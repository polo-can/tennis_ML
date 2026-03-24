"""
Hybrid Latency Arbitrage Engine for ATP Tennis.

Orchestrates three data sources:
  1. predict.py     — ML model win probabilities (Elo + features)
  2. scrape_sharp   — Pinnacle odds via The Odds API (sharp baseline)
  3. scrape_loro    — Loterie Romande odds via Playwright interceptor

When model + sharp bookmaker agree a player is favored, but LORO offers
stale/high odds on that player, the engine flags an arbitrage opportunity
and sends a Telegram alert.

Usage:
    python3 arbitrage_engine.py                     # Single scan, print results
    python3 arbitrage_engine.py --dry-run           # No Telegram alerts
    python3 arbitrage_engine.py --min-edge 8        # Only flag 8%+ edges
    python3 arbitrage_engine.py --loop --interval 5 # Continuous scanning every 5 min
    python3 arbitrage_engine.py --skip-loro         # Test without LORO (sharp vs model only)
    python3 arbitrage_engine.py --skip-model        # Test without ML model (sharp vs LORO only)

Requires .env with: ODDS_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from scrape_sharp import (get_sharp_lines, get_soft_lines,
                         fetch_all_tennis_odds, extract_pinnacle_lines)
from transform_scraped import (
    build_player_index,
    build_recent_player_ids,
    harmonize_name,
    normalize_str,
)

load_dotenv()

DATA_DIR = Path('data')
PLAYERS_CSV = 'atp_players.csv'

# ── Edge Detection ────────────────────────────────────────────────────────────

DEFAULT_MIN_EDGE = 5.0  # minimum edge % to flag
AGREEMENT_THRESHOLD = 0.05  # model & sharp must agree within 5% to confirm signal


def load_predictions(predictions_csv):
    """Load model predictions from CSV.

    Returns:
        dict keyed by (normalized_p1, normalized_p2) with win probabilities
    """
    preds = {}
    try:
        with open(predictions_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                p1 = normalize_str(row.get('player1_name', ''))
                p2 = normalize_str(row.get('player2_name', ''))
                try:
                    p1_prob = float(row.get('p1_win_prob', 0))
                    p2_prob = float(row.get('p2_win_prob', 0))
                except (ValueError, TypeError):
                    continue

                preds[(p1, p2)] = {
                    'p1_name': row.get('player1_name', ''),
                    'p2_name': row.get('player2_name', ''),
                    'p1_prob': p1_prob,
                    'p2_prob': p2_prob,
                    'tournament': row.get('tourney_name', ''),
                    'surface': row.get('surface', ''),
                    'confidence': row.get('confidence', ''),
                }
    except FileNotFoundError:
        print(f"  Warning: {predictions_csv} not found")

    return preds


def run_predictions(input_csv=None, output_csv=None, load_state=True):
    """Run predict.py as subprocess and return path to predictions CSV."""
    input_csv = input_csv or str(DATA_DIR / 'upcoming_matches.csv')
    output_csv = output_csv or str(DATA_DIR / 'predictions.csv')

    cmd = ['python3', 'predict.py', '--input', input_csv, '--output', output_csv]
    if load_state:
        cmd.append('--load-state')

    print("Running predict.py...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  predict.py error: {result.stderr[:500]}")
            return None
        print("  predict.py completed")
    except subprocess.TimeoutExpired:
        print("  predict.py timed out")
        return None
    except FileNotFoundError:
        print("  predict.py not found")
        return None

    return output_csv


def match_across_sources(model_preds, sharp_lines, loro_lines,
                         player_data, last_initial_index, full_name_index,
                         recent_ids):
    """Match players across all three data sources using player_id as key.

    Returns list of unified match dicts with data from all available sources.
    """
    # Build player_id-keyed indexes for each source

    # Model predictions: names are Sackmann format ("Carlos Alcaraz")
    model_by_players = {}
    for (p1_norm, p2_norm), pred in model_preds.items():
        pid1 = harmonize_name(pred['p1_name'], 'odds_api',
                              last_initial_index, full_name_index,
                              player_data, recent_ids)
        pid2 = harmonize_name(pred['p2_name'], 'odds_api',
                              last_initial_index, full_name_index,
                              player_data, recent_ids)
        if pid1 and pid2:
            key = tuple(sorted([pid1, pid2]))
            model_by_players[key] = {**pred, 'pid1': pid1, 'pid2': pid2}

    # Sharp lines: names from Odds API ("Carlos Alcaraz")
    sharp_by_players = {}
    for line in sharp_lines:
        pid_home = harmonize_name(line['home'], 'odds_api',
                                  last_initial_index, full_name_index,
                                  player_data, recent_ids)
        pid_away = harmonize_name(line['away'], 'odds_api',
                                  last_initial_index, full_name_index,
                                  player_data, recent_ids)
        if pid_home and pid_away:
            key = tuple(sorted([pid_home, pid_away]))
            sharp_by_players[key] = {**line, 'pid_home': pid_home, 'pid_away': pid_away}

    # LORO lines: names from OpenBet (format TBD)
    loro_by_players = {}
    for line in loro_lines:
        pid_home = harmonize_name(line['home'], 'loro',
                                  last_initial_index, full_name_index,
                                  player_data, recent_ids)
        pid_away = harmonize_name(line['away'], 'loro',
                                  last_initial_index, full_name_index,
                                  player_data, recent_ids)
        if pid_home and pid_away:
            key = tuple(sorted([pid_home, pid_away]))
            loro_by_players[key] = {**line, 'pid_home': pid_home, 'pid_away': pid_away}

    # Find matches present in multiple sources
    all_keys = set(model_by_players) | set(sharp_by_players) | set(loro_by_players)

    unified = []
    for key in all_keys:
        pid1, pid2 = key
        p1_info = player_data.get(pid1, {})
        p2_info = player_data.get(pid2, {})
        p1_display = f"{p1_info.get('name_first', '')} {p1_info.get('name_last', '')}".strip()
        p2_display = f"{p2_info.get('name_first', '')} {p2_info.get('name_last', '')}".strip()

        entry = {
            'pid1': pid1,
            'pid2': pid2,
            'p1_name': p1_display or str(pid1),
            'p2_name': p2_display or str(pid2),
            'model': None,
            'sharp': None,
            'loro': None,
            'sources': 0,
        }

        if key in model_by_players:
            m = model_by_players[key]
            # Align player order: pid1 is always the sorted-first
            if m.get('pid1') == pid1:
                entry['model'] = {'p1_prob': m['p1_prob'], 'p2_prob': m['p2_prob']}
            else:
                entry['model'] = {'p1_prob': m['p2_prob'], 'p2_prob': m['p1_prob']}
            entry['sources'] += 1
            entry['tournament'] = m.get('tournament', '')
            entry['surface'] = m.get('surface', '')

        if key in sharp_by_players:
            s = sharp_by_players[key]
            if s.get('pid_home') == pid1:
                entry['sharp'] = {
                    'p1_prob': s['home_prob'], 'p2_prob': s['away_prob'],
                    'p1_odds': s['home_odds'], 'p2_odds': s['away_odds'],
                }
            else:
                entry['sharp'] = {
                    'p1_prob': s['away_prob'], 'p2_prob': s['home_prob'],
                    'p1_odds': s['away_odds'], 'p2_odds': s['home_odds'],
                }
            entry['sources'] += 1

        if key in loro_by_players:
            lo = loro_by_players[key]
            if lo.get('pid_home') == pid1:
                entry['loro'] = {
                    'p1_odds': lo['home_odds'], 'p2_odds': lo['away_odds'],
                    'p1_prob': lo['home_prob'], 'p2_prob': lo['away_prob'],
                }
            else:
                entry['loro'] = {
                    'p1_odds': lo['away_odds'], 'p2_odds': lo['home_odds'],
                    'p1_prob': lo['away_prob'], 'p2_prob': lo['home_prob'],
                }
            entry['sources'] += 1
            if not entry.get('tournament'):
                entry['tournament'] = lo.get('tournament', '')

        unified.append(entry)

    return unified


def find_opportunities(unified_matches, min_edge=DEFAULT_MIN_EDGE):
    """Identify arbitrage opportunities.

    An opportunity exists when:
    - Model and sharp both agree player X has probability P
    - LORO offers odds implying probability Q, where Q < P
    - Edge = P - Q >= min_edge

    If LORO is unavailable, falls back to model vs sharp comparison.
    """
    opportunities = []

    for match in unified_matches:
        for player_key in ['p1', 'p2']:
            opp_key = 'p2' if player_key == 'p1' else 'p1'
            player_name = match[f'{player_key}_name']

            # Get consensus probability from model + sharp
            probs = []
            if match['model']:
                probs.append(match['model'][f'{player_key}_prob'])
            if match['sharp']:
                probs.append(match['sharp'][f'{player_key}_prob'])

            if not probs:
                continue

            consensus_prob = sum(probs) / len(probs)

            # Check model-sharp agreement
            if len(probs) == 2 and abs(probs[0] - probs[1]) > AGREEMENT_THRESHOLD * 2:
                continue  # Model and sharp disagree too much

            # Calculate edge against LORO
            if match['loro']:
                loro_prob = match['loro'][f'{player_key}_prob']
                loro_odds = match['loro'][f'{player_key}_odds']
                edge = (consensus_prob - loro_prob) * 100

                if edge >= min_edge:
                    opportunities.append({
                        'player': player_name,
                        'opponent': match[f'{opp_key}_name'],
                        'tournament': match.get('tournament', ''),
                        'surface': match.get('surface', ''),
                        'model_prob': match['model'][f'{player_key}_prob'] if match['model'] else None,
                        'sharp_prob': match['sharp'][f'{player_key}_prob'] if match['sharp'] else None,
                        'consensus_prob': consensus_prob,
                        'loro_prob': loro_prob,
                        'loro_odds': loro_odds,
                        'edge': edge,
                        'sources': match['sources'],
                    })

    # Sort by edge descending
    opportunities.sort(key=lambda x: x['edge'], reverse=True)
    return opportunities


# ── Telegram Alerts ───────────────────────────────────────────────────────────

def send_telegram_alert(message):
    """Send a message via Telegram Bot API."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("  Telegram not configured (missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID)")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"  Telegram error: {e}")
        return False


def format_alert(opp):
    """Format an opportunity as a Telegram message."""
    lines = [
        f"*EDGE DETECTED* ({opp['edge']:.1f}%)",
        f"",
        f"*{opp['player']}* vs {opp['opponent']}",
        f"Tournament: {opp.get('tournament', 'Unknown')}",
        f"Surface: {opp.get('surface', '?')}",
        f"",
    ]

    if opp['model_prob'] is not None:
        lines.append(f"Model: {opp['model_prob']:.1%}")
    if opp['sharp_prob'] is not None:
        lines.append(f"Pinnacle: {opp['sharp_prob']:.1%}")
    lines.append(f"Consensus: {opp['consensus_prob']:.1%}")
    lines.append(f"")
    lines.append(f"Soft odds: *{opp['loro_odds']:.2f}* ({opp['loro_prob']:.1%} implied)")
    lines.append(f"Edge: *{opp['edge']:.1f}%*")
    lines.append(f"")
    lines.append(f"Sources: {opp['sources']}/3")
    lines.append(f"_Check LORO for similar or better odds_")

    return "\n".join(lines)


# ── Display ───────────────────────────────────────────────────────────────────

def print_scan_results(unified, opportunities, min_edge):
    """Print a summary of the scan."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*70}")
    print(f"ARBITRAGE SCAN — {now}")
    print(f"{'='*70}")

    # Source summary
    with_model = sum(1 for m in unified if m['model'])
    with_sharp = sum(1 for m in unified if m['sharp'])
    with_loro = sum(1 for m in unified if m['loro'])
    with_all = sum(1 for m in unified if m['model'] and m['sharp'] and m['loro'])

    print(f"\n  Matches found:  {len(unified)} total")
    print(f"  With model:     {with_model}")
    print(f"  With Pinnacle:  {with_sharp}")
    print(f"  With LORO:      {with_loro}")
    print(f"  In all 3:       {with_all}")

    if not opportunities:
        print(f"\n  No edges >= {min_edge}% found.")
        return

    print(f"\n  OPPORTUNITIES ({len(opportunities)} found, min edge {min_edge}%):")
    print(f"  {'Player':<25} {'vs':<25} {'Consensus':>10} {'LORO':>10} {'Edge':>8}")
    print(f"  {'-'*80}")

    for opp in opportunities:
        player = opp['player'][:23]
        opponent = opp['opponent'][:23]
        print(f"  {player:<25} {opponent:<25} "
              f"{opp['consensus_prob']:>9.1%} "
              f"{opp['loro_odds']:>9.2f} "
              f"{opp['edge']:>7.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def scan_once(args, player_data, last_initial_index, full_name_index, recent_ids):
    """Run a single arbitrage scan."""

    # 1. Model predictions
    model_preds = {}
    if not args.skip_model:
        predictions_csv = args.predictions
        if not predictions_csv:
            predictions_csv = run_predictions(load_state=True)
        if predictions_csv:
            model_preds = load_predictions(predictions_csv)
            print(f"  Model: {len(model_preds)} predictions loaded")
    else:
        print("  Model: skipped")

    # 2+3. Fetch all odds in ONE API call, then split sharp vs soft
    sharp_lines = []
    loro_lines = []
    if not args.skip_sharp or not args.skip_loro:
        print("Fetching odds from all bookmakers...")
        events = fetch_all_tennis_odds()

        all_books = set()
        for event in events:
            for bm in event.get("bookmakers", []):
                all_books.add(bm["key"])
        if all_books:
            print(f"  Available: {', '.join(sorted(all_books))}")

        if not args.skip_sharp:
            # Extract sharp lines (try in order of sharpness)
            sharp_priority = ["pinnacle", "matchbook", "betfair_ex_eu",
                              "marathon_bet", "williamhill"]
            for book in sharp_priority:
                if book in all_books:
                    sharp_lines = extract_pinnacle_lines(events, bookmaker=book)
                    if sharp_lines:
                        print(f"  Sharp source: {book} ({len(sharp_lines)} matches)")
                        break
            if not sharp_lines:
                print("  Sharp: no bookmaker found")

        if not args.skip_loro:
            # Extract soft lines as LORO proxy
            soft_priority = ["unibet", "unibet_eu", "sport888", "betclic",
                             "nordicbet", "williamhill"]
            for book in soft_priority:
                if book in all_books:
                    loro_lines = extract_pinnacle_lines(events, bookmaker=book)
                    if loro_lines:
                        for m in loro_lines:
                            m["source"] = "loro_proxy"
                            m["proxy_book"] = book
                        print(f"  Soft source: {book} ({len(loro_lines)} matches)")
                        break
            if not loro_lines:
                print("  Soft: no bookmaker found")
    else:
        print("  Sharp: skipped")
        print("  LORO: skipped")

    # 4. Match across sources
    unified = match_across_sources(
        model_preds, sharp_lines, loro_lines,
        player_data, last_initial_index, full_name_index, recent_ids
    )

    # 5. Find opportunities
    opportunities = find_opportunities(unified, min_edge=args.min_edge)

    # 6. Display
    print_scan_results(unified, opportunities, args.min_edge)

    # 7. Send alerts
    if opportunities and not args.dry_run:
        print(f"\n  Sending {len(opportunities)} Telegram alert(s)...")
        for opp in opportunities:
            msg = format_alert(opp)
            sent = send_telegram_alert(msg)
            status = "sent" if sent else "FAILED"
            print(f"    {opp['player']} ({opp['edge']:.1f}% edge): {status}")
    elif opportunities and args.dry_run:
        print(f"\n  [DRY RUN] Would send {len(opportunities)} alert(s)")
        for opp in opportunities:
            print(f"\n{format_alert(opp)}")

    return opportunities


def main():
    parser = argparse.ArgumentParser(description="Tennis Latency Arbitrage Engine")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print opportunities without sending Telegram alerts")
    parser.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE,
                        help=f"Minimum edge %% to flag (default: {DEFAULT_MIN_EDGE})")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to existing predictions CSV (skip running predict.py)")
    parser.add_argument("--loop", action="store_true",
                        help="Run continuously")
    parser.add_argument("--interval", type=int, default=5,
                        help="Minutes between scans in loop mode (default: 5)")
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip ML model predictions")
    parser.add_argument("--skip-sharp", action="store_true",
                        help="Skip Pinnacle odds fetch")
    parser.add_argument("--skip-loro", action="store_true",
                        help="Skip LORO odds interception")
    args = parser.parse_args()

    # Load player database (once)
    players_csv = PLAYERS_CSV
    if not Path(players_csv).exists():
        players_csv = str(DATA_DIR / 'atp_players.csv')
    print(f"Loading player database...")
    player_data, last_initial_index, full_name_index = build_player_index(players_csv)
    recent_ids = build_recent_player_ids(Path('.') if Path('atp_matches_2024.csv').exists() else DATA_DIR)
    print(f"  {len(player_data)} players, {len(recent_ids)} recent")

    if args.loop:
        print(f"\nStarting continuous scan (every {args.interval} min)...")
        print("Press Ctrl+C to stop.\n")
        while True:
            try:
                scan_once(args, player_data, last_initial_index,
                          full_name_index, recent_ids)
                print(f"\n  Next scan in {args.interval} minutes...")
                time.sleep(args.interval * 60)
            except KeyboardInterrupt:
                print("\n\nStopped.")
                break
    else:
        scan_once(args, player_data, last_initial_index,
                  full_name_index, recent_ids)


if __name__ == "__main__":
    main()
