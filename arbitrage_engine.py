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
from scrape_loro import intercept_loro_odds
from scrape_polymarket import fetch_polymarket_odds
from transform_scraped import (
    build_player_index,
    build_recent_player_ids,
    harmonize_name,
    normalize_str,
)

load_dotenv()

DATA_DIR = Path('data')
PLAYERS_CSV = 'atp_players.csv'
PAPER_TRADES_CSV = DATA_DIR / 'paper_trades.csv'
SCAN_LOG_CSV = DATA_DIR / 'scan_log.csv'

# ── Edge Detection ────────────────────────────────────────────────────────────

DEFAULT_MIN_EDGE = 3.0  # minimum edge % to flag
AGREEMENT_THRESHOLD = 0.05  # model & sharp must agree within 5% to confirm signal


# ── Paper Trading Logger ─────────────────────────────────────────────────────

PAPER_FIELDS = [
    'timestamp', 'player', 'opponent', 'tournament', 'surface',
    'model_prob', 'sharp_prob', 'consensus_prob',
    'loro_prob', 'loro_odds', 'sharp_odds',
    'edge_pct', 'sources', 'sharp_source', 'soft_source',
    'bet_size', 'expected_value',
    # Filled in later when result is known:
    'result', 'pnl', 'settled_at',
]

SCAN_FIELDS = [
    'timestamp', 'total_matches', 'with_model', 'with_sharp', 'with_loro',
    'with_all_3', 'opportunities', 'max_edge', 'api_quota_remaining',
    'sharp_source', 'soft_source', 'loro_direct',
]


def log_paper_trade(opp, sharp_source='', soft_source='', bankroll=100.0):
    """Log a detected edge to paper_trades.csv for later backtesting.

    Uses Kelly criterion (quarter-Kelly) to size the bet.
    """
    DATA_DIR.mkdir(exist_ok=True)
    file_exists = PAPER_TRADES_CSV.exists()

    # Quarter-Kelly bet sizing
    edge = opp['edge'] / 100.0
    odds = opp['loro_odds']
    kelly_fraction = edge / (odds - 1) if odds > 1 else 0
    quarter_kelly = kelly_fraction * 0.25
    bet_size = round(bankroll * max(0, min(quarter_kelly, 0.10)), 2)  # cap at 10%
    ev = round(bet_size * edge, 2)

    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'player': opp['player'],
        'opponent': opp['opponent'],
        'tournament': opp.get('tournament', ''),
        'surface': opp.get('surface', ''),
        'model_prob': f"{opp['model_prob']:.4f}" if opp['model_prob'] else '',
        'sharp_prob': f"{opp['sharp_prob']:.4f}" if opp['sharp_prob'] else '',
        'consensus_prob': f"{opp['consensus_prob']:.4f}",
        'loro_prob': f"{opp['loro_prob']:.4f}",
        'loro_odds': f"{opp['loro_odds']:.2f}",
        'sharp_odds': f"{opp.get('sharp_odds', 0):.2f}" if opp.get('sharp_odds') else '',
        'edge_pct': f"{opp['edge']:.2f}",
        'sources': opp['sources'],
        'sharp_source': sharp_source,
        'soft_source': soft_source,
        'bet_size': bet_size,
        'expected_value': ev,
        'result': '',
        'pnl': '',
        'settled_at': '',
    }

    with open(PAPER_TRADES_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=PAPER_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"    Paper trade logged: {opp['player']} @ {opp['loro_odds']:.2f} "
          f"(edge {opp['edge']:.1f}%, bet CHF {bet_size})")


def log_scan(unified, opportunities, sharp_source='', soft_source='',
             loro_direct=False, api_quota=None):
    """Log scan summary to scan_log.csv."""
    DATA_DIR.mkdir(exist_ok=True)
    file_exists = SCAN_LOG_CSV.exists()

    with_model = sum(1 for m in unified if m['model'])
    with_sharp = sum(1 for m in unified if m['sharp'])
    with_loro = sum(1 for m in unified if m['loro'])
    with_all = sum(1 for m in unified if m['model'] and m['sharp'] and m['loro'])
    max_edge = max((o['edge'] for o in opportunities), default=0)

    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_matches': len(unified),
        'with_model': with_model,
        'with_sharp': with_sharp,
        'with_loro': with_loro,
        'with_all_3': with_all,
        'opportunities': len(opportunities),
        'max_edge': f"{max_edge:.2f}" if max_edge else '0',
        'api_quota_remaining': api_quota or '',
        'sharp_source': sharp_source,
        'soft_source': soft_source,
        'loro_direct': loro_direct,
    }

    with open(SCAN_LOG_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SCAN_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def print_paper_summary():
    """Print summary of paper trading performance."""
    if not PAPER_TRADES_CSV.exists():
        return

    total = 0
    settled = 0
    wins = 0
    total_pnl = 0.0
    total_bet = 0.0

    with open(PAPER_TRADES_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            bet = float(row.get('bet_size', 0) or 0)
            total_bet += bet
            if row.get('result'):
                settled += 1
                pnl = float(row.get('pnl', 0) or 0)
                total_pnl += pnl
                if row['result'] == 'W':
                    wins += 1

    print(f"\n  Paper Trading: {total} trades logged, {settled} settled")
    if settled:
        roi = (total_pnl / total_bet * 100) if total_bet else 0
        print(f"  Record: {wins}W-{settled-wins}L ({wins/settled:.0%} win rate)")
        print(f"  PnL: CHF {total_pnl:+.2f} | ROI: {roi:+.1f}%")


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


def generate_upcoming_csv(events, output_csv=None):
    """Generate upcoming_matches.csv from Odds API events.

    Maps Odds API sport keys to tournament info so predict.py can
    run without needing the Playwright-based scrape_upcoming.py.
    """
    output_csv = output_csv or str(DATA_DIR / 'upcoming_matches.csv')

    # Map sport keys to surface and level
    TOURNEY_MAP = {
        'australian_open': ('Hard', 'G', 'Australian Open'),
        'french_open': ('Clay', 'G', 'French Open'),
        'wimbledon': ('Grass', 'G', 'Wimbledon'),
        'us_open': ('Hard', 'G', 'US Open'),
        'indian_wells': ('Hard', 'M', 'Indian Wells'),
        'miami_open': ('Hard', 'M', 'Miami Open'),
        'monte_carlo': ('Clay', 'M', 'Monte Carlo'),
        'madrid_open': ('Clay', 'M', 'Madrid Open'),
        'rome': ('Clay', 'M', 'Rome'),
        'canadian_open': ('Hard', 'M', 'Canadian Open'),
        'cincinnati': ('Hard', 'M', 'Cincinnati'),
        'shanghai': ('Hard', 'M', 'Shanghai'),
        'paris': ('Hard', 'M', 'Paris'),
    }

    matches = []
    for event in events:
        sport_key = event.get('sport_key', '')
        # Extract tournament slug from sport_key (e.g. "tennis_atp_miami_open" -> "miami_open")
        slug = sport_key.replace('tennis_atp_', '')

        surface, level, tourney_name = 'Hard', 'A', ''
        for key, (s, l, n) in TOURNEY_MAP.items():
            if key in slug:
                surface, level, tourney_name = s, l, n
                break
        if not tourney_name:
            tourney_name = event.get('sport_title', slug).replace('ATP ', '')

        p1 = event.get('home_team', '')
        p2 = event.get('away_team', '')
        if p1 and p2:
            matches.append({
                'player1_name': p1,
                'player2_name': p2,
                'surface': surface,
                'tourney_level': level,
                'tourney_name': tourney_name,
                'round': '',
                'best_of': 5 if level == 'G' else 3,
            })

    if not matches:
        return None

    DATA_DIR.mkdir(exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=matches[0].keys())
        writer.writeheader()
        writer.writerows(matches)

    print(f"  Generated {len(matches)} upcoming matches from Odds API")
    return output_csv


def run_predictions(events=None, input_csv=None, output_csv=None, load_state=True):
    """Run predict.py as subprocess and return path to predictions CSV.

    If events are provided, generates upcoming_matches.csv from Odds API data
    instead of requiring scrape_upcoming.py.
    """
    output_csv = output_csv or str(DATA_DIR / 'predictions.csv')

    if events:
        input_csv = generate_upcoming_csv(events)
        if not input_csv:
            print("  No matches to predict")
            return None
    else:
        input_csv = input_csv or str(DATA_DIR / 'upcoming_matches.csv')

    cmd = [sys.executable, 'predict.py', '--input', input_csv, '--output', output_csv]
    if load_state:
        cmd.extend(['--load-state', '--save-state'])

    print("Running predict.py...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  predict.py error: {result.stderr[:500]}")
            return None
        print(f"  predict.py completed")
    except subprocess.TimeoutExpired:
        print("  predict.py timed out")
        return None
    except FileNotFoundError:
        print("  predict.py not found")
        return None

    return output_csv


def match_across_sources(model_preds, sharp_lines, loro_lines,
                         player_data, last_initial_index, full_name_index,
                         recent_ids, poly_lines=None):
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

    # Polymarket lines: names like "Carlos Alcaraz"
    poly_by_players = {}
    for line in (poly_lines or []):
        pid_home = harmonize_name(line['home'], 'odds_api',
                                  last_initial_index, full_name_index,
                                  player_data, recent_ids)
        pid_away = harmonize_name(line['away'], 'odds_api',
                                  last_initial_index, full_name_index,
                                  player_data, recent_ids)
        if pid_home and pid_away:
            key = tuple(sorted([pid_home, pid_away]))
            poly_by_players[key] = {**line, 'pid_home': pid_home, 'pid_away': pid_away}

    # Find matches present in multiple sources
    all_keys = (set(model_by_players) | set(sharp_by_players)
                | set(loro_by_players) | set(poly_by_players))

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
            'poly': None,
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

        if key in poly_by_players:
            po = poly_by_players[key]
            if po.get('pid_home') == pid1:
                entry['poly'] = {
                    'p1_prob': po['home_prob'], 'p2_prob': po['away_prob'],
                    'p1_odds': po['home_odds'], 'p2_odds': po['away_odds'],
                    'volume': po.get('volume', 0),
                    'liquidity': po.get('liquidity', 0),
                }
            else:
                entry['poly'] = {
                    'p1_prob': po['away_prob'], 'p2_prob': po['home_prob'],
                    'p1_odds': po['away_odds'], 'p2_odds': po['home_odds'],
                    'volume': po.get('volume', 0),
                    'liquidity': po.get('liquidity', 0),
                }
            entry['sources'] += 1

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

            # Get consensus probability from model + sharp + polymarket
            probs = []
            if match['model']:
                probs.append(match['model'][f'{player_key}_prob'])
            if match['sharp']:
                probs.append(match['sharp'][f'{player_key}_prob'])
            if match['poly']:
                probs.append(match['poly'][f'{player_key}_prob'])

            if not probs:
                continue

            consensus_prob = sum(probs) / len(probs)

            # Check sharp sources agreement (skip if they diverge too much)
            sharp_probs = []
            if match['sharp']:
                sharp_probs.append(match['sharp'][f'{player_key}_prob'])
            if match['poly']:
                sharp_probs.append(match['poly'][f'{player_key}_prob'])
            if len(sharp_probs) == 2 and abs(sharp_probs[0] - sharp_probs[1]) > AGREEMENT_THRESHOLD * 2:
                continue  # Sharp sources disagree too much

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
                        'sharp_odds': match['sharp'][f'{player_key}_odds'] if match['sharp'] else None,
                        'poly_prob': match['poly'][f'{player_key}_prob'] if match['poly'] else None,
                        'poly_volume': match['poly'].get('volume', 0) if match['poly'] else None,
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
        lines.append(f"Sharp: {opp['sharp_prob']:.1%}")
    if opp.get('poly_prob') is not None:
        vol = opp.get('poly_volume', 0)
        lines.append(f"Polymarket: {opp['poly_prob']:.1%} (${vol:,.0f} vol)")
    lines.append(f"Consensus: {opp['consensus_prob']:.1%}")
    lines.append(f"")
    lines.append(f"Soft odds: *{opp['loro_odds']:.2f}* ({opp['loro_prob']:.1%} implied)")
    lines.append(f"Edge: *{opp['edge']:.1f}%*")
    lines.append(f"")
    lines.append(f"Sources: {opp['sources']}/4")
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
    with_poly = sum(1 for m in unified if m['poly'])
    with_all = sum(1 for m in unified if m['sharp'] and m['loro'])

    print(f"\n  Matches found:  {len(unified)} total")
    print(f"  With model:     {with_model}")
    print(f"  With Sharp:     {with_sharp}")
    print(f"  With LORO/Soft: {with_loro}")
    print(f"  With Polymarket:{with_poly}")
    print(f"  Sharp+Soft:     {with_all}")

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

def _find_pre_edges(poly_lines, loro_lines, player_data,
                    last_initial_index, full_name_index, recent_ids,
                    min_edge):
    """Quick pre-scan: compare Polymarket vs LORO to find potential edges.

    Returns list of (player_name, opponent_name, poly_prob, loro_prob, edge)
    for matches where Polymarket and LORO diverge enough to warrant
    burning an Odds API credit to confirm.
    """
    # Build player-keyed indexes
    poly_by = {}
    for line in poly_lines:
        pid_h = harmonize_name(line['home'], 'odds_api',
                               last_initial_index, full_name_index,
                               player_data, recent_ids)
        pid_a = harmonize_name(line['away'], 'odds_api',
                               last_initial_index, full_name_index,
                               player_data, recent_ids)
        if pid_h and pid_a:
            key = tuple(sorted([pid_h, pid_a]))
            poly_by[key] = {**line, 'pid_home': pid_h, 'pid_away': pid_a}

    loro_by = {}
    for line in loro_lines:
        pid_h = harmonize_name(line['home'], 'loro',
                               last_initial_index, full_name_index,
                               player_data, recent_ids)
        pid_a = harmonize_name(line['away'], 'loro',
                               last_initial_index, full_name_index,
                               player_data, recent_ids)
        if pid_h and pid_a:
            key = tuple(sorted([pid_h, pid_a]))
            loro_by[key] = {**line, 'pid_home': pid_h, 'pid_away': pid_a}

    edges = []
    common_keys = set(poly_by) & set(loro_by)
    for key in common_keys:
        pid1, pid2 = key
        po = poly_by[key]
        lo = loro_by[key]

        # Align player order
        if po.get('pid_home') == pid1:
            poly_p1, poly_p2 = po['home_prob'], po['away_prob']
        else:
            poly_p1, poly_p2 = po['away_prob'], po['home_prob']

        if lo.get('pid_home') == pid1:
            loro_p1, loro_p2 = lo['home_prob'], lo['away_prob']
        else:
            loro_p1, loro_p2 = lo['away_prob'], lo['home_prob']

        # Check edge for each player
        for poly_p, loro_p in [(poly_p1, loro_p1), (poly_p2, loro_p2)]:
            edge = (poly_p - loro_p) * 100
            if edge >= min_edge:
                edges.append(edge)

    return edges


def scan_once(args, player_data, last_initial_index, full_name_index, recent_ids):
    """Run a single arbitrage scan.

    Flow: Polymarket + LORO first (both free), then only call the paid
    Odds API if a potential edge is detected.
    """

    # Track sources for logging
    sharp_source = ''
    soft_source = ''
    loro_direct = False
    api_quota = None

    # ── Phase 1: Free sources (Polymarket + LORO) ──────────────────────

    # 1a. Polymarket (free, no API key, no quota)
    poly_lines = []
    if not args.skip_poly:
        print("Fetching Polymarket odds...")
        try:
            poly_lines = fetch_polymarket_odds(min_liquidity=1000)
            if poly_lines:
                print(f"  Polymarket: {len(poly_lines)} ATP markets")
            else:
                print("  Polymarket: no active ATP markets")
        except Exception as e:
            print(f"  Polymarket error: {e}")
    else:
        print("  Polymarket: skipped")

    # 1b. LORO direct API (free)
    loro_lines = []
    if not args.skip_loro:
        print("Fetching LORO odds...")
        try:
            loro_lines = intercept_loro_odds(headless=True)
            if loro_lines:
                loro_direct = True
                soft_source = 'loro_direct'
                print(f"  LORO direct: {len(loro_lines)} matches")
        except Exception as e:
            print(f"  LORO direct failed: {e}")

    # ── Phase 2: Check for pre-edges before using paid API ─────────────

    events = []
    sharp_lines = []
    need_sharp = False

    if poly_lines and loro_lines:
        pre_edges = _find_pre_edges(
            poly_lines, loro_lines, player_data,
            last_initial_index, full_name_index, recent_ids,
            min_edge=args.min_edge
        )
        if pre_edges:
            need_sharp = True
            print(f"\n  Pre-scan: {len(pre_edges)} potential edge(s) detected "
                  f"(max {max(pre_edges):.1f}%) — confirming with sharp odds...")
        else:
            print(f"\n  Pre-scan: no Poly vs LORO edges >= {args.min_edge}% — "
                  f"skipping Odds API (saving credits)")
    elif not poly_lines and not args.skip_poly:
        # No Polymarket data — fall back to Odds API
        need_sharp = True
        print("  No Polymarket data — will use Odds API")
    elif args.skip_poly:
        need_sharp = True

    # ── Phase 3: Paid Odds API (only if needed) ───────────────────────

    if need_sharp and not args.skip_sharp:
        print("Fetching sharp odds (Odds API)...")
        events = fetch_all_tennis_odds()

        all_books = set()
        for event in events:
            for bm in event.get("bookmakers", []):
                all_books.add(bm["key"])

        # Extract sharp lines
        sharp_priority = ["pinnacle", "matchbook", "betfair_ex_eu",
                          "marathon_bet", "williamhill"]
        for book in sharp_priority:
            if book in all_books:
                sharp_lines = extract_pinnacle_lines(events, bookmaker=book)
                if sharp_lines:
                    sharp_source = book
                    print(f"  Sharp source: {book} ({len(sharp_lines)} matches)")
                    break
        if not sharp_lines:
            print("  Sharp: no bookmaker found")

        # If LORO failed, use soft proxy from the same API call
        if not loro_lines and not args.skip_loro:
            print("  LORO failed — using soft book proxy...")
            soft_priority = ["unibet", "unibet_eu", "sport888", "betclic",
                             "nordicbet", "williamhill"]
            for book in soft_priority:
                if book in all_books:
                    loro_lines = extract_pinnacle_lines(events, bookmaker=book)
                    if loro_lines:
                        soft_source = book
                        for m in loro_lines:
                            m["source"] = "loro_proxy"
                            m["proxy_book"] = book
                        print(f"  Soft source: {book} ({len(loro_lines)} matches)")
                        break
    elif not need_sharp:
        print("  Sharp: skipped (no pre-edge)")

    # ── Phase 4: Model predictions ────────────────────────────────────

    model_preds = {}
    if not args.skip_model:
        predictions_csv = args.predictions
        if not predictions_csv:
            predictions_csv = run_predictions(events=events or None, load_state=True)
        if predictions_csv:
            model_preds = load_predictions(predictions_csv)
            print(f"  Model: {len(model_preds)} predictions loaded")
    else:
        print("  Model: skipped")

    # ── Phase 5: Match across sources + find opportunities ────────────

    unified = match_across_sources(
        model_preds, sharp_lines, loro_lines,
        player_data, last_initial_index, full_name_index, recent_ids,
        poly_lines=poly_lines
    )

    opportunities = find_opportunities(unified, min_edge=args.min_edge)

    # 6. Display
    print_scan_results(unified, opportunities, args.min_edge)

    # 7. Log scan
    log_scan(unified, opportunities, sharp_source=sharp_source,
             soft_source=soft_source, loro_direct=loro_direct,
             api_quota=api_quota)

    # 8. Send alerts + log paper trades
    if opportunities:
        print(f"\n  Logging {len(opportunities)} paper trade(s)...")
        for opp in opportunities:
            log_paper_trade(opp, sharp_source=sharp_source,
                           soft_source=soft_source)

        if not args.dry_run:
            print(f"  Sending {len(opportunities)} Telegram alert(s)...")
            for opp in opportunities:
                msg = format_alert(opp)
                sent = send_telegram_alert(msg)
                status = "sent" if sent else "FAILED"
                print(f"    {opp['player']} ({opp['edge']:.1f}% edge): {status}")
        else:
            print(f"\n  [DRY RUN] Would send {len(opportunities)} alert(s)")
            for opp in opportunities:
                print(f"\n{format_alert(opp)}")

    # 9. Paper trading summary
    print_paper_summary()

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
    parser.add_argument("--skip-poly", action="store_true",
                        help="Skip Polymarket odds fetch")
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
