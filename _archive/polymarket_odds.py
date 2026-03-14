"""
Fetch Polymarket tennis match odds and compare against our model predictions.

Queries the Polymarket Gamma API for active ATP match markets, extracts
moneyline implied probabilities, and compares them to our model's win
probabilities to find value discrepancies.

Usage:
    python3 polymarket_odds.py                              # Full pipeline: fetch odds, run model, compare
    python3 polymarket_odds.py --predictions data/predictions.csv  # Use existing predictions
    python3 polymarket_odds.py --fetch-only                 # Just show Polymarket odds
    python3 polymarket_odds.py --min-edge 5                 # Only show edges >= 5%
    python3 polymarket_odds.py --min-liquidity 5000         # Only markets with >= $5K liquidity
    python3 polymarket_odds.py --no-predict                 # Skip running predict.py (use existing predictions)
    python3 polymarket_odds.py --check-results              # Check resolved markets & show accuracy/ROI
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

DATA_DIR = Path('data')

# ── Polymarket API ────────────────────────────────────────────────────────────

GAMMA_API = "https://gamma-api.polymarket.com"

# Tag 864 = Tennis (all individual match events, ATP + WTA)
TENNIS_TAG_ID = 864


def fetch_tennis_events(limit=100):
    """Fetch all active tennis match events from Polymarket Gamma API."""
    params = urlencode({
        'active': 'true',
        'closed': 'false',
        'limit': str(limit),
        'tag_id': str(TENNIS_TAG_ID),
        'order': 'startDate',
        'ascending': 'false',
    })
    url = f"{GAMMA_API}/events?{params}"

    req = Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
        'Accept': 'application/json',
    })

    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data if isinstance(data, list) else []
    except (HTTPError, URLError, json.JSONDecodeError) as e:
        print(f"  Error fetching events: {e}")
        return []


def fetch_event_by_slug(slug):
    """Fetch a single event by its slug for detailed data."""
    params = urlencode({'slug': slug})
    url = f"{GAMMA_API}/events?{params}"

    req = Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
        'Accept': 'application/json',
    })

    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            if isinstance(data, list) and len(data) > 0:
                return data[0]
            return None
    except (HTTPError, URLError, json.JSONDecodeError) as e:
        print(f"  Error fetching event {slug}: {e}")
        return None


def extract_atp_matches(events):
    """Extract ATP moneyline match data from tennis events.

    Returns list of dicts with player names and implied probabilities.
    """
    matches = []

    for event in events:
        slug = event.get('slug', '')
        title = event.get('title', '')

        # Only ATP matches (skip WTA)
        if not slug.startswith('atp-'):
            continue

        # Get markets for this event
        markets = event.get('markets', [])
        if not markets:
            continue

        # Find the moneyline market (main match winner)
        # It's typically the first market, or the one whose question matches the title
        moneyline = None
        for m in markets:
            question = m.get('question', '')
            market_type = m.get('marketType', '')
            # Moneyline market type, or the main event market (same as title)
            if market_type == 'moneyline' or question == title:
                moneyline = m
                break

        # Fallback: first market with exactly 2 player-name outcomes
        if not moneyline:
            for m in markets:
                outcomes = m.get('outcomes', '')
                if isinstance(outcomes, str):
                    try:
                        outcomes = json.loads(outcomes)
                    except json.JSONDecodeError:
                        continue
                if (isinstance(outcomes, list) and len(outcomes) == 2
                        and 'Over' not in outcomes[0]
                        and 'Under' not in outcomes[0]):
                    moneyline = m
                    break

        if not moneyline:
            continue

        # Parse outcomes and prices
        outcomes = moneyline.get('outcomes', '[]')
        prices = moneyline.get('outcomePrices', '[]')

        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except json.JSONDecodeError:
                continue
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except json.JSONDecodeError:
                continue

        if len(outcomes) != 2 or len(prices) != 2:
            continue

        try:
            p1_price = float(prices[0])
            p2_price = float(prices[1])
        except (ValueError, TypeError):
            continue

        # Extract tournament name from title (e.g., "BNP Paribas Open: A vs B")
        tourney = ''
        if ':' in title:
            tourney = title.split(':')[0].strip()

        # Extract volume and liquidity
        volume = moneyline.get('volume', 0) or 0
        volume_24h = moneyline.get('volume24hr', 0) or 0
        liquidity = moneyline.get('liquidity', 0) or 0

        # Extract start date
        start_date = event.get('startDate', '')

        matches.append({
            'player1': outcomes[0],
            'player2': outcomes[1],
            'p1_implied_prob': p1_price,
            'p2_implied_prob': p2_price,
            'tournament': tourney,
            'slug': slug,
            'event_id': event.get('id', ''),
            'volume': float(volume),
            'volume_24h': float(volume_24h),
            'liquidity': float(liquidity),
            'start_date': start_date,
        })

    return matches


# ── Name Matching ─────────────────────────────────────────────────────────────

def normalize_name(name):
    """Normalize a player name for fuzzy matching.

    Handles different formats:
      Polymarket: "Carlos Alcaraz", "Alexander Zverev"
      Our model:  "Alcaraz C.", "Zverev A."
    """
    name = name.strip()
    # Remove common suffixes/prefixes
    name = re.sub(r'\s*\(.*?\)\s*', '', name)  # Remove parentheticals
    name = name.replace('.', '').strip()

    # Split into parts
    parts = name.split()
    if not parts:
        return ''

    # Return lowercase parts for comparison
    return ' '.join(p.lower() for p in parts)


def extract_last_name(name):
    """Extract the last name from a player name.

    Handles:
      "Carlos Alcaraz" -> "alcaraz"
      "Alcaraz C" -> "alcaraz"
      "Roberto Bautista Agut" -> "bautista agut" (multi-part)
    """
    name = name.strip().replace('.', '').strip()
    parts = name.split()
    if not parts:
        return ''

    # If format is "LastName FirstInitial" (our model format)
    if len(parts) == 2 and len(parts[1]) <= 2:
        return parts[0].lower()

    # If format is "FirstName LastName" (Polymarket format)
    if len(parts) >= 2:
        # Return everything after the first name as the last name
        return ' '.join(parts[1:]).lower()

    return parts[0].lower()


def match_players(poly_match, predictions):
    """Try to match a Polymarket match to a prediction row.

    Returns the matched prediction dict or None.
    """
    poly_p1 = poly_match['player1']
    poly_p2 = poly_match['player2']

    poly_p1_last = extract_last_name(poly_p1)
    poly_p2_last = extract_last_name(poly_p2)

    for pred in predictions:
        pred_p1 = pred['player1_name']
        pred_p2 = pred['player2_name']

        pred_p1_last = extract_last_name(pred_p1)
        pred_p2_last = extract_last_name(pred_p2)

        # Check both orderings (p1/p2 might be swapped)
        if (pred_p1_last == poly_p1_last and pred_p2_last == poly_p2_last):
            return pred, False  # Same order
        elif (pred_p1_last == poly_p2_last and pred_p2_last == poly_p1_last):
            return pred, True  # Swapped order

    return None, None


# ── Display ───────────────────────────────────────────────────────────────────

def format_pct(val):
    """Format a probability as percentage."""
    return f"{val * 100:.1f}%"


def print_comparison(comparisons, min_edge=0):
    """Print a formatted comparison table."""
    if not comparisons:
        print("\nNo matched markets found.")
        return

    # Group by tournament
    by_tourney = {}
    for c in comparisons:
        t = c['tournament'] or 'Unknown'
        if t not in by_tourney:
            by_tourney[t] = []
        by_tourney[t].append(c)

    total_shown = 0
    value_bets = []

    for tourney, matches in by_tourney.items():
        header_printed = False

        for m in matches:
            edge1 = m['model_p1'] - m['market_p1']
            edge2 = m['model_p2'] - m['market_p2']
            max_edge = max(abs(edge1), abs(edge2))

            if max_edge * 100 < min_edge:
                continue

            if not header_printed:
                print(f"\n{'='*80}")
                print(f"  {tourney}")
                print(f"{'='*80}")
                header_printed = True

            total_shown += 1

            # Determine which player has the value edge
            if edge1 > 0:
                value_player = m['player1']
                edge = edge1
            else:
                value_player = m['player2']
                edge = edge2

            # Edge indicator
            if abs(edge) >= 0.10:
                edge_label = "*** STRONG VALUE ***"
            elif abs(edge) >= 0.05:
                edge_label = "** VALUE **"
            elif abs(edge) >= 0.02:
                edge_label = "* edge *"
            else:
                edge_label = ""

            print(f"\n  {m['player1']} vs {m['player2']}")
            print(f"  {'─'*50}")
            print(f"  {'':30s} {'Model':>10s} {'Market':>10s} {'Edge':>10s}")
            print(f"  {m['player1']:30s} {format_pct(m['model_p1']):>10s} {format_pct(m['market_p1']):>10s} {edge1*100:>+9.1f}%")
            print(f"  {m['player2']:30s} {format_pct(m['model_p2']):>10s} {format_pct(m['market_p2']):>10s} {edge2*100:>+9.1f}%")

            if edge_label:
                print(f"  --> {edge_label}: {value_player} ({format_pct(abs(edge))} edge)")

            vol_str = f"${m['volume']:,.0f}" if m['volume'] else "N/A"
            liq_str = f"${m['liquidity']:,.0f}" if m['liquidity'] else "N/A"
            print(f"  Volume: {vol_str} | Liquidity: {liq_str}")

            if abs(edge) >= 0.02:
                value_bets.append({
                    'player': value_player,
                    'opponent': m['player2'] if value_player == m['player1'] else m['player1'],
                    'tournament': tourney,
                    'model_prob': m['model_p1'] if edge1 > 0 else m['model_p2'],
                    'market_prob': m['market_p1'] if edge1 > 0 else m['market_p2'],
                    'edge': abs(edge),
                    'liquidity': m['liquidity'],
                })

    # Summary
    print(f"\n{'='*80}")
    print(f"  SUMMARY: {total_shown} matches compared")
    print(f"{'='*80}")

    if value_bets:
        # Sort by edge descending
        value_bets.sort(key=lambda x: x['edge'], reverse=True)
        print(f"\n  Top Value Bets (model edge >= 2%):")
        print(f"  {'Player':25s} {'vs':5s} {'Opponent':25s} {'Model':>8s} {'Market':>8s} {'Edge':>8s}")
        print(f"  {'─'*80}")
        for vb in value_bets:
            print(f"  {vb['player']:25s} {'vs':5s} {vb['opponent']:25s} "
                  f"{format_pct(vb['model_prob']):>8s} {format_pct(vb['market_prob']):>8s} "
                  f"{vb['edge']*100:>+7.1f}%")
    else:
        print("\n  No significant value bets found (edge < 2%).")


# ── CSV Output ────────────────────────────────────────────────────────────────

def save_comparisons(comparisons, output_path):
    """Save comparison data to CSV."""
    if not comparisons:
        return

    fieldnames = [
        'player1', 'player2', 'tournament',
        'model_p1', 'model_p2', 'market_p1', 'market_p2',
        'edge_p1', 'edge_p2', 'max_edge',
        'value_player', 'confidence',
        'volume', 'liquidity', 'slug',
    ]

    rows = []
    for c in comparisons:
        edge1 = c['model_p1'] - c['market_p1']
        edge2 = c['model_p2'] - c['market_p2']
        max_edge = max(abs(edge1), abs(edge2))
        value_player = c['player1'] if edge1 > 0 else c['player2']

        rows.append({
            'player1': c['player1'],
            'player2': c['player2'],
            'tournament': c['tournament'],
            'model_p1': round(c['model_p1'], 4),
            'model_p2': round(c['model_p2'], 4),
            'market_p1': round(c['market_p1'], 4),
            'market_p2': round(c['market_p2'], 4),
            'edge_p1': round(edge1, 4),
            'edge_p2': round(edge2, 4),
            'max_edge': round(max_edge, 4),
            'value_player': value_player,
            'confidence': c.get('confidence', ''),
            'volume': round(c['volume'], 2),
            'liquidity': round(c['liquidity'], 2),
            'slug': c['slug'],
        })

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Comparison data saved to {output_path}")


# ── Prediction Log (Accuracy Tracking) ───────────────────────────────────────

PREDICTION_LOG = DATA_DIR / 'prediction_log.csv'

LOG_FIELDS = [
    'timestamp', 'slug', 'player1', 'player2', 'tournament',
    'model_p1', 'model_p2', 'market_p1', 'market_p2',
    'edge', 'value_player', 'confidence', 'liquidity', 'volume',
    'result',       # '' = pending, 'p1' or 'p2' = resolved winner, 'void' = voided
    'result_date',  # When the result was recorded
]


def load_prediction_log():
    """Load the persistent prediction log."""
    if not PREDICTION_LOG.exists():
        return []
    rows = []
    with open(PREDICTION_LOG, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_prediction_log(rows):
    """Save the prediction log."""
    with open(PREDICTION_LOG, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def log_predictions(comparisons):
    """Append new predictions to the persistent log, skipping duplicates."""
    existing = load_prediction_log()
    existing_slugs = {row['slug'] for row in existing}

    new_count = 0
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    for c in comparisons:
        if c['slug'] in existing_slugs:
            continue

        edge1 = c['model_p1'] - c['market_p1']
        edge2 = c['model_p2'] - c['market_p2']
        if abs(edge1) >= abs(edge2):
            edge = edge1
            value_player = c['player1']
        else:
            edge = edge2
            value_player = c['player2']

        existing.append({
            'timestamp': now,
            'slug': c['slug'],
            'player1': c['player1'],
            'player2': c['player2'],
            'tournament': c['tournament'],
            'model_p1': round(c['model_p1'], 4),
            'model_p2': round(c['model_p2'], 4),
            'market_p1': round(c['market_p1'], 4),
            'market_p2': round(c['market_p2'], 4),
            'edge': round(edge, 4),
            'value_player': value_player,
            'confidence': c.get('confidence', ''),
            'liquidity': round(c['liquidity'], 2),
            'volume': round(c['volume'], 2),
            'result': '',
            'result_date': '',
        })
        existing_slugs.add(c['slug'])
        new_count += 1

    save_prediction_log(existing)
    return new_count


def fetch_resolved_events(slugs):
    """Fetch events by slug to check if they've resolved."""
    results = {}
    for slug in slugs:
        event = fetch_event_by_slug(slug)
        if not event:
            continue

        markets = event.get('markets', [])
        if not markets:
            continue

        # Find moneyline market
        title = event.get('title', '')
        moneyline = None
        for m in markets:
            question = m.get('question', '')
            market_type = m.get('marketType', '')
            if market_type == 'moneyline' or question == title:
                moneyline = m
                break
        if not moneyline:
            for m in markets:
                outcomes = m.get('outcomes', '')
                if isinstance(outcomes, str):
                    try:
                        outcomes = json.loads(outcomes)
                    except json.JSONDecodeError:
                        continue
                if (isinstance(outcomes, list) and len(outcomes) == 2
                        and 'Over' not in outcomes[0]
                        and 'Under' not in outcomes[0]):
                    moneyline = m
                    break

        if not moneyline:
            continue

        # Check if resolved
        prices = moneyline.get('outcomePrices', '[]')
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except json.JSONDecodeError:
                continue

        if len(prices) != 2:
            continue

        try:
            p1_price = float(prices[0])
            p2_price = float(prices[1])
        except (ValueError, TypeError):
            continue

        # Resolved if one side is ~1.0 and other is ~0.0
        if p1_price >= 0.95 and p2_price <= 0.05:
            results[slug] = 'p1'
        elif p2_price >= 0.95 and p1_price <= 0.05:
            results[slug] = 'p2'
        # Check if voided (both around 0.5)
        elif moneyline.get('closed', False) and abs(p1_price - 0.5) < 0.05:
            results[slug] = 'void'

    return results


def check_results():
    """Check resolved markets and calculate accuracy/ROI stats."""
    log = load_prediction_log()
    if not log:
        print("\n  No predictions in log yet. Run the pipeline first to start tracking.")
        return

    # Find unresolved entries
    pending = [r for r in log if not r['result']]
    resolved = [r for r in log if r['result'] and r['result'] != 'void']
    voided = [r for r in log if r['result'] == 'void']

    print(f"\n{'='*80}")
    print(f"  PREDICTION LOG STATUS")
    print(f"{'='*80}")
    print(f"  Total logged:   {len(log)}")
    print(f"  Resolved:       {len(resolved)}")
    print(f"  Voided:         {len(voided)}")
    print(f"  Pending:        {len(pending)}")

    # Try to resolve pending entries
    if pending:
        print(f"\n  Checking {len(pending)} pending matches for results...")
        pending_slugs = [r['slug'] for r in pending]
        results = fetch_resolved_events(pending_slugs)

        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        newly_resolved = 0
        for row in log:
            if row['slug'] in results and not row['result']:
                row['result'] = results[row['slug']]
                row['result_date'] = now
                newly_resolved += 1

        if newly_resolved:
            save_prediction_log(log)
            print(f"  Newly resolved: {newly_resolved}")
            # Refresh counts
            resolved = [r for r in log if r['result'] and r['result'] != 'void']
            voided = [r for r in log if r['result'] == 'void']
            pending = [r for r in log if not r['result']]
            print(f"  Still pending:  {len(pending)}")
        else:
            print(f"  No new results found.")

    # Calculate accuracy/ROI for resolved entries
    if not resolved:
        print("\n  No resolved matches yet — accuracy stats will appear after matches complete.")
        return

    print(f"\n{'='*80}")
    print(f"  ACCURACY & ROI ANALYSIS ({len(resolved)} resolved matches)")
    print(f"{'='*80}")

    # Overall model accuracy (did the model-favorite win?)
    model_correct = 0
    market_correct = 0
    value_bet_wins = 0
    value_bet_total = 0
    total_roi = 0.0  # Simulated flat-bet ROI on value side

    # Breakdowns
    by_confidence = {'HIGH': [0, 0], 'MEDIUM': [0, 0], 'LOW': [0, 0]}
    by_edge_bucket = {'2-5%': [0, 0], '5-10%': [0, 0], '10%+': [0, 0]}
    by_liquidity = {'<$5K': [0, 0], '$5K-$20K': [0, 0], '$20K+': [0, 0]}

    results_detail = []

    for r in resolved:
        model_p1 = float(r['model_p1'])
        model_p2 = float(r['model_p2'])
        market_p1 = float(r['market_p1'])
        market_p2 = float(r['market_p2'])
        edge = float(r['edge'])
        liquidity = float(r['liquidity']) if r['liquidity'] else 0
        winner = r['result']  # 'p1' or 'p2'
        value_player = r['value_player']
        confidence = r.get('confidence', 'LOW')

        # Model favorite = player with higher model prob
        model_fav = 'p1' if model_p1 > model_p2 else 'p2'
        market_fav = 'p1' if market_p1 > market_p2 else 'p2'

        model_was_right = (model_fav == winner)
        market_was_right = (market_fav == winner)

        if model_was_right:
            model_correct += 1
        if market_was_right:
            market_correct += 1

        # Value bet analysis: did the value side (model edge) win?
        value_side = 'p1' if value_player == r['player1'] else 'p2'
        abs_edge = abs(edge)

        if abs_edge >= 0.02:
            value_bet_total += 1
            value_won = (value_side == winner)
            if value_won:
                value_bet_wins += 1
                # ROI: bet $1 at market price, win $(1/market_price - 1) profit
                market_price = market_p1 if value_side == 'p1' else market_p2
                if market_price > 0:
                    total_roi += (1.0 / market_price) - 1.0
            else:
                total_roi -= 1.0  # Lost the $1 bet

            # Confidence breakdown
            if confidence in by_confidence:
                by_confidence[confidence][1] += 1
                if value_won:
                    by_confidence[confidence][0] += 1

            # Edge bucket breakdown
            if abs_edge >= 0.10:
                bucket = '10%+'
            elif abs_edge >= 0.05:
                bucket = '5-10%'
            else:
                bucket = '2-5%'
            by_edge_bucket[bucket][1] += 1
            if value_won:
                by_edge_bucket[bucket][0] += 1

            # Liquidity breakdown
            if liquidity >= 20000:
                liq_bucket = '$20K+'
            elif liquidity >= 5000:
                liq_bucket = '$5K-$20K'
            else:
                liq_bucket = '<$5K'
            by_liquidity[liq_bucket][1] += 1
            if value_won:
                by_liquidity[liq_bucket][0] += 1

            results_detail.append({
                'player1': r['player1'],
                'player2': r['player2'],
                'tournament': r['tournament'],
                'value_player': value_player,
                'edge': abs_edge,
                'won': value_won,
                'market_price': market_p1 if value_side == 'p1' else market_p2,
                'liquidity': liquidity,
            })

    # Print overall stats
    n = len(resolved)
    print(f"\n  Model accuracy:    {model_correct}/{n} ({model_correct/n*100:.1f}%)")
    print(f"  Market accuracy:   {market_correct}/{n} ({market_correct/n*100:.1f}%)")

    if value_bet_total:
        vb_pct = value_bet_wins / value_bet_total * 100
        roi_pct = total_roi / value_bet_total * 100
        print(f"\n  Value Bets (edge >= 2%):")
        print(f"    Record:          {value_bet_wins}/{value_bet_total} ({vb_pct:.1f}%)")
        print(f"    Flat-bet ROI:    {roi_pct:+.1f}%")
        print(f"    Total P/L:       ${total_roi:+.2f} (on ${value_bet_total} wagered)")

        # Breakdown tables
        print(f"\n  By Confidence Level:")
        print(f"    {'Level':12s} {'W-L':10s} {'Win%':>8s}")
        print(f"    {'─'*32}")
        for level in ['HIGH', 'MEDIUM', 'LOW']:
            w, t = by_confidence[level]
            if t > 0:
                print(f"    {level:12s} {w}-{t-w:<8d} {w/t*100:>7.1f}%")
            else:
                print(f"    {level:12s} {'—':10s} {'—':>8s}")

        print(f"\n  By Edge Size:")
        print(f"    {'Bucket':12s} {'W-L':10s} {'Win%':>8s}")
        print(f"    {'─'*32}")
        for bucket in ['2-5%', '5-10%', '10%+']:
            w, t = by_edge_bucket[bucket]
            if t > 0:
                print(f"    {bucket:12s} {w}-{t-w:<8d} {w/t*100:>7.1f}%")
            else:
                print(f"    {bucket:12s} {'—':10s} {'—':>8s}")

        print(f"\n  By Liquidity:")
        print(f"    {'Bucket':12s} {'W-L':10s} {'Win%':>8s}")
        print(f"    {'─'*32}")
        for bucket in ['<$5K', '$5K-$20K', '$20K+']:
            w, t = by_liquidity[bucket]
            if t > 0:
                print(f"    {bucket:12s} {w}-{t-w:<8d} {w/t*100:>7.1f}%")
            else:
                print(f"    {bucket:12s} {'—':10s} {'—':>8s}")

        # Individual results
        print(f"\n  Recent Results:")
        print(f"    {'Match':45s} {'Value On':20s} {'Edge':>7s} {'Result':>8s}")
        print(f"    {'─'*82}")
        for rd in results_detail[-20:]:  # Show last 20
            match_str = f"{rd['player1']} vs {rd['player2']}"
            if len(match_str) > 43:
                match_str = match_str[:42] + '…'
            result_str = "✓ WON" if rd['won'] else "✗ LOST"
            print(f"    {match_str:45s} {rd['value_player']:20s} "
                  f"{rd['edge']*100:>+6.1f}% {result_str:>8s}")
    else:
        print("\n  No value bets (edge >= 2%) resolved yet.")


# ── Tournament metadata ───────────────────────────────────────────────────────

# Map tournament names (from Polymarket titles) to surface/level/slug
TOURNEY_META = {
    'BNP Paribas Open': {'surface': 'Hard', 'level': 'M', 'slug': 'indian-wells'},
    'Indian Wells': {'surface': 'Hard', 'level': 'M', 'slug': 'indian-wells'},
    'Miami Open': {'surface': 'Hard', 'level': 'M', 'slug': 'miami'},
    'Monte Carlo': {'surface': 'Clay', 'level': 'M', 'slug': 'monte-carlo'},
    'Monte-Carlo Masters': {'surface': 'Clay', 'level': 'M', 'slug': 'monte-carlo'},
    'Madrid Open': {'surface': 'Clay', 'level': 'M', 'slug': 'madrid'},
    'Italian Open': {'surface': 'Clay', 'level': 'M', 'slug': 'rome'},
    'Rome': {'surface': 'Clay', 'level': 'M', 'slug': 'rome'},
    'Canadian Open': {'surface': 'Hard', 'level': 'M', 'slug': 'canada'},
    'Cincinnati Open': {'surface': 'Hard', 'level': 'M', 'slug': 'cincinnati'},
    'Shanghai Masters': {'surface': 'Hard', 'level': 'M', 'slug': 'shanghai'},
    'Paris Masters': {'surface': 'Hard', 'level': 'M', 'slug': 'paris'},
    'Australian Open': {'surface': 'Hard', 'level': 'G', 'slug': 'australian-open'},
    'French Open': {'surface': 'Clay', 'level': 'G', 'slug': 'roland-garros'},
    'Roland Garros': {'surface': 'Clay', 'level': 'G', 'slug': 'roland-garros'},
    'Wimbledon': {'surface': 'Grass', 'level': 'G', 'slug': 'wimbledon'},
    'US Open': {'surface': 'Hard', 'level': 'G', 'slug': 'us-open'},
    'Dubai Tennis Championships': {'surface': 'Hard', 'level': 'A', 'slug': 'dubai'},
    'Dubai': {'surface': 'Hard', 'level': 'A', 'slug': 'dubai'},
    'Phoenix': {'surface': 'Hard', 'level': 'A', 'slug': 'phoenix'},
    'Cap Cana': {'surface': 'Hard', 'level': 'A', 'slug': 'cap-cana'},
    'Santiago': {'surface': 'Clay', 'level': 'A', 'slug': 'santiago'},
    'Cherbourg': {'surface': 'Hard', 'level': 'A', 'slug': 'cherbourg'},
    'Kigali': {'surface': 'Hard', 'level': 'A', 'slug': 'kigali'},
    'Kigali 2': {'surface': 'Hard', 'level': 'A', 'slug': 'kigali'},
    'Hersonissos': {'surface': 'Hard', 'level': 'A', 'slug': 'hersonissos'},
    'Hersonissos 2': {'surface': 'Hard', 'level': 'A', 'slug': 'hersonissos'},
    'Lugano': {'surface': 'Clay', 'level': 'A', 'slug': 'lugano'},
    'Marrakech': {'surface': 'Clay', 'level': 'A', 'slug': 'marrakech'},
    'Munich': {'surface': 'Clay', 'level': 'A', 'slug': 'munich'},
    'Barcelona': {'surface': 'Clay', 'level': 'A', 'slug': 'barcelona'},
    'Houston': {'surface': 'Clay', 'level': 'A', 'slug': 'houston'},
    'Estoril': {'surface': 'Clay', 'level': 'A', 'slug': 'estoril'},
    'Lyon': {'surface': 'Clay', 'level': 'A', 'slug': 'lyon'},
    'Geneva': {'surface': 'Clay', 'level': 'A', 'slug': 'geneva'},
    'Stuttgart': {'surface': 'Grass', 'level': 'A', 'slug': 'stuttgart'},
    "Queen's Club": {'surface': 'Grass', 'level': 'A', 'slug': 'queens-club'},
    'Halle': {'surface': 'Grass', 'level': 'A', 'slug': 'halle'},
    'Eastbourne': {'surface': 'Grass', 'level': 'A', 'slug': 'eastbourne'},
    'Newport': {'surface': 'Grass', 'level': 'A', 'slug': 'newport'},
    'Washington': {'surface': 'Hard', 'level': 'A', 'slug': 'washington'},
    'Atlanta': {'surface': 'Hard', 'level': 'A', 'slug': 'atlanta'},
    'Los Cabos': {'surface': 'Hard', 'level': 'A', 'slug': 'los-cabos'},
    'Winston-Salem': {'surface': 'Hard', 'level': 'A', 'slug': 'winston-salem'},
}


def get_tourney_meta(tourney_name):
    """Get tournament metadata, with fallback defaults."""
    if tourney_name in TOURNEY_META:
        return TOURNEY_META[tourney_name]
    # Fuzzy match: check if any key is contained in the tourney name
    for key, meta in TOURNEY_META.items():
        if key.lower() in tourney_name.lower():
            return meta
    # Default fallback
    return {'surface': 'Hard', 'level': 'A', 'slug': tourney_name.lower().replace(' ', '-')}


# ── Match generation for predict.py ───────────────────────────────────────────

def full_name_to_flashscore(full_name):
    """Convert Polymarket full name to Flashscore-like format.

    "Carlos Alcaraz" -> "Alcaraz C."
    "Roberto Bautista Agut" -> "Bautista Agut R."
    "Jay Dylan Friend" -> "Friend J."
    """
    parts = full_name.strip().split()
    if not parts:
        return full_name

    if len(parts) == 1:
        return parts[0]

    # Common known multi-word first names (given names with 2 words)
    known_two_word_first = {
        'jay dylan', 'pedro boscardin', 'nicolas moreno',
        'santiago rodriguez', 'genaro alberto', 'felipe meligeni',
        'nikolas sanchez', 'diego dedura',
    }

    first_lower = ' '.join(parts[:2]).lower()
    if len(parts) >= 3 and first_lower in known_two_word_first:
        # First 2 words are the given name, rest is surname
        first_initial = parts[0][0].upper()
        last_name = ' '.join(parts[2:])
        return f"{last_name} {first_initial}."

    # Default: first word is given name, rest is surname
    first_initial = parts[0][0].upper()
    last_name = ' '.join(parts[1:])
    return f"{last_name} {first_initial}."


def generate_upcoming_matches_csv(atp_matches, output_path):
    """Write Polymarket ATP matches to CSV in predict.py's expected format."""
    fieldnames = [
        'player1_name', 'player2_name', 'tourney_name', 'tourney_slug',
        'surface', 'round', 'best_of', 'tourney_level', 'match_date',
        'scheduled_time',
    ]

    rows = []
    today = datetime.now()
    for m in atp_matches:
        # Skip already-resolved matches (price at 0 or 1)
        if m['p1_implied_prob'] <= 0.01 or m['p2_implied_prob'] <= 0.01:
            continue

        meta = get_tourney_meta(m['tournament'])

        # Convert full names to Flashscore format for predict.py
        p1_flash = full_name_to_flashscore(m['player1'])
        p2_flash = full_name_to_flashscore(m['player2'])

        # Parse date from slug (atp-player1-player2-YYYY-MM-DD)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})$', m['slug'])
        if date_match:
            match_date = date_match.group(1).replace('-', '')
        else:
            match_date = today.strftime('%Y%m%d')

        rows.append({
            'player1_name': p1_flash,
            'player2_name': p2_flash,
            'tourney_name': m['tournament'],
            'tourney_slug': meta['slug'],
            'surface': meta['surface'],
            'round': '',
            'best_of': 3,
            'tourney_level': meta['level'],
            'match_date': match_date,
            'scheduled_time': '',
        })

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def run_predictions(matches_csv, predictions_csv, load_state=True):
    """Run predict.py on the generated matches CSV."""
    cmd = [
        sys.executable, 'predict.py',
        '--input', str(matches_csv),
        '--output', str(predictions_csv),
    ]
    if load_state and Path('player_state.pkl').exists():
        cmd.append('--load-state')
    else:
        cmd.extend(['--save-state'])

    print(f"\n  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"  predict.py failed (exit code {result.returncode}):")
        print(result.stderr[-500:] if result.stderr else "No error output")
        return False

    # Print last few lines of output
    lines = result.stdout.strip().split('\n')
    for line in lines[-15:]:
        print(f"  {line}")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def load_predictions(path):
    """Load model predictions from CSV."""
    predictions = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append(row)
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description='Compare model predictions with Polymarket tennis odds')
    parser.add_argument('--predictions', type=str,
                        default=str(DATA_DIR / 'predictions.csv'),
                        help='Path to model predictions CSV')
    parser.add_argument('--output', type=str,
                        default=str(DATA_DIR / 'value_bets.csv'),
                        help='Output CSV for comparison data')
    parser.add_argument('--fetch-only', action='store_true',
                        help='Only fetch and display Polymarket odds (no comparison)')
    parser.add_argument('--no-predict', action='store_true',
                        help='Skip running predict.py (use existing predictions)')
    parser.add_argument('--min-edge', type=float, default=0,
                        help='Minimum edge percentage to display (e.g., 5 for 5%%)')
    parser.add_argument('--min-liquidity', type=float, default=0,
                        help='Minimum market liquidity in $ (e.g., 5000 for $5K)')
    parser.add_argument('--check-results', action='store_true',
                        help='Check resolved matches and show accuracy/ROI stats')
    args = parser.parse_args()

    # ── Check results mode ────────────────────────────────────────────────
    if args.check_results:
        check_results()
        return

    # ── Step 1: Fetch Polymarket tennis odds ──────────────────────────────
    print("Fetching Polymarket tennis markets...")
    events = fetch_tennis_events(limit=100)
    print(f"  Found {len(events)} tennis events")

    atp_matches = extract_atp_matches(events)
    print(f"  ATP matches with moneyline odds: {len(atp_matches)}")

    if not atp_matches:
        print("\nNo active ATP match markets found on Polymarket.")
        sys.exit(0)

    # Filter out already-resolved markets (price at 0/1)
    active_matches = [m for m in atp_matches
                      if 0.01 < m['p1_implied_prob'] < 0.99]
    resolved = len(atp_matches) - len(active_matches)
    if resolved:
        print(f"  Skipping {resolved} already-resolved markets")

    # Liquidity filter
    if args.min_liquidity > 0:
        pre_filter = len(active_matches)
        active_matches = [m for m in active_matches
                          if m['liquidity'] >= args.min_liquidity]
        filtered = pre_filter - len(active_matches)
        if filtered:
            print(f"  Filtered {filtered} markets below ${args.min_liquidity:,.0f} liquidity")

    print(f"  Active markets: {len(active_matches)}")

    if not active_matches:
        print("\nNo markets meet the filters.")
        sys.exit(0)

    # Display Polymarket odds
    print(f"\n{'='*70}")
    print(f"  POLYMARKET ATP MATCH ODDS")
    if args.min_liquidity > 0:
        print(f"  (liquidity >= ${args.min_liquidity:,.0f})")
    print(f"{'='*70}")

    by_tourney = {}
    for m in active_matches:
        t = m['tournament'] or 'Unknown'
        if t not in by_tourney:
            by_tourney[t] = []
        by_tourney[t].append(m)

    for tourney, matches in by_tourney.items():
        print(f"\n  {tourney}")
        print(f"  {'─'*60}")
        for m in matches:
            vol_str = f"${m['volume']:,.0f}" if m['volume'] else ""
            liq_str = f"[${m['liquidity']:,.0f}]" if m['liquidity'] else ""
            print(f"    {m['player1']:25s} {format_pct(m['p1_implied_prob']):>7s}  |  "
                  f"{m['player2']:25s} {format_pct(m['p2_implied_prob']):>7s}  {vol_str}  {liq_str}")

    if args.fetch_only:
        return

    # ── Step 2: Generate upcoming matches & run predictions ───────────────
    matches_csv = DATA_DIR / 'upcoming_matches.csv'
    pred_path = Path(args.predictions)

    if not args.no_predict:
        print(f"\nGenerating match file for predict.py...")
        n_matches = generate_upcoming_matches_csv(active_matches, matches_csv)
        print(f"  Wrote {n_matches} matches to {matches_csv}")

        print(f"\nRunning model predictions...")
        success = run_predictions(matches_csv, pred_path)
        if not success:
            print("Failed to generate predictions. Use --no-predict with existing predictions.")
            sys.exit(1)

    # ── Step 3: Load model predictions ────────────────────────────────────
    if not pred_path.exists():
        print(f"\nPredictions file not found: {pred_path}")
        print("Run without --no-predict to auto-generate predictions.")
        sys.exit(1)

    predictions = load_predictions(pred_path)
    print(f"\n  Loaded {len(predictions)} model predictions from {pred_path}")

    # ── Step 4: Match and compare ─────────────────────────────────────────
    comparisons = []
    unmatched_poly = []
    matched_count = 0

    for pm in active_matches:
        result, swapped = match_players(pm, predictions)
        if result is None:
            unmatched_poly.append(pm)
            continue

        matched_count += 1

        # Get model probabilities (handle swapped order)
        try:
            if swapped:
                model_p1 = float(result['p2_win_prob'])
                model_p2 = float(result['p1_win_prob'])
            else:
                model_p1 = float(result['p1_win_prob'])
                model_p2 = float(result['p2_win_prob'])
        except (ValueError, KeyError):
            continue

        comparisons.append({
            'player1': pm['player1'],
            'player2': pm['player2'],
            'tournament': pm['tournament'],
            'slug': pm['slug'],
            'model_p1': model_p1,
            'model_p2': model_p2,
            'market_p1': pm['p1_implied_prob'],
            'market_p2': pm['p2_implied_prob'],
            'volume': pm['volume'],
            'liquidity': pm['liquidity'],
            'confidence': result.get('confidence', ''),
        })

    print(f"  Matched: {matched_count}/{len(active_matches)} Polymarket markets")

    if unmatched_poly:
        print(f"\n  Unmatched Polymarket markets:")
        for um in unmatched_poly:
            print(f"    - {um['player1']} vs {um['player2']} ({um['tournament']})")

    # ── Step 5: Display comparison ────────────────────────────────────────
    print_comparison(comparisons, min_edge=args.min_edge)

    # ── Step 6: Save to CSV ───────────────────────────────────────────────
    save_comparisons(comparisons, args.output)

    # ── Step 7: Log predictions for accuracy tracking ─────────────────────
    new_logged = log_predictions(comparisons)
    total_logged = len(load_prediction_log())
    print(f"\n  Prediction log: {new_logged} new entries added ({total_logged} total)")
    print(f"  Run with --check-results to see accuracy after matches resolve.")


if __name__ == '__main__':
    main()
