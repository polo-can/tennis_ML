"""
Fetch Polymarket tennis odds as a read-only sharp signal.

No betting — just reads the prediction market price for consensus building.
Free API, no key needed, no rate limit concerns.

Usage:
    python3 scrape_polymarket.py              # Show active ATP markets
    python3 scrape_polymarket.py --json       # Output as JSON
    python3 scrape_polymarket.py --min-liq 5000  # Only high-liquidity markets
"""

import argparse
import json
import re
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

# ── Configuration ────────────────────────────────────────────────────────────

GAMMA_API = "https://gamma-api.polymarket.com"
TENNIS_TAG_ID = 864
MIN_LIQUIDITY = 1000  # $1K minimum to consider the price meaningful


def fetch_polymarket_odds(min_liquidity=MIN_LIQUIDITY):
    """Fetch ATP tennis match odds from Polymarket.

    Returns:
        List of match dicts compatible with the arbitrage engine:
        [{
            'home': str,           # Player 1
            'away': str,           # Player 2
            'home_prob': float,    # Implied probability
            'away_prob': float,
            'home_odds': float,    # Decimal odds
            'away_odds': float,
            'tournament': str,
            'volume': float,       # Total volume in USD
            'liquidity': float,    # Current liquidity in USD
            'source': 'polymarket',
        }]
    """
    params = urlencode({
        'active': 'true',
        'closed': 'false',
        'limit': '100',
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
            events = json.loads(resp.read().decode())
    except (HTTPError, URLError, json.JSONDecodeError) as e:
        print(f"  Polymarket API error: {e}")
        return []

    if not isinstance(events, list):
        return []

    matches = []
    for event in events:
        slug = event.get('slug', '')

        # Only ATP matches
        if not slug.startswith('atp-'):
            continue

        title = event.get('title', '')
        markets = event.get('markets', [])
        if not markets:
            continue

        # Find the moneyline market
        moneyline = None
        for m in markets:
            question = m.get('question', '')
            market_type = m.get('marketType', '')
            if market_type == 'moneyline' or question == title:
                moneyline = m
                break

        # Fallback: first market with 2 player outcomes
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
            p1_prob = float(prices[0])
            p2_prob = float(prices[1])
        except (ValueError, TypeError):
            continue

        # Skip low-probability noise
        if p1_prob <= 0.01 or p2_prob <= 0.01:
            continue

        # Liquidity filter
        liquidity = float(moneyline.get('liquidity', 0) or 0)
        volume = float(moneyline.get('volume', 0) or 0)
        if liquidity < min_liquidity:
            continue

        # Convert to decimal odds
        p1_odds = round(1.0 / p1_prob, 2) if p1_prob > 0 else 0
        p2_odds = round(1.0 / p2_prob, 2) if p2_prob > 0 else 0

        # Tournament from title
        tourney = ''
        if ':' in title:
            tourney = title.split(':')[0].strip()

        matches.append({
            'home': outcomes[0],
            'away': outcomes[1],
            'home_prob': round(p1_prob, 4),
            'away_prob': round(p2_prob, 4),
            'home_odds': p1_odds,
            'away_odds': p2_odds,
            'tournament': tourney,
            'volume': volume,
            'liquidity': liquidity,
            'source': 'polymarket',
        })

    return matches


def print_matches(matches):
    """Display Polymarket matches."""
    if not matches:
        print("No Polymarket ATP tennis markets found.")
        return

    print(f"\n{'Match':<40} {'Prob':>12} {'Odds':>12} {'Volume':>10} {'Liq':>10}")
    print("-" * 88)

    for m in matches:
        matchup = f"{m['home']} vs {m['away']}"
        if len(matchup) > 38:
            matchup = matchup[:35] + "..."
        prob_str = f"{m['home_prob']:.0%}/{m['away_prob']:.0%}"
        odds_str = f"{m['home_odds']:.2f}/{m['away_odds']:.2f}"
        vol = f"${m['volume']:,.0f}"
        liq = f"${m['liquidity']:,.0f}"
        print(f"  {matchup:<38} {prob_str:>12} {odds_str:>12} {vol:>10} {liq:>10}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Polymarket ATP tennis odds")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--min-liq", type=float, default=MIN_LIQUIDITY,
                        help=f"Minimum liquidity in USD (default: {MIN_LIQUIDITY})")
    args = parser.parse_args()

    print("Fetching Polymarket tennis odds...")
    matches = fetch_polymarket_odds(min_liquidity=args.min_liq)

    if args.json:
        print(json.dumps(matches, indent=2))
    else:
        print_matches(matches)

    print(f"\n  Total: {len(matches)} Polymarket ATP markets")


if __name__ == "__main__":
    main()
