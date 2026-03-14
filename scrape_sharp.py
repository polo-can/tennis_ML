"""
Fetch sharp bookmaker odds (Pinnacle) via The Odds API.

Queries the-odds-api.com for ATP tennis moneyline odds, filters for Pinnacle,
and returns implied probabilities with vig removed.

Usage:
    python3 scrape_sharp.py                  # Fetch and display Pinnacle odds
    python3 scrape_sharp.py --sport tennis_wta  # WTA instead of ATP
    python3 scrape_sharp.py --all-books      # Show all bookmakers, not just Pinnacle
    python3 scrape_sharp.py --json           # Output as JSON

Requires ODDS_API_KEY in .env file. Free tier: 500 requests/month.
Sign up at https://the-odds-api.com
"""

import argparse
import json
import os
import sys
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE = "https://api.the-odds-api.com/v4"
DEFAULT_BOOKMAKER = "pinnacle"


def _get_api_key():
    """Get API key from environment."""
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        print("Error: ODDS_API_KEY not set in .env file")
        sys.exit(1)
    return api_key


def fetch_active_tennis_sports():
    """Discover all currently active tennis sport keys.

    The Odds API uses per-tournament keys like 'tennis_atp_indian_wells'
    rather than a generic 'tennis_atp' key.

    Returns:
        list of sport key strings (e.g. ['tennis_atp_indian_wells', ...])
    """
    api_key = _get_api_key()
    url = f"{API_BASE}/sports/"
    try:
        resp = requests.get(url, params={"apiKey": api_key}, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  Error fetching sports list: {e}")
        return []

    sports = resp.json()
    tennis = [s["key"] for s in sports
              if s["key"].startswith("tennis_atp") and s.get("active")]
    print(f"  Active ATP tennis sports: {len(tennis)}")
    for t in tennis:
        title = next((s["title"] for s in sports if s["key"] == t), t)
        print(f"    {t} — {title}")
    return tennis


def fetch_odds(sport, regions="eu", markets="h2h", bookmakers=None):
    """Fetch odds from The Odds API for a single sport key.

    Args:
        sport: Sport key (e.g. 'tennis_atp_indian_wells')
        regions: Odds region ('eu' for decimal, 'us' for American)
        markets: Market type ('h2h' for moneyline)
        bookmakers: Comma-separated bookmaker keys (e.g. 'pinnacle')

    Returns:
        list of event dicts, or empty list on error
    """
    api_key = _get_api_key()

    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "decimal",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers

    url = f"{API_BASE}/sports/{sport}/odds/"

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  Error fetching odds for {sport}: {e}")
        return []

    # Log remaining API quota
    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    print(f"  API quota: {remaining} remaining, {used} used")

    return resp.json()


def fetch_all_tennis_odds(bookmakers=None):
    """Fetch odds across all active ATP tennis tournaments.

    Auto-discovers active tournament keys and queries each one.
    Each API call counts against the monthly quota.

    Returns:
        list of all event dicts across all tournaments
    """
    sports = fetch_active_tennis_sports()
    if not sports:
        return []

    all_events = []
    for sport in sports:
        events = fetch_odds(sport, bookmakers=bookmakers)
        all_events.extend(events)

    return all_events


def implied_probability(decimal_odds):
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 1.0:
        return 1.0
    return 1.0 / decimal_odds


def remove_vig(prob1, prob2):
    """Remove bookmaker vig by normalizing probabilities to sum to 1.0."""
    total = prob1 + prob2
    if total == 0:
        return 0.5, 0.5
    return prob1 / total, prob2 / total


def extract_pinnacle_lines(events, bookmaker=DEFAULT_BOOKMAKER):
    """Extract moneyline odds for a specific bookmaker from API response.

    Args:
        events: List of event dicts from fetch_odds()
        bookmaker: Bookmaker key to filter for

    Returns:
        List of dicts with match info and fair probabilities:
        [{
            'home': str,        # Home player name
            'away': str,        # Away player name
            'commence': str,    # Match start time (ISO)
            'home_odds': float, # Raw decimal odds
            'away_odds': float,
            'home_prob': float, # Vig-free implied probability
            'away_prob': float,
            'bookmaker': str,
        }]
    """
    matches = []

    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        commence = event.get("commence_time", "")

        for bm in event.get("bookmakers", []):
            if bm["key"] != bookmaker:
                continue

            for market in bm.get("markets", []):
                if market["key"] != "h2h":
                    continue

                outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                home_odds = outcomes.get(home, 0)
                away_odds = outcomes.get(away, 0)

                if home_odds <= 1.0 or away_odds <= 1.0:
                    continue

                raw_home = implied_probability(home_odds)
                raw_away = implied_probability(away_odds)
                fair_home, fair_away = remove_vig(raw_home, raw_away)

                matches.append({
                    "home": home,
                    "away": away,
                    "commence": commence,
                    "home_odds": home_odds,
                    "away_odds": away_odds,
                    "home_prob": round(fair_home, 4),
                    "away_prob": round(fair_away, 4),
                    "bookmaker": bookmaker,
                })
                break  # Only one h2h market per bookmaker
            break  # Found our bookmaker

    return matches


def get_sharp_lines(bookmaker=DEFAULT_BOOKMAKER):
    """Main entry point: fetch Pinnacle odds across all active ATP tournaments.

    Returns:
        List of match dicts with vig-free probabilities.
    """
    events = fetch_all_tennis_odds(bookmakers=bookmaker)
    if not events:
        print("  No events found")
        return []

    matches = extract_pinnacle_lines(events, bookmaker=bookmaker)
    print(f"  Found {len(matches)} matches with {bookmaker} odds")
    return matches


def print_matches(matches):
    """Display matches in a readable table."""
    if not matches:
        print("No matches to display.")
        return

    print(f"\n{'Match':<45} {'Odds':>12} {'Fair Prob':>14} {'Book':>10}")
    print("-" * 85)

    for m in sorted(matches, key=lambda x: x["commence"]):
        try:
            dt = datetime.fromisoformat(m["commence"].replace("Z", "+00:00"))
            time_str = dt.strftime("%b %d %H:%M")
        except (ValueError, AttributeError):
            time_str = m["commence"][:16] if m["commence"] else "?"

        matchup = f"{m['home']} vs {m['away']}"
        if len(matchup) > 42:
            matchup = matchup[:39] + "..."

        odds_str = f"{m['home_odds']:.2f}/{m['away_odds']:.2f}"
        prob_str = f"{m['home_prob']:.1%}/{m['away_prob']:.1%}"

        print(f"  {matchup:<43} {odds_str:>12} {prob_str:>14} {m['bookmaker']:>10}")
        print(f"  {'':43} {time_str:>12}")


def main():
    parser = argparse.ArgumentParser(description="Fetch sharp bookmaker odds")
    parser.add_argument("--sport", default=None,
                        help="Specific sport key (default: auto-discover all ATP)")
    parser.add_argument("--all-books", action="store_true",
                        help="Fetch all bookmakers, not just Pinnacle")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args()

    bookmaker = None if args.all_books else DEFAULT_BOOKMAKER

    if args.sport:
        print(f"Fetching {args.sport} odds...")
        events = fetch_odds(sport=args.sport,
                            bookmakers=bookmaker if bookmaker else None)
    else:
        print("Fetching all active ATP tennis odds...")
        events = fetch_all_tennis_odds(bookmakers=bookmaker if bookmaker else None)

    if not events:
        print("No events found.")
        return

    if args.all_books:
        all_books = set()
        for event in events:
            for bm in event.get("bookmakers", []):
                all_books.add(bm["key"])
        print(f"  Available bookmakers: {', '.join(sorted(all_books))}")

    matches = extract_pinnacle_lines(events, bookmaker=DEFAULT_BOOKMAKER)

    if args.json:
        print(json.dumps(matches, indent=2))
    else:
        print_matches(matches)

    print(f"\n  Total: {len(matches)} matches with Pinnacle lines")


if __name__ == "__main__":
    main()
