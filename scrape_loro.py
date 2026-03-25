"""
Fetch Loterie Romande (JouezSport) tennis odds via their public API.

Uses the calendar endpoint to get ALL matches (today + tomorrow),
not just the featured "best-bets" selection.

No browser, no captcha, no cookies — just HTTP requests.

Usage:
    python3 scrape_loro.py              # Fetch LORO tennis odds
    python3 scrape_loro.py --json       # Output as JSON
    python3 scrape_loro.py --all        # Show all sports, not just tennis
    python3 scrape_loro.py --days 3     # Look ahead 3 days
"""

import argparse
import json
import sys
from datetime import datetime, timedelta

import requests

# ── Configuration ────────────────────────────────────────────────────────────

LORO_CALENDAR = "https://jeux.loro.ch/api/sport/sports/events/calendar"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "fr-CH",
    "Referer": "https://jeux.loro.ch/sports",
}

# ATP categories to include (skip doubles "DH" and other non-singles)
ATP_SINGLES_LEAGUES = {"ATP Miami", "ATP"}


def fetch_loro_odds(sport_filter="TENN", atp_only=True, days_ahead=2):
    """Fetch odds from LORO's calendar API for today + upcoming days.

    Args:
        sport_filter: Sport code to filter ("TENN", "FOOT", None for all).
        atp_only: If True, only return ATP singles matches.
        days_ahead: Number of days to look ahead (default: 2 = today + tomorrow).

    Returns:
        List of match dicts with odds.
    """
    matches = []
    seen = set()

    for day_offset in range(days_ahead):
        date = (datetime.now() + timedelta(days=day_offset)).strftime("%Y-%m-%d")
        params = {"date": date}
        if sport_filter:
            params["sportCode"] = sport_filter

        try:
            resp = requests.get(LORO_CALENDAR, headers=HEADERS,
                                params=params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  LORO API error ({date}): {e}")
            continue

        data = resp.json()
        if not isinstance(data, dict):
            continue

        events = data.get("events", [])

        for event in events:
            if not isinstance(event, dict):
                continue

            sport_code = event.get("sportCode", "")
            if sport_filter and sport_code != sport_filter:
                continue

            category = event.get("sportCategory", "")
            league = event.get("leagueName", "")

            # Filter: ATP singles only (skip doubles "DH", "DF")
            if atp_only and sport_filter == "TENN":
                if "DH" in league or "DF" in league:
                    continue
                if category != "ATP":
                    continue

            markets = event.get("markets", [])
            for market in markets:
                if not isinstance(market, dict):
                    continue

                outcomes = market.get("outcomes", [])
                if len(outcomes) != 2:
                    continue

                try:
                    p1, p2 = outcomes[0], outcomes[1]
                    name1 = p1.get("opponent", "")
                    name2 = p2.get("opponent", "")
                    odds1 = float(p1.get("price", 0))
                    odds2 = float(p2.get("price", 0))

                    if not name1 or not name2 or odds1 <= 1 or odds2 <= 1:
                        continue

                    key = (name1, name2)
                    if key in seen:
                        continue
                    seen.add(key)

                    matches.append({
                        "home": name1,
                        "away": name2,
                        "home_odds": odds1,
                        "away_odds": odds2,
                        "home_prob": round(1.0 / odds1, 4),
                        "away_prob": round(1.0 / odds2, 4),
                        "tournament": league,
                        "date": date,
                        "source": "loro",
                    })
                except (ValueError, TypeError):
                    continue

    return matches


# Backwards compatibility with arbitrage_engine
intercept_loro_odds = lambda **kwargs: fetch_loro_odds()


def print_matches(matches):
    """Display LORO matches."""
    if not matches:
        print("No LORO tennis matches found.")
        return

    print(f"\n{'Match':<45} {'LORO Odds':>12} {'Implied':>14} {'Tournament':>20}")
    print("-" * 95)

    for m in matches:
        matchup = f"{m['home']} vs {m['away']}"
        if len(matchup) > 42:
            matchup = matchup[:39] + "..."
        odds_str = f"{m['home_odds']:.2f}/{m['away_odds']:.2f}"
        prob_str = f"{m['home_prob']:.1%}/{m['away_prob']:.1%}"
        tourney = m.get("tournament", "")
        if len(tourney) > 18:
            tourney = tourney[:15] + "..."

        print(f"  {matchup:<43} {odds_str:>12} {prob_str:>14} {tourney:>20}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Loterie Romande tennis odds")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--all", action="store_true",
                        help="Show all sports, not just tennis")
    parser.add_argument("--days", type=int, default=2,
                        help="Days to look ahead (default: 2)")
    args = parser.parse_args()

    sport = None if args.all else "TENN"
    atp = not args.all
    print("Fetching LORO odds...")
    matches = fetch_loro_odds(sport_filter=sport, atp_only=atp,
                              days_ahead=args.days)

    if args.json:
        print(json.dumps(matches, indent=2))
    else:
        print_matches(matches)

    print(f"\n  Total: {len(matches)} LORO matches")


if __name__ == "__main__":
    main()
