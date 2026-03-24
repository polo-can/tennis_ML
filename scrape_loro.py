"""
Fetch Loterie Romande (JouezSport) tennis odds via their public API.

No browser, no captcha, no cookies — just a single HTTP request.

Usage:
    python3 scrape_loro.py              # Fetch LORO tennis odds
    python3 scrape_loro.py --json       # Output as JSON
    python3 scrape_loro.py --all        # Show all sports, not just tennis
"""

import argparse
import json
import sys

import requests

# ── Configuration ────────────────────────────────────────────────────────────

LORO_API = "https://jeux.loro.ch/api/sport/sports/events/best-bets"

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


def fetch_loro_odds(sport_filter="TENN"):
    """Fetch odds from LORO's public API.

    Args:
        sport_filter: Sport code to filter for ("TENN", "FOOT", etc.)
                      Use None for all sports.

    Returns:
        List of match dicts with odds.
    """
    try:
        resp = requests.get(LORO_API, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  LORO API error: {e}")
        return []

    data = resp.json()
    if not isinstance(data, list):
        print(f"  LORO API: unexpected response type {type(data).__name__}")
        return []

    matches = []
    seen = set()

    for sport in data:
        if not isinstance(sport, dict):
            continue

        sport_code = sport.get("sportCode", "")
        if sport_filter and sport_code != sport_filter:
            continue

        for path in sport.get("eventPaths", []):
            if not isinstance(path, dict):
                continue

            league = path.get("leagueName", "")

            for event in path.get("events", []):
                if not isinstance(event, dict):
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
                            "source": "loro",
                        })
                    except (ValueError, TypeError):
                        continue

    return matches


# Keep old function name for backwards compatibility with arbitrage_engine
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
    args = parser.parse_args()

    sport = None if args.all else "TENN"
    print("Fetching LORO odds...")
    matches = fetch_loro_odds(sport_filter=sport)

    if args.json:
        print(json.dumps(matches, indent=2))
    else:
        print_matches(matches)

    print(f"\n  Total: {len(matches)} LORO matches")


if __name__ == "__main__":
    main()
