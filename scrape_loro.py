"""
Intercept Loterie Romande (JouezSport) tennis odds via Playwright.

Instead of parsing the DOM, this script listens to background network traffic
when loading the JouezSport tennis page, intercepts the OpenBet JSON payload
containing match odds, and extracts them programmatically.

Usage:
    python3 scrape_loro.py                     # Fetch LORO tennis odds
    python3 scrape_loro.py --discover          # Discovery mode: log all network responses
    python3 scrape_loro.py --headed            # Run with visible browser (for debugging)
    python3 scrape_loro.py --json              # Output as JSON
    python3 scrape_loro.py --timeout 30        # Wait longer for page load (seconds)

Requires: playwright install chromium
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

# ── Configuration ────────────────────────────────────────────────────────────

LORO_BASE = "https://jeux.loro.ch"
LORO_SPORTS = f"{LORO_BASE}/sports"
LORO_LIVE = f"{LORO_BASE}/sports/online/live"

# Alternative: scrape LORO odds from the-odds-api.com if available,
# or from the LORO mobile/API endpoint which may skip captcha
LORO_MOBILE = "https://jeux.loro.ch/api/v2/sportsbook"

# URL patterns that typically contain OpenBet/SBTech odds payloads
# These will be refined after running --discover mode
ODDS_URL_PATTERNS = [
    r"/api/",
    r"/sportsbook/",
    r"/openbet/",
    r"/sbtech/",
    r"/odds",
    r"/events",
    r"/matches",
    r"/GetEvents",
    r"/sports-data/",
    r"/feed/",
    r"\.json",
]

# Keywords that indicate a response contains odds data
ODDS_KEYWORDS = [
    "odds", "price", "outcome", "selection", "market",
    "tennis", "match", "event", "handicap", "moneyline",
]

DEFAULT_TIMEOUT = 20  # seconds to wait for network traffic


def is_odds_url(url):
    """Check if a URL likely contains odds data."""
    return any(re.search(p, url, re.IGNORECASE) for p in ODDS_URL_PATTERNS)


def looks_like_odds_payload(data):
    """Heuristic check if a JSON response contains odds data."""
    text = json.dumps(data).lower() if isinstance(data, (dict, list)) else str(data).lower()
    matches = sum(1 for kw in ODDS_KEYWORDS if kw in text)
    return matches >= 3


def discover_endpoints(headed=True, timeout=DEFAULT_TIMEOUT):
    """Discovery mode: log ALL network responses to identify the odds endpoint.

    Run this first to find which URLs return the tennis odds JSON payload,
    then update ODDS_URL_PATTERNS accordingly.
    """
    captured = []

    def handle_response(response):
        url = response.url
        content_type = response.headers.get("content-type", "")

        # Only interested in JSON/text responses
        if not any(t in content_type for t in ["json", "text", "javascript"]):
            return

        status = response.status
        size = len(response.headers.get("content-length", "?"))

        try:
            body = response.text()
            body_len = len(body)
        except Exception:
            body = ""
            body_len = 0

        # Try to parse as JSON
        is_json = False
        json_data = None
        try:
            json_data = json.loads(body)
            is_json = True
        except (json.JSONDecodeError, ValueError):
            pass

        entry = {
            "url": url,
            "status": status,
            "content_type": content_type,
            "body_length": body_len,
            "is_json": is_json,
            "has_odds_keywords": looks_like_odds_payload(json_data or body) if body else False,
        }
        captured.append(entry)

        # Log interesting responses in real-time
        flag = " *** ODDS?" if entry["has_odds_keywords"] else ""
        if is_json or entry["has_odds_keywords"]:
            print(f"  [{status}] {body_len:>8} bytes  {url[:120]}{flag}")

    print(f"Discovery mode: loading {LORO_LIVE}")
    print(f"Logging all network responses for {timeout}s...\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not headed)
        context = browser.new_context(
            locale="fr-CH",
            timezone_id="Europe/Zurich",
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()
        page.on("response", handle_response)

        try:
            page.goto(LORO_LIVE, wait_until="networkidle", timeout=timeout * 1000)
        except Exception as e:
            print(f"  Page load warning: {e}")

        # Give extra time for lazy-loaded XHR requests
        print(f"\n  Waiting {timeout}s for additional network traffic...")
        time.sleep(timeout)

        # Try navigating to tennis specifically
        print("\n  Attempting to find tennis section...")
        try:
            # Look for tennis links/buttons
            tennis_links = page.query_selector_all(
                'a[href*="tennis"], button:has-text("Tennis"), '
                '[data-sport*="tennis"], [class*="tennis"]'
            )
            if tennis_links:
                print(f"  Found {len(tennis_links)} tennis-related elements")
                tennis_links[0].click()
                time.sleep(10)
            else:
                print("  No tennis-specific links found on page")
        except Exception as e:
            print(f"  Tennis nav error: {e}")

        browser.close()

    # Summary
    print(f"\n{'='*80}")
    print(f"DISCOVERY SUMMARY: {len(captured)} responses captured")
    print(f"{'='*80}")

    json_responses = [r for r in captured if r["is_json"]]
    odds_candidates = [r for r in captured if r["has_odds_keywords"]]

    print(f"\n  JSON responses: {len(json_responses)}")
    print(f"  Potential odds payloads: {len(odds_candidates)}")

    if odds_candidates:
        print(f"\n  CANDIDATE ODDS ENDPOINTS:")
        for r in odds_candidates:
            print(f"    [{r['status']}] {r['body_length']:>8} bytes  {r['url'][:150]}")

    return captured


COOKIES_FILE = Path(__file__).parent / "data" / "loro_cookies.json"


def save_cookies(cookies, path=None):
    """Save browser cookies to JSON file for reuse."""
    path = Path(path or COOKIES_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cookies, f, indent=2)
    print(f"  Saved {len(cookies)} cookies to {path}")


def load_cookies(path=None):
    """Load saved cookies from JSON file."""
    path = Path(path or COOKIES_FILE)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            cookies = json.load(f)
        print(f"  Loaded {len(cookies)} cookies from {path}")
        return cookies
    except Exception:
        return None


def login_and_save_cookies(timeout=60):
    """Open LORO in headed browser, let user solve captcha, save cookies.

    Run this once on your local machine:
        python3 scrape_loro.py --login
    """
    print("Opening LORO in browser — solve the captcha manually...")
    print(f"You have {timeout} seconds. Navigate to the tennis section if possible.\n")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        context = browser.new_context(
            locale="fr-CH",
            timezone_id="Europe/Zurich",
        )
        page = context.new_page()
        page.goto(LORO_SPORTS, timeout=timeout * 1000)

        # Wait for user to solve captcha and page to load
        print("  Waiting for you to solve the captcha...")
        try:
            # Wait until we see actual sports content (not captcha page)
            page.wait_for_selector(
                'a[href*="tennis"], [class*="sport"], [data-sport]',
                timeout=timeout * 1000
            )
            print("  Captcha solved! Page loaded.")
        except Exception:
            print("  Timeout waiting for captcha — saving cookies anyway...")

        # Give extra time for all requests to complete
        time.sleep(3)

        cookies = context.cookies()
        save_cookies(cookies)

        # Try navigating to tennis to warm up the session
        try:
            tennis_el = page.query_selector(
                'a[href*="tennis"], button:has-text("Tennis")'
            )
            if tennis_el:
                tennis_el.click()
                time.sleep(5)
                # Save cookies again after tennis navigation
                cookies = context.cookies()
                save_cookies(cookies)
                print("  Navigated to tennis section, cookies updated.")
        except Exception:
            pass

        browser.close()

    print("\nCookies saved! You can now run the scraper (including on the server).")
    print("Copy data/loro_cookies.json to the server.")


def intercept_loro_odds(headless=True, timeout=DEFAULT_TIMEOUT):
    """Intercept LORO tennis odds from network traffic.

    Uses saved cookies to bypass captcha if available.

    Returns:
        List of match dicts:
        [{
            'home': str,
            'away': str,
            'home_odds': float,
            'away_odds': float,
            'home_prob': float,  # implied probability
            'away_prob': float,
            'tournament': str,
            'source': 'loro',
        }]
    """
    odds_payloads = []

    def handle_response(response):
        url = response.url
        content_type = response.headers.get("content-type", "")

        if "json" not in content_type and "javascript" not in content_type:
            return

        try:
            data = response.json()
        except Exception:
            return

        # Capture ALL JSON responses — the OpenBet endpoint URL patterns
        # may not match our guesses, so cast a wide net
        odds_payloads.append({
            "url": url,
            "data": data,
        })

    cookies = load_cookies()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        context = browser.new_context(
            locale="fr-CH",
            timezone_id="Europe/Zurich",
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )

        # Load saved cookies to bypass captcha
        if cookies:
            context.add_cookies(cookies)
            print("  Using saved cookies to bypass captcha")

        page = context.new_page()
        page.on("response", handle_response)

        try:
            page.goto(LORO_LIVE, wait_until="networkidle",
                      timeout=timeout * 1000)
        except Exception:
            pass

        # Try to navigate to tennis section
        try:
            tennis_el = page.query_selector(
                'a[href*="tennis"], button:has-text("Tennis"), '
                '[data-sport*="tennis"]'
            )
            if tennis_el:
                tennis_el.click()
                page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass

        # Extra wait for lazy XHR
        time.sleep(5)

        # Save updated cookies for next run
        try:
            new_cookies = context.cookies()
            save_cookies(new_cookies)
        except Exception:
            pass

        browser.close()

    # Parse all captured odds payloads
    matches = []
    for payload in odds_payloads:
        parsed = parse_loro_payload(payload["data"])
        matches.extend(parsed)

    print(f"  Intercepted {len(odds_payloads)} JSON responses, "
          f"extracted {len(matches)} tennis matches")
    if not matches and odds_payloads:
        # Debug: show captured URLs so we can refine
        print("  DEBUG — captured JSON endpoints:")
        for p in odds_payloads[:10]:
            url = p["url"][:120]
            data = p["data"]
            size = len(json.dumps(data)) if data else 0
            dtype = type(data).__name__
            print(f"    {size:>8} bytes  [{dtype}]  {url}")

        # Dump large payloads for analysis
        debug_dir = Path(__file__).parent / "data"
        debug_dir.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(odds_payloads):
            size = len(json.dumps(p["data"])) if p["data"] else 0
            if size > 5000:
                fname = debug_dir / f"loro_debug_{i}.json"
                with open(fname, "w") as f:
                    json.dump({"url": p["url"], "data": p["data"]}, f, indent=2)
                print(f"    Dumped {fname.name} ({size} bytes)")
    return matches


def _recursive_find_events(data, depth=0, max_depth=6):
    """Recursively search a JSON structure for lists that look like events."""
    if depth > max_depth:
        return []

    results = []
    if isinstance(data, list):
        # Check if this list contains event-like dicts
        if len(data) > 0 and isinstance(data[0], dict):
            results.append(data)
        for item in data:
            results.extend(_recursive_find_events(item, depth + 1, max_depth))
    elif isinstance(data, dict):
        for val in data.values():
            results.extend(_recursive_find_events(val, depth + 1, max_depth))
    return results


def _extract_odds_from_item(item):
    """Try to extract a 2-player match with odds from any dict structure."""
    if not isinstance(item, dict):
        return None

    text = json.dumps(item).lower()

    # Must look like a tennis match (check for common tennis terms)
    tennis_signals = ["tennis", "atp", "wta", "challenger", "itf",
                      "roland", "wimbledon", "open", "masters",
                      # French terms used by LORO
                      "vainqueur", "set", "jeu"]
    is_tennis = any(s in text for s in tennis_signals)

    # Also check: does the item contain exactly 2 names with odds?
    # This catches cases where sport type isn't labeled

    # Try multiple structural patterns to extract players + odds
    candidates = []

    # Pattern 1: item has markets/selections
    for mkey in ["markets", "eventMarkets", "betOffers", "bets", "odds"]:
        markets = item.get(mkey, [])
        if isinstance(markets, dict):
            markets = list(markets.values())
        if not isinstance(markets, list):
            continue
        for market in markets:
            if not isinstance(market, dict):
                continue
            for skey in ["selections", "eventSelections", "outcomes",
                         "runners", "picks", "options", "choices"]:
                sels = market.get(skey, [])
                if isinstance(sels, list) and len(sels) == 2:
                    candidates.append(sels)

    # Pattern 2: item directly has selections/outcomes
    for skey in ["selections", "outcomes", "runners", "participants"]:
        sels = item.get(skey, [])
        if isinstance(sels, list) and len(sels) == 2:
            candidates.append(sels)

    # Pattern 3: item has nested "odds" with player keys
    if "odds" in item and isinstance(item["odds"], dict):
        odds_dict = item["odds"]
        if len(odds_dict) == 2:
            sels = [{"name": k, "odds": v} for k, v in odds_dict.items()]
            candidates.append(sels)

    for sels in candidates:
        try:
            players = []
            for sel in sels:
                if not isinstance(sel, dict):
                    continue
                name = ""
                for nk in ["name", "selectionName", "label", "runnerName",
                           "participant", "player", "competitorName",
                           "teamName", "description"]:
                    if sel.get(nk):
                        name = str(sel[nk])
                        break

                odds = 0
                for ok in ["odds", "price", "decimalOdds", "trueOdds",
                           "decimal", "dec", "backOdds", "payout"]:
                    val = sel.get(ok)
                    if val:
                        if isinstance(val, dict):
                            val = val.get("decimal", val.get("dec", 0))
                        odds = float(val)
                        if odds > 1.0:
                            break

                if name and odds > 1.0:
                    players.append({"name": name, "odds": odds})

            if len(players) == 2:
                p1, p2 = players
                tournament = ""
                for tk in ["competition", "tournament", "league",
                           "category", "sport", "competitionName"]:
                    t = item.get(tk, "")
                    if isinstance(t, dict):
                        t = t.get("name", "")
                    if t:
                        tournament = str(t)
                        break

                return {
                    "home": p1["name"],
                    "away": p2["name"],
                    "home_odds": p1["odds"],
                    "away_odds": p2["odds"],
                    "home_prob": round(1.0 / p1["odds"], 4),
                    "away_prob": round(1.0 / p2["odds"], 4),
                    "tournament": tournament,
                    "is_tennis": is_tennis,
                    "source": "loro",
                }
        except (ValueError, TypeError, KeyError):
            continue

    return None


def parse_loro_payload(data):
    """Parse LORO JSON payload to extract tennis match odds.

    LORO structure:
    [
      { "sportCode": "TENN", "eventPaths": [
          { "leagueName": "ATP Miami", "events": [
              { "description": "Lehecka, Jiri vs Fritz, Taylor",
                "markets": [{ "outcomes": [
                    {"opponent": "Lehecka, Jiri", "price": "2.45"},
                    {"opponent": "Fritz, Taylor", "price": "1.40"}
                ]}]
              }
          ]}
      ]}
    ]
    """
    matches = []
    seen = set()

    # Handle list of sports (best-bets, initial-request)
    sports_list = data if isinstance(data, list) else [data]

    for sport in sports_list:
        if not isinstance(sport, dict):
            continue

        # Filter for tennis
        sport_code = sport.get("sportCode", "")
        if sport_code and sport_code != "TENN":
            continue

        event_paths = sport.get("eventPaths", [])
        if not isinstance(event_paths, list):
            continue

        for path in event_paths:
            if not isinstance(path, dict):
                continue

            league = path.get("leagueName", "")
            events = path.get("events", [])

            for event in events:
                if not isinstance(event, dict):
                    continue

                desc = event.get("description", "")
                markets = event.get("markets", [])

                for market in markets:
                    if not isinstance(market, dict):
                        continue

                    # Only "Face à Face" / head-to-head / TWO_OUTCOME
                    style = market.get("style", "")
                    outcomes = market.get("outcomes", [])

                    if len(outcomes) != 2:
                        continue

                    try:
                        p1 = outcomes[0]
                        p2 = outcomes[1]
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
        description="Intercept Loterie Romande tennis odds")
    parser.add_argument("--discover", action="store_true",
                        help="Discovery mode: log all network responses")
    parser.add_argument("--headed", action="store_true",
                        help="Run with visible browser")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"Page load timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--login", action="store_true",
                        help="Open browser to solve captcha and save cookies")
    args = parser.parse_args()

    if args.login:
        login_and_save_cookies(timeout=args.timeout or 60)
        return

    if args.discover:
        discover_endpoints(headed=args.headed or True, timeout=args.timeout)
        return

    print("Intercepting LORO tennis odds...")
    matches = intercept_loro_odds(headless=not args.headed, timeout=args.timeout)

    if args.json:
        print(json.dumps(matches, indent=2))
    else:
        print_matches(matches)

    print(f"\n  Total: {len(matches)} LORO tennis matches")


if __name__ == "__main__":
    main()
