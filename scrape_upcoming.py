"""
Scrape upcoming/scheduled ATP matches from Flashscore.

Visits the ATP Singles fixtures overview page (single page load) and extracts
all scheduled matches grouped by tournament.

Usage:
    python3 scrape_upcoming.py                           # All upcoming matches
    python3 scrape_upcoming.py --tournament indian-wells  # Specific tournament only
    python3 scrape_upcoming.py --output my_matches.csv    # Custom output
"""

import argparse
import csv
import re
import time
from datetime import datetime
from pathlib import Path
from playwright.sync_api import sync_playwright

DATA_DIR = Path('data')

# Import shared constants from the results scraper
from scrape_flashscore import SURFACE_MAP, ROUND_MAP, ALL_ATP_SLUGS, EXCLUDE_SLUGS

# Tournament level mapping
TOURNEY_LEVEL_MAP = {
    'australian-open': 'G',
    'french-open': 'G',
    'wimbledon': 'G',
    'us-open': 'G',
    'indian-wells': 'M',
    'miami': 'M',
    'monte-carlo': 'M',
    'madrid': 'M',
    'rome': 'M',
    'montreal': 'M',
    'toronto': 'M',
    'cincinnati': 'M',
    'shanghai': 'M',
    'paris': 'M',
    'finals-turin': 'F',
    'atp-finals': 'F',
}

GRAND_SLAM_SLUGS = {'australian-open', 'french-open', 'wimbledon', 'us-open'}


def scrape_all_fixtures(page):
    """Scrape ALL upcoming ATP matches from the fixtures overview page.

    Returns a list of match dicts grouped by tournament, scraped from a
    single page load (much faster than visiting each tournament separately).
    """
    url = "https://www.flashscore.com/tennis/atp-singles/fixtures/"
    page.goto(url, timeout=30000)
    page.wait_for_timeout(3000)

    # Scroll down to load lazy content
    for _ in range(5):
        page.evaluate("window.scrollBy(0, 1000)")
        page.wait_for_timeout(500)

    # Extract all matches from the fixtures page
    # Flashscore groups matches by tournament with headerLeague sections
    data = page.evaluate("""() => {
        const results = [];
        let currentTourney = '';
        let currentSlug = '';
        let currentRound = '';

        // The fixtures page uses headerLeague for tournament sections
        // and event__match for individual matches
        const container = document.querySelector('.sportName') ||
                          document.querySelector('[id="live-table"]') ||
                          document.body;

        const allEls = container.querySelectorAll(
            '[class*="headerLeague"], [class*="event__round"], [class*="event__match"]'
        );

        for (const el of allEls) {
            const cls = (typeof el.className === 'string') ? el.className : '';

            // Tournament header (headerLeague__wrapper or headerLeague--has-star)
            if (cls.indexOf('headerLeague') >= 0 && cls.indexOf('headerLeague__') < 0) {
                // This is the main header container
                const titleLink = el.querySelector('a.headerLeague__title') ||
                                  el.querySelector('a[href*="/tennis/atp-singles/"]');
                if (titleLink) {
                    // Extract tournament name: "Indian Wells (USA), hard"
                    const rawName = titleLink.textContent.trim();
                    // Remove surface and country info to get clean name
                    currentTourney = rawName.replace(/\\s*\\(.*?\\).*$/, '').trim();

                    // Extract slug from href
                    const href = titleLink.getAttribute('href') || '';
                    const match = href.match(/\\/tennis\\/atp-singles\\/([a-z0-9-]+)\\//);
                    if (match) {
                        currentSlug = match[1];
                    }
                }
                currentRound = '';
                continue;
            }

            // Skip headerLeague child elements (title, body, meta, etc.)
            if (cls.indexOf('headerLeague__') >= 0) continue;

            // Round header
            if (cls.indexOf('event__round') >= 0) {
                currentRound = el.textContent.trim().toLowerCase();
                continue;
            }

            // Match row
            if (cls.indexOf('event__match') < 0) continue;

            const homeEl = el.querySelector('[class*="participant--home"]');
            const awayEl = el.querySelector('[class*="participant--away"]');
            const timeEl = el.querySelector('[class*="event__time"]');

            // Check if match has scores (already played/in progress)
            const scoreHome = el.querySelector('[class*="event__score--home"]');
            const scoreAway = el.querySelector('[class*="event__score--away"]');
            const hasScore = (scoreHome && scoreHome.textContent.trim() !== '') ||
                             (scoreAway && scoreAway.textContent.trim() !== '');

            const home = homeEl ? homeEl.textContent.trim() : '';
            const away = awayEl ? awayEl.textContent.trim() : '';
            const timeText = timeEl ? timeEl.textContent.trim() : '';

            if (home && away) {
                results.push({
                    home: home,
                    away: away,
                    round: currentRound,
                    timeText: timeText,
                    hasScore: hasScore,
                    tourneyName: currentTourney,
                    tourneySlug: currentSlug,
                });
            }
        }
        return results;
    }""")

    return data


def scrape_tournament_fixtures(page, slug):
    """Scrape upcoming matches for a specific tournament (single tournament mode)."""
    url = f"https://www.flashscore.com/tennis/atp-singles/{slug}/fixtures/"
    page.goto(url, timeout=30000)
    page.wait_for_timeout(3000)

    tourney_name = page.evaluate("""() => {
        const el = document.querySelector('[class*="heading__name"]') ||
                   document.querySelector('[class*="tournamentHeader__country"]') ||
                   document.querySelector('h2');
        return el ? el.textContent.trim().replace(/\\s*-\\s*Singles.*$/i, '').trim() : '';
    }""")

    if not tourney_name:
        tourney_name = slug.replace('-', ' ').title()

    raw_matches = page.evaluate("""() => {
        const results = [];
        let currentRound = '';

        const container = document.querySelector('.sportName') ||
                          document.querySelector('[class*="leagues--static"]') ||
                          document.querySelector('[id="live-table"]') ||
                          document.body;

        const allEls = container.querySelectorAll('[class*="event__round"], [class*="event__match"]');

        for (const el of allEls) {
            const cls = el.className || '';

            if (cls.includes('event__round')) {
                currentRound = el.textContent.trim().toLowerCase();
                continue;
            }

            if (!cls.includes('event__match')) continue;

            const homeEl = el.querySelector('[class*="participant--home"]');
            const awayEl = el.querySelector('[class*="participant--away"]');
            const timeEl = el.querySelector('[class*="event__time"]');
            const scoreHome = el.querySelector('[class*="event__score--home"]');
            const scoreAway = el.querySelector('[class*="event__score--away"]');
            const hasScore = (scoreHome && scoreHome.textContent.trim() !== '') ||
                             (scoreAway && scoreAway.textContent.trim() !== '');

            const home = homeEl ? homeEl.textContent.trim() : '';
            const away = awayEl ? awayEl.textContent.trim() : '';
            const timeText = timeEl ? timeEl.textContent.trim() : '';

            if (home && away) {
                results.push({
                    home: home,
                    away: away,
                    round: currentRound,
                    timeText: timeText,
                    hasScore: hasScore,
                    tourneyName: '',
                    tourneySlug: '',
                });
            }
        }
        return results;
    }""")

    # Tag with tournament info
    for m in raw_matches:
        m['tourneyName'] = tourney_name
        m['tourneySlug'] = slug

    return raw_matches


def process_fixtures(raw_matches):
    """Convert raw fixture data into output rows, filtering out played matches."""
    today = datetime.now().strftime('%Y%m%d')
    results = []

    for m in raw_matches:
        # Skip matches that already have scores
        if m.get('hasScore'):
            continue

        p1 = m['home']
        p2 = m['away']

        # Skip TBD / qualifier placeholders
        skip_names = {'tbd', 'qualifier', 'bye', 'to be determined', 'winner match'}
        if p1.lower() in skip_names or p2.lower() in skip_names:
            continue
        if 'qualifier' in p1.lower() or 'qualifier' in p2.lower():
            continue
        if 'winner' in p1.lower() or 'winner' in p2.lower():
            continue

        slug = m.get('tourneySlug', '')
        tourney_name = m.get('tourneyName', slug.replace('-', ' ').title())
        surface = SURFACE_MAP.get(slug, '')
        tourney_level = TOURNEY_LEVEL_MAP.get(slug, 'A')
        best_of = 5 if slug in GRAND_SLAM_SLUGS else 3

        # Map round
        round_text = m.get('round', '').lower().strip()
        match_round = ROUND_MAP.get(round_text, round_text.upper()[:3] if round_text else '')

        row = {
            'player1_name': p1,
            'player2_name': p2,
            'tourney_name': tourney_name,
            'tourney_slug': slug,
            'surface': surface,
            'round': match_round,
            'best_of': best_of,
            'tourney_level': tourney_level,
            'match_date': today,
            'scheduled_time': m.get('timeText', ''),
        }
        results.append(row)

    return results


def main():
    parser = argparse.ArgumentParser(description='Scrape upcoming ATP matches from Flashscore')
    parser.add_argument('--tournament', type=str, default=None,
                        help='Specific tournament slug (e.g., "indian-wells")')
    parser.add_argument('--output', type=str, default=str(DATA_DIR / 'upcoming_matches.csv'),
                        help='Output CSV filename')
    args = parser.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        if args.tournament:
            # Scrape specific tournament
            print(f"Scraping fixtures for: {args.tournament}")
            raw_matches = scrape_tournament_fixtures(page, args.tournament)
        else:
            # Scrape all upcoming from the overview page (single page load!)
            print("Scraping ATP fixtures overview page...")
            raw_matches = scrape_all_fixtures(page)

        browser.close()

    print(f"  Raw matches found: {len(raw_matches)}")

    # Process and filter
    matches = process_fixtures(raw_matches)

    # Group by tournament for display
    tourneys = {}
    for m in matches:
        key = m['tourney_name']
        if key not in tourneys:
            tourneys[key] = []
        tourneys[key].append(m)

    for tourney, tmatches in tourneys.items():
        slug = tmatches[0]['tourney_slug']
        surface = tmatches[0]['surface']
        print(f"\n  {tourney} ({slug}) - {surface}")
        for m in tmatches:
            print(f"    {m['round']:>4s}  {m['player1_name']} vs {m['player2_name']}  [{m['scheduled_time']}]")

    # Write output
    if matches:
        fieldnames = [
            'player1_name', 'player2_name', 'tourney_name', 'tourney_slug',
            'surface', 'round', 'best_of', 'tourney_level', 'match_date',
            'scheduled_time',
        ]

        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(matches)

        print(f"\n{'='*60}")
        print(f"DONE: {len(matches)} upcoming matches saved to {args.output}")
        print(f"{'='*60}")
    else:
        print("\nNo upcoming matches found!")


if __name__ == '__main__':
    main()
