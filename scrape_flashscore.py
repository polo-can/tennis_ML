"""
Scrape recent ATP match results from Flashscore.

Outputs a CSV compatible with the Sackmann atp_matches format.
Scrapes by tournament for complete coverage.

Usage:
    python3 scrape_flashscore.py                           # Scrape all 2025+ tournaments
    python3 scrape_flashscore.py --tournament indian-wells  # Specific tournament
    python3 scrape_flashscore.py --no-stats                # Skip per-match stats (faster)
"""

import argparse
import csv
import re
import time
from datetime import datetime
from pathlib import Path
from playwright.sync_api import sync_playwright

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)

ROUND_MAP = {
    'final': 'F',
    'semi-finals': 'SF',
    'quarter-finals': 'QF',
    '4th round': 'R16',
    '3rd round': 'R32',
    '2nd round': 'R64',
    '1st round': 'R128',
    '1/2-finals': 'SF',
    '1/4-finals': 'QF',
    '1/16-finals': 'R32',
    '1/32-finals': 'R64',
    '1/8-finals': 'R16',
    'round of 16': 'R16',
    'round of 32': 'R32',
    'round of 64': 'R64',
    'round of 128': 'R128',
    'round robin': 'RR',
}

# Tournament slug -> surface mapping (more reliable than scraping headers)
SURFACE_MAP = {
    'australian-open': 'Hard',
    'indian-wells': 'Hard',
    'miami': 'Hard',
    'us-open': 'Hard',
    'brisbane': 'Hard',
    'adelaide': 'Hard',
    'auckland': 'Hard',
    'hong-kong': 'Hard',
    'montpellier': 'Hard',
    'dallas': 'Hard',
    'marseille': 'Hard',
    'doha': 'Hard',
    'dubai': 'Hard',
    'rotterdam': 'Hard',
    'delray-beach': 'Hard',
    'acapulco': 'Hard',
    'shanghai': 'Hard',
    'beijing': 'Hard',
    'tokyo': 'Hard',
    'basel': 'Hard',
    'vienna': 'Hard',
    'paris': 'Hard',
    'atp-finals': 'Hard',
    'sofia': 'Hard',
    'metz': 'Hard',
    'astana': 'Hard',
    'stockholm': 'Hard',
    'antwerp': 'Hard',
    'cincinnati': 'Hard',
    'montreal': 'Hard',
    'toronto': 'Hard',
    'washington': 'Hard',
    'atlanta': 'Hard',
    'los-cabos': 'Hard',
    'winston-salem': 'Hard',
    'san-diego': 'Hard',
    'french-open': 'Clay',
    'rome': 'Clay',
    'madrid': 'Clay',
    'monte-carlo': 'Clay',
    'barcelona': 'Clay',
    'buenos-aires': 'Clay',
    'rio-de-janeiro': 'Clay',
    'santiago': 'Clay',
    'estoril': 'Clay',
    'lyon': 'Clay',
    'geneva': 'Clay',
    'hamburg': 'Clay',
    'bastad': 'Clay',
    'gstaad': 'Clay',
    'kitzbuhel': 'Clay',
    'umag': 'Clay',
    'cordoba': 'Clay',
    'marrakech': 'Clay',
    'bucharest': 'Clay',
    'wimbledon': 'Grass',
    'halle': 'Grass',
    'queens': 'Grass',
    'stuttgart': 'Grass',
    'eastbourne': 'Grass',
    'mallorca': 'Grass',
    's-hertogenbosch': 'Grass',
    'newport': 'Grass',
    'nottingham': 'Grass',
    'london': 'Grass',
    # Additional 250s/500s
    'pune': 'Hard',
    'dallas': 'Hard',
    'san-jose': 'Hard',
    'atlanta': 'Hard',
    'sydney': 'Hard',
    'bangkok': 'Hard',
    'antalya': 'Hard',
    'copenhagen': 'Hard',
    'istanbul': 'Hard',
    'sao-paulo': 'Hard',
    'tel-aviv': 'Hard',
    'tashkent': 'Hard',
    'milan': 'Hard',
    'finals-turin': 'Hard',
    'munich': 'Clay',
    'geneva': 'Clay',
    'budapest': 'Clay',
    'cordoba': 'Clay',
    'belgrade': 'Clay',
    'bogota': 'Clay',
    'casablanca': 'Clay',
    'nice': 'Clay',
}


def parse_serve_stat(text):
    """Parse '82% (27/33)' or '2/8' into (numerator, denominator)."""
    m = re.search(r'(\d+)/(\d+)', text)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def get_match_detail(page, match_url):
    """Get detailed stats from a match page."""
    stats = {}
    try:
        page.goto(f"{match_url}#/match-summary/match-statistics/0", timeout=15000)
        page.wait_for_timeout(2000)

        # Rankings
        for i, r in enumerate(page.evaluate("""() =>
            Array.from(document.querySelectorAll('[class*="participantRank"]'))
            .map(e => e.textContent.trim()).filter(t => /\\d/.test(t))
        """)):
            rank_num = re.search(r'(\d+)', r)
            if rank_num:
                stats['home_rank' if i == 0 else 'away_rank'] = int(rank_num.group(1))

        # Serve statistics
        stat_rows = page.evaluate("""() =>
            Array.from(document.querySelectorAll('[data-testid="wcl-statistics"]'))
            .map(el => Array.from(el.querySelectorAll('*'))
                .filter(c => c.children.length === 0)
                .map(c => c.textContent.trim())
                .filter(Boolean))
        """)

        for row in stat_rows:
            cat = ' '.join(row).lower()

            if 'aces' in cat:
                nums = [int(x) for x in row if x.isdigit()]
                if len(nums) >= 2:
                    stats['home_ace'], stats['away_ace'] = nums[0], nums[-1]

            elif 'double faults' in cat:
                nums = [int(x) for x in row if x.isdigit()]
                if len(nums) >= 2:
                    stats['home_df'], stats['away_df'] = nums[0], nums[-1]

            elif '1st serve' in cat and 'won' in cat:
                pairs = [(parse_serve_stat(item)) for item in row]
                valid = [(w, t) for w, t in pairs if w is not None]
                if len(valid) >= 2:
                    stats['home_1stWon'], stats['home_1stIn'] = valid[0]
                    stats['away_1stWon'], stats['away_1stIn'] = valid[1]

            elif '2nd serve' in cat and 'won' in cat:
                pairs = [(parse_serve_stat(item)) for item in row]
                valid = [(w, t) for w, t in pairs if w is not None]
                if len(valid) >= 2:
                    stats['home_2ndWon'] = valid[0][0]
                    stats['home_svpt'] = stats.get('home_1stIn', 0) + valid[0][1]
                    stats['away_2ndWon'] = valid[1][0]
                    stats['away_svpt'] = stats.get('away_1stIn', 0) + valid[1][1]

            elif 'break point' in cat:
                pairs = [(parse_serve_stat(item)) for item in row]
                valid = [(w, t) for w, t in pairs if w is not None]
                if len(valid) >= 2:
                    # BP won by home = BP faced by away's serve
                    stats['away_bpFaced'] = valid[0][1]
                    stats['away_bpSaved'] = valid[0][1] - valid[0][0]
                    stats['home_bpFaced'] = valid[1][1]
                    stats['home_bpSaved'] = valid[1][1] - valid[1][0]

    except Exception as e:
        pass  # Stats are optional, continue without them

    return stats


def scrape_tournament_results(page, slug):
    """Scrape all results for a tournament. Returns list of match dicts."""
    url = f"https://www.flashscore.com/tennis/atp-singles/{slug}/results/"
    page.goto(url, timeout=30000)
    page.wait_for_timeout(3000)

    # Expand all results
    for _ in range(20):
        btn = page.query_selector('.event__more, a.event__more')
        if btn:
            btn.click()
            page.wait_for_timeout(1500)
        else:
            break

    # Surface from lookup table (most reliable)
    surface = SURFACE_MAP.get(slug, '')

    # If not in lookup, try to detect from page header
    if not surface:
        header = page.evaluate("""() => {
            const els = document.querySelectorAll('[class*="heading"] span, [class*="breadcrumb"] a, [class*="tournamentHeader"]');
            return Array.from(els).map(e => e.textContent.trim()).join(' ');
        }""") or ''
        for surf_key in ['hard', 'clay', 'grass']:
            if surf_key in header.lower():
                surface = surf_key.capitalize()

    tourney_name = slug.replace('-', ' ').title()

    # Extract matches with full structured DOM parsing including set scores and rounds
    raw = page.evaluate("""() => {
        const results = [];
        let currentRound = '';

        // Iterate through all event children (round headers + matches)
        const container = document.querySelector('[class*="sportName"]') ||
                          document.querySelector('[class*="event--results"]') ||
                          document.querySelector('.leagues--static');
        if (!container) return [];

        const allChildren = container.children;
        for (let i = 0; i < allChildren.length; i++) {
            const el = allChildren[i];
            const cls = el.className || '';

            // Round header detection
            if (cls.includes('event__round')) {
                currentRound = el.textContent.trim().toLowerCase();
                continue;
            }

            // Match row
            if (!cls.includes('event__match')) continue;

            const homeEl = el.querySelector('[class*="participant--home"]');
            const awayEl = el.querySelector('[class*="participant--away"]');
            const homeScoreEl = el.querySelector('[class*="score--home"]');
            const awayScoreEl = el.querySelector('[class*="score--away"]');
            const link = el.querySelector('a[href*="match"]');

            const home = homeEl ? homeEl.textContent.trim() : '';
            const away = awayEl ? awayEl.textContent.trim() : '';
            const homeSets = homeScoreEl ? parseInt(homeScoreEl.textContent.trim()) || 0 : 0;
            const awaySets = awayScoreEl ? parseInt(awayScoreEl.textContent.trim()) || 0 : 0;

            // Extract individual set scores from part* elements
            // These have classes like event__part--1, event__part--2, etc.
            const homeParts = [];
            const awayParts = [];
            for (let s = 1; s <= 5; s++) {
                const homePartEl = el.querySelector(`.event__part--home.event__part--${s}`);
                const awayPartEl = el.querySelector(`.event__part--away.event__part--${s}`);
                if (homePartEl && awayPartEl) {
                    const hVal = homePartEl.textContent.trim();
                    const aVal = awayPartEl.textContent.trim();
                    if (hVal !== '' && aVal !== '') {
                        homeParts.push(hVal);
                        awayParts.push(aVal);
                    }
                }
            }

            // If part-specific selectors didn't work, try generic approach
            if (homeParts.length === 0) {
                const allParts = el.querySelectorAll('[class*="event__part"]');
                const homeP = [];
                const awayP = [];
                allParts.forEach(p => {
                    const pcls = p.className || '';
                    if (pcls.includes('home')) homeP.push(p.textContent.trim());
                    else if (pcls.includes('away')) awayP.push(p.textContent.trim());
                });
                if (homeP.length > 0 && homeP.length === awayP.length) {
                    homeP.forEach((v, idx) => {
                        homeParts.push(v);
                        awayParts.push(awayP[idx]);
                    });
                }
            }

            // Extract date from the match element's time info
            const timeEl = el.querySelector('[class*="event__time"]');
            let dateText = timeEl ? timeEl.textContent.trim() : '';

            results.push({
                home: home,
                away: away,
                homeSets: homeSets,
                awaySets: awaySets,
                homeParts: homeParts,
                awayParts: awayParts,
                round: currentRound,
                dateText: dateText,
                link: link ? link.getAttribute('href') : '',
            });
        }
        return results;
    }""")

    return raw, tourney_name, surface


def parse_date(date_text, year=None):
    """Parse date text like 'DD.MM. HH:MM' or 'DD.MM.YYYY' into YYYYMMDD.

    Flashscore only shows day.month (no year).  When the resulting date would
    fall in the future, it almost certainly belongs to the *previous* season,
    so we subtract one year.
    """
    if not year:
        year = datetime.now().year
    dm = re.search(r'(\d{2})\.(\d{2})\.', date_text)
    if dm:
        day, month = dm.groups()
        candidate = f"{year}{month}{day}"
        # If this date is in the future, it's from last season
        try:
            from datetime import date as _date
            parsed = _date(int(year), int(month), int(day))
            if parsed > _date.today():
                candidate = f"{year - 1}{month}{day}"
        except ValueError:
            pass
        return candidate
    return ''


def parse_set_score(raw_score):
    """Parse a raw set score like '77' or '6' into (games, tiebreak_or_none).

    Flashscore DOM concatenates game score + tiebreak score in the same element.
    '77' = 7 games, tiebreak score 7  (the set was 7-6 with TB 7)
    '65' = 6 games, tiebreak score 5  (the set was 6-7 with TB 5)
    '6'  = 6 games, no tiebreak
    '710' = 7 games, tiebreak score 10 (super tiebreak 10)
    '69' = 6 games, tiebreak score 9
    """
    raw = str(raw_score).strip()
    if not raw or not raw.isdigit():
        return raw, None

    # Single digit: just a game score (0-7)
    if len(raw) == 1:
        return raw, None

    # Two+ digits: first digit is games, rest is tiebreak score
    # E.g., "77" -> games=7, tb=7; "65" -> games=6, tb=5; "710" -> games=7, tb=10
    games = raw[0]
    tb = raw[1:]
    return games, tb


def build_score(home_parts, away_parts, winner_is_away):
    """Build score string from winner's perspective, e.g. '6-3 7-6(5)'.

    Handles tiebreak scores embedded in the DOM values.
    """
    if not home_parts or len(home_parts) != len(away_parts):
        return ''
    sets = []
    for h_raw, a_raw in zip(home_parts, away_parts):
        h_games, h_tb = parse_set_score(h_raw)
        a_games, a_tb = parse_set_score(a_raw)

        # Determine which tiebreak score to show (loser's TB score, per convention)
        tb_str = ''
        if h_tb is not None or a_tb is not None:
            # The loser's tiebreak points are shown in parentheses
            # If home won the set (7-6), show away's TB points
            # If away won the set (6-7), show home's TB points
            if int(h_games) > int(a_games):
                # Home won this set — show away's (loser's) TB score
                tb_str = f"({a_tb})" if a_tb is not None else f"({h_tb})" if h_tb is not None else ''
            else:
                # Away won this set — show home's (loser's) TB score
                tb_str = f"({h_tb})" if h_tb is not None else f"({a_tb})" if a_tb is not None else ''

        if winner_is_away:
            sets.append(f"{a_games}-{h_games}{tb_str}")
        else:
            sets.append(f"{h_games}-{a_games}{tb_str}")
    return ' '.join(sets)


def find_recent_tournaments(page):
    """Find all ATP tournament slugs that have results on Flashscore."""
    page.goto("https://www.flashscore.com/tennis/atp-singles/fixtures/", timeout=30000)
    page.wait_for_timeout(3000)

    links = page.evaluate("""() => {
        const els = document.querySelectorAll('a[href*="/tennis/atp-singles/"]');
        const seen = new Set();
        const results = [];
        for (const el of els) {
            const href = el.getAttribute('href');
            const text = el.textContent.trim();
            const match = href.match(/\\/tennis\\/atp-singles\\/([a-z0-9-]+)\\//);
            if (match && !seen.has(match[1]) && match[1] !== 'results' && match[1] !== 'fixtures' && match[1] !== 'archive') {
                seen.add(match[1]);
                results.push({slug: match[1], name: text});
            }
        }
        return results;
    }""")

    return links


# Exclude non-standard events from auto-discovery
EXCLUDE_SLUGS = {
    'davis-cup-world-group', 'laver-cup', 'atp-cup', 'next-gen-finals-jeddah',
    'olympic-games',
}

# Full ATP tour slugs — all tournaments that could have results
ALL_ATP_SLUGS = [
    # Grand Slams
    'australian-open', 'french-open', 'wimbledon', 'us-open',
    # Masters 1000
    'indian-wells', 'miami', 'monte-carlo', 'madrid', 'rome',
    'montreal', 'toronto', 'cincinnati', 'shanghai', 'paris',
    # ATP Finals
    'finals-turin',
    # ATP 500
    'rotterdam', 'dubai', 'acapulco', 'barcelona', 'hamburg',
    'halle', 'london', 'washington', 'tokyo', 'beijing',
    'vienna', 'basel',
    # ATP 250
    'brisbane', 'adelaide', 'auckland', 'hong-kong', 'pune',
    'montpellier', 'dallas', 'buenos-aires', 'rio-de-janeiro',
    'marseille', 'doha', 'delray-beach', 'santiago', 'estoril',
    'munich', 'geneva', 'lyon', 'eastbourne', 'mallorca',
    's-hertogenbosch', 'stuttgart', 'newport', 'bastad',
    'gstaad', 'kitzbuhel', 'umag', 'los-cabos', 'atlanta',
    'winston-salem', 'metz', 'chengdu', 'zhuhai', 'sofia',
    'antwerp', 'stockholm', 'san-jose', 'cordoba',
    'marrakech', 'budapest', 'antalya', 'bangkok',
    'belgrade', 'bogota', 'casablanca', 'copenhagen',
    'istanbul', 'nice', 'nottingham', 'sao-paulo', 'sydney',
    'tel-aviv', 'tashkent', 'milan',
]


def load_existing_matches(csv_path):
    """Load existing scraped matches to enable incremental updates.

    Returns set of (tourney_name, winner_name, loser_name, date) tuples.
    """
    existing = set()
    if not Path(csv_path).exists():
        return existing
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    row.get('tourney_name', ''),
                    row.get('winner_name', ''),
                    row.get('loser_name', ''),
                    row.get('tourney_date', ''),
                )
                existing.add(key)
    except Exception:
        pass
    return existing


def load_existing_rows(csv_path):
    """Load all existing rows from CSV for merging."""
    rows = []
    if not Path(csv_path).exists():
        return rows
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        pass
    return rows


def process_raw_matches(raw_matches, slug, surface, tourney_name, year,
                        page=None, no_stats=True, existing_keys=None):
    """Convert raw scraped match data into output rows.

    Returns list of dicts, one per match.
    """
    results = []
    for i, m in enumerate(raw_matches):
        if not m['home'] or not m['away']:
            continue

        # Determine winner from set scores
        home_sets = m.get('homeSets', 0)
        away_sets = m.get('awaySets', 0)
        winner_is_away = away_sets > home_sets

        winner = m['away'] if winner_is_away else m['home']
        loser = m['home'] if winner_is_away else m['away']

        # Parse date
        date_str = parse_date(m.get('dateText', ''), year)

        # Skip if already in existing data (incremental mode)
        if existing_keys is not None:
            match_key = (tourney_name, winner, loser, date_str)
            if match_key in existing_keys:
                continue

        # Build score from set parts
        score = build_score(
            m.get('homeParts', []),
            m.get('awayParts', []),
            winner_is_away
        )

        # Map round
        round_text = m.get('round', '').lower().strip()
        match_round = ROUND_MAP.get(round_text, round_text.upper()[:3] if round_text else '')

        # Get detailed stats if requested
        stats = {}
        if not no_stats and page and m.get('link'):
            link = m['link']
            if not link.startswith('http'):
                link = f"https://www.flashscore.com{link}"
            stats = get_match_detail(page, link)
            time.sleep(0.3)

        # Map stats to winner/loser perspective
        w_prefix = 'away' if winner_is_away else 'home'
        l_prefix = 'home' if winner_is_away else 'away'

        row = {
            'tourney_name': tourney_name,
            'tourney_date': date_str,
            'surface': surface,
            'round': match_round,
            'best_of': 5 if slug in ('australian-open', 'french-open', 'us-open', 'wimbledon') else 3,
            'winner_name': winner,
            'loser_name': loser,
            'score': score,
            'winner_rank': stats.get(f'{w_prefix}_rank', ''),
            'loser_rank': stats.get(f'{l_prefix}_rank', ''),
            'w_ace': stats.get(f'{w_prefix}_ace', ''),
            'w_df': stats.get(f'{w_prefix}_df', ''),
            'w_svpt': stats.get(f'{w_prefix}_svpt', ''),
            'w_1stIn': stats.get(f'{w_prefix}_1stIn', ''),
            'w_1stWon': stats.get(f'{w_prefix}_1stWon', ''),
            'w_2ndWon': stats.get(f'{w_prefix}_2ndWon', ''),
            'w_bpSaved': stats.get(f'{w_prefix}_bpSaved', ''),
            'w_bpFaced': stats.get(f'{w_prefix}_bpFaced', ''),
            'l_ace': stats.get(f'{l_prefix}_ace', ''),
            'l_df': stats.get(f'{l_prefix}_df', ''),
            'l_svpt': stats.get(f'{l_prefix}_svpt', ''),
            'l_1stIn': stats.get(f'{l_prefix}_1stIn', ''),
            'l_1stWon': stats.get(f'{l_prefix}_1stWon', ''),
            'l_2ndWon': stats.get(f'{l_prefix}_2ndWon', ''),
            'l_bpSaved': stats.get(f'{l_prefix}_bpSaved', ''),
            'l_bpFaced': stats.get(f'{l_prefix}_bpFaced', ''),
        }

        results.append(row)
        print(f"  [{i+1}/{len(raw_matches)}] {winner} d. {loser} {score}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Scrape ATP results from Flashscore')
    parser.add_argument('--tournament', type=str, default=None,
                        help='Specific tournament slug (e.g., "indian-wells", "australian-open")')
    parser.add_argument('--list', action='store_true',
                        help='List all available tournaments and exit')
    parser.add_argument('--no-stats', action='store_true',
                        help='Skip per-match detailed stats (much faster)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV filename')
    parser.add_argument('--year', type=int, default=datetime.now().year,
                        help='Filter tournaments by year in date')
    parser.add_argument('--all', action='store_true',
                        help='Scrape ALL known ATP tournaments (auto-discover)')
    parser.add_argument('--backfill-stats', action='store_true',
                        help='Re-visit tournament pages and fill in missing stats for existing matches')
    args = parser.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        def _save_backfill(rows, filepath):
            """Save current backfill progress to CSV."""
            fnames = [
                'tourney_name', 'tourney_date', 'surface', 'round', 'best_of',
                'winner_name', 'loser_name', 'score',
                'winner_rank', 'loser_rank',
                'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
                'w_bpSaved', 'w_bpFaced',
                'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon',
                'l_bpSaved', 'l_bpFaced',
            ]
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(rows)

        # Backfill stats mode
        if args.backfill_stats:
            input_file = args.output or str(DATA_DIR / f"atp_matches_scraped_{args.year}.csv")
            print(f"Backfill stats mode: reading {input_file}")
            existing_rows = load_existing_rows(input_file)
            if not existing_rows:
                print("No existing data found!")
                browser.close()
                return

            # Group rows by tournament name for batch processing
            from collections import OrderedDict
            tourney_groups = OrderedDict()
            for i, row in enumerate(existing_rows):
                tname = row['tourney_name']
                if tname not in tourney_groups:
                    tourney_groups[tname] = []
                tourney_groups[tname].append((i, row))

            # Check which rows need stats
            stats_cols = ['w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'winner_rank']
            need_stats = sum(1 for r in existing_rows
                             if not any(r.get(c) for c in stats_cols))
            print(f"  Total matches: {len(existing_rows)}")
            print(f"  Missing stats: {need_stats}")

            if need_stats == 0:
                print("All matches already have stats!")
                browser.close()
                return

            # Map tournament names (from CSV) to slugs
            name_to_slug = {}
            for s in ALL_ATP_SLUGS:
                pretty = s.replace('-', ' ').title()
                name_to_slug[pretty] = s
            # Add special mappings for names that don't match slug.title()
            name_to_slug.update({
                'Us Open': 'us-open',
                'French Open': 'french-open',
                'Australian Open': 'australian-open',
                'Wimbledon': 'wimbledon',
                'Monte Carlo': 'monte-carlo',
                'Indian Wells': 'indian-wells',
                'Rio De Janeiro': 'rio-de-janeiro',
                'Los Cabos': 'los-cabos',
                'Winston Salem': 'winston-salem',
                'Delray Beach': 'delray-beach',
                'Hong Kong': 'hong-kong',
                'Finals Turin': 'atp-finals',
                'Sao Paulo': 'sao-paulo',
                'San Jose': 'san-jose',
                'Tel Aviv': 'tel-aviv',
            })

            filled = 0
            skipped_tournaments = 0

            for tname, row_group in tourney_groups.items():
                # Check if any rows in this group need stats
                group_needs = [i for i, r in row_group
                               if not any(r.get(c) for c in stats_cols)]
                if not group_needs:
                    continue

                slug = name_to_slug.get(tname)
                if not slug:
                    # Try lowercase match
                    for k, v in name_to_slug.items():
                        if k.lower() == tname.lower():
                            slug = v
                            break
                if not slug:
                    skipped_tournaments += 1
                    print(f"\n  Skipping {tname}: no slug mapping")
                    continue

                print(f"\n{'='*60}")
                print(f"Backfilling: {tname} ({slug})")
                print(f"  {len(group_needs)} matches need stats")
                print(f"{'='*60}")

                try:
                    raw_matches, _, surface = scrape_tournament_results(page, slug)
                    if not raw_matches:
                        print("  No matches found on page")
                        continue
                except Exception as e:
                    print(f"  Error: {e}")
                    continue

                # Build lookup from raw matches: (winner, loser) -> match link
                link_map = {}
                for m in raw_matches:
                    if not m['home'] or not m['away']:
                        continue
                    home_sets = m.get('homeSets', 0)
                    away_sets = m.get('awaySets', 0)
                    winner_is_away = away_sets > home_sets
                    winner = m['away'] if winner_is_away else m['home']
                    loser = m['home'] if winner_is_away else m['away']
                    w_prefix = 'away' if winner_is_away else 'home'
                    l_prefix = 'home' if winner_is_away else 'away'
                    link_map[(winner, loser)] = (m.get('link', ''), w_prefix, l_prefix)

                # Now backfill each row that needs stats
                for idx, row in row_group:
                    if any(row.get(c) for c in stats_cols):
                        continue  # Already has stats

                    key = (row['winner_name'], row['loser_name'])
                    if key not in link_map:
                        continue

                    link, w_prefix, l_prefix = link_map[key]
                    if not link:
                        continue

                    if not link.startswith('http'):
                        link = f"https://www.flashscore.com{link}"

                    stats = get_match_detail(page, link)
                    time.sleep(0.3)

                    if stats:
                        existing_rows[idx]['winner_rank'] = stats.get(f'{w_prefix}_rank', '')
                        existing_rows[idx]['loser_rank'] = stats.get(f'{l_prefix}_rank', '')
                        existing_rows[idx]['w_ace'] = stats.get(f'{w_prefix}_ace', '')
                        existing_rows[idx]['w_df'] = stats.get(f'{w_prefix}_df', '')
                        existing_rows[idx]['w_svpt'] = stats.get(f'{w_prefix}_svpt', '')
                        existing_rows[idx]['w_1stIn'] = stats.get(f'{w_prefix}_1stIn', '')
                        existing_rows[idx]['w_1stWon'] = stats.get(f'{w_prefix}_1stWon', '')
                        existing_rows[idx]['w_2ndWon'] = stats.get(f'{w_prefix}_2ndWon', '')
                        existing_rows[idx]['w_bpSaved'] = stats.get(f'{w_prefix}_bpSaved', '')
                        existing_rows[idx]['w_bpFaced'] = stats.get(f'{w_prefix}_bpFaced', '')
                        existing_rows[idx]['l_ace'] = stats.get(f'{l_prefix}_ace', '')
                        existing_rows[idx]['l_df'] = stats.get(f'{l_prefix}_df', '')
                        existing_rows[idx]['l_svpt'] = stats.get(f'{l_prefix}_svpt', '')
                        existing_rows[idx]['l_1stIn'] = stats.get(f'{l_prefix}_1stIn', '')
                        existing_rows[idx]['l_1stWon'] = stats.get(f'{l_prefix}_1stWon', '')
                        existing_rows[idx]['l_2ndWon'] = stats.get(f'{l_prefix}_2ndWon', '')
                        existing_rows[idx]['l_bpSaved'] = stats.get(f'{l_prefix}_bpSaved', '')
                        existing_rows[idx]['l_bpFaced'] = stats.get(f'{l_prefix}_bpFaced', '')
                        filled += 1
                        print(f"  [{filled}] {row['winner_name']} d. {row['loser_name']} - stats filled")

                # Save after each tournament for resumability
                _save_backfill(existing_rows, input_file)
                print(f"  Saved progress ({filled} total stats filled)")

            browser.close()

            print(f"\n{'='*60}")
            print(f"BACKFILL COMPLETE")
            print(f"  Stats filled: {filled}/{need_stats}")
            print(f"  Skipped tournaments: {skipped_tournaments}")
            print(f"  Saved to: {input_file}")
            print(f"{'='*60}")
            return

        # List mode
        if args.list:
            tournaments = find_recent_tournaments(page)
            print(f"Available tournaments ({len(tournaments)}):")
            for t in tournaments:
                print(f"  {t['slug']:30s} {t['name']}")
            browser.close()
            return

        # Determine which tournaments to scrape
        if args.tournament:
            slugs = [args.tournament]
        elif args.all:
            slugs = [s for s in ALL_ATP_SLUGS if s not in EXCLUDE_SLUGS]
        else:
            slugs = ALL_ATP_SLUGS[:40]  # Default: first 40 (majors + 500s + main 250s)

        output_file = args.output or str(DATA_DIR / f"atp_matches_scraped_{args.year}.csv")

        # Load existing data for incremental mode
        existing_keys = load_existing_matches(output_file)
        existing_rows = load_existing_rows(output_file)
        if existing_keys:
            print(f"Incremental mode: {len(existing_keys)} existing matches found in {output_file}")
        else:
            print(f"Full scrape mode (no existing data at {output_file})")

        new_results = []
        empty_tournaments = 0

        for slug in slugs:
            print(f"\n{'='*60}")
            print(f"Scraping: {slug}")
            print(f"{'='*60}")

            try:
                raw_matches, tourney_name, surface = scrape_tournament_results(page, slug)
                if not raw_matches:
                    empty_tournaments += 1
                    print("  No matches found, skipping.")
                    continue
                print(f"  Found {len(raw_matches)} matches | Surface: {surface or '?'}")
            except Exception as e:
                empty_tournaments += 1
                print(f"  Error loading tournament: {e}")
                continue

            matches = process_raw_matches(
                raw_matches, slug, surface, tourney_name, args.year,
                page=page, no_stats=args.no_stats,
                existing_keys=existing_keys if existing_keys else None,
            )

            if existing_keys and not matches:
                print(f"  All matches already scraped, skipping.")
            else:
                new_results.extend(matches)

        browser.close()

    # Merge with existing data and write
    fieldnames = [
        'tourney_name', 'tourney_date', 'surface', 'round', 'best_of',
        'winner_name', 'loser_name', 'score',
        'winner_rank', 'loser_rank',
        'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
        'w_bpSaved', 'w_bpFaced',
        'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon',
        'l_bpSaved', 'l_bpFaced',
    ]

    all_results = existing_rows + new_results

    if all_results:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\n{'='*60}")
        print(f"DONE")
        print(f"  Existing matches: {len(existing_rows)}")
        print(f"  New matches:      {len(new_results)}")
        print(f"  Total matches:    {len(all_results)}")
        print(f"  Tournaments with no data: {empty_tournaments}")
        print(f"  Saved to: {output_file}")
        print(f"{'='*60}")
    else:
        print("\nNo matches found!")


if __name__ == "__main__":
    main()
