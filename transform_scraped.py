"""
Transform Flashscore-scraped ATP match data into Sackmann-compatible format.

Handles:
  - Player name matching (Flashscore "Alcaraz C." → Sackmann player_id 207989)
  - Tournament level inference (Grand Slam, Masters, ATP 250/500)
  - Player demographics from atp_players.csv (age, height, hand)
  - tourney_id and match_num generation

Usage:
    python3 transform_scraped.py                              # Transform default scraped file
    python3 transform_scraped.py --input data/scraped.csv     # Custom input
    python3 transform_scraped.py --year 2025                  # Output as atp_matches_2025.csv
"""

import argparse
import csv
import re
import unicodedata
from datetime import datetime
from pathlib import Path

try:
    from thefuzz import fuzz
    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False

DATA_DIR = Path('data')

# ── Tournament level mapping ─────────────────────────────────────────────────
# G = Grand Slam, M = Masters 1000, A = ATP 250/500, F = Tour Finals
TOURNEY_LEVEL_MAP = {
    'australian open': 'G',
    'french open': 'G',
    'roland garros': 'G',
    'wimbledon': 'G',
    'us open': 'G',
    'indian wells': 'M',
    'miami': 'M',
    'monte carlo': 'M',
    'madrid': 'M',
    'rome': 'M',
    'montreal': 'M',
    'toronto': 'M',
    'cincinnati': 'M',
    'shanghai': 'M',
    'paris': 'M',
    'atp finals': 'F',
    # Everything else defaults to 'A' (ATP 250/500)
}


def normalize_str(s):
    """Remove accents, lowercase, strip punctuation for fuzzy matching."""
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    s = s.lower().strip()
    s = re.sub(r'[^a-z\s]', '', s)
    return s


def normalize_player_name(name):
    """Normalize any player name format to canonical 'firstname lastname'.

    Handles:
        "Gauff, Cori"       -> "cori gauff"
        "Coco Gauff"        -> "coco gauff"
        "C. Alcaraz"        -> "c alcaraz"
        "Polona Hercog"     -> "polona hercog"
        "Lazaro Garcia, Andrea" -> "andrea lazaro garcia"
    """
    name = name.strip()
    if not name:
        return ''

    # "Last, First" -> "First Last"
    if ',' in name:
        parts = name.split(',', 1)
        name = f"{parts[1].strip()} {parts[0].strip()}"

    return normalize_str(name)


# Known nickname -> real name mappings
NICKNAME_MAP = {
    'coco': 'cori',         # Coco Gauff -> Cori Gauff
    'sasha': 'alexander',   # Sasha Zverev -> Alexander Zverev
    'rafa': 'rafael',       # Rafa Nadal -> Rafael Nadal
    'andy': 'andrew',       # Andy Murray
    'alex': 'alexander',    # Alex de Minaur
}


def names_match(name_a, name_b):
    """Check if two player names refer to the same person.

    Uses normalized comparison with nickname support and last-name matching.
    """
    norm_a = normalize_player_name(name_a)
    norm_b = normalize_player_name(name_b)

    if not norm_a or not norm_b:
        return False

    # Exact match
    if norm_a == norm_b:
        return True

    # Split into parts
    parts_a = norm_a.split()
    parts_b = norm_b.split()

    if not parts_a or not parts_b:
        return False

    # Last name must match (last word)
    if parts_a[-1] != parts_b[-1]:
        # Try compound last names: "lazaro garcia" vs "garcia"
        if not (parts_a[-1] in parts_b or parts_b[-1] in parts_a):
            return False

    # First name/initial match
    first_a = parts_a[0]
    first_b = parts_b[0]

    # Direct match
    if first_a == first_b:
        return True

    # One starts with the other (initial vs full name)
    if first_a.startswith(first_b) or first_b.startswith(first_a):
        return True

    # Nickname match
    resolved_a = NICKNAME_MAP.get(first_a, first_a)
    resolved_b = NICKNAME_MAP.get(first_b, first_b)
    if resolved_a == resolved_b:
        return True
    if resolved_a.startswith(resolved_b) or resolved_b.startswith(resolved_a):
        return True

    return False


def build_player_index(players_csv):
    """Build multiple lookup indexes from atp_players.csv for name matching.

    Returns:
        player_data: dict {player_id: {name_first, name_last, hand, height, dob, ioc}}
        name_index: dict for matching Flashscore names to player_ids
    """
    player_data = {}
    # Index: (normalized_last_name, first_initial) -> list of player_ids
    last_initial_index = {}
    # Index: normalized_full_name -> list of player_ids
    full_name_index = {}

    with open(players_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row['player_id'])
            first = row.get('name_first', '').strip()
            last = row.get('name_last', '').strip()
            if not last:
                continue

            player_data[pid] = {
                'name_first': first,
                'name_last': last,
                'hand': row.get('hand', 'U') or 'U',
                'height': row.get('height', ''),
                'dob': row.get('dob', ''),
                'ioc': row.get('ioc', ''),
            }

            # Normalized last name + first initial index
            norm_last = normalize_str(last)
            norm_first = normalize_str(first)
            first_init = norm_first[0] if norm_first else ''

            key = (norm_last, first_init)
            if key not in last_initial_index:
                last_initial_index[key] = []
            last_initial_index[key].append(pid)

            # Also index without hyphens/spaces for compound names
            # "Auger Aliassime" -> "augeraliassime"
            compact_last = norm_last.replace(' ', '')
            if compact_last != norm_last:
                key2 = (compact_last, first_init)
                if key2 not in last_initial_index:
                    last_initial_index[key2] = []
                last_initial_index[key2].append(pid)

            # Full name index
            norm_full = f"{normalize_str(first)} {norm_last}"
            if norm_full not in full_name_index:
                full_name_index[norm_full] = []
            full_name_index[norm_full].append(pid)

    return player_data, last_initial_index, full_name_index


def parse_flashscore_name(name):
    """Parse Flashscore name format into (last_name, first_initials).

    Examples:
        "Alcaraz C."       -> ("Alcaraz", "C")
        "De Minaur A."     -> ("De Minaur", "A")
        "Auger-Aliassime F." -> ("Auger-Aliassime", "F")
        "Cerundolo J. M."  -> ("Cerundolo", "J")
        "Mpetshi Perricard G." -> ("Mpetshi Perricard", "G")
    """
    name = name.strip()
    # Pattern: everything before the last single-letter-dot sequences = last name
    # Match trailing initials: one or more (letter + ".")
    m = re.match(r'^(.+?)\s+([A-Z])\.\s*(?:[A-Z]\.\s*)*$', name)
    if m:
        last_name = m.group(1).strip()
        first_initial = m.group(2).lower()
        return last_name, first_initial

    # Fallback: split on last space
    parts = name.rsplit(' ', 1)
    if len(parts) == 2:
        return parts[0], parts[1][0].lower() if parts[1] else ''
    return name, ''


def match_player(flash_name, last_initial_index, full_name_index, player_data,
                 recent_ids=None):
    """Match a Flashscore player name to a Sackmann player_id.

    Args:
        flash_name: e.g. "Alcaraz C."
        last_initial_index: {(norm_last, first_init): [player_ids]}
        full_name_index: {norm_full: [player_ids]}
        player_data: {player_id: {...}}
        recent_ids: set of player_ids active in recent years (for disambiguation)

    Returns:
        player_id or None
    """
    last_name, first_init = parse_flashscore_name(flash_name)
    norm_last = normalize_str(last_name)

    # Try exact last + initial match
    key = (norm_last, first_init)
    candidates = last_initial_index.get(key, [])

    # Also try without hyphens (Auger-Aliassime -> augeraliassime)
    compact_last = norm_last.replace(' ', '')
    if compact_last != norm_last:
        candidates = candidates + last_initial_index.get((compact_last, first_init), [])

    # Also try hyphen -> space conversion
    if '-' in last_name:
        alt_last = normalize_str(last_name.replace('-', ' '))
        candidates = candidates + last_initial_index.get((alt_last, first_init), [])
        alt_compact = alt_last.replace(' ', '')
        candidates = candidates + last_initial_index.get((alt_compact, first_init), [])

    # Deduplicate
    candidates = list(dict.fromkeys(candidates))

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Multiple candidates — disambiguate using recent_ids
    if recent_ids:
        recent = [pid for pid in candidates if pid in recent_ids]
        if len(recent) == 1:
            return recent[0]
        if recent:
            candidates = recent

    # Disambiguate by most recent DOB (youngest active player is most likely)
    best_pid = None
    best_dob = ''
    for pid in candidates:
        dob = player_data[pid].get('dob', '')
        if dob and dob > best_dob:
            best_dob = dob
            best_pid = pid

    return best_pid


def match_full_name(full_name, last_initial_index, full_name_index, player_data,
                    recent_ids=None):
    """Match a full Western-order name to a Sackmann player_id.

    Handles formats like "Carlos Alcaraz" from The Odds API.

    Args:
        full_name: e.g. "Carlos Alcaraz", "Felix Auger-Aliassime"

    Returns:
        player_id or None
    """
    full_name = full_name.strip()
    if not full_name:
        return None

    # Try direct full name lookup
    norm_full = normalize_str(full_name)
    candidates = full_name_index.get(norm_full, [])
    if len(candidates) == 1:
        return candidates[0]

    # Try splitting into first/last and using last_initial_index
    parts = full_name.strip().split()
    if len(parts) >= 2:
        # Assume last word(s) are the last name — try progressively
        for split_at in range(1, len(parts)):
            first_part = ' '.join(parts[:split_at])
            last_part = ' '.join(parts[split_at:])
            norm_last = normalize_str(last_part)
            first_init = normalize_str(first_part)[0] if first_part else ''

            key = (norm_last, first_init)
            hits = last_initial_index.get(key, [])

            # Also try compact
            compact_last = norm_last.replace(' ', '')
            if compact_last != norm_last:
                hits = hits + last_initial_index.get((compact_last, first_init), [])

            hits = list(dict.fromkeys(hits))
            if len(hits) == 1:
                return hits[0]
            if hits and len(hits) < len(candidates) or not candidates:
                candidates = hits

    # Deduplicate
    candidates = list(dict.fromkeys(candidates))

    if not candidates:
        # Fuzzy fallback
        return _fuzzy_match(full_name, player_data, recent_ids)

    if len(candidates) == 1:
        return candidates[0]

    # Disambiguate
    return _disambiguate(candidates, player_data, recent_ids)


def match_loro_name(loro_name, last_initial_index, full_name_index, player_data,
                    recent_ids=None):
    """Match a LORO/OpenBet player name to a Sackmann player_id.

    Handles various formats:
        "C. Alcaraz"          → initial-first format
        "Alcaraz, C."         → last-comma-initial format
        "Carlos Alcaraz"      → full name (delegates to match_full_name)
        "Alcaraz C."          → Flashscore-style (delegates to match_player)
    """
    loro_name = loro_name.strip()
    if not loro_name:
        return None

    # Format: "Alcaraz, C." or "Alcaraz, Carlos"
    if ',' in loro_name:
        parts = loro_name.split(',', 1)
        last_name = parts[0].strip()
        first_part = parts[1].strip().rstrip('.')
        norm_last = normalize_str(last_name)
        first_init = normalize_str(first_part)[0] if first_part else ''

        key = (norm_last, first_init)
        candidates = last_initial_index.get(key, [])
        compact = norm_last.replace(' ', '')
        if compact != norm_last:
            candidates = candidates + last_initial_index.get((compact, first_init), [])
        candidates = list(dict.fromkeys(candidates))

        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            return _disambiguate(candidates, player_data, recent_ids)

    # Format: "C. Alcaraz" (initial dot space lastname)
    m = re.match(r'^([A-Z])\.\s+(.+)$', loro_name)
    if m:
        first_init = m.group(1).lower()
        last_name = m.group(2).strip()
        norm_last = normalize_str(last_name)

        key = (norm_last, first_init)
        candidates = last_initial_index.get(key, [])
        compact = norm_last.replace(' ', '')
        if compact != norm_last:
            candidates = candidates + last_initial_index.get((compact, first_init), [])
        candidates = list(dict.fromkeys(candidates))

        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            return _disambiguate(candidates, player_data, recent_ids)

    # Format: "Alcaraz C." — Flashscore-style, delegate
    if re.search(r'\s[A-Z]\.\s*$', loro_name):
        return match_player(loro_name, last_initial_index, full_name_index,
                            player_data, recent_ids)

    # Format: full name "Carlos Alcaraz"
    return match_full_name(loro_name, last_initial_index, full_name_index,
                           player_data, recent_ids)


def harmonize_name(name, source, last_initial_index, full_name_index,
                   player_data, recent_ids=None):
    """Unified name matching dispatcher.

    Args:
        name: Player name string
        source: One of 'flashscore', 'odds_api', 'loro'
        (remaining args: indexes from build_player_index)

    Returns:
        player_id or None
    """
    dispatch = {
        'flashscore': match_player,
        'odds_api': match_full_name,
        'loro': match_loro_name,
    }
    fn = dispatch.get(source, match_full_name)
    return fn(name, last_initial_index, full_name_index, player_data, recent_ids)


def _disambiguate(candidates, player_data, recent_ids=None):
    """Disambiguate multiple player_id candidates."""
    if recent_ids:
        recent = [pid for pid in candidates if pid in recent_ids]
        if len(recent) == 1:
            return recent[0]
        if recent:
            candidates = recent

    # Pick youngest
    best_pid = None
    best_dob = ''
    for pid in candidates:
        dob = player_data.get(pid, {}).get('dob', '')
        if dob and dob > best_dob:
            best_dob = dob
            best_pid = pid
    return best_pid


def _fuzzy_match(name, player_data, recent_ids=None, threshold=85):
    """Fuzzy fallback: compare against all player full names."""
    if not HAS_FUZZY:
        return None

    norm_input = normalize_str(name)
    best_pid = None
    best_score = 0

    # Only search recent players for performance
    search_pool = recent_ids if recent_ids else player_data.keys()

    for pid in search_pool:
        info = player_data.get(pid)
        if not info:
            continue
        full = f"{normalize_str(info['name_first'])} {normalize_str(info['name_last'])}"
        score = fuzz.ratio(norm_input, full)
        if score > best_score:
            best_score = score
            best_pid = pid

    return best_pid if best_score >= threshold else None


def get_tourney_level(tourney_name):
    """Map tournament name to Sackmann tourney_level code."""
    name_lower = tourney_name.lower().strip()
    for key, level in TOURNEY_LEVEL_MAP.items():
        if key in name_lower:
            return level
    return 'A'  # Default to ATP 250/500


def compute_age(dob_str, tourney_date_str):
    """Compute age in years from DOB string (YYYYMMDD) and tournament date (YYYYMMDD)."""
    if not dob_str or not tourney_date_str or len(dob_str) < 8 or len(tourney_date_str) < 8:
        return ''
    try:
        dob = datetime.strptime(dob_str[:8], '%Y%m%d')
        tdate = datetime.strptime(tourney_date_str[:8], '%Y%m%d')
        age = (tdate - dob).days / 365.25
        return f"{age:.1f}"
    except (ValueError, TypeError):
        return ''


def build_recent_player_ids(data_dir):
    """Build set of player IDs that appear in recent Sackmann match files."""
    recent_ids = set()
    for year in range(2020, 2025):
        fp = data_dir / f'atp_matches_{year}.csv'
        if fp.exists():
            with open(fp) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('winner_id'):
                        recent_ids.add(int(row['winner_id']))
                    if row.get('loser_id'):
                        recent_ids.add(int(row['loser_id']))
    return recent_ids


def main():
    parser = argparse.ArgumentParser(description='Transform scraped data to Sackmann format')
    parser.add_argument('--input', type=str,
                        default=str(DATA_DIR / 'atp_matches_scraped_2025_2026.csv'),
                        help='Input scraped CSV')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV (default: data/atp_matches_YEAR.csv)')
    parser.add_argument('--year', type=int, default=None,
                        help='Year label for output file')
    args = parser.parse_args()

    # Load player database
    players_csv = DATA_DIR / 'atp_players.csv'
    print(f"Loading player database from {players_csv}...")
    player_data, last_initial_index, full_name_index = build_player_index(players_csv)
    print(f"  Loaded {len(player_data)} players")

    # Build recent player IDs for disambiguation
    print("Building recent player ID set...")
    recent_ids = build_recent_player_ids(DATA_DIR)
    print(f"  {len(recent_ids)} players active 2020-2024")

    # Load scraped data
    print(f"\nLoading scraped data from {args.input}...")
    with open(args.input) as f:
        reader = csv.DictReader(f)
        scraped_rows = list(reader)
    print(f"  {len(scraped_rows)} matches loaded")

    # Build name cache to avoid repeated lookups
    name_cache = {}
    unmatched_names = set()

    def resolve_player(flash_name):
        if flash_name in name_cache:
            return name_cache[flash_name]
        pid = match_player(flash_name, last_initial_index, full_name_index,
                           player_data, recent_ids)
        name_cache[flash_name] = pid
        if pid is None:
            unmatched_names.add(flash_name)
        return pid

    # Transform rows
    output_rows = []
    skipped = 0
    tourney_counters = {}

    for row in scraped_rows:
        winner_name = row['winner_name']
        loser_name = row['loser_name']

        winner_id = resolve_player(winner_name)
        loser_id = resolve_player(loser_name)

        if winner_id is None or loser_id is None:
            skipped += 1
            continue

        w_info = player_data[winner_id]
        l_info = player_data[loser_id]

        tourney_name = row['tourney_name']
        tourney_date = row.get('tourney_date', '')
        tourney_level = get_tourney_level(tourney_name)

        # Generate tourney_id
        year_str = tourney_date[:4] if len(tourney_date) >= 4 else str(args.year or datetime.now().year)
        tourney_slug = tourney_name.lower().replace(' ', '-')
        tourney_id = f"{year_str}-{tourney_slug}"

        # Match number (sequential within tournament)
        if tourney_id not in tourney_counters:
            tourney_counters[tourney_id] = 0
        tourney_counters[tourney_id] += 1
        match_num = tourney_counters[tourney_id]

        # Compute ages
        winner_age = compute_age(w_info['dob'], tourney_date)
        loser_age = compute_age(l_info['dob'], tourney_date)

        out = {
            'tourney_id': tourney_id,
            'tourney_name': tourney_name,
            'surface': row.get('surface', ''),
            'draw_size': '',
            'tourney_level': tourney_level,
            'tourney_date': tourney_date,
            'match_num': match_num,
            'winner_id': winner_id,
            'winner_seed': '',
            'winner_entry': '',
            'winner_name': f"{w_info['name_first']} {w_info['name_last']}",
            'winner_hand': w_info['hand'],
            'winner_ht': w_info['height'],
            'winner_ioc': w_info['ioc'],
            'winner_age': winner_age,
            'loser_id': loser_id,
            'loser_seed': '',
            'loser_entry': '',
            'loser_name': f"{l_info['name_first']} {l_info['name_last']}",
            'loser_hand': l_info['hand'],
            'loser_ht': l_info['height'],
            'loser_ioc': l_info['ioc'],
            'loser_age': loser_age,
            'score': row.get('score', ''),
            'best_of': row.get('best_of', '3'),
            'round': row.get('round', ''),
            'minutes': '',
            'w_ace': row.get('w_ace', ''),
            'w_df': row.get('w_df', ''),
            'w_svpt': row.get('w_svpt', ''),
            'w_1stIn': row.get('w_1stIn', ''),
            'w_1stWon': row.get('w_1stWon', ''),
            'w_2ndWon': row.get('w_2ndWon', ''),
            'w_SvGms': '',
            'w_bpSaved': row.get('w_bpSaved', ''),
            'w_bpFaced': row.get('w_bpFaced', ''),
            'l_ace': row.get('l_ace', ''),
            'l_df': row.get('l_df', ''),
            'l_svpt': row.get('l_svpt', ''),
            'l_1stIn': row.get('l_1stIn', ''),
            'l_1stWon': row.get('l_1stWon', ''),
            'l_2ndWon': row.get('l_2ndWon', ''),
            'l_SvGms': '',
            'l_bpSaved': row.get('l_bpSaved', ''),
            'l_bpFaced': row.get('l_bpFaced', ''),
            'winner_rank': row.get('winner_rank', ''),
            'winner_rank_points': '',
            'loser_rank': row.get('loser_rank', ''),
            'loser_rank_points': '',
        }

        output_rows.append(out)

    # Determine output file
    if args.output:
        output_file = args.output
    elif args.year:
        output_file = str(DATA_DIR / f'atp_matches_{args.year}.csv')
    else:
        # Auto-detect year from data
        years = set()
        for row in output_rows:
            d = row.get('tourney_date', '')
            if len(d) >= 4:
                years.add(d[:4])
        year_label = max(years) if years else str(datetime.now().year)
        output_file = str(DATA_DIR / f'atp_matches_{year_label}.csv')

    # Write output
    fieldnames = [
        'tourney_id', 'tourney_name', 'surface', 'draw_size', 'tourney_level',
        'tourney_date', 'match_num',
        'winner_id', 'winner_seed', 'winner_entry', 'winner_name', 'winner_hand',
        'winner_ht', 'winner_ioc', 'winner_age',
        'loser_id', 'loser_seed', 'loser_entry', 'loser_name', 'loser_hand',
        'loser_ht', 'loser_ioc', 'loser_age',
        'score', 'best_of', 'round', 'minutes',
        'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms',
        'w_bpSaved', 'w_bpFaced',
        'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms',
        'l_bpSaved', 'l_bpFaced',
        'winner_rank', 'winner_rank_points', 'loser_rank', 'loser_rank_points',
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    # Print summary
    print(f"\n{'='*60}")
    print(f"TRANSFORMATION SUMMARY")
    print(f"{'='*60}")
    print(f"Input matches:    {len(scraped_rows)}")
    print(f"Output matches:   {len(output_rows)}")
    print(f"Skipped (no ID):  {skipped}")
    print(f"Unique players:   {len(name_cache)}")
    print(f"Matched players:  {len(name_cache) - len(unmatched_names)}")
    print(f"Unmatched players: {len(unmatched_names)}")

    if unmatched_names:
        print(f"\nUnmatched names:")
        for name in sorted(unmatched_names):
            print(f"  - {name}")

    # Show some sample matches
    print(f"\nSample output (first 3 matches):")
    for row in output_rows[:3]:
        print(f"  {row['winner_name']} (id={row['winner_id']}, age={row['winner_age']}, "
              f"ht={row['winner_ht']}, hand={row['winner_hand']}) d. "
              f"{row['loser_name']} | {row['surface']} {row['tourney_level']}")

    print(f"\nOutput saved to: {output_file}")


if __name__ == '__main__':
    main()
