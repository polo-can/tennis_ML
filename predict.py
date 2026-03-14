"""
Predict outcomes for upcoming ATP matches.

Replays all historical match data to build current player state (Elo ratings,
form, head-to-head, serve stats, etc.), then computes features for each upcoming
match and outputs win probabilities with confidence levels.

Usage:
    python3 predict.py                                    # Default: reads data/upcoming_matches.csv
    python3 predict.py --input data/my_matches.csv        # Custom input
    python3 predict.py --output predictions.csv           # Custom output
    python3 predict.py --save-state                       # Cache player state for faster re-runs
    python3 predict.py --load-state                       # Load cached state instead of replaying
"""

import argparse
import pickle
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from transform_scraped import (
    build_player_index,
    match_player,
    build_recent_player_ids,
)

DATA_DIR = Path('data')

# ── Elo System (from notebook cell 7) ───────────────────────────────────────

K_FACTORS = {'G': 32, 'M': 24, 'A': 16, 'D': 12, 'F': 24}
DEFAULT_K = 16
INITIAL_ELO = 1500.0


def elo_expected(player_elo, opponent_elo):
    """Expected win probability."""
    return 1.0 / (1.0 + 10.0 ** ((opponent_elo - player_elo) / 400.0))


def elo_update(player_elo, opponent_elo, won, k):
    """Return new Elo after a match result."""
    expected = elo_expected(player_elo, opponent_elo)
    return player_elo + k * (won - expected)


# ── Player State Storage (from notebook cell 9) ────────────────────────────

elo_overall = defaultdict(lambda: INITIAL_ELO)
elo_surface = defaultdict(lambda: defaultdict(lambda: INITIAL_ELO))
match_history = defaultdict(list)
h2h_record = defaultdict(lambda: {'wins': 0, 'losses': 0})
h2h_surface_record = defaultdict(lambda: {'wins': 0, 'losses': 0})
surface_record = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0}))
level_record = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0}))
win_streak = defaultdict(int)
lose_streak = defaultdict(int)
total_wins = defaultdict(int)
total_losses = defaultdict(int)


# ── Feature Extraction Functions (from notebook cells 11-18) ────────────────

def get_elo_features(player_id, opponent_id, surface):
    """Elo ratings and expected win probability."""
    p_elo = elo_overall[player_id]
    o_elo = elo_overall[opponent_id]
    p_surf_elo = elo_surface[player_id][surface]
    o_surf_elo = elo_surface[opponent_id][surface]

    return {
        'elo': p_elo,
        'elo_opponent': o_elo,
        'elo_diff': p_elo - o_elo,
        'elo_expected': elo_expected(p_elo, o_elo),
        'surface_elo': p_surf_elo,
        'surface_elo_opponent': o_surf_elo,
        'surface_elo_diff': p_surf_elo - o_surf_elo,
        'surface_elo_expected': elo_expected(p_surf_elo, o_surf_elo),
    }


def get_form_features(player_id, match_date):
    """Recent form: win rates over different windows."""
    history = match_history[player_id]
    if not history:
        return {
            'form_last10': np.nan,
            'form_last20': np.nan,
            'form_52w': np.nan,
            'form_weighted': np.nan,
            'matches_played': 0,
        }

    last10 = history[-10:]
    last20 = history[-20:]

    cutoff_52w = match_date - timedelta(weeks=52)
    recent_52w = [m for m in history if m['date'] >= cutoff_52w]

    decay = 0.9
    last30 = history[-30:]
    if last30:
        weights = np.array([decay ** i for i in range(len(last30) - 1, -1, -1)])
        results = np.array([m['won'] for m in last30], dtype=float)
        weighted_form = np.sum(weights * results) / np.sum(weights)
    else:
        weighted_form = np.nan

    def win_rate(matches):
        if not matches:
            return np.nan
        return np.mean([m['won'] for m in matches])

    return {
        'form_last10': win_rate(last10),
        'form_last20': win_rate(last20),
        'form_52w': win_rate(recent_52w),
        'form_weighted': weighted_form,
        'matches_played': len(history),
    }


def get_h2h_features(player_id, opponent_id, surface):
    """Head-to-head record between the two players."""
    rec = h2h_record[(player_id, opponent_id)]
    total = rec['wins'] + rec['losses']

    surf_rec = h2h_surface_record[(player_id, opponent_id, surface)]
    surf_total = surf_rec['wins'] + surf_rec['losses']

    return {
        'h2h_wins': rec['wins'],
        'h2h_losses': rec['losses'],
        'h2h_total': total,
        'h2h_win_pct': rec['wins'] / total if total > 0 else np.nan,
        'h2h_surface_wins': surf_rec['wins'],
        'h2h_surface_losses': surf_rec['losses'],
        'h2h_surface_win_pct': surf_rec['wins'] / surf_total if surf_total > 0 else np.nan,
    }


def get_surface_features(player_id, surface, match_date):
    """Surface-specific performance."""
    rec = surface_record[player_id][surface]
    total = rec['wins'] + rec['losses']

    cutoff = match_date - timedelta(weeks=52)
    recent = [m for m in match_history[player_id]
              if m['surface'] == surface and m['date'] >= cutoff]

    return {
        'surface_wins': rec['wins'],
        'surface_losses': rec['losses'],
        'surface_matches': total,
        'surface_win_pct': rec['wins'] / total if total > 0 else np.nan,
        'surface_recent_win_pct': np.mean([m['won'] for m in recent]) if recent else np.nan,
        'surface_recent_matches': len(recent),
    }


def get_fatigue_features(player_id, match_date):
    """Fatigue and scheduling features."""
    history = match_history[player_id]
    if not history:
        return {
            'days_since_last': np.nan,
            'matches_last_7d': 0,
            'matches_last_14d': 0,
            'matches_last_30d': 0,
        }

    last_date = history[-1]['date']
    days_since = (match_date - last_date).days

    m7 = sum(1 for m in history if (match_date - m['date']).days <= 7)
    m14 = sum(1 for m in history if (match_date - m['date']).days <= 14)
    m30 = sum(1 for m in history if (match_date - m['date']).days <= 30)

    return {
        'days_since_last': days_since,
        'matches_last_7d': m7,
        'matches_last_14d': m14,
        'matches_last_30d': m30,
    }


def get_momentum_features(player_id):
    """Current winning/losing streak."""
    return {
        'win_streak': win_streak[player_id],
        'lose_streak': lose_streak[player_id],
    }


def get_level_features(player_id, tourney_level):
    """Win rate by tournament level."""
    rec = level_record[player_id][tourney_level]
    total = rec['wins'] + rec['losses']

    gs = level_record[player_id]['G']
    gs_total = gs['wins'] + gs['losses']

    return {
        'level_wins': rec['wins'],
        'level_losses': rec['losses'],
        'level_win_pct': rec['wins'] / total if total > 0 else np.nan,
        'gs_win_pct': gs['wins'] / gs_total if gs_total > 0 else np.nan,
    }


def get_serve_features(player_id):
    """Rolling average serve statistics from past matches."""
    history = match_history[player_id]
    with_serve = [m for m in history if m.get('svpt') and m['svpt'] > 0]
    recent = with_serve[-30:]

    if not recent:
        return {
            'avg_ace_rate': np.nan,
            'avg_df_rate': np.nan,
            'avg_first_serve_pct': np.nan,
            'avg_first_serve_win_pct': np.nan,
            'avg_second_serve_win_pct': np.nan,
            'avg_bp_saved_pct': np.nan,
        }

    n = len(recent)
    weights = np.array([0.95 ** (n - 1 - i) for i in range(n)])
    weights /= weights.sum()

    def wavg(key):
        vals = [m.get(key, np.nan) for m in recent]
        mask = ~np.isnan(vals)
        if not mask.any():
            return np.nan
        return np.average(np.array(vals)[mask], weights=weights[mask])

    return {
        'avg_ace_rate': wavg('ace_rate'),
        'avg_df_rate': wavg('df_rate'),
        'avg_first_serve_pct': wavg('first_serve_pct'),
        'avg_first_serve_win_pct': wavg('first_serve_win_pct'),
        'avg_second_serve_win_pct': wavg('second_serve_win_pct'),
        'avg_bp_saved_pct': wavg('bp_saved_pct'),
    }


# ── State Update Functions (from notebook cells 22-23) ──────────────────────

def compute_serve_stats(row, prefix):
    """Compute serve stat ratios for a player from a match row."""
    svpt = row.get(f'{prefix}_svpt', 0)
    if not svpt or (isinstance(svpt, float) and np.isnan(svpt)) or svpt == 0:
        return {}

    svpt = float(svpt)
    first_in = float(row.get(f'{prefix}_1stIn', 0) or 0)
    second_attempts = svpt - first_in
    bp_faced = float(row.get(f'{prefix}_bpFaced', 0) or 0)

    stats = {
        'svpt': svpt,
        'ace_rate': float(row.get(f'{prefix}_ace', 0) or 0) / svpt,
        'df_rate': float(row.get(f'{prefix}_df', 0) or 0) / svpt,
        'first_serve_pct': first_in / svpt if svpt > 0 else np.nan,
        'first_serve_win_pct': float(row.get(f'{prefix}_1stWon', 0) or 0) / first_in if first_in > 0 else np.nan,
        'second_serve_win_pct': float(row.get(f'{prefix}_2ndWon', 0) or 0) / second_attempts if second_attempts > 0 else np.nan,
        'bp_saved_pct': float(row.get(f'{prefix}_bpSaved', 0) or 0) / bp_faced if bp_faced > 0 else np.nan,
    }
    return stats


def update_state(player_id, opponent_id, won, surface, tourney_level,
                 match_date, k_factor, row, prefix):
    """Update all player state after a match."""
    # 1. Update Elo
    opp_elo = elo_overall[opponent_id]
    elo_overall[player_id] = elo_update(elo_overall[player_id], opp_elo, won, k_factor)

    opp_surf_elo = elo_surface[opponent_id][surface]
    elo_surface[player_id][surface] = elo_update(
        elo_surface[player_id][surface], opp_surf_elo, won, k_factor
    )

    # 2. Compute serve stats for this match
    serve = compute_serve_stats(row, prefix)

    # 3. Get rank from this match
    if prefix == 'w':
        rank_val = row.get('winner_rank', np.nan)
    else:
        rank_val = row.get('loser_rank', np.nan)
    if isinstance(rank_val, str):
        try:
            rank_val = float(rank_val) if rank_val else np.nan
        except ValueError:
            rank_val = np.nan

    # 4. Append to match history
    entry = {
        'date': match_date,
        'won': won,
        'surface': surface,
        'tourney_level': tourney_level,
        'opponent_id': opponent_id,
        'rank': rank_val,
    }
    entry.update(serve)
    match_history[player_id].append(entry)

    # 5. Update h2h
    if won:
        h2h_record[(player_id, opponent_id)]['wins'] += 1
        h2h_surface_record[(player_id, opponent_id, surface)]['wins'] += 1
    else:
        h2h_record[(player_id, opponent_id)]['losses'] += 1
        h2h_surface_record[(player_id, opponent_id, surface)]['losses'] += 1

    # 6. Update surface record
    if won:
        surface_record[player_id][surface]['wins'] += 1
    else:
        surface_record[player_id][surface]['losses'] += 1

    # 7. Update level record
    if won:
        level_record[player_id][tourney_level]['wins'] += 1
    else:
        level_record[player_id][tourney_level]['losses'] += 1

    # 8. Update streaks
    if won:
        win_streak[player_id] += 1
        lose_streak[player_id] = 0
        total_wins[player_id] += 1
    else:
        lose_streak[player_id] += 1
        win_streak[player_id] = 0
        total_losses[player_id] += 1


# ── Profile Features for Prediction (adapted from notebook cell 19) ─────────

def get_profile_features(player_id, opponent_id, match_date, player_data):
    """Player profile features using current state + player database.

    Adapted from notebook's get_player_profile() to work without a match result
    row (since we don't know the winner/loser yet).
    """
    p_info = player_data.get(player_id, {})
    o_info = player_data.get(opponent_id, {})

    # Rank from most recent match history entry
    rank = np.nan
    for m in reversed(match_history[player_id]):
        r = m.get('rank', np.nan)
        if r is not None and not (isinstance(r, float) and np.isnan(r)):
            rank = float(r)
            break

    opp_rank = np.nan
    for m in reversed(match_history[opponent_id]):
        r = m.get('rank', np.nan)
        if r is not None and not (isinstance(r, float) and np.isnan(r)):
            opp_rank = float(r)
            break

    # Rank trajectory: compare current rank to rank 90 days ago
    rank_90d_ago = np.nan
    cutoff = match_date - timedelta(days=90)
    old_matches = [m for m in match_history[player_id]
                   if m['date'] <= cutoff and
                   m.get('rank') is not None and
                   not (isinstance(m.get('rank'), float) and np.isnan(m.get('rank')))]
    if old_matches:
        rank_90d_ago = float(old_matches[-1]['rank'])

    # Age from DOB
    def compute_age_from_dob(info):
        dob = info.get('dob', '')
        if dob and len(str(dob)) >= 8:
            try:
                dob_dt = datetime.strptime(str(dob)[:8], '%Y%m%d')
                return (match_date - dob_dt).days / 365.25
            except (ValueError, TypeError):
                pass
        return np.nan

    age = compute_age_from_dob(p_info)
    opp_age = compute_age_from_dob(o_info)

    # Height
    def get_height(info):
        ht = info.get('height', '')
        if ht and str(ht).strip():
            try:
                return float(ht)
            except (ValueError, TypeError):
                pass
        return np.nan

    ht = get_height(p_info)
    opp_ht = get_height(o_info)

    # Hand encoding
    hand = p_info.get('hand', 'U') or 'U'
    opp_hand = o_info.get('hand', 'U') or 'U'
    hand_map = {'R': 0, 'L': 1, 'U': 2}
    hand_enc = hand_map.get(hand, 2)
    opp_hand_enc = hand_map.get(opp_hand, 2)

    same_hand = int(hand == opp_hand)
    is_lefty = int(hand == 'L')
    opp_is_lefty = int(opp_hand == 'L')

    return {
        'rank': rank,
        'rank_points': np.nan,  # Not available for prediction
        'age': age,
        'height': ht,
        'hand': hand_enc,
        'opp_rank': opp_rank,
        'opp_rank_points': np.nan,
        'opp_age': opp_age,
        'opp_height': opp_ht,
        'opp_hand': opp_hand_enc,
        'rank_diff': rank - opp_rank if not (np.isnan(rank) or np.isnan(opp_rank)) else np.nan,
        'rank_points_diff': np.nan,
        'age_diff': age - opp_age if not (np.isnan(age) or np.isnan(opp_age)) else np.nan,
        'height_diff': ht - opp_ht if not (np.isnan(ht) or np.isnan(opp_ht)) else np.nan,
        'same_hand': same_hand,
        'is_lefty': is_lefty,
        'opp_is_lefty': opp_is_lefty,
        'rank_trajectory': rank - rank_90d_ago if not (np.isnan(rank) or np.isnan(rank_90d_ago)) else np.nan,
    }


def extract_prediction_features(player_id, opponent_id, surface, tourney_level,
                                 match_date, player_data):
    """Extract ALL features for one player before a predicted match.

    Same as notebook's extract_all_features() but uses get_profile_features()
    instead of get_player_profile() (which requires a match result row).
    """
    features = {}
    features.update(get_elo_features(player_id, opponent_id, surface))
    features.update(get_form_features(player_id, match_date))
    features.update(get_h2h_features(player_id, opponent_id, surface))
    features.update(get_surface_features(player_id, surface, match_date))
    features.update(get_fatigue_features(player_id, match_date))
    features.update(get_momentum_features(player_id))
    features.update(get_level_features(player_id, tourney_level))
    features.update(get_serve_features(player_id))
    features.update(get_profile_features(player_id, opponent_id, match_date, player_data))
    return features


# ── History Replay ──────────────────────────────────────────────────────────

def load_match_data():
    """Load and concatenate all historical match CSVs (2015-2026)."""
    dfs = []
    for year in range(2015, 2027):
        fp = DATA_DIR / f'atp_matches_{year}.csv'
        if fp.exists():
            df_year = pd.read_csv(fp)
            dfs.append(df_year)
            print(f'  {year}: {len(df_year)} matches')
        else:
            print(f'  {year}: no file found, skipping')

    if not dfs:
        print('ERROR: No match data files found!')
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
    df = df.sort_values(['tourney_date', 'tourney_id', 'match_num']).reset_index(drop=True)
    df['surface'] = df['surface'].replace('Carpet', 'Hard')

    print(f'  Total: {len(df)} matches ({df["tourney_date"].min().date()} to {df["tourney_date"].max().date()})')
    return df


def replay_history(df):
    """Replay all historical matches to build current player state."""
    print(f'\nReplaying {len(df)} matches...')
    for idx, row in df.iterrows():
        winner_id = row['winner_id']
        loser_id = row['loser_id']
        surface = row['surface']
        tourney_level = row['tourney_level']
        match_date = row['tourney_date']
        k = K_FACTORS.get(tourney_level, DEFAULT_K)

        update_state(winner_id, loser_id, 1, surface, tourney_level, match_date, k, row, 'w')
        update_state(loser_id, winner_id, 0, surface, tourney_level, match_date, k, row, 'l')

        if (idx + 1) % 5000 == 0:
            print(f'  {idx + 1}/{len(df)} matches replayed...')

    n_players = len(set(list(elo_overall.keys())))
    print(f'  Done! {n_players} players in state.')


def save_state(filepath):
    """Save player state to pickle for fast re-loading."""
    state = {
        'elo_overall': dict(elo_overall),
        'elo_surface': {k: dict(v) for k, v in elo_surface.items()},
        'match_history': dict(match_history),
        'h2h_record': dict(h2h_record),
        'h2h_surface_record': dict(h2h_surface_record),
        'surface_record': {k: dict(v) for k, v in surface_record.items()},
        'level_record': {k: dict(v) for k, v in level_record.items()},
        'win_streak': dict(win_streak),
        'lose_streak': dict(lose_streak),
        'total_wins': dict(total_wins),
        'total_losses': dict(total_losses),
    }
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)
    print(f'  State saved to {filepath}')


def load_state(filepath):
    """Load player state from pickle."""
    global elo_overall, elo_surface, match_history, h2h_record, h2h_surface_record
    global surface_record, level_record, win_streak, lose_streak, total_wins, total_losses

    with open(filepath, 'rb') as f:
        state = pickle.load(f)

    elo_overall.update(state['elo_overall'])
    for k, v in state['elo_surface'].items():
        elo_surface[k].update(v)
    match_history.update(state['match_history'])
    h2h_record.update(state['h2h_record'])
    h2h_surface_record.update(state['h2h_surface_record'])
    for k, v in state['surface_record'].items():
        surface_record[k].update(v)
    for k, v in state['level_record'].items():
        level_record[k].update(v)
    win_streak.update(state['win_streak'])
    lose_streak.update(state['lose_streak'])
    total_wins.update(state['total_wins'])
    total_losses.update(state['total_losses'])

    n_players = len(state['elo_overall'])
    print(f'  State loaded from {filepath} ({n_players} players)')


# ── Prediction ──────────────────────────────────────────────────────────────

LEVEL_NAMES = {'G': 'Grand Slam', 'M': 'Masters 1000', 'A': 'ATP 250/500', 'F': 'Tour Finals', 'D': 'Davis Cup'}


def main():
    parser = argparse.ArgumentParser(description='Predict upcoming ATP match outcomes')
    parser.add_argument('--input', type=str, default=str(DATA_DIR / 'upcoming_matches.csv'),
                        help='Input CSV of upcoming matches')
    parser.add_argument('--output', type=str, default=str(DATA_DIR / 'predictions.csv'),
                        help='Output CSV for predictions')
    parser.add_argument('--save-state', action='store_true',
                        help='Save player state to pickle after replay')
    parser.add_argument('--load-state', action='store_true',
                        help='Load player state from pickle instead of replaying')
    parser.add_argument('--state-file', type=str, default='player_state.pkl',
                        help='State pickle file path')
    args = parser.parse_args()

    # ── Phase 1: Build player state ─────────────────────────────────────────
    if args.load_state and Path(args.state_file).exists():
        print('Phase 1: Loading cached player state...')
        load_state(args.state_file)
    else:
        print('Phase 1: Loading historical match data...')
        df = load_match_data()
        replay_history(df)
        if args.save_state:
            save_state(args.state_file)

    # ── Phase 2: Load model ─────────────────────────────────────────────────
    print('\nPhase 2: Loading trained model...')
    model_path = Path('tennis_model.pkl')
    if not model_path.exists():
        print(f'ERROR: Model file not found: {model_path}')
        sys.exit(1)

    with open(model_path, 'rb') as f:
        artifacts = pickle.load(f)
        

    model = artifacts['model']
    feature_names = artifacts['feature_names']
    print(f'  Model loaded ({len(feature_names)} features)')

    # ── Phase 3: Load player database for name resolution ───────────────────
    print('\nPhase 3: Resolving player names...')
    players_csv = DATA_DIR / 'atp_players.csv'
    player_data, last_initial_index, full_name_index = build_player_index(players_csv)
    recent_ids = build_recent_player_ids(DATA_DIR)
    print(f'  {len(player_data)} players in database, {len(recent_ids)} recent')

    # ── Phase 4: Load upcoming matches ──────────────────────────────────────
    print(f'\nPhase 4: Loading upcoming matches from {args.input}...')
    try:
        upcoming = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f'ERROR: File not found: {args.input}')
        print('Run scrape_upcoming.py first to generate upcoming match data.')
        sys.exit(1)

    if len(upcoming) == 0:
        print('No upcoming matches found!')
        sys.exit(0)

    print(f'  {len(upcoming)} upcoming matches loaded')

    # ── Phase 5: Resolve names and predict ──────────────────────────────────
    print('\nPhase 5: Computing predictions...\n')
    today = datetime.now()
    name_cache = {}
    predictions = []
    unmatched = []

    for _, match in upcoming.iterrows():
        p1_name = match['player1_name']
        p2_name = match['player2_name']
        surface = match.get('surface', 'Hard')
        tourney_level = match.get('tourney_level', 'A')
        tourney_name = match.get('tourney_name', '')
        match_round = match.get('round', '')
        best_of = int(match.get('best_of', 3))

        # Resolve player names to IDs
        if p1_name not in name_cache:
            name_cache[p1_name] = match_player(
                p1_name, last_initial_index, full_name_index, player_data, recent_ids
            )
        if p2_name not in name_cache:
            name_cache[p2_name] = match_player(
                p2_name, last_initial_index, full_name_index, player_data, recent_ids
            )

        p1_id = name_cache[p1_name]
        p2_id = name_cache[p2_name]

        if p1_id is None or p2_id is None:
            unmatched.append((p1_name if p1_id is None else '', p2_name if p2_id is None else ''))
            continue

        # Extract features for both players
        p1_feat = extract_prediction_features(
            p1_id, p2_id, surface, tourney_level, today, player_data
        )
        p2_feat = extract_prediction_features(
            p2_id, p1_id, surface, tourney_level, today, player_data
        )

        # Compute diff features (p1 - p2) — same as notebook cell 27
        row_data = {}
        for k in p1_feat:
            v1 = p1_feat[k]
            v2 = p2_feat[k]
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                row_data[f'diff_{k}'] = v1 - v2

        # Add surface dummies and best_of
        for surf in ['Hard', 'Clay', 'Grass']:
            row_data[f'surface_{surf}'] = int(surface == surf)
        row_data['best_of'] = best_of

        # Build feature vector in correct order
        X = np.array([[row_data.get(f, np.nan) for f in feature_names]])

        # Predict
        prob = model.predict_proba(X)[0]
        p1_win_prob = prob[1]  # Probability that p1 wins (target=1 means p1 wins)
        p2_win_prob = prob[0]

        # Confidence level based on max probability
        max_prob = max(p1_win_prob, p2_win_prob)
        if max_prob >= 0.70:
            confidence = 'HIGH'
        elif max_prob >= 0.60:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        # Get Elo ratings for display
        p1_elo = elo_overall.get(p1_id, INITIAL_ELO)
        p2_elo = elo_overall.get(p2_id, INITIAL_ELO)

        # Get ranks for display
        p1_rank = np.nan
        for m in reversed(match_history.get(p1_id, [])):
            r = m.get('rank', np.nan)
            if r is not None and not (isinstance(r, float) and np.isnan(r)):
                p1_rank = int(r)
                break
        p2_rank = np.nan
        for m in reversed(match_history.get(p2_id, [])):
            r = m.get('rank', np.nan)
            if r is not None and not (isinstance(r, float) and np.isnan(r)):
                p2_rank = int(r)
                break

        predictions.append({
            'player1_name': p1_name,
            'player2_name': p2_name,
            'tourney_name': tourney_name,
            'tourney_slug': match.get('tourney_slug', ''),
            'surface': surface,
            'round': match_round,
            'best_of': best_of,
            'tourney_level': tourney_level,
            'p1_win_prob': p1_win_prob,
            'p2_win_prob': p2_win_prob,
            'confidence': confidence,
            'p1_elo': round(p1_elo, 1),
            'p2_elo': round(p2_elo, 1),
            'p1_rank': p1_rank,
            'p2_rank': p2_rank,
            'predicted_winner': p1_name if p1_win_prob > p2_win_prob else p2_name,
            'scheduled_time': match.get('scheduled_time', ''),
        })

    # ── Print Results ───────────────────────────────────────────────────────
    if not predictions:
        print('No predictions could be made (all players unmatched).')
        if unmatched:
            print(f'\nUnmatched players:')
            for p1, p2 in unmatched:
                names = [n for n in [p1, p2] if n]
                print(f'  {", ".join(names)}')
        sys.exit(1)

    # Group by tournament
    pred_df = pd.DataFrame(predictions)
    print('=' * 72)
    print('  MATCH PREDICTIONS')
    print('=' * 72)

    for tourney_name, group in pred_df.groupby('tourney_name', sort=False):
        surface = group.iloc[0]['surface']
        level = group.iloc[0]['tourney_level']
        level_name = LEVEL_NAMES.get(level, level)
        print(f'\n  {tourney_name} - {surface} ({level_name})')
        print(f'  {"-" * 68}')

        for _, pred in group.iterrows():
            p1 = pred['player1_name']
            p2 = pred['player2_name']
            p1_pct = pred['p1_win_prob'] * 100
            p2_pct = pred['p2_win_prob'] * 100
            conf = pred['confidence']
            rnd = pred['round']

            # Format rank display
            p1_rank_str = f"#{int(pred['p1_rank'])}" if not (isinstance(pred['p1_rank'], float) and np.isnan(pred['p1_rank'])) else '#?'
            p2_rank_str = f"#{int(pred['p2_rank'])}" if not (isinstance(pred['p2_rank'], float) and np.isnan(pred['p2_rank'])) else '#?'

            # Ensure round is a string
            rnd = str(rnd) if rnd and not (isinstance(rnd, float) and np.isnan(rnd)) else ''

            line = f'  {rnd:>4s}  {p1:<22s} ({p1_rank_str:>4s}) vs  {p2:<22s} ({p2_rank_str:>4s})  →  {p1_pct:5.1f}% / {p2_pct:5.1f}%  [{conf}]'

            print(line)

    # Summary stats
    high_conf = pred_df[pred_df['confidence'] == 'HIGH']
    med_conf = pred_df[pred_df['confidence'] == 'MEDIUM']
    low_conf = pred_df[pred_df['confidence'] == 'LOW']

    print(f'\n{"=" * 72}')
    print(f'  SUMMARY: {len(predictions)} predictions')
    print(f'    HIGH confidence:   {len(high_conf):3d} matches (≥70%)')
    print(f'    MEDIUM confidence: {len(med_conf):3d} matches (60-70%)')
    print(f'    LOW confidence:    {len(low_conf):3d} matches (50-60%)')
    print(f'{"=" * 72}')

    if unmatched:
        print(f'\n  ⚠ {len(unmatched)} matches skipped (unresolved player names):')
        for p1, p2 in unmatched[:10]:
            names = [n for n in [p1, p2] if n]
            print(f'    {", ".join(names)}')
        if len(unmatched) > 10:
            print(f'    ... and {len(unmatched) - 10} more')

    # ── Save predictions CSV ───────────────────────────────────────────────
    pred_df.to_csv(args.output, index=False)
    print(f'\nPredictions saved to {args.output}')


if __name__ == '__main__':
    main()
