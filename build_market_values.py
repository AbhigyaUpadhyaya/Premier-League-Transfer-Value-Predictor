# ==========================================================
# build_market_values.py
# ==========================================================
from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

# pyrefly: ignore [missing-import]
import duckdb
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np

# ============================================================
# PATHS & CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
STATS_FILE = RAW_DIR / "seasonal_player_stats.csv"
OUTPUT_FILE = RAW_DIR / "player_market_values.csv"
DB_FILE = PROJECT_ROOT / "transfermarkt-datasets.duckdb"

MIN_REAL_VALUES = 30
EUR_TO_GBP = 0.86

# ============================================================
# DYNAMIC SEASON HELPERS
# ============================================================

def get_current_ongoing_season() -> str:
    now = datetime.now()
    year = now.year
    if now.month >= 7:
        return f"{year}-{str(year + 1)[-2:]}"
    else:
        return f"{year - 1}-{str(year)[-2:]}"

CURRENT_ONGOING_SEASON = get_current_ongoing_season()

def extract_season_year(season_str: Any) -> int:
    try:
        return int(str(season_str).split("-")[0].strip())
    except Exception:
        return 0

def is_season_completed(season_str: Any) -> bool:
    season_year = extract_season_year(season_str)
    current_year = extract_season_year(CURRENT_ONGOING_SEASON)
    return 1900 < season_year < current_year

def season_from_date(value):
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return None
    year = int(date.year)
    if date.month >= 7:
        start = year
        end = year + 1
    else:
        start = year - 1
        end = year
    return f"{start}-{str(end)[-2:]}"

# ============================================================
# ROBUST NAME NORMALIZATION & MATCHING
# ============================================================

PARTICLES = {"de", "da", "do", "dos", "das", "del", "van", "von", "bin", "ibn", "mac", "mc"}

def normalize_name(value: Any) -> str:
    if not value: return ""
    text = str(value)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = text.lower()
    text = text.replace("-", " ").replace("'", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())

def remove_particles(norm_name: str) -> str:
    tokens = [t for t in norm_name.split() if t not in PARTICLES]
    return " ".join(tokens)

def get_first_last(norm_name: str) -> str:
    tokens = norm_name.split()
    if len(tokens) <= 1: return norm_name
    return f"{tokens[0]} {tokens[-1]}"

def generate_name_candidates(raw_name: str) -> set[str]:
    """Generates multiple representations of a name to maximize match probability."""
    candidates = set()
    norm = normalize_name(raw_name)
    if not norm: return candidates
    
    candidates.add(norm)
    
    no_parts = remove_particles(norm)
    if no_parts: candidates.add(no_parts)
    
    fl = get_first_last(no_parts)
    if fl: candidates.add(fl)
    
    return candidates

# ============================================================
# LOAD DATA
# ============================================================

def load_stats():
    if not STATS_FILE.exists():
        raise FileNotFoundError(f"Dataset not found:\n{STATS_FILE}\nRun: python collect_data.py")
    stats = pd.read_csv(STATS_FILE)
    stats["season"] = stats["season"].astype(str).str.strip()
    return stats

def load_tm_data(con):
    players = con.execute("""
        SELECT player_id, name, first_name, last_name
        FROM players
    """).fetchdf()

    valuations = con.execute("""
        SELECT pv.player_id, pv.date, pv.market_value_in_eur, c.name as current_club_name
        FROM player_valuations pv
        LEFT JOIN clubs c ON pv.current_club_id = c.club_id
        WHERE pv.market_value_in_eur IS NOT NULL AND pv.market_value_in_eur > 0
    """).fetchdf()
    
    appearances = con.execute("""
        SELECT player_id, date, minutes_played, goals, assists
        FROM appearances
    """).fetchdf()
    
    return players, valuations, appearances

# ============================================================
# MAIN BUILD PROCESS
# ============================================================

def main():
    print("=" * 64)
    print("BUILDING ALL-COMPETITION MARKET VALUE DATA")
    print("=" * 64)

    stats = load_stats()
    unique_fpl_players = stats.drop_duplicates("player_id")[["player_id", "player"]].copy()
    print(f"Loaded PL stats records: {len(stats):,} (Unique players: {len(unique_fpl_players):,})")

    if not DB_FILE.exists():
        raise FileNotFoundError(f"Transfermarkt database not found: {DB_FILE}")

    con = duckdb.connect(str(DB_FILE), read_only=True)
    try:
        print("\nLoading Transfermarkt database tables...")
        tm_players, valuations, appearances = load_tm_data(con)
    finally:
        con.close()

    print(f"Transfermarkt players loaded: {len(tm_players):,}")

    # ==========================================
    # 1. MULTI-STAGE IDENTITY MATCHING
    # ==========================================
    print("\nExecuting multi-stage player identity matching...")
    
    # Build TM candidate dictionary for fast lookups
    tm_candidate_map = {} # candidate_string -> set of tm_player_ids
    tm_token_map = {}     # tm_player_id -> set of name tokens
    
    for _, row in tm_players.iterrows():
        pid = str(row["player_id"])
        
        # Combine TM name fields
        names_to_parse = [row.get("name"), row.get("first_name"), row.get("last_name")]
        if pd.notna(row.get("first_name")) and pd.notna(row.get("last_name")):
            names_to_parse.append(f"{row['first_name']} {row['last_name']}")
            
        all_tokens = set()
        for raw_name in names_to_parse:
            if pd.isna(raw_name): continue
            
            # Generate representations (norm, no_particles, first_last)
            candidates = generate_name_candidates(raw_name)
            for c in candidates:
                tm_candidate_map.setdefault(c, set()).add(pid)
                
            # Store base tokens for subset matching fallback
            norm = normalize_name(raw_name)
            all_tokens.update(remove_particles(norm).split())
            
        tm_token_map[pid] = all_tokens

    # Match FPL players to TM players
    identity_map = {} # fpl_id -> tm_player_id
    
    match_stats = {"exact_norm": 0, "first_last": 0, "token_subset": 0, "unmatched": 0}

    for _, row in unique_fpl_players.iterrows():
        fpl_id = str(row["player_id"])
        raw_name = str(row["player"])
        
        candidates = generate_name_candidates(raw_name)
        fpl_tokens = set(remove_particles(normalize_name(raw_name)).split())
        
        matched_tm_id = None
        
        # Strategy 1 & 2: Direct Candidate Overlap (Exact, No Particles, First+Last)
        candidate_matches = set()
        for c in candidates:
            if c in tm_candidate_map:
                candidate_matches.update(tm_candidate_map[c])
                
        if len(candidate_matches) == 1:
            matched_tm_id = list(candidate_matches)[0]
            match_stats["exact_norm"] += 1
            
        # Strategy 3: Token Subset Fallback (e.g. "bruno borges fernandes" contains "bruno fernandes")
        if not matched_tm_id:
            subset_matches = []
            for tm_pid, tm_tokens in tm_token_map.items():
                if len(tm_tokens) > 1 and tm_tokens.issubset(fpl_tokens):
                    subset_matches.append(tm_pid)
                    
            if len(subset_matches) == 1:
                matched_tm_id = subset_matches[0]
                match_stats["token_subset"] += 1

        if matched_tm_id:
            identity_map[fpl_id] = matched_tm_id
        else:
            match_stats["unmatched"] += 1

    print("\nPLAYER MATCHING DIAGNOSTICS")
    print("-" * 64)
    print(f"Exact / First+Last Matches:    {match_stats['exact_norm']:,}")
    print(f"Token Subset Matches:          {match_stats['token_subset']:,}")
    print(f"Unmatched Players:             {match_stats['unmatched']:,}")

    # ==========================================
    # 2. AGGREGATE ALL-COMPETITION STATS
    # ==========================================
    print("\nAggregating all-competition match performance by season...")
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
    appearances["season"] = appearances["date"].apply(season_from_date)
    appearances["tm_player_id"] = appearances["player_id"].astype(str).str.strip()
    
    tm_stats_agg = appearances.groupby(["tm_player_id", "season"]).agg({
        "date": "count",
        "minutes_played": "sum",
        "goals": "sum",
        "assists": "sum"
    }).reset_index().rename(columns={"date": "tm_appearances"})

    # ==========================================
    # 3. PREPARE VALUATIONS
    # ==========================================
    print("Extracting actual season-end market valuations...")
    valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")
    valuations["season"] = valuations["date"].apply(season_from_date)
    valuations["tm_player_id"] = valuations["player_id"].astype(str).str.strip()
    
    valuations = valuations.sort_values(["tm_player_id", "season", "date"])
    player_season_values = valuations.groupby(["tm_player_id", "season"], as_index=False).tail(1).copy()

    # Build Value Lookup
    value_lookup = {}
    for _, row in player_season_values.iterrows():
        pid = str(row["tm_player_id"])
        season = str(row["season"])
        
        match_stats_df = tm_stats_agg[(tm_stats_agg["tm_player_id"] == pid) & (tm_stats_agg["season"] == season)]
        if not match_stats_df.empty:
            s = match_stats_df.iloc[0]
            tm_apps, tm_mins, tm_goals, tm_asts = s["tm_appearances"], s["minutes_played"], s["goals"], s["assists"]
        else:
            tm_apps, tm_mins, tm_goals, tm_asts = 0, 0, 0, 0
            
        value_lookup[(pid, season)] = {
            "market_value_gbp": float(row["market_value_in_eur"]) * EUR_TO_GBP,
            "transfermarkt_club": row.get("current_club_name"),
            "tm_appearances": tm_apps,
            "tm_minutes": tm_mins,
            "tm_goals": tm_goals,
            "tm_assists": tm_asts
        }

    # ==========================================
    # 4. GENERATE OUTPUT
    # ==========================================
    print("Merging Transfermarkt data into Premier League roster...")
    output_rows = []

    for _, row in stats.iterrows():
        fpl_id = str(row["player_id"])
        season = str(row["season"])
        
        tm_id = identity_map.get(fpl_id)
        value = value_lookup.get((tm_id, season)) if tm_id else None

        output_rows.append({
            "player_id": fpl_id,
            "player": row["player"],
            "season": season,
            "market_value_gbp": value["market_value_gbp"] if value else None,
            "transfermarkt_player_id": tm_id,
            "transfermarkt_club": value["transfermarkt_club"] if value else None,
            "total_appearances": value["tm_appearances"] if value else None,
            "total_minutes": value["tm_minutes"] if value else None,
            "total_goals": value["tm_goals"] if value else None,
            "total_assists": value["tm_assists"] if value else None,
            "is_completed_season": is_season_completed(season)
        })

    result = pd.DataFrame(output_rows)
    
    valid_completed = result[result["is_completed_season"] & result["market_value_gbp"].notna() & (result["market_value_gbp"] > 0)]
    
    print("\n" + "=" * 64)
    print("MARKET VALUE INTEGRATION DIAGNOSTICS")
    print("=" * 64)
    print(f"Total input player-seasons:              {len(result):,}")
    print(f"Valid completed records (with target):   {len(valid_completed):,}")

    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\nSaved market-value dataset to:\n{OUTPUT_FILE}")

    if len(valid_completed) < MIN_REAL_VALUES:
        raise SystemExit(f"Build failed. Only {len(valid_completed)} valid records created.")

if __name__ == "__main__":
    main()