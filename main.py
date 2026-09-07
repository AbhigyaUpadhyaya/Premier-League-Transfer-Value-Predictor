# ==========================================================
# main.py
# ==========================================================
from __future__ import annotations

import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
STATS_FILE = RAW_DIR / "seasonal_player_stats.csv"
MARKET_FILE = RAW_DIR / "player_market_values.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "transfer_value_model.joblib"
PICKLE_MODEL_FILE = MODEL_DIR / "transfer_value_model.pkl"

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

# ============================================================
# ROBUST NAME NORMALIZATION & SEARCHING
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

def format_currency(value) -> str:
    if value is None: return "Unknown"
    try:
        value = float(value)
    except (TypeError, ValueError): return "Unknown"
    
    if np.isnan(value): return "Unknown"
    if value < 1_000_000: return f"£{value:,.0f}"
    if value < 1_000_000_000: return f"£{value / 1_000_000:.2f}M"
    return f"£{value / 1_000_000_000:.2f}B"

def numeric(value):
    try:
        val = float(value)
        return np.nan if np.isnan(val) else val
    except (TypeError, ValueError):
        return np.nan

# ============================================================
# LOAD DATA & MODEL
# ============================================================

def load_data():
    if not STATS_FILE.exists() or not MARKET_FILE.exists():
        raise FileNotFoundError("Data files missing. Please run collect_data.py and build_market_values.py.")

    stats = pd.read_csv(STATS_FILE)
    market = pd.read_csv(MARKET_FILE)

    stats["player_id"] = stats["player_id"].astype(str).str.strip()
    market["player_id"] = market["player_id"].astype(str).str.strip()
    stats["season"] = stats["season"].astype(str).str.strip()
    market["season"] = market["season"].astype(str).str.strip()

    market = market.drop_duplicates(["player_id", "season"])
    merged = stats.merge(market, on=["player_id", "season"], how="left", suffixes=("", "_market"))

    merged["team"] = merged["team"].astype(str)
    mask = (merged["team"] == "Unknown") & merged["transfermarkt_club"].notna()
    merged.loc[mask, "team"] = merged.loc[mask, "transfermarkt_club"].astype(str)

    return merged

def load_model():
    if MODEL_FILE.exists():
        return joblib.load(MODEL_FILE)
    if PICKLE_MODEL_FILE.exists():
        return joblib.load(PICKLE_MODEL_FILE)
    return None

# ============================================================
# PLAYER SEARCH & SEASON SELECTION
# ============================================================

def find_player(df, query):
    """
    Finds a player using multi-stage identity resolution.
    Handles exact matches, first+last extraction, token subsets, and partial starts.
    """
    q_norm = normalize_name(query)
    q_no_parts = remove_particles(q_norm)
    q_first_last = get_first_last(q_no_parts)
    q_tokens = set(q_no_parts.split())
    
    search = df.copy()
    search["_norm"] = search["player"].astype(str).map(normalize_name)
    search["_no_parts"] = search["_norm"].map(remove_particles)
    search["_first_last"] = search["_no_parts"].map(get_first_last)
    
    # 1. Exact Match
    exact = search[search["_norm"] == q_norm]
    if not exact.empty: return exact
    
    # 2. First + Last Match
    fl_match = search[search["_first_last"] == q_first_last]
    if not fl_match.empty: return fl_match

    # 3. Token Subset Match (Is query a subset of db name, or vice versa)
    def is_subset_match(db_no_parts: str) -> bool:
        db_tokens = set(str(db_no_parts).split())
        if not db_tokens or not q_tokens: return False
        # Match if input is "Bruno Fernandes" and db is "Bruno Borges Fernandes"
        return q_tokens.issubset(db_tokens) or db_tokens.issubset(q_tokens)

    subset_mask = search["_no_parts"].apply(is_subset_match)
    subset = search[subset_mask]
    if not subset.empty: return subset
    
    # 4. Partial Starts-With (Fallback)
    starts = search[search["_norm"].str.startswith(q_norm, na=False)]
    if not starts.empty: return starts

    return pd.DataFrame()

def get_valid_completed_records(matches: pd.DataFrame) -> pd.DataFrame:
    """Filters for completed seasons that possess BOTH TM valuations and TM performance stats."""
    valid = matches.copy()
    valid = valid[valid["season"].apply(is_season_completed)]
    valid["market_value_gbp"] = pd.to_numeric(valid["market_value_gbp"], errors="coerce")
    valid = valid[valid["market_value_gbp"].notna() & (valid["market_value_gbp"] > 0)]

    for col in ["total_minutes", "total_appearances", "total_goals", "total_assists"]:
        valid[col] = pd.to_numeric(valid[col], errors="coerce")
        valid = valid[valid[col].notna()]

    valid["_season_order"] = valid["season"].apply(extract_season_year)
    return valid.sort_values("_season_order", ascending=False)

# ============================================================
# PREDICTION & DISPLAY
# ============================================================

def predict_player(model, player_record):
    if model is None: return None

    mins = numeric(player_record.get("total_minutes"))
    apps = numeric(player_record.get("total_appearances"))
    gls = numeric(player_record.get("total_goals"))
    asts = numeric(player_record.get("total_assists"))

    if any(np.isnan(x) for x in [mins, apps, gls, asts]): return None

    row = pd.DataFrame([{
        "total_minutes": mins,
        "total_appearances": apps,
        "total_goals": gls,
        "total_assists": asts,
        "position": player_record.get("position", "Unknown"),
        "team": player_record.get("team", "Unknown"),
    }])

    try:
        pred = model.predict(row)[0]
        return max(float(pred), 0)
    except Exception as exc:
        print(f"\nPrediction error: {exc}")
        return None

def display_player(matches, model, query_name=""):
    # Identity diagnostic logic
    player_name = matches.iloc[0].get("player", query_name) if not matches.empty else query_name
    
    # Check if TM identity exists
    tm_id = matches.iloc[0].get("transfermarkt_player_id")
    if pd.isna(tm_id):
        print("\n" + "=" * 64)
        print("PLAYER PROFILE")
        print("=" * 64)
        print(f"Name: {player_name}")
        print("\nDIAGNOSTIC: Player found in FPL roster, but Transfermarkt identity could not be resolved.")
        print("Market value comparison cannot be performed.")
        return

    valid_completed = get_valid_completed_records(matches)
    if valid_completed.empty:
        print("\n" + "=" * 64)
        print("PLAYER PROFILE")
        print("=" * 64)
        print(f"Name: {player_name}")
        print("\nDIAGNOSTIC: Transfermarkt player identified, but no valid completed-season")
        print("market value and/or performance data is available for this player.")
        return

    # Select the LATEST VALID COMPLETED SEASON
    player = valid_completed.iloc[0]

    name = player.get("player", "Unknown")
    team = player.get("team", "Unknown")
    position = player.get("position", "Unknown")
    season = player.get("season", "Unknown")

    tm_mins = player.get("total_minutes")
    tm_apps = player.get("total_appearances")
    tm_goals = player.get("total_goals")
    tm_asts = player.get("total_assists")

    actual_value = player.get("market_value_gbp")
    prediction = predict_player(model, player)

    print("\n" + "=" * 64)
    print("PLAYER PROFILE")
    print("=" * 64)
    print(f"Name:                {name}")
    print(f"Team:                {team}")
    print(f"Position:            {position}")
    print(f"Season:              {season} (Latest Completed Season)")
    
    print("\nAll-Competition Performance:")
    print(f"  Total Appearances: {int(tm_apps):,}")
    print(f"  Total Minutes:     {int(tm_mins):,}")
    print(f"  Total Goals:       {int(tm_goals):,}")
    print(f"  Total Assists:     {int(tm_asts):,}")

    photo_url = player.get("photo_url")
    if pd.notna(photo_url) and str(photo_url).strip():
        print(f"\nPhoto:               {str(photo_url).strip()}")

    print("\n" + "-" * 64)
    print("TRANSFER VALUE")
    print("-" * 64)
    print(f"Actual Transfermarkt value: {format_currency(actual_value)}")
    print(f"Predicted market value:     {format_currency(prediction) if prediction is not None else 'Unknown'}")

    print("\n" + "-" * 64)
    print("DIFFERENCE")
    print("-" * 64)

    if prediction is None or pd.isna(actual_value) or float(actual_value) <= 0:
        print("Difference:                 Unknown (missing predicted or actual value)")
    else:
        diff = prediction - float(actual_value)
        pct = (diff / float(actual_value)) * 100

        print(f"Difference:                 {format_currency(diff)}")
        print(f"Percentage difference:      {pct:+.1f}%")
        
        if diff > 0:
            print("\nThe model is OVERESTIMATING the player's value.")
        elif diff < 0:
            print("\nThe model is UNDERESTIMATING the player's value.")
        else:
            print("\nThe model's prediction matches the actual value exactly.")

    # Historical Completed Records
    if len(valid_completed) > 1:
        print("\n" + "=" * 64)
        print("HISTORICAL COMPLETED SEASONS")
        print("=" * 64)
        for _, hist_row in valid_completed.iloc[1:].iterrows():
            print(f"\nSeason: {hist_row.get('season')}")
            print(f"  Team:          {hist_row.get('team')}")
            print(f"  Total Minutes: {int(hist_row.get('total_minutes')):,}")
            print(f"  Total Goals:   {int(hist_row.get('total_goals')):,}")
            print(f"  Actual Value:  {format_currency(hist_row.get('market_value_gbp'))}")

def display_unknown_player(query):
    print(f"\nDIAGNOSTIC: No player found matching '{query}'.")

def interactive():
    print("=" * 64)
    print("PREMIER LEAGUE TRANSFER MARKET VALUE PREDICTOR")
    print("=" * 64)

    try:
        df = load_data()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        return

    model = load_model()
    print(f"\nTotal player-season records loaded: {len(df):,}")
    
    if model is None:
        print("\nWARNING: No trained model available. Run train_model.py first.")
    
    print("\nEnter a Premier League player name (e.g., Bruno Borges Fernandes, Rúben Dias).")
    print("Type 'exit' to quit.")

    while True:
        try:
            query = input("\nPlayer: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query: continue
        if query.lower() in {"exit", "quit", "q"}: break

        matches = find_player(df, query)
        if matches.empty:
            display_unknown_player(query)
            continue

        unique_names = matches["player"].dropna().astype(str).unique()
        if len(unique_names) > 1:
            print("\nMultiple players matched:")
            for idx, name in enumerate(unique_names, 1): print(f"  {idx}. {name}")
            try:
                choice = int(input("\nChoose a number: ").strip()) - 1
                if 0 <= choice < len(unique_names):
                    matches = matches[matches["player"] == unique_names[choice]]
                else:
                    print("Invalid selection.")
                    continue
            except (ValueError, EOFError, KeyboardInterrupt):
                continue

        display_player(matches, model, query_name=query)

def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        df = load_data()
        matches = find_player(df, query)
        if matches.empty:
            display_unknown_player(query)
        else:
            display_player(matches, load_model(), query_name=query)
    else:
        interactive()

if __name__ == "__main__":
    main()