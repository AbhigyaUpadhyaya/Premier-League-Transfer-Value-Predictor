# ==========================================================
# collect_data.py
# ==========================================================
from __future__ import annotations

import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================================================
# PATHS & CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RAW_DIR / "seasonal_player_stats.csv"

FPL_BASE_URL = "https://fantasy.premierleague.com/api"
BOOTSTRAP_ENDPOINT = f"{FPL_BASE_URL}/bootstrap-static/"
ELEMENT_SUMMARY_ENDPOINT = f"{FPL_BASE_URL}/element-summary/_player_id_placeholder/"

POSITION_MAP = {1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Forward"}

REQUEST_TIMEOUT = 30
REQUEST_DELAY_BASE = 0.3
REQUEST_DELAY_JITTER = 0.2
MAX_ATTEMPTS_PER_REQUEST = 3
RETRY_BACKOFF_SECONDS = 1.5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# ============================================================
# DYNAMIC SEASON HELPERS
# ============================================================

def get_current_ongoing_season() -> str:
    """
    Dynamically identifies the current ongoing football season based on system date.
    Cutover month is July (month >= 7).
    E.g., In September 2026, returns '2026-27'.
    """
    now = datetime.now()
    year = now.year
    if now.month >= 7:
        return f"{year}-{str(year + 1)[-2:]}"
    else:
        return f"{year - 1}-{str(year)[-2:]}"

CURRENT_ONGOING_SEASON = get_current_ongoing_season()

def extract_season_year(season_str: Any) -> int:
    """Safely extracts the starting year integer from a season string (e.g. '2025-26' -> 2025)."""
    try:
        return int(str(season_str).split("-")[0].strip())
    except Exception:
        return 0

def is_season_completed(season_str: Any) -> bool:
    """Returns True if the season is a completed historical season."""
    season_year = extract_season_year(season_str)
    current_year = extract_season_year(CURRENT_ONGOING_SEASON)
    return 1900 < season_year < current_year

# ============================================================
# HTTP SESSION
# ============================================================

def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def fetch_json(session: requests.Session, url: str) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS_PER_REQUEST + 1):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            if response.text.strip():
                return response.json()
            last_exc = requests.RequestException(f"Empty body (attempt {attempt}).")
        except requests.RequestException as e:
            last_exc = e
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise last_exc

def polite_sleep() -> None:
    time.sleep(REQUEST_DELAY_BASE + random.uniform(0, REQUEST_DELAY_JITTER))

# ============================================================
# UTILITIES
# ============================================================

def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        val = float(value)
        return max(0.0, val)
    except (TypeError, ValueError):
        return None

def normalize_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def format_season(fpl_season: str) -> str:
    return fpl_season.replace("/", "-").strip()

# ============================================================
# DATA COLLECTION
# ============================================================

def collect_data():
    session = make_session()
    
    print("\nFetching FPL bootstrap-static (teams and current players)...")
    bootstrap = fetch_json(session, BOOTSTRAP_ENDPOINT)
    
    teams = {team["id"]: team["name"] for team in bootstrap.get("teams", [])}
    elements = bootstrap.get("elements", [])
    
    print(f"Current FPL players found: {len(elements):,}")
    print(f"Dynamic Current Ongoing Season: {CURRENT_ONGOING_SEASON}")
    
    rows = []
    total = len(elements)
    
    print("\n" + "=" * 64)
    print("COLLECTING PLAYER ROSTER & FPL HISTORICAL DATA")
    print("=" * 64)
    
    for index, element in enumerate(elements):
        fpl_id = element.get("id")
        first_name = element.get("first_name", "")
        second_name = element.get("second_name", "")
        player_name = f"{first_name} {second_name}".strip()
        team_id = element.get("team")
        team_name = teams.get(team_id, "Unknown")
        position = POSITION_MAP.get(element.get("element_type"), "Unknown")
        fpl_price = element.get("now_cost", 0) / 10.0
        
        photo_url = f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{str(element.get('code'))}.png"
        
        print(f"[{index + 1:4d}/{total}] {player_name[:25]:<25}", end="", flush=True)
        
        # Current ongoing season entry
        rows.append({
            "player_id": str(fpl_id),
            "player": player_name,
            "season": CURRENT_ONGOING_SEASON,
            "team": team_name,
            "position": position,
            "fpl_minutes": safe_float(element.get("minutes")),
            "fpl_appearances": safe_float(element.get("starts")), 
            "fpl_starts": safe_float(element.get("starts")),
            "fpl_goals": safe_float(element.get("goals_scored")),
            "fpl_assists": safe_float(element.get("assists")),
            "fpl_price_gbp": fpl_price * 1_000_000, 
            "photo_url": photo_url,
            "is_completed_season": False
        })
        
        # Fetch historical past seasons
        url = ELEMENT_SUMMARY_ENDPOINT.replace("_player_id_placeholder", str(fpl_id))
        try:
            summary = fetch_json(session, url)
            history = summary.get("history_past", [])
            print(f" | past seasons: {len(history)}")
            
            for past in history:
                season_name = format_season(past.get("season_name", ""))
                if not season_name:
                    continue
                    
                rows.append({
                    "player_id": str(fpl_id),
                    "player": player_name,
                    "season": season_name,
                    "team": "Unknown", 
                    "position": position, 
                    "fpl_minutes": safe_float(past.get("minutes")),
                    "fpl_appearances": safe_float(past.get("starts")), 
                    "fpl_starts": safe_float(past.get("starts")),
                    "fpl_goals": safe_float(past.get("goals_scored")),
                    "fpl_assists": safe_float(past.get("assists")),
                    "fpl_price_gbp": None,
                    "photo_url": photo_url,
                    "is_completed_season": is_season_completed(season_name)
                })
        except requests.RequestException as exc:
            print(f" | API error: {exc}")
            
        polite_sleep()
        
    df = pd.DataFrame(rows)
    return df

def validate_output(df: pd.DataFrame) -> None:
    print("\n" + "=" * 64)
    print("STATISTICS DIAGNOSTICS & VALIDATION")
    print("=" * 64)
    print(f"Total player-season records: {len(df):,}")
    
    print("\nBreakdown by Season:")
    for season, group in df.groupby("season"):
        completed_tag = "Completed" if is_season_completed(season) else "In Progress"
        print(f"  {season:10s} ({completed_tag:11s}): {len(group):4d} records")

    if len(df) == 0:
        raise RuntimeError("\nNO PLAYER STATISTICS WERE COLLECTED.")

def save_output(df: pd.DataFrame) -> None:
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\nSaved dataset to:\n{OUTPUT_FILE}")

def main() -> None:
    print("=" * 64)
    print("PREMIER LEAGUE PLAYER DATA COLLECTOR")
    print("=" * 64)
    result = collect_data()
    validate_output(result)
    save_output(result)
    print("\nNext step:\n  python build_market_values.py")

if __name__ == "__main__":
    main()