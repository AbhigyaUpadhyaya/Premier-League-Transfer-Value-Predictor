# ==========================================================
# train_model.py
# ==========================================================
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

# pyrefly: ignore [missing-import]
import joblib
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ============================================================
# PATHS & CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"

STATS_FILE = RAW_DIR / "seasonal_player_stats.csv"
MARKET_FILE = RAW_DIR / "player_market_values.csv"

MODEL_FILE = MODEL_DIR / "transfer_value_model.joblib"
PICKLE_MODEL_FILE = MODEL_DIR / "transfer_value_model.pkl"
METADATA_FILE = MODEL_DIR / "model_metadata.json"
TRAINING_FILE = MODEL_DIR / "training_data.csv"

NUMERIC_FEATURES = ["total_minutes", "total_appearances", "total_goals", "total_assists"]
CATEGORICAL_FEATURES = ["position", "team"]
TARGET = "market_value_gbp"
MIN_TRAINING_ROWS = 30

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

def parse_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df

def parse_player_id(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text

# ============================================================
# LOAD & MATCH DATA
# ============================================================

def load_data() -> pd.DataFrame:
    if not STATS_FILE.exists() or not MARKET_FILE.exists():
        raise FileNotFoundError("Raw datasets missing. Run collect_data.py and build_market_values.py first.")

    stats = pd.read_csv(STATS_FILE)
    market = pd.read_csv(MARKET_FILE)

    stats["player_id"] = stats["player_id"].apply(parse_player_id)
    market["player_id"] = market["player_id"].apply(parse_player_id)
    stats["season"] = stats["season"].astype(str).str.strip()
    market["season"] = market["season"].astype(str).str.strip()

    # STRICT FILTER: Exclude in-progress season AND require valid Transfermarkt target
    market[TARGET] = pd.to_numeric(market[TARGET], errors="coerce")
    market = market[
        market["season"].apply(is_season_completed) &
        market[TARGET].notna() &
        (market[TARGET] > 0)
    ].copy()

    market = market.sort_values(["player_id", "season"]).drop_duplicates(["player_id", "season"], keep="last")

    merged = stats[["player_id", "player", "season", "team", "position"]].merge(
        market,
        on=["player_id", "season"],
        how="inner",
    )
    
    # Backfill "Unknown" historical teams using actual Transfermarkt club
    merged["team"] = merged["team"].astype(str)
    mask = (merged["team"] == "Unknown") & merged["transfermarkt_club"].notna()
    merged.loc[mask, "team"] = merged.loc[mask, "transfermarkt_club"].astype(str)

    return merged

# ============================================================
# PREPARE TRAINING DATA
# ============================================================

def prepare_training_data(merged: pd.DataFrame) -> pd.DataFrame:
    training = merged.copy()
    training = parse_numeric(training, NUMERIC_FEATURES)

    # Ensure no training rows contain missing numerical performance metrics
    training = training.dropna(subset=NUMERIC_FEATURES)

    for column in CATEGORICAL_FEATURES:
        training[column] = training[column].fillna("Unknown").astype(str).str.strip()
        training.loc[training[column] == "", column] = "Unknown"

    print("\n" + "=" * 60)
    print("NUMERIC FEATURE COVERAGE")
    print("=" * 60)
    for column in NUMERIC_FEATURES:
        valid_count = training[column].notna().sum()
        print(f"{column:20s} valid={valid_count:,} / {len(training):,}")

    training.to_csv(TRAINING_FILE, index=False)
    return training

# ============================================================
# BUILD & TRAIN MODEL
# ============================================================

def build_model() -> Pipeline:
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0,
    )

    regressor = TransformedTargetRegressor(
        regressor=LinearRegression(),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])

def train(training: pd.DataFrame):
    X = training[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = training[TARGET]

    if len(training) < MIN_TRAINING_ROWS:
        raise RuntimeError(f"Not enough training records. Valid: {len(training):,} Required: {MIN_TRAINING_ROWS:,}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows:  {len(X_test):,}")

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"MAE:  £{mae:,.0f}")
    print(f"RMSE: £{rmse:,.0f}")
    print(f"R²:   {r2:.4f}")

    metrics = {
        "model_type": "LinearRegression (log1p target transform)",
        "target": TARGET,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "total_training_rows": int(len(training)),
        "mae_gbp": float(mae),
        "rmse_gbp": float(rmse),
        "r2": float(r2),
    }
    return model, metrics

def save_model(model, metrics):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(model, PICKLE_MODEL_FILE)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nModel saved successfully:\n  {MODEL_FILE}")

def main():
    print("=" * 60)
    print("TRANSFER VALUE MODEL TRAINING")
    print("=" * 60)

    training = load_data()
    print(f"\nMatched completed training rows with actual market value: {len(training):,}")

    training = prepare_training_data(training)
    model, metrics = train(training)
    save_model(model, metrics)

if __name__ == "__main__":
    main()