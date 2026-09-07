# Premier League Transfer Market Value Predictor

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)](tests/)

An end-to-end Machine Learning pipeline and interactive CLI application that estimates the transfer market value of Premier League players. Built with Scikit-learn's `LinearRegression` using target log-transformation, trained on historical player-season match statistics, and validated against official Transfermarkt valuations.

---

## 📖 Table of Contents
- [Overview](#overview)
- [System Architecture & Documentation](#system-architecture--documentation)
- [Project Directory Structure](#project-directory-structure)
- [Installation & Setup](#installation--setup)
- [How to Run the Complete Pipeline](#how-to-run-the-complete-pipeline)
- [Machine Learning Modeling & Performance](#machine-learning-modeling--performance)
- [Interactive CLI Application](#interactive-cli-application)
- [Running Automated Tests](#running-automated-tests)
- [Limitations & Ethical Considerations](#limitations--ethical-considerations)

---

## Overview

In professional football, transfer valuations are often swayed by hype, speculation, and media narratives. This project builds a transparent, reproducible, and interpretable statistical baseline to determine what a player's raw on-pitch output is worth on the open market.

### Key Capabilities:
- **Automated Data Harvesting:** Ingests live player rosters, positions, and high-resolution official photos directly from the Official Fantasy Premier League (FPL) API.
- **Entity Resolution:** Bridges player identity differences between FPL and Transfermarkt using a 4-stage Unicode normalisation, token subset, and particle-stripping algorithm.
- **All-Competition Aggregation:** Queries match appearance logs from a local columnar DuckDB database to incorporate all domestic and European cup performances.
- **Log-Transformed Regression:** Uses `sklearn.compose.TransformedTargetRegressor` with `np.log1p` to account for the heavy right-skew of elite player valuations.
- **Instant Interactive Inference:** Allows command-line player lookup with interactive disambiguation, photographic headshots, and over/under-valuation indicators.

---

## System Architecture & Documentation

For in-depth explanations tailored to different audiences:
- 📘 **[architecture.md](file:///home/zrex/Documents/Transfer%20Value%20Predicter/architecture.md)** — **For Non-Technical & Analytical Readers**: Understand the football scouting analogy, tech stack, data journey, and model intuition in plain English.
- 🛠️ **[production.md](file:///home/zrex/Documents/Transfer%20Value%20Predicter/production.md)** — **For Software Engineers & MLOps**: FastAPI REST microservice design, containerization, Redis caching, drift detection, and monitoring.
- 💡 **[IDEA.md](file:///home/zrex/Documents/Transfer%20Value%20Predicter/IDEA.md)** — Original concept proposal and core design requirements.

---

## Project Directory Structure

```text
Transfer Value Predicter/
├── architecture.md                     # System architecture & non-technical guide
├── production.md                       # Technical MLOps & deployment blueprint
├── README.md                           # Project documentation and quickstart
├── IDEA.md                             # Original concept & requirements
├── AGENTS.md                           # Project specifications & rules
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Git exclusions (DuckDB, caches, data, models)
├── transfermarkt-datasets.duckdb       # Transfermarkt database (player valuations & match logs)
│
├── collect_data.py                     # Step 1: Harvest FPL rosters, stats & photo URLs
├── build_market_values.py              # Step 2: Entity matching & valuation data synthesis
├── train_model.py                      # Step 3: Train & serialize log-linear regression pipeline
├── main.py                             # Step 4: Interactive search & prediction CLI app
│
├── data/
│   ├── raw/
│   │   ├── seasonal_player_stats.csv   # Ingested player season statistics
│   │   └── player_market_values.csv    # Cross-matched Transfermarkt valuations
│   └── processed/                      # Transformed datasets (if needed)
│
├── models/
│   ├── transfer_value_model.joblib     # Serialized trained Scikit-learn Pipeline
│   ├── transfer_value_model.pkl        # Pickle backup of trained Pipeline
│   ├── model_metadata.json             # Model training parameters & evaluation metrics
│   └── training_data.csv               # Processed dataset snapshot used during training
│
├── src/                                # Modular source packages
│   ├── data_collection/
│   ├── preprocessing/
│   ├── features/
│   ├── modeling/
│   ├── prediction/
│   ├── visualization/
│   └── utils/
│
└── tests/
    └── test_basic.py                   # Automated tests verifying structure & packages
```

---

## Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Linux, macOS, or Windows WSL

### 1. Clone & Navigate to Repository
```bash
cd "Transfer Value Predicter"
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate       # On Linux / macOS
# or: venv\Scripts\activate    # On Windows
```

### 3. Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## How to Run the Complete Pipeline

The application can be executed from scratch or run directly using the pre-trained model:

### Step 1: Harvest Player Roster & Statistics
Collects current Premier League players, historical season stats, and headshots from the FPL API:
```bash
python collect_data.py
```
*Outputs: `data/raw/seasonal_player_stats.csv`*

### Step 2: Cross-Reference Transfer Valuations
Resolves player names against the Transfermarkt DuckDB database and calculates season-end valuations in GBP:
```bash
python build_market_values.py
```
*Outputs: `data/raw/player_market_values.csv`*

### Step 3: Train the Machine Learning Model
Fits the log-linear pipeline on completed historical player-seasons and saves weights:
```bash
python train_model.py
```
*Outputs: `models/transfer_value_model.joblib` and `models/model_metadata.json`*

### Step 4: Run the Prediction CLI
Run interactively:
```bash
python main.py
```
Or query a player directly via command-line arguments:
```bash
python main.py "Erling Haaland"
python main.py "Bruno Fernandes"
python main.py "Bukayo Saka"
```

---

## Machine Learning Modeling & Performance

### Feature Architecture
- **Numeric Features (Median Imputed):**
  - `total_minutes`: Total minutes played across all competitions
  - `total_appearances`: Total match appearances
  - `total_goals`: Total goals scored
  - `total_assists`: Total assists provided
- **Categorical Features (One-Hot Encoded):**
  - `position`: Goalkeeper, Defender, Midfielder, Forward
  - `team`: Club affiliation / club prestige premium
- **Target Variable:**
  - `market_value_gbp`: Log-transformed using `np.log1p` during training and inverted via `np.expm1` during prediction.

### Evaluation Metrics (Test Set, 20% Holdout)
| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Model Type** | `LinearRegression (log1p)` | Interpretable, strictly positive valuations |
| **Training Records** | `1,788` player-seasons | Historical completed seasons |
| **Mean Absolute Error (MAE)** | **£8,748,025** | Average error margin across all player tiers |
| **Root Mean Squared Error (RMSE)** | **£13,742,633** | Standard deviation of valuation residuals |
| **Coefficient of Determination ($R^2$)** | **0.4378** | Explains ~44% of valuation variance using basic stats |

---

## Interactive CLI Application

When querying a player, the application produces a detailed valuation dossier:

```text
================================================================
PLAYER PROFILE
================================================================
Name:                Erling Haaland
Team:                Manchester City
Position:            Forward
Season:              2024-25 (Latest Completed Season)

All-Competition Performance:
  Total Appearances: 45
  Total Minutes:     3,745
  Total Goals:       38
  Total Assists:     6

Photo:               https://resources.premierleague.com/premierleague/photos/players/110x140/p223094.png

----------------------------------------------------------------
TRANSFER VALUE
----------------------------------------------------------------
Actual Transfermarkt value: £154.80M
Predicted market value:     £148.20M

----------------------------------------------------------------
DIFFERENCE
----------------------------------------------------------------
Difference:                 -£6.60M
Percentage difference:      -4.3%

The model is UNDERESTIMATING the player's value.
```

---

## Running Automated Tests

Run the automated test suite using `pytest`:
```bash
python -m pytest -v
```

Tests verify:
- Existence of all required directories (`data/`, `models/`, `src/`)
- Presence of entrypoint execution scripts
- Dependency consistency across `requirements.txt`

---

## Limitations & Ethical Considerations

1. **Contextual Drivers:** The model does not currently ingest contract length, buyout clauses, player age, injury history, or commercial marketing appeal.
2. **Positional Variance:** Defenders and Goalkeepers have fewer counting stats (goals/assists) compared to forwards; clean sheets and defensive actions will be added in future iterations.
3. **Educational Intent:** This system is an analytical exploration tool designed to evaluate statistical efficiency, not financial or legal advice for football transfers.