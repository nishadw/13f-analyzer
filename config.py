from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent

# ── EDGAR ────────────────────────────────────────────────────────────────────
EDGAR_USER_AGENT: str = os.environ.get("EDGAR_USER_AGENT", "13f-analyzer contact@example.com")
EDGAR_BASE_URL = "https://data.sec.gov"
EDGAR_SUBMISSIONS_URL = f"{EDGAR_BASE_URL}/submissions"
EDGAR_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

# ── FRED ─────────────────────────────────────────────────────────────────────
FRED_API_KEY: str = os.environ.get("FRED_API_KEY", "")

# ── Market data provider (yfinance-only) ─────────────────────────────────────
MARKET_DATA_PROVIDER: str = "yfinance"

# ── Targets ──────────────────────────────────────────────────────────────────
# Tracked funds:
# 1350694 = Bridgewater Associates, 1423053 = Citadel,
# 1336528 = Pershing Square, 2045724 = Situational Awareness LP,
# 1037389 = Renaissance Technologies,
# 1179392 = Two Sigma Investments LP, 1478735 = Two Sigma Advisers LP,
# 1009207 = D. E. Shaw & Co., 1167557 = AQR Capital Management,
# 1603465 = Cubist Systematic Strategies, 1603466 = Point72 Asset Management,
# 1564702 = PDT Partners
TARGET_CIKS: list[str] = [
    c.strip() for c in os.environ.get(
        "TARGET_CIKS",
        "1350694,1423053,1336528,2045724,1037389,"
        "1179392,1478735,1009207,1167557,1603465,1603466,1564702",
    ).split(",")
]

PROXY_ETFS: list[str] = [
    e.strip() for e in os.environ.get(
        "PROXY_ETFS", "SMH,XLU,XLK,XLF,XLE,XLV,XLI,XLB,XLRE,XLC,XLY,XLP"
    ).split(",")
]

# ── FRED series for macro features ───────────────────────────────────────────
FRED_SERIES: dict[str, str] = {
    "fed_funds_rate": "FEDFUNDS",
    "unemployment": "UNRATE",
    "cpi_yoy": "CPIAUCSL",
    "treasury_10y": "DGS10",
    "treasury_2y": "DGS2",
    "yield_spread": "T10Y2Y",
    "vix": "VIXCLS",
    "credit_spread_hy": "BAMLH0A0HYM2",
    "m2_money_supply": "M2SL",
    "industrial_production": "INDPRO",
}

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "checkpoints"))
FEATURE_STORE_PATH = Path(os.environ.get("FEATURE_STORE_PATH", ROOT / "data" / "features.parquet"))
DATA_DIR = ROOT / "data"

for _p in (MODEL_DIR, FEATURE_STORE_PATH.parent, DATA_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ── TFT hyperparameters ───────────────────────────────────────────────────────
TFT_MAX_ENCODER_LENGTH = 52   # one year of weekly data as context
TFT_MAX_PREDICTION_LENGTH = 13  # one quarter ahead
TFT_BATCH_SIZE = 32
TFT_MAX_EPOCHS = 100
TFT_LEARNING_RATE = 3e-4
TFT_HIDDEN_SIZE = 128
TFT_ATTENTION_HEAD_SIZE = 4
TFT_DROPOUT = 0.1
TFT_HIDDEN_CONTINUOUS_SIZE = 64

# ── EMA decay for conviction scores ──────────────────────────────────────────
CONVICTION_EMA_SPAN = 4  # weeks
