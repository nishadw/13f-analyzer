# 13F Analyzer

**Institutional portfolio intelligence — SEC 13F filings + ML conviction signals.**

Track quarterly 13F filings from top hedge funds (Bridgewater, Citadel, Renaissance Technologies, Pershing Square, Situational Awareness), analyse position changes, and view ML-driven buy/sell conviction scores in a clean web dashboard.

![Dashboard](https://img.shields.io/badge/status-active-brightgreen) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Next.js](https://img.shields.io/badge/next.js-14-black) ![License](https://img.shields.io/badge/license-MIT-blue)

---

## Features

- **Live 13F ingestion** — pulls the latest filings directly from SEC EDGAR, including Q4 2025 data
- **ML conviction signals** — `HistGradientBoostingClassifier` trained on 4-quarter position transitions + yfinance price momentum, outputting a signal ∈ [−1, +1]
- **Position history** — 4-filing weight histogram for any individual holding
- **Macro context** — FRED economic indicators (Fed funds rate, VIX, yield spread, CPI)
- **Sector ETF proxies** — SMH, XLK, XLF, XLV, XLE, and 7 more
- **Candidate new-buys** — stocks held by peer funds that the model predicts as likely new positions

---

## Architecture

```
SEC EDGAR ──▶ edgar_client.py    ─┐
yfinance  ──▶ market_client.py   ─┼──▶ features/ ──▶ model/stock_predictor.py ──▶ API ──▶ Frontend
FRED      ──▶ fred_client.py     ─┘
```

| Stage | Module | Description |
|-------|--------|-------------|
| Ingestion | `data/ingestion/` | 13F XML parsing (SEC EDGAR), ETF OHLCV, FRED macro |
| Features | `features/` | Weekly VWAP, net flows, EMA conviction → parquet store |
| Model | `model/stock_predictor.py` | HistGBC on 13F transitions + price data |
| API | `api_backend/main.py` | FastAPI with HTTPS, all data endpoints |
| Frontend | `frontend/` | Next.js 14 dashboard |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/nishadw/13f-analyzer.git
cd 13f-analyzer

# 2. Configure
cp .env.example .env
# Edit .env — set EDGAR_USER_AGENT, FRED_API_KEY, GROQ_API_KEY

# 3. Run (installs deps + starts both servers)
./run.sh --bootstrap

# 4. Open
open http://localhost:3001
```

> On subsequent runs, just `./run.sh` — no `--bootstrap` needed.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `EDGAR_USER_AGENT` | Yes | Your name + email, e.g. `John Doe john@example.com` |
| `FRED_API_KEY` | Yes | From [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `GROQ_API_KEY` | Yes | From [console.groq.com](https://console.groq.com) |
| `TARGET_CIKS` | No | Comma-separated CIK list (defaults to 5 funds) |

---

## Scripts

| Command | Description |
|---------|-------------|
| `./run.sh` | Start backend + frontend |
| `./run.sh --bootstrap` | Install deps first, then start |
| `./run.sh --ingest` | Re-ingest all 13F data before starting |
| `./run.sh --train` | Re-train TFT model before starting |
| `python scripts/ingest.py` | Bootstrap data ingestion (run once) |
| `python scripts/serve.py --reload` | Backend only (dev) |

---

## Data Sources

- **SEC EDGAR** — official submissions API (`data.sec.gov/submissions`) — no key required
- **yfinance** — stock and ETF price history — no key required
- **FRED** — Federal Reserve macroeconomic data — free API key required

---

## License

MIT — see [LICENSE](LICENSE). Data from SEC EDGAR is public domain. This tool is for informational purposes only and is not investment advice.
