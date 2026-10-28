# 13F Analyzer

**Institutional positioning from 13F filings: what twelve books hold, what they moved, and what they are likely to open next.**

Quarterly 13F filings pulled straight from SEC EDGAR, parsed into position
histories, and scored for conviction. The dashboard reads what changed, what the
model expects next quarter, and which names held by peer funds it thinks are
about to appear. FRED macro series sit alongside as context, so a position change
is read against the rate environment it was made in.

![status](https://img.shields.io/badge/status-active-1F2328)
![python](https://img.shields.io/badge/python-3.10%2B-1F2328)
![next.js](https://img.shields.io/badge/next.js-14-1F2328)
[![license](https://img.shields.io/badge/license-MIT-6B7280)](LICENSE)

---

## What it does

- **Live 13F ingestion** — straight from the SEC EDGAR submissions API, no
  aggregator in between, so the numbers are the ones that were filed.
- **Twelve tracked institutions** — Bridgewater, Citadel, Renaissance, Pershing
  Square, Two Sigma (both entities), D. E. Shaw, AQR, Cubist, Point72, PDT and
  Situational Awareness. Override with `TARGET_CIKS`.
- **Conviction scores** — a gradient-boosted regressor over four-quarter position
  transitions and price momentum, predicting next-quarter weight change in
  percentage points.
- **A sequence model for the whole book** — a Temporal Fusion Transformer over
  static fund covariates, past weights, weekly market proxy flows and FRED macro,
  predicting the full weight distribution rather than one position at a time.
- **Position history** — a four-filing weight histogram for any holding.
- **Candidate new buys** — names held by peer funds that the model expects to
  turn up in a given book next quarter.
- **Macro context** — Fed funds, unemployment, CPI, 2s, 10s and the 2s10s spread.
- **Sector proxies** — twelve sector ETFs for relative-strength context.

## Why a simplex projection

A portfolio is a point on the probability simplex: weights are non-negative and
sum to one. A softmax head respects both constraints but can never produce a true
zero, so it cannot represent a position being *exited* — only shrunk to an
infinitesimal residual.

So the TFT's output is projected onto the simplex after each backward pass, by
the exact O(n log n) algorithm of Duchi et al. (2008). The projection produces
genuine zeros, which is what lets the model learn exits as exits.

---

## Architecture

```
SEC EDGAR ──▶ edgar_client.py    ─┐
yfinance  ──▶ market_client.py   ─┼──▶ features/ ──▶ model/ ──▶ API ──▶ Frontend
FRED      ──▶ fred_client.py     ─┘
```

| Stage | Module | |
|---|---|---|
| Ingestion | `data/ingestion/` | 13F XML parsing, ETF OHLCV, FRED macro |
| Features | `features/` | Weekly VWAP, net flows, EMA conviction → parquet store |
| Models | `model/stock_predictor.py` | `HistGradientBoostingRegressor` on 13F transitions + price data |
| | `model/tft_model.py` | Temporal Fusion Transformer over the full book |
| | `model/simplex_projector.py` | Exact projection onto the probability simplex |
| API | `api_backend/main.py` | FastAPI, all data endpoints |
| Frontend | `frontend/` | Next.js 14 dashboard |

---

## Quick Start

```bash
git clone https://github.com/nishadw/13f-analyzer.git
cd 13f-analyzer

cp .env.example .env
# Edit .env — set EDGAR_USER_AGENT, FRED_API_KEY, GROQ_API_KEY

./run.sh --bootstrap        # installs deps, then starts both servers
open http://localhost:3001
```

On subsequent runs, `./run.sh` alone.

## Environment

| Variable | Required | |
|---|---|---|
| `EDGAR_USER_AGENT` | yes | Your name and email — the SEC requires it, e.g. `Jane Doe jane@example.com` |
| `FRED_API_KEY` | yes | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `GROQ_API_KEY` | yes | [console.groq.com](https://console.groq.com) |
| `TARGET_CIKS` | no | Comma-separated CIKs; defaults to the twelve above |
| `PROXY_ETFS` | no | Comma-separated tickers; defaults to twelve sector ETFs |

## Scripts

| Command | |
|---|---|
| `./run.sh` | Start backend and frontend |
| `./run.sh --bootstrap` | Install dependencies first |
| `./run.sh --ingest` | Re-ingest all 13F data before starting |
| `./run.sh --train` | Re-train before starting |
| `python scripts/ingest.py` | Bootstrap ingestion, run once |
| `python scripts/serve.py --reload` | Backend only, for development |

## Data sources

- **SEC EDGAR** — `data.sec.gov/submissions`, public domain, no key
- **yfinance** — price history, no key
- **FRED** — macro series, free key

---

## A caveat worth stating plainly

13F filings are a lagged, long-only, US-equity-only picture, filed up to
forty-five days after quarter end. They do not show shorts, derivatives
positioning, or anything closed inside the quarter. A model fitted to them is a
model of what gets *disclosed*, which is not the same thing as what gets *held*.

Informational only. Not investment advice.

## License

MIT — see [LICENSE](LICENSE). Data from SEC EDGAR is public domain.
