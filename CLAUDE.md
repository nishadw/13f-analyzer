# 13F Analyzer — Codebase Guide

## What this is
A production system for predicting institutional portfolio weights by fusing
SEC 13F filings (quarterly ground truth) with high-frequency market proxies,
macroeconomic indicators, and LLM-extracted narrative conviction scores.

## Architecture (5-stage pipeline)

| Stage | Module | Purpose |
|-------|--------|---------|
| 1 — Ingestion | `data/ingestion/` | SEC EDGAR 13F/8-K, yfinance ETF proxies, FRED macro |
| 2 — Narrative | `narrative/` | GraphRAG knowledge graph + Groq LLM-as-judge scorer |
| 3 — Features | `features/` | Weekly VWAP, net flows, EMA conviction → parquet store |
| 4 — Model | `model/tft_model.py` | Temporal Fusion Transformer forecasting |
| 5 — Constraints | `model/simplex_projector.py` | Projected Gradient Descent onto probability simplex |

Production orchestration: `production/weekly_loop.py`

## Quick start

```bash
cp .env.example .env          # fill in API keys
pip install -r requirements.txt
python scripts/ingest.py      # bootstrap data (run once)
python scripts/train.py       # train TFT (run once after ingestion)
python scripts/scheduler.py --now   # test full pipeline immediately
python scripts/scheduler.py         # start the weekly cron daemon

# API backend
python scripts/serve.py --reload    # dev server at http://localhost:7779
python scripts/serve.py             # production
```

## Key design decisions

**Why PGD over Softmax for portfolio constraints?**
Softmax cannot output true zeros (exited positions). The Duchi et al. O(n log n)
simplex projection in `model/simplex_projector.py` allows learned sparse weight
vectors while guaranteeing ∑w = 1 and w ≥ 0 after every backward pass.

**Why GraphRAG over standard RAG?**
Standard vector RAG does single-hop retrieval. GraphRAG builds entity/relationship
graphs enabling multi-hop, corpus-wide "Global Queries" for thematic summarisation
across weeks of earnings transcripts and policy news.

**Why TFT over RL/MDP?**
Predicting a 13F filing is an observational forecasting problem, not a sequential
decision problem. TFT natively handles static fund metadata, known past 13F weights,
and dynamic proxy flows with interpretable attention weights.

**Why weekly aggregation?**
Institutional funds execute over weeks. Daily frequency fits market-maker noise
rather than fund rebalancing signals. Weekly VWAP + net flows align with actual
execution timelines.

## API keys needed

| Key | Source | Required |
|-----|--------|----------|
| `FRED_API_KEY` | fred.stlouisfed.org | Yes |
| `GROQ_API_KEY` | console.groq.com | Yes |
| `EDGAR_USER_AGENT` | No key — just your name + email | Yes |

## File layout

```
13f-analyzer/
├── config.py                    # All env vars + hyperparameters
├── requirements.txt
├── .env.example
├── data/
│   ├── ingestion/
│   │   ├── edgar_client.py      # 13F XML parsing, rate-limited
│   │   ├── market_client.py     # ETF OHLCV + flow proxies via yfinance
│   │   └── fred_client.py       # Macro series → weekly DataFrame
│   └── pipeline.py              # Ingestion orchestrator
├── narrative/
│   ├── scraper.py               # 8-K exhibits + FOMC statement scraper
│   ├── graphrag_processor.py    # GraphRAG init, indexing, global query
│   └── conviction_scorer.py     # Groq LLM-as-judge with JSON rubric
├── features/
│   ├── aggregator.py            # VWAP, net flows, EMA conviction
│   └── feature_store.py         # Unified parquet feature store
├── model/
│   ├── tft_model.py             # TemporalFusionTransformer wrapper
│   ├── simplex_projector.py     # PGD + tracking error loss
│   └── trainer.py               # Training entry point
├── production/
│   └── weekly_loop.py           # 4-stage Saturday pipeline
└── scripts/
    ├── ingest.py                # Bootstrap ingestion
    ├── train.py                 # Model training
    └── scheduler.py             # Cron daemon
```
