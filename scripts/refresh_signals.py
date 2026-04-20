"""
Pre-compute ML signals and write to data/signals_cache.json.

Run after every ingestion (or whenever new 13F filings land):
    python scripts/refresh_signals.py

The cache is committed to the repo so Render serves it instantly on cold start
without re-training. The /stock_signals endpoint falls back to this file first.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import config
from model.stock_predictor import backtest_last_n_quarters, generate_signals

_CACHE_PATH = config.DATA_DIR / "signals_cache.json"


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def refresh(top_n: int = 20, force: bool = False) -> None:
    holdings = _load_parquet(config.DATA_DIR / "13f_holdings.parquet")
    if holdings.empty:
        print("[refresh_signals] No holdings data — run ingest.py first.")
        return

    feature_store = _load_parquet(config.DATA_DIR / "features.parquet")

    latest_period = str(holdings["period"].max())[:10]

    # Check if cache is already up to date
    if not force and _CACHE_PATH.exists():
        try:
            cached = json.loads(_CACHE_PATH.read_text())
            if cached.get("latest_period") == latest_period:
                print(f"[refresh_signals] Cache already current for period {latest_period}. Use --force to override.")
                return
        except Exception:
            pass

    print("[refresh_signals] Running required 4-quarter walk-forward backtest …")
    backtest = backtest_last_n_quarters(
        holdings=holdings,
        feature_store=feature_store,
        n_quarters=4,
    )
    if backtest.get("status") != "ok" or int(backtest.get("n_test_periods", 0)) < 4:
        raise RuntimeError(
            "Backtest requirement failed: expected 4 valid test quarters before signal generation."
        )

    summary = backtest.get("summary", {})
    print(
        "[refresh_signals] Backtest OK "
        f"(quarters={backtest.get('n_test_periods')}, "
        f"avg_mae_pp={summary.get('avg_mae_pp')}, "
        f"avg_dir_acc={summary.get('avg_directional_accuracy')})"
    )

    print(f"[refresh_signals] Training on period {latest_period} …")
    result = generate_signals(
        cik="",
        holdings=holdings,
        feature_store=feature_store,
        top_n=top_n,
        include_candidates=True,
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "latest_period": latest_period,
        "backtest": backtest,
        **result,
    }
    _CACHE_PATH.write_text(json.dumps(payload, indent=2))
    print(f"[refresh_signals] Wrote {len(result['data'])} signals → {_CACHE_PATH}")


if __name__ == "__main__":
    force = "--force" in sys.argv
    refresh(top_n=20, force=force)
