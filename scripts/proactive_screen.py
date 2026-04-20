"""
Proactive screening pipeline for net-new stock ideas.

Loop:
1) Build investable universe from sector ETFs.
2) Run quantitative tripwire (z-score anomalies) across that universe.
3) Produce candidate payload suitable for TFT zero-weight injection.

Run:
    python scripts/proactive_screen.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from model.stock_predictor import get_ticker

OUT_PATH = config.DATA_DIR / "proactive_candidates.json"


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _fetch_etf_constituents(etf: str) -> set[str]:
    """
    Best-effort ETF constituents using yfinance metadata.
    Falls back gracefully if holdings are unavailable.
    """
    out: set[str] = set()
    try:
        t = yf.Ticker(etf)
    except Exception:
        return out

    # Try modern funds_data.top_holdings path first.
    try:
        fd = getattr(t, "funds_data", None)
        top = getattr(fd, "top_holdings", None)
        if isinstance(top, pd.DataFrame) and not top.empty:
            symbol_col = "symbol" if "symbol" in top.columns else ("holding" if "holding" in top.columns else None)
            if symbol_col:
                out |= {str(s).strip().upper() for s in top[symbol_col].dropna().tolist() if str(s).strip()}
    except Exception:
        pass

    # Try info.holdings fallback.
    try:
        info = t.info if hasattr(t, "info") else {}
        holdings = info.get("holdings", []) if isinstance(info, dict) else []
        if isinstance(holdings, list):
            for h in holdings:
                if isinstance(h, dict):
                    sym = str(h.get("symbol", "")).strip().upper()
                    if sym:
                        out.add(sym)
    except Exception:
        pass

    return {s for s in out if 1 <= len(s) <= 6}


def build_investable_universe(etfs: Iterable[str], max_size: int = 350) -> list[str]:
    symbols: set[str] = set()
    for etf in etfs:
        symbols |= _fetch_etf_constituents(str(etf).strip().upper())

    # Fallback: if ETF constituents are unavailable, use mapped 13F tickers from latest holdings.
    if not symbols:
        p = config.DATA_DIR / "13f_holdings.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df = df[df.get("put_call", "").fillna("") == ""]
            latest = df[df["period"] == df["period"].max()].copy()
            mapped = latest.apply(
                lambda r: get_ticker(str(r.get("name", "")), str(r.get("cusip", ""))) or "",
                axis=1,
            )
            symbols = {str(s).strip().upper() for s in mapped.tolist() if str(s).strip()}

    return sorted(symbols)[:max_size]


def _compute_volume_zscores(universe: list[str], lookback: str = "6mo") -> dict[str, float]:
    if not universe:
        return {}

    try:
        data = yf.download(
            universe,
            period=lookback,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception:
        return {}

    if data is None or data.empty:
        return {}

    out: dict[str, float] = {}

    def _series_for(sym: str) -> pd.Series:
        if isinstance(data.columns, pd.MultiIndex):
            if sym in data.columns.get_level_values(0) and (sym, "Volume") in data.columns:
                return pd.to_numeric(data[(sym, "Volume")], errors="coerce").dropna()
            return pd.Series(dtype=float)
        if "Volume" in data.columns and len(universe) == 1:
            return pd.to_numeric(data["Volume"], errors="coerce").dropna()
        return pd.Series(dtype=float)

    for sym in universe:
        s = _series_for(sym)
        if len(s) < 25:
            continue
        tail = s.tail(20)
        mu = float(tail.mean())
        sd = float(tail.std(ddof=0))
        if sd <= 0:
            out[sym] = 0.0
        else:
            out[sym] = (float(s.iloc[-1]) - mu) / sd

    return out


def run_tripwire(universe: list[str], z_thresh: float, max_candidates: int) -> list[dict]:
    vol_z = _compute_volume_zscores(universe)
    if not vol_z:
        return []

    # Options/dark-pool placeholders for future OpenBB integration.
    options_z = {s: 0.0 for s in vol_z}
    dark_pool_z = {s: 0.0 for s in vol_z}

    rows = []
    all_rows = []
    for sym in vol_z:
        score = max(_safe_float(vol_z.get(sym)), _safe_float(options_z.get(sym)), _safe_float(dark_pool_z.get(sym)))
        row = {
            "ticker": sym,
            "tripwire_score_z": round(score, 4),
            "volume_z": round(_safe_float(vol_z.get(sym)), 4),
            "options_z": round(_safe_float(options_z.get(sym)), 4),
            "dark_pool_z": round(_safe_float(dark_pool_z.get(sym)), 4),
        }
        all_rows.append(row)
        if score >= z_thresh:
            rows.append(row)

    rows.sort(key=lambda r: -r["tripwire_score_z"])
    if not rows:
        # Quiet-week fallback: keep strongest positive anomalies so pipeline keeps learning.
        rows = sorted([r for r in all_rows if r["tripwire_score_z"] > 1.5], key=lambda r: -r["tripwire_score_z"])
    return rows[:max_candidates]


def to_tft_injection_payload(candidates: list[dict]) -> list[dict]:
    """
    Output shape for weekly model injection.
    Last 13F Wt is forced to 0 by design for net-new ideas.
    Thesis conviction is left neutral until targeted GraphRAG scoring is attached.
    """
    out = []
    for c in candidates:
        out.append(
            {
                "ticker": c["ticker"],
                "last_13f_wt_pct": 0.0,
                "flow_signal_z": c["tripwire_score_z"],
                "thesis_conviction": 0.5,
                "source": "proactive_tripwire",
            }
        )
    return out


def run() -> dict:
    universe = build_investable_universe(config.SCREEN_UNIVERSE_ETFS)
    candidates = run_tripwire(
        universe=universe,
        z_thresh=config.SCREEN_TRIPWIRE_Z,
        max_candidates=config.SCREEN_MAX_CANDIDATES,
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe_etfs": config.SCREEN_UNIVERSE_ETFS,
        "universe_size": len(universe),
        "tripwire_threshold_z": config.SCREEN_TRIPWIRE_Z,
        "candidates": candidates,
        "tft_injection": to_tft_injection_payload(candidates),
        "notes": [
            "Attach targeted 8-K + transcript GraphRAG scoring for candidates before training refresh.",
            "OpenBB options/dark-pool z-scores are placeholder 0.0 until a data provider is connected.",
        ],
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = run()
    print(
        f"[proactive_screen] universe={result['universe_size']} "
        f"candidates={len(result['candidates'])} -> {OUT_PATH}"
    )
