"""
Market data client for ETF proxy features.

This module uses yfinance directly.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

import config


def fetch_etf_ohlcv(
    symbols: list[str],
    start: date,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """Fetch daily OHLCV for ETF symbols. Returns tidy DataFrame."""
    end = end or date.today()
    if not symbols:
        return pd.DataFrame()

    data = yf.download(
        symbols,
        start=str(start),
        end=str(end),
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if data is None or data.empty:
        return pd.DataFrame()

    frames = []

    # Multi-ticker shape: columns are MultiIndex(level0=ticker, level1=field)
    if isinstance(data.columns, pd.MultiIndex):
        for sym in symbols:
            if sym not in data.columns.get_level_values(0):
                continue
            sub = data[sym].copy().dropna(how="all")
            if sub.empty:
                continue
            sub = sub.rename(columns=str.lower)
            sub["symbol"] = sym
            sub.index.name = "date"
            frames.append(sub.reset_index())
    else:
        # Single ticker shape
        sub = data.copy().dropna(how="all")
        if not sub.empty:
            sub = sub.rename(columns=str.lower)
            sub["symbol"] = symbols[0]
            sub.index.name = "date"
            frames.append(sub.reset_index())

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_options_flow(
    symbols: list[str],
    start: date,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """Placeholder for options flow (not available in yfinance for this pipeline)."""
    _ = (symbols, start, end)
    return pd.DataFrame()


def fetch_dark_pool_prints(
    symbols: list[str],
    start: date,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """Placeholder for dark pool prints (not available in yfinance)."""
    _ = (symbols, start, end)
    return pd.DataFrame()


def fetch_etf_flows(
    symbols: list[str],
    start: date,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """
    ETF flow proxy from OHLCV using dollar volume.
    """
    ohlcv = fetch_etf_ohlcv(symbols, start, end)
    if ohlcv.empty:
        return ohlcv

    df = ohlcv.copy()
    if "close" in df.columns and "volume" in df.columns:
        df["dollar_flow"] = pd.to_numeric(df["close"], errors="coerce") * pd.to_numeric(df["volume"], errors="coerce")
    return df


def pull_weekly_proxy_data(weeks_back: int = 1) -> dict[str, pd.DataFrame]:
    """
    Pull weekly proxy data for all configured ETFs.
    Returns dict: ohlcv, options_flow, dark_pool, etf_flows.
    """
    end = date.today()
    start = end - timedelta(weeks=weeks_back + 1)
    symbols = config.PROXY_ETFS

    ohlcv = fetch_etf_ohlcv(symbols, start, end)
    return {
        "ohlcv": ohlcv,
        "options_flow": fetch_options_flow(symbols, start, end),
        "dark_pool": fetch_dark_pool_prints(symbols, start, end),
        "etf_flows": fetch_etf_flows(symbols, start, end) if not ohlcv.empty else pd.DataFrame(),
    }
