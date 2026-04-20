"""
Step 3: Temporal Aggregation & Feature Engineering.

Converts daily market proxy data → weekly VWAP + net cumulative flows.
Applies EMA smoothing to GraphRAG conviction scores.
Merges with FRED macro features to produce the unified weekly feature matrix.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config


def compute_weekly_vwap(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weekly VWAP per symbol from daily OHLCV data.
    VWAP = sum(typical_price * volume) / sum(volume)
    where typical_price = (high + low + close) / 3
    """
    df = ohlcv.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["dollar_volume"] = df["typical_price"] * df["volume"]

    weekly = (
        df.groupby(["symbol", pd.Grouper(key="date", freq="W-FRI")])
        .agg(
            vwap=("dollar_volume", lambda x: x.sum() / df.loc[x.index, "volume"].sum()),
            volume_sum=("volume", "sum"),
            close_last=("close", "last"),
            high_max=("high", "max"),
            low_min=("low", "min"),
        )
        .reset_index()
    )
    weekly.rename(columns={"date": "week_ending"}, inplace=True)
    return weekly


def compute_net_flows(etf_flows: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weekly net cumulative dollar flows per ETF.
    Uses volume * close as a dollar flow proxy, then sums weekly.
    """
    df = ohlcv.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dollar_flow"] = df["close"] * df["volume"]

    # Net flow = sum of daily dollar volume (positive = inflows proxy)
    weekly = (
        df.groupby(["symbol", pd.Grouper(key="date", freq="W-FRI")])
        .agg(net_dollar_flow=("dollar_flow", "sum"))
        .reset_index()
    )
    weekly.rename(columns={"date": "week_ending"}, inplace=True)

    # Normalize by rolling 4-week mean to create a z-score
    weekly = weekly.sort_values(["symbol", "week_ending"])
    weekly["flow_zscore"] = (
        weekly.groupby("symbol")["net_dollar_flow"]
        .transform(lambda s: (s - s.rolling(4, min_periods=1).mean()) / (s.rolling(4, min_periods=1).std() + 1e-9))
    )
    return weekly


def apply_ema_to_conviction(
    scores: list[dict],
    span: int | None = None,
) -> pd.DataFrame:
    """
    Convert a list of weekly conviction score dicts into a DataFrame and
    apply an EMA with the configured span so narrative shifts decay smoothly.
    """
    span = span or config.CONVICTION_EMA_SPAN
    df = pd.DataFrame(scores)
    if df.empty:
        return df

    df["week_date"] = pd.to_datetime(df["week_date"])
    df = df.sort_values("week_date")

    numeric_cols = [c for c in df.columns if c not in ("week_date", "rationale")]
    for col in numeric_cols:
        df[f"{col}_ema"] = df[col].ewm(span=span, adjust=False).mean()

    return df


def build_etf_feature_matrix(
    ohlcv: pd.DataFrame,
    etf_flows: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pivot the weekly VWAP and flow features into a wide matrix where each
    column is {symbol}_{feature}, indexed by week_ending date.
    """
    vwap_df = compute_weekly_vwap(ohlcv)
    flow_df = compute_net_flows(etf_flows, ohlcv)

    vwap_pivot = vwap_df.pivot(index="week_ending", columns="symbol", values="vwap")
    vwap_pivot.columns = [f"{c}_vwap" for c in vwap_pivot.columns]

    flow_pivot = flow_df.pivot(index="week_ending", columns="symbol", values="flow_zscore")
    flow_pivot.columns = [f"{c}_flow_z" for c in flow_pivot.columns]

    volume_pivot = vwap_df.pivot(index="week_ending", columns="symbol", values="volume_sum")
    volume_pivot.columns = [f"{c}_vol" for c in volume_pivot.columns]

    combined = pd.concat([vwap_pivot, flow_pivot, volume_pivot], axis=1)
    combined.index.name = "week_ending"
    return combined


def add_momentum_features(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Add 4-week and 13-week price momentum for each ETF VWAP column.
    """
    df = feature_matrix.copy()
    vwap_cols = [c for c in df.columns if c.endswith("_vwap")]
    for col in vwap_cols:
        sym = col.replace("_vwap", "")
        df[f"{sym}_mom4w"] = df[col].pct_change(4)
        df[f"{sym}_mom13w"] = df[col].pct_change(13)
    return df
