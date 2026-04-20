"""
Unified feature store.

Merges the weekly ETF proxy matrix, FRED macro features, and conviction
scores into a single parquet-backed feature table.  The TFT training loop
reads from here.

Schema (wide format, one row per week_ending):
  - week_ending: datetime index
  - {ETF}_vwap, {ETF}_flow_z, {ETF}_vol, {ETF}_mom4w, {ETF}_mom13w
  - {FRED_series}: fed_funds_rate, unemployment, ...
  - conviction_score_ema, thesis_clarity_ema, ...
  - cik: str (for multi-fund training; fund-level static metadata)
  - prior_weight_{cusip}: float (previous quarter 13F weight per position)
  - target_weight_{cusip}: float (next quarter 13F weight — training target)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from features.aggregator import (
    build_etf_feature_matrix,
    add_momentum_features,
    apply_ema_to_conviction,
)


def load_feature_store() -> pd.DataFrame:
    if config.FEATURE_STORE_PATH.exists():
        return pd.read_parquet(config.FEATURE_STORE_PATH)
    return pd.DataFrame()


def save_feature_store(df: pd.DataFrame) -> None:
    df.to_parquet(config.FEATURE_STORE_PATH)
    print(f"[feature_store] saved {len(df):,} rows → {config.FEATURE_STORE_PATH}")


def build_feature_store(
    ohlcv: pd.DataFrame,
    etf_flows: pd.DataFrame,
    macro: pd.DataFrame,
    conviction_scores: list[dict],
    holdings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full rebuild of the feature store from raw inputs.
    Typically called after initial data ingestion or a quarterly reset.
    """
    # ETF proxy features
    etf_matrix = build_etf_feature_matrix(ohlcv, etf_flows)
    etf_matrix = add_momentum_features(etf_matrix)

    # FRED macro (already weekly-indexed)
    macro.index = pd.to_datetime(macro.index)
    macro.index.name = "week_ending"

    # Conviction EMA
    conviction_df = apply_ema_to_conviction(conviction_scores)
    if not conviction_df.empty:
        conviction_df = conviction_df.set_index("week_date")
        conviction_df.index.name = "week_ending"
        # Keep only EMA columns
        conviction_df = conviction_df[[c for c in conviction_df.columns if c.endswith("_ema")]]

    # Merge on week_ending
    combined = etf_matrix.join(macro, how="outer").ffill()
    if not conviction_df.empty:
        combined = combined.join(conviction_df, how="left").ffill()

    # Attach prior quarter 13F weights as static known inputs
    if not holdings.empty:
        combined = _attach_13f_weights(combined, holdings)

    combined = combined.dropna(how="all").sort_index()
    return combined


_TOP_N_POSITIONS = 20  # limit column count to top holdings by average value


def _attach_13f_weights(
    feature_matrix: pd.DataFrame,
    holdings: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each week, attach:
      - prior_weight_{cusip}: weights from the most recently *filed* 13F on or before that week
      - target_weight_{cusip}: weights from the next filing (training label; NaN for latest period)
    Uses filing_date (not report period) so the feature matrix only sees information
    that was publicly available at each point in time.
    """
    if holdings.empty or "filing_date" not in holdings.columns:
        return feature_matrix

    top_cusips = (
        holdings.groupby("cusip")["value_thousands"]
        .mean()
        .nlargest(_TOP_N_POSITIONS)
        .index.tolist()
    )
    sub = holdings[holdings["cusip"].isin(top_cusips)].copy()
    sub["filing_date"] = pd.to_datetime(sub["filing_date"])

    pivot = (
        sub.pivot_table(index="filing_date", columns="cusip", values="weight", aggfunc="first")
        .fillna(0.0)
        .sort_index()
    )
    pivot.columns = [str(c) for c in pivot.columns]
    cusip_cols = pivot.columns.tolist()

    weeks_df = (
        feature_matrix.index
        .to_frame(name="week_ending")
        .reset_index(drop=True)
        .sort_values("week_ending")
    )
    pivot_reset = pivot.reset_index().rename(columns={"filing_date": "week_ending"})

    prior = pd.merge_asof(weeks_df, pivot_reset, on="week_ending", direction="backward")
    prior = prior.set_index("week_ending")
    for c in cusip_cols:
        if c in prior.columns:
            feature_matrix[f"prior_weight_{c}"] = prior[c].reindex(feature_matrix.index).values

    target = pd.merge_asof(weeks_df, pivot_reset, on="week_ending", direction="forward")
    target = target.set_index("week_ending")
    for c in cusip_cols:
        if c in target.columns:
            feature_matrix[f"target_weight_{c}"] = target[c].reindex(feature_matrix.index).values

    return feature_matrix


def get_latest_13f_period(holdings_path: Path | None = None) -> str | None:
    """Return the max report period from the local 13F parquet, or None if absent."""
    path = holdings_path or (Path(__file__).parent.parent / "data" / "13f_holdings.parquet")
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["period"])
    if df.empty:
        return None
    return str(pd.to_datetime(df["period"]).max().date())


def append_weekly_features(
    new_ohlcv: pd.DataFrame,
    new_macro: pd.DataFrame,
    new_conviction: dict,
) -> pd.DataFrame:
    """
    Incremental update: append one week of new features to the store.
    Called every Saturday evening.
    """
    existing = load_feature_store()

    etf_matrix = build_etf_feature_matrix(new_ohlcv, new_ohlcv)
    etf_matrix = add_momentum_features(etf_matrix)

    new_macro.index = pd.to_datetime(new_macro.index)
    new_macro.index.name = "week_ending"
    week_row = etf_matrix.join(new_macro, how="left")

    # Attach conviction
    if new_conviction:
        for k, v in new_conviction.items():
            if isinstance(v, (int, float)):
                week_row[k] = v

    if existing.empty:
        updated = week_row
    else:
        updated = pd.concat([existing, week_row])
        updated = updated[~updated.index.duplicated(keep="last")]
        updated = updated.sort_index()

    save_feature_store(updated)
    return updated
