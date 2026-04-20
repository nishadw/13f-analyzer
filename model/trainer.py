"""
Training orchestrator.

Loads the feature store, classifies columns into TFT input groups,
synthesises a target column when no 13F weight targets exist yet,
then calls train_model.
"""
from __future__ import annotations

import pandas as pd
from rich.console import Console

import config
from features.feature_store import load_feature_store
from model.tft_model import train_model, PortfolioTFT

console = Console()

# Columns that are never features or targets
_META_COLS = {"week_ending", "date", "time_idx", "dummy_group"}


def _add_synthetic_target(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    When no 13F weight targets exist, create a synthetic target:
    the next-week return of the first available VWAP column.
    This gives the TFT something meaningful to minimise until real
    13F targets are available.
    """
    vwap_cols = [c for c in df.columns if c.endswith("_vwap")]
    if vwap_cols:
        col = vwap_cols[0]
        target_name = f"target_{col}_return"
        df[target_name] = df[col].pct_change().shift(-1)
    else:
        # Last resort: use the first FRED numeric series
        numeric = [c for c in df.columns if df[c].dtype == float and c not in _META_COLS]
        col = numeric[0]
        target_name = f"target_{col}_delta"
        df[target_name] = df[col].diff().shift(-1)

    df[target_name] = df[target_name].fillna(0.0)
    return df, target_name


def _classify_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Classify every column into one of four TFT input groups.
    Adds a synthetic target and a dummy group column if needed.
    Returns the (possibly augmented) DataFrame and the column groups dict.
    """
    # ── Add dummy group so TFT always has a group_ids column ──────────────────
    if "cik" not in df.columns:
        df = df.copy()
        df["dummy_group"] = "fund_0"

    group_col = "cik" if "cik" in df.columns else "dummy_group"

    all_cols = set(df.columns)

    # Explicit target columns (from 13F weight pivots)
    target_cols = [c for c in all_cols if c.startswith("target_weight_")]

    # If no real targets, synthesise one
    if not target_cols:
        df, synthetic = _add_synthetic_target(df)
        target_cols = [synthetic]
        console.print(f"  [yellow]No 13F weight targets — using synthetic target:[/yellow] {synthetic}")

    static_cols = [c for c in all_cols if c in ("cik", "dummy_group")]
    known_past  = [c for c in all_cols if c.startswith("prior_weight_")]

    exclude = set(target_cols) | set(static_cols) | set(known_past) | _META_COLS
    unknown = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_float_dtype(df[c])
    ]

    return df, {
        "target_cols":               target_cols,
        "static_cols":               static_cols,
        "time_varying_known_cols":   known_past,
        "time_varying_unknown_cols": unknown,
        "group_col":                 group_col,
    }


def run_training() -> PortfolioTFT:
    console.rule("[bold]TFT Training[/bold]")

    features = load_feature_store()
    if features.empty:
        raise ValueError("Feature store is empty. Run data ingestion first.")

    console.print(f"[cyan]Feature store[/cyan]: {features.shape[0]} weeks × {features.shape[1]} features")

    features, col_groups = _classify_columns(features)

    console.print(f"  targets:      {col_groups['target_cols']}")
    console.print(f"  static:       {len(col_groups['static_cols'])}")
    console.print(f"  known past:   {len(col_groups['time_varying_known_cols'])}")
    console.print(f"  unknown dyn:  {len(col_groups['time_varying_unknown_cols'])}")

    model = train_model(features=features, **col_groups)
    console.rule("[bold green]Training Complete[/bold green]")
    return model


if __name__ == "__main__":
    run_training()
