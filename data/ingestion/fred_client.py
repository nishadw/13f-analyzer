"""
FRED API client for macroeconomic features.

Uses fredapi to pull the series defined in config.FRED_SERIES.
All series are resampled to weekly frequency (last observation of each week)
and forward-filled to handle publication lags.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

import config


def _get_fred():
    from fredapi import Fred  # type: ignore

    if not config.FRED_API_KEY:
        raise EnvironmentError("FRED_API_KEY is not set. See .env.example.")
    return Fred(api_key=config.FRED_API_KEY)


def fetch_macro_series(
    start: date,
    end: date | None = None,
    series: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Fetch FRED macro series and return a weekly DataFrame indexed by date.
    Columns are the friendly names from config.FRED_SERIES.
    """
    end = end or date.today()
    series = series or config.FRED_SERIES
    fred = _get_fred()

    frames: dict[str, pd.Series] = {}
    for name, series_id in series.items():
        try:
            s = fred.get_series(series_id, observation_start=str(start), observation_end=str(end))
            s.name = name
            frames[name] = s
        except Exception as exc:
            print(f"[fred] failed to fetch {series_id} ({name}): {exc}")

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    # Resample to weekly (Friday close), forward-fill publication lags
    df = df.resample("W-FRI").last().ffill()
    return df.loc[str(start):str(end)]


def fetch_weekly_macro(weeks_back: int = 4) -> pd.DataFrame:
    """Pull the last `weeks_back` weeks of macro data."""
    end = date.today()
    start = end - timedelta(weeks=weeks_back + 4)  # extra buffer for ffill
    df = fetch_macro_series(start=start, end=end)
    if df.empty:
        return df
    cutoff = pd.Timestamp(end) - pd.Timedelta(weeks=weeks_back)
    return df[df.index >= cutoff]


def compute_yield_curve_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived yield curve features if raw series are present."""
    out = df.copy()
    if "treasury_10y" in out.columns and "treasury_2y" in out.columns:
        out["yield_spread_10y2y"] = out["treasury_10y"] - out["treasury_2y"]
        # Inversion flag: 1 when curve is inverted
        out["yield_curve_inverted"] = (out["yield_spread_10y2y"] < 0).astype(float)
    if "cpi_yoy" in out.columns:
        out["cpi_mom"] = out["cpi_yoy"].pct_change()
    return out
