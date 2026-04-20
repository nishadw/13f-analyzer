"""
Data ingestion orchestrator.

Runs all three sources (EDGAR, market proxies, FRED) and persists results to disk.
Called every Friday evening as part of the production loop.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from rich.console import Console

import config
from data.ingestion.edgar_client import fetch_13f_history
from data.ingestion.fred_client import fetch_macro_series, compute_yield_curve_features
from data.ingestion.market_client import pull_weekly_proxy_data

console = Console()


def run_edgar_ingestion(ciks: list[str] | None = None, n_quarters: int = 8) -> pd.DataFrame:
    ciks = ciks or config.TARGET_CIKS
    frames = []
    for cik in ciks:
        console.print(f"[cyan]EDGAR[/cyan] fetching 13F history for CIK {cik}...")
        df = fetch_13f_history(cik, n_quarters=n_quarters)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    out = config.DATA_DIR / "13f_holdings.parquet"
    result.to_parquet(out, index=False)
    console.print(f"[green]EDGAR[/green] saved {len(result):,} rows → {out}")
    return result


def run_market_ingestion(weeks_back: int = 1) -> dict[str, pd.DataFrame]:
    console.print("[cyan]Market[/cyan] pulling weekly proxy data...")
    data = pull_weekly_proxy_data(weeks_back=weeks_back)
    for key, df in data.items():
        if df.empty:
            console.print(f"[yellow]Market[/yellow] {key}: empty result")
            continue
        out = config.DATA_DIR / f"market_{key}.parquet"
        df.to_parquet(out, index=False)
        console.print(f"[green]Market[/green] {key}: {len(df):,} rows → {out}")
    return data


def run_fred_ingestion(lookback_years: int = 5) -> pd.DataFrame:
    console.print("[cyan]FRED[/cyan] pulling macro series...")
    from datetime import timedelta
    start = date.today().replace(month=1, day=1) - pd.DateOffset(years=lookback_years - 1)
    df = fetch_macro_series(start=start.date() if hasattr(start, "date") else start)
    df = compute_yield_curve_features(df)
    if df.empty:
        console.print("[yellow]FRED[/yellow] no data returned")
        return df
    out = config.DATA_DIR / "fred_macro.parquet"
    df.to_parquet(out)
    console.print(f"[green]FRED[/green] {len(df):,} weeks → {out}")
    return df


def run_full_ingestion() -> dict:
    console.rule("[bold]Data Ingestion Pipeline[/bold]")
    holdings = run_edgar_ingestion()
    proxy = run_market_ingestion()
    macro = run_fred_ingestion()
    console.rule("[bold green]Ingestion Complete[/bold green]")
    return {"holdings": holdings, "proxy": proxy, "macro": macro}


if __name__ == "__main__":
    run_full_ingestion()
