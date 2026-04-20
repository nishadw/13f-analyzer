"""
Data ingestion orchestrator.

Runs all three sources (EDGAR, market proxies, FRED) and persists results to disk.
Called every Friday evening as part of the production loop.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
from rich.console import Console

import config
from data.ingestion.edgar_client import fetch_13f_history, fetch_fund_name
from data.ingestion.fred_client import fetch_macro_series, compute_yield_curve_features
from data.ingestion.market_client import pull_weekly_proxy_data

console = Console()

_FUND_NAMES_PATH = config.DATA_DIR / "fund_names.json"


def _load_fund_names() -> dict[str, str]:
    if _FUND_NAMES_PATH.exists():
        try:
            return json.loads(_FUND_NAMES_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_fund_names(names: dict[str, str]) -> None:
    _FUND_NAMES_PATH.write_text(json.dumps(names, indent=2))


def run_edgar_ingestion(ciks: list[str] | None = None, n_quarters: int = 8) -> pd.DataFrame:
    ciks = ciks or config.TARGET_CIKS
    frames = []
    fund_names = _load_fund_names()

    for cik in ciks:
        cik_norm = str(int(cik))
        console.print(f"[cyan]EDGAR[/cyan] fetching CIK {cik_norm}...")
        df = fetch_13f_history(cik, n_quarters=n_quarters)
        if not df.empty:
            frames.append(df)
        # Fetch and cache the fund's legal name from EDGAR
        if cik_norm not in fund_names:
            name = fetch_fund_name(cik)
            fund_names[cik_norm] = name
            console.print(f"  name: {name}")

    _save_fund_names(fund_names)

    if not frames:
        return pd.DataFrame()

    # Merge with any existing data for CIKs not being re-fetched
    out = config.DATA_DIR / "13f_holdings.parquet"
    new_data = pd.concat(frames, ignore_index=True)
    if out.exists():
        existing = pd.read_parquet(out)
        fetched_ciks = {str(int(c)) for c in ciks}
        kept = existing[~existing["cik"].astype(str).map(lambda c: str(int(c)) if c.strip().lstrip("0") else "0").isin(fetched_ciks)]
        result = pd.concat([kept, new_data], ignore_index=True)
    else:
        result = new_data

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
