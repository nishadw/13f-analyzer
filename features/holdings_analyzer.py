"""
Holdings analyzer and time series tracker.

Groups 13F holdings by fund and provides:
- Current portfolio allocation (%). (what they actually hold)
- Historical changes (quarter-over-quarter)
- Top positions per fund
- Inter-fund comparison
"""
from __future__ import annotations

import pandas as pd

import config
from features.sector_map import name_to_sector


def _normalize_cik(cik: str) -> str:
    """Normalize CIK so values with/without leading zeros match."""
    s = str(cik or "").strip()
    if not s:
        return ""
    return s.lstrip("0") or "0"


def _infer_ticker_from_name(name: str) -> str:
    """Best-effort ticker inference from issuer name for table display."""
    if not name:
        return ""

    n = str(name).upper()
    mapping = {
        "ALPHABET INC": "GOOGL",
        "AMAZON COM INC": "AMZN",
        "APPLE INC": "AAPL",
        "BROADCOM INC": "AVGO",
        "MICROSOFT CORP": "MSFT",
        "META PLATFORMS INC": "META",
        "NVIDIA CORPORATION": "NVDA",
        "TESLA INC": "TSLA",
        "UBER TECHNOLOGIES INC": "UBER",
        "VISA INC": "V",
        "ADOBE INC": "ADBE",
        "BOOKING HOLDINGS INC": "BKNG",
        "SALESFORCE INC": "CRM",
        "LAM RESEARCH CORP": "LRCX",
        "HILTON WORLDWIDE HLDGS INC": "HLT",
        "RESTAURANT BRANDS INTL INC": "QSR",
        "SPDR S&P 500 ETF TR": "SPY",
        "SPDR DOW JONES INDL AVERAGE": "DIA",
        "SPDR GOLD TR": "GLD",
        "ISHARES TR": "",
        "ISHARES INC": "",
        "BROOKFIELD CORP": "BN",
    }
    return mapping.get(n, "")


def load_all_holdings() -> pd.DataFrame:
    """Load all stored 13F holdings."""
    try:
        return pd.read_parquet(config.DATA_DIR / "13f_holdings.parquet")
    except Exception:
        return pd.DataFrame()


def get_funds_summary() -> list[dict]:
    """
    Get summary of all tracked funds: latest filing date, positions count, total value.
    """
    holdings = load_all_holdings()
    if holdings.empty:
        return []

    holdings = holdings.copy()
    holdings["cik_normalized"] = holdings["cik"].astype(str).map(_normalize_cik)

    summaries = []
    for cik in config.TARGET_CIKS:
        cik_norm = _normalize_cik(cik)
        fund_holdings = holdings[holdings["cik_normalized"] == cik_norm]
        if fund_holdings.empty:
            summaries.append({
                "cik": cik,
                "name": _get_fund_name(cik),
                "latest_period": "",
                "num_positions": 0,
                "total_value_usd_mm": 0.0,
                "diversification": 0,
            })
            continue

        latest_period = fund_holdings["period"].max()
        latest_holdings = fund_holdings[fund_holdings["period"] == latest_period]

        summaries.append({
            "cik": cik,
            "name": _get_fund_name(cik),
            "latest_period": str(latest_period)[:10],
            "num_positions": len(latest_holdings[latest_holdings["put_call"].fillna("") == ""]),
            "total_value_usd_mm": round(
                latest_holdings["value_thousands"].sum() / 1000, 0
            ),
            "diversification": latest_holdings["cusip"].nunique(),
        })

    return summaries


def get_fund_current_holdings(
    cik: str,
    top_n: int = 25,
    include_options: bool = False,
) -> list[dict]:
    """
    Get current holdings for a specific fund, sorted by value descending.

    Args:
        cik: Fund CIK
        top_n: Return top N positions
        include_options: Include options positions (put/call)

    Returns:
        List of dicts: {name, cusip, shares, value_usd_mm, weight_pct, sector, ...}
    """
    holdings = load_all_holdings()
    if holdings.empty:
        return []

    cik_norm = _normalize_cik(cik)
    holdings = holdings.copy()
    holdings["cik_normalized"] = holdings["cik"].astype(str).map(_normalize_cik)

    fund_holdings = holdings[holdings["cik_normalized"] == cik_norm]
    if fund_holdings.empty:
        return []

    # Get latest period
    latest = fund_holdings["period"].max()
    latest_holdings = fund_holdings[fund_holdings["period"] == latest]

    # Filter equity only unless requested
    if not include_options:
        latest_holdings = latest_holdings[latest_holdings["put_call"].fillna("") == ""]

    # Sort by value
    latest_holdings = latest_holdings.nlargest(top_n, "value_thousands")

    rows = []
    for _, row in latest_holdings.iterrows():
        rows.append({
            "rank": len(rows) + 1,
            "name": row.get("name", ""),
            "ticker": _infer_ticker_from_name(row.get("name", "")),
            "cusip": row.get("cusip", ""),
            "shares": int(row.get("shares", 0)) if pd.notna(row.get("shares")) else 0,
            "value_usd_mm": round(float(row.get("value_thousands", 0)) / 1000, 2),
            "weight_pct": round(float(row.get("weight", 0)) * 100, 2),
            "sector": name_to_sector(str(row.get("name", ""))),
            "put_call": row.get("put_call", ""),
        })

    return rows


def get_fund_holdings_history(cik: str, n_periods: int = 8) -> dict:
    """
    Get holdings changes over N most recent quarters for a fund.

    Returns:
      {
        periods: [date1, date2, ...],
        top_movers: [{name, cusip, values: [v1, v2, ...], change_pct}, ...],
        sector_exposure: {quarter: {sector: weight_pct}, ...},
      }
    """
    holdings = load_all_holdings()
    if holdings.empty:
        return {}

    cik_norm = _normalize_cik(cik)
    fund_holdings = holdings.copy()
    fund_holdings["cik_normalized"] = fund_holdings["cik"].astype(str).map(_normalize_cik)
    fund_holdings = fund_holdings[fund_holdings["cik_normalized"] == cik_norm].copy()
    if fund_holdings.empty:
        return {}

    # Get N most recent periods
    periods = sorted(fund_holdings["period"].unique(), reverse=True)[:n_periods]
    periods = sorted(periods)  # Chronological order for chart

    if len(periods) < 2:
        return {}

    period_strs = [str(p)[:10] for p in periods]

    # Find top movers: stocks that changed most in absolute weight
    all_cusips = set()
    for p in periods:
        all_cusips.update(fund_holdings[fund_holdings["period"] == p]["cusip"].dropna())

    top_movers = []
    for cusip in all_cusips:
        cusip_history = fund_holdings[fund_holdings["cusip"] == cusip].sort_values("period")
        if len(cusip_history) < 2:
            continue

        values = []
        for p in periods:
            row = cusip_history[cusip_history["period"] == p]
            if not row.empty:
                values.append(float(row.iloc[0].get("weight", 0)) * 100)
            else:
                values.append(0.0)

        change = values[-1] - values[0]
        if abs(change) > 0.1:  # Moved more than 0.1%
            top_movers.append({
                "name": cusip_history.iloc[0].get("name", ""),
                "cusip": cusip,
                "values": [round(v, 2) for v in values],
                "change_pct": round(change, 2),
                "latest_weight": round(values[-1], 2),
            })

    # Sort by absolute change
    top_movers.sort(key=lambda x: abs(x["change_pct"]), reverse=True)
    top_movers = top_movers[:10]  # Top 10 movers

    # Sector exposure over time
    sector_exposure = {}
    for p in periods:
        period_holdings = fund_holdings[fund_holdings["period"] == p]
        sector_weights = {}
        total = period_holdings["value_thousands"].sum()

        for _, row in period_holdings.iterrows():
            if row.get("put_call", "") != "":  # Skip options
                continue
            sector = name_to_sector(str(row.get("name", "")))
            sector_weights[sector] = sector_weights.get(sector, 0) + float(row.get("value_thousands", 0))

        # Normalize to weights
        for sector in sector_weights:
            sector_weights[sector] = round(sector_weights[sector] / total * 100, 1) if total > 0 else 0.0

        sector_exposure[str(p)[:10]] = sector_weights

    return {
        "periods": period_strs,
        "top_movers": top_movers,
        "sector_exposure": sector_exposure,
    }


def compare_funds(ciks: list[str], top_n: int = 15) -> dict:
    """
    Compare holdings across multiple funds.

    Returns common holdings, unique to each fund, and allocation differences.
    """
    holdings = load_all_holdings()
    if holdings.empty:
        return {}

    holdings = holdings.copy()
    holdings["cik_normalized"] = holdings["cik"].astype(str).map(_normalize_cik)

    normalized_to_requested = {_normalize_cik(c): c for c in ciks}

    fund_data = {}
    all_cusips = set()

    for cik in ciks:
        cik_norm = _normalize_cik(cik)
        fund_h = holdings[holdings["cik_normalized"] == cik_norm]
        if fund_h.empty:
            continue

        latest = fund_h["period"].max()
        latest_h = fund_h[
            (fund_h["period"] == latest) & (fund_h["put_call"].fillna("") == "")
        ].nlargest(top_n, "value_thousands")

        requested_key = normalized_to_requested.get(cik_norm, cik)
        fund_data[requested_key] = {}
        for _, row in latest_h.iterrows():
            cusip = row.get("cusip", "")
            fund_data[requested_key][cusip] = {
                "name": row.get("name", ""),
                "weight_pct": round(float(row.get("weight", 0)) * 100, 2),
                "value_usd_mm": round(float(row.get("value_thousands", 0)) / 1000, 2),
            }
            all_cusips.add(cusip)

    # Create comparison table
    comparison = []
    for cusip in sorted(all_cusips):
        row = {"cusip": cusip, "name": ""}
        for cik in ciks:
            if cusip in fund_data.get(cik, {}):
                data = fund_data[cik][cusip]
                row["name"] = data["name"]
                row[f"{cik}_weight"] = data["weight_pct"]
                row[f"{cik}_value"] = data["value_usd_mm"]
            else:
                row[f"{cik}_weight"] = 0.0
                row[f"{cik}_value"] = 0.0

        # Only include if held by at least 2 funds or in top holdings of any fund
        if sum(1 for cik in ciks if row[f"{cik}_weight"] > 0) >= 1:
            comparison.append(row)

    return {
        "comparison": comparison,
        "fund_data": fund_data,
    }


def _get_fund_name(cik: str) -> str:
    """Look up fund name from fund_names.json (populated during ingestion)."""
    import json as _json
    cik_norm = _normalize_cik(cik)
    path = config.DATA_DIR / "fund_names.json"
    if path.exists():
        try:
            names = _json.loads(path.read_text())
            if cik_norm in names:
                return names[cik_norm]
        except Exception:
            pass
    return cik_norm


def get_sector_breakdown_for_fund(cik: str) -> dict[str, float]:
    """
    Get sector allocation (%) for a fund's current holdings.
    """
    holdings = load_all_holdings()
    if holdings.empty:
        return {}

    cik_norm = _normalize_cik(cik)
    fund_h = holdings.copy()
    fund_h["cik_normalized"] = fund_h["cik"].astype(str).map(_normalize_cik)
    fund_h = fund_h[fund_h["cik_normalized"] == cik_norm]
    if fund_h.empty:
        return {}

    latest = fund_h["period"].max()
    latest_h = fund_h[
        (fund_h["period"] == latest) & (fund_h["put_call"].fillna("") == "")
    ]

    sector_values = {}
    total = latest_h["value_thousands"].sum()

    for _, row in latest_h.iterrows():
        sector = name_to_sector(str(row.get("name", "")))
        sector_values[sector] = sector_values.get(sector, 0) + float(row.get("value_thousands", 0))

    # Convert to percentages
    sector_pcts = {
        s: round(v / total * 100, 1) for s, v in sector_values.items()
    } if total > 0 else {}

    return sector_pcts
