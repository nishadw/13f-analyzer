"""
Custom API backend for the 13F Analyzer.

Run:
    uvicorn api_backend.main:app --reload --host 0.0.0.0 --port 7779
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from features.sector_map import name_to_sector

app = FastAPI(
    title="13F Analyzer",
    description="Institutional portfolio prediction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_HERE = Path(__file__).parent

_EMPTY_CHART = {
    "data": [],
    "layout": {
        "annotations": [{
            "text": "No data yet - run the pipeline first",
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5,
            "showarrow": False,
            "font": {"size": 14},
        }],
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
    },
}


# ── Auth ───────────────────────────────────────────────────────────────────────

def _check_api_key(request: Request) -> None:
    valid_keys_raw = os.getenv("VALID_API_KEYS", "")
    if not valid_keys_raw:
        return
    valid_keys = [k.strip() for k in valid_keys_raw.split(",") if k.strip()]
    if request.headers.get("X-API-KEY", "") not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid or missing X-API-KEY header")


# ── Config endpoints ───────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"info": "13F Analyzer API Backend", "version": "1.0.0"}


# ── Sector lookup ──────────────────────────────────────────────────────────────

# Each entry: (list of uppercase name substrings, sector ETF proxy)
_SECTOR_MAP: list[tuple[list[str], str]] = [
    (["NVIDIA", "INTEL", "AMD", "QUALCOMM", "BROADCOM", "APPLIED MATERIALS",
      "LAM RESEARCH", "MICRON", "KLA CORP", "ASML", "MARVELL", "SKYWORKS",
      "MONOLITHIC POWER", "QORVO", "ON SEMICONDUCTOR", "LATTICE SEMI",
      "ENTEGRIS", "AXCELIS", "ONTO INNOVATION", "ADVANCED ENERGY",
      "PHOTRONICS", "KULICKE"], "SMH"),
    (["APPLE", "MICROSOFT", "ORACLE", "SALESFORCE", "SERVICENOW", "ADOBE",
      "INTUIT", "CISCO", "IBM", "ACCENTURE", "PALO ALTO", "CROWDSTRIKE",
      "SNOWFLAKE", "WORKDAY", "AUTODESK", "CADENCE", "SYNOPSYS", "FISERV",
      "PAYCOM", "PAYCHEX", "FORTINET", "DATADOG", "CLOUDFLARE", "PALANTIR",
      "SCIENCE APPLICATIONS", "LEIDOS", "BOOZ ALLEN", "KEYSIGHT", "GARTNER",
      "FAIR ISAAC", "VERISIGN", "TRIMBLE"], "XLK"),
    (["META", "ALPHABET", "GOOGLE", "NETFLIX", "DISNEY", "COMCAST", "CHARTER",
      "VERIZON", "AT&T", "T-MOBILE", "ELECTRONIC ARTS", "ACTIVISION",
      "TAKE-TWO", "MATCH GROUP", "PINTEREST", "SNAP", "LIVE NATION",
      "NEWS CORP", "FOX CORP", "PARAMOUNT", "WARNER BROS", "NEXSTAR",
      "INTERPUBLIC", "OMNICOM"], "XLC"),
    (["AMAZON", "TESLA", "HOME DEPOT", "NIKE", "MCDONALD", "STARBUCKS",
      "LOWE'S", "BOOKING HOLDINGS", "AIRBNB", "EXPEDIA", "ROSS STORES",
      "TJX COMPANIES", "DOLLAR GENERAL", "DOLLAR TREE", "AUTOZONE",
      "O'REILLY", "ADVANCE AUTO", "GENUINE PARTS", "ULTA BEAUTY",
      "POOL CORP", "TRACTOR SUPPLY", "EBAY", "ETSY", "WAYFAIR", "CHEWY"], "XLY"),
    (["JPMORGAN", "BANK OF AMERICA", "WELLS FARGO", "CITIGROUP",
      "GOLDMAN SACHS", "MORGAN STANLEY", "BERKSHIRE", "VISA", "MASTERCARD",
      "AMERICAN EXPRESS", "CHARLES SCHWAB", "BLACKROCK", "U.S. BANCORP",
      "TRUIST", "CAPITAL ONE", "DISCOVER", "SYNCHRONY", "ALLY FINANCIAL",
      "REGIONS FINANCIAL", "HUNTINGTON", "CITIZENS FINANCIAL", "FIFTH THIRD",
      "KEYCORP", "M&T BANK", "AMERIPRISE", "T. ROWE PRICE",
      "CBOE", "CME GROUP", "INTERCONTINENTAL EXCHANGE", "NASDAQ INC",
      "GLOBAL PAYMENTS", "FLEETCOR", "WEX INC", "HARTFORD", "TRAVELERS",
      "ALLSTATE", "CHUBB", "AIG", "METLIFE", "AFLAC", "MARKEL"], "XLF"),
    (["UNITEDHEALTH", "JOHNSON & JOHNSON", "ELI LILLY", "PFIZER", "ABBVIE",
      "MERCK", "THERMO FISHER", "DANAHER", "ABBOTT LABORATORIES", "MEDTRONIC",
      "STRYKER", "BECTON DICKINSON", "BOSTON SCIENTIFIC", "EDWARDS LIFESCIENCES",
      "INTUITIVE SURGICAL", "GILEAD", "BIOGEN", "REGENERON", "AMGEN", "VERTEX",
      "MODERNA", "HUMANA", "CVS HEALTH", "ANTHEM", "CIGNA", "CENTENE", "MOLINA",
      "LABORATORY CORP", "QUEST DIAGNOSTICS", "IQVIA", "CARDINAL HEALTH",
      "MCKESSON", "AMERISOURCEBERGEN", "HOLOGIC", "ALIGN TECHNOLOGY"], "XLV"),
    (["EXXON", "CHEVRON", "PIONEER NATURAL", "CONOCOPHILLIPS", "EOG RESOURCES",
      "SCHLUMBERGER", "HALLIBURTON", "BAKER HUGHES", "HESS CORP", "VALERO",
      "PHILLIPS 66", "MARATHON PETROLEUM", "OCCIDENTAL PETROLEUM",
      "DEVON ENERGY", "DIAMONDBACK", "COTERRA", "APA CORP", "OVINTIV"], "XLE"),
    (["CATERPILLAR", "DEERE", "HONEYWELL", "GENERAL ELECTRIC", "BOEING",
      "LOCKHEED MARTIN", "RAYTHEON", "NORTHROP GRUMMAN", "GENERAL DYNAMICS",
      "L3HARRIS", "TRANSDIGM", "HEICO", "3M COMPANY", "PARKER HANNIFIN",
      "EATON CORP", "ILLINOIS TOOL", "EMERSON ELECTRIC", "ROPER TECHNOLOGIES",
      "DOVER CORP", "XYLEM", "FLOWSERVE", "IDEX CORP", "GRACO", "NORDSON",
      "CSX CORP", "UNION PACIFIC", "NORFOLK SOUTHERN", "UNITED RENTALS",
      "CINTAS", "COPART", "STERICYCLE", "MANPOWERGROUP", "ADP"], "XLI"),
    (["NEXTERA", "DUKE ENERGY", "SOUTHERN COMPANY", "DOMINION ENERGY",
      "AMERICAN ELECTRIC POWER", "XCEL ENERGY", "CONSOLIDATED EDISON",
      "ENTERGY", "SEMPRA", "AMEREN", "WEC ENERGY", "EVERGY",
      "EVERSOURCE", "ALLIANT ENERGY", "CMS ENERGY", "ATMOS ENERGY"], "XLU"),
    (["FREEPORT", "NUCOR", "STEEL DYNAMICS", "U.S. STEEL", "CENTURY ALUMINUM",
      "AIR PRODUCTS", "LINDE", "PRAXAIR", "CORTEVA", "CF INDUSTRIES",
      "MOSAIC", "NUTRIEN", "EASTMAN CHEMICAL", "CELANESE", "INTERNATIONAL PAPER",
      "WEYERHAEUSER", "PACKAGING CORP", "SEALED AIR", "AVERY DENNISON",
      "SONOCO", "GRAPHIC PACKAGING"], "XLB"),
    (["PROLOGIS", "AMERICAN TOWER", "CROWN CASTLE", "SBA COMMUNICATIONS",
      "DIGITAL REALTY", "EQUINIX", "IRON MOUNTAIN", "SIMON PROPERTY",
      "EQUITY RESIDENTIAL", "AVALONBAY", "UDR INC", "CAMDEN PROPERTY",
      "ESSEX PROPERTY", "REALTY INCOME", "NATIONAL RETAIL PROPERTIES",
      "AGREE REALTY", "STORE CAPITAL", "WELLTOWER", "VENTAS", "HEALTHPEAK",
      "NATIONAL HEALTH INVESTORS"], "XLRE"),
    (["PROCTER & GAMBLE", "COCA-COLA", "WALMART", "COSTCO", "PEPSICO",
      "PHILIP MORRIS", "ALTRIA GROUP", "KRAFT HEINZ", "MONDELEZ",
      "GENERAL MILLS", "KELLOGG", "CAMPBELL SOUP", "CONAGRA", "HERSHEY",
      "COLGATE-PALMOLIVE", "CLOROX", "CHURCH & DWIGHT", "DIAGEO",
      "CONSTELLATION BRANDS", "MOLSON COORS", "BOSTON BEER", "TYSON FOODS",
      "HORMEL", "ARCHER-DANIELS", "SPROUTS FARMERS"], "XLP"),
]


def _name_to_sector(name: str) -> str:
    # Backward-compatible alias used by existing endpoints.
    return name_to_sector(name)


def _equity_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop options rows (put_call populated) so pie/table show only equity positions."""
    if "put_call" in df.columns:
        df = df[df["put_call"].fillna("") == ""]
    return df


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _load_latest_portfolio() -> dict:
    p = config.DATA_DIR / "portfolio_latest.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _normalize_cik(cik: str) -> str:
    s = str(cik or "").strip()
    if not s:
        return ""
    return s.lstrip("0") or "0"


def _filter_by_cik(df: pd.DataFrame, cik: str) -> pd.DataFrame:
    if df.empty or not cik:
        return df
    if "cik" not in df.columns:
        return df

    normalized = _normalize_cik(cik)
    tmp = df.copy()
    tmp["_cik_norm"] = tmp["cik"].astype(str).map(_normalize_cik)
    return tmp[tmp["_cik_norm"] == normalized].drop(columns=["_cik_norm"], errors="ignore")


# ── Fund-specific holdings endpoints ────────────────────────────────────────────

@app.get("/funds_summary")
def funds_summary(request: Request):
    """List all tracked funds with latest filing date, positions count, AUM."""
    _check_api_key(request)
    try:
        from features.holdings_analyzer import get_funds_summary
        return JSONResponse(content=get_funds_summary())
    except Exception as e:
        print(f"[funds_summary] Error: {e}")
        return JSONResponse(content=[])


@app.get("/funds_latest_filings")
def funds_latest_filings(request: Request):
    """Latest available 13F filing metadata for each tracked fund CIK."""
    _check_api_key(request)
    try:
        from data.ingestion.edgar_client import list_13f_filings
        from features.holdings_analyzer import _get_fund_name

        rows = []
        for cik in config.TARGET_CIKS:
            filings = list_13f_filings(cik)
            latest = filings[0] if filings else {}
            rows.append({
                "cik": cik,
                "name": _get_fund_name(cik),
                "has_filings": bool(latest),
                "latest_report_date": latest.get("reportDate", ""),
                "latest_filing_date": latest.get("filingDate", ""),
                "latest_accession": latest.get("accessionNumber", ""),
                "form_type": latest.get("formType", ""),
            })

        return JSONResponse(content=rows)
    except Exception as e:
        print(f"[funds_latest_filings] Error: {e}")
        return JSONResponse(content=[])


@app.get("/fund_holdings")
def fund_holdings(
    request: Request,
    cik: str = Query(default=""),
    top_n: int = Query(default=25),
):
    """
    Get current holdings for a specific fund, sorted by value descending.
    """
    _check_api_key(request)
    if not cik and config.TARGET_CIKS:
        cik = config.TARGET_CIKS[0]

    try:
        from features.holdings_analyzer import get_fund_current_holdings
        holdings = get_fund_current_holdings(cik, top_n=top_n)
        return JSONResponse(content=holdings)
    except Exception as e:
        print(f"[fund_holdings] Error: {e}")
        return JSONResponse(content=[])


@app.get("/fund_holdings_pie")
def fund_holdings_pie(
    request: Request,
    cik: str = Query(default=""),
    top_n: int = Query(default=15),
):
    """
    Pie chart of top holdings for a fund (actual stocks, not sectors).
    """
    _check_api_key(request)
    if not cik and config.TARGET_CIKS:
        cik = config.TARGET_CIKS[0]

    try:
        from features.holdings_analyzer import get_fund_current_holdings, _get_fund_name

        holdings = get_fund_current_holdings(cik, top_n=top_n)
        if not holdings:
            return JSONResponse(content=_EMPTY_CHART)

        # Calculate "Other" for positions beyond top_n
        total_weight = sum(h["weight_pct"] for h in holdings)
        other_weight = max(0, 100 - total_weight)

        labels = [h["name"] for h in holdings]
        values = [h["weight_pct"] for h in holdings]

        if other_weight > 0.5:
            labels.append("Other")
            values.append(other_weight)

        fund_name = _get_fund_name(cik)
        figure = {
            "data": [{
                "type": "pie",
                "labels": labels,
                "values": values,
                "hole": 0.35,
                "textinfo": "label+percent",
                "hovertemplate": "%{label}<br>%{value:.2f}%<extra></extra>",
            }],
            "layout": {
                "title": f"{fund_name} — Current Holdings",
                "showlegend": True,
                "plot_bgcolor": "rgba(0,0,0,0)",
                "paper_bgcolor": "rgba(0,0,0,0)",
            },
        }
        return JSONResponse(content=figure)
    except Exception as e:
        print(f"[fund_holdings_pie] Error: {e}")
        return JSONResponse(content=_EMPTY_CHART)


@app.get("/fund_holdings_history")
def fund_holdings_history(
    request: Request,
    cik: str = Query(default=""),
):
    """
    Historical holdings changes for a fund over past N quarters.
    Shows top movers and sector exposure over time.
    """
    _check_api_key(request)
    if not cik and config.TARGET_CIKS:
        cik = config.TARGET_CIKS[0]

    try:
        from features.holdings_analyzer import get_fund_holdings_history
        history = get_fund_holdings_history(cik)
        return JSONResponse(content=history)
    except Exception as e:
        print(f"[fund_holdings_history] Error: {e}")
        return JSONResponse(content={})


@app.get("/stock_holdings_history")
def stock_holdings_history(
    request: Request,
    cik: str = Query(default=""),
    cusip: str = Query(default=""),
    name: str = Query(default=""),
    ticker: str = Query(default=""),
    n_periods: int = Query(default=4),
):
    """
    Historical weight changes for a single holding across the most recent filings.
    """
    _check_api_key(request)

    df = _load_parquet(config.DATA_DIR / "13f_holdings.parquet")
    if df.empty:
        return JSONResponse(content={})

    df = _equity_only(df)
    if cik:
        cik_norm = _normalize_cik(cik)
        df["cik_norm"] = df["cik"].astype(str).map(_normalize_cik)
        df = df[df["cik_norm"] == cik_norm].drop(columns=["cik_norm"], errors="ignore")

    if df.empty:
        return JSONResponse(content={})

    periods = sorted(df["period"].dropna().unique())[-max(2, n_periods):]
    if len(periods) < 2:
        return JSONResponse(content={})

    window = df[df["period"].isin(periods)].copy()

    def _infer_ticker(value: str) -> str:
        return str(value or "").strip().upper()

    ticker_norm = _infer_ticker(ticker)
    cusip_norm = str(cusip or "").strip().upper()
    name_norm = str(name or "").strip().upper()

    if cusip_norm:
        match = window[window["cusip"].astype(str).str.upper() == cusip_norm]
    elif ticker_norm:
        match = window[window["name"].astype(str).str.upper().map(lambda v: _infer_ticker(v) == ticker_norm)]
    elif name_norm:
        match = window[window["name"].astype(str).str.upper() == name_norm]
    else:
        return JSONResponse(content={})

    if match.empty:
        return JSONResponse(content={})

    match = match.sort_values("period")
    values = []
    for period in periods:
        row = match[match["period"] == period]
        if row.empty:
            values.append(0.0)
        else:
            values.append(round(float(row.iloc[0].get("weight", 0)) * 100, 2))

    deltas = []
    increase_total = 0.0
    decrease_total = 0.0
    for idx in range(1, len(values)):
        delta = round(values[idx] - values[idx - 1], 2)
        deltas.append(delta)
        if delta > 0:
            increase_total += delta
        elif delta < 0:
            decrease_total += abs(delta)

    latest_row = match.iloc[-1]
    payload = {
        "periods": [str(p)[:10] for p in periods],
        "values": values,
        "deltas": deltas,
        "increase_total": round(increase_total, 2),
        "decrease_total": round(decrease_total, 2),
        "net_change": round(values[-1] - values[0], 2),
        "name": str(latest_row.get("name", name or "")),
        "cusip": str(latest_row.get("cusip", cusip or "")),
        "ticker": ticker_norm,
    }
    return JSONResponse(content=payload)


@app.get("/fund_holdings_history_chart")
def fund_holdings_history_chart(
    request: Request,
    cik: str = Query(default=""),
):
    """
    Multi-line chart showing top movers in holdings over past quarters.
    """
    _check_api_key(request)
    if not cik and config.TARGET_CIKS:
        cik = config.TARGET_CIKS[0]

    try:
        from features.holdings_analyzer import get_fund_holdings_history, _get_fund_name

        history = get_fund_holdings_history(cik)
        if not history or not history.get("top_movers"):
            return JSONResponse(content=_EMPTY_CHART)

        periods = history.get("periods", [])
        top_movers = history.get("top_movers", [])[:8]  # Limit to 8 for chart clarity

        traces = []
        for mover in top_movers:
            traces.append({
                "type": "scatter",
                "mode": "lines+markers",
                "x": periods,
                "y": mover["values"],
                "name": mover["name"][:30],  # Truncate long names
            })

        fund_name = _get_fund_name(cik)
        figure = {
            "data": traces,
            "layout": {
                "title": f"{fund_name} — Top 8 Movers (Holdings % Over Time)",
                "xaxis": {"title": "Period"},
                "yaxis": {"title": "Portfolio Weight (%)"},
                "hovermode": "x unified",
                "plot_bgcolor": "rgba(0,0,0,0)",
                "paper_bgcolor": "rgba(0,0,0,0)",
            },
        }
        return JSONResponse(content=figure)
    except Exception as e:
        print(f"[fund_holdings_history_chart] Error: {e}")
        return JSONResponse(content=_EMPTY_CHART)


@app.get("/fund_sector_exposure_chart")
def fund_sector_exposure_chart(
    request: Request,
    cik: str = Query(default=""),
):
    """
    Stacked bar chart of sector exposure over time for a fund.
    """
    _check_api_key(request)
    if not cik and config.TARGET_CIKS:
        cik = config.TARGET_CIKS[0]

    try:
        from features.holdings_analyzer import get_fund_holdings_history, _get_fund_name

        history = get_fund_holdings_history(cik)
        if not history or not history.get("sector_exposure"):
            return JSONResponse(content=_EMPTY_CHART)

        periods = list(history.get("sector_exposure", {}).keys())
        sector_exposure = history.get("sector_exposure", {})

        # Get all sectors
        all_sectors = set()
        for pe in sector_exposure.values():
            all_sectors.update(pe.keys())

        # Build stacked bar traces
        traces = []
        for sector in sorted(all_sectors):
            values = [sector_exposure.get(p, {}).get(sector, 0) for p in periods]
            traces.append({
                "type": "bar",
                "x": periods,
                "y": values,
                "name": sector,
                "stackgroup": "sector",
            })

        fund_name = _get_fund_name(cik)
        figure = {
            "data": traces,
            "layout": {
                "title": f"{fund_name} — Sector Exposure Over Time",
                "xaxis": {"title": "Period"},
                "yaxis": {"title": "Sector Weight (%)"},
                "barmode": "stack",
                "plot_bgcolor": "rgba(0,0,0,0)",
                "paper_bgcolor": "rgba(0,0,0,0)",
            },
        }
        return JSONResponse(content=figure)
    except Exception as e:
        print(f"[fund_sector_exposure_chart] Error: {e}")
        return JSONResponse(content=_EMPTY_CHART)


@app.get("/funds_comparison")
def funds_comparison(
    request: Request,
    ciks: str = Query(default=""),
    top_n: int = Query(default=15),
):
    """
    Compare holdings across multiple funds.
    ciks: Comma-separated CIK list, e.g., "1350694,1336528,1037389"
    """
    _check_api_key(request)

    if not ciks:
        # Default to top 3 funds
        cik_list = config.TARGET_CIKS[:3]
    else:
        cik_list = [c.strip() for c in ciks.split(",")]

    try:
        from features.holdings_analyzer import compare_funds
        comparison = compare_funds(cik_list, top_n=top_n)
        return JSONResponse(content=comparison)
    except Exception as e:
        print(f"[funds_comparison] Error: {e}")
        return JSONResponse(content={})


# ── Widget endpoints ───────────────────────────────────────────────────────────

@app.get("/portfolio_weights")
def portfolio_weights(request: Request):
    _check_api_key(request)
    latest = _load_latest_portfolio()
    if not latest:
        return JSONResponse(content={"as_of": "pending", "data": []})

    weights: dict = latest.get("weights", {})
    rows = sorted(
        [{"asset": k, "weight": round(v, 6), "weight_pct": f"{v*100:.2f}%"} for k, v in weights.items()],
        key=lambda r: r["weight"],
        reverse=True,
    )
    return JSONResponse(content={"as_of": latest.get("date", ""), "data": rows})


@app.get("/portfolio_chart")
def portfolio_chart(request: Request):
    _check_api_key(request)
    latest = _load_latest_portfolio()
    if not latest:
        return JSONResponse(content=_EMPTY_CHART)

    weights = latest.get("weights", {})
    sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    figure = {
        "data": [{
            "type": "bar",
            "x": [i[0] for i in sorted_items],
            "y": [round(i[1] * 100, 2) for i in sorted_items],
            "marker": {"color": "rgba(50, 171, 96, 0.85)"},
            "name": "Weight %",
        }],
        "layout": {
            "title": f"Predicted Portfolio Weights - {latest.get('date', '')}",
            "xaxis": {"title": "Asset"},
            "yaxis": {"title": "Weight (%)"},
            "plot_bgcolor": "rgba(0,0,0,0)",
            "paper_bgcolor": "rgba(0,0,0,0)",
        },
    }
    return JSONResponse(content=figure)


@app.get("/holdings")
def holdings(
    request: Request,
    cik: str = Query(default=""),
    period: str = Query(default=""),
    top_n: int = Query(default=20),
):
    _check_api_key(request)
    df = _load_parquet(config.DATA_DIR / "13f_holdings.parquet")
    if df.empty:
        return JSONResponse(content=[])

    df = _equity_only(df)
    if cik:
        df = df[df["cik"] == cik]
    if period:
        df = df[df["period"].astype(str).str.startswith(period)]
    else:
        df = df[df["period"] == df["period"].max()]

    df = df.nlargest(top_n, "value_thousands")
    df["value_usd_mm"] = (df["value_thousands"] / 1000).round(2)
    df["weight_pct"] = (df["weight"] * 100).round(2)
    if "period" in df.columns:
        df["period"] = df["period"].astype(str).str[:10]
    if "filing_date" in df.columns:
        df["filing_date"] = df["filing_date"].astype(str).str[:10]

    cols = ["name", "cusip", "weight_pct", "value_usd_mm", "shares",
            "period", "filing_date", "cik", "investment_discretion"]
    out = df[[c for c in cols if c in df.columns]].to_dict(orient="records")
    return JSONResponse(content=out)


@app.get("/holdings_chart")
def holdings_chart(
    request: Request,
    cik: str = Query(default=""),
    top_n: int = Query(default=15),
):
    _check_api_key(request)
    df = _load_parquet(config.DATA_DIR / "13f_holdings.parquet")
    if df.empty:
        return JSONResponse(content=_EMPTY_CHART)

    df = _equity_only(df)
    if cik:
        df = df[df["cik"] == cik]

    latest_period = df["period"].max()
    df = df[df["period"] == latest_period].nlargest(top_n, "value_thousands")

    top_weight = df["weight"].sum()
    other_weight = max(0.0, 1.0 - top_weight)

    labels = list(df["name"]) + (["Other"] if other_weight > 0.001 else [])
    values = list((df["weight"] * 100).round(2)) + ([round(other_weight * 100, 2)] if other_weight > 0.001 else [])
    period_str = str(latest_period)[:10]

    figure = {
        "data": [{
            "type": "pie",
            "labels": labels,
            "values": values,
            "hole": 0.35,
            "textinfo": "label+percent",
            "hovertemplate": "%{label}<br>%{value:.2f}%<extra></extra>",
        }],
        "layout": {
            "title": f"Current Holdings  ({period_str})  —  CIK {cik or 'all'}",
            "showlegend": False,
            "plot_bgcolor": "rgba(0,0,0,0)",
            "paper_bgcolor": "rgba(0,0,0,0)",
        },
    }
    return JSONResponse(content=figure)


@app.get("/holdings_predicted_change")
def holdings_predicted_change(
    request: Request,
    cik: str = Query(default=""),
    top_n: int = Query(default=25),
):
    """
    For each current 13F equity position, show the model's sector signal:
    maps the stock to a sector ETF proxy, then compares the TFT-predicted
    sector weight against the equal-weight baseline.
    """
    _check_api_key(request)
    df = _load_parquet(config.DATA_DIR / "13f_holdings.parquet")
    if df.empty:
        return JSONResponse(content=[])

    df = _equity_only(df)
    if cik:
        df = df[df["cik"] == cik]

    df = df[df["period"] == df["period"].max()].nlargest(top_n, "value_thousands")

    portfolio = _load_latest_portfolio()
    predicted = portfolio.get("weights", {})
    n_sectors = len(config.PROXY_ETFS)
    eq_weight = 1.0 / n_sectors if n_sectors > 0 else 0.0

    rows = []
    for _, row in df.iterrows():
        name = str(row.get("name", ""))
        sector = _name_to_sector(name)
        pred_w = predicted.get(sector, eq_weight)
        delta = pred_w - eq_weight

        if delta > 0.03:
            signal = "▲ OVERWEIGHT"
        elif delta < -0.03:
            signal = "▼ UNDERWEIGHT"
        else:
            signal = "= NEUTRAL"

        rows.append({
            "name": name,
            "cusip": row.get("cusip", ""),
            "current_weight_pct": round(float(row.get("weight", 0)) * 100, 2),
            "value_usd_mm": round(float(row.get("value_thousands", 0)) / 1000, 2),
            "sector_etf": sector,
            "predicted_sector_pct": round(pred_w * 100, 2),
            "vs_equal_weight_pct": round(delta * 100, 2),
            "signal": signal,
        })

    return JSONResponse(content=rows)


@app.get("/etf_flows")
def etf_flows(
    request: Request,
    symbols: str = Query(default=""),
    weeks: int = Query(default=13),
):
    _check_api_key(request)
    features = _load_parquet(config.FEATURE_STORE_PATH)
    if features.empty:
        return JSONResponse(content=[])

    flow_cols = [c for c in features.columns if c.endswith("_flow_z")]
    if symbols:
        requested = [s.strip().upper() for s in symbols.split(",")]
        flow_cols = [c for c in flow_cols if any(c.startswith(s) for s in requested)]

    df = features[flow_cols].tail(weeks).reset_index()
    if "week_ending" in df.columns:
        df = df.rename(columns={"week_ending": "date"})
    df["date"] = df["date"].astype(str).str[:10]
    return JSONResponse(content=df.to_dict(orient="records"))


@app.get("/etf_flows_chart")
def etf_flows_chart(
    request: Request,
    symbols: str = Query(default="SMH,XLU,XLK,XLF"),
    weeks: int = Query(default=26),
):
    _check_api_key(request)
    features = _load_parquet(config.FEATURE_STORE_PATH)
    if features.empty:
        return JSONResponse(content=_EMPTY_CHART)

    requested = [s.strip().upper() for s in symbols.split(",")]
    flow_cols = [c for c in features.columns if c.endswith("_flow_z") and any(c.startswith(s) for s in requested)]
    df = features[flow_cols].tail(weeks)
    dates = [str(i)[:10] for i in df.index]

    traces = [
        {"type": "scatter", "mode": "lines", "x": dates,
         "y": df[col].round(3).tolist(), "name": col.replace("_flow_z", "")}
        for col in flow_cols
    ]
    return JSONResponse(content={
        "data": traces,
        "layout": {
            "title": "ETF Net Flow Z-Scores (Institutional Proxy)",
            "xaxis": {"title": "Week"},
            "yaxis": {"title": "Flow Z-Score"},
            "hovermode": "x unified",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "paper_bgcolor": "rgba(0,0,0,0)",
        },
    })


@app.get("/macro_metrics")
def macro_metrics(request: Request):
    _check_api_key(request)
    df = _load_parquet(config.DATA_DIR / "fred_macro.parquet")
    if df.empty:
        return JSONResponse(content=[])

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    labels = {
        "fed_funds_rate": "Fed Funds Rate",
        "unemployment": "Unemployment Rate",
        "cpi_yoy": "CPI (YoY)",
        "treasury_10y": "10Y Treasury",
        "treasury_2y": "2Y Treasury",
        "yield_spread_10y2y": "10Y-2Y Spread",
        "vix": "VIX",
        "credit_spread_hy": "HY Credit Spread",
    }
    rows = []
    for key, label in labels.items():
        if key not in latest:
            continue
        val, prev_val = latest[key], prev[key]
        change = round(float(val) - float(prev_val), 3) if pd.notna(val) and pd.notna(prev_val) else None
        rows.append({
            "indicator": label,
            "value": round(float(val), 3) if pd.notna(val) else None,
            "weekly_change": change,
            "direction": "up" if change and change > 0 else ("down" if change and change < 0 else "flat"),
        })
    return JSONResponse(content=rows)


@app.get("/macro_chart")
def macro_chart(
    request: Request,
    series: str = Query(default="treasury_10y,treasury_2y,fed_funds_rate"),
    weeks: int = Query(default=52),
):
    _check_api_key(request)
    df = _load_parquet(config.DATA_DIR / "fred_macro.parquet")
    if df.empty:
        return JSONResponse(content=_EMPTY_CHART)

    cols = [s.strip() for s in series.split(",") if s.strip() in df.columns]
    if not cols:
        return JSONResponse(content=_EMPTY_CHART)

    df = df[cols].tail(weeks)
    dates = [str(i)[:10] for i in df.index]
    traces = [
        {"type": "scatter", "mode": "lines", "x": dates,
         "y": df[col].round(3).tolist(), "name": col.replace("_", " ").title()}
        for col in cols
    ]
    return JSONResponse(content={
        "data": traces,
        "layout": {
            "title": "Macro Indicators (FRED)",
            "xaxis": {"title": "Week"},
            "yaxis": {"title": "Rate / Level"},
            "hovermode": "x unified",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "paper_bgcolor": "rgba(0,0,0,0)",
        },
    })


@app.get("/portfolio_history")
def portfolio_history(request: Request):
    _check_api_key(request)
    history_dir = config.DATA_DIR / "portfolio_history"
    if not history_dir.exists():
        return JSONResponse(content=[])

    frames = []
    for f in sorted(history_dir.glob("portfolio_*.parquet")):
        try:
            df = pd.read_parquet(f)
            df["run_date"] = f.stem.replace("portfolio_", "")
            frames.append(df)
        except Exception:
            pass

    if not frames:
        return JSONResponse(content=[])

    combined = pd.concat(frames, ignore_index=True)
    return JSONResponse(content=combined.to_dict(orient="records"))


@app.get("/tracked_funds")
def tracked_funds(request: Request):
    _check_api_key(request)
    df = _load_parquet(config.DATA_DIR / "13f_holdings.parquet")
    if df.empty:
        return JSONResponse(content=[{"cik": c} for c in config.TARGET_CIKS])

    summary = (
        df.groupby("cik")
        .agg(
            latest_period=("period", "max"),
            n_positions=("cusip", "nunique"),
            total_value_mm=("value_thousands", lambda x: round(x.sum() / 1000, 1)),
        )
        .reset_index()
    )
    summary["latest_period"] = summary["latest_period"].astype(str).str[:10]
    return JSONResponse(content=summary.to_dict(orient="records"))


# ── Stock signals (unified GBC model) ─────────────────────────────────────────

@app.get("/stock_signals")
def stock_signals(
    request: Request,
    cik: str = Query(default=""),
    top_n: int = Query(default=30),
    include_candidates: bool = Query(default=True),
):
    """
    Unified stock signal table for a fund.

    Trains a GradientBoostingClassifier on all available 13F quarterly
    transitions (increase / hold / decrease / new_buy / exited) combined with
    real price features (momentum, volatility) fetched from yfinance.

    Returns signal ∈ [-1, +1]:
      signal = tanh((P(increase) - P(decrease)) * 2.5)

    Positive → model predicts the fund will grow this position.
    Negative → model predicts a reduction or exit.
    Candidates (source="candidate") are stocks not currently held that the
    model scores as likely new buys based on other funds' patterns.
    """
    _check_api_key(request)

    try:
        from model.stock_predictor import generate_signals

        holdings = _load_parquet(config.DATA_DIR / "13f_holdings.parquet")
        if holdings.empty:
            return JSONResponse(content={"data": [], "columnsDefs": []})

        if not cik and config.TARGET_CIKS:
            cik = str(config.TARGET_CIKS[0])

        feature_store = _load_parquet(config.DATA_DIR / "features.parquet")

        result = generate_signals(
            cik=cik,
            holdings=holdings,
            feature_store=feature_store,
            top_n=top_n,
            include_candidates=include_candidates,
        )
        return JSONResponse(content=result)

    except Exception as e:
        print(f"[stock_signals] Error: {e}")
        import traceback; traceback.print_exc()
        return JSONResponse(content={"data": [], "columnsDefs": []})


