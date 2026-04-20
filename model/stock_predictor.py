"""
Stock-level ML prediction engine.

Trains a HistGradientBoostingClassifier on historical 13F position transitions
combined with sector ETF and individual stock price momentum features.

Output: continuous signal ∈ [-1, +1]
  +1.0 = very strong buy conviction
   0.0 = neutral / hold
  -1.0 = very strong sell conviction
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier

import config
from features.sector_map import name_to_sector

warnings.filterwarnings("ignore", category=FutureWarning)

_MODEL_CACHE: dict[str, object] = {
    "signature": None,
    "model": None,
}

_PRICE_CACHE: dict[tuple[str, ...], dict[str, dict[str, float]]] = {}

# ── CUSIP → ticker ─────────────────────────────────────────────────────────────

_CUSIP_TO_TICKER: dict[str, str] = {
    "037833100": "AAPL",   "594918104": "MSFT",   "67066G104": "NVDA",
    "023135106": "AMZN",   "30303M102": "META",   "02079K305": "GOOGL",
    "02079K107": "GOOG",   "88160R101": "TSLA",   "46090E103": "QQQ",
    "78462F103": "SPY",    "464287655": "IWM",    "78463V107": "GLD",
    "11135F101": "AVGO",   "532457108": "LLY",    "46625H100": "JPM",
    "92826C839": "V",      "91324P102": "UNH",    "30231G102": "XOM",
    "57636Q104": "MA",     "437076102": "HD",     "742718109": "PG",
    "478160104": "JNJ",    "22160K105": "COST",   "00287Y109": "ABBV",
    "64110L106": "NFLX",   "007903107": "AMD",    "747525103": "QCOM",
    "882184100": "TXN",    "458140100": "INTC",   "79466L302": "CRM",
    "68389X105": "ORCL",   "81762P102": "NOW",    "00724F101": "ADBE",
    "595112103": "MU",     "67066G104": "NVDA",   "46090E103": "QQQ",
    "17275R102": "CSCO",   "459200101": "IBM",    "G01767105": "ACN",
    "74762E102": "PYPL",   "38141G104": "GS",     "61166W101": "MS",
    "172967424": "BRK-B",  "025816109": "AXP",    "808513105": "SCHW",
    "09247X101": "BLK",    "747590107": "USB",    "13646108":  "TFC",
    "17275R102": "CSCO",   "345370860": "F",      "370442105": "GM",
    "035240100": "BA",     "526057104": "LMT",    "763901101": "RTX",
    "670346105": "NOC",    "812523105": "GD",     "14170T101": "CAT",
    "244199105": "DE",     "438516106": "HON",    "369550108": "GE",
    "064058100": "BAX",    "110122108": "BMY",    "60871R209": "MRNA",
    "552463106": "MCD",    "85208M102": "SBUX",   "741503207": "PFE",
    "58155Q103": "MRK",    "931142103": "WAB",    "256219106": "DIS",
    "20030N101": "CMCSA",  "92343V104": "VZ",     "87236Y108": "TMUS",
    "23331A109": "UBER",   "G162491024": "BN",    "023135106": "AMZN",
    "26210C104": "DVN",    "337932107": "FCX",    "67887L107": "OXY",
    "169905106": "CHTR",
}

_NAME_TO_TICKER: dict[str, str] = {
    "ALPHABET INC":          "GOOGL", "AMAZON COM INC":         "AMZN",
    "APPLE INC":             "AAPL",  "BROADCOM INC":           "AVGO",
    "MICROSOFT CORP":        "MSFT",  "META PLATFORMS INC":     "META",
    "NVIDIA CORPORATION":    "NVDA",  "TESLA INC":              "TSLA",
    "UBER TECHNOLOGIES INC": "UBER",  "VISA INC":               "V",
    "ADOBE INC":             "ADBE",  "BOOKING HOLDINGS INC":   "BKNG",
    "SALESFORCE INC":        "CRM",   "LAM RESEARCH CORP":      "LRCX",
    "HILTON WORLDWIDE HLDGS INC": "HLT",
    "RESTAURANT BRANDS INTL INC": "QSR",
    "BROOKFIELD CORP":       "BN",    "NETFLIX INC":            "NFLX",
    "ORACLE CORP":           "ORCL",  "SERVICENOW INC":         "NOW",
    "MASTERCARD INC":        "MA",    "HOME DEPOT INC":         "HD",
    "JPMORGAN CHASE & CO":   "JPM",   "UNITEDHEALTH GROUP INC": "UNH",
    "ELI LILLY & CO":        "LLY",   "COSTCO WHSL CORP":       "COST",
    "JOHNSON & JOHNSON":     "JNJ",   "EXXON MOBIL CORP":       "XOM",
    "PROCTER & GAMBLE CO":   "PG",    "ABBVIE INC":             "ABBV",
    "ADVANCED MICRO DEVICES": "AMD",  "QUALCOMM INC":           "QCOM",
    "CISCO SYSTEMS INC":     "CSCO",  "INTEL CORP":             "INTC",
    "PALO ALTO NETWORKS INC": "PANW", "CROWDSTRIKE HLDGS INC":  "CRWD",
    "SPDR S&P 500 ETF TR":   "SPY",   "INVESCO QQQ TR":         "QQQ",
    "SPDR DOW JONES INDL AVERAGE": "DIA",
    "SPDR GOLD TR":          "GLD",   "ISHARES TR":             "IVV",
}


def get_ticker(name: str, cusip: str) -> Optional[str]:
    t = _CUSIP_TO_TICKER.get(cusip.strip())
    if t:
        return t
    return _NAME_TO_TICKER.get(name.strip().upper())


# ── Feature engineering ────────────────────────────────────────────────────────

def _sector_features(sector: str, feature_store: pd.DataFrame) -> dict[str, float]:
    """Pull latest sector ETF features from the weekly feature store."""
    out = {"sector_flow_z": 0.0, "sector_mom4w": 0.0, "sector_mom13w": 0.0}
    if feature_store.empty or not sector or sector == "Other":
        return out
    row = feature_store.iloc[-1]
    for suf, key in [("_flow_z", "sector_flow_z"), ("_mom4w", "sector_mom4w"), ("_mom13w", "sector_mom13w")]:
        col = f"{sector}{suf}"
        if col in row.index and pd.notna(row[col]):
            out[key] = float(row[col])
    return out


def _macro_features(feature_store: pd.DataFrame) -> dict[str, float]:
    out = {"vix": 20.0, "yield_spread": 0.0}
    if feature_store.empty:
        return out
    row = feature_store.iloc[-1]
    for col, key in [("vix", "vix"), ("yield_spread_10y2y", "yield_spread"), ("yield_spread", "yield_spread")]:
        if col in row.index and pd.notna(row[col]):
            out[key] = float(row[col])
    return out


def _fetch_price_features_bulk(tickers: list[str]) -> dict[str, dict[str, float]]:
    """
    Batch-download 9 months of weekly closes for all tickers at once.
    Returns per-ticker dict of {ret_3m, ret_6m, vol_3m, rel_ret_sector}.
    """
    clean = sorted({t.upper() for t in tickers if t and 1 <= len(t) <= 6})
    if not clean:
        return {}

    cache_key = tuple(clean)
    cached = _PRICE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        data = yf.download(
            clean, period="9mo", interval="1wk",
            auto_adjust=True, progress=False,
        )
    except Exception:
        return {}

    if data is None or data.empty:
        return {}

    close = data["Close"] if "Close" in data else data
    if isinstance(close, pd.Series):
        close = close.to_frame(name=clean[0])

    results: dict[str, dict[str, float]] = {}
    for t in clean:
        if t not in close.columns:
            continue
        px = close[t].dropna()
        if len(px) < 10:
            continue
        last = float(px.iloc[-1])
        def _ret(n: int) -> float:
            if len(px) < n:
                return 0.0
            base = float(px.iloc[-n])
            return (last / base - 1.0) if base > 0 else 0.0
        ret_3m = _ret(13)
        ret_6m = _ret(26)
        vol_3m = float(px.pct_change().dropna().tail(13).std()) if len(px) >= 14 else 0.0
        results[t] = {"ret_3m": ret_3m, "ret_6m": ret_6m, "vol_3m": vol_3m}

    # Add sector-relative return (vs SPY as broad market proxy)
    spy_ret_3m = results.get("SPY", {}).get("ret_3m", 0.0)
    for t in results:
        results[t]["rel_ret_3m"] = results[t]["ret_3m"] - spy_ret_3m

    _PRICE_CACHE[cache_key] = results
    return results


# ── ML model ───────────────────────────────────────────────────────────────────

_INCREASE_THRESH =  0.15   # weight change > +15% → label "increase"
_DECREASE_THRESH = -0.15   # weight change < -15% or exit → label "decrease"

_FEATURE_COLS = [
    "holding_streak", "weight_latest", "weight_change_1q", "weight_change_2q",
    "weight_trend", "port_pct_rank",
    "sector_flow_z", "sector_mom4w", "sector_mom13w",
    "vix", "yield_spread",
    "stock_ret_3m", "stock_ret_6m", "stock_vol_3m", "stock_rel_ret_3m",
]

_CLASSES = ["decrease", "hold", "increase"]


def _build_training_data(
    holdings: pd.DataFrame,
    feature_store: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build (X, y) training pairs from all CIKs and all consecutive period transitions.
    For each (cik, cusip, period_t) that has a next period, compute features and label.
    """
    eq = holdings[holdings["put_call"].fillna("") == ""].copy()
    eq = eq.sort_values("value_thousands", ascending=False).drop_duplicates(
        subset=["cik", "period", "cusip"]
    )

    rows = []
    all_tickers: set[str] = set()

    for cik, fund in eq.groupby("cik"):
        periods = sorted(fund["period"].unique())
        for i in range(len(periods) - 1):
            p_curr, p_next = periods[i], periods[i + 1]
            curr_df = fund[fund["period"] == p_curr].set_index("cusip")
            next_df = fund[fund["period"] == p_next].set_index("cusip")
            all_cusips = curr_df.index.union(next_df.index)
            port_weights = curr_df["weight"].astype(float)

            for cusip in all_cusips:
                w_curr = float(curr_df["weight"].get(cusip, 0.0))
                w_next = float(next_df["weight"].get(cusip, 0.0))

                if w_curr == 0.0 and w_next == 0.0:
                    continue

                if w_curr == 0.0:
                    label = "increase"
                elif w_next == 0.0:
                    label = "decrease"
                else:
                    chg = (w_next - w_curr) / max(w_curr, 1e-8)
                    if chg > _INCREASE_THRESH:
                        label = "increase"
                    elif chg < _DECREASE_THRESH:
                        label = "decrease"
                    else:
                        label = "hold"

                name = ""
                if cusip in curr_df.index:
                    name = str(curr_df.loc[cusip, "name"] if "name" in curr_df.columns else "")
                elif cusip in next_df.index:
                    name = str(next_df.loc[cusip, "name"] if "name" in next_df.columns else "")

                ticker = get_ticker(name, cusip) or ""
                if ticker:
                    all_tickers.add(ticker)

                # Weight history
                hist = fund[fund["cusip"] == cusip].sort_values("period")
                hist = hist[hist["period"] <= p_curr]["weight"].astype(float).values[-4:]
                wpad = np.pad(hist, (max(0, 4 - len(hist)), 0))
                streak = int(len(hist))
                changes = np.diff(wpad)
                w_chg1 = float(changes[-1]) if len(changes) > 0 else 0.0
                w_chg2 = float(changes[-2]) if len(changes) > 1 else 0.0
                w_trend = float(np.mean(changes)) if len(changes) > 0 else 0.0
                ranked = port_weights.rank(pct=True)
                port_rank = float(ranked.get(cusip, 0.5))

                sector = name_to_sector(name)
                s_feats = {"sector_flow_z": 0.0, "sector_mom4w": 0.0, "sector_mom13w": 0.0}
                m_feats = {"vix": 20.0, "yield_spread": 0.0}
                if not feature_store.empty:
                    fs_before = feature_store[feature_store.index <= p_curr]
                    if not fs_before.empty:
                        # Use a temporary single-row df for _sector_features
                        tmp = feature_store.loc[[fs_before.index[-1]]]
                        s_feats = _sector_features(sector, tmp)
                        m_feats = _macro_features(tmp)

                rows.append({
                    "cusip": cusip, "name": name, "ticker": ticker,
                    "cik": cik, "period": p_curr, "label": label,
                    "holding_streak": streak,
                    "weight_latest": w_curr,
                    "weight_change_1q": w_chg1,
                    "weight_change_2q": w_chg2,
                    "weight_trend": w_trend,
                    "port_pct_rank": port_rank,
                    **s_feats, **m_feats,
                    "stock_ret_3m": 0.0, "stock_ret_6m": 0.0,
                    "stock_vol_3m": 0.0, "stock_rel_ret_3m": 0.0,
                })

    if not rows:
        return pd.DataFrame(columns=_FEATURE_COLS), pd.Series(dtype=str)

    df = pd.DataFrame(rows)

    # Batch-fetch price data for all mapped tickers
    price_map = _fetch_price_features_bulk(list(all_tickers))
    for col in ("stock_ret_3m", "stock_ret_6m", "stock_vol_3m", "stock_rel_ret_3m"):
        df[col] = df["ticker"].map(
            lambda t, c=col: price_map.get(t, {}).get(c.replace("stock_", ""), 0.0)
        )

    X = df[_FEATURE_COLS].fillna(0.0)
    y = df["label"]
    return X, y


def _train(X: pd.DataFrame, y: pd.Series) -> Optional[HistGradientBoostingClassifier]:
    if X.empty or len(y) < 30:
        return None
    clf = HistGradientBoostingClassifier(
        max_iter=100, max_depth=4, learning_rate=0.05,
        min_samples_leaf=20, random_state=42,
    )
    clf.fit(X, y)
    return clf


def _holdings_signature(holdings: pd.DataFrame, feature_store: pd.DataFrame) -> tuple:
    latest_period = str(pd.to_datetime(holdings["period"]).max()) if not holdings.empty else ""
    latest_filing = str(pd.to_datetime(holdings["filing_date"]).max()) if not holdings.empty and "filing_date" in holdings.columns else ""
    fund_count = int(holdings["cik"].astype(str).nunique()) if not holdings.empty and "cik" in holdings.columns else 0
    feature_shape = tuple(feature_store.shape) if not feature_store.empty else (0, 0)
    feature_stamp = str(feature_store.index.max()) if not feature_store.empty else ""
    return (len(holdings), latest_period, latest_filing, fund_count, feature_shape, feature_stamp)


def _get_or_train_model(holdings: pd.DataFrame, feature_store: pd.DataFrame) -> Optional[HistGradientBoostingClassifier]:
    signature = _holdings_signature(holdings, feature_store)
    if _MODEL_CACHE["signature"] == signature:
        return _MODEL_CACHE["model"]  # type: ignore[return-value]

    X_train, y_train = _build_training_data(holdings, feature_store)
    clf = _train(X_train, y_train)
    _MODEL_CACHE["signature"] = signature
    _MODEL_CACHE["model"] = clf
    return clf


# ── Public interface ───────────────────────────────────────────────────────────

# Column definitions shown on hover in UI tables
COLUMN_DEFS = [
    {"field": "name",             "headerName": "Company",        "headerTooltip": "Company name as reported in the 13F-HR filing"},
    {"field": "ticker",           "headerName": "Ticker",         "headerTooltip": "Exchange ticker symbol inferred from CUSIP or company name"},
    {"field": "signal",           "headerName": "Signal",         "headerTooltip": "ML conviction score: +1 = strong buy, 0 = neutral, -1 = strong sell. Computed as tanh(P(increase) − P(decrease)) from a GradientBoosting model trained on 4-quarter 13F transitions"},
    {"field": "current_weight_pct","headerName": "Weight %",      "headerTooltip": "Current portfolio weight percentage as reported in the most recent 13F filing. 0% means this is a predicted new-buy candidate not currently held."},
    {"field": "sector",           "headerName": "Sector ETF",     "headerTooltip": "Sector ETF proxy this stock maps to (e.g. SMH=Semiconductors, XLK=Technology, XLF=Financials)"},
    {"field": "momentum_3m",      "headerName": "Momentum 3M",    "headerTooltip": "Stock's 3-month price return from yfinance weekly data. Positive = price trending up."},
    {"field": "momentum_6m",      "headerName": "Momentum 6M",    "headerTooltip": "Stock's 6-month price return. Used as a longer-term trend signal in the model."},
    {"field": "rel_momentum",     "headerName": "Rel. Momentum",  "headerTooltip": "3-month return relative to SPY (S&P 500). Positive = outperforming the broad market."},
    {"field": "holding_streak",   "headerName": "Quarters Held",  "headerTooltip": "Consecutive quarters this fund has held the position. Longer streaks suggest higher conviction from the manager."},
    {"field": "weight_trend",     "headerName": "Weight Trend",   "headerTooltip": "Average quarter-over-quarter change in portfolio weight over the last 4 filings. Positive = manager has been consistently adding."},
    {"field": "sector_flow_z",    "headerName": "Sector Flow Z",  "headerTooltip": "Z-score of sector ETF net flows vs. 52-week history. Values above +1 indicate unusual institutional inflows into this sector."},
    {"field": "source",           "headerName": "Source",         "headerTooltip": "held = stock is in the fund's current 13F portfolio; candidate = predicted new-buy based on the fund's historical methodology"},
    {"field": "cusip",            "headerName": "CUSIP",          "headerTooltip": "Committee on Uniform Securities Identification Procedures code — the unique identifier used by SEC filings"},
]


def generate_signals(
    cik: str,
    holdings: pd.DataFrame,
    feature_store: pd.DataFrame,
    top_n: int = 30,
    include_candidates: bool = True,
) -> dict:
    """
    Unified signal generation combining current holdings + predicted new buys.

    Returns frontend-compatible dict:
      {
        "data": [...],
        "columnsDefs": [...]
      }

    Signal ∈ [-1, +1]:
      Trained GBC → P(increase), P(hold), P(decrease)
      signal = tanh((P(increase) - P(decrease)) * 2.5)
    """
    holdings = holdings.copy()
    holdings["put_call"] = holdings.get("put_call", pd.Series("", index=holdings.index)).fillna("")

    # ── Train model on all available historical data ───────────────────────────
    clf = _get_or_train_model(holdings, feature_store)

    # ── Current holdings features ──────────────────────────────────────────────
    eq = holdings[holdings["put_call"] == ""].copy()
    eq = eq.sort_values("value_thousands", ascending=False).drop_duplicates(
        subset=["cik", "period", "cusip"]
    )
    if cik:
        fund = eq[eq["cik"] == cik]
    else:
        fund = eq
    latest_period = fund["period"].max()
    current = fund[fund["period"] == latest_period].nlargest(top_n, "value_thousands")

    port_weights = current.set_index("cusip")["weight"].astype(float)
    ranked = port_weights.rank(pct=True)

    m_feats = _macro_features(feature_store)

    # Batch price fetch for all tickers in current holdings
    current_tickers = [
        get_ticker(str(r.get("name", "")), str(r.get("cusip", "")))
        for _, r in current.iterrows()
    ]
    price_map = _fetch_price_features_bulk([t for t in current_tickers if t])

    # ── Build feature matrix for current holdings ──────────────────────────────
    rows = []
    for idx, row in current.iterrows():
        name  = str(row.get("name", ""))
        cusip = str(row.get("cusip", ""))
        ticker = get_ticker(name, cusip) or ""
        sector = name_to_sector(name)
        w_curr = float(row.get("weight", 0.0))

        # 13F history
        hist_all = eq[(eq["cik"] == cik) & (eq["cusip"] == cusip)] if cik else \
                   eq[eq["cusip"] == cusip]
        hist = hist_all.sort_values("period")["weight"].astype(float).values[-4:]
        wpad = np.pad(hist, (max(0, 4 - len(hist)), 0))
        streak = int(len(hist))
        changes = np.diff(wpad)
        w_chg1  = float(changes[-1]) if len(changes) > 0 else 0.0
        w_chg2  = float(changes[-2]) if len(changes) > 1 else 0.0
        w_trend = float(np.mean(changes)) if len(changes) > 0 else 0.0
        port_rank = float(ranked.get(cusip, 0.5))

        s_feats = _sector_features(sector, feature_store)
        pf = price_map.get(ticker, {})

        feat_row = {
            "holding_streak":   streak,
            "weight_latest":    w_curr,
            "weight_change_1q": w_chg1,
            "weight_change_2q": w_chg2,
            "weight_trend":     w_trend,
            "port_pct_rank":    port_rank,
            **s_feats, **m_feats,
            "stock_ret_3m":    pf.get("ret_3m", 0.0),
            "stock_ret_6m":    pf.get("ret_6m", 0.0),
            "stock_vol_3m":    pf.get("vol_3m", 0.0),
            "stock_rel_ret_3m": pf.get("rel_ret_3m", 0.0),
        }
        rows.append({
            "cusip": cusip, "name": name, "ticker": ticker, "sector": sector,
            "current_weight_pct": round(w_curr * 100, 2),
            "holding_streak": streak,
            "weight_trend": round(w_trend * 100, 4),
            "sector_flow_z": round(s_feats.get("sector_flow_z", 0.0), 3),
            "momentum_3m": round(pf.get("ret_3m", 0.0) * 100, 2),
            "momentum_6m": round(pf.get("ret_6m", 0.0) * 100, 2),
            "rel_momentum": round(pf.get("rel_ret_3m", 0.0) * 100, 2),
            "source": "held",
            "_features": feat_row,
        })

    # ── Deduplicate holdings by ticker (keep highest-value row per ticker) ───────
    seen_tickers: set[str] = set()
    deduped_rows = []
    for row in rows:
        t = row["ticker"]
        key = t if t else row["cusip"]  # fall back to CUSIP when ticker is unknown
        if key and key not in seen_tickers:
            seen_tickers.add(key)
            deduped_rows.append(row)
    rows = deduped_rows

    # ── Compute signals ────────────────────────────────────────────────────────
    if clf is not None and rows:
        X_pred = pd.DataFrame([r["_features"] for r in rows])[_FEATURE_COLS].fillna(0.0)
        proba = clf.predict_proba(X_pred)
        class_order = list(clf.classes_)
        idx_dec = class_order.index("decrease") if "decrease" in class_order else -1
        idx_inc = class_order.index("increase") if "increase" in class_order else -1
        for i, row in enumerate(rows):
            p_inc = float(proba[i, idx_inc]) if idx_inc >= 0 else 0.33
            p_dec = float(proba[i, idx_dec]) if idx_dec >= 0 else 0.33
            raw = (p_inc - p_dec) * 2.5
            row["signal"] = round(float(np.tanh(raw)), 3)
    else:
        # Fallback: simple heuristic when no model
        for row in rows:
            f = row["_features"]
            s = (f["weight_trend"] * 5.0 + f["stock_ret_3m"] * 1.5 + f["sector_flow_z"] * 0.3)
            row["signal"] = round(float(np.tanh(s)), 3)

    # ── Add predicted-buy candidates ───────────────────────────────────────────
    if include_candidates:
        current_cusips = {r["cusip"] for r in rows}
        # Also pass known tickers so candidates don't duplicate held positions
        current_tickers_set = {r["ticker"] for r in rows if r["ticker"]}
        candidate_rows = _predict_new_buys(
            cik, holdings, feature_store, clf, current_cusips, price_map, m_feats,
            n=10, exclude_tickers=current_tickers_set,
        )
        rows.extend(candidate_rows)

    # Sort by signal descending, clean up internal fields
    for row in rows:
        row.pop("_features", None)
    rows.sort(key=lambda r: -r["signal"])

    return {"data": rows, "columnsDefs": COLUMN_DEFS}


def _predict_new_buys(
    cik: str,
    holdings: pd.DataFrame,
    feature_store: pd.DataFrame,
    clf: Optional[HistGradientBoostingClassifier],
    exclude_cusips: set[str],
    existing_price_map: dict,
    m_feats: dict,
    n: int = 10,
    exclude_tickers: Optional[set[str]] = None,
) -> list[dict]:
    """
    Identify stocks not currently held that match the fund's historical buy pattern.
    Universe: stocks held by OTHER tracked funds but not by this one.
    """
    eq = holdings[holdings["put_call"].fillna("") == ""].copy()
    eq = eq.sort_values("value_thousands", ascending=False).drop_duplicates(
        subset=["cik", "period", "cusip"]
    )

    # Find fund's most-bought sectors and characteristics
    if cik:
        fund_hist = eq[eq["cik"] == cik]
    else:
        fund_hist = eq

    # Universe: stocks in other funds' LATEST period that this fund doesn't hold
    other = eq[eq["cik"] != cik] if cik else eq
    if other.empty:
        return []

    _excl_tickers = exclude_tickers or set()
    latest_other = other[other["period"] == other["period"].max()]
    candidates = latest_other[~latest_other["cusip"].isin(exclude_cusips)].nlargest(50, "value_thousands")
    # Further filter by ticker to avoid duplicating held positions under a different CUSIP
    candidates = candidates[
        candidates.apply(
            lambda r: (get_ticker(str(r.get("name", "")), str(r.get("cusip", ""))) or "") not in _excl_tickers,
            axis=1,
        )
    ]

    if candidates.empty:
        return []

    # Fetch price data for candidates
    cand_tickers = [
        get_ticker(str(r.get("name", "")), str(r.get("cusip", "")))
        for _, r in candidates.iterrows()
    ]
    new_tickers = [t for t in cand_tickers if t and t not in existing_price_map]
    price_map = {**existing_price_map, **_fetch_price_features_bulk(new_tickers)}

    # Use fund's average holding characteristics as a "template"
    fund_latest = fund_hist[fund_hist["period"] == fund_hist["period"].max()]
    avg_streak = 2  # New positions start at streak=1 but are predicted with 0 history
    avg_sector_flow = 0.0
    if not feature_store.empty:
        for etf in config.PROXY_ETFS:
            col = f"{etf}_flow_z"
            if col in feature_store.columns:
                avg_sector_flow = float(feature_store[col].iloc[-1])
                break

    # Build metadata and feature rows together, then batch-predict
    meta_rows = []
    feat_rows = []
    for _, row in candidates.iterrows():
        name   = str(row.get("name", ""))
        cusip  = str(row.get("cusip", ""))
        ticker = get_ticker(name, cusip) or ""
        sector = name_to_sector(name)
        s_feats = _sector_features(sector, feature_store)
        pf = price_map.get(ticker, {})
        meta_rows.append({
            "cusip": cusip, "name": name, "ticker": ticker, "sector": sector,
            "s_feats": s_feats, "pf": pf,
        })
        feat_rows.append({
            "holding_streak": 0, "weight_latest": 0.0,
            "weight_change_1q": 0.0, "weight_change_2q": 0.0,
            "weight_trend": 0.0, "port_pct_rank": 0.0,
            **s_feats, **m_feats,
            "stock_ret_3m":     pf.get("ret_3m", 0.0),
            "stock_ret_6m":     pf.get("ret_6m", 0.0),
            "stock_vol_3m":     pf.get("vol_3m", 0.0),
            "stock_rel_ret_3m": pf.get("rel_ret_3m", 0.0),
        })

    if not feat_rows:
        return []

    if clf is not None:
        X_all = pd.DataFrame(feat_rows)[_FEATURE_COLS].fillna(0.0)
        probas = clf.predict_proba(X_all)
        class_order = list(clf.classes_)
        idx_dec = class_order.index("decrease") if "decrease" in class_order else -1
        idx_inc = class_order.index("increase") if "increase" in class_order else -1
    else:
        probas = None

    cand_rows = []
    for i, meta in enumerate(meta_rows):
        pf = meta["pf"]
        s_feats = meta["s_feats"]
        if probas is not None:
            p_inc = float(probas[i, idx_inc]) if idx_inc >= 0 else 0.33
            p_dec = float(probas[i, idx_dec]) if idx_dec >= 0 else 0.33
            signal = round(float(np.tanh((p_inc - p_dec) * 2.5)), 3)
        else:
            signal = round(float(np.tanh(pf.get("ret_3m", 0.0) * 2.0 + s_feats.get("sector_flow_z", 0.0) * 0.5)), 3)

        if signal > 0.05:
            cand_rows.append({
                "cusip":              meta["cusip"],
                "name":               meta["name"],
                "ticker":             meta["ticker"],
                "sector":             meta["sector"],
                "current_weight_pct": 0.0,
                "holding_streak":     0,
                "weight_trend":       0.0,
                "sector_flow_z":      round(s_feats.get("sector_flow_z", 0.0), 3),
                "momentum_3m":        round(pf.get("ret_3m", 0.0) * 100, 2),
                "momentum_6m":        round(pf.get("ret_6m", 0.0) * 100, 2),
                "rel_momentum":       round(pf.get("rel_ret_3m", 0.0) * 100, 2),
                "source":             "candidate",
                "signal":             signal,
            })

    seen: set[str] = set()
    deduped: list[dict] = []
    for r in sorted(cand_rows, key=lambda x: -x["signal"]):
        key = r["ticker"] if r["ticker"] else r["cusip"]
        if key and key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped[:n]
