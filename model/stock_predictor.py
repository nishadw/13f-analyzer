"""
Stock-level ML prediction engine.

Trains a HistGradientBoostingRegressor on historical 13F position transitions
to predict next-quarter portfolio weight change in percentage points.
"""
from __future__ import annotations

import warnings
import json
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

import config
from features.sector_map import name_to_sector

warnings.filterwarnings("ignore", category=FutureWarning)

_MODEL_CACHE: dict[str, object] = {
    "signature": None,
    "model": None,
    "delta_scale": 0.6,
}

_MODEL_ARTIFACT_PATH = config.DATA_DIR / "stock_model.joblib"
_MODEL_META_PATH = config.DATA_DIR / "stock_model_meta.json"
_PROACTIVE_CANDIDATES_PATH = config.DATA_DIR / "proactive_candidates.json"

_PRICE_CACHE: dict[tuple[str, ...], dict[str, dict[str, float]]] = {}

_ETF_INDEX_TICKERS = {
    "SPY", "IVV", "VOO", "QQQ", "IWM", "DIA", "GLD", "XLF", "XLK", "XLY", "XLI", "XLP", "XLE", "XLC", "XLV", "XLU", "XLB", "XLRE", "SMH", "VTI",
}
_ETF_INDEX_NAME_HINTS = (
    " ETF", " ETN", " TRUST", " ISHARES", " SPDR", " VANGUARD", " INDEX", " FUND", " MSCI", " TOTAL MARKET",
)

_CIK_TO_PORTFOLIO = {
    "1350694": "Bridgewater",
    "1423053": "Citadel",
    "1336528": "Pershing Square",
    "2045724": "Situational Awareness",
    "1037389": "Renaissance",
    "1179392": "Two Sigma Investments",
    "1478735": "Two Sigma Advisers",
    "1009207": "D. E. Shaw",
    "1167557": "AQR",
    "1603465": "Cubist",
    "1603466": "Point72",
    "1564702": "PDT Partners",
}

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


def _is_stock_security(name: str, ticker: str) -> bool:
    t = str(ticker or "").strip().upper()
    n = f" {str(name or '').strip().upper()} "

    if t and t in _ETF_INDEX_TICKERS:
        return False
    for hint in _ETF_INDEX_NAME_HINTS:
        if hint in n:
            return False
    return True


def _portfolio_names_for(ciks: set[str]) -> str:
    if not ciks:
        return "Not in tracked portfolios"
    names = sorted({_CIK_TO_PORTFOLIO.get(str(c), f"CIK {str(c).lstrip('0') or '0'}") for c in ciks})
    if len(names) <= 2:
        return ", ".join(names)
    return f"{names[0]}, {names[1]} (+{len(names) - 2})"


def _build_portfolio_position_maps(latest_all: pd.DataFrame) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """
    Build lookup maps for expandable per-stock portfolio details.
    Keys are CUSIP and ticker.
    """
    if latest_all.empty:
        return {}, {}

    tmp = latest_all.copy()
    tmp["weight"] = pd.to_numeric(tmp.get("weight", 0.0), errors="coerce").fillna(0.0)
    tmp["shares"] = pd.to_numeric(tmp.get("shares", 0.0), errors="coerce").fillna(0.0)
    tmp["value_thousands"] = pd.to_numeric(tmp.get("value_thousands", 0.0), errors="coerce").fillna(0.0)

    def _row_detail(r: pd.Series) -> dict:
        shares = float(r.get("shares", 0.0))
        value_usd = float(r.get("value_thousands", 0.0)) * 1000.0
        implied_price = (value_usd / shares) if shares > 0 else 0.0
        cik = str(r.get("cik", ""))
        return {
            "cik": cik,
            "portfolio": _CIK_TO_PORTFOLIO.get(cik, f"CIK {cik.lstrip('0') or '0'}"),
            "weight_pct": round(float(r.get("weight", 0.0)) * 100.0, 4),
            "shares": int(round(shares)),
            "value_usd_mm": round(value_usd / 1_000_000.0, 3),
            "reported_implied_price": round(implied_price, 4),
        }

    by_cusip: dict[str, list[dict]] = {}
    by_ticker: dict[str, list[dict]] = {}

    for _, grp in tmp.groupby("cusip"):
        details = [_row_detail(r) for _, r in grp.sort_values("value_thousands", ascending=False).iterrows()]
        by_cusip[str(grp.iloc[0]["cusip"])] = details

    for t, grp in tmp[tmp["ticker"].astype(str).str.len() > 0].groupby("ticker"):
        details = [_row_detail(r) for _, r in grp.sort_values("value_thousands", ascending=False).iterrows()]
        by_ticker[str(t)] = details

    return by_cusip, by_ticker


def _status_from_weights(last_wt: float, pred_wt: float, delta_bps: int) -> str:
    if last_wt <= 0.01 and pred_wt > 0.05:
        return "NEW POSITION"
    if last_wt > 0.05 and pred_wt <= 0.01:
        return "EXITED"
    if delta_bps >= 10:
        return "ACCUMULATING"
    return "TRIMMING"


def _load_proactive_candidates(max_n: int = 10) -> list[dict]:
    if not _PROACTIVE_CANDIDATES_PATH.exists():
        return []
    try:
        payload = json.loads(_PROACTIVE_CANDIDATES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

    rows = payload.get("tft_injection", []) if isinstance(payload, dict) else []
    out = []
    for r in rows[:max_n]:
        t = str(r.get("ticker", "")).strip().upper()
        if not t:
            continue
        out.append(
            {
                "ticker": t,
                "flow_signal_z": float(r.get("flow_signal_z", 0.0) or 0.0),
                "thesis_conviction": float(r.get("thesis_conviction", 0.5) or 0.5),
            }
        )
    return out


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

_FEATURE_COLS = [
    "holding_streak", "weight_latest", "weight_change_1q", "weight_change_2q",
    "weight_trend", "port_pct_rank",
    "sector_flow_z", "sector_mom4w", "sector_mom13w",
    "vix", "yield_spread",
    "stock_ret_3m", "stock_ret_6m", "stock_vol_3m", "stock_rel_ret_3m",
]

_MIN_TRAIN_ROWS = 120


def _build_training_data(
    holdings: pd.DataFrame,
    feature_store: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build regression training pairs from all CIKs and consecutive period transitions.
    y is next-quarter weight change in percentage points.
    Also returns period_next for time-based backtesting splits.
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

                delta_pp = (w_next - w_curr) * 100.0

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
                    "cik": cik, "period": p_curr, "period_next": p_next,
                    "holding_streak": streak,
                    "weight_latest": w_curr,
                    "weight_change_1q": w_chg1,
                    "weight_change_2q": w_chg2,
                    "weight_trend": w_trend,
                    "port_pct_rank": port_rank,
                    **s_feats, **m_feats,
                    "stock_ret_3m": 0.0, "stock_ret_6m": 0.0,
                    "stock_vol_3m": 0.0, "stock_rel_ret_3m": 0.0,
                    "target_delta_pp": delta_pp,
                })

    if not rows:
        return pd.DataFrame(columns=_FEATURE_COLS), pd.Series(dtype=float), pd.Series(dtype=str)

    df = pd.DataFrame(rows)

    # Batch-fetch price data for all mapped tickers
    price_map = _fetch_price_features_bulk(list(all_tickers))
    for col in ("stock_ret_3m", "stock_ret_6m", "stock_vol_3m", "stock_rel_ret_3m"):
        df[col] = df["ticker"].map(
            lambda t, c=col: price_map.get(t, {}).get(c.replace("stock_", ""), 0.0)
        )

    X = df[_FEATURE_COLS].fillna(0.0)
    y = df["target_delta_pp"].astype(float)
    period_next = df["period_next"].astype(str)
    return X, y, period_next


def _train(X: pd.DataFrame, y: pd.Series) -> Optional[HistGradientBoostingRegressor]:
    if X.empty or len(y) < _MIN_TRAIN_ROWS:
        return None
    clf = HistGradientBoostingRegressor(
        max_iter=240,
        max_depth=6,
        learning_rate=0.04,
        min_samples_leaf=15,
        l2_regularization=0.05,
        random_state=42,
    )
    clf.fit(X, y)
    return clf


def _holdings_signature(holdings: pd.DataFrame, feature_store: pd.DataFrame) -> tuple:
    del feature_store  # retrain policy is tied to 13F filings, not weekly market updates
    latest_period = str(pd.to_datetime(holdings["period"]).max()) if not holdings.empty else ""
    latest_filing = str(pd.to_datetime(holdings["filing_date"]).max()) if not holdings.empty and "filing_date" in holdings.columns else ""
    fund_count = int(holdings["cik"].astype(str).nunique()) if not holdings.empty and "cik" in holdings.columns else 0
    return (len(holdings), latest_period, latest_filing, fund_count)


def _read_model_meta() -> dict:
    if not _MODEL_META_PATH.exists():
        return {}
    try:
        return json.loads(_MODEL_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_model_from_disk(signature: tuple) -> Optional[HistGradientBoostingRegressor]:
    if not _MODEL_ARTIFACT_PATH.exists():
        return None

    meta = _read_model_meta()
    if tuple(meta.get("signature", ())) != signature:
        return None

    try:
        clf = joblib.load(_MODEL_ARTIFACT_PATH)
        if isinstance(clf, HistGradientBoostingRegressor):
            _MODEL_CACHE["delta_scale"] = float(meta.get("delta_scale", 0.6) or 0.6)
            return clf
    except Exception:
        return None
    return None


def _persist_model(clf: Optional[HistGradientBoostingRegressor], signature: tuple, delta_scale: float) -> None:
    if clf is None:
        return
    try:
        _MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, _MODEL_ARTIFACT_PATH)
        _MODEL_META_PATH.write_text(
            json.dumps(
                {
                    "signature": list(signature),
                    "latest_period": str(signature[1])[:10] if len(signature) > 1 else "",
                    "artifact": str(_MODEL_ARTIFACT_PATH),
                    "model_type": "HistGradientBoostingRegressor",
                    "delta_scale": float(delta_scale),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[stock_predictor] Failed to persist model artifact: {e}")


def _get_or_train_model(holdings: pd.DataFrame, feature_store: pd.DataFrame) -> Optional[HistGradientBoostingRegressor]:
    signature = _holdings_signature(holdings, feature_store)
    if _MODEL_CACHE["signature"] == signature:
        return _MODEL_CACHE["model"]  # type: ignore[return-value]

    disk_model = _load_model_from_disk(signature)
    if disk_model is not None:
        _MODEL_CACHE["signature"] = signature
        _MODEL_CACHE["model"] = disk_model
        return disk_model

    X_train, y_train, _ = _build_training_data(holdings, feature_store)
    clf = _train(X_train, y_train)
    delta_scale = float(np.clip(y_train.std(ddof=0), 0.2, 2.5)) if len(y_train) else 0.6
    _MODEL_CACHE["signature"] = signature
    _MODEL_CACHE["model"] = clf
    _MODEL_CACHE["delta_scale"] = delta_scale
    _persist_model(clf, signature, delta_scale)
    return clf


def backtest_last_n_quarters(
    holdings: pd.DataFrame,
    feature_store: pd.DataFrame,
    n_quarters: int = 4,
) -> dict:
    """
    Walk-forward backtest on the last n predicted quarters.
    Train on periods strictly before each test period.
    """
    X, y, period_next = _build_training_data(holdings, feature_store)
    if X.empty or len(y) < _MIN_TRAIN_ROWS:
        return {
            "status": "insufficient_data",
            "n_test_periods": 0,
            "period_results": [],
            "summary": {},
        }

    periods = sorted(pd.Series(period_next).dropna().astype(str).unique())
    test_periods = periods[-n_quarters:]
    period_results: list[dict] = []

    for test_period in test_periods:
        train_mask = period_next < test_period
        test_mask = period_next == test_period
        if int(train_mask.sum()) < _MIN_TRAIN_ROWS or int(test_mask.sum()) < 10:
            continue

        model = _train(X.loc[train_mask], y.loc[train_mask])
        if model is None:
            continue

        pred = pd.Series(model.predict(X.loc[test_mask]), index=y.loc[test_mask].index).astype(float)
        actual = y.loc[test_mask].astype(float)
        err = pred - actual

        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(np.square(err))))
        deadband = 0.02
        pred_dir = np.where(pred > deadband, 1, np.where(pred < -deadband, -1, 0))
        true_dir = np.where(actual > deadband, 1, np.where(actual < -deadband, -1, 0))
        dir_acc = float(np.mean(pred_dir == true_dir))
        corr = float(np.corrcoef(pred, actual)[0, 1]) if len(pred) > 3 and np.std(pred) > 0 and np.std(actual) > 0 else 0.0

        period_results.append(
            {
                "test_period": str(test_period)[:10],
                "n_rows": int(test_mask.sum()),
                "mae_pp": round(mae, 4),
                "rmse_pp": round(rmse, 4),
                "directional_accuracy": round(dir_acc, 4),
                "corr": round(corr, 4),
            }
        )

    if not period_results:
        return {
            "status": "insufficient_data",
            "n_test_periods": 0,
            "period_results": [],
            "summary": {},
        }

    mae_vals = [r["mae_pp"] for r in period_results]
    rmse_vals = [r["rmse_pp"] for r in period_results]
    dir_vals = [r["directional_accuracy"] for r in period_results]
    corr_vals = [r["corr"] for r in period_results]

    return {
        "status": "ok",
        "n_test_periods": len(period_results),
        "period_results": period_results,
        "summary": {
            "avg_mae_pp": round(float(np.mean(mae_vals)), 4),
            "avg_rmse_pp": round(float(np.mean(rmse_vals)), 4),
            "avg_directional_accuracy": round(float(np.mean(dir_vals)), 4),
            "avg_corr": round(float(np.mean(corr_vals)), 4),
        },
    }


# ── Public interface ───────────────────────────────────────────────────────────

# Column definitions shown on hover in UI tables
COLUMN_DEFS = [
    {"field": "asset", "headerName": "Asset", "headerTooltip": "Security name from the 13F filing (company/issuer name)."},
    {"field": "ticker", "headerName": "Ticker", "headerTooltip": "Exchange ticker inferred from CUSIP/company mapping."},
    {"field": "sector", "headerName": "Sector", "headerTooltip": "Mapped economic sector proxy used in feature engineering."},
    {"field": "status", "headerName": "Status", "headerTooltip": "Action badge for fast scanning: NEW POSITION, ACCUMULATING, TRIMMING, or EXITED."},
    {"field": "portfolio_count", "headerName": "In Portfolios", "headerTooltip": "How many tracked fund portfolios currently hold this stock. Click row for full holder breakdown."},
    {"field": "thesis_conviction", "headerName": "Thesis Conviction", "headerTooltip": "A 0 to 1 strength score for how strongly current signals align with the strategy thesis; higher = stronger alignment."},
    {"field": "top_tft_driver", "headerName": "Top TFT Driver", "headerTooltip": "The #1 driver behind the prediction (for example momentum, flow pressure, or weight-trend behavior)."},
    {"field": "flow_signal", "headerName": "Flow Signal", "headerTooltip": "Institutional flow pressure regime. Strong positive z-scores indicate unusually aggressive buying activity."},
]


def _infer_top_driver(weight_trend: float, stock_ret_3m: float, rel_ret_3m: float, sector_flow_z: float) -> str:
    impacts = {
        "Weight Trend": weight_trend * 5.0,
        "Momentum 3M": stock_ret_3m * 1.8,
        "Relative Momentum": rel_ret_3m * 1.2,
        "Sector Flow": sector_flow_z * 0.8,
    }
    best = max(impacts.items(), key=lambda kv: abs(kv[1]))
    direction = "Bullish" if best[1] >= 0 else "Bearish"
    return f"{best[0]} ({direction})"


def _flow_regime(sector_flow_z: float, rel_momentum_pct: float = 0.0) -> str:
    # Prefer sector flow z-score when available; otherwise fall back to
    # stock relative momentum (% vs SPY) so regimes are still informative.
    if abs(sector_flow_z) >= 0.05:
        if sector_flow_z >= 0.25:
            return "Inflow"
        if sector_flow_z <= -0.25:
            return "Outflow"
        return "Neutral"

    if rel_momentum_pct >= 1.25:
        return "Inflow"
    if rel_momentum_pct <= -1.25:
        return "Outflow"
    return "Neutral"


def generate_signals(
    cik: str,
    holdings: pd.DataFrame,
    feature_store: pd.DataFrame,
    top_n: int = 20,
    include_candidates: bool = True,
) -> dict:
    """
    Unified signal generation combining current holdings + predicted new buys.

    Returns frontend-compatible dict:
      {
        "data": [...],
        "columnsDefs": [...]
      }

        Thesis conviction ∈ [-1, +1] from predicted weight delta (pp) scaled by
        training-distribution volatility.
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
    display_n = int(np.clip(top_n, 10, 20))
    current = fund[fund["period"] == latest_period].nlargest(max(display_n * 3, 30), "value_thousands")

    latest_all = eq[eq["period"] == latest_period].copy()
    latest_all["ticker"] = latest_all.apply(
        lambda r: get_ticker(str(r.get("name", "")), str(r.get("cusip", ""))) or "",
        axis=1,
    )
    latest_all = latest_all[
        latest_all.apply(lambda r: _is_stock_security(str(r.get("name", "")), str(r.get("ticker", ""))), axis=1)
    ]
    cusip_to_ciks = latest_all.groupby("cusip")["cik"].apply(lambda s: set(s.astype(str))).to_dict()
    ticker_to_ciks = latest_all.groupby("ticker")["cik"].apply(lambda s: set(s.astype(str))).to_dict()
    cusip_to_positions, ticker_to_positions = _build_portfolio_position_maps(latest_all)

    # Rank by row/value so lookup is always scalar even when CUSIP repeats across funds.
    ranked_by_row = current["value_thousands"].astype(float).rank(pct=True)

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
        if not _is_stock_security(name, ticker):
            continue
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
        port_rank = float(ranked_by_row.get(idx, 0.5))

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

    # ── Predict next-quarter weight deltas ────────────────────────────────────
    if clf is not None and rows:
        X_pred = pd.DataFrame([r["_features"] for r in rows])[_FEATURE_COLS].fillna(0.0)
        pred_delta_pp = clf.predict(X_pred)
        scale = float(_MODEL_CACHE.get("delta_scale", 0.6) or 0.6)
        for i, row in enumerate(rows):
            dpp = float(pred_delta_pp[i])
            row["pred_delta_pp"] = round(dpp, 4)
            row["signal"] = round(float(np.tanh(dpp / max(scale, 1e-6))), 3)
    else:
        # Fallback: simple heuristic when no model
        for row in rows:
            f = row["_features"]
            dpp = (f["weight_trend"] * 100.0) + (f["stock_ret_3m"] * 20.0) + (f["sector_flow_z"] * 8.0)
            row["pred_delta_pp"] = round(float(dpp), 4)
            row["signal"] = round(float(np.tanh(dpp / 0.6)), 3)

    # ── Derive user-facing model columns ──────────────────────────────────────
    for row in rows:
        last_wt = float(row.get("current_weight_pct", 0.0))
        pred_delta_pp = float(row.get("pred_delta_pp", 0.0))
        signal = float(row.get("signal", 0.0))
        pred_wt = max(0.0, last_wt + pred_delta_pp)
        weight_delta_bps = round((pred_wt - last_wt) * 100.0, 0)
        sector_flow_z = float(row.get("sector_flow_z", 0.0))
        momentum_3m = float(row.get("momentum_3m", 0.0)) / 100.0
        rel_momentum = float(row.get("rel_momentum", 0.0)) / 100.0

        row["asset"] = row.get("name", "")
        portfolio_ciks = set(cusip_to_ciks.get(str(row.get("cusip", "")), set()))
        if row.get("ticker"):
            portfolio_ciks |= set(ticker_to_ciks.get(str(row.get("ticker", "")), set()))
        row["portfolio"] = _portfolio_names_for(portfolio_ciks)
        details = list(cusip_to_positions.get(str(row.get("cusip", "")), []))
        if row.get("ticker"):
            details = ticker_to_positions.get(str(row.get("ticker", "")), details)
        row["portfolio_positions"] = details
        row["portfolio_count"] = len(details)
        row["pred_wt_pct"] = round(pred_wt, 2)
        row["weight_delta_bps"] = int(weight_delta_bps)
        row["status"] = _status_from_weights(last_wt, pred_wt, int(weight_delta_bps))
        row["thesis_conviction"] = round(abs(signal), 3)
        row["top_tft_driver"] = _infer_top_driver(float(row.get("weight_trend", 0.0)) / 100.0, momentum_3m, rel_momentum, sector_flow_z)
        row["flow_signal"] = _flow_regime(sector_flow_z, rel_momentum * 100.0)
        row["last_13f_wt_pct"] = round(last_wt, 2)

    # ── Add predicted-buy candidates ───────────────────────────────────────────
    if include_candidates:
        counts = latest_all.groupby("cusip")["cik"].nunique().to_dict()

        current_cusips = {r["cusip"] for r in rows}
        # Also pass known tickers so candidates don't duplicate held positions
        current_tickers_set = {r["ticker"] for r in rows if r["ticker"]}
        candidate_rows = _predict_new_buys(
            cik, holdings, feature_store, clf, current_cusips, price_map, m_feats,
            n=max(display_n, 20), exclude_tickers=current_tickers_set,
            cusip_to_ciks=cusip_to_ciks, ticker_to_ciks=ticker_to_ciks,
            cusip_to_positions=cusip_to_positions, ticker_to_positions=ticker_to_positions,
        )
        for c in candidate_rows:
            c["fund_overlap_count"] = int(counts.get(str(c.get("cusip", "")), 0))
        # Keep candidates that are portfolio-alike: present across tracked funds.
        candidate_rows = [c for c in candidate_rows if int(c.get("fund_overlap_count", 0)) >= 2]
        rows.extend(candidate_rows)

        # Inject proactive tripwire ideas (zero-weight seeds) when available.
        proactive = _load_proactive_candidates(max_n=10)
        seen_tickers = {str(r.get("ticker", "")).strip().upper() for r in rows if r.get("ticker")}
        for p in proactive:
            t = p["ticker"]
            if t in seen_tickers:
                continue
            flow_z = float(p.get("flow_signal_z", 0.0))
            conv = float(np.clip(p.get("thesis_conviction", 0.5), 0.0, 1.0))
            pred_wt = float(np.clip((conv * 0.9) + (max(flow_z, 0.0) * 0.25), 0.1, 2.5))
            delta_bps = int(round(pred_wt * 100.0, 0))

            pos_details = list(ticker_to_positions.get(t, []))
            rows.append(
                {
                    "cusip": "",
                    "name": t,
                    "asset": t,
                    "ticker": t,
                    "sector": "Other",
                    "source": "proactive_candidate",
                    "signal": round((conv * 2.0) - 1.0, 3),
                    "pred_delta_pp": pred_wt,
                    "pred_wt_pct": round(pred_wt, 2),
                    "weight_delta_bps": delta_bps,
                    "status": "NEW POSITION",
                    "thesis_conviction": round(conv, 3),
                    "top_tft_driver": "Tripwire Flow Anomaly",
                    "flow_signal": _flow_regime(flow_z, 0.0),
                    "last_13f_wt_pct": 0.0,
                    "portfolio": _portfolio_names_for(set(ticker_to_ciks.get(t, set()))),
                    "portfolio_positions": pos_details,
                    "portfolio_count": len(pos_details),
                    "fund_overlap_count": len(ticker_to_ciks.get(t, set())),
                }
            )
            seen_tickers.add(t)

    rows = [r for r in rows if _is_stock_security(str(r.get("asset", r.get("name", ""))), str(r.get("ticker", "")))]

    # Sort by signal descending, clean up internal fields
    for row in rows:
        row.pop("_features", None)
    rows.sort(
        key=lambda r: (
            -float(r.get("thesis_conviction", r.get("signal", 0.0))),
            -float(r.get("pred_wt_pct", 0.0)),
            -int(r.get("fund_overlap_count", 0)),
        )
    )

    proactive_rows = [r for r in rows if str(r.get("source", "")) == "proactive_candidate"]
    top_rows = rows[:display_n]
    if proactive_rows and not any(str(r.get("source", "")) == "proactive_candidate" for r in top_rows):
        top_rows = top_rows[:-1] + [proactive_rows[0]] if len(top_rows) >= 1 else [proactive_rows[0]]

    rows = top_rows

    return {"data": rows, "columnsDefs": COLUMN_DEFS}


def _predict_new_buys(
    cik: str,
    holdings: pd.DataFrame,
    feature_store: pd.DataFrame,
    clf: Optional[HistGradientBoostingRegressor],
    exclude_cusips: set[str],
    existing_price_map: dict,
    m_feats: dict,
    n: int = 20,
    exclude_tickers: Optional[set[str]] = None,
    cusip_to_ciks: Optional[dict[str, set[str]]] = None,
    ticker_to_ciks: Optional[dict[str, set[str]]] = None,
    cusip_to_positions: Optional[dict[str, list[dict]]] = None,
    ticker_to_positions: Optional[dict[str, list[dict]]] = None,
) -> list[dict]:
    """
    Identify stocks not currently held that match the fund's historical buy pattern.
    Universe: stocks held by OTHER tracked funds but not by this one.
    """
    eq = holdings[holdings["put_call"].fillna("") == ""].copy()
    eq = eq.sort_values("value_thousands", ascending=False).drop_duplicates(
        subset=["cik", "period", "cusip"]
    )

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

    # Build metadata and feature rows together, then batch-predict
    meta_rows = []
    feat_rows = []
    for _, row in candidates.iterrows():
        name   = str(row.get("name", ""))
        cusip  = str(row.get("cusip", ""))
        ticker = get_ticker(name, cusip) or ""
        if not _is_stock_security(name, ticker):
            continue
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
        pred_delta_pp_all = clf.predict(X_all)
        scale = float(_MODEL_CACHE.get("delta_scale", 0.6) or 0.6)
    else:
        pred_delta_pp_all = None

    cand_rows = []
    for i, meta in enumerate(meta_rows):
        pf = meta["pf"]
        s_feats = meta["s_feats"]
        if pred_delta_pp_all is not None:
            dpp = float(pred_delta_pp_all[i])
            signal = round(float(np.tanh(dpp / max(scale, 1e-6))), 3)
        else:
            dpp = float((pf.get("ret_3m", 0.0) * 20.0) + (s_feats.get("sector_flow_z", 0.0) * 8.0))
            signal = round(float(np.tanh(dpp / 0.6)), 3)

        if dpp > 0.04:
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
                "pred_delta_pp":      round(dpp, 4),
                "signal":             signal,
            })

    for row in cand_rows:
        last_wt = float(row.get("current_weight_pct", 0.0))
        pred_delta_pp = max(0.0, float(row.get("pred_delta_pp", 0.0)))
        signal = float(row.get("signal", 0.0))
        pred_wt = max(0.0, pred_delta_pp)
        weight_delta_bps = round((pred_wt - last_wt) * 100.0, 0)
        sector_flow_z = float(row.get("sector_flow_z", 0.0))
        momentum_3m = float(row.get("momentum_3m", 0.0)) / 100.0
        rel_momentum = float(row.get("rel_momentum", 0.0)) / 100.0

        row["asset"] = row.get("name", "")
        _cusip_map = cusip_to_ciks or {}
        _ticker_map = ticker_to_ciks or {}
        portfolio_ciks = set(_cusip_map.get(str(row.get("cusip", "")), set()))
        if row.get("ticker"):
            portfolio_ciks |= set(_ticker_map.get(str(row.get("ticker", "")), set()))
        row["portfolio"] = _portfolio_names_for(portfolio_ciks)
        _cusip_pos = cusip_to_positions or {}
        _ticker_pos = ticker_to_positions or {}
        details = list(_cusip_pos.get(str(row.get("cusip", "")), []))
        if row.get("ticker"):
            details = _ticker_pos.get(str(row.get("ticker", "")), details)
        row["portfolio_positions"] = details
        row["portfolio_count"] = len(details)
        row["pred_wt_pct"] = round(pred_wt, 2)
        row["weight_delta_bps"] = int(weight_delta_bps)
        row["status"] = _status_from_weights(last_wt, pred_wt, int(weight_delta_bps))
        row["thesis_conviction"] = round(abs(signal), 3)
        row["top_tft_driver"] = _infer_top_driver(0.0, momentum_3m, rel_momentum, sector_flow_z)
        row["flow_signal"] = _flow_regime(sector_flow_z, rel_momentum * 100.0)
        row["last_13f_wt_pct"] = round(last_wt, 2)

    seen: set[str] = set()
    deduped: list[dict] = []
    for r in sorted(cand_rows, key=lambda x: -x["signal"]):
        key = r["ticker"] if r["ticker"] else r["cusip"]
        if key and key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped[:n]
