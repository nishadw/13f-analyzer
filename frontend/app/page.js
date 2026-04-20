"use client";

import { useEffect, useMemo, useState } from "react";

const FUND_PRIORITY = [
  { aliases: ["bridgewater associates", "bridgewater"] },
  { aliases: ["citadel"] },
  { aliases: ["pershing square"] },
  { ciks: ["1037389"], aliases: ["renaissance technologies", "renaissance tech"] },
  { ciks: ["2045724"], aliases: ["situational awareness", "situational awareness lp", "leopold's situational awareness"] },
];

function normalizeText(v) {
  return String(v || "").toLowerCase().replace(/[^a-z0-9 ]/g, " ").replace(/\s+/g, " ").trim();
}

function uniqueByKey(rows, key) {
  const seen = new Set();
  const out = [];
  for (const row of (Array.isArray(rows) ? rows : [])) {
    const k = String(row?.[key] || "").trim().toUpperCase();
    if (!k || seen.has(k)) continue;
    seen.add(k);
    out.push(row);
  }
  return out;
}

function pickPriorityFunds(allFunds) {
  const rows = Array.isArray(allFunds) ? allFunds : [];
  const used = new Set();
  const selected = [];
  for (const priority of FUND_PRIORITY) {
    const match = rows.find((f) => {
      if (!f?.cik || used.has(f.cik)) return false;
      const cikNorm = normalizeText(f.cik);
      const name = normalizeText(f.name);
      const cikMatch = Array.isArray(priority.ciks) && priority.ciks.some((cik) => cikNorm === normalizeText(cik));
      const nameMatch = Array.isArray(priority.aliases) && priority.aliases.some((alias) => name.includes(normalizeText(alias)));
      return cikMatch || nameMatch;
    });
    if (match) { used.add(match.cik); selected.push(match); }
  }
  if (selected.length === 0) return [...rows].sort((a, b) => String(a.name).localeCompare(String(b.name)));
  const remaining = rows.filter((f) => !used.has(f.cik)).sort((a, b) => String(a.name).localeCompare(String(b.name)));
  return [...selected, ...remaining];
}

function fmtMoney(v)    { return `$${Number(v || 0).toLocaleString(undefined, { maximumFractionDigits: 2 })}M`; }
function fmtPct(v, d=2) { return `${Number(v || 0).toFixed(d)}%`; }
function fmtCik(v) { return String(v || "").replace(/^0+/, "") || "0"; }
function displayName(name) {
  const n = String(name || "").trim();
  if (/situational awareness/i.test(n)) return "Situational Awareness";
  return n;
}

function fmtSignal(v) {
  const n = Number(v || 0);
  const cls = n >= 0.1 ? "signal-positive" : n <= -0.1 ? "signal-negative" : "signal-neutral";
  return <span className={cls}>{n >= 0 ? "+" : ""}{n.toFixed(3)}</span>;
}

// ── Summary: top conviction signals ──────────────────────────────────────────

function ConvictionSection({ rows, emptyMsg, fundNames }) {
  if (!rows || rows.length === 0) return <p className="meta">{emptyMsg}</p>;
  return (
    <table>
      <thead>
        <tr>
          <th>Ticker</th>
          <th>Company</th>
          <th title="ML conviction score: +1 = strong increase, -1 = strong decrease">Signal</th>
          <th title="3-month price return %">Mom 3M</th>
          <th title="6-month price return %">Mom 6M</th>
          <th title="Sector ETF proxy">Sector</th>
          <th title="Which tracked fund drove this signal">Top Fund</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={`${r.ticker || r.cusip}-${i}`}>
            <td><strong>{r.ticker || "-"}</strong></td>
            <td>{r.name || "-"}</td>
            <td>{fmtSignal(r.signal)}</td>
            <td>{r.momentum_3m != null ? fmtPct(r.momentum_3m) : "-"}</td>
            <td>{r.momentum_6m != null ? fmtPct(r.momentum_6m) : "-"}</td>
            <td>{r.sector || "-"}</td>
            <td style={{ fontSize: "0.8em", color: "#64748b" }}>
              {fundNames?.[r.fund_cik] || r.fund_cik || "-"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function TrackedFunds({ funds, onFundClick }) {
  return (
    <div className="card">
      <h2>Tracked Funds</h2>
      <p className="meta">Click a row to open that fund's holdings tab.</p>
      <table>
        <thead>
          <tr>
            <th>Fund</th><th>CIK</th><th>Latest Filing</th><th>Positions</th><th>Total Value</th>
          </tr>
        </thead>
        <tbody>
          {funds.map((f) => (
            <tr key={f.cik} className="clickable-row" onClick={() => onFundClick?.(f.cik)}>
              <td>{displayName(f.name)}</td>
              <td>{fmtCik(f.cik)}</td>
              <td>{f.latest_period || "-"}</td>
              <td>{Number(f.num_positions || 0).toLocaleString()}</td>
              <td>{fmtMoney(f.total_value_usd_mm)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Per-fund holdings ─────────────────────────────────────────────────────────

function HoldingsTable({ fund, rows, loading, onRowClick, selectedStock }) {
  return (
    <div className="card">
      <h2>{displayName(fund.name)} — Holdings</h2>
      <p className="meta">CIK: {fmtCik(fund.cik)} | Last Filing: {fund.latest_period || "N/A"}</p>
      {loading ? <p>Loading holdings…</p> : rows.length === 0 ? <p>No holdings found.</p> : (
        <table>
          <thead>
            <tr>
              <th>Rank</th><th>Name</th><th>Ticker</th><th>CUSIP</th>
              <th>Weight</th><th>Value</th><th>Shares</th><th>Sector</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={`${fund.cik}-${r.rank}-${r.cusip}`}
                className={`clickable-row ${selectedStock?.cusip === r.cusip ? "selected" : ""}`}
                onClick={() => onRowClick?.(r)}
              >
                <td>{r.rank}</td>
                <td>{r.name}</td>
                <td>{r.ticker || "-"}</td>
                <td>{r.cusip || "-"}</td>
                <td>{fmtPct(r.weight_pct)}</td>
                <td>{fmtMoney(r.value_usd_mm)}</td>
                <td>{Number(r.shares || 0).toLocaleString()}</td>
                <td>{r.sector || "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function StockHistogram({ history, loading, stock }) {
  const values  = Array.isArray(history?.values)  ? history.values  : [];
  const periods = Array.isArray(history?.periods) ? history.periods : [];
  const maxVal  = Math.max(1, ...values.map((v) => Number(v || 0)));

  return (
    <div className="card history-card">
      <h2>4-Filing Size Histogram</h2>
      <p className="meta">
        {stock ? `${stock.name || "Selected stock"}${stock.ticker ? ` (${stock.ticker})` : ""}` : "Click a holding row to inspect its weight history."}
      </p>
      {loading ? <p>Loading…</p> : !history || values.length === 0 ? (
        <p>Click any holding to view its weight over the past 4 filings.</p>
      ) : (
        <>
          <div className="histogram-shell">
            <div className="histogram-boxes">
              {values.map((value, i) => {
                const heightPct = Math.max(18, Math.round((Number(value || 0) / maxVal) * 100));
                const delta = i === 0 ? 0 : Number((value - values[i - 1]).toFixed(2));
                return (
                  <div className="histogram-box-wrap" key={`${periods[i] || i}-${value}`}>
                    <div className="histogram-label">{periods[i] || `Q${i + 1}`}</div>
                    <div className="histogram-box-track">
                      <div className="histogram-box" style={{ height: `${heightPct}%` }}>
                        <span>{fmtPct(value)}</span>
                      </div>
                    </div>
                    <div className={`histogram-change ${delta >= 0 ? "positive" : "negative"}`}>
                      {i === 0 ? "Start" : `${delta >= 0 ? "+" : ""}${fmtPct(delta)}`}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
          <div className="history-summary">
            <span>Increase total: {fmtPct(history.increase_total || 0)}</span>
            <span>Decrease total: {fmtPct(history.decrease_total || 0)}</span>
            <span>Net change: {fmtPct(history.net_change || 0)}</span>
          </div>
        </>
      )}
    </div>
  );
}

// ── Models tab ────────────────────────────────────────────────────────────────

const SIGNAL_TOOLTIPS = {
  name:               "Company name as reported in the 13F-HR filing",
  ticker:             "Exchange ticker symbol inferred from CUSIP or company name",
  signal:             "ML conviction score ∈ [−1,+1]. +1 = model strongly predicts fund will grow this position; −1 = likely reduction or exit. tanh((P(increase)−P(decrease))×2.5) from GBC trained on 4-quarter 13F transitions",
  current_weight_pct: "Current portfolio weight % from most recent 13F. 0% = predicted new-buy candidate not yet held.",
  sector:             "Sector ETF proxy (SMH=Semiconductors, XLK=Technology, XLF=Financials, XLV=Healthcare…)",
  momentum_3m:        "Stock 3-month price return % (yfinance weekly)",
  momentum_6m:        "Stock 6-month price return %",
  rel_momentum:       "3M return relative to SPY. Positive = outperforming the broad market.",
  holding_streak:     "Consecutive quarters this fund has held the position. Longer = higher manager conviction.",
  weight_trend:       "Average quarter-over-quarter weight change over last 4 filings. Positive = consistently adding.",
  sector_flow_z:      "Z-score of sector ETF net flows vs. 52-week history. Above +1 = unusual institutional inflows.",
  source:             "held = in current 13F portfolio; candidate = predicted new-buy from historical patterns",
  cusip:              "CUSIP — unique SEC identifier for the security",
};

function SignalsTable({ rows, loading, fundName }) {
  if (loading) return <div className="card"><p>Running GBC model — may take 30–60 s on first load…</p></div>;
  if (!rows || rows.length === 0) return <div className="card"><p>No signal data available for this fund yet.</p></div>;

  const cols = Object.entries(SIGNAL_TOOLTIPS);

  return (
    <div className="card">
      <h2>{fundName} — Stock Signals</h2>
      <p className="meta">
        GradientBoosting model on 4-quarter 13F transitions + yfinance price data. Signal ∈ [−1,+1].
        <strong> Candidates</strong> (source=candidate) are predicted new buys not currently held.
        Hover any column header for its definition.
      </p>
      <table>
        <thead>
          <tr>
            {cols.map(([col, tip]) => (
              <th key={col} title={tip}>
                {col === "current_weight_pct" ? "Weight %" :
                 col === "momentum_3m"        ? "Mom 3M %" :
                 col === "momentum_6m"        ? "Mom 6M %" :
                 col === "rel_momentum"       ? "Rel Mom %" :
                 col === "holding_streak"     ? "Qtrs Held" :
                 col === "weight_trend"       ? "Wt Trend" :
                 col === "sector_flow_z"      ? "Sector Flow Z" :
                 col.charAt(0).toUpperCase() + col.slice(1)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={`${r.cusip || r.ticker || r.name}-${i}`}>
              <td>{r.name || "-"}</td>
              <td><strong>{r.ticker || "-"}</strong></td>
              <td>{fmtSignal(r.signal)}</td>
              <td>{fmtPct(r.current_weight_pct)}</td>
              <td>{r.sector || "-"}</td>
              <td>{r.momentum_3m != null ? fmtPct(r.momentum_3m) : "-"}</td>
              <td>{r.momentum_6m != null ? fmtPct(r.momentum_6m) : "-"}</td>
              <td>{r.rel_momentum != null ? fmtPct(r.rel_momentum) : "-"}</td>
              <td>{r.holding_streak ?? "-"}</td>
              <td>{r.weight_trend != null ? Number(r.weight_trend).toFixed(3) : "-"}</td>
              <td>{r.sector_flow_z != null ? Number(r.sector_flow_z).toFixed(2) : "-"}</td>
              <td>
                <span className={`pill ${r.source === "candidate" ? "hold" : "buy"}`}>
                  {r.source || "held"}
                </span>
              </td>
              <td><code style={{ fontSize: "0.75em" }}>{r.cusip || "-"}</code></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function HomePage() {
  const [funds, setFunds]           = useState([]);
  const [activeView, setActiveView] = useState("summary");
  const [fundDataCache, setFundDataCache] = useState({});

  // Fund tab state
  const [holdingsRows, setHoldingsRows]   = useState([]);
  const [selectedStock, setSelectedStock] = useState(null);
  const [stockHistory, setStockHistory]   = useState(null);
  const [loadingStockHistory, setLoadingStockHistory] = useState(false);
  const [loadingTabData, setLoadingTabData] = useState(false);

  // Models tab state
  const [modelsCik, setModelsCik]       = useState("");
  const [modelsData, setModelsData]     = useState({ data: [], columnsDefs: [] });
  const [loadingModels, setLoadingModels] = useState(false);

  const [loadingFunds, setLoadingFunds] = useState(true);
  const [error, setError]               = useState("");

  const selectedFunds = useMemo(() => pickPriorityFunds(funds), [funds]);
  const activeFund    = useMemo(() => selectedFunds.find((f) => f.cik === activeView) || null, [selectedFunds, activeView]);
  const isFundView    = activeFund !== null;

  // Load fund list
  useEffect(() => {
    async function load() {
      try {
        setLoadingFunds(true);
        const res  = await fetch("api/funds", { cache: "no-store" });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Failed to load funds");
        setFunds(Array.isArray(data) ? data : []);
      } catch (e) {
        setError(String(e));
      } finally {
        setLoadingFunds(false);
      }
    }
    load();
  }, []);

  // Default models CIK
  useEffect(() => {
    if (!selectedFunds.length) return;
    if (!modelsCik || !selectedFunds.some((f) => f.cik === modelsCik)) {
      const first = selectedFunds.find((f) => Number(f.num_positions || 0) > 0) || selectedFunds[0];
      if (first) setModelsCik(first.cik);
    }
  }, [selectedFunds, modelsCik]);

  // Reset stock selection when switching tabs
  useEffect(() => {
    setSelectedStock(null);
    setStockHistory(null);
  }, [activeView]);

  // Guard invalid active view after funds load
  useEffect(() => {
    if (!selectedFunds.length) return;
    const valid = ["summary", "models"].includes(activeView) || selectedFunds.some((f) => f.cik === activeView);
    if (!valid) {
      const first = selectedFunds.find((f) => Number(f.num_positions || 0) > 0) || selectedFunds[0];
      if (first) setActiveView(first.cik);
    }
  }, [selectedFunds, activeView]);

  // Load holdings for fund tab
  useEffect(() => {
    if (!activeView || !isFundView) return;
    const cached = fundDataCache[activeView];
    if (cached?.holdingsRows) { setHoldingsRows(cached.holdingsRows); return; }

    async function load() {
      try {
        setLoadingTabData(true);
        const res  = await fetch(`api/holdings?cik=${encodeURIComponent(activeView)}&top_n=25`, { cache: "no-store" });
        const data = await res.json();
        const rows = uniqueByKey(Array.isArray(data) ? data : [], "cusip");
        setHoldingsRows(rows);
        setFundDataCache((prev) => ({ ...prev, [activeView]: { ...(prev[activeView] || {}), holdingsRows: rows } }));
      } catch (e) {
        setError(String(e));
      } finally {
        setLoadingTabData(false);
      }
    }
    load();
  }, [activeView, isFundView, fundDataCache]);

  // Load models signals
  useEffect(() => {
    if (!modelsCik) return;
    const key = `models:${modelsCik}`;
    if (fundDataCache[key]) { setModelsData(fundDataCache[key]); return; }

    async function load() {
      try {
        setLoadingModels(true);
        const res  = await fetch(`api/recommendations?cik=${encodeURIComponent(modelsCik)}&top_n=30`, { cache: "no-store" });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Failed");
        const result = { data: Array.isArray(data?.data) ? data.data : [], columnsDefs: data?.columnsDefs || [] };
        setModelsData(result);
        setFundDataCache((prev) => ({ ...prev, [key]: result }));
      } catch (e) {
        setError(String(e));
      } finally {
        setLoadingModels(false);
      }
    }
    load();
  }, [modelsCik, fundDataCache]);

  async function loadStockHistory(stock) {
    if (!stock || !isFundView) return;
    const key = `${activeView}:${String(stock.cusip || stock.ticker || stock.name || "").toUpperCase()}`;
    if (fundDataCache[key]) { setSelectedStock(stock); setStockHistory(fundDataCache[key]); return; }
    try {
      setLoadingStockHistory(true);
      setSelectedStock(stock);
      const params = new URLSearchParams({ cik: activeView, cusip: stock.cusip || "", name: stock.name || "", ticker: stock.ticker || "" });
      const res  = await fetch(`api/stock-history?${params}`, { cache: "no-store" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed");
      setStockHistory(data);
      setFundDataCache((prev) => ({ ...prev, [key]: data }));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoadingStockHistory(false);
    }
  }

  const modelsFund = selectedFunds.find((f) => f.cik === modelsCik);

  return (
    <main className="container">
      <header>
        <h1>13F Analyzer</h1>
        <p>Institutional portfolio tracker — 13F filings + ML signals.</p>
      </header>

      {error && <div className="error">{error}</div>}

      {loadingFunds ? <p>Loading dashboard…</p> : (
        <>
          {/* ── Tab bar ─────────────────────────────────────────────────────── */}
          <div className="card tabs-card">
            <div className="tabs">
              {/* Summary */}
              <button
                type="button"
                className={`tab-btn ${activeView === "summary" ? "active" : ""}`}
                onClick={() => setActiveView("summary")}
              >
                Summary
              </button>

              {/* Models */}
              <button
                type="button"
                className={`tab-btn ${activeView === "models" ? "active" : ""}`}
                onClick={() => setActiveView("models")}
              >
                Models
              </button>

              {/* Funds dropdown */}
              <div className="tab-dropdown-wrapper">
                <button
                  type="button"
                  className={`tab-btn ${isFundView ? "active" : ""}`}
                >
                  Funds ▾
                </button>
                <div className="tab-dropdown">
                  {selectedFunds.map((fund) => (
                    <button
                      key={fund.cik}
                      type="button"
                      className={`tab-dropdown-item ${activeView === fund.cik ? "active" : ""}`}
                      onClick={() => setActiveView(fund.cik)}
                    >
                      {displayName(fund.name)}
                      {Number(fund.num_positions || 0) === 0 ? " · empty" : ""}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* ── Tab content ─────────────────────────────────────────────────── */}
          {activeView === "summary" ? (
            <>
              <TrackedFunds funds={selectedFunds} onFundClick={(cik) => setActiveView(cik)} />
            </>

          ) : activeView === "models" ? (
            <div>
              <div className="card" style={{ paddingBottom: "8px" }}>
                <h2>ML Stock Signals</h2>
                <p className="meta">Select a fund. Hover any column header for its definition.</p>
                <div style={{ display: "flex", alignItems: "center", gap: "10px", marginTop: "6px" }}>
                  <label htmlFor="models-fund-select"><strong>Fund</strong></label>
                  <select
                    id="models-fund-select"
                    value={modelsCik}
                    onChange={(e) => setModelsCik(e.target.value)}
                  >
                    {selectedFunds.map((f) => (
                      <option key={f.cik} value={f.cik}>{displayName(f.name)}</option>
                    ))}
                  </select>
                </div>
              </div>
              <SignalsTable rows={modelsData.data} loading={loadingModels} fundName={modelsFund?.name || modelsCik} />
            </div>

          ) : activeFund ? (
            <>
              <HoldingsTable
                fund={activeFund}
                rows={holdingsRows}
                loading={loadingTabData}
                onRowClick={loadStockHistory}
                selectedStock={selectedStock}
              />
              <StockHistogram stock={selectedStock} history={stockHistory} loading={loadingStockHistory} />
            </>

          ) : (
            <div className="card"><p>No fund data available.</p></div>
          )}
        </>
      )}
    </main>
  );
}
