"use client";

import { useEffect, useMemo, useState } from "react";

const GITHUB_URL = "https://github.com/nishadw/13f-analyzer";

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

function sortFunds(funds) {
  return [...(Array.isArray(funds) ? funds : [])].sort(
    (a, b) => Number(b.total_value_usd_mm || 0) - Number(a.total_value_usd_mm || 0)
  );
}

function fmtMoney(v)    { return `$${Number(v || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}M`; }
function fmtPct(v, d=2) { return `${Number(v || 0).toFixed(d)}%`; }
function fmtCik(v) { return String(v || "").replace(/^0+/, "") || "0"; }
function displayName(name) { return String(name || "").trim() || "—"; }

function fmtSignal(v) {
  const n = Number(v || 0);
  const cls = n >= 0.1 ? "signal-positive" : n <= -0.1 ? "signal-negative" : "signal-neutral";
  return <span className={cls}>{n >= 0 ? "+" : ""}{n.toFixed(3)}</span>;
}

// ── Nav ───────────────────────────────────────────────────────────────────────

function TopNav({ activeView, setActiveView, selectedFunds, isFundView }) {
  return (
    <nav className="topnav">
      <div className="topnav-inner">
        <a className="topnav-brand" href="/">
          <div className="topnav-logo">13F</div>
          <span className="topnav-title">13F Analyzer</span>
        </a>
        <div className="topnav-tabs">
          <button
            type="button"
            className={`tab-btn ${activeView === "summary" ? "active" : ""}`}
            onClick={() => setActiveView("summary")}
          >
            Summary
          </button>
          <button
            type="button"
            className={`tab-btn ${activeView === "models" ? "active" : ""}`}
            onClick={() => setActiveView("models")}
          >
            Models
          </button>
          <div className="tab-dropdown-wrapper">
            <button type="button" className={`tab-btn ${isFundView ? "active" : ""}`}>
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
                  {Number(fund.num_positions || 0) === 0 ? " · no data" : ""}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}

// ── Hero strip ─────────────────────────────────────────────────────────────────

function Hero({ subtitle }) {
  return (
    <div className="hero">
      <div className="hero-inner">
        <h1>Institutional Portfolio Intelligence</h1>
        <p>{subtitle || "Track 13F filings from top hedge funds, analyse position changes, and view ML-driven conviction signals in real time."}</p>
      </div>
    </div>
  );
}

// ── Footer ─────────────────────────────────────────────────────────────────────

function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <div className="footer-brand">
          <div className="name">13F Analyzer</div>
          <p>Open-source tool for tracking institutional 13F filings, analysing position changes, and generating ML conviction signals across top hedge funds.</p>
        </div>
        <div className="footer-links">
          <a href={GITHUB_URL} target="_blank" rel="noopener noreferrer">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844a9.59 9.59 0 0 1 2.504.337c1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.02 10.02 0 0 0 22 12.017C22 6.484 17.522 2 12 2z"/></svg>
            GitHub — nishadw/13f-analyzer
          </a>
          <a href={`${GITHUB_URL}/issues`} target="_blank" rel="noopener noreferrer">
            Issues &amp; feedback
          </a>
          <a href="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=13F" target="_blank" rel="noopener noreferrer">
            SEC EDGAR 13F Filings
          </a>
        </div>
      </div>
      <div className="footer-bottom">
        <span>© {new Date().getFullYear()} 13F Analyzer. Data sourced from SEC EDGAR — for informational purposes only, not investment advice.</span>
        <a href={GITHUB_URL} target="_blank" rel="noopener noreferrer">View on GitHub ↗</a>
      </div>
    </footer>
  );
}

// ── Summary tab ────────────────────────────────────────────────────────────────

function TrackedFunds({ funds, onFundClick }) {
  const totalFunds = funds.length;
  const totalPositions = funds.reduce((s, f) => s + Number(f.num_positions || 0), 0);

  return (
    <>
      <div className="stat-row">
        <div className="stat-chip">
          <div className="label">Tracked Funds</div>
          <div className="value">{totalFunds}</div>
        </div>
        <div className="stat-chip">
          <div className="label">Total Positions</div>
          <div className="value">{totalPositions.toLocaleString()}</div>
        </div>
        <div className="stat-chip">
          <div className="label">Latest Filing</div>
          <div className="value">{funds[0]?.latest_period?.slice(0, 7) || "—"}</div>
        </div>
      </div>
      <div className="card">
        <h2>Tracked Funds</h2>
        <p className="meta">Click a row to inspect that fund's current holdings.</p>
        <table>
          <thead>
            <tr>
              <th>Fund</th><th>CIK</th><th>Latest Filing</th><th>Positions</th><th>AUM (approx.)</th>
            </tr>
          </thead>
          <tbody>
            {funds.map((f) => (
              <tr key={f.cik} className="clickable-row" onClick={() => onFundClick?.(f.cik)}>
                <td><strong>{displayName(f.name)}</strong></td>
                <td style={{ color: "#64748b" }}>{fmtCik(f.cik)}</td>
                <td>{f.latest_period || "—"}</td>
                <td>{Number(f.num_positions || 0).toLocaleString()}</td>
                <td>{fmtMoney(f.total_value_usd_mm)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

// ── Fund holdings with inline expandable histogram ────────────────────────────

function InlineHistogram({ history, loading }) {
  const values  = Array.isArray(history?.values)  ? history.values  : [];
  const periods = Array.isArray(history?.periods) ? history.periods : [];
  const maxVal  = Math.max(1, ...values.map((v) => Number(v || 0)));

  if (loading) return <p style={{ color: "#64748b", padding: "12px 0" }}>Loading history…</p>;
  if (!values.length) return <p style={{ color: "#94a3b8", padding: "12px 0" }}>No history data.</p>;

  return (
    <div style={{ padding: "16px 8px 8px" }}>
      <div className="histogram-shell">
        <div className="histogram-boxes">
          {values.map((value, i) => {
            const heightPct = Math.max(14, Math.round((Number(value || 0) / maxVal) * 100));
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
                  {i === 0 ? "start" : `${delta >= 0 ? "+" : ""}${fmtPct(delta)}`}
                </div>
              </div>
            );
          })}
        </div>
      </div>
      <div className="history-summary">
        <span>Increase total: <strong>{fmtPct(history.increase_total || 0)}</strong></span>
        <span>Decrease total: <strong>{fmtPct(history.decrease_total || 0)}</strong></span>
        <span>Net change: <strong>{fmtPct(history.net_change || 0)}</strong></span>
      </div>
    </div>
  );
}

function HoldingsTable({ fund, rows, loading, onRowClick, expandedCusip, historyMap, loadingCusip }) {
  const COLS = 9; // # of td columns including the triangle column
  return (
    <div className="card">
      <h2>{displayName(fund.name)} — Holdings</h2>
      <p className="meta">CIK {fmtCik(fund.cik)} · Latest filing {fund.latest_period || "N/A"} · Click a row to expand position history</p>
      {loading ? <p>Loading holdings…</p> : rows.length === 0 ? <p>No holdings data available.</p> : (
        <table>
          <thead>
            <tr>
              <th style={{ width: 24 }}></th>
              <th>#</th><th>Company</th><th>Ticker</th><th>CUSIP</th>
              <th>Weight</th><th>Value</th><th>Shares</th><th>Sector</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const isOpen = expandedCusip === r.cusip;
              return (
                <>
                  <tr
                    key={`${fund.cik}-${r.rank}-${r.cusip}`}
                    className={`clickable-row ${isOpen ? "selected" : ""}`}
                    onClick={() => onRowClick?.(r)}
                  >
                    <td style={{ color: "#94a3b8", textAlign: "center", fontSize: "0.7em" }}>
                      {isOpen ? "▼" : "▶"}
                    </td>
                    <td style={{ color: "#94a3b8" }}>{r.rank}</td>
                    <td>{r.name}</td>
                    <td><strong>{r.ticker || "—"}</strong></td>
                    <td style={{ color: "#94a3b8", fontSize: "0.78em" }}>{r.cusip || "—"}</td>
                    <td>{fmtPct(r.weight_pct)}</td>
                    <td>{fmtMoney(r.value_usd_mm)}</td>
                    <td>{Number(r.shares || 0).toLocaleString()}</td>
                    <td>{r.sector || "—"}</td>
                  </tr>
                  {isOpen && (
                    <tr key={`hist-${r.cusip}`} className="history-inline-row">
                      <td colSpan={COLS} style={{ padding: 0, background: "#f8fafc", borderBottom: "2px solid #e0f2fe" }}>
                        <InlineHistogram
                          history={historyMap?.[r.cusip]}
                          loading={loadingCusip === r.cusip}
                        />
                      </td>
                    </tr>
                  )}
                </>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}

// ── Models tab ─────────────────────────────────────────────────────────────────

const DEFAULT_MODEL_COLUMNS = [
  { field: "asset", headerName: "Asset", headerTooltip: "Security name from the 13F filing." },
  { field: "ticker", headerName: "Ticker", headerTooltip: "Exchange ticker inferred from CUSIP/company mapping." },
  { field: "sector", headerName: "Sector", headerTooltip: "Mapped sector proxy used by the model." },
  { field: "status", headerName: "Status", headerTooltip: "Action badge: NEW POSITION, ACCUMULATING, TRIMMING, EXITED." },
  { field: "portfolio_count", headerName: "In Portfolios", headerTooltip: "How many tracked funds currently hold this stock. Click row for full holder details." },
  { field: "thesis_conviction", headerName: "Thesis Conviction", headerTooltip: "0 to 1 score for strategy alignment strength." },
  { field: "top_tft_driver", headerName: "Top TFT Driver", headerTooltip: "The #1 reason the model produced this prediction." },
  { field: "flow_signal", headerName: "Flow Signal", headerTooltip: "Institutional flow pressure regime from z-score behavior." },
];

function renderModelCell(row, field) {
  switch (field) {
    case "asset":
      return row.asset || row.name || "—";
    case "ticker":
      return <strong>{row.ticker || "—"}</strong>;
    case "pred_wt_pct":
    case "last_13f_wt_pct":
      return row[field] != null ? fmtPct(row[field]) : "—";
    case "weight_delta_bps": {
      const n = Number(row.weight_delta_bps || 0);
      return `${n >= 0 ? "+" : ""}${n.toFixed(0)}`;
    }
    case "status": {
      const v = String(row.status || "ACCUMULATING");
      const cls = v === "NEW POSITION" || v === "ACCUMULATING" ? "buy" : "sell";
      return <span className={`pill ${cls}`}>{v}</span>;
    }
    case "thesis_conviction":
      return row.thesis_conviction != null ? Number(row.thesis_conviction).toFixed(3) : "—";
    case "top_tft_driver":
      return row.top_tft_driver || "—";
    case "flow_signal": {
      const v = String(row.flow_signal || "Neutral");
      const cls = v === "Inflow" ? "buy" : v === "Outflow" ? "sell" : "hold";
      return <span className={`pill ${cls}`}>{v}</span>;
    }
    case "sector":
      return row.sector || "—";
    case "portfolio_count":
      return Number(row.portfolio_count || 0).toLocaleString();
    default:
      return row[field] != null && row[field] !== "" ? String(row[field]) : "—";
  }
}

function SignalsTable({ rows, loading, columnsDefs }) {
  const [expanded, setExpanded] = useState(null);

  if (loading) return <div className="card"><p style={{ color: "#64748b" }}>Running model — may take 30–60 s on first load…</p></div>;
  if (!rows || rows.length === 0) return <div className="card"><p style={{ color: "#64748b" }}>No signal data available yet.</p></div>;

  const cols = Array.isArray(columnsDefs) && columnsDefs.length > 0 ? columnsDefs : DEFAULT_MODEL_COLUMNS;

  return (
    <div className="card">
      <h2>ML Stock Signals — All Funds</h2>
      <p className="meta">
        HistGradientBoosting model trained on all tracked funds' 4-quarter 13F transitions + yfinance price data.
        Hover any column header for its definition.
        <strong> Thesis Conviction</strong> is normalized to a 0..1 strength score.
        <br />
        New idea pipeline (next step): GraphRAG entity extraction + flow anomaly scan + zero-weight initialization for net-new names.
      </p>
      <table>
        <thead>
          <tr>
            {cols.map((col) => (
              <th key={col.field} title={col.headerTooltip || ""}>
                {col.headerName || col.field}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const key = `${r.cusip || r.ticker || r.name}-${i}`;
            const isOpen = expanded === key;
            const positions = Array.isArray(r.portfolio_positions) ? r.portfolio_positions : [];
            return [
                <tr key={key} className={`clickable-row ${isOpen ? "selected" : ""}`} onClick={() => setExpanded(isOpen ? null : key)}>
                  {cols.map((col) => (
                    <td key={`${col.field}-${key}`}>
                      {renderModelCell(r, col.field)}
                    </td>
                  ))}
                </tr>,
                isOpen && (
                  <tr key={`details-${key}`} className="history-inline-row">
                    <td colSpan={cols.length} style={{ padding: 0, background: "#f8fafc", borderBottom: "2px solid #e0f2fe" }}>
                      <div style={{ padding: "12px 14px" }}>
                        <div style={{ display: "flex", gap: 18, flexWrap: "wrap", marginBottom: 10 }}>
                          <span>Pred Wt: <strong>{fmtPct(r.pred_wt_pct)}</strong></span>
                          <span>Wt Δ: <strong>{`${Number(r.weight_delta_bps || 0) >= 0 ? "+" : ""}${Number(r.weight_delta_bps || 0).toFixed(0)} bps`}</strong></span>
                          <span>Last 13F Wt: <strong>{fmtPct(r.last_13f_wt_pct)}</strong></span>
                        </div>
                        {!positions.length ? (
                          <p style={{ color: "#64748b", margin: 0 }}>Not currently present in tracked portfolios.</p>
                        ) : (
                          <table>
                            <thead>
                              <tr>
                                <th>Portfolio</th><th>CIK</th><th>Weight %</th><th>Shares</th><th>Value (MM)</th><th>Reported Implied Price</th>
                              </tr>
                            </thead>
                            <tbody>
                              {positions.map((p, idx) => (
                                <tr key={`${key}-p-${idx}`}>
                                  <td>{p.portfolio || "—"}</td>
                                  <td>{fmtCik(p.cik)}</td>
                                  <td>{fmtPct(p.weight_pct)}</td>
                                  <td>{Number(p.shares || 0).toLocaleString()}</td>
                                  <td>{`$${Number(p.value_usd_mm || 0).toFixed(2)}M`}</td>
                                  <td>{p.reported_implied_price ? `$${Number(p.reported_implied_price).toFixed(2)}` : "—"}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        )}
                      </div>
                    </td>
                  </tr>
                ),
            ];
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── Main ───────────────────────────────────────────────────────────────────────

export default function HomePage() {
  const [funds, setFunds]           = useState([]);
  const [activeView, setActiveView] = useState("summary");
  const [fundDataCache, setFundDataCache] = useState({});

  const [holdingsRows, setHoldingsRows]     = useState([]);
  const [expandedCusip, setExpandedCusip]   = useState(null);
  const [historyMap, setHistoryMap]         = useState({});
  const [loadingCusip, setLoadingCusip]     = useState(null);
  const [loadingTabData, setLoadingTabData] = useState(false);

  const [modelsData, setModelsData]       = useState({ data: [], columnsDefs: [] });
  const [loadingModels, setLoadingModels] = useState(false);

  const [loadingFunds, setLoadingFunds] = useState(true);
  const [error, setError]               = useState("");

  const selectedFunds = useMemo(() => sortFunds(funds), [funds]);
  const activeFund    = useMemo(() => selectedFunds.find((f) => f.cik === activeView) || null, [selectedFunds, activeView]);
  const isFundView    = activeFund !== null;

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

  useEffect(() => {
    setExpandedCusip(null);
  }, [activeView]);

  useEffect(() => {
    if (!selectedFunds.length) return;
    const valid = ["summary", "models"].includes(activeView) || selectedFunds.some((f) => f.cik === activeView);
    if (!valid) {
      const first = selectedFunds.find((f) => Number(f.num_positions || 0) > 0) || selectedFunds[0];
      if (first) setActiveView(first.cik);
    }
  }, [selectedFunds, activeView]);

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

  useEffect(() => {
    if (fundDataCache["models"]) { setModelsData(fundDataCache["models"]); return; }

    async function load() {
      try {
        setLoadingModels(true);
        const res  = await fetch("api/recommendations?top_n=20", { cache: "no-store" });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Failed");
        const result = { data: Array.isArray(data?.data) ? data.data : [], columnsDefs: data?.columnsDefs || [] };
        setModelsData(result);
        setFundDataCache((prev) => ({ ...prev, models: result }));
      } catch (e) {
        setError(String(e));
      } finally {
        setLoadingModels(false);
      }
    }
    load();
  }, [fundDataCache]);

  async function loadStockHistory(stock) {
    if (!stock || !isFundView) return;
    const cusip = String(stock.cusip || "").toUpperCase();
    if (!cusip) return;

    // toggle closed
    if (expandedCusip === cusip) { setExpandedCusip(null); return; }

    setExpandedCusip(cusip);
    if (historyMap[cusip]) return; // already cached

    try {
      setLoadingCusip(cusip);
      const params = new URLSearchParams({ cik: activeView, cusip: stock.cusip || "", name: stock.name || "", ticker: stock.ticker || "" });
      const res  = await fetch(`api/stock-history?${params}`, { cache: "no-store" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed");
      setHistoryMap((prev) => ({ ...prev, [cusip]: data }));
    } catch (e) {
      setError(String(e));
    } finally {
      setLoadingCusip(null);
    }
  }

  return (
    <>
      <TopNav
        activeView={activeView}
        setActiveView={setActiveView}
        selectedFunds={selectedFunds}
        isFundView={isFundView}
      />

      <div className="page-body">
        <Hero />

        <div className="container">
          {error && <div className="error">{error}</div>}

          {loadingFunds ? (
            <p style={{ color: "#64748b" }}>Loading dashboard…</p>
          ) : (
            <>
              {activeView === "summary" && (
                <TrackedFunds funds={selectedFunds} onFundClick={(cik) => setActiveView(cik)} />
              )}

              {activeView === "models" && (
                <SignalsTable rows={modelsData.data} loading={loadingModels} columnsDefs={modelsData.columnsDefs} />
              )}

              {isFundView && activeFund && (
                <>
                  <HoldingsTable
                    fund={activeFund}
                    rows={holdingsRows}
                    loading={loadingTabData}
                    onRowClick={loadStockHistory}
                    expandedCusip={expandedCusip}
                    historyMap={historyMap}
                    loadingCusip={loadingCusip}
                  />
                </>
              )}

              {!["summary", "models"].includes(activeView) && !activeFund && (
                <div className="card"><p style={{ color: "#64748b" }}>No fund data available.</p></div>
              )}
            </>
          )}
        </div>
      </div>

      <Footer />
    </>
  );
}
