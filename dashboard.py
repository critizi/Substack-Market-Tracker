# dashboard.py — Single-page interactive HTML dashboard for hedge fund investors

from __future__ import annotations

import json
import math
import os
from datetime import datetime

import pandas as pd


def _signal_score(row: dict, xref_set: set[str]) -> float:
    """Composite signal quality score 0–100."""
    r = row.get("pearson_r") or 0.0
    p = row.get("pearson_p") or 1.0
    spike = row.get("volume_spike_ratio") or 0.0
    mentions = max(1, row.get("mention_count") or 1)
    ticker = row.get("ticker", "")

    if isinstance(r, float) and math.isnan(r):
        r = 0.0
    if isinstance(p, float) and math.isnan(p):
        p = 1.0
    if isinstance(spike, float) and math.isnan(spike):
        spike = 0.0

    r_score = max(0.0, r) * 40.0
    sig_score = 20.0 if p < 0.05 else (10.0 if p < 0.10 else 0.0)
    spike_score = min(20.0, max(0.0, (spike - 1.0) * 20.0))
    mention_score = min(10.0, math.log10(mentions) / math.log10(100) * 10.0)
    xref_score = 10.0 if ticker in xref_set else 0.0

    return round(min(100.0, r_score + sig_score + spike_score + mention_score + xref_score), 1)


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    except (TypeError, ValueError):
        return None


def _get_recent_mentions(conn, limit: int = 60, slugs: list[str] | None = None) -> list[dict]:
    if slugs:
        placeholders = ",".join("?" * len(slugs))
        rows = conn.execute(
            f"""
            SELECT
                tm.ticker,
                tm.mention_context,
                tm.match_type,
                p.substack_slug,
                p.title,
                p.url,
                p.published_date
            FROM ticker_mentions tm
            JOIN posts p ON p.id = tm.post_id
            WHERE p.substack_slug IN ({placeholders})
            ORDER BY p.published_date DESC, tm.id DESC
            LIMIT ?
            """,
            (*slugs, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT
                tm.ticker,
                tm.mention_context,
                tm.match_type,
                p.substack_slug,
                p.title,
                p.url,
                p.published_date
            FROM ticker_mentions tm
            JOIN posts p ON p.id = tm.post_id
            ORDER BY p.published_date DESC, tm.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def generate_dashboard(
    conn,
    corr_df: pd.DataFrame,
    xref_rows: list[dict],
    output_dir: str,
    slugs: list[str] | None = None,
) -> str:
    """
    Generate a self-contained dashboard.html in output_dir.
    Returns the output file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "dashboard.html")

    xref_set = {r["ticker"] for r in xref_rows}
    xref_map = {r["ticker"]: r for r in xref_rows}

    records: list[dict] = []
    if not corr_df.empty:
        for _, row in corr_df.iterrows():
            r = row.to_dict()
            for k in list(r.keys()):
                if isinstance(r[k], float):
                    r[k] = _safe_float(r[k])
            r["signal_score"] = _signal_score(r, xref_set)
            r["is_consensus"] = r["ticker"] in xref_set
            r["pub_count"] = xref_map.get(r["ticker"], {}).get("slug_count", 1)
            r["slugs"] = xref_map.get(r["ticker"], {}).get("slugs", "")
            records.append(r)

    records.sort(key=lambda x: x["signal_score"], reverse=True)

    recent_mentions = _get_recent_mentions(conn, limit=60, slugs=slugs)

    top_signal = records[0]["ticker"] if records else "—"
    top_score = records[0]["signal_score"] if records else 0
    valid_spikes = [r["volume_spike_ratio"] for r in records if r["volume_spike_ratio"] is not None]
    avg_spike = round(sum(valid_spikes) / len(valid_spikes), 4) if valid_spikes else 0.0

    try:
        post_count = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
        mention_count_total = conn.execute("SELECT COUNT(*) FROM ticker_mentions").fetchone()[0]
    except Exception:
        post_count = mention_count_total = 0

    run_date = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    payload = {
        "records": records,
        "xref": xref_rows,
        "recent_mentions": recent_mentions,
        "kpis": {
            "total_tickers": len(records),
            "top_signal": top_signal,
            "top_score": top_score,
            "consensus_count": len(xref_rows),
            "avg_spike": avg_spike,
            "post_count": post_count,
            "mention_count": mention_count_total,
            "run_date": run_date,
        },
    }

    data_json = json.dumps(payload, default=str, ensure_ascii=False)
    html = _HTML_TEMPLATE.replace("__DATA_JSON__", data_json)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    return out_path


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Alpha Intelligence — Substack Signal Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" crossorigin="anonymous"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:'SF Mono','Cascadia Code','Fira Mono',monospace;background:#070d1a;color:#c9d4e8;min-height:100vh;font-size:13px}
a{color:#38bdf8;text-decoration:none}
a:hover{color:#7dd3fc;text-decoration:underline}

/* HEADER */
.header{display:flex;align-items:center;justify-content:space-between;padding:14px 28px;background:#0b1627;border-bottom:1px solid #1a2d45;position:sticky;top:0;z-index:100}
.logo{font-size:15px;font-weight:700;letter-spacing:.18em;color:#38bdf8;text-transform:uppercase}
.logo span{color:#334155;font-weight:400}
.header-meta{font-size:11px;color:#475569;text-align:right;line-height:1.7}
.header-meta strong{color:#94a3b8}

/* KPI ROW */
.kpi-row{display:flex;gap:10px;padding:18px 28px;overflow-x:auto}
.kpi-card{background:#0d1b2e;border:1px solid #1a2d45;border-radius:8px;padding:14px 18px;min-width:150px;flex:1}
.kpi-label{font-size:9px;text-transform:uppercase;letter-spacing:.14em;color:#334155;margin-bottom:6px}
.kpi-value{font-size:26px;font-weight:800;letter-spacing:-.02em;color:#e2e8f0}
.kpi-sub{font-size:10px;color:#475569;margin-top:3px}
.kpi-card.c-blue .kpi-value{color:#38bdf8}
.kpi-card.c-green .kpi-value{color:#34d399}
.kpi-card.c-yellow .kpi-value{color:#fbbf24}
.kpi-card.c-purple .kpi-value{color:#a78bfa}

/* TABS */
.tab-bar{display:flex;padding:0 28px;background:#0b1627;border-bottom:1px solid #1a2d45}
.tab-btn{background:none;border:none;border-bottom:2px solid transparent;color:#475569;font-family:inherit;font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;padding:12px 16px;cursor:pointer;transition:color .15s,border-color .15s}
.tab-btn:hover{color:#94a3b8}
.tab-btn.active{color:#38bdf8;border-bottom-color:#38bdf8}

/* CONTENT */
.content{padding:20px 28px}
.tab-panel{display:none}
.tab-panel.active{display:block}

/* TOOLBAR */
.toolbar{display:flex;gap:10px;margin-bottom:14px;align-items:center;flex-wrap:wrap}
.search-box{background:#0d1b2e;border:1px solid #1e3a5f;border-radius:6px;color:#c9d4e8;font-family:inherit;font-size:12px;padding:7px 11px;width:200px;outline:none;transition:border-color .15s}
.search-box:focus{border-color:#38bdf8}
.search-box::placeholder{color:#1e3a5f}
.filter-select{background:#0d1b2e;border:1px solid #1e3a5f;border-radius:6px;color:#94a3b8;font-family:inherit;font-size:11px;padding:7px 10px;outline:none;cursor:pointer}
.filter-select:focus{border-color:#38bdf8}
.count-label{font-size:10px;color:#334155;margin-left:auto}

/* DATA TABLE */
.tbl-wrap{overflow-x:auto;border-radius:8px;border:1px solid #1a2d45}
table{width:100%;border-collapse:collapse;font-size:12px}
thead th{background:#0a1625;color:#475569;font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;padding:10px 13px;text-align:left;cursor:pointer;white-space:nowrap;user-select:none;border-bottom:1px solid #1a2d45;transition:color .15s}
thead th:hover{color:#94a3b8}
thead th.sort-asc::after{content:" ▲";color:#38bdf8}
thead th.sort-desc::after{content:" ▼";color:#38bdf8}
tbody tr{border-bottom:1px solid #0f1e32;transition:background .1s}
tbody tr:hover{background:#0d1b2e}
tbody td{padding:9px 13px;vertical-align:middle}
.ticker-lnk{font-weight:800;font-size:13px;color:#7dd3fc;letter-spacing:.05em}
.ticker-lnk:hover{color:#38bdf8}

/* COLUMN HEADER TOOLTIPS */
th[data-tip]{position:relative;cursor:help}
th[data-tip]::after{
  content:attr(data-tip);
  position:absolute;top:calc(100% + 6px);left:50%;transform:translateX(-50%);
  background:#0d1b2e;border:1px solid #1a2d45;color:#94a3b8;
  font-size:11px;font-weight:400;line-height:1.5;white-space:pre-wrap;
  width:220px;padding:8px 10px;border-radius:6px;
  pointer-events:none;opacity:0;transition:opacity .15s;z-index:200;
  box-shadow:0 4px 16px rgba(0,0,0,.5)
}
th[data-tip]:hover::after{opacity:1}

/* SIGNAL SCORE CELL */
.score-cell{display:flex;align-items:center;gap:8px;min-width:100px}
.bar-bg{width:56px;height:5px;background:#1a2d45;border-radius:3px;overflow:hidden;flex-shrink:0}
.bar-fill{height:100%;border-radius:3px}
.bar-hi{background:linear-gradient(90deg,#34d399,#059669)}
.bar-md{background:linear-gradient(90deg,#fbbf24,#d97706)}
.bar-lo{background:linear-gradient(90deg,#f87171,#dc2626)}
.score-num{font-weight:800;font-size:12px;min-width:32px}
.c-hi{color:#34d399}.c-md{color:#fbbf24}.c-lo{color:#f87171}

/* BADGES */
.badge{display:inline-block;font-size:9px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;padding:2px 6px;border-radius:3px}
.b-sig{background:#064e3b;color:#34d399;border:1px solid #065f46}
.b-weak{background:#451a03;color:#fbbf24;border:1px solid #78350f}
.b-ns{background:#1c1917;color:#57534e;border:1px solid #292524}
.b-cons{background:#172554;color:#60a5fa;border:1px solid #1e3a8a}
.b-na{background:#111;color:#333;border:1px solid #222}

/* CONSENSUS CARDS */
.cons-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px}
.cons-card{background:#0d1b2e;border:1px solid #1a2d45;border-radius:8px;padding:16px;transition:border-color .15s}
.cons-card:hover{border-color:#38bdf8}
.cons-ticker{font-size:22px;font-weight:800;color:#7dd3fc;letter-spacing:.05em;margin-bottom:4px}
.pub-pills{margin-bottom:10px}
.pub-pill{display:inline-block;background:#0b1a2c;border:1px solid #1e3a5f;border-radius:10px;font-size:9px;color:#94a3b8;padding:2px 8px;margin:2px 2px 2px 0}
.cons-stats{display:flex;gap:16px;margin-top:8px;font-size:11px}
.stat{display:flex;flex-direction:column}
.stat-label{color:#334155;font-size:9px;text-transform:uppercase;letter-spacing:.1em}
.stat-val{color:#c9d4e8;font-weight:700}

/* MENTIONS FEED */
.feed{display:flex;flex-direction:column;gap:8px}
.mention-card{background:#0d1b2e;border:1px solid #1a2d45;border-radius:6px;padding:12px 14px;display:grid;grid-template-columns:58px 1fr auto;gap:12px;align-items:start}
.m-ticker{font-size:14px;font-weight:800;color:#7dd3fc}
.m-title{font-size:12px;color:#94a3b8;margin-bottom:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.m-ctx{font-size:11px;color:#475569;font-style:italic;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.m-meta{text-align:right;min-width:100px}
.m-date{font-size:11px;color:#475569}
.m-slug{font-size:10px;color:#38bdf8}

/* CHART */
.chart-wrap{background:#0d1b2e;border:1px solid #1a2d45;border-radius:8px;padding:16px;min-height:500px}

/* EMPTY / MISC */
.empty{color:#1e3a5f;text-align:center;padding:60px 20px;font-size:13px}
.section-title{font-size:10px;text-transform:uppercase;letter-spacing:.14em;color:#334155;margin-bottom:14px}

/* SCROLLBAR */
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:#070d1a}
::-webkit-scrollbar-thumb{background:#1a2d45;border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:#1e3a5f}
</style>
</head>
<body>

<header class="header">
  <div class="logo">&#9670;&nbsp;Alpha<span>/Intelligence</span></div>
  <div class="header-meta">
    <strong id="hdr-date">—</strong><br>
    <span id="hdr-stats">—</span>
  </div>
</header>

<div class="kpi-row">
  <div class="kpi-card c-blue">
    <div class="kpi-label">Tickers Tracked</div>
    <div class="kpi-value" id="kpi-tickers">—</div>
    <div class="kpi-sub" id="kpi-posts-sub">— posts scraped</div>
  </div>
  <div class="kpi-card c-blue">
    <div class="kpi-label">Top Signal</div>
    <div class="kpi-value" id="kpi-top">—</div>
    <div class="kpi-sub" id="kpi-top-score">Score —</div>
  </div>
  <div class="kpi-card c-green">
    <div class="kpi-label">Consensus Picks</div>
    <div class="kpi-value" id="kpi-cons">—</div>
    <div class="kpi-sub">2+ publications</div>
  </div>
  <div class="kpi-card c-yellow">
    <div class="kpi-label">Avg Vol Spike</div>
    <div class="kpi-value" id="kpi-spike">—</div>
    <div class="kpi-sub">vs. 30-day baseline</div>
  </div>
  <div class="kpi-card c-purple">
    <div class="kpi-label">Total Mentions</div>
    <div class="kpi-value" id="kpi-mentions">—</div>
    <div class="kpi-sub">ticker occurrences</div>
  </div>
</div>

<div class="tab-bar">
  <button class="tab-btn active" data-tab="signals">Signal Intelligence</button>
  <button class="tab-btn" data-tab="consensus">Consensus Picks</button>
  <button class="tab-btn" data-tab="overview">Market Overview</button>
  <button class="tab-btn" data-tab="mentions">Recent Mentions</button>
</div>

<div class="content">

  <div class="tab-panel active" id="panel-signals">
    <div class="toolbar">
      <input class="search-box" id="tbl-search" placeholder="Search ticker…" oninput="applyTable()">
      <select class="filter-select" id="tbl-filter" onchange="applyTable()">
        <option value="all">All Signals</option>
        <option value="sig">Significant (p &lt; 0.05)</option>
        <option value="consensus">Consensus Only</option>
        <option value="spike">Vol Spike &gt; 1×</option>
        <option value="strong">Score ≥ 60</option>
      </select>
      <span class="count-label" id="tbl-count"></span>
    </div>
    <div class="tbl-wrap">
      <table id="sig-table">
        <thead><tr>
          <th data-col="rank">#</th>
          <th data-col="ticker">Ticker</th>
          <th data-col="signal_score" class="sort-desc" data-tip="Composite score 0–100.&#10;&#10;Pearson r  → up to 40 pts&#10;p &lt; 0.05   → 20 pts (p &lt; 0.10 → 10 pts)&#10;Vol spike  → up to 20 pts&#10;Mentions   → up to 10 pts (log-scaled)&#10;Cross-ref  → 10 pts if 2+ sources">Signal Score</th>
          <th data-col="pearson_r" data-tip="Pearson correlation between mention frequency and price returns over the lookback window. Range −1 to +1. Positive = price tends to rise after mentions.">Pearson r</th>
          <th data-col="pearson_p" data-tip="Two-tailed p-value for the Pearson r. Below 0.05 = statistically significant (shown in green). Below 0.10 = marginal (yellow).">p-value</th>
          <th data-col="volume_spike_ratio" data-tip="Average trading volume in the days after a mention divided by the prior baseline volume. 2.0× means volume doubled. Higher = more market attention post-mention.">Vol Spike</th>
          <th data-col="avg_price_return_pct" data-tip="Average % price change in the window after a mention across all posts. Positive = stock tended to rise after being mentioned.">Avg Return</th>
          <th data-col="mention_count" data-tip="Total number of times this ticker was mentioned across all scraped posts in the lookback window.">Mentions</th>
          <th data-col="pub_count" data-tip="Number of distinct Substack publications that mentioned this ticker. Higher = broader coverage across sources.">Sources</th>
          <th data-col="is_consensus" data-tip="Marked ✓ if the ticker was mentioned by 2 or more different publications. Consensus picks get a +10 pt bonus to their Signal Score.">Consensus</th>
        </tr></thead>
        <tbody id="sig-tbody"></tbody>
      </table>
    </div>
  </div>

  <div class="tab-panel" id="panel-consensus">
    <div class="section-title">Tickers mentioned across multiple publications</div>
    <div class="cons-grid" id="cons-grid"></div>
  </div>

  <div class="tab-panel" id="panel-overview">
    <div class="chart-wrap" id="scatter-div"></div>
  </div>

  <div class="tab-panel" id="panel-mentions">
    <div class="toolbar">
      <input class="search-box" id="feed-search" placeholder="Filter by ticker or source…" oninput="applyFeed()">
      <span class="count-label" id="feed-count"></span>
    </div>
    <div class="feed" id="feed-list"></div>
  </div>

</div>

<script>
const RAW = __DATA_JSON__;
const {records, xref, recent_mentions, kpis} = RAW;

// ── KPIs ──────────────────────────────────────────────────────────────────
document.getElementById('hdr-date').textContent = kpis.run_date;
document.getElementById('hdr-stats').textContent =
  kpis.post_count.toLocaleString() + ' posts · ' + kpis.mention_count.toLocaleString() + ' mentions indexed';
document.getElementById('kpi-tickers').textContent = kpis.total_tickers;
document.getElementById('kpi-posts-sub').textContent = kpis.post_count.toLocaleString() + ' posts scraped';
document.getElementById('kpi-top').textContent = kpis.top_signal;
document.getElementById('kpi-top-score').textContent = 'Score ' + kpis.top_score;
document.getElementById('kpi-cons').textContent = kpis.consensus_count;
document.getElementById('kpi-spike').textContent = Number(kpis.avg_spike).toFixed(3) + '×';
document.getElementById('kpi-mentions').textContent = kpis.mention_count.toLocaleString();

// ── Tabs ──────────────────────────────────────────────────────────────────
let scatterDrawn = false;
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const panel = document.getElementById('panel-' + btn.dataset.tab);
    if (panel) panel.classList.add('active');
    if (btn.dataset.tab === 'overview' && !scatterDrawn) { drawScatter(); scatterDrawn = true; }
  });
});

// ── Helpers ───────────────────────────────────────────────────────────────
function fmt(v, d) { return (v == null) ? '—' : Number(v).toFixed(d); }

function scoreClasses(s) {
  if (s >= 60) return ['bar-hi', 'c-hi'];
  if (s >= 35) return ['bar-md', 'c-md'];
  return ['bar-lo', 'c-lo'];
}

function pBadge(p) {
  if (p == null) return '<span class="badge b-na">N/A</span>';
  if (p < 0.05)  return '<span class="badge b-sig">SIG</span>';
  if (p < 0.10)  return '<span class="badge b-weak">WEAK</span>';
  return '<span class="badge b-ns">NS</span>';
}

function numColor(v, pos, neg) {
  if (v == null) return '#475569';
  return v > 0 ? pos : v < 0 ? neg : '#94a3b8';
}

function rColor(v) {
  if (v == null) return '#475569';
  if (v >= 0.7) return '#34d399';
  if (v >= 0.4) return '#fbbf24';
  return '#f87171';
}

function slugLabel(s) {
  return (s || '').replace(/^www\./, '').replace(/\.substack\.com$/, '').replace(/\.com$/, '');
}

// ── Signal Table ──────────────────────────────────────────────────────────
let sortCol = 'signal_score', sortDir = -1;

function renderTable(rows) {
  const tbody = document.getElementById('sig-tbody');
  document.getElementById('tbl-count').textContent = rows.length + ' of ' + records.length + ' tickers';
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="10" class="empty">No results match your filters.</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map((r, i) => {
    const sc = r.signal_score;
    const [barCls, numCls] = scoreClasses(sc);
    const spikeVal = r.volume_spike_ratio != null ? fmt(r.volume_spike_ratio, 3) + '×' : '—';
    const spikeColor = r.volume_spike_ratio == null ? '#475569'
      : r.volume_spike_ratio > 1.2 ? '#34d399'
      : r.volume_spike_ratio > 0.9 ? '#fbbf24' : '#f87171';
    const retSign = r.avg_price_return_pct != null && r.avg_price_return_pct >= 0 ? '+' : '';
    const retVal = r.avg_price_return_pct != null ? retSign + fmt(r.avg_price_return_pct, 2) + '%' : '—';
    return `<tr>
      <td style="color:#334155">${i + 1}</td>
      <td><a class="ticker-lnk" href="${r.ticker}_timeseries.html">${r.ticker}</a></td>
      <td>
        <div class="score-cell">
          <div class="bar-bg"><div class="bar-fill ${barCls}" style="width:${Math.min(100, sc)}%"></div></div>
          <span class="score-num ${numCls}">${sc}</span>
        </div>
      </td>
      <td style="color:${rColor(r.pearson_r)};font-weight:700">${fmt(r.pearson_r, 4)}</td>
      <td>${pBadge(r.pearson_p)}</td>
      <td style="color:${spikeColor};font-weight:700">${spikeVal}</td>
      <td style="color:${numColor(r.avg_price_return_pct,'#34d399','#f87171')};font-weight:700">${retVal}</td>
      <td style="color:#94a3b8">${r.mention_count}</td>
      <td style="color:#475569">${r.pub_count}</td>
      <td>${r.is_consensus ? '<span class="badge b-cons">&#10003; CONSENSUS</span>' : ''}</td>
    </tr>`;
  }).join('');
}

function applyTable() {
  const q = document.getElementById('tbl-search').value.toUpperCase().trim();
  const f = document.getElementById('tbl-filter').value;
  let rows = records.filter(r => {
    if (q && !r.ticker.includes(q)) return false;
    if (f === 'sig'       && (r.pearson_p == null || r.pearson_p >= 0.05)) return false;
    if (f === 'consensus' && !r.is_consensus) return false;
    if (f === 'spike'     && (r.volume_spike_ratio == null || r.volume_spike_ratio <= 1)) return false;
    if (f === 'strong'    && r.signal_score < 60) return false;
    return true;
  });
  rows = [...rows].sort((a, b) => {
    const av = a[sortCol], bv = b[sortCol];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    return sortDir * (typeof av === 'string' ? av.localeCompare(bv) : av - bv);
  });
  renderTable(rows);
}

document.querySelectorAll('#sig-table thead th').forEach(th => {
  th.addEventListener('click', () => {
    const col = th.dataset.col;
    if (!col) return;
    if (sortCol === col) { sortDir *= -1; }
    else { sortCol = col; sortDir = col === 'ticker' ? 1 : -1; }
    document.querySelectorAll('#sig-table thead th').forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
    th.classList.add(sortDir === 1 ? 'sort-asc' : 'sort-desc');
    applyTable();
  });
});

// ── Consensus Cards ───────────────────────────────────────────────────────
function renderConsensus() {
  const grid = document.getElementById('cons-grid');
  const recMap = Object.fromEntries(records.map(r => [r.ticker, r]));
  if (!xref.length) { grid.innerHTML = '<p class="empty">No consensus picks in this dataset.</p>'; return; }
  grid.innerHTML = xref.map(x => {
    const rec = recMap[x.ticker] || {};
    const sc = rec.signal_score != null ? rec.signal_score : null;
    const scColor = sc == null ? '#475569' : sc >= 60 ? '#34d399' : sc >= 35 ? '#fbbf24' : '#f87171';
    const pills = (x.slugs || '').split(',')
      .map(s => `<span class="pub-pill">${slugLabel(s.trim())}</span>`).join('');
    return `<div class="cons-card">
      <div class="cons-ticker"><a href="${x.ticker}_timeseries.html" style="color:inherit">${x.ticker}</a></div>
      <div class="pub-pills">${pills}</div>
      <div class="cons-stats">
        <div class="stat"><span class="stat-label">Signal Score</span><span class="stat-val" style="color:${scColor}">${sc ?? '—'}</span></div>
        <div class="stat"><span class="stat-label">Publications</span><span class="stat-val">${x.slug_count}</span></div>
        <div class="stat"><span class="stat-label">Mentions</span><span class="stat-val">${x.total_mentions}</span></div>
        <div class="stat"><span class="stat-label">Pearson r</span><span class="stat-val">${rec.pearson_r != null ? Number(rec.pearson_r).toFixed(3) : '—'}</span></div>
      </div>
    </div>`;
  }).join('');
}

// ── Scatter Chart ─────────────────────────────────────────────────────────
function drawScatter() {
  const el = document.getElementById('scatter-div');
  const valid = records.filter(r => r.mention_count >= 3 && r.volume_spike_ratio != null);
  if (!valid.length) { el.innerHTML = '<p class="empty">Insufficient data.</p>'; return; }
  Plotly.newPlot(el, [{
    x: valid.map(r => r.mention_count),
    y: valid.map(r => r.volume_spike_ratio),
    text: valid.map(r => r.ticker),
    mode: 'markers+text',
    textposition: 'top center',
    marker: {
      size: valid.map(r => Math.max(8, Math.min(22, r.mention_count / 15))),
      color: valid.map(r => r.pearson_r != null ? r.pearson_r : 0),
      colorscale: 'RdYlGn',
      showscale: true,
      colorbar: {title: 'Pearson r', tickfont: {color: '#94a3b8'}, titlefont: {color: '#94a3b8'}},
      line: {width: 1, color: '#1a2d45'}
    },
    hovertemplate: '<b>%{text}</b><br>Mentions: %{x}<br>Vol Spike: %{y:.4f}<extra></extra>',
    textfont: {color: '#475569', size: 10}
  }], {
    paper_bgcolor: '#0d1b2e',
    plot_bgcolor: '#0a1525',
    font: {color: '#94a3b8', family: 'SF Mono,monospace', size: 11},
    title: {text: 'Mention Count vs. Volume Spike Ratio', font: {color: '#c9d4e8', size: 13}},
    xaxis: {title: 'Mention Count', gridcolor: '#1a2d45', zerolinecolor: '#1a2d45', color: '#475569'},
    yaxis: {title: 'Volume Spike Ratio', gridcolor: '#1a2d45', zerolinecolor: '#1a2d45', color: '#475569'},
    hovermode: 'closest',
    margin: {t: 48, r: 24, b: 52, l: 64}
  }, {responsive: true, displayModeBar: false});
}

// ── Recent Mentions Feed ──────────────────────────────────────────────────
function renderFeed(rows) {
  const list = document.getElementById('feed-list');
  document.getElementById('feed-count').textContent = rows.length + ' mentions';
  if (!rows.length) { list.innerHTML = '<p class="empty">No mentions found.</p>'; return; }
  list.innerHTML = rows.map(m => {
    const date = (m.published_date || '').substring(0, 10);
    const slug = slugLabel(m.substack_slug || '');
    const ctx = m.mention_context ? '“…' + m.mention_context.substring(0, 110) + '…”' : '';
    const titleHtml = m.url
      ? `<a href="${m.url}" target="_blank" rel="noopener">${m.title || 'Untitled'}</a>`
      : (m.title || 'Untitled');
    return `<div class="mention-card">
      <div class="m-ticker">${m.ticker}</div>
      <div>
        <div class="m-title">${titleHtml}</div>
        <div class="m-ctx">${ctx}</div>
      </div>
      <div class="m-meta">
        <div class="m-date">${date}</div>
        <div class="m-slug">${slug}</div>
      </div>
    </div>`;
  }).join('');
}

function applyFeed() {
  const q = document.getElementById('feed-search').value.toUpperCase().trim();
  if (!q) { renderFeed(recent_mentions); return; }
  renderFeed(recent_mentions.filter(m =>
    (m.ticker || '').toUpperCase().includes(q) ||
    (m.substack_slug || '').toUpperCase().includes(q)
  ));
}

// ── Init ──────────────────────────────────────────────────────────────────
applyTable();
renderConsensus();
renderFeed(recent_mentions);
</script>
</body>
</html>"""
