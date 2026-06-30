"use strict";

// Vacío = mismo origen: el backend FastAPI sirve este frontend en `/`, así que
// las requests van al mismo host (anda igual local y desplegado).
const API_BASE = "";

// Etiquetas cortas por categoría (se completan desde /categories).
let CATEGORY_LABELS = {
  us: "US", arg_stock: "Acción ARG", arg_cedear: "CEDEAR ARG",
  arg_bond: "Bono soberano ARG", arg_corp: "ON ARG",
};

const DEFAULT_POSITIONS = [
  { ticker: "GGAL", source: "arg_stock", value: 200000 },
  { ticker: "YPFD", source: "arg_stock", value: 150000 },
  { ticker: "AL30", source: "arg_bond", value: 200000 },
  { ticker: "GD30", source: "arg_bond", value: 150000 },
  { ticker: "AAPL", source: "us", value: 1000 },
];

// Estado: posiciones de la cartera + cache del universo por categoría.
let positions = [];
const universeCache = {};
// La tasa libre de riesgo se carga en vivo; marcamos si el usuario la editó a mano.
let rfTouched = false;

const PLOT_FONT = { family: "Inter, sans-serif", color: "#e8eef7" };
const PLOT_BG = "rgba(0,0,0,0)";

function baseLayout(extra = {}) {
  return Object.assign({
    paper_bgcolor: PLOT_BG,
    plot_bgcolor: PLOT_BG,
    font: PLOT_FONT,
    margin: { l: 55, r: 20, t: 20, b: 45 },
    xaxis: { gridcolor: "#1e3a63", zerolinecolor: "#1e3a63" },
    yaxis: { gridcolor: "#1e3a63", zerolinecolor: "#1e3a63" },
    legend: { orientation: "h", y: -0.2, font: { size: 11 } },
  }, extra);
}
const PLOT_CONFIG = { responsive: true, displayModeBar: false };

const pct = (x) => (x * 100).toFixed(1) + "%";
const pct2 = (x) => (x * 100).toFixed(2) + "%";

// ---------- Posiciones (categoría → universo → monto) ----------
function renderPositions() {
  const el = document.getElementById("positions");
  if (!positions.length) {
    el.innerHTML = `<div class="positions-empty">Todavía no agregaste activos.</div>`;
    return;
  }
  el.innerHTML = positions.map((p, i) => `
    <div class="position">
      <span class="pos-badge">${escapeHtml(CATEGORY_LABELS[p.source] || p.source)}</span>
      <span class="pos-ticker">${escapeHtml(p.ticker)}</span>
      <input class="pos-amount" type="number" min="0" value="${p.value}" data-i="${i}" />
      <button class="pos-remove" data-i="${i}" title="Quitar">✕</button>
    </div>`).join("");
  el.querySelectorAll(".pos-remove").forEach(b =>
    b.addEventListener("click", () => { positions.splice(+b.dataset.i, 1); renderPositions(); }));
  el.querySelectorAll(".pos-amount").forEach(inp =>
    inp.addEventListener("change", () => {
      const v = parseFloat(inp.value);
      positions[+inp.dataset.i].value = v > 0 ? v : 0;
    }));
}

function addPosition(ticker, source, value) {
  ticker = String(ticker || "").trim().toUpperCase();
  value = parseFloat(value);
  if (!ticker || !(value > 0)) return false;
  // Si ya existe el mismo ticker+categoría, sumamos el monto.
  const ex = positions.find(p => p.ticker === ticker && p.source === source);
  if (ex) ex.value += value; else positions.push({ ticker, source, value });
  renderPositions();
  return true;
}

// Universo de la categoría actual + estado del combo buscable.
let currentUniverse = [];
let comboActive = -1;

async function loadUniverse(category) {
  const status = document.getElementById("universe-status");
  status.textContent = "Cargando universo…";
  currentUniverse = [];
  try {
    if (!universeCache[category]) {
      const res = await fetch(`${API_BASE}/universe?category=${category}`);
      universeCache[category] = (await res.json()).instruments || [];
    }
    currentUniverse = universeCache[category];
    status.textContent = `${currentUniverse.length} instrumentos en ${CATEGORY_LABELS[category] || category}. Escribí para filtrar.`;
  } catch (e) {
    status.textContent = "No se pudo cargar el universo de esta categoría.";
  }
}

const RENDER_CAP = 200;

function fmtPrice(p) {
  return p == null ? "" : Number(p).toLocaleString("es-AR", { maximumFractionDigits: 2 });
}

function renderCombo(query) {
  const panel = document.getElementById("combo-panel");
  const q = (query || "").trim().toUpperCase();
  const matches = currentUniverse.filter(it =>
    !q || it.symbol.toUpperCase().includes(q) ||
    (it.name && it.name.toUpperCase().includes(q)));
  comboActive = -1;

  if (!currentUniverse.length) {
    panel.innerHTML = `<div class="combo-empty">Universo no disponible.</div>`;
  } else if (!matches.length) {
    panel.innerHTML = `<div class="combo-empty">Sin resultados para “${escapeHtml(query)}”.</div>`;
  } else {
    const shown = matches.slice(0, RENDER_CAP);
    const head = `<div class="combo-count">${matches.length} resultado${matches.length === 1 ? "" : "s"}${matches.length > RENDER_CAP ? ` · mostrando ${RENDER_CAP}` : ""}</div>`;
    panel.innerHTML = head + shown.map((it, i) => {
      const nm = it.name && it.name !== it.symbol ? it.name : "";
      const right = it.price != null ? `<span class="px">${fmtPrice(it.price)}</span>` : `<span class="nm">${escapeHtml(nm)}</span>`;
      return `<div class="combo-opt" data-sym="${escapeHtml(it.symbol)}" data-i="${i}">
        <span class="sym">${escapeHtml(it.symbol)}</span>${right}</div>`;
    }).join("");
    panel.querySelectorAll(".combo-opt").forEach(opt =>
      opt.addEventListener("mousedown", e => {   // mousedown: antes que blur
        e.preventDefault();
        pickTicker(opt.dataset.sym);
      }));
  }
  panel.classList.remove("hidden");
}

function pickTicker(sym) {
  document.getElementById("ticker-pick").value = sym;
  hideCombo();
  document.getElementById("amount").focus();
}

function hideCombo() {
  document.getElementById("combo-panel").classList.add("hidden");
  comboActive = -1;
}

function moveCombo(delta) {
  const opts = [...document.querySelectorAll("#combo-panel .combo-opt")];
  if (!opts.length) return;
  comboActive = (comboActive + delta + opts.length) % opts.length;
  opts.forEach((o, i) => o.classList.toggle("active", i === comboActive));
  opts[comboActive].scrollIntoView({ block: "nearest" });
}

// ---------- Analyze ----------
async function analyze() {
  const status = document.getElementById("status");
  status.className = "status";

  if (positions.length < 2) {
    status.className = "status error";
    status.textContent = "Agregá al menos 2 activos a la cartera.";
    return;
  }
  const holdings = positions.map(p => ({ ticker: p.ticker, source: p.source, value: p.value }));

  const rfRaw = document.getElementById("rf").value;
  // Si el usuario no tocó el campo, mandamos null para usar (y atribuir) la tasa en vivo.
  const rf = (!rfTouched || rfRaw === "") ? null : parseFloat(rfRaw) / 100;
  const mwRaw = document.getElementById("max-weight").value;
  const body = {
    holdings,
    period: document.getElementById("period").value,
    risk_free_rate: rf,
    normalize_currency: document.getElementById("normalize").checked,
    fx_kind: document.getElementById("fx-kind").value,
    return_method: document.getElementById("return-method").value,
    // Vacío = sin tope (1.0); si no, porcentaje a fracción.
    max_weight: mwRaw === "" ? 1.0 : Math.min(Math.max(parseFloat(mwRaw) / 100, 0.05), 1.0),
    run_ai: document.getElementById("run-ai").checked,
  };

  const btn = document.getElementById("analyze");
  btn.disabled = true;
  status.innerHTML = `<span class="spinner"></span>Descargando datos de mercado y optimizando…`;

  try {
    const res = await fetch(`${API_BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Error ${res.status}`);
    }
    const data = await res.json();
    render(data);
    const dropped = (data.period.dropped || []).map(d => d.symbol);
    status.textContent = `Datos: ${data.period.start} → ${data.period.end}` +
      (dropped.length ? ` · Fuera del optimizador: ${dropped.join(", ")}` : "");
  } catch (e) {
    status.className = "status error";
    status.textContent = "Error: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

// ---------- Render ----------
function render(d) {
  document.getElementById("results").classList.remove("hidden");
  if (d.risk_free) {
    const rf = d.risk_free;
    document.getElementById("rf").value = rf.percent;
    document.getElementById("rf-source").textContent =
      `${rf.percent}% · ${rf.source}${rf.as_of ? " · " + rf.as_of : ""}`;
  }
  renderWarnings(d.warnings);
  renderOnInfo(d.on_info);
  renderCards(d);
  renderRiskGauge(d.risk);
  renderFrontier(d);
  renderAllocations(d);
  renderCorrelation(d.market.correlation, d.tickers);
  renderRebalancing(d.rebalancing, d.tickers);
  renderBacktest(d.walk_forward, d.backtest);
  renderRiskDims(d.risk);
  renderDiagnostics(d.diagnostics);
  renderAI(d.ai_analysis);
  document.getElementById("results").scrollIntoView({ behavior: "smooth" });
}

function renderWarnings(warnings) {
  const el = document.getElementById("warnings");
  if (!warnings || !warnings.length) { el.innerHTML = ""; return; }
  el.innerHTML = warnings.map(w => `<div class="warn">⚠ ${escapeHtml(w)}</div>`).join("");
}

function renderOnInfo(onInfo) {
  const panel = document.getElementById("on-panel");
  const syms = Object.keys(onInfo || {});
  if (!syms.length) { panel.classList.add("hidden"); return; }
  panel.classList.remove("hidden");
  const fmt = (x) => (x == null ? "—" : Number(x).toLocaleString("es-AR", { maximumFractionDigits: 2 }));
  const rows = syms.map(s => {
    const o = onInfo[s];
    return `<tr>
      <td class="sym">${escapeHtml(s)}</td>
      <td class="num">${fmt(o.last_price)}</td>
      <td>${escapeHtml(o.currency || "ARS")}</td>
      <td>${escapeHtml(o.maturity_date || "—")}</td>
      <td class="num">${o.days_to_maturity ?? "—"}</td>
      <td class="num">${fmt(o.trade_volume)}</td>
    </tr>`;
  }).join("");
  document.getElementById("on-table").innerHTML = `
    <table class="on-table">
      <thead><tr><th>ON</th><th>Último</th><th>Moneda</th><th>Vencimiento</th><th>Días</th><th>Volumen</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function renderCards(d) {
  const cur = d.current_portfolio, opt = d.markowitz.max_sharpe, imp = d.rebalancing.improvement;
  const deltaTag = (v, invert = false) => {
    const good = invert ? v < 0 : v > 0;
    const arrow = v >= 0 ? "▲" : "▼";
    return `<div class="delta ${good ? "up" : "down"}">${arrow} ${pct(Math.abs(v))} vs actual</div>`;
  };
  const cards = [
    { label: "Sharpe actual", value: cur.sharpe.toFixed(2), delta: "" },
    { label: "Sharpe óptimo", value: opt.sharpe.toFixed(2), delta: deltaTag(imp.sharpe) },
    { label: "Retorno esp. (óptimo)", value: pct(opt.expected_return), delta: deltaTag(imp.expected_return) },
    { label: "Volatilidad (óptimo)", value: pct(opt.volatility), delta: deltaTag(imp.volatility, true) },
  ];
  document.getElementById("summary-cards").innerHTML = cards.map(c => `
    <div class="card">
      <div class="label">${c.label}</div>
      <div class="value">${c.value}</div>
      ${c.delta}
    </div>`).join("");
}

function riskColor(score) {
  if (score < 20) return "#22c55e";
  if (score < 40) return "#84cc16";
  if (score < 60) return "#f5a623";
  if (score < 80) return "#f97316";
  return "#ef4444";
}

function renderRiskGauge(risk) {
  const data = [{
    type: "indicator", mode: "gauge+number", value: risk.risk_score,
    number: { suffix: "/100", font: { size: 34 } },
    gauge: {
      axis: { range: [0, 100], tickcolor: "#8aa0c0" },
      bar: { color: riskColor(risk.risk_score) },
      bordercolor: "#1e3a63", borderwidth: 1,
      steps: [
        { range: [0, 20], color: "rgba(34,197,94,.15)" },
        { range: [20, 40], color: "rgba(132,204,22,.15)" },
        { range: [40, 60], color: "rgba(245,166,35,.15)" },
        { range: [60, 80], color: "rgba(249,115,22,.15)" },
        { range: [80, 100], color: "rgba(239,68,68,.15)" },
      ],
    },
  }];
  Plotly.newPlot("risk-gauge", data, baseLayout({ margin: { l: 20, r: 20, t: 10, b: 10 }, height: 240 }), PLOT_CONFIG);
  const band = document.getElementById("risk-band");
  band.textContent = `Nivel de riesgo: ${risk.risk_band.toUpperCase()}`;
  band.style.color = riskColor(risk.risk_score);
}

function renderFrontier(d) {
  // Submuestreo de la nube para usar SVG (scatter), no scattergl: la capa WebGL
  // se dibuja por encima de los marcadores SVG y tapaba el círculo rojo. Con SVG
  // el orden de las trazas define el z-order, así la cartera actual queda arriba.
  let mc = d.markowitz.monte_carlo;
  if (mc.length > 1500) {
    const step = Math.ceil(mc.length / 1500);
    mc = mc.filter((_, i) => i % step === 0);
  }
  const cloud = {
    x: mc.map(p => p.volatility), y: mc.map(p => p.expected_return),
    mode: "markers", type: "scatter", name: "Carteras simuladas",
    marker: { size: 5, color: mc.map(p => p.sharpe), colorscale: "Viridis", showscale: true,
      colorbar: { title: "Sharpe", thickness: 10, len: 0.6 }, opacity: 0.45 },
    hovertemplate: "σ %{x:.1%}<br>R %{y:.1%}<extra></extra>",
  };
  const fr = d.markowitz.frontier;
  const frontier = {
    x: fr.map(p => p.volatility), y: fr.map(p => p.expected_return),
    mode: "lines", type: "scatter", name: "Frontera eficiente",
    line: { color: "#2dd4bf", width: 3 },
  };
  const point = (p, name, color, symbol) => ({
    x: [p.volatility], y: [p.expected_return], mode: "markers", type: "scatter", name,
    marker: { size: 14, color, symbol, line: { color: "#fff", width: 1 } },
  });
  const cur = d.current_portfolio;
  const ms = d.markowitz.max_sharpe;

  // Capital Market Line: desde (0, rf) por la cartera tangente (máx Sharpe).
  const rf = (d.risk_free && typeof d.risk_free.rate === "number") ? d.risk_free.rate : 0;
  const slope = ms.volatility > 0 ? (ms.expected_return - rf) / ms.volatility : 0;
  const xMax = Math.max(...mc.map(p => p.volatility), ms.volatility, cur.volatility) * 1.1;
  const cml = {
    x: [0, xMax], y: [rf, rf + slope * xMax],
    mode: "lines", type: "scatter", name: "CML (tangencia)",
    line: { color: "#f5a623", width: 1.5, dash: "dash" },
    hovertemplate: "CML · Sharpe %{customdata:.2f}<extra></extra>",
    customdata: [slope, slope],
  };

  // Cartera actual: círculo rojo grande con borde blanco. Va como ÚLTIMA traza
  // para quedar siempre por encima de la nube y del resto de los puntos.
  const current = {
    x: [cur.volatility], y: [cur.expected_return], mode: "markers", type: "scatter",
    name: "Tu cartera (actual)",
    marker: { size: 18, color: "#ef4444", symbol: "circle",
      line: { color: "#fff", width: 2.5 } },
    hovertemplate: `Tu cartera<br>σ ${(cur.volatility * 100).toFixed(1)}%<br>R ${(cur.expected_return * 100).toFixed(1)}%<extra></extra>`,
  };

  const traces = [
    cloud, cml, frontier,
    point(d.markowitz.min_variance, "Mín Varianza", "#3b82f6", "diamond"),
    point(ms, "Máx Sharpe (tangencia)", "#f5a623", "star"),
    { x: [0], y: [rf], mode: "markers", type: "scatter", name: "Tasa libre de riesgo",
      marker: { size: 9, color: "#8aa0c0", symbol: "x" },
      hovertemplate: `rf ${(rf * 100).toFixed(2)}%<extra></extra>` },
    current,  // última = siempre arriba
  ];
  Plotly.newPlot("frontier", traces, baseLayout({
    xaxis: { title: "Volatilidad (σ anual)", tickformat: ".0%", gridcolor: "#1e3a63", rangemode: "tozero" },
    yaxis: { title: "Retorno esperado", tickformat: ".0%", gridcolor: "#1e3a63" },
  }), PLOT_CONFIG);
}

function renderAllocations(d) {
  const tickers = d.tickers;
  const cur = tickers.map(t => (d.current_portfolio.weights[t] || 0) * 100);
  const opt = tickers.map(t => (d.markowitz.max_sharpe.weights[t] || 0) * 100);
  const traces = [
    { x: tickers, y: cur, type: "bar", name: "Actual", marker: { color: "#3b82f6" } },
    { x: tickers, y: opt, type: "bar", name: "Óptimo (Max Sharpe)", marker: { color: "#2dd4bf" } },
  ];
  Plotly.newPlot("allocations", traces, baseLayout({
    barmode: "group",
    yaxis: { title: "Peso (%)", ticksuffix: "%", gridcolor: "#1e3a63" },
  }), PLOT_CONFIG);
}

const BT_META = {
  current: { label: "Actual", color: "#3b82f6" },
  max_sharpe: { label: "Óptimo (Max Sharpe)", color: "#2dd4bf" },
  min_variance: { label: "Mín varianza", color: "#f5a623" },
  equal_weight: { label: "Equal-weight (1/N)", color: "#8aa0c0" },
};

function renderRebalancing(reb, tickers) {
  if (!reb || !reb.actions) return;
  document.getElementById("rebal-ccy").textContent = reb.currency === "USD" ? "USD" : "moneda nativa";
  const money = (x) => (x < 0 ? "-" : "") + Math.abs(x).toLocaleString("es-AR", { maximumFractionDigits: 0 });
  const pfx = reb.currency === "USD" ? "US$ " : "";          // base del cómputo
  const actColor = { comprar: "#22c55e", vender: "#ef4444", mantener: "#8aa0c0" };
  // Celda con monto base + (si el activo es ARG) equivalente en pesos.
  const cell = (val, native) => `${pfx}${money(val)}` +
    (native != null ? `<br><span class="ars">$ ${money(native)} ARS</span>` : "");
  const rows = tickers.map(t => {
    const a = reb.actions[t]; if (!a) return "";
    const sign = a.delta_value > 0 ? "+" : "";
    const deltaNative = a.delta_native != null ? `<br><span class="ars">${sign}$ ${money(a.delta_native)} ARS</span>` : "";
    return `<tr>
      <td class="sym">${escapeHtml(t)}</td>
      <td class="num">${cell(a.current_value, a.current_native)}</td>
      <td class="num">${cell(a.target_value, a.target_native)}</td>
      <td style="color:${actColor[a.action]};font-weight:600">${a.action}</td>
      <td class="num" style="color:${actColor[a.action]}">${sign}${pfx}${money(a.delta_value)}${deltaNative}</td>
    </tr>`;
  }).join("");
  const total = `<div class="rebal-total">Total cartera: <strong>${pfx}${money(reb.invested)}</strong>` +
    ` — se mantiene constante; el rebalanceo solo redistribuye los pesos.</div>`;
  document.getElementById("rebal-table").innerHTML = total + `
    <table class="bt-table">
      <thead><tr><th>Activo</th><th>Actual</th><th>Objetivo</th><th>Acción</th><th>Monto</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function renderBacktest(wf, inSample) {
  const el = document.getElementById("backtest");
  const note = document.getElementById("backtest-note");
  // Preferimos walk-forward (out-of-sample); si la ventana no alcanza, in-sample.
  let bt;
  if (wf && wf.available && wf.dates && wf.dates.length) {
    bt = wf;
    note.innerHTML = `<strong>Out-of-sample (walk-forward):</strong> en cada rebalanceo (cada ${wf.params.rebalance} ruedas, ${wf.params.rebalances} en total) se re-optimiza usando <em>solo datos pasados</em> y se evalúa en los días siguientes. Sin look-ahead. Incluye benchmark equal-weight (1/N). No garantiza rendimiento futuro.`;
  } else {
    bt = inSample;
    const why = wf && wf.reason ? ` (${escapeHtml(wf.reason)})` : "";
    note.innerHTML = `<strong>In-sample (referencial):</strong> la óptima se calculó sobre el mismo período graficado → sesgo de look-ahead${why}. Usá un período más largo para ver el walk-forward out-of-sample.`;
  }
  if (!bt || !bt.dates || !bt.dates.length) { el.innerHTML = ""; note.textContent = ""; return; }
  const traces = Object.keys(bt.series).map(name => ({
    x: bt.dates, y: bt.series[name], type: "scatter", mode: "lines",
    name: (BT_META[name] || { label: name }).label,
    line: { color: (BT_META[name] || { color: "#8aa0c0" }).color, width: 2 },
  }));
  Plotly.newPlot("backtest", traces, baseLayout({
    yaxis: { title: "Valor (base 100)", gridcolor: "#1e3a63" },
    xaxis: { gridcolor: "#1e3a63", type: "date" },
    margin: { l: 55, r: 20, t: 20, b: 40 },
  }), PLOT_CONFIG);

  // Tabla de métricas
  const pctf = (x) => (x * 100).toFixed(1) + "%";
  const rows = Object.keys(bt.metrics).map(name => {
    const m = bt.metrics[name];
    const meta = BT_META[name] || { label: name, color: "#8aa0c0" };
    return `<tr>
      <td><span class="bt-dot" style="background:${meta.color}"></span>${escapeHtml(meta.label)}</td>
      <td class="num">${pctf(m.total_return)}</td>
      <td class="num">${pctf(m.cagr)}</td>
      <td class="num">${pctf(m.volatility)}</td>
      <td class="num">${m.sharpe.toFixed(2)}</td>
      <td class="num">${pctf(m.max_drawdown)}</td>
    </tr>`;
  }).join("");
  document.getElementById("backtest-metrics").innerHTML = `
    <table class="bt-table">
      <thead><tr><th>Cartera</th><th>Retorno total</th><th>CAGR</th><th>Volatilidad</th><th>Sharpe</th><th>Máx drawdown</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function renderCorrelation(corr, tickers) {
  const z = tickers.map(a => tickers.map(b => corr[a][b]));
  const data = [{
    z, x: tickers, y: tickers, type: "heatmap",
    colorscale: [[0, "#3b82f6"], [0.5, "#0f1d36"], [1, "#ef4444"]],
    zmin: -1, zmax: 1, hovertemplate: "%{y} · %{x}: %{z:.2f}<extra></extra>",
    colorbar: { thickness: 10, len: 0.7 },
  }];
  const annotations = [];
  tickers.forEach((a, i) => tickers.forEach((b, j) => {
    annotations.push({ x: b, y: a, text: z[i][j].toFixed(2), showarrow: false,
      font: { size: 10, color: "#e8eef7" } });
  }));
  Plotly.newPlot("correlation", data, baseLayout({
    annotations, margin: { l: 60, r: 20, t: 20, b: 60 },
    yaxis: { autorange: "reversed", gridcolor: "#1e3a63" },
  }), PLOT_CONFIG);
}

function bandColor(band) {
  return { bajo: "#22c55e", moderado: "#f5a623", elevado: "#f97316", alto: "#ef4444", "crítico": "#dc2626" }[band] || "#8aa0c0";
}

function renderRiskDims(risk) {
  const html = risk.dimensions.map((dim, i) => {
    const color = bandColor(dim.band);
    const num = String(i + 1).padStart(2, "0");
    return `
      <div class="dim" style="border-left-color:${color}">
        <div class="dim-head">
          <span class="dim-title">${num} · ${dim.title}</span>
          <span class="dim-sev" style="background:${color}22;color:${color}">${dim.band} · ${dim.severity}</span>
        </div>
        <div class="dim-summary">${dim.summary}</div>
        <div class="bar"><span style="width:${dim.severity}%;background:${color}"></span></div>
      </div>`;
  }).join("");
  document.getElementById("risk-dims").innerHTML = html;
}

function renderDiagnostics(diag) {
  const panel = document.getElementById("diagnostics-panel");
  if (!diag || !panel) return;
  if (!diag.available) {
    panel.classList.remove("hidden");
    document.getElementById("diag-body").innerHTML =
      `<p class="hint">${escapeHtml(diag.reason || "No disponible.")}</p>`;
    return;
  }
  panel.classList.remove("hidden");
  // Cada test: verde si NO rechaza el supuesto, ámbar si lo rechaza.
  const rows = diag.tests.map(t => {
    const ok = !t.reject;
    const color = ok ? "#22c55e" : "#f5a623";
    const tag = ok ? "se cumple" : "no se cumple";
    return `
      <div class="diag" style="border-left-color:${color}">
        <div class="diag-head">
          <span class="diag-title">${escapeHtml(t.title)}</span>
          <span class="diag-tag" style="background:${color}22;color:${color}">
            ${tag} · p=${t.p_value.toFixed(3)}
          </span>
        </div>
        <div class="diag-summary">${escapeHtml(t.verdict)}</div>
      </div>`;
  }).join("");
  document.getElementById("diag-body").innerHTML =
    `<div class="diag-grid">${rows}</div>
     <p class="diag-interp">${escapeHtml(diag.interpretation)}</p>`;
}

function renderAI(ai) {
  const badge = document.getElementById("ai-badge");
  const report = document.getElementById("ai-report");
  if (ai.skipped) {
    badge.textContent = ""; report.innerHTML = `<p class="hint">El análisis con IA fue desactivado.</p>`;
    return;
  }
  badge.className = "badge " + (ai.is_mock ? "mock" : "real");
  badge.textContent = ai.is_mock ? "MOCK (sin API key)" : ai.model;

  const findings = (ai.key_findings || []).map(f => `<li>${escapeHtml(f)}</li>`).join("");
  const hedges = (ai.hedging_strategies || []).map(h => `
    <div class="hedge">
      <div class="risk">${escapeHtml(h.risk)}</div>
      <div class="instr">→ ${escapeHtml(h.instrument)}</div>
      <div class="why">${escapeHtml(h.rationale)}</div>
    </div>`).join("");

  report.innerHTML = `
    <div class="verdict"><strong>Veredicto:</strong> ${escapeHtml(ai.risk_verdict)}</div>
    <div class="ai-section"><h4>Resumen ejecutivo</h4><p>${escapeHtml(ai.executive_summary)}</p></div>
    <div class="ai-section"><h4>Hallazgos clave</h4><ul>${findings}</ul></div>
    <div class="ai-section"><h4>Estrategias de cobertura</h4>${hedges}</div>
    <div class="ai-section"><h4>Justificación del rebalanceo</h4><p>${escapeHtml(ai.rebalancing_rationale)}</p></div>
    <div class="disclaimer">${escapeHtml(ai.disclaimer || "")}</div>`;
}

function escapeHtml(s) {
  return String(s || "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------- Categorías + atajos + tasa en vivo ----------
async function loadCategories() {
  const sel = document.getElementById("cat");
  try {
    const res = await fetch(`${API_BASE}/categories`);
    const cats = (await res.json()).categories || [];
    CATEGORY_LABELS = Object.fromEntries(cats.map(c => [c.value, c.label]));
    sel.innerHTML = cats.map(c =>
      `<option value="${c.value}">${escapeHtml(c.label)}</option>`).join("");
  } catch (e) {
    sel.innerHTML = Object.entries(CATEGORY_LABELS).map(([v, l]) =>
      `<option value="${v}">${l}</option>`).join("");
  }
  loadUniverse(sel.value);
}

function wireAddForm() {
  const cat = document.getElementById("cat");
  const pick = document.getElementById("ticker-pick");
  const amount = document.getElementById("amount");
  cat.addEventListener("change", () => { pick.value = ""; loadUniverse(cat.value); hideCombo(); });

  // Combo buscable
  pick.addEventListener("input", () => renderCombo(pick.value));
  pick.addEventListener("focus", () => renderCombo(pick.value));
  pick.addEventListener("keydown", e => {
    const open = !document.getElementById("combo-panel").classList.contains("hidden");
    if (e.key === "ArrowDown") { e.preventDefault(); if (!open) renderCombo(pick.value); else moveCombo(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); moveCombo(-1); }
    else if (e.key === "Escape") { hideCombo(); }
    else if (e.key === "Enter") {
      e.preventDefault();
      const active = document.querySelector("#combo-panel .combo-opt.active");
      if (active) { pickTicker(active.dataset.sym); }
      else { document.getElementById("add-pos").click(); }
    }
  });
  document.addEventListener("click", e => {
    if (!e.target.closest(".combo")) hideCombo();
  });

  document.getElementById("add-pos").addEventListener("click", () => {
    if (addPosition(pick.value, cat.value, amount.value)) {
      pick.value = ""; amount.value = ""; hideCombo(); pick.focus();
    }
  });
  amount.addEventListener("keydown", e => {
    if (e.key === "Enter") document.getElementById("add-pos").click();
  });
  document.querySelectorAll(".chip[data-add]").forEach(chip => {
    chip.addEventListener("click", () => {
      const source = chip.dataset.add;
      chip.dataset.tickers.split(",").forEach(t => addPosition(t.trim(), source, 10000));
    });
  });
  document.getElementById("rf").addEventListener("input", () => { rfTouched = true; });
}

async function loadLiveRate() {
  const src = document.getElementById("rf-source");
  try {
    const res = await fetch(`${API_BASE}/risk-free-rate`);
    const rf = await res.json();
    document.getElementById("rf").value = rf.percent;
    src.textContent = `${rf.percent}% · ${rf.source}${rf.as_of ? " · " + rf.as_of : ""}`;
  } catch (e) {
    document.getElementById("rf").value = 4.0;
    src.textContent = "No se pudo cargar la tasa en vivo; usando 4% (editable).";
  }
}

// ---------- Init ----------
document.getElementById("footer-year").textContent = new Date().getFullYear();
positions = DEFAULT_POSITIONS.slice();
renderPositions();
loadCategories();
wireAddForm();
loadLiveRate();
document.getElementById("analyze").addEventListener("click", analyze);
