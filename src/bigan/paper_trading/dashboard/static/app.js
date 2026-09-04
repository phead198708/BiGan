"use strict";

// No external data is interpreted as markup; all values become text nodes.
const $ = (id) => document.getElementById(id);
const number = (v, digits = 2) => typeof v === "number" && Number.isFinite(v)
  ? v.toLocaleString("en-US", {maximumFractionDigits: digits, minimumFractionDigits: digits}) : "—";
const money = (v) => typeof v === "number" && Number.isFinite(v) ? "$" + number(v) : "—";
const percent = (v) => typeof v === "number" && Number.isFinite(v) ? number(v * 100) + "%" : "—";
const when = (v) => typeof v === "number" && Number.isFinite(v) && !Number.isNaN(new Date(v).valueOf())
  ? new Date(v).toISOString().replace("T", " ").replace("Z", " UTC") : "—";
const text = (v) => v === null || v === undefined || v === "" ? "—" : String(v);
const flag = (v) => v === true ? "✓ Yes" : v === false ? "× No" : "— unavailable";
const age = (v) => typeof v === "number" && Number.isFinite(v) ? number(v / 1000, 1) + " s" : "—";
const binanceName = (status) => status.feeds?.binance?.venue === "us" ? "Binance.US"
  : status.feeds?.binance?.venue === "global" ? "Binance Global" : "Binance · venue unavailable";
function el(tag, value, className) {
  const node = document.createElement(tag);
  if (value !== undefined) node.textContent = text(value);
  if (className) node.className = className;
  return node;
}
function fields(id, values) {
  const target = $(id);
  target.replaceChildren();
  for (const [label, value] of values) target.append(el("dt", label), el("dd", value));
}
function table(id, rows, columns, empty) {
  const target = $(id);
  target.replaceChildren();
  if (!Array.isArray(rows) || rows.length === 0) {
    target.append(el("p", rows === null || rows === undefined ? "unavailable — retrying on the next refresh" : empty, "empty"));
    return;
  }
  const tbl = el("table"), head = el("thead"), hr = el("tr"), body = el("tbody");
  for (const [label] of columns) { const th = el("th", label); th.scope = "col"; hr.append(th); }
  head.append(hr);
  for (const row of rows) {
    const tr = el("tr");
    for (const [, getter] of columns) tr.append(el("td", getter(row)));
    body.append(tr);
  }
  tbl.append(head, body); target.append(tbl);
}

let last = null, receivedAt = 0, cursor = null, olderCursor = null, failed = false, busy = false, timer;
let priceClocks = [];
const price = (v, digits = 4) => typeof v === "number" && Number.isFinite(v) ? "$" + number(v, digits) : "—";
function priceCard(target, key, label, value, unit, rows, observation) {
  const card = el("article", undefined, "price-card " + key);
  card.id = "price-" + key;
  const badge = el("span", "— unavailable", "badge warn"), dl = el("dl", undefined, "fields");
  card.append(el("h3", label), el("span", value, "price-value"), el("span", unit, "price-unit"), el("br"), badge, dl);
  for (const [name, v] of rows) dl.append(el("dt", name), el("dd", v));
  if (observation) {
    const ageNode = el("dd", age(observation.age_ms));
    dl.append(el("dt", "数据年龄 / Age"), ageNode);
    priceClocks.push({badge, ageNode, observation});
  } else {
    badge.textContent = value === "—" ? "不可用 · Unavailable" : "固定开盘参考 · Fixed at window start";
  }
  $(target).append(card);
}
function renderPrices(view) {
  priceClocks = [];
  $("reference-prices").replaceChildren(); $("contract-prices").replaceChildren();
  const market = view.active_market, data = view.status.market_data;
  // Do not mix a previous window's observations with the new opening reference.
  const current = market && data?.window_id === market.window_id && data?.market_id === market.market_id ? data : {};
  $("price-market").textContent = market ? text(market.slug) + " · " + when(market.window_start_ts_ms) + " → " + when(market.window_end_ts_ms) : "No active market · 暂无市场";
  const proof = market?.opening_reference;
  const lookback = market?.oracle_twap_lookback_seconds;
  priceCard("reference-prices", "opening", "开盘价 · Price to beat", price(market?.reference_price_at_start),
    lookback ? "USD · 开盘 " + lookback + "s TWAP" : "USD · 窗口开盘参考价", [
      ["Source", proof?.source_endpoint || market?.source_endpoint],
      ["开窗 (UTC)", when(market?.window_start_ts_ms)],
      ["来源时间 (UTC)", when(proof?.source_ts_ms ?? market?.discovered_at_ms)],
      ["来源校验", proof ? "Public endpoint · identity / payload hash (not a signed oracle report)" : "Market metadata · no separate opening proof"],
    ]);
  for (const [key, label] of [["oracle", "当前参考价 · Chainlink"], ["spot", "当前现货价 · " + binanceName(view.status)]]) {
    const o = current[key] || {};
    const model = key === "spot" ? "bid/ask midpoint" : o.kind === "published_twap" ? text(o.lookback_seconds) + "s published TWAP" : "Oracle price";
    priceCard("reference-prices", key, label, price(o.value), text(o.symbol) + " · " + text(o.quote_currency) + " · " + model, [
      ["Source", o.source], ["事件 (UTC)", when(o.timestamp_ms)], ["接收 (UTC)", when(o.received_at_ms)],
    ], {...o, available: typeof o.value === "number"});
  }
  for (const [key, label, token] of [["up", "UP / YES · 上涨", market?.yes_token_id], ["down", "DOWN / NO · 下跌", market?.no_token_id]]) {
    const candidate = current[key];
    const o = token && candidate?.token_id === token ? candidate : {};
    priceCard("contract-prices", key, label, price(o.ask, 6), "买入 Ask · USDC / share", [
      ["卖出 Bid", price(o.bid, 6)], ["Ask 数量 / shares", number(o.ask_size, 4)], ["Bid 数量 / shares", number(o.bid_size, 4)],
      ["Source", o.source], ["Token", token], ["事件 (UTC)", when(o.timestamp_ms)],
    ], {...o, available: typeof o.ask === "number" && typeof o.bid === "number"});
  }
}
function clocks() {
  if (!last) return;
  const elapsed = performance.now() - receivedAt;
  const statusAge = last.status_age_ms + elapsed;
  const stale = last.stale || statusAge > last.stale_after_ms;
  const banner = $("error-banner");
  banner.hidden = !failed && !stale;
  banner.textContent = failed
    ? "⚠ Refresh failed. Keeping the last successful view; data may be stale. Retrying automatically."
    : "⚠ Operator status is stale. Values below are the last reported observations, not live data.";
  $("poll-status").textContent = "Status age " + age(statusAge) + " · " + (stale ? "STALE" : "within freshness limit") + " · polls every 2 s";
  const end = last.active_market?.window_end_ts_ms;
  if (typeof end === "number") {
    const remaining = Math.max(0, Math.floor((end - last.generated_at_ms - elapsed) / 1000));
    $("countdown").textContent = remaining === 0 ? "Window ended" : Math.floor(remaining / 60) + "m " + remaining % 60 + "s remaining";
  } else $("countdown").textContent = "—";
  for (const {badge, ageNode, observation: o} of priceClocks) {
    const currentAge = typeof o.age_ms === "number" ? o.age_ms + statusAge : null;
    ageNode.textContent = age(currentAge);
    const stopped = ["STOPPING", "STOPPED", "FAILED", "EXHAUSTED"].includes(last.status.state);
    const expired = typeof end === "number" && last.generated_at_ms + elapsed >= end;
    const fresh = o.fresh === true && !failed && !stale && !stopped && !expired && currentAge !== null && currentAge >= 0 && currentAge <= o.max_age_ms;
    badge.textContent = !o.available ? (o.connected === true && o.confirmed === false ? "待确认 · Syncing" : "不可用 · Unavailable")
      : stopped || o.connected === false ? "已断开 · Disconnected" : fresh ? "新鲜 · Fresh" : "已过期 / 未确认 · Stale";
    badge.className = "badge " + (fresh && o.available ? "good" : "warn");
  }
}

function render(view) {
  const status = view.status, account = view.account, market = view.active_market;
  last = view; receivedAt = performance.now();
  renderPrices(view);
  const badge = $("state-badge");
  const state = status.state;
  badge.textContent = (state === "RUNNING" ? "✓ " : state === "FAILED" ? "× " : "◷ ") + text(state);
  badge.className = "badge " + (state === "RUNNING" ? "good" : state === "FAILED" ? "bad" : "warn");
  const warnings = $("warnings"); warnings.replaceChildren();
  for (const warning of view.warnings || []) {
    if (warning.code !== "STATUS_STALE") warnings.append(el("div", "⚠ " + warning.message, "notice"));
  }
  const kpis = $("account-kpis"); kpis.replaceChildren();
  for (const [label, key, format] of [
    ["Equity", "equity", money], ["Cash", "cash", money], ["Total PnL", "total_pnl", money], ["Initial bankroll", "initial_bankroll", money],
    ["Realized PnL", "realized_pnl", money], ["Unrealized PnL", "unrealized_pnl", money], ["Commission / fees", "total_fees", money], ["Drawdown · current run", "drawdown", percent],
  ]) {
    const card = el("div", undefined, "kpi");
    card.append(el("span", label, "label"), el("span", format(account?.[key]), "value")); kpis.append(card);
  }
  fields("operator-fields", [
    ["Reason", status.state_reason], ["Operator", status.operator_id], ["Strategy", status.strategy_id], ["Run", status.run_id],
    ["Source commit", status.source_commit], ["Paper only", flag(status.paper_only)],
    ["Capital at risk", flag(status.safety?.capital_at_risk)], ["Wallet signing", flag(status.safety?.wallet_signing_enabled)],
    ["Exchange writes", flag(status.safety?.live_exchange_write_enabled)], ["Updated (UTC)", when(status.updated_at_ms)],
    ["Account observed (UTC)", when(account?.timestamp_ms)],
  ]);
  $("market-name").textContent = text(market?.title);
  fields("market-fields", [
    ["Slug", market?.slug], ["Market / window", market ? text(market.market_id) + " / " + text(market.window_id) : "—"],
    ["Starts (UTC)", when(market?.window_start_ts_ms)], ["Ends (UTC)", when(market?.window_end_ts_ms)],
    ["YES token", market?.yes_token_id], ["NO token", market?.no_token_id],
    ["Settlement", status.settlement?.status], ["Source", market?.resolution_source], ["Reference", status.settlement?.source_reference],
  ]);
  table("positions-table", view.positions, [
    ["Market / window", r => text(r.market_symbol) + " / " + text(r.window_id)], ["Side", r => r.side], ["Shares", r => number(r.shares, 4)],
    ["Avg entry", r => number(r.average_entry_price, 4)], ["Current mark (bid)", r => number(r.mark_bid, 4)],
    ["Cost basis", r => money(r.cost_usdc)], ["Unrealized PnL", r => money(r.unrealized_pnl)], ["Opened (UTC)", r => when(r.opened_at_ms)],
  ], "No open positions.");
  const feeds = $("feed-cards"); feeds.replaceChildren();
  for (const [name, label] of [["binance", binanceName(status)], ["polymarket", "Polymarket"], ["chainlink", "Chainlink"]]) {
    const feed = status.feeds?.[name];
    const card = el("div", undefined, "feed-card"), dl = el("dl", undefined, "fields");
    card.append(el("h3", label), dl);
    for (const [key, value] of [
      ["State", feed?.state], ["Connected", flag(feed?.connected)], ["Synchronized", flag(feed?.synchronized)], ["Fresh", flag(feed?.fresh)],
      ["Event (UTC)", when(feed?.last_event_ts_ms)], ["Reported age", age(feed?.age_ms)], ["Gaps", feed?.gap_count],
      ["Reconnects", feed?.reconnect_count], ["Errors", feed?.error_count],
    ]) dl.append(el("dt", key), el("dd", value));
    feeds.append(card);
  }
  const alpha = status.alpha, pricing = status.pricing_inputs;
  fields("alpha-fields", [
    ["Alpha venue / source", text(alpha?.venue) + " / " + text(alpha?.source)],
    ["OFI Z-score", number(alpha?.z_score, 4)], ["Alpha timestamp (UTC)", when(alpha?.timestamp_ms)], ["Alpha age", age(alpha?.age_ms)], ["Alpha fresh", flag(alpha?.fresh)],
    ["Pricing ready / fresh", flag(pricing?.ready) + " / " + flag(pricing?.fresh)], ["Pricing time (UTC)", when(pricing?.timestamp_ms)], ["Pricing age", age(pricing?.age_ms)],
    ["Spot samples", pricing?.spot_sample_count], ["Oracle / TWAP samples", pricing?.oracle_sample_count], ["Volatility return samples", pricing?.return_sample_count],
    ["Oracle feed fresh", flag(status.feeds?.chainlink?.fresh)],
    ["Spot / Oracle observations", "See Market data above; volatility readiness is reported separately"],
  ]);
  const input = status.last_decision;
  fields("inputs-fields", [
    ["Decision (UTC)", when(input?.timestamp_ms)], ["Input time (UTC)", when(input?.pricing_inputs_timestamp_ms)], ["Fresh at decision", flag(input?.pricing_inputs_are_fresh)],
    ["Spot", money(input?.spot_price)], ["Oracle TWAP", money(input?.oracle_twap_so_far)], ["TWAP weight", percent(input?.twap_weight)],
    ["Annualized volatility", percent(input?.volatility_annualized)],
  ]);
  const recent = view.recent;
  table("decisions-table", recent.decisions?.slice().reverse(), [
    ["Time (UTC)", r => when(r.decision.timestamp_ms)], ["Run", r => r.run_id], ["Direction", r => r.decision.direction],
    ["Disposition / reason", r => text(r.decision.disposition) + " / " + text(r.decision.reason_code)], ["Model P", r => percent(r.decision.model_probability)],
    ["Market price", r => number(r.decision.market_price, 4)], ["Effective strike", r => money(r.decision.effective_strike)],
    ["Edge", r => percent(r.decision.edge)], ["EV", r => number(r.decision.ev, 4)], ["Kelly size", r => percent(r.decision.recommended_size_pct)],
    ["Order / fill", r => r.decision.order_status], ["Cash before → after", r => money(r.decision.cash_before) + " → " + money(r.decision.cash_after)],
  ], "No decisions in this bounded page.");
  table("fills-table", recent.fills?.slice().reverse(), [
    ["Time (UTC)", r => when(r.decision.timestamp_ms)], ["Run", r => r.run_id], ["Side", r => r.decision.order_side], ["Shares", r => number(r.decision.shares, 4)],
    ["Fill price", r => number(r.decision.fill_price, 4)], ["Fee", r => money(r.decision.fee_usdc)], ["Order ID", r => r.decision.order_id],
    ["Signal event ID", r => r.event_id], ["Cash after", r => money(r.decision.cash_after)],
  ], "No fills in this bounded page. This is not an error.");
  table("settlements-table", recent.settlements?.slice().reverse(), [
    ["Run / window", r => text(r.run_id) + " / " + text(r.settlement.window_id)],
    ["Outcome", r => r.settlement.yes_payout === 1 ? "YES" : r.settlement.yes_payout === 0 ? "NO" : "Fractional payout"],
    ["YES payout", r => number(r.settlement.yes_payout, 4)], ["Proceeds", r => money(r.proceeds_usdc)], ["Realized PnL delta", r => money(r.realized_pnl_delta)],
    ["Cash after", r => money(r.cash_after)], ["Source / reference", r => text(r.settlement.source) + " / " + text(r.settlement.source_reference)], ["Settled (UTC)", r => when(r.settlement.settlement_ts_ms)],
  ], "No settlements in this bounded page.");
  table("runs-table", recent.runs, [
    ["Index", r => r.run_index], ["Run ID", r => r.run_id], ["Market / window", r => text(r.market_title) + " / " + text(r.window_ids?.join(", "))],
    ["Opening cash", r => money(r.opening_cash)], ["Settled", r => flag(r.settled)], ["Cash", r => money(r.cash)], ["Predecessor", r => r.predecessor_run_id],
  ], "No runs in this bounded page. Use Latest to return.");
  const oldest = recent.runs?.at(-1);
  olderCursor = oldest?.run_index > 0 ? oldest.run_id : null;
  $("history-page").textContent = cursor ? "Older than " + cursor + " · exclusive run cursor" : "Newest bounded page · up to " + view.query_defaults.limit + " rows per section";
  $("older").disabled = !olderCursor;
  clocks();
}

async function refresh() {
  if (busy) return;
  busy = true; clearTimeout(timer);
  $("older").disabled = true; $("latest").disabled = true;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 8000);
  try {
    const query = cursor ? "?before_run_id=" + encodeURIComponent(cursor) : "";
    const response = await fetch("/api/v1/dashboard" + query, {cache: "no-store", signal: controller.signal});
    if (!response.ok) throw new Error("unavailable");
    const view = await response.json();
    if (view.schema_version !== 1 || !view.status || !view.recent) throw new Error("invalid response");
    failed = false; render(view);
  } catch {
    failed = true;
    if (!last) { $("error-banner").hidden = false; $("error-banner").textContent = "⚠ Operator status unavailable. Check the local operator and config. Retrying automatically."; }
    else clocks();
  } finally {
    clearTimeout(timeout); busy = false; $("latest").disabled = false; $("older").disabled = !olderCursor;
    timer = setTimeout(refresh, 2000);
  }
}
$("older").addEventListener("click", () => { if (!busy && olderCursor) { cursor = olderCursor; refresh(); } });
$("latest").addEventListener("click", () => { if (!busy) { cursor = null; refresh(); } });
setInterval(clocks, 1000);
refresh();
