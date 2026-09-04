// Optional browser acceptance with an externally installed Playwright.
// No npm/build pipeline or browser dependency is added to the package.
// node browser_smoke.cjs http://127.0.0.1:8088 docs/paper_trading/images
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const path = require("node:path");
const fs = require("node:fs/promises");

(async () => {
  const url = process.argv[2] || "http://127.0.0.1:8088";
  assert(["127.0.0.1", "[::1]"].includes(new URL(url).hostname));
  const directory = path.resolve(process.argv[3] || "dashboard-screenshots");
  await fs.mkdir(directory, {recursive: true});
  const browser = await chromium.launch({headless: true, channel: process.env.BROWSER_CHANNEL || undefined});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1100}});
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.goto(url);
    await page.waitForFunction(() => document.querySelectorAll(".kpi").length === 8);
    assert.equal(await page.locator(".price-card").count(), 5);
    assert.match(await page.locator("#price-up").innerText(), /Ask.*USDC \/ share/);
    assert(await page.locator(".paper-banner").isVisible());
    assert.match(await page.locator(".paper-banner").innerText(), /PAPER \/ SIMULATED — NO REAL FUNDS/);
    assert.equal(await page.locator("#account-kpis").innerText().then(t => t.includes("NaN")), false);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({path: path.join(directory, "dashboard-desktop.png")});
    await page.setViewportSize({width: 390, height: 844});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({path: path.join(directory, "dashboard-mobile.png")});

    const before = await page.locator("#account-kpis").innerText();
    await page.route("**/api/v1/dashboard*", route => route.fulfill({status: 503, contentType: "application/json", body: "{}"}));
    await page.waitForFunction(() => document.getElementById("error-banner").textContent.includes("Refresh failed"));
    assert.equal(await page.locator("#account-kpis").innerText(), before);
    assert.equal(await page.locator(".price-card .badge.good").count(), 0);
    await page.unroute("**/api/v1/dashboard*");
    await page.waitForFunction(() => !document.getElementById("error-banner").textContent.includes("Refresh failed"));

    const response = await page.request.get(url + "/api/v1/dashboard");
    const view = await response.json();
    // Independent observations: no decision, no Binance, but Oracle and both
    // contracts are inspectable. Explicit injected fixture, never a live feed.
    view.stale = false; view.status_age_ms = 0; view.stale_after_ms = 60_000;
    view.status.state = "SYNCING";
    view.status.last_decision = null;
    view.status.feeds.binance.venue = "us";
    view.status.alpha.venue = "us";
    view.status.alpha.source = "binance_us_depth:BTCUSDT";
    view.active_market.reference_price_at_start = 100_000.125;
    view.active_market.oracle_twap_lookback_seconds = 60;
    const observation = {value: 100_010.25, fresh: true, connected: true, timestamp_ms: view.generated_at_ms,
      age_ms: 0, max_age_ms: 60_000, received_at_ms: view.generated_at_ms,
      source: '<img src=x onerror="window.xss=true">', symbol: "btc/usd", quote_currency: "USD",
      kind: "published_twap", lookback_seconds: 60};
    const quote = {bid: 0.38, ask: 0.4, bid_size: 80, ask_size: 120, confirmed: true, ...observation};
    view.status.market_data = {window_id: view.active_market.window_id, market_id: view.active_market.market_id,
      spot: {value: null}, oracle: observation,
      up: {...quote, token_id: view.active_market.yes_token_id},
      down: {...quote, bid: 0.58, ask: 0.6, token_id: view.active_market.no_token_id}};
    await page.route("**/api/v1/dashboard*", route => route.fulfill({status: 200, contentType: "application/json", body: JSON.stringify(view)}));
    await page.waitForFunction(() => document.querySelector("#price-opening .price-value").textContent === "$100,000.1250");
    assert.equal(await page.locator("#price-oracle .price-value").innerText(), "$100,010.2500");
    assert.match(await page.locator("#price-oracle").innerText(), /60s published TWAP/);
    assert.equal(await page.locator("#price-spot .price-value").innerText(), "—");
    assert.match(await page.locator("#price-spot h3").innerText(), /Binance\.US/);
    assert.match(await page.locator("#feed-cards h3").first().innerText(), /Binance\.US/);
    assert.match(await page.locator("#alpha-fields").innerText(), /binance_us_depth:BTCUSDT/);
    assert.equal(await page.locator("#price-up .price-value").innerText(), "$0.400000");
    assert.equal(await page.locator("#price-down .price-value").innerText(), "$0.600000");
    assert.equal(await page.locator(".price-card img").count(), 0);
    assert.equal(await page.evaluate(() => window.xss), undefined);
    // Age advances between successful fetches, not just when new data arrives.
    observation.max_age_ms = 200;
    await page.waitForFunction(() => document.querySelector("#price-oracle .badge").textContent.includes("Stale"));
    // A rollover must not combine a new opening with old token quotes.
    view.active_market.window_id = "next-window";
    view.active_market.reference_price_at_start = 100_020;
    await page.waitForFunction(() => document.querySelector("#price-opening .price-value").textContent === "$100,020.0000");
    assert.equal(await page.locator("#price-up .price-value").innerText(), "—");
    assert.equal(await page.locator("#price-oracle .price-value").innerText(), "—");
    delete view.status.market_data;
    view.active_market.reference_price_at_start = null;
    await page.waitForFunction(() => document.querySelector("#price-opening .price-value").textContent === "—");
    assert.equal(await page.locator(".price-card .badge.good").count(), 0);
    await page.unroute("**/api/v1/dashboard*");
    view.active_market.title = '<img src=x onerror="window.xss=true">';
    view.account = null; view.positions = null;
    view.status.last_decision = null;
    await page.route("**/api/v1/dashboard*", route => route.fulfill({status: 200, contentType: "application/json", body: JSON.stringify(view)}));
    await page.waitForFunction(() => document.getElementById("market-name").textContent.startsWith("<img"));
    assert.equal(await page.locator("#market-name img").count(), 0);
    assert.equal(await page.evaluate(() => window.xss), undefined);
    assert.equal(await page.locator(".kpi .value").first().innerText(), "—");
    assert(! (await page.locator("#inputs-fields").innerText()).includes("$0"));
    await page.unroute("**/api/v1/dashboard*");
    await page.setViewportSize({width: 1440, height: 1100});
    await page.locator("#latest").focus();
    await page.keyboard.press("Enter");
    assert.deepEqual(errors, []);
    console.log("Browser acceptance passed: desktop/mobile, prices without decisions, per-source missing/stale data, rollover fencing, legacy status, no overflow, failure retention/recovery, XSS-safe text, keyboard control.");
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
