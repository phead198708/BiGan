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
    await page.unroute("**/api/v1/dashboard*");
    await page.waitForFunction(() => !document.getElementById("error-banner").textContent.includes("Refresh failed"));

    const response = await page.request.get(url + "/api/v1/dashboard");
    const view = await response.json();
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
    console.log("Browser acceptance passed: desktop/mobile, no horizontal overflow, failure retention/recovery, null values, XSS-safe text, keyboard control.");
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
