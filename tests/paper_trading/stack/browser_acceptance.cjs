// Optional manual acceptance with external Playwright + installed Chrome.
// Run against an already-started two-minute mock stack; no outbound requests.
const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs/promises");
const path = require("node:path");

(async () => {
  const url = process.argv[2];
  assert.equal(new URL(url).hostname, "127.0.0.1");
  const out = path.resolve(process.argv[3]);
  await fs.mkdir(out, {recursive: true});
  const browser = await chromium.launch({headless: true, channel: process.env.BROWSER_CHANNEL || "chrome"});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1100}});
    const errors = [];
    page.on("pageerror", e => errors.push(e.message));
    await page.goto(url);
    await page.waitForFunction(() => document.querySelectorAll(".kpi").length === 8);
    assert(await page.locator(".paper-banner").isVisible());
    await page.screenshot({path: path.join(out, "stack-running.png")});
    await page.setViewportSize({width: 390, height: 844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({path: path.join(out, "stack-mobile.png")});
    await page.setViewportSize({width: 1440, height: 1100});
    await page.waitForFunction(() => document.querySelectorAll("#runs-table tbody tr").length >= 2, null, {timeout: 150000});
    await page.screenshot({path: path.join(out, "stack-rollover.png")});
    await page.waitForFunction(() => document.getElementById("state-badge").textContent.includes("STOPPED"), null, {timeout: 150000});
    await page.screenshot({path: path.join(out, "stack-stopped.png")});
    await page.waitForFunction(() => document.getElementById("error-banner").textContent.includes("Refresh failed"), null, {timeout: 20000});
    assert((await page.locator("#state-badge").innerText()).includes("STOPPED"));
    assert(await page.locator(".paper-banner").isVisible());
    assert.equal(await page.locator(".kpi").count(), 8);
    await page.screenshot({path: path.join(out, "stack-disconnected.png")});
    assert.deepEqual(errors, []);
    console.log("PASS: running, mobile, rollover, STOPPED, retained view after disconnect; no page errors.");
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
