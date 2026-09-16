/* Run against a local console:
   NODE_PATH=/path/to/playwright-core/node_modules node tests/console_frontend.cjs
   Requires Playwright's Chromium installation. No database writes are made. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright-core");

const base = process.env.MPM_CONSOLE_URL || "http://127.0.0.1:8765";
const output = path.resolve("outputs/frontend-check");
fs.mkdirSync(output, { recursive: true });

(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    const ready = () => page.waitForFunction(() => !document.querySelector("main").hasAttribute("aria-busy"));
    const noOverflow = async () => assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "Page has horizontal overflow");
    const shot = name => page.screenshot({ path: path.join(output, name + ".png"), fullPage: true });

    await page.goto(base);
    await ready();
    assert(await page.getByRole("heading", { level: 1 }).textContent());
    await noOverflow();
    await shot("home-desktop");
    await page.locator('nav a[href="#memories"]').click();
    await ready();
    await page.locator("#memory-results:not([aria-busy])").waitFor();
    if (await page.locator(".memory-row").count()) {
      const row = page.locator(".memory-row").first();
      await row.click();
      await page.locator("#detail-body:not([aria-busy])").waitFor();
      assert(await page.locator("dialog").isVisible());
      await shot("memory-detail");
      await page.keyboard.press("Escape");
      assert(!await page.locator("dialog").isVisible());
    }
    await page.getByLabel("Search memories").fill("no-match-7c02");
    await page.getByRole("heading", { name: "No matching memories" }).waitFor();
    await page.getByRole("button", { name: "Clear filters", exact: true }).first().click();
    await page.locator("#memory-results:not([aria-busy])").waitFor();
    await shot("memories-desktop");
    await page.locator('nav a[href="#decisions"]').click();
    await ready();
    await page.locator("#decision-results:not([aria-busy])").waitFor();
    await noOverflow();
    await shot("decisions-desktop");
    if (await page.locator(".decision-row").count()) {
      await page.getByLabel("Operation").selectOption("NOOP");
      await page.locator("#decision-results:not([aria-busy])").waitFor();
      assert(!(await page.locator(".decision-row .tag").allTextContents()).some(t => t === "Saved"), "Operation filter leaked other operations");
      await page.getByRole("button", { name: "Clear filters", exact: true }).first().click();
      await page.locator("#decision-results:not([aria-busy])").waitFor();
    }
    await page.locator('nav a[href="#audit"]').click();
    await ready();
    await shot("history-desktop");
    if (await page.getByRole("button", { name: "Next", exact: true }).isEnabled()) {
      const firstPage = await page.locator("main .timeline").textContent();
      await page.getByRole("button", { name: "Next", exact: true }).click();
      await ready();
      assert.notEqual(await page.locator("main .timeline").textContent(), firstPage);
      await page.getByRole("button", { name: "Previous", exact: true }).click();
      await ready();
      assert.equal(await page.locator("main .timeline").textContent(), firstPage);
    }
    await page.locator('nav a[href="#checkpoints"]').click();
    await ready();
    await shot("learning-desktop");
    await page.getByLabel("Appearance").selectOption("dark");
    await shot("learning-dark");
    await page.locator('nav a[href="#overview"]').click();
    await ready();
    await shot("home-dark");
    await page.getByLabel("Appearance").selectOption("light");
    await page.setViewportSize({ width: 390, height: 844 });
    for (const route of ["overview", "memories", "decisions", "audit", "checkpoints"]) {
      await page.goto(base + "/#" + route);
      await ready();
      await noOverflow();
      await shot(route + "-mobile");
    }
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto(base + "/#memories");
    await ready();
    await page.locator("#memory-results:not([aria-busy])").waitFor();
    if (await page.locator(".memory-row").count()) {
      await page.locator(".memory-row").first().click();
      await page.locator("#detail-body:not([aria-busy])").waitFor();
      assert(await page.locator("dialog").evaluate(el => el.scrollWidth <= el.clientWidth));
      await page.keyboard.press("Escape");
    }
    // Failure and retry must be usable without a full page reload.
    await page.route("**/api/status", route => route.fulfill({ status: 503, body: "{}" }));
    await page.goto(base);
    await page.getByRole("heading", { name: "We couldn’t load this information" }).waitFor();
    await page.unroute("**/api/status");
    await page.getByRole("button", { name: "Try again" }).click();
    await ready();
    assert(await page.locator(".stats").count());

    // Slow responses from an older search must not replace newer results.
    await page.route("**/api/memories?**", async route => {
      const q = new URL(route.request().url()).searchParams.get("q");
      if (q === "slow") await new Promise(resolve => setTimeout(resolve, 650));
      if (q === "slow" || q === "new") {
        await route.fulfill({ json: { items: [{
          memory_id: q, preview: { content: q + " <script>unsafe()</script>" },
          status: "active", n_retrievals: 0, total_reward: 0,
        }], total: 1, offset: 0, limit: 20 } }).catch(() => {});
      } else await route.continue();
    });
    await page.locator('nav a[href="#memories"]').click();
    await ready();
    await page.getByLabel("Search memories").fill("slow");
    await page.waitForTimeout(250);
    await page.getByLabel("Search memories").fill("new");
    await page.waitForTimeout(850);
    assert.match(await page.locator(".memory-list").textContent(), /new <script>/);
    assert.equal(await page.locator(".memory-list script").count(), 0);
    assert.equal(errors.length, 0, errors.join("\n"));
    console.log("PASS: desktop/mobile pages, dark mode, details, decisions, search, pagination, retry, stale requests, safe text, and no browser exceptions.");
    console.log("Screenshots: " + output);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
