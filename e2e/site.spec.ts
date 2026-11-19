import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

/**
 * Smoke + accessibility checks of the built site in a real browser. Every test also
 * fails on uncaught exceptions or console errors.
 */

const SECTIONS = ["top", "about", "domains", "toolbox", "projects", "contact"];
/** WCAG 2.0–2.2 A/AA rules plus axe's best practices (e.g. heading order). */
const AXE_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"];

let problems: string[] = [];

test.beforeEach(async ({ page }) => {
  problems = [];
  page.on("pageerror", (e) => problems.push(`uncaught: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() === "error") problems.push(`console: ${m.text()}`);
  });
});

test.afterEach(() => {
  expect(problems, "page errors / console errors").toEqual([]);
});

/** Open a route and wait for the intro loader to finish (it waits for the 3D scene). */
async function open(page: Page, route = "") {
  await page.goto(route);
  await expect(page.locator(".loader")).toHaveCount(0, { timeout: 45_000 });
}

/** Scroll the whole page so every scroll-triggered reveal has played, then settle. */
async function revealAll(page: Page) {
  const height = await page.evaluate(() => document.documentElement.scrollHeight);
  for (let y = 0; y <= height; y += 600) {
    await page.evaluate((top) => window.scrollTo(0, top), y);
    await page.waitForTimeout(80);
  }
  await page.waitForTimeout(1500);
}

/** Run axe and fail with a readable list of violations. */
async function expectAccessible(page: Page) {
  // preload: false — axe would otherwise fetch cross-origin stylesheets, which the
  // site's CSP (connect-src 'self') rightly blocks and logs as a console error
  const { violations } = await new AxeBuilder({ page }).withTags(AXE_TAGS).options({ preload: false }).analyze();
  const summary = violations.map((v) => `${v.id} (${v.impact}) ×${v.nodes.length}: ${v.help} — ${v.nodes[0]?.target.join(" ")}`);
  expect(summary, "axe accessibility violations").toEqual([]);
}

/** Nothing may be wider than the viewport (no sideways scrolling on phones). */
async function expectNoHorizontalScroll(page: Page) {
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow, "horizontal overflow in px").toBeLessThanOrEqual(1);
}

test("home page: every section renders, fits the screen and passes axe", async ({ page }) => {
  await open(page);
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Alexi Kehayias");
  for (const id of SECTIONS) await expect(page.locator(`section#${id}`)).toHaveCount(1);
  await revealAll(page);
  await expect(page.getByRole("link", { name: /download cv/i })).toBeVisible();
  await expectNoHorizontalScroll(page);
  await expectAccessible(page);
});

test("a project card opens its page, and Back returns to the work section", async ({ page }) => {
  await open(page);
  await page.locator("#projects").scrollIntoViewIfNeeded();
  const link = page.locator(".project-card__link").first();
  // ScrambleText keeps the real text in an .sr-only span (the visible glyphs may be mid-scramble)
  const title = (await link.locator(".sr-only").first().textContent())?.trim() ?? "";
  // Click the middle of the card (not the title) — the stretched link must catch it
  await link.locator("xpath=ancestor::article").click();
  await expect(page).toHaveURL(/#\/project\/[a-zA-Z]+\/\d+$/);
  await expect(page.getByRole("heading", { level: 1 })).toContainText(title);
  await page.getByRole("link", { name: /back to portfolio/i }).click();
  await expect(page).toHaveURL(/alexikehayiasportoflio\/(#\/?)?$/);
  await expect(page.locator("#projects")).toBeInViewport({ ratio: 0.1 });
});

test("project page: network cover, fits the screen and passes axe", async ({ page }) => {
  await open(page, "#/project/nlp/1");
  await expect(page.getByRole("img", { name: /neural network diagram/i })).toBeVisible();
  await revealAll(page);
  await expectNoHorizontalScroll(page);
  await expectAccessible(page);
});

test("the image viewer opens from the gallery and closes with Escape", async ({ page }) => {
  await open(page, "#/project/computerVision/0");
  await page.getByRole("button", { name: /open image 1 of/i }).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole("button", { name: "Close image viewer" })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
});

test("unknown routes show the 404 page with a way home, and it passes axe", async ({ page }) => {
  await open(page, "#/this-page-does-not-exist");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Lost in the ice");
  await expect(page.getByRole("link", { name: /return to home/i })).toBeVisible();
  await page.waitForTimeout(1500);
  await expectAccessible(page);
});
