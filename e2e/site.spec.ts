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
  // Count every Web Audio context the page creates: the site must stay silent
  await page.addInitScript(() => {
    const w = window as unknown as Record<string, unknown> & { __audioContexts: number };
    w.__audioContexts = 0;
    for (const name of ["AudioContext", "webkitAudioContext"]) {
      const Original = w[name] as (new (...args: unknown[]) => object) | undefined;
      if (!Original) continue;
      w[name] = class extends Original {
        constructor(...args: unknown[]) {
          super(...args);
          w.__audioContexts++;
        }
      };
    }
  });
  page.on("pageerror", (e) => problems.push(`uncaught: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() === "error") problems.push(`console: ${m.text()}`);
    // GSAP's warning for an animation scoped to an element React has already removed
    if (m.type() === "warning" && /Invalid scope/.test(m.text())) problems.push(`console warning: ${m.text()}`);
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

/** No sound: no media elements and no Web Audio contexts. */
async function expectSilent(page: Page) {
  const audio = await page.evaluate(() => ({
    media: document.querySelectorAll("audio, video").length,
    contexts: (window as unknown as { __audioContexts: number }).__audioContexts,
  }));
  expect(audio, "audio on the page").toEqual({ media: 0, contexts: 0 });
}

/**
 * No personal location (city, country code or map coordinates) in the visible text. The
 * IKEA Breda dashboard keeps its client's name: that's the store, not where Alexi lives.
 */
async function expectNoLocation(page: Page) {
  const text = await page.evaluate(() => document.body.innerText.replace(/IKEA Breda/g, "IKEA"));
  expect(text).not.toMatch(/breda|\bNL\b|°\s*[NSEW]/i);
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
  await expectSilent(page);
  await expectNoLocation(page);
  await expectAccessible(page);
});

test("the phone menu's links close the menu and scroll to their section", async ({ page, isMobile }) => {
  test.skip(!isMobile, "the full-screen menu is for small screens");
  await open(page);
  await page.getByRole("button", { name: "Open menu" }).click();
  await expect(page.locator("#mobile-menu")).toHaveClass(/is-open/);
  await page.locator("#mobile-menu").getByRole("link", { name: /work/i }).click();
  await expect(page.locator("#mobile-menu")).not.toHaveClass(/is-open/);
  await expect(page.locator("#projects")).toBeInViewport({ ratio: 0.1 });
});

test("a reload always opens on the intro, also after coming back from a project", async ({ page }) => {
  const atIntro = async () => {
    await expect(page.locator(".loader")).toHaveCount(0, { timeout: 45_000 });
    await expect.poll(() => page.evaluate(() => window.scrollY)).toBe(0);
    await expect(page.locator(".hud-section__idx")).toHaveText("00");
  };
  await open(page);
  // Scrolled down, then reloaded
  await page.locator("#projects").scrollIntoViewIfNeeded();
  await expect.poll(() => page.evaluate(() => window.scrollY)).toBeGreaterThan(1000);
  await page.waitForTimeout(400); // the position is saved shortly after scrolling stops
  await page.reload();
  await atIntro();
  // Back from a project lands on the work section; a reload there still opens on the intro
  await page.locator("#projects").scrollIntoViewIfNeeded();
  await page.locator(".project-card__link").first().click();
  await expect(page).toHaveURL(/#\/project\//);
  await page.getByRole("link", { name: /back to portfolio/i }).click();
  await expect(page.locator("#projects")).toBeInViewport({ ratio: 0.1 });
  await page.reload();
  await atIntro();
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

test("the scene follows the time of day (?time= pins it) and the link-preview image is served", async ({ page }) => {
  await open(page, "?time=13:00#/");
  await expect(page.locator("html")).toHaveAttribute("data-sky", "day");
  await open(page, "?time=23:00#/");
  await expect(page.locator("html")).toHaveAttribute("data-sky", "night");
  const og = await page.request.get("og-image.jpg");
  expect(og.status()).toBe(200);
  expect(og.headers()["content-type"]).toBe("image/jpeg");
});

test("keyboard: ←/→ step through projects and ? opens the shortcuts help", async ({ page, isMobile }) => {
  test.skip(isMobile, "keyboard shortcuts are for desktop");
  await open(page, "#/project/nlp/1");
  await page.keyboard.press("ArrowRight");
  await expect(page).toHaveURL(/#\/project\/research\/0$/);
  await page.keyboard.press("ArrowLeft");
  await expect(page).toHaveURL(/#\/project\/nlp\/1$/);
  await page.keyboard.press("?");
  const help = page.getByRole("dialog", { name: "Keyboard shortcuts" });
  await expect(help).toBeVisible();
  await expect(help.getByRole("button", { name: "Close keyboard shortcuts" })).toBeFocused();
  await expectAccessible(page);
  await page.keyboard.press("Escape");
  await expect(help).toHaveCount(0);
});
