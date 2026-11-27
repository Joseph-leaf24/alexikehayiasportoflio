import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

/**
 * Smoke and accessibility tests of the built site in a real browser. Every test also fails on
 * uncaught exceptions or console errors.
 */

const SECTIONS = ["top", "about", "domains", "toolbox", "projects", "contact", "watch"];
/** WCAG 2.0–2.2 A/AA rules plus axe's best practices, such as heading order. */
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

/** Open a route and wait for the intro loader, which waits for the pixel scene, to finish. */
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
  const summary = violations.map(
    (v) => `${v.id} (${v.impact}) ×${v.nodes.length}: ${v.help} — ${v.nodes[0]?.target.join(" ")}`
  );
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
 * No personal location, such as a city, country code or map coordinates, in the visible text. The
 * IKEA Breda dashboard keeps its client's name: that's the store, not where Alexi lives.
 */
async function expectNoLocation(page: Page) {
  const text = await page.evaluate(() => document.body.innerText.replace(/IKEA Breda/g, "IKEA"));
  expect(text).not.toMatch(/breda|\bNL\b|°\s*[NSEW]/i);
}

/** Nothing may be wider than the viewport, so phones never scroll sideways. */
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
  // The position is saved shortly after scrolling stops; wait for it so the test proves it is ignored
  await expect.poll(() => page.evaluate(() => Number(sessionStorage.getItem("ak:homeY")))).toBeGreaterThan(1000);
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
  // ScrambleText keeps the real text in an .sr-only span; the visible glyphs may be mid-scramble
  const title = (await link.locator(".sr-only").first().textContent())?.trim() ?? "";
  // Click the middle of the card, not the title: the stretched link must catch it
  await link.locator("xpath=ancestor::article").click();
  await expect(page).toHaveURL(/#\/project\/[a-zA-Z]+\/\d+$/);
  await expect(page.getByRole("heading", { level: 1 })).toContainText(title);
  await page.getByRole("link", { name: /back to portfolio/i }).click();
  await expect(page).toHaveURL(/:\d+\/(#\/?)?$/);
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
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Mission failed");
  await expect(page.getByRole("link", { name: /return to home/i })).toBeVisible();
  await page.waitForTimeout(1500);
  await expectAccessible(page);
});

/** A fingerprint of what the battle's canvas shows right now; it changes whenever the picture does. */
const frame = (page: Page) =>
  page.evaluate(() => {
    const canvas = document.querySelector<HTMLCanvasElement>("canvas.pixel-canvas");
    if (!canvas) return "no canvas";
    const url = canvas.toDataURL("image/png");
    let h = 0;
    for (let i = 0; i < url.length; i += 3) h = (h * 31 + url.charCodeAt(i)) | 0;
    return `${url.length}:${h}`;
  });

test("the battle plays behind the page by itself, and a click on the scenery doesn't break it", async ({
  page,
  isMobile,
}) => {
  await open(page);
  const first = await frame(page);
  expect(first).not.toBe("no canvas");
  await expect.poll(() => frame(page), { timeout: 10_000 }).not.toBe(first);
  if (!isMobile) {
    // A click in the open strip calls in a blast; the fight goes on after it
    const { width, height } = page.viewportSize()!;
    await page.mouse.click(width * 0.6, height * 0.62);
    const after = await frame(page);
    await expect.poll(() => frame(page), { timeout: 10_000 }).not.toBe(after);
  }
});

test("the spectator space at the bottom brings up the scene controls (pause, speed, zoom), and they go with it", async ({
  page,
}) => {
  await open(page, "?level=temple");
  const controls = page.getByRole("region", { name: "Background animation" });
  await expect(controls).toHaveCount(0);
  await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
  await expect(page.locator("section#watch")).toBeInViewport();
  await expect(controls).toBeVisible();
  // Pause holds the picture still; Resume sets it going again
  await controls.getByRole("button", { name: "Pause the background" }).click();
  await page.waitForTimeout(400);
  const still = await frame(page);
  await page.waitForTimeout(1000);
  expect(await frame(page), "the picture while paused").toBe(still);
  await controls.getByRole("button", { name: "Resume the background" }).click();
  await expect.poll(() => frame(page), { timeout: 10_000 }).not.toBe(still);
  // One speed at a time
  const speeds = controls.getByRole("group", { name: "Speed" });
  await speeds.getByRole("button", { name: "4×", exact: true }).click();
  await expect(speeds.getByRole("button", { name: "4×", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect(speeds.getByRole("button", { name: "1×", exact: true })).toHaveAttribute("aria-pressed", "false");
  // Zooming out shows more of the battlefield, at a smaller percentage
  const zoom = controls.locator(".scene-controls__zoom");
  const before = (await zoom.textContent())!;
  await controls.getByRole("button", { name: /^Zoom out/ }).click();
  await expect(zoom).not.toHaveText(before);
  await expectNoHorizontalScroll(page);
  await expectAccessible(page);
  // Leaving the spectator space puts them away
  await page.evaluate(() => window.scrollTo(0, 0));
  await expect(controls).toHaveCount(0);
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
