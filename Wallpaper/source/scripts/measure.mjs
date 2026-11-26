/**
 * How hard the built wallpaper works: opens dist/index.html off the disk in Chrome at a desktop's
 * size for a while and reports the frames it drew a second and the time each one took on the
 * main thread (the game and the drawing). `npm run measure -- 2560 1440 30` for another screen
 * size and run length (seconds).
 */
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { chromium } from "@playwright/test";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const [w = 1920, h = 1080, secs = 20] = process.argv.slice(2).map(Number);
const browser = await chromium.launch({ channel: "chrome" });
const page = await browser.newPage({ viewport: { width: w, height: h } });
// (time every animation-frame callback: the scene does all its work in them)
await page.addInitScript(() => {
  const stats = { frames: 0, busy: 0, worst: 0 };
  window.__stats = stats;
  const raf = window.requestAnimationFrame.bind(window);
  window.requestAnimationFrame = (cb) =>
    raf((t) => {
      const t0 = performance.now();
      cb(t);
      const d = performance.now() - t0;
      if (d > 0.3) {
        stats.frames++;
        stats.busy += d;
        stats.worst = Math.max(stats.worst, d);
      }
    });
});
await page.goto(pathToFileURL(join(root, "dist", "index.html")).href);
await page.waitForTimeout(6000);
await page.evaluate(() => Object.assign(window.__stats, { frames: 0, busy: 0, worst: 0 }));
await page.waitForTimeout(secs * 1000);
const s = await page.evaluate(() => window.__stats);
await browser.close();
console.log(`${w}x${h}, ${secs}s: ${(s.frames / secs).toFixed(1)} frames drawn a second, ${(s.busy / Math.max(1, s.frames)).toFixed(2)} ms each on average (worst ${s.worst.toFixed(1)} ms), ${((s.busy / (secs * 1000)) * 100).toFixed(1)}% of one CPU core`);
