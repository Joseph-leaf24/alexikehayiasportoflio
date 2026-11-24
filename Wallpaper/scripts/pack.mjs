/**
 * After `vite build`: makes dist/index.html the whole battle on its own.
 *
 * 1. Inlines the script and the styles into it, so it's one file that runs wherever it's put
 *    (opened straight off the disk too, where the browser won't load a module script from a
 *    separate file).
 * 2. Opens it off the disk in Chrome, checks it runs without errors, and takes a picture of it
 *    (dist/preview.jpg).
 */
import { readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const dist = join(root, "dist");

/* 1. One self-contained page */
let html = readFileSync(join(dist, "index.html"), "utf8");
html = html.replace(/<script type="module"[^>]*\ssrc="\.\/(assets\/[^"]+\.js)"[^>]*><\/script>/g, (_, file) => {
  const js = readFileSync(join(dist, file), "utf8").replace(/<\/script/gi, "<\\/script");
  return `<script type="module">${js}</script>`;
});
html = html.replace(/<link rel="stylesheet"[^>]*\shref="\.\/(assets\/[^"]+\.css)"[^>]*>/g, (_, file) => `<style>${readFileSync(join(dist, file), "utf8")}</style>`);
if (/\.\/assets\//.test(html)) throw new Error("pack: index.html still loads something from assets/");
writeFileSync(join(dist, "index.html"), html);
rmSync(join(dist, "assets"), { recursive: true, force: true });

/* 2. Run it off the disk and take its picture */
try {
  const { chromium } = await import("@playwright/test");
  const browser = await chromium.launch({ channel: "chrome" });
  const page = await browser.newPage({ viewport: { width: 960, height: 540 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  await page.goto(pathToFileURL(join(dist, "index.html")).href);
  await page.waitForTimeout(7000);
  await page.screenshot({ path: join(dist, "preview.jpg"), type: "jpeg", quality: 82 });
  await browser.close();
  if (errors.length) throw new Error(`the battle had errors running off the disk:\n${errors.join("\n")}`);
  console.log("pack: dist/index.html runs on its own without errors (preview: dist/preview.jpg)");
} catch (e) {
  if (/had errors/.test(e.message)) throw e;
  console.warn(`pack: not checked in a browser (${e.message.split("\n")[0]})`);
}
