/**
 * After `vite build`: turns dist/ into a wallpaper.
 *
 * 1. Inlines the script and the styles into index.html. Wallpaper apps open the page straight off
 *    the disk (file://), where the browser won't load a module script from a separate file.
 * 2. Adds what Lively Wallpaper (LivelyInfo.json, LivelyProperties.json) and Wallpaper Engine
 *    (project.json) need, with the settings their panels offer.
 * 3. Opens the finished page off the disk in Chrome, the way a wallpaper app will, checks it runs
 *    without errors, and takes its preview picture.
 * 4. Zips it all up for Lively (drag the zip onto Lively's window to add it).
 */
import { existsSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { execFileSync } from "node:child_process";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const dist = join(root, "dist");
const TITLE = "Pixel Battle";
const DESC =
  "An endless eight-player pixel-art battle through random countries: cities with traffic, day and night and weather, metros, rail yards, great mountains, caverns, gunships, nukes. Runs at 2x speed, zoomed out to 75%. Click the desktop to call in a blast; hold the button for a barrage.";

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

/* 2. For the wallpaper apps */
const json = (name, data) => writeFileSync(join(dist, name), JSON.stringify(data, null, 2) + "\n");
json("LivelyInfo.json", {
  AppVersion: "1.0.0.0",
  Title: TITLE,
  Thumbnail: "thumbnail.jpg",
  Preview: "preview.jpg",
  Desc: DESC,
  Author: "Alexi",
  License: "",
  Contact: "",
  Type: 1,
  FileName: "index.html",
  Arguments: "",
  IsAbsolutePath: false,
});
json("LivelyProperties.json", {
  speed: { type: "dropdown", value: 1, text: "Speed", items: ["1x", "2x", "4x"] },
  zoom: { type: "slider", value: 75, min: 50, max: 100, step: 5, text: "Zoom (%)" },
  fps: { type: "dropdown", value: 0, text: "Frame rate", items: ["30 fps (lightest)", "60 fps"] },
  blasts: { type: "checkbox", value: true, text: "Clicking the desktop calls in a blast" },
});
json("project.json", {
  title: TITLE,
  description: DESC,
  type: "web",
  file: "index.html",
  preview: "preview.jpg",
  general: {
    properties: {
      speed: { order: 0, text: "Speed", type: "combo", value: "2", options: [{ label: "1x", value: "1" }, { label: "2x", value: "2" }, { label: "4x", value: "4" }] },
      zoom: { order: 1, text: "Zoom (%)", type: "slider", value: 75, min: 50, max: 100, fraction: false },
      fps: { order: 2, text: "Frame rate", type: "combo", value: "30", options: [{ label: "30 fps (lightest)", value: "30" }, { label: "60 fps", value: "60" }] },
      blasts: { order: 3, text: "Clicking the desktop calls in a blast", type: "bool", value: true },
    },
  },
});

/* 3. Run it off the disk, the way the wallpaper apps will, and take its picture */
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
  await page.screenshot({ path: join(dist, "thumbnail.jpg"), type: "jpeg", quality: 70 });
  await browser.close();
  if (errors.length) throw new Error(`the wallpaper had errors running off the disk:\n${errors.join("\n")}`);
  console.log("pack: runs off the disk without errors; preview taken");
} catch (e) {
  if (/had errors/.test(e.message)) throw e;
  console.warn(`pack: no preview picture (${e.message.split("\n")[0]})`);
}

/* 4. The zip for Lively */
const zip = join(root, "PixelBattle-Lively.zip");
if (process.platform === "win32") {
  if (existsSync(zip)) rmSync(zip);
  execFileSync("powershell", ["-NoProfile", "-Command", `Compress-Archive -Path '${dist}\\*' -DestinationPath '${zip}' -Force`]);
  console.log(`pack: ${zip}`);
}
console.log(`pack: dist/ (${readdirSync(dist).join(", ")})`);
