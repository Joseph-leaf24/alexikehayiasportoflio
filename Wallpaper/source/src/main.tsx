import { createRoot } from "react-dom/client";
import PixelScene from "@/components/pixel/PixelScene";
import { canvasOptions } from "@/components/pixel/art";
import { sceneState } from "@/lib/store";
import "./wallpaper.css";

/**
 * The pixel battle as the desktop's wallpaper. The same scene as the site's, on its own and full
 * screen: at twice the speed, zoomed out by a quarter (more of the battlefield on the desktop),
 * the whole battle as the site's spectator space shows it, capped at 30 frames a second and on a
 * lighter budget so it goes easy on the machine.
 *
 * PixelBattle.exe (host/) shows this page behind the desktop icons. It passes its settings in the
 * address (?speed=2&zoom=75&fps=30) and changes them, and passes on clicks on the desktop, through
 * `window.battle`. Opened in a browser on its own, clicks on the page itself call in blasts.
 */
const params = new URLSearchParams(window.location.search);
const num = (key: string, fallback: number) => {
  const v = Number(params.get(key));
  return Number.isFinite(v) && v > 0 ? v : fallback;
};

const settings = {
  speed: (v: number) => (sceneState.speed = Math.min(8, Math.max(0.5, v))),
  zoom: (pct: number) => (sceneState.scaleK = Math.min(1, Math.max(0.5, pct / 100))),
  fps: (v: number) => (sceneState.maxFps = Math.min(120, Math.max(10, v))),
};
settings.speed(num("speed", 2));
settings.zoom(num("zoom", 75));
settings.fps(num("fps", 30));
// (drawn in software: cheaper than the GPU for a canvas this small, drawn in thousands of little pieces; ?gpu=1 to compare)
if (!params.has("gpu")) canvasOptions.willReadFrequently = true;
// (always on show: no slowing down when nobody's touched anything for a while)
sceneState.watching = true;
// (the wildest of the site's sections: the spectator space at the bottom of the page)
sceneState.stops = [{ y: -1e9, layout: "watch" }];

/* ---- Blasts: a click calls one in there, holding the button down brings a barrage ---- */
let hold = 0;
const aim = (nx: number, ny: number) => {
  sceneState.pointer.x = nx * 2 - 1;
  sceneState.pointer.y = -(ny * 2 - 1);
  sceneState.pointer.lastMove = performance.now();
};
const blast = (nx: number, ny: number) => {
  aim(nx, ny);
  sceneState.burst++;
  window.clearTimeout(hold);
  hold = window.setTimeout(() => (sceneState.press = true), 450);
};
const press = (on: boolean) => {
  if (on) return;
  window.clearTimeout(hold);
  sceneState.press = false;
};

declare global {
  interface Window {
    /** What PixelBattle.exe talks to: settings changed from its tray menu, clicks on the desktop (0–1 across the screen). */
    battle: { set: (name: keyof typeof settings, value: number) => void; blast: (nx: number, ny: number) => void; press: (on: boolean) => void };
  }
}
window.battle = { set: (name, value) => settings[name]?.(Number(value)), blast, press };

// (in a browser on its own: the page's own clicks)
window.addEventListener("pointermove", (e) => aim(e.clientX / window.innerWidth, e.clientY / window.innerHeight), { passive: true });
window.addEventListener("pointerdown", (e) => blast(e.clientX / window.innerWidth, e.clientY / window.innerHeight));
window.addEventListener("pointerup", () => press(false));
window.addEventListener("blur", () => press(false));

// (for PixelBattle.exe's self-test: whether the battle's running, how far on it's got, what it can see)
const webview = (window as unknown as { chrome?: { webview?: { postMessage: (m: string) => void } } }).chrome?.webview;
if (webview && params.has("report")) {
  // (and how many frames a second it's drawing)
  let frames = sceneState.frames;
  window.setInterval(() => {
    webview.postMessage(JSON.stringify({ ready: sceneState.ready, x: Math.round(sceneState.cam.x), fps: (sceneState.frames - frames) / 2, w: window.innerWidth, h: window.innerHeight, dpr: window.devicePixelRatio, hidden: document.hidden }));
    frames = sceneState.frames;
  }, 2000);
}

createRoot(document.getElementById("root")!).render(<PixelScene />);
