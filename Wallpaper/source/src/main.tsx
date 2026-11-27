import { createRoot } from "react-dom/client";
import PixelScene from "@/components/pixel/PixelScene";
import { canvasOptions } from "@/components/pixel/art";
import { sceneState } from "@/lib/store";
import "./wallpaper.css";

/**
 * The Site 2 pixel battle on its own, full screen, as PixelBattle.exe shows it behind the desktop icons.
 * Settings arrive in the query string (?speed=2&zoom=75&fps=30) and later through `window.battle`.
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

// A software canvas is cheaper than the GPU for a small canvas drawn in thousands of tiny pieces.
if (!params.has("gpu")) canvasOptions.willReadFrequently = true;
// Keep the full frame rate even when nobody has touched anything for a while.
sceneState.watching = true;
sceneState.stops = [{ y: -1e9, layout: "watch" }];

// A click calls in a blast; holding the button for 450 ms turns it into a barrage until release.
const BARRAGE_DELAY_MS = 450;
let barrageTimer = 0;
const aim = (nx: number, ny: number) => {
  sceneState.pointer.x = nx * 2 - 1;
  sceneState.pointer.y = -(ny * 2 - 1);
  sceneState.pointer.lastMove = performance.now();
};
const blast = (nx: number, ny: number) => {
  aim(nx, ny);
  sceneState.burst++;
  window.clearTimeout(barrageTimer);
  barrageTimer = window.setTimeout(() => (sceneState.press = true), BARRAGE_DELAY_MS);
};
const release = () => {
  window.clearTimeout(barrageTimer);
  sceneState.press = false;
};

declare global {
  interface Window {
    /** The interface PixelBattle.exe calls: settings from its tray menu, and desktop clicks (0–1 across the screen). */
    battle: {
      set: (name: keyof typeof settings, value: number) => void;
      blast: (nx: number, ny: number) => void;
      release: () => void;
    };
  }
}
window.battle = { set: (name, value) => settings[name]?.(Number(value)), blast, release };

// Opened in a browser instead, the page handles its own clicks.
window.addEventListener("pointermove", (e) => aim(e.clientX / window.innerWidth, e.clientY / window.innerHeight), {
  passive: true,
});
window.addEventListener("pointerdown", (e) => blast(e.clientX / window.innerWidth, e.clientY / window.innerHeight));
window.addEventListener("pointerup", release);
window.addEventListener("blur", release);

// For PixelBattle.exe --selftest: what the battle is doing, every 2 seconds.
const webview = (window as unknown as { chrome?: { webview?: { postMessage: (m: string) => void } } }).chrome?.webview;
if (webview && params.has("report")) {
  const REPORT_MS = 2000;
  let frames = sceneState.frames;
  window.setInterval(() => {
    webview.postMessage(
      JSON.stringify({
        ready: sceneState.ready,
        x: Math.round(sceneState.cam.x),
        fps: (sceneState.frames - frames) / (REPORT_MS / 1000),
        speed: sceneState.speed,
        zoom: Math.round(sceneState.scaleK * 100),
        maxFps: sceneState.maxFps,
        w: window.innerWidth,
        h: window.innerHeight,
        dpr: window.devicePixelRatio,
        hidden: document.hidden,
      })
    );
    frames = sceneState.frames;
  }, REPORT_MS);
}

createRoot(document.getElementById("root")!).render(<PixelScene />);
