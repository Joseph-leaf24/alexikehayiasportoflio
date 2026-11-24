import { createRoot } from "react-dom/client";
import PixelScene from "@/components/pixel/PixelScene";
import { sceneState } from "@/lib/store";
import "./wallpaper.css";

/**
 * The pixel battle as a live wallpaper. The same scene as the site's, on its own and full
 * screen: at twice the speed, zoomed out by a quarter (more of the battlefield on the desktop),
 * the whole battle as the site's spectator space shows it, capped at 30 frames a second and on
 * a lighter budget so it goes easy on the machine. A click on the desktop (if the wallpaper app
 * passes clicks on) calls in a blast there; holding the button down brings a barrage in.
 * Lively Wallpaper's and Wallpaper Engine's settings panels can change the speed, the zoom, the
 * frame rate and the clicking.
 */
const SPEEDS = [1, 2, 4];
const RATES = [30, 60];

sceneState.speed = 2;
sceneState.scaleK = 0.75;
sceneState.maxFps = 30;
// (always on show: no slowing down when nobody's touched anything for a while)
sceneState.watching = true;
// (the wildest of the site's sections: the spectator space at the bottom of the page)
sceneState.stops = [{ y: -1e9, layout: "watch" }];
let blasts = true;

/* ---- Clicks on the desktop ---- */
const at = (e: PointerEvent) => {
  sceneState.pointer.x = (e.clientX / window.innerWidth) * 2 - 1;
  sceneState.pointer.y = -(e.clientY / window.innerHeight) * 2 + 1;
  sceneState.pointer.lastMove = performance.now();
};
let hold = 0;
window.addEventListener("pointermove", at, { passive: true });
window.addEventListener("pointerdown", (e) => {
  at(e);
  if (!blasts) return;
  sceneState.burst++;
  window.clearTimeout(hold);
  hold = window.setTimeout(() => (sceneState.press = true), 450);
});
const release = () => {
  window.clearTimeout(hold);
  sceneState.press = false;
};
window.addEventListener("pointerup", release);
window.addEventListener("pointercancel", release);
window.addEventListener("blur", release);

/* ---- Settings ---- */
const setZoom = (pct: number) => (sceneState.scaleK = Math.min(1, Math.max(0.5, pct / 100)));

declare global {
  interface Window {
    /** Lively Wallpaper calls this with each of LivelyProperties.json's settings, at the start and whenever one's changed. */
    livelyPropertyListener?: (name: string, value: unknown) => void;
    /** Wallpaper Engine calls these with project.json's settings, and with its own (its frame rate limit). */
    wallpaperPropertyListener?: {
      applyUserProperties?: (props: Record<string, { value: unknown }>) => void;
      applyGeneralProperties?: (props: { fps?: number }) => void;
    };
  }
}

window.livelyPropertyListener = (name, value) => {
  if (name === "speed") sceneState.speed = SPEEDS[Number(value)] ?? 2;
  else if (name === "zoom") setZoom(Number(value));
  else if (name === "fps") sceneState.maxFps = RATES[Number(value)] ?? 30;
  else if (name === "blasts") blasts = !!value;
};

window.wallpaperPropertyListener = {
  applyUserProperties(props) {
    if (props.speed) sceneState.speed = Number(props.speed.value) || 2;
    if (props.zoom) setZoom(Number(props.zoom.value));
    if (props.fps) sceneState.maxFps = Number(props.fps.value) || 30;
    if (props.blasts) blasts = !!props.blasts.value;
  },
  // (Wallpaper Engine's own frame rate limit, for web wallpapers to keep to: never above ours)
  applyGeneralProperties(props) {
    if (props.fps) sceneState.maxFps = Math.min(sceneState.maxFps || 60, props.fps);
  },
};

createRoot(document.getElementById("root")!).render(<PixelScene />);
