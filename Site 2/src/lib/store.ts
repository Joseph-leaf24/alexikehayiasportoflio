/**
 * Shared app state: a tiny reactive store for the UI and a plain mutable object the pixel scene
 * reads every frame.
 */
import { useSyncExternalStore } from "react";

/* ------------------------------------------------------------------ */
/* Tiny reactive store                                                 */
/* ------------------------------------------------------------------ */

type Listener = () => void;

/** A minimal observable store: `set` merges a patch and notifies only if a value changed. */
export function createStore<T extends object>(initial: T) {
  let state = initial;
  const listeners = new Set<Listener>();
  return {
    get: () => state,
    set(patch: Partial<T>) {
      let changed = false;
      for (const k in patch) {
        if (!Object.is(state[k], patch[k])) {
          changed = true;
          break;
        }
      }
      if (!changed) return;
      state = { ...state, ...patch };
      listeners.forEach((l) => l());
    },
    subscribe(l: Listener) {
      listeners.add(l);
      return () => listeners.delete(l);
    },
  };
}

/** Subscribe a component to a slice of a store. */
export function useStore<T extends object, S>(store: ReturnType<typeof createStore<T>>, select: (s: T) => S): S {
  return useSyncExternalStore(
    store.subscribe,
    () => select(store.get()),
    () => select(store.get())
  );
}

/* ------------------------------------------------------------------ */
/* UI state                                                            */
/* ------------------------------------------------------------------ */

export const ui = createStore({
  /** Loader finished and the page is revealed. */
  ready: false,
  /** Index and label shown in the HUD for the section or page in view. */
  sectionIndex: "00",
  sectionLabel: "INTRO",
  /** Mobile menu is open. */
  menuOpen: false,
});

/** Run `cb` once the loader has finished, or straight away if it already has. */
export function onAppReady(cb: () => void): () => void {
  if (ui.get().ready) {
    cb();
    return () => {};
  }
  let done = false;
  const unsub = ui.subscribe(() => {
    if (!done && ui.get().ready) {
      done = true;
      unsub();
      cb();
    }
  });
  return () => {
    done = true;
    unsub();
  };
}

/* ------------------------------------------------------------------ */
/* Pixel scene state — mutable on purpose, read every frame             */
/* ------------------------------------------------------------------ */

/**
 * Scene zones: the home page's sections in order, ending with "watch", the spectator space at the
 * bottom; then "cube" for project pages and "lost" for the 404, whose level is a cave.
 */
export type LayoutKey =
  | "village"
  | "shatter"
  | "clusters"
  | "network"
  | "bars"
  | "villageTop"
  | "watch"
  | "cube"
  | "lost";

/** One finished level. */
interface LevelLog {
  no: number;
  name: string;
  kills: number;
  /** Seconds it took */
  time: number;
}

/** Details of the level in play, written by the scene every frame. */
export const blankIntel = () => ({
  /** 0 at the drop-in, 1 at the enemy stronghold */
  progress: 0,
  objective: "",
  /** The level's boss ("" if none) and how that fight is going */
  boss: "",
  bossState: "none" as "none" | "ahead" | "engaged" | "destroyed" | "bypassed",
  levelKills: 0,
  levelTime: 0,
  /** Enemies on screen right now */
  hostiles: 0,
  /** Levels finished, and the last few */
  cleared: 0,
  log: [] as LevelLog[],
  /** Seconds the battle has been running */
  uptime: 0,
  weather: "",
  light: "",
  /** How hard the fighting is right now: a lull, a skirmish, an assault, an onslaught */
  wave: "",
});

/** Written by the page and input handlers and read by the scene every frame. Not reactive. */
export const sceneState = {
  /** Scroll positions where each home section becomes dominant. */
  stops: [] as { y: number; layout: LayoutKey }[],
  /** Set on project and 404 pages; a change starts a new level. */
  override: null as LayoutKey | null,
  /** Pointer in normalised device coords (-1..1); lastMove is performance.now() of the last mouse move. */
  pointer: { x: 0, y: 0, lastMove: -1e9 },
  /** Hold on the hero: calls in an airstrike. */
  press: false,
  /** A page-wipe transition is running. */
  transition: false,
  /** A full-screen overlay hides the scene, so there is no need to draw. */
  covered: false,
  /** performance.now() / 1000 of the last UI ping, which flashes up the players' tags. */
  pulse: -100,
  /** Scroll velocity in px/s, written by the smooth-scroll driver and decayed by the scene. */
  velocity: 0,
  /** Scene has built its world and drawn its first frame. */
  ready: false,
  /** The visitor paused the scene, and how fast it plays (0.5× to 8×), from the scene controls. */
  paused: false,
  speed: 1,
  /** Most frames a second the scene draws; 0 means as many as the display does. The wallpaper caps it. */
  maxFps: 0,
  /** Frames the scene has drawn, for the wallpaper's self-test */
  frames: 0,
  /** How big the scene's pixels are against the usual; 0.75 is zoomed out by a quarter. Set by the wallpaper. */
  scaleK: 1,
  /** Endless: the levels run on from biome to biome, with no finale and no "Level complete". */
  endless: false,
  /** Zoom-out steps from the normal view (0): more of the battlefield, in smaller pixels. */
  zoom: 0,
  /** False if the 2D canvas is unavailable — the site still works without it. */
  canvas: true,
  /** Enemies down and explosions so far, written by the scene. */
  kills: 0,
  booms: 0,
  /** P1–P8: score, health (0–1) and whether they're up. */
  players: Array.from({ length: 8 }, () => ({ score: 0, hp: 1, alive: true })),
  /** The level being played, counting from 1, and its theme's name. */
  level: 1,
  stage: "",
  /** The level in play, in detail. */
  intel: blankIntel(),
  /** The spectator space at the bottom of the home page is in view: the scene keeps its full frame rate. */
  watching: false,
  /** The end-of-level white-out over the scene (0–1), so the page can fade its HUD with it. */
  white: 0,
  /** Domain card being hovered: that domain's player is pointed out in the scene (-1 = none). */
  highlight: -1,
  /** Incremented on a click outside the controls — the scene answers with a blast under the cursor. */
  burst: 0,
  /** Camera position: x, y in scene pixels, z the pixel scale. */
  cam: { x: 0, y: 0, z: 0 },
};

/* ------------------------------------------------------------------ */
/* Navigation within this page load                                    */
/* ------------------------------------------------------------------ */

/**
 * Whether the visitor has moved within the site since the page loaded, by a link or the browser's
 * back and forward buttons. Until then the home page opens on the intro: the router state and the
 * saved scroll position describe moves within the site, but both outlive a reload.
 */
export const nav = { inApp: false };
// Registered at import, before the router's own listener, so it is set before the new route renders
if (typeof window !== "undefined") window.addEventListener("popstate", () => (nav.inApp = true));

/** Ping the scene on UI events: the loader exit, page transitions, card hovers and tab changes. */
export const triggerPulse = () => {
  sceneState.pulse = performance.now() / 1000;
};
