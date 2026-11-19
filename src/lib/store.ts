/**
 * Shared app state: a tiny reactive store for the UI (loader, HUD, menu) and a plain
 * mutable object the WebGL scene reads every frame.
 */
import { useSyncExternalStore } from "react";

/* ------------------------------------------------------------------ */
/* Tiny reactive store (no dependency needed)                          */
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
  return useSyncExternalStore(store.subscribe, () => select(store.get()), () => select(store.get()));
}

/* ------------------------------------------------------------------ */
/* UI state                                                            */
/* ------------------------------------------------------------------ */

export const ui = createStore({
  /** Loader finished and the page is revealed. */
  ready: false,
  /** Index + label shown in the HUD for the section (or page) currently in view. */
  sectionIndex: "00",
  sectionLabel: "INTRO",
  /** Mobile menu is open. */
  menuOpen: false,
});

/** Run `cb` once the loader has finished (immediately if it already has). */
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
/* WebGL scene state — mutable on purpose, read every frame            */
/* ------------------------------------------------------------------ */

/**
 * Formations the cubes morph between: one per home section (village → shatter → clusters →
 * network → bars → villageTop), plus "cube" for project pages and "lost" for the 404.
 */
export type LayoutKey = "village" | "shatter" | "clusters" | "network" | "bars" | "villageTop" | "cube" | "lost";

/** Written by the page / input handlers and read by the scene on every frame (not reactive). */
export const sceneState = {
  /** Scroll positions where each home section becomes dominant. */
  stops: [] as { y: number; layout: LayoutKey }[],
  /** When set (detail / 404 pages) the scene ignores scroll and holds this layout. */
  override: null as LayoutKey | null,
  /** Pointer in normalised device coords (-1..1); lastMove is performance.now() of the last mouse move. */
  pointer: { x: 0, y: 0, lastMove: -1e9 },
  /** Hold-to-shatter on the hero. */
  press: false,
  /** A page-wipe transition is running (slow devices pause the 3D render meanwhile). */
  transition: false,
  /** performance.now() / 1000 of the last ripple pulse. */
  pulse: -100,
  /** Scroll velocity in px/s (written by the smooth-scroll driver, decayed by the scene). */
  velocity: 0,
  /** Scene has compiled and rendered its first frame. */
  ready: false,
  /** False if WebGL is unavailable — the site still works without it. */
  webgl: true,
  /** Number of cubes in the scene (filled in by the scene, shown by the loader and HUD). */
  blocks: 0,
  /** Domain card being hovered (index into the clusters formation's anchors, -1 = none). */
  highlight: -1,
  /** Incremented on every click — the scene answers with a burst of ice dust. */
  burst: 0,
  /** Frames per second the scene is drawing, and the resolution (device-pixel ratio) it draws at. */
  fps: 0,
  res: 1,
  /** Dominant formation + camera position, read by the HUD telemetry. */
  formation: "village" as LayoutKey,
  cam: { x: 0, y: 0, z: 0 },
};

/** Send a ripple through the scene (loader exit, page transitions, card hovers, tab changes). */
export const triggerPulse = () => {
  sceneState.pulse = performance.now() / 1000;
};
