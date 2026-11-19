import { useSyncExternalStore } from "react";

/* ------------------------------------------------------------------ */
/* Tiny reactive store (no dependency needed)                          */
/* ------------------------------------------------------------------ */

type Listener = () => void;

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

export function useStore<T extends object, S>(store: ReturnType<typeof createStore<T>>, select: (s: T) => S): S {
  return useSyncExternalStore(store.subscribe, () => select(store.get()), () => select(store.get()));
}

/* ------------------------------------------------------------------ */
/* UI state                                                            */
/* ------------------------------------------------------------------ */

export const ui = createStore({
  /** Loader finished and the page is revealed. */
  ready: false,
  /** Label shown in the HUD for the section currently in view. */
  sectionIndex: "00",
  sectionLabel: "INTRO",
  menuOpen: false,
});

export const isAppReady = () => ui.get().ready;

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

export type LayoutKey = "village" | "shatter" | "clusters" | "network" | "bars" | "villageTop" | "cube" | "lost";

export const sceneState = {
  /** Scroll positions where each home section becomes dominant. */
  stops: [] as { y: number; layout: LayoutKey }[],
  /** When set (detail / 404 pages) the scene ignores scroll and holds this layout. */
  override: null as LayoutKey | null,
  /** Pointer in normalised device coords (-1..1). */
  pointer: { x: 0, y: 0, lastMove: -1e9 },
  /** Hold-to-shatter on the hero. */
  press: false,
  /** performance.now() / 1000 of the last ripple pulse. */
  pulse: -100,
  /** Smoothed scroll velocity in px/s (written by the smooth-scroll driver). */
  velocity: 0,
  /** Scene has compiled and rendered its first frame. */
  ready: false,
  /** False if WebGL is unavailable — the site still works without it. */
  webgl: true,
  /** Number of ice blocks (filled in by the scene, shown by the loader). */
  blocks: 0,
  /** Domain card being hovered (index into the clusters formation's anchors, -1 = none). */
  highlight: -1,
  /** Incremented on every click — the scene answers with a burst of ice dust. */
  burst: 0,
  /** Dominant formation + camera position, read by the HUD telemetry. */
  formation: "village" as LayoutKey,
  cam: { x: 0, y: 0, z: 0 },
};

export const triggerPulse = () => {
  sceneState.pulse = performance.now() / 1000;
};
