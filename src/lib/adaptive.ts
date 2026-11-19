/**
 * Adaptive render quality for the 3D scene. The aim: a steady 60 fps on every device, at
 * the best quality that device can hold. The scene moves up and down a ladder of render
 * settings while it runs: it climbs while frames are clean, steps down as soon as they
 * start to miss, and remembers rungs that failed so it doesn't keep re-trying them (each
 * failed try costs a moment of stutter). Pure logic: the scene feeds in frame intervals and
 * applies the rung the controller picks.
 */
import type { Tier } from "./device";

/** One set of render settings. */
export interface Rung {
  /** Device-pixel ratio the canvas renders at */
  dpr: number;
  /** Grain / aberration / glitch composite pass */
  fx: boolean;
  bloom: boolean;
  /** Share of the snow particles drawn (0–1) */
  snow: number;
  /** Render every other display refresh (30 fps on a 60 Hz screen): the last resort */
  half: boolean;
}

/** The frame budget: 60 fps. */
export const TARGET_MS = 1000 / 60;

/**
 * The ladder for a screen with this device-pixel ratio, cheapest first: resolution up to
 * 1×, then the grain pass, then bloom, then resolution above 1× up to the screen's own
 * (at most 2×; beyond that extra pixels don't show under bloom and grain).
 */
export function buildLadder(nativeDpr: number): Rung[] {
  const top = Math.min(2, Math.max(1, nativeDpr || 1));
  const r = (dpr: number, fx: boolean, bloom: boolean, snow: number, half = false): Rung => ({ dpr, fx, bloom, snow, half });
  const ladder = [
    r(0.5, false, false, 0.35, true),
    r(0.5, false, false, 0.35),
    r(0.65, false, false, 0.5),
    r(0.8, true, false, 0.75),
    r(1, true, false, 1),
    r(1, true, true, 1),
  ];
  for (const d of [1.25, 1.5, 1.75, 2]) if (d <= top + 1e-6) ladder.push(r(d, true, true, 1));
  // A screen between the steps (say 1.6×) gets its exact ratio as the top rung
  if (top - ladder[ladder.length - 1].dpr > 0.05) ladder.push(r(top, true, true, 1));
  return ladder;
}

/** Where each device tier starts, before anything has been measured. */
export function startRung(ladder: Rung[], tier: Tier): number {
  const last = (ok: (r: Rung) => boolean) => ladder.reduce((best, r, i) => (ok(r) ? i : best), 0);
  if (tier === "low") return 2; // 0.65×, no effects
  if (tier === "mid") return last((r) => r.bloom && r.dpr <= 1);
  return last((r) => r.dpr <= 1.5);
}

/**
 * Display pacing: estimates the screen's refresh rate from the quickest recent rAF
 * intervals and returns how many refreshes to wait between rendered frames, so a 120 or
 * 144 Hz screen renders every other refresh (about 60 fps, leaving GPU time for quality)
 * and a 60–100 Hz screen renders every refresh.
 */
export function createPacer() {
  const ring: number[] = [];
  let divisor = 1;
  return {
    /** Feed one rAF interval (ms); returns the current divisor. */
    sample(dt: number) {
      if (dt > 0 && dt < 100) ring.push(dt);
      if (ring.length >= 30) {
        const sorted = [...ring].sort((a, b) => a - b);
        const hz = 1000 / sorted[Math.floor(sorted.length * 0.2)];
        divisor = hz > 100 ? Math.min(4, Math.max(1, Math.floor(hz / 60 + 0.15))) : 1;
        ring.length = 0;
      }
      return divisor;
    },
  };
}

export interface ControllerOptions {
  /** Rung to start on */
  start: number;
  /** Rendered frames per measurement (a measurement also ends after ~1 s, so slow devices decide quickly) */
  window?: number;
  /** Frames to ignore at start-up (shader compiles, the intro); at most 1.5 s */
  warmup?: number;
  onChange: (level: number) => void;
  /** Still missing frames on the lowest rung: the build-time budget is too big */
  onBottom?: () => void;
  /** Holding 60 fps with ease on the top rung: there's room for a bigger budget */
  onTop?: () => void;
}

const RUNG_KEY = "ak:rung";

/** The rung this session last settled on (a reload starts there), if valid for this ladder. */
export function rememberedRung(size: number): number | null {
  try {
    const raw = sessionStorage.getItem(RUNG_KEY);
    const v = Number(raw);
    return raw !== null && Number.isInteger(v) && v >= 0 && v < size ? v : null;
  } catch {
    return null;
  }
}

export function rememberRung(level: number) {
  try {
    sessionStorage.setItem(RUNG_KEY, String(level));
  } catch {
    /* storage blocked */
  }
}

/**
 * A frame longer than this, and much longer than the device's typical frame, is a one-off
 * hitch (a route change, GC), not a trend. (Relative, so a device that is slow on every
 * frame still gets measured.)
 */
const HITCH_MS = 250;

export function createController(ladder: Rung[], o: ControllerOptions) {
  const size = ladder.length;
  const W = o.window ?? 40;
  let level = Math.min(size - 1, Math.max(0, o.start));
  /** Frames to skip (ends early once settleMs have passed, counted from the first one) */
  let settleFrames = o.warmup ?? 90;
  let settleMs = 1500;
  let settleUntil = -1;
  const settle = (frames: number, ms: number) => {
    settleFrames = frames;
    settleMs = ms;
    settleUntil = -1;
  };
  let n = 0;
  let sum = 0;
  let misses = 0;
  let windowStart = 0;
  /** Running typical frame interval (ms), hitches included */
  let typical = TARGET_MS;
  /** Consecutive clean windows */
  let clean = 0;
  /** Windows left in which bad frames count against the last step up */
  let probe = 0;
  const fails = new Array<number>(size).fill(0);
  const retryAt = new Array<number>(size).fill(0);
  let ceiling = size - 1;
  /** Don't go below this (until floorUntil): lower rungs were found not to help */
  let floor = 0;
  let floorUntil = 0;
  let floorHits = 0;
  /** The current run of steps down: where it began and how fast frames were then */
  let descent: { level: number; mean: number; steps: number } | null = null;
  let bottomWindows = 0;
  let topWindows = 0;
  let firedBottom = false;
  let firedTop = false;

  const set = (to: number) => {
    level = to;
    settle(20, 350); // the change itself (render targets resized) costs a frame or two
    n = sum = misses = 0;
    clean = 0;
    o.onChange(level);
  };

  const evaluate = (now: number) => {
    const budget = TARGET_MS * (ladder[level].half ? 2 : 1);
    const mean = sum / n;
    const miss = misses / n;
    n = sum = misses = 0;
    const bad = miss > 0.2 || mean > budget * 1.15;
    const good = miss < 0.03 && mean < budget * 1.06;

    if (bad) {
      clean = 0;
      topWindows = 0;
      if (probe > 0) {
        // The last step up was one too many: go back, and wait longer each time before retrying it
        probe = 0;
        const failed = level;
        fails[failed]++;
        retryAt[failed] = now + 8000 * 2 ** (fails[failed] - 1);
        if (fails[failed] >= 3) ceiling = Math.min(ceiling, failed - 1);
        set(failed - 1);
        return;
      }
      const lowest = now < floorUntil ? floor : 0;
      if (!descent) descent = { level, mean, steps: 0 };
      // (only full-rate rungs compare: on the 30 fps rung intervals are longer by design)
      else if (!ladder[level].half && descent.steps >= 2 && mean > descent.mean * 0.9) {
        // Two steps down and frames are no faster: the bottleneck is elsewhere (the CPU, or
        // a screen below 60 Hz), so lower settings only cost quality. Undo them.
        floor = descent.level;
        floorUntil = ++floorHits >= 2 ? Infinity : now + 30_000;
        const back = descent.level;
        descent = null;
        if (back !== level) set(back);
        return;
      }
      if (level <= lowest) {
        if (level === 0 && ++bottomWindows >= 3 && !firedBottom) {
          firedBottom = true;
          o.onBottom?.();
        }
        return;
      }
      descent.steps++;
      set(Math.max(lowest, level - (mean > budget * 1.8 ? 2 : 1)));
      return;
    }

    descent = null;
    bottomWindows = 0;
    if (probe > 0) probe--;
    if (!good) {
      clean = 0;
      return;
    }
    clean++;
    if (level === size - 1) {
      if (++topWindows >= 15 && !firedTop) {
        firedTop = true;
        o.onTop?.();
      }
      return;
    }
    if (clean >= 3 && level < ceiling && now >= retryAt[level + 1]) {
      set(level + 1);
      probe = 2;
    }
  };

  return {
    get level() {
      return level;
    },
    get rung() {
      return ladder[level];
    },
    /** Feed one rendered frame: ms since the previous rendered frame, and the time now (ms). */
    frame(interval: number, now: number) {
      const hitch = interval > Math.max(HITCH_MS, typical * 3);
      typical += (Math.min(interval, 2000) - typical) * 0.1;
      if (settleFrames > 0) {
        if (settleUntil < 0) settleUntil = now + settleMs;
        settleFrames--;
        if (now < settleUntil) return;
        settleFrames = 0;
      }
      if (hitch) return;
      if (n === 0) windowStart = now - interval;
      n++;
      sum += interval;
      // A frame "misses" when it takes noticeably longer than the budget (a skipped refresh)
      if (interval > TARGET_MS * (ladder[level].half ? 2 : 1) * 1.4) misses++;
      if (n >= W || (n >= 8 && now - windowStart >= 1000)) evaluate(now);
    },
    /** Start measuring afresh (after a hidden tab or a paused page transition). */
    reset() {
      n = sum = misses = 0;
      clean = 0;
      if (settleFrames < 10) settle(10, 200);
    },
  };
}
