/** Stepping a world without a browser or a renderer: tests, soak runs and benchmarks. */

import { tick, TICK } from "./core/scheduler";
import { now } from "./core/stats";
import type { World } from "./core/world";

export interface HeadlessOptions {
  /** After every tick; return false to stop early. */
  onTick?: (world: World, tick: number) => boolean | void;
  /** Stop once this much wall-clock time has passed (a guard for soak runs). */
  maxWallMs?: number;
}

export interface HeadlessReport {
  ticks: number;
  simSeconds: number;
  wallMs: number;
  /** Mean wall-clock milliseconds per tick. */
  msPerTick: number;
  /** 95th-percentile and worst tick. */
  p95TickMs: number;
  maxTickMs: number;
  /** Whether `onTick` or `maxWallMs` ended the run before `seconds`. */
  stoppedEarly: boolean;
}

/**
 * Run `seconds` of simulation as fast as possible, one fixed tick at a time. Ignores pause and the time scale,
 * which only pace real-time play; the result is identical to the same number of ticks under the frame loop.
 */
export function runHeadless(world: World, seconds: number, opts: HeadlessOptions = {}): HeadlessReport {
  const total = Math.max(0, Math.round(seconds / TICK));
  const times = new Float64Array(total);
  const start = now();
  let ran = 0,
    stoppedEarly = false;
  while (ran < total) {
    const t0 = now();
    tick(world);
    const t1 = now();
    times[ran++] = t1 - t0;
    if (
      opts.onTick?.(world, world.time.tick) === false ||
      (opts.maxWallMs !== undefined && t1 - start >= opts.maxWallMs)
    ) {
      stoppedEarly = ran < total;
      break;
    }
  }
  world.time.alpha = 0;
  const wallMs = now() - start;
  const sorted = times.subarray(0, ran).sort();
  return {
    ticks: ran,
    simSeconds: ran * TICK,
    wallMs,
    msPerTick: ran > 0 ? wallMs / ran : 0,
    p95TickMs: ran > 0 ? sorted[Math.min(ran - 1, Math.floor(ran * 0.95))] : 0,
    maxTickMs: ran > 0 ? sorted[ran - 1] : 0,
    stoppedEarly,
  };
}
