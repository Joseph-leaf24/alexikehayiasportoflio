/** Debug statistics the scheduler and loop keep up to date, read by the debug overlay. */

import type { Ecs } from "./ecs";

export const now: () => number =
  typeof performance !== "undefined" && typeof performance.now === "function"
    ? () => performance.now()
    : () => Date.now();

/** Weight of the newest sample in the rolling averages (about a 20-sample window). */
const EMA = 0.05;
/** Samples per peak window; `max` covers the current and the previous window. */
const PEAK_WINDOW = 60;

/** A rolling average and recent peak of one timing, in milliseconds. */
export class Timing {
  /** Rolling average. */
  ms = 0;
  /** Latest sample. */
  last = 0;
  /** Highest sample over the last one to two windows. */
  max = 0;
  samples = 0;
  private peak = 0;
  private prevPeak = 0;

  sample(ms: number): void {
    this.last = ms;
    this.ms = this.samples === 0 ? ms : this.ms + (ms - this.ms) * EMA;
    if (ms > this.peak) this.peak = ms;
    if (++this.samples % PEAK_WINDOW === 0) {
      this.prevPeak = this.peak;
      this.peak = 0;
    }
    this.max = this.peak > this.prevPeak ? this.peak : this.prevPeak;
  }

  reset(): void {
    this.ms = this.last = this.max = this.peak = this.prevPeak = 0;
    this.samples = 0;
  }
}

export class SystemStat extends Timing {
  /** Runs that threw (with `world.onError` set, the loop carries on). */
  errors = 0;
  constructor(
    readonly name: string,
    readonly phase: string
  ) {
    super();
  }
}

export interface ComponentStat {
  readonly name: string;
  count: number;
}

export class WorldStats {
  /** Time each system: two clock reads per system per tick. Off, systems report nothing. */
  timing = true;
  /** Per-system timings, in run order. */
  systems: SystemStat[] = [];
  /** Members per component, in definition order (refreshed every tick). */
  readonly components: ComponentStat[] = [];
  entities = 0;
  /** Ticks run since the world was made. */
  ticks = 0;
  /** Whole-tick time (systems, event delivery, destroys). */
  readonly tick = new Timing();
  /** Ticks per second of wall-clock time, over the last second. */
  tps = 0;
  /** Ticks the step cap dropped to stay real-time (a slow machine or a long stall). */
  droppedTicks = 0;
  /** Events delivered in the last tick. */
  events = 0;
  /* Filled by the frame loop */
  fps = 0;
  readonly frame = new Timing();
  readonly render = new Timing();
  ticksLastFrame = 0;

  private windowStart = -1;
  private windowTicks = 0;

  /** Called by the scheduler at the end of every tick. */
  endTick(ecs: Ecs, ms: number, delivered: number): void {
    this.ticks++;
    this.events = delivered;
    if (this.timing) this.tick.sample(ms);
    this.entities = ecs.count;
    const comps = ecs.components;
    for (let k = this.components.length; k < comps.length; k++)
      this.components.push({ name: comps[k].$.name, count: 0 });
    for (let k = 0; k < comps.length; k++) this.components[k].count = comps[k].$.size;
    // Ticks per second over windows of at least a second, reading the clock every 16th tick only
    this.windowTicks++;
    if (this.windowStart >= 0 && (this.ticks & 15) !== 0) return;
    const t = now();
    if (this.windowStart < 0) {
      this.windowStart = t;
      this.windowTicks = 0;
    } else if (t - this.windowStart >= 1000) {
      this.tps = (this.windowTicks * 1000) / (t - this.windowStart);
      this.windowStart = t;
      this.windowTicks = 0;
    }
  }

  /** A system's timing by name. */
  system(name: string): SystemStat | undefined {
    return this.systems.find((s) => s.name === name);
  }
}
