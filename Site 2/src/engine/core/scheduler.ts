/**
 * Systems in a fixed phase order, run on a fixed timestep. The simulation only ever advances in whole ticks of
 * `TICK` seconds, so its state depends on the tick count and never on frame timing: the same seed and inputs give
 * the same world whether it runs at 144 fps, 30 fps or headless.
 */

import type { Query } from "./ecs";
import { now, SystemStat } from "./stats";
import type { World } from "./world";

export const PHASES = [
  "input",
  "director",
  "spawn",
  "ai",
  "weapons",
  "motion",
  "physics",
  "projectiles",
  "damage",
  "death",
  "effects",
  "terrain",
  "camera",
  "late",
] as const;
export type Phase = (typeof PHASES)[number];

/** Simulation step in seconds. */
export const TICK = 1 / 30;
/** Most ticks one `stepFixed` call runs; beyond it the backlog is dropped rather than chased (spiral of death). */
export const MAX_TICKS_PER_STEP = 12;
/** Float slack, so 1/60 s frames add up to a tick every second frame. */
const EPS = 1e-9;

export interface System {
  readonly name: string;
  readonly phase: Phase;
  run(world: World, dt: number): void;
  /** Sort key within the phase (default 0); ties run in the order the systems were added. */
  readonly order?: number;
  /** Run every n ticks (default 1) with dt = n × TICK: budgeted thinking, such as AI at 10 Hz. */
  readonly every?: number;
  /** Which tick of the `every` cycle it runs on, to stagger systems that share a cadence. */
  readonly offset?: number;
}

interface Entry {
  readonly sys: System;
  readonly phase: number;
  readonly order: number;
  readonly seq: number;
  readonly every: number;
  readonly offset: number;
  readonly dt: number;
  readonly stat: SystemStat;
  enabled: boolean;
}

export class Scheduler {
  private entries: Entry[] = [];
  /** entries[phaseEnd[p - 1] .. phaseEnd[p]) are phase p's systems. */
  private phaseEnd = new Int32Array(PHASES.length);
  private seq = 0;
  private running = false;
  private onChange: ((stats: SystemStat[]) => void) | null = null;

  /** @internal The world wires its stats list here. */
  bind(onChange: (stats: SystemStat[]) => void): void {
    this.onChange = onChange;
    onChange(this.entries.map((e) => e.stat));
  }

  add(sys: System): System {
    if (this.entries.some((e) => e.sys.name === sys.name)) throw new Error(`Scheduler: "${sys.name}" already added`);
    const phase = PHASES.indexOf(sys.phase);
    if (phase < 0) throw new Error(`Scheduler: unknown phase "${sys.phase}"`);
    const every = Math.max(1, Math.floor(sys.every ?? 1));
    this.entries.push({
      sys,
      phase,
      order: sys.order ?? 0,
      seq: this.seq++,
      every,
      offset: (((sys.offset ?? 0) % every) + every) % every,
      dt: every * TICK,
      stat: new SystemStat(sys.name, sys.phase),
      enabled: true,
    });
    this.rebuild();
    return sys;
  }

  remove(name: string): boolean {
    const k = this.entries.findIndex((e) => e.sys.name === name);
    if (k < 0) return false;
    this.entries.splice(k, 1);
    this.rebuild();
    return true;
  }

  get(name: string): System | undefined {
    return this.entries.find((e) => e.sys.name === name)?.sys;
  }

  /** Switch a system off or on without losing its place; false when there is no such system. */
  setEnabled(name: string, enabled: boolean): boolean {
    const e = this.entries.find((x) => x.sys.name === name);
    if (e) e.enabled = enabled;
    return e !== undefined;
  }

  isEnabled(name: string): boolean {
    return this.entries.find((x) => x.sys.name === name)?.enabled ?? false;
  }

  /** System names in run order. */
  get order(): string[] {
    return this.entries.map((e) => e.sys.name);
  }

  private rebuild(): void {
    if (this.running) throw new Error("Scheduler: systems can't be added or removed mid-tick");
    this.entries.sort((a, b) => a.phase - b.phase || a.order - b.order || a.seq - b.seq);
    let k = 0;
    for (let p = 0; p < PHASES.length; p++) {
      while (k < this.entries.length && this.entries[k].phase === p) k++;
      this.phaseEnd[p] = k;
    }
    this.onChange?.(this.entries.map((e) => e.stat));
  }

  /** @internal One tick of every system; use `tick(world)`. */
  run(world: World): number {
    if (this.running) throw new Error("Scheduler: tick() called from inside a tick");
    this.running = true;
    const timing = world.stats.timing,
      tickNo = world.time.tick,
      events = world.events;
    let delivered = 0;
    try {
      let k = 0;
      // One clock read per system: each one's end is the next one's start, unless events ran in between
      let t = timing ? now() : 0;
      for (let p = 0; p < PHASES.length; p++) {
        const end = this.phaseEnd[p];
        for (; k < end; k++) {
          const e = this.entries[k];
          if (!e.enabled || (e.every > 1 && tickNo % e.every !== e.offset)) continue;
          try {
            // The constant needs no boxing for the call; a dt loaded from the entry would allocate a number
            e.sys.run(world, e.every === 1 ? TICK : e.dt);
          } catch (err) {
            e.stat.errors++;
            if (!world.onError) throw err;
            world.onError(err, e.sys.name);
          }
          if (timing) {
            const t1 = now();
            e.stat.sample(t1 - t);
            t = t1;
          }
        }
        const n = events.flush();
        if (n > 0) {
          delivered += n;
          if (timing) t = now();
        }
      }
    } finally {
      this.running = false;
    }
    return delivered;
  }
}

const prevQueries = new WeakMap<World, Query>();

/** Copy pos into prev for every entity that has both, so rendering can interpolate across the tick. */
function snapshotPrev(world: World): void {
  const { pos, prev } = world.core;
  let q = prevQueries.get(world);
  if (!q) prevQueries.set(world, (q = world.query(pos, prev)));
  const n = q.collect(),
    idx = q.idx,
    px = pos.x,
    py = pos.y,
    qx = prev.x,
    qy = prev.y;
  for (let k = 0; k < n; k++) {
    const i = idx[k];
    qx[i] = px[i];
    qy[i] = py[i];
  }
}

/** Run exactly one tick: prev snapshot, every phase (events flushed after each), then the deferred destroys. */
export function tick(world: World): void {
  const t0 = world.stats.timing ? now() : 0;
  snapshotPrev(world);
  let delivered = world.systems.run(world);
  try {
    world.flushDestroyed();
  } catch (err) {
    // A destroy hook threw (every queued entity is gone regardless)
    if (!world.onError) throw err;
    world.onError(err, "destroy");
  }
  // Destroy notifications go out now rather than in the next tick's first phase
  delivered += world.events.flush();
  const time = world.time;
  time.tick++;
  time.now = time.tick * TICK;
  world.stats.endTick(world, world.stats.timing ? now() - t0 : 0, delivered);
}

/**
 * Advance by `dt` seconds of simulation time (real time × the time scale): 0..MAX_TICKS_PER_STEP ticks, the
 * remainder kept for next time. Sets `world.time.alpha`, the fraction of a tick left over, for interpolation.
 * Returns how many ticks ran.
 */
export function stepFixed(world: World, dt: number): number {
  const time = world.time;
  if (dt > 0 && Number.isFinite(dt)) time.acc += dt;
  let n = 0;
  while (time.acc >= TICK - EPS) {
    if (n >= MAX_TICKS_PER_STEP) {
      const dropped = Math.floor((time.acc + EPS) / TICK);
      world.stats.droppedTicks += dropped;
      time.acc -= dropped * TICK;
      break;
    }
    tick(world);
    time.acc -= TICK;
    n++;
  }
  if (time.acc < 0) time.acc = 0;
  time.alpha = time.acc >= TICK ? 1 : time.acc / TICK;
  return n;
}
