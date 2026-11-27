/**
 * The browser frame loop: turns real time into fixed simulation ticks, then renders with the leftover fraction
 * of a tick for interpolation. With `maxFps` > 0 it sleeps on a timer until a frame is nearly due and only then
 * asks for an animation frame, as the old PixelScene `schedule()` did, so a 30 fps Wallpaper doesn't wake for
 * every display refresh.
 */

import { stepFixed } from "./core/scheduler";
import { now } from "./core/stats";
import type { World } from "./core/world";

interface VisibilityDoc {
  readonly hidden: boolean;
  addEventListener(type: "visibilitychange", fn: () => void): void;
  removeEventListener(type: "visibilitychange", fn: () => void): void;
}

/** The browser services the loop uses; tests pass fakes. */
export interface LoopHost {
  now(): number;
  requestFrame(cb: (ms: number) => void): number;
  cancelFrame(id: number): void;
  setTimer(cb: () => void, ms: number): number;
  clearTimer(id: number): void;
  /** For the visibility pause; null where there is no document. */
  doc: VisibilityDoc | null;
}

export interface LoopOptions {
  /** Draw the world. `alpha` is how far (0..1) to interpolate from prev to pos; `frameDt` is real seconds. */
  render?: (alpha: number, frameDt: number) => void;
  /** 0 (default) = every display frame; > 0 = timer-scheduled frames at about that rate. Settable later. */
  maxFps?: number;
  /** An extra slow-motion factor read every frame, on top of `world.time.slowmo`. */
  slowmo?: () => number;
  /** While this returns true the loop sleeps: no ticks, no renders (the Wallpaper when windows cover it). */
  sleep?: () => boolean;
  /** Longest real frame counted, in seconds (default 0.25), so a stall or a breakpoint doesn't fast-forward. */
  maxFrameDt?: number;
  /** After every frame that drew, with the number of ticks it ran. */
  onFrame?: (ticks: number) => void;
  host?: Partial<LoopHost>;
}

/** Leave this much of the frame budget to rAF, which then lands the frame on a display refresh. */
const TIMER_SLACK_MS = 6;
/** While asleep or hidden, check back this often instead of every frame. */
const SLEEP_POLL_MS = 100;

function defaultHost(): LoopHost {
  const g = globalThis as typeof globalThis & Partial<Window>;
  const raf = typeof g.requestAnimationFrame === "function" ? g.requestAnimationFrame.bind(g) : null;
  const caf = typeof g.cancelAnimationFrame === "function" ? g.cancelAnimationFrame.bind(g) : null;
  return {
    now,
    requestFrame: raf ?? ((cb) => setTimeout(() => cb(now()), 16) as unknown as number),
    cancelFrame: caf ?? ((id) => clearTimeout(id)),
    setTimer: (cb, ms) => setTimeout(cb, ms) as unknown as number,
    clearTimer: (id) => clearTimeout(id),
    doc: typeof document !== "undefined" ? document : null,
  };
}

export class Loop {
  /** 0 = every display frame; > 0 = timer-scheduled frames at about that rate. */
  maxFps: number;
  /** Frames drawn since start. */
  frames = 0;
  private readonly host: LoopHost;
  private raf = 0;
  private timer = 0;
  private last = -1;
  private isRunning = false;
  private hidden = false;
  /** Draw on the next frame even if nothing moved (paused, after a resize). */
  private dirty = true;
  private fpsStart = -1;
  private fpsFrames = 0;

  constructor(
    readonly world: World,
    private readonly opts: LoopOptions = {}
  ) {
    this.maxFps = opts.maxFps ?? 0;
    this.host = { ...defaultHost(), ...opts.host };
  }

  get running(): boolean {
    return this.isRunning;
  }

  start(): void {
    if (this.isRunning) return;
    this.isRunning = true;
    this.last = -1;
    this.dirty = true;
    const doc = this.host.doc;
    if (doc) {
      this.hidden = doc.hidden;
      doc.addEventListener("visibilitychange", this.onVisibility);
    }
    this.schedule();
  }

  stop(): void {
    if (!this.isRunning) return;
    this.isRunning = false;
    if (this.raf) this.host.cancelFrame(this.raf);
    if (this.timer) this.host.clearTimer(this.timer);
    this.raf = this.timer = 0;
    this.host.doc?.removeEventListener("visibilitychange", this.onVisibility);
  }

  get paused(): boolean {
    return this.world.time.paused;
  }

  /** Freeze the simulation; the screen holds on the last frame. */
  pause(): void {
    this.world.time.paused = true;
    this.dirty = true;
  }

  resume(): void {
    this.world.time.paused = false;
  }

  get speed(): number {
    return this.world.time.speed;
  }

  set speed(v: number) {
    this.world.time.speed = v > 0 ? v : 0;
  }

  /** Draw on the next frame even while paused (after a resize or a settings change). */
  invalidate(): void {
    this.dirty = true;
  }

  private readonly onVisibility = (): void => {
    this.hidden = this.host.doc?.hidden ?? false;
    // Coming back, don't count the time away as one huge frame, and don't wait out the sleep poll
    this.last = -1;
    if (!this.hidden && this.isRunning) {
      if (this.timer) this.host.clearTimer(this.timer);
      this.timer = 0;
      this.schedule();
    }
  };

  private readonly onFrameCb = (ms: number): void => {
    this.raf = 0;
    this.frame(ms);
  };

  private readonly onTimer = (): void => {
    this.timer = 0;
    if (this.isRunning) this.raf = this.host.requestFrame(this.onFrameCb);
  };

  private schedule(): void {
    if (this.raf || this.timer) return;
    if (this.maxFps > 0)
      this.timer = this.host.setTimer(this.onTimer, Math.max(0, 1000 / this.maxFps - TIMER_SLACK_MS));
    else this.raf = this.host.requestFrame(this.onFrameCb);
  }

  /** One frame at time `ms` (the rAF timestamp). Public so hosts that own their frame clock can drive it. */
  frame(ms: number): void {
    if (!this.isRunning) return;
    if (this.hidden || this.opts.sleep?.()) {
      this.last = -1;
      if (!this.raf && !this.timer) this.timer = this.host.setTimer(this.onTimer, SLEEP_POLL_MS);
      return;
    }
    this.schedule();
    const world = this.world,
      stats = world.stats;
    const maxDt = this.opts.maxFrameDt ?? 0.25;
    const frameDt = this.last < 0 ? 0 : Math.min(maxDt, Math.max(0, (ms - this.last) / 1000));
    this.last = ms;
    const hook = this.opts.slowmo ? this.opts.slowmo() : 1;
    const scale = world.time.scale * (hook > 0 ? hook : 0);
    const t0 = now();
    const ticks = scale > 0 ? stepFixed(world, frameDt * scale) : 0;
    stats.ticksLastFrame = ticks;
    // Moving time redraws every frame (alpha changes even without a tick); paused, draw once and hold
    if (scale > 0 || ticks > 0 || this.dirty) {
      this.dirty = false;
      if (this.opts.render) {
        const r0 = now();
        this.opts.render(world.time.alpha, frameDt);
        stats.render.sample(now() - r0);
      }
      this.frames++;
      this.fpsFrames++;
      stats.frame.sample(now() - t0);
      this.opts.onFrame?.(ticks);
    }
    if (this.fpsStart < 0) {
      // The window opens at this frame, so it doesn't count towards it
      this.fpsStart = ms;
      this.fpsFrames = 0;
    } else if (ms - this.fpsStart >= 1000) {
      stats.fps = (this.fpsFrames * 1000) / (ms - this.fpsStart);
      this.fpsStart = ms;
      this.fpsFrames = 0;
    }
  }
}

export const createLoop = (world: World, opts?: LoopOptions): Loop => new Loop(world, opts);
