import type { RenderFrame, Renderer } from "./types";

/** The parts of the core's `world.stats` the overlay shows; any object of this shape works. */
export interface DebugWorldStats {
  systems?: ReadonlyArray<{ readonly name: string; ms: number; max?: number }>;
  tps?: number;
  entities?: number;
  tick?: { ms: number };
}

/** True when the page was opened with `?debug=1`. */
export function debugFromUrl(): boolean {
  try {
    return typeof location !== "undefined" && new URLSearchParams(location.search).get("debug") === "1";
  } catch {
    return false;
  }
}

const COLOR = 0xffe27aff;
const DIM = 0xc8d0e0ff;
/** Text is rebuilt this often (ms); in between the same strings are pushed, so the overlay allocates almost nothing. */
const REFRESH = 250;
const MAX_SYSTEMS = 14;

const ms = (v: number) => (v >= 10 ? v.toFixed(1) : v.toFixed(2));

/**
 * The `?debug=1` overlay: fps, frame ms, per-system ms from the world's stats, sprite, particle and light counts and
 * draw calls, in the pixel font at the top left of the overlay.
 */
export class DebugOverlay {
  enabled: boolean;
  /** Frames per second over the last refresh window. */
  fps = 0;
  /** Rolling average wall-clock ms between frames, and the worst in the last window. */
  frameMs = 0;
  frameMax = 0;
  private renderMs = 0;
  private lines: string[] = [];
  private windowStart = -1;
  private windowFrames = 0;
  private windowMax = 0;

  constructor(enabled = debugFromUrl()) {
    this.enabled = enabled;
  }

  /** Call once per displayed frame with the wall-clock ms since the previous one. */
  tick(dtMs: number) {
    this.frameMs = this.frameMs === 0 ? dtMs : this.frameMs + (dtMs - this.frameMs) * 0.05;
    if (dtMs > this.windowMax) this.windowMax = dtMs;
    this.windowFrames++;
  }

  /** Pushes the overlay into `frame.text`; reads the renderer's stats from its previous frame. */
  draw(frame: RenderFrame, renderer: Renderer, world?: DebugWorldStats | null, now = performance.now()) {
    if (!this.enabled) return;
    const r = renderer.stats;
    this.renderMs = this.renderMs === 0 ? r.cpuMs : this.renderMs + (r.cpuMs - this.renderMs) * 0.05;
    if (this.windowStart < 0 || now - this.windowStart >= REFRESH || this.lines.length === 0) {
      const span = this.windowStart < 0 ? 0 : now - this.windowStart;
      if (span > 0) this.fps = (this.windowFrames * 1000) / span;
      this.frameMax = this.windowMax;
      this.windowStart = now;
      this.windowFrames = 0;
      this.windowMax = 0;
      this.rebuild(renderer, world ?? null);
    }
    for (let i = 0; i < this.lines.length; i++)
      frame.text.push(this.lines[i], 3, 3 + i * 7, i < 3 ? COLOR : DIM, 1, 0, 15);
  }

  private rebuild(renderer: Renderer, world: DebugWorldStats | null) {
    const r = renderer.stats;
    const L = this.lines;
    L.length = 0;
    L.push(`FPS ${Math.round(this.fps)}  FRAME ${ms(this.frameMs)}MS  MAX ${ms(this.frameMax)}`);
    const gpu = r.gpuMs !== undefined && r.gpuMs >= 0 ? `  GPU ${ms(r.gpuMs)}MS` : "";
    L.push(`${renderer.kind.toUpperCase()} ${renderer.width}X${renderer.height}  CPU ${ms(this.renderMs)}MS${gpu}`);
    L.push(`DRAWS ${r.drawCalls}  SPRITES ${r.sprites}  PARTS ${r.particles}  LIGHTS ${r.lights}  CHUNKS ${r.chunks}`);
    if (!world) return;
    const tick = world.tick ? `  TICK ${ms(world.tick.ms)}MS` : "";
    L.push(`TPS ${Math.round(world.tps ?? 0)}  ENTITIES ${world.entities ?? 0}${tick}`);
    const sys = world.systems ?? [];
    for (let i = 0; i < sys.length && i < MAX_SYSTEMS; i++) {
      const s = sys[i];
      L.push(`${s.name} ${ms(s.ms)}${s.max !== undefined ? ` / ${ms(s.max)}` : ""}`);
    }
  }
}
