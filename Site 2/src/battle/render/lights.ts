/**
 * Short-lived lights the simulation makes (muzzle flashes, blasts, impacts), aged in ticks and drawn into the
 * renderer's light buffer, plus the steady glow of burning terrain cells. At night these are what you see by.
 */
import type { LightBuffer, Terrain } from "@/engine";

export class Flashes {
  readonly cap: number;
  n = 0;
  readonly x: Float32Array;
  readonly y: Float32Array;
  readonly r: Float32Array;
  readonly color: Uint32Array;
  readonly power: Float32Array;
  readonly life: Float32Array;
  readonly max: Float32Array;

  constructor(cap: number) {
    this.cap = cap;
    this.x = new Float32Array(cap);
    this.y = new Float32Array(cap);
    this.r = new Float32Array(cap);
    this.color = new Uint32Array(cap);
    this.power = new Float32Array(cap);
    this.life = new Float32Array(cap);
    this.max = new Float32Array(cap);
  }

  /** A light at (x, y): RGB colour, radius px, intensity (1 = daylight), lasting `life` seconds. */
  add(x: number, y: number, radius: number, color: number, intensity: number, life: number): void {
    let k = this.n;
    if (k >= this.cap) {
      // Full: replace the one closest to going out
      k = 0;
      for (let j = 1; j < this.n; j++) if (this.life[j] < this.life[k]) k = j;
    } else this.n++;
    this.x[k] = x;
    this.y[k] = y;
    this.r[k] = radius;
    this.color[k] = color >>> 0;
    this.power[k] = intensity;
    this.life[k] = this.max[k] = Math.max(1e-3, life);
  }

  step(dt: number): void {
    let k = 0;
    while (k < this.n) {
      this.life[k] -= dt;
      if (this.life[k] > 0) {
        k++;
        continue;
      }
      const last = --this.n;
      this.x[k] = this.x[last];
      this.y[k] = this.y[last];
      this.r[k] = this.r[last];
      this.color[k] = this.color[last];
      this.power[k] = this.power[last];
      this.life[k] = this.life[last];
      this.max[k] = this.max[last];
    }
  }

  /** Push the lights within `margin` of the view, fading with their remaining life; returns how many. */
  draw(out: LightBuffer, budget: number, left: number, top: number, right: number, bottom: number): number {
    let pushed = 0;
    for (let k = 0; k < this.n && pushed < budget; k++) {
      const r = this.r[k];
      const x = this.x[k],
        y = this.y[k];
      if (x + r < left || x - r > right || y + r < top || y - r > bottom) continue;
      out.push(x, y, r, this.color[k], this.power[k] * (this.life[k] / this.max[k]));
      pushed++;
    }
    return pushed;
  }

  shiftX(dx: number): void {
    for (let k = 0; k < this.n; k++) this.x[k] += dx;
  }
}

const burning: number[] = [];
/** Blocks already lit this frame (a small ring: cells arrive column by column, so repeats are recent). */
const seen = new Int32Array(32);

/**
 * A glow for burning terrain: cells are bucketed into 4×4-cell blocks and one flickering light is pushed per block
 * that burns, up to `budget`, so a forest fire costs a handful of lights rather than one per cell.
 */
export function drawBurning(
  out: LightBuffer,
  terrain: Terrain,
  budget: number,
  colFrom: number,
  colTo: number,
  time: number
): number {
  const n = terrain.collectBurning(colFrom, colTo, burning);
  if (n === 0 || budget <= 0) return 0;
  let pushed = 0;
  seen.fill(-1);
  search: for (let k = 0; k < n && pushed < budget; k++) {
    const c = burning[k * 2],
      r = burning[k * 2 + 1];
    const key = ((c >> 2) << 8) | (r >> 2);
    for (let s = 0; s < seen.length; s++) if (seen[s] === key) continue search;
    seen[pushed % seen.length] = key;
    const flicker = 0.75 + 0.25 * Math.sin(time * 13 + c * 1.7 + r);
    out.push(c * 8 + 4, r * 8 + 4, 34, 0xff8a30, 0.9 * flicker);
    pushed++;
  }
  return pushed;
}
