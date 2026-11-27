/**
 * CPU particles in typed arrays: debris, sparks, smoke, flames. Simulated in fixed ticks (gravity, drag, fade,
 * optional terrain collision) so they interpolate smoothly and stay deterministic on the "fx" stream; drawn by
 * pushing into the renderer's ParticleBuffer. The pool never grows past its budget: when it is full, new particles
 * are dropped, which keeps a huge blast from costing more than the cap.
 */
import { LEVEL_H, type ParticleBuffer, type Terrain } from "@/engine";

/* Particle behaviour bits */
/** Colours run from c0 to c1 over its life (a spark cooling), instead of staying c0. */
export const P_RAMP = 1;
/** Alpha fades out over the last half of its life. */
export const P_FADE = 2;
/** Stops dead on terrain. */
export const P_COLLIDE = 4;
/** Dies on terrain (rain, snow). */
export const P_POP = 8;

const lerpRgba = (a: number, b: number, t: number) => {
  const k = t < 0 ? 0 : t > 1 ? 1 : t;
  const r = ((a >>> 24) & 255) + ((((b >>> 24) & 255) - ((a >>> 24) & 255)) * k + 0.5);
  const g = ((a >>> 16) & 255) + ((((b >>> 16) & 255) - ((a >>> 16) & 255)) * k + 0.5);
  const bl = ((a >>> 8) & 255) + ((((b >>> 8) & 255) - ((a >>> 8) & 255)) * k + 0.5);
  const al = (a & 255) + (((b & 255) - (a & 255)) * k + 0.5);
  return (((r & 255) << 24) | ((g & 255) << 16) | ((bl & 255) << 8) | (al & 255)) >>> 0;
};

export class Particles {
  readonly cap: number;
  /** Live particles, packed in [0, n). */
  n = 0;
  /** Particles dropped because the pool was full (since the start). */
  dropped = 0;
  readonly x: Float32Array;
  readonly y: Float32Array;
  readonly px: Float32Array;
  readonly py: Float32Array;
  readonly vx: Float32Array;
  readonly vy: Float32Array;
  readonly life: Float32Array;
  readonly max: Float32Array;
  readonly grav: Float32Array;
  readonly drag: Float32Array;
  readonly c0: Uint32Array;
  readonly c1: Uint32Array;
  readonly size: Uint8Array;
  readonly flags: Uint8Array;
  readonly bits: Uint8Array;

  constructor(cap: number) {
    const n = (this.cap = Math.max(0, cap | 0));
    this.x = new Float32Array(n);
    this.y = new Float32Array(n);
    this.px = new Float32Array(n);
    this.py = new Float32Array(n);
    this.vx = new Float32Array(n);
    this.vy = new Float32Array(n);
    this.life = new Float32Array(n);
    this.max = new Float32Array(n);
    this.grav = new Float32Array(n);
    this.drag = new Float32Array(n);
    this.c0 = new Uint32Array(n);
    this.c1 = new Uint32Array(n);
    this.size = new Uint8Array(n);
    this.flags = new Uint8Array(n);
    this.bits = new Uint8Array(n);
  }

  get free(): number {
    return this.cap - this.n;
  }

  /** One particle; returns its slot, or -1 when the pool is full. Colours are RGBA. */
  spawn(
    x: number,
    y: number,
    vx: number,
    vy: number,
    life: number,
    size: number,
    c0: number,
    c1: number,
    grav: number,
    drag: number,
    flags: number,
    bits: number
  ): number {
    if (this.n >= this.cap || !(life > 0)) {
      this.dropped++;
      return -1;
    }
    const k = this.n++;
    this.x[k] = this.px[k] = x;
    this.y[k] = this.py[k] = y;
    this.vx[k] = vx;
    this.vy[k] = vy;
    this.life[k] = this.max[k] = life;
    this.size[k] = size;
    this.c0[k] = c0 >>> 0;
    this.c1[k] = c1 >>> 0;
    this.grav[k] = grav;
    this.drag[k] = drag;
    this.flags[k] = flags;
    this.bits[k] = bits;
    return k;
  }

  /** One tick: age, move, collide; dead particles are swapped out so the live ones stay packed. */
  step(dt: number, terrain: Terrain | null): void {
    const { x, y, px, py, vx, vy, life, grav, drag, bits } = this;
    let k = 0;
    while (k < this.n) {
      const left = life[k] - dt;
      if (left <= 0 || y[k] > LEVEL_H + 16) {
        this.remove(k);
        continue;
      }
      life[k] = left;
      px[k] = x[k];
      py[k] = y[k];
      let ux = vx[k],
        uy = vy[k] + grav[k] * dt;
      const d = drag[k];
      if (d > 0) {
        const f = Math.max(0, 1 - d * dt);
        ux *= f;
        uy *= f;
      }
      const nx = x[k] + ux * dt,
        ny = y[k] + uy * dt;
      if (bits[k] & (P_COLLIDE | P_POP) && terrain !== null && (ux !== 0 || uy !== 0) && terrain.solidAt(nx, ny)) {
        if (bits[k] & P_POP) {
          this.remove(k);
          continue;
        }
        // Settle where it struck: debris comes to rest on the ground instead of sinking into it
        ux = uy = 0;
      } else {
        x[k] = nx;
        y[k] = ny;
      }
      vx[k] = ux;
      vy[k] = uy;
      k++;
    }
  }

  private remove(k: number): void {
    const last = --this.n;
    if (k === last) return;
    this.x[k] = this.x[last];
    this.y[k] = this.y[last];
    this.px[k] = this.px[last];
    this.py[k] = this.py[last];
    this.vx[k] = this.vx[last];
    this.vy[k] = this.vy[last];
    this.life[k] = this.life[last];
    this.max[k] = this.max[last];
    this.grav[k] = this.grav[last];
    this.drag[k] = this.drag[last];
    this.c0[k] = this.c0[last];
    this.c1[k] = this.c1[last];
    this.size[k] = this.size[last];
    this.flags[k] = this.flags[last];
    this.bits[k] = this.bits[last];
  }

  /** Push every particle inside the view rectangle, interpolated `alpha` into the tick. */
  draw(out: ParticleBuffer, alpha: number, left: number, top: number, right: number, bottom: number): number {
    const { x, y, px, py, life, max, c0, c1, size, flags, bits } = this;
    let pushed = 0;
    for (let k = 0; k < this.n; k++) {
      const wx = px[k] + (x[k] - px[k]) * alpha;
      const wy = py[k] + (y[k] - py[k]) * alpha;
      if (wx < left || wx > right || wy < top || wy > bottom) continue;
      const u = 1 - life[k] / max[k];
      const b = bits[k];
      let c = b & P_RAMP ? lerpRgba(c0[k], c1[k], u) : c0[k];
      if (b & P_FADE && u > 0.5) {
        // Fade in four steps, so it reads as pixel art rather than a smooth gradient
        const a = c & 255;
        const steps = Math.ceil((1 - u) * 2 * 4) / 4;
        c = ((c & 0xffffff00) | Math.round(a * steps)) >>> 0;
      }
      out.push(wx, wy, size[k], c, flags[k]);
      pushed++;
    }
    return pushed;
  }

  shiftX(dx: number): void {
    for (let k = 0; k < this.n; k++) {
      this.x[k] += dx;
      this.px[k] += dx;
    }
  }

  clear(): void {
    this.n = 0;
  }
}
