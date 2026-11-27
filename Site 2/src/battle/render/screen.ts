/**
 * Full-screen effects the simulation drives: shockwave rings, the white flash of a big blast, the end-of-level
 * white-out, letterbox bars and a chromatic punch. Strengths ease in ticks; `apply` copies them into the frame.
 */
import type { RenderFrame, Shockwave } from "@/engine";

/** The renderer takes at most this many rings. */
export const MAX_SHOCKWAVES = 16;

export class ScreenFx {
  /** Additive white, 0..1, decaying (big blasts). */
  flash = 0;
  /** White-out, 0..1, and the level it eases to. */
  white = 0;
  whiteTarget = 0;
  letterbox = 0;
  letterboxTarget = 0;
  chroma = 0;
  vignette = 0.3;
  bloom = 0.7;
  private readonly waves: Shockwave[] = [];
  private readonly age = new Float32Array(MAX_SHOCKWAVES).fill(-1);
  private readonly size = new Float32Array(MAX_SHOCKWAVES);

  constructor() {
    for (let k = 0; k < MAX_SHOCKWAVES; k++) this.waves.push({ x: 0, y: 0, radius: 0, width: 0, strength: 0 });
  }

  /** A ring of distortion from (x, y); `size` scales its reach and push (1 = a grenade). */
  shockwave(x: number, y: number, size = 1): void {
    let slot = 0;
    for (let k = 0; k < MAX_SHOCKWAVES; k++) {
      if (this.age[k] < 0) {
        slot = k;
        break;
      }
      if (this.age[k] > this.age[slot]) slot = k;
    }
    const w = this.waves[slot];
    w.x = x;
    w.y = y;
    w.radius = 0;
    w.strength = 0;
    this.age[slot] = 0;
    this.size[slot] = size;
  }

  /** Live rings. */
  get rings(): number {
    let n = 0;
    for (let k = 0; k < MAX_SHOCKWAVES; k++) if (this.age[k] >= 0) n++;
    return n;
  }

  step(dt: number): void {
    this.flash = Math.max(0, this.flash - dt * 1.6);
    this.chroma = Math.max(0, this.chroma - dt * 2.5);
    this.white += (this.whiteTarget - this.white) * Math.min(1, dt * 3);
    this.letterbox += (this.letterboxTarget - this.letterbox) * Math.min(1, dt * 4);
    for (let k = 0; k < MAX_SHOCKWAVES; k++) {
      if (this.age[k] < 0) continue;
      const s = this.size[k];
      const life = 0.45 + 0.35 * Math.min(2, s);
      const u = (this.age[k] += dt) / life;
      const w = this.waves[k];
      if (u >= 1) {
        this.age[k] = -1;
        w.strength = 0;
        continue;
      }
      w.radius = 6 + u * 120 * s;
      w.width = 10 + u * 10 * s;
      w.strength = 5 * Math.min(2, s) * (1 - u) * (1 - u);
    }
  }

  /** Copy into the frame's effects. */
  apply(f: RenderFrame): void {
    const fx = f.effects;
    fx.shockwaves = this.waves;
    fx.flash = this.flash;
    fx.white = this.white;
    fx.letterbox = this.letterbox;
    fx.chroma = this.chroma;
    fx.vignette = this.vignette;
    fx.bloom = this.bloom;
  }

  shiftX(dx: number): void {
    for (const w of this.waves) w.x += dx;
  }
}
