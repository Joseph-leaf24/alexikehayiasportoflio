/**
 * Beams, lightning and muzzle flashes: the old scene's `beams`, `zaps` and `flashes`. The simulation adds them
 * (`beam`, `zap`, `muzzle`) and they age in ticks; the beams pass draws them as emissive pixels along each segment,
 * as the old scene did with "lighter" fills: a soft glow, the beam in its colour, a white core, and a flare where a
 * heavy beam strikes. Lightning is a jagged white-blue line, re-forked 30 times a second.
 *
 * System: fx.beams (effects). Pass: beams (350).
 */
import { hash2, PARTICLE_ADDITIVE, PARTICLE_EMISSIVE, resource, type ParticleBuffer } from "@/engine";
import { PLAYER_COLORS } from "@/data/squad";
import type { BattleGame } from "../game/BattleGame";
import type { RenderPass } from "./extract";

const BEAM = 0;
const ZAP = 1;
const MUZZLE = 2;

const GLOW = PARTICLE_EMISSIVE | PARTICLE_ADDITIVE;
const WHITE = 0xffffffff;
const ZAP_COLOR = 0xc8e8ffff;
const FLASH_CORE = 0xfff6c8ff;

/** The squad's colours as RGB, for beams and flashes in a player's colour. */
export const PLAYER_RGB: readonly number[] = PLAYER_COLORS.map((c) => parseInt(c.slice(1), 16));

/** RGB plus an alpha from 0 to 1, as RGBA. */
const rgba = (rgb: number, a: number) => ((rgb << 8) | Math.round(a * 255)) >>> 0;

export class Beams {
  readonly cap: number;
  n = 0;
  readonly kind: Uint8Array;
  readonly x0: Float32Array;
  readonly y0: Float32Array;
  readonly x1: Float32Array;
  readonly y1: Float32Array;
  readonly color: Uint32Array;
  readonly width: Uint8Array;
  readonly life: Float32Array;

  constructor(cap = 128) {
    this.cap = cap;
    this.kind = new Uint8Array(cap);
    this.x0 = new Float32Array(cap);
    this.y0 = new Float32Array(cap);
    this.x1 = new Float32Array(cap);
    this.y1 = new Float32Array(cap);
    this.color = new Uint32Array(cap);
    this.width = new Uint8Array(cap);
    this.life = new Float32Array(cap);
  }

  add(kind: number, x0: number, y0: number, x1: number, y1: number, color: number, width: number, life: number): void {
    let k = this.n;
    if (k >= this.cap) {
      // Full: replace the one closest to going out
      k = 0;
      for (let j = 1; j < this.n; j++) if (this.life[j] < this.life[k]) k = j;
    } else this.n++;
    this.kind[k] = kind;
    this.x0[k] = x0;
    this.y0[k] = y0;
    this.x1[k] = x1;
    this.y1[k] = y1;
    this.color[k] = color >>> 0;
    this.width[k] = width;
    this.life[k] = life;
  }

  step(dt: number): void {
    let k = 0;
    while (k < this.n) {
      if ((this.life[k] -= dt) > 0) {
        k++;
        continue;
      }
      const last = --this.n;
      this.kind[k] = this.kind[last];
      this.x0[k] = this.x0[last];
      this.y0[k] = this.y0[last];
      this.x1[k] = this.x1[last];
      this.y1[k] = this.y1[last];
      this.color[k] = this.color[last];
      this.width[k] = this.width[last];
      this.life[k] = this.life[last];
    }
  }

  shiftX(dx: number): void {
    for (let k = 0; k < this.n; k++) {
      this.x0[k] += dx;
      this.x1[k] += dx;
    }
  }
}

export const BEAMS = resource<Beams>("beams");

/** A beam from (x0, y0) to (x1, y1) in an RGB colour, `w` px thick (1–3), shown for `life` seconds. */
export function beam(
  game: BattleGame,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  color: number,
  w: number,
  life: number
): void {
  game.world.tryRes(BEAMS)?.add(BEAM, x0, y0, x1, y1, color, w, life);
}

/** One leg of a lightning chain. */
export function zap(game: BattleGame, x0: number, y0: number, x1: number, y1: number, life: number): void {
  game.world.tryRes(BEAMS)?.add(ZAP, x0, y0, x1, y1, 0, 1, life);
}

/** A muzzle flash: a small bright cross in an RGB colour. */
export function muzzle(game: BattleGame, x: number, y: number, color: number, life = 0.05): void {
  game.world.tryRes(BEAMS)?.add(MUZZLE, x, y, x, y, color, 1, life);
}

/** Squares of side `s` centred along a segment, one every `s` px of its longer axis: a band `s` px thick. */
function band(out: ParticleBuffer, x0: number, y0: number, x1: number, y1: number, s: number, color: number): void {
  const dx = x1 - x0,
    dy = y1 - y0;
  const m = Math.max(Math.abs(dx), Math.abs(dy));
  const n = Math.max(1, Math.ceil(m / s));
  const h = s >> 1;
  for (let k = 0; k < n; k++) {
    const t = m > 0 ? (k * s) / m : 0;
    out.push(x0 + dx * t - h, y0 + dy * t - h, s, color, GLOW);
  }
}

/** Glow, coloured beam and white core; heavy beams (w > 1) are wider, flicker and flare where they strike. */
function drawBeam(out: ParticleBuffer, b: Beams, k: number, time: number): void {
  const x0 = b.x0[k],
    y0 = b.y0[k],
    x1 = b.x1[k],
    y1 = b.y1[k];
  const w = b.width[k];
  const rgb = b.color[k];
  const fl = w > 1 && Math.sin(time * 50 + x0) > 0 ? 1 : 0;
  band(out, x0, y0, x1, y1, w * 2 + 3, rgba(rgb, 0.3));
  // A 1 px beam is all core: white over its colour, added, is white
  if (w > 1) band(out, x0, y0, x1, y1, w + fl * 2, rgba(rgb, 0.8));
  band(out, x0, y0, x1, y1, w > 2 ? 2 : 1, WHITE);
  if (w > 1) {
    const ex = Math.round(x1),
      ey = Math.round(y1);
    out.push(ex - w - 1, ey - w - 1, w * 2 + 3, rgba(rgb, 0.27), GLOW);
    out.push(ex - w, ey - w, w * 2 + 1, rgba(rgb, 0.9), GLOW);
    out.push(ex - w + 1, ey - w + 1, w * 2 - 1, WHITE, GLOW);
  }
}

/** A lightning leg: five steps, each a fifth of the way left, the first four thrown up to 5 px aside. */
function drawZap(out: ParticleBuffer, b: Beams, k: number, time: number): void {
  const x1 = b.x1[k],
    y1 = b.y1[k];
  let ax = b.x0[k],
    ay = b.y0[k];
  const seed = Math.floor(time * 30) * 977 + Math.floor(ax);
  for (let s = 1; s <= 5; s++) {
    const off = s < 5 ? 10 : 0;
    const nx = ax + (x1 - ax) / (6 - s) + (hash2(seed, s * 2) - 0.5) * off;
    const ny = ay + (y1 - ay) / (6 - s) + (hash2(seed, s * 2 + 1) - 0.5) * off;
    band(out, ax, ay, nx, ny, 1, ZAP_COLOR);
    ax = nx;
    ay = ny;
  }
}

function drawMuzzle(out: ParticleBuffer, b: Beams, k: number): void {
  const x = Math.round(b.x0[k]),
    y = Math.round(b.y0[k]);
  const c = rgba(b.color[k], 1);
  out.push(x - 1, y - 1, 3, FLASH_CORE, PARTICLE_EMISSIVE);
  for (let d = -2; d <= 2; d++) {
    out.push(x + d, y, 1, c, PARTICLE_EMISSIVE);
    if (d !== 0) out.push(x, y + d, 1, c, PARTICLE_EMISSIVE);
  }
}

export function beamsPass(): RenderPass {
  return {
    name: "beams",
    order: 350,
    draw(g, f, v) {
      const b = g.world.tryRes(BEAMS);
      if (!b) return;
      for (let k = 0; k < b.n; k++) {
        const x0 = b.x0[k],
          x1 = b.x1[k],
          y0 = b.y0[k],
          y1 = b.y1[k];
        if (Math.max(x0, x1) < v.left - 8 || Math.min(x0, x1) > v.right + 8) continue;
        if (Math.max(y0, y1) < v.top - 8 || Math.min(y0, y1) > v.bottom + 8) continue;
        const kind = b.kind[k];
        if (kind === BEAM) drawBeam(f.particles, b, k, v.time);
        else if (kind === ZAP) drawZap(f.particles, b, k, v.time);
        else drawMuzzle(f.particles, b, k);
      }
    },
  };
}
