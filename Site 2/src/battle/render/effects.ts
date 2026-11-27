/**
 * Effect emission: an `EffectDef` turned into particles (and a flash of light) on the cosmetic "fx" stream, so
 * effects never shift the gameplay random sequence. Debris from a terrain carve uses the carve's own colours and
 * sample points, so a brick wall throws brick-red chips and a hill throws dirt.
 */
import type { CarveResult } from "@/engine";
import type { BattleGame } from "../game/BattleGame";
import type { EffectDef } from "../game/defs";
import { P_COLLIDE, P_FADE, P_RAMP } from "./particles";

export interface EffectOptions {
  /** Multiplies the def's particle count. */
  count?: number;
  /** Overrides the launch direction (radians) and cone. */
  angle?: number;
  spread?: number;
  /** Multiplies launch speeds. */
  speed?: number;
  /** Velocity added to every particle (inherit a shooter's or a projectile's motion). */
  vx?: number;
  vy?: number;
  /** Replace the def's colours (RGBA). */
  colors?: readonly number[];
}

const TAU = Math.PI * 2;

/** Resolve an effect by id or index (0 when unknown, which emits nothing). */
export function effectIndex(game: BattleGame, def: string | number): number {
  return typeof def === "number" ? def : game.content.effects.index(def);
}

export function emitEffect(game: BattleGame, def: string | number, x: number, y: number, o?: EffectOptions): void {
  const k = effectIndex(game, def);
  if (k === 0) return;
  emitDef(game, game.content.effects.get(k), x, y, o);
}

export function emitDef(game: BattleGame, d: EffectDef, x: number, y: number, o?: EffectOptions): void {
  const r = game.rng.fx;
  const p = game.particles;
  const base = typeof d.count === "number" ? d.count : r.int(d.count[0], d.count[1] + 1);
  const count = Math.round(base * (o?.count ?? 1));
  const colors = o?.colors ?? d.colors;
  const angle = o?.angle ?? d.angle ?? 0;
  const spread = o?.spread ?? d.spread ?? TAU;
  const sm = o?.speed ?? 1;
  const vx0 = o?.vx ?? 0;
  const vy0 = o?.vy ?? 0;
  const bits = (d.ramp ? P_RAMP : 0) | (d.fade ? P_FADE : 0) | (d.collide ? P_COLLIDE : 0);
  const s0 = d.size?.[0] ?? 1;
  const s1 = d.size?.[1] ?? s0;
  const jitter = d.jitter ?? 0;
  for (let n = 0; n < count && p.free > 0; n++) {
    const a = angle + (r.next() - 0.5) * spread;
    const s = r.range(d.speed[0], d.speed[1]) * sm;
    const c0 = d.ramp ? colors[0] : colors[r.int(colors.length)];
    const c1 = d.ramp ? colors[colors.length - 1] : c0;
    p.spawn(
      x + (jitter > 0 ? (r.next() - 0.5) * 2 * jitter : 0),
      y + (jitter > 0 ? (r.next() - 0.5) * 2 * jitter : 0),
      vx0 + Math.cos(a) * s,
      vy0 + Math.sin(a) * s - (d.lift ?? 0),
      r.range(d.life[0], d.life[1]),
      s0 === s1 ? s0 : r.int(s0, s1 + 1),
      c0,
      c1,
      d.gravity ?? 0,
      d.drag ?? 0,
      d.flags ?? 0,
      bits
    );
  }
  if (d.light) game.flashes.add(x, y, d.light.radius, d.light.color, d.light.intensity, d.light.life);
}

/** Reused colour list for debris, so a blast allocates nothing. */
const debrisColors: number[] = [];

/** Chips of the destroyed cells, flung from the carve's sample points away from the blast centre. */
export function emitDebris(game: BattleGame, carve: CarveResult, x: number, y: number, force: number): void {
  if (carve.destroyed === 0 || carve.kinds === 0) return;
  const r = game.rng.fx;
  const p = game.particles;
  debrisColors.length = 0;
  for (let k = 0; k < carve.kinds; k++) debrisColors.push(((carve.colors[k] << 8) | 0xff) >>> 0);
  // Fewer chips per cell for big blasts, so the pool isn't spent on one crater
  const per = carve.destroyed > 60 ? 1 : carve.destroyed > 20 ? 2 : 3;
  for (let s = 0; s < carve.samples && p.free > 0; s++) {
    const sx = carve.sx[s],
      sy = carve.sy[s];
    const away = Math.atan2(sy - y, sx - x);
    for (let n = 0; n < per; n++) {
      const a = away + (r.next() - 0.5) * 1.6;
      const v = (40 + r.next() * 120) * force;
      const c = debrisColors[r.int(debrisColors.length)];
      p.spawn(
        sx,
        sy,
        Math.cos(a) * v,
        Math.sin(a) * v - 60 * force,
        0.8 + r.next() * 0.9,
        r.chance(0.3) ? 2 : 1,
        c,
        c,
        520,
        0.4,
        0,
        P_COLLIDE
      );
    }
  }
}
