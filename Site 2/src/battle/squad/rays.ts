/**
 * The squad's beam weapons, the old `fire()` cases that strike at once (`components/pixel/game.ts`), with the old
 * numbers: the laser (burns through every body up to the first wall), the tesla gun (lightning jumping from enemy to
 * enemy), the prism laser (splits into three where it strikes, `beamTo`) and the death ray (a held beam that swings
 * onto its target and cuts through cover and ground, `deathRay`). Each fires through `hitscan` and is drawn by
 * `render/beams.ts`, in the player's colour.
 *
 * System: squad.rays (weapons), the death rays being held.
 */
import { indexOf, PARTICLE_ADDITIVE, PARTICLE_EMISSIVE, resource, type System } from "@/engine";
import type { BattleGame } from "../game/BattleGame";
import type { Content } from "../game/content";
import type { EffectDef, FireFn, WeaponDef } from "../game/defs";
import { hostileTo } from "../game/teams";
import { playerOf } from "../physics/damage";
import { hitscan } from "../physics/hitscan";
import { beam, PLAYER_RGB, zap } from "../render/beams";

/** Beams burn through every body in their way. */
const ALL = 255;
const TAU = Math.PI * 2;
/** A beam's colour when it isn't a player's. */
const OTHER_RGB = 0xff5a6a;
/** The prism's split beams. */
const RAINBOW = [0xff5a6a, 0xffd23f, 0x5aff8a, 0x5ab0ff, 0xc87aff];
const opaque = (rgb: number) => ((rgb << 8) | 0xff) >>> 0;
const RAINBOW_RGBA = RAINBOW.map(opaque);
/** Glint colours for the death ray's end: white, or the player's. */
const WHITE_GLINT = [0xffffffff];
const PLAYER_GLINT = PLAYER_RGB.map((c) => [opaque(c)]);

const EFFECTS: Readonly<Record<string, EffectDef>> = {
  /** Twinkling glints (the old GLINT): the prism's rainbow, the death ray's end; colours given per burst. */
  "squad.glint": {
    count: 1,
    speed: [20, 60],
    angle: -Math.PI / 2,
    spread: 2.4,
    life: [0.35, 0.5],
    gravity: 34,
    drag: 2.2,
    size: [1, 2],
    colors: WHITE_GLINT,
    flags: PARTICLE_EMISSIVE | PARTICLE_ADDITIVE,
  },
};

const rgbOf = (game: BattleGame, id: number) => {
  const p = playerOf(game, id);
  return p >= 0 ? PLAYER_RGB[p] : OTHER_RGB;
};
const teamOf = (game: BattleGame, id: number) => game.core.team.id[indexOf(id)];

/** The laser: everyone along it for 230 px or to the first wall takes 2.2. */
const laser: FireFn = (game, shooter, gx, gy, ax, ay) => {
  const a = Math.atan2(ay - gy, ax - gx);
  const h = hitscan(game, shooter, teamOf(game, shooter), gx, gy, gx + Math.cos(a) * 230, gy + Math.sin(a) * 230, {
    damage: 2.2,
    pierce: ALL,
    width: 6,
  });
  beam(game, gx, gy, h.x, h.y, rgbOf(game, shooter), 1, 0.1);
  game.effect("squad.spark", h.x, h.y);
};

const chain: number[] = [];
const near: number[] = [];

/** The nearest live enemy of `team` within `r` of (x, y) by body centre, not in `skip` and not `not`; or 0. */
function nearestOther(game: BattleGame, team: number, x: number, y: number, r: number, skip: number[], not = 0) {
  const { pos, body } = game.core;
  const n = game.world.spatial.inRadius(x, y, r, hostileTo(team), near);
  let best = 0,
    bd = r;
  for (let k = 0; k < n; k++) {
    const id = near[k];
    if (id === not || skip.includes(id) || !game.fighting(id)) continue;
    const j = indexOf(id);
    const d = Math.hypot(pos.x[j] - x, pos.y[j] - body.h[j] / 2 - y);
    if (d < bd) {
      bd = d;
      best = id;
    }
  }
  return best;
}

/** The tesla gun: lightning to its target, then on to the nearest enemy within 80 px, four in all, 1.6 each. */
const tesla: FireFn = (game, shooter, gx, gy, ax, ay) => {
  const { pos, body } = game.core;
  const team = teamOf(game, shooter);
  let cur = game.c.intent.target[indexOf(shooter)];
  if (!game.fighting(cur)) cur = 0;
  chain.length = 0;
  let x = gx,
    y = gy;
  for (let k = 0; k < 4 && cur !== 0; k++) {
    const j = indexOf(cur);
    const qx = pos.x[j],
      qy = pos.y[j] - body.h[j] / 2;
    zap(game, x, y, qx, qy, 0.14);
    chain.push(cur);
    game.hurt(cur, 1.6, shooter);
    x = qx;
    y = qy;
    cur = nearestOther(game, team, qx, qy, 80, chain);
  }
  // Nothing to strike: the bolt still crackles out to the aim point
  if (chain.length === 0) zap(game, gx, gy, ax, ay, 0.14);
};

/**
 * The prism laser: a 2 px beam (2.6 to everyone on it) that splits where it strikes its first body, or at its end,
 * into three thin rainbow beams to the nearest others within 130 px (1.7 each).
 */
const prism: FireFn = (game, shooter, gx, gy, ax, ay) => {
  const { pos, body } = game.core;
  const team = teamOf(game, shooter);
  const a = Math.atan2(ay - gy, ax - gx);
  const h = hitscan(game, shooter, team, gx, gy, gx + Math.cos(a) * 210, gy + Math.sin(a) * 210, {
    damage: 2.6,
    pierce: ALL,
    width: 6,
  });
  beam(game, gx, gy, h.x, h.y, rgbOf(game, shooter), 2, 0.12);
  game.effect("squad.spark", h.x, h.y);
  const first = h.first;
  let hx = h.x,
    hy = h.y;
  if (first !== 0) {
    const j = indexOf(first);
    hx = pos.x[j];
    hy = pos.y[j] - body.h[j] / 2;
  }
  const shift = Math.max(0, playerOf(game, shooter));
  chain.length = 0;
  for (let s = 0; s < 3; s++) {
    const id = nearestOther(game, team, hx, hy, 130, chain, first);
    if (id === 0) break;
    chain.push(id);
  }
  for (let s = 0; s < chain.length; s++) {
    const j = indexOf(chain[s]);
    const qx = pos.x[j],
      qy = pos.y[j] - body.h[j] / 2;
    game.hurt(chain[s], 1.7, shooter);
    beam(game, hx, hy, qx, qy, RAINBOW[(s + shift) % RAINBOW.length], 1, 0.14);
    game.effect("squad.glint", qx, qy, { count: 3, colors: RAINBOW_RGBA });
  }
  game.effect("squad.glint", hx, hy, { count: 6, colors: RAINBOW_RGBA, speed: 1.4 });
};

/** A death ray being held: its shooter, seconds left, its angle, and the time since it last cut the block it hits. */
interface Ray {
  shooter: number;
  t: number;
  a: number;
  cut: number;
}

const RAYS = resource<Ray[]>("squad.rays");

/** The death ray: held for 0.9 s (the `squad.rays` system burns), then 1.3 s to recharge. */
const deathRay: FireFn = (game, shooter, gx, gy, ax, ay) => {
  const rays = game.world.res(RAYS);
  const a = Math.atan2(ay - gy, ax - gx);
  const held = rays.find((r) => r.shooter === shooter);
  if (held) {
    held.t = 0.9;
    held.a = a;
  } else rays.push({ shooter, t: 0.9, a, cut: 0 });
};

/**
 * One tick of a held ray: swing towards the target at up to 2.4 rad/s, 10 a second to everyone along it for 280 px
 * or to the first wall, and the block it strikes cut by 2 every 0.06 s; glints and flames where it ends.
 */
function burn(game: BattleGame, r: Ray, dt: number): void {
  const { pos, body } = game.core;
  const { unit, intent } = game.c;
  const i = indexOf(r.shooter);
  const gx = pos.x[i] + unit.facing[i] * 7,
    gy = pos.y[i] - Math.min(9, body.h[i] * 0.6);
  const tgt = intent.target[i];
  if (game.fighting(tgt)) {
    const j = indexOf(tgt);
    let d = Math.atan2(pos.y[j] - body.h[j] / 2 - gy, pos.x[j] - gx) - r.a;
    d -= Math.round(d / TAU) * TAU;
    r.a += Math.max(-2.4 * dt, Math.min(2.4 * dt, d));
  }
  const cut = r.cut + dt > 0.06;
  const ux = Math.cos(r.a),
    uy = Math.sin(r.a);
  const h = hitscan(game, r.shooter, teamOf(game, r.shooter), gx, gy, gx + ux * 280, gy + uy * 280, {
    damage: 10 * dt,
    pierce: ALL,
    width: 8,
    chip: cut ? 2 : 0,
  });
  if (h.wall) r.cut = cut ? 0 : r.cut + dt;
  beam(game, gx, gy, h.x, h.y, rgbOf(game, r.shooter), 3, 0.05);
  // Twice the old per-frame odds' worth, at 30 ticks a second
  const fx = game.rng.fx;
  if (fx.chance(0.6)) {
    const p = playerOf(game, r.shooter);
    game.effect("squad.glint", h.x, h.y, {
      count: 2,
      colors: p >= 0 && fx.chance(0.5) ? PLAYER_GLINT[p] : WHITE_GLINT,
    });
  }
  if (fx.chance(0.4)) game.effect("flame", h.x, h.y, { count: 2 });
}

function raySystem(game: BattleGame): System {
  return {
    name: "squad.rays",
    phase: "weapons",
    run(w, dt) {
      const rays = w.res(RAYS);
      for (let k = rays.length - 1; k >= 0; k--) {
        const r = rays[k];
        if ((r.t -= dt) < 0 || !game.fighting(r.shooter)) rays.splice(k, 1);
        else burn(game, r, dt);
      }
    },
  };
}

const RAY_GUNS: Readonly<Record<string, WeaponDef>> = {
  "squad.laser": { name: "Laser", range: 230, cadence: 0.42, fire: laser },
  "squad.tesla": { name: "Tesla", range: 140, cadence: 0.38, fire: tesla },
  "squad.deathray": { name: "Death ray", range: 280, cadence: 2.2, fire: deathRay },
  "squad.prism": { name: "Prism laser", range: 210, cadence: 0.55, fire: prism },
};

export function registerRays(content: Content): void {
  for (const [id, def] of Object.entries(EFFECTS)) content.effects.register(id, def);
  for (const [id, def] of Object.entries(RAY_GUNS)) content.weapons.register(id, def);
}

/** The death rays' state and the system that holds them. */
export function installRays(game: BattleGame): void {
  game.world.setRes(RAYS, []);
  game.world.addSystem(raySystem(game));
}
