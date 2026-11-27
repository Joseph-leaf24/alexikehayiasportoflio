/**
 * Placeholder Regime infantry (B3 ports the ~46 old kinds): the old mook, as a small state machine. It patrols
 * near where it was placed; once a squad member is on screen, in range and in sight it engages, closing to its
 * preferred range and holding it while it fires; hurt or under fire it ducks back into cover (a spot nearby the
 * target can't see) and pops out again; now and then it lobs a grenade, by preference into a cluster.
 */
import { indexOf, PARTICLE_EMISSIVE } from "@/engine";
import type { BattleGame } from "../game/BattleGame";
import { machine } from "../game/core";
import type { AiMachine, ProjectileDef, UnitDef, WeaponDef } from "../game/defs";
import { TAG_INFANTRY } from "../game/defs";
import { UNIT_DEAD } from "../game/components";
import { hostileTo, TEAM_REGIME } from "../game/teams";
import { troopArt } from "../art";
import { lobAt } from "../physics/weapons";
import { GRAVITY } from "../physics/locomotion";

/** The old mook's numbers: 15 px tall, 2 hp × 1.6, 34 px/s pushing in, 15 px/s on patrol, 150 px range. */
const RANGE = 150;
const PUSH = 34;
const PATROL = 15;

export const REGIME_RIFLE: WeaponDef = {
  name: "Regime rifle",
  range: RANGE,
  cadence: 0.55,
  pauseJitter: 0.6,
  spread: 0.18,
  lead: true,
  projectile: "regime.bullet",
  flash: { color: 0xff9a2a, radius: 22 },
  muzzleFx: "casing",
};

export const REGIME_BULLET: ProjectileDef = {
  speed: 170,
  life: 1.6,
  damage: 1,
  chip: 1,
  impact: "hit.spark",
  look: { colors: [0xff5a3aff, 0xff5a3aff, 0xb8321aff], spacing: 2, flags: PARTICLE_EMISSIVE },
};

export const REGIME_GRENADE: ProjectileDef = {
  speed: 150,
  life: 1.4,
  damage: 0,
  gravity: GRAVITY,
  bounce: 0.45,
  fuse: true,
  radius: 1,
  explode: { r: 20 },
  look: { colors: [0x4a5a30ff], size: 2, blink: 0xff3b3bff },
};

const near: number[] = [];
const lob = { vx: 0, vy: 0 };

/** The nearest squad member on screen, in range and in sight (the old `nearestFoe`), or 0. */
function scan(game: BattleGame, i: number): number {
  const { pos, body } = game.core;
  const x = pos.x[i],
    ey = pos.y[i] - 12;
  if (!game.camera.inViewX(x, 12)) return 0;
  const n = game.world.spatial.inRadius(x, pos.y[i] - 8, RANGE, hostileTo(game.core.team.id[i]), near);
  let best = 0,
    bestD = RANGE;
  for (let k = 0; k < n; k++) {
    const id = near[k];
    const j = indexOf(id);
    if (game.c.unit.$.sparse[j] < 0 || game.c.unit.flags[j] & UNIT_DEAD) continue;
    const d = Math.abs(pos.x[j] - x);
    if (d >= bestD || Math.abs(pos.y[j] - pos.y[i]) > 120) continue;
    if (!game.terrain.lineOfSight(x, ey, pos.x[j], pos.y[j] - body.h[j] * 0.6)) continue;
    best = id;
    bestD = d;
  }
  return best;
}

/** Re-scan at about 7 Hz; returns the live target or 0. */
function target(game: BattleGame, i: number, dt: number): number {
  const { ai, intent } = game.c;
  if ((ai.scanT[i] -= dt) <= 0) {
    ai.scanT[i] = 0.15;
    const t = scan(game, i);
    if (t && !ai.alert[i]) {
      ai.alert[i] = 1;
      game.popup("!", game.core.pos.x[i], game.core.pos.y[i] - 22, 0xff3b3bff);
    }
    intent.target[i] = t;
  }
  const t = intent.target[i];
  if (t !== 0 && !game.fighting(t)) intent.target[i] = 0;
  return intent.target[i];
}

/** Lob a grenade at the target now and then, by preference at two or more squad members together. */
function grenade(game: BattleGame, id: number, i: number, tgt: number): void {
  const { weapon } = game.c;
  const { pos } = game.core;
  if (weapon.alt[i] > 0) return;
  const j = indexOf(tgt);
  const r = game.rng.ai;
  weapon.alt[i] = 5 + r.next() * 4;
  if (Math.abs(pos.x[j] - pos.x[i]) > 170) return;
  const cluster = game.world.spatial.inRadius(pos.x[j], pos.y[j] - 8, 40, hostileTo(TEAM_REGIME), near);
  if (cluster < 2 && !r.chance(0.35)) return;
  const v = lob;
  lobAt(pos.x[i], pos.y[i] - 12, pos.x[j] + (r.next() - 0.5) * 12, pos.y[j] - 2, GRAVITY, 150, v);
  game.shoot("regime.grenade", pos.x[i], pos.y[i] - 12, v.vx, v.vy, { owner: id });
}

/** Ground ahead to walk on and no wall: patrols turn back at edges instead of walking off them. */
function walkable(game: BattleGame, x: number, y: number, dir: number): boolean {
  const ahead = x + dir * 5;
  return !game.terrain.solidAt(ahead, y - 4) && game.terrain.solidAt(ahead, y + 2);
}

/** A spot within reach, away from the target, where the target can't see this trooper's head; or NaN. */
function coverSpot(game: BattleGame, i: number, tgt: number): number {
  const { pos } = game.core;
  const j = indexOf(tgt);
  const tx = pos.x[j],
    ty = pos.y[j] - 9;
  const away = pos.x[i] >= tx ? 1 : -1;
  for (let d = 8; d <= 56; d += 8) {
    const x = pos.x[i] + away * d;
    const y = game.terrain.surfaceY(x, pos.y[i] - 24);
    if (Math.abs(y - pos.y[i]) > 16) continue;
    if (!game.terrain.lineOfSight(tx, ty, x, y - 12)) return x;
  }
  return NaN;
}

export const INFANTRY_AI: AiMachine = {
  initial: "patrol",
  states: {
    patrol: {
      enter(game, _id, i) {
        game.c.intent.fire[i] = 0;
      },
      run(game, _id, i, dt) {
        const { intent, unit, ai } = game.c;
        const { pos } = game.core;
        if (target(game, i, dt)) return "engage";
        // A few steps either side of home, turning at walls and edges
        const f = unit.facing[i];
        if (Math.abs(pos.x[i] + f * 5 - ai.home[i]) > 18 || !walkable(game, pos.x[i], pos.y[i], f)) unit.facing[i] = -f;
        intent.move[i] = (unit.facing[i] * PATROL) / PUSH;
      },
    },
    engage: {
      run(game, id, i, dt) {
        const { intent, unit, ai } = game.c;
        const { pos, health } = game.core;
        const tgt = target(game, i, dt);
        if (!tgt) return "patrol";
        const j = indexOf(tgt);
        const dx = pos.x[j] - pos.x[i];
        const face = dx >= 0 ? 1 : -1;
        unit.facing[i] = face;
        intent.fire[i] = 1;
        grenade(game, id, i, tgt);
        const gap = Math.abs(dx);
        const want = RANGE * 0.55;
        let move = 0;
        if (gap > want + 16) move = face;
        else if (gap < Math.min(28, want * 0.5)) move = -face * 0.6;
        intent.move[i] = move !== 0 && walkable(game, pos.x[i], pos.y[i], Math.sign(move)) ? move : 0;
        // Hurt badly, or just hit: duck back into cover
        const hurt = health.hp[i] < health.max[i] * 0.4;
        if ((hurt || unit.hitT[i] > 0) && ai.cover[i] <= 0 && game.rng.ai.chance(hurt ? 0.3 : 0.15)) return "cover";
        ai.cover[i] -= dt;
      },
    },
    cover: {
      enter(game, _id, i) {
        const tgt = game.c.intent.target[i];
        const spot = tgt ? coverSpot(game, i, tgt) : NaN;
        // Where it's heading; with no cover in reach it just falls back a little way
        game.c.ai.home[i] = Number.isNaN(spot) ? game.core.pos.x[i] - game.c.unit.facing[i] * 24 : spot;
        game.c.intent.fire[i] = 0;
      },
      run(game, _id, i) {
        const { intent, ai } = game.c;
        const x = game.core.pos.x[i];
        const dx = ai.home[i] - x;
        intent.move[i] = Math.abs(dx) > 3 ? Math.sign(dx) : 0;
        if ((intent.move[i] === 0 && ai.t[i] > 1.2) || ai.t[i] > 2.4) {
          ai.cover[i] = 3;
          ai.home[i] = x;
          return "engage";
        }
      },
    },
  },
};

/** Compiled once: every trooper shares it. */
const THINK = machine(INFANTRY_AI);

export const TROOPER: UnitDef = {
  name: "Trooper",
  hw: 3,
  h: 15,
  hp: 2 * 1.6,
  team: TEAM_REGIME,
  score: 100,
  art: troopArt("jungle", "mook"),
  weapon: "regime.rifle",
  ai: { think: THINK, every: 3 },
  move: { speed: PUSH, jump: 230 },
  tags: TAG_INFANTRY,
  blastKills: true,
  deathFx: "death.trooper",
  corpse: 1.4,
};
