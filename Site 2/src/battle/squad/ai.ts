/**
 * The squad's placeholder AI (B2 ports the full one): keep a formation slot off the advance point, shoot the
 * nearest enemy on screen in range and in sight, lob a grenade now and then, and get unstuck: back off from an
 * overhang and jump for its edge, blast a way through a wall, and drop back in if no progress comes.
 */
import { indexOf } from "@/engine";
import type { BattleGame } from "../game/BattleGame";
import { MODE_CHUTE, MODE_CLIMB, UNIT_DEAD } from "../game/components";
import { TAG_BOSS, TAG_FLYING } from "../game/defs";
import { hostileTo, TEAM_SQUAD } from "../game/teams";
import { OFFSETS, queueDrop, SQUAD } from "./squad";

const near: number[] = [];

/**
 * The best target for player index `i`: nearest by a distance that favours targets ahead and bosses, on screen,
 * and in sight unless the weapon lobs or the target flies (the old `findTarget`).
 */
export function findTarget(game: BattleGame, i: number, range: number, lobbed: boolean): number {
  const { pos, body } = game.core;
  const { unit } = game.c;
  const cam = game.camera;
  const x = pos.x[i],
    gy = pos.y[i] - 9;
  const n = game.world.spatial.inRadius(x, gy, range, hostileTo(TEAM_SQUAD), near);
  let best = 0,
    bestScore = range;
  for (let k = 0; k < n; k++) {
    const id = near[k];
    const j = indexOf(id);
    if (unit.$.sparse[j] < 0 || unit.flags[j] & UNIT_DEAD) continue;
    const ax = pos.x[j],
      ay = pos.y[j] - body.h[j] * 0.5;
    if (ax > cam.right - 6 || ax < cam.left + 4) continue;
    const tags = game.unitDef(id).tags ?? 0;
    const d = Math.hypot(ax - x, (ay - gy) * 0.6);
    const score = (ax < x - 10 ? d * 1.6 : d) * (tags & TAG_BOSS ? 0.7 : 1);
    if (score >= bestScore) continue;
    if (!lobbed && !(tags & TAG_FLYING) && !game.terrain.lineOfSight(x, gy, ax, ay)) continue;
    best = id;
    bestScore = score;
  }
  return best;
}

/** The old `grenade`: a lob towards x, 30–140 px out. */
export function throwGrenade(game: BattleGame, id: number, tx: number): void {
  const i = indexOf(id);
  const x = game.core.pos.x[i],
    y = game.core.pos.y[i];
  const dist = Math.max(30, Math.min(140, Math.abs(tx - x)));
  const dir = tx >= x ? 1 : -1;
  game.shoot("squad.grenade", x, y - 12, dir * dist * 1.05, -170, { owner: id });
}

export function playerThink(game: BattleGame, id: number, i: number, dt: number): void {
  const sq = game.world.res(SQUAD);
  const { player, intent, motor, unit, weapon, ai } = game.c;
  const { pos, vel } = game.core;
  const r = game.rng.ai;
  const p = player.id[i];
  const x = pos.x[i];
  const desired = sq.advance + OFFSETS[p] + Math.sin(game.now * 0.8 + p * 1.7) * 5;
  const dx = desired - x;
  let dir = Math.abs(dx) > 4 ? Math.sign(dx) : 0;
  if (ai.cover[i] > 0) {
    // Out from under the overhang, then a jump for its edge
    ai.cover[i] -= dt;
    dir = dx >= 0 ? -1 : 1;
    if (ai.cover[i] <= 0) intent.jump[i] = 1;
  }
  intent.move[i] = dir;
  const chute = motor.mode[i] === MODE_CHUTE;

  const wd = game.content.weapons.get(weapon.def[i]);
  if ((ai.scanT[i] -= dt) <= 0) {
    ai.scanT[i] = 0.15 + r.next() * 0.1;
    intent.target[i] = findTarget(game, i, wd.range, wd.lob !== undefined);
  }
  const tgt = intent.target[i];
  const live = tgt !== 0 && game.fighting(tgt);
  if (!live) intent.target[i] = 0;
  const tx = live ? pos.x[indexOf(tgt)] : x;
  unit.facing[i] = live ? (tx >= x ? 1 : -1) : dir !== 0 ? dir : unit.facing[i];
  intent.fire[i] = live && !chute ? 1 : 0;
  if (live && !chute && weapon.alt[i] <= 0 && Math.abs(tx - x) < 130) {
    throwGrenade(game, id, tx);
    weapon.alt[i] = 6 + r.next() * 6;
  }

  // Climbing into an overhang counts as stuck at once: back off and come at it again
  if (motor.mode[i] === MODE_CLIMB && vel.vy[i] === 0 && ai.cover[i] <= 0 && motor.stuckT[i] > 0.3) ai.cover[i] = 0.4;
  if (motor.stuckT[i] > 1.6) {
    // Blow a way through, or try the jump again
    if (r.chance(0.5)) throwGrenade(game, id, x + (dir || 1) * 24);
    else intent.jump[i] = 1;
    motor.stuckT[i] = 0;
  }
  // Well behind its place and making no progress: blast a way on, then drop back in
  if (dx < 40 || chute) {
    ai.home[i] = x;
    ai.t[i] = 0;
  } else if (x > ai.home[i] + 2) {
    ai.home[i] = x;
    ai.t[i] = 0;
  } else if ((ai.t[i] += dt) > 4 && ai.t[i] - dt <= 4) throwGrenade(game, id, x + 20);
  if (ai.t[i] > 9) queueDrop(game, id);
}
