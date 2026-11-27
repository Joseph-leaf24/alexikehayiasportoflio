/**
 * The squad's AI (the old `updatePlayer`): keep a formation slot off the advance point, shoot the nearest enemy on
 * screen in range and in sight (or a barrel with enemies next to it), lob a grenade now and then, take a big climb on
 * the jetpack, and get unstuck: back off from an overhang and jump for its edge, fly over or blast through a wall,
 * and drop back in if no progress comes.
 */
import { BODY_GROUNDED, indexOf } from "@/engine";
import type { BattleGame } from "../game/BattleGame";
import { MODE_CHUTE, MODE_CLIMB, MODE_FLY, MODE_WALK, UNIT_DEAD } from "../game/components";
import { TAG_BOSS, TAG_FLYING, TAG_PROP } from "../game/defs";
import { hostileTo, TEAM_SQUAD } from "../game/teams";
import { OFFSETS, queueDrop, SQUAD } from "./squad";

const near: number[] = [];
const props: number[] = [];

/** An enemy (not a boss) close enough to a barrel at (x, y) to go up with it. */
function enemyBeside(game: BattleGame, x: number, y: number): boolean {
  const n = game.world.spatial.rect(x - 30, y - 18, x + 30, y + 18, hostileTo(TEAM_SQUAD), near);
  for (let k = 0; k < n; k++) {
    const id = near[k];
    if (game.fighting(id) && !((game.unitDef(id).tags ?? 0) & (TAG_PROP | TAG_BOSS))) return true;
  }
  return false;
}

/**
 * The best target for player index `i`: nearest by a distance that favours targets ahead and bosses, on screen,
 * and in sight unless the weapon lobs or the target flies; but a barrel in sight with an enemy beside it comes
 * first (the old `findTarget`).
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
  props.length = 0;
  for (let k = 0; k < n; k++) {
    const id = near[k];
    const j = indexOf(id);
    if (unit.$.sparse[j] < 0 || unit.flags[j] & UNIT_DEAD) continue;
    const ax = pos.x[j],
      ay = pos.y[j] - body.h[j] * 0.5;
    if (ax > cam.right - 6 || ax < cam.left + 4) continue;
    const tags = game.unitDef(id).tags ?? 0;
    if (tags & TAG_PROP) {
      props.push(id);
      continue;
    }
    const d = Math.hypot(ax - x, (ay - gy) * 0.6);
    const score = (ax < x - 10 ? d * 1.6 : d) * (tags & TAG_BOSS ? 0.7 : 1);
    if (score >= bestScore) continue;
    if (!lobbed && !(tags & TAG_FLYING) && !game.terrain.lineOfSight(x, gy, ax, ay)) continue;
    best = id;
    bestScore = score;
  }
  for (const id of props) {
    const j = indexOf(id);
    const d = Math.hypot(pos.x[j] - x, pos.y[j] - pos.y[i]);
    if (d > range * 0.9 || d > bestScore + 20) continue;
    if (enemyBeside(game, pos.x[j], pos.y[j]) && game.terrain.lineOfSight(x, gy, pos.x[j], pos.y[j] - 4)) return id;
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

/** Jetpack flight (the old `jet`): along a few blocks over the ground ahead, over anything in the way. */
function fly(game: BattleGame, i: number, dir: number, dt: number): void {
  const { pos } = game.core;
  const { motor, intent, player, unit } = game.c;
  const t = game.terrain;
  const x = pos.x[i],
    y = pos.y[i];
  player.jet[i] -= dt;
  motor.mode[i] = MODE_FLY;
  const below = Math.min(t.surfaceY(x, y - 100), t.surfaceY(x + dir * 14, y - 100));
  const ty = below - 34 + Math.sin(game.now * 3 + player.id[i]) * 4;
  // Locomotion eases towards up × speed: the old climb, 3 px/s for every px off, up to 120
  intent.up[i] = Math.max(-120, Math.min(120, (y - ty) * 3)) / motor.speed[i];
  const fx = game.rng.fx;
  const bx = x - unit.facing[i] * 3;
  if (fx.chance(0.9)) game.effect("squad.jet", bx, y - 6);
  if (fx.chance(0.3)) game.effect("squad.smoke", bx, y - 2);
}

export function playerThink(game: BattleGame, id: number, i: number, dt: number): void {
  const sq = game.world.res(SQUAD);
  const { player, intent, motor, unit, weapon, ai } = game.c;
  const { pos, vel, body } = game.core;
  const r = game.rng.ai;
  const p = player.id[i];
  const x = pos.x[i],
    y = pos.y[i];
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

  // A big climb ahead: take it on the jetpack now and then
  if (
    !chute &&
    player.jet[i] <= 0 &&
    body.flags[i] & BODY_GROUNDED &&
    dir !== 0 &&
    r.chance(dt * 0.15) &&
    game.terrain.surfaceY(x + dir * 40, y - 120) < y - 32
  )
    player.jet[i] = 2.5;
  if (player.jet[i] > 0 && !chute) fly(game, i, dir, dt);
  else if (motor.mode[i] === MODE_FLY) {
    // Out of fuel: down to earth
    motor.mode[i] = MODE_WALK;
    intent.up[i] = 0;
  }

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
    // Over it on the jetpack, or blow a way through
    if (r.chance(0.5)) player.jet[i] = 3;
    else throwGrenade(game, id, x + (dir || 1) * 24);
    motor.stuckT[i] = 0;
  }
  // Well behind its place and making no progress: blast a way on and fly, then drop back in
  if (dx < 40 || chute) {
    ai.home[i] = x;
    ai.t[i] = 0;
  } else if (x > ai.home[i] + 2) {
    ai.home[i] = x;
    ai.t[i] = 0;
  } else if ((ai.t[i] += dt) > 4 && ai.t[i] - dt <= 4) {
    throwGrenade(game, id, x + 20);
    player.jet[i] = 3;
  }
  if (ai.t[i] > 9) queueDrop(game, id);
}
