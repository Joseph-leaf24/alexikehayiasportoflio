/**
 * The squad: eight players P1–P8 with stable entities for the whole run, the advance point their formation moves
 * along (the old `moveSquad`: it pushes on, holds while there's a fight just ahead, never runs away from the squad),
 * downed players coming back by parachute, and the camera's framing of them (the old `cameraTarget`).
 */
import { BODY_SOLID_TO_SHOTS, BODY_UNINDEXED, indexOf, LEVEL_H, place, resource, TILE, type System } from "@/engine";
import { PLAYER_COLORS } from "@/data/squad";
import type { BattleGame } from "../game/BattleGame";
import { MODE_CHUTE, MODE_WALK, UNIT_DEAD, UNIT_HIDDEN } from "../game/components";
import { TAG_FLYING, TAG_PROP } from "../game/defs";
import { hostileTo, TEAM_SQUAD } from "../game/teams";

export const SQUAD_SIZE = 8;
export const PLAYER_HP = 6;
/** Each player's offset from the advance point: the flamethrower leads, the sniper hangs back. */
export const OFFSETS = [36, 48, 10, -16, -42, 62, -4, 22];
/** Where the advance point starts on a new level. */
export const START_X = 260;
/** Seconds between drops, and the least a downed player stays out before dropping back in. */
const DROP_GAP = 0.4;

export class Squad {
  /** The formation's anchor, world px. */
  advance = START_X;
  /** How fast it moves on, px/s (the old jungle theme's 38). */
  speed = 38;
  /** Seconds a fight just ahead has held the squad up; past 4 it pushes on regardless. */
  holdT = 0;
  /** Never past this x (a feature holding the squad: a boss arena, a defence). */
  holdAt = Infinity;
  /** Player entities by player id 0–7. */
  readonly players: number[] = [];
  /** Player ids waiting to drop back in, and seconds until the next drop. */
  readonly queue: number[] = [];
  dropT = 0;
  /** Supply crates waiting to be picked up: entity → a weapon number, or -1 for a jetpack; and seconds to the next. */
  readonly crates = new Map<number, number>();
  crateT = 5;
  /** Player colours as RGBA, for popups and tags. */
  readonly colors = PLAYER_COLORS.map((c) => ((parseInt(c.slice(1), 16) << 8) | 0xff) >>> 0);
}

export const SQUAD = resource<Squad>("squad");

/** Feet y of the first open spot at x from `y` down with room to stand (a drop into a cavern lands inside it). */
export function openAir(game: BattleGame, x: number, y: number, h: number): number {
  const t = game.terrain;
  for (let yy = Math.max(h + 1, y); yy < LEVEL_H - TILE; yy += TILE) if (!t.overlapsSolid(x, yy, 3, h)) return yy;
  return y;
}

/** Bring a player back: full health, spawn protection, under a parachute at (x, y). */
export function release(game: BattleGame, id: number, x: number, y: number): void {
  const i = indexOf(id);
  const { unit, player, motor, intent, weapon } = game.c;
  const { body, health, vel } = game.core;
  place(game.world, id, x, openAir(game, x, y, body.h[i]));
  unit.flags[i] &= ~(UNIT_DEAD | UNIT_HIDDEN);
  unit.hitT[i] = 0;
  health.hp[i] = health.max[i];
  body.flags[i] = (body.flags[i] & ~BODY_UNINDEXED) | BODY_SOLID_TO_SHOTS;
  motor.mode[i] = MODE_CHUTE;
  motor.stuckT[i] = 0;
  vel.vx[i] = vel.vy[i] = 0;
  intent.target[i] = 0;
  intent.fire[i] = 0;
  weapon.cd[i] = 0.5;
  player.inv[i] = 1.5;
  player.queued[i] = 0;
  player.jet[i] = 0;
  game.world.events.emit("playerRespawned", { id, player: player.id[i], x, y });
}

/**
 * Take a player out until the next drop: hidden, off to the left of the view, queued. One still up (stuck, or left
 * behind) is benched, not killed: no "DOWN", as in the old game.
 */
export function queueDrop(game: BattleGame, id: number): void {
  const sq = game.world.res(SQUAD);
  const i = indexOf(id);
  const { unit, player, motor, intent } = game.c;
  const body = game.core.body;
  // Out of every query for targets and hits until it is released
  body.flags[i] = (body.flags[i] | BODY_UNINDEXED) & ~BODY_SOLID_TO_SHOTS;
  intent.fire[i] = intent.move[i] = intent.target[i] = 0;
  player.jet[i] = 0;
  unit.flags[i] |= UNIT_DEAD | UNIT_HIDDEN;
  player.queued[i] = 1;
  motor.mode[i] = MODE_WALK;
  place(game.world, id, game.camera.left - 200, game.camera.top);
  game.core.vel.vx[i] = game.core.vel.vy[i] = 0;
  if (!sq.queue.includes(player.id[i])) sq.queue.push(player.id[i]);
  // Out for at least one gap, even when the queue was idle (else hidden and dropped back in on the same tick).
  sq.dropT = Math.max(sq.dropT, DROP_GAP);
}

const near: number[] = [];

/** The advance point ("director" phase), after the old `moveSquad`. */
export function advanceSystem(game: BattleGame): System {
  return {
    name: "squad.advance",
    phase: "director",
    run(w, dt) {
      const sq = w.res(SQUAD);
      const { pos } = game.core;
      let front = -Infinity,
        sum = 0,
        n = 0;
      for (const id of sq.players) {
        if (!game.fighting(id)) continue;
        const x = pos.x[indexOf(id)];
        if (x > front) front = x;
        sum += x;
        n++;
      }
      if (n === 0) front = sq.advance;
      // A fight on the ground just ahead holds the advance a while
      const m = w.spatial.rect(
        front - 10,
        game.camera.top,
        front + 80,
        game.camera.bottom,
        hostileTo(TEAM_SQUAD),
        near
      );
      let threat = false;
      for (let k = 0; k < m && !threat; k++)
        threat = game.fighting(near[k]) && ((game.unitDef(near[k]).tags ?? 0) & (TAG_FLYING | TAG_PROP)) === 0;
      sq.holdT = threat ? sq.holdT + dt : 0;
      if (!threat || sq.holdT > 4) sq.advance += sq.speed * dt;
      const mean = n > 0 ? sum / n : sq.advance;
      sq.advance = Math.min(sq.advance, front + 50, mean + 40, sq.holdAt);
    },
  };
}

/** Downed players wait, get queued, and drop back in one by one by parachute ("spawn" phase). */
export function respawnSystem(game: BattleGame): System {
  return {
    name: "squad.respawn",
    phase: "spawn",
    run(w, dt) {
      const sq = w.res(SQUAD);
      const { player, unit } = game.c;
      const { pos } = game.core;
      const cam = game.camera;
      for (const id of sq.players) {
        const i = indexOf(id);
        if (player.queued[i]) continue;
        if (unit.flags[i] & UNIT_DEAD) {
          if ((player.downT[i] -= dt) <= 0) queueDrop(game, id);
        } else if (pos.x[i] < cam.left - 40 || pos.y[i] > LEVEL_H - 4) queueDrop(game, id);
      }
      sq.dropT -= dt;
      if (sq.queue.length === 0 || sq.dropT > 0) return;
      const p = sq.queue.shift() as number;
      sq.dropT = DROP_GAP;
      const x = Math.max(cam.left + 24, sq.advance + OFFSETS[p] * 0.5 + (game.rng.squad.next() - 0.5) * 30);
      release(game, sq.players[p], x, cam.top - 24);
    },
  };
}

/**
 * Frame the feet of the squad on foot, a third of the way across; with the squad split up and down a shaft, frame
 * the larger group, or the one further on. With nobody on foot, the ground under the advance point.
 */
const framing = { x: 0, y: 0 };

export function squadCamera(game: BattleGame): { x: number; y: number } {
  const sq = game.world.res(SQUAD);
  const cam = game.camera;
  const { pos } = game.core;
  const { motor } = game.c;
  const W = cam.viewW / cam.zoom;
  const H = cam.viewH / cam.zoom;
  let feet = game.terrain.surfaceY(sq.advance, 0);
  let most = 0,
    lead = -Infinity;
  for (const a of sq.players) {
    if (!game.fighting(a) || motor.mode[indexOf(a)] === MODE_CHUTE) continue;
    const ay = pos.y[indexOf(a)];
    let n = 0,
      sum = 0;
    for (const b of sq.players) {
      if (!game.fighting(b) || motor.mode[indexOf(b)] === MODE_CHUTE) continue;
      const by = pos.y[indexOf(b)];
      if (Math.abs(by - ay) < H * 0.3) {
        n++;
        sum += by;
      }
    }
    const ax = pos.x[indexOf(a)];
    if (n > most || (n === most && ax > lead)) {
      most = n;
      lead = ax;
      feet = sum / n;
    }
  }
  const left = Math.max(game.terrain.startCol * TILE, sq.advance - W * 0.32);
  const top = Math.max(-80, Math.min(LEVEL_H - H, feet - H * game.sky.horizon));
  framing.x = left + W / 2;
  framing.y = top + H / 2;
  return framing;
}
