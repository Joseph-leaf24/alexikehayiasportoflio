/**
 * Unit locomotion on the cell terrain: everything that walks, swims, climbs or falls moves here, ported from the
 * old engine's `updatePlayer`/`integrate`/`swim`/`tumble` and made robust:
 *
 * - bodies move in sub-steps of at most half a cell per axis, so no speed skips a floor or a wall;
 * - ledges up to `stepUp` cells are stepped up without a jump; taller walls are jumped, then climbed;
 * - a gap or lava ahead is jumped; ladders are climbed; water is swum with the head just out of it;
 * - parachutes drift down; blasts fling a unit (no control until it lands); corpses tumble and slide to rest;
 * - a body that ends up inside terrain (sand poured on it, rubble landing on it) is pushed out to the nearest room.
 *
 * AI writes `intent` (move, up, jump); locomotion turns it into motion and reports what it ran into in `motor.out`.
 */
import { BODY_GROUNDED, indexOf, materials, TILE, type System, type Terrain } from "@/engine";
import type { BattleGame } from "../game/BattleGame";
import {
  AUTO_JUMP,
  CAN_CLIMB,
  CAN_JUMP,
  CAN_LADDER,
  CAN_SWIM,
  FLIES,
  HIT_CEILING,
  HIT_WALL,
  IN_ACID,
  IN_LAVA,
  IN_WATER,
  LANDED,
  MODE_CHUTE,
  MODE_CLIMB,
  MODE_FLY,
  MODE_KNOCK,
  MODE_SWIM,
  MODE_TUMBLE,
  MODE_WALK,
  ON_LADDER,
  PUSHED_OUT,
  UNIT_DEAD,
  UNIT_HELD,
} from "../game/components";
import type { MoveSpec } from "../game/defs";

const { LADDER, LAVA, ACID } = materials;

/** px/s², the old engine's GRAV. */
export const GRAVITY = 620;

/** The old squad's feel: RUN 56, JUMP 200, CLIMB 70, a 420 px/s terminal fall and a 55 px/s parachute. */
export const DEFAULT_MOVE: Readonly<MoveSpec> = {
  speed: 56,
  jump: 200,
  climb: 70,
  stepUp: 1,
  gravity: 1,
  maxFall: 420,
  swim: true,
  autoJump: true,
  ladders: true,
  fly: false,
  chuteFall: 55,
};

/** Most px moved per sub-step on either axis: half a cell, so a move can never pass through one. */
export const SUB_STEP = TILE / 2;
/** Sub-steps per tick at most (a body flung at 1,900 px/s still moves exactly). */
const MAX_SUB = 16;
/** Seconds a blast takes control away for, at least, before landing gives it back. */
const KNOCK_TIME = 0.25;
/** Swimmers float with their head this many px above the water. */
const FLOAT = 4;

/** Copy a MoveSpec into a unit's motor columns. */
export function initMotor(game: BattleGame, i: number, spec: Readonly<MoveSpec>): void {
  const m = game.c.motor;
  m.speed[i] = spec.speed;
  m.jump[i] = spec.jump;
  m.climb[i] = spec.climb;
  m.step[i] = spec.stepUp;
  m.grav[i] = spec.gravity;
  m.maxFall[i] = spec.maxFall;
  m.chute[i] = spec.chuteFall;
  m.abil[i] =
    (spec.jump > 0 ? CAN_JUMP : 0) |
    (spec.climb > 0 ? CAN_CLIMB : 0) |
    (spec.swim ? CAN_SWIM : 0) |
    (spec.autoJump ? AUTO_JUMP : 0) |
    (spec.ladders && spec.climb > 0 ? CAN_LADDER : 0) |
    (spec.fly ? FLIES : 0);
  m.mode[i] = spec.fly ? MODE_FLY : MODE_WALK;
}

/** Fling a unit (a blast, a ram): adds velocity and takes control away until it lands. */
export function knock(game: BattleGame, id: number, vx: number, vy: number): void {
  if (!game.world.has(id, game.c.motor)) return;
  const i = indexOf(id);
  const m = game.c.motor;
  const mode = m.mode[i];
  if (mode === MODE_TUMBLE || mode === MODE_FLY) {
    game.core.vel.vx[i] += vx;
    game.core.vel.vy[i] = Math.min(game.core.vel.vy[i], vy);
    return;
  }
  game.core.vel.vx[i] = vx;
  game.core.vel.vy[i] = Math.min(game.core.vel.vy[i], vy);
  game.core.body.flags[i] &= ~BODY_GROUNDED;
  m.mode[i] = MODE_KNOCK;
  m.timer[i] = KNOCK_TIME;
}

/* ---------------------------------------------------------------- the box mover */

/** A moving box: feet position, velocity, and what it hit. Module scratch, so moving allocates nothing. */
export interface Mover {
  x: number;
  y: number;
  vx: number;
  vy: number;
  grounded: boolean;
  out: number;
}

/** Room for a box with its feet at (x, y)? */
const free = (t: Terrain, x: number, y: number, hw: number, h: number) => !t.overlapsSolid(x, y, hw, h);

/** Step up onto a ledge of at most `step` cells at `nx`, with headroom where the body stands. */
function stepUp(t: Terrain, s: Mover, nx: number, hw: number, h: number, step: number): boolean {
  const top = Math.ceil(s.y / TILE - 1e-4);
  for (let k = 1; k <= step; k++) {
    const ty = (top - k) * TILE;
    if (free(t, nx, ty, hw, h) && free(t, s.x, ty, hw, h)) {
      s.x = nx;
      s.y = ty;
      return true;
    }
  }
  return false;
}

/**
 * Move a box by its velocity for `dt`, in sub-steps of at most half a cell: x first (stepping up low ledges
 * when grounded), then y (landing flush on floors, stopping flush under ceilings). Sets `grounded` from the cell
 * under the feet afterwards.
 */
export function moveBox(t: Terrain, s: Mover, hw: number, h: number, step: number, dt: number): void {
  const dist = Math.max(Math.abs(s.vx), Math.abs(s.vy)) * dt;
  const n = dist > SUB_STEP ? Math.min(MAX_SUB, Math.ceil(dist / SUB_STEP)) : 1;
  const sdt = dt / n;
  for (let k = 0; k < n; k++) {
    if (s.vx !== 0) {
      const nx = s.x + s.vx * sdt;
      if (free(t, nx, s.y, hw, h)) s.x = nx;
      else if (!(step > 0 && s.grounded && stepUp(t, s, nx, hw, h, step))) {
        // Flush against the wall: the box's edge on the wall's cell boundary
        const cand =
          s.vx > 0 ? Math.floor((nx + hw) / TILE) * TILE - hw : (Math.floor((nx - hw) / TILE) + 1) * TILE + hw;
        if ((s.vx > 0 ? cand > s.x : cand < s.x) && free(t, cand, s.y, hw, h)) s.x = cand;
        s.vx = 0;
        s.out |= HIT_WALL;
      }
    }
    if (s.vy !== 0) {
      const ny = s.y + s.vy * sdt;
      if (free(t, s.x, ny, hw, h)) {
        s.y = ny;
        if (s.vy < 0) s.grounded = false;
      } else if (s.vy > 0) {
        // Land on top of the cell the feet reached
        const fy = Math.floor(ny / TILE) * TILE;
        if (fy >= s.y && free(t, s.x, fy, hw, h)) s.y = fy;
        s.vy = 0;
        s.grounded = true;
        s.out |= LANDED;
      } else {
        const cy = (Math.floor((ny - h) / TILE) + 1) * TILE + h;
        if (cy <= s.y && free(t, s.x, cy, hw, h)) s.y = cy;
        s.vy = 0;
        s.out |= HIT_CEILING;
      }
    }
  }
  s.grounded = t.overlapsSolid(s.x, s.y + 1, hw, 1);
}

/**
 * Find the nearest room for a box that overlaps terrain: straight up first (out of sand or rubble), then a cell or
 * two to either side, then down, then anywhere up the column. Writes the spot into `s`; false when it was clear.
 */
export function pushOut(t: Terrain, s: Mover, hw: number, h: number): boolean {
  const x = s.x;
  if (free(t, x, s.y, hw, h)) return false;
  const base = Math.floor(s.y / TILE) * TILE;
  for (let k = 0; k <= 3; k++)
    if (free(t, x, base - k * TILE, hw, h)) {
      s.y = base - k * TILE;
      return true;
    }
  for (let side = 1; side <= 2; side++)
    for (let d = -1; d <= 1; d += 2)
      for (let k = 0; k <= 2; k++) {
        const tx = x + d * side * TILE;
        if (free(t, tx, base - k * TILE, hw, h)) {
          s.x = tx;
          s.y = base - k * TILE;
          return true;
        }
      }
  if (free(t, x, base + TILE, hw, h)) {
    s.y = base + TILE;
    return true;
  }
  for (let y = base - 4 * TILE; y > h; y -= TILE)
    if (free(t, x, y, hw, h)) {
      s.y = y;
      return true;
    }
  s.y = h + 1;
  return true;
}

/* ---------------------------------------------------------------- per-unit movement */

const mover: Mover = { x: 0, y: 0, vx: 0, vy: 0, grounded: false, out: 0 };

/** A wall taller than the unit can step, just ahead in direction `dir`. */
function wallAhead(t: Terrain, x: number, y: number, hw: number, h: number, dir: number, step: number): boolean {
  const probe = x + dir * (hw + 3);
  const low = y - step * TILE - 2;
  if (t.solidAt(probe, low) || t.solidAt(probe, y - h + 2)) return true;
  const mid = y - h / 2;
  return mid < low && t.solidAt(probe, mid);
}

/** A drop of more than a cell, or lava, just ahead. */
function hazardAhead(t: Terrain, x: number, y: number, dir: number): boolean {
  const ahead = x + dir * 7;
  if (!t.solidAt(ahead, y + 2) && !t.solidAt(ahead, y + 10)) return true;
  return t.liquidAt(ahead, y + 6) === LAVA || t.liquidAt(ahead, y + 12) === LAVA;
}

/** Feet y a swimmer floats at: its head just out of the water above it. */
function floatY(t: Terrain, x: number, y: number, h: number): number {
  let r = Math.floor((y - FLOAT) / TILE);
  while (r > 0 && t.liquidAt(x, (r - 1) * TILE + 4) !== 0) r--;
  return r * TILE + h - FLOAT;
}

/** Move one unit for one tick. */
export function moveUnit(game: BattleGame, i: number, dt: number): void {
  const t = game.terrain;
  const { pos, vel, body } = game.core;
  const { motor, intent, unit } = game.c;
  const fl = unit.flags[i];
  if (fl & UNIT_HELD) return;
  const hw = body.hw[i],
    h = body.h[i];
  const s = mover;
  s.x = pos.x[i];
  s.y = pos.y[i];
  s.vx = vel.vx[i];
  s.vy = vel.vy[i];
  s.grounded = (body.flags[i] & BODY_GROUNDED) !== 0;
  s.out = 0;
  const abil = motor.abil[i];
  let mode = motor.mode[i];
  const dir = fl & UNIT_DEAD ? 0 : Math.max(-1, Math.min(1, intent.move[i]));
  const speed = motor.speed[i];
  const g = GRAVITY * motor.grav[i];
  const maxFall = motor.maxFall[i];

  // What it stands in: water deep enough to swim at chest and feet, lava or acid at the feet
  const feet = t.liquidAt(s.x, s.y - 1);
  const chest = t.liquidAt(s.x, s.y - Math.min(6, h * 0.45));
  if (feet === LAVA) s.out |= IN_LAVA;
  if (feet === ACID || chest === ACID) s.out |= IN_ACID;
  const deep = feet !== 0 && chest !== 0 && feet !== LAVA;
  if (deep) s.out |= IN_WATER;

  if (fl & UNIT_DEAD && mode !== MODE_TUMBLE) mode = MODE_TUMBLE;
  switch (mode) {
    case MODE_TUMBLE: {
      // A corpse: bounces and slides to rest, spinning while airborne (the old `tumble`)
      const vx0 = s.vx;
      s.vy = deep ? Math.min(s.vy + g * 0.2 * dt, 30) : Math.min(s.vy + g * dt, maxFall);
      moveBox(t, s, hw, h, 0, dt);
      if (s.grounded) s.vx = vx0 * 0.6;
      else unit.spin[i] += dt * 14;
      break;
    }
    case MODE_KNOCK: {
      s.vy = Math.min(s.vy + g * dt, maxFall);
      moveBox(t, s, hw, h, 0, dt);
      motor.timer[i] -= dt;
      if (s.grounded) s.vx *= 0.7;
      if ((s.grounded && motor.timer[i] <= 0) || deep) mode = deep && abil & CAN_SWIM ? MODE_SWIM : MODE_WALK;
      break;
    }
    case MODE_CHUTE: {
      s.vx = dir * speed * 0.35;
      s.vy = Math.min(s.vy + g * 0.25 * dt, motor.chute[i]);
      moveBox(t, s, hw, h, 0, dt);
      if (s.grounded) mode = MODE_WALK;
      else if (deep) mode = abil & CAN_SWIM ? MODE_SWIM : MODE_WALK;
      break;
    }
    case MODE_FLY: {
      const k = Math.min(1, dt * 4);
      s.vx += (dir * speed - s.vx) * k;
      s.vy += (-intent.up[i] * speed - s.vy) * k;
      moveBox(t, s, hw, h, 0, dt);
      break;
    }
    default:
      mode = walk(game, i, s, mode, abil, deep, dir, speed, g, maxFall, dt);
  }

  if (pushOut(t, s, hw, h)) {
    s.out |= PUSHED_OUT;
    s.grounded = t.overlapsSolid(s.x, s.y + 1, hw, 1);
  }
  // A big drop into water throws up a splash
  if (deep && !(motor.out[i] & IN_WATER) && vel.vy[i] > 120) game.effect("splash", s.x, s.y - 2);
  pos.x[i] = s.x;
  pos.y[i] = s.y;
  vel.vx[i] = s.vx;
  vel.vy[i] = s.vy;
  if (s.grounded) body.flags[i] |= BODY_GROUNDED;
  else body.flags[i] &= ~BODY_GROUNDED;
  motor.mode[i] = mode;
  motor.out[i] = s.out;
  unit.anim[i] += dt * Math.abs(s.vx) * 0.21;
  // Stuck: pushing on without getting anywhere (climbing a cliff counts as getting somewhere)
  if (dir !== 0 && Math.abs(s.x - motor.lastX[i]) < 0.3 * dt * 30 && !(mode === MODE_CLIMB && s.vy < 0))
    motor.stuckT[i] += dt;
  else motor.stuckT[i] = 0;
  motor.lastX[i] = s.x;
}

/** Walking, climbing and swimming: the controlled modes. Returns the mode it ends in. */
function walk(
  game: BattleGame,
  i: number,
  s: Mover,
  mode: number,
  abil: number,
  deep: boolean,
  dir: number,
  speed: number,
  g: number,
  maxFall: number,
  dt: number
): number {
  const t = game.terrain;
  const { body } = game.core;
  const { motor, intent } = game.c;
  const hw = body.hw[i],
    h = body.h[i];
  const step = motor.step[i];
  if (deep && abil & CAN_SWIM) {
    // Swim: stroke along with the head just out, and hop out where the bank is near the surface. The water
    // takes most of a plunge's speed, so a jump in doesn't carry it to the bottom
    if (!(motor.out[i] & IN_WATER) && s.vy > 0) s.vy *= 0.35;
    s.vx = dir * speed * 0.7;
    const fy = floatY(t, s.x, s.y, h);
    s.vy = Math.max(-110, Math.min(110, s.vy + ((fy - s.y) * 9 - s.vy * 6) * dt));
    moveBox(t, s, hw, h, step, dt);
    if (s.out & HIT_WALL && Math.abs(s.y - fy) < 6 && abil & CAN_JUMP) {
      s.vy = -motor.jump[i] * 0.8;
      return MODE_WALK;
    }
    return MODE_SWIM;
  }
  s.vx = dir * speed;
  const up = intent.up[i];
  const ladder =
    abil & CAN_LADDER && (t.materialAt(s.x, s.y - 2) === LADDER || t.materialAt(s.x, s.y - h * 0.6) === LADDER);
  if (ladder) s.out |= ON_LADDER;
  // Airborne, nothing can be stepped onto, so any wall at the feet blocks: a climber keeps on until it is clear
  const blocked = dir !== 0 && wallAhead(t, s.x, s.y, hw, h, dir, s.grounded ? step : 0);
  const auto = (abil & AUTO_JUMP) !== 0;
  if (ladder && (up > 0.3 || (auto && (blocked || (dir !== 0 && t.overlapsSolid(s.x + dir, s.y, hw, h)))))) {
    // Up the ladder, edging forward so it steps off onto the top
    s.vx = dir * speed * 0.4;
    s.vy = -motor.climb[i];
    moveBox(t, s, hw, h, step, dt);
    return MODE_CLIMB;
  }
  if (ladder && up < -0.3) {
    s.vx = 0;
    s.vy = motor.climb[i];
    moveBox(t, s, hw, h, step, dt);
    return MODE_CLIMB;
  }
  let next = MODE_WALK;
  const wantJump = intent.jump[i] !== 0;
  intent.jump[i] = 0;
  if (blocked) {
    s.vx = 0;
    s.out |= HIT_WALL;
    if (s.grounded && abil & CAN_JUMP && (auto || wantJump)) {
      s.vy = -motor.jump[i];
      s.grounded = false;
    } else if (!s.grounded && abil & CAN_CLIMB && auto && s.vy > -motor.climb[i] && s.vy < 140) {
      // Grab the wall while rising or near the top of a jump, not while falling fast down a shaft
      s.vy = -motor.climb[i];
      next = MODE_CLIMB;
    }
  } else if (s.grounded && abil & CAN_JUMP && (wantJump || (auto && dir !== 0 && hazardAhead(t, s.x, s.y, dir)))) {
    s.vy = -motor.jump[i];
    s.grounded = false;
  }
  s.vy = Math.min(s.vy + g * dt, maxFall);
  moveBox(t, s, hw, h, step, dt);
  return next;
}

/** Moves every unit, each tick ("motion" phase). */
export function locomotionSystem(game: BattleGame): System {
  const { pos, vel, body } = game.core;
  const { motor, unit } = game.c;
  const q = game.world.query(pos, vel, body, motor, unit);
  return {
    name: "locomotion.move",
    phase: "motion",
    run(_w, dt) {
      const n = q.collect(true),
        idx = q.idx;
      for (let k = 0; k < n; k++) moveUnit(game, idx[k], dt);
    },
  };
}

/**
 * After the terrain has moved (sand settling, rubble landing, chunks shattering): push every unit that is now
 * inside it back out, so no unit ever ends a tick inside solid ground ("terrain" phase, after the terrain step).
 */
export function settleSystem(game: BattleGame): System {
  const { pos, body } = game.core;
  const { motor, unit } = game.c;
  const q = game.world.query(pos, body, motor, unit);
  return {
    name: "locomotion.settle",
    phase: "terrain",
    order: 100,
    run() {
      const t = game.terrain;
      const n = q.collect(true),
        idx = q.idx;
      const s = mover;
      for (let k = 0; k < n; k++) {
        const i = idx[k];
        if (unit.flags[i] & UNIT_HELD) continue;
        s.x = pos.x[i];
        s.y = pos.y[i];
        if (!pushOut(t, s, body.hw[i], body.h[i])) continue;
        pos.x[i] = s.x;
        pos.y[i] = s.y;
        motor.out[i] |= PUSHED_OUT;
        game.core.vel.vy[i] = Math.min(0, game.core.vel.vy[i]);
      }
    },
  };
}
