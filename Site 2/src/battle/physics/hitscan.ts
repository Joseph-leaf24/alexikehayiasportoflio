/**
 * Instant shots along a line (lasers, beams, the sniper's crack): the same sweep a projectile does in one tick,
 * over the whole range at once. Bodies are hit nearest first up to the first wall, as many as it pierces, and the
 * wall cell is chipped. Weapons with a `fire` hook call this and draw their own beam from the result.
 */
import { createRayHit, indexOf, TEAM_ALL, type RayHit } from "@/engine";
import type { BattleGame } from "../game/BattleGame";
import { UNIT_DEAD } from "../game/components";
import { hostileTo } from "../game/teams";
import { hurt } from "./damage";

export interface HitscanOptions {
  damage: number;
  /** Extra bodies it passes through (default 0). */
  pierce?: number;
  /** Beam thickness, px (default 2). */
  width?: number;
  /** Hits knocked off the wall cell it stops at (default 0). */
  chip?: number;
  /** Hurts the shooter's own team too. */
  friendly?: boolean;
}

/** Where a hitscan stopped and what it did. Reused: read it before the next call. */
export interface HitscanResult {
  x: number;
  y: number;
  /** Bodies hit. */
  hits: number;
  /** It stopped at a wall (rather than at its range or at its last body). */
  wall: boolean;
}

const ray: RayHit = createRayHit();
const bodies: number[] = [];
const result: HitscanResult = { x: 0, y: 0, hits: 0, wall: false };

export function hitscan(
  game: BattleGame,
  owner: number,
  team: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  o: HitscanOptions
): HitscanResult {
  const w = game.world;
  const { unit } = game.c;
  const dx = x1 - x0,
    dy = y1 - y0;
  const len = Math.hypot(dx, dy);
  result.x = x1;
  result.y = y1;
  result.hits = 0;
  result.wall = false;
  if (!(len > 0)) return result;
  let tEnd = 1;
  if (game.terrain.raycast(x0, y0, dx, dy, len, ray)) {
    tEnd = ray.dist / len;
    result.wall = true;
  }
  const mask = o.friendly ? TEAM_ALL : hostileTo(team);
  const n = w.spatial.segment(x0, y0, x0 + dx * tEnd, y0 + dy * tEnd, o.width ?? 2, mask, bodies);
  let left = (o.pierce ?? 0) + 1;
  for (let k = 0; k < n && left > 0; k++) {
    const id = bodies[k];
    const i = indexOf(id);
    if (id === owner || unit.$.sparse[i] < 0 || unit.flags[i] & UNIT_DEAD) continue;
    hurt(game, id, o.damage, owner);
    result.hits++;
    if (--left === 0) {
      // Stopped by the last body it could take
      const t = w.spatial.hitT[k] * tEnd;
      result.x = x0 + dx * t;
      result.y = y0 + dy * t;
      result.wall = false;
      return result;
    }
  }
  if (result.wall) {
    result.x = ray.x;
    result.y = ray.y;
    if (o.chip) game.terrain.damage(ray.col, ray.row, o.chip);
  }
  return result;
}
