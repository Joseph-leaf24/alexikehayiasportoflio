/** The components every module shares, defined on each world at construction, plus their flag bits. */

import type { Ecs } from "./ecs";
import { INDEX_MASK } from "./ecs";

/* body.flags */
export const BODY_GROUNDED = 1 << 0;
export const BODY_FLYING = 1 << 1;
export const BODY_SWIMMING = 1 << 2;
export const BODY_NO_GRAVITY = 1 << 3;
export const BODY_COLLIDES_TERRAIN = 1 << 4;
export const BODY_SOLID_TO_SHOTS = 1 << 5;
export const BODY_SOLID_TO_UNITS = 1 << 6;
/** Left out of the spatial hash (bullets, debris, props nobody queries for). */
export const BODY_UNINDEXED = 1 << 7;
export const BODY_CLIMBING = 1 << 8;
export const BODY_INVULNERABLE = 1 << 9;
/** Bits from here up are the game's own. */
export const BODY_USER = 1 << 16;

/**
 * sprite.flags: the low bits are the renderer's SPRITE_* flags passed straight through (flipX = 1, emissive = 2,
 * …); this one is the core's, and tells the sprite pass to skip the entity.
 */
export const SPRITE_HIDDEN = 1 << 7;

export const TEAM_NEUTRAL = 0;
export const TEAM_SQUAD = 1;
/** Factions take 2 and up (spatial team masks cover 0..31). */
export const TEAM_FIRST_FACTION = 2;

export function defineCoreComponents(ecs: Ecs) {
  return {
    /** Feet position in world px (y down). */
    pos: ecs.defineComponent("pos", { x: "f32", y: "f32" }),
    /** Position at the start of the tick, for render interpolation; `place()` sets both. */
    prev: ecs.defineComponent("prev", { x: "f32", y: "f32" }),
    vel: ecs.defineComponent("vel", { vx: "f32", vy: "f32" }),
    /** Box [x - hw, x + hw] × [y - h, y]. */
    body: ecs.defineComponent("body", { hw: "f32", h: "f32", flags: "u32" }),
    health: ecs.defineComponent("health", { hp: "f32", max: "f32" }),
    team: ecs.defineComponent("team", { id: "u8" }),
    /** Index into a content registry (1-based; 0 = none). */
    kind: ecs.defineComponent("kind", { def: "u16" }),
    sprite: ecs.defineComponent("sprite", {
      frame: "u16",
      layer: "u8",
      flags: "u8",
      tint: ["u32", 0xffffffff],
      scale: ["f32", 1],
    }),
    /** Seconds left; the core life system destroys the entity at 0. */
    life: ecs.defineComponent("life", { t: "f32" }),
  };
}

export type CoreComponents = ReturnType<typeof defineCoreComponents>;

/** Put an entity at (x, y) with no interpolation smear (spawns, teleports); adds pos and prev if missing. */
export function place(world: Ecs & { readonly core: CoreComponents }, id: number, x: number, y: number): boolean {
  const { pos, prev } = world.core;
  if (!world.add(id, pos) || !world.add(id, prev)) return false;
  const i = id & INDEX_MASK;
  pos.x[i] = prev.x[i] = x;
  pos.y[i] = prev.y[i] = y;
  return true;
}

/**
 * Shift every position by (dx, dy): a floating origin for endless runs. f32 positions move in 1/64 px steps up to
 * 131,072 px and 1/16 px steps up to 524,288 px, so an endless mode should rebase by a multiple of the terrain
 * ring (512 × 8 = 4096 px) before then, moving the terrain's start column and the camera by the same amount.
 */
export function shiftOrigin(world: Ecs & { readonly core: CoreComponents }, dx: number, dy: number): void {
  const { pos, prev } = world.core;
  for (const c of [pos, prev]) {
    const { dense, size } = c.$;
    for (let k = 0; k < size; k++) {
      const i = dense[k];
      c.x[i] += dx;
      c.y[i] += dy;
    }
  }
}
