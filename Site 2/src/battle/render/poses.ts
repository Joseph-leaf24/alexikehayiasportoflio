/**
 * Pose pickers: which frame of a unit's art set to draw from its state. The old troop and player sprites share one
 * layout (idle, four running frames, jumping), so `humanoidPose` serves every soldier.
 */
import { BODY_GROUNDED } from "@/engine";
import type { BattleGame } from "../game/BattleGame";
import { MODE_CHUTE, MODE_CLIMB, MODE_SWIM, UNIT_DEAD } from "../game/components";

export const POSE_IDLE = 0;
export const POSE_RUN = 1;
export const POSE_JUMP = 5;

export function humanoidPose(game: BattleGame, i: number): number {
  const { unit, motor } = game.c;
  if (unit.flags[i] & UNIT_DEAD) return POSE_IDLE;
  const mode = motor.mode[i];
  if (mode === MODE_CLIMB || mode === MODE_CHUTE) return POSE_JUMP;
  const grounded = (game.core.body.flags[i] & BODY_GROUNDED) !== 0;
  if (!grounded && mode !== MODE_SWIM) return POSE_JUMP;
  const vx = game.core.vel.vx[i];
  if (vx > 1 || vx < -1) return POSE_RUN + (Math.floor(unit.anim[i]) & 3);
  return POSE_IDLE;
}

/** One frame whatever the state (turrets, props). */
export const stillPose = (): number => 0;

/** Two frames alternating with the run clock (dogs, crawlers, heavy troopers). */
export function twoFramePose(game: BattleGame, i: number): number {
  return Math.floor(game.c.unit.anim[i]) & 1;
}
