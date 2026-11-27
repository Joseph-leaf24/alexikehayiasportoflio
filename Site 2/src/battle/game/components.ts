/**
 * The game's shared components, on top of the core's pos/prev/vel/body/health/team/kind/sprite/life. Hot numbers
 * are typed columns indexed by entity index (`c.motor.speed[i]`); a feature with rare or structured state defines
 * its own object component (`world.defineObject<T>`) in its install instead of widening these.
 */
import type { World } from "@/engine";

/* unit.flags */
/** Down: out of the fight, no longer targetable; a corpse tumbling, or a player waiting to respawn. */
export const UNIT_DEAD = 1;
/** Not drawn (a player queued for a drop, a crew member inside a vehicle). */
export const UNIT_HIDDEN = 2;
/** Doesn't think, move or fire: held by something else (a vehicle, a cutscene). */
export const UNIT_HELD = 4;
/** Counted against the enemy budget (set at spawn for every non-squad unit). */
export const UNIT_COUNTED = 8;

/* motor.mode */
export const MODE_WALK = 0;
export const MODE_CLIMB = 1;
export const MODE_SWIM = 2;
export const MODE_CHUTE = 3;
export const MODE_FLY = 4;
/** Flung by a blast: no control until it lands. */
export const MODE_KNOCK = 5;
/** A corpse: bounces, slides and spins to rest. */
export const MODE_TUMBLE = 6;

/* motor.abil */
export const CAN_JUMP = 1;
export const CAN_CLIMB = 2;
export const CAN_SWIM = 4;
export const AUTO_JUMP = 8;
export const CAN_LADDER = 16;
export const FLIES = 32;

/* motor.out: what the last move ran into */
export const HIT_WALL = 1;
export const HIT_CEILING = 2;
export const LANDED = 4;
export const IN_WATER = 8;
export const IN_LAVA = 16;
export const IN_ACID = 32;
export const ON_LADDER = 64;
/** It ended a move inside terrain and was pushed out. */
export const PUSHED_OUT = 128;

export function defineGameComponents(world: World) {
  return {
    /** Anything that fights: players, troops, vehicles, bosses. `kind.def` is its UnitDef index. */
    unit: world.defineComponent("unit", {
      flags: "u8",
      facing: ["i8", 1],
      /** Art set index overriding the def's (a player's current weapon, a faction's uniform); 0 = the def's. */
      art: "u16",
      /** Animation clock (run cycle), advanced by locomotion. */
      anim: "f32",
      /** Seconds of hit flash left. */
      hitT: "f32",
      /** Seconds since it went down. */
      deadT: "f32",
      /** Corpse spin, quarter turns. */
      spin: "f32",
      /** Who brought it down (entity id). */
      killer: "u32",
    }),
    /** Locomotion state and tuning (copied from the def's MoveSpec at spawn; per-unit so pickups can change it). */
    motor: world.defineComponent("motor", {
      mode: "u8",
      abil: "u8",
      out: "u8",
      step: "u8",
      speed: "f32",
      jump: "f32",
      climb: "f32",
      grav: ["f32", 1],
      maxFall: ["f32", 420],
      chute: ["f32", 55],
      /** Seconds left in a timed mode (a blast's knockback). */
      timer: "f32",
      /** Seconds pushing on without moving, for the stuck handling of AIs. */
      stuckT: "f32",
      lastX: "f32",
    }),
    /** What the unit wants this tick, written by its AI (or a player's input) and read by locomotion and weapons. */
    intent: world.defineComponent("intent", {
      /** Run direction and strength, -1..1 (× motor.speed). */
      move: "f32",
      /** Climb wish, -1..1 (up is positive); fliers use it as vertical move. */
      up: "f32",
      /** Jump now (cleared by locomotion). */
      jump: "u8",
      /** Fire at `target`, or at (aimX, aimY) when there is none. */
      fire: "u8",
      target: "u32",
      aimX: "f32",
      aimY: "f32",
    }),
    /** The unit's gun: weapon def index, cooldown and burst count. */
    weapon: world.defineComponent("weapon", {
      def: "u16",
      cd: "f32",
      shots: "u16",
      /** A second cooldown for an alternate attack (grenades). */
      alt: "f32",
    }),
    /** One of the eight squad members. `id` is 0–7, stable for the whole run. */
    player: world.defineComponent("player", {
      id: "u8",
      score: "f32",
      kills: "u32",
      /** Seconds of spawn protection left. */
      inv: "f32",
      /** Seconds until a downed player is queued for a drop. */
      downT: "f32",
      queued: "u8",
    }),
    /** Generic AI scratch for the shared behaviours; bespoke AIs keep their state in their own object component. */
    ai: world.defineComponent("ai", {
      state: "u8",
      /** Seconds in the current state. */
      t: "f32",
      scanT: "f32",
      cover: "f32",
      /** Where it was placed or last anchored (world x). */
      home: "f32",
      alert: "u8",
    }),
    /** A shot in flight. Its team is in `team`, its motion in pos/prev/vel. */
    projectile: world.defineComponent("projectile", {
      def: "u16",
      owner: "u32",
      /** The owner's player id (0–7), or -1; kept so a kill is credited after the owner is gone. */
      pid: ["i8", -1],
      pierce: "u8",
      bounces: "u8",
      dmg: "f32",
      /** Seconds before it expires. */
      life: "f32",
      /** Last body hit, so a piercing shot doesn't hit the same one twice in a row. */
      last: "u32",
      /** Trail emission accumulator. */
      trail: "f32",
    }),
  };
}

export type GameComponents = ReturnType<typeof defineGameComponents>;
