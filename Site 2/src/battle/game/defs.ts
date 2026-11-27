/**
 * Content definitions: what a unit, weapon, projectile, effect or art set is, as data plus small behaviour hooks.
 * Defs are registered by id into the game's registries at install time and frozen at boot, so shared code only
 * ever reads them. Hot numbers are copied into component columns at spawn; defs are consulted for the rest.
 */
import type { BattleGame } from "./BattleGame";

/* ---------------------------------------------------------------- units */

/** Unit categories (`UnitDef.tags`), for targeting rules, blast reactions and faction rosters. */
export const TAG_INFANTRY = 1;
export const TAG_ARMOURED = 2;
export const TAG_FLYING = 4;
export const TAG_SHIP = 8;
export const TAG_BOSS = 16;
export const TAG_STRUCTURE = 32;

/** Locomotion tuning, copied into the unit's `motor` columns at spawn (see `physics/locomotion.ts`). */
export interface MoveSpec {
  /** Run speed, px/s. */
  speed: number;
  /** Jump launch speed, px/s; 0 = can't jump. */
  jump: number;
  /** Wall and ladder climbing speed, px/s; 0 = can't climb. */
  climb: number;
  /** Ledges up to this many cells high are stepped up without a jump. */
  stepUp: number;
  /** Gravity scale (1 = 620 px/s²). */
  gravity: number;
  /** Terminal fall speed, px/s. */
  maxFall: number;
  /** Floats and strokes along in water; false sinks and walks the bottom. */
  swim: boolean;
  /** Jumps walls and gaps on its way without being told. */
  autoJump: boolean;
  /** Climbs ladders it overlaps when it wants to go up. */
  ladders: boolean;
  /** No gravity: moves where its intent points, in 2D. */
  fly: boolean;
  /** Parachute descent speed, px/s. */
  chuteFall: number;
}

/**
 * A unit's thinking: reads the world and writes the unit's `intent` (move, aim, target, fire). `dt` is the time
 * since it last thought. `i` is the entity index for column access.
 */
export type Think = (game: BattleGame, id: number, i: number, dt: number) => void;

/** One state of an `AiMachine`: `run` returns the next state's name, or nothing to stay. */
export interface AiState {
  enter?: Think;
  run: (game: BattleGame, id: number, i: number, dt: number) => string | void;
}

/** A small state machine, compiled to a `Think` by `machine()`; state numbers live in the `ai.state` column. */
export interface AiMachine {
  readonly initial: string;
  readonly states: Readonly<Record<string, AiState>>;
}

export interface AiSpec {
  readonly think: Think | AiMachine;
  /** Think every n ticks (default 3: 10 Hz), staggered across units. */
  readonly every?: number;
}

/** Which frame of a unit's art set to draw (see `render/poses.ts`). */
export type PoseFn = (game: BattleGame, i: number) => number;

/** Draws extra sprites for a unit after its body (a parachute, a jetpack flame, a turret on a hull). */
export type UnitDrawFn = (game: BattleGame, i: number, x: number, y: number, ctx: DrawSink) => void;

/** What a draw hook can push into, in world space; frames come from art ids via the art book. */
export interface DrawSink {
  sprite(art: number, pose: number, facing: number, x: number, y: number, layer: number, tint?: number): void;
  particle(x: number, y: number, size: number, color: number, flags: number): void;
  light(x: number, y: number, radius: number, color: number, intensity: number): void;
}

export interface UnitDef {
  readonly name: string;
  /** Collision box: half width and height in px; `pos` is the feet. */
  readonly hw: number;
  readonly h: number;
  readonly hp: number;
  /** Default team (spawn options may override). */
  readonly team: number;
  readonly score?: number;
  /** Art set id (`content.art`). */
  readonly art: string;
  /** Default: the humanoid poses (idle, run ×4, jump), as the old troop and player sprites are laid out. */
  readonly pose?: PoseFn;
  readonly draw?: UnitDrawFn;
  /** Sprite layer (default by team: players 8, everyone else 7). */
  readonly layer?: number;
  /** Weapon id (`content.weapons`). */
  readonly weapon?: string;
  readonly ai?: AiSpec;
  readonly move?: Partial<MoveSpec>;
  /** TAG_* bits. */
  readonly tags?: number;
  /** Multiplies incoming damage (0.5 takes half). */
  readonly armour?: number;
  /** Killed and flung outright by a blast that reaches it (light infantry), instead of only damaged. */
  readonly blastKills?: boolean;
  /** Takes this fixed damage from any hostile blast in reach (the squad's 3), instead of the falloff. */
  readonly blastTaken?: number;
  /** Effect id played where it dies. */
  readonly deathFx?: string;
  /** Seconds the body tumbles before it is removed (default 1.2). */
  readonly corpse?: number;
  /** Adjusts incoming damage; return the damage to apply. */
  readonly onHurt?: (game: BattleGame, id: number, dmg: number, source: number) => number;
  readonly onSpawn?: (game: BattleGame, id: number) => void;
  readonly onDeath?: (game: BattleGame, id: number, killer: number) => void;
}

/* ---------------------------------------------------------------- weapons */

/** Fires something other than plain projectiles (a beam, a chain, a sweep): replaces the projectile spawn. */
export type FireFn = (
  game: BattleGame,
  shooter: number,
  gunX: number,
  gunY: number,
  aimX: number,
  aimY: number
) => void;

export interface WeaponDef {
  readonly name: string;
  readonly range: number;
  /** Seconds between shots within a burst (or between shots, when `burst` is 1). */
  readonly cadence: number;
  /** Shots per burst (default 1) and the pause after a burst, seconds. */
  readonly burst?: number;
  readonly burstPause?: number;
  /** Random extra on the burst pause, as a fraction (0.5 = up to half as long again). */
  readonly pauseJitter?: number;
  /** Projectiles per shot (a shotgun's pellets). */
  readonly pellets?: number;
  /** Cone, radians, across which shots scatter. */
  readonly spread?: number;
  /** Random extra launch speed, px/s. */
  readonly speedJitter?: number;
  /** Projectile id (`content.projectiles`). */
  readonly projectile?: string;
  /** Launch speed override; otherwise the projectile's. */
  readonly speed?: number;
  /** Aims a ballistic arc (needs no line of sight), at this horizontal speed. */
  readonly lob?: number;
  /** Leads a moving target by its velocity. */
  readonly lead?: boolean;
  /** Gun offset from the feet (x forward, y up negative), px; default (7, -9). */
  readonly muzzleX?: number;
  readonly muzzleY?: number;
  /** Muzzle flash light: colour (RGB), radius px; none when absent. */
  readonly flash?: { readonly color: number; readonly radius: number };
  /** Effect id at the muzzle (smoke, casings). */
  readonly muzzleFx?: string;
  readonly fire?: FireFn;
  /** Multiplies projectile damage. */
  readonly damage?: number;
}

/* ---------------------------------------------------------------- projectiles and blasts */

/** An explosion: terrain carve, damage and knockback in a radius, and its look. */
export interface BlastSpec {
  /** Radius, px. */
  readonly r: number;
  /** Terrain hits at the centre (default r / 7, as the old engine); 255 clears outright. */
  readonly power?: number;
  /** Carve radius as a fraction of `r` (default 0.62, the old dig). */
  readonly dig?: number;
  /** Damage at the centre, on top of 3 anywhere in reach, to units not killed outright (default 10; the old ×r/22). */
  readonly damage?: number;
  /** Fling speed at the centre, px/s (default 220). */
  readonly knock?: number;
  /** Hurts the owner's own team too. */
  readonly friendly?: boolean;
  /** Effect id for the fireball (default "blast.fire"). */
  readonly fx?: string;
  /** Sets flammable cells alight within this fraction of `r` (default 0). */
  readonly ignite?: number;
}

/** How a projectile is drawn when it has no sprite: a streak of pixels behind its head, or a dot. */
export interface ProjectileLook {
  /** RGBA colours from head to tail; the streak is as long as the list (times `spacing`). */
  readonly colors: readonly number[];
  readonly spacing?: number;
  readonly size?: number;
  /** PARTICLE_* flags (tracers glow at night with EMISSIVE). */
  readonly flags?: number;
  /** Second colour for the head, alternating at 6 Hz (a blinking fuse). */
  readonly blink?: number;
}

export interface ProjectileDef {
  readonly speed: number;
  /** Seconds before it expires. */
  readonly life: number;
  readonly damage: number;
  /** px/s² downwards (negative floats up, as flames do). */
  readonly gravity?: number;
  /** Speed multiplier growth per second (rockets accelerate: 2.2). */
  readonly accel?: number;
  /** Velocity lost per second, as a fraction. */
  readonly drag?: number;
  /** Sweep half-thickness, px: how close it passes to count as a hit. */
  readonly radius?: number;
  /** Extra bodies it passes through. */
  readonly pierce?: number;
  /** Hits knocked off the cell it strikes when it stops there (bullets chip at blocks). */
  readonly chip?: number;
  /** Bounces off terrain with this restitution instead of stopping (grenades 0.45). */
  readonly bounce?: number;
  /** Most bounces before it stops (or explodes); default unlimited while it has life. */
  readonly bounces?: number;
  /** Explodes when its life runs out (a fused grenade), not only on impact. */
  readonly fuse?: boolean;
  readonly explode?: BlastSpec;
  /** Explode on striking a unit or terrain (default true when `explode` is set and there's no fuse). */
  readonly impactExplode?: boolean;
  /** Passes through terrain (beams, ghosts). */
  readonly ghost?: boolean;
  /** Effect ids: a trail emitted per second of flight, and a puff where it strikes terrain. */
  readonly trail?: string;
  readonly trailRate?: number;
  readonly impact?: string;
  /** Art id (`content.art`) drawn rotated along its flight; otherwise `look`. */
  readonly art?: string;
  readonly look?: ProjectileLook;
  /** A light it carries (RGB colour, radius px, intensity). */
  readonly glow?: { readonly color: number; readonly radius: number; readonly intensity: number };
  /** Hits the shooter's own team too. */
  readonly friendly?: boolean;
  /** Steering, homing and the like, each tick before it moves. */
  readonly update?: (game: BattleGame, id: number, i: number, dt: number) => void;
  /** Return false to stop the default hit (damage, pierce, explode). */
  readonly onHit?: (game: BattleGame, id: number, target: number) => boolean | void;
  readonly onExpire?: (game: BattleGame, id: number) => void;
}

/* ---------------------------------------------------------------- effects */

/** A burst of CPU particles (`render/particles.ts`), optionally with a flash of light. */
export interface EffectDef {
  /** Particles per emission: a fixed count or [min, max]. */
  readonly count: number | readonly [number, number];
  /** Launch speed range, px/s. */
  readonly speed: readonly [number, number];
  /** Launch direction, radians (0 = right, -π/2 = up), and the cone around it; default a full circle. */
  readonly angle?: number;
  readonly spread?: number;
  /** Extra upward launch speed, px/s. */
  readonly lift?: number;
  readonly life: readonly [number, number];
  /** px/s²; negative rises (smoke). */
  readonly gravity?: number;
  /** Velocity lost per second, as a fraction. */
  readonly drag?: number;
  /** Particle size range, px (1–3). */
  readonly size?: readonly [number, number];
  /** RGBA colours: one is picked per particle, or with `ramp` they run head to tail over its life. */
  readonly colors: readonly number[];
  readonly ramp?: boolean;
  /** Alpha fades out over the last part of life. */
  readonly fade?: boolean;
  /** PARTICLE_* flags. */
  readonly flags?: number;
  /** Stops on terrain instead of passing through it. */
  readonly collide?: boolean;
  /** Spawn scatter radius, px. */
  readonly jitter?: number;
  /** A flash of light with the burst: RGB colour, radius px, intensity, seconds. */
  readonly light?: {
    readonly color: number;
    readonly radius: number;
    readonly intensity: number;
    readonly life: number;
  };
}

/* ---------------------------------------------------------------- art */

export type ArtSource = HTMLCanvasElement | OffscreenCanvas;

/** One painted frame and its pivot (the point placed at the sprite's position: feet for units). */
export interface ArtFrame {
  readonly canvas: ArtSource;
  readonly ox: number;
  readonly oy: number;
}

/** A set of frames painted at boot by one of the pixel-art painters (`art/`). */
export interface ArtDef {
  /** Paints every frame, in pose order; called once, in a browser, the first time the set is needed. */
  readonly paint: () => readonly ArtFrame[];
  /**
   * Frames per pose: 2 when the painter supplies a left-facing copy after each right-facing frame (the old
   * painters do), 1 when the renderer mirrors it.
   */
  readonly facings?: 1 | 2;
  /** Paint at boot rather than on first use (backdrops and anything drawn every frame). */
  readonly eager?: boolean;
}
