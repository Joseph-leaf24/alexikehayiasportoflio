/**
 * The game's events, added to the engine's `WorldEvents` by declaration merging. Hot ones (kills, hurts, blasts,
 * shots, spawns) are pooled: emit them with `world.events.claim(type)`, which returns null when nobody listens, and
 * copy what you keep in a listener, because the payload object is reused after delivery. A feature adds its own
 * events the same way, in its own file.
 */
import type { EventBus, WorldEvents } from "@/engine";

export interface KillEvent {
  id: number;
  /** UnitDef index. */
  def: number;
  team: number;
  killer: number;
  /** The killer's player id (0–7), or -1. */
  player: number;
  x: number;
  y: number;
  score: number;
}

export interface HurtEvent {
  id: number;
  dmg: number;
  source: number;
  /** Health left. */
  hp: number;
}

export interface BlastEvent {
  x: number;
  y: number;
  r: number;
  source: number;
  team: number;
  /** Terrain cells destroyed. */
  cells: number;
}

export interface FireEvent {
  id: number;
  /** WeaponDef index. */
  weapon: number;
  x: number;
  y: number;
  aimX: number;
  aimY: number;
}

export interface SpawnEvent {
  id: number;
  def: number;
  team: number;
}

export interface PlayerEvent {
  id: number;
  /** 0–7. */
  player: number;
  x: number;
  y: number;
}

export interface BannerEvent {
  text: string;
  sub: string;
  color: number;
}

export interface LevelEvent {
  no: number;
  name: string;
  /** Seconds it lasted (levelEnd), 0 at the start. */
  time: number;
}

export interface OriginShiftEvent {
  /** World px every position moved by (negative: the world slid left). */
  dx: number;
  /** Terrain columns, dx / 8. */
  cols: number;
}

declare module "@/engine/core/events" {
  interface WorldEvents {
    unitSpawned: SpawnEvent;
    unitHurt: HurtEvent;
    unitKilled: KillEvent;
    weaponFired: FireEvent;
    playerDown: PlayerEvent;
    playerRespawned: PlayerEvent;
    blast: BlastEvent;
    banner: BannerEvent;
    levelStart: LevelEvent;
    levelEnd: LevelEvent;
    /** Delivered at once (dispatched), so state holding world x can move before anything reads it. */
    originShift: OriginShiftEvent;
  }
}

/** Register the pools for the hot event types. */
export function definePooledEvents(bus: EventBus<WorldEvents>): void {
  bus.define("unitKilled", () => ({ id: 0, def: 0, team: 0, killer: 0, player: -1, x: 0, y: 0, score: 0 }));
  bus.define("unitHurt", () => ({ id: 0, dmg: 0, source: 0, hp: 0 }));
  bus.define("blast", () => ({ x: 0, y: 0, r: 0, source: 0, team: 0, cells: 0 }));
  bus.define("weaponFired", () => ({ id: 0, weapon: 0, x: 0, y: 0, aimX: 0, aimY: 0 }));
  bus.define("unitSpawned", () => ({ id: 0, def: 0, team: 0 }));
}
