/**
 * Options and budgets for one battle. Budgets come from the device quality (`getQuality()` in a browser; the
 * Wallpaper shims it to a lighter one), so every budgeted system has a cap it never exceeds.
 */
import { qualityFor, type Quality } from "@/lib/device";
import type { BattleGame } from "./BattleGame";

export interface Budgets {
  /** Most non-squad units alive at once. */
  maxEnemies: number;
  /** Most CPU particles alive at once. */
  maxParticles: number;
  /** Clouds drifting across the sky. */
  clouds: number;
  /** Most projectiles in flight; past it, new shots are dropped. */
  maxProjectiles: number;
  /** Lights pushed per frame. */
  maxLights: number;
}

export function budgetsFromQuality(q: Quality): Budgets {
  return {
    maxEnemies: q.enemies,
    maxParticles: q.particles,
    clouds: q.clouds,
    maxProjectiles: Math.max(200, q.enemies * 4),
    maxLights: q.tier === "low" ? 32 : 64,
  };
}

/**
 * Knobs for individual features, keyed by feature. A feature adds its own by declaration merging in its own file:
 * `declare module "@/battle/game/options" { interface FeatureOptions { enemies?: EnemyOptions } }`.
 */
// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface FeatureOptions {}

/** A feature folder's entry point: registers content, adds systems and subscribes to events. */
export interface Feature {
  readonly name: string;
  readonly install: (game: BattleGame) => void;
}

export interface BattleOptions {
  /** Same seed, same battle. */
  seed?: number;
  /** View size in world px at zoom 1 (the renderer's internal size); on-screen rules use it. */
  viewW?: number;
  viewH?: number;
  budgets?: Partial<Budgets>;
  /** Per-system timings for the debug overlay (`?debug=1`); off, systems aren't timed. */
  debug?: boolean;
  /** Report a throwing system and carry on (`world.onError`), for long unattended runs. */
  unattended?: boolean;
  /** Seconds per day–night cycle (the old game's 240); and where in it the battle starts, 0..1. */
  dayLength?: number;
  timeOfDay?: number;
  /** The feature list to install, in order (default `INSTALLS`). */
  features?: readonly Feature[];
  /** Per-feature knobs. */
  feature?: FeatureOptions;
}

export interface ResolvedOptions {
  readonly seed: number;
  readonly viewW: number;
  readonly viewH: number;
  readonly budgets: Readonly<Budgets>;
  readonly debug: boolean;
  readonly unattended: boolean;
  readonly dayLength: number;
  readonly timeOfDay: number;
  readonly feature: FeatureOptions;
}

/** Headless and test defaults: the full ("high") budget and a 480×270 view, as at 4× on a 1080p screen. */
export function resolveOptions(o: BattleOptions): ResolvedOptions {
  return {
    seed: (o.seed ?? 1) | 0,
    viewW: o.viewW ?? 480,
    viewH: o.viewH ?? 270,
    budgets: { ...budgetsFromQuality(qualityFor("high")), ...o.budgets },
    debug: o.debug ?? false,
    unattended: o.unattended ?? false,
    dayLength: o.dayLength ?? 240,
    timeOfDay: o.timeOfDay ?? 0.15,
    feature: o.feature ?? {},
  };
}
