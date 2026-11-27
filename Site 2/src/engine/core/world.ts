/**
 * The world: entities and components (from `Ecs`) plus everything a tick needs — the core components, seeded
 * random streams, the event bus, resources, the fixed-step clock, the systems, the spatial hash and debug stats.
 */

import { defineCoreComponents, type CoreComponents } from "./components";
import { Ecs } from "./ecs";
import { EventBus, type WorldEvents } from "./events";
import { RngStreams, type Rng } from "./rng";
import { Scheduler, TICK, type System } from "./scheduler";
import { SpatialHash, type SpatialOptions } from "./spatial";
import { WorldStats } from "./stats";
import { installCoreSystems, type CoreSystemOptions } from "./systems";

/** A typed handle for a resource: `const CAMERA = resource<Camera>("camera")`, then `world.res(CAMERA)`. */
export interface ResourceKey<T> {
  readonly key: string;
  /** Type-only marker; never set. */
  readonly type?: T;
}
export const resource = <T>(key: string): ResourceKey<T> => ({ key });

export class WorldTime {
  /** Ticks completed. */
  tick = 0;
  /** Simulation seconds at the start of the current tick (tick × TICK, so it never drifts). */
  now = 0;
  /** Simulation seconds waiting to become ticks. */
  acc = 0;
  /** Fraction of a tick between the last tick and the frame being drawn, for interpolation. */
  alpha = 0;
  /** Game speed (1 = normal); the frame loop multiplies real time by `scale`. */
  speed = 1;
  /** Slow-motion factor set by the game (cinematics); 1 = none. */
  slowmo = 1;
  paused = false;

  /** Simulation seconds per real second: speed × slow-motion, or 0 when paused. */
  get scale(): number {
    return this.paused ? 0 : this.speed * this.slowmo;
  }

  readonly dt = TICK;
}

export interface WorldOptions {
  seed?: number;
  /** Initial entity capacity, default 4096 (grows by doubling; size it to the peak to keep columns fixed). */
  capacity?: number;
  spatial?: SpatialOptions;
  /** Install the core systems (life, spatial rebuild); default on. Pass options to tune them. */
  coreSystems?: boolean | CoreSystemOptions;
}

export class World extends Ecs {
  readonly seed: number;
  /** Independent random streams by name: `world.rng.get("ai")`. */
  readonly rng: RngStreams;
  /** The default simulation stream ("sim"). */
  readonly random: Rng;
  readonly events = new EventBus<WorldEvents>();
  readonly time = new WorldTime();
  readonly stats = new WorldStats();
  readonly systems = new Scheduler();
  readonly spatial: SpatialHash;
  readonly core: CoreComponents;
  /**
   * When set, a throwing system or listener is reported here and the tick carries on (unattended runs); unset,
   * errors propagate (tests, development).
   */
  onError: ((err: unknown, where: string) => void) | null = null;

  private readonly resources = new Map<string, unknown>();

  constructor(opts: WorldOptions = {}) {
    super(opts.capacity ?? 4096);
    this.seed = (opts.seed ?? 1) | 0;
    this.rng = new RngStreams(this.seed);
    this.random = this.rng.get("sim");
    this.spatial = new SpatialHash(opts.spatial);
    this.core = defineCoreComponents(this);
    this.systems.bind((list) => (this.stats.systems = list));
    this.onDestroy((id) => this.events.emit("destroyed", id));
    this.events.onError = (err, type) => {
      if (!this.onError) throw err;
      this.onError(err, `event:${String(type)}`);
    };
    const core = opts.coreSystems ?? true;
    if (core !== false) installCoreSystems(this, core === true ? {} : core);
  }

  addSystem(sys: System): System {
    return this.systems.add(sys);
  }

  /** A resource; throws when it was never set. */
  res<T>(key: string | ResourceKey<T>): T {
    const k = typeof key === "string" ? key : key.key;
    if (!this.resources.has(k)) throw new Error(`World: no resource "${k}"`);
    return this.resources.get(k) as T;
  }

  /** A resource, or undefined. */
  tryRes<T>(key: string | ResourceKey<T>): T | undefined {
    return this.resources.get(typeof key === "string" ? key : key.key) as T | undefined;
  }

  setRes<T>(key: string | ResourceKey<T>, value: T): T {
    this.resources.set(typeof key === "string" ? key : key.key, value);
    return value;
  }

  hasRes(key: string | ResourceKey<unknown>): boolean {
    return this.resources.has(typeof key === "string" ? key : key.key);
  }

  deleteRes(key: string | ResourceKey<unknown>): boolean {
    return this.resources.delete(typeof key === "string" ? key : key.key);
  }

  /** Deliver queued events and run the deferred destroys now (outside a tick: setup code and tests). */
  flush(): void {
    this.events.flush();
    this.flushDestroyed();
    this.events.flush();
  }
}

export const createWorld = (opts?: WorldOptions): World => new World(opts);
