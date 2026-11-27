/**
 * The composition root of one battle: the world, the terrain, the content registries, the per-game resources and
 * budgets, and a small facade of the helpers content code calls most (spawn, hurt, explode, effect, banner), so a
 * unit or weapon stays a few lines. Everything that changes the simulation runs inside a system; the facade only
 * calls into the physics and render modules. Features plug in through `install(game)` (see `installs.ts`).
 */
import {
  clearFrame,
  indexOf,
  createWorld,
  stepFixed,
  Terrain,
  type AtlasBuilder,
  type CoreComponents,
  type RenderFrame,
  type Rng,
  type TerrainGenerator,
  type World,
} from "@/engine";
import { ArtBook } from "../art/ArtBook";
import { explode } from "../physics/explosions";
import { hurt, kill } from "../physics/damage";
import { fireWeapon, shoot, type ShotOptions } from "../physics/weapons";
import { spawnUnit, type SpawnOptions } from "../physics/spawn";
import { Banners, Popups, type BannerOptions } from "../render/banners";
import { Camera } from "../render/camera";
import { emitEffect, type EffectOptions } from "../render/effects";
import { FrameBuilder } from "../render/extract";
import { Flashes } from "../render/lights";
import { Particles } from "../render/particles";
import { ScreenFx } from "../render/screen";
import { SkyClock } from "../render/sky";
import { Content, type Links } from "./content";
import { defineGameComponents, UNIT_DEAD, type GameComponents } from "./components";
import { installCore, Timers } from "./core";
import type { BlastSpec, UnitDef } from "./defs";
import { definePooledEvents } from "./events";
import { INSTALLS } from "./installs";
import { Origin } from "./origin";
import { resolveOptions, type BattleOptions, type Budgets, type ResolvedOptions } from "./options";
import { TEAM_NEUTRAL } from "./teams";

/** Gameplay random streams; effects and anything cosmetic use `fx`, never these. */
export interface GameStreams {
  readonly ai: Rng;
  readonly spawn: Rng;
  readonly combat: Rng;
  readonly squad: Rng;
  readonly level: Rng;
  /** Cosmetic only (particles, flicker): never read by gameplay code. */
  readonly fx: Rng;
}

/** Running totals for the HUD and the site contract (`sceneState.kills`, `booms`). */
export interface BattleStats {
  kills: number;
  booms: number;
  shots: number;
  /** Units alive that count against the enemy budget, and projectiles in flight. */
  enemies: number;
  projectiles: number;
}

/** The level in play, and the slot its generator fills (the level feature sets it at install). */
export interface LevelState {
  no: number;
  name: string;
  /** Seconds since it started. */
  t: number;
  /** Fills new terrain columns; gets absolute columns (add `game.origin.cols` for the endless column). */
  generate: TerrainGenerator | null;
}

const errorReporter = () => {
  let reported = 0;
  return (err: unknown, where: string) => {
    // Unattended runs: say what broke, a few times, then stay quiet rather than flood the console every tick
    if (reported++ < 20) console.error(`battle: ${where} failed`, err);
  };
};

export class BattleGame {
  readonly options: ResolvedOptions;
  readonly budgets: Readonly<Budgets>;
  readonly world: World;
  readonly core: CoreComponents;
  readonly c: GameComponents;
  readonly terrain: Terrain;
  readonly content = new Content();
  readonly rng: GameStreams;
  readonly stats: BattleStats = { kills: 0, booms: 0, shots: 0, enemies: 0, projectiles: 0 };
  readonly level: LevelState = { no: 0, name: "", t: 0, generate: null };

  /* Per-game resources (render-facing ones are simulated in ticks and only read when drawing) */
  readonly camera: Camera;
  readonly sky: SkyClock;
  readonly particles: Particles;
  readonly flashes: Flashes;
  readonly screen = new ScreenFx();
  readonly banners = new Banners();
  readonly popups = new Popups();
  readonly origin: Origin;
  readonly timers = new Timers();
  readonly art: ArtBook;
  readonly frames = new FrameBuilder();

  private readonly bootHooks: ((game: BattleGame) => void)[] = [];
  private booted = false;

  constructor(opts: BattleOptions = {}) {
    const o = (this.options = resolveOptions(opts));
    this.budgets = o.budgets;
    const world = (this.world = createWorld({ seed: o.seed, capacity: 4096 }));
    world.stats.timing = o.debug;
    if (o.unattended) world.onError = errorReporter();
    definePooledEvents(world.events);
    this.core = world.core;
    this.c = defineGameComponents(world);
    const r = world.rng;
    this.rng = {
      ai: r.get("ai"),
      spawn: r.get("spawn"),
      combat: r.get("combat"),
      squad: r.get("squad"),
      level: r.get("level"),
      fx: r.get("fx"),
    };
    this.terrain = new Terrain({
      seed: o.seed,
      generate: (t, from, to) => (this.level.generate ? this.level.generate(t, from, to) : undefined),
    });
    this.camera = new Camera(o.viewW, o.viewH);
    this.sky = new SkyClock(o.dayLength, o.timeOfDay);
    this.particles = new Particles(o.budgets.maxParticles);
    this.flashes = new Flashes(96);
    this.origin = new Origin(this);
    this.art = new ArtBook(this.content.art);

    installCore(this);
    for (const f of opts.features ?? INSTALLS) f.install(this);
    this.content.freeze();
    this.booted = true;
    for (const hook of this.bootHooks) hook(this);
    this.bootHooks.length = 0;
    world.flush();
  }

  /** Resolved def references (after boot). */
  get links(): Links {
    return this.content.links;
  }

  /** Simulation seconds since the battle began. */
  get now(): number {
    return this.world.time.now;
  }

  /** Run once content is frozen and linked: spawning the first units, starting the level. */
  onBoot(hook: (game: BattleGame) => void): void {
    if (this.booted) hook(this);
    else this.bootHooks.push(hook);
  }

  /* ---------------------------------------------------------------- lifecycle */

  /** Advance by `realDt` real seconds × speed × slow-motion, in fixed ticks. Returns the ticks run. */
  step(realDt: number): number {
    return stepFixed(this.world, realDt * this.world.time.scale);
  }

  /** Fill `frame` (from `createFrame()`) for drawing, interpolated `alpha` (0..1) of the way into the next tick. */
  buildFrame(frame: RenderFrame, alpha = this.world.time.alpha): void {
    clearFrame(frame);
    this.frames.build(this, frame, alpha);
  }

  /** The view in world px at zoom 1 (the renderer's internal size); on-screen rules follow it. */
  resize(viewW: number, viewH: number, zoom = 1): void {
    this.camera.resize(viewW, viewH, zoom);
  }

  /** Paint the art book into the renderer's atlas (browser only) and fill the terrain's tile table. */
  paintArt(atlas: AtlasBuilder): void {
    this.art.attach(atlas, this.terrain);
  }

  dispose(): void {
    this.world.destroyAll();
    this.world.flush();
    this.world.events.clear();
    this.art.detach();
    this.particles.clear();
  }

  /* ---------------------------------------------------------------- facade */

  /** A unit by def id or index at (x, feet y); 0 when the enemy budget is full (unless forced). */
  spawn(def: string | number, x: number, y: number, opts?: SpawnOptions): number {
    return spawnUnit(this, def, x, y, opts);
  }

  /** Damage a unit; true when this killed it. */
  hurt(target: number, dmg: number, source = 0): boolean {
    return hurt(this, target, dmg, source);
  }

  /** Bring a unit down now, flung by (vx, vy). */
  kill(target: number, source = 0, vx = 0, vy = -120): void {
    kill(this, target, source, vx, vy);
  }

  /** An explosion owned by `source` on `team` (its own team is spared unless the spec is friendly). */
  explode(x: number, y: number, spec: BlastSpec, source = 0, team = TEAM_NEUTRAL): void {
    explode(this, x, y, spec, source, team);
  }

  /** A projectile by def id or index; 0 when the projectile budget is full. */
  shoot(def: string | number, x: number, y: number, vx: number, vy: number, opts?: ShotOptions): number {
    return shoot(this, def, x, y, vx, vy, opts);
  }

  /** Fire a unit's weapon at a point now, ignoring its cooldown. */
  fire(shooter: number, aimX: number, aimY: number): void {
    fireWeapon(this, shooter, aimX, aimY);
  }

  /** Particles (and light) from an effect def. */
  effect(def: string | number, x: number, y: number, opts?: EffectOptions): void {
    emitEffect(this, def, x, y, opts);
  }

  /** A short light (muzzle flash, blast): RGB colour, radius px, intensity, seconds. */
  flash(x: number, y: number, radius: number, color: number, intensity: number, life: number): void {
    this.flashes.add(x, y, radius, color, intensity, life);
  }

  /** Shake the camera by about `amount` px. */
  shake(amount: number): void {
    this.camera.shake = Math.max(this.camera.shake, amount);
  }

  /** Queue a banner (title text across the screen). */
  banner(text: string, sub = "", color = 0xffffffff, opts?: BannerOptions): void {
    this.banners.push(text, sub, color, opts);
  }

  /** A short floating label at a world point ("P2 DOWN", "!"). */
  popup(text: string, x: number, y: number, color = 0xffffffff): void {
    this.popups.add(text, x, y, color);
  }

  /** Run `fn` after `delay` simulation seconds (inside a tick, in the "late" phase). */
  at(delay: number, fn: (game: BattleGame) => void): void {
    this.timers.add(this.now + delay, fn);
  }

  /** A unit's def. */
  unitDef(id: number): UnitDef {
    return this.content.units.get(this.core.kind.def[indexOf(id)]);
  }

  /** Alive and still in the fight (not down, not a corpse). */
  fighting(id: number): boolean {
    return this.world.has(id, this.c.unit) && (this.c.unit.flags[indexOf(id)] & UNIT_DEAD) === 0;
  }
}
