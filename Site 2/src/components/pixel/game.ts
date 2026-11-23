/**
 * The game playing out behind the site. Eight AI players push right through a level
 * (level.ts) in a random theme (themes.ts): they run, jump gaps, wade rivers, climb walls and
 * cliffs, shoot, throw grenades and parachute back in when they go down. Enemies shoot back
 * from bunkers, towers, temples, machine-gun nests, jeeps, boats and helicopters; attack dogs
 * run at them, riot shields turn their bullets, flamethrowers and grenadiers close in, trucks
 * unload reinforcements and drone swarms sweep over. Barrels and fuel tankers chain-react;
 * artillery, jets, paratroopers and lightning come down. Half way there may be a boss (a siege
 * tank, a walker, a gunship or a burrowing worm); at the end the squad calls ICBMs down on the
 * enemy stronghold, the whole scene goes up, the screen fades to white for "Level complete",
 * and the next random level fades in.
 * Plain logic with no drawing (PixelScene renders it), so it's unit-tested.
 */
import {
  AIR,
  BASE_ROW,
  blastCells,
  cell,
  createLevel,
  damageCell,
  extend,
  groundOf,
  lavaAt,
  LEVEL_H,
  lineOfSight,
  solidAt,
  surfaceY,
  TILE,
  waterAt,
  type Level,
  type LevelOptions,
  type Spawn,
} from "./level";
import { nextTheme, THEMES, type Backdrop, type BossKind, type Theme } from "./themes";
import { rng } from "./rng";

const GRAV = 620;
const RUN = 56;
const JUMP = 200;
const CLIMB = 70;
export const PLAYER_HP = 6;
export const SQUAD_SIZE = 8;
/** Where each player runs relative to the squad's advance point (the flamethrower leads, the sniper hangs back). */
const OFFSETS = [36, 48, 10, -16, -42, 62, -4, 22];
/** Weapon range per player: rifle, shotgun, plasma, rockets, sniper, flamethrower, grenade launcher, minigun. */
const RANGE = [170, 115, 160, 190, 300, 72, 210, 165];
/** Player height (head to feet) and half width. */
const PH = 15;
const PW = 3;

export interface Body {
  x: number;
  /** Feet */
  y: number;
  vx: number;
  vy: number;
  ground: boolean;
  facing: 1 | -1;
}

export interface Player extends Body {
  id: number;
  hp: number;
  dead: boolean;
  deadT: number;
  /** Invulnerable after a hit or a respawn */
  inv: number;
  /** Floating down under a parachute */
  chute: boolean;
  climbing: boolean;
  fireCd: number;
  burst: number;
  specialCd: number;
  scanT: number;
  target: Enemy | Barrel | null;
  stuckT: number;
  /** Seconds without getting anywhere (long enough: respawn) */
  stall: number;
  lastX: number;
  anim: number;
  spin: number;
  /** Recoil kick for the sprite (seconds left) */
  kick: number;
  score: number;
  kills: number;
}

export type EnemyKind =
  | "mook"
  | "heavy"
  | "rocket"
  | "sniper"
  | "mortar"
  | "mg"
  | "turret"
  | "crawler"
  | "dog"
  | "shield"
  | "flamer"
  | "grenadier"
  | "drone"
  | "jeep"
  | "truck"
  | "boat"
  | "heli"
  | BossKind;
export const BOSS_KINDS: BossKind[] = ["tank", "mech", "gunship", "worm"];
export const BOSS_NAMES: Record<BossKind, string> = { tank: "Siege tank", mech: "Walker", gunship: "Gunship", worm: "Burrower" };
const isBoss = (k: EnemyKind): k is BossKind => (BOSS_KINDS as string[]).includes(k);
/** Things in the air: no gravity, shot at without needing a clear line, crash when downed. */
const FLYING = new Set<EnemyKind>(["heli", "gunship", "drone"]);
/** Vehicles a blast damages rather than flings. */
const ARMOURED = new Set<EnemyKind>(["heli", "jeep", "truck", "boat", "mg"]);

export interface Segment {
  x: number;
  y: number;
  /** Above ground (visible, can be hit) */
  up: boolean;
}

export interface Enemy extends Body {
  kind: EnemyKind;
  boss: boolean;
  hp: number;
  maxHp: number;
  alive: boolean;
  deadT: number;
  spin: number;
  fireCd: number;
  altCd: number;
  burst: number;
  scanT: number;
  target: Player | null;
  alert: boolean;
  home: number;
  anim: number;
  hit: number;
  /** Helicopters: seconds left before they fly off, and their hover height */
  life: number;
  hover: number;
  /** Which theme's uniform it wears */
  look: Backdrop;
  /** Floating down under a parachute (reinforcements) */
  chute: boolean;
  /** Snipers: seconds spent lining up the shot (the laser sight is on) */
  aim: number;
  /** Where the gun points (radians), for turrets, tanks and walkers */
  aimA: number;
  /** The worm's body */
  segs: Segment[];
  /** Contact-damage cooldown (crawlers, the worm) */
  bite: number;
}

export type BarrelKind = "barrel" | "fuel" | "tanker" | "ammo" | "egg";

export interface Barrel extends Body {
  type: BarrelKind;
  hp: number;
  /** Seconds until it goes off (-1: not lit) */
  fuse: number;
  owner: number;
  alive: boolean;
}

/* Shot kinds */
export const BULLET = 0;
export const PELLET = 1;
export const PLASMA = 2;
export const ROCKET = 3;
export const GRENADE = 4;
export const SHELL = 5;
export const METEOR = 6;
export const ENEMY_BULLET = 7;
export const SNIPE = 8;
export const FLAMESHOT = 9;
export const LOB = 10;
export const CANNON = 11;
export const MORTAR = 12;
export const FIREBALL = 13;

const EXPLOSIVE = new Set([ROCKET, GRENADE, SHELL, METEOR, LOB, CANNON, MORTAR, FIREBALL]);
/** Gravity per shot kind (px/s²): lobbed things arc, flames drift up */
const SHOT_GRAV: Record<number, number> = { [GRENADE]: GRAV, [LOB]: 500, [CANNON]: 300, [MORTAR]: 400, [FIREBALL]: 350, [FLAMESHOT]: -70 };
const BLAST_R: Record<number, number> = { [ROCKET]: 18, [GRENADE]: 22, [SHELL]: 18, [METEOR]: 24, [LOB]: 20, [CANNON]: 24, [MORTAR]: 18, [FIREBALL]: 16 };

export interface Shot {
  kind: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  dmg: number;
  /** Who fired: player 0–7; -1 enemies, -2 the visitor's clicks, -3 the level (artillery), -4 the ICBMs */
  pid: number;
  pierce: number;
}

/* Particle kinds */
export const DEBRIS = 0;
export const SPARK = 1;
export const SMOKE = 2;
export const EMBER = 3;
export const CASING = 4;
export const FLAME = 5;
export const ASHFALL = 6;
export const FIREFLY = 7;
export const SNOWFLAKE = 8;
export const RAIN = 9;
export const SANDGRAIN = 10;
export const SPORE = 11;
export const BIGSMOKE = 12;
/** Spray at the foot of a waterfall */
export const MIST = 13;

export interface Particle {
  k: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  max: number;
  c: string;
  s: number;
}

export interface Blast {
  x: number;
  y: number;
  r: number;
  t: number;
}

export interface Popup {
  x: number;
  y: number;
  text: string;
  c: string;
  t: number;
}

/** An ICBM coming in on the stronghold (or, `up`, an enemy missile leaving a silo). */
export interface Missile {
  x: number;
  y: number;
  tx: number;
  ty: number;
  speed: number;
  up: boolean;
}

export interface Banner {
  text: string;
  sub: string;
  /** A third, smaller line (what lies ahead) */
  info?: string;
  color: string;
  /** Seconds shown (it waits while negative) */
  t: number;
}

/** One finished level, for the "Level complete" card and the spectator log. */
export interface LevelReport {
  no: number;
  name: string;
  kills: number;
  /** Seconds it took */
  time: number;
}

/** How the level's boss fight went. */
export type BossState = "none" | "ahead" | "engaged" | "destroyed" | "bypassed";

/** What the camera sees (world px). */
export interface View {
  x: number;
  y: number;
  w: number;
  h: number;
}

export type Phase = "play" | "boss" | "nuke" | "clear";

export interface GameOptions {
  seed?: number;
  maxEnemies?: number;
  maxParticles?: number;
  /** Player colours, for score popups */
  colors?: string[];
  /** Level options for the first level (tests) */
  level?: LevelOptions;
}

const SCORE: Record<EnemyKind, number> = {
  mook: 100,
  heavy: 300,
  rocket: 150,
  sniper: 200,
  mortar: 200,
  mg: 300,
  turret: 250,
  crawler: 50,
  dog: 80,
  shield: 250,
  flamer: 200,
  grenadier: 150,
  drone: 120,
  jeep: 500,
  truck: 800,
  boat: 900,
  heli: 1000,
  tank: 5000,
  mech: 6000,
  gunship: 5500,
  worm: 6000,
};
const HP: Record<EnemyKind, number> = {
  mook: 2,
  heavy: 9,
  rocket: 3,
  sniper: 2,
  mortar: 3,
  mg: 12,
  turret: 6,
  crawler: 2,
  dog: 2,
  shield: 4,
  flamer: 4,
  grenadier: 3,
  drone: 2,
  jeep: 14,
  truck: 20,
  boat: 22,
  heli: 24,
  tank: 240,
  mech: 300,
  gunship: 260,
  worm: 170,
};
/** Half width and height of each body, for collisions and hits. */
export const SIZE: Record<EnemyKind | BarrelKind | "player", [number, number]> = {
  player: [PW, PH],
  mook: [3, 15],
  heavy: [5, 17],
  rocket: [3, 15],
  sniper: [3, 15],
  mortar: [5, 10],
  mg: [7, 12],
  turret: [6, 8],
  crawler: [5, 7],
  dog: [5, 8],
  shield: [4, 15],
  flamer: [3, 15],
  grenadier: [3, 15],
  drone: [6, 5],
  jeep: [14, 14],
  truck: [20, 18],
  boat: [18, 10],
  heli: [16, 12],
  tank: [28, 26],
  mech: [16, 56],
  gunship: [30, 16],
  worm: [9, 9],
  barrel: [3, 8],
  fuel: [7, 11],
  tanker: [19, 17],
  ammo: [4, 6],
  egg: [4, 9],
};
const BARREL_BLAST: Record<BarrelKind, number> = { barrel: 20, fuel: 34, tanker: 60, ammo: 16, egg: 0 };

const sign = (v: number) => (v < 0 ? -1 : 1);
const clamp = (v: number, a: number, b: number) => (v < a ? a : v > b ? b : v);
/** Launch velocity for a lobbed shot from (x0, y0) to land on (x1, y1) under gravity g. */
function lobAt(x0: number, y0: number, x1: number, y1: number, g: number, speed: number) {
  const t = clamp(Math.abs(x1 - x0) / speed, 0.4, 1.6);
  return { vx: (x1 - x0) / t, vy: (y1 - y0 - 0.5 * g * t * t) / t };
}
/** Centre of a body's hit box (a boat's y is its water line). */
const centreY = (e: Enemy) => (FLYING.has(e.kind) ? e.y : e.y - SIZE[e.kind][1] / 2);

export class Game {
  level: Level;
  players: Player[];
  enemies: Enemy[] = [];
  barrels: Barrel[] = [];
  shots: Shot[] = [];
  parts: Particle[] = [];
  blasts: Blast[] = [];
  popups: Popup[] = [];
  missiles: Missile[] = [];
  /** Mushroom clouds and shockwaves from the ICBMs */
  mushrooms: { x: number; y: number; t: number }[] = [];
  waves: { x: number; y: number; t: number }[] = [];
  /** Short-lived fires left by explosions */
  fires: { x: number; y: number; t: number }[] = [];
  jets: { x: number; y: number; vx: number; bombs: number; dropT: number }[] = [];
  /** Muzzle flashes */
  flashes: { x: number; y: number; t: number; c: string }[] = [];
  /** Blocks that changed (column, row pairs) and columns that were generated, for the renderer */
  dirty: number[] = [];
  fresh: number[] = [];
  /** A whole new level: the renderer redraws everything */
  reset = false;
  /** Where the squad is pushing towards (world px). It starts a screen into the landing zone. */
  advance = 260;
  t = 0;
  kills = 0;
  booms = 0;
  levelNo = 1;
  phase: Phase = "play";
  phaseT = 0;
  banner: Banner | null = null;
  /** Flash from each ICBM hit (0..1, decays) */
  flash = 0;
  /**
   * The level's end: the whole screen fades to white (0 → 1), holds while the "Level complete"
   * card is up, and fades back out over the next level.
   */
  white = 0;
  private whiteUp = false;
  /** The card shown on the white-out at the end of a level (null otherwise) */
  complete: LevelReport | null = null;
  /** The levels finished so far (most recent first, the last six) and how many */
  log: LevelReport[] = [];
  cleared = 0;
  /** The boss this level holds (null: none), and how that fight is going */
  bossKind: BossKind | null = null;
  bossState: BossState = "none";
  /** Lightning strikes in a storm (x in world px, seconds since the strike) */
  bolts: { x: number; t: number; seed: number }[] = [];
  private killsAt = 0;
  /** Seconds since this level began (the drop-in helicopter flies over at the start) */
  levelT = 0;
  /** Multiplies enemies, artillery and the rest (the page sets it: deeper sections are wilder) */
  intensity = 1;
  /** Camera shake the renderer should apply (it decays it) */
  shake = 0;
  private rand: () => number;
  private holdT = 0;
  private holdAt = Infinity;
  private timeline: { at: number; fn: () => void }[] = [];
  private timers = { artillery: 2, meteor: 3, heli: 12, para: 6, jet: 8, silo: 6, drone: 10, bolt: 4 };
  private weather = 0;
  private lastIcbm = false;
  private lastView: View = { x: 0, y: 0, w: 400, h: 300 };
  private maxEnemies: number;
  private maxParticles: number;
  private colors: string[];

  constructor(theme: Theme, opts: GameOptions = {}) {
    this.rand = rng((opts.seed ?? 1) * 7919 + 3);
    this.maxEnemies = opts.maxEnemies ?? 70;
    this.maxParticles = opts.maxParticles ?? 2600;
    this.colors = opts.colors ?? ["#4aa3ff", "#ffcc33", "#c77dff", "#ff6a3d", "#5fd16a", "#3fd8e0", "#ff6fae", "#e8e2d0"];
    this.level = createLevel(theme, opts.seed ?? 1, opts.level);
    this.players = Array.from({ length: SQUAD_SIZE }, (_, id) => ({
      id,
      x: this.advance + OFFSETS[id],
      y: BASE_ROW * TILE,
      vx: 0,
      vy: 0,
      ground: true,
      facing: 1 as const,
      hp: PLAYER_HP,
      dead: false,
      deadT: 0,
      inv: 0,
      chute: false,
      climbing: false,
      fireCd: 0.2 * id,
      burst: 0,
      specialCd: 3 + id,
      scanT: 0,
      target: null,
      stuckT: 0,
      stall: 0,
      lastX: 0,
      anim: id,
      spin: 0,
      kick: 0,
      score: 0,
      kills: 0,
    }));
    this.meetBoss();
    this.say(`Level ${this.levelNo}`, theme.name, "#ffcc33", this.briefing());
  }

  get theme() {
    return this.level.theme;
  }

  /** Enemies down since this level began. */
  get levelKills() {
    return this.kills - this.killsAt;
  }

  /** How far through the level the squad is (0 at the drop, 1 at the stronghold). */
  get progress() {
    if (this.phase === "nuke" || this.phase === "clear") return 1;
    const end = this.level.plan.stronghold * TILE - 120;
    return clamp((this.advance - 260) / Math.max(1, end - 260), 0, 1);
  }

  /** The squad's current orders, for the spectator panel. */
  get objective() {
    const boss = this.bossKind ? BOSS_NAMES[this.bossKind] : "";
    switch (this.phase) {
      case "boss":
        return `Destroy the ${boss}`;
      case "nuke":
        return "ICBMs inbound - danger close";
      case "clear":
        return "Level complete";
      default:
        return this.bossState === "ahead" ? `Push on - ${boss} ahead` : "Reach the enemy stronghold";
    }
  }

  /** Note which boss (if any) this level holds. */
  private meetBoss() {
    this.bossKind = this.level.plan.boss?.kind ?? null;
    this.bossState = this.bossKind ? "ahead" : "none";
  }

  /** The level banner's third line: what lies ahead. */
  private briefing() {
    return this.bossKind ? `Boss ahead: ${BOSS_NAMES[this.bossKind]}` : "Objective: the enemy stronghold";
  }

  solid = (x: number, y: number) => solidAt(this.level, x, y);

  /** Run `fn` in `delay` seconds of game time. */
  private at(delay: number, fn: () => void) {
    this.timeline.push({ at: this.t + delay, fn });
  }

  private say(text: string, sub: string, color: string, info?: string, delay = 0) {
    this.banner = { text, sub, info, color, t: -delay };
  }

  /* -------------------------------------------------------------- */
  /* Levels                                                          */
  /* -------------------------------------------------------------- */

  /** On to the next level: a fresh, random theme (or the one given), the squad parachuting in. */
  nextLevel(theme?: Theme, opts?: LevelOptions) {
    this.levelNo++;
    const th = theme ?? nextTheme(this.rand, this.theme.key);
    this.level = createLevel(th, Math.floor(this.rand() * 1e6) + 1, opts);
    this.enemies = [];
    this.barrels = [];
    this.shots = [];
    this.missiles = [];
    this.mushrooms = [];
    this.waves = [];
    this.fires = [];
    this.jets = [];
    this.flashes = [];
    this.popups = [];
    this.blasts = [];
    this.bolts = [];
    this.timeline = [];
    this.parts.length = 0;
    this.dirty.length = 0;
    this.fresh.length = 0;
    this.lastIcbm = false;
    this.advance = 260;
    this.phase = "play";
    this.phaseT = 0;
    this.holdAt = Infinity;
    this.levelT = 0;
    this.reset = true;
    // Out of the white: the new level fades in (a page change starts one this way too)
    this.white = 1;
    this.whiteUp = false;
    this.complete = null;
    this.killsAt = this.kills;
    this.meetBoss();
    for (const p of this.players) {
      Object.assign(p, { dead: false, hp: PLAYER_HP, inv: 2.5, chute: true, ground: false, vx: 0, vy: 0, stuckT: 0, stall: 0, spin: 0, target: null });
      p.x = this.advance + OFFSETS[p.id] + (this.rand() - 0.5) * 8;
      p.y = BASE_ROW * TILE - 90 - p.id * 9;
    }
    this.say(`Level ${this.levelNo}`, th.name, "#ffcc33", this.briefing(), 0.8);
  }

  /* -------------------------------------------------------------- */
  /* Tick                                                            */
  /* -------------------------------------------------------------- */

  /** Advance the game by `dt` seconds. With `frozen` only effects already in motion play out (reduced motion). */
  step(dt: number, view: View, frozen = false) {
    this.t += dt;
    this.levelT += dt;
    this.phaseT += dt;
    this.flash = Math.max(0, this.flash - dt * 1.1);
    // (rises over 0.6 s, so it's a fade rather than a flash; falls over about a second)
    this.white = this.whiteUp ? Math.min(1, this.white + dt / 0.6) : Math.max(0, this.white - dt / 1.1);
    if (this.banner && (this.banner.t += dt) > 2.6) this.banner = null;
    for (const b of this.bolts) b.t += dt;
    if (this.bolts.length) this.bolts = this.bolts.filter((b) => b.t < 0.5);
    this.lastView = view;
    const L = this.level;
    L.intensity = this.intensity * (1 + 0.1 * Math.min(this.levelNo - 1, 6));

    // Keep the level built a little past the right edge
    const before = L.next;
    const spawns = extend(L, Math.floor((view.x + view.w) / TILE) + 40);
    for (let c = before; c < L.next; c++) this.fresh.push(c);
    for (const s of spawns) this.spawn(s);

    // Scheduled events (the ICBM sequence)
    if (this.timeline.length) {
      const due = this.timeline.filter((e) => e.at <= this.t);
      this.timeline = this.timeline.filter((e) => e.at > this.t);
      for (const e of due) {
        e.fn();
        if (this.reset) return;
      }
    }

    if (!frozen) {
      this.flow(view);
      this.moveSquad(dt);
      for (const p of this.players) this.updatePlayer(p, dt, view);
      for (const e of this.enemies) this.updateEnemy(e, dt, view);
      this.ambient(dt, view);
    }
    for (const b of this.barrels) this.updateBarrel(b, dt);
    this.updateShots(dt, view);
    this.updateMissiles(dt);
    this.updateEffects(dt, view);

    // Forget what fell behind
    const left = view.x - 140;
    this.enemies = this.enemies.filter((e) => (e.alive || e.deadT > 0) && (e.x > left || FLYING.has(e.kind)) && e.y < LEVEL_H + 80);
    this.barrels = this.barrels.filter((b) => b.alive && b.x > left);
  }

  /** The level's beats: the boss when the squad reaches the arena, the ICBMs at the stronghold. */
  private flow(view: View) {
    const plan = this.level.plan;
    if (this.phase === "play" && plan.boss && plan.bossPlaced && this.advance > plan.boss.col * TILE - 150) {
      this.spawnBoss(plan.boss.kind, view);
      this.holdAt = plan.boss.col * TILE - 40;
      this.phase = "boss";
      this.phaseT = 0;
      this.bossState = "engaged";
      plan.boss = null;
    } else if (this.phase === "boss" && (!this.enemies.some((e) => e.boss && e.alive) || this.phaseT > 45)) {
      this.phase = "play";
      this.holdAt = Infinity;
      this.bossState = this.phaseT <= 45 ? "destroyed" : "bypassed";
      if (this.phaseT <= 45) this.say("Boss down", "", "#5fd16a");
    } else if (this.phase === "play" && plan.fortPlaced && this.advance > plan.targets[0] - 200) this.startNuke(view);
  }

  /** The squad's advance point creeps forward, holding while enemies are close ahead (but never for long). */
  private moveSquad(dt: number) {
    const alive = this.players.filter((p) => !p.dead);
    const front = alive.length ? Math.max(...alive.map((p) => p.x)) : this.advance;
    const threat = this.enemies.some((e) => e.alive && !e.boss && !FLYING.has(e.kind) && e.x > front - 10 && e.x < front + 80);
    this.holdT = threat ? this.holdT + dt : 0;
    if (!threat || this.holdT > 4) this.advance += this.theme.speed * dt;
    // Never far ahead of the squad as a whole (stragglers still catch up: see updatePlayer)
    const mean = alive.length ? alive.reduce((s, p) => s + p.x, 0) / alive.length : this.advance;
    this.advance = Math.min(this.advance, front + 50, mean + 40, this.holdAt);
  }

  private spawn(s: Spawn) {
    if (s.kind === "barrel" || s.kind === "fuel" || s.kind === "tanker" || s.kind === "ammo" || s.kind === "egg") {
      const hp = { barrel: 1, fuel: 4, tanker: 8, ammo: 1, egg: 3 }[s.kind];
      this.barrels.push({ x: s.x, y: s.y, vx: 0, vy: 0, ground: true, facing: -1, type: s.kind, hp, fuse: -1, owner: -3, alive: true });
      return;
    }
    if (this.enemies.length >= this.maxEnemies) return;
    let y = s.y;
    while (y > TILE && this.solid(s.x, y - 1)) y -= TILE;
    this.enemies.push(this.makeEnemy(s.kind, s.x, y));
  }

  private makeEnemy(kind: EnemyKind, x: number, y: number): Enemy {
    const boss = isBoss(kind);
    const hp = HP[kind] * (boss ? Math.min(2, 1 + 0.12 * (this.levelNo - 1)) : 1);
    return {
      kind,
      boss,
      x,
      y,
      vx: 0,
      vy: 0,
      ground: true,
      facing: -1,
      hp,
      maxHp: hp,
      alive: true,
      deadT: 0,
      spin: 0,
      fireCd: 0.6 + this.rand(),
      altCd: 1.5 + this.rand(),
      burst: 0,
      scanT: this.rand() * 0.3,
      target: null,
      alert: false,
      home: x,
      anim: this.rand() * 4,
      hit: 0,
      life: 22 + this.rand() * 8,
      hover: 0,
      look: this.theme.look,
      chute: false,
      aim: 0,
      aimA: Math.PI,
      segs: [],
      bite: 0,
    };
  }

  private spawnBoss(kind: BossKind, view: View) {
    const x = kind === "worm" ? this.advance + 190 : view.x + view.w + 40;
    const e = this.makeEnemy(kind, x, surfaceY(this.level, x, view.y));
    if (kind === "gunship") {
      e.y = view.y + 10;
      e.hover = 48;
    }
    if (kind === "worm") {
      e.home = x;
      e.y += 30;
      e.segs = Array.from({ length: 14 }, () => ({ x, y: e.y, up: false }));
    }
    e.alert = true;
    this.enemies.push(e);
    this.say("Warning!", BOSS_NAMES[kind], "#ff3b3b");
    this.shake = Math.max(this.shake, 3);
  }

  /** The end of the level: hold, call the ICBMs down on the stronghold, then everything goes. */
  private startNuke(view: View) {
    this.phase = "nuke";
    this.phaseT = 0;
    const targets = this.level.plan.targets;
    this.holdAt = targets[0] - 200;
    this.say("Missiles inbound", "Danger close", "#ff3b3b");
    targets.forEach((tx, i) => this.at(1.4 + i * 0.55, () => this.launchICBM(tx, view, i === targets.length - 1)));
  }

  private launchICBM(tx: number, view: View, last: boolean) {
    const ty = surfaceY(this.level, tx, view.y);
    this.missiles.push({ x: tx + 240 + this.rand() * 60, y: view.y - 200, tx, ty, speed: 260, up: false });
    if (last) this.lastIcbm = true;
  }

  private detonate(x: number, y: number) {
    this.explode(x, y, 64, -4, false);
    this.mushrooms.push({ x, y, t: 0 });
    this.waves.push({ x, y, t: 0 });
    this.flash = Math.max(this.flash, 0.6);
    this.shake = 12;
    for (const e of this.enemies) if (e.alive && Math.abs(e.x - x) < 220) this.hurtEnemy(e, 999, -4, e.x - x);
    if (this.lastIcbm && !this.missiles.some((m) => !m.up && m.tx !== x)) {
      this.lastIcbm = false;
      this.blowUpEverything();
    }
  }

  /**
   * After the last ICBM: the whole scene goes up, the screen fades to white, "Level complete"
   * comes up on the white with the level's numbers, then the next level fades in.
   */
  private blowUpEverything() {
    const v = this.lastView;
    for (let k = 0; k < 18; k++)
      this.at(0.15 + k * 0.09, () => {
        const x = Math.max(this.advance + 70, v.x + v.w * (0.25 + this.rand() * 0.8));
        this.explode(x, surfaceY(this.level, x, v.y) - this.rand() * 40, 18 + this.rand() * 18, -4, false);
      });
    this.at(0.4, () => {
      for (const e of this.enemies) if (e.alive) this.hurtEnemy(e, 999, -4, 1);
      for (const b of this.barrels) this.lightBarrel(b, -4, 0.1 + this.rand() * 0.6);
    });
    this.at(1.1, () => (this.whiteUp = true));
    this.at(1.8, () => {
      this.phase = "clear";
      this.banner = null;
      const report = { no: this.levelNo, name: this.theme.name, kills: this.levelKills, time: this.levelT };
      this.complete = report;
      this.log = [report, ...this.log].slice(0, 6);
      this.cleared++;
      for (const p of this.players) p.score += 5000;
    });
    this.at(5.2, () => this.nextLevel());
  }

  /* -------------------------------------------------------------- */
  /* Bodies                                                          */
  /* -------------------------------------------------------------- */

  /** Gravity, then move along x and y, stopping at blocks. */
  private integrate(b: Body, dt: number, hw: number, hh: number, gravity = 1, maxFall = 420) {
    b.vy = Math.min(b.vy + GRAV * gravity * dt, maxFall);
    if (b.vx) {
      const nx = b.x + b.vx * dt;
      const side = nx + sign(b.vx) * hw;
      if (!this.solid(side, b.y - 2) && !this.solid(side, b.y - hh + 2)) b.x = nx;
      else b.vx = 0;
    }
    const ny = b.y + b.vy * dt;
    if (b.vy > 0) {
      if (this.solid(b.x - hw + 1, ny) || this.solid(b.x + hw - 1, ny)) {
        b.y = Math.floor(ny / TILE) * TILE;
        b.vy = 0;
        b.ground = true;
      } else {
        b.y = ny;
        b.ground = false;
      }
    } else if (b.vy < 0) {
      if (this.solid(b.x, ny - hh)) b.vy = 0;
      else b.y = ny;
      b.ground = false;
    }
  }

  /** A body flung by a blast or a death: tumbles and bounces. */
  private tumble(b: Body & { spin: number }, dt: number, hw: number, hh: number) {
    const vx = b.vx;
    this.integrate(b, dt, hw, hh);
    if (b.ground) b.vx = vx * 0.6;
    else b.spin += dt * 14;
  }

  /* -------------------------------------------------------------- */
  /* Players                                                         */
  /* -------------------------------------------------------------- */

  private updatePlayer(p: Player, dt: number, view: View) {
    if (p.dead) {
      p.deadT -= dt;
      this.tumble(p, dt, PW, PH);
      if (p.deadT <= 0) this.respawn(p, view);
      return;
    }
    p.inv = Math.max(0, p.inv - dt);
    p.kick = Math.max(0, p.kick - dt);
    if (p.x < view.x - 20 || p.y > LEVEL_H) {
      this.respawn(p, view);
      return;
    }

    p.scanT -= dt;
    if (p.scanT <= 0) {
      p.scanT = 0.15 + this.rand() * 0.1;
      p.target = this.findTarget(p);
    }
    const tgt = p.target?.alive ? p.target : null;

    const desired = this.advance + OFFSETS[p.id] + Math.sin(this.t * 0.8 + p.id * 1.7) * 5;
    const dx = desired - p.x;
    const dir = Math.abs(dx) > 4 ? sign(dx) : 0;
    p.facing = tgt ? (tgt.x >= p.x ? 1 : -1) : ((dir || p.facing) as 1 | -1);

    p.fireCd -= dt;
    if (tgt && p.fireCd <= 0 && !p.chute) this.fire(p, tgt);
    p.specialCd -= dt;
    if (tgt && p.specialCd <= 0 && Math.abs(tgt.x - p.x) < 130) {
      this.grenade(p, tgt.x);
      p.specialCd = 6 + this.rand() * 6;
    }

    // Run; jump walls and gaps; climb what's too tall to jump
    let vx = dir * RUN;
    // (looks further ahead than one step can reach, so a wall the step would hit is always seen first)
    const probe = p.x + dir * 6;
    const blocked = dir !== 0 && (this.solid(probe, p.y - 2) || this.solid(probe, p.y - PH + 3));
    p.climbing = false;
    if (blocked) {
      vx = 0;
      if (p.ground) {
        p.vy = -JUMP;
        p.ground = false;
      } else if (p.vy > -CLIMB) {
        p.vy = -CLIMB;
        p.climbing = true;
      }
    } else if (dir && p.ground) {
      const ahead = p.x + dir * 7;
      const drop = !this.solid(ahead, p.y + 2) && !this.solid(ahead, p.y + 10);
      if (drop || lavaAt(this.level, ahead, p.y + 6) || lavaAt(this.level, ahead, p.y + 12)) {
        p.vy = -JUMP;
        p.ground = false;
      }
    }
    p.vx = vx;
    const climbing = p.climbing;
    if (p.chute) {
      this.integrate(p, dt, PW, PH, 0.25, 55);
      if (p.ground) p.chute = false;
    } else this.integrate(p, dt, PW, PH);
    // Climbing into an overhang: count it as stuck straight away (the grenade comes sooner)
    if (climbing && p.vy === 0) p.stuckT += dt;
    p.anim += dt * (vx ? 12 : 0);
    // Wading through a river throws up spray
    if (vx && p.ground && waterAt(this.level, p.x, p.y - 3) && this.rand() < 0.35)
      this.add({ k: SPARK, x: p.x - p.facing * 3, y: p.y - 8, vx: -p.facing * 20 + (this.rand() - 0.5) * 30, vy: -50 - this.rand() * 40, life: 0.35, c: this.rand() < 0.5 ? "#e0f6ff" : "#8fd3ff", s: 1 });

    if (lavaAt(this.level, p.x, p.y - 2)) {
      this.burst(p.x, p.y - 4, 10, ["#ffd23f", "#ff8a1a"]);
      this.hurtPlayer(p, 99);
      return;
    }

    // Stuck against something: blow it up; still stuck after a while: drop back in
    if (dir && Math.abs(p.x - p.lastX) < 0.3) {
      p.stuckT += dt;
      p.stall += dt;
    } else p.stuckT = p.stall = 0;
    p.lastX = p.x;
    if (p.stuckT > 1.6) {
      this.grenade(p, p.x + dir * 24);
      p.stuckT = 0;
    }
    if (p.stall > 6) this.respawn(p, view);
  }

  /** Nearest enemy in range and in sight (lobbed weapons don't need sight); a barrel if an enemy stands next to it. */
  findTarget(p: Player): Enemy | Barrel | null {
    const range = RANGE[p.id];
    const gx = p.x;
    const gy = p.y - 9;
    const lobs = p.id === 6;
    let best: Enemy | Barrel | null = null;
    let bestD = range;
    for (const e of this.enemies) {
      if (!e.alive) continue;
      const aim = this.aimPoint(e);
      if (!aim) continue;
      const d = Math.hypot(aim.x - p.x, (aim.y - gy) * 0.6);
      // Prefer what's ahead (and bosses): things behind count as further away
      const score = (e.x < p.x - 10 ? d * 1.6 : d) * (e.boss ? 0.7 : 1);
      if (score >= bestD) continue;
      if (!lobs && !FLYING.has(e.kind) && !lineOfSight(this.level, gx, gy, aim.x, aim.y)) continue;
      best = e;
      bestD = score;
    }
    for (const b of this.barrels) {
      if (!b.alive || b.fuse >= 0 || b.type === "egg") continue;
      const d = Math.hypot(b.x - p.x, b.y - p.y);
      if (d > range * 0.9 || d > bestD + 20) continue;
      const near = this.enemies.some((e) => e.alive && !e.boss && Math.abs(e.x - b.x) < 30 && Math.abs(e.y - b.y) < 18);
      if (near && lineOfSight(this.level, gx, gy, b.x, b.y - 4)) return b;
    }
    return best;
  }

  /** Where to aim at an enemy (null: can't be hit right now, e.g. the worm underground). */
  aimPoint(e: Enemy): { x: number; y: number } | null {
    if (e.kind === "worm") {
      const s = e.segs.find((q) => q.up);
      return s ? { x: s.x, y: s.y } : null;
    }
    return { x: e.x, y: centreY(e) };
  }

  private fire(p: Player, tgt: Enemy | Barrel) {
    const gx = p.x + p.facing * 7;
    const gy = p.y - 9;
    const aim = "kind" in tgt ? (this.aimPoint(tgt) ?? { x: tgt.x, y: tgt.y }) : { x: tgt.x, y: tgt.y - SIZE[tgt.type][1] / 2 };
    const a = Math.atan2(aim.y - gy, aim.x - gx);
    const r = this.rand;
    const shot = (ang: number, speed: number, kind: number, dmg: number, life: number, pierce = 0) =>
      this.shots.push({ kind, x: gx, y: gy, vx: Math.cos(ang) * speed, vy: Math.sin(ang) * speed, life, dmg, pid: p.id, pierce });
    p.kick = 0.06;
    switch (p.id) {
      case 0: // rifle: bursts of five
        shot(a + (r() - 0.5) * 0.08, 400, BULLET, 1, 0.6);
        p.burst++;
        p.fireCd = p.burst % 5 === 0 ? 0.45 : 0.11;
        break;
      case 1: // shotgun
        for (let k = 0; k < 6; k++) shot(a + (r() - 0.5) * 0.6, 300 + r() * 60, PELLET, 1, 0.3);
        p.fireCd = 0.6;
        break;
      case 2: // plasma: punches through
        shot(a, 260, PLASMA, 2, 0.8, 2);
        p.fireCd = 0.3;
        break;
      case 3: // rockets
        shot(a, 140, ROCKET, 0, 1.6);
        p.fireCd = 1.1;
        this.puff(p.x - p.facing * 7, gy, 3);
        break;
      case 4: // sniper: one round through several
        shot(a, 800, SNIPE, 7, 0.5, 3);
        p.fireCd = 1.5;
        break;
      case 5: // flamethrower: a stream of fire
        for (let k = 0; k < 2; k++) shot(a + (r() - 0.5) * 0.3, 140 + r() * 40, FLAMESHOT, 0.45, 0.45, 2);
        p.fireCd = 0.05;
        break;
      case 6: {
        // grenade launcher: lobbed over cover
        const v = lobAt(gx, gy, aim.x + (r() - 0.5) * 10, aim.y, SHOT_GRAV[LOB], 190);
        this.shots.push({ kind: LOB, x: gx, y: gy, vx: v.vx, vy: v.vy, life: 2.5, dmg: 0, pid: p.id, pierce: 0 });
        p.fireCd = 0.85;
        break;
      }
      default: // minigun: long hosing bursts
        shot(a + (r() - 0.5) * 0.14, 380, BULLET, 0.6, 0.55);
        p.burst++;
        p.fireCd = p.burst % 24 === 0 ? 0.8 : 0.045;
    }
    if (p.id !== 5) this.flashes.push({ x: gx + p.facing * 2, y: gy, t: 0.05, c: this.colors[p.id] });
    if (p.id === 0 || p.id === 1 || p.id === 4 || p.id === 7) this.add({ k: CASING, x: p.x, y: gy, vx: -p.facing * (20 + r() * 30), vy: -60 - r() * 40, life: 0.8, c: "#ffd27a", s: 1 });
  }

  private grenade(p: Player, tx: number) {
    const dist = Math.max(30, Math.min(140, Math.abs(tx - p.x)));
    const dir = tx >= p.x ? 1 : -1;
    this.shots.push({ kind: GRENADE, x: p.x, y: p.y - 12, vx: dir * dist * 1.05, vy: -170, life: 1.1, dmg: 0, pid: p.id, pierce: 0 });
  }

  hurtPlayer(p: Player, dmg: number) {
    if (p.dead || (p.inv > 0 && dmg < 50)) return;
    p.hp -= dmg;
    p.inv = 0.4;
    if (p.hp > 0) return;
    p.dead = true;
    p.deadT = 2.6;
    p.vy = -170;
    p.vx = -p.facing * 70;
    p.spin = 0;
    p.chute = false;
    this.popups.push({ x: p.x, y: p.y - 20, text: `P${p.id + 1} DOWN`, c: this.colors[p.id], t: 0 });
  }

  /** Back in the fight: parachuting down near the squad. */
  private respawn(p: Player, view: View) {
    p.dead = false;
    p.hp = PLAYER_HP;
    p.inv = 1.5;
    p.x = Math.max(view.x + 24, Math.min(view.x + view.w - 24, this.advance + OFFSETS[p.id]));
    p.y = Math.max(8, view.y + 12);
    p.vx = 0;
    p.vy = 0;
    p.ground = false;
    p.chute = true;
    p.stuckT = p.stall = 0;
    p.spin = 0;
  }

  /* -------------------------------------------------------------- */
  /* Enemies                                                         */
  /* -------------------------------------------------------------- */

  private nearestPlayer(e: Enemy, range: number, sight: boolean, dy = 120): Player | null {
    let best: Player | null = null;
    let bestD = range;
    const ey = e.y - Math.min(14, SIZE[e.kind][1] - 4);
    for (const p of this.players) {
      if (p.dead) continue;
      const d = Math.abs(p.x - e.x);
      if (d >= bestD || Math.abs(p.y - e.y) > dy) continue;
      if (sight && !lineOfSight(this.level, e.x, ey, p.x, p.y - 9)) continue;
      best = p;
      bestD = d;
    }
    return best;
  }

  private updateEnemy(e: Enemy, dt: number, view: View) {
    e.hit = Math.max(0, e.hit - dt);
    e.bite = Math.max(0, e.bite - dt);
    if (!e.alive) {
      e.deadT -= dt;
      if (FLYING.has(e.kind)) return this.crash(e, dt);
      if (e.kind === "boat") {
        e.y += dt * 4;
        return;
      }
      if (!e.boss) this.tumble(e, dt, 3, 11);
      return;
    }
    switch (e.kind) {
      case "heli":
        return this.updateHeli(e, dt, view);
      case "gunship":
        return this.updateGunship(e, dt, view);
      case "drone":
        return this.updateDrone(e, dt, view);
      case "dog":
        return this.updateDog(e, dt);
      case "truck":
        return this.updateTruck(e, dt);
      case "boat":
        return this.updateBoat(e, dt);
      case "tank":
      case "mech":
        return this.updateWalker(e, dt);
      case "worm":
        return this.updateWorm(e, dt);
      case "crawler":
        return this.updateCrawler(e, dt);
      case "jeep":
        return this.updateJeep(e, dt);
    }
    const ranges: Partial<Record<EnemyKind, number>> = { heavy: 170, turret: 190, mg: 230, sniper: 290, mortar: 330, rocket: 210, flamer: 75, grenadier: 210, shield: 150 };
    const range = ranges[e.kind] ?? 150;
    e.scanT -= dt;
    if (e.scanT <= 0) {
      e.scanT = 0.25;
      // (lobbed weapons don't need a clear line)
      e.target = this.nearestPlayer(e, range, e.kind !== "mortar" && e.kind !== "grenadier", e.kind === "sniper" ? 200 : 120);
      if (e.target && !e.alert) {
        e.alert = true;
        this.popups.push({ x: e.x, y: e.y - SIZE[e.kind][1] - 6, text: "!", c: "#ff3b3b", t: 0 });
      }
    }
    const tgt = e.target && !e.target.dead ? e.target : null;
    e.vx = 0;
    if (tgt) {
      e.facing = tgt.x >= e.x ? 1 : -1;
      e.aimA = Math.atan2(tgt.y - 9 - (e.y - 8), tgt.x - e.x);
      e.fireCd -= dt;
      if (e.kind === "sniper") {
        // Line the shot up (the laser sight shows), then fire
        e.aim += dt;
        if (e.aim > 0.9 && e.fireCd <= 0) {
          this.enemyShot(e, tgt, 520, 2, 0.02);
          e.aim = 0;
          e.fireCd = 2.4;
        }
      } else if (e.fireCd <= 0) this.enemyFire(e, tgt);
      // Riflemen close in from far off; shields advance behind their shield; flamers rush in
      const gap = Math.abs(tgt.x - e.x);
      if (e.ground && !e.chute) {
        if (e.kind === "mook" && gap > 110) e.vx = e.facing * 16;
        else if (e.kind === "shield" && gap > 50) e.vx = e.facing * 11;
        else if (e.kind === "flamer" && gap > 50) e.vx = e.facing * 22;
      }
    } else {
      e.aim = 0;
      if (e.kind !== "turret" && e.kind !== "mg" && e.kind !== "mortar" && !e.chute) {
        // Patrol a few steps either side of where it was placed
        const ahead = e.x + e.facing * 5;
        if (Math.abs(ahead - e.home) > 18 || this.solid(ahead, e.y - 4) || !this.solid(ahead, e.y + 2)) e.facing = e.facing === 1 ? -1 : 1;
        else e.vx = e.facing * (e.kind === "heavy" ? 6 : 10);
      }
    }
    const [hw, hh] = SIZE[e.kind];
    if (e.chute) {
      e.vx = 0;
      this.integrate(e, dt, hw, hh, 0.25, 45);
      if (e.ground) e.chute = false;
    } else this.integrate(e, dt, hw, hh);
    e.anim += dt * (e.vx ? 8 : 0);
    if (lavaAt(this.level, e.x, e.y - 2)) this.killEnemy(e, -3, 0, -150);
  }

  /** One enemy bullet from `e` at `p`. */
  private enemyShot(e: Enemy, p: Player, speed: number, dmg: number, spread: number, fromY = 9) {
    const gx = e.x + e.facing * 7;
    const gy = e.kind === "heli" || e.kind === "gunship" ? e.y + 6 : e.kind === "drone" ? e.y + 3 : e.y - fromY;
    const a = Math.atan2(p.y - 9 - gy, p.x - gx) + (this.rand() - 0.5) * spread;
    this.shots.push({ kind: ENEMY_BULLET, x: gx, y: gy, vx: Math.cos(a) * speed, vy: Math.sin(a) * speed, life: 1.6, dmg, pid: -1, pierce: 0 });
    this.flashes.push({ x: gx, y: gy, t: 0.05, c: "#ff9a2a" });
  }

  private enemyFire(e: Enemy, p: Player) {
    const r = this.rand;
    if (e.kind === "rocket") {
      const a = Math.atan2(p.y - 9 - (e.y - 11), p.x - e.x);
      this.shots.push({ kind: ROCKET, x: e.x + e.facing * 6, y: e.y - 11, vx: Math.cos(a) * 110, vy: Math.sin(a) * 110, life: 2, dmg: 0, pid: -1, pierce: 0 });
      e.fireCd = 3 + r();
      return;
    }
    if (e.kind === "mortar") {
      const v = lobAt(e.x, e.y - 8, p.x + (r() - 0.5) * 40, p.y, SHOT_GRAV[MORTAR], 150);
      this.shots.push({ kind: MORTAR, x: e.x, y: e.y - 8, vx: v.vx, vy: v.vy, life: 4, dmg: 0, pid: -1, pierce: 0 });
      this.puff(e.x, e.y - 10, 3);
      e.fireCd = 3.4 + r();
      return;
    }
    if (e.kind === "flamer") {
      // A stream of fire, a second or so at a time
      const a = Math.atan2(p.y - 9 - (e.y - 9), p.x - e.x);
      for (let k = 0; k < 2; k++) {
        const ang = a + (r() - 0.5) * 0.3;
        const v = 130 + r() * 40;
        this.shots.push({ kind: FLAMESHOT, x: e.x + e.facing * 7, y: e.y - 9, vx: Math.cos(ang) * v, vy: Math.sin(ang) * v, life: 0.45, dmg: 1, pid: -1, pierce: 0 });
      }
      e.burst++;
      if (e.burst < 24) e.fireCd = 0.05;
      else {
        e.burst = 0;
        e.fireCd = 1.6 + r();
      }
      return;
    }
    if (e.kind === "grenadier") {
      // A grenade lobbed over cover: it bounces, then goes off
      const v = lobAt(e.x, e.y - 12, p.x + (r() - 0.5) * 24, p.y - 2, GRAV, 150);
      this.shots.push({ kind: GRENADE, x: e.x, y: e.y - 12, vx: v.vx, vy: v.vy, life: 1.5, dmg: 0, pid: -1, pierce: 0 });
      e.fireCd = 2.6 + r();
      return;
    }
    this.enemyShot(e, p, 170, 1, 0.18, e.kind === "turret" ? 5 : e.kind === "mg" ? 7 : 9);
    // Mooks and shields fire single shots; heavies, turrets and MG nests fire bursts
    const single = e.kind === "mook" || e.kind === "shield";
    const burstLen = e.kind === "heavy" ? 6 : e.kind === "mg" ? 8 : single ? 1 : 4;
    e.burst++;
    if (e.burst < burstLen) e.fireCd = e.kind === "mg" ? 0.07 : e.kind === "heavy" ? 0.09 : 0.12;
    else {
      e.burst = 0;
      e.fireCd = (single ? 1.1 : 1.5) * (0.8 + r() * 0.5);
    }
  }

  private updateDog(e: Enemy, dt: number) {
    // Runs the squad down and leaps at it
    const p = this.nearestPlayer(e, 230, false, 60);
    e.altCd -= dt;
    if (p) {
      e.facing = p.x >= e.x ? 1 : -1;
      if (!e.alert) {
        e.alert = true;
        this.popups.push({ x: e.x, y: e.y - 14, text: "!", c: "#ff3b3b", t: 0 });
      }
      e.vx = e.facing * 84;
      if (e.ground && e.altCd <= 0 && Math.abs(p.x - e.x) < 30) {
        e.vy = -150;
        e.altCd = 1.2;
      }
    } else e.vx = 0;
    if (e.vx && e.ground && this.solid(e.x + e.facing * 7, e.y - 3)) e.vy = -190;
    this.integrate(e, dt, 5, 8);
    e.anim += dt * (e.vx ? 16 : 0);
    if (p && e.bite <= 0 && Math.abs(p.x - e.x) < 9 && Math.abs(p.y - e.y) < 12) {
      this.hurtPlayer(p, 1);
      e.bite = 0.7;
    }
    if (lavaAt(this.level, e.x, e.y - 2)) this.killEnemy(e, -3, 0, -120);
  }

  private updateDrone(e: Enemy, dt: number, view: View) {
    // Hovers in a loose swarm over the front of the squad, firing and dropping grenades
    e.life -= dt;
    const tx = e.life > 0 ? this.advance + 50 + ((e.home * 37) % 130) : view.x + view.w + 90;
    const ty = view.y + 26 + e.hover + Math.sin(this.t * 3 + e.home) * 5;
    e.x += (tx - e.x) * (1 - Math.exp(-dt * 1.6));
    e.y += (ty - e.y) * (1 - Math.exp(-dt * 2));
    e.facing = tx < e.x ? -1 : 1;
    e.anim += dt;
    if (e.life < 0 && e.x > view.x + view.w + 50) {
      e.alive = false;
      e.deadT = 0;
      return;
    }
    e.fireCd -= dt;
    const p = this.players.find((q) => !q.dead && Math.abs(q.x - e.x) < 160);
    if (p && e.fireCd <= 0 && e.life > 0) {
      if (Math.abs(p.x - e.x) < 14 && this.rand() < 0.5)
        this.shots.push({ kind: GRENADE, x: e.x, y: e.y + 5, vx: 0, vy: 20, life: 1.2, dmg: 0, pid: -1, pierce: 0 });
      else this.enemyShot(e, p, 170, 1, 0.15);
      e.fireCd = 0.9 + this.rand() * 0.6;
    }
  }

  private updateTruck(e: Enemy, dt: number) {
    // Drives up, parks near the squad and the troops in the back pile out, one at a time
    e.vx = e.x > this.advance + 190 && e.x < this.advance + 400 ? -30 : 0;
    e.facing = -1;
    this.integrate(e, dt, 20, 18);
    e.anim += dt * Math.abs(e.vx) * 0.3;
    if (e.vx || e.x > this.advance + 260 || e.burst >= 5) return;
    e.altCd -= dt;
    if (e.altCd > 0 || this.enemies.length >= this.maxEnemies) return;
    e.altCd = 0.35;
    e.burst++;
    const n = this.rand();
    const m = this.makeEnemy(n < 0.2 ? "rocket" : n < 0.32 ? "heavy" : n < 0.45 ? "grenadier" : "mook", e.x + 18, e.y - 4);
    m.vy = -110;
    m.vx = 30;
    m.ground = false;
    m.alert = true;
    m.home = e.x + 30;
    this.enemies.push(m);
  }

  private updateBoat(e: Enemy, dt: number) {
    // Patrols up and down the river, its gunner hosing the banks
    // (rivers with a boat are at least eight blocks wide: it stays clear of the banks)
    if (Math.abs(e.x - e.home) > 10) e.facing = e.x > e.home ? -1 : 1;
    e.x += e.facing * 8 * dt;
    e.anim += dt;
    e.scanT -= dt;
    if (e.scanT <= 0) {
      e.scanT = 0.3;
      e.target = this.nearestPlayer(e, 220, true, 140);
    }
    e.fireCd -= dt;
    if (e.target && !e.target.dead && e.fireCd <= 0) {
      this.enemyShot(e, e.target, 180, 1, 0.2, 12);
      e.burst++;
      e.fireCd = e.burst % 6 === 0 ? 1.5 : 0.1;
    }
  }

  private updateCrawler(e: Enemy, dt: number) {
    // Scuttles straight at the nearest player and bites
    const p = this.nearestPlayer(e, 200, false, 80);
    e.vx = p ? sign(p.x - e.x) * 58 : 0;
    if (p) e.facing = p.x >= e.x ? 1 : -1;
    if (e.vx && e.ground && this.solid(e.x + e.facing * 7, e.y - 3)) e.vy = -170;
    this.integrate(e, dt, 5, 7);
    e.anim += dt * 14;
    if (p && e.bite <= 0 && Math.abs(p.x - e.x) < 8 && Math.abs(p.y - e.y) < 12) {
      this.hurtPlayer(p, 1);
      e.bite = 0.6;
    }
    if (lavaAt(this.level, e.x, e.y - 2)) this.killEnemy(e, -3, 0, -100);
  }

  private updateJeep(e: Enemy, dt: number) {
    // Drives at the squad, stops short and hoses it with the mounted gun
    const near = this.advance + 90;
    e.vx = e.x > near && e.x < this.advance + 320 ? -26 : 0;
    e.facing = -1;
    this.integrate(e, dt, 14, 14);
    e.anim += dt * Math.abs(e.vx) * 0.3;
    e.scanT -= dt;
    if (e.scanT <= 0) {
      e.scanT = 0.3;
      e.target = this.nearestPlayer(e, 210, true);
    }
    e.fireCd -= dt;
    if (e.target && !e.target.dead && e.fireCd <= 0) {
      this.enemyShot(e, e.target, 180, 1, 0.2, 16);
      e.burst++;
      e.fireCd = e.burst % 6 === 0 ? 1.4 : 0.09;
    }
  }

  private updateHeli(e: Enemy, dt: number, view: View) {
    e.life -= dt;
    const tx = e.life > 0 ? view.x + view.w * (0.68 + Math.sin(this.t * 0.4) * 0.1) : view.x + view.w + 120;
    e.x += (tx - e.x) * (1 - Math.exp(-dt * 1.2));
    e.y = view.y + e.hover + Math.sin(this.t * 2) * 3;
    e.facing = -1;
    if (e.life < 0 && e.x > view.x + view.w + 60) {
      e.alive = false;
      e.deadT = 0;
      return;
    }
    e.fireCd -= dt;
    const p = this.players.find((q) => !q.dead && Math.abs(q.x - e.x) < 220);
    if (p && e.fireCd <= 0 && e.life > 0) {
      this.enemyShot(e, p, 170, 1, 0.18);
      e.burst++;
      e.fireCd = e.burst % 4 === 0 ? 1.4 : 0.12;
    }
  }

  private updateGunship(e: Enemy, dt: number, view: View) {
    const tx = view.x + view.w * (0.7 + Math.sin(this.t * 0.35) * 0.12);
    e.x += (tx - e.x) * (1 - Math.exp(-dt * 0.9));
    e.y += (view.y + e.hover + Math.sin(this.t * 1.6) * 6 - e.y) * (1 - Math.exp(-dt * 2));
    e.facing = -1;
    const p = this.players.find((q) => !q.dead && Math.abs(q.x - e.x) < 300);
    if (!p) return;
    e.fireCd -= dt;
    e.altCd -= dt;
    if (e.fireCd <= 0) {
      this.enemyShot(e, p, 200, 1, 0.22);
      e.burst++;
      e.fireCd = e.burst % 8 === 0 ? 1.2 : 0.07;
    }
    if (e.altCd <= 0) {
      // A rocket salvo
      for (let k = 0; k < 4; k++)
        this.at(k * 0.14, () => {
          if (!e.alive) return;
          const a = Math.atan2(p.y - e.y, p.x - e.x) + (this.rand() - 0.5) * 0.3;
          this.shots.push({ kind: ROCKET, x: e.x - 10, y: e.y + 8, vx: Math.cos(a) * 120, vy: Math.sin(a) * 120, life: 2.5, dmg: 0, pid: -1, pierce: 0 });
        });
      e.altCd = 3.2;
    }
  }

  /** Tanks and walkers: roll or stride in, stop short of the squad, then cannon / rockets and machine guns. */
  private updateWalker(e: Enemy, dt: number) {
    const tank = e.kind === "tank";
    const stop = this.advance + (tank ? 170 : 150);
    e.vx = e.x > stop ? (tank ? -16 : -12) : 0;
    e.facing = -1;
    const [hw, hh] = SIZE[e.kind];
    const was = Math.floor(e.anim);
    this.integrate(e, dt, hw, hh);
    e.anim += dt * (e.vx ? (tank ? 6 : 2.2) : 0.2);
    // A walker's stride shakes the ground
    if (!tank && e.vx && Math.floor(e.anim) !== was) {
      this.shake = Math.max(this.shake, 2);
      this.puff(e.x - 10, e.y, 3);
      this.puff(e.x + 10, e.y, 3);
    }
    const p = this.nearestPlayer(e, 320, false, 200);
    if (!p) return;
    const gunY = e.y - (tank ? 20 : 38);
    e.aimA = Math.atan2(p.y - 9 - gunY, p.x - e.x);
    e.fireCd -= dt;
    e.altCd -= dt;
    if (e.fireCd <= 0) {
      this.enemyShot(e, p, 190, 1, 0.2, tank ? 16 : 34);
      e.burst++;
      e.fireCd = e.burst % (tank ? 4 : 7) === 0 ? 1.3 : 0.08;
    }
    if (e.altCd <= 0) {
      if (tank) {
        const mx = e.x - 30;
        const v = lobAt(mx, gunY - 2, p.x, p.y, SHOT_GRAV[CANNON], 220);
        this.shots.push({ kind: CANNON, x: mx, y: gunY - 2, vx: v.vx, vy: v.vy, life: 4, dmg: 0, pid: -1, pierce: 0 });
        this.flashes.push({ x: mx, y: gunY - 2, t: 0.12, c: "#ffe27a" });
        this.puff(mx, gunY, 5);
        this.shake = Math.max(this.shake, 2.5);
        e.altCd = 2.2;
      } else {
        for (let k = 0; k < 3; k++)
          this.at(k * 0.16, () => {
            if (!e.alive) return;
            const a = Math.atan2(p.y - 9 - (e.y - 50), p.x - e.x) + (this.rand() - 0.5) * 0.25;
            this.shots.push({ kind: ROCKET, x: e.x + 4, y: e.y - 50, vx: Math.cos(a) * 110, vy: Math.sin(a) * 110 - 40, life: 2.5, dmg: 0, pid: -1, pierce: 0 });
          });
        e.altCd = 3.2;
      }
    }
  }

  /** The worm swims through the ground in long arcs, bursting out to spit fire, and bites what it touches. */
  private updateWorm(e: Enemy, dt: number) {
    e.anim += dt * 1.5;
    e.home += (this.advance + 200 - e.home) * (1 - Math.exp(-dt * 0.4));
    const g = groundOf(this.level, Math.floor(e.home / TILE)) * TILE;
    const hx = e.home + Math.cos(e.anim * 0.7) * 70;
    const hy = g + 26 - Math.max(-0.4, Math.sin(e.anim)) * 120;
    e.facing = hx < e.x ? -1 : 1;
    e.x = hx;
    e.y = hy;
    // The body follows the head's path
    const segs = e.segs;
    segs[0].x = hx;
    segs[0].y = hy;
    for (let i = 1; i < segs.length; i++) {
      const a = segs[i - 1];
      const s = segs[i];
      const d = Math.hypot(a.x - s.x, a.y - s.y) || 1;
      if (d > 9) {
        s.x = a.x + ((s.x - a.x) / d) * 9;
        s.y = a.y + ((s.y - a.y) / d) * 9;
      }
    }
    for (const s of segs) s.up = !this.solid(s.x, s.y);
    // Underground it tunnels; breaking the surface throws dirt up
    if (!segs[0].up) {
      const c = Math.floor(hx / TILE);
      const r = Math.floor(hy / TILE);
      for (const [dc, dr] of [
        [0, 0],
        [1, 0],
        [-1, 0],
        [0, -1],
      ]) {
        const broke = damageCell(this.level, c + dc, r + dr, 99);
        if (broke) {
          this.dirty.push(c + dc, r + dr);
          this.dirtyAround(c + dc, r + dr);
          if (this.rand() < 0.5) this.debris(c + dc, r + dr, broke, 1);
        }
      }
    }
    e.fireCd -= dt;
    const p = this.nearestPlayer(e, 320, false, 220);
    if (p && segs[0].up && e.fireCd <= 0) {
      const v = lobAt(hx, hy, p.x, p.y, SHOT_GRAV[FIREBALL], 170);
      this.shots.push({ kind: FIREBALL, x: hx, y: hy, vx: v.vx, vy: v.vy, life: 3, dmg: 0, pid: -1, pierce: 0 });
      e.fireCd = 1.3;
    }
    for (const q of this.players) if (!q.dead && Math.hypot(q.x - hx, q.y - 8 - hy) < 13) this.hurtPlayer(q, 2);
  }

  private crash(e: Enemy, dt: number) {
    e.vy = Math.min(e.vy + GRAV * 0.5 * dt, 260);
    e.x += e.vx * dt;
    e.y += e.vy * dt;
    e.spin += dt * 6;
    if (this.rand() < 0.6) this.puff(e.x, e.y, 3);
    if (this.rand() < 0.4) this.flame(e.x, e.y);
    if (this.solid(e.x, e.y + 8) || e.y > LEVEL_H) {
      e.deadT = 0;
      const y = Math.min(e.y, LEVEL_H - 8);
      e.y = LEVEL_H + 200;
      this.explode(e.x, y, e.kind === "gunship" ? 44 : e.kind === "drone" ? 12 : 32, -3, true);
    }
  }

  private killEnemy(e: Enemy, pid: number, vx: number, vy: number) {
    if (!e.alive) return;
    e.alive = false;
    e.chute = false;
    e.deadT = FLYING.has(e.kind) ? 99 : e.boss || e.kind === "boat" ? 2 : 1.4;
    e.vx = vx;
    e.vy = vy;
    e.spin = 0;
    this.kills++;
    if (pid >= 0) {
      const p = this.players[pid];
      p.score += SCORE[e.kind];
      p.kills++;
      this.popups.push({ x: e.x, y: e.y - SIZE[e.kind][1] - 6, text: `+${SCORE[e.kind]}`, c: this.colors[pid], t: 0 });
    }
    if (e.boss) {
      // Bosses go up in a long chain of explosions, and everyone shares the bounty
      for (const q of this.players) q.score += Math.round(SCORE[e.kind] / SQUAD_SIZE);
      const [hw, hh] = SIZE[e.kind];
      const pts =
        e.kind === "worm"
          ? e.segs.map((s) => ({ x: s.x, y: s.y }))
          : Array.from({ length: 8 }, () => ({ x: e.x + (this.rand() - 0.5) * hw * 2, y: centreY(e) + (this.rand() - 0.5) * hh }));
      pts.forEach((q, i) => this.at(i * 0.12, () => this.explode(q.x, q.y, 16 + this.rand() * 10, pid, false)));
      this.at(pts.length * 0.12 + 0.1, () => this.explode(e.x, centreY(e), 44, pid, false));
      if (e.kind !== "gunship") this.at(pts.length * 0.12 + 0.2, () => this.fires.push({ x: e.x, y: e.y, t: 8 }));
      return;
    }
    if (e.kind === "turret" || e.kind === "heavy" || e.kind === "mg") this.explode(e.x, e.y - 4, e.kind === "heavy" ? 10 : 14, pid, false);
    if (e.kind === "jeep") this.explode(e.x, e.y - 8, 26, pid, true);
    if (e.kind === "truck") {
      this.explode(e.x, e.y - 10, 30, pid, true);
      this.fires.push({ x: e.x - 8, y: e.y, t: 5 }, { x: e.x + 10, y: e.y, t: 3 });
    }
    if (e.kind === "boat") this.explode(e.x, e.y - 6, 24, pid, false);
    if (e.kind === "flamer") {
      // The fuel tank on its back goes up
      this.explode(e.x, e.y - 8, 14, pid, false);
      this.fires.push({ x: e.x, y: e.y, t: 2 });
    }
    if (e.kind === "drone") this.burst(e.x, e.y, 8, ["#ffe27a", "#ff3b3b"]);
    if (e.kind === "crawler") this.burst(e.x, e.y - 4, 10, ["#8aff6a", "#3a8a2a"]);
  }

  /** Damage an enemy (bosses soak it up; anything else at `dmg` ≥ its health is thrown). */
  hurtEnemy(e: Enemy, dmg: number, pid: number, vx: number) {
    if (!e.alive) return;
    e.hp -= dmg;
    e.hit = 0.06;
    if (e.hp <= 0) this.killEnemy(e, pid, sign(vx) * 50, -110);
  }

  /* -------------------------------------------------------------- */
  /* Barrels, shots, explosions                                      */
  /* -------------------------------------------------------------- */

  private updateBarrel(b: Barrel, dt: number) {
    if (!b.alive) return;
    const [hw, hh] = SIZE[b.type];
    if (b.type === "egg") {
      // Hatches when the squad gets close
      if (this.players.some((p) => !p.dead && Math.abs(p.x - b.x) < 60 && Math.abs(p.y - b.y) < 30)) this.hatch(b, 2);
      return;
    }
    const vx = b.vx;
    this.integrate(b, dt, hw, hh);
    if (b.ground) b.vx = vx * 0.5;
    if (b.fuse >= 0) {
      b.fuse -= dt;
      if (this.rand() < 0.6) this.add({ k: FLAME, x: b.x + (this.rand() - 0.5) * hw * 2, y: b.y - hh, vx: 0, vy: -30, life: 0.3, c: "", s: 1 });
      if (b.fuse < 0) {
        b.alive = false;
        this.explode(b.x, b.y - hh / 2, BARREL_BLAST[b.type], b.owner, true);
        if (b.type === "ammo") this.burst(b.x, b.y - 6, 30, ["#ffe27a", "#ff9a2a", "#ffffff"]);
        if (b.type === "tanker") this.fires.push({ x: b.x, y: b.y, t: 6 }, { x: b.x - 20, y: b.y, t: 4 }, { x: b.x + 20, y: b.y, t: 4 });
      }
    }
  }

  private hatch(b: Barrel, n: number) {
    if (!b.alive) return;
    b.alive = false;
    this.burst(b.x, b.y - 6, 12, ["#c8ff8a", "#6a2a8a"]);
    for (let k = 0; k < n && this.enemies.length < this.maxEnemies; k++) {
      const e = this.makeEnemy("crawler", b.x + (k - 0.5) * 6, b.y);
      e.vy = -120;
      this.enemies.push(e);
    }
  }

  private lightBarrel(b: Barrel, owner: number, delay: number) {
    if (!b.alive || b.fuse >= 0) return;
    if (b.type === "egg") return this.hatch(b, 1);
    b.fuse = delay;
    b.owner = owner;
  }

  private updateShots(dt: number, view: View) {
    const L = this.level;
    const keep: Shot[] = [];
    for (const s of this.shots) {
      s.life -= dt;
      if (s.kind === ROCKET) {
        s.vx *= 1 + 2.2 * dt;
        s.vy *= 1 + 2.2 * dt;
        if (this.rand() < 0.7) this.puff(s.x, s.y, 1);
      } else if (s.kind === SHELL) s.vy = Math.min(s.vy + 300 * dt, 420);
      else if (SHOT_GRAV[s.kind]) s.vy += SHOT_GRAV[s.kind] * dt;
      if (s.kind === METEOR || s.kind === FIREBALL) {
        if (this.rand() < 0.8) this.add({ k: FLAME, x: s.x, y: s.y, vx: 0, vy: -10, life: 0.4, c: "", s: 2 });
      } else if (s.kind === FLAMESHOT) {
        this.add({ k: FLAME, x: s.x, y: s.y, vx: s.vx * 0.2, vy: -20, life: 0.25, c: "", s: 1 + Math.round(this.rand()) });
      } else if ((s.kind === MORTAR || s.kind === CANNON) && this.rand() < 0.4) this.puff(s.x, s.y, 1);
      const explosive = EXPLOSIVE.has(s.kind);
      if (s.life <= 0) {
        if (explosive) this.explodeShot(s);
        continue;
      }
      const nx = s.x + s.vx * dt;
      const ny = s.y + s.vy * dt;
      if (this.solid(nx, ny)) {
        if (s.kind === GRENADE) {
          if (this.solid(nx, s.y)) s.vx *= -0.45;
          if (this.solid(s.x, ny)) s.vy *= -0.4;
          s.vx *= 0.85;
          keep.push(s);
          continue;
        }
        if (explosive) {
          this.explodeShot(s, nx, ny);
          continue;
        }
        if (s.kind === FLAMESHOT) continue;
        // Bullets chip at blocks
        const c = Math.floor(nx / TILE);
        const r = Math.floor(ny / TILE);
        const broke = damageCell(L, c, r, s.kind === SNIPE ? 3 : s.dmg);
        if (broke) {
          this.dirty.push(c, r);
          this.debris(c, r, broke, 1);
          this.dirtyAround(c, r);
        }
        this.add({ k: SPARK, x: s.x, y: s.y, vx: -s.vx * 0.1 + (this.rand() - 0.5) * 40, vy: -40 * this.rand(), life: 0.15, c: "#ffe27a", s: 1 });
        continue;
      }
      s.x = nx;
      s.y = ny;
      if (this.hitBodies(s)) continue;
      if (s.x < view.x - 80 || s.x > view.x + view.w + 120 || s.y > LEVEL_H + 20 || s.y < view.y - 260) continue;
      keep.push(s);
    }
    this.shots = keep;
  }

  /** Does a point hit an enemy's body? */
  private hits(e: Enemy, x: number, y: number) {
    if (e.kind === "worm") return e.segs.some((s, i) => s.up && Math.hypot(s.x - x, s.y - y) < (i === 0 ? 11 : 8));
    const [hw, hh] = SIZE[e.kind];
    return Math.abs(e.x - x) <= hw + 1 && Math.abs(centreY(e) - y) <= hh / 2 + 1;
  }

  /** Did a shot hit someone? (true: it's used up) */
  private hitBodies(s: Shot) {
    if (s.pid === -1) {
      for (const p of this.players) {
        if (p.dead || Math.abs(p.x - s.x) > 4 || s.y < p.y - PH || s.y > p.y) continue;
        if (EXPLOSIVE.has(s.kind)) {
          this.explodeShot(s);
          return true;
        }
        this.hurtPlayer(p, s.dmg);
        this.add({ k: SPARK, x: s.x, y: s.y, vx: 0, vy: -30, life: 0.12, c: "#ffffff", s: 1 });
        return true;
      }
      return false;
    }
    if (s.kind === SHELL || s.kind === METEOR) return false;
    for (const e of this.enemies) {
      if (!e.alive || !this.hits(e, s.x, s.y)) continue;
      if (EXPLOSIVE.has(s.kind)) {
        this.explodeShot(s);
        return true;
      }
      // A riot shield turns bullets and pellets from the front (go round it, or use fire or explosives)
      if (e.kind === "shield" && (s.kind === BULLET || s.kind === PELLET) && sign(s.vx) === -e.facing) {
        for (let k = 0; k < 3; k++) this.add({ k: SPARK, x: s.x, y: s.y, vx: -s.vx * 0.15 + (this.rand() - 0.5) * 60, vy: -30 - this.rand() * 50, life: 0.2, c: k ? "#ffe27a" : "#ffffff", s: 1 });
        return true;
      }
      this.hurtEnemy(e, s.dmg, s.pid, s.vx);
      this.add({ k: SPARK, x: s.x, y: s.y, vx: -s.vx * 0.05, vy: -30, life: 0.12, c: e.kind === "crawler" || e.kind === "worm" ? "#8aff6a" : "#ffffff", s: 1 });
      if (s.pierce > 0) {
        s.pierce--;
        continue;
      }
      return true;
    }
    for (const b of this.barrels) {
      if (!b.alive) continue;
      const [hw, hh] = SIZE[b.type];
      if (Math.abs(b.x - s.x) > hw + 1 || s.y < b.y - hh || s.y > b.y) continue;
      if (EXPLOSIVE.has(s.kind)) {
        this.explodeShot(s);
        return true;
      }
      b.hp -= s.kind === FLAMESHOT ? 1 : s.dmg;
      if (b.hp <= 0) this.lightBarrel(b, s.pid, 0.05);
      return true;
    }
    return false;
  }

  private explodeShot(s: Shot, x = s.x, y = s.y) {
    const r = s.kind === ROCKET && s.pid === -1 ? 14 : (BLAST_R[s.kind] ?? 18);
    // Enemy fire and artillery hurt the squad; its own explosives, the visitor's and the ICBMs don't
    this.explode(x, y, r, s.pid, s.pid === -3 || s.pid === -1);
  }

  /** An explosion: breaks blocks, scorches what's left, throws everyone about, sets off barrels. */
  explode(x: number, y: number, r: number, pid: number, hurtsPlayers: boolean) {
    const L = this.level;
    this.booms++;
    // The ground takes less than the people do, so the level stays runnable (the visitor's own blasts dig full craters)
    const dig = pid === -2 ? r : pid === -4 ? r * 0.85 : r * 0.6;
    for (const [c, row] of blastCells(L, x, y, dig)) {
      const i = cell(c, row);
      const kind = L.fg[i];
      L.fg[i] = AIR;
      L.hp[i] = 0;
      this.dirty.push(c, row);
      this.dirtyAround(c, row);
      if (this.rand() < (r > 40 ? 0.3 : 1)) this.debris(c, row, kind, r > 40 ? 1 : 2);
    }
    for (const [c, row] of blastCells(L, x, y, dig + 8)) {
      const i = cell(c, row);
      if (L.burn[i] < 3) {
        L.burn[i]++;
        this.dirty.push(c, row);
      }
    }
    for (const e of this.enemies) {
      if (!e.alive) continue;
      const aim = this.aimPoint(e) ?? { x: e.x, y: e.y };
      const d = Math.hypot(aim.x - x, aim.y - y);
      const reach = r + (e.boss ? SIZE[e.kind][0] : 8);
      if (d > reach) continue;
      const f = 1 - d / reach;
      if (e.boss || ARMOURED.has(e.kind)) this.hurtEnemy(e, (e.boss ? 14 : 10) * f + 3, pid, e.x - x);
      else this.killEnemy(e, pid, sign(e.x - x) * (80 + 140 * f), -(140 + 160 * f));
    }
    for (const b of this.barrels) {
      if (!b.alive) continue;
      const d = Math.hypot(b.x - x, b.y - 4 - y);
      if (d > r + 10) continue;
      this.lightBarrel(b, pid, 0.1 + this.rand() * 0.3);
      if (b.type !== "tanker") {
        b.vy = -140 - this.rand() * 120;
        b.vx = sign(b.x - x) * (30 + this.rand() * 60);
      }
    }
    for (const p of this.players) {
      if (p.dead) continue;
      const d = Math.hypot(p.x - x, p.y - 8 - y);
      if (d > r + 4) continue;
      p.vx = sign(p.x - x) * 80;
      p.vy = Math.min(p.vy, -120);
      if (hurtsPlayers) this.hurtPlayer(p, 3);
    }
    this.blasts.push({ x, y, r: r * 0.9, t: 0 });
    const sparks = r > 40 ? 30 : 14;
    for (let n = 0; n < sparks; n++) {
      const a = this.rand() * Math.PI * 2;
      const v = 60 + this.rand() * (r > 40 ? 300 : 160);
      this.add({ k: SPARK, x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v - 50, life: 0.3 + this.rand() * 0.5, c: this.rand() < 0.5 ? "#ffe27a" : "#ff9a2a", s: 1 });
    }
    for (let n = 0; n < (r > 40 ? 12 : 6); n++) this.puff(x + (this.rand() - 0.5) * r, y + (this.rand() - 0.5) * r * 0.6, 2 + this.rand() * (r > 40 ? 8 : 3));
    if (this.rand() < 0.5) {
      const gy = groundOf(L, Math.floor(x / TILE)) * TILE;
      if (Math.abs(gy - y) < 40) this.fires.push({ x, y: gy, t: 1.5 + this.rand() * 2.5 });
    }
    this.shake = Math.min(pid === -2 ? 6 : pid === -4 ? 12 : 3, this.shake + r * (pid === -2 ? 0.2 : 0.05));
  }

  /** The visitor's click: a blast exactly where they clicked. */
  strike(x: number, y: number) {
    this.explode(x, y, 18, -2, false);
  }

  /** The visitor's airstrike: a shell falling from the top of the view. */
  dropShell(x: number, view: View) {
    this.shots.push({ kind: SHELL, x, y: view.y - 8, vx: 0, vy: 380, life: 3, dmg: 0, pid: -2, pierce: 0 });
  }

  private updateMissiles(dt: number) {
    const keep: Missile[] = [];
    for (const m of this.missiles) {
      m.speed = Math.min(m.speed + 520 * dt, 720);
      const dx = m.tx - m.x;
      const dy = m.ty - m.y;
      const d = Math.hypot(dx, dy) || 1;
      const step = m.speed * dt;
      // A thick trail of smoke and fire behind it
      for (let k = 0; k < 2; k++)
        this.add({ k: BIGSMOKE, x: m.x - (dx / d) * 10, y: m.y - (dy / d) * 10, vx: (this.rand() - 0.5) * 10, vy: (this.rand() - 0.5) * 10, life: 2.5 + this.rand(), c: "", s: 2 + this.rand() * 3 });
      this.add({ k: FLAME, x: m.x - (dx / d) * 8, y: m.y - (dy / d) * 8, vx: 0, vy: 0, life: 0.25, c: "", s: 2 });
      if (m.up) {
        m.y -= step;
        if (m.y > this.lastView.y - 300) keep.push(m);
        continue;
      }
      if (d <= step || this.solid(m.x + (dx / d) * step, m.y + (dy / d) * step)) {
        this.detonate(m.tx, m.ty);
        continue;
      }
      m.x += (dx / d) * step;
      m.y += (dy / d) * step;
      keep.push(m);
    }
    this.missiles = keep;
  }

  /* -------------------------------------------------------------- */
  /* Ambience and effects                                            */
  /* -------------------------------------------------------------- */

  private ambient(dt: number, view: View) {
    const s = this.theme;
    const r = this.rand;
    const T = this.timers;
    const k = this.intensity * (1 + 0.08 * Math.min(this.levelNo - 1, 6));
    const calm = this.phase === "nuke" || this.phase === "clear";
    if (!calm && s.artillery > 0 && (T.artillery -= dt) <= 0) {
      T.artillery = (0.4 + r() * 1.2) / (s.artillery * k);
      this.shots.push({ kind: SHELL, x: view.x + view.w * (0.4 + r() * 0.7), y: view.y - 10, vx: -15, vy: 260, life: 4, dmg: 0, pid: -3, pierce: 0 });
    }
    if (s.meteors > 0 && (T.meteor -= dt) <= 0) {
      T.meteor = (0.5 + r()) / (s.meteors * k);
      this.shots.push({ kind: METEOR, x: view.x + view.w * (0.5 + r() * 0.8), y: view.y - 20, vx: -90, vy: 230, life: 4, dmg: 0, pid: -3, pierce: 0 });
    }
    if (!calm && s.paras > 0 && (T.para -= dt) <= 0) {
      T.para = (40 + r() * 40) / (s.paras * k);
      // Reinforcements: a stick of paratroopers ahead of the squad
      for (let n = 2 + Math.floor(r() * 3); n > 0 && this.enemies.length < this.maxEnemies; n--) {
        const e = this.makeEnemy(r() < 0.2 ? "heavy" : r() < 0.3 ? "rocket" : "mook", view.x + view.w * (0.55 + r() * 0.45), view.y - 10 - n * 16);
        e.chute = true;
        e.ground = false;
        this.enemies.push(e);
      }
    }
    if (s.jets > 0 && (T.jet -= dt) <= 0) {
      T.jet = (40 + r() * 40) / (s.jets * k);
      this.jets.push({ x: view.x + view.w + 30, y: view.y + 14 + r() * 24, vx: -240 - r() * 60, bombs: calm ? 0 : 5 + Math.floor(r() * 5), dropT: 0.3 });
    }
    for (const j of this.jets) {
      j.x += j.vx * dt;
      j.dropT -= dt;
      if (j.bombs > 0 && j.dropT <= 0 && j.x < view.x + view.w * 0.95 && j.x > view.x + 20) {
        j.bombs--;
        j.dropT = 0.11 + r() * 0.1;
        this.shots.push({ kind: SHELL, x: j.x, y: j.y + 3, vx: j.vx * 0.25, vy: 40, life: 4, dmg: 0, pid: -3, pierce: 0 });
      }
    }
    this.jets = this.jets.filter((j) => j.x > view.x - 80);
    if (!calm && s.helis > 0 && (T.heli -= dt) <= 0) {
      T.heli = (40 + r() * 40) / (s.helis * k);
      if (this.enemies.filter((e) => e.kind === "heli" && e.alive).length < 2) {
        const h = this.makeEnemy("heli", view.x + view.w + 40, view.y + 40);
        h.hover = 34 + r() * 40;
        this.enemies.push(h);
      }
    }
    // Drone swarms sweep in from ahead
    if (!calm && s.drones > 0 && (T.drone -= dt) <= 0) {
      T.drone = (40 + r() * 40) / (s.drones * k);
      if (this.enemies.filter((e) => e.kind === "drone" && e.alive).length < 6)
        for (let n = 2 + Math.floor(r() * 3); n > 0 && this.enemies.length < this.maxEnemies; n--) {
          const d = this.makeEnemy("drone", view.x + view.w + 20 + n * 18, view.y + 30 + r() * 30);
          d.home = n * 2 + Math.floor(r() * 10);
          d.hover = r() * 40;
          d.life = 18 + r() * 10;
          d.ground = false;
          this.enemies.push(d);
        }
    }
    // Lightning in a storm (a few seconds apart: a flash in the sky, never a strobe)
    if (s.ambient === "storm" && (T.bolt -= dt) <= 0) {
      T.bolt = 5 + r() * 9;
      this.bolts.push({ x: view.x + view.w * (0.1 + r() * 0.8), t: 0, seed: Math.floor(r() * 1e6) });
      this.shake = Math.max(this.shake, 1.5);
    }
    // Missile silos send one up now and then
    if ((T.silo -= dt) <= 0) {
      T.silo = 4 + r() * 6;
      const silo = this.level.props.find((p) => p.kind === "silo" && p.x > view.x && p.x < view.x + view.w);
      if (silo) this.missiles.push({ x: silo.x, y: silo.y, tx: silo.x, ty: -999, speed: 80, up: true });
    }
    // Weather (particles per second)
    const rate = { none: 0, fireflies: 3, smoke: 4, embers: 14, ash: 28, snow: 34, rain: 60, storm: 90, sand: 34, spores: 10 }[s.ambient];
    this.weather += rate * dt;
    for (; this.weather >= 1; this.weather--) {
      const x = view.x + r() * view.w;
      switch (s.ambient) {
        case "fireflies":
          this.add({ k: FIREFLY, x, y: view.y + view.h * (0.3 + r() * 0.4), vx: (r() - 0.5) * 8, vy: (r() - 0.5) * 6, life: 2 + r() * 2, c: "#d8ff7a", s: 1 });
          break;
        case "ash":
          this.add({ k: ASHFALL, x, y: view.y - 4, vx: -10 - r() * 10, vy: 14 + r() * 12, life: 8, c: r() < 0.3 ? "#ff8a3a" : "#8a8088", s: 1 });
          break;
        case "embers":
          this.add({ k: EMBER, x, y: view.y + view.h, vx: (r() - 0.5) * 10, vy: -20 - r() * 20, life: 4 + r() * 2, c: "", s: 1 });
          break;
        case "snow":
          this.add({ k: SNOWFLAKE, x: x + 40, y: view.y - 4, vx: -12 - r() * 14, vy: 18 + r() * 16, life: 12, c: r() < 0.3 ? "#ffffff" : "#d8e4f4", s: r() < 0.2 ? 2 : 1 });
          break;
        case "rain":
        case "storm":
          this.add({ k: RAIN, x: x + 60, y: view.y - 4, vx: s.ambient === "storm" ? -110 : -60, vy: 320 + r() * 60, life: 1.4, c: "#8ab0d0", s: 1 });
          break;
        case "sand":
          this.add({ k: SANDGRAIN, x: view.x + view.w + 4, y: view.y + r() * view.h, vx: -160 - r() * 80, vy: (r() - 0.5) * 20, life: 4, c: r() < 0.5 ? "#e8c890" : "#c8a060", s: 1 });
          break;
        case "spores":
          this.add({ k: SPORE, x, y: view.y + view.h, vx: (r() - 0.5) * 10, vy: -10 - r() * 14, life: 8, c: r() < 0.5 ? "#8aff6a" : "#d88aff", s: 1 });
          break;
        default:
          this.add({ k: SMOKE, x, y: view.y + view.h * 0.7, vx: 4, vy: -6, life: 3, c: "", s: 3 + r() * 3 });
      }
    }
    // Burning wrecks, campfires and braziers; spray at the foot of waterfalls
    for (const p of this.level.props) {
      if (p.x < view.x - 20 || p.x > view.x + view.w + 20) continue;
      if ((p.kind === "fire" || p.kind === "brazier") && r() < 0.5) this.flame(p.x, p.kind === "brazier" ? p.y - 8 : p.y);
      else if (p.kind === "waterfall" && r() < 0.7)
        this.add({ k: MIST, x: p.x + r() * p.w, y: p.y + p.h - 2, vx: (r() - 0.5) * 16 - 4, vy: -8 - r() * 14, life: 1.2 + r(), c: "", s: 2 + r() * 2 });
    }
  }

  private updateEffects(dt: number, view: View) {
    for (const f of this.fires) {
      f.t -= dt;
      if (this.rand() < 0.6) this.flame(f.x + (this.rand() - 0.5) * 10, f.y);
    }
    this.fires = this.fires.filter((f) => f.t > 0 && f.x > view.x - 40);
    for (const b of this.blasts) b.t += dt;
    this.blasts = this.blasts.filter((b) => b.t < 0.5);
    for (const m of this.mushrooms) {
      m.t += dt;
      if (m.t < 3 && this.rand() < 0.8)
        this.add({ k: BIGSMOKE, x: m.x + (this.rand() - 0.5) * 20, y: m.y - this.rand() * Math.min(1, m.t) * 110, vx: (this.rand() - 0.5) * 12, vy: -10, life: 2 + this.rand() * 2, c: "", s: 3 + this.rand() * 4 });
    }
    this.mushrooms = this.mushrooms.filter((m) => m.t < 6);
    for (const w of this.waves) w.t += dt;
    this.waves = this.waves.filter((w) => w.t < 1);
    for (const f of this.flashes) f.t -= dt;
    this.flashes = this.flashes.filter((f) => f.t > 0);
    for (const p of this.popups) {
      p.t += dt;
      p.y -= 14 * dt;
    }
    this.popups = this.popups.filter((p) => p.t < 1);

    const parts = this.parts;
    for (let i = parts.length - 1; i >= 0; i--) {
      const p = parts[i];
      p.life -= dt;
      if (p.life <= 0 || p.x < view.x - 80 || p.y > view.y + view.h + 40) {
        parts[i] = parts[parts.length - 1];
        parts.pop();
        continue;
      }
      if (p.k === DEBRIS || p.k === CASING) {
        p.vy += 400 * dt;
        const nx = p.x + p.vx * dt;
        const ny = p.y + p.vy * dt;
        if (this.solid(nx, ny)) {
          p.vy *= -0.3;
          p.vx *= 0.5;
          if (Math.abs(p.vy) < 20) p.vy = p.vx = 0;
        } else {
          p.x = nx;
          p.y = ny;
        }
      } else {
        if (p.k === SPARK) p.vy += 160 * dt;
        else if (p.k === FIREFLY || p.k === SPORE) {
          p.vx += (this.rand() - 0.5) * 20 * dt;
          p.vy += (this.rand() - 0.5) * 20 * dt;
        } else if (p.k === SNOWFLAKE) p.vx += Math.sin(this.t * 2 + p.y * 0.05) * 6 * dt;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        if ((p.k === SNOWFLAKE || p.k === RAIN || p.k === ASHFALL) && this.solid(p.x, p.y)) p.life = 0;
      }
    }
  }

  private add(p: Omit<Particle, "max">) {
    if (this.parts.length < this.maxParticles) this.parts.push({ ...p, max: p.life });
  }

  private puff(x: number, y: number, s: number) {
    this.add({ k: SMOKE, x, y, vx: (this.rand() - 0.5) * 16, vy: -8 - this.rand() * 14, life: 1 + this.rand(), c: "", s });
  }

  private flame(x: number, y: number) {
    this.add({ k: FLAME, x: x + (this.rand() - 0.5) * 8, y, vx: (this.rand() - 0.5) * 10, vy: -26 - this.rand() * 20, life: 0.4 + this.rand() * 0.3, c: "", s: 1 + Math.round(this.rand()) });
  }

  private burst(x: number, y: number, n: number, colors: string[]) {
    for (let k = 0; k < n; k++) this.add({ k: SPARK, x, y, vx: (this.rand() - 0.5) * 140, vy: -this.rand() * 140, life: 0.5, c: colors[k % colors.length], s: 1 });
  }

  private debris(c: number, r: number, kind: number, n: number) {
    for (let k = 0; k < n; k++)
      this.add({ k: DEBRIS, x: c * TILE + this.rand() * TILE, y: r * TILE + this.rand() * TILE, vx: (this.rand() - 0.5) * 140, vy: -60 - this.rand() * 140, life: 1.2 + this.rand(), c: String(kind), s: this.rand() < 0.3 ? 2 : 1 });
  }

  private dirtyAround(c: number, r: number) {
    this.dirty.push(c - 1, r, c + 1, r, c, r - 1, c, r + 1);
  }
}

/** The theme for `?level=` (or the 404's cave), else a random one. */
export function startTheme(key: string | null, rand: () => number) {
  return (key && THEMES[key]) || nextTheme(rand);
}
