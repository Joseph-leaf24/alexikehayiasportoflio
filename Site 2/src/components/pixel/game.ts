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
  ASH,
  kindAt,
  WATER,
  BASE_ROW,
  CONCRETE,
  DIRT,
  MUD,
  REDROCK,
  SAND,
  SANDSTONE,
  SNOW,
  STONE,
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
import { nextTheme, THEMES, themeDeck, variant, type Backdrop, type BossKind, type Theme } from "./themes";
import { asset } from "./assets";
import { rng } from "./rng";

const GRAV = 620;
const RUN = 56;
const JUMP = 200;
const CLIMB = 70;
export const PLAYER_HP = 6;
export const SQUAD_SIZE = 8;
/** Where each player runs relative to the squad's advance point (the flamethrower leads, the sniper hangs back). */
const OFFSETS = [36, 48, 10, -16, -42, 62, -4, 22];
/**
 * The fourteen weapons. Each player starts with a signature one (P1 rifle … P8 minigun) and
 * picks up the others from supply drops for a while.
 */
export const WEAPONS = ["Rifle", "Shotgun", "Plasma", "Rockets", "Sniper", "Flamethrower", "Grenades", "Minigun", "Laser", "Tesla", "Pistols", "Crossbow", "Saw blades", "Mortar"] as const;
/** Range per weapon. */
const W_RANGE = [170, 115, 160, 190, 300, 72, 210, 165, 230, 140, 150, 200, 150, 260];
/** Weapons that lob over cover (no clear line needed). */
const LOBBED = new Set([6, 13]);
/** How much more of everything there is than the themes ask for. */
const CHAOS = 3;
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
  /** The weapon in hand (an index into WEAPONS) and seconds left on a picked-up one */
  weapon: number;
  weaponT: number;
  /** Seconds of jetpack left */
  jet: number;
  /** Down, waiting for a transport to drop them back in */
  queued: boolean;
  /** Seconds left on fire after a warhead threw them (they trail flame as they fly) */
  burn: number;
  score: number;
  kills: number;
}

/** Anything an enemy can shoot at: a player or an allied vehicle. */
export type Foe = { x: number; y: number; dead: boolean; vx?: number };

/** The allies' war machines: tanks, fighting vehicles, rocket artillery, flak, attack helicopters, a supply lifter. */
export type AllyKind = "mbt" | "ifv" | "mlrs" | "aa" | "scoutcar" | "aheli" | "lifter";
export interface Ally extends Body {
  kind: AllyKind;
  hp: number;
  maxHp: number;
  dead: boolean;
  deadT: number;
  fireCd: number;
  altCd: number;
  burst: number;
  scanT: number;
  aimA: number;
  anim: number;
  /** Its place in the column */
  slot: number;
  hit: number;
  target: Enemy | null;
}
const ALLY_HP: Record<AllyKind, number> = { mbt: 40, ifv: 26, mlrs: 18, aa: 20, scoutcar: 18, aheli: 16, lifter: 24 };
export const ALLY_SIZE: Record<AllyKind, [number, number]> = { mbt: [20, 14], ifv: [19, 13], mlrs: [19, 14], aa: [18, 13], scoutcar: [16, 12], aheli: [15, 9], lifter: [22, 10] };
export const ALLY_AIR = new Set<AllyKind>(["aheli", "lifter"]);
/** Blast radius of a missile by size: small rockets, tactical missiles, cruise missiles, huge warheads. */
export const MISSILE_R = [16, 30, 52, 96];
/** The enemy's tanks and armour (driven by updateArmour). */
export const ARMOR_KINDS = new Set(["apc", "lighttank", "medtank", "heavytank", "flametank", "rockettank", "aatank", "launcher", "armoredcar", "artytruck", "hovercraft"]);
export type PlaneType = "bomber" | "fighter" | "attacker" | "transport";

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
  | "jetpacker"
  | "minigunner"
  | "officer"
  | "dynamiter"
  | "rusher"
  | "engineer"
  | "laser"
  | "exosuit"
  | "drone"
  | "jeep"
  | "truck"
  | "boat"
  | "battleship"
  | "apc"
  | "lighttank"
  | "medtank"
  | "heavytank"
  | "flametank"
  | "rockettank"
  | "aatank"
  | "launcher"
  | "attackheli"
  | "transport"
  | "scoutheli"
  | "flakgun"
  | "howitzer"
  | "samsite"
  | "rocketpod"
  | "submarine"
  | "train"
  | "armoredcar"
  | "artytruck"
  | "hovercraft"
  | "heli"
  | BossKind;
export const BOSS_KINDS: BossKind[] = ["tank", "mech", "gunship", "worm", "airship", "cannon", "spider"];
export const BOSS_NAMES: Record<BossKind, string> = { tank: "Siege tank", mech: "Walker", gunship: "Gunship", worm: "Burrower", airship: "Sky fortress", cannon: "Doom cannon", spider: "Spider tank" };
const isBoss = (k: EnemyKind): k is BossKind => (BOSS_KINDS as string[]).includes(k);
/** Things in the air: no gravity, shot at without needing a clear line, crash when downed. */
const FLYING = new Set<EnemyKind>(["heli", "gunship", "drone", "jetpacker", "airship", "attackheli", "transport", "scoutheli"]);
/** Vehicles a blast damages rather than flings. */
const ARMOURED = new Set<EnemyKind>(["heli", "jeep", "truck", "boat", "battleship", "mg", "exosuit", "apc", "lighttank", "medtank", "heavytank", "flametank", "rockettank", "aatank", "launcher", "attackheli", "transport", "scoutheli", "flakgun", "howitzer", "samsite", "rocketpod", "submarine", "train", "armoredcar", "artytruck", "hovercraft"]);

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
  target: Foe | null;
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
  /** Seconds left on fire after a blast flung it (it trails flame and smoke as it tumbles) */
  burn: number;
  /** How big it is (0.75 small and quick … 2.2 a giant brute), and whether it carries explosives that go off when it dies */
  scale: number;
  sapper: boolean;
  /** Seconds left falling back to cover, and until its next grenade */
  cover: number;
  nade: number;
}

export type BarrelKind = "barrel" | "fuel" | "tanker" | "ammo" | "egg" | "propane" | "canister" | "cylinder" | "mine" | "tnt";

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
/** Explosive crossbow bolt, ricocheting saw blade, enemy laser bolt */
export const BOLT = 14;
export const SAW = 15;
export const ELASER = 16;
/** A napalm canister (fire on impact) and a flak shell (bursts in the air at its range) */
export const NAPALM = 17;
export const FLAK = 18;
/** A huge bomb: a big blast, a mushroom cloud, a crater */
export const BIGBOMB = 19;

const EXPLOSIVE = new Set([ROCKET, GRENADE, SHELL, METEOR, LOB, CANNON, MORTAR, FIREBALL, BOLT, NAPALM, FLAK, BIGBOMB]);
/** Gravity per shot kind (px/s²): lobbed things arc, flames drift up */
const SHOT_GRAV: Record<number, number> = { [GRENADE]: GRAV, [LOB]: 500, [CANNON]: 300, [MORTAR]: 400, [FIREBALL]: 350, [FLAMESHOT]: -70, [BOLT]: 120, [NAPALM]: 420, [BIGBOMB]: 240 };
const BLAST_R: Record<number, number> = { [ROCKET]: 18, [GRENADE]: 22, [SHELL]: 18, [METEOR]: 24, [LOB]: 20, [CANNON]: 24, [MORTAR]: 18, [FIREBALL]: 16, [BOLT]: 13, [NAPALM]: 14, [FLAK]: 11, [BIGBOMB]: 72 };

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
/** One ball of an explosion's fireball: white hot, then yellow, orange, red, and it cools to smoke as it swells and rises */
export const BLAZE = 14;
/** A burning chunk thrown out of a blast: arcs, bounces, trails flame */
export const CINDER = 15;
/** Dust kicked along the ground by a blast (its colour in `c`) */
export const DUST = 16;

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
  /** Blast radius (the big ones throw up a mushroom cloud) */
  r?: number;
  /** Part of the level's finale (the level ends once every one of those has landed) */
  fin?: boolean;
  /** Splits into this many warheads over the target area */
  split?: number;
  /** Top speed */
  max?: number;
  /** Fired from the ground: climbs first, then turns over and dives on its target */
  boost?: boolean;
  /** Whose it is (none: the level's finale) and how big: 0 small … 3 huge (MISSILE_R) */
  owner?: "enemy" | "ally";
  cls?: number;
  t?: number;
}

/** The ways a level can end. Every level picks one at random (never the last one again). */
export type Finale = "salvo" | "nuke" | "cruise" | "silo" | "mirv";
export const FINALES: Finale[] = ["salvo", "nuke", "cruise", "silo", "mirv"];
/** Banner, sub-line and the line on the "Level complete" card for each ending. */
export const FINALE_TEXT: Record<Finale, [string, string, string]> = {
  salvo: ["Missiles inbound", "Danger close", "Called in: ICBM salvo"],
  nuke: ["Nuke inbound", "Brace brace brace", "Called in: one very big warhead"],
  cruise: ["Cruise missiles", "Low and fast - get down", "Called in: cruise missile strike"],
  silo: ["Launch detected", "Their own silos are firing", "Called in: the enemy's own missiles"],
  mirv: ["Warheads splitting", "Everybody down", "Called in: MIRV, twelve warheads"],
};
/** The "Level complete" card's title, picked at random. */
const CARD_TITLES = ["Level complete", "Mission accomplished", "Stronghold flattened", "Sector cleared", "Target destroyed", "Nothing left standing", "Area secured"];

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
  /** The card's title, and how the level ended */
  title?: string;
  finale?: Finale;
  /** The squad bonus each player got */
  bonus?: number;
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
  battleship: 4000,
  jetpacker: 200,
  minigunner: 400,
  officer: 350,
  dynamiter: 200,
  rusher: 150,
  engineer: 200,
  laser: 300,
  exosuit: 900,
  apc: 1200,
  lighttank: 1500,
  medtank: 2000,
  heavytank: 3000,
  flametank: 1800,
  rockettank: 1800,
  aatank: 1600,
  launcher: 2500,
  attackheli: 1500,
  transport: 1400,
  scoutheli: 700,
  flakgun: 900,
  howitzer: 1400,
  samsite: 1200,
  rocketpod: 1100,
  submarine: 3000,
  train: 5000,
  armoredcar: 900,
  artytruck: 1300,
  hovercraft: 1200,
  airship: 6000,
  cannon: 6500,
  spider: 6500,
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
  battleship: 90,
  jetpacker: 3,
  minigunner: 10,
  officer: 4,
  dynamiter: 3,
  rusher: 3,
  engineer: 3,
  laser: 4,
  exosuit: 28,
  apc: 30,
  lighttank: 36,
  medtank: 50,
  heavytank: 80,
  flametank: 45,
  rockettank: 40,
  aatank: 40,
  launcher: 36,
  attackheli: 30,
  transport: 34,
  scoutheli: 14,
  flakgun: 30,
  howitzer: 40,
  samsite: 34,
  rocketpod: 30,
  submarine: 60,
  train: 150,
  armoredcar: 28,
  artytruck: 30,
  hovercraft: 34,
  airship: 300,
  cannon: 320,
  spider: 300,
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
  battleship: [40, 26],
  jetpacker: [3, 15],
  minigunner: [5, 17],
  officer: [3, 15],
  dynamiter: [3, 15],
  rusher: [3, 15],
  engineer: [3, 15],
  laser: [3, 15],
  exosuit: [7, 20],
  apc: [22, 15],
  lighttank: [20, 14],
  medtank: [20, 15],
  heavytank: [25, 17],
  flametank: [20, 15],
  rockettank: [20, 16],
  aatank: [19, 15],
  launcher: [27, 16],
  attackheli: [16, 9],
  transport: [24, 10],
  scoutheli: [11, 7],
  flakgun: [12, 12],
  howitzer: [16, 14],
  samsite: [14, 14],
  rocketpod: [12, 12],
  submarine: [32, 9],
  train: [64, 15],
  armoredcar: [16, 12],
  artytruck: [24, 15],
  hovercraft: [22, 10],
  tnt: [4, 8],
  airship: [46, 16],
  cannon: [30, 30],
  spider: [26, 34],
  mine: [3, 2],
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
  propane: [3, 10],
  canister: [3, 5],
  cylinder: [2, 11],
};
const BARREL_BLAST: Record<BarrelKind, number> = { barrel: 20, fuel: 34, tanker: 60, ammo: 16, egg: 0, propane: 24, canister: 18, cylinder: 22, mine: 22, tnt: 32 };
/** Dust a blast kicks up, by the block the ground is made of (others: a dun grey-brown). */
const DUST_COLOR: Partial<Record<number, string>> = {
  [DIRT]: "#9a7a58",
  [SAND]: "#e0c08a",
  [SANDSTONE]: "#d0a878",
  [SNOW]: "#eef2f8",
  [ASH]: "#6a6068",
  [CONCRETE]: "#a8a8a4",
  [STONE]: "#8a8a92",
  [MUD]: "#6a5a40",
  [REDROCK]: "#b8704e",
};

const sign = (v: number) => (v < 0 ? -1 : 1);
const clamp = (v: number, a: number, b: number) => (v < a ? a : v > b ? b : v);
/** Launch velocity for a lobbed shot from (x0, y0) to land on (x1, y1) under gravity g. */
function lobAt(x0: number, y0: number, x1: number, y1: number, g: number, speed: number) {
  const t = clamp(Math.abs(x1 - x0) / speed, 0.4, 1.6);
  return { vx: (x1 - x0) / t, vy: (y1 - y0 - 0.5 * g * t * t) / t };
}
/** Distance from a point to a line segment. */
function segDist(px: number, py: number, x0: number, y0: number, x1: number, y1: number) {
  const dx = x1 - x0;
  const dy = y1 - y0;
  const u = clamp(((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy || 1), 0, 1);
  return Math.hypot(px - (x0 + dx * u), py - (y0 + dy * u));
}

/** A body's half width and height, at its size. */
const dims = (e: Enemy): [number, number] => {
  const [w, h] = SIZE[e.kind];
  return e.scale === 1 ? [w, h] : [w * e.scale, h * e.scale];
};
/** Centre of a body's hit box (a boat's y is its water line). */
const centreY = (e: Enemy) => (FLYING.has(e.kind) ? e.y : e.y - dims(e)[1] / 2);
/** Foot soldiers: the ones that come in sizes, and can carry explosives. */
const INFANTRY = new Set<EnemyKind>(["mook", "heavy", "rocket", "sniper", "shield", "flamer", "grenadier", "minigunner", "officer", "dynamiter", "rusher", "engineer", "laser"]);

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
  /** Bombers, fighters and attack jets on both sides */
  jets: { x: number; y: number; vx: number; bombs: number; dropT: number; kind: PlaneType; ally: boolean; fireCd: number; hp: number; down: boolean; vy: number; dead: boolean; big?: boolean }[] = [];
  /** Players waiting for the next transport to drop them back in */
  private dropQueue: number[] = [];
  /** The allied column fighting alongside the squad */
  allies: Ally[] = [];
  /** Manholes in the road that enemies climb out of */
  hatches: { x: number; y: number; n: number; cd: number; open: number }[] = [];
  private allyCd = 1;
  /** Muzzle flashes */
  flashes: { x: number; y: number; t: number; c: string }[] = [];
  /** Laser beams and tesla lightning (seconds left to show) */
  beams: { x0: number; y0: number; x1: number; y1: number; t: number; c: string }[] = [];
  zaps: { pts: [number, number][]; t: number }[] = [];
  /** Supply crates parachuting in: a weapon (index into WEAPONS) or a jetpack (-1) */
  drops: { x: number; y: number; vy: number; weapon: number; ground: boolean; t: number }[] = [];
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
  private timers = { artillery: 2, meteor: 3, heli: 8, para: 6, jet: 8, silo: 6, drone: 10, bolt: 4, drop: 3, armor: 12, jetpack: 14, strike: 10, estrike: 12 };
  private weather = 0;
  /** How this level ends (picked when the squad reaches the stronghold) and how the last one did */
  finale: Finale = "salvo";
  private lastFinale: Finale | null = null;
  private current: Theme;
  /** The biomes, dealt from a shuffled deck */
  private deck: (current?: string) => Theme;
  /** Finale warheads still to land (scheduled or in the air) */
  private pending = 0;
  private lastView: View = { x: 0, y: 0, w: 400, h: 300 };
  private maxEnemies: number;
  private maxParticles: number;
  private colors: string[];

  constructor(theme: Theme, opts: GameOptions = {}) {
    this.rand = rng((opts.seed ?? 1) * 7919 + 3);
    this.maxEnemies = opts.maxEnemies ?? 70;
    this.maxParticles = opts.maxParticles ?? 2600;
    this.colors = opts.colors ?? ["#4aa3ff", "#ffcc33", "#c77dff", "#ff6a3d", "#5fd16a", "#3fd8e0", "#ff6fae", "#e8e2d0"];
    this.current = theme;
    this.level = createLevel(theme, opts.seed ?? 1, opts.level);
    this.deck = themeDeck(this.rand);
    this.level.pickTheme = (cur) => variant(this.deck(cur.key), this.rand);
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
      burn: 0,
      weapon: id,
      weaponT: 0,
      jet: 0,
      queued: false,
      score: 0,
      kills: 0,
    }));
    this.meetBoss();
    this.say(`Level ${this.levelNo}`, theme.name, "#ffcc33", this.briefing());
  }

  /** The biome the squad is in (the land ahead may already be the next one) */
  get theme() {
    return this.current;
  }

  /** Endless: the land runs on from biome to biome and the finale never comes. */
  get endless() {
    return this.level.endless;
  }
  set endless(v: boolean) {
    this.level.endless = v;
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
        return `${FINALE_TEXT[this.finale][0]} - danger close`;
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

  /** How many enemies at once: more when the view is wider (zoomed out). */
  private cap() {
    return Math.round(this.maxEnemies * clamp(this.lastView.w / 560, 1, 2.6));
  }

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
    const th = theme ?? variant(this.deck(this.theme.key), this.rand);
    const endless = this.level.endless;
    this.level = createLevel(th, Math.floor(this.rand() * 1e6) + 1, { endless, ...opts });
    this.level.pickTheme = (cur) => variant(this.deck(cur.key), this.rand);
    this.current = th;
    this.enemies = [];
    this.barrels = [];
    this.shots = [];
    this.missiles = [];
    this.mushrooms = [];
    this.waves = [];
    this.fires = [];
    this.jets = [];
    this.allies = [];
    this.hatches = [];
    this.flashes = [];
    this.beams = [];
    this.zaps = [];
    this.drops = [];
    this.popups = [];
    this.blasts = [];
    this.bolts = [];
    this.timeline = [];
    this.parts.length = 0;
    this.dirty.length = 0;
    this.fresh.length = 0;
    this.pending = 0;
    this.dropQueue = [];
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
      Object.assign(p, { dead: false, hp: PLAYER_HP, inv: 2.5, chute: true, ground: false, vx: 0, vy: 0, stuckT: 0, stall: 0, spin: 0, target: null, burn: 0, weapon: p.id, weaponT: 0, jet: 0, queued: false });
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
    L.intensity = this.intensity * CHAOS * (1 + 0.1 * Math.min(this.levelNo - 1, 6));

    // Over a border: the land has run on into the next biome, and this is the next level
    while (L.borders.length && this.advance > L.borders[0].col * TILE) {
      const b = L.borders.shift()!;
      const report = { no: this.levelNo, name: this.current.name, kills: this.levelKills, time: this.levelT };
      this.log = [report, ...this.log].slice(0, 6);
      this.cleared++;
      this.levelNo++;
      this.current = b.theme;
      this.levelT = 0;
      this.killsAt = this.kills;
      this.bossKind = b.boss;
      this.bossState = b.boss ? "ahead" : "none";
      this.say(`Level ${this.levelNo}`, b.theme.name, "#ffcc33", this.briefing());
    }
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
      this.manageAllies(dt, view);
      for (const a of this.allies) this.updateAlly(a, dt, view);
      this.updateHatches(dt, view);
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
    this.allies = this.allies.filter((a) => (!a.dead || a.deadT > 0) && a.x > left - 60);
    this.hatches = this.hatches.filter((h) => h.x > left);
  }

  /** The level's beats: the boss when the squad reaches the arena, the ICBMs at the stronghold. */
  private flow(view: View) {
    const plan = this.level.plan;
    const arena = this.level.bossQueue[0];
    if (this.phase === "play" && arena && this.advance > arena.col * TILE - 150) {
      this.level.bossQueue.shift();
      this.spawnBoss(arena.kind, view);
      this.holdAt = arena.col * TILE - 40;
      this.phase = "boss";
      this.phaseT = 0;
      this.bossState = "engaged";
    } else if (this.phase === "boss" && (!this.enemies.some((e) => e.boss && e.alive) || this.phaseT > 45)) {
      this.phase = "play";
      this.holdAt = Infinity;
      this.bossState = this.phaseT <= 45 ? "destroyed" : "bypassed";
      if (this.phaseT <= 45) this.say("Boss down", "", "#5fd16a");
    } else if (this.phase === "play" && plan.fortPlaced && this.advance > plan.targets[0] - 200) this.startNuke(view);
    else if (this.phase === "nuke" && this.phaseT > 22 && this.pending > 0) {
      // (a warhead that never came down: end it anyway)
      this.pending = 0;
      this.blowUpEverything();
    }
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
    if (s.kind === "manhole") {
      this.hatches.push({ x: s.x, y: s.y, n: 2 + Math.floor(this.rand() * 3), cd: 0, open: 0 });
      return;
    }
    if (s.kind === "barrel" || s.kind === "fuel" || s.kind === "tanker" || s.kind === "ammo" || s.kind === "egg" || s.kind === "propane" || s.kind === "canister" || s.kind === "cylinder" || s.kind === "mine" || s.kind === "tnt") {
      const hp = { barrel: 1, fuel: 4, tanker: 8, ammo: 1, egg: 3, propane: 1, canister: 1, cylinder: 1, mine: 1, tnt: 1 }[s.kind];
      this.barrels.push({ x: s.x, y: s.y, vx: 0, vy: 0, ground: true, facing: -1, type: s.kind, hp, fuse: -1, owner: -3, alive: true });
      return;
    }
    if (this.enemies.length >= this.cap() && s.kind !== "battleship") return;
    let y = s.y;
    // (ships ride in the water under the pier: they aren't lifted on top of it)
    if (s.kind !== "battleship" && s.kind !== "submarine") while (y > TILE && this.solid(s.x, y - 1)) y -= TILE;
    this.enemies.push(this.vary(this.makeEnemy(s.kind, s.x, y)));
  }

  /**
   * Some foot soldiers come small and quick, some big, the odd one a giant brute (slow, very
   * hard to put down, hits hard); some carry explosives that go off when they die.
   */
  private vary(e: Enemy) {
    if (!INFANTRY.has(e.kind)) return e;
    const r = this.rand();
    const scale = r < 0.05 ? 2.2 : r < 0.15 ? 1.5 : r < 0.3 ? 0.75 : 1;
    e.scale = scale;
    e.hp = e.maxHp = e.maxHp * (scale === 2.2 ? 7 : scale === 1.5 ? 3 : scale === 0.75 ? 0.5 : 1);
    e.sapper = this.rand() < 0.12;
    return e;
  }

  private makeEnemy(kind: EnemyKind, x: number, y: number): Enemy {
    const boss = isBoss(kind);
    // (tougher than they used to be: more health all round)
    const hp = HP[kind] * (boss ? 1.3 * Math.min(2, 1 + 0.12 * (this.levelNo - 1)) : 1.6);
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
      burn: 0,
      scale: 1,
      sapper: false,
      cover: 0,
      nade: 3 + this.rand() * 4,
    };
  }

  private spawnBoss(kind: BossKind, view: View) {
    const x = kind === "worm" ? this.advance + 190 : kind === "cannon" ? this.advance + 230 : view.x + view.w + 40;
    const e = this.makeEnemy(kind, x, surfaceY(this.level, x, view.y));
    if (kind === "gunship" || kind === "airship") {
      e.y = view.y + 10;
      e.hover = kind === "airship" ? 8 : 48;
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

  /**
   * The end of the level: the squad holds, and one of the endings (at random, never the one
   * before) comes down on the stronghold, and on the squad too. Once every warhead has landed,
   * everything goes.
   */
  private startNuke(view: View) {
    this.phase = "nuke";
    this.phaseT = 0;
    const pool = FINALES.filter((f) => f !== this.lastFinale);
    const f = pool[Math.floor(this.rand() * pool.length)];
    this.finale = this.lastFinale = f;
    const targets = this.level.plan.targets;
    this.holdAt = targets[0] - 200;
    this.say(FINALE_TEXT[f][0], FINALE_TEXT[f][1], "#ff3b3b");
    const r = this.rand;
    const order = [...targets].sort(() => r() - 0.5);
    /** Somewhere among the squad (when it lands, not now) */
    const squad = () => {
      const up = this.players.filter((p) => !p.dead);
      const x = up.length ? up.reduce((s, p) => s + p.x, 0) / up.length : this.advance;
      return x + (r() - 0.5) * 40;
    };
    const top = () => this.lastView.y - 220;
    /** Schedule one finale warhead: it's launched after `delay`, aimed at whatever `aim` gives then */
    const send = (delay: number, aim: () => number, m: (tx: number, ty: number) => Partial<Missile> & { x: number; y: number }) => {
      this.pending++;
      this.at(delay, () => {
        const tx = aim();
        const ty = surfaceY(this.level, tx, 0);
        this.missiles.push({ tx, ty, speed: 260, up: false, r: 56, fin: true, ...m(tx, ty) });
      });
    };
    const fromAbove = (tx: number) => ({ x: tx + 160 + r() * 160, y: top() });
    switch (f) {
      case "salvo": {
        // ICBMs on every target, in a random order, then two on the squad's heads
        let t = 1.2;
        for (const tx of order) {
          send(t, () => tx, (x) => fromAbove(x));
          t += 0.3 + r() * 0.45;
        }
        send(t + 0.2, squad, (x) => fromAbove(x));
        send(t + 0.5 + r() * 0.3, squad, (x) => ({ ...fromAbove(x), r: 64 }));
        break;
      }
      case "nuke": {
        // A couple of small ones to soften it up, then one enormous warhead straight down on the squad
        send(1, () => order[0], (x) => ({ ...fromAbove(x), r: 36 }));
        send(1.5 + r() * 0.4, () => order[1], (x) => ({ ...fromAbove(x), r: 36 }));
        send(2.6 + r() * 0.6, squad, (x) => ({ x: x + 20, y: top() - 200, r: 110, speed: 140, max: 520 }));
        break;
      }
      case "cruise": {
        // Low and fast from behind the squad, walking along the stronghold and back onto the squad
        const n = 6 + Math.floor(r() * 4);
        for (let k = 0; k < n; k++) {
          const last = k >= n - 2;
          const aim = last ? squad : () => targets[0] + ((targets[targets.length - 1] - targets[0]) * k) / (n - 3) + (r() - 0.5) * 30;
          send(1 + k * (0.22 + r() * 0.12), aim, (x, y) => ({ x: this.lastView.x - 60, y: Math.min(y, this.lastView.y + 40) - 30 - r() * 50, r: last ? 52 : 34, speed: 380, max: 820 }));
        }
        break;
      }
      case "silo": {
        // Their own missiles go up from the stronghold, turn over and come down on everything
        const n = 4 + Math.floor(r() * 3);
        for (let k = 0; k < n; k++)
          this.at(0.8 + k * 0.25, () => {
            const sx = order[k % order.length] + (r() - 0.5) * 30;
            this.missiles.push({ x: sx, y: surfaceY(this.level, sx, 0) - 4, tx: sx, ty: -999, speed: 90, up: true });
            this.shake = Math.max(this.shake, 2);
          });
        let t = 2.8;
        for (let k = 0; k < n; k++) {
          const onSquad = k >= n - 2;
          send(t, onSquad ? squad : () => order[k % order.length] + (r() - 0.5) * 40, (x) => ({ x: x + (r() - 0.5) * 120, y: top(), r: onSquad ? 58 : 46 }));
          t += 0.25 + r() * 0.35;
        }
        break;
      }
      case "mirv": {
        // One missile that bursts over the battlefield into a dozen warheads, the last few on the squad
        this.pending++;
        this.at(1.2, () => {
          const tx = (targets[0] + squad()) / 2;
          this.missiles.push({ x: tx + 260, y: top(), tx, ty: this.lastView.y + 30, speed: 300, up: false, fin: true, split: 12 });
        });
        break;
      }
    }
  }

  /** A MIRV bursting: its warheads fan out over the stronghold and the squad. */
  private splitMirv(m: Missile) {
    const targets = this.level.plan.targets;
    const up = this.players.filter((p) => !p.dead);
    const sx = up.length ? up.reduce((s, p) => s + p.x, 0) / up.length : this.advance;
    const n = m.split ?? 8;
    this.pending += n - 1;
    this.flash = Math.max(this.flash, 0.3);
    this.burst(m.x, m.y, 24, ["#fff6c8", "#ffe27a", "#ff9a2a"]);
    for (let k = 0; k < n; k++) {
      const onSquad = k >= n - 3;
      const tx = onSquad ? sx + (k - n + 2) * 36 + (this.rand() - 0.5) * 20 : targets[0] - 60 + this.rand() * (targets[targets.length - 1] - targets[0] + 120);
      this.missiles.push({ x: m.x + (this.rand() - 0.5) * 20, y: m.y, tx, ty: surfaceY(this.level, tx, 0), speed: 200 + this.rand() * 120, up: false, r: onSquad ? 50 : 30 + this.rand() * 16, fin: true });
    }
  }

  /**
   * A missile hits (not the finale's): it blows a crater in the land the size of its class, wrecks
   * the vehicles near it, and an enemy one throws the squad about. The big ones put up a
   * shockwave, the biggest a mushroom cloud.
   */
  private impact(m: Missile, x: number, y: number) {
    const cls = m.cls ?? 1;
    const r = MISSILE_R[cls];
    const theirs = m.owner === "enemy";
    this.explode(x, y, r, -4, theirs);
    if (r >= 30) this.waves.push({ x, y, t: 0.4 - cls * 0.12 });
    if (r >= 50) this.mushrooms.push({ x, y, t: cls === 3 ? 0 : 1.2 });
    this.flash = Math.max(this.flash, cls === 3 ? 0.75 : cls === 2 ? 0.25 : 0);
    this.shake = Math.max(this.shake, [3, 6, 10, 16][cls]);
    if (!theirs && cls >= 2)
      for (const e of this.enemies) if (e.alive && !e.boss && Math.abs(e.x - x) < r * 1.6) this.hurtEnemy(e, cls === 3 ? 60 : 24, -4, e.x - x);
    if (theirs && cls >= 1)
      for (const p of this.players) {
        if (p.dead || p.chute || Math.hypot(p.x - x, p.y - y) > r * 1.4) continue;
        p.vx = sign(p.x - x) * (120 + r * 2);
        p.vy = -(200 + r * 3);
        p.ground = false;
      }
  }

  /** A warhead lands: a huge blast, a mushroom cloud for the big ones, and anyone near it is thrown through the air. */
  private detonate(m: Missile, x = m.tx, y = m.ty) {
    const r = m.r ?? 56;
    this.explode(x, y, r, -4, false);
    if (r >= 40) {
      this.mushrooms.push({ x, y, t: 0 });
      this.waves.push({ x, y, t: 0 });
    }
    this.flash = Math.max(this.flash, r >= 90 ? 0.95 : r >= 40 ? 0.6 : 0.35);
    this.shake = r >= 90 ? 18 : r >= 40 ? 12 : 7;
    for (const e of this.enemies) if (e.alive && Math.abs(e.x - x) < r * 3.6) this.hurtEnemy(e, 999, -4, e.x - x);
    for (const p of this.players) {
      if (p.dead || p.chute) continue;
      const d = Math.hypot(p.x - x, p.y - 8 - y);
      if (d > r * 2.6) continue;
      // Thrown through the air, the closest on fire (they parachute into the next level)
      const f = 1 - d / (r * 2.6);
      p.dead = true;
      p.deadT = 99;
      p.ground = false;
      p.spin = this.rand() * 4;
      p.vx = sign(p.x - x || this.rand() - 0.5) * (110 + 280 * f + this.rand() * 60);
      p.vy = -(240 + 360 * f + this.rand() * 80);
      p.burn = f > 0.25 ? 1 + f * 1.5 : 0;
    }
    if (m.fin && this.pending > 0 && --this.pending === 0 && this.phase === "nuke") this.blowUpEverything();
  }

  /**
   * After the last warhead: the whole scene goes up, the screen fades to white, "Level complete"
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
      const bonus = 3000 + Math.floor(this.rand() * 13) * 500;
      const title = CARD_TITLES[Math.floor(this.rand() * CARD_TITLES.length)];
      const report = { no: this.levelNo, name: this.theme.name, kills: this.levelKills, time: this.levelT, title, finale: this.finale, bonus };
      this.complete = report;
      this.log = [report, ...this.log].slice(0, 6);
      this.cleared++;
      for (const p of this.players) p.score += bonus;
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

  /** In the water (deep enough to swim). */
  private inWater(b: Body) {
    return waterAt(this.level, b.x, b.y - 6) && waterAt(this.level, b.x, b.y - 1);
  }

  /** Swimming: the body floats with its head just out of the water and strokes along. */
  private swim(b: Body, dt: number, hw: number, hh: number) {
    const c = Math.floor(b.x / TILE);
    let r = Math.floor((b.y - 4) / TILE);
    while (r > 0 && kindAt(this.level, c, r - 1) === WATER) r--;
    const float = r * TILE + hh - 4;
    b.vy = clamp(b.vy + ((float - b.y) * 9 - b.vy * 3) * dt, -110, 110);
    this.integrate(b, dt, hw, hh, 0);
    if (Math.abs(b.y - float) < 4 && this.rand() < 0.15) this.add({ k: SPARK, x: b.x - b.facing * 3, y: float - hh + 4, vx: (this.rand() - 0.5) * 30, vy: -30, life: 0.3, c: "#e0f6ff", s: 1 });
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
      if (p.queued) return;
      p.deadT -= dt;
      this.tumble(p, dt, PW, PH);
      if (p.burn > 0) {
        p.burn -= dt;
        this.add({ k: FLAME, x: p.x + (this.rand() - 0.5) * 4, y: p.y - 8, vx: -p.vx * 0.15, vy: -20, life: 0.4 + this.rand() * 0.25, c: "", s: 2 });
        if (this.rand() < 0.4) this.puff(p.x, p.y - 10, 2);
      }
      if (p.deadT <= 0) this.respawn(p, view);
      return;
    }
    p.inv = Math.max(0, p.inv - dt);
    p.kick = Math.max(0, p.kick - dt);
    if (p.weaponT > 0 && (p.weaponT -= dt) <= 0) p.weapon = p.id;
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
    // A big climb ahead: take it on the jetpack now and then
    if (!p.chute && !p.jet && p.ground && dir && this.rand() < dt * 0.15 && surfaceY(this.level, p.x + dir * 40, p.y - 120) < p.y - 32) p.jet = 2.5;
    if (p.chute) {
      this.integrate(p, dt, PW, PH, 0.25, 55);
      if (p.ground) p.chute = false;
    } else if (this.inWater(p) && p.jet <= 0) {
      p.vx = dir * RUN * 0.7;
      this.swim(p, dt, PW, PH);
    } else if (p.jet > 0) {
      // Jetpack: fly along a few blocks over the ground, over anything in the way
      p.jet -= dt;
      const below = Math.min(surfaceY(this.level, p.x, p.y - 100), surfaceY(this.level, p.x + dir * 14, p.y - 100));
      const ty = below - 34 + Math.sin(this.t * 3 + p.id) * 4;
      p.vy = clamp((ty - p.y) * 3, -120, 120);
      p.vx = dir * RUN * 1.2;
      this.integrate(p, dt, PW, PH, 0);
      if (this.rand() < 0.9) this.add({ k: FLAME, x: p.x - p.facing * 3, y: p.y - 6, vx: -p.facing * 10, vy: 60 + this.rand() * 40, life: 0.2, c: "", s: 1 + Math.round(this.rand()) });
      if (this.rand() < 0.3) this.puff(p.x - p.facing * 3, p.y - 2, 2);
    } else this.integrate(p, dt, PW, PH);
    // Picking up a supply crate
    for (let i = this.drops.length - 1; i >= 0; i--) {
      const d = this.drops[i];
      if (Math.abs(d.x - p.x) > 10 || Math.abs(d.y - p.y) > 18) continue;
      this.drops.splice(i, 1);
      if (d.weapon < 0) p.jet = 7;
      else {
        p.weapon = d.weapon;
        p.weaponT = 16;
      }
      this.popups.push({ x: p.x, y: p.y - 26, text: `P${p.id + 1} ${d.weapon < 0 ? "JETPACK" : WEAPONS[d.weapon].toUpperCase()}`, c: this.colors[p.id], t: 0 });
      this.burst(p.x, p.y - 10, 10, [this.colors[p.id], "#ffffff"]);
    }
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
      // Over it on the jetpack, or blow a way through
      if (this.rand() < 0.5) p.jet = 3;
      else this.grenade(p, p.x + dir * 24);
      p.stuckT = 0;
    }
    if (p.stall > 6) this.respawn(p, view);
  }

  /** Nearest enemy in range and in sight (lobbed weapons don't need sight); a barrel if an enemy stands next to it. */
  findTarget(p: Player): Enemy | Barrel | null {
    const range = W_RANGE[p.weapon];
    const gx = p.x;
    const gy = p.y - 9;
    const lobs = LOBBED.has(p.weapon);
    // (only what's on screen: the fight stays where it can be seen)
    const v = this.lastView;
    let best: Enemy | Barrel | null = null;
    let bestD = range;
    for (const e of this.enemies) {
      if (!e.alive) continue;
      const aim = this.aimPoint(e);
      if (!aim || aim.x > v.x + v.w - 6 || aim.x < v.x + 4) continue;
      const d = Math.hypot(aim.x - p.x, (aim.y - gy) * 0.6);
      // Prefer what's ahead (and bosses): things behind count as further away
      const score = (e.x < p.x - 10 ? d * 1.6 : d) * (e.boss ? 0.7 : 1);
      if (score >= bestD) continue;
      if (!lobs && !FLYING.has(e.kind) && !lineOfSight(this.level, gx, gy, aim.x, aim.y)) continue;
      best = e;
      bestD = score;
    }
    for (const b of this.barrels) {
      if (!b.alive || b.fuse >= 0 || b.type === "egg" || b.type === "mine") continue;
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
    // (a submarine under water can't be hit)
    if (e.kind === "submarine" && e.spin > 0) return null;
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
    switch (p.weapon) {
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
      case 7: // minigun: long hosing bursts
        shot(a + (r() - 0.5) * 0.14, 380, BULLET, 0.6, 0.55);
        p.burst++;
        p.fireCd = p.burst % 24 === 0 ? 0.8 : 0.045;
        break;
      case 8: {
        // laser: a beam that burns through everything in its path, up to the first wall
        const len = W_RANGE[8];
        let end = len;
        for (let d = 6; d < len; d += 4)
          if (this.solid(gx + Math.cos(a) * d, gy + Math.sin(a) * d)) {
            end = d;
            break;
          }
        const x1 = gx + Math.cos(a) * end;
        const y1 = gy + Math.sin(a) * end;
        for (const e of this.enemies) {
          if (!e.alive) continue;
          const q = this.aimPoint(e);
          if (q && segDist(q.x, q.y, gx, gy, x1, y1) < SIZE[e.kind][0] + 3) this.hurtEnemy(e, 2.2, p.id, Math.cos(a) * 60);
        }
        this.beams.push({ x0: gx, y0: gy, x1, y1, t: 0.1, c: this.colors[p.id] });
        for (let k = 0; k < 4; k++) this.add({ k: SPARK, x: x1, y: y1, vx: (r() - 0.5) * 80, vy: -r() * 80, life: 0.25, c: "#ffffff", s: 1 });
        p.fireCd = 0.42;
        break;
      }
      case 9: {
        // tesla gun: lightning that jumps from one enemy to the next
        const pts: [number, number][] = [[gx, gy]];
        const hit = new Set<Enemy>();
        let cur: Enemy | null = "kind" in tgt ? tgt : null;
        for (let k = 0; k < 4 && cur; k++) {
          const q = this.aimPoint(cur) ?? { x: cur.x, y: cur.y };
          pts.push([q.x, q.y]);
          hit.add(cur);
          this.hurtEnemy(cur, 1.6, p.id, 0);
          let next: Enemy | null = null;
          let nd = 80;
          for (const e of this.enemies) {
            if (!e.alive || hit.has(e)) continue;
            const d = Math.hypot(e.x - q.x, centreY(e) - q.y);
            if (d < nd) {
              nd = d;
              next = e;
            }
          }
          cur = next;
        }
        if (!("kind" in tgt)) {
          pts.push([tgt.x, tgt.y - 4]);
          this.lightBarrel(tgt, p.id, 0.05);
        }
        this.zaps.push({ pts, t: 0.14 });
        p.fireCd = 0.38;
        break;
      }
      case 10: // dual pistols: left, right, left, right
        shot(a + (r() - 0.5) * 0.1, 380, BULLET, 1, 0.5);
        p.burst++;
        p.fireCd = p.burst % 2 ? 0.09 : 0.2;
        break;
      case 11: // crossbow: explosive bolts
        shot(a - 0.05, 330, BOLT, 0, 1.2);
        p.fireCd = 0.75;
        break;
      case 12: // saw blades that ricochet round the room, cutting through whoever's there
        shot(a, 230, SAW, 1.2, 1.8, 8);
        p.fireCd = 0.6;
        break;
      default: {
        // mortar tube: shells lobbed high, onto the enemy's head
        const v = lobAt(gx, gy, aim.x + (r() - 0.5) * 16, aim.y, SHOT_GRAV[MORTAR], 150);
        this.shots.push({ kind: MORTAR, x: gx, y: gy - 2, vx: v.vx, vy: v.vy, life: 4, dmg: 0, pid: p.id, pierce: 0 });
        this.puff(gx, gy - 4, 3);
        p.fireCd = 1.3;
      }
    }
    if (p.weapon !== 5 && p.weapon !== 8 && p.weapon !== 9) this.flashes.push({ x: gx + p.facing * 2, y: gy, t: 0.05, c: this.colors[p.id] });
    if (p.weapon === 0 || p.weapon === 1 || p.weapon === 4 || p.weapon === 7 || p.weapon === 10) this.add({ k: CASING, x: p.x, y: gy, vx: -p.facing * (20 + r() * 30), vy: -60 - r() * 40, life: 0.8, c: "#ffd27a", s: 1 });
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

  /** Back in the fight: the next transport over drops them by parachute (until then they're out of it). */
  private respawn(p: Player, view: View) {
    p.dead = true;
    p.queued = true;
    p.deadT = 999;
    p.vx = p.vy = 0;
    p.x = view.x - 200;
    if (!this.dropQueue.includes(p.id)) this.dropQueue.push(p.id);
  }

  /** A player jumps from a transport: back in, on a parachute. */
  private release(p: Player, x: number, y: number) {
    Object.assign(p, { dead: false, queued: false, hp: PLAYER_HP, inv: 1.5, x, y, vx: 0, vy: 0, ground: false, chute: true, stuckT: 0, stall: 0, spin: 0, burn: 0, jet: 0, target: null });
  }

  /** The nearest player or allied vehicle in range (and in sight, if asked). */
  private nearestFoe(e: Enemy, range: number, sight: boolean, dy = 120): Foe | null {
    let best: Foe | null = this.nearestPlayer(e, range, sight, dy);
    let bestD = best ? Math.abs(best.x - e.x) : range;
    const ey = e.y - Math.min(14, SIZE[e.kind][1] - 4);
    for (const a of this.allies) {
      if (a.dead) continue;
      const d = Math.abs(a.x - e.x);
      if (d >= bestD || Math.abs(a.y - e.y) > dy + 40) continue;
      if (sight && !ALLY_AIR.has(a.kind) && !lineOfSight(this.level, e.x, ey, a.x, a.y - 8)) continue;
      best = a;
      bestD = d;
    }
    return best;
  }

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
      if (e.kind === "boat" || e.kind === "battleship" || e.kind === "submarine") {
        e.y += dt * (e.kind === "battleship" ? 6 : 4);
        if (e.kind === "battleship" && this.rand() < 0.5) this.flame(e.x + (this.rand() - 0.5) * 60, e.y - 10);
        return;
      }
      if (!e.boss) this.tumble(e, dt, 3, 11);
      if (e.burn > 0) {
        // A comet of flame and smoke behind the body
        e.burn -= dt;
        const y = e.y - 7;
        this.add({ k: FLAME, x: e.x + (this.rand() - 0.5) * 4, y, vx: -e.vx * 0.15, vy: -20, life: 0.35 + this.rand() * 0.25, c: "", s: 2 });
        if (this.rand() < 0.5) this.add({ k: FLAME, x: e.x, y: y + 2, vx: 0, vy: -10, life: 0.25, c: "", s: 1 });
        if (this.rand() < 0.35) this.puff(e.x, y - 2, 2);
      }
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
      case "battleship":
        return this.updateBattleship(e, dt);
      case "jetpacker":
        return this.updateJetpacker(e, dt, view);
      case "rusher":
        return this.updateRusher(e, dt);
      case "apc":
      case "lighttank":
      case "medtank":
      case "heavytank":
      case "flametank":
      case "rockettank":
      case "aatank":
      case "launcher":
        return this.updateArmour(e, dt);
      case "attackheli":
      case "transport":
      case "scoutheli":
        return this.updateHeliType(e, dt, view);
      case "flakgun":
      case "howitzer":
      case "samsite":
      case "rocketpod":
        return this.updateEmplacement(e, dt);
      case "submarine":
        return this.updateSubmarine(e, dt);
      case "train":
        return this.updateTrain(e, dt);
      case "airship":
        return this.updateAirship(e, dt, view);
      case "cannon":
        return this.updateCannon(e, dt);
      case "spider":
        return this.updateSpider(e, dt);
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
    const ranges: Partial<Record<EnemyKind, number>> = {
      heavy: 170,
      turret: 190,
      mg: 230,
      sniper: 290,
      mortar: 330,
      rocket: 210,
      flamer: 75,
      grenadier: 210,
      shield: 150,
      minigunner: 200,
      officer: 260,
      dynamiter: 160,
      engineer: 180,
      laser: 240,
      exosuit: 230,
    };
    const range = ranges[e.kind] ?? 150;
    e.scanT -= dt;
    if (e.scanT <= 0) {
      e.scanT = 0.15;
      // (lobbed weapons don't need a clear line)
      // (only once it's on screen: off to the right, it waits for the squad to come into view)
      e.target = this.inView(e) ? this.nearestFoe(e, range, e.kind !== "mortar" && e.kind !== "grenadier" && e.kind !== "dynamiter" && e.kind !== "officer", e.kind === "sniper" ? 200 : 120) : null;
      if (e.target && !e.alert) {
        e.alert = true;
        this.popups.push({ x: e.x, y: e.y - dims(e)[1] - 6, text: "!", c: "#ff3b3b", t: 0 });
      }
    }
    const tgt = e.target && !e.target.dead ? e.target : null;
    e.vx = 0;
    e.cover = Math.max(0, e.cover - dt);
    e.nade -= dt;
    if (tgt) {
      e.facing = tgt.x >= e.x ? 1 : -1;
      e.aimA = Math.atan2(tgt.y - 9 - (e.y - 8), tgt.x - e.x);
      // Quicker on the trigger than they used to be (a brute, slower)
      e.fireCd -= dt * 1.6;
      // Grenades, at a cluster of the squad by preference
      if (e.nade <= 0 && e.kind !== "sniper" && e.kind !== "flamer" && Math.abs(tgt.x - e.x) < 170) {
        const cluster = this.players.filter((q) => !q.dead && Math.hypot(q.x - tgt.x, q.y - tgt.y) < 40).length;
        if (cluster >= 2 || this.rand() < 0.35) {
          const v = lobAt(e.x, e.y - 12, tgt.x + (this.rand() - 0.5) * 12, tgt.y - 2, GRAV, 150);
          this.shots.push({ kind: GRENADE, x: e.x, y: e.y - 12, vx: v.vx, vy: v.vy, life: 1.4, dmg: 0, pid: -1, pierce: 0 });
        }
        e.nade = 5 + this.rand() * 4;
      }
      // Hurt badly: fall back a little way, still firing, then come on again
      if (e.hp < e.maxHp * 0.35 && e.cover <= 0 && this.rand() < dt * 0.6) e.cover = 1.4;
      if (e.kind === "sniper") {
        // Line the shot up (the laser sight shows), then fire
        e.aim += dt;
        if (e.aim > 0.9 && e.fireCd <= 0) {
          this.enemyShot(e, tgt, 520, 2, 0.02);
          e.aim = 0;
          e.fireCd = 2.4;
        }
      } else if (e.fireCd <= 0) this.enemyFire(e, tgt);
      // Close to their best range and hold it: everyone pushes in (fast), backs off if crowded,
      // jumps what's in the way; the big ones lumber, the small ones scamper
      const gap = Math.abs(tgt.x - e.x);
      const ADV: Partial<Record<EnemyKind, number>> = { mook: 34, heavy: 18, rocket: 24, shield: 22, flamer: 46, grenadier: 26, minigunner: 20, officer: 24, dynamiter: 42, engineer: 28, laser: 30, exosuit: 22 };
      const fixed = e.kind === "turret" || e.kind === "mg" || e.kind === "mortar" || e.kind === "sniper";
      if (e.ground && !e.chute && !fixed) {
        const want = e.kind === "flamer" ? 40 : e.kind === "shield" ? 50 : range * 0.55;
        const speed = (ADV[e.kind] ?? 24) * (e.scale > 2 ? 1.25 : e.scale > 1 ? 1.2 : e.scale < 1 ? 1.4 : 1);
        if (e.cover > 0) e.vx = -e.facing * speed * 0.8;
        else if (gap > want + 16) e.vx = e.facing * speed;
        else if (gap < Math.min(28, want * 0.5)) e.vx = -e.facing * speed * 0.6;
        if (e.vx && this.solid(e.x + sign(e.vx) * (dims(e)[0] + 3), e.y - 3)) e.vy = -230;
        if (e.vx && this.rand() < 0.04) this.add({ k: DUST, x: e.x, y: e.y - 1, vx: -e.vx * 0.3, vy: -10, life: 0.5, c: "#9a8a78", s: 1.5 });
      }
    } else {
      e.aim = 0;
      if (e.kind !== "turret" && e.kind !== "mg" && e.kind !== "mortar" && !e.chute) {
        // Patrol a few steps either side of where it was placed
        const ahead = e.x + e.facing * 5;
        if (Math.abs(ahead - e.home) > 18 || this.solid(ahead, e.y - 4) || !this.solid(ahead, e.y + 2)) e.facing = e.facing === 1 ? -1 : 1;
        else e.vx = e.facing * (e.kind === "heavy" ? 9 : 15);
      }
    }
    const [hw, hh] = dims(e);
    if (e.chute) {
      e.vx = 0;
      this.integrate(e, dt, hw, hh, 0.25, 45);
      if (e.ground || this.inWater(e)) e.chute = false;
    } else if (this.inWater(e)) this.swim(e, dt, hw, hh);
    else this.integrate(e, dt, hw, hh);
    e.anim += dt * (e.vx ? 8 : 0);
    if (lavaAt(this.level, e.x, e.y - 2)) this.killEnemy(e, -3, 0, -150);
  }

  /** One enemy bullet from `e` at `p`. */
  private enemyShot(e: Enemy, p: Foe, speed: number, dmg: number, spread: number, fromY = 9) {
    const gx = e.x + e.facing * 7;
    const gy = e.kind === "heli" || e.kind === "gunship" ? e.y + 6 : e.kind === "drone" ? e.y + 3 : e.y - fromY;
    // (leading a moving target, and steadier hands than they used to have)
    const lead = ((p.vx ?? 0) * Math.hypot(p.x - gx, p.y - gy)) / speed;
    const a = Math.atan2(p.y - 9 - gy, p.x + lead - gx) + (this.rand() - 0.5) * spread * 0.6;
    // (the big ones hit much harder: giants and heavy armour, bosses)
    const heavy = e.boss ? 2.5 : e.scale > 1 ? e.scale * 1.6 : e.kind === "heavytank" || e.kind === "exosuit" || e.kind === "heavy" || e.kind === "minigunner" ? 1.6 : 1;
    this.shots.push({ kind: ENEMY_BULLET, x: gx, y: gy, vx: Math.cos(a) * speed, vy: Math.sin(a) * speed, life: 1.6, dmg: dmg * heavy, pid: -1, pierce: 0 });
    this.flashes.push({ x: gx, y: gy, t: 0.05, c: "#ff9a2a" });
    if (this.rand() < 0.3) this.add({ k: CASING, x: e.x, y: gy, vx: -e.facing * (20 + this.rand() * 30), vy: -60 - this.rand() * 40, life: 0.7, c: "#c8a050", s: 1 });
  }

  /** On screen (or nearly): enemies off to the right hold their fire until the squad can see them. */
  private inView(e: Enemy) {
    const v = this.lastView;
    return e.x > v.x - 12 && e.x < v.x + v.w + 12;
  }

  private enemyFire(e: Enemy, p: Foe) {
    const r = this.rand;
    switch (e.kind) {
      case "minigunner":
        // Spins up and hoses the squad
        this.enemyShot(e, p, 190, 1, 0.28, 11);
        e.burst++;
        e.fireCd = e.burst % 22 === 0 ? 1.8 : 0.05;
        return;
      case "officer":
        // A pistol, and now and then a flare: artillery comes down where it lands
        this.enemyShot(e, p, 170, 1, 0.15);
        e.burst++;
        e.fireCd = 1.1;
        if (e.burst % 4 === 0) {
          this.popups.push({ x: e.x, y: e.y - 22, text: "FIRE MISSION", c: "#ff3b3b", t: 0 });
          const tx = p.x;
          for (let k = 0; k < 4; k++)
            this.at(1 + k * 0.25, () => this.shots.push({ kind: SHELL, x: tx + (this.rand() - 0.5) * 70, y: this.lastView.y - 10, vx: 0, vy: 280, life: 4, dmg: 0, pid: -3, pierce: 0 }));
        }
        return;
      case "dynamiter": {
        // Lit sticks of dynamite, lobbed
        const v = lobAt(e.x, e.y - 12, p.x + (r() - 0.5) * 20, p.y, SHOT_GRAV[LOB], 160);
        this.shots.push({ kind: LOB, x: e.x, y: e.y - 12, vx: v.vx, vy: v.vy, life: 2.5, dmg: 0, pid: -1, pierce: 0 });
        e.fireCd = 2 + r();
        return;
      }
      case "engineer":
        // Plants mines in the squad's path, and keeps them off with a pistol
        e.burst++;
        if (e.burst % 4 === 0 && this.barrels.filter((b) => b.type === "mine").length < 14)
          this.barrels.push({ x: e.x - 12, y: e.y, vx: 0, vy: 0, ground: true, facing: -1, type: "mine", hp: 1, fuse: -1, owner: -3, alive: true });
        else this.enemyShot(e, p, 170, 1, 0.15);
        e.fireCd = 1 + r() * 0.6;
        return;
      case "laser": {
        // A bolt of light: fast and hard-hitting
        const gx = e.x + e.facing * 8;
        const gy = e.y - 9;
        const a = Math.atan2(p.y - 9 - gy, p.x - gx) + (r() - 0.5) * 0.08;
        this.shots.push({ kind: ELASER, x: gx, y: gy, vx: Math.cos(a) * 420, vy: Math.sin(a) * 420, life: 1, dmg: 2, pid: -1, pierce: 0 });
        this.flashes.push({ x: gx, y: gy, t: 0.06, c: "#5ff0e0" });
        e.burst++;
        e.fireCd = e.burst % 3 === 0 ? 1.6 : 0.25;
        return;
      }
      case "exosuit":
        // Machine guns, and a pair of rockets every few bursts
        this.enemyShot(e, p, 190, 1, 0.2, 14);
        e.burst++;
        e.fireCd = e.burst % 6 === 0 ? 1.2 : 0.08;
        if (e.burst % 18 === 0)
          for (let k = 0; k < 2; k++) {
            const a = Math.atan2(p.y - 9 - (e.y - 24), p.x - e.x) + (k - 0.5) * 0.2;
            this.shots.push({ kind: ROCKET, x: e.x, y: e.y - 24, vx: Math.cos(a) * 110, vy: Math.sin(a) * 110, life: 2.2, dmg: 0, pid: -1, pierce: 0 });
          }
        return;
    }
    if (e.kind === "rocket" && this.rand() < 0.4) {
      // A guided missile now and then
      this.missiles.push({ x: e.x + e.facing * 6, y: e.y - 11, tx: p.x, ty: surfaceY(this.level, p.x, p.y - 30), speed: 150, max: 420, up: false, owner: "enemy", cls: 0 });
      e.fireCd = 2.6 + r();
      return;
    }
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
      e.vx = e.facing * 105;
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

  /** A trooper on a jetpack: flies in over the front of the squad and shoots down at it. */
  private updateJetpacker(e: Enemy, dt: number, view: View) {
    const tx = this.advance + 70 + ((e.home * 37) % 140);
    const below = surfaceY(this.level, e.x, view.y);
    const ty = Math.max(view.y + 20, below - 46 - (e.home % 3) * 10) + Math.sin(this.t * 2.5 + e.home) * 6;
    e.x += (tx - e.x) * (1 - Math.exp(-dt * 0.9));
    e.y += (ty - e.y) * (1 - Math.exp(-dt * 2));
    e.anim += dt * 4;
    if (this.rand() < 0.9) this.add({ k: FLAME, x: e.x - e.facing * 3, y: e.y - 4, vx: 0, vy: 60 + this.rand() * 30, life: 0.18, c: "", s: 1 + Math.round(this.rand()) });
    const p = this.nearestPlayer(e, 200, false, 160);
    if (!p) return;
    e.facing = p.x >= e.x ? 1 : -1;
    e.fireCd -= dt;
    if (e.fireCd <= 0 && this.inView(e)) {
      this.enemyShot(e, p, 180, 1, 0.2, 9);
      e.burst++;
      e.fireCd = e.burst % 4 === 0 ? 1.3 : 0.12;
    }
  }

  /** Charges the squad with a machete, jumping whatever's in the way. */
  private updateRusher(e: Enemy, dt: number) {
    const p = this.inView(e) ? this.nearestPlayer(e, 260, false, 80) : null;
    if (p) {
      e.facing = p.x >= e.x ? 1 : -1;
      if (!e.alert) {
        e.alert = true;
        this.popups.push({ x: e.x, y: e.y - 20, text: "!", c: "#ff3b3b", t: 0 });
      }
      e.vx = e.facing * 95;
    } else e.vx = 0;
    if (e.vx && e.ground && this.solid(e.x + e.facing * 6, e.y - 3)) e.vy = -230;
    this.integrate(e, dt, 3, 15);
    e.anim += dt * (e.vx ? 14 : 0);
    if (p && e.bite <= 0 && Math.abs(p.x - e.x) < 8 && Math.abs(p.y - e.y) < 14) {
      this.hurtPlayer(p, 2);
      e.bite = 0.6;
    }
    if (lavaAt(this.level, e.x, e.y - 2)) this.killEnemy(e, -3, 0, -150);
  }

  /**
   * The enemy's armour: each kind drives up to its own distance from the squad and fights its
   * own way. Carriers unload troops; light, medium and heavy tanks shell (the heavy with twin
   * guns); flame tanks close in and hose; rocket tanks fire salvos of missiles; flak tanks shoot
   * at anything in the air; launchers raise a missile and fire it, small, big or enormous.
   */
  private updateArmour(e: Enemy, dt: number) {
    const k = e.kind;
    const STOP: Record<string, number> = { apc: 210, lighttank: 240, medtank: 230, heavytank: 250, flametank: 80, rockettank: 320, aatank: 260, launcher: 300, armoredcar: 200, artytruck: 380, hovercraft: 220 };
    const SPEED: Record<string, number> = { apc: 40, lighttank: 30, medtank: 32, heavytank: 26, flametank: 38, rockettank: 26, aatank: 30, launcher: 24, armoredcar: 52, artytruck: 30, hovercraft: 44 };
    e.vx = e.x > this.advance + STOP[k] ? -SPEED[k] : 0;
    e.facing = -1;
    const [hw, hh] = SIZE[e.kind];
    if (k === "hovercraft" && this.inWater(e)) {
      // Over water: riding on the surface
      let row = Math.floor((e.y - 4) / TILE);
      while (row > 0 && kindAt(this.level, Math.floor(e.x / TILE), row - 1) === WATER) row--;
      e.y = row * TILE;
      e.vy = 0;
      e.x += e.vx * dt;
    } else this.integrate(e, dt, hw, hh);
    e.anim += dt * Math.abs(e.vx) * 0.3;
    if (!this.inView(e)) return;
    const p = this.nearestFoe(e, k === "rockettank" || k === "launcher" ? 460 : 340, false, 200);
    // A flak tank looks for something in the air first: an allied helicopter, a squad member on a jetpack
    let air: Foe | null = null;
    if (k === "aatank") {
      air = this.allies.find((a) => !a.dead && ALLY_AIR.has(a.kind) && Math.abs(a.x - e.x) < 380) ?? this.players.find((q) => !q.dead && q.jet > 0 && Math.abs(q.x - e.x) < 320) ?? null;
    }
    const tgt = air ?? p;
    if (tgt) e.aimA = Math.atan2(tgt.y - 9 - (e.y - 18), tgt.x - e.x);
    e.fireCd -= dt * 1.4;
    e.altCd -= dt * 1.3;
    const gunY = e.y - (k === "heavytank" ? 20 : 17);
    const gx = e.x + Math.cos(e.aimA) * 16;
    const gy = gunY + Math.sin(e.aimA) * 16;
    if (tgt && e.fireCd <= 0 && k !== "rockettank" && k !== "launcher") {
      if (k === "flametank") {
        if (Math.abs(tgt.x - e.x) < 100) {
          for (let j = 0; j < 2; j++) {
            const a = e.aimA + (this.rand() - 0.5) * 0.3;
            const v = 150 + this.rand() * 40;
            this.shots.push({ kind: FLAMESHOT, x: gx, y: gy, vx: Math.cos(a) * v, vy: Math.sin(a) * v, life: 0.55, dmg: 1, pid: -1, pierce: 0 });
          }
          e.burst++;
          e.fireCd = e.burst % 30 === 0 ? 1.4 : 0.05;
        }
      } else if (air) {
        // Flak: shells that burst at the target's range
        const d = Math.hypot(air.x - gx, air.y - gy);
        const a = e.aimA + (this.rand() - 0.5) * 0.12;
        this.shots.push({ kind: FLAK, x: gx, y: gy, vx: Math.cos(a) * 320, vy: Math.sin(a) * 320, life: d / 320 + (this.rand() - 0.5) * 0.08, dmg: 0, pid: -1, pierce: 0 });
        this.flashes.push({ x: gx, y: gy, t: 0.05, c: "#ffe27a" });
        e.burst++;
        e.fireCd = e.burst % 6 === 0 ? 1 : 0.14;
      } else {
        this.enemyShot(e, tgt, 190, 1, 0.2, 18);
        e.burst++;
        e.fireCd = e.burst % 6 === 0 ? 1.4 : 0.09;
      }
    }
    if (!tgt || e.altCd > 0) {
      if (k === "launcher") e.hover = Math.max(0, e.hover - dt * 0.5);
      return;
    }
    switch (k) {
      case "lighttank":
      case "medtank":
      case "heavytank": {
        // Shells: flatter and faster from the bigger guns; the heavy fires both barrels
        const speed = k === "lighttank" ? 230 : k === "medtank" ? 290 : 260;
        for (let j = 0; j < (k === "heavytank" ? 2 : 1); j++) {
          const v = lobAt(gx, gy - j * 4, tgt.x + (this.rand() - 0.5) * 16, tgt.y, SHOT_GRAV[CANNON], speed);
          this.shots.push({ kind: CANNON, x: gx, y: gy - j * 4, vx: v.vx, vy: v.vy, life: 4, dmg: 0, pid: -1, pierce: 0 });
        }
        this.flashes.push({ x: gx, y: gy, t: 0.1, c: "#ffe27a" });
        this.puff(gx, gy, 5);
        this.shake = Math.max(this.shake, k === "heavytank" ? 4 : 2);
        e.altCd = (k === "heavytank" ? 3.4 : k === "medtank" ? 2.5 : 2.8) + this.rand();
        break;
      }
      case "armoredcar": {
        const v = lobAt(gx, gy, tgt.x, tgt.y, SHOT_GRAV[CANNON], 300);
        this.shots.push({ kind: CANNON, x: gx, y: gy, vx: v.vx, vy: v.vy, life: 3, dmg: 0, pid: -1, pierce: 0 });
        this.flashes.push({ x: gx, y: gy, t: 0.08, c: "#ffe27a" });
        e.altCd = 2.2 + this.rand();
        break;
      }
      case "artytruck": {
        // High arcing shells from far back
        const v = lobAt(gx, gy - 6, tgt.x + (this.rand() - 0.5) * 30, tgt.y, SHOT_GRAV[CANNON], 160);
        this.shots.push({ kind: CANNON, x: gx, y: gy - 6, vx: v.vx, vy: v.vy, life: 6, dmg: 0, pid: -1, pierce: 0 });
        this.flashes.push({ x: gx, y: gy - 6, t: 0.14, c: "#ffe27a" });
        for (let j = 0; j < 3; j++) this.puff(gx, gy - 6, 5);
        this.shake = Math.max(this.shake, 3);
        e.altCd = 3.2 + this.rand();
        break;
      }
      case "hovercraft":
        for (let j = 0; j < 2; j++) {
          const a = e.aimA + (j - 0.5) * 0.12;
          this.shots.push({ kind: ROCKET, x: e.x - 8, y: e.y - 14, vx: Math.cos(a) * 120, vy: Math.sin(a) * 120, life: 2.4, dmg: 0, pid: -1, pierce: 0 });
        }
        e.altCd = 2.6 + this.rand();
        break;
      case "rockettank":
        // A ripple of six rockets, up and over onto the target
        for (let j = 0; j < 6; j++)
          this.at(j * 0.16, () => {
            if (!e.alive) return;
            this.missiles.push({ x: e.x - 6, y: e.y - 22, tx: tgt.x + (this.rand() - 0.5) * 60, ty: surfaceY(this.level, tgt.x, 0), speed: 160, max: 520, up: false, boost: true, owner: "enemy", cls: 0 });
            this.puff(e.x - 6, e.y - 22, 3);
          });
        e.altCd = 7 + this.rand() * 3;
        break;
      case "launcher": {
        // Raise the missile, then fire it: usually a big one, now and then an enormous one
        e.hover = Math.min(1, e.hover + dt * 0.5);
        if (e.hover < 1) return;
        const roll = this.rand();
        const cls = roll < 0.14 ? 3 : roll < 0.55 ? 2 : 1;
        this.missiles.push({ x: e.x + 18, y: e.y - 30, tx: tgt.x + (this.rand() - 0.5) * 30, ty: surfaceY(this.level, tgt.x, 0), speed: 90, max: [520, 560, 600, 520][cls], up: false, boost: true, owner: "enemy", cls });
        this.popups.push({ x: e.x, y: e.y - 40, text: cls === 3 ? "ICBM LAUNCH!" : "MISSILE LAUNCH", c: "#ff3b3b", t: 0 });
        for (let j = 0; j < 8; j++) this.puff(e.x + 18 + (this.rand() - 0.5) * 20, e.y - 10, 5);
        this.shake = Math.max(this.shake, cls === 3 ? 6 : 3);
        e.hover = 0;
        e.altCd = 11 + this.rand() * 5;
        break;
      }
    }
    // A carrier, parked: its troops pile out of the back, a mix of everything
    if (k === "apc" && !e.vx && e.burst < 200 && e.altCd <= 0 && e.spin < 6 && this.enemies.length < this.cap()) {
      e.altCd = 0.4;
      e.spin++;
      const pool: EnemyKind[] = ["mook", "minigunner", "rusher", "dynamiter", "laser", "engineer", "officer", "rocket", "shield"];
      const m = this.vary(this.makeEnemy(pool[Math.floor(this.rand() * pool.length)], e.x + 20, e.y - 4));
      m.vy = -120;
      m.vx = 30;
      m.ground = false;
      m.alert = true;
      m.home = e.x + 30;
      this.enemies.push(m);
    }
  }

  /**
   * Helicopters: an attack ship with rocket pods and missiles, a twin-rotor transport that drops a
   * stick of paratroopers and a door gunner's fire, a fast little scout with a minigun.
   */
  private updateHeliType(e: Enemy, dt: number, view: View) {
    const k = e.kind;
    e.life -= dt;
    e.anim += dt;
    const weave = k === "scoutheli" ? Math.sin(this.t * 1.6 + e.home) * 0.18 : Math.sin(this.t * 0.4 + e.home) * 0.08;
    const tx = e.life > 0 ? view.x + view.w * ((k === "transport" ? 0.55 : 0.7) + weave) : view.x + view.w + 160;
    e.x += (tx - e.x) * (1 - Math.exp(-dt * (k === "scoutheli" ? 2 : 1)));
    e.y += (view.y + e.hover + Math.sin(this.t * 2 + e.home) * 4 - e.y) * (1 - Math.exp(-dt * 2));
    e.facing = -1;
    if (e.life < 0 && e.x > view.x + view.w + 80) {
      e.alive = false;
      e.deadT = 0;
      return;
    }
    const p = this.nearestFoe(e, 300, false, 300);
    if (!p || e.life <= 0) return;
    e.aimA = Math.atan2(p.y - e.y, p.x - e.x);
    e.fireCd -= dt;
    e.altCd -= dt;
    if (e.fireCd <= 0) {
      this.enemyShot(e, p, 200, 1, k === "scoutheli" ? 0.25 : 0.18);
      e.burst++;
      e.fireCd = k === "scoutheli" ? (e.burst % 16 === 0 ? 1 : 0.05) : e.burst % 5 === 0 ? 1.3 : 0.1;
    }
    if (e.altCd > 0) return;
    if (k === "attackheli") {
      // A pair of rockets, and every other time a missile
      for (let j = 0; j < 2; j++) {
        const a = e.aimA + (j - 0.5) * 0.1;
        this.shots.push({ kind: ROCKET, x: e.x - 6, y: e.y + 6, vx: Math.cos(a) * 120, vy: Math.sin(a) * 120, life: 2.4, dmg: 0, pid: -1, pierce: 0 });
      }
      e.burst++;
      if (e.burst % 2) this.missiles.push({ x: e.x, y: e.y + 6, tx: p.x, ty: surfaceY(this.level, p.x, 0), speed: 140, max: 480, up: false, owner: "enemy", cls: 0 });
      e.altCd = 2.6 + this.rand();
    } else if (k === "transport") {
      // A stick of paratroopers out of the back
      for (let n = 0; n < 4 && this.enemies.length < this.cap(); n++) {
        const m = this.makeEnemy((["mook", "minigunner", "rusher", "dynamiter", "laser"] as EnemyKind[])[Math.floor(this.rand() * 5)], e.x + 8 + n * 10, e.y + 8);
        m.chute = true;
        m.ground = false;
        this.enemies.push(m);
      }
      e.altCd = 9;
      e.life = Math.min(e.life, 4);
    } else e.altCd = 99;
  }

  /**
   * Gun emplacements, dug in: flak guns that fill the sky with bursts, howitzers lobbing heavy
   * shells from far off, SAM sites firing missiles at aircraft (or anything), rocket batteries.
   */
  private updateEmplacement(e: Enemy, dt: number) {
    const k = e.kind;
    e.facing = -1;
    this.integrate(e, dt, SIZE[k][0], SIZE[k][1]);
    e.hover = Math.max(0, e.hover - dt);
    if (!this.inView(e)) return;
    const air: Foe | null = k === "flakgun" || k === "samsite" ? (this.allies.find((a) => !a.dead && ALLY_AIR.has(a.kind) && Math.abs(a.x - e.x) < 420) ?? this.jets.find((j) => j.ally && !j.down && Math.abs(j.x - e.x) < 380) ?? this.players.find((q) => !q.dead && q.jet > 0 && Math.abs(q.x - e.x) < 360) ?? null) : null;
    const tgt = air ?? this.nearestFoe(e, k === "howitzer" ? 620 : 420, false, 300);
    if (!tgt) return;
    e.aimA = Math.atan2(tgt.y - 10 - (e.y - 14), tgt.x - e.x);
    e.fireCd -= dt * 1.4;
    if (e.fireCd > 0) return;
    const gx = e.x + Math.cos(e.aimA) * 12;
    const gy = e.y - 14 + Math.sin(e.aimA) * 12;
    switch (k) {
      case "flakgun":
        if (air) {
          const d = Math.hypot(air.x - gx, air.y - gy);
          this.shots.push({ kind: FLAK, x: gx, y: gy, vx: Math.cos(e.aimA) * 330, vy: Math.sin(e.aimA) * 330, life: d / 330 + (this.rand() - 0.5) * 0.1, dmg: 0, pid: -1, pierce: 0 });
        } else this.enemyShot(e, tgt, 220, 1, 0.2, 14);
        this.flashes.push({ x: gx, y: gy, t: 0.05, c: "#ffe27a" });
        e.burst++;
        e.fireCd = e.burst % 8 === 0 ? 0.9 : 0.1;
        break;
      case "howitzer": {
        const v = lobAt(gx, gy, tgt.x + (this.rand() - 0.5) * 30, tgt.y, SHOT_GRAV[CANNON], 170);
        this.shots.push({ kind: CANNON, x: gx, y: gy, vx: v.vx, vy: v.vy, life: 6, dmg: 0, pid: -1, pierce: 0 });
        this.flashes.push({ x: gx, y: gy, t: 0.14, c: "#ffe27a" });
        for (let j = 0; j < 3; j++) this.puff(gx, gy, 5);
        this.shake = Math.max(this.shake, 3);
        e.hover = 0.3;
        e.fireCd = 3 + this.rand();
        break;
      }
      case "samsite":
        for (let j = 0; j < 2; j++)
          this.at(j * 0.3, () => {
            if (!e.alive || tgt.dead) return;
            const flying = air !== null;
            this.missiles.push({ x: e.x, y: e.y - 18, tx: tgt.x, ty: flying ? tgt.y : surfaceY(this.level, tgt.x, 0), speed: 120, max: 620, up: false, boost: !flying, owner: "enemy", cls: flying ? 0 : 1 });
            this.puff(e.x, e.y - 16, 4);
          });
        e.fireCd = 3.6 + this.rand();
        break;
      default:
        for (let j = 0; j < 8; j++)
          this.at(j * 0.12, () => {
            if (!e.alive) return;
            this.missiles.push({ x: e.x, y: e.y - 16, tx: tgt.x + (this.rand() - 0.5) * 70, ty: surfaceY(this.level, tgt.x, 0), speed: 170, max: 520, up: false, boost: true, owner: "enemy", cls: 0 });
          });
        e.fireCd = 7 + this.rand() * 3;
    }
  }

  /**
   * An armoured train coming down the line: an engine and its cars, machine guns all along it,
   * a flak car shooting at anything in the air, a rocket car firing salvos.
   */
  private updateTrain(e: Enemy, dt: number) {
    e.facing = -1;
    e.vx = e.x > Math.max(e.home - 220, this.advance + 150) ? -22 : 0;
    this.integrate(e, dt, 64, 15);
    e.anim += dt * Math.abs(e.vx) * 0.3;
    if (this.rand() < 0.4) this.puff(e.x - 52, e.y - 30, 3);
    if (!this.inView(e)) return;
    const p = this.nearestFoe(e, 360, false, 220);
    if (!p) return;
    e.aimA = Math.atan2(p.y - 9 - (e.y - 18), p.x - e.x);
    e.fireCd -= dt * 1.4;
    e.altCd -= dt;
    if (e.fireCd <= 0) {
      // Guns all down its length
      const gx = e.x - 40 + this.rand() * 80;
      const a = Math.atan2(p.y - 9 - (e.y - 18), p.x - gx) + (this.rand() - 0.5) * 0.15;
      this.shots.push({ kind: ENEMY_BULLET, x: gx, y: e.y - 18, vx: Math.cos(a) * 210, vy: Math.sin(a) * 210, life: 1.6, dmg: 1, pid: -1, pierce: 0 });
      this.flashes.push({ x: gx, y: e.y - 18, t: 0.05, c: "#ff9a2a" });
      e.burst++;
      e.fireCd = e.burst % 10 === 0 ? 0.8 : 0.07;
    }
    if (e.altCd <= 0) {
      for (let j = 0; j < 6; j++)
        this.at(j * 0.14, () => {
          if (!e.alive) return;
          this.missiles.push({ x: e.x + 30, y: e.y - 22, tx: p.x + (this.rand() - 0.5) * 60, ty: surfaceY(this.level, p.x, 0), speed: 160, max: 520, up: false, boost: true, owner: "enemy", cls: 0 });
        });
      e.altCd = 6 + this.rand() * 3;
    }
  }

  /**
   * A submarine: patrols the ocean, surfaces to fire missiles and its deck gun, then dives again
   * (under water it can't be hit). `spin` 0: on the surface, 1: submerged.
   */
  private updateSubmarine(e: Enemy, dt: number) {
    e.anim += dt;
    if (Math.abs(e.x - e.home) > 30) e.facing = e.x > e.home ? -1 : 1;
    e.x += e.facing * 6 * dt;
    e.life -= dt;
    if (e.life <= 0) {
      e.spin = e.spin ? 0 : 1;
      e.life = e.spin ? 4 + this.rand() * 3 : 7 + this.rand() * 4;
      for (let k = 0; k < 10; k++) this.add({ k: MIST, x: e.x + (this.rand() - 0.5) * 50, y: e.y - 2, vx: (this.rand() - 0.5) * 20, vy: -20 - this.rand() * 30, life: 1, c: "", s: 2 });
    }
    if (e.spin || !this.inView(e)) return;
    const p = this.nearestFoe(e, 460, false, 300);
    if (!p) return;
    e.fireCd -= dt * 1.3;
    e.altCd -= dt;
    if (e.fireCd <= 0) {
      this.enemyShot(e, p, 200, 1, 0.2, 10);
      e.burst++;
      e.fireCd = e.burst % 6 === 0 ? 1.2 : 0.1;
    }
    if (e.altCd <= 0) {
      for (let j = 0; j < 2; j++)
        this.at(j * 0.4, () => {
          if (!e.alive || e.spin) return;
          this.missiles.push({ x: e.x + 4, y: e.y - 8, tx: p.x + (this.rand() - 0.5) * 30, ty: surfaceY(this.level, p.x, 0), speed: 100, max: 560, up: false, boost: true, owner: "enemy", cls: 1 });
          for (let k = 0; k < 4; k++) this.puff(e.x + 4, e.y - 8, 4);
        });
      e.altCd = 5 + this.rand() * 3;
    }
  }

  /** Keep the allied column up to strength: more of it when the view is wide (zoomed out). */
  private manageAllies(dt: number, view: View) {
    if ((this.allyCd -= dt) > 0 || this.phase === "clear") return;
    const ground = this.allies.filter((a) => !a.dead && !ALLY_AIR.has(a.kind));
    const air = this.allies.filter((a) => !a.dead && ALLY_AIR.has(a.kind));
    const wantG = clamp(Math.round(view.w / 200), 2, 9);
    const wantA = clamp(Math.round(view.w / 460), 1, 3);
    const r = this.rand;
    this.allyCd = 2 + r() * 2;
    const slotOf = (list: Ally[]) => {
      let s = 0;
      while (list.some((a) => a.slot === s)) s++;
      return s;
    };
    if (ground.length < wantG) {
      const roll = r();
      const kind: AllyKind = roll < 0.34 ? "mbt" : roll < 0.56 ? "ifv" : roll < 0.7 ? "scoutcar" : roll < 0.86 ? "mlrs" : "aa";
      const x = Math.max(this.level.start * TILE + 24, view.x - 20);
      const y = surfaceY(this.level, x, 0);
      // (not into the sea)
      if (kindAt(this.level, Math.floor(x / TILE), Math.floor(y / TILE) - 1) !== WATER) this.allies.push(this.makeAlly(kind, x, y, slotOf(ground)));
    } else if (air.length < wantA) {
      const kind: AllyKind = !air.some((a) => a.kind === "lifter") && r() < 0.3 ? "lifter" : "aheli";
      this.allies.push(this.makeAlly(kind, view.x - 40, view.y + 30, slotOf(air)));
    }
  }

  private makeAlly(kind: AllyKind, x: number, y: number, slot: number): Ally {
    return { kind, x, y, vx: 0, vy: 0, ground: true, facing: 1, hp: ALLY_HP[kind], maxHp: ALLY_HP[kind], dead: false, deadT: 0, fireCd: 1 + this.rand(), altCd: 2 + this.rand() * 3, burst: 0, scanT: 0, aimA: 0, anim: 0, slot, hit: 0, target: null };
  }

  /** Allied vehicles hold their place in the column behind (or, in the air, over) the squad and fight whatever's in view. */
  private updateAlly(a: Ally, dt: number, view: View) {
    const L = this.level;
    const air = ALLY_AIR.has(a.kind);
    a.hit = Math.max(0, a.hit - dt);
    if (a.dead) {
      a.deadT -= dt;
      if (air) {
        a.vy = Math.min(a.vy + 300 * dt, 220);
        a.x += 30 * dt;
        a.y += a.vy * dt;
        if (this.rand() < 0.6) this.puff(a.x, a.y, 3);
        if (this.solid(a.x, a.y + 6) || a.y > LEVEL_H) {
          this.explode(a.x, a.y, 26, -5, false);
          a.deadT = 0;
        }
      } else if (this.rand() < 0.3) this.puff(a.x, a.y - 10, 3);
      return;
    }
    if (air) {
      const tx = Math.max(view.x + 30, this.advance + 30 + a.slot * 50 + Math.sin(this.t * 0.5 + a.slot) * 20);
      const ty = Math.max(view.y + 22, surfaceY(L, a.x, view.y) - 70 - (a.slot % 2) * 20) + Math.sin(this.t * 2 + a.slot) * 3;
      a.x += (tx - a.x) * (1 - Math.exp(-dt * 1.1));
      a.y += (ty - a.y) * (1 - Math.exp(-dt * 1.5));
      a.anim += dt;
    } else {
      // Drive to its place in the column, climbing what it can, waiting at what it can't
      const tx = Math.max(view.x + 28, this.advance - 40 - a.slot * 36);
      const dx = tx - a.x;
      a.vx = Math.abs(dx) > 6 ? sign(dx) * Math.min(44, Math.abs(dx) * 1.5) : 0;
      if (a.vx) {
        const nx = a.x + a.vx * dt;
        const ground = surfaceY(L, nx, a.y - 44);
        // (no further than the shore)
        if (ground > a.y - 42 && kindAt(L, Math.floor(nx / TILE), Math.floor(ground / TILE) - 1) !== WATER) a.x = nx;
        else a.vx = 0;
      }
      const g = surfaceY(L, a.x, a.y - 44);
      a.y = g < a.y ? g : Math.min(g, a.y + 200 * dt);
      a.anim += dt * Math.abs(a.vx) * 0.3;
    }
    a.facing = 1;
    // What to shoot
    a.scanT -= dt;
    if (a.scanT <= 0) {
      a.scanT = 0.3;
      a.target = this.allyTarget(a, view);
    }
    const tgt = a.target?.alive ? a.target : null;
    if (!tgt) return;
    const aim = this.aimPoint(tgt) ?? { x: tgt.x, y: tgt.y };
    const gy = air ? a.y + 4 : a.y - 16;
    const gx = a.x + 10;
    a.aimA = Math.atan2(aim.y - gy, aim.x - gx);
    a.fireCd -= dt;
    a.altCd -= dt;
    const bullet = (dmg: number, speed: number, spread: number) => {
      const ang = a.aimA + (this.rand() - 0.5) * spread;
      this.shots.push({ kind: BULLET, x: gx, y: gy, vx: Math.cos(ang) * speed, vy: Math.sin(ang) * speed, life: 1.2, dmg, pid: -5, pierce: 0 });
      this.flashes.push({ x: gx, y: gy, t: 0.05, c: "#9ad8ff" });
    };
    if (a.fireCd <= 0) {
      if (a.kind === "aa" && FLYING.has(tgt.kind)) {
        const d = Math.hypot(aim.x - gx, aim.y - gy);
        this.shots.push({ kind: FLAK, x: gx, y: gy, vx: Math.cos(a.aimA) * 320, vy: Math.sin(a.aimA) * 320, life: d / 320, dmg: 0, pid: -5, pierce: 0 });
        a.burst++;
        a.fireCd = a.burst % 6 === 0 ? 0.9 : 0.14;
      } else if (a.kind !== "lifter" && a.kind !== "mlrs") {
        bullet(a.kind === "ifv" ? 2 : 1, 380, 0.12);
        a.burst++;
        a.fireCd = a.burst % (a.kind === "ifv" ? 3 : 5) === 0 ? 0.9 : a.kind === "ifv" ? 0.15 : 0.1;
      }
    }
    if (a.altCd > 0) return;
    switch (a.kind) {
      case "mbt": {
        const v = lobAt(gx, gy, aim.x, aim.y, SHOT_GRAV[CANNON], 320);
        this.shots.push({ kind: CANNON, x: gx, y: gy, vx: v.vx, vy: v.vy, life: 4, dmg: 0, pid: -5, pierce: 0 });
        this.flashes.push({ x: gx, y: gy, t: 0.1, c: "#ffe27a" });
        this.puff(gx, gy, 5);
        a.altCd = 2.4 + this.rand();
        break;
      }
      case "mlrs":
        for (let j = 0; j < 6; j++)
          this.at(j * 0.15, () => {
            if (a.dead || !tgt.alive) return;
            this.missiles.push({ x: a.x - 4, y: a.y - 22, tx: tgt.x + (this.rand() - 0.5) * 50, ty: surfaceY(this.level, tgt.x, 0), speed: 160, max: 520, up: false, boost: true, owner: "ally", cls: 0 });
          });
        a.altCd = 8 + this.rand() * 3;
        break;
      case "aheli":
        for (let j = 0; j < 2; j++) {
          const ang = a.aimA + (j - 0.5) * 0.1;
          this.shots.push({ kind: ROCKET, x: gx, y: gy + 2, vx: Math.cos(ang) * 120, vy: Math.sin(ang) * 120, life: 2.4, dmg: 0, pid: -5, pierce: 0 });
        }
        a.altCd = 2.6 + this.rand();
        break;
      case "lifter":
        // Drops the squad a supply crate
        this.drops.push({ x: a.x, y: a.y + 10, vy: 0, weapon: this.rand() < 0.2 ? -1 : Math.floor(this.rand() * WEAPONS.length), ground: false, t: 0 });
        a.altCd = 10 + this.rand() * 5;
        break;
    }
  }

  /** An ally's target: something in view and in range; flak goes for aircraft, everyone else for armour first. */
  private allyTarget(a: Ally, view: View): Enemy | null {
    const range = a.kind === "mlrs" ? 520 : a.kind === "aa" ? 380 : ALLY_AIR.has(a.kind) ? 300 : 340;
    let best: Enemy | null = null;
    let bestS = Infinity;
    const gy = ALLY_AIR.has(a.kind) ? a.y : a.y - 16;
    for (const e of this.enemies) {
      if (!e.alive || e.x < view.x || e.x > view.x + view.w - 4) continue;
      const q = this.aimPoint(e);
      if (!q) continue;
      const d = Math.hypot(q.x - a.x, q.y - gy);
      if (d > range) continue;
      const flying = FLYING.has(e.kind);
      if (a.kind === "aa" ? !flying && best && FLYING.has(best.kind) : false) continue;
      if ((a.kind === "mbt" || a.kind === "ifv") && !flying && !lineOfSight(this.level, a.x + 10, gy, q.x, q.y)) continue;
      const score = d * (a.kind === "aa" && flying ? 0.3 : ARMOURED.has(e.kind) || e.boss ? 0.6 : 1);
      if (score < bestS) {
        bestS = score;
        best = e;
      }
    }
    return best;
  }

  hurtAlly(a: Ally, dmg: number) {
    if (a.dead) return;
    a.hp -= dmg;
    a.hit = 0.08;
    if (a.hp > 0) return;
    a.dead = true;
    a.deadT = ALLY_AIR.has(a.kind) ? 8 : 4;
    a.vy = -30;
    if (!ALLY_AIR.has(a.kind)) {
      this.explode(a.x, a.y - 8, 26, -5, false);
      this.fires.push({ x: a.x, y: a.y, t: 4 });
    }
    this.popups.push({ x: a.x, y: a.y - 24, text: "ALLY DOWN", c: "#5ab0ff", t: 0 });
    this.allyCd = Math.max(this.allyCd, 5);
  }

  /** Manholes: once the squad is near, enemies climb out of the sewers, the cover flying off. */
  private updateHatches(dt: number, view: View) {
    for (const h of this.hatches) {
      h.open = Math.max(0, h.open - dt);
      if (h.n <= 0 || (h.cd -= dt) > 0 || h.x > view.x + view.w || this.enemies.length >= this.cap()) continue;
      if (!this.players.some((p) => !p.dead && Math.abs(p.x - h.x) < 170)) continue;
      const pool: EnemyKind[] = ["mook", "rusher", "dynamiter", "minigunner", "engineer", "flamer"];
      const e = this.vary(this.makeEnemy(pool[Math.floor(this.rand() * pool.length)], h.x, h.y));
      e.vy = -260;
      e.vx = 20;
      e.ground = false;
      e.alert = true;
      this.enemies.push(e);
      h.n--;
      h.cd = 0.9 + this.rand() * 0.8;
      h.open = 0.6;
      this.popups.push({ x: h.x, y: h.y - 12, text: "!", c: "#ff3b3b", t: 0 });
      for (let k = 0; k < 4; k++) this.add({ k: DEBRIS, x: h.x, y: h.y - 2, vx: (this.rand() - 0.5) * 80, vy: -120 - this.rand() * 80, life: 1, c: "#3a3a40", s: 2 });
    }
  }

  /** The sky fortress: an armoured airship that drifts over, carpet-bombing and dropping paratroopers. */
  private updateAirship(e: Enemy, dt: number, view: View) {
    const tx = view.x + view.w * (0.62 + Math.sin(this.t * 0.25) * 0.18);
    e.x += (tx - e.x) * (1 - Math.exp(-dt * 0.6));
    e.y += (view.y + 30 + e.hover + Math.sin(this.t * 0.9) * 4 - e.y) * (1 - Math.exp(-dt * 1.5));
    e.facing = -1;
    e.anim += dt;
    e.fireCd -= dt;
    e.altCd -= dt;
    const p = this.players.find((q) => !q.dead && Math.abs(q.x - e.x) < 260);
    // Now and then, one enormous bomb
    e.nade -= dt;
    if (e.nade <= 0) {
      e.nade = 9 + this.rand() * 6;
      this.shots.push({ kind: BIGBOMB, x: e.x, y: e.y + 16, vx: -12, vy: 30, life: 8, dmg: 0, pid: -1, pierce: 0 });
      this.popups.push({ x: e.x, y: e.y + 26, text: "BIG BOMB!", c: "#ff3b3b", t: 0 });
    }
    if (e.fireCd <= 0) {
      // A stick of bombs out of the belly
      for (let k = 0; k < 4; k++) this.at(k * 0.18, () => e.alive && this.shots.push({ kind: SHELL, x: e.x - 20 + k * 12, y: e.y + 14, vx: -20, vy: 60, life: 4, dmg: 0, pid: -1, pierce: 0 }));
      e.fireCd = 2.2 + this.rand();
    }
    if (p && e.burst % 50 < 8 && this.rand() < dt * 10) this.enemyShot(e, p, 180, 1, 0.2, -12);
    e.burst++;
    if (e.altCd <= 0) {
      // Paratroopers from the gondola
      for (let n = 0; n < 3 && this.enemies.length < this.cap(); n++) {
        const m = this.makeEnemy(this.rand() < 0.3 ? "minigunner" : "mook", e.x + (n - 1) * 14, e.y + 16);
        m.chute = true;
        m.ground = false;
        this.enemies.push(m);
      }
      e.altCd = 7 + this.rand() * 3;
    }
  }

  /** The doom cannon: a huge gun on a carriage that throws explosive drums in high arcs, a flak gun beside it. */
  private updateCannon(e: Enemy, dt: number) {
    e.facing = -1;
    this.integrate(e, dt, 30, 30);
    const p = this.nearestPlayer(e, 420, false, 300);
    e.anim += dt;
    if (!p) return;
    e.aimA = Math.atan2(p.y - (e.y - 30), p.x - e.x);
    e.fireCd -= dt;
    e.altCd -= dt;
    if (e.altCd <= 0) {
      // A volley of three
      for (let k = 0; k < 3; k++)
        this.at(k * 0.35, () => {
          if (!e.alive) return;
          const gx = e.x - 26;
          const gy = e.y - 44;
          const v = lobAt(gx, gy, p.x + (k - 1) * 30, p.y, SHOT_GRAV[CANNON], 170);
          this.shots.push({ kind: CANNON, x: gx, y: gy, vx: v.vx, vy: v.vy, life: 5, dmg: 0, pid: -1, pierce: 0 });
          this.flashes.push({ x: gx, y: gy, t: 0.14, c: "#ffe27a" });
          for (let j = 0; j < 3; j++) this.puff(gx, gy, 5);
          this.shake = Math.max(this.shake, 4);
          e.hover = 0.25;
        });
      e.altCd = 3 + this.rand();
    }
    e.hover = Math.max(0, e.hover - dt);
    if (e.fireCd <= 0) {
      this.enemyShot(e, p, 200, 1, 0.3, 24);
      e.burst++;
      e.fireCd = e.burst % 10 === 0 ? 1.5 : 0.07;
    }
  }

  /** The spider tank: six legs, a plasma cannon, and it lays crawler eggs as it comes. */
  private updateSpider(e: Enemy, dt: number) {
    e.vx = e.x > this.advance + 160 ? -14 : 0;
    e.facing = -1;
    this.integrate(e, dt, 26, 34);
    e.anim += dt * (e.vx ? 3 : 0.5);
    const p = this.nearestPlayer(e, 340, false, 220);
    if (!p) return;
    e.aimA = Math.atan2(p.y - 9 - (e.y - 30), p.x - e.x);
    e.fireCd -= dt;
    e.altCd -= dt;
    if (e.fireCd <= 0) {
      const gx = e.x - 20;
      const gy = e.y - 30;
      const a = e.aimA + (this.rand() - 0.5) * 0.1;
      this.shots.push({ kind: ELASER, x: gx, y: gy, vx: Math.cos(a) * 380, vy: Math.sin(a) * 380, life: 1.2, dmg: 2, pid: -1, pierce: 0 });
      this.flashes.push({ x: gx, y: gy, t: 0.06, c: "#c77dff" });
      e.burst++;
      e.fireCd = e.burst % 5 === 0 ? 1.4 : 0.18;
    }
    if (e.altCd <= 0 && this.enemies.length < this.cap()) {
      this.barrels.push({ x: e.x, y: e.y, vx: 0, vy: 0, ground: true, facing: -1, type: "egg", hp: 3, fuse: -1, owner: -3, alive: true });
      e.altCd = 6 + this.rand() * 3;
    }
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
    if (e.altCd > 0 || this.enemies.length >= this.cap()) return;
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

  /**
   * A battleship riding at anchor in the harbour: its two turrets lob heavy shells at the squad,
   * its deck gun sprays the pier, smoke pours from its funnel.
   */
  private updateBattleship(e: Enemy, dt: number) {
    e.anim += dt;
    e.facing = -1;
    e.scanT -= dt;
    if (e.scanT <= 0) {
      e.scanT = 0.4;
      e.target = this.nearestPlayer(e, 420, false, 260);
    }
    if (this.rand() < 0.5) this.puff(e.x + 6, e.y - 34, 3);
    const p = e.target && !e.target.dead ? e.target : null;
    if (p) e.aimA = Math.atan2(p.y - (e.y - 22), p.x - e.x);
    e.fireCd -= dt;
    if (p && e.fireCd <= 0) {
      // A salvo from both turrets
      e.fireCd = 2.4 + this.rand() * 1.4;
      for (const dx of [-22, 18]) {
        const gx = e.x + dx + Math.cos(e.aimA) * 10;
        const gy = e.y - 20;
        const v = lobAt(gx, gy, p.x + (this.rand() - 0.5) * 40, p.y, SHOT_GRAV[CANNON], 200);
        this.shots.push({ kind: CANNON, x: gx, y: gy, vx: v.vx, vy: v.vy, life: 4, dmg: 0, pid: -1, pierce: 0 });
        this.flashes.push({ x: gx, y: gy, t: 0.08, c: "#ffb020" });
        this.puff(gx, gy, 4);
      }
      this.shake = Math.max(this.shake, 2);
    }
    e.altCd -= dt;
    // Anti-aircraft mounts: flak at helicopters, jets and anyone on a jetpack
    e.hover = Math.max(0, e.hover - dt);
    const air: Foe | null =
      this.allies.find((a) => !a.dead && ALLY_AIR.has(a.kind) && Math.abs(a.x - e.x) < 420) ??
      this.jets.find((j) => j.ally && !j.down && Math.abs(j.x - e.x) < 380) ??
      this.players.find((q) => !q.dead && q.jet > 0 && Math.abs(q.x - e.x) < 320) ??
      null;
    if (air && e.hover <= 0) {
      for (const dx of [-30, 8, 30]) {
        const gx = e.x + dx;
        const gy = e.y - 16;
        const a = Math.atan2(air.y - gy, air.x - gx) + (this.rand() - 0.5) * 0.12;
        const d = Math.hypot(air.x - gx, air.y - gy);
        this.shots.push({ kind: FLAK, x: gx, y: gy, vx: Math.cos(a) * 340, vy: Math.sin(a) * 340, life: d / 340, dmg: 0, pid: -1, pierce: 0 });
      }
      e.hover = 0.35;
    }
    if (p && e.altCd <= 0) {
      this.enemyShot(e, p, 200, 1, 0.25, 14);
      e.burst++;
      e.altCd = e.burst % 8 === 0 ? 1.6 : 0.12;
    }
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
    e.vx = e.x > stop ? (tank ? -26 : -20) : 0;
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
      this.explode(e.x, y, e.kind === "gunship" || e.kind === "airship" ? 50 : e.kind === "drone" || e.kind === "jetpacker" ? 14 : e.kind === "transport" ? 40 : 32, -3, true);
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
    if (ARMOR_KINDS.has(e.kind) || e.kind === "exosuit") {
      this.explode(e.x, e.y - 10, e.kind === "exosuit" ? 22 : 32, pid, true);
      this.fires.push({ x: e.x, y: e.y, t: 4 });
    }
    if (e.kind === "dynamiter") this.explode(e.x, e.y - 8, 16, pid, false);
    if (e.sapper) this.at(0.15, () => this.explode(e.x, e.y - 8, 22 * Math.max(1, e.scale), pid, false));
    if (e.scale > 2) {
      this.shake = Math.max(this.shake, 3);
      this.popups.push({ x: e.x, y: e.y - 40, text: "BRUTE DOWN", c: "#ffcc33", t: 0 });
    }
    if (e.kind === "truck") {
      this.explode(e.x, e.y - 10, 30, pid, true);
      this.fires.push({ x: e.x - 8, y: e.y, t: 5 }, { x: e.x + 10, y: e.y, t: 3 });
    }
    if (e.kind === "boat") this.explode(e.x, e.y - 6, 24, pid, false);
    if (e.kind === "train") {
      // Car after car goes up
      e.deadT = 1.6;
      for (let k = 0; k < 8; k++) this.at(k * 0.14, () => this.explode(e.x - 56 + k * 16, e.y - 10, 22, pid, false));
      this.fires.push({ x: e.x - 30, y: e.y, t: 6 }, { x: e.x + 20, y: e.y, t: 6 });
    }
    if (e.kind === "submarine") {
      e.deadT = 4;
      for (let k = 0; k < 5; k++) this.at(k * 0.2, () => this.explode(e.x - 24 + this.rand() * 48, e.y - 4, 18, pid, false));
    }
    if (e.kind === "flakgun" || e.kind === "howitzer" || e.kind === "samsite" || e.kind === "rocketpod") {
      this.explode(e.x, e.y - 8, 26, pid, true);
      this.fires.push({ x: e.x, y: e.y, t: 4 });
    }
    if (e.kind === "battleship") {
      // It goes up stem to stern, then the magazine, then it settles and sinks
      e.deadT = 5;
      for (let k = 0; k < 9; k++) this.at(k * 0.16, () => this.explode(e.x - 40 + this.rand() * 80, e.y - 6 - this.rand() * 24, 14 + this.rand() * 12, pid, false));
      this.at(1.6, () => this.explode(e.x, e.y - 12, 46, pid, false));
    }
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
    if (b.type === "mine" && b.fuse < 0 && this.players.some((p) => !p.dead && Math.abs(p.x - b.x) < 6 && Math.abs(p.y - b.y) < 8)) {
      // Click.
      this.popups.push({ x: b.x, y: b.y - 12, text: "!", c: "#ff3b3b", t: 0 });
      this.lightBarrel(b, -3, 0.12);
    }
    const vx = b.vx;
    const rocket = (b.type === "propane" || b.type === "cylinder") && b.fuse >= 0;
    this.integrate(b, dt, hw, hh, rocket ? 0.35 : 1);
    if (b.ground) b.vx = vx * 0.5;
    if (rocket) {
      // A jet of flame and smoke out of the bottom
      for (let k = 0; k < 2; k++) this.add({ k: FLAME, x: b.x + (this.rand() - 0.5) * 3, y: b.y + 1, vx: (this.rand() - 0.5) * 20, vy: 60 + this.rand() * 40, life: 0.25, c: "", s: 2 });
      this.puff(b.x, b.y + 3, 2);
    }
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
    for (let k = 0; k < n && this.enemies.length < this.cap(); k++) {
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
    if (b.type === "propane" || b.type === "cylinder") {
      // The valve blows: it takes off on a jet of flame, then goes up at the top of its climb
      b.fuse = 0.55 + this.rand() * 0.35;
      b.vy = -330 - this.rand() * 90;
      b.vx = (this.rand() - 0.5) * 60;
      b.ground = false;
    }
  }

  private updateShots(dt: number, view: View) {
    const L = this.level;
    const keep: Shot[] = [];
    for (const s of this.shots) {
      s.life -= dt;
      if (s.kind === ROCKET) {
        s.vx *= 1 + 2.2 * dt;
        s.vy *= 1 + 2.2 * dt;
        if (this.rand() < 0.8) this.puff(s.x, s.y, 1 + this.rand());
        this.add({ k: FLAME, x: s.x - sign(s.vx) * 2, y: s.y, vx: -s.vx * 0.1, vy: 0, life: 0.12, c: "", s: 1 });
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
        if (s.kind === SAW) {
          // Ricochets
          if (this.solid(nx, s.y)) s.vx *= -1;
          if (this.solid(s.x, ny)) s.vy *= -1;
          this.add({ k: SPARK, x: s.x, y: s.y, vx: (this.rand() - 0.5) * 60, vy: -40, life: 0.15, c: "#ffffff", s: 1 });
          keep.push(s);
          continue;
        }
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
    if (e.kind === "submarine" && e.spin > 0) return false;
    const [hw, hh] = dims(e);
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
      for (const a of this.allies) {
        if (a.dead) continue;
        const [hw, hh] = ALLY_SIZE[a.kind];
        const cy = ALLY_AIR.has(a.kind) ? a.y : a.y - hh / 2;
        if (Math.abs(a.x - s.x) > hw || Math.abs(cy - s.y) > hh / 2 + 1) continue;
        if (EXPLOSIVE.has(s.kind)) {
          this.explodeShot(s);
          return true;
        }
        this.hurtAlly(a, s.dmg * 0.5);
        this.add({ k: SPARK, x: s.x, y: s.y, vx: 0, vy: -30, life: 0.12, c: "#ffe27a", s: 1 });
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
    if (s.kind === BIGBOMB) {
      this.mushrooms.push({ x, y, t: 0.8 });
      this.waves.push({ x, y, t: 0.25 });
      this.flash = Math.max(this.flash, 0.4);
      this.shake = Math.max(this.shake, 14);
    }
    if (s.kind === NAPALM) {
      // A wall of fire along the ground
      const gy = surfaceY(this.level, x, y - 10);
      for (let k = -2; k <= 2; k++) this.fires.push({ x: x + k * 9, y: gy, t: 3 + this.rand() * 2 });
      for (let k = 0; k < 6; k++) this.add({ k: CINDER, x, y: gy - 4, vx: (this.rand() - 0.5) * 220, vy: -60 - this.rand() * 120, life: 1, c: "", s: 2 });
    }
    const r = s.kind === ROCKET && s.pid === -1 ? 14 : (BLAST_R[s.kind] ?? 18);
    // Enemy fire and artillery hurt the squad; its own explosives, the visitor's and the ICBMs don't
    this.explode(x, y, r, s.pid, s.pid === -3 || s.pid === -1);
  }

  /** An explosion: breaks blocks, scorches what's left, throws everyone about, sets off barrels. */
  explode(x: number, y: number, r: number, pid: number, hurtsPlayers: boolean) {
    const L = this.level;
    this.booms++;
    // The ground takes less than the people do, so the level stays runnable (the visitor's own blasts dig full craters)
    // The ground is tougher than it was: a blast wears blocks down by how close they are and how
    // big it is, so soft soil goes but stone, concrete and the rest hold against all but the
    // big ones (and granite against everything). The visitor's own clicks still dig clean.
    const dig = pid === -2 ? r : pid === -4 ? r * 0.6 : r * 0.45;
    for (const [c, row] of blastCells(L, x, y, dig)) {
      const i = cell(c, row);
      const kind = L.fg[i];
      if (pid === -2) {
        L.fg[i] = AIR;
        L.hp[i] = 0;
      } else {
        const d = Math.hypot(c * TILE + TILE / 2 - x, row * TILE + TILE / 2 - y);
        if (!damageCell(L, c, row, Math.max(1, Math.round((r / 9) * (1 - d / (dig + 4)))))) continue;
      }
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
    this.smashProps(x, y, r);
    for (const e of this.enemies) {
      if (!e.alive) continue;
      const aim = this.aimPoint(e) ?? { x: e.x, y: e.y };
      const d = Math.hypot(aim.x - x, aim.y - y);
      const reach = r + (e.boss ? SIZE[e.kind][0] : 8 * e.scale);
      if (d > reach) continue;
      const f = 1 - d / reach;
      // (bosses, armour and the big troopers take damage from blasts; the rest go flying)
      if (e.boss || ARMOURED.has(e.kind) || e.scale > 1.4) this.hurtEnemy(e, ((e.boss ? 14 : 10) * f + 3) * Math.max(1, r / 22), pid, e.x - x);
      else {
        this.killEnemy(e, pid, sign(e.x - x) * (80 + 140 * f), -(140 + 160 * f));
        // Close to the blast: they go flying on fire
        if (f > 0.25 && r >= 12) e.burn = 0.6 + f;
      }
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
    if (hurtsPlayers)
      for (const a of this.allies) {
        if (a.dead) continue;
        const d = Math.hypot(a.x - x, a.y - 8 - y);
        const reach = r + ALLY_SIZE[a.kind][0];
        if (d < reach) this.hurtAlly(a, (8 * (1 - d / reach) + 2) * Math.max(1, r / 24));
      }
    // Aircraft caught in it (flak, mostly): the enemy's guns bring down ours, ours theirs
    for (const j of this.jets) {
      if (j.down || Math.hypot(j.x - x, j.y - y) > r + 14) continue;
      if (j.ally ? hurtsPlayers : pid === -5 || pid >= 0) {
        j.hp -= 1;
        if (j.hp <= 0) {
          j.down = true;
          this.popups.push({ x: j.x, y: j.y - 10, text: j.ally ? "JET DOWN" : "+800", c: j.ally ? "#5ab0ff" : "#ffcc33", t: 0 });
        }
      }
    }
    this.blasts.push({ x, y, r: r * 0.9, t: 0 });
    this.fireball(x, y, r);
    const big = r > 40;
    const sparks = big ? 40 : 18;
    for (let n = 0; n < sparks; n++) {
      const a = this.rand() * Math.PI * 2;
      const v = 80 + this.rand() * (big ? 340 : 200);
      this.add({ k: SPARK, x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v - 60, life: 0.3 + this.rand() * 0.6, c: this.rand() < 0.5 ? "#ffe27a" : "#ff9a2a", s: 1 });
    }
    // Burning chunks thrown out in arcs
    for (let n = big ? 14 : 3 + Math.floor(r / 6); n > 0; n--) {
      const a = -Math.PI / 2 + (this.rand() - 0.5) * 2.4;
      const v = 90 + this.rand() * (big ? 260 : 150);
      this.add({ k: CINDER, x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v, life: 0.9 + this.rand() * 0.9, c: "", s: this.rand() < 0.3 ? 2 : 1 });
    }
    for (let n = 0; n < (big ? 12 : 6); n++) this.puff(x + (this.rand() - 0.5) * r, y + (this.rand() - 0.5) * r * 0.6, 2 + this.rand() * (big ? 8 : 3));
    const gy = groundOf(L, Math.floor(x / TILE)) * TILE;
    if (Math.abs(gy - y) < r + 8) {
      // Dust rolling out along the ground
      const dust = DUST_COLOR[this.theme.fill] ?? "#9a8a78";
      for (let n = big ? 16 : 6; n > 0; n--) {
        const dir = n % 2 ? 1 : -1;
        this.add({ k: DUST, x: x + dir * this.rand() * 6, y: gy - 2 - this.rand() * 4, vx: dir * (40 + this.rand() * (big ? 160 : 90)), vy: -8 - this.rand() * 22, life: 0.8 + this.rand() * 0.7, c: dust, s: 2 + this.rand() * (big ? 4 : 2) });
      }
      if (this.rand() < 0.75) this.fires.push({ x, y: gy, t: 1.5 + this.rand() * 3 });
    }
    this.shake = Math.min(pid === -2 ? 6 : pid === -4 ? 12 : 3, this.shake + r * (pid === -2 ? 0.2 : 0.05));
  }

  /** Props caught in a blast break into pieces (and the explosive ones go off too). */
  private smashProps(x: number, y: number, r: number) {
    const props = this.level.props;
    for (let i = props.length - 1; i >= 0; i--) {
      const p = props[i];
      if (p.kind !== "sprite" || p.back) continue;
      const a = asset(p.id);
      if (Math.abs(p.x - x) > r + a.w / 2 || y < p.y - a.h - r || y > p.y + r) continue;
      props.splice(i, 1);
      for (let k = Math.min(14, 4 + ((a.w * a.h) >> 5)); k > 0; k--)
        this.add({ k: DEBRIS, x: p.x + (this.rand() - 0.5) * a.w, y: p.y - this.rand() * a.h, vx: (this.rand() - 0.5) * 160, vy: -80 - this.rand() * 160, life: 1.2 + this.rand(), c: a.debris, s: this.rand() < 0.3 ? 2 : 1 });
      if (a.boom) {
        const boom = a.boom;
        this.at(0.06 + this.rand() * 0.12, () => this.explode(p.x, p.y - a.h / 2, boom, -3, true));
      }
    }
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
      m.speed = Math.min(m.speed + 520 * dt, m.max ?? 720);
      const dx = m.tx - m.x;
      const dy = m.ty - m.y;
      const d = Math.hypot(dx, dy) || 1;
      const step = m.speed * dt;
      // A trail of smoke and fire behind it (thick for the big ones)
      const cls = m.owner ? (m.cls ?? 1) : 3;
      const ux = m.boost ? 0 : dx / d;
      const uy = m.boost ? -1 : dy / d;
      if (cls >= 2 || this.rand() < 0.5 + cls * 0.25)
        this.add({ k: cls >= 2 ? BIGSMOKE : SMOKE, x: m.x - ux * 10, y: m.y - uy * 10, vx: (this.rand() - 0.5) * 10, vy: (this.rand() - 0.5) * 10, life: 1.2 + cls * 0.6 + this.rand(), c: "", s: 1 + cls + this.rand() * 2 });
      this.add({ k: FLAME, x: m.x - ux * 8, y: m.y - uy * 8, vx: 0, vy: 0, life: 0.2, c: "", s: cls >= 2 ? 2 : 1 });
      if (m.boost) {
        // Climbing off the launcher, then over and down
        m.t = (m.t ?? 0) + dt;
        m.y -= step;
        m.x += (m.owner === "ally" ? 1 : -1) * step * 0.12;
        if (m.y < this.lastView.y - 30 || m.t > 2.6) m.boost = false;
        keep.push(m);
        continue;
      }
      if (m.up) {
        m.y -= step;
        if (m.y > this.lastView.y - 300) keep.push(m);
        continue;
      }
      if (m.split && m.y >= m.ty - 4) {
        this.splitMirv(m);
        continue;
      }
      if (d <= step) {
        if (m.owner) this.impact(m, m.tx, m.ty);
        else this.detonate(m);
        continue;
      }
      if (this.solid(m.x + (dx / d) * step, m.y + (dy / d) * step)) {
        // (hit something on the way in: it goes off there)
        if (m.owner) this.impact(m, m.x + (dx / d) * step, m.y + (dy / d) * step);
        else this.detonate(m, m.x + (dx / d) * step, m.y + (dy / d) * step);
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
      T.para = (40 + r() * 40) / (s.paras * k * 1.8);
      // Reinforcements: a transport plane drops a stick of paratroopers ahead of the squad
      this.jets.push({ x: view.x + view.w + 60, y: view.y + 14 + r() * 16, vx: -320, bombs: 3 + Math.floor(r() * 4), dropT: 0, kind: "transport", ally: false, fireCd: 0, hp: 6, down: false, vy: 0, dead: false });
    }
    if (s.jets > 0 && (T.jet -= dt) <= 0) {
      // (often, and low enough to be seen: across the middle of the view)
      T.jet = (3 + r() * 4) / Math.max(0.6, s.jets) / Math.sqrt(k);
      const kind: PlaneType = (["bomber", "fighter", "attacker"] as const)[Math.floor(r() * 3)];
      const ally = r() < 0.5;
      const speed = kind === "bomber" ? 170 : kind === "fighter" ? 300 : 230;
      this.jets.push({
        x: ally ? view.x - 60 : view.x + view.w + 60,
        y: view.y + 24 + r() * view.h * 0.4,
        vx: (ally ? 1 : -1) * (speed + r() * 40),
        bombs: calm ? 0 : kind === "bomber" ? 8 + Math.floor(r() * 5) : kind === "attacker" ? 4 : 2,
        dropT: 0.3,
        kind,
        ally,
        fireCd: 0.2,
        hp: kind === "bomber" ? 4 : 2,
        down: false,
        vy: 0,
        dead: false,
        // (some bombers carry one huge bomb instead of a stick of small ones)
        big: kind === "bomber" && r() < 0.35,
      });
    }
    // Players to drop back in: a transport comes over from behind
    if (this.dropQueue.length && !this.jets.some((j) => j.kind === "transport" && j.ally && !j.down)) {
      this.jets.push({ x: view.x - 50, y: view.y + 16, vx: 380, bombs: 0, dropT: 0, kind: "transport", ally: true, fireCd: 0, hp: 6, down: false, vy: 0, dead: false });
    }
    for (const j of this.jets) {
      j.x += j.vx * dt;
      if (j.kind === "transport" && !j.down) {
        j.dropT -= dt;
        if (j.ally) {
          // Over the squad: out they go, one after another
          const id = this.dropQueue[0];
          if (id !== undefined && j.dropT <= 0 && j.x >= Math.min(view.x + view.w * 0.6, Math.max(view.x + 20, this.advance + OFFSETS[id] - 30))) {
            this.dropQueue.shift();
            this.release(this.players[id], j.x, j.y + 8);
            j.dropT = 0.12;
          }
        } else if (j.bombs > 0 && j.dropT <= 0 && j.x < view.x + view.w * 0.92 && j.x > this.advance + 40) {
          // The enemy's: a stick of paratroopers
          j.bombs--;
          j.dropT = 0.16;
          const pool: EnemyKind[] = ["mook", "mook", "heavy", "rocket", "minigunner", "rusher", "laser", "dynamiter"];
          const e = this.vary(this.makeEnemy(pool[Math.floor(this.rand() * pool.length)], j.x, j.y + 8));
          e.chute = true;
          e.ground = false;
          if (this.enemies.length < this.cap()) this.enemies.push(e);
        }
        continue;
      }
      if (j.down) {
        // Shot down: trailing fire and smoke, into the ground
        j.vy = Math.min(j.vy + 260 * dt, 260);
        j.y += j.vy * dt;
        this.puff(j.x, j.y, 3);
        if (this.rand() < 0.6) this.flame(j.x, j.y);
        if (this.solid(j.x, j.y + 4) || j.y > LEVEL_H) {
          j.dead = true;
          this.explode(j.x, Math.min(j.y, LEVEL_H - 8), j.kind === "bomber" ? 40 : 28, j.ally ? -5 : -3, !j.ally);
        }
        continue;
      }
      j.dropT -= dt;
      j.fireCd -= dt;
      // Over its target area: the enemy's over the squad and the column, ours over the enemy ahead
      const over = j.ally ? j.x > this.advance + 70 && j.x < view.x + view.w * 0.96 : j.x > this.advance - 120 && j.x < this.advance + 170;
      if (!over) continue;
      const pid = j.ally ? -5 : -3;
      if (j.kind === "bomber" && j.big && j.bombs > 0 && j.x * sign(j.vx) > (j.ally ? this.advance + 140 : this.advance + 30) * sign(j.vx)) {
        j.bombs = 0;
        this.shots.push({ kind: BIGBOMB, x: j.x, y: j.y + 6, vx: j.vx * 0.2, vy: 30, life: 8, dmg: 0, pid, pierce: 0 });
        this.popups.push({ x: j.x, y: j.y + 16, text: "BIG BOMB!", c: j.ally ? "#5ab0ff" : "#ff3b3b", t: 0 });
      } else if (j.kind === "bomber" && !j.big && j.bombs > 0 && j.dropT <= 0) {
        j.bombs--;
        j.dropT = 0.1 + r() * 0.1;
        this.shots.push({ kind: SHELL, x: j.x, y: j.y + 4, vx: j.vx * 0.25, vy: 40, life: 4, dmg: 0, pid, pierce: 0 });
      } else if (j.kind === "attacker" && j.bombs > 0 && j.dropT <= 0) {
        j.bombs--;
        j.dropT = 0.16;
        this.shots.push({ kind: NAPALM, x: j.x, y: j.y + 4, vx: j.vx * 0.35, vy: 30, life: 4, dmg: 0, pid, pierce: 0 });
      } else if (j.kind === "fighter" && j.fireCd <= 0) {
        // Strafing runs, and a pair of missiles at whatever's worth it
        const dir = sign(j.vx);
        const a = Math.atan2(0.55, dir) + (r() - 0.5) * 0.1;
        this.shots.push({ kind: j.ally ? BULLET : ENEMY_BULLET, x: j.x, y: j.y + 2, vx: Math.cos(a) * 420, vy: Math.sin(a) * 420, life: 0.9, dmg: 1, pid: j.ally ? -5 : -1, pierce: 0 });
        j.fireCd = 0.05;
        if (j.bombs > 0 && r() < 0.06) {
          j.bombs--;
          const tgt: Foe | Enemy | null = j.ally ? (this.enemies.find((e) => e.alive && (ARMOURED.has(e.kind) || FLYING.has(e.kind)) && e.x > j.x && e.x < view.x + view.w) ?? null) : (this.allies.find((q) => !q.dead) ?? this.players.find((q) => !q.dead) ?? null);
          if (tgt) this.missiles.push({ x: j.x, y: j.y + 4, tx: tgt.x, ty: "boss" in tgt && FLYING.has(tgt.kind) ? tgt.y : surfaceY(this.level, tgt.x, 0), speed: 300, max: 640, up: false, owner: j.ally ? "ally" : "enemy", cls: 0 });
        }
      }
    }
    this.jets = this.jets.filter((j) => !j.dead && j.x > view.x - 120 && j.x < view.x + view.w + 160);
    if (!calm && s.helis > 0 && (T.heli -= dt) <= 0) {
      T.heli = (40 + r() * 40) / (s.helis * k * 2.2);
      if (this.enemies.filter((e) => FLYING.has(e.kind) && !e.boss && e.kind !== "drone" && e.kind !== "jetpacker" && e.alive).length < clamp(Math.round(view.w / 200), 3, 7)) {
        const roll = r();
        const kind: EnemyKind = roll < 0.3 ? "heli" : roll < 0.55 ? "attackheli" : roll < 0.75 ? "scoutheli" : "transport";
        const h = this.makeEnemy(kind, view.x + view.w + 40, view.y + 40);
        h.hover = 30 + r() * 44;
        h.home = Math.floor(r() * 100);
        this.enemies.push(h);
      }
    }
    // Armour rolls in from ahead: carriers full of troops, light tanks
    if (!calm && (s.helis > 0 || s.jets > 0) && (T.armor -= dt) <= 0) {
      T.armor = (14 + r() * 14) / k;
      if (this.enemies.filter((e) => ARMOR_KINDS.has(e.kind) && e.alive).length < clamp(Math.round(view.w / 260), 3, 9) && this.enemies.length < this.cap()) {
        const x = view.x + view.w + 50;
        const gy = surfaceY(this.level, x, view.y - 60);
        const wet = kindAt(this.level, Math.floor(x / TILE), Math.floor(gy / TILE) - 1) === WATER;
        const kinds: EnemyKind[] = wet ? ["boat", "battleship", "submarine", "hovercraft"] : ["apc", "apc", "armoredcar", "armoredcar", "artytruck", "hovercraft", "lighttank", "medtank", "medtank", "heavytank", "flametank", "rockettank", "aatank", "launcher"];
        const kind = kinds[Math.floor(r() * kinds.length)];
        let y = gy;
        if (wet) {
          // At sea: ships, on the surface
          let row = Math.floor(gy / TILE) - 1;
          while (row > 0 && kindAt(this.level, Math.floor(x / TILE), row - 1) === WATER) row--;
          y = row * TILE;
        }
        this.enemies.push(this.makeEnemy(kind, x, y));
      }
    }
    // Allied missile strikes on the enemy ahead: salvos of small ones, tactical, cruise, and now and then an enormous one
    if (!calm && (T.strike -= dt) <= 0) {
      T.strike = (14 + r() * 14) / k;
      const ahead = this.enemies.filter((e) => e.alive && !FLYING.has(e.kind) && e.x > this.advance + 80 && e.x < view.x + view.w - 10);
      const tgt = ahead.find((e) => ARMOURED.has(e.kind) || e.boss) ?? ahead[Math.floor(r() * ahead.length)];
      if (tgt) {
        const roll = r();
        let cls = roll < 0.45 ? 0 : roll < 0.75 ? 1 : roll < 0.93 ? 2 : 3;
        if (cls === 3 && this.players.some((p) => !p.dead && Math.abs(p.x - tgt.x) < 230)) cls = 2;
        const n = cls === 0 ? 5 : cls === 1 ? 2 : 1;
        for (let j = 0; j < n; j++)
          this.at(j * 0.22, () => {
            const tx = tgt.x + (n > 1 ? (r() - 0.5) * 60 : 0);
            this.missiles.push({ x: view.x - 30 - r() * 60, y: view.y - 50 - r() * 30, tx, ty: surfaceY(this.level, tx, 0), speed: 200, max: [700, 640, 560, 480][cls], up: false, owner: "ally", cls });
          });
        if (cls === 3) this.popups.push({ x: tgt.x, y: tgt.y - 40, text: "BIG ONE INBOUND", c: "#5ab0ff", t: 0 });
      }
    }
    // Enemy missile strikes on the squad and the allied column, from small rockets to enormous warheads
    if (!calm && s.artillery > 0 && (T.estrike -= dt) <= 0) {
      T.estrike = (9 + r() * 9) / k;
      const foes: Foe[] = [...this.players.filter((p) => !p.dead && !p.chute), ...this.allies.filter((a) => !a.dead)];
      const tgt = foes[Math.floor(r() * foes.length)];
      if (tgt) {
        const roll = r();
        const cls = roll < 0.5 ? 0 : roll < 0.8 ? 1 : roll < 0.95 ? 2 : 3;
        const n = cls === 0 ? 5 : cls === 1 ? 2 : 1;
        for (let j = 0; j < n; j++)
          this.at(j * 0.22, () => {
            const tx = tgt.x + (r() - 0.5) * 70;
            this.missiles.push({ x: view.x + view.w + 30 + r() * 60, y: view.y - 50 - r() * 30, tx, ty: surfaceY(this.level, tx, 0), speed: 200, max: [700, 640, 560, 480][cls], up: false, owner: "enemy", cls });
          });
        if (cls >= 2) this.popups.push({ x: tgt.x, y: tgt.y - 30, text: cls === 3 ? "HUGE ONE INBOUND" : "INCOMING", c: "#ff3b3b", t: 0 });
      }
    }
    // Jetpack troopers swoop in
    if (!calm && s.paras > 0 && (T.jetpack -= dt) <= 0) {
      T.jetpack = (24 + r() * 24) / k;
      for (let n = 2 + Math.floor(r() * 3); n > 0 && this.enemies.length < this.cap(); n--) {
        const j = this.makeEnemy("jetpacker", view.x + view.w + 20 + n * 16, view.y + 20 + r() * 30);
        j.home = n * 3 + Math.floor(r() * 10);
        j.ground = false;
        this.enemies.push(j);
      }
    }
    // Supply drops for the squad: a new weapon, or a jetpack
    if (!calm && (T.drop -= dt) <= 0) {
      T.drop = 5 + r() * 6;
      this.drops.push({ x: Math.max(this.advance + 30, view.x + view.w * (0.3 + r() * 0.4)), y: view.y - 10, vy: 0, weapon: r() < 0.2 ? -1 : Math.floor(r() * WEAPONS.length), ground: false, t: 0 });
    }
    // Drone swarms sweep in from ahead
    if (!calm && s.drones > 0 && (T.drone -= dt) <= 0) {
      T.drone = (40 + r() * 40) / (s.drones * k);
      if (this.enemies.filter((e) => e.kind === "drone" && e.alive).length < 6)
        for (let n = 2 + Math.floor(r() * 3); n > 0 && this.enemies.length < this.cap(); n--) {
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
      else if (p.kind === "sprite" && !p.back && r() < 0.5) {
        const a = asset(p.id);
        if (a.flames) for (const [fx, fy] of a.flames) this.flame(p.x + (p.flip ? a.w / 2 - fx : fx - a.w / 2), p.y - a.h + fy);
        if (a.smoke && r() < 0.3) for (const [fx, fy] of a.smoke) this.puff(p.x + (p.flip ? a.w / 2 - fx : fx - a.w / 2), p.y - a.h + fy, 2 + r() * 2);
      }
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
    for (const b of this.beams) b.t -= dt;
    this.beams = this.beams.filter((b) => b.t > 0);
    for (const z of this.zaps) z.t -= dt;
    this.zaps = this.zaps.filter((z) => z.t > 0);
    for (const d of this.drops) {
      d.t += dt;
      if (d.ground) continue;
      d.vy = Math.min(d.vy + 200 * dt, 45);
      d.y += d.vy * dt;
      if (this.solid(d.x, d.y + 1)) {
        d.ground = true;
        d.y = Math.floor((d.y + 1) / TILE) * TILE;
      }
    }
    this.drops = this.drops.filter((d) => d.t < 22 && d.x > view.x - 40);
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
      } else if (p.k === CINDER) {
        p.vy += 380 * dt;
        const nx = p.x + p.vx * dt;
        const ny = p.y + p.vy * dt;
        if (this.solid(nx, ny)) {
          p.vy *= -0.35;
          p.vx *= 0.55;
        } else {
          p.x = nx;
          p.y = ny;
        }
        if (this.rand() < 0.45) this.add({ k: FLAME, x: p.x, y: p.y, vx: 0, vy: -8, life: 0.2 + this.rand() * 0.15, c: "", s: 1 });
      } else {
        if (p.k === SPARK) p.vy += 160 * dt;
        else if (p.k === BLAZE) {
          const k = Math.exp(-dt * 3.2);
          p.vx *= k;
          p.vy = p.vy * k - 26 * dt;
        } else if (p.k === DUST) {
          p.vx *= Math.exp(-dt * 2.6);
          p.vy += 6 * dt;
        }
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

  /** A particle, if the budget allows (`force`: always; for the few that make up an explosion). */
  private add(p: Omit<Particle, "max">, force = false) {
    if (force || this.parts.length < this.maxParticles) this.parts.push({ ...p, max: p.life });
  }

  /**
   * A cluster of fireballs (the renderer draws them as one lumpy, outlined blob): big ones
   * in the middle, smaller ones round the edge, all swelling, rising and cooling.
   */
  private fireball(x: number, y: number, r: number) {
    const balls = Math.min(22, 4 + Math.round(r / 2.5));
    const slow = r > 40 ? 1.7 : 1;
    const reach = r * 0.55 + 1;
    for (let n = 0; n < balls; n++) {
      const a = this.rand() * Math.PI * 2;
      const d = Math.sqrt(this.rand()) * r * 0.55;
      const ux = Math.cos(a);
      const uy = Math.sin(a);
      const core = 1 - d / reach;
      this.add({
        k: BLAZE,
        x: x + ux * d,
        y: y + uy * d * 0.8,
        vx: ux * (10 + d * 1.8),
        vy: uy * d * 1.2 - 18 - this.rand() * 28,
        life: (0.4 + this.rand() * 0.45 + core * 0.25) * slow,
        c: "",
        s: Math.max(2, r * (0.18 + this.rand() * 0.2) * (0.65 + core * 0.5)),
      }, true);
    }
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
      this.add({ k: DEBRIS, x: c * TILE + this.rand() * TILE, y: r * TILE + this.rand() * TILE, vx: (this.rand() - 0.5) * 140, vy: -60 - this.rand() * 140, life: 1.2 + this.rand(), c: String(kind), s: this.rand() < 0.15 ? 3 : this.rand() < 0.4 ? 2 : 1 });
  }

  private dirtyAround(c: number, r: number) {
    this.dirty.push(c - 1, r, c + 1, r, c, r - 1, c, r + 1);
  }
}

/** The theme for `?level=` (or the 404's cave), else a random one. */
export function startTheme(key: string | null, rand: () => number) {
  return (key && THEMES[key]) || nextTheme(rand);
}
