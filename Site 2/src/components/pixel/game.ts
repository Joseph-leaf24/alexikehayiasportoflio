/**
 * The battle simulation behind the site: eight AI players fight through one random level after another.
 * Plain logic with no drawing; PixelScene renders it, which keeps it unit-testable.
 */
import {
  AIR,
  ASH,
  kindAt,
  WATER,
  BASE_ROW,
  CLAY,
  CONCRETE,
  DIRT,
  GRAVEL,
  LADDER,
  LOAM,
  LOOSE,
  MUD,
  PEAT,
  PERMAFROST,
  RAIL,
  REDROCK,
  RING,
  SAND,
  SANDSTONE,
  SNOW,
  STONE,
  TILE_HP,
  TUFF,
  blastCells,
  cell,
  createLevel,
  crustOf,
  damageCell,
  extend,
  groundOf,
  lavaAt,
  LEVEL_H,
  ROWS,
  lineOfSight,
  SOLID,
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
/** Each player's offset from the squad's advance point: the flamethrower leads, the sniper hangs back. */
const OFFSETS = [36, 48, 10, -16, -42, 62, -4, 22];
/** The squad's weapons. Player n starts with weapon n; supply crates hand out the rest for a while. */
export const WEAPONS = [
  "Rifle",
  "Shotgun",
  "Plasma",
  "Rockets",
  "Sniper",
  "Flamethrower",
  "Grenades",
  "Minigun",
  "Laser",
  "Tesla",
  "Pistols",
  "Crossbow",
  "Saw blades",
  "Mortar",
  "Pulse laser",
  "Death ray",
  "Prism laser",
] as const;
/** Range per weapon. */
const W_RANGE = [170, 115, 160, 190, 300, 72, 210, 165, 230, 140, 150, 200, 150, 260, 220, 280, 210];
/** The prism laser's split beams */
const RAINBOW = ["#ff5a6a", "#ffd23f", "#5aff8a", "#5ab0ff", "#c87aff"];

/** The battle's waves; `k` scales how many enemies there are. */
export const WAVES = [
  { name: "Lull", k: 0.5 },
  { name: "Skirmish", k: 1 },
  { name: "Assault", k: 1.8 },
  { name: "Onslaught", k: 3 },
] as const;
/** Weapons that lob over cover, needing no clear line. */
const LOBBED = new Set([6, 13]);
/** How much more of everything there is than the themes ask for. */
const CHAOS = 3;
/** Player height from head to feet, and half width. */
const PH = 15;
const PW = 3;

interface Body {
  x: number;
  /** Feet */
  y: number;
  vx: number;
  vy: number;
  ground: boolean;
  facing: 1 | -1;
}

interface Player extends Body {
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
  /** Seconds without getting anywhere; too long and it respawns */
  stall: number;
  lastX: number;
  anim: number;
  spin: number;
  /** Seconds left of the sprite's recoil kick */
  kick: number;
  /** Seconds left backing off from an overhang before jumping for its edge */
  back: number;
  /**
   * Best x and y reached while behind its place in the squad, and seconds since either improved,
   * so circling in a blasted hole is caught however busy it looks
   */
  bestX: number;
  bestY: number;
  progT: number;
  /** Death ray: seconds left on, its angle, and time towards its next bite out of a block */
  ray: number;
  rayA: number;
  rayCut: number;
  /** Index into WEAPONS of the weapon in hand, and seconds left on a picked-up one */
  weapon: number;
  weaponT: number;
  /** Seconds of jetpack left */
  jet: number;
  /** Down, waiting for a transport to drop them back in */
  queued: boolean;
  /** Riding in the gunship or a metro train, out of the fight except for the gunship's door guns */
  aboard: boolean;
  /** Seconds left on fire after a warhead threw them */
  burn: number;
  score: number;
  kills: number;
}

/** Anything an enemy can shoot at: a player or an allied vehicle. */
type Foe = { x: number; y: number; dead: boolean; vx?: number };

/** Allied vehicles fighting alongside the squad, its gunship included. */
type AllyKind = "mbt" | "ifv" | "mlrs" | "aa" | "scoutcar" | "aheli" | "lifter" | "gunship";
interface Ally extends Body {
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
const ALLY_HP: Record<AllyKind, number> = {
  mbt: 40,
  ifv: 26,
  mlrs: 18,
  aa: 20,
  scoutcar: 18,
  aheli: 16,
  lifter: 24,
  gunship: 220,
};
const ALLY_SIZE: Record<AllyKind, [number, number]> = {
  mbt: [20, 14],
  ifv: [19, 13],
  mlrs: [19, 14],
  aa: [18, 13],
  scoutcar: [16, 12],
  aheli: [15, 9],
  lifter: [22, 10],
  gunship: [30, 14],
};
export const ALLY_AIR = new Set<AllyKind>(["aheli", "lifter", "gunship"]);
/** Missile blast radius by size class: small rocket, tactical, cruise, huge warhead. */
const MISSILE_R = [16, 30, 52, 96];
/** Enemy armour, driven by updateArmour. */
export const ARMOR_KINDS = new Set([
  "apc",
  "lighttank",
  "medtank",
  "heavytank",
  "flametank",
  "rockettank",
  "aatank",
  "launcher",
  "armoredcar",
  "artytruck",
  "hovercraft",
]);
/** Kinds of aircraft flying over the battle, on either side. */
export type PlaneType = "bomber" | "fighter" | "attacker" | "transport" | "heavy" | "stealth" | "dive" | "rocketjet";
export interface Jet {
  x: number;
  y: number;
  vx: number;
  bombs: number;
  dropT: number;
  kind: PlaneType;
  ally: boolean;
  fireCd: number;
  hp: number;
  down: boolean;
  vy: number;
  dead: boolean;
  big?: boolean;
  /** Dive bombers: cruising height, and dive phase: 0 cruising, 1 diving, 2 pulling out, 3 done */
  alt?: number;
  phase?: number;
}

/** Per aircraft kind: speed, bombs carried plus up to `more` extra, hit points, and whether it flies high. */
const PLANES: Record<PlaneType, { speed: number; bombs: number; more: number; hp: number; high?: boolean }> = {
  bomber: { speed: 170, bombs: 8, more: 4, hp: 4 },
  fighter: { speed: 300, bombs: 2, more: 0, hp: 2 },
  attacker: { speed: 230, bombs: 4, more: 0, hp: 2 },
  transport: { speed: 320, bombs: 3, more: 3, hp: 6 },
  heavy: { speed: 120, bombs: 18, more: 8, hp: 7, high: true },
  stealth: { speed: 250, bombs: 3, more: 0, hp: 3, high: true },
  dive: { speed: 210, bombs: 1, more: 0, hp: 2 },
  rocketjet: { speed: 260, bombs: 3, more: 1, hp: 2 },
};
/** Weighted pool for bombing runs; transports have their own schedule. */
const PLANE_MIX: PlaneType[] = [
  "bomber",
  "bomber",
  "fighter",
  "fighter",
  "attacker",
  "attacker",
  "heavy",
  "stealth",
  "dive",
  "dive",
  "rocketjet",
  "rocketjet",
];

/** An allied submarine that fires missiles at enemy aircraft. */
interface Sub {
  x: number;
  /** The middle of its hull */
  y: number;
  facing: 1 | -1;
  /** Target depth in px below the surface, 0 when surfaced, and seconds until it picks another */
  dive: number;
  diveT: number;
  samCd: number;
  t: number;
}

/** Something in the air a submarine's missile can go after. */
interface AirTarget {
  /** The aircraft itself, so no two missiles chase the same one */
  ref: object;
  x(): number;
  y(): number;
  vx(): number;
  up(): boolean;
  /** Called when a missile hits it */
  hit(): void;
}

/** A submarine-launched missile: rises through the water, breaks the surface, climbs, then homes in. */
interface Sam {
  x: number;
  y: number;
  vx: number;
  vy: number;
  /** The side that fired it; it hunts the other side's aircraft */
  side: "ally" | "enemy";
  target: AirTarget | null;
  t: number;
  /** Still under water, and the time `t` at which it surfaced */
  wet: boolean;
  out: number;
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
const BOSS_KINDS: BossKind[] = ["tank", "mech", "gunship", "worm", "airship", "cannon", "spider"];
export const BOSS_NAMES: Record<BossKind, string> = {
  tank: "Siege tank",
  mech: "Walker",
  gunship: "Gunship",
  worm: "Burrower",
  airship: "Sky fortress",
  cannon: "Doom cannon",
  spider: "Spider tank",
};
const isBoss = (k: EnemyKind): k is BossKind => (BOSS_KINDS as string[]).includes(k);
/** Things in the air: no gravity, shot at without needing a clear line, crash when downed. */
const FLYING = new Set<EnemyKind>([
  "heli",
  "gunship",
  "drone",
  "jetpacker",
  "airship",
  "attackheli",
  "transport",
  "scoutheli",
]);
/** Ships sail back and forth across whatever water they're on. */
const SHIPS = new Set<EnemyKind>(["boat", "battleship", "submarine"]);
/** Kinds a blast damages rather than flings. */
const ARMOURED = new Set<EnemyKind>([
  "heli",
  "jeep",
  "truck",
  "boat",
  "battleship",
  "mg",
  "exosuit",
  "apc",
  "lighttank",
  "medtank",
  "heavytank",
  "flametank",
  "rockettank",
  "aatank",
  "launcher",
  "attackheli",
  "transport",
  "scoutheli",
  "flakgun",
  "howitzer",
  "samsite",
  "rocketpod",
  "submarine",
  "train",
  "armoredcar",
  "artytruck",
  "hovercraft",
]);

interface Segment {
  x: number;
  y: number;
  /** Above ground, so visible and hittable */
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
  /** Floating down under a parachute */
  chute: boolean;
  /** Snipers: seconds spent lining up the shot, laser sight on */
  aim: number;
  /** Gun angle in radians, for turrets, tanks and walkers */
  aimA: number;
  /** The worm's body */
  segs: Segment[];
  /** Contact-damage cooldown */
  bite: number;
  /** Seconds left on fire after a blast flung it */
  burn: number;
  /** Size multiplier from 0.75 to 2.2, and whether it carries explosives that go off when it dies */
  scale: number;
  sapper: boolean;
  /** Falling back during a lull: it runs off ahead, out of the fight */
  retreat?: boolean;
  /** Seconds left falling back to cover, and until its next grenade */
  cover: number;
  nade: number;
  /** Submarines: seconds until the next missile at our aircraft */
  samCd?: number;
}

type BarrelKind = "barrel" | "fuel" | "tanker" | "ammo" | "egg" | "propane" | "canister" | "cylinder" | "mine" | "tnt";

interface Barrel extends Body {
  type: BarrelKind;
  hp: number;
  /** Seconds until it goes off; -1 when not lit */
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
/** A napalm canister that starts fires, and a flak shell that bursts at its range */
export const NAPALM = 17;
export const FLAK = 18;
/** A huge bomb: a big blast, a mushroom cloud, a crater */
export const BIGBOMB = 19;
/** A pulse laser's bolt, in its firer's colour */
export const PULSE = 20;

const EXPLOSIVE = new Set([ROCKET, GRENADE, SHELL, METEOR, LOB, CANNON, MORTAR, FIREBALL, BOLT, NAPALM, FLAK, BIGBOMB]);
/** Gravity per shot kind in px/s²; negative values drift up */
const SHOT_GRAV: Record<number, number> = {
  [GRENADE]: GRAV,
  [LOB]: 500,
  [CANNON]: 300,
  [MORTAR]: 400,
  [FIREBALL]: 350,
  [FLAMESHOT]: -70,
  [BOLT]: 120,
  [NAPALM]: 420,
  [BIGBOMB]: 240,
};
const BLAST_R: Record<number, number> = {
  [ROCKET]: 18,
  [GRENADE]: 22,
  [SHELL]: 18,
  [METEOR]: 24,
  [LOB]: 20,
  [CANNON]: 24,
  [MORTAR]: 18,
  [FIREBALL]: 16,
  [BOLT]: 13,
  [NAPALM]: 14,
  [FLAK]: 11,
  [BIGBOMB]: 72,
};

export interface Shot {
  kind: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  dmg: number;
  /** Who fired: player 0–7, -1 enemies, -2 the visitor, -3 the level's artillery, -4 missiles, -5 allies */
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
/** One ball of an explosion's fireball, cooling from white hot to smoke as it swells and rises */
export const BLAZE = 14;
/** A burning chunk thrown out of a blast: arcs, bounces, trails flame */
export const CINDER = 15;
/** Dust kicked along the ground by a blast; colour in `c` */
export const DUST = 16;
/** Air rising through water, popping at the surface */
export const BUBBLE = 17;
/** A sparkle drifting down after a blast; colour in `c` */
export const GLINT = 18;

/** Explosion palettes; the renderer has each one's fireball, flash and smoke colours. */
export const PAL_FIRE = 0;
const PAL_NUKE = 1;
export const PAL_CHEM = 2;
const PAL_GAS = 3;
const PAL_PLASMA = 4;
const PAL_NAPALM = 5;
const PAL_MAGMA = 6;
/** Spark and glint colours per palette */
const PAL_SPARKS: string[][] = [
  ["#fff6c8", "#ffe27a", "#ffb43a", "#ff7a2a", "#ff4a5a", "#ffd0f0"],
  ["#ffffff", "#fff3b0", "#ffd23f", "#ff9ad0", "#9ad8ff", "#ffb43a"],
  ["#f0ffc0", "#b8ff6a", "#5aff4a", "#2ad86a", "#d8ff3a", "#5affd8"],
  ["#ffffff", "#c8f0ff", "#7ac8ff", "#4a7aff", "#a08aff", "#ffd27a"],
  ["#ffffff", "#aafcff", "#4ae0ff", "#7a9aff", "#c87aff", "#ff7af0"],
  ["#fff0b0", "#ffc23a", "#ff7a1a", "#ff4a1a", "#d82a1a", "#ffe27a"],
  ["#ffe8a0", "#ffa21f", "#ff5a1a", "#e8301a", "#ff8a5a", "#ffd23f"],
];

interface Particle {
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

interface Blast {
  x: number;
  y: number;
  r: number;
  t: number;
  /** Palette, one of PAL_* */
  pal: number;
}

/** A mine cart rolling between `lo` and `hi` (world px); `load` indexes CART_LOAD. */
interface Cart {
  x: number;
  y: number;
  vx: number;
  lo: number;
  hi: number;
  load: 0 | 1 | 2 | 3;
  anim: number;
}

/** Flight phases of the squad's gunship. */
type ShipMode = "inbound" | "board" | "fly" | "land" | "leave";

interface Popup {
  x: number;
  y: number;
  text: string;
  c: string;
  t: number;
}

/** A missile in flight. Without an `owner` it's a finale warhead, or with `up` one rising from a silo. */
interface Missile {
  x: number;
  y: number;
  tx: number;
  ty: number;
  speed: number;
  up: boolean;
  /** Blast radius */
  r?: number;
  /** Part of the finale; the level ends once all of these have landed */
  fin?: boolean;
  /** Splits into this many warheads over the target area */
  split?: number;
  /** Top speed */
  max?: number;
  /** Fired from the ground: climbs first, then turns over and dives on its target */
  boost?: boolean;
  /** Whose it is, unset for the finale; and its size class, an index into MISSILE_R */
  owner?: "enemy" | "ally";
  cls?: number;
  t?: number;
  /** Unit heading, as it curves in on its target; and its trail as (x, y) pairs, newest last */
  hx?: number;
  hy?: number;
  trail?: number[];
}

/** A train on a stretch of main line: `x` is its front (world px), `y` the rails. */
interface Freight {
  x: number;
  y: number;
  dir: 1 | -1;
  speed: number;
  /** Cars from the front: 0 locomotive, 1 boxcar, 2 tank car, 3 container flat, 4 hopper, 5 coach */
  cars: number[];
  livery: number;
  lo: number;
  hi: number;
  /** Seconds until the next train, while none is running */
  wait: number;
  anim: number;
  /** Station x (world px) where passenger trains stop; whether this one has; seconds left standing there */
  stop?: number;
  halted?: boolean;
  halt?: number;
  passenger?: boolean;
}
/** Length of a freight car in px */
export const CAR_LEN = 42;

/** Road vehicle kinds */
type CarKind = "sedan" | "taxi" | "van" | "bus" | "truck" | "firetruck" | "police" | "ambulance";
/** Vehicle length in px */
export const CAR_SIZE: Record<CarKind, number> = {
  sedan: 18,
  taxi: 18,
  van: 22,
  bus: 40,
  truck: 30,
  firetruck: 34,
  police: 18,
  ambulance: 24,
};
/**
 * A vehicle on a road. `x` is its middle, `wait` seconds stopped at a bus stop or fire, `last` the
 * last bus stop it called at, and `spray` where a fire engine is spraying, NaN when it isn't.
 */
interface Car {
  x: number;
  dir: 1 | -1;
  v: number;
  kind: CarKind;
  color: number;
  wait: number;
  riders: number;
  anim: number;
  siren: boolean;
  last: number;
  spray: number;
}
/** A road and its traffic; `fast` roads are expressways with no lights or stops. */
interface Road {
  lo: number;
  hi: number;
  y: number;
  lights: number[];
  stops: number[];
  fast: boolean;
  cars: Car[];
  spawnT: number;
}
/** Traffic light at x at time t: 0 green, 1 amber, 2 red. Each crossing runs its own cycle. */
export const lightState = (t: number, x: number) => {
  const u = (((t / 14 + x * 0.00071) % 1) + 1) % 1;
  return u < 0.46 ? 0 : u < 0.54 ? 1 : 2;
};
/** Weather in themes with a day cycle */
type Sky = "clear" | "cloud" | "rain" | "storm" | "fog";

/** A metro train: `x` its middle, `y` the tunnel floor, `wait` seconds left at a station, `doors` how open (0–1). */
interface Metro {
  x: number;
  y: number;
  lo: number;
  hi: number;
  stops: number[];
  dir: 1 | -1;
  v: number;
  wait: number;
  doors: number;
  cars: number;
  anim: number;
  /** Stopped at the end of the line; it reverses next */
  end: boolean;
  /** The squad boards at this stop if it's on the platform */
  board: boolean;
}
/** Length of a metro car in px */
const METRO_CAR = 34;

/** How a level ends; picked at random, never the same twice running. */
type Finale = "salvo" | "nuke" | "cruise" | "silo" | "mirv";
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
const CARD_TITLES = [
  "Level complete",
  "Mission accomplished",
  "Stronghold flattened",
  "Sector cleared",
  "Target destroyed",
  "Nothing left standing",
  "Area secured",
];

interface Banner {
  text: string;
  sub: string;
  /** A third, smaller line */
  info?: string;
  color: string;
  /** Seconds shown; negative while waiting to appear */
  t: number;
}

/** One finished level, for the "Level complete" card and the spectator log. */
interface LevelReport {
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
type BossState = "none" | "ahead" | "engaged" | "destroyed" | "bypassed";

/** What the camera sees (world px). */
export interface View {
  x: number;
  y: number;
  w: number;
  h: number;
}

type Phase = "play" | "boss" | "nuke" | "clear";

interface GameOptions {
  seed?: number;
  maxEnemies?: number;
  maxParticles?: number;
  /** Player colours, for score popups */
  colors?: string[];
  /** Options for the first level, for tests */
  level?: LevelOptions;
  /** False: the squad's gunship never comes */
  gunship?: boolean;
  /** False: the battle stays a skirmish throughout */
  waves?: boolean;
  /** How the first level ends; random if unset */
  finale?: Finale;
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
const SIZE: Record<EnemyKind | BarrelKind | "player", [number, number]> = {
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
const BARREL_BLAST: Record<BarrelKind, number> = {
  barrel: 20,
  fuel: 34,
  tanker: 60,
  ammo: 16,
  egg: 0,
  propane: 24,
  canister: 18,
  cylinder: 22,
  mine: 22,
  tnt: 32,
};
/** Blast dust colour by ground block; anything else uses a dun grey-brown. */
const DUST_COLOR: Partial<Record<number, string>> = {
  [CLAY]: "#c0906a",
  [LOAM]: "#7a5a3a",
  [GRAVEL]: "#9a948a",
  [PEAT]: "#5a4632",
  [PERMAFROST]: "#a8b0b8",
  [TUFF]: "#c8bca8",
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

/** Debris colour of a mine cart's load: coal, gold, copper ore, or empty */
const CART_LOAD = ["#2a2a30", "#ffd23f", "#c8906a", "#6a5a4a"];

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

/** A body's half width and height, scaled by its size. */
const dims = (e: Enemy): [number, number] => {
  const [w, h] = SIZE[e.kind];
  return e.scale === 1 ? [w, h] : [w * e.scale, h * e.scale];
};
/** Centre of a body's hit box: fliers are placed by their centre, everything else by its feet. */
const centreY = (e: Enemy) => (FLYING.has(e.kind) ? e.y : e.y - dims(e)[1] / 2);
/** Foot soldiers: the ones that come in sizes, and can carry explosives. */
const INFANTRY = new Set<EnemyKind>([
  "mook",
  "heavy",
  "rocket",
  "sniper",
  "shield",
  "flamer",
  "grenadier",
  "minigunner",
  "officer",
  "dynamiter",
  "rusher",
  "engineer",
  "laser",
]);

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
  /** Mushroom clouds, `s` their scale */
  mushrooms: { x: number; y: number; t: number; s?: number }[] = [];
  /** Roads with their traffic; smoke rising off the big blasts for a while after */
  roads: Road[] = [];
  plumes: { x: number; y: number; t: number; s: number }[] = [];
  /** Themes with a day cycle: time of day (0 noon, 0.5 midnight), the weather, and seconds until it changes */
  clock = 0;
  sky: Sky = "clear";
  private skyT = 40;
  /** Freight trains on the main lines, metro trains in the tunnels, and the one the squad's riding */
  freights: Freight[] = [];
  metros: Metro[] = [];
  ride: Metro | null = null;
  private rideHops = 0;
  /**
   * Station x the squad is waiting at for a train (-1 none), seconds left to wait, the train sent
   * for them, and the stations already considered
   */
  private platform = -1;
  private platformT = 0;
  private called: Metro | null = null;
  private platforms = new Set<number>();
  /** Index into WAVES, seconds until it changes, and seconds until the next reinforcements */
  wave = 1;
  private waveT = 24;
  private surge = 0;
  /** Running average of enemies on screen in skirmishes; the other waves scale from it */
  private usual = 10;
  /** Shockwave rings */
  waves: { x: number; y: number; t: number }[] = [];
  /** Short-lived fires left by explosions */
  fires: { x: number; y: number; t: number }[] = [];
  /** Bombers, fighters and attack jets on both sides */
  jets: Jet[] = [];
  /** Our submarines out in the sea, and the missiles both sides' submarines have sent up at aircraft */
  navy: Sub[] = [];
  sams: Sam[] = [];
  private navyCd = 0;
  /** Players waiting for the next transport to drop them back in */
  private dropQueue: number[] = [];
  /** The allied column fighting alongside the squad */
  allies: Ally[] = [];
  /** Mine carts rolling on the rails */
  carts: Cart[] = [];
  /** The squad's gunship while it's out (also in `allies`), its phase, and ids of the squad aboard it or a train */
  ship: Ally | null = null;
  shipMode: ShipMode = "inbound";
  crew: number[] = [];
  private shipT = 0;
  private shipCd = 22;
  /** Where it sets down for the squad to board (world px) */
  private shipPad = 0;
  private noShip = false;
  private noWaves = false;
  private forceFinale: Finale | null = null;
  /** It left empty, so it comes back soon */
  private shipEmpty = false;
  /** Cells to check for falling (column, row pairs), and cells of cut-off pieces that fall whatever they're made of */
  private unstable: number[] = [];
  private crumbling = new Set<number>();
  private seen = new Uint32Array(RING * ROWS);
  private stamp = 0;
  /** Manholes in the road that enemies climb out of */
  hatches: { x: number; y: number; n: number; cd: number; open: number }[] = [];
  private allyCd = 1;
  /** Muzzle flashes */
  flashes: { x: number; y: number; t: number; c: string }[] = [];
  /** Laser beams and tesla lightning, `t` seconds left to show */
  beams: { x0: number; y0: number; x1: number; y1: number; t: number; c: string; w?: number }[] = [];
  zaps: { pts: [number, number][]; t: number }[] = [];
  /** Supply crates parachuting in: `weapon` indexes WEAPONS, or -1 for a jetpack */
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
  private phaseT = 0;
  banner: Banner | null = null;
  /** Screen flash from big hits, 0 to 1, decaying */
  flash = 0;
  /** White-out from 0 to 1: up at a level's end while the card shows, back down as the next level fades in */
  white = 0;
  private whiteUp = false;
  /** The card shown during the end-of-level white-out */
  complete: LevelReport | null = null;
  /** The last six levels finished, most recent first, and how many have been finished */
  log: LevelReport[] = [];
  cleared = 0;
  /** This level's boss, if any, and how that fight is going */
  bossKind: BossKind | null = null;
  bossState: BossState = "none";
  /** Lightning strikes: x in world px, `t` seconds since the strike */
  bolts: { x: number; t: number; seed: number }[] = [];
  private killsAt = 0;
  /** Seconds since this level began */
  levelT = 0;
  /** Multiplies enemies, artillery and the rest; the page raises it for deeper sections */
  intensity = 1;
  /** Camera shake for the renderer, which decays it */
  shake = 0;
  private rand: () => number;
  private holdT = 0;
  private holdAt = Infinity;
  private timeline: { at: number; fn: () => void }[] = [];
  private timers = {
    artillery: 2,
    meteor: 3,
    heli: 8,
    para: 6,
    jet: 8,
    silo: 6,
    drone: 10,
    bolt: 4,
    drop: 3,
    armor: 12,
    jetpack: 14,
    strike: 10,
    estrike: 12,
  };
  private weather = 0;
  /** How this level ends, picked at the stronghold, and how the last one did */
  finale: Finale = "salvo";
  private lastFinale: Finale | null = null;
  private current: Theme;
  /** The biomes, dealt from a shuffled deck */
  private deck: (current?: string) => Theme;
  /** Finale warheads still to land, scheduled or in the air */
  private pending = 0;
  private lastView: View = { x: 0, y: 0, w: 400, h: 300 };
  private maxEnemies: number;
  private maxParticles: number;
  private colors: string[];

  constructor(theme: Theme, opts: GameOptions = {}) {
    this.rand = rng((opts.seed ?? 1) * 7919 + 3);
    this.maxEnemies = opts.maxEnemies ?? 70;
    this.maxParticles = opts.maxParticles ?? 2600;
    this.noShip = opts.gunship === false;
    this.noWaves = opts.waves === false;
    // Each seed starts at a different time of day.
    this.clock = ((opts.seed ?? 1) * 0.6180339) % 1;
    this.forceFinale = opts.finale ?? null;
    this.colors = opts.colors ?? [
      "#4aa3ff",
      "#ffcc33",
      "#c77dff",
      "#ff6a3d",
      "#5fd16a",
      "#3fd8e0",
      "#ff6fae",
      "#e8e2d0",
    ];
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
      back: 0,
      bestX: 0,
      bestY: 0,
      progT: 0,
      ray: 0,
      rayA: 0,
      rayCut: 0,
      burn: 0,
      weapon: id,
      weaponT: 0,
      jet: 0,
      queued: false,
      aboard: false,
      score: 0,
      kills: 0,
    }));
    this.meetBoss();
    this.say(`Level ${this.levelNo}`, theme.name, "#ffcc33", this.briefing());
  }

  /** The biome the squad is in; the land ahead may already be the next one */
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

  /** How far through the level the squad is, from 0 at the drop to 1 at the stronghold. */
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

  /** Record which boss, if any, this level holds. */
  private meetBoss() {
    this.bossKind = this.level.plan.boss?.kind ?? null;
    this.bossState = this.bossKind ? "ahead" : "none";
  }

  /** The level banner's third line: what lies ahead. */
  private briefing() {
    return this.bossKind ? `Boss ahead: ${BOSS_NAMES[this.bossKind]}` : "Objective: the enemy stronghold";
  }

  solid = (x: number, y: number) => solidAt(this.level, x, y);

  /** How many enemies at once: more when the view is wider, scaled by the current wave. */
  private cap() {
    return Math.round(this.maxEnemies * clamp(this.lastView.w / 560, 1, 2.6) * WAVES[this.wave].k);
  }

  /** How dark it is: the theme's own darkness, or from the time of day in themes with a day cycle. */
  get night() {
    if (!this.theme.cycle) return this.theme.night;
    return 0.7 * (0.5 - 0.5 * Math.cos(this.clock * Math.PI * 2));
  }

  /** What the squad is riding, for the camera to follow: a metro train, or the gunship (`air`). */
  get carrier(): { x: number; y: number; air: boolean } | null {
    if (!this.crew.length) return null;
    if (this.ride) return { x: this.ride.x, y: this.ride.y, air: false };
    if (this.ship && !this.ship.dead) return { x: this.ship.x, y: this.ship.y, air: true };
    return null;
  }

  /** One of the theme's foot soldiers, by its weights. */
  private troop(): EnemyKind {
    const entries = Object.entries(this.theme.troops) as [EnemyKind, number][];
    let x = this.rand() * entries.reduce((s, [, w]) => s + w, 0);
    for (const [k, w] of entries) if ((x -= w) < 0) return k;
    return "mook";
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

  /** Start the next level, in the given theme or a random one, with the squad parachuting in. */
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
    this.navy = [];
    this.sams = [];
    this.allies = [];
    this.hatches = [];
    this.carts = [];
    this.freights = [];
    this.roads = [];
    this.plumes = [];
    this.metros = [];
    this.ride = null;
    this.platform = -1;
    this.called = null;
    this.platforms.clear();
    this.ship = null;
    this.crew = [];
    this.shipCd = 22;
    this.unstable = [];
    this.crumbling.clear();
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
    // Start fully white so the new level fades in.
    this.white = 1;
    this.whiteUp = false;
    this.complete = null;
    this.killsAt = this.kills;
    this.meetBoss();
    for (const p of this.players) {
      Object.assign(p, {
        dead: false,
        hp: PLAYER_HP,
        inv: 2.5,
        chute: true,
        ground: false,
        vx: 0,
        vy: 0,
        stuckT: 0,
        stall: 0,
        spin: 0,
        target: null,
        burn: 0,
        weapon: p.id,
        weaponT: 0,
        jet: 0,
        queued: false,
        aboard: false,
      });
      p.x = this.advance + OFFSETS[p.id] + (this.rand() - 0.5) * 8;
      p.y = BASE_ROW * TILE - 90 - p.id * 9;
    }
    this.say(`Level ${this.levelNo}`, th.name, "#ffcc33", this.briefing(), 0.8);
  }

  /* -------------------------------------------------------------- */
  /* Tick                                                            */
  /* -------------------------------------------------------------- */

  /** Advance the game by `dt` seconds. With `frozen`, for reduced motion, only effects already in motion play out. */
  step(dt: number, view: View, frozen = false) {
    this.t += dt;
    this.levelT += dt;
    this.phaseT += dt;
    this.flash = Math.max(0, this.flash - dt * 1.1);
    // Fades up over 0.6 s and back down over 1.1 s.
    this.white = this.whiteUp ? Math.min(1, this.white + dt / 0.6) : Math.max(0, this.white - dt / 1.1);
    if (this.banner && (this.banner.t += dt) > 2.6) this.banner = null;
    for (const b of this.bolts) b.t += dt;
    if (this.bolts.length) this.bolts = this.bolts.filter((b) => b.t < 0.5);
    this.lastView = view;
    const L = this.level;
    L.intensity = this.intensity * CHAOS * (1 + 0.1 * Math.min(this.levelNo - 1, 6)) * WAVES[this.wave].k;

    // Past a biome border the land has run on into the next theme: count it as a new level.
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

    // Scheduled events; one that starts a new level ends this step.
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
      this.updateWave(dt, view);
      this.moveSquad(dt);
      for (const p of this.players) this.updatePlayer(p, dt, view);
      for (const e of this.enemies) {
        this.updateEnemy(e, dt, view);
        if (e.alive && FLYING.has(e.kind)) this.aboveLand(e);
      }
      this.manageAllies(dt, view);
      for (const a of this.allies) {
        this.updateAlly(a, dt, view);
        if (!a.dead && ALLY_AIR.has(a.kind)) {
          // Keep allied aircraft above the land surface too.
          const top = this.landTop(a.x, ALLY_SIZE[a.kind][0]) * TILE - ALLY_SIZE[a.kind][1] - 6;
          if (a.y > top) a.y = top;
        }
      }
      this.manageShip(dt, view);
      this.manageNavy(dt, view);
      for (const s of this.navy) this.updateSub(s, dt);
      this.updateCarts(dt);
      this.updateFreights(dt);
      this.updateMetros(dt);
      this.updateRoads(dt);
      this.updateSky(dt);
      this.updateHatches(dt, view);
      this.ambient(dt, view);
    }
    for (const b of this.barrels) this.updateBarrel(b, dt);
    this.settle();
    this.updateShots(dt, view);
    this.updateMissiles(dt);
    this.updateSams(dt, view);
    this.updateEffects(dt, view);

    // Forget what fell behind
    const left = view.x - 140;
    this.enemies = this.enemies.filter(
      (e) => (e.alive || e.deadT > 0) && (e.x > left || FLYING.has(e.kind)) && e.y < LEVEL_H + 80
    );
    this.barrels = this.barrels.filter((b) => b.alive && b.x > left);
    this.allies = this.allies.filter((a) => (!a.dead || a.deadT > 0) && a.x > left - 60);
    this.hatches = this.hatches.filter((h) => h.x > left);
    this.freights = this.freights.filter((f) => f.hi > left - 200);
    this.roads = this.roads.filter((rd) => rd.hi > left - 200);
    this.metros = this.metros.filter((m) => m.hi > left - 200 || m === this.ride);
    this.carts = this.carts.filter((k) => k.x > left);
  }

  /** The level's beats: the boss fight at its arena, the finale at the stronghold. */
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
    } else if (this.phase === "play" && plan.fortPlaced && this.advance > plan.targets[0] - 200) this.startNuke();
    else if (this.phase === "nuke" && this.phaseT > 22 && this.pending > 0) {
      // A warhead never landed: end the level anyway.
      this.pending = 0;
      this.blowUpEverything();
    }
  }

  /** Move the squad's advance point forward, holding for up to 4 s while enemies are close ahead. */
  private moveSquad(dt: number) {
    const alive = this.players.filter((p) => !p.dead);
    const front = alive.length ? Math.max(...alive.map((p) => p.x)) : this.advance;
    const threat = this.enemies.some(
      (e) => e.alive && !e.boss && !FLYING.has(e.kind) && e.x > front - 10 && e.x < front + 80
    );
    this.holdT = threat ? this.holdT + dt : 0;
    if (!threat || this.holdT > 4) this.advance += this.theme.speed * dt;
    // Never far ahead of the squad as a whole; stragglers catch up in updatePlayer.
    const mean = alive.length ? alive.reduce((s, p) => s + p.x, 0) / alive.length : this.advance;
    // Also hold at a platform while waiting for a train.
    this.advance = Math.min(
      this.advance,
      front + 50,
      mean + 40,
      this.holdAt,
      this.platform >= 0 ? this.platform + 30 : Infinity
    );
  }

  private spawn(s: Spawn) {
    if (s.kind === "minecart") {
      this.addCart(s.x, s.y);
      return;
    }
    if (s.kind === "metro") {
      this.addMetro(s);
      return;
    }
    if (s.kind === "freight") {
      this.addLine(s);
      return;
    }
    if (s.kind === "road" || s.kind === "motorway") {
      const [lo, hi, n = 0, ...rest] = s.extra ?? [s.x, s.x + 400, 0];
      this.roads.push({
        lo,
        hi,
        y: s.y,
        lights: rest.slice(0, n),
        stops: rest.slice(n),
        fast: s.kind === "motorway",
        cars: [],
        spawnT: this.rand() * 2,
      });
      return;
    }
    if (s.kind === "manhole") {
      this.hatches.push({ x: s.x, y: s.y, n: 2 + Math.floor(this.rand() * 3), cd: 0, open: 0 });
      return;
    }
    if (
      s.kind === "barrel" ||
      s.kind === "fuel" ||
      s.kind === "tanker" ||
      s.kind === "ammo" ||
      s.kind === "egg" ||
      s.kind === "propane" ||
      s.kind === "canister" ||
      s.kind === "cylinder" ||
      s.kind === "mine" ||
      s.kind === "tnt"
    ) {
      const hp = {
        barrel: 1,
        fuel: 4,
        tanker: 8,
        ammo: 1,
        egg: 3,
        propane: 1,
        canister: 1,
        cylinder: 1,
        mine: 1,
        tnt: 1,
      }[s.kind];
      this.barrels.push({
        x: s.x,
        y: s.y,
        vx: 0,
        vy: 0,
        ground: true,
        facing: -1,
        type: s.kind,
        hp,
        fuse: -1,
        owner: -3,
        alive: true,
      });
      return;
    }
    if (this.enemies.length >= this.cap() && s.kind !== "battleship") return;
    let y = s.y;
    // Lift spawns out of solid ground; battleships and submarines stay in the water under piers.
    if (s.kind !== "battleship" && s.kind !== "submarine") while (y > TILE && this.solid(s.x, y - 1)) y -= TILE;
    const e = this.vary(this.makeEnemy(s.kind, s.x, y));
    // Ships start off heading either way.
    if (SHIPS.has(e.kind)) e.facing = this.rand() < 0.5 ? -1 : 1;
    this.enemies.push(e);
  }

  /** Pick a train's cars, speed and livery; lines with a station mostly run passenger trains. */
  private makeUp(f: Freight) {
    const r = this.rand;
    f.cars = [0];
    f.passenger = f.stop !== undefined && r() < 0.7;
    f.halted = false;
    if (f.passenger) {
      for (let k = 3 + Math.floor(r() * 4); k > 0; k--) f.cars.push(5);
      f.speed = 60 + r() * 20;
    } else {
      if (r() < 0.35) f.cars.push(0);
      for (let k = 4 + Math.floor(r() * 6); k > 0; k--) f.cars.push(1 + Math.floor(r() * 4));
      f.speed = 46 + r() * 30;
    }
    f.livery = Math.floor(r() * 4);
  }

  /** Add a stretch of main line, extending the previous line if the two meet. */
  private addLine(s: Spawn) {
    const [lo, hi, stop] = s.extra ?? [s.x, s.x + 400];
    const prev = this.freights.find(
      (f) => Math.abs(f.y - s.y) <= TILE * 2 && f.hi >= lo - 3 * TILE && f.hi <= lo + 3 * TILE
    );
    if (prev) {
      prev.hi = Math.max(prev.hi, hi);
      if (stop !== undefined && prev.stop === undefined) prev.stop = stop;
      return;
    }
    const dir = this.rand() < 0.5 ? 1 : -1;
    const f: Freight = {
      x: dir > 0 ? lo : hi,
      y: s.y,
      dir,
      speed: 0,
      cars: [],
      livery: 0,
      lo,
      hi,
      wait: this.rand() * 4,
      anim: 0,
      stop,
    };
    this.makeUp(f);
    this.freights.push(f);
  }

  /** Run each train from one end of its line to the other, then send another after a wait. */
  private updateFreights(dt: number) {
    for (const f of this.freights) {
      if (f.wait > 0) {
        if ((f.wait -= dt) <= 0) {
          f.dir = this.rand() < 0.5 ? 1 : -1;
          f.x = f.dir > 0 ? f.lo : f.hi;
          this.makeUp(f);
        }
        continue;
      }
      const len = f.cars.length * CAR_LEN;
      // Passenger trains brake into the station and stand there a few seconds.
      let v = f.speed;
      if (f.passenger && f.stop !== undefined && !f.halted) {
        const to = (f.stop - (f.x - (f.dir * len) / 2)) * f.dir;
        if (to <= 0) {
          f.halted = true;
          f.halt = 5 + this.rand() * 3;
        } else v = f.speed * clamp(to / 140, 0.12, 1);
      }
      if (f.halt && f.halt > 0) {
        f.halt -= dt;
        continue;
      }
      f.x += f.dir * v * dt;
      f.anim += v * dt * 0.3;
      if (f.dir > 0 ? f.x - len > f.hi : f.x + len < f.lo) {
        f.wait = 4 + this.rand() * 10;
        continue;
      }
      // Exhaust from the engines, on the open line
      for (let k = 0; k < f.cars.length && f.cars[k] === 0; k++) {
        const ex = f.x - f.dir * (k * CAR_LEN + 12);
        if (ex > f.lo && ex < f.hi && this.rand() < dt * 5) this.puff(ex, f.y - 24, 2 + this.rand() * 2);
      }
    }
  }

  /** A metro train on its line, waiting at its first station. */
  private addMetro(s: Spawn) {
    const [lo, hi, ...stops] = s.extra ?? [];
    if (lo === undefined || hi === undefined) return;
    const cars = 4;
    const half = (cars * METRO_CAR) / 2;
    const dir: 1 | -1 = s.x <= (lo + hi) / 2 ? 1 : -1;
    this.metros.push({
      x: clamp(s.x, lo + half + 4, hi - half - 4),
      y: s.y,
      lo,
      hi,
      stops,
      dir,
      v: 0,
      wait: 1 + this.rand() * 3,
      doors: 1,
      cars,
      anim: 0,
      end: false,
      board: true,
    });
  }

  /** Run metro trains station to station, reversing at the ends; the squad may ride one a stop or two. */
  private updateMetros(dt: number) {
    // At some stations the squad, gathered on the platform, waits for a train.
    const up = this.players.filter((p) => !p.dead && !p.aboard && !p.chute);
    if (!this.ride && this.platform < 0 && up.length >= 3) {
      const mx = up.reduce((s, p) => s + p.x, 0) / up.length;
      const my = up.reduce((s, p) => s + p.y, 0) / up.length;
      for (const m of this.metros) {
        if (Math.abs(my - m.y) > 16 || mx < m.lo || mx > m.hi) continue;
        const s = m.stops.find((sx) => Math.abs(sx - mx) < 48 && sx < m.hi - 240);
        if (s === undefined || this.platforms.has(s)) break;
        this.platforms.add(s);
        if (this.rand() < 0.65) {
          this.platform = s;
          this.platformT = 16;
          this.called = m;
          this.popups.push({ x: s, y: m.y - 34, text: "WAIT FOR THE TRAIN", c: "#9ad8ff", t: 0 });
        }
        break;
      }
    }
    if (this.platform >= 0 && ((this.platformT -= dt) <= 0 || this.ride)) {
      this.platform = -1;
      this.called = null;
    }
    for (const m of this.metros) {
      const half = (m.cars * METRO_CAR) / 2;
      const a = m.lo + half + 4;
      const b = m.hi - half - 4;
      if (b <= a) continue;
      if (m.wait > 0) {
        m.doors = Math.min(1, m.doors + dt * 3);
        m.wait -= dt;
        if (m.doors >= 1) {
          if (this.ride === m && (m.end || this.rideHops <= 0)) this.alightAll(m);
          else if (!this.ride && m.board && m.dir > 0 && !m.end && m.wait > 0.6) this.boardMetro(m);
        }
        if (m.wait <= 0 && m.end) {
          m.dir = m.dir > 0 ? -1 : 1;
          m.end = false;
        }
        continue;
      }
      if (m.doors > 0) {
        // Close the doors before pulling away.
        m.doors = Math.max(0, m.doors - dt * 3);
        continue;
      }
      // Next stop: the next station ahead or the end of the line, or the squad's platform if sent for them.
      const pickup = this.called === m && !this.ride ? clamp(this.platform, a, b) : NaN;
      if (!Number.isNaN(pickup) && Math.abs(pickup - m.x) > 1) m.dir = pickup > m.x ? 1 : -1;
      const last = m.dir > 0 ? b : a;
      let target = last;
      for (const s of m.stops) {
        const sx = clamp(s, a, b);
        if ((sx - m.x) * m.dir > 2 && Math.abs(sx - m.x) < Math.abs(target - m.x)) target = sx;
      }
      if (!Number.isNaN(pickup)) target = pickup;
      if (Math.abs(target - m.x) < 1 && !Number.isNaN(pickup)) {
        // Already at their platform: open up, heading their way.
        m.dir = 1;
        m.wait = 3;
        m.end = false;
        m.board = true;
        continue;
      }
      const dist = Math.abs(target - m.x);
      m.v = Math.min(Math.min(130, Math.sqrt(2 * 80 * dist) + 6), m.v + 55 * dt);
      const step = m.v * dt;
      if (step >= dist) {
        m.x = target;
        m.v = 0;
        m.wait = 2.6 + this.rand() * 1.6;
        m.end = Math.abs(target - last) < 1;
        m.board = this.rand() < 0.8;
        if (this.ride === m) this.rideHops--;
        if (target === pickup) {
          // Arrived for the squad: open up, heading their way.
          m.dir = 1;
          m.end = false;
          m.board = true;
          m.wait = 3.5;
        }
      } else m.x += m.dir * step;
      m.anim += step * 0.3;
      if (this.ride === m) this.advance = Math.max(this.advance, m.x);
    }
  }

  /** Board the squad onto a train at their platform, if enough of them are there. */
  private boardMetro(m: Metro) {
    if (this.ship) return;
    const half = (m.cars * METRO_CAR) / 2;
    const up = this.players.filter((p) => !p.dead && !p.aboard && !p.chute);
    const near = up.filter((p) => Math.abs(p.x - m.x) < half + 30 && Math.abs(p.y - m.y) < 14);
    if (!up.length || near.length < Math.max(3, up.length * 0.6)) return;
    // Those close enough get on too.
    this.ride = m;
    for (const p of up) if (Math.abs(p.y - m.y) < 40 && Math.abs(p.x - m.x) < 320) this.board(p);
    this.rideHops = 1 + Math.floor(this.rand() * 2);
    this.popups.push({ x: m.x, y: m.y - 30, text: "ALL ABOARD", c: "#5ab0ff", t: 0 });
  }

  /** Let the squad off the train onto the platform. */
  private alightAll(m: Metro) {
    const half = (m.cars * METRO_CAR) / 2;
    for (const id of [...this.crew]) {
      const p = this.players[id];
      p.aboard = false;
      const x = m.x - half + 10 + this.rand() * (half * 2 - 20);
      Object.assign(p, {
        x,
        y: m.y,
        vx: 0,
        vy: 0,
        ground: true,
        chute: false,
        inv: 1,
        stuckT: 0,
        stall: 0,
        progT: 0,
        back: 0,
        jet: 0,
        bestX: x,
        bestY: m.y,
        lastX: x,
      });
    }
    this.crew = [];
    this.ride = null;
  }

  /** Change the wave every so often, and send reinforcements during assaults and onslaughts. */
  private updateWave(dt: number, view: View) {
    if (this.noWaves) return;
    if ((this.waveT -= dt) <= 0) {
      const r = this.rand();
      let next = r < 0.22 ? 0 : r < 0.58 ? 1 : r < 0.86 ? 2 : 3;
      if (next === this.wave) next = (next + 1 + Math.floor(this.rand() * 3)) % 4;
      // Never go straight between a lull and an onslaught.
      if ((this.wave === 3 && next === 0) || (this.wave === 0 && next === 3)) next = 1;
      this.wave = next;
      this.waveT = [14, 24, 20, 16][next] + this.rand() * 12;
      if (next === 3) this.say("Onslaught", "Enemy forces x3 - hold the line", "#ff3b3b");
      else
        this.popups.push({
          x: view.x + view.w / 2,
          y: view.y + 50,
          text: ["LULL IN THE FIGHTING", "SKIRMISH", "ENEMY ASSAULT"][next],
          c: next === 2 ? "#ff9a2a" : "#9ad8ff",
          t: 0,
        });
    }
    if (this.phase !== "play" || this.theme.enemies <= 0) return;
    // Enemies on screen now; skirmishes set the usual number.
    let here = 0;
    for (const e of this.enemies) if (e.alive && !e.boss && e.x > view.x && e.x < view.x + view.w) here++;
    if (this.wave === 1) this.usual += (here - this.usual) * (1 - Math.exp(-dt / 20));
    if (this.wave === 0 && (this.surge -= dt) <= 0) {
      // In a lull, surplus foot soldiers fall back, furthest ahead first.
      this.surge = 0.5;
      if (here > Math.max(3, this.usual * WAVES[0].k)) {
        let far: Enemy | null = null;
        for (const e of this.enemies)
          if (
            e.alive &&
            !e.boss &&
            !e.retreat &&
            e.scale <= 1.4 &&
            !ARMOURED.has(e.kind) &&
            !FLYING.has(e.kind) &&
            e.kind !== "turret" &&
            e.kind !== "mortar" &&
            e.kind !== "worm" &&
            !e.chute &&
            e.x > view.x &&
            e.x < view.x + view.w &&
            (!far || e.x > far.x)
          )
            far = e;
        if (far) far.retreat = true;
      }
    }
    if (this.wave < 2 || this.buried(view)) return;
    // In an assault or onslaught, reinforce until there are `k` times the usual number.
    if (
      (this.surge -= dt) > 0 ||
      here >= Math.max(8, this.usual) * WAVES[this.wave].k ||
      this.enemies.length >= this.cap()
    )
      return;
    this.surge = this.wave === 3 ? 0.3 : 0.9;
    const n = this.wave === 3 ? 6 + Math.floor(this.rand() * 5) : 3 + Math.floor(this.rand() * 4);
    if (this.rand() < 0.4) {
      // Paratroopers over the squad's heads
      for (let k = 0; k < n; k++) {
        const m = this.makeEnemy(
          this.troop(),
          view.x + view.w * (0.35 + this.rand() * 0.6),
          view.y + 6 + this.rand() * 20
        );
        m.chute = true;
        m.ground = false;
        this.enemies.push(this.vary(m));
      }
      return;
    }
    // Troops running in from just ahead, with armour now and then in an onslaught
    const x = view.x + view.w - 10 + this.rand() * 40;
    const y = surfaceY(this.level, x, view.y - 40);
    if (y >= LEVEL_H || waterAt(this.level, x, y - 2)) return;
    for (let k = 0; k < n; k++) this.enemies.push(this.vary(this.makeEnemy(this.troop(), x + k * 9, y)));
    if (this.wave === 3 && this.rand() < 0.3)
      this.enemies.push(
        this.vary(
          this.makeEnemy(
            (["apc", "lighttank", "armoredcar", "medtank"] as const)[Math.floor(this.rand() * 4)],
            x + 50,
            y
          )
        )
      );
  }

  /** Add a mine cart on the rails at (x, y), y at its wheels; it stays put on a short stretch. */
  private addCart(x: number, y: number) {
    const L = this.level;
    const row = Math.floor(y / TILE) - 1;
    const c = Math.floor(x / TILE);
    if (kindAt(L, c, row) !== RAIL) return;
    let lo = c;
    let hi = c;
    while (kindAt(L, lo - 1, row) === RAIL && c - lo < 90) lo--;
    while (kindAt(L, hi + 1, row) === RAIL && hi - c < 90) hi++;
    const r = this.rand;
    const loads = [0, 0, 1, 2, 3] as const;
    this.carts.push({
      x,
      y: (row + 1) * TILE,
      vx: hi - lo >= 4 ? (r() < 0.5 ? -1 : 1) * (16 + r() * 26) : 0,
      lo: lo * TILE + 8,
      hi: (hi + 1) * TILE - 8,
      load: loads[Math.floor(r() * loads.length)],
      anim: 0,
    });
  }

  /** Roll mine carts between the ends of their rails, running down small enemies in the way. */
  private updateCarts(dt: number) {
    const L = this.level;
    for (const k of this.carts) {
      // Stop if the rail under it is gone.
      if (kindAt(L, Math.floor(k.x / TILE), Math.floor(k.y / TILE) - 1) !== RAIL) k.vx = 0;
      if (!k.vx) continue;
      k.x += k.vx * dt;
      k.anim += k.vx * dt * 0.35;
      if (k.x < k.lo || k.x > k.hi) {
        k.x = clamp(k.x, k.lo, k.hi);
        k.vx = -k.vx;
        this.burst(k.x - sign(k.vx) * 7, k.y - 3, 3, ["#ffe27a", "#ffffff"]);
      }
      if (this.rand() < dt * 3)
        this.add({
          k: SPARK,
          x: k.x + (this.rand() - 0.5) * 10,
          y: k.y - 1,
          vx: -k.vx * 0.4 + (this.rand() - 0.5) * 20,
          vy: -20 - this.rand() * 30,
          life: 0.2,
          c: "#ffd27a",
          s: 1,
        });
      if (Math.abs(k.vx) > 12)
        for (const e of this.enemies)
          if (
            e.alive &&
            !e.boss &&
            e.scale <= 1.4 &&
            !ARMOURED.has(e.kind) &&
            !FLYING.has(e.kind) &&
            Math.abs(e.x - k.x) < 9 &&
            Math.abs(e.y - k.y) < 8
          )
            this.killEnemy(e, -5, k.vx * 3, -150);
    }
  }

  /**
   * A dive bomber's run: dive over the target, release the bomb low down and climb back to cruising
   * height. `phase` 0 cruising, 1 diving, 2 pulling out, 3 done.
   */
  private diveBomb(j: Jet, dt: number, over: boolean, pid: number, view: View) {
    const alt = j.alt ?? j.y;
    // The first ground or water below
    let gy = Math.max(0, view.y);
    while (gy < LEVEL_H && !this.solid(j.x, gy) && !waterAt(this.level, j.x, gy)) gy += TILE;
    const low = gy - 64;
    if (!j.phase && over && j.bombs > 0) j.phase = 1;
    if (j.phase === 1) {
      j.vy = Math.min(j.vy + 420 * dt, 190);
      if (j.y >= low) {
        j.bombs = 0;
        j.phase = 2;
        this.shots.push({
          kind: SHELL,
          x: j.x,
          y: j.y + 5,
          vx: j.vx * 0.55,
          vy: Math.max(60, j.vy * 0.8),
          life: 4,
          dmg: 0,
          pid,
          pierce: 0,
        });
      }
    } else if (j.phase === 2) {
      j.vy = Math.max(j.vy - 520 * dt, -150);
      if (j.y + j.vy * dt <= alt) {
        j.y = alt;
        j.vy = 0;
        j.phase = 3;
      }
    }
    j.y += j.vy * dt;
  }

  /** Surface and floor y (px) of the water column at (x, y), or null if (x, y) isn't water. */
  private waterSpan(x: number, y: number) {
    const L = this.level;
    const c = Math.floor(x / TILE);
    const r = Math.floor(y / TILE);
    if (kindAt(L, c, r) !== WATER) return null;
    let top = r;
    while (top > 0 && kindAt(L, c, top - 1) === WATER) top--;
    let bot = r;
    while (bot < ROWS - 1 && kindAt(L, c, bot + 1) === WATER) bot++;
    return { top: top * TILE, floor: (bot + 1) * TILE };
  }

  /**
   * Move a submarine along its heading, turning where the water ends and easing towards `dive` px
   * below the surface without touching the seabed. `s.y` is the middle of its hull.
   */
  private roam(s: { x: number; y: number; facing: 1 | -1 }, dive: number, dt: number, speed: number) {
    const span = this.waterSpan(s.x, s.y) ?? this.waterSpan(s.x, s.y + TILE);
    if (!span) return;
    const want = clamp(span.top + 2 + dive, span.top + 2, span.floor - 9);
    s.y += clamp(want - s.y, -14 * dt, 14 * dt);
    const ax = s.x + s.facing * 40;
    const clear =
      waterAt(this.level, ax, Math.max(span.top + 1, s.y - 5)) &&
      waterAt(this.level, ax, s.y) &&
      waterAt(this.level, ax, s.y + 6);
    if (clear) s.x += s.facing * speed * dt;
    else s.facing = s.facing > 0 ? -1 : 1;
  }

  /** Air rising through the water. */
  private bubble(x: number, y: number) {
    this.add({
      k: BUBBLE,
      x,
      y,
      vx: (this.rand() - 0.5) * 6,
      vy: -22 - this.rand() * 18,
      life: 3,
      c: "",
      s: this.rand() < 0.3 ? 2 : 1,
    });
  }

  /** Keep a few allied submarines in deep water in view, entering from the edges. */
  private manageNavy(dt: number, view: View) {
    if ((this.navyCd -= dt) > 0) return;
    this.navyCd = 2.5;
    this.navy = this.navy.filter((s) => s.x > view.x - 220 && s.x < view.x + view.w + 420);
    if (this.navy.length >= clamp(Math.round(view.w / 420), 1, 3)) return;
    for (const x of [view.x + view.w + 50, view.x - 50]) {
      // The water surface must be in view, with at least six blocks of water under it.
      const c = Math.floor(x / TILE);
      let r = Math.max(0, Math.floor(view.y / TILE));
      while (r < ROWS && kindAt(this.level, c, r) !== WATER) r++;
      const span = r < ROWS ? this.waterSpan(x, r * TILE + 1) : null;
      if (!span || span.top > view.y + view.h || span.floor - span.top < 6 * TILE) continue;
      const depth = 24 + this.rand() * Math.max(8, span.floor - span.top - 50);
      this.navy.push({
        x,
        y: span.top + depth,
        facing: x > view.x ? -1 : 1,
        dive: depth,
        diveT: 6 + this.rand() * 6,
        samCd: 1 + this.rand() * 2,
        t: 0,
      });
      return;
    }
  }

  /** Move an allied submarine, change its depth now and then, and fire at enemy aircraft. */
  private updateSub(s: Sub, dt: number) {
    s.t += dt;
    if ((s.diveT -= dt) <= 0) {
      s.dive = this.rand() < 0.15 ? 0 : 20 + this.rand() * 100;
      s.diveT = 6 + this.rand() * 8;
    }
    this.roam(s, s.dive, dt, 14);
    if (this.rand() < 0.08) this.bubble(s.x + (this.rand() - 0.5) * 40, s.y - 4);
    if ((s.samCd -= dt) <= 0) s.samCd = this.launchSam(s.x - s.facing * 8, s.y - 6, "ally") ? 3.5 + this.rand() * 3 : 1;
  }

  /** Aircraft that a missile fired by `side` can go after. */
  private airTargets(side: "ally" | "enemy"): AirTarget[] {
    const out: AirTarget[] = [];
    for (const j of this.jets)
      if (j.ally !== (side === "ally") && !j.down && !j.dead)
        out.push({
          ref: j,
          x: () => j.x,
          y: () => j.y,
          vx: () => j.vx,
          up: () => !j.down && !j.dead,
          hit: () => {
            j.hp = 0;
            j.down = true;
            this.popups.push({
              x: j.x,
              y: j.y - 10,
              text: j.ally ? "JET DOWN" : "SPLASH ONE",
              c: j.ally ? "#5ab0ff" : "#ffcc33",
              t: 0,
            });
          },
        });
    if (side === "ally") {
      for (const e of this.enemies)
        if (e.alive && FLYING.has(e.kind) && e.kind !== "drone" && e.kind !== "jetpacker")
          out.push({
            ref: e,
            x: () => e.x,
            y: () => centreY(e),
            vx: () => e.vx,
            up: () => e.alive,
            hit: () => this.hurtEnemy(e, e.boss ? 40 : 999, -5, 1),
          });
    } else {
      for (const a of this.allies)
        if (!a.dead && ALLY_AIR.has(a.kind))
          out.push({
            ref: a,
            x: () => a.x,
            y: () => a.y,
            vx: () => a.vx,
            up: () => !a.dead,
            hit: () => this.hurtAlly(a, a.kind === "gunship" ? 26 : 999),
          });
    }
    return out;
  }

  /** Fire a missile from (x, y) at the nearest unchased enemy aircraft in reach; false if there is none. */
  private launchSam(x: number, y: number, side: "ally" | "enemy") {
    if (this.sams.length >= 10) return false;
    const v = this.lastView;
    const chased = new Set(this.sams.map((s) => s.target?.ref));
    let best: AirTarget | null = null;
    let bestD = 560;
    for (const a of this.airTargets(side)) {
      const ax = a.x();
      if (chased.has(a.ref) || ax < v.x - 40 || ax > v.x + v.w + 160 || a.y() > y - 40) continue;
      const d = Math.abs(ax - x);
      if (d < bestD) {
        best = a;
        bestD = d;
      }
    }
    if (!best) return false;
    const wet = waterAt(this.level, x, y);
    this.sams.push({ x, y, vx: 0, vy: wet ? -70 : -160, side, target: best, t: 0, wet, out: wet ? -1 : 0 });
    for (let k = 0; k < 6; k++) this.bubble(x + (this.rand() - 0.5) * 8, y);
    return true;
  }

  /** Submarine missiles: rise through the water, break the surface, climb, then home in. */
  private updateSams(dt: number, view: View) {
    const keep: Sam[] = [];
    for (const s of this.sams) {
      s.t += dt;
      if (s.wet) {
        s.y += s.vy * dt;
        if (this.rand() < 0.7) this.bubble(s.x + (this.rand() - 0.5) * 4, s.y + 6);
        if (!waterAt(this.level, s.x, s.y)) {
          // Out of the sea: spray, and the motor lights
          s.wet = false;
          s.out = s.t;
          s.vy = -180;
          for (let k = 0; k < 16; k++)
            this.add({
              k: SPARK,
              x: s.x,
              y: s.y + 2,
              vx: (this.rand() - 0.5) * 90,
              vy: -60 - this.rand() * 120,
              life: 0.5,
              c: this.rand() < 0.5 ? "#e0f6ff" : "#8fd3ff",
              s: 1,
            });
          for (let k = 0; k < 6; k++)
            this.add({
              k: MIST,
              x: s.x + (this.rand() - 0.5) * 16,
              y: s.y + 2,
              vx: (this.rand() - 0.5) * 30,
              vy: -10 - this.rand() * 20,
              life: 1,
              c: "",
              s: 2,
            });
          this.flashes.push({ x: s.x, y: s.y + 4, t: 0.08, c: "#ffe27a" });
        }
        if (s.t < 5) keep.push(s);
        continue;
      }
      const tgt = s.target && s.target.up() ? s.target : null;
      const speed = Math.min(440, Math.hypot(s.vx, s.vy) + 520 * dt);
      let a = Math.atan2(s.vy, s.vx);
      // Climb briefly, then turn towards where the target is heading.
      if (tgt && s.t - s.out > 0.3) {
        const d = Math.hypot(tgt.x() - s.x, tgt.y() - s.y);
        const lead = d / Math.max(120, speed);
        const want = Math.atan2(tgt.y() - s.y, tgt.x() + tgt.vx() * lead - s.x);
        let turn = want - a;
        while (turn > Math.PI) turn -= Math.PI * 2;
        while (turn < -Math.PI) turn += Math.PI * 2;
        a += clamp(turn, -4.5 * dt, 4.5 * dt);
      }
      s.vx = Math.cos(a) * speed;
      s.vy = Math.sin(a) * speed;
      s.x += s.vx * dt;
      s.y += s.vy * dt;
      this.add({
        k: SMOKE,
        x: s.x - Math.cos(a) * 12,
        y: s.y - Math.sin(a) * 12,
        vx: (this.rand() - 0.5) * 10,
        vy: (this.rand() - 0.5) * 10,
        life: 1.4 + this.rand(),
        c: "",
        s: 2 + this.rand() * 2,
      });
      this.add({
        k: FLAME,
        x: s.x - Math.cos(a) * 10,
        y: s.y - Math.sin(a) * 10,
        vx: 0,
        vy: 0,
        life: 0.15,
        c: "",
        s: 2,
      });
      const pid = s.side === "ally" ? -5 : -1;
      if (tgt && Math.hypot(tgt.x() - s.x, tgt.y() - s.y) < 14) {
        tgt.hit();
        this.explode(s.x, s.y, 20, pid, false);
        continue;
      }
      // Lost its target or out of fuel: burst in the air.
      if (s.t - s.out > 6 || s.y < view.y - 260 || (!tgt && s.t - s.out > 1.2)) {
        this.explode(s.x, s.y, 14, pid, false);
        continue;
      }
      keep.push(s);
    }
    this.sams = keep;
  }

  /**
   * Sail a ship along the water at `speed`, turning where the water ends. `half` is half its length;
   * big ships also need two blocks of water under them.
   */
  private sail(e: Enemy, dt: number, speed: number, half: number) {
    const row = Math.floor(e.y / TILE);
    const c = Math.floor((e.x + e.facing * (half + 6)) / TILE);
    const deep = half > 30;
    const open = kindAt(this.level, c, row) === WATER && (!deep || kindAt(this.level, c, row + 1) === WATER);
    if (open) e.x += e.facing * speed * dt;
    else e.facing = e.facing > 0 ? -1 : 1;
  }

  /** Give a foot soldier a random size, with health to match, and maybe explosives. */
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

  /** Start the level's finale: hold the squad and schedule the chosen ending's warheads. */
  private startNuke() {
    this.phase = "nuke";
    this.phaseT = 0;
    const pool = FINALES.filter((f) => f !== this.lastFinale);
    const f = this.forceFinale ?? pool[Math.floor(this.rand() * pool.length)];
    this.forceFinale = null;
    this.finale = this.lastFinale = f;
    const targets = this.level.plan.targets;
    this.holdAt = targets[0] - 200;
    this.say(FINALE_TEXT[f][0], FINALE_TEXT[f][1], "#ff3b3b");
    const r = this.rand;
    const order = [...targets].sort(() => r() - 0.5);
    /** A point among the squad, taken at launch time */
    const squad = () => {
      const up = this.players.filter((p) => !p.dead);
      const x = up.length ? up.reduce((s, p) => s + p.x, 0) / up.length : this.advance;
      return x + (r() - 0.5) * 40;
    };
    const top = () => this.lastView.y - 220;
    /** Schedule one finale warhead: it's launched after `delay`, aimed at whatever `aim` gives then */
    const send = (
      delay: number,
      aim: () => number,
      m: (tx: number, ty: number) => Partial<Missile> & { x: number; y: number }
    ) => {
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
          send(
            t,
            () => tx,
            (x) => fromAbove(x)
          );
          t += 0.3 + r() * 0.45;
        }
        send(t + 0.2, squad, (x) => fromAbove(x));
        send(t + 0.5 + r() * 0.3, squad, (x) => ({ ...fromAbove(x), r: 64 }));
        break;
      }
      case "nuke": {
        // A couple of small ones to soften it up, then one enormous warhead straight down on the squad
        send(
          1,
          () => order[0],
          (x) => ({ ...fromAbove(x), r: 36 })
        );
        send(
          1.5 + r() * 0.4,
          () => order[1],
          (x) => ({ ...fromAbove(x), r: 36 })
        );
        send(2.6 + r() * 0.6, squad, (x) => ({ x: x + 20, y: top() - 200, r: 110, speed: 140, max: 520 }));
        break;
      }
      case "cruise": {
        // Low and fast from behind the squad, walking along the stronghold and back onto the squad
        const n = 6 + Math.floor(r() * 4);
        for (let k = 0; k < n; k++) {
          const last = k >= n - 2;
          const aim = last
            ? squad
            : () => targets[0] + ((targets[targets.length - 1] - targets[0]) * k) / (n - 3) + (r() - 0.5) * 30;
          send(1 + k * (0.22 + r() * 0.12), aim, (_x, y) => ({
            x: this.lastView.x - 60,
            y: Math.min(y, this.lastView.y + 40) - 30 - r() * 50,
            r: last ? 52 : 34,
            speed: 380,
            max: 820,
          }));
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
          send(t, onSquad ? squad : () => order[k % order.length] + (r() - 0.5) * 40, (x) => ({
            x: x + (r() - 0.5) * 120,
            y: top(),
            r: onSquad ? 58 : 46,
          }));
          t += 0.25 + r() * 0.35;
        }
        break;
      }
      case "mirv": {
        // One missile that bursts over the battlefield into a dozen warheads, the last few on the squad
        this.pending++;
        this.at(1.2, () => {
          const tx = (targets[0] + squad()) / 2;
          this.missiles.push({
            x: tx + 260,
            y: top(),
            tx,
            ty: this.lastView.y + 30,
            speed: 300,
            up: false,
            fin: true,
            split: 12,
          });
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
      const tx = onSquad
        ? sx + (k - n + 2) * 36 + (this.rand() - 0.5) * 20
        : targets[0] - 60 + this.rand() * (targets[targets.length - 1] - targets[0] + 120);
      this.missiles.push({
        x: m.x + (this.rand() - 0.5) * 20,
        y: m.y,
        tx,
        ty: surfaceY(this.level, tx, 0),
        speed: 200 + this.rand() * 120,
        up: false,
        r: onSquad ? 50 : 30 + this.rand() * 16,
        fin: true,
      });
    }
  }

  /**
   * A non-finale missile lands: a blast sized by its class. Big allied ones also hurt enemies nearby;
   * enemy ones throw the squad about.
   */
  private impact(m: Missile, x: number, y: number) {
    const cls = m.cls ?? 1;
    const r = MISSILE_R[cls];
    const theirs = m.owner === "enemy";
    this.explode(x, y, r, -4, theirs);
    if (r >= 30) this.waves.push({ x, y, t: 0.4 - cls * 0.12 });
    // A mushroom cloud from the biggest and some cruise-sized ones, at most two at a time.
    if (r >= 50 && (cls === 3 || this.rand() < 0.3) && this.mushrooms.filter((m) => m.t < 6).length < 2) {
      this.mushrooms.push({ x, y, t: cls === 3 ? 0 : 0.35, s: cls === 3 ? 1.1 : 0.65 });
      this.surgeDust(x, y, r);
    }
    this.flash = Math.max(this.flash, cls === 3 ? 0.75 : cls === 2 ? 0.25 : 0);
    this.shake = Math.max(this.shake, [3, 6, 10, 16][cls]);
    if (!theirs && cls >= 2)
      for (const e of this.enemies)
        if (e.alive && !e.boss && Math.abs(e.x - x) < r * 1.6) this.hurtEnemy(e, cls === 3 ? 60 : 24, -4, e.x - x);
    if (theirs && cls >= 1)
      for (const p of this.players) {
        if (p.dead || p.chute || p.aboard || Math.hypot(p.x - x, p.y - y) > r * 1.4) continue;
        p.vx = sign(p.x - x) * (120 + r * 2);
        p.vy = -(200 + r * 3);
        p.ground = false;
      }
  }

  /** Pick a vehicle kind for a road; fire engines are likelier where there's a fire. */
  private carKind(rd: Road): CarKind {
    const r = this.rand();
    if (rd.fast)
      return r < 0.45 ? "sedan" : r < 0.6 ? "van" : r < 0.82 ? "truck" : r < 0.9 ? "taxi" : r < 0.96 ? "bus" : "police";
    if (this.fires.some((f) => f.x > rd.lo && f.x < rd.hi && Math.abs(f.y - rd.y) < 24) && r < 0.35) return "firetruck";
    return r < 0.34
      ? "sedan"
      : r < 0.5
        ? "taxi"
        : r < 0.62
          ? "van"
          : r < 0.78
            ? "bus"
            : r < 0.84
              ? "truck"
              : r < 0.91
                ? "police"
                : r < 0.96
                  ? "ambulance"
                  : "firetruck";
  }

  /**
   * Road traffic: vehicles enter at either end and brake for the one ahead and for red lights unless
   * their siren is on. Buses call at stops; fire engines stop to put out fires.
   */
  private updateRoads(dt: number) {
    for (const rd of this.roads) {
      if ((rd.spawnT -= dt) <= 0) {
        rd.spawnT = rd.fast ? 0.7 + this.rand() * 1.3 : 2 + this.rand() * 3.4;
        const dir = this.rand() < 0.5 ? 1 : -1;
        const kind = this.carKind(rd);
        const x = dir > 0 ? rd.lo - 24 : rd.hi + 24;
        const emergency = kind === "police" || kind === "ambulance" || kind === "firetruck";
        if (!rd.cars.some((c) => c.dir === dir && Math.abs(c.x - x) < 70))
          rd.cars.push({
            x,
            dir,
            v: rd.fast ? 120 : 50,
            kind,
            color: Math.floor(this.rand() * 8),
            wait: 0,
            riders: kind === "bus" ? 2 + Math.floor(this.rand() * 8) : 0,
            anim: 0,
            siren: emergency && this.rand() < 0.8,
            last: NaN,
            spray: NaN,
          });
      }
      for (const car of rd.cars) {
        if (car.wait > 0) {
          car.wait -= dt;
          car.v = 0;
          if (!Number.isNaN(car.spray)) this.hose(car, rd, dt);
          if (car.wait <= 0) car.spray = NaN;
          continue;
        }
        const half = CAR_SIZE[car.kind] / 2;
        const top = rd.fast
          ? 150 + (car.color % 3) * 12
          : car.siren
            ? 110
            : car.kind === "bus"
              ? 48
              : car.kind === "truck"
                ? 55
                : 68 + (car.color % 3) * 6;
        let gap = Infinity;
        for (const o of rd.cars)
          if (o !== car && o.dir === car.dir && (o.x - car.x) * car.dir > 0)
            gap = Math.min(gap, Math.abs(o.x - car.x) - half - CAR_SIZE[o.kind] / 2 - 6);
        if (!car.siren)
          for (const lx of rd.lights) {
            const d = (lx - car.x) * car.dir - half - 6;
            if (d >= -1 && d < 70 && lightState(this.t, lx) !== 0) gap = Math.min(gap, d);
          }
        if (car.kind === "bus")
          for (const sx of rd.stops) {
            const d = (sx - car.x) * car.dir;
            if (d < -2 || d > 60 || car.last === sx) continue;
            if (d < 2) {
              car.wait = 2.4 + this.rand() * 1.6;
              car.last = sx;
              car.riders = clamp(car.riders + Math.floor(this.rand() * 6) - 2, 0, 12);
            } else gap = Math.min(gap, d - 1);
          }
        if (car.kind === "firetruck")
          for (const f of this.fires) {
            if (Math.abs(f.y - rd.y) > 24 || f.t <= 0.3) continue;
            const d = (f.x - car.x) * car.dir;
            if (d < -8 || d > 90) continue;
            if (d < 34) {
              car.wait = 2.5;
              car.spray = f.x;
            } else gap = Math.min(gap, d - 34);
          }
        const want = gap <= 0 ? 0 : Math.min(top, Math.sqrt(2 * 160 * gap));
        car.v = want < car.v ? want : Math.min(want, car.v + 70 * dt);
        car.x += car.dir * car.v * dt;
        car.anim += car.v * dt * 0.3;
      }
      rd.cars = rd.cars.filter((c) => (c.dir > 0 ? c.x < rd.hi + 40 : c.x > rd.lo - 40));
    }
  }

  /** A fire engine sprays its hose at a fire, putting it out. */
  private hose(car: Car, rd: Road, dt: number) {
    const nx = car.x + car.dir * 8;
    const ny = rd.y - 14;
    for (let k = 0; k < 3; k++) {
      const d = car.spray - nx;
      this.add({
        k: SPARK,
        x: nx,
        y: ny,
        vx: d * 1.6 + (this.rand() - 0.5) * 20,
        vy: -70 - this.rand() * 30,
        life: 0.5 + this.rand() * 0.2,
        c: this.rand() < 0.5 ? "#cdeef8" : "#7ac8f0",
        s: 1,
      });
    }
    if (this.rand() < 0.3)
      this.add({
        k: MIST,
        x: car.spray + (this.rand() - 0.5) * 10,
        y: rd.y - 6,
        vx: (this.rand() - 0.5) * 10,
        vy: -12,
        life: 0.8,
        c: "",
        s: 2,
      });
    for (const f of this.fires) if (Math.abs(f.x - car.spray) < 16 && Math.abs(f.y - rd.y) < 24) f.t -= dt * 1.6;
  }

  /** Advance the day cycle, one day per 240 s, and change the weather now and then. */
  private updateSky(dt: number) {
    if (!this.theme.cycle) return;
    this.clock = (this.clock + dt / 240) % 1;
    if ((this.skyT -= dt) > 0) return;
    const r = this.rand();
    const next: Sky = r < 0.34 ? "clear" : r < 0.58 ? "cloud" : r < 0.78 ? "rain" : r < 0.9 ? "storm" : "fog";
    this.sky = next === this.sky ? "clear" : next;
    this.skyT = 35 + this.rand() * 40;
  }

  /** The base surge of a big blast: dust rolling out along the ground, both ways, far and fast. */
  private surgeDust(x: number, y: number, r: number) {
    const gy = surfaceY(this.level, x, y - 30);
    const dust = DUST_COLOR[this.theme.fill] ?? "#9a8a78";
    for (let n = 0; n < 26; n++) {
      const dir = n % 2 ? 1 : -1;
      this.add(
        {
          k: DUST,
          x: x + dir * this.rand() * r * 0.3,
          y: gy - 3 - this.rand() * 8,
          vx: dir * (80 + this.rand() * (140 + r * 2)),
          vy: -6 - this.rand() * 14,
          life: 1.4 + this.rand() * 1.2,
          c: dust,
          s: 3 + this.rand() * 4,
        },
        true
      );
    }
  }

  /** A finale warhead lands: a huge blast that wipes out enemies nearby and throws the squad through the air. */
  private detonate(m: Missile, x = m.tx, y = m.ty) {
    const r = m.r ?? 56;
    this.explode(x, y, r, -4, false);
    if (r >= 40) {
      this.mushrooms.push({ x, y, t: 0, s: r >= 90 ? 1.5 : r >= 56 ? 1 : 0.8 });
      this.waves.push({ x, y, t: 0 });
      this.surgeDust(x, y, r);
    }
    this.flash = Math.max(this.flash, r >= 90 ? 0.95 : r >= 40 ? 0.6 : 0.35);
    this.shake = r >= 90 ? 18 : r >= 40 ? 12 : 7;
    for (const e of this.enemies) if (e.alive && Math.abs(e.x - x) < r * 3.6) this.hurtEnemy(e, 999, -4, e.x - x);
    for (const p of this.players) {
      if (p.dead || p.chute || p.aboard) continue;
      const d = Math.hypot(p.x - x, p.y - 8 - y);
      if (d > r * 2.6) continue;
      // Thrown through the air, the closest on fire; they parachute into the next level.
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

  /** After the last warhead: blow up the scene, white out, show the level card, then start the next level. */
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
    // Give a fresh big mushroom cloud a few seconds to rise before the white-out.
    const hold = this.mushrooms.some((m) => m.t < 1.5 && (m.s ?? 1) >= 1) ? 3.4 : 0;
    this.at(1.1 + hold, () => (this.whiteUp = true));
    this.at(1.8 + hold, () => {
      this.phase = "clear";
      this.banner = null;
      const bonus = 3000 + Math.floor(this.rand() * 13) * 500;
      const title = CARD_TITLES[Math.floor(this.rand() * CARD_TITLES.length)];
      const report = {
        no: this.levelNo,
        name: this.theme.name,
        kills: this.levelKills,
        time: this.levelT,
        title,
        finale: this.finale,
        bonus,
      };
      this.complete = report;
      this.log = [report, ...this.log].slice(0, 6);
      this.cleared++;
      for (const p of this.players) p.score += bonus;
    });
    this.at(5.2 + hold, () => this.nextLevel());
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

  /** Whether the body is in water deep enough to swim. */
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
    if (Math.abs(b.y - float) < 4 && this.rand() < 0.15)
      this.add({
        k: SPARK,
        x: b.x - b.facing * 3,
        y: float - hh + 4,
        vx: (this.rand() - 0.5) * 30,
        vy: -30,
        life: 0.3,
        c: "#e0f6ff",
        s: 1,
      });
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
        this.add({
          k: FLAME,
          x: p.x + (this.rand() - 0.5) * 4,
          y: p.y - 8,
          vx: -p.vx * 0.15,
          vy: -20,
          life: 0.4 + this.rand() * 0.25,
          c: "",
          s: 2,
        });
        if (this.rand() < 0.4) this.puff(p.x, p.y - 10, 2);
      }
      if (p.deadT <= 0) this.respawn(p, view);
      return;
    }
    if (p.aboard) {
      // Riding a metro train or the gunship: move with it, out of harm's way.
      const a = this.ride ?? (this.ship && !this.ship.dead ? this.ship : null);
      if (a) {
        p.x = a.x + ((p.id % 4) - 1.5) * (this.ride ? 14 : 4);
        p.y = this.ride ? a.y - 2 : a.y + 6;
        p.vx = p.vy = 0;
        p.lastX = p.x;
        p.inv = Math.max(p.inv, 0.5);
        return;
      }
      p.aboard = false;
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
    if (p.ray > 0) this.deathRay(p, dt, tgt);

    // While the gunship waits to board, everyone runs for its door.
    const ship = this.ship && this.shipMode === "board" && !this.ship.dead ? this.ship : null;
    const desired = ship
      ? ship.x + ((p.id % 4) - 1.5) * 3
      : this.advance + OFFSETS[p.id] + Math.sin(this.t * 0.8 + p.id * 1.7) * 5;
    const dx = desired - p.x;
    let dir = Math.abs(dx) > 4 ? sign(dx) : 0;
    if (p.back > 0) {
      // Out from under the overhang; then a jump for its edge
      p.back -= dt;
      dir = -sign(dx);
      if (p.back <= 0 && p.ground) {
        p.vy = -JUMP;
        p.ground = false;
      }
    }
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
    // Probe further than one step reaches, so a wall is always seen before the step hits it.
    const probe = p.x + dir * 6;
    // Test the same heights as integrate(), so a climber is never clear of the wall by one check
    // and blocked by the other, stuck at the top.
    const blocked =
      dir !== 0 && (this.solid(probe, p.y - 2) || this.solid(probe, p.y - PH / 2) || this.solid(probe, p.y - PH + 2));
    p.climbing = false;
    if (blocked) {
      vx = 0;
      // Under a parachute, only drift down; climbing would leave it hanging off the rock.
      if (p.chute) {
        /* down first */
      } else if (p.ground) {
        p.vy = -JUMP;
        p.ground = false;
      } else if (p.vy > -CLIMB && p.vy < 140) {
        // Grab the wall only while rising or near the top of a jump, not falling fast down a shaft.
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
    if (
      !p.chute &&
      !p.jet &&
      p.ground &&
      dir &&
      this.rand() < dt * 0.15 &&
      surfaceY(this.level, p.x + dir * 40, p.y - 120) < p.y - 32
    )
      p.jet = 2.5;
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
      if (this.rand() < 0.9)
        this.add({
          k: FLAME,
          x: p.x - p.facing * 3,
          y: p.y - 6,
          vx: -p.facing * 10,
          vy: 60 + this.rand() * 40,
          life: 0.2,
          c: "",
          s: 1 + Math.round(this.rand()),
        });
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
      this.popups.push({
        x: p.x,
        y: p.y - 26,
        text: `P${p.id + 1} ${d.weapon < 0 ? "JETPACK" : WEAPONS[d.weapon].toUpperCase()}`,
        c: this.colors[p.id],
        t: 0,
      });
      this.burst(p.x, p.y - 10, 10, [this.colors[p.id], "#ffffff"]);
    }
    // Climbing into an overhang counts as stuck at once, so the grenade comes sooner.
    if (climbing && p.vy === 0) {
      p.stuckT += dt;
      // Back off and come at it again rather than pressing on underneath.
      if (p.back <= 0 && p.stuckT > 0.3) p.back = 0.4;
    }
    p.anim += dt * (vx ? 12 : 0);
    // Wading through a river throws up spray
    if (vx && p.ground && waterAt(this.level, p.x, p.y - 3) && this.rand() < 0.35)
      this.add({
        k: SPARK,
        x: p.x - p.facing * 3,
        y: p.y - 8,
        vx: -p.facing * 20 + (this.rand() - 0.5) * 30,
        vy: -50 - this.rand() * 40,
        life: 0.35,
        c: this.rand() < 0.5 ? "#e0f6ff" : "#8fd3ff",
        s: 1,
      });

    if (lavaAt(this.level, p.x, p.y - 2)) {
      this.burst(p.x, p.y - 4, 10, ["#ffd23f", "#ff8a1a"]);
      this.hurtPlayer(p, 99);
      return;
    }

    // Stuck against something: blow it up, and respawn if still stuck. Climbing a cliff counts as moving.
    if (dir && Math.abs(p.x - p.lastX) < 0.3 && !(climbing && p.vy < 0)) {
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
    // Well behind its place and making no progress, such as circling in a blasted pocket or stuck
    // under a ledge: blast a way on, then respawn.
    if (dx < 40 || p.chute) {
      p.bestX = p.x;
      p.bestY = p.y;
      p.progT = 0;
    } else if (p.x > p.bestX + 2 || p.y < p.bestY - 4) {
      // Each best is kept separately, so a loop that only gets higher further back is no progress.
      p.bestX = Math.max(p.bestX, p.x);
      p.bestY = Math.min(p.bestY, p.y);
      p.progT = 0;
    } else if ((p.progT += dt) > 4 && p.progT - dt <= 4) {
      this.grenade(p, p.x + sign(dx) * 20);
      p.jet = 3;
    }
    if (p.stall > 6 || p.progT > 9) this.respawn(p, view);
  }

  /** Nearest enemy in range and sight, lobbed weapons needing no sight; or a barrel with an enemy next to it. */
  private findTarget(p: Player): Enemy | Barrel | null {
    const range = W_RANGE[p.weapon];
    const gx = p.x;
    const gy = p.y - 9;
    const lobs = LOBBED.has(p.weapon);
    // Only targets on screen, so the fight stays in view.
    const v = this.lastView;
    let best: Enemy | Barrel | null = null;
    let bestD = range;
    for (const e of this.enemies) {
      if (!e.alive) continue;
      const aim = this.aimPoint(e);
      if (!aim || aim.x > v.x + v.w - 6 || aim.x < v.x + 4) continue;
      const d = Math.hypot(aim.x - p.x, (aim.y - gy) * 0.6);
      // Prefer targets ahead, and bosses.
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

  /** Where to aim at an enemy, or null if it can't be hit right now. */
  aimPoint(e: Enemy): { x: number; y: number } | null {
    if (e.kind === "worm") {
      const s = e.segs.find((q) => q.up);
      return s ? { x: s.x, y: s.y } : null;
    }
    // A submerged submarine can't be hit.
    if (e.kind === "submarine" && e.spin > 0) return null;
    return { x: e.x, y: centreY(e) };
  }

  private fire(p: Player, tgt: Enemy | Barrel) {
    const gx = p.x + p.facing * 7;
    const gy = p.y - 9;
    const aim =
      "kind" in tgt ? (this.aimPoint(tgt) ?? { x: tgt.x, y: tgt.y }) : { x: tgt.x, y: tgt.y - SIZE[tgt.type][1] / 2 };
    const a = Math.atan2(aim.y - gy, aim.x - gx);
    const r = this.rand;
    const shot = (ang: number, speed: number, kind: number, dmg: number, life: number, pierce = 0) =>
      this.shots.push({
        kind,
        x: gx,
        y: gy,
        vx: Math.cos(ang) * speed,
        vy: Math.sin(ang) * speed,
        life,
        dmg,
        pid: p.id,
        pierce,
      });
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
          if (q && segDist(q.x, q.y, gx, gy, x1, y1) < SIZE[e.kind][0] + 3)
            this.hurtEnemy(e, 2.2, p.id, Math.cos(a) * 60);
        }
        this.beams.push({ x0: gx, y0: gy, x1, y1, t: 0.1, c: this.colors[p.id] });
        for (let k = 0; k < 4; k++)
          this.add({ k: SPARK, x: x1, y: y1, vx: (r() - 0.5) * 80, vy: -r() * 80, life: 0.25, c: "#ffffff", s: 1 });
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
      case 12: // saw blades: ricochet and cut through
        shot(a, 230, SAW, 1.2, 1.8, 8);
        p.fireCd = 0.6;
        break;
      case 14: // pulse laser: bursts of bolts that pierce one target
        shot(a + (r() - 0.5) * 0.04, 560, PULSE, 1.3, 0.45, 1);
        this.flashes.push({ x: gx + p.facing * 2, y: gy, t: 0.04, c: "#ffffff" });
        p.burst++;
        p.fireCd = p.burst % 4 === 0 ? 0.34 : 0.07;
        break;
      case 15: // death ray: a held beam that burns through cover and ground
        p.ray = 0.9;
        p.rayA = a;
        p.rayCut = 0;
        p.fireCd = 2.2;
        this.deathRay(p, 1 / 60, tgt);
        break;
      case 16: {
        // prism laser: the beam splits into three where it strikes
        const hit = this.beamTo(p, gx, gy, a, W_RANGE[16], 2.6, this.colors[p.id], 2);
        const near = this.enemies
          .filter((e) => e.alive && e !== hit.first && Math.hypot(e.x - hit.x, centreY(e) - hit.y) < 130)
          .sort((u, v) => Math.hypot(u.x - hit.x, u.y - hit.y) - Math.hypot(v.x - hit.x, v.y - hit.y))
          .slice(0, 3);
        near.forEach((e, i) => {
          const q = this.aimPoint(e) ?? { x: e.x, y: centreY(e) };
          this.hurtEnemy(e, 1.7, p.id, q.x - hit.x);
          this.beams.push({
            x0: hit.x,
            y0: hit.y,
            x1: q.x,
            y1: q.y,
            t: 0.14,
            c: RAINBOW[(i + p.id) % RAINBOW.length],
            w: 1,
          });
          for (let k = 0; k < 3; k++)
            this.add({
              k: GLINT,
              x: q.x,
              y: q.y,
              vx: (r() - 0.5) * 60,
              vy: -r() * 60,
              life: 0.4,
              c: RAINBOW[(i + k) % RAINBOW.length],
              s: 1,
            });
        });
        for (let k = 0; k < 6; k++)
          this.add({
            k: GLINT,
            x: hit.x,
            y: hit.y,
            vx: (r() - 0.5) * 90,
            vy: -r() * 80,
            life: 0.5,
            c: RAINBOW[k % RAINBOW.length],
            s: k % 2 ? 2 : 1,
          });
        p.fireCd = 0.55;
        break;
      }
      default: {
        // mortar tube: shells lobbed high, onto the enemy's head
        const v = lobAt(gx, gy, aim.x + (r() - 0.5) * 16, aim.y, SHOT_GRAV[MORTAR], 150);
        this.shots.push({ kind: MORTAR, x: gx, y: gy - 2, vx: v.vx, vy: v.vy, life: 4, dmg: 0, pid: p.id, pierce: 0 });
        this.puff(gx, gy - 4, 3);
        p.fireCd = 1.3;
      }
    }
    if (p.weapon !== 5 && p.weapon !== 8 && p.weapon !== 9 && p.weapon < 14)
      this.flashes.push({ x: gx + p.facing * 2, y: gy, t: 0.05, c: this.colors[p.id] });
    if (p.weapon === 0 || p.weapon === 1 || p.weapon === 4 || p.weapon === 7 || p.weapon === 10)
      this.add({
        k: CASING,
        x: p.x,
        y: gy,
        vx: -p.facing * (20 + r() * 30),
        vy: -60 - r() * 40,
        life: 0.8,
        c: "#ffd27a",
        s: 1,
      });
  }

  /**
   * Fire a laser beam from (gx, gy) at angle `a`, up to `len` or the first wall, hurting everyone on
   * it. Returns the first enemy struck and where, or the beam's end if it struck none.
   */
  private beamTo(p: Player, gx: number, gy: number, a: number, len: number, dmg: number, c: string, w = 1) {
    let end = len;
    for (let d = 6; d < len; d += 4)
      if (this.solid(gx + Math.cos(a) * d, gy + Math.sin(a) * d)) {
        end = d;
        break;
      }
    const x1 = gx + Math.cos(a) * end;
    const y1 = gy + Math.sin(a) * end;
    let first: Enemy | null = null;
    let fd = Infinity;
    for (const e of this.enemies) {
      if (!e.alive) continue;
      const q = this.aimPoint(e);
      if (!q || segDist(q.x, q.y, gx, gy, x1, y1) >= SIZE[e.kind][0] + 3) continue;
      this.hurtEnemy(e, dmg, p.id, Math.cos(a) * 60);
      const d = Math.hypot(q.x - gx, q.y - gy);
      if (d < fd) {
        fd = d;
        first = e;
      }
    }
    // The prism splits from the first enemy struck.
    const hx = first ? (this.aimPoint(first) ?? { x: first.x, y: centreY(first) }).x : x1;
    const hy = first ? (this.aimPoint(first) ?? { x: first.x, y: centreY(first) }).y : y1;
    this.beams.push({ x0: gx, y0: gy, x1, y1, t: 0.12, c, w });
    for (let k = 0; k < 4; k++)
      this.add({
        k: SPARK,
        x: x1,
        y: y1,
        vx: (this.rand() - 0.5) * 80,
        vy: -this.rand() * 80,
        life: 0.25,
        c: "#ffffff",
        s: 1,
      });
    return { x: hx, y: hy, first };
  }

  /** The held death ray: swing onto the target and burn through enemies, barrels and blocks along it. */
  private deathRay(p: Player, dt: number, tgt: Enemy | Barrel | null) {
    p.ray -= dt;
    const L = this.level;
    const gx = p.x + p.facing * 7;
    const gy = p.y - 9;
    if (tgt) {
      const aim = "kind" in tgt ? (this.aimPoint(tgt) ?? { x: tgt.x, y: tgt.y }) : { x: tgt.x, y: tgt.y - 6 };
      let d = Math.atan2(aim.y - gy, aim.x - gx) - p.rayA;
      while (d > Math.PI) d -= Math.PI * 2;
      while (d < -Math.PI) d += Math.PI * 2;
      p.rayA += clamp(d, -2.4 * dt, 2.4 * dt);
    }
    const ux = Math.cos(p.rayA);
    const uy = Math.sin(p.rayA);
    const len = W_RANGE[15];
    let end = len;
    for (let d = 6; d < len; d += 3) {
      const x = gx + ux * d;
      const y = gy + uy * d;
      if (!this.solid(x, y)) continue;
      end = d;
      // Scorch the block it hits and break it bit by bit.
      if ((p.rayCut += dt) > 0.06) {
        p.rayCut = 0;
        const c = Math.floor(x / TILE);
        const row = Math.floor(y / TILE);
        const kind = kindAt(L, c, row);
        if (damageCell(L, c, row, 2)) {
          this.debris(c, row, kind, 2);
          this.dirtyAround(c, row);
          this.loosen(c, row);
        } else if (L.burn[cell(c, row)] < 3) L.burn[cell(c, row)]++;
        this.dirty.push(c, row);
      }
      break;
    }
    const x1 = gx + ux * end;
    const y1 = gy + uy * end;
    for (const e of this.enemies) {
      if (!e.alive) continue;
      const q = this.aimPoint(e);
      if (q && segDist(q.x, q.y, gx, gy, x1, y1) < SIZE[e.kind][0] + 4) this.hurtEnemy(e, 10 * dt, p.id, ux * 40);
    }
    for (const b of this.barrels)
      if (b.alive && segDist(b.x, b.y - 4, gx, gy, x1, y1) < 8) this.lightBarrel(b, p.id, 0.05);
    this.beams.push({ x0: gx, y0: gy, x1, y1, t: 0.05, c: this.colors[p.id], w: 3 });
    if (this.rand() < 0.6)
      this.add({
        k: GLINT,
        x: x1,
        y: y1,
        vx: (this.rand() - 0.5) * 70,
        vy: -this.rand() * 70,
        life: 0.35,
        c: this.rand() < 0.5 ? "#ffffff" : this.colors[p.id],
        s: 2,
      });
    if (this.rand() < 0.4) this.flame(x1, y1);
  }

  private grenade(p: Player, tx: number) {
    const dist = Math.max(30, Math.min(140, Math.abs(tx - p.x)));
    const dir = tx >= p.x ? 1 : -1;
    this.shots.push({
      kind: GRENADE,
      x: p.x,
      y: p.y - 12,
      vx: dir * dist * 1.05,
      vy: -170,
      life: 1.1,
      dmg: 0,
      pid: p.id,
      pierce: 0,
    });
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

  /** Take a player out of the fight until the next transport drops them back in. */
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
    // Deep in a cavern, drop in just above the nearest teammate: the squad's average height could be
    // inside a pocket in the cavern roof.
    let mate: Player | null = null;
    for (const q of this.players)
      if (!q.dead && !q.chute && q !== p && (!mate || Math.abs(q.x - x) < Math.abs(mate.x - x))) mate = q;
    let from = y;
    if (mate && mate.y - crustOf(this.level, Math.floor(mate.x / TILE)) * TILE > 12 * TILE) {
      x = clamp(x, mate.x - 30, mate.x + 30);
      from = Math.max(y, mate.y - 40);
    }
    Object.assign(p, {
      dead: false,
      queued: false,
      hp: PLAYER_HP,
      inv: 1.5,
      x,
      y: this.openAir(x, from),
      vx: 0,
      vy: 0,
      ground: false,
      chute: true,
      stuckT: 0,
      stall: 0,
      spin: 0,
      burn: 0,
      jet: 0,
      target: null,
    });
  }

  /** The first y at x, from `y` down, with room to stand, so a player dropping into a cavern lands inside it. */
  private openAir(x: number, y: number) {
    for (let yy = Math.max(PH, y); yy < LEVEL_H - TILE; yy += TILE)
      if (!this.solid(x, yy) && !this.solid(x, yy - PH + 2)) return yy;
    return y;
  }

  /** Keep a flier above the land, so helicopters at view height don't sink into mountains or cavern roofs. */
  private aboveLand(e: Enemy) {
    const top = this.landTop(e.x, SIZE[e.kind][0]) * TILE - SIZE[e.kind][1] - 6;
    if (e.y > top) e.y = top;
  }

  /** The highest surface row within `hw` px either side of x. */
  private landTop(x: number, hw: number) {
    const L = this.level;
    return Math.min(
      crustOf(L, Math.floor((x - hw) / TILE)),
      crustOf(L, Math.floor(x / TILE)),
      crustOf(L, Math.floor((x + hw) / TILE))
    );
  }

  /** Whether the view's top is below the land surface, deep in a cavern with no sky above. */
  private buried(view: View) {
    return view.y > crustOf(this.level, Math.floor((view.x + view.w / 2) / TILE)) * TILE + 24;
  }

  /** The nearest player or allied vehicle in range, and in sight if `sight`. */
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
      if (p.dead || p.aboard) continue;
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
        // Trail flame and smoke.
        e.burn -= dt;
        const y = e.y - 7;
        this.add({
          k: FLAME,
          x: e.x + (this.rand() - 0.5) * 4,
          y,
          vx: -e.vx * 0.15,
          vy: -20,
          life: 0.35 + this.rand() * 0.25,
          c: "",
          s: 2,
        });
        if (this.rand() < 0.5) this.add({ k: FLAME, x: e.x, y: y + 2, vx: 0, vy: -10, life: 0.25, c: "", s: 1 });
        if (this.rand() < 0.35) this.puff(e.x, y - 2, 2);
      }
      return;
    }
    if (e.retreat) {
      // Falling back in a lull: off ahead at a run, gone once it's out of sight
      e.facing = 1;
      e.vx = 70;
      this.integrate(e, dt, 3, 14);
      e.anim += dt * 10;
      if (e.x > view.x + view.w + 30) {
        e.alive = false;
        e.deadT = 0;
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
      // Only once on screen; lobbed weapons don't need a clear line.
      e.target = this.inView(e)
        ? this.nearestFoe(
            e,
            range,
            e.kind !== "mortar" && e.kind !== "grenadier" && e.kind !== "dynamiter" && e.kind !== "officer",
            e.kind === "sniper" ? 200 : 120
          )
        : null;
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
      e.fireCd -= dt * 1.6;
      // Grenades, at a cluster of the squad by preference
      if (e.nade <= 0 && e.kind !== "sniper" && e.kind !== "flamer" && Math.abs(tgt.x - e.x) < 170) {
        const cluster = this.players.filter((q) => !q.dead && Math.hypot(q.x - tgt.x, q.y - tgt.y) < 40).length;
        if (cluster >= 2 || this.rand() < 0.35) {
          const v = lobAt(e.x, e.y - 12, tgt.x + (this.rand() - 0.5) * 12, tgt.y - 2, GRAV, 150);
          this.shots.push({
            kind: GRENADE,
            x: e.x,
            y: e.y - 12,
            vx: v.vx,
            vy: v.vy,
            life: 1.4,
            dmg: 0,
            pid: -1,
            pierce: 0,
          });
        }
        e.nade = 5 + this.rand() * 4;
      }
      // Hurt badly: fall back a little way, still firing, then come on again
      if (e.hp < e.maxHp * 0.35 && e.cover <= 0 && this.rand() < dt * 0.6) e.cover = 1.4;
      if (e.kind === "sniper") {
        // Line up the shot with the laser sight on, then fire
        e.aim += dt;
        if (e.aim > 0.9 && e.fireCd <= 0) {
          this.enemyShot(e, tgt, 520, 2, 0.02);
          e.aim = 0;
          e.fireCd = 2.4;
        }
      } else if (e.fireCd <= 0) this.enemyFire(e, tgt);
      // Close to their preferred range and hold it: push in, back off if too close, jump obstacles.
      const gap = Math.abs(tgt.x - e.x);
      const ADV: Partial<Record<EnemyKind, number>> = {
        mook: 34,
        heavy: 18,
        rocket: 24,
        shield: 22,
        flamer: 46,
        grenadier: 26,
        minigunner: 20,
        officer: 24,
        dynamiter: 42,
        engineer: 28,
        laser: 30,
        exosuit: 22,
      };
      const fixed = e.kind === "turret" || e.kind === "mg" || e.kind === "mortar" || e.kind === "sniper";
      if (e.ground && !e.chute && !fixed) {
        const want = e.kind === "flamer" ? 40 : e.kind === "shield" ? 50 : range * 0.55;
        const speed = (ADV[e.kind] ?? 24) * (e.scale > 2 ? 1.25 : e.scale > 1 ? 1.2 : e.scale < 1 ? 1.4 : 1);
        if (e.cover > 0) e.vx = -e.facing * speed * 0.8;
        else if (gap > want + 16) e.vx = e.facing * speed;
        else if (gap < Math.min(28, want * 0.5)) e.vx = -e.facing * speed * 0.6;
        if (e.vx && this.solid(e.x + sign(e.vx) * (dims(e)[0] + 3), e.y - 3)) e.vy = -230;
        if (e.vx && this.rand() < 0.04)
          this.add({ k: DUST, x: e.x, y: e.y - 1, vx: -e.vx * 0.3, vy: -10, life: 0.5, c: "#9a8a78", s: 1.5 });
      }
    } else {
      e.aim = 0;
      if (e.kind !== "turret" && e.kind !== "mg" && e.kind !== "mortar" && !e.chute) {
        // Patrol a few steps either side of where it was placed
        const ahead = e.x + e.facing * 5;
        if (Math.abs(ahead - e.home) > 18 || this.solid(ahead, e.y - 4) || !this.solid(ahead, e.y + 2))
          e.facing = e.facing === 1 ? -1 : 1;
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
    // Lead a moving target.
    const lead = ((p.vx ?? 0) * Math.hypot(p.x - gx, p.y - gy)) / speed;
    const a = Math.atan2(p.y - 9 - gy, p.x + lead - gx) + (this.rand() - 0.5) * spread * 0.6;
    // Bosses, giants and heavy units hit harder.
    const heavy = e.boss
      ? 2.5
      : e.scale > 1
        ? e.scale * 1.6
        : e.kind === "heavytank" || e.kind === "exosuit" || e.kind === "heavy" || e.kind === "minigunner"
          ? 1.6
          : 1;
    this.shots.push({
      kind: ENEMY_BULLET,
      x: gx,
      y: gy,
      vx: Math.cos(a) * speed,
      vy: Math.sin(a) * speed,
      life: 1.6,
      dmg: dmg * heavy,
      pid: -1,
      pierce: 0,
    });
    this.flashes.push({ x: gx, y: gy, t: 0.05, c: "#ff9a2a" });
    if (this.rand() < 0.3)
      this.add({
        k: CASING,
        x: e.x,
        y: gy,
        vx: -e.facing * (20 + this.rand() * 30),
        vy: -60 - this.rand() * 40,
        life: 0.7,
        c: "#c8a050",
        s: 1,
      });
  }

  /** Whether an enemy is on screen or nearly; off-screen enemies hold their fire. */
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
            this.at(1 + k * 0.25, () =>
              this.shots.push({
                kind: SHELL,
                x: tx + (this.rand() - 0.5) * 70,
                y: this.lastView.y - 10,
                vx: 0,
                vy: 280,
                life: 4,
                dmg: 0,
                pid: -3,
                pierce: 0,
              })
            );
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
          this.barrels.push({
            x: e.x - 12,
            y: e.y,
            vx: 0,
            vy: 0,
            ground: true,
            facing: -1,
            type: "mine",
            hp: 1,
            fuse: -1,
            owner: -3,
            alive: true,
          });
        else this.enemyShot(e, p, 170, 1, 0.15);
        e.fireCd = 1 + r() * 0.6;
        return;
      case "laser": {
        // A bolt of light: fast and hard-hitting
        const gx = e.x + e.facing * 8;
        const gy = e.y - 9;
        const a = Math.atan2(p.y - 9 - gy, p.x - gx) + (r() - 0.5) * 0.08;
        this.shots.push({
          kind: ELASER,
          x: gx,
          y: gy,
          vx: Math.cos(a) * 420,
          vy: Math.sin(a) * 420,
          life: 1,
          dmg: 2,
          pid: -1,
          pierce: 0,
        });
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
            this.shots.push({
              kind: ROCKET,
              x: e.x,
              y: e.y - 24,
              vx: Math.cos(a) * 110,
              vy: Math.sin(a) * 110,
              life: 2.2,
              dmg: 0,
              pid: -1,
              pierce: 0,
            });
          }
        return;
    }
    if (e.kind === "rocket" && this.rand() < 0.4) {
      // A guided missile now and then
      this.missiles.push({
        x: e.x + e.facing * 6,
        y: e.y - 11,
        tx: p.x,
        ty: surfaceY(this.level, p.x, p.y - 30),
        speed: 150,
        max: 420,
        up: false,
        owner: "enemy",
        cls: 0,
      });
      e.fireCd = 2.6 + r();
      return;
    }
    if (e.kind === "rocket") {
      const a = Math.atan2(p.y - 9 - (e.y - 11), p.x - e.x);
      this.shots.push({
        kind: ROCKET,
        x: e.x + e.facing * 6,
        y: e.y - 11,
        vx: Math.cos(a) * 110,
        vy: Math.sin(a) * 110,
        life: 2,
        dmg: 0,
        pid: -1,
        pierce: 0,
      });
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
        this.shots.push({
          kind: FLAMESHOT,
          x: e.x + e.facing * 7,
          y: e.y - 9,
          vx: Math.cos(ang) * v,
          vy: Math.sin(ang) * v,
          life: 0.45,
          dmg: 1,
          pid: -1,
          pierce: 0,
        });
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
      this.shots.push({
        kind: GRENADE,
        x: e.x,
        y: e.y - 12,
        vx: v.vx,
        vy: v.vy,
        life: 1.5,
        dmg: 0,
        pid: -1,
        pierce: 0,
      });
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
    if (this.rand() < 0.9)
      this.add({
        k: FLAME,
        x: e.x - e.facing * 3,
        y: e.y - 4,
        vx: 0,
        vy: 60 + this.rand() * 30,
        life: 0.18,
        c: "",
        s: 1 + Math.round(this.rand()),
      });
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

  /** Enemy armour: each kind drives to its own stand-off distance from the squad and fights its own way. */
  private updateArmour(e: Enemy, dt: number) {
    const k = e.kind;
    const STOP: Record<string, number> = {
      apc: 210,
      lighttank: 240,
      medtank: 230,
      heavytank: 250,
      flametank: 80,
      rockettank: 320,
      aatank: 260,
      launcher: 300,
      armoredcar: 200,
      artytruck: 380,
      hovercraft: 220,
    };
    const SPEED: Record<string, number> = {
      apc: 40,
      lighttank: 30,
      medtank: 32,
      heavytank: 26,
      flametank: 38,
      rockettank: 26,
      aatank: 30,
      launcher: 24,
      armoredcar: 52,
      artytruck: 30,
      hovercraft: 44,
    };
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
      air =
        this.allies.find((a) => !a.dead && ALLY_AIR.has(a.kind) && Math.abs(a.x - e.x) < 380) ??
        this.players.find((q) => !q.dead && q.jet > 0 && Math.abs(q.x - e.x) < 320) ??
        null;
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
            this.shots.push({
              kind: FLAMESHOT,
              x: gx,
              y: gy,
              vx: Math.cos(a) * v,
              vy: Math.sin(a) * v,
              life: 0.55,
              dmg: 1,
              pid: -1,
              pierce: 0,
            });
          }
          e.burst++;
          e.fireCd = e.burst % 30 === 0 ? 1.4 : 0.05;
        }
      } else if (air) {
        // Flak: shells that burst at the target's range
        const d = Math.hypot(air.x - gx, air.y - gy);
        const a = e.aimA + (this.rand() - 0.5) * 0.12;
        this.shots.push({
          kind: FLAK,
          x: gx,
          y: gy,
          vx: Math.cos(a) * 320,
          vy: Math.sin(a) * 320,
          life: d / 320 + (this.rand() - 0.5) * 0.08,
          dmg: 0,
          pid: -1,
          pierce: 0,
        });
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
          this.shots.push({
            kind: CANNON,
            x: gx,
            y: gy - j * 4,
            vx: v.vx,
            vy: v.vy,
            life: 4,
            dmg: 0,
            pid: -1,
            pierce: 0,
          });
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
          this.shots.push({
            kind: ROCKET,
            x: e.x - 8,
            y: e.y - 14,
            vx: Math.cos(a) * 120,
            vy: Math.sin(a) * 120,
            life: 2.4,
            dmg: 0,
            pid: -1,
            pierce: 0,
          });
        }
        e.altCd = 2.6 + this.rand();
        break;
      case "rockettank":
        // A ripple of six rockets, up and over onto the target
        for (let j = 0; j < 6; j++)
          this.at(j * 0.16, () => {
            if (!e.alive) return;
            this.missiles.push({
              x: e.x - 6,
              y: e.y - 22,
              tx: tgt.x + (this.rand() - 0.5) * 60,
              ty: surfaceY(this.level, tgt.x, 0),
              speed: 160,
              max: 520,
              up: false,
              boost: true,
              owner: "enemy",
              cls: 0,
            });
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
        this.missiles.push({
          x: e.x + 18,
          y: e.y - 30,
          tx: tgt.x + (this.rand() - 0.5) * 30,
          ty: surfaceY(this.level, tgt.x, 0),
          speed: 90,
          max: [520, 560, 600, 520][cls],
          up: false,
          boost: true,
          owner: "enemy",
          cls,
        });
        this.popups.push({
          x: e.x,
          y: e.y - 40,
          text: cls === 3 ? "ICBM LAUNCH!" : "MISSILE LAUNCH",
          c: "#ff3b3b",
          t: 0,
        });
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
      const pool: EnemyKind[] = [
        "mook",
        "minigunner",
        "rusher",
        "dynamiter",
        "laser",
        "engineer",
        "officer",
        "rocket",
        "shield",
      ];
      const m = this.vary(this.makeEnemy(pool[Math.floor(this.rand() * pool.length)], e.x + 20, e.y - 4));
      m.vy = -120;
      m.vx = 30;
      m.ground = false;
      m.alert = true;
      m.home = e.x + 30;
      this.enemies.push(m);
    }
  }

  /** Attack, transport and scout helicopters: hover over the squad and fire, then fly off. */
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
        this.shots.push({
          kind: ROCKET,
          x: e.x - 6,
          y: e.y + 6,
          vx: Math.cos(a) * 120,
          vy: Math.sin(a) * 120,
          life: 2.4,
          dmg: 0,
          pid: -1,
          pierce: 0,
        });
      }
      e.burst++;
      if (e.burst % 2)
        this.missiles.push({
          x: e.x,
          y: e.y + 6,
          tx: p.x,
          ty: surfaceY(this.level, p.x, 0),
          speed: 140,
          max: 480,
          up: false,
          owner: "enemy",
          cls: 0,
        });
      e.altCd = 2.6 + this.rand();
    } else if (k === "transport") {
      // A stick of paratroopers out of the back
      for (let n = 0; n < 4 && this.enemies.length < this.cap(); n++) {
        const m = this.makeEnemy(
          (["mook", "minigunner", "rusher", "dynamiter", "laser"] as EnemyKind[])[Math.floor(this.rand() * 5)],
          e.x + 8 + n * 10,
          e.y + 8
        );
        m.chute = true;
        m.ground = false;
        this.enemies.push(m);
      }
      e.altCd = 9;
      e.life = Math.min(e.life, 4);
    } else e.altCd = 99;
  }

  /** Dug-in emplacements: flak guns, howitzers, SAM sites and rocket pods. */
  private updateEmplacement(e: Enemy, dt: number) {
    const k = e.kind;
    e.facing = -1;
    this.integrate(e, dt, SIZE[k][0], SIZE[k][1]);
    e.hover = Math.max(0, e.hover - dt);
    if (!this.inView(e)) return;
    const air: Foe | null =
      k === "flakgun" || k === "samsite"
        ? (this.allies.find((a) => !a.dead && ALLY_AIR.has(a.kind) && Math.abs(a.x - e.x) < 420) ??
          this.jets.find((j) => j.ally && !j.down && Math.abs(j.x - e.x) < 380) ??
          this.players.find((q) => !q.dead && q.jet > 0 && Math.abs(q.x - e.x) < 360) ??
          null)
        : null;
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
          this.shots.push({
            kind: FLAK,
            x: gx,
            y: gy,
            vx: Math.cos(e.aimA) * 330,
            vy: Math.sin(e.aimA) * 330,
            life: d / 330 + (this.rand() - 0.5) * 0.1,
            dmg: 0,
            pid: -1,
            pierce: 0,
          });
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
            this.missiles.push({
              x: e.x,
              y: e.y - 18,
              tx: tgt.x,
              ty: flying ? tgt.y : surfaceY(this.level, tgt.x, 0),
              speed: 120,
              max: 620,
              up: false,
              boost: !flying,
              owner: "enemy",
              cls: flying ? 0 : 1,
            });
            this.puff(e.x, e.y - 16, 4);
          });
        e.fireCd = 3.6 + this.rand();
        break;
      default:
        for (let j = 0; j < 8; j++)
          this.at(j * 0.12, () => {
            if (!e.alive) return;
            this.missiles.push({
              x: e.x,
              y: e.y - 16,
              tx: tgt.x + (this.rand() - 0.5) * 70,
              ty: surfaceY(this.level, tgt.x, 0),
              speed: 170,
              max: 520,
              up: false,
              boost: true,
              owner: "enemy",
              cls: 0,
            });
          });
        e.fireCd = 7 + this.rand() * 3;
    }
  }

  /** An armoured train: shuttles near the squad, firing guns along its length and rocket salvos. */
  private updateTrain(e: Enemy, dt: number) {
    e.facing = -1;
    const tx = Math.max(e.home - 320, this.advance + 150 + Math.sin(this.t * 0.3 + e.home * 0.013) * 110);
    e.vx = clamp((tx - e.x) * 0.7, -30, 30);
    if (Math.abs(e.vx) < 2) e.vx = 0;
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
      this.shots.push({
        kind: ENEMY_BULLET,
        x: gx,
        y: e.y - 18,
        vx: Math.cos(a) * 210,
        vy: Math.sin(a) * 210,
        life: 1.6,
        dmg: 1,
        pid: -1,
        pierce: 0,
      });
      this.flashes.push({ x: gx, y: e.y - 18, t: 0.05, c: "#ff9a2a" });
      e.burst++;
      e.fireCd = e.burst % 10 === 0 ? 0.8 : 0.07;
    }
    if (e.altCd <= 0) {
      for (let j = 0; j < 6; j++)
        this.at(j * 0.14, () => {
          if (!e.alive) return;
          this.missiles.push({
            x: e.x + 30,
            y: e.y - 22,
            tx: p.x + (this.rand() - 0.5) * 60,
            ty: surfaceY(this.level, p.x, 0),
            speed: 160,
            max: 520,
            up: false,
            boost: true,
            owner: "enemy",
            cls: 0,
          });
        });
      e.altCd = 6 + this.rand() * 3;
    }
  }

  /**
   * An enemy submarine: roams at varying depths, fires missiles at allied aircraft, and attacks the
   * squad when surfaced. `spin` is 1 when submerged, `hover` its target depth, `life` seconds until
   * it picks another.
   */
  private updateSubmarine(e: Enemy, dt: number) {
    e.anim += dt;
    if (e.samCd === undefined) {
      // First update: set its timers and dive straight away.
      e.samCd = 2 + this.rand() * 3;
      e.life = this.rand() * 3;
      e.hover = 24 + this.rand() * 60;
    }
    e.life -= dt;
    if (e.life <= 0) {
      e.hover = this.rand() < 0.3 ? 0 : 18 + this.rand() * 100;
      e.life = e.hover ? 6 + this.rand() * 7 : 5 + this.rand() * 4;
    }
    this.roam(e, e.hover, dt, e.hover ? 15 : 10);
    const span = this.waterSpan(e.x, e.y);
    const was = e.spin;
    e.spin = span && e.y - span.top > 9 ? 1 : 0;
    // Breaking the surface, or going under: a burst of spray
    if (was !== e.spin && span)
      for (let k = 0; k < 10; k++)
        this.add({
          k: MIST,
          x: e.x + (this.rand() - 0.5) * 50,
          y: span.top - 1,
          vx: (this.rand() - 0.5) * 20,
          vy: -20 - this.rand() * 30,
          life: 1,
          c: "",
          s: 2,
        });
    if (e.spin && this.rand() < 0.1) this.bubble(e.x + (this.rand() - 0.5) * 40, e.y - 4);
    e.samCd -= dt;
    if (e.samCd <= 0)
      e.samCd = this.inView(e) && this.launchSam(e.x - e.facing * 8, e.y - 6, "enemy") ? 4 + this.rand() * 3 : 1;
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
          this.missiles.push({
            x: e.x + 4,
            y: e.y - 8,
            tx: p.x + (this.rand() - 0.5) * 30,
            ty: surfaceY(this.level, p.x, 0),
            speed: 100,
            max: 560,
            up: false,
            boost: true,
            owner: "enemy",
            cls: 1,
          });
          for (let k = 0; k < 4; k++) this.puff(e.x + 4, e.y - 8, 4);
        });
      e.altCd = 5 + this.rand() * 3;
    }
  }

  /** Keep the allied column up to strength, with more of it when the view is wider. */
  private manageAllies(dt: number, view: View) {
    if ((this.allyCd -= dt) > 0 || this.phase === "clear") return;
    const ground = this.allies.filter((a) => !a.dead && !ALLY_AIR.has(a.kind));
    const air = this.allies.filter((a) => !a.dead && ALLY_AIR.has(a.kind) && a.kind !== "gunship");
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
      const kind: AllyKind =
        roll < 0.34 ? "mbt" : roll < 0.56 ? "ifv" : roll < 0.7 ? "scoutcar" : roll < 0.86 ? "mlrs" : "aa";
      const x = Math.max(this.level.start * TILE + 24, view.x - 20);
      const y = surfaceY(this.level, x, 0);
      // Not into the sea.
      if (kindAt(this.level, Math.floor(x / TILE), Math.floor(y / TILE) - 1) !== WATER)
        this.allies.push(this.makeAlly(kind, x, y, slotOf(ground)));
    } else if (air.length < wantA) {
      const kind: AllyKind = !air.some((a) => a.kind === "lifter") && r() < 0.3 ? "lifter" : "aheli";
      this.allies.push(this.makeAlly(kind, view.x - 40, view.y + 30, slotOf(air)));
    }
  }

  private makeAlly(kind: AllyKind, x: number, y: number, slot: number): Ally {
    return {
      kind,
      x,
      y,
      vx: 0,
      vy: 0,
      ground: true,
      facing: 1,
      hp: ALLY_HP[kind],
      maxHp: ALLY_HP[kind],
      dead: false,
      deadT: 0,
      fireCd: 1 + this.rand(),
      altCd: 2 + this.rand() * 3,
      burst: 0,
      scanT: 0,
      aimA: 0,
      anim: 0,
      slot,
      hit: 0,
      target: null,
    };
  }

  /** Allied vehicles keep their place in the column behind or above the squad and fight whatever's in view. */
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
        if (a.kind === "gunship") {
          // Spinning down in flames
          a.anim += dt * 3;
          if (this.rand() < 0.8)
            this.add({
              k: FLAME,
              x: a.x + (this.rand() - 0.5) * 14,
              y: a.y - 4,
              vx: -20,
              vy: -30,
              life: 0.35,
              c: "",
              s: 2,
            });
        }
        if (this.solid(a.x, a.y + 6) || a.y > LEVEL_H) {
          this.explode(
            a.x,
            a.y,
            a.kind === "gunship" ? 44 : 26,
            -5,
            false,
            a.kind === "gunship" ? PAL_NAPALM : PAL_FIRE
          );
          if (a.kind === "gunship") this.fires.push({ x: a.x, y: surfaceY(L, a.x, a.y - 20), t: 6 });
          a.deadT = 0;
        }
      } else if (this.rand() < 0.3) this.puff(a.x, a.y - 10, 3);
      return;
    }
    if (a.kind === "gunship") this.flyShip(a, dt, view);
    else if (air) {
      const tx = Math.max(view.x + 30, this.advance + 30 + a.slot * 50 + Math.sin(this.t * 0.5 + a.slot) * 20);
      const ty =
        Math.max(view.y + 22, surfaceY(L, a.x, view.y) - 70 - (a.slot % 2) * 20) + Math.sin(this.t * 2 + a.slot) * 3;
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
        // Stop at the shore.
        if (ground > a.y - 42 && kindAt(L, Math.floor(nx / TILE), Math.floor(ground / TILE) - 1) !== WATER) a.x = nx;
        else a.vx = 0;
      }
      const g = surfaceY(L, a.x, a.y - 44);
      a.y = g < a.y ? g : Math.min(g, a.y + 200 * dt);
      a.anim += dt * Math.abs(a.vx) * 0.3;
    }
    a.facing = 1;
    a.scanT -= dt;
    if (a.scanT <= 0) {
      a.scanT = 0.3;
      a.target = this.allyTarget(a, view);
    }
    const tgt = a.target?.alive ? a.target : null;
    if (!tgt) return;
    const aim = this.aimPoint(tgt) ?? { x: tgt.x, y: tgt.y };
    const gy = a.kind === "gunship" ? a.y + 7 : air ? a.y + 4 : a.y - 16;
    const gx = a.kind === "gunship" ? a.x + 16 : a.x + 10;
    if (a.kind === "gunship" && this.crew.length && (a.burst += dt) > 0.12) {
      // The squad aboard fire the door guns; the kills are theirs.
      a.burst = 0;
      const id = this.crew[Math.floor(this.rand() * this.crew.length)];
      const dx = a.x - 2;
      const dy = a.y + 4;
      const ang = Math.atan2(aim.y - dy, aim.x - dx) + (this.rand() - 0.5) * 0.12;
      this.shots.push({
        kind: BULLET,
        x: dx,
        y: dy,
        vx: Math.cos(ang) * 400,
        vy: Math.sin(ang) * 400,
        life: 1.2,
        dmg: 2,
        pid: id,
        pierce: 0,
      });
      this.flashes.push({ x: dx, y: dy, t: 0.05, c: this.colors[id] });
    }
    a.aimA = Math.atan2(aim.y - gy, aim.x - gx);
    a.fireCd -= dt;
    a.altCd -= dt;
    const bullet = (dmg: number, speed: number, spread: number) => {
      const ang = a.aimA + (this.rand() - 0.5) * spread;
      this.shots.push({
        kind: BULLET,
        x: gx,
        y: gy,
        vx: Math.cos(ang) * speed,
        vy: Math.sin(ang) * speed,
        life: 1.2,
        dmg,
        pid: -5,
        pierce: 0,
      });
      this.flashes.push({ x: gx, y: gy, t: 0.05, c: "#9ad8ff" });
    };
    if (a.fireCd <= 0) {
      if (a.kind === "aa" && FLYING.has(tgt.kind)) {
        const d = Math.hypot(aim.x - gx, aim.y - gy);
        this.shots.push({
          kind: FLAK,
          x: gx,
          y: gy,
          vx: Math.cos(a.aimA) * 320,
          vy: Math.sin(a.aimA) * 320,
          life: d / 320,
          dmg: 0,
          pid: -5,
          pierce: 0,
        });
        a.burst++;
        a.fireCd = a.burst % 6 === 0 ? 0.9 : 0.14;
      } else if (a.kind === "gunship") {
        // Its chin cannon, in long bursts
        bullet(3, 420, 0.08);
        a.fireCd = this.rand() < 0.1 ? 0.6 : 0.08;
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
            this.missiles.push({
              x: a.x - 4,
              y: a.y - 22,
              tx: tgt.x + (this.rand() - 0.5) * 50,
              ty: surfaceY(this.level, tgt.x, 0),
              speed: 160,
              max: 520,
              up: false,
              boost: true,
              owner: "ally",
              cls: 0,
            });
          });
        a.altCd = 8 + this.rand() * 3;
        break;
      case "aheli":
        for (let j = 0; j < 2; j++) {
          const ang = a.aimA + (j - 0.5) * 0.1;
          this.shots.push({
            kind: ROCKET,
            x: gx,
            y: gy + 2,
            vx: Math.cos(ang) * 120,
            vy: Math.sin(ang) * 120,
            life: 2.4,
            dmg: 0,
            pid: -5,
            pierce: 0,
          });
        }
        a.altCd = 2.6 + this.rand();
        break;
      case "gunship":
        // A salvo from its rocket pods, and now and then a guided missile
        for (let j = 0; j < 4; j++)
          this.at(j * 0.09, () => {
            if (a.dead) return;
            const ang = a.aimA + (this.rand() - 0.5) * 0.16;
            this.shots.push({
              kind: ROCKET,
              x: a.x + 6,
              y: a.y + 9,
              vx: Math.cos(ang) * 130,
              vy: Math.sin(ang) * 130,
              life: 2.4,
              dmg: 0,
              pid: -5,
              pierce: 0,
            });
          });
        if (this.rand() < 0.35)
          this.missiles.push({
            x: a.x,
            y: a.y + 10,
            tx: tgt.x,
            ty: surfaceY(this.level, tgt.x, 0),
            speed: 170,
            max: 520,
            up: false,
            boost: true,
            owner: "ally",
            cls: 0,
          });
        a.altCd = 2 + this.rand() * 0.8;
        break;
      case "lifter":
        // Drops the squad a supply crate
        this.drops.push({
          x: a.x,
          y: a.y + 10,
          vy: 0,
          weapon: this.rand() < 0.2 ? -1 : Math.floor(this.rand() * WEAPONS.length),
          ground: false,
          t: 0,
        });
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
      if ((a.kind === "mbt" || a.kind === "ifv") && !flying && !lineOfSight(this.level, a.x + 10, gy, q.x, q.y))
        continue;
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
    if (a.kind === "gunship") this.shipDown(a);
    else this.popups.push({ x: a.x, y: a.y - 24, text: "ALLY DOWN", c: "#5ab0ff", t: 0 });
    this.allyCd = Math.max(this.allyCd, 5);
  }

  /* -------------------------------------------------------------- */
  /* The squad's gunship                                             */
  /* -------------------------------------------------------------- */

  /** Now and then, out in the open, send the squad's gunship in to pick them up. */
  private manageShip(dt: number, view: View) {
    const s = this.ship;
    if (s && (!this.allies.includes(s) || (s.dead && s.deadT <= 0))) this.ship = null;
    if (this.ship || this.noShip || this.ride) return;
    if ((this.shipCd -= dt) > 0) return;
    this.shipCd = 3;
    const plan = this.level.plan;
    const arena = this.level.bossQueue[0];
    if (
      this.phase !== "play" ||
      this.levelT < 10 ||
      this.buried(view) ||
      plan.fortPlaced ||
      (arena && arena.col * TILE - this.advance < 900)
    )
      return;
    const up = this.players.filter((p) => !p.dead && !p.chute && !p.aboard);
    if (up.length < 5) return;
    // It needs dry ground just ahead of the squad, near their height.
    const tx = this.advance + 30;
    const g = surfaceY(this.level, tx, view.y - 40);
    if (waterAt(this.level, tx, g - 2) || Math.abs(g - up.reduce((m, p) => m + p.y, 0) / up.length) > 40) return;
    const ship = this.makeAlly("gunship", view.x - 70, Math.max(view.y + 10, g - 120), 9);
    this.ship = ship;
    this.allies.push(ship);
    this.shipMode = "inbound";
    this.shipT = 0;
    // Just ahead of the squad, so they reach it about when it lands.
    this.shipPad = this.advance + 110;
    this.crew = [];
    this.popups.push({ x: this.advance, y: g - 70, text: "GUNSHIP INBOUND", c: "#5ab0ff", t: 0 });
  }

  /** The highest y (px) of the land or buildings between `x0` and `x1`, looking down from well above `y`. */
  private shipFloor(x0: number, x1: number, y: number) {
    let top = Infinity;
    const from = Math.max(0, y - 260);
    for (let x = x0; x <= x1; x += 12) top = Math.min(top, surfaceY(this.level, x, from), this.landTop(x, 0) * TILE);
    return top;
  }

  /** The gunship's hover y for setting down at x: skids just above the ground, never below allied flight height. */
  private hoverY(x: number, y: number) {
    const [hw, hh] = ALLY_SIZE.gunship;
    return Math.min(this.shipFloor(x - hw, x + hw, y), this.landTop(x, hw) * TILE) - hh - 7;
  }

  /** Fly the gunship through its phases: inbound, board, fly, land and leave. */
  private flyShip(a: Ally, dt: number, view: View) {
    this.shipT += dt;
    a.anim += dt;
    const ease = (v: number, to: number, rate: number) => v + (to - v) * (1 - Math.exp(-dt * rate));
    const plan = this.level.plan;
    const arena = this.level.bossQueue[0];
    switch (this.shipMode) {
      case "inbound": {
        // In over the squad's heads to the landing spot ahead of them, and down to just over the ground there
        const tx = this.shipPad;
        const ty = this.hoverY(tx, a.y);
        a.x = Math.min(a.x + 170 * dt, ease(a.x, tx, 2.2));
        const near = Math.abs(a.x - tx) < 70;
        a.y += clamp((near ? ty : Math.min(a.y, ty - 50)) - a.y, -110 * dt, 100 * dt);
        if (Math.abs(a.x - tx) < 4 && Math.abs(a.y - ty) < 4) {
          this.shipMode = "board";
          this.shipT = 0;
          this.popups.push({ x: a.x, y: a.y - 28, text: "ALL ABOARD", c: "#5ab0ff", t: 0 });
        } else if (this.shipT > 16) this.shipMode = "leave";
        break;
      }
      case "board": {
        // Hover with the doors open while the squad climbs in; after 9 s anyone nearby is pulled up.
        const hy = this.hoverY(a.x, a.y);
        a.y = ease(a.y, hy - 1 + Math.sin(this.t * 3) * 1.5, 3);
        for (const p of this.players)
          if (
            !p.dead &&
            !p.aboard &&
            !p.chute &&
            ((Math.abs(p.x - a.x) < 14 && Math.abs(p.y - (hy + 21)) < 20) ||
              (this.shipT > 9 && Math.abs(p.x - a.x) < 220))
          )
            this.board(p);
        if (this.players.every((p) => p.dead || p.aboard || p.chute) || this.shipT > 9.5) {
          if (!this.crew.length) {
            this.shipMode = "leave";
            this.shipEmpty = true;
          } else {
            this.shipMode = "fly";
            this.shipT = 0;
            this.advance = Math.max(this.advance, a.x);
          }
        }
        break;
      }
      case "fly": {
        // Fly on low over the land, climbing over what's ahead; the squad's advance moves with it.
        const floor = this.shipFloor(a.x - 36, a.x + 180, a.y);
        const ty = floor - 62 + Math.sin(this.t * 1.7) * 4;
        const climb = a.y - ty;
        a.vx = 96 * clamp(1 - (climb - 30) / 140, 0.25, 1);
        a.x += a.vx * dt;
        a.y += clamp(ty - a.y, -140 * dt, 100 * dt);
        this.advance = Math.max(this.advance, a.x);
        // Pick up anyone waiting to respawn.
        while (this.dropQueue.length) {
          const p = this.players[this.dropQueue.shift()!];
          Object.assign(p, { dead: false, queued: false, hp: PLAYER_HP, burn: 0, spin: 0 });
          this.board(p);
        }
        // The enemy sends helicopters up after it, the longer it's up the more of them
        if (this.shipT > 24 && this.shipT % 10 < dt && this.enemies.length < this.cap()) {
          const h = this.makeEnemy(
            this.rand() < 0.6 ? "attackheli" : "heli",
            view.x + view.w + 40,
            a.y - 20 - this.rand() * 30
          );
          h.hover = clamp(a.y - view.y - 10 + (this.rand() - 0.5) * 40, 20, view.h - 60);
          h.home = Math.floor(this.rand() * 100);
          this.enemies.push(h);
        }
        // Where it has to set them down: a boss ahead, the stronghold, the fuel low
        const end = plan.fortPlaced ? (plan.targets[0] ?? plan.stronghold * TILE) : plan.stronghold * TILE;
        const low = this.shipT > 80;
        if (
          this.phase !== "play" ||
          (arena && arena.col * TILE - a.x < 420) ||
          (plan.fortPlaced && end - a.x < 520) ||
          low
        ) {
          if (low) this.popups.push({ x: a.x, y: a.y - 26, text: "FUEL LOW", c: "#ffcc33", t: 0 });
          this.shipMode = "land";
          this.shipT = 0;
        }
        break;
      }
      case "land": {
        // Down to the ground, and everyone out
        const ty = this.hoverY(a.x, a.y);
        a.x += 20 * dt;
        a.y += clamp(ty - a.y, -80 * dt, 90 * dt);
        this.advance = Math.max(this.advance, a.x - 20);
        if (Math.abs(a.y - ty) < 6 || this.shipT > 6) {
          const wet = waterAt(this.level, a.x, ty + 2);
          for (const id of [...this.crew]) this.unload(this.players[id], a, wet || this.shipT > 6);
          this.shipMode = "leave";
          this.shipT = 0;
        }
        break;
      }
      case "leave":
        // Up and away
        a.x += 110 * dt;
        a.y -= 70 * dt;
        if (a.y < view.y - 80 || a.x > view.x + view.w + 120) {
          a.dead = true;
          a.deadT = 0;
          this.ship = null;
          this.shipCd = this.shipEmpty ? 10 : 50 + this.rand() * 40;
          this.shipEmpty = false;
        }
        break;
    }
    a.facing = 1;
  }

  /** Put a player aboard the gunship or the metro train the squad is riding. */
  private board(p: Player) {
    if (p.aboard) return;
    p.aboard = true;
    p.chute = false;
    p.jet = 0;
    p.vx = p.vy = 0;
    p.ray = 0;
    const host = this.ride ?? this.ship;
    if (host) {
      p.x = host.x;
      p.y = this.ride ? host.y - 2 : host.y + 6;
    }
    if (!this.crew.includes(p.id)) this.crew.push(p.id);
  }

  /** Drop a player out of the gunship, by parachute if `chute`. */
  private unload(p: Player, a: Ally, chute: boolean) {
    p.aboard = false;
    this.crew = this.crew.filter((id) => id !== p.id);
    const k = this.rand();
    Object.assign(p, {
      x: a.x - 10 + k * 20,
      y: a.y + 14,
      vx: (k - 0.5) * 90,
      vy: chute ? -80 - this.rand() * 60 : 0,
      ground: false,
      chute,
      inv: 2,
      stuckT: 0,
      stall: 0,
      progT: 0,
      back: 0,
      jet: 0,
    });
    p.bestX = p.x;
    p.bestY = p.y;
    p.lastX = p.x;
  }

  /** The gunship's going down: everyone aboard bails out, and it's a while before another comes. */
  private shipDown(a: Ally) {
    for (const id of [...this.crew]) this.unload(this.players[id], a, true);
    this.popups.push({ x: a.x, y: a.y - 30, text: "GUNSHIP DOWN - BAIL OUT", c: "#ff6a3d", t: 0 });
    this.say("Gunship down", "Bail out!", "#ff6a3d");
    this.shake = Math.max(this.shake, 6);
    this.shipCd = 45 + this.rand() * 45;
    this.shipMode = "leave";
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
      for (let k = 0; k < 4; k++)
        this.add({
          k: DEBRIS,
          x: h.x,
          y: h.y - 2,
          vx: (this.rand() - 0.5) * 80,
          vy: -120 - this.rand() * 80,
          life: 1,
          c: "#3a3a40",
          s: 2,
        });
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
      for (let k = 0; k < 4; k++)
        this.at(
          k * 0.18,
          () =>
            e.alive &&
            this.shots.push({
              kind: SHELL,
              x: e.x - 20 + k * 12,
              y: e.y + 14,
              vx: -20,
              vy: 60,
              life: 4,
              dmg: 0,
              pid: -1,
              pierce: 0,
            })
        );
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

  /** The doom cannon: lobs volleys of three heavy shells and sprays the squad with its machine gun. */
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
      this.shots.push({
        kind: ELASER,
        x: gx,
        y: gy,
        vx: Math.cos(a) * 380,
        vy: Math.sin(a) * 380,
        life: 1.2,
        dmg: 2,
        pid: -1,
        pierce: 0,
      });
      this.flashes.push({ x: gx, y: gy, t: 0.06, c: "#c77dff" });
      e.burst++;
      e.fireCd = e.burst % 5 === 0 ? 1.4 : 0.18;
    }
    if (e.altCd <= 0 && this.enemies.length < this.cap()) {
      this.barrels.push({
        x: e.x,
        y: e.y,
        vx: 0,
        vy: 0,
        ground: true,
        facing: -1,
        type: "egg",
        hp: 3,
        fuse: -1,
        owner: -3,
        alive: true,
      });
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
    const m = this.makeEnemy(
      n < 0.2 ? "rocket" : n < 0.32 ? "heavy" : n < 0.45 ? "grenadier" : "mook",
      e.x + 18,
      e.y - 4
    );
    m.vy = -110;
    m.vx = 30;
    m.ground = false;
    m.alert = true;
    m.home = e.x + 30;
    this.enemies.push(m);
  }

  /**
   * A battleship sailing slowly back and forth, shelling the squad from two turrets, with flak and a
   * deck gun. Gun and funnel offsets are for facing left; `fore` mirrors them.
   */
  private updateBattleship(e: Enemy, dt: number) {
    e.anim += dt;
    this.sail(e, dt, 10, 56);
    const fore = -e.facing;
    e.scanT -= dt;
    if (e.scanT <= 0) {
      e.scanT = 0.4;
      e.target = this.nearestPlayer(e, 420, false, 260);
    }
    if (this.rand() < 0.5) this.puff(e.x + 14 * fore, e.y - 34, 3);
    const p = e.target && !e.target.dead ? e.target : null;
    if (p) e.aimA = Math.atan2(p.y - (e.y - 22), p.x - e.x);
    e.fireCd -= dt;
    if (p && e.fireCd <= 0) {
      // A salvo from both turrets
      e.fireCd = 2.4 + this.rand() * 1.4;
      for (const dx of [-30 * fore, 38 * fore]) {
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
      for (const dx of [-14 * fore, 14 * fore, 28 * fore]) {
        const gx = e.x + dx;
        const gy = e.y - 16;
        const a = Math.atan2(air.y - gy, air.x - gx) + (this.rand() - 0.5) * 0.12;
        const d = Math.hypot(air.x - gx, air.y - gy);
        this.shots.push({
          kind: FLAK,
          x: gx,
          y: gy,
          vx: Math.cos(a) * 340,
          vy: Math.sin(a) * 340,
          life: d / 340,
          dmg: 0,
          pid: -1,
          pierce: 0,
        });
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
    // Races up and down the water it's on, bank to bank, its gunner hosing the shore
    this.sail(e, dt, 26, 19);
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
          this.shots.push({
            kind: ROCKET,
            x: e.x - 10,
            y: e.y + 8,
            vx: Math.cos(a) * 120,
            vy: Math.sin(a) * 120,
            life: 2.5,
            dmg: 0,
            pid: -1,
            pierce: 0,
          });
        });
      e.altCd = 3.2;
    }
  }

  /** The tank and walker bosses: advance, stop short of the squad, and fire a cannon or rockets plus machine guns. */
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
            this.shots.push({
              kind: ROCKET,
              x: e.x + 4,
              y: e.y - 50,
              vx: Math.cos(a) * 110,
              vy: Math.sin(a) * 110 - 40,
              life: 2.5,
              dmg: 0,
              pid: -1,
              pierce: 0,
            });
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
      this.explode(
        e.x,
        y,
        e.kind === "gunship" || e.kind === "airship"
          ? 50
          : e.kind === "drone" || e.kind === "jetpacker"
            ? 14
            : e.kind === "transport"
              ? 40
              : 32,
        -3,
        true
      );
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
          : Array.from({ length: 8 }, () => ({
              x: e.x + (this.rand() - 0.5) * hw * 2,
              y: centreY(e) + (this.rand() - 0.5) * hh,
            }));
      pts.forEach((q, i) => this.at(i * 0.12, () => this.explode(q.x, q.y, 16 + this.rand() * 10, pid, false)));
      this.at(pts.length * 0.12 + 0.1, () => this.explode(e.x, centreY(e), 44, pid, false));
      if (e.kind !== "gunship") this.at(pts.length * 0.12 + 0.2, () => this.fires.push({ x: e.x, y: e.y, t: 8 }));
      return;
    }
    if (e.kind === "turret" || e.kind === "heavy" || e.kind === "mg")
      this.explode(e.x, e.y - 4, e.kind === "heavy" ? 10 : 14, pid, false);
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
      for (let k = 0; k < 5; k++)
        this.at(k * 0.2, () => this.explode(e.x - 24 + this.rand() * 48, e.y - 4, 18, pid, false));
    }
    if (e.kind === "flakgun" || e.kind === "howitzer" || e.kind === "samsite" || e.kind === "rocketpod") {
      this.explode(e.x, e.y - 8, 26, pid, true);
      this.fires.push({ x: e.x, y: e.y, t: 4 });
    }
    if (e.kind === "battleship") {
      // It goes up stem to stern, then the magazine, then it settles and sinks
      e.deadT = 5;
      for (let k = 0; k < 9; k++)
        this.at(k * 0.16, () =>
          this.explode(e.x - 40 + this.rand() * 80, e.y - 6 - this.rand() * 24, 14 + this.rand() * 12, pid, false)
        );
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

  /** Damage an enemy; at zero health it dies and is flung. */
  private hurtEnemy(e: Enemy, dmg: number, pid: number, vx: number) {
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
    if (
      b.type === "mine" &&
      b.fuse < 0 &&
      this.players.some((p) => !p.dead && Math.abs(p.x - b.x) < 6 && Math.abs(p.y - b.y) < 8)
    ) {
      // Stepped on: it goes off a moment later
      this.popups.push({ x: b.x, y: b.y - 12, text: "!", c: "#ff3b3b", t: 0 });
      this.lightBarrel(b, -3, 0.12);
    }
    const vx = b.vx;
    const rocket = (b.type === "propane" || b.type === "cylinder") && b.fuse >= 0;
    this.integrate(b, dt, hw, hh, rocket ? 0.35 : 1);
    if (b.ground) b.vx = vx * 0.5;
    if (rocket) {
      // A jet of flame and smoke out of the bottom
      for (let k = 0; k < 2; k++)
        this.add({
          k: FLAME,
          x: b.x + (this.rand() - 0.5) * 3,
          y: b.y + 1,
          vx: (this.rand() - 0.5) * 20,
          vy: 60 + this.rand() * 40,
          life: 0.25,
          c: "",
          s: 2,
        });
      this.puff(b.x, b.y + 3, 2);
    }
    if (b.fuse >= 0) {
      b.fuse -= dt;
      if (this.rand() < 0.6)
        this.add({
          k: FLAME,
          x: b.x + (this.rand() - 0.5) * hw * 2,
          y: b.y - hh,
          vx: 0,
          vy: -30,
          life: 0.3,
          c: "",
          s: 1,
        });
      if (b.fuse < 0) {
        b.alive = false;
        const pal =
          b.type === "propane" || b.type === "cylinder"
            ? PAL_GAS
            : b.type === "canister"
              ? PAL_CHEM
              : b.type === "fuel" || b.type === "tanker"
                ? PAL_NAPALM
                : PAL_FIRE;
        this.explode(b.x, b.y - hh / 2, BARREL_BLAST[b.type], b.owner, true, pal);
        if (b.type === "ammo") this.burst(b.x, b.y - 6, 30, ["#ffe27a", "#ff9a2a", "#ffffff"]);
        if (b.type === "tanker")
          this.fires.push({ x: b.x, y: b.y, t: 6 }, { x: b.x - 20, y: b.y, t: 4 }, { x: b.x + 20, y: b.y, t: 4 });
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
        this.add({
          k: FLAME,
          x: s.x,
          y: s.y,
          vx: s.vx * 0.2,
          vy: -20,
          life: 0.25,
          c: "",
          s: 1 + Math.round(this.rand()),
        });
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
          this.loosen(c, r);
        }
        this.add({
          k: SPARK,
          x: s.x,
          y: s.y,
          vx: -s.vx * 0.1 + (this.rand() - 0.5) * 40,
          vy: -40 * this.rand(),
          life: 0.15,
          c: "#ffe27a",
          s: 1,
        });
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

  /** Check a shot against bodies; true when the shot is used up. */
  private hitBodies(s: Shot) {
    if (s.pid === -1) {
      for (const p of this.players) {
        if (p.dead || p.aboard || Math.abs(p.x - s.x) > 4 || s.y < p.y - PH || s.y > p.y) continue;
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
        // The gunship is armoured against small arms.
        this.hurtAlly(a, s.dmg * (a.kind === "gunship" ? 0.3 : 0.5));
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
      // A riot shield stops bullets and pellets from the front.
      if (e.kind === "shield" && (s.kind === BULLET || s.kind === PELLET) && sign(s.vx) === -e.facing) {
        for (let k = 0; k < 3; k++)
          this.add({
            k: SPARK,
            x: s.x,
            y: s.y,
            vx: -s.vx * 0.15 + (this.rand() - 0.5) * 60,
            vy: -30 - this.rand() * 50,
            life: 0.2,
            c: k ? "#ffe27a" : "#ffffff",
            s: 1,
          });
        return true;
      }
      this.hurtEnemy(e, s.dmg, s.pid, s.vx);
      this.add({
        k: SPARK,
        x: s.x,
        y: s.y,
        vx: -s.vx * 0.05,
        vy: -30,
        life: 0.12,
        c: e.kind === "crawler" || e.kind === "worm" ? "#8aff6a" : "#ffffff",
        s: 1,
      });
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
      if (this.rand() < 0.4 && this.mushrooms.filter((m) => m.t < 6).length < 2)
        this.mushrooms.push({ x, y, t: 0.3, s: 0.55 });
      this.waves.push({ x, y, t: 0.25 });
      this.flash = Math.max(this.flash, 0.4);
      this.shake = Math.max(this.shake, 14);
    }
    if (s.kind === NAPALM) {
      // A wall of fire along the ground
      const gy = surfaceY(this.level, x, y - 10);
      for (let k = -2; k <= 2; k++) this.fires.push({ x: x + k * 9, y: gy, t: 3 + this.rand() * 2 });
      for (let k = 0; k < 6; k++)
        this.add({
          k: CINDER,
          x,
          y: gy - 4,
          vx: (this.rand() - 0.5) * 220,
          vy: -60 - this.rand() * 120,
          life: 1,
          c: "",
          s: 2,
        });
    }
    const r = s.kind === ROCKET && s.pid === -1 ? 14 : (BLAST_R[s.kind] ?? 18);
    const pal =
      s.kind === BOLT || s.kind === PLASMA || s.kind === ELASER
        ? PAL_PLASMA
        : s.kind === NAPALM
          ? PAL_NAPALM
          : s.kind === FIREBALL || s.kind === METEOR
            ? PAL_MAGMA
            : s.kind === BIGBOMB
              ? PAL_NUKE
              : PAL_FIRE;
    // Only enemy fire and the level's artillery hurt the squad
    this.explode(x, y, r, s.pid, s.pid === -3 || s.pid === -1, pal);
  }

  /**
   * An explosion: craters and scorches the ground, hurts or flings everything in reach, sets off
   * barrels and props, and goes up in palette `pal`, one of PAL_*.
   */
  explode(x: number, y: number, r: number, pid: number, hurtsPlayers: boolean, pal = pid === -4 ? PAL_NUKE : PAL_FIRE) {
    const L = this.level;
    this.booms++;
    // Damage falls off with distance, so soft ground breaks at once and hard rock needs a big or
    // repeated blast. The visitor's clicks clear blocks outright.
    const dig = pid === -2 ? r : pid === -4 ? r * 0.7 : r * 0.62;
    for (const [c, row] of blastCells(L, x, y, dig)) {
      const i = cell(c, row);
      const kind = L.fg[i];
      if (pid === -2) {
        L.fg[i] = AIR;
        L.hp[i] = 0;
      } else {
        const d = Math.hypot(c * TILE + TILE / 2 - x, row * TILE + TILE / 2 - y);
        if (!damageCell(L, c, row, Math.max(1, Math.round((r / 7) * (1 - d / (dig + 4)))))) continue;
      }
      this.dirty.push(c, row);
      this.dirtyAround(c, row);
      if (this.rand() < (r > 40 ? 0.35 : 1)) this.debris(c, row, kind, r > 40 ? 1 : 3);
      this.loosen(c, row);
    }
    for (const [c, row] of blastCells(L, x, y, dig + 8)) {
      const i = cell(c, row);
      if (L.burn[i] < 3) {
        L.burn[i]++;
        this.dirty.push(c, row);
      }
    }
    // Traffic caught in it: wrecked, burning where it stood, going up a moment later
    for (const rd of this.roads)
      for (let i = rd.cars.length - 1; i >= 0; i--) {
        const car = rd.cars[i];
        if (Math.abs(car.x - x) > r + CAR_SIZE[car.kind] / 2 || Math.abs(rd.y - 6 - y) > r + 8) continue;
        rd.cars.splice(i, 1);
        this.fires.push({ x: car.x, y: rd.y, t: 4 + this.rand() * 4 });
        // It goes up a moment later and leaves a wreck.
        this.at(0.12 + this.rand() * 0.3, () => {
          this.explode(car.x, rd.y - 6, car.kind === "truck" || car.kind === "bus" ? 22 : 16, -3, false, PAL_NAPALM);
          L.props.push({
            kind: "sprite",
            id: car.kind === "bus" || car.kind === "truck" || car.kind === "firetruck" ? "buswreck" : "carwreck",
            x: car.x,
            y: rd.y,
            flip: car.dir < 0,
            back: false,
          });
        });
      }
    for (let i = this.carts.length - 1; i >= 0; i--) {
      const k = this.carts[i];
      if (Math.hypot(k.x - x, k.y - 5 - y) > r + 8) continue;
      this.carts.splice(i, 1);
      // In pieces, its load thrown everywhere
      for (let n = 0; n < 12; n++)
        this.add({
          k: DEBRIS,
          x: k.x + (this.rand() - 0.5) * 12,
          y: k.y - 5,
          vx: (this.rand() - 0.5) * 220,
          vy: -80 - this.rand() * 180,
          life: 1.4 + this.rand(),
          c: n % 2 ? "#6a5a4a" : CART_LOAD[k.load],
          s: this.rand() < 0.3 ? 2 : 1,
        });
    }
    this.smashProps(x, y, r);
    for (const e of this.enemies) {
      if (!e.alive) continue;
      const aim = this.aimPoint(e) ?? { x: e.x, y: e.y };
      const d = Math.hypot(aim.x - x, aim.y - y);
      const reach = r + (e.boss ? SIZE[e.kind][0] : 8 * e.scale);
      if (d > reach) continue;
      const f = 1 - d / reach;
      // Bosses, armour and big troopers take damage; the rest are killed and flung.
      if (e.boss || ARMOURED.has(e.kind) || e.scale > 1.4)
        this.hurtEnemy(e, ((e.boss ? 14 : 10) * f + 3) * Math.max(1, r / 22), pid, e.x - x);
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
      if (p.dead || p.aboard) continue;
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
        const reach = r + ALLY_SIZE[a.kind][0] * (a.kind === "gunship" ? 0.45 : 1);
        // The gunship is armoured: its blast damage is reduced and capped.
        if (d < reach) {
          const hurt = (8 * (1 - d / reach) + 2) * Math.max(1, r / 24);
          this.hurtAlly(a, a.kind === "gunship" ? Math.min(9, hurt * 0.4) : hurt);
        }
      }
    // Aircraft caught in it, mostly by flak: each side's fire brings down the other's
    for (const j of this.jets) {
      if (j.down || Math.hypot(j.x - x, j.y - y) > r + 14) continue;
      if (j.ally ? hurtsPlayers : pid === -5 || pid >= 0) {
        j.hp -= 1;
        if (j.hp <= 0) {
          j.down = true;
          this.popups.push({
            x: j.x,
            y: j.y - 10,
            text: j.ally ? "JET DOWN" : "+800",
            c: j.ally ? "#5ab0ff" : "#ffcc33",
            t: 0,
          });
        }
      }
    }
    this.blasts.push({ x, y, r: r * 0.9, t: 0, pal });
    this.fireball(x, y, r, pal);
    const big = r > 40;
    const cols = PAL_SPARKS[pal] ?? PAL_SPARKS[PAL_FIRE];
    // Sparks every which way, in the fire's own colours
    for (let n = big ? 46 : 12 + Math.round(r * 0.6); n > 0; n--) {
      const a = this.rand() * Math.PI * 2;
      const v = 80 + this.rand() * (big ? 360 : 210);
      this.add({
        k: SPARK,
        x,
        y,
        vx: Math.cos(a) * v,
        vy: Math.sin(a) * v - 60,
        life: 0.3 + this.rand() * 0.6,
        c: cols[n % cols.length],
        s: 1,
      });
    }
    // Glints: sparkles hanging in the air after it, twinkling as they drift down
    for (let n = big ? 22 : 3 + Math.floor(r / 4); n > 0; n--)
      this.add({
        k: GLINT,
        x: x + (this.rand() - 0.5) * r * 1.8,
        y: y + (this.rand() - 0.6) * r * 1.2,
        vx: (this.rand() - 0.5) * 40,
        vy: -30 - this.rand() * 50,
        life: 0.6 + this.rand() * 1,
        c: cols[Math.floor(this.rand() * cols.length)],
        s: this.rand() < 0.35 ? 2 : 1,
      });
    // Burning chunks thrown out in arcs, trailing flame and smoke
    for (let n = big ? 16 : 3 + Math.floor(r / 5); n > 0; n--) {
      const a = -Math.PI / 2 + (this.rand() - 0.5) * 2.4;
      const v = 90 + this.rand() * (big ? 280 : 160);
      this.add({
        k: CINDER,
        x,
        y,
        vx: Math.cos(a) * v,
        vy: Math.sin(a) * v,
        life: 0.9 + this.rand() * 1,
        c: cols[Math.floor(this.rand() * 3)],
        s: this.rand() < 0.3 ? 2 : 1,
      });
    }
    for (let n = 0; n < (big ? 14 : 6); n++)
      this.puff(x + (this.rand() - 0.5) * r, y + (this.rand() - 0.5) * r * 0.6, 2 + this.rand() * (big ? 8 : 3));
    // Bigger blasts: thick smoke, and a plume that keeps rising for a while
    if (r >= 22) {
      for (let n = Math.round(r / 3); n > 0; n--) {
        const a = this.rand() * Math.PI * 2;
        const d = this.rand() * r * 0.7;
        this.add(
          {
            k: this.rand() < 0.5 ? SMOKE : BIGSMOKE,
            x: x + Math.cos(a) * d,
            y: y + Math.sin(a) * d * 0.6,
            vx: Math.cos(a) * 18,
            vy: -16 - this.rand() * 26,
            life: 1.8 + this.rand() * 1.8,
            c: "",
            s: 3 + this.rand() * (r / 6),
          },
          true
        );
      }
      this.plumes.push({
        x,
        y: Math.min(y, groundOf(L, Math.floor(x / TILE)) * TILE),
        t: 2.5 + r / 10,
        s: Math.min(3, r / 28),
      });
      // Shrapnel streaking out, fast and far
      for (let n = Math.round(r / 4); n > 0; n--) {
        const a = this.rand() * Math.PI * 2;
        const v = 260 + this.rand() * 260;
        this.add({
          k: SPARK,
          x,
          y,
          vx: Math.cos(a) * v,
          vy: Math.sin(a) * v - 40,
          life: 0.25 + this.rand() * 0.3,
          c: "#fff6c8",
          s: 2,
        });
      }
    }
    // A ring of smoke rolling out of the bigger ones
    if (r >= 20)
      for (let n = 0, m = big ? 18 : 10; n < m; n++) {
        const a = (n / m) * Math.PI * 2;
        this.add({
          k: SMOKE,
          x: x + Math.cos(a) * r * 0.3,
          y: y + Math.sin(a) * r * 0.15,
          vx: Math.cos(a) * (40 + r),
          vy: Math.sin(a) * (12 + r * 0.3) - 10,
          life: 0.8 + this.rand() * 0.6,
          c: "",
          s: 2 + this.rand() * 2,
        });
      }
    // Little secondary bursts round it a moment after, as what's in it goes up
    if (r >= 16)
      for (let n = big ? 4 + Math.floor(this.rand() * 3) : 1 + Math.floor(this.rand() * 2); n > 0; n--) {
        const a = this.rand() * Math.PI * 2;
        const d = r * (0.4 + this.rand() * 0.6);
        const bx = x + Math.cos(a) * d;
        const by = y + Math.sin(a) * d * 0.7;
        this.at(0.06 + this.rand() * (big ? 0.5 : 0.25), () => {
          this.fireball(bx, by, r * (0.3 + this.rand() * 0.2), pal);
          for (let k = 0; k < 6; k++)
            this.add({
              k: SPARK,
              x: bx,
              y: by,
              vx: (this.rand() - 0.5) * 200,
              vy: -this.rand() * 160,
              life: 0.3 + this.rand() * 0.3,
              c: cols[k % cols.length],
              s: 1,
            });
          this.blasts.push({ x: bx, y: by, r: r * 0.3, t: 0.07, pal });
        });
      }
    const gy = groundOf(L, Math.floor(x / TILE)) * TILE;
    if (Math.abs(gy - y) < r + 8) {
      // Dust rolling out along the ground
      const dust = DUST_COLOR[this.theme.fill] ?? "#9a8a78";
      for (let n = big ? 18 : 7; n > 0; n--) {
        const dir = n % 2 ? 1 : -1;
        this.add({
          k: DUST,
          x: x + dir * this.rand() * 6,
          y: gy - 2 - this.rand() * 4,
          vx: dir * (40 + this.rand() * (big ? 170 : 100)),
          vy: -8 - this.rand() * 22,
          life: 0.8 + this.rand() * 0.7,
          c: dust,
          s: 2 + this.rand() * (big ? 4 : 2),
        });
      }
      if (this.rand() < 0.75) this.fires.push({ x, y: gy, t: 1.5 + this.rand() * 3 });
    }
    this.shake = Math.min(pid === -2 ? 6 : pid === -4 ? 12 : 3, this.shake + r * (pid === -2 ? 0.2 : 0.05));
  }

  /** Break props caught in a blast into pieces; explosive ones go off too. */
  private smashProps(x: number, y: number, r: number) {
    const props = this.level.props;
    for (let i = props.length - 1; i >= 0; i--) {
      const p = props[i];
      if (p.kind !== "sprite" || p.back) continue;
      const a = asset(p.id);
      if (Math.abs(p.x - x) > r + a.w / 2 || y < p.y - a.h - r || y > p.y + r) continue;
      props.splice(i, 1);
      for (let k = Math.min(14, 4 + ((a.w * a.h) >> 5)); k > 0; k--)
        this.add({
          k: DEBRIS,
          x: p.x + (this.rand() - 0.5) * a.w,
          y: p.y - this.rand() * a.h,
          vx: (this.rand() - 0.5) * 160,
          vy: -80 - this.rand() * 160,
          life: 1.2 + this.rand(),
          c: a.debris,
          s: this.rand() < 0.3 ? 2 : 1,
        });
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
      // Trail for the renderer
      const trail = (m.trail ??= []);
      trail.push(m.x, m.y);
      if (trail.length > 72) trail.splice(0, 2);
      // Smoke left hanging behind it, thicker for the big ones, and motor sparks
      const cls = m.owner ? (m.cls ?? 1) : 3;
      const ux = m.boost ? 0 : (m.hx ?? dx / d);
      const uy = m.boost ? -1 : (m.hy ?? dy / d);
      if (this.rand() < (cls >= 2 ? 0.6 : 0.25))
        this.add({
          k: cls >= 2 ? BIGSMOKE : SMOKE,
          x: m.x - ux * 14,
          y: m.y - uy * 14,
          vx: (this.rand() - 0.5) * 10,
          vy: (this.rand() - 0.5) * 10,
          life: 1.4 + cls * 0.7 + this.rand(),
          c: "",
          s: 1 + cls + this.rand() * 2,
        });
      if (this.rand() < 0.5)
        this.add({
          k: SPARK,
          x: m.x - ux * 10,
          y: m.y - uy * 10,
          vx: -ux * 60 + (this.rand() - 0.5) * 50,
          vy: -uy * 60 + (this.rand() - 0.5) * 50,
          life: 0.2 + this.rand() * 0.2,
          c: this.rand() < 0.5 ? "#ffe27a" : "#ff9a2a",
          s: 1,
        });
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
      // Curve in on the target, turning harder as it closes, then fly straight for the last 70 px.
      if (m.hx === undefined || m.hy === undefined) {
        // Start off to one side or the other.
        const a0 = Math.atan2(dy, dx) + (this.rand() < 0.5 ? -1 : 1) * (0.45 + this.rand() * 0.4) * (m.owner ? 0.6 : 1);
        m.hx = Math.cos(a0);
        m.hy = Math.sin(a0);
      }
      if (d < 70) {
        m.hx = dx / d;
        m.hy = dy / d;
      } else {
        let cur = Math.atan2(m.hy, m.hx);
        let da = Math.atan2(dy, dx) - cur;
        while (da > Math.PI) da -= Math.PI * 2;
        while (da < -Math.PI) da += Math.PI * 2;
        const turn = (0.5 + (1.1 * m.speed) / Math.max(40, d)) * dt;
        cur += clamp(da, -turn, turn);
        m.hx = Math.cos(cur);
        m.hy = Math.sin(cur);
      }
      if (d <= step) {
        if (m.owner) this.impact(m, m.tx, m.ty);
        else this.detonate(m);
        continue;
      }
      if (this.solid(m.x + m.hx * step, m.y + m.hy * step)) {
        // Hit something on the way in.
        if (m.owner) this.impact(m, m.x + m.hx * step, m.y + m.hy * step);
        else this.detonate(m, m.x + m.hx * step, m.y + m.hy * step);
        continue;
      }
      m.x += m.hx * step;
      m.y += m.hy * step;
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
    const k = this.intensity * (1 + 0.08 * Math.min(this.levelNo - 1, 6)) * WAVES[this.wave].k;
    // Deep in a cavern there's no sky: no shells, paratroopers, aircraft or weather.
    const buried = this.buried(view);
    const calm = this.phase === "nuke" || this.phase === "clear" || buried;
    if (buried && this.dropQueue.length) {
      // Down there the fallen come straight back in, near the squad
      const id = this.dropQueue.shift()!;
      this.release(
        this.players[id],
        clamp(this.advance + OFFSETS[id] - 30, view.x + 20, view.x + view.w - 20),
        view.y + 12
      );
    }
    if (buried) this.weather = 0;
    if (!calm && s.artillery > 0 && (T.artillery -= dt) <= 0) {
      T.artillery = (0.4 + r() * 1.2) / (s.artillery * k);
      this.shots.push({
        kind: SHELL,
        x: view.x + view.w * (0.4 + r() * 0.7),
        y: view.y - 10,
        vx: -15,
        vy: 260,
        life: 4,
        dmg: 0,
        pid: -3,
        pierce: 0,
      });
    }
    if (!buried && s.meteors > 0 && (T.meteor -= dt) <= 0) {
      T.meteor = (0.5 + r()) / (s.meteors * k);
      this.shots.push({
        kind: METEOR,
        x: view.x + view.w * (0.5 + r() * 0.8),
        y: view.y - 20,
        vx: -90,
        vy: 230,
        life: 4,
        dmg: 0,
        pid: -3,
        pierce: 0,
      });
    }
    if (!calm && s.paras > 0 && (T.para -= dt) <= 0) {
      T.para = (40 + r() * 40) / (s.paras * k * 1.8);
      // Reinforcements: a transport plane drops a stick of paratroopers ahead of the squad
      this.jets.push({
        x: view.x + view.w + 60,
        y: view.y + 14 + r() * 16,
        vx: -320,
        bombs: 3 + Math.floor(r() * 4),
        dropT: 0,
        kind: "transport",
        ally: false,
        fireCd: 0,
        hp: 6,
        down: false,
        vy: 0,
        dead: false,
      });
    }
    if (!buried && s.jets > 0 && (T.jet -= dt) <= 0) {
      T.jet = (3 + r() * 4) / Math.max(0.6, s.jets) / Math.sqrt(k);
      const kind = PLANE_MIX[Math.floor(r() * PLANE_MIX.length)];
      const ally = r() < 0.5;
      const spec = PLANES[kind];
      // Heavy and stealth bombers fly high; the rest cross the middle of the view.
      const y = spec.high ? view.y + 12 + r() * 18 : view.y + 24 + r() * view.h * 0.4;
      this.jets.push({
        x: ally ? view.x - 60 : view.x + view.w + 60,
        y,
        vx: (ally ? 1 : -1) * (spec.speed + r() * 40),
        bombs: calm ? 0 : spec.bombs + Math.floor(r() * (spec.more + 1)),
        dropT: 0.3,
        kind,
        ally,
        fireCd: 0.2,
        hp: spec.hp,
        down: false,
        vy: 0,
        dead: false,
        // Some bombers carry one huge bomb instead of a stick.
        big: kind === "bomber" && r() < 0.35,
        alt: y,
        phase: 0,
      });
    }
    // Players to drop back in: a transport comes over from behind
    if (!buried && this.dropQueue.length && !this.jets.some((j) => j.kind === "transport" && j.ally && !j.down)) {
      this.jets.push({
        x: view.x - 50,
        y: view.y + 16,
        vx: 380,
        bombs: 0,
        dropT: 0,
        kind: "transport",
        ally: true,
        fireCd: 0,
        hp: 6,
        down: false,
        vy: 0,
        dead: false,
      });
    }
    for (const j of this.jets) {
      j.x += j.vx * dt;
      if (!j.down) {
        // Climb over mountains rather than fly into them.
        const L = this.level;
        const top =
          Math.min(
            crustOf(L, Math.floor((j.x - 30) / TILE)),
            crustOf(L, Math.floor(j.x / TILE)),
            crustOf(L, Math.floor((j.x + 30) / TILE))
          ) *
            TILE -
          28;
        if (j.y > top) j.y = Math.max(top, j.y - 260 * dt);
      }
      if (j.kind === "transport" && !j.down) {
        j.dropT -= dt;
        if (j.ally) {
          // Over the squad: out they go, one after another
          const id = this.dropQueue[0];
          if (
            id !== undefined &&
            j.dropT <= 0 &&
            j.x >= Math.min(view.x + view.w * 0.6, Math.max(view.x + 20, this.advance + OFFSETS[id] - 30))
          ) {
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
          this.explode(
            j.x,
            Math.min(j.y, LEVEL_H - 8),
            j.kind === "bomber" || j.kind === "heavy" ? 40 : 28,
            j.ally ? -5 : -3,
            !j.ally
          );
        }
        continue;
      }
      j.dropT -= dt;
      j.fireCd -= dt;
      // Over its target area: the enemy's over the squad and the column, ours over the enemy ahead
      const over = j.ally
        ? j.x > this.advance + 70 && j.x < view.x + view.w * 0.96
        : j.x > this.advance - 120 && j.x < this.advance + 170;
      const pid = j.ally ? -5 : -3;
      if (j.kind === "dive") {
        this.diveBomb(j, dt, over, pid, view);
        continue;
      }
      if (!over) continue;
      if (j.kind === "heavy" && j.bombs > 0 && j.dropT <= 0) {
        // Carpet bombing: a long, close-spaced stick from high up
        j.bombs--;
        j.dropT = 0.06 + r() * 0.05;
        this.shots.push({
          kind: SHELL,
          x: j.x + (r() - 0.5) * 8,
          y: j.y + 5,
          vx: j.vx * 0.3,
          vy: 30,
          life: 5,
          dmg: 0,
          pid,
          pierce: 0,
        });
        continue;
      }
      if (j.kind === "stealth" && j.bombs > 0 && j.dropT <= 0) {
        // A few precise bombs, each released right over something worth hitting
        const dir = sign(j.vx);
        // Its bombs fall almost straight down, so release right over the target.
        const below = (tx: number) => Math.abs(tx - j.x) < 10;
        const hit = j.ally
          ? this.enemies.some((e) => e.alive && !FLYING.has(e.kind) && below(e.x))
          : this.players.some((p) => !p.dead && below(p.x)) ||
            this.allies.some((a) => !a.dead && !ALLY_AIR.has(a.kind) && below(a.x));
        if (hit) {
          j.bombs--;
          j.dropT = 0.35;
          this.shots.push({ kind: SHELL, x: j.x, y: j.y + 4, vx: dir * 4, vy: 60, life: 5, dmg: 0, pid, pierce: 0 });
        }
        continue;
      }
      if (j.kind === "rocketjet" && j.bombs > 0 && j.fireCd <= 0) {
        // A salvo of four rockets down at the ground ahead
        j.bombs--;
        j.fireCd = 0.9 + r() * 0.5;
        const dir = sign(j.vx);
        for (let n = 0; n < 4; n++)
          this.at(n * 0.07, () => {
            if (j.down) return;
            const a = Math.atan2(0.62, dir) + (this.rand() - 0.5) * 0.12;
            this.shots.push({
              kind: ROCKET,
              x: j.x,
              y: j.y + 3,
              vx: Math.cos(a) * 220,
              vy: Math.sin(a) * 220,
              life: 2.5,
              dmg: 0,
              pid: j.ally ? -5 : -1,
              pierce: 0,
            });
            this.flashes.push({ x: j.x, y: j.y + 3, t: 0.05, c: "#ffe27a" });
          });
        continue;
      }
      if (
        j.kind === "bomber" &&
        j.big &&
        j.bombs > 0 &&
        j.x * sign(j.vx) > (j.ally ? this.advance + 140 : this.advance + 30) * sign(j.vx)
      ) {
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
        this.shots.push({
          kind: j.ally ? BULLET : ENEMY_BULLET,
          x: j.x,
          y: j.y + 2,
          vx: Math.cos(a) * 420,
          vy: Math.sin(a) * 420,
          life: 0.9,
          dmg: 1,
          pid: j.ally ? -5 : -1,
          pierce: 0,
        });
        j.fireCd = 0.05;
        if (j.bombs > 0 && r() < 0.06) {
          j.bombs--;
          const tgt: Foe | Enemy | null = j.ally
            ? (this.enemies.find(
                (e) => e.alive && (ARMOURED.has(e.kind) || FLYING.has(e.kind)) && e.x > j.x && e.x < view.x + view.w
              ) ?? null)
            : (this.allies.find((q) => !q.dead) ?? this.players.find((q) => !q.dead) ?? null);
          if (tgt)
            this.missiles.push({
              x: j.x,
              y: j.y + 4,
              tx: tgt.x,
              ty: "boss" in tgt && FLYING.has(tgt.kind) ? tgt.y : surfaceY(this.level, tgt.x, 0),
              speed: 300,
              max: 640,
              up: false,
              owner: j.ally ? "ally" : "enemy",
              cls: 0,
            });
        }
      }
    }
    this.jets = this.jets.filter((j) => !j.dead && j.x > view.x - 120 && j.x < view.x + view.w + 160);
    if (!calm && s.helis > 0 && (T.heli -= dt) <= 0) {
      T.heli = (40 + r() * 40) / (s.helis * k * 2.2);
      if (
        this.enemies.filter(
          (e) => FLYING.has(e.kind) && !e.boss && e.kind !== "drone" && e.kind !== "jetpacker" && e.alive
        ).length < clamp(Math.round(view.w / 200), 3, 7)
      ) {
        const roll = r();
        const kind: EnemyKind =
          roll < 0.3 ? "heli" : roll < 0.55 ? "attackheli" : roll < 0.75 ? "scoutheli" : "transport";
        const h = this.makeEnemy(kind, view.x + view.w + 40, view.y + 40);
        h.hover = 30 + r() * 44;
        h.home = Math.floor(r() * 100);
        this.enemies.push(h);
      }
    }
    // Armour rolls in from ahead, or ships at sea
    if (!calm && (s.helis > 0 || s.jets > 0) && (T.armor -= dt) <= 0) {
      T.armor = (14 + r() * 14) / k;
      if (
        this.enemies.filter((e) => ARMOR_KINDS.has(e.kind) && e.alive).length < clamp(Math.round(view.w / 260), 3, 9) &&
        this.enemies.length < this.cap()
      ) {
        const x = view.x + view.w + 50;
        const gy = surfaceY(this.level, x, view.y - 60);
        const wet = kindAt(this.level, Math.floor(x / TILE), Math.floor(gy / TILE) - 1) === WATER;
        const kinds: EnemyKind[] = wet
          ? ["boat", "battleship", "submarine", "hovercraft"]
          : [
              "apc",
              "apc",
              "armoredcar",
              "armoredcar",
              "artytruck",
              "hovercraft",
              "lighttank",
              "medtank",
              "medtank",
              "heavytank",
              "flametank",
              "rockettank",
              "aatank",
              "launcher",
            ];
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
    // Allied missile strikes on the enemy ahead, from salvos of small rockets to the odd enormous one
    if (!calm && (T.strike -= dt) <= 0) {
      T.strike = (14 + r() * 14) / k;
      const ahead = this.enemies.filter(
        (e) => e.alive && !FLYING.has(e.kind) && e.x > this.advance + 80 && e.x < view.x + view.w - 10
      );
      const tgt = ahead.find((e) => ARMOURED.has(e.kind) || e.boss) ?? ahead[Math.floor(r() * ahead.length)];
      if (tgt) {
        const roll = r();
        let cls = roll < 0.45 ? 0 : roll < 0.75 ? 1 : roll < 0.93 ? 2 : 3;
        if (cls === 3 && this.players.some((p) => !p.dead && Math.abs(p.x - tgt.x) < 230)) cls = 2;
        const n = cls === 0 ? 5 : cls === 1 ? 2 : 1;
        for (let j = 0; j < n; j++)
          this.at(j * 0.22, () => {
            const tx = tgt.x + (n > 1 ? (r() - 0.5) * 60 : 0);
            this.missiles.push({
              x: view.x - 30 - r() * 60,
              y: view.y - 50 - r() * 30,
              tx,
              ty: surfaceY(this.level, tx, 0),
              speed: 200,
              max: [700, 640, 560, 480][cls],
              up: false,
              owner: "ally",
              cls,
            });
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
            this.missiles.push({
              x: view.x + view.w + 30 + r() * 60,
              y: view.y - 50 - r() * 30,
              tx,
              ty: surfaceY(this.level, tx, 0),
              speed: 200,
              max: [700, 640, 560, 480][cls],
              up: false,
              owner: "enemy",
              cls,
            });
          });
        if (cls >= 2)
          this.popups.push({
            x: tgt.x,
            y: tgt.y - 30,
            text: cls === 3 ? "HUGE ONE INBOUND" : "INCOMING",
            c: "#ff3b3b",
            t: 0,
          });
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
      this.drops.push({
        x: Math.max(this.advance + 30, view.x + view.w * (0.3 + r() * 0.4)),
        y: view.y - 10,
        vy: 0,
        weapon: r() < 0.2 ? -1 : Math.floor(r() * WEAPONS.length),
        ground: false,
        t: 0,
      });
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
    // With a day cycle, rain and storms follow the current weather.
    const amb = s.cycle
      ? this.sky === "rain"
        ? "rain"
        : this.sky === "storm"
          ? "storm"
          : s.ambient === "rain" || s.ambient === "storm"
            ? "none"
            : s.ambient
      : s.ambient;
    // Lightning in a storm, seconds apart so it never strobes
    if (amb === "storm" && (T.bolt -= dt) <= 0) {
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
    // Weather, in particles per second
    const rate = {
      none: 0,
      fireflies: 3,
      smoke: 4,
      embers: 14,
      ash: 28,
      snow: 34,
      rain: 60,
      storm: 90,
      sand: 34,
      spores: 10,
    }[amb];
    this.weather += buried ? 0 : rate * dt;
    for (; this.weather >= 1; this.weather--) {
      const x = view.x + r() * view.w;
      switch (amb) {
        case "fireflies":
          this.add({
            k: FIREFLY,
            x,
            y: view.y + view.h * (0.3 + r() * 0.4),
            vx: (r() - 0.5) * 8,
            vy: (r() - 0.5) * 6,
            life: 2 + r() * 2,
            c: "#d8ff7a",
            s: 1,
          });
          break;
        case "ash":
          this.add({
            k: ASHFALL,
            x,
            y: view.y - 4,
            vx: -10 - r() * 10,
            vy: 14 + r() * 12,
            life: 8,
            c: r() < 0.3 ? "#ff8a3a" : "#8a8088",
            s: 1,
          });
          break;
        case "embers":
          this.add({
            k: EMBER,
            x,
            y: view.y + view.h,
            vx: (r() - 0.5) * 10,
            vy: -20 - r() * 20,
            life: 4 + r() * 2,
            c: "",
            s: 1,
          });
          break;
        case "snow":
          this.add({
            k: SNOWFLAKE,
            x: x + 40,
            y: view.y - 4,
            vx: -12 - r() * 14,
            vy: 18 + r() * 16,
            life: 12,
            c: r() < 0.3 ? "#ffffff" : "#d8e4f4",
            s: r() < 0.2 ? 2 : 1,
          });
          break;
        case "rain":
        case "storm":
          this.add({
            k: RAIN,
            x: x + 60,
            y: view.y - 4,
            vx: amb === "storm" ? -110 : -60,
            vy: 320 + r() * 60,
            life: 1.4,
            c: "#8ab0d0",
            s: 1,
          });
          break;
        case "sand":
          this.add({
            k: SANDGRAIN,
            x: view.x + view.w + 4,
            y: view.y + r() * view.h,
            vx: -160 - r() * 80,
            vy: (r() - 0.5) * 20,
            life: 4,
            c: r() < 0.5 ? "#e8c890" : "#c8a060",
            s: 1,
          });
          break;
        case "spores":
          this.add({
            k: SPORE,
            x,
            y: view.y + view.h,
            vx: (r() - 0.5) * 10,
            vy: -10 - r() * 14,
            life: 8,
            c: r() < 0.5 ? "#8aff6a" : "#d88aff",
            s: 1,
          });
          break;
        default:
          this.add({ k: SMOKE, x, y: view.y + view.h * 0.7, vx: 4, vy: -6, life: 3, c: "", s: 3 + r() * 3 });
      }
    }
    // Burning wrecks, campfires and braziers; spray at the foot of waterfalls
    for (const p of this.level.props) {
      if (p.x < view.x - 20 || p.x > view.x + view.w + 20) continue;
      if ((p.kind === "fire" || p.kind === "brazier") && r() < 0.5)
        this.flame(p.x, p.kind === "brazier" ? p.y - 8 : p.y);
      else if (p.kind === "sprite" && !p.back && r() < 0.5) {
        const a = asset(p.id);
        if (a.flames)
          for (const [fx, fy] of a.flames) this.flame(p.x + (p.flip ? a.w / 2 - fx : fx - a.w / 2), p.y - a.h + fy);
        if (a.smoke && r() < 0.3)
          for (const [fx, fy] of a.smoke)
            this.puff(p.x + (p.flip ? a.w / 2 - fx : fx - a.w / 2), p.y - a.h + fy, 2 + r() * 2);
      } else if (p.kind === "waterfall" && r() < 0.7)
        this.add({
          k: MIST,
          x: p.x + r() * p.w,
          y: p.y + p.h - 2,
          vx: (r() - 0.5) * 16 - 4,
          vy: -8 - r() * 14,
          life: 1.2 + r(),
          c: "",
          s: 2 + r() * 2,
        });
    }
  }

  private updateEffects(dt: number, view: View) {
    // Smoke columns rising off the big blasts, thinning out as they go
    for (const pl of this.plumes) {
      pl.t -= dt;
      if (this.rand() < dt * 9 * pl.s)
        this.add({
          k: this.rand() < 0.6 ? SMOKE : BIGSMOKE,
          x: pl.x + (this.rand() - 0.5) * 12 * pl.s,
          y: pl.y - 6,
          vx: 6 + (this.rand() - 0.5) * 8,
          vy: -22 - this.rand() * 18,
          life: 2.6 + this.rand() * 2,
          c: "",
          s: 2 + this.rand() * 3 * pl.s,
        });
      if (this.rand() < dt * 3)
        this.add({
          k: EMBER,
          x: pl.x + (this.rand() - 0.5) * 10,
          y: pl.y - 4,
          vx: (this.rand() - 0.5) * 16,
          vy: -30 - this.rand() * 30,
          life: 1.5,
          c: "",
          s: 1,
        });
    }
    this.plumes = this.plumes.filter((pl) => pl.t > 0 && pl.x > view.x - 100);
    for (const f of this.fires) {
      f.t -= dt;
      if (this.rand() < 0.6) this.flame(f.x + (this.rand() - 0.5) * 10, f.y);
    }
    this.fires = this.fires.filter((f) => f.t > 0 && f.x > view.x - 40);
    for (const b of this.blasts) b.t += dt;
    this.blasts = this.blasts.filter((b) => b.t < 0.5);
    for (const m of this.mushrooms) {
      m.t += dt;
      // The renderer draws the cloud; add smoke drifting off it and embers falling out.
      const s = m.s ?? 1;
      if (m.t < 4 && this.rand() < 0.35)
        this.add({
          k: BIGSMOKE,
          x: m.x + (this.rand() - 0.5) * 50 * s,
          y: m.y - this.rand() * Math.min(1, m.t / 2) * 140 * s,
          vx: (this.rand() - 0.5) * 14,
          vy: -8,
          life: 2 + this.rand() * 2,
          c: "",
          s: 3 + this.rand() * 4,
        });
      if (m.t > 0.6 && m.t < 5 && this.rand() < 0.5)
        this.add({
          k: EMBER,
          x: m.x + (this.rand() - 0.5) * 80 * s,
          y: m.y - (60 + this.rand() * 80) * s * Math.min(1, m.t / 2.5),
          vx: (this.rand() - 0.5) * 20,
          vy: 20 + this.rand() * 30,
          life: 1.5 + this.rand(),
          c: "",
          s: 1,
        });
    }
    this.mushrooms = this.mushrooms.filter((m) => m.t < 9);
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
          // Now and then a landing cinder starts a small fire.
          if (p.life > p.max * 0.4 && Math.abs(p.vy) > 60 && this.rand() < 0.06 && this.fires.length < 60)
            this.fires.push({ x: p.x, y: Math.floor(ny / TILE) * TILE, t: 1 + this.rand() * 2 });
          p.vy *= -0.35;
          p.vx *= 0.55;
        } else {
          p.x = nx;
          p.y = ny;
        }
        if (this.rand() < 0.45)
          this.add({ k: FLAME, x: p.x, y: p.y, vx: 0, vy: -8, life: 0.2 + this.rand() * 0.15, c: "", s: 1 });
        // Smoke trail while it's still hot
        if (p.life > p.max * 0.4 && this.rand() < 0.22)
          this.add({ k: SMOKE, x: p.x, y: p.y, vx: 0, vy: -6, life: 0.6 + this.rand() * 0.4, c: "", s: 1 });
      } else {
        if (p.k === SPARK) p.vy += 160 * dt;
        else if (p.k === GLINT) {
          p.vx *= Math.exp(-dt * 2.4);
          p.vy = p.vy * Math.exp(-dt * 2) + 34 * dt;
        } else if (p.k === BLAZE) {
          const k = Math.exp(-dt * 3.2);
          p.vx *= k;
          p.vy = p.vy * k - 26 * dt;
        } else if (p.k === DUST) {
          p.vx *= Math.exp(-dt * 2.6);
          p.vy += 6 * dt;
        } else if (p.k === FIREFLY || p.k === SPORE) {
          p.vx += (this.rand() - 0.5) * 20 * dt;
          p.vy += (this.rand() - 0.5) * 20 * dt;
        } else if (p.k === SNOWFLAKE) p.vx += Math.sin(this.t * 2 + p.y * 0.05) * 6 * dt;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        if ((p.k === SNOWFLAKE || p.k === RAIN || p.k === ASHFALL) && this.solid(p.x, p.y)) p.life = 0;
        // Bubbles wobble up and pop at the surface.
        if (p.k === BUBBLE) {
          p.vx += (this.rand() - 0.5) * 30 * dt;
          if (!waterAt(this.level, p.x, p.y)) p.life = 0;
        }
      }
    }
  }

  /** Add a particle if the budget allows; `force` skips the check, for explosion particles. */
  private add(p: Omit<Particle, "max">, force = false) {
    if (force || this.parts.length < this.maxParticles) this.parts.push({ ...p, max: p.life });
  }

  /** A cluster of BLAZE particles, bigger in the middle, that the renderer draws as one outlined blob. */
  private fireball(x: number, y: number, r: number, pal = PAL_FIRE) {
    const balls = Math.min(22, 4 + Math.round(r / 2.5));
    const slow = r > 40 ? 1.7 : 1;
    const reach = r * 0.55 + 1;
    for (let n = 0; n < balls; n++) {
      const a = this.rand() * Math.PI * 2;
      const d = Math.sqrt(this.rand()) * r * 0.55;
      const ux = Math.cos(a);
      const uy = Math.sin(a);
      const core = 1 - d / reach;
      this.add(
        {
          k: BLAZE,
          x: x + ux * d,
          y: y + uy * d * 0.8,
          vx: ux * (10 + d * 1.8),
          vy: uy * d * 1.2 - 18 - this.rand() * 28,
          life: (0.4 + this.rand() * 0.45 + core * 0.25) * slow,
          c: String(pal),
          s: Math.max(2, r * (0.18 + this.rand() * 0.2) * (0.65 + core * 0.5)),
        },
        true
      );
    }
  }

  private puff(x: number, y: number, s: number) {
    this.add({
      k: SMOKE,
      x,
      y,
      vx: (this.rand() - 0.5) * 16,
      vy: -8 - this.rand() * 14,
      life: 1 + this.rand(),
      c: "",
      s,
    });
  }

  private flame(x: number, y: number) {
    this.add({
      k: FLAME,
      x: x + (this.rand() - 0.5) * 8,
      y,
      vx: (this.rand() - 0.5) * 10,
      vy: -26 - this.rand() * 20,
      life: 0.4 + this.rand() * 0.3,
      c: "",
      s: 1 + Math.round(this.rand()),
    });
  }

  private burst(x: number, y: number, n: number, colors: string[]) {
    for (let k = 0; k < n; k++)
      this.add({
        k: SPARK,
        x,
        y,
        vx: (this.rand() - 0.5) * 140,
        vy: -this.rand() * 140,
        life: 0.5,
        c: colors[k % colors.length],
        s: 1,
      });
  }

  private debris(c: number, r: number, kind: number, n: number) {
    for (let k = 0; k < n; k++)
      this.add({
        k: DEBRIS,
        x: c * TILE + this.rand() * TILE,
        y: r * TILE + this.rand() * TILE,
        vx: (this.rand() - 0.5) * 140,
        vy: -60 - this.rand() * 140,
        life: 1.2 + this.rand(),
        c: String(kind),
        s: this.rand() < 0.15 ? 3 : this.rand() < 0.4 ? 2 : 1,
      });
  }

  private dirtyAround(c: number, r: number) {
    this.dirty.push(c - 1, r, c + 1, r, c, r - 1, c, r + 1);
  }

  /** The block at (c, r) just broke: let loose ground around it settle, and drop any small piece left floating. */
  private loosen(c: number, r: number) {
    this.unstable.push(c, r - 1, c - 1, r - 1, c + 1, r - 1, c - 1, r, c + 1, r);
    for (const [nc, nr] of [
      [c - 1, r],
      [c + 1, r],
      [c, r - 1],
      [c, r + 1],
    ])
      this.cutOff(nc, nr);
  }

  /** If the block at (c, r) is part of a small piece of the land touching nothing else, set the whole piece falling. */
  private cutOff(c: number, r: number) {
    const L = this.level;
    if (r < 0 || r >= ROWS - 3 || c < L.start || c >= L.next || !SOLID[L.fg[cell(c, r)]]) return;
    const seen = this.seen;
    const mark = ++this.stamp;
    const piece: number[] = [];
    const stack = [c, r];
    seen[cell(c, r)] = mark;
    while (stack.length) {
      const y = stack.pop()!;
      const x = stack.pop()!;
      const i = cell(x, y);
      // Anchored: too big, at bedrock, unbreakable, or at the edge of the stored land.
      if (piece.length > 90 || y >= ROWS - 3 || TILE_HP[L.fg[i]] === 255 || x <= L.start || x >= L.next - 1) return;
      piece.push(x, y);
      for (const [nx, ny] of [
        [x - 1, y],
        [x + 1, y],
        [x, y - 1],
        [x, y + 1],
      ]) {
        if (ny < 0) continue;
        const j = cell(nx, ny);
        if (seen[j] === mark || !SOLID[L.fg[j]]) continue;
        seen[j] = mark;
        stack.push(nx, ny);
      }
    }
    // Bottom row first, so each block has somewhere to fall.
    const order: [number, number][] = [];
    for (let k = 0; k < piece.length; k += 2) order.push([piece[k], piece[k + 1]]);
    order.sort((a, b) => b[1] - a[1]);
    for (const [x, y] of order) {
      this.crumbling.add(cell(x, y));
      this.unstable.push(x, y);
    }
  }

  /** Whether a player occupies block (c, r); falling ground stops short of them. */
  private occupied(c: number, r: number) {
    for (const p of this.players) {
      if (p.dead || p.aboard || Math.floor(p.x / TILE) !== c) continue;
      if (r >= Math.floor((p.y - PH) / TILE) && r <= Math.floor((p.y - 1) / TILE)) return true;
    }
    return false;
  }

  /**
   * Move falling ground one row per step: loose blocks drop or slide off slopes, cut-off pieces drop
   * whatever they're made of, and whatever rested on a moved block is checked next.
   */
  private settle() {
    if (!this.unstable.length) return;
    const L = this.level;
    const todo = this.unstable;
    this.unstable = [];
    let moves = 0;
    for (let k = 0; k < todo.length; k += 2) {
      const c = todo[k];
      const r = todo[k + 1];
      if (r < 0 || r >= ROWS - 3 || c <= L.start || c >= L.next - 1) continue;
      const at = cell(c, r);
      const kind = L.fg[at];
      const falling = this.crumbling.has(at);
      if (!kind || !SOLID[kind] || (!LOOSE[kind] && !falling)) {
        if (falling) this.crumbling.delete(at);
        continue;
      }
      if (moves >= 260) {
        this.unstable.push(c, r);
        continue;
      }
      // Falls through air and soft blocks like leaves and vines, not into water, lava or ladders.
      const open = (i: number) => {
        const k = L.fg[i];
        return k === AIR || (!SOLID[k] && k !== WATER && k !== LADDER && TILE_HP[k] !== 255);
      };
      let to = -1;
      if (open(cell(c, r + 1))) to = c;
      else if (LOOSE[kind] && !falling) {
        const d = this.rand() < 0.5 ? 1 : -1;
        for (const dir of [d, -d])
          if (open(cell(c + dir, r)) && open(cell(c + dir, r + 1)) && c + dir > L.start && c + dir < L.next - 1) {
            to = c + dir;
            break;
          }
      }
      if (to < 0) {
        // Landed
        if (falling) {
          this.crumbling.delete(at);
          if (this.rand() < 0.5)
            this.add({
              k: DUST,
              x: c * TILE + 4,
              y: r * TILE + 6,
              vx: (this.rand() - 0.5) * 30,
              vy: -6,
              life: 0.6,
              c: DUST_COLOR[this.theme.fill] ?? "#9a8a78",
              s: 2,
            });
        }
        continue;
      }
      if (this.occupied(to, r + 1)) {
        this.unstable.push(c, r);
        continue;
      }
      const dst = cell(to, r + 1);
      L.fg[dst] = kind;
      L.hp[dst] = L.hp[at];
      L.burn[dst] = L.burn[at];
      L.fg[at] = AIR;
      L.hp[at] = 0;
      L.burn[at] = 0;
      if (falling) {
        this.crumbling.delete(at);
        this.crumbling.add(dst);
      }
      this.dirty.push(c, r, to, r + 1);
      this.dirtyAround(c, r);
      this.dirtyAround(to, r + 1);
      moves++;
      // Keep it falling, and check what was resting on it.
      this.unstable.push(to, r + 1, c, r - 1, c - 1, r - 1, c + 1, r - 1);
      if (this.rand() < 0.08) this.debris(c, r, kind, 1);
    }
  }
}

/** The theme for `?level=` or the 404's cave, else a random one. */
export function startTheme(key: string | null, rand: () => number) {
  return (key && THEMES[key]) || nextTheme(rand);
}
