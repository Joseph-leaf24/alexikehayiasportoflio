/**
 * Level generator for the pixel run-and-gun: a side-scrolling grid of 8×8 blocks built a feature at
 * a time ahead of the camera from seeded noise. Columns live in a ring buffer, so memory stays fixed
 * however long a level runs. Pure data: game.ts plays it and PixelScene draws it.
 */
import { int, rng } from "./rng";
import { fbm1, noise2, ridged1, smoothstep } from "./noise";
import { asset, pickAsset, type AssetId } from "./assets";
import { buildingsFor, SPEC_BY_KEY, type BuildingSpec, type Material } from "./buildings";
import type { BossKind, Theme, Troop } from "./themes";

/** Edge of one block, in scene pixels. */
export const TILE = 8;
/** Rows of blocks, with room above BASE_ROW for the tallest mountains and below it for deep caverns. */
export const ROWS = 240;
/** Columns kept in memory; a power of two so `c & MASK` indexes the ring. */
export const RING = 512;
export const MASK = RING - 1;
/** Ground row of the flat lowlands and the landing zone. */
export const BASE_ROW = 148;
/** Limits of terrainAt's hills and valleys; set-piece mountains go far higher. */
export const PEAK_ROW = BASE_ROW - 38;
export const VALLEY_ROW = BASE_ROW + 6;
export const LEVEL_H = ROWS * TILE;

/* Block kinds. The back-wall grid uses the same ids. */
export const AIR = 0;
export const GRASS = 1;
export const DIRT = 2;
export const STONE = 3;
export const BEDROCK = 4;
export const BRICK = 5;
export const CONCRETE = 6;
export const CRATE = 7;
export const SANDBAG = 8;
export const WOOD = 9;
export const STEEL = 10;
export const WINDOW = 11;
export const LEAF = 12;
export const TRUNK = 13;
export const CRYSTAL = 14;
export const ASH = 15;
export const ROCK = 16;
export const METAL = 17;
export const LAVA = 18;
export const HAZARD = 19;
export const PIPE = 20;
export const SAND = 21;
export const SANDSTONE = 22;
export const SNOW = 23;
export const ICE = 24;
export const PINE = 25;
export const HIVE = 26;
export const SLIME = 27;
export const MUD = 28;
export const REDROCK = 29;
// Jungle temple stone, vines, water and bamboo
export const MOSS = 30;
export const VINE = 31;
export const WATER = 32;
export const BAMBOO = 33;
// Ladders, chain-link fencing and corrugated tin roofs
export const LADDER = 34;
export const FENCE = 35;
export const TIN = 36;
// Building materials, roads, and underground ore, rails and sewers
export const GLASS = 37;
export const MARBLE = 38;
export const PLASTER = 39;
export const SIDING = 40;
export const STUCCO = 41;
export const TILES = 42;
export const DARKBRICK = 43;
export const ASPHALT = 44;
export const PAVEMENT = 45;
export const COAL = 46;
export const IRON = 47;
export const GOLD = 48;
export const RAIL = 49;
export const SEWER = 50;
export const PARQUET = 51;
export const CARPET = 52;
// Traditional architecture: half-timbering, adobe, roof tiles, glazed green tiles, slate, paper screens, painted tiles
export const TIMBER = 53;
export const ADOBE = 54;
export const TERRACOTTA = 55;
export const JADE = 56;
export const SLATE = 57;
export const SHOJI = 58;
export const AZULEJO = 59;
export const LIMESTONE = 60;
// Soils and rocks, then autumn and blossom leaves
export const CLAY = 61;
export const GRAVEL = 62;
export const GRANITE = 63;
export const BASALT = 64;
export const LOAM = 65;
export const CHALK = 66;
export const REDLEAF = 67;
export const PINKLEAF = 68;
// Regional rocks, soils and ores; MAGMA glows in its cracks and MYCELIUM is the glowing cave fungus
export const SHALE = 69;
export const QUARTZ = 70;
export const OBSIDIAN = 71;
export const MAGMA = 72;
export const SULFUR = 73;
export const PERMAFROST = 74;
export const GLACIER = 75;
export const SCREE = 76;
export const PEAT = 77;
export const LATERITE = 78;
export const SALT = 79;
export const CORAL = 80;
export const COBBLE = 81;
export const FOSSIL = 82;
export const COPPER = 83;
export const AMETHYST = 84;
export const MYCELIUM = 85;
export const MOSSROCK = 86;
export const DRYGRASS = 87;
export const ROOTS = 88;
export const TUFF = 89;
export const DIAMOND = 90;
/** Reinforced concrete for hardened buildings, bridge decks and overpasses; nearly unbreakable. */
export const REINFORCED = 91;
export const KINDS = 92;

/** 1 for block kinds that stop people and bullets; foliage, trunks, liquids, ladders, fences and rails don't. */
export const SOLID = new Uint8Array(KINDS).map((_, k) =>
  k === AIR ||
  k === LEAF ||
  k === TRUNK ||
  k === LAVA ||
  k === PINE ||
  k === VINE ||
  k === WATER ||
  k === BAMBOO ||
  k === LADDER ||
  k === FENCE ||
  k === RAIL ||
  k === REDLEAF ||
  k === PINKLEAF
    ? 0
    : 1
);
/** Hits each block kind takes before it breaks. 255 never breaks, so blasts can't drain water or lava. */
export const TILE_HP = Uint8Array.from([
  0, 2, 2, 4, 255, 3, 5, 1, 2, 1, 6, 1, 1, 1, 4, 2, 5, 6, 255, 3, 3, 1, 3, 1, 3, 1, 3, 2, 2, 4, 5, 1, 255, 1, 1, 1, 3,
  1, 5, 2, 2, 3, 3, 4, 4, 3, 4, 5, 5, 1, 4, 2, 2, 3, 3, 2, 3, 3, 1, 3, 4, 2, 1, 9, 8, 2, 2, 1, 1,
  // shale, quartz, obsidian, magma, sulphur, permafrost, glacier, scree, peat, laterite, salt
  3, 5, 7, 4, 2, 3, 4, 1, 2, 2, 2,
  // coral, cobble, fossil, copper, amethyst, mycelium, moss rock, dry grass, roots, tuff, diamond
  2, 4, 4, 5, 4, 2, 4, 2, 2, 2, 6,
  // reinforced concrete
  32,
]);
/** 1 for loose ground that slides down into a crater dug beneath it. */
export const LOOSE = new Uint8Array(KINDS).map((_, k) =>
  k === SAND || k === GRAVEL || k === SNOW || k === SCREE || k === ASH || k === SALT ? 1 : 0
);

/** An entity for game.ts to create, at world-px x and feet y; a boat's y is the water line. */
export interface Spawn {
  kind:
    | Troop
    | "mortar"
    | "mg"
    | "turret"
    | "jeep"
    | "truck"
    | "boat"
    | "barrel"
    | "fuel"
    | "tanker"
    | "ammo"
    | "egg"
    | "propane"
    | "canister"
    | "cylinder"
    | "battleship"
    | "apc"
    | "lighttank"
    | "mine"
    | "medtank"
    | "heavytank"
    | "flametank"
    | "rockettank"
    | "aatank"
    | "launcher"
    | "manhole"
    | "tnt"
    | "submarine"
    | "flakgun"
    | "howitzer"
    | "samsite"
    | "rocketpod"
    | "train"
    | "armoredcar"
    | "artytruck"
    | "hovercraft"
    | "minecart"
    /** `extra`: the tunnel's two ends, then each station's x, in world px */
    | "metro"
    /** `extra`: the line's two ends in world px, optionally followed by a station's x and 1 for passenger trains */
    | "freight"
    /** `extra`: the road's ends in world px, the traffic light count, each light's x, then each bus stop's x */
    | "road"
    /** Like road, but an expressway with faster traffic and no lights or stops */
    | "motorway";
  x: number;
  y: number;
  extra?: number[];
}

/** Common city buildings, picked more often than the rest of the theme's list. */
const CITY_ESSENTIAL = [
  "hospital",
  "gasstation",
  "police",
  "firestation",
  "school",
  "supermarket",
  "apartments",
  "officetower",
  "bank",
  "diner",
  "cinema",
  "hotel",
  "parkinggarage",
  "pharmacy",
  "postoffice",
  "library",
  "skyscraper",
  "pizzeria",
  "convenience",
  "tenement",
  "brownstone",
  "decotower",
  "neontenement",
  "busdepot",
  "garage",
  "clinic",
  "townhouse",
  "haussmann",
  "cafe",
  "gym",
  "bakery",
  "pub",
  "trainstation",
  "stockexchange",
  "museum",
];
/** Buildings always built of reinforced concrete. */
const HARDENED = new Set([
  "bank",
  "police",
  "stockexchange",
  "datacentre",
  "commandhq",
  "armory",
  "bunkerhq",
  "embassy",
  "courthouse",
  "powerstation",
  "reactor",
  "brutalist",
  "townhall",
  "radarstation",
]);
/** Street furniture and vehicles placed between city buildings. */
const STREET_KIT = [
  "trafficlight",
  "busstop",
  "bench",
  "trashbin",
  "hydrant",
  "treeplanter",
  "streetlamp",
  "newsbox",
  "mailbox",
  "dumpster",
  "sedan",
  "taxi",
  "van",
  "policecar",
  "ambulance",
  "newsstand",
  "foodtruck",
  "kiosk",
  "vending",
  "metrosign",
];
/** Building keys for industrial works. */
const WORKS = [
  "factory",
  "warehouse",
  "brewery",
  "printworks",
  "powerstation",
  "waterworks",
  "recycling",
  "refinery",
  "chemplant",
  "sawmill",
];
/** Metro and railway station names. */
const STATIONS = [
  "CENTRAL",
  "MARKET",
  "HARBOUR",
  "PARK",
  "UNION",
  "CIVIC",
  "DOCKS",
  "OLD TOWN",
  "MUSEUM",
  "NORTH",
  "BRIDGE",
  "STADIUM",
  "GARDENS",
  "TOWER",
  "RIVER",
  "ARSENAL",
];

/** A static decoration drawn by PixelScene, positioned in world px. */
type Prop =
  | { kind: "legs"; x: number; y: number; w: number; h: number }
  | { kind: "support"; x: number; y: number; h: number }
  | { kind: "searchlight"; x: number; y: number }
  | { kind: "alarm"; x: number; y: number }
  | { kind: "fire"; x: number; y: number }
  | { kind: "flag"; x: number; y: number }
  | { kind: "derrick"; x: number; y: number }
  | { kind: "radar"; x: number; y: number }
  | { kind: "silo"; x: number; y: number }
  | { kind: "antenna"; x: number; y: number; h: number }
  | { kind: "lamp"; x: number; y: number }
  | { kind: "wire"; x: number; y: number; w: number }
  /** Falling water from x, y, `w` px wide and `h` px tall */
  | { kind: "waterfall"; x: number; y: number; w: number; h: number }
  | { kind: "tent"; x: number; y: number }
  /** Rope handrails `w` px long over a bridge deck at y */
  | { kind: "rope"; x: number; y: number; w: number }
  | { kind: "brazier"; x: number; y: number }
  /** An assets.ts sprite centred on x with its feet at y; `back` draws it in shadow behind the blocks */
  | { kind: "sprite"; id: AssetId; x: number; y: number; flip: boolean; back: boolean }
  /** A building's sign centred on x with its bottom at y; neon signs glow */
  | { kind: "sign"; x: number; y: number; text: string; ink: string; board: string; neon: boolean }
  /** A striped awning sticking out left of x */
  | { kind: "awning"; x: number; y: number; w: number; color: string }
  /** Traffic lights at a crossing; road traffic stops at red */
  | { kind: "stoplight"; x: number; y: number }
  | { kind: "streetlight"; x: number; y: number }
  /** A suspension bridge cable from x, y to x2, y2, sagging `sag` px, with hangers down to the deck at y = `deck` */
  | { kind: "cable"; x: number; y: number; x2: number; y2: number; sag: number; deck: number };

/** Where the current stretch's boss arena and stronghold go. */
interface Plan {
  /** Columns before the stronghold */
  length: number;
  boss: { col: number; kind: BossKind } | null;
  bossPlaced: boolean;
  /** First column of the stronghold */
  stronghold: number;
  fortPlaced: boolean;
  /** World-px x positions the ICBMs aim at, set when the stronghold is placed */
  targets: number[];
}

/** A level's block grids, stored in the column ring, and the generator's state. */
export interface Level {
  fg: Uint8Array;
  bg: Uint8Array;
  /** Hits left before a block breaks */
  hp: Uint8Array;
  /** Scorch level 0–3 */
  burn: Uint8Array;
  /** The stretch being generated, counting from 1 */
  segment: number;
  /** Every `every`th stretch ends in a stronghold and finale */
  every: number;
  /** Never end a stretch in a stronghold */
  endless: boolean;
  /** End the current stretch in a stronghold whatever `every` says */
  forceFort: boolean;
  /** Biome changes: the border column and the theme and boss that follow it */
  borders: { col: number; theme: Theme; boss: BossKind | null }[];
  /** Boss arenas placed but not yet reached */
  bossQueue: { col: number; kind: BossKind }[];
  /** The last feature generated was open sea with no shore yet */
  openSea: boolean;
  /** Picks the next biome; set by the game. Without it a level keeps its theme. */
  pickTheme?: (current: Theme) => Theme;
  /** Walkable ground row of each stored column; in a cavern, its floor */
  ground: Int16Array;
  /** Surface row open to the sky in each stored column; PixelScene darkens blocks by depth below it */
  crust: Int16Array;
  /** Oldest stored column and the next one to generate, as absolute column numbers */
  start: number;
  next: number;
  /** The generator's current ground row */
  h: number;
  /** Noise seed for the hills, rock and caves */
  seed: number;
  theme: Theme;
  plan: Plan;
  /** Enemy count multiplier, set by the game */
  intensity: number;
  rand: () => number;
  props: Prop[];
}

/** Index into the stored grids for an absolute column and row. */
export const cell = (c: number, r: number) => r * RING + (c & MASK);

export interface LevelOptions {
  /** Force this boss, or null for none, instead of rolling one */
  boss?: BossKind | null;
  length?: number;
  /** End the first stretch at a stronghold whatever `every` says */
  stronghold?: boolean;
  /** Stretches per stronghold and finale; default 20 */
  every?: number;
  /** Never place a stronghold; biomes run on forever */
  endless?: boolean;
}

/** A new, empty level in `theme`; extend generates its columns. */
export function createLevel(theme: Theme, seed = 1, opts: LevelOptions = {}): Level {
  const rand = rng(seed);
  const length = opts.length ?? Math.round((240 + int(rand, 160)) * (theme.long ?? 1));
  const kind =
    opts.boss !== undefined
      ? opts.boss
      : theme.bosses.length && rand() < 0.8
        ? theme.bosses[int(rand, theme.bosses.length)]
        : null;
  return {
    fg: new Uint8Array(RING * ROWS),
    bg: new Uint8Array(RING * ROWS),
    hp: new Uint8Array(RING * ROWS),
    burn: new Uint8Array(RING * ROWS),
    ground: new Int16Array(RING).fill(BASE_ROW),
    crust: new Int16Array(RING).fill(BASE_ROW),
    start: 0,
    next: 0,
    h: BASE_ROW,
    seed: Math.floor(rand() * 1e9),
    theme,
    plan: {
      length,
      boss: kind ? { col: Math.floor(length * 0.48), kind } : null,
      bossPlaced: false,
      stronghold: length,
      fortPlaced: false,
      targets: [],
    },
    segment: 1,
    every: opts.every ?? 20,
    endless: !!opts.endless,
    forceFort: !!opts.stronghold,
    borders: [],
    bossQueue: [],
    openSea: false,
    intensity: 1,
    rand,
    props: [],
  };
}

/** The block at an absolute column and row: AIR outside the stored columns, BEDROCK below the grid. */
export function kindAt(L: Level, c: number, r: number) {
  if (r < 0 || c < L.start || c >= L.next) return AIR;
  if (r >= ROWS) return BEDROCK;
  return L.fg[cell(c, r)];
}

/** Whether the block at a world-pixel position is solid. */
export const solidAt = (L: Level, x: number, y: number) => {
  const r = Math.floor(y / TILE);
  if (r >= ROWS) return true;
  return SOLID[kindAt(L, Math.floor(x / TILE), r)] === 1;
};

export const lavaAt = (L: Level, x: number, y: number) =>
  kindAt(L, Math.floor(x / TILE), Math.floor(y / TILE)) === LAVA;
export const waterAt = (L: Level, x: number, y: number) =>
  kindAt(L, Math.floor(x / TILE), Math.floor(y / TILE)) === WATER;

/** Ground row of a column, or BASE_ROW if it isn't stored. */
export const groundOf = (L: Level, c: number) => (c >= L.start && c < L.next ? L.ground[c & MASK] : BASE_ROW);
/** Surface row open to the sky of a column, or BASE_ROW if it isn't stored. */
export const crustOf = (L: Level, c: number) => (c >= L.start && c < L.next ? L.crust[c & MASK] : BASE_ROW);

/** World-pixel y of the first solid block at or below `fromY` in a column. */
export function surfaceY(L: Level, x: number, fromY = 0) {
  for (let r = Math.max(0, Math.floor(fromY / TILE)); r < ROWS; r++) if (solidAt(L, x, r * TILE + 1)) return r * TILE;
  return LEVEL_H;
}

/** Whether the straight line between two points is clear of solid blocks, sampled every 4 px. */
export function lineOfSight(L: Level, x0: number, y0: number, x1: number, y1: number) {
  const steps = Math.ceil(Math.hypot(x1 - x0, y1 - y0) / 4);
  for (let k = 1; k < steps; k++) {
    const u = k / steps;
    if (solidAt(L, x0 + (x1 - x0) * u, y0 + (y1 - y0) * u)) return false;
  }
  return true;
}

/** Breakable blocks whose centre lies within `radius` of (x, y), as [column, row] pairs. */
export function blastCells(L: Level, x: number, y: number, radius: number): [number, number][] {
  const out: [number, number][] = [];
  const c0 = Math.max(L.start, Math.floor((x - radius) / TILE));
  const c1 = Math.min(L.next - 1, Math.floor((x + radius) / TILE));
  const r0 = Math.max(0, Math.floor((y - radius) / TILE));
  const r1 = Math.min(ROWS - 1, Math.floor((y + radius) / TILE));
  for (let r = r0; r <= r1; r++)
    for (let c = c0; c <= c1; c++) {
      const k = L.fg[cell(c, r)];
      if (k === AIR || TILE_HP[k] === 255) continue;
      if (Math.hypot(c * TILE + TILE / 2 - x, r * TILE + TILE / 2 - y) <= radius) out.push([c, r]);
    }
  return out;
}

/** Knock `dmg` hits off a block. Returns the kind that broke, or 0 if it held or was empty. */
export function damageCell(L: Level, c: number, r: number, dmg: number) {
  if (r < 0 || r >= ROWS || c < L.start || c >= L.next) return 0;
  const i = cell(c, r);
  const k = L.fg[i];
  if (!k || TILE_HP[k] === 255) return 0;
  if (L.hp[i] > dmg) {
    L.hp[i] -= dmg;
    return 0;
  }
  L.fg[i] = AIR;
  L.hp[i] = 0;
  return k;
}

/* ------------------------------------------------------------------ */
/* Generation                                                          */
/* ------------------------------------------------------------------ */

function put(L: Level, c: number, r: number, k: number, back = -1) {
  if (r < 0 || r >= ROWS) return;
  const i = cell(c, r);
  L.fg[i] = k;
  L.hp[i] = TILE_HP[k];
  if (back >= 0) L.bg[i] = back;
}

function fill(L: Level, c0: number, r0: number, w: number, h: number, k: number, back = -1) {
  for (let r = r0; r < r0 + h; r++) for (let c = c0; c < c0 + w; c++) put(L, c, r, k, back);
}

/** Fill the back-wall grid only, wherever the foreground is empty. */
function backWall(L: Level, c0: number, r0: number, w: number, h: number, k: number) {
  for (let r = Math.max(0, r0); r < Math.min(ROWS, r0 + h); r++)
    for (let c = c0; c < c0 + w; c++) {
      const i = cell(c, r);
      if (L.fg[i] === AIR) L.bg[i] = k;
    }
}

/**
 * Target ground row at column `c`: slow noise picks lowland or mountain range, ridged noise adds
 * peaks inside ranges, and hills and roughness go on top. Flat over the landing zone.
 */
export function terrainAt(L: Level, c: number) {
  const s = L.seed;
  const relief = L.theme.relief * smoothstep(40, 120, c);
  if (relief <= 0) return BASE_ROW;
  // 0 in the lowlands, 1 in the heart of a range; ranges are about 600 columns apart
  const range = smoothstep(-0.15, 0.45, fbm1(s + 11, c / 300, 2));
  const peaks = ridged1(s + 23, c / 70, 3, 0.45) * (6 + 30 * range);
  const hills = (fbm1(s + 37, c / 34, 3) * 0.5 + 0.5) * 9;
  const rough = fbm1(s + 51, c / 7, 2) * 1.4;
  const lift = (peaks + hills + rough) * relief - 4 * (1 - range) * relief;
  return Math.max(PEAK_ROW, Math.min(VALLEY_ROW, Math.round(BASE_ROW - lift)));
}

/** A theme's broad soil and rock family, which picks its strata. */
type Ground = "volcanic" | "frozen" | "arid" | "wet" | "tropic" | "alien" | "temperate";
function groundKind(t: Theme): Ground {
  if (t.deep === HIVE) return "alien";
  if (t.deep === ROCK || t.top === ASH) return "volcanic";
  if (t.top === SNOW) return "frozen";
  if (t.deep === SANDSTONE || t.deep === REDROCK || t.top === SAND) return "arid";
  if (t.fill === MUD || t.top === MUD) return "wet";
  if (t.fill === CLAY || t.decor === "jungle" || t.decor === "palms") return "tropic";
  return "temperate";
}

/**
 * Write one fresh column with its top at row `h`, dropping the oldest stored column if the ring is
 * full. Soil, strata and ore depend on the theme's ground kind. `granite` fills under the topsoil
 * with granite and no caves, so climbable cliffs have no holes; `caves` carves noise tunnels.
 */
function column(L: Level, c: number, h: number, granite = false, caves = !granite) {
  if (c - L.start >= RING) L.start = c - RING + 1;
  const t = L.theme;
  const sc = c & MASK;
  const s = L.seed;
  for (let r = 0; r < ROWS; r++) {
    const i = r * RING + sc;
    L.fg[i] = L.bg[i] = L.hp[i] = L.burn[i] = 0;
  }
  const kind = groundKind(t);
  // Cliffs show bare fill and high summits bare rock, except in snow themes
  const left = c > L.start ? L.ground[(c - 1) & MASK] : h;
  const steep = Math.abs(left - h) >= 2;
  const alt = BASE_ROW - h;
  const summit = alt > 24 && t.top !== SNOW;
  let top = summit ? t.deep : steep ? t.fill : t.top;
  if (!granite && t.top !== CONCRETE && kind !== "alien") {
    const meadow = noise2(s + 111, c / 18, 0);
    if (t.top === GRASS && !steep && (kind === "tropic" && t.fill === CLAY ? meadow > 0.3 : meadow > 0.7))
      top = DRYGRASS;
    // High ground: snow when cold or temperate, scree when dry, moss rock when wet
    if (alt > 30 && (kind === "temperate" || kind === "frozen")) top = SNOW;
    else if (alt > 22 && (kind === "arid" || kind === "frozen")) top = SCREE;
    else if (alt > 18 && (kind === "temperate" || kind === "wet" || kind === "tropic") && meadow < 0.45) top = MOSSROCK;
    else if (steep && kind === "temperate" && noise2(s + 113, c / 5, 0) > 0.7) top = MOSSROCK;
  }
  // Patches of hard country have granite under the soil
  const hard = granite || noise2(s + 71, c / 45, 0.5) > 0.6;
  // Soil is thinnest on summits
  const soil = summit ? 1 : 3 + Math.round(noise2(s + 3, c / 9, 0) * 3);
  for (let r = h; r < ROWS; r++) {
    let k = r >= ROWS - 2 ? BEDROCK : r === h ? top : r <= h + soil ? t.fill : t.deep;
    const depth = r - h;
    if (k === t.fill && r > h) {
      // Pockets of other soils in the topsoil
      const soilN = noise2(s + 81, c / 12, r / 5);
      if (kind === "frozen" && depth <= soil) k = soilN > 0.62 ? GRAVEL : PERMAFROST;
      else if (soilN > 0.74) k = kind === "volcanic" ? TUFF : kind === "arid" ? SALT : GRAVEL;
      else if (soilN > 0.62)
        k =
          kind === "wet"
            ? PEAT
            : kind === "tropic"
              ? LATERITE
              : kind === "volcanic"
                ? BASALT
                : t.fill === DIRT || t.fill === LOAM
                  ? CLAY
                  : LOAM;
      if (depth === 1 && (t.top === GRASS || t.top === MUD) && noise2(s + 117, c / 4, 0) > 0.55) k = ROOTS;
      if (kind === "arid" && depth <= 2 && noise2(s + 119, c / 9, r / 3) > 0.8) k = SALT;
    }
    if (k === t.deep && r < ROWS - 2) {
      // Rock strata: granite in hard country, noise bands of other rock elsewhere
      if (hard && depth > soil + 1) k = granite && noise2(s + 121, c / 6, r / 2) > 0.8 ? QUARTZ : GRANITE;
      else {
        const rockN = noise2(s + 91, c / 10, r / 6);
        const band = noise2(s + 101, c / 40, r / 2.2);
        if (kind === "volcanic") {
          if (band > 0.74) k = OBSIDIAN;
          else if (depth > 18 && rockN > 0.76) k = MAGMA;
          else if (depth <= 3 && rockN > 0.6) k = SULFUR;
          else if (rockN < 0.18) k = BASALT;
        } else if (kind === "alien") {
          if (rockN > 0.82) k = AMETHYST;
        } else {
          if (rockN > 0.72) k = BASALT;
          else if (rockN < 0.16) k = kind === "arid" ? LIMESTONE : CHALK;
          else if (band > 0.76) k = kind === "arid" ? LIMESTONE : SHALE;
          else if (noise2(s + 103, c / 7, r / 5) > 0.86) k = QUARTZ;
          if (
            (k === SHALE || k === LIMESTONE || k === CHALK || k === t.deep) &&
            noise2(s + 107, c / 2.5, r / 2.5) > 0.9
          )
            k = FOSSIL;
        }
      }
    }
    if (k === t.deep && r < ROWS - 2 && noise2(s + 7, c / 6, r / 3) > 0.78) k = t.fill;
    else if ((k === t.deep || k === GRANITE || k === SHALE) && r < ROWS - 3) {
      // Ore veins get richer with depth: coal, then copper and iron, gold, and diamond deepest
      const ore = noise2(s + 41, c / 3.5, r / 2.5);
      if (ore > 0.86) {
        const deep = noise2(s + 43, c / 9, r / 9);
        k =
          depth > 40 && deep > 0.72
            ? DIAMOND
            : depth > 22 && deep > 0.6
              ? GOLD
              : depth > 12
                ? deep > 0.5
                  ? COPPER
                  : IRON
                : depth > 6 && deep > 0.6
                  ? COPPER
                  : COAL;
      } else if (depth > 28 && noise2(s + 127, c / 5, r / 4) > 0.9) k = AMETHYST;
    }
    put(L, c, r, k, r <= h + soil ? t.fill : t.deep);
  }
  // Caves well under the surface. None under cities, so a crater in a street never drops into one.
  if (!t.ceiling && c >= 40 && caves && t.backdrop !== "city" && t.backdrop !== "megacity")
    for (let r = h + 7; r < ROWS - 4; r++) {
      const worm = Math.abs(noise2(L.seed + 13, c / 11, r / 6) - 0.5);
      const hall = noise2(L.seed + 17, c / 26, r / 9);
      if (worm < 0.01 || hall > 0.92) {
        const i = cell(c, r);
        L.fg[i] = L.hp[i] = 0;
        L.bg[i] = t.deep;
      }
    }
  if (t.ceiling) {
    // Cave roof for ceiling themes, with crystals along its underside
    const roof = Math.max(3, h - 22 + Math.round(Math.sin(c * 0.5) * 2 + L.rand() * 3));
    for (let r = 0; r <= roof; r++) put(L, c, r, r === roof && L.rand() < 0.18 ? CRYSTAL : STONE, STONE);
  } else if (L.rand() < 0.05) put(L, c, Math.min(ROWS - 3, h + 6 + int(L.rand, 8)), CRYSTAL);
  L.ground[sc] = h;
  L.crust[sc] = h;
}

/** A random key from a table of relative weights. */
const pick = <K extends string>(weights: Partial<Record<K, number>>, r: () => number): K => {
  const entries = Object.entries(weights) as [K, number][];
  let x = r() * entries.reduce((s, [, w]) => s + w, 0);
  for (const [f, w] of entries) if ((x -= w) < 0) return f;
  return entries[0][0];
};

/** Block kind for each buildings.ts material. */
const MATERIAL: Record<Material, number> = {
  brick: BRICK,
  darkbrick: DARKBRICK,
  concrete: CONCRETE,
  glass: GLASS,
  marble: MARBLE,
  plaster: PLASTER,
  siding: SIDING,
  stucco: STUCCO,
  tiles: TILES,
  wood: WOOD,
  steel: STEEL,
  tin: TIN,
  sandstone: SANDSTONE,
  stone: STONE,
  metal: METAL,
  parquet: PARQUET,
  carpet: CARPET,
  moss: MOSS,
  timber: TIMBER,
  adobe: ADOBE,
  terracotta: TERRACOTTA,
  jade: JADE,
  slate: SLATE,
  shoji: SHOJI,
  azulejo: AZULEJO,
  limestone: LIMESTONE,
  snow: SNOW,
  gold: GOLD,
};

/**
 * Place an asset sprite on the floor at the top of row `row`, from column c, raised `lift` px for
 * wall and ceiling items. Skipped if the space above is blocked or it would overlap a recent sprite.
 */
function placeAt(L: Level, id: AssetId, c: number, row: number, lift = 0) {
  const a = asset(id);
  const cols = Math.ceil(a.w / TILE);
  for (let k = 0; k < cols; k++) {
    const above = L.fg[cell(c + k, row - 1)];
    if (above !== AIR && above !== LADDER && above !== RAIL) return false;
  }
  const x = c * TILE + (cols * TILE) / 2;
  const y = row * TILE - lift;
  for (let i = L.props.length - 1; i >= Math.max(0, L.props.length - 8); i--) {
    const q = L.props[i];
    if (q.kind === "sprite" && Math.abs(q.y - y) < 4 && Math.abs(q.x - x) < (asset(q.id).w + a.w) / 2 + 1) return false;
  }
  L.props.push({ kind: "sprite", id, x, y, flip: L.rand() < 0.5, back: false });
  return true;
}

/**
 * Build a cut-away buildings.ts building on flat ground from column c0, with open doorways through
 * the ground floor, a ladder through every floor, furniture, roof, sign and guards. Returns its width.
 */
function building(L: Level, c0: number, spec: BuildingSpec, out: Spawn[], troop: () => Spawn["kind"]) {
  const r = L.rand;
  const t = L.theme;
  const g = L.h;
  // Hardened buildings, and a few others at random, use reinforced concrete
  const hard = HARDENED.has(spec.key) || r() < 0.1;
  const wall = hard ? REINFORCED : MATERIAL[spec.wall];
  const back = MATERIAL[spec.back];
  const slab = hard ? REINFORCED : spec.floor ? MATERIAL[spec.floor] : wall;
  const fh = spec.fh;
  const bw = spec.width[0] + int(r, spec.width[1] - spec.width[0] + 1);
  const room = 12;
  const floors = Math.max(
    1,
    Math.min(spec.floors[0] + int(r, spec.floors[1] - spec.floors[0] + 1), Math.floor((g - room) / fh))
  );
  const w = bw + 2;
  const x0 = c0 + 1;
  const x1 = x0 + bw - 1;
  for (let c = c0; c < c0 + w; c++) column(L, c, g);
  const top = g - floors * fh;
  const spawnAt = (c: number, row: number) => out.push({ kind: troop(), x: c * TILE + TILE / 2, y: row * TILE });
  const pickItem = (list: string[]) => list[int(r, list.length)];
  for (let f = 0; f < floors; f++) {
    const base = g - f * fh;
    const ceil = base - fh;
    // Each storey: back wall with windows, side walls, then the floor above
    for (let row = ceil + 1; row < base; row++) {
      const y = row - ceil;
      for (let c = x0 + 1; c < x1; c++) {
        const i = cell(c, row);
        L.fg[i] = AIR;
        L.hp[i] = 0;
        const k = c - x0;
        let win = false;
        switch (spec.windows) {
          case "grid":
            win = y >= 2 && y <= 3 && k % 3 === 1;
            break;
          case "pairs":
            win = y >= 1 && y <= 2 && k % 4 !== 0 && k % 4 !== 3;
            break;
          case "strip":
            win = y === 2;
            break;
          case "glass":
            win = true;
            break;
          case "arched":
            win = (y === 2 || y === 3) && k % 3 === 1;
            break;
          case "porthole":
            win = y === 2 && k % 3 === 1;
            break;
          case "slits":
            win = y === 1 && k % 4 === 2;
            break;
          case "shop":
            win = f === 0 ? y >= 1 && y <= 3 && k > 1 && k < bw - 2 : y === 2 && k % 3 === 1;
            break;
          default:
            win = false;
        }
        L.bg[i] = win ? (spec.windows === "glass" ? GLASS : WINDOW) : back;
      }
      for (const c of [x0, x1]) put(L, c, row, wall, back);
    }
    // Open doorways through the ground floor so the squad can run through; maybe a side window upstairs
    if (f === 0)
      for (const c of [x0, x1])
        for (let row = base - 3; row < base; row++) {
          const i = cell(c, row);
          L.fg[i] = AIR;
          L.hp[i] = 0;
          L.bg[i] = back;
        }
    else if (r() < 0.6) put(L, r() < 0.5 ? x0 : x1, base - 2, WINDOW, back);
    fill(L, x0, ceil, bw, 1, slab, back);
    // Furniture
    const list = f > 0 && spec.upper ? spec.upper : spec.items;
    for (let k = Math.max(2, Math.round(bw / 4)) + int(r, 2); k > 0; k--)
      placeAt(L, pickItem(list), x0 + 1 + int(r, Math.max(1, bw - 3)), base);
    if (spec.walls?.length)
      for (let k = 1 + int(r, 2); k > 0; k--) {
        const id = pickItem(spec.walls);
        const a = asset(id);
        const lift = a.hang ? (fh - 1) * TILE - a.h - 1 : Math.min((fh - 1) * TILE - a.h - 2, 12);
        if (lift >= 0) placeAt(L, id, x0 + 1 + int(r, Math.max(1, bw - 3)), base, lift);
      }
    for (const id of list) {
      const a = asset(id);
      if (a.hang && r() < 0.5) placeAt(L, id, x0 + 1 + int(r, Math.max(1, bw - 3)), base, (fh - 1) * TILE - a.h - 1);
    }
    // Guards
    for (let k = (spec.guards ?? 1) * (f === 0 ? 0.6 : 1) * (0.4 + r()) * t.enemies * L.intensity * 0.25; k >= 1; k--)
      spawnAt(x0 + 2 + int(r, Math.max(1, bw - 4)), base);
  }
  ladder(L, x1 - 1, top, g);
  // Roof
  const cx = x0 + (bw >> 1);
  let roofTop = top;
  switch (spec.roof) {
    case "parapet":
      put(L, x0, top - 1, wall);
      put(L, x1, top - 1, wall);
      break;
    case "pitched": {
      const roofK =
        spec.wall === "wood" || spec.wall === "siding"
          ? WOOD
          : spec.wall === "tin"
            ? TIN
            : t.build === WOOD
              ? WOOD
              : BRICK;
      for (let k = 1; k <= 5; k++) {
        const from = x0 - 1 + k;
        const width = bw + 2 - k * 2;
        if (width <= 0) break;
        fill(L, from, top - k, width, 1, roofK, roofK);
        roofTop = top - k;
      }
      break;
    }
    case "dome": {
      const rad = Math.min(5, bw >> 1);
      for (let dy = 1; dy <= rad; dy++) {
        const half = Math.round(Math.sqrt(rad * rad - (dy - 0.5) ** 2));
        fill(
          L,
          cx - half,
          top - dy,
          half * 2 + (bw % 2),
          1,
          dy === rad ? GOLD : spec.roofMat ? MATERIAL[spec.roofMat] : MARBLE,
          MARBLE
        );
      }
      roofTop = top - rad;
      break;
    }
    case "spire": {
      fill(L, cx - 1, top - 6, 3, 6, wall, back);
      put(L, cx, top - 4, WINDOW, back);
      fill(L, cx - 2, top - 7, 5, 1, wall);
      for (let k = 1; k <= 3; k++) put(L, cx, top - 7 - k, k === 3 ? GOLD : TIN);
      placeAt(L, "bellframe", cx - 1, top - 5);
      roofTop = top - 10;
      break;
    }
    case "sawtooth":
      for (let c = x0; c <= x1; c++) {
        const k = (c - x0) % 4;
        if (k < 3) put(L, c, top - 1 - Math.min(k, 1), k === 0 ? wall : GLASS);
        if (k === 1) put(L, c, top - 1, GLASS);
      }
      break;
    case "stepped":
      fill(L, x0 + 2, top - 3, bw - 4, 3, wall, back);
      fill(L, x0 + 4, top - 5, Math.max(1, bw - 8), 2, wall, back);
      roofTop = top - 5;
      break;
    case "garden":
      put(L, x0, top - 1, wall);
      put(L, x1, top - 1, wall);
      placeAt(L, "roofgarden", x0 + 1, top);
      placeAt(L, "hedge", x0 + 4, top);
      placeAt(L, "treeplanter", x1 - 2, top);
      break;
    case "helipad":
      placeAt(L, "helipad", x0 + 1, top);
      break;
    case "tank":
      placeAt(L, "roofwatertank", x0 + 1 + int(r, Math.max(1, bw - 3)), top);
      break;
    case "billboard":
      placeAt(L, "billboardroof", x0 + int(r, Math.max(1, bw - 4)), top);
      break;
    case "antenna":
      L.props.push({ kind: "antenna", x: (x1 - 2) * TILE + 4, y: top * TILE, h: 30 + int(r, 30) });
      break;
    case "chimney":
      placeAt(L, "chimney", x0 + 1, top);
      if (bw > 9) placeAt(L, "chimney", x1 - 2, top);
      break;
    case "solar":
      placeAt(L, "solarpanels", x0 + 1, top);
      break;
    case "bell":
      placeAt(L, "bellframe", cx - 1, top);
      break;
    case "neon":
      break;
    case "pagoda": {
      // Tiers flaring at the eaves, a gold finial on top
      const roofK = spec.roofMat ? MATERIAL[spec.roofMat] : JADE;
      let y = top;
      for (let tier = 0, half = (bw >> 1) + 2; tier < 3 && half > 1; tier++, half -= 2) {
        fill(L, cx - half, y - 1, half * 2 + 1, 1, roofK);
        put(L, cx - half - 1, y - 2, roofK);
        put(L, cx + half + 1, y - 2, roofK);
        fill(L, cx - half + 2, y - 3, Math.max(1, half * 2 - 3), 2, wall, back);
        y -= 3;
      }
      fill(L, cx, y - 3, 1, 3, GOLD);
      roofTop = y - 3;
      break;
    }
    case "onion": {
      // Drum and onion dome with a gold spire, and small corner turrets
      const domeK = spec.roofMat ? MATERIAL[spec.roofMat] : GOLD;
      fill(L, cx - 1, top - 4, 3, 4, wall, back);
      const rows = [3, 5, 5, 5, 3, 1];
      rows.forEach((wd, k) => fill(L, cx - (wd >> 1), top - 5 - k, wd, 1, domeK));
      fill(L, cx, top - 13, 1, 2, GOLD);
      for (const side of [x0 + 1, x1 - 1]) {
        fill(L, side, top - 3, 1, 3, wall, back);
        put(L, side, top - 4, domeK);
      }
      roofTop = top - 13;
      break;
    }
    case "minaret": {
      // Dome over the hall and a tall corner tower with a balcony
      const rad = Math.min(4, bw >> 1);
      for (let dy = 1; dy <= rad; dy++) {
        const half = Math.round(Math.sqrt(rad * rad - (dy - 0.5) ** 2));
        fill(L, cx - half, top - dy, half * 2 + (bw % 2), 1, spec.roofMat ? MATERIAL[spec.roofMat] : AZULEJO);
      }
      fill(L, x1, top - 14, 1, 14, wall, back);
      fill(L, x1 - 1, top - 11, 3, 1, wall);
      put(L, x1, top - 15, GOLD);
      roofTop = top - 15;
      break;
    }
    case "mansard": {
      // Steep slate roof with dormer windows
      for (let k = 1; k <= 3; k++) {
        fill(L, x0 + k - 1, top - k, bw - (k - 1) * 2, 1, SLATE);
        if (k === 2) for (let c = x0 + 3; c < x1 - 2; c += 3) put(L, c, top - k, WINDOW, SLATE);
      }
      roofTop = top - 3;
      break;
    }
    case "stepgable": {
      for (let k = 1, half = bw >> 1; half > 0; k += 2, half -= 2) {
        fill(L, cx - half, top - k - 1, half * 2 + (bw % 2), 2, wall, back);
        roofTop = top - k - 1;
      }
      put(L, cx, roofTop - 1, wall);
      break;
    }
    case "columns": {
      // Temple front: columns, an entablature and a pediment
      for (let c = x0; c <= x1; c += 2) put(L, c, top - 1, LIMESTONE);
      fill(L, x0 - 1, top - 2, bw + 2, 1, LIMESTONE);
      for (let k = 1, half = (bw >> 1) + 1; half > 0; k++, half -= 2)
        fill(L, cx - half, top - 2 - k, half * 2 + (bw % 2), 1, LIMESTONE);
      roofTop = top - 2 - Math.ceil((bw >> 1) / 2) - 1;
      break;
    }
    case "pyramid": {
      const rk = spec.roofMat ? MATERIAL[spec.roofMat] : SANDSTONE;
      let half = (bw >> 1) + 1;
      let y = top - 1;
      while (half > 0 && y > 4) {
        fill(L, cx - half, y, half * 2 + (bw % 2), 1, rk);
        half--;
        y--;
      }
      put(L, cx, y, GOLD);
      roofTop = y;
      break;
    }
    case "castle": {
      // Battlements, and a capped turret at each end
      for (let c = x0; c <= x1; c += 2) put(L, c, top - 1, wall);
      for (const tx of [x0 - 1, x1]) {
        fill(L, tx, top - 5, 2, 5, wall, back);
        put(L, tx, top - 3, WINDOW, back);
        fill(L, tx, top - 6, 2, 1, spec.roofMat ? MATERIAL[spec.roofMat] : SLATE);
        put(L, tx + (tx < cx ? 0 : 1), top - 7, spec.roofMat ? MATERIAL[spec.roofMat] : SLATE);
      }
      roofTop = top - 7;
      break;
    }
    case "chalet": {
      // Wide, shallow roof overhanging the walls
      for (let k = 1; k <= 3; k++) fill(L, x0 - 3 + k * 2, top - k, bw + 6 - k * 4, 1, WOOD);
      roofTop = top - 3;
      break;
    }
    case "igloo": {
      const rad = Math.min(4, (bw >> 1) + 1);
      for (let dy = 1; dy <= rad; dy++) {
        const half = Math.round(Math.sqrt(rad * rad - (dy - 0.5) ** 2));
        fill(L, cx - half, top - dy, half * 2 + (bw % 2), 1, SNOW);
      }
      roofTop = top - rad;
      break;
    }
    case "stupa": {
      // Bell shape with a tall gold spire
      [7, 7, 5, 5, 3].forEach((wd, k) => fill(L, cx - (wd >> 1), top - 1 - k, wd, 1, k > 2 ? GOLD : PLASTER));
      fill(L, cx, top - 10, 1, 5, GOLD);
      roofTop = top - 10;
      break;
    }
    case "clock": {
      // Tower with the clock high on its face and a spire
      fill(L, cx - 1, top - 9, 3, 9, wall, back);
      placeAt(L, "wallclock", cx - 1, top - 5);
      fill(L, cx - 2, top - 10, 5, 1, wall);
      for (let k = 0; k < 3; k++) put(L, cx, top - 11 - k, k === 2 ? GOLD : SLATE);
      roofTop = top - 13;
      break;
    }
    case "artdeco": {
      // Stepped setbacks and a spire
      fill(L, x0 + 1, top - 4, bw - 2, 4, wall, back);
      fill(L, x0 + 3, top - 7, Math.max(1, bw - 6), 3, wall, back);
      for (let c = x0 + 2; c < x1 - 1; c += 2) put(L, c, top - 2, WINDOW, back);
      fill(L, cx, top - 13, 1, 6, METAL);
      roofTop = top - 13;
      break;
    }
    case "brutalist":
      // Heavy slab overhanging the walls
      fill(L, x0 - 2, top - 2, bw + 4, 2, CONCRETE);
      roofTop = top - 2;
      break;
  }
  if (
    spec.roof === "flat" ||
    spec.roof === "parapet" ||
    spec.roof === "neon" ||
    spec.roof === "helipad" ||
    spec.roof === "antenna" ||
    spec.roof === "solar" ||
    spec.roof === "tank"
  )
    for (let k = int(r, 3); k > 0; k--)
      placeAt(L, spec.top ? spec.top[int(r, spec.top.length)] : "acunit", x0 + 1 + int(r, Math.max(1, bw - 3)), top);
  // Sign, awning and a prop out front
  if (spec.sign)
    L.props.push({
      kind: "sign",
      x: cx * TILE + (bw % 2 ? TILE / 2 : 0),
      y: (spec.roof === "neon" || spec.roof === "flat" || spec.roof === "parapet" ? top : roofTop) * TILE - 1,
      text: spec.sign,
      ink: spec.ink ?? "#ffffff",
      board: spec.board ?? "#1e1e24",
      neon: !!spec.neon,
    });
  if (spec.awning) L.props.push({ kind: "awning", x: x0 * TILE + 2, y: (g - 3) * TILE + 1, w: 14, color: spec.awning });
  if (spec.outside) placeProp(L, spec.outside[int(r, spec.outside.length)], c0 - 3);
  // Maybe a guard on the roof
  if (floors > 1 && r() < 0.6) spawnAt(x0 + 2 + int(r, Math.max(1, bw - 4)), top);
  return w;
}

/**
 * Place an asset sprite on the ground at column c if the ground under it is level and clear and it
 * doesn't overlap the last few props. `back` sprites sit in shadow behind the blocks, uneven ground or not.
 */
function placeProp(L: Level, id: AssetId, c: number, back = false) {
  const a = asset(id);
  const cols = Math.ceil(a.w / TILE);
  const gr = (k: number) => L.ground[k & MASK];
  let g = gr(c);
  for (let k = 1; k < cols; k++) {
    const gk = gr(c + k);
    if (!back && gk !== g) return false;
    g = Math.max(g, gk);
  }
  if (!back)
    for (let k = 0; k < cols; k++)
      if (L.fg[cell(c + k, g - 1)] !== AIR && L.fg[cell(c + k, g - 1)] !== RAIL) return false;
  const x = c * TILE + (cols * TILE) / 2;
  for (let i = L.props.length - 1; i >= Math.max(0, L.props.length - 4); i--) {
    const q = L.props[i];
    if (q.kind === "sprite" && q.back === back && Math.abs(q.x - x) < (asset(q.id).w + a.w) / 2 + 2) return false;
  }
  L.props.push({ kind: "sprite", id, x, y: g * TILE, flip: L.rand() < 0.5, back });
  return true;
}

/** Ladder in column c from row `top` down to, but not including, row `bottom`, through air and floors. */
function ladder(L: Level, c: number, top: number, bottom: number) {
  for (let row = Math.max(0, top); row < Math.min(ROWS, bottom); row++) {
    const k = L.fg[cell(c, row)];
    if (k === AIR || k === L.theme.build || k === WOOD || k === METAL || k === CONCRETE) put(L, c, row, LADDER);
  }
}

/** Generate columns up to `until`, exclusive. Returns the entities to spawn in them. */
export function extend(L: Level, until: number): Spawn[] {
  const out: Spawn[] = [];
  while (L.next < until) feature(L, out);
  // Drop props whose columns have left the ring
  if (L.props.length > 300 || (L.props[0] && L.props[0].x < L.start * TILE))
    L.props = L.props.filter((p) => p.x >= L.start * TILE);
  return out;
}

/** Generate the next feature from L.next and move L.next past it. */
function feature(L: Level, out: Spawn[]) {
  const t = L.theme;
  const r = L.rand;
  const plan = L.plan;
  const c0 = L.next;
  const px = (c: number) => c * TILE + TILE / 2;
  // Read the ring directly: L.next only moves past this feature once it's finished, so groundOf would miss it
  const gr = (c: number) => L.ground[c & MASK];
  const troop = (): Spawn["kind"] => pick(t.troops, r);
  const enemies = (from: number, w: number, row = -1, scale = 1) => {
    const n = Math.round(((t.enemies * L.intensity * scale * w) / 12) * (0.5 + r()));
    for (let k = 0; k < n; k++) {
      const c = from + int(r, w);
      out.push({ kind: troop(), x: px(c) + (r() - 0.5) * 4, y: (row >= 0 ? row : gr(c)) * TILE });
    }
  };
  const spawn = (kind: Spawn["kind"], c: number, row = gr(c)) => out.push({ kind, x: px(c), y: row * TILE });
  const barrels = (from: number, w: number) => {
    if (r() >= Math.min(0.95, t.barrels * 1.9)) return;
    const n = 2 + int(r, 4);
    const c = from + int(r, Math.max(1, w - n));
    for (let k = 0; k < n; k++) {
      const roll = r();
      spawn(
        roll < 0.16
          ? "ammo"
          : roll < 0.3
            ? "propane"
            : roll < 0.41
              ? "canister"
              : roll < 0.5
                ? "cylinder"
                : roll < 0.62
                  ? "tnt"
                  : "barrel",
        c + k
      );
    }
  };
  const prop = (id: AssetId, c: number, back = false) => placeProp(L, id, c, back);
  /** Scatter `n` of the theme's props over open ground. */
  const scatter = (from: number, w: number, n: number) => {
    for (let k = 0; k < n; k++) prop(pickAsset(r, t.backdrop), from + int(r, Math.max(1, w - 2)));
  };
  /** Sometimes place one large background prop. */
  const standing = (from: number, w: number) => {
    if (t.ceiling || r() > 0.5) return;
    prop(pickAsset(r, t.backdrop, true), from + int(r, Math.max(1, w - 3)), true);
  };
  const climb = (c: number, top: number, bottom: number) => ladder(L, c, top, bottom);
  /** Maybe bury a room under this stretch, deep enough to leave the surface intact. */
  const underground = (from: number, w: number) => {
    if (t.ceiling || from < 50 || w < 12) return;
    let low = 0;
    for (let c = from; c < from + w; c++) low = Math.max(low, gr(c));
    const top = low + 4;
    const rh = 4 + int(r, 2);
    const floor = top + rh + 1;
    if (floor >= ROWS - 4) return;
    const x0 = from + 1;
    const rw = w - 2;
    const kinds = [
      "mine",
      "mine",
      "mine",
      "mine",
      "mine",
      "crypt",
      "bunker",
      "sewer",
      "subway",
      "cavern",
      "lab",
      "vault",
    ] as const;
    const kind = kinds[int(r, kinds.length)];
    const shell: Record<"mine" | "crypt" | "bunker" | "sewer" | "subway" | "cavern" | "lab" | "vault", number> = {
      mine: t.deep,
      crypt: STONE,
      bunker: CONCRETE,
      sewer: SEWER,
      subway: TILES,
      cavern: t.deep,
      lab: STEEL,
      vault: STEEL,
    };
    const k = shell[kind];
    fill(L, x0, top, rw, rh + 2, k, k);
    fill(L, x0 + 1, top + 1, rw - 2, rh, AIR, kind === "cavern" || kind === "mine" ? t.deep : k);
    if (kind === "cavern") {
      // Ragged walls instead of a box
      for (let c = x0 + 1; c < x0 + rw - 1; c++) {
        if (r() < 0.4) put(L, c, top + 1, t.deep, t.deep);
        if (r() < 0.25) put(L, c, floor - 1, t.deep, t.deep);
      }
    }
    const item = (list: string[], n: number, lift = 0) => {
      for (let j = 0; j < n; j++) {
        const id = list[int(r, list.length)];
        const a = asset(id);
        placeAt(L, id, x0 + 1 + int(r, Math.max(1, rw - 3)), floor, a.hang ? rh * TILE - a.h - 1 : lift);
      }
    };
    switch (kind) {
      case "mine": {
        // Ladder shaft from the surface with a headframe over it
        const sx = x0 + 2;
        const sg = gr(sx);
        for (let row = sg; row < floor; row++) {
          const i = cell(sx, row);
          L.fg[i] = LADDER;
          L.hp[i] = TILE_HP[LADDER];
          L.bg[i] = t.deep;
          const j = cell(sx + 1, row);
          if (row >= sg + 1) {
            L.fg[j] = AIR;
            L.hp[j] = 0;
            L.bg[j] = t.deep;
          }
        }
        placeProp(L, "headframe", sx - 1, true);
        for (let c = x0 + 1; c < x0 + rw - 1; c++) put(L, c, floor - 1, RAIL);
        for (let c = x0 + 2; c < x0 + rw - 2; c += 5) placeAt(L, "pitprops", c, floor);
        spawn("minecart", x0 + 3 + int(r, Math.max(1, rw - 6)), floor);
        item(["cartcoal", "orepile", "toolsrack", "orepile", "drillrig", "timbers", "dynamite", "carttipped"], 3);
        item(["minelantern"], 2);
        break;
      }
      case "crypt":
        item(["sarcophagus", "amphora", "treasurechest", "skeleton", "amphora"], 4);
        item(["fossilfish"], 1, 14);
        L.props.push({ kind: "brazier", x: (x0 + 2) * TILE, y: floor * TILE });
        break;
      case "bunker":
        item(["bunkerbed", "radio", "weaponrack", "maptable", "ammocrate", "crates"], 4);
        item(["tunnellamp"], 1);
        if (t.enemies > 0)
          for (let j = 1 + int(r, 2); j > 0; j--)
            out.push({ kind: troop(), x: (x0 + 2 + int(r, rw - 4)) * TILE + 4, y: floor * TILE });
        break;
      case "sewer":
        for (let c = x0 + 1; c < x0 + rw - 1; c++) put(L, c, floor - 1, WATER, k);
        put(L, x0, floor - 1, SEWER);
        item(["sewerpipe"], 1, 8);
        item(["roots"], 1);
        break;
      case "subway":
        for (let c = x0 + 1; c < x0 + rw - 1; c++) put(L, c, floor - 1, RAIL);
        if (rw > 10) placeAt(L, "subwaycar", x0 + 2, floor);
        item(["tunnellamp"], 2);
        item(["platformbench", "ticketmachine"], 1);
        break;
      case "cavern":
        if (rw > 6) placeAt(L, "undergroundpool", x0 + 2, floor);
        item(["stalagmites", "crystals", "orepile", "skeleton", "mushrooms"], 3);
        item(["roots"], 1);
        break;
      case "lab":
        item(["labbench", "serverrack", "controlpanel", "toxicdrum", "microscope"], 4);
        item(["tunnellamp"], 1);
        break;
      case "vault":
        item(["safe", "treasurechest", "orepile", "safe"], 3);
        item(["tunnellamp"], 1);
        break;
    }
  };
  const ground = (from: number, w: number, h = L.h) => {
    for (let c = from; c < from + w; c++) column(L, c, h);
  };
  /**
   * Ground that follows terrainAt, changing at most two rows per column so climbs stay jumpable and
   * slopes don't read as pits.
   */
  const natural = (from: number, w: number) => {
    for (let c = from; c < from + w; c++) {
      const d = terrainAt(L, c) - L.h;
      L.h += d > 0 ? Math.min(2, d) : Math.max(-2, d);
      column(L, c, L.h);
    }
  };
  const night = t.night > 0.3;
  /** Sometimes a ruined back-wall structure, for depth. */
  const skyline = (from: number, w: number) => {
    if (!t.skyline || r() > 0.45) return;
    const bw = 5 + int(r, Math.max(1, w - 5));
    const bh = 8 + int(r, 14);
    const top = L.h - bh;
    backWall(L, from, top, bw, bh, t.skyline);
    // Broken windows: holes in the back wall
    for (let rr = top + 2; rr < L.h - 2; rr += 3)
      for (let cc = from + 1; cc < from + bw - 1; cc += 2) if (r() < 0.55) L.bg[cell(cc, rr)] = 0;
    // Ragged top
    for (let cc = from; cc < from + bw; cc++) for (let k = int(r, 3); k > 0; k--) L.bg[cell(cc, top + k - 1)] = 0;
  };
  /** City towers on the back-wall grid behind the street, with rows of windows and the odd mast. */
  const towers = (from: number, w: number) => {
    if (t.backdrop !== "city" && t.backdrop !== "megacity") return;
    const tall = t.backdrop === "megacity" ? 40 : 24;
    for (let c = from; c < from + w; ) {
      const bw = 5 + int(r, 8);
      const bh = 16 + int(r, tall);
      const top = Math.max(2, L.h - bh);
      const mat = [CONCRETE, DARKBRICK, GLASS, CONCRETE, STONE][int(r, 5)];
      backWall(L, c, top, bw, L.h - top, mat);
      for (let row = top + 2; row < L.h - 2; row += 3)
        for (let x = c + 1; x < c + bw - 1; x += 2) {
          const i = cell(x, row);
          if (!L.fg[i] && r() < 0.8) L.bg[i] = WINDOW;
        }
      if (r() < 0.3) backWall(L, c + (bw >> 1), top - 5, 1, 5, STEEL);
      c += bw + int(r, 4);
    }
  };
  const pine = (c: number, g: number) => {
    const ht = 7 + int(r, 5);
    fill(L, c, g - 2, 1, 2, TRUNK);
    for (let k = 0; k < ht - 2; k++) {
      const half = Math.floor((ht - 2 - k) / 2.2);
      fill(L, c - half, g - 3 - k, half * 2 + 1, 1, PINE);
    }
  };
  const deadTree = (c: number, g: number) => {
    const ht = 6 + int(r, 5);
    fill(L, c, g - ht, 1, ht, TRUNK);
    for (let k = 2; k < ht - 1; k += 2) {
      const dir = k % 4 === 0 ? 1 : -1;
      put(L, c + dir, g - k - 1, TRUNK);
      put(L, c + dir * 2, g - k - 2, TRUNK);
    }
  };
  let w = 0;
  /** Put a non-solid block such as a leaf or vine into open air only, within columns lo..hi-1. */
  const soft = (c: number, row: number, k: number, lo = c0, hi = c0 + w) => {
    if (c < lo || c >= hi || row < 0 || row >= ROWS) return;
    if (L.fg[cell(c, row)] === AIR) put(L, c, row, k);
  };
  const palm = (c: number, g: number) => {
    fill(L, c, g - 7, 1, 7, TRUNK);
    fill(L, c - 2, g - 8, 5, 1, LEAF);
    fill(L, c - 1, g - 9, 3, 1, LEAF);
    put(L, c - 3, g - 7, LEAF);
    put(L, c + 3, g - 7, LEAF);
  };
  const crown = t.decor === "autumn" ? REDLEAF : t.decor === "sakura" ? PINKLEAF : LEAF;
  /** A tall tree with a wide crown and hanging vines. */
  const bigTree = (c: number, g: number, lo: number, hi: number) => {
    const th = 9 + int(r, 6);
    fill(L, c, g - th, 1, th, TRUNK);
    const top = g - th - 2;
    [3, 7, 9, 7].forEach((cw, k) => {
      for (let x = c - (cw >> 1); x <= c + (cw >> 1); x++)
        soft(x, top + k, crown === LEAF || r() < 0.75 ? crown : LEAF, lo, hi);
    });
    for (const dx of [-4, -3, 2, 4]) {
      if (r() > 0.55) continue;
      for (let k = 0, len = 2 + int(r, 5); k < len; k++) soft(c + dx, top + 3 + k, VINE, lo, hi);
    }
  };
  const fern = (c: number, g: number, lo: number, hi: number) => {
    soft(c, g - 1, LEAF, lo, hi);
    soft(c + 1, g - 1, LEAF, lo, hi);
    if (r() < 0.6) soft(c, g - 2, LEAF, lo, hi);
  };
  const deco = (from: number, w: number) => {
    const c = from + 2 + int(r, Math.max(1, w - 4));
    const g = gr(c);
    if (t.decor === "palms" && r() < 0.6) palm(c, g);
    else if (t.decor === "autumn" || t.decor === "sakura") {
      if (r() < 0.7) bigTree(c, g, from, from + w);
      else fern(c, g, from, from + w);
    } else if (t.decor === "jungle") {
      const n = r();
      if (n < 0.45) bigTree(c, g, from, from + w);
      else if (n < 0.75) fern(c, g, from, from + w);
      else palm(c, g);
    } else if (t.decor === "pines" && r() < 0.7) pine(c, g);
    else if (t.decor === "deadwood" && r() < 0.6) deadTree(c, g);
    else if (t.decor === "fires" && r() < 0.5) L.props.push({ kind: "fire", x: px(c), y: g * TILE });
    else if (t.decor === "crystals" && r() < 0.6) put(L, c, g - 1, CRYSTAL);
    else if ((t.decor === "none" || t.decor === "rubble" || t.decor === "fires") && t.top === GRASS && r() < 0.7) {
      // Plain grassy themes get woods
      if (r() < 0.55) bigTree(c, g, from, from + w);
      else pine(c, g);
    } else if (t.decor === "rubble" && r() < 0.6) {
      put(L, c, g - 1, t.wall);
      if (r() < 0.5) put(L, c + 1, g - 1, CRATE);
    }
  };
  const hut = (x0: number, g: number, build: number) => {
    fill(L, x0 + 1, g - 4, 6, 4, AIR, build);
    fill(L, x0, g - 4, 1, 2, build);
    fill(L, x0 + 7, g - 4, 1, 2, build);
    fill(L, x0, g - 5, 8, 1, build);
  };

  /** `ow` columns of open sea from `from`, down to a rolling sandy seabed, with ships on it. */
  const sea = (from: number, ow: number) => {
    const g = L.h;
    // Seabed about 28 rows below the surface
    const bed = Math.min(ROWS - 4, g + 28);
    for (let c = from; c < from + ow; c++) {
      const floor = Math.min(ROWS - 3, Math.max(g + 6, bed - Math.round((noise2(L.seed + 61, c / 8, 0) - 0.4) * 10)));
      column(L, c, floor);
      put(L, c, floor, SAND);
      for (let row = g + 1; row < floor; row++) put(L, c, row, WATER, 0);
      // Coral reefs in banks
      const reef = noise2(L.seed + 63, c / 6, 0);
      if (reef > 0.62 && floor > g + 6) {
        put(L, c, floor, CORAL);
        if (reef > 0.74) put(L, c, floor - 1, CORAL, 0);
        if (reef > 0.86 && floor > g + 8) put(L, c, floor - 2, CORAL, 0);
      }
    }
    for (let c = from + 2; c < from + ow - 2; c += 3 + int(r, 6)) {
      const floor = L.ground[c & MASK];
      if (floor > g + 6) placeAt(L, r() < 0.6 ? "seaweed" : "coral", c, floor);
    }
    for (let c = from + 12; c < from + ow - 8; c += 40 + int(r, 30))
      out.push({ kind: "battleship", x: c * TILE, y: (g + 1) * TILE });
    for (let k = 1 + int(r, 3); k > 0; k--)
      out.push({ kind: "submarine", x: (from + 6 + int(r, ow - 12)) * TILE, y: (g + 1) * TILE });
    for (let k = 1 + int(r, 3); k > 0; k--)
      out.push({ kind: "boat", x: (from + 6 + int(r, ow - 12)) * TILE, y: (g + 1) * TILE });
  };

  /* ---- Out at sea: more sea, or come ashore ---- */
  if (L.openSea) {
    const total = (Object.values(t.features) as number[]).reduce((s, v) => s + v, 0);
    const more = (t.features.ocean ?? 0) / Math.max(1e-6, total);
    // Land must return for a boss arena, the stronghold or a biome border
    const ashore =
      (plan.boss && !plan.bossPlaced && c0 >= plan.boss.col - 4) || c0 >= plan.stronghold || plan.fortPlaced;
    if (!ashore && r() < more) {
      // More sea, with no shore in between
      w = 60 + int(r, 100);
      sea(c0, w);
    } else {
      // A beach with a lighthouse
      w = 4;
      ground(c0, w);
      prop("lighthousebig", c0 + 1, true);
      L.openSea = false;
    }
    L.next = c0 + w;
    return;
  }

  /* ---- The level's own shape: landing zone, boss arena, stronghold, run-off ---- */
  if (c0 < 40) {
    w = 40 - c0;
    ground(c0, w);
    deco(c0 + 6, w - 6);
    scatter(c0 + 4, w - 6, 4);
    standing(c0 + 10, w - 12);
    L.next = c0 + w;
    return;
  }
  if (plan.boss && !plan.bossPlaced && c0 >= plan.boss.col - 4) {
    // A wide, flat arena for the boss fight
    w = 40;
    ground(c0, w);
    plan.boss.col = c0 + 14;
    plan.bossPlaced = true;
    L.bossQueue.push({ col: c0 + 14, kind: plan.boss.kind });
    barrels(c0 + 4, 8);
    barrels(c0 + 26, 10);
    skyline(c0, 20);
    L.next = c0 + w;
    return;
  }
  if (!plan.fortPlaced && c0 >= plan.stronghold) {
    // End of the stretch: every `every`th one, unless endless, is the stronghold and finale;
    // otherwise a river marks the border into the next biome
    if (L.forceFort || (!L.endless && L.segment % L.every === 0)) {
      stronghold(L, c0, out, troop, night);
      return;
    }
    const rw = 6 + int(r, 6);
    w = rw + 4;
    ground(c0, 2);
    const g = L.h;
    ground(c0 + 2, rw, g + 2);
    for (let c = c0 + 2; c < c0 + 2 + rw; c++) {
      put(L, c, g + 2, MUD);
      put(L, c, g + 1, WATER);
    }
    ground(c0 + 2 + rw, 2);
    for (const c of [c0 + 1, c0 + 2 + rw]) soft(c, g - 1, LEAF, c0, c0 + w);
    L.next = c0 + w;
    const next = L.pickTheme ? L.pickTheme(t) : t;
    const length = Math.round((240 + int(r, 160)) * (next.long ?? 1));
    const kind = next.bosses.length && r() < 0.7 ? next.bosses[int(r, next.bosses.length)] : null;
    L.segment++;
    L.theme = next;
    L.plan = {
      length,
      boss: kind ? { col: L.next + Math.floor(length * 0.48), kind } : null,
      bossPlaced: false,
      stronghold: L.next + length,
      fortPlaced: false,
      targets: [],
    };
    L.borders.push({ col: c0 + 2 + (rw >> 1), theme: next, boss: kind });
    return;
  }
  if (plan.fortPlaced) {
    // Past the stronghold: filler ground; the level ends before anyone gets here
    w = 16;
    ground(c0, w);
    L.next = c0 + w;
    return;
  }

  // Between set pieces: a waterfall where the land climbs steeply, hill country more often in
  // mountainous themes, the odd military site, or else the theme's weighted pick
  let choice = pick(t.features, r);
  if (t.features.waterfall && L.h - terrainAt(L, c0 + 10) >= 4 && r() < 0.5) choice = "waterfall";
  else if (r() < 0.3 * t.relief * smoothstep(40, 120, c0)) {
    w = 12 + int(r, 18);
    natural(c0, w);
    for (let k = 2 + int(r, 4); k > 0; k--) deco(c0, w);
    if (t.top === GRASS)
      for (let k = int(r, 3); k > 0; k--)
        prop((["hedge", "flowers", "stump", "logs", "boulders"] as const)[int(r, 5)], c0 + int(r, w - 2));
    scatter(c0, w, 2 + int(r, 3));
    standing(c0, w);
    if (r() < 0.45) underground(c0, w);
    if (t.enemies > 0 && c0 > 70 && r() < 0.3)
      for (let k = 1 + int(r, 3); k > 0; k--) spawn("mine", c0 + 2 + int(r, w - 4));
    enemies(c0, w);
    barrels(c0, w);
    L.next = c0 + w;
    return;
  } else if (!t.ceiling && t.enemies > 0 && c0 > 50 && r() < 0.1) {
    // Gun emplacement: heavy weapons behind sandbags
    w = 12;
    ground(c0, w);
    const g = L.h;
    fill(L, c0 + 1, g - 1, 1, 1, SANDBAG);
    fill(L, c0 + w - 2, g - 1, 1, 1, SANDBAG);
    const guns: Spawn["kind"][] = ["flakgun", "howitzer", "samsite", "rocketpod"];
    spawn(guns[int(r, 4)], c0 + 4);
    if (r() < 0.6) spawn(guns[int(r, 4)], c0 + 8);
    prop(r() < 0.5 ? "ammocrate" : "camonet", c0 + 2);
    enemies(c0 + 2, w - 4, -1, 1.2);
    L.next = c0 + w;
    return;
  } else if (!t.ceiling && t.enemies > 0 && c0 > 50 && r() < 0.06) {
    // Bunker line: two sunken pillboxes with machine guns, and a trench between
    w = 22;
    ground(c0, w);
    const g = L.h;
    for (const bx of [c0 + 2, c0 + 13]) {
      fill(L, bx, g - 3, 7, 6, CONCRETE, CONCRETE);
      fill(L, bx + 1, g - 2, 5, 4, AIR, CONCRETE);
      // Firing slit facing the squad's approach
      put(L, bx, g - 1, AIR, CONCRETE);
      fill(L, bx, g - 4, 7, 1, SANDBAG);
      out.push({ kind: "mg", x: (bx + 3) * TILE + 4, y: (g + 2) * TILE });
      if (r() < 0.6) out.push({ kind: troop(), x: (bx + 5) * TILE, y: (g + 2) * TILE });
      L.props.push({ kind: "lamp", x: (bx + 3) * TILE + 4, y: (g - 2) * TILE + 1 });
    }
    fill(L, c0 + 9, g, 4, 2, AIR, t.fill);
    fill(L, c0 + 9, g + 2, 4, 1, WOOD);
    L.props.push({ kind: "wire", x: c0 * TILE, y: g * TILE, w: 2 * TILE });
    enemies(c0 + 9, 4, g + 2, 1.2);
    L.next = c0 + w;
    return;
  } else if (!t.ceiling && t.enemies > 0 && c0 > 80 && r() < 0.04) {
    // Missile launch pad with a gantry and a launcher truck
    w = 18;
    ground(c0, w);
    const g = L.h;
    fill(L, c0 + 1, g, w - 2, 1, CONCRETE);
    fill(L, c0 + 3, g - 1, 4, 1, HAZARD);
    for (let row = g - 10; row < g; row++) put(L, c0 + 8, row, LADDER);
    fill(L, c0 + 8, g - 11, 3, 1, METAL);
    placeProp(L, "missiletrailer", c0 + 11);
    L.props.push(
      { kind: "silo", x: (c0 + 5) * TILE, y: g * TILE },
      { kind: "antenna", x: (c0 + 9) * TILE + 4, y: (g - 11) * TILE, h: 24 },
      { kind: "alarm", x: (c0 + 10) * TILE, y: (g - 11) * TILE }
    );
    spawn("launcher", c0 + 14);
    enemies(c0 + 1, w - 2, -1, 1.2);
    L.next = c0 + w;
    return;
  } else if (!t.ceiling && t.enemies > 0 && c0 > 60 && r() < 0.07) {
    // Fenced watchtower outpost
    w = 12;
    ground(c0, w);
    const g = L.h;
    const tx = c0 + 4;
    const ht = 8 + int(r, 3);
    fill(L, tx, g - ht, 4, 1, WOOD);
    fill(L, tx - 1, g - ht - 5, 6, 1, TIN);
    put(L, tx, g - ht - 1, SANDBAG);
    put(L, tx + 3, g - ht - 1, SANDBAG);
    L.props.push({ kind: "legs", x: tx * TILE, y: (g - ht + 1) * TILE, w: 4 * TILE, h: (ht - 1) * TILE });
    L.props.push(
      { kind: "support", x: tx * TILE + 1, y: (g - ht - 4) * TILE, h: 4 * TILE },
      { kind: "support", x: (tx + 4) * TILE - 1, y: (g - ht - 4) * TILE, h: 4 * TILE }
    );
    climb(tx + 1, g - ht, g);
    spawn(r() < 0.5 ? "sniper" : troop(), tx + 2, g - ht);
    if (night) L.props.push({ kind: "searchlight", x: (tx + 4) * TILE, y: (g - ht) * TILE - 3 });
    for (const c of [c0, c0 + 1, c0 + w - 2, c0 + w - 1]) fill(L, c, g - 2, 1, 2, FENCE);
    prop(r() < 0.5 ? "radio" : "generator", c0 + 9);
    prop("crates", c0 + 2);
    enemies(c0 + 2, w - 4);
    L.next = c0 + w;
    return;
  }

  switch (choice) {
    case "metropolis": {
      // City blocks: buildings shoulder to shoulder with side streets, taller towers behind
      const specs = buildingsFor(t.backdrop);
      let c = c0;
      const street = (from: number, n: number) => {
        ground(from, n);
        for (let k = from; k < from + n; k++) put(L, k, L.h, ASPHALT, t.fill);
      };
      street(c, 4);
      c += 4;
      // Big cities get more blocks
      const many = t.backdrop === "city" || t.backdrop === "megacity";
      for (let n = many ? 7 + int(r, 7) : 4 + int(r, 5); n > 0; n--) {
        const spec =
          r() < 0.65 ? SPEC_BY_KEY.get(CITY_ESSENTIAL[int(r, CITY_ESSENTIAL.length)]) : specs[int(r, specs.length)];
        if (!spec || L.h - spec.fh * spec.floors[0] < 12) continue;
        const bw = building(L, c, spec, out, troop);
        for (let k = c; k < c + bw; k++) put(L, k, L.h, PAVEMENT, t.fill);
        c += bw;
        if (r() < (many ? 0.65 : 0.55)) continue;
        // A side street
        const gap = 3 + int(r, 4);
        street(c, gap);
        for (let j = 1 + int(r, 2); j > 0; j--)
          prop(STREET_KIT[int(r, STREET_KIT.length)], c + int(r, Math.max(1, gap - 2)));
        if (r() < 0.5) spawn("manhole", c + 1 + int(r, Math.max(1, gap - 2)));
        if (r() < 0.35) {
          // Sandbags and a machine gun at the corner
          put(L, c + 1, L.h - 1, SANDBAG);
          spawn("mg", c + 2);
        } else if (r() < 0.3) spawn(r() < 0.5 ? "apc" : "armoredcar", c + (gap >> 1));
        c += gap;
      }
      street(c, 3);
      c += 3;
      w = c - c0;
      towers(c0, w);
      if (w >= 14 && r() < 0.5) underground(c0, w);
      enemies(c0, w, -1, 0.6);
      break;
    }
    case "metro": {
      // City blocks over a metro line with two to four stations. Half the time the squad goes down
      // into the tunnel instead of along the street.
      const specs = buildingsFor(t.backdrop);
      const under = r() < 0.5;
      const g = L.h;
      const sw = 16;
      const stops: number[] = [];
      let c = c0;
      const street = (from: number, n: number, k = ASPHALT) => {
        ground(from, n, g);
        for (let x = from; x < from + n; x++) put(L, x, g, k, t.fill);
      };
      street(c, 4);
      c += 4;
      const n = 2 + int(r, 3);
      for (let i = 0; i < n; i++) {
        // Station square
        stops.push(c);
        street(c, sw, PAVEMENT);
        c += sw;
        if (i === n - 1) break;
        // Blocks between stations
        for (let b = 2 + int(r, 3); b > 0; b--) {
          const spec =
            r() < 0.6 ? SPEC_BY_KEY.get(CITY_ESSENTIAL[int(r, CITY_ESSENTIAL.length)]) : specs[int(r, specs.length)];
          if (!spec || g - spec.fh * spec.floors[0] < 12) continue;
          const bw = building(L, c, spec, out, troop);
          for (let x = c; x < c + bw; x++) put(L, x, g, PAVEMENT, t.fill);
          c += bw;
          if (r() < 0.5) {
            const gap = 3 + int(r, 3);
            street(c, gap);
            prop(STREET_KIT[int(r, STREET_KIT.length)], c + int(r, Math.max(1, gap - 2)));
            c += gap;
          }
        }
        street(c, 3);
        c += 3;
      }
      street(c, 4);
      c += 4;
      w = c - c0;
      L.h = g;
      // Tunnel from the first station to the last
      const tf = Math.min(ROWS - 10, g + 15);
      const top = tf - 6;
      const t0 = stops[0] + 1;
      const t1 = stops[n - 1] + sw - 2;
      for (let x = t0; x < t1; x++) {
        for (let row = top; row < tf; row++) {
          const i = cell(x, row);
          L.fg[i] = L.hp[i] = L.burn[i] = 0;
          L.bg[i] = CONCRETE;
        }
        // Reinforced roof and floor, so street craters can't break into the tunnel
        put(L, x, top - 1, REINFORCED, CONCRETE);
        put(L, x, tf, REINFORCED, CONCRETE);
        put(L, x, tf - 1, RAIL, CONCRETE);
        if (under) L.ground[x & MASK] = tf;
      }
      for (let x = t0 + 4; x < t1 - 2; x += 8) L.props.push({ kind: "lamp", x: px(x), y: top * TILE + 1 });
      const names = [...STATIONS].sort(() => r() - 0.5);
      stops.forEach((s0, i) => {
        // Platform: tiled walls, furniture and the station name
        for (let x = Math.max(t0, s0); x < Math.min(t1, s0 + sw); x++) {
          for (let row = top; row < tf - 1; row++) L.bg[cell(x, row)] = TILES;
          put(L, x, tf, PAVEMENT, CONCRETE);
        }
        placeAt(L, "platformbench", s0 + 3, tf);
        placeAt(L, "ticketmachine", s0 + 8, tf);
        placeAt(L, r() < 0.5 ? "vending" : "metromap", s0 + 11, tf);
        placeAt(L, "arrivalboard", s0 + 6, tf, (tf - top) * TILE - 9);
        L.props.push({
          kind: "sign",
          x: px(s0 + (sw >> 1)),
          y: (top + 2) * TILE,
          text: names[i % names.length],
          ink: "#ffffff",
          board: "#2a4a9a",
          neon: false,
        });
        for (let x = s0 + 2; x < s0 + sw - 1; x += 4) L.props.push({ kind: "lamp", x: px(x), y: top * TILE + 1 });
        // Stairwell to the street; the first and last are open when the squad goes underground
        const sx = i === 0 ? s0 + 2 : i === n - 1 ? t1 - 4 : s0 + 6;
        const open = under && (i === 0 || i === n - 1);
        for (let x = sx; x < sx + 4; x++)
          for (let row = open ? g : g + 1; row < top; row++) {
            const k = cell(x, row);
            L.fg[k] = L.hp[k] = 0;
            L.bg[k] = TILES;
          }
        if (!open) for (let x = sx; x < sx + 4; x++) put(L, x, g, METAL, t.fill);
        climb(sx + 3, open ? g : g + 1, tf);
        prop("metrosign", Math.max(c0, sx - 1));
        if (under) enemies(s0 + 2, sw - 4, tf, 0.7);
      });
      // One train per line, two on lines with three or more stations
      const stopsX = stops.map((s0) => (s0 + (sw >> 1)) * TILE);
      out.push({ kind: "metro", x: stopsX[0], y: tf * TILE, extra: [t0 * TILE, t1 * TILE, ...stopsX] });
      if (n >= 3) out.push({ kind: "metro", x: stopsX[n - 1], y: tf * TILE, extra: [t0 * TILE, t1 * TILE, ...stopsX] });
      towers(c0, w);
      if (!under) enemies(c0, w, g, 0.5);
      break;
    }
    case "railyard": {
      // Rail yard: freight on the main line, wagons on the sidings, signals and an armoured train
      w = 64 + int(r, 56);
      ground(c0, w);
      const g = L.h;
      for (let c = c0; c < c0 + w; c++) {
        put(L, c, g, GRAVEL, t.fill);
        put(L, c, g - 1, RAIL);
      }
      out.push({ kind: "freight", x: px(c0), y: g * TILE, extra: [c0 * TILE, (c0 + w) * TILE] });
      for (let c = c0 + 4; c < c0 + w - 8; c += 6 + int(r, 10))
        prop((["boxcar", "tankcar", "flatcar", "hopper", "boxcar"] as const)[int(r, 5)], c, true);
      prop("signalgantry", c0 + 2 + int(r, 6), true);
      prop("railtunnel", c0, true);
      prop("railsignal", c0 + 1);
      prop("railsignal", c0 + w - 2);
      prop(r() < 0.5 ? "watercrane" : "signalbox", c0 + (w >> 1), true);
      for (let c = c0 + 5; c < c0 + w; c += 12) prop("streetlamp", c);
      for (let k = 1 + int(r, 3); k > 0; k--)
        prop(
          (["pallets", "cablespool", "drums", "containerred", "containerblue", "forklift"] as const)[int(r, 6)],
          c0 + 3 + int(r, w - 6)
        );
      if (t.enemies > 0) out.push({ kind: "train", x: (c0 + w - 16) * TILE, y: g * TILE });
      enemies(c0, w, -1, 0.9);
      barrels(c0, w);
      break;
    }
    case "factory": {
      // Works buildings, industrial props behind and in the yard, and a pipe bridge
      w = 34 + int(r, 30);
      ground(c0, w);
      const g = L.h;
      for (let c = c0; c < c0 + w; c++) put(L, c, g, CONCRETE, t.fill);
      let c = c0 + 1;
      for (let k = 1 + int(r, 2); k > 0; k--) {
        const spec = SPEC_BY_KEY.get(WORKS[int(r, WORKS.length)]);
        if (!spec || g - spec.fh * spec.floors[0] <= 12 || c + spec.width[1] + 2 > c0 + w - 8) continue;
        c += building(L, c, spec, out, troop) + 1;
      }
      for (let k = c0; k < c0 + w - 4; k += 6 + int(r, 8))
        prop(
          (
            [
              "smokestack",
              "gasholder",
              "factoryshed",
              "storagetank",
              "pylon",
              "smokestack",
              "coolingtower",
              "pipebridge",
            ] as const
          )[int(r, 8)],
          k,
          true
        );
      for (let k = 2 + int(r, 3); k > 0; k--)
        prop(
          (
            [
              "transformer",
              "conveyor",
              "boiler",
              "coalheap",
              "forklift",
              "pallets",
              "containeryellow",
              "drums",
              "cablespool",
              "toxicdrum",
            ] as const
          )[int(r, 10)],
          c + int(r, Math.max(1, c0 + w - c - 3))
        );
      // Pipe bridge across the yard
      if (c0 + w - c > 14) {
        const py = g - 7;
        for (let x = c + 1; x < c0 + w - 2; x++) put(L, x, py, PIPE);
        L.props.push({ kind: "legs", x: (c + 1) * TILE, y: (py + 1) * TILE, w: (c0 + w - c - 3) * TILE, h: 6 * TILE });
      }
      enemies(c0, w, -1, 1);
      barrels(c0, w);
      break;
    }
    case "highway": {
      // Elevated expressway with traffic, over a street with parked vehicles and a convoy
      w = 40 + int(r, 30);
      ground(c0, w);
      const g = L.h;
      for (let c = c0; c < c0 + w; c++) put(L, c, g, ASPHALT, t.fill);
      const deck = g - 9;
      for (let c = c0 + 1; c < c0 + w - 1; c++) put(L, c, deck, REINFORCED, 0);
      for (let c = c0 + 3; c < c0 + w - 2; c += 10) backWall(L, c, deck + 1, 2, g - deck - 1, CONCRETE);
      for (let c = c0 + 3; c < c0 + w - 1; c += 9) L.props.push({ kind: "streetlight", x: px(c), y: deck * TILE });
      for (let c = c0 + 6; c < c0 + w - 10; c += 24 + int(r, 10)) placeAt(L, "hwysign", c, deck);
      climb(c0 + 1, deck, g);
      out.push({ kind: "motorway", x: px(c0 + 1), y: deck * TILE, extra: [(c0 + 1) * TILE, (c0 + w - 1) * TILE, 0] });
      enemies(c0 + 2, w - 4, deck, 0.4);
      for (let c = c0 + 2; c < c0 + w; c += 8) L.props.push({ kind: "streetlight", x: px(c), y: g * TILE });
      for (let k = 2 + int(r, 3); k > 0; k--)
        prop(
          (
            [
              "sedan",
              "sedanblue",
              "hatchback",
              "taxi",
              "van",
              "deliverytruck",
              "pickup",
              "bus",
              "carwreck",
              "jerseybarrier",
              "trafficbarrels",
            ] as const
          )[int(r, 11)],
          c0 + 3 + int(r, w - 8)
        );
      standing(c0, w);
      if (r() < 0.6) prop("billboard", c0 + int(r, w - 4), true);
      spawn(r() < 0.5 ? "armoredcar" : "apc", c0 + w - 6);
      if (r() < 0.5) spawn((["truck", "medtank", "artytruck", "jeep"] as const)[int(r, 4)], c0 + w - 14);
      enemies(c0, w, -1, 0.8);
      break;
    }
    case "airfield": {
      // Airfield: runway, control tower, hangar, parked aircraft and air defence
      w = 56 + int(r, 24);
      ground(c0, w);
      const g = L.h;
      for (let c = c0; c < c0 + w; c++) put(L, c, g, ASPHALT, t.fill);
      let c = c0 + 1;
      const tower = SPEC_BY_KEY.get("controltower");
      if (tower && g - tower.fh * tower.floors[0] > 12) c += building(L, c, tower, out, troop);
      for (let k = c + 2; k < c0 + w - 4; k += 6) prop("runwaylight", k);
      for (let k = 2 + int(r, 3); k > 0; k--)
        prop(r() < 0.6 ? "parkedjet" : "parkedheli", c + 3 + int(r, Math.max(1, c0 + w - c - 24)));
      prop("cargoplane", c + 10 + int(r, 20), true);
      prop("windsock", c0 + w - 20);
      const hangar = SPEC_BY_KEY.get("hangar");
      if (hangar && g - hangar.fh * hangar.floors[0] > 12) building(L, c0 + w - 18, hangar, out, troop);
      spawn(r() < 0.5 ? "flakgun" : "samsite", c0 + (w >> 1));
      spawn("aatank", c0 + w - 24);
      enemies(c0, w, -1, 0.9);
      break;
    }
    case "downtown": {
      // A street of buildings with street furniture between them, maybe something underground
      const specs = buildingsFor(t.backdrop);
      let c = c0;
      const road = (from: number, n: number) => {
        ground(from, n);
        for (let k = from; k < from + n; k++) put(L, k, L.h, ASPHALT, t.fill);
      };
      road(c, 3);
      c += 3;
      for (let n = 1 + int(r, t.backdrop === "city" ? 3 : 2); n > 0; n--) {
        const spec = specs[int(r, specs.length)];
        if (L.h - spec.fh * spec.floors[0] < 12) break;
        const bw = building(L, c, spec, out, troop);
        for (let k = c; k < c + bw; k++) put(L, k, L.h, PAVEMENT, t.fill);
        c += bw;
        const gap = 4 + int(r, 5);
        road(c, gap);
        if (r() < 0.6) spawn("manhole", c + 1 + int(r, Math.max(1, gap - 2)));
        for (let j = 1 + int(r, 2); j > 0; j--) prop(pickAsset(r, t.backdrop), c + int(r, Math.max(1, gap - 2)));
        c += gap;
      }
      w = c - c0;
      towers(c0, w);
      if (w >= 14 && r() < 0.55) underground(c0, w);
      enemies(c0, 3);
      break;
    }
    case "ocean": {
      // Open sea; the next feature may carry it on (see the openSea branch above)
      const ow = 120 + int(r, 120);
      w = ow + 4;
      ground(c0, 4);
      sea(c0 + 4, ow);
      L.openSea = true;
      break;
    }
    case "lake": {
      // A deep lake to swim or jetpack over
      const lw = 18 + int(r, 22);
      w = lw + 6;
      ground(c0, 3);
      const g = L.h;
      for (let c = c0 + 3; c < c0 + 3 + lw; c++) {
        const k = c - c0 - 3;
        const depth = Math.round(Math.sin((Math.PI * (k + 0.5)) / lw) * (8 + (lw >> 2)));
        const floor = g + 1 + Math.max(1, depth);
        column(L, c, floor);
        put(L, c, floor, MUD);
        for (let row = g + 1; row < floor; row++) put(L, c, row, WATER, 0);
        if (r() < 0.3 && floor > g + 4) placeAt(L, "seaweed", c, floor);
      }
      ground(c0 + 3 + lw, 3);
      for (const c of [c0 + 1, c0 + 4 + lw]) soft(c, g - 1, LEAF);
      deco(c0, 3);
      if (r() < 0.6) out.push({ kind: "boat", x: (c0 + 3 + (lw >> 1)) * TILE, y: (g + 1) * TILE });
      break;
    }
    case "massif": {
      // A mountain 90 to 130 rows high: terraces and cliffs up, ladders on the tall cliffs, an outpost
      // on the summit, then big drops down the far side
      const g0 = L.h;
      const peak = Math.max(10, g0 - (90 + int(r, 40)));
      let c = c0;
      ground(c, 4);
      c += 4;
      const ledge = (lw: number) => {
        for (let x = c; x < c + lw; x++) column(L, x, L.h, true);
        const g = L.h;
        const alt = BASE_ROW - g;
        // Surface by altitude: moss rock low down, scree banks higher, then snow over ice
        const scree = alt > 20 && alt <= 62 && r() < 0.45;
        for (let x = c; x < c + lw; x++) {
          if (alt > 62) {
            put(L, x, g, SNOW);
            if (r() < 0.25) put(L, x, g + 1, ICE);
            else if (alt > 85 && r() < 0.6) put(L, x, g + 1, GLACIER);
          } else if (scree) {
            put(L, x, g, SCREE);
            if (r() < 0.5) put(L, x, g + 1, SCREE);
          } else if (alt <= 30) put(L, x, g, r() < 0.7 ? MOSSROCK : GRASS);
          else if (r() < 0.3) put(L, x, g, SHALE);
        }
        if (alt < 40 && lw >= 4 && r() < 0.5) pine(c + 1 + int(r, lw - 2), g);
        else if (r() < 0.3) prop("boulders", c + int(r, lw));
        if (lw >= 5 && r() < 0.15) L.props.push({ kind: "tent", x: px(c + 2), y: g * TILE });
        if (r() < 0.55) enemies(c, lw, g, 0.6);
        if (alt > 30 && r() < 0.18) spawn("sniper", c + lw - 1, g);
        c += lw;
      };
      while (L.h > peak) {
        const ch = Math.min(L.h - peak, r() < 0.22 ? 9 + int(r, 6) : 2 + int(r, 5));
        if (ch >= 8) climb(c - 1, L.h - ch, L.h);
        L.h -= ch;
        ledge(3 + int(r, 4));
      }
      // Summit: a snowfield over glacier, with the enemy outpost
      const sw = 14 + int(r, 8);
      const s0 = c;
      for (let x = s0; x < s0 + sw; x++) column(L, x, L.h, true);
      for (let x = s0; x < s0 + sw; x++) {
        put(L, x, L.h, SNOW);
        for (let d = 1, n = 1 + int(r, 3); d <= n; d++) put(L, x, L.h + d, GLACIER);
      }
      // Mine adit under the summit
      if (sw >= 16) {
        const ar = L.h + 6;
        for (let x = s0 + 2; x < s0 + sw - 2; x++) for (let row = ar; row < ar + 4; row++) put(L, x, row, AIR, GRANITE);
        for (let x = s0 + 2; x < s0 + sw - 2; x++) put(L, x, ar + 3, RAIL, GRANITE);
        for (let x = s0 + 3; x < s0 + sw - 3; x += 5)
          L.props.push({ kind: "support", x: px(x), y: ar * TILE, h: 3 * TILE });
        spawn("minecart", s0 + 4 + int(r, sw - 9), ar + 4);
        placeAt(L, "orepile", s0 + sw - 6, ar + 4);
        L.props.push({ kind: "lamp", x: px(s0 + (sw >> 1)), y: ar * TILE + 1 });
      }
      L.props.push(
        { kind: "flag", x: px(s0 + 2), y: L.h * TILE },
        { kind: "radar", x: px(s0 + (sw >> 1)), y: L.h * TILE },
        { kind: "antenna", x: px(s0 + sw - 3), y: L.h * TILE, h: 40 }
      );
      if (night) L.props.push({ kind: "searchlight", x: px(s0 + 5), y: L.h * TILE - 3 });
      put(L, s0 + 4, L.h - 1, SANDBAG);
      fill(L, s0 + 7, L.h - 2, 1, 2, SANDBAG);
      spawn("mg", s0 + 5);
      spawn(r() < 0.5 ? "turret" : "mortar", s0 + sw - 5);
      enemies(s0, sw, L.h, 1.2);
      c += sw;
      // Down the far side
      while (L.h < g0) {
        L.h += Math.min(g0 - L.h, 5 + int(r, 10));
        ledge(3 + int(r, 3));
      }
      ground(c, 4);
      c += 4;
      w = c - c0;
      break;
    }
    case "abyss": {
      // A sinkhole dropping about 60 to 80 rows into a vast cavern, and a chimney to climb back out
      const g = L.h;
      const deep = Math.min(ROWS - 10, g + 58 + int(r, 22));
      const rock = t.deep;
      /** Hollow out a column from row `top` to the walkable `floor`; `open` shows sky at the top of a shaft. */
      const hollow = (c: number, top: number, floor: number, open = false) => {
        column(L, c, g);
        for (let row = top; row < floor; row++) {
          const i = cell(c, row);
          L.fg[i] = L.hp[i] = L.burn[i] = 0;
          L.bg[i] = open && row < g + 3 ? 0 : rock;
        }
        for (let row = floor; row < floor + 3; row++) if (!L.fg[cell(c, row)]) put(L, c, row, rock, rock);
        L.ground[c & MASK] = floor;
      };
      /** Row of the cavern roof above column x. */
      const roofOf = (x: number) => {
        let row = L.ground[x & MASK] - 1;
        while (row > g && !L.fg[cell(x, row)]) row--;
        return row;
      };
      let c = c0;
      ground(c, 3);
      c += 3;
      // Sinkhole with vines at the top and a ladder down the far wall
      const sw = 9 + int(r, 5);
      const h0 = c;
      for (let k = 0; k < sw; k++) hollow(h0 + k, g, deep, true);
      /** Noise bulges of rock out from a shaft wall at x, with `dir` pointing into the shaft. */
      const rough = (x: number, dir: 1 | -1, top: number, bottom: number, salt: number) => {
        for (let row = top; row < bottom; row++) {
          const bulge = Math.round(noise2(L.seed + salt, row / 5, x / 7) * 2.6);
          for (let k = 0; k < bulge; k++) put(L, x + dir * k, row, rock, rock);
        }
      };
      // Only the near wall is rough; the far wall stays straight for the ladder
      rough(h0, 1, g + 3, deep - 14, 141);
      // Ledges on the near wall only: on the far wall the squad would stand on them instead of dropping down
      for (let row = g + 8; row < deep - 14; row += 9 + int(r, 6))
        for (let k = 0, lw = 2 + int(r, 2); k < lw; k++) put(L, h0 + k, row, rock, rock);
      for (let k = 0; k < sw; k++)
        if (r() < 0.5) for (let d = 0, n = 2 + int(r, 4); d < n; d++) soft(h0 + k, g + d, VINE, h0, h0 + sw);
      climb(h0 + sw - 1, g, deep);
      c += sw;
      // Cavern
      const cw = 70 + int(r, 60);
      const cs = c;
      for (let k = 0; k < cw; k++) {
        const x = cs + k;
        // Floor noise fades out at the ends to meet the sinkhole and chimney floors
        const ends = Math.min(1, k / 8, (cw - 1 - k) / 8);
        const f = deep + Math.round((noise2(L.seed + 131, x / 14, 0) - 0.5) * 6 * ends);
        hollow(x, f - 9 - Math.round(noise2(L.seed + 137, x / 9, 0) * 6), f);
      }
      for (let k = 2; k < cw - 2; k++) {
        const x = cs + k;
        const f = L.ground[x & MASK];
        const roof = roofOf(x);
        if (r() < 0.18) for (let d = 1, n = 1 + int(r, 3); d <= n; d++) put(L, x, roof + d, rock, rock);
        else if (r() < 0.07) put(L, x, roof, CRYSTAL);
        if (r() < 0.07) put(L, x, f - 1, rock, rock);
        else if (r() < 0.05) put(L, x, f, CRYSTAL);
        if (k % 11 === 5) L.props.push({ kind: "lamp", x: px(x), y: (roof + 1) * TILE });
        if (k % 23 === 12) L.props.push({ kind: "brazier", x: px(x), y: f * TILE });
      }
      // Amethyst geodes in the roof; glowing fungus patches with mushrooms on the floor
      for (let n = 2 + int(r, 3); n > 0; n--) {
        const x = cs + 4 + int(r, cw - 8);
        const row = roofOf(x) - 1 - int(r, 3);
        for (let dx = -2; dx <= 2; dx++)
          for (let dy = -1; dy <= 1; dy++)
            if (Math.abs(dx) + Math.abs(dy) < 3 && L.fg[cell(x + dx, row + dy)])
              put(L, x + dx, row + dy, AMETHYST, rock);
      }
      for (let n = 2 + int(r, 3); n > 0; n--) {
        const x0 = cs + 6 + int(r, cw - 16);
        for (let x = x0; x < x0 + 5 + int(r, 6); x++) {
          const f = L.ground[x & MASK];
          if (L.fg[cell(x, f)] && L.fg[cell(x, f)] !== CRYSTAL) put(L, x, f, MYCELIUM, rock);
        }
        prop("mushrooms", x0 + 1);
      }
      // Lava pits with obsidian beneath and magma rims
      for (let n = 1 + int(r, 2); n > 0; n--) {
        const x0 = cs + 10 + int(r, cw - 24);
        const f = L.ground[x0 & MASK];
        for (let x = x0; x < x0 + 3; x++) {
          const i = cell(x, f);
          L.fg[i] = L.hp[i] = 0;
          L.bg[i] = rock;
          put(L, x, f + 1, LAVA);
          put(L, x, f + 2, LAVA);
          put(L, x, f + 3, OBSIDIAN, rock);
          L.ground[x & MASK] = f + 3;
        }
        for (const x of [x0 - 1, x0 + 3])
          for (let row = f; row < f + 4; row++)
            if (SOLID[L.fg[cell(x, row)]] && L.fg[cell(x, row)] !== OBSIDIAN) put(L, x, row, MAGMA, rock);
      }
      // Sometimes an underground lake
      if (r() < 0.5) {
        const lw = 8 + int(r, 6);
        const x0 = cs + 20 + int(r, Math.max(1, cw - 40));
        const f = L.ground[x0 & MASK];
        for (let x = x0; x < x0 + lw; x++) {
          const own = L.ground[x & MASK];
          for (let row = Math.min(own, f); row < f; row++) {
            const i = cell(x, row);
            L.fg[i] = L.hp[i] = 0;
            L.bg[i] = rock;
          }
          for (let row = f; row < f + 3; row++) put(L, x, row, WATER);
          put(L, x, f + 3, rock, rock);
          L.ground[x & MASK] = f + 3;
        }
      }
      // Old mine workings
      const m0 = cs + 6 + int(r, Math.max(1, cw - 30));
      for (let x = m0; x < m0 + 20; x += 6) {
        const roof = roofOf(x);
        L.props.push({ kind: "support", x: px(x), y: (roof + 1) * TILE, h: (L.ground[x & MASK] - roof - 1) * TILE });
      }
      for (let x = m0; x < m0 + 20; x++)
        if (!L.fg[cell(x, L.ground[x & MASK] - 1)]) put(L, x, L.ground[x & MASK] - 1, RAIL);
      spawn("minecart", m0 + 4);
      if (r() < 0.6) spawn("minecart", m0 + 14);
      prop(r() < 0.5 ? "carttipped" : "minecart", m0 + 21);
      prop("orepile", m0 + 15);
      prop(r() < 0.5 ? "railbuffer" : "dynamite", m0 - 2);
      // Garrison
      enemies(cs, cw, -1, 1.1);
      spawn("mg", cs + (cw >> 1));
      c += cw;
      // Way out: a chimney with a ladder up the far wall
      const chw = 7;
      for (let k = 0; k < chw; k++) hollow(c + k, g, deep, true);
      climb(c + chw - 1, g, deep);
      // Rough near wall; the far wall carries the ladder
      rough(c, 1, g + 3, deep - 14, 149);
      c += chw;
      // Solid granite beside the exit, so the climb has no holes in it
      for (let x = c; x < c + 4; x++) column(L, x, g, true);
      c += 4;
      w = c - c0;
      L.h = g;
      break;
    }
    case "mineshaft": {
      // A working mine: down a shaft to a long gallery with carts and ore, worked-out drifts above and
      // below it, and up a far shaft
      const g = L.h;
      const floor = Math.min(ROWS - 14, g + 20 + int(r, 12));
      const rock = t.deep;
      const gh = 6;
      /** Cut a column out from row `top` to the walkable floor; `open` shows sky at the top of a shaft. */
      const dig = (x: number, top: number, open: boolean) => {
        column(L, x, g, false, false);
        for (let row = top; row < floor; row++) {
          const i = cell(x, row);
          L.fg[i] = L.hp[i] = L.burn[i] = 0;
          L.bg[i] = open && row < g + 3 ? 0 : row >= floor - gh ? WOOD : rock;
        }
        L.ground[x & MASK] = floor;
      };
      let c = c0;
      ground(c, 4);
      prop("winch", c0);
      c += 4;
      // Entry shaft with a ladder and the cage at the bottom
      const sw = 7;
      const s0 = c;
      for (let k = 0; k < sw; k++) dig(s0 + k, g, true);
      climb(s0 + sw - 1, g, floor);
      placeProp(L, "headframe", s0 + 2, true);
      placeAt(L, "minecage", s0 + 1, floor);
      c += sw;
      // Main gallery with rails and timber sets
      const gw = 40 + int(r, 34);
      const gs = c;
      for (let k = 0; k < gw; k++) dig(gs + k, floor - gh, false);
      for (let k = 0; k < gw; k++) put(L, gs + k, floor - 1, RAIL, WOOD);
      for (let k = 2; k < gw - 1; k += 5) {
        L.props.push({ kind: "support", x: px(gs + k), y: (floor - gh) * TILE, h: (gh - 1) * TILE });
        for (let dx = -1; dx <= 1; dx++) put(L, gs + k + dx, floor - gh - 1, WOOD, rock);
      }
      for (let k = 6; k < gw - 2; k += 9) L.props.push({ kind: "lamp", x: px(gs + k), y: (floor - gh) * TILE + 1 });
      // Ore veins in the gallery's roof and floor
      for (let n = 3 + int(r, 4); n > 0; n--) {
        const x = gs + int(r, gw);
        const row = r() < 0.5 ? floor - gh - 2 - int(r, 3) : floor + 1 + int(r, 2);
        const ore = r() < 0.4 ? COAL : r() < 0.6 ? COPPER : r() < 0.8 ? IRON : GOLD;
        for (let dx = 0; dx < 2 + int(r, 2); dx++)
          for (let dy = 0; dy < 2; dy++) if (L.fg[cell(x + dx, row + dy)]) put(L, x + dx, row + dy, ore, rock);
      }
      // Carts, mining kit and guards
      for (let k = 0, n = 2 + int(r, 2); k < n; k++) spawn("minecart", gs + 3 + int(r, gw - 6), floor);
      const kit = [
        "cartcoal",
        "orepile",
        "toolsrack",
        "drillrig",
        "timbers",
        "dynamite",
        "carttipped",
        "orepile",
        "cartcoal",
      ];
      for (let k = 0, n = 4 + int(r, 4); k < n; k++)
        placeAt(L, kit[int(r, kit.length)], gs + 2 + int(r, gw - 4), floor);
      placeAt(L, "railbuffer", gs + gw - 2, floor);
      enemies(gs + 4, gw - 8, floor, 1);
      if (t.enemies > 0 && r() < 0.6) spawn("mg", gs + (gw >> 1), floor);
      barrels(gs + 8, gw - 16);
      // Sealed-off drifts above and below the gallery
      const drift = (x0: number, dw: number, top: number) => {
        for (let x = x0; x < x0 + dw; x++)
          for (let row = top; row < top + 4; row++) {
            const i = cell(x, row);
            L.fg[i] = L.hp[i] = 0;
            L.bg[i] = WOOD;
          }
        for (let x = x0; x < x0 + dw; x++) put(L, x, top + 3, RAIL, WOOD);
        for (let x = x0 + 1; x < x0 + dw - 1; x += 5)
          L.props.push({ kind: "support", x: px(x), y: top * TILE, h: 3 * TILE });
        placeAt(L, r() < 0.5 ? "cartcoal" : "minecart", x0 + 2 + int(r, Math.max(1, dw - 6)), top + 4);
        L.props.push({ kind: "lamp", x: px(x0 + (dw >> 1)), y: top * TILE + 1 });
      };
      if (floor + 9 < ROWS - 3) drift(gs + 4 + int(r, 10), 10 + int(r, 8), floor + 4);
      if (floor - gh - 12 > g + 4) drift(gs + gw - 26 + int(r, 8), 10 + int(r, 8), floor - gh - 9);
      c += gw;
      // Exit shaft with a ladder, and the ore bin at the top
      const ew = 7;
      for (let k = 0; k < ew; k++) dig(c + k, g, true);
      climb(c + ew - 1, g, floor);
      placeProp(L, "headframe", c + 2, true);
      c += ew;
      // Solid ground beside the exit, so the climb has no holes in it
      for (let x = c; x < c + 5; x++) column(L, x, g, true);
      prop("orebin", c + 1);
      c += 5;
      w = c - c0;
      L.h = g;
      break;
    }
    case "forest": {
      // Dense woods with undergrowth
      w = 18 + int(r, 14);
      natural(c0, w);
      for (let c = c0 + 2; c < c0 + w - 2; c += 3 + int(r, 3)) {
        if (t.decor === "pines" || t.top === SNOW) pine(c, gr(c));
        else if (t.decor === "deadwood") deadTree(c, gr(c));
        else bigTree(c, gr(c), c0, c0 + w);
      }
      for (let c = c0; c < c0 + w; c++) if (r() < 0.35) soft(c, gr(c) - 1, LEAF);
      for (let k = 1 + int(r, 3); k > 0; k--)
        prop((["stump", "logs", "flowers", "hedge"] as const)[int(r, 4)], c0 + int(r, w - 2));
      enemies(c0, w, -1, 1.1);
      break;
    }
    case "railway": {
      // Track with freight cars on the sidings, freight trains and an armoured train
      w = 44 + int(r, 20);
      ground(c0, w);
      const g = L.h;
      for (let c = c0; c < c0 + w; c++) put(L, c, g - 1, RAIL);
      out.push({ kind: "freight", x: px(c0), y: g * TILE, extra: [c0 * TILE, (c0 + w) * TILE] });
      for (let c = c0 + 3; c < c0 + w - 8; c += 7 + int(r, 6))
        prop((["boxcar", "tankcar", "flatcar", "boxcar"] as const)[int(r, 4)], c, true);
      prop("railsignal", c0 + 1);
      prop(r() < 0.5 ? "watertower" : "crane", c0 + (w >> 1), true);
      out.push({ kind: "train", x: (c0 + w - 12) * TILE, y: g * TILE });
      enemies(c0, w, -1, 0.8);
      break;
    }
    case "nuclear": {
      // Nuclear power station: cooling towers, reactor dome, hardened halls, switchyard, fence and air defence
      w = 48 + int(r, 18);
      ground(c0, w);
      const g = L.h;
      for (let c = c0; c < c0 + w; c++) put(L, c, g, CONCRETE, t.fill);
      prop("coolingtower", c0 + 1, true);
      prop("coolingtower", c0 + 9, true);
      if (r() < 0.6) prop("coolingtower", c0 + w - 14, true);
      prop("reactordome", c0 + 18, true);
      let c = c0 + 2;
      for (const key of ["reactor", "powerstation"]) {
        const spec = SPEC_BY_KEY.get(key);
        if (spec && g - spec.fh * spec.floors[0] > 12 && c + spec.width[1] + 2 < c0 + w - 10)
          c += building(L, c, spec, out, troop) + 1;
      }
      // Switchyard
      for (let k = c; k < c0 + w - 4; k += 5) prop(r() < 0.6 ? "transformer" : "hazardsign", k);
      for (let k = c + 2; k < c0 + w - 2; k += 12) prop("pylon", k, true);
      for (let k = c; k < c0 + w - 2; k += 6) L.props.push({ kind: "streetlight", x: px(k), y: g * TILE });
      for (const fx of [c0, c0 + w - 1]) fill(L, fx, g - 2, 1, 2, FENCE);
      for (let k = 2 + int(r, 3); k > 0; k--)
        prop(r() < 0.7 ? "toxicdrum" : "hazardsign", c + int(r, Math.max(1, c0 + w - c - 2)));
      spawn(r() < 0.5 ? "samsite" : "flakgun", c0 + w - 4);
      enemies(c0, w, -1, 1);
      break;
    }
    case "boulevard": {
      // A long avenue with two-way traffic, traffic lights, bus stops and street furniture
      w = 64 + int(r, 56);
      ground(c0, w);
      const g = L.h;
      for (let c = c0; c < c0 + w; c++) put(L, c, g, (c - c0) % 18 < 2 ? PAVEMENT : ASPHALT, t.fill);
      const lights: number[] = [];
      const stops: number[] = [];
      for (let c = c0 + 9; c < c0 + w - 6; c += 17 + int(r, 10)) {
        L.props.push({ kind: "stoplight", x: px(c), y: g * TILE });
        lights.push(px(c));
      }
      for (let c = c0 + 14; c < c0 + w - 8; c += 26 + int(r, 14)) if (placeProp(L, "busstop", c)) stops.push(px(c) + 4);
      for (let c = c0 + 3; c < c0 + w; c += 9) L.props.push({ kind: "streetlight", x: px(c), y: g * TILE });
      for (let c = c0 + 5; c < c0 + w - 3; c += 4 + int(r, 6)) prop(STREET_KIT[int(r, STREET_KIT.length)], c);
      // Maybe one building on the avenue itself
      const specs = buildingsFor(t.backdrop);
      if (r() < 0.6) {
        const spec =
          r() < 0.6 ? SPEC_BY_KEY.get(CITY_ESSENTIAL[int(r, CITY_ESSENTIAL.length)]) : specs[int(r, specs.length)];
        const at = c0 + (w >> 1) + int(r, 8);
        if (spec && g - spec.fh * spec.floors[0] > 12 && at + spec.width[1] + 3 < c0 + w)
          building(L, at, spec, out, troop);
      }
      towers(c0, w);
      out.push({
        kind: "road",
        x: px(c0),
        y: g * TILE,
        extra: [c0 * TILE, (c0 + w) * TILE, lights.length, ...lights, ...stops],
      });
      enemies(c0, w, -1, 0.6);
      if (t.enemies > 0 && r() < 0.5) spawn(r() < 0.5 ? "apc" : "armoredcar", c0 + w - 8);
      break;
    }
    case "park": {
      // City park with a pond, big trees, lamps and park furniture
      w = 40 + int(r, 34);
      ground(c0, w);
      const g = L.h;
      for (let c = c0; c < c0 + w; c++) put(L, c, g, GRASS, t.fill);
      // Pond
      const pw = 7 + int(r, 6);
      const p0 = c0 + 6 + int(r, Math.max(1, w - pw - 12));
      for (let x = p0; x < p0 + pw; x++) {
        column(L, x, g + 2);
        put(L, x, g + 2, MUD);
        put(L, x, g, WATER, 0);
        put(L, x, g + 1, WATER, 0);
        L.ground[x & MASK] = g + 2;
        L.crust[x & MASK] = g;
      }
      placeAt(L, "reeds", p0, g + 2);
      for (let c = c0 + 2; c < c0 + w - 2; c += 6 + int(r, 5))
        if (c < p0 - 2 || c > p0 + pw + 1) bigTree(c, g, c0, c0 + w);
      for (let c = c0 + 1; c < c0 + w - 1; c += 7) L.props.push({ kind: "streetlight", x: px(c), y: g * TILE });
      const kit = [
        "fountain",
        "statue",
        "gazebo",
        "playground",
        "bench",
        "bench",
        "flowerbed",
        "flowerbed",
        "picnictable",
        "icecreamcart",
        "hedge",
        "flowers",
      ];
      for (let k = 6 + int(r, 5); k > 0; k--) prop(kit[int(r, kit.length)], c0 + 1 + int(r, w - 3));
      towers(c0, w);
      enemies(c0, w, -1, 0.5);
      break;
    }
    case "citybridge": {
      // A river through the city crossed by a suspension bridge, with traffic on the deck and boats below
      const rw = 24 + int(r, 22);
      w = rw + 12;
      ground(c0, 6);
      const g = L.h;
      for (let c = c0; c < c0 + 6; c++) put(L, c, g, ASPHALT, t.fill);
      for (let c = c0 + 6; c < c0 + 6 + rw; c++) {
        column(L, c, g + 11);
        put(L, c, g + 11, MUD);
        for (let row = g + 1; row <= g + 10; row++) put(L, c, row, WATER, 0);
        put(L, c, g, REINFORCED, 0);
        L.ground[c & MASK] = g;
        L.crust[c & MASK] = g;
      }
      for (let c = c0 + 10; c < c0 + 6 + rw - 2; c += 11)
        for (let row = g + 1; row <= g + 10; row++) put(L, c, row, CONCRETE, 0);
      ground(c0 + 6 + rw, 6, g);
      for (let c = c0 + 6 + rw; c < c0 + w; c++) put(L, c, g, ASPHALT, t.fill);
      // Towers and cables
      const ta = c0 + 9;
      const tb = c0 + 6 + rw - 4;
      for (const tx of [ta, tb]) backWall(L, tx, g - 22, 2, 22, STEEL);
      const top = (g - 21) * TILE;
      L.props.push({
        kind: "cable",
        x: ta * TILE + 8,
        y: top,
        x2: tb * TILE + 8,
        y2: top,
        sag: 17 * TILE,
        deck: g * TILE,
      });
      L.props.push({
        kind: "cable",
        x: (c0 + 1) * TILE,
        y: g * TILE - 6,
        x2: ta * TILE + 8,
        y2: top,
        sag: 3 * TILE,
        deck: g * TILE,
      });
      L.props.push({
        kind: "cable",
        x: tb * TILE + 8,
        y: top,
        x2: (c0 + w - 1) * TILE,
        y2: g * TILE - 6,
        sag: 3 * TILE,
        deck: g * TILE,
      });
      for (let c = c0 + 2; c < c0 + w; c += 8) L.props.push({ kind: "streetlight", x: px(c), y: g * TILE });
      out.push({ kind: "road", x: px(c0), y: g * TILE, extra: [c0 * TILE, (c0 + w) * TILE, 0] });
      for (let k = 1 + int(r, 2); k > 0; k--)
        out.push({ kind: "boat", x: (c0 + 8 + int(r, rw - 4)) * TILE, y: (g + 1) * TILE });
      enemies(c0, w, g, 0.5);
      break;
    }
    case "waterplant": {
      // Waterworks: pumping station, sunken settling basins, tanks and pipe runs
      w = 42 + int(r, 24);
      ground(c0, w);
      const g = L.h;
      for (let c = c0; c < c0 + w; c++) put(L, c, g, CONCRETE, t.fill);
      let c = c0 + 1;
      const spec = SPEC_BY_KEY.get("waterworks");
      if (spec && g - spec.fh * spec.floors[0] > 12) c += building(L, c, spec, out, troop) + 1;
      // Settling basins, each walled
      for (let n = 2 + int(r, 2); n > 0 && c + 8 < c0 + w - 4; n--) {
        put(L, c, g - 1, CONCRETE);
        for (let x = c + 1; x < c + 7; x++) {
          column(L, x, g + 3);
          for (let row = g; row < g + 3; row++) put(L, x, row, WATER, CONCRETE);
          put(L, x, g + 3, CONCRETE);
          L.ground[x & MASK] = g + 3;
          L.crust[x & MASK] = g;
        }
        put(L, c + 7, g - 1, CONCRETE);
        c += 9;
      }
      prop("watertower", c0 + 2, true);
      for (let k = c0 + 4; k < c0 + w - 6; k += 10 + int(r, 8)) prop(r() < 0.5 ? "storagetank" : "clarifier", k, true);
      for (let k = 2 + int(r, 2); k > 0; k--)
        prop(
          (["clarifier", "valve", "generator", "toxicdrum", "pallets"] as const)[int(r, 5)],
          c + int(r, Math.max(1, c0 + w - c - 4))
        );
      const py = g - 6;
      for (let x = c0 + 2; x < c0 + w - 2; x++) if (!L.fg[cell(x, py)]) put(L, x, py, PIPE);
      for (let x = c0 + 3; x < c0 + w - 2; x += 8)
        L.props.push({ kind: "support", x: px(x), y: (py + 1) * TILE, h: 5 * TILE });
      for (let k = c0 + 1; k < c0 + w; k += 9) L.props.push({ kind: "streetlight", x: px(k), y: g * TILE });
      enemies(c0, w, -1, 0.8);
      break;
    }
    case "station": {
      // Railway station: the building, a platform under a canopy, and passenger trains calling
      w = 56 + int(r, 30);
      ground(c0, w);
      const g = L.h;
      for (let c = c0; c < c0 + w; c++) {
        put(L, c, g, PAVEMENT, t.fill);
        put(L, c, g - 1, RAIL);
      }
      let c = c0 + 1;
      const spec = SPEC_BY_KEY.get(r() < 0.5 ? "trainstation" : "grandstation");
      if (spec && g - spec.fh * spec.floors[0] > 12) c += building(L, c, spec, out, troop) + 1;
      const s0 = c;
      const s1 = c0 + w - 2;
      for (let x = s0; x < s1; x++) put(L, x, g - 9, x % 2 ? GLASS : TIN);
      for (let x = s0 + 1; x < s1; x += 6) L.props.push({ kind: "support", x: px(x), y: (g - 8) * TILE, h: 8 * TILE });
      for (let x = s0 + 3; x < s1 - 3; x += 7 + int(r, 5))
        placeAt(L, (["platformbench", "ticketmachine", "vending", "bench", "newsstand"] as const)[int(r, 5)], x, g);
      placeAt(L, "arrivalboard", s0 + 4, g, 8 * TILE - 9);
      placeAt(L, "arrivalboard", s1 - 10, g, 8 * TILE - 9);
      L.props.push({
        kind: "sign",
        x: px(s0 + ((s1 - s0) >> 1)),
        y: (g - 6) * TILE,
        text: STATIONS[int(r, STATIONS.length)],
        ink: "#ffffff",
        board: "#1e2a3a",
        neon: false,
      });
      for (let x = s0; x < s1; x += 8) L.props.push({ kind: "lamp", x: px(x), y: (g - 8) * TILE + 1 });
      // Passenger trains stop at the middle of the platform
      out.push({ kind: "freight", x: px(c0), y: g * TILE, extra: [c0 * TILE, (c0 + w) * TILE, (s0 + s1) * 4, 1] });
      enemies(c0, w, -1, 0.6);
      break;
    }
    case "harbour": {
      // Harbour basin under a long pier, a battleship in it, docks either side
      const hw = 18 + int(r, 8);
      w = hw + 10;
      ground(c0, 5);
      const g = L.h;
      prop(r() < 0.5 ? "containerstack" : "containerred", c0);
      prop("capstan", c0 + 4);
      for (let c = c0 + 5; c < c0 + 5 + hw; c++) {
        column(L, c, g + 6);
        put(L, c, g + 6, MUD);
        for (let row = g + 1; row <= g + 5; row++) put(L, c, row, WATER);
      }
      fill(L, c0 + 5, g, hw, 1, WOOD);
      L.props.push({ kind: "legs", x: (c0 + 5) * TILE, y: (g + 1) * TILE, w: hw * TILE, h: 5 * TILE });
      L.props.push({ kind: "rope", x: (c0 + 5) * TILE, y: g * TILE, w: hw * TILE });
      for (let c = c0 + 7; c < c0 + 3 + hw; c += 8) placeAt(L, "pierlamp", c, g);
      out.push({ kind: "battleship", x: (c0 + 5 + (hw >> 1)) * TILE, y: (g + 1) * TILE });
      ground(c0 + 5 + hw, 5);
      prop(r() < 0.5 ? "dockcrane" : "lighthousebig", c0 + 6 + hw, true);
      prop(pickAsset(r, t.backdrop), c0 + 6 + hw);
      enemies(c0 + 5, hw, g, 0.6);
      enemies(c0 + 5 + hw, 5);
      break;
    }
    case "flat": {
      w = 8 + int(r, 8);
      natural(c0, w);
      skyline(c0, w);
      deco(c0, w);
      deco(c0, w);
      scatter(c0, w, 1 + int(r, 3));
      standing(c0, w);
      if (r() < 0.4) underground(c0, w);
      if (t.enemies > 0 && c0 > 70 && r() < 0.25)
        for (let k = 1 + int(r, 2); k > 0; k--) spawn("mine", c0 + 1 + int(r, w - 2));
      enemies(c0, w);
      barrels(c0, w);
      break;
    }
    case "slope": {
      w = 12 + int(r, 10);
      natural(c0, w);
      skyline(c0, w);
      deco(c0, w);
      deco(c0, w);
      scatter(c0, w, 1 + int(r, 3));
      standing(c0, w);
      enemies(c0, w);
      break;
    }
    case "gap": {
      const gw = 2 + int(r, 2);
      ground(c0, 1);
      ground(c0 + 1, gw, Math.min(ROWS - 3, L.h + 7));
      ground(c0 + 1 + gw, 3);
      w = gw + 4;
      break;
    }
    case "lava": {
      const gw = 3;
      ground(c0, 2);
      ground(c0 + 2, gw, L.h + 3);
      fill(L, c0 + 2, L.h + 1, gw, 2, LAVA);
      ground(c0 + 2 + gw, 3);
      w = gw + 5;
      enemies(c0 + 2 + gw, 3);
      break;
    }
    case "bridge": {
      const gw = 4 + int(r, 3);
      ground(c0, 1);
      ground(c0 + 1, gw, Math.min(ROWS - 3, L.h + 9));
      fill(L, c0 + 1, L.h, gw, 1, WOOD);
      L.props.push({ kind: "legs", x: (c0 + 1) * TILE, y: (L.h + 1) * TILE, w: gw * TILE, h: 6 * TILE });
      ground(c0 + 1 + gw, 2);
      w = gw + 3;
      enemies(c0 + 1, gw, L.h);
      break;
    }
    case "wall": {
      w = 4 + int(r, 2);
      ground(c0, w);
      const tall = r() < 0.25 ? 4 + int(r, 2) : 2 + int(r, 2);
      const ww = 1 + int(r, 2);
      fill(L, c0 + 1, L.h - tall, ww, tall, t.wall);
      enemies(c0 + 1, ww, L.h - tall);
      enemies(c0 + 1 + ww, w - 1 - ww);
      break;
    }
    case "huts": {
      w = 10;
      ground(c0, w);
      hut(c0 + 1, L.h, t.build);
      scatter(c0 + 2, 6, 1);
      enemies(c0 + 2, 6);
      if (r() < 0.6) spawn(troop(), c0 + 5, L.h - 5);
      break;
    }
    case "stilts": {
      // Hut on stilts: the squad runs underneath while its guards shoot down
      w = 11;
      ground(c0, w);
      const floor = L.h - 5;
      fill(L, c0 + 2, floor, 8, 1, WOOD);
      hut(c0 + 2, floor, t.build);
      L.props.push({ kind: "legs", x: (c0 + 2) * TILE, y: (floor + 1) * TILE, w: 8 * TILE, h: 4 * TILE });
      climb(c0 + 9, floor, L.h);
      scatter(c0 + 3, 5, 1);
      enemies(c0 + 3, 6, floor, 1.4);
      break;
    }
    case "tower": {
      w = 7;
      ground(c0, w);
      const g = L.h;
      const ht = 6 + int(r, 4);
      fill(L, c0 + 2, g - ht, 4, 1, WOOD);
      put(L, c0 + 2, g - ht - 1, SANDBAG);
      L.props.push({ kind: "legs", x: (c0 + 2) * TILE, y: (g - ht + 1) * TILE, w: 4 * TILE, h: (ht - 1) * TILE });
      climb(c0 + 3, g - ht, g);
      if (r() < 0.5) {
        // A tin roof on posts
        fill(L, c0 + 1, g - ht - 5, 6, 1, TIN);
        L.props.push(
          { kind: "support", x: (c0 + 2) * TILE + 1, y: (g - ht - 4) * TILE, h: 4 * TILE },
          { kind: "support", x: (c0 + 6) * TILE - 1, y: (g - ht - 4) * TILE, h: 4 * TILE }
        );
      }
      spawn(r() < 0.5 ? "sniper" : troop(), c0 + 4, g - ht);
      if (night) L.props.push({ kind: "searchlight", x: (c0 + 6) * TILE - 2, y: (g - ht) * TILE - 3 });
      enemies(c0, w);
      break;
    }
    case "bunker": {
      const bw = 10 + int(r, 4);
      const bh = 5 + int(r, 2);
      w = bw + 2;
      ground(c0, w);
      const g = L.h;
      const x0 = c0 + 1;
      const x1 = x0 + bw - 1;
      const top = g - bh;
      fill(L, x0, top, bw, bh, AIR, t.build);
      fill(L, x0, top, 1, bh - 2, t.build, t.build);
      fill(L, x1, top, 1, bh - 2, t.build, t.build);
      fill(L, x0, top - 1, bw, 1, t.build, t.build);
      put(L, x0, top + 1, WINDOW);
      put(L, x1, top + 1, WINDOW);
      put(L, x0, top - 2, SANDBAG);
      put(L, x1, top - 2, SANDBAG);
      spawn("turret", x0 + (bw >> 1), top - 1);
      climb(x1 + 1, top - 1, g);
      scatter(x0 + 2, bw - 4, 1 + int(r, 2));
      if (r() < 0.7) spawn(troop(), x0 + 2, top - 1);
      enemies(x0 + 1, bw - 2);
      L.props.push({ kind: "lamp", x: px(x0 + (bw >> 1)), y: top * TILE + 2 });
      if (night || r() < 0.3) L.props.push({ kind: "alarm", x: px(x1 - 1), y: (top - 1) * TILE });
      break;
    }
    case "tunnel": {
      // Buried tunnel with a garrison, reached by blasting through the roof
      w = 16 + int(r, 6);
      ground(c0, w);
      const g = L.h;
      fill(L, c0 + 2, g + 3, w - 4, 3, AIR, t.fill);
      fill(L, c0 + 2, g + 2, 1, 1, WOOD);
      fill(L, c0 + w - 3, g + 2, 1, 1, WOOD);
      for (let c = c0 + 4; c < c0 + w - 3; c += 6) L.props.push({ kind: "lamp", x: px(c), y: (g + 3) * TILE + 1 });
      enemies(c0 + 3, w - 6, g + 6, 1.2);
      spawn(r() < 0.5 ? "ammo" : "barrel", c0 + 3 + int(r, w - 6), g + 6);
      enemies(c0, w, -1, 0.6);
      deco(c0, w);
      break;
    }
    case "mgnest": {
      w = 8;
      ground(c0, w);
      const g = L.h;
      put(L, c0 + 2, g - 1, SANDBAG);
      fill(L, c0 + 5, g - 2, 1, 2, SANDBAG);
      spawn("mg", c0 + 3);
      L.props.push({ kind: "wire", x: c0 * TILE, y: g * TILE, w: 2 * TILE });
      prop(r() < 0.5 ? "ammocrate" : "sandbags", c0 + 6);
      enemies(c0 + 5, 3);
      break;
    }
    case "catwalk": {
      const cw = 12 + int(r, 6);
      w = cw + 2;
      ground(c0, w);
      const row = L.h - 5;
      fill(L, c0 + 1, row, cw, 1, METAL);
      fill(L, c0 + 1, row - 3, cw, 1, PIPE);
      for (let c = c0 + 2; c < c0 + cw; c += 5)
        L.props.push({ kind: "support", x: px(c), y: (row + 1) * TILE, h: 4 * TILE });
      climb(c0 + cw, row, L.h);
      scatter(c0 + 2, cw - 2, 1);
      enemies(c0 + 1, cw, row);
      enemies(c0 + 1, cw);
      barrels(c0 + 1, cw);
      break;
    }
    case "tank": {
      // Fuel depot
      w = 12;
      ground(c0, w);
      fill(L, c0, L.h - 4, w, 1, PIPE);
      spawn("fuel", c0 + 3);
      spawn("fuel", c0 + 8);
      enemies(c0 + 9, 3);
      break;
    }
    case "tanker": {
      w = 9;
      ground(c0, w);
      spawn("tanker", c0 + 4);
      enemies(c0 + 7, 2);
      break;
    }
    case "convoy": {
      w = 18;
      ground(c0, w);
      spawn(r() < 0.5 ? "jeep" : "apc", c0 + 4);
      const armour: Spawn["kind"][] = [
        "jeep",
        "truck",
        "lighttank",
        "medtank",
        "heavytank",
        "flametank",
        "rockettank",
        "aatank",
        "tanker",
      ];
      spawn(armour[int(r, armour.length)], c0 + 13);
      break;
    }
    case "road": {
      // Dirt track with a troop truck
      w = 18;
      ground(c0, w);
      for (let c = c0; c < c0 + w; c++) put(L, c, L.h, t.fill);
      spawn("truck", c0 + 12);
      if (r() < 0.4) spawn("jeep", c0 + 4);
      break;
    }
    case "canopy": {
      // Rainforest giants, undergrowth and sometimes a sniper platform in the trees
      w = 14 + int(r, 8);
      natural(c0, w);
      for (let c = c0 + 3; c < c0 + w - 3; c += 5 + int(r, 3)) bigTree(c, gr(c), c0, c0 + w);
      for (let c = c0; c < c0 + w; c++) if (r() < 0.25) soft(c, gr(c) - 1, LEAF);
      if (r() < 0.55) {
        const pc = c0 + 2 + int(r, w - 7);
        // On a slope the platform clears the higher end and its legs reach the lower
        let hi = ROWS;
        let lo = 0;
        for (let c = pc; c < pc + 4; c++) {
          hi = Math.min(hi, gr(c));
          lo = Math.max(lo, gr(c));
        }
        const row = hi - 6 - int(r, 2);
        fill(L, pc, row, 4, 1, WOOD);
        L.props.push({ kind: "legs", x: pc * TILE, y: (row + 1) * TILE, w: 4 * TILE, h: (lo - row - 1) * TILE });
        spawn(r() < 0.6 ? "sniper" : troop(), pc + 2, row);
      }
      enemies(c0, w, -1, 1.1);
      break;
    }
    case "bamboo": {
      w = 12 + int(r, 6);
      natural(c0, w);
      for (let c = c0 + 1; c < c0 + w - 1; c++) {
        if (r() > 0.55) continue;
        const g = gr(c);
        const ht = 5 + int(r, 6);
        fill(L, c, g - ht, 1, ht, BAMBOO);
        soft(c + (r() < 0.5 ? -1 : 1), g - ht + int(r, 2), LEAF);
        if (r() < 0.4) soft(c + (r() < 0.5 ? -1 : 1), g - ht + 2 + int(r, 3), LEAF);
      }
      enemies(c0, w, -1, 1.3);
      break;
    }
    case "river": {
      // Waist-deep river with a rope bridge or sometimes a patrol boat
      const rw = 6 + int(r, 5);
      w = rw + 4;
      ground(c0, 2);
      const g = L.h;
      ground(c0 + 2, rw, g + 2);
      for (let c = c0 + 2; c < c0 + 2 + rw; c++) {
        put(L, c, g + 2, MUD);
        put(L, c, g + 1, WATER);
      }
      ground(c0 + 2 + rw, 2);
      for (const c of [c0 + 1, c0 + 2 + rw]) if (r() < 0.6) soft(c, g - 1, LEAF);
      if (r() < 0.45) {
        fill(L, c0 + 2, g, rw, 1, WOOD);
        L.props.push({ kind: "rope", x: (c0 + 2) * TILE, y: g * TILE, w: rw * TILE });
        enemies(c0 + 2, rw, g, 0.8);
      } else if (rw >= 8 && r() < 0.7) spawn("boat", c0 + 2 + (rw >> 1), g + 1);
      enemies(c0 + 2 + rw, 2);
      break;
    }
    case "waterfall": {
      // A cliff with a stream pouring over it into a pool
      const rise = Math.min(4 + int(r, 3), L.h - (PEAK_ROW + 4));
      if (rise < 3) {
        w = 8;
        ground(c0, w);
        deco(c0, w);
        break;
      }
      w = 11;
      const g = L.h;
      ground(c0, 2);
      ground(c0 + 2, 3, g + 1);
      for (let c = c0 + 2; c < c0 + 5; c++) put(L, c, g, WATER);
      const top = g - rise;
      ground(c0 + 5, 6, top + 1);
      for (let c = c0 + 5; c < c0 + 11; c++) put(L, c, top, WATER);
      L.props.push({ kind: "waterfall", x: (c0 + 4) * TILE - 2, y: top * TILE, w: 10, h: (g - top) * TILE });
      L.h = top;
      enemies(c0 + 6, 5, top + 1, 0.9);
      break;
    }
    case "temple": {
      // Stepped mossy temple, guarded on every tier, with a shrine and a gun on top
      const tiers = 3 + int(r, 2);
      const bw = 6 + tiers * 4;
      w = bw + 2;
      ground(c0, w);
      const g = L.h;
      const x0 = c0 + 1;
      for (let k = 0; k < tiers; k++) {
        const tx = x0 + k * 2;
        const tw = bw - k * 4;
        const top = g - (k + 1) * 2;
        fill(L, tx, top, tw, 2, MOSS, MOSS);
        for (const c of [tx - 1, tx + tw]) if (r() < 0.55) for (let k2 = 0; k2 < 2; k2++) soft(c, top + k2, VINE);
        enemies(tx + 1, Math.max(1, tw - 2), top, k === 0 ? 0.8 : 0.6);
      }
      const sx = x0 + (tiers - 1) * 2;
      const sw = bw - (tiers - 1) * 4;
      const roof = g - tiers * 2;
      const lintel = roof - 4;
      fill(L, sx + 1, lintel + 1, 1, 3, MOSS, MOSS);
      fill(L, sx + sw - 2, lintel + 1, 1, 3, MOSS, MOSS);
      fill(L, sx, lintel, sw, 1, MOSS, MOSS);
      backWall(L, sx + 2, lintel + 1, sw - 4, 3, MOSS);
      L.props.push(
        { kind: "brazier", x: px(sx), y: roof * TILE },
        { kind: "brazier", x: px(sx + sw - 1), y: roof * TILE }
      );
      spawn(r() < 0.5 ? "mg" : "turret", sx + (sw >> 1), lintel);
      spawn(troop(), sx + (sw >> 1), roof);
      break;
    }
    case "statues": {
      // Overgrown ruins: broken pillars with vines, sometimes a fallen one
      w = 12 + int(r, 5);
      ground(c0, w);
      const g = L.h;
      for (let c = c0 + 2; c < c0 + w - 2; c += 3 + int(r, 3)) {
        const ht = 2 + int(r, 4);
        fill(L, c, g - ht, 1, ht, MOSS, MOSS);
        if (r() < 0.5) for (let x = c - 1; x <= c + 1; x++) soft(x, g - ht - 1, MOSS);
        for (let k = 0; k < ht - 1; k++) if (r() < 0.5) soft(c + 1, g - ht + k, VINE);
      }
      if (r() < 0.6) {
        const c = c0 + 1 + int(r, w - 5);
        for (let x = c; x < c + 3; x++) soft(x, g - 1, MOSS);
      }
      enemies(c0, w);
      barrels(c0, w);
      break;
    }
    case "camp": {
      // Enemy camp round a campfire
      w = 16;
      ground(c0, w);
      const g = L.h;
      L.props.push({ kind: "tent", x: px(c0 + 3), y: g * TILE });
      if (r() < 0.6) L.props.push({ kind: "tent", x: px(c0 + 13), y: g * TILE });
      L.props.push({ kind: "fire", x: px(c0 + 8), y: g * TILE });
      L.props.push({ kind: "flag", x: (c0 + 5) * TILE, y: g * TILE });
      put(L, c0 + 6, g - 1, CRATE);
      if (r() < 0.6) prop(r() < 0.5 ? "cage" : "camonet", c0 + 9);
      scatter(c0 + 1, w - 2, 2);
      spawn("ammo", c0 + 10);
      if (r() < 0.6) spawn("barrel", c0 + 11);
      enemies(c0 + 1, w - 2, -1, 1.4);
      if (t.troops.dog) spawn("dog", c0 + 9);
      break;
    }
    case "derrick": {
      w = 9;
      ground(c0, w);
      fill(L, c0 + 2, L.h - 1, 5, 1, METAL);
      L.props.push({ kind: "derrick", x: (c0 + 4) * TILE + 4, y: (L.h - 1) * TILE });
      spawn("barrel", c0 + 7);
      enemies(c0, 2);
      break;
    }
    case "silo": {
      // Missile silo hatch in a concrete pad
      w = 10;
      ground(c0, w);
      fill(L, c0 + 2, L.h, 6, 1, CONCRETE);
      fill(L, c0 + 3, L.h + 1, 4, 5, AIR, METAL);
      fill(L, c0 + 3, L.h, 4, 1, HAZARD);
      L.props.push({ kind: "silo", x: (c0 + 5) * TILE, y: L.h * TILE });
      enemies(c0 + 7, 3);
      break;
    }
    case "radar": {
      w = 8;
      ground(c0, w);
      fill(L, c0 + 3, L.h - 6, 3, 1, METAL);
      L.props.push({ kind: "legs", x: (c0 + 3) * TILE, y: (L.h - 5) * TILE, w: 3 * TILE, h: 5 * TILE });
      L.props.push({ kind: "radar", x: (c0 + 4) * TILE + 4, y: (L.h - 6) * TILE });
      spawn(troop(), c0 + 4, L.h - 6);
      break;
    }
    case "ruins": {
      // Shelled building with enemies on every floor
      const bw = 10 + int(r, 5);
      w = bw + 2;
      ground(c0, w);
      const g = L.h;
      const floors = 2 + int(r, 2);
      const x0 = c0 + 1;
      for (let f = 1; f <= floors; f++) {
        const row = g - f * 5;
        for (let c = x0; c < x0 + bw; c++) if (r() > 0.15) put(L, c, row, t.build, t.build);
        backWall(L, x0, row + 1, bw, 4, t.build);
        fill(L, x0, row + 1, 1, 2, t.build);
        if (r() < 0.6) put(L, x0, row + 2, WINDOW);
        enemies(x0 + 1, bw - 2, row, 0.9);
      }
      backWall(L, x0, g - 4, bw, 4, t.build);
      climb(x0 + bw - 2, g - floors * 5, g);
      if (r() < 0.6) spawn("sniper", x0 + bw - 2, g - floors * 5);
      enemies(x0 + 1, bw - 2, -1, 0.8);
      break;
    }
    case "mesa": {
      const mw = 6 + int(r, 5);
      const mh = 5 + int(r, 5);
      w = mw + 3;
      ground(c0, w);
      fill(L, c0 + 1, L.h - mh, mw, mh, t.wall, t.wall);
      enemies(c0 + 1, mw, L.h - mh, 1.3);
      if (r() < 0.5) spawn("sniper", c0 + mw, L.h - mh);
      break;
    }
    case "pines": {
      w = 12 + int(r, 6);
      natural(c0, w);
      for (let c = c0 + 2; c < c0 + w - 1; c += 4 + int(r, 3)) pine(c, gr(c));
      enemies(c0, w);
      break;
    }
    case "deadtrees": {
      w = 12 + int(r, 6);
      natural(c0, w);
      for (let c = c0 + 2; c < c0 + w - 1; c += 5 + int(r, 3)) deadTree(c, gr(c));
      enemies(c0, w);
      barrels(c0, w);
      break;
    }
    case "eggs": {
      w = 10 + int(r, 5);
      ground(c0, w);
      for (let k = 2 + int(r, 3); k > 0; k--) spawn("egg", c0 + 1 + int(r, w - 2));
      for (let c = c0; c < c0 + w; c += 3) if (r() < 0.5) put(L, c, L.h - 1, t.wall);
      enemies(c0, w, -1, 0.6);
      break;
    }
    case "spires": {
      w = 7 + int(r, 3);
      natural(c0, w);
      for (let k = 1 + int(r, 2); k > 0; k--) {
        const c = c0 + 1 + int(r, w - 3);
        const ht = 3 + int(r, 6);
        const wide = r() < 0.4 && c + 1 < c0 + w ? 2 : 1;
        const g = Math.min(gr(c), gr(c + wide - 1));
        fill(L, c, g - ht, wide, ht, t.wall);
      }
      enemies(c0, w);
      break;
    }
    case "wreck": {
      w = 8;
      ground(c0, w);
      fill(L, c0 + 2, L.h - 1, 4, 1, METAL);
      fill(L, c0 + 3, L.h - 2, 2, 1, METAL);
      L.props.push({ kind: "fire", x: (c0 + 4) * TILE, y: (L.h - 2) * TILE });
      enemies(c0 + 5, 3);
      barrels(c0, 2);
      break;
    }
    case "fort": {
      w = 9;
      ground(c0, w);
      const g = L.h;
      const ht = 7 + int(r, 2);
      fill(L, c0 + 3, g - ht, 3, ht, t.build, t.build);
      fill(L, c0 + 3, g - 3, 3, 3, AIR, t.build);
      put(L, c0 + 3, g - ht - 1, t.build);
      put(L, c0 + 5, g - ht - 1, t.build);
      put(L, c0 + 4, g - ht + 2, WINDOW);
      spawn("turret", c0 + 4, g - ht);
      spawn(troop(), c0 + 1);
      if (night) L.props.push({ kind: "alarm", x: (c0 + 6) * TILE, y: (g - ht) * TILE });
      L.props.push({ kind: "flag", x: (c0 + 3) * TILE + 2, y: (g - ht - 1) * TILE });
      enemies(c0 + 6, 3);
      break;
    }
  }
  L.next = c0 + w;
}

/** The stronghold that ends a level. The squad stops short and calls in ICBMs, which aim at plan.targets. */
function stronghold(L: Level, c0: number, out: Spawn[], troop: () => Spawn["kind"], night: boolean) {
  const t = L.theme;
  const r = L.rand;
  const px = (c: number) => c * TILE + TILE / 2;
  const w = 46;
  for (let c = c0; c < c0 + w; c++) column(L, c, L.h);
  const g = L.h;
  const spawn = (kind: Spawn["kind"], c: number, row = g) => out.push({ kind, x: px(c), y: row * TILE });
  const crowd = (from: number, width: number, row: number, n: number) => {
    for (let k = 0; k < n; k++) spawn(troop(), from + int(r, width), row);
  };
  // Approach
  fill(L, c0 + 2, g - 2, 1, 2, SANDBAG);
  put(L, c0 + 5, g - 1, SANDBAG);
  spawn("mg", c0 + 6);
  L.props.push({ kind: "wire", x: (c0 + 3) * TILE, y: g * TILE, w: 3 * TILE });
  placeProp(L, "hedgehog", c0);
  fill(L, c0 + 7, g - 3, 2, 3, FENCE);
  // Outer wall with a gate
  const ow = c0 + 9;
  fill(L, ow, g - 10, 3, 10, t.build, t.build);
  fill(L, ow, g - 3, 3, 3, AIR, t.build);
  for (const c of [ow, ow + 2]) put(L, c, g - 11, t.build);
  spawn("turret", ow + 1, g - 10);
  ladder(L, ow - 1, g - 10, g);
  L.props.push({ kind: "flag", x: ow * TILE + 2, y: (g - 11) * TILE });
  if (night) L.props.push({ kind: "searchlight", x: (ow + 3) * TILE, y: (g - 10) * TILE - 3 });
  // Courtyard
  spawn("fuel", c0 + 14);
  spawn("tanker", c0 + 18);
  spawn("barrel", c0 + 21);
  spawn("ammo", c0 + 22);
  crowd(c0 + 13, 8, g, 3);
  placeProp(L, "drums", c0 + 12);
  placeProp(L, "loudspeaker", c0 + 23);
  placeProp(L, "radiotower", c0 + 43, true);
  // Keep: three floors, with turrets and a radar on the roof
  const kx = c0 + 24;
  const kw = 14;
  const kh = 14;
  fill(L, kx, g - kh, kw, kh, AIR, t.build);
  fill(L, kx, g - kh, 1, kh - 2, t.build, t.build);
  fill(L, kx + kw - 1, g - kh, 1, kh - 2, t.build, t.build);
  fill(L, kx, g - kh - 1, kw, 1, t.build, t.build);
  ladder(L, kx + kw - 3, g - kh - 1, g);
  for (const row of [g - 5, g - 10]) {
    fill(L, kx + 1, row, kw - 2, 1, t.build, t.build);
    crowd(kx + 2, kw - 4, row, 2);
  }
  for (let row = g - kh + 2; row < g - 2; row += 3) {
    put(L, kx, row, WINDOW);
    put(L, kx + kw - 1, row, WINDOW);
  }
  spawn("turret", kx + 2, g - kh - 1);
  spawn("turret", kx + kw - 3, g - kh - 1);
  spawn("heavy", kx + 6, g);
  L.props.push({ kind: "radar", x: (kx + 7) * TILE, y: (g - kh - 1) * TILE });
  L.props.push({ kind: "antenna", x: (kx + 11) * TILE, y: (g - kh - 1) * TILE, h: 40 });
  L.props.push({ kind: "flag", x: (kx + 4) * TILE, y: (g - kh - 1) * TILE });
  L.props.push({ kind: "alarm", x: (kx + 9) * TILE, y: (g - kh - 1) * TILE });
  for (let c = kx + 2; c < kx + kw - 2; c += 4) L.props.push({ kind: "lamp", x: px(c), y: (g - kh) * TILE + 2 });
  // Back wall
  fill(L, c0 + 41, g - 9, 3, 9, t.build, t.build);
  spawn("turret", c0 + 42, g - 9);
  backWall(L, c0 + 20, g - 22, 22, 8, t.skyline || t.build);
  L.plan.targets = [px(ow + 1), px(kx + 4), px(kx + kw - 3), px(c0 + 42)];
  L.plan.stronghold = c0;
  L.plan.fortPlaced = true;
  L.next = c0 + w;
}
