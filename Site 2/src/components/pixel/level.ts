/**
 * One level of the run-and-gun behind the site: a side-scrolling stretch of 8×8 blocks
 * generated a chunk at a time just ahead of the camera, in a theme (themes.ts). The lie of
 * the land comes from seeded fractal noise (noise.ts), like a voxel game's world generator:
 * lowlands, rolling hills and whole mountain ranges, different on every level and every
 * visit, with rock strata and winding caves under the ground. Every level
 * opens on a landing zone, may have a boss arena half way, and ends at an enemy stronghold
 * that the squad flattens with ICBMs before moving on to the next, random level. Blocks
 * live in a ring buffer of columns, so a level of any length never grows in memory.
 * Pure data and maths: game.ts plays it, PixelScene draws it.
 */
import { int, rng } from "./rng";
import { fbm1, noise2, ridged1, smoothstep } from "./noise";
import { asset, pickAsset, type AssetId } from "./assets";
import { buildingsFor, SPEC_BY_KEY, type BuildingSpec, type Material } from "./buildings";
import type { BossKind, Theme, Troop } from "./themes";

/** Edge of one block, in scene pixels. */
export const TILE = 8;
/** Rows of blocks: room for mountains over the ground, and tunnels and caves under it. */
export const ROWS = 112;
/** Columns kept in memory (a power of two). */
export const RING = 512;
export const MASK = RING - 1;
/** Where the lowland ground is (the landing zone, the plains between the hills). */
export const BASE_ROW = 80;
/** The highest a mountain top gets, and the lowest a valley floor. */
export const PEAK_ROW = BASE_ROW - 38;
export const VALLEY_ROW = BASE_ROW + 6;
export const LEVEL_H = ROWS * TILE;

/* Block kinds (the foreground grid; the back-wall grid reuses the same ids) */
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
/** Mossy temple stone, hanging vines, river water, bamboo (the jungle levels) */
export const MOSS = 30;
export const VINE = 31;
export const WATER = 32;
export const BAMBOO = 33;
/** Ladders up towers and between floors, chain-link fencing, corrugated tin roofs */
export const LADDER = 34;
export const FENCE = 35;
export const TIN = 36;
/** Building materials, roads, and what's underground: ore, rails, sewers */
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
/** The world's old architecture: half-timbering, adobe, roof tiles, glazed green tiles, slate, paper screens, painted tiles, limestone */
export const TIMBER = 53;
export const ADOBE = 54;
export const TERRACOTTA = 55;
export const JADE = 56;
export const SLATE = 57;
export const SHOJI = 58;
export const AZULEJO = 59;
export const LIMESTONE = 60;
/** More soils and rocks: clay, gravel, granite (it doesn't break), basalt, loam, chalk; and autumn and blossom leaves */
export const CLAY = 61;
export const GRAVEL = 62;
export const GRANITE = 63;
export const BASALT = 64;
export const LOAM = 65;
export const CHALK = 66;
export const REDLEAF = 67;
export const PINKLEAF = 68;
export const KINDS = 69;

/** What stops people and bullets. Leaves, trunks, pine boughs, vines, bamboo, water, lava, ladders and fences don't. */
export const SOLID = new Uint8Array(KINDS).map((_, k) =>
  k === AIR || k === LEAF || k === TRUNK || k === LAVA || k === PINE || k === VINE || k === WATER || k === BAMBOO || k === LADDER || k === FENCE || k === RAIL || k === REDLEAF || k === PINKLEAF ? 0 : 1
);
/** Hits a block takes before it breaks (255: never, so blasts can't drain rivers or lava). */
export const TILE_HP = Uint8Array.from([0, 2, 2, 4, 255, 3, 5, 1, 2, 1, 6, 1, 1, 1, 4, 2, 5, 6, 255, 3, 3, 1, 3, 1, 3, 1, 3, 2, 2, 4, 5, 1, 255, 1, 1, 1, 3, 1, 5, 2, 2, 3, 3, 4, 4, 3, 4, 5, 5, 1, 4, 2, 2, 3, 3, 2, 3, 3, 1, 3, 4, 2, 1, 255, 8, 2, 2, 1, 1]);

/** Something the generator placed for game.ts to bring to life. x, y: world px (y = feet; a boat's y is the water line). */
export interface Spawn {
  kind: Troop | "mortar" | "mg" | "turret" | "jeep" | "truck" | "boat" | "barrel" | "fuel" | "tanker" | "ammo" | "egg" | "propane" | "canister" | "cylinder" | "battleship" | "apc" | "lighttank" | "mine" | "medtank" | "heavytank" | "flametank" | "rockettank" | "aatank" | "launcher" | "manhole" | "tnt" | "submarine" | "flakgun" | "howitzer" | "samsite" | "rocketpod" | "train" | "armoredcar" | "artytruck" | "hovercraft";
  x: number;
  y: number;
}

/** Static decorations drawn over the blocks (world px). */
export type Prop =
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
  /** Falling water from (x, y) down `h` px, `w` px wide */
  | { kind: "waterfall"; x: number; y: number; w: number; h: number }
  | { kind: "tent"; x: number; y: number }
  /** Rope handrails over a bridge deck at y, `w` px long */
  | { kind: "rope"; x: number; y: number; w: number }
  | { kind: "brazier"; x: number; y: number }
  /** One of the props in assets.ts, centred on x, standing at y; `back`: in shadow behind the ground */
  | { kind: "sprite"; id: AssetId; x: number; y: number; flip: boolean; back: boolean }
  /** A building's sign, centred on x with its bottom at y (`neon`: it glows) */
  | { kind: "sign"; x: number; y: number; text: string; ink: string; board: string; neon: boolean }
  /** A striped awning over a door, sticking out left of x */
  | { kind: "awning"; x: number; y: number; w: number; color: string };

/** The shape of a level: how long, where the boss waits, where the stronghold is. */
export interface Plan {
  /** Columns before the stronghold */
  length: number;
  boss: { col: number; kind: BossKind } | null;
  bossPlaced: boolean;
  /** First column of the stronghold (once placed) and the points the ICBMs aim for (world px) */
  stronghold: number;
  fortPlaced: boolean;
  targets: number[];
}

export interface Level {
  fg: Uint8Array;
  bg: Uint8Array;
  /** Hits left before a block breaks */
  hp: Uint8Array;
  /** Scorch level 0–3 */
  burn: Uint8Array;
  /** Seamless levels: the stretch being generated (1, 2, …), every how many a stronghold and finale comes, whether never (endless), or forced */
  segment: number;
  every: number;
  endless: boolean;
  forceFort: boolean;
  /** Where the land runs on into the next biome: the column, the new theme, its boss */
  borders: { col: number; theme: Theme; boss: BossKind | null }[];
  /** Boss arenas placed, waiting for the squad */
  bossQueue: { col: number; kind: BossKind }[];
  /** Picks the next biome (the game sets it; levels on their own keep their theme) */
  pickTheme?: (current: Theme) => Theme;
  /** Ground row of each stored column */
  ground: Int16Array;
  /** Oldest column still stored, and the next one to generate (absolute column numbers) */
  start: number;
  next: number;
  /** The generator's current ground row */
  h: number;
  /** The level's world seed (the noise the hills and caves come from) */
  seed: number;
  theme: Theme;
  plan: Plan;
  /** Multiplies how many enemies the generator places (set by the game) */
  intensity: number;
  rand: () => number;
  props: Prop[];
}

/** Index of a block in the stored grid (absolute column, row). */
export const cell = (c: number, r: number) => r * RING + (c & MASK);

export interface LevelOptions {
  /** Force a boss (or none) instead of rolling for one */
  boss?: BossKind | null;
  length?: number;
  /** End this first stretch at a stronghold (the finale) whatever the count */
  stronghold?: boolean;
  /** A stronghold and finale every this many stretches (default 20) */
  every?: number;
  /** Never: the land runs on from biome to biome for ever */
  endless?: boolean;
}

export function createLevel(theme: Theme, seed = 1, opts: LevelOptions = {}): Level {
  const rand = rng(seed);
  const length = opts.length ?? 240 + int(rand, 160);
  const kind = opts.boss !== undefined ? opts.boss : theme.bosses.length && rand() < 0.8 ? theme.bosses[int(rand, theme.bosses.length)] : null;
  return {
    fg: new Uint8Array(RING * ROWS),
    bg: new Uint8Array(RING * ROWS),
    hp: new Uint8Array(RING * ROWS),
    burn: new Uint8Array(RING * ROWS),
    ground: new Int16Array(RING).fill(BASE_ROW),
    start: 0,
    next: 0,
    h: BASE_ROW,
    seed: Math.floor(rand() * 1e9),
    theme,
    plan: { length, boss: kind ? { col: Math.floor(length * 0.48), kind } : null, bossPlaced: false, stronghold: length, fortPlaced: false, targets: [] },
    segment: 1,
    every: opts.every ?? 20,
    endless: !!opts.endless,
    forceFort: !!opts.stronghold,
    borders: [],
    bossQueue: [],
    intensity: 1,
    rand,
    props: [],
  };
}

/** The block at an absolute column and row (outside the stored columns: open air; below the level: bedrock). */
export function kindAt(L: Level, c: number, r: number) {
  if (r < 0 || c < L.start || c >= L.next) return AIR;
  if (r >= ROWS) return BEDROCK;
  return L.fg[cell(c, r)];
}

/** Is there something solid at this world-pixel position? */
export const solidAt = (L: Level, x: number, y: number) => {
  const r = Math.floor(y / TILE);
  if (r >= ROWS) return true;
  return SOLID[kindAt(L, Math.floor(x / TILE), r)] === 1;
};

export const lavaAt = (L: Level, x: number, y: number) => kindAt(L, Math.floor(x / TILE), Math.floor(y / TILE)) === LAVA;
export const waterAt = (L: Level, x: number, y: number) => kindAt(L, Math.floor(x / TILE), Math.floor(y / TILE)) === WATER;

/** Ground row of a stored column (BASE_ROW outside them). */
export const groundOf = (L: Level, c: number) => (c >= L.start && c < L.next ? L.ground[c & MASK] : BASE_ROW);

/** World-pixel y of the first solid block at or below `fromY` in a column. */
export function surfaceY(L: Level, x: number, fromY = 0) {
  for (let r = Math.max(0, Math.floor(fromY / TILE)); r < ROWS; r++) if (solidAt(L, x, r * TILE + 1)) return r * TILE;
  return LEVEL_H;
}

/** Nothing solid on the straight line between two points (sampled every 4 px). */
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

/** Knock `dmg` hits off a block. Returns the kind that broke, or 0 if it held (or there was nothing). */
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

/** A back wall only (behind the action, where there's nothing in front). */
function backWall(L: Level, c0: number, r0: number, w: number, h: number, k: number) {
  for (let r = Math.max(0, r0); r < Math.min(ROWS, r0 + h); r++)
    for (let c = c0; c < c0 + w; c++) {
      const i = cell(c, r);
      if (L.fg[i] === AIR) L.bg[i] = k;
    }
}

/**
 * The ground row the land wants at column `c`: broad lowlands and mountain ranges (a slow
 * noise decides which), ridged peaks inside the ranges, rolling hills everywhere, a little
 * roughness on top. The landing zone is flat and the hills grow in after it.
 */
export function terrainAt(L: Level, c: number) {
  const s = L.seed;
  const relief = L.theme.relief * smoothstep(40, 120, c);
  if (relief <= 0) return BASE_ROW;
  // 0 lowlands … 1 the heart of a range (about 600 columns from one range to the next)
  const range = smoothstep(-0.15, 0.45, fbm1(s + 11, c / 300, 2));
  const peaks = ridged1(s + 23, c / 70, 3, 0.45) * (6 + 30 * range);
  const hills = (fbm1(s + 37, c / 34, 3) * 0.5 + 0.5) * 9;
  const rough = fbm1(s + 51, c / 7, 2) * 1.4;
  const lift = (peaks + hills + rough) * relief - 4 * (1 - range) * relief;
  return Math.max(PEAK_ROW, Math.min(VALLEY_ROW, Math.round(BASE_ROW - lift)));
}

/** Write one fresh column of ground with its top at row `h` (dropping the oldest stored column if the ring is full). */
function column(L: Level, c: number, h: number) {
  if (c - L.start >= RING) L.start = c - RING + 1;
  const t = L.theme;
  const sc = c & MASK;
  for (let r = 0; r < ROWS; r++) {
    const i = r * RING + sc;
    L.fg[i] = L.bg[i] = L.hp[i] = L.burn[i] = 0;
  }
  // Cliffs show bare earth; high summits bare rock (snow stays snow)
  const left = c > L.start ? L.ground[(c - 1) & MASK] : h;
  const steep = Math.abs(left - h) >= 2;
  const summit = h < BASE_ROW - 24 && t.top !== SNOW;
  const top = summit ? t.deep : steep ? t.fill : t.top;
  // Hard country: granite under the soil, which no blast can dig through
  const hard = noise2(L.seed + 71, c / 45, 0.5) > 0.6;
  // The soil gets thinner up a mountain, and rock shows through in seams
  const soil = summit ? 1 : 3 + Math.round(noise2(L.seed + 3, c / 9, 0) * 3);
  for (let r = h; r < ROWS; r++) {
    let k = r >= ROWS - 2 ? BEDROCK : r === h ? top : r <= h + soil ? t.fill : t.deep;
    if (k === t.fill && r > h) {
      // Pockets of other soils in the topsoil: loam, clay, gravel
      const soilN = noise2(L.seed + 81, c / 12, r / 5);
      if (soilN > 0.74) k = GRAVEL;
      else if (soilN > 0.64) k = t.fill === DIRT || t.fill === LOAM ? CLAY : LOAM;
    }
    if (k === t.deep && r < ROWS - 2) {
      // The rock beneath: granite where the country's hard (it won't break), bands of basalt and chalk elsewhere
      if (hard && r > h + soil + 1) k = GRANITE;
      else {
        const rockN = noise2(L.seed + 91, c / 10, r / 6);
        if (rockN > 0.72) k = BASALT;
        else if (rockN < 0.16) k = CHALK;
      }
    }
    if (k === t.deep && r < ROWS - 2 && noise2(L.seed + 7, c / 6, r / 3) > 0.78) k = t.fill;
    else if (k === t.deep && r < ROWS - 3) {
      // Veins of ore: coal fairly near the top, iron deeper, a little gold deepest of all
      const ore = noise2(L.seed + 41, c / 3.5, r / 2.5);
      if (ore > 0.86) k = r > h + 22 && noise2(L.seed + 43, c / 9, r / 9) > 0.6 ? GOLD : r > h + 12 ? IRON : COAL;
    }
    put(L, c, r, k, r <= h + soil ? t.fill : t.deep);
  }
  // Caves: winding tunnels and the odd cavern, well under the surface (blast down to find them)
  if (!t.ceiling && c >= 40)
    for (let r = h + 7; r < ROWS - 4; r++) {
      const worm = Math.abs(noise2(L.seed + 13, c / 11, r / 6) - 0.5);
      const hall = noise2(L.seed + 17, c / 26, r / 9);
      // (only now and then: the ground is mostly solid, with mineshafts dug into it instead)
      if (worm < 0.01 || hall > 0.92) {
        const i = cell(c, r);
        L.fg[i] = L.hp[i] = 0;
        L.bg[i] = t.deep;
      }
    }
  if (t.ceiling) {
    // A cave roof with stalactites, crystals along its underside
    const roof = Math.max(3, h - 22 + Math.round(Math.sin(c * 0.5) * 2 + L.rand() * 3));
    for (let r = 0; r <= roof; r++) put(L, c, r, r === roof && L.rand() < 0.18 ? CRYSTAL : STONE, STONE);
  } else if (L.rand() < 0.05) put(L, c, Math.min(ROWS - 3, h + 6 + int(L.rand, 8)), CRYSTAL);
  L.ground[sc] = h;
}

const pick = <K extends string>(weights: Partial<Record<K, number>>, r: () => number): K => {
  const entries = Object.entries(weights) as [K, number][];
  let x = r() * entries.reduce((s, [, w]) => s + w, 0);
  for (const [f, w] of entries) if ((x -= w) < 0) return f;
  return entries[0][0];
};

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
 * Stand a prop on a floor: the top of row `row`, columns from c (`lift`: raised off the floor,
 * for things on the wall or hanging from the ceiling). Only where it's open above that floor
 * and it doesn't stand on the last thing put there.
 */
function placeAt(L: Level, id: AssetId, c: number, row: number, lift = 0) {
  const a = asset(id);
  const cols = Math.ceil(a.w / TILE);
  for (let k = 0; k < cols; k++) {
    const above = L.fg[cell(c + k, row - 1)];
    if (above !== AIR && above !== LADDER) return false;
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
 * Build one of the hundred-odd kinds of building (buildings.ts) on flat ground from column c0,
 * cut away so every storey shows: walls with doors right through the ground floor (the squad
 * runs through), floors, a ladder up through them all, windows in the back wall, furniture on
 * every floor, its roof and what's on it, a sign, an awning. Returns the columns it took.
 */
function building(L: Level, c0: number, spec: BuildingSpec, out: Spawn[], troop: () => Spawn["kind"]) {
  const r = L.rand;
  const t = L.theme;
  const g = L.h;
  const wall = MATERIAL[spec.wall];
  const back = MATERIAL[spec.back];
  const slab = spec.floor ? MATERIAL[spec.floor] : wall;
  const fh = spec.fh;
  const bw = spec.width[0] + int(r, spec.width[1] - spec.width[0] + 1);
  const room = 12;
  const floors = Math.max(1, Math.min(spec.floors[0] + int(r, spec.floors[1] - spec.floors[0] + 1), Math.floor((g - room) / fh)));
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
    // The storey: back wall with its windows, side walls, the floor above
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
    // Doors right through the ground floor (the squad runs through); a window up the side upstairs
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
    // What's in it
    const list = f > 0 && spec.upper ? spec.upper : spec.items;
    for (let k = Math.max(2, Math.round(bw / 4)) + int(r, 2); k > 0; k--) placeAt(L, pickItem(list), x0 + 1 + int(r, Math.max(1, bw - 3)), base);
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
    // Who's in it
    for (let k = (spec.guards ?? 1) * (f === 0 ? 0.6 : 1) * (0.4 + r()) * t.enemies * L.intensity * 0.25; k >= 1; k--) spawnAt(x0 + 2 + int(r, Math.max(1, bw - 4)), base);
  }
  // A ladder all the way up, through every floor to the roof
  ladder(L, x1 - 1, top, g);
  // The roof
  const cx = x0 + (bw >> 1);
  let roofTop = top;
  switch (spec.roof) {
    case "parapet":
      put(L, x0, top - 1, wall);
      put(L, x1, top - 1, wall);
      break;
    case "pitched": {
      const roofK = spec.wall === "wood" || spec.wall === "siding" ? WOOD : spec.wall === "tin" ? TIN : t.build === WOOD ? WOOD : BRICK;
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
        fill(L, cx - half, top - dy, half * 2 + (bw % 2), 1, dy === rad ? GOLD : spec.roofMat ? MATERIAL[spec.roofMat] : MARBLE, MARBLE);
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
      // Stacked roofs flaring out at the eaves, a finial on top
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
      // A drum, an onion dome, a gilt cross-less spire
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
      // A dome over the hall and a slender tower at one corner, a balcony near its top
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
      // A steep slate roof with dormer windows in it
      for (let k = 1; k <= 3; k++) {
        fill(L, x0 + k - 1, top - k, bw - (k - 1) * 2, 1, SLATE);
        if (k === 2) for (let c = x0 + 3; c < x1 - 2; c += 3) put(L, c, top - k, WINDOW, SLATE);
      }
      roofTop = top - 3;
      break;
    }
    case "stepgable": {
      // A stepped gable, the way they build along the canals
      for (let k = 1, half = bw >> 1; half > 0; k += 2, half -= 2) {
        fill(L, cx - half, top - k - 1, half * 2 + (bw % 2), 2, wall, back);
        roofTop = top - k - 1;
      }
      put(L, cx, roofTop - 1, wall);
      break;
    }
    case "columns": {
      // A temple front: an entablature and a pediment over a row of columns
      for (let c = x0; c <= x1; c += 2) put(L, c, top - 1, LIMESTONE);
      fill(L, x0 - 1, top - 2, bw + 2, 1, LIMESTONE);
      for (let k = 1, half = (bw >> 1) + 1; half > 0; k++, half -= 2) fill(L, cx - half, top - 2 - k, half * 2 + (bw % 2), 1, LIMESTONE);
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
      // Battlements, and a turret at each end with a pointed cap
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
      // A wide, shallow roof hanging well out past the walls
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
      // A bell-shaped stupa with a tall gilt spire
      [7, 7, 5, 5, 3].forEach((wd, k) => fill(L, cx - (wd >> 1), top - 1 - k, wd, 1, k > 2 ? GOLD : PLASTER));
      fill(L, cx, top - 10, 1, 5, GOLD);
      roofTop = top - 10;
      break;
    }
    case "clock": {
      // A clock tower: the clock high on its face, a spire over it
      fill(L, cx - 1, top - 9, 3, 9, wall, back);
      placeAt(L, "wallclock", cx - 1, top - 5);
      fill(L, cx - 2, top - 10, 5, 1, wall);
      for (let k = 0; k < 3; k++) put(L, cx, top - 11 - k, k === 2 ? GOLD : SLATE);
      roofTop = top - 13;
      break;
    }
    case "artdeco": {
      // Setbacks, tier on tier, and a spire
      fill(L, x0 + 1, top - 4, bw - 2, 4, wall, back);
      fill(L, x0 + 3, top - 7, Math.max(1, bw - 6), 3, wall, back);
      for (let c = x0 + 2; c < x1 - 1; c += 2) put(L, c, top - 2, WINDOW, back);
      fill(L, cx, top - 13, 1, 6, METAL);
      roofTop = top - 13;
      break;
    }
    case "brutalist":
      // A heavy slab, overhanging the walls
      fill(L, x0 - 2, top - 2, bw + 4, 2, CONCRETE);
      roofTop = top - 2;
      break;
  }
  if (spec.roof === "flat" || spec.roof === "parapet" || spec.roof === "neon" || spec.roof === "helipad" || spec.roof === "antenna" || spec.roof === "solar" || spec.roof === "tank")
    for (let k = int(r, 3); k > 0; k--) placeAt(L, spec.top ? spec.top[int(r, spec.top.length)] : "acunit", x0 + 1 + int(r, Math.max(1, bw - 3)), top);
  // Sign on top, awning over the door, things out front
  if (spec.sign) L.props.push({ kind: "sign", x: cx * TILE + (bw % 2 ? TILE / 2 : 0), y: (spec.roof === "neon" || spec.roof === "flat" || spec.roof === "parapet" ? top : roofTop) * TILE - 1, text: spec.sign, ink: spec.ink ?? "#ffffff", board: spec.board ?? "#1e1e24", neon: !!spec.neon });
  if (spec.awning) L.props.push({ kind: "awning", x: x0 * TILE + 2, y: (g - 3) * TILE + 1, w: 14, color: spec.awning });
  if (spec.outside) placeProp(L, spec.outside[int(r, spec.outside.length)], c0 - 3);
  // A few up on the roof
  if (floors > 1 && r() < 0.6) spawnAt(x0 + 2 + int(r, Math.max(1, bw - 4)), top);
  return w;
}

/**
 * Stand a prop on the ground at column c (`back`: in shadow behind the ground), if the ground
 * is level under all of it, nothing's in the way and it wouldn't stand on the last one.
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
  if (!back) for (let k = 0; k < cols; k++) if (L.fg[cell(c + k, g - 1)] !== AIR) return false;
  const x = c * TILE + (cols * TILE) / 2;
  for (let i = L.props.length - 1; i >= Math.max(0, L.props.length - 4); i--) {
    const q = L.props[i];
    if (q.kind === "sprite" && q.back === back && Math.abs(q.x - x) < (asset(q.id).w + a.w) / 2 + 2) return false;
  }
  L.props.push({ kind: "sprite", id, x, y: g * TILE, flip: L.rand() < 0.5, back });
  return true;
}

/** A ladder up column c from row `top` down to (not including) row `bottom`, through floors and open air. */
function ladder(L: Level, c: number, top: number, bottom: number) {
  for (let row = Math.max(0, top); row < Math.min(ROWS, bottom); row++) {
    const k = L.fg[cell(c, row)];
    if (k === AIR || k === L.theme.build || k === WOOD || k === METAL || k === CONCRETE) put(L, c, row, LADDER);
  }
}

/** Generate columns until `until` (exclusive). Returns what should come to life there. */
export function extend(L: Level, until: number): Spawn[] {
  const out: Spawn[] = [];
  while (L.next < until) feature(L, out);
  // Forget decorations that dropped out of the ring
  if (L.props.length > 300 || (L.props[0] && L.props[0].x < L.start * TILE)) L.props = L.props.filter((p) => p.x >= L.start * TILE);
  return out;
}

function feature(L: Level, out: Spawn[]) {
  const t = L.theme;
  const r = L.rand;
  const plan = L.plan;
  const c0 = L.next;
  const px = (c: number) => c * TILE + TILE / 2;
  // (this feature's columns aren't counted as generated until it's finished: read them directly)
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
      spawn(roll < 0.16 ? "ammo" : roll < 0.3 ? "propane" : roll < 0.41 ? "canister" : roll < 0.5 ? "cylinder" : roll < 0.62 ? "tnt" : "barrel", c + k);
    }
  };
  const prop = (id: AssetId, c: number, back = false) => placeProp(L, id, c, back);
  /** Scatter a few of the land's props over open ground. */
  const scatter = (from: number, w: number, n: number) => {
    for (let k = 0; k < n; k++) prop(pickAsset(r, t.backdrop), from + int(r, Math.max(1, w - 2)));
  };
  /** Now and then something big in the background: a water tower, a radio mast, a wreck. */
  const standing = (from: number, w: number) => {
    if (t.ceiling || r() > 0.5) return;
    prop(pickAsset(r, t.backdrop, true), from + int(r, Math.max(1, w - 3)), true);
  };
  const climb = (c: number, top: number, bottom: number) => ladder(L, c, top, bottom);
  /**
   * Something buried under this stretch: a mine with pit props, rails and a mine cart; a crypt
   * with a sarcophagus and treasure; a bunker; a sewer; a subway platform with a train in it; a
   * cavern with a pool; a secret lab; a vault. Deep enough to leave the ground above whole.
   */
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
    const kinds = ["mine", "mine", "mine", "crypt", "bunker", "sewer", "subway", "cavern", "lab", "vault"] as const;
    const kind = kinds[int(r, kinds.length)];
    const shell: Record<"mine" | "crypt" | "bunker" | "sewer" | "subway" | "cavern" | "lab" | "vault", number> = { mine: t.deep, crypt: STONE, bunker: CONCRETE, sewer: SEWER, subway: TILES, cavern: t.deep, lab: STEEL, vault: STEEL };
    const k = shell[kind];
    fill(L, x0, top, rw, rh + 2, k, k);
    fill(L, x0 + 1, top + 1, rw - 2, rh, AIR, kind === "cavern" || kind === "mine" ? t.deep : k);
    if (kind === "cavern") {
      // Ragged, not a box
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
        // A shaft from the surface: a headframe over it, a ladder all the way down
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
        item(["minecart", "orepile", "toolsrack", "orepile"], 3);
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
        if (t.enemies > 0) for (let j = 1 + int(r, 2); j > 0; j--) out.push({ kind: troop(), x: (x0 + 2 + int(r, rw - 4)) * TILE + 4, y: floor * TILE });
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
   * Open country: the ground follows the noise, climbing at most two blocks a column (a
   * jump) and dropping at most two (so a slope never reads as a pit).
   */
  const natural = (from: number, w: number) => {
    for (let c = from; c < from + w; c++) {
      const d = terrainAt(L, c) - L.h;
      L.h += d > 0 ? Math.min(2, d) : Math.max(-2, d);
      column(L, c, L.h);
    }
  };
  const night = t.night > 0.3;
  /** Big ruined structures in the background: depth behind the fight */
  const skyline = (from: number, w: number) => {
    if (!t.skyline || r() > 0.45) return;
    const bw = 5 + int(r, Math.max(1, w - 5));
    const bh = 8 + int(r, 14);
    const top = L.h - bh;
    backWall(L, from, top, bw, bh, t.skyline);
    // Broken windows: holes in the back wall
    for (let rr = top + 2; rr < L.h - 2; rr += 3) for (let cc = from + 1; cc < from + bw - 1; cc += 2) if (r() < 0.55) L.bg[cell(cc, rr)] = 0;
    // A ragged top
    for (let cc = from; cc < from + bw; cc++) for (let k = int(r, 3); k > 0; k--) L.bg[cell(cc, top + k - 1)] = 0;
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
  /** Put a see-through block (leaf, vine, bamboo) only into open air, within columns lo..hi-1. */
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
  /** A rainforest giant: a tall trunk, a wide crown, vines hanging from it (in autumn red, in a blossom valley pink). */
  const crown = t.decor === "autumn" ? REDLEAF : t.decor === "sakura" ? PINKLEAF : LEAF;
  const bigTree = (c: number, g: number, lo: number, hi: number) => {
    const th = 9 + int(r, 6);
    fill(L, c, g - th, 1, th, TRUNK);
    const top = g - th - 2;
    [3, 7, 9, 7].forEach((cw, k) => {
      for (let x = c - (cw >> 1); x <= c + (cw >> 1); x++) soft(x, top + k, crown === LEAF || r() < 0.75 ? crown : LEAF, lo, hi);
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
      // Grassy land with nothing of its own: woods
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
    // The end of this stretch: once in so many (unless endless), the enemy stronghold and the
    // finale; otherwise the land just runs on across a river into the next biome
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
    const length = 240 + int(r, 160);
    const kind = next.bosses.length && r() < 0.7 ? next.bosses[int(r, next.bosses.length)] : null;
    L.segment++;
    L.theme = next;
    L.plan = { length, boss: kind ? { col: L.next + Math.floor(length * 0.48), kind } : null, bossPlaced: false, stronghold: L.next + length, fortPlaced: false, targets: [] };
    L.borders.push({ col: c0 + 2 + (rw >> 1), theme: next, boss: kind });
    return;
  }
  if (plan.fortPlaced) {
    // After the stronghold: quiet ground (the level ends before anyone gets here)
    w = 16;
    ground(c0, w);
    L.next = c0 + w;
    return;
  }

  // Hill country between the set pieces: the more mountainous the theme, the more of it.
  // (Where the land climbs steeply and the theme has them, a stream pours over the cliff.)
  let choice = pick(t.features, r);
  if (t.features.waterfall && L.h - terrainAt(L, c0 + 10) >= 4 && r() < 0.5) choice = "waterfall";
  else if (r() < 0.3 * t.relief * smoothstep(40, 120, c0)) {
    w = 12 + int(r, 18);
    natural(c0, w);
    for (let k = 2 + int(r, 4); k > 0; k--) deco(c0, w);
    if (t.top === GRASS) for (let k = int(r, 3); k > 0; k--) prop((["hedge", "flowers", "stump", "logs", "boulders"] as const)[int(r, 5)], c0 + int(r, w - 2));
    scatter(c0, w, 2 + int(r, 3));
    standing(c0, w);
    if (r() < 0.35) underground(c0, w);
    if (t.enemies > 0 && c0 > 70 && r() < 0.3) for (let k = 1 + int(r, 3); k > 0; k--) spawn("mine", c0 + 2 + int(r, w - 4));
    enemies(c0, w);
    barrels(c0, w);
    L.next = c0 + w;
    return;
  } else if (!t.ceiling && t.enemies > 0 && c0 > 50 && r() < 0.1) {
    // A gun emplacement: a sandbagged pit with heavy weapons in it, their crews, ammunition
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
    // A bunker line: concrete pillboxes sunk into the ground, machine guns behind the slits, a trench between
    w = 22;
    ground(c0, w);
    const g = L.h;
    for (const bx of [c0 + 2, c0 + 13]) {
      fill(L, bx, g - 3, 7, 6, CONCRETE, CONCRETE);
      fill(L, bx + 1, g - 2, 5, 4, AIR, CONCRETE);
      // the firing slit, facing the way the squad comes
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
    // A missile launch pad: a concrete apron, a gantry, a missile on it fuelled and ready, and a launcher truck
    w = 18;
    ground(c0, w);
    const g = L.h;
    fill(L, c0 + 1, g, w - 2, 1, CONCRETE);
    fill(L, c0 + 3, g - 1, 4, 1, HAZARD);
    for (let row = g - 10; row < g; row++) put(L, c0 + 8, row, LADDER);
    fill(L, c0 + 8, g - 11, 3, 1, METAL);
    placeProp(L, "missiletrailer", c0 + 11);
    L.props.push({ kind: "silo", x: (c0 + 5) * TILE, y: g * TILE }, { kind: "antenna", x: (c0 + 9) * TILE + 4, y: (g - 11) * TILE, h: 24 }, { kind: "alarm", x: (c0 + 10) * TILE, y: (g - 11) * TILE });
    spawn("launcher", c0 + 14);
    enemies(c0 + 1, w - 2, -1, 1.2);
    L.next = c0 + w;
    return;
  } else if (!t.ceiling && t.enemies > 0 && c0 > 60 && r() < 0.07) {
    // A lone outpost: a tall watchtower with a ladder and a tin roof, fenced, guarded
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
    L.props.push({ kind: "support", x: tx * TILE + 1, y: (g - ht - 4) * TILE, h: 4 * TILE }, { kind: "support", x: (tx + 4) * TILE - 1, y: (g - ht - 4) * TILE, h: 4 * TILE });
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
      // City blocks: buildings shoulder to shoulder, side streets between them with their
      // traffic and street furniture, everything a city has (a hospital, a fuel station, police,
      // a fire station, a school, shops, offices, flats...), defended street by street
      const ESSENTIAL = ["hospital", "gasstation", "police", "firestation", "school", "supermarket", "apartments", "officetower", "bank", "diner", "cinema", "hotel", "parkinggarage", "pharmacy", "postoffice", "library", "skyscraper", "pizzeria", "convenience", "tenement", "brownstone", "decotower", "neontenement", "busdepot", "garage", "clinic", "townhouse", "haussmann", "cafe", "gym", "bakery", "pub", "trainstation", "stockexchange", "museum"];
      const specs = buildingsFor(t.backdrop);
      let c = c0;
      const street = (from: number, n: number) => {
        ground(from, n);
        for (let k = from; k < from + n; k++) put(L, k, L.h, ASPHALT, t.fill);
      };
      street(c, 4);
      c += 4;
      for (let n = 4 + int(r, 5); n > 0; n--) {
        const spec = r() < 0.65 ? SPEC_BY_KEY.get(ESSENTIAL[int(r, ESSENTIAL.length)]) : specs[int(r, specs.length)];
        if (!spec || L.h - spec.fh * spec.floors[0] < 12) continue;
        const bw = building(L, c, spec, out, troop);
        for (let k = c; k < c + bw; k++) put(L, k, L.h, PAVEMENT, t.fill);
        c += bw;
        if (r() < 0.55) continue;
        // A side street
        const gap = 3 + int(r, 4);
        street(c, gap);
        for (let j = 1 + int(r, 2); j > 0; j--) prop((["trafficlight", "busstop", "bench", "trashbin", "hydrant", "treeplanter", "streetlamp", "newsbox", "mailbox", "dumpster", "sedan", "taxi", "van", "policecar", "ambulance"] as const)[int(r, 15)], c + int(r, Math.max(1, gap - 2)));
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
      if (w >= 14 && r() < 0.5) underground(c0, w);
      enemies(c0, w, -1, 0.6);
      break;
    }
    case "highway": {
      // A highway: asphalt with lane markings, lamps along it, traffic stopped on it, signs, a convoy
      w = 32 + int(r, 22);
      ground(c0, w);
      const g = L.h;
      for (let c = c0; c < c0 + w; c++) put(L, c, g, ASPHALT, t.fill);
      for (let c = c0 + 2; c < c0 + w; c += 8) prop("streetlamp", c);
      for (let k = 2 + int(r, 3); k > 0; k--) prop((["sedan", "sedanblue", "hatchback", "taxi", "van", "deliverytruck", "pickup", "bus", "carwreck", "jerseybarrier", "trafficbarrels"] as const)[int(r, 11)], c0 + 3 + int(r, w - 8));
      standing(c0, w);
      if (r() < 0.6) prop("billboard", c0 + int(r, w - 4), true);
      spawn(r() < 0.5 ? "armoredcar" : "apc", c0 + w - 6);
      if (r() < 0.5) spawn((["truck", "medtank", "artytruck", "jeep"] as const)[int(r, 4)], c0 + w - 14);
      enemies(c0, w, -1, 0.8);
      break;
    }
    case "airfield": {
      // An airfield: a long runway, a control tower and hangars, aircraft parked on the apron,
      // runway lights, a windsock, its flak and its guards
      w = 56 + int(r, 24);
      ground(c0, w);
      const g = L.h;
      for (let c = c0; c < c0 + w; c++) put(L, c, g, ASPHALT, t.fill);
      let c = c0 + 1;
      const tower = SPEC_BY_KEY.get("controltower");
      if (tower && g - tower.fh * tower.floors[0] > 12) c += building(L, c, tower, out, troop);
      for (let k = c + 2; k < c0 + w - 4; k += 6) prop("runwaylight", k);
      for (let k = 2 + int(r, 3); k > 0; k--) prop(r() < 0.6 ? "parkedjet" : "parkedheli", c + 3 + int(r, Math.max(1, c0 + w - c - 24)));
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
      // A street: road and pavement, a row of buildings with the street furniture between them,
      // parked cars, and maybe a sewer or a subway under it all
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
      if (w >= 14 && r() < 0.55) underground(c0, w);
      enemies(c0, 3);
      break;
    }
    case "ocean": {
      // Open sea, as deep as the world: everyone but the vehicles swims it. Reefs and islands,
      // battleships, submarines and patrol boats out on it.
      const ow = 120 + int(r, 120);
      w = ow + 8;
      ground(c0, 4);
      const g = L.h;
      const bed = ROWS - 4;
      let island = 30 + int(r, 30);
      for (let c = c0 + 4; c < c0 + 4 + ow; c++) {
        const k = c - c0 - 4;
        const isle = k >= island && k < island + 8;
        const floor = isle ? g : Math.min(ROWS - 3, Math.max(g + 6, bed - Math.round((noise2(L.seed + 61, c / 8, 0) - 0.4) * 10)));
        column(L, c, floor);
        put(L, c, floor, isle ? t.top : SAND);
        for (let row = g + 1; row < floor; row++) put(L, c, row, WATER, 0);
        if (k === island + 8) island += 50 + int(r, 40);
      }
      for (let c = c0 + 10; c < c0 + ow; c += 3 + int(r, 6)) {
        const floor = L.ground[c & MASK];
        if (floor > g + 6) placeAt(L, r() < 0.6 ? "seaweed" : "coral", c, floor);
        else if (floor === g && r() < 0.5) deco(c - 2, 5);
      }
      for (let c = c0 + 16; c < c0 + ow - 10; c += 28 + int(r, 14)) out.push({ kind: L.ground[c & MASK] > g ? "battleship" : "flakgun", x: c * TILE, y: (g + 1) * TILE });
      for (let k = 2 + int(r, 3); k > 0; k--) out.push({ kind: "submarine", x: (c0 + 10 + int(r, ow - 20)) * TILE, y: (g + 1) * TILE });
      for (let k = 1 + int(r, 3); k > 0; k--) out.push({ kind: "boat", x: (c0 + 10 + int(r, ow - 20)) * TILE, y: (g + 1) * TILE });
      ground(c0 + 4 + ow, 4);
      prop("lighthousebig", c0 + 5 + ow, true);
      break;
    }
    case "lake": {
      // A deep lake: swim it, or go round on the jetpack
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
    case "forest": {
      // Deep woods: big trees close together, undergrowth, a few troops among them
      w = 18 + int(r, 14);
      natural(c0, w);
      for (let c = c0 + 2; c < c0 + w - 2; c += 3 + int(r, 3)) {
        if (t.decor === "pines" || t.top === SNOW) pine(c, gr(c));
        else if (t.decor === "deadwood") deadTree(c, gr(c));
        else bigTree(c, gr(c), c0, c0 + w);
      }
      for (let c = c0; c < c0 + w; c++) if (r() < 0.35) soft(c, gr(c) - 1, LEAF);
      for (let k = 1 + int(r, 3); k > 0; k--) prop((["stump", "logs", "flowers", "hedge"] as const)[int(r, 4)], c0 + int(r, w - 2));
      enemies(c0, w, -1, 1.1);
      break;
    }
    case "railway": {
      // A rail yard: track along the ground, freight cars on the sidings behind, a signal, a
      // water tower, and an armoured train coming down the line
      w = 44 + int(r, 20);
      ground(c0, w);
      const g = L.h;
      for (let c = c0; c < c0 + w; c++) put(L, c, g - 1, RAIL);
      for (let c = c0 + 3; c < c0 + w - 8; c += 7 + int(r, 6)) prop((["boxcar", "tankcar", "flatcar", "boxcar"] as const)[int(r, 4)], c, true);
      prop("railsignal", c0 + 1);
      prop(r() < 0.5 ? "watertower" : "crane", c0 + (w >> 1), true);
      out.push({ kind: "train", x: (c0 + w - 12) * TILE, y: g * TILE });
      enemies(c0, w, -1, 0.8);
      break;
    }
    case "nuclear": {
      // A nuclear plant: cooling towers steaming behind, the reactor hall, waste drums, fences, guns
      w = 30 + int(r, 10);
      ground(c0, w);
      const g = L.h;
      prop("coolingtower", c0 + 1, true);
      prop("coolingtower", c0 + 9, true);
      const spec = SPEC_BY_KEY.get("reactor");
      let c = c0 + 2;
      if (spec && g - spec.fh * spec.floors[0] > 12) c += building(L, c, spec, out, troop);
      for (const fx of [c0, c0 + w - 1]) fill(L, fx, g - 2, 1, 2, FENCE);
      for (let k = 2 + int(r, 3); k > 0; k--) prop(r() < 0.7 ? "toxicdrum" : "hazardsign", c + int(r, Math.max(1, c0 + w - c - 2)));
      spawn(r() < 0.5 ? "samsite" : "flakgun", c0 + w - 4);
      enemies(c0, w, -1, 1);
      break;
    }
    case "harbour": {
      // A harbour basin, deep water under a long pier, a battleship riding in it, docks either side
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
      if (r() < 0.3) underground(c0, w);
      if (t.enemies > 0 && c0 > 70 && r() < 0.25) for (let k = 1 + int(r, 2); k > 0; k--) spawn("mine", c0 + 1 + int(r, w - 2));
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
      // A hut on stilts: you run underneath, they shoot down from it
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
        L.props.push({ kind: "support", x: (c0 + 2) * TILE + 1, y: (g - ht - 4) * TILE, h: 4 * TILE }, { kind: "support", x: (c0 + 6) * TILE - 1, y: (g - ht - 4) * TILE, h: 4 * TILE });
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
      // A tunnel under the ground with its own garrison: blow the roof in to reach them
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
      for (let c = c0 + 2; c < c0 + cw; c += 5) L.props.push({ kind: "support", x: px(c), y: (row + 1) * TILE, h: 4 * TILE });
      climb(c0 + cw, row, L.h);
      scatter(c0 + 2, cw - 2, 1);
      enemies(c0 + 1, cw, row);
      enemies(c0 + 1, cw);
      barrels(c0 + 1, cw);
      break;
    }
    case "tank": {
      // A fuel depot: tanks and pipes
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
      const armour: Spawn["kind"][] = ["jeep", "truck", "lighttank", "medtank", "heavytank", "flametank", "rockettank", "aatank", "tanker"];
      spawn(armour[int(r, armour.length)], c0 + 13);
      break;
    }
    case "road": {
      // A dirt track: a troop truck is coming down it
      w = 18;
      ground(c0, w);
      for (let c = c0; c < c0 + w; c++) put(L, c, L.h, t.fill);
      spawn("truck", c0 + 12);
      if (r() < 0.4) spawn("jeep", c0 + 4);
      break;
    }
    case "canopy": {
      // Rainforest giants with vines, undergrowth, sometimes a lookout platform up in the trees
      w = 14 + int(r, 8);
      natural(c0, w);
      for (let c = c0 + 3; c < c0 + w - 3; c += 5 + int(r, 3)) bigTree(c, gr(c), c0, c0 + w);
      for (let c = c0; c < c0 + w; c++) if (r() < 0.25) soft(c, gr(c) - 1, LEAF);
      if (r() < 0.55) {
        const pc = c0 + 2 + int(r, w - 7);
        // (on a slope: high enough over the higher end, its legs down to the lower)
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
      // A bamboo thicket: good cover for an ambush
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
      // A river, waist deep: wade across, or take the rope bridge; a patrol boat may be on it
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
      // A cliff to climb, a stream pouring over it into a pool at its foot
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
      // A stepped temple of mossy stone, guarded on every tier, a shrine and a gun on top
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
      L.props.push({ kind: "brazier", x: px(sx), y: roof * TILE }, { kind: "brazier", x: px(sx + sw - 1), y: roof * TILE });
      spawn(r() < 0.5 ? "mg" : "turret", sx + (sw >> 1), lintel);
      spawn(troop(), sx + (sw >> 1), roof);
      break;
    }
    case "statues": {
      // Overgrown ruins: broken pillars wrapped in vines, a fallen one on the ground
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
      // An enemy camp: tents round a campfire, crates and barrels, plenty of guards
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
      // A missile silo: a hatch in a concrete pad (sometimes a missile goes up out of it)
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
      // A shelled building, several floors high, with enemies on every floor
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

/**
 * The enemy stronghold at the end of every level: sandbags and a machine-gun nest, an outer
 * wall with a gate, a courtyard of fuel tanks, a tall keep bristling with turrets, a back
 * wall. The squad stops short of it and calls in the ICBMs (they aim at plan.targets).
 */
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
  // The keep: three floors, turrets and a radar on the roof
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
