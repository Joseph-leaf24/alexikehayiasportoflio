/**
 * One level of the run-and-gun behind the site: a side-scrolling stretch of 8×8 blocks
 * generated a chunk at a time just ahead of the camera, in a theme (themes.ts). Every level
 * opens on a landing zone, may have a boss arena half way, and ends at an enemy stronghold
 * that the squad flattens with ICBMs before moving on to the next, random level. Blocks
 * live in a ring buffer of columns, so a level of any length never grows in memory.
 * Pure data and maths: game.ts plays it, PixelScene draws it.
 */
import { int, rng } from "./rng";
import type { BossKind, Theme, Troop } from "./themes";

/** Edge of one block, in scene pixels. */
export const TILE = 8;
/** Rows of blocks: deep enough for tunnels under the ground and tall structures over it. */
export const ROWS = 64;
/** Columns kept in memory (a power of two). */
export const RING = 512;
export const MASK = RING - 1;
/** Where the ground usually is. */
export const BASE_ROW = 44;
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
export const KINDS = 34;

/** What stops people and bullets. Leaves, trunks, pine boughs, vines, bamboo, water and lava don't. */
export const SOLID = new Uint8Array(KINDS).map((_, k) =>
  k === AIR || k === LEAF || k === TRUNK || k === LAVA || k === PINE || k === VINE || k === WATER || k === BAMBOO ? 0 : 1
);
/** Hits a block takes before it breaks (255: never, so blasts can't drain rivers or lava). */
export const TILE_HP = Uint8Array.from([0, 2, 2, 4, 255, 3, 5, 1, 2, 1, 6, 1, 1, 1, 4, 2, 5, 6, 255, 3, 3, 1, 3, 1, 3, 1, 3, 2, 2, 4, 5, 1, 255, 1]);

/** Something the generator placed for game.ts to bring to life. x, y: world px (y = feet; a boat's y is the water line). */
export interface Spawn {
  kind: Troop | "mortar" | "mg" | "turret" | "jeep" | "truck" | "boat" | "barrel" | "fuel" | "tanker" | "ammo" | "egg";
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
  | { kind: "brazier"; x: number; y: number };

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
  /** Ground row of each stored column */
  ground: Int16Array;
  /** Oldest column still stored, and the next one to generate (absolute column numbers) */
  start: number;
  next: number;
  /** The generator's current ground row */
  h: number;
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
}

export function createLevel(theme: Theme, seed = 1, opts: LevelOptions = {}): Level {
  const rand = rng(seed);
  const length = opts.length ?? 150 + int(rand, 100);
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
    theme,
    plan: { length, boss: kind ? { col: Math.floor(length * 0.48), kind } : null, bossPlaced: false, stronghold: length, fortPlaced: false, targets: [] },
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

/** Write one fresh column of ground with its top at row `h` (dropping the oldest stored column if the ring is full). */
function column(L: Level, c: number, h: number) {
  if (c - L.start >= RING) L.start = c - RING + 1;
  const t = L.theme;
  const sc = c & MASK;
  for (let r = 0; r < ROWS; r++) {
    const i = r * RING + sc;
    L.fg[i] = L.bg[i] = L.hp[i] = L.burn[i] = 0;
  }
  for (let r = h; r < ROWS; r++) {
    const k = r >= ROWS - 2 ? BEDROCK : r === h ? t.top : r < h + 4 ? t.fill : t.deep;
    put(L, c, r, k, r < h + 4 ? t.fill : t.deep);
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
    if (r() >= t.barrels) return;
    const n = 1 + int(r, 3);
    const c = from + int(r, Math.max(1, w - n));
    for (let k = 0; k < n; k++) spawn(r() < 0.25 ? "ammo" : "barrel", c + k);
  };
  const ground = (from: number, w: number, h = L.h) => {
    for (let c = from; c < from + w; c++) column(L, c, h);
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
  /** A rainforest giant: a tall trunk, a wide crown, vines hanging from it. */
  const bigTree = (c: number, g: number, lo: number, hi: number) => {
    const th = 9 + int(r, 6);
    fill(L, c, g - th, 1, th, TRUNK);
    const top = g - th - 2;
    [3, 7, 9, 7].forEach((cw, k) => {
      for (let x = c - (cw >> 1); x <= c + (cw >> 1); x++) soft(x, top + k, LEAF, lo, hi);
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
    else if (t.decor === "jungle") {
      const n = r();
      if (n < 0.45) bigTree(c, g, from, from + w);
      else if (n < 0.75) fern(c, g, from, from + w);
      else palm(c, g);
    } else if (t.decor === "pines" && r() < 0.7) pine(c, g);
    else if (t.decor === "deadwood" && r() < 0.6) deadTree(c, g);
    else if (t.decor === "fires" && r() < 0.5) L.props.push({ kind: "fire", x: px(c), y: g * TILE });
    else if (t.decor === "crystals" && r() < 0.6) put(L, c, g - 1, CRYSTAL);
    else if (t.decor === "rubble" && r() < 0.6) {
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
    L.next = c0 + w;
    return;
  }
  if (plan.boss && !plan.bossPlaced && c0 >= plan.boss.col - 4) {
    // A wide, flat arena for the boss fight
    w = 40;
    ground(c0, w);
    plan.boss.col = c0 + 14;
    plan.bossPlaced = true;
    barrels(c0 + 4, 8);
    barrels(c0 + 26, 10);
    skyline(c0, 20);
    L.next = c0 + w;
    return;
  }
  if (!plan.fortPlaced && c0 >= plan.stronghold) {
    stronghold(L, c0, out, troop, night);
    return;
  }
  if (plan.fortPlaced) {
    // After the stronghold: quiet ground (the level ends before anyone gets here)
    w = 16;
    ground(c0, w);
    L.next = c0 + w;
    return;
  }

  switch (pick(t.features, r)) {
    case "flat": {
      w = 8 + int(r, 8);
      ground(c0, w);
      skyline(c0, w);
      deco(c0, w);
      enemies(c0, w);
      barrels(c0, w);
      break;
    }
    case "slope": {
      const dir = L.h <= BASE_ROW - 5 ? 1 : L.h >= BASE_ROW + 3 ? -1 : r() < 0.5 ? -1 : 1;
      let c = c0;
      for (let k = 1 + int(r, 4); k > 0; k--) {
        const run = 2 + int(r, 2);
        ground(c, run);
        c += run;
        L.h += dir;
      }
      ground(c, 3);
      c += 3;
      w = c - c0;
      skyline(c0, w);
      deco(c0, w);
      enemies(c - 4, 4);
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
      w = 16;
      ground(c0, w);
      spawn("jeep", c0 + 4);
      const n = r();
      if (n < 0.4) spawn("jeep", c0 + 10);
      else if (n < 0.75) spawn("truck", c0 + 12);
      else spawn("tanker", c0 + 11);
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
      ground(c0, w);
      const g = L.h;
      for (let c = c0 + 3; c < c0 + w - 3; c += 5 + int(r, 3)) bigTree(c, g, c0, c0 + w);
      for (let c = c0; c < c0 + w; c++) if (r() < 0.25) soft(c, g - 1, LEAF);
      if (r() < 0.55) {
        const pc = c0 + 2 + int(r, w - 7);
        const row = g - 6 - int(r, 2);
        fill(L, pc, row, 4, 1, WOOD);
        L.props.push({ kind: "legs", x: pc * TILE, y: (row + 1) * TILE, w: 4 * TILE, h: (g - row - 1) * TILE });
        spawn(r() < 0.6 ? "sniper" : troop(), pc + 2, row);
      }
      enemies(c0, w, -1, 1.1);
      break;
    }
    case "bamboo": {
      // A bamboo thicket: good cover for an ambush
      w = 12 + int(r, 6);
      ground(c0, w);
      const g = L.h;
      for (let c = c0 + 1; c < c0 + w - 1; c++) {
        if (r() > 0.55) continue;
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
      const rise = Math.min(4 + int(r, 3), L.h - (BASE_ROW - 9));
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
      ground(c0, w);
      for (let c = c0 + 2; c < c0 + w - 1; c += 4 + int(r, 3)) pine(c, L.h);
      enemies(c0, w);
      break;
    }
    case "deadtrees": {
      w = 12 + int(r, 6);
      ground(c0, w);
      for (let c = c0 + 2; c < c0 + w - 1; c += 5 + int(r, 3)) deadTree(c, L.h);
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
      ground(c0, w);
      for (let k = 1 + int(r, 2); k > 0; k--) {
        const c = c0 + 1 + int(r, w - 3);
        const ht = 3 + int(r, 6);
        fill(L, c, L.h - ht, r() < 0.4 ? 2 : 1, ht, t.wall);
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
  // Outer wall with a gate
  const ow = c0 + 9;
  fill(L, ow, g - 10, 3, 10, t.build, t.build);
  fill(L, ow, g - 3, 3, 3, AIR, t.build);
  for (const c of [ow, ow + 2]) put(L, c, g - 11, t.build);
  spawn("turret", ow + 1, g - 10);
  L.props.push({ kind: "flag", x: ow * TILE + 2, y: (g - 11) * TILE });
  if (night) L.props.push({ kind: "searchlight", x: (ow + 3) * TILE, y: (g - 10) * TILE - 3 });
  // Courtyard
  spawn("fuel", c0 + 14);
  spawn("tanker", c0 + 18);
  spawn("barrel", c0 + 21);
  spawn("ammo", c0 + 22);
  crowd(c0 + 13, 8, g, 3);
  // The keep: three floors, turrets and a radar on the roof
  const kx = c0 + 24;
  const kw = 14;
  const kh = 14;
  fill(L, kx, g - kh, kw, kh, AIR, t.build);
  fill(L, kx, g - kh, 1, kh - 2, t.build, t.build);
  fill(L, kx + kw - 1, g - kh, 1, kh - 2, t.build, t.build);
  fill(L, kx, g - kh - 1, kw, 1, t.build, t.build);
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
