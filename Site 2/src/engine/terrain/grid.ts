/**
 * The destructible cell world: a column ring of material cells with dirty tracking for the renderer, chunk wake-ups
 * for the physics, and an event ring for the game's particles. Physics, carving, queries and structural collapse
 * live in their own files and work on this class's arrays; its methods are thin entry points to them.
 */
import type { TerrainView } from "../render/types";
import {
  CELLS,
  CHUNK,
  CHUNK_ROWS,
  CHUNKS,
  chunkIndex,
  F_BURNING,
  F_LOOSE,
  F_REINFORCED,
  F_STRUCTURAL,
  MASK,
  RING,
  ROWS,
  TERRAIN_TICK,
  TILE,
  VARIANTS,
  EV_IGNITE,
  F_WET,
  WAKE,
} from "./constants";
import {
  AIR,
  BEDROCK,
  EMISSIVE,
  FLAMMABLE,
  FUEL,
  HP,
  LOOSE,
  MATERIAL_COUNT,
  RUBBLE,
  RUBBLE_CHANCE,
  SOLID,
  STRUCTURAL,
} from "./materials";
import { hashCell, TerrainRng } from "./random";
import { Physics, type PhysicsConfig } from "./physics";
import { Structure, type StructureConfig } from "./structure";
import type { FallingChunk } from "./falling";
import { carve, createCarveResult, damageCell, flood, ignite, type CarveResult } from "./carve";
import * as Q from "./query";

/**
 * Fills columns [colFrom, colTo) of a fresh stretch. It may run on (to finish a feature) by up to the terrain's
 * `overshoot` columns, and returns where it stopped.
 */
export type TerrainGenerator = (terrain: Terrain, colFrom: number, colTo: number) => number | void;

export interface TerrainOptions {
  /** Seeds tile variants and every random choice the simulation makes. */
  seed?: number;
  /** Called by `ensure` to fill new columns, like the old engine's `extend()`. */
  generate?: TerrainGenerator;
  /**
   * Wake the physics on freshly generated columns so loose ground and liquids settle at once. Off by default:
   * generated terrain stays as placed (as in the old engine) until something disturbs it.
   */
  wakeOnGenerate?: boolean;
  /** Columns a generator may write past the stretch it was asked for (default 128); kept free ahead of it. */
  overshoot?: number;
  physics?: Partial<PhysicsConfig>;
  structure?: Partial<StructureConfig>;
}

export interface SetOptions {
  /** Background material to write too. */
  bg?: number;
  /** Hit points; default the material's full hp. 255 makes the cell unbreakable. */
  hp?: number;
  /** Exact flags; default the material's (loose for powders, structural for building materials). */
  flags?: number;
  /** Force the structural flag on or off (a stone castle wall is structural, stone ground isn't). */
  structural?: boolean;
  reinforced?: boolean;
  /** Tile variant; default a hash of the position, so regenerated levels look the same. */
  variant?: number;
}

/** What the last `step` did, for the debug overlay and the budget tests. */
export interface TerrainStats {
  tick: number;
  /** Cell moves (falls, slides, flows, swaps). Capped by `physics.maxMoves`. */
  moves: number;
  /** Chunks simulated, and chunks still awake afterwards. */
  chunks: number;
  awake: number;
  /** Burning cells seen. */
  burning: number;
  /** Structure work: flood-fill visits plus cells lifted, shattered and moved. Capped by `structure.budget`. */
  work: number;
  /** Cells visited by flood fills alone. */
  fill: number;
  /** Seeds waiting for a support check. */
  queued: number;
  /** Rigid chunks in flight or shattering, and their cells. */
  falling: number;
  fallingCells: number;
}

const EVENT_CAP = 1024;

const DEFAULT_FLAGS = Uint8Array.from(
  { length: 256 },
  (_, m) => (LOOSE[m] ? F_LOOSE : 0) | (STRUCTURAL[m] ? F_STRUCTURAL : 0)
);

export class Terrain implements TerrainView {
  readonly rows = ROWS;
  readonly ring = RING;
  readonly tile = TILE;
  readonly chunkSize = CHUNK;
  readonly variants = VARIANTS;

  /** Material per cell. */
  readonly fg = new Uint8Array(CELLS);
  /** Background (back-wall) material per cell. */
  readonly bg = new Uint8Array(CELLS);
  /** Hits left; 255 never breaks. */
  readonly hp = new Uint8Array(CELLS);
  /** F_* bits. */
  readonly flags = new Uint8Array(CELLS);
  /** Tile variant 0..VARIANTS-1. */
  readonly variant = new Uint8Array(CELLS);
  /** Fire: ticks of fuel left in a burning cell. */
  readonly heat = new Uint8Array(CELLS);

  /** Atlas frame per material tile: `tiles[material * VARIANTS + variant]`, -1 until the game paints it. */
  readonly tiles = new Int32Array(MATERIAL_COUNT * VARIANTS).fill(-1);
  readonly emissive = EMISSIVE;
  readonly maxHp = HP;

  /** Oldest stored column, and one past the newest. Columns outside are air. */
  startCol = 0;
  endCol = 0;
  tick = 0;
  readonly seed: number;
  readonly rng: TerrainRng;
  readonly physics: Physics;
  readonly structure: Structure;
  readonly stats: TerrainStats = {
    tick: 0,
    moves: 0,
    chunks: 0,
    awake: 0,
    burning: 0,
    work: 0,
    fill: 0,
    queued: 0,
    falling: 0,
    fallingCells: 0,
  };

  /** Per chunk: ticks left awake (0 = asleep). */
  readonly awake = new Uint8Array(CHUNKS);
  private readonly dirty = new Uint32Array(Math.ceil(CHUNKS / 32));
  /** Per ring column: row of the topmost solid cell, or -1 if unknown. */
  private readonly top = new Int16Array(RING).fill(-1);
  private readonly events = new Int32Array(EVENT_CAP * 4);
  private evStart = 0;
  private evCount = 0;
  private generating = false;
  private readonly generator?: TerrainGenerator;
  private readonly wakeOnGenerate: boolean;
  private readonly overshoot: number;
  private readonly carveOut = createCarveResult();

  constructor(opts: TerrainOptions = {}) {
    this.seed = (opts.seed ?? 1) | 0;
    this.rng = new TerrainRng(hashCell(this.seed, 0x7e44, 0x5eed));
    this.generator = opts.generate;
    this.wakeOnGenerate = !!opts.wakeOnGenerate;
    this.overshoot = Math.max(0, Math.min(RING >> 2, opts.overshoot ?? 128));
    this.physics = new Physics(this, opts.physics);
    this.structure = new Structure(this, opts.structure);
  }

  /* ---------------------------------------------------------------- cells */

  /** Ring index of a column and row (no bounds checks). */
  index(col: number, row: number): number {
    return (col & MASK) * ROWS + row;
  }

  /** Whether a column is stored (generated and not yet recycled). */
  stored(col: number): boolean {
    return col >= this.startCol && col < this.endCol;
  }

  /** Absolute column of a ring column. */
  colOfRing(ringCol: number): number {
    return this.startCol + ((ringCol - this.startCol) & MASK);
  }

  /** Material at a cell: air above the world and outside the window, bedrock below it. */
  get(col: number, row: number): number {
    if (row < 0) return AIR;
    if (row >= ROWS) return BEDROCK;
    if (col < this.startCol || col >= this.startCol + RING) return AIR;
    return this.fg[(col & MASK) * ROWS + row];
  }

  /**
   * Write a cell. Outside generation it wakes the physics around it, marks it for upload and, if it removed
   * something that held other cells up, queues a support check. Returns false outside the writable window
   * (`startCol` to `startCol + RING`).
   */
  set(col: number, row: number, mat: number, opts?: SetOptions): boolean {
    if (row < 0 || row >= ROWS || col < this.startCol || col >= this.startCol + RING) return false;
    const i = (col & MASK) * ROWS + row;
    const was = this.fg[i];
    const wasFlags = this.flags[i];
    let fl = DEFAULT_FLAGS[mat];
    if (opts) {
      if (opts.flags !== undefined) fl = opts.flags;
      if (opts.structural !== undefined) fl = opts.structural ? fl | F_STRUCTURAL : fl & ~F_STRUCTURAL;
      if (opts.reinforced !== undefined) fl = opts.reinforced ? fl | F_REINFORCED : fl & ~F_REINFORCED;
      if (opts.bg !== undefined) this.bg[i] = opts.bg;
    }
    this.fg[i] = mat;
    this.hp[i] = opts?.hp ?? HP[mat];
    this.flags[i] = mat === AIR ? 0 : fl;
    this.variant[i] = opts?.variant ?? hashCell(col, row, this.seed) & (VARIANTS - 1);
    this.heat[i] = 0;
    if (this.generating) {
      this.top[col & MASK] = -1;
      return true;
    }
    this.touch(col, row);
    if (was !== AIR && (SOLID[was] || wasFlags & F_STRUCTURAL)) {
      const nowFlags = this.flags[i];
      if (!SOLID[mat] || (wasFlags ^ nowFlags) & (F_STRUCTURAL | F_LOOSE))
        this.structure.removed(col, row, was, wasFlags);
    }
    return true;
  }

  /** Write a background cell. */
  setBg(col: number, row: number, mat: number): boolean {
    if (row < 0 || row >= ROWS || col < this.startCol || col >= this.startCol + RING) return false;
    this.bg[(col & MASK) * ROWS + row] = mat;
    if (!this.generating) this.markDirty(col, row);
    return true;
  }

  /** Ring index of the cell under a world-px point, or -1 outside the stored world. */
  cellAt(px: number, py: number): number {
    const col = Math.floor(px / TILE);
    const row = Math.floor(py / TILE);
    if (row < 0 || row >= ROWS || col < this.startCol || col >= this.endCol) return -1;
    return (col & MASK) * ROWS + row;
  }

  /**
   * Break a cell for good: masonry may leave loose rubble, everything else leaves air. Wakes the cells around it
   * and queues a support check for what it held up. Internal to carving, acid and the physics.
   */
  breakCell(i: number, col: number, row: number): number {
    const m = this.fg[i];
    const fl = this.flags[i];
    const chance = RUBBLE_CHANCE[m];
    const left = chance && (this.rng.u32() & 255) < chance ? RUBBLE : AIR;
    this.fg[i] = left;
    this.hp[i] = HP[left];
    this.flags[i] = left ? F_LOOSE : 0;
    this.heat[i] = 0;
    this.touch(col, row);
    if (SOLID[m] || fl & F_STRUCTURAL) this.structure.removed(col, row, m, fl);
    return m;
  }

  /** Set a flammable cell alight (a wet one only dries). Returns whether it caught. */
  igniteCell(i: number, col: number, row: number): boolean {
    const m = this.fg[i];
    const fl = this.flags[i];
    if (!FLAMMABLE[m] || fl & F_BURNING) return false;
    if (fl & F_WET) {
      this.flags[i] = fl & ~F_WET;
      return false;
    }
    this.flags[i] = fl | F_BURNING;
    this.heat[i] = FUEL[m];
    this.touch(col, row);
    this.emit(EV_IGNITE, col, row, m);
    return true;
  }

  /* ---------------------------------------------------------------- generation and the ring */

  /**
   * Make sure columns up to `colEnd` exist: the generator fills each new stretch (at most half a ring at a time),
   * after the oldest columns are recycled to leave room for it plus `overshoot`. New columns are marked dirty.
   */
  ensure(colEnd: number): void {
    while (this.endCol < colEnd) {
      const from = this.endCol;
      const to = Math.min(colEnd, from + (RING >> 1));
      const room = to + this.overshoot - RING;
      if (room > this.startCol) this.recycle(room);
      let reached = to;
      if (this.generator) {
        this.generating = true;
        try {
          const r = this.generator(this, from, to);
          if (typeof r === "number" && r > reached) reached = r;
        } finally {
          this.generating = false;
        }
      }
      this.endCol = Math.min(reached, this.startCol + RING);
      this.columnsChanged(from, this.endCol, this.wakeOnGenerate);
    }
  }

  /** Drop columns left of `newStart`, clearing their ring slots for reuse. */
  private recycle(newStart: number): void {
    const n = Math.min(newStart - this.startCol, RING);
    for (let k = 0; k < n; k++) {
      const s = ((this.startCol + k) & MASK) * ROWS;
      const e = s + ROWS;
      this.fg.fill(0, s, e);
      this.bg.fill(0, s, e);
      this.hp.fill(0, s, e);
      this.flags.fill(0, s, e);
      this.variant.fill(0, s, e);
      this.heat.fill(0, s, e);
    }
    this.columnsChanged(this.startCol, this.startCol + n, false);
    this.startCol = newStart;
    if (this.endCol < newStart) this.endCol = newStart;
    this.structure.recycled(newStart);
  }

  /** Mark every chunk over [from, to) dirty (and awake if asked). */
  private columnsChanged(from: number, to: number, wake: boolean): void {
    for (let c = from; c < to; c++) {
      this.top[c & MASK] = -1;
      if (c !== from && (c & (CHUNK - 1)) !== 0) continue;
      const k0 = ((c & MASK) >> 4) * CHUNK_ROWS;
      for (let k = k0; k < k0 + CHUNK_ROWS; k++) {
        this.dirty[k >> 5] |= 1 << (k & 31);
        if (wake) this.awake[k] = WAKE;
      }
    }
  }

  /* ---------------------------------------------------------------- dirty chunks and waking */

  /**
   * A cell changed: queue its chunk for upload, wake it, and forget its column top. The nearer horizontal neighbour
   * wakes too, since a liquid there may now see a drop (liquids look up to 8 cells, half a chunk, sideways), and the
   * chunk above or below when the cell is on that edge.
   */
  touch(col: number, row: number): void {
    const k = ((col & MASK) >> 4) * CHUNK_ROWS + (row >> 4);
    this.dirty[k >> 5] |= 1 << (k & 31);
    const awake = this.awake;
    awake[k] = WAKE;
    this.top[col & MASK] = -1;
    const side = (col & 15) < 8 ? col - 8 : col + 8;
    awake[chunkIndex(side, row)] = WAKE;
    const lr = row & 15;
    if (lr !== 0 && lr !== 15) return;
    const r = lr === 0 ? row - 1 : row + 1;
    if (r < 0 || r >= ROWS) return;
    awake[chunkIndex(col, r)] = WAKE;
    awake[chunkIndex(side, r)] = WAKE;
  }

  /**
   * Wake the chunks within `reach` columns of a cell, on its row and the row above: a liquid cell that moved may
   * have opened (or closed) the drop another liquid cell up to `reach` away was looking for.
   */
  wakeSpan(col: number, row: number, reach: number): void {
    const awake = this.awake;
    const top = Math.max(0, row - 1) >> 4;
    const bottom = row >> 4;
    for (let c = col - reach; ; c += CHUNK) {
      const cc = Math.min(c, col + reach);
      const k0 = ((cc & MASK) >> 4) * CHUNK_ROWS;
      awake[k0 + top] = WAKE;
      awake[k0 + bottom] = WAKE;
      if (cc === col + reach) break;
    }
  }

  /** Queue a cell's chunk for upload without waking anything. */
  markDirty(col: number, row: number): void {
    const k = chunkIndex(col, row);
    this.dirty[k >> 5] |= 1 << (k & 31);
  }

  /** Wake every chunk overlapping a cell rectangle (inclusive). */
  wakeRect(c0: number, r0: number, c1: number, r1: number): void {
    r0 = Math.max(0, r0) >> 4;
    r1 = Math.min(ROWS - 1, r1) >> 4;
    c0 = Math.max(c0, this.startCol);
    c1 = Math.min(c1, this.endCol - 1);
    for (let c = c0; c <= c1; c = ((c >> 4) + 1) << 4) {
      const k0 = ((c & MASK) >> 4) * CHUNK_ROWS;
      for (let r = r0; r <= r1; r++) this.awake[k0 + r] = WAKE;
    }
  }

  /** Hand the renderer the chunks changed since the last call (`TerrainView` contract). */
  drainDirty(out: number[]): number {
    let n = 0;
    const d = this.dirty;
    for (let w = 0; w < d.length; w++) {
      let bits = d[w];
      if (bits === 0) continue;
      d[w] = 0;
      while (bits !== 0) {
        const low = bits & -bits;
        out[n++] = w * 32 + (31 - Math.clz32(low));
        bits ^= low;
      }
    }
    out.length = n;
    return n;
  }

  /**
   * Pack one chunk for a texture laid out like the ring (x = ring column, y = row): 16×16 RGBA8 texels, row-major,
   * R = fg, G = bg, B = variant, A = hp. `out` needs 1,024 bytes. Returns the chunk's ring column and row origin
   * in `origin` if given.
   */
  packChunk(k: number, out: Uint8Array, origin?: { x: number; y: number }): void {
    const c0 = ((k / CHUNK_ROWS) | 0) << 4;
    const r0 = k % CHUNK_ROWS << 4;
    for (let lc = 0; lc < CHUNK; lc++) {
      const base = (c0 + lc) * ROWS + r0;
      for (let lr = 0; lr < CHUNK; lr++) {
        const o = (lr * CHUNK + lc) * 4;
        out[o] = this.fg[base + lr];
        out[o + 1] = this.bg[base + lr];
        out[o + 2] = this.variant[base + lr];
        out[o + 3] = this.hp[base + lr];
      }
    }
    if (origin) {
      origin.x = c0;
      origin.y = r0;
    }
  }

  /** Chunks currently awake. */
  awakeChunks(): number {
    let n = 0;
    for (let k = 0; k < CHUNKS; k++) if (this.awake[k]) n++;
    return n;
  }

  /* ---------------------------------------------------------------- events */

  /** Queue a game event; the oldest is dropped when the ring is full. */
  emit(type: number, col: number, row: number, value: number): void {
    let slot: number;
    if (this.evCount < EVENT_CAP) slot = (this.evStart + this.evCount++) % EVENT_CAP;
    else {
      slot = this.evStart;
      this.evStart = (this.evStart + 1) % EVENT_CAP;
    }
    const o = slot * 4;
    this.events[o] = type;
    this.events[o + 1] = col;
    this.events[o + 2] = row;
    this.events[o + 3] = value;
  }

  /** Move queued events into `out` as (type, col, row, value) quadruples (EV_*). Returns the event count. */
  drainEvents(out: number[]): number {
    const n = this.evCount;
    for (let e = 0; e < n; e++) {
      const o = ((this.evStart + e) % EVENT_CAP) * 4;
      out[e * 4] = this.events[o];
      out[e * 4 + 1] = this.events[o + 1];
      out[e * 4 + 2] = this.events[o + 2];
      out[e * 4 + 3] = this.events[o + 3];
    }
    out.length = n * 4;
    this.evStart = 0;
    this.evCount = 0;
    return n;
  }

  /* ---------------------------------------------------------------- simulation */

  /** One fixed tick: cellular physics on awake chunks, then support checks and falling chunks, all budgeted. */
  step(dt = TERRAIN_TICK): void {
    this.tick++;
    this.physics.step();
    this.structure.step(dt);
    const s = this.stats;
    s.tick = this.tick;
    s.moves = this.physics.moves;
    s.chunks = this.physics.chunks;
    s.burning = this.physics.burning;
    s.awake = this.awakeChunks();
    s.work = this.structure.work;
    s.fill = this.structure.fill;
    s.queued = this.structure.queued;
    s.falling = this.structure.falling.chunks.length;
    s.fallingCells = this.structure.falling.cells();
  }

  /** Rigid chunks in flight (or shattering on the ground), for drawing and for crushing whatever they hit. */
  get falling(): readonly FallingChunk[] {
    return this.structure.falling.chunks;
  }

  /** Whether anything is still settling: awake chunks, queued checks, a check in progress or falling chunks. */
  get busy(): boolean {
    return this.awakeChunks() > 0 || this.structure.pending;
  }

  /**
   * Burning cells in stored columns [colFrom, colTo), as (col, row) pairs in `out`, for flames and lights. Only
   * awake chunks can hold fire, so this skips the sleeping world. Returns the count.
   */
  collectBurning(colFrom: number, colTo: number, out: number[]): number {
    let n = 0;
    const from = Math.max(colFrom, this.startCol);
    const to = Math.min(colTo, this.endCol);
    for (let c = from; c < to; c++) {
      const k0 = ((c & MASK) >> 4) * CHUNK_ROWS;
      const base = (c & MASK) * ROWS;
      for (let cr = 0; cr < CHUNK_ROWS; cr++) {
        if (!this.awake[k0 + cr]) continue;
        for (let r = cr << 4, e = r + CHUNK; r < e; r++)
          if (this.flags[base + r] & F_BURNING) {
            out[n * 2] = c;
            out[n * 2 + 1] = r;
            n++;
          }
      }
    }
    out.length = n * 2;
    return n;
  }

  /* ---------------------------------------------------------------- destruction (carve.ts) */

  /** A blast: see `carve` in carve.ts. Reuses one result object unless given `out`. */
  carve(x: number, y: number, r: number, power: number, out: CarveResult = this.carveOut): CarveResult {
    return carve(this, x, y, r, power, out);
  }

  /** Knock `hits` off one cell; returns the material that broke, or 0. */
  damage(col: number, row: number, hits: number): number {
    return damageCell(this, col, row, hits);
  }

  /** Set flammable cells within `r` px alight; returns how many caught. */
  ignite(x: number, y: number, r: number): number {
    return ignite(this, x, y, r);
  }

  /** Fill air within `r` px with `mat` (a flood of water, a lava burst); returns cells filled. */
  flood(x: number, y: number, r: number, mat: number): number {
    return flood(this, x, y, r, mat);
  }

  /* ---------------------------------------------------------------- queries (query.ts) */

  solidAt(px: number, py: number): boolean {
    return Q.solidAt(this, px, py);
  }
  liquidAt(px: number, py: number): number {
    return Q.liquidAt(this, px, py);
  }
  materialAt(px: number, py: number): number {
    return Q.materialAt(this, px, py);
  }
  surfaceY(px: number, fromY = 0): number {
    return Q.surfaceY(this, px, fromY);
  }
  lineOfSight(x0: number, y0: number, x1: number, y1: number): boolean {
    return Q.lineOfSight(this, x0, y0, x1, y1);
  }
  raycast(x0: number, y0: number, dx: number, dy: number, maxDist: number, out: Q.RayHit): boolean {
    return Q.raycast(this, x0, y0, dx, dy, maxDist, out);
  }
  overlapsSolid(x: number, y: number, hw: number, h: number): boolean {
    return Q.overlapsSolid(this, x, y, hw, h);
  }

  /** Row of the topmost solid cell in a column (ROWS if none, or if the column isn't stored). Cached per column. */
  columnTop(col: number): number {
    if (col < this.startCol || col >= this.endCol) return ROWS;
    const slot = col & MASK;
    const cached = this.top[slot];
    if (cached >= 0) return cached;
    const base = slot * ROWS;
    let r = 0;
    while (r < ROWS && !SOLID[this.fg[base + r]]) r++;
    this.top[slot] = r;
    return r;
  }

  /* ---------------------------------------------------------------- determinism */

  /** FNV-1a over the stored state (cells, rng, falling chunks): equal hashes mean equal worlds. */
  hash(): number {
    let h = 0x811c9dc5;
    const mixArr = (a: Uint8Array) => {
      for (let i = 0; i < a.length; i++) h = Math.imul(h ^ a[i], 0x01000193);
    };
    mixArr(this.fg);
    mixArr(this.bg);
    mixArr(this.hp);
    mixArr(this.flags);
    mixArr(this.variant);
    mixArr(this.heat);
    h = Math.imul(h ^ this.rng.state, 0x01000193);
    h = Math.imul(h ^ this.startCol, 0x01000193);
    for (const ch of this.falling) {
      h = Math.imul(h ^ ch.col, 0x01000193);
      h = Math.imul(h ^ Math.round(ch.y * 16), 0x01000193);
      mixArr(ch.fg);
    }
    return h >>> 0;
  }
}
