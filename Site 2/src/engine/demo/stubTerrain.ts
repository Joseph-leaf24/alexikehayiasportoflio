import type { SpriteBuffer, TerrainView } from "../render/types";
import type { DemoGround } from "./a2Ground";

/** The demo's materials (its own small table; the real ids come from `terrain/materials.ts`). */
export const M = {
  AIR: 0,
  DIRT: 1,
  GRASS: 2,
  STONE: 3,
  SAND: 4,
  BRICK: 5,
  LAVA: 6,
  CRYSTAL: 7,
  WATER: 8,
  WOOD: 9,
  METAL: 10,
} as const;
export const MATERIALS = 11;
export const VARIANTS = 4;

const ROWS = 240;
const RING = 512;
const CHUNK = 16;
const CHUNK_ROWS = ROWS / CHUNK;

const hash = (a: number, b: number) => {
  let h = Math.imul(a | 0, 374761393) ^ Math.imul(b | 0, 668265263);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
};

/** Smooth 1D value noise. */
const noise = (x: number, seed: number) => {
  const i = Math.floor(x);
  const f = x - i;
  const u = f * f * (3 - 2 * f);
  return hash(i, seed) * (1 - u) + hash(i + 1, seed) * u;
};

/**
 * A stand-in for A2's terrain: the same layout (8 px cells, 240 rows, a 512-column ring, 16×16 chunks), filled by a
 * procedural generator as the view moves right, with a crude carve for clicks.
 */
export class StubTerrain implements TerrainView, DemoGround {
  readonly rows = ROWS;
  readonly ring = RING;
  readonly tile = 8;
  readonly chunkSize = CHUNK;
  readonly variants = VARIANTS;
  startCol = 0;
  /** One past the last generated column. */
  next = 0;
  fg = new Uint8Array(RING * ROWS);
  bg = new Uint8Array(RING * ROWS);
  variant = new Uint8Array(RING * ROWS);
  hp = new Uint8Array(RING * ROWS);
  tiles = new Int32Array(MATERIALS * VARIANTS).fill(-1);
  emissive = new Uint8Array(256);
  maxHp = new Uint8Array(256).fill(255);
  private dirty = new Uint8Array((RING / CHUNK) * CHUNK_ROWS);
  private dirtyList: number[] = [];

  constructor(private seed = 7) {
    this.emissive[M.LAVA] = 255;
    this.emissive[M.CRYSTAL] = 200;
    const hp: [number, number][] = [
      [M.DIRT, 6],
      [M.GRASS, 6],
      [M.SAND, 4],
      [M.STONE, 14],
      [M.BRICK, 10],
      [M.WOOD, 5],
      [M.CRYSTAL, 12],
      [M.METAL, 20],
    ];
    for (const [m, h] of hp) this.maxHp[m] = h;
    this.ensure(RING);
  }

  get view(): TerrainView {
    return this;
  }

  /** The stub has no physics. */
  step(_dt: number) {}

  sprites(_out: SpriteBuffer, _layer: number) {}

  drainDirty(out: number[]): number {
    const n = this.dirtyList.length;
    for (let i = 0; i < n; i++) {
      out.push(this.dirtyList[i]);
      this.dirty[this.dirtyList[i]] = 0;
    }
    this.dirtyList.length = 0;
    return n;
  }

  private mark(col: number, row: number) {
    const k = ((col & (RING - 1)) >> 4) * CHUNK_ROWS + (row >> 4);
    if (!this.dirty[k]) {
      this.dirty[k] = 1;
      this.dirtyList.push(k);
    }
  }

  /** Generates columns up to `colEnd`, recycling the oldest ring slots. */
  ensure(colEnd: number) {
    while (this.next < colEnd) {
      this.column(this.next++);
      if (this.next - this.startCol > RING) this.startCol = this.next - RING;
    }
  }

  surface(col: number) {
    const s = this.seed;
    return Math.round(120 + (noise(col / 40, s) - 0.5) * 50 + (noise(col / 9, s + 1) - 0.5) * 10);
  }

  private column(col: number) {
    const base = (col & (RING - 1)) * ROWS;
    const top = this.surface(col);
    const s = this.seed;
    const town = noise(col / 30, s + 5) > 0.66;
    const tower =
      town && hash(Math.floor(col / 7), s + 9) > 0.35 ? 8 + Math.floor(hash(Math.floor(col / 7), s + 10) * 22) : 0;
    const pond = !town && noise(col / 22, s + 7) > 0.72;
    for (let r = 0; r < ROWS; r++) {
      const i = base + r;
      let f: number = M.AIR;
      let b: number = M.AIR;
      if (r >= top) {
        const d = r - top;
        const cave = noise(col / 11 + r * 0.37, s + 3) > 0.62 && d > 14 && noise(r / 6 + col * 0.21, s + 4) > 0.45;
        f = d === 0 ? M.GRASS : d < 6 ? M.DIRT : d < 9 && hash(col, r) > 0.6 ? M.SAND : M.STONE;
        if (d > 30 && hash(col >> 2, r >> 2) > 0.985) f = M.CRYSTAL;
        if (cave) {
          b = M.STONE;
          f = d > 60 && r > 200 ? M.LAVA : M.AIR;
        }
        if (r > 228) f = M.LAVA;
      } else if (pond && r >= top - 3) {
        f = M.WATER;
        b = M.SAND;
      } else if (tower && r >= top - tower) {
        // Towers: brick walls with back walls inside, metal roofs, wooden floors
        const lx = col % 7;
        const h = top - r;
        b = M.BRICK;
        f = lx === 0 || lx === 6 ? M.BRICK : h % 6 === 0 ? M.WOOD : M.AIR;
        if (h === tower) f = M.METAL;
        if ((lx === 2 || lx === 4) && h % 6 === 3) f = M.AIR;
      }
      this.fg[i] = f;
      this.bg[i] = b;
      this.variant[i] = (hash(col, r * 7 + 1) * VARIANTS) | 0;
      this.hp[i] = this.maxHp[f];
      if (r % CHUNK === 0) this.mark(col, r);
    }
  }

  solidAt(px: number, py: number) {
    const c = Math.floor(px / 8);
    const r = Math.floor(py / 8);
    if (r < 0 || r >= ROWS || c < this.startCol || c >= this.next) return false;
    const m = this.fg[(c & (RING - 1)) * ROWS + r];
    return m !== M.AIR && m !== M.WATER && m !== M.LAVA;
  }

  /** First solid row at or below `fromRow` in a column, in px. */
  groundY(px: number) {
    const c = Math.floor(px / 8);
    if (c < this.startCol || c >= this.next) return 960;
    const base = (c & (RING - 1)) * ROWS;
    for (let r = 0; r < ROWS; r++) {
      const m = this.fg[base + r];
      if (m !== M.AIR && m !== M.WATER) return r * 8;
    }
    return ROWS * 8;
  }

  /** Knocks hp off cells in a radius, removing those that reach 0; returns how many cells it removed. */
  carve(x: number, y: number, r: number, power: number) {
    const c0 = Math.floor((x - r) / 8);
    const c1 = Math.floor((x + r) / 8);
    const r0 = Math.max(0, Math.floor((y - r) / 8));
    const r1 = Math.min(ROWS - 1, Math.floor((y + r) / 8));
    let removed = 0;
    for (let c = Math.max(c0, this.startCol); c <= Math.min(c1, this.next - 1); c++)
      for (let row = r0; row <= r1; row++) {
        const d = Math.hypot(c * 8 + 4 - x, row * 8 + 4 - y) / r;
        if (d > 1) continue;
        const i = (c & (RING - 1)) * ROWS + row;
        const m = this.fg[i];
        if (m === M.AIR || this.hp[i] === 255) continue;
        const dmg = Math.ceil(power * (1 - d * d));
        if (this.hp[i] <= dmg) {
          this.fg[i] = M.AIR;
          if (!this.bg[i] && row > this.surface(c)) this.bg[i] = M.STONE;
          removed++;
        } else this.hp[i] -= dmg;
        this.mark(c, row);
      }
    return removed;
  }
}
