/** Shared worlds and helpers for the engine-terrain tests and benchmark. */
import { Terrain, type TerrainOptions } from "@/engine/terrain/grid";
import { F_STRUCTURAL, MASK, RING, ROWS } from "@/engine/terrain/constants";
import { BEDROCK, CONCRETE, DIRT, STONE } from "@/engine/terrain/materials";

/** Ground row of the test worlds (the old engine's BASE_ROW). */
export const GROUND = 148;

/** Flat ground from `ground` down: 4 rows of dirt, stone below, bedrock on the bottom row. */
export function flatWorld(opts: TerrainOptions & { ground?: number; cols?: number } = {}): Terrain {
  const g = opts.ground ?? GROUND;
  const t = new Terrain({
    seed: 7,
    ...opts,
    generate: (w, from, to) => {
      for (let c = from; c < to; c++)
        for (let r = g; r < ROWS; r++) w.set(c, r, r === ROWS - 1 ? BEDROCK : r < g + 4 ? DIRT : STONE);
    },
  });
  t.ensure(opts.cols ?? RING - 128);
  return t;
}

/** Fill a rectangle of cells. */
export function fill(
  t: Terrain,
  c0: number,
  r0: number,
  w: number,
  h: number,
  mat: number,
  opts?: Parameters<Terrain["set"]>[3]
): void {
  for (let c = c0; c < c0 + w; c++) for (let r = r0; r < r0 + h; r++) t.set(c, r, mat, opts);
}

/**
 * A building standing on `ground` (its bottom row is ground - 1): side walls `wall` wide, a slab every `fh` rows
 * and a roof, all structural. Returns its top row.
 */
export function building(
  t: Terrain,
  c0: number,
  w: number,
  floors: number,
  fh = 5,
  mat = CONCRETE,
  ground = GROUND,
  wall = 1
): number {
  const top = ground - floors * fh - 1;
  for (let r = top; r < ground; r++)
    for (let c = c0; c < c0 + w; c++) {
      const edge = c - c0 < wall || c >= c0 + w - wall;
      const slab = (ground - 1 - r) % fh === fh - 1 || r === top;
      if (edge || slab) t.set(c, r, mat, { structural: true });
    }
  return top;
}

/** Step `n` ticks. `each` runs after every tick. */
export function run(t: Terrain, n: number, each?: (t: Terrain) => void): void {
  for (let k = 0; k < n; k++) {
    t.step();
    each?.(t);
  }
}

/** Step until nothing is settling, or `max` ticks. Returns ticks taken. */
export function settle(t: Terrain, max = 3000, each?: (t: Terrain) => void): number {
  for (let k = 0; k < max; k++) {
    if (!t.busy) return k;
    t.step();
    each?.(t);
  }
  return max;
}

/** Cells of a material (or matching a predicate on material and flags) in the stored columns. */
export function count(t: Terrain, mat: number | ((m: number, flags: number) => boolean)): number {
  let n = 0;
  for (let c = t.startCol; c < t.endCol; c++) {
    const base = (c & MASK) * ROWS;
    for (let r = 0; r < ROWS; r++) {
      const m = t.fg[base + r];
      if (typeof mat === "number" ? m === mat : mat(m, t.flags[base + r])) n++;
    }
  }
  return n;
}

/** Structural cells within a rectangle. */
export function structuralIn(t: Terrain, c0: number, r0: number, w: number, h: number): number {
  let n = 0;
  for (let c = c0; c < c0 + w; c++)
    for (let r = r0; r < r0 + h; r++) if (t.get(c, r) && t.flags[t.index(c, r)] & F_STRUCTURAL) n++;
  return n;
}

/** Air everywhere but a bedrock floor (handy for exact physics checks). */
export function emptyWorld(opts: TerrainOptions = {}): Terrain {
  return flatWorld({ ...opts, ground: ROWS - 1 });
}
