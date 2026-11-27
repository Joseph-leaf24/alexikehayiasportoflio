/**
 * Terrain layout, shared by every terrain file. The scale matches the old engine (`components/pixel/level.ts`), so
 * its generator and art port unchanged: 8 px cells, 240 rows, a ring of 512 columns that scrolls endlessly right.
 */

/** Edge of one cell, in world px. */
export const TILE = 8;
/** Rows of cells; row 0 is the top. */
export const ROWS = 240;
/** Columns kept in memory; a power of two so `col & MASK` picks the ring slot. */
export const RING = 512;
export const MASK = RING - 1;
/** Cells in the ring. Index = `(col & MASK) * ROWS + row`: column-major, so a whole column is contiguous. */
export const CELLS = RING * ROWS;
/** Height of the world in px. */
export const LEVEL_H = ROWS * TILE;

/** Chunks are 16×16 cells: the unit of dirty tracking (renderer uploads) and of physics sleeping. */
export const CHUNK = 16;
export const CHUNK_SHIFT = 4;
export const CHUNK_ROWS = ROWS / CHUNK;
export const CHUNK_COLS = RING / CHUNK;
export const CHUNKS = CHUNK_ROWS * CHUNK_COLS;

/** Tile variants per material, as in the old art (`art.ts` VARIANTS). */
export const VARIANTS = 4;

/** Ticks a chunk stays awake after its last change. */
export const WAKE = 2;

/** The fixed simulation step the scheduler runs (`TICK` in the core). */
export const TERRAIN_TICK = 1 / 30;

/* Per-cell flags (`Terrain.flags`). */
/** Falls: sand-like materials always; any cell while it crumbles down after being cut off. */
export const F_LOOSE = 1;
export const F_BURNING = 2;
/** Doused: the next spark only dries it. */
export const F_WET = 4;
/** Part of a building, bridge or tower: checked for support after nearby damage. */
export const F_STRUCTURAL = 8;
/** Takes half damage from blasts. */
export const F_REINFORCED = 16;
/** Cosmetic: blackened by a blast. */
export const F_SCORCHED = 32;
/** Moved this tick or last; internal, so falling cells don't count as support. */
export const F_MOVING = 64;
/** Debris from a shattered chunk: slides down slopes like a powder while loose, then rests as it was. */
export const F_TUMBLE = 128;

/** Chunk index of a cell, column-major like the cells (the `TerrainView.drainDirty` contract). */
export const chunkIndex = (col: number, row: number): number =>
  ((col & MASK) >> CHUNK_SHIFT) * CHUNK_ROWS + (row >> CHUNK_SHIFT);
/** First ring column of a chunk. */
export const chunkCol0 = (k: number): number => ((k / CHUNK_ROWS) | 0) << CHUNK_SHIFT;
/** First row of a chunk. */
export const chunkRow0 = (k: number): number => k % CHUNK_ROWS << CHUNK_SHIFT;

/* Terrain events (`Terrain.drainEvents`), as (type, col, row, value) quadruples. */
/** Lava met water: the lava turned to stone and the water to steam. value: 0. */
export const EV_STEAM = 1;
/** A cell caught fire. value: material. */
export const EV_IGNITE = 2;
/** Water put a fire out. value: material. */
export const EV_EXTINGUISH = 3;
/** A burning cell burnt out. value: the material it was. */
export const EV_BURNOUT = 4;
/** Acid ate a cell. value: the material it was. */
export const EV_DISSOLVE = 5;
/** A piece lost its support and started to fall. value: cells. */
export const EV_DETACH = 6;
/** A rigid falling chunk hit the ground. value: cells. */
export const EV_IMPACT = 7;
/** An overloaded or unbalanced structure gave way at its supports. value: cells crushed. */
export const EV_CRUSH = 8;
