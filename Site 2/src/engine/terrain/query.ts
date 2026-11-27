/**
 * Read-only questions about the terrain in world px, for bodies, shots and AI. Below the world is solid (bedrock);
 * above it and outside the stored columns is air, as in the old engine's `kindAt`.
 */
import type { Terrain } from "./grid";
import { LEVEL_H, MASK, ROWS, TILE } from "./constants";
import { BEDROCK, LIQUID, SOLID } from "./materials";

/** Where a ray stopped. Reused: create once with `createRayHit`. */
export interface RayHit {
  hit: boolean;
  /** Hit point (or the ray's end), world px, and distance travelled. */
  x: number;
  y: number;
  dist: number;
  /** Face normal of the hit cell (an axis unit vector); 0, 0 if the ray started inside it. */
  nx: number;
  ny: number;
  col: number;
  row: number;
  /** Material hit (BEDROCK below the world). */
  mat: number;
}

export const createRayHit = (): RayHit => ({ hit: false, x: 0, y: 0, dist: 0, nx: 0, ny: 0, col: 0, row: 0, mat: 0 });

/** Material of a cell as queries see it. */
function cellMat(t: Terrain, col: number, row: number): number {
  if (row >= ROWS) return BEDROCK;
  if (row < 0 || col < t.startCol || col >= t.endCol) return 0;
  return t.fg[(col & MASK) * ROWS + row];
}

export function solidAt(t: Terrain, px: number, py: number): boolean {
  return SOLID[cellMat(t, Math.floor(px / TILE), Math.floor(py / TILE))] === 1;
}

/** The liquid at a point (WATER, LAVA, ACID), or 0. */
export function liquidAt(t: Terrain, px: number, py: number): number {
  const m = cellMat(t, Math.floor(px / TILE), Math.floor(py / TILE));
  return LIQUID[m] ? m : 0;
}

export function materialAt(t: Terrain, px: number, py: number): number {
  return cellMat(t, Math.floor(px / TILE), Math.floor(py / TILE));
}

/** World-px y of the top of the first solid cell at or below `fromY` in the column under `px`; LEVEL_H if none. */
export function surfaceY(t: Terrain, px: number, fromY = 0): number {
  const col = Math.floor(px / TILE);
  if (col < t.startCol || col >= t.endCol) return LEVEL_H;
  const base = (col & MASK) * ROWS;
  for (let r = Math.max(0, Math.floor(fromY / TILE)); r < ROWS; r++) if (SOLID[t.fg[base + r]]) return r * TILE;
  return LEVEL_H;
}

/**
 * Whether the segment between two points is clear of solid cells, walking every cell it crosses (a DDA). The cells
 * holding the two end points don't count, as in the old engine, so a shooter pressed against a wall can still fire.
 */
export function lineOfSight(t: Terrain, x0: number, y0: number, x1: number, y1: number): boolean {
  let col = Math.floor(x0 / TILE);
  let row = Math.floor(y0 / TILE);
  const dx = x1 - x0;
  const dy = y1 - y0;
  const sx = dx > 0 ? 1 : dx < 0 ? -1 : 0;
  const sy = dy > 0 ? 1 : dy < 0 ? -1 : 0;
  // Progress along the segment, 0..1, at the next column and row boundary
  const stepX = sx !== 0 ? TILE / Math.abs(dx) : Infinity;
  const stepY = sy !== 0 ? TILE / Math.abs(dy) : Infinity;
  let nextX = sx > 0 ? ((col + 1) * TILE - x0) / dx : sx < 0 ? (col * TILE - x0) / dx : Infinity;
  let nextY = sy > 0 ? ((row + 1) * TILE - y0) / dy : sy < 0 ? (row * TILE - y0) / dy : Infinity;
  let n = Math.abs(Math.floor(x1 / TILE) - col) + Math.abs(Math.floor(y1 / TILE) - row);
  while (n-- > 1) {
    if (nextX < nextY) {
      col += sx;
      nextX += stepX;
    } else {
      row += sy;
      nextY += stepY;
    }
    if (SOLID[cellMat(t, col, row)]) return false;
  }
  return true;
}

/**
 * Cast a ray from (x0, y0) along (dx, dy) for up to `maxDist` px, cell by cell (a DDA). Fills `out` with the first
 * solid cell hit (point, face normal, material) and returns whether it hit.
 */
export function raycast(
  t: Terrain,
  x0: number,
  y0: number,
  dx: number,
  dy: number,
  maxDist: number,
  out: RayHit
): boolean {
  out.hit = false;
  out.nx = 0;
  out.ny = 0;
  out.mat = 0;
  const len = Math.hypot(dx, dy);
  if (!(len > 0) || !(maxDist >= 0)) {
    out.x = x0;
    out.y = y0;
    out.dist = 0;
    return false;
  }
  const ux = dx / len;
  const uy = dy / len;
  let col = Math.floor(x0 / TILE);
  let row = Math.floor(y0 / TILE);
  const sx = ux > 0 ? 1 : ux < 0 ? -1 : 0;
  const sy = uy > 0 ? 1 : uy < 0 ? -1 : 0;
  const stepX = sx !== 0 ? TILE / Math.abs(ux) : Infinity;
  const stepY = sy !== 0 ? TILE / Math.abs(uy) : Infinity;
  let nextX = sx > 0 ? ((col + 1) * TILE - x0) / ux : sx < 0 ? (col * TILE - x0) / ux : Infinity;
  let nextY = sy > 0 ? ((row + 1) * TILE - y0) / uy : sy < 0 ? (row * TILE - y0) / uy : Infinity;
  let dist = 0;
  let nx = 0;
  let ny = 0;
  const limit = Math.min(4096, Math.ceil(maxDist / TILE) * 2 + 2);
  for (let n = 0; n <= limit; n++) {
    const m = cellMat(t, col, row);
    if (SOLID[m]) {
      out.hit = true;
      out.dist = dist;
      out.x = x0 + ux * dist;
      out.y = y0 + uy * dist;
      out.nx = nx;
      out.ny = ny;
      out.col = col;
      out.row = row;
      out.mat = m;
      return true;
    }
    // Nothing left to hit: in the sky heading up, or off the stored land heading away from it
    if (row < 0 && sy <= 0) break;
    if (sy <= 0 && ((col < t.startCol && sx <= 0) || (col >= t.endCol && sx >= 0))) break;
    if (nextX < nextY) {
      dist = nextX;
      nextX += stepX;
      col += sx;
      nx = -sx;
      ny = 0;
    } else {
      dist = nextY;
      nextY += stepY;
      row += sy;
      nx = 0;
      ny = -sy;
    }
    if (dist > maxDist) break;
  }
  out.dist = maxDist;
  out.x = x0 + ux * maxDist;
  out.y = y0 + uy * maxDist;
  return false;
}

/** Whether a body's box overlaps a solid cell: centred on `x`, feet at `y`, half width `hw`, height `h` (px). */
export function overlapsSolid(t: Terrain, x: number, y: number, hw: number, h: number): boolean {
  const c0 = Math.floor((x - hw) / TILE);
  const c1 = Math.floor((x + hw - 1e-3) / TILE);
  const r0 = Math.floor((y - h) / TILE);
  const r1 = Math.floor((y - 1e-3) / TILE);
  for (let c = c0; c <= c1; c++) for (let r = r0; r <= r1; r++) if (SOLID[cellMat(t, c, r)]) return true;
  return false;
}
