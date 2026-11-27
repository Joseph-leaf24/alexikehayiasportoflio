import type { Atlas, AtlasSource } from "./atlas";
import { makeCanvas } from "./canvas";
import type { TerrainView } from "./types";

/**
 * The terrain's material tiles copied into one sheet, a column per variant and a row per material, so the terrain
 * pass finds a tile with one fetch: sheet (variant × tile + px, material × tile + py). Built from `tiles` (the
 * material → frame table) whenever that table changes, which is normally once, at boot.
 */
export class TileSheet {
  canvas: AtlasSource | null = null;
  materials = 0;
  private copy = new Int32Array(0);
  private atlasVersion = -1;
  /** Bumped when the sheet is rebuilt, so the GPU copy knows to re-upload. */
  version = 0;

  /** Rebuilds the sheet if the table or the atlas changed; returns true when it did. */
  sync(view: TerrainView, atlas: Atlas): boolean {
    const t = view.tiles;
    if (atlas.version === this.atlasVersion && sameTable(t, this.copy)) return false;
    this.atlasVersion = atlas.version;
    if (this.copy.length !== t.length) this.copy = new Int32Array(t.length);
    this.copy.set(t);
    const T = view.tile;
    const V = Math.max(1, view.variants);
    this.materials = Math.ceil(t.length / V);
    if (typeof document === "undefined" && typeof OffscreenCanvas === "undefined") return false;
    const [c, ctx] = makeCanvas(V * T, Math.max(1, this.materials) * T);
    for (let i = 0; i < t.length; i++) {
      const src = t[i] >= 0 ? atlas.sources[t[i]] : undefined;
      if (!src || src.width === 0 || src.height === 0) continue;
      const v = i % V;
      const m = (i / V) | 0;
      ctx.drawImage(src, 0, 0, Math.min(T, src.width), Math.min(T, src.height), v * T, m * T, T, T);
    }
    this.canvas = c;
    this.version++;
    return true;
  }
}

function sameTable(a: Int32Array, b: Int32Array) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** Splits a chunk index (column-major, see `TerrainView.drainDirty`) into its chunk column and row. */
export function chunkOf(index: number, view: TerrainView, out: Int32Array) {
  const rowsOfChunks = Math.ceil(view.rows / view.chunkSize);
  out[0] = (index / rowsOfChunks) | 0;
  out[1] = index - out[0] * rowsOfChunks;
  return out;
}

/** The first non-air row of a ring column, or `rows` if it's all air. */
export function columnTop(view: TerrainView, ringCol: number) {
  const base = ringCol * view.rows;
  const fg = view.fg;
  for (let r = 0; r < view.rows; r++) if (fg[base + r] !== 0) return r;
  return view.rows;
}

/**
 * Copies the cells of ring columns [x0, x1) and rows [y0, y1) from the terrain's column-major arrays into a
 * row-major RGBA staging array (fg, bg, variant, hp), ready for `texSubImage2D`.
 */
export function packCells(view: TerrainView, x0: number, x1: number, y0: number, y1: number, out: Uint8Array) {
  const w = x1 - x0;
  const { fg, bg, variant, hp, rows } = view;
  for (let x = x0; x < x1; x++) {
    const base = x * rows;
    for (let y = y0; y < y1; y++) {
      const i = base + y;
      const o = ((y - y0) * w + (x - x0)) * 4;
      out[o] = fg[i];
      out[o + 1] = bg[i];
      out[o + 2] = variant[i];
      out[o + 3] = hp[i];
    }
  }
  return out;
}
