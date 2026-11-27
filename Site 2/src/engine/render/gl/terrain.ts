import type { Atlas } from "../atlas";
import { columnTop, packCells, TileSheet, chunkOf } from "../tiles";
import type { TerrainView } from "../types";
import type { View } from "../view";
import { TERRAIN_FS, FULLSCREEN_VS } from "./shaders";
import { program, type Program } from "./util";

/**
 * The terrain on the GPU: the ring of cells as an RGBA8UI texture (fg, bg, variant, hp), each column's top row, the
 * tile sheet and a per-material table (emissive, max hp). Only dirty chunks are re-uploaded; drawing it is one
 * full-screen pass whatever it contains.
 */
export class TerrainPass {
  private prog: Program;
  private cells: WebGLTexture | null = null;
  private top: WebGLTexture | null = null;
  private tiles: WebGLTexture | null = null;
  private mat: WebGLTexture | null = null;
  private view: TerrainView | null = null;
  private ring = 0;
  private rows = 0;
  private sheet = new TileSheet();
  private sheetVersion = -1;
  private dirty: number[] = [];
  private colMask = new Uint32Array(0);
  private staging = new Uint8Array(0);
  private tops = new Uint16Array(0);
  private matData = new Uint8Array(256 * 4);
  private chunkXY = new Int32Array(2);
  private emissiveRef: Uint8Array | undefined;
  private maxHpRef: Uint8Array | undefined;
  /** Chunks uploaded by the last `sync`. */
  uploaded = 0;

  constructor(private gl: WebGL2RenderingContext) {
    this.prog = program(gl, "terrain", FULLSCREEN_VS, TERRAIN_FS, [
      "uCells",
      "uTop",
      "uTiles",
      "uMat",
      "uBase",
      "uFrac",
      "uInvZoom",
      "uH",
      "uTime",
      "uRows",
      "uRing",
      "uStart",
      "uT",
      "uVariants",
    ]);
  }

  /** Forgets the uploaded state, so the next `sync` re-uploads everything (after a resize of the ring or a new view). */
  reset() {
    this.view = null;
    this.sheetVersion = -1;
  }

  /** Brings the GPU copy up to date with the terrain: dirty chunks, column tops, tiles and the material table. */
  sync(view: TerrainView, atlas: Atlas) {
    const gl = this.gl;
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    this.uploaded = 0;
    if (view !== this.view || view.ring !== this.ring || view.rows !== this.rows) this.full(view);
    else this.partial(view);
    if (this.sheet.sync(view, atlas) || this.sheetVersion !== this.sheet.version) {
      this.sheetVersion = this.sheet.version;
      if (this.sheet.canvas) {
        if (this.tiles) gl.deleteTexture(this.tiles);
        this.tiles = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.tiles);
        setNearest(gl);
        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, this.sheet.canvas);
      }
    }
    this.material(view);
  }

  private full(view: TerrainView) {
    const gl = this.gl;
    this.view = view;
    this.ring = view.ring;
    this.rows = view.rows;
    // Anything queued is covered by the full upload
    view.drainDirty(this.dirty);
    this.dirty.length = 0;
    const n = view.ring * view.rows * 4;
    if (this.staging.length < n) this.staging = new Uint8Array(n);
    packCells(view, 0, view.ring, 0, view.rows, this.staging);
    if (this.cells) gl.deleteTexture(this.cells);
    this.cells = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.cells);
    setNearest(gl);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA8UI,
      view.ring,
      view.rows,
      0,
      gl.RGBA_INTEGER,
      gl.UNSIGNED_BYTE,
      this.staging
    );
    this.tops = new Uint16Array(view.ring);
    for (let c = 0; c < view.ring; c++) this.tops[c] = columnTop(view, c);
    if (this.top) gl.deleteTexture(this.top);
    this.top = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.top);
    setNearest(gl);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16UI, view.ring, 1, 0, gl.RED_INTEGER, gl.UNSIGNED_SHORT, this.tops);
    this.colMask = new Uint32Array(Math.ceil(view.ring / view.chunkSize));
    this.uploaded = this.colMask.length * Math.ceil(view.rows / view.chunkSize);
  }

  private partial(view: TerrainView) {
    const gl = this.gl;
    const n = view.drainDirty(this.dirty);
    if (n === 0) return;
    const cs = view.chunkSize;
    const mask = this.colMask;
    const xy = this.chunkXY;
    for (let i = 0; i < n; i++) {
      chunkOf(this.dirty[i], view, xy);
      const cc = xy[0];
      const cr = xy[1];
      if (cc >= 0 && cc < mask.length && cr < 32) mask[cc] |= 1 << cr;
    }
    this.dirty.length = 0;
    // One upload per chunk column, spanning its lowest to highest dirty chunk
    for (let cc = 0; cc < mask.length; cc++) {
      const m = mask[cc];
      if (!m) continue;
      mask[cc] = 0;
      const lo = 31 - Math.clz32(m & -m);
      const hi = 31 - Math.clz32(m);
      const x0 = cc * cs;
      const x1 = Math.min(view.ring, x0 + cs);
      const y0 = lo * cs;
      const y1 = Math.min(view.rows, (hi + 1) * cs);
      packCells(view, x0, x1, y0, y1, this.staging);
      gl.bindTexture(gl.TEXTURE_2D, this.cells);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, x0, y0, x1 - x0, y1 - y0, gl.RGBA_INTEGER, gl.UNSIGNED_BYTE, this.staging);
      for (let c = x0; c < x1; c++) this.tops[c] = columnTop(view, c);
      gl.bindTexture(gl.TEXTURE_2D, this.top);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, x0, 0, x1 - x0, 1, gl.RED_INTEGER, gl.UNSIGNED_SHORT, this.tops, x0);
      for (let b = m; b; b &= b - 1) this.uploaded++;
    }
  }

  /** Re-uploads the material table when the terrain hands over different tables (normally once). */
  private material(view: TerrainView) {
    if (view.emissive === this.emissiveRef && view.maxHp === this.maxHpRef && this.mat) return;
    const gl = this.gl;
    this.emissiveRef = view.emissive;
    this.maxHpRef = view.maxHp;
    const d = this.matData;
    for (let m = 0; m < 256; m++) {
      d[m * 4] = view.emissive?.[m] ?? 0;
      d[m * 4 + 1] = view.maxHp?.[m] ?? 255;
      d[m * 4 + 2] = 0;
      d[m * 4 + 3] = 255;
    }
    if (!this.mat) {
      this.mat = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.mat);
      setNearest(gl);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, d);
    } else {
      gl.bindTexture(gl.TEXTURE_2D, this.mat);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 256, 1, gl.RGBA, gl.UNSIGNED_BYTE, d);
    }
  }

  /** Draws the terrain into the bound scene target (premultiplied blending already set). Returns draw calls. */
  draw(view: View, time: number, empty: WebGLVertexArrayObject): number {
    const t = this.view;
    if (!t || !this.cells || !this.tiles || !this.top || !this.mat) return 0;
    const gl = this.gl;
    const { u } = this.prog;
    const T = t.tile;
    const bx = Math.floor(view.left / T);
    const by = Math.floor(view.top / T);
    gl.useProgram(this.prog.prog);
    bindUnit(gl, 0, this.cells, u.uCells);
    bindUnit(gl, 1, this.top, u.uTop);
    bindUnit(gl, 2, this.tiles, u.uTiles);
    bindUnit(gl, 3, this.mat, u.uMat);
    gl.uniform2i(u.uBase, bx, by);
    gl.uniform2f(u.uFrac, view.left - bx * T, view.top - by * T);
    gl.uniform1f(u.uInvZoom, 1 / view.zoom);
    gl.uniform1f(u.uH, view.h);
    gl.uniform1f(u.uTime, time % 3600);
    gl.uniform1i(u.uRows, t.rows);
    gl.uniform1i(u.uRing, t.ring);
    gl.uniform1i(u.uStart, t.startCol);
    gl.uniform1i(u.uT, T);
    gl.uniform1i(u.uVariants, Math.max(1, t.variants));
    gl.bindVertexArray(empty);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.activeTexture(gl.TEXTURE0);
    return 1;
  }

  destroy() {
    const gl = this.gl;
    for (const t of [this.cells, this.top, this.tiles, this.mat]) if (t) gl.deleteTexture(t);
    gl.deleteProgram(this.prog.prog);
    this.cells = this.top = this.tiles = this.mat = null;
  }
}

function bindUnit(gl: WebGL2RenderingContext, unit: number, tex: WebGLTexture, loc: WebGLUniformLocation | null) {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.uniform1i(loc, unit);
}

function setNearest(gl: WebGL2RenderingContext) {
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
}
