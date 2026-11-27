import type { Atlas } from "../atlas";
import type { SpriteList } from "../buffers";
import { buildRuns, countingSort } from "../sort";
import { program, type Program } from "./util";
import { SPRITE_FS, SPRITE_VS } from "./shaders";

/** Bytes per sprite instance: pos 2×f32, uv rect 4×u16, pivot 2×i16, tint 4×u8, scale and rotation 2×f32, flags. */
export const SPRITE_STRIDE = 36;
const WORDS = SPRITE_STRIDE / 4;
/** Layers a sprite list can use. */
export const LAYERS = 16;

/** The atlas pages as GPU textures, uploaded lazily as frames are added. */
export class AtlasTextures {
  textures: WebGLTexture[] = [];
  private version = -1;

  constructor(
    private gl: WebGL2RenderingContext,
    private atlas: Atlas
  ) {}

  /** Uploads frames added since the last call; a no-op when nothing changed. */
  sync() {
    const { gl, atlas } = this;
    if (this.version === atlas.version) return;
    this.version = atlas.version;
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
    while (this.textures.length < atlas.pages) {
      const t = gl.createTexture();
      if (!t) return;
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, atlas.pageSize, atlas.pageSize);
      this.textures.push(t);
    }
    for (const i of atlas.pending) {
      const f = atlas.frames[i];
      const src = atlas.sources[i];
      if (!f || !src || f.w === 0 || f.h === 0 || src.width !== f.w || src.height !== f.h) continue;
      gl.bindTexture(gl.TEXTURE_2D, this.textures[f.page]);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, f.x, f.y, f.w, f.h, gl.RGBA, gl.UNSIGNED_BYTE, src);
    }
    atlas.pending.length = 0;
  }

  destroy() {
    for (const t of this.textures) this.gl.deleteTexture(t);
    this.textures.length = 0;
  }
}

/** The shared sprite program; every batch draws with it. */
export function spriteProgram(gl: WebGL2RenderingContext): Program {
  return program(gl, "sprite", SPRITE_VS, SPRITE_FS, ["uClip", "uTexel", "uTex"]);
}

/**
 * One sprite list's instances on the GPU. `prepare` sorts by (layer, page) and streams the instances in one
 * `bufferSubData`; `draw` issues one instanced call per run whose layer is in range. WebGL2 has no base instance,
 * so each run re-points the attributes at its slice of the buffer.
 */
export class SpriteBatch {
  readonly capacity: number;
  private vao: WebGLVertexArrayObject;
  private vbo: WebGLBuffer;
  private data: ArrayBuffer;
  private f32: Float32Array;
  private u32: Uint32Array;
  private u16: Uint16Array;
  private i16: Int16Array;
  private u8: Uint8Array;
  private keys: Uint16Array;
  private order: Uint32Array;
  private counts = new Uint32Array(LAYERS * 4 + 2);
  private runKey = new Uint16Array(LAYERS * 4 + 1);
  private runStart = new Uint32Array(LAYERS * 4 + 1);
  private runCount = new Uint32Array(LAYERS * 4 + 1);
  private runs = 0;
  private pages = 1;
  /** Instances written by the last `prepare` (after dropping any past capacity or with unknown frames). */
  count = 0;
  /** Sprites dropped by the last `prepare` because the list outgrew the capacity. */
  dropped = 0;

  constructor(
    private gl: WebGL2RenderingContext,
    capacity: number
  ) {
    this.capacity = capacity;
    this.data = new ArrayBuffer(capacity * SPRITE_STRIDE);
    this.f32 = new Float32Array(this.data);
    this.u32 = new Uint32Array(this.data);
    this.u16 = new Uint16Array(this.data);
    this.i16 = new Int16Array(this.data);
    this.u8 = new Uint8Array(this.data);
    this.keys = new Uint16Array(capacity);
    this.order = new Uint32Array(capacity);
    const vao = gl.createVertexArray();
    const vbo = gl.createBuffer();
    if (!vao || !vbo) throw new Error("render: could not create sprite buffers");
    this.vao = vao;
    this.vbo = vbo;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, this.data.byteLength, gl.STREAM_DRAW);
    for (let a = 0; a < 6; a++) {
      gl.enableVertexAttribArray(a);
      gl.vertexAttribDivisor(a, 1);
    }
    this.point(0);
    gl.bindVertexArray(null);
  }

  private point(base: number) {
    const gl = this.gl;
    const o = base * SPRITE_STRIDE;
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, SPRITE_STRIDE, o);
    gl.vertexAttribPointer(1, 4, gl.UNSIGNED_SHORT, false, SPRITE_STRIDE, o + 8);
    gl.vertexAttribPointer(2, 2, gl.SHORT, false, SPRITE_STRIDE, o + 16);
    gl.vertexAttribPointer(3, 4, gl.UNSIGNED_BYTE, true, SPRITE_STRIDE, o + 20);
    gl.vertexAttribPointer(4, 2, gl.FLOAT, false, SPRITE_STRIDE, o + 24);
    gl.vertexAttribIPointer(5, 1, gl.UNSIGNED_BYTE, SPRITE_STRIDE, o + 32);
  }

  /**
   * Sorts and uploads `list`, with positions made relative to (ox, oy) on the CPU in doubles, so far-flung world
   * coordinates keep full precision on the GPU. Given a view size (w×h from the origin), sprites that can't touch it
   * at any rotation are dropped.
   */
  prepare(list: SpriteList, atlas: Atlas, ox: number, oy: number, w = -1, h = -1) {
    const n = Math.min(list.count, this.capacity);
    this.dropped = list.count - n;
    const pages = Math.max(1, atlas.pages);
    const buckets = LAYERS * pages;
    if (this.counts.length < buckets + 2) {
      this.counts = new Uint32Array(buckets + 2);
      this.runKey = new Uint16Array(buckets + 1);
      this.runStart = new Uint32Array(buckets + 1);
      this.runCount = new Uint32Array(buckets + 1);
    }
    this.pages = pages;
    const { keys, order, f32, u32, u16, i16, u8 } = this;
    const frames = atlas.frames;
    const cull = w >= 0 && h >= 0;
    // Unknown or empty frames, and sprites out of view, go to a last bucket that is never drawn
    for (let i = 0; i < n; i++) {
      const f = frames[list.frame[i]];
      if (!f || f.w === 0) {
        keys[i] = buckets;
        continue;
      }
      const r = (f.w + f.h + 2) * Math.abs(list.scale[i]);
      const x = list.x[i] - ox;
      const y = list.y[i] - oy;
      const out = cull && (x + r < 0 || y + r < 0 || x - r > w || y - r > h);
      keys[i] = out ? buckets : list.layer[i] * pages + f.page;
    }
    countingSort(keys, n, buckets + 1, this.counts, order);
    const drawn = this.counts[buckets];
    for (let k = 0; k < drawn; k++) {
      const i = order[k];
      const f = frames[list.frame[i]];
      const w = k * WORDS;
      f32[w] = list.x[i] - ox;
      f32[w + 1] = list.y[i] - oy;
      u16[w * 2 + 4] = f.x;
      u16[w * 2 + 5] = f.y;
      u16[w * 2 + 6] = f.w;
      u16[w * 2 + 7] = f.h;
      i16[w * 2 + 8] = f.ox;
      i16[w * 2 + 9] = f.oy;
      u32[w + 5] = list.tint[i];
      f32[w + 6] = list.scale[i];
      f32[w + 7] = list.rot[i];
      u8[(w + 8) * 4] = list.flags[i];
    }
    this.count = drawn;
    this.runs = buildRuns(this.counts, buckets, this.runKey, this.runStart, this.runCount);
    if (drawn > 0) {
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.u8, 0, drawn * SPRITE_STRIDE);
    }
  }

  /**
   * Draws the prepared runs with layers in [from, to]. The caller has bound the sprite program, set `uClip`, the
   * blend mode and the target. Returns the draw calls issued.
   */
  draw(prog: Program, tex: AtlasTextures, from: number, to: number, pageSize: number): number {
    const gl = this.gl;
    let calls = 0;
    let bound = -1;
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.uniform2f(prog.u.uTexel, 1 / pageSize, 1 / pageSize);
    gl.activeTexture(gl.TEXTURE0);
    for (let r = 0; r < this.runs; r++) {
      const key = this.runKey[r];
      const layer = (key / this.pages) | 0;
      if (layer < from || layer > to) continue;
      const page = key - layer * this.pages;
      if (page !== bound) {
        gl.bindTexture(gl.TEXTURE_2D, tex.textures[page] ?? null);
        bound = page;
      }
      this.point(this.runStart[r]);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.runCount[r]);
      calls++;
    }
    gl.bindVertexArray(null);
    return calls;
  }

  destroy() {
    this.gl.deleteBuffer(this.vbo);
    this.gl.deleteVertexArray(this.vao);
  }
}
