import type { LightList, ParticleList } from "../buffers";
import { LIGHT_FS, LIGHT_VS, PARTICLE_FS, PARTICLE_VS } from "./shaders";
import { program, type Program } from "./util";

/** Bytes per particle instance: pos 2×f32, colour 4×u8, size and flags 2×u8, 2 bytes padding. */
const PARTICLE_STRIDE = 16;
/** Bytes per light instance: pos 2×f32, radius f32, colour 4×u8, intensity f32. */
const LIGHT_STRIDE = 20;

function instanced(gl: WebGL2RenderingContext, bytes: number) {
  const vao = gl.createVertexArray();
  const vbo = gl.createBuffer();
  if (!vao || !vbo) throw new Error("render: could not create instance buffers");
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, bytes, gl.STREAM_DRAW);
  return { vao, vbo };
}

/** World particles as instanced 1–3 px quads, normal and additive mixed in one draw (premultiplied alpha). */
export class ParticleBatch {
  readonly prog: Program;
  private vao: WebGLVertexArrayObject;
  private vbo: WebGLBuffer;
  private f32: Float32Array;
  private u32: Uint32Array;
  private u8: Uint8Array;
  count = 0;

  constructor(
    private gl: WebGL2RenderingContext,
    readonly capacity: number
  ) {
    this.prog = program(gl, "particle", PARTICLE_VS, PARTICLE_FS, ["uClip"]);
    const data = new ArrayBuffer(capacity * PARTICLE_STRIDE);
    this.f32 = new Float32Array(data);
    this.u32 = new Uint32Array(data);
    this.u8 = new Uint8Array(data);
    const { vao, vbo } = instanced(gl, data.byteLength);
    this.vao = vao;
    this.vbo = vbo;
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, PARTICLE_STRIDE, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.UNSIGNED_BYTE, true, PARTICLE_STRIDE, 8);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribIPointer(2, 2, gl.UNSIGNED_BYTE, PARTICLE_STRIDE, 12);
    for (let a = 0; a < 3; a++) gl.vertexAttribDivisor(a, 1);
    gl.bindVertexArray(null);
  }

  /** Streams the particles in view (w×h world px), relative to the view origin (ox, oy). */
  prepare(list: ParticleList, ox: number, oy: number, w: number, h: number) {
    const { f32, u32, u8 } = this;
    let n = 0;
    for (let i = 0; i < list.count && n < this.capacity; i++) {
      const x = list.x[i] - ox;
      const y = list.y[i] - oy;
      if (x < -4 || y < -4 || x > w || y > h || (list.color[i] & 255) === 0) continue;
      const o = n * 4;
      f32[o] = x;
      f32[o + 1] = y;
      u32[o + 2] = list.color[i];
      u8[o * 4 + 12] = list.size[i];
      u8[o * 4 + 13] = list.flags[i];
      n++;
    }
    this.count = n;
    if (n > 0) {
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, u8, 0, n * PARTICLE_STRIDE);
    }
  }

  /** Draws the prepared particles; the caller set the target and blending. Returns draw calls. */
  draw(clipX: number, clipY: number): number {
    if (this.count === 0) return 0;
    const gl = this.gl;
    gl.useProgram(this.prog.prog);
    gl.uniform2f(this.prog.u.uClip, clipX, clipY);
    gl.bindVertexArray(this.vao);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.count);
    gl.bindVertexArray(null);
    return 1;
  }

  destroy() {
    this.gl.deleteBuffer(this.vbo);
    this.gl.deleteVertexArray(this.vao);
    this.gl.deleteProgram(this.prog.prog);
  }
}

/** Point lights as instanced quads drawn additively into the light buffer, falloff in the shader. */
export class LightBatch {
  readonly prog: Program;
  private vao: WebGLVertexArrayObject;
  private vbo: WebGLBuffer;
  private f32: Float32Array;
  private u32: Uint32Array;
  private u8: Uint8Array;
  count = 0;

  constructor(
    private gl: WebGL2RenderingContext,
    readonly capacity: number
  ) {
    this.prog = program(gl, "light", LIGHT_VS, LIGHT_FS, ["uClip"]);
    const data = new ArrayBuffer(capacity * LIGHT_STRIDE);
    this.f32 = new Float32Array(data);
    this.u32 = new Uint32Array(data);
    this.u8 = new Uint8Array(data);
    const { vao, vbo } = instanced(gl, data.byteLength);
    this.vao = vao;
    this.vbo = vbo;
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, LIGHT_STRIDE, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 1, gl.FLOAT, false, LIGHT_STRIDE, 8);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 4, gl.UNSIGNED_BYTE, true, LIGHT_STRIDE, 12);
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 1, gl.FLOAT, false, LIGHT_STRIDE, 16);
    for (let a = 0; a < 4; a++) gl.vertexAttribDivisor(a, 1);
    gl.bindVertexArray(null);
  }

  /** Streams the lights, relative to the view origin, skipping any that can't touch the view. */
  prepare(list: LightList, ox: number, oy: number, viewW: number, viewH: number) {
    const { f32, u32 } = this;
    let n = 0;
    for (let i = 0; i < list.count && n < this.capacity; i++) {
      const r = list.radius[i];
      const x = list.x[i] - ox;
      const y = list.y[i] - oy;
      if (r <= 0 || list.intensity[i] <= 0 || x + r < 0 || y + r < 0 || x - r > viewW || y - r > viewH) continue;
      const w = n * 5;
      f32[w] = x;
      f32[w + 1] = y;
      f32[w + 2] = r;
      u32[w + 3] = list.color[i];
      f32[w + 4] = list.intensity[i];
      n++;
    }
    this.count = n;
    if (n > 0) {
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.u8, 0, n * LIGHT_STRIDE);
    }
  }

  draw(clipX: number, clipY: number): number {
    if (this.count === 0) return 0;
    const gl = this.gl;
    gl.useProgram(this.prog.prog);
    gl.uniform2f(this.prog.u.uClip, clipX, clipY);
    gl.bindVertexArray(this.vao);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.count);
    gl.bindVertexArray(null);
    return 1;
  }

  destroy() {
    this.gl.deleteBuffer(this.vbo);
    this.gl.deleteVertexArray(this.vao);
    this.gl.deleteProgram(this.prog.prog);
  }
}
