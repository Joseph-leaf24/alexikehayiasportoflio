/** Small WebGL2 helpers: programs, textures and framebuffers. */

export interface Program {
  prog: WebGLProgram;
  u: Record<string, WebGLUniformLocation | null>;
}

function compile(gl: WebGL2RenderingContext, type: number, src: string, name: string) {
  const s = gl.createShader(type);
  if (!s) throw new Error(`render: could not create shader ${name}`);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS) && !gl.isContextLost())
    throw new Error(`render: ${name} shader failed: ${gl.getShaderInfoLog(s)}`);
  return s;
}

/** Compiles and links a program, then looks up the named uniforms once. */
export function program(gl: WebGL2RenderingContext, name: string, vs: string, fs: string, uniforms: string[]): Program {
  const prog = gl.createProgram();
  if (!prog) throw new Error(`render: could not create program ${name}`);
  const v = compile(gl, gl.VERTEX_SHADER, vs, `${name}.vs`);
  const f = compile(gl, gl.FRAGMENT_SHADER, fs, `${name}.fs`);
  gl.attachShader(prog, v);
  gl.attachShader(prog, f);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS) && !gl.isContextLost())
    throw new Error(`render: ${name} failed to link: ${gl.getProgramInfoLog(prog)}`);
  gl.deleteShader(v);
  gl.deleteShader(f);
  const u: Record<string, WebGLUniformLocation | null> = {};
  for (const n of uniforms) u[n] = gl.getUniformLocation(prog, n);
  return { prog, u };
}

/** An empty texture with nearest filtering and clamped edges. */
export function texture(
  gl: WebGL2RenderingContext,
  w: number,
  h: number,
  internal: number,
  format: number,
  type: number,
  filter: number = gl.NEAREST
) {
  const t = gl.createTexture();
  if (!t) throw new Error("render: could not create texture");
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, null);
  return t;
}

/** A render target: its framebuffer, colour textures and size. */
export interface Target {
  fb: WebGLFramebuffer;
  tex: WebGLTexture[];
  w: number;
  h: number;
}

/** A framebuffer with `count` RGBA8 colour attachments. */
export function target(
  gl: WebGL2RenderingContext,
  w: number,
  h: number,
  count = 1,
  filter: number = gl.NEAREST
): Target {
  const fb = gl.createFramebuffer();
  if (!fb) throw new Error("render: could not create framebuffer");
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  const tex: WebGLTexture[] = [];
  const bufs: number[] = [];
  for (let i = 0; i < count; i++) {
    const t = texture(gl, w, h, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, filter);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0);
    tex.push(t);
    bufs.push(gl.COLOR_ATTACHMENT0 + i);
  }
  gl.drawBuffers(bufs);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE && !gl.isContextLost())
    throw new Error(`render: framebuffer incomplete (0x${status.toString(16)})`);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { fb, tex, w, h };
}

export function freeTarget(gl: WebGL2RenderingContext, t: Target | null) {
  if (!t) return;
  for (const x of t.tex) gl.deleteTexture(x);
  gl.deleteFramebuffer(t.fb);
}

/** Names for GL error codes, for development messages. */
const GL_ERRORS: Record<number, string> = {
  0x0500: "INVALID_ENUM",
  0x0501: "INVALID_VALUE",
  0x0502: "INVALID_OPERATION",
  0x0505: "OUT_OF_MEMORY",
  0x0506: "INVALID_FRAMEBUFFER_OPERATION",
  0x9242: "CONTEXT_LOST_WEBGL",
};

/** Reads and reports the GL error flag; returns true if there was one. Development only: it stalls the pipeline. */
export function checkGl(gl: WebGL2RenderingContext, where: string): boolean {
  const e = gl.getError();
  if (e === gl.NO_ERROR || e === gl.CONTEXT_LOST_WEBGL) return false;
  console.error(`render: GL error ${GL_ERRORS[e] ?? e} after ${where}`);
  return true;
}
