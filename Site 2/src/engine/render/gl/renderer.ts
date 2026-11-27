import { Atlas } from "../atlas";
import { SpriteList, TextList, ParticleList, LightList } from "../buffers";
import { layoutText, registerFont, type FontFrames } from "../font";
import { PARTICLE_LAYER, TERRAIN_LAYER } from "../frame";
import type { RenderFrame, Renderer, RendererOptions, RenderStats } from "../types";
import { clipScale, computeView, internalSize, makeView, parallaxOrigin, rgbInto } from "../view";
import { LightBatch, ParticleBatch } from "./batches";
import { BLUR_FS, BRIGHT_FS, FULLSCREEN_VS, MAX_WAVES, POST_FS, SKY_FS } from "./shaders";
import { AtlasTextures, SpriteBatch, spriteProgram } from "./sprites";
import { TerrainPass } from "./terrain";
import { checkGl, freeTarget, program, target, type Program, type Target } from "./util";

const DEV = import.meta.env?.DEV === true;
/** In development, GL errors are checked after every pass on one frame in this many (getError stalls). */
const CHECK_EVERY = 60;
/** Light buffer encoding: stored = light × 128/255, so daylight (1.0) is exact and lights reach ~2× daylight. */
const LIGHT_ENCODE = 128 / 255;

/** Every GPU resource, rebuilt as a whole after a lost context. */
interface Gpu {
  sprite: Program;
  sky: Program;
  bright: Program;
  blur: Program;
  post: Program;
  empty: WebGLVertexArrayObject;
  atlas: AtlasTextures;
  bg: SpriteBatch;
  world: SpriteBatch;
  overlay: SpriteBatch;
  particles: ParticleBatch;
  lights: LightBatch;
  terrain: TerrainPass;
  scene: Target | null;
  light: Target | null;
  bloomA: Target | null;
  bloomB: Target | null;
  out: Target | null;
}

/**
 * Pixel-perfect WebGL2 renderer. The world draws into a low-resolution two-attachment scene target (colour and its
 * emissive part); lights go into a smaller light buffer; one post pass lights the scene and applies the effects; the
 * overlay (HUD sprites and text) draws on top at the same resolution; a nearest-filtered blit scales it to the canvas.
 */
export class WebGLRenderer implements Renderer {
  readonly kind = "webgl2" as const;
  readonly atlas = new Atlas(2048);
  readonly stats: RenderStats = { drawCalls: 0, sprites: 0, particles: 0, lights: 0, cpuMs: 0, chunks: 0 };
  width = 1;
  height = 1;
  private gpu: Gpu | null = null;
  private lost = false;
  private font: FontFrames;
  private view = makeView();
  private clip = new Float32Array(2);
  private tmp2 = new Float32Array(2);
  private rgb = new Float32Array(3);
  private waves = new Float32Array(MAX_WAVES * 4);
  private waveS = new Float32Array(MAX_WAVES);
  private bgList = new SpriteList(256);
  private overlayList = new SpriteList(1024);
  private frameNo = 0;
  private quality: "high" | "mid" | "low";
  private maxSprites: number;
  private maxParticles: number;
  private pixelScale: number;
  private timer: GpuTimer | null = null;

  constructor(
    private canvas: HTMLCanvasElement,
    private gl: WebGL2RenderingContext,
    opts: RendererOptions
  ) {
    this.quality = opts.quality ?? "high";
    this.maxSprites = opts.maxSprites ?? 16384;
    this.maxParticles = opts.maxParticles ?? 16384;
    this.pixelScale = opts.pixelScale > 0 ? opts.pixelScale : 3;
    this.font = registerFont(this.atlas);
    canvas.addEventListener("webglcontextlost", this.onLost);
    canvas.addEventListener("webglcontextrestored", this.onRestored);
    this.gpu = this.build();
    if (opts.gpuTiming) this.timer = GpuTimer.create(gl);
    this.stats.gpuMs = -1;
    const s = internalSize(canvas.width || 1, canvas.height || 1, 1, this.pixelScale);
    this.setSize(s.w, s.h);
  }

  private onLost = (e: Event) => {
    // Without preventDefault the browser never offers the context back
    e.preventDefault();
    this.lost = true;
    this.gpu = null;
  };

  private onRestored = () => {
    this.lost = false;
    this.atlas.invalidate();
    this.gpu = this.build();
    if (this.timer) this.timer = GpuTimer.create(this.gl);
    this.makeTargets();
  };

  private build(): Gpu {
    const gl = this.gl;
    const empty = gl.createVertexArray();
    if (!empty) throw new Error("render: could not create a vertex array");
    return {
      sprite: spriteProgram(gl),
      sky: program(gl, "sky", FULLSCREEN_VS, SKY_FS, ["uTop", "uMid", "uBot", "uStars", "uTime", "uH", "uStarOff"]),
      bright: program(gl, "bright", FULLSCREEN_VS, BRIGHT_FS, ["uScene", "uEmit", "uLight", "uInvSize", "uThreshold"]),
      blur: program(gl, "blur", FULLSCREEN_VS, BLUR_FS, ["uSrc", "uInvSize", "uDir"]),
      post: program(gl, "post", FULLSCREEN_VS, POST_FS, [
        "uScene",
        "uEmit",
        "uLight",
        "uBloom",
        "uSize",
        "uWaves",
        "uWave",
        "uWaveS",
        "uChroma",
        "uBloomK",
        "uVignette",
        "uFlash",
        "uWhite",
        "uLetterbox",
        "uLightSteps",
        "uGrade",
        "uGradeA",
        "uTint",
      ]),
      empty,
      atlas: new AtlasTextures(gl, this.atlas),
      bg: new SpriteBatch(gl, 1024),
      world: new SpriteBatch(gl, this.maxSprites),
      overlay: new SpriteBatch(gl, 8192),
      particles: new ParticleBatch(gl, this.maxParticles),
      lights: new LightBatch(gl, 1024),
      terrain: new TerrainPass(gl),
      scene: null,
      light: null,
      bloomA: null,
      bloomB: null,
      out: null,
    };
  }

  /** Sizes the canvas to whole pixel-scale multiples (its CSS size set to match) and the targets to the view. */
  resize(cssWidth: number, cssHeight: number, dpr: number, pixelScale: number) {
    this.pixelScale = pixelScale > 0 ? pixelScale : this.pixelScale;
    const s = internalSize(cssWidth, cssHeight, dpr, this.pixelScale);
    const cw = Math.round(s.w * this.pixelScale);
    const ch = Math.round(s.h * this.pixelScale);
    if (this.canvas.width !== cw) this.canvas.width = cw;
    if (this.canvas.height !== ch) this.canvas.height = ch;
    if (this.canvas.style) {
      this.canvas.style.width = `${cw / dpr}px`;
      this.canvas.style.height = `${ch / dpr}px`;
    }
    this.setSize(s.w, s.h);
  }

  private setSize(w: number, h: number) {
    if (w === this.width && h === this.height && this.gpu?.scene) return;
    this.width = w;
    this.height = h;
    this.makeTargets();
  }

  private makeTargets() {
    const gpu = this.gpu;
    const gl = this.gl;
    if (!gpu || this.lost) return;
    for (const t of [gpu.scene, gpu.light, gpu.bloomA, gpu.bloomB, gpu.out]) freeTarget(gl, t);
    const { width: w, height: h } = this;
    const lightDiv = this.quality === "low" ? 4 : 2;
    const bloomDiv = this.quality === "high" ? 2 : 4;
    gpu.scene = target(gl, w, h, 2, gl.LINEAR);
    gpu.light = target(gl, Math.ceil(w / lightDiv), Math.ceil(h / lightDiv), 1, gl.LINEAR);
    gpu.out = target(gl, w, h, 1, gl.NEAREST);
    if (this.quality !== "low") {
      gpu.bloomA = target(gl, Math.ceil(w / bloomDiv), Math.ceil(h / bloomDiv), 1, gl.LINEAR);
      gpu.bloomB = target(gl, Math.ceil(w / bloomDiv), Math.ceil(h / bloomDiv), 1, gl.LINEAR);
    } else gpu.bloomA = gpu.bloomB = null;
  }

  render(frame: RenderFrame) {
    const t0 = performance.now();
    const st = this.stats;
    st.drawCalls = st.sprites = st.particles = st.lights = st.chunks = 0;
    const gpu = this.gpu;
    const gl = this.gl;
    if (!gpu || this.lost || !gpu.scene || !gpu.light || !gpu.out) {
      st.cpuMs = performance.now() - t0;
      return;
    }
    const check = DEV && this.frameNo++ % CHECK_EVERY === 0;
    this.timer?.begin();
    const W = this.width;
    const H = this.height;
    const view = computeView(frame.camera, W, H, this.view);
    const clip = clipScale(view, this.clip);
    const ww = W / view.zoom;
    const wh = H / view.zoom;
    const terrainLayer = frame.terrainLayer ?? TERRAIN_LAYER;
    const particleLayer = frame.particleLayer ?? PARTICLE_LAYER;

    // Upload and stream everything first, then draw
    gpu.atlas.sync();
    this.buildParallax(frame);
    gpu.bg.prepare(this.bgList, this.atlas, 0, 0, ww, wh);
    const sprites = frame.sprites instanceof SpriteList ? frame.sprites : null;
    if (sprites) gpu.world.prepare(sprites, this.atlas, view.left, view.top, ww, wh);
    const particles = frame.particles instanceof ParticleList ? frame.particles : null;
    if (particles) gpu.particles.prepare(particles, view.left, view.top, ww, wh);
    const lights = frame.lights instanceof LightList ? frame.lights : null;
    if (lights) gpu.lights.prepare(lights, view.left, view.top, ww, wh);
    this.buildOverlay(frame);
    gpu.overlay.prepare(this.overlayList, this.atlas, 0, 0, W, H);
    if (frame.terrain) {
      gpu.terrain.sync(frame.terrain, this.atlas);
      st.chunks = gpu.terrain.uploaded;
    }
    if (check) checkGl(gl, "uploads");

    // Scene: sky, parallax, back sprites, terrain, sprites, particles, front sprites
    gl.bindFramebuffer(gl.FRAMEBUFFER, gpu.scene.fb);
    gl.viewport(0, 0, W, H);
    gl.disable(gl.BLEND);
    gl.useProgram(gpu.sky.prog);
    const su = gpu.sky.u;
    gl.uniform3fv(su.uTop, rgbInto(frame.sky.top, this.rgb));
    gl.uniform3fv(su.uMid, rgbInto(frame.sky.mid, this.rgb));
    gl.uniform3fv(su.uBot, rgbInto(frame.sky.bottom, this.rgb));
    gl.uniform1f(su.uStars, frame.sky.stars);
    gl.uniform1f(su.uTime, frame.time % 3600);
    gl.uniform1f(su.uH, H);
    gl.uniform2i(su.uStarOff, Math.floor(frame.camera.x * 0.02) % 4096, Math.floor(frame.camera.y * 0.02) % 4096);
    gl.bindVertexArray(gpu.empty);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    let calls = 1;

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    const ps = this.atlas.pageSize;
    this.useSprites(gpu, clip[0], clip[1]);
    calls += gpu.bg.draw(gpu.sprite, gpu.atlas, 0, 15, ps);
    calls += gpu.world.draw(gpu.sprite, gpu.atlas, 0, terrainLayer - 1, ps);
    if (frame.terrain) {
      calls += gpu.terrain.draw(view, frame.time, gpu.empty);
      this.useSprites(gpu, clip[0], clip[1]);
    }
    calls += gpu.world.draw(gpu.sprite, gpu.atlas, terrainLayer, particleLayer, ps);
    calls += gpu.particles.draw(clip[0], clip[1]);
    this.useSprites(gpu, clip[0], clip[1]);
    calls += gpu.world.draw(gpu.sprite, gpu.atlas, particleLayer + 1, 15, ps);
    if (check) checkGl(gl, "scene");

    // Lights: ambient, plus each light added on top
    gl.bindFramebuffer(gl.FRAMEBUFFER, gpu.light.fb);
    gl.viewport(0, 0, gpu.light.w, gpu.light.h);
    rgbInto(frame.ambient, this.rgb);
    gl.clearColor(this.rgb[0] * LIGHT_ENCODE, this.rgb[1] * LIGHT_ENCODE, this.rgb[2] * LIGHT_ENCODE, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.blendFunc(gl.ONE, gl.ONE);
    calls += gpu.lights.draw(clip[0], clip[1]);
    gl.disable(gl.BLEND);
    if (check) checkGl(gl, "lights");

    const fx = frame.effects;
    const bloom = fx.bloom > 0 && gpu.bloomA && gpu.bloomB ? fx.bloom : 0;
    if (bloom > 0 && gpu.bloomA && gpu.bloomB)
      calls += this.bloomPass(gpu, gpu.scene, gpu.light, gpu.bloomA, gpu.bloomB);

    // Post: lighting composite and effects into the output target
    gl.bindFramebuffer(gl.FRAMEBUFFER, gpu.out.fb);
    gl.viewport(0, 0, W, H);
    calls += this.postPass(gpu, frame, bloom);
    if (check) checkGl(gl, "post");

    // Overlay at the same resolution, unaffected by camera and post
    if (gpu.overlay.count > 0) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      this.useSprites(gpu, 2 / W, -2 / H);
      calls += gpu.overlay.draw(gpu.sprite, gpu.atlas, 0, 15, ps);
      gl.disable(gl.BLEND);
    }

    // Present: nearest-neighbour upscale to the canvas
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, gpu.out.fb);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
    gl.blitFramebuffer(0, 0, W, H, 0, 0, this.canvas.width, this.canvas.height, gl.COLOR_BUFFER_BIT, gl.NEAREST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (check) checkGl(gl, "present");
    if (this.timer) {
      this.timer.end();
      st.gpuMs = this.timer.ms;
    }

    st.drawCalls = calls;
    st.sprites = gpu.world.count;
    st.particles = gpu.particles.count;
    st.lights = gpu.lights.count;
    st.cpuMs = performance.now() - t0;
  }

  private useSprites(gpu: Gpu, cx: number, cy: number) {
    const gl = this.gl;
    gl.useProgram(gpu.sprite.prog);
    gl.uniform2f(gpu.sprite.u.uClip, cx, cy);
    gl.uniform1i(gpu.sprite.u.uTex, 0);
  }

  /** Parallax layers as sprite instances relative to the view, one layer index each so they keep their order. */
  private buildParallax(frame: RenderFrame) {
    const list = this.bgList;
    list.clear();
    const view = this.view;
    const ww = view.w / view.zoom;
    const cam = frame.camera;
    for (let li = 0; li < frame.parallax.length && li < 16; li++) {
      const p = frame.parallax[li];
      const f = this.atlas.frame(p.frame);
      if (f.w === 0) continue;
      const o = parallaxOrigin(view, cam.x + cam.shakeX, cam.y + cam.shakeY, p.factor, p.y, this.tmp2);
      if (!p.repeatX) {
        list.push(p.frame, o[0], o[1], li, 0, p.tint, 1, 0);
        continue;
      }
      // Tile across the view: copies at whole frame widths from the pivot
      const left = o[0] - f.ox;
      const k0 = Math.floor(-left / f.w);
      const k1 = Math.ceil((ww - left) / f.w);
      for (let k = k0; k < k1; k++) list.push(p.frame, o[0] + k * f.w, o[1], li, 0, p.tint, 1, 0);
    }
  }

  /** Overlay sprites plus every text run laid out as glyph sprites. */
  private buildOverlay(frame: RenderFrame) {
    const list = this.overlayList;
    list.clear();
    const src = frame.overlay;
    if (src instanceof SpriteList)
      for (let i = 0; i < src.count; i++)
        list.push(src.frame[i], src.x[i], src.y[i], src.layer[i], src.flags[i], src.tint[i], src.scale[i], src.rot[i]);
    const text = frame.text;
    if (text instanceof TextList)
      for (let i = 0; i < text.count; i++)
        layoutText(
          this.font,
          text.text[i],
          text.x[i],
          text.y[i],
          text.color[i],
          text.scale[i],
          text.align[i],
          text.layer[i],
          list
        );
  }

  private bloomPass(gpu: Gpu, scene: Target, light: Target, a: Target, b: Target): number {
    const gl = this.gl;
    gl.bindVertexArray(gpu.empty);
    gl.bindFramebuffer(gl.FRAMEBUFFER, a.fb);
    gl.viewport(0, 0, a.w, a.h);
    gl.useProgram(gpu.bright.prog);
    const bu = gpu.bright.u;
    bindTex(gl, 0, scene.tex[0], bu.uScene);
    bindTex(gl, 1, scene.tex[1], bu.uEmit);
    bindTex(gl, 2, light.tex[0], bu.uLight);
    gl.uniform2f(bu.uInvSize, 1 / a.w, 1 / a.h);
    // Only light pushed past daylight blooms, so a bright day scene doesn't haze over
    gl.uniform1f(bu.uThreshold, 1.0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.useProgram(gpu.blur.prog);
    const lu = gpu.blur.u;
    gl.uniform2f(lu.uInvSize, 1 / a.w, 1 / a.h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, b.fb);
    bindTex(gl, 0, a.tex[0], lu.uSrc);
    gl.uniform2f(lu.uDir, 1 / a.w, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindFramebuffer(gl.FRAMEBUFFER, a.fb);
    bindTex(gl, 0, b.tex[0], lu.uSrc);
    gl.uniform2f(lu.uDir, 0, 1 / a.h);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    return 3;
  }

  private postPass(gpu: Gpu, frame: RenderFrame, bloom: number): number {
    const gl = this.gl;
    const fx = frame.effects;
    const view = this.view;
    const { u } = gpu.post;
    const scene = gpu.scene as Target;
    gl.useProgram(gpu.post.prog);
    bindTex(gl, 0, scene.tex[0], u.uScene);
    bindTex(gl, 1, scene.tex[1], u.uEmit);
    bindTex(gl, 2, (gpu.light as Target).tex[0], u.uLight);
    bindTex(gl, 3, (gpu.bloomA ?? scene).tex[0], u.uBloom);
    gl.uniform2f(u.uSize, view.w, view.h);
    // Shockwaves in output pixels, y up as gl_FragCoord has it
    let n = 0;
    const z = view.zoom;
    for (let i = 0; i < fx.shockwaves.length && n < MAX_WAVES; i++) {
      const s = fx.shockwaves[i];
      if (s.strength === 0 || s.radius + s.width < 0) continue;
      this.waves[n * 4] = (s.x - view.left) * z;
      this.waves[n * 4 + 1] = view.h - (s.y - view.top) * z;
      this.waves[n * 4 + 2] = s.radius * z;
      this.waves[n * 4 + 3] = s.width * z;
      this.waveS[n] = s.strength * z;
      n++;
    }
    gl.uniform1i(u.uWaves, n);
    if (n > 0) {
      gl.uniform4fv(u.uWave, this.waves, 0, n * 4);
      gl.uniform1fv(u.uWaveS, this.waveS, 0, n);
    }
    gl.uniform1f(u.uChroma, fx.chroma);
    gl.uniform1f(u.uBloomK, bloom);
    gl.uniform1f(u.uVignette, fx.vignette);
    gl.uniform1f(u.uFlash, fx.flash);
    gl.uniform1f(u.uWhite, fx.white);
    gl.uniform1f(u.uLetterbox, fx.letterbox);
    gl.uniform1f(u.uLightSteps, 16);
    const g = fx.grade;
    const graded = g.exposure !== 1 || g.contrast !== 1 || g.saturation !== 1 || (g.tint & 0xffffff) !== 0xffffff;
    gl.uniform1i(u.uGrade, graded ? 1 : 0);
    gl.uniform4f(u.uGradeA, g.exposure, g.contrast, g.saturation, 0);
    gl.uniform3fv(u.uTint, rgbInto(g.tint, this.rgb));
    gl.bindVertexArray(gpu.empty);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.activeTexture(gl.TEXTURE0);
    return 1;
  }

  destroy() {
    this.canvas.removeEventListener("webglcontextlost", this.onLost);
    this.canvas.removeEventListener("webglcontextrestored", this.onRestored);
    const gpu = this.gpu;
    const gl = this.gl;
    if (gpu && !gl.isContextLost()) {
      for (const t of [gpu.scene, gpu.light, gpu.bloomA, gpu.bloomB, gpu.out]) freeTarget(gl, t);
      for (const p of [gpu.sprite, gpu.sky, gpu.bright, gpu.blur, gpu.post]) gl.deleteProgram(p.prog);
      gpu.atlas.destroy();
      gpu.bg.destroy();
      gpu.world.destroy();
      gpu.overlay.destroy();
      gpu.particles.destroy();
      gpu.lights.destroy();
      gpu.terrain.destroy();
      gl.deleteVertexArray(gpu.empty);
    }
    this.gpu = null;
  }
}

/**
 * GPU frame time from `EXT_disjoint_timer_query_webgl2`, read a few frames late from a small ring of queries so it
 * never waits on the GPU.
 */
class GpuTimer {
  ms = -1;
  private queries: WebGLQuery[] = [];
  private busy: boolean[] = [];
  private slot = 0;
  private active = false;

  private constructor(
    private gl: WebGL2RenderingContext,
    private ext: { TIME_ELAPSED_EXT: number; GPU_DISJOINT_EXT: number }
  ) {
    for (let i = 0; i < 4; i++) {
      const q = gl.createQuery();
      if (q) this.queries.push(q);
      this.busy.push(false);
    }
  }

  static create(gl: WebGL2RenderingContext): GpuTimer | null {
    const ext = gl.getExtension("EXT_disjoint_timer_query_webgl2") as {
      TIME_ELAPSED_EXT: number;
      GPU_DISJOINT_EXT: number;
    } | null;
    return ext ? new GpuTimer(gl, ext) : null;
  }

  begin() {
    const gl = this.gl;
    const n = this.queries.length;
    if (n === 0) return;
    // Collect any finished results first, oldest first
    for (let k = 1; k <= n; k++) {
      const i = (this.slot + k) % n;
      const q = this.queries[i];
      if (!this.busy[i] || !gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) continue;
      const ns = gl.getQueryParameter(q, gl.QUERY_RESULT) as number;
      this.busy[i] = false;
      if (!gl.getParameter(this.ext.GPU_DISJOINT_EXT))
        this.ms = this.ms < 0 ? ns / 1e6 : this.ms + (ns / 1e6 - this.ms) * 0.1;
    }
    if (this.busy[this.slot]) return;
    gl.beginQuery(this.ext.TIME_ELAPSED_EXT, this.queries[this.slot]);
    this.active = true;
  }

  end() {
    if (!this.active) return;
    this.gl.endQuery(this.ext.TIME_ELAPSED_EXT);
    this.busy[this.slot] = true;
    this.active = false;
    this.slot = (this.slot + 1) % this.queries.length;
  }
}

function bindTex(gl: WebGL2RenderingContext, unit: number, tex: WebGLTexture, loc: WebGLUniformLocation | null) {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.uniform1i(loc, unit);
}
