import { Atlas, type AtlasSource } from "../atlas";
import { LightList, ParticleList, SpriteList, TextList } from "../buffers";
import { cssRgb, cssRgba, makeCanvas, type Canvas2D } from "../canvas";
import { layoutText, registerFont, type FontFrames } from "../font";
import { PARTICLE_LAYER, TERRAIN_LAYER } from "../frame";
import { countingSort } from "../sort";
import { chunkOf, TileSheet } from "../tiles";
import {
  SPRITE_FLIP_X,
  SPRITE_SILHOUETTE,
  type RenderFrame,
  type Renderer,
  type RendererOptions,
  type RenderStats,
  type TerrainView,
} from "../types";
import { computeView, internalSize, makeView, parallaxOrigin } from "../view";

/** Most tinted copies kept (glyphs, silhouettes); the cache is dropped whole when it fills. */
const TINT_CACHE = 1024;

/**
 * The fallback when WebGL2 is missing: the same interface on Canvas 2D at the same low resolution. Terrain comes
 * from cached chunk canvases, sprites from their atlas sources, particles are `fillRect`s. Lighting, bloom,
 * shockwaves, chroma and grading are skipped; ambient darkening, flash, white-out and letterbox are kept.
 */
export class Canvas2DRenderer implements Renderer {
  readonly kind = "canvas2d" as const;
  readonly atlas = new Atlas(2048);
  readonly stats: RenderStats = { drawCalls: 0, sprites: 0, particles: 0, lights: 0, cpuMs: 0, chunks: 0 };
  width = 1;
  height = 1;
  private low: AtlasSource;
  private lc: Canvas2D;
  private font: FontFrames;
  private view = makeView();
  private tmp2 = new Float32Array(2);
  private pixelScale: number;
  private sheet = new TileSheet();
  private chunks = new Map<number, [AtlasSource, Canvas2D]>();
  private terrain: TerrainView | null = null;
  private dirty: number[] = [];
  private xy = new Int32Array(2);
  private keys = new Uint16Array(1024);
  private order = new Uint32Array(1024);
  private counts = new Uint32Array(18);
  private colors = new Map<number, string>();
  private tinted = new Map<number, AtlasSource>();
  private overlayList = new SpriteList(256);
  private sky = { key: "", grad: null as CanvasGradient | null };

  constructor(
    private canvas: HTMLCanvasElement,
    private ctx: CanvasRenderingContext2D,
    opts: RendererOptions
  ) {
    this.pixelScale = opts.pixelScale > 0 ? opts.pixelScale : 3;
    this.font = registerFont(this.atlas);
    [this.low, this.lc] = makeCanvas(1, 1);
    const s = internalSize(canvas.width || 1, canvas.height || 1, 1, this.pixelScale);
    this.setSize(s.w, s.h);
  }

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
    this.width = w;
    this.height = h;
    this.low.width = w;
    this.low.height = h;
    this.lc.imageSmoothingEnabled = false;
    this.sky.key = "";
  }

  render(frame: RenderFrame) {
    const t0 = performance.now();
    const st = this.stats;
    const c = this.lc;
    const W = this.width;
    const H = this.height;
    const view = computeView(frame.camera, W, H, this.view);
    const z = view.zoom;
    let calls = 0;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.globalAlpha = 1;
    c.globalCompositeOperation = "source-over";
    this.drawSky(frame);
    calls++;

    // Parallax
    const cam = frame.camera;
    for (let li = 0; li < frame.parallax.length; li++) {
      const p = frame.parallax[li];
      const f = this.atlas.frame(p.frame);
      const src = this.atlas.sources[p.frame];
      if (f.w === 0 || !src) continue;
      const o = parallaxOrigin(view, cam.x + cam.shakeX, cam.y + cam.shakeY, p.factor, p.y, this.tmp2);
      c.globalAlpha = (p.tint & 255) / 255;
      const y = Math.round((o[1] - f.oy) * z);
      const left = o[0] - f.ox;
      const k0 = p.repeatX ? Math.floor(-left / f.w) : 0;
      const k1 = p.repeatX ? Math.ceil((W / z - left) / f.w) : 1;
      for (let k = k0; k < k1; k++) {
        c.drawImage(src, Math.round((left + k * f.w) * z), y, f.w * z, f.h * z);
        calls++;
      }
    }
    c.globalAlpha = 1;

    const sprites = frame.sprites instanceof SpriteList ? frame.sprites : null;
    const n = sprites ? this.sortSprites(sprites) : 0;
    const terrainLayer = frame.terrainLayer ?? TERRAIN_LAYER;
    const particleLayer = frame.particleLayer ?? PARTICLE_LAYER;
    let k = 0;
    if (sprites) {
      calls += this.drawSprites(sprites, 0, n, terrainLayer - 1, view.left, view.top, z);
      k = this.cursor;
    }
    if (frame.terrain) calls += this.drawTerrain(frame.terrain);
    if (sprites) {
      calls += this.drawSprites(sprites, k, n, particleLayer, view.left, view.top, z);
      k = this.cursor;
    }
    const parts = frame.particles instanceof ParticleList ? frame.particles : null;
    if (parts) calls += this.drawParticles(parts);
    if (sprites) calls += this.drawSprites(sprites, k, n, 15, view.left, view.top, z);

    // Ambient darkening stands in for the light pass
    const amb = frame.ambient & 0xffffff;
    if (amb !== 0xffffff) {
      c.globalCompositeOperation = "multiply";
      c.fillStyle = cssRgb(amb);
      c.fillRect(0, 0, W, H);
      c.globalCompositeOperation = "source-over";
      calls++;
    }
    const fx = frame.effects;
    if (fx.flash > 0) {
      c.globalCompositeOperation = "lighter";
      c.globalAlpha = Math.min(1, fx.flash);
      c.fillStyle = "#fff";
      c.fillRect(0, 0, W, H);
      c.globalCompositeOperation = "source-over";
    }
    if (fx.white > 0) {
      c.globalAlpha = Math.min(1, fx.white);
      c.fillStyle = "#fff";
      c.fillRect(0, 0, W, H);
    }
    c.globalAlpha = 1;
    if (fx.letterbox > 0) {
      const bar = Math.floor(fx.letterbox * H * 0.12);
      c.fillStyle = "#000";
      c.fillRect(0, 0, W, bar);
      c.fillRect(0, H - bar, W, bar);
    }
    calls += this.drawOverlay(frame);

    // Present
    const out = this.ctx;
    out.imageSmoothingEnabled = false;
    out.drawImage(this.low, 0, 0, this.canvas.width, this.canvas.height);
    st.drawCalls = calls + 1;
    st.sprites = n;
    st.particles = parts?.count ?? 0;
    st.lights = frame.lights instanceof LightList ? frame.lights.count : 0;
    st.cpuMs = performance.now() - t0;
  }

  private drawSky(frame: RenderFrame) {
    const c = this.lc;
    const s = frame.sky;
    const key = `${s.top},${s.mid},${s.bottom},${this.height}`;
    if (key !== this.sky.key || !this.sky.grad) {
      const g = c.createLinearGradient(0, 0, 0, this.height);
      g.addColorStop(0, cssRgb(s.top));
      g.addColorStop(0.6, cssRgb(s.mid));
      g.addColorStop(1, cssRgb(s.bottom));
      this.sky.grad = g;
      this.sky.key = key;
    }
    c.fillStyle = this.sky.grad;
    c.fillRect(0, 0, this.width, this.height);
    if (s.stars > 0.05) {
      c.fillStyle = "#fff8e8";
      const n = Math.round((this.width * this.height) / 1100);
      for (let i = 0; i < n; i++) {
        const h = Math.imul(i + 1, 2654435761) >>> 0;
        c.globalAlpha = s.stars * (0.5 + 0.5 * Math.sin(frame.time * (0.7 + (h & 7) * 0.3) + i));
        c.fillRect(h % this.width, (h >>> 12) % Math.max(1, Math.floor(this.height * 0.6)) | 0, 1, 1);
      }
      c.globalAlpha = 1;
    }
  }

  /** Where the last `drawSprites` stopped in sorted order. */
  private cursor = 0;

  private sortSprites(list: SpriteList) {
    const n = list.count;
    if (this.keys.length < n) {
      this.keys = new Uint16Array(n * 2);
      this.order = new Uint32Array(n * 2);
    }
    for (let i = 0; i < n; i++) this.keys[i] = list.layer[i];
    countingSort(this.keys, n, 16, this.counts, this.order);
    this.cursor = 0;
    return n;
  }

  /** Draws sorted sprites from position `from` while their layer is ≤ `maxLayer`; returns drawImage calls. */
  private drawSprites(list: SpriteList, from: number, n: number, maxLayer: number, ox: number, oy: number, z: number) {
    const c = this.lc;
    let k = from;
    let calls = 0;
    for (; k < n; k++) {
      const i = this.order[k];
      if (list.layer[i] > maxLayer) break;
      const fi = list.frame[i];
      const f = this.atlas.frames[fi];
      let src = this.atlas.sources[fi];
      if (!f || !src || f.w === 0) continue;
      const tint = list.tint[i];
      if (list.flags[i] & SPRITE_SILHOUETTE) src = this.tint(fi, src, tint | 255);
      const s = list.scale[i] * z;
      const flip = list.flags[i] & SPRITE_FLIP_X ? -1 : 1;
      const x = Math.round((list.x[i] - ox) * z);
      const y = Math.round((list.y[i] - oy) * z);
      c.globalAlpha = (tint & 255) / 255;
      // Rotation after the flip and scale, as the GPU path does
      const cr = Math.cos(list.rot[i]);
      const sr = Math.sin(list.rot[i]);
      c.setTransform(s * flip * cr, s * flip * sr, -s * sr, s * cr, x, y);
      c.drawImage(src, -f.ox, -f.oy);
      calls++;
    }
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.globalAlpha = 1;
    this.cursor = k;
    return calls;
  }

  /** A tinted copy of a frame (silhouettes and text), cached by frame and colour. */
  private tint(frame: number, src: AtlasSource, color: number): AtlasSource {
    const key = frame * 4294967296 + (color >>> 0);
    let t = this.tinted.get(key);
    if (t) return t;
    if (this.tinted.size >= TINT_CACHE) this.tinted.clear();
    const [cv, cx] = makeCanvas(src.width, src.height);
    cx.drawImage(src, 0, 0);
    cx.globalCompositeOperation = "source-in";
    cx.fillStyle = cssRgba(color | 255);
    cx.fillRect(0, 0, src.width, src.height);
    t = cv;
    this.tinted.set(key, t);
    return t;
  }

  private drawTerrain(t: TerrainView): number {
    const c = this.lc;
    const view = this.view;
    const z = view.zoom;
    const T = t.tile;
    const cs = t.chunkSize;
    const span = cs * T;
    if (t !== this.terrain) {
      this.terrain = t;
      this.chunks.clear();
      t.drainDirty(this.dirty);
    } else {
      const n = t.drainDirty(this.dirty);
      for (let i = 0; i < n; i++) this.chunks.delete(this.dirty[i]);
    }
    this.dirty.length = 0;
    if (this.sheet.sync(t, this.atlas)) this.chunks.clear();
    const sheet = this.sheet.canvas;
    if (!sheet) return 0;
    const rowsOfChunks = Math.ceil(t.rows / cs);
    const c0 = Math.max(t.startCol, Math.floor(view.left / T));
    const c1 = Math.min(t.startCol + t.ring, Math.ceil((view.left + view.w / z) / T) + 1);
    const r0 = Math.max(0, Math.floor(view.top / T / cs));
    const r1 = Math.min(rowsOfChunks, Math.ceil((view.top + view.h / z) / T / cs));
    let calls = 0;
    let built = 0;
    for (let wc = Math.floor(c0 / cs) * cs; wc < c1; wc += cs) {
      const cc = (((wc % t.ring) + t.ring) % t.ring) / cs;
      for (let cr = r0; cr < r1; cr++) {
        const idx = cc * rowsOfChunks + cr;
        let ch = this.chunks.get(idx);
        if (!ch) {
          ch = this.paintChunk(t, sheet, idx, makeCanvas(span, span));
          this.chunks.set(idx, ch);
          built++;
        }
        c.drawImage(
          ch[0],
          Math.round((wc * T - view.left) * z),
          Math.round((cr * span - view.top) * z),
          span * z,
          span * z
        );
        calls++;
      }
    }
    this.stats.chunks = built;
    return calls;
  }

  private paintChunk(t: TerrainView, sheet: AtlasSource, idx: number, ch: [AtlasSource, Canvas2D]) {
    const [, x] = ch;
    const T = t.tile;
    const cs = t.chunkSize;
    const V = Math.max(1, t.variants);
    chunkOf(idx, t, this.xy);
    const col0 = this.xy[0] * cs;
    const row0 = this.xy[1] * cs;
    for (let dc = 0; dc < cs && col0 + dc < t.ring; dc++) {
      const base = (col0 + dc) * t.rows;
      for (let dr = 0; dr < cs && row0 + dr < t.rows; dr++) {
        const i = base + row0 + dr;
        const fg = t.fg[i];
        const bg = t.bg[i];
        if (!fg && !bg) continue;
        const v = t.variant[i] % V;
        const m = fg || bg;
        x.drawImage(sheet, v * T, m * T, T, T, dc * T, dr * T, T, T);
        if (!fg) {
          x.fillStyle = "rgba(10,8,22,0.62)";
          x.fillRect(dc * T, dr * T, T, T);
        } else if (row0 + dr > 0 && t.fg[i - 1] === 0) {
          x.fillStyle = "rgba(255,255,255,0.2)";
          x.fillRect(dc * T, dr * T, T, 1);
        }
      }
    }
    return ch;
  }

  private drawParticles(p: ParticleList): number {
    const c = this.lc;
    const view = this.view;
    const z = view.zoom;
    const ww = view.w / z;
    const wh = view.h / z;
    let last = -1;
    for (let i = 0; i < p.count; i++) {
      const x = p.x[i] - view.left;
      const y = p.y[i] - view.top;
      if (x < -4 || y < -4 || x > ww || y > wh) continue;
      const col = p.color[i];
      if (col !== last) {
        let s = this.colors.get(col);
        if (!s) {
          if (this.colors.size > 4096) this.colors.clear();
          s = cssRgba(col);
          this.colors.set(col, s);
        }
        c.fillStyle = s;
        last = col;
      }
      c.fillRect(Math.floor(x) * z, Math.floor(y) * z, p.size[i] * z, p.size[i] * z);
    }
    return 1;
  }

  private drawOverlay(frame: RenderFrame): number {
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
    // Text glyphs are white and need their colour: draw them as silhouettes
    for (let i = 0; i < list.count; i++)
      if ((list.tint[i] | 0xff) >>> 0 !== 0xffffffff) list.flags[i] |= SPRITE_SILHOUETTE;
    const n = this.sortSprites(list);
    return this.drawSprites(list, 0, n, 15, 0, 0, 1);
  }

  destroy() {
    this.chunks.clear();
    this.tinted.clear();
    this.colors.clear();
  }
}
