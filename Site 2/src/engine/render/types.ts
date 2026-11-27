/**
 * The renderer contract (engine spec A3). Core, terrain and game code build against these types; extend only by
 * adding fields. `createRenderer` and `createFrame` are implemented in `render/index.ts` (a `.ts` file cannot hold
 * body-less function declarations), typed by `CreateRenderer` and `CreateFrame` below.
 *
 * Colours: sprite, particle, parallax and text tints are RGBA (`0xRRGGBBAA`, so `0xffffffff` is neutral); light,
 * sky, ambient and grade colours are RGB (`0xRRGGBB`). World space is y-down, in world px.
 *
 * Additions to the spec's contract (fields only): `RenderFrame.terrainLayer` and `particleLayer`,
 * `TerrainView.emissive` and `maxHp`, `RendererOptions.prefer` and `gpuTiming`, `Renderer.stats` with `RenderStats`,
 * the `PARTICLE_*` flags, and the doc comments giving units.
 */

/** A rectangle of an atlas page, with its pivot. */
export interface Frame {
  page: number;
  x: number;
  y: number;
  w: number;
  h: number;
  ox: number;
  oy: number;
}

/** Builds texture atlases from canvases painted at boot (the game's pixel-art painters). */
export interface AtlasBuilder {
  add(id: string, source: HTMLCanvasElement | OffscreenCanvas, ox?: number, oy?: number): number; // returns frame index
  frame(index: number): Frame;
  index(id: string): number;
}

/** Struct-of-arrays sprite list for one frame; the game pushes, the renderer draws. Reused every frame. */
export interface SpriteBuffer {
  count: number;
  push(
    frame: number,
    x: number,
    y: number,
    layer: number,
    flags: number,
    tint: number,
    scale: number,
    rot: number
  ): void;
  clear(): void;
}
export const SPRITE_FLIP_X = 1,
  SPRITE_EMISSIVE = 2,
  SPRITE_ADDITIVE = 4,
  SPRITE_OUTLINE = 8,
  SPRITE_SILHOUETTE = 16;

/** Particle flags share the sprite bits they mean the same as. */
export const PARTICLE_EMISSIVE = SPRITE_EMISSIVE,
  PARTICLE_ADDITIVE = SPRITE_ADDITIVE;

/** Particles as coloured pixels or tiny quads (no texture), batched. */
export interface ParticleBuffer {
  count: number;
  push(x: number, y: number, size: number, color: number, flags: number): void; // flags: emissive/additive
  clear(): void;
}

/** Point lights for the light pass. */
export interface LightBuffer {
  count: number;
  push(x: number, y: number, radius: number, color: number, intensity: number): void;
  clear(): void;
}

/** Screen-space text in the pixel font (atlas glyphs). */
export interface TextBuffer {
  push(text: string, x: number, y: number, color: number, scale: number, align: 0 | 1 | 2, layer: number): void;
  clear(): void;
}

/**
 * A backdrop layer. `factor` is how far it moves with the camera (0 = fixed to the screen, 1 = part of the world);
 * the frame's pivot sits at screen x = `(0 - camera.x × factor)` and y = `(y - camera.y × factor)`, both measured from
 * the view centre, so at factor 1 `y` is a world y. `repeatX` tiles it across the view.
 */
export interface ParallaxLayer {
  frame: number;
  factor: number;
  y: number;
  tint: number;
  repeatX: boolean;
}

/** A ring of distortion: centre (world px), current radius and ring width (world px), and peak push in world px. */
export interface Shockwave {
  x: number;
  y: number;
  radius: number;
  width: number;
  strength: number;
}

/** Colour grade; neutral is exposure 1, contrast 1, saturation 1, tint 0xffffff. */
export interface Grade {
  exposure: number;
  contrast: number;
  saturation: number;
  tint: number /* RGB */;
}

/** Everything the renderer needs to draw one frame. */
export interface RenderFrame {
  time: number;
  /** World-space centre of the view, zoom (1 = native pixel scale), screen shake in world px. */
  camera: { x: number; y: number; zoom: number; shakeX: number; shakeY: number };
  sky: { top: number; mid: number; bottom: number; stars: number /* 0..1 */ };
  ambient: number; // RGB ambient light; 0xffffff = full daylight
  parallax: ParallaxLayer[];
  terrain: TerrainView | null;
  sprites: SpriteBuffer; // layers: 0 back … 15 front; world space
  particles: ParticleBuffer; // world space
  lights: LightBuffer; // world space
  text: TextBuffer; // screen space (overlay)
  overlay: SpriteBuffer; // screen-space sprites (HUD, cards)
  effects: {
    shockwaves: Shockwave[];
    flash: number; // 0..1 additive white
    white: number; // 0..1 white-out
    letterbox: number; // 0..1
    vignette: number; // 0..1
    chroma: number; // 0..1 chromatic aberration punch
    bloom: number; // 0..1
    grade: Grade;
  };
  /** Sprite layers below this draw behind the terrain (default 4). */
  terrainLayer?: number;
  /** World particles draw after this sprite layer (default 11), so layers above it cover them. */
  particleLayer?: number;
}

/** What the terrain gives the renderer (A2 implements it). */
export interface TerrainView {
  rows: number;
  ring: number;
  tile: number;
  startCol: number;
  fg: Uint8Array;
  bg: Uint8Array;
  variant: Uint8Array;
  hp: Uint8Array;
  /**
   * Chunk indices (chunk = 16×16 cells), column-major like cells:
   * `((col & (ring - 1)) / chunkSize | 0) * (rows / chunkSize) + (row / chunkSize | 0)`; returns count.
   */
  drainDirty(out: number[]): number; // chunk indices (chunk = 16×16 cells); returns count
  chunkSize: number;
  /** Atlas frames for material tiles: tiles[material * VARIANTS + variant] = frame index (8×8). */
  tiles: Int32Array;
  variants: number;
  /** Per material id, glow 0..255 (lava, crystals): lit regardless of ambient and fed to bloom. */
  emissive?: Uint8Array;
  /** Per material id, the full hp, so the renderer can draw cracks as `hp` falls; 255 means unbreakable. */
  maxHp?: Uint8Array;
}

export interface RendererOptions {
  pixelScale: number;
  maxSprites?: number;
  maxParticles?: number;
  quality?: "high" | "mid" | "low";
  /** Force the Canvas 2D fallback (for testing it, or a machine where WebGL misbehaves). */
  prefer?: "webgl2" | "canvas2d";
  /** Measure GPU time per frame with a timer query where the browser offers one (for the debug overlay). */
  gpuTiming?: boolean;
}

export interface Renderer {
  readonly kind: "webgl2" | "canvas2d";
  readonly atlas: AtlasBuilder;
  resize(cssWidth: number, cssHeight: number, dpr: number, pixelScale: number): void;
  /** Internal (low-res) size in world px at zoom 1. */
  readonly width: number;
  readonly height: number;
  render(frame: RenderFrame): void;
  destroy(): void;
  /** Counters from the last `render`, for the debug overlay. */
  readonly stats: RenderStats;
}

/** What the last frame cost the renderer. */
export interface RenderStats {
  drawCalls: number;
  sprites: number;
  particles: number;
  lights: number;
  /** CPU time spent inside `render`, ms. */
  cpuMs: number;
  /** Terrain chunks re-uploaded this frame. */
  chunks: number;
  /** GPU time of a recent frame, ms, when `gpuTiming` is on and supported; otherwise -1. */
  gpuMs?: number;
}

/** WebGL2, else canvas2d. Implemented by `createRenderer` in `render/index.ts`. */
export type CreateRenderer = (canvas: HTMLCanvasElement, opts: RendererOptions) => Renderer;
/** Allocates the reusable buffers. Implemented by `createFrame` in `render/index.ts`. */
export type CreateFrame = () => RenderFrame;
