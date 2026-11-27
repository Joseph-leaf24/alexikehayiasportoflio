import { LightList, ParticleList, SpriteList, TextList } from "./buffers";
import type { RenderFrame } from "./types";

/** Default sprite layer the terrain draws at: layers below it are behind the ground. */
export const TERRAIN_LAYER = 4;
/** Default sprite layer world particles draw after. */
export const PARTICLE_LAYER = 11;

/** Allocates a frame with its reusable buffers and neutral settings: daylight, no effects, neutral grade. */
export function createFrame(): RenderFrame {
  return {
    time: 0,
    camera: { x: 0, y: 0, zoom: 1, shakeX: 0, shakeY: 0 },
    sky: { top: 0x3a78d0, mid: 0x86b6e6, bottom: 0xd8eaf6, stars: 0 },
    ambient: 0xffffff,
    parallax: [],
    terrain: null,
    sprites: new SpriteList(4096),
    particles: new ParticleList(8192),
    lights: new LightList(128),
    text: new TextList(64),
    overlay: new SpriteList(256),
    effects: {
      shockwaves: [],
      flash: 0,
      white: 0,
      letterbox: 0,
      vignette: 0,
      chroma: 0,
      bloom: 0,
      grade: { exposure: 1, contrast: 1, saturation: 1, tint: 0xffffff },
    },
    terrainLayer: TERRAIN_LAYER,
    particleLayer: PARTICLE_LAYER,
  };
}

/** Empties every per-frame buffer, ready for the next frame's pushes. */
export function clearFrame(f: RenderFrame) {
  f.sprites.clear();
  f.particles.clear();
  f.lights.clear();
  f.text.clear();
  f.overlay.clear();
}
