/**
 * The renderer: pixel-perfect WebGL2 with a Canvas 2D fallback. Build a `RenderFrame` once with `createFrame`, fill
 * its buffers each frame, and hand it to `renderer.render`.
 */
import { Canvas2DRenderer } from "./canvas2d/renderer";
import { WebGLRenderer } from "./gl/renderer";
import type { CreateFrame, CreateRenderer, Renderer, RendererOptions } from "./types";
import { createFrame as makeFrame } from "./frame";

export * from "./types";
export { Atlas } from "./atlas";
export { SpriteList, ParticleList, LightList, TextList } from "./buffers";
export { clearFrame, TERRAIN_LAYER, PARTICLE_LAYER } from "./frame";
export { measureText, GLYPHS, INK } from "./font";
export { DebugOverlay, debugFromUrl, type DebugWorldStats } from "./debug";
export { makeCanvas } from "./canvas";
export { mixRgb } from "./view";

const GL_ATTRS: WebGLContextAttributes = {
  alpha: false,
  antialias: false,
  depth: false,
  stencil: false,
  premultipliedAlpha: true,
  preserveDrawingBuffer: false,
  powerPreference: "default",
};

/** A fresh canvas in place of one whose WebGL setup failed (a canvas keeps the first context type it gave out). */
function replaceCanvas(canvas: HTMLCanvasElement): HTMLCanvasElement | null {
  const parent = canvas.parentNode;
  if (!parent || typeof document === "undefined") return null;
  const fresh = document.createElement("canvas");
  for (const a of Array.from(canvas.attributes)) fresh.setAttribute(a.name, a.value);
  parent.replaceChild(fresh, canvas);
  return fresh;
}

/**
 * WebGL2 when the browser has it, else Canvas 2D. If WebGL2 exists but fails to set up, the canvas is swapped for
 * a fresh one (same attributes, same place in the page) for the fallback. Throws when neither is available.
 */
export const createRenderer: CreateRenderer = (canvas: HTMLCanvasElement, opts: RendererOptions): Renderer => {
  let target = canvas;
  if (opts.prefer !== "canvas2d") {
    const gl = canvas.getContext("webgl2", GL_ATTRS);
    if (gl) {
      try {
        return new WebGLRenderer(canvas, gl, opts);
      } catch (e) {
        console.error("render: WebGL2 setup failed, falling back to Canvas 2D", e);
        target = replaceCanvas(canvas) ?? canvas;
      }
    }
  }
  const ctx = target.getContext("2d", { alpha: false });
  if (!ctx) throw new Error("render: neither WebGL2 nor Canvas 2D is available");
  return new Canvas2DRenderer(target, ctx, opts);
};

/** Allocates a frame and its reusable buffers, with daylight and no effects. */
export const createFrame: CreateFrame = makeFrame;
