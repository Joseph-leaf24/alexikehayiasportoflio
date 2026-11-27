import type { RenderFrame } from "./types";

/** The visible world rectangle for one frame: its top-left in world px, the zoom and the internal size. */
export interface View {
  left: number;
  top: number;
  zoom: number;
  w: number;
  h: number;
}

export function makeView(): View {
  return { left: 0, top: 0, zoom: 1, w: 1, h: 1 };
}

/**
 * Works out the view from the camera. The top-left is snapped so world pixels land on whole internal pixels (at
 * zoom 1, whole world px), which keeps a panning camera from shimmering; shake is applied before the snap.
 */
export function computeView(cam: RenderFrame["camera"], w: number, h: number, out: View): View {
  const zoom = cam.zoom > 0 && Number.isFinite(cam.zoom) ? cam.zoom : 1;
  const left = cam.x + cam.shakeX - w / (2 * zoom);
  const top = cam.y + cam.shakeY - h / (2 * zoom);
  out.left = Math.round(left * zoom) / zoom;
  out.top = Math.round(top * zoom) / zoom;
  out.zoom = zoom;
  out.w = w;
  out.h = h;
  return out;
}

/** Clip-space scale for view-relative world px: `clip = rel × scale + (-1, 1)` (y down on screen). */
export function clipScale(view: View, out: Float32Array): Float32Array {
  out[0] = (2 * view.zoom) / view.w;
  out[1] = (-2 * view.zoom) / view.h;
  return out;
}

/** Internal size for a canvas: device pixels divided by the pixel scale, at least 1×1. */
export function internalSize(cssWidth: number, cssHeight: number, dpr: number, pixelScale: number) {
  const s = pixelScale > 0 ? pixelScale : 1;
  return {
    w: Math.max(1, Math.ceil((cssWidth * dpr) / s)),
    h: Math.max(1, Math.ceil((cssHeight * dpr) / s)),
  };
}

/** Where a parallax layer's pivot sits relative to the view's top-left, in world px at the view's zoom. */
export function parallaxOrigin(view: View, camX: number, camY: number, factor: number, y: number, out: Float32Array) {
  const hw = view.w / (2 * view.zoom);
  const hh = view.h / (2 * view.zoom);
  out[0] = hw - camX * factor;
  out[1] = hh + y - camY * factor;
  return out;
}

/** Splits a colour into floats: RGB `0xRRGGBB` → [r, g, b] in 0..1. */
export function rgbInto(c: number, out: Float32Array, o = 0) {
  out[o] = ((c >>> 16) & 255) / 255;
  out[o + 1] = ((c >>> 8) & 255) / 255;
  out[o + 2] = (c & 255) / 255;
  return out;
}

/** Mixes two RGB colours, t in 0..1. */
export function mixRgb(a: number, b: number, t: number) {
  const k = t < 0 ? 0 : t > 1 ? 1 : t;
  const r = ((a >>> 16) & 255) + ((((b >>> 16) & 255) - ((a >>> 16) & 255)) * k + 0.5);
  const g = ((a >>> 8) & 255) + ((((b >>> 8) & 255) - ((a >>> 8) & 255)) * k + 0.5);
  const bl = (a & 255) + (((b & 255) - (a & 255)) * k + 0.5);
  return ((r & 255) << 16) | ((g & 255) << 8) | (bl & 255);
}
