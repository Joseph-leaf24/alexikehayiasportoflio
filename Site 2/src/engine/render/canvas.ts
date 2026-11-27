export type Canvas2D = OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;

/** A 2D canvas for painting at boot: an OffscreenCanvas where there is one, else a detached <canvas>. */
export function makeCanvas(w: number, h: number): [HTMLCanvasElement | OffscreenCanvas, Canvas2D] {
  const c: HTMLCanvasElement | OffscreenCanvas =
    typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(w, h) : document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d") as Canvas2D | null;
  if (!ctx) throw new Error("render: no 2D canvas");
  ctx.imageSmoothingEnabled = false;
  return [c, ctx];
}

/** CSS colour for an RGBA `0xRRGGBBAA` number. */
export function cssRgba(c: number) {
  const a = (c & 255) / 255;
  return `rgba(${(c >>> 24) & 255},${(c >>> 16) & 255},${(c >>> 8) & 255},${Math.round(a * 1000) / 1000})`;
}

/** CSS colour for an RGB `0xRRGGBB` number. */
export function cssRgb(c: number) {
  return `#${((c & 0xffffff) | 0x1000000).toString(16).slice(1)}`;
}
