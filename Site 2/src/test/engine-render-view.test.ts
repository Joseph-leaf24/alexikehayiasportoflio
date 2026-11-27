import { describe, expect, it } from "vitest";
import { clipScale, computeView, internalSize, makeView, mixRgb, parallaxOrigin, rgbInto } from "@/engine/render/view";

const cam = (x: number, y: number, zoom = 1, shakeX = 0, shakeY = 0) => ({ x, y, zoom, shakeX, shakeY });

describe("view", () => {
  it("centres the camera and snaps the top-left to whole pixels at zoom 1", () => {
    const v = computeView(cam(100.4, 50.6), 480, 300, makeView());
    expect(v.left).toBe(Math.round(100.4 - 240));
    expect(v.top).toBe(Math.round(50.6 - 150));
    expect(Number.isInteger(v.left)).toBe(true);
    expect(v.zoom).toBe(1);
  });

  it("applies shake before snapping", () => {
    const v = computeView(cam(0, 0, 1, 2.6, -1.2), 100, 100, makeView());
    expect(v.left).toBe(Math.round(-50 + 2.6));
    expect(v.top).toBe(Math.round(-50 - 1.2));
  });

  it("snaps to whole output pixels at a fractional zoom", () => {
    const v = computeView(cam(10.3, 7.7, 1.5), 480, 300, makeView());
    expect(v.zoom).toBe(1.5);
    expect(v.left * 1.5).toBeCloseTo(Math.round(v.left * 1.5), 9);
    // The view spans the internal size divided by the zoom
    expect(v.left).toBeCloseTo(10.3 - 160, 0);
  });

  it("falls back to zoom 1 for a bad zoom", () => {
    expect(computeView(cam(0, 0, 0), 10, 10, makeView()).zoom).toBe(1);
    expect(computeView(cam(0, 0, NaN), 10, 10, makeView()).zoom).toBe(1);
  });

  it("maps view-relative world px to clip space, y down", () => {
    const v = computeView(cam(0, 0, 2), 400, 200, makeView());
    const s = clipScale(v, new Float32Array(2));
    // The view's top-left is clip (-1, 1); its far corner (w/zoom, h/zoom) is (1, -1)
    expect(0 * s[0] - 1).toBe(-1);
    expect(200 * s[0] - 1).toBeCloseTo(1);
    expect(100 * s[1] + 1).toBeCloseTo(-1);
  });

  it("sizes the internal buffer from CSS size, device pixel ratio and pixel scale", () => {
    expect(internalSize(1440, 900, 1, 3)).toEqual({ w: 480, h: 300 });
    expect(internalSize(1440, 900, 2, 5)).toEqual({ w: 576, h: 360 });
    expect(internalSize(1000, 10, 1, 3)).toEqual({ w: 334, h: 4 });
    expect(internalSize(0, 0, 1, 3)).toEqual({ w: 1, h: 1 });
  });

  it("places parallax layers by factor: 0 pins to the screen, 1 moves with the world", () => {
    const v = computeView(cam(1000, 500), 480, 300, makeView());
    const o = new Float32Array(2);
    parallaxOrigin(v, 1000, 500, 0, 20, o);
    expect([o[0], o[1]]).toEqual([240, 170]);
    parallaxOrigin(v, 1000, 500, 1, 520, o);
    // At factor 1 the layer's y is a world y: world 520 is 20 px below the centre
    expect([o[0], o[1]]).toEqual([240 - 1000, 170]);
  });

  it("splits and mixes RGB colours", () => {
    const out = rgbInto(0xff8000, new Float32Array(3));
    expect(out[0]).toBe(1);
    expect(out[1]).toBeCloseTo(128 / 255);
    expect(out[2]).toBe(0);
    expect(mixRgb(0x000000, 0xffffff, 0)).toBe(0x000000);
    expect(mixRgb(0x000000, 0xffffff, 1)).toBe(0xffffff);
    expect(mixRgb(0x204060, 0x204060, 0.37)).toBe(0x204060);
    expect(mixRgb(0x000000, 0x0000ff, 0.5)).toBe(0x000080);
    expect(mixRgb(0x000000, 0xffffff, 2)).toBe(0xffffff);
  });
});
