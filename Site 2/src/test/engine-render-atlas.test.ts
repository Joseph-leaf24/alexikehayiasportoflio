import { describe, expect, it } from "vitest";
import { Atlas } from "@/engine/render/atlas";
import { Skyline } from "@/engine/render/packer";

/** A stand-in canvas: the atlas only reads its size until it uploads. */
const box = (width: number, height: number) => ({ width, height }) as unknown as HTMLCanvasElement;

type Rect = { x: number; y: number; w: number; h: number };
const overlaps = (a: Rect, b: Rect, pad: number) =>
  a.x < b.x + b.w + pad && b.x < a.x + a.w + pad && a.y < b.y + b.h + pad && b.y < a.y + a.h + pad;

describe("skyline packer", () => {
  it("packs mixed rectangles inside the page without overlaps, keeping a 1 px gap", () => {
    const k = new Skyline(256, 256, 1);
    const placed: Rect[] = [];
    let seed = 1;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 30;
    for (let i = 0; i < 120; i++) {
      const w = 1 + Math.floor(rnd());
      const h = 1 + Math.floor(rnd());
      if (!k.insert(w, h)) break;
      placed.push({ x: k.x, y: k.y, w, h });
    }
    expect(placed.length).toBeGreaterThan(40);
    for (const r of placed) {
      expect(r.x).toBeGreaterThanOrEqual(0);
      expect(r.y).toBeGreaterThanOrEqual(0);
      expect(r.x + r.w).toBeLessThanOrEqual(256);
      expect(r.y + r.h).toBeLessThanOrEqual(256);
    }
    for (let i = 0; i < placed.length; i++)
      for (let j = i + 1; j < placed.length; j++) expect(overlaps(placed[i], placed[j], 1)).toBe(false);
  });

  it("puts each rectangle where its top is lowest, filling a row before starting the next", () => {
    const k = new Skyline(32, 32, 1);
    expect(k.insert(15, 8)).toBe(true);
    expect([k.x, k.y]).toEqual([0, 0]);
    expect(k.insert(15, 8)).toBe(true);
    expect([k.x, k.y]).toEqual([16, 0]);
    expect(k.insert(15, 8)).toBe(true);
    expect([k.x, k.y]).toEqual([0, 9]);
  });

  it("uses the page edge without padding and reports a full page", () => {
    const k = new Skyline(16, 16, 1);
    expect(k.insert(16, 16)).toBe(true);
    expect(k.insert(1, 1)).toBe(false);
    const k2 = new Skyline(16, 16, 1);
    expect(k2.insert(17, 1)).toBe(false);
  });

  it("merges segments of equal height", () => {
    const k = new Skyline(64, 64, 0);
    k.insert(32, 4);
    k.insert(32, 4);
    expect(k.segments).toBe(1);
  });
});

describe("atlas", () => {
  it("hands out frame indices in order, with the pivot and page position", () => {
    const a = new Atlas(64);
    const i = a.add("a", box(8, 8), 4, 8);
    const j = a.add("b", box(10, 5));
    expect([i, j]).toEqual([0, 1]);
    expect(a.frame(i)).toMatchObject({ page: 0, x: 0, y: 0, w: 8, h: 8, ox: 4, oy: 8 });
    expect(a.frame(j)).toMatchObject({ page: 0, w: 10, h: 5, ox: 0, oy: 0 });
    expect(a.index("b")).toBe(1);
    expect(a.index("nope")).toBe(-1);
    expect(a.pending).toEqual([0, 1]);
  });

  it("grows a new page when one is full", () => {
    const a = new Atlas(32);
    for (let n = 0; n < 4; n++) a.add(`q${n}`, box(15, 15));
    expect(a.pages).toBe(1);
    a.add("more", box(15, 15));
    expect(a.pages).toBe(2);
    expect(a.frame(4).page).toBe(1);
  });

  it("keeps an id's index when it is added again, moving it only if the size changed", () => {
    const a = new Atlas(64);
    const i = a.add("x", box(8, 8));
    const at = { ...a.frame(i) };
    const v = a.version;
    expect(a.add("x", box(8, 8), 1, 2)).toBe(i);
    expect(a.frame(i)).toMatchObject({ x: at.x, y: at.y, ox: 1, oy: 2 });
    expect(a.add("x", box(12, 8))).toBe(i);
    expect(a.frame(i).w).toBe(12);
    expect(a.version).toBeGreaterThan(v);
  });

  it("returns an empty frame for unknown indices and refuses canvases bigger than a page", () => {
    const a = new Atlas(64);
    expect(a.frame(99)).toMatchObject({ w: 0, h: 0 });
    expect(() => a.add("big", box(65, 1))).toThrow(/larger than a page/);
  });

  it("re-queues every frame after a lost context", () => {
    const a = new Atlas(64);
    a.add("a", box(2, 2));
    a.add("b", box(2, 2));
    a.pending.length = 0;
    a.invalidate();
    expect(a.pending).toEqual([0, 1]);
  });
});
