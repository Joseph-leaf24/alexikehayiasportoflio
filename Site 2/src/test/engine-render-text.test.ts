import { describe, expect, it } from "vitest";
import { SpriteList } from "@/engine/render/buffers";
import { DebugOverlay, debugFromUrl, type DebugWorldStats } from "@/engine/render/debug";
import { createFrame } from "@/engine/render/frame";
import { GLYPH_OF, GLYPHS, INK, layoutText, measureText, outlineBits, type FontFrames } from "@/engine/render/font";
import type { TextList } from "@/engine/render/buffers";
import type { Renderer } from "@/engine/render/types";
import { WorldStats } from "@/engine/core/stats";

/** Glyph n's frame is 100 + n and its outline's 200 + n, so tests can tell them apart. */
const font: FontFrames = {
  glyph: Int32Array.from(Object.keys(GLYPHS), (_, i) => 100 + i),
  outline: Int32Array.from(Object.keys(GLYPHS), (_, i) => 200 + i),
};

describe("pixel font", () => {
  it("keeps the old engine's 3×5 glyphs", () => {
    expect(GLYPHS.A).toBe("010101111101101");
    expect(GLYPHS["0"]).toBe("111101101101111");
    expect(GLYPHS.W).toBe("101101111111101");
    expect(GLYPHS["?"]).toBe("111001010000010");
    for (const bits of Object.values(GLYPHS)) expect(bits).toMatch(/^[01]{15}$/);
  });

  it("maps lower case to upper case and draws nothing for spaces and unknown characters", () => {
    expect(GLYPH_OF["a".charCodeAt(0)]).toBe(GLYPH_OF["A".charCodeAt(0)]);
    expect(GLYPH_OF[" ".charCodeAt(0)]).toBe(-1);
    expect(GLYPH_OF["~".charCodeAt(0)]).toBe(-1);
    expect(GLYPH_OF["·".charCodeAt(0)]).toBeGreaterThanOrEqual(0);
  });

  it("measures 4 px per character less the last gap, times the scale", () => {
    expect(measureText("ABC")).toBe(11);
    expect(measureText("ABC", 2)).toBe(22);
    expect(measureText("")).toBe(0);
  });

  it("builds the outline from the glyph moved left, right, up, down and diagonally down", () => {
    const m = outlineBits(GLYPHS["."]);
    // "." is the single pixel (1, 4); in the 5×7 outline cell that pixel sits at (2, 5)
    const set = Array.from(m).flatMap((v, k) => (v ? [[k % 5, (k / 5) | 0]] : []));
    expect(set).toEqual(
      expect.arrayContaining([
        [1, 5],
        [3, 5],
        [2, 4],
        [2, 6],
        [3, 6],
      ])
    );
    expect(set).toHaveLength(5);
  });

  it("lays text out as outlines first, then glyphs in the text colour", () => {
    const out = new SpriteList(16);
    const n = layoutText(font, "Hi!", 10, 20, 0xff0000ff, 1, 0, 7, out);
    expect(n).toBe(6);
    const g = (ch: string) => GLYPH_OF[ch.charCodeAt(0)];
    expect(Array.from(out.frame.slice(0, 6))).toEqual([
      200 + g("H"),
      200 + g("I"),
      200 + g("!"),
      100 + g("H"),
      100 + g("I"),
      100 + g("!"),
    ]);
    expect(Array.from(out.x.slice(3, 6))).toEqual([10, 14, 18]);
    expect(out.y[0]).toBe(20);
    expect(out.tint[0]).toBe(INK);
    expect(out.tint[3]).toBe(0xff0000ff);
    expect(out.layer[0]).toBe(7);
  });

  it("aligns centre and right, scales the advance, and skips spaces but keeps their width", () => {
    const out = new SpriteList(16);
    layoutText(font, "A B", 100, 0, 0xffffff80, 2, 1, 0, out);
    // Width (3 × 4 − 1) × 2 = 22, so centred at 100 it starts at 89
    expect(out.count).toBe(4);
    expect(out.x[2]).toBe(89);
    expect(out.x[3]).toBe(89 + 16);
    expect(out.scale[2]).toBe(2);
    // The outline takes the text's alpha
    expect(out.tint[0] & 255).toBe(0x80);
    out.clear();
    layoutText(font, "AB", 100, 0, 0xffffffff, 1, 2, 0, out);
    expect(out.x[2]).toBe(93);
    out.clear();
    expect(layoutText(font, "AB", 0, 0, 0xffffff00, 1, 0, 0, out)).toBe(0);
  });
});

describe("debug overlay", () => {
  const renderer = {
    kind: "webgl2",
    width: 480,
    height: 300,
    stats: { drawCalls: 18, sprites: 2000, particles: 8000, lights: 64, cpuMs: 0.4, chunks: 2, gpuMs: 0.3 },
  } as unknown as Renderer;

  it("is off unless the page has ?debug=1", () => {
    expect(debugFromUrl()).toBe(false);
    const f = createFrame();
    new DebugOverlay().draw(f, renderer);
    expect((f.text as TextList).count).toBe(0);
  });

  it("shows fps, frame ms, renderer counters, draw calls and per-system ms", () => {
    const d = new DebugOverlay(true);
    const f = createFrame();
    d.draw(f, renderer, undefined, 0);
    for (let i = 0; i < 30; i++) d.tick(16.7);
    f.text.clear();
    d.draw(f, renderer, { systems: [{ name: "ai", ms: 0.25, max: 0.5 }], tps: 30, entities: 400 }, 500);
    const text = f.text as TextList;
    const lines = text.text.slice(0, text.count);
    const joined = lines.join("\n");
    expect(joined).toMatch(/FPS 60/);
    expect(joined).toMatch(/FRAME 16\.7MS/);
    expect(joined).toMatch(/DRAWS 18 {2}SPRITES 2000 {2}PARTS 8000 {2}LIGHTS 64/);
    expect(joined).toMatch(/GPU 0\.30MS/);
    expect(joined).toMatch(/TPS 30 {2}ENTITIES 400/);
    expect(joined).toMatch(/ai 0\.25 \/ 0\.50/);
  });

  it("takes the core's world.stats as it is", () => {
    const stats: DebugWorldStats = new WorldStats();
    const f = createFrame();
    new DebugOverlay(true).draw(f, renderer, stats, 0);
    const text = f.text as TextList;
    expect(text.text.slice(0, text.count)).toContainEqual(expect.stringMatching(/TPS 0 {2}ENTITIES 0/));
  });
});
