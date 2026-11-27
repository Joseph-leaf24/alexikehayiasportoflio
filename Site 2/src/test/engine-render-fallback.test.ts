import { afterEach, describe, expect, it, vi } from "vitest";
import { StubTerrain } from "@/engine/demo/stubTerrain";
import { createFrame, createRenderer, SPRITE_FLIP_X, SPRITE_SILHOUETTE } from "@/engine/render";
import { chunkOf, columnTop, packCells } from "@/engine/render/tiles";
import { chunkIndex as chunkOfCell } from "@/engine/terrain/constants";

const cv = (width: number, height: number) => Object.assign(document.createElement("canvas"), { width, height });

/** A Canvas 2D context that accepts every call, counting drawImage and fillRect. */
function fakeContext() {
  const calls = { drawImage: 0, fillRect: 0 };
  const target: Record<string, unknown> = {
    createLinearGradient: () => ({ addColorStop: () => {} }),
    drawImage: () => calls.drawImage++,
    fillRect: () => calls.fillRect++,
  };
  const ctx = new Proxy(target, {
    get: (t, k: string) => (k in t ? t[k] : () => {}),
    set: (t, k: string, v) => ((t[k] = v), true),
  });
  return { ctx, calls };
}

function mockContexts(webgl: unknown) {
  const { ctx, calls } = fakeContext();
  const asked: string[] = [];
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(((type: string) => {
    asked.push(type);
    return type === "2d" ? ctx : type === "webgl2" ? webgl : null;
  }) as unknown as typeof HTMLCanvasElement.prototype.getContext);
  return { calls, asked };
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("createRenderer", () => {
  it("falls back to Canvas 2D without WebGL2", () => {
    const { asked } = mockContexts(null);
    const r = createRenderer(document.createElement("canvas"), { pixelScale: 3 });
    expect(r.kind).toBe("canvas2d");
    expect(asked[0]).toBe("webgl2");
  });

  it("uses Canvas 2D straight away when asked to", () => {
    const { asked } = mockContexts(null);
    expect(createRenderer(document.createElement("canvas"), { pixelScale: 3, prefer: "canvas2d" }).kind).toBe(
      "canvas2d"
    );
    expect(asked).not.toContain("webgl2");
  });

  it("swaps in a fresh canvas when WebGL2 exists but fails to set up", () => {
    mockContexts({});
    const err = vi.spyOn(console, "error").mockImplementation(() => {});
    const canvas = document.createElement("canvas");
    canvas.id = "scene";
    document.body.appendChild(canvas);
    const r = createRenderer(canvas, { pixelScale: 3 });
    expect(r.kind).toBe("canvas2d");
    expect(err).toHaveBeenCalled();
    const now = document.getElementById("scene");
    expect(now).not.toBe(canvas);
    expect(now?.tagName).toBe("CANVAS");
  });

  it("throws when there is no canvas at all", () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => null);
    expect(() => createRenderer(document.createElement("canvas"), { pixelScale: 3 })).toThrow(/neither WebGL2/);
  });
});

describe("Canvas 2D renderer", () => {
  it("sizes the canvas to whole pixel-scale multiples and reports its internal size", () => {
    mockContexts(null);
    const canvas = document.createElement("canvas");
    const r = createRenderer(canvas, { pixelScale: 3 });
    r.resize(1000, 600, 1, 3);
    expect([r.width, r.height]).toEqual([334, 200]);
    expect([canvas.width, canvas.height]).toEqual([1002, 600]);
    expect(canvas.style.width).toBe("1002px");
  });

  it("draws a full frame: sky, terrain chunks, sprites, particles, text and effects", () => {
    const { calls } = mockContexts(null);
    const r = createRenderer(document.createElement("canvas"), { pixelScale: 3 });
    r.resize(480, 300, 1, 2);
    const terrain = new StubTerrain(3);
    for (let i = 0; i < terrain.tiles.length; i++) terrain.tiles[i] = r.atlas.add(`t${i}`, cv(8, 8));
    const unit = r.atlas.add("unit", cv(7, 10), 3, 10);
    const f = createFrame();
    f.terrain = terrain;
    f.camera.x = 1000;
    f.camera.y = terrain.surface(125) * 8;
    f.ambient = 0x404060;
    f.sky.stars = 1;
    f.effects.flash = 0.5;
    f.effects.white = 0.2;
    f.effects.letterbox = 1;
    for (let i = 0; i < 50; i++)
      f.sprites.push(unit, 900 + i * 4, f.camera.y, i % 16, i % 3 === 0 ? SPRITE_FLIP_X : 0, 0xffffffff, 1, 0);
    f.sprites.push(unit, 1000, f.camera.y, 8, SPRITE_SILHOUETTE, 0xff0000ff, 1, 0.3);
    for (let i = 0; i < 100; i++) f.particles.push(950 + i, f.camera.y - 10, 2, 0xffd23fff, 0);
    f.text.push("HELLO", 10, 10, 0xffffffff, 1, 0, 15);
    r.render(f);
    expect(r.stats.sprites).toBe(51);
    expect(r.stats.particles).toBe(100);
    expect(r.stats.chunks).toBeGreaterThan(0);
    expect(r.stats.drawCalls).toBeGreaterThan(50);
    expect(calls.fillRect).toBeGreaterThan(100);
    // A second frame reuses the cached chunks
    r.render(f);
    expect(r.stats.chunks).toBe(0);
    // Zoomed, with a fixed backdrop and daylight
    f.camera.zoom = 1.5;
    f.ambient = 0xffffff;
    f.parallax.push({ frame: unit, factor: 0, y: 0, tint: 0xffffff80, repeatX: false });
    f.parallax.push({ frame: unit, factor: 0.3, y: 40, tint: 0xffffffff, repeatX: true });
    f.effects.flash = f.effects.white = f.effects.letterbox = 0;
    f.sky.stars = 0;
    r.render(f);
    expect(r.stats.drawCalls).toBeGreaterThan(50);
    // Damage re-paints only the chunks it touched
    terrain.carve(1000, terrain.surface(125) * 8 + 16, 20, 50);
    r.render(f);
    expect(r.stats.chunks).toBeGreaterThan(0);
    r.destroy();
  });
});

describe("terrain upload helpers", () => {
  it("splits chunk indices column-major, as the terrain marks them", () => {
    const t = new StubTerrain(1);
    const xy = new Int32Array(2);
    // Column 37, row 200: chunk column 2, chunk row 12 of 15, numbered as the terrain numbers them
    expect(chunkOfCell(37, 200)).toBe(2 * 15 + 12);
    expect(Array.from(chunkOf(chunkOfCell(37, 200), t, xy))).toEqual([2, 12]);
    expect(Array.from(chunkOf(chunkOfCell(511 + 512, 239), t, xy))).toEqual([31, 14]);
  });

  it("repacks column-major cells into row-major RGBA texels", () => {
    const t = new StubTerrain(1);
    const i = 5 * t.rows + 7;
    t.fg[i] = 9;
    t.bg[i] = 8;
    t.variant[i] = 3;
    t.hp[i] = 77;
    const out = packCells(t, 4, 6, 6, 9, new Uint8Array(2 * 3 * 4));
    // Column 5 is x 1 of the 2-wide span; row 7 is y 1 of the 3-high span
    const o = (1 * 2 + 1) * 4;
    expect(Array.from(out.slice(o, o + 4))).toEqual([9, 8, 3, 77]);
  });

  it("finds the first non-air row of a column", () => {
    const t = new StubTerrain(1);
    const c = 10;
    t.fg.fill(0, c * t.rows, (c + 1) * t.rows);
    expect(columnTop(t, c)).toBe(t.rows);
    t.fg[c * t.rows + 42] = 1;
    expect(columnTop(t, c)).toBe(42);
  });
});
