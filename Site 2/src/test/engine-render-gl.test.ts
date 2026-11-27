import { describe, expect, it, vi } from "vitest";
import { StubTerrain } from "@/engine/demo/stubTerrain";
import { Atlas } from "@/engine/render/atlas";
import { SpriteList } from "@/engine/render/buffers";
import { createFrame } from "@/engine/render/frame";
import { WebGLRenderer } from "@/engine/render/gl/renderer";
import { SPRITE_STRIDE, SpriteBatch } from "@/engine/render/gl/sprites";
import { SPRITE_FLIP_X, SPRITE_SILHOUETTE } from "@/engine/render/types";

type Call = { name: string; args: unknown[] };

/**
 * A WebGL2 stand-in that records every call: upper-case names are distinct enum numbers, creators hand back
 * objects, status queries say yes, and everything else does nothing. Enough to drive the passes in jsdom.
 */
function mockGl(timer = false) {
  const calls: Call[] = [];
  const enums = new Map<string, number>();
  const num = (k: string) => enums.get(k) ?? enums.set(k, 0x1000 + enums.size).get(k)!;
  const special: Record<string, (...a: unknown[]) => unknown> = {
    getShaderParameter: () => true,
    getProgramParameter: () => true,
    checkFramebufferStatus: () => num("FRAMEBUFFER_COMPLETE"),
    getError: () => num("NO_ERROR"),
    isContextLost: () => false,
    getExtension: (name) =>
      timer && name === "EXT_disjoint_timer_query_webgl2" ? { TIME_ELAPSED_EXT: 1, GPU_DISJOINT_EXT: 2 } : null,
    getQueryParameter: (_q, p) => (p === num("QUERY_RESULT_AVAILABLE") ? true : 2e6),
    getParameter: () => false,
    getUniformLocation: () => ({}),
  };
  const gl = new Proxy(
    {},
    {
      get: (_t, k: string) => {
        if (/^[A-Z0-9_]+$/.test(k)) return num(k);
        return (...args: unknown[]) => {
          calls.push({ name: k, args });
          if (special[k]) return special[k](...args);
          return k.startsWith("create") ? {} : undefined;
        };
      },
    }
  ) as unknown as WebGL2RenderingContext;
  return { gl, calls, named: (n: string) => calls.filter((c) => c.name === n) };
}

const box = (width: number, height: number) => ({ width, height }) as unknown as HTMLCanvasElement;

describe("sprite batch", () => {
  it("streams instances sorted by layer then page, packed as the shader reads them", () => {
    const { gl, named } = mockGl();
    const atlas = new Atlas(64);
    const a = atlas.add("a", box(8, 10), 3, 10);
    const b = atlas.add("b", box(60, 60), 1, 2); // forces a second page
    expect(atlas.frame(b).page).toBe(1);
    const list = new SpriteList(16);
    list.push(b, 110, 210, 5, SPRITE_SILHOUETTE, 0x11223344, 2, 0.5);
    list.push(a, 100.4, 200.6, 5, SPRITE_FLIP_X, 0xff8000ff, 1, 0);
    list.push(a, 50, 60, 2, 0, 0xffffffff, 1, 0);
    list.push(-1, 0, 0, 0, 0, 0, 1, 0);
    const batch = new SpriteBatch(gl, 64);
    batch.prepare(list, atlas, 100, 200);
    expect(batch.count).toBe(3);
    const up = named("bufferSubData").at(-1)!;
    expect(up.args[4]).toBe(3 * SPRITE_STRIDE);
    const bytes = up.args[2] as Uint8Array;
    const f32 = new Float32Array(bytes.buffer);
    const u16 = new Uint16Array(bytes.buffer);
    const i16 = new Int16Array(bytes.buffer);
    const u32 = new Uint32Array(bytes.buffer);
    const W = SPRITE_STRIDE / 4;
    // Order: layer 2 (a), then layer 5 page 0 (a), then layer 5 page 1 (b)
    expect([f32[0], f32[1]]).toEqual([-50, -140]);
    expect(f32[W]).toBeCloseTo(0.4, 4);
    expect(f32[W + 1]).toBeCloseTo(0.6, 4);
    const fa = atlas.frame(a);
    expect(Array.from(u16.slice(W * 2 + 4, W * 2 + 8))).toEqual([fa.x, fa.y, 8, 10]);
    expect([i16[W * 2 + 8], i16[W * 2 + 9]]).toEqual([3, 10]);
    expect(u32[W + 5]).toBe(0xff8000ff);
    expect(bytes[(W + 8) * 4]).toBe(SPRITE_FLIP_X);
    const k = 2 * W;
    expect([f32[k], f32[k + 1], f32[k + 6], f32[k + 7]]).toEqual([10, 10, 2, 0.5]);
    expect(bytes[(k + 8) * 4]).toBe(SPRITE_SILHOUETTE);
    // The tint's bytes in memory are A, B, G, R: the shader swizzles them back (wzyx)
    expect(Array.from(bytes.slice((k + 5) * 4, (k + 5) * 4 + 4))).toEqual([0x44, 0x33, 0x22, 0x11]);
  });

  it("draws one instanced call per (layer, page) run, re-pointing the attributes at each slice", () => {
    const { gl, named } = mockGl();
    const atlas = new Atlas(64);
    const a = atlas.add("a", box(8, 8));
    const list = new SpriteList(16);
    for (let i = 0; i < 5; i++) list.push(a, i, 0, i % 2 ? 3 : 1, 0, 0xffffffff, 1, 0);
    const batch = new SpriteBatch(gl, 64);
    batch.prepare(list, atlas, 0, 0);
    const prog = { prog: {} as WebGLProgram, u: {} };
    const tex = { textures: [{}] } as never;
    expect(batch.draw(prog, tex, 0, 15, 64)).toBe(2);
    const draws = named("drawArraysInstanced");
    expect(draws.map((d) => d.args[3])).toEqual([3, 2]);
    const posPointers = named("vertexAttribPointer").filter((c) => c.args[0] === 0);
    expect(posPointers.at(-1)!.args[5]).toBe(3 * SPRITE_STRIDE);
    // Layer filtering: only layer 3
    expect(batch.draw(prog, tex, 2, 15, 64)).toBe(1);
  });

  it("culls sprites that can't reach the view and drops any past its capacity", () => {
    const { gl } = mockGl();
    const atlas = new Atlas(64);
    const a = atlas.add("a", box(8, 8));
    const list = new SpriteList(16);
    list.push(a, 50, 50, 0, 0, 0xffffffff, 1, 0);
    list.push(a, -500, 50, 0, 0, 0xffffffff, 1, 0);
    list.push(a, 50, 900, 0, 0, 0xffffffff, 1, 0);
    const batch = new SpriteBatch(gl, 2);
    batch.prepare(list, atlas, 0, 0, 100, 100);
    expect(batch.dropped).toBe(1);
    expect(batch.count).toBe(1);
  });
});

describe("WebGL2 renderer passes", () => {
  it("draws a full frame through every pass with a fixed handful of draw calls, then blits", () => {
    const { ctxCalls } = mock2d();
    const { gl, named } = mockGl();
    const canvas = document.createElement("canvas");
    const r = new WebGLRenderer(canvas, gl, { pixelScale: 3 });
    expect(ctxCalls()).toBeGreaterThan(0);
    r.resize(1440, 900, 1, 3);
    expect([r.width, r.height]).toEqual([480, 300]);
    const terrain = new StubTerrain(2);
    for (let i = 0; i < terrain.tiles.length; i++) terrain.tiles[i] = r.atlas.add(`t${i}`, box(8, 8));
    const unit = r.atlas.add("unit", box(7, 10), 3, 10);
    const f = createFrame();
    f.terrain = terrain;
    f.camera.x = 1000;
    f.camera.y = 900;
    for (let i = 0; i < 2000; i++) f.sprites.push(unit, 800 + (i % 400), 850 + (i % 90), i % 16, 0, 0xffffffff, 1, 0);
    for (let i = 0; i < 8000; i++) f.particles.push(800 + (i % 400), 800 + (i % 200), 1 + (i % 3), 0xffd23fff, i & 6);
    for (let i = 0; i < 64; i++) f.lights.push(800 + i * 6, 900, 40, 0xffb060, 1);
    f.effects.bloom = 0.8;
    f.effects.chroma = 0.5;
    f.effects.vignette = 0.3;
    f.effects.shockwaves.push({ x: 1000, y: 900, radius: 40, width: 12, strength: 6 });
    f.text.push("FPS 60", 2, 2, 0xffffffff, 1, 0, 15);
    // At factor 0.2 the camera's y of 900 lifts the layer by 180: y 180 puts its pivot mid-view
    f.parallax.push({ frame: unit, factor: 0.2, y: 180, tint: 0xffffffff, repeatX: true });
    r.render(f);
    expect(r.stats.sprites).toBe(2000);
    expect(r.stats.particles).toBe(8000);
    expect(r.stats.lights).toBe(64);
    // The first frame uploads the whole terrain ring: 32 chunk columns × 15 chunk rows
    expect(r.stats.chunks).toBe(480);
    // sky + parallax + 16 sprite layers + terrain + particles + lights + bloom (3) + post + overlay
    expect(r.stats.drawCalls).toBe(1 + 1 + 16 + 1 + 1 + 1 + 3 + 1 + 1);
    expect(named("blitFramebuffer")).toHaveLength(1);
    expect(named("blitFramebuffer")[0].args.slice(4, 8)).toEqual([0, 0, 1440, 900]);
    // Steady state: nothing re-uploads
    r.render(f);
    expect(r.stats.chunks).toBe(0);
    expect(terrain.carve(1000, terrain.surface(125) * 8 + 16, 20, 50)).toBeGreaterThan(0);
    r.render(f);
    expect(r.stats.chunks).toBeGreaterThan(0);
    r.destroy();
  });

  it("times the GPU with a timer query, and at low quality skips bloom", () => {
    mock2d();
    const { gl, named } = mockGl(true);
    const r = new WebGLRenderer(document.createElement("canvas"), gl, {
      pixelScale: 3,
      quality: "low",
      gpuTiming: true,
    });
    r.resize(300, 200, 2, 3);
    const f = createFrame();
    f.effects.bloom = 1;
    f.effects.letterbox = 1;
    f.effects.grade.exposure = 1.2;
    f.camera.zoom = 1.5;
    r.render(f);
    r.render(f);
    expect(named("beginQuery").length).toBeGreaterThanOrEqual(2);
    expect(r.stats.gpuMs).toBeCloseTo(2);
    // Sky and post only: no terrain, sprites, particles, lights, bloom or overlay
    expect(r.stats.drawCalls).toBe(2);
    r.destroy();
  });

  it("stops drawing while the context is lost and rebuilds everything when it comes back", () => {
    mock2d();
    const { gl, named } = mockGl();
    const canvas = document.createElement("canvas");
    const r = new WebGLRenderer(canvas, gl, { pixelScale: 3 });
    r.resize(300, 200, 1, 3);
    const f = createFrame();
    const lost = new Event("webglcontextlost", { cancelable: true });
    canvas.dispatchEvent(lost);
    expect(lost.defaultPrevented).toBe(true);
    const before = named("drawArrays").length;
    r.render(f);
    expect(r.stats.drawCalls).toBe(0);
    expect(named("drawArrays").length).toBe(before);
    const programs = named("createProgram").length;
    canvas.dispatchEvent(new Event("webglcontextrestored"));
    expect(named("createProgram").length).toBeGreaterThan(programs);
    r.render(f);
    expect(r.stats.drawCalls).toBeGreaterThan(0);
    r.destroy();
  });
});

/** A 2D context for the font painter (jsdom has none). */
function mock2d() {
  let n = 0;
  const ctx = new Proxy({} as Record<string, unknown>, {
    get: (t, k: string) => (k in t ? t[k] : () => void n++),
    set: (t, k: string, v) => ((t[k] = v), true),
  });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(((type: string) =>
    type === "2d" ? ctx : null) as unknown as typeof HTMLCanvasElement.prototype.getContext);
  return { ctxCalls: () => n };
}
