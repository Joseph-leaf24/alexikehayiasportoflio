import { describe, expect, it } from "vitest";
import { LightList, ParticleList, SpriteList, TextList } from "@/engine/render/buffers";
import { clearFrame, createFrame, PARTICLE_LAYER, TERRAIN_LAYER } from "@/engine/render/frame";
import { buildRuns, countingSort } from "@/engine/render/sort";

describe("counting sort", () => {
  it("orders by key and keeps push order within a key (stable)", () => {
    const n = 500;
    const keys = new Uint16Array(n);
    let seed = 7;
    for (let i = 0; i < n; i++) keys[i] = (seed = (seed * 48271) % 2147483647) % 12;
    const counts = new Uint32Array(13);
    const order = new Uint32Array(n);
    countingSort(keys, n, 12, counts, order);
    const expected = Array.from({ length: n }, (_, i) => i).sort((a, b) => keys[a] - keys[b] || a - b);
    expect(Array.from(order)).toEqual(expected);
    // counts[k]..counts[k + 1] spans bucket k
    for (let k = 0; k < 12; k++) for (let p = counts[k]; p < counts[k + 1]; p++) expect(keys[order[p]]).toBe(k);
    expect(counts[12]).toBe(n);
  });

  it("turns buckets into one run per non-empty (layer, page) key", () => {
    const keys = Uint16Array.from([3, 0, 3, 5, 0]);
    const counts = new Uint32Array(7);
    const order = new Uint32Array(5);
    countingSort(keys, 5, 6, counts, order);
    const runKey = new Uint16Array(6);
    const runStart = new Uint32Array(6);
    const runCount = new Uint32Array(6);
    const runs = buildRuns(counts, 6, runKey, runStart, runCount);
    expect(runs).toBe(3);
    expect(Array.from(runKey.slice(0, 3))).toEqual([0, 3, 5]);
    expect(Array.from(runStart.slice(0, 3))).toEqual([0, 2, 4]);
    expect(Array.from(runCount.slice(0, 3))).toEqual([2, 2, 1]);
  });

  it("handles an empty list", () => {
    const counts = new Uint32Array(5);
    countingSort(new Uint16Array(0), 0, 4, counts, new Uint32Array(0));
    expect(buildRuns(counts, 4, new Uint16Array(4), new Uint32Array(4), new Uint32Array(4))).toBe(0);
  });
});

describe("frame buffers", () => {
  it("sprite list stores its fields, clamps layers, keeps tints unsigned and grows past capacity", () => {
    const s = new SpriteList(16);
    for (let i = 0; i < 40; i++) s.push(i, i * 2, -i, i === 0 ? -3 : 20, 1, 0xff0000ff, 2, 0.5);
    expect(s.count).toBe(40);
    expect(s.capacity).toBeGreaterThanOrEqual(40);
    expect(s.frame[39]).toBe(39);
    expect(s.x[39]).toBe(78);
    expect(s.y[39]).toBe(-39);
    expect(s.layer[0]).toBe(0);
    expect(s.layer[1]).toBe(15);
    expect(s.tint[5]).toBe(0xff0000ff);
    expect(s.scale[5]).toBe(2);
    expect(s.rot[5]).toBeCloseTo(0.5);
    s.clear();
    expect(s.count).toBe(0);
    const cap = s.capacity;
    for (let i = 0; i < 40; i++) s.push(0, 0, 0, 0, 0, 0, 1, 0);
    // A steady load reuses the same arrays
    expect(s.capacity).toBe(cap);
  });

  it("particle, light and text lists grow and clear", () => {
    const p = new ParticleList(16);
    for (let i = 0; i < 33; i++) p.push(i, 1, i === 0 ? 0 : 3, 0xffffffff, 2);
    expect(p.count).toBe(33);
    expect(p.size[0]).toBe(1);
    expect(p.color[32]).toBe(0xffffffff);
    const l = new LightList(16);
    for (let i = 0; i < 20; i++) l.push(i, 2, 30, 0xffb060, 1.5);
    expect(l.count).toBe(20);
    expect(l.intensity[19]).toBeCloseTo(1.5);
    const t = new TextList(16);
    for (let i = 0; i < 18; i++) t.push(`T${i}`, 1, 2, 0xffffffff, 2.4, 1, 30);
    expect(t.count).toBe(18);
    expect(t.text[17]).toBe("T17");
    expect(t.scale[0]).toBe(2);
    expect(t.layer[0]).toBe(15);
    t.clear();
    expect(t.count).toBe(0);
    expect(t.text[0]).toBe("");
  });

  it("createFrame gives neutral settings and clearFrame empties every buffer", () => {
    const f = createFrame();
    expect(f.ambient).toBe(0xffffff);
    expect(f.camera.zoom).toBe(1);
    expect(f.effects.grade).toEqual({ exposure: 1, contrast: 1, saturation: 1, tint: 0xffffff });
    expect([f.effects.flash, f.effects.white, f.effects.bloom, f.effects.chroma]).toEqual([0, 0, 0, 0]);
    expect(f.terrainLayer).toBe(TERRAIN_LAYER);
    expect(f.particleLayer).toBe(PARTICLE_LAYER);
    f.sprites.push(0, 0, 0, 0, 0, 0xffffffff, 1, 0);
    f.particles.push(0, 0, 1, 0xffffffff, 0);
    f.lights.push(0, 0, 10, 0xffffff, 1);
    f.overlay.push(0, 0, 0, 0, 0, 0xffffffff, 1, 0);
    f.text.push("HI", 0, 0, 0xffffffff, 1, 0, 0);
    clearFrame(f);
    expect([f.sprites.count, f.particles.count, f.lights.count, f.overlay.count]).toEqual([0, 0, 0, 0]);
    expect((f.text as TextList).count).toBe(0);
  });
});
