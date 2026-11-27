import { describe, expect, it } from "vitest";
import { Rng, hash2, hashString, streams } from "@/engine/core";

const draw = (r: Rng, n: number) => Array.from({ length: n }, () => r.u32());

describe("rng", () => {
  it("is deterministic per seed and differs across seeds and salts", () => {
    expect(draw(new Rng(42), 50)).toEqual(draw(new Rng(42), 50));
    expect(draw(new Rng(42), 50)).not.toEqual(draw(new Rng(43), 50));
    expect(draw(new Rng(42, 1), 50)).not.toEqual(draw(new Rng(42, 2), 50));
    // Nearby seeds decorrelate immediately
    expect(new Rng(1).u32()).not.toBe(new Rng(2).u32());
  });

  it("next() is uniform in [0, 1)", () => {
    const r = new Rng(7);
    const bins = new Array(10).fill(0);
    let min = 1,
      max = 0;
    for (let k = 0; k < 100_000; k++) {
      const v = r.next();
      if (v < min) min = v;
      if (v > max) max = v;
      bins[Math.floor(v * 10)]++;
    }
    expect(min).toBeGreaterThanOrEqual(0);
    expect(max).toBeLessThan(1);
    for (const b of bins) expect(Math.abs(b - 10_000)).toBeLessThan(500);
  });

  it("helpers stay in range", () => {
    const r = new Rng(3);
    for (let k = 0; k < 2000; k++) {
      const f = r.range(-5, 5);
      expect(f >= -5 && f < 5).toBe(true);
      const i = r.int(6);
      expect(Number.isInteger(i) && i >= 0 && i < 6).toBe(true);
      const j = r.int(-3, 3);
      expect(Number.isInteger(j) && j >= -3 && j < 3).toBe(true);
      expect([-1, 1]).toContain(r.sign());
      const a = r.angle();
      expect(a >= 0 && a < Math.PI * 2).toBe(true);
    }
    expect(r.chance(0)).toBe(false);
    expect(r.chance(1)).toBe(true);
    expect(["a", "b", "c"]).toContain(r.pick(["a", "b", "c"]));
  });

  it("weighted picks follow the weights and skip zero and negative ones", () => {
    const r = new Rng(11);
    const counts = [0, 0, 0, 0];
    for (let k = 0; k < 40_000; k++) counts[r.weighted([1, 0, 3, -2])]++;
    expect(counts[1]).toBe(0);
    expect(counts[3]).toBe(0);
    expect(counts[2] / counts[0]).toBeGreaterThan(2.7);
    expect(counts[2] / counts[0]).toBeLessThan(3.3);
    expect(r.weighted([0, 0])).toBe(-1);
    expect(r.weighted([])).toBe(-1);
    const items = [
      { id: "a", w: 0 },
      { id: "b", w: 2 },
    ];
    for (let k = 0; k < 20; k++) expect(r.pickWeighted(items, (i) => i.w)?.id).toBe("b");
    expect(r.pickWeighted(items, () => 0)).toBeUndefined();
  });

  it("shuffle permutes in place, typed arrays too", () => {
    const r = new Rng(5);
    const a = Array.from({ length: 30 }, (_, i) => i);
    const s = r.shuffle([...a]);
    expect(s).not.toEqual(a);
    expect([...s].sort((x, y) => x - y)).toEqual(a);
    const t = r.shuffle(Int32Array.from(a));
    expect(Array.from(t).sort((x, y) => x - y)).toEqual(a);
  });

  it("gauss has the right mean and spread", () => {
    const r = new Rng(9);
    let sum = 0,
      sq = 0;
    const n = 50_000;
    for (let k = 0; k < n; k++) {
      const v = r.gauss(10, 2);
      sum += v;
      sq += v * v;
    }
    const mean = sum / n;
    expect(mean).toBeCloseTo(10, 1);
    expect(Math.sqrt(sq / n - mean * mean)).toBeCloseTo(2, 1);
  });

  it("save, load and clone continue the same sequence", () => {
    const r = new Rng(77);
    draw(r, 10);
    const state = new Int32Array(4);
    r.save(state);
    const c = r.clone();
    const next = draw(r, 20);
    expect(draw(c, 20)).toEqual(next);
    expect(draw(new Rng().load(state), 20)).toEqual(next);
  });
});

describe("streams", () => {
  it("are independent: draws on one never shift another", () => {
    const a = streams(1234);
    const b = streams(1234);
    draw(a.get("fx"), 1000);
    expect(draw(a.get("ai"), 20)).toEqual(draw(b.get("ai"), 20));
    expect(draw(a.get("ai"), 5)).not.toEqual(draw(a.get("spawn"), 5));
    expect(a.get("ai")).toBe(a.get("ai"));
    expect(a.names()).toEqual(["ai", "fx", "spawn"]);
  });

  it("hash tracks state regardless of creation order; reset rewinds", () => {
    const a = streams(5),
      b = streams(5);
    a.get("x");
    a.get("y");
    b.get("y");
    b.get("x");
    expect(a.hash()).toBe(b.hash());
    a.get("x").next();
    expect(a.hash()).not.toBe(b.hash());
    a.reset();
    expect(a.hash()).toBe(b.hash());
  });

  it("hashes are stable and spread", () => {
    expect(hashString("ai")).toBe(hashString("ai"));
    expect(hashString("ai")).not.toBe(hashString("ia"));
    const v = hash2(3, 4, 1);
    expect(v).toBe(hash2(3, 4, 1));
    expect(v >= 0 && v < 1).toBe(true);
    expect(hash2(3, 4, 1)).not.toBe(hash2(4, 3, 1));
  });
});
