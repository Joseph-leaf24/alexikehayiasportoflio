import { describe, expect, it } from "vitest";
import { createCarveResult } from "@/engine/terrain/carve";
import { CHUNK_ROWS, F_BURNING, F_SCORCHED, MASK, ROWS, TILE } from "@/engine/terrain/constants";
import {
  AIR,
  BEDROCK,
  BRICK,
  DEBRIS,
  DIRT,
  LAVA,
  RUBBLE,
  SAND,
  STEEL,
  STONE,
  WATER,
  WOOD,
} from "@/engine/terrain/materials";
import { count, fill, flatWorld, GROUND, settle } from "./engine-terrain-fixtures";

const X = 100 * TILE + 4;
const Y = (GROUND + 6) * TILE + 4;

describe("carve", () => {
  it("clears breakable cells within the radius and leaves bedrock, water and lava alone", () => {
    const t = flatWorld();
    fill(t, 96, GROUND + 4, 9, 1, BEDROCK);
    t.set(100, GROUND + 7, WATER);
    t.set(101, GROUND + 7, LAVA);
    const out = t.carve(X, Y, 24, 255);
    expect(out.destroyed).toBeGreaterThan(20);
    expect(t.get(100, GROUND + 6)).toBe(AIR);
    expect(t.get(100, GROUND + 4)).toBe(BEDROCK);
    expect(t.get(100, GROUND + 7)).toBe(WATER);
    expect(t.get(101, GROUND + 7)).toBe(LAVA);
    // Only cells whose centre is in range
    for (let c = 90; c < 111; c++)
      for (let r = GROUND; r < GROUND + 13; r++) {
        const d = Math.hypot(c * TILE + 4 - X, r * TILE + 4 - Y);
        if (d > 24) expect(t.get(c, r), `${c},${r}`).not.toBe(AIR);
      }
  });

  it("falls off with distance: soft ground goes at once, stone needs a big or repeated blast", () => {
    const t = flatWorld();
    const out = t.carve(X, Y, 40, 3);
    // Centre stone (hp 4) took 3 hits and held; nearer the edge less
    const centre = t.hp[t.index(100, GROUND + 6)];
    const edge = t.hp[t.index(104, GROUND + 6)];
    expect(centre).toBeLessThan(edge);
    expect(out.damaged).toBeGreaterThan(0);
    for (let k = 0; k < 4; k++) t.carve(X, Y, 40, 3);
    expect(t.get(100, GROUND + 6)).toBe(AIR);
    // The same blast in the dirt (hp 2) breaks it outright
    const u = flatWorld();
    u.carve(X, (GROUND + 1) * TILE + 4, 40, 3);
    expect(u.get(100, GROUND + 1)).toBe(AIR);
    expect(u.get(100, GROUND + 5)).toBe(STONE);
  });

  it("is resisted by hard materials and by reinforced cells", () => {
    const t = flatWorld();
    fill(t, 40, 20, 1, 1, STONE, { hp: 200 });
    fill(t, 60, 20, 1, 1, STEEL, { hp: 200, structural: false });
    fill(t, 80, 20, 1, 1, STONE, { hp: 200, reinforced: true });
    for (const c of [40, 60, 80]) t.carve(c * TILE + 4, 20 * TILE + 4, 16, 60);
    const taken = (c: number) => 200 - t.hp[t.index(c, 20)];
    expect(taken(60)).toBeLessThan(taken(40));
    expect(taken(80)).toBeLessThan(taken(40));
    expect(taken(80)).toBeGreaterThan(0);
  });

  it("reports debris colours, counts and sample points for particles", () => {
    const t = flatWorld();
    const out = createCarveResult();
    t.carve(X, GROUND * TILE + 4, 44, 255, out);
    expect(out.kinds).toBeGreaterThanOrEqual(2);
    const colors = [...out.colors.slice(0, out.kinds)];
    expect(colors).toContain(DEBRIS[DIRT]);
    expect(colors).toContain(DEBRIS[STONE]);
    expect([...out.counts.slice(0, out.kinds)].reduce((a, b) => a + b, 0)).toBe(out.destroyed);
    expect(out.samples).toBe(Math.min(32, out.destroyed));
    for (let k = 0; k < out.samples; k++) {
      expect(Math.hypot(out.sx[k] - X, out.sy[k] - (GROUND * TILE + 4))).toBeLessThanOrEqual(44);
      expect([DIRT, STONE]).toContain(out.smat[k]);
    }
    // The shared result is reused
    expect(t.carve(X, 10, 4, 255)).toBe(t.carve(X, 10, 4, 255));
  });

  it("leaves rubble from masonry, which then falls", () => {
    const t = flatWorld();
    fill(t, 60, GROUND - 30, 30, 30, BRICK, { structural: false });
    t.carve(75 * TILE, (GROUND - 15) * TILE, 60, 255);
    expect(count(t, RUBBLE)).toBeGreaterThan(10);
    settle(t);
    for (let c = 55; c < 95; c++)
      for (let r = 0; r < GROUND; r++) if (t.get(c, r) === RUBBLE) expect(t.get(c, r + 1)).not.toBe(AIR);
  });

  it("scorches what it doesn't break and wakes the chunks around it", () => {
    const t = flatWorld();
    settle(t);
    expect(t.awakeChunks()).toBe(0);
    t.carve(X, Y, 16, 255);
    expect(t.awakeChunks()).toBeGreaterThan(0);
    let scorched = 0;
    for (let c = 95; c < 106; c++)
      for (let r = GROUND; r < GROUND + 13; r++) if (t.flags[t.index(c, r)] & F_SCORCHED) scorched++;
    expect(scorched).toBeGreaterThan(0);
    // The renderer is told
    const out: number[] = [];
    t.drainDirty(out);
    expect(out).toContain(((100 & MASK) >> 4) * CHUNK_ROWS + ((GROUND + 6) >> 4));
  });

  it("ignores empty blasts and the edges of the world", () => {
    const t = flatWorld();
    expect(t.carve(X, -500, 20, 255).destroyed).toBe(0);
    expect(t.carve(X, Y, 0, 255).destroyed).toBe(0);
    expect(t.carve(-1000, Y, 20, 255).destroyed).toBe(0);
    // At the floor of the world it breaks the stone but never the bedrock
    expect(t.carve(X, ROWS * TILE + 2, 30, 255).destroyed).toBeGreaterThan(0);
    for (let c = 95; c < 106; c++) expect(t.get(c, ROWS - 1)).toBe(BEDROCK);
  });
});

describe("single hits, fire and floods", () => {
  it("damage knocks hits off one cell and reports what broke", () => {
    const t = flatWorld();
    expect(t.damage(10, GROUND + 6, 1)).toBe(0);
    expect(t.hp[t.index(10, GROUND + 6)]).toBe(3);
    expect(t.damage(10, GROUND + 6, 3)).toBe(STONE);
    expect(t.get(10, GROUND + 6)).toBe(AIR);
    expect(t.damage(10, ROWS - 1, 99)).toBe(0);
    expect(t.damage(10, 5, 9)).toBe(0);
    expect(t.damage(-4, GROUND, 9)).toBe(0);
  });

  it("ignite lights only flammable cells in range", () => {
    const t = flatWorld();
    fill(t, 30, GROUND - 3, 6, 3, WOOD, { structural: false });
    const n = t.ignite(32 * TILE, (GROUND - 2) * TILE, 10);
    expect(n).toBeGreaterThan(0);
    expect(t.flags[t.index(32, GROUND - 2)] & F_BURNING).toBe(F_BURNING);
    expect(t.flags[t.index(32, GROUND + 1)] & F_BURNING).toBe(0);
  });

  it("flood fills only air, and the liquid then flows", () => {
    const t = flatWorld();
    const n = t.flood(50 * TILE, (GROUND - 1) * TILE, 20, WATER);
    expect(n).toBeGreaterThan(5);
    expect(count(t, WATER)).toBe(n);
    expect(t.get(50, GROUND + 1)).toBe(DIRT);
    t.set(70, GROUND - 1, SAND);
    expect(t.flood(70 * TILE + 4, (GROUND - 1) * TILE + 4, 2, WATER)).toBe(0);
    settle(t);
    expect(count(t, WATER)).toBe(n);
  });
});
