import { describe, expect, it } from "vitest";
import { Terrain } from "@/engine/terrain/grid";
import { createRayHit } from "@/engine/terrain/query";
import { LEVEL_H, ROWS, TILE } from "@/engine/terrain/constants";
import { AIR, BEDROCK, LAVA, LEAF, SOLID, STONE, WATER } from "@/engine/terrain/materials";
import { TerrainRng } from "@/engine/terrain/random";

/** A random 120-column world: scattered stone, some water and lava, leaves (not solid), bedrock floor. */
function noisy(seed: number): Terrain {
  const r = new TerrainRng(seed);
  const t = new Terrain({
    seed,
    generate: (w, from, to) => {
      for (let c = from; c < to; c++)
        for (let row = 0; row < ROWS; row++) {
          const p = r.next();
          const m =
            row === ROWS - 1 ? BEDROCK : p < 0.12 ? STONE : p < 0.14 ? WATER : p < 0.15 ? LAVA : p < 0.17 ? LEAF : AIR;
          if (m) w.set(c, row, m);
        }
    },
  });
  t.ensure(120);
  return t;
}

/** The old-engine semantics, by hand. */
function solidCell(t: Terrain, c: number, r: number): boolean {
  if (r >= ROWS) return true;
  if (r < 0 || c < t.startCol || c >= t.endCol) return false;
  return SOLID[t.get(c, r)] === 1;
}

/** Exact segment-vs-cell test (slab method): the length of the segment inside the cell's box is positive. */
function crosses(x0: number, y0: number, x1: number, y1: number, c: number, r: number): boolean {
  let lo = 0;
  let hi = 1;
  for (const [p, d, min] of [
    [x0, x1 - x0, c * TILE],
    [y0, y1 - y0, r * TILE],
  ]) {
    if (d === 0) {
      if (p < min || p >= min + TILE) return false;
      continue;
    }
    let a = (min - p) / d;
    let b = (min + TILE - p) / d;
    if (a > b) [a, b] = [b, a];
    lo = Math.max(lo, a);
    hi = Math.min(hi, b);
  }
  return hi - lo > 1e-9;
}

describe("point queries", () => {
  it("solidAt, liquidAt and materialAt agree with the cells", () => {
    const t = noisy(1);
    const r = new TerrainRng(99);
    for (let k = 0; k < 3000; k++) {
      const x = r.next() * 140 * TILE - 10 * TILE;
      const y = r.next() * (ROWS + 4) * TILE - 2 * TILE;
      const c = Math.floor(x / TILE);
      const row = Math.floor(y / TILE);
      expect(t.solidAt(x, y)).toBe(solidCell(t, c, row));
      const m = row >= ROWS ? BEDROCK : c < 0 || c >= 120 || row < 0 ? AIR : t.get(c, row);
      expect(t.materialAt(x, y)).toBe(m);
      expect(t.liquidAt(x, y)).toBe(m === WATER || m === LAVA ? m : 0);
    }
  });

  it("surfaceY and columnTop find the first solid cell", () => {
    const t = noisy(2);
    for (let c = 0; c < 120; c++) {
      let top = 0;
      while (top < ROWS && !solidCell(t, c, top)) top++;
      expect(t.columnTop(c)).toBe(top);
      expect(t.surfaceY(c * TILE + 3)).toBe(top * TILE);
      let below = 100;
      while (below < ROWS && !solidCell(t, c, below)) below++;
      expect(t.surfaceY(c * TILE + 3, 100 * TILE + 5)).toBe(below * TILE);
    }
    expect(t.surfaceY(-50)).toBe(LEVEL_H);
  });

  it("overlapsSolid matches a cell-by-cell check of the box", () => {
    const t = noisy(3);
    const r = new TerrainRng(5);
    for (let k = 0; k < 2000; k++) {
      const x = r.next() * 120 * TILE;
      const y = r.next() * ROWS * TILE;
      const hw = 1 + r.next() * 10;
      const h = 1 + r.next() * 24;
      let hit = false;
      for (let c = Math.floor((x - hw) / TILE); c <= Math.floor((x + hw - 1e-3) / TILE); c++)
        for (let row = Math.floor((y - h) / TILE); row <= Math.floor((y - 1e-3) / TILE); row++)
          hit ||= solidCell(t, c, row);
      expect(t.overlapsSolid(x, y, hw, h)).toBe(hit);
    }
    // Feet exactly on a cell top don't overlap it
    const flat = new Terrain({
      generate: (w, f, to) => void [...Array(to - f)].forEach((_, k) => w.set(f + k, 100, STONE)),
    });
    flat.ensure(10);
    expect(flat.overlapsSolid(40, 100 * TILE, 4, 12)).toBe(false);
    expect(flat.overlapsSolid(40, 100 * TILE + 0.5, 4, 12)).toBe(true);
  });
});

describe("rays and sight lines", () => {
  it("lineOfSight matches an exact walk of the cells between the end points", () => {
    const t = noisy(4);
    const r = new TerrainRng(6);
    let blocked = 0;
    for (let k = 0; k < 1500; k++) {
      const x0 = r.next() * 110 * TILE + 40;
      const y0 = r.next() * 200 * TILE;
      const x1 = x0 + (r.next() - 0.5) * 300;
      const y1 = y0 + (r.next() - 0.5) * 300;
      const sc = Math.floor(x0 / TILE);
      const sr = Math.floor(y0 / TILE);
      const ec = Math.floor(x1 / TILE);
      const er = Math.floor(y1 / TILE);
      let clear = true;
      for (let c = Math.min(sc, ec); c <= Math.max(sc, ec); c++)
        for (let row = Math.min(sr, er); row <= Math.max(sr, er); row++) {
          if ((c === sc && row === sr) || (c === ec && row === er)) continue;
          if (solidCell(t, c, row) && crosses(x0, y0, x1, y1, c, row)) clear = false;
        }
      if (!clear) blocked++;
      expect(t.lineOfSight(x0, y0, x1, y1), `${x0},${y0} → ${x1},${y1}`).toBe(clear);
    }
    expect(blocked).toBeGreaterThan(300);
  });

  it("raycast returns the first solid cell, its entry point and face normal", () => {
    const t = noisy(5);
    const r = new TerrainRng(8);
    const hit = createRayHit();
    for (let k = 0; k < 1500; k++) {
      const x0 = 20 * TILE + r.next() * 80 * TILE;
      const y0 = r.next() * 230 * TILE;
      if (t.solidAt(x0, y0)) continue;
      const a = r.next() * Math.PI * 2;
      const dx = Math.cos(a);
      const dy = Math.sin(a);
      const max = 160;
      // Brute force: the nearest solid cell the ray enters within range
      let best = Infinity;
      let bc = 0;
      let br = 0;
      for (let c = Math.floor((x0 - max) / TILE); c <= Math.floor((x0 + max) / TILE); c++)
        for (let row = Math.floor((y0 - max) / TILE); row <= Math.floor((y0 + max) / TILE); row++) {
          if (!solidCell(t, c, row)) continue;
          const tx0 = dx !== 0 ? (c * TILE - x0) / dx : -Infinity;
          const tx1 = dx !== 0 ? ((c + 1) * TILE - x0) / dx : Infinity;
          const ty0 = dy !== 0 ? (row * TILE - y0) / dy : -Infinity;
          const ty1 = dy !== 0 ? ((row + 1) * TILE - y0) / dy : Infinity;
          const enter = Math.max(Math.min(tx0, tx1), Math.min(ty0, ty1));
          const exit = Math.min(Math.max(tx0, tx1), Math.max(ty0, ty1));
          if (enter <= exit && enter >= 0 && enter <= max && enter < best) {
            best = enter;
            bc = c;
            br = row;
          }
        }
      const got = t.raycast(x0, y0, dx * 3, dy * 3, max, hit);
      expect(got).toBe(best !== Infinity);
      if (!got) {
        expect(hit.dist).toBe(max);
        continue;
      }
      expect(hit.dist).toBeCloseTo(best, 6);
      expect([hit.col, hit.row]).toEqual([bc, br]);
      expect(hit.mat).toBe(br >= ROWS ? BEDROCK : t.get(bc, br));
      // The normal points back out of the face it entered
      expect(Math.abs(hit.nx) + Math.abs(hit.ny)).toBe(1);
      expect(hit.nx * dx + hit.ny * dy).toBeLessThan(0);
      expect(t.solidAt(hit.x - hit.nx * 0.01, hit.y - hit.ny * 0.01)).toBe(true);
      expect(t.solidAt(hit.x + hit.nx * 0.01, hit.y + hit.ny * 0.01)).toBe(false);
    }
  });

  it("handles rays from inside solid, into the sky and off the world", () => {
    const t = noisy(6);
    const hit = createRayHit();
    t.set(50, 50, STONE);
    expect(t.raycast(50 * TILE + 4, 50 * TILE + 4, 1, 0, 100, hit)).toBe(true);
    expect(hit.dist).toBe(0);
    expect([hit.nx, hit.ny]).toEqual([0, 0]);
    for (let r = 0; r < 10; r++) t.set(60, r, AIR);
    expect(t.raycast(60 * TILE + 4, 9 * TILE, 0, -1, 1e9, hit)).toBe(false);
    expect(t.raycast(-100, 10 * TILE, -1, 0, 1e9, hit)).toBe(false);
    expect(t.raycast(-100, 10 * TILE, 0, 1, 1e9, hit)).toBe(true);
    expect(hit.mat).toBe(BEDROCK);
    expect(t.raycast(0, 0, 0, 0, 10, hit)).toBe(false);
  });
});
