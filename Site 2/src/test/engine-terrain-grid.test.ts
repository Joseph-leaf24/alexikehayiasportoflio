import { describe, expect, it } from "vitest";
import type { TerrainView } from "@/engine/render/types";
import { Terrain } from "@/engine/terrain/grid";
import {
  CHUNK,
  CHUNK_ROWS,
  CHUNKS,
  EV_IMPACT,
  EV_STEAM,
  F_LOOSE,
  F_REINFORCED,
  F_STRUCTURAL,
  RING,
  ROWS,
  TILE,
  VARIANTS,
} from "@/engine/terrain/constants";
import { AIR, BEDROCK, BRICK, DIRT, GLASS, MATERIAL_COUNT, SAND, STONE, WATER } from "@/engine/terrain/materials";
import { flatWorld, GROUND } from "./engine-terrain-fixtures";

describe("terrain layout and the ring", () => {
  it("indexes column-major, so a whole column is contiguous, and wraps the ring", () => {
    const t = new Terrain();
    expect(t.index(0, 0)).toBe(0);
    expect(t.index(0, 1)).toBe(1);
    expect(t.index(1, 0)).toBe(ROWS);
    expect(t.index(3, 17)).toBe(3 * ROWS + 17);
    expect(t.index(RING, 5)).toBe(t.index(0, 5));
    expect(t.index(RING + 7, 5)).toBe(t.index(7, 5));
    expect(t.fg.length).toBe(RING * ROWS);
  });

  it("reads air above the world and outside the window, bedrock below it", () => {
    const t = flatWorld();
    expect(t.get(10, -1)).toBe(AIR);
    expect(t.get(10, ROWS)).toBe(BEDROCK);
    expect(t.get(-5, GROUND + 1)).toBe(AIR);
    expect(t.get(RING + 3, GROUND + 1)).toBe(AIR);
    expect(t.get(10, GROUND)).toBe(DIRT);
    expect(t.get(10, GROUND + 10)).toBe(STONE);
    expect(t.get(10, ROWS - 1)).toBe(BEDROCK);
  });

  it("sets cells with the material's hp and flags, or the options given", () => {
    const t = flatWorld();
    expect(t.set(5, 10, SAND)).toBe(true);
    const i = t.index(5, 10);
    expect(t.fg[i]).toBe(SAND);
    expect(t.flags[i] & F_LOOSE).toBe(F_LOOSE);
    t.set(6, 10, BRICK);
    expect(t.flags[t.index(6, 10)] & F_STRUCTURAL).toBe(F_STRUCTURAL);
    expect(t.hp[t.index(6, 10)]).toBe(3);
    t.set(7, 10, STONE, { structural: true, reinforced: true, hp: 255, bg: GLASS, variant: 2 });
    const j = t.index(7, 10);
    expect(t.flags[j] & (F_STRUCTURAL | F_REINFORCED)).toBe(F_STRUCTURAL | F_REINFORCED);
    expect(t.hp[j]).toBe(255);
    expect(t.bg[j]).toBe(GLASS);
    expect(t.variant[j]).toBe(2);
    t.set(6, 10, BRICK, { structural: false });
    expect(t.flags[t.index(6, 10)] & F_STRUCTURAL).toBe(0);
    // Outside the writable window
    expect(t.set(-1, 10, STONE)).toBe(false);
    expect(t.set(RING, 10, STONE)).toBe(false);
    expect(t.set(3, ROWS, STONE)).toBe(false);
    expect(t.set(3, -1, STONE)).toBe(false);
  });

  it("gives every cell a variant hashed from its position, the same on every run", () => {
    const a = flatWorld({ seed: 3 });
    const b = flatWorld({ seed: 3 });
    const c = flatWorld({ seed: 4 });
    expect(a.variant).toEqual(b.variant);
    expect(a.variant).not.toEqual(c.variant);
    const seen = new Set<number>();
    for (let col = 0; col < 64; col++) seen.add(a.variant[a.index(col, GROUND + 2)]);
    expect([...seen].sort()).toEqual([0, 1, 2, 3].slice(0, VARIANTS));
  });

  it("ensure generates consecutive stretches, at most half a ring at a time", () => {
    const calls: [number, number][] = [];
    const t = new Terrain({ generate: (_w, from, to) => void calls.push([from, to]) });
    t.ensure(100);
    t.ensure(100);
    t.ensure(700);
    expect(calls).toEqual([
      [0, 100],
      [100, 356],
      [356, 612],
      [612, 700],
    ]);
    expect(t.endCol).toBe(700);
    // Room is kept for a generator to run 128 columns past what it was asked for
    expect(t.startCol).toBe(700 + 128 - RING);
  });

  it("recycles columns that scroll off the left, clearing their slots", () => {
    const t = new Terrain({
      generate: (w, from, to) => {
        for (let c = from; c < to; c++) w.set(c, 10, c % 2 ? STONE : DIRT, { bg: BRICK });
      },
    });
    t.ensure(RING - 128);
    expect(t.startCol).toBe(0);
    expect(t.get(0, 10)).toBe(DIRT);
    t.ensure(RING);
    expect(t.startCol).toBe(128);
    const keep = new Terrain({ overshoot: 0, generate: (w, from) => void w.set(from, 0, STONE) });
    keep.ensure(RING + 40);
    expect(keep.startCol).toBe(40);
    t.ensure(RING + 40);
    expect(t.startCol).toBe(40 + 128);
    expect(keep.get(RING + 39, 0)).toBe(AIR);
    expect(t.get(0, 10)).toBe(AIR);
    expect(t.get(167, 10)).toBe(AIR);
    expect(t.get(168, 10)).toBe(DIRT);
    // The recycled slots now hold the new columns, written fresh
    expect(t.fg[t.index(RING + 1, 10)]).toBe(STONE);
    expect(t.get(RING + 1, 10)).toBe(STONE);
    expect(t.bg[t.index(RING + 1, 10)]).toBe(BRICK);
    expect(t.bg[t.index(RING + 1, 11)]).toBe(AIR);
    // Far ahead: still continuous, never more than a ring stored
    t.ensure(10_000);
    expect(t.endCol).toBe(10_000);
    expect(t.endCol - t.startCol).toBe(RING - 128);
    for (let c = t.startCol; c < t.endCol; c++) expect(t.get(c, 10)).toBe(c % 2 ? STONE : DIRT);
  });

  it("lets a generator run past the stretch it was asked for, up to the window", () => {
    const t = new Terrain({
      generate: (w, from, to) => {
        const end = to + 30;
        for (let c = from; c < end; c++) w.set(c, 5, STONE);
        // Reading back what it wrote, even past `to`, sees it
        expect(w.get(end - 1, 5)).toBe(STONE);
        return end;
      },
    });
    t.ensure(50);
    expect(t.endCol).toBe(80);
    t.ensure(RING + 200);
    expect(t.endCol - t.startCol).toBeLessThanOrEqual(RING);
    for (let c = t.startCol; c < t.endCol; c++) expect(t.get(c, 5)).toBe(STONE);
  });

  it("cellAt maps world px to a ring index, or -1 outside", () => {
    const t = flatWorld();
    expect(t.cellAt(3 * TILE + 2, 7 * TILE + 7)).toBe(t.index(3, 7));
    expect(t.cellAt(-1, 10)).toBe(-1);
    expect(t.cellAt(10, -1)).toBe(-1);
    expect(t.cellAt(10, ROWS * TILE)).toBe(-1);
    expect(t.cellAt(RING * TILE, 10)).toBe(-1);
  });

  it("caches column tops and forgets them when a column changes", () => {
    const t = flatWorld();
    expect(t.columnTop(20)).toBe(GROUND);
    t.set(20, 100, STONE);
    expect(t.columnTop(20)).toBe(100);
    t.set(20, 100, AIR);
    expect(t.columnTop(20)).toBe(GROUND);
    t.set(20, 50, WATER);
    expect(t.columnTop(20)).toBe(GROUND);
    expect(t.columnTop(-3)).toBe(ROWS);
  });
});

describe("dirty chunks for the renderer", () => {
  it("marks every chunk of newly generated columns, then only what changes", () => {
    const t = new Terrain({ generate: (w, f) => void w.set(f, 0, STONE) });
    const out: number[] = [];
    t.ensure(40);
    // Columns 0..39 touch chunk columns 0, 1 and 2, all 15 rows of chunks
    expect(t.drainDirty(out)).toBe(3 * CHUNK_ROWS);
    expect(new Set(out).size).toBe(out.length);
    expect(t.drainDirty(out)).toBe(0);
    expect(out.length).toBe(0);
    t.set(20, 100, STONE);
    expect(t.drainDirty(out)).toBe(1);
    // Column-major chunk index, as TerrainView documents
    expect(out[0]).toBe((((20 & (RING - 1)) / CHUNK) | 0) * (ROWS / CHUNK) + ((100 / CHUNK) | 0));
  });

  it("packs a chunk as row-major RGBA (fg, bg, variant, hp) for texSubImage2D", () => {
    const t = flatWorld();
    t.set(17, 35, BRICK, { bg: GLASS, variant: 3 });
    const k = (17 >> 4) * CHUNK_ROWS + (35 >> 4);
    const px = new Uint8Array(CHUNK * CHUNK * 4);
    const origin = { x: -1, y: -1 };
    t.packChunk(k, px, origin);
    expect(origin).toEqual({ x: 16, y: 32 });
    const o = ((35 - 32) * CHUNK + (17 - 16)) * 4;
    expect([...px.slice(o, o + 4)]).toEqual([BRICK, GLASS, 3, 3]);
  });

  it("implements the renderer's TerrainView", () => {
    const view: TerrainView = new Terrain();
    expect(view.rows).toBe(ROWS);
    expect(view.ring).toBe(RING);
    expect(view.tile).toBe(TILE);
    expect(view.chunkSize).toBe(CHUNK);
    expect(view.variants).toBe(VARIANTS);
    expect(view.tiles.length).toBe(MATERIAL_COUNT * VARIANTS);
    expect(view.emissive?.length).toBe(256);
    expect(view.maxHp?.[BEDROCK]).toBe(255);
    expect(view.drainDirty([])).toBe(0);
    expect(CHUNKS).toBe((RING / CHUNK) * CHUNK_ROWS);
  });
});

describe("terrain events", () => {
  it("drains in order as quadruples and drops the oldest when full", () => {
    const t = new Terrain();
    t.emit(EV_STEAM, 1, 2, 3);
    t.emit(EV_IMPACT, 4, 5, 6);
    const out: number[] = [];
    expect(t.drainEvents(out)).toBe(2);
    expect(out).toEqual([EV_STEAM, 1, 2, 3, EV_IMPACT, 4, 5, 6]);
    expect(t.drainEvents(out)).toBe(0);
    for (let k = 0; k < 1100; k++) t.emit(EV_STEAM, k, 0, 0);
    expect(t.drainEvents(out)).toBe(1024);
    expect(out[1]).toBe(1100 - 1024);
    expect(out[out.length - 3]).toBe(1099);
  });
});
