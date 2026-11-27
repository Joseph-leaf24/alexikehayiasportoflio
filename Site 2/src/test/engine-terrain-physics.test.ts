import { describe, expect, it } from "vitest";
import { Terrain } from "@/engine/terrain/grid";
import {
  EV_BURNOUT,
  EV_DISSOLVE,
  EV_EXTINGUISH,
  EV_STEAM,
  F_BURNING,
  F_WET,
  ROWS,
  TILE,
} from "@/engine/terrain/constants";
import {
  ACID,
  AIR,
  ASH,
  BEDROCK,
  GLASS,
  LAVA,
  LEAF,
  RUBBLE,
  SAND,
  STONE,
  WATER,
  WOOD,
} from "@/engine/terrain/materials";
import { count, emptyWorld, fill, flatWorld, GROUND, run, settle } from "./engine-terrain-fixtures";

const FLOOR = ROWS - 1;

/** Event types drained so far. */
function events(t: Terrain): number[] {
  const out: number[] = [];
  const n = t.drainEvents(out);
  return Array.from({ length: n }, (_, k) => out[k * 4]);
}

describe("powders", () => {
  it("sand falls, piles no steeper than 45 degrees, keeps its mass and goes to sleep", () => {
    const t = emptyWorld();
    fill(t, 100, 50, 10, 30, SAND);
    const before = count(t, SAND);
    const ticks = settle(t);
    expect(ticks).toBeLessThan(3000);
    expect(count(t, SAND)).toBe(before);
    for (let c = 80; c < 130; c++)
      for (let r = 0; r < FLOOR; r++) {
        if (t.get(c, r) !== SAND) continue;
        // Resting: something under it, and no open slope beside it to slide down
        expect(t.get(c, r + 1)).not.toBe(AIR);
        for (const d of [-1, 1]) expect(t.get(c + d, r) === AIR && t.get(c + d, r + 1) === AIR).toBe(false);
      }
    // It spread into a pile wider than it started
    let width = 0;
    for (let c = 80; c < 130; c++) if (t.get(c, FLOOR - 1) === SAND) width++;
    expect(width).toBeGreaterThan(10);
    expect(t.awakeChunks()).toBe(0);
    expect(t.busy).toBe(false);
  });

  it("sand slides into a crater dug next to it", () => {
    const t = flatWorld();
    fill(t, 60, GROUND - 6, 12, 6, SAND);
    settle(t);
    t.carve(54 * TILE, (GROUND + 2) * TILE, 28, 255);
    settle(t);
    // Some of the dune ran into the hole
    let inHole = 0;
    for (let c = 50; c < 58; c++) for (let r = GROUND; r < GROUND + 6; r++) if (t.get(c, r) === SAND) inHole++;
    expect(inHole).toBeGreaterThan(0);
  });

  it("sinks through water, while the water rises", () => {
    const t = emptyWorld();
    fill(t, 40, FLOOR - 10, 1, 10, WATER);
    t.set(40, FLOOR - 12, SAND);
    // A wall either side keeps the water in its column
    fill(t, 39, FLOOR - 14, 1, 14, STONE);
    fill(t, 41, FLOOR - 14, 1, 14, STONE);
    settle(t);
    expect(t.get(40, FLOOR - 1)).toBe(SAND);
    expect(count(t, WATER)).toBe(10);
  });

  it("crushes foliage it falls onto", () => {
    const t = emptyWorld();
    t.set(10, FLOOR - 1, LEAF);
    t.set(10, FLOOR - 20, SAND);
    settle(t);
    expect(t.get(10, FLOOR - 1)).toBe(SAND);
    expect(count(t, LEAF)).toBe(0);
  });
});

describe("liquids", () => {
  it("water poured into a basin spreads out level, keeps its volume and sleeps", () => {
    const t = emptyWorld();
    fill(t, 20, FLOOR - 8, 1, 8, STONE);
    fill(t, 61, FLOOR - 8, 1, 8, STONE);
    fill(t, 38, 150, 4, 20, WATER);
    const volume = count(t, WATER);
    const ticks = settle(t);
    expect(ticks).toBeLessThan(3000);
    expect(count(t, WATER)).toBe(volume);
    // 80 cells over a 40-wide basin: two full rows, flat
    let top = ROWS;
    let lowestTop = 0;
    for (let c = 21; c < 61; c++) {
      let r = 0;
      while (r < FLOOR && t.get(c, r) !== WATER) r++;
      top = Math.min(top, r);
      lowestTop = Math.max(lowestTop, r);
    }
    expect(lowestTop - top).toBeLessThanOrEqual(1);
    expect(t.busy).toBe(false);
  });

  it("drains out of a pool through a hole blown in its wall", () => {
    const t = emptyWorld();
    // A raised tank: floor and walls of stone, full of water, with open ground below and to the right
    fill(t, 30, 200, 20, 1, STONE);
    fill(t, 30, 190, 1, 10, STONE);
    fill(t, 49, 190, 1, 10, STONE);
    fill(t, 31, 190, 18, 10, WATER);
    settle(t);
    const volume = count(t, WATER);
    expect(t.get(40, 199)).toBe(WATER);
    t.carve(49 * TILE + 4, 198 * TILE, 10, 255);
    settle(t);
    expect(count(t, WATER)).toBe(volume);
    let below = 0;
    for (let c = 0; c < 200; c++) for (let r = 201; r < FLOOR; r++) if (t.get(c, r) === WATER) below++;
    expect(below).toBeGreaterThan(volume / 2);
  });

  it("lava next to water makes stone and steam", () => {
    const t = emptyWorld();
    fill(t, 50, FLOOR - 3, 6, 3, LAVA);
    fill(t, 56, FLOOR - 3, 6, 3, WATER);
    fill(t, 49, FLOOR - 6, 1, 6, BEDROCK);
    fill(t, 62, FLOOR - 6, 1, 6, BEDROCK);
    const water = count(t, WATER);
    settle(t, 2000);
    expect(count(t, STONE)).toBeGreaterThan(0);
    expect(count(t, WATER)).toBeLessThan(water);
    expect(events(t)).toContain(EV_STEAM);
    // Wherever lava and water still meet, they're separated by stone now
    for (let c = 50; c < 62; c++)
      for (let r = FLOOR - 6; r < FLOOR; r++)
        if (t.get(c, r) === LAVA) for (const d of [-1, 1]) expect(t.get(c + d, r)).not.toBe(WATER);
  });

  it("lava flows more slowly than water", () => {
    const flowAfter = (mat: number) => {
      const t = emptyWorld();
      fill(t, 100, FLOOR - 6, 2, 6, mat);
      run(t, 20);
      let far = 0;
      for (let c = 100; c < 160; c++) if (t.get(c, FLOOR - 1) === mat) far = c;
      return far - 100;
    };
    expect(flowAfter(LAVA)).toBeLessThan(flowAfter(WATER));
  });

  it("acid eats down through the ground until it is spent, but not through glass", () => {
    const t = flatWorld();
    fill(t, 40, GROUND - 3, 3, 3, ACID);
    // A glass cup standing on the ground, half full of acid
    fill(t, 79, GROUND - 1, 5, 1, GLASS);
    fill(t, 79, GROUND - 6, 1, 5, GLASS);
    fill(t, 83, GROUND - 6, 1, 5, GLASS);
    fill(t, 80, GROUND - 4, 3, 3, ACID);
    const before = count(t, (m) => m === STONE || m === 2 /* dirt */);
    settle(t, 6000);
    expect(count(t, ACID)).toBeLessThan(9 + 9);
    expect(count(t, (m) => m === STONE || m === 2)).toBeLessThan(before);
    expect(events(t)).toContain(EV_DISSOLVE);
    // The glass cup held its acid
    expect(count(t, GLASS)).toBe(5 + 5 + 5);
    expect(count(t, (m) => m === ACID)).toBeGreaterThanOrEqual(9);
    expect(t.get(81, GROUND - 2)).toBe(ACID);
  });
});

describe("fire", () => {
  it("spreads through wood and burns it out to ash, which falls", () => {
    const t = emptyWorld();
    fill(t, 30, FLOOR - 10, 10, 10, WOOD, { structural: false });
    t.ignite(30 * TILE + 4, (FLOOR - 1) * TILE + 4, 6);
    const burning: number[] = [];
    let most = 0;
    settle(t, 20000, (w) => {
      most = Math.max(most, w.collectBurning(0, 400, burning));
    });
    expect(most).toBeGreaterThan(5);
    expect(count(t, WOOD)).toBeLessThan(10);
    expect(count(t, ASH)).toBeGreaterThan(0);
    expect(events(t)).toContain(EV_BURNOUT);
    // Ash is loose: none of it hangs in the air
    for (let c = 25; c < 45; c++)
      for (let r = 0; r < FLOOR; r++) if (t.get(c, r) === ASH) expect(t.get(c, r + 1)).not.toBe(AIR);
    expect(t.busy).toBe(false);
  });

  it("is put out by water, and a wet cell only dries when a spark reaches it", () => {
    const t = emptyWorld();
    t.set(10, FLOOR - 1, WOOD, { structural: false });
    t.set(11, FLOOR - 1, WATER);
    fill(t, 12, FLOOR - 2, 1, 2, STONE);
    t.ignite(10 * TILE + 4, (FLOOR - 1) * TILE + 4, 2);
    run(t, 2);
    const i = t.index(10, FLOOR - 1);
    expect(t.flags[i] & F_BURNING).toBe(0);
    expect(t.flags[i] & F_WET).toBe(F_WET);
    expect(events(t)).toContain(EV_EXTINGUISH);
    t.set(11, FLOOR - 1, AIR);
    expect(t.ignite(10 * TILE + 4, (FLOOR - 1) * TILE + 4, 2)).toBe(0);
    expect(t.flags[i] & F_WET).toBe(0);
    expect(t.ignite(10 * TILE + 4, (FLOOR - 1) * TILE + 4, 2)).toBe(1);
  });

  it("is lit by lava", () => {
    const t = emptyWorld();
    t.set(20, FLOOR - 1, LAVA);
    fill(t, 19, FLOOR - 2, 1, 2, BEDROCK);
    fill(t, 21, FLOOR - 1, 1, 1, WOOD, { structural: false });
    fill(t, 22, FLOOR - 2, 1, 2, BEDROCK);
    t.set(20, FLOOR - 2, BEDROCK);
    let lit = false;
    run(t, 200, (w) => {
      lit ||= (w.flags[w.index(21, FLOOR - 1)] & F_BURNING) !== 0;
    });
    expect(lit).toBe(true);
  });
});

describe("sleeping and budgets", () => {
  it("a settled world costs nothing: no chunk awake, none simulated", () => {
    const t = flatWorld({ wakeOnGenerate: true });
    expect(t.awakeChunks()).toBeGreaterThan(0);
    settle(t);
    t.step();
    expect(t.stats.chunks).toBe(0);
    expect(t.stats.moves).toBe(0);
    expect(t.stats.awake).toBe(0);
  });

  it("wakes only the chunks around a change", () => {
    const t = flatWorld();
    settle(t);
    t.set(100, 100, SAND);
    expect(t.awakeChunks()).toBeLessThanOrEqual(4);
    settle(t);
    expect(t.get(100, GROUND - 1)).toBe(SAND);
    expect(t.awakeChunks()).toBe(0);
  });

  it("caps moves and chunks per tick, and still finishes a huge avalanche", () => {
    const t = emptyWorld({ physics: { maxMoves: 3000, maxActiveChunks: 64 } });
    fill(t, 0, 20, 380, 60, SAND);
    const sand = count(t, SAND);
    let peakMoves = 0;
    let peakChunks = 0;
    const ticks = settle(t, 5000, (w) => {
      peakMoves = Math.max(peakMoves, w.stats.moves);
      peakChunks = Math.max(peakChunks, w.stats.chunks);
    });
    expect(ticks).toBeLessThan(5000);
    expect(peakMoves).toBeLessThanOrEqual(3000);
    expect(peakMoves).toBeGreaterThan(2000);
    expect(peakChunks).toBeLessThanOrEqual(64);
    expect(count(t, SAND)).toBe(sand);
  });
});

describe("determinism", () => {
  const scene = (seed: number) => {
    const t = flatWorld({ seed });
    fill(t, 40, 60, 30, 20, SAND);
    fill(t, 120, 100, 20, 20, WATER);
    fill(t, 200, GROUND - 8, 10, 8, WOOD, { structural: false });
    fill(t, 260, GROUND - 4, 4, 4, ACID);
    fill(t, 300, GROUND - 5, 6, 5, LAVA);
    t.ignite(204 * TILE, (GROUND - 1) * TILE, 10);
    for (let k = 0; k < 200; k++) {
      if (k % 10 === 0) t.carve((40 + ((k * 37) % 300)) * TILE, (GROUND + 2) * TILE, 20, 3);
      t.step();
    }
    return t;
  };

  it("same seed and inputs, same world", () => {
    const a = scene(11);
    const b = scene(11);
    expect(a.hash()).toBe(b.hash());
    expect(a.fg).toEqual(b.fg);
    expect(count(a, RUBBLE) + count(a, ASH)).toBeGreaterThan(0);
  });

  it("a different seed plays out differently", () => {
    expect(scene(11).hash()).not.toBe(scene(12).hash());
  });
});
