import { describe, expect, it } from "vitest";
import { Terrain } from "@/engine/terrain/grid";
import { EV_CRUSH, EV_DETACH, EV_IMPACT, F_LOOSE, F_STRUCTURAL, TILE } from "@/engine/terrain/constants";
import { AIR, BRICK, CONCRETE, DIRT, GLASS, RUBBLE, SAND, STONE, STEEL } from "@/engine/terrain/materials";
import { building, count, fill, flatWorld, GROUND, run, settle, structuralIn } from "./engine-terrain-fixtures";

/** Run until settled, collecting event types and the peak per-tick costs. */
function collapse(t: Terrain, max = 4000) {
  const seen = new Map<number, number>();
  const out: number[] = [];
  const peak = { work: 0, moves: 0, falling: 0, ticks: 0 };
  peak.ticks = settle(t, max, (w) => {
    peak.work = Math.max(peak.work, w.stats.work);
    peak.moves = Math.max(peak.moves, w.stats.moves);
    peak.falling = Math.max(peak.falling, w.stats.falling);
    const n = w.drainEvents(out);
    for (let k = 0; k < n; k++) seen.set(out[k * 4], (seen.get(out[k * 4]) ?? 0) + 1);
  });
  return { seen, peak };
}

/** Destroy a rectangle of cells as blasts would. */
function knockOut(t: Terrain, c0: number, r0: number, w: number, h: number) {
  for (let c = c0; c < c0 + w; c++) for (let r = r0; r < r0 + h; r++) t.damage(c, r, 255);
}

describe("cut-off ground", () => {
  it("a small piece left floating crumbles and falls, as in the old engine", () => {
    const t = flatWorld();
    // A rock ledge on a cliff, then the cliff under and beside it is blasted away
    fill(t, 60, GROUND - 20, 8, 20, STONE);
    fill(t, 68, GROUND - 14, 5, 2, STONE);
    settle(t);
    knockOut(t, 60, GROUND - 14, 8, 2);
    const { seen } = collapse(t);
    expect(seen.get(EV_DETACH)).toBeGreaterThan(0);
    // The ledge landed on the ground, intact blocks
    expect(t.get(70, GROUND - 13)).toBe(AIR);
    expect(t.get(70, GROUND - 1)).toBe(STONE);
    expect(count(t, (m, f) => m === STONE && (f & F_LOOSE) !== 0)).toBe(0);
  });

  it("a big overhang stays up", () => {
    const t = flatWorld();
    fill(t, 60, GROUND - 30, 10, 30, STONE);
    fill(t, 70, GROUND - 30, 40, 4, STONE);
    settle(t);
    t.damage(69, GROUND - 1, 255);
    collapse(t);
    expect(count(t, (m) => m === STONE) - count(flatWorld(), STONE)).toBe(10 * 30 + 40 * 4 - 1);
    expect(t.get(100, GROUND - 28)).toBe(STONE);
  });
});

describe("buildings", () => {
  it("collapse when their ground floor is carved away, falling as rigid chunks that pancake into rubble", () => {
    const t = flatWorld();
    const top = building(t, 100, 20, 6, 5, CONCRETE);
    const cells = structuralIn(t, 100, top, 20, GROUND - top);
    settle(t);
    t.carve(100 * TILE + 4, (GROUND - 3) * TILE, 22, 255);
    t.carve(119 * TILE + 4, (GROUND - 3) * TILE, 22, 255);
    let fell = false;
    const { seen, peak } = collapse(t, 4000);
    expect(peak.falling).toBeGreaterThan(0);
    expect(seen.get(EV_DETACH)).toBeGreaterThan(0);
    expect(seen.get(EV_IMPACT)).toBeGreaterThan(0);
    // Nothing structural is left standing where the storeys were, and it all came down as debris
    expect(structuralIn(t, 95, top - 2, 30, GROUND - 6 - top)).toBe(0);
    for (let c = 95; c < 125; c++) for (let r = 0; r < GROUND - 1; r++) if (t.get(c, r) !== AIR) fell = true;
    expect(fell).toBe(true);
    expect(count(t, RUBBLE)).toBeGreaterThan(cells / 10);
    // Debris rests: nothing floats
    for (let c = 80; c < 140; c++)
      for (let r = 0; r < GROUND - 1; r++) if (t.get(c, r) !== AIR) expect(t.get(c, r + 1), `${c},${r}`).not.toBe(AIR);
    expect(t.busy).toBe(false);
  });

  it("stand with a hole blown in a wall", () => {
    const t = flatWorld();
    const top = building(t, 100, 20, 6, 5, BRICK);
    const before = structuralIn(t, 100, top, 20, GROUND - top);
    settle(t);
    knockOut(t, 119, GROUND - 14, 1, 3);
    knockOut(t, 100, GROUND - 22, 1, 3);
    collapse(t);
    expect(structuralIn(t, 100, top, 20, GROUND - top)).toBe(before - 6);
  });

  it("topple when only one side is left holding them up", () => {
    const t = flatWorld();
    const top = building(t, 100, 24, 4, 5, BRICK);
    settle(t);
    // The right wall's ground floor goes; the left wall alone can't balance the rest
    knockOut(t, 123, GROUND - 4, 1, 4);
    const { seen } = collapse(t);
    expect(seen.get(EV_CRUSH)).toBeGreaterThan(0);
    expect(structuralIn(t, 100, top, 24, GROUND - 6 - top)).toBe(0);
  });

  it("give way when overloaded: a tower on a single slim pillar", () => {
    const t = flatWorld();
    // A solid glass block (weak) held up by one pillar under its middle
    fill(t, 140, GROUND - 3, 1, 3, GLASS);
    fill(t, 130, GROUND - 30, 21, 27, GLASS);
    settle(t);
    // Any damage to it triggers a check
    t.damage(150, GROUND - 4, 255);
    const { seen } = collapse(t);
    expect(seen.get(EV_CRUSH)).toBeGreaterThan(0);
    expect(structuralIn(t, 130, GROUND - 30, 21, 20)).toBe(0);
  });

  it("but stand on a pillar strong enough for them", () => {
    const t = flatWorld();
    fill(t, 139, GROUND - 3, 3, 3, STEEL);
    fill(t, 130, GROUND - 13, 21, 10, CONCRETE);
    settle(t);
    t.damage(150, GROUND - 4, 255);
    const { seen } = collapse(t);
    expect(seen.get(EV_CRUSH)).toBeUndefined();
    expect(structuralIn(t, 130, GROUND - 13, 21, 10)).toBe(21 * 10 - 1);
  });

  it("aren't held up by loose junk inside them, only by real ground", () => {
    const t = flatWorld();
    const top = building(t, 100, 12, 3, 5, CONCRETE);
    // A stone block on the first-floor slab, touching the slab above too
    fill(t, 104, GROUND - 9, 2, 4, STONE);
    settle(t);
    knockOut(t, 100, GROUND - 5, 1, 5);
    knockOut(t, 111, GROUND - 5, 1, 5);
    collapse(t);
    expect(structuralIn(t, 100, top, 12, GROUND - 6 - top)).toBe(0);
  });

  it("standing on sand stay up", () => {
    const t = flatWorld();
    fill(t, 90, GROUND - 4, 40, 4, SAND);
    settle(t);
    const top = building(t, 100, 16, 3, 5, BRICK, GROUND - 4);
    const before = structuralIn(t, 100, top, 16, GROUND - top);
    // Chip the roof: the check that follows finds it resting on settled sand, which holds
    t.damage(115, top, 255);
    const { seen } = collapse(t);
    expect(seen.get(EV_DETACH)).toBeUndefined();
    expect(structuralIn(t, 100, top, 16, GROUND - top)).toBe(before - 1);
  });

  it("bridges hold between two cliffs and fall when one end is cut", () => {
    const t = flatWorld();
    fill(t, 50, GROUND - 20, 10, 20, STONE);
    fill(t, 90, GROUND - 20, 10, 20, STONE);
    fill(t, 60, GROUND - 20, 30, 2, CONCRETE, { structural: true });
    settle(t);
    // A hole in the middle of the deck: both ends still bear on the cliffs
    t.damage(75, GROUND - 19, 255);
    collapse(t);
    expect(structuralIn(t, 60, GROUND - 20, 30, 2)).toBe(59);
    // Blow the left end off the cliff
    knockOut(t, 60, GROUND - 20, 2, 2);
    collapse(t);
    expect(structuralIn(t, 60, GROUND - 20, 30, 2)).toBe(0);
  });
});

describe("rigid falling chunks", () => {
  it("fall as one piece, accelerating, carrying the back wall, and crush into the ground", () => {
    const t = flatWorld();
    // A box of brick with a back wall inside, hung from a thin neck that is then cut
    fill(t, 200, 40, 10, 1, BRICK);
    fill(t, 200, 49, 10, 1, BRICK);
    fill(t, 200, 41, 1, 8, BRICK);
    fill(t, 209, 41, 1, 8, BRICK);
    for (let c = 201; c < 209; c++) for (let r = 41; r < 49; r++) t.setBg(c, r, DIRT);
    fill(t, 204, 20, 1, 20, STONE);
    settle(t);
    t.damage(204, 39, 255);
    const ys: number[] = [];
    let chunkBg = 0;
    for (let k = 0; k < 6; k++) {
      t.step();
      const ch = t.falling[0];
      if (!ch) break;
      ys.push(ch.y);
      chunkBg = ch.bg.reduce((s, v) => s + (v ? 1 : 0), 0);
      // The whole chunk moves together: its cells left the grid
      expect(t.get(200, 40)).toBe(AIR);
    }
    expect(ys.length).toBeGreaterThan(3);
    const steps = ys.slice(1).map((y, k) => y - ys[k]);
    for (let k = 1; k < steps.length; k++) expect(steps[k]).toBeGreaterThan(steps[k - 1]);
    expect(chunkBg).toBe(64);
    expect(t.bg[t.index(204, 45)]).toBe(AIR);
    const { seen } = collapse(t);
    expect(seen.get(EV_IMPACT)).toBeGreaterThan(0);
    expect(t.falling.length).toBe(0);
    // It came to rest on the ground as debris
    let debris = 0;
    for (let c = 190; c < 220; c++) for (let r = GROUND - 12; r < GROUND; r++) if (t.get(c, r) !== AIR) debris++;
    expect(debris).toBeGreaterThan(10);
  });

  it("crumble as loose cells instead when too many are in flight", () => {
    const t = flatWorld({ structure: { maxFalling: 0 } });
    const top = building(t, 100, 10, 3, 5, BRICK);
    settle(t);
    knockOut(t, 100, GROUND - 4, 1, 4);
    knockOut(t, 109, GROUND - 4, 1, 4);
    let flying = 0;
    collapse(t);
    run(t, 1, (w) => (flying += w.falling.length));
    expect(flying).toBe(0);
    expect(structuralIn(t, 100, top, 10, GROUND - 6 - top)).toBe(0);
  });
});

describe("budgets", () => {
  it("a giant collapse finishes over several ticks without one tick exceeding the caps", () => {
    const t = flatWorld({ ground: 232, structure: { budget: 4000 }, physics: { maxMoves: 3000 } });
    // A solid 50 × 200 tower: 10,000 structural cells
    fill(t, 150, 32, 50, 200, CONCRETE, { structural: true });
    settle(t);
    knockOut(t, 150, 229, 50, 3);
    const { peak } = collapse(t, 6000);
    expect(peak.ticks).toBeLessThan(6000);
    expect(peak.work).toBeLessThanOrEqual(4000);
    expect(peak.moves).toBeLessThanOrEqual(3000);
    // It fell as more than one band, lifted over more than one tick
    expect(peak.falling).toBeGreaterThan(1);
    expect(structuralIn(t, 150, 32, 50, 200)).toBe(0);
  });

  it("support checks of a big structure spread over ticks", () => {
    const works: number[] = [];
    // 2,400 cells checked at 500 per tick: several ticks, never over budget, and it still stands
    const u = flatWorld({ structure: { budget: 500 } });
    fill(u, 100, GROUND - 60, 40, 60, CONCRETE, { structural: true });
    settle(u);
    u.damage(100, GROUND - 1, 255);
    for (let k = 0; k < 20; k++) {
      u.step();
      works.push(u.stats.fill);
    }
    expect(works.filter((w) => w > 0).length).toBeGreaterThanOrEqual(4);
    expect(Math.max(...works)).toBeLessThanOrEqual(500);
    expect(structuralIn(u, 100, GROUND - 60, 40, 60)).toBe(40 * 60 - 1);
  });
});

describe("determinism", () => {
  it("the same collapse plays out the same way", () => {
    const go = () => {
      const t = flatWorld({ seed: 5 });
      building(t, 100, 20, 8, 5, CONCRETE);
      building(t, 130, 14, 5, 6, BRICK);
      settle(t);
      t.carve(100 * TILE, (GROUND - 3) * TILE, 24, 6);
      t.carve(130 * TILE, (GROUND - 3) * TILE, 24, 6);
      for (let k = 0; k < 300; k++) t.step();
      return t;
    };
    const a = go();
    const b = go();
    expect(a.hash()).toBe(b.hash());
    expect(count(a, (_m, f) => (f & F_STRUCTURAL) !== 0)).toBe(count(b, (_m, f) => (f & F_STRUCTURAL) !== 0));
    expect(count(a, RUBBLE)).toBeGreaterThan(0);
  });
});
