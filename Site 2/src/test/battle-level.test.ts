import { describe, expect, it } from "vitest";
import { cell } from "@/components/pixel/level";
import { F_STRUCTURAL, materials as M, ROWS, runHeadless, TILE, type Terrain } from "@/engine";
import { BattleGame, type Feature } from "@/battle";
import { install as art } from "@/battle/art";
import { install as flow } from "@/battle/flow";
import { install as level, LEVEL } from "@/battle/level";
import { THEMES } from "@/battle/level/themes";
import { install as physics } from "@/battle/physics";
import { install as render } from "@/battle/render";

/** A level with nobody in it: the generator's terrain and the physics, and no fighting to break things. */
function emptyLevel(theme: string, seed: number): BattleGame {
  const features: Feature[] = [
    { name: "art", install: art },
    { name: "render", install: render },
    { name: "physics", install: physics },
    { name: "level", install: level },
    { name: "flow", install: flow },
  ];
  return new BattleGame({ seed, features, feature: { level: { theme } } });
}

const structural = (t: Terrain, c: number, r: number) =>
  t.get(c, r) !== M.AIR && (t.flags[t.index(c, r)] & F_STRUCTURAL) !== 0;

/**
 * One cell of each structure in the stored window that rests on something. Pieces the old art left touching nothing
 * at all (a shelled ruin's slab between two shell holes) are debris, and fall at their first check.
 */
function standing(t: Terrain): [number, number][] {
  const seen = new Set<number>();
  const out: [number, number][] = [];
  for (let c = t.startCol; c < t.endCol; c++)
    for (let r = 0; r < ROWS; r++) {
      if (!structural(t, c, r) || seen.has(t.index(c, r))) continue;
      seen.add(t.index(c, r));
      const piece = [c, r];
      let touches = false;
      for (let n = 0; n < piece.length; n += 2)
        for (const [dc, dr] of [
          [1, 0],
          [-1, 0],
          [0, 1],
          [0, -1],
        ]) {
          const nc = piece[n] + dc;
          const nr = piece[n + 1] + dr;
          if (nc < t.startCol || nc >= t.endCol || nr < 0 || nr >= ROWS) continue;
          if (!structural(t, nc, nr)) touches ||= t.get(nc, nr) !== M.AIR;
          else if (!seen.has(t.index(nc, nr))) {
            seen.add(t.index(nc, nr));
            piece.push(nc, nr);
          }
        }
      if (touches) out.push([c, r]);
    }
  return out;
}

/** Run until the support checks are done; the most structures seen falling at once. */
function settle(g: BattleGame): number {
  const t = g.terrain;
  let fell = 0;
  runHeadless(g.world, 10, {
    onTick: () => {
      fell = Math.max(fell, t.structure.falling.chunks.length);
      return t.structure.pending;
    },
  });
  expect(t.structure.pending).toBe(false);
  return fell;
}

describe("battle level generator", () => {
  it("copies the old generator's cells at their true columns, adding only door leaves, beams and joints", () => {
    const added = new Map<number, number>();
    for (const [theme, seed] of [
      ["jungle", 1],
      ["city", 3],
      ["cave", 5],
      ["archipelago", 7],
      ["monsoon", 1],
      ["megacity", 1],
    ] as const) {
      const g = new BattleGame({ seed, feature: { level: { theme } } });
      const t = g.terrain;
      // Generate past a ring, so the window has recycled and the old ring has wrapped
      t.ensure(t.endCol + 700);
      const L = g.world.res(LEVEL).old;
      const off = g.origin.cols;
      const from = Math.max(t.startCol, L.start - off);
      const to = Math.min(t.endCol, L.next - off);
      expect(to - from).toBeGreaterThan(200);
      for (let c = from; c < to; c++)
        for (let r = 0; r < ROWS; r++) {
          const want = L.fg[cell(c + off, r)];
          const got = t.get(c, r);
          if (got === want) continue;
          // Beams go through air and foliage; door leaves and joints fill air with a structure's material
          const ok = got === M.BEAM ? want === M.AIR || M.SOFT[want] === 1 : want === M.AIR && structural(t, c, r);
          expect([theme, c, r, want, got, ok]).toEqual([theme, c, r, want, got, true]);
          added.set(got, (added.get(got) ?? 0) + 1);
        }
      // What the old generator placed stays within its ring, so the list can't grow without end
      const spawns = g.world.res(LEVEL).spawns;
      expect(spawns.length).toBeGreaterThan(0);
      expect(spawns.every((s) => s.x >= L.start * TILE)).toBe(true);
    }
    expect(added.get(M.WOOD)).toBeGreaterThan(0);
    expect(added.get(M.BEAM)).toBeGreaterThan(0);
  });

  it("everything that rests on something stands up to a support check, in every theme", () => {
    for (const theme of Object.keys(THEMES))
      for (const seed of [2, 3]) {
        const g = emptyLevel(theme, seed);
        for (const [c, r] of standing(g.terrain)) g.terrain.structure.queue(c, r);
        expect([theme, seed, settle(g)]).toEqual([theme, seed, 0]);
      }
  });

  it("a stilt hut stands on beams, and comes down when they're blasted", () => {
    const g = emptyLevel("monsoon", 1);
    const t = g.terrain;
    const legs = g.world.res(LEVEL).old.props.find((p) => p.kind === "legs");
    if (legs?.kind !== "legs") throw new Error("no stilt hut");
    const left = legs.x / TILE - g.origin.cols;
    const right = left + legs.w / TILE - 1;
    const floor = legs.y / TILE - 1;
    expect(t.get(left, floor + 1)).toBe(M.BEAM);
    // The far leg is the hut's ladder, which bears load already
    expect([M.BEAM, M.LADDER]).toContain(t.get(right, floor + 1));
    // Bodies and shots pass a beam
    expect(M.SOLID[M.BEAM]).toBe(0);
    t.structure.queue(left, floor);
    expect(settle(g)).toBe(0);
    // Blast the near leg: the hut tips off the far one
    for (let r = floor + 1; t.get(left, r) === M.BEAM; r++) t.damage(left, r, 255);
    expect(settle(g)).toBeGreaterThan(0);
    expect(structural(t, left + 1, floor)).toBe(false);
  });
});
