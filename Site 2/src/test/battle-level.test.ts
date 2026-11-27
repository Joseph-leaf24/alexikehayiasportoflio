import { describe, expect, it } from "vitest";
import { cell } from "@/components/pixel/level";
import { F_STRUCTURAL, materials as M, ROWS, runHeadless, TILE } from "@/engine";
import { BattleGame, type Feature } from "@/battle";
import { install as art } from "@/battle/art";
import { install as flow } from "@/battle/flow";
import { install as level, LEVEL } from "@/battle/level";
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

describe("battle level generator", () => {
  it("copies the old generator's cells at their true columns, closing doorways with door leaves", () => {
    let doors = 0;
    for (const [theme, seed] of [
      ["jungle", 1],
      ["city", 3],
      ["cave", 5],
      ["archipelago", 7],
    ] as const) {
      const g = new BattleGame({ seed, feature: { level: { theme } } });
      const t = g.terrain;
      // Generate past a ring, so the window has recycled and the old ring has wrapped
      t.ensure(t.endCol + 700);
      const L = g.world.res(LEVEL).old;
      const off = g.origin.cols;
      const from = Math.max(t.startCol, L.start - off);
      const to = Math.min(t.endCol, L.next - off);
      expect(to - from).toBeGreaterThan(300);
      for (let c = from; c < to; c++)
        for (let r = 0; r < ROWS; r++) {
          const want = L.fg[cell(c + off, r)];
          const got = t.get(c, r);
          if (got === want) continue;
          // The only difference: a door leaf where the old doorway was open
          expect([theme, c, r, want, got]).toEqual([theme, c, r, M.AIR, M.WOOD]);
          doors++;
        }
      // What the old generator placed stays within its ring, so the list can't grow without end
      const spawns = g.world.res(LEVEL).spawns;
      expect(spawns.length).toBeGreaterThan(0);
      expect(spawns.every((s) => s.x >= L.start * TILE)).toBe(true);
    }
    expect(doors).toBeGreaterThan(0);
  });

  it("every building stands up to a support check", () => {
    for (const [theme, seed] of [
      ["city", 3],
      ["village", 4],
      ["megacity", 9],
    ] as const) {
      const g = emptyLevel(theme, seed);
      const t = g.terrain;
      // Check each structure from the top of every column it reaches
      let checked = 0;
      for (let c = t.startCol; c < t.endCol; c++)
        for (let r = 0; r < ROWS; r++)
          if (t.flags[t.index(c, r)] & F_STRUCTURAL) {
            t.structure.queue(c, r);
            checked++;
            break;
          }
      expect(checked).toBeGreaterThan(0);
      let fell = 0;
      runHeadless(g.world, 10, {
        onTick: () => {
          fell = Math.max(fell, t.structure.falling.chunks.length);
          return t.structure.pending;
        },
      });
      expect(t.structure.pending).toBe(false);
      expect([theme, fell]).toEqual([theme, 0]);
    }
  });
});
