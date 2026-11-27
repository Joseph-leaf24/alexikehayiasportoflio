import { describe, expect, it } from "vitest";
import { F_STRUCTURAL, materials as M, runHeadless } from "@/engine";
import { BattleGame, INSTALLS, TEAM_REGIME, UNIT_DEAD, type Feature } from "@/battle";
import { GROUND, GROUND_Y, ix, stagedGame } from "./battle-fixtures";

const victims: Feature = {
  name: "victims",
  install: (g) => {
    g.content.units.register("test.victim", {
      name: "Victim",
      hw: 3,
      h: 15,
      hp: 10,
      team: TEAM_REGIME,
      art: "troop.jungle.mook",
    });
  },
};

/** Storey height in rows. */
const STOREY = 6;

/**
 * A three-storey building like the generator's: walls standing on the ground through wooden door leaves, a slab
 * every storey, all structural.
 */
function building(g: BattleGame, c0: number, c1: number, floors = 3): number {
  const t = g.terrain;
  const roof = GROUND - floors * STOREY;
  for (let c = c0; c <= c1; c++)
    for (let r = roof; r < GROUND; r++) {
      const h = GROUND - 1 - r;
      const wall = c === c0 || c === c1;
      const slab = h % STOREY === STOREY - 1;
      if (wall) t.set(c, r, h < 3 ? M.WOOD : M.BRICK, { structural: true });
      else if (slab) t.set(c, r, M.BRICK, { structural: true });
    }
  return roof;
}

const structural = (g: BattleGame, c0: number, c1: number) => {
  let n = 0;
  for (let c = c0; c <= c1; c++)
    for (let r = GROUND - 30; r < GROUND; r++) n += g.terrain.get(c, r) === M.BRICK ? 1 : 0;
  return n;
};

describe("battle structures", () => {
  it("a building stands until its ground floor is blasted, then comes down and crushes whoever is under it", () => {
    const g = stagedGame({ extra: [victims] });
    const c0 = 150,
      c1 = 166;
    building(g, c0, c1);
    const victim = g.spawn("test.victim", 158 * 8 + 4, GROUND_Y, { force: true });
    runHeadless(g.world, 2);
    // Standing: nothing has moved
    expect(g.terrain.falling.length).toBe(0);
    const before = structural(g, c0, c1);
    expect(before).toBeGreaterThan(60);
    expect(g.c.unit.flags[ix(victim)] & UNIT_DEAD).toBe(0);

    // Blow out both ground-floor walls
    g.explode(c0 * 8 + 4, GROUND_Y - 20, { r: 40, power: 255 });
    g.explode(c1 * 8 + 4, GROUND_Y - 20, { r: 40, power: 255 });
    expect(g.c.unit.flags[ix(victim)] & UNIT_DEAD).toBe(0);
    let fell = 0;
    let crushedBy = -1;
    g.world.events.on("unitKilled", (e) => {
      if (e.id === victim) crushedBy = e.killer;
    });
    runHeadless(g.world, 6, { onTick: () => void (fell = Math.max(fell, g.terrain.falling.length)) });
    expect(fell).toBeGreaterThan(0);
    expect(g.c.unit.flags[ix(victim)] & UNIT_DEAD).toBeTruthy();
    // Crushed by the falling floors, not by either blast
    expect(crushedBy).toBe(0);
    expect(structural(g, c0, c1)).toBeLessThan(before / 2);
  });

  it("a generated level's buildings stand untouched, and stand a support check", () => {
    const g = new BattleGame({
      seed: 5,
      features: INSTALLS.filter((f) => f.name !== "squad"),
      feature: { enemies: { spawn: false }, level: { theme: "city" } },
    });
    const t = g.terrain;
    g.camera.place(1500, GROUND_Y - 60);
    t.ensure(400);
    runHeadless(g.world, 3);
    expect(t.falling.length).toBe(0);
    expect(t.structure.pending).toBe(false);
    // Check every building's support as if something nearby had been blasted: the door leaves hold them up
    let checked = 0;
    for (let c = t.startCol; c < 400; c++)
      for (let r = 0; r < t.rows; r++)
        if (t.flags[(c & (t.ring - 1)) * t.rows + r] & F_STRUCTURAL) {
          t.structure.queue(c, r);
          checked++;
          break;
        }
    expect(checked).toBeGreaterThan(20);
    runHeadless(g.world, 3);
    expect(t.falling.length).toBe(0);
  });
});
