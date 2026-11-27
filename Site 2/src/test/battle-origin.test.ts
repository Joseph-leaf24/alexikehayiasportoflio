import { describe, expect, it } from "vitest";
import { materials as M, ROWS, runHeadless } from "@/engine";
import { BattleGame, ORIGIN_STEP } from "@/battle";
import { SQUAD } from "@/battle/squad";
import { buried, GROUND, GROUND_Y, stagedGame } from "./battle-fixtures";

/** Run until the game's first origin shift; returns the ticks run. */
function untilShift(g: BattleGame, maxSeconds = 600): number {
  const r = runHeadless(g.world, maxSeconds, { onTick: () => g.origin.shifts === 0 });
  return r.ticks;
}

describe("battle floating origin", () => {
  it("shifting by a ring keeps entities, terrain, camera, particles and tracked columns consistent", () => {
    // Sooner than the default two rings, but still with everything past one ring out, so the shift is exact
    const a = new BattleGame({ seed: 6 });
    a.origin.threshold = 1.5 * ORIGIN_STEP;
    const ticks = untilShift(a);
    expect(a.origin.shifts).toBe(1);
    expect(a.origin.x).toBe(ORIGIN_STEP);
    expect(a.origin.cols).toBe(ORIGIN_STEP / 8);
    // The twin never shifts: run it the same number of ticks
    const b = new BattleGame({ seed: 6 });
    b.origin.threshold = Infinity;
    runHeadless(b.world, ticks / 30);
    expect(b.world.time.tick).toBe(a.world.time.tick);

    // Every position slid back by exactly one ring (positions shrink in magnitude, so f32 keeps every bit)
    const { pos, prev } = a.core;
    for (let k = 0; k < pos.$.size; k++) {
      const i = pos.$.dense[k];
      expect(pos.x[i]).toBe(b.core.pos.x[i] - ORIGIN_STEP);
      expect(prev.x[i]).toBe(b.core.prev.x[i] - ORIGIN_STEP);
      expect(pos.y[i]).toBe(b.core.pos.y[i]);
    }
    expect(a.world.count).toBe(b.world.count);
    for (let k = 0; k < a.c.ai.$.size; k++) {
      const i = a.c.ai.$.dense[k];
      expect(a.c.ai.home[i]).toBe(b.c.ai.home[i] - ORIGIN_STEP);
    }
    expect(a.camera.x).toBeCloseTo(b.camera.x - ORIGIN_STEP, 6);
    expect(a.world.res(SQUAD).advance).toBeCloseTo(b.world.res(SQUAD).advance - ORIGIN_STEP, 6);
    expect(a.particles.n).toBe(b.particles.n);
    for (let k = 0; k < a.particles.n; k++) expect(a.particles.x[k]).toBe(b.particles.x[k] - ORIGIN_STEP);

    // The terrain window moved by a whole ring: the same cells, under the same (shifted) columns
    const dc = ORIGIN_STEP / 8;
    expect(a.terrain.startCol).toBe(b.terrain.startCol - dc);
    expect(a.terrain.endCol).toBe(b.terrain.endCol - dc);
    for (let c = a.terrain.startCol; c < a.terrain.endCol; c++)
      for (let r = 0; r < ROWS; r += 3) expect(a.terrain.get(c, r)).toBe(b.terrain.get(c + dc, r));

    // And the battle carries on: new columns come out as the unshifted run's, nobody ends up in the ground
    let offender = 0;
    runHeadless(a.world, 20, { onTick: () => void (offender ||= buried(a)) });
    runHeadless(b.world, 20);
    expect(offender).toBe(0);
    const ahead = Math.min(a.terrain.endCol, b.terrain.endCol - dc) - 4;
    expect(ahead).toBeGreaterThan(Math.ceil(a.camera.right / 8) + 40);
    for (let r = 0; r < ROWS; r++) expect(a.terrain.get(ahead, r)).toBe(b.terrain.get(ahead + dc, r));
    expect(a.world.res(SQUAD).advance + a.origin.x).toBeGreaterThan(b.world.res(SQUAD).advance - 600);
  }, 30_000);

  it("waits while structures are settling, then shifts", () => {
    const g = stagedGame({});
    // A slab on two posts, then the posts blown away: the slab falls for a while
    for (let c = 150; c <= 160; c++) g.terrain.set(c, GROUND - 12, M.CONCRETE, { structural: true });
    for (let r = GROUND - 11; r < GROUND; r++) {
      g.terrain.set(150, r, M.CONCRETE, { structural: true });
      g.terrain.set(160, r, M.CONCRETE, { structural: true });
    }
    runHeadless(g.world, 0.5);
    g.explode(150 * 8 + 4, GROUND_Y - 30, { r: 40, power: 255 });
    g.explode(160 * 8 + 4, GROUND_Y - 30, { r: 40, power: 255 });
    g.origin.threshold = 0;
    let shiftedWhileBusy = false;
    let sawBusy = false;
    runHeadless(g.world, 6, {
      onTick: () => {
        const busy = g.terrain.structure.pending;
        sawBusy ||= busy;
        if (busy && g.origin.shifts > 0) shiftedWhileBusy = true;
        return g.origin.shifts === 0;
      },
    });
    expect(sawBusy).toBe(true);
    expect(shiftedWhileBusy).toBe(false);
    expect(g.origin.shifts).toBe(1);
    expect(g.origin.x % ORIGIN_STEP).toBe(0);
    expect(() => g.origin.shift(100)).toThrow();
  });
});
