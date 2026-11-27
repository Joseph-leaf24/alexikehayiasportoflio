import { describe, expect, it } from "vitest";
import { createFrame, hashWorld, runHeadless, TICK } from "@/engine";
import { BattleGame, MODE_CHUTE, UNIT_HIDDEN } from "@/battle";
import { SQUAD, SQUAD_SIZE, START_X } from "@/battle/squad";
import { ix } from "./battle-fixtures";

describe("battle vertical slice (headless)", () => {
  it("starts level 1 with its banner and the squad parachuting in", () => {
    const g = new BattleGame({ seed: 4 });
    runHeadless(g.world, TICK);
    expect(g.level.no).toBe(1);
    expect(g.banners.current?.text).toBe("Level 1");
    const sq = g.world.res(SQUAD);
    expect(sq.players).toHaveLength(SQUAD_SIZE);
    for (const id of sq.players) expect(g.c.motor.mode[ix(id)]).toBe(MODE_CHUTE);
    // Every player wears their own colours: eight different art sets
    expect(new Set(sq.players.map((id) => g.c.unit.art[ix(id)])).size).toBe(SQUAD_SIZE);
  });

  it.each([3, 7, 21])("the squad fights its way at least 1,000 px forward in 60 s, with kills (seed %i)", (seed) => {
    const g = new BattleGame({ seed });
    runHeadless(g.world, 60);
    const sq = g.world.res(SQUAD);
    expect(sq.advance - START_X).toBeGreaterThan(1000);
    expect(g.stats.kills).toBeGreaterThan(5);
    expect(g.stats.booms).toBeGreaterThan(3);
    expect(g.stats.enemies).toBeLessThanOrEqual(g.budgets.maxEnemies);
  });

  it("the same seed gives the same world, tick for tick; another seed doesn't", () => {
    const run = (seed: number) => {
      const g = new BattleGame({ seed });
      runHeadless(g.world, 30);
      return { world: hashWorld(g.world), terrain: g.terrain.hash(), particles: g.particles.n, kills: g.stats.kills };
    };
    const a = run(9);
    const b = run(9);
    expect(b).toEqual(a);
    expect(run(10).world).not.toBe(a.world);
  });

  it("drawing never changes the simulation", () => {
    const plain = new BattleGame({ seed: 12 });
    runHeadless(plain.world, 10);
    const drawn = new BattleGame({ seed: 12 });
    const frame = createFrame();
    runHeadless(drawn.world, 10, { onTick: () => drawn.buildFrame(frame, 0.5) });
    expect(hashWorld(drawn.world)).toBe(hashWorld(plain.world));
  });

  it("downed players drop back in by parachute", () => {
    const g = new BattleGame({ seed: 3 });
    const down: number[] = [];
    const back: number[] = [];
    g.world.events.on("playerDown", (e) => down.push(e.player));
    g.world.events.on("playerRespawned", (e) => back.push(e.player));
    let sawHidden = false;
    let sawChute = false;
    runHeadless(g.world, 60, {
      onTick: () => {
        if (down.length === 0) return;
        const id = g.world.res(SQUAD).players[down[0]];
        sawHidden ||= (g.c.unit.flags[ix(id)] & UNIT_HIDDEN) !== 0;
        sawChute ||= back.includes(down[0]) && g.c.motor.mode[ix(id)] === MODE_CHUTE;
      },
    });
    expect(down.length).toBeGreaterThan(0);
    expect(back).toContain(down[0]);
    expect(sawHidden).toBe(true);
    expect(sawChute).toBe(true);
  });
});
