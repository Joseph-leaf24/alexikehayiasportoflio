import { describe, expect, it } from "vitest";
import { BODY_GROUNDED, materials as M, runHeadless, TICK } from "@/engine";
import {
  BattleGame,
  MODE_CLIMB,
  MODE_KNOCK,
  MODE_SWIM,
  MODE_WALK,
  TEAM_SQUAD,
  type Feature,
  type UnitDef,
} from "@/battle";
import { knock, DEFAULT_MOVE } from "@/battle/physics";
import { at, buried, clear, fill, GROUND, GROUND_Y, ix, stagedGame } from "./battle-fixtures";

/** A body with the squad's feel and no mind of its own: the tests drive its intent. */
const WALKER: UnitDef = { name: "Walker", hw: 3, h: 15, hp: 50, team: TEAM_SQUAD, art: "player.0.0" };
const walkers: Feature = {
  name: "walkers",
  install: (g) => {
    g.content.units.register("test.walker", WALKER);
  },
};

function setup(): BattleGame {
  return stagedGame({ extra: [walkers] });
}

/** Spawn a walker standing at column `col` and send it right. */
function walker(game: BattleGame, col: number, move = 1, y = GROUND_Y): number {
  const id = game.spawn("test.walker", col * 8 + 4, y);
  game.c.intent.move[ix(id)] = move;
  return id;
}

describe("battle locomotion", () => {
  it("runs at its speed over flat ground, feet on the surface", () => {
    const g = setup();
    const id = walker(g, 125);
    runHeadless(g.world, 2);
    const p = at(g, id);
    expect(p.x).toBeGreaterThan(125 * 8 + 4 + DEFAULT_MOVE.speed * 2 - 4);
    expect(p.x).toBeLessThan(125 * 8 + 4 + DEFAULT_MOVE.speed * 2 + 4);
    expect(p.y).toBe(GROUND_Y);
    expect(g.core.body.flags[ix(id)] & BODY_GROUNDED).toBeTruthy();
  });

  it("steps up a one-cell ledge without jumping", () => {
    const g = setup();
    fill(g.terrain, 130, GROUND - 1, 40, 1, M.DIRT);
    const id = walker(g, 125);
    let minVy = 0;
    runHeadless(g.world, 1.5, { onTick: () => void (minVy = Math.min(minVy, g.core.vel.vy[ix(id)])) });
    const p = at(g, id);
    expect(p.x).toBeGreaterThan(131 * 8);
    expect(p.y).toBe(GROUND_Y - 8);
    expect(minVy).toBeGreaterThan(-50);
  });

  it("jumps a two-cell ledge and climbs a wall too tall to jump", () => {
    const g = setup();
    fill(g.terrain, 130, GROUND - 2, 10, 2, M.STONE);
    fill(g.terrain, 140, GROUND - 8, 10, 8, M.STONE);
    const id = walker(g, 125);
    let climbed = false;
    let top = GROUND_Y;
    const watch = () => {
      climbed ||= g.c.motor.mode[ix(id)] === MODE_CLIMB;
      top = Math.min(top, at(g, id).y);
    };
    runHeadless(g.world, 2, { onTick: watch });
    expect(at(g, id).y).toBeLessThanOrEqual(GROUND_Y - 16);
    runHeadless(g.world, 4, { onTick: watch });
    expect(climbed).toBe(true);
    // Over the top of the wall and down its far side
    expect(top).toBeLessThanOrEqual(GROUND_Y - 64);
    expect(at(g, id).x).toBeGreaterThan(150 * 8);
    expect(at(g, id).y).toBe(GROUND_Y);
  });

  it("jumps a narrow pit instead of falling in", () => {
    const g = setup();
    clear(g.terrain, 130, GROUND, 3, 10);
    const id = walker(g, 124);
    let lowest = 0;
    runHeadless(g.world, 2, { onTick: () => void (lowest = Math.max(lowest, at(g, id).y)) });
    expect(at(g, id).x).toBeGreaterThan(133 * 8);
    expect(lowest).toBeLessThanOrEqual(GROUND_Y);
  });

  it("climbs a ladder up a cliff and walks off the top", () => {
    const g = setup();
    fill(g.terrain, 131, GROUND - 14, 12, 14, M.STONE);
    fill(g.terrain, 130, GROUND - 14, 1, 14, M.LADDER);
    const id = walker(g, 126);
    let onLadder = false;
    runHeadless(g.world, 3, { onTick: () => void (onLadder ||= g.c.motor.mode[ix(id)] === MODE_CLIMB) });
    expect(onLadder).toBe(true);
    const p = at(g, id);
    expect(p.y).toBe(GROUND_Y - 14 * 8);
    expect(p.x).toBeGreaterThan(132 * 8);
  });

  it("swims with its head out of the water and climbs out at the far bank", () => {
    const g = setup();
    clear(g.terrain, 128, GROUND, 14, 6);
    fill(g.terrain, 128, GROUND, 14, 6, M.WATER);
    const id = walker(g, 124);
    let swimT = 0;
    let headOut = true;
    runHeadless(g.world, 6, {
      onTick: () => {
        const i = ix(id);
        if (g.c.motor.mode[i] !== MODE_SWIM) return;
        // Once it has bobbed up from the plunge, it floats with its head out of the water
        const head = g.core.pos.y[i] - 15 - GROUND_Y;
        if ((swimT += TICK) > 0.8 && head > 1) headOut = false;
      },
    });
    expect(swimT).toBeGreaterThan(1);
    expect(headOut).toBe(true);
    expect(at(g, id).x).toBeGreaterThan(142 * 8);
    expect(at(g, id).y).toBe(GROUND_Y);
  });

  it("drifts down under a parachute no faster than its chute allows, and lands", () => {
    const g = setup();
    const id = g.spawn("test.walker", 1004, GROUND_Y - 300, { chute: true });
    let fastest = 0;
    runHeadless(g.world, 8, { onTick: () => void (fastest = Math.max(fastest, g.core.vel.vy[ix(id)])) });
    expect(fastest).toBeLessThanOrEqual(DEFAULT_MOVE.chuteFall + 1e-3);
    expect(g.c.motor.mode[ix(id)]).toBe(MODE_WALK);
    expect(at(g, id).y).toBe(GROUND_Y);
  });

  it("is flung by a blast with no control until it lands, then walks on", () => {
    const g = setup();
    const id = walker(g, 125, 0);
    runHeadless(g.world, 0.2);
    knock(g, id, 150, -220);
    expect(g.c.motor.mode[ix(id)]).toBe(MODE_KNOCK);
    g.c.intent.move[ix(id)] = -1;
    runHeadless(g.world, 0.3);
    // Still going the way the blast threw it, whatever it wants
    expect(g.core.vel.vx[ix(id)]).toBeGreaterThan(0);
    runHeadless(g.world, 2);
    expect(g.c.motor.mode[ix(id)]).toBe(MODE_WALK);
    expect(g.core.vel.vx[ix(id)]).toBeLessThan(0);
  });

  it("never falls through a one-cell floor, however fast", () => {
    const g = setup();
    fill(g.terrain, 120, GROUND - 40, 20, 1, M.STONE);
    const id = g.spawn("test.walker", 1004, GROUND_Y - 400);
    const i = ix(id);
    g.c.motor.maxFall[i] = 4000;
    g.core.vel.vy[i] = 3000;
    runHeadless(g.world, 1);
    expect(at(g, id).y).toBe((GROUND - 40) * 8);
  });

  it("is pushed out when terrain fills in on top of it", () => {
    const g = setup();
    const id = walker(g, 125, 0);
    runHeadless(g.world, 0.2);
    fill(g.terrain, 124, GROUND - 3, 3, 3, M.DIRT);
    expect(buried(g)).toBe(id);
    runHeadless(g.world, TICK);
    expect(buried(g)).toBe(0);
    expect(at(g, id).y).toBeLessThanOrEqual(GROUND_Y - 24);
  });

  it("never ends a tick inside solid terrain over a long random run (soak)", () => {
    const g = new BattleGame({ seed: 11 });
    const r = g.world.rng.get("test");
    let checks = 0;
    let offender = 0;
    runHeadless(g.world, 60, {
      onTick: (_w, tick) => {
        // Every tick ends with nobody inside the ground, whatever was done to it since the last one
        checks++;
        offender ||= buried(g);
        // Rough up the ground around the camera: blasts, and sand poured from above onto whoever is there
        if (tick % 9 === 0) {
          const x = g.camera.left + r.next() * (g.camera.right - g.camera.left);
          g.terrain.carve(x, g.terrain.surfaceY(x, 0) + r.range(-8, 16), r.range(8, 30), 6);
        }
        if (tick % 23 === 0) {
          const x = g.camera.x + r.range(-120, 120);
          const y = g.terrain.surfaceY(x, 0) - r.range(10, 40);
          for (let k = 0; k < 12; k++) g.terrain.set(Math.floor(x / 8) + (k % 4), Math.floor(y / 8) - (k >> 2), M.SAND);
        }
      },
    });
    expect(checks).toBe(1800);
    expect(offender).toBe(0);
    expect(g.stats.kills).toBeGreaterThan(0);
  });
});
