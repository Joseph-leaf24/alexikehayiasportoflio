/** Shared set-ups for the battle tests: whole battles, and quiet ones on hand-built ground for staged fights. */
import { indexOf, materials as M, ROWS, type Terrain } from "@/engine";
import { BattleGame, type BattleOptions, type Feature } from "@/battle";
import { install as art } from "@/battle/art";
import { install as enemies } from "@/battle/enemies";
import { install as flow } from "@/battle/flow";
import { install as physics } from "@/battle/physics";
import { install as render } from "@/battle/render";
import { install as squad } from "@/battle/squad";

/** Ground row of the staged worlds (the old engine's BASE_ROW). */
export const GROUND = 148;
export const GROUND_Y = GROUND * 8;

/** Flat ground from `GROUND` down: grass, dirt, stone, bedrock. */
export const flatGround: Feature = {
  name: "flat",
  install(game) {
    game.level.generate = (t, from, to) => {
      for (let c = from; c < to; c++)
        for (let r = GROUND; r < ROWS; r++)
          t.set(c, r, r === ROWS - 1 ? M.BEDROCK : r === GROUND ? M.GRASS : r < GROUND + 5 ? M.DIRT : M.STONE);
    };
    game.world.events.on("levelStart", () => game.terrain.ensure(400));
  },
};

/**
 * A battle with no squad, no enemy spawns and no level generator of its own: flat ground, the trooper defs, and
 * whatever `extra` features add. The camera is parked over x = 1,200.
 */
export function stagedGame(opts: BattleOptions & { extra?: Feature[]; withSquad?: boolean } = {}): BattleGame {
  const features: Feature[] = [
    { name: "art", install: art },
    { name: "render", install: render },
    { name: "physics", install: physics },
    flatGround,
    ...(opts.withSquad ? [{ name: "squad", install: squad }] : []),
    { name: "enemies", install: enemies },
    ...(opts.extra ?? []),
    { name: "flow", install: flow },
  ];
  const game = new BattleGame({
    seed: 1,
    ...opts,
    features,
    feature: { ...opts.feature, enemies: { spawn: false, ...opts.feature?.enemies } },
  });
  if (!opts.withSquad) game.camera.place(1200, GROUND_Y - 60);
  return game;
}

/** Fill a rectangle of cells. */
export function fill(t: Terrain, c0: number, r0: number, w: number, h: number, mat: number, structural = false): void {
  for (let c = c0; c < c0 + w; c++) for (let r = r0; r < r0 + h; r++) t.set(c, r, mat, { structural });
}

/** Clear a rectangle to air. */
export const clear = (t: Terrain, c0: number, r0: number, w: number, h: number): void => fill(t, c0, r0, w, h, M.AIR);

/** Entity index shorthand. */
export const ix = indexOf;

/** A unit's feet position. */
export function at(game: BattleGame, id: number): { x: number; y: number } {
  return { x: game.core.pos.x[ix(id)], y: game.core.pos.y[ix(id)] };
}

/** Whether any live unit's box overlaps solid terrain. Returns the first offender's id, or 0. */
export function buried(game: BattleGame): number {
  const { pos, body } = game.core;
  const u = game.c.unit;
  for (let k = 0; k < u.$.size; k++) {
    const i = u.$.dense[k];
    if (game.terrain.overlapsSolid(pos.x[i], pos.y[i], body.hw[i], body.h[i])) return game.world.idOf(i);
  }
  return 0;
}
