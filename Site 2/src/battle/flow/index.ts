/**
 * The level loop (minimal until B4 ports phases, intensity waves, finales, the white-out and cards): start a level
 * once the game has booted, keep its clock, announce it, and end it on request.
 *
 * Systems: flow.clock (director).
 */
import type { BattleGame } from "../game/BattleGame";

export interface FlowOptions {
  /** The first level's name, shown under its banner. */
  name?: string;
}

declare module "../game/options" {
  interface FeatureOptions {
    flow?: FlowOptions;
  }
}

/** Start level `no`: its clock, the `levelStart` event (delivered at once), and the "Level n" banner. */
export function startLevel(game: BattleGame, no: number, name: string): void {
  const L = game.level;
  L.no = no;
  L.name = name;
  L.t = 0;
  game.world.events.dispatch("levelStart", { no, name, time: 0 });
  game.banner(`Level ${no}`, name, 0xffffffff, { hold: 2.4, priority: 10 });
}

/** End the level in play: the `levelEnd` event, with how long it lasted. */
export function endLevel(game: BattleGame): void {
  const L = game.level;
  game.world.events.dispatch("levelEnd", { no: L.no, name: L.name, time: L.t });
}

export function install(game: BattleGame): void {
  game.world.addSystem({
    name: "flow.clock",
    phase: "director",
    order: -100,
    run(_w, dt) {
      game.level.t += dt;
    },
  });
  game.onBoot(() => startLevel(game, 1, game.options.feature.flow?.name ?? "Jungle landing"));
}
