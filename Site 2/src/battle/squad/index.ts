/**
 * The squad feature: P1–P8 in their own colours and loadout sprites, each with the old weapon of its number, the
 * formation's advance, respawns by parachute, grenades and the jetpack, supply crates (`supply.ts`), the level's
 * barrels and mines (`barrels.ts`), and the camera's framing.
 *
 * Systems: squad.advance (director), squad.respawn and squad.supply (spawn), squad.rays (weapons), squad.barrels
 * (death); players think every tick through `ai.think`.
 */
import { indexOf, place, TILE } from "@/engine";
import type { BattleGame } from "../game/BattleGame";
import type { UnitDef } from "../game/defs";
import { TEAM_SQUAD } from "../game/teams";
import { playerArt } from "../art";
import { playerThink } from "./ai";
import { equip, installArms } from "./arms";
import { installBarrels } from "./barrels";
import { drawJetpack, installSupply } from "./supply";
import {
  advanceSystem,
  OFFSETS,
  openAir,
  PLAYER_HP,
  respawnSystem,
  SQUAD,
  Squad,
  SQUAD_SIZE,
  squadCamera,
  START_X,
} from "./squad";

export { findTarget, throwGrenade } from "./ai";
export { ARSENAL, equip } from "./arms";
export { OFFSETS, PLAYER_HP, queueDrop, release, SQUAD, Squad, SQUAD_SIZE, START_X } from "./squad";

/** One def for all eight: each player's sprite is its own (`unit.art`), by player id and weapon. */
export const PLAYER: UnitDef = {
  name: "Player",
  hw: 3,
  h: 15,
  hp: PLAYER_HP,
  team: TEAM_SQUAD,
  art: playerArt(0, 0),
  weapon: "squad.rifle",
  ai: { think: playerThink, every: 1 },
  draw: drawJetpack,
  // The old game's 3 a blast downed the squad too often with grenades about (B2's lethality tuning)
  blastTaken: 2,
  crushTaken: 2,
  corpse: 0,
};

/** Bring the eight in over the landing zone, under parachutes, staggered so they land one after another. */
function dropIn(game: BattleGame): void {
  const sq = game.world.res(SQUAD);
  const { player } = game.c;
  sq.advance = START_X;
  sq.holdT = 0;
  sq.queue.length = 0;
  for (let p = 0; p < SQUAD_SIZE; p++) {
    const x = START_X + OFFSETS[p];
    const ground = game.terrain.surfaceY(x, 0);
    const y = openAir(game, x, ground - 150 - p * 10, PLAYER.h);
    let id = sq.players[p];
    if (id === undefined) {
      id = game.spawn("squad.player", x, y, { chute: true });
      sq.players[p] = id;
      const i = indexOf(id);
      game.world.add(id, player);
      player.id[i] = p;
      // Player n starts with weapon n, as in the old game
      equip(game, id, p);
    } else place(game.world, id, x, y);
  }
  // Frame the landing zone straight away, then snap to the squad's framing on the first tick
  game.camera.place(START_X + game.camera.halfW * 0.36, game.terrain.surfaceY(START_X, 0) - TILE * 4);
  game.camera.snap = true;
}

export function install(game: BattleGame): void {
  game.world.setRes(SQUAD, new Squad());
  installArms(game);
  game.content.units.register("squad.player", PLAYER);
  game.world.addSystem(advanceSystem(game));
  game.world.addSystem(respawnSystem(game));
  installSupply(game);
  installBarrels(game);
  game.camera.follow = squadCamera;
  game.world.events.on("levelStart", () => dropIn(game));
  game.world.events.on("originShift", (e) => {
    const sq = game.world.res(SQUAD);
    sq.advance += e.dx;
    sq.holdAt += e.dx;
  });
  game.world.events.on("playerDown", (e) =>
    game.popup(`P${e.player + 1} DOWN`, e.x, e.y - 20, game.world.res(SQUAD).colors[e.player])
  );
}
