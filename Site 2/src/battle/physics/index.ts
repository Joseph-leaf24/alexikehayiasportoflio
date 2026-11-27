/**
 * The physics feature: weapons, locomotion, projectiles, the damage pipeline and the terrain's tick.
 *
 * Systems: weapons.fire (weapons); locomotion.move (motion); projectiles.step (projectiles); damage.hazards
 * (damage); damage.clocks (death); terrain.ensure -10, terrain.step 0, terrain.crush 50, locomotion.settle 100,
 * terrain.events 200 (terrain).
 */
import type { BattleGame } from "../game/BattleGame";
import { hazardSystem, unitClockSystem } from "./damage";
import { locomotionSystem, settleSystem } from "./locomotion";
import { projectileSystem } from "./projectiles";
import { terrainSystems } from "./terrain";
import { weaponsSystem } from "./weapons";

export { hurt, kill, playerOf, LETHAL } from "./damage";
export { explode } from "./explosions";
export { DEFAULT_MOVE, GRAVITY, knock, moveBox, moveUnit, pushOut, SUB_STEP, type Mover } from "./locomotion";
export { spawnUnit, unitIndex, type SpawnOptions } from "./spawn";
export { fireWeapon, lobAt, shoot, type ShotOptions } from "./weapons";
export { hitscan, type HitscanOptions, type HitscanResult } from "./hitscan";
export { CRUSH_DAMAGE, CRUSH_SPEED, GENERATE_AHEAD } from "./terrain";

export function install(game: BattleGame): void {
  const w = game.world;
  w.addSystem(weaponsSystem(game));
  w.addSystem(locomotionSystem(game));
  w.addSystem(projectileSystem(game));
  w.addSystem(hazardSystem(game));
  w.addSystem(unitClockSystem(game));
  for (const s of terrainSystems(game)) w.addSystem(s);
  w.addSystem(settleSystem(game));
}
