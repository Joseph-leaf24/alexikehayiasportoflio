/**
 * The enemies feature (placeholder until B3 ports the Regime army, its vehicles and bosses): troopers spawned
 * ahead of the squad in groups, some by parachute, within the enemy budget, and cleared away once far behind.
 *
 * Systems: enemies.spawn (spawn, every 15 ticks), enemies.cull (late, every 30 ticks).
 */
import { TILE } from "@/engine";
import type { BattleGame } from "../game/BattleGame";
import { UNIT_DEAD } from "../game/components";
import { TEAM_SQUAD } from "../game/teams";
import { REGIME_BULLET, REGIME_GRENADE, REGIME_RIFLE, TROOPER } from "./infantry";

export { INFANTRY_AI, REGIME_BULLET, REGIME_GRENADE, REGIME_RIFLE, TROOPER } from "./infantry";

export interface EnemyOptions {
  /** Spawn troopers ahead of the squad (default true; tests turn it off to stage their own fights). */
  spawn?: boolean;
  /** Troopers wanted per 100 px of view width (default 2.5). */
  density?: number;
}

declare module "../game/options" {
  interface FeatureOptions {
    enemies?: EnemyOptions;
  }
}

const SPAWN_EVERY = 15;

export function install(game: BattleGame): void {
  const c = game.content;
  c.weapons.register("regime.rifle", REGIME_RIFLE);
  c.projectiles.register("regime.bullet", REGIME_BULLET);
  c.projectiles.register("regime.grenade", REGIME_GRENADE);
  c.units.register("regime.trooper", TROOPER);
  const opts = game.options.feature.enemies ?? {};
  const { pos, team } = game.core;
  const { unit } = game.c;
  const q = game.world.query(unit, pos, team);

  if (opts.spawn !== false)
    game.world.addSystem({
      name: "enemies.spawn",
      phase: "spawn",
      every: SPAWN_EVERY,
      run() {
        const cam = game.camera;
        const r = game.rng.spawn;
        // How many are already in the fight ahead of the squad's back
        const n = q.collect(true),
          idx = q.idx;
        let ahead = 0;
        for (let k = 0; k < n; k++) {
          const i = idx[k];
          if (team.id[i] !== TEAM_SQUAD && !(unit.flags[i] & UNIT_DEAD) && pos.x[i] > cam.left - 60) ahead++;
        }
        const want = Math.min(
          game.budgets.maxEnemies,
          Math.round(((cam.viewW / cam.zoom) * (opts.density ?? 2.5)) / 100)
        );
        if (ahead >= want || game.now < 1.5) return;
        const group = Math.min(want - ahead, 2 + r.int(3));
        const chute = r.chance(0.25);
        const x0 = cam.right + 40 + r.next() * 160;
        for (let g = 0; g < group; g++) {
          const x = Math.floor((x0 + g * (10 + r.next() * 14)) / TILE) * TILE + TILE / 2;
          const y = chute ? cam.top - 30 - g * 12 : game.terrain.surfaceY(x, 0);
          game.spawn("regime.trooper", x, y, { facing: -1, chute });
        }
      },
    });

  game.world.addSystem({
    // Far behind the squad nobody will ever see them again: gone, without counting as kills
    name: "enemies.cull",
    phase: "late",
    every: 30,
    run(w) {
      const left = game.camera.left - 320;
      const n = q.collect(true),
        idx = q.idx;
      for (let k = 0; k < n; k++) {
        const i = idx[k];
        if (team.id[i] !== TEAM_SQUAD && pos.x[i] < left) w.destroy(w.idOf(i));
      }
    },
  });
}
