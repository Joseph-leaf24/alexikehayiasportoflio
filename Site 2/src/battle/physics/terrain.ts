/**
 * The terrain's place in the tick ("terrain" phase): keep columns generated ahead of the camera, step the cellular
 * physics and structures, crush units under falling chunks, push everyone out of whatever moved onto them, and turn
 * the terrain's events into effects (steam, flames catching, a building's impact dust and shake).
 */
import { EV_CRUSH, EV_DETACH, EV_IGNITE, EV_IMPACT, EV_STEAM, indexOf, TEAM_ALL, TILE, type System } from "@/engine";
import type { BattleGame } from "../game/BattleGame";
import { UNIT_DEAD } from "../game/components";
import { hurt } from "./damage";

/** Columns generated past the right edge of the view. */
export const GENERATE_AHEAD = 96;
/** A falling chunk crushes what it lands on above this speed, px/s. */
export const CRUSH_SPEED = 40;
/** Crushing damage: past spawn protection, through any infantry's health. */
export const CRUSH_DAMAGE = 80;

export function terrainSystems(game: BattleGame): System[] {
  const events: number[] = [];
  const under: number[] = [];
  return [
    {
      name: "terrain.ensure",
      phase: "terrain",
      order: -10,
      run() {
        game.terrain.ensure(Math.ceil(game.camera.right / TILE) + GENERATE_AHEAD);
      },
    },
    { name: "terrain.step", phase: "terrain", order: 0, run: (_w, dt) => game.terrain.step(dt) },
    {
      name: "terrain.crush",
      phase: "terrain",
      order: 50,
      run(w) {
        const { pos, body } = game.core;
        const { unit } = game.c;
        for (const ch of game.terrain.falling) {
          if (ch.vy < CRUSH_SPEED) continue;
          const x0 = ch.col * TILE;
          const n = w.spatial.rect(x0, ch.y, x0 + ch.pxW, ch.y + ch.pxH + 2, TEAM_ALL, under);
          for (let k = 0; k < n; k++) {
            const id = under[k];
            const i = indexOf(id);
            if (unit.$.sparse[i] < 0 || unit.flags[i] & UNIT_DEAD) continue;
            // Under one of the chunk's columns, and reached by that column's lowest cell
            const top = pos.y[i] - body.h[i];
            const c0 = Math.max(0, Math.floor((pos.x[i] - body.hw[i]) / TILE) - ch.col);
            const c1 = Math.min(ch.w - 1, Math.floor((pos.x[i] + body.hw[i] - 1e-3) / TILE) - ch.col);
            for (let lc = c0; lc <= c1; lc++) {
              const b = ch.bottom[lc];
              if (b < 0) continue;
              const face = ch.y + (b + 1) * TILE;
              if (face > top && ch.y < pos.y[i]) {
                hurt(game, id, CRUSH_DAMAGE);
                break;
              }
            }
          }
        }
      },
    },
    {
      name: "terrain.events",
      phase: "terrain",
      order: 200,
      run() {
        const n = game.terrain.drainEvents(events);
        for (let e = 0; e < n; e++) {
          const type = events[e * 4],
            x = events[e * 4 + 1] * TILE + TILE / 2,
            y = events[e * 4 + 2] * TILE,
            v = events[e * 4 + 3];
          if (type === EV_STEAM) game.effect("steam", x, y);
          else if (type === EV_IGNITE) game.effect("flame", x, y);
          else if (type === EV_IMPACT) {
            game.effect("impact.dust", x, y, { count: Math.min(3, 0.5 + v / 200) });
            game.shake(Math.min(10, 2 + v / 60));
          } else if (type === EV_DETACH || type === EV_CRUSH) game.effect("impact.dust", x, y, { count: 0.5 });
        }
      },
    },
  ];
}
