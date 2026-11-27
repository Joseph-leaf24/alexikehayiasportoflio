/**
 * Projectiles in flight ("projectiles" phase, after the spatial hash has this tick's positions). Each tick a shot's
 * whole path is swept: a terrain raycast and a spatial `segment` query along the same line, hits taken nearest
 * first, so no speed tunnels through a wall or a body. On a body: damage, then pierce on, explode, or stop. On
 * terrain: bounce (grenades), explode, or chip the cell and stop. Team masks keep a side's shots off its own.
 */
import { createRayHit, LEVEL_H, TEAM_ALL, type System } from "@/engine";
import type { BattleGame } from "../game/BattleGame";
import { UNIT_DEAD } from "../game/components";
import type { ProjectileDef } from "../game/defs";
import { hostileTo } from "../game/teams";
import { emitDef } from "../render/effects";
import { hurt } from "./damage";

const ray = createRayHit();
const bodies: number[] = [];

/** End a shot: its blast if it has one, its expiry hook, and the entity. */
function finish(game: BattleGame, id: number, i: number, def: ProjectileDef, x: number, y: number, blast: boolean) {
  const p = game.c.projectile;
  if (blast && def.explode) game.explode(x, y, def.explode, p.owner[i], game.core.team.id[i]);
  def.onExpire?.(game, id);
  game.world.destroy(id);
}

export function projectileSystem(game: BattleGame): System {
  const { pos, vel, team } = game.core;
  const { projectile: p, unit } = game.c;
  const q = game.world.query(pos, vel, p);
  /** Only bodies still in the fight take hits (the hash may hold units killed earlier this tick). */
  const live = (_id: number, i: number) => unit.$.sparse[i] >= 0 && (unit.flags[i] & UNIT_DEAD) === 0;
  return {
    name: "projectiles.step",
    phase: "projectiles",
    run(w, dt) {
      const t = game.terrain;
      const defs = game.content.projectiles.defs;
      const links = game.links;
      const cam = game.camera;
      const n = q.collect(true),
        idx = q.idx;
      for (let k = 0; k < n; k++) {
        const i = idx[k];
        const id = w.idOf(i);
        const def = defs[p.def[i] - 1];
        const explodes = def.explode !== undefined;
        const life = (p.life[i] -= dt);
        if (life <= 0) {
          finish(game, id, i, def, pos.x[i], pos.y[i], explodes && (def.fuse === true || def.impactExplode !== false));
          continue;
        }
        def.update?.(game, id, i, dt);
        if (w.doomed(id)) continue;
        let vx = vel.vx[i],
          vy = vel.vy[i] + (def.gravity ?? 0) * dt;
        if (def.accel) {
          const f = 1 + def.accel * dt;
          vx *= f;
          vy *= f;
        }
        if (def.drag) {
          const f = Math.max(0, 1 - def.drag * dt);
          vx *= f;
          vy *= f;
        }
        const x0 = pos.x[i],
          y0 = pos.y[i];
        const dx = vx * dt,
          dy = vy * dt;
        const len = Math.hypot(dx, dy);
        // Terrain along the path (ghosts pass through), then bodies before that point
        let tHit = 2;
        if (!def.ghost && len > 0 && t.raycast(x0, y0, dx, dy, len, ray)) tHit = ray.dist / len;
        const mask = def.friendly ? TEAM_ALL : hostileTo(team.id[i]);
        const width = (def.radius ?? 1) * 2;
        const m = len > 0 ? w.spatial.segment(x0, y0, x0 + dx, y0 + dy, width, mask, bodies, live) : 0;
        let done = false;
        for (let b = 0; b < m && !done; b++) {
          const ht = w.spatial.hitT[b];
          if (ht > tHit) break;
          const target = bodies[b];
          if (target === p.last[i] || target === p.owner[i]) continue;
          if (def.onHit?.(game, id, target) === false) continue;
          const hx = x0 + dx * ht,
            hy = y0 + dy * ht;
          if (p.dmg[i] > 0) {
            hurt(game, target, p.dmg[i], p.owner[i]);
            game.effect("hit.flesh", hx, hy, { vx: vx * 0.1, vy: vy * 0.1 });
          }
          p.last[i] = target;
          // Explosive shots go off on whoever they reach, fused grenades included
          if (explodes && def.impactExplode !== false) {
            finish(game, id, i, def, hx, hy, true);
            done = true;
          } else if (p.pierce[i] > 0) p.pierce[i]--;
          else {
            finish(game, id, i, def, hx, hy, false);
            done = true;
          }
        }
        if (done) continue;
        if (tHit <= 1) {
          // Back off the face it struck, so a bounce or a blast starts in the open
          const ux = dx / len,
            uy = dy / len;
          const hx = ray.x - ux * 0.5 + ray.nx * 0.5,
            hy = ray.y - uy * 0.5 + ray.ny * 0.5;
          const maxB = def.bounces ?? 255;
          if (def.bounce !== undefined && p.bounces[i] < maxB) {
            if (ray.nx !== 0) vx = -vx * def.bounce;
            if (ray.ny !== 0) vy = -vy * def.bounce * 0.9;
            vx *= 0.85;
            p.bounces[i]++;
            // A ricochet throws its impact puff too (saw blades spark)
            const bfx = links.projectileImpact[p.def[i]];
            if (bfx) emitDef(game, game.content.effects.get(bfx), hx, hy, { vx: vx * 0.05 });
            pos.x[i] = hx;
            pos.y[i] = hy;
            vel.vx[i] = vx;
            vel.vy[i] = vy;
            continue;
          }
          if (explodes && def.impactExplode !== false && !def.fuse) {
            finish(game, id, i, def, hx, hy, true);
            continue;
          }
          if (def.fuse) {
            // A fused shot that can't bounce any more lies where it struck until it goes off
            pos.x[i] = hx;
            pos.y[i] = hy;
            vel.vx[i] = vel.vy[i] = 0;
            continue;
          }
          if (def.chip) {
            const broke = t.damage(ray.col, ray.row, def.chip);
            if (broke) game.effect("death.trooper", ray.col * 8 + 4, ray.row * 8 + 4, { count: 0.5 });
          }
          const fx = links.projectileImpact[p.def[i]];
          if (fx) emitDef(game, game.content.effects.get(fx), hx, hy, { vx: -vx * 0.05 });
          finish(game, id, i, def, hx, hy, false);
          continue;
        }
        pos.x[i] = x0 + dx;
        pos.y[i] = y0 + dy;
        vel.vx[i] = vx;
        vel.vy[i] = vy;
        // Far off screen or out of the world: gone
        const nx = pos.x[i];
        if (nx < cam.left - 320 || nx > cam.right + 320 || pos.y[i] > LEVEL_H + 40 || pos.y[i] < cam.top - 480) {
          w.destroy(id);
          continue;
        }
        const trail = links.projectileTrail[p.def[i]];
        if (trail) {
          p.trail[i] += dt * (def.trailRate ?? 30);
          if (p.trail[i] >= 1) {
            const c = Math.floor(p.trail[i]);
            p.trail[i] -= c;
            emitDef(game, game.content.effects.get(trail), nx, pos.y[i], { count: c, vx: vx * 0.1, vy: vy * 0.1 });
          }
        }
      }
    },
  };
}
