/** The core's own systems: lifetimes, the spatial hash rebuild and an optional plain velocity integrator. */

import { BODY_COLLIDES_TERRAIN } from "./components";
import type { Phase, System } from "./scheduler";
import type { World } from "./world";

export interface CoreSystemOptions {
  /** Count `life` down and destroy at 0 (default on, in "death"). */
  life?: boolean;
  /** Rebuild the spatial hash once per tick (default on, last in `spatialPhase`). */
  spatial?: boolean;
  /** Default "physics": after movement, so projectiles, damage and next tick's AI see where bodies ended up. */
  spatialPhase?: Phase;
  /** pos += vel × dt for entities whose body doesn't collide with terrain (default off: games bring physics). */
  motion?: boolean;
}

/** f32 rounding can leave a lifetime of exactly n ticks a hair above 0 after n ticks; this absorbs it. */
const LIFE_EPS = 1e-4;

/** Counts `life.t` down and destroys the entity once it reaches 0. */
export function lifeSystem(world: World): System {
  const { life } = world.core;
  const q = world.query(life);
  return {
    name: "core.life",
    phase: "death",
    order: -1000,
    run(w, dt) {
      const n = q.collect(true),
        idx = q.idx,
        t = life.t;
      for (let k = 0; k < n; k++) {
        const i = idx[k];
        const left = t[i] - dt;
        t[i] = left;
        if (left <= LIFE_EPS) w.destroy(w.idOf(i));
      }
    },
  };
}

/** Rebuilds `world.spatial` from bodies. */
export function spatialSystem(phase: Phase = "physics"): System {
  return {
    name: "core.spatial",
    phase,
    order: 1_000_000,
    run(w) {
      w.spatial.rebuild(w);
    },
  };
}

/** Integrates velocity for free-moving entities (no body, or a body without BODY_COLLIDES_TERRAIN). */
export function motionSystem(world: World): System {
  const { pos, vel, body } = world.core;
  const q = world.query(pos, vel);
  return {
    name: "core.motion",
    phase: "motion",
    order: 1000,
    run(_w, dt) {
      const n = q.collect(),
        idx = q.idx,
        x = pos.x,
        y = pos.y,
        vx = vel.vx,
        vy = vel.vy,
        bsparse = body.$.sparse,
        bf = body.flags;
      for (let k = 0; k < n; k++) {
        const i = idx[k];
        if (bsparse[i] >= 0 && bf[i] & BODY_COLLIDES_TERRAIN) continue;
        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;
      }
    },
  };
}

export function installCoreSystems(world: World, opts: CoreSystemOptions = {}): void {
  if (opts.life !== false) world.addSystem(lifeSystem(world));
  if (opts.spatial !== false) world.addSystem(spatialSystem(opts.spatialPhase));
  if (opts.motion) world.addSystem(motionSystem(world));
}
