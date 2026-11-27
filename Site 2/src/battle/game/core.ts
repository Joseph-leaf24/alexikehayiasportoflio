/**
 * The systems every battle has whatever features are installed: AI dispatch (each unit def's `ai` runs at its own
 * rate, staggered across units), delayed calls (`game.at`), budget counters, and the floating origin.
 */
import { TICK, type System } from "@/engine";
import type { BattleGame } from "./BattleGame";
import { UNIT_COUNTED, UNIT_DEAD, UNIT_HELD } from "./components";
import type { AiMachine, Think } from "./defs";

/** Compile a state machine spec into a `Think`; state numbers are kept in `ai.state` (0 = not started). */
export function machine(spec: AiMachine): Think {
  const names = Object.keys(spec.states);
  const index = new Map(names.map((n, k) => [n, k]));
  const states = names.map((n) => spec.states[n]);
  const initial = index.get(spec.initial);
  if (initial === undefined) throw new Error(`machine: no initial state "${spec.initial}"`);
  const go = (game: BattleGame, id: number, i: number, dt: number, k: number) => {
    game.c.ai.state[i] = k + 1;
    game.c.ai.t[i] = 0;
    states[k].enter?.(game, id, i, dt);
  };
  return (game, id, i, dt) => {
    const ai = game.c.ai;
    if (ai.state[i] === 0) go(game, id, i, dt, initial);
    ai.t[i] += dt;
    const cur = ai.state[i] - 1;
    const next = states[cur].run(game, id, i, dt);
    if (typeof next !== "string" || next === names[cur]) return;
    const k = index.get(next);
    if (k === undefined) throw new Error(`machine: no state "${next}"`);
    go(game, id, i, dt, k);
  };
}

/** Delayed calls in simulation time, run in order of their due time (then of scheduling). */
export class Timers {
  private due: number[] = [];
  private fns: ((game: BattleGame) => void)[] = [];

  add(at: number, fn: (game: BattleGame) => void): void {
    let k = this.due.length;
    while (k > 0 && this.due[k - 1] > at) k--;
    this.due.splice(k, 0, at);
    this.fns.splice(k, 0, fn);
  }

  get pending(): number {
    return this.due.length;
  }

  /** Run everything due by `now`; calls scheduled meanwhile for now or earlier run in the same pass. */
  run(game: BattleGame, now: number): void {
    let n = 0;
    while (this.due.length > 0 && this.due[0] <= now + 1e-9 && n++ < 1000) {
      this.due.shift();
      const fn = this.fns.shift() as (game: BattleGame) => void;
      fn(game);
    }
  }

  clear(): void {
    this.due.length = 0;
    this.fns.length = 0;
  }
}

function aiSystem(game: BattleGame): System {
  const { unit, intent } = game.c;
  const { kind } = game.core;
  const q = game.world.query(unit, intent, kind);
  let thinks: (Think | null)[] = [];
  let every = new Uint8Array(0);
  game.onBoot(() => {
    const units = game.content.units;
    thinks = [null];
    every = new Uint8Array(units.size + 1);
    for (let k = 1; k <= units.size; k++) {
      const spec = units.get(k).ai;
      thinks.push(spec ? (typeof spec.think === "function" ? spec.think : machine(spec.think)) : null);
      every[k] = Math.max(1, spec?.every ?? 3);
    }
  });
  return {
    name: "ai.think",
    phase: "ai",
    run(w) {
      const tick = w.time.tick;
      const n = q.collect(true),
        idx = q.idx,
        flags = unit.flags,
        def = kind.def;
      for (let k = 0; k < n; k++) {
        const i = idx[k];
        if (flags[i] & (UNIT_DEAD | UNIT_HELD)) continue;
        const d = def[i];
        const think = thinks[d];
        if (!think) continue;
        const ev = every[d];
        if (ev > 1 && (tick + i) % ev !== 0) continue;
        think(game, w.idOf(i), i, ev * TICK);
      }
    },
  };
}

export function installCore(game: BattleGame): void {
  const { world } = game;
  const { unit, projectile } = game.c;
  // Budget counters stay right however an entity goes: killed, despawned or swept up by a level reset
  world.onDestroy((_id, i) => {
    if (unit.$.sparse[i] >= 0 && unit.flags[i] & UNIT_COUNTED) {
      unit.flags[i] &= ~UNIT_COUNTED;
      game.stats.enemies--;
    }
    if (projectile.$.sparse[i] >= 0) game.stats.projectiles--;
  });
  world.addSystem(aiSystem(game));
  world.addSystem({
    name: "game.timers",
    phase: "late",
    order: 100,
    run: () => game.timers.run(game, game.now),
  });
  world.addSystem({
    name: "game.origin",
    phase: "late",
    order: 1000,
    run: () => game.origin.update(),
  });
}
