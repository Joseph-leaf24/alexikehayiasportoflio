/**
 * The level's barrels and mines (the old `barrels`): oil drums, fuel tanks, ammo boxes, propane tanks and gas
 * cylinders (which take off on a jet of flame first), chemical canisters, TNT, tankers and land mines, placed where
 * the old generator put them as they come into view. Shot to nothing, or caught in any blast, one goes off a moment
 * later and sets off its neighbours in a chain; a mine goes off when a squad member steps on it. The squad shoots a
 * barrel with enemies next to it (`findTarget`).
 *
 * Barrels stand on the Regime's side, so only the squad's shots hit them; mines are neutral, which nobody aims at.
 * A lit barrel stays standing (not a corpse) until its fuse runs out, then blows up for whoever lit it.
 *
 * System: squad.barrels (death): fuses, mines underfoot, and placing the level's barrels.
 */
import { drawTanker } from "@/components/pixel/art";
import { indexOf, PARTICLE_EMISSIVE, resource, TEAM_ALL, TILE, type System } from "@/engine";
import { registerShapeArt } from "../art";
import type { BattleGame } from "../game/BattleGame";
import { TAG_PROP, type BlastSpec, type EffectDef, type UnitDef, type UnitDrawFn } from "../game/defs";
import { TEAM_NEUTRAL, TEAM_REGIME } from "../game/teams";
import { LEVEL } from "../level";
import { knock } from "../physics/locomotion";
import { EFFECTS as LIBRARY } from "../render/library";
import { LAYER } from "../render/layers";
import { SQUAD } from "./squad";

/** A barrel kind, from the old tables (`SIZE`, the hp at spawn, `BARREL_BLAST`, the blast palettes). */
interface Barrel {
  readonly hw: number;
  readonly h: number;
  readonly hp: number;
  readonly blast: BlastSpec;
  /** Takes off on a jet of flame when lit, and goes up near the top of its climb. */
  readonly rocket?: boolean;
  /** Throws up a shower of sparks too (ammo). */
  readonly sparks?: boolean;
  /** Too heavy for a blast to throw (a tanker). */
  readonly heavy?: boolean;
}

const KINDS: Readonly<Record<string, Barrel>> = {
  barrel: { hw: 3, h: 8, hp: 1, blast: { r: 20 } },
  fuel: { hw: 7, h: 11, hp: 4, blast: { r: 34, fx: "barrel.napalm", ignite: 0.4 } },
  tanker: { hw: 19, h: 17, hp: 8, heavy: true, blast: { r: 60, fx: "barrel.napalm", ignite: 0.6 } },
  ammo: { hw: 4, h: 6, hp: 1, sparks: true, blast: { r: 16 } },
  propane: { hw: 3, h: 10, hp: 1, rocket: true, blast: { r: 24, fx: "barrel.gas" } },
  canister: { hw: 3, h: 5, hp: 1, blast: { r: 18, fx: "barrel.chem" } },
  cylinder: { hw: 2, h: 11, hp: 1, rocket: true, blast: { r: 22, fx: "barrel.gas" } },
  mine: { hw: 3, h: 2, hp: 1, blast: { r: 22 } },
  tnt: { hw: 4, h: 8, hp: 1, blast: { r: 32 } },
};

/** The old fireball palettes: gas, chemical and napalm blasts. */
const fireball = (colors: readonly number[], light: number): EffectDef => ({
  ...LIBRARY["blast.fire"],
  colors,
  light: { color: light, radius: 70, intensity: 1.8, life: 0.3 },
});
const EFFECTS: Readonly<Record<string, EffectDef>> = {
  "barrel.gas": fireball([0xffffffff, 0xc8f0ffff, 0x7ac8ffff, 0x4a7affff], 0x7ac8ff),
  "barrel.chem": fireball([0xf0ffc0ff, 0xb8ff6aff, 0x5aff4aff, 0x2ad86aff], 0x8aff4a),
  "barrel.napalm": fireball([0xfff0b0ff, 0xffc23aff, 0xff7a1aff, 0xd82a1aff], 0xff7a1a),
};
const AMMO_SPARKS = [0xffe27aff, 0xff9a2aff, 0xffffffff];
const TANKER_ART = "barrel.tanker";

interface Fuse {
  /** Seconds left. */
  t: number;
  /** Who lit it: its blast is theirs. */
  owner: number;
}

class Barrels {
  /** Lit barrels by entity. */
  readonly lit = new Map<number, Fuse>();
  readonly mines: number[] = [];
  placeT = 0;
}

const BARRELS = resource<Barrels>("barrels");

/** Each barrel def's kind, for the fuse and the blast. */
const kindOf = new Map<UnitDef, Barrel>();

/** Light a barrel's fuse (`delay` s); a propane tank or gas cylinder takes off first. */
function light(game: BattleGame, id: number, owner: number, delay: number): void {
  const lit = game.world.res(BARRELS).lit;
  if (lit.has(id)) return;
  const b = kindOf.get(game.unitDef(id));
  if (!b) return;
  const fuse: Fuse = { t: delay, owner };
  if (b.rocket) {
    // The valve blows: up on a jet of flame, going off near the top of the climb
    const r = game.rng.combat;
    fuse.t = 0.55 + r.next() * 0.35;
    game.c.motor.grav[indexOf(id)] = 0.35;
    knock(game, id, (r.next() - 0.5) * 60, -330 - r.next() * 90);
  }
  lit.set(id, fuse);
}

/** The fuse ran out: the barrel goes up, for whoever lit it, and anything near catches. */
function detonate(game: BattleGame, id: number, owner: number): void {
  const b = kindOf.get(game.unitDef(id)) as Barrel;
  const i = indexOf(id);
  const x = game.core.pos.x[i],
    y = game.core.pos.y[i] - b.h / 2;
  game.kill(id, owner, 0, 0);
  game.world.destroy(id);
  game.explode(x, y, b.blast, owner, TEAM_NEUTRAL);
  if (b.sparks) game.effect("squad.spark", x, y, { count: 7, colors: AMMO_SPARKS });
}

/** Shot down to nothing, it's lit rather than killed: it stands until it goes off. */
const onHurt = (game: BattleGame, id: number, dmg: number, source: number): number => {
  if (game.world.res(BARRELS).lit.has(id)) return 0;
  if (game.core.health.hp[indexOf(id)] - dmg > 0) return dmg;
  light(game, id, source, 0.05);
  return 0;
};

/** A lit barrel's fuse spark blinks; an armed mine's light blinks red. */
const drawFuse =
  (mine: boolean): UnitDrawFn =>
  (game, i, x, y, sink) => {
    const t = game.now;
    if (game.world.res(BARRELS).lit.has(game.world.idOf(i))) {
      if (Math.floor(t * 20) & 1) sink.particle(x, y - game.core.body.h[i] + 1, 1, 0xffffffff, PARTICLE_EMISSIVE);
    } else if (mine && Math.floor(t * 3 + x) % 3 === 0) sink.particle(x, y - 3, 1, 0xff3b3bff, PARTICLE_EMISSIVE);
  };

const STILL = { speed: 0, jump: 0, climb: 0, swim: false, autoJump: false, ladders: false };

/** Unit ids by the old spawn kind. */
const IDS: Readonly<Record<string, string>> = Object.fromEntries(Object.keys(KINDS).map((k) => [k, `barrel.${k}`]));

/** Mines underfoot, lit fuses burning down, and the level's barrels placed as they come into view. */
function barrelSystem(game: BattleGame): System {
  const { pos } = game.core;
  return {
    name: "squad.barrels",
    phase: "death",
    run(w, dt) {
      const st = w.res(BARRELS);
      const sq = w.res(SQUAD);
      for (let k = st.mines.length - 1; k >= 0; k--) {
        const m = st.mines[k];
        if (!game.fighting(m)) {
          st.mines.splice(k, 1);
          continue;
        }
        const mi = indexOf(m);
        for (const p of sq.players) {
          const i = indexOf(p);
          if (st.lit.has(m) || !game.fighting(p)) continue;
          if (Math.abs(pos.x[i] - pos.x[mi]) < 6 && Math.abs(pos.y[i] - pos.y[mi]) < 8) {
            // Stepped on: it goes off a moment later
            game.popup("!", pos.x[mi], pos.y[mi] - 12, 0xff3b3bff);
            light(game, m, 0, 0.12);
          }
        }
      }
      const fx = game.rng.fx;
      for (const [id, f] of st.lit) {
        if (!w.has(id, game.c.unit)) {
          st.lit.delete(id);
          continue;
        }
        const i = indexOf(id);
        const b = kindOf.get(game.unitDef(id)) as Barrel;
        // Flames off the top while it burns; a rocket's jet out of the bottom
        if (b.rocket) {
          game.effect("squad.jet", pos.x[i], pos.y[i] + 1, { count: 2 });
          game.effect("squad.smoke", pos.x[i], pos.y[i] + 3);
        } else if (fx.chance(0.6)) game.effect("squad.fire", pos.x[i], pos.y[i] - b.h);
        if ((f.t -= dt) > 0) continue;
        st.lit.delete(id);
        detonate(game, id, f.owner);
      }
      if ((st.placeT -= dt) > 0) return;
      st.placeT = 0.5;
      const level = w.tryRes(LEVEL);
      if (!level) return;
      const off = game.origin.cols * TILE;
      const cam = game.camera;
      const spawns = level.spawns;
      let n = 0;
      for (const s of spawns) {
        const def = IDS[s.kind];
        const x = s.x - off;
        if (def === undefined || x > cam.right + 120) spawns[n++] = s;
        else if (x > cam.left - 40) {
          const id = game.spawn(def, x, s.y);
          if (s.kind === "mine") st.mines.push(id);
        }
      }
      spawns.length = n;
    },
  };
}

export function installBarrels(game: BattleGame): void {
  const c = game.content;
  for (const [id, def] of Object.entries(EFFECTS)) c.effects.register(id, def);
  registerShapeArt(c, TANKER_ART, { w: 41, h: 20, ox: 20, oy: 20, paint: (ctx) => drawTanker(ctx, 0, 0, false) });
  for (const [kind, b] of Object.entries(KINDS)) {
    const def: UnitDef = {
      name: kind,
      hw: b.hw,
      h: b.h,
      hp: b.hp,
      team: kind === "mine" ? TEAM_NEUTRAL : TEAM_REGIME,
      art: kind === "tanker" ? TANKER_ART : `prop.${kind}`,
      pose: () => 0,
      layer: LAYER.PROPS,
      tags: TAG_PROP,
      move: STILL,
      onHurt,
      draw: drawFuse(kind === "mine"),
    };
    kindOf.set(def, b);
    c.units.register(IDS[kind], def);
  }
  game.world.setRes(BARRELS, new Barrels());
  game.world.addSystem(barrelSystem(game));
  // Any blast lights the barrels in reach, a moment apart so a row goes up in a ripple, and throws the light ones
  const hit: number[] = [];
  game.world.events.on("blast", (e) => {
    const n = game.world.spatial.inRadius(e.x, e.y, e.r + 30, TEAM_ALL, hit);
    const r = game.rng.combat;
    for (let k = 0; k < n; k++) {
      const id = hit[k];
      const b = game.fighting(id) ? kindOf.get(game.unitDef(id)) : undefined;
      const i = indexOf(id);
      if (!b || Math.hypot(game.core.pos.x[i] - e.x, game.core.pos.y[i] - 4 - e.y) > e.r + 10) continue;
      if (game.world.res(BARRELS).lit.has(id)) continue;
      light(game, id, e.source, 0.1 + r.next() * 0.3);
      if (!b.heavy && !b.rocket)
        knock(game, id, Math.sign(game.core.pos.x[i] - e.x) * (30 + r.next() * 60), -140 - r.next() * 120);
    }
  });
}
