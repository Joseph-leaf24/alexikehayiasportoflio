/**
 * What the squad is given: supply crates parachuted in ahead of it every 5–11 s (the old `drops`), each a random
 * weapon for 16 s or, one in five, a jetpack for 7 s, picked up by whoever passes; and the jetpack's look, its pack
 * and flame drawn on a flying player (the flight itself is the AI's, `ai.ts`).
 *
 * System: squad.supply (spawn).
 */
import { INK } from "@/components/pixel/art";
import { indexOf, PARTICLE_ADDITIVE, PARTICLE_EMISSIVE, type System } from "@/engine";
import { registerShapeArt } from "../art";
import type { BattleGame } from "../game/BattleGame";
import { UNIT_DEAD } from "../game/components";
import { TAG_PROP, type EffectDef, type UnitDef, type UnitDrawFn } from "../game/defs";
import { TEAM_NEUTRAL } from "../game/teams";
import { LAYER } from "../render/layers";
import { ARSENAL, equip } from "./arms";
import { SQUAD } from "./squad";

/** Seconds a crate's weapon lasts, and a jetpack's flight. */
const CRATE_WEAPON = 16;
const CRATE_JET = 7;
/** Seconds a crate waits to be picked up. */
const CRATE_LIFE = 22;
/** A crate's contents for a jetpack. */
const JETPACK = -1;

export const CRATE: UnitDef = {
  name: "Supply crate",
  hw: 6,
  h: 9,
  hp: 1,
  team: TEAM_NEUTRAL,
  art: "prop.crate",
  pose: () => 0,
  layer: LAYER.PROPS,
  tags: TAG_PROP,
  // Nothing breaks it (lava, a falling chunk)
  onHurt: () => 0,
  onSpawn: (game, id) => {
    game.world.add(id, game.core.life);
    game.core.life.t[indexOf(id)] = CRATE_LIFE;
  },
  move: { speed: 0, jump: 0, climb: 0, swim: false, autoJump: false, ladders: false, chuteFall: 45 },
};

/** The jetpack's flame, licking down out of the pack (the old FLAME particles). */
const JET_FLAME: EffectDef = {
  count: 1,
  speed: [60, 100],
  angle: Math.PI / 2,
  spread: 0.5,
  life: [0.2, 0.2],
  size: [1, 2],
  colors: [0xfff3b0ff, 0xffb020ff, 0xff5a1aff],
  ramp: true,
  flags: PARTICLE_EMISSIVE | PARTICLE_ADDITIVE,
};

/** The pack on a player's back: pose = flame colour (bit 0) and flame length 2–4 (pose >> 1), as the old scene. */
const JETPACK_ART = "squad.jetpack";

/** Draws the jetpack on a player who is flying on it (`PLAYER.draw`). */
export const drawJetpack: UnitDrawFn = (game, i, x, y, sink) => {
  if (game.c.player.jet[i] <= 0 || game.c.unit.flags[i] & UNIT_DEAD) return;
  const t = game.now;
  const pose = (Math.floor(t * 20) & 1) | (Math.floor(t * 30) % 3 << 1);
  sink.sprite(game.art.index(JETPACK_ART), pose, 1, x - game.c.unit.facing[i] * 5 - 1, y, LAYER.ENEMIES);
};

/** Drop a crate in, ahead of the squad, under a parachute. */
function dropCrate(game: BattleGame): void {
  const sq = game.world.res(SQUAD);
  const cam = game.camera;
  const r = game.rng.squad;
  const x = Math.max(sq.advance + 30, cam.left + (cam.right - cam.left) * (0.3 + r.next() * 0.4));
  const id = game.spawn("squad.crate", x, cam.top - 10, { chute: true });
  if (id) sq.crates.set(id, r.chance(0.2) ? JETPACK : r.int(ARSENAL.length));
}

/** Whoever passes a crate takes what's in it. */
function pickUp(game: BattleGame, crate: number, contents: number): boolean {
  const sq = game.world.res(SQUAD);
  const { pos } = game.core;
  const { player } = game.c;
  const c = indexOf(crate);
  for (const id of sq.players) {
    const i = indexOf(id);
    if (!game.fighting(id) || Math.abs(pos.x[c] - pos.x[i]) > 10 || Math.abs(pos.y[c] - pos.y[i]) > 18) continue;
    const p = player.id[i];
    if (contents === JETPACK) player.jet[i] = CRATE_JET;
    else {
      equip(game, id, contents);
      player.crateT[i] = CRATE_WEAPON;
    }
    const weapons = game.content.weapons;
    const name = contents === JETPACK ? "JETPACK" : weapons.get(weapons.index(ARSENAL[contents])).name.toUpperCase();
    game.popup(`P${p + 1} ${name}`, pos.x[i], pos.y[i] - 26, sq.colors[p]);
    game.effect("squad.spark", pos.x[i], pos.y[i] - 10, { count: 2.5, colors: [sq.colors[p], 0xffffffff] });
    return true;
  }
  return false;
}

function supplySystem(game: BattleGame): System {
  return {
    name: "squad.supply",
    phase: "spawn",
    run(w, dt) {
      const sq = w.res(SQUAD);
      const { player } = game.c;
      if ((sq.crateT -= dt) <= 0) {
        sq.crateT = 5 + game.rng.squad.next() * 6;
        dropCrate(game);
      }
      for (const [crate, contents] of sq.crates)
        if (!w.has(crate, game.c.unit) || pickUp(game, crate, contents)) {
          sq.crates.delete(crate);
          w.destroy(crate);
        }
      // A crate's weapon runs out: back to the player's own
      for (const id of sq.players) {
        const i = indexOf(id);
        if (player.crateT[i] > 0 && (player.crateT[i] -= dt) <= 0) equip(game, id, player.id[i]);
      }
    },
  };
}

export function installSupply(game: BattleGame): void {
  const c = game.content;
  c.units.register("squad.crate", CRATE);
  c.effects.register("squad.jet", JET_FLAME);
  registerShapeArt(c, JETPACK_ART, {
    w: 5,
    h: 11,
    ox: 1,
    oy: 14,
    poses: 6,
    paint(ctx, pose) {
      ctx.fillStyle = INK;
      ctx.fillRect(-1, -14, 5, 8);
      ctx.fillStyle = "#8a90a0";
      ctx.fillRect(0, -13, 3, 6);
      ctx.fillStyle = pose & 1 ? "#ffe27a" : "#ff8a1a";
      ctx.fillRect(0, -7, 3, 2 + (pose >> 1));
    },
  });
  game.world.addSystem(supplySystem(game));
}
