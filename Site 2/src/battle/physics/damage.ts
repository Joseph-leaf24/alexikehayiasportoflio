/**
 * The damage pipeline: `hurt` (def hook, armour, spawn protection, hit flash, `unitHurt`) and, at zero health,
 * `kill`: the unit goes down (out of the spatial hash, its weapon silent), tumbles as a corpse, plays its death
 * effect, credits the killer's score and emits `unitKilled` (and `playerDown` for the squad). Corpses are removed
 * after their time; players stay, for the squad feature to respawn.
 */
import { BODY_INVULNERABLE, BODY_SOLID_TO_SHOTS, BODY_UNINDEXED, indexOf, type System } from "@/engine";
import type { BattleGame } from "../game/BattleGame";
import { IN_ACID, IN_LAVA, MODE_TUMBLE, UNIT_COUNTED, UNIT_DEAD } from "../game/components";
import { TEAM_SQUAD } from "../game/teams";

/** Damage at or above this ignores spawn protection (lava, crushing, a nuke). */
export const LETHAL = 50;
/** Seconds of hit flash. */
const HIT_FLASH = 0.12;

/** The player id (0–7) of an entity, or -1. */
export function playerOf(game: BattleGame, id: number): number {
  return id !== 0 && game.world.has(id, game.c.player) ? game.c.player.id[indexOf(id)] : -1;
}

/** Damage a unit; returns true when this killed it. `source` is the entity responsible (a shooter, not a bullet). */
export function hurt(game: BattleGame, id: number, dmg: number, source = 0): boolean {
  const w = game.world;
  const { unit, player } = game.c;
  if (!(dmg > 0) || !w.has(id, unit) || !w.has(id, game.core.health)) return false;
  const i = indexOf(id);
  if (unit.flags[i] & UNIT_DEAD || game.core.body.flags[i] & BODY_INVULNERABLE) return false;
  const isPlayer = player.$.sparse[i] >= 0;
  if (isPlayer && player.inv[i] > 0 && dmg < LETHAL) return false;
  const def = game.unitDef(id);
  if (def.onHurt) dmg = def.onHurt(game, id, dmg, source);
  if (def.armour !== undefined) dmg *= def.armour;
  if (!(dmg > 0)) return false;
  const hp = (game.core.health.hp[i] -= dmg);
  unit.hitT[i] = HIT_FLASH;
  // Players get a moment's grace after each hit, as in the old game
  if (isPlayer) player.inv[i] = Math.max(player.inv[i], 0.4);
  const ev = w.events.claim("unitHurt");
  if (ev) {
    ev.id = id;
    ev.dmg = dmg;
    ev.source = source;
    ev.hp = hp;
  }
  if (hp > 0) return false;
  const sx = game.world.has(source, game.core.pos) ? game.core.pos.x[indexOf(source)] : game.core.pos.x[i];
  const away = game.core.pos.x[i] >= sx ? 1 : -1;
  kill(game, id, source, away * (40 + game.rng.combat.next() * 40), -(90 + game.rng.combat.next() * 60));
  return true;
}

/** Bring a unit down now: a corpse flung by (vx, vy), or a downed player. */
export function kill(game: BattleGame, id: number, source = 0, vx = 0, vy = -120): void {
  const w = game.world;
  const { unit, motor, intent, player } = game.c;
  const { body, vel, pos, team, kind, health, life } = game.core;
  if (!w.has(id, unit)) return;
  const i = indexOf(id);
  if (unit.flags[i] & UNIT_DEAD) return;
  unit.flags[i] |= UNIT_DEAD;
  unit.killer[i] = source;
  unit.deadT[i] = 0;
  unit.spin[i] = 0;
  health.hp[i] = Math.min(health.hp[i], 0);
  // Out of every query for targets and hits from now on
  body.flags[i] = (body.flags[i] | BODY_UNINDEXED) & ~BODY_SOLID_TO_SHOTS;
  motor.mode[i] = MODE_TUMBLE;
  vel.vx[i] = vx;
  vel.vy[i] = vy;
  intent.fire[i] = 0;
  intent.move[i] = 0;
  intent.target[i] = 0;
  if (unit.flags[i] & UNIT_COUNTED) {
    unit.flags[i] &= ~UNIT_COUNTED;
    game.stats.enemies--;
  }
  const k = kind.def[i];
  const def = game.content.units.get(k);
  const victimTeam = team.id[i];
  const killer = playerOf(game, source);
  const score = def.score ?? 0;
  if (victimTeam !== TEAM_SQUAD) {
    game.stats.kills++;
    if (killer >= 0) {
      const ki = indexOf(source);
      player.score[ki] += score;
      player.kills[ki]++;
    }
  }
  const fx = game.links.unitDeathFx[k];
  if (fx) game.effect(fx, pos.x[i], pos.y[i] - def.h / 2);
  def.onDeath?.(game, id, source);
  const ev = w.events.claim("unitKilled");
  if (ev) {
    ev.id = id;
    ev.def = k;
    ev.team = victimTeam;
    ev.killer = source;
    ev.player = killer;
    ev.x = pos.x[i];
    ev.y = pos.y[i];
    ev.score = score;
  }
  if (player.$.sparse[i] >= 0) {
    // Players are never destroyed: the squad feature brings them back
    player.downT[i] = 2.6;
    const p = player.id[i];
    w.events.emit("playerDown", { id, player: p, x: pos.x[i], y: pos.y[i] });
  } else {
    w.add(id, life);
    life.t[i] = def.corpse ?? 1.2;
  }
}

/** Hit flashes, corpse clocks and spawn protection wear off ("death" phase). */
export function unitClockSystem(game: BattleGame): System {
  const { unit } = game.c;
  const q = game.world.query(unit);
  return {
    name: "damage.clocks",
    phase: "death",
    run(_w, dt) {
      const n = q.collect(),
        idx = q.idx;
      for (let k = 0; k < n; k++) {
        const i = idx[k];
        if (unit.hitT[i] > 0) unit.hitT[i] = Math.max(0, unit.hitT[i] - dt);
        if (unit.flags[i] & UNIT_DEAD) unit.deadT[i] += dt;
      }
      const p = game.c.player;
      const pn = p.$.size;
      for (let k = 0; k < pn; k++) {
        const i = p.$.dense[k];
        if (p.inv[i] > 0) p.inv[i] = Math.max(0, p.inv[i] - dt);
      }
    },
  };
}

/** Lava kills, acid burns ("damage" phase), from what locomotion found the unit standing in. */
export function hazardSystem(game: BattleGame): System {
  const { unit, motor } = game.c;
  const q = game.world.query(unit, motor);
  return {
    name: "damage.hazards",
    phase: "damage",
    run(w, dt) {
      const n = q.collect(true),
        idx = q.idx;
      for (let k = 0; k < n; k++) {
        const i = idx[k];
        const out = motor.out[i];
        if (!(out & (IN_LAVA | IN_ACID)) || unit.flags[i] & UNIT_DEAD) continue;
        const id = w.idOf(i);
        if (out & IN_LAVA) {
          game.effect("blast.fire", game.core.pos.x[i], game.core.pos.y[i] - 4, { count: 0.5 });
          hurt(game, id, 99);
        } else hurt(game, id, 3 * dt);
      }
    },
  };
}
