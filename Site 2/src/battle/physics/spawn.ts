/**
 * Making a unit from its def: the core components (feet position, box, health, team, kind) and the game's (unit,
 * motor tuned from the def's MoveSpec, intent, weapon, ai), within the enemy budget.
 */
import { BODY_COLLIDES_TERRAIN, BODY_SOLID_TO_SHOTS, indexOf, place } from "@/engine";
import type { BattleGame } from "../game/BattleGame";
import { MODE_CHUTE, UNIT_COUNTED } from "../game/components";
import { TEAM_SQUAD } from "../game/teams";
import { DEFAULT_MOVE, initMotor } from "./locomotion";

export interface SpawnOptions {
  /** Team other than the def's (a captured unit, a persuaded trooper). */
  team?: number;
  /** 1 faces right (default), -1 left. */
  facing?: number;
  /** Drops in under a parachute. */
  chute?: boolean;
  vx?: number;
  vy?: number;
  /** Spawn even when the enemy budget is full (bosses, scripted set pieces). */
  force?: boolean;
  /** Health multiplier. */
  hp?: number;
}

/** Resolve a unit def by id or index; throws for an unknown id, since that is a content bug. */
export function unitIndex(game: BattleGame, def: string | number): number {
  if (typeof def === "number") return def;
  const k = game.content.units.index(def);
  if (k === 0) throw new Error(`spawn: unknown unit "${def}"`);
  return k;
}

export function spawnUnit(game: BattleGame, def: string | number, x: number, y: number, o?: SpawnOptions): number {
  const k = unitIndex(game, def);
  const d = game.content.units.get(k);
  const team = o?.team ?? d.team;
  const counted = team !== TEAM_SQUAD;
  if (counted && !o?.force && game.stats.enemies >= game.budgets.maxEnemies) return 0;
  const w = game.world;
  const { vel, body, health, kind } = game.core;
  const c = game.c;
  const id = w.create();
  const i = indexOf(id);
  place(w, id, x, y);
  w.add(id, vel);
  vel.vx[i] = o?.vx ?? 0;
  vel.vy[i] = o?.vy ?? 0;
  w.add(id, body);
  body.hw[i] = d.hw;
  body.h[i] = d.h;
  body.flags[i] = BODY_COLLIDES_TERRAIN | BODY_SOLID_TO_SHOTS;
  w.add(id, health);
  health.hp[i] = health.max[i] = d.hp * (o?.hp ?? 1);
  w.add(id, game.core.team);
  game.core.team.id[i] = team;
  w.add(id, kind);
  kind.def[i] = k;
  w.add(id, c.unit);
  c.unit.facing[i] = (o?.facing ?? 1) < 0 ? -1 : 1;
  if (counted) {
    c.unit.flags[i] |= UNIT_COUNTED;
    game.stats.enemies++;
  }
  w.add(id, c.motor);
  initMotor(game, i, d.move ? { ...DEFAULT_MOVE, ...d.move } : DEFAULT_MOVE);
  c.motor.lastX[i] = x;
  if (o?.chute) c.motor.mode[i] = MODE_CHUTE;
  w.add(id, c.intent);
  const weapon = game.links.unitWeapon[k];
  if (weapon) {
    w.add(id, c.weapon);
    c.weapon.def[i] = weapon;
    c.weapon.cd[i] = 0.3 + game.rng.combat.next() * 0.5;
  }
  if (d.ai) {
    w.add(id, c.ai);
    c.ai.home[i] = x;
  }
  d.onSpawn?.(game, id);
  const ev = w.events.claim("unitSpawned");
  if (ev) {
    ev.id = id;
    ev.def = k;
    ev.team = team;
  }
  return id;
}
