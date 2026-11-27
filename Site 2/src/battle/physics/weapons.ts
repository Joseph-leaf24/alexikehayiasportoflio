/**
 * Weapons: the cadence and burst logic every gun shares, aiming (straight, led, or lobbed on a ballistic arc), and
 * the projectile spawn. A unit fires when its AI sets `intent.fire` and its cooldown is up; the aim follows its
 * target's current body centre, so 10 Hz thinking still shoots straight. Custom weapons (beams, chains) replace the
 * spawn with their def's `fire`.
 */
import { indexOf, place, type System } from "@/engine";
import type { BattleGame } from "../game/BattleGame";
import { UNIT_DEAD, UNIT_HELD } from "../game/components";
import type { WeaponDef } from "../game/defs";
import { muzzle, PLAYER_RGB } from "../render/beams";
import { playerOf } from "./damage";
import { GRAVITY } from "./locomotion";

export interface ShotOptions {
  /** Who fired it (kill credit and the team it spares). */
  owner?: number;
  /** Its team; default the owner's. */
  team?: number;
  /** Multiplies the def's damage. */
  damage?: number;
}

/** Launch velocity to land a shot with gravity `g` on (x1, y1) at about `speed` px/s across (the old `lobAt`). */
export function lobAt(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  g: number,
  speed: number,
  out: { vx: number; vy: number }
): void {
  const t = Math.max(0.4, Math.min(1.6, Math.abs(x1 - x0) / speed));
  out.vx = (x1 - x0) / t;
  out.vy = (y1 - y0 - 0.5 * g * t * t) / t;
}

/** Resolve a projectile by id or index; 0 when unknown. */
const projectileIndex = (game: BattleGame, def: string | number) =>
  typeof def === "number" ? def : game.content.projectiles.index(def);

/** Spawn a projectile; returns 0 when the def is unknown or the projectile budget is full. */
export function shoot(
  game: BattleGame,
  def: string | number,
  x: number,
  y: number,
  vx: number,
  vy: number,
  o?: ShotOptions
): number {
  const k = projectileIndex(game, def);
  if (k === 0 || game.stats.projectiles >= game.budgets.maxProjectiles) return 0;
  const d = game.content.projectiles.get(k);
  const w = game.world;
  const owner = o?.owner ?? 0;
  const { vel, team } = game.core;
  const p = game.c.projectile;
  const id = w.create();
  const i = indexOf(id);
  place(w, id, x, y);
  w.add(id, vel);
  vel.vx[i] = vx;
  vel.vy[i] = vy;
  w.add(id, team);
  team.id[i] = o?.team ?? (w.has(owner, team) ? team.id[indexOf(owner)] : 0);
  w.add(id, p);
  p.def[i] = k;
  p.owner[i] = owner;
  p.pid[i] = playerOf(game, owner);
  p.pierce[i] = d.pierce ?? 0;
  p.dmg[i] = d.damage * (o?.damage ?? 1);
  p.life[i] = d.life;
  game.stats.projectiles++;
  game.stats.shots++;
  return id;
}

const lob = { vx: 0, vy: 0 };

/** Fire a unit's weapon at (aimX, aimY) now: pellets, spread, lob or custom fire, muzzle flash and the event. */
export function fireWeapon(game: BattleGame, shooter: number, aimX: number, aimY: number): void {
  const { pos, body } = game.core;
  const { weapon, unit } = game.c;
  if (!game.world.has(shooter, weapon)) return;
  const i = indexOf(shooter);
  const wk = weapon.def[i];
  const def = game.content.weapons.get(wk);
  const facing = aimX >= pos.x[i] ? 1 : -1;
  unit.facing[i] = facing;
  const gx = pos.x[i] + facing * (def.muzzleX ?? 7);
  const gy = pos.y[i] + (def.muzzleY ?? -Math.min(9, body.h[i] * 0.6));
  if (def.fire) def.fire(game, shooter, gx, gy, aimX, aimY);
  else launch(game, shooter, def, wk, gx, gy, aimX, aimY);
  if (def.flash) {
    game.flash(gx + facing * 2, gy, def.flash.radius, def.flash.color, 1.4, 0.08);
    // The old scene's muzzle cross, in the player's colour for the squad
    const pid = playerOf(game, shooter);
    muzzle(game, gx + facing * 2, gy, pid >= 0 ? PLAYER_RGB[pid] : def.flash.color);
  }
  const fx = game.links.weaponMuzzleFx[wk];
  if (fx) game.effect(fx, gx, gy, { vx: -facing * 20 });
  const ev = game.world.events.claim("weaponFired");
  if (ev) {
    ev.id = shooter;
    ev.weapon = wk;
    ev.x = gx;
    ev.y = gy;
    ev.aimX = aimX;
    ev.aimY = aimY;
  }
}

function launch(
  game: BattleGame,
  shooter: number,
  def: WeaponDef,
  wk: number,
  gx: number,
  gy: number,
  aimX: number,
  aimY: number
): void {
  const pk = game.links.weaponProjectile[wk];
  if (pk === 0) return;
  const pd = game.content.projectiles.get(pk);
  const r = game.rng.combat;
  const pellets = def.pellets ?? 1;
  const spread = def.spread ?? 0;
  const base = def.speed ?? pd.speed;
  const dmg = def.damage ?? 1;
  for (let k = 0; k < pellets; k++) {
    let vx: number, vy: number;
    if (def.lob) {
      lobAt(gx, gy, aimX + (r.next() - 0.5) * spread * 40, aimY, pd.gravity ?? GRAVITY, def.lob, lob);
      vx = lob.vx;
      vy = lob.vy;
    } else {
      const a = Math.atan2(aimY - gy, aimX - gx) + (r.next() - 0.5) * spread;
      const s = base + (def.speedJitter ? r.next() * def.speedJitter : 0);
      vx = Math.cos(a) * s;
      vy = Math.sin(a) * s;
    }
    shoot(game, pk, gx, gy, vx, vy, { owner: shooter, damage: dmg });
  }
}

/** Where a unit aims at a target: its body centre, led by its velocity when the weapon leads. */
function aimAt(
  game: BattleGame,
  i: number,
  target: number,
  def: WeaponDef,
  speed: number,
  out: { x: number; y: number }
): boolean {
  const w = game.world;
  if (!game.fighting(target)) return false;
  const { pos, body, vel } = game.core;
  const j = indexOf(target);
  out.x = pos.x[j];
  out.y = pos.y[j] - body.h[j] * 0.5;
  if (def.lead && w.has(target, vel) && speed > 0) {
    const d = Math.hypot(out.x - pos.x[i], out.y - pos.y[i]);
    out.x += (vel.vx[j] * d) / speed;
  }
  return true;
}

const aim = { x: 0, y: 0 };

/** Cooldowns, bursts and firing for every armed unit ("weapons" phase). */
export function weaponsSystem(game: BattleGame): System {
  const { weapon, intent, unit } = game.c;
  const { pos } = game.core;
  const q = game.world.query(weapon, intent, unit, pos);
  return {
    name: "weapons.fire",
    phase: "weapons",
    run(w, dt) {
      const defs = game.content.weapons.defs;
      const n = q.collect(true),
        idx = q.idx;
      for (let k = 0; k < n; k++) {
        const i = idx[k];
        weapon.alt[i] -= dt;
        const cd = (weapon.cd[i] -= dt);
        if (cd > 0 || !intent.fire[i] || unit.flags[i] & (UNIT_DEAD | UNIT_HELD)) continue;
        const wk = weapon.def[i];
        const def = defs[wk - 1];
        const target = intent.target[i];
        const speed = def.speed ?? game.content.projectiles.tryGet(game.links.weaponProjectile[wk])?.speed ?? 0;
        if (target !== 0) {
          if (!aimAt(game, i, target, def, speed, aim)) {
            intent.target[i] = 0;
            intent.fire[i] = 0;
            continue;
          }
        } else {
          aim.x = intent.aimX[i];
          aim.y = intent.aimY[i];
        }
        const dx = aim.x - pos.x[i],
          dy = aim.y - pos.y[i];
        if (dx * dx + dy * dy > def.range * def.range * 1.3) continue;
        fireWeapon(game, w.idOf(i), aim.x, aim.y);
        const shots = ++weapon.shots[i];
        const burst = def.burst ?? 1;
        const pause = def.burstPause ?? def.cadence;
        const jitter = def.pauseJitter ? 1 + game.rng.combat.next() * def.pauseJitter : 1;
        weapon.cd[i] = burst > 1 && shots % burst !== 0 ? def.cadence : (burst > 1 ? pause : def.cadence) * jitter;
      }
    },
  };
}
