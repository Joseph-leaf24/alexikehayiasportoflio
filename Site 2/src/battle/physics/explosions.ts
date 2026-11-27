/**
 * Explosions, after the old engine's `explode`: carve the terrain (soft ground breaks at once, rock needs a big or
 * repeated blast), throw debris in the colours of what broke, kill or hurt and fling the units in reach, then the
 * look: fireball and smoke, a shockwave, a flash of light, screen shake and, for big ones, a white flash. Emits
 * `blast`. Light infantry caught in a blast is killed and flung; armour and bosses take damage.
 */
import { indexOf, teamBit, TEAM_ALL } from "@/engine";
import type { BattleGame } from "../game/BattleGame";
import { UNIT_DEAD } from "../game/components";
import { TAG_ARMOURED, TAG_BOSS, type BlastSpec } from "../game/defs";
import { hostileTo, TEAM_NEUTRAL } from "../game/teams";
import { emitDebris, emitEffect } from "../render/effects";
import { kill, hurt } from "./damage";
import { knock } from "./locomotion";

/** Hit lists per nesting depth: a death hook may set off another blast while this one is still dealing damage. */
const lists: number[][] = [];
let depth = 0;

/** Fireball and smoke scaled to the radius (a grenade is about 22 px). */
function blastLook(game: BattleGame, x: number, y: number, r: number, fx: string | number): void {
  const k = Math.max(0.5, Math.min(4, r / 22));
  emitEffect(game, fx, x, y, { count: k, speed: Math.sqrt(k) });
  emitEffect(game, "blast.smoke", x, y - 2, { count: k });
  game.screen.shockwave(x, y, k);
  game.flash(x, y - 4, 40 + r * 2.4, 0xffb060, 1.4 + k * 0.3, 0.22 + k * 0.06);
  game.shake(Math.min(14, 2 + r / 4));
  if (r >= 40) game.screen.flash = Math.max(game.screen.flash, Math.min(0.6, r / 120));
  game.screen.chroma = Math.min(1, game.screen.chroma + k * 0.25);
}

export function explode(
  game: BattleGame,
  x: number,
  y: number,
  spec: BlastSpec,
  source = 0,
  team = TEAM_NEUTRAL
): void {
  const r = spec.r;
  game.stats.booms++;
  const t = game.terrain;
  const carve = t.carve(x, y, r * (spec.dig ?? 0.62), spec.power ?? r / 7);
  const cells = carve.destroyed;
  emitDebris(game, carve, x, y, Math.min(1.6, 0.6 + r / 40));
  if (spec.ignite) t.ignite(x, y, r * spec.ignite);

  // Units in reach: the source's own team is spared unless the blast is friendly; everyone is pushed
  const { pos, body, team: teams } = game.core;
  const { unit } = game.c;
  const mask = spec.friendly || team === TEAM_NEUTRAL ? TEAM_ALL : hostileTo(team);
  const hits = (lists[depth] ??= []);
  const n = game.world.spatial.inRadius(x, y, r + 8, TEAM_ALL, hits);
  depth++;
  const knockAt = spec.knock ?? 220;
  const dmgAt = spec.damage ?? 10;
  try {
    for (let k = 0; k < n; k++) {
      const id = hits[k];
      const i = indexOf(id);
      if (unit.$.sparse[i] < 0 || unit.flags[i] & UNIT_DEAD) continue;
      const cx = pos.x[i],
        cy = pos.y[i] - body.h[i] / 2;
      const d = Math.hypot(cx - x, cy - y);
      const reach = r + Math.max(8, body.hw[i]);
      if (d >= reach) continue;
      const f = 1 - d / reach;
      const side = cx >= x ? 1 : -1;
      const def = game.unitDef(id);
      const tough = ((def.tags ?? 0) & (TAG_ARMOURED | TAG_BOSS)) !== 0;
      if ((teamBit(teams.id[i]) & mask) !== 0) {
        if (def.blastKills && !tough) {
          kill(game, id, source, side * (80 + 140 * f), -(140 + 160 * f));
          continue;
        }
        const dmg = def.blastTaken ?? (dmgAt * f + 3) * Math.max(1, r / 22);
        if (hurt(game, id, dmg, source)) continue;
      }
      if (!tough) knock(game, id, side * knockAt * 0.4 * (0.5 + f), -knockAt * 0.6 * (0.4 + f));
    }
  } finally {
    depth--;
  }

  const fx = spec.fx ?? "blast.fire";
  blastLook(game, x, y, r, fx);
  const ev = game.world.events.claim("blast");
  if (ev) {
    ev.x = x;
    ev.y = y;
    ev.r = r;
    ev.source = source;
    ev.team = team;
    ev.cells = cells;
  }
}
