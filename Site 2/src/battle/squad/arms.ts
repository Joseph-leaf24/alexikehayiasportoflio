/**
 * The squad's arms: the old game's 17 weapons (`WEAPONS`) in their old order, the projectile guns in `guns.ts` and
 * the beams in `rays.ts`, plus the hand grenade thrown as each player's special. Player n starts with weapon n;
 * supply crates hand out the rest for a while.
 */
import { indexOf } from "@/engine";
import type { BattleGame } from "../game/BattleGame";
import type { ProjectileDef } from "../game/defs";
import { playerArt } from "../art";
import { registerGuns } from "./guns";
import { installRays, registerRays } from "./rays";

/** Weapon ids by the old weapon number. */
export const ARSENAL = [
  "squad.rifle",
  "squad.shotgun",
  "squad.plasma",
  "squad.rockets",
  "squad.sniper",
  "squad.flamethrower",
  "squad.grenades",
  "squad.minigun",
  "squad.laser",
  "squad.tesla",
  "squad.pistols",
  "squad.crossbow",
  "squad.saws",
  "squad.mortar",
  "squad.pulse",
  "squad.deathray",
  "squad.prism",
] as const;

/** The hand grenade: bounces, then goes off after its fuse (BLAST_R 22). */
export const GRENADE: ProjectileDef = {
  speed: 170,
  life: 1.1,
  damage: 0,
  gravity: 620,
  bounce: 0.45,
  fuse: true,
  radius: 1,
  explode: { r: 22, ignite: 0.4 },
  look: { colors: [0x3a5a2aff], size: 2, blink: 0xff3b3bff },
};

/** Put weapon `w` (the old number, 0–16) in a unit's hands, starting a fresh burst; a player is drawn holding it. */
export function equip(game: BattleGame, id: number, w: number): void {
  const i = indexOf(id);
  const { weapon, player, unit } = game.c;
  weapon.def[i] = game.content.weapons.index(ARSENAL[w]);
  weapon.shots[i] = 0;
  if (player.$.sparse[i] >= 0) unit.art[i] = game.art.index(playerArt(player.id[i], w));
}

export function installArms(game: BattleGame): void {
  registerGuns(game.content);
  registerRays(game.content);
  game.content.projectiles.register("squad.grenade", GRENADE);
  installRays(game);
}
