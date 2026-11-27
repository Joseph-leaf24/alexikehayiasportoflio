/**
 * The art feature: registers the art sets content draws with (painted lazily in a browser by the old engine's
 * painters) and the terrain tile painter. Ids: `player.<0-7>.<weapon 0-16>`, `troop.<look>.<kind>`,
 * `backdrop.<theme>.<peaks|far|mid|near>`, `prop.<name>`, `sky.cloud`, `fx.chute`. Other features add sets the
 * same way (`registerTroopArt` for another theme's uniforms, or `game.content.art.register` for their own).
 */
import type { Backdrop } from "@/components/pixel/themes";
import { PLAYER_COLORS } from "@/data/squad";
import type { BattleGame } from "../game/BattleGame";
import type { Content } from "../game/content";
import {
  paintBackdrop,
  paintChutes,
  paintClouds,
  paintPlayer,
  paintProp,
  paintTiles,
  paintTroop,
  type BackdropLayerName,
  type PropName,
  type TroopKind,
} from "./painters";

export { ArtBook, tilesFromSheet, type TilePainter } from "./ArtBook";
export { HUMANOID_POSES, type TroopKind } from "./painters";

/** The old game's weapon count: every player can be painted holding any of them. */
export const WEAPON_ART = 17;

const TROOP_KINDS: readonly TroopKind[] = [
  "mook",
  "rocket",
  "flamer",
  "grenadier",
  "shield",
  "sniper",
  "heavy",
  "mortar",
  "mg",
  "turret",
  "crawler",
  "dog",
  "jetpacker",
  "minigunner",
  "officer",
  "dynamiter",
  "rusher",
  "engineer",
  "laser",
  "exosuit",
];

const PROPS: readonly PropName[] = [
  "barrel",
  "fuel",
  "ammo",
  "egg",
  "propane",
  "canister",
  "cylinder",
  "mine",
  "crate",
  "tnt",
];

/** Art id of player `p` (0–7) holding weapon `w`. */
export const playerArt = (p: number, w: number): string => `player.${p}.${w}`;
export const troopArt = (look: string, kind: TroopKind): string => `troop.${look}.${kind}`;

/** Every troop kind in a theme's uniform. */
export function registerTroopArt(content: Content, look: Backdrop): void {
  for (const kind of TROOP_KINDS)
    content.art.register(troopArt(look, kind), { paint: () => paintTroop(look, kind), facings: 2 });
}

/** A theme's backdrop layers (painted at boot: they are drawn every frame). */
export function registerBackdropArt(content: Content, kind: Backdrop): void {
  for (const layer of ["peaks", "far", "mid", "near"] as const satisfies readonly BackdropLayerName[])
    content.art.register(`backdrop.${kind}.${layer}`, { paint: () => paintBackdrop(kind, layer), eager: true });
}

export function install(game: BattleGame): void {
  const art = game.content.art;
  for (let p = 0; p < PLAYER_COLORS.length; p++)
    for (let w = 0; w < WEAPON_ART; w++)
      art.register(playerArt(p, w), { paint: () => paintPlayer(p, PLAYER_COLORS[p], w), facings: 2 });
  registerTroopArt(game.content, "jungle");
  registerBackdropArt(game.content, "jungle");
  for (const name of PROPS) art.register(`prop.${name}`, { paint: () => paintProp(name) });
  art.register("sky.cloud", { paint: () => paintClouds(game.budgets.clouds), eager: true });
  art.register("fx.chute", { paint: () => paintChutes(PLAYER_COLORS), eager: true });
  game.art.tilePainter = paintTiles;
}
