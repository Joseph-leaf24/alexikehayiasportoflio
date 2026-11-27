/**
 * The art feature: registers the art sets content draws with (painted lazily in a browser by the old engine's
 * painters) and the terrain tile painter. Ids: `player.<0-7>.<weapon 0-16>`, `troop.<look>.<kind>`,
 * `backdrop.<theme>.<peaks|far|mid|near>`, `prop.<name>`, `asset.<id>`, `sky.cloud`, `fx.chute`. Other features add sets the
 * same way (`registerTroopArt` for another theme's uniforms, `registerStripArt` for runs of colour drawn at any size,
 * `registerShapeArt` for small shapes drawn by hand, or `game.content.art.register` for their own).
 */
import { ASSETS } from "@/components/pixel/assets";
import type { Backdrop } from "@/components/pixel/themes";
import { PLAYER_COLORS } from "@/data/squad";
import type { BattleGame } from "../game/BattleGame";
import type { Content } from "../game/content";
import {
  paintAsset,
  paintBackdrop,
  paintChutes,
  paintClouds,
  paintPlayer,
  paintProp,
  paintShape,
  paintStrip,
  paintTiles,
  paintTroop,
  type BackdropLayerName,
  type PropName,
  type ShapeSpec,
  type TroopKind,
} from "./painters";

export { ArtBook, tilesFromSheet, type TilePainter } from "./ArtBook";
export { HUMANOID_POSES, STRIP_PIECE, type ShapeSpec, type TroopKind } from "./painters";

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

export const backdropArt = (kind: string, layer: BackdropLayerName): string => `backdrop.${kind}.${layer}`;
/** Art id of an old scene prop (`ASSETS`): pose 0 lit, pose 1 in shadow, two facings. */
export const assetArt = (id: string): string => `asset.${id}`;

/** A run of plain colour (`paintStrip`), for things drawn at any size; one already registered is left as it is. */
export function registerStripArt(content: Content, id: string, axis: "x" | "y", colors: readonly string[]): void {
  if (!content.art.has(id)) content.art.register(id, { paint: () => paintStrip(axis, colors) });
}

/** A fixed shape drawn by hand (`paintShape`); one already registered is left as it is. */
export function registerShapeArt(content: Content, id: string, shape: ShapeSpec): void {
  if (!content.art.has(id)) content.art.register(id, { paint: () => paintShape(shape) });
}

/** Every troop kind in a theme's uniform; a look already registered is left as it is. */
export function registerTroopArt(content: Content, look: Backdrop): void {
  for (const kind of TROOP_KINDS) {
    const id = troopArt(look, kind);
    if (!content.art.has(id)) content.art.register(id, { paint: () => paintTroop(look, kind), facings: 2 });
  }
}

/**
 * A theme's backdrop layers, painted on first use, or at boot when `eager` (the first level's, drawn from the first
 * frame); a backdrop already registered is left as it is.
 */
export function registerBackdropArt(content: Content, kind: Backdrop, eager = false): void {
  for (const layer of ["peaks", "far", "mid", "near"] as const satisfies readonly BackdropLayerName[]) {
    const id = backdropArt(kind, layer);
    if (!content.art.has(id)) content.art.register(id, { paint: () => paintBackdrop(kind, layer), eager });
  }
}

export function install(game: BattleGame): void {
  const art = game.content.art;
  for (let p = 0; p < PLAYER_COLORS.length; p++)
    for (let w = 0; w < WEAPON_ART; w++)
      art.register(playerArt(p, w), { paint: () => paintPlayer(p, PLAYER_COLORS[p], w), facings: 2 });
  // The default uniform; the level feature adds every theme's, and their backdrops
  registerTroopArt(game.content, "jungle");
  for (const name of PROPS) art.register(`prop.${name}`, { paint: () => paintProp(name) });
  for (const id of Object.keys(ASSETS)) art.register(assetArt(id), { paint: () => paintAsset(id), facings: 2 });
  art.register("sky.cloud", { paint: () => paintClouds(game.budgets.clouds), eager: true });
  art.register("fx.chute", { paint: () => paintChutes(PLAYER_COLORS), eager: true });
  game.art.tilePainter = paintTiles;
}
