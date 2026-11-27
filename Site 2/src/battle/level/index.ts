/**
 * The level feature: deals the level's theme, fills terrain columns from the old generator, sets the backdrop
 * (still the jungle's until the per-theme backdrops are registered), and makes sure the ground exists before
 * anything spawns on it.
 */
import { BASE_ROW } from "@/components/pixel/level";
import { resource, TILE } from "@/engine";
import type { BattleGame } from "../game/BattleGame";
import type { BackdropLayer } from "../render/sky";
import { LevelSource, levelGenerator } from "./generator";
import { THEMES, themeDeck, variant } from "./themes";

export { LevelSource } from "./generator";

export interface LevelOptions {
  /** Play this theme (a `THEMES` key) instead of dealing one from the deck. */
  theme?: string;
}

declare module "../game/options" {
  interface FeatureOptions {
    level?: LevelOptions;
  }
}

/** The level's layout, for features that place things by it (`game.world.res(LEVEL)`). */
export const LEVEL = resource<LevelSource>("level");

/** The old jungle backdrop's factors and lifts (`PixelScene` drawWorld). */
const JUNGLE: BackdropLayer[] = [
  { art: "backdrop.jungle.peaks", factor: 0.04, lift: 6 },
  { art: "backdrop.jungle.far", factor: 0.08, lift: 20 },
  { art: "backdrop.jungle.mid", factor: 0.2, lift: 24 },
  { art: "backdrop.jungle.near", factor: 0.4, lift: 30 },
];

export function install(game: BattleGame): void {
  const rand = () => game.rng.level.next();
  const deck = themeDeck(rand);
  const key = game.options.feature.level?.theme;
  // As in the old game: every level, and every biome border inside one, is a random variant of a dealt theme
  const theme = variant(key ? THEMES[key] : deck(), rand);
  const level = game.world.setRes(LEVEL, new LevelSource(theme, game.options.seed));
  level.old.pickTheme = (cur) => variant(deck(cur.key), rand);
  game.level.generate = levelGenerator(level, () => game.origin.cols);
  game.sky.backdrop = JUNGLE;
  game.sky.horizonY = BASE_ROW * TILE;
  game.world.events.on("levelStart", () => {
    game.terrain.ensure(Math.ceil((game.camera.right + 600) / TILE));
  });
}
