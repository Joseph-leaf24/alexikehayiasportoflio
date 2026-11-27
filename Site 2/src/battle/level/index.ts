/**
 * The level feature: deals the level's theme, fills terrain columns from the old generator, registers every theme's
 * backdrop and uniforms, shows the backdrop of the biome the camera is in (fading over the last one at a biome
 * border, as the old scene did) with its light and weather (`weather.ts`), makes the props burn, smoke and spray
 * (`ambient.ts`), and makes sure the ground exists before anything spawns on it.
 *
 * Systems: level.ambient (effects, every 2 ticks), level.weather (effects), level.biome (late).
 */
import { BASE_ROW } from "@/components/pixel/level";
import { resource, TILE } from "@/engine";
import { backdropArt, registerBackdropArt, registerTroopArt } from "../art";
import type { BattleGame } from "../game/BattleGame";
import type { BackdropLayer } from "../render/sky";
import { ambientSystem, registerAmbientEffects } from "./ambient";
import { LevelSource, levelGenerator } from "./generator";
import { registerDrawnArt } from "./drawn";
import { propsPass } from "./props";
import { THEMES, themeDeck, variant, type Backdrop, type Theme } from "./themes";
import { installWeather, skyLook } from "./weather";

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

/** The old scene's backdrop layers, far to near: parallax factor, and how far below the horizon each stands. */
const LAYERS = [
  { layer: "peaks", factor: 0.04, lift: 6 },
  { layer: "far", factor: 0.08, lift: 20 },
  { layer: "mid", factor: 0.2, lift: 24 },
  { layer: "near", factor: 0.4, lift: 30 },
] as const;
/** Backdrops with a roof over them: no clouds drift there. */
const ROOFED: ReadonlySet<Backdrop> = new Set(["cave", "volcano", "hive"]);

/** A backdrop's parallax layers; the cave's far rock hangs from the top of the view. */
export function backdropLayers(kind: Backdrop): BackdropLayer[] {
  return LAYERS.map(({ layer, factor, lift }) => ({
    art: backdropArt(kind, layer),
    factor,
    lift,
    top: kind === "cave" && layer === "far",
  }));
}

/** Show a biome's sky: its backdrop and light, faded in or at once, and clouds unless it's roofed over. */
function showBiome(game: BattleGame, theme: Theme, fadeIn: boolean): void {
  game.sky.show(backdropLayers(theme.backdrop), fadeIn, skyLook(theme));
  game.sky.clouds = !ROOFED.has(theme.backdrop);
}

export function install(game: BattleGame): void {
  const rand = () => game.rng.level.next();
  const deck = themeDeck(rand);
  const key = game.options.feature.level?.theme;
  // As in the old game: every level, and every biome border inside one, is a random variant of a dealt theme
  const theme = variant(key ? THEMES[key] : deck(), rand);
  const level = game.world.setRes(LEVEL, new LevelSource(theme, game.options.seed));
  level.old.pickTheme = (cur) => variant(deck(cur.key), rand);
  game.level.generate = levelGenerator(level, () => game.origin.cols);
  // Any theme can follow at a border: its art is painted when first drawn, the first level's at boot
  for (const t of Object.values(THEMES)) {
    registerBackdropArt(game.content, t.backdrop, t.backdrop === theme.backdrop);
    registerTroopArt(game.content, t.look);
  }
  registerDrawnArt(game.content);
  showBiome(game, theme, false);
  game.sky.horizonY = BASE_ROW * TILE;
  game.frames.add(propsPass(game, level));
  registerAmbientEffects(game.content);
  game.world.addSystem(ambientSystem(game, level));
  installWeather(game, level);
  game.world.addSystem({
    name: "level.biome",
    phase: "late",
    run() {
      const cam = game.camera;
      if (level.enter(Math.floor((cam.left + cam.right) / 2 / TILE) + game.origin.cols))
        showBiome(game, level.biome, true);
    },
  });
  game.world.events.on("levelStart", () => {
    game.terrain.ensure(Math.ceil((game.camera.right + 600) / TILE));
  });
}
