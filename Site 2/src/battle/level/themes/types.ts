/** The theme types and the defaults every theme spreads (ported from `components/pixel/themes.ts`). */

/** Set pieces the generator can place (the old `level.ts` builds each one in `feature`). */
export type SetPiece =
  | "flat"
  | "slope"
  | "gap"
  | "bridge"
  | "wall"
  | "huts"
  | "stilts"
  | "tower"
  | "bunker"
  | "tunnel"
  | "mgnest"
  | "catwalk"
  | "tank"
  | "tanker"
  | "convoy"
  | "road"
  | "derrick"
  | "silo"
  | "radar"
  | "ruins"
  | "mesa"
  | "pines"
  | "deadtrees"
  | "eggs"
  | "lava"
  | "spires"
  | "wreck"
  | "fort"
  | "canopy"
  | "bamboo"
  | "river"
  | "waterfall"
  | "temple"
  | "statues"
  | "camp"
  | "metropolis"
  | "highway"
  | "airfield"
  | "downtown"
  | "harbour"
  | "ocean"
  | "lake"
  | "forest"
  | "railway"
  | "nuclear"
  | "massif"
  | "abyss"
  | "mineshaft"
  | "metro"
  | "railyard"
  | "factory"
  | "boulevard"
  | "park"
  | "citybridge"
  | "waterplant"
  | "station";

/** Enemy foot-soldier kinds. */
export type Troop =
  | "mook"
  | "heavy"
  | "rocket"
  | "sniper"
  | "crawler"
  | "mortar"
  | "dog"
  | "shield"
  | "flamer"
  | "grenadier"
  | "jetpacker"
  | "minigunner"
  | "officer"
  | "dynamiter"
  | "rusher"
  | "engineer"
  | "laser"
  | "exosuit";
export type BossKind = "tank" | "mech" | "gunship" | "worm" | "airship" | "cannon" | "spider";
export type Backdrop =
  | "jungle"
  | "deepjungle"
  | "temple"
  | "monsoon"
  | "desert"
  | "canyon"
  | "snow"
  | "fort"
  | "factory"
  | "city"
  | "volcano"
  | "hive"
  | "swamp"
  | "silo"
  | "cave"
  | "savanna"
  | "autumn"
  | "sakura"
  | "tundra"
  | "badlands"
  | "farmland"
  | "megacity"
  | "airbase"
  | "archipelago"
  | "mangrove";
/** Ambient particles and weather. */
export type Ambient = "fireflies" | "smoke" | "embers" | "ash" | "snow" | "rain" | "storm" | "sand" | "spores" | "none";

export interface Theme {
  key: string;
  /** Shown in the HUD and the level banner */
  name: string;
  /** Sky gradient, top → horizon */
  sky: [string, string, string];
  /** How dark the world is tinted, from 0 for day to 1 for black */
  night: number;
  /** An rgba colour wash over the world, or empty for none */
  tint: string;
  sun: "sun" | "red" | "moon" | "none";
  backdrop: Backdrop;
  /** Which backdrop's enemy uniforms to use */
  look: Backdrop;
  /** Ground: top block, the few below it, and the deep rock */
  top: number;
  fill: number;
  deep: number;
  /** Block kinds for obstacles, buildings and background ruins; skyline 0 means no ruins */
  wall: number;
  build: number;
  skyline: number;
  decor: "palms" | "jungle" | "pines" | "deadwood" | "fires" | "crystals" | "rubble" | "none" | "autumn" | "sakura";
  /** Relative weight of each feature */
  features: Partial<Record<SetPiece, number>>;
  troops: Partial<Record<Troop, number>>;
  /** Enemies per 12 columns */
  enemies: number;
  /** Scales the chance of a barrel cluster in each feature */
  barrels: number;
  /** Artillery shells and meteors per second; helicopters, paratroopers, bombing jets and drone swarms per minute */
  artillery: number;
  meteors: number;
  helis: number;
  paras: number;
  jets: number;
  drones: number;
  ambient: Ambient;
  /** Intensity of the distant background battle, 0 to 1 */
  distant: number;
  /** How fast the squad pushes forward, px/s */
  speed: number;
  /** Hill and mountain scale for terrainAt, from 0 for flat to about 1.5 for alpine */
  relief: number;
  bosses: BossKind[];
  /** A cave roof over the level */
  ceiling?: boolean;
  /** Level length multiplier */
  long?: number;
  /** Run a day–night and weather cycle instead of fixed light and weather */
  cycle?: boolean;
}

export const base = {
  tint: "",
  sun: "sun" as Theme["sun"],
  skyline: 0,
  decor: "none" as Theme["decor"],
  meteors: 0,
  drones: 0,
  distant: 0.6,
  speed: 38,
  relief: 1,
  ceiling: false,
};
