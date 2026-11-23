/**
 * The level themes. Every level picks one at random (never the same twice in a row), so the
 * scenery keeps changing and is different on every visit: what the ground is made of, what
 * gets built on it, who defends it, which boss waits half way, the sky, the weather, how hard
 * the artillery, jets, drones and paratroopers come down, and how big the battle raging in the
 * distance is. Seven of them are jungles of one kind or another. The cave is kept for the 404.
 */
import {
  ASH,
  CLAY,
  LOAM,
  GRAVEL,
  BRICK,
  CONCRETE,
  CRATE,
  DIRT,
  GRASS,
  HIVE,
  ICE,
  METAL,
  MOSS,
  MUD,
  REDROCK,
  ROCK,
  SANDBAG,
  SAND,
  SANDSTONE,
  SLIME,
  SNOW,
  STONE,
  WOOD,
} from "./level";

export type Feature =
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
  | "nuclear";

/** Foot soldiers a theme fields. */
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
export type BossKind =
  "tank" | "mech" | "gunship" | "worm" | "airship" | "cannon" | "spider";
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
export type Ambient =
  | "fireflies"
  | "smoke"
  | "embers"
  | "ash"
  | "snow"
  | "rain"
  | "storm"
  | "sand"
  | "spores"
  | "none";

export interface Theme {
  key: string;
  /** Shown in the HUD and the level banner */
  name: string;
  /** Sky gradient, top → horizon */
  sky: [string, string, string];
  /** How dark the world is tinted (0 day … 1 black) */
  night: number;
  /** A colour wash over the world (rgba), e.g. a warm dawn */
  tint: string;
  sun: "sun" | "red" | "moon" | "none";
  backdrop: Backdrop;
  /** Enemy uniforms */
  look: Backdrop;
  /** Ground: top block, the few below it, and the deep rock */
  top: number;
  fill: number;
  deep: number;
  /** Obstacles and buildings; `skyline` builds big ruins in the background (0: none) */
  wall: number;
  build: number;
  skyline: number;
  decor:
    | "palms"
    | "jungle"
    | "pines"
    | "deadwood"
    | "fires"
    | "crystals"
    | "rubble"
    | "none"
    | "autumn"
    | "sakura";
  /** How often each feature comes up (relative weights) */
  features: Partial<Record<Feature, number>>;
  troops: Partial<Record<Troop, number>>;
  /** Enemies per 12 columns */
  enemies: number;
  /** Chance of a barrel cluster per chunk */
  barrels: number;
  /** Artillery shells and meteors per second; enemy helicopters, paratroopers, bombing jets and drone swarms per minute */
  artillery: number;
  meteors: number;
  helis: number;
  paras: number;
  jets: number;
  drones: number;
  ambient: Ambient;
  /** How much of a battle rages in the distance behind the level (0 none … 1 a lot) */
  distant: number;
  /** How fast the squad pushes forward, px/s */
  speed: number;
  /** How mountainous the ground is (0 flat … 1.5 alpine): scales the hills and ranges the generator raises */
  relief: number;
  bosses: BossKind[];
  /** A cave roof over the level */
  ceiling?: boolean;
}

const base = {
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

export const THEMES: Record<string, Theme> = {
  jungle: {
    ...base,
    key: "jungle",
    name: "Jungle landing",
    sky: ["#2a2d6e", "#b8506e", "#ffa65a"],
    night: 0.1,
    tint: "rgba(255,110,40,0.1)",
    sun: "red",
    backdrop: "jungle",
    look: "jungle",
    top: GRASS,
    fill: DIRT,
    deep: STONE,
    wall: SANDBAG,
    build: WOOD,
    decor: "palms",
    features: {
      flat: 3,
      slope: 2,
      gap: 1,
      bridge: 1,
      wall: 2,
      huts: 3,
      tower: 2,
      tunnel: 1,
      mgnest: 1,
      canopy: 2,
      river: 3.0,
      camp: 1,
      bamboo: 1,
      downtown: 4.0,
      harbour: 1,
      waterfall: 1,
      ocean: 2.5,
      forest: 2,
      lake: 1.5,
      railway: 0.5,
      metropolis: 1.5,
      highway: 1,
      airfield: 0.5,
    },
    troops: {
      mook: 6,
      heavy: 1,
      rocket: 1,
      sniper: 1,
      dog: 1,
      flamer: 1,
      rusher: 1.5,
      dynamiter: 1,
      minigunner: 1,
      officer: 0.6,
      engineer: 0.6,
      jetpacker: 0.8,
    },
    enemies: 4.8,
    barrels: 0.5,
    artillery: 0.5,
    helis: 0.8,
    paras: 0.8,
    jets: 0.8,
    ambient: "fireflies",
    bosses: ["gunship", "tank", "airship"],
  },
  deepjungle: {
    ...base,
    key: "deepjungle",
    name: "Deep jungle",
    sky: ["#1e4a44", "#6aa070", "#e0e8a0"],
    night: 0.12,
    tint: "rgba(90,210,120,0.07)",
    backdrop: "deepjungle",
    relief: 1.2,
    look: "deepjungle",
    top: GRASS,
    fill: DIRT,
    deep: STONE,
    wall: WOOD,
    build: WOOD,
    decor: "jungle",
    features: {
      flat: 2,
      canopy: 5,
      bamboo: 3,
      river: 4.0,
      waterfall: 2,
      camp: 2,
      huts: 1,
      mgnest: 1,
      tunnel: 1,
      slope: 1,
      downtown: 1.5,
      forest: 1.5,
      lake: 1.5,
      ocean: 0.5,
    },
    troops: {
      mook: 5,
      dog: 2,
      flamer: 1,
      sniper: 2,
      grenadier: 1,
      rusher: 2,
      dynamiter: 1,
      officer: 0.6,
      engineer: 0.6,
    },
    enemies: 5.8,
    barrels: 0.5,
    artillery: 0.6,
    helis: 1.2,
    paras: 1.2,
    jets: 1,
    ambient: "fireflies",
    distant: 0.5,
    bosses: ["gunship", "tank", "worm", "spider"],
  },
  temple: {
    ...base,
    key: "temple",
    name: "Lost temple",
    sky: ["#3a2a5a", "#c8706a", "#ffc888"],
    night: 0.08,
    tint: "rgba(255,160,80,0.08)",
    sun: "red",
    backdrop: "temple",
    relief: 1.1,
    look: "temple",
    top: GRASS,
    fill: DIRT,
    deep: STONE,
    wall: MOSS,
    build: MOSS,
    skyline: MOSS,
    decor: "jungle",
    features: {
      temple: 5,
      statues: 3,
      canopy: 2,
      waterfall: 2,
      river: 2.0,
      camp: 1,
      flat: 1,
      mgnest: 1,
      slope: 1,
      downtown: 2.0,
      forest: 1,
      lake: 1,
    },
    troops: {
      mook: 5,
      shield: 1,
      flamer: 1,
      grenadier: 1,
      sniper: 1,
      mortar: 1,
      rusher: 1.5,
      dynamiter: 1,
      minigunner: 0.8,
      laser: 0.5,
    },
    enemies: 6.2,
    barrels: 0.5,
    artillery: 0.8,
    helis: 1.4,
    paras: 1.5,
    jets: 1.2,
    ambient: "fireflies",
    distant: 0.5,
    bosses: ["mech", "worm", "spider"],
  },
  monsoon: {
    ...base,
    key: "monsoon",
    name: "Monsoon river",
    sky: ["#141c26", "#33424e", "#5e6e74"],
    night: 0.3,
    tint: "rgba(110,160,200,0.07)",
    sun: "none",
    backdrop: "monsoon",
    relief: 0.85,
    look: "monsoon",
    top: GRASS,
    fill: MUD,
    deep: STONE,
    wall: SANDBAG,
    build: WOOD,
    decor: "jungle",
    features: {
      river: 6.0,
      stilts: 3,
      bridge: 2,
      bamboo: 2,
      canopy: 2,
      huts: 2,
      mgnest: 1,
      road: 1,
      flat: 1,
      downtown: 3.0,
      harbour: 2,
      waterfall: 1,
      ocean: 4.0,
      forest: 1,
      lake: 1.5,
    },
    troops: {
      mook: 5,
      dog: 1,
      rocket: 1,
      grenadier: 1,
      sniper: 1,
      rusher: 1,
      dynamiter: 1,
      minigunner: 1,
      engineer: 0.6,
      jetpacker: 0.6,
    },
    enemies: 6,
    barrels: 0.5,
    artillery: 0.8,
    helis: 1.6,
    paras: 1.5,
    jets: 1,
    ambient: "storm",
    distant: 0.6,
    bosses: ["gunship", "tank", "airship"],
  },
  nightraid: {
    ...base,
    key: "nightraid",
    name: "Night raid",
    sky: ["#03050b", "#0b1a20", "#1a3030"],
    night: 0.5,
    sun: "moon",
    backdrop: "deepjungle",
    look: "deepjungle",
    top: GRASS,
    fill: DIRT,
    deep: STONE,
    wall: SANDBAG,
    build: WOOD,
    decor: "jungle",
    features: {
      canopy: 4,
      camp: 4,
      tower: 2,
      bamboo: 2,
      river: 3.5,
      mgnest: 2,
      road: 1,
      tunnel: 1,
      flat: 1,
      downtown: 2.5,
      waterfall: 1,
      ocean: 2.0,
      forest: 1.5,
      lake: 1,
    },
    troops: {
      mook: 5,
      dog: 2,
      flamer: 2,
      sniper: 1,
      shield: 1,
      rusher: 1.5,
      minigunner: 1,
      officer: 0.8,
      laser: 0.6,
    },
    enemies: 6.6,
    barrels: 0.6,
    artillery: 1,
    helis: 2,
    paras: 2,
    jets: 1.4,
    drones: 0.6,
    ambient: "fireflies",
    distant: 0.8,
    bosses: ["gunship", "mech", "airship"],
  },
  village: {
    ...base,
    key: "village",
    name: "River village",
    sky: ["#2f6fcf", "#5aa7e6", "#a9dcf0"],
    night: 0,
    backdrop: "jungle",
    look: "jungle",
    top: GRASS,
    fill: DIRT,
    deep: STONE,
    wall: CRATE,
    build: WOOD,
    decor: "palms",
    features: {
      flat: 2,
      slope: 2,
      bridge: 2,
      huts: 4,
      stilts: 2,
      tower: 2,
      wall: 2,
      mgnest: 1,
      tunnel: 1,
      river: 3.5,
      camp: 1,
      downtown: 7.0,
      harbour: 1,
      waterfall: 1,
      ocean: 2.0,
      forest: 2,
      lake: 1.5,
      railway: 1,
      metropolis: 2,
      highway: 1.5,
      airfield: 0.5,
    },
    troops: {
      mook: 6,
      heavy: 1,
      rocket: 1,
      sniper: 1,
      dog: 1,
      grenadier: 1,
      rusher: 1,
      dynamiter: 1.5,
      minigunner: 1,
      engineer: 0.8,
    },
    enemies: 5.2,
    barrels: 0.6,
    artillery: 0.7,
    helis: 1,
    paras: 1.5,
    jets: 1,
    ambient: "none",
    distant: 0.4,
    bosses: ["gunship", "tank", "airship"],
  },
  desert: {
    ...base,
    key: "desert",
    name: "Desert outpost",
    sky: ["#3a7ad0", "#8ec4e8", "#f4dca8"],
    night: 0,
    tint: "rgba(255,190,90,0.08)",
    backdrop: "desert",
    relief: 0.8,
    look: "desert",
    top: SAND,
    fill: SANDSTONE,
    deep: SANDSTONE,
    wall: SANDBAG,
    build: SANDSTONE,
    skyline: SANDSTONE,
    decor: "rubble",
    features: {
      flat: 2,
      slope: 3,
      wall: 2,
      bunker: 2,
      derrick: 3,
      tanker: 2,
      mgnest: 2,
      convoy: 2,
      tunnel: 1,
      road: 1,
      downtown: 5.0,
      river: 2.0,
      lake: 0.5,
      railway: 1.5,
      nuclear: 1.5,
      ocean: 1,
      metropolis: 1.5,
      highway: 2.5,
      airfield: 1.5,
    },
    troops: {
      mook: 5,
      heavy: 1,
      rocket: 2,
      sniper: 1,
      mortar: 1,
      dog: 1,
      grenadier: 1,
      minigunner: 1.5,
      dynamiter: 1.2,
      engineer: 1,
      officer: 0.8,
      exosuit: 0.5,
      jetpacker: 0.8,
    },
    enemies: 5.6,
    barrels: 0.7,
    artillery: 0.9,
    helis: 1.4,
    paras: 2,
    jets: 1.4,
    ambient: "sand",
    bosses: ["tank", "mech", "cannon"],
  },
  canyon: {
    ...base,
    key: "canyon",
    name: "Red canyon",
    sky: ["#4a6fb0", "#e8a070", "#ffd0a0"],
    night: 0.04,
    tint: "rgba(255,120,60,0.08)",
    backdrop: "canyon",
    relief: 1.25,
    look: "desert",
    top: SAND,
    fill: REDROCK,
    deep: REDROCK,
    wall: REDROCK,
    build: WOOD,
    decor: "rubble",
    features: {
      flat: 1,
      slope: 2,
      mesa: 4,
      bridge: 3,
      gap: 2,
      mgnest: 1,
      tower: 2,
      convoy: 1,
      downtown: 2.5,
      river: 2.5,
      waterfall: 1,
      forest: 0.5,
      lake: 0.5,
      railway: 1,
    },
    troops: {
      mook: 5,
      sniper: 2,
      rocket: 1,
      mortar: 1,
      grenadier: 1,
      minigunner: 1.2,
      dynamiter: 1.5,
      rusher: 1,
      officer: 0.6,
    },
    enemies: 5.4,
    barrels: 0.6,
    artillery: 0.8,
    helis: 1.6,
    paras: 1.5,
    jets: 1.2,
    ambient: "sand",
    distant: 0.5,
    bosses: ["mech", "gunship", "cannon"],
  },
  snow: {
    ...base,
    key: "snow",
    name: "Frozen peaks",
    sky: ["#7a90b8", "#bccadc", "#eef2f8"],
    night: 0.06,
    tint: "rgba(200,220,255,0.08)",
    sun: "none",
    backdrop: "snow",
    relief: 1.45,
    look: "snow",
    top: SNOW,
    fill: STONE,
    deep: STONE,
    wall: ICE,
    build: CONCRETE,
    decor: "pines",
    features: {
      flat: 2,
      slope: 3,
      pines: 3,
      bunker: 2,
      mgnest: 2,
      gap: 1,
      tower: 1,
      tunnel: 1,
      radar: 1,
      downtown: 3.5,
      river: 3.0,
      waterfall: 2,
      ocean: 2.0,
      forest: 3,
      lake: 1.5,
      railway: 1,
      nuclear: 1.5,
      metropolis: 1.5,
      highway: 1.5,
      airfield: 1,
    },
    troops: {
      mook: 6,
      heavy: 2,
      sniper: 1,
      rocket: 1,
      dog: 1,
      shield: 1,
      minigunner: 1.2,
      engineer: 1,
      laser: 0.8,
      exosuit: 0.6,
      jetpacker: 0.8,
    },
    enemies: 5.8,
    barrels: 0.5,
    artillery: 1,
    helis: 1.6,
    paras: 2.5,
    jets: 1.4,
    ambient: "snow",
    distant: 0.5,
    bosses: ["mech", "tank", "airship", "cannon"],
  },
  fortress: {
    ...base,
    key: "fortress",
    name: "Fortress",
    sky: ["#3a6fb8", "#8ab8d8", "#f0d8a8"],
    night: 0.04,
    backdrop: "fort",
    relief: 0.9,
    look: "fort",
    top: GRASS,
    fill: DIRT,
    deep: STONE,
    wall: SANDBAG,
    build: CONCRETE,
    skyline: CONCRETE,
    features: {
      flat: 1,
      wall: 2,
      bunker: 4,
      tower: 1,
      mgnest: 2,
      tunnel: 2,
      fort: 1,
      radar: 1,
      road: 1,
      downtown: 5.0,
      harbour: 1,
      river: 2.0,
      ocean: 2.0,
      forest: 1,
      lake: 1,
      railway: 1.5,
      nuclear: 2.5,
      metropolis: 2,
      highway: 1.5,
      airfield: 2,
    },
    troops: {
      mook: 5,
      heavy: 2,
      rocket: 1,
      mortar: 1,
      shield: 2,
      dog: 1,
      grenadier: 1,
      minigunner: 1.5,
      officer: 1.2,
      engineer: 1,
      laser: 0.8,
      exosuit: 1,
      jetpacker: 1,
    },
    enemies: 6.4,
    barrels: 0.7,
    artillery: 1.1,
    helis: 1.6,
    paras: 2.5,
    jets: 1.6,
    drones: 0.5,
    ambient: "smoke",
    distant: 0.8,
    bosses: ["tank", "gunship", "cannon", "airship"],
  },
  refinery: {
    ...base,
    key: "refinery",
    name: "Refinery",
    sky: ["#1e1838", "#8a3a4a", "#ff8a3a"],
    night: 0.2,
    tint: "rgba(255,110,40,0.1)",
    sun: "red",
    backdrop: "factory",
    relief: 0.55,
    look: "factory",
    top: CONCRETE,
    fill: DIRT,
    deep: STONE,
    wall: METAL,
    build: METAL,
    skyline: METAL,
    features: {
      flat: 1,
      gap: 1,
      wall: 2,
      bunker: 1,
      catwalk: 4,
      tank: 3,
      tanker: 2,
      mgnest: 1,
      downtown: 8.0,
      harbour: 2,
      river: 2.0,
      ocean: 3.0,
      forest: 0.5,
      lake: 1,
      railway: 3,
      nuclear: 4.0,
      metropolis: 3,
      highway: 2,
      airfield: 0.5,
    },
    troops: {
      mook: 5,
      heavy: 2,
      rocket: 2,
      flamer: 2,
      shield: 1,
      minigunner: 1.2,
      engineer: 1.5,
      dynamiter: 1,
      exosuit: 0.6,
      jetpacker: 0.8,
    },
    enemies: 6.8,
    barrels: 0.85,
    artillery: 1.4,
    helis: 2.4,
    paras: 3,
    jets: 1.8,
    drones: 0.6,
    ambient: "embers",
    distant: 0.8,
    bosses: ["mech", "gunship", "cannon"],
  },
  city: {
    ...base,
    key: "city",
    name: "City ruins",
    sky: ["#05060f", "#141b3a", "#2a2f5a"],
    night: 0.5,
    sun: "moon",
    backdrop: "city",
    relief: 0.5,
    look: "city",
    top: CONCRETE,
    fill: DIRT,
    deep: STONE,
    wall: SANDBAG,
    build: BRICK,
    skyline: BRICK,
    decor: "rubble",
    features: {
      ruins: 5,
      flat: 1,
      wreck: 2,
      mgnest: 2,
      convoy: 1,
      tanker: 1,
      tunnel: 1,
      wall: 1,
      road: 1,
      downtown: 17.0,
      harbour: 2,
      river: 2.5,
      ocean: 4.0,
      forest: 0.5,
      lake: 1,
      railway: 2,
      nuclear: 2.5,
      metropolis: 8,
      highway: 3,
      airfield: 1,
    },
    troops: {
      mook: 5,
      heavy: 2,
      sniper: 2,
      rocket: 2,
      shield: 2,
      grenadier: 1,
      flamer: 1,
      minigunner: 1.5,
      rusher: 1,
      officer: 1,
      laser: 1,
      exosuit: 0.8,
      jetpacker: 1.2,
    },
    enemies: 7,
    barrels: 0.8,
    artillery: 1.4,
    helis: 2.4,
    paras: 3,
    jets: 2.2,
    drones: 1,
    ambient: "embers",
    distant: 1,
    bosses: ["mech", "gunship", "airship", "spider"],
  },
  inferno: {
    ...base,
    key: "inferno",
    name: "Inferno",
    sky: ["#1a0508", "#6a1010", "#ff5a1a"],
    night: 0.26,
    tint: "rgba(255,90,30,0.14)",
    sun: "red",
    backdrop: "volcano",
    relief: 1.3,
    look: "volcano",
    top: ASH,
    fill: ROCK,
    deep: ROCK,
    wall: ROCK,
    build: METAL,
    decor: "fires",
    features: {
      flat: 1,
      gap: 1,
      wall: 1,
      bunker: 1,
      lava: 4,
      spires: 3,
      wreck: 2,
      mgnest: 1,
      downtown: 0.5,
      lake: 0.5,
      nuclear: 1,
    },
    troops: {
      mook: 4,
      heavy: 2,
      crawler: 3,
      rocket: 1,
      flamer: 2,
      minigunner: 1,
      dynamiter: 1.2,
      exosuit: 0.8,
      laser: 0.6,
    },
    enemies: 8.4,
    barrels: 0.7,
    artillery: 2,
    meteors: 0.8,
    helis: 2.6,
    paras: 3.5,
    jets: 2.4,
    ambient: "ash",
    distant: 0.9,
    bosses: ["worm", "spider"],
  },
  hive: {
    ...base,
    key: "hive",
    name: "Alien hive",
    sky: ["#120a20", "#3a1450", "#7a2a8a"],
    night: 0.32,
    tint: "rgba(120,255,140,0.06)",
    sun: "none",
    backdrop: "hive",
    relief: 0.9,
    look: "hive",
    top: SLIME,
    fill: HIVE,
    deep: HIVE,
    wall: HIVE,
    build: HIVE,
    skyline: HIVE,
    decor: "crystals",
    features: {
      flat: 2,
      slope: 2,
      eggs: 4,
      spires: 3,
      gap: 1,
      tunnel: 1,
      downtown: 0.5,
    },
    troops: { crawler: 6, mook: 1, heavy: 1, laser: 2, exosuit: 1 },
    enemies: 7,
    barrels: 0.3,
    artillery: 0.6,
    meteors: 0.3,
    helis: 0.8,
    paras: 1,
    jets: 1.6,
    ambient: "spores",
    distant: 0.3,
    bosses: ["worm", "spider"],
  },
  swamp: {
    ...base,
    key: "swamp",
    name: "Black swamp",
    sky: ["#1e2a24", "#3e5a44", "#8aa07a"],
    night: 0.24,
    tint: "rgba(120,200,120,0.07)",
    sun: "none",
    backdrop: "swamp",
    relief: 0.6,
    look: "swamp",
    top: MUD,
    fill: DIRT,
    deep: STONE,
    wall: WOOD,
    build: WOOD,
    decor: "deadwood",
    features: {
      flat: 2,
      deadtrees: 3,
      stilts: 3,
      gap: 1,
      bridge: 2,
      mgnest: 1,
      huts: 1,
      river: 3.0,
      bamboo: 1,
      downtown: 2.5,
      waterfall: 1,
      ocean: 1.5,
      forest: 1.5,
      lake: 2.5,
    },
    troops: {
      mook: 5,
      crawler: 2,
      rocket: 1,
      sniper: 1,
      flamer: 1,
      dog: 1,
      rusher: 1.5,
      dynamiter: 1,
      engineer: 0.6,
    },
    enemies: 6,
    barrels: 0.5,
    artillery: 0.8,
    helis: 1.6,
    paras: 1.5,
    jets: 1.2,
    ambient: "rain",
    distant: 0.4,
    bosses: ["worm", "gunship", "spider"],
  },
  silos: {
    ...base,
    key: "silos",
    name: "Missile silos",
    sky: ["#2a1e3a", "#9a4a4a", "#ffb070"],
    night: 0.14,
    tint: "rgba(255,140,80,0.08)",
    sun: "red",
    backdrop: "silo",
    relief: 0.8,
    look: "fort",
    top: CONCRETE,
    fill: DIRT,
    deep: STONE,
    wall: SANDBAG,
    build: CONCRETE,
    skyline: CONCRETE,
    features: {
      silo: 4,
      bunker: 2,
      mgnest: 2,
      radar: 2,
      convoy: 2,
      tanker: 1,
      wall: 1,
      flat: 1,
      road: 1,
      downtown: 5.0,
      harbour: 1,
      ocean: 2.0,
      forest: 1,
      lake: 1,
      river: 1.5,
      railway: 1.5,
      nuclear: 4.0,
      metropolis: 1.5,
      highway: 2,
      airfield: 2.5,
    },
    troops: {
      mook: 5,
      heavy: 2,
      rocket: 2,
      mortar: 1,
      shield: 1,
      grenadier: 1,
      laser: 1.5,
      officer: 1,
      engineer: 1,
      exosuit: 1,
      jetpacker: 1,
    },
    enemies: 6.6,
    barrels: 0.7,
    artillery: 1.2,
    helis: 2,
    paras: 2.5,
    jets: 2,
    drones: 0.8,
    ambient: "none",
    distant: 0.8,
    bosses: ["mech", "tank", "cannon", "airship"],
  },
  hq: {
    ...base,
    key: "hq",
    name: "HQ assault",
    sky: ["#07060f", "#1a1030", "#3a1a3a"],
    night: 0.48,
    sun: "moon",
    backdrop: "fort",
    relief: 0.7,
    look: "city",
    top: CONCRETE,
    fill: CONCRETE,
    deep: STONE,
    wall: SANDBAG,
    build: CONCRETE,
    skyline: CONCRETE,
    features: {
      flat: 1,
      wall: 2,
      bunker: 3,
      fort: 3,
      catwalk: 1,
      mgnest: 2,
      radar: 1,
      tunnel: 1,
      road: 1,
      downtown: 6.0,
      harbour: 1,
      river: 2.0,
      ocean: 2.5,
      forest: 1,
      lake: 0.5,
      railway: 1.5,
      nuclear: 3.0,
      metropolis: 2.5,
      highway: 2,
      airfield: 2.5,
    },
    troops: {
      mook: 5,
      heavy: 2,
      rocket: 2,
      sniper: 1,
      shield: 2,
      flamer: 1,
      dog: 1,
      minigunner: 1.5,
      officer: 1.5,
      laser: 1.2,
      exosuit: 1.2,
      jetpacker: 1.2,
    },
    enemies: 7,
    barrels: 0.7,
    artillery: 1.2,
    helis: 2,
    paras: 3,
    jets: 2,
    drones: 1.2,
    ambient: "smoke",
    distant: 1,
    bosses: ["tank", "gunship", "airship", "spider"],
  },
  savanna: {
    ...base,
    key: "savanna",
    name: "Savanna",
    sky: ["#3a6ab0", "#e8b060", "#ffd890"],
    night: 0.05,
    tint: "rgba(255,200,80,0.1)",
    backdrop: "savanna",
    relief: 0.6,
    look: "desert",
    top: GRASS,
    fill: CLAY,
    deep: REDROCK,
    wall: SANDBAG,
    build: WOOD,
    decor: "palms",
    features: {
      flat: 3,
      slope: 1,
      huts: 2,
      camp: 2,
      river: 2,
      lake: 1,
      tower: 1,
      convoy: 2,
      mgnest: 2,
      airfield: 1,
      highway: 1,
      downtown: 2,
      forest: 0.5,
    },
    troops: {
      mook: 5,
      heavy: 1,
      rocket: 1,
      sniper: 1,
      dog: 2,
      rusher: 1.5,
      dynamiter: 1,
      minigunner: 1,
    },
    enemies: 5,
    barrels: 0.5,
    artillery: 0.6,
    helis: 1,
    paras: 1,
    jets: 1,
    ambient: "none",
    bosses: ["tank", "airship", "cannon"],
  },
  autumn: {
    ...base,
    key: "autumn",
    name: "Autumn woods",
    sky: ["#2a3a6a", "#c87850", "#f0c070"],
    night: 0.08,
    tint: "rgba(255,120,40,0.08)",
    backdrop: "autumn",
    relief: 0.9,
    look: "jungle",
    top: GRASS,
    fill: LOAM,
    deep: STONE,
    wall: WOOD,
    build: WOOD,
    decor: "autumn",
    features: {
      flat: 2,
      slope: 2,
      forest: 4,
      river: 2,
      lake: 1.5,
      waterfall: 1,
      huts: 2,
      camp: 1,
      mgnest: 2,
      downtown: 2,
      highway: 1,
      railway: 1,
    },
    troops: {
      mook: 5,
      sniper: 2,
      dog: 1,
      grenadier: 1,
      rusher: 1,
      minigunner: 1,
      engineer: 1,
    },
    enemies: 5,
    barrels: 0.5,
    artillery: 0.6,
    helis: 1,
    paras: 1,
    jets: 1,
    ambient: "none",
    bosses: ["mech", "tank", "spider"],
  },
  sakura: {
    ...base,
    key: "sakura",
    name: "Blossom valley",
    sky: ["#5a7ac8", "#f0b8d0", "#fff0f4"],
    night: 0.04,
    tint: "rgba(255,170,210,0.08)",
    backdrop: "sakura",
    relief: 1,
    look: "temple",
    top: GRASS,
    fill: LOAM,
    deep: STONE,
    wall: WOOD,
    build: WOOD,
    decor: "sakura",
    features: {
      flat: 2,
      slope: 2,
      forest: 3,
      river: 2,
      waterfall: 2,
      lake: 2,
      downtown: 3,
      bridge: 1,
      camp: 1,
      mgnest: 1,
      temple: 1,
    },
    troops: {
      mook: 4,
      rusher: 2,
      sniper: 1,
      laser: 1,
      minigunner: 1,
      officer: 1,
    },
    enemies: 5,
    barrels: 0.4,
    artillery: 0.5,
    helis: 1,
    paras: 1,
    jets: 1,
    ambient: "none",
    bosses: ["mech", "gunship", "spider"],
  },
  tundra: {
    ...base,
    key: "tundra",
    name: "Frozen tundra",
    sky: ["#8aa0c0", "#d8e4f0", "#f4f8ff"],
    night: 0.1,
    backdrop: "tundra",
    relief: 0.4,
    look: "snow",
    top: SNOW,
    fill: GRAVEL,
    deep: STONE,
    wall: ICE,
    build: CONCRETE,
    decor: "pines",
    features: {
      flat: 3,
      slope: 1,
      pines: 2,
      bunker: 2,
      mgnest: 2,
      radar: 2,
      lake: 2,
      railway: 1,
      airfield: 1,
      nuclear: 1,
      convoy: 2,
      highway: 1,
    },
    troops: {
      mook: 4,
      heavy: 2,
      sniper: 2,
      minigunner: 1,
      exosuit: 1,
      engineer: 1,
    },
    enemies: 5,
    barrels: 0.5,
    artillery: 0.8,
    helis: 1.2,
    paras: 1,
    jets: 1.2,
    ambient: "snow",
    bosses: ["tank", "airship", "cannon"],
  },
  badlands: {
    ...base,
    key: "badlands",
    name: "Badlands",
    sky: ["#4a6ab0", "#e0905a", "#ffd0a0"],
    night: 0.06,
    tint: "rgba(255,130,60,0.08)",
    backdrop: "badlands",
    relief: 1.3,
    look: "canyon",
    top: REDROCK,
    fill: CLAY,
    deep: SANDSTONE,
    wall: ROCK,
    build: WOOD,
    decor: "deadwood",
    features: {
      flat: 2,
      slope: 3,
      mesa: 3,
      gap: 1,
      bridge: 2,
      mgnest: 2,
      convoy: 2,
      railway: 2,
      downtown: 1,
      highway: 1,
    },
    troops: {
      mook: 4,
      dynamiter: 2,
      minigunner: 1,
      rocket: 1,
      sniper: 2,
      rusher: 1,
    },
    enemies: 5,
    barrels: 0.6,
    artillery: 0.7,
    helis: 1,
    paras: 1,
    jets: 1,
    ambient: "sand",
    bosses: ["cannon", "tank", "spider"],
  },
  farmland: {
    ...base,
    key: "farmland",
    name: "Farmland",
    sky: ["#4a8ad0", "#a8d0f0", "#f0f8ff"],
    night: 0.02,
    backdrop: "farmland",
    relief: 0.5,
    look: "jungle",
    top: GRASS,
    fill: LOAM,
    deep: STONE,
    wall: WOOD,
    build: WOOD,
    decor: "jungle",
    features: {
      flat: 3,
      slope: 1,
      downtown: 4,
      forest: 1.5,
      river: 2,
      lake: 1.5,
      railway: 1.5,
      highway: 2,
      mgnest: 1,
      camp: 1,
      airfield: 0.5,
    },
    troops: {
      mook: 5,
      heavy: 1,
      dog: 1,
      rocket: 1,
      grenadier: 1,
      minigunner: 1,
    },
    enemies: 5,
    barrels: 0.6,
    artillery: 0.6,
    helis: 1,
    paras: 1.2,
    jets: 1.2,
    ambient: "none",
    bosses: ["tank", "mech", "airship"],
  },
  suburbs: {
    ...base,
    key: "suburbs",
    name: "Suburbia",
    sky: ["#3a7ac8", "#90c0e8", "#e8f4ff"],
    night: 0.03,
    backdrop: "farmland",
    relief: 0.4,
    look: "city",
    top: GRASS,
    fill: LOAM,
    deep: STONE,
    wall: SANDBAG,
    build: BRICK,
    decor: "jungle",
    features: {
      flat: 1,
      downtown: 6,
      metropolis: 1,
      highway: 3,
      forest: 1,
      lake: 1,
      river: 1,
      mgnest: 2,
      convoy: 1,
    },
    troops: {
      mook: 5,
      heavy: 1,
      rocket: 1,
      shield: 1,
      rusher: 1,
      engineer: 1,
      minigunner: 1,
    },
    enemies: 5,
    barrels: 0.6,
    artillery: 0.7,
    helis: 1.2,
    paras: 1.2,
    jets: 1.2,
    ambient: "none",
    bosses: ["tank", "mech", "airship"],
  },
  megacity: {
    ...base,
    key: "megacity",
    name: "Megacity",
    sky: ["#0a0a1a", "#2a1a4a", "#6a3a6a"],
    night: 0.42,
    sun: "moon",
    backdrop: "megacity",
    relief: 0.3,
    look: "city",
    top: CONCRETE,
    fill: DIRT,
    deep: STONE,
    wall: SANDBAG,
    build: CONCRETE,
    skyline: CONCRETE,
    features: {
      metropolis: 8,
      downtown: 3,
      highway: 3,
      railway: 1,
      harbour: 1,
      ocean: 0.5,
      nuclear: 0.5,
      airfield: 0.5,
      mgnest: 2,
    },
    troops: {
      mook: 5,
      heavy: 2,
      rocket: 2,
      shield: 2,
      laser: 1.5,
      minigunner: 1.5,
      exosuit: 1,
      officer: 1,
      jetpacker: 1,
    },
    enemies: 6,
    barrels: 0.7,
    artillery: 1,
    helis: 2,
    paras: 2,
    jets: 2,
    drones: 1.5,
    ambient: "rain",
    distant: 1,
    bosses: ["airship", "spider", "gunship", "mech"],
  },
  airbase: {
    ...base,
    key: "airbase",
    name: "Air base",
    sky: ["#3a5a90", "#8ab0d8", "#e0f0ff"],
    night: 0.05,
    backdrop: "airbase",
    relief: 0.3,
    look: "fort",
    top: GRASS,
    fill: DIRT,
    deep: STONE,
    wall: SANDBAG,
    build: CONCRETE,
    features: {
      airfield: 5,
      bunker: 2,
      mgnest: 2,
      radar: 2,
      downtown: 1,
      highway: 1,
      flat: 1,
      convoy: 1,
    },
    troops: {
      mook: 4,
      heavy: 1,
      rocket: 2,
      sniper: 1,
      officer: 1,
      engineer: 1,
      exosuit: 1,
      jetpacker: 1,
    },
    enemies: 6,
    barrels: 0.8,
    artillery: 1,
    helis: 2,
    paras: 2,
    jets: 2.5,
    drones: 1,
    ambient: "none",
    bosses: ["gunship", "airship", "tank"],
  },
  archipelago: {
    ...base,
    key: "archipelago",
    name: "Archipelago",
    sky: ["#2a7ac8", "#8ad0f0", "#e8fcff"],
    night: 0.03,
    backdrop: "archipelago",
    relief: 0.5,
    look: "monsoon",
    top: SAND,
    fill: SAND,
    deep: SANDSTONE,
    wall: SANDBAG,
    build: WOOD,
    decor: "palms",
    features: {
      ocean: 5,
      lake: 1,
      flat: 1,
      huts: 2,
      harbour: 2,
      camp: 1,
      mgnest: 1,
      downtown: 1,
    },
    troops: {
      mook: 5,
      rocket: 1,
      sniper: 1,
      grenadier: 1,
      rusher: 1,
      dynamiter: 1,
    },
    enemies: 5,
    barrels: 0.5,
    artillery: 0.6,
    helis: 1.4,
    paras: 1,
    jets: 1.4,
    ambient: "none",
    bosses: ["gunship", "airship"],
  },
  mangrove: {
    ...base,
    key: "mangrove",
    name: "Mangrove delta",
    sky: ["#2a4a40", "#6a9a7a", "#c8e0b0"],
    night: 0.1,
    tint: "rgba(90,200,140,0.06)",
    backdrop: "mangrove",
    relief: 0.3,
    look: "swamp",
    top: MUD,
    fill: MUD,
    deep: STONE,
    wall: WOOD,
    build: WOOD,
    decor: "jungle",
    features: {
      river: 4,
      lake: 3,
      stilts: 3,
      bamboo: 1,
      ocean: 1,
      camp: 1,
      flat: 1,
      canopy: 1,
      downtown: 1,
    },
    troops: { mook: 5, flamer: 1, rusher: 2, dog: 1, dynamiter: 1, sniper: 1 },
    enemies: 5,
    barrels: 0.5,
    artillery: 0.5,
    helis: 1.2,
    paras: 1,
    jets: 1,
    ambient: "fireflies",
    bosses: ["worm", "spider", "gunship"],
  },
  cave: {
    ...base,
    key: "cave",
    name: "Underground",
    sky: ["#05040a", "#0b0a18", "#141228"],
    night: 0.55,
    sun: "none",
    backdrop: "cave",
    look: "cave",
    top: STONE,
    fill: STONE,
    deep: STONE,
    wall: ROCK,
    build: STONE,
    decor: "crystals",
    features: { flat: 2, slope: 2, gap: 1, wall: 1, spires: 2 },
    troops: { mook: 3, crawler: 2 },
    enemies: 3,
    barrels: 0.3,
    artillery: 0,
    helis: 0,
    paras: 0,
    jets: 0,
    ambient: "none",
    distant: 0,
    speed: 29,
    relief: 0.45,
    bosses: [],
    ceiling: true,
  },
};

/** The themes levels are drawn from (the cave is the 404's). */
export const ROTATION = Object.keys(THEMES).filter((k) => k !== "cave");

/**
 * Biomes dealt like cards from a shuffled deck: every one comes up once before any comes up
 * again, so there's no bias towards any of them (and never the same one twice running).
 */
export function themeDeck(rand: () => number) {
  let deck: string[] = [];
  return (current?: string): Theme => {
    if (!deck.length) {
      deck = [...ROTATION];
      for (let i = deck.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        [deck[i], deck[j]] = [deck[j], deck[i]];
      }
      if (deck[deck.length - 1] === current)
        [deck[0], deck[deck.length - 1]] = [deck[deck.length - 1], deck[0]];
    }
    let key = deck.pop()!;
    if (key === current && deck.length) {
      const other = deck.pop()!;
      deck.unshift(key);
      key = other;
    }
    return THEMES[key];
  };
}

const mixHex = (a: string, b: string, t: number) => {
  const pa = parseInt(a.slice(1), 16);
  const pb = parseInt(b.slice(1), 16);
  const ch = (sh: number) =>
    Math.round(
      ((pa >> sh) & 255) + (((pb >> sh) & 255) - ((pa >> sh) & 255)) * t,
    );
  return (
    "#" + [16, 8, 0].map((sh) => ch(sh).toString(16).padStart(2, "0")).join("")
  );
};

/**
 * A one-off version of a biome, so no two levels are quite alike: a time of day (dawn, dusk,
 * night), weather (rain, a storm, fog, snow where it's cold), hills higher or lower, and its
 * features in a different mix. The name says what's different.
 */
export function variant(t: Theme, rand: () => number): Theme {
  const v: Theme = {
    ...t,
    features: { ...t.features },
    sky: [...t.sky] as Theme["sky"],
  };
  const time = rand();
  const notes: string[] = [];
  if (t.backdrop !== "cave" && t.night < 0.3) {
    if (time < 0.14) {
      v.night = 0.42;
      v.sun = "moon";
      v.sky = [
        mixHex(t.sky[0], "#05060f", 0.8),
        mixHex(t.sky[1], "#14102a", 0.75),
        mixHex(t.sky[2], "#2a1a3a", 0.7),
      ];
      v.tint = "";
      notes.push("at night");
    } else if (time < 0.3) {
      v.sun = "red";
      v.night = Math.max(t.night, 0.12);
      v.sky = [
        mixHex(t.sky[0], "#2a2d6e", 0.5),
        mixHex(t.sky[1], "#c85a5a", 0.55),
        mixHex(t.sky[2], "#ffa65a", 0.6),
      ];
      v.tint = "rgba(255,110,40,0.1)";
      notes.push("at dusk");
    } else if (time < 0.4) {
      v.sky = [
        mixHex(t.sky[0], "#3a3a7a", 0.4),
        mixHex(t.sky[1], "#e898a8", 0.5),
        mixHex(t.sky[2], "#ffe0c0", 0.5),
      ];
      v.tint = "rgba(255,160,180,0.07)";
      notes.push("at dawn");
    }
  }
  const weather = rand();
  const cold =
    t.top === SNOW || t.backdrop === "snow" || t.backdrop === "tundra";
  if (weather < 0.12 && t.backdrop !== "cave") {
    v.ambient = "storm";
    notes.push("in a storm");
  } else if (weather < 0.24) {
    v.ambient = cold ? "snow" : "rain";
    notes.push(cold ? "in the snow" : "in the rain");
  } else if (weather < 0.3 && t.backdrop !== "cave") {
    v.tint = "rgba(200,210,220,0.16)";
    v.ambient = "smoke";
    notes.push("in fog");
  }
  v.relief = Math.max(0.2, Math.min(1.6, t.relief * (0.6 + rand() * 0.8)));
  for (const k of Object.keys(v.features) as Feature[])
    v.features[k] = (v.features[k] ?? 0) * (0.5 + rand());
  v.name = notes.length ? `${t.name} ${notes[0]}` : t.name;
  return v;
}

/** A random theme for the next level, never the one just played. */
export function nextTheme(rand: () => number, current?: string) {
  const pool = ROTATION.filter((k) => k !== current);
  return THEMES[pool[Math.floor(rand() * pool.length)]];
}

/** How the HUD names a theme's weather and light. */
export const WEATHER: Record<Ambient, string> = {
  fireflies: "Fireflies",
  smoke: "Smoke",
  embers: "Embers",
  ash: "Ash fall",
  snow: "Snow",
  rain: "Rain",
  storm: "Monsoon storm",
  sand: "Sandstorm",
  spores: "Spores",
  none: "Clear",
};
export const lightOf = (t: Theme) =>
  t.backdrop === "cave"
    ? "Underground"
    : t.night >= 0.45
      ? "Night"
      : t.night >= 0.2
        ? "Dusk"
        : t.sun === "red"
          ? "Sunset"
          : "Day";
