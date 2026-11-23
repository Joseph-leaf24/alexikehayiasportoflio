/**
 * The level themes. Every level picks one at random (never the same twice in a row), so the
 * scenery keeps changing and is different on every visit: what the ground is made of, what
 * gets built on it, who defends it, which boss waits half way, the sky, the weather, how hard
 * the artillery, jets, drones and paratroopers come down, and how big the battle raging in the
 * distance is. Seven of them are jungles of one kind or another. The cave is kept for the 404.
 */
import {
  ASH,
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
  | "camp";

/** Foot soldiers a theme fields. */
export type Troop = "mook" | "heavy" | "rocket" | "sniper" | "crawler" | "mortar" | "dog" | "shield" | "flamer" | "grenadier";
export type BossKind = "tank" | "mech" | "gunship" | "worm";
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
  | "cave";
export type Ambient = "fireflies" | "smoke" | "embers" | "ash" | "snow" | "rain" | "storm" | "sand" | "spores" | "none";

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
  decor: "palms" | "jungle" | "pines" | "deadwood" | "fires" | "crystals" | "rubble" | "none";
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
  speed: 32,
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
    features: { flat: 3, slope: 2, gap: 1, bridge: 1, wall: 2, huts: 3, tower: 2, tunnel: 1, mgnest: 1, canopy: 2, river: 1, camp: 1, bamboo: 1 },
    troops: { mook: 6, heavy: 1, rocket: 1, sniper: 1, dog: 1, flamer: 1 },
    enemies: 4.8,
    barrels: 0.5,
    artillery: 0.5,
    helis: 0.8,
    paras: 0.8,
    jets: 0.8,
    ambient: "fireflies",
    bosses: ["gunship", "tank"],
  },
  deepjungle: {
    ...base,
    key: "deepjungle",
    name: "Deep jungle",
    sky: ["#1e4a44", "#6aa070", "#e0e8a0"],
    night: 0.12,
    tint: "rgba(90,210,120,0.07)",
    backdrop: "deepjungle",
    look: "deepjungle",
    top: GRASS,
    fill: DIRT,
    deep: STONE,
    wall: WOOD,
    build: WOOD,
    decor: "jungle",
    features: { flat: 2, canopy: 5, bamboo: 3, river: 3, waterfall: 2, camp: 2, huts: 1, mgnest: 1, tunnel: 1, slope: 1 },
    troops: { mook: 5, dog: 2, flamer: 1, sniper: 2, grenadier: 1 },
    enemies: 5.8,
    barrels: 0.5,
    artillery: 0.6,
    helis: 1.2,
    paras: 1.2,
    jets: 1,
    ambient: "fireflies",
    distant: 0.5,
    bosses: ["gunship", "tank", "worm"],
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
    look: "temple",
    top: GRASS,
    fill: DIRT,
    deep: STONE,
    wall: MOSS,
    build: MOSS,
    skyline: MOSS,
    decor: "jungle",
    features: { temple: 5, statues: 3, canopy: 2, waterfall: 2, river: 1, camp: 1, flat: 1, mgnest: 1, slope: 1 },
    troops: { mook: 5, shield: 1, flamer: 1, grenadier: 1, sniper: 1, mortar: 1 },
    enemies: 6.2,
    barrels: 0.5,
    artillery: 0.8,
    helis: 1.4,
    paras: 1.5,
    jets: 1.2,
    ambient: "fireflies",
    distant: 0.5,
    bosses: ["mech", "worm"],
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
    look: "monsoon",
    top: GRASS,
    fill: MUD,
    deep: STONE,
    wall: SANDBAG,
    build: WOOD,
    decor: "jungle",
    features: { river: 5, stilts: 3, bridge: 2, bamboo: 2, canopy: 2, huts: 2, mgnest: 1, road: 1, flat: 1 },
    troops: { mook: 5, dog: 1, rocket: 1, grenadier: 1, sniper: 1 },
    enemies: 6,
    barrels: 0.5,
    artillery: 0.8,
    helis: 1.6,
    paras: 1.5,
    jets: 1,
    ambient: "storm",
    distant: 0.6,
    bosses: ["gunship", "tank"],
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
    features: { canopy: 4, camp: 4, tower: 2, bamboo: 2, river: 2, mgnest: 2, road: 1, tunnel: 1, flat: 1 },
    troops: { mook: 5, dog: 2, flamer: 2, sniper: 1, shield: 1 },
    enemies: 6.6,
    barrels: 0.6,
    artillery: 1,
    helis: 2,
    paras: 2,
    jets: 1.4,
    drones: 0.6,
    ambient: "fireflies",
    distant: 0.8,
    bosses: ["gunship", "mech"],
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
    features: { flat: 2, slope: 2, bridge: 2, huts: 4, stilts: 2, tower: 2, wall: 2, mgnest: 1, tunnel: 1, river: 2, camp: 1 },
    troops: { mook: 6, heavy: 1, rocket: 1, sniper: 1, dog: 1, grenadier: 1 },
    enemies: 5.2,
    barrels: 0.6,
    artillery: 0.7,
    helis: 1,
    paras: 1.5,
    jets: 1,
    ambient: "none",
    distant: 0.4,
    bosses: ["gunship", "tank"],
  },
  desert: {
    ...base,
    key: "desert",
    name: "Desert outpost",
    sky: ["#3a7ad0", "#8ec4e8", "#f4dca8"],
    night: 0,
    tint: "rgba(255,190,90,0.08)",
    backdrop: "desert",
    look: "desert",
    top: SAND,
    fill: SANDSTONE,
    deep: SANDSTONE,
    wall: SANDBAG,
    build: SANDSTONE,
    skyline: SANDSTONE,
    decor: "rubble",
    features: { flat: 2, slope: 3, wall: 2, bunker: 2, derrick: 3, tanker: 2, mgnest: 2, convoy: 2, tunnel: 1, road: 1 },
    troops: { mook: 5, heavy: 1, rocket: 2, sniper: 1, mortar: 1, dog: 1, grenadier: 1 },
    enemies: 5.6,
    barrels: 0.7,
    artillery: 0.9,
    helis: 1.4,
    paras: 2,
    jets: 1.4,
    ambient: "sand",
    bosses: ["tank", "mech"],
  },
  canyon: {
    ...base,
    key: "canyon",
    name: "Red canyon",
    sky: ["#4a6fb0", "#e8a070", "#ffd0a0"],
    night: 0.04,
    tint: "rgba(255,120,60,0.08)",
    backdrop: "canyon",
    look: "desert",
    top: SAND,
    fill: REDROCK,
    deep: REDROCK,
    wall: REDROCK,
    build: WOOD,
    decor: "rubble",
    features: { flat: 1, slope: 2, mesa: 4, bridge: 3, gap: 2, mgnest: 1, tower: 2, convoy: 1 },
    troops: { mook: 5, sniper: 2, rocket: 1, mortar: 1, grenadier: 1 },
    enemies: 5.4,
    barrels: 0.6,
    artillery: 0.8,
    helis: 1.6,
    paras: 1.5,
    jets: 1.2,
    ambient: "sand",
    distant: 0.5,
    bosses: ["mech", "gunship"],
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
    look: "snow",
    top: SNOW,
    fill: STONE,
    deep: STONE,
    wall: ICE,
    build: CONCRETE,
    decor: "pines",
    features: { flat: 2, slope: 3, pines: 3, bunker: 2, mgnest: 2, gap: 1, tower: 1, tunnel: 1, radar: 1 },
    troops: { mook: 6, heavy: 2, sniper: 1, rocket: 1, dog: 1, shield: 1 },
    enemies: 5.8,
    barrels: 0.5,
    artillery: 1,
    helis: 1.6,
    paras: 2.5,
    jets: 1.4,
    ambient: "snow",
    distant: 0.5,
    bosses: ["mech", "tank"],
  },
  fortress: {
    ...base,
    key: "fortress",
    name: "Fortress",
    sky: ["#3a6fb8", "#8ab8d8", "#f0d8a8"],
    night: 0.04,
    backdrop: "fort",
    look: "fort",
    top: GRASS,
    fill: DIRT,
    deep: STONE,
    wall: SANDBAG,
    build: CONCRETE,
    skyline: CONCRETE,
    features: { flat: 1, wall: 2, bunker: 4, tower: 1, mgnest: 2, tunnel: 2, fort: 1, radar: 1, road: 1 },
    troops: { mook: 5, heavy: 2, rocket: 1, mortar: 1, shield: 2, dog: 1, grenadier: 1 },
    enemies: 6.4,
    barrels: 0.7,
    artillery: 1.1,
    helis: 1.6,
    paras: 2.5,
    jets: 1.6,
    drones: 0.5,
    ambient: "smoke",
    distant: 0.8,
    bosses: ["tank", "gunship"],
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
    look: "factory",
    top: CONCRETE,
    fill: DIRT,
    deep: STONE,
    wall: METAL,
    build: METAL,
    skyline: METAL,
    features: { flat: 1, gap: 1, wall: 2, bunker: 1, catwalk: 4, tank: 3, tanker: 2, mgnest: 1 },
    troops: { mook: 5, heavy: 2, rocket: 2, flamer: 2, shield: 1 },
    enemies: 6.8,
    barrels: 0.85,
    artillery: 1.4,
    helis: 2.4,
    paras: 3,
    jets: 1.8,
    drones: 0.6,
    ambient: "embers",
    distant: 0.8,
    bosses: ["mech", "gunship"],
  },
  city: {
    ...base,
    key: "city",
    name: "City ruins",
    sky: ["#05060f", "#141b3a", "#2a2f5a"],
    night: 0.5,
    sun: "moon",
    backdrop: "city",
    look: "city",
    top: CONCRETE,
    fill: DIRT,
    deep: STONE,
    wall: SANDBAG,
    build: BRICK,
    skyline: BRICK,
    decor: "rubble",
    features: { ruins: 5, flat: 1, wreck: 2, mgnest: 2, convoy: 1, tanker: 1, tunnel: 1, wall: 1, road: 1 },
    troops: { mook: 5, heavy: 2, sniper: 2, rocket: 2, shield: 2, grenadier: 1, flamer: 1 },
    enemies: 7,
    barrels: 0.8,
    artillery: 1.4,
    helis: 2.4,
    paras: 3,
    jets: 2.2,
    drones: 1,
    ambient: "embers",
    distant: 1,
    bosses: ["mech", "gunship"],
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
    look: "volcano",
    top: ASH,
    fill: ROCK,
    deep: ROCK,
    wall: ROCK,
    build: METAL,
    decor: "fires",
    features: { flat: 1, gap: 1, wall: 1, bunker: 1, lava: 4, spires: 3, wreck: 2, mgnest: 1 },
    troops: { mook: 4, heavy: 2, crawler: 3, rocket: 1, flamer: 2 },
    enemies: 8.4,
    barrels: 0.7,
    artillery: 2,
    meteors: 0.8,
    helis: 2.6,
    paras: 3.5,
    jets: 2.4,
    ambient: "ash",
    distant: 0.9,
    bosses: ["worm"],
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
    look: "hive",
    top: SLIME,
    fill: HIVE,
    deep: HIVE,
    wall: HIVE,
    build: HIVE,
    skyline: HIVE,
    decor: "crystals",
    features: { flat: 2, slope: 2, eggs: 4, spires: 3, gap: 1, tunnel: 1 },
    troops: { crawler: 6, mook: 1, heavy: 1 },
    enemies: 7,
    barrels: 0.3,
    artillery: 0.6,
    meteors: 0.3,
    helis: 0.8,
    paras: 1,
    jets: 1.6,
    ambient: "spores",
    distant: 0.3,
    bosses: ["worm"],
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
    look: "swamp",
    top: MUD,
    fill: DIRT,
    deep: STONE,
    wall: WOOD,
    build: WOOD,
    decor: "deadwood",
    features: { flat: 2, deadtrees: 3, stilts: 3, gap: 1, bridge: 2, mgnest: 1, huts: 1, river: 2, bamboo: 1 },
    troops: { mook: 5, crawler: 2, rocket: 1, sniper: 1, flamer: 1, dog: 1 },
    enemies: 6,
    barrels: 0.5,
    artillery: 0.8,
    helis: 1.6,
    paras: 1.5,
    jets: 1.2,
    ambient: "rain",
    distant: 0.4,
    bosses: ["worm", "gunship"],
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
    look: "fort",
    top: CONCRETE,
    fill: DIRT,
    deep: STONE,
    wall: SANDBAG,
    build: CONCRETE,
    skyline: CONCRETE,
    features: { silo: 4, bunker: 2, mgnest: 2, radar: 2, convoy: 2, tanker: 1, wall: 1, flat: 1, road: 1 },
    troops: { mook: 5, heavy: 2, rocket: 2, mortar: 1, shield: 1, grenadier: 1 },
    enemies: 6.6,
    barrels: 0.7,
    artillery: 1.2,
    helis: 2,
    paras: 2.5,
    jets: 2,
    drones: 0.8,
    ambient: "none",
    distant: 0.8,
    bosses: ["mech", "tank"],
  },
  hq: {
    ...base,
    key: "hq",
    name: "HQ assault",
    sky: ["#07060f", "#1a1030", "#3a1a3a"],
    night: 0.48,
    sun: "moon",
    backdrop: "fort",
    look: "city",
    top: CONCRETE,
    fill: CONCRETE,
    deep: STONE,
    wall: SANDBAG,
    build: CONCRETE,
    skyline: CONCRETE,
    features: { flat: 1, wall: 2, bunker: 3, fort: 3, catwalk: 1, mgnest: 2, radar: 1, tunnel: 1, road: 1 },
    troops: { mook: 5, heavy: 2, rocket: 2, sniper: 1, shield: 2, flamer: 1, dog: 1 },
    enemies: 7,
    barrels: 0.7,
    artillery: 1.2,
    helis: 2,
    paras: 3,
    jets: 2,
    drones: 1.2,
    ambient: "smoke",
    distant: 1,
    bosses: ["tank", "gunship"],
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
    speed: 24,
    bosses: [],
    ceiling: true,
  },
};

/** The themes levels are drawn from (the cave is the 404's). */
export const ROTATION = Object.keys(THEMES).filter((k) => k !== "cave");

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
export const lightOf = (t: Theme) => (t.backdrop === "cave" ? "Underground" : t.night >= 0.45 ? "Night" : t.night >= 0.2 ? "Dusk" : t.sun === "red" ? "Sunset" : "Day");
