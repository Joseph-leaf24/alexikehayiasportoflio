/**
 * Level themes, ported from `components/pixel/themes.ts` with the same keys and values: ground, buildings,
 * defenders, boss, sky, weather and how hard the air and artillery attacks come. The data lives in `themes/`;
 * levels draw themes from a deck, and the cave is kept for the 404 page.
 */
import { materials } from "@/engine";
import { SET1 } from "./themes/set1";
import { SET2 } from "./themes/set2";
import { SET3 } from "./themes/set3";
import type { Ambient, SetPiece, Theme } from "./themes/types";

export type { Ambient, Backdrop, BossKind, SetPiece, Theme, Troop } from "./themes/types";

export const THEMES: Record<string, Theme> = { ...SET1, ...SET2, ...SET3 };

/**
 * Weights for the long set pieces in each theme, as a share of its other features' total. They span
 * 50 to 100-odd columns, so even a small share comes up often.
 */
type Big = "massif" | "abyss" | "mineshaft";
const BIG: Partial<Record<string, Partial<Record<Big, number>>>> = {
  snow: { massif: 0.05, abyss: 0.015, mineshaft: 0.03 },
  tundra: { massif: 0.04, abyss: 0.015, mineshaft: 0.04 },
  canyon: { massif: 0.04, abyss: 0.02, mineshaft: 0.04 },
  badlands: { massif: 0.03, abyss: 0.03, mineshaft: 0.04 },
  sakura: { massif: 0.04, mineshaft: 0.015 },
  autumn: { massif: 0.035, abyss: 0.015, mineshaft: 0.03 },
  savanna: { massif: 0.025, abyss: 0.015, mineshaft: 0.03 },
  temple: { massif: 0.02, abyss: 0.02, mineshaft: 0.015 },
  jungle: { abyss: 0.02, mineshaft: 0.02 },
  deepjungle: { abyss: 0.02, massif: 0.015, mineshaft: 0.025 },
  desert: { abyss: 0.025, mineshaft: 0.035 },
  inferno: { abyss: 0.04, mineshaft: 0.03 },
  hive: { abyss: 0.04 },
  fortress: { abyss: 0.015, mineshaft: 0.02 },
  farmland: { abyss: 0.015, mineshaft: 0.02 },
  village: { massif: 0.015, mineshaft: 0.025 },
  swamp: { mineshaft: 0.02 },
  monsoon: { mineshaft: 0.015 },
  nightraid: { mineshaft: 0.02 },
};
for (const [key, add] of Object.entries(BIG)) {
  const t = THEMES[key];
  if (!t || !add) continue;
  const total = (Object.values(t.features) as number[]).reduce((s, v) => s + v, 0);
  for (const [f, share] of Object.entries(add) as [Big, number][])
    t.features[f] = Math.round(total * share * 100) / 100;
}

/** Themes levels are drawn from; the cave is kept for the 404 page. */
export const ROTATION = Object.keys(THEMES).filter((k) => k !== "cave");

/** Deals themes from a shuffled deck: each comes up once per pass, and never twice in a row. */
export function themeDeck(rand: () => number) {
  let deck: string[] = [];
  return (current?: string): Theme => {
    if (!deck.length) {
      deck = [...ROTATION];
      for (let i = deck.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        [deck[i], deck[j]] = [deck[j], deck[i]];
      }
      if (deck[deck.length - 1] === current) [deck[0], deck[deck.length - 1]] = [deck[deck.length - 1], deck[0]];
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
  const ch = (sh: number) => Math.round(((pa >> sh) & 255) + (((pb >> sh) & 255) - ((pa >> sh) & 255)) * t);
  return "#" + [16, 8, 0].map((sh) => ch(sh).toString(16).padStart(2, "0")).join("");
};

/** A randomised copy of a theme with its own time of day, weather, relief and feature mix. */
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
      v.sky = [mixHex(t.sky[0], "#05060f", 0.8), mixHex(t.sky[1], "#14102a", 0.75), mixHex(t.sky[2], "#2a1a3a", 0.7)];
      v.tint = "";
      notes.push("at night");
    } else if (time < 0.3) {
      v.sun = "red";
      v.night = Math.max(t.night, 0.12);
      v.sky = [mixHex(t.sky[0], "#2a2d6e", 0.5), mixHex(t.sky[1], "#c85a5a", 0.55), mixHex(t.sky[2], "#ffa65a", 0.6)];
      v.tint = "rgba(255,110,40,0.1)";
      notes.push("at dusk");
    } else if (time < 0.4) {
      v.sky = [mixHex(t.sky[0], "#3a3a7a", 0.4), mixHex(t.sky[1], "#e898a8", 0.5), mixHex(t.sky[2], "#ffe0c0", 0.5)];
      v.tint = "rgba(255,160,180,0.07)";
      notes.push("at dawn");
    }
  }
  const weather = rand();
  const cold = t.top === materials.SNOW || t.backdrop === "snow" || t.backdrop === "tundra";
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
  for (const k of Object.keys(v.features) as SetPiece[]) v.features[k] = (v.features[k] ?? 0) * (0.5 + rand());
  v.name = notes.length ? `${t.name} ${notes[0]}` : t.name;
  return v;
}

/** HUD label for each kind of weather. */
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

/** HUD label for a theme's light. */
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
