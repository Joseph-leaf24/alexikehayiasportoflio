/**
 * Weather, and the sky's look per biome, as the old game and scene made them (`Game.ambient`, `updateSky`; the
 * scene's `skyOf`, overlays and `drawBolts`):
 * - a biome's theme sets its light and sky colours, fixed or from the day–night cycle (`skyLook`, `sky.show`);
 * - its weather particles fall at the old rates (`theme.ambient`), none while the view is buried under the crust;
 * - themes with a day cycle change their sky now and then (clear, cloud, rain, storm, fog): cloud cover greys the
 *   sky, rain and storms fall, and fog and wet ease in as the grade (`sky.fog`, `sky.wet`);
 * - storms throw lightning seconds apart: a flash, a shake and a forked bolt down from the top of the view.
 *
 * System: level.weather (effects). Pass: weather (50), the bolts.
 */
import { crustOf } from "@/components/pixel/level";
import { hash2, PARTICLE_ADDITIVE, PARTICLE_EMISSIVE, SPRITE_EMISSIVE, TILE } from "@/engine";
import type { BattleGame } from "../game/BattleGame";
import type { EffectDef } from "../game/defs";
import { artLookup, emitEffect, LAYER, type SkyLook } from "../render";
import { shape } from "./drawn";
import type { LevelSource } from "./generator";
import type { Ambient, Theme } from "./themes";

type Sky = "clear" | "cloud" | "rain" | "storm" | "fog";

/** The cycle's cloud cover per sky (the old scene's `OVERCAST`): its grey, and how far it greys the sky. */
const OVERCAST: Record<Sky, [number, number]> = {
  clear: [0x6a7280, 0],
  cloud: [0x6a7280, 0.35],
  rain: [0x4a5260, 0.55],
  storm: [0x2a2e38, 0.68],
  fog: [0xb8bec6, 0.6],
};

/** Weather particles per second. */
const RATE: Record<Ambient, number> = {
  none: 0,
  fireflies: 3,
  smoke: 4,
  embers: 14,
  ash: 28,
  snow: 34,
  rain: 60,
  storm: 90,
  sand: 34,
  spores: 10,
};

/** The weather's particles: launched with the velocity `fall` gives them (no speed of their own). */
const STILL = [0, 0] as const;
const GLOW = PARTICLE_EMISSIVE | PARTICLE_ADDITIVE;
const EFFECTS: Readonly<Record<string, EffectDef>> = {
  "weather.firefly": { count: 1, speed: STILL, life: [2, 4], colors: [0xd8ff7aff], flags: GLOW, fade: true },
  "weather.ash": { count: 1, speed: STILL, life: [8, 8], colors: [0xff8a3aff, 0x8a8088ff, 0x8a8088ff], pop: true },
  "weather.ember": {
    count: 1,
    speed: STILL,
    life: [4, 6],
    colors: [0xffe27aff, 0xb8321aff],
    ramp: true,
    flags: GLOW,
  },
  "weather.snow": {
    count: 1,
    speed: STILL,
    life: [12, 12],
    size: [1, 2],
    colors: [0xffffffff, 0xd8e4f4ff, 0xd8e4f4ff],
    pop: true,
  },
  "weather.rain": { count: 1, speed: STILL, life: [1.4, 1.4], colors: [0x8ab0d099], pop: true },
  "weather.sand": { count: 1, speed: STILL, life: [4, 4], colors: [0xe8c890ff, 0xc8a060ff] },
  "weather.spore": { count: 1, speed: STILL, life: [8, 8], colors: [0x8aff6aff, 0xd88affff], flags: GLOW },
  "weather.smoke": {
    count: 1,
    speed: STILL,
    life: [3, 3],
    size: [2, 3],
    colors: [0x2b283080, 0x5b586640],
    ramp: true,
    fade: true,
  },
};

/** A lightning bolt forking down from the top of the view (the old `drawBolts`), a few to pick from. */
const BOLT_POSES = 4;
const BOLT_W = 64;
const BOLT_H = 150;
const BOLT = shape("weather.bolt", {
  w: BOLT_W,
  h: BOLT_H,
  ox: BOLT_W / 2,
  oy: 0,
  poses: BOLT_POSES,
  paint(c, pose) {
    c.fillStyle = "#f4f0ff";
    let x = 0;
    let y = 0;
    for (let n = 0; y < BOLT_H; n += 2) {
      const nx = Math.max(2 - BOLT_W / 2, Math.min(BOLT_W / 2 - 2, x + Math.round((hash2(pose, n) - 0.5) * 8)));
      const ny = y + 4 + Math.floor(hash2(pose, n + 1) * 6);
      const m = Math.max(Math.abs(nx - x), ny - y);
      for (let k = 0; k <= m; k++)
        c.fillRect(Math.round(x + ((nx - x) * k) / m), Math.round(y + ((ny - y) * k) / m), 1, 1);
      x = nx;
      y = ny;
    }
  },
});

const hex = (s: string) => parseInt(s.slice(1), 16);

/** A theme's light and sky colours; its `tint` is the old overlay, "rgba(r,g,b,a)" or "". */
export function skyLook(t: Theme): SkyLook {
  const m = /^rgba\((\d+),(\d+),(\d+),([\d.]+)\)$/.exec(t.tint);
  return {
    cycle: !!t.cycle,
    night: t.night,
    colors: t.sky.map(hex),
    stars: t.backdrop !== "cave",
    tint: m ? (Number(m[1]) << 16) | (Number(m[2]) << 8) | Number(m[3]) : 0xffffff,
    tintA: m ? Number(m[4]) : 0,
  };
}

/** What falls: with a day cycle, rain and storms follow the sky instead of the theme. */
function ambientOf(t: Theme, sky: Sky): Ambient {
  if (!t.cycle) return t.ambient;
  if (sky === "rain" || sky === "storm") return sky;
  return t.ambient === "rain" || t.ambient === "storm" ? "none" : t.ambient;
}

/** One weather particle, placed and launched as the old ones were. */
function fall(game: BattleGame, amb: Ambient): void {
  const r = game.rng.fx;
  const cam = game.camera;
  const L = cam.left;
  const T = cam.top;
  const W = cam.right - L;
  const H = cam.bottom - T;
  const x = L + r.next() * W;
  const at = (id: string, px: number, py: number, vx: number, vy: number) => emitEffect(game, id, px, py, { vx, vy });
  switch (amb) {
    case "fireflies":
      return at("weather.firefly", x, T + H * (0.3 + r.next() * 0.4), (r.next() - 0.5) * 8, (r.next() - 0.5) * 6);
    case "ash":
      return at("weather.ash", x, T - 4, -10 - r.next() * 10, 14 + r.next() * 12);
    case "embers":
      return at("weather.ember", x, T + H, (r.next() - 0.5) * 10, -20 - r.next() * 20);
    case "snow":
      return at("weather.snow", x + 40, T - 4, -12 - r.next() * 14, 18 + r.next() * 16);
    case "rain":
    case "storm": {
      // A drop is a 1×3 streak: three particles falling together
      const vy = 320 + r.next() * 60;
      for (let k = 0; k < 3; k++) at("weather.rain", x + 60, T - 4 + k, amb === "storm" ? -110 : -60, vy);
      return;
    }
    case "sand":
      return at("weather.sand", L + W + 4, T + r.next() * H, -160 - r.next() * 80, (r.next() - 0.5) * 20);
    case "spores":
      return at("weather.spore", x, T + H, (r.next() - 0.5) * 10, -10 - r.next() * 14);
    default:
      at("weather.smoke", x, T + H * 0.7, 4, -6);
  }
}

interface Bolt {
  x: number;
  t: number;
  pose: number;
}

export function installWeather(game: BattleGame, level: LevelSource): void {
  for (const [id, def] of Object.entries(EFFECTS)) game.content.effects.register(id, def);
  let sky: Sky = "clear";
  let skyT = 0;
  let acc = 0;
  let boltT = 0;
  const bolts: Bolt[] = [];
  game.world.addSystem({
    name: "level.weather",
    phase: "effects",
    run(_w, dt) {
      const theme = level.biome;
      const r = game.rng.fx;
      const s = game.sky;
      if (theme.cycle && (skyT -= dt) <= 0) {
        const u = r.next();
        const next: Sky = u < 0.34 ? "clear" : u < 0.58 ? "cloud" : u < 0.78 ? "rain" : u < 0.9 ? "storm" : "fog";
        sky = next === sky ? "clear" : next;
        skyT = 35 + r.next() * 40;
        [s.overcast, s.overcastK] = OVERCAST[sky];
      }
      // Fog and wet ease towards the sky's, as the old 2% a frame did
      const cyc = theme.cycle ? sky : "clear";
      const ease = Math.min(1, dt * 1.2);
      s.fog += ((cyc === "fog" ? 1 : 0) - s.fog) * ease;
      s.wet += ((cyc === "rain" ? 0.6 : cyc === "storm" ? 1 : cyc === "cloud" ? 0.25 : 0) - s.wet) * ease;
      const amb = ambientOf(theme, sky);
      const cam = game.camera;
      if (amb === "storm" && (boltT -= dt) <= 0) {
        // Seconds apart, so it never strobes
        boltT = 5 + r.next() * 9;
        bolts.push({ x: cam.left + (cam.right - cam.left) * (0.1 + r.next() * 0.8), t: 0, pose: r.int(BOLT_POSES) });
        game.shake(1.5);
        game.screen.flash = Math.max(game.screen.flash, 0.24);
      }
      for (const b of bolts) b.t += dt;
      while (bolts.length > 0 && bolts[0].t >= 0.35) bolts.shift();
      // Deep in a cavern there's no sky
      const tc = Math.floor((cam.left + cam.right) / 2 / TILE) + game.origin.cols;
      const buried = cam.top > crustOf(level.old, tc) * TILE + 24;
      acc = buried ? 0 : acc + RATE[amb] * dt;
      for (; acc >= 1; acc--) fall(game, amb);
    },
  });
  const art = artLookup(game);
  game.frames.add({
    name: "weather",
    order: 50,
    draw(g, f, v) {
      const k = art(BOLT.id);
      for (const b of bolts) {
        // The bolt flickers out once before it goes
        if (k <= 0 || b.t > 0.22 || (b.t > 0.08 && b.t < 0.12)) continue;
        const fr = g.art.frame(k, b.pose, 1);
        if (fr >= 0) f.sprites.push(fr, b.x, v.top, LAYER.SKY, SPRITE_EMISSIVE, 0xffffffff, 1, 0);
      }
    },
  });
}
