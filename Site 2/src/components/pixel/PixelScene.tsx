import { useEffect, useRef } from "react";
import { sceneState, type LayoutKey } from "@/lib/store";
import { prefersReducedMotion } from "@/lib/smoothScroll";
import { getQuality } from "@/lib/device";
import { PLAYER_COLORS } from "@/data/squad";
import { resolveBlend } from "./blend";
import { AIR, AMETHYST, BASE_ROW, cell, CRYSTAL, DIAMOND, kindAt, LAVA, LEVEL_H, MAGMA, MASK, MYCELIUM, RING, ROWS, SOLID, TILE, WATER, WINDOW } from "./level";
import { lightOf, THEMES, WEATHER, type Backdrop, type Theme } from "./themes";
import { rng } from "./rng";
import { asset } from "./assets";
import type { ArmorType, HeliType } from "./art";
import {
  ASHFALL,
  BIGBOMB,
  BIGSMOKE,
  BLAZE,
  BUBBLE,
  BOLT,
  BOSS_NAMES,
  BULLET,
  CANNON,
  CASING,
  CINDER,
  DEBRIS,
  DUST,
  ALLY_AIR,
  ARMOR_KINDS,
  ELASER,
  EMBER,
  FLAK,
  NAPALM,
  ENEMY_BULLET,
  FIREBALL,
  FINALE_TEXT,
  FIREFLY,
  FLAME,
  CAR_LEN,
  FINALES,
  FLAMESHOT,
  Game,
  lightState,
  GLINT,
  PULSE,
  WAVES,
  GRENADE,
  LOB,
  METEOR,
  MIST,
  MORTAR,
  PELLET,
  PLASMA,
  PLAYER_HP,
  RAIN,
  ROCKET,
  SAW,
  SANDGRAIN,
  SHELL,
  SMOKE,
  SNIPE,
  SNOWFLAKE,
  SPARK,
  SPORE,
  startTheme,
  type Enemy,
  type View,
} from "./game";
import {
  BAYER,
  DEBRIS_COLOR,
  disc,
  drawAirship,
  canvasOptions,
  drawAllyGunship,
  drawCar,
  drawFreightCar,
  drawMetroTrain,
  drawApc,
  drawArmor,
  drawHeliType,
  drawPlane,
  drawBattleship,
  drawBoat,
  drawCannon,
  drawEmplacement,
  drawSubmarine,
  drawTrain,
  drawLightTank,
  drawSpider,
  drawDrone,
  drawGunship,
  drawHeli,
  drawIcbm,
  drawJeep,
  drawMech,
  drawTank,
  drawTanker,
  drawTruck,
  drawWorm,
  FRAMES,
  INK,
  makeAssets,
  makeAtlas,
  makeLoadouts,
  makeBackdrop,
  makeCanvas,
  makeCloud,
  makeProps,
  makeTroops,
  mix,
  OUTLINED,
  pixelText,
  rgb,
  ring,
  VARIANTS,
  type BackdropArt,
} from "./art";

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
/** The battle plays this much faster than real time at "1×" (and the rest of the speeds in proportion). */
const PACE = 1.25;

/**
 * Each explosion palette's fireball, hot core to cooling edge (game.ts PAL_*): its outline, its
 * highlights, the smoke it cools to, its flash and the light it throws. Plain fire, a warhead's
 * white and gold, a chemical drum's green, gas burning blue, plasma, napalm, magma.
 */
const FIRE_PALS = [
  { core: "#fffbe0", hot: "#ffe14a", mid: "#ffa21f", cool: "#e8501a", edge: "#b8263a", rim: ["#6a1a0c", "#3a1410"], hi: ["#ffffff", "#ffe14a", "#ff8a2a"], smoke: ["#3e3034", "#5a5258"], flash: "#ffe7a0", glow: "#ff8a3a" },
  { core: "#ffffff", hot: "#fff6c8", mid: "#ffd23f", cool: "#ff9a3a", edge: "#d8504a", rim: ["#8a3a2a", "#4a2030"], hi: ["#ffffff", "#ffffff", "#ffe27a"], smoke: ["#5a4a4a", "#7a6a6a"], flash: "#fffbe8", glow: "#ffd27a" },
  { core: "#f4ffd0", hot: "#c8ff6a", mid: "#6ae84a", cool: "#2aa84a", edge: "#1a6a4a", rim: ["#1a4a2a", "#10301c"], hi: ["#ffffff", "#e8ff9a", "#8aff6a"], smoke: ["#2e3a2a", "#4a5a44"], flash: "#e0ffa0", glow: "#7aff4a" },
  { core: "#ffffff", hot: "#c8f0ff", mid: "#6ab8ff", cool: "#ff9a3a", edge: "#d84a1a", rim: ["#1a2a6a", "#101838"], hi: ["#ffffff", "#e8f8ff", "#9ad8ff"], smoke: ["#30343e", "#4a5060"], flash: "#d8f0ff", glow: "#5ab0ff" },
  { core: "#ffffff", hot: "#bafcff", mid: "#4ad8ff", cool: "#8a6aff", edge: "#c84ae8", rim: ["#2a1a6a", "#1a1040"], hi: ["#ffffff", "#e0ffff", "#9af0ff"], smoke: ["#2e2a40", "#4a4460"], flash: "#d0fcff", glow: "#6ae8ff" },
  { core: "#fff0c0", hot: "#ffc23a", mid: "#ff6a1a", cool: "#c8281a", edge: "#6a0e10", rim: ["#4a0c08", "#200808"], hi: ["#ffffff", "#ffd23f", "#ff7a2a"], smoke: ["#1e1a1c", "#3a3436"], flash: "#ffd890", glow: "#ff5a1a" },
  { core: "#ffe8a0", hot: "#ffb43a", mid: "#ff6a1a", cool: "#d8301a", edge: "#8a1414", rim: ["#5a1408", "#301008"], hi: ["#fff6c8", "#ffd23f", "#ff8a3a"], smoke: ["#3a2a2a", "#5a4444"], flash: "#ffc890", glow: "#ff6a2a" },
];
const firePal = (c: string) => FIRE_PALS[+c] ?? FIRE_PALS[0];

/** Frame-rate independent smoothing factor: `x += (target - x) * damp(dt, rate)` */
const damp = (dt: number, rate: number) => 1 - Math.exp(-dt * rate);
const mod = (a: number, n: number) => ((a % n) + n) % n;
/** The screen is about this many scene pixels tall, whatever the device: a wide view of the battle. */
const TARGET_ROWS = 360;
/** The squad's feet sit this far down the screen (higher on portrait screens, above the stacked hero panels). */
const HORIZON = 0.68;
const HORIZON_PORTRAIT = 0.5;
/** After this long without input (ms) the scene drops to about 30 frames a second. */
const IDLE_MS = 8000;
const INPUT_EVENTS = ["pointermove", "pointerdown", "wheel", "keydown", "touchstart", "scroll"] as const;
/** Simulated seconds before the first frame, so the scene opens mid-fight. */
const WARM_UP = 5;
/** Sections in page order: deeper ones turn the intensity up (the spectator space at the bottom most of all). */
const SECTION_ORDER: LayoutKey[] = ["village", "shatter", "clusters", "network", "bars", "villageTop", "watch"];
/** The far-off battle: things go off beyond the mountains (plane 0) and between the far and mid hills (plane 1). */
const FAR_BOOM = 0;
const FAR_FLAK = 1;
const FAR_TRACER = 2;
const FAR_PLANE = 3;
const FAR_BIRDS = 4;
const FAR_SMOKE = 5;
const FAR_BEAM = 6;
/** How far each plane slides as the camera moves (the far mountains move at 0.08, the mid hills at 0.2). */
const FAR_F = [0.08, 0.15];
interface FarFx {
  k: number;
  depth: 0 | 1;
  /** x in the plane's own coordinates; y up from its horizon line */
  x: number;
  y: number;
  t: number;
  life: number;
  s: number;
  vx: number;
  seed: number;
}
const JUNGLES: Backdrop[] = ["jungle", "deepjungle", "temple", "monsoon", "swamp"];
const clock = (sec: number) => `${Math.floor(sec / 60)}:${String(Math.floor(sec % 60)).padStart(2, "0")}`;
/** The sea's colour by depth below its surface (px): sunlit teal at the top, deepening to navy. */
const SEA_COLOUR: [number, string][] = [
  [0, "#6cc4dc"],
  [6, "#4aaed2"],
  [20, "#3290c0"],
  [44, "#2474aa"],
  [80, "#1a5890"],
  [130, "#143f72"],
  [200, "#0e2c56"],
  [320, "#0a1e3e"],
  [LEVEL_H, "#08162e"],
];
/** How much of what's in the water still shows through, by depth: plenty near the top, little deep down. */
const SEA_ALPHA: [number, number][] = [
  [0, 0.3],
  [8, 0.42],
  [24, 0.58],
  [56, 0.74],
  [110, 0.86],
  [200, 0.93],
  [LEVEL_H, 0.96],
];
/** Piecewise-linear lookup in a list of [depth, value] stops. */
function alongDepth<T>(stops: [number, T][], d: number, lerp: (a: T, b: T, u: number) => T) {
  for (let i = 1; i < stops.length; i++) if (d <= stops[i][0]) return lerp(stops[i - 1][1], stops[i][1], (d - stops[i - 1][0]) / (stops[i][0] - stops[i - 1][0]));
  return stops[stops.length - 1][1];
}
/**
 * The sea from its surface down, one block wide: an ordered-dither gradient through 28 shades
 * (no hard bands). One opaque copy goes behind everything; one see-through copy is laid over
 * whatever is in the water, so swimmers and submarines look submerged, fading the deeper they go.
 */
function makeSea() {
  const K = 28;
  const shades = Array.from({ length: K }, (_, i) =>
    alongDepth(SEA_COLOUR.map(([d, c]) => [d, rgb(c)] as [number, number[]]), LEVEL_H * (i / (K - 1)) ** 2, (a, b, u) => a.map((v, n) => v + (b[n] - v) * u))
  );
  const [back, bctx] = makeCanvas(TILE, LEVEL_H);
  const [front, fctx] = makeCanvas(TILE, LEVEL_H);
  const bi = bctx.createImageData(TILE, LEVEL_H);
  const fi = fctx.createImageData(TILE, LEVEL_H);
  for (let y = 0; y < LEVEL_H; y++) {
    const s = (K - 1) * Math.sqrt(y / LEVEL_H);
    const a = Math.round(alongDepth(SEA_ALPHA, y, (p, q, u) => p + (q - p) * u) * 255);
    for (let x = 0; x < TILE; x++) {
      const i = Math.min(K - 1, s - Math.floor(s) > (BAYER[(y & 3) * 4 + (x & 3)] + 0.5) / 16 ? Math.ceil(s) : Math.floor(s));
      const o = (y * TILE + x) * 4;
      for (let n = 0; n < 3; n++) bi.data[o + n] = fi.data[o + n] = Math.round(shades[i][n]);
      bi.data[o + 3] = 255;
      fi.data[o + 3] = a;
    }
  }
  bctx.putImageData(bi, 0, 0);
  fctx.putImageData(fi, 0, 0);
  return { back, front };
}
/** The ground darkens with depth below its own surface (on a mountain as in a valley), down to this. */
const DEPTH_SHADE = Array.from({ length: 16 }, (_, k) => `rgba(6,5,14,${Math.min(0.62, k * 0.045).toFixed(3)})`);

/** Deterministic 0..1 hash of two integers (block variants, flicker). */
const hash2 = (a: number, b: number) => {
  let h = Math.imul(a | 0, 374761393) ^ Math.imul(b | 0, 668265263);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
};

/**
 * The fixed pixel-art backdrop: eight players (P1–P4 the domains, P5–P8 specialists)
 * fighting through endless random levels (game.ts, level.ts, themes.ts), drawn on a
 * low-resolution canvas scaled up with hard pixel edges, a battle raging in the distance
 * behind the hills. Each level ends with ICBMs on the enemy stronghold: the whole screen
 * fades to white, "Level complete" comes up, and the next level fades in. Deeper sections of
 * the page make it wilder (the spectator space at the bottom most of all, at full frame rate);
 * project pages start a new level, the 404 drops the squad in a cave. A click calls in a
 * blast, holding on the hero calls in an airstrike, hovering a domain card or a Work tab
 * points out its player. With reduced motion the fight is shown frozen mid-action, without
 * the white-out. Without a 2D canvas it adds `no-canvas` to <html> and draws nothing.
 */
export default function PixelScene() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d", canvasOptions);
    if (!canvas || !ctx) {
      sceneState.canvas = false;
      sceneState.ready = true;
      document.documentElement.classList.add("no-canvas");
      return;
    }

    const q = getQuality();
    const reduced = prefersReducedMotion();
    const rand = Math.random;
    const params = new URLSearchParams(window.location.search);

    /* ------------------------------------------------------------ */
    /* Art                                                           */
    /* ------------------------------------------------------------ */
    const atlas = makeAtlas();
    const loadouts = makeLoadouts(PLAYER_COLORS);
    const props = makeProps();
    const assetArt = makeAssets();
    const troopArt = new Map<Backdrop, ReturnType<typeof makeTroops>>();
    const troops = (k: Backdrop) => troopArt.get(k) ?? troopArt.set(k, makeTroops(k)).get(k)!;
    const backdrops = new Map<Backdrop, BackdropArt>();
    const backdrop = (k: Backdrop) => backdrops.get(k) ?? backdrops.set(k, makeBackdrop(k)).get(k)!;
    const clouds = Array.from({ length: q.clouds }, (_, i) => ({ img: makeCloud(i + 1), x: hash2(i, 1) * 1600, alt: 40 + hash2(i, 2) * 110, speed: 3 + hash2(i, 3) * 5 }));

    /* ------------------------------------------------------------ */
    /* Game + its blocks on a ring-buffer canvas                     */
    /* ------------------------------------------------------------ */
    let game: Game | null = null;
    const [terrain, tctx] = makeCanvas(RING * TILE, ROWS * TILE);
    const sea = makeSea();
    const drawCell = (c: number, r: number) => {
      const L = game!.level;
      if (r < 0 || r >= ROWS || c < L.start || c >= L.next) return;
      const i = cell(c, r);
      const x = (c & MASK) * TILE;
      const y = r * TILE;
      tctx.clearRect(x, y, TILE, TILE);
      const k = L.fg[i];
      if (k === WATER) {
        // The sea behind whatever is in it, shaded by how far below its own surface this block is
        let d = 0;
        while (d < ROWS && kindAt(L, c, r - d - 1) === WATER) d++;
        tctx.drawImage(sea.back, 0, Math.min(d * TILE, LEVEL_H - TILE), TILE, TILE, x, y, TILE, TILE);
        return;
      }
      const v = (hash2(c, r) * VARIANTS) | 0;
      if (k !== AIR) tctx.drawImage(atlas, k * TILE, v * TILE, TILE, TILE, x, y, TILE, TILE);
      else {
        const b = L.bg[i];
        if (!b) return;
        // Back wall: the same block, in shadow
        tctx.drawImage(atlas, b * TILE, v * TILE, TILE, TILE, x, y, TILE, TILE);
        tctx.fillStyle = "rgba(10,8,22,0.62)";
        tctx.fillRect(x, y, TILE, TILE);
      }
      if (L.burn[i]) {
        tctx.fillStyle = `rgba(20,10,6,${0.22 * L.burn[i]})`;
        tctx.fillRect(x, y, TILE, TILE);
      }
      // Deeper below the land's surface (not the floor of a cavern) is darker; a cavern's own back
      // wall half as much, so it stays readable down there
      const depth = r - L.crust[c & MASK] - 5;
      if (depth > 0) {
        tctx.fillStyle = DEPTH_SHADE[Math.min(15, k !== AIR ? depth : depth >> 1)];
        tctx.fillRect(x, y, TILE, TILE);
      }
      if (k !== AIR && OUTLINED[k]) {
        tctx.fillStyle = INK;
        if (kindAt(L, c, r - 1) === AIR) tctx.fillRect(x, y, TILE, 1);
        if (r < ROWS - 1 && kindAt(L, c, r + 1) === AIR) tctx.fillRect(x, y + TILE - 1, TILE, 1);
        if (kindAt(L, c - 1, r) === AIR) tctx.fillRect(x, y, 1, TILE);
        if (kindAt(L, c + 1, r) === AIR) tctx.fillRect(x + TILE - 1, y, 1, TILE);
      }
    };
    const drawColumn = (c: number) => {
      for (let r = 0; r < ROWS; r++) drawCell(c, r);
    };
    const redrawAll = (g: Game) => {
      tctx.clearRect(0, 0, terrain.width, terrain.height);
      for (let c = g.level.start; c < g.level.next; c++) drawColumn(c);
    };

    /* ------------------------------------------------------------ */
    /* Screen buffers                                                */
    /* ------------------------------------------------------------ */
    let W = 1;
    let H = 1;
    /** The zoom the view was last sized for */
    let zoomed = 0;
    let scaleK = 1;
    let horizon = HORIZON;
    let scale = 1;
    let dpr = 1;
    const [view, vctx] = makeCanvas(1, 1);
    const [sky, sctx] = makeCanvas(1, 1);
    let stars: { x: number; y: number; b: number }[] = [];
    let skyKey = "";
    let snap = true;
    /** Portrait screens keep the fight up above the stacked hero panels, except in the open spectator space. */
    const horizonFor = () => (W / H < 0.8 && !sceneState.watching ? HORIZON_PORTRAIT : HORIZON);

    const resize = () => {
      dpr = window.devicePixelRatio || 1;
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      // Zoomed out (sceneState.zoom steps): smaller pixels, so more of the battlefield fits. Never
      // so far that the view outgrows the level's stored columns or its height.
      const base = Math.max(2, Math.round((vh * dpr) / TARGET_ROWS));
      const least = Math.max(1, Math.ceil((vh * dpr) / 1000), Math.ceil((vw * dpr) / (RING * TILE - 520)));
      const steps = Math.max(0, Math.min(sceneState.zoom, base - least));
      // (and by a fraction more or less, if asked: whole pixels where that comes out near enough)
      const want = Math.max(1, (base - steps) * sceneState.scaleK);
      scale = Math.abs(want - Math.round(want)) < 0.12 ? Math.round(want) : want;
      scaleK = sceneState.scaleK;
      // (as far as this screen can go: so one press of + always zooms back in)
      sceneState.zoom = zoomed = steps;
      window.dispatchEvent(new CustomEvent("scenezoom", { detail: { pct: Math.round((scale / base) * 100), canIn: steps > 0, canOut: base - steps > least } }));
      W = Math.max(1, Math.ceil((vw * dpr) / scale));
      H = Math.max(1, Math.ceil((vh * dpr) / scale));
      horizon = horizonFor();
      canvas.width = view.width = sky.width = W;
      canvas.height = view.height = sky.height = H;
      canvas.style.width = `${(W * scale) / dpr}px`;
      canvas.style.height = `${(H * scale) / dpr}px`;
      ctx.imageSmoothingEnabled = vctx.imageSmoothingEnabled = false;
      stars = Array.from({ length: Math.round((W * H) / 1100) }, (_, i) => ({ x: Math.floor(hash2(i, 7) * W), y: Math.floor(hash2(i, 8) * H * 0.6), b: hash2(i, 9) }));
      skyKey = "";
      snap = true;
      held = false;
    };

    /* ------------------------------------------------------------ */
    /* Sky (follows the level's theme, crossfading when it changes)  */
    /* ------------------------------------------------------------ */
    let theme: Theme = THEMES.jungle;
    let prevTheme: Theme = theme;
    let fade = 1;
    const lerpTheme = (f: (s: Theme) => number) => f(prevTheme) + (f(theme) - f(prevTheme)) * fade;
    /** How dark it is now: the theme's own darkness, or where the day turns, the time of day */
    const nightNow = () => lerpTheme((s) => (s.cycle && game ? game.night : s.night));
    /** Where the day turns: the sky's colours at this time of day, in this weather */
    const DAY_SKY = ["#3a78d0", "#86b6e6", "#d8eaf6"];
    const DUSK_SKY = ["#2a2a5a", "#c8607a", "#ffb070"];
    const NIGHT_SKY = ["#04050e", "#0c1030", "#1e2848"];
    const OVERCAST: Record<string, [string, number]> = { clear: ["#6a7280", 0], cloud: ["#6a7280", 0.35], rain: ["#4a5260", 0.55], storm: ["#2a2e38", 0.68], fog: ["#b8bec6", 0.6] };
    const skyOf = (s: Theme) => {
      if (!s.cycle || !game) return s.sky;
      const u = Math.round(clamp01(game.night / 0.7) * 25) / 25;
      const [grey, k] = OVERCAST[game.sky];
      return [0, 1, 2].map((i) => mix(u < 0.5 ? mix(DAY_SKY[i], DUSK_SKY[i], u * 2) : mix(DUSK_SKY[i], NIGHT_SKY[i], (u - 0.5) * 2), mix(grey, NIGHT_SKY[i], u), k));
    };
    /** Fog and rain over the streets, easing in and out with the weather */
    let fogA = 0;
    let wetA = 0;

    const regenSky = () => {
      const from = skyOf(prevTheme);
      const to = skyOf(theme);
      const k = `${W}x${H}:${prevTheme.key}>${theme.key}:${fade.toFixed(2)}:${from.join()}:${to.join()}`;
      if (k === skyKey) return;
      skyKey = k;
      const stops = [0, 1, 2].map((i) => rgb(mix(from[i], to[i], fade)));
      const N = 10;
      const bands = Array.from({ length: N }, (_, j) => {
        const t = j / (N - 1);
        const [a, b, u] = t < 0.6 ? [stops[0], stops[1], t / 0.6] : [stops[1], stops[2], (t - 0.6) / 0.4];
        return a.map((v, n) => Math.round(v + (b[n] - v) * u));
      });
      const img = sctx.createImageData(W, H);
      const d = img.data;
      const hz = Math.max(1, Math.round(H * horizon));
      for (let y = 0; y < H; y++) {
        const p = Math.min(1, y / hz) * (N - 1);
        const band = Math.floor(p);
        const fr = clamp01((p - band - 0.62) / 0.38);
        for (let x = 0; x < W; x++) {
          const c = bands[Math.min(N - 1, fr > (BAYER[(y & 3) * 4 + (x & 3)] + 0.5) / 16 ? band + 1 : band)];
          const o = (y * W + x) * 4;
          d[o] = c[0];
          d[o + 1] = c[1];
          d[o + 2] = c[2];
          d[o + 3] = 255;
        }
      }
      sctx.putImageData(img, 0, 0);
    };

    /* ------------------------------------------------------------ */
    /* Camera                                                        */
    /* ------------------------------------------------------------ */
    let camX = 0;
    let camY = 0;
    let shake = 0;
    const viewRect = (): View => ({ x: camX, y: camY, w: W, h: H });
    const cameraTarget = (g: Game) => {
      const alive = g.players.filter((p) => !p.dead && !p.chute && !p.aboard);
      // (with the whole squad thrown through the air by a warhead, the camera holds where it is)
      // (and with the whole squad up in the air, coming down by parachute, it frames the ground they're dropping onto)
      // (and with them all aboard the gunship, the gunship and the land it's flying over)
      const below = g.level.ground[((Math.floor(g.advance / TILE) % RING) + RING) % RING] * TILE;
      // (all aboard: the gunship and the land under it, or the metro train in its tunnel)
      const ride = g.carrier;
      let feet = g.phase === "nuke" ? camY + H * horizon : ride ? ride.y + (ride.air ? 58 : 0) : below;
      // (a squad split up and down a shaft frames its bigger group, or the one further on if they're even,
      // not the solid rock halfway between them; all together, that's just the squad's average height)
      let most = 0;
      let lead = -Infinity;
      for (const p of alive) {
        let n = 0;
        let sum = 0;
        for (const q of alive)
          if (Math.abs(q.y - p.y) < H * 0.3) {
            n++;
            sum += q.y;
          }
        if (n > most || (n === most && p.x > lead)) {
          most = n;
          lead = p.x;
          feet = sum / n;
        }
      }
      // (never left of the oldest stored column: there's nothing to see there)
      // (the squad a third of the way across: room to see what it's fighting, up ahead)
      return { x: Math.max(g.level.start * TILE, g.advance - W * 0.32), y: Math.max(-80, Math.min(LEVEL_H - H, feet - H * horizon)) };
    };
    const pointerWorld = () => {
      const p = sceneState.pointer;
      const sx = ((p.x + 1) / 2) * window.innerWidth;
      const sy = ((1 - p.y) / 2) * window.innerHeight;
      return { x: camX + (sx * dpr) / scale, y: camY + (sy * dpr) / scale };
    };

    let lastOverride = sceneState.override;
    const intensity = () => {
      if (sceneState.override) return 1;
      const { a, b, t } = resolveBlend(window.scrollY, window.innerHeight);
      const k = SECTION_ORDER.indexOf(t < 0.5 ? a : b);
      return 1 + 0.12 * Math.max(0, k);
    };
    const warm = (g: Game, seconds: number) => {
      // Open mid-fight: play a few seconds before the first frame
      for (let s = 0; s < seconds * 30; s++) {
        const tg = cameraTarget(g);
        camX = tg.x;
        camY = tg.y;
        g.step(1 / 30, viewRect());
        if (g.reset) {
          g.reset = false;
          redrawAll(g);
        }
      }
      g.shake = 0;
      g.fresh.length = 0;
      g.dirty.length = 0;
      redrawAll(g);
      snap = true;
    };
    const start = () => {
      lastOverride = sceneState.override;
      const first = sceneState.override === "lost" ? THEMES.cave : startTheme(params.get("level"), rand);
      // (?finale=nuke and the like: a short first level, ending that way, to see it)
      const finale = FINALES.find((f) => f === params.get("finale"));
      const g = new Game(first, {
        seed: 1 + Math.floor(rand() * 100000),
        maxEnemies: q.enemies,
        maxParticles: q.particles,
        colors: PLAYER_COLORS,
        ...(finale ? { finale, level: { length: 70, stronghold: true, boss: null } } : {}),
      });
      g.intensity = intensity();
      game = g;
      theme = prevTheme = first;
      fade = 1;
      warm(g, WARM_UP);
    };

    /* ------------------------------------------------------------ */
    /* Drawing                                                       */
    /* ------------------------------------------------------------ */
    const parallaxBase = (cy: number, f: number) => Math.round(H * horizon - (cy - (BASE_ROW * TILE - H * horizon)) * f);
    const motion = reduced ? 0 : 1;

    /* ------------------------------------------------------------ */
    /* The battle in the distance                                    */
    /* ------------------------------------------------------------ */
    const farFx: FarFx[] = [];
    /** Each plane's horizon line on screen: about where the far mountains' and the mid hills' ridges run. */
    const farBase = (cy: number, depth: number) => (depth ? parallaxBase(cy, 0.15) - 60 : parallaxBase(cy, 0.08) - 96);
    const spawnFar = (dt: number, cx: number) => {
      const d = theme.distant;
      if (d <= 0 || farFx.length > 90) return;
      const at = (depth: 0 | 1) => rand() * (W + 120) - 60 + cx * FAR_F[depth];
      const roll = (perSec: number) => rand() < perSec * dt;
      if (roll(1.1 * d)) {
        const depth = rand() < 0.55 ? 0 : 1;
        farFx.push({ k: FAR_BOOM, depth, x: at(depth), y: -rand() * 26, t: 0, life: 2.2, s: 2 + rand() * (depth ? 5 : 3), vx: 0, seed: rand() });
      }
      if (roll(0.35 * d)) {
        // A cluster of flak bursts
        const x = at(0);
        const y = -50 - rand() * 60;
        for (let n = 3 + Math.floor(rand() * 4); n > 0; n--)
          farFx.push({ k: FAR_FLAK, depth: 0, x: x + (rand() - 0.5) * 40, y: y + (rand() - 0.5) * 24, t: -rand() * 1.2, life: 1.6, s: 1 + rand() * 2, vx: 0, seed: rand() });
      }
      if (roll((0.12 + theme.night * 0.3) * d)) {
        const depth = rand() < 0.5 ? 0 : 1;
        farFx.push({ k: FAR_TRACER, depth, x: at(depth), y: -rand() * 10, t: 0, life: 1.4 + rand() * 1.4, s: rand() < 0.3 ? 1 : 0, vx: 0, seed: rand() });
      }
      if (roll(0.04 * d)) farFx.push({ k: FAR_PLANE, depth: 0, x: cx * FAR_F[0] + W + 20, y: -70 - rand() * 50, t: 0, life: 60, s: rand() < 0.25 ? 1 : 0, vx: -(10 + rand() * 12), seed: rand() });
      if (roll(JUNGLES.includes(theme.backdrop) ? 0.07 : 0.015))
        farFx.push({ k: FAR_BIRDS, depth: 1, x: cx * FAR_F[1] + W + 20, y: -30 - rand() * 50, t: 0, life: 60, s: 4 + Math.floor(rand() * 6), vx: -(16 + rand() * 14), seed: rand() });
      // Smoke columns stand over the horizon (kept topped up); searchlights sweep the sky at night
      let smokes = 0;
      let beams = 0;
      for (const f of farFx) {
        if (f.k === FAR_SMOKE) smokes++;
        else if (f.k === FAR_BEAM) beams++;
      }
      if (smokes < Math.round(3 * d) && roll(0.5)) {
        const depth = rand() < 0.5 ? 0 : 1;
        farFx.push({ k: FAR_SMOKE, depth, x: at(depth), y: 0, t: rand() * 10, life: Infinity, s: 0.6 + rand() * 0.8, vx: 0, seed: rand() });
      }
      if (theme.night > 0.3 && beams < 2 && roll(0.3)) farFx.push({ k: FAR_BEAM, depth: 0, x: at(0), y: 0, t: 0, life: Infinity, s: 0, vx: 0, seed: rand() * 10 });
    };
    const updateFar = (dt: number, cx: number) => {
      for (let i = farFx.length - 1; i >= 0; i--) {
        const f = farFx[i];
        f.t += dt;
        f.x += f.vx * dt;
        const sx = f.x - cx * FAR_F[f.depth];
        if (f.t > f.life || sx < -140 || sx > W + 260) {
          farFx[i] = farFx[farFx.length - 1];
          farFx.pop();
        }
      }
    };
    const drawFar = (c: CanvasRenderingContext2D, depth: 0 | 1, cx: number, cy: number, t: number) => {
      const base = farBase(cy, depth);
      const night = theme.night > 0.3;
      for (const f of farFx) {
        if (f.depth !== depth || f.t < 0) continue;
        const x = Math.round(f.x - cx * FAR_F[depth]);
        const y = Math.round(base + f.y);
        if (x < -100 || x > W + 100) continue;
        switch (f.k) {
          case FAR_BOOM: {
            if (f.t < 0.08) {
              c.fillStyle = "#fff6d0";
              disc(c, x, y, f.s + 1);
              break;
            }
            if (f.t < 0.5) {
              const a = 1 - (f.t - 0.08) / 0.42;
              c.globalAlpha = a;
              c.fillStyle = f.t < 0.25 ? "#ffd23f" : "#ff8a1a";
              disc(c, x, y, Math.round(f.s * (1 + f.t * 2)));
              if (night) {
                c.globalAlpha = a * 0.25;
                disc(c, x, y, Math.round(f.s * 4));
              }
              c.globalAlpha = 1;
            }
            const sa = Math.max(0, 1 - f.t / f.life) * 0.5;
            if (f.t > 0.2 && sa > 0.05) {
              c.globalAlpha = Math.round(sa * 4) / 4;
              c.fillStyle = night ? "#3a3640" : "#7a7480";
              disc(c, x + Math.round(f.t * 2), y - Math.round(f.t * 9), Math.round(f.s * (0.8 + f.t * 0.8)));
              c.globalAlpha = 1;
            }
            break;
          }
          case FAR_FLAK: {
            if (f.t < 0.06) {
              c.fillStyle = "#fff2b0";
              c.fillRect(x - 1, y - 1, 3, 3);
              break;
            }
            c.globalAlpha = Math.round(Math.max(0, 1 - f.t / f.life) * 0.7 * 4) / 4;
            c.fillStyle = night ? "#2a2830" : "#4a4650";
            disc(c, x, y + Math.round(f.t * 2), Math.round(f.s + f.t));
            c.globalAlpha = 1;
            break;
          }
          case FAR_TRACER: {
            // Anti-aircraft fire hosing the sky
            const lean = (f.seed - 0.5) * 1.2 + Math.sin(f.t * 1.5 + f.seed * 6) * 0.15;
            const ux = Math.sin(lean);
            const uy = -Math.cos(lean);
            const head = f.t * 140;
            c.globalAlpha = 0.85 * (f.t > f.life - 0.3 ? (f.life - f.t) / 0.3 : 1);
            c.fillStyle = f.s ? "#9aff6a" : "#ffb84a";
            for (let k = head % 9; k < Math.min(head, 160); k += 9) c.fillRect(Math.round(x + ux * k), Math.round(y + uy * k), 1, 2);
            c.globalAlpha = 1;
            break;
          }
          case FAR_PLANE: {
            c.fillStyle = night ? "#0c0e18" : "#3a3e4c";
            c.fillRect(x, y, 6, 1);
            c.fillRect(x + 2, y - 1, 2, 1);
            c.fillRect(x + 5, y - 2, 1, 2);
            if (f.s) {
              // Hit: trailing smoke and fire
              c.fillStyle = "#5a5660";
              for (let k = 1; k < 8; k++) if ((k + Math.floor(t * 8)) % 2) c.fillRect(x + 6 + k * 2, y - Math.round(k * 0.3), 1, 1);
              if (Math.floor(t * 10) % 2) {
                c.fillStyle = "#ff8a1a";
                c.fillRect(x + 6, y, 1, 1);
              }
            }
            break;
          }
          case FAR_BIRDS: {
            c.fillStyle = night ? "#0c0e18" : "#1e2228";
            for (let b = 0; b < f.s; b++) {
              const bx = x + ((b * 7) % 23) - (b % 2) * 3;
              const by = y + Math.round(Math.sin(f.t * 1.4 + b) * 3) + (b % 3) * 3;
              const up = Math.floor(f.t * 5 + b) % 2;
              c.fillRect(bx - 1, by - up, 1, 1);
              c.fillRect(bx, by, 1, 1);
              c.fillRect(bx + 1, by - up, 1, 1);
            }
            break;
          }
          case FAR_SMOKE: {
            // Something burning beyond the hills: a column of smoke leaning in the wind
            c.fillStyle = night ? "#1c1a22" : "#5a5662";
            for (let k = 0; k < 12; k++) {
              const h = k * 7 + ((f.t * 6) % 7);
              c.globalAlpha = Math.round((1 - k / 12) * 0.45 * 4) / 4;
              disc(c, Math.round(x + h * 0.35 * f.s + Math.sin(f.t * 0.5 + k) * 1.5), Math.round(y - h), Math.round((2 + k * 0.6) * f.s));
            }
            c.globalAlpha = 1;
            if (Math.sin(t * 7 + f.seed * 9) > 0.3) {
              c.fillStyle = "#ff8a1a";
              c.fillRect(x - 1, y - 1, 3, 2);
            }
            break;
          }
          case FAR_BEAM: {
            const a = -Math.PI / 2 + Math.sin(t * 0.35 + f.seed) * 0.55;
            c.globalCompositeOperation = "lighter";
            c.fillStyle = `rgba(200,220,255,${0.08 * theme.night})`;
            c.beginPath();
            c.moveTo(x, y);
            c.lineTo(x + Math.cos(a - 0.05) * 260, y + Math.sin(a - 0.05) * 260);
            c.lineTo(x + Math.cos(a + 0.05) * 260, y + Math.sin(a + 0.05) * 260);
            c.closePath();
            c.fill();
            c.globalCompositeOperation = "source-over";
            break;
          }
        }
      }
    };

    const troopSprite = (e: Enemy) => {
      const art = troops(e.look);
      const face = e.facing > 0 ? 0 : 1;
      switch (e.kind) {
        case "turret":
          return art.turret[face];
        case "mg":
          return art.mg[face];
        case "mortar":
          return art.mortar[face];
        case "heavy":
          return art.heavy[Math.floor(e.anim) % 2][face];
        case "minigunner":
          return art.minigunner[Math.floor(e.anim) % 2][face];
        case "exosuit":
          return art.exosuit[Math.floor(e.anim) % 2][face];
        case "jetpacker":
          return art.jetpacker[FRAMES.jump][face];
        case "sniper":
          return art.sniper[0][face];
        case "crawler":
          return art.crawler[Math.floor(e.anim) % 2][face];
        case "dog":
          return art.dog[Math.floor(e.anim) % 2][face];
        default: {
          const set =
            e.kind === "rocket"
              ? art.rocket
              : e.kind === "flamer"
                ? art.flamer
                : e.kind === "grenadier"
                  ? art.grenadier
                  : e.kind === "shield"
                    ? art.shield
                    : e.kind === "officer"
                      ? art.officer
                      : e.kind === "dynamiter"
                        ? art.dynamiter
                        : e.kind === "rusher"
                          ? art.rusher
                          : e.kind === "engineer"
                            ? art.engineer
                            : e.kind === "laser"
                              ? art.laser
                              : art.mook;
          return set[e.vx ? FRAMES.run[Math.floor(e.anim) % 4] : e.ground ? FRAMES.idle : FRAMES.jump][face];
        }
      }
    };

    /** The surface (world px) of the water at (x, y), or null if (x, y) isn't in water. */
    const surfaceAbove = (g: Game, x: number, y: number) => {
      const c = Math.floor(x / TILE);
      let r = Math.floor(y / TILE);
      if (kindAt(g.level, c, r) !== WATER) return null;
      while (r > 0 && kindAt(g.level, c, r - 1) === WATER) r--;
      return r * TILE;
    };

    const drawWorld = (g: Game, cx: number, cy: number, t: number) => {
      const v = vctx;
      v.globalCompositeOperation = "source-over";
      v.globalAlpha = 1;
      v.clearRect(0, 0, W, H);
      const bd = backdrop(theme.backdrop);

      // Clouds in the open-air themes
      const cloudy = theme.backdrop === "cave" || theme.backdrop === "volcano" || theme.backdrop === "hive" ? 0 : 1 - clamp01(theme.night * 2);
      if (cloudy > 0.05) {
        v.globalAlpha = cloudy;
        const span = Math.max(W + 160, 1000);
        for (const c of clouds) v.drawImage(c.img, Math.round(mod(c.x + t * c.speed * (reduced ? 0 : 1) - cx * 0.05, span) - 80), Math.round(parallaxBase(cy, 0.03) - 60 - c.alt));
        v.globalAlpha = 1;
      }
      // Far, mid and near backdrop layers, with the distant battle going on between the far and mid hills
      const layer = (img: HTMLCanvasElement, f: number, lift: number) => {
        const off = Math.round(mod(cx * f, img.width));
        const y = theme.backdrop === "cave" && img === bd.far ? 0 : parallaxBase(cy, f) - img.height + lift;
        // (the nearest hills carry on down to the bottom of the screen, so deep pits don't show the sky)
        const below = f === 0.4 ? H - (y + img.height) : 0;
        for (let x = -off; x < W; x += img.width) {
          v.drawImage(img, x, y);
          if (below > 0) v.drawImage(img, 0, img.height - 1, img.width, 1, x, y + img.height, img.width, below);
        }
      };
      if (bd.peaks) layer(bd.peaks, 0.04, 6);
      // (crossing into a new biome: the old backdrop fades out under the new one)
      const old = fade < 1 && prevTheme.backdrop !== theme.backdrop ? backdrop(prevTheme.backdrop) : null;
      const pair = (pick: (b: BackdropArt) => HTMLCanvasElement | undefined, f: number, lift: number) => {
        if (old) {
          const o = pick(old);
          if (o) {
            v.globalAlpha = 1 - fade;
            layer(o, f, lift);
          }
          v.globalAlpha = fade;
        }
        const img = pick(bd);
        if (img) layer(img, f, lift);
        v.globalAlpha = 1;
      };
      pair((b) => b.far, 0.08, 20);
      drawFar(v, 1, cx, cy, t);
      pair((b) => b.mid, 0.2, 24);
      pair((b) => b.near, 0.4, 30);
      // Jets crossing the sky, ICBMs coming in
      for (const j of g.jets) {
        const x = Math.round(j.x - cx);
        const y = Math.round(j.y - cy);
        v.globalAlpha = 0.25;
        v.fillStyle = "#e8eef6";
        v.fillRect(j.ally ? x - 80 : x + 20, y, 60, 1);
        v.globalAlpha = 1;
        drawPlane(v, x, y, j.kind, j.ally, t);
      }
      // Each missile's trail: white-hot exhaust just behind it, cooling to smoke that swells and fades
      for (const m of g.missiles) {
        const tr = m.trail;
        if (!tr || tr.length < 6) continue;
        const n = tr.length / 2;
        const big = !m.owner || (m.cls ?? 1) >= 2;
        for (let i = 0; i < n; i++) {
          const age = 1 - i / (n - 1);
          const x = Math.round(tr[i * 2] - cx);
          const y = Math.round(tr[i * 2 + 1] - cy);
          v.globalAlpha = Math.max(0.25, Math.round((1 - age * 0.85) * (big ? 0.9 : 0.7) * 4) / 4);
          v.fillStyle = age < 0.06 ? "#fff3c0" : age < 0.14 ? "#ffb43a" : age < 0.22 ? "#c8704a" : mix("#e0dce4", "#7a7480", age);
          disc(v, x, y, Math.max(1, Math.round((big ? 1.6 : 0.8) + age * (big ? 5.5 : 3))));
        }
        v.globalAlpha = 1;
      }
      for (const m of g.missiles) {
        // (pointing the way it's flying: it curves in on its target)
        const dx = m.up ? 0 : (m.hx ?? m.tx - m.x);
        const dy = m.up ? -1 : (m.hy ?? m.ty - m.y);
        const d = Math.hypot(dx, dy) || 1;
        {
          // Its motor's flame, flickering, out of the back of it
          const ux = m.boost ? 0 : dx / d;
          const uy = m.boost ? -1 : dy / d;
          const len = m.owner && (m.cls ?? 1) < 2 ? 5 : 9;
          const bx = m.x - cx - ux * (len + (m.owner && (m.cls ?? 1) < 3 ? 2 : 20));
          const by = m.y - cy - uy * (len + (m.owner && (m.cls ?? 1) < 3 ? 2 : 20));
          const fl = 0.75 + 0.25 * Math.sin(t * 60 + m.x);
          v.fillStyle = "#ff7a1a";
          disc(v, Math.round(bx - ux * 2), Math.round(by - uy * 2), Math.round((len >= 9 ? 3 : 2) * fl));
          v.fillStyle = "#ffe27a";
          disc(v, Math.round(bx), Math.round(by), Math.round((len >= 9 ? 2 : 1) * fl));
          v.fillStyle = "#ffffff";
          v.fillRect(Math.round(bx + ux * 2), Math.round(by + uy * 2), 1, 1);
        }
        if (!m.owner && !m.up && (m.r ?? 0) < 90) {
          // A warhead coming in: its nose glowing with the heat of the way down
          v.globalAlpha = 0.5;
          v.fillStyle = "#ff9a3a";
          disc(v, Math.round(m.x - cx), Math.round(m.y - cy), 4 + Math.round(Math.sin(t * 24) * 1));
          v.globalAlpha = 1;
        }
        if ((m.r ?? 0) >= 90) {
          // The big one: a hot halo round it all the way down
          v.globalAlpha = 0.35;
          v.fillStyle = "#ffe27a";
          disc(v, Math.round(m.x - cx), Math.round(m.y - cy), 9 + Math.round(Math.sin(t * 20) * 1.5));
          v.globalAlpha = 1;
          drawIcbm(v, Math.round(m.x - cx) - 1, Math.round(m.y - cy), dx / d, dy / d);
          drawIcbm(v, Math.round(m.x - cx) + 1, Math.round(m.y - cy), dx / d, dy / d);
        }
        if (m.owner && (m.cls ?? 1) < 3) {
          // Smaller missiles: a body this long, a red nose
          const cls = m.cls ?? 1;
          const len = [6, 11, 16][cls];
          const ux = m.boost ? 0 : dx / d;
          const uy = m.boost ? -1 : dy / d;
          const mx = Math.round(m.x - cx);
          const my = Math.round(m.y - cy);
          for (let k = 0; k < len; k++) {
            v.fillStyle = k < 2 ? "#c8302a" : m.owner === "ally" ? "#c8d8e8" : "#d8dce4";
            v.fillRect(Math.round(mx - ux * k), Math.round(my - uy * k), cls ? 2 : 1, cls ? 2 : 1);
          }
        } else drawIcbm(v, Math.round(m.x - cx), Math.round(m.y - cy), m.boost ? 0 : dx / d, m.boost ? -1 : dy / d);
      }
      // The worm swims behind the ground: only what's above it shows
      for (const e of g.enemies)
        if (e.kind === "worm" && e.alive)
          drawWorm(
            v,
            e.segs.map((s) => ({ x: s.x - cx, y: s.y - cy, up: s.up })),
            t,
            e.hit > 0
          );

      // Props standing in the background, in shadow behind the ground
      for (const p of g.level.props) {
        if (p.kind !== "sprite" || !p.back || p.x < cx - 60 || p.x > cx + W + 60) continue;
        const img = assetArt(p.id).dark[p.flip ? 1 : 0];
        v.drawImage(img, Math.round(p.x - cx - img.width / 2), Math.round(p.y - cy - img.height + 2));
      }

      // Blocks (the ring buffer may wrap inside the view)
      const ringW = RING * TILE;
      const sy = Math.max(0, cy);
      const sh = Math.min(LEVEL_H, cy + H) - sy;
      if (sh > 0) {
        let dx = 0;
        while (dx < W) {
          const sx = mod(cx + dx, ringW);
          const w = Math.min(W - dx, ringW - sx);
          v.drawImage(terrain, sx, sy, w, sh, dx, sy - cy, w, sh);
          dx += w;
        }
      }

      // Decorations
      const dark = nightNow();
      for (const p of g.level.props) {
        if ((p.kind === "cable" ? p.x2 : p.x) < cx - 50 || p.x > cx + W + 50) continue;
        const x = Math.round(p.x - cx);
        const y = Math.round(p.y - cy);
        switch (p.kind) {
          case "stoplight": {
            // Traffic lights: a pole, the box of three lamps, the one lit for where the crossing is in its cycle
            const st = lightState(g.t, p.x);
            v.fillStyle = "#3a3f4f";
            v.fillRect(x, y - 18, 1, 18);
            v.fillStyle = INK;
            v.fillRect(x - 2, y - 27, 5, 10);
            v.fillStyle = "#1e1e24";
            v.fillRect(x - 1, y - 26, 3, 8);
            v.fillStyle = st === 2 ? "#ff3b3b" : "#4a1414";
            v.fillRect(x - 1, y - 26, 3, 2);
            v.fillStyle = st === 1 ? "#ffb020" : "#4a3410";
            v.fillRect(x - 1, y - 23, 3, 2);
            v.fillStyle = st === 0 ? "#5aff6a" : "#123a1a";
            v.fillRect(x - 1, y - 20, 3, 2);
            break;
          }
          case "streetlight":
            // A tall lamp post, its arm out over the street, the lamp lit at night
            v.fillStyle = "#4a4e5a";
            v.fillRect(x, y - 24, 1, 24);
            v.fillRect(x - 1, y - 2, 3, 2);
            v.fillRect(x, y - 24, 6, 1);
            v.fillStyle = dark > 0.3 ? "#fff3c0" : "#c8ced9";
            v.fillRect(x + 4, y - 23, 3, 1);
            break;
          case "cable": {
            // A suspension bridge's main cable, hanging between its towers, the hangers from it down to the deck
            const x2 = p.x2 - cx;
            const y2 = p.y2 - cy;
            const len = Math.max(1, x2 - x);
            const deck = p.deck - cy;
            for (let k = 0; k <= len; k++) {
              const u = k / len;
              const yy = Math.round(y + (y2 - y) * u + p.sag * 4 * u * (1 - u));
              v.fillStyle = INK;
              v.fillRect(x + k, yy, 1, 2);
              v.fillStyle = "#8a909c";
              v.fillRect(x + k, yy, 1, 1);
              if (k % 8 === 4 && yy < deck - 3) {
                v.fillStyle = "#5a606c";
                v.fillRect(x + k, yy + 2, 1, Math.round(deck - yy - 2));
              }
            }
            break;
          }
          case "sign": {
            // A board with the building's name on it, on two little posts
            const w = p.text.length * 4 + 3;
            const sx = Math.round(x - w / 2);
            v.fillStyle = INK;
            v.fillRect(sx - 1, y - 11, w + 2, 10);
            v.fillRect(sx + 2, y - 2, 1, 2);
            v.fillRect(sx + w - 3, y - 2, 1, 2);
            v.fillStyle = p.board;
            v.fillRect(sx, y - 10, w, 8);
            pixelText(v, p.text, sx + 2, y - 9, p.ink, false);
            break;
          }
          case "awning": {
            // Stripes, sloping down and out over the door
            for (let k = 0; k < p.w; k++) {
              v.fillStyle = (k >> 1) % 2 ? "#f4f0e8" : p.color;
              v.fillRect(x - k - 1, y + Math.floor(k / 5), 1, 3);
            }
            v.fillStyle = INK;
            v.fillRect(x - p.w - 1, y + Math.floor(p.w / 5) + 3, p.w + 1, 1);
            break;
          }
          case "sprite": {
            if (p.back) break;
            const img = assetArt(p.id).img[p.flip ? 1 : 0];
            v.drawImage(img, Math.round(x - img.width / 2), y - img.height + 2);
            break;
          }
          case "legs":
            v.fillStyle = "#6a421c";
            v.fillRect(x, y, 2, p.h);
            v.fillRect(x + p.w - 2, y, 2, p.h);
            v.fillStyle = "#5a3816";
            for (let k = 0; k < p.h - 6; k += 8) {
              v.fillRect(x + 2, y + k + 4, p.w - 4, 1);
              for (let s = 0; s < 6 && s < p.w - 4; s += 2) v.fillRect(x + 2 + Math.round(((p.w - 4) * s) / 6), y + k + 4 + s, 1, 1);
            }
            break;
          case "support":
            v.fillStyle = "#4a5468";
            v.fillRect(x - 1, y, 2, p.h);
            break;
          case "flag":
            v.fillStyle = "#c9ced8";
            v.fillRect(x, y - 18, 1, 18);
            for (let i = 0; i < 10; i++) {
              const o = Math.round(Math.sin(t * 6 * (reduced ? 0 : 1) - i * 0.6) * (i / 9) * 1.5);
              v.fillStyle = "#c8302a";
              v.fillRect(x + 1 + i, y - 18 + o, 1, 6);
              if (i > 2 && i < 7) {
                v.fillStyle = "#1a1a1a";
                v.fillRect(x + 1 + i, y - 16 + o, 1, 2);
              }
            }
            break;
          case "searchlight":
          case "alarm":
          case "lamp":
            v.fillStyle = "#3a3f4f";
            v.fillRect(x - 1, y - (p.kind === "lamp" ? 0 : 2), 4, p.kind === "lamp" ? 2 : 3);
            break;
          case "derrick": {
            // A lattice tower with a nodding pumpjack beside it
            v.fillStyle = "#4a4038";
            for (let k = 0; k < 44; k++) {
              const half = Math.round(7 - (k / 44) * 6);
              v.fillRect(x - half, y - k, 1, 1);
              v.fillRect(x + half, y - k, 1, 1);
              if (k % 7 === 0) v.fillRect(x - half, y - k, half * 2 + 1, 1);
            }
            v.fillRect(x - 2, y - 46, 5, 2);
            const nod = Math.sin(t * 2 * (reduced ? 0 : 1)) * 4;
            v.fillStyle = "#2e2a26";
            v.fillRect(x + 14, y - 12, 2, 12);
            for (let k = -8; k <= 8; k++) v.fillRect(x + 15 + k, Math.round(y - 13 - (nod * k) / 8), 1, 2);
            v.fillRect(x + 22, Math.round(y - 15 - nod), 3, 5);
            break;
          }
          case "radar": {
            v.fillStyle = "#5d6578";
            v.fillRect(x, y - 8, 2, 8);
            const w = Math.round(Math.abs(Math.cos(t * 1.5 * (reduced ? 0 : 1))) * 10) + 2;
            v.fillStyle = INK;
            v.fillRect(x + 1 - w / 2 - 1, y - 16, w + 2, 9);
            v.fillStyle = "#c8ced9";
            v.fillRect(x + 1 - w / 2, y - 15, w, 7);
            v.fillStyle = "#8a90a0";
            v.fillRect(x + 1 - w / 2, y - 10, w, 2);
            break;
          }
          case "silo":
            v.fillStyle = "#14161e";
            v.fillRect(x - 14, y, 28, 2);
            break;
          case "antenna":
            v.fillStyle = "#5d6578";
            v.fillRect(x, y - p.h, 1, p.h);
            for (let k = 6; k < p.h; k += 8) v.fillRect(x - 2, y - k, 5, 1);
            break;
          case "wire":
            v.fillStyle = "#9aa1b0";
            for (let k = 0; k < p.w; k++) v.fillRect(x + k, y - 3 - ((k >> 1) & 1), 1, 1);
            v.fillStyle = "#5d6880";
            for (let k = 0; k < p.w; k += 8) v.fillRect(x + k, y - 5, 1, 5);
            break;
          case "tent": {
            // A ridge tent in canvas, its door flap open
            v.fillStyle = INK;
            for (let k = 0; k < 12; k++) v.fillRect(x - (12 - k) - 1, y - k - 1, (12 - k) * 2 + 3, 1);
            for (let k = 0; k < 11; k++) {
              v.fillStyle = "#5a6a3a";
              v.fillRect(x - (11 - k), y - k - 1, (11 - k) * 2 + 1, 1);
              v.fillStyle = "#6e7e4a";
              v.fillRect(x - (11 - k), y - k - 1, 11 - k, 1);
            }
            v.fillStyle = "#1a1e14";
            for (let k = 0; k < 7; k++) {
              const half = Math.floor((7 - k) / 2);
              v.fillRect(x - half, y - k - 1, half * 2 + 1, 1);
            }
            v.fillStyle = "#8a7a50";
            v.fillRect(x, y - 14, 1, 3);
            break;
          }
          case "rope":
            // Rope handrails sagging between two posts, tied down to the deck
            v.fillStyle = "#5a3816";
            v.fillRect(x - 1, y - 11, 2, 11);
            v.fillRect(x + p.w - 1, y - 11, 2, 11);
            for (let k = 0; k <= p.w; k++) {
              const sag = Math.round(Math.sin((k / p.w) * Math.PI) * 4);
              v.fillStyle = "#a08050";
              v.fillRect(x + k, y - 10 + sag, 1, 1);
              if (k % 6 === 3) {
                v.fillStyle = "#7a5a30";
                v.fillRect(x + k, y - 9 + sag, 1, 9 - sag);
              }
            }
            break;
          case "brazier":
            v.fillStyle = INK;
            v.fillRect(x - 4, y - 9, 9, 3);
            v.fillRect(x - 2, y - 6, 5, 6);
            v.fillStyle = "#6a6e60";
            v.fillRect(x - 3, y - 8, 7, 1);
            v.fillStyle = "#4a4e44";
            v.fillRect(x - 1, y - 6, 3, 6);
            break;
          case "waterfall": {
            // Bands of water racing down; foam where it lands
            for (let k = 0; k < p.w; k++) {
              const ph = hash2(k, 99) * 12 - t * 80 * motion;
              for (let yy = 0; yy < p.h; yy += 3) {
                const band = mod(Math.floor((yy + ph) / 3), 4);
                v.fillStyle = band === 0 ? "#e8f8ff" : band === 1 ? "#9fd8f0" : band === 2 ? "#5ab0d8" : "#4aa0c8";
                v.fillRect(x + k, y + yy, 1, Math.min(3, p.h - yy));
              }
            }
            const tick = Math.floor(t * 6 * motion);
            v.fillStyle = "#f4fcff";
            for (let k = -3; k < p.w + 3; k++) if (hash2(k + p.x, tick) > 0.5) v.fillRect(x + k, y + p.h - 2 - Math.floor(hash2(k, tick + 1) * 3), 1, 1);
            break;
          }
        }
      }

      // Barrels, vehicles, enemies, bosses, players
      for (const b of g.barrels) {
        if (b.x < cx - 40 || b.x > cx + W + 40) continue;
        const x = Math.round(b.x - cx);
        const y = Math.round(b.y - cy);
        if (b.type === "tanker") {
          drawTanker(v, x, y, b.fuse >= 0 && Math.floor(t * 20) % 2 === 0);
          continue;
        }
        const img = props[b.type];
        v.drawImage(img, x - Math.floor(img.width / 2), y - img.height + 1);
        if (b.type === "mine" && Math.floor(t * 3 + b.x) % 3 === 0) {
          v.fillStyle = "#ff3b3b";
          v.fillRect(x, y - 3, 1, 1);
        }
        if (b.fuse >= 0 && Math.floor(t * 20) % 2) {
          v.fillStyle = "#ffffff";
          v.fillRect(x, y - img.height + 2, 1, 1);
        }
      }
      // Supply crates, floating down under their parachutes
      for (const d of g.drops) {
        const x = Math.round(d.x - cx);
        const y = Math.round(d.y - cy);
        if (x < -20 || x > W + 20) continue;
        v.drawImage(props.crate, x - 6, y - props.crate.height + 1);
        if (!d.ground) {
          v.fillStyle = INK;
          v.fillRect(x - 10, y - 22, 21, 5);
          v.fillStyle = "#e8e4dc";
          v.fillRect(x - 9, y - 21, 19, 3);
          v.fillStyle = "#c8302a";
          v.fillRect(x - 3, y - 21, 7, 3);
          v.fillStyle = "#c9ced8";
          v.fillRect(x - 9, y - 18, 1, 10);
          v.fillRect(x + 9, y - 18, 1, 10);
        } else if (Math.floor(t * 4) % 2) {
          v.fillStyle = "#ffe27a";
          v.fillRect(x - 1, y - 11, 3, 1);
        }
      }
      // Freight trains on the main lines, behind the fight: only the part of them out on the open
      // line shows (the rest is still in the tunnel at its end)
      for (const f of g.freights) {
        if (f.wait > 0 || f.hi < cx - 20 || f.lo > cx + W + 20) continue;
        v.save();
        v.beginPath();
        v.rect(Math.round(f.lo - cx), Math.round(f.y - cy) - 40, Math.round(f.hi - f.lo), 44);
        v.clip();
        f.cars.forEach((kind, k) => {
          const left = f.dir > 0 ? f.x - (k + 1) * CAR_LEN : f.x + k * CAR_LEN;
          if (left > cx + W + 50 || left + CAR_LEN < cx - 50) return;
          drawFreightCar(v, Math.round(left - cx), Math.round(f.y - cy) - 1, kind, f.livery, f.anim, f.dir, k * 7 + f.livery + Math.floor(f.lo / 64));
        });
        v.restore();
      }
      // Traffic on the roads: the near lane's in front of the far one; at night their lamps lit
      for (const rd of g.roads) {
        if (rd.hi < cx - 40 || rd.lo > cx + W + 40) continue;
        for (const lane of [1, -1])
          for (const car of rd.cars) {
            if (car.dir !== lane || car.x < cx - 40 || car.x > cx + W + 40) continue;
            drawCar(v, Math.round(car.x - cx), Math.round(rd.y - cy) - (lane > 0 ? 2 : 0), car.kind, car.color, car.dir, car.anim, car.siren, t, car.riders);
          }
      }
      // Metro trains in their tunnels (the squad in its windows, when they're riding)
      for (const m of g.metros) {
        const half = (m.cars * 34) / 2;
        if (m.x + half < cx - 20 || m.x - half > cx + W + 20) continue;
        drawMetroTrain(v, Math.round(m.x - cx), Math.round(m.y - cy), m.cars, m.doors, g.ride === m ? g.crew.map((id) => PLAYER_COLORS[id]) : [], m.dir, Math.floor(m.lo / 8));
      }
      for (const e of g.enemies) {
        if (e.x < cx - 80 || e.x > cx + W + 80 || e.kind === "worm") continue;
        const x = Math.round(e.x - cx);
        const y = Math.round(e.y - cy);
        const hit = e.hit > 0;
        if (e.kind === "heli") {
          drawHeli(v, x, y, t, true);
          continue;
        }
        if (e.kind === "gunship") {
          drawGunship(v, x, y, t, hit);
          continue;
        }
        if (e.kind === "tank") {
          if (e.alive) drawTank(v, x, y, e.anim, e.aimA, hit);
          continue;
        }
        if (e.kind === "mech") {
          if (e.alive) drawMech(v, x, y, e.anim, e.aimA, hit, t);
          continue;
        }
        if (e.kind === "jeep") {
          if (e.alive) drawJeep(v, x, y, e.anim, e.look, t);
          continue;
        }
        if (e.kind === "truck") {
          if (e.alive) drawTruck(v, x, y, e.anim, e.look, e.burst > 0);
          continue;
        }
        if (ARMOR_KINDS.has(e.kind) && e.kind !== "apc" && e.kind !== "lighttank") {
          if (e.alive) drawArmor(v, x, y, e.kind as ArmorType, e.anim, e.aimA, hit, e.hover);
          continue;
        }
        if (e.kind === "flakgun" || e.kind === "howitzer" || e.kind === "samsite" || e.kind === "rocketpod") {
          if (e.alive) drawEmplacement(v, x, y, e.kind, e.aimA, hit, e.hover);
          continue;
        }
        if (e.kind === "train") {
          if (e.alive) drawTrain(v, x, y, e.anim, hit);
          continue;
        }
        if (e.kind === "submarine") {
          const top = surfaceAbove(g, e.x, e.y);
          const under = e.alive && e.spin > 0;
          drawSubmarine(v, x, y, t, under, hit, e.alive ? 0 : clamp01(1 - e.deadT / 4), e.facing, false, under && top !== null && e.y - top < 44 ? top - cy : null);
          continue;
        }
        if (e.kind === "attackheli" || e.kind === "transport" || e.kind === "scoutheli") {
          drawHeliType(v, x, y, t, e.kind as HeliType, hit);
          continue;
        }
        if (e.kind === "apc") {
          if (e.alive) drawApc(v, x, y, e.anim, e.aimA, hit);
          continue;
        }
        if (e.kind === "lighttank") {
          if (e.alive) drawLightTank(v, x, y, e.anim, e.aimA, hit);
          continue;
        }
        if (e.kind === "airship") {
          drawAirship(v, x, y, t, hit);
          continue;
        }
        if (e.kind === "cannon") {
          if (e.alive) drawCannon(v, x, y, e.aimA, hit, e.hover);
          continue;
        }
        if (e.kind === "spider") {
          if (e.alive) drawSpider(v, x, y, e.anim, e.aimA, hit, t);
          continue;
        }
        if (e.kind === "battleship") {
          drawBattleship(v, x, y, t, e.aimA, e.alive ? 0 : clamp01(1 - e.deadT / 5), hit, e.facing);
          continue;
        }
        if (e.kind === "boat") {
          drawBoat(v, x, y, t, e.look, e.facing, e.alive ? 0 : clamp01(1 - e.deadT / 2));
          continue;
        }
        if (e.kind === "drone") {
          drawDrone(v, x, y, t, hit);
          continue;
        }
        const img = troopSprite(e);
        // (troopers come small, large and giant: the sprite is drawn at their size, in whole pixels where it can be)
        const k = e.scale;
        const iw = Math.round(img.width * k);
        const ih = Math.round(img.height * k);
        if (!e.alive) {
          v.globalAlpha = clamp01(e.deadT / 0.6);
          v.save();
          v.translate(x, y - 6 * k);
          v.rotate((e.ground ? -e.facing : Math.floor(e.spin) % 4) * (Math.PI / 2));
          v.drawImage(img, -Math.floor(iw / 2), -Math.floor(ih / 2), iw, ih);
          v.restore();
          v.globalAlpha = 1;
          continue;
        }
        const face = e.facing > 0 ? 0 : 1;
        const ox = (e.kind === "turret" || e.kind === "crawler" || e.kind === "dog" || e.kind === "mg" || e.kind === "mortar" ? img.width / 2 : face ? img.width - 7 : 6) * k;
        const sx = x - Math.round(ox);
        const sy = y - ih + 1;
        if (e.sapper) {
          // An explosive pack on its back
          const bx = x - e.facing * Math.round(5 * k) - Math.round(2 * k);
          v.fillStyle = INK;
          v.fillRect(bx - 1, Math.round(y - 13 * k) - 1, Math.round(4 * k) + 2, Math.round(6 * k) + 2);
          v.fillStyle = "#c8302a";
          v.fillRect(bx, Math.round(y - 13 * k), Math.round(4 * k), Math.round(6 * k));
          if (Math.floor(t * 6 + e.home) % 2) {
            v.fillStyle = "#ffe27a";
            v.fillRect(bx + 1, Math.round(y - 13 * k), 1, 1);
          }
        }
        v.drawImage(img, sx, sy, iw, ih);
        if (hit) {
          // The whole body flashes
          v.globalCompositeOperation = "lighter";
          v.globalAlpha = 0.7;
          v.drawImage(img, sx, sy, iw, ih);
          v.globalAlpha = 1;
          v.globalCompositeOperation = "source-over";
        }
        if (e.chute) {
          const top = y - img.height - 10;
          v.fillStyle = INK;
          v.fillRect(x - 8, top - 1, 17, 4);
          v.fillStyle = "#6a7048";
          v.fillRect(x - 7, top, 15, 2);
          v.fillStyle = "#8a9060";
          v.fillRect(x - 5, top, 11, 1);
          v.fillStyle = "#9aa1b0";
          v.fillRect(x - 7, top + 3, 1, 9);
          v.fillRect(x + 7, top + 3, 1, 9);
        }
      }
      // The allied column
      for (const a of g.allies) {
        if (a.x < cx - 60 || a.x > cx + W + 60) continue;
        const x = Math.round(a.x - cx);
        const y = Math.round(a.y - cy);
        if (a.dead) v.globalAlpha = 0.55;
        if (a.kind === "gunship") {
          const boarding = g.ship === a && g.shipMode === "board";
          const rope = boarding ? Math.max(0, Math.min(40, g.level.ground[Math.floor(a.x / TILE) & MASK] * TILE - (a.y + 8))) : 0;
          drawAllyGunship(v, x, y, t, a.aimA, a.hit > 0, g.ship === a ? g.crew.map((id) => PLAYER_COLORS[id]) : [], Math.round(rope));
        } else if (ALLY_AIR.has(a.kind)) drawHeliType(v, x, y, t, a.kind as HeliType, a.hit > 0);
        else drawArmor(v, x, y, a.kind as ArmorType, a.anim, a.aimA, a.hit > 0);
        v.globalAlpha = 1;
      }
      // Mine carts rolling on their rails: the load heaped in them, their wheels going round
      for (const k of g.carts) {
        if (k.x < cx - 20 || k.x > cx + W + 20) continue;
        const x = Math.round(k.x - cx);
        const y = Math.round(k.y - cy);
        const img = assetArt(k.load === 0 ? "cartcoal" : "minecart").img[k.vx < 0 ? 1 : 0];
        v.drawImage(img, x - Math.round(img.width / 2), y - img.height + 2);
        if (k.load === 1 || k.load === 2) {
          v.fillStyle = k.load === 1 ? "#ffd23f" : "#c8906a";
          for (let i = -4; i <= 4; i += 2) v.fillRect(x + i, y - 10 - (Math.abs(i) < 3 ? 1 : 0), 2, 2);
          v.fillStyle = "#fff6c8";
          v.fillRect(x - 1, y - 11, 1, 1);
        }
        v.fillStyle = "#c8ced9";
        for (const wx of [-4, 3]) v.fillRect(x + wx + Math.round(Math.cos(k.anim + wx)), y - 1 + Math.round(Math.sin(k.anim + wx)), 1, 1);
      }
      // Manholes: a cover flush with the road, thrown open when something climbs out
      for (const h of g.hatches) {
        const x = Math.round(h.x - cx);
        const y = Math.round(h.y - cy);
        if (x < -10 || x > W + 10) continue;
        v.fillStyle = INK;
        v.fillRect(x - 4, y - 1, 9, 2);
        v.fillStyle = h.open > 0 ? "#0d0b14" : "#5a5a62";
        v.fillRect(x - 3, y - 1, 7, 1);
        if (h.open > 0) {
          v.fillStyle = "#5a5a62";
          v.fillRect(x + 4, y - 7, 2, 6);
        }
      }
      // Our submarines out in the sea (the water over them comes later)
      for (const s of g.navy) {
        if (s.x < cx - 60 || s.x > cx + W + 60) continue;
        const top = surfaceAbove(g, s.x, s.y);
        const under = top !== null && s.y - top > 9;
        drawSubmarine(v, Math.round(s.x - cx), Math.round(s.y - cy), t, under, false, 0, s.facing, true, under && s.y - top < 44 ? top - cy : null);
      }
      // The submarines' missiles, under the water and out of it (drawn here so the sea covers the wet ones)
      for (const s of g.sams) {
        const sp = Math.hypot(s.vx, s.vy) || 1;
        drawIcbm(v, Math.round(s.x - cx), Math.round(s.y - cy), s.vx / sp, s.vy / sp);
      }
      for (const p of g.players) {
        if (p.x < cx - 20 || p.x > cx + W + 20) continue;
        if (p.inv > 0 && !p.dead && Math.floor(t * 18) % 2 === 0) continue;
        const x = Math.round(p.x - cx);
        const y = Math.round(p.y - cy);
        const frames = loadouts(p.id, p.weapon);
        const face = p.facing > 0 ? 0 : 1;
        if (p.dead) {
          const img = frames[FRAMES.idle][face];
          v.save();
          v.translate(x, y - 6);
          v.rotate((p.ground ? -p.facing : Math.floor(p.spin) % 4) * (Math.PI / 2));
          v.drawImage(img, -8, -10);
          v.restore();
          continue;
        }
        const frame = !p.ground ? FRAMES.jump : p.vx ? FRAMES.run[Math.floor(p.anim) % 4] : FRAMES.idle;
        const kick = p.kick > 0 ? -p.facing : 0;
        if (p.jet > 0) {
          // The jetpack on their back, burning
          const jx = x - p.facing * 5 - 1;
          v.fillStyle = INK;
          v.fillRect(jx - 1, y - 14, 5, 8);
          v.fillStyle = "#8a90a0";
          v.fillRect(jx, y - 13, 3, 6);
          v.fillStyle = Math.floor(t * 20) % 2 ? "#ffe27a" : "#ff8a1a";
          v.fillRect(jx, y - 7, 3, 2 + (Math.floor(t * 30) % 3));
        }
        v.drawImage(frames[frame][face], x - (face ? 9 : 6) + kick, y - 19);
        if (p.chute) {
          // Parachute in the player's colour
          v.fillStyle = INK;
          v.fillRect(x - 9, y - 34, 19, 5);
          v.fillStyle = PLAYER_COLORS[p.id];
          v.fillRect(x - 8, y - 33, 17, 3);
          v.fillStyle = mix(PLAYER_COLORS[p.id], "#ffffff", 0.4);
          v.fillRect(x - 6, y - 33, 13, 1);
          v.fillStyle = "#c9ced8";
          v.fillRect(x - 8, y - 29, 1, 8);
          v.fillRect(x + 8, y - 29, 1, 8);
          v.fillRect(x, y - 29, 1, 9);
        }
      }

      // The sea over whatever is in it (see makeSea), and its surface: a swell running along it,
      // crests and glints, light rippling just under it, foam lapping at the shore
      {
        const L = g.level;
        const c0 = Math.floor(cx / TILE);
        const c1 = Math.ceil((cx + W) / TILE);
        const r0 = Math.max(0, Math.floor(cy / TILE));
        const r1 = Math.min(ROWS - 1, Math.ceil((cy + H) / TILE));
        const tm = t * motion;
        for (let c = c0; c <= c1; c++) {
          const x = c * TILE - cx;
          for (let r = r0; r <= r1; r++) {
            if (kindAt(L, c, r) !== WATER) continue;
            // One run of water down this column: its surface, and how far down it goes in view
            let top = r;
            while (top > 0 && kindAt(L, c, top - 1) === WATER) top--;
            let end = r;
            while (end < r1 && kindAt(L, c, end + 1) === WATER) end++;
            const h = (end - r + 1) * TILE;
            v.drawImage(sea.front, 0, (r - top) * TILE, TILE, h, x, r * TILE - cy, TILE, h);
            if (top === r) {
              const y = r * TILE - cy;
              const roofed = SOLID[kindAt(L, c, r - 1)] === 1;
              for (let k = 0; k < TILE; k++) {
                const wx = c * TILE + k;
                const swell = Math.sin(wx * 0.19 + tm * 2.1) + 0.8 * Math.sin(wx * 0.067 - tm * 1.3);
                if (swell > 1.15 && !roofed) {
                  v.fillStyle = "#d8f4fa";
                  v.fillRect(x + k, y - 1, 1, 1);
                }
                v.fillStyle = swell > 0.3 ? "#e4f8fc" : swell < -0.9 ? "#7ccbe2" : "#aee2f0";
                v.fillRect(x + k, y, 1, 1);
              }
              const glint = hash2(c, Math.floor(tm * 1.5 + c * 0.37));
              if (glint > 0.82) {
                v.fillStyle = "#ffffff";
                v.fillRect(x + (((glint * 61) | 0) % 6), y + 1, 2, 1);
              }
              const off = mod(Math.floor(hash2(c, 5) * 12 + tm * 6), 14);
              if (off < 8) {
                v.fillStyle = "rgba(200,242,250,0.22)";
                v.fillRect(x + off, y + 3 + ((hash2(c, 9) * 10) | 0), Math.min(3, 8 - off), 1);
              }
              const lap = Math.floor(tm * 2 + c) % 3;
              v.fillStyle = "#f0fbfe";
              if (SOLID[kindAt(L, c - 1, r)]) {
                v.fillRect(x, y, lap === 0 ? 3 : 2, 1);
                v.fillRect(x, y + 1, 1, 1);
              }
              if (SOLID[kindAt(L, c + 1, r)]) {
                v.fillRect(x + TILE - (lap === 1 ? 3 : 2), y, lap === 1 ? 3 : 2, 1);
                v.fillRect(x + TILE - 1, y + 1, 1, 1);
              }
            }
            r = end;
          }
        }
      }

      // Debris, shell casings, smoke and spray; then the foreground strip
      for (const p of g.parts) {
        if (p.k === DEBRIS || p.k === CASING) {
          v.fillStyle = p.k === CASING || p.c[0] === "#" ? p.c : (DEBRIS_COLOR[+p.c] ?? "#8a5a2b");
          v.fillRect(Math.round(p.x - cx), Math.round(p.y - cy), p.s, p.s);
        } else if (p.k === BUBBLE) {
          // A little ring of air, lit on its upper side
          const x = Math.round(p.x - cx);
          const y = Math.round(p.y - cy);
          v.fillStyle = "#cdeef8";
          if (p.s > 1) {
            v.fillRect(x, y - 1, 2, 1);
            v.fillRect(x - 1, y, 1, 2);
            v.fillRect(x + 2, y, 1, 2);
            v.fillRect(x, y + 2, 2, 1);
          } else v.fillRect(x, y, 1, 1);
        } else if (p.k === MIST) {
          const a = p.life / p.max;
          v.globalAlpha = Math.round(a * 0.6 * 4) / 4;
          v.fillStyle = "#e8f6fa";
          disc(v, Math.round(p.x - cx), Math.round(p.y - cy), Math.round(p.s + (1 - a) * 3));
          v.globalAlpha = 1;
        } else if (p.k === DUST) {
          const a = p.life / p.max;
          v.globalAlpha = Math.round(a * 0.7 * 4) / 4;
          v.fillStyle = p.c;
          disc(v, Math.round(p.x - cx), Math.round(p.y - cy), Math.round(p.s + (1 - a) * 4));
          v.globalAlpha = 1;
        } else if (p.k === SMOKE || p.k === BIGSMOKE) {
          const a = p.life / p.max;
          v.globalAlpha = Math.round(a * (p.k === BIGSMOKE ? 0.6 : 0.75) * 4) / 4;
          v.fillStyle = p.k === BIGSMOKE ? (a > 0.6 ? "#6a6470" : "#8a8490") : a > 0.6 ? "#2b2830" : "#5b5866";
          disc(v, Math.round(p.x - cx), Math.round(p.y - cy), Math.round(p.s + (1 - a) * (p.k === BIGSMOKE ? 8 : 3)));
          v.globalAlpha = 1;
        }
      }
      const fo = Math.round(mod(cx * 1.25, bd.front.width));
      for (let x = -fo; x < W; x += bd.front.width) v.drawImage(bd.front, x, H - bd.front.height + 6);
      // Jungles: leaves and vines hanging into the top of the view, closest of all
      if (bd.top) {
        const to = Math.round(mod(cx * 1.4, bd.top.width));
        for (let x = -to; x < W; x += bd.top.width) v.drawImage(bd.top, x, 0);
      }

      // The theme's tint over the whole world layer (depth below the ground is shaded in the blocks themselves)
      v.globalCompositeOperation = "source-atop";
      const night = nightNow();
      if (night > 0.01) {
        v.fillStyle = `rgba(8,10,34,${night})`;
        v.fillRect(0, 0, W, H);
      }
      if (theme.tint) {
        v.fillStyle = theme.tint;
        v.fillRect(0, 0, W, H);
      }
      v.globalCompositeOperation = "source-over";
    };

    const glow = (x: number, y: number, w: number, h: number, col: string, a: number) => {
      ctx.globalAlpha = Math.min(1, a) * 0.3;
      ctx.fillStyle = col;
      ctx.fillRect(x - 1, y - 1, w + 2, h + 2);
      ctx.globalAlpha = Math.min(1, a);
      ctx.fillRect(x, y, w, h);
      ctx.globalAlpha = 1;
    };

    /**
     * Explosions: every fireball ball of every blast, drawn in passes (cooled smoke behind;
     * then the dark outlines, the hot bodies, the highlights) so a cluster reads as one
     * lumpy, outlined blob of fire.
     */
    /**
     * A mushroom cloud. First the fireball, swelling white hot off the ground; then it rises on a
     * stem of smoke and dust (its skirt spreading over the ground, a white condensation ring
     * round it), and rolls over into the cap: a ring of cloud turning over on itself, lit from
     * inside by the fire while it's hot, its fiery heart showing through underneath. As it climbs
     * it cools to brown and grey smoke and slowly drifts apart. All of it built of shaded,
     * outlined puffs of cloud, the same lumps every frame.
     */
    const drawMushroom = (m: { x: number; y: number; t: number; s?: number }, cx: number, cy: number) => {
      const s = m.s ?? 1;
      const T = m.t;
      const fade = T > 6 ? clamp01(1 - (T - 6) / 3) : 1;
      if (fade <= 0) return;
      const R = rng(Math.floor(m.x * 7 + m.y));
      const ease = (u: number) => 1 - Math.pow(1 - clamp01(u), 2.2);
      const x0 = Math.round(m.x - cx);
      const gy = Math.round(m.y - cy);
      const heat = clamp01(1 - T / 2.6);
      const cool = clamp01((T - 1.2) / 3);
      const smokeD = mix("#2a1e22", "#3e383c", cool);
      const smokeM = mix("#6a4436", "#706a6a", cool);
      const smokeL = mix("#a0705a", "#a49e9c", cool);
      const fire = (u: number) => (u > 0.75 ? "#fff6c8" : u > 0.5 ? "#ffd23f" : u > 0.25 ? "#ff8a1a" : "#c8341a");
      // (the biggest spread wider, but rise no higher than will stay in view)
      const capH = (24 + 112 * ease(T / 2.6) + T * 3) * Math.min(s, 1.1);
      const capY = gy - capH;
      const capRx = (14 + 30 * ease(T / 2) + T * 2.2) * s;
      const capRy = capRx * 0.42;
      /** A puff of cloud: where, how big, its colour and its lit side's, the fire's glow under it and how strong */
      type Puff = [number, number, number, string, string, string, number];
      /**
       * A layer of the cloud's puffs, drawn as one: one dark outline round the lot, then their
       * bodies, the light on their tops, the fire glowing under them (no seams between them).
       */
      const layer = (ps: Puff[]) => {
        ctx.fillStyle = smokeD;
        for (const [px, py, r] of ps) disc(ctx, px, py, r + 1);
        for (const [px, py, r, body] of ps) {
          ctx.fillStyle = body;
          disc(ctx, px, py, r);
        }
        // (a shade on the side away from the light, then the light on top)
        ctx.fillStyle = mix(smokeM, smokeD, 0.45);
        for (const [px, py, r] of ps) if (r >= 4) disc(ctx, px + Math.round(r * 0.3), py + Math.round(r * 0.35), Math.round(r * 0.55));
        for (const [px, py, r, body] of ps) {
          ctx.fillStyle = body;
          if (r >= 4) disc(ctx, px - Math.round(r * 0.08), py - Math.round(r * 0.1), Math.round(r * 0.72));
        }
        for (const [px, py, r, , light] of ps) {
          ctx.fillStyle = light;
          disc(ctx, px - Math.round(r * 0.3), py - Math.round(r * 0.35), Math.max(1, Math.round(r * 0.45)));
        }
        for (const [px, py, r, , , glow, glowA] of ps) {
          if (glowA <= 0.05) continue;
          ctx.globalAlpha = Math.round(glowA * fade * 4) / 4;
          ctx.fillStyle = glow;
          disc(ctx, px, py + Math.round(r * 0.35), Math.max(1, Math.round(r * 0.6)));
        }
        ctx.globalAlpha = fade;
      };
      ctx.globalAlpha = fade;
      // The base surge: dust and smoke rolling out over the ground, both ways
      const surge = ease(T / 3.2) * 110 * s;
      const base: Puff[] = [];
      for (let k = 0; k < 16; k++) {
        const side = k % 2 ? 1 : -1;
        const u = (k >> 1) / 7;
        const r = Math.round((4 + R() * 4 + u * 3) * s * (0.6 + 0.4 * ease(T)));
        base.push([x0 + side * Math.round(surge * (0.2 + u * 0.8)), gy - r + 2 - Math.round(R() * 3), r, smokeM, smokeL, "", 0]);
      }
      layer(base);
      // The stem, from the ground up to the cap: its skirt wide at the foot, narrow under the cap,
      // hot near the top while the fire's in it
      if (T > 0.25) {
        const stemW = (5 + 4 * ease(T / 2)) * s;
        const stem: Puff[] = [];
        for (let y = gy - 4; y > capY + capRy * 0.5; y -= 4) {
          const u = (gy - y) / Math.max(1, gy - capY);
          const w = stemW * (1 + 1.6 * Math.pow(1 - u, 3)) * (0.8 + 0.2 * Math.sin(y * 0.4 + T * 3));
          const wob = Math.sin(y * 0.11 + T * 1.6) * 2 * s;
          const hot = heat * u;
          stem.push([Math.round(x0 + wob + (R() - 0.5) * w * 0.4), Math.round(y), Math.max(1, Math.round(w)), hot > 0.3 ? mix(smokeM, "#ff7a1a", hot) : smokeM, hot > 0.3 ? mix(smokeL, "#ffd23f", hot) : smokeL, fire(hot), hot * 0.6]);
        }
        layer(stem);
        // The condensation ring round the stem, for a moment
        if (T > 0.7 && T < 3.2) {
          const ra = clamp01(Math.min((T - 0.7) / 0.4, (3.2 - T) / 0.8));
          ctx.globalAlpha = Math.round(0.6 * ra * fade * 4) / 4;
          ctx.fillStyle = "#f4f6fa";
          const ry = Math.round(gy - capH * 0.55);
          const rw = Math.round(stemW * 3.4);
          for (let dx = -rw; dx <= rw; dx++) {
            const e = Math.sqrt(Math.max(0, 1 - (dx / rw) ** 2));
            ctx.fillRect(x0 + dx, ry - Math.round(e * 3), 1, 1);
            ctx.fillRect(x0 + dx, ry + Math.round(e * 3), 1, 1);
          }
          ctx.globalAlpha = fade;
        }
      }
      // The cap: the back of its ring of cloud, a billowing dome over the middle, then the front of
      // the ring rolling over on itself, lit from inside by the fire while it's hot
      const back: Puff[] = [];
      const front: Puff[] = [];
      for (let k = 0; k < 20; k++) {
        const a = (k / 20) * Math.PI * 2 + T * 0.7 + (R() - 0.5) * 0.25;
        const depth = Math.sin(a);
        const under = depth > 0.2;
        const r = Math.max(1, Math.round(capRy * (0.55 + R() * 0.4)));
        const p: Puff = [
          Math.round(x0 + Math.cos(a) * capRx * (0.92 + R() * 0.16)),
          Math.round(capY + depth * capRy * 0.9 + (R() - 0.5) * 3 * s),
          r,
          under && heat > 0.3 ? mix(smokeM, "#ff8a1a", heat * 0.7) : smokeM,
          under ? mix(smokeL, "#ffb43a", heat * 0.5) : smokeL,
          fire(heat * (under ? 1 : 0.6)),
          under ? heat : heat * 0.4,
        ];
        (depth < 0 ? back : front).push(p);
      }
      layer(back);
      const dome: Puff[] = [[x0, Math.round(capY - capRy * 0.35), Math.round(capRx * 0.6), smokeM, smokeL, fire(heat), heat * 0.9]];
      for (let k = 0; k < 6; k++) {
        const a = Math.PI * (1.1 + (k / 5) * 0.8);
        dome.push([Math.round(x0 + Math.cos(a) * capRx * 0.45), Math.round(capY - capRy * 0.35 + Math.sin(a) * capRy * 0.9), Math.max(1, Math.round(capRx * (0.2 + R() * 0.12))), smokeM, smokeL, "", 0]);
      }
      layer(dome);
      layer(front);
      // Its fiery heart, showing through underneath while it's hot
      if (heat > 0.05) {
        ctx.globalAlpha = Math.round(heat * fade * 4) / 4;
        ctx.fillStyle = fire(heat);
        disc(ctx, x0, Math.round(capY + capRy * 0.2), Math.max(1, Math.round(capRx * 0.45 * (0.6 + heat * 0.4))));
        ctx.fillStyle = "#fff6c8";
        disc(ctx, x0, Math.round(capY + capRy * 0.1), Math.max(1, Math.round(capRx * 0.22 * heat)));
      }
      // The first instant: the fireball itself, swelling white hot off the ground
      if (T < 0.7) {
        const u = T / 0.7;
        const fr = Math.round((10 + 30 * u) * s);
        const fy = gy - Math.round(fr * 0.8);
        ctx.globalAlpha = Math.round((1 - u * 0.6) * 4) / 4;
        ctx.fillStyle = "#ff8a1a";
        disc(ctx, x0, fy, fr + 3);
        ctx.fillStyle = "#ffd23f";
        disc(ctx, x0, fy, fr + 1);
        ctx.fillStyle = "#fff6c8";
        disc(ctx, x0, fy, fr);
        ctx.fillStyle = "#ffffff";
        disc(ctx, x0, fy, Math.round(fr * 0.6));
      }
      ctx.globalAlpha = 1;
    };
    /** Rays of light out of a big blast's flash. */
    const rays = (x: number, y: number, len: number, col: string, seed: number, a: number) => {
      ctx.globalCompositeOperation = "lighter";
      ctx.fillStyle = col;
      ctx.globalAlpha = Math.round(a * 4) / 4;
      for (let k = 0; k < 10; k++) {
        const ang = (k / 10) * Math.PI * 2 + (seed % 7) * 0.31;
        const l = len * (k % 2 ? 0.55 : 1);
        const ux = Math.cos(ang);
        const uy = Math.sin(ang);
        for (let d = 5; d < l; d += 2) ctx.fillRect(Math.round(x + ux * d), Math.round(y + uy * d), 1, 1);
      }
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = "source-over";
    };
    const blazeR = (u: number, s: number) => Math.max(1, Math.round(s * (0.55 + 0.6 * Math.min(1, u * 4)) * (u > 0.75 ? 1 - (u - 0.75) * 1.6 : 1)));
    const HOT = 0.62;
    const drawFireballs = (g: Game, cx: number, cy: number) => {
      const parts = g.parts;
      for (const p of parts) {
        if (p.k !== BLAZE) continue;
        const u = 1 - p.life / p.max;
        if (u < HOT) continue;
        ctx.globalAlpha = Math.max(0.25, Math.round((1 - (u - HOT) / (1 - HOT)) * 0.8 * 4) / 4);
        ctx.fillStyle = firePal(p.c).smoke[u < 0.8 ? 0 : 1];
        disc(ctx, Math.round(p.x - cx), Math.round(p.y - cy), blazeR(u, p.s) + 1);
      }
      ctx.globalAlpha = 1;
      for (let pass = 0; pass < 3; pass++)
        for (const p of parts) {
          if (p.k !== BLAZE) continue;
          const u = 1 - p.life / p.max;
          if (u >= HOT) continue;
          const R = blazeR(u, p.s);
          const x = Math.round(p.x - cx);
          const y = Math.round(p.y - cy);
          const P = firePal(p.c);
          if (pass === 0) {
            ctx.fillStyle = P.rim[u < 0.4 ? 0 : 1];
            disc(ctx, x, y, R + 1);
          } else if (pass === 1) {
            // White hot, then through the fire's colours as it cools
            ctx.fillStyle = u < 0.08 ? P.core : u < 0.2 ? P.hot : u < 0.34 ? P.mid : u < 0.5 ? P.cool : P.edge;
            disc(ctx, x, y, R);
          } else if (R >= 2) {
            ctx.fillStyle = P.hi[u < 0.24 ? 0 : u < 0.42 ? 1 : 2];
            disc(ctx, x - Math.round(R * 0.3), y - Math.round(R * 0.3), Math.max(1, Math.round(R * 0.45)));
          }
        }
    };

    const drawLights = (g: Game, cx: number, cy: number, t: number) => {
      const night = nightNow();
      const L = g.level;

      // Lit windows at night, lava, crystals
      const c0 = Math.floor(cx / TILE);
      const c1 = Math.ceil((cx + W) / TILE);
      const r0 = Math.max(0, Math.floor(cy / TILE));
      const r1 = Math.min(ROWS - 1, Math.ceil((cy + H) / TILE));
      for (let c = c0; c <= c1; c++)
        for (let r = r0; r <= r1; r++) {
          const k = kindAt(L, c, r);
          const x = c * TILE - cx;
          const y = r * TILE - cy;
          // (the windows of the towers behind the streets, lit up at night, not every one)
          if (k === AIR && night > 0.3 && c >= L.start && c < L.next && L.bg[cell(c, r)] === WINDOW) {
            if (hash2(c, r) > 0.45) glow(x + 2, y + 2, 4, 3, hash2(r, c) > 0.8 ? "#bfe0ff" : "#ffd479", (night - 0.2) * 1.1);
            continue;
          }
          if (k !== WINDOW && k !== LAVA && k !== CRYSTAL && k !== MAGMA && k !== AMETHYST && k !== MYCELIUM && k !== DIAMOND) continue;
          if (k === MAGMA) {
            // Magma rock's cracks glowing, slowly
            if (hash2(c, r) > 0.35) glow(x + 2 + Math.floor(hash2(r, c) * 4), y + 3, 2, 2, "#ff7a1a", 0.35 + 0.25 * Math.sin(t * 2.4 + c * 1.3 + r));
            continue;
          }
          if (k === AMETHYST) {
            glow(x + 3, y + 3, 2, 2, "#d88aff", 0.45 + 0.3 * Math.sin(t * 1.7 + c + r));
            continue;
          }
          if (k === MYCELIUM) {
            glow(x + Math.floor(hash2(c, r) * 6), y + 3, 1, 1, "#7affd8", 0.55 + 0.4 * Math.sin(t * 2 + c * 0.7));
            continue;
          }
          if (k === DIAMOND) {
            if (Math.sin(t * 4 + c * 7 + r * 3) > 0.8) glow(x + 2, y + 2, 1, 1, "#ffffff", 1);
            continue;
          }
          if (k === WINDOW) {
            if (night > 0.15 && hash2(c, r) > 0.3) glow(x + 2, y + 2, 4, 4, "#ffd479", night * 1.4);
          } else if (k === LAVA) {
            if (kindAt(L, c, r - 1) === LAVA) continue;
            ctx.globalAlpha = 0.35 + 0.15 * Math.sin(t * 3 + c);
            ctx.fillStyle = "#ff8a1a";
            ctx.fillRect(x, y - 3, TILE, 3);
            ctx.globalAlpha = 1;
            if (hash2(c, Math.floor(t * 3)) > 0.9) glow(x + 3, y - 2, 2, 2, "#ffe27a", 1);
          } else glow(x + 3, y + 3, 2, 2, "#e6fffc", 0.7 + 0.3 * Math.sin(t * 2 + c));
        }

      ctx.globalCompositeOperation = "lighter";
      for (const p of L.props) {
        if (p.x < cx - 140 || p.x > cx + W + 140) continue;
        const x = Math.round(p.x - cx);
        const y = Math.round(p.y - cy);
        if (p.kind === "searchlight" && night > 0.2) {
          const a = Math.PI / 2 + 0.55 + Math.sin(t * 0.7 + p.x) * 0.6;
          ctx.fillStyle = `rgba(255,244,200,${0.16 * night})`;
          ctx.beginPath();
          ctx.moveTo(x + 1, y);
          ctx.lineTo(x + 1 + Math.cos(a - 0.12) * 150, y + Math.sin(a - 0.12) * 150);
          ctx.lineTo(x + 1 + Math.cos(a + 0.12) * 150, y + Math.sin(a + 0.12) * 150);
          ctx.closePath();
          ctx.fill();
        } else if (p.kind === "alarm") {
          const on = Math.floor(t * 4 + p.x) % 2 === 0;
          glow(x - 1, y - 3, 4, 2, on ? "#ff3b3b" : "#5a1a1a", 1);
          if (on) {
            ctx.globalAlpha = 0.18;
            ctx.fillStyle = "#ff3b3b";
            disc(ctx, x + 1, y - 2, 10);
            ctx.globalAlpha = 1;
          }
        } else if (p.kind === "lamp") {
          glow(x, y + 1, 2, 1, "#ffd479", 1);
          ctx.globalAlpha = 0.1;
          ctx.fillStyle = "#ffd479";
          disc(ctx, x + 1, y + 6, 9);
          ctx.globalAlpha = 1;
        } else if (p.kind === "streetlight") {
          // At night: the lamp lit, a cone of light down from it, a pool of it on the street
          if (night < 0.3) continue;
          const a = Math.min(1, (night - 0.3) * 2.5);
          glow(x + 4, y - 23, 3, 1, "#fff3c0", a);
          ctx.globalAlpha = 0.07 * a;
          ctx.fillStyle = "#ffe8a0";
          for (let k = 0; k < 22; k++) ctx.fillRect(x + 5 - Math.round(k * 0.4), y - 22 + k, 1 + Math.round(k * 0.8), 1);
          ctx.globalAlpha = 0.12 * a;
          ctx.fillRect(x - 5, y - 1, 21, 2);
          ctx.globalAlpha = 1;
        } else if (p.kind === "stoplight") {
          const st = lightState(t, p.x);
          glow(x - 1, y - 26 + (2 - st) * 3, 3, 2, st === 2 ? "#ff3b3b" : st === 1 ? "#ffb020" : "#5aff6a", 0.5 + night * 0.5);
        } else if (p.kind === "sign") {
          if (!p.neon) continue;
          // Neon: a glow round the board that buzzes now and then
          const w = p.text.length * 4 + 3;
          const on = hash2(Math.floor(t * 8), Math.round(p.x)) > 0.04 ? 1 : 0.2;
          ctx.globalAlpha = (0.1 + night * 0.25) * on;
          ctx.fillStyle = p.ink;
          ctx.fillRect(Math.round(x - w / 2) - 2, y - 12, w + 4, 12);
          ctx.globalAlpha = 1;
          if (night > 0.2) pixelText(ctx, p.text, Math.round(x - w / 2) + 2, y - 9, p.ink, false);
        } else if (p.kind === "sprite") {
          const a = asset(p.id);
          if (!a.lights) continue;
          const x0 = Math.round(p.x - cx - a.w / 2);
          const y0 = y - a.h + 1;
          for (const [lx, ly, col, rate] of a.lights) {
            const on = rate === 0 ? 1 : 0.35 + 0.65 * (0.5 + 0.5 * Math.sin(t * rate * 6 + p.x));
            const gx = x0 + (p.flip ? a.w - 1 - lx : lx);
            glow(gx, y0 + ly, 1, 1, col, on * (p.back ? 0.5 : 1));
            if (night > 0.2 || rate === 0) {
              ctx.globalAlpha = 0.08 * on * (0.5 + night);
              ctx.fillStyle = col;
              disc(ctx, gx, y0 + ly, 6);
              ctx.globalAlpha = 1;
            }
          }
        } else if (p.kind === "antenna") glow(x - 1, y - p.h - 2, 2, 2, Math.floor(t * 1.5) % 2 ? "#ff3b3b" : "#5a1a1a", 1);
        else if (p.kind === "fire" || p.kind === "brazier") {
          ctx.globalAlpha = 0.14 + 0.06 * Math.sin(t * 9 + p.x) + (p.kind === "brazier" ? night * 0.12 : 0);
          ctx.fillStyle = "#ff8a1a";
          disc(ctx, x, y - (p.kind === "brazier" ? 11 : 4), 10);
          ctx.globalAlpha = 1;
        }
      }
      for (const f of g.fires) {
        ctx.globalAlpha = 0.12;
        ctx.fillStyle = "#ff8a1a";
        disc(ctx, Math.round(f.x - cx), Math.round(f.y - cy - 4), 9);
        ctx.globalAlpha = 1;
      }
      ctx.globalCompositeOperation = "source-over";

      // Sniper laser sights
      for (const e of g.enemies) {
        if (e.kind !== "sniper" || !e.alive || e.aim <= 0 || !e.target) continue;
        const x0 = e.x + e.facing * 7 - cx;
        const y0 = e.y - 9 - cy;
        const x1 = e.target.x - cx;
        const y1 = e.target.y - 9 - cy;
        const n = Math.ceil(Math.hypot(x1 - x0, y1 - y0) / 2);
        ctx.fillStyle = "#ff3b3b";
        ctx.globalAlpha = 0.25 + e.aim * 0.6;
        for (let k = 0; k < n; k++) ctx.fillRect(Math.round(x0 + ((x1 - x0) * k) / n), Math.round(y0 + ((y1 - y0) * k) / n), 1, 1);
        ctx.globalAlpha = 1;
      }

      // Laser beams and tesla lightning
      ctx.globalCompositeOperation = "lighter";
      // Laser beams: a glow along them, the beam in its colour, a white-hot core; the heavy ones
      // wider, flickering, flaring where they strike
      for (const b of g.beams) {
        const n = Math.ceil(Math.hypot(b.x1 - b.x0, b.y1 - b.y0));
        const w = b.w ?? 1;
        const fl = w > 1 ? (Math.sin(t * 50 + b.x0) > 0 ? 1 : 0) : 0;
        for (let k = 0; k < n; k++) {
          const px = Math.round(b.x0 + ((b.x1 - b.x0) * k) / n - cx);
          const py = Math.round(b.y0 + ((b.y1 - b.y0) * k) / n - cy);
          ctx.fillStyle = b.c;
          ctx.globalAlpha = 0.3;
          ctx.fillRect(px, py - w - 1, 1, w * 2 + 3);
          ctx.globalAlpha = 0.8;
          ctx.fillRect(px, py - (w >> 1) - fl, 1, w + fl * 2);
          ctx.globalAlpha = 1;
          ctx.fillStyle = "#ffffff";
          ctx.fillRect(px, py - (w > 2 ? 1 : 0), 1, w > 2 ? 2 : 1);
        }
        if (w > 1) {
          const ex = Math.round(b.x1 - cx);
          const ey = Math.round(b.y1 - cy);
          glow(ex - w, ey - w, w * 2 + 1, w * 2 + 1, b.c, 0.9);
          ctx.fillStyle = "#ffffff";
          disc(ctx, ex, ey, w);
        }
      }
      for (const z of g.zaps) {
        const R = rng(Math.floor(t * 30) + z.pts.length);
        ctx.fillStyle = "#c8e8ff";
        for (let i = 1; i < z.pts.length; i++) {
          let [ax, ay] = z.pts[i - 1];
          const [bx, by] = z.pts[i];
          for (let s2 = 1; s2 <= 5; s2++) {
            const nx = ax + (bx - ax) / (6 - s2) + (s2 < 5 ? (R() - 0.5) * 10 : 0);
            const ny = ay + (by - ay) / (6 - s2) + (s2 < 5 ? (R() - 0.5) * 10 : 0);
            const m = Math.ceil(Math.max(Math.abs(nx - ax), Math.abs(ny - ay)));
            for (let k = 0; k <= m; k++) ctx.fillRect(Math.round(ax + ((nx - ax) * k) / m - cx), Math.round(ay + ((ny - ay) * k) / m - cy), 1, 1);
            ax = nx;
            ay = ny;
          }
        }
      }
      ctx.globalCompositeOperation = "source-over";

      // Muzzle flashes
      for (const f of g.flashes) {
        const x = Math.round(f.x - cx);
        const y = Math.round(f.y - cy);
        ctx.fillStyle = "#fff6c8";
        ctx.fillRect(x - 1, y - 1, 3, 3);
        ctx.fillStyle = f.c;
        ctx.fillRect(x - 2, y, 5, 1);
        ctx.fillRect(x, y - 2, 1, 5);
      }

      // Shots
      for (const s of g.shots) {
        const x = Math.round(s.x - cx);
        const y = Math.round(s.y - cy);
        const d = Math.hypot(s.vx, s.vy) || 1;
        const ux = s.vx / d;
        const uy = s.vy / d;
        switch (s.kind) {
          case BULLET:
          case ENEMY_BULLET:
            ctx.fillStyle = s.kind === BULLET ? "#ffe27a" : "#ff5a3a";
            ctx.fillRect(x, y, 1, 1);
            ctx.fillRect(Math.round(x - ux * 2), Math.round(y - uy * 2), 1, 1);
            ctx.fillStyle = s.kind === BULLET ? "#ffb020" : "#b8321a";
            ctx.fillRect(Math.round(x - ux * 4), Math.round(y - uy * 4), 1, 1);
            break;
          case SNIPE:
            ctx.fillStyle = "#ffffff";
            for (let k = 0; k < 10; k++) ctx.fillRect(Math.round(x - ux * k), Math.round(y - uy * k), 1, 1);
            break;
          case PELLET:
            ctx.fillStyle = "#ffd27a";
            ctx.fillRect(x, y, 1, 1);
            break;
          case PLASMA:
            glow(x - 1, y - 1, 3, 3, "#c77dff", 0.9);
            ctx.fillStyle = "#f4e0ff";
            ctx.fillRect(x, y, 1, 1);
            break;
          case PULSE: {
            // A searing bolt in its firer's colour, a white-hot head
            const col = PLAYER_COLORS[s.pid] ?? "#ff5a6a";
            ctx.globalAlpha = 0.45;
            ctx.fillStyle = col;
            for (let k = 0; k < 8; k++) ctx.fillRect(Math.round(x - ux * k), Math.round(y - uy * k) - 1, 1, 3);
            ctx.globalAlpha = 1;
            for (let k = 0; k < 6; k++) ctx.fillRect(Math.round(x - ux * k), Math.round(y - uy * k), 1, 1);
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(x, y, 1, 1);
            ctx.fillRect(Math.round(x - ux), Math.round(y - uy), 1, 1);
            break;
          }
          case FLAMESHOT:
            break;
          case BIGBOMB:
            // A fat bomb, fins at the back, a red band, nose down as it falls
            ctx.fillStyle = INK;
            ctx.fillRect(x - 3, y - 7, 7, 13);
            ctx.fillStyle = "#4a5030";
            ctx.fillRect(x - 2, y - 6, 5, 11);
            ctx.fillStyle = "#6a7048";
            ctx.fillRect(x - 2, y - 6, 1, 11);
            ctx.fillStyle = "#c8302a";
            ctx.fillRect(x - 2, y - 2, 5, 1);
            ctx.fillStyle = INK;
            ctx.fillRect(x - 4, y - 9, 9, 2);
            break;
          case NAPALM:
            ctx.fillStyle = INK;
            ctx.fillRect(x - 2, y - 1, 5, 3);
            ctx.fillStyle = "#c8302a";
            ctx.fillRect(x - 1, y, 3, 1);
            if (Math.floor(t * 20) % 2) glow(x + 2, y, 1, 1, "#ff8a1a", 1);
            break;
          case FLAK:
            ctx.fillStyle = "#ffe27a";
            ctx.fillRect(x, y, 1, 1);
            break;
          case BOLT:
            ctx.fillStyle = "#c8a060";
            for (let k = 0; k < 5; k++) ctx.fillRect(Math.round(x - ux * k), Math.round(y - uy * k), 1, 1);
            ctx.fillStyle = "#ff3b3b";
            ctx.fillRect(x, y, 1, 1);
            break;
          case SAW: {
            // A spinning blade
            const a = t * 30;
            ctx.fillStyle = INK;
            disc(ctx, x, y, 3);
            ctx.fillStyle = "#c8ced9";
            disc(ctx, x, y, 2);
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(Math.round(x + Math.cos(a) * 2), Math.round(y + Math.sin(a) * 2), 1, 1);
            ctx.fillRect(Math.round(x - Math.cos(a) * 2), Math.round(y - Math.sin(a) * 2), 1, 1);
            break;
          }
          case ELASER:
            glow(Math.round(x - ux * 3), Math.round(y - uy * 3), 2, 1, "#5ff0e0", 1);
            ctx.fillStyle = "#e6fffc";
            for (let k = 0; k < 7; k++) ctx.fillRect(Math.round(x - ux * k), Math.round(y - uy * k), 1, 1);
            break;
          case ROCKET:
            ctx.fillStyle = "#c8ced9";
            ctx.fillRect(Math.round(x - ux * 2) - 1, y, 3, 1);
            glow(Math.round(x - ux * 4), y, 1, 1, "#ff9a2a", 1);
            break;
          case GRENADE:
          case LOB:
            // (an enemy's lobs are lit sticks of dynamite)
            ctx.fillStyle = s.kind === LOB ? (s.pid === -1 ? "#c8302a" : "#4a5a30") : "#3a5a2a";
            ctx.fillRect(x - 1, y - 1, 2, 2);
            if (Math.floor(t * 12) % 2) {
              ctx.fillStyle = "#ff3b3b";
              ctx.fillRect(x, y - 1, 1, 1);
            }
            break;
          case SHELL:
            ctx.fillStyle = "#ff9a2a";
            ctx.fillRect(x, y - 8, 1, 4);
            ctx.fillStyle = "#fff2c0";
            ctx.fillRect(x, y - 4, 1, 4);
            break;
          case CANNON:
          case MORTAR:
            ctx.fillStyle = INK;
            ctx.fillRect(x - 1, y - 1, 3, 3);
            ctx.fillStyle = "#c8c8c0";
            ctx.fillRect(x, y, 1, 1);
            break;
          case METEOR:
          case FIREBALL:
            ctx.fillStyle = s.kind === FIREBALL ? "#8aff6a" : "#ff5a1a";
            disc(ctx, x, y, 3);
            ctx.fillStyle = s.kind === FIREBALL ? "#e0ffd0" : "#ffe27a";
            disc(ctx, x, y, 1.5);
            break;
        }
      }
      for (const m of g.missiles) {
        const dx = m.up ? 0 : m.tx - m.x;
        const dy = m.up ? -1 : m.ty - m.y;
        const d = Math.hypot(dx, dy) || 1;
        glow(Math.round(m.x - cx - (dx / d) * 24) - 1, Math.round(m.y - cy - (dy / d) * 24) - 1, 3, 3, "#ffe27a", 1);
      }
      // (the submarines' missiles only light up once they're out of the water)
      for (const s of g.sams) {
        if (s.wet) continue;
        const sp = Math.hypot(s.vx, s.vy) || 1;
        glow(Math.round(s.x - cx - (s.vx / sp) * 24) - 1, Math.round(s.y - cy - (s.vy / sp) * 24) - 1, 3, 3, "#ffe27a", 1);
      }

      // Fire: each blast's light on everything round it, its flash, rays and shock rings, its
      // fireball, then flames, embers, glints and weather; all in the blast's own colours
      for (const b of g.blasts) {
        const a = b.t / 0.5;
        const x = Math.round(b.x - cx);
        const y = Math.round(b.y - cy);
        const P = FIRE_PALS[b.pal] ?? FIRE_PALS[0];
        if (b.t < 0.2) {
          // (a soft glow, brightest at the middle: no more than a screen-filling wash for the biggest)
          ctx.globalCompositeOperation = "lighter";
          ctx.fillStyle = P.glow;
          const f = 1 - b.t / 0.2;
          ctx.globalAlpha = Math.round(f * 0.16 * 8) / 8;
          disc(ctx, x, y, Math.round(Math.min(90, b.r * 1.8)));
          ctx.globalAlpha = Math.round(f * 0.22 * 8) / 8;
          disc(ctx, x, y, Math.round(Math.min(56, b.r * 1.1)));
          ctx.globalAlpha = 1;
          ctx.globalCompositeOperation = "source-over";
        }
        if (b.r >= 22 && b.t < 0.16) rays(x, y, b.r * 2.6, P.flash, b.x + b.y, (1 - b.t / 0.16) * 0.8);
        if (b.t < 0.06) {
          ctx.fillStyle = b.pal ? P.core : "#fffbe8";
          disc(ctx, x, y, Math.round(b.r * 0.85));
          continue;
        }
        if (a < 0.3) {
          ctx.globalAlpha = 1 - a / 0.3;
          ctx.fillStyle = P.flash;
          ring(ctx, x, y, Math.round(b.r * (0.9 + 2.2 * a)));
          // (and a fainter one racing out ahead of it, in the fire's colour)
          if (b.r >= 12) {
            ctx.globalAlpha = (1 - a / 0.3) * 0.6;
            ctx.fillStyle = P.glow;
            ring(ctx, x, y, Math.round(b.r * (1.2 + 4 * a)));
          }
          ctx.globalAlpha = 1;
        }
      }
      drawFireballs(g, cx, cy);
      // Mushroom clouds, from the fireball off the ground to the smoke they leave hanging
      for (const m of g.mushrooms) drawMushroom(m, cx, cy);
      for (const w of g.waves) {
        ctx.fillStyle = `rgba(255,250,230,${(1 - w.t) * 0.8})`;
        ring(ctx, Math.round(w.x - cx), Math.round(w.y - cy), Math.round(w.t * 420));
        ring(ctx, Math.round(w.x - cx), Math.round(w.y - cy), Math.round(w.t * 420) + 1);
      }
      for (const p of g.parts) {
        if (p.k === DEBRIS || p.k === CASING || p.k === SMOKE || p.k === BIGSMOKE || p.k === MIST || p.k === DUST || p.k === BLAZE || p.k === BUBBLE) continue;
        const a = p.life / p.max;
        const x = Math.round(p.x - cx);
        const y = Math.round(p.y - cy);
        switch (p.k) {
          case FLAME:
            ctx.fillStyle = a > 0.6 ? "#fff3b0" : a > 0.35 ? "#ffb020" : "#ff5a1a";
            ctx.fillRect(x, y, p.s, p.s);
            break;
          case EMBER:
            ctx.fillStyle = a > 0.6 ? "#ffe27a" : a > 0.3 ? "#ff9a2a" : "#b8321a";
            ctx.fillRect(x, y, 1, 1);
            break;
          case SPARK: {
            // A streak along its flight, white hot at first
            const d = Math.hypot(p.vx, p.vy) || 1;
            ctx.fillStyle = p.c;
            if (d > 50) ctx.fillRect(Math.round(x - (p.vx / d) * 2), Math.round(y - (p.vy / d) * 2), 1, 1);
            ctx.fillStyle = a > 0.7 ? "#fff6c8" : p.c;
            ctx.fillRect(x, y, 1, 1);
            break;
          }
          case CINDER:
            glow(x, y, p.s, p.s, a > 0.55 ? p.c || "#ffd23f" : a > 0.25 ? "#ff7a1a" : "#b8321a", 1);
            break;
          case GLINT:
            // A sparkle, twinkling: a little star when it's bright
            if (Math.sin(t * 22 + p.x * 3 + p.y) < -0.3) break;
            ctx.fillStyle = p.c;
            if (p.s > 1 && a > 0.35) {
              ctx.fillRect(x - 1, y, 3, 1);
              ctx.fillRect(x, y - 1, 1, 3);
              ctx.fillStyle = "#ffffff";
            }
            ctx.fillRect(x, y, 1, 1);
            break;
          case FIREFLY:
          case SPORE:
            if (Math.sin(t * 5 + p.x) > -0.2) glow(x, y, 1, 1, p.c, 0.9);
            break;
          case RAIN:
            ctx.fillStyle = p.c;
            ctx.globalAlpha = 0.6;
            ctx.fillRect(x, y, 1, 3);
            ctx.globalAlpha = 1;
            break;
          case SNOWFLAKE:
          case ASHFALL:
          case SANDGRAIN:
            ctx.fillStyle = p.c;
            ctx.fillRect(x, y, p.s, p.s);
            break;
          default:
            ctx.fillStyle = p.c;
            ctx.fillRect(x, y, 1, 1);
        }
      }

      // The squad's helicopter flies over as a level starts, dropping everyone in
      if (g.levelT < 4.5) {
        const hx = Math.round(-40 + (g.levelT / 4) * (W + 80));
        drawHeli(ctx, hx, 28 + Math.round(Math.sin(t * 2) * 2), t, false);
      }

      // Player markers; the hovered player gets a beam and its tag
      const hi = sceneState.highlight;
      const pulseAge = performance.now() / 1000 - sceneState.pulse;
      const tags = pulseAge >= 0 && pulseAge < 0.8;
      for (const p of g.players) {
        if (p.dead) continue;
        const x = Math.round(p.x - cx);
        const y = Math.round(p.y - cy) - (p.chute ? 38 : 22);
        const col = PLAYER_COLORS[p.id];
        if (hi === p.id) {
          ctx.globalCompositeOperation = "lighter";
          ctx.globalAlpha = 0.12;
          ctx.fillStyle = col;
          ctx.fillRect(x - 4, 0, 9, y);
          ctx.globalAlpha = 0.3;
          ctx.fillRect(x - 1, 0, 3, y);
          ctx.globalAlpha = 1;
          ctx.globalCompositeOperation = "source-over";
        }
        const bob = Math.floor(t * 4) % 2;
        ctx.fillStyle = INK;
        ctx.fillRect(x - 2, y - 1 + bob, 5, 3);
        ctx.fillStyle = col;
        ctx.fillRect(x - 1, y + bob, 3, 1);
        ctx.fillRect(x, y + 1 + bob, 1, 1);
        if (hi === p.id || tags) pixelText(ctx, `P${p.id + 1}`, x - 3, y - 7 + bob, col);
      }
      for (const p of g.popups) {
        const w = p.text.length * 4 - 1;
        ctx.globalAlpha = p.t > 0.7 ? 1 - (p.t - 0.7) / 0.3 : 1;
        pixelText(ctx, p.text, Math.round(p.x - cx - w / 2), Math.round(p.y - cy), p.c);
        ctx.globalAlpha = 1;
      }

      // A health bar floating over the boss, with its name
      for (const e of g.enemies) {
        if (!e.boss || !e.alive) continue;
        const aim = g.aimPoint(e) ?? { x: e.x, y: e.y };
        const top =
          e.kind === "worm"
            ? aim.y - 22
            : e.kind === "gunship"
              ? e.y - 26
              : e.kind === "airship"
                ? e.y - 30
                : e.y - (e.kind === "mech" ? 74 : e.kind === "spider" ? 60 : e.kind === "cannon" ? 52 : 44);
        const bw = 60;
        const bx = Math.round(aim.x - cx - bw / 2);
        const by = Math.round(top - cy);
        const name = BOSS_NAMES[e.kind as keyof typeof BOSS_NAMES];
        pixelText(ctx, name, Math.round(aim.x - cx - (name.length * 4) / 2), by - 8, "#ff3b3b");
        ctx.fillStyle = INK;
        ctx.fillRect(bx - 1, by - 1, bw + 2, 6);
        ctx.fillStyle = "#3a1414";
        ctx.fillRect(bx, by, bw, 4);
        ctx.fillStyle = "#ff3b3b";
        ctx.fillRect(bx, by, Math.round(bw * clamp01(e.hp / e.maxHp)), 4);
        ctx.fillStyle = "#ff8a6a";
        ctx.fillRect(bx, by, Math.round(bw * clamp01(e.hp / e.maxHp)), 1);
      }

      // Level banners ("Level 3 · Desert outpost · Boss ahead: Walker", "Warning!", "Missiles inbound")
      const b = g.banner;
      if (b && b.t >= 0 && (b.color !== "#ff3b3b" || Math.floor(b.t * 4) % 2 === 0 || b.t > 1.6)) {
        const s = W < 260 ? 2 : 3;
        const w = (b.text.length * 4 - 1) * s;
        // (just above the squad, where the page leaves the scene open)
        const y = Math.round(H * horizon - 26 * s - 60);
        pixelText(ctx, b.text, Math.round((W - w) / 2), y, b.color, true, s);
        if (b.sub) {
          const sw = b.sub.length * 4 - 1;
          pixelText(ctx, b.sub, Math.round((W - sw) / 2), y + 6 * s + 4, "#ffffff");
        }
        if (b.info) {
          const iw = b.info.length * 4 - 1;
          pixelText(ctx, b.info, Math.round((W - iw) / 2), y + 6 * s + 12, "#9ab4d8");
        }
      }

      // White-out: each ICBM's flash; then at the end of the level the whole screen goes white,
      // "Level complete" comes up on it, and it fades away over the next level
      if (!reduced) {
        const white = Math.max(Math.min(0.9, g.flash), g.white);
        if (white > 0.01) {
          ctx.fillStyle = `rgba(255,250,235,${Math.min(1, white)})`;
          ctx.fillRect(0, 0, W, H);
        }
        if (g.complete && g.white > 0.9) drawComplete(g);
      }
    };

    /** The "Level complete" card, dark pixel type on the white. */
    const drawComplete = (g: Game) => {
      const c = g.complete!;
      ctx.globalAlpha = clamp01((g.white - 0.9) / 0.1);
      const title = c.title ?? "Level complete";
      const s = Math.max(2, Math.min(7, Math.floor((W * 0.84) / (Math.max(14, title.length) * 4))));
      const tw = (title.length * 4 - 1) * s;
      const top = Math.round(H * 0.34 - 3 * s);
      pixelText(ctx, title, Math.round((W - tw) / 2), top, "#1a1626", false, s);
      // A bar under it in the eight players' colours
      const bx = Math.round((W - tw) / 2);
      PLAYER_COLORS.forEach((col, i) => {
        ctx.fillStyle = col;
        ctx.fillRect(bx + Math.round((tw * i) / 8), top + 6 * s + 2, Math.ceil(tw / 8), Math.max(2, s >> 1));
      });
      const s2 = s >= 5 ? 2 : 1;
      let y = top + 6 * s + 12;
      const lines = [`Level ${String(c.no).padStart(2, "0")} - ${c.name}`, `Enemies down ${c.kills}`, `Time ${clock(c.time)}`];
      if (c.finale) lines.push(FINALE_TEXT[c.finale][2]);
      lines.push(`Squad bonus +${c.bonus ?? 5000} each`);
      for (const line of lines) {
        pixelText(ctx, line, Math.round((W - (line.length * 4 - 1) * s2) / 2), y, "#3a3450", false, s2);
        y += 7 * s2 + 3;
      }
      if (Math.floor(g.t * 2) % 2 === 0) {
        const next = "Next level incoming";
        pixelText(ctx, next, Math.round((W - (next.length * 4 - 1) * s2) / 2), y + 4, "#c8302a", false, s2);
      }
      ctx.globalAlpha = 1;
    };

    /** Lightning: the sky lights up for a moment and a bolt forks down behind the hills. */
    const drawBolts = (g: Game, cx: number, cy: number) => {
      for (const b of g.bolts) {
        const a = 0.24 * (1 - b.t / 0.35);
        if (a > 0.01) {
          ctx.fillStyle = `rgba(225,232,255,${a})`;
          ctx.fillRect(0, 0, W, H);
        }
        if (b.t > 0.22 || (b.t > 0.08 && b.t < 0.12)) continue;
        const R = rng(b.seed);
        let x = Math.round(b.x - cx);
        let y = 0;
        const end = farBase(cy, 0) + 10;
        ctx.fillStyle = "#f4f0ff";
        while (y < end) {
          const nx = x + Math.round((R() - 0.5) * 8);
          const ny = y + 4 + Math.floor(R() * 6);
          const n = Math.max(Math.abs(nx - x), ny - y);
          for (let k = 0; k <= n; k++) ctx.fillRect(Math.round(x + ((nx - x) * k) / n), Math.round(y + ((ny - y) * k) / n), 1, 1);
          x = nx;
          y = ny;
        }
      }
    };

    const drawSky = (g: Game, cx: number, cy: number) => {
      const t = g.t;
      regenSky();
      ctx.drawImage(sky, 0, 0);
      const night = nightNow();
      const hz = H * horizon;
      if (night > 0.25 && theme.backdrop !== "cave") {
        ctx.fillStyle = "#ffffff";
        stars.forEach((s, i) => {
          const tw = s.b > 0.7 ? 0.55 + 0.45 * Math.sin(t * (1 + s.b * 3) + i) : 1;
          ctx.globalAlpha = Math.min(1, (night - 0.25) * 2 * (0.35 + 0.65 * s.b) * tw);
          ctx.fillRect(s.x, s.y, 1, 1);
        });
        ctx.globalAlpha = 1;
      }
      if (theme.cycle) {
        // The sun across the sky by day, the moon by night (hidden behind the cloud)
        const a = g.clock * Math.PI * 2;
        const cover = g.sky === "clear" ? 1 : g.sky === "cloud" ? 0.45 : 0.15;
        for (const [ang, moon] of [[a, false], [a + Math.PI, true]] as const) {
          const x = Math.round(W * (0.5 + 0.42 * Math.sin(ang)));
          const y = Math.round(hz * (0.12 + (0.92 * (1 - Math.cos(ang))) / 2));
          if (y > hz + 10) continue;
          ctx.globalAlpha = cover;
          if (moon) {
            ctx.fillStyle = "#e9edf6";
            disc(ctx, x, y, 8);
            ctx.fillStyle = "#c3c9d8";
            disc(ctx, x + 2, y - 2, 1.5);
            disc(ctx, x - 3, y + 2, 1);
          } else {
            const low = clamp01((y / hz - 0.45) / 0.55);
            ctx.globalAlpha = 0.35 * cover;
            ctx.fillStyle = mix("#ffe08a", "#ff5a2a", low);
            disc(ctx, x, y, 17);
            ctx.globalAlpha = cover;
            disc(ctx, x, y, 12);
            ctx.fillStyle = mix("#fffbe6", "#ffb060", low);
            disc(ctx, x, y, 9);
          }
          ctx.globalAlpha = 1;
        }
      } else if (theme.sun === "sun" || theme.sun === "red") {
        const red = theme.sun === "red" ? 1 : 0;
        const x = Math.round(W * 0.74);
        const y = Math.round(hz * (0.4 + red * 0.22));
        ctx.globalAlpha = 0.35;
        ctx.fillStyle = mix("#ffe08a", "#ff5a2a", red);
        disc(ctx, x, y, 17);
        ctx.globalAlpha = 1;
        disc(ctx, x, y, 12);
        ctx.fillStyle = mix("#fffbe6", "#ffb060", red);
        disc(ctx, x, y, 9);
      } else if (theme.sun === "moon") {
        const x = Math.round(W * 0.2);
        const y = Math.round(hz * 0.22);
        ctx.fillStyle = "#e9edf6";
        disc(ctx, x, y, 8);
        ctx.fillStyle = "#c3c9d8";
        disc(ctx, x + 2, y - 2, 1.5);
        disc(ctx, x - 3, y + 2, 1);
      }
      if (!reduced) drawBolts(g, cx, cy);
      // The far-off battle, beyond the mountains (they're drawn over it)
      drawFar(ctx, 0, cx, cy, t);
    };

    /* ------------------------------------------------------------ */
    /* Loop                                                          */
    /* ------------------------------------------------------------ */
    let raf = 0;
    let running = !document.hidden;
    let lastFrame = performance.now();
    let lastInput = performance.now();
    let lastBurst = sceneState.burst;
    let holdT = 0;
    let strikeT = 0;
    /** The paused frame is on screen (a resize redraws it) */
    let held = false;
    const onInput = () => (lastInput = performance.now());

    const step = (g: Game, dt: number) => {
      // A different page (project, 404, back home): on to a new level behind the page wipe
      if (sceneState.override !== lastOverride) {
        lastOverride = sceneState.override;
        g.nextLevel(sceneState.override === "lost" ? THEMES.cave : undefined);
      }
      g.intensity = intensity();
      g.endless = sceneState.endless;
      if (sceneState.zoom !== zoomed || sceneState.scaleK !== scaleK) resize();
      const hz = horizonFor();
      if (hz !== horizon) {
        // (the camera eases over to the new framing; the sky's bands follow it)
        horizon = hz;
        skyKey = "";
      }

      // Visitor input: a click blasts the spot, holding on the hero brings shells down
      if (sceneState.burst !== lastBurst) {
        lastBurst = sceneState.burst;
        const p = pointerWorld();
        g.strike(p.x, p.y);
      }
      if (sceneState.press) {
        holdT += dt;
        strikeT -= dt;
        if (holdT > 0.25 && strikeT <= 0) {
          strikeT = 0.08 + rand() * 0.07;
          g.dropShell(camX + 10 + rand() * (W - 20), viewRect());
        }
      } else holdT = 0;

      g.step(dt, viewRect(), reduced);

      if (g.reset) {
        // A new level (it fades in from the white): redraw every block, jump the camera, a new battle in the distance
        g.reset = false;
        prevTheme = g.theme;
        theme = g.theme;
        fade = 1;
        redrawAll(g);
        g.fresh.length = 0;
        g.dirty.length = 0;
        snap = true;
        farFx.length = 0;
      }
      if (theme !== g.theme) {
        prevTheme = theme;
        theme = g.theme;
        fade = 0;
      }
      fade = Math.min(1, fade + dt / 1.2);
      if (!reduced) {
        spawnFar(dt, camX);
        updateFar(dt, camX);
      }

      const tg = cameraTarget(g);
      if (snap || reduced) {
        camX = tg.x;
        camY = tg.y;
        snap = false;
      } else {
        // (during the nuke the camera holds still to watch)
        camX += (tg.x - camX) * damp(dt, g.phase === "nuke" ? 1 : 3);
        camY += (tg.y - camY) * damp(dt, 2);
      }
      if (!reduced) shake = Math.max(shake, g.shake);
      g.shake = 0;
      shake *= Math.exp(-dt * 8);

      for (const c of g.fresh) {
        drawColumn(c);
        drawColumn(c - 1);
      }
      g.fresh.length = 0;
      const d = g.dirty;
      for (let i = 0; i < d.length; i += 2) drawCell(d[i], d[i + 1]);
      d.length = 0;

      sceneState.cam.x = Math.round(camX);
      sceneState.cam.y = Math.round(camY);
      sceneState.cam.z = scale;
      sceneState.kills = g.kills;
      sceneState.booms = g.booms;
      sceneState.level = g.levelNo;
      sceneState.stage = g.theme.name;
      sceneState.white = reduced ? 0 : g.white;
      const intel = sceneState.intel;
      intel.progress = g.progress;
      intel.objective = g.objective;
      intel.boss = g.bossKind ? BOSS_NAMES[g.bossKind] : "";
      intel.bossState = g.bossState;
      intel.levelKills = g.levelKills;
      intel.levelTime = g.levelT;
      intel.cleared = g.cleared;
      intel.log = g.log;
      intel.uptime = g.t;
      intel.weather = g.theme.cycle ? { clear: "Clear", cloud: "Overcast", rain: "Rain", storm: "Thunderstorm", fog: "Fog" }[g.sky] : WEATHER[g.theme.ambient];
      intel.light = lightOf(g.theme);
      intel.wave = WAVES[g.wave].name;
      let hostiles = 0;
      for (const e of g.enemies) if (e.alive && e.x > camX && e.x < camX + W) hostiles++;
      intel.hostiles = hostiles;
      g.players.forEach((p, i) => {
        const s = sceneState.players[i];
        s.score = p.score;
        s.hp = p.dead ? 0 : p.hp / PLAYER_HP;
        s.alive = !p.dead;
      });
      sceneState.velocity *= Math.exp(-dt * 4);
    };

    const draw = (g: Game) => {
      const sx = shake > 0.4 ? Math.round((rand() - 0.5) * 2 * shake) : 0;
      const sy = shake > 0.4 ? Math.round((rand() - 0.5) * 2 * shake) : 0;
      const cx = Math.round(camX) + sx;
      const cy = Math.round(camY) + sy;
      drawSky(g, cx, cy);
      drawWorld(g, cx, cy, g.t);
      ctx.drawImage(view, 0, 0);
      // The weather over the streets: fog thick in them, the gloom of the rain
      const cyc = theme.cycle ? g.sky : "clear";
      fogA += ((cyc === "fog" ? 1 : 0) - fogA) * 0.02;
      wetA += ((cyc === "rain" ? 0.6 : cyc === "storm" ? 1 : cyc === "cloud" ? 0.25 : 0) - wetA) * 0.02;
      if (fogA > 0.02) {
        ctx.fillStyle = `rgba(196,202,212,${(0.3 * fogA).toFixed(3)})`;
        ctx.fillRect(0, 0, W, H);
        ctx.fillStyle = `rgba(214,218,226,${(0.18 * fogA).toFixed(3)})`;
        ctx.fillRect(0, Math.round(H * 0.55), W, H);
      }
      if (wetA > 0.02) {
        ctx.fillStyle = `rgba(20,28,46,${(0.16 * wetA).toFixed(3)})`;
        ctx.fillRect(0, 0, W, H);
      }
      drawLights(g, cx, cy, g.t);
    };

    /**
     * The next frame. Capped (the wallpaper), it's only asked for when it's due, so the browser
     * isn't woken for every one the display could show (120 a second on some) to skip most of them.
     */
    let timer = 0;
    const schedule = () => {
      if (sceneState.maxFps > 0) timer = window.setTimeout(() => (raf = requestAnimationFrame(loop)), Math.max(0, 1000 / sceneState.maxFps - 6));
      else raf = requestAnimationFrame(loop);
    };
    const loop = (ms: number) => {
      schedule();
      if (!running || sceneState.covered) {
        lastFrame = ms;
        return;
      }
      if (!game) start();
      // (someone in the spectator space is watching: keep the full frame rate however long they sit there)
      const busy = ms - lastInput < IDLE_MS || sceneState.press || sceneState.transition || sceneState.watching;
      if (!busy && ms - lastFrame < 30) return;
      const dt = Math.min(0.05, Math.max(0, (ms - lastFrame) / 1000));
      lastFrame = ms;
      // Paused: the frame holds still (drawn once). Faster or slower: the same frame time, in more or
      // fewer game seconds, in steps small enough to stay stable
      if (sceneState.paused) {
        if (held) return;
        held = true;
        shake = 0;
      } else {
        held = false;
        const total = dt * sceneState.speed * PACE;
        const n = Math.max(1, Math.ceil(total / 0.034));
        for (let k = 0; k < n; k++) step(game!, total / n);
      }
      draw(game!);
      sceneState.frames++;
      sceneState.ready = true;
    };

    const onVisibility = () => {
      running = !document.hidden;
      lastFrame = performance.now();
    };

    resize();
    window.addEventListener("resize", resize);
    document.addEventListener("visibilitychange", onVisibility);
    for (const ev of INPUT_EVENTS) window.addEventListener(ev, onInput, { passive: true });
    raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      window.clearTimeout(timer);
      window.removeEventListener("resize", resize);
      document.removeEventListener("visibilitychange", onVisibility);
      for (const ev of INPUT_EVENTS) window.removeEventListener(ev, onInput);
    };
  }, []);

  return (
    <div className="pixel-stage" aria-hidden="true">
      <canvas ref={canvasRef} className="pixel-canvas" />
    </div>
  );
}
