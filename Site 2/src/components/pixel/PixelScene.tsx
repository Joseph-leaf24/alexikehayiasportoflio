import { useEffect, useRef } from "react";
import { sceneState, type LayoutKey } from "@/lib/store";
import { prefersReducedMotion } from "@/lib/smoothScroll";
import { getQuality } from "@/lib/device";
import { PLAYER_COLORS } from "@/data/squad";
import { resolveBlend } from "./blend";
import { AIR, cell, CRYSTAL, kindAt, LAVA, LEVEL_H, MASK, RING, ROWS, TILE, WATER, WINDOW } from "./level";
import { lightOf, THEMES, WEATHER, type Backdrop, type Theme } from "./themes";
import { rng } from "./rng";
import {
  ASHFALL,
  BIGSMOKE,
  BOSS_NAMES,
  BULLET,
  CANNON,
  CASING,
  DEBRIS,
  EMBER,
  ENEMY_BULLET,
  FIREBALL,
  FIREFLY,
  FLAME,
  FLAMESHOT,
  Game,
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
  SANDGRAIN,
  SHELL,
  SMOKE,
  SNIPE,
  SNOWFLAKE,
  SPORE,
  startTheme,
  type Enemy,
  type View,
} from "./game";
import {
  BAYER,
  DEBRIS_COLOR,
  disc,
  drawBoat,
  drawDrone,
  drawGunship,
  drawHeli,
  drawIcbm,
  drawJeep,
  drawJet,
  drawMech,
  drawTank,
  drawTanker,
  drawTruck,
  drawWorm,
  FRAMES,
  INK,
  makeAtlas,
  makeBackdrop,
  makeCanvas,
  makeCloud,
  makeProps,
  makeSquad,
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
/** Frame-rate independent smoothing factor: `x += (target - x) * damp(dt, rate)` */
const damp = (dt: number, rate: number) => 1 - Math.exp(-dt * rate);
const mod = (a: number, n: number) => ((a % n) + n) % n;
/** The screen is about this many scene pixels tall, whatever the device: a wide view of the battle. */
const TARGET_ROWS = 340;
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
    const ctx = canvas?.getContext("2d");
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
    const squad = makeSquad(PLAYER_COLORS);
    const props = makeProps();
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
    const drawCell = (c: number, r: number) => {
      const L = game!.level;
      if (r < 0 || r >= ROWS || c < L.start || c >= L.next) return;
      const i = cell(c, r);
      const x = (c & MASK) * TILE;
      const y = r * TILE;
      tctx.clearRect(x, y, TILE, TILE);
      const k = L.fg[i];
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
      scale = Math.max(2, Math.round((vh * dpr) / TARGET_ROWS));
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
    };

    /* ------------------------------------------------------------ */
    /* Sky (follows the level's theme, crossfading when it changes)  */
    /* ------------------------------------------------------------ */
    let theme: Theme = THEMES.jungle;
    let prevTheme: Theme = theme;
    let fade = 1;
    const lerpTheme = (f: (s: Theme) => number) => f(prevTheme) + (f(theme) - f(prevTheme)) * fade;

    const regenSky = () => {
      const k = `${W}x${H}:${prevTheme.key}>${theme.key}:${fade.toFixed(2)}`;
      if (k === skyKey) return;
      skyKey = k;
      const stops = [0, 1, 2].map((i) => rgb(mix(prevTheme.sky[i], theme.sky[i], fade)));
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
      const alive = g.players.filter((p) => !p.dead && !p.chute);
      const feet = alive.length ? alive.reduce((s, p) => s + p.y, 0) / alive.length : g.level.h * TILE;
      // (never left of the oldest stored column: there's nothing to see there)
      return { x: Math.max(g.level.start * TILE, g.advance - W * 0.42), y: Math.max(-80, Math.min(LEVEL_H - H, feet - H * horizon)) };
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
      const g = new Game(first, { seed: 1 + Math.floor(rand() * 100000), maxEnemies: q.enemies, maxParticles: q.particles, colors: PLAYER_COLORS });
      g.intensity = intensity();
      game = g;
      theme = prevTheme = first;
      fade = 1;
      warm(g, WARM_UP);
    };

    /* ------------------------------------------------------------ */
    /* Drawing                                                       */
    /* ------------------------------------------------------------ */
    const parallaxBase = (cy: number, f: number) => Math.round(H * horizon - (cy - (44 * TILE - H * horizon)) * f);
    const motion = reduced ? 0 : 1;

    /* ------------------------------------------------------------ */
    /* The battle in the distance                                    */
    /* ------------------------------------------------------------ */
    const farFx: FarFx[] = [];
    /** Each plane's horizon line on screen: about where the far mountains' and the mid hills' ridges run. */
    const farBase = (cy: number, depth: number) => (depth ? parallaxBase(cy, 0.15) - 46 : parallaxBase(cy, 0.08) - 56);
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
        case "sniper":
          return art.sniper[0][face];
        case "crawler":
          return art.crawler[Math.floor(e.anim) % 2][face];
        case "dog":
          return art.dog[Math.floor(e.anim) % 2][face];
        default: {
          const set =
            e.kind === "rocket" ? art.rocket : e.kind === "flamer" ? art.flamer : e.kind === "grenadier" ? art.grenadier : e.kind === "shield" ? art.shield : art.mook;
          return set[e.vx ? FRAMES.run[Math.floor(e.anim) % 4] : e.ground ? FRAMES.idle : FRAMES.jump][face];
        }
      }
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
        const below = img === bd.near ? H - (y + img.height) : 0;
        for (let x = -off; x < W; x += img.width) {
          v.drawImage(img, x, y);
          if (below > 0) v.drawImage(img, 0, img.height - 1, img.width, 1, x, y + img.height, img.width, below);
        }
      };
      layer(bd.far, 0.08, 20);
      drawFar(v, 1, cx, cy, t);
      layer(bd.mid, 0.2, 24);
      layer(bd.near, 0.4, 30);
      // Jets crossing the sky, ICBMs coming in
      for (const j of g.jets) {
        const x = Math.round(j.x - cx);
        const y = Math.round(j.y - cy);
        v.globalAlpha = 0.25;
        v.fillStyle = "#e8eef6";
        v.fillRect(x + 21, y + 1, 60, 1);
        v.globalAlpha = 1;
        drawJet(v, x, y);
      }
      for (const m of g.missiles) {
        const dx = m.up ? 0 : m.tx - m.x;
        const dy = m.up ? -1 : m.ty - m.y;
        const d = Math.hypot(dx, dy) || 1;
        drawIcbm(v, Math.round(m.x - cx), Math.round(m.y - cy), dx / d, dy / d);
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
      for (const p of g.level.props) {
        if (p.x < cx - 50 || p.x > cx + W + 50) continue;
        const x = Math.round(p.x - cx);
        const y = Math.round(p.y - cy);
        switch (p.kind) {
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
        if (b.fuse >= 0 && Math.floor(t * 20) % 2) {
          v.fillStyle = "#ffffff";
          v.fillRect(x, y - img.height + 2, 1, 1);
        }
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
        if (e.kind === "boat") {
          drawBoat(v, x, y, t, e.look, e.facing, e.alive ? 0 : clamp01(1 - e.deadT / 2));
          continue;
        }
        if (e.kind === "drone") {
          drawDrone(v, x, y, t, hit);
          continue;
        }
        const img = troopSprite(e);
        if (!e.alive) {
          v.globalAlpha = clamp01(e.deadT / 0.6);
          v.save();
          v.translate(x, y - 6);
          v.rotate((e.ground ? -e.facing : Math.floor(e.spin) % 4) * (Math.PI / 2));
          v.drawImage(img, -Math.floor(img.width / 2), -Math.floor(img.height / 2));
          v.restore();
          v.globalAlpha = 1;
          continue;
        }
        const face = e.facing > 0 ? 0 : 1;
        const ox = e.kind === "turret" || e.kind === "crawler" || e.kind === "dog" || e.kind === "mg" || e.kind === "mortar" ? img.width / 2 : face ? img.width - 7 : 6;
        v.drawImage(img, x - Math.round(ox), y - img.height + 1);
        if (hit) {
          v.fillStyle = "#ffffff";
          v.fillRect(x - 1, y - 11, 3, 3);
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
      for (const p of g.players) {
        if (p.x < cx - 20 || p.x > cx + W + 20) continue;
        if (p.inv > 0 && !p.dead && Math.floor(t * 18) % 2 === 0) continue;
        const x = Math.round(p.x - cx);
        const y = Math.round(p.y - cy);
        const frames = squad[p.id];
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

      // Water over whoever is wading in it, with a glinting surface
      {
        const L = g.level;
        const c0 = Math.floor(cx / TILE);
        const c1 = Math.ceil((cx + W) / TILE);
        const r0 = Math.max(0, Math.floor(cy / TILE));
        const r1 = Math.min(ROWS - 1, Math.ceil((cy + H) / TILE));
        const tick = Math.floor(t * 3 * motion);
        for (let c = c0; c <= c1; c++)
          for (let r = r0; r <= r1; r++) {
            if (kindAt(L, c, r) !== WATER) continue;
            const x = c * TILE - cx;
            const y = r * TILE - cy;
            v.fillStyle = "rgba(40,120,165,0.62)";
            v.fillRect(x, y, TILE, TILE);
            if (kindAt(L, c, r - 1) === WATER) continue;
            v.fillStyle = "#a8e4f4";
            for (let k = 0; k < TILE; k++) if (hash2(c * TILE + k, tick + k) > 0.55) v.fillRect(x + k, y, 1, 1);
          }
      }

      // Debris, shell casings, smoke and spray; then the foreground strip
      for (const p of g.parts) {
        if (p.k === DEBRIS || p.k === CASING) {
          v.fillStyle = p.k === CASING ? p.c : (DEBRIS_COLOR[+p.c] ?? "#8a5a2b");
          v.fillRect(Math.round(p.x - cx), Math.round(p.y - cy), p.s, p.s);
        } else if (p.k === MIST) {
          const a = p.life / p.max;
          v.globalAlpha = Math.round(a * 0.6 * 4) / 4;
          v.fillStyle = "#e8f6fa";
          disc(v, Math.round(p.x - cx), Math.round(p.y - cy), Math.round(p.s + (1 - a) * 3));
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

      // Deep underground gets darker; then the theme's tint over the whole world layer
      v.globalCompositeOperation = "source-atop";
      for (let r = Math.max(50, Math.floor(cy / TILE)); r * TILE < cy + H; r++) {
        v.fillStyle = `rgba(6,5,14,${Math.min(0.6, (r - 49) * 0.08)})`;
        v.fillRect(0, r * TILE - cy, W, TILE);
      }
      const night = lerpTheme((s) => s.night);
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

    const drawLights = (g: Game, cx: number, cy: number, t: number) => {
      const night = lerpTheme((s) => s.night);
      const L = g.level;

      // Lit windows at night, lava, crystals
      const c0 = Math.floor(cx / TILE);
      const c1 = Math.ceil((cx + W) / TILE);
      const r0 = Math.max(0, Math.floor(cy / TILE));
      const r1 = Math.min(ROWS - 1, Math.ceil((cy + H) / TILE));
      for (let c = c0; c <= c1; c++)
        for (let r = r0; r <= r1; r++) {
          const k = kindAt(L, c, r);
          if (k !== WINDOW && k !== LAVA && k !== CRYSTAL) continue;
          const x = c * TILE - cx;
          const y = r * TILE - cy;
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
          case FLAMESHOT:
            break;
          case ROCKET:
            ctx.fillStyle = "#c8ced9";
            ctx.fillRect(Math.round(x - ux * 2) - 1, y, 3, 1);
            glow(Math.round(x - ux * 4), y, 1, 1, "#ff9a2a", 1);
            break;
          case GRENADE:
          case LOB:
            ctx.fillStyle = s.kind === LOB ? "#4a5a30" : "#3a5a2a";
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

      // Fire: blasts, then flames, embers and weather
      for (const b of g.blasts) {
        const a = b.t / 0.5;
        const x = Math.round(b.x - cx);
        const y = Math.round(b.y - cy);
        if (b.t < 0.07) {
          ctx.fillStyle = "#fffbe8";
          disc(ctx, x, y, b.r);
          continue;
        }
        const out = a > 0.7 ? Math.max(0, 1 - (a - 0.7) / 0.3) : 1;
        const R = Math.round(b.r * (0.75 + 0.45 * a) * out);
        ctx.fillStyle = "#8a2414";
        disc(ctx, x, y, R + 1);
        ctx.fillStyle = a < 0.4 ? "#ffd23f" : a < 0.7 ? "#ff8a1a" : "#c2341a";
        disc(ctx, x, y, R);
        ctx.fillStyle = "#fff3b0";
        disc(ctx, x, y, Math.round(b.r * (0.55 - 0.7 * a)));
        if (a < 0.35) {
          ctx.fillStyle = "#ffe7a0";
          ring(ctx, x, y, Math.round(b.r * (1 + 1.6 * a)));
        }
      }
      // Mushroom clouds: a fireball that rises on a stem and cools to smoke
      for (const m of g.mushrooms) {
        const x = Math.round(m.x - cx);
        const base = Math.round(m.y - cy);
        const rise = Math.min(1, m.t / 1.4) * 120;
        const R = Math.round(14 + Math.min(1, m.t / 1.8) * 38);
        const top = base - rise;
        const hot = clamp01(1 - m.t / 1.6);
        ctx.globalAlpha = m.t > 4 ? Math.max(0, 1 - (m.t - 4) / 2) : 1;
        const smoke = mix("#5a4a44", "#8a7a70", clamp01((m.t - 1.5) / 2));
        ctx.fillStyle = hot > 0.2 ? mix(smoke, "#ff8a1a", hot) : smoke;
        for (let y = top; y < base; y += 2) {
          const w = 7 + Math.round(Math.sin(y * 0.3 + m.t * 4) * 2) + Math.round(((base - y) / Math.max(1, rise)) * 4);
          ctx.fillRect(x - w, y, w * 2, 2);
        }
        ctx.fillStyle = mix(smoke, "#000000", 0.3);
        disc(ctx, x, top, R + 2);
        ctx.fillStyle = hot > 0.1 ? mix(smoke, "#ffb020", hot) : smoke;
        disc(ctx, x, top, R);
        ctx.fillStyle = hot > 0.1 ? mix(smoke, "#fff3b0", hot) : mix(smoke, "#ffffff", 0.15);
        disc(ctx, x - Math.round(R * 0.2), top - Math.round(R * 0.25), Math.round(R * 0.55));
        ctx.fillStyle = mix(smoke, "#000000", 0.4);
        ctx.fillRect(x - R, top + Math.round(R * 0.5), R * 2, 3);
        ctx.globalAlpha = 1;
      }
      for (const w of g.waves) {
        ctx.fillStyle = `rgba(255,250,230,${(1 - w.t) * 0.8})`;
        ring(ctx, Math.round(w.x - cx), Math.round(w.y - cy), Math.round(w.t * 420));
        ring(ctx, Math.round(w.x - cx), Math.round(w.y - cy), Math.round(w.t * 420) + 1);
      }
      for (const p of g.parts) {
        if (p.k === DEBRIS || p.k === CASING || p.k === SMOKE || p.k === BIGSMOKE || p.k === MIST) continue;
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
        const top = e.kind === "worm" ? aim.y - 22 : e.kind === "gunship" ? e.y - 26 : e.y - (e.kind === "mech" ? 74 : 44);
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
      const s = Math.max(2, Math.min(7, Math.floor((W * 0.84) / 55)));
      const title = "Level complete";
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
      for (const line of [`Level ${String(c.no).padStart(2, "0")} - ${c.name}`, `Enemies down ${c.kills}`, `Time ${clock(c.time)}`, "Squad bonus +5000 each"]) {
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
      const night = lerpTheme((s) => s.night);
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
      if (theme.sun === "sun" || theme.sun === "red") {
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
    const onInput = () => (lastInput = performance.now());

    const step = (g: Game, dt: number) => {
      // A different page (project, 404, back home): on to a new level behind the page wipe
      if (sceneState.override !== lastOverride) {
        lastOverride = sceneState.override;
        g.nextLevel(sceneState.override === "lost" ? THEMES.cave : undefined);
      }
      g.intensity = intensity();
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
      intel.weather = WEATHER[g.theme.ambient];
      intel.light = lightOf(g.theme);
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
      drawLights(g, cx, cy, g.t);
    };

    const loop = (ms: number) => {
      raf = requestAnimationFrame(loop);
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
      step(game!, dt);
      draw(game!);
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
