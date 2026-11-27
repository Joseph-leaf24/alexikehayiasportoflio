/**
 * The battle's dev page (`battle-dev.html`, dev only): the vertical slice on the real renderer and frame loop.
 * Query parameters: seed, scale (pixel scale), fps (cap, like the Wallpaper), day (seconds per day–night cycle,
 * fast by default), renderer=canvas2d, quality (low|mid|high), debug=1 (overlay with the slowest systems).
 * Keys: P pause, N half a day on, F fast-forward 10 s, Z zoom punch. Click: a blast under the cursor.
 */
import { createFrame, createRenderer, DebugOverlay, Loop, runHeadless, type DebugWorldStats } from "@/engine";
import { getQuality } from "@/lib/device";
import { BattleGame, budgetsFromQuality } from "@/battle";
import { SQUAD } from "@/battle/squad";

const params = new URLSearchParams(location.search);
const num = (k: string, d: number) => {
  const v = Number(params.get(k));
  return params.has(k) && Number.isFinite(v) ? v : d;
};
const dpr = window.devicePixelRatio || 1;
const debug = params.get("debug") === "1";
const quality = getQuality();
/** The old scene's scale: about 360 rows of pixels on any screen. */
const scale = num("scale", Math.max(2, Math.round((window.innerHeight * dpr) / 360)));

const canvas = document.getElementById("view") as HTMLCanvasElement;
const renderer = createRenderer(canvas, {
  pixelScale: scale,
  quality: quality.tier,
  maxSprites: 16384,
  maxParticles: 16384,
  gpuTiming: debug,
  prefer: params.get("renderer") === "canvas2d" ? "canvas2d" : "webgl2",
});
renderer.resize(window.innerWidth, window.innerHeight, dpr, scale);

const game = new BattleGame({
  seed: num("seed", 1 + Math.floor(Math.random() * 100000)),
  viewW: renderer.width,
  viewH: renderer.height,
  budgets: budgetsFromQuality(quality),
  debug,
  unattended: true,
  dayLength: num("day", 90),
  timeOfDay: num("time", 0.15),
});
game.paintArt(renderer.atlas);

const frame = createFrame();
const overlay = new DebugOverlay(debug);
/** The overlay lists at most 14 systems: show the slowest, refreshed with its text. */
const top: DebugWorldStats = { systems: [], tps: 0, entities: 0, tick: { ms: 0 } };
let lastTop = 0;
const slowest = () => {
  const s = game.world.stats;
  top.tps = s.tps;
  top.entities = s.entities;
  top.tick = s.tick;
  const now = performance.now();
  if (now - lastTop > 250) {
    lastTop = now;
    top.systems = [...s.systems].sort((a, b) => b.ms - a.ms).slice(0, 11);
  }
  return top;
};

const loop = new Loop(game.world, {
  maxFps: num("fps", 0),
  render(alpha, frameDt) {
    overlay.tick(frameDt * 1000);
    game.buildFrame(frame, alpha);
    overlay.draw(frame, renderer, debug ? slowest() : null);
    renderer.render(frame);
  },
});
loop.start();

window.addEventListener("resize", () => {
  renderer.resize(window.innerWidth, window.innerHeight, dpr, scale);
  game.resize(renderer.width, renderer.height);
  loop.invalidate();
});

/** Screen point (CSS px) to world px, through the camera the last frame used. */
const toWorld = (cx: number, cy: number) => {
  const v = game.frames.view;
  return { x: v.left + (cx * dpr) / scale / v.zoom, y: v.top + (cy * dpr) / scale / v.zoom };
};
canvas.addEventListener("pointerdown", (e) => {
  const p = toWorld(e.clientX, e.clientY);
  game.explode(p.x, p.y, { r: e.shiftKey ? 40 : 24, power: e.shiftKey ? 12 : 6 });
});
window.addEventListener("keydown", (e) => {
  const k = e.key.toLowerCase();
  if (k === "p") {
    if (loop.paused) loop.resume();
    else loop.pause();
  } else if (k === "n") game.sky.clock = (game.sky.clock + 0.5) % 1;
  else if (k === "f") runHeadless(game.world, 10);
  else if (k === "z") game.camera.zoom = game.camera.zoom === 1 ? 1.25 : 1;
});

declare global {
  interface Window {
    battleDev?: unknown;
  }
}
/** For scripted runs (Playwright): the game, and its numbers. */
window.battleDev = {
  game,
  renderer,
  loop,
  /** Run `seconds` of simulation now, as fast as possible. */
  advance: (seconds: number) => runHeadless(game.world, seconds),
  sample: () => {
    const s = game.world.stats;
    const sq = game.world.res(SQUAD);
    return {
      kind: renderer.kind,
      width: renderer.width,
      height: renderer.height,
      fps: overlay.fps,
      frameMs: overlay.frameMs,
      tickMs: s.tick.ms,
      renderCpuMs: renderer.stats.cpuMs,
      gpuMs: renderer.stats.gpuMs,
      drawCalls: renderer.stats.drawCalls,
      sprites: renderer.stats.sprites,
      particles: renderer.stats.particles,
      lights: renderer.stats.lights,
      entities: s.entities,
      enemies: game.stats.enemies,
      kills: game.stats.kills,
      booms: game.stats.booms,
      advance: Math.round(sq.advance),
      night: game.sky.night,
      falling: game.terrain.falling.length,
      time: game.now,
    };
  },
};
