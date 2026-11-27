/**
 * The renderer demo (dev only, `engine-demo.html`): a synthetic battle at the spec's target load, with the debug
 * overlay on. Query parameters: sprites, particles, lights (counts), scale (pixel scale), fps (cap, like the
 * Wallpaper), quality (high|mid|low), renderer=canvas2d, terrain=stub (the stand-in instead of A2's terrain), auto=0
 * (no automatic blasts), debug=0, day (seconds).
 * Keys: B bloom, V vignette, L letterbox, W white-out, G grade, Z zoom punch, P pause, N jump to night.
 */
import { clearFrame, createFrame, createRenderer, DebugOverlay } from "../render";
import { paintParallax, paintSprites, paintTiles } from "./art";
import { A2Ground, type DemoGround } from "./a2Ground";
import { StubTerrain } from "./stubTerrain";
import { DemoWorld } from "./world";

const params = new URLSearchParams(location.search);
const num = (k: string, d: number) => {
  const v = Number(params.get(k));
  return params.has(k) && Number.isFinite(v) ? v : d;
};
const scale = num("scale", 3);
const maxFps = num("fps", 0);
const quality = (["high", "mid", "low"] as const).find((q) => q === params.get("quality")) ?? "high";

const canvas = document.getElementById("view") as HTMLCanvasElement;
const renderer = createRenderer(canvas, {
  pixelScale: scale,
  quality,
  maxSprites: 16384,
  gpuTiming: true,
  maxParticles: 16384,
  prefer: params.get("renderer") === "canvas2d" ? "canvas2d" : "webgl2",
});
const frame = createFrame();
const debug = new DebugOverlay(params.get("debug") !== "0");

// A2's terrain by default; ?terrain=stub uses the stand-in
let terrain: DemoGround;
if (params.get("terrain") === "stub") {
  const stub = new StubTerrain();
  paintTiles(renderer.atlas, stub.tiles);
  terrain = stub;
} else terrain = new A2Ground(renderer.atlas);
const sprites = paintSprites(renderer.atlas);
const parallax = paintParallax(renderer.atlas);
frame.terrain = terrain.view;

const world = new DemoWorld(terrain, sprites, parallax, {
  sprites: num("sprites", 2000),
  particles: num("particles", 8000),
  lights: num("lights", 64),
});
world.dayLength = num("day", 40);
if (params.get("auto") === "0") world.autoBlast = 0;

const resize = () => {
  renderer.resize(window.innerWidth, window.innerHeight, window.devicePixelRatio || 1, scale);
  world.viewW = renderer.width;
  world.viewH = renderer.height;
};
resize();
window.addEventListener("resize", resize);

/** Screen point (CSS px) to world px, through the same view the renderer uses. */
const toWorld = (cx: number, cy: number) => {
  const dpr = window.devicePixelRatio || 1;
  const ix = (cx * dpr) / scale;
  const iy = (cy * dpr) / scale;
  return {
    x: world.camX + (ix - renderer.width / 2) / world.zoom,
    y: world.camY + (iy - renderer.height / 2) / world.zoom,
  };
};
canvas.addEventListener("pointerdown", (e) => {
  const p = toWorld(e.clientX, e.clientY);
  world.blast(p.x, p.y, e.shiftKey ? 2 : 1.2);
});

let paused = false;
window.addEventListener("keydown", (e) => {
  const k = e.key.toLowerCase();
  if (k === "b") world.bloom = world.bloom > 0 ? 0 : 0.8;
  else if (k === "v") world.vignette = world.vignette > 0 ? 0 : 0.35;
  else if (k === "l") world.letterboxTarget = world.letterboxTarget > 0 ? 0 : 1;
  else if (k === "w") world.whiteTarget = world.whiteTarget > 0 ? 0 : 1;
  else if (k === "g") world.graded = !world.graded;
  else if (k === "z") world.zoom = world.zoom === 1 ? 1.5 : 1;
  else if (k === "p") paused = !paused;
  else if (k === "n") world.time += world.dayLength / 2;
});

/** Rolling timings for the overlay's system lines and for scripts reading `window.engineDemo`. */
const timing = {
  sim: { name: "DEMO SIM", ms: 0, max: 0 },
  fill: { name: "DEMO FILL", ms: 0, max: 0 },
};
const stat = (s: { ms: number; max: number }, v: number) => {
  s.ms += (v - s.ms) * 0.05;
  s.max = Math.max(v, s.max * 0.995);
};
const worldStats = { systems: [timing.sim, timing.fill], tps: 0, entities: 0 };
const title = "ENGINE DEMO · CLICK: BLAST · B BLOOM · L LETTERBOX · W WHITE · Z ZOOM · N NIGHT";

let last = performance.now();
let raf = 0;
let timer = 0;
const loop = () => {
  const now = performance.now();
  const dtMs = Math.min(250, now - last);
  last = now;
  debug.tick(dtMs);
  const t0 = performance.now();
  if (!paused) world.update(Math.min(0.1, dtMs / 1000));
  const t1 = performance.now();
  clearFrame(frame);
  world.fill(frame);
  frame.text.push(title, renderer.width / 2, renderer.height - 9, 0xffffffd0, 1, 1, 14);
  const t2 = performance.now();
  stat(timing.sim, t1 - t0);
  stat(timing.fill, t2 - t1);
  worldStats.tps = debug.fps;
  worldStats.entities = world.counts.sprites;
  debug.draw(frame, renderer, worldStats, now);
  renderer.render(frame);
  schedule();
};
/** With an fps cap (the Wallpaper's 30), wait on a timer until the next frame is nearly due, as PixelScene does. */
const schedule = () => {
  if (maxFps > 0) timer = window.setTimeout(() => (raf = requestAnimationFrame(loop)), Math.max(0, 1000 / maxFps - 6));
  else raf = requestAnimationFrame(loop);
};
schedule();

declare global {
  interface Window {
    engineDemo?: unknown;
  }
}
/** For scripted runs (Playwright): read the numbers, or stop the loop. */
window.engineDemo = {
  renderer,
  world,
  debug,
  sample: () => ({
    kind: renderer.kind,
    width: renderer.width,
    height: renderer.height,
    fps: debug.fps,
    frameMs: debug.frameMs,
    frameMax: debug.frameMax,
    renderCpuMs: renderer.stats.cpuMs,
    simMs: timing.sim.ms,
    fillMs: timing.fill.ms,
    ...renderer.stats,
  }),
  stop: () => {
    cancelAnimationFrame(raf);
    clearTimeout(timer);
  },
};
