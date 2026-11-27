import type { AtlasBuilder } from "../render/types";
import { makeCanvas, type Canvas2D } from "../render/canvas";
import { M, MATERIALS, VARIANTS } from "./stubTerrain";

/** Placeholder pixel art for the demo, painted at boot like the game's painters. */

const hash = (a: number, b: number) => {
  let h = Math.imul(a | 0, 374761393) ^ Math.imul(b | 0, 668265263);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
};

const hex = (c: number) => `#${(c | 0x1000000).toString(16).slice(1)}`;
const shade = (c: number, k: number) => {
  const f = (v: number) => Math.max(0, Math.min(255, Math.round(v * k)));
  return (f((c >> 16) & 255) << 16) | (f((c >> 8) & 255) << 8) | f(c & 255);
};

const BASE: Record<number, number> = {
  [M.DIRT]: 0x7a5236,
  [M.GRASS]: 0x4f9a3a,
  [M.STONE]: 0x6c6f7c,
  [M.SAND]: 0xd8c07a,
  [M.BRICK]: 0x9a4a3a,
  [M.LAVA]: 0xff7a20,
  [M.CRYSTAL]: 0x5ff0e0,
  [M.WATER]: 0x3a7ad0,
  [M.WOOD]: 0x8a6034,
  [M.METAL]: 0x8a9aa8,
};

function paintTile(x: Canvas2D, m: number, v: number) {
  const base = BASE[m];
  for (let py = 0; py < 8; py++)
    for (let px = 0; px < 8; px++) {
      const n = hash(px + v * 13 + m * 101, py + v * 7);
      // Two-pixel clusters keep the noise from reading as grain at 3× scale
      const n2 = hash((px >> 1) + v * 13 + m * 101, (py >> 1) + v * 7);
      let c = shade(base, 0.9 + n2 * 0.16 + (n - 0.5) * 0.04);
      let a = 1;
      if (m === M.GRASS && py < 2 + ((hash(px, v) * 2) | 0)) c = shade(0x6cc84a, 0.9 + n * 0.2);
      if (m === M.GRASS && py >= 3) c = shade(BASE[M.DIRT], 0.86 + n * 0.28);
      if (m === M.BRICK && (py % 4 === 3 || (px + (py >> 2) * 4 + v) % 8 === 0)) c = shade(base, 0.55);
      if (m === M.WOOD && py % 4 === 0) c = shade(base, 0.7);
      if (m === M.METAL && (px === 0 || py === 0)) c = shade(base, 1.25);
      if (m === M.LAVA) c = n > 0.7 ? 0xffd23f : n > 0.3 ? 0xff7a20 : 0xc8341a;
      if (m === M.CRYSTAL) c = n > 0.75 ? 0xe6fffc : (px + py + v) % 5 === 0 ? 0x2a8a9a : 0x5ff0e0;
      if (m === M.WATER) {
        c = py === 0 ? 0x9ad0ff : shade(base, 0.9 + n * 0.15);
        a = py === 0 ? 0.9 : 0.62;
      }
      x.globalAlpha = a;
      x.fillStyle = hex(c);
      x.fillRect(px, py, 1, 1);
    }
  x.globalAlpha = 1;
}

/** Paints every material tile and returns the material × variant → frame table. */
export function paintTiles(atlas: AtlasBuilder, tiles: Int32Array) {
  for (let m = 1; m < MATERIALS; m++)
    for (let v = 0; v < VARIANTS; v++) {
      const [c, x] = makeCanvas(8, 8);
      paintTile(x, m, v);
      tiles[m * VARIANTS + v] = atlas.add(`tile:${m}:${v}`, c);
    }
}

export const TEAM_COLORS = [0x4ab0ff, 0xff5a4a, 0x7aff4a, 0xffd23f, 0xc77dff, 0xff9a2a];

/** A soldier: 7×10, four run frames, pivot at the feet. */
function paintSoldier(x: Canvas2D, color: number, frame: number) {
  const ink = "#0d0b14";
  const body = hex(color);
  const dark = hex(shade(color, 0.6));
  x.fillStyle = ink;
  x.fillRect(2, 0, 3, 3);
  x.fillRect(1, 3, 5, 4);
  x.fillStyle = "#f0c8a0";
  x.fillRect(3, 1, 2, 1);
  x.fillStyle = body;
  x.fillRect(2, 3, 3, 3);
  x.fillStyle = dark;
  x.fillRect(2, 0, 3, 1);
  // Gun
  x.fillStyle = "#3a3a44";
  x.fillRect(4, 4, 3, 1);
  // Legs by frame
  x.fillStyle = ink;
  const legs = [
    [2, 4],
    [1, 5],
    [2, 4],
    [3, 3],
  ][frame];
  x.fillRect(legs[0], 7, 1, 3);
  x.fillRect(legs[1], 7, 1, 3);
}

/** A drone: 9×5, two rotor frames, pivot at the centre. */
function paintDrone(x: Canvas2D, frame: number) {
  x.fillStyle = "#0d0b14";
  x.fillRect(1, 2, 7, 3);
  x.fillStyle = "#8a9aa8";
  x.fillRect(2, 2, 5, 2);
  x.fillStyle = "#ff3b3b";
  x.fillRect(4, 3, 1, 1);
  x.fillStyle = "#c8d0e0";
  if (frame) x.fillRect(0, 0, 4, 1);
  else x.fillRect(5, 0, 4, 1);
  x.fillRect(4, 1, 1, 1);
}

/** A glowing orb: 5×5, pulsing over two frames. */
function paintOrb(x: Canvas2D, frame: number) {
  x.fillStyle = frame ? "#ffd23f" : "#ff8a3a";
  x.fillRect(1, 0, 3, 5);
  x.fillRect(0, 1, 5, 3);
  x.fillStyle = "#fff6c8";
  x.fillRect(1, 1, 2 + frame, 2);
}

/** A tank: 16×9, two tread frames, pivot at the bottom centre. */
function paintTank(x: Canvas2D, color: number, frame: number) {
  x.fillStyle = "#0d0b14";
  x.fillRect(0, 5, 16, 4);
  x.fillRect(3, 2, 9, 4);
  x.fillRect(10, 3, 6, 1);
  x.fillStyle = hex(shade(color, 0.8));
  x.fillRect(1, 5, 14, 2);
  x.fillRect(4, 3, 7, 2);
  x.fillStyle = "#3a3a44";
  for (let k = 0; k < 4; k++) x.fillRect(1 + k * 4 + frame * 2, 7, 2, 1);
}

export interface DemoSprites {
  /** soldier[team][frame] */
  soldier: number[][];
  tank: number[][];
  drone: number[];
  orb: number[];
}

export function paintSprites(atlas: AtlasBuilder): DemoSprites {
  const soldier = TEAM_COLORS.map((col, t) =>
    [0, 1, 2, 3].map((f) => {
      const [c, x] = makeCanvas(7, 10);
      paintSoldier(x, col, f);
      return atlas.add(`soldier:${t}:${f}`, c, 3, 10);
    })
  );
  const tank = TEAM_COLORS.map((col, t) =>
    [0, 1].map((f) => {
      const [c, x] = makeCanvas(16, 9);
      paintTank(x, col, f);
      return atlas.add(`tank:${t}:${f}`, c, 8, 9);
    })
  );
  const drone = [0, 1].map((f) => {
    const [c, x] = makeCanvas(9, 5);
    paintDrone(x, f);
    return atlas.add(`drone:${f}`, c, 4, 2);
  });
  const orb = [0, 1].map((f) => {
    const [c, x] = makeCanvas(5, 5);
    paintOrb(x, f);
    return atlas.add(`orb:${f}`, c, 2, 2);
  });
  return { soldier, tank, drone, orb };
}

/** A repeating silhouette band: ridges from layered noise, lighter towards the top edge. */
function paintRidge(w: number, h: number, color: number, rough: number, seed: number, trees: boolean) {
  const [c, x] = makeCanvas(w, h);
  for (let px = 0; px < w; px++) {
    const t = px / w;
    let y = 0;
    for (let o = 1; o <= 4; o++) y += Math.sin((t * o * 2 + hash(o, seed)) * Math.PI * 2) * (rough / o);
    let top = Math.round(h * 0.45 + y);
    if (trees && hash(px >> 2, seed) > 0.6)
      top -= 4 + Math.round(hash(px >> 2, seed + 1) * 10) - Math.abs((px % 4) - 2) * 2;
    top = Math.max(0, Math.min(h - 1, top));
    x.fillStyle = hex(shade(color, 1.15));
    x.fillRect(px, top, 1, 1);
    x.fillStyle = hex(color);
    x.fillRect(px, top + 1, 1, h - top - 1);
  }
  return c;
}

export interface DemoParallax {
  far: number;
  mid: number;
  near: number;
  clouds: number;
}

export function paintParallax(atlas: AtlasBuilder): DemoParallax {
  const [cl, cx] = makeCanvas(320, 40);
  cx.fillStyle = "#f4f6ff";
  for (let k = 0; k < 6; k++) {
    const x0 = Math.round(hash(k, 3) * 280);
    const y0 = Math.round(8 + hash(k, 4) * 20);
    for (let j = 0; j < 5; j++) cx.fillRect(x0 + j * 5 - 4, y0 - (j % 2) * 3, 16, 6);
  }
  return {
    far: atlas.add("bg:far", paintRidge(512, 140, 0x56608a, 34, 1, false), 0, 140),
    mid: atlas.add("bg:mid", paintRidge(384, 120, 0x3c5a5a, 22, 2, true), 0, 120),
    near: atlas.add("bg:near", paintRidge(256, 110, 0x26402e, 14, 3, true), 0, 110),
    clouds: atlas.add("bg:clouds", cl, 0, 0),
  };
}
