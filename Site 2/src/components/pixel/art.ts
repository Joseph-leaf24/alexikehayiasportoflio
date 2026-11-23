/**
 * Procedural pixel art for the levels: an 8×8 sprite per block kind (four variants each),
 * four-layer backdrops for every theme, the eight players, the enemy troops in each theme's
 * uniform, vehicles and bosses (drawn live, so their guns can aim and their legs can walk),
 * and a 3×5 pixel font. Everything is painted pixel by pixel at start-up: no image files.
 */
import {
  ASH,
  BAMBOO,
  BEDROCK,
  BRICK,
  CONCRETE,
  CRATE,
  CRYSTAL,
  DIRT,
  GRASS,
  HAZARD,
  HIVE,
  ICE,
  KINDS,
  LAVA,
  LEAF,
  METAL,
  MOSS,
  MUD,
  PINE,
  PIPE,
  REDROCK,
  ROCK,
  SAND,
  SANDBAG,
  SANDSTONE,
  SLIME,
  SNOW,
  STEEL,
  STONE,
  TILE,
  TRUNK,
  VINE,
  WATER,
  WINDOW,
  WOOD,
} from "./level";
import type { Backdrop } from "./themes";
import { rng } from "./rng";

/** The outline colour used everywhere (site and scene). */
export const INK = "#0d0b14";
export const VARIANTS = 4;

/** Blocks that get a dark outline where they meet open air (not see-through ones). */
export const OUTLINED = new Uint8Array(KINDS).map((_, k) =>
  k === LEAF || k === STEEL || k === TRUNK || k === LAVA || k === PIPE || k === PINE || k === VINE || k === WATER || k === BAMBOO ? 0 : 1
);

/** A representative colour per block kind, for the debris an explosion throws out. */
export const DEBRIS_COLOR: string[] = (() => {
  const c = new Array<string>(KINDS).fill("#8a5a2b");
  const set: [number, string][] = [
    [GRASS, "#5bbf3a"],
    [STONE, "#6d7184"],
    [BEDROCK, "#2f2b3a"],
    [BRICK, "#a24c36"],
    [CONCRETE, "#8a90a0"],
    [CRATE, "#b07a3a"],
    [SANDBAG, "#c4ae78"],
    [STEEL, "#5d6880"],
    [WINDOW, "#3d5a80"],
    [LEAF, "#3f9a3a"],
    [TRUNK, "#7a5230"],
    [CRYSTAL, "#5ff0e0"],
    [ASH, "#8a8088"],
    [ROCK, "#3a2f33"],
    [METAL, "#6c7488"],
    [HAZARD, "#ffcc33"],
    [PIPE, "#5a7a6a"],
    [SAND, "#e0c080"],
    [SANDSTONE, "#c89a60"],
    [SNOW, "#eef2f8"],
    [ICE, "#a8d8f0"],
    [PINE, "#2a5a40"],
    [HIVE, "#5a2a6a"],
    [SLIME, "#7aff5a"],
    [MUD, "#4a3a24"],
    [REDROCK, "#b8543a"],
    [MOSS, "#5a7a4a"],
    [VINE, "#3f9a3a"],
    [WATER, "#5ab0e0"],
    [BAMBOO, "#9ab040"],
  ];
  for (const [k, col] of set) c[k] = col;
  return c;
})();

/** "#rrggbb" → [r, g, b] */
export const rgb = (hex: string): [number, number, number] => {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};

/** Mix two "#rrggbb" colours (t = 0 → a, 1 → b), as "#rrggbb". */
export function mix(a: string, b: string, t: number) {
  const x = rgb(a);
  const y = rgb(b);
  return "#" + x.map((v, i) => Math.round(v + (y[i] - v) * t).toString(16).padStart(2, "0")).join("");
}

/** A canvas and its 2D context. */
export function makeCanvas(w: number, h: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const x = c.getContext("2d")!;
  x.imageSmoothingEnabled = false;
  return [c, x];
}

/* ------------------------------------------------------------------ */
/* Blocks                                                              */
/* ------------------------------------------------------------------ */

type Paint = (x: number, y: number) => string | null;

function paintBlock(ctx: CanvasRenderingContext2D, kind: number, variant: number, ox: number, oy: number) {
  const r = rng(kind * 97 + variant * 13 + 1);
  const dot = (x: number, y: number, col: string) => {
    ctx.fillStyle = col;
    ctx.fillRect(ox + x, oy + y, 1, 1);
  };
  const all = (fn: Paint) => {
    for (let y = 0; y < TILE; y++)
      for (let x = 0; x < TILE; x++) {
        const col = fn(x, y);
        if (col) dot(x, y, col);
      }
  };
  const speckle = (base: string, dark: string, light: string, pd = 0.14, pl = 0.1) => () => {
    const n = r();
    return n < pd ? dark : n < pd + pl ? light : base;
  };
  const dirt = speckle("#8a5a2b", "#6e4520", "#9c6834", 0.12, 0.08);
  const rock = () => {
    const n = r();
    return n < 0.14 ? "#4e4046" : n < 0.22 ? "#2a2226" : n < 0.25 ? "#7a2a14" : "#3a2f33";
  };
  const plate = (x: number, y: number, base: string, light: string, dark: string) => (y === 0 || x === 0 ? light : y === 7 || x === 7 ? dark : base);
  const sandstone = (x: number, y: number) => (y === 3 || (y === 7 && x % 4 === 0) ? "#a87a48" : speckle("#c89a60", "#b48650", "#dab070")());

  switch (kind) {
    case DIRT:
      all(dirt);
      break;
    case GRASS:
      all((_, y) => {
        const n = r();
        if (y === 0) return n < 0.5 ? "#8be05a" : "#5bbf3a";
        if (y === 1) return n < 0.2 ? "#4aa32f" : "#5bbf3a";
        if (y === 2 && n < 0.45) return "#4aa32f";
        return dirt();
      });
      break;
    case STONE: {
      all(speckle("#6d7184", "#5a5e70", "#81869a", 0.15, 0.1));
      const cx = 1 + Math.floor(r() * 5);
      const cy = 1 + Math.floor(r() * 5);
      dot(cx, cy, "#4b4f60");
      dot(cx + 1, cy + 1, "#4b4f60");
      break;
    }
    case BEDROCK:
      all(speckle("#2f2b3a", "#1f1c28", "#46405a", 0.15, 0.25));
      break;
    case BRICK:
      all((x, y) => {
        const seam = y < 4 ? x === 0 : x === 4;
        if (y === 3 || y === 7 || seam) return "#4e2a20";
        if (y === 0 || y === 4) return "#b95f45";
        return r() < 0.18 ? "#8a3e2c" : "#a24c36";
      });
      break;
    case CONCRETE:
    case WINDOW:
      all((x, y) => plate(x, y, r() < 0.1 ? "#7b8191" : "#8a90a0", "#a3a9b8", "#6e7484"));
      if (kind === WINDOW) {
        ctx.fillStyle = "#233049";
        ctx.fillRect(ox + 2, oy + 2, 4, 4);
        dot(2, 2, "#3d5a80");
        dot(3, 2, "#3d5a80");
        dot(2, 3, "#3d5a80");
      }
      break;
    case CRATE:
      all((x, y) => {
        if (x === 0 || y === 0 || x === 7 || y === 7) return "#6e4519";
        if (x === y || x === 7 - y) return "#7a4e20";
        return r() < 0.15 ? "#c08a48" : "#b07a3a";
      });
      break;
    case SANDBAG:
      all((x, y) => {
        if (y === 0) return "#d8c58f";
        if (y === 7) return "#8e7a48";
        if (y === 3 || (y < 3 ? x === 3 : x === 6)) return "#9c8752";
        return r() < 0.1 ? "#b8a26c" : "#c4ae78";
      });
      break;
    case WOOD:
      all((_, y) => {
        if (y === 0) return "#a06a35";
        if (y === 3 || y === 6) return "#6a421c";
        return r() < 0.06 ? "#5a3816" : "#8b5a2b";
      });
      break;
    case STEEL:
      all((x, y) => {
        if (x === 0) return "#7d89a3";
        if (x === 7) return "#4a5468";
        if (x === y || x === 7 - y || y === 0) return "#5d6880";
        return null;
      });
      break;
    case LEAF:
      all((x, y) => {
        const n = r();
        const edge = x === 0 || y === 0 || x === 7 || y === 7;
        if (edge && n < 0.3) return null;
        if (y <= 1 && n < 0.5) return "#5bbf3a";
        if (y >= 6) return n < 0.5 ? "#2c7330" : "#3f9a3a";
        return n < 0.15 ? "#5bbf3a" : n < 0.3 ? "#2c7330" : "#3f9a3a";
      });
      break;
    case TRUNK:
      all((x, y) => {
        if (x < 2 || x > 5) return null;
        if (x === 2 || x === 5) return "#5c3c20";
        return y % 3 === 0 && r() < 0.6 ? "#6a4626" : "#7a5230";
      });
      break;
    case CRYSTAL:
      paintBlock(ctx, STONE, variant, ox, oy);
      all((x, y) => {
        const d = Math.abs(x - 3.5) + Math.abs(y - 3.5);
        if (d < 1.5) return "#d8fffb";
        if (d < 2.6) return "#5ff0e0";
        if (d < 3.4) return "#1e8f9e";
        return null;
      });
      break;
    case ASH:
      all((_, y) => {
        const n = r();
        if (y === 0) return n < 0.4 ? "#b0a6a8" : "#8a8088";
        if (y === 1) return n < 0.5 ? "#6a6068" : rock();
        return rock();
      });
      break;
    case ROCK:
      all(rock);
      break;
    case METAL:
      all((x, y) => {
        if ((x === 1 || x === 6) && (y === 1 || y === 6)) return "#a6afc2";
        return plate(x, y, r() < 0.08 ? "#626a7e" : "#6c7488", "#8a93a8", "#454b5c");
      });
      break;
    case LAVA:
      all((x, y) => {
        const n = r();
        if (y === 0) return n < 0.5 ? "#ffe27a" : "#ffb020";
        const swirl = Math.sin(x * 1.3 + y * 0.9 + variant) > 0.3;
        return swirl ? (n < 0.5 ? "#ff8a1a" : "#ffb020") : n < 0.3 ? "#c2341a" : "#ff6a1a";
      });
      break;
    case HAZARD:
      all((x, y) => (y === 0 ? "#3a3646" : (x + y) % 4 < 2 ? "#ffcc33" : "#23202b"));
      break;
    case PIPE:
      all((x, y) => {
        if (y < 2 || y > 5) return x % 4 === 0 && y > 5 ? "#3a4048" : null;
        if (x === 0 || x === 7) return "#3a5044";
        return y === 2 ? "#8fb09e" : y === 5 ? "#3a5044" : "#5a7a6a";
      });
      break;
    case SAND:
      all((_, y) => {
        const n = r();
        if (y === 0) return n < 0.4 ? "#f4dca0" : "#e8c888";
        if (y < 3) return n < 0.3 ? "#d8b070" : "#e0c080";
        return sandstone(0, y);
      });
      break;
    case SANDSTONE:
      all(sandstone);
      break;
    case SNOW:
      all((x, y) => {
        const n = r();
        if (y < 2) return n < 0.3 ? "#ffffff" : "#eef2f8";
        if (y === 2) return n < 0.5 ? "#d0dcec" : "#eef2f8";
        if (y === 3 && n < 0.5) return "#b8c6d8";
        return speckle("#6d7184", "#5a5e70", "#81869a")();
      });
      break;
    case ICE:
      all((x, y) => {
        if (y === 0 || x === 0) return "#e0f4ff";
        if (x === y + 2 || x === y - 3) return "#ffffff";
        return r() < 0.2 ? "#7ab8e0" : "#a8d8f0";
      });
      break;
    case PINE:
      all((x, y) => {
        const n = r();
        if ((x + y) % 7 === 0 && n < 0.4) return null;
        if (y === 0 && n < 0.6) return "#eef2f8";
        return n < 0.3 ? "#1e4a34" : n < 0.4 ? "#3a6a4a" : "#2a5a40";
      });
      break;
    case HIVE:
      all((x, y) => {
        const n = r();
        const vein = Math.abs(Math.sin(x * 0.9 + y * 1.4 + variant * 2)) < 0.18;
        if (vein) return n < 0.5 ? "#c84aff" : "#9a2ad8";
        return n < 0.2 ? "#3a1a4a" : n < 0.3 ? "#7a3a8a" : "#5a2a6a";
      });
      break;
    case SLIME:
      all((x, y) => {
        const n = r();
        if (y === 0) return n < 0.5 ? "#baff7a" : "#7aff5a";
        if (y === 1 && n < 0.6) return "#4ad83a";
        if (y === 2 && n < 0.3) return "#4ad83a";
        return n < 0.2 ? "#3a1a4a" : "#5a2a6a";
      });
      break;
    case MUD:
      all((_, y) => {
        const n = r();
        if (y === 0) return n < 0.4 ? "#6a5a3a" : "#5a4a2e";
        if (y === 1 && n < 0.3) return "#3a5a3a";
        return n < 0.2 ? "#3a2e1c" : n < 0.3 ? "#5a4a30" : "#4a3a24";
      });
      break;
    case REDROCK:
      all((x, y) => {
        const n = r();
        if (y % 3 === 1 && n < 0.6) return "#c8644a";
        return n < 0.15 ? "#8a3a24" : n < 0.25 ? "#d8805a" : "#b8543a";
      });
      break;
    case MOSS:
      // Carved temple stone in courses, moss creeping over the top
      all((x, y) => {
        const n = r();
        if (y === 0) return n < 0.6 ? "#6aa040" : "#4a8030";
        if (y === 1 && n < 0.45) return "#4a8030";
        if (y === 4 || (y < 4 ? x === 0 : x === 4)) return "#3e4638";
        if (n < 0.07) return "#4a8030";
        return n < 0.2 ? "#5e6656" : n < 0.3 ? "#7a8270" : "#6a7262";
      });
      if (variant === 2) {
        // A carved glyph
        for (const [x, y] of [
          [5, 1],
          [6, 1],
          [6, 2],
          [5, 3],
          [1, 5],
          [2, 6],
          [3, 5],
        ])
          dot(x, y, "#3e4638");
      }
      break;
    case VINE:
      all((x, y) => {
        const cx = 3 + Math.round(Math.sin((y + variant * 2) * 0.9));
        if (x === cx) return y % 3 === 0 ? "#2c7330" : "#3f9a3a";
        if ((x === cx - 1 || x === cx + 1) && (y + variant + (x > cx ? 2 : 0)) % 4 === 1) return "#5bbf3a";
        return null;
      });
      break;
    case WATER:
      // Deep water (PixelScene washes a see-through layer over it, so what wades in looks submerged)
      all((x, y) => {
        const n = r();
        if (y === 0) return n < 0.5 ? "#4aa8d0" : "#3a90b8";
        if (Math.sin(x * 0.9 + y * 1.7 + variant) > 0.6) return "#2e7aa0";
        return n < 0.2 ? "#1a4e6a" : "#1e5a7a";
      });
      break;
    case BAMBOO:
      all((x, y) => {
        if (x === 5 && y === (variant * 2 + 1) % 8) return "#7aa830";
        if (x < 3 || x > 4) return null;
        if (y === (variant % 2 ? 2 : 6)) return "#5a7a20";
        return x === 3 ? "#b8d060" : "#88a838";
      });
      break;
  }
}

/** Every block sprite: column = kind, row = variant. */
export function makeAtlas() {
  const [cv, ctx] = makeCanvas(KINDS * TILE, VARIANTS * TILE);
  for (let k = 1; k < KINDS; k++) for (let v = 0; v < VARIANTS; v++) paintBlock(ctx, k, v, k * TILE, v * TILE);
  return cv;
}

/* ------------------------------------------------------------------ */
/* Sky and backdrops                                                   */
/* ------------------------------------------------------------------ */

/** Ordered 4×4 Bayer matrix, for dithering the sky's colour bands. */
export const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];

/** A puffy pixel cloud: a few overlapping blobs, lit from the top left. */
export function makeCloud(seed: number) {
  const r = rng(seed * 31 + 5);
  const w = 28 + Math.floor(r() * 40);
  const h = 12 + Math.floor(r() * 9);
  const n = 3 + Math.floor(r() * 3);
  const blobs = Array.from({ length: n }, (_, i) => {
    const rad = h * 0.35 + r() * h * 0.3;
    return { x: (w * (i + 0.5)) / n + (r() - 0.5) * 4, y: h - rad - 1, r: rad };
  });
  const inside = (x: number, y: number) => y <= h - 2 && blobs.some((b) => (x - b.x) ** 2 + (y - b.y) ** 2 <= b.r * b.r);
  const [cv, ctx] = makeCanvas(w, h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      if (!inside(x, y)) continue;
      ctx.fillStyle = y >= h - 3 ? "#b5bfd1" : y > h * 0.6 ? "#d3d9e5" : !inside(x - 1, y - 1) ? "#ffffff" : "#f1efe8";
      ctx.fillRect(x, y, 1, 1);
    }
  return cv;
}

/** Heights of a seamlessly repeating skyline (sum of whole-period sines, so the ends meet). */
function ridge(w: number, seed: number, floor: number, waves: [number, number][]) {
  const r = rng(seed);
  const phases = waves.map(() => r() * Math.PI * 2);
  return (x: number) => Math.round(floor - waves.reduce((s, [k, a], i) => s + a * (0.5 + 0.5 * Math.sin((2 * Math.PI * k * x) / w + phases[i])), 0));
}

type Ctx = CanvasRenderingContext2D;
const col = (c: Ctx, x: number, y0: number, y1: number, color: string) => {
  c.fillStyle = color;
  c.fillRect(x, y0, 1, Math.max(0, y1 - y0));
};
const box = (c: Ctx, x: number, y: number, w: number, h: number, color: string) => {
  c.fillStyle = color;
  c.fillRect(x, y, w, h);
};
/** Fill under a ridge line, with a lit edge. */
function hills(c: Ctx, w: number, h: number, top: (x: number) => number, body: string, edge: string, snow?: { below: number; color: string }) {
  for (let x = 0; x < w; x++) {
    const t = top(x);
    col(c, x, t, h, body);
    col(c, x, t, t + 2, edge);
    if (snow && t < snow.below) col(c, x, t, t + 3, snow.color);
  }
}
function palm(c: Ctx, x: number, base: number, tall: number, lean: number, color: string) {
  c.fillStyle = color;
  for (let y = 0; y < tall; y++) c.fillRect(x + Math.round((lean * y * y) / (tall * 6)), base - y, 1, 1);
  const tx = x + Math.round((lean * tall) / 6);
  const ty = base - tall;
  for (const dir of [1, -1]) for (const dx of [1, 2]) for (let s = 0; s < 6; s++) c.fillRect(tx + dir * (s + dx), ty + Math.round((s * s) / 9) + (dx - 1), 1, 1);
}
function pineTree(c: Ctx, x: number, base: number, tall: number, color: string, tip?: string) {
  for (let y = 0; y < tall; y++) {
    const half = Math.floor(((tall - y) / tall) * (tall / 3.2));
    box(c, x - half, base - y, half * 2 + 1, 1, color);
    if (tip && y % 4 === 3) box(c, x - half, base - y, half * 2 + 1, 1, tip);
  }
}
function deadTree(c: Ctx, x: number, base: number, tall: number, color: string, r: () => number) {
  col(c, x, base - tall, base, color);
  col(c, x + 1, base - tall + 4, base, color);
  for (let k = 6; k < tall; k += 5) {
    const dir = r() < 0.5 ? -1 : 1;
    for (let s = 0; s < 6; s++) box(c, x + dir * s, base - k - Math.floor(s / 2), 1, 1, color);
  }
}

/** A row of round tree crowns along `base`, lit from the top left, solid below (tiles seamlessly). */
function crowns(c: Ctx, w: number, h: number, base: number, rmin: number, rmax: number, body: string, edge: string, r: () => number) {
  for (let x = 0; x < w; x += 7 + Math.floor(r() * 9)) {
    const y = base - Math.floor(r() * 8);
    const rad = rmin + Math.floor(r() * (rmax - rmin));
    for (const o of [0, -w, w]) {
      c.fillStyle = edge;
      disc(c, x + o - 1, y - 1, rad);
      c.fillStyle = body;
      disc(c, x + o + 1, y + 1, rad);
    }
  }
  box(c, 0, base, w, h - base, body);
}

/** Stepped temple silhouette with a shrine on top, centred on x. */
function stepped(c: Ctx, x: number, base: number, tiers: number, w0: number, th: number, body: string, edge: string) {
  for (let k = 0; k < tiers; k++) {
    const tw = w0 - k * Math.round(w0 / (tiers + 1));
    const y = base - (k + 1) * th;
    box(c, x - (tw >> 1), y, tw, th, body);
    box(c, x - (tw >> 1), y, tw, 1, edge);
  }
  const top = base - tiers * th;
  box(c, x - 4, top - 7, 8, 7, body);
  box(c, x - 4, top - 8, 8, 1, edge);
}

/** Jungle undergrowth: fans of fronds rising from the bottom of a strip. */
function ferns(c: Ctx, w: number, h: number, color: string, r: () => number) {
  for (let x = 0; x < w; x += 4 + Math.floor(r() * 7)) {
    const ht = 8 + Math.floor(r() * 16);
    for (let s = -3; s <= 3; s++) for (let k = 0; k < ht - Math.abs(s) * 2; k++) box(c, (x + Math.round((s * k) / 5) + w) % w, h - k, 1, 1, color);
  }
}

/** Leaves and vines hanging into the top of the screen, in front of everything. */
function canopyTop(w: number, h: number, color: string, density: number, r: () => number) {
  const [cv, c] = makeCanvas(w, h);
  const depth = ridge(w, 7, 14, [[3, 5], [7, 4], [13, 3]]);
  for (let x = 0; x < w; x++) col(c, x, 0, depth(x), color);
  c.fillStyle = color;
  for (let x = 0; x < w; x += 3 + Math.floor(r() * 4)) for (const o of [0, -w, w]) disc(c, x + o, depth(x), 2 + Math.floor(r() * 2));
  for (let x = 2; x < w; x += Math.round((5 + r() * 9) / density)) {
    const len = 6 + Math.floor(r() * (h - 8));
    const end = Math.min(h - 1, depth(x) + len);
    col(c, x, depth(x), end, color);
    if (r() < 0.6) box(c, x - 1, end - 2, 3, 2, color);
  }
  return cv;
}

/** The parallax layers of a backdrop (far → near, a foreground strip, and for jungles a canopy across the top). */
export interface BackdropArt {
  far: HTMLCanvasElement;
  mid: HTMLCanvasElement;
  near: HTMLCanvasElement;
  front: HTMLCanvasElement;
  top?: HTMLCanvasElement;
}

export function makeBackdrop(kind: Backdrop): BackdropArt {
  const FW = 640;
  const FH = 130;
  const [far, f] = makeCanvas(FW, FH);
  const [mid, m] = makeCanvas(560, 110);
  const [near, n] = makeCanvas(512, 100);
  const [front, fr] = makeCanvas(480, 40);
  let top: HTMLCanvasElement | undefined;
  const r = rng(kind.length * 131 + kind.charCodeAt(0));

  const mountains = (c: Ctx, seed: number, body: string, edge: string, snowCol?: string) =>
    hills(c, FW, FH, ridge(FW, seed, FH - 10, [[1, 30], [2, 22], [3, 14], [7, 9], [13, 4]]), body, edge, snowCol ? { below: 50, color: snowCol } : undefined);
  const rolling = (c: Ctx, w: number, h: number, seed: number, body: string, edge: string, amp = 1) =>
    hills(c, w, h, ridge(w, seed, h - 8, [[2, 14 * amp], [3, 10 * amp], [7, 6 * amp], [11, 3 * amp]]), body, edge);
  const rubble = (c: Ctx, w: number, h: number, color: string, top = 12) => {
    const t = ridge(w, 99, h - 2, [[5, top], [11, top / 2], [23, top / 3]]);
    for (let x = 0; x < w; x++) col(c, x, t(x), h, color);
  };

  switch (kind) {
    case "jungle": {
      mountains(f, 21, "#86a0c0", "#a3b8d3", "#e8eef6");
      rolling(m, 560, 110, 33, "#4b7a52", "#5a8a5e");
      for (let i = 0; i < 18; i++) palm(m, 6 + Math.floor(r() * 548), 92, 16 + Math.floor(r() * 14), r() < 0.5 ? -1 : 1, "#335a40");
      rolling(n, 512, 100, 34, "#24402e", "#2e5038", 1.3);
      for (let i = 0; i < 12; i++) palm(n, Math.floor(r() * 500), 80, 24 + Math.floor(r() * 16), r() < 0.5 ? -1 : 1, "#1a3022");
      rubble(fr, 480, 40, "#10200f", 14);
      break;
    }
    case "deepjungle": {
      // Misty peaks with waterfalls, then wall after wall of rainforest, vines hanging off it
      const peaks = ridge(FW, 23, FH - 10, [[1, 30], [2, 22], [3, 14], [7, 9], [13, 4]]);
      hills(f, FW, FH, peaks, "#6e8e80", "#88a898");
      for (let i = 0; i < 4; i++) {
        const x = 30 + Math.floor(r() * 580);
        for (let y = peaks(x) + 6; y < FH; y++) if (y % 4) box(f, x, y, 2, 1, "#d8ece8");
      }
      box(f, 0, FH - 34, FW, 10, "rgba(220,240,230,0.22)");
      box(m, 0, 64, 560, 46, "#22402e");
      for (let x = 0; x < 560; x += 5 + Math.floor(r() * 7)) col(m, x, 60, 110, "#1a3324");
      crowns(m, 560, 110, 66, 9, 16, "#2e5a40", "#3e6e4e", r);
      crowns(n, 512, 100, 58, 10, 18, "#173222", "#21402c", r);
      for (let x = 0; x < 512; x += 5 + Math.floor(r() * 9)) col(n, x, 58, 58 + 6 + Math.floor(r() * 22), "#0e2416");
      rubble(fr, 480, 40, "#0a1a0e", 10);
      ferns(fr, 480, 40, "#0a1a0e", r);
      top = canopyTop(480, 40, "#08140a", 1, r);
      break;
    }
    case "temple": {
      // Dusk haze over the hills, temple tops breaking the canopy, broken pillars up close
      const hillsTop = ridge(FW, 29, FH - 12, [[1, 22], [2, 16], [5, 8], [9, 4]]);
      hills(f, FW, FH, hillsTop, "#9a7a8a", "#b08e98");
      for (const x of [120, 420]) stepped(f, x, hillsTop(x) + 8, 4, 60, 7, "#8a6a7a", "#a8889a");
      crowns(m, 560, 110, 76, 8, 14, "#3a4a36", "#4a5c44", r);
      stepped(m, 300, 84, 5, 110, 9, "#4a5040", "#6a7058");
      box(m, 297, 84 - 5 * 9 - 5, 6, 4, "#ffb060");
      crowns(n, 512, 100, 66, 10, 16, "#1e2a1c", "#283624", r);
      for (let i = 0; i < 6; i++) {
        const x = Math.floor(r() * 500);
        const h = 18 + Math.floor(r() * 22);
        box(n, x, 66 - h, 6, h, "#1a2418");
        box(n, x - 2, 66 - h - 3, 10, 3, "#1a2418");
      }
      rubble(fr, 480, 40, "#0e160c", 12);
      ferns(fr, 480, 40, "#0e160c", r);
      top = canopyTop(480, 36, "#0c120a", 0.7, r);
      break;
    }
    case "monsoon": {
      // Low, rain-dark hills, palms and stilt villages along the river
      hills(f, FW, FH, ridge(FW, 37, FH - 14, [[1, 18], [2, 14], [4, 8], [9, 3]]), "#3a4a50", "#4a5a60");
      box(f, 0, FH - 40, FW, 12, "rgba(160,180,190,0.18)");
      crowns(m, 560, 110, 80, 7, 12, "#26383a", "#30464a", r);
      for (let i = 0; i < 10; i++) palm(m, Math.floor(r() * 550), 84, 18 + Math.floor(r() * 12), r() < 0.5 ? -1 : 1, "#1e2e30");
      crowns(n, 512, 100, 72, 8, 14, "#16242a", "#1e3036", r);
      for (let i = 0; i < 5; i++) {
        // Stilt huts
        const x = Math.floor(r() * 470);
        box(n, x, 44, 26, 12, "#141e20");
        for (let k = 0; k < 15; k++) box(n, x - 2 + k * 2, 44 - Math.min(k, 14 - k), 2, 1, "#141e20");
        for (const lx of [x + 2, x + 12, x + 22]) box(n, lx, 56, 2, 20, "#141e20");
      }
      for (let x = 0; x < 480; x += 2 + Math.floor(r() * 3)) col(fr, x, 40 - 8 - Math.floor(r() * 26), 40, "#0a1214");
      top = canopyTop(480, 30, "#0a1214", 0.5, r);
      break;
    }
    case "desert":
    case "canyon": {
      const desert = kind === "desert";
      if (desert) {
        hills(f, FW, FH, ridge(FW, 41, FH - 12, [[1, 18], [2, 12], [5, 6]]), "#e0b880", "#ecc890");
        for (let i = 0; i < 5; i++) {
          const x = Math.floor(r() * 600);
          box(f, x, FH - 60, 30 + Math.floor(r() * 30), 50, "#d0a070");
        }
        hills(m, 560, 110, ridge(560, 42, 100, [[2, 20], [3, 12], [7, 5]]), "#c89a60", "#d8aa70");
        for (let i = 0; i < 6; i++) {
          // Distant oil derricks
          const x = 20 + Math.floor(r() * 520);
          for (let y = 0; y < 30; y++) {
            const half = Math.floor((30 - y) / 6);
            box(m, x - half, 70 - y, 1, 1, "#8a6a40");
            box(m, x + half, 70 - y, 1, 1, "#8a6a40");
            if (y % 6 === 0) box(m, x - half, 70 - y, half * 2 + 1, 1, "#8a6a40");
          }
        }
        rolling(n, 512, 100, 43, "#9a6a3a", "#aa7a4a", 1.2);
        for (let i = 0; i < 10; i++) {
          // Cactus silhouettes
          const x = Math.floor(r() * 500);
          const h = 12 + Math.floor(r() * 10);
          box(n, x, 84 - h, 3, h, "#6a4a2a");
          box(n, x - 3, 84 - h + 5, 3, 2, "#6a4a2a");
          box(n, x - 3, 84 - h + 1, 1, 5, "#6a4a2a");
          box(n, x + 3, 84 - h + 7, 3, 2, "#6a4a2a");
          box(n, x + 5, 84 - h + 3, 1, 5, "#6a4a2a");
        }
        rubble(fr, 480, 40, "#4a3018", 10);
      } else {
        // Buttes and mesas with striped walls
        for (let x = 0; x < FW; ) {
          const w = 30 + Math.floor(r() * 60);
          const h = 30 + Math.floor(r() * 60);
          box(f, x, FH - h, w, h, "#d8987a");
          for (let y = FH - h + 6; y < FH; y += 9) box(f, x, y, w, 2, "#c8846a");
          x += w + Math.floor(r() * 50);
        }
        for (let x = 0; x < 560; ) {
          const w = 20 + Math.floor(r() * 40);
          const h = 40 + Math.floor(r() * 50);
          box(m, x, 110 - h, w, h, "#a8543a");
          box(m, x, 110 - h, w, 3, "#c8744a");
          for (let y = 110 - h + 8; y < 110; y += 7) box(m, x + 2, y, w - 4, 1, "#904430");
          x += w + 10 + Math.floor(r() * 40);
        }
        for (let x = 0; x < 512; ) {
          const w = 6 + Math.floor(r() * 10);
          const h = 30 + Math.floor(r() * 50);
          for (let k = 0; k < w; k++) col(n, x + k, 100 - h + Math.abs(k - w / 2) * 2, 100, "#6a2a1e");
          x += w + 8 + Math.floor(r() * 30);
        }
        rubble(fr, 480, 40, "#3a160e", 12);
      }
      break;
    }
    case "snow": {
      mountains(f, 51, "#a8b8d0", "#d8e4f0", "#ffffff");
      for (let x = 0; x < 560; x += 6 + Math.floor(r() * 8)) pineTree(m, x, 100, 18 + Math.floor(r() * 16), "#4a6a7a", "#e8eef6");
      box(m, 0, 98, 560, 12, "#d8e4f0");
      for (let x = 0; x < 512; x += 10 + Math.floor(r() * 12)) pineTree(n, x, 92, 30 + Math.floor(r() * 24), "#1e3a44", "#c8d8e8");
      box(n, 0, 90, 512, 10, "#e8eef6");
      rubble(fr, 480, 40, "#e8eef6", 12);
      break;
    }
    case "fort":
    case "silo": {
      mountains(f, 61, kind === "fort" ? "#7a8aa8" : "#b87a6a", kind === "fort" ? "#98a8c0" : "#d08a70");
      box(m, 0, 76, 560, 34, "#3c4660");
      for (let x = 0; x < 560; x += 8) box(m, x, 72, 4, 4, "#3c4660");
      for (let i = 0; i < 7; i++) {
        const x = Math.floor(r() * 520);
        const h = 30 + Math.floor(r() * 30);
        box(m, x, 76 - h, 18, h, "#343d55");
        for (let k = 0; k < 18; k += 6) box(m, x + k, 72 - h, 4, 4, "#343d55");
        box(m, x + 7, 82 - h, 3, 5, "#ffd479");
      }
      if (kind === "silo") {
        for (let i = 0; i < 5; i++) {
          // Radar domes and missiles on their gantries
          const x = Math.floor(r() * 480);
          for (let k = -10; k <= 10; k++) col(n, x + k, 70 - Math.round(Math.sqrt(100 - k * k)), 74, "#2a2a3a");
          box(n, x + 30, 30, 4, 44, "#2a2a3a");
          box(n, x + 29, 26, 6, 6, "#2a2a3a");
          box(n, x + 36, 36, 2, 38, "#2a2a3a");
        }
      } else
        for (let i = 0; i < 6; i++) {
          // Watchtowers
          const x = Math.floor(r() * 490);
          box(n, x, 40, 14, 8, "#22283a");
          box(n, x + 1, 48, 2, 30, "#22283a");
          box(n, x + 11, 48, 2, 30, "#22283a");
          box(n, x - 2, 38, 18, 2, "#22283a");
        }
      box(n, 0, 76, 512, 24, "#22283a");
      for (let x = 0; x < 480; x += 3) box(fr, x, 26 + ((x >> 1) % 2), 2, 1, "#1a1e2a");
      rubble(fr, 480, 40, "#14182a", 8);
      break;
    }
    case "factory": {
      box(f, 0, 104, FW, 26, "#3a2a3a");
      for (let x = 0; x < FW; ) {
        const w = 14 + Math.floor(r() * 30);
        const h = 20 + Math.floor(r() * 60);
        box(f, x, FH - h - 20, w, h + 20, "#3a2a3a");
        if (r() < 0.5) box(f, x + 4, FH - h - 56, 5, 38, "#3a2a3a");
        for (let y = FH - h - 14; y < FH - 24; y += 7) if (r() < 0.5) box(f, x + 3 + Math.floor(r() * (w - 6)), y, 2, 2, "#ffb84a");
        x += w + Math.floor(r() * 10);
      }
      box(m, 0, 84, 560, 26, "#2a2030");
      for (let x = 0; x < 560; x += 70) {
        box(m, x + 6, 56, 30, 28, "#2a2030");
        box(m, x + 4, 54, 34, 4, "#2a2030");
        col(m, x + 52, 10, 84, "#2a2030");
        box(m, x + 30, 10, 36, 3, "#2a2030");
      }
      for (let x = 0; x < 512; x += 3) box(n, x, 66 + Math.round(Math.sin(x * 0.05) * 3), 3, 6, "#1a1420");
      for (let x = 0; x < 512; x += 40) box(n, x, 60, 4, 40, "#1a1420");
      box(n, 0, 84, 512, 16, "#1a1420");
      rubble(fr, 480, 40, "#100c14", 8);
      break;
    }
    case "city": {
      for (let x = 0; x < FW; ) {
        const w = 16 + Math.floor(r() * 26);
        const h = 40 + Math.floor(r() * 80);
        box(f, x, FH - h, w, h, "#1a1e34");
        for (let k = 0; k < w; k += 3) box(f, x + k, FH - h - Math.floor(r() * 4), 3, 4, "#1a1e34");
        for (let y = FH - h + 4; y < FH - 4; y += 6) for (let k = 3; k < w - 3; k += 5) if (r() < 0.14) box(f, x + k, y, 2, 3, "#ffd479");
        x += w + 2 + Math.floor(r() * 6);
      }
      for (let x = 0; x < 560; ) {
        const w = 24 + Math.floor(r() * 30);
        const h = 40 + Math.floor(r() * 50);
        box(m, x, 110 - h, w, h, "#12152a");
        // Shell holes and the odd burning window
        for (let k = 0; k < 3; k++) box(m, x + Math.floor(r() * (w - 6)), 110 - h + Math.floor(r() * (h - 10)), 5, 4, "#1e2440");
        if (r() < 0.5) box(m, x + 4 + Math.floor(r() * (w - 8)), 110 - h + 8, 3, 3, "#ff8a3a");
        x += w + 4 + Math.floor(r() * 16);
      }
      rubble(n, 512, 100, "#0c0e1c", 16);
      rubble(fr, 480, 40, "#08090f", 14);
      break;
    }
    case "volcano": {
      hills(f, FW, FH, ridge(FW, 71, FH - 6, [[1, 50], [2, 26], [5, 10], [11, 4]]), "#2a1014", "#3a161a");
      for (let x = 0; x < FW; x += 53) col(f, x, 40, 100, "#b8321a");
      for (let x = 0; x < 560; ) {
        const w = 3 + Math.floor(r() * 7);
        const h = 20 + Math.floor(r() * 50);
        for (let k = 0; k < w; k++) col(m, x + k, 110 - h + Math.abs(k - w / 2) * 2, 110, "#1e0c0e");
        col(m, x, 110 - h + w, 110, "#5a1a10");
        x += w + 6 + Math.floor(r() * 18);
      }
      rubble(n, 512, 100, "#140608", 20);
      for (let x = 0; x < 512; x += 23) col(n, x, 86, 100, "#ff5a1a");
      rubble(fr, 480, 40, "#0c0405", 12);
      break;
    }
    case "hive": {
      for (let x = 0; x < FW; ) {
        const w = 10 + Math.floor(r() * 20);
        const h = 40 + Math.floor(r() * 80);
        for (let k = 0; k < w; k++) col(f, x + k, FH - h + Math.round((Math.abs(k - w / 2) / (w / 2)) ** 2 * 30), FH, "#2a1440");
        for (let k = 0; k < 4; k++) box(f, x + Math.floor(r() * w), FH - h + 20 + Math.floor(r() * (h - 30)), 2, 2, "#c84aff");
        x += w + Math.floor(r() * 20);
      }
      for (let x = 0; x < 560; x += 4 + Math.floor(r() * 10)) {
        const h = 20 + Math.floor(r() * 60);
        for (let y = 0; y < h; y++) box(m, x + Math.round(Math.sin(y * 0.15 + x) * 3), 110 - y, 3, 1, "#1e0c2e");
      }
      rubble(n, 512, 100, "#14081e", 22);
      for (let x = 0; x < 512; x += 31) box(n, x, 80 + (x % 7), 3, 3, "#7aff5a");
      rubble(fr, 480, 40, "#0a0410", 16);
      break;
    }
    case "swamp": {
      rolling(f, FW, FH, 81, "#5a7a64", "#6a8a74", 1.4);
      for (let x = 0; x < 560; x += 12 + Math.floor(r() * 20)) deadTree(m, x, 104, 30 + Math.floor(r() * 30), "#2e4436", r);
      box(m, 0, 100, 560, 10, "#2e4436");
      for (let x = 0; x < 512; x += 20 + Math.floor(r() * 30)) deadTree(n, x, 96, 40 + Math.floor(r() * 40), "#16241c", r);
      box(n, 0, 92, 512, 8, "#16241c");
      for (let x = 0; x < 480; x += 2) col(fr, x, 26 - ((x * 7) % 11), 40, "#0c160e");
      break;
    }
    default: {
      // Cave: a rock wall, stalagmites and columns
      box(f, 0, 0, FW, FH, "#141220");
      for (let i = 0; i < 180; i++) box(f, Math.floor(r() * FW), Math.floor(r() * FH), 3 + Math.floor(r() * 8), 2 + Math.floor(r() * 4), "#1c1a2c");
      for (let x = 0; x < 560; ) {
        const w = 4 + Math.floor(r() * 8);
        const h = 14 + Math.floor(r() * 40);
        for (let k = 0; k < w; k++) col(m, x + k, 110 - h + Math.abs(k - w / 2) * 3, 110, "#0e0c18");
        if (r() < 0.3) box(m, x + (w >> 1), 110 - h + 4, 2, 2, "#5ff0e0");
        x += w + 4 + Math.floor(r() * 20);
      }
      rubble(n, 512, 100, "#09080f", 18);
      rubble(fr, 480, 40, "#050408", 12);
    }
  }
  return { far, mid, near, front, top };
}

/* ------------------------------------------------------------------ */
/* People                                                              */
/* ------------------------------------------------------------------ */

/** Heads (rows 0–5) of the eight players; then the enemies' masked one and the sniper's hood. */
const HEADS = [
  ["....HHHH", "...HhHHHH", "...HHHHHHH", "...dSSSES", "...dSSSSs", ".....sS"], // P1 combat helmet
  ["....HHHH", "...HhHHHHHH", "....SSSSS", "....SGGGG", "....SSSSs", ".....sS"], // P2 cap and shades
  ["....HHHH", "...HhHHHH", "...HHVVVVV", "...HHVvVVV", "....SSSSs", ".....sS"], // P3 visor helmet
  ["...HHHHH", "..HhHHHHH", "....SSSSS", "....SSSES", "....SSSSs", "....RRRR"], // P4 beret and scarf
  ["....HHHH", "..HHHhHHHH", "..HHHHHHHHH", "....SSSES", "....SSSSs", ".....sS"], // P5 boonie hat
  ["....HHHH", "...HHhHHH", "...HHMMMM", "...HMMVVM", "...HMMMMF", "....HHH"], // P6 hood and gas mask
  ["....HHHH", "...HHHHHHR", "...hVVVVh", "....SSSES", "....SSSSs", ".....sS"], // P7 bandana and goggles
  ["....SSSS", "...SSSSSS", "...dSSSSSd", "...dSSSES", "....YYYYs", ".....YY"], // P8 headset and beard
  ["....HHHH", "...HhHHHH", "...HHHHHHH", "....FFFRF", "....FfFFF", ".....FF"], // enemy: gas mask, red eye
  ["...HHHHH", "..HHhHHHHH", "..HHHHHHHHH", "...HHFFRF", "...HHFFFF", "....HHH"], // enemy sniper: ghillie hood
];
const TORSO = ["...TCCCT", "..cTCCCTC", "..cTCCCTC", "..cTTTTTc", "...BBPBB", "...LLLLL"];
/** Weapons over the torso (rows 6–9). */
const GUNS = [
  ["", ".......SWWWWWW", ".......SWwW", ".........W"], // rifle
  ["", ".......SWWWWW", "......SWWwWW"], // shotgun
  ["", ".......SWWWXXF", ".......SWWWW", "........W"], // plasma
  ["WWWWWWWWWWWWWR", ".......S", "", ""], // rocket tube on the shoulder
  [".........www", ".......SWWWWWW", "........W"], // sniper rifle with scope
  ["YY", "YY.....SWWWXF", "YY.....SW", "YY"], // flamethrower and its tank
  ["", ".......SWWWWW", "......SWOOWW", "........WW"], // grenade launcher with a drum
  ["", "......SWWwWwWw", "......SWWWWWWW", ".....SSWWwWwW"], // minigun
];
/** Legs (rows 12–17): idle, four running frames, jumping. */
const LEGS = [
  ["...LLLLL", "...LLl.LL", "...LL..LL", "...LL..LL", "...KK..KK", "..KKKk.KKKk"],
  ["...LLLLL", "..LLL..LL", "..LL....LL", ".LL......LL", ".KK.......KK", "KKk.......KKk"],
  ["...LLLLL", "...LLLLL", "....LLLl", "....LL.LL", "....KK..KK", "...KKKk"],
  ["...LLLLL", "...LL.LLL", "..LL...LL", "..LL....LL", "..KKk....KK", ".KKk.....KKk"],
  ["...LLLLL", "...LLLLL", "...lLLL", "...LL.LL", "..KK...KK", "........KKKk"],
  ["...LLLLL", "..LLLLLLL", ".LLl....LL", ".LL......LL", "KKk.......KK", ""],
];
export const FRAMES = { idle: 0, run: [1, 2, 3, 4], jump: 5 } as const;

const SKIN = ["#f2c29b", "#c68a5a", "#e0a878", "#8d5a3b", "#f0b890", "#b07a50", "#e8b088", "#d09068"];

type Pal = Record<string, string>;

/** Paint a character map (rows of palette letters) with a 1 px ink outline, facing right or left. */
function paintMap(rows: string[], pal: Pal, flip: boolean, width?: number) {
  const h = rows.length;
  const w = width ?? Math.max(...rows.map((r) => r.length));
  const grid = rows.map((r) => r.padEnd(w, "."));
  const [cv, ctx] = makeCanvas(w + 2, h + 2);
  const at = (x: number, y: number) => {
    if (y < 0 || y >= h || x < 0 || x >= w) return false;
    const ch = grid[y][flip ? w - 1 - x : x];
    return ch !== "." && !!pal[ch];
  };
  ctx.fillStyle = INK;
  for (let y = -1; y <= h; y++)
    for (let x = -1; x <= w; x++) if (!at(x, y) && (at(x - 1, y) || at(x + 1, y) || at(x, y - 1) || at(x, y + 1))) ctx.fillRect(x + 1, y + 1, 1, 1);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      if (!at(x, y)) continue;
      ctx.fillStyle = pal[grid[y][flip ? w - 1 - x : x]];
      ctx.fillRect(x + 1, y + 1, 1, 1);
    }
  return cv;
}

/** A soldier map: head, torso with a gun over it, legs. */
function bodyMap(head: string[], gun: string[], legs: string[]) {
  const W = 14;
  const torso = TORSO.map((row, i) => {
    const g = (gun[i] ?? "").padEnd(W, ".");
    return Array.from(row.padEnd(W, "."), (ch, x) => (g[x] !== "." ? g[x] : ch)).join("");
  });
  return [...head, ...torso, ...legs].map((r) => r.padEnd(W, "."));
}

/** The palette for player `id` (0–7) in `color`. */
function playerPalette(id: number, color: string): Pal {
  return {
    H: id === 4 ? mix(color, "#5a5a30", 0.5) : mix(color, "#000000", 0.3),
    h: mix(color, "#ffffff", 0.35),
    d: "#2a2430",
    S: SKIN[id],
    s: mix(SKIN[id], "#000000", 0.25),
    E: "#1a1a1a",
    G: "#1a1a24",
    V: id === 5 ? "#ffd23f" : "#7fe3ff",
    v: "#e0fbff",
    M: "#5a5e6a",
    F: "#2e3240",
    R: id === 6 ? color : "#c8302a",
    Y: "#5a3a20",
    C: color,
    c: mix(color, "#000000", 0.3),
    T: "#2e2c28",
    B: "#1e1c22",
    P: "#4a4434",
    L: mix(color, "#1c1c2a", 0.62),
    l: mix(color, "#000000", 0.7),
    K: "#1a1820",
    k: "#3a3640",
    W: "#2e323e",
    w: "#5d6578",
    X: "#c77dff",
    O: "#4a4e5a",
    Y2: "#c8302a",
  };
}

/** One player's sprite (16×20 with outline), for the scene and the portraits on the page. */
export function playerSprite(id: number, color: string, frame = 0, flip = false) {
  const pal = playerPalette(id, color);
  // (the flamethrower's back tank is red)
  if (id === 5) pal.Y = "#c8302a";
  return paintMap(bodyMap(HEADS[id], GUNS[id], LEGS[frame]), pal, flip);
}

/** sprites[player][frame][facing: 0 right, 1 left] */
export function makeSquad(colors: string[]) {
  return colors.map((c, id) => LEGS.map((_, frame) => [playerSprite(id, c, frame), playerSprite(id, c, frame, true)]));
}

/** Enemy uniforms per theme. */
const LOOKS: Record<Backdrop, { helmet: string; uniform: string; legs: string }> = {
  jungle: { helmet: "#4a5230", uniform: "#7a7a50", legs: "#4a4a32" },
  deepjungle: { helmet: "#3a4a24", uniform: "#5a6a30", legs: "#3a4220" },
  temple: { helmet: "#6a5030", uniform: "#a08050", legs: "#5a4428" },
  monsoon: { helmet: "#2a3428", uniform: "#3e4e3a", legs: "#262e22" },
  desert: { helmet: "#8a7448", uniform: "#c8a870", legs: "#8a6a44" },
  canyon: { helmet: "#8a7448", uniform: "#c8a870", legs: "#8a6a44" },
  snow: { helmet: "#c8d0dc", uniform: "#e8ecf2", legs: "#8a94a4" },
  fort: { helmet: "#3a3e48", uniform: "#7a7e8a", legs: "#4a4e5a" },
  silo: { helmet: "#3a3e48", uniform: "#7a7e8a", legs: "#4a4e5a" },
  factory: { helmet: "#22283a", uniform: "#3e4a6a", legs: "#262c40" },
  volcano: { helmet: "#2a0a0e", uniform: "#8a1e1e", legs: "#3a1012" },
  city: { helmet: "#16161c", uniform: "#3a3a44", legs: "#1e1e26" },
  hive: { helmet: "#3a1a4a", uniform: "#6a3a8a", legs: "#2a1a3a" },
  swamp: { helmet: "#2e3a24", uniform: "#4e5a3a", legs: "#2a3220" },
  cave: { helmet: "#2a1e3a", uniform: "#4e3a6a", legs: "#2a2040" },
};

const HEAVY = (legs: number) => [
  "......HHHHH",
  ".....HhHHHHH",
  ".....HHHHHHHH",
  "......FFFFRF",
  "......FfFFFF",
  "...TTTTTTTTTT",
  "..TTCCCCCCCTT",
  "..TCCCCCCCCCTWWW",
  "..TCCCCCCCCSWwww",
  "..TCCCCCCCCSWWWW",
  "..TTTTTTTTTTWWW",
  "...BBBPPBBB",
  "...LLLLLLLL",
  ...(legs ? ["...LLLL.LLLL", "...LLL...LLL", "..LLL.....LLL", "..KKK.....KKK", ".KKKKk...KKKKk"] : ["...LLLLLLLL", "....LLL.LLL", "....LLL.LLL", "....KKK.KKK", "...KKKKkKKKKk"]),
];
const MORTAR = [
  "",
  "....HHHH",
  "...HhHHHH",
  "....FFRF.....W",
  "....FFFF....WW",
  "...CCCCCC..WW",
  "..cCCCCCCSWW",
  "..cCCCCCC.W",
  "...BBBBB.WWW",
  "..LLLLLLWWWWW",
  ".LLLLLLLL.W.W",
  "KKKK.KKKK",
];
const MG = [
  "",
  "......HHHH",
  ".....HhHHHH",
  ".....HHHHHHH",
  "......FFFRF",
  "......FFFFF",
  ".....CCCCCCS",
  "....cCCCCCCSWWWWWWWW",
  "....cCCCCCCSWWwWWWW",
  ".....TTTTT.WWW",
  "......BBB..W.W",
  ".....LLLLL.W..W",
  "....LLLLLLW....W",
  "...KKKK.KKK.....",
];
const TURRET = ["....TTTT", "...TtTTTT", "...TTTTTTGGGG", "..TTTTTTTT", ".BBBBBBBBBB", "BBBBBBBBBBBB", "BBbBBBBbBBBB"];
const CRAWLER = [
  ["...AAAAA", "..AaaAAAAA", ".AAAAAAAAAE", "AAAAAAAAAAAA", ".dAAAAAAAAd", "d.d.d..d.d.d", ""],
  ["...AAAAA", "..AaaAAAAA", ".AAAAAAAAAE", "AAAAAAAAAAAA", ".dAAAAAAAAd", ".d.d.dd.d.d.", ""],
];
/** An attack dog at full stretch and gathering itself (faces right). */
const DOG = [
  ["...........dd", "..........dDDd", "d.........DDEDN", ".dDdddddddDDD", "..DDDDDDDDDD", "..DD.....DD", ".D..D...D..D", "D....D.D....D"],
  ["...........dd", "..........dDDd", ".d........DDEDN", "ddDdddddddDDD", "..DDDDDDDDDD", "...DD...DD", "...DD...DD", "..DD...DD"],
];
/** A riot shield held in front of a trooper (columns 10–13 of the soldier map, from row 2). */
const SHIELD_PLATE = [".qq.", "qQzq", "qQzq", "qQQq", "qQQq", "qQQq", "qQQq", "qQQq", "qQQq", "qQQq", "qQQq", ".qq."];
/** A sidearm, for the shield trooper. */
const PISTOL = ["", ".......SWW", "", ""];

/** Lay `patch` over a character map at column x, row y (dots in the patch leave the map alone). */
function overlay(rows: string[], x: number, y: number, patch: string[]) {
  return rows.map((row, i) => {
    const p = patch[i - y];
    if (!p) return row;
    const chars = row.padEnd(x + p.length, ".").split("");
    for (let k = 0; k < p.length; k++) if (p[k] !== ".") chars[x + k] = p[k];
    return chars.join("");
  });
}

/** Enemy sprites in a theme's uniform; each [right, left]. */
export function makeTroops(look: Backdrop) {
  const p = LOOKS[look];
  const pal: Pal = {
    H: p.helmet,
    h: mix(p.helmet, "#ffffff", 0.25),
    F: "#22202a",
    f: "#4a4854",
    R: "#ff4040",
    C: p.uniform,
    c: mix(p.uniform, "#000000", 0.3),
    T: mix(p.uniform, "#000000", 0.55),
    B: "#1e1c24",
    P: "#3a3630",
    L: p.legs,
    l: mix(p.legs, "#000000", 0.3),
    K: "#15131a",
    k: "#2e2a34",
    S: "#2e2a36",
    W: "#2e3240",
    w: "#4e5464",
    X: "#ff6a3d",
    // The flamethrower's fuel tank, the grenadier's drum, the riot shield
    Y: "#c8302a",
    O: "#4a4e5a",
    Q: "#7a8698",
    q: "#2e3442",
    z: "#c8d0dc",
  };
  const both = (rows: string[], palette = pal) => [paintMap(rows, palette, false), paintMap(rows, palette, true)];
  const gun = (i: number) => GUNS[i];
  const turretPal: Pal = { T: "#6a7080", t: "#8a90a0", B: "#4a4f5f", b: "#ff3b3b", G: "#2e3240" };
  const bug: Pal = look === "volcano" ? { A: "#8a2a14", a: "#c84a24", d: "#3a0a06", E: "#ffe27a" } : { A: "#4a2a6a", a: "#8a5aaa", d: "#1e0e2e", E: "#8aff6a" };
  const dogPal: Pal = { D: "#8a6a40", d: "#3a2a1c", E: "#e8d8b0", N: "#1a1418" };
  return {
    mook: LEGS.map((legs) => both(bodyMap(HEADS[8], gun(0), legs))),
    rocket: LEGS.map((legs) => both(bodyMap(HEADS[8], gun(3), legs))),
    flamer: LEGS.map((legs) => both(bodyMap(HEADS[8], gun(5), legs))),
    grenadier: LEGS.map((legs) => both(bodyMap(HEADS[8], gun(6), legs))),
    shield: LEGS.map((legs) => both(overlay(bodyMap(HEADS[8], PISTOL, legs), 10, 2, SHIELD_PLATE))),
    sniper: [both(bodyMap(HEADS[9], gun(4), LEGS[0]))],
    heavy: [0, 1].map((k) => both(HEAVY(k))),
    mortar: both(MORTAR),
    mg: both(MG),
    turret: both(TURRET, turretPal),
    crawler: CRAWLER.map((rows) => both(rows, bug)),
    dog: DOG.map((rows) => both(rows, dogPal)),
  };
}

/** Barrels, the fuel tank, ammo crates and alien egg pods. */
export function makeProps() {
  const barrel = paintMap([".RRRR.", "RrrrrR", "RRRRRR", "WWWWWW", "RRKRRR", "RRRRRR", "RRRRRR", ".RRRR."], { R: "#c8302a", r: "#e8584a", W: "#e8e0d0", K: "#1a1a1a" }, false);
  const fuel = paintMap(
    ["..YYYYYYYYYY..", ".YyyyyyyyyyyY.", "YYYYYYYYYYYYYY", "YKKYYKKYYKKYYK", "KKYYKKYYKKYYKK", "YYYYYYYYYYYYYY", "YYYYYKKYYYYYYY", "YYYYYKKYYYYYYY", "YYYYYYYYYYYYYY", ".YYYYYYYYYYYY.", "..GG......GG.."],
    { Y: "#e8b830", y: "#ffd860", K: "#1a1a1a", G: "#3a3f4f" },
    false
  );
  const ammo = paintMap(["GGGGGGGG", "GgggggGG", "GYYYYYYG", "GGGGGGGG", "GGgGGgGG", "GGGGGGGG"], { G: "#4a5a30", g: "#6a7a48", Y: "#ffcc33" }, false);
  const egg = paintMap(
    ["...EEE..", "..EeeEE.", ".EeEEEEE", ".EEEEGEE", "EEEEEEEEE", "EEEGEEEEE", "EEEEEEEGE", ".EEEEEEE.", "..dddddd."],
    { E: "#7a3a8a", e: "#b87ad0", G: "#8aff6a", d: "#3a1a4a" },
    false
  );
  return { barrel, fuel, ammo, egg };
}

/* ------------------------------------------------------------------ */
/* Vehicles and bosses (drawn live: their guns aim, their legs walk)   */
/* ------------------------------------------------------------------ */

/** A boxy drawing helper, mirrored when `flip`. */
function painter(c: Ctx, x: number, y: number, flip: boolean) {
  return (a: number, b: number, w: number, h: number, color: string) => {
    c.fillStyle = color;
    c.fillRect(flip ? x - a - w : x + a, y + b, w, h);
  };
}

/** A helicopter centred on (x, y): `enemy` ones are dark with a red stripe, the squad's is olive. */
export function drawHeli(c: Ctx, x: number, y: number, t: number, enemy: boolean, flip = false) {
  const body = enemy ? "#3a4050" : "#5a6b3a";
  const dark = enemy ? "#262a36" : "#3f4a29";
  const p = painter(c, x, y, flip);
  p(-21, -7, 3, 6, dark);
  p(-20, -2, 13, 2, dark);
  p(-9, -4, 16, 8, INK);
  p(-8, -3, 14, 6, body);
  p(-8, -3, 14, 1, enemy ? "#525a6e" : "#6f8247");
  p(-8, 2, 14, 1, dark);
  if (enemy) p(-8, 0, 14, 1, "#c8302a");
  p(2, -2, 4, 3, enemy ? "#ff8a6a" : "#8fd3ff");
  p(-6, 5, 1, 1, INK);
  p(3, 5, 1, 1, INK);
  p(-8, 6, 13, 1, INK);
  p(-1, -6, 2, 2, INK);
  if (enemy) p(5, 3, 4, 1, INK);
  const blade = Math.round(18 * Math.abs(Math.cos(t * 30))) + 2;
  c.globalAlpha = 0.35;
  p(-20, -7, 41, 1, INK);
  c.globalAlpha = 1;
  p(-blade, -7, blade * 2 + 1, 1, INK);
  if (Math.sin(t * 40) > 0) p(-20, -10, 1, 7, INK);
  else p(-23, -7, 7, 1, INK);
}

/** The boss gunship: twice the size, twin rotors, rocket pods and a chin gun. Faces left. */
export function drawGunship(c: Ctx, x: number, y: number, t: number, hit: boolean) {
  const p = painter(c, x, y, true);
  const body = hit ? "#8a8a9a" : "#2e3442";
  p(-44, -10, 4, 10, "#1e222c");
  p(-42, -3, 26, 4, "#1e222c");
  p(-18, -9, 40, 18, INK);
  p(-17, -8, 38, 16, body);
  p(-17, -8, 38, 2, "#4a5264");
  p(-17, 4, 38, 2, "#1a1e28");
  p(-12, -2, 30, 2, "#c8302a");
  p(12, -6, 8, 6, "#ff8a6a");
  p(13, -6, 3, 2, "#ffd0c0");
  p(-6, 8, 22, 3, INK);
  for (const k of [-2, 10]) {
    p(k, 9, 6, 4, "#3a4050");
    p(k + 6, 10, 2, 2, "#c8302a");
  }
  p(18, 8, 6, 2, INK);
  p(0, -13, 3, 4, INK);
  const blade = Math.round(34 * Math.abs(Math.cos(t * 24))) + 4;
  c.globalAlpha = 0.3;
  p(-36, -14, 76, 1, INK);
  c.globalAlpha = 1;
  p(1 - blade, -14, blade * 2, 2, INK);
  if (Math.sin(t * 40) > 0) p(-43, -18, 2, 12, INK);
  else p(-48, -13, 12, 2, INK);
}

/** A jeep with a gunner on a mounted machine gun. Faces left. */
export function drawJeep(c: Ctx, x: number, y: number, anim: number, look: Backdrop, t: number) {
  const L = LOOKS[look];
  const p = painter(c, x, y, true);
  const body = mix(L.uniform, "#3a3a30", 0.5);
  p(-15, -12, 30, 8, INK);
  p(-14, -11, 28, 6, body);
  p(-14, -11, 28, 1, mix(body, "#ffffff", 0.2));
  p(4, -15, 2, 4, INK);
  p(6, -14, 6, 3, "#8fd3ff");
  // Gunner and gun
  p(-6, -22, 6, 10, INK);
  p(-5, -21, 4, 4, L.helmet);
  p(-5, -17, 4, 5, L.uniform);
  p(-2, -18, 14, 2, INK);
  p(-2, -18, 12, 1, "#4e5464");
  for (const wx of [-10, 9]) {
    p(wx - 3, -6, 7, 7, INK);
    p(wx - 2, -5, 5, 5, "#2a2a30");
    const a = anim * 2;
    p(wx + Math.round(Math.cos(a) * 1.5), -3 + Math.round(Math.sin(a) * 1.5), 1, 1, "#8a8a90");
  }
  void t;
}

/** A fuel tanker truck. Faces left. */
export function drawTanker(c: Ctx, x: number, y: number, lit: boolean) {
  const p = painter(c, x, y, true);
  p(-20, -18, 30, 13, INK);
  p(-19, -17, 28, 11, "#c8c8c0");
  p(-19, -17, 28, 2, "#e8e8e0");
  for (let k = -16; k < 6; k += 6) p(k, -13, 3, 3, "#c8302a");
  p(-14, -10, 18, 1, "#1a1a1a");
  p(10, -14, 10, 9, INK);
  p(11, -13, 8, 7, "#c8302a");
  p(15, -12, 3, 3, "#8fd3ff");
  for (const wx of [-15, -6, 14]) {
    p(wx - 3, -6, 7, 6, INK);
    p(wx - 2, -5, 5, 4, "#2a2a30");
  }
  if (lit) p(-4, -20, 2, 2, "#ffffff");
}

/** A quadcopter drone centred on (x, y): spinning rotors, a gun pod, a blinking light. */
export function drawDrone(c: Ctx, x: number, y: number, t: number, hit: boolean) {
  const p = (a: number, b: number, w: number, h: number, color: string) => {
    c.fillStyle = color;
    c.fillRect(x + a, y + b, w, h);
  };
  p(-5, -2, 10, 4, INK);
  p(-4, -1, 8, 2, hit ? "#c8ccd8" : "#4a5064");
  p(-4, -1, 8, 1, hit ? "#ffffff" : "#6a7288");
  p(-1, 2, 3, 2, INK);
  p(0, 3, 1, 1, "#ff3b3b");
  p(-8, -2, 3, 1, INK);
  p(5, -2, 3, 1, INK);
  const blur = Math.round(3 * Math.abs(Math.cos(t * 40))) + 1;
  c.globalAlpha = 0.6;
  p(-7 - blur, -4, blur * 2 + 1, 1, INK);
  p(6 - blur, -4, blur * 2 + 1, 1, INK);
  c.globalAlpha = 1;
  p(-7, -3, 1, 1, INK);
  p(6, -3, 1, 1, INK);
  if (Math.floor(t * 3) % 2) p(3, -1, 1, 1, "#ff3b3b");
}

/** A troop truck with a canvas-covered back (its flap open once the troops pile out). Faces left. */
export function drawTruck(c: Ctx, x: number, y: number, anim: number, look: Backdrop, open: boolean) {
  const L = LOOKS[look];
  const p = painter(c, x, y, true);
  const body = mix(L.uniform, "#3a3a30", 0.45);
  const cover = mix(L.uniform, "#5a5a40", 0.3);
  // The covered bed (at the back, on the right)
  p(-22, -27, 30, 19, INK);
  p(-21, -26, 28, 17, cover);
  p(-21, -26, 28, 2, mix(cover, "#ffffff", 0.18));
  for (let k = -16; k < 6; k += 6) p(k, -26, 1, 17, mix(cover, "#000000", 0.25));
  if (open) p(-22, -23, 3, 14, "#14121a");
  // Chassis and cab
  p(-23, -9, 46, 4, INK);
  p(-22, -8, 44, 2, body);
  p(8, -21, 14, 13, INK);
  p(9, -20, 12, 11, body);
  p(9, -20, 12, 1, mix(body, "#ffffff", 0.2));
  p(14, -18, 6, 4, "#8fd3ff");
  p(20, -11, 3, 3, "#ffe27a");
  for (const wx of [-16, -8, 14]) {
    p(wx - 4, -6, 8, 7, INK);
    p(wx - 3, -5, 6, 5, "#2a2a30");
    const a = anim * 2;
    p(wx + Math.round(Math.cos(a) * 1.5), -3 + Math.round(Math.sin(a) * 1.5), 1, 1, "#8a8a90");
  }
}

/**
 * A river patrol boat at the water line y: a grey hull with its bow pointing the way it goes, a
 * cabin, a gunner on the stern, a wake. `sink` (0..1) as it goes down.
 */
export function drawBoat(c: Ctx, x: number, y: number, t: number, look: Backdrop, facing: number, sink: number) {
  const L = LOOKS[look];
  const by = Math.round(y + Math.sin(t * 2.4) + sink * 5);
  const p = painter(c, x, by, facing < 0);
  if (sink > 0) c.globalAlpha = Math.max(0, 1 - sink);
  p(-18, -6, 33, 7, INK);
  p(-17, -5, 31, 5, "#3a4250");
  p(-17, -5, 31, 1, "#5a6474");
  for (let k = 0; k < 4; k++) p(15 + k, -6 + k * 2, 1, 7 - k * 2, INK);
  p(-17, -1, 31, 1, "#c8302a");
  p(-2, -12, 10, 7, INK);
  p(-1, -11, 8, 6, "#5a6474");
  p(3, -10, 3, 2, "#8fd3ff");
  // The gunner on the stern
  p(-14, -16, 6, 11, INK);
  p(-13, -15, 4, 4, L.helmet);
  p(-13, -11, 4, 5, L.uniform);
  p(-10, -12, 9, 2, INK);
  p(-10, -12, 8, 1, "#4e5464");
  c.globalAlpha = 1;
  if (!sink) {
    c.fillStyle = "#e8f8ff";
    for (let k = 0; k < 6; k++) if (Math.sin(t * 9 + k) > 0) c.fillRect(facing < 0 ? x + 18 + k * 2 : x - 20 - k * 2, y, 1, 1);
  }
}

/** The siege tank boss: treads, hull, a turret whose long cannon tracks the squad. Faces left. */
export function drawTank(c: Ctx, x: number, y: number, anim: number, aim: number, hit: boolean) {
  const hull = hit ? "#9a9a8a" : "#4a5230";
  const dark = "#2e3420";
  const p = (a: number, b: number, w: number, h: number, color: string) => {
    c.fillStyle = color;
    c.fillRect(x + a, y + b, w, h);
  };
  // Treads
  p(-30, -12, 60, 12, INK);
  p(-29, -11, 58, 10, "#2a2a28");
  for (let k = 0; k < 9; k++) {
    const wx = -25 + k * 6 + ((Math.floor(anim * 4) % 2) ? 1 : 0);
    p(wx, -9, 4, 6, "#4a4a44");
  }
  // Hull
  p(-28, -22, 56, 11, INK);
  p(-27, -21, 54, 9, hull);
  p(-27, -21, 54, 2, mix(hull, "#ffffff", 0.2));
  p(-20, -16, 6, 3, dark);
  p(10, -16, 6, 3, dark);
  // Turret and cannon (pointing toward `aim`, kept to the left half)
  p(-14, -31, 26, 10, INK);
  p(-13, -30, 24, 8, hull);
  p(-13, -30, 24, 2, mix(hull, "#ffffff", 0.25));
  p(4, -34, 6, 4, dark);
  const a = Math.max(Math.PI * 0.75, Math.min(Math.PI * 1.15, aim < 0 ? aim + Math.PI * 2 : aim));
  for (let k = 0; k < 28; k++) {
    const bx = Math.round(-10 + Math.cos(a) * k);
    const by = Math.round(-26 + Math.sin(a) * k);
    p(bx, by - 1, 1, 3, INK);
    p(bx, by, 1, 1, "#3a3a34");
  }
  p(-26, -22, 8, 2, "#c8302a");
}

/** The walker boss: two big stomping legs, a cockpit with a red visor, a gun arm and a rocket pod. Faces left. */
export function drawMech(c: Ctx, x: number, y: number, anim: number, aim: number, hit: boolean, t: number) {
  const metal = hit ? "#a0a4b0" : "#5a6070";
  const dark = "#2e323e";
  const p = (a: number, b: number, w: number, h: number, color: string) => {
    c.fillStyle = color;
    c.fillRect(x + a, y + b, w, h);
  };
  // Legs: each lifts in turn
  for (const side of [0, 1]) {
    const phase = anim * Math.PI + side * Math.PI;
    const lift = Math.max(0, Math.sin(phase)) * 6;
    const fx = -6 + side * 12 + Math.round(Math.cos(phase) * 6);
    const hip = { x: -4 + side * 8, y: -34 };
    const foot = { x: fx, y: -lift };
    const knee = { x: Math.round((hip.x + foot.x) / 2) - 6, y: -18 - Math.round(lift / 2) };
    for (const [a, b] of [
      [hip, knee],
      [knee, foot],
    ] as const) {
      const steps = Math.max(Math.abs(b.x - a.x), Math.abs(b.y - a.y));
      for (let k = 0; k <= steps; k++) {
        const u = k / steps;
        p(Math.round(a.x + (b.x - a.x) * u) - 2, Math.round(a.y + (b.y - a.y) * u) - 2, 5, 5, INK);
      }
      for (let k = 0; k <= steps; k++) {
        const u = k / steps;
        p(Math.round(a.x + (b.x - a.x) * u) - 1, Math.round(a.y + (b.y - a.y) * u) - 1, 3, 3, side ? metal : dark);
      }
    }
    p(foot.x - 6, foot.y - 3, 12, 3, INK);
    p(foot.x - 5, foot.y - 2, 10, 2, metal);
  }
  // Body
  p(-17, -58, 34, 26, INK);
  p(-16, -57, 32, 24, metal);
  p(-16, -57, 32, 3, mix(metal, "#ffffff", 0.25));
  p(-16, -38, 32, 2, dark);
  p(-14, -52, 12, 5, INK);
  p(-13, -51, 10, 3, Math.sin(t * 6) > 0 ? "#ff3b3b" : "#c82a2a");
  p(4, -50, 10, 8, dark);
  p(6, -48, 2, 2, "#ffcc33");
  p(10, -48, 2, 2, "#ffcc33");
  // Rocket pod on top, gun arm in front (aiming)
  p(-8, -64, 18, 7, INK);
  p(-7, -63, 16, 5, dark);
  for (let k = 0; k < 4; k++) p(-6 + k * 4, -62, 2, 2, "#c8302a");
  const a = Math.max(Math.PI * 0.7, Math.min(Math.PI * 1.2, aim < 0 ? aim + Math.PI * 2 : aim));
  for (let k = 0; k < 22; k++) {
    const bx = Math.round(-16 + Math.cos(a) * k);
    const by = Math.round(-40 + Math.sin(a) * k);
    p(bx, by - 1, 2, 4, INK);
    p(bx, by, 2, 2, "#3a3e4a");
  }
}

/** The burrowing worm: segments from the tail forward, a jawed head on top. `segs` in screen space. */
export function drawWorm(c: Ctx, segs: { x: number; y: number; up: boolean }[], t: number, hit: boolean) {
  for (let i = segs.length - 1; i >= 0; i--) {
    const s = segs[i];
    const r = i === 0 ? 9 : 8 - (i / segs.length) * 3;
    c.fillStyle = INK;
    disc(c, Math.round(s.x), Math.round(s.y), r + 1);
    c.fillStyle = hit ? "#e0d0e8" : i % 2 ? "#6a3a7a" : "#7a4a8a";
    disc(c, Math.round(s.x), Math.round(s.y), r);
    c.fillStyle = "#a87ab8";
    c.fillRect(Math.round(s.x - r / 2), Math.round(s.y - r + 1), Math.max(1, Math.round(r)), 2);
  }
  const h = segs[0];
  if (!h) return;
  const n = segs[1] ?? { x: h.x - 1, y: h.y };
  const a = Math.atan2(h.y - n.y, h.x - n.x);
  const open = 3 + Math.round(Math.abs(Math.sin(t * 8)) * 3);
  c.fillStyle = "#ffe27a";
  for (const s of [-1, 1]) {
    const ex = h.x + Math.cos(a + s * 0.6) * 5;
    const ey = h.y + Math.sin(a + s * 0.6) * 5;
    c.fillRect(Math.round(ex), Math.round(ey), 2, 2);
  }
  c.fillStyle = "#e8e0d0";
  for (const s of [-1, 1]) {
    for (let k = 0; k < 5; k++) {
      const jx = h.x + Math.cos(a) * (8 + k) + Math.cos(a + (s * Math.PI) / 2) * (open - k * 0.5);
      const jy = h.y + Math.sin(a) * (8 + k) + Math.sin(a + (s * Math.PI) / 2) * (open - k * 0.5);
      c.fillRect(Math.round(jx), Math.round(jy), 2, 2);
    }
  }
}

/** A jet flying left. */
export function drawJet(c: Ctx, x: number, y: number) {
  const p = (a: number, b: number, w: number, h: number, color: string) => {
    c.fillStyle = color;
    c.fillRect(x + a, y + b, w, h);
  };
  p(-1, -1, 22, 5, INK);
  p(14, -5, 6, 5, INK);
  p(0, 0, 20, 3, "#8a93a8");
  p(1, 0, 19, 1, "#c9ced8");
  p(6, 3, 8, 2, "#5d6578");
  p(15, -4, 3, 4, "#5d6578");
  p(2, 0, 4, 1, "#3d5a80");
  p(-2, 1, 2, 1, "#c8ced9");
}

/** An ICBM pointing along (ux, uy): a white body, red nose band, fins. */
export function drawIcbm(c: Ctx, x: number, y: number, ux: number, uy: number) {
  const px = -uy;
  const py = ux;
  for (let k = 0; k < 22; k++) {
    const cx = x - ux * k;
    const cy = y - uy * k;
    for (const o of [-2, -1, 0, 1, 2]) {
      c.fillStyle = Math.abs(o) === 2 ? INK : k < 4 ? "#c8302a" : k > 17 ? "#3a3f4f" : o === -1 ? "#ffffff" : "#d8dce4";
      c.fillRect(Math.round(cx + px * o), Math.round(cy + py * o), 1, 1);
    }
  }
  for (const s of [-1, 1]) {
    for (let k = 0; k < 4; k++) {
      c.fillStyle = INK;
      c.fillRect(Math.round(x - ux * (18 + k) + px * s * (3 + k)), Math.round(y - uy * (18 + k) + py * s * (3 + k)), 2, 2);
    }
  }
}

/* ------------------------------------------------------------------ */
/* Pixel font                                                          */
/* ------------------------------------------------------------------ */

/** 3×5 glyphs: each is five rows of three bits. */
const GLYPHS: Record<string, string> = {
  "0": "111101101101111",
  "1": "010110010010111",
  "2": "111001111100111",
  "3": "111001111001111",
  "4": "101101111001001",
  "5": "111100111001111",
  "6": "111100111101111",
  "7": "111001010010010",
  "8": "111101111101111",
  "9": "111101111001111",
  A: "010101111101101",
  B: "110101110101110",
  C: "011100100100011",
  D: "110101101101110",
  E: "111100110100111",
  F: "111100110100100",
  G: "011100101101011",
  H: "101101111101101",
  I: "111010010010111",
  J: "001001001101010",
  K: "101101110101101",
  L: "100100100100111",
  M: "101111111101101",
  N: "110101101101101",
  O: "010101101101010",
  P: "110101110100100",
  Q: "010101101110011",
  R: "110101110101101",
  S: "011100010001110",
  T: "111010010010010",
  U: "101101101101111",
  V: "101101101101010",
  W: "101101111111101",
  X: "101101010101101",
  Y: "101101010010010",
  Z: "111001010100111",
  "+": "000010111010000",
  "-": "000000111000000",
  "!": "010010010000010",
  ",": "000000000010100",
  ".": "000000000000010",
  ":": "000010000010000",
  "?": "111001010000010",
  " ": "000000000000000",
};

/** Draw text in the 3×5 pixel font (1 px between letters, scaled by `scale`); returns its width. */
export function pixelText(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string, outline = true, scale = 1) {
  const chars = text.toUpperCase().split("");
  const draw = (ox: number, oy: number, colr: string) => {
    ctx.fillStyle = colr;
    chars.forEach((ch, i) => {
      const g = GLYPHS[ch] ?? GLYPHS[" "];
      for (let b = 0; b < 15; b++) if (g[b] === "1") ctx.fillRect(x + ox + (i * 4 + (b % 3)) * scale, y + oy + Math.floor(b / 3) * scale, scale, scale);
    });
  };
  if (outline)
    for (const [dx, dy] of [
      [-1, 0],
      [1, 0],
      [0, -1],
      [0, 1],
      [1, 1],
      [scale, scale],
    ])
      draw(dx, dy, INK);
  draw(0, 0, color);
  return (chars.length * 4 - 1) * scale;
}

/** Filled pixel circle (scanlines, so it stays crisp). */
export function disc(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number) {
  if (r <= 0) return;
  for (let dy = -Math.floor(r); dy <= Math.floor(r); dy++) {
    const w = Math.floor(Math.sqrt(r * r - dy * dy));
    ctx.fillRect(cx - w, cy + dy, w * 2 + 1, 1);
  }
}

/** One-pixel circle outline (midpoint algorithm). */
export function ring(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number) {
  let x = Math.round(r);
  let y = 0;
  let err = 1 - x;
  while (x >= y) {
    for (const [a, b] of [
      [x, y],
      [y, x],
      [-y, x],
      [-x, y],
      [-x, -y],
      [-y, -x],
      [y, -x],
      [x, -y],
    ])
      ctx.fillRect(cx + a, cy + b, 1, 1);
    y++;
    if (err < 0) err += 2 * y + 1;
    else {
      x--;
      err += 2 * (y - x) + 1;
    }
  }
}
