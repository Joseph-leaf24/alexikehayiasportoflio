/**
 * Procedural pixel art: block sprites, theme backdrops, players, enemies and props, painted once in
 * code (there are no image files), plus the vehicles and bosses, which are drawn live every frame.
 */
import {
  REINFORCED,
  AMETHYST,
  COBBLE,
  COPPER,
  CORAL,
  DIAMOND,
  DRYGRASS,
  FOSSIL,
  GLACIER,
  LATERITE,
  MAGMA,
  MOSSROCK,
  MYCELIUM,
  OBSIDIAN,
  PEAT,
  PERMAFROST,
  QUARTZ,
  ROOTS,
  SALT,
  SCREE,
  SHALE,
  SULFUR,
  TUFF,
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
  BASALT,
  CHALK,
  CLAY,
  GRANITE,
  GRAVEL,
  LOAM,
  PINKLEAF,
  REDLEAF,
  ADOBE,
  AZULEJO,
  JADE,
  LIMESTONE,
  SHOJI,
  SLATE,
  TERRACOTTA,
  TIMBER,
  ASPHALT,
  CARPET,
  COAL,
  DARKBRICK,
  GLASS,
  GOLD,
  IRON,
  MARBLE,
  PARQUET,
  PAVEMENT,
  PLASTER,
  RAIL,
  SEWER,
  SIDING,
  STUCCO,
  TILES,
  FENCE,
  LADDER,
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
  TIN,
  TRUNK,
  VINE,
  WATER,
  WINDOW,
  WOOD,
} from "./level";
import type { Backdrop } from "./themes";
import { rng } from "./rng";
import { fbm1, hash, ridged1 } from "./noise";
import { ASSETS, type AssetId } from "./assets";
import { CATALOG, paintCatalog } from "./catalog";

/** The outline colour used everywhere, on the site and in the scene. */
export const INK = "#0d0b14";
export const VARIANTS = 4;

/** 1 for blocks that get a dark outline where they meet open air; see-through blocks get none. */
export const OUTLINED = new Uint8Array(KINDS).map((_, k) =>
  k === LEAF ||
  k === STEEL ||
  k === TRUNK ||
  k === LAVA ||
  k === PIPE ||
  k === PINE ||
  k === VINE ||
  k === WATER ||
  k === BAMBOO ||
  k === LADDER ||
  k === FENCE ||
  k === RAIL ||
  k === REDLEAF ||
  k === PINKLEAF
    ? 0
    : 1
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
    [LADDER, "#9c6834"],
    [FENCE, "#9aa1b0"],
    [TIN, "#8a90a0"],
    [GLASS, "#8fd3ff"],
    [MARBLE, "#e8e4dc"],
    [PLASTER, "#e0d4b8"],
    [SIDING, "#5a8ab0"],
    [STUCCO, "#e8a8a0"],
    [TILES, "#e8eef0"],
    [DARKBRICK, "#5a3a3a"],
    [ASPHALT, "#3a3a40"],
    [PAVEMENT, "#a8a8b0"],
    [COAL, "#2a2a30"],
    [IRON, "#c8906a"],
    [GOLD, "#ffd23f"],
    [RAIL, "#7a6a5a"],
    [SEWER, "#4a5a4a"],
    [PARQUET, "#b07a40"],
    [CARPET, "#8a3a4a"],
    [TIMBER, "#e8dcc0"],
    [ADOBE, "#c08a5a"],
    [TERRACOTTA, "#c8603a"],
    [JADE, "#3a8a6a"],
    [SLATE, "#4a4e5a"],
    [SHOJI, "#f0e8d8"],
    [AZULEJO, "#3a6ab0"],
    [LIMESTONE, "#e0d8c0"],
    [CLAY, "#b87a5a"],
    [GRAVEL, "#8a8478"],
    [GRANITE, "#7a6a70"],
    [BASALT, "#2e2e36"],
    [LOAM, "#5a3a22"],
    [CHALK, "#e8e4d8"],
    [REDLEAF, "#d8642a"],
    [PINKLEAF, "#f0a0c8"],
    [SHALE, "#525a68"],
    [QUARTZ, "#e8e4ee"],
    [OBSIDIAN, "#2a2036"],
    [MAGMA, "#ff6a1a"],
    [SULFUR, "#e8d040"],
    [PERMAFROST, "#7a6a5a"],
    [GLACIER, "#7ab8e8"],
    [SCREE, "#8a8a8e"],
    [PEAT, "#3a2a1c"],
    [LATERITE, "#b85a30"],
    [SALT, "#f0e8e8"],
    [CORAL, "#e86a7a"],
    [COBBLE, "#8a8a90"],
    [FOSSIL, "#e8dcc0"],
    [COPPER, "#4ab08a"],
    [AMETHYST, "#a85ae8"],
    [MYCELIUM, "#7affd8"],
    [MOSSROCK, "#6a9a3a"],
    [DRYGRASS, "#d8b450"],
    [ROOTS, "#a07a50"],
    [TUFF, "#8a7e72"],
    [DIAMOND, "#aaf4ff"],
    [REINFORCED, "#7a7e88"],
  ];
  for (const [k, col] of set) c[k] = col;
  return c;
})();

/** "#rrggbb" → [r, g, b] */
export const rgb = (hex: string): [number, number, number] => {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};

/** Mix two "#rrggbb" colours into one: t = 0 gives a, 1 gives b. */
export function mix(a: string, b: string, t: number) {
  const x = rgb(a);
  const y = rgb(b);
  return (
    "#" +
    x
      .map((v, i) =>
        Math.round(v + (y[i] - v) * t)
          .toString(16)
          .padStart(2, "0")
      )
      .join("")
  );
}

/**
 * How the scene's canvases are drawn. On the GPU by default; the wallpaper draws them in software
 * (`willReadFrequently`), which costs less for a canvas this small drawn in thousands of little
 * pieces: the GPU then only gets the finished frame, not every piece of it.
 */
export const canvasOptions: CanvasRenderingContext2DSettings = {};

/** A canvas with image smoothing off, and its 2D context. */
export function makeCanvas(w: number, h: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const x = c.getContext("2d", canvasOptions)!;
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
  const speckle =
    (base: string, dark: string, light: string, pd = 0.14, pl = 0.1) =>
    () => {
      const n = r();
      return n < pd ? dark : n < pd + pl ? light : base;
    };
  const dirt = speckle("#8a5a2b", "#6e4520", "#9c6834", 0.12, 0.08);
  const rock = () => {
    const n = r();
    return n < 0.14 ? "#4e4046" : n < 0.22 ? "#2a2226" : n < 0.25 ? "#7a2a14" : "#3a2f33";
  };
  const plate = (x: number, y: number, base: string, light: string, dark: string) =>
    y === 0 || x === 0 ? light : y === 7 || x === 7 ? dark : base;
  const sandstone = (x: number, y: number) =>
    y === 3 || (y === 7 && x % 4 === 0) ? "#a87a48" : speckle("#c89a60", "#b48650", "#dab070")();

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
      all((_, y) => {
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
      all((_, y) => {
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
      all((_, y) => {
        const n = r();
        if (y % 3 === 1 && n < 0.6) return "#c8644a";
        return n < 0.15 ? "#8a3a24" : n < 0.25 ? "#d8805a" : "#b8543a";
      });
      break;
    case MOSS:
      // Temple stone in courses, moss creeping over the top
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
      // The scene paints the sea itself, by depth; this is only its sunlit top, for wherever a block swatch shows
      all((x, y) => ((x + y + variant) % 4 === 0 ? "#3290c0" : "#4aaed2"));
      break;
    case LADDER:
      // Two rails and two rungs, placed so they line up from block to block
      all((x, y) => {
        if (x === 1) return "#b07a40";
        if (x === 6) return "#6e4520";
        if ((y === 1 || y === 5) && x > 1 && x < 6) return y === 1 ? "#9c6834" : "#8a5a2b";
        return null;
      });
      break;
    case FENCE:
      // Chain link on a top rail
      all((x, y) => {
        if (y === 0) return "#5d6578";
        if ((x + y) % 4 === 0 || (x - y + 8) % 4 === 0) return "#9aa1b0";
        return null;
      });
      break;
    case TIN:
      // Corrugated sheet, rusting at the edges
      all((x, y) => {
        if (y === 0) return "#b0b6c4";
        if (r() < 0.07 || (y === 7 && r() < 0.5)) return "#8a4a2a";
        return x % 2 ? "#6a7080" : "#8a90a0";
      });
      break;
    case GLASS:
      // Curtain wall: sky-blue panes with a diagonal streak of light
      all((x, y) => {
        if (x === 0 || y === 0) return "#3a4a5a";
        if (x - y === variant - 1 || x - y === variant) return "#e8f8ff";
        return y < 4 ? "#8fd3ff" : "#5aa0d0";
      });
      break;
    case MARBLE:
      all((x, y) =>
        r() < 0.06 ? "#b8b4ac" : (x + y * 2 + variant) % 11 === 0 ? "#c8c4bc" : y === 0 ? "#ffffff" : "#e8e4dc"
      );
      break;
    case PLASTER:
      all(speckle("#e0d4b8", "#c8bca0", "#f0e8d0", 0.08, 0.08));
      break;
    case SIDING: {
      // Painted weatherboard, a different colour per variant
      const tone = ["#5a8ab0", "#c8a050", "#6aa070", "#c86a5a"][variant];
      all((_, y) => (y % 3 === 2 ? mix(tone, "#000000", 0.3) : y % 3 === 0 ? mix(tone, "#ffffff", 0.2) : tone));
      break;
    }
    case STUCCO:
      all(speckle("#e8a8a0", "#d09088", "#f8c0b8", 0.1, 0.08));
      break;
    case TILES:
      all((x, y) => (x === 3 || x === 7 || y === 3 || y === 7 ? "#b8c4c8" : "#e8eef0"));
      break;
    case DARKBRICK:
      all((x, y) => {
        const seam = y < 4 ? x === 0 : x === 4;
        if (y === 3 || y === 7 || seam) return "#2a1a1a";
        return r() < 0.2 ? "#4a2a2a" : "#5a3a3a";
      });
      break;
    case ASPHALT:
      // Road; the top row carries the lane markings
      all((x, y) => {
        if (y === 0 && variant % 2 === 0 && x > 1 && x < 6) return "#e8e4dc";
        if (y === 0 && variant === 3 && x > 0 && x < 7) return "#ffcc33";
        const n = r();
        return n < 0.1 ? "#2a2a30" : n < 0.16 ? "#4a4a52" : "#3a3a40";
      });
      break;
    case PAVEMENT:
      all((x, y) => (x === 7 || y === 7 ? "#7a7a84" : y === 0 ? "#c8c8d0" : r() < 0.08 ? "#98989f" : "#a8a8b0"));
      break;
    case COAL:
    case IRON:
    case GOLD: {
      const ore =
        kind === COAL ? ["#1e1e24", "#3a3a44"] : kind === IRON ? ["#c8906a", "#e8b090"] : ["#ffd23f", "#fff6c8"];
      all(() => {
        const n = r();
        return n < 0.22 ? ore[0] : n < 0.3 ? ore[1] : n < 0.4 ? "#5a5e70" : "#6d7184";
      });
      break;
    }
    case RAIL:
      // Track along the bottom of the block: sleepers and a steel rail
      all((x, y) => (y === 5 ? "#9aa1b0" : y === 6 ? "#5d6578" : y === 7 && x % 3 !== 2 ? "#6e4520" : null));
      break;
    case SEWER:
      all((x, y) => {
        const seam = y < 4 ? x === 2 : x === 6;
        if (y === 3 || y === 7 || seam) return "#2a3a2a";
        return r() < 0.15 ? "#3a4a3a" : r() < 0.05 ? "#5a8a4a" : "#4a5a4a";
      });
      break;
    case PARQUET:
      all((x, y) => (((x >> 1) + y) % 4 === 0 ? "#8a5a2b" : y === 0 ? "#d09a5a" : "#b07a40"));
      break;
    case CARPET:
      all((x, y) => (y === 0 ? "#aa5a6a" : (x + y) % 4 === 0 ? "#7a2a3a" : "#8a3a4a"));
      break;
    case CLAY:
      all((_, y) => (y === 2 || y === 6 ? "#a06a4a" : speckle("#b87a5a", "#a06a4a", "#c88a68", 0.1, 0.08)()));
      break;
    case GRAVEL:
      all(() => {
        const n = r();
        return n < 0.25 ? "#6a6458" : n < 0.45 ? "#a8a294" : n < 0.55 ? "#5a544a" : "#8a8478";
      });
      break;
    case GRANITE:
      all(() => {
        const n = r();
        return n < 0.12 ? "#c8a8a8" : n < 0.22 ? "#3a3036" : n < 0.3 ? "#f0e0e0" : "#7a6a70";
      });
      break;
    case BASALT:
      all((x, y) => ((x + y * 3 + variant) % 7 === 0 ? "#1a1a20" : r() < 0.1 ? "#46464e" : "#2e2e36"));
      break;
    case LOAM:
      all(speckle("#5a3a22", "#4a2e18", "#6e4a2c", 0.14, 0.08));
      break;
    case CHALK:
      all((_, y) => (r() < 0.06 ? "#2a2a30" : y % 4 === 3 ? "#d0ccc0" : "#e8e4d8"));
      break;
    case REDLEAF:
    case PINKLEAF: {
      const [a, b, c2] = kind === REDLEAF ? ["#d8642a", "#f0a03a", "#a8401a"] : ["#f0a0c8", "#ffd0e4", "#c870a0"];
      all((_, y) => {
        const n = r();
        if (n < 0.12) return null;
        return n < 0.3 ? b : n < 0.42 ? c2 : y === 0 && n < 0.6 ? b : a;
      });
      break;
    }
    case TIMBER:
      // Half-timbering: dark beams across white plaster, a brace in every other block
      all((x, y) =>
        x === 0 || y === 0 || (variant % 2 === 0 && x === y) ? "#4a3018" : r() < 0.05 ? "#d0c4a8" : "#ece4d0"
      );
      break;
    case ADOBE:
      all((_, y) => (y === 3 || y === 7 ? "#a8744a" : speckle("#c08a5a", "#a8744a", "#d0a070", 0.1, 0.08)()));
      break;
    case TERRACOTTA:
      // Curved roof tiles in rows
      all((x, y) => ((x + (y >> 1) * 2) % 4 === 0 ? "#8a3a20" : y % 2 ? "#c8603a" : "#e0805a"));
      break;
    case JADE:
      // Glazed green tiles
      all((x, y) => ((x + (y >> 1) * 2) % 4 === 0 ? "#1e5a40" : y % 2 ? "#3a8a6a" : "#6ac09a"));
      break;
    case SLATE:
      all((x, y) => (y % 3 === 2 ? "#2a2e38" : (x + (Math.floor(y / 3) % 2) * 2) % 4 === 0 ? "#3a3e4a" : "#4a4e5a"));
      break;
    case SHOJI:
      // Paper screens in a wooden lattice
      all((x, y) => (x === 0 || y === 0 || x === 4 || y === 4 ? "#5a3816" : "#f0e8d8"));
      break;
    case AZULEJO:
      // Blue-and-white painted tiles
      all((x, y) =>
        x === 0 || y === 0 ? "#c8d8e8" : x === y || x === 8 - y ? "#2a4a9a" : (x + y) % 4 === 0 ? "#5a8ad0" : "#f0f4f8"
      );
      break;
    case LIMESTONE:
      all((x, y) =>
        y === 3 || y === 7 || (y < 4 ? x === 1 : x === 5) ? "#b8b098" : r() < 0.06 ? "#c8c0a8" : "#e0d8c0"
      );
      break;
    case SHALE:
      all((_, y) => {
        const n = r();
        const band = (y + (variant & 1)) % 3;
        if (band === 0) return n < 0.15 ? "#6e7888" : "#3e4552";
        if (band === 1) return n < 0.1 ? "#3e4552" : "#58606e";
        return n < 0.12 ? "#6e7888" : "#4c5462";
      });
      break;
    case QUARTZ:
      all((x, y) => {
        const n = r();
        if ((x + y * 2 + variant) % 7 === 0) return n < 0.5 ? "#c8a8c8" : "#b0a8b8";
        if (n < 0.06) return "#ffffff";
        return n < 0.2 ? "#d8d4e0" : "#ece8f2";
      });
      break;
    case OBSIDIAN:
      all((x, y) => {
        const n = r();
        if (x - y === variant * 2 - 3) return "#7a6a9a";
        if (x - y === variant * 2 - 2 && n < 0.6) return "#4a3a6a";
        return n < 0.08 ? "#a898d8" : n < 0.3 ? "#2a2036" : "#1a1422";
      });
      break;
    case MAGMA:
      // Dark crust with glowing cracks
      all((x, y) => {
        const n = r();
        const crack = Math.abs(Math.sin(x * 1.15 + y * 0.75 + variant * 1.7) + Math.sin(y * 1.3 - x * 0.4)) < 0.32;
        if (crack) return n < 0.4 ? "#ffd23f" : n < 0.8 ? "#ff8a1a" : "#ff4a1a";
        return n < 0.15 ? "#4a2a26" : n < 0.25 ? "#1a1012" : "#2e1c1c";
      });
      break;
    case SULFUR:
      all((_, y) => {
        const n = r();
        if (y === 0) return n < 0.5 ? "#fff08a" : "#f0dc50";
        return n < 0.12 ? "#b8a020" : n < 0.2 ? "#fff08a" : n < 0.24 ? "#6a6a50" : "#e8d040";
      });
      break;
    case PERMAFROST:
      all(() => {
        const n = r();
        return n < 0.14 ? "#d8ecf8" : n < 0.24 ? "#9ab8d0" : n < 0.34 ? "#5a4a3c" : n < 0.42 ? "#8a8a90" : "#7a6a5a";
      });
      break;
    case GLACIER:
      all((x, y) => {
        const n = r();
        if ((x * 2 + y + variant * 3) % 9 === 0) return "#2a5a98";
        if (y === (variant + 2) % 8 && n < 0.7) return "#e8f8ff";
        return n < 0.25 ? "#5a9ad8" : n < 0.35 ? "#a8d8f8" : "#7ab8e8";
      });
      break;
    case SCREE:
      all((x, y) => {
        const n = r();
        const gap = (x + (y >> 1) * 3 + variant) % 4 === 0 && (y + x) % 3 === 0;
        if (gap) return "#3e3e44";
        if (y === 0 && n < 0.3) return "#b8b8bc";
        return n < 0.25 ? "#6a6a70" : n < 0.45 ? "#a8a8ac" : n < 0.55 ? "#7a7068" : "#8a8a8e";
      });
      break;
    case PEAT:
      all((x, y) => {
        const n = r();
        if ((y + variant) % 3 === 0 && (x + y) % 4 !== 0 && n < 0.6) return "#5a4a2e";
        return n < 0.2 ? "#2a1e14" : n < 0.3 ? "#4a3624" : "#3a2a1c";
      });
      break;
    case LATERITE:
      all(() => {
        const n = r();
        return n < 0.1 ? "#d8804a" : n < 0.18 ? "#7a3018" : n < 0.34 ? "#a04a28" : "#b85a30";
      });
      break;
    case SALT:
      all((x, y) => {
        const n = r();
        if (y === 0) return n < 0.5 ? "#ffffff" : "#f4eeee";
        if ((x + y + variant) % 5 === 0) return "#e8c0c8";
        return n < 0.12 ? "#ffffff" : n < 0.3 ? "#e0d0d4" : "#f0e8e8";
      });
      break;
    case CORAL:
      all((x, y) => {
        const n = r();
        if ((x * 3 + y * 5 + variant) % 7 === 0) return "#a83a4a";
        if (y === 0 && n < 0.6) return "#ffc0a0";
        return n < 0.3 ? "#f08a5a" : n < 0.4 ? "#ffb0c0" : "#e86a7a";
      });
      break;
    case COBBLE:
      all((x, y) => {
        const sx = y < 4 ? x : (x + 2) % 8;
        if (y === 3 || y === 7 || sx === 3 || sx === 7) return "#4a4a50";
        const n = r();
        if ((y === 0 || y === 4) && n < 0.6) return "#b0b0b6";
        return n < 0.2 ? "#78787e" : (variant + (sx >> 2)) % 2 ? "#8a8a90" : "#9a948c";
      });
      break;
    case FOSSIL:
      // An ammonite spiral on even variants, fish bones on odd ones
      all(() => (r() < 0.12 ? "#988a72" : "#a89a80"));
      if (variant % 2 === 0)
        for (const [x, y] of [
          [3, 2],
          [4, 2],
          [5, 3],
          [5, 4],
          [4, 5],
          [3, 5],
          [2, 4],
          [2, 3],
          [3, 3],
          [4, 4],
        ])
          dot(x, y, "#ece0c4");
      else {
        for (let x = 1; x < 7; x++) dot(x, 4, "#ece0c4");
        for (const x of [2, 3, 4, 5]) {
          dot(x, 3, "#d8ccb0");
          dot(x, 5, "#d8ccb0");
        }
        dot(6, 3, "#ece0c4");
        dot(6, 5, "#ece0c4");
      }
      break;
    case COPPER:
      all(() => {
        const n = r();
        return n < 0.14 ? "#4ab08a" : n < 0.22 ? "#2a7a5a" : n < 0.3 ? "#d8843a" : n < 0.4 ? "#5a5e70" : "#6d7184";
      });
      break;
    case AMETHYST:
      all((x, y) => {
        const d = Math.abs(x - 3.5) + Math.abs(y - 3.5);
        const n = r();
        if (d < 1.6) return n < 0.5 ? "#f0d0ff" : "#c88aff";
        if (d < 3.2) return (x + y + variant) % 3 === 0 ? "#5a2a8a" : n < 0.4 ? "#b06aff" : "#8a3ad0";
        return n < 0.3 ? "#2a2232" : "#3a3044";
      });
      break;
    case MYCELIUM:
      all((x, y) => {
        const n = r();
        if (y === 0) return n < 0.5 ? "#8a4aaa" : "#6a3a8a";
        if ((x * 5 + y * 3 + variant * 7) % 13 === 0) return "#7affd8";
        if (y === 1 && n < 0.5) return "#5a2a6a";
        return n < 0.2 ? "#2a1a2a" : "#3a2a34";
      });
      break;
    case MOSSROCK:
      all((x, y) => {
        const n = r();
        if (y === 0) return n < 0.6 ? "#7aaa4a" : "#5a8a3a";
        if (y === 1 && n < 0.55) return "#4a7a30";
        if ((x + variant) % 5 === 0 && n < 0.4) return "#4a7a30";
        return n < 0.15 ? "#5a5e66" : n < 0.25 ? "#8a8e98" : "#70747e";
      });
      break;
    case DRYGRASS:
      all((_, y) => {
        const n = r();
        if (y === 0) return n < 0.5 ? "#f0d070" : "#d8b450";
        if (y === 1) return n < 0.3 ? "#b89030" : "#d8b450";
        if (y === 2 && n < 0.4) return "#a88a40";
        return dirt();
      });
      break;
    case ROOTS:
      all((x, y) => {
        const cx = 3 + Math.round(Math.sin((y + variant * 2) * 0.8) * 2);
        if (x === cx || (y === (variant * 3 + 2) % 8 && x > cx)) return r() < 0.5 ? "#a07a50" : "#c8a070";
        return dirt();
      });
      break;
    case TUFF:
      all((x, y) => {
        const n = r();
        if ((x * 3 + y * 7 + variant * 5) % 11 === 0) return "#4e4640";
        return n < 0.15 ? "#6a6058" : n < 0.25 ? "#a09486" : "#8a7e72";
      });
      break;
    case DIAMOND:
      all((x, y) => {
        const n = r();
        if ((x === 2 && y === 2) || (x === 5 && y === 5) || (x === 5 && y === 1 && variant % 2)) return "#ffffff";
        if (Math.abs(x - 2) + Math.abs(y - 2) === 1 || Math.abs(x - 5) + Math.abs(y - 5) === 1) return "#7ae8ff";
        return n < 0.1 ? "#3a3e50" : n < 0.4 ? "#5a5e70" : "#6d7184";
      });
      break;
    case REINFORCED:
      // Cast panels, the steel bars showing through where the face is chipped
      all((x, y) => {
        if (x === 0 || y === 0) return "#9aa0ac";
        if (x === 7 || y === 7) return "#4e525c";
        if ((x === 3 || y === 4) && r() < 0.55) return "#5a3e30";
        return r() < 0.12 ? "#6a6e78" : "#7a7e88";
      });
      dot(2 + (variant % 3), 2, "#3a3e48");
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
  const inside = (x: number, y: number) =>
    y <= h - 2 && blobs.some((b) => (x - b.x) ** 2 + (y - b.y) ** 2 <= b.r * b.r);
  const [cv, ctx] = makeCanvas(w, h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      if (!inside(x, y)) continue;
      ctx.fillStyle = y >= h - 3 ? "#b5bfd1" : y > h * 0.6 ? "#d3d9e5" : !inside(x - 1, y - 1) ? "#ffffff" : "#f1efe8";
      ctx.fillRect(x, y, 1, 1);
    }
  return cv;
}

/** Heights of a seamlessly repeating skyline: a sum of whole-period sines, so the ends meet. */
function ridge(w: number, seed: number, floor: number, waves: [number, number][]) {
  const r = rng(seed);
  const phases = waves.map(() => r() * Math.PI * 2);
  return (x: number) =>
    Math.round(
      floor - waves.reduce((s, [k, a], i) => s + a * (0.5 + 0.5 * Math.sin((2 * Math.PI * k * x) / w + phases[i])), 0)
    );
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
function hills(c: Ctx, w: number, h: number, top: (x: number) => number, body: string, edge: string) {
  for (let x = 0; x < w; x++) {
    const t = top(x);
    col(c, x, t, h, body);
    col(c, x, t, t + 2, edge);
  }
}
function palm(c: Ctx, x: number, base: number, tall: number, lean: number, color: string) {
  c.fillStyle = color;
  for (let y = 0; y < tall; y++) c.fillRect(x + Math.round((lean * y * y) / (tall * 6)), base - y, 1, 1);
  const tx = x + Math.round((lean * tall) / 6);
  const ty = base - tall;
  for (const dir of [1, -1])
    for (const dx of [1, 2])
      for (let s = 0; s < 6; s++) c.fillRect(tx + dir * (s + dx), ty + Math.round((s * s) / 9) + (dx - 1), 1, 1);
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

/** A row of round tree crowns along `base`, lit from the top left and solid below; it tiles seamlessly. */
function crowns(
  c: Ctx,
  w: number,
  h: number,
  base: number,
  rmin: number,
  rmax: number,
  body: string,
  edge: string,
  r: () => number
) {
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
    for (let s = -3; s <= 3; s++)
      for (let k = 0; k < ht - Math.abs(s) * 2; k++) box(c, (x + Math.round((s * k) / 5) + w) % w, h - k, 1, 1, color);
  }
}

/** Leaves and vines hanging into the top of the screen, in front of everything. */
function canopyTop(w: number, h: number, color: string, density: number, r: () => number) {
  const [cv, c] = makeCanvas(w, h);
  const depth = ridge(w, 7, 14, [
    [3, 5],
    [7, 4],
    [13, 3],
  ]);
  for (let x = 0; x < w; x++) col(c, x, 0, depth(x), color);
  c.fillStyle = color;
  for (let x = 0; x < w; x += 3 + Math.floor(r() * 4))
    for (const o of [0, -w, w]) disc(c, x + o, depth(x), 2 + Math.floor(r() * 2));
  for (let x = 2; x < w; x += Math.round((5 + r() * 9) / density)) {
    const len = 6 + Math.floor(r() * (h - 8));
    const end = Math.min(h - 1, depth(x) + len);
    col(c, x, depth(x), end, color);
    if (r() < 0.6) box(c, x - 1, end - 2, 3, 2, color);
  }
  return cv;
}

/** The parallax layers of a backdrop, back to front; `top` is a canopy hung across the top of jungle scenes. */
export interface BackdropArt {
  /** The furthest, palest mountains; absent underground */
  peaks?: HTMLCanvasElement;
  far: HTMLCanvasElement;
  mid: HTMLCanvasElement;
  near: HTMLCanvasElement;
  front: HTMLCanvasElement;
  top?: HTMLCanvasElement;
}

interface RangeOpts {
  seed: number;
  /** Where the foot of the range is, in px from the top of the strip, and how high its crests reach */
  base: number;
  height: number;
  /** Ranges across the strip's width; a whole number, so it tiles */
  lumps: number;
  /** Faces turned to the sun, and those in shadow */
  lit: string;
  shade: string;
  /** Snow above this many px from the top, in this colour */
  snow?: string;
  snowline?: number;
  /** The haze colour the range fades to near its foot, and how much */
  haze?: string;
  hazeAmount?: number;
  /** Rock strata across the faces */
  strata?: boolean;
}

/**
 * A seamlessly tiling mountain range painted straight into the pixels: ridged crests lit on the sun
 * side, snow on the tops, strata and haze towards the foot. Returns the crest height at each x.
 */
function range(c: Ctx, w: number, h: number, o: RangeOpts) {
  const T = new Int16Array(w);
  for (let x = 0; x < w; x++) {
    const u = (x / w) * o.lumps;
    const crest =
      ridged1(o.seed, u, 4, 0.5, o.lumps) * 0.8 +
      (fbm1(o.seed + 5, u * 0.5, 2, 0.5, Math.max(1, o.lumps >> 1)) * 0.5 + 0.5) * 0.35;
    T[x] = Math.round(o.base - o.height * Math.min(1, crest));
  }
  const slope = (x: number) => T[(x + 3) % w] - T[(x - 3 + w) % w];
  const img = c.createImageData(w, h);
  const d = img.data;
  const lit = rgb(o.lit);
  const shade = rgb(o.shade);
  const lip = rgb(mix(o.lit, "#ffffff", 0.3));
  const dark = rgb(mix(o.shade, "#000000", 0.18));
  const snowLit = o.snow ? rgb(o.snow) : lit;
  const snowShade = o.snow ? rgb(mix(o.snow, o.shade, 0.4)) : shade;
  const haze = o.haze ? rgb(o.haze) : null;
  const hazeAmount = o.hazeAmount ?? 0.45;
  for (let x = 0; x < w; x++) {
    const top = T[x];
    // A ragged snow edge
    const snowTo =
      o.snow && o.snowline !== undefined && top < o.snowline
        ? top + Math.round((o.snowline - top) * 0.55 + hash(o.seed, x >> 1) * 3)
        : -1;
    for (let y = Math.max(0, top); y < h; y++) {
      const depth = y - top;
      const sunny = slope((x + Math.round(depth * 0.7)) % w) > 0;
      let col = sunny ? lit : shade;
      if (y < snowTo) col = sunny ? snowLit : snowShade;
      else if (
        o.strata &&
        !sunny &&
        ((y + hash(o.seed + 1, x >> 3) * 6) | 0) % 9 === 0 &&
        hash(o.seed + 2, x, y >> 3) < 0.7
      )
        col = dark;
      if (depth === 0 && sunny) col = y < snowTo ? [255, 255, 255] : lip;
      let r = col[0];
      let g = col[1];
      let b = col[2];
      if (haze) {
        // Thicker towards the foot, dithered in whole pixels
        const f = Math.min(1, Math.max(0, (y - (o.base - o.height * 0.6)) / (o.height * 0.8))) * hazeAmount;
        const step = f > (BAYER[(y & 3) * 4 + (x & 3)] + 0.5) / 16 ? Math.ceil(f * 3) / 3 : Math.floor(f * 3) / 3;
        r += (haze[0] - r) * step;
        g += (haze[1] - g) * step;
        b += (haze[2] - b) * step;
      }
      const i = (y * w + x) * 4;
      d[i] = r;
      d[i + 1] = g;
      d[i + 2] = b;
      d[i + 3] = 255;
    }
  }
  c.putImageData(img, 0, 0);
  return (x: number) => T[((Math.round(x) % w) + w) % w];
}

/** A filled disc in a colour. */
const c2disc = (c: Ctx, x: number, y: number, rad: number, color: string) => {
  c.fillStyle = color;
  disc(c, x, y, rad);
};

/** Paint the parallax layers of a theme's backdrop. */
export function makeBackdrop(kind: Backdrop): BackdropArt {
  const FW = 960;
  const FH = 200;
  const PW = 1200;
  const PH = 240;
  const [far, f] = makeCanvas(FW, FH);
  const [mid, m] = makeCanvas(560, 110);
  const [near, n] = makeCanvas(512, 100);
  const [front, fr] = makeCanvas(480, 40);
  let top: HTMLCanvasElement | undefined;
  let peaks: HTMLCanvasElement | undefined;
  const r = rng(kind.length * 131 + kind.charCodeAt(0));
  const seed = kind.length * 977 + kind.charCodeAt(1) * 31;

  /** The furthest mountains: tall, pale, mostly haze. */
  const distant = (lit: string, shade: string, haze: string, snow?: string) => {
    const [cv, pc] = makeCanvas(PW, PH);
    range(pc, PW, PH, {
      seed: seed + 1,
      base: PH,
      height: 190,
      lumps: 3,
      lit,
      shade,
      haze,
      hazeAmount: 0.7,
      snow,
      snowline: 90,
    });
    peaks = cv;
  };
  /** The main range behind the fight. */
  const mountains = (lit: string, shade: string, o: Partial<RangeOpts> = {}) =>
    range(f, FW, FH, { seed: seed + 2, base: FH, height: 150, lumps: 4, lit, shade, strata: true, ...o });
  const rolling = (c: Ctx, w: number, h: number, sd: number, body: string, edge: string, amp = 1) =>
    hills(
      c,
      w,
      h,
      ridge(w, sd, h - 8, [
        [2, 14 * amp],
        [3, 10 * amp],
        [7, 6 * amp],
        [11, 3 * amp],
      ]),
      body,
      edge
    );
  const rubble = (c: Ctx, w: number, h: number, color: string, height = 12) => {
    const t = ridge(w, 99, h - 2, [
      [5, height],
      [11, height / 2],
      [23, height / 3],
    ]);
    for (let x = 0; x < w; x++) col(c, x, t(x), h, color);
  };

  switch (kind) {
    case "jungle": {
      distant("#b88aa8", "#8a6a94", "#e8a07a", "#f0d8e0");
      mountains("#8aa0c4", "#5a6a98", { snow: "#eef2fa", snowline: 92, haze: "#d89a8a" });
      rolling(m, 560, 110, 33, "#4b7a52", "#5a8a5e", 1.5);
      for (let i = 0; i < 18; i++)
        palm(m, 6 + Math.floor(r() * 548), 92, 16 + Math.floor(r() * 14), r() < 0.5 ? -1 : 1, "#335a40");
      rolling(n, 512, 100, 34, "#24402e", "#2e5038", 1.6);
      for (let i = 0; i < 12; i++)
        palm(n, Math.floor(r() * 500), 80, 24 + Math.floor(r() * 16), r() < 0.5 ? -1 : 1, "#1a3022");
      rubble(fr, 480, 40, "#10200f", 14);
      break;
    }
    case "deepjungle": {
      // Misty peaks with waterfalls, then rainforest hung with vines
      distant("#9ab8a8", "#7a9a8e", "#e0ecd0");
      const crest = mountains("#7a9e8a", "#4e6e62", { haze: "#c8dcc0", hazeAmount: 0.55 });
      for (let i = 0; i < 6; i++) {
        const x = 30 + Math.floor(r() * (FW - 60));
        for (let y = crest(x) + 8; y < FH; y++) if (y % 4) box(f, x, y, 2, 1, "#e0f0ec");
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
      // Temples on the ridges, broken pillars up close
      distant("#c09aa8", "#9a7a90", "#f0b890");
      const crest = mountains("#b08e98", "#7a5a72", { height: 120, haze: "#e0a890" });
      for (const x of [180, 640]) stepped(f, x, crest(x) + 10, 4, 60, 7, "#8a6a7a", "#a8889a");
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
      distant("#5a6a74", "#46545e", "#8a9aa4");
      mountains("#4e6068", "#36464e", { height: 110, haze: "#7a8a94", hazeAmount: 0.6 });
      box(f, 0, FH - 40, FW, 12, "rgba(160,180,190,0.18)");
      crowns(m, 560, 110, 80, 7, 12, "#26383a", "#30464a", r);
      for (let i = 0; i < 10; i++)
        palm(m, Math.floor(r() * 550), 84, 18 + Math.floor(r() * 12), r() < 0.5 ? -1 : 1, "#1e2e30");
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
        // Mesas, then dunes, oil derricks and cactus
        distant("#f0caa0", "#d8aa80", "#fff0d0");
        mountains("#ecc890", "#c09060", { height: 110, lumps: 5, haze: "#f8e0b0" });
        for (let i = 0; i < 6; i++) {
          const x = Math.floor(r() * (FW - 60));
          const h = 30 + Math.floor(r() * 30);
          box(f, x, FH - h, 30 + Math.floor(r() * 30), h, "#d0a070");
          box(f, x, FH - h, 30, 2, "#e0b880");
        }
        hills(
          m,
          560,
          110,
          ridge(560, 42, 100, [
            [2, 26],
            [3, 16],
            [7, 6],
          ]),
          "#c89a60",
          "#d8aa70"
        );
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
        rolling(n, 512, 100, 43, "#9a6a3a", "#aa7a4a", 1.4);
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
        // Red peaks, then buttes and mesas with striped walls
        distant("#e8a890", "#c07a6a", "#ffd0a8");
        mountains("#d88a6a", "#a05a48", { height: 90, lumps: 6, haze: "#f0b090", hazeAmount: 0.35 });
        for (let x = 0; x < FW; ) {
          const w = 30 + Math.floor(r() * 60);
          const h = 30 + Math.floor(r() * 60);
          box(f, x, FH - h, w, h, "#d8987a");
          for (let y = FH - h + 6; y < FH; y += 9) box(f, x, y, w, 2, "#c8846a");
          box(f, x, FH - h, w, 2, "#f0b090");
          x += w + Math.floor(r() * 80);
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
      distant("#d8e4f4", "#a8b8d4", "#eef2fa", "#ffffff");
      mountains("#c8d4e8", "#7a8cac", {
        height: 170,
        lumps: 5,
        snow: "#ffffff",
        snowline: 150,
        haze: "#e0e8f4",
        hazeAmount: 0.35,
      });
      for (let x = 0; x < 560; x += 6 + Math.floor(r() * 8))
        pineTree(m, x, 100, 18 + Math.floor(r() * 16), "#4a6a7a", "#e8eef6");
      box(m, 0, 98, 560, 12, "#d8e4f0");
      for (let x = 0; x < 512; x += 10 + Math.floor(r() * 12))
        pineTree(n, x, 92, 30 + Math.floor(r() * 24), "#1e3a44", "#c8d8e8");
      box(n, 0, 90, 512, 10, "#e8eef6");
      rubble(fr, 480, 40, "#e8eef6", 12);
      break;
    }
    case "fort":
    case "silo": {
      const fort = kind === "fort";
      if (fort) distant("#8a94b4", "#6a7494", "#a0a0c0", "#c8d0e4");
      else distant("#d8a090", "#b07a70", "#f0b8a0");
      mountains(fort ? "#98a8c0" : "#d08a70", fort ? "#5a6a88" : "#9a5a4a", {
        height: 130,
        haze: fort ? "#8a90b0" : "#e0a088",
        snow: fort ? "#dce4f0" : undefined,
        snowline: 80,
      });
      box(m, 0, 76, 560, 34, "#3c4660");
      for (let x = 0; x < 560; x += 8) box(m, x, 72, 4, 4, "#3c4660");
      for (let i = 0; i < 7; i++) {
        const x = Math.floor(r() * 520);
        const h = 30 + Math.floor(r() * 30);
        box(m, x, 76 - h, 18, h, "#343d55");
        for (let k = 0; k < 18; k += 6) box(m, x + k, 72 - h, 4, 4, "#343d55");
        box(m, x + 7, 82 - h, 3, 5, "#ffd479");
      }
      if (!fort) {
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
      // Smog-brown hills behind a skyline of stacks
      distant("#6a5060", "#54404e", "#8a6a6a");
      box(f, 0, FH - 26, FW, 26, "#3a2a3a");
      for (let x = 0; x < FW; ) {
        const w = 14 + Math.floor(r() * 30);
        const h = 20 + Math.floor(r() * 60);
        box(f, x, FH - h - 20, w, h + 20, "#3a2a3a");
        if (r() < 0.5) box(f, x + 4, FH - h - 56, 5, 38, "#3a2a3a");
        for (let y = FH - h - 14; y < FH - 24; y += 7)
          if (r() < 0.5) box(f, x + 3 + Math.floor(r() * (w - 6)), y, 2, 2, "#ffb84a");
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
      // Dark hills ringing the city, then towers, then the shelled blocks
      distant("#2a2e48", "#22253c", "#3a3a5a");
      for (let x = 0; x < FW; ) {
        const w = 16 + Math.floor(r() * 26);
        const h = 40 + Math.floor(r() * 80);
        box(f, x, FH - h, w, h, "#1a1e34");
        for (let k = 0; k < w; k += 3) box(f, x + k, FH - h - Math.floor(r() * 4), 3, 4, "#1a1e34");
        for (let y = FH - h + 4; y < FH - 4; y += 6)
          for (let k = 3; k < w - 3; k += 5) if (r() < 0.14) box(f, x + k, y, 2, 3, "#ffd479");
        x += w + 2 + Math.floor(r() * 6);
      }
      for (let x = 0; x < 560; ) {
        const w = 24 + Math.floor(r() * 30);
        const h = 40 + Math.floor(r() * 50);
        box(m, x, 110 - h, w, h, "#12152a");
        // Shell holes and the odd burning window
        for (let k = 0; k < 3; k++)
          box(m, x + Math.floor(r() * (w - 6)), 110 - h + Math.floor(r() * (h - 10)), 5, 4, "#1e2440");
        if (r() < 0.5) box(m, x + 4 + Math.floor(r() * (w - 8)), 110 - h + 8, 3, 3, "#ff8a3a");
        x += w + 4 + Math.floor(r() * 16);
      }
      rubble(n, 512, 100, "#0c0e1c", 16);
      rubble(fr, 480, 40, "#08090f", 14);
      break;
    }
    case "volcano": {
      // Black volcanic peaks with lava running down their flanks
      distant("#4a1a1e", "#3a1216", "#8a2a1a");
      const crest = mountains("#4a1c20", "#2a1014", { height: 170, lumps: 3, haze: "#6a2016", hazeAmount: 0.3 });
      // Lava runs a little way down from the higher crests, cooling as it goes
      for (let x = 20; x < FW; x += 60 + Math.floor(r() * 50)) {
        const y0 = crest(x) + 3;
        if (y0 > FH - 70) continue;
        const len = 24 + Math.floor(r() * 46);
        let lx = x;
        for (let y = y0; y < Math.min(FH, y0 + len); y++) {
          if (y % 5 === 0) lx += r() < 0.5 ? -1 : 1;
          box(f, lx, y, 1, 1, y < y0 + 5 ? "#ffd23f" : y < y0 + len * 0.6 ? "#d8481a" : "#7a2414");
        }
      }
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
      distant("#3a1a54", "#2a1240", "#5a2a6a");
      for (let x = 0; x < FW; ) {
        const w = 10 + Math.floor(r() * 20);
        const h = 40 + Math.floor(r() * 110);
        for (let k = 0; k < w; k++)
          col(f, x + k, FH - h + Math.round((Math.abs(k - w / 2) / (w / 2)) ** 2 * 30), FH, "#2a1440");
        for (let k = 0; k < 4; k++)
          box(f, x + Math.floor(r() * w), FH - h + 20 + Math.floor(r() * (h - 30)), 2, 2, "#c84aff");
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
      distant("#7a9488", "#62807a", "#a8c0b0");
      mountains("#6a8a74", "#4a6a5a", { height: 90, lumps: 5, haze: "#98b0a0", hazeAmount: 0.6 });
      for (let x = 0; x < 560; x += 12 + Math.floor(r() * 20))
        deadTree(m, x, 104, 30 + Math.floor(r() * 30), "#2e4436", r);
      box(m, 0, 100, 560, 10, "#2e4436");
      for (let x = 0; x < 512; x += 20 + Math.floor(r() * 30))
        deadTree(n, x, 96, 40 + Math.floor(r() * 40), "#16241c", r);
      box(n, 0, 92, 512, 8, "#16241c");
      for (let x = 0; x < 480; x += 2) col(fr, x, 26 - ((x * 7) % 11), 40, "#0c160e");
      break;
    }
    case "savanna": {
      // Golden grassland and flat-topped acacias
      distant("#e8c890", "#c8a070", "#fff0d0");
      mountains("#d8a868", "#a87848", { height: 100, lumps: 5, haze: "#ffe0b0" });
      rolling(m, 560, 110, 51, "#c8a050", "#d8b868", 1.2);
      for (let i = 0; i < 10; i++) {
        const x = 20 + Math.floor(r() * 520);
        col(m, x, 72, 96, "#5a4028");
        box(m, x - 10, 68, 21, 3, "#6a7a30");
        box(m, x - 7, 66, 15, 2, "#7a8a3a");
      }
      rolling(n, 512, 100, 52, "#8a7040", "#9a8050", 1.4);
      for (let x = 0; x < 512; x += 3) col(n, x, 74 - Math.floor(r() * 6), 82, "#7a6034");
      rubble(fr, 480, 40, "#4a3a20", 10);
      break;
    }
    case "autumn": {
      distant("#9a8aa8", "#7a6a90", "#e8c0a0");
      mountains("#a8a0b8", "#6a6080", { height: 120, haze: "#e8b090" });
      crowns(m, 560, 110, 70, 8, 14, "#c86a2a", "#e08a3a", r);
      for (let x = 0; x < 560; x += 13 + Math.floor(r() * 20)) c2disc(m, x, 66 + Math.floor(r() * 10), 6, "#e8a040");
      crowns(n, 512, 100, 62, 10, 18, "#8a3a1a", "#a84a22", r);
      rubble(fr, 480, 40, "#3a1e10", 12);
      ferns(fr, 480, 40, "#3a1e10", r);
      break;
    }
    case "sakura": {
      distant("#c8b8d8", "#a898c0", "#ffe8f0", "#ffffff");
      mountains("#b8b0d0", "#8078a8", { snow: "#ffffff", snowline: 80, haze: "#ffd8e8" });
      crowns(m, 560, 110, 72, 8, 13, "#f0a8c8", "#ffd0e0", r);
      for (const x of [140, 400]) stepped(m, x, 74, 3, 40, 6, "#5a4a5a", "#7a6a7a");
      crowns(n, 512, 100, 64, 10, 16, "#c8709a", "#e890b8", r);
      rubble(fr, 480, 40, "#3a2a30", 10);
      break;
    }
    case "tundra": {
      distant("#c8d4e4", "#a8b8cc", "#f4f8ff", "#ffffff");
      mountains("#d8e0ec", "#9aa8bc", { height: 70, lumps: 6, snow: "#ffffff", snowline: 160, haze: "#eef2f8" });
      box(m, 0, 96, 560, 14, "#e8eef6");
      for (let x = 0; x < 560; x += 30 + Math.floor(r() * 40))
        pineTree(m, x, 98, 14 + Math.floor(r() * 10), "#3a5a6a", "#e8eef6");
      rolling(n, 512, 100, 53, "#e8eef6", "#ffffff", 0.6);
      for (let x = 0; x < 512; x += 40 + Math.floor(r() * 50))
        pineTree(n, x, 92, 24 + Math.floor(r() * 12), "#1e3a44", "#c8d8e8");
      rubble(fr, 480, 40, "#e8eef6", 10);
      break;
    }
    case "badlands": {
      // Banded buttes and hoodoos
      distant("#e8a888", "#c07a68", "#ffd0a8");
      for (let x = 0; x < FW; ) {
        const w = 40 + Math.floor(r() * 70);
        const h = 40 + Math.floor(r() * 80);
        box(f, x, FH - h, w, h, "#c8805a");
        for (let y = FH - h + 4; y < FH; y += 7) box(f, x, y, w, 3, y % 14 < 7 ? "#e0a070" : "#a8603a");
        box(f, x, FH - h, w, 2, "#f0c090");
        x += w + Math.floor(r() * 40);
      }
      for (let x = 0; x < 560; ) {
        const w = 8 + Math.floor(r() * 12);
        const h = 40 + Math.floor(r() * 50);
        for (let k = 0; k < w; k++) col(m, x + k, 110 - h + Math.abs(k - w / 2), 110, "#8a4a2e");
        box(m, x - 2, 110 - h - 3, w + 4, 4, "#a8603a");
        x += w + 16 + Math.floor(r() * 30);
      }
      rolling(n, 512, 100, 54, "#6a3420", "#7a4028", 1.2);
      rubble(fr, 480, 40, "#3a1a0e", 12);
      break;
    }
    case "farmland": {
      // A patchwork of fields and hedgerows
      distant("#a8c0d8", "#8aa0c0", "#e8f4ff");
      mountains("#90b0a0", "#6a8a7a", { height: 80, haze: "#d8ecf0" });
      const fields = ridge(560, 55, 100, [
        [2, 14],
        [3, 8],
        [7, 4],
      ]);
      for (let x = 0; x < 560; x++) {
        const top = fields(x);
        col(
          m,
          x,
          top,
          110,
          Math.floor(x / 46) % 3 === 0 ? "#d8c060" : Math.floor(x / 46) % 3 === 1 ? "#7ab050" : "#9ac868"
        );
        col(m, x, top, top + 1, "#b8d880");
      }
      for (let x = 0; x < 560; x += 46) col(m, x, fields(x) - 2, 110, "#3a6a2a");
      rolling(n, 512, 100, 56, "#4a8a3a", "#5a9a48", 1.1);
      crowns(n, 512, 100, 74, 6, 10, "#2e6a2a", "#3a7a34", r);
      ferns(fr, 480, 40, "#1e3a14", r);
      break;
    }
    case "megacity": {
      // Towers of lit windows and neon
      distant("#2a2040", "#201a34", "#3a2a50");
      for (let x = 0; x < FW; ) {
        const w = 12 + Math.floor(r() * 26);
        const h = 70 + Math.floor(r() * 120);
        box(f, x, FH - h, w, h, "#16142a");
        if (r() < 0.4) col(f, x + (w >> 1), FH - h - 14, FH - h, "#16142a");
        for (let y = FH - h + 3; y < FH - 3; y += 4)
          for (let k = 2; k < w - 2; k += 3) if (r() < 0.35) box(f, x + k, y, 1, 2, r() < 0.15 ? "#ff6fae" : "#ffd479");
        x += w + 1 + Math.floor(r() * 4);
      }
      for (let x = 0; x < 560; ) {
        const w = 20 + Math.floor(r() * 30);
        const h = 50 + Math.floor(r() * 50);
        box(m, x, 110 - h, w, h, "#0e0c1c");
        for (let y = 110 - h + 4; y < 108; y += 5)
          for (let k = 3; k < w - 3; k += 4) if (r() < 0.25) box(m, x + k, y, 2, 2, r() < 0.2 ? "#5ff0e0" : "#ffb84a");
        if (r() < 0.5) box(m, x + 3, 110 - h + 6, w - 6, 3, r() < 0.5 ? "#ff6fae" : "#5ab0ff");
        x += w + 3 + Math.floor(r() * 10);
      }
      rubble(n, 512, 100, "#08080f", 14);
      rubble(fr, 480, 40, "#050508", 12);
      break;
    }
    case "airbase": {
      // Hangars and a control tower along the field, then the perimeter fence
      distant("#9ab0c8", "#7a90a8", "#d8e8f4");
      mountains("#8aa0b8", "#5a7088", { height: 90, haze: "#c8d8e8" });
      box(m, 0, 92, 560, 18, "#4a5a48");
      for (let x = 20; x < 560; x += 110 + Math.floor(r() * 40)) {
        for (let k = -24; k <= 24; k++) col(m, x + k, 92 - Math.round(Math.sqrt(576 - k * k) * 0.8), 92, "#3a4450");
        box(m, x - 8, 80, 16, 12, "#2a3038");
      }
      box(m, 300, 50, 8, 42, "#3a4450");
      box(m, 294, 44, 20, 8, "#3a4450");
      box(m, 296, 46, 16, 3, "#8fd3ff");
      box(n, 0, 84, 512, 16, "#2e3a2c");
      for (let x = 0; x < 512; x += 10) col(n, x, 74, 84, "#3a3f4f");
      for (let x = 0; x < 512; x += 2) box(n, x, 76 + (x % 4 === 0 ? 0 : 3), 1, 1, "#5d6578");
      rubble(fr, 480, 40, "#1a2018", 8);
      break;
    }
    case "archipelago": {
      // Sea out to the horizon with islands on it, palms close by
      distant("#a8d8f0", "#88c0e0", "#e8fcff");
      box(f, 0, FH - 50, FW, 50, "#3a8ac8");
      box(f, 0, FH - 50, FW, 2, "#8ad0f0");
      for (let i = 0; i < 6; i++) {
        const x = Math.floor(r() * (FW - 80));
        const w = 30 + Math.floor(r() * 60);
        for (let k = 0; k < w; k++)
          col(
            f,
            x + k,
            FH - 50 - Math.round(Math.sin((k / w) * Math.PI) * (8 + r() * 0.5 + w / 8)),
            FH - 49,
            "#5a9a6a"
          );
      }
      box(m, 0, 90, 560, 20, "#2a7ab8");
      for (let i = 0; i < 4; i++) {
        const x = Math.floor(r() * 480);
        for (let k = 0; k < 60; k++) col(m, x + k, 90 - Math.round(Math.sin((k / 60) * Math.PI) * 14), 91, "#3a7a4a");
        for (let j = 0; j < 3; j++)
          palm(m, x + 12 + j * 14, 82, 14 + Math.floor(r() * 8), r() < 0.5 ? -1 : 1, "#2a5a34");
      }
      box(n, 0, 88, 512, 12, "#1e5a8a");
      for (let i = 0; i < 8; i++)
        palm(n, Math.floor(r() * 500), 88, 24 + Math.floor(r() * 14), r() < 0.5 ? -1 : 1, "#1a3a22");
      rubble(fr, 480, 40, "#c8a870", 8);
      break;
    }
    case "mangrove": {
      // Mangroves standing on their roots in the water
      distant("#7aa090", "#5a8070", "#b8d8c0");
      mountains("#6a9080", "#4a6a5a", { height: 60, lumps: 6, haze: "#a8c8b0", hazeAmount: 0.6 });
      box(m, 0, 96, 560, 14, "#3a6a5a");
      for (let x = 0; x < 560; x += 18 + Math.floor(r() * 20)) {
        for (let k = -4; k <= 4; k += 2) col(m, x + k, 88, 98, "#2a3a2a");
        c2disc(m, x, 82, 9, "#2e5a3a");
      }
      crowns(m, 560, 110, 84, 6, 10, "#2e5a3a", "#3e6a4a", r);
      box(n, 0, 90, 512, 10, "#244a3a");
      for (let x = 0; x < 512; x += 26 + Math.floor(r() * 26)) {
        for (let k = -6; k <= 6; k += 3) col(n, x + k, 76, 92, "#16241a");
        c2disc(n, x, 66, 14, "#16301e");
      }
      for (let x = 0; x < 480; x += 3) col(fr, x, 26 - ((x * 5) % 9), 40, "#0c160e");
      break;
    }
    default: {
      // Cave: a rock wall, stalagmites and columns
      box(f, 0, 0, FW, FH, "#141220");
      for (let i = 0; i < 260; i++)
        box(f, Math.floor(r() * FW), Math.floor(r() * FH), 3 + Math.floor(r() * 8), 2 + Math.floor(r() * 4), "#1c1a2c");
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
  return { peaks, far, mid, near, front, top };
}

/* ------------------------------------------------------------------ */
/* People                                                              */
/* ------------------------------------------------------------------ */

/** Rows 0–5 of the map: the eight players' heads, then the enemies' masked one and the sniper's hood. */
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
/** Weapons drawn over the torso, rows 6–9. */
const GUNS = [
  ["", ".......SWWWWWW", ".......SWwW", ".........W"], // rifle
  ["", ".......SWWWWW", "......SWWwWW"], // shotgun
  ["", ".......SWWWXXF", ".......SWWWW", "........W"], // plasma
  ["WWWWWWWWWWWWWR", ".......S", "", ""], // rocket tube on the shoulder
  [".........www", ".......SWWWWWW", "........W"], // sniper rifle with scope
  ["YY", "YY.....SWWWXF", "YY.....SW", "YY"], // flamethrower and its tank
  ["", ".......SWWWWW", "......SWOOWW", "........WW"], // grenade launcher with a drum
  ["", "......SWWwWwWw", "......SWWWWWWW", ".....SSWWwWwW"], // minigun
  ["", ".......SWWWVVVv", ".......SWWW", "........W"], // laser
  ["........XX", ".......SWWXwX", ".......SWWX", "........W"], // tesla gun with its coil
  ["", ".......SWW", "", ".......SWW"], // dual pistols
  ["..........w", ".......SWWWWWwR", "..........w", ""], // crossbow
  ["", ".......SWWOOO", ".......SWWOwO", "..........OOO"], // saw-blade launcher
  ["...........WW", ".......SWWWW", "......SWW", ""], // mortar tube
  ["", ".......SWWVVVV", ".......SWWWw", "........W"], // pulse laser
  [".........XXX", ".......SWWWXXXV", ".......SWWWWWW", ".......WW...W"], // death ray
  ["", ".......SWWWXVX", ".......SWWWW", "........W"], // prism laser
];
/** Legs, rows 12–17: idle, four running frames, jumping. */
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

/** Paint a character map, rows of palette letters, with a 1 px ink outline, facing right or left. */
function paintMap(rows: string[], pal: Pal, flip: boolean) {
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const grid = rows.map((r) => r.padEnd(w, "."));
  const [cv, ctx] = makeCanvas(w + 2, h + 2);
  const at = (x: number, y: number) => {
    if (y < 0 || y >= h || x < 0 || x >= w) return false;
    const ch = grid[y][flip ? w - 1 - x : x];
    return ch !== "." && !!pal[ch];
  };
  ctx.fillStyle = INK;
  for (let y = -1; y <= h; y++)
    for (let x = -1; x <= w; x++)
      if (!at(x, y) && (at(x - 1, y) || at(x + 1, y) || at(x, y - 1) || at(x, y + 1))) ctx.fillRect(x + 1, y + 1, 1, 1);
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

/** The palette for player `id`, 0–7, in `color`. */
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
  };
}

/** One player's 16×20 outlined sprite holding `weapon`, which defaults to their own. */
export function playerSprite(id: number, color: string, frame = 0, flip = false, weapon = id) {
  const pal = playerPalette(id, color);
  // The flamethrower's back tank is red
  if (weapon === 5) pal.Y = "#c8302a";
  return paintMap(bodyMap(HEADS[id], GUNS[weapon], LEGS[frame]), pal, flip);
}

/** Each player's frames with each weapon, painted the first time they're needed: get(id, weapon)[frame][facing]. */
export function makeLoadouts(colors: string[]) {
  const cache = new Map<number, HTMLCanvasElement[][]>();
  return (id: number, weapon: number) => {
    const key = id * 32 + weapon;
    let frames = cache.get(key);
    if (!frames)
      cache.set(
        key,
        (frames = LEGS.map((_, f) => [
          playerSprite(id, colors[id], f, false, weapon),
          playerSprite(id, colors[id], f, true, weapon),
        ]))
      );
    return frames;
  };
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
  savanna: { helmet: "#6a5a30", uniform: "#b89858", legs: "#6a5030" },
  autumn: { helmet: "#4a3a20", uniform: "#7a5a30", legs: "#3a2a18" },
  sakura: { helmet: "#3a3a4a", uniform: "#5a5a7a", legs: "#2a2a3a" },
  tundra: { helmet: "#c8d0dc", uniform: "#e8ecf2", legs: "#8a94a4" },
  badlands: { helmet: "#7a4a30", uniform: "#b8704e", legs: "#5a3020" },
  farmland: { helmet: "#3a4a2a", uniform: "#5a6a3a", legs: "#2a3220" },
  megacity: { helmet: "#101018", uniform: "#2a2a34", legs: "#18181e" },
  airbase: { helmet: "#3a4a5a", uniform: "#6a7a8a", legs: "#3a4450" },
  archipelago: { helmet: "#3a5a4a", uniform: "#5a8a6a", legs: "#2a4a3a" },
  mangrove: { helmet: "#2e3a24", uniform: "#4e5a3a", legs: "#2a3220" },
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
  ...(legs
    ? ["...LLLL.LLLL", "...LLL...LLL", "..LLL.....LLL", "..KKK.....KKK", ".KKKKk...KKKKk"]
    : ["...LLLLLLLL", "....LLL.LLL", "....LLL.LLL", "....KKK.KKK", "...KKKKkKKKKk"]),
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
/** An attack dog at full stretch and gathering itself, facing right. */
const DOG = [
  [
    "...........dd",
    "..........dDDd",
    "d.........DDEDN",
    ".dDdddddddDDD",
    "..DDDDDDDDDD",
    "..DD.....DD",
    ".D..D...D..D",
    "D....D.D....D",
  ],
  [
    "...........dd",
    "..........dDDd",
    ".d........DDEDN",
    "ddDdddddddDDD",
    "..DDDDDDDDDD",
    "...DD...DD",
    "...DD...DD",
    "..DD...DD",
  ],
];
/** A riot shield held in front of a trooper, over columns 10–13 of the soldier map from row 2. */
const SHIELD_PLATE = [".qq.", "qQzq", "qQzq", "qQQq", "qQQq", "qQQq", "qQQq", "qQQq", "qQQq", "qQQq", "qQQq", ".qq."];
/** A sidearm, for the shield trooper. */
const PISTOL = ["", ".......SWW", "", ""];
/** Heads and kit for the specialist troopers. */
const OFFICER_HEAD = ["....DDDD", "...DDDDDD", "...HHHHHHH", "....FFFRF", "....FfFFF", ".....FF"];
const HARDHAT = ["....ZZZZ", "...ZZZZZZ", "..ZZZZZZZZ", "....FFFRF", "....FfFFF", ".....FF"];
const BANDANA = ["....DDDD", "...DDDDDDD.D", "....FFFRF", "....FfFFF", "....FFFF", ".....FF"];
const DYNAMITE = ["", ".......SDDk", ".......SDD", ""];
const MACHETE = ["......ww", ".......Sw", ".......S", ""];
const LASER_GUN = ["", ".......SWWWVVv", ".......SWW", ""];
const JETPACK = ["JJ", "JJ", "JJ", "jj"];

/** Lay `patch` over a character map at column x, row y; dots in the patch leave the map alone. */
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
    J: "#6a7080",
    j: "#3a3f4f",
    Z: "#ffcc33",
    D: "#c8302a",
    V: "#5ff0e0",
    v: "#e6fffc",
  };
  const exoPal: Pal = { ...pal, C: "#6a7080", c: "#4a5060", T: "#3a3f4f", L: "#5d6578", l: "#3a3f4f" };
  const both = (rows: string[], palette = pal) => [paintMap(rows, palette, false), paintMap(rows, palette, true)];
  const gun = (i: number) => GUNS[i];
  const turretPal: Pal = { T: "#6a7080", t: "#8a90a0", B: "#4a4f5f", b: "#ff3b3b", G: "#2e3240" };
  const bug: Pal =
    look === "volcano"
      ? { A: "#8a2a14", a: "#c84a24", d: "#3a0a06", E: "#ffe27a" }
      : { A: "#4a2a6a", a: "#8a5aaa", d: "#1e0e2e", E: "#8aff6a" };
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
    jetpacker: LEGS.map((legs) => both(overlay(bodyMap(HEADS[8], gun(0), legs), 0, 6, JETPACK))),
    minigunner: [0, 1].map((k) => both(overlay(HEAVY(k), 12, 7, ["WWWWWW", "wwwwww", "WWWWWW", "..ZZ"]))),
    officer: LEGS.map((legs) => both(bodyMap(OFFICER_HEAD, PISTOL, legs))),
    dynamiter: LEGS.map((legs) => both(overlay(bodyMap(HEADS[8], DYNAMITE, legs), 3, 10, ["DDDDD"]))),
    rusher: LEGS.map((legs) => both(bodyMap(BANDANA, MACHETE, legs))),
    engineer: LEGS.map((legs) => both(overlay(bodyMap(HARDHAT, PISTOL, legs), 3, 10, ["ZZZZZ"]))),
    laser: LEGS.map((legs) => both(bodyMap(HEADS[8], LASER_GUN, legs))),
    exosuit: [0, 1].map((k) => both(overlay(HEAVY(k), 2, 4, ["JJJJ", "RRRJ"]), exoPal)),
  };
}

/** Barrels, the fuel tank, ammo crates and alien egg pods. */
export function makeProps() {
  const barrel = paintMap(
    [".RRRR.", "RrrrrR", "RRRRRR", "WWWWWW", "RRKRRR", "RRRRRR", "RRRRRR", ".RRRR."],
    { R: "#c8302a", r: "#e8584a", W: "#e8e0d0", K: "#1a1a1a" },
    false
  );
  const fuel = paintMap(
    [
      "..YYYYYYYYYY..",
      ".YyyyyyyyyyyY.",
      "YYYYYYYYYYYYYY",
      "YKKYYKKYYKKYYK",
      "KKYYKKYYKKYYKK",
      "YYYYYYYYYYYYYY",
      "YYYYYKKYYYYYYY",
      "YYYYYKKYYYYYYY",
      "YYYYYYYYYYYYYY",
      ".YYYYYYYYYYYY.",
      "..GG......GG..",
    ],
    { Y: "#e8b830", y: "#ffd860", K: "#1a1a1a", G: "#3a3f4f" },
    false
  );
  const ammo = paintMap(
    ["GGGGGGGG", "GgggggGG", "GYYYYYYG", "GGGGGGGG", "GGgGGgGG", "GGGGGGGG"],
    { G: "#4a5a30", g: "#6a7a48", Y: "#ffcc33" },
    false
  );
  const egg = paintMap(
    ["...EEE..", "..EeeEE.", ".EeEEEEE", ".EEEEGEE", "EEEEEEEEE", "EEEGEEEEE", "EEEEEEEGE", ".EEEEEEE.", "..dddddd."],
    { E: "#7a3a8a", e: "#b87ad0", G: "#8aff6a", d: "#3a1a4a" },
    false
  );
  const propane = paintMap(
    ["..KK..", ".SSSS.", "SsSSSS", "SsSRSS", "SsRRRS", "SsSRSS", "SsSSSS", "SsSSSS", "SsSSSS", ".SSSS.", ".K..K."],
    { S: "#d8dce4", s: "#ffffff", R: "#c8302a", K: "#3a3f4f" },
    false
  );
  const canister = paintMap(
    ["..KKK.", ".RRRRK", "RrRRRR", "RRrRRR", "RRRrRR", "RRRRRR"],
    { R: "#c8302a", r: "#e8584a", K: "#3a3f4f" },
    false
  );
  const cylinder = paintMap(
    ["..K.", ".KK.", ".GG.", "GgGG", "GgGG", "GgGG", "GgGG", "WWWW", "GgGG", "GgGG", "GgGG", ".GG."],
    { G: "#3a7a4a", g: "#6aba7a", K: "#3a3f4f", W: "#e8e4dc" },
    false
  );
  const mine = paintMap(["..KK..", ".MMMMM", "MMMMMMM"], { M: "#4a5030", K: "#3a3f4f" }, false);
  const crate = paintMap(
    ["WWWWWWWWWW", "WwwwwwwwwW", "WwYYYYYYwW", "WwYKKKKYwW", "WwYYYYYYwW", "WwwwwwwwwW", "WWWWWWWWWW"],
    { W: "#5a6a3a", w: "#7a8a4a", Y: "#ffcc33", K: "#1e1e24" },
    false
  );
  const tnt = paintMap(
    ["WWWWWWW", "WRRRRRW", "WRYYYRW", "WRYRYRW", "WRYYYRW", "WRRRRRW", "WWWWWWW", "WRRRRRW"],
    { W: "#8a5a2b", R: "#c8302a", Y: "#ffcc33" },
    false
  );
  return { barrel, fuel, ammo, egg, propane, canister, cylinder, mine, crate, tnt };
}

/* ------------------------------------------------------------------ */
/* Vehicles and bosses, drawn live so their guns aim and legs walk     */
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
export function drawJeep(c: Ctx, x: number, y: number, anim: number, look: Backdrop, _t: number) {
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

/** A troop truck with a canvas-covered back, its flap open once the troops pile out. Faces left. */
export function drawTruck(c: Ctx, x: number, y: number, anim: number, look: Backdrop, open: boolean) {
  const L = LOOKS[look];
  const p = painter(c, x, y, true);
  const body = mix(L.uniform, "#3a3a30", 0.45);
  const cover = mix(L.uniform, "#5a5a40", 0.3);
  // The covered bed at the back, on the right
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

/** A box with a 1 px ink outline and an optional lit top edge. */
function hull(c: Ctx, x: number, y: number, w: number, h: number, col: string, hi?: string) {
  c.fillStyle = INK;
  c.fillRect(x - 1, y - 1, w + 2, h + 2);
  c.fillStyle = col;
  c.fillRect(x, y, w, h);
  if (hi) {
    c.fillStyle = hi;
    c.fillRect(x, y, w, 1);
  }
}
/** A gun barrel from (x, y) along `aim`, `len` long, `t` thick. */
function barrel(c: Ctx, x: number, y: number, aim: number, len: number, col: string, t = 1) {
  const ux = Math.cos(aim);
  const uy = Math.sin(aim);
  c.fillStyle = INK;
  for (let k = 0; k < len; k++) c.fillRect(Math.round(x + ux * k) - 1, Math.round(y + uy * k) - 1, t + 2, t + 2);
  c.fillStyle = col;
  for (let k = 0; k < len; k++) c.fillRect(Math.round(x + ux * k), Math.round(y + uy * k), t, t);
}
const wheels = (c: Ctx, x: number, y: number, n: number, gap: number, r: number, anim: number) => {
  for (let k = 0; k < n; k++) {
    c.fillStyle = INK;
    disc(c, x + k * gap, y, r + 1);
    c.fillStyle = "#26262e";
    disc(c, x + k * gap, y, r);
    c.fillStyle = "#6a6a74";
    const a = anim + k;
    c.fillRect(Math.round(x + k * gap + Math.cos(a) * (r - 1)), Math.round(y + Math.sin(a) * (r - 1)), 1, 1);
  }
};

/** An armoured personnel carrier: eight wheels, a sloped glacis on the left, a little MG turret. */
export function drawApc(c: Ctx, x: number, y: number, anim: number, aim: number, hit: boolean) {
  const body = hit ? "#c8ced9" : "#5a6a3a";
  const hi = hit ? "#ffffff" : "#7a8a4a";
  hull(c, x - 20, y - 14, 42, 9, body, hi);
  c.fillStyle = INK;
  for (let k = 0; k < 6; k++) c.fillRect(x - 22 - k + 6, y - 14 + k, 1, 1);
  hull(c, x - 6, y - 19, 12, 5, body, hi);
  barrel(c, x - 2, y - 17, aim, 12, "#3a3f4f");
  c.fillStyle = "#2a3020";
  c.fillRect(x + 8, y - 12, 6, 4);
  c.fillRect(x - 14, y - 11, 3, 2);
  wheels(c, x - 15, y - 3, 4, 10, 3, anim);
}

/** A light tank: tracks, a hull, a turret whose gun trains on the target. */
export function drawLightTank(c: Ctx, x: number, y: number, anim: number, aim: number, hit: boolean) {
  const body = hit ? "#c8ced9" : "#6a5a3a";
  const hi = hit ? "#ffffff" : "#8a7a4a";
  hull(c, x - 20, y - 7, 40, 7, "#2a2a30");
  wheels(c, x - 16, y - 3, 5, 8, 2, anim);
  hull(c, x - 18, y - 12, 36, 5, body, hi);
  hull(c, x - 8, y - 18, 16, 6, body, hi);
  barrel(c, x - 4, y - 15, aim, 18, "#3a3f4f", 2);
  c.fillStyle = "#c8302a";
  c.fillRect(x + 4, y - 17, 2, 2);
}

/** The sky fortress: a long armoured airship, fins at the back, a gondola with guns underneath, props turning. */
export function drawAirship(c: Ctx, x: number, y: number, t: number, hit: boolean) {
  const skin = hit ? "#e8eef6" : "#7a7266";
  const lite = hit ? "#ffffff" : "#9a9284";
  const dark = "#4a443c";
  c.fillStyle = INK;
  for (let k = -15; k <= 15; k++) {
    const half = Math.round(Math.sqrt(1 - (k / 15.5) ** 2) * 46);
    c.fillRect(x - half - 1, y + k - 1, half * 2 + 2, 3);
  }
  for (let k = -14; k <= 14; k++) {
    const half = Math.round(Math.sqrt(1 - (k / 15) ** 2) * 45);
    c.fillStyle = k < -8 ? lite : k > 8 ? dark : skin;
    c.fillRect(x - half, y + k, half * 2, 1);
  }
  c.fillStyle = dark;
  for (let k = -36; k < 40; k += 9) c.fillRect(x + k, y - 13, 1, 26);
  // Fins
  c.fillStyle = INK;
  c.fillRect(x + 34, y - 22, 12, 9);
  c.fillRect(x + 34, y + 13, 12, 9);
  c.fillStyle = "#c8302a";
  c.fillRect(x + 35, y - 21, 10, 7);
  c.fillRect(x + 35, y + 14, 10, 7);
  // Gondola
  hull(c, x - 16, y + 15, 30, 7, "#3a3f4f", "#5d6578");
  c.fillStyle = "#ffd479";
  for (let k = -12; k < 12; k += 5) c.fillRect(x + k, y + 17, 2, 2);
  // Props
  for (const px of [x - 20, x + 16]) {
    const blade = Math.round(Math.sin(t * 40 + px) * 4);
    c.fillStyle = INK;
    c.fillRect(px - 1, y + 21 - Math.abs(blade), 3, Math.abs(blade) * 2 + 1);
  }
  c.fillStyle = "#ffcc33";
  c.fillRect(x - 30, y - 3, 14, 5);
  c.fillStyle = INK;
  c.fillRect(x - 28, y - 2, 10, 3);
}

/** The doom cannon: a huge mortar on a wheeled carriage, its stubby triple barrel raised. */
export function drawCannon(c: Ctx, x: number, y: number, aim: number, hit: boolean, recoil: number) {
  const body = hit ? "#e8eef6" : "#7a4a2a";
  const hi = hit ? "#ffffff" : "#a86a40";
  hull(c, x - 30, y - 12, 60, 12, "#3a3f4f", "#5d6578");
  wheels(c, x - 22, y - 3, 4, 15, 4, 0);
  hull(c, x - 20, y - 30, 40, 18, body, hi);
  c.fillStyle = "#c8302a";
  c.fillRect(x + 6, y - 26, 10, 10);
  c.fillStyle = INK;
  c.fillRect(x + 10, y - 24, 2, 6);
  c.fillRect(x + 8, y - 22, 6, 2);
  const a = Math.min(-0.5, Math.max(-2.6, aim));
  const back = recoil * 16;
  for (const dy of [-6, 0, 6])
    barrel(c, x - 6 - Math.cos(a) * back + dy * 0.3, y - 34 + dy * 0.4 - Math.sin(a) * back, a, 26, "#3a2a20", 3);
  c.fillStyle = "#ffcc33";
  c.globalAlpha = 0.8;
  c.fillRect(x - 6, y - 42, 6, 6);
  c.globalAlpha = 1;
}

/** The spider tank: a domed body on six jointed legs that step in turn, a plasma cannon in front. */
export function drawSpider(c: Ctx, x: number, y: number, anim: number, aim: number, hit: boolean, t: number) {
  const body = hit ? "#e8eef6" : "#4a3a5a";
  const hi = hit ? "#ffffff" : "#6a5a8a";
  const bodyY = y - 30 + Math.round(Math.sin(anim * 2) * 1.5);
  for (let k = 0; k < 6; k++) {
    const side = k < 3 ? -1 : 1;
    const i = k % 3;
    const hipX = x + (i - 1) * 10;
    const lift = Math.max(0, Math.sin(anim * Math.PI + k * 1.05)) * 4;
    const footX = hipX + side * (16 + i * 4) + Math.sin(anim * Math.PI + k) * 3;
    const kneeX = (hipX + footX) / 2 + side * 3;
    const kneeY = bodyY - 12;
    for (const [ax, ay, bx, by] of [
      [hipX, bodyY, kneeX, kneeY],
      [kneeX, kneeY, footX, y - lift],
    ] as const) {
      const n = Math.max(Math.abs(bx - ax), Math.abs(by - ay), 1);
      for (let s = 0; s <= n; s++) {
        const px = Math.round(ax + ((bx - ax) * s) / n);
        const py = Math.round(ay + ((by - ay) * s) / n);
        c.fillStyle = INK;
        c.fillRect(px - 1, py - 1, 3, 3);
        c.fillStyle = "#3a3f4f";
        c.fillRect(px, py, 1, 1);
      }
    }
  }
  c.fillStyle = INK;
  disc(c, x, bodyY, 15);
  c.fillStyle = body;
  disc(c, x, bodyY, 14);
  c.fillStyle = hi;
  disc(c, x - 4, bodyY - 5, 6);
  c.fillStyle = "#ff3b3b";
  for (const ex of [-7, -3]) c.fillRect(x + ex, bodyY - 1, 2, 2);
  barrel(c, x - 12, bodyY + 2, aim, 14, "#8a4ac8", 2);
  c.fillStyle = "#e0b8ff";
  if (Math.sin(t * 8) > 0)
    c.fillRect(Math.round(x - 12 + Math.cos(aim) * 14), Math.round(bodyY + 2 + Math.sin(aim) * 14), 2, 2);
}

/**
 * A battleship broadside on, its three main turrets trained on `aim`. It is drawn bow to the left
 * and mirrored when `facing` > 0; `sink` 0..1 settles it into the water.
 */
export function drawBattleship(
  c: Ctx,
  x: number,
  y: number,
  t: number,
  aim: number,
  sink: number,
  hit: boolean,
  facing = -1
) {
  mirrored(c, x, facing, () => battleshipFacingLeft(c, x, y, t, facing > 0 ? Math.PI - aim : aim, sink, hit));
  // The hull number near the bow, painted after mirroring so it never reads backwards
  const by = Math.round(y + Math.sin(t * 1.2) * 0.6 + sink * 22);
  if (sink > 0) c.globalAlpha = Math.max(0, 1 - sink * 0.8);
  const nx = facing > 0 ? x + 37 : x - 40;
  c.fillStyle = "#f4f0e8";
  c.fillRect(nx, by - 6, 1, 3);
  c.fillRect(nx + 2, by - 6, 2, 1);
  c.fillRect(nx + 2, by - 4, 2, 1);
  c.fillRect(nx + 3, by - 6, 1, 3);
  c.globalAlpha = 1;
}

/** Draw something made facing left, mirrored about x when it faces right. */
function mirrored(c: Ctx, x: number, facing: number, draw: () => void) {
  if (facing <= 0) return draw();
  c.save();
  c.translate(2 * x, 0);
  c.scale(-1, 1);
  draw();
  c.restore();
}

function battleshipFacingLeft(c: Ctx, x: number, y: number, t: number, aim: number, sink: number, hit: boolean) {
  const by = Math.round(y + Math.sin(t * 1.2) * 0.6 + sink * 22);
  if (sink > 0) c.globalAlpha = Math.max(0, 1 - sink * 0.8);
  const hullC = hit ? "#c8ced9" : "#6a7280";
  const deck = hit ? "#e8eef6" : "#8a92a0";
  const dark = "#4a525e";
  const box2 = (bx: number, bw: number, top: number, h: number, col: string) => {
    c.fillStyle = INK;
    c.fillRect(x + bx - 1, by - top - 1, bw + 2, h + 2);
    c.fillStyle = col;
    c.fillRect(x + bx, by - top, bw, h);
  };
  // Hull
  c.fillStyle = INK;
  for (let k = 0; k < 10; k++) c.fillRect(x - 54 + k, by - 10 + Math.floor(k / 3), 110 - k, 1);
  c.fillRect(x - 50, by - 7, 104, 11);
  c.fillStyle = hullC;
  for (let k = 0; k < 9; k++) c.fillRect(x - 53 + k, by - 9 + Math.floor(k / 3), 107 - k, 1);
  c.fillRect(x - 49, by - 6, 101, 9);
  c.fillStyle = dark;
  c.fillRect(x - 49, by - 1, 101, 1);
  c.fillStyle = "#a02a24";
  c.fillRect(x - 49, by + 1, 101, 2);
  c.fillStyle = deck;
  c.fillRect(x - 52, by - 10, 106, 1);
  c.fillStyle = "#2a2e3a";
  // Portholes, leaving a gap for the hull number that drawBattleship paints on
  for (let k = -44; k < 50; k += 6) if (k !== -38) c.fillRect(x + k, by - 4, 1, 1);
  // Superstructure: a stepped bridge, the funnel, a lattice mast with a radar
  box2(-10, 22, 19, 9, hullC);
  box2(-7, 16, 26, 7, deck);
  box2(-4, 10, 31, 5, deck);
  c.fillStyle = "#8fd3ff";
  for (let k = -6; k < 8; k += 3) c.fillRect(x + k, by - 24, 2, 2);
  c.fillRect(x - 3, by - 29, 8, 1);
  box2(10, 8, 38, 19, dark);
  c.fillStyle = "#1e1e24";
  c.fillRect(x + 10, by - 38, 8, 2);
  c.fillStyle = "#c8302a";
  c.fillRect(x + 10, by - 34, 8, 1);
  c.fillStyle = INK;
  c.fillRect(x, by - 52, 1, 21);
  for (let k = 0; k < 20; k += 4) c.fillRect(x - 2 + (k % 8 ? 1 : 0), by - 50 + k, 5, 1);
  c.fillStyle = "#c8ced9";
  const sweep = Math.round(Math.cos(t * 2) * 4);
  c.fillRect(x - Math.abs(sweep), by - 53, Math.abs(sweep) * 2 + 1, 2);
  c.fillStyle = "#c8302a";
  c.fillRect(x + 1, by - 52, 5, 3);
  // Lifeboats on their davits
  for (const lx of [-20, 22]) {
    box2(lx, 7, 13, 3, "#e8e4dc");
    c.fillStyle = "#c8302a";
    c.fillRect(x + lx, by - 11, 7, 1);
  }
  // Main turrets, guns trained on the target: two forward, one aft
  const ux = Math.cos(aim);
  const uy = Math.sin(aim);
  for (const tx of [-36, -24, 38]) {
    box2(tx - 5, 11, 15, 5, deck);
    for (const dy of [-1, 1]) {
      c.fillStyle = INK;
      for (let k = 0; k < 14; k++)
        c.fillRect(Math.round(x + tx + ux * k) - 1, Math.round(by - 13 + dy + uy * k) - 1, 3, 2);
      c.fillStyle = dark;
      for (let k = 0; k < 14; k++) c.fillRect(Math.round(x + tx + ux * k), Math.round(by - 13 + dy + uy * k), 1, 1);
    }
  }
  // Quad anti-aircraft mounts, barrels raised
  for (const ax of [-14, 14, 28]) {
    box2(ax - 2, 5, 13, 3, dark);
    c.fillStyle = INK;
    for (let k = 0; k < 4; k++) c.fillRect(x + ax - 2 + k, by - 17 - k, 1, 4);
  }
  if (!sink) {
    // The bow wave, and the wake churning out behind the stern
    c.fillStyle = "#e8f8ff";
    for (let k = 0; k < 10; k++) if (Math.sin(t * 6 + k) > 0) c.fillRect(x - 56 - k, by + 1 - (k % 2), 1, 1);
    for (let k = 0; k < 18; k++)
      if (Math.sin(t * 5 - k * 0.9) > 0.1) c.fillRect(x + 54 + k * 2, by + 1 - (k % 3 === 0 ? 1 : 0), k < 8 ? 2 : 1, 1);
  }
  c.globalAlpha = 1;
}

/** An armoured train: an engine at the front, then a flak car, a gun car with its slits, a rocket car. */
export function drawTrain(c: Ctx, x: number, y: number, anim: number, hit: boolean) {
  const body = hit ? "#e8eef6" : "#4a5040";
  const hi = hit ? "#ffffff" : "#6a7058";
  // Engine
  hull(c, x - 64, y - 16, 26, 12, body, hi);
  hull(c, x - 46, y - 22, 8, 6, body, hi);
  hull(c, x - 58, y - 26, 4, 10, "#2a2a30");
  c.fillStyle = "#ffe27a";
  c.fillRect(x - 65, y - 12, 2, 2);
  c.fillStyle = "#c8302a";
  c.fillRect(x - 64, y - 6, 26, 1);
  // Cars
  hull(c, x - 36, y - 14, 26, 10, body, hi);
  hull(c, x - 28, y - 20, 10, 6, "#3a3f4f");
  c.fillStyle = INK;
  for (let k = 0; k < 3; k++) c.fillRect(x - 26 + k * 3, y - 26, 1, 6);
  hull(c, x - 8, y - 15, 30, 11, body, hi);
  c.fillStyle = INK;
  for (let k = -4; k < 20; k += 5) c.fillRect(x + k, y - 11, 3, 1);
  hull(c, x + 24, y - 13, 32, 9, body, hi);
  hull(c, x + 28, y - 20, 22, 7, "#5d6578", "#7a8294");
  c.fillStyle = "#c8302a";
  for (let k = 0; k < 6; k++) c.fillRect(x + 29 + k * 3, y - 19, 2, 2);
  // Wheels along the whole length
  for (let k = -60; k < 56; k += 9) {
    c.fillStyle = INK;
    disc(c, x + k, y - 2, 3);
    c.fillStyle = "#26262e";
    disc(c, x + k, y - 2, 2);
    c.fillStyle = "#6a6a74";
    c.fillRect(Math.round(x + k + Math.cos(anim + k) * 1.5), Math.round(y - 2 + Math.sin(anim + k) * 1.5), 1, 1);
  }
}

/** A river patrol boat at the water line y, bow towards `facing`, a gunner on the stern; `sink` 0..1 as it goes down. */
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
    for (let k = 0; k < 6; k++)
      if (Math.sin(t * 9 + k) > 0) c.fillRect(facing < 0 ? x + 18 + k * 2 : x - 20 - k * 2, y, 1, 1);
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
    const wx = -25 + k * 6 + (Math.floor(anim * 4) % 2 ? 1 : 0);
    p(wx, -9, 4, 6, "#4a4a44");
  }
  // Hull
  p(-28, -22, 56, 11, INK);
  p(-27, -21, 54, 9, hull);
  p(-27, -21, 54, 2, mix(hull, "#ffffff", 0.2));
  p(-20, -16, 6, 3, dark);
  p(10, -16, 6, 3, dark);
  // Turret and cannon; the cannon follows `aim` but stays in the left half
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
  // Rocket pod on top, the aiming gun arm in front
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
      c.fillRect(
        Math.round(x - ux * (18 + k) + px * s * (3 + k)),
        Math.round(y - uy * (18 + k) + py * s * (3 + k)),
        2,
        2
      );
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

/** Draw text in the 3×5 pixel font, 1 px between letters, scaled by `scale`; returns its width. */
export function pixelText(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  color: string,
  outline = true,
  scale = 1
) {
  const chars = text.toUpperCase().split("");
  const draw = (ox: number, oy: number, colr: string) => {
    ctx.fillStyle = colr;
    chars.forEach((ch, i) => {
      const g = GLYPHS[ch] ?? GLYPHS[" "];
      for (let b = 0; b < 15; b++)
        if (g[b] === "1")
          ctx.fillRect(x + ox + (i * 4 + (b % 3)) * scale, y + oy + Math.floor(b / 3) * scale, scale, scale);
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

/** Filled pixel circle, drawn in scanlines so it stays crisp. */
export function disc(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number) {
  if (r <= 0) return;
  for (let dy = -Math.floor(r); dy <= Math.floor(r); dy++) {
    const w = Math.floor(Math.sqrt(r * r - dy * dy));
    ctx.fillRect(cx - w, cy + dy, w * 2 + 1, 1);
  }
}

/** One-pixel circle outline, by the midpoint algorithm. */
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

/* ------------------------------------------------------------------ */
/* Props: assets.ts defines them, these paint the basic set            */
/* ------------------------------------------------------------------ */

const dot = (c: Ctx, x: number, y: number, color: string) => box(c, x, y, 1, 1, color);
/** A line of whole pixels from (x0, y0) to (x1, y1), `t` px thick. */
function seg(c: Ctx, x0: number, y0: number, x1: number, y1: number, color: string, t = 1) {
  const n = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
  for (let k = 0; k <= n; k++)
    box(c, Math.round(x0 + ((x1 - x0) * k) / n), Math.round(y0 + ((y1 - y0) * k) / n), t, t, color);
}
const crate = (c: Ctx, x: number, y: number, w: number, h: number) => {
  box(c, x, y, w, h, "#9c6834");
  box(c, x, y, w, 1, "#c08a4a");
  box(c, x, y, 1, h, "#c08a4a");
  box(c, x, y + h - 1, w, 1, "#5a3816");
  box(c, x + w - 1, y, 1, h, "#5a3816");
  for (let k = 1; k < w - 1; k++) dot(c, x + k, y + Math.round((k * (h - 1)) / (w - 1)), "#6e4520");
};
const drum = (c: Ctx, x: number, y: number, body: string, light: string, band: string) => {
  box(c, x, y, 5, 7, body);
  box(c, x, y, 5, 1, light);
  box(c, x, y, 1, 7, light);
  box(c, x, y + 2, 5, 1, band);
  box(c, x, y + 5, 5, 1, band);
};
const camo = ["#4a5a30", "#6a7a48", "#2e3a1e", "#8a7a4a"];

const PAINT: Record<string, (c: Ctx) => void> = {
  crates: (c) => {
    crate(c, 0, 6, 7, 7);
    crate(c, 7, 6, 7, 7);
    crate(c, 3, 0, 7, 6);
  },
  ammocrate: (c) => {
    box(c, 0, 1, 12, 7, "#4a5a30");
    box(c, 0, 0, 12, 2, "#6a7a48");
    box(c, 3, 4, 6, 1, "#ffcc33");
    box(c, 0, 4, 1, 2, "#2e3a1e");
    box(c, 11, 4, 1, 2, "#2e3a1e");
    box(c, 5, 1, 2, 1, "#3a3f4f");
  },
  drums: (c) => {
    drum(c, 0, 7, "#3a6ab0", "#7ab0e8", "#2a4a80");
    drum(c, 6, 7, "#3a6ab0", "#7ab0e8", "#2a4a80");
    drum(c, 3, 0, "#c8302a", "#e8584a", "#7a1a16");
  },
  sandbags: (c) => {
    const bag = (x: number, y: number) => {
      box(c, x, y, 4, 3, "#c4ae78");
      box(c, x, y, 4, 1, "#d8c890");
      box(c, x, y + 2, 4, 1, "#9a8858");
    };
    for (let x = 0; x < 16; x += 4) bag(x, 3);
    for (let x = 2; x < 14; x += 4) bag(x, 0);
  },
  tires: (c) => {
    const tire = (x: number, y: number) => {
      box(c, x, y, 6, 4, "#1e1e24");
      box(c, x, y, 6, 1, "#3a3a44");
      box(c, x + 2, y + 1, 2, 2, "#5d6578");
    };
    tire(0, 4);
    tire(6, 4);
    tire(3, 0);
  },
  jerrycans: (c) => {
    for (const x of [0, 5]) {
      box(c, x, 1, 5, 7, "#c8302a");
      box(c, x, 1, 1, 7, "#e8584a");
      box(c, x + 1, 0, 2, 1, "#7a1a16");
      seg(c, x + 1, 3, x + 3, 6, "#9a2020");
      seg(c, x + 3, 3, x + 1, 6, "#9a2020");
    }
  },
  radio: (c) => {
    box(c, 1, 0, 1, 8, "#3a3f4f");
    box(c, 0, 7, 8, 9, "#4a5a30");
    box(c, 0, 7, 8, 1, "#6a7a48");
    box(c, 1, 9, 3, 2, "#ffcc33");
    for (let y = 12; y < 15; y += 2) box(c, 1, y, 6, 1, "#2e3a1e");
    box(c, 5, 9, 2, 2, "#1e1e24");
  },
  generator: (c) => {
    box(c, 2, 0, 2, 3, "#3a3f4f");
    box(c, 0, 2, 14, 7, "#5a6a3a");
    box(c, 0, 2, 14, 1, "#7a8a5a");
    for (let x = 2; x < 9; x += 2) box(c, x, 4, 1, 4, "#3a4a24");
    box(c, 10, 3, 3, 3, "#2a2e24");
    box(c, 1, 9, 3, 1, "#1e1e24");
    box(c, 10, 9, 3, 1, "#1e1e24");
  },
  dish: (c) => {
    box(c, 7, 8, 1, 7, "#5d6578");
    box(c, 4, 14, 7, 2, "#3a3f4f");
    c.fillStyle = "#c8ced9";
    disc(c, 6, 6, 5.5);
    c.fillStyle = "#9aa1b0";
    disc(c, 7, 5, 3.5);
    seg(c, 6, 6, 11, 1, "#5d6578");
    dot(c, 11, 1, "#ff3b3b");
  },
  watertower: (c) => {
    box(c, 4, 0, 10, 1, "#5a3816");
    box(c, 2, 1, 14, 1, "#5a3816");
    box(c, 1, 2, 16, 1, "#5a3816");
    box(c, 1, 3, 16, 12, "#8a5a2b");
    for (let x = 2; x < 17; x += 3) box(c, x, 3, 1, 12, "#6e4520");
    box(c, 1, 5, 16, 1, "#3a3f4f");
    box(c, 1, 12, 16, 1, "#3a3f4f");
    box(c, 1, 3, 1, 12, "#b07a40");
    for (const x of [2, 7, 10, 15]) box(c, x, 15, 1, 21, "#5a3816");
    for (let y = 17; y < 34; y += 6) {
      seg(c, 2, y, 7, y + 5, "#6e4520");
      seg(c, 15, y, 10, y + 5, "#6e4520");
    }
    for (let y = 16; y < 36; y += 2) box(c, 12, y, 2, 1, "#9a9aa0");
  },
  streetlamp: (c) => {
    box(c, 1, 2, 1, 21, "#3a3f4f");
    box(c, 1, 2, 5, 1, "#3a3f4f");
    box(c, 4, 2, 3, 2, "#5d6578");
    dot(c, 5, 4, "#ffe27a");
    box(c, 0, 22, 3, 2, "#2a2e3a");
  },
  dangersign: (c) => {
    box(c, 4, 8, 1, 5, "#8a5a2b");
    for (let y = 0; y < 8; y++) {
      const half = Math.floor((y * 4) / 7);
      box(c, 4 - half, y, half * 2 + 1, 1, y === 7 ? "#c89a20" : "#ffcc33");
    }
    box(c, 4, 2, 1, 3, "#0d0b14");
    dot(c, 4, 6, "#0d0b14");
  },
  signpost: (c) => {
    box(c, 5, 2, 2, 12, "#6e4520");
    box(c, 1, 2, 9, 3, "#b07a40");
    dot(c, 0, 3, "#b07a40");
    box(c, 3, 6, 8, 3, "#9c6834");
    dot(c, 11, 7, "#9c6834");
    for (const x of [3, 5, 7]) dot(c, x, 3, "#5a3816");
    for (const x of [5, 7, 9]) dot(c, x, 7, "#5a3816");
  },
  hedgehog: (c) => {
    seg(c, 0, 7, 8, 0, "#5d6578", 2);
    seg(c, 3, 0, 10, 7, "#5d6578", 2);
    box(c, 1, 5, 10, 2, "#4a5264");
    seg(c, 1, 7, 8, 1, "#9aa1b0");
  },
  carwreck: (c) => {
    box(c, 0, 4, 24, 4, "#8a4a2a");
    box(c, 0, 4, 24, 1, "#a86a40");
    box(c, 5, 0, 12, 4, "#6a3a22");
    box(c, 7, 1, 3, 3, "#1e1e24");
    box(c, 12, 1, 4, 3, "#1e1e24");
    box(c, 9, 5, 5, 2, "#3a2a20");
    box(c, 19, 5, 3, 1, "#3a2a20");
    box(c, 3, 8, 5, 2, "#1e1e24");
    box(c, 17, 8, 4, 2, "#1e1e24");
  },
  heliwreck: (c) => {
    box(c, 4, 5, 18, 7, "#4a5a30");
    box(c, 4, 5, 18, 1, "#6a7a48");
    box(c, 2, 7, 3, 4, "#4a5a30");
    box(c, 3, 5, 5, 3, "#2a3040");
    box(c, 21, 7, 7, 2, "#3e4c28");
    box(c, 27, 3, 2, 5, "#3e4c28");
    seg(c, 6, 4, 18, 0, "#3a3f4f");
    seg(c, 13, 4, 10, 2, "#3a3f4f");
    box(c, 5, 13, 16, 1, "#3a3f4f");
    box(c, 7, 12, 1, 1, "#3a3f4f");
    box(c, 18, 12, 1, 1, "#3a3f4f");
    box(c, 10, 7, 6, 3, "#2a2e20");
    box(c, 12, 8, 2, 1, "#1e1e24");
  },
  cage: (c) => {
    box(c, 6, 4, 2, 2, "#e0a878");
    box(c, 5, 6, 4, 4, "#a89868");
    box(c, 5, 10, 1, 3, "#5a4a30");
    box(c, 8, 10, 1, 3, "#5a4a30");
    box(c, 9, 3, 1, 3, "#e0a878");
    box(c, 0, 0, 14, 2, "#5a3816");
    box(c, 0, 13, 14, 2, "#5a3816");
    for (let x = 0; x < 14; x += 3) box(c, x, 2, 1, 11, "#8a8a92");
    box(c, 13, 2, 1, 11, "#8a8a92");
  },
  tikitorch: (c) => {
    box(c, 1, 3, 1, 11, "#8a5a2b");
    box(c, 0, 1, 3, 3, "#5a3816");
    dot(c, 1, 1, "#ffcc33");
  },
  idol: (c) => {
    box(c, 1, 0, 10, 15, "#7a7a84");
    box(c, 2, 0, 8, 1, "#a0a0aa");
    box(c, 1, 1, 1, 14, "#a0a0aa");
    box(c, 10, 1, 1, 14, "#4a4a54");
    box(c, 1, 4, 10, 1, "#4a4a54");
    box(c, 2, 5, 3, 2, "#2a2a32");
    box(c, 7, 5, 3, 2, "#2a2a32");
    box(c, 5, 7, 2, 4, "#5a5a64");
    box(c, 3, 12, 6, 2, "#2a2a32");
    for (const x of [4, 6]) dot(c, x, 12, "#a0a0aa");
    box(c, 0, 15, 12, 1, "#4a4a54");
    for (const [x, y] of [
      [2, 2],
      [9, 9],
      [3, 10],
      [8, 1],
    ])
      dot(c, x, y, "#5a7a4a");
  },
  bunting: (c) => {
    box(c, 0, 0, 2, 14, "#6e4520");
    box(c, 24, 0, 2, 14, "#6e4520");
    const flags = ["#c8302a", "#ffcc33", "#3a6ab0", "#5bbf3a"];
    for (let x = 2; x < 24; x++) {
      const y = 1 + Math.round(Math.sin((Math.PI * (x - 2)) / 21) * 4);
      dot(c, x, y, "#c8c0a8");
      if (x % 3 === 0) {
        const f = flags[(x / 3) % 4];
        box(c, x, y + 1, 2, 1, f);
        box(c, x, y + 2, 2, 1, f);
        dot(c, x, y + 3, f);
      }
    }
  },
  well: (c) => {
    box(c, 4, 0, 6, 1, "#8a3e2c");
    box(c, 2, 1, 10, 1, "#8a3e2c");
    box(c, 0, 2, 14, 1, "#6a2e20");
    box(c, 1, 3, 1, 5, "#6e4520");
    box(c, 12, 3, 1, 5, "#6e4520");
    box(c, 4, 4, 6, 1, "#5a3816");
    box(c, 7, 5, 1, 3, "#c8b888");
    box(c, 1, 8, 12, 5, "#7a7a84");
    box(c, 1, 8, 12, 1, "#a0a0aa");
    for (const [x, y] of [
      [4, 10],
      [9, 10],
      [2, 11],
      [7, 12],
      [11, 12],
    ])
      box(c, x, y, 1, 1, "#4a4a54");
  },
  boulders: (c) => {
    c.fillStyle = "#6d7184";
    disc(c, 4, 4, 3.6);
    c.fillStyle = "#5a5e70";
    disc(c, 10, 5, 2.8);
    box(c, 2, 2, 2, 1, "#81869a");
    box(c, 9, 3, 1, 1, "#81869a");
  },
  cactus: (c) => {
    box(c, 3, 0, 3, 16, "#4a8a3a");
    box(c, 3, 0, 1, 16, "#6aaa4a");
    box(c, 1, 7, 2, 2, "#4a8a3a");
    box(c, 0, 3, 2, 5, "#4a8a3a");
    box(c, 6, 9, 2, 2, "#4a8a3a");
    box(c, 7, 5, 2, 5, "#4a8a3a");
    for (const [x, y] of [
      [5, 3],
      [3, 8],
      [5, 12],
      [0, 4],
      [8, 6],
    ])
      dot(c, x, y, "#c8e0a0");
    dot(c, 4, 0, "#ff6fae");
  },
  logs: (c) => {
    const log = (x: number, y: number, w: number) => {
      box(c, x, y, w, 3, "#7a5230");
      box(c, x, y, w, 1, "#9a7250");
      box(c, x + w - 2, y, 2, 3, "#c8a070");
      dot(c, x + w - 1, y + 1, "#7a5230");
    };
    log(0, 4, 14);
    log(2, 1, 10);
  },
  cone: (c) => {
    box(c, 2, 0, 1, 2, "#ff8a1a");
    box(c, 1, 2, 3, 1, "#f4f0e8");
    box(c, 1, 3, 3, 2, "#ff8a1a");
    box(c, 0, 5, 5, 1, "#ff8a1a");
    box(c, 0, 6, 5, 1, "#3a3a44");
  },
  hydrant: (c) => {
    box(c, 1, 0, 3, 1, "#e8584a");
    box(c, 1, 1, 3, 6, "#c8302a");
    box(c, 1, 1, 1, 6, "#e8584a");
    box(c, 0, 3, 1, 2, "#a02020");
    box(c, 4, 3, 1, 2, "#a02020");
    box(c, 0, 7, 5, 1, "#7a1a16");
  },
  phonebooth: (c) => {
    box(c, 0, 0, 8, 18, "#c8302a");
    box(c, 1, 1, 6, 2, "#1e1e24");
    box(c, 1, 4, 6, 11, "#7ab0e8");
    box(c, 1, 8, 6, 1, "#a02020");
    box(c, 1, 12, 6, 1, "#a02020");
    box(c, 4, 4, 1, 11, "#a02020");
    box(c, 0, 0, 1, 18, "#e8584a");
    box(c, 0, 16, 8, 2, "#7a1a16");
  },
  billboard: (c) => {
    box(c, 5, 14, 2, 8, "#3a3f4f");
    box(c, 19, 14, 2, 8, "#3a3f4f");
    box(c, 3, 0, 3, 2, "#5d6578");
    box(c, 20, 0, 3, 2, "#5d6578");
    box(c, 0, 2, 26, 12, "#5d6578");
    box(c, 1, 3, 24, 10, "#3a6ab0");
    c.fillStyle = "#ffcc33";
    disc(c, 7, 8, 3);
    box(c, 13, 5, 10, 2, "#f4f0e8");
    box(c, 13, 8, 7, 1, "#f4f0e8");
    box(c, 13, 10, 9, 1, "#f4f0e8");
    for (const [x, y] of [
      [4, 4],
      [17, 11],
      [22, 6],
    ])
      dot(c, x, y, "#1e1e24");
  },
  missiletrailer: (c) => {
    box(c, 0, 8, 30, 2, "#3a3f4f");
    for (const x of [3, 10, 22]) box(c, x, 10, 4, 2, "#1e1e24");
    box(c, 8, 7, 2, 1, "#5d6578");
    box(c, 18, 7, 2, 1, "#5d6578");
    box(c, 2, 4, 24, 3, "#c8ced9");
    box(c, 2, 4, 24, 1, "#e8eef6");
    box(c, 26, 5, 2, 2, "#c8ced9");
    dot(c, 28, 5, "#c8302a");
    box(c, 0, 2, 3, 2, "#c8302a");
    box(c, 0, 7, 3, 1, "#c8302a");
    box(c, 12, 4, 1, 3, "#c8302a");
    box(c, 16, 5, 6, 1, "#5d6578");
  },
  fuelpump: (c) => {
    box(c, 0, 0, 6, 2, "#e8584a");
    box(c, 0, 2, 6, 11, "#c8302a");
    box(c, 1, 3, 4, 2, "#ffe27a");
    box(c, 1, 7, 4, 3, "#a02020");
    box(c, 6, 4, 1, 7, "#1e1e24");
    box(c, 5, 10, 2, 1, "#3a3f4f");
    box(c, 0, 13, 7, 1, "#7a1a16");
  },
  outhouse: (c) => {
    box(c, 0, 0, 9, 2, "#5a3816");
    box(c, 1, 2, 7, 14, "#8a5a2b");
    box(c, 2, 4, 5, 12, "#9c6834");
    for (let x = 3; x < 7; x += 2) box(c, x, 4, 1, 12, "#7a4a22");
    box(c, 4, 5, 2, 1, "#1e1e24");
    dot(c, 4, 6, "#1e1e24");
    dot(c, 6, 10, "#3a3f4f");
  },
  loudspeaker: (c) => {
    box(c, 4, 3, 2, 17, "#5d6578");
    box(c, 2, 20, 6, 2, "#3a3f4f");
    box(c, 2, 4, 6, 1, "#5d6578");
    box(c, 0, 0, 4, 4, "#9aa1b0");
    box(c, 0, 0, 1, 4, "#5d6578");
    box(c, 6, 0, 4, 4, "#9aa1b0");
    box(c, 9, 0, 1, 4, "#5d6578");
    box(c, 3, 8, 1, 6, "#1e1e24");
  },
  valve: (c) => {
    box(c, 0, 5, 12, 3, "#5a7a6a");
    box(c, 0, 5, 12, 1, "#7a9a8a");
    box(c, 3, 4, 1, 5, "#3a4a40");
    box(c, 8, 4, 1, 5, "#3a4a40");
    box(c, 5, 2, 2, 3, "#3a3f4f");
    box(c, 3, 0, 6, 2, "#c8302a");
    dot(c, 5, 0, "#e8584a");
  },
  mushrooms: (c) => {
    box(c, 0, 1, 4, 2, "#c84aff");
    box(c, 1, 3, 2, 3, "#e8d8f0");
    dot(c, 1, 1, "#f0c8ff");
    box(c, 5, 2, 5, 2, "#5ac83a");
    box(c, 7, 4, 1, 2, "#e8d8f0");
    dot(c, 8, 2, "#c8ff9a");
  },
  reeds: (c) => {
    for (const [x, h, head] of [
      [1, 8, 1],
      [3, 11, 0],
      [5, 9, 1],
      [7, 10, 0],
      [9, 7, 1],
      [11, 9, 0],
    ] as const) {
      box(c, x, 11 - h, 1, h, x % 4 === 1 ? "#6a8a3a" : "#8aaa4a");
      if (head) box(c, x, 11 - h, 1, 3, "#6a4a2a");
    }
  },
  stump: (c) => {
    box(c, 1, 1, 6, 5, "#7a5230");
    box(c, 1, 1, 6, 1, "#c8a070");
    dot(c, 3, 1, "#9a7250");
    box(c, 0, 4, 1, 2, "#5a3a20");
    box(c, 7, 4, 1, 2, "#5a3a20");
    box(c, 2, 3, 1, 3, "#5a3a20");
  },
  flowers: (c) => {
    box(c, 0, 4, 10, 2, "#3a7a3a");
    for (const [x, y, col] of [
      [1, 2, "#ff6fae"],
      [4, 1, "#ffcc33"],
      [7, 2, "#c77dff"],
      [9, 3, "#ff6fae"],
    ] as const) {
      box(c, x, y + 1, 1, 4 - y, "#2c6330");
      box(c, x - (x ? 1 : 0), y, 2, 1, col);
      dot(c, x, y - 1 < 0 ? 0 : y - 1, col);
    }
  },
  pillar: (c) => {
    box(c, 0, 1, 3, 1, "#b8b8c2");
    box(c, 0, 2, 6, 2, "#b8b8c2");
    box(c, 0, 4, 6, 12, "#a0a0aa");
    box(c, 1, 4, 1, 12, "#8a8a94");
    box(c, 4, 4, 1, 12, "#8a8a94");
    box(c, 5, 4, 1, 12, "#7a7a84");
    box(c, 0, 16, 6, 2, "#7a7a84");
    for (const [x, y] of [
      [0, 9],
      [2, 12],
      [5, 6],
      [1, 15],
    ])
      dot(c, x, y, "#5a7a4a");
  },
  crystals: (c) => {
    for (const [x, h, w] of [
      [0, 6, 2],
      [3, 10, 3],
      [7, 5, 3],
    ] as const) {
      box(c, x, 10 - h + 1, w, h - 1, "#5ff0e0");
      box(c, x, 10 - h + 1, 1, h - 1, "#c8fff8");
      box(c, x + w - 1, 10 - h + 1, 1, h - 1, "#2a9a90");
      dot(c, x + (w >> 1), 10 - h, "#e6fffc");
    }
  },
  pod: (c) => {
    for (let y = 0; y < 11; y++) {
      const half = Math.max(1, Math.round(Math.sqrt(Math.max(0, 1 - ((y - 6) / 6) ** 2)) * 4.5));
      box(c, 5 - half, y, half * 2, 1, "#7a3a8a");
    }
    seg(c, 2, 8, 4, 1, "#b87ad0");
    seg(c, 8, 8, 6, 2, "#b87ad0");
    box(c, 4, 4, 2, 3, "#8aff6a");
    box(c, 1, 11, 8, 1, "#3a1a4a");
  },
  lavavent: (c) => {
    box(c, 1, 2, 8, 4, "#3a2f33");
    box(c, 3, 1, 4, 1, "#3a2f33");
    box(c, 4, 0, 2, 2, "#ffd23f");
    seg(c, 4, 2, 2, 5, "#ff8a1a");
    seg(c, 6, 2, 8, 5, "#d8481a");
  },
  toxicdrum: (c) => {
    box(c, 0, 0, 6, 9, "#d8c030");
    box(c, 0, 0, 1, 9, "#f0e070");
    box(c, 0, 2, 6, 1, "#8a7a20");
    box(c, 0, 6, 6, 1, "#8a7a20");
    box(c, 2, 3, 2, 2, "#0d0b14");
    box(c, 1, 0, 4, 1, "#8aff6a");
    dot(c, 4, 1, "#8aff6a");
  },
  camonet: (c) => {
    box(c, 4, 5, 7, 5, "#9c6834");
    box(c, 13, 4, 8, 6, "#4a5a30");
    box(c, 1, 1, 1, 9, "#5a3816");
    box(c, 24, 1, 1, 9, "#5a3816");
    for (let x = 0; x < 26; x++) {
      const y = 1 + (x < 2 || x > 23 ? 0 : Math.round(Math.sin(x * 0.9) + 1));
      for (let k = 0; k < 2; k++) dot(c, x, y + k, camo[(x * 7 + k * 3 + y) % 4]);
      if ((x < 3 || x > 22) && x % 2 === 0)
        for (let k = 2; k < 8; k++) if ((x + k) % 3) dot(c, x, y + k, camo[(x + k) % 4]);
    }
  },
  radiotower: (c) => {
    seg(c, 1, 45, 6, 1, "#5d6578");
    seg(c, 12, 45, 7, 1, "#5d6578");
    for (let y = 5; y < 45; y += 5) {
      const half = Math.round(0.5 + ((y - 1) / 44) * 5.5);
      box(c, 7 - half, y, half * 2, 1, "#5d6578");
      seg(c, 7 - half, y, 7 + half, Math.min(45, y + 5), "#4a5264");
    }
    box(c, 6, 0, 2, 2, "#5d6578");
    box(c, 3, 12, 8, 1, "#3a3f4f");
  },
};

/** One prop's art: facing right and left, and the same in shadow for the background. */
interface AssetArt {
  img: [HTMLCanvasElement, HTMLCanvasElement];
  dark: [HTMLCanvasElement, HTMLCanvasElement];
}

/**
 * Every prop's art, painted the first time it's needed: outlined in ink by stamping its
 * silhouette around it, mirrored, and darkened for the background.
 */
export function makeAssets(): (id: AssetId) => AssetArt {
  const cache = new Map<AssetId, AssetArt>();
  return (id) => {
    const hit = cache.get(id);
    if (hit) return hit;
    const { w, h } = ASSETS[id];
    const [src, sc] = makeCanvas(w, h);
    if (PAINT[id]) PAINT[id](sc);
    else if (CATALOG[id]) paintCatalog(id, sc);
    const [sil, s] = makeCanvas(w + 2, h + 2);
    for (const [dx, dy] of [
      [0, 1],
      [2, 1],
      [1, 0],
      [1, 2],
    ])
      s.drawImage(src, dx, dy);
    s.globalCompositeOperation = "source-in";
    s.fillStyle = INK;
    s.fillRect(0, 0, w + 2, h + 2);
    const [img, ic] = makeCanvas(w + 2, h + 2);
    ic.drawImage(sil, 0, 0);
    ic.drawImage(src, 1, 1);
    const mirror = (from: HTMLCanvasElement) => {
      const [m, mc] = makeCanvas(w + 2, h + 2);
      mc.translate(w + 2, 0);
      mc.scale(-1, 1);
      mc.drawImage(from, 0, 0);
      return m;
    };
    const [dark, dc] = makeCanvas(w + 2, h + 2);
    dc.drawImage(img, 0, 0);
    dc.globalCompositeOperation = "source-atop";
    dc.fillStyle = "rgba(14,12,30,0.62)";
    dc.fillRect(0, 0, w + 2, h + 2);
    const art: AssetArt = { img: [img, mirror(img)], dark: [dark, mirror(dark)] };
    cache.set(id, art);
    return art;
  };
}

/* ------------------------------------------------------------------ */
/* Tanks, helicopters and planes, enemy and allied                      */
/* ------------------------------------------------------------------ */

export type ArmorType =
  | "medtank"
  | "heavytank"
  | "flametank"
  | "rockettank"
  | "aatank"
  | "launcher"
  | "armoredcar"
  | "artytruck"
  | "hovercraft"
  | "mbt"
  | "ifv"
  | "mlrs"
  | "aa"
  | "scoutcar";
/** The enemy's paint and the allies'; allies are blue-grey with a white star so the two sides read apart. */
const SIDE = {
  enemy: { body: "#5a6248", hi: "#7a8462", dark: "#3a4030", track: "#24242a" },
  ally: { body: "#5a7a9a", hi: "#8aaac8", dark: "#3a5068", track: "#20242c" },
};
const star = (c: Ctx, x: number, y: number) => {
  c.fillStyle = "#f4f0e8";
  c.fillRect(x - 1, y, 3, 1);
  c.fillRect(x, y - 1, 1, 3);
};
/** Draw in a vehicle's own frame, facing left like the enemy's; `flip` faces it right. */
function framed(c: Ctx, x: number, flip: boolean, fn: () => void) {
  c.save();
  c.translate(x, 0);
  if (flip) c.scale(-1, 1);
  fn();
  c.restore();
}

/**
 * Every kind of tank and armoured vehicle, enemy or allied. `aim` is the gun's angle in the world;
 * `raise`, from 0 to 1, lifts a launcher's missile.
 */
export function drawArmor(
  c: Ctx,
  x: number,
  y: number,
  type: ArmorType,
  anim: number,
  aim: number,
  hit: boolean,
  raise = 0
) {
  const ally = type === "mbt" || type === "ifv" || type === "mlrs" || type === "aa" || type === "scoutcar";
  const P = SIDE[ally ? "ally" : "enemy"];
  const body = hit
    ? "#e8eef6"
    : type === "flametank"
      ? "#6a4a30"
      : type === "aatank"
        ? "#4a5a5a"
        : type === "heavytank"
          ? "#4a5040"
          : P.body;
  const hi = hit ? "#ffffff" : P.hi;
  // An ally's frame is mirrored, so its aim in the world is mirrored into the frame too
  const a = ally ? Math.PI - aim : aim;
  framed(c, x, ally, () => {
    const big = type === "heavytank";
    const small = type === "armoredcar" || type === "scoutcar";
    const len = big ? 50 : type === "launcher" ? 54 : type === "artytruck" ? 48 : small ? 32 : 40;
    const half = len >> 1;
    const wheeled = type === "launcher" || type === "ifv" || type === "artytruck" || small;
    if (type === "hovercraft") {
      // Skirt, hull, and the ducted fan at the back
      hull(c, -half, y - 6, len, 6, "#1e1e24");
      hull(c, -half + 2, y - 13, len - 4, 7, body, hi);
      c.fillStyle = INK;
      disc(c, half - 6, y - 17, 6);
      c.fillStyle = "#5d6578";
      disc(c, half - 6, y - 17, 5);
      c.fillStyle = INK;
      const blade = Math.round(Math.sin(anim * 20) * 4);
      c.fillRect(half - 7, y - 17 - Math.abs(blade), 2, Math.abs(blade) * 2 + 1);
    } else if (wheeled && small) {
      hull(c, -half, y - 11, len, 7, body, hi);
      hull(c, -half + 4, y - 15, len - 12, 4, body, hi);
      c.fillStyle = "#8fd3ff";
      c.fillRect(-half + 6, y - 14, 4, 2);
      wheels(c, -half + 6, y - 3, 3, 10, 3, anim);
    } else if (wheeled) {
      hull(c, -half, y - 12, len, 8, body, hi);
      wheels(c, -half + 6, y - 3, type === "launcher" ? 5 : 4, type === "launcher" ? 10 : 9, 3, anim);
    } else {
      hull(c, -half, y - 7, len, 7, P.track);
      wheels(c, -half + 4, y - 3, big ? 7 : 5, big ? 7 : 8, 2, anim);
      hull(c, -half + 2, y - (big ? 14 : 12), len - 4, big ? 7 : 5, body, hi);
    }
    const top = big ? y - 14 : small ? y - 15 : type === "hovercraft" ? y - 13 : y - 12;
    switch (type) {
      case "armoredcar":
      case "scoutcar":
        hull(c, -3, top - 4, 8, 4, body, hi);
        barrel(c, -2, top - 2, a, 12, P.dark, 1);
        break;
      case "hovercraft":
        hull(c, -8, top - 4, 10, 4, body, hi);
        barrel(c, -6, top - 2, a, 12, P.dark, 1);
        hull(c, 4, top - 5, 6, 3, "#3a3f4f");
        break;
      case "artytruck": {
        // A cab, and a howitzer on the bed
        hull(c, -half, y - 20, 12, 9, body, hi);
        c.fillStyle = "#8fd3ff";
        c.fillRect(-half + 2, y - 18, 5, 3);
        hull(c, 2, y - 18, 12, 6, P.dark);
        barrel(c, 6, y - 17, Math.min(-0.4, Math.max(-1.3, a)), 22, "#3a3f4f", 2);
        break;
      }
      case "medtank":
      case "mbt":
        hull(c, -8, top - 6, 18, 6, body, hi);
        barrel(c, -6, top - 3, a, 20, P.dark, 2);
        break;
      case "heavytank":
        hull(c, -10, top - 8, 24, 8, body, hi);
        barrel(c, -8, top - 6, a, 24, P.dark, 2);
        barrel(c, -8, top - 2, a, 24, P.dark, 2);
        c.fillStyle = "#c8302a";
        c.fillRect(6, top - 6, 3, 3);
        break;
      case "flametank":
        hull(c, -6, top - 6, 14, 6, body, hi);
        barrel(c, -4, top - 3, a, 10, "#3a3f4f", 2);
        // The fuel drums on the back
        hull(c, 8, top - 8, 9, 8, "#c8302a", "#e8584a");
        c.fillStyle = "#ffcc33";
        c.fillRect(11, top - 5, 3, 1);
        break;
      case "rockettank":
      case "mlrs": {
        // A box of rockets on a mount, tilted so its front points up and never into the ground
        c.save();
        c.translate(2, top - 4);
        c.rotate(0.45);
        hull(c, -16, -8, 22, 9, body, hi);
        c.fillStyle = "#c8302a";
        for (let j = 0; j < 3; j++) c.fillRect(-17, -7 + j * 3, 1, 2);
        c.restore();
        hull(c, -4, top - 4, 10, 4, P.dark);
        break;
      }
      case "aatank":
      case "aa":
        hull(c, -7, top - 7, 15, 7, body, hi);
        barrel(c, -5, top - 6, Math.min(-0.6, a), 18, P.dark, 1);
        barrel(c, -3, top - 4, Math.min(-0.6, a), 18, P.dark, 1);
        // Radar dish
        c.fillStyle = INK;
        c.fillRect(8, top - 13, 1, 7);
        c.fillStyle = "#c8ced9";
        c.fillRect(5 + Math.round(Math.sin(anim * 2) * 2), top - 14, 7, 2);
        break;
      case "ifv":
        hull(c, -half - 2, y - 10, 4, 6, body);
        hull(c, -6, top - 5, 13, 5, body, hi);
        barrel(c, -4, top - 3, a, 16, P.dark, 1);
        break;
      case "launcher": {
        // A cab on the front, the missile on its erector behind
        hull(c, -half, y - 20, 12, 9, body, hi);
        c.fillStyle = "#8fd3ff";
        c.fillRect(-half + 2, y - 18, 5, 3);
        // Raising tilts the missile's nose up, away from the ground
        const ang = raise * 1.25;
        c.save();
        c.translate(half - 4, y - 13);
        c.rotate(ang);
        hull(c, -38, -4, 38, 5, "#c8ced9", "#e8eef6");
        c.fillStyle = "#c8302a";
        c.fillRect(-42, -3, 4, 3);
        c.fillRect(-4, -6, 3, 2);
        c.fillRect(-4, 2, 3, 2);
        c.fillStyle = INK;
        c.fillRect(-26, -4, 1, 5);
        c.restore();
        break;
      }
    }
    if (ally) star(c, -half + 10, y - 9);
  });
  if (ally) {
    // A friendly marker over it
    c.fillStyle = "#5ab0ff";
    c.fillRect(x - 2, y - 30, 5, 1);
    c.fillRect(x - 1, y - 29, 3, 1);
    c.fillRect(x, y - 28, 1, 1);
  }
}

export type HeliType = "attackheli" | "transport" | "scoutheli" | "aheli" | "lifter";
/** Enemy and allied helicopters: attack ships, twin-rotor transports and scouts. */
export function drawHeliType(c: Ctx, x: number, y: number, t: number, type: HeliType, hit: boolean) {
  const ally = type === "aheli" || type === "lifter";
  const P = SIDE[ally ? "ally" : "enemy"];
  const body = hit ? "#e8eef6" : type === "scoutheli" ? "#6a5a3a" : P.body;
  const hi = hit ? "#ffffff" : P.hi;
  const blade = (bx: number, by: number, w: number) => {
    const k = Math.abs(Math.round(Math.sin(t * 50 + bx) * w));
    c.fillStyle = INK;
    c.fillRect(bx - k - 1, by - 1, k * 2 + 3, 3);
    c.fillStyle = "#9aa1b0";
    c.fillRect(bx - k, by, k * 2 + 1, 1);
  };
  framed(c, x, ally, () => {
    if (type === "transport" || type === "lifter") {
      hull(c, -22, y - 5, 40, 10, body, hi);
      hull(c, -24, y - 2, 4, 6, body);
      c.fillStyle = "#8fd3ff";
      for (let k = -14; k < 14; k += 6) c.fillRect(k, y - 3, 3, 2);
      hull(c, -20, y - 8, 4, 3, P.dark);
      hull(c, 14, y - 9, 4, 4, P.dark);
      blade(-18, y - 10, 18);
      blade(16, y - 11, 18);
      c.fillStyle = INK;
      c.fillRect(-12, y + 6, 22, 1);
    } else {
      const small = type === "scoutheli";
      const L = small ? 22 : 30;
      // Cockpit forward on the left, tail boom back
      hull(c, -L / 2, y - 4, small ? 12 : 16, small ? 7 : 8, body, hi);
      c.fillStyle = "#8fd3ff";
      c.fillRect(-L / 2 + 1, y - 3, small ? 4 : 5, 3);
      hull(c, -L / 2 + (small ? 12 : 16), y - 2, L / 2 + 2, 3, body);
      hull(c, L / 2 + 1, y - 6, 3, 6, body);
      blade(-L / 2 + 6, y - 6, small ? 11 : 15);
      if (!small) {
        // Stub wings, a rocket pod and a missile under each
        hull(c, -6, y + 3, 12, 2, P.dark);
        hull(c, -7, y + 5, 5, 3, "#3a3f4f");
        hull(c, 2, y + 5, 6, 2, "#c8ced9");
        c.fillStyle = "#c8302a";
        c.fillRect(1, y + 5, 1, 2);
        c.fillStyle = INK;
        c.fillRect(-L / 2 - 2, y + 2, 2, 2);
      }
      c.fillStyle = INK;
      c.fillRect(-L / 2 + 1, y + 5, small ? 10 : 14, 1);
    }
    if (ally) star(c, 0, y);
  });
  if (ally) {
    c.fillStyle = "#5ab0ff";
    c.fillRect(x - 2, y - 18, 5, 1);
    c.fillRect(x - 1, y - 17, 3, 1);
  }
}

/**
 * The squad's gunship, facing right, its chin cannon aimed at `aim` and the squad, in their `crew`
 * colours, in its open door. `rope` is the length in px of the ladder hanging from the door.
 */
export function drawAllyGunship(
  c: Ctx,
  x: number,
  y: number,
  t: number,
  aim: number,
  hit: boolean,
  crew: string[],
  rope: number
) {
  // A hit only brightens it: it's under fire the whole way, so a white flash would never stop
  const P = SIDE.ally;
  const body = hit ? mix(P.body, "#ffffff", 0.35) : P.body;
  const hi = hit ? "#ffffff" : P.hi;
  const dark = hit ? mix(P.dark, "#ffffff", 0.3) : P.dark;
  if (rope > 0) {
    // The rope ladder, drawn first so it hangs behind the skids
    c.fillStyle = "#8a6a3a";
    c.fillRect(x - 3, y + 8, 1, rope);
    c.fillRect(x + 2, y + 8, 1, rope);
    c.fillStyle = "#b08a4a";
    for (let k = 3; k < rope; k += 3) c.fillRect(x - 2, y + 8 + k, 4, 1);
  }
  framed(c, x, true, () => {
    // Tail boom, fin and tail rotor, to the right in its own frame
    hull(c, 14, y - 3, 22, 4, body, hi);
    hull(c, 34, y - 11, 4, 9, body, hi);
    hull(c, 30, y + 1, 6, 2, dark);
    c.fillStyle = Math.floor(t * 4) % 2 ? "#ff3b3b" : "#6a1a1a";
    c.fillRect(35, y - 12, 2, 1);
    c.fillStyle = INK;
    if (Math.sin(t * 46) > 0) c.fillRect(38, y - 13, 1, 11);
    else c.fillRect(33, y - 7, 11, 1);
    // Engines on top, their exhausts
    hull(c, -9, y - 11, 20, 4, dark);
    hull(c, -4, y - 13, 9, 2, dark);
    c.fillStyle = "#2a2a30";
    c.fillRect(11, y - 10, 2, 2);
    // The cabin, armoured
    hull(c, -16, y - 7, 31, 14, body, hi);
    c.fillStyle = dark;
    for (let k = -12; k < 14; k += 6) c.fillRect(k, y + 5, 3, 1);
    // The open door, the squad in it and a door gun
    c.fillStyle = "#141820";
    c.fillRect(-6, y - 5, 13, 9);
    crew.slice(0, 4).forEach((col, i) => {
      const cx = 4 - i * 3;
      c.fillStyle = col;
      c.fillRect(cx, y - 1, 2, 4);
      c.fillRect(cx, y - 4, 2, 1);
      c.fillStyle = "#e0a878";
      c.fillRect(cx, y - 3, 2, 2);
    });
    if (crew.length) {
      c.fillStyle = "#3a3f4f";
      c.fillRect(-9, y + 1, 4, 1);
    }
    // The nose, glazed, and the chin turret under it
    hull(c, -27, y - 5, 12, 10, body, hi);
    c.fillStyle = "#8fd3ff";
    c.fillRect(-26, y - 4, 7, 4);
    c.fillStyle = "#d8f4ff";
    c.fillRect(-26, y - 4, 2, 1);
    hull(c, -24, y + 5, 6, 3, "#3a3f4f");
    // Stub wings: a rocket pod and a pair of missiles under each
    hull(c, -9, y + 6, 18, 2, dark);
    hull(c, -10, y + 9, 7, 3, "#3a3f4f");
    c.fillStyle = "#c8302a";
    c.fillRect(-11, y + 9, 1, 3);
    hull(c, 1, y + 9, 8, 2, "#c8ced9");
    c.fillStyle = "#c8302a";
    c.fillRect(0, y + 9, 1, 2);
    // Skids
    c.fillStyle = INK;
    c.fillRect(-15, y + 13, 26, 1);
    c.fillRect(-10, y + 8, 1, 5);
    c.fillRect(7, y + 8, 1, 5);
    star(c, 10, y - 1);
    // The main rotor, a blur
    c.fillStyle = INK;
    c.fillRect(0, y - 15, 2, 3);
    const k = Math.abs(Math.round(Math.sin(t * 50) * 30));
    c.fillRect(1 - k - 1, y - 16, k * 2 + 3, 3);
    c.fillStyle = "#9aa1b0";
    c.fillRect(1 - k, y - 15, k * 2 + 1, 1);
    c.globalAlpha = 0.25;
    c.fillStyle = INK;
    c.fillRect(-30, y - 15, 62, 1);
    c.globalAlpha = 1;
  });
  // The chin cannon, aimed in the world
  barrel(c, x + 21, y + 7, aim, 7, "#5a6070");
  c.fillStyle = "#5ab0ff";
  c.fillRect(x - 2, y - 22, 5, 1);
  c.fillRect(x - 1, y - 21, 3, 1);
}

/** Freight locomotive liveries: body, light edge, dark edge */
const LIVERY: [string, string, string][] = [
  ["#c8302a", "#e8584a", "#7a1a16"],
  ["#e8b020", "#ffd85a", "#8a6a10"],
  ["#3a6ab0", "#7ab0e8", "#2a4a80"],
  ["#4a8a3a", "#6aaa4a", "#2c5a24"],
];
const CONTAINER = ["#c8302a", "#3a6ab0", "#e8b020", "#4a8a3a", "#8a8a92", "#d8682a"];
/** A pair of wheels on a bogie, turning. */
function bogie(c: Ctx, x: number, y: number, anim: number) {
  c.fillStyle = INK;
  c.fillRect(x - 1, y - 5, 12, 2);
  for (const wx of [x + 2, x + 8]) {
    c.fillStyle = INK;
    disc(c, wx, y - 2, 2.6);
    c.fillStyle = "#2a2a30";
    disc(c, wx, y - 2, 1.6);
    c.fillStyle = "#8a8a92";
    c.fillRect(Math.round(wx + Math.cos(anim + wx) * 1.4), Math.round(y - 2 + Math.sin(anim + wx) * 1.4), 1, 1);
  }
}

/**
 * One car of a train, its left end at x and wheels on y. `kind` 0 is a diesel locomotive, 1 a
 * boxcar, 2 a tank car, 3 a container flat, 5 a passenger coach and anything else a coal hopper.
 * A locomotive's cab is at its right end when `front` is 1, its left when -1; `seed` picks colours.
 */
export function drawFreightCar(
  c: Ctx,
  x: number,
  y: number,
  kind: number,
  livery: number,
  anim: number,
  front: number,
  seed: number
) {
  const L = LIVERY[livery % LIVERY.length];
  const W = 40;
  // Underframe, coupler and bogies
  c.fillStyle = INK;
  c.fillRect(x, y - 7, W + 2, 2);
  c.fillRect(x - 1, y - 8, 2, 2);
  bogie(c, x + 4, y, anim);
  bogie(c, x + W - 14, y, anim);
  switch (kind) {
    case 0: {
      // Diesel locomotive: long hood, cab at the front, exhaust stack, headlamp
      hull(c, x + 2, y - 18, W - 2, 11, L[0], L[1]);
      c.fillStyle = L[2];
      c.fillRect(x + 2, y - 11, W - 2, 2);
      c.fillStyle = "#f0f0e8";
      c.fillRect(x + 2, y - 13, W - 2, 1);
      const cab = front > 0 ? x + W - 12 : x + 2;
      hull(c, cab, y - 23, 12, 16, L[0], L[1]);
      c.fillStyle = "#8fd3ff";
      c.fillRect(cab + 2, y - 21, 8, 4);
      c.fillStyle = "#d8f4ff";
      c.fillRect(cab + 2, y - 21, 2, 1);
      c.fillStyle = INK;
      c.fillRect(front > 0 ? x + 8 : x + W - 10, y - 21, 3, 3);
      for (let k = 0; k < 4; k++) c.fillRect(front > 0 ? x + 14 + k * 4 : x + 12 + k * 4, y - 16, 2, 3);
      c.fillStyle = "#ffe27a";
      c.fillRect(front > 0 ? x + W : x, y - 12, 2, 2);
      break;
    }
    case 1: {
      // Boxcar: planks, sliding door, a stencilled code
      const col = ["#8a3a24", "#6a4a3a", "#3a5a6a", "#7a2a2a"][seed % 4];
      hull(c, x + 1, y - 19, W, 12, col, mix(col, "#ffffff", 0.2));
      c.fillStyle = mix(col, "#000000", 0.25);
      for (let k = x + 3; k < x + W; k += 3) c.fillRect(k, y - 18, 1, 10);
      c.fillStyle = mix(col, "#000000", 0.4);
      c.fillRect(x + 16, y - 18, 9, 11);
      c.fillStyle = "#e8e4dc";
      c.fillRect(x + 4, y - 16, 7, 1);
      c.fillRect(x + 4, y - 14, 5, 1);
      break;
    }
    case 2: {
      // Tank car: barrel, dome, hazard plate
      hull(c, x + 2, y - 16, W - 2, 9, "#2a2a30", "#4a4a54");
      c.fillStyle = "#4a4a54";
      c.fillRect(x + 4, y - 15, W - 6, 1);
      hull(c, x + 18, y - 19, 6, 3, "#2a2a30");
      c.fillStyle = seed % 2 ? "#ff8a1a" : "#ffcc33";
      c.fillRect(x + 6, y - 13, 4, 4);
      break;
    }
    case 3: {
      // A flat car with a container on it
      const col = CONTAINER[seed % CONTAINER.length];
      hull(c, x + 1, y - 9, W, 2, "#5a4a3a");
      hull(c, x + 3, y - 20, W - 4, 11, col, mix(col, "#ffffff", 0.2));
      c.fillStyle = mix(col, "#000000", 0.3);
      for (let k = x + 5; k < x + W - 2; k += 2) c.fillRect(k, y - 19, 1, 9);
      c.fillStyle = "#e8e4dc";
      c.fillRect(x + 6, y - 17, 6, 2);
      break;
    }
    case 5: {
      // Passenger coach: windows with people in them, doors at the ends, the line's colours
      hull(c, x + 1, y - 21, W, 14, "#c8ced9", "#eef2f8");
      c.fillStyle = L[0];
      c.fillRect(x + 1, y - 11, W, 2);
      c.fillStyle = "#9aa1b0";
      c.fillRect(x + 1, y - 22, W, 1);
      const heads = ["#e0a878", "#8d5a3b", "#f2c29b", "#c68a5a", "#3a2a20"];
      for (let k = 0; k < 5; k++) {
        const wx = x + 6 + k * 6;
        c.fillStyle = "#1e2a3a";
        c.fillRect(wx, y - 19, 4, 5);
        if ((seed * 7 + k * 3) % 5 < 3) {
          c.fillStyle = heads[(seed + k) % heads.length];
          c.fillRect(wx + 1, y - 17, 2, 2);
        }
      }
      c.fillStyle = "#9aa1b0";
      c.fillRect(x + 2, y - 19, 2, 10);
      c.fillRect(x + W - 3, y - 19, 2, 10);
      break;
    }
    default: {
      // A hopper heaped with coal
      hull(c, x + 2, y - 17, W - 2, 10, "#4a4a54", "#6a6a74");
      c.fillStyle = "#1e1e24";
      for (let k = 0; k < W - 4; k++)
        c.fillRect(x + 3 + k, y - 18 - Math.round(Math.sin(((k + 0.5) / (W - 4)) * Math.PI) * 2), 1, 2);
      c.fillStyle = "#2a2a30";
      for (const k of [8, 18, 28]) c.fillRect(x + k, y - 8, 5, 2);
    }
  }
}

/**
 * A metro train, its middle at x and wheels on y, heading `dir`. The windows show passengers, or
 * the squad's colours when `riders` is non-empty; `doors` from 0 to 1 slides the doors open.
 */
export function drawMetroTrain(
  c: Ctx,
  x: number,
  y: number,
  cars: number,
  doors: number,
  riders: string[],
  dir: number,
  seed: number
) {
  const W = 34;
  const x0 = Math.round(x - (cars * W) / 2);
  const r = rng(seed);
  const heads = ["#e0a878", "#8d5a3b", "#f2c29b", "#c68a5a", "#3a2a20"];
  for (let i = 0; i < cars; i++) {
    const cx = x0 + i * W;
    const first = i === 0;
    const last = i === cars - 1;
    // Body, its roof and stripe
    hull(c, cx + 1, y - 22, W - 2, 16, "#c8ced9", "#eef2f8");
    c.fillStyle = "#9aa1b0";
    c.fillRect(cx + 1, y - 23, W - 2, 1);
    c.fillStyle = "#2a6ad8";
    c.fillRect(cx + 1, y - 10, W - 2, 2);
    // Windows, someone in most of them
    for (let k = 0; k < 3; k++) {
      const wx = cx + 3 + k * 10;
      c.fillStyle = "#1e2a3a";
      c.fillRect(wx, y - 19, 7, 6);
      const rider = riders.length
        ? riders[(i * 3 + k) % riders.length]
        : r() < 0.6
          ? heads[Math.floor(r() * heads.length)]
          : "";
      if (rider && (riders.length ? i * 3 + k < riders.length * 2 : true)) {
        c.fillStyle = riders.length ? rider : "#3a4a5a";
        c.fillRect(wx + 2, y - 15, 3, 2);
        c.fillStyle = riders.length ? "#e0a878" : rider;
        c.fillRect(wx + 2, y - 17, 3, 2);
      }
      c.fillStyle = "#5a7a9a";
      c.fillRect(wx, y - 19, 7, 1);
    }
    // Its doors: two per car, sliding apart
    const gap = Math.round(doors * 3);
    for (const dx of [8, 24]) {
      c.fillStyle = "#9aa1b0";
      c.fillRect(cx + dx - 2 - gap, y - 20, 2, 13);
      c.fillRect(cx + dx + 1 + gap, y - 20, 2, 13);
      if (gap) {
        c.fillStyle = "#ffe8b0";
        c.fillRect(cx + dx, y - 20, gap * 2, 13);
      }
    }
    // A cab at each end, sloped
    if (first || last) {
      const nose = first ? cx - 2 : cx + W;
      c.fillStyle = INK;
      c.fillRect(nose, y - 18, 3, 12);
      c.fillStyle = "#c8ced9";
      c.fillRect(first ? nose + 1 : nose, y - 17, 2, 10);
      const ahead = (first && dir < 0) || (last && dir > 0);
      c.fillStyle = ahead ? "#fff6c8" : "#ff3b3b";
      c.fillRect(first ? cx + 1 : cx + W - 3, y - 9, 2, 1);
    }
    // Couplings, bogies
    if (!last) {
      c.fillStyle = INK;
      c.fillRect(cx + W - 1, y - 12, 2, 2);
    }
    bogie(c, cx + 3, y, seed + i);
    bogie(c, cx + W - 13, y, seed + i);
  }
}

const PAINT_CAR = ["#c8302a", "#3a6ab0", "#e8e4dc", "#2a2a30", "#4a8a3a", "#8a8a92", "#d8a020", "#6a3a8a"];
/**
 * Road traffic facing `dir`, wheels on y. `siren` flashes the emergency vehicles' lights and
 * `riders` fills that many bus windows.
 */
export function drawCar(
  c: Ctx,
  x: number,
  y: number,
  kind: string,
  color: number,
  dir: number,
  anim: number,
  siren: boolean,
  t: number,
  riders: number
) {
  const wheel = (wx: number, r = 2.4) => {
    c.fillStyle = INK;
    disc(c, wx, y - 2, r);
    c.fillStyle = "#4a4a54";
    disc(c, wx, y - 2, r - 1.2);
    c.fillStyle = "#9aa1b0";
    c.fillRect(Math.round(wx + Math.cos(anim + wx) * 1.2), Math.round(y - 2 + Math.sin(anim + wx) * 1.2), 1, 1);
  };
  const flash = Math.floor(t * 8) % 2;
  framed(c, x, dir > 0, () => {
    // Drawn facing left, its nose at -half
    switch (kind) {
      case "bus": {
        hull(c, -20, y - 17, 40, 13, "#d8a020", "#f0c84a");
        c.fillStyle = "#c8302a";
        c.fillRect(-20, y - 8, 40, 2);
        for (let k = 0; k < 6; k++) {
          const wx = -17 + k * 6;
          c.fillStyle = "#1e2a3a";
          c.fillRect(wx, y - 15, 4, 5);
          if (k < riders) {
            c.fillStyle = ["#e0a878", "#8d5a3b", "#f2c29b", "#c68a5a"][k % 4];
            c.fillRect(wx + 1, y - 13, 2, 2);
          }
        }
        c.fillStyle = "#8fd3ff";
        c.fillRect(-20, y - 15, 2, 6);
        c.fillStyle = "#1e1e24";
        c.fillRect(-16, y - 19, 8, 2);
        c.fillStyle = "#ffcc33";
        c.fillRect(-15, y - 19, 6, 1);
        wheel(-13);
        wheel(13);
        break;
      }
      case "truck": {
        hull(c, -4, y - 16, 19, 11, PAINT_CAR[color % PAINT_CAR.length], "#ffffff");
        hull(c, -15, y - 13, 10, 8, "#5a6070", "#8a90a0");
        c.fillStyle = "#8fd3ff";
        c.fillRect(-14, y - 12, 4, 3);
        wheel(-10);
        wheel(5);
        wheel(11);
        break;
      }
      case "firetruck": {
        hull(c, -17, y - 13, 34, 9, "#c8302a", "#e8584a");
        hull(c, -17, y - 16, 10, 4, "#c8302a", "#e8584a");
        c.fillStyle = "#8fd3ff";
        c.fillRect(-16, y - 15, 4, 2);
        c.fillStyle = "#e8e4dc";
        c.fillRect(-6, y - 9, 22, 1);
        // The ladder along the top
        c.fillStyle = "#c8ced9";
        c.fillRect(-6, y - 16, 22, 1);
        c.fillRect(-6, y - 14, 22, 1);
        for (let k = -5; k < 16; k += 3) c.fillRect(k, y - 15, 1, 1);
        c.fillStyle = siren && flash ? "#4ab0ff" : "#ff3b3b";
        c.fillRect(-13, y - 17, 3, 1);
        wheel(-11);
        wheel(8);
        wheel(13);
        break;
      }
      case "ambulance": {
        hull(c, -12, y - 15, 24, 11, "#eef2f8", "#ffffff");
        c.fillStyle = "#c8302a";
        c.fillRect(2, y - 12, 5, 1);
        c.fillRect(4, y - 14, 1, 5);
        c.fillRect(-12, y - 7, 24, 1);
        c.fillStyle = "#8fd3ff";
        c.fillRect(-11, y - 13, 4, 3);
        c.fillStyle = siren ? (flash ? "#ff3b3b" : "#4ab0ff") : "#8a8a92";
        c.fillRect(-9, y - 17, 4, 1);
        wheel(-7);
        wheel(7);
        break;
      }
      default: {
        // Cars: a saloon, a taxi, police, a van
        const van = kind === "van";
        const body = kind === "taxi" ? "#e8c020" : kind === "police" ? "#1e2a4a" : PAINT_CAR[color % PAINT_CAR.length];
        const half = van ? 11 : 9;
        hull(c, -half, y - 9, half * 2, 5, body, mix(body, "#ffffff", 0.3));
        hull(
          c,
          van ? -half : -half + 3,
          y - (van ? 14 : 12),
          van ? half * 2 - 2 : half * 2 - 7,
          van ? 5 : 3,
          body,
          mix(body, "#ffffff", 0.3)
        );
        c.fillStyle = "#8fd3ff";
        c.fillRect(van ? -half + 1 : -half + 4, y - (van ? 13 : 11), 3, 2);
        if (!van) c.fillRect(-half + 8, y - 11, 3, 2);
        if (kind === "police") {
          c.fillStyle = "#e8eef6";
          c.fillRect(-half, y - 7, half * 2, 1);
          c.fillStyle = siren ? (flash ? "#ff3b3b" : "#4ab0ff") : "#5a5a64";
          c.fillRect(-2, y - 14, 2, 1);
          c.fillStyle = siren ? (flash ? "#4ab0ff" : "#ff3b3b") : "#5a5a64";
          c.fillRect(0, y - 14, 2, 1);
        }
        if (kind === "taxi") {
          c.fillStyle = "#1e1e24";
          c.fillRect(-2, y - 14, 4, 1);
          c.fillStyle = "#fff6c8";
          c.fillRect(-1, y - 14, 2, 1);
        }
        wheel(-half + 4, 2);
        wheel(half - 4, 2);
      }
    }
    // Headlamps forward, tail lamps behind
    c.fillStyle = "#fff6c8";
    c.fillRect(-CAR_HALF(kind) - 1, y - 8, 1, 1);
    c.fillStyle = "#ff3b3b";
    c.fillRect(CAR_HALF(kind), y - 8, 1, 1);
  });
}
const CAR_HALF = (kind: string) =>
  kind === "bus"
    ? 20
    : kind === "firetruck"
      ? 17
      : kind === "truck"
        ? 15
        : kind === "ambulance"
          ? 12
          : kind === "van"
            ? 11
            : 9;

export type PlaneType = "bomber" | "fighter" | "attacker" | "transport" | "heavy" | "stealth" | "dive" | "rocketjet";
/** Aircraft of every type; enemy ones fly left, allied ones right. */
export function drawPlane(c: Ctx, x: number, y: number, type: PlaneType, ally: boolean, t: number) {
  const P = SIDE[ally ? "ally" : "enemy"];
  const body = ally ? P.body : "#4a5060";
  const hi = ally ? P.hi : "#6a7080";
  framed(c, x, ally, () => {
    if (type === "heavy") {
      // Glazed nose, four engines under a long wing, twin tail
      hull(c, -38, y - 4, 76, 8, body, hi);
      hull(c, -40, y - 2, 3, 5, body);
      c.fillStyle = "#8fd3ff";
      c.fillRect(-39, y - 2, 3, 2);
      c.fillRect(-34, y - 4, 4, 1);
      hull(c, -16, y - 1, 36, 2, P.dark);
      for (const ex of [-12, -3, 8, 17]) {
        hull(c, ex, y + 2, 5, 3, "#3a3f4f");
        const k = Math.abs(Math.round(Math.sin(t * 46 + ex) * 3));
        c.fillStyle = INK;
        c.fillRect(ex - 1, y + 3 - k, 1, k * 2 + 1);
      }
      for (const tx of [32, 36]) hull(c, tx, y - 14, 3, 10, body);
      hull(c, 28, y - 6, 14, 2, P.dark);
      c.fillStyle = INK;
      for (const gx of [-22, 4]) c.fillRect(gx, y + 4, 1, 3);
      if (ally) star(c, -4, y - 1);
      return;
    }
    if (type === "stealth") {
      // A flying wing with a low cockpit hump and a sawtooth trailing edge
      const skin = ally ? "#2e3c52" : "#22242c";
      const edge = ally ? "#4a5c78" : "#3a3e4a";
      hull(c, -26, y - 1, 52, 3, skin, edge);
      hull(c, -12, y - 4, 16, 3, skin, edge);
      c.fillStyle = "#4a6a8a";
      c.fillRect(-9, y - 3, 5, 1);
      c.fillStyle = skin;
      for (let k = 0; k < 6; k++) c.fillRect(6 + k * 4, y + 2, 2, k % 2 ? 2 : 1);
      // No afterburner, just a shimmer at the exhausts
      if (Math.floor(t * 12) % 2) {
        c.fillStyle = "#6a7080";
        c.fillRect(20, y - 2, 2, 1);
      }
      return;
    }
    if (type === "dive") {
      // Propeller, long canopy, gull wings, fixed wheels
      hull(c, -15, y - 2, 30, 5, body, hi);
      hull(c, -17, y - 1, 2, 3, "#3a3f4f");
      const k = Math.abs(Math.round(Math.sin(t * 60) * 4));
      c.fillStyle = INK;
      c.fillRect(-19, y + 1 - k, 1, k * 2 + 1);
      c.fillStyle = "#8fd3ff";
      c.fillRect(-9, y - 4, 10, 2);
      c.fillStyle = INK;
      c.fillRect(-10, y - 5, 12, 1);
      for (let n = 0; n < 4; n++) c.fillRect(-6 + n * 3, y + 3 + Math.min(n, 3 - n), 3, 1);
      hull(c, -6, y + 5, 3, 3, P.dark);
      hull(c, 4, y + 5, 3, 3, P.dark);
      hull(c, 12, y - 7, 3, 5, body);
      hull(c, 10, y - 2, 8, 2, P.dark);
      if (ally) star(c, 2, y);
      return;
    }
    if (type === "rocketjet") {
      // A rocket pod under each wing, twin tails
      hull(c, -16, y - 2, 32, 5, body, hi);
      c.fillStyle = INK;
      for (let k = 0; k < 4; k++) c.fillRect(-1 + k, y + 2 + k, 9, 1);
      c.fillStyle = P.dark;
      for (let k = 0; k < 3; k++) c.fillRect(k, y + 2 + k, 7, 1);
      for (const px of [-8, 3]) {
        hull(c, px, y + 5, 7, 2, "#3a3f4f");
        c.fillStyle = "#c8302a";
        c.fillRect(px - 1, y + 5, 1, 2);
      }
      hull(c, 10, y - 8, 3, 6, body);
      hull(c, 13, y - 6, 2, 4, body);
      c.fillStyle = "#8fd3ff";
      c.fillRect(-12, y - 3, 4, 1);
      c.fillStyle = Math.floor(t * 30) % 2 ? "#ffe27a" : "#ff8a1a";
      c.fillRect(16, y - 1, 2, 2);
      if (ally) star(c, 0, y);
      return;
    }
    if (type === "transport") {
      // High wing with four props under it, tall tail
      hull(c, -32, y - 4, 62, 9, body, hi);
      hull(c, -34, y - 2, 4, 5, body);
      hull(c, -10, y - 8, 26, 3, P.dark);
      for (const ex of [-6, 2, 8, 14]) {
        hull(c, ex, y - 6, 3, 3, "#3a3f4f");
        const k = Math.abs(Math.round(Math.sin(t * 50 + ex) * 3));
        c.fillStyle = INK;
        c.fillRect(ex, y - 6 - k, 1, k * 2 + 1);
      }
      hull(c, 24, y - 15, 5, 11, body);
      hull(c, 20, y - 5, 12, 2, P.dark);
      c.fillStyle = "#8fd3ff";
      c.fillRect(-31, y - 3, 4, 2);
      c.fillStyle = INK;
      c.fillRect(-20, y + 6, 1, 2);
      c.fillRect(10, y + 6, 1, 2);
    } else if (type === "bomber") {
      hull(c, -26, y - 3, 52, 6, body, hi);
      c.fillStyle = INK;
      for (let k = 0; k < 7; k++) c.fillRect(-6 + k, y - 3 - k, 18, 1);
      for (let k = 0; k < 4; k++) c.fillRect(-4 + k, y + 3 + k, 14, 1);
      c.fillStyle = body;
      for (let k = 0; k < 6; k++) c.fillRect(-5 + k, y - 3 - k, 16, 1);
      for (const ex of [0, 6]) hull(c, ex, y + 2, 4, 2, P.dark);
      hull(c, 22, y - 9, 4, 6, body);
      c.fillStyle = "#8fd3ff";
      c.fillRect(-25, y - 2, 3, 2);
    } else if (type === "fighter") {
      hull(c, -14, y - 2, 28, 4, body, hi);
      c.fillStyle = INK;
      for (let k = 0; k < 6; k++) c.fillRect(-2 + k, y + 1 + k, 8, 1);
      c.fillStyle = body;
      for (let k = 0; k < 5; k++) c.fillRect(-1 + k, y + 1 + k, 6, 1);
      hull(c, 10, y - 7, 3, 5, body);
      c.fillStyle = "#8fd3ff";
      c.fillRect(-10, y - 3, 4, 1);
      c.fillStyle = INK;
      c.fillRect(-16, y, 3, 1);
    } else {
      hull(c, -16, y - 2, 30, 5, body, hi);
      hull(c, -6, y - 1, 12, 2, P.dark);
      c.fillStyle = INK;
      c.fillRect(-4, y + 3, 1, 3);
      c.fillRect(3, y + 3, 1, 3);
      hull(c, 6, y - 5, 6, 3, P.dark);
      hull(c, 12, y - 8, 3, 6, body);
      c.fillStyle = "#8fd3ff";
      c.fillRect(-12, y - 3, 3, 1);
    }
    // Afterburner
    c.fillStyle = Math.floor(t * 30) % 2 ? "#ffe27a" : "#ff8a1a";
    c.fillRect(type === "bomber" ? 26 : type === "fighter" ? 14 : 15, y - 1, 2, 2);
    if (ally) star(c, 0, y);
  });
}

/** Dug-in weapon emplacements: a sandbagged ring with the gun on its mount. */
export function drawEmplacement(
  c: Ctx,
  x: number,
  y: number,
  kind: "flakgun" | "howitzer" | "samsite" | "rocketpod",
  aim: number,
  hit: boolean,
  recoil: number
) {
  const metal = hit ? "#e8eef6" : "#5a6248";
  const hi = hit ? "#ffffff" : "#7a8462";
  // Sandbags round the pit
  for (let k = -14; k <= 10; k += 5) hull(c, x + k, y - 5, 5, 4, "#c4ae78", "#d8c890");
  switch (kind) {
    case "flakgun":
      hull(c, x - 5, y - 11, 10, 6, metal, hi);
      barrel(c, x - 3, y - 11, Math.min(-0.5, aim), 16, "#3a3f4f", 1);
      barrel(c, x, y - 10, Math.min(-0.5, aim), 16, "#3a3f4f", 1);
      break;
    case "howitzer": {
      hull(c, x - 8, y - 10, 16, 5, metal, hi);
      const back = recoil * 12;
      barrel(c, x - Math.cos(aim) * back, y - 12 - Math.sin(aim) * back, Math.min(-0.3, aim), 24, "#3a3f4f", 2);
      hull(c, x - 2, y - 16, 6, 6, metal, hi);
      break;
    }
    case "samsite":
      hull(c, x - 6, y - 10, 12, 5, metal, hi);
      c.save();
      c.translate(x, y - 12);
      c.rotate(Math.max(-1.2, Math.min(-0.4, aim < -Math.PI / 2 ? -(Math.PI + aim) : aim)));
      for (const dy of [-5, 0]) {
        hull(c, -18, dy - 2, 18, 3, "#c8ced9", "#e8eef6");
        c.fillStyle = "#c8302a";
        c.fillRect(-20, dy - 2, 2, 3);
      }
      c.restore();
      break;
    default:
      hull(c, -8 + x, y - 11, 16, 6, metal, hi);
      hull(c, x - 9, y - 19, 16, 8, metal, hi);
      c.fillStyle = INK;
      for (let k = 0; k < 4; k++) for (let j = 0; j < 2; j++) c.fillRect(x - 8 + k * 4, y - 18 + j * 4, 2, 2);
  }
}

/**
 * A submarine centred on (x, y). Submerged, the whole boat is drawn for the scene's water to tint,
 * and with `scope` set its periscope breaks the surface at that screen y. Allied boats carry a star.
 */
export function drawSubmarine(
  c: Ctx,
  x: number,
  y: number,
  t: number,
  submerged: boolean,
  hit: boolean,
  sink: number,
  facing = -1,
  ally = false,
  scope: number | null = null
) {
  mirrored(c, x, facing, () => submarineFacingLeft(c, x, y, t, submerged, hit, sink, ally, scope));
}

function submarineFacingLeft(
  c: Ctx,
  x: number,
  y: number,
  t: number,
  submerged: boolean,
  hit: boolean,
  sink: number,
  ally: boolean,
  scope: number | null
) {
  const by = Math.round(y + (submerged ? 0 : Math.sin(t * 1.4) * 0.6) + sink * 14);
  if (submerged && scope !== null && scope < by - 15) {
    // The periscope, up through the water to the surface
    c.fillStyle = INK;
    c.fillRect(x + 1, scope - 3, 3, by - 12 - scope);
    c.fillStyle = "#4a525e";
    c.fillRect(x + 2, scope - 2, 1, by - 14 - scope);
    c.fillRect(x + 1, scope - 3, 3, 1);
    c.fillStyle = "#e8f8ff";
    for (let k = 0; k < 6; k++) if (Math.sin(t * 7 + k) > 0) c.fillRect(x + 6 + k * 2, scope, 1, 1);
  }
  if (sink) c.globalAlpha = Math.max(0, 1 - sink);
  const body = hit ? "#c8ced9" : ally ? "#3a5068" : "#2e343e";
  c.fillStyle = INK;
  for (let k = -3; k <= 3; k++) {
    const half = Math.round(Math.sqrt(1 - (k / 3.6) ** 2) * 33);
    c.fillRect(x - half - 1, by + k - 1, half * 2 + 2, 3);
  }
  c.fillStyle = body;
  for (let k = -3; k <= 3; k++) {
    const half = Math.round(Math.sqrt(1 - (k / 3.6) ** 2) * 32);
    c.fillRect(x - half, by + k, half * 2, 1);
  }
  hull(c, x - 2, by - 10, 10, 7, body, hit ? "#ffffff" : ally ? "#5a7a9a" : "#4a525e");
  c.fillStyle = INK;
  c.fillRect(x + 2, by - 15, 1, 5);
  if (ally) {
    c.fillStyle = "#c8d8e8";
    c.fillRect(x - 28, by - 1, 14, 1);
    star(c, x + 3, by - 7);
  } else {
    c.fillStyle = "#c8302a";
    c.fillRect(x - 26, by - 1, 4, 1);
  }
  c.globalAlpha = 1;
}
