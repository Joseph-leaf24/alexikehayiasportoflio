/**
 * The old engine's pixel-art painters (`components/pixel/art.ts`, read-only) wrapped as art sets: player loadouts,
 * troops in each theme's uniform, terrain blocks cut into 8×8 tiles, props, clouds and backdrops. Painting happens
 * only in a browser (the painters need a canvas), on first use of each set.
 */
import {
  FRAMES,
  makeAssets,
  makeAtlas,
  makeBackdrop,
  makeCanvas,
  makeCloud,
  makeProps,
  makeTroops,
  mix,
  playerSprite,
  INK,
} from "@/components/pixel/art";
import type { Backdrop } from "@/components/pixel/themes";
import { materials, type AtlasBuilder } from "@/engine";
import type { ArtFrame } from "../game/defs";
import { tilesFromSheet } from "./ArtBook";

/** Poses in the old humanoid layout: idle, four running frames, jumping. */
export const HUMANOID_POSES = FRAMES.jump + 1;

/** One player's frames with a weapon, [pose][facing] flattened, pivoted at the feet as the old scene drew them. */
export function paintPlayer(id: number, color: string, weapon: number): ArtFrame[] {
  const out: ArtFrame[] = [];
  for (let f = 0; f < HUMANOID_POSES; f++) {
    const right = playerSprite(id, color, f, false, weapon);
    const left = playerSprite(id, color, f, true, weapon);
    out.push({ canvas: right, ox: 6, oy: right.height - 1 }, { canvas: left, ox: 9, oy: left.height - 1 });
  }
  return out;
}

export type TroopKind = keyof ReturnType<typeof makeTroops>;

const troopCache = new Map<Backdrop, ReturnType<typeof makeTroops>>();
const troopsFor = (look: Backdrop) => {
  let t = troopCache.get(look);
  if (!t) troopCache.set(look, (t = makeTroops(look)));
  return t;
};

/** Kinds pivoted at their middle rather than at a soldier's feet (the old scene's `troopSprite` offsets). */
const CENTRED = new Set<TroopKind>(["turret", "crawler", "dog", "mg", "mortar"]);

/** A troop kind in a theme's uniform, [pose][facing] flattened. */
export function paintTroop(look: Backdrop, kind: TroopKind): ArtFrame[] {
  const set = troopsFor(look)[kind] as HTMLCanvasElement[] | HTMLCanvasElement[][];
  const poses = (Array.isArray(set[0]) ? set : [set]) as HTMLCanvasElement[][];
  const centred = CENTRED.has(kind);
  const out: ArtFrame[] = [];
  for (const [right, left] of poses) {
    out.push(
      { canvas: right, ox: centred ? right.width >> 1 : 6, oy: right.height - 1 },
      { canvas: left, ox: centred ? left.width >> 1 : left.width - 7, oy: left.height - 1 }
    );
  }
  return out;
}

/** Parachutes after the old scene's: a canopy in each player's colour, then an olive one for troops. */
export function paintChutes(colors: readonly string[]): ArtFrame[] {
  return [...colors, "#6a7048"].map((color) => {
    const [cv, c] = makeCanvas(19, 14);
    c.fillStyle = INK;
    c.fillRect(0, 0, 19, 5);
    c.fillStyle = color;
    c.fillRect(1, 1, 17, 3);
    c.fillStyle = mix(color, "#ffffff", 0.4);
    c.fillRect(3, 1, 13, 1);
    c.fillStyle = "#c9ced8";
    c.fillRect(1, 5, 1, 8);
    c.fillRect(17, 5, 1, 8);
    c.fillRect(9, 5, 1, 9);
    // Pivot at the wearer's feet, as the old scene drew it 34 px above them
    return { canvas: cv, ox: 9, oy: 34 };
  });
}

/** The longest piece of a strip (`paintStrip`): a longer run repeats it. */
export const STRIP_PIECE = 8;

/**
 * A run of plain colour along x or y at every length from 1 to `STRIP_PIECE` px (pose = length - 1), so a run of any
 * length is whole pieces plus one. `colors` are its rows (along x) or columns (along y), one px each; the pivot is
 * the run's top-left.
 */
export function paintStrip(axis: "x" | "y", colors: readonly string[]): ArtFrame[] {
  const out: ArtFrame[] = [];
  for (let len = 1; len <= STRIP_PIECE; len++) {
    const [c, ctx] = axis === "x" ? makeCanvas(len, colors.length) : makeCanvas(colors.length, len);
    colors.forEach((color, k) => {
      ctx.fillStyle = color;
      if (axis === "x") ctx.fillRect(0, k, len, 1);
      else ctx.fillRect(k, 0, 1, len);
    });
    out.push({ canvas: c, ox: 0, oy: 0 });
  }
  return out;
}

/** A fixed shape drawn by hand, in `poses` poses; `paint` draws pose k with (0, 0) at the pivot. */
export interface ShapeSpec {
  readonly w: number;
  readonly h: number;
  /** The pivot, from the canvas's top-left. */
  readonly ox: number;
  readonly oy: number;
  readonly poses?: number;
  readonly paint: (ctx: CanvasRenderingContext2D, pose: number) => void;
}

export function paintShape(s: ShapeSpec): ArtFrame[] {
  return Array.from({ length: s.poses ?? 1 }, (_, k) => {
    const [c, ctx] = makeCanvas(s.w, s.h);
    ctx.translate(s.ox, s.oy);
    s.paint(ctx, k);
    return { canvas: c, ox: s.ox, oy: s.oy };
  });
}

export function paintClouds(n: number): ArtFrame[] {
  return Array.from({ length: Math.max(1, n) }, (_, i) => ({ canvas: makeCloud(i + 1), ox: 0, oy: 0 }));
}

export type BackdropLayerName = "peaks" | "far" | "mid" | "near";

/** How far the nearest backdrop layer's bottom row is stretched down, so a deep valley never shows sky below it. */
const NEAR_SKIRT = 360;

let assets: ReturnType<typeof makeAssets> | null = null;

/**
 * One of the old scene's props (`ASSETS`, outlined): lit, then in shadow for behind the blocks, each facing right
 * then left, pivoted at its foot as the old scene drew it.
 */
export function paintAsset(id: string): ArtFrame[] {
  assets ??= makeAssets();
  const a = assets(id);
  return [a.img[0], a.img[1], a.dark[0], a.dark[1]].map((c) => ({ canvas: c, ox: c.width >> 1, oy: c.height - 2 }));
}

/** The last backdrop painted: its four layers are asked for one after another. */
let lastBackdrop: { kind: Backdrop; art: ReturnType<typeof makeBackdrop> } | null = null;

/**
 * One layer of a theme's backdrop, pivoted at its bottom-left so it stands on the horizon; the cave's far rock is
 * pivoted at its top-left, since it hangs from the top of the view.
 */
export function paintBackdrop(kind: Backdrop, layer: BackdropLayerName): ArtFrame[] {
  if (lastBackdrop?.kind !== kind) lastBackdrop = { kind, art: makeBackdrop(kind) };
  const art = lastBackdrop.art[layer];
  if (!art) return [];
  if (kind === "cave" && layer === "far") return [{ canvas: art, ox: 0, oy: 0 }];
  if (layer !== "near") return [{ canvas: art, ox: 0, oy: art.height }];
  // The old scene stretched the near layer's bottom row down the screen; here it is painted on once
  const [c, ctx] = makeCanvas(art.width, art.height + NEAR_SKIRT);
  ctx.drawImage(art, 0, 0);
  ctx.drawImage(art, 0, art.height - 1, art.width, 1, 0, art.height, art.width, NEAR_SKIRT);
  return [{ canvas: c, ox: 0, oy: art.height }];
}

export type PropName = keyof ReturnType<typeof makeProps>;
let props: ReturnType<typeof makeProps> | null = null;

export function paintProp(name: PropName): ArtFrame[] {
  props ??= makeProps();
  const c = props[name];
  return [{ canvas: c, ox: c.width >> 1, oy: c.height - 1 }];
}

/**
 * Materials the old painter has no block for, drawn as their nearest old kind. A beam gets no tile: what it holds up
 * is drawn on legs, posts or a back wall, which show through it.
 */
const STAND_IN: Record<number, number> = {
  [materials.ACID]: materials.SLIME,
  [materials.RUBBLE]: materials.GRAVEL,
  [materials.BEAM]: materials.AIR,
};

/**
 * The old `makeAtlas` block sheet cut into 8×8 tiles per material and variant. Water is drawn see-through, since
 * the old scene's depth-shaded sea isn't reproduced yet and swimmers should show through it.
 */
export function paintTiles(atlas: AtlasBuilder, tiles: Int32Array): void {
  const sheet = makeAtlas();
  const make = (w: number, h: number) => makeCanvas(w, h);
  tilesFromSheet(atlas, tiles, sheet, make, materials.MATERIAL_COUNT, (m) => STAND_IN[m] ?? m);
  for (let v = 0; v < 4; v++) {
    const [c, ctx] = makeCanvas(8, 8);
    ctx.globalAlpha = 0.72;
    ctx.drawImage(sheet, materials.WATER * 8, v * 8, 8, 8, 0, 0, 8, 8);
    tiles[materials.WATER * 4 + v] = atlas.add(`tile:water:${v}`, c);
  }
}
