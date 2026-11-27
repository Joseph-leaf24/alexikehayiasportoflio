/**
 * The art book: every art set the content registered (`content.art`), painted into the renderer's atlas and looked
 * up by (art index, pose, facing). Sets are painted on first use (or at boot when `eager`), so the 136 player
 * loadouts cost nothing until a player picks one up. Headless there is no atlas: lookups return -1 and the
 * simulation never notices, because it never touches art.
 */
import { TILE, VARIANTS, type AtlasBuilder, type ReadonlyRegistry, type Terrain } from "@/engine";
import type { ArtDef } from "../game/defs";

/** Paints the terrain's material tiles and fills `terrain.tiles` (material × VARIANTS + variant → frame). */
export type TilePainter = (atlas: AtlasBuilder, tiles: Int32Array) => void;

const NONE = new Int32Array(0);

export class ArtBook {
  /** Set by the art feature: the old engine's block painter, cut into 8×8 tiles. */
  tilePainter: TilePainter | null = null;
  private atlas: AtlasBuilder | null = null;
  private frames: Int32Array[] = [];
  private facings = new Uint8Array(0);
  /** Sets that failed to paint, so a broken painter is reported once and then drawn as nothing. */
  private failed = new Uint8Array(0);

  constructor(private readonly registry: ReadonlyRegistry<ArtDef>) {}

  get attached(): boolean {
    return this.atlas !== null;
  }

  /** Paint the eager sets and the terrain tiles into `atlas`; later lookups paint the rest on demand. */
  attach(atlas: AtlasBuilder, terrain?: Terrain): void {
    this.atlas = atlas;
    const n = this.registry.size + 1;
    this.frames = new Array<Int32Array>(n).fill(NONE);
    this.facings = new Uint8Array(n);
    this.failed = new Uint8Array(n);
    for (let k = 1; k < n; k++) if (this.registry.get(k).eager) this.paint(k);
    if (terrain && this.tilePainter) this.tilePainter(atlas, terrain.tiles);
  }

  detach(): void {
    this.atlas = null;
    this.frames = [];
  }

  /** The art index for an id (0 when unknown). */
  index(id: string): number {
    return this.registry.index(id);
  }

  /** Atlas frame for a set's pose facing right (facing ≥ 0) or left; -1 when unavailable. */
  frame(art: number, pose: number, facing: number): number {
    const f = this.ensure(art);
    if (f.length === 0) return -1;
    const two = this.facings[art] === 2;
    const poses = two ? f.length >> 1 : f.length;
    const p = pose < 0 ? 0 : pose >= poses ? poses - 1 : pose;
    return two ? f[p * 2 + (facing < 0 ? 1 : 0)] : f[p];
  }

  /** Whether a left-facing draw of this set needs the renderer's mirror flag (single-facing sets). */
  mirrors(art: number): boolean {
    return this.facings[art] !== 2;
  }

  /** Poses in a set (0 until painted). */
  poses(art: number): number {
    const f = this.ensure(art);
    return this.facings[art] === 2 ? f.length >> 1 : f.length;
  }

  private ensure(art: number): Int32Array {
    if (this.atlas === null || art <= 0 || art >= this.frames.length) return NONE;
    const f = this.frames[art];
    return f.length > 0 || this.failed[art] ? f : this.paint(art);
  }

  private paint(art: number): Int32Array {
    const atlas = this.atlas;
    if (!atlas) return NONE;
    const def = this.registry.get(art);
    const id = this.registry.id(art);
    try {
      const painted = def.paint();
      const out = new Int32Array(painted.length);
      for (let k = 0; k < painted.length; k++) {
        const p = painted[k];
        out[k] = atlas.add(`art:${id}:${k}`, p.canvas, p.ox, p.oy);
      }
      this.frames[art] = out;
      this.facings[art] = def.facings ?? 1;
      return out;
    } catch (err) {
      this.failed[art] = 1;
      console.error(`art: painting "${id}" failed`, err);
      return NONE;
    }
  }
}

/** Cut a sheet of `TILE`-sized blocks (column = material, row = variant, as the old `makeAtlas`) into tiles. */
export function tilesFromSheet(
  atlas: AtlasBuilder,
  tiles: Int32Array,
  sheet: CanvasImageSource & { width: number; height: number },
  make: (w: number, h: number) => [HTMLCanvasElement | OffscreenCanvas, CanvasDrawImage],
  materials: number,
  remap: (material: number) => number = (m) => m
): void {
  const cols = Math.floor(sheet.width / TILE);
  for (let m = 1; m < materials; m++) {
    const src = remap(m);
    if (src <= 0 || src >= cols) continue;
    for (let v = 0; v < VARIANTS; v++) {
      const [c, ctx] = make(TILE, TILE);
      ctx.drawImage(sheet, src * TILE, v * TILE, TILE, TILE, 0, 0, TILE, TILE);
      tiles[m * VARIANTS + v] = atlas.add(`tile:${m}:${v}`, c);
    }
  }
}
