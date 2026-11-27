import { Skyline } from "./packer";
import type { AtlasBuilder, Frame } from "./types";

export type AtlasSource = HTMLCanvasElement | OffscreenCanvas;

/** A frame that draws nothing, returned for unknown indices so a bad id never throws mid-frame. */
const EMPTY: Frame = Object.freeze({ page: 0, x: 0, y: 0, w: 0, h: 0, ox: 0, oy: 0 });

/**
 * Packs boot-time canvases into square pages. Pages exist only on the GPU (or as the sources themselves for the
 * Canvas 2D fallback): the atlas keeps every source, so a lost WebGL context can re-upload them. Don't repaint a
 * source after adding it; add it again under the same id to replace it (its index stays the same).
 */
export class Atlas implements AtlasBuilder {
  readonly frames: Frame[] = [];
  readonly sources: AtlasSource[] = [];
  private readonly packers: Skyline[] = [];
  private readonly ids = new Map<string, number>();
  /** Frame indices added or replaced since the renderer last uploaded. */
  readonly pending: number[] = [];
  /** Bumped on every change, so renderers can tell cheaply whether anything needs uploading. */
  version = 0;

  constructor(readonly pageSize = 2048) {}

  get pages() {
    return this.packers.length;
  }

  /** Adds a canvas with its pivot (default top-left); returns its frame index. */
  add(id: string, source: AtlasSource, ox = 0, oy = 0): number {
    const w = source.width;
    const h = source.height;
    if (w > this.pageSize || h > this.pageSize) throw new Error(`atlas: ${id} (${w}×${h}) is larger than a page`);
    const known = this.ids.get(id);
    if (known !== undefined) {
      const f = this.frames[known];
      // Same size: repaint in place; otherwise find it a new home under the same index
      if (f.w !== w || f.h !== h) this.frames[known] = this.place(w, h, ox, oy);
      else this.frames[known] = { ...f, ox, oy };
      this.sources[known] = source;
      this.pending.push(known);
      this.version++;
      return known;
    }
    const index = this.frames.length;
    this.frames.push(this.place(w, h, ox, oy));
    this.sources.push(source);
    this.ids.set(id, index);
    this.pending.push(index);
    this.version++;
    return index;
  }

  frame(index: number): Frame {
    return this.frames[index] ?? EMPTY;
  }

  index(id: string): number {
    return this.ids.get(id) ?? -1;
  }

  /** Marks every frame for upload again, after a lost context. */
  invalidate() {
    this.pending.length = 0;
    for (let i = 0; i < this.frames.length; i++) this.pending.push(i);
    this.version++;
  }

  private place(w: number, h: number, ox: number, oy: number): Frame {
    // Zero-sized canvases take no space
    if (w === 0 || h === 0) return { page: 0, x: 0, y: 0, w, h, ox, oy };
    for (let p = 0; p < this.packers.length; p++) {
      const k = this.packers[p];
      if (k.insert(w, h)) return { page: p, x: k.x, y: k.y, w, h, ox, oy };
    }
    const k = new Skyline(this.pageSize, this.pageSize, 1);
    this.packers.push(k);
    k.insert(w, h);
    return { page: this.packers.length - 1, x: k.x, y: k.y, w, h, ox, oy };
  }
}
