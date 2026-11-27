import { makeCanvas, type Canvas2D } from "../render/canvas";
import type { AtlasBuilder, SpriteBuffer, TerrainView } from "../render/types";
import { TILE, VARIANTS } from "../terrain/constants";
import { Terrain } from "../terrain/grid";
import * as Mat from "../terrain/materials";

/** What the demo world needs from its ground: the renderer's view plus a few queries and actions. */
export interface DemoGround {
  readonly view: TerrainView;
  readonly ring: number;
  /** Top of the first solid cell under x, in px. */
  groundY(x: number): number;
  carve(x: number, y: number, r: number, power: number): void;
  ensure(colEnd: number): void;
  /** Advances the ground's own simulation (sand, water, fire, collapse). */
  step(dt: number): void;
  /** Pushes anything the terrain owns but the terrain pass can't draw (rigid falling pieces). */
  sprites(out: SpriteBuffer, layer: number): void;
}

const hash = (a: number, b: number) => {
  let h = Math.imul(a | 0, 374761393) ^ Math.imul(b | 0, 668265263);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
};
const noise = (x: number, seed: number) => {
  const i = Math.floor(x);
  const f = x - i;
  const u = f * f * (3 - 2 * f);
  return hash(i, seed) * (1 - u) + hash(i + 1, seed) * u;
};
const hex = (c: number) => `#${(c | 0x1000000).toString(16).slice(1)}`;
const shade = (c: number, k: number) => {
  const f = (v: number) => Math.max(0, Math.min(255, Math.round(v * k)));
  return (f((c >> 16) & 255) << 16) | (f((c >> 8) & 255) << 8) | f(c & 255);
};

/** A generic 8×8 tile from a material's colour and a hint of its kind (bricks, planks, liquids, glass). */
function paintTile(x: Canvas2D, m: Mat.Material, v: number) {
  const base = m.debris;
  const name = m.name;
  const bricky = /brick|block|stone|tiles|adobe|marble|slate/.test(name);
  const planks = /wood|timber|crate|parquet|plank|bamboo/.test(name);
  for (let py = 0; py < 8; py++)
    for (let px = 0; px < 8; px++) {
      const n = hash((px >> 1) + v * 13 + m.id * 101, (py >> 1) + v * 7);
      let c = shade(base, 0.88 + n * 0.2);
      let a = 1;
      if (bricky && (py % 4 === 3 || (px + (py >> 2) * 4 + v) % 8 === 0)) c = shade(base, 0.6);
      if (planks && py % 4 === 0) c = shade(base, 0.72);
      if (m.emissive > 0) c = n > 0.7 ? shade(base, 1.35) : n > 0.3 ? base : shade(base, 0.75);
      if (m.liquid) {
        c = py === 0 ? shade(base, 1.4) : shade(base, 0.92 + n * 0.12);
        a = m.emissive > 0 ? 1 : py === 0 ? 0.9 : 0.62;
      }
      if (name === "window" || name === "glass") a = px === 0 || py === 0 ? 1 : 0.45;
      x.globalAlpha = a;
      x.fillStyle = hex(c);
      x.fillRect(px, py, 1, 1);
    }
  x.globalAlpha = 1;
}

/**
 * The demo's ground on A2's real terrain: a generator writing hills, caves, crystals, lava, ponds, trees and brick
 * towers (structural, so carving their lower floors brings them down), with tiles painted for every material.
 */
export class A2Ground implements DemoGround {
  readonly view: Terrain;
  readonly ring: number;
  private acc = 0;

  constructor(atlas: AtlasBuilder, seed = 7) {
    const t = new Terrain({ seed, generate: (tt, from, to) => this.generate(tt, from, to, seed) });
    this.view = t;
    this.ring = t.ring;
    for (let m = 1; m < Mat.MATERIAL_COUNT; m++) {
      const mat = Mat.MATERIALS[m];
      if (!mat) continue;
      for (let v = 0; v < VARIANTS; v++) {
        const [c, x] = makeCanvas(TILE, TILE);
        paintTile(x, mat, v);
        t.tiles[m * VARIANTS + v] = atlas.add(`a2tile:${m}:${v}`, c);
      }
    }
  }

  private surface(col: number, seed: number) {
    return Math.round(120 + (noise(col / 40, seed) - 0.5) * 50 + (noise(col / 9, seed + 1) - 0.5) * 10);
  }

  private generate(t: Terrain, from: number, to: number, s: number) {
    for (let col = from; col < to; col++) {
      const top = this.surface(col, s);
      const town = noise(col / 30, s + 5) > 0.62;
      const block = Math.floor(col / 9);
      const tower = town && hash(block, s + 9) > 0.3 ? 10 + Math.floor(hash(block, s + 10) * 24) : 0;
      const pond = !town && noise(col / 22, s + 7) > 0.72;
      const tree = !town && !pond && hash(col, s + 11) > 0.93;
      for (let r = 0; r < t.rows; r++) {
        if (r >= top) {
          const d = r - top;
          const cave = d > 14 && noise(col / 11 + r * 0.37, s + 3) > 0.62 && noise(r / 6 + col * 0.21, s + 4) > 0.45;
          let m = d === 0 ? Mat.GRASS : d < 6 ? Mat.DIRT : d < 9 && hash(col, r) > 0.6 ? Mat.SAND : Mat.STONE;
          if (d > 30 && hash(col >> 2, r >> 2) > 0.985) m = Mat.CRYSTAL;
          if (r > 228) m = Mat.LAVA;
          else if (cave) m = d > 60 ? Mat.LAVA : Mat.AIR;
          t.set(col, r, m, { bg: cave || r > 228 ? Mat.STONE : Mat.AIR, structural: false });
        } else if (pond && r >= top - 3) {
          t.set(col, r, Mat.WATER, { bg: Mat.AIR });
        } else if (tower && r >= top - tower) {
          // A tower 9 columns wide: walls at the sides, a floor every 6 rows, windows, a steel roof
          const lx = col - block * 9;
          const h = top - r;
          let m: number = lx === 0 || lx === 8 ? Mat.BRICK : h % 6 === 0 ? Mat.WOOD : Mat.AIR;
          if (h === tower) m = Mat.STEEL;
          else if ((lx === 0 || lx === 8) && h % 6 === 3) m = Mat.WINDOW;
          if (lx > 0 && lx < 8 && h % 6 === 0 && h !== tower && (lx === 3 || lx === 4)) m = Mat.AIR;
          t.set(col, r, m, { bg: Mat.DARKBRICK, structural: true });
        } else if (tree && r >= top - 8) {
          t.set(col, r, r >= top - 4 ? Mat.TRUNK : Mat.LEAF);
        }
      }
    }
  }

  groundY(x: number) {
    return this.view.surfaceY(x, 0);
  }

  carve(x: number, y: number, r: number, power: number) {
    this.view.carve(x, y, r, power);
  }

  ensure(colEnd: number) {
    this.view.ensure(colEnd);
  }

  step(dt: number) {
    this.acc = Math.min(this.acc + dt, 0.2);
    while (this.acc >= 1 / 30) {
      this.view.step();
      this.acc -= 1 / 30;
    }
  }

  sprites(out: SpriteBuffer, layer: number) {
    const t = this.view;
    for (const f of t.falling) {
      for (let lc = 0; lc < f.w; lc++)
        for (let lr = 0; lr < f.h; lr++) {
          const k = lc * f.h + lr;
          const m = f.fg[k];
          if (!m) continue;
          const frame = t.tiles[m * VARIANTS + (f.variant[k] & (VARIANTS - 1))];
          out.push(frame, (f.col + lc) * TILE, f.y + lr * TILE, layer, 0, 0xffffffff, 1, 0);
        }
    }
  }
}
