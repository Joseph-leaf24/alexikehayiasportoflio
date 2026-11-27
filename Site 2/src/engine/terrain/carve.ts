/**
 * Destruction: blasts, single hits, fire and floods. A blast knocks hit points off every breakable cell whose
 * centre is in range, falling off with distance and with each material's blast resistance, with a seeded dither
 * so crater edges come out ragged rather than perfect circles. Broken masonry may leave loose rubble behind.
 */
import type { Terrain } from "./grid";
import { F_REINFORCED, F_SCORCHED, MASK, ROWS, TILE } from "./constants";
import { BLAST_TAKEN, DEBRIS, FLAMMABLE, HP, SOLID } from "./materials";

/** What a blast did, for particles and sound. Reused between blasts; read it before the next one. */
export interface CarveResult {
  /** Cells destroyed, and cells only damaged. */
  destroyed: number;
  damaged: number;
  /** Up to 8 debris colours (0xRRGGBB) of the destroyed materials, with how many cells each. */
  kinds: number;
  colors: Uint32Array;
  counts: Uint16Array;
  /** Up to 32 destroyed cells' centres (world px) and materials, to spawn particles at. */
  samples: number;
  sx: Float32Array;
  sy: Float32Array;
  smat: Uint8Array;
}

const KINDS = 8;
const SAMPLES = 32;

export function createCarveResult(): CarveResult {
  return {
    destroyed: 0,
    damaged: 0,
    kinds: 0,
    colors: new Uint32Array(KINDS),
    counts: new Uint16Array(KINDS),
    samples: 0,
    sx: new Float32Array(SAMPLES),
    sy: new Float32Array(SAMPLES),
    smat: new Uint8Array(SAMPLES),
  };
}

function record(out: CarveResult, col: number, row: number, m: number): void {
  out.destroyed++;
  const color = DEBRIS[m];
  let k = 0;
  while (k < out.kinds && out.colors[k] !== color) k++;
  if (k < KINDS) {
    if (k === out.kinds) {
      out.colors[k] = color;
      out.counts[k] = 0;
      out.kinds++;
    }
    out.counts[k]++;
  }
  // Keep a spread of samples: the first 32, then every other one replaces an older slot
  const s = out.samples < SAMPLES ? out.samples++ : out.destroyed % SAMPLES;
  out.sx[s] = col * TILE + TILE / 2;
  out.sy[s] = row * TILE + TILE / 2;
  out.smat[s] = m;
}

/**
 * Blast at world px (x, y) with radius `r` px. `power` is the hits dealt at the centre (the old engine used
 * radius / 7); it falls off linearly to the edge, is divided by the material's blast resistance and halved for
 * reinforced cells. `power` 255 or more clears every breakable cell in range outright. Unbreakable cells (hp 255:
 * bedrock, water, lava) are never touched. Cells just outside the radius are scorched.
 */
export function carve(t: Terrain, x: number, y: number, r: number, power: number, out: CarveResult): CarveResult {
  out.destroyed = 0;
  out.damaged = 0;
  out.kinds = 0;
  out.samples = 0;
  if (!(r > 0)) return out;
  const scorch = r + TILE;
  const c0 = Math.max(t.startCol, Math.floor((x - scorch) / TILE));
  const c1 = Math.min(t.endCol - 1, Math.floor((x + scorch) / TILE));
  const r0 = Math.max(0, Math.floor((y - scorch) / TILE));
  const r1 = Math.min(ROWS - 1, Math.floor((y + scorch) / TILE));
  if (c0 > c1 || r0 > r1) return out;
  const clear = power >= 255;
  const r2 = r * r;
  const s2 = scorch * scorch;
  const reach = r + TILE / 2;
  const fg = t.fg;
  const hp = t.hp;
  const flags = t.flags;
  for (let c = c0; c <= c1; c++) {
    const base = (c & MASK) * ROWS;
    const dx = c * TILE + TILE / 2 - x;
    for (let row = r0; row <= r1; row++) {
      const dy = row * TILE + TILE / 2 - y;
      const d2 = dx * dx + dy * dy;
      if (d2 > s2) continue;
      const i = base + row;
      const m = fg[i];
      if (m === 0 || hp[i] === 255 || HP[m] === 255) continue;
      if (d2 > r2) {
        if (SOLID[m] && (flags[i] & F_SCORCHED) === 0) {
          flags[i] |= F_SCORCHED;
          t.markDirty(c, row);
        }
        continue;
      }
      let hits: number;
      if (clear) hits = 255;
      else {
        let dmg = power * (1 - Math.sqrt(d2) / reach) * BLAST_TAKEN[m];
        if (flags[i] & F_REINFORCED) dmg *= 0.5;
        hits = Math.floor(dmg + t.rng.next());
      }
      if (hits <= 0) continue;
      if (hits >= hp[i]) {
        t.breakCell(i, c, row);
        record(out, c, row, m);
      } else {
        hp[i] -= hits;
        flags[i] |= F_SCORCHED;
        out.damaged++;
        t.markDirty(c, row);
      }
    }
  }
  // Loose ground and liquids around the crater may now move
  t.wakeRect(c0 - 1, r0 - 1, c1 + 1, r1 + 1);
  return out;
}

/** Knock `hits` off one cell (a bullet, a pickaxe). Returns the material that broke, or 0 if it held or was empty. */
export function damageCell(t: Terrain, col: number, row: number, hits: number): number {
  if (row < 0 || row >= ROWS || col < t.startCol || col >= t.endCol || hits <= 0) return 0;
  const i = (col & MASK) * ROWS + row;
  const m = t.fg[i];
  if (m === 0 || t.hp[i] === 255 || HP[m] === 255) return 0;
  if (t.hp[i] > hits) {
    t.hp[i] -= hits;
    t.markDirty(col, row);
    return 0;
  }
  t.breakCell(i, col, row);
  return m;
}

/** Set flammable cells within `r` px of (x, y) alight. Returns how many caught. */
export function ignite(t: Terrain, x: number, y: number, r: number): number {
  let n = 0;
  const c0 = Math.max(t.startCol, Math.floor((x - r) / TILE));
  const c1 = Math.min(t.endCol - 1, Math.floor((x + r) / TILE));
  const r0 = Math.max(0, Math.floor((y - r) / TILE));
  const r1 = Math.min(ROWS - 1, Math.floor((y + r) / TILE));
  for (let c = c0; c <= c1; c++)
    for (let row = r0; row <= r1; row++) {
      const dx = c * TILE + TILE / 2 - x;
      const dy = row * TILE + TILE / 2 - y;
      if (dx * dx + dy * dy > r * r) continue;
      const i = (c & MASK) * ROWS + row;
      if (FLAMMABLE[t.fg[i]] && t.igniteCell(i, c, row)) n++;
    }
  return n;
}

/** Fill air within `r` px of (x, y) with `mat` (a flood, a lava burst, an acid splash). Returns cells filled. */
export function flood(t: Terrain, x: number, y: number, r: number, mat: number): number {
  let n = 0;
  const c0 = Math.max(t.startCol, Math.floor((x - r) / TILE));
  const c1 = Math.min(t.endCol - 1, Math.floor((x + r) / TILE));
  const r0 = Math.max(0, Math.floor((y - r) / TILE));
  const r1 = Math.min(ROWS - 1, Math.floor((y + r) / TILE));
  for (let c = c0; c <= c1; c++)
    for (let row = r0; row <= r1; row++) {
      const dx = c * TILE + TILE / 2 - x;
      const dy = row * TILE + TILE / 2 - y;
      if (dx * dx + dy * dy > r * r) continue;
      const i = (c & MASK) * ROWS + row;
      if (t.fg[i] !== 0) continue;
      t.set(c, row, mat);
      n++;
    }
  return n;
}
