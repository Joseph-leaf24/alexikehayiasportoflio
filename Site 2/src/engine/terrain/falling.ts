/**
 * Rigid falling chunks: a detached structure lifted out of the grid as a small cell sprite that falls as one, under
 * gravity. Where it lands, its bottom few rows are crushed into loose debris (budgeted) and the rest keeps falling
 * onto that debris, so a tall building pancakes down into its own rubble pile instead of turning into a tower of
 * loose cells. The game draws chunks from their cells and uses their bounds to crush whatever is underneath.
 */
import type { Terrain } from "./grid";
import type { StructureConfig } from "./structure";
import { EV_IMPACT, F_LOOSE, F_MOVING, F_STRUCTURAL, F_TUMBLE, MASK, ROWS, TILE } from "./constants";
import { HP, LIQUID, RUBBLE, SOFT, SOLID } from "./materials";

export class FallingChunk {
  /** World row of the top edge, in whole cells (where it is for collision). */
  row: number;
  /** Top edge in world px, smooth, and last tick's value for render interpolation. Never below `row * TILE`. */
  y: number;
  prevY: number;
  /** Fall speed, px/s. */
  vy: number;
  /** Cells left (crushing removes them). */
  cells = 0;
  /** It hit something and is crushing its bottom rows into the grid. */
  landed = false;
  /** Local cells, column-major: (lc, lr) at `lc * h + lr`; fg 0 is empty. bg is the back wall carried with it. */
  readonly fg: Uint8Array;
  readonly bg: Uint8Array;
  readonly hp: Uint8Array;
  readonly variant: Uint8Array;
  /** Lowest occupied local row per column, -1 if none: the faces that land. */
  readonly bottom: Int16Array;
  /** While landed: the local rows being crushed, and the next column to crush. */
  crushLo = 0;
  crushHi = -1;
  next = 0;

  constructor(
    /** Absolute column of the left edge. */
    readonly col: number,
    row: number,
    readonly w: number,
    readonly h: number,
    vy: number
  ) {
    this.row = row;
    this.y = this.prevY = row * TILE;
    this.vy = vy;
    const n = w * h;
    this.fg = new Uint8Array(n);
    this.bg = new Uint8Array(n);
    this.hp = new Uint8Array(n);
    this.variant = new Uint8Array(n);
    this.bottom = new Int16Array(w).fill(-1);
  }

  /** Width and height in px. */
  get pxW(): number {
    return this.w * TILE;
  }
  get pxH(): number {
    return this.h * TILE;
  }
  /** Left edge in px. */
  get x(): number {
    return this.col * TILE;
  }
}

export class Falling {
  readonly chunks: FallingChunk[] = [];
  /** Work done by the last `lift`. */
  lastWork = 0;
  private landedCount = 0;

  constructor(
    private readonly t: Terrain,
    private readonly cfg: StructureConfig
  ) {}

  /** Cells in all chunks. */
  cells(): number {
    let n = 0;
    for (const ch of this.chunks) n += ch.cells;
    return n;
  }

  /** Cell indices in `list[lo..hi)` that are still standing structure. */
  private standing(i: number): boolean {
    const fl = this.t.flags[i];
    return this.t.fg[i] !== 0 && (fl & F_STRUCTURAL) !== 0 && (fl & (F_LOOSE | F_MOVING)) === 0;
  }

  /**
   * Lift the cells `list[lo..hi)` out of the grid as one rigid chunk (with the back wall between its top and bottom
   * cells in each column), starting at speed `vy`. Returns null if none of them still stands.
   */
  lift(list: Int32Array, lo: number, hi: number, vy: number): FallingChunk | null {
    const t = this.t;
    this.lastWork = 0;
    let minC = Infinity;
    let maxC = -Infinity;
    let minR = ROWS;
    let maxR = -1;
    for (let k = lo; k < hi; k++) {
      const i = list[k];
      if (!this.standing(i)) {
        list[k] = -1;
        continue;
      }
      const rc = (i / ROWS) | 0;
      const c = t.colOfRing(rc);
      const r = i - rc * ROWS;
      if (c < minC) minC = c;
      if (c > maxC) maxC = c;
      if (r < minR) minR = r;
      if (r > maxR) maxR = r;
    }
    if (maxR < 0) return null;
    const w = maxC - minC + 1;
    const h = maxR - minR + 1;
    const ch = new FallingChunk(minC, minR, w, h, vy);
    const top = new Int16Array(w).fill(h);
    const fg = t.fg;
    for (let k = lo; k < hi; k++) {
      const i = list[k];
      if (i < 0) continue;
      const rc = (i / ROWS) | 0;
      const c = t.colOfRing(rc);
      const r = i - rc * ROWS;
      const lc = c - minC;
      const lr = r - minR;
      const li = lc * h + lr;
      ch.fg[li] = fg[i];
      ch.hp[li] = t.hp[i];
      ch.variant[li] = t.variant[i];
      ch.cells++;
      fg[i] = 0;
      t.hp[i] = 0;
      t.flags[i] = 0;
      t.heat[i] = 0;
      t.touch(c, r);
      if (lr < top[lc]) top[lc] = lr;
      if (lr > ch.bottom[lc]) ch.bottom[lc] = lr;
      // Ground blocks it carried on its roof are left hanging: check them
      if (r > 0) {
        const a = fg[i - 1];
        if (a !== 0 && SOLID[a] && (t.flags[i - 1] & (F_STRUCTURAL | F_LOOSE)) === 0) t.structure.queue(c, r - 1);
      }
    }
    let work = ch.cells;
    // Carry the back wall inside it, so a fallen building doesn't leave its interior hanging in the sky
    for (let lc = 0; lc < w; lc++) {
      if (ch.bottom[lc] < 0) continue;
      const c = minC + lc;
      const base = (c & MASK) * ROWS + minR;
      for (let lr = top[lc]; lr <= ch.bottom[lc]; lr++) {
        const gi = base + lr;
        if (fg[gi] !== 0 || t.bg[gi] === 0) continue;
        ch.bg[lc * h + lr] = t.bg[gi];
        t.bg[gi] = 0;
        t.markDirty(c, minR + lr);
        work++;
      }
    }
    this.lastWork = work;
    this.chunks.push(ch);
    return ch;
  }

  /** Too many chunks in flight: let `list[lo..hi)` crumble where it stands instead. Returns the work. */
  crumble(list: Int32Array, lo: number, hi: number): number {
    const t = this.t;
    let work = 0;
    for (let k = lo; k < hi; k++) {
      const i = list[k];
      if (!this.standing(i)) continue;
      const rc = (i / ROWS) | 0;
      t.flags[i] = (t.flags[i] & ~F_STRUCTURAL) | F_LOOSE | F_MOVING;
      t.touch(t.colOfRing(rc), i - rc * ROWS);
      work++;
    }
    return work;
  }

  /** Move every falling chunk, then crush the bottoms of landed ones while the budget lasts. Returns the work. */
  step(dt: number, budget: number): number {
    let work = 0;
    const { gravity, maxFallSpeed } = this.cfg;
    for (const ch of this.chunks) {
      if (ch.landed) continue;
      ch.prevY = ch.y;
      ch.vy = Math.min(ch.vy + gravity * dt, maxFallSpeed);
      const target = ch.y + ch.vy * dt;
      const need = Math.ceil(target / TILE);
      let hit = false;
      while (ch.row < need) {
        work += ch.w;
        if (this.blocked(ch, ch.row + 1)) {
          hit = true;
          break;
        }
        ch.row++;
      }
      if (!hit) {
        ch.y = target;
        continue;
      }
      // Landed: crush the lowest `crushRows` rows into debris; the rest falls on after them
      const fast = ch.vy;
      ch.y = ch.row * TILE;
      ch.vy = 0;
      ch.landed = true;
      this.landedCount++;
      let low = -1;
      for (let lc = 0; lc < ch.w; lc++) if (ch.bottom[lc] > low) low = ch.bottom[lc];
      ch.crushHi = low;
      ch.crushLo = Math.max(0, low - this.cfg.crushRows + 1);
      ch.next = 0;
      if (fast > 60) this.t.emit(EV_IMPACT, ch.col + (ch.w >> 1), ch.row + low, ch.cells);
    }
    for (let n = 0; n < this.chunks.length && work < budget; ) {
      const ch = this.chunks[n];
      if (ch.landed) work += this.crush(ch, budget - work);
      if (ch.cells === 0) {
        if (ch.landed) this.landedCount--;
        this.chunks.splice(n, 1);
      } else n++;
    }
    return work;
  }

  /** Whether the chunk's landing faces would hit something one row down. */
  private blocked(ch: FallingChunk, newRow: number): boolean {
    const t = this.t;
    for (let lc = 0; lc < ch.w; lc++) {
      const b = ch.bottom[lc];
      if (b < 0) continue;
      const wr = newRow + b;
      if (wr >= ROWS) return true;
      const wc = ch.col + lc;
      if (wc < t.startCol || wc >= t.endCol) return true;
      const i = (wc & MASK) * ROWS + wr;
      const m = t.fg[i];
      if (m !== 0 && !LIQUID[m] && !SOFT[m]) return true;
      if (this.landedCount > 0 && this.restingAt(wc, wr, ch)) return true;
    }
    return false;
  }

  /** Whether a landed chunk still holds a cell at (col, row). */
  private restingAt(col: number, row: number, self: FallingChunk): boolean {
    for (const o of this.chunks) {
      if (o === self || !o.landed) continue;
      const lc = col - o.col;
      const lr = row - o.row;
      if (lc >= 0 && lc < o.w && lr >= 0 && lr < o.h && o.fg[lc * o.h + lr] !== 0) return true;
    }
    return false;
  }

  /**
   * Crush a landed chunk's bottom rows into loose debris, a column at a time, bottom first. When the band is done the
   * chunk falls on. Returns the work.
   */
  private crush(ch: FallingChunk, budget: number): number {
    let work = 0;
    const h = ch.h;
    while (ch.next < ch.w) {
      if (work >= budget) return work;
      const lc = ch.next++;
      for (let lr = ch.crushHi; lr >= ch.crushLo; lr--) {
        const li = lc * h + lr;
        const m = ch.fg[li];
        if (m === 0) continue;
        work++;
        ch.fg[li] = 0;
        ch.cells--;
        this.place(ch.col + lc, ch.row + lr, m, ch.hp[li], ch.variant[li]);
      }
      // Its new landing face
      let b = ch.crushLo - 1;
      while (b >= 0 && ch.fg[lc * h + b] === 0) b--;
      if (ch.bottom[lc] >= ch.crushLo) ch.bottom[lc] = b;
    }
    ch.landed = false;
    this.landedCount--;
    return work;
  }

  /** One shattered cell: dust, rubble, or the original block crumbling, pushing liquid up out of its way. */
  private place(col: number, row: number, m: number, hp: number, variant: number): void {
    const t = this.t;
    const roll = t.rng.next();
    if (roll < this.cfg.dust) return;
    if (row < 0 || row >= ROWS || col < t.startCol || col >= t.endCol) return;
    const mat = roll < this.cfg.dust + this.cfg.rubble ? RUBBLE : m;
    const base = (col & MASK) * ROWS;
    let r = row;
    const here = t.fg[base + r];
    if (here !== 0) {
      const reach = LIQUID[here] ? 24 : 6;
      let up = r - 1;
      while (up >= 0 && r - up <= reach && t.fg[base + up] !== 0) up--;
      if (up < 0 || r - up > reach) {
        if (!LIQUID[here]) return;
      } else if (LIQUID[here]) {
        // The liquid rises to the first open cell above
        t.fg[base + up] = here;
        t.hp[base + up] = t.hp[base + r];
        t.flags[base + up] = F_MOVING;
        t.touch(col, up);
      } else r = up;
    }
    const i = base + r;
    t.fg[i] = mat;
    t.hp[i] = mat === RUBBLE ? HP[RUBBLE] : Math.max(1, Math.min(hp, HP[mat]));
    t.flags[i] = F_LOOSE | F_MOVING | F_TUMBLE;
    t.variant[i] = variant;
    t.heat[i] = 0;
    t.touch(col, r);
  }

  /** Columns left of `newStart` were recycled: chunks over them vanish. */
  recycled(newStart: number): void {
    for (let n = this.chunks.length - 1; n >= 0; n--) {
      const ch = this.chunks[n];
      if (ch.col >= newStart) continue;
      if (ch.landed) this.landedCount--;
      this.chunks.splice(n, 1);
    }
  }
}
