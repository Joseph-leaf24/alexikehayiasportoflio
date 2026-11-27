/**
 * Budgeted cellular physics over awake chunks only: powders fall and slide, liquids fall and run toward the nearest
 * drop (no pressure, so they settle and sleep), lava meeting water turns to stone, acid eats what it touches and
 * fire spreads through flammable cells until they burn out. A chunk that passes a whole tick with nothing to do
 * goes to sleep and costs nothing until something wakes it.
 */
import type { Terrain } from "./grid";
import {
  CELLS,
  CHUNK,
  CHUNKS,
  chunkCol0,
  chunkRow0,
  EV_BURNOUT,
  EV_DISSOLVE,
  EV_EXTINGUISH,
  EV_STEAM,
  F_BURNING,
  F_LOOSE,
  F_MOVING,
  F_SCORCHED,
  F_STRUCTURAL,
  F_TUMBLE,
  F_WET,
  MASK,
  ROWS,
  WAKE,
} from "./constants";
import {
  ACID,
  ACIDPROOF,
  BEHAVIOUR,
  BURNS_TO,
  DENSITY,
  FLAMMABLE,
  FLOW,
  HP,
  LAVA,
  LIQUID,
  LOOSE,
  SOFT,
  SOLID,
  STONE,
  STRUCTURAL,
  VISCOSITY,
  WATER,
} from "./materials";

export interface PhysicsConfig {
  /** Cap on cell moves (falls, slides, flows, swaps) per tick. */
  maxMoves: number;
  /** Cap on chunks simulated per tick; the rest wait their turn, round robin. */
  maxActiveChunks: number;
  /** Chance per tick that a burning cell tries to light a neighbour. */
  fireSpread: number;
  /** Chance per tick that acid bites the cell it touches. */
  acidRate: number;
  /** Chance that acid is used up when it destroys a cell. */
  acidUse: number;
  /** Chance per lava turn that it lights a flammable neighbour. */
  lavaIgnite: number;
}

export const DEFAULT_PHYSICS: PhysicsConfig = {
  maxMoves: 3000,
  maxActiveChunks: 64,
  fireSpread: 0.08,
  acidRate: 0.25,
  acidUse: 0.35,
  lavaIgnite: 0.2,
};

export class Physics {
  readonly cfg: PhysicsConfig;
  /** Moves, chunks simulated and burning cells in the last tick. */
  moves = 0;
  chunks = 0;
  burning = 0;
  /** Per cell, the tick stamp of its last move, so nothing moves twice in one tick. */
  private readonly clock = new Uint16Array(CELLS);
  private stamp = 0;
  private cursor = CHUNKS - 1;
  private busy = false;
  /** Target of the last `liquidTarget` call. */
  private tc = 0;
  private tr = 0;

  constructor(
    private readonly t: Terrain,
    cfg?: Partial<PhysicsConfig>
  ) {
    this.cfg = { ...DEFAULT_PHYSICS, ...cfg };
  }

  step(): void {
    this.stamp = this.stamp >= 65535 ? 1 : this.stamp + 1;
    this.moves = 0;
    this.chunks = 0;
    this.burning = 0;
    const awake = this.t.awake;
    const maxChunks = this.cfg.maxActiveChunks;
    const maxMoves = this.cfg.maxMoves;
    // Descending order runs each chunk column bottom-up, so falling cells clear the way for those above.
    let k = this.cursor;
    let resume = -1;
    for (let n = 0; n < CHUNKS; n++, k = k === 0 ? CHUNKS - 1 : k - 1) {
      if (awake[k] === 0) continue;
      if (this.chunks >= maxChunks || this.moves >= maxMoves) {
        resume = k;
        break;
      }
      this.chunks++;
      this.runChunk(k);
    }
    this.cursor = resume >= 0 ? resume : CHUNKS - 1;
  }

  private runChunk(k: number): void {
    const t = this.t;
    const awake = t.awake;
    // Any change in the pass re-arms it through `touch`
    awake[k]--;
    const rc0 = chunkCol0(k);
    const r0 = chunkRow0(k);
    const start = t.startCol;
    const end = t.endCol;
    const fg = t.fg;
    const flags = t.flags;
    const clock = this.clock;
    const stamp = this.stamp;
    const flip = (t.tick & 1) === 1;
    const maxMoves = this.cfg.maxMoves;
    this.busy = false;
    let live = false;
    for (let n = 0; n < CHUNK; n++) {
      const rc = rc0 + (flip ? CHUNK - 1 - n : n);
      const col = start + ((rc - start) & MASK);
      if (col >= end) continue;
      live = true;
      const base = rc * ROWS;
      for (let row = r0 + CHUNK - 1; row >= r0; row--) {
        const i = base + row;
        let m = fg[i];
        if (m === 0) continue;
        let fl = flags[i];
        const b = BEHAVIOUR[m];
        if (b === 0 && (fl & (F_LOOSE | F_BURNING | F_MOVING)) === 0) continue;
        if (fl & F_BURNING) {
          this.fire(i, col, row, m);
          if (fg[i] !== m) continue;
          m = fg[i];
          fl = flags[i];
        }
        if (clock[i] === stamp) continue;
        if (b === 2) this.liquid(i, col, row, m);
        else if (fl & F_LOOSE) this.powder(i, col, row, m);
        else if (fl & F_MOVING) flags[i] = fl & ~F_MOVING;
        if (this.moves >= maxMoves) {
          awake[k] = WAKE;
          return;
        }
      }
    }
    if (!live) awake[k] = 0;
    else if (this.busy) awake[k] = WAKE;
  }

  /* ---------------------------------------------------------------- moving cells */

  private move(i: number, j: number, c0: number, r0: number, c1: number, r1: number): void {
    const t = this.t;
    const fg = t.fg;
    const flags = t.flags;
    fg[j] = fg[i];
    t.hp[j] = t.hp[i];
    flags[j] = flags[i] | F_MOVING;
    t.variant[j] = t.variant[i];
    t.heat[j] = t.heat[i];
    fg[i] = 0;
    t.hp[i] = 0;
    flags[i] = 0;
    t.heat[i] = 0;
    this.clock[j] = this.stamp;
    this.moves++;
    t.touch(c0, r0);
    t.touch(c1, r1);
    // Whatever rested on the cell that left may have lost its footing
    if (r0 > 0) {
      const a = i - 1;
      const ma = fg[a];
      if (ma !== 0 && (flags[a] & F_LOOSE) === 0 && (SOLID[ma] || flags[a] & F_STRUCTURAL))
        t.structure.queue(c0, r0 - 1);
    }
  }

  private swap(i: number, j: number, c0: number, r0: number, c1: number, r1: number): void {
    const t = this.t;
    const a = t.fg[i];
    t.fg[i] = t.fg[j];
    t.fg[j] = a;
    const h = t.hp[i];
    t.hp[i] = t.hp[j];
    t.hp[j] = h;
    const f = t.flags[i];
    t.flags[i] = t.flags[j] | F_MOVING;
    t.flags[j] = f | F_MOVING;
    const v = t.variant[i];
    t.variant[i] = t.variant[j];
    t.variant[j] = v;
    const e = t.heat[i];
    t.heat[i] = t.heat[j];
    t.heat[j] = e;
    this.clock[i] = this.stamp;
    this.clock[j] = this.stamp;
    this.moves++;
    t.touch(c0, r0);
    t.touch(c1, r1);
  }

  /** Sand, gravel, snow, rubble, ash, debris, and any cell crumbling after being cut off. */
  private powder(i: number, col: number, row: number, m: number): void {
    const t = this.t;
    const fg = t.fg;
    if (row + 1 < ROWS) {
      const mb = fg[i + 1];
      if (mb === 0 || SOFT[mb]) return this.move(i, i + 1, col, row, col, row + 1);
      if (LIQUID[mb] && DENSITY[m] > DENSITY[mb]) return this.swap(i, i + 1, col, row, col, row + 1);
      // Powders and tumbling debris slide off slopes; a crumbling piece of ground drops straight
      if (LOOSE[m] || t.flags[i] & F_TUMBLE) {
        const d = ((t.tick ^ col ^ row) & 1) === 0 ? 1 : -1;
        if (this.slide(i, col, row, m, d) || this.slide(i, col, row, m, -d)) return;
      }
    }
    const fl = t.flags[i];
    t.flags[i] = LOOSE[m] ? fl & ~F_MOVING : fl & ~(F_MOVING | F_LOOSE | F_TUMBLE);
  }

  private slide(i: number, col: number, row: number, m: number, d: number): boolean {
    const t = this.t;
    const nc = col + d;
    if (nc < t.startCol || nc >= t.endCol) return false;
    const ni = (nc & MASK) * ROWS + row;
    const side = t.fg[ni];
    if (side !== 0 && !SOFT[side] && !LIQUID[side]) return false;
    const below = t.fg[ni + 1];
    if (below === 0 || SOFT[below]) this.move(i, ni + 1, col, row, nc, row + 1);
    else if (LIQUID[below] && DENSITY[m] > DENSITY[below]) this.swap(i, ni + 1, col, row, nc, row + 1);
    else return false;
    return true;
  }

  /** Water, acid and lava: react with neighbours, then fall or run toward the nearest drop. */
  private liquid(i: number, col: number, row: number, m: number): void {
    if (m === LAVA) {
      if (this.lava(i, col, row)) return;
    } else if (m === ACID && this.acid(i, col, row)) return;
    const j = this.liquidTarget(i, col, row, m);
    if (j < 0) {
      this.t.flags[i] &= ~F_MOVING;
      return;
    }
    // Viscous liquids move only on their turn, but stay awake while they could move
    const visc = VISCOSITY[m];
    if (visc > 1 && (this.t.tick + col) % visc !== 0) {
      this.busy = true;
      return;
    }
    if (this.t.fg[j] === 0) this.move(i, j, col, row, this.tc, this.tr);
    else this.swap(i, j, col, row, this.tc, this.tr);
    this.t.wakeSpan(col, row, FLOW[m]);
  }

  /** Where a liquid cell would move this tick (sets tc, tr), or -1 to stay. */
  private liquidTarget(i: number, col: number, row: number, m: number): number {
    const t = this.t;
    const fg = t.fg;
    if (row + 1 >= ROWS) return -1;
    const mb = fg[i + 1];
    if (mb === 0 || (LIQUID[mb] && mb !== m && DENSITY[m] > DENSITY[mb])) {
      this.tc = col;
      this.tr = row + 1;
      return i + 1;
    }
    const start = t.startCol;
    const end = t.endCol;
    const d = ((t.tick ^ col ^ row) & 1) === 0 ? 1 : -1;
    // Diagonally down, through an open side only (no squeezing between corners)
    for (let s = d, n = 0; n < 2; n++, s = -s) {
      const nc = col + s;
      if (nc < start || nc >= end) continue;
      const ni = (nc & MASK) * ROWS + row;
      if (fg[ni] === 0 && fg[ni + 1] === 0) {
        this.tc = nc;
        this.tr = row + 1;
        return ni + 1;
      }
    }
    // Sideways, one step toward the nearest drop within `flow` cells (a drop one cell away was the diagonal)
    const reach = FLOW[m];
    let best = 0;
    let dir = 0;
    for (let s = d, n = 0; n < 2; n++, s = -s) {
      for (let k = 1; k <= reach && (best === 0 || k < best); k++) {
        const nc = col + s * k;
        if (nc < start || nc >= end) break;
        const ni = (nc & MASK) * ROWS + row;
        if (fg[ni] !== 0) break;
        if (fg[ni + 1] === 0) {
          best = k;
          dir = s;
          break;
        }
      }
    }
    if (best === 0) return -1;
    this.tc = col + dir;
    this.tr = row;
    return (this.tc & MASK) * ROWS + row;
  }

  /** Lava cools to stone against water (the water flashes to steam) and lights flammable neighbours. */
  private lava(i: number, col: number, row: number): boolean {
    const t = this.t;
    const fg = t.fg;
    for (let n = 0; n < 4; n++) {
      const nc = n === 2 ? col - 1 : n === 3 ? col + 1 : col;
      const nr = n === 0 ? row - 1 : n === 1 ? row + 1 : row;
      if (nr < 0 || nr >= ROWS || nc < t.startCol || nc >= t.endCol) continue;
      const ni = (nc & MASK) * ROWS + nr;
      const nm = fg[ni];
      if (nm === WATER) {
        fg[i] = STONE;
        t.hp[i] = HP[STONE];
        t.flags[i] = 0;
        fg[ni] = 0;
        t.hp[ni] = 0;
        t.flags[ni] = 0;
        t.touch(col, row);
        t.touch(nc, nr);
        t.emit(EV_STEAM, nc, nr, 0);
        return true;
      }
      if (FLAMMABLE[nm] && (t.flags[ni] & F_BURNING) === 0) {
        this.busy = true;
        if (t.rng.next() < this.cfg.lavaIgnite) t.igniteCell(ni, nc, nr);
      }
    }
    return false;
  }

  /** Acid bites the first edible neighbour (below first, so it digs); returns true if it bit instead of flowing. */
  private acid(i: number, col: number, row: number): boolean {
    const t = this.t;
    const fg = t.fg;
    for (let n = 0; n < 4; n++) {
      const nc = n === 1 ? col - 1 : n === 2 ? col + 1 : col;
      const nr = n === 0 ? row + 1 : n === 3 ? row - 1 : row;
      if (nr < 0 || nr >= ROWS || nc < t.startCol || nc >= t.endCol) continue;
      const ni = (nc & MASK) * ROWS + nr;
      const nm = fg[ni];
      if (nm === 0 || LIQUID[nm] || ACIDPROOF[nm] || t.hp[ni] === 255) continue;
      this.busy = true;
      if (t.rng.next() >= this.cfg.acidRate) return false;
      if (t.hp[ni] > 1) {
        t.hp[ni]--;
        t.markDirty(nc, nr);
        return true;
      }
      t.emit(EV_DISSOLVE, nc, nr, nm);
      t.hp[ni] = 0;
      fg[ni] = 0;
      const fl = t.flags[ni];
      t.flags[ni] = 0;
      t.heat[ni] = 0;
      t.touch(nc, nr);
      if (SOLID[nm] || fl & F_STRUCTURAL) t.structure.removed(nc, nr, nm, fl);
      if (t.rng.next() < this.cfg.acidUse) {
        fg[i] = 0;
        t.flags[i] = 0;
        t.hp[i] = 0;
        t.touch(col, row);
      }
      return true;
    }
    return false;
  }

  /** A burning cell: doused by water, spreads (mostly upward), and burns out when its fuel runs dry. */
  private fire(i: number, col: number, row: number, m: number): void {
    const t = this.t;
    const flags = t.flags;
    this.busy = true;
    this.burning++;
    if (!FLAMMABLE[m]) {
      flags[i] &= ~F_BURNING;
      return;
    }
    if (this.nextToWater(col, row)) {
      flags[i] = (flags[i] & ~F_BURNING) | F_WET;
      t.heat[i] = 0;
      t.touch(col, row);
      t.emit(EV_EXTINGUISH, col, row, m);
      return;
    }
    if (t.rng.next() < this.cfg.fireSpread) {
      const r = t.rng.u32();
      const dx = (r % 3) - 1;
      const up = (r >>> 8) & 3;
      const dy = up === 0 ? 1 : up === 1 ? 0 : -1;
      const nc = col + dx;
      const nr = row + dy;
      if ((dx | dy) !== 0 && nr >= 0 && nr < ROWS && nc >= t.startCol && nc < t.endCol)
        t.igniteCell((nc & MASK) * ROWS + nr, nc, nr);
    }
    if (t.heat[i] > 1) {
      t.heat[i]--;
      return;
    }
    // Burnt out: wood to ash (which falls), leaves to nothing, dry grass to dirt
    const to = BURNS_TO[m];
    const fl = flags[i];
    t.fg[i] = to;
    t.hp[i] = HP[to];
    flags[i] = to === 0 ? 0 : (LOOSE[to] ? F_LOOSE : 0) | (STRUCTURAL[to] ? F_STRUCTURAL : 0) | (fl & F_SCORCHED);
    t.heat[i] = 0;
    t.touch(col, row);
    t.emit(EV_BURNOUT, col, row, m);
    if (SOLID[m] || fl & F_STRUCTURAL) t.structure.removed(col, row, m, fl);
  }

  private nextToWater(col: number, row: number): boolean {
    const t = this.t;
    const fg = t.fg;
    const base = (col & MASK) * ROWS + row;
    if (row > 0 && fg[base - 1] === WATER) return true;
    if (row + 1 < ROWS && fg[base + 1] === WATER) return true;
    if (col > t.startCol && fg[((col - 1) & MASK) * ROWS + row] === WATER) return true;
    return col + 1 < t.endCol && fg[((col + 1) & MASK) * ROWS + row] === WATER;
  }
}
