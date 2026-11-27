/**
 * Structural collapse. Removing a cell queues its neighbours for a support check, run as a budgeted, resumable
 * flood fill (a few thousand cells per tick, so a skyscraper takes a few ticks, never one long one):
 *
 * - **Ground** (non-structural solid cells): a piece of at most `smallPiece` cells that reaches neither the world's
 *   floor, an unbreakable cell nor the edge of the stored land has been cut off, and crumbles: its cells fall as
 *   loose cells, as the old engine's `cutOff` did. Bigger pieces count as anchored. Fills that prove a cell anchored
 *   leave a proof on it, so the next fill from the same blast stops as soon as it touches one.
 * - **Structures** (cells flagged structural: buildings, bridges, towers): the whole connected structure is
 *   gathered with its load (the cells' weight) and its supports (ground it rests on, each verified by a ground fill).
 *   No support: it falls. Load beyond what the supports carry, or a centre of mass outside them (it would topple):
 *   the supporting cells give way (the storey above them is crushed) and it falls. Falling structures are lifted
 *   out as rigid chunks (`falling.ts`) in bands of rows, bottom first, and pancake into rubble where they land.
 */
import type { Terrain } from "./grid";
import { Falling, type FallingChunk } from "./falling";
import { CELLS, EV_CRUSH, EV_DETACH, F_LOOSE, F_MOVING, F_STRUCTURAL, MASK, ROWS } from "./constants";
import { LIQUID, SOFT, SOLID, STRENGTH, WEIGHT } from "./materials";

export interface StructureConfig {
  /** Work units per tick: flood-fill visits, cells lifted, crushed, shattered, and falling-chunk collision checks. */
  budget: number;
  /** A structure bigger than this is taken as standing (bounds memory and time). */
  maxPiece: number;
  /** Ground pieces up to this many cells crumble when cut off; bigger ones count as anchored. */
  smallPiece: number;
  /** Cells per rigid falling chunk; taller structures fall as a stack of bands. At most `budget`. */
  band: number;
  /** Rigid chunks in flight at once; past it, detached structures crumble as loose cells instead. */
  maxFalling: number;
  /** Supporting contacts recorded per structure; a structure with more is taken as standing. */
  maxContacts: number;
  /** Scales every support's strength: lower makes buildings give way sooner. */
  loadFactor: number;
  /** Falling chunks: gravity (px/s², the old game's 620) and top speed (px/s). */
  gravity: number;
  maxFallSpeed: number;
  /** Rows a landing chunk crushes into debris each time it comes to rest. */
  crushRows: number;
  /** Fractions of crushed cells that vanish as dust and that become rubble; the rest tumbles down as it was. */
  dust: number;
  rubble: number;
  /** Cells above a supporting contact crushed when a structure gives way, up to the floor above. */
  crushHeight: number;
}

export const DEFAULT_STRUCTURE: StructureConfig = {
  budget: 4000,
  maxPiece: 32768,
  smallPiece: 64,
  band: 2048,
  maxFalling: 48,
  maxContacts: 8192,
  loadFactor: 1,
  gravity: 620,
  maxFallSpeed: 720,
  crushRows: 4,
  dust: 0.3,
  rubble: 0.4,
  crushHeight: 12,
};

/* Job states; one job runs at a time, resuming across ticks. */
const IDLE = 0;
const GROUND = 1;
const CRUMBLE = 2;
const BUILD = 3;
const VERIFY = 4;
const VERDICT = 5;
const PREP = 6;
const LIFT = 7;

/** Neighbour offsets, with the share of a contact's strength each direction carries (resting on, beside, hanging). */
const DC = [0, -1, 1, 0];
const DR = [1, 0, 0, -1];
const SHARE = [1, 0.5, 0.5, 0.25];

export class Structure {
  readonly cfg: StructureConfig;
  readonly falling: Falling;
  /** Work and flood-fill visits in the last tick. */
  work = 0;
  fill = 0;

  // Seeds waiting for a check: a ring of cell indices, each at most once, stamped with the fill count when queued
  private readonly queueBuf = new Int32Array(CELLS);
  private qHead = 0;
  private qLen = 0;
  private readonly inQueue = new Uint8Array(CELLS);
  private readonly qseq = new Uint32Array(CELLS);
  /** Per cell: the last fill that visited it, and the last ground fill that proved it anchored. */
  private readonly mark = new Uint32Array(CELLS);
  private readonly proof = new Uint32Array(CELLS);
  /** Fills started so far; proofs older than the last ground cut are stale. */
  private seq = 0;
  private lastCut = 0;

  private job = IDLE;
  private cur = 0;
  private readonly stack: Int32Array;
  private sp = 0;

  // Ground fill
  private readonly ground: Int32Array;
  private gn = 0;
  private anchored = false;
  private subFill = false;

  // Structure fill and its supports
  private readonly piece: Int32Array;
  private pn = 0;
  private overflow = false;
  private mass = 0;
  private sumX = 0;
  private origin = 0;
  private bottomRow = 0;
  private readonly supCell: Int32Array;
  private readonly supFrom: Int32Array;
  private readonly supCol: Int32Array;
  private readonly supDir: Uint8Array;
  private readonly supW: Float32Array;
  private cn = 0;
  private vi = 0;
  private cap = 0;
  private supMin = 0;
  private supMax = 0;

  // Lifting: the piece sorted by row, bottom first, and cut into bands
  private readonly sorted: Int32Array;
  private readonly rowCount = new Int32Array(ROWS);
  private readonly rowFill = new Int32Array(ROWS);
  private readonly bounds = new Int32Array(ROWS + 1);
  private nb = 0;
  private bandAt = 0;
  private pi = 0;
  private valid = 0;
  private prepPhase = 0;
  private firstChunk: FallingChunk | null = null;

  constructor(
    private readonly t: Terrain,
    cfg?: Partial<StructureConfig>
  ) {
    const c = { ...DEFAULT_STRUCTURE, ...cfg };
    c.band = Math.max(1, Math.min(c.band, c.budget));
    this.cfg = c;
    this.falling = new Falling(t, c);
    this.stack = new Int32Array((Math.max(c.maxPiece, 4 * c.smallPiece) + 16) * 2);
    this.ground = new Int32Array(c.smallPiece + 1);
    this.piece = new Int32Array(c.maxPiece);
    this.sorted = new Int32Array(c.maxPiece);
    this.supCell = new Int32Array(c.maxContacts);
    this.supFrom = new Int32Array(c.maxContacts);
    this.supCol = new Int32Array(c.maxContacts);
    this.supDir = new Uint8Array(c.maxContacts);
    this.supW = new Float32Array(c.maxContacts);
  }

  /** Seeds waiting for a check. */
  get queued(): number {
    return this.qLen;
  }

  /** Whether any check, lift or falling chunk is still in progress. */
  get pending(): boolean {
    return this.job !== IDLE || this.qLen > 0 || this.falling.chunks.length > 0;
  }

  /* ---------------------------------------------------------------- queueing */

  /** A cell that held others up was removed (or changed kind): check its neighbours. */
  removed(col: number, row: number, mat: number, flags: number): void {
    // Ground proofs rest on ground cells only; structural and loose cells never carry them
    if (SOLID[mat] && (flags & (F_STRUCTURAL | F_LOOSE)) === 0) this.lastCut = this.seq;
    this.queue(col - 1, row);
    this.queue(col + 1, row);
    this.queue(col, row - 1);
    this.queue(col, row + 1);
  }

  /** Queue one cell for a support check, if it can hold anything up. */
  queue(col: number, row: number): void {
    const t = this.t;
    if (row < 0 || row >= ROWS || col < t.startCol || col >= t.endCol) return;
    const i = (col & MASK) * ROWS + row;
    const m = t.fg[i];
    const fl = t.flags[i];
    if (m === 0 || fl & F_LOOSE || ((fl & F_STRUCTURAL) === 0 && !SOLID[m])) return;
    this.qseq[i] = this.seq;
    if (this.inQueue[i]) return;
    this.inQueue[i] = 1;
    let at = this.qHead + this.qLen;
    if (at >= CELLS) at -= CELLS;
    this.queueBuf[at] = i;
    this.qLen++;
  }

  /** Columns left of `newStart` were recycled: their proofs and chunks are gone. */
  recycled(newStart: number): void {
    this.lastCut = this.seq;
    this.falling.recycled(newStart);
  }

  /* ---------------------------------------------------------------- the tick */

  step(dt: number): void {
    const budget = this.cfg.budget;
    this.fill = 0;
    this.work = this.falling.step(dt, budget);
    while (this.work < budget) {
      const job = this.job;
      if (job === GROUND) {
        if (!this.runGround(budget)) break;
        this.job = this.anchored ? IDLE : CRUMBLE;
      } else if (job === CRUMBLE) {
        if (this.work > 0 && this.work + this.gn > budget) break;
        this.crumbleGround();
        this.job = IDLE;
      } else if (job === BUILD) {
        if (!this.runBuild(budget)) break;
        this.job = this.anchored || this.overflow ? IDLE : this.cn === 0 ? PREP : VERIFY;
        this.startPhase();
      } else if (job === VERIFY) {
        if (!this.runVerify(budget)) break;
        this.job = VERDICT;
      } else if (job === VERDICT) {
        if (this.work > 0 && this.work + this.cfg.crushHeight * 64 > budget) break;
        this.job = this.verdict() ? PREP : IDLE;
        this.startPhase();
      } else if (job === PREP) {
        if (!this.runPrep(budget)) break;
        this.job = LIFT;
      } else if (job === LIFT) {
        if (!this.runLift(budget)) break;
        this.job = IDLE;
      } else if (!this.popSeed()) break;
    }
  }

  /** Reset the per-stage cursors when a job moves on. */
  private startPhase(): void {
    this.vi = 0;
    this.cap = 0;
    this.supMin = Infinity;
    this.supMax = -Infinity;
    this.subFill = false;
    this.pi = 0;
    this.valid = 0;
    this.prepPhase = 0;
    this.bandAt = 0;
    this.nb = 0;
    this.rowCount.fill(0);
  }

  /** Take the next seed and start the check it needs. Returns false when the queue is empty. */
  private popSeed(): boolean {
    if (this.qLen === 0) return false;
    const t = this.t;
    const i = this.queueBuf[this.qHead];
    this.qHead = this.qHead + 1 === CELLS ? 0 : this.qHead + 1;
    this.qLen--;
    this.inQueue[i] = 0;
    this.work++;
    // Answered already by a fill that started after it was queued
    if (this.mark[i] > this.qseq[i]) return true;
    const m = t.fg[i];
    const fl = t.flags[i];
    if (m === 0 || fl & (F_LOOSE | F_MOVING)) return true;
    const rc = (i / ROWS) | 0;
    const col = t.colOfRing(rc);
    if (col >= t.endCol) return true;
    const row = i - rc * ROWS;
    if (fl & F_STRUCTURAL) {
      if (LIQUID[m]) return true;
      this.startBuild(i, col, row);
    } else if (SOLID[m] && this.proof[i] <= this.lastCut) this.startGround(i, col, row, false);
    return true;
  }

  private push(col: number, row: number): void {
    this.stack[this.sp++] = col;
    this.stack[this.sp++] = row;
  }

  /* ---------------------------------------------------------------- ground */

  private startGround(i: number, col: number, row: number, sub: boolean): void {
    this.cur = ++this.seq;
    this.mark[i] = this.cur;
    this.sp = 0;
    this.push(col, row);
    this.gn = 0;
    this.anchored = false;
    this.subFill = sub;
    if (!sub) this.job = GROUND;
  }

  /** Depth-first, preferring down, so ground usually proves itself anchored within a short dive. */
  private runGround(budget: number): boolean {
    const t = this.t;
    const fg = t.fg;
    const flags = t.flags;
    const mark = this.mark;
    const proof = this.proof;
    const stack = this.stack;
    const cur = this.cur;
    const lastCut = this.lastCut;
    const small = this.cfg.smallPiece;
    const start = t.startCol;
    const end = t.endCol;
    search: while (this.sp > 0) {
      if (this.work >= budget) return false;
      this.work++;
      this.fill++;
      const row = stack[--this.sp];
      const col = stack[--this.sp];
      const i = (col & MASK) * ROWS + row;
      const m = fg[i];
      if (m === 0 || !SOLID[m] || flags[i] & (F_STRUCTURAL | F_LOOSE | F_MOVING)) continue;
      if (row >= ROWS - 1 || t.hp[i] === 255 || col <= start || col >= end - 1 || proof[i] > lastCut) {
        this.anchored = true;
        break;
      }
      if (this.gn >= small) {
        this.anchored = true;
        break;
      }
      this.ground[this.gn++] = i;
      for (let n = 3; n >= 0; n--) {
        const nr = row + DR[n];
        if (nr < 0) continue;
        const nc = col + DC[n];
        const ni = (nc & MASK) * ROWS + nr;
        if (mark[ni] === cur) continue;
        const nm = fg[ni];
        if (nm === 0 || !SOLID[nm]) continue;
        const nfl = flags[ni];
        // Resting on a settled loose pile, which stands on something (as runVerify counts it)
        if (n === 0 && (nfl & (F_LOOSE | F_MOVING)) === F_LOOSE) {
          this.anchored = true;
          break search;
        }
        if (nfl & (F_STRUCTURAL | F_LOOSE | F_MOVING)) continue;
        if (proof[ni] > lastCut) {
          this.anchored = true;
          break search;
        }
        mark[ni] = cur;
        this.push(nc, nr);
      }
    }
    if (this.anchored) for (let k = 0; k < this.gn; k++) proof[this.ground[k]] = cur;
    return true;
  }

  /** A cut-off ground piece crumbles: its cells fall straight down and land where they stop. */
  private crumbleGround(): void {
    const t = this.t;
    let bottom = 0;
    let sumC = 0;
    for (let k = 0; k < this.gn; k++) {
      const i = this.ground[k];
      if (t.fg[i] === 0) continue;
      const rc = (i / ROWS) | 0;
      const col = t.colOfRing(rc);
      const row = i - rc * ROWS;
      t.flags[i] |= F_LOOSE | F_MOVING;
      t.touch(col, row);
      sumC += col;
      if (row > bottom) bottom = row;
      // A building resting on it must check its footing again
      for (let n = 0; n < 4; n++) {
        const nc = col + DC[n];
        const nr = row + DR[n];
        if (nr >= 0 && nr < ROWS && t.flags[(nc & MASK) * ROWS + nr] & F_STRUCTURAL) this.queue(nc, nr);
      }
    }
    this.work += this.gn;
    this.lastCut = this.seq;
    if (this.gn) t.emit(EV_DETACH, Math.round(sumC / this.gn), bottom, this.gn);
  }

  /* ---------------------------------------------------------------- structures */

  private startBuild(i: number, col: number, row: number): void {
    this.cur = ++this.seq;
    this.mark[i] = this.cur;
    this.sp = 0;
    this.push(col, row);
    this.pn = 0;
    this.mass = 0;
    this.sumX = 0;
    this.origin = col;
    this.bottomRow = row;
    this.cn = 0;
    this.anchored = false;
    this.overflow = false;
    this.job = BUILD;
  }

  /** Gather the whole structure, its load, centre of mass and supporting contacts. */
  private runBuild(budget: number): boolean {
    const t = this.t;
    const fg = t.fg;
    const flags = t.flags;
    const mark = this.mark;
    const cur = this.cur;
    const start = t.startCol;
    const end = t.endCol;
    const maxPiece = this.cfg.maxPiece;
    const maxContacts = this.cfg.maxContacts;
    search: while (this.sp > 0) {
      if (this.work >= budget) return false;
      this.work++;
      this.fill++;
      const row = this.stack[--this.sp];
      const col = this.stack[--this.sp];
      const i = (col & MASK) * ROWS + row;
      const m = fg[i];
      const fl = flags[i];
      if (m === 0 || LIQUID[m] || (fl & F_STRUCTURAL) === 0 || fl & (F_LOOSE | F_MOVING)) continue;
      if (t.hp[i] === 255) {
        this.anchored = true;
        break;
      }
      if (this.pn >= maxPiece) {
        this.overflow = true;
        break;
      }
      this.piece[this.pn++] = i;
      const w = WEIGHT[m];
      this.mass += w;
      this.sumX += w * (col - this.origin);
      if (row > this.bottomRow) this.bottomRow = row;
      for (let n = 0; n < 4; n++) {
        const nr = row + DR[n];
        if (nr >= ROWS) {
          this.anchored = true;
          break search;
        }
        if (nr < 0) continue;
        const nc = col + DC[n];
        if (nc < start || nc >= end) {
          this.anchored = true;
          break search;
        }
        const ni = (nc & MASK) * ROWS + nr;
        const nm = fg[ni];
        const nfl = flags[ni];
        if (nm !== 0 && !LIQUID[nm] && nfl & F_STRUCTURAL && (nfl & (F_LOOSE | F_MOVING)) === 0) {
          if (mark[ni] !== cur) {
            mark[ni] = cur;
            this.push(nc, nr);
          }
          continue;
        }
        if (nm === 0 || LIQUID[nm] || SOFT[nm] || nfl & F_MOVING) continue;
        if (t.hp[ni] === 255) {
          this.anchored = true;
          break search;
        }
        if (this.cn >= maxContacts) {
          this.overflow = true;
          break search;
        }
        const c = this.cn++;
        this.supCell[c] = ni;
        this.supFrom[c] = i;
        this.supCol[c] = nc - this.origin;
        this.supDir[c] = n;
        this.supW[c] = STRENGTH[m] * SHARE[n];
      }
    }
    return true;
  }

  /** Keep only supports that really stand on something: loose piles do, ground must prove itself anchored. */
  private runVerify(budget: number): boolean {
    const t = this.t;
    while (this.vi < this.cn) {
      if (this.work >= budget) return false;
      if (this.subFill) {
        if (!this.runGround(budget)) return false;
        this.subFill = false;
        if (this.anchored) this.count(this.vi);
        else {
          // The "ground" it stood on is itself a loose fragment: that crumbles, and doesn't count
          this.supW[this.vi] = 0;
          this.crumbleGround();
        }
        this.vi++;
        continue;
      }
      this.work++;
      const s = this.supCell[this.vi];
      const sm = t.fg[s];
      const sfl = t.flags[s];
      if (sm === 0 || LIQUID[sm] || SOFT[sm] || sfl & F_MOVING) {
        this.supW[this.vi++] = 0;
        continue;
      }
      if (sfl & (F_LOOSE | F_STRUCTURAL) || !SOLID[sm] || this.proof[s] > this.lastCut) {
        this.count(this.vi++);
        continue;
      }
      const rc = (s / ROWS) | 0;
      this.startGround(s, t.colOfRing(rc), s - rc * ROWS, true);
    }
    return true;
  }

  private count(v: number): void {
    this.cap += this.supW[v];
    const c = this.supCol[v];
    if (c < this.supMin) this.supMin = c;
    if (c > this.supMax) this.supMax = c;
  }

  /**
   * Decide a gathered structure's fate. Returns true if it falls (its failing supports already crushed).
   * Columns are measured from `origin`, where the fill started.
   */
  private verdict(): boolean {
    if (this.cap <= 0) return true;
    const com = this.mass > 0 ? this.sumX / this.mass : 0;
    const overloaded = this.mass > this.cap * this.cfg.loadFactor;
    const toppling = com < this.supMin - 1 || com > this.supMax + 1;
    if (!overloaded && !toppling) return false;
    this.crush(com);
    return true;
  }

  /** The supports give way: each contact it rests on is crushed up to the floor above it. */
  private crush(com: number): void {
    const t = this.t;
    const fg = t.fg;
    const flags = t.flags;
    const height = this.cfg.crushHeight;
    // The contacts fall with the rest anyway; crushing only opens the gap, so a bounded amount is enough
    const most = height * 64;
    let crushed = 0;
    let lowest = 0;
    for (let v = 0; v < this.cn && crushed < most; v++) {
      if (this.supW[v] <= 0) continue;
      const i = this.supFrom[v];
      const rc = (i / ROWS) | 0;
      const col = t.colOfRing(rc);
      const row = i - rc * ROWS;
      if (row > lowest) lowest = row;
      const run = this.supDir[v] === 0 ? height : 1;
      for (let k = 0; k < run && row - k >= 0; k++) {
        const j = i - k;
        if (fg[j] === 0 || (flags[j] & F_STRUCTURAL) === 0) break;
        // Stop at a junction with a floor: the storey above is what comes down
        if (k > 0 && (this.structuralAt(col - 1, row - k) || this.structuralAt(col + 1, row - k))) break;
        fg[j] = 0;
        t.hp[j] = 0;
        flags[j] = 0;
        t.heat[j] = 0;
        t.touch(col, row - k);
        crushed++;
      }
    }
    this.work += crushed;
    t.emit(EV_CRUSH, Math.round(this.origin + com), lowest, crushed);
  }

  private structuralAt(col: number, row: number): boolean {
    const t = this.t;
    if (col < t.startCol || col >= t.endCol) return false;
    const i = (col & MASK) * ROWS + row;
    return t.fg[i] !== 0 && (t.flags[i] & F_STRUCTURAL) !== 0;
  }

  /* ---------------------------------------------------------------- lifting */

  /** Sort the piece by row, bottom first (counting sort, resumable), and cut it into bands. */
  private runPrep(budget: number): boolean {
    const t = this.t;
    if (this.prepPhase === 0) {
      while (this.pi < this.pn) {
        if (this.work >= budget) return false;
        this.work++;
        const i = this.piece[this.pi++];
        const fl = t.flags[i];
        if (t.fg[i] === 0 || (fl & F_STRUCTURAL) === 0 || fl & (F_LOOSE | F_MOVING)) continue;
        this.piece[this.valid++] = i;
        this.rowCount[i % ROWS]++;
      }
      this.pn = this.valid;
      let at = 0;
      let acc = 0;
      this.nb = 0;
      this.bounds[0] = 0;
      for (let r = ROWS - 1; r >= 0; r--) {
        const c = this.rowCount[r];
        this.rowFill[r] = at;
        if (c === 0) continue;
        if (acc > 0 && acc + c > this.cfg.band) {
          this.bounds[++this.nb] = at;
          acc = 0;
        }
        acc += c;
        at += c;
      }
      if (at > 0) this.bounds[++this.nb] = at;
      this.prepPhase = 1;
      this.pi = 0;
    }
    while (this.pi < this.pn) {
      if (this.work >= budget) return false;
      this.work++;
      const i = this.piece[this.pi++];
      this.sorted[this.rowFill[i % ROWS]++] = i;
    }
    return true;
  }

  /** Lift one band per call while the budget allows; each band falls as its own rigid chunk. */
  private runLift(budget: number): boolean {
    const t = this.t;
    while (this.bandAt < this.nb) {
      const lo = this.bounds[this.bandAt];
      const hi = this.bounds[this.bandAt + 1];
      if (this.work > 0 && this.work + (hi - lo) > budget) return false;
      if (this.falling.chunks.length >= this.cfg.maxFalling) this.work += this.falling.crumble(this.sorted, lo, hi);
      else {
        // Later bands set off as fast as the first is already falling, so the stack stays together
        const lead = this.bandAt === 0 ? null : this.firstChunk;
        const ch = this.falling.lift(this.sorted, lo, hi, lead && !lead.landed ? lead.vy : 0);
        if (this.bandAt === 0) this.firstChunk = ch;
        this.work += this.falling.lastWork;
      }
      this.bandAt++;
    }
    this.firstChunk = null;
    if (this.pn > 0) {
      const com = this.mass > 0 ? this.sumX / this.mass : 0;
      t.emit(EV_DETACH, Math.round(this.origin + com), this.bottomRow, this.pn);
    }
    return true;
  }
}
