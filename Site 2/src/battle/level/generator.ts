/**
 * The level generator: the old engine's own (`components/pixel/level.ts` `createLevel` and `extend`, which runs
 * `feature`), used read-only like the old painters, and copied column by column into `Terrain`. The engine keeps the
 * old block ids, so cells copy straight across. The old generator is stateful and only runs forward, which is how
 * `Terrain.ensure` asks for columns; it works in true columns (`col + origin.cols`), so an origin shift doesn't
 * disturb it.
 *
 * Everything has to stand up to the engine's support check, which the old art never faced:
 * - Buildings stand on their walls. The old ones have open doorways through both ground-floor walls, which in a side
 *   view leaves the floors above on nothing, so each doorway gets a wooden door leaf in its frame. The squad climbs
 *   over or blasts through; blasting a door away takes that side's support, and the building comes down. A building
 *   walled in stone, sandstone or limestone (terrain to the engine) is made one structure first, found from its
 *   doorway.
 * - What the old game drew standing on legs, posts or back-wall pillars (stilt huts, towers, station canopies, tin
 *   roofs, motorway decks) gets beams there: load-bearing cells that bodies and shots pass and blasts break.
 * - Cells that touch only at a corner (a ruin's slab over a wall stub, through a shell hole) are joined.
 */
import {
  cell,
  createLevel,
  extend,
  groundOf,
  type Level,
  type LevelOptions,
  type Spawn,
} from "@/components/pixel/level";
import { F_STRUCTURAL, materials as M, ROWS, TILE, type Terrain, type TerrainGenerator } from "@/engine";
import type { Theme } from "./themes";

/** The old doorways are 3 rows tall. */
const DOOR_ROWS = 3;
/**
 * The least wall over a gap that gets a door leaf: a ruin's 2-row wall stub hangs under a floor slab, and must stand
 * on something when the slab over it has a hole; a one-row slab over a room is left open.
 */
const WALL_ROWS = 2;
/** A door leaf breaks sooner than the wall around it. */
const DOOR_HP = 6;
/** The most cells a stone building's frame has; past it, the "frame" is terrain (a cave roof over a stalactite). */
const MAX_FRAME = 2000;
/**
 * Columns generated past the one being copied: a feature reaches a few columns back (a palm's fronds, a pine's
 * boughs), so a column is only final once the next ones exist. The old game kept 40 columns built past the screen.
 */
const LOOKAHEAD = 16;

/** A run of beams in one true column, under something the old generator drew standing on a support. */
interface Post {
  tc: number;
  top: number;
  /** At most this many rows; the run stops at the first cell that isn't air. */
  rows: number;
  /** The `legs` or `support` prop it stands for, and which of its ends (0 left, 1 right). */
  prop?: PostProp;
  end?: 0 | 1;
}

type PostProp = Extract<Level["props"][number], { kind: "legs" | "support" }>;

/** True column of a `legs` prop's left (0) or right (1) leg, or of a `support` post. */
const postColumn = (p: PostProp, end: 0 | 1): number =>
  Math.floor((p.kind === "legs" && end ? p.x + p.w - 1 : p.x) / TILE);

export class LevelSource {
  readonly old: Level;
  /**
   * What the old generator placed (troops, barrels, crates), in true-column px; dropped once the ring passes it, or
   * taken out by the feature that places it (the squad's barrels).
   */
  readonly spawns: Spawn[] = [];
  /** Beams waiting for their column to be copied. */
  private posts: Post[] = [];
  /** The old `legs` and `support` props already turned into posts. */
  private readonly propped = new WeakSet<object>();
  /** Which ends of those props got beams (bit 0 the left, bit 1 the right). */
  private readonly held = new WeakMap<object, number>();
  /** Old-ring cells (`cell(tc, row)`) of stone frames, made structural when their column is copied. */
  private readonly framed = new Set<number>();
  private current: Theme;

  constructor(
    readonly theme: Theme,
    seed: number,
    opts?: LevelOptions
  ) {
    this.old = createLevel(theme, seed, opts);
    this.current = theme;
  }

  /** Walkable ground row of a true column (a cavern's floor); the base row outside the old ring. */
  ground(tc: number): number {
    return groundOf(this.old, tc);
  }

  /** The biome at the camera: the level's theme, then the one after each biome border passed. */
  get biome(): Theme {
    return this.current;
  }

  /** Pass the biome borders up to true column `tc`; true when the biome changed. */
  enter(tc: number): boolean {
    const borders = this.old.borders;
    const from = this.current;
    for (let b = borders[0]; b && tc >= b.col; b = borders[0]) {
      borders.shift();
      this.current = b.theme;
    }
    return this.current !== from;
  }

  /** Fill stored columns [from, to); `off` is the true column of stored column 0. */
  fill(t: Terrain, from: number, to: number, off: number): void {
    const L = this.old;
    for (let c = from; c < to; c++) {
      // A feature at a time, only when needed: the old ring holds 512 columns, so generating a whole long stretch
      // first would overwrite its start before it was copied
      const until = c + off + LOOKAHEAD + 1;
      if (until > L.next) {
        for (const s of extend(L, until)) {
          this.spawns.push(s);
          if (s.kind === "motorway") this.pillars(s);
        }
        this.supports();
      }
      this.column(t, c, c + off);
    }
    // Drop what has left the old ring (a feature lists its troops in no particular order)
    const left = L.start * TILE;
    let n = 0;
    for (const s of this.spawns) if (s.x >= left) this.spawns[n++] = s;
    this.spawns.length = n;
  }

  private column(t: Terrain, col: number, tc: number): void {
    const L = this.old;
    for (let r = 0; r < ROWS; r++) {
      const i = cell(tc, r);
      const k = L.fg[i];
      const b = L.bg[i];
      // The ring slot was cleared when it was recycled
      if (k === M.AIR && b === M.AIR) continue;
      // Part of a stone building found from its doorway in an earlier column
      t.set(col, r, k, this.framed.delete(i) ? { bg: b, structural: true } : { bg: b });
    }
    this.beams(t, col, tc);
    this.doors(t, col, tc);
    this.joints(t, col);
  }

  /**
   * Whether the post at a `legs` or `support` prop's `end` still stands, in a terrain whose stored column 0 is true
   * column `cols`: its beams are there, or it never had any (legs into water hold nothing up, but the old game drew
   * them all the same).
   */
  stands(t: Terrain, p: PostProp, end: 0 | 1, cols: number): boolean {
    if (!((this.held.get(p) ?? 0) & (1 << end))) return true;
    return t.get(postColumn(p, end) - cols, Math.round(p.y / TILE)) === M.BEAM;
  }

  /** Posts under the old `legs` (one at each end of their span) and `support` props the generator has added. */
  private supports(): void {
    for (const p of this.old.props) {
      if ((p.kind !== "legs" && p.kind !== "support") || this.propped.has(p)) continue;
      this.propped.add(p);
      const top = Math.round(p.y / TILE);
      const rows = Math.round(p.h / TILE);
      const left = postColumn(p, 0);
      const right = postColumn(p, 1);
      this.posts.push({ tc: left, top, rows, prop: p, end: 0 });
      if (right !== left) this.posts.push({ tc: right, top, rows, prop: p, end: 1 });
    }
  }

  /** A motorway deck stands on back-wall concrete pillars: a post in each pillar column, down to the street. */
  private pillars(s: Spawn): void {
    const L = this.old;
    const [x0, x1] = s.extra ?? [0, 0];
    const top = Math.round(s.y / TILE) + 1;
    for (let tc = Math.floor(x0 / TILE); tc < x1 / TILE; tc++) {
      const i = cell(tc, top);
      if (L.fg[i] === M.AIR && L.bg[i] === M.CONCRETE) this.posts.push({ tc, top, rows: ROWS });
    }
  }

  /**
   * Write the beams of this column's posts, through air and foliage, where they land on something that bears load
   * as the engine's support check counts it (legs into water or a pit hold nothing up); drop posts whose column has
   * passed.
   */
  private beams(t: Terrain, col: number, tc: number): void {
    const open = (k: number) => k === M.AIR || M.SOFT[k] === 1;
    let n = 0;
    for (const p of this.posts) {
      if (p.tc > tc) this.posts[n++] = p;
      if (p.tc !== tc) continue;
      let end = p.top;
      while (end < Math.min(ROWS, p.top + p.rows) && open(t.get(col, end))) end++;
      const below = t.get(col, end);
      if (open(below) || M.LIQUID[below]) continue;
      for (let r = p.top; r < end; r++) t.set(col, r, M.BEAM);
      if (p.prop && end > p.top) this.held.set(p.prop, (this.held.get(p.prop) ?? 0) | (1 << (p.end ?? 0)));
    }
    this.posts.length = n;
  }

  /**
   * Close every doorway: a gap of at most `DOOR_ROWS` on solid ground, under a wall at least `WALL_ROWS` tall. A
   * stone wall only counts where it's one cell thick (read from the old level, which has the next column already)
   * over the street, which keeps tunnels, shafts and overhangs open, and only once its building is a structure.
   */
  private doors(t: Terrain, col: number, tc: number): void {
    const L = this.old;
    for (let r = WALL_ROWS; r < ROWS - 1; r++) {
      const above = t.get(col, r - 1);
      if (t.get(col, r) !== M.AIR || !M.SOLID[above] || M.LOOSE[above]) continue;
      const stone = !structuralAt(t, col, r - 1);
      if (stone && (M.SOLID[L.fg[cell(tc - 1, r - 1)]] || M.SOLID[L.fg[cell(tc + 1, r - 1)]])) continue;
      let gap = 1;
      while (gap <= DOOR_ROWS && t.get(col, r + gap) === M.AIR) gap++;
      if (gap > DOOR_ROWS || !M.SOLID[t.get(col, r + gap)]) continue;
      let wall = 1;
      const isWall = (row: number) => (stone ? M.SOLID[t.get(col, row)] === 1 : structuralAt(t, col, row));
      while (wall < WALL_ROWS && isWall(r - 1 - wall)) wall++;
      if (wall < WALL_ROWS) continue;
      // A stone building's doorway opens onto the street, the walkable ground
      if (stone && (r + gap !== this.ground(tc) || !this.frame(t, tc - col, tc, r - 1, r + gap))) continue;
      for (let k = 0; k < gap; k++) t.set(col, r + k, M.WOOD, { structural: true, hp: DOOR_HP });
      r += gap;
    }
  }

  /**
   * Make a stone building one structure: everything solid joined to the wall at (tc, row) above its floor row,
   * flagged structural now in copied columns and as the rest are copied. False, changing nothing, if it's too big
   * to be a building.
   */
  private frame(t: Terrain, off: number, tc: number, row: number, floor: number): boolean {
    const L = this.old;
    const seen = new Set<number>([cell(tc, row)]);
    // Column and row pairs, also the flood's stack
    const found = [tc, row];
    for (let n = 0; n < found.length; n += 2) {
      const c = found[n];
      const r = found[n + 1];
      for (const [nc, nr] of [
        [c - 1, r],
        [c + 1, r],
        [c, r - 1],
        [c, r + 1],
      ]) {
        const i = cell(nc, nr);
        if (nr < 0 || nr >= floor || nc < L.start || nc >= L.next || seen.has(i)) continue;
        const k = L.fg[i];
        if (!M.SOLID[k] || M.LOOSE[k]) continue;
        if (seen.size >= MAX_FRAME) return false;
        seen.add(i);
        found.push(nc, nr);
      }
    }
    for (let n = 0; n < found.length; n += 2) {
      const c = found[n];
      const r = found[n + 1];
      if (c > tc) this.framed.add(cell(c, r));
      else if (!structuralAt(t, c - off, r)) t.set(c - off, r, L.fg[cell(c, r)], { structural: true });
    }
    return true;
  }

  /**
   * Join structural cells that touch only at a corner with the stored column to the left: support runs through
   * edges, so the upper cell is carried over the lower one.
   */
  private joints(t: Terrain, col: number): void {
    for (let r = 1; r < ROWS - 1; r++) {
      if (!structuralAt(t, col, r) || t.get(col - 1, r) !== M.AIR) continue;
      if (structuralAt(t, col - 1, r + 1) && t.get(col, r + 1) === M.AIR) t.set(col - 1, r, t.get(col, r));
      else if (structuralAt(t, col - 1, r - 1) && t.get(col, r - 1) === M.AIR) t.set(col, r - 1, t.get(col - 1, r - 1));
    }
  }
}

/** Whether a stored cell carries the structural flag. */
function structuralAt(t: Terrain, col: number, row: number): boolean {
  return t.get(col, row) !== M.AIR && (t.flags[t.index(col, row)] & F_STRUCTURAL) !== 0;
}

/** The generator callback for `Terrain`, reading the battle's column offset at call time. */
export function levelGenerator(level: LevelSource, colOffset: () => number): TerrainGenerator {
  return (t, from, to) => level.fill(t, from, to, colOffset());
}
