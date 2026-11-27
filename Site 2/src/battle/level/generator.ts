/**
 * The level generator: the old engine's own (`components/pixel/level.ts` `createLevel` and `extend`, which runs
 * `feature`), used read-only like the old painters, and copied column by column into `Terrain`. The engine keeps the
 * old block ids, so cells copy straight across. The old generator is stateful and only runs forward, which is how
 * `Terrain.ensure` asks for columns; it works in true columns (`col + origin.cols`), so an origin shift doesn't
 * disturb it.
 *
 * Buildings stand on their walls. The old ones have open doorways through both ground-floor walls, which in a side
 * view leaves the floors above on nothing, so each doorway gets a wooden door leaf in its masonry frame. The squad
 * climbs over or blasts through; blasting a door away takes that side's support, and the building comes down.
 */
import { cell, createLevel, extend, MASK, type Level, type LevelOptions, type Spawn } from "@/components/pixel/level";
import { materials, ROWS, SOLID, STRUCTURAL, TILE, type Terrain, type TerrainGenerator } from "@/engine";
import type { Theme } from "./themes";

const M = materials;

/** The old doorways are 3 rows tall. */
const DOOR_ROWS = 3;
/** A door leaf breaks sooner than the wall around it. */
const DOOR_HP = 6;

export class LevelSource {
  readonly old: Level;
  /** What the old generator placed (troops, barrels, crates), in true-column px; dropped once the ring passes it. */
  readonly spawns: Spawn[] = [];

  constructor(
    readonly theme: Theme,
    seed: number,
    opts?: LevelOptions
  ) {
    this.old = createLevel(theme, seed, opts);
  }

  /** Walkable ground row of a generated true column (a cavern's floor), as the old `groundOf`. */
  ground(tc: number): number {
    return this.old.ground[tc & MASK];
  }

  /** Fill stored columns [from, to); `off` is the true column of stored column 0. */
  fill(t: Terrain, from: number, to: number, off: number): void {
    const L = this.old;
    // One column more than asked, so a doorway's outside neighbour exists
    for (const s of extend(L, to + off + 1)) this.spawns.push(s);
    while (this.spawns.length && this.spawns[0].x < L.start * TILE) this.spawns.shift();
    for (let c = from; c < to; c++) this.column(t, c, c + off);
  }

  private column(t: Terrain, col: number, tc: number): void {
    const L = this.old;
    for (let r = 0; r < ROWS; r++) {
      const i = cell(tc, r);
      const k = L.fg[i];
      const b = L.bg[i];
      // The ring slot was cleared when it was recycled
      if (k !== M.AIR || b !== M.AIR) t.set(col, r, k, { bg: b });
    }
    this.doors(t, col);
  }

  /**
   * Close every doorway: a gap of at most `DOOR_ROWS` on solid ground, under a structural run at least as tall (a
   * wall, not a one-row slab over a room).
   */
  private doors(t: Terrain, col: number): void {
    for (let r = DOOR_ROWS; r < ROWS - 1; r++) {
      if (t.get(col, r) !== M.AIR || !STRUCTURAL[t.get(col, r - 1)]) continue;
      let gap = 1;
      while (gap <= DOOR_ROWS && t.get(col, r + gap) === M.AIR) gap++;
      if (gap > DOOR_ROWS || !SOLID[t.get(col, r + gap)]) continue;
      let wall = 1;
      while (wall < DOOR_ROWS && STRUCTURAL[t.get(col, r - 1 - wall)]) wall++;
      if (wall < DOOR_ROWS) continue;
      for (let k = 0; k < gap; k++) t.set(col, r + k, M.WOOD, { structural: true, hp: DOOR_HP });
      r += gap;
    }
  }
}

/** The generator callback for `Terrain`, reading the battle's column offset at call time. */
export function levelGenerator(level: LevelSource, colOffset: () => number): TerrainGenerator {
  return (t, from, to) => level.fill(t, from, to, colOffset());
}
