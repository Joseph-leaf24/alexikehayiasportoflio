/**
 * The floating origin for the endless level. Positions are f32, so once the camera is two terrain rings
 * (8,192 px) out, everything slides back by one ring (4,096 px): entity positions, the terrain's column window (a
 * whole ring, so every cell keeps its ring slot), the camera, particles, lights and every tracked world-x column.
 * It waits for the terrain's structure work to be idle, because a support check in progress holds absolute columns.
 */
import { RING, shiftOrigin, TILE, type ComponentBase } from "@/engine";
import type { BattleGame } from "./BattleGame";

/** One shift, in world px: a whole terrain ring. */
export const ORIGIN_STEP = RING * TILE;

export class Origin {
  /** World px shifted so far: a position's distance from the battle's true origin is `x + origin.x`. */
  x = 0;
  /** Terrain columns shifted so far: the true column of `col` is `col + origin.cols`. */
  cols = 0;
  /** The camera centre past which a shift happens. */
  threshold = 2 * ORIGIN_STEP;
  /** Shifts done. */
  shifts = 0;
  private readonly fields: { readonly comp: ComponentBase; readonly k: number }[] = [];

  constructor(private readonly game: BattleGame) {
    const { motor, ai, intent } = game.c;
    this.track(motor, "lastX");
    this.track(ai, "home");
    this.track(intent, "aimX");
  }

  /** A column holding world x that must move with the origin (besides pos and prev, which always do). */
  track(comp: ComponentBase, field: string): void {
    const k = comp.$.fields.indexOf(field);
    if (k < 0) throw new Error(`Origin: ${comp.$.name} has no field "${field}"`);
    this.fields.push({ comp, k });
  }

  /** The true (unshifted) world x of a position. */
  trueX(x: number): number {
    return x + this.x;
  }

  update(): void {
    const g = this.game;
    if (g.camera.x < this.threshold || g.terrain.structure.pending) return;
    this.shift(-ORIGIN_STEP * Math.floor((g.camera.x - this.threshold) / ORIGIN_STEP + 1));
  }

  /** Slide the world by `dx` px (a multiple of `ORIGIN_STEP`). Call between ticks or from a late system. */
  shift(dx: number): void {
    if (dx === 0) return;
    if (dx % ORIGIN_STEP !== 0) throw new Error(`Origin: shift ${dx} is not a multiple of ${ORIGIN_STEP}`);
    const g = this.game;
    const dc = dx / TILE;
    shiftOrigin(g.world, dx, 0);
    g.terrain.startCol += dc;
    g.terrain.endCol += dc;
    g.camera.shiftX(dx);
    g.particles.shiftX(dx);
    g.flashes.shiftX(dx);
    g.screen.shiftX(dx);
    g.popups.shiftX(dx);
    for (const { comp, k } of this.fields) {
      // Read the column now: storage is replaced when the world grows
      const { dense, size, columns } = comp.$;
      const col = columns[k];
      for (let j = 0; j < size; j++) col[dense[j]] += dx;
    }
    this.x -= dx;
    this.cols -= dc;
    this.shifts++;
    g.world.events.dispatch("originShift", { dx, cols: dc });
  }
}
