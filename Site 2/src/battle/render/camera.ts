/**
 * The camera, simulated in the "camera" phase so on-screen rules (targeting, spawning ahead, respawns) are part of
 * the deterministic simulation. It eases towards a target that a feature provides (the squad sets the old
 * `cameraTarget` framing), never shows columns before the oldest stored one, and carries the shake amplitude.
 */
import { LEVEL_H, TILE, type System } from "@/engine";
import type { BattleGame } from "../game/BattleGame";

/** Where the camera wants its centre to be this tick, or null to hold still. */
export type CameraTarget = (game: BattleGame) => { x: number; y: number } | null;

const damp = (dt: number, rate: number) => 1 - Math.exp(-dt * rate);

export class Camera {
  /** Centre, world px, now and at the start of the tick (for interpolation). */
  x = 0;
  y = 0;
  prevX = 0;
  prevY = 0;
  /** View size in world px at zoom 1, and the zoom (1 = native; a punch is 1.0–1.25). */
  viewW: number;
  viewH: number;
  zoom = 1;
  /** Shake amplitude, px; decays every tick. */
  shake = 0;
  /** Easing rates towards the target, per second (the old scene's 3 and 2). */
  rateX = 3;
  rateY = 2;
  /** Jump straight to the next target instead of easing (a new level, a teleport). */
  snap = true;
  follow: CameraTarget | null = null;

  constructor(viewW: number, viewH: number) {
    this.viewW = viewW;
    this.viewH = viewH;
  }

  resize(w: number, h: number, zoom = 1): void {
    this.viewW = Math.max(1, w);
    this.viewH = Math.max(1, h);
    this.zoom = zoom > 0 ? zoom : 1;
  }

  /** Half the view, in world px at the current zoom. */
  get halfW(): number {
    return this.viewW / (2 * this.zoom);
  }
  get halfH(): number {
    return this.viewH / (2 * this.zoom);
  }
  get left(): number {
    return this.x - this.halfW;
  }
  get right(): number {
    return this.x + this.halfW;
  }
  get top(): number {
    return this.y - this.halfH;
  }
  get bottom(): number {
    return this.y + this.halfH;
  }

  /** Whether a world x is on screen, give or take `margin` px. */
  inViewX(x: number, margin = 0): boolean {
    return x > this.left - margin && x < this.right + margin;
  }

  inView(x: number, y: number, margin = 0): boolean {
    return this.inViewX(x, margin) && y > this.top - margin && y < this.bottom + margin;
  }

  /** Put the centre at (x, y) with no easing or interpolation smear. */
  place(x: number, y: number): void {
    this.x = this.prevX = x;
    this.y = this.prevY = y;
    this.snap = false;
  }

  shiftX(dx: number): void {
    this.x += dx;
    this.prevX += dx;
  }

  update(game: BattleGame, dt: number): void {
    this.prevX = this.x;
    this.prevY = this.y;
    const tg = this.follow?.(game) ?? null;
    if (tg) {
      // Never show columns before the oldest stored one, or below the world
      const minX = game.terrain.startCol * TILE + this.halfW;
      const tx = Math.max(minX, tg.x);
      const ty = Math.min(LEVEL_H - this.halfH, Math.max(this.halfH - 80, tg.y));
      if (this.snap) this.place(tx, ty);
      else {
        this.x += (tx - this.x) * damp(dt, this.rateX);
        this.y += (ty - this.y) * damp(dt, this.rateY);
      }
    }
    this.shake *= Math.exp(-dt * 8);
    if (this.shake < 0.05) this.shake = 0;
  }
}

export function cameraSystem(game: BattleGame): System {
  return { name: "camera.follow", phase: "camera", run: (_w, dt) => game.camera.update(game, dt) };
}
