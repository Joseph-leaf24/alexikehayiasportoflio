/**
 * Render extraction: the simulation turned into one `RenderFrame`. It works out the interpolated view, then runs
 * the draw passes in order; each pass reads game state and pushes sprites, particles, lights and text. A feature
 * adds a draw by adding a pass (`game.frames.add`), never by editing another pass. Passes never change the
 * simulation and never draw on the gameplay random streams.
 */
import { Rng, type RenderFrame } from "@/engine";
import type { BattleGame } from "../game/BattleGame";

/** The visible world rectangle for this frame, interpolated, in world px. */
export interface View {
  /** Camera centre, including shake. */
  x: number;
  y: number;
  left: number;
  top: number;
  right: number;
  bottom: number;
  /** Internal size in px and the zoom. */
  w: number;
  h: number;
  zoom: number;
  /** Interpolation fraction between the last tick and the next. */
  alpha: number;
  /** Simulation seconds at this frame, interpolated. */
  time: number;
}

export interface RenderPass {
  readonly name: string;
  /** Ascending; the built-in passes use multiples of 100 (see `render/index.ts`). */
  readonly order: number;
  draw(game: BattleGame, frame: RenderFrame, view: View): void;
}

export class FrameBuilder {
  readonly passes: RenderPass[] = [];
  readonly view: View = {
    x: 0,
    y: 0,
    left: 0,
    top: 0,
    right: 0,
    bottom: 0,
    w: 1,
    h: 1,
    zoom: 1,
    alpha: 0,
    time: 0,
  };
  /** Cosmetic randomness for drawing (shake jitter): never the world's streams, so drawing can't change a run. */
  readonly rng = new Rng(0x5eed, 0xd4a3);

  add(pass: RenderPass): void {
    if (this.passes.some((p) => p.name === pass.name)) throw new Error(`FrameBuilder: "${pass.name}" already added`);
    this.passes.push(pass);
    this.passes.sort((a, b) => a.order - b.order);
  }

  remove(name: string): boolean {
    const k = this.passes.findIndex((p) => p.name === name);
    if (k >= 0) this.passes.splice(k, 1);
    return k >= 0;
  }

  build(game: BattleGame, frame: RenderFrame, alpha: number): void {
    const cam = game.camera;
    const v = this.view;
    const a = alpha < 0 ? 0 : alpha > 1 ? 1 : alpha;
    const shake = cam.shake;
    const sx = shake > 0.4 ? Math.round((this.rng.next() - 0.5) * 2 * shake) : 0;
    const sy = shake > 0.4 ? Math.round((this.rng.next() - 0.5) * 2 * shake) : 0;
    v.alpha = a;
    v.zoom = cam.zoom;
    v.w = cam.viewW;
    v.h = cam.viewH;
    v.x = cam.prevX + (cam.x - cam.prevX) * a;
    v.y = cam.prevY + (cam.y - cam.prevY) * a;
    v.left = v.x + sx - cam.halfW;
    v.right = v.x + sx + cam.halfW;
    v.top = v.y + sy - cam.halfH;
    v.bottom = v.y + sy + cam.halfH;
    v.time = game.now + a * game.world.time.dt;
    frame.time = v.time;
    frame.camera.x = v.x;
    frame.camera.y = v.y;
    frame.camera.zoom = cam.zoom;
    frame.camera.shakeX = sx;
    frame.camera.shakeY = sy;
    frame.terrain = game.terrain;
    for (const p of this.passes) p.draw(game, frame, v);
  }
}
