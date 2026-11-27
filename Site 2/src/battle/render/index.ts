/**
 * The render feature: the camera, day–night clock, particles, lights, screen effects, banners and popups (all
 * simulated in ticks), the shared effect library, and the draw passes that turn a tick into a `RenderFrame`.
 *
 * Systems: camera.follow (camera); fx.particles, fx.flashes, fx.screen, fx.fire (effects); sky.clock, fx.text (late).
 * Passes: sky 0, chunks 100, units 200, projectiles 300, particles 400, lights 500, screen 600, text 700.
 */
import type { BattleGame } from "../game/BattleGame";
import { cameraSystem } from "./camera";
import { registerEffects } from "./library";
import {
  chunksPass,
  lightsPass,
  particlesPass,
  projectilesPass,
  screenPass,
  skyPass,
  textPass,
  unitsPass,
} from "./passes";
import { emitEffect } from "./effects";

export { Camera, type CameraTarget } from "./camera";
export { FrameBuilder, type RenderPass, type View } from "./extract";
export { LAYER, TEXT_LAYER } from "./layers";
export { humanoidPose, stillPose, twoFramePose, POSE_IDLE, POSE_JUMP, POSE_RUN } from "./poses";
export { emitDebris, emitDef, emitEffect, type EffectOptions } from "./effects";
export { P_COLLIDE, P_FADE, P_RAMP, Particles } from "./particles";
export type { BackdropLayer } from "./sky";

const burning: number[] = [];

export function install(game: BattleGame): void {
  const { world } = game;
  registerEffects(game.content);
  world.addSystem(cameraSystem(game));
  world.addSystem({ name: "fx.particles", phase: "effects", run: (_w, dt) => game.particles.step(dt, game.terrain) });
  world.addSystem({ name: "fx.flashes", phase: "effects", run: (_w, dt) => game.flashes.step(dt) });
  world.addSystem({ name: "fx.screen", phase: "effects", run: (_w, dt) => game.screen.step(dt) });
  world.addSystem({
    // Flames licking up from burning cells in view, a few a tick
    name: "fx.fire",
    phase: "effects",
    every: 2,
    run() {
      const cam = game.camera;
      const n = game.terrain.collectBurning(Math.floor(cam.left / 8), Math.ceil(cam.right / 8) + 1, burning);
      const r = game.rng.fx;
      for (let k = 0; k < 6 && n > 0; k++) {
        const j = r.int(n);
        emitEffect(game, "flame", burning[j * 2] * 8 + 4, burning[j * 2 + 1] * 8);
      }
    },
  });
  world.addSystem({ name: "sky.clock", phase: "late", run: (_w, dt) => game.sky.step(dt) });
  world.addSystem({
    name: "fx.text",
    phase: "late",
    run(_w, dt) {
      game.banners.step(dt);
      game.popups.step(dt);
    },
  });
  game.banners.onShow = (b) => world.events.emit("banner", { text: b.text, sub: b.sub, color: b.color });

  const f = game.frames;
  f.add(skyPass(game));
  f.add(chunksPass());
  f.add(unitsPass(game));
  f.add(projectilesPass(game));
  f.add(particlesPass());
  f.add(lightsPass());
  f.add(screenPass());
  f.add(textPass());
}
