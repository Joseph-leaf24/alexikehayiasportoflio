/**
 * The level's ambient effects, as the end of the old `Game.ambient` made them (`components/pixel/game.ts`): flames over
 * fires, braziers and the sprite props that burn (`asset(id).flames`), smoke from their `smoke`, and spray at a
 * waterfall's foot. The old ran every frame, about 60 times a second; this runs 15 times, so it makes four times as
 * much per run (two flames where the old made one at even odds, and so on).
 */
import { asset } from "@/components/pixel/assets";
import { PARTICLE_ADDITIVE, PARTICLE_EMISSIVE, TILE, type System } from "@/engine";
import type { BattleGame } from "../game/BattleGame";
import type { Content } from "../game/content";
import type { EffectDef } from "../game/defs";
import { emitEffect } from "../render";
import type { LevelSource } from "./generator";

const UP = -Math.PI / 2;

const EFFECTS: Readonly<Record<string, EffectDef>> = {
  /** The old `flame`: a lick of fire rising off a campfire, a brazier or a wreck. */
  "level.flame": {
    count: 1,
    speed: [26, 46],
    angle: UP,
    spread: 0.3,
    life: [0.4, 0.7],
    gravity: -20,
    size: [1, 2],
    colors: [0xfff3b0ff, 0xffb020ff, 0xff5a1aff],
    ramp: true,
    flags: PARTICLE_EMISSIVE | PARTICLE_ADDITIVE,
  },
  /** The old `puff`: dark smoke drifting up. */
  "level.smoke": {
    count: 1,
    speed: [8, 22],
    angle: UP,
    spread: 1,
    life: [1, 2],
    gravity: -4,
    size: [2, 4],
    colors: [0x2b2830c0, 0x5b5866a0],
    ramp: true,
    fade: true,
  },
  /** Spray at a waterfall's foot, blown a little left. */
  "level.mist": {
    count: 1,
    speed: [8, 22],
    angle: UP,
    spread: 1,
    life: [1.2, 2.2],
    size: [2, 4],
    colors: [0xe8f6fa99],
    fade: true,
  },
};

export function registerAmbientEffects(content: Content): void {
  for (const [id, def] of Object.entries(EFFECTS)) content.effects.register(id, def);
}

/** Two flames at (x, y), each up to 4 px either side, as two frames of the old. */
function flames(game: BattleGame, x: number, y: number): void {
  for (let k = 0; k < 2; k++) emitEffect(game, "level.flame", x + (game.rng.fx.next() - 0.5) * 8, y);
}

/** The `level.ambient` system: the props in view (and 20 px either side) burn, smoke and spray. */
export function ambientSystem(game: BattleGame, level: LevelSource): System {
  return {
    name: "level.ambient",
    phase: "effects",
    every: 2,
    run() {
      const r = game.rng.fx;
      const cam = game.camera;
      const off = game.origin.cols * TILE;
      for (const p of level.old.props) {
        const x = p.x - off;
        if (x < cam.left - 20 || x > cam.right + 20) continue;
        if (p.kind === "fire" || p.kind === "brazier") flames(game, x, p.kind === "brazier" ? p.y - 8 : p.y);
        else if (p.kind === "sprite" && !p.back) {
          const a = asset(p.id);
          // Spots are from its top-left, mirrored when it faces left
          const at = (fx: number) => x + (p.flip ? a.w / 2 - fx : fx - a.w / 2);
          if (a.flames) for (const [fx, fy] of a.flames) flames(game, at(fx), p.y - a.h + fy);
          if (a.smoke && r.next() < 0.6)
            for (const [fx, fy] of a.smoke) emitEffect(game, "level.smoke", at(fx), p.y - a.h + fy);
        } else if (p.kind === "waterfall")
          for (let k = 0; k < 3; k++) emitEffect(game, "level.mist", x + r.next() * p.w, p.y + p.h - 2, { vx: -4 });
      }
    },
  };
}
