/**
 * The shared effect library: fireballs, smoke, sparks, dust, splashes and casings, in the old engine's palettes.
 * Content refers to them by id ("blast.fire"); a feature adds its own effects the same way, under its own prefix.
 */
import { PARTICLE_ADDITIVE, PARTICLE_EMISSIVE } from "@/engine";
import type { Content } from "../game/content";
import type { EffectDef } from "../game/defs";

const GLOW = PARTICLE_EMISSIVE | PARTICLE_ADDITIVE;
const UP = -Math.PI / 2;

export const EFFECTS: Readonly<Record<string, EffectDef>> = {
  /** The fireball of an ordinary blast: white-hot to red, rising a little. */
  "blast.fire": {
    count: [16, 22],
    speed: [30, 150],
    life: [0.25, 0.6],
    gravity: -40,
    drag: 2.2,
    size: [1, 3],
    colors: [0xfff3b0ff, 0xffd23fff, 0xff8a1aff, 0xc8341aff],
    ramp: true,
    flags: GLOW,
    jitter: 3,
    light: { color: 0xffa050, radius: 70, intensity: 1.8, life: 0.3 },
  },
  "blast.smoke": {
    count: [6, 10],
    speed: [8, 40],
    life: [0.9, 1.6],
    gravity: -26,
    drag: 1.2,
    size: [2, 3],
    colors: [0x5b5866d0, 0x8a8490a0],
    fade: true,
    jitter: 6,
  },
  /** A bullet striking rock. */
  "hit.spark": {
    count: [2, 4],
    speed: [30, 90],
    angle: UP,
    spread: 2.4,
    life: [0.1, 0.22],
    gravity: 300,
    colors: [0xffe27aff, 0xffb020ff],
    ramp: true,
    flags: GLOW,
  },
  /** A bullet striking a body. */
  "hit.flesh": {
    count: [2, 3],
    speed: [20, 60],
    angle: UP,
    spread: 2,
    life: [0.15, 0.3],
    gravity: 400,
    colors: [0xc8302aff, 0x8a1e1eff],
  },
  "dust.kick": {
    count: 1,
    speed: [4, 14],
    angle: UP,
    spread: 1.5,
    life: [0.35, 0.6],
    gravity: -8,
    colors: [0x9a8a78c0],
    fade: true,
  },
  /** Landing in water, or a body falling in. */
  splash: {
    count: [6, 10],
    speed: [40, 110],
    angle: UP,
    spread: 1.3,
    life: [0.3, 0.55],
    gravity: 420,
    colors: [0xe0f6ffff, 0x8fd3ffff],
  },
  casing: {
    count: 1,
    speed: [30, 60],
    angle: UP,
    spread: 0.9,
    life: [0.5, 0.8],
    gravity: 620,
    colors: [0xc8a050ff],
    collide: true,
  },
  "muzzle.smoke": {
    count: 1,
    speed: [4, 12],
    life: [0.3, 0.5],
    gravity: -20,
    size: [1, 2],
    colors: [0x8a8490a0],
    fade: true,
  },
  /** A soldier going down: a puff of dust and a few dark chips. */
  "death.trooper": {
    count: [4, 6],
    speed: [20, 70],
    angle: UP,
    spread: 2.2,
    life: [0.3, 0.7],
    gravity: 380,
    colors: [0x5a4a3aff, 0x8a7a68ff, 0x3a3630ff],
    collide: true,
  },
  /** A falling structure hitting the ground. */
  "impact.dust": {
    count: [14, 20],
    speed: [20, 70],
    angle: UP,
    spread: 2.6,
    life: [0.8, 1.6],
    gravity: -10,
    drag: 1.5,
    size: [2, 3],
    colors: [0x9a9088c0, 0x7e7670a0],
    fade: true,
    jitter: 12,
  },
  /** Steam where lava meets water. */
  steam: {
    count: [3, 5],
    speed: [10, 30],
    angle: UP,
    spread: 1,
    life: [0.6, 1.1],
    gravity: -30,
    colors: [0xe8eef4b0],
    fade: true,
  },
  /** Fire licking up from a burning cell. */
  flame: {
    count: 1,
    speed: [6, 20],
    angle: UP,
    spread: 0.8,
    life: [0.25, 0.45],
    gravity: -40,
    colors: [0xffd23fff, 0xff8a1aff, 0xc8341aff],
    ramp: true,
    flags: GLOW,
  },
};

export function registerEffects(content: Content): void {
  for (const [id, def] of Object.entries(EFFECTS)) content.effects.register(id, def);
}
