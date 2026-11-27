/**
 * The squad's placeholder arms (B2 ports the 17 old weapons): the rifle every player carries in the slice, its
 * tracer rounds, and the hand grenade thrown as each player's special. Numbers are the old engine's.
 */
import { PARTICLE_EMISSIVE } from "@/engine";
import type { Content } from "../game/content";
import type { ProjectileDef, WeaponDef } from "../game/defs";

/** The old rifle: bursts of five, 0.11 s apart, then a 0.45 s pause. */
export const RIFLE: WeaponDef = {
  name: "Rifle",
  range: 170,
  cadence: 0.11,
  burst: 5,
  burstPause: 0.45,
  spread: 0.08,
  projectile: "squad.bullet",
  flash: { color: 0xffd080, radius: 26 },
  muzzleFx: "casing",
};

export const BULLET: ProjectileDef = {
  speed: 400,
  life: 0.6,
  damage: 1,
  chip: 1,
  impact: "hit.spark",
  look: { colors: [0xffe27aff, 0xffe27aff, 0xffb020ff], spacing: 2, flags: PARTICLE_EMISSIVE },
};

/** The hand grenade: bounces, then goes off after its fuse (BLAST_R 22). */
export const GRENADE: ProjectileDef = {
  speed: 170,
  life: 1.1,
  damage: 0,
  gravity: 620,
  bounce: 0.45,
  fuse: true,
  radius: 1,
  explode: { r: 22, ignite: 0.4 },
  look: { colors: [0x3a5a2aff], size: 2, blink: 0xff3b3bff },
};

export function registerArms(content: Content): void {
  content.weapons.register("squad.rifle", RIFLE);
  content.projectiles.register("squad.bullet", BULLET);
  content.projectiles.register("squad.grenade", GRENADE);
}
