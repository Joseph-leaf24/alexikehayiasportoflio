/**
 * The squad's guns: the old `fire()` cases that launch shots (`components/pixel/game.ts`), with the old numbers
 * (cadence, bursts, spread, speeds, damage, pierce, life), and each shot kind as the old shots update and scene made
 * it: gravity (`SHOT_GRAV`), blasts (`BLAST_R`), saw blades ricocheting, the flame stream licking fire, rockets
 * speeding up in smoke, and the old tracers and streaks. Weapons with a light `flash` also get the old muzzle cross
 * in the player's colour (`physics/weapons.ts`).
 */
import { disc, INK as INK_CSS } from "@/components/pixel/art";
import { indexOf, INK, PARTICLE_ADDITIVE, PARTICLE_EMISSIVE } from "@/engine";
import { registerShapeArt } from "../art";
import type { Content } from "../game/content";
import type { EffectDef, FireFn, ProjectileDef, ProjectileLook, WeaponDef } from "../game/defs";
import { playerOf } from "../physics/damage";
import { muzzle, PLAYER_RGB } from "../render/beams";
import { EFFECTS as LIBRARY } from "../render/library";

const GLOW = PARTICLE_EMISSIVE | PARTICLE_ADDITIVE;
const UP = -Math.PI / 2;
const WHITE = 0xffffffff;
const rgba = (rgb: number, a: number) => ((rgb << 8) | Math.round(a * 255)) >>> 0;

const EFFECTS: Readonly<Record<string, EffectDef>> = {
  /** A launch puff (rockets, mortar). */
  "squad.puff": {
    count: [3, 4],
    speed: [4, 16],
    angle: UP,
    spread: 1.4,
    life: [1, 2],
    gravity: -10,
    drag: 1,
    size: [2, 3],
    colors: [0x5b5866d0, 0x8a8490a0],
    fade: true,
    jitter: 2,
  },
  /** Smoke behind a mortar shell. */
  "squad.smoke": {
    count: 1,
    speed: [4, 12],
    angle: UP,
    spread: 1.2,
    life: [1, 2],
    gravity: -10,
    size: [1, 2],
    colors: [0x5b5866c0, 0x8a8490a0],
    fade: true,
  },
  /** A rocket's exhaust: fire turning to smoke. */
  "squad.exhaust": {
    count: 1,
    speed: [2, 10],
    life: [0.4, 1],
    gravity: -14,
    size: [1, 2],
    colors: [0xff9a2aff, 0x8a849080],
    ramp: true,
    fade: true,
  },
  /** The flamethrower's stream: fire licking up off each blob (the old FLAME particles). */
  "squad.fire": {
    count: 1,
    speed: [0, 10],
    angle: UP,
    spread: 1.6,
    lift: 20,
    life: [0.2, 0.3],
    size: [1, 2],
    colors: [0xfff3b0ff, 0xffb020ff, 0xff5a1aff],
    ramp: true,
    flags: GLOW,
  },
  /** White sparks: a beam's end, a saw blade's ricochet. */
  "squad.spark": {
    count: 4,
    speed: [20, 80],
    angle: UP,
    spread: 2.4,
    life: [0.15, 0.25],
    gravity: 160,
    colors: [WHITE, 0xfff6c8ff],
    flags: GLOW,
  },
  /** A crossbow bolt's blast, in the old plasma palette. */
  "squad.blast.plasma": {
    ...LIBRARY["blast.fire"],
    colors: [0xbafcffff, 0x4ad8ffff, 0x8a6affff, 0xc84ae8ff],
    light: { color: 0x6ae8ff, radius: 70, intensity: 1.8, life: 0.3 },
  },
};

/** The old tracer: two yellow pixels and an orange tail. */
const TRACER: ProjectileLook = { colors: [0xffe27aff, 0xffe27aff, 0xffb020ff], spacing: 2, flags: PARTICLE_EMISSIVE };

const bullet = (speed: number, life: number, damage: number): ProjectileDef => ({
  speed,
  life,
  damage,
  chip: 1,
  impact: "hit.spark",
  look: TRACER,
});

/** A pulse bolt in its firer's colour, with a white head and a faint tail. */
const pulse = (rgb: number): ProjectileDef => {
  const c = rgba(rgb, 1),
    dim = rgba(rgb, 0.45);
  return {
    speed: 560,
    life: 0.45,
    damage: 1.3,
    pierce: 1,
    chip: 1,
    impact: "hit.spark",
    look: { colors: [WHITE, WHITE, c, c, c, c, dim, dim], spacing: 1, flags: PARTICLE_EMISSIVE },
  };
};

const PROJECTILES: Readonly<Record<string, ProjectileDef>> = {
  "squad.bullet": bullet(400, 0.6, 1),
  /** The minigun's rounds. */
  "squad.round": bullet(380, 0.55, 0.6),
  "squad.pistol": bullet(380, 0.5, 1),
  "squad.pellet": {
    speed: 300,
    life: 0.3,
    damage: 1,
    chip: 1,
    impact: "hit.spark",
    look: { colors: [0xffd27aff], flags: PARTICLE_EMISSIVE },
  },
  /** Punches through two bodies. */
  "squad.plasma": {
    speed: 260,
    life: 0.8,
    damage: 2,
    pierce: 2,
    chip: 2,
    impact: "hit.spark",
    look: { colors: [0xf4e0ffff, 0xc77dffe6, 0xc77dff80], spacing: 1, size: 2, flags: GLOW },
    glow: { color: 0xc77dff, radius: 14, intensity: 0.9 },
  },
  "squad.rocket": {
    speed: 140,
    life: 1.6,
    damage: 0,
    accel: 2.2,
    explode: { r: 18 },
    trail: "squad.exhaust",
    trailRate: 60,
    look: { colors: [0xc8ced9ff, 0xc8ced9ff, 0xc8ced9ff, 0xff9a2aff], spacing: 1 },
    glow: { color: 0xff9a2a, radius: 16, intensity: 1 },
  },
  /** One round through three bodies, a long white streak; it knocks 3 off the block it strikes. */
  "squad.snipe": {
    speed: 800,
    life: 0.5,
    damage: 7,
    pierce: 3,
    chip: 3,
    impact: "hit.spark",
    look: { colors: Array<number>(10).fill(WHITE), spacing: 1, flags: PARTICLE_EMISSIVE },
  },
  /** An unseen blob of fire, drifting up, through two bodies; it shows by the fire it sheds and dies on a wall. */
  "squad.flame": { speed: 140, life: 0.45, damage: 0.45, pierce: 2, gravity: -70, trail: "squad.fire", trailRate: 60 },
  /** The grenade launcher's round: lobbed, it goes off where it lands. */
  "squad.lob": {
    speed: 190,
    life: 2.5,
    damage: 0,
    gravity: 500,
    radius: 1,
    explode: { r: 20 },
    look: { colors: [0x4a5a30ff], size: 2, blink: 0xff3b3bff },
  },
  /** An explosive bolt, dropping a little. */
  "squad.bolt": {
    speed: 330,
    life: 1.2,
    damage: 0,
    gravity: 120,
    explode: { r: 13, fx: "squad.blast.plasma" },
    look: { colors: [0xff3b3bff, 0xc8a060ff, 0xc8a060ff, 0xc8a060ff, 0xc8a060ff], spacing: 1 },
  },
  /** Ricochets off walls and cuts through eight bodies. */
  "squad.saw": {
    speed: 230,
    life: 1.8,
    damage: 1.2,
    pierce: 8,
    bounce: 1,
    radius: 2,
    impact: "squad.spark",
    art: "squad.saw",
  },
  /** A mortar shell, lobbed high onto the enemy's head. */
  "squad.shell": {
    speed: 150,
    life: 4,
    damage: 0,
    gravity: 400,
    explode: { r: 18 },
    trail: "squad.smoke",
    trailRate: 24,
    look: { colors: [INK], size: 3 },
  },
};

/** A pulse bolt in the firer's colour, and a white flash in place of the player's. */
const pulseFire: FireFn = (game, shooter, gx, gy, ax, ay) => {
  const a = Math.atan2(ay - gy, ax - gx) + (game.rng.combat.next() - 0.5) * 0.04;
  const pid = playerOf(game, shooter);
  game.shoot(pid >= 0 ? `squad.pulse.${pid}` : "squad.pulse", gx, gy, Math.cos(a) * 560, Math.sin(a) * 560, {
    owner: shooter,
  });
  muzzle(game, gx + game.c.unit.facing[indexOf(shooter)] * 2, gy, 0xffffff, 0.04);
};

/** The bolt leaves turned 0.05 rad anticlockwise, as the old did. */
const crossbowFire: FireFn = (game, shooter, gx, gy, ax, ay) => {
  const a = Math.atan2(ay - gy, ax - gx) - 0.05;
  game.shoot("squad.bolt", gx, gy, Math.cos(a) * 330, Math.sin(a) * 330, { owner: shooter });
};

const FLASH = { color: 0xffd080, radius: 26 };

const GUNS: Readonly<Record<string, WeaponDef>> = {
  /** Bursts of five. */
  "squad.rifle": {
    name: "Rifle",
    range: 170,
    cadence: 0.11,
    burst: 5,
    burstPause: 0.45,
    spread: 0.08,
    projectile: "squad.bullet",
    flash: FLASH,
    muzzleFx: "casing",
  },
  "squad.shotgun": {
    name: "Shotgun",
    range: 115,
    cadence: 0.6,
    pellets: 6,
    spread: 0.6,
    speedJitter: 60,
    projectile: "squad.pellet",
    flash: FLASH,
    muzzleFx: "casing",
  },
  "squad.plasma": {
    name: "Plasma",
    range: 160,
    cadence: 0.3,
    projectile: "squad.plasma",
    flash: { color: 0xc77dff, radius: 24 },
  },
  "squad.rockets": {
    name: "Rockets",
    range: 190,
    cadence: 1.1,
    projectile: "squad.rocket",
    flash: FLASH,
    muzzleFx: "squad.puff",
  },
  "squad.sniper": {
    name: "Sniper",
    range: 300,
    cadence: 1.5,
    projectile: "squad.snipe",
    flash: FLASH,
    muzzleFx: "casing",
  },
  /** A stream of fire, two blobs every 0.05 s; no muzzle flash. */
  "squad.flamethrower": {
    name: "Flamethrower",
    range: 72,
    cadence: 0.05,
    pellets: 2,
    spread: 0.3,
    speedJitter: 40,
    projectile: "squad.flame",
  },
  /** The grenade launcher: lobbed over cover. */
  "squad.grenades": {
    name: "Grenades",
    range: 210,
    cadence: 0.85,
    lob: 190,
    spread: 0.25,
    projectile: "squad.lob",
    flash: FLASH,
  },
  /** Long hosing bursts of 24. */
  "squad.minigun": {
    name: "Minigun",
    range: 165,
    cadence: 0.045,
    burst: 24,
    burstPause: 0.8,
    spread: 0.14,
    projectile: "squad.round",
    flash: FLASH,
    muzzleFx: "casing",
  },
  /** Left, right, left, right. */
  "squad.pistols": {
    name: "Pistols",
    range: 150,
    cadence: 0.09,
    burst: 2,
    burstPause: 0.2,
    spread: 0.1,
    projectile: "squad.pistol",
    flash: FLASH,
    muzzleFx: "casing",
  },
  "squad.crossbow": {
    name: "Crossbow",
    range: 200,
    cadence: 0.75,
    projectile: "squad.bolt",
    fire: crossbowFire,
    flash: FLASH,
  },
  "squad.saws": {
    name: "Saw blades",
    range: 150,
    cadence: 0.6,
    projectile: "squad.saw",
    flash: FLASH,
  },
  /** Shells lobbed high from a tube held above the gun. */
  "squad.mortar": {
    name: "Mortar",
    range: 260,
    cadence: 1.3,
    lob: 150,
    spread: 0.4,
    muzzleY: -11,
    projectile: "squad.shell",
    flash: FLASH,
    muzzleFx: "squad.puff",
  },
  /** Bursts of four bolts that each pierce one body. */
  "squad.pulse": {
    name: "Pulse laser",
    range: 220,
    cadence: 0.07,
    burst: 4,
    burstPause: 0.34,
    projectile: "squad.pulse",
    fire: pulseFire,
  },
};

export function registerGuns(content: Content): void {
  for (const [id, def] of Object.entries(EFFECTS)) content.effects.register(id, def);
  for (const [id, def] of Object.entries(PROJECTILES)) content.projectiles.register(id, def);
  content.projectiles.register("squad.pulse", pulse(0xff5a6a));
  PLAYER_RGB.forEach((rgb, p) => content.projectiles.register(`squad.pulse.${p}`, pulse(rgb)));
  for (const [id, def] of Object.entries(GUNS)) content.weapons.register(id, def);
  // The saw blade, drawn turned along its flight
  registerShapeArt(content, "squad.saw", {
    w: 7,
    h: 7,
    ox: 3,
    oy: 3,
    paint(c) {
      c.fillStyle = INK_CSS;
      disc(c, 0, 0, 3);
      c.fillStyle = "#c8ced9";
      disc(c, 0, 0, 2);
      c.fillStyle = "#ffffff";
      c.fillRect(2, 0, 1, 1);
      c.fillRect(-2, 0, 1, 1);
    },
  });
}
