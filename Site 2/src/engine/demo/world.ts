import {
  PARTICLE_ADDITIVE,
  PARTICLE_EMISSIVE,
  SPRITE_EMISSIVE,
  SPRITE_FLIP_X,
  SPRITE_OUTLINE,
  SPRITE_SILHOUETTE,
  type ParallaxLayer,
  type RenderFrame,
  type Shockwave,
} from "../render/types";
import { mixRgb } from "../render/view";
import type { DemoParallax, DemoSprites } from "./art";
import { TEAM_COLORS } from "./art";
import type { DemoGround } from "./a2Ground";

/** The synthetic battle the demo draws: walking units, drones, glowing orbs, particles, lights, day and night. */

const SOLDIER = 0;
const TANK = 1;
const DRONE = 2;
const ORB = 3;

const SPARK = 0;
const SMOKE = 1;
const EMBER = 2;
const DUST = 3;

const DAY_SKY = [0x3a78d0, 0x86b6e6, 0xd8eaf6];
const DUSK_SKY = [0x2a2a5a, 0xc8607a, 0xffb070];
const NIGHT_SKY = [0x04050e, 0x0c1030, 0x1e2848];
const LIGHT_COLORS = [0xffb060, 0xff7a3a, 0x6ae8ff, 0xc77dff, 0xffe27a, 0x7aff9a];

export interface DemoCounts {
  sprites: number;
  particles: number;
  lights: number;
}

export class DemoWorld {
  time = 0;
  /** Seconds per full day–night cycle. */
  dayLength = 40;
  night = 0;
  camX = 1200;
  camY = 900;
  viewW = 480;
  viewH = 300;
  zoom = 1;
  shake = 0;
  flash = 0;
  chroma = 0;
  white = 0;
  whiteTarget = 0;
  letterbox = 0;
  letterboxTarget = 0;
  bloom = 0.8;
  vignette = 0.35;
  graded = true;
  /** Seconds between automatic blasts; 0 turns them off. */
  autoBlast = 1.1;
  private nextBlast = 0.5;
  private rand = mulberry(12345);

  // Units
  private n: number;
  private ux: Float32Array;
  private uy: Float32Array;
  private uv: Float32Array;
  private uk: Uint8Array;
  private ut: Uint8Array;
  private ua: Float32Array;
  private uh: Float32Array;
  private ufl: Uint8Array;

  // Particles
  private p: number;
  private px: Float32Array;
  private py: Float32Array;
  private pvx: Float32Array;
  private pvy: Float32Array;
  private plife: Float32Array;
  private pmax: Float32Array;
  private pk: Uint8Array;
  private ps: Uint8Array;
  private burst = 0;

  // Lights
  private l: number;
  private lphase: Float32Array;

  private waves: Shockwave[] = [];
  private waveT: number[] = [];
  /** Ground height per ring column, and the world column it was measured for. */
  private ground: Int16Array;
  private groundKey: Int32Array;
  private burstX = 0;
  private burstY = 0;
  private layers: ParallaxLayer[];
  private orbs: number[] = [];

  constructor(
    private terrain: DemoGround,
    private art: DemoSprites,
    bg: DemoParallax,
    counts: DemoCounts
  ) {
    const R = this.rand;
    this.n = counts.sprites;
    this.ux = new Float32Array(this.n);
    this.uy = new Float32Array(this.n);
    this.uv = new Float32Array(this.n);
    this.uk = new Uint8Array(this.n);
    this.ut = new Uint8Array(this.n);
    this.ua = new Float32Array(this.n);
    this.uh = new Float32Array(this.n);
    this.ufl = new Uint8Array(this.n);
    for (let i = 0; i < this.n; i++) {
      const r = R();
      this.uk[i] = r < 0.7 ? SOLDIER : r < 0.78 ? TANK : r < 0.9 ? DRONE : ORB;
      this.ut[i] = (R() * TEAM_COLORS.length) | 0;
      this.ux[i] = this.camX - 300 + R() * 600;
      this.uv[i] = (this.uk[i] === TANK ? 8 : 14 + R() * 16) * (R() < 0.5 ? -1 : 1);
      this.ua[i] = R() * 10;
      this.uh[i] = 40 + R() * 120;
      if (R() < 0.02) this.ufl[i] |= SPRITE_OUTLINE;
      if (this.uk[i] === ORB) this.orbs.push(i);
    }
    this.p = counts.particles;
    this.px = new Float32Array(this.p);
    this.py = new Float32Array(this.p);
    this.pvx = new Float32Array(this.p);
    this.pvy = new Float32Array(this.p);
    this.plife = new Float32Array(this.p);
    this.pmax = new Float32Array(this.p).fill(1);
    this.pk = new Uint8Array(this.p);
    this.ps = new Uint8Array(this.p).fill(1);
    this.l = counts.lights;
    this.lphase = new Float32Array(this.l * 3);
    for (let i = 0; i < this.lphase.length; i++) this.lphase[i] = R() * 100;
    this.ground = new Int16Array(terrain.ring);
    this.groundKey = new Int32Array(terrain.ring).fill(-1);
    // Each layer sits on the horizon when the camera is at the usual ground height (y ≈ 940)
    const layer = (frame: number, factor: number, lift: number): ParallaxLayer => ({
      frame,
      factor,
      y: 940 * factor + lift,
      tint: 0xffffffff,
      repeatX: true,
    });
    this.layers = [
      layer(bg.clouds, 0.03, -120),
      layer(bg.far, 0.1, 10),
      layer(bg.mid, 0.22, 40),
      layer(bg.near, 0.4, 70),
    ];
    for (let i = 0; i < 16; i++) {
      this.waves.push({ x: 0, y: 0, radius: 0, width: 0, strength: 0 });
      this.waveT.push(-1);
    }
  }

  /** Ground height in px at a world x, cached per column until the terrain there changes. */
  groundAt(x: number) {
    const c = Math.floor(x / 8);
    const k = c & (this.terrain.ring - 1);
    if (this.groundKey[k] !== c) {
      this.ground[k] = this.terrain.groundY(x);
      this.groundKey[k] = c;
    }
    return this.ground[k];
  }

  /** A blast: carve, shockwave, flash, chroma punch, sparks and smoke. */
  blast(x: number, y: number, big = 1) {
    const t = this.terrain;
    t.carve(x, y, 22 * big, 12);
    const c0 = Math.floor((x - 30 * big) / 8);
    for (let c = c0; c <= c0 + Math.ceil((60 * big) / 8); c++) this.groundKey[c & (t.ring - 1)] = -1;
    let slot = 0;
    for (let i = 1; i < this.waveT.length; i++) if (this.waveT[i] < 0 || this.waveT[i] > this.waveT[slot]) slot = i;
    const w = this.waves[slot];
    w.x = x;
    w.y = y;
    this.waveT[slot] = 0;
    this.shake = Math.min(6, this.shake + 3 * big);
    this.flash = Math.min(0.5, this.flash + 0.18 * big);
    this.chroma = Math.min(1, this.chroma + 0.6 * big);
    this.burst = 220 * big;
    this.burstX = x;
    this.burstY = y;
  }

  update(dt: number) {
    const R = this.rand;
    this.time += dt;
    const t = this.time;
    // Day and night: a sine through dusk; nights hold a little longer than days
    const u = 0.5 - 0.5 * Math.cos((t / this.dayLength) * Math.PI * 2);
    this.night = Math.min(1, Math.max(0, (u - 0.25) / 0.6));
    this.camX += 30 * dt;
    const target = this.groundAt(this.camX) - 20;
    this.camY += (target - this.camY) * Math.min(1, dt * 1.5);
    const halfW = this.viewW / (2 * this.zoom);
    this.terrain.ensure(Math.ceil((this.camX + halfW) / 8) + 24);
    this.terrain.step(dt);
    this.shake = Math.max(0, this.shake - dt * 14);
    this.flash = Math.max(0, this.flash - dt * 1.6);
    this.chroma = Math.max(0, this.chroma - dt * 2.5);
    this.white += (this.whiteTarget - this.white) * Math.min(1, dt * 3);
    this.letterbox += (this.letterboxTarget - this.letterbox) * Math.min(1, dt * 4);
    if (this.autoBlast > 0 && (this.nextBlast -= dt) <= 0) {
      this.nextBlast = this.autoBlast * (0.6 + R() * 0.8);
      const bx = this.camX + (R() - 0.5) * this.viewW * 0.8;
      this.blast(bx, this.groundAt(bx) + 4, R() < 0.25 ? 1.8 : 1);
    }
    for (let i = 0; i < this.waves.length; i++) {
      const a = this.waveT[i];
      if (a < 0) continue;
      const w = this.waves[i];
      const k = (this.waveT[i] = a + dt) / 0.8;
      if (k >= 1) {
        this.waveT[i] = -1;
        w.strength = 0;
        continue;
      }
      w.radius = 8 + k * 220;
      w.width = 14 + k * 10;
      w.strength = 7 * (1 - k) * (1 - k);
    }

    // Units walk the ground across the view, wrapping round its edges. Where the ground is out of sight they are
    // held at the view's edge, so the whole benchmark load stays on screen.
    const span = 2 * halfW;
    const left = this.camX - halfW;
    const vTop = this.camY - this.viewH / (2 * this.zoom) + 14;
    const vBottom = this.camY + this.viewH / (2 * this.zoom) - 2;
    for (let i = 0; i < this.n; i++) {
      let x = this.ux[i] + this.uv[i] * dt;
      if (x < left) x += span;
      else if (x > left + span) x -= span;
      this.ux[i] = x;
      this.ua[i] += dt * Math.abs(this.uv[i]) * 0.4;
      const k = this.uk[i];
      const y = k === DRONE || k === ORB ? this.groundAt(x) - this.uh[i] + Math.sin(t * 2 + i) * 6 : this.groundAt(x);
      this.uy[i] = y < vTop ? vTop : y > vBottom ? vBottom : y;
    }

    // Particles: every dead one respawns, so the pool stays full; any that leave the view die early
    const g = 160;
    const vl = this.camX - halfW - 2;
    const vt = this.camY - this.viewH / (2 * this.zoom) - 2;
    const vr = vl + 2 * halfW + 4;
    const vb = vt + this.viewH / this.zoom + 4;
    for (let i = 0; i < this.p; i++) {
      const life = this.plife[i] - dt;
      if (life <= 0) {
        this.spawn(i);
        continue;
      }
      this.plife[i] = life;
      const k = this.pk[i];
      if (k === SPARK) this.pvy[i] += g * dt;
      else if (k === SMOKE) {
        this.pvy[i] -= 4 * dt;
        this.pvx[i] += 6 * dt;
      } else if (k === EMBER) this.pvx[i] += Math.sin(t * 3 + i) * 20 * dt;
      const x = (this.px[i] += this.pvx[i] * dt);
      const y = (this.py[i] += this.pvy[i] * dt);
      if (x < vl || x > vr || y < vt || y > vb) this.plife[i] = 0;
    }
  }

  private spawn(i: number) {
    const R = this.rand;
    const halfW = this.viewW / (2 * this.zoom);
    const halfH = this.viewH / (2 * this.zoom);
    if (this.burst > 0) {
      // Blast debris: sparks and smoke out of the crater
      this.burst--;
      const a = R() * Math.PI * 2;
      const s = 40 + R() * 160;
      const smoke = R() < 0.35;
      this.set(
        i,
        this.burstX,
        this.burstY,
        Math.cos(a) * s,
        Math.sin(a) * s - 60,
        smoke ? SMOKE : SPARK,
        smoke ? 1.5 : 0.8
      );
      return;
    }
    const r = R();
    if (r < 0.2 && this.orbs.length) {
      // Sparks off a random orb
      const o = this.orbs[(R() * this.orbs.length) | 0];
      this.set(i, this.ux[o], this.uy[o], (R() - 0.5) * 60, -R() * 50, SPARK, 0.4 + R() * 0.6);
      return;
    }
    // Everything else starts in view, so the pool's particles are on screen
    const top = this.camY - halfH;
    const bottom = this.camY + halfH;
    const inView = (y: number) => (y < top ? top + R() * 20 : y > bottom ? bottom - R() * 20 : y);
    if (r < 0.6) {
      const x = this.camX + (R() - 0.5) * 2 * halfW;
      this.set(i, x, inView(this.groundAt(x)), (R() - 0.5) * 6, -6 - R() * 8, SMOKE, 2 + R() * 2);
      return;
    }
    if (r < 0.82) {
      const x = this.camX + (R() - 0.5) * 2 * halfW;
      this.set(i, x, inView(this.groundAt(x)), (R() - 0.5) * 10, -20 - R() * 30, EMBER, 1.5 + R() * 1.5);
      return;
    }
    this.set(
      i,
      this.camX + (R() - 0.5) * 2 * halfW,
      this.camY - halfH + R() * halfH * 2,
      4 - R() * 2,
      6 + R() * 10,
      DUST,
      2 + R() * 3
    );
  }

  private set(i: number, x: number, y: number, vx: number, vy: number, k: number, life: number) {
    this.px[i] = x;
    this.py[i] = y;
    this.pvx[i] = vx;
    this.pvy[i] = vy;
    this.pk[i] = k;
    this.plife[i] = this.pmax[i] = life;
    this.ps[i] = k === SMOKE ? 2 + ((this.rand() * 2) | 0) : k === SPARK && this.rand() < 0.2 ? 2 : 1;
  }

  /** Writes this frame's camera, sky, effects and buffers. */
  fill(f: RenderFrame) {
    const t = this.time;
    const n = this.night;
    f.time = t;
    f.camera.x = this.camX;
    f.camera.y = this.camY;
    f.camera.zoom = this.zoom;
    f.camera.shakeX = this.shake > 0 ? (this.rand() - 0.5) * this.shake : 0;
    f.camera.shakeY = this.shake > 0 ? (this.rand() - 0.5) * this.shake : 0;
    const dusk = Math.min(1, n * 2);
    const deep = Math.max(0, n * 2 - 1);
    f.sky.top = mixRgb(mixRgb(DAY_SKY[0], DUSK_SKY[0], dusk), NIGHT_SKY[0], deep);
    f.sky.mid = mixRgb(mixRgb(DAY_SKY[1], DUSK_SKY[1], dusk), NIGHT_SKY[1], deep);
    f.sky.bottom = mixRgb(mixRgb(DAY_SKY[2], DUSK_SKY[2], dusk), NIGHT_SKY[2], deep);
    f.sky.stars = deep;
    f.ambient = mixRgb(0xffffff, 0x283058, n);
    const fx = f.effects;
    fx.shockwaves = this.waves;
    fx.flash = this.flash;
    fx.white = this.white;
    fx.letterbox = this.letterbox;
    fx.vignette = this.vignette;
    fx.chroma = this.chroma;
    fx.bloom = this.bloom;
    fx.grade.saturation = this.graded ? 1.1 - n * 0.25 : 1;
    fx.grade.contrast = this.graded ? 1.05 : 1;
    fx.grade.tint = this.graded ? mixRgb(0xfff4e8, 0xd8e4ff, n) : 0xffffff;

    const s = f.sprites;
    const a = this.art;
    for (let i = 0; i < this.n; i++) {
      const k = this.uk[i];
      const team = this.ut[i];
      const anim = this.ua[i];
      let flags = this.ufl[i] | (this.uv[i] < 0 ? SPRITE_FLIP_X : 0);
      let tint = 0xffffffff;
      // Now and then a unit flashes white, as if hit
      if (((i * 7919 + Math.floor(t * 8)) & 63) === 0) {
        flags |= SPRITE_SILHOUETTE;
        tint = 0xffffffff;
      }
      if (flags & SPRITE_OUTLINE) tint = 0xffe27aff;
      if (k === SOLDIER) s.push(a.soldier[team][(anim | 0) & 3], this.ux[i], this.uy[i], 8, flags, tint, 1, 0);
      else if (k === TANK) s.push(a.tank[team][(anim | 0) & 1], this.ux[i], this.uy[i], 6, flags, tint, 1, 0);
      else if (k === DRONE)
        s.push(a.drone[(t * 12 + i) & 1], this.ux[i], this.uy[i], 9, flags, tint, 1, Math.sin(t * 2 + i) * 0.25);
      else s.push(a.orb[(t * 4 + i) & 1], this.ux[i], this.uy[i], 10, SPRITE_EMISSIVE, 0xffffffff, 1, 0);
    }

    // Rigid pieces of collapsing buildings, drawn as cell sprites at the terrain's layer
    this.terrain.sprites(s, 4);

    const p = f.particles;
    for (let i = 0; i < this.p; i++) {
      const k = this.pk[i];
      const life = this.plife[i] / this.pmax[i];
      let color: number;
      let flags = 0;
      if (k === SPARK) {
        color = life > 0.6 ? 0xfff3b0ff : life > 0.3 ? 0xffb020ff : 0xff5a1aff;
        flags = PARTICLE_ADDITIVE | PARTICLE_EMISSIVE;
      } else if (k === SMOKE) {
        const al = Math.max(0.25, Math.round(life * 0.6 * 4) / 4);
        color = (life > 0.6 ? 0x5b586600 : 0x8a849000) | Math.round(al * 255);
      } else if (k === EMBER) {
        color = life > 0.5 ? 0xffd23fff : 0xff7a1aff;
        flags = PARTICLE_EMISSIVE;
      } else color = n > 0.5 ? 0xa0b0e040 : 0xfff4e040;
      p.push(this.px[i], this.py[i], this.ps[i], color >>> 0, flags);
    }

    const L = f.lights;
    const halfW = this.viewW / (2 * this.zoom);
    const halfH = this.viewH / (2 * this.zoom);
    for (let i = 0; i < this.l; i++) {
      const ph = this.lphase;
      const x = this.camX + Math.sin(t * 0.3 + ph[i * 3]) * halfW * 0.95;
      const y = this.camY + Math.sin(t * 0.37 + ph[i * 3 + 1]) * halfH * 0.9;
      const flicker = 0.85 + 0.15 * Math.sin(t * 9 + ph[i * 3 + 2]);
      L.push(x, y, 34 + (i % 5) * 12, LIGHT_COLORS[i % LIGHT_COLORS.length], (0.2 + n * 1.2) * flicker);
    }
    for (let i = 0; i < this.waves.length; i++)
      if (this.waveT[i] >= 0 && this.waveT[i] < 0.3)
        L.push(this.waves[i].x, this.waves[i].y - 6, 90, 0xffb060, (1 - this.waveT[i] / 0.3) * 1.6);

    f.parallax = this.layers;
    this.layers[0].tint = (0xffffff00 | Math.round((1 - n) * 220)) >>> 0;
    for (let i = 1; i < this.layers.length; i++) this.layers[i].tint = n > 0.5 ? 0xb8c0e0ff : 0xffffffff;
  }

  get counts() {
    return { sprites: this.n, particles: this.p, lights: this.l };
  }
}

/** mulberry32: a tiny seeded generator, so runs look the same. */
function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
