/**
 * The day–night clock and what it colours: the sky gradient, stars, the ambient light and the backdrop's tint.
 * The clock runs in ticks (the old game's `clock`, one cycle per `dayLength` seconds). A biome's `SkyLook` says how
 * dark it is and its sky colours, fixed or from the clock (the old themes' `night`, `sky`, `cycle`), faded across a
 * biome border; `night` is that darkness, 0 for day, 0.7 at the cycle's midnight. The backdrop is a list of parallax
 * layers a level feature sets per theme.
 */
import { mixRgb } from "@/engine";

/** One backdrop layer: an art id, how far it moves with the camera, and how far below the horizon its foot sits. */
export interface BackdropLayer {
  readonly art: string;
  readonly factor: number;
  readonly lift: number;
  /** Hangs from the top of the view instead of standing on the horizon (a cave's far rock; art pivoted at its top). */
  readonly top?: boolean;
}

/** What a biome gives the sky: fixed light and colours, or the day–night cycle's. */
export interface SkyLook {
  /** Runs the day–night cycle instead of the fixed `night` and `colors`. */
  readonly cycle: boolean;
  /** Darkness, 0 for day to 1 for black. */
  readonly night: number;
  /** The sky gradient's top, middle and bottom (RGB). */
  readonly colors: readonly number[];
  /** Stars come out at night (not under a cave's roof). */
  readonly stars: boolean;
  /** A colour cast over the world (RGB) and its strength, 0 for none. */
  readonly tint: number;
  readonly tintA: number;
}

/** Seconds a new backdrop takes to fade in over the last one (the old scene's biome change). */
const BACKDROP_FADE = 1.2;
/** The cycle's darkness at midnight. */
const CYCLE_NIGHT = 0.7;

/** Sky colours by time of day, from the old scene: top, middle and bottom of the gradient. */
const DAY = [0x3a78d0, 0x86b6e6, 0xd8eaf6];
const DUSK = [0x2a2a5a, 0xc8607a, 0xffb070];
const NIGHT = [0x04050e, 0x0c1030, 0x1e2848];
/** Ambient light at noon and at midnight: deep enough that muzzle flashes and fires carry the night. */
const AMBIENT_DAY = 0xffffff;
const AMBIENT_NIGHT = 0x464e78;

/** The look with no level feature: the plain day–night cycle. */
export const CYCLE_LOOK: SkyLook = { cycle: true, night: 0, colors: DAY, stars: true, tint: 0xffffff, tintA: 0 };

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

export class SkyClock {
  /** 0..1 through the cycle; 0 is noon. */
  clock: number;
  /** Seconds per cycle. */
  dayLength: number;
  /** The backdrop, far to near (set by the level feature with `show`; empty draws sky only). */
  backdrop: readonly BackdropLayer[] = [];
  /** The backdrop being faded out under `backdrop`, while `fade` runs from 0 to 1. */
  previous: readonly BackdropLayer[] = [];
  /** The biome's look, and the last one's, faded with the backdrop. */
  look: SkyLook = CYCLE_LOOK;
  lookFrom: SkyLook = CYCLE_LOOK;
  fade = 1;
  /** The cycle's cloud cover (the old `OVERCAST`): grey (RGB) and how far it greys the sky, set by the weather. */
  overcast = 0x6a7280;
  overcastK = 0;
  /** Fog and wet overlays, 0..1, set by the weather. */
  fog = 0;
  wet = 0;
  /** World y of the horizon the backdrop stands on (the level's base ground). */
  horizonY = 148 * 8;
  /** Fraction of the view height the horizon sits at when the camera frames the squad (the old 0.68). */
  horizon = 0.68;
  /** Clouds drift this fast, px/s, and are drawn when the art book has them and the sky is open. */
  cloudSpeed = 4;
  clouds = true;

  constructor(dayLength: number, start: number) {
    this.dayLength = Math.max(1, dayLength);
    this.clock = ((start % 1) + 1) % 1;
  }

  /** How dark it is: 0 for day, up to 0.7 at the cycle's midnight (a biome's fixed darkness can go further). */
  get night(): number {
    const to = this.nightOf(this.look);
    return this.fade < 1 ? this.nightOf(this.lookFrom) * (1 - this.fade) + to * this.fade : to;
  }

  /** How far through the cycle's day it is: 0 at noon, 1 at midnight. */
  private get dark(): number {
    return 0.5 - 0.5 * Math.cos(this.clock * Math.PI * 2);
  }

  private nightOf(l: SkyLook): number {
    return l.cycle ? CYCLE_NIGHT * this.dark : l.night;
  }

  step(dt: number): void {
    this.clock = (this.clock + dt / this.dayLength) % 1;
    this.fade = Math.min(1, this.fade + dt / BACKDROP_FADE);
  }

  /** Show a biome: its backdrop and look, faded in over the current ones (a biome border), or at once (a new level). */
  show(layers: readonly BackdropLayer[], fadeIn: boolean, look: SkyLook = this.look): void {
    this.previous = fadeIn ? this.backdrop : [];
    this.lookFrom = fadeIn ? this.look : look;
    this.backdrop = layers;
    this.look = look;
    this.fade = fadeIn ? 0 : 1;
  }

  /** A look's sky colour `i` (0 top, 1 middle, 2 bottom): the cycle's by time of day and cloud, or its own. */
  private skyOf(l: SkyLook, i: number): number {
    if (!l.cycle) return l.colors[i];
    const u = this.dark;
    const clear = u < 0.5 ? mixRgb(DAY[i], DUSK[i], u * 2) : mixRgb(DUSK[i], NIGHT[i], (u - 0.5) * 2);
    return mixRgb(clear, mixRgb(this.overcast, NIGHT[i], u), this.overcastK);
  }

  /** Sky gradient colours into `out` (top, mid, bottom) and the star strength, for the current night. */
  colors(out: { top: number; mid: number; bottom: number; stars: number }): void {
    const f = this.fade;
    out.top = mixRgb(this.skyOf(this.lookFrom, 0), this.skyOf(this.look, 0), f);
    out.mid = mixRgb(this.skyOf(this.lookFrom, 1), this.skyOf(this.look, 1), f);
    out.bottom = mixRgb(this.skyOf(this.lookFrom, 2), this.skyOf(this.look, 2), f);
    out.stars = this.look.stars ? clamp01((this.night - 0.25) * 2) : 0;
  }

  get ambient(): number {
    // Hold full daylight a little into the evening, then fall to night
    const d = this.night / CYCLE_NIGHT;
    return mixRgb(AMBIENT_DAY, AMBIENT_NIGHT, clamp01((d - 0.15) / 0.6));
  }

  /** `base` (RGB) cast by the biome's tint, faded across a border. */
  tinted(base: number): number {
    const from = mixRgb(base, this.lookFrom.tint, this.lookFrom.tintA);
    return mixRgb(from, mixRgb(base, this.look.tint, this.look.tintA), this.fade);
  }
}
