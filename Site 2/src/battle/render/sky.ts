/**
 * The day–night clock and what it colours: the sky gradient, stars, the ambient light and the backdrop's tint.
 * The clock runs in ticks (the old game's `clock`, one cycle per `dayLength` seconds); night is 0 at noon and 1 at
 * midnight. The backdrop is a list of parallax layers a level feature sets per theme.
 */
import { mixRgb } from "@/engine";

/** One backdrop layer: an art id, how far it moves with the camera, and how far below the horizon its foot sits. */
export interface BackdropLayer {
  readonly art: string;
  readonly factor: number;
  readonly lift: number;
}

/** Sky colours by time of day, from the old scene: top, middle and bottom of the gradient. */
const DAY = [0x3a78d0, 0x86b6e6, 0xd8eaf6];
const DUSK = [0x2a2a5a, 0xc8607a, 0xffb070];
const NIGHT = [0x04050e, 0x0c1030, 0x1e2848];
/** Ambient light at noon and at midnight: deep enough that muzzle flashes and fires carry the night. */
const AMBIENT_DAY = 0xffffff;
const AMBIENT_NIGHT = 0x464e78;

export class SkyClock {
  /** 0..1 through the cycle; 0 is noon. */
  clock: number;
  /** Seconds per cycle. */
  dayLength: number;
  /** The backdrop, far to near (set by the level feature; empty draws sky only). */
  backdrop: BackdropLayer[] = [];
  /** World y of the horizon the backdrop stands on (the level's base ground). */
  horizonY = 148 * 8;
  /** Fraction of the view height the horizon sits at when the camera frames the squad (the old 0.68). */
  horizon = 0.68;
  /** Clouds drift this fast, px/s, and are drawn when the art book has them. */
  cloudSpeed = 4;

  constructor(dayLength: number, start: number) {
    this.dayLength = Math.max(1, dayLength);
    this.clock = ((start % 1) + 1) % 1;
  }

  /** 0 at noon, 1 at midnight. */
  get night(): number {
    return 0.5 - 0.5 * Math.cos(this.clock * Math.PI * 2);
  }

  step(dt: number): void {
    this.clock = (this.clock + dt / this.dayLength) % 1;
  }

  /** Sky gradient colours into `out` (top, mid, bottom) and the star strength, for the current night. */
  colors(out: { top: number; mid: number; bottom: number; stars: number }): void {
    const n = this.night;
    const dusk = Math.min(1, n * 2);
    const deep = Math.max(0, n * 2 - 1);
    out.top = mixRgb(mixRgb(DAY[0], DUSK[0], dusk), NIGHT[0], deep);
    out.mid = mixRgb(mixRgb(DAY[1], DUSK[1], dusk), NIGHT[1], deep);
    out.bottom = mixRgb(mixRgb(DAY[2], DUSK[2], dusk), NIGHT[2], deep);
    out.stars = deep;
  }

  get ambient(): number {
    // Hold full daylight a little into the evening, then fall to night
    const n = this.night;
    return mixRgb(AMBIENT_DAY, AMBIENT_NIGHT, Math.min(1, Math.max(0, (n - 0.15) / 0.6)));
  }
}
