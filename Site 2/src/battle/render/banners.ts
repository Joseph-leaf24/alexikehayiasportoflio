/**
 * Banners (title text across the screen: "LEVEL 1", "BOSS AHEAD") as a small priority queue shown one after
 * another, and popups (short labels floating up from a world point: "P2 DOWN", "!"). Both age in ticks; the text
 * pass draws them in the pixel font.
 */

export interface BannerOptions {
  /** Seconds fully shown (default 2.6). */
  hold?: number;
  /** Higher goes first; a higher one arriving cuts the current one short. */
  priority?: number;
}

export interface Banner {
  text: string;
  sub: string;
  color: number;
  hold: number;
  priority: number;
  /** Seconds since it came up. */
  t: number;
}

/** Fade in and out, seconds. */
export const BANNER_FADE = 0.35;

export class Banners {
  /** The banner on screen, or null. */
  current: Banner | null = null;
  readonly queue: Banner[] = [];
  /** Called when a banner comes up (the game emits the `banner` event from it). */
  onShow: ((b: Banner) => void) | null = null;

  push(text: string, sub = "", color = 0xffffffff, o?: BannerOptions): void {
    const b: Banner = { text, sub, color, hold: o?.hold ?? 2.6, priority: o?.priority ?? 0, t: 0 };
    let k = this.queue.length;
    while (k > 0 && this.queue[k - 1].priority < b.priority) k--;
    this.queue.splice(k, 0, b);
    // A more important banner cuts the current one short (it fades out now)
    const cur = this.current;
    if (cur && b.priority > cur.priority) cur.t = Math.max(cur.t, BANNER_FADE + cur.hold);
  }

  step(dt: number): void {
    const cur = this.current;
    if (cur) {
      cur.t += dt;
      if (cur.t < cur.hold + 2 * BANNER_FADE) return;
      this.current = null;
    }
    const next = this.queue.shift();
    if (next) {
      this.current = next;
      this.onShow?.(next);
    }
  }

  /** 0..1 visibility of the current banner. */
  get alpha(): number {
    const b = this.current;
    if (!b) return 0;
    if (b.t < BANNER_FADE) return b.t / BANNER_FADE;
    const out = b.t - b.hold - BANNER_FADE;
    return out > 0 ? Math.max(0, 1 - out / BANNER_FADE) : 1;
  }

  clear(): void {
    this.current = null;
    this.queue.length = 0;
  }
}

/** Seconds a popup floats. */
export const POPUP_LIFE = 1.2;
const POPUPS = 24;

export class Popups {
  n = 0;
  readonly text: string[] = [];
  readonly x = new Float32Array(POPUPS);
  readonly y = new Float32Array(POPUPS);
  readonly t = new Float32Array(POPUPS);
  readonly color = new Uint32Array(POPUPS);

  add(text: string, x: number, y: number, color: number): void {
    let k = this.n;
    if (k >= POPUPS) {
      k = 0;
      for (let j = 1; j < this.n; j++) if (this.t[j] > this.t[k]) k = j;
    } else this.n++;
    this.text[k] = text;
    this.x[k] = x;
    this.y[k] = y;
    this.t[k] = 0;
    this.color[k] = color >>> 0;
  }

  step(dt: number): void {
    let k = 0;
    while (k < this.n) {
      this.t[k] += dt;
      if (this.t[k] < POPUP_LIFE) {
        k++;
        continue;
      }
      const last = --this.n;
      this.text[k] = this.text[last];
      this.x[k] = this.x[last];
      this.y[k] = this.y[last];
      this.t[k] = this.t[last];
      this.color[k] = this.color[last];
      this.text[last] = "";
    }
  }

  shiftX(dx: number): void {
    for (let k = 0; k < this.n; k++) this.x[k] += dx;
  }
}
