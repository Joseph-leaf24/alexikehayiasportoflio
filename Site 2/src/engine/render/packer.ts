/**
 * Skyline packer: the page's filled outline is a list of horizontal segments, and each rectangle goes where its top
 * edge would be lowest (then leftmost). Good for many small sprites of mixed sizes, packed once at boot.
 */
export class Skyline {
  /** Segment starts, heights (filled depth from the top) and widths, left to right. */
  private xs: number[] = [0];
  private ys: number[] = [0];
  private ws: number[];
  /** Where the last successful `insert` put its rectangle. */
  x = 0;
  y = 0;
  /** Pixels covered by inserted rectangles, padding included. */
  used = 0;

  constructor(
    readonly width: number,
    readonly height: number,
    /** Gap kept to the right of and below each rectangle. */
    readonly padding = 1
  ) {
    this.ws = [width];
  }

  /** Places a w×h rectangle; returns false when the page is full. On success the spot is in `x` and `y`. */
  insert(w: number, h: number): boolean {
    const pw = w + this.padding;
    const ph = h + this.padding;
    let bestI = -1;
    let bestY = Infinity;
    let bestW = Infinity;
    for (let i = 0; i < this.xs.length; i++) {
      const y = this.fit(i, w, pw, h);
      if (y < 0) continue;
      if (y < bestY || (y === bestY && this.ws[i] < bestW)) {
        bestI = i;
        bestY = y;
        bestW = this.ws[i];
      }
    }
    if (bestI < 0) return false;
    this.x = this.xs[bestI];
    this.y = bestY;
    // A rectangle that reaches the right or bottom edge needs no padding there
    this.place(bestI, Math.min(pw, this.width - this.x), Math.min(bestY + ph, this.height));
    this.used += pw * ph;
    return true;
  }

  /** The top y for a rectangle starting at segment i, or -1 if it doesn't fit there. */
  private fit(i: number, w: number, pw: number, h: number): number {
    const x = this.xs[i];
    if (x + w > this.width) return -1;
    const span = Math.min(pw, this.width - x);
    let left = span;
    let y = 0;
    for (let j = i; left > 0; j++) {
      if (j >= this.xs.length) return -1;
      y = Math.max(y, this.ys[j]);
      if (y + h > this.height) return -1;
      left -= this.ws[j];
    }
    return y;
  }

  /** Raises the skyline under [x, x+w) to `top`, trimming and merging the segments it covers. */
  private place(i: number, w: number, top: number) {
    const x = this.xs[i];
    this.xs.splice(i, 0, x);
    this.ys.splice(i, 0, top);
    this.ws.splice(i, 0, w);
    const end = x + w;
    for (let j = i + 1; j < this.xs.length; ) {
      const segEnd = this.xs[j] + this.ws[j];
      if (segEnd <= end) {
        this.xs.splice(j, 1);
        this.ys.splice(j, 1);
        this.ws.splice(j, 1);
      } else {
        if (this.xs[j] < end) {
          this.ws[j] = segEnd - end;
          this.xs[j] = end;
        }
        break;
      }
    }
    for (let j = 0; j < this.xs.length - 1; ) {
      if (this.ys[j] === this.ys[j + 1]) {
        this.ws[j] += this.ws[j + 1];
        this.xs.splice(j + 1, 1);
        this.ys.splice(j + 1, 1);
        this.ws.splice(j + 1, 1);
      } else j++;
    }
  }

  /** Segment count, for tests. */
  get segments() {
    return this.xs.length;
  }
}
