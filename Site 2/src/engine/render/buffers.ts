import type { LightBuffer, ParticleBuffer, SpriteBuffer, TextBuffer } from "./types";

/** Doubles a typed array's capacity, keeping its contents. Only runs when a buffer outgrows its capacity. */
function grow<T extends Float32Array | Uint32Array | Uint16Array | Uint8Array>(a: T, n: number): T {
  const b = new (a.constructor as new (n: number) => T)(n);
  b.set(a);
  return b;
}

/** Struct-of-arrays sprites; grows by doubling, so a steady load allocates nothing per frame. */
export class SpriteList implements SpriteBuffer {
  count = 0;
  frame: Uint32Array;
  x: Float32Array;
  y: Float32Array;
  layer: Uint8Array;
  flags: Uint8Array;
  tint: Uint32Array;
  scale: Float32Array;
  rot: Float32Array;

  constructor(capacity = 1024) {
    const n = Math.max(16, capacity);
    this.frame = new Uint32Array(n);
    this.x = new Float32Array(n);
    this.y = new Float32Array(n);
    this.layer = new Uint8Array(n);
    this.flags = new Uint8Array(n);
    this.tint = new Uint32Array(n);
    this.scale = new Float32Array(n);
    this.rot = new Float32Array(n);
  }

  get capacity() {
    return this.x.length;
  }

  push(frame: number, x: number, y: number, layer: number, flags: number, tint: number, scale: number, rot: number) {
    if (this.count === this.x.length) this.resize(this.count * 2);
    const i = this.count++;
    this.frame[i] = frame;
    this.x[i] = x;
    this.y[i] = y;
    this.layer[i] = layer < 0 ? 0 : layer > 15 ? 15 : layer;
    this.flags[i] = flags;
    this.tint[i] = tint >>> 0;
    this.scale[i] = scale;
    this.rot[i] = rot;
  }

  clear() {
    this.count = 0;
  }

  private resize(n: number) {
    this.frame = grow(this.frame, n);
    this.x = grow(this.x, n);
    this.y = grow(this.y, n);
    this.layer = grow(this.layer, n);
    this.flags = grow(this.flags, n);
    this.tint = grow(this.tint, n);
    this.scale = grow(this.scale, n);
    this.rot = grow(this.rot, n);
  }
}

/** Struct-of-arrays particles: position, size in world px (1–3), RGBA colour and flags. */
export class ParticleList implements ParticleBuffer {
  count = 0;
  x: Float32Array;
  y: Float32Array;
  size: Uint8Array;
  color: Uint32Array;
  flags: Uint8Array;

  constructor(capacity = 4096) {
    const n = Math.max(16, capacity);
    this.x = new Float32Array(n);
    this.y = new Float32Array(n);
    this.size = new Uint8Array(n);
    this.color = new Uint32Array(n);
    this.flags = new Uint8Array(n);
  }

  get capacity() {
    return this.x.length;
  }

  push(x: number, y: number, size: number, color: number, flags: number) {
    if (this.count === this.x.length) {
      const n = this.count * 2;
      this.x = grow(this.x, n);
      this.y = grow(this.y, n);
      this.size = grow(this.size, n);
      this.color = grow(this.color, n);
      this.flags = grow(this.flags, n);
    }
    const i = this.count++;
    this.x[i] = x;
    this.y[i] = y;
    this.size[i] = size < 1 ? 1 : size > 255 ? 255 : size;
    this.color[i] = color >>> 0;
    this.flags[i] = flags;
  }

  clear() {
    this.count = 0;
  }
}

/** Struct-of-arrays point lights: position, radius (world px), RGB colour and intensity (1 = full). */
export class LightList implements LightBuffer {
  count = 0;
  x: Float32Array;
  y: Float32Array;
  radius: Float32Array;
  color: Uint32Array;
  intensity: Float32Array;

  constructor(capacity = 128) {
    const n = Math.max(16, capacity);
    this.x = new Float32Array(n);
    this.y = new Float32Array(n);
    this.radius = new Float32Array(n);
    this.color = new Uint32Array(n);
    this.intensity = new Float32Array(n);
  }

  push(x: number, y: number, radius: number, color: number, intensity: number) {
    if (this.count === this.x.length) {
      const n = this.count * 2;
      this.x = grow(this.x, n);
      this.y = grow(this.y, n);
      this.radius = grow(this.radius, n);
      this.color = grow(this.color, n);
      this.intensity = grow(this.intensity, n);
    }
    const i = this.count++;
    this.x[i] = x;
    this.y[i] = y;
    this.radius[i] = radius;
    this.color[i] = color >>> 0;
    this.intensity[i] = intensity;
  }

  clear() {
    this.count = 0;
  }
}

/** Text runs for the overlay; laid out into glyph sprites at render time. Strings are held, not copied. */
export class TextList implements TextBuffer {
  count = 0;
  text: string[] = [];
  x: Float32Array;
  y: Float32Array;
  color: Uint32Array;
  scale: Uint8Array;
  align: Uint8Array;
  layer: Uint8Array;

  constructor(capacity = 64) {
    const n = Math.max(16, capacity);
    this.x = new Float32Array(n);
    this.y = new Float32Array(n);
    this.color = new Uint32Array(n);
    this.scale = new Uint8Array(n);
    this.align = new Uint8Array(n);
    this.layer = new Uint8Array(n);
  }

  /** align: 0 left, 1 centre, 2 right of x; y is the top of the glyphs. Scale is whole pixels (1–15). */
  push(text: string, x: number, y: number, color: number, scale: number, align: 0 | 1 | 2, layer: number) {
    if (this.count === this.x.length) {
      const n = this.count * 2;
      this.x = grow(this.x, n);
      this.y = grow(this.y, n);
      this.color = grow(this.color, n);
      this.scale = grow(this.scale, n);
      this.align = grow(this.align, n);
      this.layer = grow(this.layer, n);
    }
    const i = this.count++;
    this.text[i] = text;
    this.x[i] = x;
    this.y[i] = y;
    this.color[i] = color >>> 0;
    const s = Math.round(scale);
    this.scale[i] = s < 1 ? 1 : s > 15 ? 15 : s;
    this.align[i] = align;
    this.layer[i] = layer < 0 ? 0 : layer > 15 ? 15 : layer;
  }

  clear() {
    // Drop the string references so old text can be collected
    for (let i = 0; i < this.count; i++) this.text[i] = "";
    this.count = 0;
  }
}
