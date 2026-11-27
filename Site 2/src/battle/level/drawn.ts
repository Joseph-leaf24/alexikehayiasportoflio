/**
 * The old scene's hand-drawn props (`PixelScene` drawWorld's `switch (p.kind)`). Content is frozen after install but
 * props arrive as the level is generated, so every kind draws with art registered up front:
 * - things of any size (a leg, a cable, a rung) are runs of a strip: plain colour painted at every length from 1 to
 *   8 px, so a run of any length is whole pieces plus one;
 * - fixed shapes (a tent) are painted once with the old case's own code, at (0, 0) for the prop's x, y; animated
 *   ones (a flag) as a few poses, picked by time with `pen.cycle`;
 * - glows are emissive sprites (they skip the dark and bloom), plus `pen.light` for light cast around; the old
 *   lighting pass's night glows are drawn after each prop by `glows.ts`.
 *
 * To port another kind: declare its strips and shapes with the others (`strip`, `shape`), write its draw from the
 * old case (same colours and offsets; `x`, `y` are its world position), and add it to `DRAWN`.
 */
import { INK, pixelText } from "@/components/pixel/art";
import { BUILDINGS } from "@/components/pixel/buildings";
import { STATIONS, type Level } from "@/components/pixel/level";
import { hash2, SPRITE_EMISSIVE, type RenderFrame } from "@/engine";
import { registerShapeArt, registerStripArt, STRIP_PIECE, type ShapeSpec } from "../art";
import type { BattleGame } from "../game/BattleGame";
import type { Content } from "../game/content";
import { artLookup, LAYER } from "../render";
import type { LevelSource } from "./generator";

export type Prop = Level["props"][number];
export type PropOf<K extends Prop["kind"]> = Extract<Prop, { kind: K }>;
export type Draw<K extends Prop["kind"]> = (pen: Pen, p: PropOf<K>, x: number, y: number) => void;

export interface Strip {
  readonly id: string;
  readonly axis: "x" | "y";
}

export interface Shape {
  readonly id: string;
}

/** A sprite tint that only fades it, `a` from 0 to 1 (glows dimming and flickering). */
export const alpha = (a: number): number => (0xffffff00 | Math.round(Math.min(1, Math.max(0, a)) * 255)) >>> 0;

/** Registers each strip and shape declared below. */
const ART: ((content: Content) => void)[] = [];

/** A strip, art id `strip.<name>`: a run along `axis`, `colors` its rows (or columns) across, one px each. */
export function strip(name: string, axis: "x" | "y", ...colors: string[]): Strip {
  const id = `strip.${name}`;
  ART.push((content) => registerStripArt(content, id, axis, colors));
  return { id, axis };
}

/** A fixed shape, art id `shape.<name>`, painted with the old case's code at (0, 0) (see `ShapeSpec`). */
export function shape(name: string, spec: ShapeSpec): Shape {
  const id = `shape.${name}`;
  ART.push((content) => registerShapeArt(content, id, spec));
  return { id };
}

/** Register the art every drawn prop uses (at install; it's painted when first drawn). */
export function registerDrawnArt(content: Content): void {
  for (const register of ART) register(content);
}

/** What a prop's draw writes with, in world px, on the props layer. */
export class Pen {
  /** Simulation seconds, for the animated ones. */
  time = 0;
  /** 0 at noon, 1 at midnight. */
  night = 0;
  private frame: RenderFrame | null = null;
  private readonly art: (id: string) => number;

  constructor(
    private readonly game: BattleGame,
    private readonly level: LevelSource
  ) {
    this.art = artLookup(game);
  }

  begin(frame: RenderFrame, time: number): void {
    this.frame = frame;
    this.time = time;
    this.night = this.game.sky.night;
  }

  end(): void {
    this.frame = null;
  }

  /** A run of `len` px of a strip from (x, y): rightwards for an x strip, downwards for a y one. */
  run(s: Strip, x: number, y: number, len: number, flags = 0, tint = 0xffffffff): void {
    const f = this.frame;
    const a = this.art(s.id);
    if (!f || a <= 0) return;
    const book = this.game.art;
    const along = s.axis === "x";
    for (let k = 0; k < len; k += STRIP_PIECE) {
      const fr = book.frame(a, Math.min(STRIP_PIECE, len - k) - 1, 1);
      if (fr < 0) return;
      f.sprites.push(fr, along ? x + k : x, along ? y : y + k, LAYER.PROPS, flags, tint, 1, 0);
    }
  }

  /** An x strip along a curve, `n` px from x at height `yAt(k)`; each stretch at one height is a single run. */
  curve(s: Strip, x: number, n: number, yAt: (k: number) => number): void {
    let from = 0;
    let y0 = yAt(0);
    for (let k = 1; k <= n; k++) {
      const yk = k < n ? yAt(k) : NaN;
      if (yk === y0) continue;
      this.run(s, x + from, y0, k - from);
      from = k;
      y0 = yk;
    }
  }

  /** A shape's pose with its pivot at (x, y), turned `rot` radians about it. */
  shape(s: Shape, pose: number, x: number, y: number, flags = 0, tint = 0xffffffff, rot = 0): void {
    const f = this.frame;
    const a = this.art(s.id);
    const fr = f && a > 0 ? this.game.art.frame(a, pose, 1) : -1;
    if (f && fr >= 0) f.sprites.push(fr, x, y, LAYER.PROPS, flags, tint, 1, rot);
  }

  /** The pose now of a loop of `poses` poses played `rate` times a second. */
  cycle(poses: number, rate: number): number {
    return Math.floor(this.time * rate * poses) % poses;
  }

  /** A light (RGB colour), while the frame holds under a quarter of the light budget: the rest is the fighting's. */
  light(x: number, y: number, radius: number, color: number, intensity: number): void {
    const f = this.frame;
    if (f && f.lights.count < this.game.budgets.maxLights >> 2) f.lights.push(x, y, radius, color, intensity);
  }

  /** Whether a `legs` or `support` prop's post at `end` (0 left, 1 right) still stands on its beams. */
  stands(p: PropOf<"legs" | "support">, end: 0 | 1): boolean {
    return this.level.stands(this.game.terrain, p, end, this.game.origin.cols);
  }
}

// Stilt legs and posts stand on beams (which have no tile): a post whose beams are gone isn't drawn
const LEG = strip("leg", "y", "#6a421c", "#6a421c");
const BRACE = strip("brace", "x", "#5a3816");
const POST = strip("post", "y", "#4a5468", "#4a5468");
const BARB = strip("barb", "x", "#9aa1b0");
const BARB_POST = strip("barb.post", "y", "#5d6880");
const ROPE_POST = strip("rope.post", "y", "#5a3816", "#5a3816");
const ROPE = strip("rope", "x", "#a08050");
const ROPE_HANGER = strip("rope.hanger", "y", "#7a5a30");
const MAST = strip("mast", "y", "#5d6578");
const RUNG = strip("rung", "x", "#5d6578");
const BEACON_ON = strip("beacon.on", "x", "#ff3b3b", "#ff3b3b");
const BEACON_OFF = strip("beacon.off", "x", "#5a1a1a", "#5a1a1a");
const CABLE = strip("cable", "x", "#8a909c", INK);
const CABLE_HANGER = strip("cable.hanger", "y", "#5a606c");
const SILO_LID = strip("silo", "x", "#14161e", "#14161e");
/** The fittings lamps hang from and alarms and searchlights sit on; their glows are in `glows.ts`. */
const LAMP = strip("lamp", "x", "#3a3f4f", "#3a3f4f");
const MOUNT = strip("mount", "x", "#3a3f4f", "#3a3f4f", "#3a3f4f");

const BRAZIER = shape("brazier", {
  w: 9,
  h: 9,
  ox: 4,
  oy: 9,
  paint(c) {
    c.fillStyle = INK;
    c.fillRect(-4, -9, 9, 3);
    c.fillRect(-2, -6, 5, 6);
    c.fillStyle = "#6a6e60";
    c.fillRect(-3, -8, 7, 1);
    c.fillStyle = "#4a4e44";
    c.fillRect(-1, -6, 3, 6);
  },
});

const TENT = shape("tent", {
  w: 27,
  h: 14,
  ox: 13,
  oy: 14,
  paint(c) {
    c.fillStyle = INK;
    for (let k = 0; k < 12; k++) c.fillRect(-(12 - k) - 1, -k - 1, (12 - k) * 2 + 3, 1);
    for (let k = 0; k < 11; k++) {
      c.fillStyle = "#5a6a3a";
      c.fillRect(-(11 - k), -k - 1, (11 - k) * 2 + 1, 1);
      c.fillStyle = "#6e7e4a";
      c.fillRect(-(11 - k), -k - 1, 11 - k, 1);
    }
    c.fillStyle = "#1a1e14";
    for (let k = 0; k < 7; k++) {
      const half = Math.floor((7 - k) / 2);
      c.fillRect(-half, -k - 1, half * 2 + 1, 1);
    }
    c.fillStyle = "#8a7a50";
    c.fillRect(0, -14, 1, 3);
  },
});

/** A traffic light, a pose per state (`lightState`); its lit lamp's glow is in `glows.ts`. */
const STOPLIGHT = shape("stoplight", {
  w: 5,
  h: 27,
  ox: 2,
  oy: 27,
  poses: 3,
  paint(c, st) {
    c.fillStyle = "#3a3f4f";
    c.fillRect(0, -18, 1, 18);
    c.fillStyle = INK;
    c.fillRect(-2, -27, 5, 10);
    c.fillStyle = "#1e1e24";
    c.fillRect(-1, -26, 3, 8);
    c.fillStyle = st === 2 ? "#ff3b3b" : "#4a1414";
    c.fillRect(-1, -26, 3, 2);
    c.fillStyle = st === 1 ? "#ffb020" : "#4a3410";
    c.fillRect(-1, -23, 3, 2);
    c.fillStyle = st === 0 ? "#5aff6a" : "#123a1a";
    c.fillRect(-1, -20, 3, 2);
  },
});

/** The old `lightState` (`game.ts`): a traffic light at true-column x at time t, 0 green, 1 amber, 2 red. */
export function lightState(t: number, x: number): number {
  const u = (((t / 14 + x * 0.00071) % 1) + 1) % 1;
  return u < 0.46 ? 0 : u < 0.54 ? 1 : 2;
}

/** A streetlight's pole, foot and arm; its lamp is a strip of its own, which glows at night. */
const STREETLIGHT = shape("streetlight", {
  w: 7,
  h: 24,
  ox: 1,
  oy: 24,
  paint(c) {
    c.fillStyle = "#4a4e5a";
    c.fillRect(0, -24, 1, 24);
    c.fillRect(-1, -2, 3, 2);
    c.fillRect(0, -24, 6, 1);
  },
});
const STREETLAMP_ON = strip("streetlamp.on", "x", "#fff3c0");
const STREETLAMP_OFF = strip("streetlamp.off", "x", "#c8ced9");

/** A radar on its post, a pose per dish width (2–12 px) as it turns. */
const RADAR = shape("radar", {
  w: 14,
  h: 16,
  ox: 6,
  oy: 16,
  poses: 11,
  paint(c, pose) {
    const w = pose + 2;
    c.fillStyle = "#5d6578";
    c.fillRect(0, -8, 2, 8);
    c.fillStyle = INK;
    c.fillRect(1 - w / 2 - 1, -16, w + 2, 9);
    c.fillStyle = "#c8ced9";
    c.fillRect(1 - w / 2, -15, w, 7);
    c.fillStyle = "#8a90a0";
    c.fillRect(1 - w / 2, -10, w, 2);
  },
});

/** An oil derrick's lattice tower; its row at the prop's y is painted too, as the old one was. */
const DERRICK = shape("derrick", {
  w: 15,
  h: 47,
  ox: 7,
  oy: 46,
  paint(c) {
    c.fillStyle = "#4a4038";
    for (let k = 0; k < 44; k++) {
      const half = Math.round(7 - (k / 44) * 6);
      c.fillRect(-half, -k, 1, 1);
      c.fillRect(half, -k, 1, 1);
      if (k % 7 === 0) c.fillRect(-half, -k, half * 2 + 1, 1);
    }
    c.fillRect(-2, -46, 5, 2);
  },
});

/** The pumpjack beside a derrick, nodding (the old `sin(t * 2) * 4`); pivoted at the derrick's foot, left of it. */
const PUMP_POSES = 8;
const PUMP_RATE = 2 / (2 * Math.PI);
const PUMPJACK = shape("pumpjack", {
  w: 18,
  h: 19,
  ox: -7,
  oy: 19,
  poses: PUMP_POSES,
  paint(c, pose) {
    const nod = Math.sin((pose / PUMP_POSES) * 2 * Math.PI) * 4;
    c.fillStyle = "#2e2a26";
    c.fillRect(14, -12, 2, 12);
    for (let k = -8; k <= 8; k++) c.fillRect(15 + k, Math.round(-13 - (nod * k) / 8), 1, 2);
    c.fillRect(22, Math.round(-15 - nod), 3, 5);
  },
});

/**
 * A waterfall column: bands 3 px each, pose = look × 12 + length - 1. The old bands step on a fixed 3-px grid, so a
 * column has only 4 looks (which band is on top), and is drawn down its height in pieces of 12 (one round of bands).
 */
const FALL = 12;
const FALL_BANDS = ["#e8f8ff", "#9fd8f0", "#5ab0d8", "#4aa0c8"];
const WATERFALL = shape("waterfall", {
  w: 1,
  h: FALL,
  ox: 0,
  oy: 0,
  poses: FALL_BANDS.length * FALL,
  paint(c, pose) {
    const look = Math.floor(pose / FALL);
    const len = (pose % FALL) + 1;
    for (let m = 0; m * 3 < len; m++) {
      c.fillStyle = FALL_BANDS[(m + look) % FALL_BANDS.length];
      c.fillRect(0, m * 3, 1, Math.min(3, len - m * 3));
    }
  },
});
const FOAM = strip("foam", "x", "#f4fcff");

/** Awnings: a 3-px strip per colour the building specs give them, and the white of their stripes. */
const AWNING_WHITE = "#f4f0e8";
const AWNINGS = new Map(
  [...new Set([AWNING_WHITE, ...BUILDINGS.flatMap((s) => (s.awning ? [s.awning] : []))])].map((c) => [
    c,
    strip(`awning.${c.slice(1)}`, "x", c, c, c),
  ])
);
const AWNING_EDGE = strip("awning.edge", "x", INK);

/**
 * Signs: a shape per text, board and ink the old generator can place (`kind: "sign"` in `components/pixel/level.ts`):
 * the building specs' (with its defaults) and the station names, on the metro's board and the railway station's.
 */
type SignLook = Pick<PropOf<"sign">, "text" | "board" | "ink">;
const signName = (s: SignLook) => `sign.${s.text}.${s.board.slice(1)}.${s.ink.slice(1)}`;
const SIGN_LOOKS: SignLook[] = [
  ...BUILDINGS.flatMap((b) => (b.sign ? [{ text: b.sign, board: b.board ?? "#1e1e24", ink: b.ink ?? "#ffffff" }] : [])),
  ...STATIONS.flatMap((text) => ["#2a4a9a", "#1e2a3a"].map((board) => ({ text, board, ink: "#ffffff" }))),
];
const SIGNS = new Map<string, Shape>();
for (const look of SIGN_LOOKS) {
  const name = signName(look);
  if (SIGNS.has(name)) continue;
  // Centred on x (rounded as the old one was), its feet at y
  const w = look.text.length * 4 + 3;
  const sx = Math.round(-w / 2);
  SIGNS.set(
    name,
    shape(name, {
      w: w + 2,
      h: 11,
      ox: 1 - sx,
      oy: 11,
      paint(c) {
        c.fillStyle = INK;
        c.fillRect(sx - 1, -11, w + 2, 10);
        c.fillRect(sx + 2, -2, 1, 2);
        c.fillRect(sx + w - 3, -2, 1, 2);
        c.fillStyle = look.board;
        c.fillRect(sx, -10, w, 8);
        pixelText(c, look.text, sx + 2, -9, look.ink, false);
      },
    })
  );
}

/** A sign prop's art id, or "" for a look not registered (the old generator gained a sign: add it above). */
export const signArt = (p: SignLook): string => SIGNS.get(signName(p))?.id ?? "";

/** The flag's wave, a loop of poses (the old one was `sin(t * 6 - i * 0.6)`). */
const FLAG_POSES = 8;
const FLAG_RATE = 6 / (2 * Math.PI);
const FLAG = shape("flag", {
  w: 11,
  h: 20,
  ox: 0,
  oy: 20,
  poses: FLAG_POSES,
  paint(c, pose) {
    c.fillStyle = "#c9ced8";
    c.fillRect(0, -18, 1, 18);
    const phase = (pose / FLAG_POSES) * 2 * Math.PI;
    for (let i = 0; i < 10; i++) {
      const o = Math.round(Math.sin(phase - i * 0.6) * (i / 9) * 1.5);
      c.fillStyle = "#c8302a";
      c.fillRect(1 + i, -18 + o, 1, 6);
      if (i > 2 && i < 7) {
        c.fillStyle = "#1a1a1a";
        c.fillRect(1 + i, -16 + o, 1, 2);
      }
    }
  },
});

/** Two legs with cross braces between them every 8 px, while both stand. */
const legs: Draw<"legs"> = (pen, p, x, y) => {
  const left = pen.stands(p, 0);
  const right = pen.stands(p, 1);
  if (left) pen.run(LEG, x, y, p.h);
  if (right) pen.run(LEG, x + p.w - 2, y, p.h);
  if (!left || !right) return;
  const span = p.w - 4;
  for (let k = 0; k < p.h - 6; k += 8) {
    pen.run(BRACE, x + 2, y + k + 4, span);
    for (let s = 2; s < 6 && s < span; s += 2) pen.run(BRACE, x + 2 + Math.round((span * s) / 6), y + k + 4 + s, 1);
  }
};

const support: Draw<"support"> = (pen, p, x, y) => {
  if (pen.stands(p, 0)) pen.run(POST, x - 1, y, p.h);
};

/** Barbed wire: a zigzag strand, with a post every 8 px. */
const wire: Draw<"wire"> = (pen, p, x, y) => {
  for (let k = 0; k < p.w; k += 2) pen.run(BARB, x + k, y - 3 - ((k >> 1) & 1), Math.min(2, p.w - k));
  for (let k = 0; k < p.w; k += 8) pen.run(BARB_POST, x + k, y - 5, 5);
};

/** Rope handrails over a bridge deck: a post at each end, the rope sagging between, hangers every 6 px. */
const rope: Draw<"rope"> = (pen, p, x, y) => {
  pen.run(ROPE_POST, x - 1, y - 11, 11);
  pen.run(ROPE_POST, x + p.w - 1, y - 11, 11);
  const sag = (k: number) => Math.round(Math.sin((k / p.w) * Math.PI) * 4);
  pen.curve(ROPE, x, p.w + 1, (k) => y - 10 + sag(k));
  for (let k = 3; k <= p.w; k += 6) pen.run(ROPE_HANGER, x + k, y - 9 + sag(k), 9 - sag(k));
};

/** A mast with rungs, and a red beacon blinking on top. */
const antenna: Draw<"antenna"> = (pen, p, x, y) => {
  pen.run(MAST, x, y - p.h, p.h);
  for (let k = 6; k < p.h; k += 8) pen.run(RUNG, x - 2, y - k, 5);
  const on = Math.floor(pen.time * 1.5) % 2 === 1;
  pen.run(on ? BEACON_ON : BEACON_OFF, x - 1, y - p.h - 2, 2, SPRITE_EMISSIVE);
};

/** A suspension bridge cable: a parabolic sag between the towers, with hangers down to the deck. */
const cable: Draw<"cable"> = (pen, p, x, y) => {
  const len = Math.max(1, p.x2 - p.x);
  const at = (k: number) => {
    const u = k / len;
    return Math.round(y + (p.y2 - y) * u + p.sag * 4 * u * (1 - u));
  };
  pen.curve(CABLE, x, len + 1, at);
  for (let k = 4; k <= len; k += 8) {
    const yy = at(k);
    if (yy < p.deck - 3) pen.run(CABLE_HANGER, x + k, yy + 2, Math.round(p.deck - yy - 2));
  }
};

const tent: Draw<"tent"> = (pen, _p, x, y) => pen.shape(TENT, 0, x, y);

const flag: Draw<"flag"> = (pen, _p, x, y) => pen.shape(FLAG, pen.cycle(FLAG_POSES, FLAG_RATE), x, y);

/** A missile silo's hatch, flush with the ground. */
const silo: Draw<"silo"> = (pen, _p, x, y) => pen.run(SILO_LID, x - 14, y, 28);

const brazier: Draw<"brazier"> = (pen, _p, x, y) => pen.shape(BRAZIER, 0, x, y);

const lamp: Draw<"lamp"> = (pen, _p, x, y) => pen.run(LAMP, x - 1, y, 4);

const stoplight: Draw<"stoplight"> = (pen, p, x, y) => pen.shape(STOPLIGHT, lightState(pen.time, p.x), x, y);

/** A streetlight's lamp is lit (and emissive) after dusk. */
const streetlight: Draw<"streetlight"> = (pen, _p, x, y) => {
  pen.shape(STREETLIGHT, 0, x, y);
  if (pen.night > 0.3) pen.run(STREETLAMP_ON, x + 4, y - 23, 3, SPRITE_EMISSIVE);
  else pen.run(STREETLAMP_OFF, x + 4, y - 23, 3);
};

/** The dish's width follows `|cos(t * 1.5)|`, so its pose is picked from time directly. */
const radar: Draw<"radar"> = (pen, _p, x, y) =>
  pen.shape(RADAR, Math.round(Math.abs(Math.cos(pen.time * 1.5)) * 10), x, y);

const derrick: Draw<"derrick"> = (pen, _p, x, y) => {
  pen.shape(DERRICK, 0, x, y);
  pen.shape(PUMPJACK, pen.cycle(PUMP_POSES, PUMP_RATE), x, y);
};

/** Each column's bands scroll down at 80 px/s from its own phase; foam flickers at the foot, 6 times a second. */
const waterfall: Draw<"waterfall"> = (pen, p, x, y) => {
  const n = FALL_BANDS.length;
  for (let k = 0; k < p.w; k++) {
    const look = (((Math.floor((hash2(k, 99) * 12 - pen.time * 80) / 3) % n) + n) % n) * FALL;
    for (let yy = 0; yy < p.h; yy += FALL) pen.shape(WATERFALL, look + Math.min(FALL, p.h - yy) - 1, x + k, y + yy);
  }
  const tick = Math.floor(pen.time * 6);
  for (let k = -3; k < p.w + 3; k++)
    if (hash2(k + p.x, tick) > 0.5) pen.run(FOAM, x + k, y + p.h - 2 - Math.floor(hash2(k, tick + 1) * 3), 1);
};

/** 2-px stripes out left of x, a px lower every 5; a stripe is one run, or two where it steps down. */
const awning: Draw<"awning"> = (pen, p, x, y) => {
  let from = 0;
  for (let k = 1; k <= p.w; k++) {
    if (k < p.w && k >> 1 === from >> 1 && Math.floor(k / 5) === Math.floor(from / 5)) continue;
    const s = AWNINGS.get((from >> 1) % 2 ? AWNING_WHITE : p.color);
    if (s) pen.run(s, x - k, y + Math.floor(from / 5), k - from);
    from = k;
  }
  pen.run(AWNING_EDGE, x - p.w - 1, y + Math.floor(p.w / 5) + 3, p.w + 1);
};

/** A sign; a neon one's glow is in `glows.ts`. */
const sign: Draw<"sign"> = (pen, p, x, y) => {
  const s = SIGNS.get(signName(p));
  if (s) pen.shape(s, 0, x, y);
};

/** Alarms and searchlights sit on the same mount (any prop: the draw table checks the kinds). */
const mount = (pen: Pen, _p: Prop, x: number, y: number) => pen.run(MOUNT, x - 1, y - 2, 4);

const DRAWN: { [K in Prop["kind"]]?: Draw<K> } = {
  legs,
  support,
  wire,
  rope,
  antenna,
  cable,
  tent,
  flag,
  silo,
  brazier,
  lamp,
  stoplight,
  streetlight,
  radar,
  derrick,
  waterfall,
  awning,
  sign,
  alarm: mount,
  searchlight: mount,
};

/** Draw one of the old hand-drawn props at world (x, its y); kinds not ported yet draw nothing. */
export function drawProp(pen: Pen, p: Prop, x: number): void {
  (DRAWN[p.kind] as Draw<Prop["kind"]> | undefined)?.(pen, p, x, p.y);
}

/** How far right of its x a prop reaches, for culling. */
export const reach = (p: Prop): number => (p.kind === "cable" ? p.x2 - p.x : "w" in p ? p.w : 0);
