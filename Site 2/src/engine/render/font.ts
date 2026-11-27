import type { Atlas } from "./atlas";
import { makeCanvas } from "./canvas";
import type { SpriteBuffer } from "./types";

/**
 * The 3×5 pixel font, ported from the old engine's `pixelText` (`components/pixel/art.ts`): each glyph is five rows
 * of three bits. The second block adds the symbols the HUD and debug overlay need.
 */
export const GLYPHS: Readonly<Record<string, string>> = {
  "0": "111101101101111",
  "1": "010110010010111",
  "2": "111001111100111",
  "3": "111001111001111",
  "4": "101101111001001",
  "5": "111100111001111",
  "6": "111100111101111",
  "7": "111001010010010",
  "8": "111101111101111",
  "9": "111101111001111",
  A: "010101111101101",
  B: "110101110101110",
  C: "011100100100011",
  D: "110101101101110",
  E: "111100110100111",
  F: "111100110100100",
  G: "011100101101011",
  H: "101101111101101",
  I: "111010010010111",
  J: "001001001101010",
  K: "101101110101101",
  L: "100100100100111",
  M: "101111111101101",
  N: "110101101101101",
  O: "010101101101010",
  P: "110101110100100",
  Q: "010101101110011",
  R: "110101110101101",
  S: "011100010001110",
  T: "111010010010010",
  U: "101101101101111",
  V: "101101101101010",
  W: "101101111111101",
  X: "101101010101101",
  Y: "101101010010010",
  Z: "111001010100111",
  "+": "000010111010000",
  "-": "000000111000000",
  "!": "010010010000010",
  ",": "000000000010100",
  ".": "000000000000010",
  ":": "000010000010000",
  "?": "111001010000010",
  " ": "000000000000000",
  // Added for the engine
  "/": "001001010100100",
  "%": "101001010100101",
  "(": "010100100100010",
  ")": "010001001001010",
  "[": "110100100100110",
  "]": "011001001001011",
  "=": "000111000111000",
  "'": "010010000000000",
  '"': "101101000000000",
  "<": "001010100010001",
  ">": "100010001010100",
  "*": "000101010101000",
  "#": "101111101111101",
  _: "000000000000111",
  "|": "010010010010010",
  ";": "000010000010100",
  "·": "000000010000000",
};

/** The outline colour of the old font. */
export const INK = 0x0d0b14ff;
export const GLYPH_W = 3;
export const GLYPH_H = 5;
/** Glyph advance in font pixels (3 wide plus a 1 px gap). */
export const ADVANCE = 4;

/** Char code (Latin-1) to glyph number; -1 draws nothing but still advances. Lower case maps to upper. */
export const GLYPH_OF = (() => {
  const t = new Int16Array(256).fill(-1);
  Object.keys(GLYPHS).forEach((ch, i) => {
    const c = ch.charCodeAt(0);
    if (c < 256 && ch !== " ") t[c] = i;
  });
  for (let c = 97; c <= 122; c++) t[c] = t[c - 32];
  return t;
})();

/** The atlas frames for each glyph and its outline (the old font's 4-way outline plus a drop shadow). */
export interface FontFrames {
  glyph: Int32Array;
  outline: Int32Array;
}

/** The outline mask: the glyph moved by each of the old offsets, in a 5×7 cell with its pivot at (1, 1). */
export function outlineBits(bits: string): Uint8Array {
  const out = new Uint8Array(5 * 7);
  const offs = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
    [1, 1],
  ];
  for (let b = 0; b < 15; b++) {
    if (bits[b] !== "1") continue;
    const gx = b % 3;
    const gy = (b / 3) | 0;
    for (const [dx, dy] of offs) out[(gy + dy + 1) * 5 + (gx + dx + 1)] = 1;
  }
  return out;
}

/** Paints every glyph (white, for tinting) and its outline into the atlas. */
export function registerFont(atlas: Atlas): FontFrames {
  const keys = Object.keys(GLYPHS);
  const glyph = new Int32Array(keys.length).fill(-1);
  const outline = new Int32Array(keys.length).fill(-1);
  keys.forEach((ch, i) => {
    if (ch === " ") return;
    const bits = GLYPHS[ch];
    const [g, gc] = makeCanvas(GLYPH_W, GLYPH_H);
    gc.fillStyle = "#fff";
    for (let b = 0; b < 15; b++) if (bits[b] === "1") gc.fillRect(b % 3, (b / 3) | 0, 1, 1);
    glyph[i] = atlas.add(`font:${ch}`, g);
    const mask = outlineBits(bits);
    const [o, oc] = makeCanvas(5, 7);
    oc.fillStyle = "#fff";
    for (let k = 0; k < mask.length; k++) if (mask[k]) oc.fillRect(k % 5, (k / 5) | 0, 1, 1);
    outline[i] = atlas.add(`font:outline:${ch}`, o, 1, 1);
  });
  return { glyph, outline };
}

/** Width in pixels of `text` at a whole-pixel scale. */
export function measureText(text: string, scale = 1) {
  return text.length ? (text.length * ADVANCE - 1) * scale : 0;
}

/**
 * Lays one text run out as overlay sprites: all outlines first, then the glyphs tinted `color`, so no outline covers
 * a neighbouring letter. The outline takes the text's alpha. Returns the sprites pushed.
 */
export function layoutText(
  font: FontFrames,
  text: string,
  x: number,
  y: number,
  color: number,
  scale: number,
  align: number,
  layer: number,
  out: SpriteBuffer
): number {
  const s = scale < 1 ? 1 : scale | 0;
  const width = measureText(text, s);
  const x0 = Math.round(align === 1 ? x - width / 2 : align === 2 ? x - width : x);
  const y0 = Math.round(y);
  const alpha = color & 255;
  if (alpha === 0) return 0;
  const ink = ((INK & 0xffffff00) | alpha) >>> 0;
  let pushed = 0;
  for (let pass = 0; pass < 2; pass++) {
    const frames = pass === 0 ? font.outline : font.glyph;
    const tint = pass === 0 ? ink : color;
    for (let i = 0; i < text.length; i++) {
      const c = text.charCodeAt(i);
      const g = c < 256 ? GLYPH_OF[c] : -1;
      if (g < 0) continue;
      out.push(frames[g], x0 + i * ADVANCE * s, y0, layer, 0, tint, s, 0);
      pushed++;
    }
  }
  return pushed;
}
