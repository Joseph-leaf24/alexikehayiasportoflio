import type { Project } from "@/types/project";

/**
 * Geometry for the project network on the detail page (see ProjectGraph): the
 * project's tools (or its tags, if it lists no tools) feed the technologies it is
 * built on, which feed a small unlabelled hidden layer (wide frames only), which
 * produces the concepts it demonstrates.
 */

/** A neuron, positioned in px. */
export interface GraphNode {
  key: string;
  /** 0 inputs (tools) · 1 technologies · 2 unlabelled hidden layer · 3 outputs (concepts) */
  layer: number;
  /** Column index once empty layers are dropped — drives when the node fires */
  step: number;
  x: number;
  y: number;
  /** Label, wrapped onto one or two lines to fit its column */
  lines?: string[];
  /** The untruncated name */
  full?: string;
  /** True when the name had to be shortened (it is then shown as a tooltip) */
  clipped?: boolean;
  /** Technology (layer 1) nodes are drawn as labelled pills of this size */
  w?: number;
  h?: number;
}

/** A connection between neurons in consecutive columns. */
export interface GraphEdge {
  key: string;
  from: string;
  to: string;
  /** SVG path: an S-curve from `from` to `to` */
  d: string;
  /** Column the signal leaves from */
  step: number;
  /** Whether a signal travels along this edge */
  pulse: boolean;
  /** 0..1, staggers this edge's signal a little */
  jitter: number;
}

/** Fallback size before the frame is measured (and in tests): a 21:9 cover */
export const DEFAULT_SIZE = { w: 1200, h: 514 };
const MAX_PER_LAYER = 7;
/** One forward pass: each layer fires `STEP` seconds after the previous one */
export const STEP = 1.05;
/** Length of the animation loop, s (one step per column of the wide layout) */
export const PERIOD = STEP * 4;
/** Neurons in the unlabelled hidden layer */
const ANON = 5;

/** Small deterministic hash so each project gets its own (stable) firing pattern */
function hash(s: string) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}

const clip = (s: string, max: number) => (s.length > max ? s.slice(0, Math.max(1, max - 1)).trimEnd() + "…" : s);
/** Re-joins split words: spaces between them, except straight after a "/" */
const joinWords = (ws: string[]) => ws.reduce((s, w) => (s === "" || s.endsWith("/") ? s + w : `${s} ${w}`), "");

/**
 * Greedy word-wrap into at most `maxLines` lines of `max` characters. Breaks at spaces
 * and after "/" (so "TensorFlow/Keras" can split); whatever doesn't fit is cut with "…".
 */
export function wrapLabel(text: string, max: number, maxLines: number): string[] {
  const words = text.split(/\s+|(?<=\/)/).filter(Boolean);
  const rows: string[][] = [];
  let row: string[] = [];
  for (const w of words) {
    if (row.length && joinWords([...row, w]).length > max) {
      rows.push(row);
      row = [];
    }
    row.push(w);
  }
  if (row.length) rows.push(row);
  const kept = rows.length > maxLines ? [...rows.slice(0, maxLines - 1), rows.slice(maxLines - 1).flat()] : rows;
  return kept.map((r) => clip(joinWords(r), max));
}

/**
 * Builds the network geometry (in px) for a frame of the given size. Tools, technologies
 * and concepts are capped at 7 each. Narrow frames (< 640px) use a smaller font, wrap
 * labels onto two lines and drop the unlabelled hidden layer.
 */
export function layoutGraph(project: Project, w: number, h: number) {
  const narrow = w < 640;
  const font = narrow ? 9 : 11.5;
  // Mono glyph advance (0.6em) plus the label letter-spacing, with a little slack
  const charW = font * 0.64;
  const lineH = font * 1.25;
  const maxLines = narrow ? 2 : 1;
  const inputs = (project.tools?.length ? project.tools : project.tags).slice(0, MAX_PER_LAYER);
  const hidden = project.tags.slice(0, MAX_PER_LAYER);
  const outputs = project.concepts.slice(0, MAX_PER_LAYER);

  // Column positions (fractions of the width); labels live in the outer gutters
  const cols = narrow ? [0.25, 0.48, 0.48, 0.71] : [0.26, 0.44, 0.585, 0.71];
  const top = narrow ? 44 : 64;
  // Leaves room for the figure caption (two lines on phones)
  const bottom = narrow ? 58 : 54;
  const ys = (n: number) => Array.from({ length: n }, (_, i) => top + ((i + 0.5) * (h - top - bottom)) / n);

  const sideChars = Math.floor((cols[0] * w - 22) / charW);
  const outChars = Math.floor(((1 - cols[3]) * w - 26) / charW);
  const pillChars = Math.floor(((narrow ? 0.24 : 0.15) * w - 18) / charW);
  const label = (full: string, max: number) => {
    const lines = wrapLabel(full, max, maxLines);
    return { lines, full, clipped: lines.some((l) => l.endsWith("…")) };
  };

  const nodes: GraphNode[] = [];
  ys(inputs.length).forEach((y, i) => nodes.push({ key: `0-${i}`, layer: 0, step: 0, x: cols[0] * w, y, ...label(inputs[i], sideChars) }));
  ys(hidden.length).forEach((y, i) => {
    const l = label(hidden[i], pillChars);
    const longest = Math.max(...l.lines.map((s) => s.length));
    nodes.push({ key: `1-${i}`, layer: 1, step: 0, x: cols[1] * w, y, ...l, w: longest * charW + 20, h: 22 + (l.lines.length - 1) * lineH });
  });
  if (!narrow) ys(ANON).forEach((y, i) => nodes.push({ key: `2-${i}`, layer: 2, step: 0, x: cols[2] * w, y }));
  ys(outputs.length).forEach((y, i) => nodes.push({ key: `3-${i}`, layer: 3, step: 0, x: cols[3] * w, y, ...label(outputs[i], outChars) }));

  // Fully connect consecutive layers with smooth S-curves
  const layers = [0, 1, 2, 3].map((l) => nodes.filter((n) => n.layer === l)).filter((l) => l.length);
  layers.forEach((col, step) => col.forEach((n) => (n.step = step)));
  const seed = hash(project.title);
  const edges: GraphEdge[] = [];
  layers.slice(0, -1).forEach((src, li) => {
    const dst = layers[li + 1];
    for (const a of src) {
      for (const b of dst) {
        const x1 = a.x + (a.w ? a.w / 2 : 0);
        const x2 = b.x - (b.w ? b.w / 2 : 0);
        const mx = (x1 + x2) / 2;
        const key = `${a.key}>${b.key}`;
        const r = hash(key + seed);
        edges.push({
          key,
          from: a.key,
          to: b.key,
          d: `M${x1.toFixed(1)} ${a.y.toFixed(1)}C${mx.toFixed(1)} ${a.y.toFixed(1)} ${mx.toFixed(1)} ${b.y.toFixed(1)} ${x2.toFixed(1)} ${b.y.toFixed(1)}`,
          step: li,
          // ~40% of the connections carry a signal on each pass
          pulse: r % 5 < 2,
          jitter: ((r >>> 8) % 100) / 100,
        });
      }
    }
  });

  const heads = {
    y: top - (narrow ? 22 : 30),
    in: cols[0] * w - 16,
    hidden: ((cols[1] + cols[2]) / 2) * w,
    out: cols[3] * w + 16,
    // Short headers on phones so they fit their columns
    text: narrow ? ["TOOLS", "STACK", "CONCEPTS"] : ["IN · TOOLS", "HIDDEN · STACK", "OUT · CONCEPTS"],
  };
  return { nodes, edges, font, lineH, heads, counts: { inputs: inputs.length, hidden: hidden.length, outputs: outputs.length } };
}
