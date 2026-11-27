/**
 * The night glows of the old scene's lighting pass (`PixelScene` drawLights, after the glowing blocks), drawn after
 * each prop: lit lamps as emissive runs (they skip the dark and bloom), soft glows (a cone, a halo, a beam) as
 * additive emissive shapes faded by their tint, and the light they throw around with `pen.light`. Same colours and
 * offsets as the old; a kind is one draw function in `GLOWS`.
 */
import { pixelText } from "@/components/pixel/art";
import { ASSETS, asset } from "@/components/pixel/assets";
import { BUILDINGS } from "@/components/pixel/buildings";
import { hash2, SPRITE_ADDITIVE, SPRITE_EMISSIVE } from "@/engine";
import { alpha, lightState, shape, strip, type Draw, type Pen, type Prop, type Shape } from "./drawn";

const GLOW = SPRITE_EMISSIVE | SPRITE_ADDITIVE;
const rgb = (hex: string): number => parseInt(hex.slice(1), 16);

const ALARM_ON = strip("alarm.on", "x", "#ff3b3b", "#ff3b3b");
const ALARM_OFF = strip("alarm.off", "x", "#5a1a1a", "#5a1a1a");
const LAMP_GLOW = strip("lamp.glow", "x", "#ffd479");
/** A traffic light's lit lamp, by `lightState`: green, amber, red. */
const STOP_LAMPS = ["#5aff6a", "#ffb020", "#ff3b3b"].map((c, st) => strip(`stoplight.${st}`, "x", c, c));

/** A streetlight's cone of light (pose 0) and its pool on the street (pose 1). */
const STREET_GLOW = shape("streetlight.glow", {
  w: 21,
  h: 23,
  ox: 5,
  oy: 22,
  poses: 2,
  paint(c, pose) {
    c.fillStyle = "#ffe8a0";
    if (pose === 1) c.fillRect(-5, -1, 21, 2);
    else for (let k = 0; k < 22; k++) c.fillRect(5 - Math.round(k * 0.4), -22 + k, 1 + Math.round(k * 0.8), 1);
  },
});

/** A searchlight's beam: 150 px long, 0.12 rad either side, from its tip at the pivot along +x (turned when drawn). */
const BEAM = 150;
const BEAM_HALF = Math.ceil(BEAM * Math.tan(0.12));
const SEARCH_BEAM = shape("searchlight.beam", {
  w: BEAM,
  h: BEAM_HALF * 2 + 1,
  ox: 0,
  oy: BEAM_HALF,
  paint(c) {
    c.fillStyle = "#fff4c8";
    for (let k = 0; k < BEAM; k++) {
      const half = Math.round(k * Math.tan(0.12));
      c.fillRect(k, -half, 1, half * 2 + 1);
    }
  },
});

/** How far left of its x a prop's glow reaches, for culling: a searchlight's beam swings out left. */
export const glowLead = (p: Prop): number => (p.kind === "searchlight" ? BEAM : 0);

/**
 * Neon signs: a shape per text and ink a neon building spec gives (the only neon signs), centred on x as the sign is,
 * its halo over the board (pose 0) and its letters (pose 1).
 */
const neonName = (text: string, ink: string) => `neon.${text}.${ink.slice(1)}`;
const NEON = new Map<string, Shape>();
for (const b of BUILDINGS) {
  if (!b.sign || !b.neon) continue;
  const text = b.sign;
  const ink = b.ink ?? "#ffffff";
  const name = neonName(text, ink);
  if (NEON.has(name)) continue;
  const w = text.length * 4 + 3;
  const sx = Math.round(-w / 2);
  NEON.set(
    name,
    shape(name, {
      w: w + 4,
      h: 12,
      ox: 2 - sx,
      oy: 12,
      poses: 2,
      paint(c, pose) {
        c.fillStyle = ink;
        if (pose === 0) c.fillRect(sx - 2, -12, w + 4, 12);
        else pixelText(c, text, sx + 2, -9, ink, false);
      },
    })
  );
}

/** The sprite props' lit spots (`asset(id).lights`): a 1-px strip per colour they use, and its light colour. */
const SPOTS = new Map(
  [...new Set(Object.values(ASSETS).flatMap((a) => a.lights?.map((l) => l[2]) ?? []))].map((c) => [
    c,
    { strip: strip(`spot.${c.slice(1)}`, "x", c), rgb: rgb(c) },
  ])
);

/** An alarm blinks red, lighting up around it while on. */
const alarm: Draw<"alarm"> = (pen, p, x, y) => {
  const on = Math.floor(pen.time * 4 + p.x) % 2 === 0;
  pen.run(on ? ALARM_ON : ALARM_OFF, x - 1, y - 3, 4, SPRITE_EMISSIVE);
  if (on) pen.light(x + 1, y - 2, 20, 0xff3b3b, 0.8);
};

const lamp: Draw<"lamp"> = (pen, _p, x, y) => {
  pen.run(LAMP_GLOW, x, y + 1, 2, SPRITE_EMISSIVE);
  pen.light(x + 1, y + 6, 18, 0xffd479, 0.6);
};

/** After dusk a streetlight throws a cone of light onto a pool on the street, fading in. */
const streetlight: Draw<"streetlight"> = (pen, _p, x, y) => {
  if (pen.night < 0.3) return;
  const a = Math.min(1, (pen.night - 0.3) * 2.5);
  pen.shape(STREET_GLOW, 0, x, y, GLOW, alpha(0.07 * a));
  pen.shape(STREET_GLOW, 1, x, y, GLOW, alpha(0.12 * a));
  pen.light(x + 5, y - 4, 28, 0xffe8a0, 0.8 * a);
};

/** The lit lamp glows, brighter at night. */
const stoplight: Draw<"stoplight"> = (pen, p, x, y) => {
  const st = lightState(pen.time, p.x);
  pen.run(STOP_LAMPS[st], x - 1, y - 26 + (2 - st) * 3, 3, SPRITE_EMISSIVE, alpha(0.5 + pen.night * 0.5));
};

/** A neon sign's halo, flickering now and then; its letters light up after dusk. */
const sign: Draw<"sign"> = (pen, p, x, y) => {
  const s = p.neon ? NEON.get(neonName(p.text, p.ink)) : undefined;
  if (!s) return;
  const on = hash2(Math.floor(pen.time * 8), Math.round(p.x)) > 0.04 ? 1 : 0.2;
  pen.shape(s, 0, x, y, GLOW, alpha((0.1 + pen.night * 0.25) * on));
  if (pen.night > 0.2) pen.shape(s, 1, x, y, SPRITE_EMISSIVE);
};

/** A sprite prop's lit spots pulse at their rates (dimmer behind), lighting up around after dusk or if steady. */
const sprite: Draw<"sprite"> = (pen, p, x, y) => {
  const a = asset(p.id);
  if (!a.lights) return;
  const x0 = Math.round(x - a.w / 2);
  const y0 = y - a.h + 1;
  for (const [lx, ly, col, rate] of a.lights) {
    const spot = SPOTS.get(col);
    if (!spot) continue;
    const on = rate === 0 ? 1 : 0.35 + 0.65 * (0.5 + 0.5 * Math.sin(pen.time * rate * 6 + p.x));
    const gx = x0 + (p.flip ? a.w - 1 - lx : lx);
    pen.run(spot.strip, gx, y0 + ly, 1, SPRITE_EMISSIVE, alpha(on * (p.back ? 0.5 : 1)));
    if (pen.night > 0.2 || rate === 0) pen.light(gx, y0 + ly, 12, spot.rgb, on * (0.5 + pen.night) * 0.5);
  }
};

/** Fires and braziers light up around them, flickering; a brazier more after dusk (any prop: the table checks). */
const fire = (pen: Pen, p: Prop, x: number, y: number) => {
  const brazier = p.kind === "brazier";
  const a = 0.14 + 0.06 * Math.sin(pen.time * 9 + p.x) + (brazier ? pen.night * 0.12 : 0);
  pen.light(x, y - (brazier ? 11 : 4), 20, 0xff8a1a, a * 4);
};

/** After dusk a searchlight's beam sweeps the ground below its tower. */
const searchlight: Draw<"searchlight"> = (pen, p, x, y) => {
  if (pen.night <= 0.2) return;
  const a = Math.PI / 2 + 0.55 + Math.sin(pen.time * 0.7 + p.x) * 0.6;
  pen.shape(SEARCH_BEAM, 0, x + 1, y, GLOW, alpha(0.16 * pen.night), a);
};

const GLOWS: { [K in Prop["kind"]]?: Draw<K> } = {
  alarm,
  lamp,
  streetlight,
  stoplight,
  sign,
  sprite,
  fire,
  brazier: fire,
  searchlight,
};

/** Draw a prop's glow, if its kind has one, at world (x, its y). */
export function drawGlow(pen: Pen, p: Prop, x: number): void {
  (GLOWS[p.kind] as Draw<Prop["kind"]> | undefined)?.(pen, p, x, p.y);
}
