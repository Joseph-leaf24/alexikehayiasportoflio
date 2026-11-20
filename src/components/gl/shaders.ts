/**
 * GLSL for every material in the 3D scene (village pieces, voxel terrain, sea and water
 * blocks, voxel moon and sun, sky, snow, cursor dust, network edges) plus the final
 * composite pass. All colours are authored directly in display space (no tone mapping).
 *
 * Day and night: the scene sets uDay / uGolden / uSunDir and the key light (uLightDir,
 * uLightCol, uAmbient) from the real time in Amsterdam (src/lib/sun.ts). At full night
 * those uniforms equal the original constants, so the night look is unchanged.
 */
import { WAVE_AMP } from "./layouts";

/** Hash, value-noise and fBm helpers shared by several shaders. */
const NOISE = /* glsl */ `
  float hash13(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
  }
  float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
  }
  float vnoise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
      mix(mix(hash13(i), hash13(i + vec3(1, 0, 0)), f.x), mix(hash13(i + vec3(0, 1, 0)), hash13(i + vec3(1, 1, 0)), f.x), f.y),
      mix(mix(hash13(i + vec3(0, 0, 1)), hash13(i + vec3(1, 0, 1)), f.x), mix(hash13(i + vec3(0, 1, 1)), hash13(i + vec3(1, 1, 1)), f.x), f.y),
      f.z);
  }
  float fbm(vec3 p) {
    float a = 0.5, s = 0.0;
    for (int i = 0; i < 4; i++) { s += a * vnoise(p); p *= 2.03; a *= 0.5; }
    return s;
  }
`;

/** Time of day + key light, written by the scene every frame (see the header comment). */
const DAY = /* glsl */ `
  uniform float uDay;
  uniform float uGolden;
  uniform vec3 uSunDir;
  uniform vec3 uLightDir;
  uniform vec3 uLightCol;
  uniform vec3 uAmbient;
`;

/**
 * Cheap procedural environment for reflections and refraction: a night sky (gradient,
 * horizon glow, two highlights) blended into an Aegean day sky with the sun, plus a warm
 * horizon toward the sun at sunrise / sunset. Needs DAY.
 */
const ENV = /* glsl */ `
  vec3 envNight(vec3 d) {
    float y = d.y * 0.5 + 0.5;
    vec3 c = mix(vec3(0.015, 0.025, 0.045), vec3(0.30, 0.44, 0.60), smoothstep(0.25, 1.0, y));
    c += vec3(0.55, 0.78, 1.0) * pow(1.0 - abs(d.y), 8.0) * 0.45;
    vec3 L = normalize(vec3(-0.4, 0.8, 0.5));
    c += vec3(0.9, 0.97, 1.0) * pow(max(dot(d, L), 0.0), 24.0) * 1.15;
    vec3 L2 = normalize(vec3(0.7, 0.2, -0.6));
    c += vec3(0.35, 0.75, 1.0) * pow(max(dot(d, L2), 0.0), 10.0) * 0.6;
    return c;
  }
  vec3 envDay(vec3 d) {
    vec3 c = mix(vec3(0.42, 0.56, 0.66), vec3(0.14, 0.32, 0.62), smoothstep(0.0, 0.8, d.y));
    c = mix(vec3(0.04, 0.14, 0.2), c, smoothstep(-0.25, 0.02, d.y));
    c += vec3(1.0, 0.93, 0.78) * pow(max(dot(d, uSunDir), 0.0), 48.0) * 1.3;
    return c;
  }
  vec3 envColor(vec3 d) {
    // (uniform branches: only twilight pays for both skies)
    vec3 c;
    if (uDay < 0.001) c = envNight(d);
    else if (uDay > 0.999) c = envDay(d);
    else c = mix(envNight(d), envDay(d), uDay);
    if (uGolden > 0.001) {
      vec3 sunH = normalize(vec3(uSunDir.x, 0.08, uSunDir.z));
      c += vec3(1.0, 0.5, 0.28) * uGolden * pow(max(dot(d, sunH), 0.0), 5.0) * 0.55;
    }
    return c;
  }
`;

/**
 * Sea surface colour at w with normal N and view V: deep water, the sky reflected by
 * Fresnel, and a glittering path toward the moon at night or the sun by day. Shared by the
 * flat distant sea and the water blocks so they always match. Needs DAY + NOISE.
 */
const SEA = /* glsl */ `
  uniform vec3 uMoonDir;
  vec3 seaColor(vec3 w, vec3 N, vec3 V, float t) {
    vec3 R = reflect(-V, N);
    float fres = 0.03 + 0.97 * pow(1.0 - max(dot(N, V), 0.0), 5.0);
    vec3 skyNight = mix(vec3(0.015, 0.025, 0.045), vec3(0.12, 0.18, 0.26), smoothstep(-0.1, 0.5, R.y));
    vec3 skyDay = mix(vec3(0.3, 0.42, 0.52), vec3(0.1, 0.24, 0.46), smoothstep(0.0, 0.6, R.y));
    vec3 skyR = mix(skyNight, skyDay, uDay);
    float n = vnoise(vec3(w.xz * 7.0, t * 3.0));
    vec3 glint = vec3(0.0);
    if (uDay < 0.999) {
      float md = max(dot(R, uMoonDir), 0.0);
      glint += vec3(0.85, 0.92, 1.0) * (pow(md, 400.0) * 2.4 * step(0.62, n) + pow(md, 50.0) * 0.14) * (1.0 - uDay);
    }
    // The sun's glitter path: a narrow, sparser band than the moon's, so the water stays blue
    if (uSunDir.y > 0.0 && max(uDay, uGolden) > 0.001) {
      float sd = max(dot(R, uSunDir), 0.0);
      vec3 sunCol = mix(vec3(1.0, 0.95, 0.84), vec3(1.0, 0.62, 0.34), uGolden);
      glint += sunCol * (pow(sd, 700.0) * 2.0 * step(0.7, n) + pow(sd, 90.0) * 0.12) * max(uDay, uGolden);
    }
    vec3 base = mix(vec3(0.006, 0.014, 0.026), vec3(0.012, 0.13, 0.25), uDay);
    base *= mix(1.0, 0.75 + 0.5 * max(dot(N, uLightDir), 0.0), uDay);
    base = mix(base, base * vec3(1.35, 1.0, 0.8), uGolden * 0.6);
    return base + skyR * fres * 0.8 + glint;
  }
`;

/**
 * Hold-to-shatter for instanced cubes (terrain, water blocks): every cube tears loose and
 * tumbles up and out, in a wave spreading from uBreakC (the village), less so far away;
 * uBreak eases it out on press and back on release.
 */
const BREAK = /* glsl */ `
  uniform float uBreak;
  uniform vec3 uBreakC;
  mat3 rotAxis(vec3 a, float ang) {
    a = normalize(a);
    float s = sin(ang);
    float c = cos(ang);
    float oc = 1.0 - c;
    return mat3(oc * a.x * a.x + c, oc * a.x * a.y + a.z * s, oc * a.z * a.x - a.y * s,
                oc * a.x * a.y - a.z * s, oc * a.y * a.y + c, oc * a.y * a.z + a.x * s,
                oc * a.z * a.x + a.y * s, oc * a.y * a.z - a.x * s, oc * a.z * a.z + c);
  }
  // amt scales the whole effect (1 for the hills; the water blocks only splash)
  vec3 shatter(vec3 c, float rnd, float depth, float amt, out mat3 R, out float e) {
    vec3 rel = c - uBreakC;
    float rb = length(rel.xz);
    e = smoothstep(rb * 0.02, rb * 0.02 + 0.5, uBreak) * smoothstep(42.0, 6.0, rb) * amt;
    vec3 dir = normalize(vec3(rel.x * 0.08, 0.0, rel.z * 0.08) + vec3(rnd - 0.5, 1.0 + rnd, fract(rnd * 7.13) - 0.5));
    vec3 axis = vec3(fract(rnd * 3.1) - 0.5, fract(rnd * 5.7) - 0.5, fract(rnd * 9.3) - 0.5) + 0.001;
    R = rotAxis(axis, e * (1.5 + rnd * 5.0));
    return dir * e * (1.5 + rnd * 6.5) * (1.0 + depth * 0.15);
  }
`;

/* ------------------------------------------------------------------ */
/* Village pieces: whitewashed houses + blue domes that turn into ice  */
/* ------------------------------------------------------------------ */
export const pieceVertex = /* glsl */ `
  attribute float aRand;
  attribute float aMix;
  attribute vec4 aTint;
  varying vec3 vN;
  varying vec3 vW;
  varying vec3 vL;
  varying vec3 vLN;
  varying vec3 vS;
  varying float vR;
  varying float vMix;
  varying vec4 vTint;
  void main() {
    vTint = aTint;
    vL = position;
    vLN = normal;
    vS = vec3(length(instanceMatrix[0].xyz), length(instanceMatrix[1].xyz), length(instanceMatrix[2].xyz));
    mat4 m = modelMatrix * instanceMatrix;
    vec4 w = m * vec4(position, 1.0);
    vW = w.xyz;
    vN = normalize(mat3(m) * (normal / max(vS * vS, vec3(1e-4))));
    vR = aRand;
    vMix = aMix;
    gl_Position = projectionMatrix * viewMatrix * w;
  }
`;

export const pieceFragment = /* glsl */ `
  uniform vec3 uCam;
  uniform float uTime;
  uniform vec3 uMouse;
  uniform float uMouseOn;
  uniform float uPulseT;
  uniform vec3 uPulseC;
  uniform float uExposure;
  uniform vec3 uFog;
  uniform float uDome;
  uniform float uNet;
  uniform vec3 uNetC;
  uniform float uNetRot;
  uniform float uHi;
  uniform vec3 uHiC;
  varying vec3 vN;
  varying vec3 vW;
  varying vec3 vL;
  varying vec3 vLN;
  varying vec3 vS;
  varying float vR;
  varying float vMix;
  varying vec4 vTint;
  ${DAY}
  ${NOISE}
  ${ENV}
  void main() {
    vec3 N = normalize(vN);
    if (!gl_FrontFacing) N = -N;
    vec3 V = normalize(uCam - vW);
    vec3 L = uLightDir;
    float ndl = max(dot(N, L), 0.0);
    float fres = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 3.0);

    // World-space distance to the box edges (used for the frosted edge light)
    vec3 dE = (0.5 - abs(vL)) * vS;
    float dmax = max(dE.x, max(dE.y, dE.z));
    float dmin = min(dE.x, min(dE.y, dE.z));
    float dmid = dE.x + dE.y + dE.z - dmax - dmin;
    float edgeW = (1.0 - smoothstep(0.0, 0.04, dmid)) * (1.0 - uDome);

    vec3 village = vec3(0.0);
    if (vMix < 0.99) {
      if (uDome > 0.5) {
        vec3 blue = mix(vec3(0.04, 0.2, 0.6), vec3(0.05, 0.26, 0.72), uDay);
        village = blue * (0.22 * uAmbient + 0.8 * ndl * uLightCol) + envColor(reflect(-V, N)) * (0.12 + fres * 0.55);
        village += pow(max(dot(N, normalize(L + V)), 0.0), 70.0) * 0.7;
        village += vec3(0.35, 0.75, 1.0) * fres * 0.3;
      } else {
        vec3 albedo = vTint.rgb - vnoise(vW * 11.0) * 0.07;
        village = albedo * (uAmbient * (0.08 + 0.16 * (0.5 + 0.5 * N.y) + 0.16 * step(0.5, vLN.y)) + uLightCol * 0.7 * ndl);
        village *= mix(0.5, 1.0, smoothstep(0.25, 1.0, (vL.y + 0.5) * vS.y));
        village += vec3(0.55, 0.8, 1.0) * edgeW * 0.2 * (1.0 - 0.7 * uDay);
        vec3 an = abs(vLN);
        if (an.y < 0.5 && vTint.w > 0.5) {
          // Windows with blue shutters on the walls; some are lit at night
          vec3 lp = vL * vS;
          bool xf = an.x > 0.5;
          float u = xf ? lp.z : lp.x;
          float hu = (xf ? vS.z : vS.x) * 0.5;
          float hv = vS.y * 0.5;
          vec2 g = vec2(u, lp.y) / vec2(0.34, 0.4);
          vec2 cell = floor(g);
          vec2 f = fract(g) - 0.5;
          float face = xf ? sign(vLN.x) : 2.0 * sign(vLN.z);
          float h = hash12(cell + vec2(face * 17.0, vR * 113.0));
          float has = step(0.45, h) * step(abs(u), hu - 0.13) * step(lp.y, hv - 0.15);
          float frame = step(abs(f.x), 0.24) * step(abs(f.y), 0.3);
          float glass = step(abs(f.x), 0.16) * step(abs(f.y), 0.22);
          float lit = step(0.66, hash12(cell * 1.37 + vec2(face, vR * 51.0)));
          float warm = step(0.8, hash12(cell * 2.1 + vec2(vR * 7.0, face)));
          vec3 glow = mix(vec3(0.5, 0.82, 1.0), vec3(1.0, 0.7, 0.4), warm) * (1.15 + 0.15 * sin(uTime * 0.8 + h * 40.0));
          vec3 shutter = vec3(0.06, 0.2, 0.52) * (0.55 + 0.65 * ndl);
          village = mix(village, shutter, has * frame * (1.0 - glass));
          // By day the glass is dark and reflects the sky instead of glowing
          vec3 pane = mix(vec3(0.015, 0.03, 0.06), vec3(0.05, 0.1, 0.17) + envColor(reflect(-V, N)) * 0.18, uDay);
          village = mix(village, mix(pane, glow, lit * (1.0 - uDay)), has * glass);
        }
      }
    }

    vec3 ice = vec3(0.0);
    float edge = 0.0;
    if (vMix > 0.01) {
      vec3 a = abs(vL) * 2.0;
      float mx = max(a.x, max(a.y, a.z));
      float mn = min(a.x, min(a.y, a.z));
      edge = smoothstep(0.8, 0.985, a.x + a.y + a.z - mx - mn) * (1.0 - uDome);
      vec3 q = vL * vec3(1.6, 0.95, 0.88) + vR * 17.0;
      float frost = fbm(q * 2.0);
      float crack = 1.0 - smoothstep(0.0, 0.03, abs(vnoise(q * 3.1) - 0.5));
      float bubbles = smoothstep(0.82, 0.9, vnoise(q * 9.0));
      vec3 refl = envColor(reflect(-V, normalize(N + (frost - 0.5) * 0.35)));
      vec3 refr = envColor(normalize(refract(-V, N, 0.76) + (frost - 0.5) * 0.7));
      vec3 body = mix(vec3(0.035, 0.085, 0.14), refr * vec3(0.5, 0.78, 1.0), 0.55);
      body += vec3(0.55, 0.75, 0.95) * frost * 0.2 + vec3(0.75, 0.9, 1.0) * (crack * 0.22 + bubbles * 0.15);
      ice = mix(body, refl, fres * 0.8) + vec3(0.7, 0.9, 1.0) * edge * (0.5 + 0.5 * fres);
      vec3 Nf = normalize(N + (vec3(vnoise(q * 7.0), vnoise(q * 7.0 + 31.0), vnoise(q * 7.0 + 57.0)) - 0.5) * 0.55);
      ice += pow(max(dot(Nf, normalize(L + V)), 0.0), 140.0) * 1.1;
      ice *= 0.93 + 0.07 * sin(uTime * 1.3 + vR * 40.0);
    }

    vec3 col = mix(village, ice, vMix);
    col += vec3(0.4, 0.78, 1.0) * uMouseOn * smoothstep(2.2, 0.0, distance(vW, uMouse)) * (0.15 + edge * 0.9 + edgeW * 0.5);
    float pt = uTime - uPulseT;
    float ring = exp(-pow(length(vW - uPulseC) - pt * 11.0, 2.0) * 1.2) * smoothstep(2.0, 0.0, pt);
    col += vec3(0.5, 0.85, 1.0) * ring * (0.4 + edge + edgeW);
    // The cluster of the hovered domain card glows
    col += vec3(0.45, 0.85, 1.0) * uHi * smoothstep(2.0, 0.4, distance(vW, uHiC)) * (0.35 + edge * 0.9);
    // Neural net: activations sweep layer by layer (forward propagation)
    if (uNet > 0.001) {
      vec3 dn = vW - uNetC;
      float lx = dn.x * cos(uNetRot) - dn.z * sin(uNetRot);
      float fire = pow(0.5 + 0.5 * sin(lx * 1.1 - uTime * 3.2), 14.0);
      col += vec3(0.45, 0.85, 1.0) * fire * uNet * (0.22 + edge * 0.7);
    }
    col = mix(col, uFog, smoothstep(26.0, 70.0, distance(vW, uCam)));
    gl_FragColor = vec4(col * uExposure, 1.0);
  }
`;

/** Anti-aliased grid lines (width in screen pixels, via fwidth). */
const LINES = /* glsl */ `
  float gridLine(float x, float w) {
    float fw = max(fwidth(x), 1e-4);
    return 1.0 - smoothstep(0.0, fw * w, abs(fract(x - 0.5) - 0.5));
  }
`;

/* ------------------------------------------------------------------ */
/* Voxel hills: stacked ice-rock cubes <-> pulsing holographic equaliser */
/* ------------------------------------------------------------------ */
export const terrainVertex = /* glsl */ `
  attribute float aRand;
  attribute float aDepth;
  uniform float uTime;
  uniform float uLand;
  uniform float uGrow;
  uniform float uAmp;
  uniform vec3 uCenter;
  uniform vec3 uMouse;
  uniform float uMouseOn;
  uniform float uShockT;
  uniform float uPulseT;
  varying vec3 vW;
  varying vec3 vL;
  varying vec3 vLN;
  varying vec3 vN;
  varying float vR;
  varying float vLift;
  varying float vDepth;
  varying float vSize;
  ${NOISE}
  ${BREAK}
  void main() {
    vL = position;
    vLN = normal;
    float size = length(instanceMatrix[0].xyz);
    vec3 c = instanceMatrix[3].xyz;
    float r = length(c.xz - uCenter.xz);
    float holo = 1.0 - uLand;
    // Equaliser: whole columns pump with a travelling wave once the land goes holographic
    float eq = (sin(r * 0.55 - uTime * 2.2) * 0.5 + 0.5) * 0.9 + vnoise(vec3(c.xz * 0.22, uTime * 0.6)) * 1.1;
    float lift = holo * eq * uAmp * 0.6;
    lift += holo * smoothstep(3.0, 0.0, length(c.xz - uMouse.xz)) * 1.4 * uMouseOn;
    // Shockwave (hold-to-shatter) and ripple (UI pulse, e.g. hovering a card)
    float st = uTime - uShockT;
    lift += exp(-pow(r - st * 9.0, 2.0) * 0.35) * smoothstep(3.0, 0.0, st) * 1.4;
    float pt = uTime - uPulseT;
    lift += exp(-pow(r - pt * 12.0, 2.0) * 0.5) * smoothstep(1.8, 0.0, pt) * 0.5;
    // Intro: the hills rise out of the sea in a ripple from the village
    float grow = smoothstep(r * 0.025, r * 0.025 + 0.45, uGrow);
    lift -= (1.0 - grow) * (c.y + 3.0);

    // Hold-to-shatter: every cube tears loose and tumbles (see BREAK)
    mat3 R;
    float e;
    vec3 fly = shatter(c, aRand, aDepth, 1.0, R, e);
    vec3 w = c + R * (position * size * (1.0 - e * 0.25)) + fly;
    w.y += lift;

    vW = w;
    vN = normalize(R * normal);
    vR = aRand;
    vLift = lift;
    vDepth = aDepth;
    vSize = size;
    gl_Position = projectionMatrix * viewMatrix * vec4(w, 1.0);
  }
`;

export const terrainFragment = /* glsl */ `
  uniform float uTime;
  uniform float uLand;
  uniform float uExposure;
  uniform vec3 uCam;
  uniform vec3 uFog;
  varying vec3 vW;
  varying vec3 vL;
  varying vec3 vLN;
  varying vec3 vN;
  varying float vR;
  varying float vLift;
  varying float vDepth;
  varying float vSize;
  ${DAY}
  ${NOISE}
  ${ENV}
  void main() {
    vec3 N = normalize(vN);
    vec3 V = normalize(uCam - vW);
    vec3 L = uLightDir;
    float ndl = max(dot(N, L), 0.0);
    float fres = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 3.0);

    // Glowing cube edges
    vec3 dE = (0.5 - abs(vL)) * vSize;
    float dmax = max(dE.x, max(dE.y, dE.z));
    float dmin = min(dE.x, min(dE.y, dE.z));
    float edge = 1.0 - smoothstep(0.0, 0.026, dE.x + dE.y + dE.z - dmax - dmin);
    float depth = vDepth + (0.5 - vL.y) * vSize;
    float topF = step(0.5, vLN.y);
    float h = hash12(vec2(vR * 97.0, vDepth));

    // Glacial ice cubes with frosted tops: dark and moonlit at night, pale blue-grey by day
    vec3 deep = mix(mix(vec3(0.03, 0.05, 0.08), vec3(0.06, 0.09, 0.13), h), mix(vec3(0.11, 0.15, 0.2), vec3(0.16, 0.2, 0.25), h), uDay);
    vec3 real = deep * (0.55 * uAmbient + 1.2 * ndl * uLightCol) + envColor(reflect(-V, N)) * (0.05 + fres * 0.32);
    real += mix(vec3(0.16, 0.2, 0.26), vec3(0.34, 0.39, 0.45), uDay) * topF * (0.35 + 0.65 * ndl) * (0.6 + 0.4 * vnoise(vW * 5.0));
    real += vec3(0.55, 0.8, 1.0) * edge * (0.12 + 0.22 * fres) * (1.0 - 0.6 * uDay);
    real *= mix(0.4, 1.0, exp(-depth * 0.55));

    // Holographic equaliser
    float sweep = exp(-pow((fract(vW.z * 0.045 + uTime * 0.07) - 0.5) * 26.0, 2.0));
    vec3 holo = vec3(0.005, 0.01, 0.02) + vec3(0.3, 0.72, 1.0) * edge * (0.34 + sweep * 1.2);
    holo += vec3(0.25, 0.6, 1.0) * topF * clamp(vLift * 0.3, 0.0, 0.9);
    holo *= mix(0.25, 1.0, exp(-depth * 0.35));

    vec3 col = mix(holo, real, uLand);
    col = mix(col, uFog, smoothstep(24.0, 75.0, distance(vW, uCam)));
    gl_FragColor = vec4(col * uExposure, 1.0);
  }
`;

/* ------------------------------------------------------------------ */
/* Distant Aegean sea (flat) with a moon / sun path                    */
/* ------------------------------------------------------------------ */
export const seaVertex = /* glsl */ `
  varying vec3 vW;
  void main() {
    vec4 w = modelMatrix * vec4(position, 1.0);
    vW = w.xyz;
    gl_Position = projectionMatrix * viewMatrix * w;
  }
`;

export const seaFragment = /* glsl */ `
  uniform float uTime;
  uniform float uLand;
  uniform float uExposure;
  uniform vec3 uCam;
  uniform vec3 uFog;
  uniform float uShockT;
  varying vec3 vW;
  ${DAY}
  ${NOISE}
  ${SEA}
  ${LINES}
  void main() {
    vec2 p = vW.xz;
    float t = uTime * 0.35;
    vec2 wv = vec2(vnoise(vec3(p * 0.6, t)) - 0.5, vnoise(vec3(p * 0.6 + 11.0, t)) - 0.5) * 0.35
            + vec2(sin(p.x * 1.7 + t * 3.0), cos(p.y * 2.3 - t * 2.4)) * 0.05;
    // Hold-to-shatter: a ripple ring runs out across the sea from the village
    vec2 rel = p - vec2(0.0, -4.0);
    float rr = length(rel);
    float st = uTime - uShockT;
    wv += rel / max(rr, 1e-3) * sin(rr * 2.2 - st * 12.0) * exp(-pow(rr - st * 9.0, 2.0) * 0.05) * smoothstep(4.0, 0.0, st) * 0.3;
    vec3 N = normalize(vec3(wv.x, 1.0, wv.y));
    vec3 V = normalize(uCam - vW);
    float dist = distance(vW, uCam);
    vec3 real = seaColor(vW, N, V, t);
    float grid = max(gridLine(p.x * 0.5, 1.0), gridLine(p.y * 0.5, 1.0)) * smoothstep(60.0, 10.0, dist);
    vec3 holo = vec3(0.004, 0.008, 0.014) + vec3(0.3, 0.7, 1.0) * grid * 0.12;
    vec3 col = mix(holo, real, uLand);
    col = mix(col, uFog, smoothstep(30.0, 120.0, dist));
    gl_FragColor = vec4(col * uExposure, 1.0);
  }
`;

/* ------------------------------------------------------------------ */
/* Near sea: water blocks riding the swell                             */
/* ------------------------------------------------------------------ */
export const waterVertex = /* glsl */ `
  attribute float aRand;
  uniform float uTime;
  uniform float uLand;
  uniform vec3 uCam;
  varying vec3 vW;
  varying vec3 vN;
  varying vec3 vL;
  varying float vR;
  ${BREAK}
  // GLSL twin of waveHeight() in layouts.ts
  float waveHeight(vec2 p, float t) {
    return ${WAVE_AMP.toFixed(3)} * (0.5 * sin(0.55 * p.x + 0.9 * t) + 0.3 * sin(0.8 * p.y - 1.3 * t + 1.7) + 0.2 * sin(0.35 * (p.x + p.y) + 0.6 * t));
  }
  void main() {
    vec3 c = instanceMatrix[3].xyz;
    // The block's top sits on the swell; in the holographic formations the blocks sink under the flat sea
    c.y += waveHeight(c.xz, uTime) - 0.15 - (1.0 - uLand) * 0.9;
    mat3 R;
    float e;
    // On hold-to-shatter the water splashes: blocks jump and roll near the surface rather
    // than flying up and tumbling with the land, and stay down close to the camera (they'd
    // fill the lens)
    vec3 fly = shatter(c, aRand, 0.0, 0.45 * smoothstep(4.0, 14.0, length(c.xz - uCam.xz)), R, e) * 0.8;
    vec3 w = c + R * (position * (1.0 - e * 0.25)) + fly;
    vW = w;
    vN = normalize(R * normal);
    vL = position;
    vR = aRand;
    gl_Position = projectionMatrix * viewMatrix * vec4(w, 1.0);
  }
`;

export const waterFragment = /* glsl */ `
  uniform float uTime;
  uniform float uExposure;
  uniform vec3 uCam;
  uniform vec3 uFog;
  uniform vec3 uHalf;
  varying vec3 vW;
  varying vec3 vN;
  varying vec3 vL;
  varying float vR;
  ${DAY}
  ${NOISE}
  ${SEA}
  void main() {
    vec3 N = normalize(vN);
    vec3 V = normalize(uCam - vW);
    float t = uTime * 0.35;
    // Tops get a small surface ripple; the sides stay flat and a little darker
    float top = step(0.5, N.y);
    vec2 wv = vec2(vnoise(vec3(vW.xz * 0.9, t)) - 0.5, vnoise(vec3(vW.xz * 0.9 + 11.0, t)) - 0.5) * 0.22;
    N = normalize(N + vec3(wv.x, 0.0, wv.y) * top);
    vec3 col = seaColor(vW, N, V, t) * mix(0.7, 1.0, top);
    // Thin bright block edges (the voxel look)
    vec3 a = abs(vL) / uHalf;
    float mx = max(a.x, max(a.y, a.z));
    float mn = min(a.x, min(a.y, a.z));
    float edge = smoothstep(0.86, 0.98, a.x + a.y + a.z - mx - mn);
    col += vec3(0.55, 0.82, 1.0) * edge * mix(0.22, 0.12, uDay);
    col = mix(col, uFog, smoothstep(30.0, 120.0, distance(vW, uCam)));
    gl_FragColor = vec4(col * uExposure, 1.0);
  }
`;

/* ------------------------------------------------------------------ */
/* Voxel moon: a shell of glowing ice cubes that bursts into a ring    */
/* ------------------------------------------------------------------ */
export const moonVertex = /* glsl */ `
  attribute float aRand;
  attribute float aCrater;
  uniform vec3 uCenter;
  uniform float uTime;
  uniform float uBurst;
  varying vec3 vN;
  varying vec3 vL;
  varying float vR;
  varying float vCrater;
  varying float vBurst;
  mat3 rotY(float a) { float c = cos(a); float s = sin(a); return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c); }
  mat3 rotX(float a) { float c = cos(a); float s = sin(a); return mat3(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c); }
  void main() {
    vec3 o = instanceMatrix[3].xyz;
    float e = smoothstep(aRand * 0.4, aRand * 0.4 + 0.6, uBurst);
    float ang = atan(o.z, o.x) + uTime * (0.05 + aRand * 0.06) + e * 1.6;
    float rr = 10.0 + aRand * 7.0;
    vec3 ring = vec3(cos(ang) * rr, (fract(aRand * 7.31) - 0.5) * 1.6, sin(ang) * rr);
    ring = rotX(0.45) * ring;
    vec3 pos = mix(o * (1.0 - aCrater * 0.07), ring, e);
    pos = rotY(uTime * 0.025) * pos;
    mat3 tumble = rotY(e * aRand * 7.0) * rotX(e * fract(aRand * 3.7) * 5.0);
    vec3 w = uCenter + pos + tumble * position * mix(1.0, 0.55, e);
    vL = position;
    vN = normalize(tumble * normal);
    vR = aRand;
    vCrater = aCrater;
    vBurst = e;
    gl_Position = projectionMatrix * viewMatrix * vec4(w, 1.0);
  }
`;

export const moonFragment = /* glsl */ `
  uniform float uTime;
  uniform float uExposure;
  varying vec3 vN;
  varying vec3 vL;
  varying float vR;
  varying float vCrater;
  varying float vBurst;
  ${NOISE}
  void main() {
    vec3 N = normalize(vN);
    vec3 a = abs(vL) / 0.45;
    float mx = max(a.x, max(a.y, a.z));
    float mn = min(a.x, min(a.y, a.z));
    float edge = smoothstep(0.8, 0.97, a.x + a.y + a.z - mx - mn);
    float facing = 0.72 + 0.28 * max(dot(N, normalize(vec3(-0.5, 0.35, 1.0))), 0.0);
    vec3 base = mix(vec3(0.8, 0.87, 0.97), vec3(0.36, 0.44, 0.58), vCrater);
    base *= 0.86 + 0.28 * hash13(vec3(vR * 91.0, vR * 17.0, 3.0));
    vec3 col = base * facing + vec3(0.6, 0.86, 1.0) * edge * 0.45;
    col *= 0.93 + 0.07 * sin(uTime * 2.0 + vR * 50.0);
    col = mix(col, col * vec3(0.55, 0.85, 1.25), vBurst * 0.6);
    gl_FragColor = vec4(col * uExposure, 1.0);
  }
`;

/* ------------------------------------------------------------------ */
/* Voxel sun: a glowing shell of cubes that bursts into a ring (as the moon does) */
/* ------------------------------------------------------------------ */
export const sunVertex = /* glsl */ `
  attribute float aRand;
  uniform vec3 uCenter;
  uniform float uTime;
  uniform float uBurst;
  varying vec3 vL;
  varying float vR;
  mat3 rotY(float a) { float c = cos(a); float s = sin(a); return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c); }
  mat3 rotX(float a) { float c = cos(a); float s = sin(a); return mat3(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c); }
  void main() {
    vec3 o = instanceMatrix[3].xyz;
    // Cubes breathe in and out a little, like a boiling surface
    o *= 1.0 + 0.04 * sin(uTime * 1.3 + aRand * 40.0);
    // Hold-to-shatter and the holographic formations: the shell flies apart into a tumbling ring
    float e = smoothstep(aRand * 0.4, aRand * 0.4 + 0.6, uBurst);
    float ang = atan(o.z, o.x) + uTime * (0.05 + aRand * 0.06) + e * 1.6;
    float rr = 8.0 + aRand * 6.0;
    vec3 ring = rotX(0.45) * vec3(cos(ang) * rr, (fract(aRand * 7.31) - 0.5) * 1.4, sin(ang) * rr);
    vec3 pos = rotY(uTime * 0.03) * mix(o, ring, e);
    mat3 tumble = rotY(e * aRand * 7.0) * rotX(e * fract(aRand * 3.7) * 5.0);
    vL = position;
    vR = aRand;
    gl_Position = projectionMatrix * viewMatrix * vec4(uCenter + pos + tumble * position * mix(1.0, 0.6, e), 1.0);
  }
`;

export const sunFragment = /* glsl */ `
  uniform float uTime;
  uniform float uExposure;
  uniform float uGolden;
  uniform float uHalf;
  varying vec3 vL;
  varying float vR;
  ${NOISE}
  void main() {
    vec3 a = abs(vL) / uHalf;
    float mx = max(a.x, max(a.y, a.z));
    float mn = min(a.x, min(a.y, a.z));
    float edge = smoothstep(0.82, 0.97, a.x + a.y + a.z - mx - mn);
    vec3 hot = mix(vec3(1.0, 0.86, 0.55), vec3(1.0, 0.58, 0.3), uGolden);
    vec3 col = hot * (1.05 + 0.25 * hash13(vec3(vR * 91.0, 3.0, vR * 17.0))) + vec3(1.0, 0.95, 0.8) * edge * 0.5;
    col *= 0.94 + 0.06 * sin(uTime * 2.4 + vR * 60.0);
    gl_FragColor = vec4(col * uExposure, 1.0);
  }
`;

/* ------------------------------------------------------------------ */
/* Sky dome, built from cubes: night (aurora + stars) ↔ day ↔ sunset   */
/* ------------------------------------------------------------------ */
export const skyVertex = /* glsl */ `
  varying vec3 vDir;
  void main() {
    vDir = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

/** Angular size of one sky cube (radians, ~1.6°: about 35 px on screen). */
const SKY_CELL = 0.028;

export const skyFragment = /* glsl */ `
  uniform float uTime;
  uniform float uExposure;
  uniform vec3 uMoonDir;
  varying vec3 vDir;
  ${DAY}
  ${NOISE}

  /** The smooth sky colour toward d (stars apart): night, day and sunset, moon halo, sun glow. */
  vec3 skyAt(vec3 d) {
    float y = d.y;
    // Night: deep blue-black with a faint horizon glow and aurora (skipped in full daylight,
    // where it's invisible: the aurora's noise is the sky's most expensive part)
    vec3 col = vec3(0.0);
    if (uDay < 0.999) {
      col = mix(vec3(0.01, 0.014, 0.022), vec3(0.028, 0.045, 0.075), smoothstep(-0.2, 0.7, y));
      col += vec3(0.07, 0.11, 0.16) * exp(-abs(y - 0.02) * 9.0);
      float a = fbm(vec3(d.x * 2.2 + uTime * 0.018, d.y * 5.0, d.z * 2.2 - uTime * 0.012));
      float band = smoothstep(0.5, 0.85, a) * smoothstep(0.05, 0.4, y) * (1.0 - smoothstep(0.65, 1.0, y));
      col += mix(vec3(0.1, 0.42, 0.5), vec3(0.25, 0.3, 0.7), d.x * 0.5 + 0.5) * band * 0.32;
      // Soft halo around the (voxel) moon
      float md = max(dot(d, uMoonDir), 0.0);
      col += vec3(0.3, 0.45, 0.68) * pow(md, 160.0) * 0.3 + vec3(0.08, 0.12, 0.2) * pow(md, 14.0) * 0.3;
    }
    // Day: Aegean blue overhead, hazy toward the horizon (deep enough for the white text on top)
    if (uDay > 0.001) {
      vec3 day = mix(vec3(0.3, 0.42, 0.52), vec3(0.07, 0.19, 0.42), smoothstep(-0.02, 0.7, y));
      col = mix(col, day, uDay);
    }
    // Sunrise / sunset: a warm band along the horizon, strongest toward the sun
    if (uGolden > 0.001) {
      vec3 hz = normalize(vec3(d.x, 0.0, d.z) + vec3(1e-4, 0.0, 0.0));
      vec3 sunHz = normalize(vec3(uSunDir.x, 0.0, uSunDir.z) + vec3(1e-4, 0.0, 0.0));
      float toward = pow(max(dot(hz, sunHz), 0.0), 3.0);
      float horizon = exp(-max(y, 0.0) * 5.0) * smoothstep(-0.2, 0.0, y);
      col = mix(col, mix(vec3(0.72, 0.36, 0.38), vec3(1.0, 0.56, 0.26), toward), uGolden * horizon * (0.35 + 0.55 * toward));
    }
    // Glow around the sun (the disc itself is the voxel sun)
    float sd = max(dot(d, uSunDir), 0.0);
    vec3 sunCol = mix(vec3(1.0, 0.94, 0.8), vec3(1.0, 0.62, 0.32), uGolden);
    col += sunCol * (pow(sd, 60.0) * 0.35 + pow(sd, 8.0) * 0.1) * max(uDay, uGolden) * smoothstep(-0.1, 0.02, uSunDir.y);
    return col;
  }

  /** Turn c's hue by a radians (YIQ), keeping its brightness; sat scales the colourfulness. */
  vec3 hueTurn(vec3 c, float a, float sat) {
    vec3 yiq = mat3(0.299, 0.596, 0.211, 0.587, -0.274, -0.523, 0.114, -0.322, 0.312) * c;
    float h = atan(yiq.z, yiq.y) + a;
    float chroma = length(yiq.yz) * sat;
    return max(mat3(1.0, 1.0, 1.0, 0.956, -0.272, -1.106, 0.621, -0.647, 1.703) * vec3(yiq.x, chroma * cos(h), chroma * sin(h)), 0.0);
  }

  void main() {
    vec3 d = normalize(vDir);

    // The sky as a wall of cubes: a grid of square cells by compass bearing and height, each
    // one a single colour (the sky toward its centre), so the aurora, the sunset band and the
    // sun and moon glows all step from cube to cube
    vec2 g = vec2(atan(d.x, -d.z), asin(clamp(d.y, -1.0, 1.0))) / ${SKY_CELL.toFixed(3)};
    vec2 cell = floor(g);
    vec2 f = fract(g);
    vec2 cc = (cell + 0.5) * ${SKY_CELL.toFixed(3)};
    vec3 dc = vec3(sin(cc.x) * cos(cc.y), sin(cc.y), -cos(cc.x) * cos(cc.y));
    vec3 col = skyAt(dc);

    // Colour, cube by cube: a slow-drifting field turns each cube's hue through a range that
    // suits the hour (teal to violet at night, cyan to periwinkle by day, gold to rose at sunset)
    // (a broad, slowly travelling sweep across the sky plus drifting patches, so neighbouring
    // cubes step through the range a little at a time)
    float sweep = sin(cell.x * 0.045 + cell.y * 0.07 - uTime * 0.09);
    float patches = vnoise(vec3(cell * 0.05, uTime * 0.05)) * 2.0 - 1.0;
    float n = clamp(0.5 + 0.28 * sweep + 0.34 * patches + 0.12 * (vnoise(vec3(cell * 0.14 + 7.0, uTime * 0.08)) - 0.5), 0.0, 1.0);
    float swing = mix(mix(1.5, 1.1, uDay), 0.9, uGolden);
    col = hueTurn(col, (n - 0.5) * 2.0 * swing, mix(1.7, 1.4, uDay));
    // …and each cube catches the light a little differently; now and then one glints
    float h = hash12(cell);
    col *= 0.9 + 0.2 * h;
    col *= 1.0 + pow(hash12(cell + 3.7), 40.0) * (0.5 + 0.5 * sin(uTime * 1.3 + h * 50.0)) * 0.9;

    // Cube faces: bevelled edges (lit from above) and dark seams, a few screen pixels wide
    vec2 px = fwidth(g);
    float top = smoothstep(1.0 - 3.0 * px.y, 1.0, f.y);
    float bottom = smoothstep(3.0 * px.y, 0.0, f.y);
    float left = smoothstep(3.0 * px.x, 0.0, f.x);
    float right = smoothstep(1.0 - 3.0 * px.x, 1.0, f.x);
    col *= (0.96 + 0.08 * f.y) * (1.0 + 0.32 * top + 0.12 * left - 0.28 * bottom - 0.12 * right);
    col += vec3(0.25, 0.5, 0.75) * top * 0.02 * (1.0 - uDay); // icy rim on the top edges at night
    float seam = smoothstep(0.0, 1.2 * px.x, min(f.x, 1.0 - f.x)) * smoothstep(0.0, 1.2 * px.y, min(f.y, 1.0 - f.y));
    col *= mix(0.6, 1.0, seam);

    // Stars: tiny square pixels on the cube faces (night only)
    if (uDay < 0.999) {
      vec3 sp = d * 260.0;
      float s = hash13(floor(sp));
      vec3 o = abs(fract(sp) - 0.5);
      float star = step(0.9965, s) * (1.0 - smoothstep(0.16, 0.26, max(o.x, max(o.y, o.z))));
      col += star * seam * smoothstep(0.0, 0.25, d.y) * (0.55 + 0.45 * sin(uTime * 2.0 + s * 60.0)) * 0.9 * (1.0 - uDay);
    }

    gl_FragColor = vec4(col * uExposure, 1.0);
  }
`;

/* ------------------------------------------------------------------ */
/* Snow particles                                                      */
/* ------------------------------------------------------------------ */
export const snowVertex = /* glsl */ `
  attribute float aRand;
  attribute float aSize;
  uniform float uTime;
  uniform float uOffset;
  uniform float uPx;
  varying float vAlpha;
  void main() {
    vec3 p = position;
    p.y = mod(p.y - uTime * (0.18 + aRand * 0.3) - uOffset * (0.6 + aRand), 20.0) - 1.0;
    p.x = mod(p.x + sin(uTime * 0.35 + aRand * 30.0) * 0.7 + uTime * 0.08 + 22.0, 44.0) - 22.0;
    p.z += cos(uTime * 0.3 + aRand * 20.0) * 0.6;
    vec4 mv = viewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = min(aSize * uPx * (10.0 / max(-mv.z, 0.5)), 48.0);
    vAlpha = smoothstep(0.3, 2.0, -mv.z) * (1.0 - smoothstep(18.0, 30.0, -mv.z));
  }
`;

export const snowFragment = /* glsl */ `
  uniform float uOpacity;
  varying float vAlpha;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    float a = smoothstep(0.5, 0.0, d);
    gl_FragColor = vec4(vec3(0.85, 0.93, 1.0), a * a * vAlpha * uOpacity);
  }
`;

/* ------------------------------------------------------------------ */
/* Cursor ice dust: soft core + four-point sparkle                     */
/* ------------------------------------------------------------------ */
export const dustVertex = /* glsl */ `
  attribute float aAlpha;
  attribute float aSize;
  uniform float uPx;
  varying float vA;
  void main() {
    vA = aAlpha;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = min(aSize * uPx * (16.0 / max(-mv.z, 0.5)) * (0.35 + aAlpha * 0.65), 42.0);
  }
`;

export const dustFragment = /* glsl */ `
  varying float vA;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float core = smoothstep(0.5, 0.0, length(c));
    core *= core;
    float flare = max(0.0, 1.0 - abs(c.x) * 16.0) * max(0.0, 1.0 - abs(c.y) * 2.2)
                + max(0.0, 1.0 - abs(c.y) * 16.0) * max(0.0, 1.0 - abs(c.x) * 2.2);
    float a = (core + flare * 0.55) * vA;
    if (a < 0.01) discard;
    gl_FragColor = vec4(vec3(0.72, 0.92, 1.0) * 1.5, a);
  }
`;

/* ------------------------------------------------------------------ */
/* Neural-network edges with travelling signals                        */
/* ------------------------------------------------------------------ */
export const lineVertex = /* glsl */ `
  attribute float aT;
  attribute float aRand;
  uniform vec3 uNetC;
  uniform float uNetRot;
  varying float vT;
  varying float vR;
  varying float vLX;
  void main() {
    vT = aT;
    vR = aRand;
    vec3 dn = (modelMatrix * vec4(position, 1.0)).xyz - uNetC;
    vLX = dn.x * cos(uNetRot) - dn.z * sin(uNetRot);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

export const lineFragment = /* glsl */ `
  uniform float uTime;
  uniform float uOpacity;
  varying float vT;
  varying float vR;
  varying float vLX;
  void main() {
    float sig = smoothstep(0.09, 0.0, abs(fract(uTime * (0.35 + vR * 0.4) + vR) - vT));
    float wave = pow(0.5 + 0.5 * sin(vLX * 1.1 - uTime * 3.2), 10.0);
    float a = (0.05 + 0.18 * vR) + sig * 0.55 + wave * 0.75;
    gl_FragColor = vec4(vec3(0.5, 0.82, 1.0) * a, a * uOpacity);
  }
`;

/* ------------------------------------------------------------------ */
/* Final composite: aberration, glitch, vignette, grain, scanlines     */
/* ------------------------------------------------------------------ */
export const FinalShader = {
  uniforms: {
    tDiffuse: { value: null },
    uTime: { value: 0 },
    uAberr: { value: 0.0015 },
    uGlitch: { value: 0 },
    uRes: { value: [1, 1] },
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse;
    uniform float uTime;
    uniform float uAberr;
    uniform float uGlitch;
    uniform vec2 uRes;
    varying vec2 vUv;
    float rnd(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    void main() {
      vec2 uv = vUv;
      float band = step(0.965, rnd(vec2(floor(uv.y * 38.0), floor(uTime * 18.0))));
      uv.x += band * uGlitch * (rnd(vec2(floor(uv.y * 38.0), uTime)) - 0.5) * 0.06;
      vec2 c = uv - 0.5;
      float r2 = dot(c, c);
      vec2 off = c * (uAberr + uGlitch * 0.012) * (0.6 + r2 * 2.5);
      vec3 col;
      col.r = texture2D(tDiffuse, uv + off).r;
      col.g = texture2D(tDiffuse, uv).g;
      col.b = texture2D(tDiffuse, uv - off).b;
      col *= 1.0 - smoothstep(0.12, 0.62, r2) * 0.55;
      col += (rnd(uv * uRes + fract(uTime) * 97.0) - 0.5) * 0.045;
      col += vec3(0.0, 0.01, 0.02) * sin(uv.y * uRes.y * 1.4) * 0.5;
      gl_FragColor = vec4(col, 1.0);
    }
  `,
};
