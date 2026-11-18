/* All colours are authored directly in display space (no tone mapping). */

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

const ENV = /* glsl */ `
  vec3 envColor(vec3 d) {
    float y = d.y * 0.5 + 0.5;
    vec3 c = mix(vec3(0.015, 0.025, 0.045), vec3(0.30, 0.44, 0.60), smoothstep(0.25, 1.0, y));
    c += vec3(0.55, 0.78, 1.0) * pow(1.0 - abs(d.y), 8.0) * 0.45;
    vec3 L = normalize(vec3(-0.4, 0.8, 0.5));
    c += vec3(0.9, 0.97, 1.0) * pow(max(dot(d, L), 0.0), 24.0) * 1.15;
    vec3 L2 = normalize(vec3(0.7, 0.2, -0.6));
    c += vec3(0.35, 0.75, 1.0) * pow(max(dot(d, L2), 0.0), 10.0) * 0.6;
    return c;
  }
`;

/* ------------------------------------------------------------------ */
/* Village pieces: whitewashed houses + blue domes that turn into ice  */
/* ------------------------------------------------------------------ */
export const pieceVertex = /* glsl */ `
  attribute float aRand;
  attribute float aMix;
  varying vec3 vN;
  varying vec3 vW;
  varying vec3 vL;
  varying vec3 vLN;
  varying vec3 vS;
  varying float vR;
  varying float vMix;
  void main() {
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
  varying vec3 vN;
  varying vec3 vW;
  varying vec3 vL;
  varying vec3 vLN;
  varying vec3 vS;
  varying float vR;
  varying float vMix;
  ${NOISE}
  ${ENV}
  void main() {
    vec3 N = normalize(vN);
    if (!gl_FrontFacing) N = -N;
    vec3 V = normalize(uCam - vW);
    vec3 L = normalize(vec3(-0.8, 0.5, 0.33));
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
        vec3 blue = vec3(0.04, 0.2, 0.6);
        village = blue * (0.22 + 0.8 * ndl) + envColor(reflect(-V, N)) * (0.12 + fres * 0.55);
        village += pow(max(dot(N, normalize(L + V)), 0.0), 70.0) * 0.7;
        village += vec3(0.35, 0.75, 1.0) * fres * 0.3;
      } else {
        vec3 albedo = vec3(0.85, 0.89, 0.96) - vnoise(vW * 11.0) * 0.07;
        village = albedo * (0.08 + 0.7 * ndl + 0.16 * (0.5 + 0.5 * N.y) + 0.16 * step(0.5, vLN.y));
        village *= mix(0.5, 1.0, smoothstep(0.25, 1.0, (vL.y + 0.5) * vS.y));
        village += vec3(0.55, 0.8, 1.0) * edgeW * 0.2;
        vec3 an = abs(vLN);
        if (an.y < 0.5) {
          // Windows with blue shutters on the walls; some are lit
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
          village = mix(village, mix(vec3(0.015, 0.03, 0.06), glow, lit), has * glass);
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
    col = mix(col, uFog, smoothstep(26.0, 70.0, distance(vW, uCam)));
    gl_FragColor = vec4(col * uExposure, 1.0);
  }
`;

const LINES = /* glsl */ `
  float gridLine(float x, float w) {
    float fw = max(fwidth(x), 1e-4);
    return 1.0 - smoothstep(0.0, fw * w, abs(fract(x - 0.5) - 0.5));
  }
`;

/* ------------------------------------------------------------------ */
/* Caldera terrain: moonlit volcanic rock <-> holographic contour map  */
/* ------------------------------------------------------------------ */
export const terrainVertex = /* glsl */ `
  varying vec3 vW;
  varying vec3 vN;
  void main() {
    vec4 w = modelMatrix * vec4(position, 1.0);
    vW = w.xyz;
    vN = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * viewMatrix * w;
  }
`;

export const terrainFragment = /* glsl */ `
  uniform float uTime;
  uniform float uLand;
  uniform float uExposure;
  uniform vec3 uCam;
  uniform vec3 uFog;
  varying vec3 vW;
  varying vec3 vN;
  ${NOISE}
  ${LINES}
  void main() {
    vec3 N = normalize(vN);
    vec3 L = normalize(vec3(-0.8, 0.5, 0.33));
    float ndl = max(dot(N, L), 0.0);
    float slope = 1.0 - N.y;
    float strata = vnoise(vec3(vW.x * 0.25, vW.y * 5.0, vW.z * 0.25));
    vec3 rock = mix(vec3(0.03, 0.034, 0.045), vec3(0.09, 0.088, 0.1), strata);
    rock = mix(rock, vec3(0.14, 0.07, 0.055), smoothstep(0.6, 0.85, vnoise(vec3(vW.xz * 0.18, vW.y * 1.4))) * 0.55);
    vec3 top = vec3(0.13, 0.14, 0.16) + vnoise(vW * 3.0) * 0.03;
    vec3 real = mix(top, rock, smoothstep(0.22, 0.5, slope)) * (0.16 + 0.75 * ndl);
    real += vec3(0.4, 0.75, 1.0) * gridLine(vW.y * 2.0, 1.2) * 0.05;

    float contour = gridLine(vW.y * 4.0, 1.4);
    float grid = max(gridLine(vW.x * 1.2, 1.0), gridLine(vW.z * 1.2, 1.0));
    float sweep = exp(-pow((fract(vW.z * 0.045 + uTime * 0.07) - 0.5) * 26.0, 2.0));
    vec3 holo = vec3(0.008, 0.016, 0.03) + vec3(0.3, 0.72, 1.0) * (contour * 0.5 + grid * 0.14) * (0.6 + sweep * 1.6);
    holo += vec3(0.2, 0.5, 0.8) * sweep * 0.05 + vec3(0.3, 0.7, 1.0) * pow(slope, 3.0) * 0.06;

    vec3 col = mix(holo, real, uLand);
    col = mix(col, uFog, smoothstep(24.0, 75.0, distance(vW, uCam)));
    gl_FragColor = vec4(col * uExposure, 1.0);
  }
`;

/* ------------------------------------------------------------------ */
/* Aegean sea with a moon path                                         */
/* ------------------------------------------------------------------ */
export const seaFragment = /* glsl */ `
  uniform float uTime;
  uniform float uLand;
  uniform float uExposure;
  uniform vec3 uCam;
  uniform vec3 uFog;
  uniform vec3 uMoonDir;
  varying vec3 vW;
  ${NOISE}
  ${LINES}
  void main() {
    vec2 p = vW.xz;
    float t = uTime * 0.35;
    vec2 wv = vec2(vnoise(vec3(p * 0.6, t)) - 0.5, vnoise(vec3(p * 0.6 + 11.0, t)) - 0.5) * 0.35
            + vec2(sin(p.x * 1.7 + t * 3.0), cos(p.y * 2.3 - t * 2.4)) * 0.05;
    vec3 N = normalize(vec3(wv.x, 1.0, wv.y));
    vec3 V = normalize(uCam - vW);
    vec3 R = reflect(-V, N);
    float fres = 0.03 + 0.97 * pow(1.0 - max(dot(N, V), 0.0), 5.0);
    vec3 skyR = mix(vec3(0.015, 0.025, 0.045), vec3(0.12, 0.18, 0.26), smoothstep(-0.1, 0.5, R.y));
    float md = max(dot(R, uMoonDir), 0.0);
    vec3 moon = vec3(0.85, 0.92, 1.0) * (pow(md, 700.0) * 2.2 + pow(md, 60.0) * 0.22) * (0.55 + 0.9 * vnoise(vec3(p * 3.0, t * 2.0)));
    float dist = distance(vW, uCam);
    vec3 real = vec3(0.006, 0.014, 0.026) + skyR * fres * 0.8 + moon;
    float grid = max(gridLine(p.x * 0.5, 1.0), gridLine(p.y * 0.5, 1.0)) * smoothstep(60.0, 10.0, dist);
    vec3 holo = vec3(0.004, 0.008, 0.014) + vec3(0.3, 0.7, 1.0) * grid * 0.12 + moon * 0.4;
    vec3 col = mix(holo, real, uLand);
    col = mix(col, uFog, smoothstep(30.0, 120.0, dist));
    gl_FragColor = vec4(col * uExposure, 1.0);
  }
`;

/* ------------------------------------------------------------------ */
/* Windmills and other landscape props                                 */
/* ------------------------------------------------------------------ */
export const propVertex = /* glsl */ `
  varying vec3 vW;
  varying vec3 vN;
  void main() {
    vec4 w = modelMatrix * vec4(position, 1.0);
    vW = w.xyz;
    vN = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * viewMatrix * w;
  }
`;

export const propFragment = /* glsl */ `
  uniform vec3 uColor;
  uniform float uLand;
  uniform float uExposure;
  uniform vec3 uCam;
  uniform vec3 uFog;
  varying vec3 vW;
  varying vec3 vN;
  void main() {
    vec3 N = normalize(vN);
    if (!gl_FrontFacing) N = -N;
    vec3 L = normalize(vec3(-0.8, 0.5, 0.33));
    vec3 real = uColor * (0.08 + 0.6 * max(dot(N, L), 0.0) + 0.15 * (0.5 + 0.5 * N.y));
    float fres = pow(1.0 - abs(dot(N, normalize(uCam - vW))), 2.0);
    vec3 col = mix(vec3(0.35, 0.75, 1.0) * fres * 0.9, real, uLand);
    col = mix(col, uFog, smoothstep(26.0, 75.0, distance(vW, uCam)));
    gl_FragColor = vec4(col * uExposure, 1.0);
  }
`;

/* ------------------------------------------------------------------ */
/* Sky dome with aurora + stars                                        */
/* ------------------------------------------------------------------ */
export const skyVertex = /* glsl */ `
  varying vec3 vDir;
  void main() {
    vDir = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

export const skyFragment = /* glsl */ `
  uniform float uTime;
  uniform float uExposure;
  uniform vec3 uMoonDir;
  varying vec3 vDir;
  ${NOISE}
  void main() {
    vec3 d = normalize(vDir);
    float y = d.y;
    vec3 col = mix(vec3(0.01, 0.014, 0.022), vec3(0.028, 0.045, 0.075), smoothstep(-0.2, 0.7, y));
    col += vec3(0.07, 0.11, 0.16) * exp(-abs(y - 0.02) * 9.0);

    float a = fbm(vec3(d.x * 2.2 + uTime * 0.018, d.y * 5.0, d.z * 2.2 - uTime * 0.012));
    float band = smoothstep(0.5, 0.85, a) * smoothstep(0.05, 0.4, y) * (1.0 - smoothstep(0.65, 1.0, y));
    col += mix(vec3(0.1, 0.42, 0.5), vec3(0.25, 0.3, 0.7), d.x * 0.5 + 0.5) * band * 0.32;

    vec3 sp = d * 260.0;
    vec3 cell = floor(sp);
    float s = hash13(cell);
    float star = step(0.9965, s) * smoothstep(0.32, 0.0, length(fract(sp) - 0.5));
    col += star * smoothstep(0.0, 0.25, y) * (0.55 + 0.45 * sin(uTime * 2.0 + s * 60.0)) * 0.9;

    // Moon with a soft halo
    float md = dot(d, uMoonDir);
    float disc = smoothstep(0.99955, 0.99968, md);
    float craters = vnoise(d * 900.0) * 0.22 + vnoise(d * 2600.0) * 0.1;
    col += vec3(0.86, 0.9, 1.0) * disc * (0.92 - craters);
    col += vec3(0.35, 0.5, 0.72) * pow(max(md, 0.0), 260.0) * 0.28 + vec3(0.08, 0.12, 0.2) * pow(max(md, 0.0), 14.0) * 0.3;

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
  varying float vT;
  varying float vR;
  void main() {
    vT = aT;
    vR = aRand;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

export const lineFragment = /* glsl */ `
  uniform float uTime;
  uniform float uOpacity;
  varying float vT;
  varying float vR;
  void main() {
    float sig = smoothstep(0.09, 0.0, abs(fract(uTime * (0.35 + vR * 0.4) + vR) - vT));
    float a = (0.1 + 0.35 * vR) + sig * 1.4;
    gl_FragColor = vec4(vec3(0.5, 0.82, 1.0) * a, a * uOpacity);
  }
`;

/* ------------------------------------------------------------------ */
/* Final composite: chromatic aberration, glitch, grain, vignette      */
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
