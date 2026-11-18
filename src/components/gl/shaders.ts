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
/* Ice blocks                                                          */
/* ------------------------------------------------------------------ */
export const iceVertex = /* glsl */ `
  attribute float aRand;
  varying vec3 vN;
  varying vec3 vW;
  varying vec3 vL;
  varying float vR;
  void main() {
    vL = position;
    mat4 m = modelMatrix * instanceMatrix;
    vec4 w = m * vec4(position, 1.0);
    vW = w.xyz;
    vN = normalize(mat3(m) * normal);
    vR = aRand;
    gl_Position = projectionMatrix * viewMatrix * w;
  }
`;

export const iceFragment = /* glsl */ `
  uniform vec3 uCam;
  uniform float uTime;
  uniform vec3 uMouse;
  uniform float uMouseOn;
  uniform float uPulseT;
  uniform vec3 uHalf;
  uniform float uExposure;
  uniform vec3 uFog;
  varying vec3 vN;
  varying vec3 vW;
  varying vec3 vL;
  varying float vR;
  ${NOISE}
  ${ENV}
  void main() {
    vec3 N = normalize(vN);
    vec3 V = normalize(uCam - vW);
    float ndv = clamp(dot(N, V), 0.0, 1.0);
    float fres = pow(1.0 - ndv, 3.0);

    // Glowing bevelled edges: two local coordinates close to the box boundary
    vec3 a = abs(vL) / uHalf;
    float mx = max(a.x, max(a.y, a.z));
    float mn = min(a.x, min(a.y, a.z));
    float md = a.x + a.y + a.z - mx - mn;
    float edge = smoothstep(0.80, 0.985, md);

    // Frozen interior: cloudy frost + thin crack sheets
    vec3 q = vL * 2.6 + vR * 17.0;
    float frost = fbm(q * 2.0);
    float crack = 1.0 - smoothstep(0.0, 0.03, abs(vnoise(q * 3.1) - 0.5));
    float bubbles = smoothstep(0.82, 0.9, vnoise(q * 9.0));

    vec3 R = reflect(-V, normalize(N + (frost - 0.5) * 0.35));
    vec3 T = refract(-V, N, 0.76);
    vec3 refl = envColor(R);
    vec3 refr = envColor(normalize(T + (frost - 0.5) * 0.7));

    vec3 deep = vec3(0.035, 0.085, 0.14);
    vec3 body = mix(deep, refr * vec3(0.5, 0.78, 1.0), 0.55);
    body += vec3(0.55, 0.75, 0.95) * frost * 0.2;
    body += vec3(0.75, 0.9, 1.0) * (crack * 0.22 + bubbles * 0.15);
    vec3 col = mix(body, refl, fres * 0.8);
    col += vec3(0.7, 0.9, 1.0) * edge * (0.5 + 0.5 * fres);

    // Frosted micro-normals break flat faces into sparkles instead of a solid white flash
    vec3 Nf = normalize(N + (vec3(vnoise(q * 7.0), vnoise(q * 7.0 + 31.0), vnoise(q * 7.0 + 57.0)) - 0.5) * 0.55);
    vec3 L = normalize(vec3(-0.4, 0.8, 0.5));
    col += pow(max(dot(Nf, normalize(L + V)), 0.0), 140.0) * 1.1;

    // Pointer proximity glow
    float dm = distance(vW, uMouse);
    col += vec3(0.4, 0.78, 1.0) * uMouseOn * smoothstep(2.2, 0.0, dm) * (0.22 + edge * 0.9);

    // Ripple pulse (hover on projects, page transitions, intro)
    float pt = uTime - uPulseT;
    float ring = exp(-pow(length(vW - vec3(0.0, 2.0, 0.0)) - pt * 9.0, 2.0) * 1.4) * smoothstep(1.8, 0.0, pt);
    col += vec3(0.5, 0.85, 1.0) * ring * (0.5 + edge);

    col *= 0.93 + 0.07 * sin(uTime * 1.3 + vR * 40.0);
    float fogF = smoothstep(14.0, 32.0, distance(vW, uCam));
    col = mix(col, uFog, fogF);
    gl_FragColor = vec4(col * uExposure, 1.0);
  }
`;

/* ------------------------------------------------------------------ */
/* Snow ground                                                         */
/* ------------------------------------------------------------------ */
export const groundVertex = /* glsl */ `
  varying vec3 vW;
  void main() {
    vec4 w = modelMatrix * vec4(position, 1.0);
    vW = w.xyz;
    gl_Position = projectionMatrix * viewMatrix * w;
  }
`;

export const groundFragment = /* glsl */ `
  uniform float uTime;
  uniform float uOpacity;
  varying vec3 vW;
  ${NOISE}
  void main() {
    float d = length(vW.xz);
    float fade = smoothstep(24.0, 3.0, d);
    float n = fbm(vec3(vW.xz * 0.55, 0.0));
    vec3 col = mix(vec3(0.03, 0.045, 0.065), vec3(0.11, 0.15, 0.2), n);
    col += vec3(0.25, 0.42, 0.58) * smoothstep(6.0, 0.0, d) * 0.3;
    col *= mix(0.45, 1.0, smoothstep(2.5, 3.6, d));

    // HUD rings on the ice
    float ringD = abs(fract(d / 2.5 + 0.5) - 0.5) * 2.5;
    col += vec3(0.4, 0.7, 0.95) * smoothstep(0.025, 0.0, ringD) * 0.18 * smoothstep(3.0, 4.0, d);

    // Glitter
    vec2 cell = floor(vW.xz * 14.0);
    float h = hash12(cell);
    float tw = step(0.982, h) * (0.5 + 0.5 * sin(uTime * 3.0 + h * 120.0));
    float dotv = smoothstep(0.22, 0.0, length(fract(vW.xz * 14.0) - 0.5));
    col += vec3(0.8, 0.92, 1.0) * tw * dotv * 1.6;

    gl_FragColor = vec4(col, fade * uOpacity);
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
    p.y = mod(p.y - uTime * (0.22 + aRand * 0.4) - uOffset * (0.6 + aRand), 17.0) - 3.0;
    p.x = mod(p.x + sin(uTime * 0.35 + aRand * 30.0) * 0.7 + uTime * 0.08 + 16.0, 32.0) - 16.0;
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
