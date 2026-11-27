/**
 * GLSL ES 3.0 sources. Conventions: everything renders y-down (clip y is flipped), colours are premultiplied, and
 * the scene target has two attachments: 0 is colour, with alpha meaning "lit by the light pass" (the sky writes 0),
 * 1 is the emissive part of that colour, which skips darkening and feeds bloom.
 */

const HEAD = `#version 300 es
precision highp float;
precision highp int;
`;

/** 4×4 ordered dither, as in the old engine's sky bands. */
const BAYER = `
const float BAYER[16] = float[16](0., 8., 2., 10., 12., 4., 14., 6., 3., 11., 1., 9., 15., 7., 13., 5.);
float bayer(ivec2 p) { return (BAYER[(p.y & 3) * 4 + (p.x & 3)] + 0.5) / 16.0; }
`;

const HASH = `
uint hashu(uvec2 p) {
  uint h = p.x * 374761393u + p.y * 668265263u;
  h = (h ^ (h >> 13u)) * 1274126177u;
  return h ^ (h >> 16u);
}
float hash(ivec2 p) { return float(hashu(uvec2(p)) & 0xffffffu) / 16777216.0; }
`;

/** A full-screen triangle from gl_VertexID; no attributes. */
export const FULLSCREEN_VS = `${HEAD}
void main() {
  vec2 p = vec2(float((gl_VertexID & 1) << 2) - 1.0, float((gl_VertexID & 2) << 1) - 1.0);
  gl_Position = vec4(p, 0.0, 1.0);
}`;

export const SPRITE_VS = `${HEAD}
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec4 aUv;
layout(location = 2) in vec2 aPivot;
layout(location = 3) in vec4 aTint;
layout(location = 4) in vec2 aXform;
layout(location = 5) in uint aFlags;
uniform vec2 uClip;
uniform vec2 uTexel;
out vec2 vUv;
flat out vec4 vRect;
flat out vec4 vTint;
flat out uint vFlags;
void main() {
  vec2 c = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
  // Outlined sprites grow by a texel on each side for the outline to land in
  float pad = (aFlags & 8u) != 0u ? 1.0 : 0.0;
  vec2 t = c * (aUv.zw + 2.0 * pad) - pad;
  vec2 local = vec2((aFlags & 1u) != 0u ? aPivot.x - t.x : t.x - aPivot.x, t.y - aPivot.y) * aXform.x;
  if (aXform.y != 0.0) {
    float s = sin(aXform.y), k = cos(aXform.y);
    local = vec2(local.x * k - local.y * s, local.x * s + local.y * k);
  }
  vec2 p = floor(aPos + 0.5) + local;
  gl_Position = vec4(p * uClip + vec2(-1.0, 1.0), 0.0, 1.0);
  vUv = (aUv.xy + t) * uTexel;
  vRect = vec4(aUv.xy, aUv.xy + aUv.zw) * uTexel.xyxy;
  vTint = aTint.wzyx;
  vFlags = aFlags;
}`;

export const SPRITE_FS = `${HEAD}
uniform sampler2D uTex;
uniform vec2 uTexel;
in vec2 vUv;
flat in vec4 vRect;
flat in vec4 vTint;
flat in uint vFlags;
layout(location = 0) out vec4 oColor;
layout(location = 1) out vec4 oEmit;
float alphaAt(vec2 uv) {
  return (uv.x < vRect.x || uv.y < vRect.y || uv.x >= vRect.z || uv.y >= vRect.w) ? 0.0 : texture(uTex, uv).a;
}
void main() {
  vec4 c = (vUv.x < vRect.x || vUv.y < vRect.y || vUv.x >= vRect.z || vUv.y >= vRect.w)
    ? vec4(0.0) : texture(uTex, vUv);
  if ((vFlags & 8u) != 0u) {
    // Outline: the sprite as painted, ringed by one texel of the tint colour
    if (c.a < 0.5) {
      float n = alphaAt(vUv + vec2(uTexel.x, 0.0)) + alphaAt(vUv - vec2(uTexel.x, 0.0))
        + alphaAt(vUv + vec2(0.0, uTexel.y)) + alphaAt(vUv - vec2(0.0, uTexel.y));
      if (n <= 0.0) discard;
      c = vTint;
    }
  } else if ((vFlags & 16u) != 0u) {
    c = vec4(vTint.rgb, c.a * vTint.a);
  } else {
    c *= vTint;
  }
  if (c.a <= 0.003) discard;
  // Additive: premultiplied colour with zero alpha adds without covering what's below
  float cover = (vFlags & 4u) != 0u ? 0.0 : c.a;
  vec3 pm = c.rgb * c.a;
  oColor = vec4(pm, cover);
  oEmit = vec4((vFlags & 2u) != 0u ? pm : vec3(0.0), cover);
}`;

export const PARTICLE_VS = `${HEAD}
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec4 aColor;
layout(location = 2) in uvec2 aSizeFlags;
uniform vec2 uClip;
flat out vec4 vColor;
flat out uint vFlags;
void main() {
  vec2 c = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
  vec2 p = floor(aPos) + c * float(aSizeFlags.x);
  gl_Position = vec4(p * uClip + vec2(-1.0, 1.0), 0.0, 1.0);
  vColor = aColor.wzyx;
  vFlags = aSizeFlags.y;
}`;

export const PARTICLE_FS = `${HEAD}
flat in vec4 vColor;
flat in uint vFlags;
layout(location = 0) out vec4 oColor;
layout(location = 1) out vec4 oEmit;
void main() {
  float cover = (vFlags & 4u) != 0u ? 0.0 : vColor.a;
  vec3 pm = vColor.rgb * vColor.a;
  oColor = vec4(pm, cover);
  oEmit = vec4((vFlags & 2u) != 0u ? pm : vec3(0.0), cover);
}`;

export const LIGHT_VS = `${HEAD}
layout(location = 0) in vec2 aPos;
layout(location = 1) in float aRadius;
layout(location = 2) in vec4 aColor;
layout(location = 3) in float aIntensity;
uniform vec2 uClip;
out vec2 vLocal;
flat out vec3 vColor;
void main() {
  vec2 c = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;
  vec2 p = aPos + c * aRadius;
  gl_Position = vec4(p * uClip + vec2(-1.0, 1.0), 0.0, 1.0);
  vLocal = c;
  // Light buffer values are stored at 128/255, so 1.0 (daylight) is exact and lights can reach twice that
  vColor = aColor.zyx * aIntensity * 0.50196078;
}`;

export const LIGHT_FS = `${HEAD}
in vec2 vLocal;
flat in vec3 vColor;
out vec4 oColor;
void main() {
  float d = length(vLocal);
  if (d >= 1.0) discard;
  float f = 1.0 - d;
  oColor = vec4(vColor * f * f, 0.0);
}`;

export const SKY_FS = `${HEAD}${BAYER}${HASH}
uniform vec3 uTop, uMid, uBot;
uniform float uStars, uTime, uH;
uniform ivec2 uStarOff;
layout(location = 0) out vec4 oColor;
layout(location = 1) out vec4 oEmit;
vec3 grad(float t) { return t < 0.6 ? mix(uTop, uMid, t / 0.6) : mix(uMid, uBot, (t - 0.6) / 0.4); }
void main() {
  vec2 sp = vec2(gl_FragCoord.x, uH - gl_FragCoord.y);
  ivec2 ip = ivec2(sp);
  float t = clamp(sp.y / uH, 0.0, 1.0);
  // Ten flat bands with dithered steps between them, like the old sky
  float p = t * 9.0;
  float band = floor(p);
  if (clamp((p - band - 0.62) / 0.38, 0.0, 1.0) > bayer(ip)) band += 1.0;
  vec3 c = grad(min(band, 9.0) / 9.0);
  if (uStars > 0.0 && t < 0.65) {
    ivec2 sc = ip + uStarOff;
    float h = hash(sc);
    if (h > 0.99905) {
      float k = hash(sc + ivec2(17, 31));
      float tw = 0.55 + 0.45 * sin(uTime * (0.7 + k * 2.6) + k * 40.0);
      c = mix(c, vec3(1.0, 0.97, 0.9), clamp(uStars * tw * (1.0 - t / 0.65) * 1.6, 0.0, 1.0));
    }
  }
  oColor = vec4(c, 0.0);
  oEmit = vec4(0.0);
}`;

export const TERRAIN_FS = `${HEAD}${HASH}
precision highp usampler2D;
uniform usampler2D uCells;
uniform usampler2D uTop;
uniform sampler2D uTiles;
uniform sampler2D uMat;
uniform ivec2 uBase;
uniform vec2 uFrac;
uniform float uInvZoom, uH, uTime;
uniform int uRows, uRing, uStart, uT, uVariants;
layout(location = 0) out vec4 oColor;
layout(location = 1) out vec4 oEmit;
int ringCol(int c) { return ((c % uRing) + uRing) % uRing; }
uvec4 cellAt(ivec2 c) {
  if (c.y < 0 || c.y >= uRows || c.x < uStart || c.x >= uStart + uRing) return uvec4(0u);
  return texelFetch(uCells, ivec2(ringCol(c.x), c.y), 0);
}
vec4 tile(uint mat, uint variant, ivec2 px) {
  return texelFetch(uTiles, ivec2(int(variant % uint(uVariants)) * uT + px.x, int(mat) * uT + px.y), 0);
}
void main() {
  vec2 sp = vec2(gl_FragCoord.x, uH - gl_FragCoord.y);
  ivec2 local = ivec2(floor(uFrac + sp * uInvZoom));
  ivec2 cell = uBase + local / uT;
  ivec2 px = local - (local / uT) * uT;
  uvec4 c = cellAt(cell);
  if (c.r == 0u && c.g == 0u) discard;
  int top = int(texelFetch(uTop, ivec2(ringCol(cell.x), 0), 0).r);
  int depth = cell.y - top - 6;
  vec4 col;
  float glow = 0.0;
  int last = uT - 1;
  if (c.r != 0u) {
    col = tile(c.r, c.b, px);
    // Edges facing air: a lit top, shaded sides and bottom
    if (px.y == 0 && cellAt(cell + ivec2(0, -1)).r == 0u) col.rgb = mix(col.rgb, vec3(1.0), 0.2);
    else if (px.y == last && cellAt(cell + ivec2(0, 1)).r == 0u) col.rgb *= 0.55;
    if ((px.x == 0 && cellAt(cell + ivec2(-1, 0)).r == 0u) || (px.x == last && cellAt(cell + ivec2(1, 0)).r == 0u))
      col.rgb *= 0.72;
    vec4 m = texelFetch(uMat, ivec2(int(c.r), 0), 0);
    float maxHp = m.g * 255.0;
    float hp = float(c.a);
    if (maxHp < 254.5 && hp < maxHp) {
      // Cracks grow in as the cell loses hp: one diagonal, then a second, then chips
      float d = 1.0 - hp / max(maxHp, 1.0);
      uint h = hashu(uvec2(cell));
      int o1 = int(h & 3u) - 1;
      int o2 = int((h >> 2u) & 3u) - 1;
      bool crack = (d > 0.2 && px.x - px.y == o1 && px.y >= 2)
        || (d > 0.5 && (last - px.x) - px.y == o2 && px.y <= last - 2)
        || (d > 0.75 && hash(cell * uT + px) < (d - 0.75) * 1.6);
      if (crack) col.rgb *= 0.42;
    }
    glow = m.r * (0.85 + 0.15 * sin(uTime * 3.0 + float(cell.x) * 0.7 + float(cell.y) * 1.3));
    if (depth > 0) col.rgb *= 1.0 - min(0.6, float(depth) * 0.04);
    if (col.a < 1.0 && c.g != 0u) {
      // Translucent tiles (water, glass) show the back wall through them
      vec4 b = tile(c.g, c.b, px);
      b.rgb *= vec3(0.36, 0.36, 0.42);
      col.rgb = col.rgb * col.a + b.rgb * b.a * (1.0 - col.a);
      col.a = col.a + b.a * (1.0 - col.a);
      col.rgb /= max(col.a, 0.001);
    }
  } else {
    // Back wall: the same tile in shadow, darkening with depth at half the rate so caverns stay readable
    col = tile(c.g, c.b, px);
    col.rgb *= vec3(0.36, 0.36, 0.42);
    if (depth > 0) col.rgb *= 1.0 - min(0.6, float(depth) * 0.02);
  }
  if (col.a <= 0.003) discard;
  vec3 pm = col.rgb * col.a;
  oColor = vec4(pm, col.a);
  oEmit = vec4(pm * glow, col.a);
}`;

/** Bright pass for bloom: emissive light plus whatever the lit world pushes past the threshold. */
export const BRIGHT_FS = `${HEAD}
uniform sampler2D uScene, uEmit, uLight;
uniform vec2 uInvSize;
uniform float uThreshold;
out vec4 oColor;
void main() {
  vec2 uv = gl_FragCoord.xy * uInvSize;
  vec4 s = texture(uScene, uv);
  vec3 e = texture(uEmit, uv).rgb;
  vec3 l = texture(uLight, uv).rgb * 1.9921875;
  vec3 lit = (s.rgb - e) * l + e;
  oColor = vec4(e + max(lit - vec3(uThreshold), 0.0) * s.a, 1.0);
}`;

/** Separable 9-tap Gaussian, five fetches thanks to linear filtering. */
export const BLUR_FS = `${HEAD}
uniform sampler2D uSrc;
uniform vec2 uInvSize, uDir;
out vec4 oColor;
void main() {
  vec2 uv = gl_FragCoord.xy * uInvSize;
  vec2 o1 = uDir * 1.3846153846, o2 = uDir * 3.2307692308;
  vec3 c = texture(uSrc, uv).rgb * 0.2270270270;
  c += (texture(uSrc, uv + o1).rgb + texture(uSrc, uv - o1).rgb) * 0.3162162162;
  c += (texture(uSrc, uv + o2).rgb + texture(uSrc, uv - o2).rgb) * 0.0702702703;
  oColor = vec4(c, 1.0);
}`;

export const MAX_WAVES = 16;

/** Lighting composite and every post effect in one pass; each is skipped by a uniform branch when off. */
export const POST_FS = `${HEAD}${BAYER}
uniform sampler2D uScene, uEmit, uLight, uBloom;
uniform vec2 uSize;
uniform int uWaves;
uniform vec4 uWave[${MAX_WAVES}];
uniform float uWaveS[${MAX_WAVES}];
uniform float uChroma, uBloomK, uVignette, uFlash, uWhite, uLetterbox, uLightSteps;
uniform int uGrade;
uniform vec4 uGradeA;
uniform vec3 uTint;
out vec4 oColor;
vec3 litAt(ivec2 p, vec2 uv) {
  p = clamp(p, ivec2(0), ivec2(uSize) - 1);
  vec4 s = texelFetch(uScene, p, 0);
  vec3 e = texelFetch(uEmit, p, 0).rgb;
  vec3 l = texture(uLight, uv).rgb * 1.9921875;
  // Stepped light keeps the pixel-art look: no smooth gradients across a wall
  l = floor(l * uLightSteps + bayer(p)) / uLightSteps;
  return mix(s.rgb, (s.rgb - e) * l + e, s.a);
}
void main() {
  vec2 fp = gl_FragCoord.xy;
  ivec2 ip = ivec2(fp);
  vec2 off = vec2(0.0);
  float crest = 0.0;
  for (int i = 0; i < ${MAX_WAVES}; i++) {
    if (i >= uWaves) break;
    vec4 w = uWave[i];
    vec2 d = fp - w.xy;
    float r = length(d);
    float x = (r - w.z) / max(w.w, 1.0);
    if (abs(x) < 1.0) {
      float k = 1.0 - x * x;
      off -= d / max(r, 0.001) * sin(x * 3.14159265) * k * uWaveS[i];
      crest += k * k * uWaveS[i] * 0.012;
    }
  }
  vec2 sp = floor(fp + off);
  vec3 c;
  if (uChroma > 0.0) {
    vec2 dir = (fp / uSize - 0.5) * uChroma * 6.0;
    vec2 a = sp + dir, b = sp - dir;
    c = vec3(litAt(ivec2(a), (a + 0.5) / uSize).r, litAt(ivec2(sp), (sp + 0.5) / uSize).g,
      litAt(ivec2(b), (b + 0.5) / uSize).b);
  } else {
    c = litAt(ivec2(sp), (sp + 0.5) / uSize);
  }
  if (uBloomK > 0.0) {
    vec3 g = texture(uBloom, (sp + 0.5) / uSize).rgb * uBloomK;
    c += floor(g * 24.0 + bayer(ip)) / 24.0;
  }
  c += crest;
  if (uGrade != 0) {
    c *= uGradeA.x;
    c = (c - 0.5) * uGradeA.y + 0.5;
    float luma = dot(c, vec3(0.299, 0.587, 0.114));
    c = mix(vec3(luma), c, uGradeA.z) * uTint;
  }
  if (uVignette > 0.0) {
    vec2 q = fp / uSize - 0.5;
    float v = smoothstep(0.3, 0.75, length(q * vec2(1.0, uSize.y / uSize.x) * 1.25));
    c *= 1.0 - uVignette * v;
  }
  c += uFlash;
  c = mix(c, vec3(1.0), uWhite);
  if (uLetterbox > 0.0) {
    float bar = floor(uLetterbox * uSize.y * 0.12);
    if (fp.y < bar || fp.y > uSize.y - bar) c = vec3(0.0);
  }
  oColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}`;
