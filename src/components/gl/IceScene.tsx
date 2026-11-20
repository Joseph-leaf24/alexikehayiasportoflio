import { useEffect, useRef } from "react";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import { sceneState, ui, type LayoutKey } from "@/lib/store";
import { prefersReducedMotion } from "@/lib/smoothScroll";
import { getQuality, MAX_DPR } from "@/lib/device";
import { dayCycle, parseTimeOverride, sceneSun, sunDirection, wallClockToday } from "@/lib/sun";
import { profile } from "@/data/profile";
import {
  buildFloes,
  buildLayouts,
  buildMoon,
  buildTerrain,
  buildWater,
  CELL,
  KIND_DOME,
  SEA_LEVEL,
  SEA_PLANE_Y,
  waveHeight,
  type Layout,
} from "./layouts";
import {
  FinalShader,
  dustFragment,
  dustVertex,
  lineFragment,
  lineVertex,
  moonFragment,
  moonVertex,
  pieceFragment,
  pieceVertex,
  seaFragment,
  seaVertex,
  skyFragment,
  skyVertex,
  snowFragment,
  snowVertex,
  sunFragment,
  sunVertex,
  terrainFragment,
  terrainVertex,
  waterFragment,
  waterVertex,
} from "./shaders";

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
const smoothstep = (a: number, b: number, x: number) => {
  const t = clamp01((x - a) / (b - a));
  return t * t * (3 - 2 * t);
};
const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
/** Frame-rate independent smoothing factor: `x += (target - x) * damp(dt, rate)` */
const damp = (dt: number, lambda: number) => 1 - Math.exp(-dt * lambda);
const DEG = Math.PI / 180;
/** Where the moon hangs at night: places the voxel moon, the sky halo and the moon path on the sea */
const MOON_DIR = new THREE.Vector3(0.62, 0.24, -1).normalize();
/** How far the moon sinks (below the horizon) by full daylight */
const MOON_SET = 20 * DEG;
/** The moon and the sun hang this far out (inside the sky dome), around this point */
const SKY_R = 118;
const SKY_ORIGIN = new THREE.Vector3(0, 4, -4);
/** Key light at night: the original moonlight. By day it turns toward the real sun. */
const NIGHT_LIGHT = new THREE.Vector3(-0.8, 0.5, 0.33).normalize();
/** Distance fog: night blue-black, day haze (matches the day sky's horizon) and a warm sunrise / sunset haze */
const NIGHT_FOG = new THREE.Vector3(0.012, 0.018, 0.028);
const DAY_FOG = new THREE.Vector3(0.27, 0.38, 0.48);
const GOLD_FOG = new THREE.Vector3(0.5, 0.36, 0.34);
/** Centre of the hold-to-shatter explosion (the village) */
const BREAK_C = new THREE.Vector3(0, 2, -4);

/**
 * Which two layouts we are between, and how far (0..1), for the current scroll. Each
 * transition runs while the next section's top moves from 95% to 25% of the viewport
 * height; a page override (detail / 404) wins outright.
 */
function resolveBlend(scrollY: number, vh: number): { a: LayoutKey; b: LayoutKey; t: number } {
  if (sceneState.override) return { a: sceneState.override, b: sceneState.override, t: 0 };
  const stops = sceneState.stops;
  if (!stops.length) return { a: "village", b: "village", t: 0 };
  let a = stops[0].layout;
  let b = a;
  let t = 0;
  for (let i = 1; i < stops.length; i++) {
    const start = stops[i].y - vh * 0.95;
    const end = stops[i].y - vh * 0.25;
    if (scrollY >= end) {
      a = b = stops[i].layout;
      t = 0;
    } else if (scrollY > start) {
      a = stops[i - 1].layout;
      b = stops[i].layout;
      t = (scrollY - start) / (end - start);
      break;
    } else break;
  }
  return { a, b, t };
}

/**
 * The fixed WebGL backdrop: a voxel Cycladic village on a caldera cliff (instanced
 * houses, domes and windmills over voxel hills, a sea of water blocks, a voxel moon and
 * sun, ice floes, snow and cursor ice dust) that morphs between formations as the page
 * scrolls (see layouts.ts). The light follows the real time of day in Amsterdam (sun by
 * day, moon by night; `?time=HH:MM` pins it for previews). Holding a click on the hero
 * shatters everything (village, hills, water blocks, floes, sun and moon) until it's
 * released. The pointer, scroll velocity and UI pulses feed in through sceneState.
 * It always renders at full quality, whatever the device. Without WebGL it adds
 * `no-webgl` to <html> and draws nothing.
 */
export default function IceScene() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false, powerPreference: "high-performance" });
    } catch {
      sceneState.webgl = false;
      sceneState.ready = true;
      document.documentElement.classList.add("no-webgl");
      return;
    }

    const reduced = prefersReducedMotion();
    const coarse = window.matchMedia("(pointer: coarse)").matches;
    const motion = reduced ? 0.25 : 1;
    // Always the full quality, on every device (src/lib/device.ts): the whole cube world,
    // bloom and grain, at the screen's own resolution up to 2×. Nothing is lowered at runtime.
    const q = getQuality();
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);

    renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.setPixelRatio(dpr);
    renderer.setClearColor(0x05070a, 1);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 260);
    const disposables: { dispose: () => void }[] = [];
    const keep = <T extends { dispose: () => void }>(x: T) => {
      disposables.push(x);
      return x;
    };
    const fog = NIGHT_FOG.clone();

    /* Time of day: shared by every lit material (see shaders.ts) and updated by updateDaylight() */
    const dayU = {
      uDay: { value: 0 },
      uGolden: { value: 0 },
      uSunDir: { value: new THREE.Vector3(0, -1, 0) },
      uLightDir: { value: NIGHT_LIGHT.clone() },
      uLightCol: { value: new THREE.Vector3(1, 1, 1) },
      uAmbient: { value: new THREE.Vector3(1, 1, 1) },
    };
    const moonDir = MOON_DIR.clone();
    const moonDirU = { value: moonDir };
    /* Hold-to-shatter, shared by the terrain and water blocks */
    const breakU = { uBreak: { value: 0 }, uBreakC: { value: BREAK_C } };

    /* ---------------- Village pieces ---------------- */
    const set = buildLayouts();
    const N = set.count;
    const L = set.layouts;

    const boxGeo = keep(new RoundedBoxGeometry(1, 1, 1, 2, 0.06));
    const domeGeo = keep(new THREE.SphereGeometry(1, 28, 14, 0, Math.PI * 2, 0, Math.PI / 2));
    const boxRand = new Float32Array(Math.max(1, set.boxCount));
    const domeRand = new Float32Array(Math.max(1, set.domeCount));
    for (let i = 0; i < N; i++) (set.kinds[i] === KIND_DOME ? domeRand : boxRand)[set.local[i]] = set.rand[i];
    const boxMix = new Float32Array(Math.max(1, set.boxCount));
    const domeMix = new Float32Array(Math.max(1, set.domeCount));
    const boxMixAttr = new THREE.InstancedBufferAttribute(boxMix, 1).setUsage(THREE.DynamicDrawUsage);
    const domeMixAttr = new THREE.InstancedBufferAttribute(domeMix, 1).setUsage(THREE.DynamicDrawUsage);
    boxGeo.setAttribute("aRand", new THREE.InstancedBufferAttribute(boxRand, 1));
    boxGeo.setAttribute("aMix", boxMixAttr);
    domeGeo.setAttribute("aRand", new THREE.InstancedBufferAttribute(domeRand, 1));
    domeGeo.setAttribute("aMix", domeMixAttr);
    const boxTint = new Float32Array(Math.max(1, set.boxCount) * 4);
    for (let i = 0; i < N; i++) if (set.kinds[i] !== KIND_DOME) boxTint.set(set.tints.subarray(i * 4, i * 4 + 4), set.local[i] * 4);
    boxGeo.setAttribute("aTint", new THREE.InstancedBufferAttribute(boxTint, 4));
    domeGeo.setAttribute("aTint", new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, set.domeCount) * 4), 4));

    const pieceUniforms = (dome: boolean) => ({
      uCam: { value: new THREE.Vector3() },
      uTime: { value: 0 },
      uMouse: { value: new THREE.Vector3(999, 999, 999) },
      uMouseOn: { value: 0 },
      uPulseT: { value: -100 },
      uPulseC: { value: new THREE.Vector3() },
      uExposure: { value: 1 },
      uFog: { value: fog },
      uDome: { value: dome ? 1 : 0 },
      uNet: { value: 0 },
      uNetC: { value: L.network.center.clone() },
      uNetRot: { value: 0 },
      uHi: { value: 0 },
      uHiC: { value: new THREE.Vector3() },
      ...dayU,
    });
    const houseMat = keep(new THREE.ShaderMaterial({ vertexShader: pieceVertex, fragmentShader: pieceFragment, uniforms: pieceUniforms(false) }));
    const domeMat = keep(
      new THREE.ShaderMaterial({ vertexShader: pieceVertex, fragmentShader: pieceFragment, uniforms: pieceUniforms(true), side: THREE.DoubleSide })
    );
    const houses = new THREE.InstancedMesh(boxGeo, houseMat, Math.max(1, set.boxCount));
    const domes = new THREE.InstancedMesh(domeGeo, domeMat, Math.max(1, set.domeCount));
    for (const m of [houses, domes]) {
      m.frustumCulled = false;
      m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      scene.add(m);
    }
    const pieceMats = [houseMat, domeMat];

    // Current (damped) transforms — the village starts in the sky and drops into place
    const cur = L.village.pos.map((p, i) => p.clone().add(new THREE.Vector3(0, 14 + set.rand[i] * 6, 0)));
    const curQ = L.village.quat.map((q) => q.clone());
    const curS = L.village.scale.map((s) => s.clone());
    const curMix = new Float32Array(N);

    /* ---------------- Landscape: voxel hills, water blocks, sea, voxel moon + sun, floes ---------------- */
    const landU = {
      uTime: { value: 0 },
      uLand: { value: 1 },
      uExposure: { value: 0 },
      uCam: { value: new THREE.Vector3() },
      uFog: { value: fog },
      uMoonDir: moonDirU,
      uShockT: { value: -100 },
      ...dayU,
    };

    // Hills: one instance per cube, so the whole landscape can break apart
    const terrain = buildTerrain(q.terrain);
    const voxGeo = keep(new THREE.BoxGeometry(1, 1, 1));
    const voxRand = new Float32Array(terrain.count);
    const voxDepth = new Float32Array(terrain.count);
    for (let i = 0; i < terrain.count; i++) {
      voxRand[i] = Math.random();
      voxDepth[i] = terrain.data[i * 4 + 3];
    }
    voxGeo.setAttribute("aRand", new THREE.InstancedBufferAttribute(voxRand, 1));
    voxGeo.setAttribute("aDepth", new THREE.InstancedBufferAttribute(voxDepth, 1));
    const terrU = {
      ...landU,
      ...breakU,
      uGrow: { value: 0 },
      uAmp: { value: 1 },
      uCenter: { value: new THREE.Vector3(0, 0, -4) },
      uMouse: { value: new THREE.Vector3(999, 999, 999) },
      uMouseOn: { value: 0 },
      uPulseT: { value: -100 },
    };
    const terrMat = keep(new THREE.ShaderMaterial({ vertexShader: terrainVertex, fragmentShader: terrainFragment, uniforms: terrU }));
    const hills = new THREE.InstancedMesh(voxGeo, terrMat, terrain.count);
    {
      const m = new THREE.Matrix4();
      const q = new THREE.Quaternion();
      const pos = new THREE.Vector3();
      const sc = new THREE.Vector3(CELL, CELL, CELL);
      for (let i = 0; i < terrain.count; i++) {
        pos.set(terrain.data[i * 4], terrain.data[i * 4 + 1], terrain.data[i * 4 + 2]);
        hills.setMatrixAt(i, m.compose(pos, q, sc));
      }
    }
    hills.frustumCulled = false;
    scene.add(hills);

    // Near sea: a field of water blocks riding the swell wherever there's no land (they break apart with the hills)
    const water = buildWater({ xMin: q.terrain.xMin, xMax: q.terrain.xMax, cell: q.waterCell });
    const waterGeo = keep(new THREE.BoxGeometry(q.waterCell * 0.96, 0.3, q.waterCell * 0.96));
    const waterRand = new Float32Array(Math.max(1, water.count));
    for (let i = 0; i < water.count; i++) waterRand[i] = water.data[i * 3 + 2];
    waterGeo.setAttribute("aRand", new THREE.InstancedBufferAttribute(waterRand, 1));
    const waterMat = keep(
      new THREE.ShaderMaterial({
        vertexShader: waterVertex,
        fragmentShader: waterFragment,
        uniforms: { ...landU, ...breakU, uHalf: { value: new THREE.Vector3(q.waterCell * 0.48, 0.15, q.waterCell * 0.48) } },
      })
    );
    const waterMesh = new THREE.InstancedMesh(waterGeo, waterMat, Math.max(1, water.count));
    {
      const m = new THREE.Matrix4();
      for (let i = 0; i < water.count; i++) waterMesh.setMatrixAt(i, m.makeTranslation(water.data[i * 3], SEA_LEVEL, water.data[i * 3 + 1]));
    }
    waterMesh.count = water.count;
    waterMesh.frustumCulled = false;
    scene.add(waterMesh);

    // Distant sea: a flat plane just under the water blocks' lowest trough, out to the horizon
    const seaGeo = keep(new THREE.PlaneGeometry(420, 420));
    seaGeo.rotateX(-Math.PI / 2);
    seaGeo.translate(0, SEA_PLANE_Y, 0);
    const seaMat = keep(new THREE.ShaderMaterial({ vertexShader: seaVertex, fragmentShader: seaFragment, uniforms: landU }));
    scene.add(new THREE.Mesh(seaGeo, seaMat));

    // Moon: a shell of glowing cubes far behind the village
    const moonData = buildMoon(7, q.moonCube);
    const moonGeo = keep(new THREE.BoxGeometry(moonData.cube * 0.94, moonData.cube * 0.94, moonData.cube * 0.94));
    const moonRand = new Float32Array(moonData.count);
    for (let i = 0; i < moonData.count; i++) moonRand[i] = Math.random();
    moonGeo.setAttribute("aRand", new THREE.InstancedBufferAttribute(moonRand, 1));
    moonGeo.setAttribute("aCrater", new THREE.InstancedBufferAttribute(moonData.crater, 1));
    const moonMat = keep(
      new THREE.ShaderMaterial({
        vertexShader: moonVertex,
        fragmentShader: moonFragment,
        uniforms: {
          uCenter: { value: MOON_DIR.clone().multiplyScalar(SKY_R).add(SKY_ORIGIN) },
          uTime: { value: 0 },
          uBurst: { value: 0 },
          uExposure: { value: 1 },
        },
      })
    );
    const moon = new THREE.InstancedMesh(moonGeo, moonMat, moonData.count);
    {
      const m = new THREE.Matrix4();
      for (let i = 0; i < moonData.count; i++) {
        m.makeTranslation(moonData.offsets[i * 3], moonData.offsets[i * 3 + 1], moonData.offsets[i * 3 + 2]);
        moon.setMatrixAt(i, m);
      }
    }
    moon.frustumCulled = false;
    scene.add(moon);

    // Sun: a smaller shell of glowing cubes on its (framed) daytime path; it sinks into the sea at sunset
    const sunData = buildMoon(5.5, q.moonCube);
    const sunGeo = keep(new THREE.BoxGeometry(sunData.cube * 0.94, sunData.cube * 0.94, sunData.cube * 0.94));
    sunGeo.setAttribute("aRand", new THREE.InstancedBufferAttribute(Float32Array.from({ length: sunData.count }, () => Math.random()), 1));
    const sunMat = keep(
      new THREE.ShaderMaterial({
        vertexShader: sunVertex,
        fragmentShader: sunFragment,
        uniforms: {
          uCenter: { value: new THREE.Vector3() },
          uTime: { value: 0 },
          uBurst: { value: 0 },
          uExposure: { value: 1 },
          uGolden: dayU.uGolden,
          uHalf: { value: sunData.cube * 0.47 },
        },
      })
    );
    const sun = new THREE.InstancedMesh(sunGeo, sunMat, sunData.count);
    {
      const m = new THREE.Matrix4();
      for (let i = 0; i < sunData.count; i++) sun.setMatrixAt(i, m.makeTranslation(sunData.offsets[i * 3], sunData.offsets[i * 3 + 1], sunData.offsets[i * 3 + 2]));
    }
    sun.frustumCulled = false;
    sun.visible = false;
    scene.add(sun);

    // Ice floes bobbing at the foot of the cliff (same material as the pieces, fully iced)
    const floes = buildFloes(q.floes);
    const floeGeo = keep(new RoundedBoxGeometry(1, 1, 1, 2, 0.08));
    floeGeo.setAttribute("aRand", new THREE.InstancedBufferAttribute(Float32Array.from(floes.map((f) => f.rand)), 1));
    floeGeo.setAttribute("aMix", new THREE.InstancedBufferAttribute(new Float32Array(floes.length).fill(1), 1));
    floeGeo.setAttribute("aTint", new THREE.InstancedBufferAttribute(new Float32Array(floes.length * 4), 4));
    const floeMesh = new THREE.InstancedMesh(floeGeo, houseMat, floes.length);
    floeMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    floeMesh.frustumCulled = false;
    scene.add(floeMesh);

    // Windmill rotors turn as a whole (their cubes are pieces in the village layout)
    const rotorAngle = new Float32Array(set.windmills.length).map(() => Math.random() * Math.PI);
    const zAxis = new THREE.Vector3(0, 0, 1);
    const qRotor = new THREE.Quaternion();
    sceneState.blocks = N + terrain.count + water.count + moonData.count + sunData.count + floes.length;

    /* ---------------- Sky ---------------- */
    const skyMat = keep(
      new THREE.ShaderMaterial({
        vertexShader: skyVertex,
        fragmentShader: skyFragment,
        uniforms: { uTime: { value: 0 }, uExposure: { value: 1 }, uMoonDir: moonDirU, ...dayU },
        side: THREE.BackSide,
        depthWrite: false,
        depthTest: false,
      })
    );
    const sky = new THREE.Mesh(keep(new THREE.SphereGeometry(200, 48, 24)), skyMat);
    sky.renderOrder = -1;
    scene.add(sky);

    /* ---------------- Drifting particles ---------------- */
    const SNOW = q.particles;
    const snowPos = new Float32Array(SNOW * 3);
    const snowRand = new Float32Array(SNOW);
    const snowSize = new Float32Array(SNOW);
    for (let i = 0; i < SNOW; i++) {
      snowPos[i * 3] = (Math.random() - 0.5) * 44;
      snowPos[i * 3 + 1] = Math.random() * 20;
      snowPos[i * 3 + 2] = -18 + Math.random() * 34;
      snowRand[i] = Math.random();
      snowSize[i] = 0.6 + Math.pow(Math.random(), 3) * 2.4;
    }
    const snowGeo = keep(new THREE.BufferGeometry());
    snowGeo.setAttribute("position", new THREE.BufferAttribute(snowPos, 3));
    snowGeo.setAttribute("aRand", new THREE.BufferAttribute(snowRand, 1));
    snowGeo.setAttribute("aSize", new THREE.BufferAttribute(snowSize, 1));
    const snowMat = keep(
      new THREE.ShaderMaterial({
        vertexShader: snowVertex,
        fragmentShader: snowFragment,
        uniforms: { uTime: { value: 0 }, uOffset: { value: 0 }, uPx: { value: dpr }, uOpacity: { value: 0.75 } },
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      })
    );
    const snow = new THREE.Points(snowGeo, snowMat);
    snow.frustumCulled = false;
    scene.add(snow);

    /* ---------------- Network edges ---------------- */
    const E = set.edges.length;
    const linePos = new Float32Array(E * 6);
    const lineT = new Float32Array(E * 2);
    const lineRand = new Float32Array(E * 2);
    for (let e = 0; e < E; e++) {
      lineT[e * 2 + 1] = 1;
      lineRand[e * 2] = lineRand[e * 2 + 1] = Math.random();
    }
    const lineGeo = keep(new THREE.BufferGeometry());
    const linePosAttr = new THREE.BufferAttribute(linePos, 3).setUsage(THREE.DynamicDrawUsage);
    lineGeo.setAttribute("position", linePosAttr);
    lineGeo.setAttribute("aT", new THREE.BufferAttribute(lineT, 1));
    lineGeo.setAttribute("aRand", new THREE.BufferAttribute(lineRand, 1));
    const lineMat = keep(
      new THREE.ShaderMaterial({
        vertexShader: lineVertex,
        fragmentShader: lineFragment,
        uniforms: { uTime: { value: 0 }, uOpacity: { value: 0 }, uNetC: { value: L.network.center.clone() }, uNetRot: { value: 0 } },
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      })
    );
    const lines = new THREE.LineSegments(lineGeo, lineMat);
    lines.frustumCulled = false;
    lines.visible = false;
    scene.add(lines);

    /* ---------------- Cursor ice-dust trail ---------------- */
    const DUST = q.dust;
    const dustCap = Math.max(1, DUST);
    const dustPos = new Float32Array(dustCap * 3).fill(999);
    const dustVel = new Float32Array(dustCap * 3);
    const dustAge = new Float32Array(dustCap).fill(99);
    const dustLife = new Float32Array(dustCap).fill(1);
    const dustAlpha = new Float32Array(dustCap);
    const dustSize = new Float32Array(dustCap);
    for (let i = 0; i < dustCap; i++) dustSize[i] = 0.5 + Math.pow(Math.random(), 2) * 1.8;
    const dustGeo = keep(new THREE.BufferGeometry());
    const dustPosAttr = new THREE.BufferAttribute(dustPos, 3).setUsage(THREE.DynamicDrawUsage);
    const dustAlphaAttr = new THREE.BufferAttribute(dustAlpha, 1).setUsage(THREE.DynamicDrawUsage);
    dustGeo.setAttribute("position", dustPosAttr);
    dustGeo.setAttribute("aAlpha", dustAlphaAttr);
    dustGeo.setAttribute("aSize", new THREE.BufferAttribute(dustSize, 1));
    const dustMat = keep(
      new THREE.ShaderMaterial({
        vertexShader: dustVertex,
        fragmentShader: dustFragment,
        uniforms: { uPx: { value: dpr } },
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      })
    );
    const dust = new THREE.Points(dustGeo, dustMat);
    dust.frustumCulled = false;
    dust.visible = DUST > 0;
    scene.add(dust);
    let dustHead = 0;
    /** Seconds until every spawned grain has faded — lets idle frames skip the update + upload */
    let dustBusy = 0;
    let emitCarry = 0;
    let lastBurst = sceneState.burst;
    const lastEmit = new THREE.Vector3(999, 999, 999);
    const rawW = new THREE.Vector3();
    const emitDir = new THREE.Vector3();
    const spawnDust = (at: THREE.Vector3, along: THREE.Vector3, speed: number) => {
      const i = dustHead++ % DUST;
      const f = Math.random();
      const o = i * 3;
      dustPos[o] = at.x - along.x * f + (Math.random() - 0.5) * 0.08;
      dustPos[o + 1] = at.y - along.y * f + (Math.random() - 0.5) * 0.08;
      dustPos[o + 2] = at.z - along.z * f + (Math.random() - 0.5) * 0.08;
      const th = Math.random() * Math.PI * 2;
      const ph = Math.acos(Math.random() * 2 - 1);
      const sp = speed * (0.3 + Math.random() * 0.7);
      dustVel[o] = Math.sin(ph) * Math.cos(th) * sp + along.x * 0.8;
      dustVel[o + 1] = Math.cos(ph) * sp + 0.15 + along.y * 0.8;
      dustVel[o + 2] = Math.sin(ph) * Math.sin(th) * sp + along.z * 0.8;
      dustAge[i] = 0;
      dustLife[i] = 0.7 + Math.random() * 1.1;
      dustBusy = 1.9;
    };

    /* ---------------- Post-processing ---------------- */
    const rt = keep(new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: q.msaa }));
    const composer = new EffectComposer(renderer, rt);
    composer.addPass(new RenderPass(scene, camera));
    const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), q.bloomStrength, 0.55, 0.72);
    composer.addPass(bloom);
    const finalPass = new ShaderPass(FinalShader);
    composer.addPass(finalPass);

    /* ---------------- Time of day ---------------- */
    // Real time in Amsterdam, or a fixed wall-clock time from ?time=HH:MM (previews, screenshots, tests)
    const fixedTime = parseTimeOverride(window.location.search);
    const lightDay = new THREE.Vector3();
    const moonFlat = new THREE.Vector3(MOON_DIR.x, 0, MOON_DIR.z).normalize();
    const moonEl = Math.asin(MOON_DIR.y);
    let daylightAt = 0;
    const updateDaylight = () => {
      daylightAt = performance.now();
      const date = fixedTime ? wallClockToday(fixedTime[0], fixedTime[1], profile.timezone) : new Date();
      const c = dayCycle(date);
      const d = c.day;
      const g = c.golden;
      dayU.uDay.value = d;
      dayU.uGolden.value = g;
      // The visible sun runs across the frame. Portrait screens get a narrower, lower arc:
      // there the hero labels span the full width just above where the sun would peak.
      const halfFov = Math.atan(Math.tan((camera.fov * DEG) / 2) * camera.aspect);
      const tall = camera.aspect < 1;
      const shown = sceneSun(date, Math.min(26 * DEG, halfFov * 0.7), tall ? 9.5 * DEG : 12.5 * DEG, tall ? 6.5 * DEG : 7 * DEG);
      dayU.uSunDir.value.set(...sunDirection(shown));
      sunMat.uniforms.uCenter.value.copy(dayU.uSunDir.value).multiplyScalar(SKY_R).add(SKY_ORIGIN);
      sun.visible = shown.elevation > -7 * DEG;
      // Key light: moonlight at night, the real sun's direction by day (never from below the horizon)
      lightDay.set(...sunDirection({ azimuth: c.azimuth, elevation: Math.max(c.elevation, 6 * DEG) }));
      dayU.uLightDir.value.copy(NIGHT_LIGHT).lerp(lightDay, d).normalize();
      // Light colours: neutral at night (the original look), warm sun + blue sky fill by day, orange at golden hour
      dayU.uLightCol.value.set(1 + 0.3 * d + 0.2 * g, 1 + 0.24 * d - 0.12 * g, 1 + 0.12 * d - 0.38 * g);
      dayU.uAmbient.value.set(1 + 1.3 * d + 0.25 * g, 1 + 1.45 * d + 0.05 * g, 1 + 1.7 * d - 0.1 * g);
      // The moon has set by sunrise (day = 0.5) and rises again in the dusk after sunset
      const el = moonEl - MOON_SET * smoothstep(0.05, 0.5, d);
      moonDir.copy(moonFlat).multiplyScalar(Math.cos(el)).setY(Math.sin(el));
      moonMat.uniforms.uCenter.value.copy(moonDir).multiplyScalar(SKY_R).add(SKY_ORIGIN);
      moon.visible = d < 0.97;
      fog.copy(NIGHT_FOG).lerp(DAY_FOG, d).lerp(GOLD_FOG, g * 0.55);
      // Less bloom by day so the white houses don't wash out
      bloom.strength = q.bloomStrength * (1 - 0.5 * d);
      bloom.threshold = 0.72 + 0.2 * d;
      // Lets the CSS give text over the bright daytime scene a stronger shadow
      document.documentElement.dataset.sky = d > 0.5 ? "day" : "night";
    };

    const resize = () => {
      const w = window.innerWidth;
      const h = window.innerHeight;
      renderer.setPixelRatio(dpr);
      renderer.setSize(w, h, false);
      composer.setPixelRatio(dpr);
      composer.setSize(w, h);
      bloom.setSize(w * dpr, h * dpr);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      finalPass.uniforms.uRes.value = [w * dpr, h * dpr];
      snowMat.uniforms.uPx.value = dpr;
      dustMat.uniforms.uPx.value = dpr;
      updateDaylight();
    };
    resize();
    window.addEventListener("resize", resize);

    /* ---------------- Interaction + animation state ---------------- */
    const pointerNdc = new THREE.Vector2();
    const raycaster = new THREE.Raycaster();
    const plane = new THREE.Plane();
    const mouseW = new THREE.Vector3(999, 999, 999);
    const mouseSmooth = new THREE.Vector2();

    const camPos = L.village.camPos.clone().add(new THREE.Vector3(0, 7, 9));
    const camTarget = L.village.camTarget.clone();
    const tmpCamPos = new THREE.Vector3();
    const tmpCamTarget = new THREE.Vector3();
    const camDir = new THREE.Vector3();
    let exposure = 0;
    let landAmt = 1;
    let lineOp = 0;
    let push = 1;
    let pressAmt = 0;
    let mouseOn = 0;
    let smoothVel = 0;
    let snowOffset = 0;

    const pA = new THREE.Vector3();
    const pB = new THREE.Vector3();
    const sA = new THREE.Vector3();
    const sB = new THREE.Vector3();
    const qA = new THREE.Quaternion();
    const qB = new THREE.Quaternion();
    const tq = new THREE.Quaternion();
    const tp = new THREE.Vector3();
    const ts = new THREE.Vector3();
    const qSpin = new THREE.Quaternion();
    const qTumble = new THREE.Quaternion();
    const yAxis = new THREE.Vector3(0, 1, 0);
    const mtx = new THREE.Matrix4();
    const tmp = new THREE.Vector3();

    const sample = (lay: Layout, i: number, time: number, outP: THREE.Vector3, outQ: THREE.Quaternion, outS: THREE.Vector3) => {
      outS.copy(lay.scale[i]);
      const mill = set.rotorMill[i];
      if (lay.rotors && mill >= 0) {
        const wm = set.windmills[mill];
        qRotor.setFromAxisAngle(zAxis, rotorAngle[mill]);
        outP.copy(set.rotorLocal[i]).applyQuaternion(qRotor).add(wm.hub).multiplyScalar(wm.scale).applyQuaternion(wm.yawQ).add(wm.base);
        outQ.copy(wm.yawQ).multiply(qRotor).multiply(set.rotorLocalQ[i]);
        return;
      }
      if (!lay.spin && !lay.bob) {
        outP.copy(lay.pos[i]);
        outQ.copy(lay.quat[i]);
        return;
      }
      qSpin.setFromAxisAngle(yAxis, lay.spin * time * motion);
      outP.copy(lay.pos[i]).sub(lay.center).applyQuaternion(qSpin).add(lay.center);
      if (lay.bob) outP.y += Math.sin(time * 0.6 * motion + set.rand[i] * 6.283) * lay.bob;
      outQ.copy(qSpin).multiply(lay.quat[i]);
    };

    const start = performance.now() / 1000;
    let last = performance.now();
    let raf = 0;
    let running = true;
    /** Smoothed time between frames (ms) */
    let frameMs = 1000 / 60;
    let firstFrame = true;
    let revealAt = -1;
    let wasPressed = false;
    let hiAmt = 0;
    const hiC = new THREE.Vector3();

    const tick = () => {
      raf = requestAnimationFrame(tick);
      if (!running) return;
      const nowMs = performance.now();
      // Where frames are already slow (software rendering, very weak GPUs), skip drawing while
      // a page wipe covers the screen (nothing visible is lost): the wipe can only advance one
      // frame at a time, so with a heavy frame behind every step it would hold navigation up
      if (sceneState.transition && frameMs > 70) {
        last = nowMs;
        return;
      }
      const interval = nowMs - last;
      const dt = Math.min(interval / 1000, 0.1);
      last = nowMs;
      const time = nowMs / 1000 - start;
      frameMs += (Math.min(interval, 250) - frameMs) * 0.1;

      // The sun moves ~0.25° a minute: refreshing every 10 s is plenty
      if (nowMs - daylightAt > 10_000) updateDaylight();

      const vh = window.innerHeight;
      const { a, b, t } = resolveBlend(window.scrollY, vh);
      const A = L[a];
      const B = L[b];
      const camT = easeInOut(clamp01(t));
      const moving = a !== b;

      /* Camera */
      tmpCamPos.lerpVectors(A.camPos, B.camPos, camT);
      tmpCamTarget.lerpVectors(A.camTarget, B.camTarget, camT);
      const aspect = window.innerWidth / vh;
      if (aspect < 1) {
        tmpCamTarget.x *= aspect;
        tmp.copy(tmpCamPos).sub(tmpCamTarget).multiplyScalar(1 + (1 - aspect) * 1.1);
        tmpCamPos.copy(tmpCamTarget).add(tmp);
      }
      const p = sceneState.pointer;
      mouseSmooth.x += (p.x - mouseSmooth.x) * damp(dt, 4);
      mouseSmooth.y += (p.y - mouseSmooth.y) * damp(dt, 4);
      tmpCamPos.x += mouseSmooth.x * 0.7 * motion + Math.sin(time * 0.1) * 0.3 * motion;
      tmpCamPos.y += mouseSmooth.y * 0.35 * motion;
      camPos.lerp(tmpCamPos, damp(dt, 2.6));
      camTarget.lerp(tmpCamTarget, damp(dt, 3));
      camera.position.copy(camPos);
      camera.lookAt(camTarget);
      sky.position.copy(camera.position);

      /* Scalars */
      const mixS = (k: "exposure" | "land" | "lines" | "push") => A[k] + (B[k] - A[k]) * camT;
      // Pieces wait in the sky until the loader lifts, then the village drops into place
      if (revealAt < 0 && ui.get().ready) revealAt = time;
      const intro = revealAt < 0 ? 0.35 : 0.35 + 0.65 * clamp01((time - revealAt) / 1.4);
      const hold = revealAt < 0;
      exposure += (mixS("exposure") * intro - exposure) * damp(dt, 3);
      landAmt += (mixS("land") - landAmt) * damp(dt, 2.4);
      lineOp += (mixS("lines") - lineOp) * damp(dt, 2.5);
      push += (mixS("push") - push) * damp(dt, 3);
      pressAmt += ((sceneState.press ? 1 : 0) - pressAmt) * damp(dt, sceneState.press ? 2.2 : 3.2);
      // (on the land clock, like uTime, so the shock ring also runs with reduced motion)
      if (sceneState.press && !wasPressed) landU.uShockT.value = time * motion;
      wasPressed = sceneState.press;

      /* Pointer → world point on a plane through the camera target */
      const fine = !coarse && nowMs - p.lastMove < 4000;
      mouseOn += ((fine ? 1 : 0) - mouseOn) * damp(dt, 3);
      pointerNdc.set(mouseSmooth.x, mouseSmooth.y);
      raycaster.setFromCamera(pointerNdc, camera);
      camera.getWorldDirection(camDir);
      plane.setFromNormalAndCoplanarPoint(camDir, camTarget);
      if (!raycaster.ray.intersectPlane(plane, mouseW)) mouseW.set(999, 999, 999);

      /* Ice dust: trail behind the cursor + a burst on click */
      if (DUST) {
        pointerNdc.set(p.x, p.y);
        raycaster.setFromCamera(pointerNdc, camera);
        const hit = raycaster.ray.intersectPlane(plane, rawW) !== null;
        if (hit && fine && !reduced) {
          if (lastEmit.x > 900) lastEmit.copy(rawW);
          emitDir.subVectors(rawW, lastEmit);
          emitCarry += Math.min(emitDir.length() * 24, 16);
          while (emitCarry >= 1) {
            emitCarry -= 1;
            spawnDust(rawW, emitDir, 0.45);
          }
          lastEmit.copy(rawW);
        } else lastEmit.set(999, 999, 999);
        if (sceneState.burst !== lastBurst) {
          lastBurst = sceneState.burst;
          if (hit && !reduced) {
            emitDir.set(0, 0, 0);
            for (let k = 0; k < 42; k++) spawnDust(rawW, emitDir, 2.6);
          }
        }
        // Simulate + upload only while grains are alive; once all have faded the buffers stay blank
        if (dustBusy > 0) {
          dustBusy -= dt;
          for (let i = 0; i < DUST; i++) {
            if (dustAge[i] >= dustLife[i]) {
              dustAlpha[i] = 0;
              continue;
            }
            dustAge[i] += dt;
            const o = i * 3;
            const drag = Math.exp(-dt * 2.2);
            dustVel[o] *= drag;
            dustVel[o + 1] = dustVel[o + 1] * drag - 0.35 * dt;
            dustVel[o + 2] *= drag;
            dustPos[o] += dustVel[o] * dt;
            dustPos[o + 1] += dustVel[o + 1] * dt;
            dustPos[o + 2] += dustVel[o + 2] * dt;
            dustAlpha[i] = Math.pow(Math.max(0, 1 - dustAge[i] / dustLife[i]), 1.5);
          }
          dustPosAttr.needsUpdate = true;
          dustAlphaAttr.needsUpdate = true;
        }
      }

      /* Pieces */
      const settling = revealAt >= 0 && time - revealAt < 3 ? 0.55 : 1;
      for (let i = 0; i < N; i++) {
        const ti = moving ? easeInOut(clamp01((t - set.delay[i]) / (1 - 0.35))) : 0;
        sample(A, i, time, pA, qA, sA);
        sample(B, i, time, pB, qB, sB);
        tp.lerpVectors(pA, pB, ti);
        tq.slerpQuaternions(qA, qB, ti);
        ts.lerpVectors(sA, sB, ti);
        const iceT = A.ice + (B.ice - A.ice) * ti;

        // Flying arc + tumble mid-transition, and hold-to-shatter
        const arc = moving ? Math.sin(Math.PI * ti) : 0;
        const fling = arc * (0.6 + set.rand[i] * 1.6) + pressAmt * (1.2 + set.rand[i] * 4.2);
        if (fling > 0.001) {
          tp.addScaledVector(set.burstDir[i], fling);
          qTumble.setFromAxisAngle(set.tumbleAxis[i], fling * 0.9);
          tq.multiply(qTumble);
        }

        if (hold) tp.y += 14 + set.rand[i] * 6;

        // Pointer: formations scatter away, the village gently levitates
        if (mouseOn > 0.01) {
          tmp.subVectors(tp, mouseW);
          const d = tmp.length();
          if (d < 2.2 && d > 1e-4) {
            const f = (1 - d / 2.2) ** 2 * push * mouseOn * 0.9 * motion;
            if (iceT < 0.5) tp.y += f;
            else tp.addScaledVector(tmp, f / d);
          }
        }

        const k = damp(dt, set.lambda[i] * settling);
        cur[i].lerp(tp, k);
        curQ[i].slerp(tq, k);
        curS[i].lerp(ts, k);
        curMix[i] += (Math.max(iceT, pressAmt * 0.85) - curMix[i]) * k;
        mtx.compose(cur[i], curQ[i], curS[i]);
        const li = set.local[i];
        if (set.kinds[i] === KIND_DOME) {
          domes.setMatrixAt(li, mtx);
          domeMix[li] = curMix[i];
        } else {
          houses.setMatrixAt(li, mtx);
          boxMix[li] = curMix[i];
        }
      }
      houses.instanceMatrix.needsUpdate = true;
      domes.instanceMatrix.needsUpdate = true;
      boxMixAttr.needsUpdate = true;
      domeMixAttr.needsUpdate = true;

      /* Network edges follow the pieces */
      lines.visible = lineOp > 0.01;
      if (lines.visible) {
        for (let e = 0; e < E; e++) {
          const [i0, i1] = set.edges[e];
          const o = e * 6;
          linePos[o] = cur[i0].x;
          linePos[o + 1] = cur[i0].y;
          linePos[o + 2] = cur[i0].z;
          linePos[o + 3] = cur[i1].x;
          linePos[o + 4] = cur[i1].y;
          linePos[o + 5] = cur[i1].z;
        }
        linePosAttr.needsUpdate = true;
      }

      /* Scroll velocity drives particles, windmills + glitch */
      smoothVel += (sceneState.velocity - smoothVel) * damp(dt, 5);
      sceneState.velocity *= Math.exp(-dt * 4);
      const speed = Math.abs(smoothVel);
      snowOffset += smoothVel * dt * 0.0022;
      for (let m = 0; m < rotorAngle.length; m++) rotorAngle[m] -= dt * (0.55 + Math.min(speed / 500, 6)) * motion;

      // Floes ride the water blocks' swell; when the land goes holographic (or on hold-to-shatter)
      // they lift into the air
      const landT = time * motion;
      const lift = Math.max(1 - landAmt, pressAmt);
      for (let i = 0; i < floes.length; i++) {
        const f = floes[i];
        tp.copy(f.pos);
        tp.y += waveHeight(f.pos.x, f.pos.z, landT) + Math.sin(time * 0.8 * motion + f.rand * 6.28) * 0.02 + lift * (1.2 + f.rand * 3.5);
        tq.setFromAxisAngle(yAxis, f.yaw + time * 0.05 * f.rand * motion + lift * f.rand * 3);
        floeMesh.setMatrixAt(i, mtx.compose(tp, tq, f.size));
      }
      floeMesh.instanceMatrix.needsUpdate = true;

      /* Telemetry for the HUD */
      sceneState.formation = camT < 0.5 ? a : b;
      sceneState.cam.x = camPos.x;
      sceneState.cam.y = camPos.y;
      sceneState.cam.z = camPos.z;

      /* Uniforms */
      const pulseT = sceneState.pulse - start;
      // Neural-net activations are visible while the network formation is on screen
      const netW = (a === "network" ? 1 - camT : 0) + (b === "network" && a !== b ? camT : 0);
      const netRot = L.network.spin * time * motion;
      lineMat.uniforms.uNetRot.value = netRot;
      // Hovering a domain card lights up its cluster (follows the formation's spin)
      const clW = (a === "clusters" ? 1 - camT : 0) + (b === "clusters" && a !== b ? camT : 0);
      const hi = sceneState.highlight;
      const anchors = L.clusters.anchors;
      if (hi >= 0 && anchors && anchors[hi]) {
        qSpin.setFromAxisAngle(yAxis, L.clusters.spin * time * motion);
        hiC.copy(anchors[hi]).sub(L.clusters.center).applyQuaternion(qSpin).add(L.clusters.center);
      }
      hiAmt += ((hi >= 0 ? clW : 0) - hiAmt) * damp(dt, 6);
      for (const m of pieceMats) {
        const u = m.uniforms;
        u.uHi.value = hiAmt;
        u.uHiC.value.copy(hiC);
        u.uNet.value = netW;
        u.uNetRot.value = netRot;
        u.uTime.value = time;
        u.uCam.value.copy(camera.position);
        u.uMouse.value.copy(mouseW);
        u.uMouseOn.value = mouseOn;
        u.uPulseT.value = pulseT;
        u.uPulseC.value.copy(camTarget);
        u.uExposure.value = exposure;
      }
      landU.uTime.value = time * motion;
      landU.uLand.value = landAmt;
      landU.uExposure.value = exposure;
      landU.uCam.value.copy(camera.position);
      terrU.uGrow.value = revealAt < 0 ? 0 : clamp01((time - revealAt) / 2.6);
      terrU.uAmp.value = 1 + Math.min(speed / 2500, 1.5);
      terrU.uCenter.value.copy(camTarget);
      terrU.uMouse.value.copy(mouseW);
      terrU.uMouseOn.value = mouseOn;
      terrU.uPulseT.value = pulseT * motion;
      breakU.uBreak.value = pressAmt;
      // The moon and sun burst into rings in the holographic formations and on hold-to-shatter
      const burst = Math.max(clamp01((1 - landAmt) * 1.15), pressAmt);
      moonMat.uniforms.uTime.value = time * motion;
      moonMat.uniforms.uBurst.value = burst;
      moonMat.uniforms.uExposure.value = 0.75 + 0.25 * exposure;
      sunMat.uniforms.uTime.value = time * motion;
      sunMat.uniforms.uBurst.value = burst;
      sunMat.uniforms.uExposure.value = 0.85 + 0.15 * exposure;
      skyMat.uniforms.uTime.value = time * motion;
      skyMat.uniforms.uExposure.value = 0.6 + exposure * 0.4;
      snowMat.uniforms.uTime.value = time * motion;
      snowMat.uniforms.uOffset.value = snowOffset;
      snowMat.uniforms.uOpacity.value = 0.6 * intro * (1 - 0.35 * dayU.uDay.value);
      lineMat.uniforms.uTime.value = time;
      lineMat.uniforms.uOpacity.value = lineOp * exposure;
      finalPass.uniforms.uTime.value = time;
      finalPass.uniforms.uAberr.value = 0.0012 + Math.min(speed / 3500, 1) * 0.008 * motion;
      finalPass.uniforms.uGlitch.value = reduced ? 0 : clamp01((speed - 2200) / 4000);

      composer.render(dt);

      if (firstFrame) {
        firstFrame = false;
        sceneState.ready = true;
      }
    };

    // Compile shaders up-front so the first visible frame doesn't hitch
    try {
      renderer.compile(scene, camera);
    } catch {
      /* compile is an optimisation only */
    }
    raf = requestAnimationFrame(tick);

    // Render only while the tab is visible and the WebGL context is alive
    let contextLost = false;
    const onVisibility = () => {
      running = !document.hidden && !contextLost;
      last = performance.now();
    };
    document.addEventListener("visibilitychange", onVisibility);
    const onLost = (e: Event) => {
      e.preventDefault();
      contextLost = true;
      running = false;
    };
    const onRestored = () => {
      contextLost = false;
      running = !document.hidden;
      last = performance.now();
    };
    canvas.addEventListener("webglcontextlost", onLost);
    canvas.addEventListener("webglcontextrestored", onRestored);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
      document.removeEventListener("visibilitychange", onVisibility);
      canvas.removeEventListener("webglcontextlost", onLost);
      canvas.removeEventListener("webglcontextrestored", onRestored);
      delete document.documentElement.dataset.sky;
      disposables.forEach((d) => d.dispose());
      bloom.dispose();
      composer.dispose();
      renderer.dispose();
    };
  }, []);

  return <canvas ref={canvasRef} className="gl-canvas" aria-hidden="true" />;
}
