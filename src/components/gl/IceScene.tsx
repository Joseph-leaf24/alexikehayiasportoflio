import { useEffect, useRef } from "react";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import { sceneState, ui, type LayoutKey } from "@/lib/store";
import { prefersReducedMotion } from "@/lib/smoothScroll";
import { buildLayouts, KIND_DOME, terrainHeight, type Layout } from "./layouts";
import {
  FinalShader,
  dustFragment,
  dustVertex,
  lineFragment,
  lineVertex,
  pieceFragment,
  pieceVertex,
  propFragment,
  propVertex,
  seaFragment,
  skyFragment,
  skyVertex,
  snowFragment,
  snowVertex,
  terrainFragment,
  terrainVertex,
} from "./shaders";

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const damp = (dt: number, lambda: number) => 1 - Math.exp(-dt * lambda);
const MOON_DIR = new THREE.Vector3(0.42, 0.18, -1).normalize();

/** Which two layouts we are between, and how far (0..1), for the current scroll. */
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
    const small = window.innerWidth < 768;
    const motion = reduced ? 0.25 : 1;
    let dpr = Math.min(window.devicePixelRatio || 1, small ? 1.25 : 1.6);

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
    const fog = new THREE.Vector3(0.012, 0.018, 0.028);

    /* ---------------- Village pieces ---------------- */
    const set = buildLayouts();
    const N = set.count;
    sceneState.blocks = N;
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

    /* ---------------- Landscape: terrain, sea, windmills ---------------- */
    const landU = {
      uTime: { value: 0 },
      uLand: { value: 1 },
      uExposure: { value: 0 },
      uCam: { value: new THREE.Vector3() },
      uFog: { value: fog },
      uMoonDir: { value: MOON_DIR },
    };
    const terrGeo = keep(new THREE.PlaneGeometry(76, 48, small ? 190 : 300, small ? 120 : 190));
    terrGeo.rotateX(-Math.PI / 2);
    terrGeo.translate(0, 0, -9);
    const tp0 = terrGeo.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < tp0.count; i++) tp0.setY(i, terrainHeight(tp0.getX(i), tp0.getZ(i)));
    terrGeo.computeVertexNormals();
    const terrMat = keep(new THREE.ShaderMaterial({ vertexShader: terrainVertex, fragmentShader: terrainFragment, uniforms: landU }));
    scene.add(new THREE.Mesh(terrGeo, terrMat));

    const seaGeo = keep(new THREE.PlaneGeometry(420, 420));
    seaGeo.rotateX(-Math.PI / 2);
    const seaMat = keep(new THREE.ShaderMaterial({ vertexShader: terrainVertex, fragmentShader: seaFragment, uniforms: landU }));
    scene.add(new THREE.Mesh(seaGeo, seaMat));

    const propMat = (r: number, g: number, b: number) =>
      keep(
        new THREE.ShaderMaterial({
          vertexShader: propVertex,
          fragmentShader: propFragment,
          uniforms: { ...landU, uColor: { value: new THREE.Vector3(r, g, b) } },
          side: THREE.DoubleSide,
        })
      );
    const whiteM = propMat(0.86, 0.89, 0.95);
    const capM = propMat(0.16, 0.12, 0.11);
    const sailM = propMat(0.9, 0.87, 0.8);
    const woodM = propMat(0.2, 0.15, 0.12);
    const doorM = propMat(0.06, 0.18, 0.48);
    const bodyGeo = keep(new THREE.CylinderGeometry(0.42, 0.5, 1.7, 24));
    bodyGeo.translate(0, 0.85, 0);
    const capGeo = keep(new THREE.ConeGeometry(0.52, 0.6, 24));
    capGeo.translate(0, 2.0, 0);
    const doorGeo = keep(new THREE.BoxGeometry(0.24, 0.42, 0.08));
    doorGeo.translate(0, 0.21, 0.47);
    const spokeGeo = keep(new THREE.BoxGeometry(0.035, 1.3, 0.035));
    spokeGeo.translate(0, 0.65, 0);
    // Narrow triangular canvas sails, like the Cycladic windmills
    const sailGeo = keep(new THREE.BufferGeometry());
    sailGeo.setAttribute("position", new THREE.Float32BufferAttribute([0, 0.28, 0, 0, 1.28, 0, 0.2, 1.16, 0], 3));
    sailGeo.computeVertexNormals();
    const hubGeo = keep(new THREE.SphereGeometry(0.08, 12, 8));
    const rotors: THREE.Group[] = [];
    for (const wm of set.windmills) {
      const g = new THREE.Group();
      g.position.copy(wm.pos);
      g.rotation.y = wm.yaw;
      g.scale.setScalar(wm.scale);
      g.add(new THREE.Mesh(bodyGeo, whiteM), new THREE.Mesh(capGeo, capM), new THREE.Mesh(doorGeo, doorM));
      const rotor = new THREE.Group();
      rotor.position.set(0, 1.62, 0.54);
      rotor.add(new THREE.Mesh(hubGeo, woodM));
      for (let k = 0; k < 8; k++) {
        const arm = new THREE.Group();
        arm.rotation.z = (k * Math.PI) / 4;
        arm.add(new THREE.Mesh(spokeGeo, woodM), new THREE.Mesh(sailGeo, sailM));
        rotor.add(arm);
      }
      rotor.rotation.z = Math.random() * Math.PI;
      g.add(rotor);
      rotors.push(rotor);
      scene.add(g);
    }

    /* ---------------- Sky ---------------- */
    const skyMat = keep(
      new THREE.ShaderMaterial({
        vertexShader: skyVertex,
        fragmentShader: skyFragment,
        uniforms: { uTime: { value: 0 }, uExposure: { value: 1 }, uMoonDir: { value: MOON_DIR } },
        side: THREE.BackSide,
        depthWrite: false,
        depthTest: false,
      })
    );
    const sky = new THREE.Mesh(keep(new THREE.SphereGeometry(200, 48, 24)), skyMat);
    sky.renderOrder = -1;
    scene.add(sky);

    /* ---------------- Drifting particles ---------------- */
    const SNOW = small ? 700 : 1800;
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
        uniforms: { uTime: { value: 0 }, uOpacity: { value: 0 } },
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
    const DUST = coarse ? 0 : 480;
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
    };

    /* ---------------- Post-processing ---------------- */
    const rt = keep(new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: small ? 0 : 4 }));
    const composer = new EffectComposer(renderer, rt);
    composer.addPass(new RenderPass(scene, camera));
    const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), small ? 0.45 : 0.6, 0.55, 0.72);
    composer.addPass(bloom);
    const finalPass = new ShaderPass(FinalShader);
    composer.addPass(finalPass);

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
    let slowFrames = 0;
    let firstFrame = true;
    let revealAt = -1;

    const tick = () => {
      raf = requestAnimationFrame(tick);
      if (!running) return;
      const nowMs = performance.now();
      const dt = Math.min((nowMs - last) / 1000, 0.1);
      last = nowMs;
      const time = nowMs / 1000 - start;

      // Adaptive resolution: drop DPR if we're consistently slow
      if (time > 3 && dpr > 0.75) {
        slowFrames = dt > 1 / 38 ? slowFrames + 1 : Math.max(0, slowFrames - 1);
        if (slowFrames > 90) {
          dpr = Math.max(0.75, dpr - 0.25);
          slowFrames = 0;
          resize();
        }
      }

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
      for (const r of rotors) r.rotation.z -= dt * (0.55 + Math.min(speed / 500, 6)) * motion;

      /* Telemetry for the HUD */
      sceneState.formation = camT < 0.5 ? a : b;
      sceneState.cam.x = camPos.x;
      sceneState.cam.y = camPos.y;
      sceneState.cam.z = camPos.z;

      /* Uniforms */
      const pulseT = sceneState.pulse - start;
      for (const m of pieceMats) {
        const u = m.uniforms;
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
      skyMat.uniforms.uTime.value = time * motion;
      skyMat.uniforms.uExposure.value = 0.6 + exposure * 0.4;
      snowMat.uniforms.uTime.value = time * motion;
      snowMat.uniforms.uOffset.value = snowOffset;
      snowMat.uniforms.uOpacity.value = 0.6 * intro;
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

    const onVisibility = () => {
      running = !document.hidden;
      last = performance.now();
    };
    document.addEventListener("visibilitychange", onVisibility);
    const onLost = (e: Event) => {
      e.preventDefault();
      running = false;
    };
    const onRestored = () => {
      running = true;
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
      disposables.forEach((d) => d.dispose());
      bloom.dispose();
      composer.dispose();
      renderer.dispose();
    };
  }, []);

  return <canvas ref={canvasRef} className="gl-canvas" aria-hidden="true" />;
}
