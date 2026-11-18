import { useEffect, useRef } from "react";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import { sceneState, ui, type LayoutKey } from "@/lib/store";
import { prefersReducedMotion } from "@/lib/smoothScroll";
import { BLOCK, buildLayouts, type Layout } from "./layouts";
import {
  FinalShader,
  groundFragment,
  groundVertex,
  iceFragment,
  iceVertex,
  lineFragment,
  lineVertex,
  skyFragment,
  skyVertex,
  snowFragment,
  snowVertex,
} from "./shaders";

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const damp = (dt: number, lambda: number) => 1 - Math.exp(-dt * lambda);

/** Which two layouts we are between, and how far (0..1), for the current scroll. */
function resolveBlend(scrollY: number, vh: number): { a: LayoutKey; b: LayoutKey; t: number } {
  if (sceneState.override) return { a: sceneState.override, b: sceneState.override, t: 0 };
  const stops = sceneState.stops;
  if (!stops.length) return { a: "dome", b: "dome", t: 0 };
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
    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);

    /* ---------------- Layouts + ice blocks ---------------- */
    const set = buildLayouts();
    const N = set.count;
    sceneState.blocks = N;
    const L = set.layouts;

    const geo = new RoundedBoxGeometry(BLOCK.x, BLOCK.y, BLOCK.z, 2, 0.045);
    geo.setAttribute("aRand", new THREE.InstancedBufferAttribute(set.rand, 1));
    const fog = new THREE.Vector3(0.012, 0.018, 0.028);
    const iceMat = new THREE.ShaderMaterial({
      vertexShader: iceVertex,
      fragmentShader: iceFragment,
      uniforms: {
        uCam: { value: new THREE.Vector3() },
        uTime: { value: 0 },
        uMouse: { value: new THREE.Vector3(999, 999, 999) },
        uMouseOn: { value: 0 },
        uPulseT: { value: -100 },
        uHalf: { value: BLOCK.clone().multiplyScalar(0.5) },
        uExposure: { value: 1 },
        uFog: { value: fog },
      },
    });
    const mesh = new THREE.InstancedMesh(geo, iceMat, N);
    mesh.frustumCulled = false;
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    scene.add(mesh);

    // Current (damped) per-instance transforms, start in the dome formation
    const cur = L.dome.pos.map((p) => p.clone().add(new THREE.Vector3(0, 9 + Math.random() * 6, 0)));
    const curQ = L.dome.quat.map((q) => q.clone());
    const curS = new Float32Array(N).fill(1);

    /* ---------------- Ground ---------------- */
    const groundMat = new THREE.ShaderMaterial({
      vertexShader: groundVertex,
      fragmentShader: groundFragment,
      uniforms: { uTime: { value: 0 }, uOpacity: { value: 1 } },
      transparent: true,
      depthWrite: false,
    });
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), groundMat);
    ground.rotation.x = -Math.PI / 2;
    scene.add(ground);

    /* ---------------- Sky ---------------- */
    const skyMat = new THREE.ShaderMaterial({
      vertexShader: skyVertex,
      fragmentShader: skyFragment,
      uniforms: { uTime: { value: 0 }, uExposure: { value: 1 } },
      side: THREE.BackSide,
      depthWrite: false,
      depthTest: false,
    });
    const sky = new THREE.Mesh(new THREE.SphereGeometry(90, 48, 24), skyMat);
    sky.renderOrder = -1;
    scene.add(sky);

    /* ---------------- Snow ---------------- */
    const SNOW = small ? 600 : 1600;
    const snowPos = new Float32Array(SNOW * 3);
    const snowRand = new Float32Array(SNOW);
    const snowSize = new Float32Array(SNOW);
    for (let i = 0; i < SNOW; i++) {
      snowPos[i * 3] = (Math.random() - 0.5) * 32;
      snowPos[i * 3 + 1] = Math.random() * 17;
      snowPos[i * 3 + 2] = (Math.random() - 0.5) * 30;
      snowRand[i] = Math.random();
      snowSize[i] = 0.6 + Math.pow(Math.random(), 3) * 2.4;
    }
    const snowGeo = new THREE.BufferGeometry();
    snowGeo.setAttribute("position", new THREE.BufferAttribute(snowPos, 3));
    snowGeo.setAttribute("aRand", new THREE.BufferAttribute(snowRand, 1));
    snowGeo.setAttribute("aSize", new THREE.BufferAttribute(snowSize, 1));
    const snowMat = new THREE.ShaderMaterial({
      vertexShader: snowVertex,
      fragmentShader: snowFragment,
      uniforms: { uTime: { value: 0 }, uOffset: { value: 0 }, uPx: { value: dpr }, uOpacity: { value: 0.75 } },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const snow = new THREE.Points(snowGeo, snowMat);
    snow.frustumCulled = false;
    scene.add(snow);

    /* ---------------- Network edges ---------------- */
    const E = set.edges.length;
    const linePos = new Float32Array(E * 6);
    const lineT = new Float32Array(E * 2);
    const lineRand = new Float32Array(E * 2);
    set.edges.forEach((_, e) => {
      lineT[e * 2] = 0;
      lineT[e * 2 + 1] = 1;
      const r = Math.random();
      lineRand[e * 2] = r;
      lineRand[e * 2 + 1] = r;
    });
    const lineGeo = new THREE.BufferGeometry();
    const linePosAttr = new THREE.BufferAttribute(linePos, 3);
    linePosAttr.setUsage(THREE.DynamicDrawUsage);
    lineGeo.setAttribute("position", linePosAttr);
    lineGeo.setAttribute("aT", new THREE.BufferAttribute(lineT, 1));
    lineGeo.setAttribute("aRand", new THREE.BufferAttribute(lineRand, 1));
    const lineMat = new THREE.ShaderMaterial({
      vertexShader: lineVertex,
      fragmentShader: lineFragment,
      uniforms: { uTime: { value: 0 }, uOpacity: { value: 0 } },
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const lines = new THREE.LineSegments(lineGeo, lineMat);
    lines.frustumCulled = false;
    lines.visible = false;
    scene.add(lines);

    /* ---------------- Post-processing ---------------- */
    const rt = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: small ? 0 : 4 });
    const composer = new EffectComposer(renderer, rt);
    composer.addPass(new RenderPass(scene, camera));
    const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), small ? 0.45 : 0.6, 0.55, 0.6);
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
    };
    resize();
    window.addEventListener("resize", resize);

    /* ---------------- Interaction ---------------- */
    const pointerNdc = new THREE.Vector2();
    const raycaster = new THREE.Raycaster();
    const plane = new THREE.Plane();
    const mouseW = new THREE.Vector3(999, 999, 999);
    const mouseSmooth = new THREE.Vector2();

    /* ---------------- Animation state ---------------- */
    const camPos = L.dome.camPos.clone().add(new THREE.Vector3(0, 6, 8));
    const camTarget = L.dome.camTarget.clone();
    const tmpCamPos = new THREE.Vector3();
    const tmpCamTarget = new THREE.Vector3();
    const camDir = new THREE.Vector3();
    let exposure = 0;
    let groundOp = 1;
    let lineOp = 0;
    let push = 1;
    let pressAmt = 0;
    let mouseOn = 0;
    let smoothVel = 0;
    let snowOffset = 0;

    const pA = new THREE.Vector3();
    const pB = new THREE.Vector3();
    const qA = new THREE.Quaternion();
    const qB = new THREE.Quaternion();
    const tq = new THREE.Quaternion();
    const tp = new THREE.Vector3();
    const qSpin = new THREE.Quaternion();
    const qTumble = new THREE.Quaternion();
    const yAxis = new THREE.Vector3(0, 1, 0);
    const scaleV = new THREE.Vector3();
    const mtx = new THREE.Matrix4();
    const tmp = new THREE.Vector3();

    const sample = (lay: Layout, i: number, time: number, outP: THREE.Vector3, outQ: THREE.Quaternion) => {
      const angle = lay.spin * time * motion;
      qSpin.setFromAxisAngle(yAxis, angle);
      outP.copy(lay.pos[i]).sub(lay.center).applyQuaternion(qSpin).add(lay.center);
      if (lay.bob) outP.y += Math.sin(time * 0.6 * motion + set.rand[i] * 6.283) * lay.bob;
      outQ.copy(qSpin).multiply(lay.quat[i]);
      return lay.scale[i];
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
      tmpCamPos.x += mouseSmooth.x * 0.55 * motion + Math.sin(time * 0.1) * 0.25 * motion;
      tmpCamPos.y += mouseSmooth.y * 0.3 * motion;
      camPos.lerp(tmpCamPos, damp(dt, 2.6));
      camTarget.lerp(tmpCamTarget, damp(dt, 3));
      camera.position.copy(camPos);
      camera.lookAt(camTarget);
      sky.position.copy(camera.position);

      /* Scalars */
      const mixS = (k: "exposure" | "ground" | "lines" | "push") => A[k] + (B[k] - A[k]) * camT;
      // Blocks wait in the sky until the loader lifts, then the igloo assembles
      if (revealAt < 0 && ui.get().ready) revealAt = time;
      const intro = revealAt < 0 ? 0.35 : 0.35 + 0.65 * clamp01((time - revealAt) / 1.4);
      const hold = revealAt < 0 ? 1 : 0;
      exposure += (mixS("exposure") * intro - exposure) * damp(dt, 3);
      groundOp += (mixS("ground") - groundOp) * damp(dt, 3);
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

      /* Instances */
      for (let i = 0; i < N; i++) {
        const ti = moving ? easeInOut(clamp01((t - set.delay[i]) / (1 - 0.35))) : 0;
        const sA = sample(A, i, time, pA, qA);
        const sB = sample(B, i, time, pB, qB);
        tp.lerpVectors(pA, pB, ti);
        tq.slerpQuaternions(qA, qB, ti);
        let s = sA + (sB - sA) * ti;

        // Flying arc + tumble mid-transition
        const arc = moving ? Math.sin(Math.PI * ti) : 0;
        const fling = arc * (0.6 + set.rand[i] * 1.4) + pressAmt * (1.2 + set.rand[i] * 4.2);
        if (fling > 0.001) {
          tp.addScaledVector(set.burstDir[i], fling);
          qTumble.setFromAxisAngle(set.tumbleAxis[i], fling * 0.9);
          tq.multiply(qTumble);
          s *= 1 - pressAmt * 0.15;
        }

        if (hold) tp.y += 14 + set.rand[i] * 6;

        // Pointer repulsion
        if (mouseOn > 0.01) {
          tmp.subVectors(tp, mouseW);
          const d = tmp.length();
          if (d < 2.2 && d > 1e-4) {
            const f = (1 - d / 2.2) ** 2 * push * mouseOn * 0.9 * motion;
            tp.addScaledVector(tmp, f / d);
          }
        }

        const k = damp(dt, set.lambda[i] * (revealAt >= 0 && time - revealAt < 3 ? 0.55 : 1));
        cur[i].lerp(tp, k);
        curQ[i].slerp(tq, k);
        curS[i] += (s - curS[i]) * k;
        scaleV.setScalar(curS[i]);
        mtx.compose(cur[i], curQ[i], scaleV);
        mesh.setMatrixAt(i, mtx);
      }
      mesh.instanceMatrix.needsUpdate = true;

      /* Network edges follow the blocks */
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

      /* Scroll velocity drives snow + glitch */
      smoothVel += (sceneState.velocity - smoothVel) * damp(dt, 5);
      sceneState.velocity *= Math.exp(-dt * 4);
      const speed = Math.abs(smoothVel);
      snowOffset += smoothVel * dt * 0.0022;

      /* Uniforms */
      const pulseT = sceneState.pulse - start;
      iceMat.uniforms.uTime.value = time;
      iceMat.uniforms.uCam.value.copy(camera.position);
      iceMat.uniforms.uMouse.value.copy(mouseW);
      iceMat.uniforms.uMouseOn.value = mouseOn;
      iceMat.uniforms.uPulseT.value = pulseT;
      iceMat.uniforms.uExposure.value = exposure;
      groundMat.uniforms.uTime.value = time;
      groundMat.uniforms.uOpacity.value = groundOp * exposure;
      skyMat.uniforms.uTime.value = time * motion;
      skyMat.uniforms.uExposure.value = 0.6 + exposure * 0.4;
      snowMat.uniforms.uTime.value = time * motion;
      snowMat.uniforms.uOffset.value = snowOffset;
      snowMat.uniforms.uOpacity.value = 0.75 * intro;
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
      geo.dispose();
      iceMat.dispose();
      groundMat.dispose();
      ground.geometry.dispose();
      skyMat.dispose();
      sky.geometry.dispose();
      snowGeo.dispose();
      snowMat.dispose();
      lineGeo.dispose();
      lineMat.dispose();
      bloom.dispose();
      composer.dispose();
      rt.dispose();
      renderer.dispose();
    };
  }, []);

  return <canvas ref={canvasRef} className="gl-canvas" aria-hidden="true" />;
}
