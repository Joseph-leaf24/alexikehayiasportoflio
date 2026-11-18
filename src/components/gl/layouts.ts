import * as THREE from "three";
import type { LayoutKey } from "@/lib/store";

/** Dimensions of a single ice block (width, height, depth). */
export const BLOCK = new THREE.Vector3(0.62, 0.36, 0.34);

export interface Layout {
  pos: THREE.Vector3[];
  quat: THREE.Quaternion[];
  scale: number[];
  /** Spin around the Y axis through `center`, rad/s. */
  spin: number;
  center: THREE.Vector3;
  /** Idle floating amplitude. */
  bob: number;
  camPos: THREE.Vector3;
  camTarget: THREE.Vector3;
  ground: number;
  lines: number;
  exposure: number;
  /** How strongly the pointer pushes blocks away. */
  push: number;
}

export interface LayoutSet {
  count: number;
  layouts: Record<LayoutKey, Layout>;
  edges: [number, number][];
  /** Per-instance helpers */
  rand: Float32Array;
  delay: Float32Array;
  lambda: Float32Array;
  burstDir: THREE.Vector3[];
  tumbleAxis: THREE.Vector3[];
}

/* Deterministic RNG so the composition is identical on every visit */
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const basis = (x: THREE.Vector3, y: THREE.Vector3, z: THREE.Vector3) =>
  new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));

const randomQuat = (rnd: () => number) => {
  // Uniform random rotation (Shoemake)
  const u1 = rnd(), u2 = rnd() * Math.PI * 2, u3 = rnd() * Math.PI * 2;
  const a = Math.sqrt(1 - u1), b = Math.sqrt(u1);
  return new THREE.Quaternion(a * Math.sin(u2), a * Math.cos(u2), b * Math.sin(u3), b * Math.cos(u3));
};

/* ------------------------------------------------------------------ */
/* Igloo: brick rings on a hemisphere + an entrance tunnel             */
/* ------------------------------------------------------------------ */
function buildIgloo() {
  const R = 2.6;
  const pos: THREE.Vector3[] = [];
  const quat: THREE.Quaternion[] = [];
  const dPhi = (BLOCK.y * 1.04) / R;

  for (let row = 0; ; row++) {
    const phi = dPhi * (row + 0.5);
    const r = R * Math.cos(phi);
    if (phi > Math.PI / 2 - dPhi * 0.9 || r < 0.45) break;
    const y = R * Math.sin(phi);
    const n = Math.max(4, Math.floor((2 * Math.PI * r) / (BLOCK.x * 1.05)));
    for (let j = 0; j < n; j++) {
      const theta = ((j + (row % 2) * 0.5) / n) * Math.PI * 2;
      // Leave a doorway facing +z (theta = PI/2)
      const d = Math.atan2(Math.sin(theta - Math.PI / 2), Math.cos(theta - Math.PI / 2));
      if (y < 1.25 && Math.abs(d) * r < 0.62) continue;
      const nrm = new THREE.Vector3(Math.cos(phi) * Math.cos(theta), Math.sin(phi), Math.cos(phi) * Math.sin(theta));
      const tan = new THREE.Vector3(-Math.sin(theta), 0, Math.cos(theta));
      const up = new THREE.Vector3().crossVectors(tan, nrm);
      const back = new THREE.Vector3().crossVectors(tan, up);
      pos.push(nrm.clone().multiplyScalar(R));
      quat.push(basis(tan, up, back));
    }
  }

  // Cap stone
  pos.push(new THREE.Vector3(0, R + 0.02, 0));
  quat.push(basis(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, -1), new THREE.Vector3(0, 1, 0)));

  // Entrance tunnel: half-arches marching out of the doorway
  const archR = 0.78;
  for (let seg = 0; seg < 4; seg++) {
    const z = 2.45 + seg * BLOCK.y * 1.05;
    const off = seg % 2 ? 0.12 : -0.12;
    for (let k = 0; k < 4; k++) {
      const a = ((k + 0.5) / 4) * Math.PI + off;
      const tan = new THREE.Vector3(-Math.sin(a), Math.cos(a), 0);
      const along = new THREE.Vector3(0, 0, 1);
      const radial = new THREE.Vector3().crossVectors(tan, along);
      pos.push(new THREE.Vector3(archR * Math.cos(a), archR * Math.sin(a), z));
      quat.push(basis(tan, along, radial));
    }
  }
  return { pos, quat };
}

export function buildLayouts(): LayoutSet {
  const rnd = mulberry32(232230);
  const igloo = buildIgloo();
  const N = igloo.pos.length;
  const ones = () => new Array(N).fill(1);

  const rand = new Float32Array(N);
  const delay = new Float32Array(N);
  const lambda = new Float32Array(N);
  const burstDir: THREE.Vector3[] = [];
  const tumbleAxis: THREE.Vector3[] = [];
  const domeCenter = new THREE.Vector3(0, 0.8, 0);
  for (let i = 0; i < N; i++) {
    rand[i] = rnd();
    // Transitions ripple from the top of the igloo downwards
    delay[i] = THREE.MathUtils.clamp((1 - igloo.pos[i].y / 2.7) * 0.22 + rnd() * 0.13, 0, 0.35);
    lambda[i] = 3.2 + rnd() * 3.5;
    burstDir.push(
      igloo.pos[i].clone().sub(domeCenter).normalize().add(new THREE.Vector3(rnd() - 0.5, rnd() - 0.3, rnd() - 0.5).multiplyScalar(0.6)).normalize()
    );
    tumbleAxis.push(new THREE.Vector3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize());
  }

  /* Exploded igloo ------------------------------------------------- */
  const shatterPos: THREE.Vector3[] = [];
  const shatterQuat: THREE.Quaternion[] = [];
  const shatterScale: number[] = [];
  const sc = new THREE.Vector3(0, 2.4, 0);
  for (let i = 0; i < N; i++) {
    const p = sc.clone()
      .addScaledVector(burstDir[i], 3.0 + rnd() * 4.8)
      .add(new THREE.Vector3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).multiplyScalar(1.4));
    shatterPos.push(p);
    shatterQuat.push(randomQuat(rnd));
    shatterScale.push(0.6 + rnd() * 0.7);
  }

  /* Lost (404): wider, sparser drift -------------------------------- */
  const lostPos = shatterPos.map((p) => p.clone().sub(sc).multiplyScalar(1.6).add(sc));

  /* Data helix ------------------------------------------------------ */
  const helixPos: THREE.Vector3[] = [];
  const helixQuat: THREE.Quaternion[] = [];
  const K = Math.ceil(N / 2);
  for (let i = 0; i < N; i++) {
    const strand = i % 2;
    const k = Math.floor(i / 2);
    const t = K > 1 ? k / (K - 1) : 0;
    const a = t * Math.PI * 2 * 2.2 + strand * Math.PI;
    const r = 1.9;
    helixPos.push(new THREE.Vector3(r * Math.cos(a), 2.6 + (t - 0.5) * 8.0, r * Math.sin(a)));
    const radial = new THREE.Vector3(Math.cos(a), 0, Math.sin(a));
    const up = new THREE.Vector3(0, 1, 0);
    helixQuat.push(basis(radial, up, new THREE.Vector3().crossVectors(radial, up)));
  }

  /* Neural network: layered discs + edges -------------------------- */
  const weights = [0.09, 0.18, 0.225, 0.225, 0.18, 0.1];
  const sizes = weights.map((w) => Math.round(w * N));
  sizes[sizes.length - 1] += N - sizes.reduce((a, b) => a + b, 0);
  const netPos: THREE.Vector3[] = [];
  const netQuat: THREE.Quaternion[] = [];
  const layerStart: number[] = [];
  const faceCam = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -0.65);
  sizes.forEach((count, l) => {
    layerStart.push(netPos.length);
    const x = (l - (sizes.length - 1) / 2) * 1.75;
    for (let j = 0; j < count; j++) {
      const rr = 0.36 * Math.sqrt(j + 0.5);
      const ang = j * 2.39996323;
      netPos.push(new THREE.Vector3(x, 2.4 + rr * Math.sin(ang), rr * Math.cos(ang)));
      netQuat.push(faceCam.clone());
    }
  });
  const edges: [number, number][] = [];
  for (let l = 0; l < sizes.length - 1; l++) {
    for (let j = 0; j < sizes[l]; j++) {
      const from = layerStart[l] + j;
      const fanOut = 2 + Math.floor(rnd() * 2);
      for (let f = 0; f < fanOut; f++) {
        edges.push([from, layerStart[l + 1] + Math.floor(rnd() * sizes[l + 1])]);
      }
    }
  }

  /* Voxel cube (detail pages) -------------------------------------- */
  const side = Math.ceil(Math.cbrt(N));
  const cells: THREE.Vector3[] = [];
  const spacing = 0.5;
  const half = ((side - 1) * spacing) / 2;
  for (let x = 0; x < side; x++)
    for (let y = 0; y < side; y++)
      for (let z = 0; z < side; z++) cells.push(new THREE.Vector3(x * spacing - half, y * spacing - half, z * spacing - half));
  for (let i = cells.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [cells[i], cells[j]] = [cells[j], cells[i]];
  }
  const tilt = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.42, 0.3, 0.18));
  const cubeCenter = new THREE.Vector3(0, 2.6, 0);
  const cubePos = cells.slice(0, N).map((c) => c.clone().applyQuaternion(tilt).add(cubeCenter));
  const cubeQuat = cubePos.map(() => tilt.clone());

  const v = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
  const origin = v(0, 0, 0);

  const layouts: Record<LayoutKey, Layout> = {
    dome: {
      pos: igloo.pos, quat: igloo.quat, scale: ones(), spin: 0.06, center: origin, bob: 0,
      camPos: v(0, 1.9, 9.2), camTarget: v(0, 1.35, 0), ground: 1, lines: 0, exposure: 1, push: 1,
    },
    shatter: {
      pos: shatterPos, quat: shatterQuat, scale: shatterScale, spin: 0.05, center: sc, bob: 0.18,
      camPos: v(2.2, 3.6, 10.5), camTarget: v(-2.2, 2.5, 0), ground: 0.35, lines: 0, exposure: 0.9, push: 1.3,
    },
    helix: {
      pos: helixPos, quat: helixQuat, scale: new Array(N).fill(0.75), spin: 0.28, center: v(0, 2.6, 0), bob: 0.04,
      camPos: v(0, 2.6, 10.5), camTarget: v(0, 2.6, 0), ground: 0.12, lines: 0, exposure: 0.85, push: 0.9,
    },
    network: {
      pos: netPos, quat: netQuat, scale: new Array(N).fill(0.5), spin: 0.07, center: v(0, 2.4, 0), bob: 0.05,
      camPos: v(-6.8, 4.2, 8.2), camTarget: v(0.4, 2.4, 0), ground: 0, lines: 1, exposure: 0.62, push: 0.7,
    },
    domeTop: {
      pos: igloo.pos, quat: igloo.quat, scale: ones(), spin: 0.05, center: origin, bob: 0,
      camPos: v(0.01, 12.5, 5.2), camTarget: v(0, 0.6, 0), ground: 1, lines: 0, exposure: 1, push: 1,
    },
    cube: {
      pos: cubePos, quat: cubeQuat, scale: new Array(N).fill(0.62), spin: 0.16, center: cubeCenter, bob: 0.03,
      camPos: v(0, 2.8, 9.6), camTarget: v(-3.0, 2.6, 0), ground: 0.2, lines: 0, exposure: 0.7, push: 0.8,
    },
    lost: {
      pos: lostPos, quat: shatterQuat, scale: shatterScale, spin: 0.03, center: sc, bob: 0.3,
      camPos: v(0, 2.5, 12), camTarget: v(0, 2.5, 0), ground: 0.3, lines: 0, exposure: 0.9, push: 1.2,
    },
  };

  return { count: N, layouts, edges, rand, delay, lambda, burstDir, tumbleAxis };
}
