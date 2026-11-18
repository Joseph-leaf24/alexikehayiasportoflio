import * as THREE from "three";
import type { LayoutKey } from "@/lib/store";

/** Proportions of a piece once it has turned into an ice block in the formations. */
export const BLOCK = new THREE.Vector3(0.62, 0.36, 0.34);

export const KIND_BOX = 0;
export const KIND_DOME = 1;

export interface Layout {
  pos: THREE.Vector3[];
  quat: THREE.Quaternion[];
  scale: THREE.Vector3[];
  /** Spin around the Y axis through `center`, rad/s. */
  spin: number;
  center: THREE.Vector3;
  /** Idle floating amplitude. */
  bob: number;
  camPos: THREE.Vector3;
  camTarget: THREE.Vector3;
  /** 1 = real moonlit landscape, 0 = holographic contour map. */
  land: number;
  lines: number;
  exposure: number;
  /** How strongly the pointer pushes pieces away. */
  push: number;
  /** 0 = whitewashed houses / blue domes, 1 = ice. */
  ice: number;
}

export interface Windmill {
  pos: THREE.Vector3;
  yaw: number;
  scale: number;
}

export interface LayoutSet {
  count: number;
  kinds: Uint8Array;
  /** Index of each piece inside its own InstancedMesh. */
  local: Uint16Array;
  boxCount: number;
  domeCount: number;
  layouts: Record<LayoutKey, Layout>;
  edges: [number, number][];
  rand: Float32Array;
  delay: Float32Array;
  lambda: Float32Array;
  burstDir: THREE.Vector3[];
  tumbleAxis: THREE.Vector3[];
  windmills: Windmill[];
}

/* ------------------------------------------------------------------ */
/* Deterministic randomness + noise                                    */
/* ------------------------------------------------------------------ */
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const hash2 = (x: number, y: number) => {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return s - Math.floor(s);
};

function noise2(x: number, y: number) {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  const a = hash2(ix, iy);
  const b = hash2(ix + 1, iy);
  const c = hash2(ix, iy + 1);
  const d = hash2(ix + 1, iy + 1);
  return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
}

const smooth = (e0: number, e1: number, x: number) => {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};

/* ------------------------------------------------------------------ */
/* Caldera terrain: sea → sheer cliff → terraced village → ridge       */
/* ------------------------------------------------------------------ */

/** z of the cliff edge for a given x (the rim curves gently like a bay). */
export const rimZ = (x: number) => -0.011 * x * x + 0.45 * Math.sin(x * 0.33);
const rimYaw = (x: number) => -Math.atan(-0.022 * x + 0.45 * 0.33 * Math.cos(x * 0.33));

const VILLAGE_DEPTH = 8.4;

export function terrainHeight(x: number, z: number) {
  const d = rimZ(x) - z; // distance inland from the cliff edge
  const rim = 2.0 + 0.35 * Math.sin(x * 0.21 + 0.6);
  const ridge = 8.4 + 0.9 * Math.sin(x * 0.26 + 1.3) + 0.6 * (noise2(x * 0.17, 4.2) - 0.5);
  if (d < 0) {
    const t = Math.min(1, -d / 2.8);
    const s = t * t * (3 - 2 * t);
    return rim + (-1.8 - rim) * s + (noise2(x * 1.1, z * 1.1) - 0.5) * 0.7 * Math.sin(Math.PI * t);
  }
  const n = (noise2(x * 0.7, z * 0.7) - 0.5) * 0.3 * smooth(0, 0.5, d);
  if (d < VILLAGE_DEPTH) {
    const t = d / VILLAGE_DEPTH;
    return rim + (ridge - rim) * (t * 0.85 + 0.15 * t * t) + n;
  }
  return ridge - (d - VILLAGE_DEPTH) * 0.22 + n;
}

/* ------------------------------------------------------------------ */
/* Village                                                             */
/* ------------------------------------------------------------------ */
interface Piece {
  kind: number;
  pos: THREE.Vector3;
  quat: THREE.Quaternion;
  scale: THREE.Vector3;
}

function buildVillage(rnd: () => number) {
  const pieces: Piece[] = [];
  const Y = new THREE.Vector3(0, 1, 0);

  /** A whitewashed box sitting on the slope (walls run down into the rock). */
  const addHouse = (x: number, z: number, w: number, dep: number, hh: number, yaw: number, baseOverride?: number) => {
    const c = Math.cos(yaw);
    const s = Math.sin(yaw);
    let hMin = Infinity;
    let hMax = -Infinity;
    for (const sx of [-0.5, 0.5])
      for (const sz of [-0.5, 0.5]) {
        const px = x + sx * w * c + sz * dep * s;
        const pz = z - sx * w * s + sz * dep * c;
        const h = terrainHeight(px, pz);
        hMin = Math.min(hMin, h);
        hMax = Math.max(hMax, h);
      }
    const bottom = baseOverride ?? hMin - 0.4;
    const top = (baseOverride ?? hMax) + hh;
    const piece: Piece = {
      kind: KIND_BOX,
      pos: new THREE.Vector3(x, (bottom + top) / 2, z),
      quat: new THREE.Quaternion().setFromAxisAngle(Y, yaw),
      scale: new THREE.Vector3(w, top - bottom, dep),
    };
    pieces.push(piece);
    return { top, piece };
  };

  const addDome = (x: number, y: number, z: number, r: number) => {
    pieces.push({
      kind: KIND_DOME,
      pos: new THREE.Vector3(x, y, z),
      quat: new THREE.Quaternion(),
      scale: new THREE.Vector3(r, r * 1.08, r),
    });
  };

  const houses: { x: number; z: number; d: number; w: number; dep: number; top: number; yaw: number }[] = [];

  // Terraced houses on a jittered grid that follows the cliff edge
  for (let gx = -14; gx <= 14.01; gx += 0.8) {
    for (let gd = 0.28; gd <= 7.9; gd += 0.74) {
      const x = gx + (rnd() - 0.5) * 0.32;
      const d = gd + (rnd() - 0.5) * 0.28;
      const keep = (0.92 - d * 0.045) * (1 - smooth(9.5, 14, Math.abs(x)));
      if (rnd() > keep) continue;
      const z = rimZ(x) - d;
      const w = 0.48 + rnd() * 0.42;
      const dep = 0.46 + rnd() * 0.36;
      const hh = 0.34 + rnd() * 0.42;
      const yaw = rimYaw(x) + (rnd() - 0.5) * 0.16;
      const { top } = addHouse(x, z, w, dep, hh, yaw);
      houses.push({ x, z, d, w, dep, top, yaw });

      // Second storey, set back from the edge
      if (rnd() < 0.28) {
        const w2 = w * (0.45 + rnd() * 0.2);
        const d2 = dep * (0.5 + rnd() * 0.2);
        const lx = (rnd() - 0.5) * (w - w2);
        const lz = -(dep - d2) / 2;
        const c = Math.cos(yaw);
        const s = Math.sin(yaw);
        addHouse(x + lx * c + lz * s, z - lx * s + lz * c, w2, d2, 0.24 + rnd() * 0.2, yaw, top - 0.02);
      }
    }
  }

  // Low white walls along the cliff edge
  for (let x = -13; x <= 13; x += 1.6 + rnd() * 0.6) {
    if (rnd() < 0.4) continue;
    addHouse(x, rimZ(x) - 0.12, 1.1 + rnd() * 0.7, 0.09, 0.16, rimYaw(x));
  }

  // Churches: a drum and a blue dome on some of the bigger houses near the edge
  const candidates = houses.filter((h) => h.d < 4.5 && h.w > 0.72 && Math.abs(h.x) < 11);
  const churches: typeof houses = [];
  for (const h of candidates) {
    if (churches.length >= 9) break;
    if (churches.some((c) => Math.hypot(c.x - h.x, c.z - h.z) < 2.6)) continue;
    churches.push(h);
  }
  for (const ch of churches) {
    const { top } = addHouse(ch.x, ch.z, 0.42, 0.42, 0.2, ch.yaw, ch.top - 0.02);
    addDome(ch.x, top - 0.01, ch.z, 0.27 + rnd() * 0.05);
  }

  // Bell towers with tiny domes
  for (const ch of churches.slice(0, 4)) {
    const bx = ch.x + Math.cos(ch.yaw) * (ch.w * 0.5 + 0.2);
    const bz = ch.z - Math.sin(ch.yaw) * (ch.w * 0.5 + 0.2);
    const { top } = addHouse(bx, bz, 0.22, 0.22, 0.8 + rnd() * 0.25, ch.yaw);
    addDome(bx, top - 0.01, bz, 0.12);
  }

  // Windmills on the ridge
  const windmills: Windmill[] = [-4.2, -1.5, 9.6].map((x, i) => {
    const z = rimZ(x) - (VILLAGE_DEPTH + 0.2 + i * 0.3);
    return { pos: new THREE.Vector3(x, terrainHeight(x, z) - 0.05, z), yaw: rimYaw(x) + (i - 1) * 0.15, scale: 1.05 };
  });

  return { pieces, windmills };
}

const basis = (x: THREE.Vector3, y: THREE.Vector3, z: THREE.Vector3) =>
  new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));

const randomQuat = (rnd: () => number) => {
  const u1 = rnd(), u2 = rnd() * Math.PI * 2, u3 = rnd() * Math.PI * 2;
  const a = Math.sqrt(1 - u1), b = Math.sqrt(u1);
  return new THREE.Quaternion(a * Math.sin(u2), a * Math.cos(u2), b * Math.sin(u3), b * Math.cos(u3));
};

export function buildLayouts(): LayoutSet {
  const rnd = mulberry32(232230);
  const { pieces, windmills } = buildVillage(rnd);
  const N = pieces.length;

  const kinds = new Uint8Array(N);
  const local = new Uint16Array(N);
  let boxCount = 0;
  let domeCount = 0;
  pieces.forEach((p, i) => {
    kinds[i] = p.kind;
    local[i] = p.kind === KIND_DOME ? domeCount++ : boxCount++;
  });

  /** Formation-sized scale for a piece (ice block proportions, or a small dome). */
  const formScale = (i: number, s: number) =>
    kinds[i] === KIND_DOME ? new THREE.Vector3(0.3 * s, 0.3 * s, 0.3 * s) : BLOCK.clone().multiplyScalar(s);

  const rand = new Float32Array(N);
  const delay = new Float32Array(N);
  const lambda = new Float32Array(N);
  const burstDir: THREE.Vector3[] = [];
  const tumbleAxis: THREE.Vector3[] = [];
  const villageCenter = new THREE.Vector3(0, 1.5, -4);
  for (let i = 0; i < N; i++) {
    rand[i] = rnd();
    // The village lifts off in a wave sweeping across the cliff
    delay[i] = THREE.MathUtils.clamp(((pieces[i].pos.x + 14) / 28) * 0.22 + rnd() * 0.13, 0, 0.35);
    lambda[i] = 3.2 + rnd() * 3.5;
    burstDir.push(
      pieces[i].pos.clone().sub(villageCenter).normalize().add(new THREE.Vector3(rnd() - 0.5, 0.6 + rnd() * 0.4, rnd() - 0.2)).normalize()
    );
    tumbleAxis.push(new THREE.Vector3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize());
  }

  // Formations float above the sea, in front of the cliff
  const F = new THREE.Vector3(0, 2.2, 5);
  const v = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
  const fv = (x: number, y: number, z: number) => v(x, y, z).add(F);

  /* Exploded ------------------------------------------------------- */
  const sc = fv(0, 2.4, 0);
  const shatterPos: THREE.Vector3[] = [];
  const shatterQuat: THREE.Quaternion[] = [];
  const shatterScale: THREE.Vector3[] = [];
  for (let i = 0; i < N; i++) {
    const dir = new THREE.Vector3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize();
    shatterPos.push(sc.clone().addScaledVector(dir, 3.0 + rnd() * 5).add(v(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).multiplyScalar(1.4)));
    shatterQuat.push(randomQuat(rnd));
    shatterScale.push(formScale(i, 0.6 + rnd() * 0.7));
  }
  const lostPos = shatterPos.map((p) => p.clone().sub(sc).multiplyScalar(1.6).add(sc));

  /* Data helix ----------------------------------------------------- */
  const helixPos: THREE.Vector3[] = [];
  const helixQuat: THREE.Quaternion[] = [];
  const K = Math.ceil(N / 2);
  for (let i = 0; i < N; i++) {
    const strand = i % 2;
    const t = K > 1 ? Math.floor(i / 2) / (K - 1) : 0;
    const a = t * Math.PI * 2 * 2.6 + strand * Math.PI;
    helixPos.push(fv(1.9 * Math.cos(a), 2.6 + (t - 0.5) * 9.0, 1.9 * Math.sin(a)));
    const radial = v(Math.cos(a), 0, Math.sin(a));
    const up = v(0, 1, 0);
    helixQuat.push(basis(radial, up, new THREE.Vector3().crossVectors(radial, up)));
  }

  /* Neural network ------------------------------------------------- */
  const weights = [0.09, 0.18, 0.225, 0.225, 0.18, 0.1];
  const sizes = weights.map((w) => Math.round(w * N));
  sizes[sizes.length - 1] += N - sizes.reduce((a, b) => a + b, 0);
  const netPos: THREE.Vector3[] = [];
  const netQuat: THREE.Quaternion[] = [];
  const layerStart: number[] = [];
  const faceCam = new THREE.Quaternion().setFromAxisAngle(v(0, 1, 0), -0.65);
  sizes.forEach((count, l) => {
    layerStart.push(netPos.length);
    const x = (l - (sizes.length - 1) / 2) * 1.75;
    for (let j = 0; j < count; j++) {
      const rr = 0.34 * Math.sqrt(j + 0.5);
      const ang = j * 2.39996323;
      netPos.push(fv(x, 2.4 + rr * Math.sin(ang), rr * Math.cos(ang)));
      netQuat.push(faceCam.clone());
    }
  });
  const edges: [number, number][] = [];
  for (let l = 0; l < sizes.length - 1; l++)
    for (let j = 0; j < sizes[l]; j++) {
      const fanOut = 2 + Math.floor(rnd() * 2);
      for (let f = 0; f < fanOut; f++) edges.push([layerStart[l] + j, layerStart[l + 1] + Math.floor(rnd() * sizes[l + 1])]);
    }

  /* Voxel cube ----------------------------------------------------- */
  const side = Math.ceil(Math.cbrt(N));
  const cells: THREE.Vector3[] = [];
  const half = ((side - 1) * 0.5) / 2;
  for (let x = 0; x < side; x++)
    for (let y = 0; y < side; y++) for (let z = 0; z < side; z++) cells.push(v(x * 0.5 - half, y * 0.5 - half, z * 0.5 - half));
  for (let i = cells.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [cells[i], cells[j]] = [cells[j], cells[i]];
  }
  const tilt = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.42, 0.3, 0.18));
  const cubeCenter = fv(0, 2.6, 0);
  const cubePos = cells.slice(0, N).map((c) => c.clone().applyQuaternion(tilt).add(cubeCenter));

  const scaled = (s: number) => pieces.map((_, i) => formScale(i, s));
  const villageScale = pieces.map((p) => p.scale.clone());
  const zero = v(0, 0, 0);

  const layouts: Record<LayoutKey, Layout> = {
    village: {
      pos: pieces.map((p) => p.pos), quat: pieces.map((p) => p.quat), scale: villageScale, spin: 0, center: zero, bob: 0,
      camPos: v(-6.5, 5.6, 15.5), camTarget: v(1.2, 4.4, -3.6), land: 1, lines: 0, exposure: 1, push: 0.35, ice: 0,
    },
    shatter: {
      pos: shatterPos, quat: shatterQuat, scale: shatterScale, spin: 0.05, center: sc, bob: 0.18,
      camPos: fv(2.2, 3.6, 10.5), camTarget: fv(-2.2, 2.5, 0), land: 0.35, lines: 0, exposure: 0.9, push: 1.3, ice: 1,
    },
    helix: {
      pos: helixPos, quat: helixQuat, scale: scaled(0.72), spin: 0.28, center: fv(0, 2.6, 0), bob: 0.04,
      camPos: fv(0, 2.6, 11), camTarget: fv(0, 2.6, 0), land: 0.12, lines: 0, exposure: 0.85, push: 0.9, ice: 1,
    },
    network: {
      pos: netPos, quat: netQuat, scale: scaled(0.5), spin: 0.07, center: fv(0, 2.4, 0), bob: 0.05,
      camPos: fv(-6.8, 4.2, 8.2), camTarget: fv(0.4, 2.4, 0), land: 0, lines: 1, exposure: 0.62, push: 0.7, ice: 1,
    },
    villageTop: {
      pos: pieces.map((p) => p.pos), quat: pieces.map((p) => p.quat), scale: villageScale, spin: 0, center: zero, bob: 0,
      camPos: v(2, 24, 13), camTarget: v(0, 2.5, -4.5), land: 1, lines: 0, exposure: 0.85, push: 0.35, ice: 0,
    },
    cube: {
      pos: cubePos, quat: cubePos.map(() => tilt.clone()), scale: scaled(0.62), spin: 0.16, center: cubeCenter, bob: 0.03,
      camPos: fv(0, 2.8, 9.6), camTarget: fv(-3.0, 2.6, 0), land: 0.1, lines: 0, exposure: 0.7, push: 0.8, ice: 1,
    },
    lost: {
      pos: lostPos, quat: shatterQuat, scale: shatterScale, spin: 0.03, center: sc, bob: 0.3,
      camPos: fv(0, 2.5, 12), camTarget: fv(0, 2.5, 0), land: 0.25, lines: 0, exposure: 0.9, push: 1.2, ice: 1,
    },
  };

  return { count: N, kinds, local, boxCount, domeCount, layouts, edges, rand, delay, lambda, burstDir, tumbleAxis, windmills };
}
