/**
 * Geometry for the 3D scene, all seeded so it is identical on every load: the caldera
 * terrain (as voxels), the moon, the ice floes, the Cycladic village built from
 * instanced pieces, and every formation those pieces morph between.
 */
import * as THREE from "three";
import type { LayoutKey } from "@/lib/store";

/** Proportions of a piece once it has turned into an ice block in the formations. */
export const BLOCK = new THREE.Vector3(0.62, 0.36, 0.34);
/** Size of one terrain voxel (the hills are stacks of cubes this big). */
export const CELL = 0.5;

/** Piece kinds: drawn by the rounded-box or the dome InstancedMesh. */
export const KIND_BOX = 0;
export const KIND_DOME = 1;

/** One formation: a target transform for every piece, plus the camera and scene settings it is shown with. */
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
  /** 1 = real moonlit landscape, 0 = holographic voxel equaliser. */
  land: number;
  /** Opacity of the neural-network edges. */
  lines: number;
  /** Overall brightness. */
  exposure: number;
  /** How strongly the pointer pushes pieces away. */
  push: number;
  /** 0 = whitewashed houses / blue domes, 1 = ice. */
  ice: number;
  /** Windmill rotors turn (only while the village is assembled). */
  rotors: boolean;
  /** Points of interest the UI can highlight (e.g. the four domain clusters). */
  anchors?: THREE.Vector3[];
}

/** Where a windmill stands; its rotor pieces are posed relative to `hub` (local, unscaled). */
export interface Windmill {
  base: THREE.Vector3;
  yawQ: THREE.Quaternion;
  hub: THREE.Vector3;
  scale: number;
}

/** Every piece's per-instance data plus all formations, indexed by piece. */
export interface LayoutSet {
  count: number;
  kinds: Uint8Array;
  /** Index of each piece inside its own InstancedMesh. */
  local: Uint16Array;
  boxCount: number;
  domeCount: number;
  /** rgb albedo + w = 1 if the piece gets windows */
  tints: Float32Array;
  layouts: Record<LayoutKey, Layout>;
  /** Piece pairs wired together in the network formation. */
  edges: [number, number][];
  rand: Float32Array;
  /** When each piece sets off during a transition (0..0.35 of its progress). */
  delay: Float32Array;
  /** Per-piece smoothing rate (1/s), so pieces settle at slightly different speeds. */
  lambda: Float32Array;
  /** Fling direction mid-transition and on hold-to-shatter. */
  burstDir: THREE.Vector3[];
  tumbleAxis: THREE.Vector3[];
  windmills: Windmill[];
  /** Rotor membership: windmill index or -1, plus position/rotation in the rotor plane. */
  rotorMill: Int8Array;
  rotorLocal: THREE.Vector3[];
  rotorLocalQ: THREE.Quaternion[];
}

/* ------------------------------------------------------------------ */
/* Deterministic randomness + noise                                    */
/* ------------------------------------------------------------------ */
/** Seeded PRNG: returns a function yielding floats in [0, 1). */
export function mulberry32(seed: number) {
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
/** Yaw that lines a building up with the cliff edge at x (from the slope of rimZ). */
const rimYaw = (x: number) => -Math.atan(-0.022 * x + 0.45 * 0.33 * Math.cos(x * 0.33));

/** How far inland the village slope climbs before the ridge. */
const VILLAGE_DEPTH = 8.4;

/** Ground height at (x, z): sea floor → cliff → terraced slope up to the ridge, then gently down behind it. */
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

/** Height of the voxel column at (x, z): the terrain snapped down to whole cubes. */
export const voxelTop = (x: number, z: number) => Math.floor(terrainHeight(x, z) / CELL) * CELL;

/** Sea surface height. Kept just below the lowest cube tops so land and water never z-fight. */
export const SEA_LEVEL = -0.15;

/**
 * Swell amplitude. Water blocks rise and fall by at most this much around SEA_LEVEL, which
 * keeps their tops below the lowest land (0) and above the bottom of the cliff columns (−0.5).
 */
export const WAVE_AMP = 0.12;

/** Swell height at (x, z) at time t. GLSL twin: `waveHeight` in shaders.ts (floes ride this one). */
export function waveHeight(x: number, z: number, t: number) {
  return WAVE_AMP * (0.5 * Math.sin(0.55 * x + 0.9 * t) + 0.3 * Math.sin(0.8 * z - 1.3 * t + 1.7) + 0.2 * Math.sin(0.35 * (x + z) + 0.6 * t));
}

/** The flat distant sea sits just under the lowest trough, so it never fights the water blocks. */
export const SEA_PLANE_Y = SEA_LEVEL - WAVE_AMP - 0.03;

/** How far toward (and past) the camera the water blocks reach. */
const WATER_Z_MAX = 18;

/**
 * The near sea as a grid of water blocks, wherever there is no land: x, z and a random
 * 0..1 per block. Same x extent as the terrain; `cell` sets the block size (per device tier).
 */
export function buildWater(ext: { xMin: number; xMax: number; cell: number } = { xMin: -28, xMax: 28, cell: 0.6 }) {
  const rnd = mulberry32(4242);
  const out: number[] = [];
  const { xMin, xMax, cell } = ext;
  for (let x = xMin; x <= xMax + 1e-6; x += cell) {
    for (let z = -10; z <= WATER_Z_MAX + 1e-6; z += cell) {
      if (voxelTop(x, z) >= 0) continue; // land or cliff here
      out.push(x, z, rnd());
    }
  }
  return { data: Float32Array.from(out), count: out.length / 3 };
}

/**
 * The hills as individual cubes (so they can break apart). Each cube: x, y, z, depth,
 * where depth is the distance from the cube's top to the top of its column.
 * Columns only go as deep as needed to close the gap to the lowest neighbour.
 */
export function buildTerrain(ext: { xMin: number; xMax: number; zMin: number } = { xMin: -28, xMax: 28, zMin: -30 }) {
  const x0 = ext.xMin;
  const z0 = ext.zMin;
  const nx = Math.round((ext.xMax - ext.xMin) / CELL) + 1;
  const nz = Math.round((5 - ext.zMin) / CELL) + 1;
  const tops = new Float32Array(nx * nz);
  for (let ix = 0; ix < nx; ix++)
    for (let iz = 0; iz < nz; iz++) tops[ix * nz + iz] = voxelTop(x0 + ix * CELL, z0 + iz * CELL);
  const cubes: number[] = [];
  for (let ix = 0; ix < nx; ix++)
    for (let iz = 0; iz < nz; iz++) {
      const top = tops[ix * nz + iz];
      if (top < 0) continue; // under the sea
      let minN = top;
      for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const jx = ix + dx;
        const jz = iz + dz;
        let t = jx >= 0 && jx < nx && jz >= 0 && jz < nz ? tops[jx * nz + jz] : -1;
        if (t < 0) t = -1;
        minN = Math.min(minN, t);
      }
      // Nothing below the waterline is visible, so no column reaches more than one cube below y = 0
      const bottom = Math.max(-CELL, minN - CELL);
      const n = Math.round((top - bottom) / CELL);
      const x = x0 + ix * CELL;
      const z = z0 + iz * CELL;
      for (let k = 0; k < n; k++) {
        const y = top - (k + 0.5) * CELL;
        cubes.push(x, y, z, k * CELL);
      }
    }
  return { data: Float32Array.from(cubes), count: cubes.length / 4 };
}

/** The moon as a shell of cubes, with a few recessed craters. */
export function buildMoon(radius = 7, cube = 0.9) {
  const rnd = mulberry32(1969);
  const craters = Array.from({ length: 9 }, () => ({
    dir: new THREE.Vector3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize(),
    cos: Math.cos(0.18 + rnd() * 0.3),
  }));
  const offsets: number[] = [];
  const crater: number[] = [];
  const n = Math.ceil(radius / cube);
  for (let x = -n; x <= n; x++)
    for (let y = -n; y <= n; y++)
      for (let z = -n; z <= n; z++) {
        const p = new THREE.Vector3(x, y, z).multiplyScalar(cube);
        const d = p.length();
        if (d > radius || d < radius - cube * 1.25) continue;
        const dir = p.clone().normalize();
        offsets.push(p.x, p.y, p.z);
        crater.push(craters.some((c) => dir.dot(c.dir) > c.cos) ? 1 : 0);
      }
  return { offsets: Float32Array.from(offsets), crater: Float32Array.from(crater), count: crater.length, cube };
}

/** Little ice floes drifting at the foot of the cliff. */
export function buildFloes(count = 70) {
  const rnd = mulberry32(77);
  return Array.from({ length: count }, () => {
    const x = (rnd() - 0.5) * 30;
    const z = rimZ(x) + 3.4 + rnd() * 7;
    const s = 0.18 + Math.pow(rnd(), 2) * 0.45;
    return { pos: new THREE.Vector3(x, SEA_LEVEL + 0.04, z), size: new THREE.Vector3(s * (0.8 + rnd() * 0.6), s * 0.45, s * (0.8 + rnd() * 0.6)), yaw: rnd() * 6.28, rand: rnd() };
  });
}

/* ------------------------------------------------------------------ */
/* Village                                                             */
/* ------------------------------------------------------------------ */
const WHITE: [number, number, number] = [0.85, 0.89, 0.96];
const WOOD: [number, number, number] = [0.24, 0.17, 0.13];
const CANVAS: [number, number, number] = [0.9, 0.87, 0.8];
const AEGEAN: [number, number, number] = [0.06, 0.2, 0.52];

interface Piece {
  kind: number;
  pos: THREE.Vector3;
  quat: THREE.Quaternion;
  scale: THREE.Vector3;
  tint: [number, number, number, number];
  rotor?: { mill: number; local: THREE.Vector3; localQ: THREE.Quaternion };
}

const basis = (x: THREE.Vector3, y: THREE.Vector3, z: THREE.Vector3) =>
  new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));

/**
 * The village as pieces: terraced houses (some two-storey), low walls along the cliff
 * edge, churches with blue domes, bell towers, and three windmills built cube by cube.
 * `primaryCount` is the number of pieces before the windmills.
 */
function buildVillage(rnd: () => number) {
  const pieces: Piece[] = [];
  const Y = new THREE.Vector3(0, 1, 0);

  /** A whitewashed box sitting on the slope (walls run down into the rock). */
  const addHouse = (
    x: number,
    z: number,
    w: number,
    dep: number,
    hh: number,
    yaw: number,
    baseOverride?: number,
    windows = 1
  ) => {
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
    // Foundations reach below the stepped voxel ground so nothing floats
    const bottom = baseOverride ?? hMin - 0.75;
    const top = (baseOverride ?? hMax) + hh;
    pieces.push({
      kind: KIND_BOX,
      pos: new THREE.Vector3(x, (bottom + top) / 2, z),
      quat: new THREE.Quaternion().setFromAxisAngle(Y, yaw),
      scale: new THREE.Vector3(w, top - bottom, dep),
      tint: [...WHITE, windows],
    });
    return { top };
  };

  const addDome = (x: number, y: number, z: number, r: number) => {
    pieces.push({
      kind: KIND_DOME,
      pos: new THREE.Vector3(x, y, z),
      quat: new THREE.Quaternion(),
      scale: new THREE.Vector3(r, r * 1.08, r),
      tint: [0, 0, 0, 0],
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
    addHouse(x, rimZ(x) - 0.12, 1.1 + rnd() * 0.7, 0.09, 0.16, rimYaw(x), undefined, 0);
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
    const { top } = addHouse(ch.x, ch.z, 0.42, 0.42, 0.2, ch.yaw, ch.top - 0.02, 0);
    addDome(ch.x, top - 0.01, ch.z, 0.27 + rnd() * 0.05);
  }

  // Bell towers with tiny domes
  for (const ch of churches.slice(0, 4)) {
    const bx = ch.x + Math.cos(ch.yaw) * (ch.w * 0.5 + 0.2);
    const bz = ch.z - Math.sin(ch.yaw) * (ch.w * 0.5 + 0.2);
    const { top } = addHouse(bx, bz, 0.22, 0.22, 0.8 + rnd() * 0.25, ch.yaw);
    addDome(bx, top - 0.01, bz, 0.12);
  }

  const primaryCount = pieces.length;

  /* Windmills, built brick by brick ---------------------------------- */
  const windmills: Windmill[] = [];
  [-4.2, -1.5, 9.6].forEach((x, mi) => {
    const z = rimZ(x) - (VILLAGE_DEPTH + 0.2 + mi * 0.3);
    const scale = 1.12;
    const base = new THREE.Vector3(x, voxelTop(x, z), z);
    const yawQ = new THREE.Quaternion().setFromAxisAngle(Y, rimYaw(x) + (mi - 1) * 0.15);
    const hub = new THREE.Vector3(0, 1.55, 0.64);
    windmills.push({ base, yawQ, hub, scale });

    const place = (local: THREE.Vector3, q: THREE.Quaternion, dims: THREE.Vector3, tint: [number, number, number, number]) => {
      pieces.push({
        kind: KIND_BOX,
        pos: local.clone().multiplyScalar(scale).applyQuaternion(yawQ).add(base),
        quat: yawQ.clone().multiply(q),
        scale: dims.clone().multiplyScalar(scale),
        tint,
      });
    };

    // Tower: rings of whitewashed bricks
    const rows = 7;
    const rowH = 0.25;
    for (let row = 0; row < rows; row++) {
      const r = 0.52 + (0.44 - 0.52) * ((row + 0.5) / rows);
      const n = Math.round((2 * Math.PI * r) / 0.3);
      for (let j = 0; j < n; j++) {
        const th = ((j + (row % 2) * 0.5) / n) * Math.PI * 2;
        const dFront = Math.atan2(Math.sin(th - Math.PI / 2), Math.cos(th - Math.PI / 2));
        if (row < 2 && Math.abs(dFront) * r < 0.16) continue; // doorway
        const nrm = new THREE.Vector3(Math.cos(th), 0, Math.sin(th));
        const tan = new THREE.Vector3(-Math.sin(th), 0, Math.cos(th));
        const up = new THREE.Vector3(0, 1, 0);
        const isWindow = row === 4 && Math.abs(dFront) * r < 0.15;
        place(
          nrm.clone().multiplyScalar(r).setY((row + 0.5) * rowH),
          basis(tan, up, new THREE.Vector3().crossVectors(tan, up)),
          new THREE.Vector3(((2 * Math.PI * r) / n) * 0.97, rowH * 0.94, 0.18),
          isWindow ? [...AEGEAN, 0] : [...WHITE, 0]
        );
      }
    }
    // Door
    place(new THREE.Vector3(0, 0.22, 0.52), new THREE.Quaternion(), new THREE.Vector3(0.26, 0.44, 0.06), [...AEGEAN, 0]);
    // Conical cap: shrinking rings of dark wood
    const yT = rows * rowH;
    for (let k = 0; k < 3; k++) {
      const r = 0.48 * (1 - k * 0.3);
      const n = Math.max(3, Math.round((2 * Math.PI * r) / 0.26));
      for (let j = 0; j < n; j++) {
        const th = ((j + k * 0.5) / n) * Math.PI * 2;
        const nrm = new THREE.Vector3(Math.cos(th), 0, Math.sin(th));
        const tan = new THREE.Vector3(-Math.sin(th), 0, Math.cos(th));
        const up = new THREE.Vector3(0, 1, 0);
        place(
          nrm.clone().multiplyScalar(r).setY(yT + 0.09 + k * 0.17),
          basis(tan, up, new THREE.Vector3().crossVectors(tan, up)),
          new THREE.Vector3(((2 * Math.PI * r) / n) * 0.97, 0.17, 0.2),
          [...WOOD, 0]
        );
      }
    }
    place(new THREE.Vector3(0, yT + 0.6, 0), new THREE.Quaternion(), new THREE.Vector3(0.16, 0.14, 0.16), [...WOOD, 0]);

    // Rotor: hub, eight spokes of cubes, and a canvas sail on each
    const Z = new THREE.Vector3(0, 0, 1);
    const addRotor = (local: THREE.Vector3, localQ: THREE.Quaternion, dims: THREE.Vector3, tint: [number, number, number, number]) => {
      pieces.push({
        kind: KIND_BOX,
        pos: local.clone().add(hub).multiplyScalar(scale).applyQuaternion(yawQ).add(base),
        quat: yawQ.clone().multiply(localQ),
        scale: dims.clone().multiplyScalar(scale),
        tint,
        rotor: { mill: mi, local, localQ },
      });
    };
    addRotor(new THREE.Vector3(), new THREE.Quaternion(), new THREE.Vector3(0.18, 0.18, 0.18), [...WOOD, 0]);
    for (let k = 0; k < 8; k++) {
      const a = (k * Math.PI) / 4;
      const dir = new THREE.Vector3(Math.cos(a), Math.sin(a), 0);
      const perp = new THREE.Vector3(-Math.sin(a), Math.cos(a), 0);
      const armQ = new THREE.Quaternion().setFromAxisAngle(Z, a - Math.PI / 2);
      for (let j = 0; j < 5; j++)
        addRotor(dir.clone().multiplyScalar(0.2 + j * 0.24), armQ.clone(), new THREE.Vector3(0.06, 0.21, 0.05), [...WOOD, 0]);
      addRotor(
        dir.clone().multiplyScalar(0.78).addScaledVector(perp, 0.12),
        armQ.clone().multiply(new THREE.Quaternion().setFromAxisAngle(Y, 0.35)),
        new THREE.Vector3(0.2, 0.86, 0.025),
        [...CANVAS, 0]
      );
    }
  });

  return { pieces, windmills, primaryCount };
}

/** Uniformly distributed random rotation (Shoemake's method). */
const randomQuat = (rnd: () => number) => {
  const u1 = rnd(), u2 = rnd() * Math.PI * 2, u3 = rnd() * Math.PI * 2;
  const a = Math.sqrt(1 - u1), b = Math.sqrt(u1);
  return new THREE.Quaternion(a * Math.sin(u2), a * Math.cos(u2), b * Math.sin(u3), b * Math.cos(u3));
};

/**
 * Builds the village pieces and every formation they morph between. In most formations
 * the village pieces make the main shape and the windmill pieces a secondary one.
 */
export function buildLayouts(): LayoutSet {
  const rnd = mulberry32(232230);
  const { pieces, windmills, primaryCount: P } = buildVillage(rnd);
  const N = pieces.length;

  const kinds = new Uint8Array(N);
  const local = new Uint16Array(N);
  const tints = new Float32Array(N * 4);
  const rotorMill = new Int8Array(N).fill(-1);
  const rotorLocal: THREE.Vector3[] = [];
  const rotorLocalQ: THREE.Quaternion[] = [];
  let boxCount = 0;
  let domeCount = 0;
  pieces.forEach((p, i) => {
    kinds[i] = p.kind;
    local[i] = p.kind === KIND_DOME ? domeCount++ : boxCount++;
    tints.set(p.tint, i * 4);
    rotorLocal.push(p.rotor?.local ?? new THREE.Vector3());
    rotorLocalQ.push(p.rotor?.localQ ?? new THREE.Quaternion());
    if (p.rotor) rotorMill[i] = p.rotor.mill;
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
    // Where the piece flies on hold-to-shatter (and arcs mid-transition): out from the village and up
    burstDir.push(
      pieces[i].pos.clone().sub(villageCenter).normalize().add(new THREE.Vector3(rnd() - 0.5, 0.6 + rnd() * 0.4, rnd() - 0.2)).normalize()
    );
    tumbleAxis.push(new THREE.Vector3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize());
  }

  // Formations float above the sea, in front of the cliff
  const F = new THREE.Vector3(0, 2.2, 5);
  const v = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
  const fv = (x: number, y: number, z: number) => v(x, y, z).add(F);
  const S = N - P; // windmill pieces: they form a secondary structure in some formations

  /* Exploded ------------------------------------------------------- */
  const sc = fv(0, 2.4, 0);
  const shatterPos: THREE.Vector3[] = [];
  const shatterQuat: THREE.Quaternion[] = [];
  const shatterScale: THREE.Vector3[] = [];
  for (let i = 0; i < N; i++) {
    const dir = new THREE.Vector3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize();
    shatterPos.push(sc.clone().addScaledVector(dir, 3.0 + rnd() * 5.5).add(v(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).multiplyScalar(1.4)));
    shatterQuat.push(randomQuat(rnd));
    shatterScale.push(formScale(i, (i < P ? 0.6 : 0.35) + rnd() * 0.6));
  }
  const lostPos = shatterPos.map((p) => p.clone().sub(sc).multiplyScalar(1.6).add(sc));

  /* 3D bar chart + a scatter plot floating above it (Work) --------- */
  const barsCenter = fv(0, 1.6, 0);
  const barsPos: THREE.Vector3[] = [];
  const barsQuat: THREE.Quaternion[] = [];
  const barsScale: THREE.Vector3[] = [];
  {
    const GX = 9;
    const GZ = 6;
    const raw: number[] = [];
    for (let i = 0; i < GX; i++)
      for (let j = 0; j < GZ; j++) {
        const u = i / (GX - 1);
        const w = j / (GZ - 1);
        raw.push(
          0.6 +
            3.2 * Math.exp(-((u - 0.68) ** 2 + (w - 0.35) ** 2) / 0.05) +
            2.2 * Math.exp(-((u - 0.22) ** 2 + (w - 0.72) ** 2) / 0.035) +
            0.5 * (Math.sin(u * 9) * 0.5 + 0.5)
        );
      }
    // Scale the bar heights so they use exactly P pieces
    const total = raw.reduce((x, y) => x + y, 0);
    const heights = raw.map((h) => Math.max(1, Math.round((h / total) * P)));
    let diff = P - heights.reduce((x, y) => x + y, 0);
    const order = heights.map((_, k) => k).sort((x, y) => heights[y] - heights[x]);
    for (let k = 0; diff !== 0; k = (k + 1) % order.length) {
      const idx = order[k];
      if (diff > 0) {
        heights[idx]++;
        diff--;
      } else if (heights[idx] > 1) {
        heights[idx]--;
        diff++;
      }
    }
    heights.forEach((h, k) => {
      const i = Math.floor(k / GZ);
      const j = k % GZ;
      for (let l = 0; l < h; l++) {
        barsPos.push(fv((i - (GX - 1) / 2) * 0.78, -0.6 + l * 0.24, (j - (GZ - 1) / 2) * 0.78));
        barsQuat.push(new THREE.Quaternion());
      }
    });
    for (let i = 0; i < P; i++) barsScale.push(kinds[i] === KIND_DOME ? v(0.2, 0.2, 0.2) : v(0.56, 0.2, 0.56));
    // Windmill pieces become a scatter plot with a rising trend
    for (let j = 0; j < S; j++) {
      const x = (rnd() - 0.5) * 6.4;
      const z = (rnd() - 0.5) * 4;
      const noise = (rnd() + rnd() + rnd() - 1.5) * 0.55;
      barsPos.push(fv(x, 3.9 + 0.35 * x + noise, z));
      barsQuat.push(randomQuat(rnd));
      barsScale.push(v(0.15, 0.15, 0.15));
    }
  }

  /* Four domain clusters joined by a lifecycle ring (Domains) ------- */
  const clusterCenter = fv(0, 2.4, 0);
  const anchors = [0, 1, 2, 3].map((k) => {
    const a = Math.PI * 0.75 - (k * Math.PI) / 2; // lifecycle order: front-left, front-right, back-right, back-left (before the spin)
    return clusterCenter.clone().add(v(Math.cos(a) * 3.2, Math.sin(k * 1.7) * 0.5, Math.sin(a) * 3.2));
  });
  const clusterPos: THREE.Vector3[] = [];
  const clusterQuat: THREE.Quaternion[] = [];
  for (let i = 0; i < P; i++) {
    const c = i % 4;
    const k = Math.floor(i / 4);
    const n = Math.ceil(P / 4);
    const y = 1 - (2 * (k + 0.5)) / n;
    const r = Math.sqrt(1 - y * y);
    const th = k * 2.39996323;
    const R = 1.05 + rnd() * 0.12;
    clusterPos.push(anchors[c].clone().add(v(Math.cos(th) * r * R, y * R, Math.sin(th) * r * R)));
    clusterQuat.push(randomQuat(rnd));
  }
  for (let j = 0; j < S; j++) {
    const a = (j / S) * Math.PI * 2;
    clusterPos.push(clusterCenter.clone().add(v(Math.cos(a) * 3.2, (rnd() - 0.5) * 0.25, Math.sin(a) * 3.2)).add(v(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).multiplyScalar(0.25)));
    clusterQuat.push(randomQuat(rnd));
  }

  /* Neural network + outer constellation ---------------------------- */
  const weights = [0.09, 0.18, 0.225, 0.225, 0.18, 0.1];
  const sizes = weights.map((w) => Math.round(w * P));
  sizes[sizes.length - 1] += P - sizes.reduce((a, b) => a + b, 0);
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
  const netCenter = fv(0, 2.4, 0);
  for (let j = 0; j < S; j++) {
    const dir = new THREE.Vector3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize();
    netPos.push(netCenter.clone().addScaledVector(dir, 7 + rnd() * 2.5));
    netQuat.push(randomQuat(rnd));
  }
  const edges: [number, number][] = [];
  for (let l = 0; l < sizes.length - 1; l++)
    for (let j = 0; j < sizes[l]; j++) {
      const fanOut = 2 + Math.floor(rnd() * 2);
      for (let f = 0; f < fanOut; f++) edges.push([layerStart[l] + j, layerStart[l + 1] + Math.floor(rnd() * sizes[l + 1])]);
    }

  /* Voxel cube ----------------------------------------------------- */
  const side = Math.ceil(Math.cbrt(N));
  const gap = 0.42;
  const cells: THREE.Vector3[] = [];
  const half = ((side - 1) * gap) / 2;
  for (let x = 0; x < side; x++)
    for (let y = 0; y < side; y++) for (let z = 0; z < side; z++) cells.push(v(x * gap - half, y * gap - half, z * gap - half));
  for (let i = cells.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [cells[i], cells[j]] = [cells[j], cells[i]];
  }
  const tilt = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.42, 0.3, 0.18));
  const cubeCenter = fv(0, 2.6, 0);
  const cubePos = cells.slice(0, N).map((c) => c.clone().applyQuaternion(tilt).add(cubeCenter));

  const scaledMix = (sp: number, ss: number) => pieces.map((_, i) => formScale(i, i < P ? sp : ss));
  const villageScale = pieces.map((p) => p.scale.clone());
  const zero = v(0, 0, 0);

  const layouts: Record<LayoutKey, Layout> = {
    village: {
      pos: pieces.map((p) => p.pos), quat: pieces.map((p) => p.quat), scale: villageScale, spin: 0, center: zero, bob: 0,
      camPos: v(-6.5, 5.6, 15.5), camTarget: v(1.2, 4.4, -3.6), land: 1, lines: 0, exposure: 1, push: 0.35, ice: 0, rotors: true,
    },
    shatter: {
      pos: shatterPos, quat: shatterQuat, scale: shatterScale, spin: 0.05, center: sc, bob: 0.18,
      camPos: fv(2.2, 3.6, 10.5), camTarget: fv(-2.2, 2.5, 0), land: 0.35, lines: 0, exposure: 0.9, push: 1.3, ice: 1, rotors: false,
    },
    bars: {
      pos: barsPos, quat: barsQuat, scale: barsScale, spin: 0.08, center: barsCenter, bob: 0.02,
      camPos: fv(-5.4, 6.4, 8.6), camTarget: fv(0.4, 1.6, 0), land: 0.08, lines: 0, exposure: 0.66, push: 0.8, ice: 1, rotors: false,
    },
    clusters: {
      pos: clusterPos, quat: clusterQuat, scale: scaledMix(0.48, 0.26), spin: 0.12, center: clusterCenter, bob: 0.06,
      camPos: fv(0.5, 5.2, 11.5), camTarget: fv(0.5, 2.2, 0), land: 0.2, lines: 0, exposure: 0.72, push: 1, ice: 1, rotors: false,
      anchors,
    },
    network: {
      pos: netPos, quat: netQuat, scale: scaledMix(0.5, 0.3), spin: 0.07, center: netCenter, bob: 0.05,
      camPos: fv(-7.6, 3.4, 9.6), camTarget: fv(0.6, 2.4, 0), land: 0, lines: 1, exposure: 0.7, push: 0.7, ice: 1, rotors: false,
    },
    villageTop: {
      pos: pieces.map((p) => p.pos), quat: pieces.map((p) => p.quat), scale: villageScale, spin: 0, center: zero, bob: 0,
      camPos: v(2, 24, 13), camTarget: v(0, 2.5, -4.5), land: 1, lines: 0, exposure: 0.85, push: 0.35, ice: 0, rotors: true,
    },
    cube: {
      pos: cubePos, quat: cubePos.map(() => tilt.clone()), scale: scaledMix(0.52, 0.52), spin: 0.16, center: cubeCenter, bob: 0.03,
      camPos: fv(0, 2.8, 10.5), camTarget: fv(-3.2, 2.6, 0), land: 0.1, lines: 0, exposure: 0.7, push: 0.8, ice: 1, rotors: false,
    },
    lost: {
      pos: lostPos, quat: shatterQuat, scale: shatterScale, spin: 0.03, center: sc, bob: 0.3,
      camPos: fv(0, 2.5, 12), camTarget: fv(0, 2.5, 0), land: 0.25, lines: 0, exposure: 0.9, push: 1.2, ice: 1, rotors: false,
    },
  };

  return {
    count: N, kinds, local, boxCount, domeCount, tints, layouts, edges, rand, delay, lambda, burstDir, tumbleAxis,
    windmills, rotorMill, rotorLocal, rotorLocalQ,
  };
}
