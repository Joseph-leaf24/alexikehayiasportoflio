import { describe, expect, it } from "vitest";
import { projectData, PROJECT_FILES_REPO } from "@/data/projects";
import { allProjects, profile, projectPath, specializations } from "@/data/profile";
import { SKILLS } from "@/data/skills";
import {
  BLOCK,
  buildFloes,
  buildLayouts,
  buildMoon,
  buildTerrain,
  CELL,
  KIND_DOME,
  mulberry32,
  rimZ,
  SEA_LEVEL,
  terrainHeight,
  voxelTop,
} from "@/components/gl/layouts";
import * as shaders from "@/components/gl/shaders";

describe("portfolio content", () => {
  it("has a complete profile without any email address", () => {
    expect(profile.name).toBe("Alexi Kehayias");
    expect(profile.linkedin).toMatch(/^https:\/\/www\.linkedin\.com\//);
    expect(profile.cv).toMatch(/Alexi_Kehayias_CV\.pdf$/);
    expect(JSON.stringify(profile)).not.toMatch(/@[a-z0-9-]+\.[a-z]{2,}/i);
  });

  it("maps every specialization to a category with projects", () => {
    expect(specializations).toHaveLength(6);
    for (const s of specializations) {
      expect(projectData[s.id].projects.length).toBeGreaterThan(0);
    }
    expect(allProjects).toHaveLength(9);
    expect(projectPath("nlp", 1)).toBe("/project/nlp/1");
  });

  it("gives every project the fields the cards and detail pages need", () => {
    for (const { project } of allProjects) {
      expect(project.title.length).toBeGreaterThan(3);
      expect(project.description.length).toBeGreaterThan(20);
      expect(project.image).toBeTruthy();
      expect(project.tags.length).toBeGreaterThan(0);
      expect(project.concepts.length).toBeGreaterThan(0);
      if (project.githubUrl) expect(project.githubUrl.startsWith(`${PROJECT_FILES_REPO}/`)).toBe(true);
    }
  });

  it("lists the toolbox", () => {
    expect(SKILLS).toHaveLength(18);
    expect(new Set(SKILLS.map((s) => s.name)).size).toBe(18);
  });
});

describe("village + formations", () => {
  const set = buildLayouts();

  it("is deterministic", () => {
    const again = buildLayouts();
    expect(again.count).toBe(set.count);
    expect(again.layouts.village.pos[42].toArray()).toEqual(set.layouts.village.pos[42].toArray());
    const a = mulberry32(1);
    const b = mulberry32(1);
    expect(a()).toBe(b());
  });

  it("gives every formation one transform per piece", () => {
    for (const [key, lay] of Object.entries(set.layouts)) {
      expect(lay.pos, key).toHaveLength(set.count);
      expect(lay.quat, key).toHaveLength(set.count);
      expect(lay.scale, key).toHaveLength(set.count);
      expect(lay.pos.every((p) => Number.isFinite(p.x + p.y + p.z)), key).toBe(true);
    }
    expect(set.boxCount + set.domeCount).toBe(set.count);
    expect(set.tints).toHaveLength(set.count * 4);
  });

  it("builds churches, windmills with rotors and a neural network for the toolbox", () => {
    expect(Array.from(set.kinds).filter((k) => k === KIND_DOME).length).toBe(set.domeCount);
    expect(set.domeCount).toBeGreaterThan(5);
    expect(set.windmills).toHaveLength(3);
    const rotorPieces = Array.from(set.rotorMill).filter((m) => m >= 0);
    expect(rotorPieces.length).toBe(3 * 49);
    expect(set.edges.length).toBeGreaterThan(100);
    for (const [a, b] of set.edges) {
      expect(set.layouts.network.pos[b].x).toBeGreaterThan(set.layouts.network.pos[a].x);
    }
    expect(set.layouts.network.lines).toBe(1);
    expect(set.layouts.village.rotors).toBe(true);
  });

  it("stacks the Work bar chart on a common baseline", () => {
    const ys = set.layouts.bars.pos.map((p) => p.y);
    expect(Math.min(...ys)).toBeCloseTo(set.layouts.bars.pos[0].y, 5);
    expect(set.layouts.bars.scale[0].toArray()).not.toEqual(BLOCK.toArray());
  });

  it("generates a continuous caldera terrain", () => {
    for (const x of [-12, 0, 9]) {
      const rim = rimZ(x);
      expect(terrainHeight(x, rim + 4)).toBeLessThan(0); // sea
      expect(terrainHeight(x, rim - 4)).toBeGreaterThan(terrainHeight(x, rim - 0.5)); // rises inland
      expect(Math.abs(terrainHeight(x, rim - 0.001) - terrainHeight(x, rim + 0.001))).toBeLessThan(0.05);
      expect(terrainHeight(x, rim - 12)).toBeLessThan(terrainHeight(x, rim - 8.4) + 0.5);
      expect(voxelTop(x, rim - 3) % CELL).toBeCloseTo(0, 6);
    }
  });

  it("voxelises the hills into cubes above the waterline", () => {
    const full = buildTerrain();
    const small = buildTerrain({ xMin: -10, xMax: 10, zMin: -10 });
    expect(full.count).toBeGreaterThan(small.count);
    for (let i = 0; i < small.count; i++) {
      expect(small.data[i * 4 + 1] + CELL / 2).toBeGreaterThan(SEA_LEVEL);
      expect(small.data[i * 4 + 3]).toBeGreaterThanOrEqual(0);
    }
  });

  it("builds the moon shell and ice floes", () => {
    const moon = buildMoon(7, 0.9);
    const coarse = buildMoon(7, 1.25);
    expect(moon.count).toBeGreaterThan(coarse.count);
    for (let i = 0; i < moon.count; i++) {
      const r = Math.hypot(moon.offsets[i * 3], moon.offsets[i * 3 + 1], moon.offsets[i * 3 + 2]);
      expect(r).toBeLessThanOrEqual(7);
    }
    expect(Array.from(moon.crater).some((c) => c === 1)).toBe(true);
    const floes = buildFloes(12);
    expect(floes).toHaveLength(12);
    expect(floes.every((f) => f.pos.z > rimZ(f.pos.x))).toBe(true);
  });

  it("ships every shader as GLSL source", () => {
    const sources = Object.entries(shaders).flatMap(([, v]) =>
      typeof v === "string" ? [v] : [(v as { vertexShader: string }).vertexShader, (v as { fragmentShader: string }).fragmentShader]
    );
    expect(sources.length).toBeGreaterThan(15);
    for (const src of sources) expect(src).toMatch(/void main\(\)/);
    expect(shaders.pieceFragment).toMatch(/forward propagation/);
  });
});
