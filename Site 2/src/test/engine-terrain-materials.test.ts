import { describe, expect, it } from "vitest";
import * as old from "@/components/pixel/level";
import { DEBRIS_COLOR } from "@/components/pixel/art";
import * as M from "@/engine/terrain/materials";

const { MATERIALS, OLD_KINDS, MATERIAL_COUNT } = M;

describe("material table", () => {
  it("keeps the old engine's 92 block kinds with the same ids", () => {
    expect(OLD_KINDS).toBe(old.KINDS);
    expect(MATERIALS.length).toBe(MATERIAL_COUNT);
    // Every block constant the old level exports has the same id here
    const oldIds = Object.entries(old).filter(
      ([k, v]) => /^[A-Z]+$/.test(k) && typeof v === "number" && v < old.KINDS && k !== "TILE" && k !== "ROWS"
    );
    expect(oldIds.length).toBeGreaterThanOrEqual(old.KINDS);
    const ours = M as unknown as Record<string, unknown>;
    for (const [name, id] of oldIds) {
      expect(ours[name], name).toBe(id);
      expect(MATERIALS[id as number].name).toBe(name.toLowerCase());
    }
  });

  it("matches the old SOLID, TILE_HP and LOOSE tables id for id", () => {
    for (let k = 0; k < old.KINDS; k++) {
      expect(M.SOLID[k], `solid ${k}`).toBe(old.SOLID[k]);
      expect(M.HP[k], `hp ${k}`).toBe(old.TILE_HP[k]);
      expect(M.LOOSE[k], `loose ${k}`).toBe(old.LOOSE[k]);
    }
  });

  it("uses the old debris colours", () => {
    for (let k = 1; k < old.KINDS; k++) {
      if (k === M.LAVA) continue; // the old table had no lava colour (lava never broke)
      expect(M.DEBRIS[k], MATERIALS[k].name).toBe(parseInt(DEBRIS_COLOR[k].slice(1), 16));
    }
  });

  it("adds acid and rubble after the old kinds", () => {
    expect(M.ACID).toBe(92);
    expect(M.RUBBLE).toBe(93);
    expect(MATERIALS[M.ACID]).toMatchObject({ liquid: true, solid: false, hp: 255 });
    expect(MATERIALS[M.RUBBLE]).toMatchObject({ loose: true, solid: true });
  });

  it("is consistent", () => {
    const names = new Set(MATERIALS.map((m) => m.name));
    expect(names.size).toBe(MATERIALS.length);
    MATERIALS.forEach((m, id) => {
      expect(m.id).toBe(id);
      expect(M.materialByName(m.name)).toBe(m);
      if (m.liquid) {
        expect(m.solid, m.name).toBe(false);
        expect(m.flow, m.name).toBeGreaterThan(0);
        expect(m.hp, m.name).toBe(255);
      }
      if (m.flammable) {
        expect(m.fuel, m.name).toBeGreaterThan(0);
        expect(m.burnsTo, m.name).toBeLessThan(MATERIAL_COUNT);
      }
      if (m.soft) expect(m.solid, m.name).toBe(false);
      expect(m.blast).toBeGreaterThan(0);
    });
    expect([M.WATER, M.LAVA, M.ACID].map((m) => M.LIQUID[m])).toEqual([1, 1, 1]);
    expect(M.EMISSIVE[M.LAVA]).toBeGreaterThan(0);
    expect(M.EMISSIVE[M.CRYSTAL]).toBeGreaterThan(0);
    expect(M.EMISSIVE[M.STONE]).toBe(0);
    // Heavier powders sink in water, snow and ash float
    expect(M.DENSITY[M.SAND]).toBeGreaterThan(M.DENSITY[M.WATER]);
    expect(M.DENSITY[M.SNOW]).toBeLessThan(M.DENSITY[M.WATER]);
    expect(M.DENSITY[M.ACID]).toBeGreaterThan(M.DENSITY[M.WATER]);
  });

  it("treats unknown ids as air in every flat table", () => {
    for (const table of [M.SOLID, M.LIQUID, M.LOOSE, M.FLAMMABLE, M.HP, M.BEHAVIOUR, M.STRUCTURAL]) {
      expect(table.length).toBe(256);
      expect(table[200]).toBe(0);
    }
    expect(M.BLAST_TAKEN[200]).toBe(1);
  });
});
