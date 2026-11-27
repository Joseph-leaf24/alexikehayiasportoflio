/**
 * The material table. Ids 0–91 are the old engine's block kinds (`components/pixel/level.ts`) with the same ids, hit
 * points, solidity and looseness, so ported levels and art match; 92+ are new (acid, rubble).
 *
 * Each property is also a flat typed array of 256 entries indexed by material id, for the hot loops: any byte read
 * from a grid indexes safely, and unknown ids behave as air.
 */

/* Old engine block kinds, same ids. */
export const AIR = 0;
export const GRASS = 1;
export const DIRT = 2;
export const STONE = 3;
export const BEDROCK = 4;
export const BRICK = 5;
export const CONCRETE = 6;
export const CRATE = 7;
export const SANDBAG = 8;
export const WOOD = 9;
export const STEEL = 10;
export const WINDOW = 11;
export const LEAF = 12;
export const TRUNK = 13;
export const CRYSTAL = 14;
export const ASH = 15;
export const ROCK = 16;
export const METAL = 17;
export const LAVA = 18;
export const HAZARD = 19;
export const PIPE = 20;
export const SAND = 21;
export const SANDSTONE = 22;
export const SNOW = 23;
export const ICE = 24;
export const PINE = 25;
export const HIVE = 26;
export const SLIME = 27;
export const MUD = 28;
export const REDROCK = 29;
export const MOSS = 30;
export const VINE = 31;
export const WATER = 32;
export const BAMBOO = 33;
export const LADDER = 34;
export const FENCE = 35;
export const TIN = 36;
export const GLASS = 37;
export const MARBLE = 38;
export const PLASTER = 39;
export const SIDING = 40;
export const STUCCO = 41;
export const TILES = 42;
export const DARKBRICK = 43;
export const ASPHALT = 44;
export const PAVEMENT = 45;
export const COAL = 46;
export const IRON = 47;
export const GOLD = 48;
export const RAIL = 49;
export const SEWER = 50;
export const PARQUET = 51;
export const CARPET = 52;
export const TIMBER = 53;
export const ADOBE = 54;
export const TERRACOTTA = 55;
export const JADE = 56;
export const SLATE = 57;
export const SHOJI = 58;
export const AZULEJO = 59;
export const LIMESTONE = 60;
export const CLAY = 61;
export const GRAVEL = 62;
export const GRANITE = 63;
export const BASALT = 64;
export const LOAM = 65;
export const CHALK = 66;
export const REDLEAF = 67;
export const PINKLEAF = 68;
export const SHALE = 69;
export const QUARTZ = 70;
export const OBSIDIAN = 71;
export const MAGMA = 72;
export const SULFUR = 73;
export const PERMAFROST = 74;
export const GLACIER = 75;
export const SCREE = 76;
export const PEAT = 77;
export const LATERITE = 78;
export const SALT = 79;
export const CORAL = 80;
export const COBBLE = 81;
export const FOSSIL = 82;
export const COPPER = 83;
export const AMETHYST = 84;
export const MYCELIUM = 85;
export const MOSSROCK = 86;
export const DRYGRASS = 87;
export const ROOTS = 88;
export const TUFF = 89;
export const DIAMOND = 90;
export const REINFORCED = 91;
/** The old engine's kind count. */
export const OLD_KINDS = 92;

/* New materials. */
/** Glowing liquid that eats whatever it touches (Brood spitters, acid rain). */
export const ACID = 92;
/** Loose building debris: what masonry leaves behind and what collapsed buildings shatter into. */
export const RUBBLE = 93;
/**
 * A load-bearing cell with no block of its own: what holds up a structure drawn on legs, posts or back-wall pillars.
 * Bodies and shots pass it; blasts break it, and then what it held comes down.
 */
export const BEAM = 94;
export const MATERIAL_COUNT = 95;

/** One material's behaviour. */
export interface Material {
  readonly id: number;
  readonly name: string;
  /** Stops bodies and shots. */
  readonly solid: boolean;
  /** Flows (water, lava, acid). */
  readonly liquid: boolean;
  /** Falls and slides like sand. */
  readonly loose: boolean;
  readonly flammable: boolean;
  /** Hits to break; 255 never breaks. */
  readonly hp: number;
  /** Relative weight: heavier loose cells sink through lighter liquids; also a structure's load. */
  readonly density: number;
  /** Glow 0..255, lit regardless of ambient light. */
  readonly emissive: number;
  /** Divides incoming blast damage; 1 is ordinary. */
  readonly blast: number;
  /** Representative colour, 0xRRGGBB, for particles. */
  readonly debris: number;
  /** Cells of this material are flagged structural when written, unless told otherwise. */
  readonly structural: boolean;
  /** Foliage that falling cells crush. */
  readonly soft: boolean;
  /** Acid can't eat it. */
  readonly acidProof: boolean;
  /** Chance 0..1 that a destroyed cell leaves loose rubble instead of air. */
  readonly rubble: number;
  /** What a burnt-out cell becomes. */
  readonly burnsTo: number;
  /** Ticks a fire burns in it. */
  readonly fuel: number;
  /** Load (in cells of density 20) one supporting contact of this material carries. */
  readonly strength: number;
  /**
   * Liquids: how far sideways (cells) a cell looks for somewhere lower to run to. With no pressure, a surface can
   * rest with 1-row steps at most this far apart, so longer reach means flatter lakes (it costs only edge cells).
   */
  readonly flow: number;
  /** Liquids: moves once every this many ticks. */
  readonly viscosity: number;
}

type Spec = Partial<Omit<Material, "id" | "name" | "debris">>;

/* Shorthands for the table below. */
const soil = (density = 16): Spec => ({ density, strength: 60 });
const rock = (density = 26): Spec => ({ density, strength: 120 });
const fuel = (ticks: number, burnsTo = ASH): Spec => ({ flammable: true, fuel: ticks, burnsTo });
const build = (density: number, strength: number, rubble = 0): Spec => ({
  structural: true,
  density,
  strength,
  rubble,
});
const foliage: Spec = { soft: true, density: 2, strength: 5, ...fuel(18, AIR) };

/** [name, debris colour, behaviour] per id. Colours are the old engine's `DEBRIS_COLOR`. */
const TABLE: [string, number, Spec][] = [
  ["air", 0x000000, { density: 0, strength: 0 }],
  ["grass", 0x5bbf3a, soil()],
  ["dirt", 0x8a5a2b, soil()],
  ["stone", 0x6d7184, rock()],
  ["bedrock", 0x2f2b3a, rock(30)],
  ["brick", 0xa24c36, build(19, 150, 0.35)],
  ["concrete", 0x8a90a0, build(24, 250, 0.4)],
  ["crate", 0xb07a3a, { ...build(5, 30), ...fuel(60, AIR) }],
  ["sandbag", 0xc4ae78, build(16, 120)],
  ["wood", 0x8a5a2b, { ...build(7, 80), ...fuel(120) }],
  ["steel", 0x5d6880, { ...build(60, 900), blast: 1.5 }],
  ["window", 0x3d5a80, build(10, 20)],
  ["leaf", 0x3f9a3a, foliage],
  ["trunk", 0x7a5230, { density: 7, ...fuel(150, AIR) }],
  ["crystal", 0x5ff0e0, { ...rock(24), emissive: 200 }],
  ["ash", 0x8a8088, { density: 4 }],
  ["rock", 0x3a2f33, rock()],
  ["metal", 0x6c7488, { ...build(50, 600), blast: 1.25 }],
  ["lava", 0xff7a20, { density: 30, emissive: 255, flow: 6, viscosity: 3 }],
  ["hazard", 0xffcc33, build(30, 200)],
  ["pipe", 0x5a7a6a, { density: 30, strength: 100 }],
  ["sand", 0xe0c080, { density: 16 }],
  ["sandstone", 0xc89a60, { ...rock(22), rubble: 0.15 }],
  ["snow", 0xeef2f8, { density: 5 }],
  ["ice", 0xa8d8f0, rock(9)],
  ["pine", 0x2a5a40, foliage],
  ["hive", 0x5a2a6a, { density: 10, strength: 60, ...fuel(70) }],
  ["slime", 0x7aff5a, soil(12)],
  ["mud", 0x4a3a24, soil(18)],
  ["redrock", 0xb8543a, rock()],
  ["moss", 0x5a7a4a, rock()],
  ["vine", 0x3f9a3a, foliage],
  ["water", 0x5ab0e0, { density: 10, flow: 32, viscosity: 1 }],
  ["bamboo", 0x9ab040, { density: 6, ...fuel(45, AIR) }],
  ["ladder", 0x9c6834, { ...build(6, 15), ...fuel(45, AIR) }],
  ["fence", 0x9aa1b0, build(8, 15)],
  ["tin", 0x8a90a0, build(12, 60)],
  ["glass", 0x8fd3ff, { ...build(25, 20), acidProof: true }],
  ["marble", 0xe8e4dc, build(27, 200, 0.3)],
  ["plaster", 0xe0d4b8, build(12, 60, 0.3)],
  ["siding", 0x5a8ab0, { ...build(8, 60), ...fuel(90) }],
  ["stucco", 0xe8a8a0, build(16, 80, 0.3)],
  ["tiles", 0xe8eef0, build(20, 100, 0.2)],
  ["darkbrick", 0x5a3a3a, build(19, 150, 0.35)],
  ["asphalt", 0x3a3a40, { ...rock(23), rubble: 0.2 }],
  ["pavement", 0xa8a8b0, { ...rock(23), rubble: 0.25 }],
  ["coal", 0x2a2a30, { ...rock(14), ...fuel(240) }],
  ["iron", 0xc8906a, rock(40)],
  ["gold", 0xffd23f, rock(60)],
  ["rail", 0x7a6a5a, { density: 40, strength: 60 }],
  ["sewer", 0x4a5a4a, rock(24)],
  ["parquet", 0xb07a40, { ...build(7, 60), ...fuel(90) }],
  ["carpet", 0x8a3a4a, { ...build(4, 20), ...fuel(60) }],
  ["timber", 0xe8dcc0, { ...build(8, 100), ...fuel(120) }],
  ["adobe", 0xc08a5a, build(17, 80, 0.4)],
  ["terracotta", 0xc8603a, build(18, 80, 0.3)],
  ["jade", 0x3a8a6a, build(27, 150, 0.2)],
  ["slate", 0x4a4e5a, build(27, 120, 0.2)],
  ["shoji", 0xf0e8d8, { ...build(3, 10), ...fuel(24, AIR) }],
  ["azulejo", 0x3a6ab0, build(20, 100, 0.2)],
  ["limestone", 0xe0d8c0, { ...rock(25), rubble: 0.15 }],
  ["clay", 0xb87a5a, soil(18)],
  ["gravel", 0x8a8478, { density: 18 }],
  ["granite", 0x7a6a70, rock(27)],
  ["basalt", 0x2e2e36, rock(29)],
  ["loam", 0x5a3a22, soil()],
  ["chalk", 0xe8e4d8, rock(22)],
  ["redleaf", 0xd8642a, foliage],
  ["pinkleaf", 0xf0a0c8, foliage],
  ["shale", 0x525a68, rock(25)],
  ["quartz", 0xe8e4ee, { ...rock(26), acidProof: true }],
  ["obsidian", 0x2a2036, { ...rock(24), blast: 1.25 }],
  ["magma", 0xff6a1a, { ...rock(28), emissive: 170 }],
  ["sulfur", 0xe8d040, { ...rock(20), ...fuel(45, AIR) }],
  ["permafrost", 0x7a6a5a, soil(18)],
  ["glacier", 0x7ab8e8, rock(9)],
  ["scree", 0x8a8a8e, { density: 18 }],
  ["peat", 0x3a2a1c, { ...soil(10), ...fuel(200) }],
  ["laterite", 0xb85a30, soil(18)],
  ["salt", 0xf0e8e8, { density: 14 }],
  ["coral", 0xe86a7a, rock(18)],
  ["cobble", 0x8a8a90, { ...rock(25), rubble: 0.3 }],
  ["fossil", 0xe8dcc0, rock(24)],
  ["copper", 0x4ab08a, rock(40)],
  ["amethyst", 0xa85ae8, rock(26)],
  ["mycelium", 0x7affd8, { ...soil(8), emissive: 150 }],
  ["mossrock", 0x6a9a3a, rock()],
  ["drygrass", 0xd8b450, { ...soil(), ...fuel(24, DIRT) }],
  ["roots", 0xa07a50, { ...soil(), ...fuel(70, AIR) }],
  ["tuff", 0x8a7e72, rock(20)],
  ["diamond", 0xaaf4ff, { ...rock(35), blast: 1.5, acidProof: true }],
  ["reinforced", 0x7a7e88, { ...build(25, 1200, 0.5), blast: 2 }],
  ["acid", 0x9aff3a, { density: 12, emissive: 140, flow: 16, viscosity: 1 }],
  ["rubble", 0x7e7670, { density: 20, strength: 40 }],
  ["beam", 0x7a7e88, build(4, 60)],
];

/* The old engine's own tables, copied verbatim so ids keep their behaviour. */
const OLD_NON_SOLID = [AIR, LEAF, TRUNK, LAVA, PINE, VINE, WATER, BAMBOO, LADDER, FENCE, RAIL, REDLEAF, PINKLEAF];
// prettier-ignore
const OLD_TILE_HP = [
  0, 2, 2, 4, 255, 3, 5, 1, 2, 1, 6, 1, 1, 1, 4, 2, 5, 6, 255, 3, 3, 1, 3, 1, 3, 1, 3, 2, 2, 4, 5, 1, 255, 1, 1, 1, 3,
  1, 5, 2, 2, 3, 3, 4, 4, 3, 4, 5, 5, 1, 4, 2, 2, 3, 3, 2, 3, 3, 1, 3, 4, 2, 1, 9, 8, 2, 2, 1, 1,
  3, 5, 7, 4, 2, 3, 4, 1, 2, 2, 2,
  2, 4, 4, 5, 4, 2, 4, 2, 2, 2, 6,
  32,
];
const OLD_LOOSE = [SAND, GRAVEL, SNOW, SCREE, ASH, SALT];

const NEW: Record<number, { solid: boolean; liquid: boolean; loose: boolean; hp: number }> = {
  [ACID]: { solid: false, liquid: true, loose: false, hp: 255 },
  [RUBBLE]: { solid: true, liquid: false, loose: true, hp: 1 },
  [BEAM]: { solid: false, liquid: false, loose: false, hp: 4 },
};

/** Every material, indexed by id. */
export const MATERIALS: readonly Material[] = TABLE.map(([name, debris, spec], id): Material => {
  const added = NEW[id];
  return Object.freeze({
    id,
    name,
    solid: added ? added.solid : !OLD_NON_SOLID.includes(id),
    liquid: added ? added.liquid : id === WATER || id === LAVA,
    loose: added ? added.loose : OLD_LOOSE.includes(id),
    flammable: false,
    hp: added ? added.hp : OLD_TILE_HP[id],
    density: 20,
    emissive: 0,
    blast: 1,
    debris,
    structural: false,
    soft: false,
    acidProof: false,
    rubble: 0,
    burnsTo: AIR,
    fuel: 0,
    strength: 100,
    flow: 0,
    viscosity: 1,
    ...spec,
  });
});

const byName = new Map(MATERIALS.map((m) => [m.name, m]));
/** A material by its lowercase name ("stone", "acid"). */
export const materialByName = (name: string): Material | undefined => byName.get(name);

const table = (pick: (m: Material) => number | boolean) => {
  const out = new Uint8Array(256);
  for (const m of MATERIALS) out[m.id] = Number(pick(m));
  return out;
};

/* Flat per-id tables for the hot loops. */
export const SOLID = table((m) => m.solid);
export const LIQUID = table((m) => m.liquid);
export const LOOSE = table((m) => m.loose);
export const FLAMMABLE = table((m) => m.flammable);
export const HP = table((m) => m.hp);
export const DENSITY = table((m) => m.density);
export const EMISSIVE = table((m) => m.emissive);
export const SOFT = table((m) => m.soft);
export const ACIDPROOF = table((m) => m.acidProof);
export const STRUCTURAL = table((m) => m.structural);
export const BURNS_TO = table((m) => m.burnsTo);
export const FUEL = table((m) => m.fuel);
export const FLOW = table((m) => m.flow);
export const VISCOSITY = table((m) => Math.max(1, m.viscosity));
/** Rubble chance scaled to 0..255, compared against a random byte. */
export const RUBBLE_CHANCE = table((m) => Math.round(m.rubble * 255));
/** Blast resistance, as a multiplier on incoming damage (1 / blast). */
export const BLAST_TAKEN = Float32Array.from({ length: 256 }, (_, k) => (MATERIALS[k] ? 1 / MATERIALS[k].blast : 1));
/** Structural load of one cell (density / 20) and what one supporting contact of it carries. */
export const WEIGHT = Float32Array.from({ length: 256 }, (_, k) => (MATERIALS[k] ? MATERIALS[k].density / 20 : 0));
export const STRENGTH = Float32Array.from({ length: 256 }, (_, k) => (MATERIALS[k] ? MATERIALS[k].strength : 0));
export const DEBRIS = Uint32Array.from({ length: 256 }, (_, k) => (MATERIALS[k] ? MATERIALS[k].debris : 0));
/** 1 for powders (fall and slide), 2 for liquids: lets the physics skip static cells with one read. */
export const BEHAVIOUR = table((m) => (m.liquid ? 2 : m.loose ? 1 : 0));
