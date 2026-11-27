/** A2: the destructible cell terrain. Materials are re-exported as a namespace from the engine root (`materials`). */
export * from "./constants";
export { Terrain, type TerrainGenerator, type TerrainOptions, type SetOptions, type TerrainStats } from "./grid";
export { DEFAULT_PHYSICS, type PhysicsConfig } from "./physics";
export { DEFAULT_STRUCTURE, type StructureConfig } from "./structure";
export { FallingChunk } from "./falling";
export { createCarveResult, type CarveResult } from "./carve";
export { createRayHit, type RayHit } from "./query";
export { TerrainRng, hashCell } from "./random";
