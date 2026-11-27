/** Engine 2: the generic engine (no React, no game content). Each area adds its own exports here. */

// A1: core, frame loop, headless runs
export * from "./core";
export * from "./loop";
export * from "./headless";

// A2: destructible terrain (material ids and tables under `materials`, e.g. `materials.STONE`)
export * from "./terrain";
export * as materials from "./terrain/materials";

// A3: renderer (WebGL2 with a Canvas 2D fallback), pixel font, debug overlay
export * from "./render";
