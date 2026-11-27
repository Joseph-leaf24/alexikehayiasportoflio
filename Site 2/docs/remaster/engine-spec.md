# Engine 2: our own engine for the Pixel Battle remaster

The owner wants the remastered game on a new, more efficient, much stronger engine, and asked for us to build
our own. This spec is the contract for building it.

**Where.** Everything lives in `c:\Users\rossb\Desktop\alexikehayiasportoflio\Site 2\`:

| Folder | Contents |
|---|---|
| `src/engine/` | The generic engine: no React, no game content |
| `src/battle/` | The game built on it (later phases) |
| `src/components/pixel/` | The old engine, which keeps running the site and the Wallpaper until the new one is at parity |

**Why our own engine.** It's tailored to pixel art: a GPU-drawn destructible cell terrain, sprite batching, 2D
lighting, post-processing and recolouring shaders. It has no third-party runtime, as the owner wants for the
Wallpaper, a smaller bundle, and full control.

## Goals

| Goal | Target |
|---|---|
| Efficiency | 60 fps with 400 units, 8,000 particles and 64 lights on a mid laptop. The Wallpaper at 30 fps should use well under half of today's CPU (30–55% of one core). |
| Strength | Data-driven content: units, weapons, abilities, factions and missions as definitions plus behaviours. Deterministic, seeded simulation. Real structural destruction. Lighting and post effects. A debug overlay and benchmarks. Headless runs for tests. |
| Robustness | Runs unattended for hours. Never stalls: every budgeted system has a per-tick cap. Falls back to Canvas 2D without WebGL2. |
| Code standard | TypeScript strict-friendly, no `any`. Prettier at 120 columns, ESLint clean (no unused variables). Short "why" doc comments; no narration. Files under about 800 lines. |

## Phases (for orientation)

- **A. Engine:** core, terrain and renderer, built in parallel by three agents against this contract.
- **B. Port the existing game** onto it: level generation and themes, the art into atlases, the squad and the
  17 weapons, the Regime army, vehicles, bosses, cities, metro, trains, roads, ships, jets, artillery, weather,
  day/night, the finales, white-out and cards, intensity waves.
- **C. Remaster content** (see `remaster-design.md`): six factions, squad specials, ultimates, ranks, the campaign,
  war map, Director and 11 missions, vehicles, spectacle.
- **D. Integration:** a `BattleScene.tsx` replacing `PixelScene.tsx` behind the same `sceneState` contract;
  Wallpaper rebuild; e2e; performance comparison; removal of the old engine.

---

## A1. Core: `src/engine/core/` and `src/engine/loop.ts`

- **`rng.ts`.** A fast seeded PRNG (sfc32 or mulberry32) plus named streams: `streams(seed).get("ai")` returns an
  independent generator per name. Helpers: `range`, `int`, `chance`, `pick`, `weighted`, `shuffle`.
- **`ecs.ts`.** A data-oriented ECS:
  - Entities: generational ids (index in the low 20 bits, generation above); `create()`, `destroy(id)`
    (deferred until the end of the tick), `alive(id)`.
  - Components, declared once with a schema:
    - **Hot numeric** components are struct-of-arrays typed columns. Example:
      `defineComponent("pos", { x: "f32", y: "f32" })` gives `pos.x[index]`. Columns grow by doubling.
    - **Cold** components are plain objects in a sparse map: `defineTag` and `defineObject<T>`.
  - Membership via sparse sets per component (a dense entity list plus a sparse index), with bitmask
    signatures for fast multi-component checks.
  - `query(...components)` iterates the smallest set and checks the others; cached query objects avoid allocation.
  - `World` holds entities, components, resources (`world.res<T>(key)`), events, rng streams and time.
- **Core components** (owned by the core, used by everyone):

  | Component | Fields | Type |
  |---|---|---|
  | `pos` | x, y | f32 |
  | `prev` | x, y | f32 (previous tick, for render interpolation) |
  | `vel` | vx, vy | f32 |
  | `body` | hw (half width), h (height), flags | flags: u32 bitfield — grounded, flying, swimming, noGravity, collidesTerrain, solidToShots, … |
  | `health` | hp, max | f32 |
  | `team` | id | u8 (0 = neutral, 1 = squad, 2+ = factions) |
  | `kind` | def | u16 (an index into a content registry) |
  | `sprite` | frame, layer, flags (flipX, emissive, hidden), tint, scale | frame: u16; layer, flags: u8; tint: u32 RGBA; scale: f32 |
  | `life` | t | f32 (seconds left; the entity is destroyed at 0) |

- **`scheduler.ts`.** Systems are `{ name, phase, run(world, dt) }`, in phase order:
  `input → director → spawn → ai → weapons → motion → physics → projectiles → damage → death → effects → terrain → camera → late`.
  - A fixed timestep: `TICK = 1/30 s`. `world.time` accumulates; `stepFixed(world, realDt × timeScale)` runs
    0..N ticks, with a cap of 12 per call to avoid a spiral of death.
  - Before each tick, copy `pos` into `prev` for rendering.
  - Time scale = speed × slow-motion; pause = 0.
- **`events.ts`.** A typed event bus: `emit(type, payload)` queues; `flush()` delivers to subscribers between
  phases. `on(type, fn)` returns an unsubscribe function. No allocation for event types that have no listeners.
- **`spatial.ts`.** A uniform-grid spatial hash (cell 64 px), rebuilt every tick from bodies (`pos` + `body` +
  `team`). Queries take a team mask and an optional predicate, and write into a reusable output array (no
  per-query allocation):
  - `nearest(x, y, r, teamMask, filter)`
  - `inRadius(x, y, r, teamMask, out)`
  - `segment(x0, y0, x1, y1, width, teamMask, out)`
- **`registry.ts`.** `Registry<Def>`: register a def by string id, get a numeric index for the `kind` column, look
  it up by index or id. A frozen view after boot.
- **`math.ts`.** clamp, lerp, damp (frame-rate independent), angle helpers, vec2 helpers without allocation.
- **`loop.ts`.** The browser frame loop:
  - `requestAnimationFrame`, or timer-scheduled frames when `maxFps > 0` (like today's
    `PixelScene` `schedule()`, which the Wallpaper depends on);
  - speed and pause, a slow-motion hook, a visibility pause;
  - it calls `stepFixed`, then `render(alpha)`;
  - it reports fps and per-system timings to the debug stats.
- **`headless.ts`.** `runHeadless(world, seconds)` steps a world without rendering, for tests, soak runs and
  benchmarks.
- **Debug stats.** `world.stats`: per-system milliseconds (rolling averages), entity count per component, ticks
  per second. Read by the debug overlay (A3).
- **Tests and benchmarks.**
  - Unit tests in `src/test/engine-core-*.test.ts`: the ECS lifecycle (create, destroy, generations, deferred
    destroy), queries, a component column growing past capacity, scheduler order and the fixed step, determinism
    (same seed, same state), spatial queries against brute force, the event bus.
  - A Node benchmark script, `scripts/bench-engine.mjs` (or a vitest bench): 400 moving units with spatial
    queries every tick, reporting ms per tick.

## A2. Terrain: `src/engine/terrain/`

Same scale as the old engine, so its level generator and art port easily.

- **Layout.** `TILE = 8` px cells, `ROWS = 240`, a column ring of `RING = 512` columns, endless to the right.
  Index = `(col & (RING - 1)) * ROWS + row`, column-major, so a whole column is contiguous.
- **`grid.ts`.** `Terrain` holds:
  - `fg`: material id, Uint8Array;
  - `bg`: background material, Uint8Array;
  - `hp`: Uint8Array, where 255 means unbreakable;
  - `flags`: Uint8Array — loose, burning, wet, structural, reinforced;
  - `variant`: Uint8Array, a per-cell random tile variant set at write time;
  - `startCol`: the first column currently in the ring.

  Methods:
  - `get(col, row)` and `set(col, row, mat, opts)`;
  - `ensure(colEnd)`, which calls a generator callback `(terrain, colFrom, colTo)` to fill new columns, as the
    old `extend()` does, and recycles columns that have scrolled away on the left;
  - `cellAt(px, py)`.
- **`materials.ts`.** A material table: `id`, `name`, `solid`, `liquid`, `loose` (falls), `flammable`, `hp`,
  `density`, `emissive` (lava, crystals), `blast` resistance, `debris` colour. Port the old engine's 92 block
  kinds and `TILE_HP` (`src/components/pixel/level.ts` ~27-160) with the same ids, so ported levels match.
- **Dirty tracking.** Changed cells mark 16×16 chunks dirty in a bitset. `drainDirty(out)` hands the renderer the
  chunks to re-upload. Writes during generation mark their columns dirty too.
- **`query.ts`.**
  - `solidAt(px, py)`, `liquidAt`, `surfaceY(px, fromY)`
  - `lineOfSight(x0, y0, x1, y1)`: a DDA over cells
  - `raycast(x0, y0, dx, dy, maxDist, out)`: hit point, normal and material
  - `columnTop(col)`
  - `overlapsSolid(x, y, hw, h)`: for body collision
- **`physics.ts`.** Budgeted cellular physics, on active chunks only:
  - loose cells (sand, gravel, rubble, snow) fall and slide;
  - liquids (water, lava, acid) flow sideways and down with a pressure-free simple model;
  - lava next to water makes stone;
  - fire spreads to flammable cells and burns them out;
  - stable chunks go to sleep and cost nothing.

  Caps: at most N cell moves per tick (default 3,000) and at most M active chunks.
- **`carve.ts`.** `carve(x, y, r, power, out?)`:
  - reduces `hp` within the radius, falling off with distance and resistance;
  - destroys cells (to air, or to rubble as a loose cell for some materials);
  - returns debris hints (count and colours) so the game can spawn particles;
  - wakes the chunks around it.
- **`structure.ts`: real structural collapse.** After a carve, check connectivity of solid cells flagged
  structural (buildings, bridges, towers) near the hole:
  - a budgeted flood fill from the cut region finds pieces no longer connected to anchored ground;
  - a detached piece becomes falling debris. Either it converts to loose cells that rain down, or it is lifted
    out as **rigid falling chunks** that move as one (preferred for buildings: stored as a small cell sprite that
    falls, then shatters into loose rubble on impact);
  - budgets: about 4,000 cells of flood fill per tick, spread over ticks for big pieces; bounded memory.
- **Tests and benchmarks.**
  - Tests in `src/test/engine-terrain-*.test.ts`: indexing and the ring, `ensure` and recycling, the queries,
    carve, sand falling, water flowing, lava with water, a building collapsing after its lower floor is carved,
    the budgets holding (a giant collapse finishes over several ticks without one tick exceeding the cap), and
    determinism.
  - Benchmark: carve 50 blasts per second on a 512×240 world; report ms per tick.

## A3. Renderer: `src/engine/render/`

Pixel-perfect WebGL2, with a Canvas 2D fallback, and a demo.

**Contract:** `src/engine/render/types.ts`. A3 writes this file first, exactly as below (extend it only by
adding fields):

```ts
/** A rectangle of an atlas page, with its pivot. */
export interface Frame { page: number; x: number; y: number; w: number; h: number; ox: number; oy: number }

/** Builds texture atlases from canvases painted at boot (the game's pixel-art painters). */
export interface AtlasBuilder {
  add(id: string, source: HTMLCanvasElement | OffscreenCanvas, ox?: number, oy?: number): number; // returns frame index
  frame(index: number): Frame;
  index(id: string): number;
}

/** Struct-of-arrays sprite list for one frame; the game pushes, the renderer draws. Reused every frame. */
export interface SpriteBuffer {
  count: number;
  push(frame: number, x: number, y: number, layer: number, flags: number, tint: number, scale: number, rot: number): void;
  clear(): void;
}
export const SPRITE_FLIP_X = 1, SPRITE_EMISSIVE = 2, SPRITE_ADDITIVE = 4, SPRITE_OUTLINE = 8, SPRITE_SILHOUETTE = 16;

/** Particles as coloured pixels or tiny quads (no texture), batched. */
export interface ParticleBuffer {
  count: number;
  push(x: number, y: number, size: number, color: number, flags: number): void; // flags: emissive/additive
  clear(): void;
}

/** Point lights for the light pass. */
export interface LightBuffer {
  count: number;
  push(x: number, y: number, radius: number, color: number, intensity: number): void;
  clear(): void;
}

/** Screen-space text in the pixel font (atlas glyphs). */
export interface TextBuffer {
  push(text: string, x: number, y: number, color: number, scale: number, align: 0 | 1 | 2, layer: number): void;
  clear(): void;
}

export interface ParallaxLayer { frame: number; factor: number; y: number; tint: number; repeatX: boolean }

export interface Shockwave { x: number; y: number; radius: number; width: number; strength: number }

export interface Grade { exposure: number; contrast: number; saturation: number; tint: number /* RGB */ }

/** Everything the renderer needs to draw one frame. */
export interface RenderFrame {
  time: number;
  /** World-space centre of the view, zoom (1 = native pixel scale), screen shake in world px. */
  camera: { x: number; y: number; zoom: number; shakeX: number; shakeY: number };
  sky: { top: number; mid: number; bottom: number; stars: number /* 0..1 */ };
  ambient: number; // RGB ambient light; 0xffffff = full daylight
  parallax: ParallaxLayer[];
  terrain: TerrainView | null;
  sprites: SpriteBuffer; // layers: 0 back … 15 front; world space
  particles: ParticleBuffer; // world space
  lights: LightBuffer; // world space
  text: TextBuffer; // screen space (overlay)
  overlay: SpriteBuffer; // screen-space sprites (HUD, cards)
  effects: {
    shockwaves: Shockwave[];
    flash: number; // 0..1 additive white
    white: number; // 0..1 white-out
    letterbox: number; // 0..1
    vignette: number; // 0..1
    chroma: number; // 0..1 chromatic aberration punch
    bloom: number; // 0..1
    grade: Grade;
  };
}

/** What the terrain gives the renderer (A2 implements it). */
export interface TerrainView {
  rows: number; ring: number; tile: number; startCol: number;
  fg: Uint8Array; bg: Uint8Array; variant: Uint8Array; hp: Uint8Array;
  drainDirty(out: number[]): number; // chunk indices (chunk = 16×16 cells); returns count
  chunkSize: number;
  /** Atlas frames for material tiles: tiles[material * VARIANTS + variant] = frame index (8×8). */
  tiles: Int32Array;
  variants: number;
}

export interface RendererOptions { pixelScale: number; maxSprites?: number; maxParticles?: number; quality?: "high" | "mid" | "low" }

export interface Renderer {
  readonly kind: "webgl2" | "canvas2d";
  readonly atlas: AtlasBuilder;
  resize(cssWidth: number, cssHeight: number, dpr: number, pixelScale: number): void;
  /** Internal (low-res) size in world px at zoom 1. */
  readonly width: number; readonly height: number;
  render(frame: RenderFrame): void;
  destroy(): void;
}

export function createRenderer(canvas: HTMLCanvasElement, opts: RendererOptions): Renderer; // WebGL2, else canvas2d
export function createFrame(): RenderFrame; // allocates the reusable buffers
```

**WebGL2 implementation:**

- **Pixel-perfect.**
  - Draw the world into a low-resolution scene framebuffer (internal size = canvas size / pixelScale; the old
    engine used 3–4 screen px per pixel).
  - Run post-processing at that resolution, then upscale to the screen with nearest filtering.
  - Camera positions are snapped to whole pixels at zoom 1. Fractional zoom is allowed for punches.
- **Atlas.**
  - Pack canvases into 2048×2048 pages with 1 px padding, using a shelf/skyline packer.
  - Upload lazily; growing adds a page.
  - Nearest filtering, with no mipmaps.
- **Sprite batch.**
  - Instanced quads with per-instance attributes (pos, size, uv rect, pivot, flags, tint, scale, rotation,
    layer), streamed into a pre-allocated buffer with `bufferSubData`.
  - Sorted by layer, then page, with a stable order within a layer.
  - One draw call per (layer, page) run.
  - Tint multiplies; `SILHOUETTE` draws a flat tint (hit flashes, team recolour); `OUTLINE` draws a 1 px outline
    in the tint colour (selection, marked enemies).
- **Terrain.**
  - Keep a GPU texture of the visible terrain window (R8 fg, R8 bg, R8 variant, R8 hp; pack them as RGBA8), sized
    to the ring (512 × 240).
  - Re-upload only dirty chunks (`texSubImage2D`).
  - A terrain shader draws each screen pixel by looking up its cell's material and variant, then sampling the
    8×8 tile from the atlas (through a material→frame lookup texture made from `tiles`).
  - It adds edge shading from the neighbours (top highlight, side shadow, crack overlay by hp) and draws `bg`
    darker behind.
  - The visible terrain costs one full-screen pass, whatever its contents.
- **Sky and parallax.** A gradient sky shader with stars at night; parallax layers are repeated textures offset by
  the camera × factor.
- **Lighting.**
  - A half-resolution light buffer: clear to ambient, then draw additive radial lights (instanced; falloff in the
    shader).
  - Composite: scene × light buffer.
  - Emissive sprites, emissive particles and emissive materials (lava, crystals) skip darkening and feed bloom.
- **Post** (one combined pass where possible; each effect skipped when its strength is 0):
  - shockwave distortion (up to 16 waves as uniforms; radial UV offset)
  - bloom (threshold, then a downsampled two-pass blur, then add)
  - chromatic aberration punch, colour grade, vignette
  - flash and white-out, letterbox bars
- **Particles.** Instanced coloured quads of 1–3 px (size in world px), additive or normal, up to `maxParticles`.
- **Text.** A pixel-font atlas: port `pixelText` and its glyphs from `src/components/pixel/art.ts` (~2881) to
  build glyph frames. Draw through the overlay pass (screen space, unaffected by the camera and post).
- **Robustness:**
  - handle `webglcontextlost` / `webglcontextrestored` by recreating all GPU resources (the atlas sources are
    kept, so it can re-upload);
  - no allocation per frame;
  - watch every GL error in development only.

**Canvas 2D fallback** (`render/canvas2d/`): the same interface. Draws terrain from cached chunk canvases, sprites
from atlas pages with `drawImage`, particles as `fillRect`. Lighting and post are skipped, apart from flash, white,
letterbox and simple ambient darkening.

**Debug overlay** (`render/debug.ts`): `?debug=1` shows fps, frame ms, per-system ms (from `world.stats`), sprite,
particle and light counts, and draw calls, in the pixel font.

**Demo** (`src/engine/demo/`): a dev-only page (`engine-demo.html` at the Site 2 root, as an extra Vite input
only in dev, or served by `vite` dev).
- It builds a synthetic world:
  - procedural terrain, with materials painted as simple 8×8 tiles;
  - 2,000 animated sprites (painted placeholder pixel sprites);
  - 8,000 particles, 64 moving lights, a day/night cycle;
  - shockwaves on click;
  - a terrain carve on click, using A2's terrain if it's ready, or a stub `TerrainView`.
- It shows the debug overlay, to prove the performance target.
- A Playwright smoke test is optional; jsdom has no WebGL, so unit-test the pure parts (packer, sort, buffers,
  matrices) in `src/test/engine-render-*.test.ts`.

## Rules for all three agents

- **Own your folder:**

  | Agent | Owns |
  |---|---|
  | A1 | `src/engine/core/`, `src/engine/loop.ts`, `src/engine/headless.ts` |
  | A2 | `src/engine/terrain/` |
  | A3 | `src/engine/render/`, `src/engine/demo/` |

  Shared: `src/engine/index.ts` (each adds its own exports; re-read before editing). Tests go in your own
  `src/test/engine-<area>-*.test.ts` files.
- **Don't touch** `src/components/pixel/` (the old engine; another agent may still be finishing there) or the site.
- **The contracts** between you are this file's types. A3 writes `render/types.ts` first, exactly as given. A2's
  `Terrain` must implement `TerrainView`. A1's components and stats are as listed. If you must change a
  contract, add fields only, and say so in your report.
- **Checks**, from `Site 2`:
  - `npx tsc -p tsconfig.app.json --noEmit` (no errors in your files; ignore others' in-progress errors and
    mention them)
  - `npx eslint src/engine src/test/engine-*`
  - `npx prettier --check src/engine`
  - your tests
- **No state-changing git commands**, no production builds. Starting the Vite dev server to try the demo is
  fine for A3; stop it when done.
- **Report:**
  - what you built, with the API signatures;
  - performance numbers;
  - test results;
  - deviations from this spec;
  - anything the next phase must know.
