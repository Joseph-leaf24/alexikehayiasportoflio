# Engine 2: what game code must know (from the Phase A reports)

The engine is in `Site 2/src/engine/`, all exported from `src/engine/index.ts`. Read the source; it's
documented. Use `engine-spec.md` for the original contract. Measured on the owner's laptop:

| Part | Result |
|---|---|
| Core | 400-unit battle tick: 0.14 ms with the spatial hash (2.3 ms brute force) |
| Terrain | 50 blasts/s: 0.37 ms per tick; skyscraper collapse worst tick under 2 ms |
| Renderer | 2,000 sprites + 8,000 particles + 64 lights: about 0.3 ms CPU and 0.3–1.2 ms GPU, 18 draw calls; Wallpaper mode about 10% of a core at 30 fps |

## Core (`engine/core`, `engine/loop.ts`, `engine/headless.ts`)

- **World:**
  - `createWorld({ seed, capacity = 4096, spatial, coreSystems })`.
  - Components are defined per world: `world.defineComponent(name, { field: "f32" | "u32" | ["u32", default] … })`;
    `defineTag`, `defineObject<T>` (`get(id)`, `at(index)`).
  - Core components are at `world.core.pos/prev/vel/body/health/team/kind/sprite/life`.
  - Metadata is under `$` (`pos.$.size`): `$` is a reserved field name.
- **Ids:**
  - Generational ids; `NULL_ENTITY = 0`.
  - `destroy` is deferred to the end of the tick; `onDestroy` hooks can still read components.
  - `place(world, id, x, y)` sets pos and prev together (for spawns and teleports).
- **Queries:**
  - Hold on to query objects: `world.query(...)` allocates its argument list on every call.
  - `q.collect()` fills `q.idx`; read it after collecting.
  - Columns and `q.idx` are replaced when capacity grows: re-read them each system run, never keep them across
    ticks. Size `capacity` or call `world.reserve()` for the peak.
- **Body flags:** `BODY_*` — GROUNDED, FLYING, SWIMMING, NO_GRAVITY, COLLIDES_TERRAIN, SOLID_TO_SHOTS,
  SOLID_TO_UNITS, UNINDEXED, CLIMBING, INVULNERABLE. The game's own flags start at `BODY_USER` (bit 16).
  `SPRITE_HIDDEN = 128`.
- **Scheduler:**
  - `TICK = 1/30`. Phases:
    `input → director → spawn → ai → weapons → motion → physics → projectiles → damage → death → effects → terrain → camera → late`.
  - A system is `{ name, phase, run(world, dt), order?, every?, offset? }`, with `every` for 10 Hz AI and similar.
  - `stepFixed(world, realDt)` runs at most 12 ticks; `world.time` has `alpha` for interpolation and
    `speed`/`slowmo`/`paused`/`scale`.
- **Spatial hash:**
  - Rebuilt at the end of "physics": AI and weapons see last tick's positions.
  - `nearest`, `inRadius`, `segment` (hits nearest first), `rect`, `segmentFirst`, with team masks (`teamBit`,
    `enemiesOf`, `TEAM_ALL`).
  - Distances are measured to the nearest point of a body's box `[x−hw, x+hw] × [y−h, y]`; `y` is the feet.
- **Events.** `world.events`: `on`, `emit` (dropped if nobody listens), `define` + `claim` for pooled payloads.
  Extend `WorldEvents` by declaration merging on `"@/engine/core/events"`.
- **Rng:**
  - `world.rng.get(name)` gives a named stream. Keep render and effects randomness on its own stream
    (`rng.get("fx")`), never on gameplay streams.
  - `Rng` has `range`, `int`, `chance`, `pick`, `weighted`, `shuffle`, `gauss`, `angle` and more.
- **Registry.** `Registry<Def>`: indices start at 1, so `kind.def` 0 means none; `freeze()`.
- **Positions are f32.** Endless runs must call `shiftOrigin(world, dx, dy)` by a multiple of 4096 px and shift
  the terrain's `startCol` and the camera by the same amount.
- **Loop.**
  - `new Loop(world, { render(alpha, frameDt), maxFps, slowmo(), sleep(), maxFrameDt, onFrame, host })` with
    `start`, `stop`, `pause`, `resume`, `speed`, `maxFps`.
  - With `maxFps` > 0 it waits on a timer (as `PixelScene` did, for the Wallpaper); it sleeps while hidden.
- **Unattended runs** set `world.onError`, and keep `stats.timing` off unless the debug overlay is up.
- `runHeadless(world, seconds, { onTick, maxWallMs })` for tests and benchmarks; `hashWorld(world)` for
  determinism checks.

## Terrain (`engine/terrain`)

- **Layout.** `TILE` 8, `ROWS` 240, `RING` 512 (column-major), `CHUNK` 16 (`CHUNKS` 480), `VARIANTS` 4.
  Materials 0–91 are the old engine's ids exactly; ACID is 92, RUBBLE 93, BEAM 94 (a load-bearing cell that isn't
  solid and has no tile, so the back wall or sky shows through: what holds up things drawn on legs or pillars).
  `materials.*` is a namespace on the engine root.
- **Construction.** `new Terrain({ seed, generate: (t, colFrom, colTo) => …, overshoot = 128, physics, structure })`.
  It implements `TerrainView`.
- **Access and simulation:**
  - `get`/`set(col, row, mat, { bg, hp, flags, structural, reinforced, variant })`, `setBg`, `cellAt`
  - `ensure(colEnd)` keeps 128 columns ahead (384 stored); `step(dt)`
  - `carve(x, y, r, power, out?)` returns destroyed and damaged counts, up to 8 debris colours and 32 sample
    points, for particles
  - `damage`, `ignite`, `flood`
- **Queries:** `solidAt`, `liquidAt`, `materialAt`, `surfaceY(px, fromY)`, `columnTop`, `lineOfSight` (ignores
  the end-point cells), `raycast(…, out: RayHit)`, `overlapsSolid(x, feetY, hw, h)`.
- **Events and effects:** `drainEvents(out)` (steam, ignite, extinguish, burn-out, dissolve, detach, impact,
  crush, as groups of four) and `collectBurning(from, to, out)`, for flames and lights.
- **Structures:**
  - Cells flagged structural (brick, concrete, glass, wood and so on by default; pass `{ structural: true }` for
    stone or sandstone used as buildings) are support-checked after damage.
  - Unsupported or overloaded structures fall as **rigid chunks** in `terrain.falling` (column-major cells, `y`
    and `prevY`). The game must draw them, and crush units with their bounds.
  - **Buildings need ground contact.** The old generator cut doorways through the bottom rows of both
    ground-floor walls, so the upper floors would topple at once. Keep door frames and pillars down to the ground.
    The battle's `level/generator.ts` does this (door leaves, beams, stone frames).
  - Ground resting on a settled loose pile (sand, gravel) counts as anchored, as a structure resting on one does.
- **Liquids flow now.** Water, lava and acid move (old water was static), so "hanging" water from old generators
  will run once something wakes it.
- **Tiles.** Fill `terrain.tiles` (frame index = material × 4 + variant) from atlas frames.

## Renderer (`engine/render`)

- **Setup.** `createRenderer(canvas, { pixelScale, maxSprites, maxParticles, quality, prefer, gpuTiming })`
  gives WebGL2, falling back to Canvas 2D. `createFrame()` and `clearFrame(frame)` each frame.
- **Buffers.** They must come from `createFrame()`: the renderer reads the struct-of-arrays.
- **Atlas.**
  - `renderer.atlas.add(id, canvas, ox, oy)` returns a frame index.
  - Paint art once at boot and push frame indices each frame. Sources are kept for context-loss re-upload.
- **Sprite layers.**
  - Layers 0–15. Layers 0–3 draw behind the terrain (`terrainLayer` default 4); particles draw after layer 11.
  - Sprite flags: `FLIP_X`, `EMISSIVE`, `ADDITIVE`, `OUTLINE`, `SILHOUETTE`.
  - Tints are `0xRRGGBBAA`; sky, light, ambient and grade colours are RGB.
- **Effects.** Shockwaves (up to 16), flash, white-out, letterbox, vignette, chroma, bloom, grade: set them on
  `frame.effects`.
- **Text.**
  - `frame.text.push` (screen space, pixel font).
  - `measureText`, `GLYPHS`, and `INK` (the font's ink colours).
  - Glyphs include `/ % ( ) [ ] = ' " < > * # _ | ; ·`.
- **`DebugOverlay`.** `debugFromUrl()` (`?debug=1`); `.tick(dtMs)`; `.draw(frame, renderer, world.stats)`.
- **Canvas sizing.** `resize(cssW, cssH, dpr, pixelScale)`: the canvas can overshoot its container by under
  one pixel-scale, so clip the container.
- **Not reproduced yet:** the old per-block ink outlines and water depth gradient (add as an optional
  per-material field if wanted). The Canvas 2D fallback ignores tint colour and outlines.
- **Demo.** `engine-demo.html` with `src/engine/demo/` is a working example of the whole pipeline (terrain, atlas
  tiles, falling chunks drawn as sprites, lights, effects, loop). Copy its patterns.
