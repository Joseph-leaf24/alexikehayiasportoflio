# Phase B0 brief: the game skeleton on Engine 2

Read first:
1. `docs/remaster/engine-notes.md`
   (what game code must know about the new engine)
2. `docs/remaster/engine-spec.md`
   (the engine's design and phases)
3. `docs/remaster/PLAN.md`
   (where the game is going)
4. `docs/remaster/old-engine-map.md`
   (a map of the OLD engine in `src/components/pixel/`, which you port from)

Project: `c:\Users\rossb\Desktop\alexikehayiasportoflio\Site 2\`. The new engine is in `src/engine/` (done and
tested). The old engine (`src/components/pixel/`) still runs the site: don't change it. Import from it read-only
where reuse saves work (its art painters especially).

## Your job

Build `src/battle/`, the game's architecture on the new engine, and prove it with a **playable vertical slice**.
After you, four agents port the old game in parallel (B1–B4), then six agents add the remaster content (C). They
all build inside the structure you set up, so it decides how good and how maintainable everything becomes. Make it
clean, fast, data-driven and easy to extend without touching shared files.

## Architecture to set up

```
src/battle/
  game/      composition root: BattleGame, options, budgets, game components, events, content registries, install list
  physics/   unit locomotion on the cell terrain; projectiles base; explosions; damage → death pipeline
  render/    ECS → RenderFrame extraction, camera, sky/parallax/day-night, CPU particles, lights, banners/text
  art/       the art book: paints sprites into the atlas at boot (reuses the old painters), id → frame lookup
  level/     placeholder generator now          → B1 ports the real generator, themes and props
  squad/     placeholder squad AI + rifle now   → B2: players, the 17 weapons, squad AI, crates, respawns
  enemies/   placeholder infantry now           → B3: the Regime army, its vehicles and bosses, allies
  world/     empty                              → B4: cities, traffic, metro, trains, ships, jets, artillery, weather
  flow/      minimal level loop now             → B4: phases, intensity waves, finales, white-out, cards
  (C later: factions/, squad abilities/, campaign/, spectacle/, vehicles/)
  dev/       battle-dev.html entry (dev-only, like engine-demo.html; never in the production build)
```

- **Feature folders.** Each one exports `install(game: BattleGame): void`, which registers its content defs, adds
  its systems (phase and order), and subscribes to events. `game/installs.ts` lists the folders in order. A later
  agent adds a folder by writing its `install` and adding one line there.
- **`BattleGame`.**
  - It owns the `World` (`createWorld`), the `Terrain`, the content registries (units, weapons, projectiles,
    effects, and so on, as typed `Registry<Def>`), the per-game resources, budgets, options and rng streams.
  - API:
    - `step(realDt)`, which uses `stepFixed`;
    - `buildFrame(frame, alpha)`, the render extraction;
    - `resize(...)`, `dispose()`;
    - a small facade of common helpers (spawn a unit, explode, emit an effect, say a banner), so content code
      stays short.
- **Content as defs plus behaviour.** For example, a `UnitDef` has id, size, hp, speed, team defaults, sprite
  set, an AI behaviour (a function or a small state-machine spec) and a weapon id. A `WeaponDef` has cadence,
  burst, spread, range and a projectile id or a hitscan behaviour. A `ProjectileDef` has speed, gravity, radius,
  damage, pierce, explode and a trail. Write the def types so the 17 old weapons and about 46 old enemy kinds can
  all be expressed without special cases in shared code.
- **Game components.** Unit (def index, AI state object), weapon state, player tag (id 0–7), projectile, effect
  emitters, and so on. Hot numbers go in typed columns, rare state in object components.
- **Events.** Extend `WorldEvents` by declaration merging (`unitKilled`, `playerDown`, `blast`, `banner`,
  `levelStart`, `levelEnd` and so on) with pooled payloads where they're hot.
- **Locomotion** (`physics/locomotion.ts`). Make it robust; everything walks on it:
  - walking on uneven cell terrain with step-up of 1–2 cells and gravity (620 px/s², as in the old engine);
  - jumping gaps and walls, climbing ladders and walls where allowed, swimming in liquids, parachutes;
  - knockback and being flung by blasts;
  - never getting stuck inside terrain (push out), and never falling through it at high speed (sub-step).

  Port the old feel from `game.ts` (`updatePlayer`, `integrate`, `solid`), using the terrain's `overlapsSolid`,
  `surfaceY` and `raycast`. Crush units under falling rigid chunks (`terrain.falling`).
- **Projectiles** (`physics/projectiles.ts`):
  - ballistic and straight shots, swept against terrain (raycast) and units (spatial `segment`) so fast shots
    never tunnel;
  - pierce, ricochet and explode hooks;
  - team masks.
- **Explosions** (`physics/explosions.ts`):
  - `terrain.carve`, damage and knockback in a radius (spatial `inRadius`), debris particles from the carve
    result's colours and points;
  - a renderer shockwave, a flash light, screen shake, and a `blast` event.
- **Damage pipeline.** `hurt(target, dmg, source)`, then death, a `unitKilled` event, the death effect and score.
  Teams: squad = 1, the classic enemy = 2, factions later.
- **Render extraction** (`render/`):
  - interpolated positions (`prev`→`pos` by alpha);
  - sprites pushed by layer;
  - falling terrain chunks drawn as sprites (copy the engine demo's `a2Ground.ts` approach);
  - lights from muzzle flashes, explosions and burning cells (`collectBurning`);
  - CPU particle simulation in typed arrays (gravity, drag, fade) feeding the ParticleBuffer, within the particle
    budget;
  - sky colours, ambient and parallax from a simple day/night clock;
  - the camera follows the squad (old `cameraTarget` in `PixelScene.tsx` ~577), with shake;
  - banner text through the text pass.
- **Art book** (`art/`):
  - at boot, paint the old sprites into the atlas: player loadouts (`makeLoadouts`/`playerSprite` in
    `components/pixel/art.ts`), troops (`makeTroops`), terrain block tiles (the old `makeAtlas` painter, as 8×8
    tiles per material and variant, filling `terrain.tiles`), props, a backdrop;
  - each art key gives frame indices, cached;
  - B1–B4 add more art through the same API.
- **Budgets and quality.**
  - Read `getQuality()` from `src/lib/device.ts` for the enemy, particle and cloud budgets (the Wallpaper shims
    it to a lighter one).
  - Respect `maxEnemies` and `maxParticles`.
  - `world.onError` is on for unattended runs; stats timing only with `?debug=1`.
- **Determinism.** Same seed, same simulation. Gameplay randomness on named streams; effects on `rng.get("fx")`.
  Note `shiftOrigin` for endless runs: the level is endless to the right, so implement the origin shift now
  (every 4,096 px), moving the terrain's `startCol` and the camera too.

## The vertical slice

It must run in `battle-dev.html?debug=1`, using `npm run dev`. You may start the dev server and drive Chrome with
Playwright (installed in Site 2, channel "chrome") to look at and measure it; stop the server when you're done.

- A placeholder level generator: hills, a cave or two, water, and a few brick and concrete buildings marked
  structural, with door frames down to the ground, so they stand until damaged and then collapse.
- **The 8 players**, with their real sprites and colours (`src/data/squad.ts` `PLAYER_COLORS`), move right in a
  formation over the terrain: jumping, climbing, swimming, parachuting in.
- Each shoots a rifle at the nearest enemy in range, with line of sight.
- **Placeholder enemy infantry** (old troop sprites) spawn ahead, take cover crudely, shoot back and die with
  debris.
- **Grenades** carve terrain; a few explosions make a building collapse.
- **Players** go down and respawn by parachute.
- **Camera, day/night** (a fast clock in dev), **lights** (night shows them), **banners** ("Level 1").
- **The `?debug=1` overlay** shows system timings.

## Tests (`src/test/battle-*.test.ts`) and benchmark

Headless, through `runHeadless`:
- the squad advances at least N px in 60 s on a fixed seed;
- kills happen;
- the same seed gives the same `hashWorld`;
- projectiles hit units and terrain without tunnelling;
- an explosion carves and damages;
- a building collapses after its ground floor is blasted, and the falling chunks crush a unit under them;
- locomotion never leaves a unit inside solid terrain across a long random run (a soak test);
- origin shift keeps everything consistent.

Add a benchmark (extend `scripts/bench-engine.mjs` or add `scripts/bench-battle.mjs`): 8 players plus 300
enemies plus 3,000 particles; report ms per tick.

## Also, read-only: the site contract

Read `src/components/pixel/PixelScene.tsx` and `src/lib/store.ts`, and list in your report every `sceneState`
and `ui` field and event the old scene reads or writes. Examples: `speed`, `paused`, `endless`, `zoom` and the
`scenezoom` event, `scaleK`, `maxFps`, `frames`, `ready`, `canvas`, `watching`, `stops`/`override`/layouts,
`pointer`/`burst`/`press`, `covered`, `white`, `cam`, `velocity`, `transition`, `intensity`. Also list what the
Wallpaper's `Wallpaper/source/src/main.tsx` uses. Phase D's `BattleScene.tsx` must keep that contract.

## Rules

- **Code standard:**
  - Prettier at 120 columns (`npx prettier --write`); ESLint clean, with no unused variables.
  - No `any`.
  - Short "why" doc comments; no narration or "(parenthetical asides)".
  - Files under about 800 lines; small focused modules.
- **Original art only.** Reuse the old engine's own painters; never copy external art.
- **Don't modify** `src/engine/` unless you find a real engine bug. If you do, fix it minimally, add a test, and
  report it. Don't modify `src/components/pixel/` or the site pages.
- **No git commands that change state.** No `npm run build`. The dev server is fine; stop it after use.
- **Checks**, from `Site 2`:
  - `npx tsc -p tsconfig.app.json --noEmit`
  - `npm run lint`
  - `npm run format:check`
  - `npx vitest run`: everything. The old 207 and the engine's 200 must still pass, plus yours.

## Report

Precise, because it becomes the brief for the next four agents:

- the architecture: folders, `BattleGame` API signatures, def types, components, events, the `install` pattern,
  system phases and order;
- how to add a unit, a weapon, a projectile, an effect, art and a level feature, each with a tiny example;
- the locomotion API and its tuning constants;
- the render extraction layers and how to add a draw;
- performance numbers;
- screenshots' file paths;
- the `sceneState` contract list;
- anything left for B1–B4.
