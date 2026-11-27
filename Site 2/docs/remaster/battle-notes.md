# The battle layer (`Site 2/src/battle/`): how it's built, from the B0 report

Read the code for details; this is the map. All checks passed at the end of B0 (450 tests).

## Structure and the `install` pattern

Each folder's `index.ts` exports `install(game: BattleGame)`. `game/installs.ts` lists them in this order:

```
art, render, physics, level, squad, enemies, world, flow
```

`flow` comes last, because it starts level 1. A new feature is a new folder plus one line in that list.

| Folder | Contents |
|---|---|
| `game/` | `BattleGame`, options (budgets, `FeatureOptions`), defs, content (registries), components, events, teams, core (AI dispatch, `machine()`, timers), origin, installs |
| `physics/` | locomotion, spawn, weapons, hitscan, projectiles, explosions, damage, terrain glue |
| `render/` | `FrameBuilder` and passes; camera, sky, particles, effects, lights, screen, banners, layers, poses; `beams.ts` (beams, lightning legs and muzzle crosses added with `beam`, `zap`, `muzzle`, aged by `fx.beams`, drawn as emissive pixels) |
| `art/` | `ArtBook` and `painters.ts`, which wraps the old painters read-only, plus `paintStrip` and `paintShape` for things drawn by hand (`registerStripArt`, `registerShapeArt`) |
| `level/` | `generator.ts`: `LevelSource` runs the old generator (a feature at a time, 16 columns ahead of the copy) and copies cells by true column, making everything stand up to support checks: door leaves in doorways, stone-walled buildings made one structure (found from their doorway), `BEAM` cells under `legs`/`support` props and motorway pillars, corner-only joints filled. Only a ruin's slab pieces between shell holes, which touch nothing, fall when disturbed. `LEVEL` holds it, with `spawns` (what the old generator placed), `ground(tc)` and `biome` (the theme at the camera; the `level.biome` system passes borders and calls `sky.show`, which fades backdrops). The level feature registers every theme's backdrop and uniforms. `props.ts` draws the old `Level.props`: the sprite ones as `asset.<id>` art, the hand-drawn kinds through `drawn.ts` (a draw function per kind in `DRAWN`, with a `Pen`: runs of `strip.<name>` art painted at every length 1–8 px for things of any size, `shape.<name>` art painted once from the old code, poses picked by time, emissive glows, lights capped at a quarter of the budget), then each prop's night glow through `glows.ts` (the old lighting pass: emissive runs and additive emissive shapes faded by tint, plus lights). `ambient.ts` is the `level.ambient` system: flames, smoke and waterfall mist from the props in view (effects `level.flame`, `level.smoke`, `level.mist`). `weather.ts`: each biome's `SkyLook` (`skyLook(theme)`: darkness and sky colours, fixed or the day cycle's, and its tint, given to `sky.show` and faded at a border), the `level.weather` system (the theme's weather particles, `weather.*` effects; the cycle themes' changing sky setting `sky.overcast`, `sky.fog`, `sky.wet`; storm lightning) and the `weather` pass (order 50, the bolts). `sky.night` is the old scale: 0 day, 0.7 at the cycle's midnight. `LevelSource.stands` says whether a leg or post still has its beams. `themes.ts` (the old themes: `THEMES`, `ROTATION`, `themeDeck`, `variant`, `WEATHER`, `lightOf`), with the data in `themes/` and block ids from `materials` (the engine keeps the old ids) |
| `squad/` | the squad, AI (placeholder until B2's), and the old 17 weapons: `arms.ts` (`ARSENAL` in the old order, `equip`; player n carries weapon n; the hand grenade special), `guns.ts` (the projectile guns), `rays.ts` (laser, tesla chain, prism split, and the held death ray, stepped by `squad.rays`) |
| `enemies/` | the infantry trooper with a state machine, the spawner and cull → B3 |
| `world/` | empty → B4 (cut from the lean plan except weather) |
| `flow/` | `startLevel`, `endLevel`, the clock → B2 |
| `dev/main.ts` | the entry for `battle-dev.html` (dev only). Query options: `?seed`, `scale`, `fps`, `day`, `time`, `renderer=canvas2d`, `quality`, `debug=1`. Keys: P pause, N half a day, F +10 s, Z zoom. A click is a blast. `window.battleDev` is exposed for scripts. |

## `BattleGame`

**Construction.** `new BattleGame({ seed, viewW = 480, viewH = 270, budgets, debug, unattended, dayLength, timeOfDay, features, feature })`.

**Budgets.** `{ maxEnemies, maxParticles, clouds, maxProjectiles, maxLights }`, from `budgetsFromQuality(getQuality())`.

**Fields:**
- `world`, `core`, `c` (the game components), `terrain`, `content` (registries, frozen after install, with checked
  references), `links`
- `rng { ai, spawn, combat, squad, level, fx }`: cosmetic randomness uses `fx` only
- `stats { kills, booms, shots, enemies, projectiles }`
- `level { no, name, t, generate }`
- `camera`, `sky`, `particles`, `flashes`, `screen`, `banners`, `popups`, `origin`, `timers`, `art`, `frames`

**Lifecycle.** `step(realDt)`, `buildFrame(frame, alpha)`, `resize(w, h, zoom)`, `paintArt(atlas)`, `dispose()`,
`onBoot(fn)`, `now`.

**Facade:**
- `spawn(def, x, feetY, { team, facing, chute, vx, vy, force, hp })` returns an id, or 0 at the budget
- `hurt(id, dmg, source)`, `kill`
- `explode(x, y, BlastSpec, source, team)`
- `shoot(projectile, x, y, vx, vy, { owner, team, damage })`, `fire(shooter, aimX, aimY)`
- `effect(id, x, y, opts)`, `flash(x, y, r, rgb, intensity, life)`, `shake(px)`
- `banner(text, sub, rgba, { hold, priority })`, `popup(text, x, y, rgba)`, `at(delay, fn)`

## Content defs (`game/defs.ts`)

- **`UnitDef`:**
  - identity and size: `name`, `hw`, `h`, `hp`, `team`, `score`
  - looks: `art`, `pose(game, i)`, `draw(game, i, x, y, sink)`, `layer`
  - behaviour: `weapon`, `ai { think: fn | machine(...), every }`, `move` (`MoveSpec`), `tags` (INFANTRY, ARMOURED,
    FLYING, SHIP, BOSS, STRUCTURE)
  - damage: `armour`, `blastKills`, `blastTaken`, `deathFx`, `corpse`
  - hooks: `onHurt`, `onSpawn`, `onDeath`
- **`WeaponDef`:** `range`, `cadence`, `burst`, `burstPause`, `pellets`, `spread`, `projectile`, `speed`, `lob`,
  `lead`, muzzle and `flash`, `damage`, and `fire(game, shooter, gx, gy, ax, ay)` for beams and chains (use
  `hitscan`).
- **`ProjectileDef`:** `speed`, `life`, `damage`, `gravity`, `accel`, `drag`, `radius`, `pierce`, `chip`,
  `bounce`/`bounces`, `fuse`, `explode` (BlastSpec), `trail`, `look` (pixel streak colours) or `art`, `glow`; hooks
  `update`, `onHit`, `onExpire`.
- **`BlastSpec`:** `r`, `power`, `dig`, `damage`, `knock`, `friendly`, `fx`, `ignite`.
- **`EffectDef`:** particle bursts (`count`, `speed`, `angle`, `spread`, `life`, `gravity`, `colors`, `ramp`,
  `flags`, `light`).
- **`ArtDef`:** `paint: () => ArtFrame[]`, `facings` 1 or 2, `eager`.
- **Registering:** `game.content.units.register("regime.dog", {...})`, then `game.spawn("regime.dog", x, y)`. The
  same goes for weapons, projectiles, effects and art. The worked AI example is `enemies/infantry.ts`.

## Components and events

**Components** (typed columns):

| Component | Fields |
|---|---|
| `unit` | `flags`, `facing`, `art`, `anim`, `hitT`, `deadT`, `spin`, `killer` |
| `motor` | `mode` (walk, climb, swim, chute, fly, knock, tumble), `abil`, `out`, plus tuning |
| `intent` | `move`, `up`, `jump`, `fire`, `target`, `aimX`, `aimY` |
| `weapon` | `def`, `cd`, `shots`, `alt` |
| `player` | `id` 0–7, `score`, `kills`, `inv`, `downT`, `queued` |
| `ai` | `state`, `t`, `scanT`, `cover`, `home`, `alert` |
| `projectile` | `def`, `owner`, `pid`, `pierce`, `bounces`, `dmg`, `life`, `last`, `trail` |

- AI writes `intent`; locomotion and weapons read it.
- Rare state goes in `world.defineObject<T>`.
- Per-feature options go in `FeatureOptions`, by declaration merging.

**Events:**
- pooled: `unitSpawned`, `unitHurt`, `unitKilled {id, def, team, killer, player, x, y, score}`, `weaponFired`,
  `blast {x, y, r, source, team, cells}`
- plain: `playerDown`, `playerRespawned`, `banner`, `levelStart`, `levelEnd`
- dispatched immediately: `originShift {dx, cols}`

**System order:**

| Phase | Systems |
|---|---|
| director | flow.clock, squad.advance |
| spawn | squad.respawn, enemies.spawn |
| ai | ai.think |
| weapons | weapons.fire, squad.rays |
| motion | locomotion.move |
| physics | spatial |
| projectiles | projectiles |
| damage | hazards |
| death | life, clocks |
| effects | particles, flashes, beams, screen, fire, level.ambient, level.weather |
| terrain | ensure, step, crush, settle, events |
| camera | camera |
| late | sky, text, cull, timers, origin |

## Locomotion

- `moveUnit`, `moveBox` (sub-stepped, 4 px), `pushOut`, `knock`, `initMotor`. A settle pass keeps units out of solid
  ground.
- Tuning: gravity 620; default speed 56, jump 200, climb 70; parachute gravity ×0.25; troopers speed 34, jump 230.

## Render

- **Passes, in order:**

  | Order | Pass |
  |---|---|
  | 0 | sky and backdrop |
  | 50 | lightning bolts (`level/weather.ts`) |
  | 100 | falling chunks |
  | 150 | the level's props (`level/props.ts`) |
  | 200 | units |
  | 300 | projectiles |
  | 350 | beams, lightning, muzzle crosses (`render/beams.ts`) |
  | 400 | particles |
  | 500 | lights |
  | 600 | screen effects |
  | 700 | text |

- **Sprite layers** (`LAYER`):

  | Layer | Use |
  |---|---|
  | 0 | sky |
  | 1 | distant |
  | 2 | back |
  | 3 | behind terrain |
  | 4 | chunks |
  | 5 | props |
  | 6 | vehicles |
  | 7 | enemies |
  | 8 | players |
  | 9 | projectiles |
  | 10 | fx |
  | 12 | front |
  | 13 | markers |

- Add a draw with `game.frames.add({ name, order, draw(game, frame, view) })`. Passes only read state.

## Performance (B0)

| Load | Result |
|---|---|
| 8 players + 300 troopers + 3,000 particles (`node scripts/bench-battle.mjs`) | about 0.16–0.18 ms per tick |
| Chrome | 120 fps; render CPU 0.1–0.4 ms; 13–18 draw calls |

## Left for the next phases (from B0)

- **B1:**
  - done (the props, their glows, flames and mist, weather and day/night); not ported: the cycle themes' sun and
    moon;
  - enemies still wear the jungle uniform: set `unit.art` from the biome's `look` at spawn (B3);
  - known visual gaps: no depth-shaded sea; trees float when the ground goes; the renderer darkens cells under trees
    and roofs (an engine follow-up: a ground row, `crust` on `TerrainView`).
- **B2:**
  - the full squad AI (jetpack, crates, barrels), weapon sprites (`unit.art`), the respawn transport jet (the 17
    weapons are done);
  - tune lethality (players went down about 40 times in 2 minutes).
- **B3:** the Regime roster as `UnitDef`s, the view-based cap, allies.
- **Origin shift:**
  - it happens every 4,096 px at a camera x of 8,192;
  - the game moves the terrain's `startCol` and `endCol` directly, waiting while structures settle;
  - a `Terrain.shiftColumns()` in the engine would be cleaner later.

## The site contract (for phase D's `BattleScene.tsx`)

**`sceneState` fields the old scene reads:**
- `override` (it starts a new level; "lost" means the cave)
- `stops` (for intensity), `pointer`, `burst` (a strike), `press` (a barrage)
- `transition`, `watching` (full frame rate), `covered` (skip frames), `pulse`, `highlight`
- `paused`, `speed`, `maxFps`, `scaleK`, `zoom`, `endless`, `velocity`

**Fields it writes:**
- `canvas` and `ready` (plus the `no-canvas` class on `<html>`), `zoom` (clamped), `cam {x, y, z}`
- `kills`, `booms`, `level`, `stage`, `white`, `intel.*`, `players[i] {score, hp, alive}`
- `velocity`, `frames`

**Other inputs and events:**
- It dispatches the window event `scenezoom {pct, canIn, canOut}`.
- After 8 s of no input it drops to about 30 fps, unless `press`, `transition` or `watching` is set.
- It reads `?level` and `?finale`, `prefersReducedMotion()` (which freezes the fight and skips the white-out),
  `getQuality()` and `PLAYER_COLORS`.

**The Wallpaper** uses:
- the `PixelScene` default export, and `canvasOptions` from `pixel/art`;
- `sceneState.speed`, `scaleK`, `maxFps`, `watching`, `stops`, `pointer`, `burst`, `press`;
- it reads `ready`, `cam.x`, `frames`, `speed`, `scaleK` and `maxFps`;
- the `@/lib/smoothScroll` and `@/lib/device` aliases.
