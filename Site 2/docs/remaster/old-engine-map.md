# Map of the old engine (src/components/pixel/)

A read-only survey from before the remaster, with line numbers as of 2026-11-27. Use it to find what to port.

(From a read-only survey; line numbers are from before your edits.)

Files, all in `Site 2/src/components/pixel/` unless noted:

| File | Lines | Contents |
|---|---|---|
| game.ts | 7,622 | The simulation: a `Game` class |
| level.ts | 3,240 | Level generation |
| themes.ts | 2,015 | Themes |
| art.ts | 4,473 | Sprites |
| PixelScene.tsx | 3,008 | Renderer and game loop |

Also relevant: rng.ts, assets.ts, buildings.ts, catalog.ts, `src/data/squad.ts`, `src/lib/store.ts`, `src/test/*`.

**The level loop.** `Level.every` defaults to 20 (level.ts:424,460). A stronghold and finale only end every 20th
stretch (level.ts:1567), or when `stronghold: true` forces one. Every other stretch ends at a river border into the
next biome: a seamless new "level" with no white-out (game.ts:1497-1509, level.ts:1571-1597).

### 1. Game (game.ts:1080)

**Construction.** `constructor(theme, opts: GameOptions)` at 1231. `GameOptions` (814-828):
- `seed`
- `maxEnemies` (default 70) and `maxParticles` (default 2600)
- `colors`
- `level: LevelOptions`
- `gunship` (false sets `noShip`) and `waves` (false sets `noWaves`)
- `finale`

The constructor:
- seeds `rand = rng(seed*7919+3)` (1232);
- builds the level with `createLevel(theme, seed, opts.level)` (1251);
- sets up the theme deck, with `level.pickTheme = variant(deck(cur.key), rand)` (1252-1253);
- creates 8 players (1254-1294);
- calls `meetBoss()` and shows the "Level 1" banner (1295-1296).

**Fields (1081-1229):**
- World: `level`, `dirty` and `fresh` (renderer redraw queues), `reset`.
- Squad: `players`, `advance` (starts at 260), `crew`/`ship`/`shipMode`, `ride`, `dropQueue`, `holdT`/`holdAt`,
  `platform`.
- Enemies: `enemies`, `barrels`, `hatches`.
- Allies: `allies`, `navy`, `sams`.
- Projectiles: `shots`, `missiles`, `jets`, `drops` (supply crates).
- Effects: `parts`, `blasts`, `popups`, `mushrooms`, `plumes`, `waves` (shock rings), `fires`, `flashes`,
  `beams`, `zaps`, `bolts` (lightning), `flash`, `shake`.
- Infrastructure: `roads`, `freights`, `metros`, `carts`.
- Sky: `clock`, `sky`, `skyT`, `weather`.
- Flow: `phase`, `phaseT`, `banner`, `white`/`whiteUp`, `complete`, `log`, `cleared`, `bossKind`/`bossState`,
  `levelNo`, `levelT`, `finale`/`lastFinale`/`forceFinale`, `pending` (finale warheads not yet landed).
- Pressure: `wave`, `waveT`, `surge`, `usual`, `intensity` (set by PixelScene), and the `timers` object
  (1202-1216).
- Other: `t`, `kills`, `booms`, `rand`, `timeline`, `lastView`.

**Helpers:**
- `at(delay, fn)` (1380) and `say(text, sub, color, info?, delay)` (1384; one banner slot, which it overwrites).
- `troop()` (1372) picks from `theme.troops`; `cap()` (1353) limits enemies by view width × `WAVES[wave].k`.
- Getters: `progress` 1318, `objective` 1325 (tests match its strings), `night` 1358, `carrier` 1364.

**`step(dt, view, frozen)` (1482-1574), in order:**
1. Timers, flash decay, white fade, banner and bolt ageing.
2. `lastView`, and `L.intensity = intensity × CHAOS(3) × (1 + 0.1·min(levelNo-1, 6)) × WAVES[wave].k`.
3. Biome-border crossings (1497-1509).
4. `extend()` and `spawn()` (1511-1514).
5. Timeline events (1517-1524).
6. If not frozen: `flow`, `updateWave`, `moveSquad`, `updatePlayer`×8, `updateEnemy`×N, `manageAllies`,
   `updateAlly`×N, `manageShip`, `manageNavy`, `updateSub`, `updateCarts`, `updateFreights`, `updateMetros`,
   `updateRoads`, `updateSky`, `updateHatches`, `ambient`.
7. Always: `updateBarrel`, `settle`, `updateShots`, `updateMissiles`, `updateSams`, `updateEffects`.
8. Off-left culling (1563-1573).

### 2. Players

**Type.** `Player` (112-161). Constants: `PLAYER_HP` = 6, `SQUAD_SIZE` = 8, formation `OFFSETS` (61), `PH` 15 and
`PW` 3, `RUN` 56, `JUMP` 200, `CLIMB` 70, `GRAV` 620. Players are created at 1254-1294 with `weapon = id`; colours
come from `opts.colors` (PixelScene passes `PLAYER_COLORS` from `src/data/squad.ts`).

**Targeting.** `findTarget` (3176): range `W_RANGE[weapon]`, on screen only, needs line of sight unless the weapon
is lobbed. `aimPoint` (3208) gives the point to aim at.

**Movement and supplies.** In `updatePlayer` (2954-3173): formation x, jumps, climbs, jetpack, swimming,
parachute, crate pickup (3101-3118), and stuck handling (3145-3172).

**Firing:**
- When `fireCd <= 0`, `fire(p, tgt)` runs (3024).
- The special is a hard-coded grenade at 3025-3029: if `specialCd <= 0` and a target is within 130 px, throw a
  grenade and set `specialCd` to 6–12 s.
- `fire()` (3218-3422) is a `switch (p.weapon)` with 17 weapons. The default case is the mortar (13).
- `WEAPONS` (63-81), `W_RANGE` (83) and `art.GUNS` (art.ts:1784) are index-parallel. Tests pin `WEAPONS` at 17,
  so don't add weapons there: build signature weapons as ability-owned fire paths using `shots`, `beams`, `zaps`
  and `missiles`.

**Supply crates.** Every 5–11 s; 20% hold a jetpack, the rest a random weapon (7051-7061). A pickup lasts 16 s.

**Death and respawn:**
- `hurtPlayer` (3545) leads to `dead` with `deadT` 2.6 and a "P{n} DOWN" popup.
- `updatePlayer` then calls `respawn` (3560): `queued` and `dropQueue`.
- An allied transport jet (`ambient`, 6721-6736) releases them with `release()` (3570: parachute).
- There are no lives and no fail state.

**Scoring.** `killEnemy` (5813) adds `kills++` and `SCORE[kind]` to the killer. Shot `pid` values (540): 0–7
players, -1 enemies, -2 visitor, -3 artillery, -4 missiles, -5 allies.

### 3. Enemies

**Type and tables:**
- `Enemy` (420-464) covers bosses too. `EnemyKind` is a string union of about 46 kinds plus `BossKind`
  (310-357). `Troop` is in themes.ts:91-109, `Spawn.kind` in level.ts:163-209.
- Category sets: `FLYING` 370, `SHIPS` 381, `ARMOURED` 383, `ARMOR_KINDS` 209 (exported; the renderer uses it),
  `INFANTRY` 1064.
- Tables `SCORE` 830, `HP` 885 and `SIZE` 941 are `Record<EnemyKind, …>`. `dims()` (1057) applies `scale`.
- `makeEnemy` (2409) multiplies hp by 1.6. `vary` (2399) handles infantry size variants and a 12% sapper chance.

**AI.** `updateEnemy` (3660):
- death handling (3663-3690) and retreat (3691-3702);
- a `switch (e.kind)` (3703-3759) into per-kind update methods;
- the generic infantry AI for everything else (3760-3878).

`enemyFire` (3927-4122) is a switch plus an if-chain. `enemyShot` (3882) leads moving targets.

**Spawning:**
- level.ts `enemies()` (1202-1208) places `t.enemies × L.intensity × width/12` troops from `t.troops`.
- Set pieces spawn their own units (see level.ts). `Game.spawn()` (1621-1699) applies the `cap()` check.
- Waves (`WAVES`, 88-93: Lull 0.5, Skirmish 1, Assault 1.8, Onslaught 3): `updateWave` (1943-2032) re-rolls every
  14–36 s and sends reinforcements (2026).
- Ambient spawns (6630-7227): para transports (6676; drop pool 6770), helicopters (6921-6936), armour and ships
  (6938-6975), jetpacks (7041), drones (7063).
- Other sources: hatches (5290), truck unloads (5478), airship paratroopers (5360).

**Damage:**
- `updateShots` (5999) moves shots and calls `hitBodies` (6093). Enemy shots (pid -1) hit players and allies;
  other shots hit enemies through `hits()` (6085). Riot shields are at 6129.
- `explode()` (6218-6493) is public.
- `hurtEnemy` (5896) and `killEnemy` (5813) end at the per-kind death chain (5844-5892).

### 4. Bosses

- `BossKind` is in themes.ts:110; game.ts has `BOSS_KINDS`, `BOSS_NAMES` and `isBoss` (358-368).
- Choice: level.ts:433-438 (80% from `theme.bosses`, placed at `length × 0.48`) and level.ts:1585 at borders (70%).
- The arena queue: level.ts:1551-1563.
- `flow()` (1580-1586) spawns the boss through `spawnBoss` (2451), sets `holdAt`, and sets phase "boss". It ends
  when the boss is dead or after 45 s (1587-1591).
- On death, `killEnemy` runs a boss chain (5828-5842).
- The renderer has a per-kind branch (PixelScene.tsx:1453-1525) and boss bars with per-kind offsets (2603-2628).

### 5. Flow

**Phases.** `Phase = "play" | "boss" | "nuke" | "clear"` (812).

**`flow()`** (1577-1598) covers the boss beat, then the finale: `startNuke()` when `fortPlaced` and the squad is
past `targets[0] - 200`.

**`nextLevel(theme?, opts?)`** (1393-1475):
- `levelNo++`, `variant(deck())`, `createLevel`;
- clears every entity array;
- `advance` = 260, phase "play", `white` = 1, `reset` = true;
- the squad parachutes in and its weapons reset.

**Finale:**
- `startNuke` (2470-2592) uses `FINALES` (757) and `FINALE_TEXT` (759); `splitMirv` is at 2595.
- `detonate` (2821) wipes the area. Once `pending` reaches 0, `blowUpEverything` (2850-2884) runs:
  `whiteUp`, phase "clear", the `complete` report (titles at 767, `LevelReport` at 788-799), the `log` push,
  then `nextLevel()` at 5.2 s.

**Themes.** `themeDeck` (themes.ts:1911), `variant` (1940), `nextTheme` (1987); `startTheme` is at game.ts:7620.

### 6. Level generation (level.ts)

- `TILE` 8, `ROWS` 240, `RING` 512 (a column ring buffer), `BASE_ROW` 148.
- 92 block kinds; `SOLID` 129, `TILE_HP` 147 (255 is unbreakable), `LOOSE` 158.
- Helpers: `cell` 415, `kindAt` 473, `solidAt` 480, `surfaceY` 497, `lineOfSight` 503, `blastCells` 513,
  `damageCell` 529.
- `Plan` (359-369) and `Level` (372-412). `createLevel` (430) uses its own `rng(seed)`.
- `extend()` (1183) calls `feature()` (1193-3168) in priority order:
  1. Open sea (1520).
  2. Landing zone (1542).
  3. Boss arena (1551).
  4. Stronghold or biome border (1564-1597).
  5. Filler (1599).
  6. Hills (1611).
  7. Military sites (1627-1705).
  8. `switch (pick(t.features))` (1708-3166).
- `building()` (787) records no extents. `stronghold()` (3171-3239) sets `plan.targets`.

**Destruction (game.ts):**
- `explode` digs via `blastCells` and `damageCell` (6224-6245).
- `loosen` 7476 leads to `cutOff` 7488 (pieces of ≤ 90 cells start crumbling), then `settle` 7539 (≤ 260 moves
  per step).
- The renderer is fed through `dirty` and `fresh`.

### 7. Vehicles and set pieces

- **Gunship ferry:** `manageShip` 5074, `flyShip` 5122, `board` 5240, `unload` 5256, `shipDown` 5280, door
  guns 4869.
- **Allies:** `manageAllies` 4739, `makeAlly` 4767, `updateAlly` 4793.
- **Trains:**
  - freight: `addLine` 1719, `updateFreights` 1748;
  - armoured: `updateTrain` 4613;
  - metro: `addMetro` 1787, `updateMetros` 1811, `boardMetro` 1900.
- **Traffic, carts and ships:**
  - roads: `updateRoads` 2677;
  - carts: `updateCarts` 2058;
  - ships: `sail` 2389;
  - allied subs: `manageNavy` 2182;
  - SAMs: `updateSams` 2293.
- **Air and strikes:**
  - jets: 224-268, in `ambient` 6694-6920;
  - artillery 6648, meteors 6662, allied strikes 6977, enemy strikes 7007, silos 7092;
  - missiles: 635, `updateMissiles` 6532.
- **Weather:** lightning 7086, `updateSky` 2788, weather particles 7098-7202.

### 8. Renderer (PixelScene.tsx)

**Setup:**
- The effect starts at 352.
- Caches (373-380):
  - `makeAtlas`;
  - `makeLoadouts(PLAYER_COLORS)` (art.ts:1893, keyed `id*32+weapon`);
  - `makeProps`, `makeAssets`;
  - `makeTroops(look)` (art.ts:2040; `LOOKS` 1911);
  - `makeBackdrop`.
- Terrain lives in a ring canvas (`drawCell` 394).

**Loop and step:**
- `loop` (2953): dt is clamped to 0.05 and scaled by `speed × PACE` (1.25), in substeps of ≤ 34 ms calling `step()`.
- `start` (640) creates the `Game`, then `warm` (622).
- `step` (2804-2916): page override, intensity (616), visitor strike, `g.step`, reset, theme crossfade, camera
  ease (2860-2869), shake (2870-2872), redraws, and the `sceneState` writes (2883-2914). No UI component reads
  `intel` now.

**Camera.** `cameraTarget` (577-607). Zoom is set only at resize (464-500).

**`draw()` (2918):** shake, `drawSky` (2720, bolts 2695), `drawWorld` (967), fog and rain, then `drawLights`
(2069).
- `drawWorld`, in order:
  - parallax, jets 1022, missiles 1032/1048, worm 1099, blocks and props 1108-1344
  - barrels 1346, drops 1366, freights 1387, roads 1410, metros 1430
  - **enemies** 1444-1595 (an if-chain per kind; fallback `troopSprite` 908-956)
  - allies 1597, carts 1622, hatches 1639, navy 1653, sams 1671
  - **players** 1675-1717
  - sea 1719, particles 1779, tint 1821
- `drawLights`, in order:
  - fires 2222, sniper sights 2230, beams 2248, zaps 2272, flashes 2292
  - **shots** 2303-2428 (a switch with no default)
  - missiles 2429, sams 2436, blasts 2451, fireballs 2487, mushrooms 2488, shock rings 2489
  - lit particles 2494-2560
  - helicopter 2564, player markers 2573, popups 2596, boss bars 2604, banners 2631-2646
  - white-out 2649-2656 and `drawComplete` 2660-2692
- Text uses `pixelText` (art.ts:2881).

### 9. What the tests rely on

**Construction.** `new Game(theme, { seed, waves: false, gunship: false, level: { boss, length, stronghold,
every }, finale? })`. Quiet themes are built by spreading `THEMES.jungle` with the spawns zeroed.

**Methods:**
- `step`, `nextLevel`, `hurtPlayer`, `hurtAlly`, `explode`, `strike`, `solid`, `aimPoint`.
- Private ones reached by casts: `makeEnemy`, `fire` (every weapon 0–16 must add a shot, beam or zap), `release`,
  `buried`.

**Fields and arrays.** The tests read many fields and push into the arrays directly. This covers players, enemies,
barrels, shots, drops, missiles, jets, beams, zaps, popups, mushrooms, navy, sams, allies, ship and more, plus
`advance`, `t`, `kills`, `booms`, `levelNo`, `phase`, `banner`, `white`, `complete`, `log`, `cleared`, `bossKind`,
`bossState`, `wave`, `night`, `sky`, `progress`, `objective`, `theme`, `level`, `endless`.

**Exports.** `WEAPONS` (17; the laser names exact), `PLAYER_HP`, `SQUAD_SIZE`, `WAVES`, and the shot constants.

**Exact strings.** "Level 1", "Jungle landing", "Boss ahead: Siege tank", "Destroy the Siege tank",
/Siege tank ahead/, "Level complete", "P2 DOWN", "P3 TESLA", "SPLASH ONE".

**Seed-tuned outcomes (any extra `this.rand()` call shifts these):**
- game.test.ts:287 (seed 2 tank), 325 (seed 11 endings), 309 and jungle 408;
- the city waves (seeds 1–2), the gunship tests (seeds 1–3), the monsoon lightning, and the truck unloading 5 troops.

**Other constraints:**
- `frozen` stops all movement (game.test.ts:506).
- squad.ts must keep 8 unique weapons, with P1–P4 in domain order.

### 10. Recommended hook points

- **Units:**
  - a def check before the switch at 3703;
  - `SCORE`/`HP`/`SIZE` lookups through functions;
  - category sets from the registry;
  - `onDeath` at 5844;
  - the renderer def draw at 1448;
  - boss registry touchpoints at themes.ts:110, game.ts:358-359, 2452-2462, the dispatch, the renderer branch,
    the bars at 2607-2614 and the death chain at 5828.
- **Faction rosters:**
  - `theme.troops` dealing at 1253 and 1395/1398;
  - `Game.troop()` 1372 and level.ts `troop` 1201;
  - spawn pools at 2026, 5295, 5363, 5491, 6770, 6929, 6947-6964, and `PLANE_MIX` 255.
- **Abilities:**
  - replace 3025-3029;
  - a squad coordinator after the player loop (1530);
  - per-player state in a parallel array.
- **Missions:**
  - a hook at the top of `flow()` (1577);
  - `progress` and `objective` delegate;
  - extract `endLevel` from `blowUpEverything` (2857-2883);
  - `LevelReport` (788) and `drawComplete` (PixelScene.tsx:2660);
  - `holdAt` for defend missions, an advance-follow for escorts;
  - level sites next to level.ts:1551, mission data in `Plan` (359);
  - fronts via `L.pickTheme` and `borders`;
  - gate the gunship (5082-5088, 5195-5206).
- **Effects:**
  - `explode`, missiles, mushrooms, shock rings, plumes, beams, `flash`, `white`, `shake` and `at` are usable as
    they are;
  - shots without a renderer default need a registry;
  - collapses need building rectangles plus a `crumbling` seed, and lifted limits (7501, 7556);
  - the camera needs a hint read in `cameraTarget` (577) and a draw-time zoom transform.
- **Hard parts:**
  - giant switches and if-chains (3703, 3927, 3238, 5844, renderer 1444, 908, level.ts `feature` 1708);
  - closed unions (`EnemyKind`, `Troop`, `Spawn.kind`, `Feature`, `BossKind`, `Phase`);
  - index-parallel weapon arrays;
  - the single banner slot;
  - no lives or fail state;
  - private helpers reached through casts in tests.

