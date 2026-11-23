# Alexi Kehayias — Portfolio (Site 2, pixel edition)

A copy of the portfolio with the same content (projects, domains, toolbox, CV) and a
completely different look: a pixel-art co-op action game. Eight players fight their way
through random levels behind the page, styled after classic 2D run-and-gun games.

This copy **runs locally only**. It has no GitHub Actions, no GitHub Pages deploy and no
Docker setup, and it is served from the root path (`/`), not `/alexikehayiasportoflio/`.

## Run it

```bash
npm install        # first time only
npm run dev        # http://localhost:8081 (opens the browser)
```

To try the production build locally:

```bash
npm run build
npm run preview    # http://localhost:4174
```

The main site's dev server uses port 8080, so both can run side by side.

## What's different from the main site

The background is an eight-player run-and-gun. Eight AI-controlled players fight their way
through a side-scrolling level behind the page, each in their own colour. P1 to P4 are the
four domains, P5 to P7 are project specialisations, and P8 is the LLM and agents engineer:

| Player | Role | Weapon |
| --- | --- | --- |
| P1 | Data Engineer (Data Engineering) | Rifle |
| P2 | Data Analyst (Data Analytics) | Shotgun |
| P3 | Data Scientist (Data Science & AI) | Plasma |
| P4 | ML Engineer (MLOps & Deployment) | Rockets |
| P5 | CV Engineer (Computer Vision projects) | Sniper |
| P6 | NLP Engineer (Natural Language Processing projects) | Flamethrower |
| P7 | Researcher (Research Methods projects) | Grenade launcher |
| P8 | LLM Engineer (LLMs and agents) | Minigun |

- **Every level is random**, and the next one never repeats the last. There are 17 themes:
  jungle landing, deep jungle, lost temple, monsoon river, night raid, river village,
  black swamp, desert outpost, red canyon, frozen peaks, fortress, refinery, city ruins,
  inferno, alien hive, missile silos and HQ assault. Each has its own sky, backdrop,
  terrain, enemies, props and weather. The 404 page is the caves ("Mission failed").
- **Jungles:** seven of the themes are jungle of some kind. They grow rainforest giants with
  hanging vines and lookout platforms, bamboo thickets, waist-deep rivers to wade (or a
  rope bridge across, or a patrol boat on them), waterfalls pouring off cliffs, stepped
  temples of mossy stone with braziers and a shrine on top, overgrown ruins, enemy camps
  with tents round a campfire, and dirt roads. Leaves and vines hang into the top of the
  screen. The monsoon brings a storm with lightning.
- **How a level plays:** the squad drops in by helicopter and fights through it. Most levels
  have a boss halfway through: a siege tank, a walker mech, a gunship or a burrowing worm.
  At the end is an enemy stronghold with walls, a gate, a keep, turrets and radar. ICBMs
  fly in and nuke it, the whole scene goes up, the screen fades to white, "LEVEL COMPLETE"
  comes up with the level's numbers (enemies down, time, the squad bonus), and the next
  random level fades in with a banner saying what's ahead.
- **The further down the page you are, the more intense it gets** (more enemies, more air
  traffic), and it escalates with every level cleared. Opening a project page starts a
  new level.
- **Everything is live:** riflemen, heavies, rocket troops, snipers with laser sights,
  attack dogs, riot-shield troopers (bullets from the front glance off), flamethrowers,
  grenadiers, mortars, machine-gun nests, turrets, jeeps, troop trucks that unload
  reinforcements, river patrol boats, drone swarms, helicopters, crawlers, paratroopers,
  bombing jets, artillery, meteors, silo launches, lava, fuel tankers, barrels and ammo
  crates that chain-react, alien eggs that hatch, and blocks that break. Players who go
  down are flown back in by a transport plane and parachuted out over the squad; enemy
  reinforcements arrive the same way. Armoured cars, APCs, self-propelled guns and
  hovercraft fight for the enemy, scout cars for the allies; the sky fortress and some
  bombers drop huge bombs that level whole blocks.
- **A battle in the distance:** behind the hills things keep going off: explosions on the
  horizon, flak, anti-aircraft tracers, smoke columns from fires far away, planes (some
  hit and trailing smoke), flocks of birds, and searchlights sweeping the sky at night.
- **Spectate:** below the contact section is open ground, nothing over it, to leave the battle
  running in. The scene keeps its full frame rate there, the screen is kept awake (where the
  browser supports it), and the scene controls appear: pause, resume, speed from ½× to 8×, and
  zoom. "Watch" in the menu goes straight there.
- **You can join in:** click anywhere off the buttons to call in a blast; hold on the
  intro for an airstrike; hover a domain or specialisation tab to point out its player.
- **Mountainous, endless terrain:** every level's land comes from seeded fractal noise, like a
  voxel game's world generator: lowlands, rolling hills and whole mountain ranges with rocky
  summits, rock strata and winding caves underground, different on every level and visit.
  Behind it, shaded mountain ranges (lit faces, shadowed faces, snow caps, haze) in five
  parallax layers.
- **Props everywhere:** 44 hand-painted props dressing each kind of land (crates, oil drums,
  sandbags, radios, generators, camo nets, POW cages, tiki torches, stone idols, wells, cactus,
  fuel pumps, cones, hydrants, phone booths, wrecks, tank traps and more), big ones standing in
  shadow behind the ground (water towers, radio masts, billboards, crashed helicopters), and
  ladders, chain-link fences and tin roofs on the towers and buildings. Blasts smash them;
  jerrycans, fuel pumps and toxic drums go off; lit propane tanks rocket up before they burst.
- **Towns and cities:** 120 kinds of building, each its own: offices, banks, a stock exchange,
  diners, pizzerias, sushi and noodle bars, pubs, clubs, supermarkets, pharmacies, record
  stores, barbers, laundromats, gas stations, hospitals, dentists, police and fire stations,
  libraries, museums, schools, cinemas, arcades, casinos, gyms, hotels, motels, apartments,
  farmhouses and barns, factories, breweries, power stations, train stations, airports,
  lighthouses, barracks and more. They're cut away so every floor shows, with its own walls,
  windows, roof (domes, spires, sawtooth skylights, rooftop gardens, helipads, water tanks,
  neon), sign and awning, and the right furniture on every floor (desks, computers and water
  coolers in an office; ovens, tables and menu boards in a pizzeria; beds, drips and heart
  monitors in a hospital). Streets of asphalt and paving between them, with traffic lights, bus
  stops, benches, hydrants, parked cars, taxis, police cars, vans and buses.
- **Under the ground:** coal, iron and gold veins in the rock, and buried rooms to blast into:
  mines with pit props, rails and carts; crypts with sarcophagi and treasure; bunkers; sewers;
  subway platforms with a train; caverns with a pool; secret labs; vaults.
- **Water:** rivers and waterfalls in most lands, and harbours: deep water under a long pier,
  docks with containers and cranes, and an enemy battleship riding in the basin, shelling the
  squad from its turrets until it breaks up and sinks.
- **306 props in all**, 16 new kinds of block (glass, marble, plaster, siding, stucco, tiles,
  dark brick, parquet, carpet, asphalt, paving, coal, iron, gold, rail, sewer brick), gas
  canisters and oxygen cylinders that rocket off when they're hit.
- **Mayhem:** half as many enemies again, helicopters three at a time, armoured personnel
  carriers that unload troops and light tanks rolling in, jetpack troopers swooping over, and
  landmines underfoot. Eight more kinds of trooper (jetpackers, minigunners, officers who call
  artillery, dynamiters, machete rushers, mine-laying engineers, laser troopers, exosuits) and
  three more bosses (a sky fortress, a doom cannon, a spider tank).
- **Full-scale war:** an allied column fights alongside the squad: main battle tanks, infantry
  fighting vehicles, rocket artillery, flak tanks, attack helicopters and a supply lifter (blue-
  grey with white stars and a blue marker). Against them: carriers, light, medium and heavy
  (twin-gun) tanks, flame tanks, rocket tanks, flak tanks and missile launchers; gunships,
  attack helicopters, twin-rotor transports dropping paratroopers, scouts. Bombers, fighters
  and ground-attack jets fly for both sides. Munitions of every kind: shells, flame, rockets,
  flak, napalm, carpet bombs, and missiles from small rockets through tactical and cruise
  missiles to enormous warheads that blow craters in the land and wreck the vehicles. Bunker
  lines of sunken pillboxes, missile launch pads, and manholes that enemies climb out of.
- **Zoom out** (the − and + by the speed buttons, or - and + on the keyboard) for a wider view
  of the battlefield in smaller pixels: more of the level, more allies, more enemies at once.
- **A competent enemy, and twice as much of it:** troopers lead their shots, close to their
  best range and hold it, jump obstacles, fall back when badly hurt, lob grenades at bunched-up
  squad members, and fire faster with more health. They come small and quick, large, or as
  giant brutes, and some carry explosive packs that go off when they die. Dug-in gun
  emplacements (flak guns, howitzers, SAM sites, rocket batteries), enemy missile strikes from
  small salvos to huge warheads, rocket troopers with guided missiles, jets crossing the view
  every few seconds, twice the explosive barrels plus TNT, and open oceans crossed on a
  causeway, with battleships riding in them and submarines that surface to fire missiles.
- **Fourteen weapons:** each player starts with their own (rifle, shotgun, plasma, rockets,
  sniper, flamethrower, grenades, minigun) and picks up the rest from supply crates that
  parachute in: a laser, a tesla gun whose lightning jumps between enemies, dual pistols, an
  explosive crossbow, ricocheting saw blades, a mortar. Some crates hold jetpacks, and the
  squad takes big climbs on them too.
- **Controls:** in the spectator space, pause, resume and play the background at ½×, 1×, 2×, 4×
  or 8× (or P, [ and ] on the keyboard). The camera keeps the squad a third of the way across,
  and enemies hold fire until they're on screen, so the fighting stays in view.
- **Seamless levels:** the land runs straight on from one biome into the next (across a river,
  the sky and backdrop crossfading), and that's the next level. The stronghold and its finale
  come once every 20 levels; switch on **Endless** (in the scene controls) and they never come.
  Biomes are dealt from a shuffled deck (every one comes up once before any comes up again, so
  there's no bias), and each is varied as it's dealt: dawn, dusk or night, rain, snow, storms or
  fog, flatter or more mountainous, a different mix of features. Among the 28: savanna, autumn
  woods, a blossom valley, tundra, badlands, farmland, suburbs, a megacity, an airbase, an
  archipelago and mangroves.
- **Cities:** dense city blocks shoulder to shoulder with side streets, traffic and street
  furniture: hospitals, fuel stations, police and fire stations, schools, shops, hotels, banks,
  offices and flats, defended by machine-gun nests and armoured cars. Also highways with stopped
  traffic and convoys, and airfields with a runway, control tower, hangars and parked aircraft.
- **Tougher ground:** soils and rocks in layers (loam, clay, gravel, basalt, chalk) and hard
  granite country that no blast can dig; ordinary explosions take a bite rather than a crater,
  so the fighting isn't always in a wasteland. Fewer caves, more mineshafts with headframes.
- **The wider world:** open oceans as deep as the world, swum by the squad and the troops (the
  vehicles wait at the shore) with reefs, islands, battleships (three turrets, anti-aircraft
  mounts, radar, lifeboats), submarines and patrol boats; deep lakes, many more rivers and
  waterfalls; forests and far more trees; rail yards with an armoured train; nuclear plants
  with cooling towers and a reactor hall; and buildings in the old styles of many lands
  (pagodas, a Japanese castle, temple halls, stupas, palaces with onion domes, a domed hall with
  a minaret, riads and bazaars, a pyramid, Greek temples, Roman villas, palazzos, haciendas and
  missions, azulejo houses, mansard apartments, canal houses with stepped gables, guild halls,
  Tudor houses, clock towers, castles, chalets, stave and onion-domed churches, brutalist blocks,
  brownstones, art deco towers, favelas, igloos, gers, shophouses and more).
- **Random endings:** every level ends a different way, never the same twice running: an ICBM
  salvo, one enormous nuke, low cruise missiles, the enemy's own silos firing, or a MIRV
  bursting into a dozen warheads. The last ones always land on the squad, throwing them
  through the air on fire. The "Level complete" card's title and bonus are random too.
- **Explosions:** lumpy, outlined fireball clusters that swell, rise and cool to smoke;
  spark streaks, burning cinders, dust rolling along the ground, enemies flung on fire.
- **Quiet HUD:** just the section, the menu and the scroll meter; the game's stats stay
  in the spectator panel. The intro shows "Portfolio Website" and the eight players in a slim
  column down the side; the name is the page's heading for screen readers only.
- **Design:** Press Start 2P, Silkscreen and Pixelify Sans fonts, notched pixel frames,
  press-down buttons, inventory-slot skills, stat bars, an ID-card portrait, an 8-player
  title-screen line-up, a squad select on the Domains cards, a diagonal block wipe between
  pages, and pictures that resolve from big blocks to sharp.
- With reduced motion the fight is shown frozen mid-action, with no white-out.

`?level=<theme>` picks the first level, for example `?level=temple` or `?level=monsoon`. The keys
are `jungle deepjungle temple monsoon nightraid village swamp desert canyon snow fortress refinery city inferno hive silos hq cave`.
`?quality=low|mid` picks a lighter budget (fewer enemies and particles) for testing.

## Where things live

- `src/components/pixel/themes.ts`: the level themes (look, enemies, bosses, weather) and the random rotation
- `src/components/pixel/level.ts`: the level generator (terrain, features, boss arena, stronghold). Pure data, unit-tested
- `src/components/pixel/game.ts`: the players, enemies, bosses, weapons, explosions, ICBM finale and level flow. Plain logic, unit-tested
- `src/components/pixel/art.ts`: procedural pixel art (blocks, backdrops, sprites, vehicles, bosses, pixel font)
- `src/components/pixel/PixelScene.tsx`: renderer, camera, the distant battle, lights pass, white-out and level changes, visitor input
- `src/components/Watch.tsx`: the spectator space and its level-intel panel (and the screen wake lock)
- `src/data/squad.ts`: the eight players and what they stand for
- `src/components/fx/PixelImage.tsx`: the mosaic reveal
- `src/index.css`: the whole pixel theme

## Checks

```bash
npm run lint
npm run typecheck
npm test             # unit tests (vitest)
npm run test:e2e     # build + browser tests (Playwright, uses your installed Chrome)
```
