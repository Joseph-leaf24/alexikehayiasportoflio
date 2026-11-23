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
  down parachute back in.
- **A battle in the distance:** behind the hills things keep going off: explosions on the
  horizon, flak, anti-aircraft tracers, smoke columns from fires far away, planes (some
  hit and trailing smoke), flocks of birds, and searchlights sweeping the sky at night.
- **Spectate:** below the contact section is open ground to leave running. Scroll right down
  and the battle fills the screen. A panel follows the level in play: objective, progress
  to the stronghold, the boss, enemies down, hostiles in sight, conditions, levels
  cleared, battle time, and the last few levels. The scene keeps its full frame rate there,
  the screen is kept awake (where the browser supports it), and at the end of each level
  the HUD fades out with the white-out so the whole screen goes white. "Watch" in the menu
  goes straight there. On phones the panel starts folded away.
- **You can join in:** click anywhere off the buttons to call in a blast; hold on the
  intro for an airstrike; hover a domain or specialisation tab to point out its player.
- **Co-op HUD:** live P1–P8 cards (health, score), the level number and name, kills and booms.
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
