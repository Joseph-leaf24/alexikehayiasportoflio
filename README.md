# Alexi Kehayias — Data Science & AI Portfolio

An interactive portfolio built as a voxel Cycladic village under a sky of slowly
colour-shifting cubes. The village is made of WebGL cubes that break apart and re-form
into new shapes as you scroll: clusters, a neural network, data bars. The light follows
the real time of day in Amsterdam (sun by
day, moon by night), and clicking and holding on the opening screen shatters the whole
scene, sun and moon included. It presents my projects across data science, data
engineering and data analytics.

**Live site:** https://joseph-leaf24.github.io/alexikehayiasportoflio/

<!-- Workflow badges use repo-relative links, so they keep working if the repo is renamed.
     The number badges read JSON that each deploy publishes under /badges/ on the live site. -->

[![Code quality](../../actions/workflows/quality.yml/badge.svg)](../../actions/workflows/quality.yml)
[![Deploy](../../actions/workflows/deploy.yml/badge.svg)](../../actions/workflows/deploy.yml)
[![Site health](../../actions/workflows/site-health.yml/badge.svg)](../../actions/workflows/site-health.yml)
[![Website](https://img.shields.io/website?url=https%3A%2F%2Fjoseph-leaf24.github.io%2Falexikehayiasportoflio%2F&label=site)](https://joseph-leaf24.github.io/alexikehayiasportoflio/)

[![Tests](https://img.shields.io/endpoint?url=https%3A%2F%2Fjoseph-leaf24.github.io%2Falexikehayiasportoflio%2Fbadges%2Ftests.json)](../../actions/workflows/deploy.yml)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fjoseph-leaf24.github.io%2Falexikehayiasportoflio%2Fbadges%2Fcoverage.json)](../../actions/workflows/deploy.yml)
[![Vulnerabilities](https://img.shields.io/endpoint?url=https%3A%2F%2Fjoseph-leaf24.github.io%2Falexikehayiasportoflio%2Fbadges%2Fvulnerabilities.json)](../../actions/workflows/deploy.yml)
[![Dependabot](https://img.shields.io/badge/Dependabot-enabled-025E8C?logo=dependabot&logoColor=white)](.github/dependabot.yml)

[![Lighthouse performance](https://img.shields.io/endpoint?url=https%3A%2F%2Fjoseph-leaf24.github.io%2Falexikehayiasportoflio%2Fbadges%2Flighthouse-performance.json&logo=lighthouse&logoColor=white)](lighthouserc.json)
[![Lighthouse accessibility](https://img.shields.io/endpoint?url=https%3A%2F%2Fjoseph-leaf24.github.io%2Falexikehayiasportoflio%2Fbadges%2Flighthouse-accessibility.json&logo=lighthouse&logoColor=white)](lighthouserc.json)
[![Lighthouse best practices](https://img.shields.io/endpoint?url=https%3A%2F%2Fjoseph-leaf24.github.io%2Falexikehayiasportoflio%2Fbadges%2Flighthouse-best-practices.json&logo=lighthouse&logoColor=white)](lighthouserc.json)
[![Lighthouse SEO](https://img.shields.io/endpoint?url=https%3A%2F%2Fjoseph-leaf24.github.io%2Falexikehayiasportoflio%2Fbadges%2Flighthouse-seo.json&logo=lighthouse&logoColor=white)](lighthouserc.json)

[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![Vite](https://img.shields.io/badge/Vite-7-646CFF?logo=vite&logoColor=white)](https://vite.dev)
[![three.js](https://img.shields.io/badge/three.js-r187-000000?logo=threedotjs&logoColor=white)](https://threejs.org)
[![GSAP](https://img.shields.io/badge/GSAP-3-88CE02?logo=greensock&logoColor=black)](https://gsap.com)
[![Docker](https://img.shields.io/badge/Docker-run%20locally-2496ED?logo=docker&logoColor=white)](#run-it-in-docker)

## Run it locally

Needs Node.js 22 or newer.

```bash
npm ci
npm run dev          # http://localhost:8080/alexikehayiasportoflio/
```

`npm run build` writes the static site to `dist/`, and `npm run preview` serves that build.
The 3D scene always renders at full quality on every device: the screen's own resolution
(up to 2×), bloom and grain, and nothing is scaled down while it runs (`src/lib/device.ts`).
Add `?quality=low` or `?quality=mid` to the URL to try a lighter budget by hand, and
`?time=HH:MM` (Amsterdam time, e.g. `?time=16:30`) to see the scene at another time of day.

## Run it in Docker

To run the site in a container (separate from the GitHub Pages deployment):

```bash
npm run docker       # same as: docker compose up --build
```

Then open http://localhost:8080/alexikehayiasportoflio/, and stop it with `docker compose down`.
The image is built in two stages. Node builds the site, then an unprivileged nginx serves
it with security headers, caching and gzip. Compose runs it read-only with no privileges.

## Tests and checks

| Command | What it does |
| --- | --- |
| `npm test` | Unit tests (Vitest + Testing Library) |
| `npm run test:coverage` | Unit tests with coverage (fails below 80%) |
| `npm run test:e2e` | Builds, then runs the Playwright smoke and axe accessibility tests (desktop and phone) |
| `npm run lint` | ESLint |
| `npm run scan:secrets` | Scans for committed keys, tokens and passwords |

Every pull request and every push runs these in GitHub Actions, along with:
- a dependency audit;
- Lighthouse CI;
- checks of the workflow files and the Docker image.

A push to `main` deploys only if all of them pass. See [DEPLOYMENT.md](DEPLOYMENT.md) for
how deployment, the checks and the weekly site-health run work.

## Project layout

```
src/
  components/      sections (Hero, About, Domains, Skills, Work, Contact) and UI
    gl/            the WebGL scene: layouts, shaders, renderer
    fx/            effects: scramble text, reveals, depth images, project network graph
    chrome/        HUD, cursor, loader, page transitions
  pages/           home, project detail, 404
  data/            profile, projects, domains, skills
  lib/             device tiers, smooth scrolling, shared state
  test/            unit tests
e2e/               Playwright browser tests
scripts/           secrets scan, README badge data
docker/            nginx config for the container
.github/           CI/CD workflows and Dependabot
```

## License

Copyright © 2025–2026 Alexi Kehayias. All rights reserved. See [LICENSE](LICENSE).
