# Deploying with a private source repo (free GitHub account)

On a free account, GitHub Pages only works for **public** repos, so the site and its
source normally sit side by side in public. The setup below fixes that with two repos.

| Repo | Visibility | Contains |
| --- | --- | --- |
| `alexikehayiasportoflio-source` | **Private** | All website source (this repo, including its full history) |
| `alexikehayiasportoflio` | Public | `gh-pages` branch: the **compiled** site only |

The public repo keeps the current name, so the site URL stays
`https://joseph-leaf24.github.io/alexikehayiasportoflio/`.

> **What this does and doesn't protect.** Anything a browser downloads (minified
> JS, CSS, images) can always be saved by a visitor. This setup keeps the
> readable source, file structure, comments and git history private. The build
> also ships no source maps and minified shaders. The LICENSE and copyright
> notices let you file a DMCA takedown if someone re-hosts your work.

---

## One-time setup (about 15 minutes)

The site goes offline between step 1 and step 6.

### 1. Make the current repo the private source repo
On github.com, open **Joseph-leaf24/alexikehayiasportoflio → Settings → General**:
1. Rename it to `alexikehayiasportoflio-source`.
2. Under **Danger Zone → Change visibility**, choose **Private**.

### 2. Point your local clone at the renamed repo *(do this before step 3)*
```bash
git remote set-url origin https://github.com/Joseph-leaf24/alexikehayiasportoflio-source.git
git remote -v   # should show ...-source.git
```
⚠️ If you skip this, your next `git push` goes to the new public repo from step 3
and publishes your source.

### 3. Create the new public repo
**New repository** → name `alexikehayiasportoflio` → **Public** → leave it empty
(no README, no .gitignore).

### 4. Create a deploy key so the private repo can publish
```bash
ssh-keygen -t ed25519 -C "pages-deploy" -f pages_deploy_key -N ""
```
- **Public repo** → Settings → **Deploy keys** → *Add deploy key*: paste the contents of
  `pages_deploy_key.pub` and tick **Allow write access**.
- **Private repo** → Settings → **Secrets and variables → Actions**:
  - **Secrets** tab → *New repository secret*, named `PAGES_DEPLOY_KEY`, with the full
    contents of `pages_deploy_key` (the file *without* `.pub`).
  - **Variables** tab → *New repository variable*, named `PUBLIC_PAGES_REPO`, with the
    value `Joseph-leaf24/alexikehayiasportoflio`.
- Then delete both key files from your computer.

### 5. Deploy
```bash
git push
```
The workflow sees `PUBLIC_PAGES_REPO`, builds the site and pushes only `dist/` to the
public repo's `gh-pages` branch, as a single commit with no history.
You can watch it under **Actions** in the private repo.

### 6. Turn on Pages in the public repo
**Public repo → Settings → Pages → Source: Deploy from a branch → `gh-pages` / `(root)`**.
The site is live again within a minute or two.

---

## Security checklist

**Source code**
- [ ] Two-repo setup above is done, with the website source in the **private** repo.
- Already in place: no source maps are shipped, JS is minified and mangled, and
  shaders are minified at build time. LICENSE (all rights reserved), a copyright
  notice in the browser console, and `"license": "UNLICENSED"` in package.json.
- Right-click blocking or "DevTools detection" doesn't stop anyone and hurts real
  visitors, so it is deliberately not used. If someone re-hosts your work, file a
  DMCA takedown with GitHub or the host. Your copyright already covers it.

**GitHub account**
- [ ] Turn on two-factor authentication: GitHub → Settings → Password and
  authentication. Use a passkey or an authenticator app.
- [ ] Don't add collaborators to the private repo unless you need them.
- [ ] The deploy key can only write to the public repo. If it ever leaks, delete it
  there and generate a new one (step 4).
- The site has no backend, API keys or secrets, so there is nothing to steal from
  the deployed page.

**Attacks and DDoS**
- The site is static files on GitHub Pages, behind GitHub's global CDN, which
  has DDoS mitigation built in. There is no server, database or form for an
  attacker to overload or exploit. The worst case of a flood is the site being
  slow or briefly unavailable, with nothing compromised.
- The production build ships a strict **Content-Security-Policy**: the page may only
  run its own scripts, load Google Fonts, and embed Power BI. This blocks injected
  scripts and third-party trackers.
- Optional extra layer: buy a custom domain (around €10 a year) and put it behind
  **Cloudflare's free plan** (proxied DNS). That adds rate limiting, bot protection
  and its own DDoS shield. In the public repo, Settings → Pages → Custom domain,
  set Cloudflare SSL to *Full*.

## Day to day
- Edit the site, then `git push`. A push to `main` runs every quality check below, then
  builds, deploys and confirms the live site is serving the new build. If any check
  fails, nothing is deployed.
- Unit tests: `npm test` (run once), `npm run test:watch` (while coding),
  `npm run test:coverage` (with the coverage table).
- Browser tests: `npm run test:e2e` builds the site and runs the Playwright smoke and
  accessibility tests in your installed Google Chrome (desktop and phone sizes).
- Lighthouse: after `npm run build`, run `npx @lhci/cli@0.15.1 autorun` for the same
  scores and thresholds as CI (`lighthouserc.json`).
- Every device gets the full quality (screen resolution up to 2×, bloom, grain, every
  page effect; `src/lib/device.ts`). Try a lighter budget by hand with `?quality=low` or
  `?quality=mid` in the URL.
- The scene follows the real time in Amsterdam. Add `?time=HH:MM` (e.g. `?time=16:30`) to
  see another time of day; the link-preview image `public/og-image.jpg` is the hero at
  `?time=16:30`, 1200×630.
- Run it in Docker instead: `npm run docker` (or `docker compose up --build`), then open
  http://localhost:8080/alexikehayiasportoflio/. This is only for running the site
  locally; GitHub Pages is deployed by the workflow as before and doesn't use the image.

## Automated checks
`.github/workflows/quality.yml` runs on pull requests, on pushes to other branches, and
inside the deploy for `main`:

| Job | What it checks | Run it locally |
| --- | --- | --- |
| Website | ESLint, TypeScript, unit tests with ≥ 80% coverage, and `npm audit` (fails on high/critical advisories in the dependencies the site ships) | `npm run lint`, `npx tsc -p tsconfig.app.json --noEmit`, `npm run test:coverage`, `npm audit --omit=dev` |
| Browser | The built site in Chromium: pages load without errors, navigation and the image viewer work, nothing overflows on a phone, zero axe accessibility violations; then Lighthouse (accessibility, best practices and SEO ≥ 95, performance reported only) | `npm run test:e2e`, Lighthouse as above |
| Workflows | actionlint (workflow mistakes, plus shellcheck on every script) and zizmor (workflow security; settings in `.github/zizmor.yml`) | not needed day to day |
| Docker | hadolint on the `Dockerfile`, then the image builds, starts locked down (read-only, no privileges) and serves the site | `npm run docker` |
| Python | PEP 8 compliance of the code in `Projects/` (flake8, settings in `.flake8`) | `pip install flake8` then `npm run lint:python` |
| Secrets | API keys, tokens or hard-coded passwords in the code | `npm run scan:secrets` |

Other automation:
- **README badges**: the deploy publishes the latest test count, coverage, audit and
  Lighthouse scores as small JSON files under `/badges/` on the live site
  (`scripts/badges.mjs`); the badges in `README.md` read them through shields.io.
- **Deploy verification** (`deploy.yml`): after each deploy, waits until the live URL
  serves the new build and fails the run if it doesn't within about 7 minutes.
- **Site health** (`site-health.yml`, Mondays and on demand from the Actions tab):
  checks the live site, app bundle and CV respond, and that every external link in the
  site still works (lychee, settings in `lychee.toml`). Failures arrive by email like any
  failed run. Set the repository variable `SITE_URL` if the address changes.
- **Dependabot** (`.github/dependabot.yml`): weekly grouped PRs for npm minor/patch
  updates, the Docker base images and the GitHub Actions used here (images and actions
  are pinned to exact digests/commits, so Dependabot is what keeps them current). Each PR
  runs the checks above. Major npm upgrades are left for you to do by hand.

Never commit credentials: use environment variables or GitHub repository secrets. The
secrets scan fails the build if a key, token or hard-coded password slips in.

### Repository settings to turn on (one-off)
- **Settings → Code security → Dependabot alerts** and **Dependabot security updates**:
  fix PRs for newly published vulnerabilities, without waiting for the weekly run.
- While the repo is **public** (free only on public repos): **Code scanning → CodeQL
  analysis → Default setup**, and **Secret Protection → Push protection**, which blocks
  pushes that contain secrets.
- With GitHub Pro (see below), also add a branch ruleset on `main` that requires the
  Code quality checks, so a failing pull request can't be merged.

## Alternative: GitHub Pro (free for students)
The [GitHub Student Developer Pack](https://education.github.com/pack) includes
GitHub Pro, which can serve Pages from a **private** repo. With Pro you can skip
the two-repo setup: make this repo private and leave `PUBLIC_PAGES_REPO` unset, and
the workflow keeps deploying from this repo.
