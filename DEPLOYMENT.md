# Deploying with a private source repo (free GitHub account)

On a free account, GitHub Pages only works for **public** repos, so the site and its
source normally sit side by side in public. The setup below fixes that with two repos.

| Repo | Visibility | Contains |
| --- | --- | --- |
| `alexikehayiasportoflio-source` | **Private** | All website source (this repo, including its full history) |
| `alexikehayiasportoflio` | Public | `gh-pages` branch: the **compiled** site only · `main` branch: the `Projects/` files |

The public repo keeps the current name, so the site URL stays
`https://joseph-leaf24.github.io/alexikehayiasportoflio/` and the "View on GitHub"
project links keep working.

> **What this does and doesn't protect.** Anything a browser downloads (minified
> JS, CSS, images) can always be saved by a visitor. This setup keeps the
> readable source, file structure, comments and git history private. The build
> also ships no source maps and minified shaders. The LICENSE and copyright
> notices let you file a DMCA takedown if someone re-hosts your work.

---

## One-time setup (about 15 minutes)

The site goes offline between step 1 and step 7.

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

### 4. Publish the project files to it
In Git Bash, from this folder:
```bash
bash scripts/publish-project-files.sh
```
This pushes only `Projects/`, a README and the LICENSE to the public repo's `main`.
It refuses to run if `origin` still points at the public repo. Re-run it whenever
your project files change.

### 5. Create a deploy key so the private repo can publish
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

### 6. Deploy
```bash
git push
```
The workflow sees `PUBLIC_PAGES_REPO`, builds the site and pushes only `dist/` to the
public repo's `gh-pages` branch, as a single commit with no history.
You can watch it under **Actions** in the private repo.

### 7. Turn on Pages in the public repo
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
  there and generate a new one (step 5).
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
- Edit the site, then `git push`. It deploys automatically.
- Changed project files? Run `bash scripts/publish-project-files.sh`.

## Alternative: GitHub Pro (free for students)
The [GitHub Student Developer Pack](https://education.github.com/pack) includes
GitHub Pro, which can serve Pages from a **private** repo. With Pro you can skip
the two-repo setup:
1. Make this repo private. Leave `PUBLIC_PAGES_REPO` unset, and the workflow keeps
   deploying from this repo.
2. Put `Projects/` in a separate public repo (`scripts/publish-project-files.sh <url>`)
   and change `PROJECT_FILES_REPO` in `src/data/projects.ts` to that repo.
