#!/usr/bin/env bash
# Publishes ONLY the Projects/ folder (plus README + LICENSE) to the PUBLIC repo's
# main branch, so the "View on GitHub" links on the site keep working while the
# website source stays in the private repo.
#
# Usage (Git Bash, from anywhere inside this repo):
#   bash scripts/publish-project-files.sh
#   bash scripts/publish-project-files.sh https://github.com/<user>/<public-repo>.git
#
# Re-run it whenever you add or change project files.
set -euo pipefail

PUBLIC_REPO="${1:-https://github.com/Joseph-leaf24/alexikehayiasportoflio.git}"
ROOT="$(git rev-parse --show-toplevel)"
ORIGIN="$(git -C "$ROOT" remote get-url origin 2>/dev/null || true)"

norm() { echo "$1" | sed -E 's#^git@github.com:#https://github.com/#; s#\.git$##; s#/$##' | tr '[:upper:]' '[:lower:]'; }

# Safety: never force-push over the private source repo.
if [ -n "$ORIGIN" ] && [ "$(norm "$ORIGIN")" = "$(norm "$PUBLIC_REPO")" ]; then
  echo "✋ Refusing: '$PUBLIC_REPO' is this repo's own origin."
  echo "   Point origin at the private source repo first (see DEPLOYMENT.md, step 2)."
  exit 1
fi

if [ ! -d "$ROOT/Projects" ]; then
  echo "No Projects/ folder found in $ROOT" >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "→ Copying Projects/ ($(du -sh "$ROOT/Projects" | cut -f1))…"
cp -r "$ROOT/Projects" "$TMP/Projects"
cp "$ROOT/LICENSE" "$TMP/LICENSE"
cat > "$TMP/README.md" <<'EOF'
# Alexi Kehayias — Project Files

Notebooks, reports and dashboards behind the projects in my portfolio:
**https://joseph-leaf24.github.io/alexikehayiasportoflio/**

© Alexi Kehayias. All rights reserved — see [LICENSE](LICENSE).
EOF

cd "$TMP"
git init -q -b main
git add -A
git -c user.name="$(git -C "$ROOT" config user.name || echo portfolio)" \
    -c user.email="$(git -C "$ROOT" config user.email || echo portfolio@users.noreply.github.com)" \
    commit -q -m "Publish project files"
echo "→ Pushing to $PUBLIC_REPO (main)…"
git push --force "$PUBLIC_REPO" main
echo "✓ Project files published."
