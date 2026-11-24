#!/bin/sh
# Rebuild the public site from the experiment artifacts and push it.
#
# The site repository lives at outputs/oss (ignored by the private repo); this script keeps it
# in sync without hand-copying files: regenerate, replace the generated parts, commit, push.
set -eu
cd "$(dirname "$0")/../.."

EXP=docs/experiments/2026-11-22-geometry-clouds
SITE_REPO=outputs/oss

MPLCONFIGDIR=$PWD/.mpl .venv/bin/python method/geometry/make_site.py \
  --summary "$EXP/summary.json" \
  --tolerance "$EXP/tolerance.json" \
  --figures "$EXP/figures" \
  --figures-dense "$EXP/figures-eth3d20" \
  --viewer "$EXP/viewer" \
  --report "$EXP/da3-omega-geometry.pdf" \
  --repo https://github.com/yu432/visualizer_phd \
  --out outputs/site

rm -rf "$SITE_REPO"/data "$SITE_REPO"/viewer "$SITE_REPO"/index.html "$SITE_REPO"/report.pdf
cp -R outputs/site/. "$SITE_REPO"/
# The pipeline and the prose live in the site repo and are not generated: refresh the code only.
rsync -a --exclude __pycache__ method/geometry/ "$SITE_REPO"/pipeline/

cd "$SITE_REPO"
git add -A
git -c user.name="yu432" -c user.email="yu432@users.noreply.github.com" \
  commit -q -m "${1:-Update site}" || echo "nothing to commit"
# The payload is a few hundred megabytes: HTTP/1.1 with a large post buffer keeps a spotty
# connection from killing the push (GitHub drops HTTP/2 streams at ~300 MB with a broken pipe).
git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 push -q origin main
echo "pushed; site https://yu432.github.io/visualizer_phd/"
