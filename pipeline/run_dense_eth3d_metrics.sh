#!/bin/sh
# Metrics for the dense ETH3D scheme (eth3d20: 20 frames per scene, 16 for any scene the guard
# could not fit). Mirrors the ten frame path so the two schemes are comparable: same voxel
# fraction, same ICP alignment as the published eth3d numbers, same 80000 point evaluation cap.
set -eu
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
EXP=docs/experiments/2026-11-22-geometry-clouds
DATASET=eth3d20
SCENES="courtyard delivery_area facade office terrace"
METHODS="da3-giant vggt-omega"

for method in $METHODS; do
  $PY method/geometry/sky_segment.py --dataset $DATASET --scenes $SCENES --method $method --force
  $PY method/geometry/build_clouds.py --dataset $DATASET --scenes $SCENES --method $method --force
  $PY method/geometry/evaluate_poses.py --dataset $DATASET --scenes $SCENES --method $method
done
for method in $METHODS; do
  $PY method/geometry/evaluate.py --dataset $DATASET --scenes $SCENES --method $method \
      --limit 80000 --force
done
MPLCONFIGDIR=$PWD/.mpl $PY method/geometry/tolerance_metrics.py --dataset $DATASET \
    --scenes $SCENES --methods $METHODS --out outputs/$DATASET/tolerance.json \
    --figure "$EXP/figures-tolerance-eth3d20.png"
MPLCONFIGDIR=$PWD/.mpl $PY method/geometry/figures.py --dataset $DATASET --scenes $SCENES \
    --methods $METHODS --out "$EXP/figures-eth3d20"
echo "metrics done"
