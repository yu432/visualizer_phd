#!/bin/sh
# Dense Tanks and Temples scheme: 100 frames per scene instead of 16.
#
# The frames, the poses and the laser GT are prepared by
#   prepare_tnt.py --dataset tnt100 --gt-source tnt --budget 100 --scenes barn courthouse ignatius meetingroom
# (the laser scan, the official polygon and <Scene>_trans.txt are shared with the 16-frame dataset,
# the COLMAP cameras are re-derived for the new frame set), and the two workers write into
# outputs/tnt100/. Everything below only reads those predictions.
set -eu
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
EXP=docs/experiments/2026-11-22-geometry-clouds
SCENES="barn courthouse ignatius meetingroom"
METHODS="da3-giant vggt-omega"
GT_SCENES="barn courthouse"

for method in $METHODS; do
  $PY method/geometry/sky_segment.py --dataset tnt100 --scenes $SCENES --method $method
  # The dense scheme uses the same metric voxel as the 16-frame one: a fraction of the prediction
  # diagonal would be 17-18 cm in GT units here (the 100-frame prediction covers more ground), i.e.
  # twelve times coarser than the 1.5 cm grid of the 16-frame clouds, and the two schemes would no
  # longer be comparable.
  $PY method/geometry/build_clouds.py --dataset tnt100 --scenes $SCENES --method $method \
      --voxel-meters 0.015 --force
  $PY method/geometry/evaluate_poses.py --dataset tnt100 --scenes $SCENES --method $method
done
for method in $METHODS; do
  # only the two scenes with a laser scan have cloud metrics; pose-only alignment, as in the report
  $PY method/geometry/evaluate.py --dataset tnt100 --scenes $GT_SCENES --method $method \
      --limit 80000 --no-icp --force
done
MPLCONFIGDIR=$PWD/.mpl $PY method/geometry/tolerance_metrics.py --dataset tnt100 \
    --scenes $GT_SCENES --methods $METHODS --out outputs/tnt100/tolerance.json
MPLCONFIGDIR=$PWD/.mpl $PY method/geometry/figures.py --dataset tnt100 --scenes $SCENES \
    --methods $METHODS --out "$EXP/figures"
$PY method/geometry/make_viewer.py --dataset tnt100 --scenes $SCENES --methods $METHODS \
    --no-index --out "$EXP/viewer"
echo "dense scheme done; now: summarize.py --methods da3-giant vggt-omega --out $EXP"
