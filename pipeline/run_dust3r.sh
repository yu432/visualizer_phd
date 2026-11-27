#!/bin/sh
# DUSt3R on the four site schemes: pairwise network, then the same post-processing every other
# method gets (SegFormer sky mask, same voxel, same evaluation limits, same tolerance table), so
# the three methods stay comparable on the eighteen pages.
#
# DUSt3R is the only method that needs a solver after the network: the global aligner optimises
# per-pair camera poses and per-frame depthmaps into one consistent set of cameras. Everything
# from sky_segment.py on is method-agnostic.
#
# Timing on this machine (M3 Max, MPS, one run at a time): ten frames ~2.5 min, twenty ~6 min,
# sixteen ~4.5 min, one hundred (cyclic sliding window of three, 600 pairs) ~8 min.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
ETH_SCENES="courtyard delivery_area facade office terrace"
TNT_SCENES="barn courthouse ignatius meetingroom"
GT_TNT="barn courthouse"

for dataset in eth3d eth3d20; do
  for scene in $ETH_SCENES; do
    $PY method/geometry/dust3r_worker.py --dataset $dataset --scene $scene --device mps || exit 1
  done
done
for dataset in tnt tnt100; do
  for scene in $TNT_SCENES; do
    $PY method/geometry/dust3r_worker.py --dataset $dataset --scene $scene --device mps || exit 1
  done
done
echo "INFERENCE-DONE"

for dataset in eth3d eth3d20; do
  $PY method/geometry/sky_segment.py --dataset $dataset --scenes $ETH_SCENES --method dust3r
  $PY method/geometry/build_clouds.py --dataset $dataset --scenes $ETH_SCENES --method dust3r --force
  $PY method/geometry/evaluate_poses.py --dataset $dataset --scenes $ETH_SCENES --method dust3r || exit 1
done
for dataset in tnt tnt100; do
  $PY method/geometry/sky_segment.py --dataset $dataset --scenes $TNT_SCENES --method dust3r
  # the metric voxel of the other two methods on Tanks and Temples: 1.5 cm in GT units
  $PY method/geometry/build_clouds.py --dataset $dataset --scenes $TNT_SCENES --method dust3r \
      --voxel-meters 0.015 --force
  $PY method/geometry/evaluate_poses.py --dataset $dataset --scenes $TNT_SCENES --method dust3r || exit 1
done
echo "CLOUDS-AND-POSES-DONE"

# ETH3D keeps the ICP polish that the published numbers use; Tanks and Temples does not, because
# ICP there repairs gross pose failures and would hide them.
for dataset in eth3d eth3d20; do
  $PY method/geometry/evaluate.py --dataset $dataset --scenes $ETH_SCENES --method dust3r \
      --limit 80000 --force || exit 1
done
for dataset in tnt tnt100; do
  $PY method/geometry/evaluate.py --dataset $dataset --scenes $GT_TNT --method dust3r \
      --limit 80000 --no-icp --force || exit 1
done
echo "METRICS-DONE"

MPLCONFIGDIR=$PWD/.mpl $PY method/geometry/tolerance_metrics.py --dataset eth3d \
    --scenes $ETH_SCENES --methods da3-giant vggt-omega dust3r --out outputs/eth3d/tolerance.json || true
MPLCONFIGDIR=$PWD/.mpl $PY method/geometry/tolerance_metrics.py --dataset eth3d20 \
    --scenes $ETH_SCENES --methods da3-giant vggt-omega dust3r --out outputs/eth3d20/tolerance.json || true
MPLCONFIGDIR=$PWD/.mpl $PY method/geometry/tolerance_metrics.py --dataset tnt \
    --scenes $GT_TNT --methods da3-giant vggt-omega dust3r --out outputs/tnt/tolerance.json || true
MPLCONFIGDIR=$PWD/.mpl $PY method/geometry/tolerance_metrics.py --dataset tnt100 \
    --scenes $GT_TNT --methods da3-giant vggt-omega dust3r --out outputs/tnt100/tolerance.json || true
echo "DUST3R-ALL-DONE"
