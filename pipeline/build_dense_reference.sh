#!/bin/sh
# Coverage-matched reference for the dense ETH3D scheme (eth3d20).
#
# The 10-frame published reference only contains the surface the official ETH3D frames see, so
# predicting from 20 frames looks "worse" there even when it is better everywhere: the extra
# frames also reconstruct surface outside the 10-frame footprint, which counts as imprecision.
# This rebuilds the GT cloud from the very same 20 frames' depth maps. Predictions are unchanged
# (identical PLYs, same voxel fraction), only the reference moves, which makes F@1 a fair
# densification test.
set -eu
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
GUARD="$PY method/geometry/mem_guard.py --min-available-gib 7 --require-free-gib 8"
SRC=eth3d20
DST=eth3d20dense
SCENES="delivery_area facade office terrace"

# 0. The reference is built from the very same 20 frames, so its inputs are eth3d20's inputs:
#    the inference manifests (frame list) and the undistorted GT depth maps are symlinked instead
#    of copied.
mkdir -p "data/inference/$DST"
for scene in courtyard $SCENES; do
  [ -e "data/inference/$DST/$scene" ] || ln -s "$PWD/data/inference/$SRC/$scene" "data/inference/$DST/$scene"
  mkdir -p "data/gt/$DST/$scene"
  [ -e "data/gt/$DST/$scene/cameras.npz" ] || ln -s "$PWD/data/gt/$SRC/$scene/cameras.npz" "data/gt/$DST/$scene/cameras.npz"
  [ -e "data/gt/$DST/$scene/depths" ] || ln -s "$PWD/data/gt/$SRC/$scene/depths" "data/gt/$DST/$scene/depths"
done

# 1. GT clouds from the 20 frames themselves (courtyard is kept if it already exists).
$GUARD --max-seconds 1800 --state outputs/_probe/logs/guard-dense-gt.json -- \
  $PY method/geometry/build_gt_cloud.py --dataset $DST --scenes courtyard $SCENES

# 2. Register the existing predictions under the new dataset root. The expensive arrays are
#    symlinked so nothing is duplicated; only the evaluation outputs are real files.
for scene in courtyard $SCENES; do
  cp -f "clouds/$SRC/$scene"/*.ply "clouds/$DST/$scene/" 2>/dev/null || {
    mkdir -p "clouds/$DST/$scene"
    cp -f "clouds/$SRC/$scene"/*.ply "clouds/$DST/$scene/"
  }
  cp -f "clouds/$SRC/$scene"/*_cloud.json "clouds/$DST/$scene/"
  for method in da3-giant vggt-omega; do
    mkdir -p "outputs/$DST/$scene/$method"
    for f in "outputs/$SRC/$scene/$method"/*; do
      base=$(basename "$f")
      case "$base" in
        evaluation.json|pose_evaluation.json|tolerance.json) continue ;;
      esac
      ln -sfn "$PWD/$f" "outputs/$DST/$scene/$method/$base"
    done
  done
done

# 3. Cloud metrics against the coverage-matched reference, same protocol as the published run
#    (voxel fraction fixed, ICP refinement on).
for method in da3-giant vggt-omega; do
  $GUARD --max-seconds 5400 --state outputs/_probe/logs/guard-dense-eval.json -- \
    $PY method/geometry/evaluate.py --dataset $DST --scenes courtyard $SCENES \
      --method $method --limit 80000 --force
done
echo "dense reference done"
