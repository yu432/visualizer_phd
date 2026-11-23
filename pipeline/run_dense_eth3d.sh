#!/bin/sh
# Dense ETH3D scheme: more than the official ten frames per scene, as many as this machine
# reliably holds. Measured on courtyard, one run at a time under the guard:
#
#   frames  da3-giant 504        vggt-omega 512
#   16      12.8 GiB / 10.9 floor
#   20      13.9 GiB /  9.6 floor   11.8 GiB / 10.3 floor
#   24      12.7 GiB /  9.4 floor   13.6 GiB / 10.2 floor
#   30                              14.4 GiB /  9.8 floor   (da3 killed at 15.6)
#   32      12.9 GiB /  8.8 floor
#   38      killed at 16.4 GiB      killed
#
# Run to run variance is a few GiB because MPS buffers land in wired pages that drain slowly,
# so the scheme targets 20 frames, twice the official set, and falls back to 16 for a scene
# that still trips the guard. ETH3D frames are 6.2 MP against roughly 1-2 MP for T&T, which is
# why T&T sustains 100 frames where ETH3D stops near 30.
#
# Frame staging: prepare_eth3d_geometry.py --budget 20 --dataset eth3d20 --gt-source eth3d
# restages the frames upstream skips and copies back the ground truth cloud back-projected
# from the ten protocol frames, so the reference is fixed and both schemes stay comparable.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=outputs/_probe/logs
PRIMARY=${PRIMARY:-eth3d20}
FALLBACK=${FALLBACK:-eth3d16}
SCENES="courtyard delivery_area facade office terrace"
CHOICE="$OUT/dense-eth3d-choice.txt"
mkdir -p "$OUT"
: > "$CHOICE"

attempt() {
  dataset=$1
  scene=$2
  kind=$3
  case $kind in
    da3)   worker="$PY method/geometry/da3_geometry_worker.py --dataset $dataset --scene $scene --device mps --process-res 504 --force" ;;
    omega) worker="$PY method/geometry/omega_worker.py --dataset $dataset --scene $scene --method vggt-omega --device mps --image-resolution 512 --force" ;;
  esac
  $PY method/geometry/mem_guard.py --min-available-gib 7 --require-free-gib 18 \
      --max-seconds 3600 --report-every 60 \
      --state "$OUT/guard-dense-$kind-$scene-$dataset.json" -- $worker
}

for kind in da3 omega; do
  for scene in $SCENES; do
    if attempt "$PRIMARY" "$scene" "$kind"; then
      echo "$kind $scene $PRIMARY" >> "$CHOICE"
      sleep 45
      continue
    fi
    echo "!! $kind $scene failed at $PRIMARY, retrying at $FALLBACK"
    sleep 90
    if attempt "$FALLBACK" "$scene" "$kind"; then
      echo "$kind $scene $FALLBACK" >> "$CHOICE"
      sleep 45
    else
      echo "!! $kind $scene failed at $FALLBACK as well"
      echo "$kind $scene FAILED" >> "$CHOICE"
    fi
  done
done

# Consolidate fallback predictions into the primary roots so every later stage sees one
# dataset. The scene keeps its own manifest, cameras and ground truth depth, so each scene
# stays internally consistent; only the frame count differs, and the choice file records it.
while read -r kind scene dataset; do
  [ "$dataset" = "$FALLBACK" ] || continue
  method=da3-giant
  [ "$kind" = omega ] && method=vggt-omega
  cp -R "outputs/$FALLBACK/$scene/$method/." "outputs/$PRIMARY/$scene/$method/"
  cp "data/inference/$FALLBACK/$scene/manifest.json" "data/inference/$PRIMARY/$scene/manifest.json"
  cp -R "data/gt/$FALLBACK/$scene/depths/." "data/gt/$PRIMARY/$scene/depths/"
  cp "data/gt/$FALLBACK/$scene/cameras.npz" "data/gt/$PRIMARY/$scene/cameras.npz"
  cp "data/gt/$FALLBACK/$scene/gt_meta.json" "data/gt/$PRIMARY/$scene/gt_meta.json"
  echo "consolidated $scene/$method from $FALLBACK"
done < "$CHOICE"

echo "inference done:"; cat "$CHOICE"
