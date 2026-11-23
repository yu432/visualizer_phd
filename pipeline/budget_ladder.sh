#!/bin/bash
# Climb the frame budget in small steps and record what each step costs in memory.
#
# The 16-frame protocol is our own convention, T&T has no official frame count, so the real
# question is where this machine stops. 100 frames already ran, so the ladder starts there
# and adds a few frames at a time, stopping at the first step that cannot finish.
#
# Every step is watched by mem_guard.py, which now trips on system available memory rather
# than process RSS: MPS activations live in Metal buffers and were invisible to RSS while a
# run drove the machine from 26 GiB free to 2 GiB. A step that gets killed stops the ladder,
# because the first failure is the answer.
#
# usage: budget_ladder.sh <omega512|omega384|da3-504> <scene> <budget> [budget...]
set -u
cd "$(dirname "$0")/../.."

method=$1
scene=$2
shift 2
budgets=$*

OUT=outputs/_probe
LOGS=$OUT/logs
mkdir -p "$LOGS"
# This Mac keeps about 15 GiB available in normal use (Chrome, editor, harness), and free
# memory is near zero all the time because macOS spends it on cache. So the entry threshold is
# what the machine actually sustains, and the tripwire fires while 6.5 GiB is still available:
# a freeze only starts once available and swap are both exhausted.
MIN_AVAILABLE=6.5
REQUIRE=15
MAX_SECONDS=3600
RESULTS=$LOGS/ladder-$method-$scene.jsonl
: > "$RESULTS"

worker_cmd () {
  budget=$1
  dataset=tntb$budget
  case $method in
    omega512) echo ".venv/bin/python method/geometry/omega_worker.py --dataset $dataset --scene $scene --out-root $OUT --method vggt-omega --device mps --image-resolution 512 --force" ;;
    omega384) echo ".venv/bin/python method/geometry/omega_worker.py --dataset $dataset --scene $scene --out-root $OUT --method vggt-omega --device mps --image-resolution 384 --force" ;;
    da3-504)  echo ".venv/bin/python method/geometry/da3_geometry_worker.py --dataset $dataset --scene $scene --out-root $OUT --device mps --process-res 504 --force" ;;
    *) echo "" ;;
  esac
}

wait_for_headroom () {
  deadline=$(( $(date +%s) + 1200 ))
  while :; do
    avail=$(.venv/bin/python -c 'import psutil; print(int(psutil.virtual_memory().available / 2**30))')
    if [ "$avail" -ge "$REQUIRE" ]; then return 0; fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "ladder: only $avail GiB available after 20 min of waiting, stopping"
      return 1
    fi
    sleep 15
  done
}

for budget in $budgets; do
  dataset=tntb$budget
  if [ ! -f "data/inference/$dataset/$scene/manifest.json" ]; then
    echo "ladder: selecting $budget frames for $scene"
    .venv/bin/python method/geometry/prepare_tnt.py --dataset "$dataset" --gt-source tnt \
      --budget "$budget" --scenes "$scene" > "$LOGS/prep-$dataset.log" 2>&1 || {
        echo "ladder: preparation failed for $dataset, see $LOGS/prep-$dataset.log"; break; }
  fi
  echo "ladder: waiting for $REQUIRE GiB available before the $budget frame step"
  wait_for_headroom || break
  cmd=$(worker_cmd "$budget")
  state="$LOGS/$method-$scene-$budget.json"
  echo "ladder: === $budget frames ==="
  .venv/bin/python method/geometry/mem_guard.py --min-available-gib $MIN_AVAILABLE \
    --require-free-gib $REQUIRE --max-seconds $MAX_SECONDS --interval 0.1 --report-every 60 \
    --state "$state" -- $cmd > "$LOGS/$method-$scene-$budget.log" 2>&1
  code=$?
  tail -2 "$LOGS/$method-$scene-$budget.log"
  .venv/bin/python - "$state" "$budget" "$code" "$RESULTS" <<'PY'
import json, sys
state, budget, code, results = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
row = json.load(open(state))
row["budget"] = budget
row["worker_returncode"] = code
open(results, "a").write(json.dumps(row, sort_keys=True) + chr(10))
print("ladder: %d frames -> %s, memory used %.1f GiB, lowest available %.1f GiB, %.0f s"
      % (budget, row["status"], row["peak_system_gib"], row["min_free_gib"], row["elapsed_seconds"]))
PY
  status=$(.venv/bin/python -c "import json,sys; print(json.load(open('$state'))['status'])")
  if [ "$status" != "completed" ]; then
    echo "ladder: stopping at $budget frames ($status)"
    break
  fi
done
