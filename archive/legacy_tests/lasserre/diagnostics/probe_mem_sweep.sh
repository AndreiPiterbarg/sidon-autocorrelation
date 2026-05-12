#!/usr/bin/env bash
# Memory culprit sweep for monolithic Lasserre SOS-dual.
#
# Runs probe_mem.py under several (threads × z2 × build-only × windows)
# combinations at a chosen d and dumps JSON per run.  Each run is a
# fresh Python process so peak RSS is isolated.
#
# Call:
#   tests/probe_mem_sweep.sh <d> [<outdir>]
# e.g.
#   tests/probe_mem_sweep.sh 12 data/probe_d12
#
# After running, aggregate with:
#   python tests/probe_mem_aggregate.py <outdir>
set -euo pipefail

D="${1:-12}"
OUT="${2:-data/probe_d${D}}"
ORDER="${ORDER:-3}"
TOL="${TOL:-1e-4}"

mkdir -p "$OUT"
PY="python3"
SCRIPT="tests/probe_mem.py"

echo "== monolithic memory sweep: d=${D} order=${ORDER} out=${OUT} =="

# -------- BUILD-ONLY: isolate task-data memory (no IPM scratch) ------
for Z in "" "--z2-full"; do
  label=$([ -n "$Z" ] && echo "z2" || echo "noz2")
  tag="${OUT}/build_${label}.json"
  echo "-- build-only ${label}"
  $PY $SCRIPT --d $D --order $ORDER --threads 1 --build-only \
      --tol $TOL --solve-form dual $Z \
      --json "$tag" || echo "  FAIL $tag"
done

# -------- OPTIMIZE: thread sweep at fixed knobs ---------------------
for T in 1 4 8 16 32 64 128; do
  tag="${OUT}/opt_t${T}_noz2.json"
  echo "-- opt threads=$T noz2"
  $PY $SCRIPT --d $D --order $ORDER --threads $T \
      --tol $TOL --solve-form dual \
      --json "$tag" || echo "  FAIL $tag"
done

# -------- Z/2 ON, same thread sweep ---------------------------------
for T in 1 4 8 16 32 64 128; do
  tag="${OUT}/opt_t${T}_z2.json"
  echo "-- opt threads=$T z2"
  $PY $SCRIPT --d $D --order $ORDER --threads $T --z2-full \
      --tol $TOL --solve-form dual \
      --json "$tag" || echo "  FAIL $tag"
done

# -------- solve_form sensitivity at one thread count ---------------
for SF in dual primal free; do
  tag="${OUT}/opt_t32_${SF}.json"
  echo "-- opt threads=32 solve_form=$SF"
  $PY $SCRIPT --d $D --order $ORDER --threads 32 \
      --tol $TOL --solve-form $SF \
      --json "$tag" || echo "  FAIL $tag"
done

# -------- window-count sensitivity at one config ------------------
for W in 32 64 128 256 0; do
  # 0 = all windows
  wlabel=$([ $W -eq 0 ] && echo "all" || echo "$W")
  tag="${OUT}/opt_w${wlabel}.json"
  echo "-- opt windows=$wlabel"
  if [ $W -eq 0 ]; then
    $PY $SCRIPT --d $D --order $ORDER --threads 32 \
        --tol $TOL --solve-form dual \
        --json "$tag" || echo "  FAIL $tag"
  else
    $PY $SCRIPT --d $D --order $ORDER --threads 32 \
        --tol $TOL --solve-form dual --active-windows $W \
        --json "$tag" || echo "  FAIL $tag"
  fi
done

echo "== done. results in ${OUT}/"
ls -la "${OUT}/"
