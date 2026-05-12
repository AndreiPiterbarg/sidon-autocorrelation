#!/bin/bash
# Run cascade_estimate.py one config at a time; if a config exceeds the
# per-config cap, kill it, log a TIMEOUT verdict, advance to next.
set -u
C_TARGET=${1:-1.25}
TOTAL_BUDGET=${2:-1500}      # 25 min total
PER_CONFIG=${3:-360}         # 6 min per config
SAMPLE_N=${4:-15}
MAX_LEVELS=${5:-2}
LEVEL_TIME=${6:-180}
shift 6 || true
# Remaining args = configs (each like "2,30")
CONFIGS=("$@")

cd /home/ubuntu
export MOSEKLM_LICENSE_FILE=/home/ubuntu/mosek/mosek.lic

OUT=/home/ubuntu/cascade_est_run_$(date +%H%M%S)
mkdir -p "$OUT"
META="$OUT/meta.json"
SUMMARY="$OUT/summary.txt"
echo "{ \"started\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"c_target\":$C_TARGET, \"configs\":[$(IFS=,; echo "\"${CONFIGS[*]}\"")] }" > "$META"

START=$(date +%s)
echo "Driver: c=$C_TARGET total=${TOTAL_BUDGET}s per_config=${PER_CONFIG}s sample=${SAMPLE_N} max_levels=${MAX_LEVELS}" | tee "$SUMMARY"

for cfg in "${CONFIGS[@]}"; do
  NOW=$(date +%s)
  REMAIN=$((TOTAL_BUDGET - (NOW - START)))
  if [ "$REMAIN" -le 60 ]; then
    echo "$(date +%H:%M:%S) [driver] total budget exhausted, stop" | tee -a "$SUMMARY"
    break
  fi
  IFS=',' read -r N M <<< "$cfg"
  TAG="n${N}_m${M}"
  LOG="$OUT/log_$TAG.log"
  echo "" | tee -a "$SUMMARY"
  echo "$(date +%H:%M:%S) [driver] launching config ($N,$M) — budget ${PER_CONFIG}s" | tee -a "$SUMMARY"
  PYCFG="[[${N},${M}]]"

  timeout "$PER_CONFIG" python3 -u _cascade_estimate.py \
       --c_target "$C_TARGET" \
       --configs "$PYCFG" \
       --max_levels "$MAX_LEVELS" \
       --level_time_sec "$LEVEL_TIME" \
       --sample_n "$SAMPLE_N" \
       --out_dir "$OUT/data_$TAG" \
       > "$LOG" 2>&1
  RC=$?
  CFG_END=$(date +%s)
  ELAPSED=$((CFG_END - NOW))

  if [ "$RC" -eq 124 ]; then
    echo "$(date +%H:%M:%S) [driver] config ($N,$M) TIMEOUT after ${ELAPSED}s" | tee -a "$SUMMARY"
  else
    echo "$(date +%H:%M:%S) [driver] config ($N,$M) exit=$RC wall=${ELAPSED}s" | tee -a "$SUMMARY"
  fi

  # Extract verdict from per-config summary if present
  CFG_SUMMARY="$OUT/data_$TAG/summary.json"
  if [ -f "$CFG_SUMMARY" ]; then
    VERDICT=$(python3 -c "import json; d=json.load(open('$CFG_SUMMARY')); r=d['results'][-1] if d.get('results') else {}; print(r.get('verdict','UNKNOWN'))" 2>/dev/null || echo "UNREADABLE")
    echo "$(date +%H:%M:%S)   verdict: $VERDICT" | tee -a "$SUMMARY"
  fi
  # Tail final lines from the log
  tail -10 "$LOG" 2>/dev/null | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
TOTAL_END=$(date +%s)
echo "$(date +%H:%M:%S) [driver] DONE — total wall $((TOTAL_END - START))s" | tee -a "$SUMMARY"
echo "Out dir: $OUT" | tee -a "$SUMMARY"
