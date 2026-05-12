#!/bin/bash
# Production proof attempt at C&S resolution: cascade reaches d=24 (matching
# C&S 2017's published 1.28 proof setup) via d0=3 → 6 → 12 → 24.
# m=50, c_target=1.2805 (just above the 1.2802 published bound).
set -u
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=/root/proof_d24_m50_c12805_${TS}
LOG=${RUN_DIR}/run.log
DATA_DIR=${RUN_DIR}/data
mkdir -p "$DATA_DIR"

# Configuration
D0=3              # cascade reaches d=24 at L3 (3→6→12→24)
M=50              # matches C&S 2017
C_TARGET=1.2805
MAX_LEVELS=4      # L0 (d=3) → L1 (d=6) → L2 (d=12) → L3 (d=24) → L4 (d=48)
WORKERS=240       # of 360 cores

cd /root

{
  echo "================================================================"
  echo "PROOF ATTEMPT: c_target = $C_TARGET (above C&S 1.2802)"
  echo "Started: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
  echo "Host: $(hostname)"
  echo "================================================================"
  echo "Config:  d0=$D0 (cascade reaches d=24 at L3), m=$M"
  echo "         c_target=$C_TARGET, max_levels=$MAX_LEVELS, workers=$WORKERS"
  echo "Output:  $RUN_DIR"
  echo "Pruning: variant F (LP-tight Δ_BB) + skip_sdp"
  echo
  echo "----- HARDWARE -----"
  echo "CPU: $(nproc) cores"
  free -h | head -2
  uname -a
  echo
  echo "----- PYTHON ENV -----"
  python3 --version
  python3 -c "import numba, numpy, scipy; print(f'numba={numba.__version__} numpy={numpy.__version__} scipy={scipy.__version__}')"
  echo
  echo "----- CASCADE CODE INFO -----"
  wc -l cloninger-steinerberger/cpu/run_cascade.py
  md5sum cloninger-steinerberger/cpu/run_cascade.py
  echo
  echo "----- BEGIN CASCADE -----"
  echo
} > "$LOG" 2>&1

{
  python3 -u cloninger-steinerberger/cpu/run_cascade.py \
      --d0 "$D0" \
      --n_half 1 \
      --m "$M" \
      --c_target "$C_TARGET" \
      --max_levels "$MAX_LEVELS" \
      --workers "$WORKERS" \
      --use_F \
      --skip_sdp \
      --output_dir "$DATA_DIR"
  EXIT=$?
  echo
  echo "================================================================"
  echo "FINISHED: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
  echo "Exit code: $EXIT"
  echo "================================================================"
  if [ "$EXIT" -eq 0 ]; then
    echo "SUCCESS - cascade completed normally."
  else
    echo "FAILURE - exit code $EXIT.  Last lines above explain why."
  fi
  echo
  echo "Checkpoints saved in: $DATA_DIR"
  ls -la "$DATA_DIR/" 2>&1
  echo
  echo "Memory at end: $(free -h | head -2 | tail -1)"
  echo "Run dir: $RUN_DIR"
} >> "$LOG" 2>&1

echo "DONE.  Log: $LOG"
