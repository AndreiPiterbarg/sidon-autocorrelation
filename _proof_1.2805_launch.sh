#!/bin/bash
# Production cascade proof attempt at c_target = 1.2805 (just above the
# published 1.2802 bound).  Optimal config: variant F + skip_sdp + max
# workers + 10-level cascade.  Comprehensive logging to a single file.
set -u
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=/root/proof_1.2805_${TS}
LOG=${RUN_DIR}/run.log
DATA_DIR=${RUN_DIR}/data
mkdir -p "$DATA_DIR"

# Configuration knobs
N_HALF=2          # initial dim d=4; cascade refines to d=8, 16, 32, 64...
M=20              # quantization (compositions sum to 4nm at L0)
C_TARGET=1.2805   # target lower bound on C_{1a}
MAX_LEVELS=10
WORKERS=240       # of 360 cores; leaves 120 for JIT + system + memory bandwidth

cd /root

# Capture environment snapshot for forensics
{
  echo "================================================================"
  echo "PROOF ATTEMPT: c_target = $C_TARGET"
  echo "Started: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
  echo "Host: $(hostname)"
  echo "================================================================"
  echo "Config:  n_half=$N_HALF, m=$M, c_target=$C_TARGET, max_levels=$MAX_LEVELS, workers=$WORKERS"
  echo "Output:  $RUN_DIR"
  echo "Pruning: variant F (LP-tight Δ_BB + standard δ²) + skip_sdp"
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

# Launch the cascade.  -u flag = unbuffered stdout (line-by-line flushes).
# Use 2>&1 to merge stderr into the same log.
# trap completion to capture exit status + final summary.
{
  python3 -u cloninger-steinerberger/cpu/run_cascade.py \
      --n_half "$N_HALF" \
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
