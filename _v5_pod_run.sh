#!/bin/bash
set -e
cd /root/sidon
TS=$(date -u +%Y%m%d_%H%M%S)
LOG=/root/sidon/v5_${TS}.log
OUT=/root/sidon/v5_run_${TS}
echo "=== START $(date -u +%Y-%m-%dT%H:%M:%SZ) c_target=1.281 v5 tight ===" | tee -a $LOG
echo "Host: $(hostname)  Cores: $(nproc)  Output: $OUT" | tee -a $LOG
echo "===" | tee -a $LOG

python3 -u cloninger-steinerberger/cpu/run_cascade_coarse_v5.py \
    --d0 2 \
    --S 60 \
    --c_target 1.281 \
    --mode adaptive \
    --tight_max_level 2 \
    --use_joint --no_sdp \
    --joint_top_K 4 --joint_iters 20 \
    --max_levels 8 \
    --n_workers 180 \
    --numba_threads 1 \
    --output_dir $OUT \
    >> $LOG 2>&1

echo "=== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a $LOG
