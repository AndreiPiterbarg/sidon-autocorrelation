#!/bin/bash
set -e
cd /root/sidon
TS=$(date -u +%Y%m%d_%H%M%S)
LOG=/root/sidon/coarse_estimator_${TS}.log
echo "=== START $(date -u +%Y-%m-%dT%H:%M:%SZ) c_target=1.281 ===" | tee -a $LOG
echo "Host: $(hostname)  Cores: $(nproc)" | tee -a $LOG
echo "===" | tee -a $LOG

# c_target=1.281, sample_n=10, max_levels=8 (run-to-convergence).
# Layers: NO + Joint dual K=4 + Shor SDP best_only.
# parent_time_sec=120 (slow-flag threshold; level_time=20min).
python3 -u _coarse_cascade_estimate.py \
    --c_target 1.281 \
    --sample_n 10 \
    --max_levels 8 \
    --joint_top_K 4 \
    --joint_iters 20 \
    --sdp_mode best_only \
    --use_joint --no_sdp \
    --level_time_sec 1200 \
    --n_workers 64 \
    --configs '[[2,30],[2,60],[2,100],[4,20],[4,30]]' \
    --out_dir /root/sidon/coarse_estimate_${TS} \
    2>&1 | grep -v 'ortools\|PDLP\|GLOP\|RuntimeError.*ortools\|Unrecognized.*ortools\|CVXPY.*ortools' \
    | tee -a $LOG

echo "=== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a $LOG
echo "Log: $LOG"
