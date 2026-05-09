#!/bin/bash
# Pod runner: v2 vs v3 vs v4 cascade head-to-head with full timestamped logging.
set -e
cd /root/sidon
LOG=/root/sidon/v2v3v4_$(date -u +%Y%m%d_%H%M%S).log
echo "=== START $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a $LOG
echo "Host: $(hostname)" | tee -a $LOG
echo "Cores: $(nproc)" | tee -a $LOG
echo "RAM:   $(free -h | grep Mem)" | tee -a $LOG
echo "===" | tee -a $LOG

# Run with -u (unbuffered Python) and tee to log; filter cvxpy noise.
python3 -u _v2_v3_v4_bench.py 2>&1 \
  | grep -v 'ortools\|PDLP\|GLOP\|RuntimeError.*ortools\|Unrecognized.*ortools\|CVXPY.*ortools' \
  | tee -a $LOG

echo "=== DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a $LOG
echo "Log: $LOG"
