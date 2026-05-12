#!/bin/bash
# Run _Q_bench.py at c=1.25 on the four bench-proven configs.
# Each config does FULL L0 enumeration (no sampling) and reports F+Q L0 survivors.
# This gives a HARD answer to "does L0 close at c=1.25" per (n,m).
set -u
cd /home/ubuntu
LOG=/home/ubuntu/qsweep_125_$(date +%H%M%S).log
> "$LOG"

# (n_half, m, expected_d) — same as bench at c=1.28
configs=(
  "3 10"   # d=6
  "4 10"   # d=8
  "5  5"   # d=10
  "6  5"   # d=12
)

for cfg in "${configs[@]}"; do
  read -r N M <<< "$cfg"
  D=$((2*N))
  echo "" | tee -a "$LOG"
  echo "##### (n=$N, m=$M, d=$D, c=1.25) #####" | tee -a "$LOG"
  T0=$(date +%s)
  timeout 1800 python3 -u _Q_bench.py --n_half "$N" --m "$M" --c_target 1.25 \
      --out /home/ubuntu/qres_n${N}m${M}_c125.json 2>&1 \
    | grep -vE 'FutureWarning|warnings\.warn' \
    | tee -a "$LOG"
  T1=$(date +%s)
  echo "  [wall: $((T1-T0))s]" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "DONE: $(date -u)" | tee -a "$LOG"
