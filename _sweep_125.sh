#!/bin/bash
# Representative sweep at c=1.25, --use_F, tight budgets per level.
# Each config naturally probes d0, 2*d0, 4*d0, ... via cascade levels.
set -u
cd /home/ubuntu
LOG=/home/ubuntu/sweep_125_$(date +%H%M%S).log
> "$LOG"
echo "Sweep at c=1.25 --use_F" | tee -a "$LOG"
echo "Start: $(date -u)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Configs: (n_half, m, level_budget_sec, sample_size)
# n_half=2 -> d0=4: cheap L0, cascade to d=8,16,32,...
# n_half=3 -> d0=6: medium L0 (234M for m=10), cascade to d=12,24
configs=(
  "2 10  90 30"
  "2 15  90 30"
  "2 20  90 30"
  "3 10 240 30"
  "3 15 240 20"
  "4 10 600 20"
)

for cfg in "${configs[@]}"; do
  read -r N M BUDGET SAMP <<< "$cfg"
  D0=$((2*N))
  echo "" | tee -a "$LOG"
  echo "######################################################################" | tee -a "$LOG"
  echo "## n_half=$N (d0=$D0), m=$M, c=1.25, sample=$SAMP, level_budget=${BUDGET}s ##" | tee -a "$LOG"
  echo "######################################################################" | tee -a "$LOG"
  T0=$(date +%s)
  timeout 1200 python3 -u tests/benchmark_sweep.py \
      --n_half "$N" --m "$M" --c_target 1.25 \
      --sample "$SAMP" --use_F --level_time_sec "$BUDGET" \
    2>&1 \
    | grep -vE 'FutureWarning|warnings\.warn|reshape|problem.py|Encountered|specify|To suppress|This default|order' \
    | tee -a "$LOG"
  T1=$(date +%s)
  echo "  [config wall: $((T1-T0))s]" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "DONE: $(date -u)" | tee -a "$LOG"
