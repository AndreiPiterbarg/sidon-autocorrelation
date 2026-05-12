#!/bin/bash
# Tightest pruning sweep at c=1.25: F + Q + L on the bench-proven configs.
# _L_bench.py runs F first, then Q on F-survivors, then L (SDP) on Q-survivors.
# Order from cheapest config to costliest.
set -u
cd /home/ubuntu
LOG=/home/ubuntu/fql_sweep_125_$(date +%H%M%S).log
> "$LOG"

# (n_half, m, timeout_sec) - in order of cheapest first
configs=(
  "3 10  600"   # d=6,  L0=1891 comps (sub-second)
  "4 10  900"   # d=8,  L0=91K comps  (5s for Q)
  "5  5 1500"   # d=10, L0=316K comps (Q+L = 6 min with MOSEK; ~30 min Clarabel)
  "6  5 1500"   # d=12, L0=8.26M comps; bench: F=7 Q=0 → expect closure
)

for cfg in "${configs[@]}"; do
  read -r N M TO <<< "$cfg"
  D=$((2*N))
  echo "" | tee -a "$LOG"
  echo "##### F+Q+L at (n=$N, m=$M, d=$D, c=1.25, solver=CLARABEL) #####" | tee -a "$LOG"
  T0=$(date +%s)
  timeout "$TO" python3 -u _L_bench.py --n_half "$N" --m "$M" --c_target 1.25 \
      --solver CLARABEL --order 1 \
      --out /home/ubuntu/lres_n${N}m${M}_c125.json 2>&1 \
    | grep -vE 'FutureWarning|warnings\.warn|UserWarning' \
    | tee -a "$LOG"
  T1=$(date +%s)
  echo "  [wall: $((T1-T0))s]" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "DONE: $(date -u)" | tee -a "$LOG"
