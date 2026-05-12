#!/bin/bash
# Launch 10 convergence probes in parallel on the pod.
# Each gets its own fresh seed (the script seeds from time_ns), its own output file,
# and its own log.
set -e
cd /root/proof
mkdir -p logs/probes

N=10
for i in $(seq 1 $N); do
  out="_conv_probe_seed${i}.json"
  log="logs/probes/probe_${i}.log"
  nohup python3 -u _convergence_probe.py \
    --input runs/d22_pod_iter5_higherK/iter_001/children_after_lp.npz \
    --d 22 --target 1.2805 \
    --split-depth 4 --subset-size 40 --max-levels 15 \
    --out "$out" > "$log" 2>&1 &
  echo "launched probe $i (PID=$!) -> $log"
  # tiny stagger so seeds differ (time_ns is sub-microsecond but be safe)
  sleep 0.05
done
echo "all 10 launched. wait with: wait"
