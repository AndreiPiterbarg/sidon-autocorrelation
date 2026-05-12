#!/bin/bash
# Run benchmark_sweep at c=1.28, n=2, m∈{10,20,30,40} and stream output
# to a single live logfile.  Use python -u for unbuffered stdout.
set -u
LOG=/root/sweep_F_$(date +%H%M%S).log
echo "Streaming to $LOG (tail -f to watch)"
> "$LOG"
for mm in 10 20 30 40; do
  echo "" | tee -a "$LOG"
  echo "######################################################################" | tee -a "$LOG"
  echo "## n=2 m=$mm c=1.28 use_F skip_sdp (default) ##" | tee -a "$LOG"
  echo "######################################################################" | tee -a "$LOG"
  python3 -u tests/benchmark_sweep.py --m "$mm" --n_half 2 \
      --c_target 1.28 --sample 30 --use_F --level_time_sec 600 \
      2>&1 \
    | grep -vE '(FutureWarning|warnings\.warn|reshape|problem.py|Encountered|Runtime|specify|To suppress|This default|order)' \
    | tee -a "$LOG"
done
echo "" | tee -a "$LOG"
echo "ALL DONE, log at $LOG" | tee -a "$LOG"
