# archive/scripts/

Utility / driver scripts (mostly Farkas-Lasserre and pod orchestration)
that supported the val(d) push attempts. Kept for archaeology; not part
of any current proof pipeline.

## Files

### Farkas-Lasserre val(d) push

- `farkas_bisect_sweep.py` — bisect Farkas infeasibility certification across
                             $(d, \text{order}, t)$ to find the tightest certifiable bound.
- `farkas_push_sweep.py`   — aggressive sweep targeted at val(d) > 1.28
                             (nthreads=64, max_denom $10^{14}$, eig_margin $10^{-10}$).
- `run_certify_sweep.py`   — certified Lasserre sweep over $(d, \text{order})$,
                             reports SDP sizes and rational bounds.
- `summarize_farkas.py`    — print summary table of all certified Farkas bounds
                             from `data/farkas_results/`.
- `push_1_28.sh`           — bash driver: try successively smaller $t$ at a
                             single $d$ until certification succeeds.

### Pod orchestration

- `pod_full_sweep.sh`  — full autonomous Farkas sweep ($d=4..16$) on the pod;
                         output to `data/farkas_results/sweep.log`.
- `pod_debug.py`       — minimal paramiko connectivity check.
- `pod_log_watcher.py` — persistent local backup: pulls pod log every poll,
                         writes heartbeat + timestamped snapshots.
