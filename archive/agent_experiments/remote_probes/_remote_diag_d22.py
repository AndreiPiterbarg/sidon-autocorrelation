"""Diagnostic harness for the d=22 dual-Farkas SDP.

Runs the dual-Farkas LP with MOSEK verbose log enabled so we can see:
  - per-iteration time
  - primal/dual residual & gap
  - any warnings (ill-conditioning, scaling, etc.)
  - factor flop count, off-diagonal nz, factor dimension

Usage:
  TB_D=22 TB_T=1.281 TB_TIME=900 TB_THREADS=48 \
    TB_EARLY_STOP=1  python3 _remote_diag_d22.py
"""
import os, sys, time, gc, traceback
sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)
import numpy as np

import mosek

from interval_bnb.windows import build_windows
from interval_bnb.bound_sdp_escalation import (
    build_sdp_escalation_cache, _build_dual_task_box, update_task_box,
)
from lasserre.dual_sdp import solve_dual_task

d = int(os.environ.get('TB_D', '22'))
target = float(os.environ.get('TB_T', '1.281'))
time_lim = float(os.environ.get('TB_TIME', '900'))
n_thr = int(os.environ.get('TB_THREADS', '48'))
early_stop = bool(int(os.environ.get('TB_EARLY_STOP', '0')))
tol = float(os.environ.get('TB_TOL', '1e-7'))

print(f"=== d={d} target={target} time={time_lim}s threads={n_thr} "
      f"early_stop={early_stop} tol={tol} ===", flush=True)

t0 = time.time()
windows = build_windows(d)
print(f"  built windows in {time.time()-t0:.2f}s ({len(windows)} windows)",
      flush=True)

# Use a fresh box around what looks like a feasible μ. We use the same
# initialization as _remote_thread_bench.py — μ near the simplex centroid
# with a small skew + radius=5e-3 hyperbox.
mu = np.full(d, 1.0/d); mu[0] *= 0.5; mu[-1] *= 1.5; mu /= mu.sum()
if mu[0] > mu[-1]:
    mu = mu[::-1]
radius = 5e-3
lo = np.maximum(mu - radius, 0.0)
hi = np.minimum(mu + radius, 1.0)
print(f"  box: lo[0]={lo[0]:.4f} hi[0]={hi[0]:.4f} sum_lo={lo.sum():.4f} "
      f"sum_hi={hi.sum():.4f}", flush=True)

# Build the task with verbose=True so MOSEK prints per-iter log.
t1 = time.time()
from lasserre.precompute import _precompute
P = _precompute(d, order=2, verbose=True, lazy_ab_eiej=True)
print(f"  precompute: {time.time()-t1:.2f}s", flush=True)
print(f"  n_basis={P['n_basis']} n_loc={P['n_loc']} n_y={P['n_y']}",
      flush=True)

env = mosek.Env()
t2 = time.time()
task, info = _build_dual_task_box(P, lo, hi, t_val=target, env=env,
                                    verbose=True)
print(f"  task build: {time.time()-t2:.2f}s  "
      f"n_bar={info['n_bar']} n_scalar={info['n_scalar']} "
      f"n_cons={info['n_cons']} n_bar_entries={info['n_bar_entries']}",
      flush=True)

# Set tolerances + threads + time limit
task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas, tol)
task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, tol)
task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, tol)
task.putdouparam(mosek.dparam.optimizer_max_time, time_lim)
task.putintparam(mosek.iparam.num_threads, n_thr)

# Make sure the verbose stream is set even if rebuilt
task.set_Stream(mosek.streamtype.log, lambda s: print(s, end='', flush=True))

# Pull out problem-info during/after optimize
print("=== STARTING task.optimize() ===", flush=True)
ts = time.time()
try:
    if early_stop:
        verdict = solve_dual_task(task, info, verbose=True,
                                    early_stop_on_clear_verdict=True,
                                    early_stop_feas_frac=0.15,
                                    early_stop_infeas_frac=0.85,
                                    early_stop_gap_tol=1e-2)
    else:
        verdict = solve_dual_task(task, info, verbose=True)
except Exception as e:
    print(f"!!! solve threw: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    verdict = None
wall = time.time() - ts

print(f"=== DONE: solve={wall:.2f}s ===", flush=True)
if verdict is not None:
    print(f"verdict={verdict.get('verdict')}  "
          f"lambda*={verdict.get('lambda_star')}", flush=True)
    print(f"status: {verdict.get('status')}", flush=True)
    if 'early_stop' in verdict:
        print(f"early_stop: {verdict['early_stop']}", flush=True)

# Pull out int/double info items for IPM stats
try:
    n_iter = task.getintinf(mosek.iinfitem.intpnt_iter)
    fac_dim = task.getintinf(mosek.iinfitem.intpnt_factor_dim)
    fac_nz = task.getintinf(mosek.iinfitem.intpnt_factor_num_nz)
    print(f"IPM iters: {n_iter}  factor_dim={fac_dim}  factor_nz={fac_nz}",
          flush=True)
    fac_flops = task.getdouinf(mosek.dinfitem.intpnt_factor_num_flops)
    print(f"factor_flops: {fac_flops:.3e}", flush=True)
    pres = task.getdouinf(mosek.dinfitem.intpnt_primal_feas)
    dres = task.getdouinf(mosek.dinfitem.intpnt_dual_feas)
    pobj = task.getdouinf(mosek.dinfitem.intpnt_primal_obj)
    dobj = task.getdouinf(mosek.dinfitem.intpnt_dual_obj)
    print(f"primal_feas={pres:.3e}  dual_feas={dres:.3e}", flush=True)
    print(f"primal_obj={pobj:.6e}  dual_obj={dobj:.6e}  "
          f"gap={abs(pobj-dobj):.3e}", flush=True)
except Exception as e:
    print(f"info-item read failed: {e}", flush=True)

# Memory check
import resource
ru = resource.getrusage(resource.RUSAGE_SELF)
print(f"max RSS: {ru.ru_maxrss/1024:.1f} MB", flush=True)
