"""Sweep diagnostic across d in {10, 12, 14, 16, 18, 20}.
Runs single-shot dual-Farkas with verbose log captured into separate file
per d, then prints a compact summary table to stdout.
"""
import os, sys, time, gc, traceback, json
sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)
import numpy as np

import mosek

from interval_bnb.windows import build_windows
from interval_bnb.bound_sdp_escalation import _build_dual_task_box
from lasserre.dual_sdp import solve_dual_task
from lasserre.precompute import _precompute

results = []
ds = [int(x) for x in os.environ.get('SWEEP_DS', '10,12,14,16,18,20').split(',')]
target = float(os.environ.get('TB_T', '1.281'))
n_thr = int(os.environ.get('TB_THREADS', '48'))
time_lim = float(os.environ.get('TB_TIME', '900'))
tol = float(os.environ.get('TB_TOL', '1e-7'))

for d in ds:
    print(f"\n========== d={d} ==========", flush=True)
    log_path = f"/tmp/diag_sweep_d{d}.log"
    log_f = open(log_path, 'w', buffering=1)
    def _stream(s, f=log_f):
        f.write(s)

    t0 = time.time()
    windows = build_windows(d)
    mu = np.full(d, 1.0/d); mu[0] *= 0.5; mu[-1] *= 1.5; mu /= mu.sum()
    if mu[0] > mu[-1]:
        mu = mu[::-1]
    radius = 5e-3
    lo = np.maximum(mu - radius, 0.0)
    hi = np.minimum(mu + radius, 1.0)

    P = _precompute(d, order=2, verbose=False, lazy_ab_eiej=True)
    print(f"  precompute={time.time()-t0:.2f}s "
          f"n_basis={P['n_basis']} n_loc={P['n_loc']} n_y={P['n_y']}",
          flush=True)
    env = mosek.Env()
    t1 = time.time()
    task, info = _build_dual_task_box(P, lo, hi, t_val=target, env=env,
                                        verbose=False)
    build_t = time.time() - t1
    print(f"  build={build_t:.2f}s n_bar_entries={info['n_bar_entries']:,}",
          flush=True)

    task.set_Stream(mosek.streamtype.log, _stream)
    task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas, tol)
    task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, tol)
    task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, tol)
    task.putdouparam(mosek.dparam.optimizer_max_time, time_lim)
    task.putintparam(mosek.iparam.num_threads, n_thr)

    ts = time.time()
    try:
        verdict = solve_dual_task(task, info, verbose=False)
        ok = True
    except Exception as e:
        verdict = {'verdict': f'EXCEPTION:{type(e).__name__}',
                   'lambda_star': float('nan')}
        ok = False
    wall = time.time() - ts

    n_iter = -1; fac_flops = float('nan'); fac_nz = -1
    try:
        n_iter = task.getintinf(mosek.iinfitem.intpnt_iter)
        fac_flops = task.getdouinf(mosek.dinfitem.intpnt_factor_num_flops)
        fac_nz = task.getintinf(mosek.iinfitem.intpnt_factor_num_nz)
    except Exception:
        pass

    log_f.close()
    rec = {
        'd': d, 'n_basis': P['n_basis'], 'n_y': P['n_y'],
        'n_bar_entries': info['n_bar_entries'],
        'build_s': build_t, 'solve_s': wall,
        'iters': n_iter, 'fac_flops': fac_flops, 'fac_nz': fac_nz,
        'per_iter_s': wall / max(n_iter, 1),
        'verdict': verdict.get('verdict'),
        'lambda_star': verdict.get('lambda_star'),
    }
    results.append(rec)
    print(f"  d={d} solve={wall:.2f}s iters={n_iter} "
          f"per_iter={rec['per_iter_s']:.2f}s "
          f"verdict={rec['verdict']} fac_nz={fac_nz} "
          f"fac_flops={fac_flops:.2e}", flush=True)
    del task, info, P
    gc.collect()

print("\n========== SUMMARY ==========", flush=True)
print(json.dumps(results, indent=2, default=str), flush=True)
print(f"{'d':>4} {'n_y':>6} {'n_bar_e':>10} {'build':>7} {'solve':>8} "
      f"{'iters':>6} {'per_it':>7} {'flops':>10} {'verdict':>10}", flush=True)
for r in results:
    print(f"{r['d']:>4} {r['n_y']:>6} {r['n_bar_entries']:>10,} "
          f"{r['build_s']:>7.2f} {r['solve_s']:>8.2f} {r['iters']:>6} "
          f"{r['per_iter_s']:>7.2f} {r['fac_flops']:>10.2e} "
          f"{str(r['verdict']):>10}", flush=True)
