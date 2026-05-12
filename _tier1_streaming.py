"""Streaming Tier-1 test: run NEW config first, OLD second, with live MOSEK
verbose output so we see iteration timings as they happen.

Cases prioritized: d=16 R=12 (reference), then R=14, R=16 if R=12 NEW is fast.
Each (d, R, config) pair is its own subprocess so a hang doesn't kill the others.
"""
import sys, os, time, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


PROBE = """
import sys, time, json, resource, io
sys.path.insert(0, '/root/sidon')
import numpy as np
from scipy import sparse as sp
import mosek
from lasserre.polya_lp.build import (
    build_handelman_lp, BuildOptions, build_window_matrices,
)
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim

d = int(sys.argv[1]); R = int(sys.argv[2]); cfg = sys.argv[3]

print(f'BUILD d={d} R={R} cfg={cfg}', flush=True)
t0 = time.time()
_, M_mats = build_window_matrices(d)
M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats, d)
d_eff = z2_dim(d)
opts = BuildOptions(R=R, use_z2=True, eliminate_c_slacks=False)
build = build_handelman_lp(d_eff, M_mats_eff, opts)
n_rows = (build.A_eq.shape[0] if build.A_eq is not None else 0) + \\
         (build.A_ub.shape[0] if build.A_ub is not None else 0)
print(f'BUILT rows={n_rows} vars={build.n_vars} nnz={build.n_nonzero_A} t={time.time()-t0:.1f}s', flush=True)

# Set up MOSEK with the requested config.
A_eq = build.A_eq
b_eq = build.b_eq
n_vars = build.n_vars
A_combined = A_eq
n_rows = A_combined.shape[0]
A_coo = A_combined.tocoo()

t_solve_start = time.time()
with mosek.Env() as env, env.Task() as task:
    task.set_Stream(mosek.streamtype.log,
                    lambda msg: print(msg, end='', flush=True))
    task.appendvars(n_vars)
    task.appendcons(n_rows)
    for j, (lo, hi) in enumerate(build.bounds):
        if lo is None and hi is None:
            bk = mosek.boundkey.fr; lb = ub = 0.0
        elif lo is None:
            bk = mosek.boundkey.up; lb = 0.0; ub = float(hi)
        elif hi is None:
            bk = mosek.boundkey.lo; lb = float(lo); ub = 0.0
        elif lo == hi:
            bk = mosek.boundkey.fx; lb = ub = float(lo)
        else:
            bk = mosek.boundkey.ra; lb = float(lo); ub = float(hi)
        task.putvarbound(j, bk, lb, ub)
        task.putcj(j, float(build.c[j]))
    for i in range(n_rows):
        task.putconbound(i, mosek.boundkey.fx, float(b_eq[i]), float(b_eq[i]))
    task.putaijlist(
        A_coo.row.astype(np.int64).tolist(),
        A_coo.col.astype(np.int64).tolist(),
        A_coo.data.astype(np.float64).tolist(),
    )
    task.putobjsense(mosek.objsense.minimize)

    # Common
    task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.intpnt)
    task.putintparam(mosek.iparam.intpnt_solve_form, mosek.solveform.dual)
    task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.never)
    task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.on)
    task.putintparam(mosek.iparam.presolve_lindep_use, mosek.onoffkey.off)
    task.putintparam(mosek.iparam.num_threads, 0)
    task.putdouparam(mosek.dparam.intpnt_tol_rel_gap, 1e-9)
    task.putdouparam(mosek.dparam.intpnt_tol_pfeas, 1e-9)
    task.putdouparam(mosek.dparam.intpnt_tol_dfeas, 1e-9)

    if cfg == 'OLD':
        task.putintparam(mosek.iparam.presolve_eliminator_max_num_tries, -1)
        task.putintparam(mosek.iparam.presolve_eliminator_max_fill, 20)
    elif cfg == 'NEW':
        task.putintparam(mosek.iparam.presolve_eliminator_max_num_tries, 1)
        task.putintparam(mosek.iparam.presolve_eliminator_max_fill, 5)
        task.putintparam(mosek.iparam.intpnt_order_method,
                         mosek.orderingtype.force_graphpar)
    elif cfg == 'NO_ELIM':
        task.putintparam(mosek.iparam.presolve_eliminator_max_num_tries, 0)
    elif cfg == 'JUST_ORDER':
        task.putintparam(mosek.iparam.presolve_eliminator_max_num_tries, -1)
        task.putintparam(mosek.iparam.presolve_eliminator_max_fill, 20)
        task.putintparam(mosek.iparam.intpnt_order_method,
                         mosek.orderingtype.force_graphpar)

    task.optimize()
    t_solve = time.time() - t_solve_start
    sol_status = task.getsolsta(mosek.soltype.itr)
    if sol_status == mosek.solsta.optimal:
        alpha = -float(task.getprimalobj(mosek.soltype.itr))
    else:
        alpha = None

rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
print(f'\\nFINAL alpha={alpha} status={sol_status} t_solve={t_solve:.1f}s '
      f'rss={rss_mb:.0f}MB', flush=True)
out = dict(d=d, R=R, cfg=cfg, alpha=alpha, status=str(sol_status),
           t_solve=t_solve, rss_mb=rss_mb, n_rows=int(n_rows),
           n_vars=int(n_vars), nnz=int(build.n_nonzero_A))
print('RESULT:', json.dumps(out, default=str), flush=True)
"""

PROBE_PATH = '_tier1_probe.py'
with open(PROBE_PATH, 'w') as f:
    f.write(PROBE)


# Run NEW first at small R to confirm working, then expand.
SCHEDULE = [
    (16, 12, 'NEW'),       # quick check NEW config works
    (16, 12, 'OLD'),       # baseline
    (16, 12, 'NO_ELIM'),   # eliminator off entirely
    (16, 12, 'JUST_ORDER'),# isolate ordering benefit
    (16, 14, 'NEW'),       # bigger
    (16, 14, 'OLD'),       # bigger baseline
    (16, 16, 'NEW'),       # even bigger
]

# 1 hour cap per probe -- if a single config takes longer it's broken.
PER_TASK_TIMEOUT = 3600

results = []
for d, R, cfg in SCHEDULE:
    print(f"\n========== d={d} R={R} cfg={cfg} ==========", flush=True)
    t_start = time.time()
    try:
        proc = subprocess.run(
            ['python3', '-u', PROBE_PATH, str(d), str(R), cfg],
            capture_output=True, text=True, timeout=PER_TASK_TIMEOUT,
        )
        wall = time.time() - t_start
        # Print captured stdout (verbose MOSEK log)
        for line in proc.stdout.splitlines():
            print(line, flush=True)
        # Find RESULT
        result = None
        for line in proc.stdout.splitlines():
            if line.startswith('RESULT:'):
                result = json.loads(line[len('RESULT:'):].strip())
        if result is None:
            err = proc.stderr[-1500:] if proc.stderr else ''
            print(f'  NO RESULT after {wall:.0f}s. stderr: {err}', flush=True)
            results.append(dict(d=d, R=R, cfg=cfg, error='no_result',
                                 wall=wall, stderr=err))
        else:
            results.append(result)
    except subprocess.TimeoutExpired:
        wall = time.time() - t_start
        print(f'  TIMEOUT after {wall:.0f}s', flush=True)
        results.append(dict(d=d, R=R, cfg=cfg, error='timeout', wall=wall))

    with open('tier1_streaming_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


print('\n\n=== SUMMARY ===\n', flush=True)
print(f"{'case':>10} {'cfg':>12} {'alpha':>10} {'wall':>8} {'rss_GB':>8}", flush=True)
print('-' * 60, flush=True)
for r in results:
    case = f"d{r['d']}R{r['R']}"
    cfg = r['cfg']
    if 'error' in r:
        print(f"{case:>10} {cfg:>12} ERROR: {r['error']}", flush=True)
        continue
    a = r.get('alpha')
    a_str = f"{a:.6f}" if a is not None else "N/A"
    print(f"{case:>10} {cfg:>12} {a_str:>10} {r['t_solve']:>8.1f} "
          f"{r['rss_mb']/1024.0:>8.2f}", flush=True)
