"""Compare OLD vs NEW MOSEK config on d=16 R=12 (and bigger) to measure
the Tier 1 speedup from:
  - presolve_eliminator_max_num_tries: -1 (cascade) -> 1 (one pass)
  - presolve_eliminator_max_fill:      20 -> 5
  - intpnt_order_method:               default -> force_graphpar (METIS)

Measured per case:
  - alpha (must agree between old & new)
  - factor nonzeros (from MOSEK log: "Factor - nonzeros after factor")
  - factor flops (from MOSEK log)
  - wall_solve seconds
  - RSS peak

Schedule:
  d=16 R=12  : reference -- known from prior run alpha=1.2362, 660s, 7.7GB
  d=16 R=15  : 4M rows, ~30 GB RAM -- stretch case
  d=16 R=18  : 12M rows, ~150 GB RAM -- big case (only on big pod)

We run the SAME LP twice -- once with OLD MOSEK params, once with NEW --
and report deltas. The build is shared (so the LP itself is identical).
"""
import sys, os, time, json, resource, re, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mosek
from scipy import sparse as sp

from lasserre.polya_lp.build import (
    build_handelman_lp, BuildOptions, build_window_matrices,
)
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim


def _solve_with_config(build, params, capture_log=True):
    """Solve the LP with explicit MOSEK params. Returns (alpha, status, wall,
    rss_mb, log_text)."""
    A_eq = build.A_eq
    b_eq = build.b_eq
    n_vars = build.n_vars
    has_ub = (build.A_ub is not None and build.A_ub.shape[0] > 0)
    if has_ub:
        A_combined = sp.vstack([build.A_ub, A_eq], format="csr")
        n_ub = build.A_ub.shape[0]
        n_eq_only = A_eq.shape[0]
    else:
        A_combined = A_eq
        n_ub = 0
        n_eq_only = A_eq.shape[0]
    n_rows = A_combined.shape[0]

    log_buf = io.StringIO()

    t0 = time.time()
    with mosek.Env() as env, env.Task() as task:
        if capture_log:
            task.set_Stream(mosek.streamtype.log,
                            lambda msg: log_buf.write(msg))
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

        for i in range(n_ub):
            task.putconbound(i, mosek.boundkey.up, 0.0, float(build.b_ub[i]))
        for i in range(n_eq_only):
            task.putconbound(n_ub + i, mosek.boundkey.fx,
                             float(b_eq[i]), float(b_eq[i]))

        A_coo = A_combined.tocoo()
        task.putaijlist(
            A_coo.row.astype(np.int64).tolist(),
            A_coo.col.astype(np.int64).tolist(),
            A_coo.data.astype(np.float64).tolist(),
        )
        task.putobjsense(mosek.objsense.minimize)

        # Apply requested params.
        for (kind, name, val) in params:
            if kind == "i":
                task.putintparam(name, val)
            else:
                task.putdouparam(name, val)

        task.optimize()
        wall = time.time() - t0
        sol_status = task.getsolsta(mosek.soltype.itr)
        if sol_status == mosek.solsta.optimal:
            alpha = -float(task.getprimalobj(mosek.soltype.itr))
        else:
            alpha = None

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return alpha, sol_status, wall, rss_mb, log_buf.getvalue()


def _parse_factor_stats(log):
    """Pull "Factor - nonzeros after factor" and "flops" out of MOSEK log."""
    out = {}
    m = re.search(r'nonzeros before factor\s*:\s*([\d.e+-]+)', log)
    if m: out['nz_before'] = float(m.group(1))
    m = re.search(r'after factor\s*:\s*([\d.e+-]+)', log)
    if m: out['nz_after'] = float(m.group(1))
    m = re.search(r'flops\s*:\s*([\d.e+-]+)', log)
    if m: out['flops'] = float(m.group(1))
    m = re.search(r'GP order time\s*:\s*([\d.e+-]+)', log)
    if m: out['gp_order_s'] = float(m.group(1))
    m = re.search(r'Factor\s+-\s+setup time\s*:\s*([\d.e+-]+)', log)
    if m: out['setup_s'] = float(m.group(1))
    # Solved problem: primal/dual?
    m = re.search(r'solved problem\s+:\s+the\s+(\w+)', log)
    if m: out['solved'] = m.group(1)
    # Iterations: count lines starting with digit
    iters = re.findall(r'^\s*(\d+)\s+[\d.e+-]+\s+[\d.e+-]+', log, re.MULTILINE)
    if iters:
        out['n_iter'] = int(iters[-1])
    return out


# Common params (always set)
COMMON = [
    ("i", mosek.iparam.optimizer, mosek.optimizertype.intpnt),
    ("i", mosek.iparam.intpnt_solve_form, mosek.solveform.dual),
    ("i", mosek.iparam.intpnt_basis, mosek.basindtype.never),
    ("i", mosek.iparam.presolve_use, mosek.presolvemode.on),
    ("i", mosek.iparam.presolve_lindep_use, mosek.onoffkey.off),
    ("i", mosek.iparam.num_threads, 0),
    ("d", mosek.dparam.intpnt_tol_rel_gap, 1e-9),
    ("d", mosek.dparam.intpnt_tol_pfeas, 1e-9),
    ("d", mosek.dparam.intpnt_tol_dfeas, 1e-9),
]

# OLD config (cascading eliminator + default ordering)
OLD = COMMON + [
    ("i", mosek.iparam.presolve_eliminator_max_num_tries, -1),
    ("i", mosek.iparam.presolve_eliminator_max_fill, 20),
    # default ordering
]

# NEW config (limited eliminator + force_graphpar)
NEW = COMMON + [
    ("i", mosek.iparam.presolve_eliminator_max_num_tries, 1),
    ("i", mosek.iparam.presolve_eliminator_max_fill, 5),
    ("i", mosek.iparam.intpnt_order_method, mosek.orderingtype.force_graphpar),
]

# Variant: just the eliminator change (isolate effect)
JUST_ELIM = COMMON + [
    ("i", mosek.iparam.presolve_eliminator_max_num_tries, 1),
    ("i", mosek.iparam.presolve_eliminator_max_fill, 5),
]

# Variant: just the ordering change
JUST_ORDER = COMMON + [
    ("i", mosek.iparam.presolve_eliminator_max_num_tries, -1),
    ("i", mosek.iparam.presolve_eliminator_max_fill, 20),
    ("i", mosek.iparam.intpnt_order_method, mosek.orderingtype.force_graphpar),
]

# Also: no eliminator at all (radical)
NO_ELIM = COMMON + [
    ("i", mosek.iparam.presolve_eliminator_max_num_tries, 0),
]


CONFIGS = [
    ("OLD",         OLD),
    ("NEW",         NEW),
    ("JUST_ELIM",   JUST_ELIM),
    ("JUST_ORDER",  JUST_ORDER),
    ("NO_ELIM",     NO_ELIM),
]


# Schedule: each case gets all configs run sequentially.
SCHEDULE = [
    (16, 12),   # baseline reference
    (16, 14),   # ~3x rows
    (16, 16),   # bigger
]


def run_case(d, R):
    print(f'\n========== d={d}, R={R} ==========', flush=True)
    t0 = time.time()
    _, M_mats = build_window_matrices(d)
    M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats, d)
    d_eff = z2_dim(d)
    opts = BuildOptions(R=R, use_z2=True, eliminate_c_slacks=False)
    build = build_handelman_lp(d_eff, M_mats_eff, opts)
    n_rows = (build.A_eq.shape[0] if build.A_eq is not None else 0) + \
             (build.A_ub.shape[0] if build.A_ub is not None else 0)
    print(f'  built in {time.time()-t0:.1f}s : rows={n_rows:,} '
          f'vars={build.n_vars:,} nnz={build.n_nonzero_A:,}', flush=True)

    out = dict(d=d, R=R, n_rows=n_rows, n_vars=build.n_vars,
               nnz=build.n_nonzero_A, configs={})
    for name, params in CONFIGS:
        print(f"\n  --- config={name} ---", flush=True)
        try:
            alpha, status, wall, rss, log = _solve_with_config(build, params)
            stats = _parse_factor_stats(log)
            out['configs'][name] = dict(
                alpha=alpha, status=str(status), wall=wall, rss_mb=rss, **stats,
            )
            nz_after = stats.get('nz_after', 0)
            flops = stats.get('flops', 0)
            print(f"  alpha={alpha}  wall={wall:.1f}s  "
                  f"factor_nnz={nz_after:.2e}  flops={flops:.2e}  "
                  f"rss={rss:.0f}MB  iters={stats.get('n_iter', '?')}",
                  flush=True)
        except Exception as e:
            out['configs'][name] = dict(error=str(e))
            print(f"  ERROR: {e}", flush=True)
    return out


print('Running Tier 1 MOSEK config A/B test', flush=True)
print('CPU count:', os.cpu_count(), flush=True)
print('Available memory:',
      f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB initial RSS",
      flush=True)

results = []
for d, R in SCHEDULE:
    rec = run_case(d, R)
    results.append(rec)
    with open('tier1_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


print('\n\n=== SUMMARY ===\n', flush=True)
print(f"{'case':>10} {'config':>12} {'alpha':>10} {'wall_s':>9} "
      f"{'factor_nnz':>13} {'speedup_vs_OLD':>14} {'iter':>5}", flush=True)
print('-' * 90, flush=True)
for rec in results:
    case = f"d{rec['d']}R{rec['R']}"
    old = rec['configs'].get('OLD', {})
    old_wall = old.get('wall', None) if isinstance(old, dict) else None
    for name in ['OLD', 'NEW', 'JUST_ELIM', 'JUST_ORDER', 'NO_ELIM']:
        c = rec['configs'].get(name, {})
        if 'error' in c:
            print(f"{case:>10} {name:>12} ERROR: {c['error'][:50]}", flush=True)
            continue
        a = c.get('alpha', None)
        a_str = f"{a:.6f}" if a is not None else "  N/A   "
        w = c.get('wall', 0)
        nz = c.get('nz_after', 0)
        sp_str = (f"{old_wall/w:.2f}x" if old_wall and w > 0 else "  -- ")
        it = c.get('n_iter', '?')
        print(f"{case:>10} {name:>12} {a_str:>10} {w:>9.1f} "
              f"{nz:>13.2e} {sp_str:>14} {it:>5}", flush=True)
    print('', flush=True)
