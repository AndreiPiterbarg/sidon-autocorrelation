"""HiGHS PDLP + MOSEK polish — does it actually speed up?

Reference: MOSEK direct on d=16 R=12 = 463 s.

Test:
  1. d=16 R=12: HiGHS PDLP alone (1e-4 tol). Time it.
  2. Same: identify active windows from PDLP primal, polish with MOSEK.
  3. Compare to MOSEK direct.
"""
import sys, os, time, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


PROBE = '''
import sys, time, json, resource
sys.path.insert(0, '/root/sidon')
import numpy as np
from scipy import sparse as sp
import highspy

from lasserre.polya_lp.build import (
    BuildOptions, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim
from lasserre.polya_lp.solve import solve_lp

d_arg = int(sys.argv[1]); R_arg = int(sys.argv[2])
mode = sys.argv[3]

t0 = time.time()
_, M_mats = build_window_matrices(d_arg)
M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats, d_arg)
d_eff = z2_dim(d_arg)

opts = BuildOptions(R=R_arg, use_z2=True, eliminate_c_slacks=False,
                    use_q_polynomial=True)
build = build_handelman_lp(d_eff, M_mats_eff, opts)
print(f"BUILD rows={build.A_eq.shape[0]} vars={build.n_vars} "
      f"nnz={build.n_nonzero_A} t={time.time()-t0:.1f}s", flush=True)


def solve_highs_pdlp(build, tol=1e-4, time_limit=600.0, verbose=False):
    """Solve via HiGHS PDLP. Returns (status, x, alpha, wall)."""
    h = highspy.Highs()
    if not verbose:
        h.silent()
    h.setOptionValue("solver", "pdlp")
    h.setOptionValue("primal_feasibility_tolerance", tol)
    h.setOptionValue("dual_feasibility_tolerance", tol)
    h.setOptionValue("time_limit", time_limit)
    h.setOptionValue("presolve", "on")

    A = build.A_eq.tocsc()
    inf = highspy.kHighsInf
    n_vars = build.n_vars

    col_lo = np.array([(-inf if lo is None else lo) for lo, _ in build.bounds],
                      dtype=np.float64)
    col_hi = np.array([(inf if hi is None else hi) for _, hi in build.bounds],
                      dtype=np.float64)

    lp = highspy.HighsLp()
    lp.num_col_ = n_vars
    lp.num_row_ = A.shape[0]
    lp.col_cost_ = build.c.copy()
    lp.col_lower_ = col_lo
    lp.col_upper_ = col_hi
    lp.row_lower_ = build.b_eq.copy()
    lp.row_upper_ = build.b_eq.copy()
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A.indices.astype(np.int32)
    lp.a_matrix_.value_ = A.data.astype(np.float64)
    lp.sense_ = highspy.ObjSense.kMinimize

    h.passModel(lp)
    t0 = time.time()
    h.run()
    wall = time.time() - t0
    status = h.getModelStatus()
    info = h.getInfo()
    obj = info.objective_function_value
    sol = h.getSolution()
    x = np.asarray(sol.col_value)
    return str(status), x, -float(obj), wall


if mode == 'MOSEK_DIRECT':
    t0 = time.time()
    sol = solve_lp(build, solver='mosek', verbose=False)
    t_solve = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    out = dict(mode=mode, d=d_arg, R=R_arg,
               alpha=sol.alpha, status=sol.status,
               n_rows=int(build.A_eq.shape[0]), n_vars=int(build.n_vars),
               nnz=int(build.n_nonzero_A),
               t_solve=t_solve, t_total=t_solve, rss_mb=rss)

elif mode == 'PDLP':
    print("PDLP (HiGHS) tol=1e-4", flush=True)
    status, x, alpha_pdlp, t_pdlp = solve_highs_pdlp(
        build, tol=1e-4, time_limit=900.0, verbose=True)
    print(f"PDLP_DONE status={status} alpha={alpha_pdlp:.6f} "
          f"t={t_pdlp:.1f}s", flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    out = dict(mode=mode, d=d_arg, R=R_arg, status=status,
               alpha=alpha_pdlp, t_solve=t_pdlp, t_total=t_pdlp, rss_mb=rss)

elif mode == 'PDLP_POLISH':
    print("PDLP (HiGHS) tol=1e-4 + MOSEK polish", flush=True)
    status, x_pdlp, alpha_pdlp, t_pdlp = solve_highs_pdlp(
        build, tol=1e-4, time_limit=900.0, verbose=True)
    print(f"PDLP_DONE status={status} alpha~{alpha_pdlp:.6f} t={t_pdlp:.1f}s",
          flush=True)

    # Active set: lambda_W > 1e-5
    lam = x_pdlp[build.lambda_idx]
    active_W = [w for w, v in enumerate(lam) if v > 1e-5]
    print(f"Active windows {len(active_W)}/{len(lam)} "
          f"max_lam={lam.max():.3e} sum={lam.sum():.6f}", flush=True)

    # Polish with MOSEK on reduced LP
    t_pol_start = time.time()
    sub_M_mats = [M_mats_eff[w] for w in active_W]
    opts_pol = BuildOptions(R=R_arg, use_z2=True,
                            eliminate_c_slacks=False, use_q_polynomial=True)
    build_pol = build_handelman_lp(d_eff, sub_M_mats, opts_pol)
    sol_pol = solve_lp(build_pol, solver='mosek', verbose=False)
    t_pol = time.time() - t_pol_start
    print(f"POLISH_DONE alpha={sol_pol.alpha} t={t_pol:.1f}s", flush=True)

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    out = dict(mode=mode, d=d_arg, R=R_arg, alpha_pdlp=alpha_pdlp,
               alpha=sol_pol.alpha, n_active_W=len(active_W),
               n_W_full=len(M_mats_eff), n_rows=int(build.A_eq.shape[0]),
               n_vars=int(build.n_vars), nnz=int(build.n_nonzero_A),
               t_pdlp=t_pdlp, t_polish=t_pol, t_total=t_pdlp + t_pol,
               rss_mb=rss)

print('RESULT:', json.dumps(out, default=str), flush=True)
'''

PROBE_PATH = '_highs_pdlp_probe.py'
with open(PROBE_PATH, 'w') as f:
    f.write(PROBE)

SCHEDULE = [
    (16, 8, 'PDLP'),         # quick PDLP test
    (16, 8, 'PDLP_POLISH'),  # full workflow
    (16, 12, 'PDLP'),        # ref scale
    (16, 12, 'PDLP_POLISH'), # the headline number
]
PER_TASK_TIMEOUT = 1800


def _run(d, R, mode):
    print(f"\n========== d={d} R={R} mode={mode} ==========", flush=True)
    t_start = time.time()
    proc = subprocess.Popen(
        ['python3', '-u', PROBE_PATH, str(d), str(R), mode],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    captured = []
    timed_out = False
    try:
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            captured.append(line)
            if time.time() - t_start > PER_TASK_TIMEOUT:
                timed_out = True
                proc.kill()
                break
        proc.wait(timeout=60)
    except Exception as e:
        try: proc.kill()
        except Exception: pass
    full = ''.join(captured)
    wall = time.time() - t_start
    if timed_out:
        return dict(d=d, R=R, mode=mode, error='timeout', wall=wall)
    result = None
    for line in full.splitlines():
        if line.startswith('RESULT:'):
            result = json.loads(line[len('RESULT:'):].strip())
    if result is None:
        return dict(d=d, R=R, mode=mode, error='no_result',
                    wall=wall, tail=full[-1500:])
    result['wall'] = wall
    return result


results = []
for case in SCHEDULE:
    r = _run(*case)
    results.append(r)
    with open('highs_pdlp_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


print('\n\n=== SUMMARY ===\n', flush=True)
print(f"{'case':>10} {'mode':>14} {'alpha':>11} {'t_pdlp':>8} {'t_pol':>8} {'t_tot':>8}", flush=True)
print('-' * 65)
for r in results:
    if 'error' in r:
        print(f"d{r['d']}R{r['R']} {r['mode']:>14} ERROR: {r['error']}", flush=True)
        continue
    case = f"d{r['d']}R{r['R']}"
    a = r.get('alpha')
    a_str = f"{a:.7f}" if a is not None else "N/A"
    tp = r.get('t_pdlp', r.get('t_solve', 0))
    tpo = r.get('t_polish', 0)
    tot = r.get('t_total', 0)
    print(f"{case:>10} {r['mode']:>14} {a_str:>11} {tp:>8.1f} {tpo:>8.1f} {tot:>8.1f}",
          flush=True)
