"""V2: OR-Tools PDLP with explicit multi-threading + iteration logging.

Uses pdlp.primal_dual_hybrid_gradient() directly (Python bindings) with
proper threading and verbosity. Skip the linear_solver wrapper which
silently disables threading.
"""
import sys, os, time, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


PROBE = '''
import sys, time, json, resource
sys.path.insert(0, '/root/sidon')
import numpy as np
from scipy import sparse as sp

from lasserre.polya_lp.build import (
    BuildOptions, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim
from lasserre.polya_lp.solve import solve_lp

d_arg = int(sys.argv[1]); R_arg = int(sys.argv[2])
mode = sys.argv[3]
n_threads = int(sys.argv[4]) if len(sys.argv) > 4 else 64

t0 = time.time()
_, M_mats = build_window_matrices(d_arg)
M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats, d_arg)
d_eff = z2_dim(d_arg)
opts = BuildOptions(R=R_arg, use_z2=True, eliminate_c_slacks=False,
                    use_q_polynomial=True)
build = build_handelman_lp(d_eff, M_mats_eff, opts)
print(f"BUILD rows={build.A_eq.shape[0]} vars={build.n_vars} "
      f"nnz={build.n_nonzero_A} t={time.time()-t0:.1f}s", flush=True)


def solve_ortools_pdlp_native(build, tol=1e-4, time_limit_s=900.0,
                               n_threads=64, verbose=True):
    """Use OR-Tools low-level PDHG bindings with explicit multi-threading."""
    from ortools.pdlp import solvers_pb2
    from ortools.pdlp.python import pdlp
    from ortools.linear_solver.python import model_builder

    # Build model via model_builder so we get LinearProgram protos
    mb = model_builder.Model()
    n_vars = build.n_vars
    vs = []
    for j, (lo, hi) in enumerate(build.bounds):
        lb = -1e30 if lo is None else lo
        ub = 1e30 if hi is None else hi
        v = mb.new_var(lb, ub, False, f"x{j}")
        vs.append(v)
    obj_terms = [(vs[j], float(build.c[j]))
                 for j in range(n_vars) if build.c[j] != 0]
    mb.minimize(sum(c * v for v, c in obj_terms))

    A = build.A_eq.tocsr()
    b = build.b_eq
    n_eq = A.shape[0]
    for i in range(n_eq):
        bi = float(b[i])
        start, end = A.indptr[i], A.indptr[i+1]
        cs = [(vs[A.indices[k]], float(A.data[k]))
              for k in range(start, end)]
        mb.add(sum(c * v for v, c in cs) == bi)

    # Convert to QuadraticProgram for PDLP
    qp = pdlp.qp_from_mpmodel_proto(mb.export_to_proto(),
                                     relax_integer_variables=True)
    params = solvers_pb2.PrimalDualHybridGradientParams()
    params.termination_criteria.eps_optimal_absolute = tol
    params.termination_criteria.eps_optimal_relative = tol
    params.termination_criteria.time_sec_limit = time_limit_s
    params.num_threads = n_threads
    params.verbosity_level = 3 if verbose else 1

    t0 = time.time()
    result = pdlp.primal_dual_hybrid_gradient(qp, params)
    wall = time.time() - t0
    primal = np.asarray(result.primal_solution, dtype=np.float64)
    obj = float(result.solve_log.solution_stats.convergence_information[0]
                .primal_objective)
    return result, primal, -obj, wall


if mode == 'ORTOOLS_PDLP':
    print(f"OR-Tools PDLP tol=1e-4 threads={n_threads}", flush=True)
    try:
        result, x, alpha, t = solve_ortools_pdlp_native(
            build, tol=1e-4, time_limit_s=900.0,
            n_threads=n_threads, verbose=True)
        print(f"OR_PDLP_DONE alpha={alpha:.6f} t={t:.1f}s "
              f"term={result.solve_log.termination_reason}", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"OR_PDLP_ERROR: {e}", flush=True)
        x = np.zeros(build.n_vars)
        alpha = None
        t = 0.0
        result = None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    out = dict(mode=mode, d=d_arg, R=R_arg, alpha=alpha,
               t_solve=t, t_total=t, rss_mb=rss)
elif mode == 'ORTOOLS_POLISH':
    print(f"OR-Tools PDLP + MOSEK polish, threads={n_threads}", flush=True)
    result, x, alpha_pdlp, t_pdlp = solve_ortools_pdlp_native(
        build, tol=1e-4, time_limit_s=900.0, n_threads=n_threads, verbose=False)
    print(f"OR_PDLP_DONE alpha~{alpha_pdlp:.6f} t={t_pdlp:.1f}s", flush=True)
    lam = x[build.lambda_idx]
    active_W = [w for w, v in enumerate(lam) if v > 1e-5]
    print(f"Active windows {len(active_W)}/{len(lam)} max_lam={lam.max():.3e}",
          flush=True)
    t_pol_start = time.time()
    if active_W:
        sub_M = [M_mats_eff[w] for w in active_W]
        opts_pol = BuildOptions(R=R_arg, use_z2=True,
                                eliminate_c_slacks=False, use_q_polynomial=True)
        build_pol = build_handelman_lp(d_eff, sub_M, opts_pol)
        sol_pol = solve_lp(build_pol, solver='mosek', verbose=False)
        polish_alpha = sol_pol.alpha
    else:
        polish_alpha = None
    t_pol = time.time() - t_pol_start
    print(f"POLISH_DONE alpha={polish_alpha} t={t_pol:.1f}s", flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    out = dict(mode=mode, d=d_arg, R=R_arg, alpha_pdlp=alpha_pdlp,
               alpha=polish_alpha, n_active_W=len(active_W),
               n_W_full=len(M_mats_eff),
               t_pdlp=t_pdlp, t_polish=t_pol,
               t_total=t_pdlp + t_pol, rss_mb=rss)

print('RESULT:', json.dumps(out, default=str), flush=True)
'''

PROBE_PATH = '_pdlp_v2_probe.py'
with open(PROBE_PATH, 'w') as f:
    f.write(PROBE)


SCHEDULE = [
    (16, 8, 'ORTOOLS_PDLP', 64),
    (16, 12, 'ORTOOLS_PDLP', 64),
    (16, 12, 'ORTOOLS_POLISH', 64),
]
PER_TASK_TIMEOUT = 1500


def _run(d, R, mode, threads):
    print(f"\n========== d={d} R={R} mode={mode} threads={threads} ==========", flush=True)
    t_start = time.time()
    proc = subprocess.Popen(
        ['python3', '-u', PROBE_PATH, str(d), str(R), mode, str(threads)],
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
    return result if result else dict(d=d, R=R, mode=mode,
                                       error='no_result', wall=wall,
                                       tail=full[-1500:])


results = []
for case in SCHEDULE:
    r = _run(*case)
    results.append(r)
    with open('pdlp_v2_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

print('\n=== SUMMARY ===\n', flush=True)
for r in results:
    print(json.dumps(r, default=str), flush=True)
