"""Measure PDLP+MOSEK-polish speedup vs MOSEK-direct on d=16 R=12.

Reference baseline (1.4 TB pod): MOSEK direct = 463 s, alpha = 1.2362297.

Test:
  1. Build LP at d=16 R=12 (full LP, ~126K rows, ~202K vars).
  2. Run PDLP-CPU to KKT 1e-4, then continue to 1e-6.
  3. Identify primal active set: lambda_W > tol, c_beta > tol.
  4. Build reduced LP keeping only active windows + active rows.
     (Rows with c_beta = 0 are TIGHT and remain in the LP. Rows with
      c_beta > 0 are slack and could be dropped, but dropping them
      removes constraints, which RAISES alpha -- bad for soundness.
      Standard approach: drop only the lambda columns.)
  5. Solve reduced LP with MOSEK at 1e-9.
  6. Report total wall time and final alpha.

For comparison: also run MOSEK direct.
"""
import sys, os, time, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


PROBE = '''
import sys, time, json, resource
sys.path.insert(0, '/root/sidon')
import numpy as np
import torch
torch.set_num_threads(64)        # MOSEK saturates around 32-64 anyway
print(f"torch threads: {torch.get_num_threads()}", flush=True)

from lasserre.polya_lp.build import (
    BuildOptions, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim
from lasserre.polya_lp.solve import solve_lp
from lasserre.polya_lp.pdlp import (
    build_gpu_lp, pdlp_solve, solve_buildresult,
)

d_arg = int(sys.argv[1]); R_arg = int(sys.argv[2])
mode = sys.argv[3]   # MOSEK_DIRECT or PDLP_POLISH

t0 = time.time()
_, M_mats = build_window_matrices(d_arg)
M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats, d_arg)
d_eff = z2_dim(d_arg)

opts = BuildOptions(R=R_arg, use_z2=True, eliminate_c_slacks=False,
                    use_q_polynomial=True)
build = build_handelman_lp(d_eff, M_mats_eff, opts)
n_rows = build.A_eq.shape[0]
print(f"BUILD rows={n_rows} vars={build.n_vars} nnz={build.n_nonzero_A} "
      f"t={time.time()-t0:.1f}s", flush=True)

if mode == 'MOSEK_DIRECT':
    t0 = time.time()
    sol = solve_lp(build, solver='mosek', verbose=False)
    t_solve = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    out = dict(mode=mode, d=d_arg, R=R_arg,
               alpha=sol.alpha, status=sol.status,
               n_rows=int(n_rows), n_vars=int(build.n_vars),
               nnz=int(build.n_nonzero_A),
               t_solve=t_solve, t_total=t_solve, rss_mb=rss)
elif mode == 'PDLP_POLISH':
    # Stage 1: PDLP to 1e-4
    t_pdlp_start = time.time()
    print(f"PDLP_START tol=1e-4", flush=True)
    pdlp_result, scaling = solve_buildresult(
        build, max_outer=400, max_inner=500, tol=1e-4,
        spectral_iter=30, log_every=5, print_log=True,
    )
    t_pdlp = time.time() - t_pdlp_start
    print(f"PDLP_DONE outer={pdlp_result.n_outer} inner={pdlp_result.n_inner_total} "
          f"kkt={pdlp_result.kkt:.3e} obj={pdlp_result.obj_primal:.6f} "
          f"alpha~={-pdlp_result.obj_primal:.6f} t={t_pdlp:.1f}s", flush=True)

    # Extract primal solution (in scaled space) and unscale
    x_scaled = pdlp_result.x.cpu().numpy()
    y_scaled = pdlp_result.y.cpu().numpy()
    # If Ruiz scaling was applied, unscale; otherwise x_scaled is the solution.
    try:
        from lasserre.polya_lp.pdlp import unscale_solution
        x_unscaled = unscale_solution(x_scaled, scaling)
    except Exception:
        x_unscaled = x_scaled

    # Identify active windows from PDLP primal: lambda_W > 1e-4
    lam = x_unscaled[build.lambda_idx]
    active_W = [w for w, v in enumerate(lam) if v > 1e-6]
    print(f"PDLP active windows: {len(active_W)}/{len(lam)} "
          f"(lam max={lam.max():.3e}, sum={lam.sum():.6f})", flush=True)

    # Stage 2: solve REDUCED LP with MOSEK at 1e-9, restricted to active W.
    t_polish_start = time.time()
    sub_M_mats = [M_mats_eff[w] for w in active_W]
    opts_polish = BuildOptions(R=R_arg, use_z2=True,
                               eliminate_c_slacks=False,
                               use_q_polynomial=True)
    build_polish = build_handelman_lp(d_eff, sub_M_mats, opts_polish)
    sol_polish = solve_lp(build_polish, solver='mosek', verbose=False)
    t_polish = time.time() - t_polish_start
    print(f"POLISH_DONE alpha={sol_polish.alpha} "
          f"t_polish={t_polish:.1f}s", flush=True)

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    out = dict(mode=mode, d=d_arg, R=R_arg,
               alpha_pdlp_approx=-pdlp_result.obj_primal,
               alpha=sol_polish.alpha,
               kkt_pdlp=pdlp_result.kkt,
               n_outer=pdlp_result.n_outer,
               n_inner=pdlp_result.n_inner_total,
               n_rows=int(n_rows), n_vars=int(build.n_vars),
               nnz=int(build.n_nonzero_A),
               n_active_W=len(active_W),
               n_W_full=len(M_mats_eff),
               t_pdlp=t_pdlp,
               t_polish=t_polish,
               t_total=t_pdlp + t_polish,
               rss_mb=rss)
print('RESULT:', json.dumps(out, default=str), flush=True)
'''

PROBE_PATH = '_pdlp_polish_probe.py'
with open(PROBE_PATH, 'w') as f:
    f.write(PROBE)


SCHEDULE = [
    (16, 8, 'MOSEK_DIRECT'),
    (16, 8, 'PDLP_POLISH'),
    (16, 12, 'MOSEK_DIRECT'),
    (16, 12, 'PDLP_POLISH'),
]
PER_TASK_TIMEOUT = 3600  # 1 h cap


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
        print(f"  STREAM_ERROR: {e}", flush=True)
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
    with open('pdlp_polish_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


print('\n\n=== SUMMARY ===\n', flush=True)
print(f"{'case':>10} {'mode':>14} {'alpha':>11} {'wall':>9} {'pdlp':>8} {'polish':>8}",
      flush=True)
print('-' * 75)
for r in results:
    if 'error' in r:
        print(f"d{r['d']}R{r['R']} {r['mode']:>14} ERROR: {r['error']}", flush=True)
        continue
    case = f"d{r['d']}R{r['R']}"
    a = r.get('alpha')
    a_str = f"{a:.7f}" if a is not None else "N/A"
    w = r.get('t_total', 0)
    tp = r.get('t_pdlp', 0)
    tpo = r.get('t_polish', 0)
    print(f"{case:>10} {r['mode']:>14} {a_str:>11} {w:>9.1f} {tp:>8.1f} {tpo:>8.1f}",
          flush=True)
