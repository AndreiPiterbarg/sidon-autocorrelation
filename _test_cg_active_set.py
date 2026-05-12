"""End-to-end test of Tier 2+3 (CG + active-set window pricing).

Validation:
  1. d=8 R=8: CG+AS alpha must equal full-LP alpha to ~1e-7.
  2. d=16 R=12: CG+AS alpha must equal full-LP alpha (1.236225...).
     Measure speedup vs full LP.
  3. d=16 R=15: scale test (full LP would be heavy; CG+AS should be fast).
  4. d=16 R=18 / R=22: push toward the breakthrough regime.

Each case runs in a subprocess so a hang doesn't kill the rest.
"""
import sys, os, time, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


PROBE = '''
import sys, time, json, resource, traceback
sys.path.insert(0, '/root/sidon')
import numpy as np
from lasserre.polya_lp.build import (
    BuildOptions, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim
from lasserre.polya_lp.solve import solve_lp
from lasserre.polya_lp.cg_active_set import solve_with_cg_active_set

d_arg = int(sys.argv[1]); R_arg = int(sys.argv[2]); mode = sys.argv[3]

t0 = time.time()
_, M_mats = build_window_matrices(d_arg)
M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats, d_arg)
d_eff = z2_dim(d_arg)
t_setup = time.time() - t0

if mode == 'FULL':
    t0 = time.time()
    opts = BuildOptions(R=R_arg, use_z2=True, eliminate_c_slacks=False,
                        use_q_polynomial=True)
    build = build_handelman_lp(d_eff, M_mats_eff, opts)
    t_build = time.time() - t0
    n_rows = (build.A_eq.shape[0] if build.A_eq is not None else 0)
    print(f"FULL_BUILD rows={n_rows} vars={build.n_vars} "
          f"nnz={build.n_nonzero_A} t_build={t_build:.1f}s", flush=True)
    t0 = time.time()
    sol = solve_lp(build, solver='mosek', verbose=False)
    t_solve = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    out = dict(mode='FULL', d=d_arg, R=R_arg, alpha=sol.alpha,
               status=sol.status, t_build=t_build, t_solve=t_solve,
               n_rows=int(n_rows), n_vars=int(build.n_vars),
               nnz=int(build.n_nonzero_A), rss_mb=rss,
               n_iter=1)
elif mode == 'CGAS':
    t0 = time.time()
    res = solve_with_cg_active_set(
        d=d_eff, M_mats=M_mats_eff, R=R_arg,
        max_iter=50, tol=1e-8,
        add_top_k_betas=-1, add_top_k_windows=-1,
        initial_W_active=None,           # start with all windows
        verbose=True,
    )
    t_total = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    last_iter = res.iterations[-1] if res.iterations else None
    out = dict(mode='CGAS', d=d_arg, R=R_arg,
               alpha=res.final_alpha,
               converged=res.converged,
               n_iter=len(res.iterations),
               n_Sigma_R_final=len(res.final_Sigma_R),
               n_B_R_final=len(res.final_B_R),
               n_W_active_final=len(res.final_W_active),
               n_W_full=res.n_W_full,
               t_total=t_total,
               t_build_total=sum(it.build_wall_s for it in res.iterations),
               t_solve_total=sum(it.solve_wall_s for it in res.iterations),
               t_pricing_total=sum(it.pricing_wall_s
                                    for it in res.iterations),
               rss_mb=rss,
               max_beta_viol_final=(last_iter.max_beta_violation
                                     if last_iter else None),
               max_W_viol_final=(last_iter.max_W_violation
                                  if last_iter else None),
               iter_log=[dict(it=it.iteration, alpha=it.alpha,
                              n_S=it.n_constraints, n_W=it.n_active_W,
                              n_bv=it.n_beta_violators,
                              n_wv=it.n_W_violators,
                              build=it.build_wall_s,
                              solve=it.solve_wall_s)
                          for it in res.iterations],
               )
else:
    raise ValueError(f"unknown mode {mode}")

print('RESULT:', json.dumps(out, default=str), flush=True)
'''

PROBE_PATH = '_cg_active_set_probe.py'
with open(PROBE_PATH, 'w') as f:
    f.write(PROBE)


# Schedule: (d, R, [modes]).
SCHEDULE = [
    (8, 8, ['FULL', 'CGAS']),       # tiny, quick correctness check
    (16, 8, ['FULL', 'CGAS']),      # validation: should match exactly
    (16, 12, ['FULL', 'CGAS']),     # reference: full ~660s on 14-core
    (16, 15, ['CGAS']),             # full would be slow; CGAS only
    (16, 18, ['CGAS']),             # push toward breakthrough
]
PER_TASK_TIMEOUT = 3600  # 1 h cap per (case, mode)


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
for d, R, modes in SCHEDULE:
    for mode in modes:
        r = _run(d, R, mode)
        results.append(r)
        with open('cg_active_set_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)


# Summary
print('\n\n=== SUMMARY ===\n', flush=True)
print(f"{'case':>10} {'mode':>5} {'alpha':>11} {'iters':>6} "
      f"{'rows':>10} {'W':>5} {'wall':>9} {'speedup':>9}", flush=True)
print('-' * 80)
for r in results:
    if 'error' in r:
        print(f"d{r['d']}R{r['R']:>2} {r['mode']:>5}  ERROR: {r['error']}", flush=True)
        continue
    case = f"d{r['d']}R{r['R']}"
    a = r.get('alpha')
    a_str = f"{a:.7f}" if a is not None else "N/A"
    n_it = r.get('n_iter', 1)
    rows = r.get('n_rows') or r.get('n_Sigma_R_final', '?')
    W = r.get('n_W_active_final') or '-'
    w = r.get('t_total') or (r.get('t_build', 0) + r.get('t_solve', 0))
    print(f"{case:>10} {r['mode']:>5} {a_str:>11} {n_it:>6} "
          f"{str(rows):>10} {str(W):>5} {w:>9.1f}",
          flush=True)
