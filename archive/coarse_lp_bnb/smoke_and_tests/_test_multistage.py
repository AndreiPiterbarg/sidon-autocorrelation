"""Test the multi-stage CG+activeSet pipeline.

Key question: does seeding W from a low-R solve give a real speedup at high R?

Schedule:
  d=16 R_seed=6, R_target=10 -- correctness (compare to full)
  d=16 R_seed=8, R_target=12 -- the BIG TEST: full was 463s; multi-stage target?
  d=16 R_seed=8, R_target=15 -- push higher
  d=16 R_seed=10, R_target=18 -- push to breakthrough territory
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
from lasserre.polya_lp.cg_multistage import solve_multistage

d_arg = int(sys.argv[1]); R_seed = int(sys.argv[2]); R_target = int(sys.argv[3])
mode = sys.argv[4]  # MULTISTAGE or FULL

t0 = time.time()
_, M_mats = build_window_matrices(d_arg)
M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats, d_arg)
d_eff = z2_dim(d_arg)

if mode == 'FULL':
    opts = BuildOptions(R=R_target, use_z2=True, eliminate_c_slacks=False,
                        use_q_polynomial=True)
    build = build_handelman_lp(d_eff, M_mats_eff, opts)
    sol = solve_lp(build, solver='mosek', verbose=False)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    out = dict(mode='FULL', d=d_arg, R=R_target,
               alpha=sol.alpha, status=sol.status,
               n_rows=int(build.A_eq.shape[0]),
               n_vars=int(build.n_vars),
               nnz=int(build.n_nonzero_A),
               t_total=time.time() - t0,
               rss_mb=rss)
elif mode == 'MULTISTAGE':
    res = solve_multistage(
        d=d_eff, M_mats=M_mats_eff,
        R_target=R_target, R_seed=R_seed,
        tol=1e-8, active_lambda_tol=1e-9,
        expand_neighbors=0,
        max_iter_per_stage=30,
        verbose=True,
    )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    out = dict(mode='MULTISTAGE', d=d_arg,
               R_seed=R_seed, R_target=R_target,
               seed_alpha=res.seed_alpha,
               seed_active_W=len(res.seed_active_W),
               seed_wall_s=res.seed_wall_s,
               target_alpha=res.target_alpha,
               target_converged=res.target_result.converged,
               target_n_iter=len(res.target_result.iterations),
               target_n_W_active_final=len(res.target_result.final_W_active),
               target_n_W_full=res.target_result.n_W_full,
               total_wall_s=res.total_wall_s,
               iter_log=[dict(it=it.iteration, alpha=it.alpha,
                              n_S=it.n_constraints,
                               n_W_act=it.n_active_W,
                               n_bv=it.n_beta_violators,
                               n_wv=it.n_W_violators,
                               build=it.build_wall_s,
                               solve=it.solve_wall_s,
                               price=it.pricing_wall_s,
                               rss_mb=it.rss_mb)
                          for it in res.target_result.iterations],
               rss_mb=rss)
print('RESULT:', json.dumps(out, default=str), flush=True)
'''

PROBE_PATH = '_multistage_probe.py'
with open(PROBE_PATH, 'w') as f:
    f.write(PROBE)


SCHEDULE = [
    (16, 6, 10, 'FULL'),
    (16, 6, 10, 'MULTISTAGE'),
    (16, 8, 12, 'MULTISTAGE'),     # <-- the big speedup test (full was 463s)
    (16, 8, 15, 'MULTISTAGE'),
    (16, 10, 18, 'MULTISTAGE'),
]
PER_TASK_TIMEOUT = 3600


def _run(d, R_seed, R_target, mode):
    print(f"\n========== d={d} R_seed={R_seed} R_target={R_target} mode={mode} ==========",
          flush=True)
    t_start = time.time()
    proc = subprocess.Popen(
        ['python3', '-u', PROBE_PATH, str(d), str(R_seed), str(R_target), mode],
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
        return dict(d=d, R_seed=R_seed, R_target=R_target, mode=mode,
                    error='timeout', wall=wall)
    result = None
    for line in full.splitlines():
        if line.startswith('RESULT:'):
            result = json.loads(line[len('RESULT:'):].strip())
    if result is None:
        return dict(d=d, R_seed=R_seed, R_target=R_target, mode=mode,
                    error='no_result', wall=wall, tail=full[-1500:])
    result['wall'] = wall
    return result


results = []
for case in SCHEDULE:
    r = _run(*case)
    results.append(r)
    with open('multistage_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


print('\n\n=== SUMMARY ===\n', flush=True)
print(f"{'case':>16} {'mode':>10} {'alpha':>11} {'W':>5} {'iters':>5} {'wall':>9}",
      flush=True)
print('-' * 75)
for r in results:
    if 'error' in r:
        print(f"d{r['d']}R{r.get('R_target',r.get('R'))}  ERROR: {r['error']}", flush=True)
        continue
    case = f"d{r['d']}R{r.get('R_target', r.get('R'))}"
    mode = r['mode']
    a = r.get('target_alpha') or r.get('alpha')
    a_str = f"{a:.7f}" if a is not None else "N/A"
    W = r.get('target_n_W_active_final', '-')
    n_it = r.get('target_n_iter', 1)
    w = r.get('total_wall_s') or r.get('t_total')
    print(f"{case:>16} {mode:>10} {a_str:>11} {str(W):>5} {n_it:>5} {w:>9.1f}",
          flush=True)
