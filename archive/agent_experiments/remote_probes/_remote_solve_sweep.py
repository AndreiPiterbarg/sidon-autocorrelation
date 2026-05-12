"""End-to-end SOLVE sweep: build + MOSEK IPM, record alpha + trend.

This is the test that tells us what we can actually PROVE at each (d, R).

Schedule prioritizes the trajectory toward 1.281:
  * Fix d, push R: shows convergence rate gap ~ C/R.
  * Fix R, push d: shows cross-d trend C(d).
  * d=48, d=64 R=6,8: are we close to the val(d) target?

Each (d, R) is run in a subprocess with a wall budget so a single hang
doesn't tank the sweep.
"""
import sys, os, json, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


PROBE = """
import sys, time, json, resource
sys.path.insert(0, '/home/ubuntu/sidon')
from lasserre.polya_lp.build import (
    build_handelman_lp, BuildOptions, build_window_matrices,
)
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim
from lasserre.polya_lp.solve import solve_lp

d = int(sys.argv[1]); R = int(sys.argv[2])
# eliminate_c_slacks=False (original equality form) -- with c_slack elimination
# + solve_form=DUAL, MOSEK builds a huge KKT factor and stalls. The equality
# form solves in seconds at the same scale.
elim = False

t0 = time.time()
_, M_mats = build_window_matrices(d)
M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats, d)
d_eff = z2_dim(d)
opts = BuildOptions(R=R, use_z2=True, eliminate_c_slacks=elim)
t_setup = time.time() - t0

t0 = time.time()
build = build_handelman_lp(d_eff, M_mats_eff, opts)
t_build = time.time() - t0

n_rows = (build.A_eq.shape[0] if build.A_eq is not None else 0) + \\
         (build.A_ub.shape[0] if build.A_ub is not None else 0)

t0 = time.time()
sol = solve_lp(build, solver='mosek', verbose=False)
t_solve = time.time() - t0

rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

out = dict(d=d, R=R, d_eff=d_eff, alpha=sol.alpha, status=sol.status,
           n_W=len(M_mats_eff),
           n_rows=n_rows, n_vars=build.n_vars, nnz=build.n_nonzero_A,
           t_setup=t_setup, t_build=t_build, t_solve=t_solve,
           rss_mb=rss_mb, elim=elim)
print('RESULT:', json.dumps(out, default=str), flush=True)
"""

with open('_solve_probe.py', 'w') as f:
    f.write(PROBE)


# Schedule: hits the question "what can we prove and at what cost?"
SCHEDULE = [
    # Trajectory at fixed d, increasing R: convergence to val(d)
    (8,  4), (8,  8), (8, 12), (8, 16),
    (16, 4), (16, 8), (16, 12), (16, 16),
    (24, 4), (24, 8), (24, 10),
    (32, 4), (32, 6), (32, 8),
    # Cross-d at fixed R: how much does adding d help?
    (48, 4), (48, 6), (48, 8),
    (64, 4), (64, 6), (64, 8),
    (80, 4), (80, 6),
    (96, 4),
    (128, 4),
]

PER_TASK_TIMEOUT = 1800  # 30 min wall

results = []
for d, R in SCHEDULE:
    print(f"\n=== d={d}, R={R} ===", flush=True)
    t_start = time.time()
    try:
        proc = subprocess.run(
            ['python3', '-u', '_solve_probe.py', str(d), str(R), '0'],
            capture_output=True, text=True, timeout=PER_TASK_TIMEOUT,
        )
        wall = time.time() - t_start
        result = None
        for line in proc.stdout.splitlines():
            if line.startswith('RESULT:'):
                result = json.loads(line[len('RESULT:'):].strip())
        if result is not None:
            print(f"  alpha={result['alpha']!s:>10}  status={result['status']}  "
                  f"build={result['t_build']:.1f}s  solve={result['t_solve']:.1f}s  "
                  f"rows={result['n_rows']:,}  nnz={result['nnz']:,}  "
                  f"rss={result['rss_mb']:.0f}MB",
                  flush=True)
            results.append(result)
        else:
            err = proc.stderr[-500:] if proc.stderr else ''
            print(f"  NO RESULT after {wall:.0f}s. stderr tail: {err}", flush=True)
            results.append(dict(d=d, R=R, error='no_result', wall=wall, stderr=err))
    except subprocess.TimeoutExpired:
        wall = time.time() - t_start
        print(f"  TIMEOUT after {wall:.0f}s", flush=True)
        results.append(dict(d=d, R=R, error='timeout', wall=wall))

    with open('solve_sweep_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


print("\n\n=== SUMMARY ===", flush=True)
print(f"{'d':>3} {'R':>3} {'alpha':>11} {'gap_to_1.281':>13} "
      f"{'rows':>10} {'build_s':>8} {'solve_s':>8} {'rss_MB':>8}", flush=True)
print('-' * 90)
for r in results:
    if 'error' in r:
        print(f"{r['d']:>3} {r['R']:>3}  {r['error']}", flush=True)
        continue
    if r.get('alpha') is None:
        print(f"{r['d']:>3} {r['R']:>3}  status={r.get('status', '?')}", flush=True)
        continue
    a = float(r['alpha'])
    gap = 1.281 - a
    print(f"{r['d']:>3} {r['R']:>3} {a:>11.6f} {gap:>+13.6f} "
          f"{r['n_rows']:>10,} {r['t_build']:>8.1f} {r['t_solve']:>8.1f} "
          f"{r['rss_mb']:>8.0f}", flush=True)
