"""End-to-end test of Schur-eliminated Polya LP.

Stage 1 (correctness): match alpha to FULL LP at d=8 R=4..8, d=16 R=6..10.
                       Tolerance ~1e-7 (within MOSEK 1e-9).
Stage 2 (speedup):     d=16 R=12 — full was 463 s on this pod.
Stage 3 (scaling):     d=16 R=15, R=18 — beyond what full LP can do quickly.

Multinomial coefficient dynamic range warnings:
  R=8:  max mult ~ 2520
  R=12: max mult ~ 3e7
  R=15: max mult ~ 5e9   (may need rescaling)
  R=18: max mult ~ 1e12  (likely needs rescaling)
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
from lasserre.polya_lp.schur_build import build_handelman_lp_schur

d_arg = int(sys.argv[1]); R_arg = int(sys.argv[2])
mode = sys.argv[3]

t0 = time.time()
_, M_mats = build_window_matrices(d_arg)
M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats, d_arg)
d_eff = z2_dim(d_arg)
t_setup = time.time() - t0

if mode == 'FULL':
    opts = BuildOptions(R=R_arg, use_z2=True, eliminate_c_slacks=False,
                        use_q_polynomial=True)
    t_b0 = time.time()
    build = build_handelman_lp(d_eff, M_mats_eff, opts)
    t_build = time.time() - t_b0
    print(f"FULL_BUILD rows={build.A_eq.shape[0]} vars={build.n_vars} "
          f"nnz={build.n_nonzero_A} t_build={t_build:.1f}s", flush=True)
    t_s0 = time.time()
    sol = solve_lp(build, solver='mosek', verbose=False)
    t_solve = time.time() - t_s0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    out = dict(mode=mode, d=d_arg, R=R_arg, alpha=sol.alpha,
               status=sol.status,
               n_rows=int(build.A_eq.shape[0]),
               n_vars=int(build.n_vars), nnz=int(build.n_nonzero_A),
               t_build=t_build, t_solve=t_solve,
               t_total=t_build + t_solve, rss_mb=rss)

elif mode == 'SCHUR':
    opts = BuildOptions(R=R_arg, use_z2=True, verbose=True)
    t_b0 = time.time()
    build = build_handelman_lp_schur(d_eff, M_mats_eff, opts)
    t_build = time.time() - t_b0
    n_rows = int(build.A_eq.shape[0])
    print(f"SCHUR_BUILD rows={n_rows} vars={build.n_vars} "
          f"nnz={build.n_nonzero_A} t_build={t_build:.1f}s", flush=True)
    t_s0 = time.time()
    sol = solve_lp(build, solver='mosek', verbose=False)
    t_solve = time.time() - t_s0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    out = dict(mode=mode, d=d_arg, R=R_arg, alpha=sol.alpha,
               status=sol.status, n_rows=n_rows,
               n_vars=int(build.n_vars), nnz=int(build.n_nonzero_A),
               t_build=t_build, t_solve=t_solve,
               t_total=t_build + t_solve, rss_mb=rss)

else:
    raise ValueError(f"unknown mode {mode}")

print('RESULT:', json.dumps(out, default=str), flush=True)
'''

PROBE_PATH = '_schur_probe.py'
with open(PROBE_PATH, 'w') as f:
    f.write(PROBE)


# Stage 1 (correctness) at small R, then Stage 2 (speedup at R=12), then Stage 3 (scaling).
SCHEDULE = [
    (8, 4, 'FULL'),
    (8, 4, 'SCHUR'),
    (8, 8, 'FULL'),
    (8, 8, 'SCHUR'),
    (16, 6, 'FULL'),
    (16, 6, 'SCHUR'),
    (16, 8, 'FULL'),
    (16, 8, 'SCHUR'),
    (16, 10, 'FULL'),
    (16, 10, 'SCHUR'),
    (16, 12, 'FULL'),
    (16, 12, 'SCHUR'),     # the headline test (full = 463s)
    (16, 15, 'SCHUR'),     # scaling
]
PER_TASK_TIMEOUT = 1500


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
    with open('schur_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


# Pair FULL/SCHUR results for correctness comparison + speedup
print('\n\n=== SUMMARY ===\n', flush=True)
print(f"{'case':>10} {'mode':>6} {'alpha':>14} {'rows':>10} {'vars':>10} "
      f"{'nnz':>11} {'wall':>8}", flush=True)
print('-' * 80)
by_case = {}
for r in results:
    if 'error' in r:
        print(f"d{r['d']}R{r['R']:>2} {r['mode']:>6}  ERROR: {r['error']}", flush=True)
        continue
    case = (r['d'], r['R'])
    by_case.setdefault(case, {})[r['mode']] = r
    a = r.get('alpha')
    a_str = f"{a:.10f}" if a is not None else "    N/A    "
    print(f"d{r['d']}R{r['R']:>2} {r['mode']:>6} {a_str:>14} "
          f"{r.get('n_rows','?'):>10} {r.get('n_vars','?'):>10} "
          f"{r.get('nnz','?'):>11} {r.get('t_total',0):>8.1f}", flush=True)

print('\n=== CORRECTNESS + SPEEDUP ===', flush=True)
print(f"{'case':>10} {'full_a':>14} {'schur_a':>14} {'diff':>10} "
      f"{'full_t':>8} {'schur_t':>8} {'speedup':>8}", flush=True)
print('-' * 80)
for case, modes in sorted(by_case.items()):
    f = modes.get('FULL'); s = modes.get('SCHUR')
    if f and s and f.get('alpha') is not None and s.get('alpha') is not None:
        diff = abs(f['alpha'] - s['alpha'])
        speedup = f['t_total'] / s['t_total'] if s['t_total'] > 0 else None
        sp = f"{speedup:.2f}x" if speedup else "  -- "
        print(f"d{case[0]}R{case[1]:>2} {f['alpha']:>14.10f} {s['alpha']:>14.10f} "
              f"{diff:>10.2e} {f['t_total']:>8.1f} {s['t_total']:>8.1f} {sp:>8}",
              flush=True)
    elif s and s.get('alpha') is not None:
        print(f"d{case[0]}R{case[1]:>2}            (no FULL) "
              f"{s['alpha']:>14.10f}              "
              f"            {s['t_total']:>8.1f}", flush=True)
