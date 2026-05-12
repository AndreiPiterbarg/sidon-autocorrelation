"""Breakthrough attempt: clear C_{1a} > 1.2802 via Polya/Handelman LP at d=16.

Schedule (in increasing R, increasing predicted alpha):
  d=16, R=28  -- M1 prediction alpha ~ 1.282 (just above 1.2802 LB; sanity)
  d=16, R=30  -- M1 prediction alpha ~ 1.285 (safe margin +0.005)
  d=16, R=33  -- M1 prediction alpha ~ 1.288 (margin +0.008)
  d=16, R=35  -- M1 prediction alpha ~ 1.290 (margin +0.010, the keeper)

Each run is isolated as a subprocess so a single crash/hang doesn't tank the
others. Per-task wall budget = 6 hours (the largest case may take 2-6 h).

CONFIG NOTES:
  * eliminate_c_slacks = False : with the eliminated form, MOSEK with
    solve_form=DUAL stalls (776M factor nonzeros at d=16 R=12). The
    original equality form solves cleanly.
  * Z/2 reduction ON.
  * MOSEK IPM with the tuned options (presolve eliminator cascading,
    basis identification OFF, num_threads = 0 = use all cores).

OUTPUT:
  * breakthrough.log   - human-readable progress (tail this)
  * breakthrough_results.json - cumulative JSON, written after each case
"""
import sys, os, json, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


PROBE = """
import sys, time, json, resource, threading, os
sys.path.insert(0, '/home/ubuntu/sidon')
from lasserre.polya_lp.build import (
    build_handelman_lp, BuildOptions, build_window_matrices,
)
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim
from lasserre.polya_lp.solve import solve_lp

d = int(sys.argv[1]); R = int(sys.argv[2])

# OS-level heartbeat: every 30s print process RSS, %CPU, threads. This way
# even if MOSEK is silent for long periods we still see *something* and can
# tell whether the process is making progress (RSS climbing or core count
# changing) versus genuinely stuck.
def heartbeat():
    pid = os.getpid()
    last_rss = 0
    t0_hb = time.time()
    while True:
        try:
            with open(f'/proc/{pid}/status') as f:
                stat = f.read()
            with open(f'/proc/{pid}/stat') as f:
                fields = f.read().split()
            rss_kb = int([l.split()[1] for l in stat.splitlines() if l.startswith('VmRSS:')][0])
            threads = int([l.split()[1] for l in stat.splitlines() if l.startswith('Threads:')][0])
            n_running_threads = 0
            try:
                for tid in os.listdir(f'/proc/{pid}/task'):
                    with open(f'/proc/{pid}/task/{tid}/stat') as f:
                        s = f.read().split()
                    if s[2] == 'R':
                        n_running_threads += 1
            except Exception:
                pass
            elapsed = time.time() - t0_hb
            d_rss = rss_kb - last_rss
            print(f'HEARTBEAT t={elapsed:.0f}s rss={rss_kb/1024.0/1024.0:.2f}GB '
                  f'd_rss={d_rss/1024.0:+.0f}MB threads={threads} '
                  f'running={n_running_threads}', flush=True)
            last_rss = rss_kb
        except Exception as e:
            print(f'HEARTBEAT_ERR: {e}', flush=True)
        time.sleep(30)
threading.Thread(target=heartbeat, daemon=True).start()

t0 = time.time()
_, M_mats = build_window_matrices(d)
M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats, d)
d_eff = z2_dim(d)
opts = BuildOptions(R=R, use_z2=True, eliminate_c_slacks=False)
t_setup = time.time() - t0

t0 = time.time()
build = build_handelman_lp(d_eff, M_mats_eff, opts)
t_build = time.time() - t0
n_rows = (build.A_eq.shape[0] if build.A_eq is not None else 0) + \\
         (build.A_ub.shape[0] if build.A_ub is not None else 0)

print(f'BUILD_DONE rows={n_rows:,} vars={build.n_vars:,} nnz={build.n_nonzero_A:,} '
      f't_build={t_build:.1f}s', flush=True)

t0 = time.time()
sol = solve_lp(build, solver='mosek', verbose=True)
t_solve = time.time() - t0
rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

out = dict(d=d, R=R, d_eff=d_eff, alpha=sol.alpha, status=sol.status,
           n_W=len(M_mats_eff), n_rows=n_rows, n_vars=build.n_vars,
           nnz=build.n_nonzero_A,
           t_setup=t_setup, t_build=t_build, t_solve=t_solve,
           rss_mb=rss_mb)
print('RESULT:', json.dumps(out, default=str), flush=True)
"""

PROBE_PATH = '_breakthrough_probe.py'
with open(PROBE_PATH, 'w') as f:
    f.write(PROBE)


# (d, R) schedule. Increasing R means increasing predicted alpha and harder LP.
SCHEDULE = [
    (16, 28),
    (16, 30),
    (16, 33),
    (16, 35),
]
# Per-task wall budget in seconds. R=35 may push 4-6 hours.
PER_TASK_TIMEOUT = 6 * 3600

CURRENT_LB = 1.2802

results = []
for d, R in SCHEDULE:
    print(f'\n========== d={d}, R={R} ==========', flush=True)
    t_start = time.time()
    # Stream the probe's stdout live to our log (and keep buffered copy for
    # parsing RESULT line). This way we never lose MOSEK log output, even
    # if the case is killed or we hit a per-task timeout.
    proc = subprocess.Popen(
        ['python3', '-u', PROBE_PATH, str(d), str(R)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    captured_lines = []
    timed_out = False
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            captured_lines.append(line)
            if time.time() - t_start > PER_TASK_TIMEOUT:
                timed_out = True
                proc.kill()
                break
        proc.wait(timeout=60)
    except Exception as e:
        print(f'  STREAM_ERROR: {e}', flush=True)
        try: proc.kill()
        except Exception: pass
    full_stdout = ''.join(captured_lines)
    wall = time.time() - t_start
    if timed_out:
        print(f'  TIMEOUT after {wall:.0f}s (live log preserved above)',
              flush=True)
        results.append(dict(d=d, R=R, error='timeout', wall=wall))
        with open('breakthrough_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        continue
    if True:
        result = None
        for line in full_stdout.splitlines():
            if line.startswith('RESULT:'):
                result = json.loads(line[len('RESULT:'):].strip())
        if result is not None:
            a = float(result['alpha']) if result.get('alpha') is not None else None
            cleared = (a is not None and a > CURRENT_LB)
            print(f"\n  alpha={a}  status={result['status']}  "
                  f"build={result['t_build']:.1f}s  solve={result['t_solve']:.1f}s  "
                  f"rows={result['n_rows']:,}  nnz={result['nnz']:,}  "
                  f"rss={result['rss_mb']:.0f}MB", flush=True)
            if cleared:
                margin = a - CURRENT_LB
                print(f"  ** CLEARS 1.2802 by +{margin:.6f} **", flush=True)
            elif a is not None:
                print(f"  short by {CURRENT_LB - a:+.6f}", flush=True)
            results.append(result)
        else:
            err = full_stdout[-1500:]
            print(f'  NO RESULT after {wall:.0f}s. tail: {err}', flush=True)
            results.append(dict(d=d, R=R, error='no_result', wall=wall, tail=err))

    with open('breakthrough_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


print('\n\n=== SUMMARY ===', flush=True)
print(f"{'d':>3} {'R':>3} {'alpha':>11} {'vs_1.2802':>12} "
      f"{'rows':>11} {'nnz':>13} {'build_s':>9} {'solve_s':>10} {'rss_GB':>8}",
      flush=True)
print('-' * 100)
best_alpha = -1.0
for r in results:
    if 'error' in r:
        print(f"{r['d']:>3} {r['R']:>3}  {r['error']}", flush=True)
        continue
    if r.get('alpha') is None:
        print(f"{r['d']:>3} {r['R']:>3}  status={r.get('status', '?')}",
              flush=True)
        continue
    a = float(r['alpha'])
    if a > best_alpha:
        best_alpha = a
    delta = a - CURRENT_LB
    print(f"{r['d']:>3} {r['R']:>3} {a:>11.6f} {delta:>+12.6f} "
          f"{r['n_rows']:>11,} {r['nnz']:>13,} "
          f"{r['t_build']:>9.1f} {r['t_solve']:>10.1f} "
          f"{r['rss_mb']/1024.0:>8.1f}", flush=True)

print('', flush=True)
if best_alpha > CURRENT_LB:
    print(f"** BREAKTHROUGH: best alpha = {best_alpha:.6f}, "
          f"new LB beats 1.2802 by +{best_alpha - CURRENT_LB:.6f} **",
          flush=True)
else:
    print(f"No case cleared 1.2802. Best alpha = {best_alpha:.6f}.",
          flush=True)
