"""Benchmark MOSEK IPM (with optimal options) on Polya LPs.

Per-iteration timeout via subprocess so a slow case doesn't kill everything.
"""
import sys, os, json, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


PROBE_SCRIPT = """
import sys, time
sys.path.insert(0, r'C:/Users/andre/OneDrive - PennO365/Desktop/compact_sidon')
from lasserre.polya_lp.runner import run_one
import json

d = int(sys.argv[1]); R = int(sys.argv[2])
t0 = time.time()
rec, _, _ = run_one(d=d, R=R, use_z2=True, solver='mosek', verbose=False)
wall = time.time() - t0
out = dict(d=d, R=R, alpha=rec.alpha, wall=wall, n_eq=rec.n_eq, n_vars=rec.n_vars,
           build_s=rec.build_wall_s, solve_s=rec.solve_wall_s)
print('RESULT:', json.dumps(out, default=str), flush=True)
"""

SCRIPT_PATH = "_mosek_ipm_probe.py"

with open(SCRIPT_PATH, "w") as f:
    f.write(PROBE_SCRIPT)


SCHEDULE = [
    (8, 4),  (8, 8),  (8, 12),
    (16, 4), (16, 6), (16, 8),
    (32, 4), (32, 6),
    (64, 4), (64, 6),
]

PER_TASK_TIMEOUT = 600  # 10 min per LP

results = []
for d, R in SCHEDULE:
    print(f"\n=== d={d}, R={R} ===", flush=True)
    t_start = time.time()
    try:
        proc = subprocess.run(
            ["python", "-u", SCRIPT_PATH, str(d), str(R)],
            capture_output=True, text=True, timeout=PER_TASK_TIMEOUT,
        )
        wall = time.time() - t_start
        # Parse last line for RESULT
        result = None
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT:"):
                result = json.loads(line[len("RESULT:"):].strip())
        if result is not None:
            print(f"  alpha={result['alpha']:.6f}  wall={result['wall']:.2f}s  "
                  f"n_eq={result['n_eq']:,}  n_vars={result['n_vars']:,}  "
                  f"build={result['build_s']:.2f}s  solve={result['solve_s']:.2f}s",
                  flush=True)
            results.append(result)
        else:
            print(f"  NO RESULT after {wall:.0f}s. stderr: {proc.stderr[:200]}",
                  flush=True)
            results.append(dict(d=d, R=R, error="no_result", wall=wall,
                               stderr=proc.stderr[:200]))
    except subprocess.TimeoutExpired:
        wall = time.time() - t_start
        print(f"  TIMEOUT after {wall:.0f}s", flush=True)
        results.append(dict(d=d, R=R, error="timeout", wall=wall))

with open("mosek_ipm_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n\n=== SUMMARY ===", flush=True)
for r in results:
    if "error" in r:
        print(f"  d={r['d']} R={r['R']}: {r['error']} after {r.get('wall', 0):.0f}s",
              flush=True)
    else:
        print(f"  d={r['d']} R={r['R']}: alpha={r['alpha']:.6f}  "
              f"wall={r['wall']:.1f}s  n_eq={r['n_eq']:,}", flush=True)
