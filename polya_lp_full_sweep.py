"""Comprehensive (d, R) sweep with per-LP isolation, RAM tracking,
and trend extrapolation.

Each (d, R) runs in its own subprocess (so an OOM doesn't kill the
sweep). Peak RAM is sampled via psutil. The schedule is ordered
smallest-LP-first so we get coverage even if late ones OOM. Within
each d, R is increased; if (d, R) OOMs we skip (d, R+) for that d.

Outputs:
  sweep_results.json : list of records, written incrementally
  sweep_summary.md   : human-readable trend table + projections
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from math import comb

import psutil


HERE = os.path.dirname(os.path.abspath(__file__))
PROBE_PATH = os.path.join(HERE, "_sweep_probe.py")
RESULTS_PATH = os.path.join(HERE, "sweep_results.json")
SUMMARY_PATH = os.path.join(HERE, "sweep_summary.md")


PROBE_SCRIPT = r"""
import sys, time, json, os
sys.path.insert(0, r"{cwd}")
from lasserre.polya_lp.runner import run_one
d = int(sys.argv[1]); R = int(sys.argv[2])
t0 = time.time()
try:
    rec, _, sol = run_one(d=d, R=R, use_z2=True, solver='mosek', verbose=False)
    wall = time.time() - t0
    out = dict(d=d, R=R, alpha=rec.alpha, wall=wall,
               n_eq=rec.n_eq, n_vars=rec.n_vars, nnz=rec.n_nonzero_A,
               build_s=rec.build_wall_s, solve_s=rec.solve_wall_s,
               status='OK')
except MemoryError as e:
    out = dict(d=d, R=R, status='OOM', error='MemoryError',
               wall=time.time()-t0)
except Exception as e:
    msg = str(e)[:300]
    status = 'OOM' if 'large memory' in msg or 'space' in msg.lower() else 'ERROR'
    out = dict(d=d, R=R, status=status, error=msg,
               wall=time.time()-t0)
print('RESULT:', json.dumps(out, default=str), flush=True)
"""


def write_probe():
    with open(PROBE_PATH, "w") as f:
        f.write(PROBE_SCRIPT.format(cwd=HERE.replace("\\", "/")))


def run_one_lp(d: int, R: int, timeout_s: int = 180) -> dict:
    """Run one (d, R) LP in subprocess. Returns dict with timing + RAM.

    On timeout: terminates and returns status='TIMEOUT'.
    """
    proc = subprocess.Popen(
        [sys.executable, "-u", PROBE_PATH, str(d), str(R)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    peak_mem = 0
    start = time.time()
    try:
        p = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        p = None

    while True:
        if proc.poll() is not None:
            break
        if time.time() - start > timeout_s:
            proc.terminate()
            time.sleep(2)
            if proc.poll() is None:
                proc.kill()
            return dict(d=d, R=R, status="TIMEOUT", wall=time.time()-start,
                        peak_mem_mb=peak_mem / 1e6)
        if p is not None:
            try:
                mem = p.memory_info().rss
                for child in p.children(recursive=True):
                    try:
                        mem += child.memory_info().rss
                    except Exception:
                        pass
                if mem > peak_mem:
                    peak_mem = mem
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        time.sleep(0.2)

    stdout, stderr = proc.communicate()
    for line in stdout.splitlines():
        if line.startswith("RESULT:"):
            try:
                r = json.loads(line[len("RESULT:"):].strip())
                r["peak_mem_mb"] = round(peak_mem / 1e6, 1)
                return r
            except json.JSONDecodeError:
                pass

    return dict(d=d, R=R, status="NO_RESULT",
                wall=time.time()-start,
                peak_mem_mb=round(peak_mem / 1e6, 1),
                stderr=stderr[:300])


def estimated_size(d: int, R: int) -> int:
    """Approximate n_eq for sorting (after Z/2 reduction)."""
    d_eff = d // 2
    return comb(d_eff + R, R)


def main():
    write_probe()

    # Schedule: a wide range of d's and R's
    D_VALUES = [4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 40, 48, 56, 64, 80, 96]
    R_VALUES = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

    # Build full schedule, then sort by size
    pairs = [(d, R) for d in D_VALUES for R in R_VALUES if d % 2 == 0]
    pairs.sort(key=lambda x: estimated_size(*x))

    print(f"Total (d, R) pairs in schedule: {len(pairs)}", flush=True)
    print(f"Smallest size: {estimated_size(*pairs[0])}, "
          f"largest: {estimated_size(*pairs[-1])}", flush=True)

    TIME_BUDGET = 3600   # 60 min total
    PER_LP_TIMEOUT = 900   # 15 min per LP
    BIG_LP_TIMEOUT = 900   # 15 min for very large too

    results = []
    oom_d_R = {}   # for each d, smallest R that OOMed
    total_time = 0
    t_start = time.time()

    for d, R in pairs:
        # Skip if a smaller R for this d already OOMed/timeout/error
        if d in oom_d_R and R >= oom_d_R[d]:
            print(f"d={d} R={R}: SKIP (previous R={oom_d_R[d]} failed)",
                  flush=True)
            continue

        # Skip if total budget exceeded
        elapsed = time.time() - t_start
        if elapsed > TIME_BUDGET:
            print(f"\nTotal budget {TIME_BUDGET}s exceeded, stopping.",
                  flush=True)
            break

        size = estimated_size(d, R)
        timeout = BIG_LP_TIMEOUT if size > 1_000_000 else PER_LP_TIMEOUT

        t0 = time.time()
        r = run_one_lp(d, R, timeout_s=timeout)
        per_wall = time.time() - t0
        total_time += per_wall

        # Decorate with extras
        r.setdefault("size_est", size)
        r.setdefault("d_eff", d // 2)

        alpha_str = (f"{r['alpha']:.6f}" if r.get("alpha") is not None
                     else f"({r.get('status', '?')})")
        wall_str = f"{r.get('wall', per_wall):.1f}s"
        mem_str = f"{r.get('peak_mem_mb', 0):.0f}MB"
        size_str = f"n_eq~{size:,}"
        print(f"d={d:>3} R={R:>3} {size_str:>20} "
              f"alpha={alpha_str:>10}  wall={wall_str:>8}  mem={mem_str:>8}  "
              f"status={r.get('status', '?'):>8}  "
              f"(elapsed {elapsed:.0f}s of {TIME_BUDGET})", flush=True)

        results.append(r)

        # Track OOM/timeout/error
        if r.get("status") in ("OOM", "TIMEOUT", "ERROR", "NO_RESULT"):
            oom_d_R[d] = R

        # Save incrementally
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\nTotal sweep wall: {time.time()-t_start:.0f}s")
    print(f"Saved {len(results)} records to {RESULTS_PATH}")

    # Run rigorous mathematical analysis
    print("\nRunning trend analysis...")
    import subprocess
    subprocess.run([sys.executable, os.path.join(HERE, "polya_lp_sweep_analysis.py")])


def write_summary(results):
    """Write a human-readable summary with trends and projections."""
    from collections import defaultdict
    import math

    VAL_D_KNOWN = {
        4: 1.102, 6: 1.171, 8: 1.205, 10: 1.241,
        12: 1.271, 14: 1.284, 16: 1.319,
        32: 1.336, 64: 1.384, 128: 1.420, 256: 1.448,
    }
    TARGET = 1.281

    by_d = defaultdict(list)
    for r in results:
        if r.get("alpha") is not None:
            by_d[r["d"]].append(r)
    for d in by_d:
        by_d[d].sort(key=lambda r: r["R"])

    lines = []
    lines.append(f"# Pólya LP sweep summary\n")
    lines.append(f"Target: certify alpha >= {TARGET}\n")
    lines.append(f"Solver: MOSEK IPM with optimal options\n")
    lines.append(f"Total records: {len(results)}\n\n")

    # Per-d trend table
    lines.append("## Results by d, R\n\n")
    lines.append("| d | R | alpha | gap_to_target | wall_s | mem_MB | n_eq | n_vars | status |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for r in sorted(results, key=lambda r: (r["d"], r["R"])):
        alpha = r.get("alpha")
        if alpha is not None:
            gap = TARGET - alpha
            astr = f"{alpha:.6f}"
            gstr = f"{gap:+.4f}"
        else:
            astr = "—"
            gstr = "—"
        lines.append(
            f"| {r['d']} | {r['R']} | {astr} | {gstr} | "
            f"{r.get('wall', 0):.1f} | {r.get('peak_mem_mb', 0):.0f} | "
            f"{r.get('n_eq', 0):,} | {r.get('n_vars', 0):,} | "
            f"{r.get('status', '?')} |\n"
        )

    # Per-d convergence fits and projections
    lines.append("\n## Convergence fits per d\n\n")
    lines.append("Fit gap(R) = C / R^a from at least 3 (R, alpha) points per d. "
                 "Project R_needed for alpha >= 1.281 (only meaningful when "
                 "val(d) > 1.281).\n\n")
    lines.append("| d | val_d | n_pts | a | C | gap@max_R | R_needed_for_1.281 |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    import numpy as np
    for d in sorted(by_d.keys()):
        runs = by_d[d]
        if len(runs) < 3:
            continue
        val_d = VAL_D_KNOWN.get(d)
        Rs = np.array([r["R"] for r in runs], dtype=float)
        alphas = np.array([r["alpha"] for r in runs])
        if val_d is None:
            val_d_use = max(alphas) + 0.05  # rough upper-bound
        else:
            val_d_use = val_d
        gaps = val_d_use - alphas
        # Filter positive gaps for log fit
        mask = gaps > 1e-12
        if mask.sum() < 2:
            continue
        log_R = np.log(Rs[mask])
        log_g = np.log(gaps[mask])
        if len(log_R) >= 2:
            slope, intercept = np.polyfit(log_R, log_g, 1)
            a = -slope
            C = float(np.exp(intercept))
        else:
            continue
        gap_at_maxR = float(gaps[mask].min())
        if val_d_use < 1.281:
            R_needed_str = f"impossible (val={val_d_use:.3f}<1.281)"
        else:
            eps = val_d_use - TARGET
            if eps <= 0:
                R_needed_str = "—"
            else:
                R_n = (C / eps) ** (1.0 / a)
                R_needed_str = f"{R_n:.1f}"
        lines.append(f"| {d} | {val_d_use:.3f} | {len(runs)} | {a:.3f} | {C:.4f} | "
                     f"{gap_at_maxR:.4f} | {R_needed_str} |\n")

    # Memory and timing trends
    lines.append("\n## Memory / timing scaling per d\n\n")
    lines.append("| d | max_R_solved | wall@max_R | mem@max_R | n_eq@max_R |\n")
    lines.append("|---|---|---|---|---|\n")
    for d in sorted(by_d.keys()):
        last = by_d[d][-1]
        lines.append(
            f"| {d} | {last['R']} | {last.get('wall', 0):.1f}s | "
            f"{last.get('peak_mem_mb', 0):.0f}MB | "
            f"{last.get('n_eq', 0):,} |\n"
        )

    # Recommended target (d, R) combinations
    lines.append("\n## Recommended targets (d, R) to clear alpha >= 1.281\n\n")
    lines.append("Based on the per-d fits, here are (d, R) combinations that "
                 "are projected to clear the threshold:\n\n")
    for d in sorted(VAL_D_KNOWN.keys()):
        if VAL_D_KNOWN[d] < TARGET:
            continue
        if d in by_d and len(by_d[d]) >= 3:
            runs = by_d[d]
            Rs = np.array([r["R"] for r in runs], dtype=float)
            alphas = np.array([r["alpha"] for r in runs])
            gaps = VAL_D_KNOWN[d] - alphas
            mask = gaps > 1e-12
            if mask.sum() >= 2:
                slope, intercept = np.polyfit(np.log(Rs[mask]),
                                               np.log(gaps[mask]), 1)
                a = -slope
                C = float(np.exp(intercept))
                eps = VAL_D_KNOWN[d] - TARGET
                R_needed = (C / eps) ** (1.0 / a)
                last_run = runs[-1]
                lines.append(f"- d={d}: val={VAL_D_KNOWN[d]:.3f}, "
                             f"R_needed≈{R_needed:.1f}, "
                             f"max_R_done={last_run['R']} "
                             f"(α={last_run['alpha']:.4f})\n")

    with open(SUMMARY_PATH, "w") as f:
        f.writelines(lines)
    print(f"Summary -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
