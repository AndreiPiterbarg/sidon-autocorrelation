"""Practical benchmark on RTX 3080 laptop: how big can we go with the LP
on CPU (HiGHS / MOSEK), and dump bigger ones to MPS for cloud.

Tests:
 - d=8 baseline at R=4..16 (small, fast)
 - d=16 at R=4..8
 - d=32 at R=4..6
 - d=64 at R=4..6 (target — 2.7M constraints at R=6)

Reports for each (d, R):
 - LP build time
 - LP size (n_eq, n_vars, nnz)
 - HiGHS solve time + alpha
 - MOSEK solve time + alpha (if HiGHS slow)
 - MPS file size (for cloud transfer)
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from lasserre.polya_lp.runner import VAL_D_KNOWN
from lasserre.polya_lp.build import (
    BuildOptions, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.solve import solve_lp
from lasserre.polya_lp.symmetry import (
    project_window_set_to_z2_rescaled, z2_dim,
)
from lasserre.polya_lp.mps_export import write_buildresult_mps


# NOTE: keep the search short on laptop; full benchmark goes to cloud
SCHEDULE = [
    (8,  4),
    (8,  8),
    (8, 12),
    (16, 4),
    (16, 6),
    (16, 8),
    (32, 4),
    (32, 6),
    (64, 4),
    (64, 6),
]


def main():
    out_dir = "polya_lp_mps"
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for d, R in SCHEDULE:
        print(f"\n=== d={d}, R={R} ===", flush=True)
        record = {"d": d, "R": R, "val_d_known": VAL_D_KNOWN.get(d)}

        # Build (with Z/2)
        t0 = time.time()
        _, M_mats_orig = build_window_matrices(d)
        M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats_orig, d)
        d_eff = z2_dim(d)
        opts = BuildOptions(R=R, use_z2=True, verbose=False)
        try:
            build = build_handelman_lp(d_eff, M_mats_eff, opts)
        except MemoryError:
            print(f"  BUILD OOM", flush=True)
            record["status"] = "BUILD_OOM"
            results.append(record); continue
        t_build = time.time() - t0
        n_eq = build.A_eq.shape[0]
        n_vars = build.n_vars
        nnz = build.A_eq.nnz
        record.update({
            "n_eq": n_eq, "n_vars": n_vars, "nnz": nnz,
            "build_wall_s": round(t_build, 3),
        })
        print(f"  build: n_eq={n_eq:,}, n_vars={n_vars:,}, nnz={nnz:,}, "
              f"{t_build:.2f}s", flush=True)

        # Skip MASSIVE LPs for solve, but DO export MPS
        size_threshold_n_vars = 2_500_000
        will_solve = n_vars < size_threshold_n_vars

        # Export MPS
        mps_path = os.path.join(out_dir, f"polya_d{d}_R{R}.mps")
        t0 = time.time()
        try:
            write_buildresult_mps(build, mps_path, sense="MIN", name=f"polya_d{d}_R{R}")
            mps_size_mb = os.path.getsize(mps_path) / 1e6
            t_mps = time.time() - t0
            record.update({
                "mps_path": mps_path,
                "mps_size_mb": round(mps_size_mb, 2),
                "mps_wall_s": round(t_mps, 2),
            })
            print(f"  MPS: {mps_path} ({mps_size_mb:.1f} MB, {t_mps:.1f}s)", flush=True)
        except Exception as e:
            print(f"  MPS export FAILED: {e}", flush=True)
            record["mps_error"] = str(e)

        # Solve via HiGHS
        if will_solve:
            print(f"  Solving with HiGHS (n_vars={n_vars:,})...", flush=True)
            t0 = time.time()
            try:
                sol = solve_lp(build)
                t_solve = time.time() - t0
                record.update({
                    "alpha": sol.alpha,
                    "solve_status": sol.status,
                    "solve_wall_s": round(t_solve, 2),
                })
                print(f"  alpha={sol.alpha}, wall={t_solve:.1f}s", flush=True)
            except Exception as e:
                t_solve = time.time() - t0
                print(f"  HiGHS FAILED after {t_solve:.0f}s: {type(e).__name__}: {e}",
                      flush=True)
                record.update({"solve_status": "ERROR",
                               "solve_error": str(e)[:200]})
        else:
            print(f"  SKIP solve (n_vars={n_vars:,} > {size_threshold_n_vars}); "
                  f"MPS exported for cloud run.", flush=True)
            record["solve_status"] = "SKIPPED_TOO_BIG"

        results.append(record)

        # Save running results
        with open(os.path.join(out_dir, "benchmark.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    print("\n\n=== SUMMARY ===", flush=True)
    print(f"{'d':>3} {'R':>3} {'n_eq':>10} {'n_vars':>10} {'nnz':>12} "
          f"{'build_s':>9} {'solve_s':>9} {'alpha':>10} {'val_d':>8}",
          flush=True)
    for r in results:
        alpha = r.get("alpha")
        alpha_str = f"{alpha:.6f}" if alpha is not None else (
            r.get("solve_status", "?")[:10])
        val_d = r.get("val_d_known", "?")
        print(f"{r['d']:>3} {r['R']:>3} {r.get('n_eq', 0):>10,} "
              f"{r.get('n_vars', 0):>10,} {r.get('nnz', 0):>12,} "
              f"{r.get('build_wall_s', 0):>9.1f} "
              f"{r.get('solve_wall_s', 0):>9.1f} "
              f"{alpha_str:>10} {str(val_d):>8}", flush=True)

    print(f"\n\nMPS files written to {out_dir}/", flush=True)
    print("Send these to a cloud GPU + cuOpt for d=64 R=8:", flush=True)
    print(f"  pip install cuopt-server-cu12 cuopt-mps-parser", flush=True)
    print(f"  cuopt -i polya_d64_R6.mps -t lp", flush=True)


if __name__ == "__main__":
    main()
