"""High-d benchmark for the GPU pod.

For each (d, R) in the grid:
  1. Build dual epigraph LP (vectorized).
  2. Solve via the requested backend(s): cuOpt, ortools PDLP, MOSEK.
  3. Capture per-backend wall, KKT, alpha, peak memory.
  4. Compare to MOSEK monolithic primal as ground truth (when feasible).

This is the script you run on the B300 pod. It:
  - Auto-detects available solvers (cuOpt, ortools, MOSEK).
  - Skips backends that aren't installed (no hard fail).
  - Skips MOSEK ground-truth on instances projected to OOM.
  - Writes JSON results AFTER EACH instance (so the run is restartable).

Usage:
    python _tier_dual_pod_bench.py                 # full default grid
    python _tier_dual_pod_bench.py --quick         # short smoke
    python _tier_dual_pod_bench.py --grid GRID     # custom grid
        where GRID = "8,4 12,8 16,12 24,8" etc.
    python _tier_dual_pod_bench.py --out result.json
    python _tier_dual_pod_bench.py --backends cuopt,ortools,mosek
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =====================================================================
# Backend detection
# =====================================================================

def detect_backends() -> dict:
    info = {}
    # cuOpt
    try:
        from lasserre.polya_lp.tier_dual.solve_cuopt import (
            CUOPT_AVAILABLE, CUOPT_IMPORT_ERROR, _CUOPT_API,
        )
        info["cuopt"] = bool(CUOPT_AVAILABLE)
        info["cuopt_api"] = _CUOPT_API
        if not CUOPT_AVAILABLE:
            info["cuopt_error"] = CUOPT_IMPORT_ERROR
    except Exception as e:
        info["cuopt"] = False
        info["cuopt_error"] = str(e)

    # ortools
    try:
        from ortools.pdlp.python import pdlp  # noqa: F401
        info["ortools"] = True
    except Exception as e:
        info["ortools"] = False
        info["ortools_error"] = str(e)

    # MOSEK
    try:
        import mosek  # noqa: F401
        info["mosek"] = True
        info["mosek_version"] = mosek.Env.getversion()
    except Exception as e:
        info["mosek"] = False
        info["mosek_error"] = str(e)

    # GPU
    try:
        import torch
        info["gpu"] = torch.cuda.is_available()
        if info["gpu"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        info["gpu"] = False
    return info


# =====================================================================
# Memory probe
# =====================================================================

def probe_rss_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except Exception:
            return -1.0


def probe_gpu_mem_mb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return None


# =====================================================================
# Build + solve
# =====================================================================

def run_one(d: int, R: int, backends: List[str], do_truth: bool,
            mosek_truth_max_n_vars: int = 5_000_000,
            cuopt_tol: float = 1e-6,
            ortools_tol: float = 1e-6,
            time_limit_per_solve: float = 600.0,
            iter_limit: int = 200000,
            verbose: bool = True) -> dict:

    from lasserre.polya_lp.build import (
        BuildOptions, build_handelman_lp, build_window_matrices,
    )
    from lasserre.polya_lp.solve import solve_lp
    from lasserre.polya_lp.symmetry import (
        project_window_set_to_z2_rescaled, z2_dim,
    )
    from lasserre.polya_lp.tier_dual.build_dual_epi_fast import (
        build_dual_epi_fast,
    )

    record: dict = {"d": d, "R": R, "ok": True}
    record["backends_attempted"] = list(backends)
    print(f"\n=== d={d} R={R} ===", flush=True)

    # Window matrices (Z/2 reduced)
    t = time.time()
    _, M_full = build_window_matrices(d)
    M_eff, _ = project_window_set_to_z2_rescaled(M_full, d)
    d_eff = z2_dim(d)
    record["d_eff"] = d_eff
    record["n_W"] = len(M_eff)
    record["wall_setup_ms"] = (time.time() - t) * 1000

    # Fast dual-epigraph build
    t = time.time()
    try:
        epi = build_dual_epi_fast(d_eff, M_eff, R, verbose=False)
    except Exception as e:
        record["ok"] = False
        record["build_error"] = str(e)
        record["build_traceback"] = traceback.format_exc()
        return record
    wall_build = time.time() - t
    record["dual_n_vars"] = epi.n_vars
    record["dual_n_eq"] = epi.n_eq
    record["dual_n_ub"] = epi.n_ub
    record["dual_nnz_eq"] = int(epi.A_eq.nnz)
    record["dual_nnz_ub"] = int(epi.A_ub.nnz)
    record["dual_nnz_total"] = int(epi.A_eq.nnz + epi.A_ub.nnz)
    record["wall_build_ms"] = wall_build * 1000
    record["rss_after_build_mb"] = probe_rss_mb()
    print(f"  build : {epi.n_vars} vars  {epi.n_eq+epi.n_ub} rows  "
          f"{epi.A_eq.nnz + epi.A_ub.nnz} nnz  {wall_build*1000:.0f}ms",
          flush=True)

    # ---- ground truth via primal MOSEK if feasible ----
    record["alpha_truth"] = None
    record["wall_truth_ms"] = None
    if do_truth and "mosek" in backends:
        # Build the primal first
        try:
            primal_build = build_handelman_lp(
                d_eff, M_eff, BuildOptions(R=R, use_z2=True),
            )
            if primal_build.n_vars > mosek_truth_max_n_vars:
                print(f"  truth : SKIPPED (primal n_vars={primal_build.n_vars} > "
                      f"max {mosek_truth_max_n_vars})", flush=True)
                record["truth_skipped_reason"] = "primal too large for MOSEK"
            else:
                t = time.time()
                sol = solve_lp(primal_build, solver="mosek", tol=1e-9,
                               verbose=False)
                wall_truth = time.time() - t
                record["alpha_truth"] = sol.alpha
                record["wall_truth_ms"] = wall_truth * 1000
                record["primal_n_vars"] = primal_build.n_vars
                print(f"  truth : alpha={sol.alpha:.10f}  "
                      f"wall={wall_truth*1000:.0f}ms", flush=True)
        except Exception as e:
            record["truth_error"] = str(e)
            print(f"  truth : ERROR {e}", flush=True)

    # ---- backend solves ----
    for backend in backends:
        t = time.time()
        try:
            if backend == "cuopt":
                from lasserre.polya_lp.tier_dual.solve_cuopt import (
                    CUOPT_AVAILABLE, solve_dual_cuopt,
                )
                if not CUOPT_AVAILABLE:
                    print(f"  cuopt : SKIPPED (not installed)", flush=True)
                    continue
                # reset peak GPU mem
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
                sol = solve_dual_cuopt(
                    epi, tol=cuopt_tol, iter_limit=iter_limit,
                    time_sec_limit=time_limit_per_solve,
                    log_level=1, is_epigraph=True,
                )
                gpu_peak = probe_gpu_mem_mb()
                if gpu_peak is not None:
                    record[f"{backend}_gpu_peak_mb"] = gpu_peak
            elif backend == "ortools":
                from lasserre.polya_lp.tier_dual.solve_ortools_pdlp import (
                    solve_dual_ortools_pdlp,
                )
                sol = solve_dual_ortools_pdlp(
                    epi, tol=ortools_tol, iter_limit=iter_limit,
                    time_sec_limit=time_limit_per_solve,
                    verbosity=0, is_epigraph=True,
                )
            elif backend == "mosek_dual":
                from lasserre.polya_lp.tier_dual.build_dual_epi import (
                    solve_epi_mosek,
                )
                sol = solve_epi_mosek(epi, tol=1e-9, verbose=False)
            else:
                print(f"  {backend}: unknown backend, SKIPPED", flush=True)
                continue
            wall = time.time() - t

            diff_truth = (abs(sol.alpha - record["alpha_truth"])
                          if (sol.alpha is not None and
                              record["alpha_truth"] is not None)
                          else None)
            record[f"{backend}_alpha"] = sol.alpha
            record[f"{backend}_wall_ms"] = wall * 1000
            record[f"{backend}_status"] = (
                str(sol.raw_status) if sol.raw_status is not None
                else sol.status
            )
            record[f"{backend}_converged"] = bool(sol.converged)
            record[f"{backend}_kkt"] = sol.kkt
            record[f"{backend}_diff_truth"] = diff_truth
            print(f"  {backend:>6s}: alpha={sol.alpha}  "
                  f"diff={diff_truth}  conv={sol.converged}  "
                  f"wall={wall*1000:.0f}ms  status={sol.status}",
                  flush=True)
        except Exception as e:
            wall = time.time() - t
            record[f"{backend}_error"] = str(e)
            record[f"{backend}_traceback"] = traceback.format_exc()
            print(f"  {backend}: ERROR {type(e).__name__}: {e}", flush=True)

    return record


# =====================================================================
# Main
# =====================================================================

DEFAULT_GRID = [
    # baselines (verify pipeline)
    (8, 4), (8, 8),
    (10, 6),
    (12, 6), (12, 8),
    # mid range -- where MOSEK is still comfy
    (16, 6), (16, 8), (16, 10), (16, 12),
    # high-d -- where GPU PDLP wins start to materialize
    (16, 16), (16, 20),
    (20, 6), (20, 8), (20, 10),
    (24, 6), (24, 8),
    # stretch -- target territory
    (16, 27),       # the breakthrough target
    (24, 12),       # approx 2.7B nnz, ~22 GB on B300
    (32, 8),        # high d, moderate R
]


QUICK_GRID = [(8, 4), (12, 6), (16, 8), (16, 12)]


def _parse_grid(s: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for tok in s.replace(",", " ").replace(";", " ").split():
        if "_" not in tok and "-" not in tok:
            # support "16,8" originally (already split above) --
            # use trailing comma-pairing
            continue
    # Try simpler form: "d,R d,R d,R"
    tokens = [t for t in s.replace(";", " ").split() if t]
    out = []
    for t in tokens:
        a, b = t.split(",")
        out.append((int(a), int(b)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=str, default="",
                    help='comma-paired (d,R) tokens; e.g. "8,4 12,8 16,16"')
    ap.add_argument("--quick", action="store_true", help="short smoke grid")
    ap.add_argument("--out", type=str, default="_tier_dual_pod_bench.json")
    ap.add_argument("--backends", type=str, default="cuopt,ortools,mosek_dual",
                    help="comma list: cuopt, ortools, mosek_dual")
    ap.add_argument("--no-truth", action="store_true",
                    help="skip MOSEK primal ground-truth (faster at high d)")
    ap.add_argument("--mosek-truth-max-vars", type=int, default=5_000_000,
                    help="skip MOSEK truth if primal n_vars exceeds this")
    ap.add_argument("--cuopt-tol", type=float, default=1e-6)
    ap.add_argument("--ortools-tol", type=float, default=1e-6)
    ap.add_argument("--iter-limit", type=int, default=200000)
    ap.add_argument("--time-limit", type=float, default=600.0,
                    help="per-solve wall time limit (s)")
    args = ap.parse_args()

    if args.grid:
        grid = _parse_grid(args.grid)
    elif args.quick:
        grid = QUICK_GRID
    else:
        grid = DEFAULT_GRID

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    do_truth = not args.no_truth

    info = detect_backends()
    info["python"] = sys.version.split()[0]
    info["platform"] = platform.platform()
    info["grid"] = grid
    info["backends"] = backends
    info["do_truth"] = do_truth
    print("=== Backend detection ===")
    for k, v in info.items():
        print(f"  {k}: {v}")

    records = []
    out_path = args.out
    for (d, R) in grid:
        rec = run_one(
            d, R, backends, do_truth=do_truth,
            mosek_truth_max_n_vars=args.mosek_truth_max_vars,
            cuopt_tol=args.cuopt_tol,
            ortools_tol=args.ortools_tol,
            time_limit_per_solve=args.time_limit,
            iter_limit=args.iter_limit,
        )
        records.append(rec)
        # Write incrementally (run is restartable)
        try:
            with open(out_path, "w") as f:
                json.dump({"info": info, "records": records}, f,
                          indent=2, default=str)
        except Exception:
            pass

    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
