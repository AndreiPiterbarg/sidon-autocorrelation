"""Deep parameter sweep of V3 SDP at finely-spaced M values and high k.

Maps lambda^{3pt}(M, k) to find the exact M* where the bound crosses 1.2802.
Uses build_2pt_bump / build_3pt_bump from lasserre.threepoint_alternatives.

Saves raw results to results/v3_deep_sweep.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from lasserre.threepoint_alternatives import build_2pt_bump, build_3pt_bump
from lasserre.threepoint_full import solve, BuildInfo


RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_PATH = RESULTS_DIR / "v3_deep_sweep.json"
LOG_PATH = RESULTS_DIR / "v3_deep_sweep.log"

THRESHOLD = 1.2802

# MOSEK params for high precision (push conditioning).
MOSEK_PARAMS_HIGHPREC = {
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-10,
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-10,
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-10,
    "MSK_IPAR_PRESOLVE_USE": "MSK_PRESOLVE_MODE_ON",
    "MSK_IPAR_INTPNT_SCALING": "MSK_SCALING_FREE",
}


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")


def run_pair(k: int, N: int, M: float, *, epsilon: float = 0.1,
              verbose: bool = False, mosek_params: Optional[Dict] = None) -> Dict:
    """Run 2pt and 3pt bump SDPs at L^infty bound M (orig coords).

    For 2pt: applies l_inf_M_rho2_orig = M^2 (rho^(2) bound).
    For 3pt: applies BOTH l_inf_M_rho2_orig = M^2 AND l_inf_M_rho3_orig = M^3
             (full constraint set per the existing report).
    """
    out: Dict = dict(k=k, N=N, M=M, epsilon=epsilon)

    # 2pt baseline (with 2D L^infty)
    try:
        t_b0 = time.time()
        p2, info2, _ = build_2pt_bump(k, N, epsilon=epsilon, l_inf_M_rho2_orig=M ** 2)
        t_b = time.time() - t_b0
        log(f"  [2pt] k={k} N={N} M={M:.4f}  built in {t_b:.1f}s, blocks={info2.block_sizes}")
        info2 = solve(p2, info2, solver="MOSEK", verbose=verbose,
                      mosek_params=mosek_params)
        log(f"  [2pt] -> status={info2.status}, lambda={info2.objective}, "
            f"solve {info2.solve_seconds:.1f}s")
        out["lambda_2pt"] = (float(info2.objective)
                              if info2.objective is not None else None)
        out["status_2pt"] = info2.status
        out["solve_seconds_2pt"] = info2.solve_seconds
    except Exception as exc:
        log(f"  [2pt] ERROR k={k} N={N} M={M:.4f}: {type(exc).__name__}: {exc}")
        out["lambda_2pt"] = None
        out["status_2pt"] = f"ERROR: {type(exc).__name__}"
        out["solve_seconds_2pt"] = 0.0

    # 3pt lift (with 2D + 3D L^infty)
    try:
        t_b0 = time.time()
        p3, info3, _ = build_3pt_bump(
            k, N, epsilon=epsilon,
            l_inf_M_rho2_orig=M ** 2,
            l_inf_M_rho3_orig=M ** 3,
        )
        t_b = time.time() - t_b0
        log(f"  [3pt] k={k} N={N} M={M:.4f}  built in {t_b:.1f}s, blocks={info3.block_sizes}")
        info3 = solve(p3, info3, solver="MOSEK", verbose=verbose,
                      mosek_params=mosek_params)
        log(f"  [3pt] -> status={info3.status}, lambda={info3.objective}, "
            f"solve {info3.solve_seconds:.1f}s")
        out["lambda_3pt"] = (float(info3.objective)
                              if info3.objective is not None else None)
        out["status_3pt"] = info3.status
        out["solve_seconds_3pt"] = info3.solve_seconds
    except Exception as exc:
        log(f"  [3pt] ERROR k={k} N={N} M={M:.4f}: {type(exc).__name__}: {exc}")
        out["lambda_3pt"] = None
        out["status_3pt"] = f"ERROR: {type(exc).__name__}"
        out["solve_seconds_3pt"] = 0.0

    return out


def append_record(rec: Dict) -> None:
    """Append rec to OUT_PATH (atomically as a list-of-records)."""
    if OUT_PATH.exists():
        try:
            data = json.load(open(OUT_PATH))
        except Exception:
            data = []
    else:
        data = []
    data.append(rec)
    tmp = OUT_PATH.with_suffix(".tmp.json")
    json.dump(data, open(tmp, "w"), indent=2)
    tmp.replace(OUT_PATH)


def existing_records() -> List[Dict]:
    if OUT_PATH.exists():
        try:
            return json.load(open(OUT_PATH))
        except Exception:
            return []
    return []


def already_done(records: List[Dict], k: int, N: int, M: float) -> bool:
    for r in records:
        if r.get("k") == k and r.get("N") == N and abs(r.get("M", -1) - M) < 1e-9:
            if r.get("lambda_3pt") is not None or r.get("status_3pt", "").startswith("ERROR"):
                return True
    return False


def find_crossover_M(records: List[Dict], k: int, N: int, threshold: float = THRESHOLD) \
        -> Tuple[Optional[float], Optional[float], Optional[Tuple[float, float]]]:
    """Linear-interp the M at which lambda_3pt crosses threshold.

    Returns (M_star, max_M_above, (M_low_above, M_high_below)).
    """
    rows = sorted(
        [r for r in records if r.get("k") == k and r.get("N") == N
         and r.get("lambda_3pt") is not None],
        key=lambda r: r["M"],
    )
    if len(rows) < 2:
        return None, None, None
    max_M_above = None
    for r in rows:
        if r["lambda_3pt"] >= threshold:
            if max_M_above is None or r["M"] > max_M_above:
                max_M_above = r["M"]
    # Find adjacent (above, below) pair
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        if a["lambda_3pt"] >= threshold and b["lambda_3pt"] < threshold:
            la, lb = a["lambda_3pt"], b["lambda_3pt"]
            t = (la - threshold) / (la - lb) if (la != lb) else 0.0
            M_star = a["M"] + t * (b["M"] - a["M"])
            return M_star, max_M_above, (a["M"], b["M"])
    return None, max_M_above, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=str, default="7,8,9",
                    help="Comma-separated k values (N=k).")
    ap.add_argument("--Ms", type=str,
                    default="2.10,2.12,2.14,2.15,2.16,2.17,2.18,2.19,2.20",
                    help="Comma-separated M values (L^inf bound on f, original coords).")
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--high_prec", action="store_true",
                    help="Use tightened MOSEK tolerances for high-k conditioning.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip (k, N, M) tuples already saved.")
    ap.add_argument("--bisect", action="store_true",
                    help="After main sweep, bisect for crossover M*.")
    ap.add_argument("--bisect_tol", type=float, default=0.005)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ks = [int(s) for s in args.ks.split(",")]
    Ms = [float(s) for s in args.Ms.split(",")]
    mosek_params = MOSEK_PARAMS_HIGHPREC if args.high_prec else None

    log(f"=== V3 deep sweep starting at {time.ctime()} ===")
    log(f"  ks = {ks}")
    log(f"  Ms = {Ms}")
    log(f"  epsilon = {args.epsilon}, high_prec = {args.high_prec}")

    records = existing_records()
    log(f"  existing records: {len(records)}")

    for k in ks:
        N = k
        for M in Ms:
            if args.skip_existing and already_done(records, k, N, M):
                log(f"\n[skip] (k={k}, N={N}, M={M:.4f}) already done")
                continue
            log(f"\n=== (k={k}, N={N}, M={M:.4f}) ===")
            rec = run_pair(k, N, M, epsilon=args.epsilon,
                           verbose=args.verbose, mosek_params=mosek_params)
            append_record(rec)
            records = existing_records()

    # Bisection for crossover
    if args.bisect:
        log("\n=== BISECTION for crossover M* ===")
        for k in ks:
            N = k
            M_star, M_above, bracket = find_crossover_M(records, k, N)
            if bracket is None:
                log(f"  k={k}: no clear bracket (above={M_above})")
                continue
            log(f"  k={k}: bracket = {bracket}, linear interp M* ~ {M_star:.5f}")
            lo, hi = bracket
            iters = 0
            while hi - lo > args.bisect_tol and iters < 6:
                mid = 0.5 * (lo + hi)
                if args.skip_existing and already_done(records, k, N, mid):
                    pass
                else:
                    log(f"    bisect mid={mid:.5f}")
                    rec = run_pair(k, N, mid, epsilon=args.epsilon,
                                   verbose=args.verbose, mosek_params=mosek_params)
                    append_record(rec)
                    records = existing_records()
                # find lambda at mid
                lam_mid = None
                for r in records:
                    if (r.get("k") == k and r.get("N") == N
                            and abs(r.get("M", -1) - mid) < 1e-9):
                        lam_mid = r.get("lambda_3pt")
                        break
                if lam_mid is None:
                    log(f"    mid solve failed; abort bisection k={k}")
                    break
                if lam_mid >= THRESHOLD:
                    lo = mid
                else:
                    hi = mid
                iters += 1
            log(f"  k={k}: bisected bracket {(lo, hi)}")

    # Final summary
    log("\n=== FINAL SUMMARY ===")
    log(f"{'k':>3} {'N':>3} {'M':>7} {'lam2pt':>10} {'lam3pt':>10} {'>1.2802?':>10}")
    for r in sorted(records, key=lambda r: (r.get("k", 0), r.get("M", 0))):
        l2 = r.get("lambda_2pt")
        l3 = r.get("lambda_3pt")
        ok3 = "Y" if (l3 is not None and l3 >= THRESHOLD) else "N"
        s2 = f"{l2:>10.5f}" if l2 is not None else f"{'FAIL':>10}"
        s3 = f"{l3:>10.5f}" if l3 is not None else f"{'FAIL':>10}"
        log(f"{r.get('k'):>3} {r.get('N'):>3} {r.get('M'):>7.4f} {s2} {s3} {ok3:>10}")

    log("\n=== CROSSOVER M* by k ===")
    for k in sorted(set(r.get("k", 0) for r in records)):
        N = k
        M_star, M_above, bracket = find_crossover_M(records, k, N)
        log(f"  k={k}: max M with lambda3pt >= 1.2802 = {M_above}, "
            f"bracket = {bracket}, linear M* = {M_star}")


if __name__ == "__main__":
    main()
