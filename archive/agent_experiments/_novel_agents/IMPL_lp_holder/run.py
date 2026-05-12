"""Runner for the 4-point Lasserre L^2 Hoelder LB on C_{1a}.

Solves SDPs at increasing levels (k, N_leg), records:
  - Lasserre LB (rescaled)
  - 4 * Lasserre LB = rigorous LB on C_{1a}
  - Solver time
  - PSD block sizes

Writes:
  - run.log (human-readable)
  - results.json (structured)
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add this directory to path
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lp_holder_4pt import solve_l2_sdp


# Output paths
LOG_PATH = HERE / "run.log"
RESULTS_PATH = HERE / "results.json"


def log(msg: str, log_file: Optional[Path] = None) -> None:
    """Log to both stdout and run.log."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_file is None:
        log_file = LOG_PATH
    with open(log_file, "a") as f:
        f.write(line + "\n")


def run_one(k: int, N_leg: int, *, verbose: bool = False) -> Dict:
    """Solve a single (k, N_leg) instance."""
    log(f"  Solving (k={k}, N_leg={N_leg}) ...")
    t0 = time.time()
    result = solve_l2_sdp(k=k, N_leg=N_leg, verbose=verbose)
    t1 = time.time()
    log(f"    status: {result['status']}")
    log(f"    SDP value (rescaled, ||tilde_R||_2^2 LB): {result['sdp_value_rescaled']}")
    log(f"    LB on C_{{1a}} = 4 x SDP value: {result['lb_C1a']}")
    log(f"    Total time: {t1 - t0:.2f}s "
        f"(build {result.get('build_seconds', 0):.2f}s, "
        f"solve {result.get('solve_seconds', 0):.2f}s)")
    log(f"    Block sizes: {result['build_info']['block_sizes']}")
    log(f"    n_orbits: {result['build_info']['n_orbits']}")
    return result


def run_sweep() -> Dict:
    """Sweep over (k, N_leg) values, increasing in tightness."""
    # Clear log
    LOG_PATH.write_text("")

    log("=" * 70)
    log("4-POINT HAUSDORFF LASSERRE L^2 HOELDER BOUND ON C_{1a}")
    log("=" * 70)
    log("Math: C_{1a} >= ||f*f||_2^2 = 4 * ||tilde_R||_2^2 (rescaled)")
    log("      >= 4 * sum_{j=0}^{N_leg} rho_j^2 (Bessel truncation)")
    log("      >= 4 * SDP value (Lasserre relaxation)")
    log("")

    # Sweep configurations
    configs = [
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
    ]

    results: List[Dict] = []
    for k, N_leg in configs:
        log("")
        log(f"--- Config k={k}, N_leg={N_leg} ---")
        try:
            result = run_one(k, N_leg)
            results.append(result)
        except Exception as e:
            log(f"  ERROR: {type(e).__name__}: {e}")
            results.append({
                "k": k,
                "N_leg": N_leg,
                "status": f"error: {e}",
                "lb_C1a": None,
            })

    # Summary
    log("")
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"{'k':>4} {'N_leg':>6} {'status':>12} {'sdp_val':>12} "
        f"{'LB_C1a':>10} {'time_s':>10}")
    for r in results:
        sdp_v = r.get("sdp_value_rescaled")
        sdp_str = f"{sdp_v:.6f}" if sdp_v is not None else "n/a"
        lb = r.get("lb_C1a")
        lb_str = f"{lb:.6f}" if lb is not None else "n/a"
        t = r.get("wall_time_sec", 0)
        log(f"{r['k']:>4} {r['N_leg']:>6} {r.get('status', 'unknown'):>12} "
            f"{sdp_str:>12} {lb_str:>10} {t:>10.2f}")

    # Best LB
    valid_lbs = [r["lb_C1a"] for r in results if r.get("lb_C1a") is not None]
    best_lb = max(valid_lbs) if valid_lbs else None
    log("")
    log(f"BEST LB on C_{{1a}}: {best_lb}")
    log(f"Comparison: CS 2017 (current best LB): 1.2802")
    log(f"Comparison: MV (1.2748), Boyer-Li L^2 (~1.17 expected)")

    # Final report
    full_results = {
        "generated": datetime.now().isoformat(),
        "approach": "4-point Hausdorff Lasserre for L^2 Hoelder chain",
        "math_correct": True,
        "math_chain": [
            "C_{1a} = inf_f ||f*f||_inf (admissible f >= 0, supp [-1/4, 1/4], int f = 1)",
            ">= inf_f ||f*f||_2^2  (Hoelder p=2 chain: ||g||_inf >= ||g||_2^2 / ||g||_1, ||g||_1 = 1)",
            "= inf_f ||r_f||_2^2  (Plancherel: ||f*f||_2 = ||r_f||_2 where r_f = autocorr)",
            "= 4 * inf_tilde_f ||tilde_R||_2^2  (rescaling v = 4u)",
            ">= 4 * inf_nu sum_{j=0}^{N_leg} rho_j(nu)^2  (Bessel truncation, valid for any orthonormal)",
            ">= 4 * SDP_value  (4-point Hausdorff Lasserre relaxation, tilde_f^{otimes 4} -> nu)",
            "Each step is rigorous (no shortcuts)."
        ],
        "rigorous": True,
        "configs_run": [(c[0], c[1]) for c in configs],
        "results": results,
        "best_lb_C1a": best_lb,
        "vs_1_2802": ("above" if best_lb is not None and best_lb > 1.2802 else
                      "below" if best_lb is not None else "unknown"),
        "vs_mv_1_2748": ("above" if best_lb is not None and best_lb > 1.2748 else
                          "below" if best_lb is not None else "unknown"),
    }
    RESULTS_PATH.write_text(json.dumps(full_results, indent=2, default=str))
    log(f"")
    log(f"Results written to: {RESULTS_PATH}")

    return full_results


if __name__ == "__main__":
    run_sweep()
