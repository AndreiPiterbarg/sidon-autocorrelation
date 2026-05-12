"""Agent M13: SCALE-RATIO STRUCTURAL ANALYSIS.

Goal: investigate whether the optimal δ_2/δ_1 ratio (≈ 0.326 from 2-scale K) has
a SPECIAL value (e.g., 1/3, 1/π, 1/φ, ratios of Bessel J_0 zeros).

Steps:
  1. Fine ratio sweep: for r ∈ {0.20, 0.22, ..., 0.42}, find optimal (δ_1, λ_1)
     via DE; report M_cert vs r.
  2. Detect structural peaks/plateaus; compare to special ratios.
  3. Geometric ladder: δ_k = δ_1 · r^(k-1), λ_k ∝ s^(k-1) for N = 3, 4, 5, 6.
  4. Identify if the optimum LIES ON a geometric ladder vs flat 1.29005.

Pipeline reused: _K26_full_sweep_reopt.eval_at.
"""
from __future__ import annotations

import json
import math
import time
import numpy as np
from scipy.optimize import differential_evolution

from _K26_full_sweep_reopt import eval_at, DELTA, U, N_QP

# Constants
DELTA_MAX = DELTA  # 0.138
DELTA_MIN = 0.005
LAM_MIN = 0.005
LAM_MAX = 0.995
MV_BASE = 1.27481
N2_BEST = 1.29005

# Bessel J_0 zeros
J0_ZEROS = [2.4048255576957727, 5.5200781102863115, 8.653727912911012,
            11.791534439014281, 14.930917708487785, 18.071063967910922]

# Special ratios to test against
SPECIAL_RATIOS = {
    "1/2": 0.5,
    "1/3": 1.0 / 3.0,
    "1/4": 0.25,
    "1/pi": 1.0 / math.pi,                              # ~0.3183
    "1/phi": 2.0 / (1.0 + math.sqrt(5.0)),              # ~0.6180
    "1/phi^2": ((math.sqrt(5.0) - 1.0) / 2.0) ** 2,     # ~0.3820
    "1/e": 1.0 / math.e,                                # ~0.3679
    "j01/j02": J0_ZEROS[0] / J0_ZEROS[1],               # ~0.4356
    "j01/j03": J0_ZEROS[0] / J0_ZEROS[2],               # ~0.2779
    "j02/j03": J0_ZEROS[1] / J0_ZEROS[2],               # ~0.6379
    "sqrt(2)-1": math.sqrt(2.0) - 1.0,                  # ~0.4142
    "ln2": math.log(2.0),                               # ~0.6931
    "ln2/2": math.log(2.0) / 2.0,                       # ~0.3466
    "2-sqrt(3)": 2.0 - math.sqrt(3.0),                  # ~0.2679
}


def safe_eval(deltas, lambdas):
    """Wrapper returning M_cert or None on error."""
    try:
        r = eval_at(deltas, lambdas)
    except Exception:
        return None, None
    Mc = r.get("M_cert")
    if Mc is None or not np.isfinite(Mc):
        return None, r
    return float(Mc), r


# --- Step 1: ratio sweep for N=2 -----------------------------------------------

def best_for_ratio_N2(r, seeds=(7, 42)):
    """For fixed ratio r = δ_2/δ_1, optimize over (δ_1, λ_1) via DE."""
    # δ_1 in [max(DELTA_MIN, DELTA_MIN/r if r<1), DELTA_MAX].
    # We always need δ_2 = r*δ_1 ∈ [DELTA_MIN, DELTA_MAX] too.
    d1_lo = max(DELTA_MIN, DELTA_MIN / max(r, 1e-6))
    d1_hi = min(DELTA_MAX, DELTA_MAX / max(r, 1e-6))
    if d1_hi <= d1_lo:
        return None, None, None
    d1_hi = min(d1_hi, DELTA_MAX)
    if d1_hi <= d1_lo + 1e-5:
        return None, None, None
    bounds = [(d1_lo, d1_hi), (0.50, 0.99)]  # δ_1, λ_1
    history = []

    def obj(x):
        d1, l1 = float(x[0]), float(x[1])
        d2 = r * d1
        if d2 < DELTA_MIN or d2 > DELTA_MAX:
            return 0.0
        Mc, _ = safe_eval([d1, d2], [l1, 1.0 - l1])
        if Mc is None:
            return 0.0
        history.append((Mc, d1, l1, d2))
        return -Mc

    best = (None, None, None, None)
    for seed in seeds:
        res = differential_evolution(
            obj, bounds, maxiter=25, popsize=16, seed=seed,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            init='sobol', updating='deferred', workers=1, disp=False)
        if history:
            top = max(history, key=lambda h: h[0])
            if best[0] is None or top[0] > best[0]:
                best = top
    if best[0] is None:
        return None, None, None
    Mc, d1_opt, l1_opt, d2_opt = best
    return Mc, {"delta_1": float(d1_opt), "delta_2": float(d2_opt),
                "lambda_1": float(l1_opt), "lambda_2": float(1.0 - l1_opt),
                "ratio_r": float(r)}, None


# --- Step 4: geometric ladder for N=N --------------------------------------

def eval_geometric(N, delta_1, r_geom, s_lam):
    """δ_k = δ_1 · r^(k-1); λ_k ∝ s^(k-1) (then normalize)."""
    deltas = [delta_1 * (r_geom ** k) for k in range(N)]
    if any(d < DELTA_MIN or d > DELTA_MAX for d in deltas):
        return None
    raw_lams = np.array([s_lam ** k for k in range(N)], dtype=float)
    if raw_lams.sum() <= 0:
        return None
    lams = (raw_lams / raw_lams.sum()).tolist()
    if any(l < 1e-4 for l in lams):
        return None
    Mc, _ = safe_eval(deltas, lams)
    return Mc


def best_geometric_ladder_N(N, seeds=(1, 13, 27)):
    """For given N, DE-search (δ_1, r, s) for the geometric ladder."""
    d1_lo = max(DELTA_MIN * 2.0, 0.02)
    d1_hi = DELTA_MAX
    r_lo, r_hi = 0.15, 0.55   # δ_2/δ_1 should be in this range
    s_lo, s_hi = 0.001, 0.999  # weight decay
    bounds = [(d1_lo, d1_hi), (r_lo, r_hi), (s_lo, s_hi)]
    history = []

    def obj(x):
        d1, r_geom, s_lam = float(x[0]), float(x[1]), float(x[2])
        Mc = eval_geometric(N, d1, r_geom, s_lam)
        if Mc is None:
            return 0.0
        history.append((Mc, d1, r_geom, s_lam))
        return -Mc

    best = None
    for seed in seeds:
        res = differential_evolution(
            obj, bounds, maxiter=40, popsize=20, seed=seed,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            init='sobol', updating='deferred', workers=1, disp=False)
        if history:
            top = max(history, key=lambda h: h[0])
            if best is None or top[0] > best[0]:
                best = top
    if best is None:
        return None
    Mc, d1, r_geom, s_lam = best
    # Reconstruct
    deltas = [d1 * (r_geom ** k) for k in range(N)]
    raw_lams = np.array([s_lam ** k for k in range(N)], dtype=float)
    lams = (raw_lams / raw_lams.sum()).tolist()
    return {"N": N, "M_cert": float(Mc), "delta_1": float(d1),
            "r_geom": float(r_geom), "s_lam": float(s_lam),
            "deltas": [float(d) for d in deltas], "lambdas": lams}


# --- Reference: unconstrained N-component baselines (compare to ladder) -------

def best_unconstrained_N(N, seeds=(1, 42, 73)):
    """DE-search all (δ_1..δ_N, λ_1..λ_{N-1}) freely. Compare to geometric ladder."""
    bounds = [(DELTA_MIN, DELTA_MAX)] * N + [(LAM_MIN, LAM_MAX)] * (N - 1)
    history = []

    def obj(x):
        deltas = list(x[:N])
        lams_free = list(x[N:])
        lam_last = 1.0 - sum(lams_free)
        if lam_last < LAM_MIN or lam_last > LAM_MAX:
            return 0.0
        lambdas = lams_free + [lam_last]
        if any(d < DELTA_MIN or d > DELTA_MAX for d in deltas):
            return 0.0
        Mc, _ = safe_eval(deltas, lambdas)
        if Mc is None:
            return 0.0
        history.append((Mc, deltas, lambdas))
        return -Mc

    best = None
    for seed in seeds:
        differential_evolution(
            obj, bounds, maxiter=50, popsize=20, seed=seed,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            init='sobol', updating='deferred', workers=1, disp=False)
        if history:
            top = max(history, key=lambda h: h[0])
            if best is None or top[0] > best[0]:
                best = top
    if best is None:
        return None
    return {"N": N, "M_cert": float(best[0]),
            "deltas": [float(d) for d in best[1]],
            "lambdas": [float(l) for l in best[2]]}


def main():
    t0 = time.time()
    print("=" * 80)
    print("Agent M13: scale-ratio structural analysis")
    print(f"  baselines: MV={MV_BASE}, 2-scale-best={N2_BEST}")
    print("=" * 80)

    # Step 1: fine ratio sweep
    ratios = [0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42]
    print("\n--- Step 1: Fine ratio sweep (r = d_2/d_1) ---")
    print(f"{'r':>7} {'M_cert':>10} {'delta_1':>10} {'lambda_1':>10}")
    ratio_results = []
    for r in ratios:
        Mc, params, _ = best_for_ratio_N2(r)
        if Mc is None:
            print(f"{r:>7.3f} {'FAIL':>10}")
            ratio_results.append({"ratio": r, "M_cert": None, "params": None})
        else:
            print(f"{r:>7.3f} {Mc:>10.5f} {params['delta_1']:>10.5f} {params['lambda_1']:>10.5f}")
            ratio_results.append({"ratio": r, "M_cert": Mc, "params": params})

    # Find max over ratios
    valid = [x for x in ratio_results if x["M_cert"] is not None]
    if not valid:
        print("ALL RATIOS FAILED")
        return
    best_ratio = max(valid, key=lambda x: x["M_cert"])
    print(f"\nBest ratio: r={best_ratio['ratio']:.3f}  M_cert={best_ratio['M_cert']:.5f}")
    print(f"  vs flat N=2 1.29005: {best_ratio['M_cert'] - N2_BEST:+.5f}")

    # Step 2: compare to special ratios
    print("\n--- Step 2: Special ratios ---")
    print(f"{'name':>14} {'r':>8} {'M_cert':>10} {'delta_1':>10} {'lambda_1':>10}")
    special_results = {}
    for name, r in sorted(SPECIAL_RATIOS.items(), key=lambda kv: kv[1]):
        Mc, params, _ = best_for_ratio_N2(r)
        if Mc is None:
            print(f"{name:>14} {r:>8.4f} {'FAIL':>10}")
            special_results[name] = {"ratio": r, "M_cert": None}
        else:
            mk = " *" if Mc > best_ratio["M_cert"] else ""
            print(f"{name:>14} {r:>8.4f} {Mc:>10.5f} {params['delta_1']:>10.5f} {params['lambda_1']:>10.5f}{mk}")
            special_results[name] = {"ratio": r, "M_cert": Mc, "params": params}

    # Step 3: identify best special
    valid_sp = [(k, v) for k, v in special_results.items() if v.get("M_cert") is not None]
    best_sp = max(valid_sp, key=lambda kv: kv[1]["M_cert"]) if valid_sp else None
    if best_sp:
        print(f"\nBest special: {best_sp[0]} (r={best_sp[1]['ratio']:.4f})  M_cert={best_sp[1]['M_cert']:.5f}")

    # Step 4: geometric ladder (N = 2..6)
    print("\n--- Step 4: Geometric ladder d_k = d_1*r^(k-1), lam_k ~ s^(k-1) ---")
    print(f"{'N':>3} {'M_cert':>10} {'delta_1':>10} {'r_geom':>10} {'s_lam':>10}")
    ladder_results = {}
    for N in [2, 3, 4, 5, 6]:
        rec = best_geometric_ladder_N(N)
        if rec is None:
            print(f"{N:>3} FAIL")
            ladder_results[N] = None
            continue
        print(f"{N:>3} {rec['M_cert']:>10.5f} {rec['delta_1']:>10.5f} "
              f"{rec['r_geom']:>10.5f} {rec['s_lam']:>10.5f}")
        ladder_results[N] = rec

    # Step 5: unconstrained N-component baselines  (compare to ladder)
    print("\n--- Step 5: Unconstrained N-component (reference for ladder) ---")
    print(f"{'N':>3} {'M_cert_unc':>12} {'M_cert_lad':>12} {'gap':>10}")
    uncon_results = {}
    for N in [2, 3, 4]:
        u = best_unconstrained_N(N)
        if u is None:
            print(f"{N:>3} FAIL")
            uncon_results[N] = None
            continue
        lad = ladder_results.get(N)
        gap = (u['M_cert'] - lad['M_cert']) if lad else float('nan')
        print(f"{N:>3} {u['M_cert']:>12.5f} "
              f"{(lad['M_cert'] if lad else float('nan')):>12.5f} {gap:>10.5f}")
        uncon_results[N] = u

    # Summary
    print("\n" + "=" * 80)
    best_overall = best_ratio["M_cert"]
    label_overall = f"ratio={best_ratio['ratio']:.3f} (N=2)"
    for N, rec in ladder_results.items():
        if rec is not None and rec["M_cert"] > best_overall:
            best_overall = rec["M_cert"]
            label_overall = f"geometric ladder N={N}"
    for N, rec in uncon_results.items():
        if rec is not None and rec["M_cert"] > best_overall:
            best_overall = rec["M_cert"]
            label_overall = f"unconstrained N={N}"
    print(f"OVERALL BEST: M_cert={best_overall:.5f}  via {label_overall}")
    print(f"  vs flat 2-scale 1.29005: {best_overall - N2_BEST:+.5f}")
    print("=" * 80)

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    out = {
        "agent": "M13",
        "description": "Scale-ratio structural analysis for multi-scale arcsine kernels",
        "baselines": {"MV": MV_BASE, "N2_flat": N2_BEST},
        "DELTA_MAX": DELTA_MAX,
        "DELTA_MIN": DELTA_MIN,
        "step1_ratio_sweep": ratio_results,
        "step2_special_ratios": special_results,
        "best_in_ratio_sweep": best_ratio,
        "best_in_special": ({"name": best_sp[0], **best_sp[1]} if best_sp else None),
        "step4_geometric_ladder": {str(N): rec for N, rec in ladder_results.items()},
        "step5_unconstrained": {str(N): rec for N, rec in uncon_results.items()},
        "overall_best": {"M_cert": best_overall, "label": label_overall},
        "elapsed_s": elapsed,
    }
    out_path = "_M13_scale_ratio.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
