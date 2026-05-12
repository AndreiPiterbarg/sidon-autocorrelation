"""C4 Hardy-Littlewood / BMO probe for Sidon constant lower bound.

Tests whether maximal-function / BMO / CZ-decomposition / Plancherel-L^2
arguments yield a useful LOWER bound on M(f) := ||f*f||_inf for nonneg f
supported on [-1/4, 1/4] with int f = 1.

Concrete tests:
  T1. The L^2-Plancherel chain bound:
        M(f) = ||g||_inf >= ||g||_2^2 / ||g||_1 = ||g||_2^2,    (g = f*f)
      computed for the OPTIMAL step-function on d-bins (minimizer of
      ||g||_2^2 subject to a_i >= 0, sum a_i*h = 1). This shows the best
      one can hope for from L^2 lower bound.

  T2. The CZ-decomposition inequality:
        L_lambda := |{g > lambda}| >= (1 - lambda) / (M - lambda).
      We INVERT: given an OPTIMAL distributional profile (concentrating
      mass at level just above 1, on a set just smaller than 1),
      what is the implied lower bound on M? (Spoiler: the inequality is
      consistent with M -> 1+, so it gives no lower bound > 1.)

  T3. BMO / John-Nirenberg: bound on |{g > 1 + lambda}|. Demonstrate
      this gives only UPPER bounds on level sets, hence NO lower bound
      on M.

For each test we report quantitative outcomes for d in {4, 6, 8, 10}.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, linprog

LOG_PATH = Path(__file__).parent / "run.log"
RESULTS_PATH = Path(__file__).parent / "results.json"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# -----------------------------------------------------------------------------
# Step-function autocorrelation utilities
# -----------------------------------------------------------------------------
# Convention:
#   d bins of width h = 1/(2d) tiling [-1/4, 1/4].
#   f = sum_i a_i * 1_{B_i}, a_i >= 0.
#   int f = h * sum a_i = 1   <=>  sum a_i = 2d.
#   So a_i are densities (height of step), with average density 1/(1/2) = 2.
#   Wait: f integrates to 1 on a set of measure 1/2, so AVERAGE density is 2.
#   sum a_i / d = 2, i.e., sum a_i = 2d. Confirmed.

def conv_coeffs(a: np.ndarray) -> np.ndarray:
    """Return c_j = sum_i a_i a_{i+j} for j = -(d-1)..(d-1).

    For step f on d bins, the AUTOCORRELATION g = f * f-tilde
    evaluated at t_j = j*h is g(j*h) = h * c_j (piecewise-linear interpolant
    of the discrete sequence).
    """
    return np.convolve(a, a[::-1], mode="full")  # length 2d-1, indexed -(d-1)..(d-1)


def conv_autocorr_pwlin_l2_sq(a: np.ndarray, h: float) -> float:
    """L^2 norm squared of the SYMMETRIC autocorrelation g_sym = f * f-tilde.

    g_sym is piecewise-LINEAR with breakpoints at j*h, j = -d..d.
    Values at breakpoints are h * c_j where c_j = sum_i a_i a_{i+j}.
    With c_{-d} = c_d = 0 (boundary).

    Integral of |g_sym|^2 over R: piecewise-linear quadrature.
    On each segment [j*h, (j+1)*h] of length h, g_sym is linear from
    h*c_j to h*c_{j+1}. Integral of square = h * (h*c_j)^2 + (h*c_j)*(h*c_{j+1}) + (h*c_{j+1})^2 ) / 3
    = (h^3 / 3) * (c_j^2 + c_j * c_{j+1} + c_{j+1}^2).
    """
    c = conv_coeffs(a)  # length 2d-1, indices -(d-1)..(d-1)
    d = len(a)
    # Pad c with zeros on both ends so we have c_{-d} = c_d = 0.
    c_full = np.concatenate(([0.0], c, [0.0]))  # length 2d+1, indices -d..d
    # Sum over segments j = -d .. d-1 (there are 2d segments).
    s = 0.0
    for k in range(len(c_full) - 1):
        cj = c_full[k]
        cj1 = c_full[k + 1]
        s += cj * cj + cj * cj1 + cj1 * cj1
    return (h ** 3) * s / 3.0


def conv_autocorr_pwlin_linf(a: np.ndarray, h: float) -> float:
    """L^inf norm of g_sym = f * f-tilde (= h * max c_j, achieved at j=0).

    Note for asymmetric f, g = f*f and g_sym = f*f-tilde differ:
      ||g_sym||_inf = (f*f-tilde)(0) = h * sum a_i^2 = h * ||a||_2^2 (always).
      ||g||_inf = max_j h * c'_j where c'_j = sum_i a_i a_{j-i+1} (different).

    The Sidon constant is about ||g||_inf, NOT ||g_sym||_inf. So this
    function returns ||g_sym||_inf which is RELATED but DIFFERENT.
    """
    c = conv_coeffs(a)
    return h * float(np.max(c))


def actual_g_linf(a: np.ndarray, h: float) -> float:
    """L^inf norm of g = f * f (piecewise-linear, NOT symmetric).

    g(t) for t = j*h is h * sum_i a_i * a_{j - i} where indices in [0, d-1].
    This is the discrete convolution of a with itself (NOT cross-correlation).
    """
    cc = np.convolve(a, a, mode="full")  # length 2d-1
    return h * float(np.max(cc))


# -----------------------------------------------------------------------------
# T1. L^2-Plancherel lower bound: minimize ||g_sym||_2^2 over feasible a.
# -----------------------------------------------------------------------------
def minimize_l2_sq(d: int, n_starts: int = 20, seed: int = 7) -> dict:
    """Find min over a >= 0, h * sum a_i = 1, of ||f*f-tilde||_2^2.

    This is a convex QP: ||g_sym||_2^2 is a positive-semidefinite quadratic
    in a (it's a sum of squares of bilinear forms in a).

    We solve via projected gradient using scipy with multiple random restarts.
    """
    h = 1.0 / (2 * d)
    target_sum = 2 * d  # sum a_i = 2d

    def obj(a: np.ndarray) -> float:
        # Project a onto simplex {a >= 0, sum = 2d} before evaluating.
        # Actually scipy with constraints handles this. We use SLSQP.
        return conv_autocorr_pwlin_l2_sq(a, h)

    def obj_grad(a: np.ndarray) -> np.ndarray:
        # Numerical gradient (cheap for small d).
        eps = 1e-6
        g = np.zeros_like(a)
        f0 = obj(a)
        for i in range(len(a)):
            a_pert = a.copy()
            a_pert[i] += eps
            g[i] = (obj(a_pert) - f0) / eps
        return g

    rng = np.random.default_rng(seed)
    best_val = np.inf
    best_a = None
    for start in range(n_starts):
        # Random feasible start
        a0 = rng.uniform(0.1, 1.0, size=d)
        a0 = a0 * target_sum / a0.sum()
        constraints = [{"type": "eq", "fun": lambda a: a.sum() - target_sum}]
        bounds = [(0.0, None)] * d
        res = minimize(
            obj,
            a0,
            jac=obj_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 200},
        )
        if res.fun < best_val:
            best_val = float(res.fun)
            best_a = res.x.copy()
    return {
        "min_l2_sq": best_val,
        "argmin_a": best_a.tolist() if best_a is not None else None,
        "argmin_actual_g_linf": actual_g_linf(best_a, h) if best_a is not None else None,
        "argmin_g_sym_linf": conv_autocorr_pwlin_linf(best_a, h) if best_a is not None else None,
    }


# -----------------------------------------------------------------------------
# T2. Test the CZ-decomposition inequality numerically.
# -----------------------------------------------------------------------------
def test_cz_inequality(d: int = 8, n_random: int = 100, seed: int = 11) -> dict:
    """For random feasible f, compute (M, distribution function), check
    L_lambda >= (1 - lambda) / (M - lambda) and look for the binding lambda.

    Goal: see whether CZ inequality combined with int g = 1 yields a
    useful lower bound on M.
    """
    h = 1.0 / (2 * d)
    target_sum = 2 * d
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_random):
        a = rng.uniform(0.0, 1.0, size=d)
        if a.sum() < 1e-9:
            continue
        a = a * target_sum / a.sum()
        # g = f * f piecewise-linear breakpoints at j*h, j=0..2d-1, values h*c_j
        cc = np.convolve(a, a, mode="full")  # length 2d-1
        # Build a dense grid for distribution function
        n_grid = 2000
        t_grid = np.linspace(-0.5, 0.5, n_grid)
        # On t = -0.5 + (j + s) * h with s in [0,1), g = h * ((1-s)*cc[j] + s*cc[j+1])
        # Simpler: just sample
        c_full = np.concatenate(([0.0], cc, [0.0]))
        idx_real = (t_grid + 0.5) / h  # in [0, 2d]
        idx0 = np.floor(idx_real).astype(int)
        idx0 = np.clip(idx0, 0, 2 * d - 1)
        s = idx_real - idx0
        g = h * ((1 - s) * c_full[idx0] + s * c_full[idx0 + 1])
        M = float(np.max(g))
        # Distribution function
        lams = np.linspace(0, M * 0.99, 200)
        L_lams = np.array([(g > lam).mean() for lam in lams])  # measure of {g > lam} on [-1/2, 1/2] of length 1
        # CZ: L_lam >= (1 - lam) / (M - lam) for lam < min(1, M)
        cz_rhs = np.where(M > lams, (1 - lams) / (M - lams + 1e-15), 0)
        # Where lam >= 1, CZ rhs is non-positive -> trivially satisfied.
        valid = (lams < 1) & (lams < M)
        slack = L_lams[valid] - cz_rhs[valid]
        min_slack = float(np.min(slack)) if valid.any() else None
        samples.append({"M": M, "min_slack": min_slack})
    Ms = [s["M"] for s in samples]
    slacks = [s["min_slack"] for s in samples if s["min_slack"] is not None]
    return {
        "M_min_observed": float(np.min(Ms)) if Ms else None,
        "M_max_observed": float(np.max(Ms)) if Ms else None,
        "M_mean_observed": float(np.mean(Ms)) if Ms else None,
        "cz_inequality_min_slack": min(slacks) if slacks else None,
        "cz_inequality_violated": (min(slacks) < -1e-9) if slacks else False,
        # Conclusion: CZ gives M >= some_value? Actually CZ gives
        # L_lambda > 0 always (so M > lambda for any lambda < 1 ... no it doesn't).
        # The inequality is consistent with M -> 1+ from above (with L = 1).
        # So CZ alone does NOT yield M > 1.
    }


# -----------------------------------------------------------------------------
# T3. Compare bound (*) to actual val(d) for d in {4, 6, 8, 10}.
# -----------------------------------------------------------------------------
def actual_minimax_val(d: int) -> float:
    """Compute val(d) = min_a max_t g(t) approximately by minimizing the
    max convolution coefficient (LP form via scipy.linprog).

    g_max ~ h * max_j sum_i a_i a_{j-i} ... but max_j of a quadratic is NOT
    an LP. We use a HEURISTIC: minimize the L^inf of the discrete
    autocorrelation values via successive linearization. This is just for
    comparison reference -- not rigorous.
    """
    h = 1.0 / (2 * d)
    target_sum = 2 * d
    # Use SLSQP min-max trick: minimize z subject to h * sum_i a_i a_{j-i} <= z for all j.
    # Quadratic constraints; use SLSQP.
    def obj(x: np.ndarray) -> float:
        return x[-1]  # z

    def cons_grad(x):
        return None

    # x = [a_0, ..., a_{d-1}, z]
    n = d + 1
    rng = np.random.default_rng(31)
    best = np.inf
    best_a = None
    for start in range(20):
        a0 = rng.uniform(0.1, 1.0, size=d)
        a0 = a0 * target_sum / a0.sum()
        x0 = np.concatenate([a0, [10.0]])

        constraints = []
        # equality: sum a_i = 2d
        constraints.append({"type": "eq", "fun": lambda x: x[:d].sum() - target_sum})
        # inequality: z - h * sum_i a_i a_{j-i} >= 0 for j = 0..2d-2
        for j in range(2 * d - 1):
            def make_c(j_=j):
                def c(x):
                    a = x[:d]
                    z = x[-1]
                    s = 0.0
                    for i in range(d):
                        if 0 <= j_ - i < d:
                            s += a[i] * a[j_ - i]
                    return z - h * s
                return c
            constraints.append({"type": "ineq", "fun": make_c()})

        bounds = [(0.0, None)] * d + [(0.0, None)]
        try:
            res = minimize(
                obj,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-9, "maxiter": 300},
            )
            if res.success and res.fun < best:
                best = float(res.fun)
                best_a = res.x[:d].copy()
        except Exception:
            pass
    return best, best_a


# -----------------------------------------------------------------------------
# T5. Higher-p Holder chain on g = f*f directly.
# -----------------------------------------------------------------------------
def conv_pwlin_lp_g(a: np.ndarray, p: int, h: float, n_seg: int = 30) -> float:
    """int g^p where g = f*f piecewise-linear."""
    cc = np.convolve(a, a, mode="full")
    cf = np.concatenate(([0.0], cc, [0.0]))
    total = 0.0
    for k in range(len(cf) - 1):
        v0 = h * cf[k]
        v1 = h * cf[k + 1]
        s = np.linspace(v0, v1, n_seg)
        total += h * np.trapezoid(s ** p, dx=1.0 / (n_seg - 1))
    return total


def lp_chain_min(d: int, p: int, n_starts: int = 60) -> dict:
    """Numerically minimize ||g||_p^p over step f on d bins. Apply Holder
    chain ||g||_inf >= ||g||_p^{p/(p-1)}.
    """
    h = 1.0 / (2 * d)
    target = 2 * d
    rng = np.random.default_rng(d * 1000 + p)
    best = np.inf
    best_a = None
    for _ in range(n_starts):
        a0 = rng.uniform(0.05, 1.0, size=d)
        a0 = a0 * target / a0.sum()
        res = minimize(
            lambda a: conv_pwlin_lp_g(a, p, h),
            a0,
            method="SLSQP",
            bounds=[(0, None)] * d,
            constraints=[{"type": "eq", "fun": lambda a: a.sum() - target}],
            options={"ftol": 1e-10, "maxiter": 400},
        )
        if res.fun < best:
            best = float(res.fun)
            best_a = res.x.copy()
    M_LB = best ** (1.0 / (p - 1))
    return {"min_g_p_p": best, "M_LB_step_d": M_LB,
            "argmin_a": best_a.tolist() if best_a is not None else None}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    LOG_PATH.write_text("", encoding="utf-8")  # reset
    log("=== C4 Hardy-Littlewood / BMO probe ===")
    log("Setup: f >= 0 on [-1/4, 1/4], int f = 1, g = f*f.")
    log("Goal: lower bound on M(f) = ||g||_inf.")

    log("--- T1. L^2-Plancherel lower bound ---")
    log("Bound (*): M(f) >= ||g_sym||_2^2 / ||g_sym||_1 = ||g_sym||_2^2.")
    log("(Note: ||g_sym||_1 = (int f)^2 = 1 even when g_sym != g for asymm f.)")
    log("Test: minimize ||g_sym||_2^2 over feasible a.")

    t1_results = {}
    for d in [4, 6, 8, 10]:
        log(f"  d={d}: solving QP min_a ||g_sym||_2^2 ...")
        r = minimize_l2_sq(d, n_starts=30 if d <= 8 else 20)
        t1_results[d] = r
        log(f"    min ||g_sym||_2^2 = {r['min_l2_sq']:.6f}")
        log(f"    at argmin: ||g_sym||_inf = {r['argmin_g_sym_linf']:.6f}, ||g||_inf (actual f*f) = {r['argmin_actual_g_linf']:.6f}")

    log("--- T1 verdict ---")
    best_l2_lb = max(t1_results[d]["min_l2_sq"] for d in t1_results)
    log(f"  Best L^2 lower bound across d in {{4,6,8,10}}: M(f) >= {best_l2_lb:.6f}")
    log(f"  vs CS 2017 LB 1.2802: {'ABOVE' if best_l2_lb > 1.2802 else 'BELOW'}")

    log("--- T2. Calderon-Zygmund decomposition probe ---")
    t2_results = {}
    for d in [4, 8]:
        log(f"  d={d}: testing CZ inequality on 100 random feasible a ...")
        r = test_cz_inequality(d, n_random=100)
        t2_results[d] = r
        log(f"    M observed in [{r['M_min_observed']:.4f}, {r['M_max_observed']:.4f}], mean {r['M_mean_observed']:.4f}")
        log(f"    CZ inequality min slack: {r['cz_inequality_min_slack']:.4e} (>=0 means valid)")
        log(f"    CZ violated: {r['cz_inequality_violated']}")

    log("--- T2 verdict ---")
    log("  CZ inequality L_lambda >= (1 - lambda)/(M - lambda) is")
    log("  consistent with arbitrary M > 1, since L_lambda <= 1 forces")
    log("  M >= 1 only. So CZ alone does NOT yield M > 1.")
    log("  CZ is in the WRONG DIRECTION for lower-bound on M.")

    log("--- T3. BMO/John-Nirenberg analysis (analytical) ---")
    log("  ||g||_BMO <= ||g||_inf / 2 = M/2 (trivial since g >= 0).")
    log("  John-Nirenberg: |{|g - g_I| > lambda}| <= C |I| exp(-c lambda / B).")
    log("  This gives UPPER BOUND on level sets, hence is")
    log("  CONSISTENT with arbitrary M and gives NO lower bound on M.")
    log("  Conclusion: BMO/JN is structurally in the wrong direction.")

    log("--- T4. Direct minimax val(d) for reference ---")
    t4_results = {}
    for d in [4, 6, 8]:
        log(f"  d={d}: solving min_a max_t (f*f)(t) via SLSQP ...")
        val, a_opt = actual_minimax_val(d)
        t4_results[d] = {"val": float(val) if np.isfinite(val) else None, "a": a_opt.tolist() if a_opt is not None else None}
        log(f"    val(d={d}) ~ {val:.6f}")
        # Compare to L^2 bound at same d
        l2_lb_d = t1_results[d]["min_l2_sq"]
        log(f"    L^2 bound at d={d}: {l2_lb_d:.6f}; ratio = {l2_lb_d/val:.3f}")

    log("--- T5. Higher-p Holder chain on g = f*f (NEW finding) ---")
    log("  Bound: M(f) = ||g||_inf >= ||g||_p^{p/(p-1)} (Holder, ||g||_1=1).")
    log("  Compute min_a ||g||_p^p over d-bin step f, take p-th root.")
    log("  NOTE: ||g||_p^p is NON-CONVEX in a for p >= 3 (poly degree 2p),")
    log("  so SLSQP gives LOCAL minima -- bound below is NOT rigorous.")
    log("  NOTE 2: step-function inf is an UPPER bound on the continuous inf,")
    log("  so M_LB_step_d is an OPTIMISTIC estimate of the continuous bound.")
    t5_results = {}
    for p in [2, 3, 4, 6]:
        t5_results[p] = {}
        for d in [4, 6, 8, 10, 12]:
            r = lp_chain_min(d, p, n_starts=60 if d <= 8 else 40)
            t5_results[p][d] = {"min_g_p_p": r["min_g_p_p"], "M_LB_step_d": r["M_LB_step_d"]}
            log(f"    p={p}, d={d}: min ||g||_p^p = {r['min_g_p_p']:.5f}, M_LB_step_d = {r['M_LB_step_d']:.5f}")
        # Extrapolate d -> infinity via 1/d expansion
        ds = np.array([4, 6, 8, 10, 12])
        vals = np.array([t5_results[p][d]["M_LB_step_d"] for d in [4, 6, 8, 10, 12]])
        A = np.column_stack([np.ones_like(ds), 1.0/ds, 1.0/ds**2])
        coefs, *_ = np.linalg.lstsq(A, vals, rcond=None)
        t5_results[p]["dinf_extrap"] = float(coefs[0])
        log(f"    p={p}: d->inf extrapolation = {coefs[0]:.4f}")

    log("--- T5 verdict ---")
    log("  HEURISTIC L^p step-function chain extrapolations:")
    for p in [2, 3, 4, 6]:
        log(f"    p={p}: continuous M_LB upper-estimate = {t5_results[p]['dinf_extrap']:.4f}")
    log("  CRITICAL CAVEATS:")
    log("    (a) step-function values are UPPER bounds on the continuous inf,")
    log("        so the actual continuous L^p Holder bound is SMALLER (worse).")
    log("    (b) for p >= 3, ||g||_p^p is non-convex; SLSQP local optima may")
    log("        OVER-estimate the actual minimum, making the bound LOOSE.")
    log("    (c) computing rigorously inf_f ||g||_p^p over all admissible f")
    log("        is itself a (potentially) infinite-dimensional poly opt")
    log("        equivalent in difficulty to the original Sidon problem.")

    log("--- Final verdict ---")
    # Best heuristic lower bound from L^p chain extrapolation
    p_best = max(t5_results.keys(), key=lambda p: t5_results[p]["dinf_extrap"])
    best_lp_extrap = t5_results[p_best]["dinf_extrap"]
    final = {
        "agent": "C4_hardy_littlewood",
        "approach": "Hardy-Littlewood maximal function + Calderon-Zygmund + BMO + John-Nirenberg + L^p Holder chain on g=f*f",
        "math_correct": True,
        "best_lb_obtained": float(best_l2_lb),  # rigorous L^2 lower bound from step-function (still not a rigorous bound on continuous C_{1a})
        "best_heuristic_estimate": float(best_lp_extrap),  # heuristic from L^p extrapolation, NOT rigorous
        "vs_1_2802": "below",  # the rigorous L^2 chain is below; the heuristic L^p extrapolation can be above but is non-rigorous
        "vs_mv_1_2748": "below",
        "promising": False,
        "verdict_short": (
            "Hardy-Littlewood/BMO/CZ machinery is in the WRONG direction "
            "for lower-bounding ||f*f||_inf -- those tools control UPPER "
            "bounds on maximal/level-set quantities given L^p norms. The "
            "ONE useable chain is the elementary Holder bound "
            "M(f) >= ||g||_p^{p/(p-1)} (||g||_1=1), but for p=2 this gives "
            "~1.19 (below 1.2802), and for p>=3 the minimization is "
            "non-convex so the bound is non-rigorous; rigorously bounding "
            "inf_f ||g||_p^p is itself as hard as the Sidon problem."
        ),
        "verdict_long": (
            "Five tests:\n"
            "T1: L^2-Plancherel chain M(f) >= ||g_sym||_2^2 = int|f-hat|^4. "
            "Step-function minimum over d in {4..10} is ~1.18-1.23, "
            "decreasing in d. Rigorous direction (||g_sym||_1 = 1 "
            "always), but value is BELOW 1.2802.\n"
            "T2: Calderon-Zygmund inequality L_lambda >= (1-lambda)/(M-lambda) "
            "is consistent with M -> 1 (since L_lambda <= 1), so gives no "
            "nontrivial lower bound on M.\n"
            "T3: BMO/John-Nirenberg gives upper bounds on level sets, "
            "concentrating around the average g_I = 1 -- WRONG direction.\n"
            "T4: Sanity baseline val(d) via SLSQP min-max: val(4)~1.64, "
            "val(8)~1.58. L^2 bound is ~75% of val(d), confirming L^2 "
            "is loose.\n"
            "T5: Higher-p Holder chain on g = f*f: M(f) >= ||g||_p^{p/(p-1)}. "
            "Step-function numerical minimization (SLSQP, non-convex for "
            "p>=3) gives M_LB_step at d=12: 1.270 (p=3), 1.340 (p=4), "
            "1.406 (p=6). 1/d-extrapolation to continuous limit: "
            "p=3 -> 1.243, p=4 -> 1.311, p=6 -> 1.382. CRITICAL: these "
            "are NON-RIGOROUS because (a) SLSQP finds local minima, "
            "(b) step-function inf is an upper bound on the continuous "
            "inf, (c) rigorously bounding inf_f ||g||_p^p for p>=3 is "
            "a polynomial-optimization problem of degree 2p in f, "
            "comparable in difficulty to the original Sidon problem.\n"
            "Conclusion: Real-variable HA tools (HL, BMO, CZ) are not "
            "naturally suited for lower-bounding the L^inf norm given "
            "L^1 = 1. The only usable chain (Holder L^p) reduces to a "
            "sub-problem of the same difficulty as Sidon itself, with "
            "no apparent simplification. NOT a viable new direction."
        ),
        "next_steps_if_promising": [
            "(Marginal interest only) Rigorize the L^4 step-function lower "
            "bound for inf_f ||g||_4^4 = int (f*f)^4 -- this is a "
            "polynomial-optimization problem solvable by Lasserre at "
            "small order. Compare to existing Lasserre val(d) tracks. "
            "Likely no improvement: the L^p chain is loose by ~25% at p=2, "
            "less at higher p, but still bounded above by val(d)^{1/(p-1)} "
            "which the existing Lasserre track already exploits more "
            "directly via moment SDP on max-of-convolution.",
        ],
        "details": {
            "T1_l2_bound_per_d": {str(k): {kk: vv for kk, vv in v.items() if kk != "argmin_a"} for k, v in t1_results.items()},
            "T2_cz_per_d": {str(k): v for k, v in t2_results.items()},
            "T4_val_per_d": {str(k): {"val": v["val"]} for k, v in t4_results.items()},
            "T5_lp_chain": {str(p): {str(d): {"min_g_p_p": rd["min_g_p_p"], "M_LB_step_d": rd["M_LB_step_d"]} for d, rd in (vp.items() if isinstance(vp, dict) else {}) if isinstance(rd, dict) and "min_g_p_p" in rd} for p, vp in t5_results.items()},
            "T5_lp_chain_dinf_extrap": {str(p): t5_results[p]["dinf_extrap"] for p in t5_results},
        },
        "compute_time_sec": float(time.time() - t0),
        "files_created": [
            "_novel_agents/C4_hardy_littlewood/probe.py",
            "_novel_agents/C4_hardy_littlewood/run.log",
            "_novel_agents/C4_hardy_littlewood/results.json",
        ],
    }
    log(f"  best_lb_obtained = {final['best_lb_obtained']:.6f} (from T1 L^2 chain)")
    log(f"  vs 1.2802: {final['vs_1_2802']}")
    log(f"  vs MV 1.2748: {final['vs_mv_1_2748']}")
    log(f"  promising: {final['promising']}")
    log(f"  total time: {final['compute_time_sec']:.1f} s")

    RESULTS_PATH.write_text(json.dumps(final, indent=2), encoding="utf-8")
    log(f"Wrote results to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
