"""Joint (delta, u) scan with MV's fixed 119 coefficients.

Idea D2: MV explicitly state "we set u = 1/2 + delta for convenience"
(MV paper, p. 4 lines 190-194). They give NO proof that this choice is optimal.
This module relaxes the convenience constraint and scans (delta, u) jointly
with u > 1/2 + delta, re-evaluating the gain parameter `a` and the Cauchy-
Schwarz master-inequality bound at each (delta, u) using MV's fixed a_j.

This is a LIGHT-COMPUTE exploration: the 119 coefficients a_j are held FIXED
at MV's QP-optimal values (at delta=0.138, u=0.638). We do NOT re-solve the
semi-infinite QP; instead, we simply re-evaluate:
    - min_{[0, 1/4]} G(x; u)         (depends on u through cos(2 pi j x/u))
    - S_1(delta, u) = sum a_j^2 / |J_0(pi j delta / u)|^2
    - gain a = (4/u) * (min_G)^2 / S_1
    - k_1(delta) = |J_0(pi delta)|^2
    - ||K||_2^2 = C_0 / delta with C_0 approx 0.5747 (independent of delta)
    - Solve MV master inequality (eq. 10, with z_1 refinement) for M.

Since a_j are NOT re-optimized, at (delta, u) != (0.138, 0.638) the constraint
min_G >= 1 may fail (min_G < 1 degrades the gain). The scan is honest: it
uses min_G^2 as-is (which may be < 1), so the gain may go down rather than
up. The scan reports whatever it finds.

Precondition (Lemma 3.1 / MV's Parseval framework)
--------------------------------------------------
The MV Cauchy-Schwarz dual framework requires:
    0 < delta <= 1/4,
    u >= 1/2 + delta        (so that the support [-delta, delta] of K fits
                             inside one period, and the Parseval identity
                             used in eq. (3)/(6) is valid on [-u/2, u/2]).
MV actually use u = 1/2 + delta (the tightest case). We relax to u > 1/2 + delta.

||K||_2^2 independence of delta
-------------------------------
MV write ||K||_2^2 < 0.5747 / delta (p. 3 line 141). Deriving:

    K(x) = (1/delta) * eta(x/delta),   eta(y) = (2/pi) / sqrt(1 - 4 y^2) on (-1/2, 1/2).

By a change of variables y = x/delta,
    ||K||_2^2 = int K(x)^2 dx = (1/delta^2) int eta(x/delta)^2 dx
              = (1/delta) int_{-1/2}^{1/2} eta(y)^2 dy.
The integral on the right is (formally; in MV's regularised sense, via
Parseval in period-u: sum of |Bessel|^2 = integral of eta^2), related to

    int_{-infty}^{infty} |J_0(pi xi)|^4 dxi = C_0 ~ 0.5747,

which is a NUMERICAL CONSTANT independent of delta. So
    ||K||_2^2 = C_0 / delta      at every delta,
not just MV's 0.138. This validates using the SAME 0.5747 across the scan.

Usage
-----
    python -m delsarte_dual.mv_delta_u_scan   # runs the baseline check and grid scan
or importable:
    from delsarte_dual.mv_delta_u_scan import evaluate_at, scan_delta_u

Do NOT modify any existing file; this file is self-contained and read-only
w.r.t. mv_bound.py.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import mpmath as mp
import numpy as np
from mpmath import mpf

# Import MV's fixed coefficients and the master-inequality solver.
# NOTE: we use absolute import so this file can be run as a script
# (python delsarte_dual/mv_delta_u_scan.py) or as a module.
try:
    from delsarte_dual.mv_bound import (
        MV_COEFFS_119,
        MV_DELTA,
        MV_U,
        MV_K2_BOUND_OVER_DELTA,
        MV_BOUND_FINAL,
        k1_value,
        solve_for_M_with_z1,
        solve_for_M_no_z1,
    )
except ImportError:
    # Fallback for when this file is executed from within delsarte_dual/.
    from mv_bound import (  # type: ignore
        MV_COEFFS_119,
        MV_DELTA,
        MV_U,
        MV_K2_BOUND_OVER_DELTA,
        MV_BOUND_FINAL,
        k1_value,
        solve_for_M_with_z1,
        solve_for_M_no_z1,
    )


# -----------------------------------------------------------------------------
# Numerical constant: C_0 = int_R |J_0(pi xi)|^4 d xi  (approx. 0.5747)
# -----------------------------------------------------------------------------

# MV use the value 0.5747 (p. 3 line 141). We keep this value to stay bit-for-bit
# faithful at the baseline (delta, u) = (0.138, 0.638) reproducing 1.27481.
# The CORRECT mathematical statement is ||K||_2^2 = C_0 / delta with
# C_0 = int |J_0(pi xi)|^4 d xi independent of delta.
MV_C0 = mpf("0.5747")


def K2_value(delta) -> mpf:
    """Return ||K||_2^2 = C_0 / delta at the given delta.

    C_0 ~ 0.5747 is INDEPENDENT of delta (see module docstring). Using the
    SAME numerical value as MV makes the baseline match their 1.27481 exactly.
    """
    return MV_C0 / mpf(delta)


# -----------------------------------------------------------------------------
# Re-evaluations (G minimum, S_1) at arbitrary (delta, u)
# -----------------------------------------------------------------------------

def min_G_on_0_quarter_at_u(
    a_coeffs: Sequence = MV_COEFFS_119,
    u: float = float(MV_U),
    n_grid: int = 40001,
) -> tuple[float, float]:
    """Compute min_{x in [0, 1/4]} G(x; u) on a fine grid, returning (value, argmin).

    G(x; u) = sum_{j=1}^n a_j cos(2 pi j x / u).

    This is a grid approximation (NOT a rigorous enclosure). For the bound
    to be mathematically rigorous one would need an interval branch-and-bound
    or a Lipschitz-controlled grid; for a LIGHT exploratory scan a fine grid
    with n_grid=40001 (step ~6.25e-6 on [0, 1/4]) is standard and matches
    `mv_bound.min_G_on_0_quarter`'s approach.
    """
    u_f = float(u)
    a_f = np.asarray([float(a) for a in a_coeffs], dtype=np.float64)
    n = a_f.size
    xs = np.linspace(0.0, 0.25, int(n_grid))
    # Vectorised evaluation: vals[i] = sum_j a_j cos(2 pi j xs[i] / u)
    # Build the n_grid x n matrix cos(2 pi j x / u) only implicitly (loop over j).
    vals = np.zeros_like(xs)
    two_pi_over_u = 2.0 * np.pi / u_f
    for j in range(1, n + 1):
        vals += a_f[j - 1] * np.cos(two_pi_over_u * j * xs)
    idx = int(vals.argmin())
    return float(vals[idx]), float(xs[idx])


def S1_sum_at(
    a_coeffs: Sequence = MV_COEFFS_119,
    delta: float = float(MV_DELTA),
    u: float = float(MV_U),
) -> mpf:
    """S_1 = sum_{j=1}^n a_j^2 / |J_0(pi j delta / u)|^2, in mpmath.

    This reproduces `mv_bound.S1_sum` but is parameterised over (delta, u).
    """
    delta_m = mpf(delta)
    u_m = mpf(u)
    pi = mp.pi
    total = mpf(0)
    for j, a in enumerate(a_coeffs, start=1):
        j0 = mp.besselj(0, pi * j * delta_m / u_m)
        total += mpf(a) ** 2 / (j0 ** 2)
    return total


# -----------------------------------------------------------------------------
# Core: evaluate MV's master-inequality bound at a single (delta, u)
# -----------------------------------------------------------------------------

def evaluate_at(
    delta: float,
    u: float,
    a_coeffs: Sequence = MV_COEFFS_119,
    n_grid_minG: int = 40001,
    z1: str = "0.50426",
    dps: int = 30,
    strict_precondition: bool = True,
) -> dict:
    """Compute MV's master-inequality bound at (delta, u) with fixed a_j.

    Parameters
    ----------
    delta, u : float
        Parameters with 0 < delta <= 1/4 and u >= 1/2 + delta.
    a_coeffs : sequence of mpf
        The n coefficients a_j of G (default: MV's 119 tabulated values).
    n_grid_minG : int
        Grid resolution for computing min G on [0, 1/4].
    z1 : str
        Lemma-3.4 ceiling on |hat f(1)|; default 0.50426 per MV.
    dps : int
        Decimal precision for mpmath.
    strict_precondition : bool
        If True (default), RAISE if u < 1/2 + delta. The MV / Lemma 3.1
        framework is NOT valid below this threshold.

    Returns
    -------
    dict with keys:
        delta, u, valid, min_G, argmin_G, S1, gain, k1, K2,
        M_no_z1, M_with_z1, feasible_minG_ge_1.

    Raises
    ------
    ValueError if strict_precondition is True and u < 1/2 + delta.
    """
    mp.mp.dps = dps
    delta_f = float(delta)
    u_f = float(u)

    # --- Precondition check: u >= 1/2 + delta (MV / Lemma 3.1 framework) ---
    if u_f < 0.5 + delta_f - 1e-15:
        msg = (
            f"Precondition violated: u={u_f} < 1/2 + delta = {0.5 + delta_f}. "
            "MV's Lemma 3.1 Parseval argument requires u >= 1/2 + delta."
        )
        if strict_precondition:
            raise ValueError(msg)
        return {
            "delta": delta_f, "u": u_f, "valid": False, "reason": msg,
            "min_G": None, "argmin_G": None, "S1": None, "gain": None,
            "k1": None, "K2": None, "M_no_z1": None, "M_with_z1": None,
            "feasible_minG_ge_1": None,
        }

    # Also must have 0 < delta <= 1/4.
    if not (0 < delta_f <= 0.25):
        raise ValueError(f"Require 0 < delta <= 1/4; got delta={delta_f}.")

    # --- Re-evaluate min_G on [0, 1/4] at the new u ---
    min_G, argmin_G = min_G_on_0_quarter_at_u(
        a_coeffs=a_coeffs, u=u_f, n_grid=n_grid_minG
    )

    # --- ADMISSIBILITY CHECK (fix for audit bug #1) ---
    # MV Lemma 3.1(4) requires G > 0 on [-1/4, 1/4].  If min G <= 0 at the
    # new (delta, u) with these fixed a_j, the triple is INADMISSIBLE and
    # the MV master inequality does not apply — the reported M would be
    # mathematically invalid.
    if min_G <= 0:
        return {
            "delta": delta_f, "u": u_f,
            "valid": False,
            "reason": (
                f"Inadmissible triple at (delta, u) = ({delta_f}, {u_f}): "
                f"min_G = {min_G} <= 0.  MV's Lemma 3.1(4) requires G > 0 "
                f"on [-1/4, 1/4]."
            ),
            "min_G": min_G, "argmin_G": argmin_G,
            "S1": None, "gain": None, "k1": None, "K2": None,
            "M_no_z1": None, "M_with_z1": None,
            "feasible_minG_ge_1": False,
        }

    # --- S_1 at new (delta, u) ---
    S1 = S1_sum_at(a_coeffs=a_coeffs, delta=delta_f, u=u_f)

    # --- Gain a = (4/u) * min_G^2 / S_1 ---
    # min_G is strictly positive (admissibility checked above).
    gain = (mpf(4) / mpf(u_f)) * mpf(min_G) ** 2 / S1

    # --- Other constants ---
    k1 = k1_value(delta_f)
    K2 = K2_value(delta_f)

    # --- Master inequality (no z_1 refinement) ---
    try:
        M_no_z1 = solve_for_M_no_z1(
            gain, delta=delta_f, u=u_f,
            K2_bound_over_delta=MV_K2_BOUND_OVER_DELTA,
        )
    except Exception as e:
        M_no_z1 = None

    # --- Master inequality with z_1 refinement (fix for audit bug #2) ---
    # The MV dual bound at the current (delta, u) requires MAX over z_1 in
    # [0, z_max(M)] of the RHS, where z_max(M) = sqrt(M sin(pi/M)/pi)
    # (Lemma 3.4).  The maximiser is either the interior critical point
    # t* = k_1 sqrt((M-1)/(B^2 + 2 k_1^2))  (in t = z_1^2 coords)
    # or the boundary t_max = z_max(M)^2, whichever is smaller (see B3
    # module mv_union_forbidden.py Phi(t) unimodality argument).
    #
    # Since the optimal z_1 depends on M, and M depends on z_1 through
    # solve_for_M_with_z1, we iterate to a self-consistent fixed point.
    # At MV's canonical (0.138, 0.638) this converges in 1-2 iterations
    # to z_1 = 0.50426, exactly reproducing MV's tabulated value.
    def best_z1_for_M(M_val: mpf) -> mpf:
        """Return the z_1 in [0, z_max(M)] that maximises RHS(M, z_1)."""
        M_val = mpf(M_val)
        # Interior critical point in t = z_1^2
        B2 = K2 - 1 - 2 * k1 * k1
        if B2 < 0:
            # Second sqrt radicand negative; master inequality ill-defined
            return mp.mpf("-inf")
        t_star = k1 * mp.sqrt((M_val - 1) / (B2 + 2 * k1 * k1))
        t_max = M_val * mp.sin(mp.pi / M_val) / mp.pi  # = z_max(M)^2
        # rad1 = M - 1 - 2 t^2 >= 0  <=>  t <= sqrt((M-1)/2)   (t = z_1^2 here)
        t_sqrt_dom = mp.sqrt((M_val - 1) / 2)
        t_opt = min(t_star, t_max, t_sqrt_dom)
        if t_opt < 0:
            t_opt = mpf(0)
        return mp.sqrt(t_opt)

    try:
        # Fixed-point iteration on (M, z_1)
        M_current = mpf("1.28")  # initial guess bracketing MV's answer
        z1_current = best_z1_for_M(M_current)
        for _it in range(50):
            M_new = solve_for_M_with_z1(
                gain, delta=delta_f, u=u_f,
                K2_bound_over_delta=MV_K2_BOUND_OVER_DELTA,
                z1=z1_current,
            )
            if abs(M_new - M_current) < mpf("1e-12"):
                M_current = M_new
                break
            M_current = M_new
            z1_current = best_z1_for_M(M_current)
        M_with_z1 = M_current
    except Exception as e:
        M_with_z1 = None

    return {
        "delta": delta_f,
        "u": u_f,
        "valid": True,
        "min_G": min_G,
        "argmin_G": argmin_G,
        "S1": S1,
        "gain": gain,
        "k1": k1,
        "K2": K2,
        "M_no_z1": M_no_z1,
        "M_with_z1": M_with_z1,
        "feasible_minG_ge_1": bool(min_G >= 1.0),
    }


# -----------------------------------------------------------------------------
# Grid scan
# -----------------------------------------------------------------------------

def _scan_worker(delta, u, a_coeffs, n_grid_minG, z1, dps):
    """Top-level worker for joblib (must be picklable)."""
    if float(u) < 0.5 + float(delta) - 1e-15:
        return None
    try:
        return evaluate_at(
            delta=delta, u=u, a_coeffs=a_coeffs,
            n_grid_minG=n_grid_minG, z1=z1, dps=dps,
            strict_precondition=True,
        )
    except Exception as exc:
        return {"delta": float(delta), "u": float(u),
                "valid": False, "reason": str(exc),
                "min_G": None, "gain": None, "M_with_z1": None}


def scan_delta_u(
    delta_values: Iterable[float],
    u_values: Iterable[float],
    a_coeffs: Sequence = MV_COEFFS_119,
    n_grid_minG: int = 40001,
    z1: str = "0.50426",
    dps: int = 30,
    verbose: bool = True,
    n_jobs: int = -1,
) -> tuple[dict, list[dict]]:
    """Scan (delta, u) in parallel via joblib.

    Respects u >= 1/2 + delta.  Returns (best, all_results).
    """
    from joblib import Parallel, delayed
    deltas = list(delta_values)
    us = list(u_values)
    pairs = [(d, u) for d in deltas for u in us]

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_scan_worker)(d, u, a_coeffs, n_grid_minG, z1, dps)
        for (d, u) in pairs
    )

    all_results = [r for r in results if r is not None]
    best: dict | None = None
    for res in all_results:
        if verbose:
            if not res.get("valid", True):
                print(f"  delta={res['delta']:.4f} u={res['u']:.4f} "
                      f"INADMISSIBLE: min_G={res.get('min_G')}")
            else:
                M_z1 = res["M_with_z1"]
                print(
                    f"  delta={res['delta']:.4f} u={res['u']:.4f} "
                    f"min_G={res['min_G']:+.5f} "
                    f"gain={float(res['gain']):.5f} "
                    f"M_z1={float(M_z1) if M_z1 is not None else float('nan'):.6f} "
                    f"feasible(minG>=1)={res['feasible_minG_ge_1']}"
                )
        if res.get("M_with_z1") is None:
            continue
        if best is None or (
            float(res["M_with_z1"]) > float(best["M_with_z1"])
        ):
            best = res

    if best is None:
        best = {"M_with_z1": mpf("-inf"), "delta": None, "u": None,
                "note": "no valid grid points"}
    return best, all_results


# -----------------------------------------------------------------------------
# Main: baseline check + scan
# -----------------------------------------------------------------------------

def run_baseline_check(dps: int = 30, n_grid_minG: int = 40001,
                       tol: float = 1e-4) -> dict:
    """Reproduce MV's 1.27481 at (delta, u) = (0.138, 0.638).

    This is the correctness anchor for the whole scan: if the baseline does
    not match, there is a bug and the scan is meaningless.
    """
    res = evaluate_at(
        delta=float(MV_DELTA), u=float(MV_U),
        a_coeffs=MV_COEFFS_119, n_grid_minG=n_grid_minG,
        z1="0.50426", dps=dps,
    )
    M = float(res["M_with_z1"])
    target = float(MV_BOUND_FINAL)
    ok = abs(M - target) < tol
    res["baseline_ok"] = bool(ok)
    res["baseline_target"] = target
    res["baseline_diff"] = M - target
    return res


if __name__ == "__main__":
    mp.mp.dps = 30

    print("=" * 72)
    print("MV joint (delta, u) scan — idea D2")
    print("=" * 72)
    print()

    # ---------------- 1. Baseline reproduction ----------------
    print("[1] Baseline check at (delta, u) = (0.138, 0.638)")
    print("-" * 72)
    base = run_baseline_check(dps=30, n_grid_minG=40001, tol=1e-4)
    print(f"  min_G         = {base['min_G']:+.8f}")
    print(f"  argmin_G      = {base['argmin_G']:.6f}")
    print(f"  S_1           = {float(base['S1']):.6f}")
    print(f"  gain a        = {float(base['gain']):.6f}")
    print(f"  k_1           = {float(base['k1']):.6f}")
    print(f"  K_2 bound     = {float(base['K2']):.6f}")
    print(f"  M_no_z1       = {float(base['M_no_z1']):.6f}")
    print(f"  M_with_z1     = {float(base['M_with_z1']):.6f}   "
          f"(MV target: {float(MV_BOUND_FINAL):.5f})")
    print(f"  baseline_ok   = {base['baseline_ok']}  "
          f"(diff = {base['baseline_diff']:+.2e})")
    print()
    if not base["baseline_ok"]:
        print("  WARNING: baseline does not match MV's 1.27481 within tol=1e-4.")
        print("           The scan results below are unreliable until this is fixed.")
        print()

    # ---------------- 2. Grid scan ----------------
    # delta in [0.10, 0.18], u in [1/2 + delta, 1.0]. 20 x 20 grid.
    # We generate u values per-delta so the precondition u >= 1/2 + delta is
    # guaranteed for every pair we try (no wasted skips).
    print("[2] Grid scan: delta in [0.10, 0.18] x u in [1/2 + delta, 1.0], 20 x 20")
    print("-" * 72)
    delta_grid = np.linspace(0.10, 0.18, 20)
    # For each delta we use a 20-point u grid from 1/2 + delta to 1.0. But
    # scan_delta_u takes a single u_values list; we therefore use a wider u
    # grid and let the precondition filter do the rest.
    # Wider u grid: union of [1/2 + delta_min, 1.0] = [0.60, 1.0].
    u_grid = np.linspace(0.60, 1.0, 20)

    best, all_results = scan_delta_u(
        delta_values=delta_grid, u_values=u_grid,
        a_coeffs=MV_COEFFS_119, n_grid_minG=40001,
        z1="0.50426", dps=30, verbose=True,
    )

    print()
    print("-" * 72)
    if best.get("delta") is None:
        print("  No valid grid points (all violated precondition). "
              "Widen the u range.")
    else:
        print(f"  BEST: delta={best['delta']:.4f} u={best['u']:.4f} "
              f"M_with_z1={float(best['M_with_z1']):.6f}")
        print(f"  MV baseline:  delta=0.1380 u=0.6380 "
              f"M_with_z1={float(MV_BOUND_FINAL):.6f}")
        improvement = float(best["M_with_z1"]) - float(MV_BOUND_FINAL)
        print(f"  Improvement over MV: {improvement:+.6f}")
        if improvement > 0:
            print("  NOTE: improvement is with FIXED MV a_j; relaxing u may be")
            print("        lucky. To certify, one should re-solve the QP at (delta, u*).")
        else:
            print("  NOTE: MV's choice u = 1/2 + delta appears (near-)optimal")
            print("        when the a_j are held fixed at their (0.138, 0.638) values.")
    print()
    print("DONE.")
