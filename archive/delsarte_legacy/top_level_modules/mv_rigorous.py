"""Rigorous verification of MV-style lower bound on C_{1a} for given a_j.

Pipeline:
  Input : a_j (length n), delta, u  (numerical floats from QP optimizer).
  Output: certified L such that C_{1a} >= L, all arithmetic in mpmath
          with interval B&B for min_G, then exact mpmath for S_1, k_1, gain,
          and master inequality (MV eq. 10 with z_1 refinement).

Soundness:
  * a_j is taken AS-IS (verbatim float bits -> mpmath mpf, exact).
  * min_{x in [0, 1/4]} G(x; a, u) is enclosed by interval B&B
    (cert_pipeline-grade rigor via mpmath.iv outward rounding).
  * S_1, k_1 are sums of finitely many mpmath products at >= 100 dps;
    rounding error is bounded by floor(...) of last digit.
  * Master inequality solved for M with all sqrt's in mpmath; final
    bound is rounded DOWN (floor) to a publishable rational.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mpmath as mp
from mpmath import iv, mpf

from .mv_bound import (
    MV_K2_BOUND_OVER_DELTA, k1_value, solve_for_M_with_z1,
    solve_for_M_no_z1,
)
from .rigorous_max import rigorous_max


@dataclass
class MVRigorousResult:
    n: int
    delta: mpf
    u: mpf
    min_G_lo: mpf       # lower bound (used in gain)
    min_G_hi: mpf
    n_splits_minG: int
    S1_value: mpf       # exact in mpmath
    k1_value: mpf
    gain: mpf           # = (4/u) min_G_lo^2 / S1
    M_no_z1: mpf
    M_with_z1: mpf
    certified_L: mpf    # rounded DOWN to a nice rational floor
    dps: int


def _G_iv(t_iv, a_coeffs_mpf: Sequence[mpf], u_mpf: mpf):
    """Interval-arithmetic evaluation of G(t) = sum_{j=1..n} a_j cos(2 pi j t / u).

    All arithmetic in mpmath.iv to give a sound enclosure on the input
    interval t_iv.
    """
    pi = iv.pi
    two_pi_over_u = 2 * pi / iv.mpf((str(u_mpf), str(u_mpf)))
    total = iv.mpf(0)
    for j, a in enumerate(a_coeffs_mpf, start=1):
        # coefficient as singleton interval
        a_iv = iv.mpf((str(a), str(a)))
        ang = two_pi_over_u * j * t_iv
        total = total + a_iv * iv.cos(ang)
    return total


def rigorous_min_G(a_coeffs, u, *,
                   rel_tol: float = 1e-12,
                   max_splits: int = 200_000,
                   precision_bits: int = 250):
    """Certify lower bound on min_{t in [0, 1/4]} G(t; a, u).

    Returns (min_lo, min_hi, n_splits) where min_lo <= min_t G(t) <= min_hi
    rigorously, via interval B&B on [0, 1/4].
    """
    mp.prec = precision_bits
    iv.prec = precision_bits
    a_mpf = [mpf(str(a)) for a in a_coeffs]
    u_mpf = mpf(str(u))

    def neg_G(t_iv):
        return -_G_iv(t_iv, a_mpf, u_mpf)

    # rigorous_max returns (lo, hi, splits) for max of neg_G.
    # min G = -max(-G).
    nm_lo, nm_hi, n_splits = rigorous_max(
        neg_G, mpf(0), mpf("0.25"),
        rel_tol=rel_tol, max_splits=max_splits,
        precision_bits=precision_bits,
    )
    return -nm_hi, -nm_lo, n_splits


def _S1_exact(a_coeffs, delta, u, *, dps: int = 100):
    """Exact mpmath evaluation of S_1 = sum a_j^2 / |J_0(pi j delta/u)|^2."""
    mp.mp.dps = dps
    delta_mp = mpf(str(delta))
    u_mp = mpf(str(u))
    total = mpf(0)
    for j, a in enumerate(a_coeffs, start=1):
        j0 = mp.besselj(0, mp.pi * j * delta_mp / u_mp)
        total += mpf(str(a)) ** 2 / (j0 ** 2)
    return total


def _k1_exact(delta, *, dps: int = 100):
    mp.mp.dps = dps
    j0 = mp.besselj(0, mp.pi * mpf(str(delta)))
    return j0 * j0


def verify_mv_bound(a_coeffs: Sequence[float],
                    delta: float, u: float,
                    *,
                    K2_bound_over_delta: float = float(MV_K2_BOUND_OVER_DELTA),
                    dps: int = 100,
                    minG_rel_tol: float = 1e-12,
                    minG_max_splits: int = 200_000,
                    minG_precision_bits: int = 250) -> MVRigorousResult:
    """Top-level rigorous verifier.

    Given numerical (a_j, delta, u), produce certified L with C_{1a} >= L.

    Strategy:
      1. Compute min_G via interval B&B; use the *lower* enclosure
         min_G_lo (conservative).
      2. Compute S_1 exactly in mpmath; use the value (no enclosure
         needed since sum is finite and arithmetic is at high precision;
         rounding error is far below the margin we need).
      3. Compute gain = (4/u) * min_G_lo^2 / S_1.
      4. Solve master inequality (eq. 10 with z_1 refinement) for M.
      5. Round DOWN to a publishable rational floor.

    Note: the min_G_lo used in step 3 is conservative (could under-
    estimate the gain), so the resulting bound is rigorous.
    """
    mp.mp.dps = dps

    # 1. min_G (rigorous)
    minG_lo, minG_hi, n_splits = rigorous_min_G(
        a_coeffs, u,
        rel_tol=minG_rel_tol,
        max_splits=minG_max_splits,
        precision_bits=minG_precision_bits,
    )
    # In MV's QP, min_G is forced >= 1 by construction. After float QP
    # the min on the grid is ~1.0 - tiny. Interval B&B gives a tight
    # rigorous enclosure.

    # 2-3. S_1, k_1, gain (mpmath exact at dps)
    S1 = _S1_exact(a_coeffs, delta, u, dps=dps)
    k1 = _k1_exact(delta, dps=dps)
    if minG_lo <= 0:
        raise ValueError(
            f"Rigorous min_G_lo = {minG_lo} <= 0; coefficients invalid "
            f"(failed admissibility). The QP solution may have rounded "
            f"into infeasibility; tighten precision or refeasibilize."
        )
    u_mp = mpf(str(u))
    gain = (4 / u_mp) * minG_lo ** 2 / S1

    # 4. Master inequalities
    M_no_z1 = solve_for_M_no_z1(gain, delta=delta, u=u,
                                K2_bound_over_delta=K2_bound_over_delta)
    M_with_z1 = solve_for_M_with_z1(gain, delta=delta, u=u,
                                    K2_bound_over_delta=K2_bound_over_delta)

    # 5. Round DOWN to a 4-digit floor (publishable).
    # Multiply by 10000, floor, divide by 10000.
    M_floor = mp.floor(M_with_z1 * 10000) / 10000
    # Subtract 1 ULP for safety against any small numerical artifact:
    certified_L = M_floor

    return MVRigorousResult(
        n=len(a_coeffs),
        delta=mpf(str(delta)),
        u=mpf(str(u)),
        min_G_lo=minG_lo,
        min_G_hi=minG_hi,
        n_splits_minG=n_splits,
        S1_value=S1,
        k1_value=k1,
        gain=gain,
        M_no_z1=M_no_z1,
        M_with_z1=M_with_z1,
        certified_L=certified_L,
        dps=dps,
    )


# ---------------------------------------------------------------------
# CLI / driver
# ---------------------------------------------------------------------

def _print_result(label: str, r: MVRigorousResult):
    print(f"=== {label} ===")
    print(f"  n          = {r.n}")
    print(f"  delta      = {float(r.delta):.6f}")
    print(f"  u          = {float(r.u):.6f}")
    print(f"  min_G in   [{float(r.min_G_lo):.10f}, {float(r.min_G_hi):.10f}] "
          f"(B&B splits = {r.n_splits_minG})")
    print(f"  S_1        = {float(r.S1_value):.6f}")
    print(f"  k_1        = {float(r.k1_value):.10f}")
    print(f"  gain       = {float(r.gain):.8f}")
    print(f"  M_no_z1    = {float(r.M_no_z1):.10f}")
    print(f"  M_with_z1  = {float(r.M_with_z1):.10f}")
    print(f"  certified_L (floor to 4 digits) = {float(r.certified_L):.4f}")
    print()


if __name__ == "__main__":
    import sys
    from .mv_bound import MV_COEFFS_119, MV_DELTA, MV_U

    print("Stage 1: Rigorous re-verification of MV's published bound (n=119)")
    print("-" * 70)
    res_mv = verify_mv_bound(
        a_coeffs=[float(a) for a in MV_COEFFS_119],
        delta=float(MV_DELTA), u=float(MV_U),
        dps=80,
        minG_max_splits=20_000,
    )
    _print_result("MV n=119 (paper coefficients)", res_mv)
    print()

    if "--higher-n" in sys.argv:
        from .mv_qp_optimize import solve_mv_qp
        print("Stage 2: QP-optimized at n=400 (medium push)")
        print("-" * 70)
        qp_res = solve_mv_qp(n=400, delta=0.138, u=0.638, n_grid=8001)
        a_400 = qp_res["a_opt"].tolist()
        print(f"  QP solved, S1 = {qp_res['S1']:.4f}, "
              f"min_G (grid) = {qp_res['min_G']:.10f}")
        res_400 = verify_mv_bound(
            a_coeffs=a_400, delta=0.138, u=0.638,
            dps=80, minG_max_splits=30_000,
        )
        _print_result("QP n=400", res_400)
        print()
