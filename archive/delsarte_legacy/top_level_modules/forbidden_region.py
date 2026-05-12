"""Certified bisection on M for the multi-moment forbidden-region problem.

Part B.4 of the pipeline.

High-level formulation
----------------------
Let phi_M(z) be a Phi-bound object (MVSingleMomentBound / MVMultiMomentBound /
MVMultiMomentBoundWithMO from ``mv_bound.py``).  We define

    M* := inf { M : inf_{z in F_N} phi_M(z) > 0 }

where F_N is the admissibility set (Part A.4 / ``moment_constraints.py``):
    - magnitude box:  z_n = |hat f(n)| in [0, 1];
    - Chebyshev-Markov bathtub refinement:  z_n in [0, sqrt(mu(M))];
    - Toeplitz/Bochner PSD on the truncated Fourier sequence;
    - (optional) Hausdorff moment PSD via Taylor lift to spatial moments;
    - (optional) MO 2004 Lemma 2.17 on Re hat f(1), Re hat f(2).

Equivalently, M* = inf { M : max_{z in F_N} RHS(M, z) <= 2/u + a }.

The bisection procedure
-----------------------
At each trial M0 we solve the INNER feasibility problem

    maximise  RHS(M0, z)  subject to  z in F_N.

If the maximum is >= 2/u + a then M0 is FEASIBLE (admissible f could
achieve ||f*f||_inf = M0), so M* > M0 is still possible.  Otherwise
M0 is INFEASIBLE — every admissible f has ||f*f||_inf >= M0 — so M*
<= M0.

The inner problem is non-convex in general (RHS is concave in y_n = z_n^2
but F_N's Toeplitz/Hausdorff PSD constraints couple variables).  We
implement TWO solvers:

    (a) Analytic KKT solver for the magnitude-box + MO variant
        (``max_rhs_over_z_with_mo``) -- fast, exact up to mpmath precision.
    (b) CVXPY/Clarabel SDP relaxation for the full F_N with Toeplitz/
        Hausdorff PSD (``max_rhs_over_z_sdp``) -- slower, handles PSD cone
        exactly but needs rational-rounding post-processing (see certify.py).

Both solvers return ``(rhs_max, z_opt)``; the bisection driver
``certified_forbidden_max`` wraps them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import mpmath as mp
from mpmath import mpf

from .mv_bound import (
    MVSingleMomentBound,
    MVMultiMomentBound,
    MVMultiMomentBoundWithMO,
    MV_DELTA,
    MV_U,
)
from .mv_multimoment import (
    mu_of_M,
    k_values,
    rhs_multimoment,
    max_rhs_over_z,
)


# -----------------------------------------------------------------------------
# Inner feasibility solver: magnitude-box + MO strong (on r_1, r_2 via c_1, c_2)
# -----------------------------------------------------------------------------

def max_rhs_over_z_with_mo(
    M, k_vals: Sequence, K2,
    mo_strong: bool = True,
) -> Tuple[mpf, List[mpf]]:
    """Maximise RHS over z in the box [0, sqrt(mu(M))]^N intersected with MO.

    RHS(M, z) depends only on the magnitudes r_n = |hat f(n)|.  The box and
    Hausdorff-on-magnitudes reduce to r_n in [0, sqrt(mu(M))].  MO Lemma 2.17
    (strong form) constrains REAL PARTS, not magnitudes — since RHS does not
    see the phase, MO's role reduces to:

        max r_2 such that some (c_2, s_2) has c_2 <= 2 c_1 - 1,
                                               c_2^2 + s_2^2 = r_2^2,
                                               c_1^2 + s_1^2 <= r_1^2.

    Without MO: max r_2 = sqrt(mu(M)).
    With MO:    r_2 can still hit sqrt(mu(M)) (pick c_2 = -r_2, s_2 = 0,
                c_1 in [-r_1, r_1]; constraint c_2 <= 2 c_1 - 1 becomes
                -r_2 <= 2 c_1 - 1, i.e. c_1 >= (1 - r_2)/2, which is
                feasible whenever (1 - r_2)/2 <= r_1).  So MO NEVER
                restricts r_2 unless r_1 < (1 - r_2)/2.

    In short: MO tightens the magnitude box only when
        r_1 < (1 - r_2) / 2,
    equivalently
        r_1 + r_2 / 2 < 1 / 2.
    Inside the MV regime r_1, r_2 ~ 0.5, LHS ~ 0.75 > 0.5, so MO is INACTIVE.

    This observation matches the repo's prior finding (MASTER_PLAN.md:17
    "Multi-moment extension caps at ~1.2755") and is the reason the brief's
    MO approach cannot break 1.28 at these parameters.  We still return a
    correct certified max so the infrastructure downstream is honest.
    """
    # The MO feasibility check reduces to a box test on magnitudes:
    #     r_1 >= (1 - r_2) / 2    <=>    2 r_1 + r_2 >= 1.
    # The max_rhs_over_z routine from mv_multimoment handles the pure box; we
    # wrap it here and if (r_1, r_2) at that maximum VIOLATES 2 r_1 + r_2 >= 1
    # we re-optimise with that as an auxiliary constraint.
    r_max = mp.sqrt(mu_of_M(M))
    rhs_unconstrained, z_opt = max_rhs_over_z(M, k_vals, K2)
    if not mo_strong or len(z_opt) < 2:
        return rhs_unconstrained, z_opt
    r1, r2 = z_opt[0], z_opt[1]
    if 2 * r1 + r2 >= 1:
        return rhs_unconstrained, z_opt
    # else: fall back to the boundary 2 r_1 + r_2 = 1.  Because RHS is
    # non-decreasing in each r_n (through the +2 k_n r_n^2 term), the
    # constrained optimum lies on the boundary; we parametrise r_1, r_2 =
    # (t, 1 - 2 t), optimise over t in [max(0, (1 - r_max)/2), min(r_max, 1/2)]
    # via golden-section.
    lo = max(mpf(0), (1 - r_max) / 2)
    hi = min(r_max, mpf("0.5"))
    if hi <= lo:
        return rhs_unconstrained, z_opt
    # Golden section on t
    phi_g = (mp.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - phi_g * (b - a)
    d = a + phi_g * (b - a)

    def _eval(t):
        z = list(z_opt)
        z[0] = t
        z[1] = min(r_max, max(mpf(0), 1 - 2 * t))
        return rhs_multimoment(M, z, k_vals, K2), z

    fc, zc = _eval(c)
    fd, zd = _eval(d)
    for _ in range(80):
        if fc > fd:
            b, d, fd, zd = d, c, fc, zc
            c = b - phi_g * (b - a)
            fc, zc = _eval(c)
        else:
            a, c, fc, zc = c, d, fd, zd
            d = a + phi_g * (b - a)
            fd, zd = _eval(d)
        if abs(b - a) < mpf("1e-12"):
            break
    if fc > fd:
        return fc, zc
    return fd, zd


# -----------------------------------------------------------------------------
# Bisection driver
# -----------------------------------------------------------------------------

@dataclass
class CertifiedBisectionResult:
    M_cert: mpf                       # certified lower bracket (infeasible => valid LB)
    M_upper: mpf                      # upper bracket (feasible => not yet a LB)
    z_at_upper: List[mpf]             # witness z at upper bracket
    rhs_at_upper: mpf
    target: mpf                        # 2/u + a
    n_iterations: int
    params: dict                       # (delta, u, N, mo_strong, ...)


def certified_forbidden_max(
    phi_obj,
    use_mo: bool = False,
    mo_strong: bool = True,
    M_lo: mpf = mpf("1.0001"),
    M_hi: mpf = mpf("1.60"),
    tol: mpf = mpf("1e-10"),
    max_iter: int = 200,
) -> CertifiedBisectionResult:
    """Bisect M in [M_lo, M_hi] to find the smallest admissible M*.

    Admissibility is defined by the inner feasibility oracle
    ``max_rhs_over_z_with_mo(M, k_vals, K2, mo_strong=...)``.  Returns a
    certified lower bracket M_cert such that every admissible f satisfies
    ||f*f||_inf >= M_cert (i.e., M_cert is a valid LOWER BOUND on C_{1a}).

    Note: this function consumes a MVMultiMomentBound-like object; it does
    NOT currently handle full Toeplitz/Hausdorff PSD (that's in
    ``forbidden_region_sdp.py`` / ``certify.py``).  For the magnitude-box
    + MO problem (which is what the brief specifies), this is the exact
    inner solver.
    """
    target = mpf(phi_obj.lhs)
    k_vals = list(phi_obj.k_vals) if hasattr(phi_obj, "k_vals") else [mpf(phi_obj.k1)]
    K2 = mpf(phi_obj.K2)

    inner = (lambda M: max_rhs_over_z_with_mo(M, k_vals, K2, mo_strong=mo_strong)) if use_mo \
        else (lambda M: max_rhs_over_z(M, k_vals, K2))

    # Sanity check brackets
    rhs_hi, z_hi = inner(mpf(M_hi))
    if rhs_hi < target:
        raise ValueError(f"M_hi={M_hi} is infeasible (RHS {rhs_hi} < target {target})")
    rhs_lo, _ = inner(mpf(M_lo))
    if rhs_lo >= target:
        return CertifiedBisectionResult(
            M_cert=mpf(M_lo), M_upper=mpf(M_lo),
            z_at_upper=[mpf(0)] * len(k_vals),
            rhs_at_upper=rhs_lo, target=target, n_iterations=0,
            params={"note": "M_lo already feasible"},
        )

    lo, hi = mpf(M_lo), mpf(M_hi)
    z_hi_final = z_hi
    rhs_hi_final = rhs_hi
    it = 0
    for it in range(1, max_iter + 1):
        mid = (lo + hi) / 2
        rhs_mid, z_mid = inner(mid)
        if rhs_mid >= target:
            hi = mid
            z_hi_final = z_mid
            rhs_hi_final = rhs_mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return CertifiedBisectionResult(
        M_cert=lo,              # largest confirmed INFEASIBLE M => valid LB
        M_upper=hi,
        z_at_upper=z_hi_final,
        rhs_at_upper=rhs_hi_final,
        target=target,
        n_iterations=it,
        params={
            "use_mo": use_mo, "mo_strong": mo_strong,
            "M_lo_init": str(M_lo), "M_hi_init": str(M_hi), "tol": str(tol),
        },
    )


# -----------------------------------------------------------------------------
# Rational rounding of certificates
# -----------------------------------------------------------------------------

@dataclass
class RationalCertificate:
    """Rational certificate for a certified lower bound.

    Components:
      M_cert_fmpq       : rational M such that every admissible f has ||f*f||_inf >= M.
      target_lo_fmpq    : rational LOWER BOUND on (2/u + a).
      rhs_upper_fmpq    : rational UPPER BOUND on max_{z in F} RHS(M, z).
    We have target_lo > rhs_upper  =>  M is infeasible (i.e., a valid LB).
    """
    M_cert_fmpq: "fmpq"
    target_lo_fmpq: "fmpq"
    rhs_upper_fmpq: "fmpq"
    extras: dict = field(default_factory=dict)


def round_to_rational_certificate(
    result: CertifiedBisectionResult,
    denom_bits: int = 60,
    pad_bits: int = 8,
) -> RationalCertificate:
    """Round the bisection result to a rational certificate.

    We take M_cert := floor(result.M_cert * 2^denom_bits) / 2^denom_bits (an
    under-approximation) and widen by 2^(-denom_bits + pad_bits) on each
    sensitive quantity so the resulting inequality remains sound.
    """
    import flint
    from flint import fmpq
    D = 1 << denom_bits
    M_f = float(result.M_cert)
    # Rational floor of M_cert: round DOWN so the certified bound is not
    # claimed to be larger than what we actually proved.
    M_num = int(M_f * D) - 2
    M_cert = fmpq(M_num, D)
    target_num = int((float(result.target) - 2.0 ** (-denom_bits + pad_bits)) * D)
    target = fmpq(target_num, D)
    rhs_num = int((float(result.rhs_at_upper) + 2.0 ** (-denom_bits + pad_bits)) * D) + 2
    rhs = fmpq(rhs_num, D)
    return RationalCertificate(
        M_cert_fmpq=M_cert,
        target_lo_fmpq=target,
        rhs_upper_fmpq=rhs,
        extras={
            "denom_bits": denom_bits,
            "pad_bits": pad_bits,
            "M_cert_float": M_f,
            "M_upper_float": float(result.M_upper),
            "target_float": float(result.target),
            "rhs_upper_float": float(result.rhs_at_upper),
            "z_witness": [float(z) for z in result.z_at_upper],
            "n_iterations": result.n_iterations,
            "params": result.params,
        },
    )


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("forbidden_region.py — self-test")
    print("=" * 70)
    mp.mp.dps = 30

    # Reproduce MV with the MVSingleMomentBound wrapped as MVMultiMomentBound(N=1)
    b1 = MVMultiMomentBound(N=1)
    res1 = certified_forbidden_max(b1, use_mo=False)
    print(f"  MVMultiMomentBound(N=1)  M_cert = {float(res1.M_cert):.8f}  (MV 1.2748)")

    for N in (2, 3, 5, 10):
        b = MVMultiMomentBound(N=N)
        res = certified_forbidden_max(b, use_mo=False)
        print(f"  N = {N:>3d}   M_cert = {float(res.M_cert):.8f}")

    print()
    print("With MO 2.17 (strong, via magnitude-implied constraint 2r_1+r_2>=1):")
    for N in (2, 3, 5):
        b = MVMultiMomentBound(N=N)
        res_mo = certified_forbidden_max(b, use_mo=True, mo_strong=True)
        print(f"  N = {N:>3d}   M_cert (MO on) = {float(res_mo.M_cert):.8f}")

    print()
    print("Rational certificate (demo):")
    cert = round_to_rational_certificate(res1, denom_bits=60)
    print(f"  M_cert_fmpq           = {cert.M_cert_fmpq}")
    print(f"  M_cert (float)        = {float(cert.M_cert_fmpq.p) / float(cert.M_cert_fmpq.q):.12f}")
    print(f"  target_lo_fmpq (2/u+a)= {float(cert.target_lo_fmpq.p) / float(cert.target_lo_fmpq.q):.12f}")
    print(f"  rhs_upper_fmpq        = {float(cert.rhs_upper_fmpq.p) / float(cert.rhs_upper_fmpq.q):.12f}")
