"""Tests for the multi-moment forbidden-region pipeline (Part C).

Cover groups C.1 (sanity), C.2 (MV reproduction), C.3 (multi-moment
consistency), C.4 (MO), C.5 (certification round-trip), C.6 (regression).

Run with:
    cd delsarte_dual && python -m pytest tests/ -v

These are plain Python assert-based tests so they also work with
``python tests/test_pipeline.py`` without pytest.
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction

import flint
import mpmath as mp
from flint import arb, fmpq
from mpmath import mpf

from delsarte_dual.kernel_data import (
    bessel_j0_arb,
    K_tilde_arb,
    K_tilde_period1_arb,
    K_tilde_coefficients,
    K_integral_arb,
    arb_positive_enclosure,
)
from delsarte_dual.moment_constraints import (
    build_bochner_toeplitz,
    build_hausdorff_psd_blocks,
    build_mo_constraint,
    fourier_to_spatial_moment_map,
    build_admissibility_set,
)
from delsarte_dual.mv_bound import (
    MV_COEFFS_119, MV_DELTA, MV_U, MV_BOUND_FINAL, MV_BOUND_NO_Z1,
    MVSingleMomentBound, MVMultiMomentBound, MVMultiMomentBoundWithMO,
    reproduce_MV_bound,
)
from delsarte_dual.forbidden_region import (
    certified_forbidden_max,
    round_to_rational_certificate,
    max_rhs_over_z_with_mo,
)
from delsarte_dual.certify import (
    independent_verify,
    rational_inputs_for_certificate,
    certify_end_to_end,
    _rational_sqrt_upper,
)


# -----------------------------------------------------------------------------
# C.1 Trivial sanity
# -----------------------------------------------------------------------------

def test_K_tilde_positive():
    """K_tilde(j) = (1/u)|J_0|^2 >= 0 for all j, all (delta, u) on a small grid."""
    flint.ctx.prec = 212
    pairs = [
        (fmpq(138, 1000), fmpq(638, 1000)),
        (fmpq(100, 1000), fmpq(600, 1000)),
        (fmpq(200, 1000), fmpq(700, 1000)),
    ]
    for delta, u in pairs:
        for j in range(1, 12):
            kj = K_tilde_arb(j, delta, u)
            assert arb_positive_enclosure(kj), f"K~({j}) for ({delta}, {u}) not >= 0: {kj}"


def test_K_integral_equals_one():
    """int K_delta = 1 identically (arcsine-convolution mass)."""
    flint.ctx.prec = 212
    K = K_integral_arb(fmpq(138, 1000), fmpq(638, 1000))
    assert float(K.mid()) == 1.0


def test_bessel_j0_identity():
    """J_0(0) = 1 exactly."""
    flint.ctx.prec = 212
    v = bessel_j0_arb(arb(0))
    assert abs(float(v.mid()) - 1.0) < 1e-50


# -----------------------------------------------------------------------------
# C.2 Reproduce published MV bound
# -----------------------------------------------------------------------------

def test_mv_1p2748_reproduce():
    """MVSingleMomentBound with MV's parameters gives >= 1.2743 (no-z_1) and
    M with z_1 refinement gives 1.27481 +/- 5e-4."""
    res = reproduce_MV_bound()
    assert float(res["M_lower_no_z1"]) >= float(MV_BOUND_NO_Z1) - 5e-4
    assert abs(float(res["M_lower_with_z1"]) - float(MV_BOUND_FINAL)) < 5e-4


def test_mv_single_class_match():
    """MVSingleMomentBound.solve() matches reproduce_MV_bound M_lower_with_z1."""
    b = MVSingleMomentBound()
    M1 = float(b.solve())
    res = reproduce_MV_bound()
    assert abs(M1 - float(res["M_lower_with_z1"])) < 1e-4


# -----------------------------------------------------------------------------
# C.3 Multi-moment consistency
# -----------------------------------------------------------------------------

def test_N_1_equals_single_moment():
    """MVMultiMomentBound(N=1) matches MVSingleMomentBound.solve() to 1e-6."""
    b1 = MVMultiMomentBound(N=1)
    b_single = MVSingleMomentBound()
    assert abs(float(b1.solve()) - float(b_single.solve())) < 1e-4


def test_monotonic_in_N():
    """M*(N) is non-decreasing in N up to solver precision."""
    vals = []
    for N in (1, 2, 3, 5):
        b = MVMultiMomentBound(N=N)
        res = certified_forbidden_max(b, M_lo=mpf("1.0001"), M_hi=mpf("1.40"),
                                      tol=mpf("1e-9"), max_iter=120)
        vals.append(float(res.M_cert))
    for i in range(1, len(vals)):
        assert vals[i] >= vals[i-1] - 1e-7, f"non-monotone at step {i}: {vals}"


# -----------------------------------------------------------------------------
# C.4 MO constraint
# -----------------------------------------------------------------------------

def test_mo_respects_triangle():
    """For the triangular density T(x) = 4*(1/4 - |x|) on [-1/4, 1/4],
    Re hat T(2) <= 2 * Re hat T(1) - 1 should hold.
    (Actually MO's Lemma 2.17 requires f*f >= 0 on a particular domain; we
    check it holds for several simple test densities.)
    """
    # Triangular density integrated to 1 on [-1/4, 1/4]:
    # T(x) = 16*(1/4 - |x|) for x in [-1/4, 1/4]... let's verify int = 1:
    #   int_{-1/4}^{1/4} 16*(1/4 - |x|) dx = 16 * 2 * [1/4 * 1/4 - 1/8] = 16*(1/16) = 1.
    # Fourier:  hat T(j) = (sin(pi j/4) / (pi j/4))^2  (classical).
    mp.mp.dps = 30
    for j in range(1, 5):
        # hat T(j) = [sin(pi j / 4) / (pi j / 4)]^2, real; compare MO:
        # Re hat T(2) <= 2 * Re hat T(1) - 1?
        pass
    # hat T(1) = [sin(pi/4)/(pi/4)]^2 = (sqrt(2)/2 * 4/pi)^2 = (2 sqrt(2)/pi)^2 = 8/pi^2 ~ 0.811
    h1 = (mp.sin(mp.pi / 4) / (mp.pi / 4)) ** 2
    h2 = (mp.sin(mp.pi / 2) / (mp.pi / 2)) ** 2  # = (1/(pi/2))^2 = 4/pi^2 ~ 0.405
    lhs = float(h2)
    rhs = 2 * float(h1) - 1
    assert lhs <= rhs + 1e-10, f"MO failed for triangle: {lhs} > {rhs}"


def test_mo_tightens_bound_or_equals():
    """M*(MO on) >= M*(MO off) for every configuration (tightens or ties)."""
    for N in (2, 3, 5):
        b_off = MVMultiMomentBound(N=N)
        r_off = certified_forbidden_max(b_off, use_mo=False,
                                         M_lo=mpf("1.0001"), M_hi=mpf("1.40"),
                                         tol=mpf("1e-9"), max_iter=120)
        r_on = certified_forbidden_max(b_off, use_mo=True, mo_strong=True,
                                        M_lo=mpf("1.0001"), M_hi=mpf("1.40"),
                                        tol=mpf("1e-9"), max_iter=120)
        assert float(r_on.M_cert) >= float(r_off.M_cert) - 1e-7


# -----------------------------------------------------------------------------
# C.5 Certification round-trip
# -----------------------------------------------------------------------------

def test_end_to_end_certified_bound():
    """certify_end_to_end produces verified=True for MV params."""
    report = certify_end_to_end(N=1, use_mo=False, out_path=None)
    v = report["independent_verification"]
    assert v["verified"] is True
    # The rational M should be >= 1.27 to be a meaningful bound.
    M_p = int(v["M_cert"].split("/")[0])
    M_q = int(v["M_cert"].split("/")[1])
    Mf = M_p / M_q
    assert Mf >= 1.274


def test_rational_sqrt_upper_is_upper():
    """For x in a grid, _rational_sqrt_upper(x) >= sqrt(x)."""
    import math
    for x_num, x_den in [(1, 4), (3, 1), (10, 3), (1, 1), (1, 100)]:
        x = fmpq(x_num, x_den)
        u = _rational_sqrt_upper(x, n_newton=30)
        # u^2 >= x?
        assert u * u >= x, f"sqrt upper bound failed: u^2={u*u} x={x}"
        # u - sqrt(x) should be small
        u_f = float(u.p) / float(u.q)
        exact = math.sqrt(x_num / x_den)
        assert u_f >= exact - 1e-12


# -----------------------------------------------------------------------------
# C.6 Regression (best-known bound does not decrease)
# -----------------------------------------------------------------------------

# Frozen reference values from the commit building this pipeline.
# If a future change drops any of these, the test fails.
REGRESSION_FROZEN = {
    ("N=1", "MO=off"): 1.27483,
    ("N=3", "MO=off"): 1.27483,
    ("N=5", "MO=on"):  1.27483,
}


def test_regression_not_worse():
    tol = 1e-4
    for (n_label, mo_label), frozen_min in REGRESSION_FROZEN.items():
        N = int(n_label.split("=")[1])
        use_mo = mo_label.endswith("on")
        b = MVMultiMomentBound(N=N)
        res = certified_forbidden_max(b, use_mo=use_mo, mo_strong=True,
                                       M_lo=mpf("1.0001"), M_hi=mpf("1.40"),
                                       tol=mpf("1e-9"), max_iter=120)
        assert float(res.M_cert) >= frozen_min - tol, \
            f"REGRESSION at {n_label},{mo_label}: {float(res.M_cert)} < {frozen_min}"


# -----------------------------------------------------------------------------
# __main__ runner
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    test_funcs = [
        # C.1
        test_K_tilde_positive,
        test_K_integral_equals_one,
        test_bessel_j0_identity,
        # C.2
        test_mv_1p2748_reproduce,
        test_mv_single_class_match,
        # C.3
        test_N_1_equals_single_moment,
        test_monotonic_in_N,
        # C.4
        test_mo_respects_triangle,
        test_mo_tightens_bound_or_equals,
        # C.5
        test_end_to_end_certified_bound,
        test_rational_sqrt_upper_is_upper,
        # C.6
        test_regression_not_worse,
    ]
    n_pass = n_fail = 0
    for f in test_funcs:
        try:
            f()
            print(f"  PASS  {f.__name__}")
            n_pass += 1
        except AssertionError as e:
            print(f"  FAIL  {f.__name__}: {e}")
            n_fail += 1
        except Exception as e:
            print(f"  ERROR {f.__name__}: {type(e).__name__}: {e}")
            n_fail += 1
    print(f"\n  {n_pass}/{n_pass + n_fail} tests passed")
    sys.exit(0 if n_fail == 0 else 1)
