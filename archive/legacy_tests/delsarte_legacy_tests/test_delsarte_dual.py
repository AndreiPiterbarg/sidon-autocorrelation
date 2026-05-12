"""Unit tests for delsarte_dual."""
from __future__ import annotations

import math

import pytest

from mpmath import iv, mp, mpf

from delsarte_dual import family_f1_selberg as f1
from delsarte_dual import family_f2_gauss_poly as f2
from delsarte_dual import rigorous_max as rm
from delsarte_dual import verify as ver


# -----------------------------------------------------------------------------
# F1 tests
# -----------------------------------------------------------------------------

def test_f1_ghat_nonneg_at_breakpoints():
    """Plain Fejer (no modulation) has ghat = triangle, nonneg."""
    p = f1.F1Params(T=1.0, omega=2.0, a=[])
    ok, _ = f1.positive_definite_certificate(p)
    assert ok


def test_f1_ghat_zero_matches_direct():
    """ghat(0) = T + sum a_k * (T - |k*omega|)_+ on the boundary."""
    p = f1.F1Params(T=1.0, omega=2.0, a=[0.3])
    # Since k*omega = 2 > T = 1, the triangle at shift 2 is zero at xi=0.
    v = float(f1.ghat_zero(p))
    assert abs(v - 1.0) < 1e-12


def test_f1_ghat_zero_with_overlap():
    """If k*omega < T, ghat(0) gets contribution from cosine."""
    p = f1.F1Params(T=2.0, omega=1.0, a=[0.5])
    # tri(0) = 2; tri(+/-1) = 1. Expected ghat(0) = 2 + 0.5 * (1 + 1) = 3.
    # But ghat_zero formula uses (1/2) * sum a_k * ... Actually re-read:
    # ghat(xi) = tri(xi) + (1/2) sum a_k (tri(xi-k*omega) + tri(xi+k*omega))
    # at xi=0: tri(0) + (1/2) a_1 * 2*tri(omega) = 2 + 0.5 * 2 * 1 = 3? No:
    # (1/2) * 0.5 * (tri(-1) + tri(1)) = 0.25 * 2 = 0.5. So ghat(0) = 2.5.
    # But ghat_zero() in the code does: T + sum a_k * tri(k*omega)
    # = 2 + 0.5 * tri(1) = 2 + 0.5 = 2.5. Consistent.
    v = float(f1.ghat_zero(p))
    assert abs(v - 2.5) < 1e-12


def test_f1_g_iv_matches_float_near_zero():
    """iv enclosure of g at t=0 should contain g(0) = T^2 (for a=[])."""
    p = f1.F1Params(T=1.5, omega=2.0, a=[])
    I = iv.mpf([mpf("-0.001"), mpf("0.001")])
    G = f1.g_iv(I, p)
    expected = 1.5 ** 2
    assert float(G.a) <= expected + 1e-6
    assert float(G.b) >= expected - 1e-6


def test_f1_g_iv_symbolic_agreement():
    """Sample a few points and compare iv enclosure with float64 eval."""
    p = f1.F1Params(T=1.0, omega=2.0, a=[-0.3])
    pi = math.pi

    def g_f64(t):
        T, omega = p.T, p.omega
        if abs(t) < 1e-12:
            fac1 = T ** 2
        else:
            fac1 = (math.sin(pi * T * t) / (pi * t)) ** 2
        fac2 = 1 + sum(ak * math.cos(2 * pi * k * omega * t)
                       for k, ak in enumerate(p.a, start=1))
        return fac1 * fac2

    for t in (0.1, 0.2, 0.37, -0.49):
        I = iv.mpf([mpf(t - 1e-8), mpf(t + 1e-8)])
        G = f1.g_iv(I, p)
        val = g_f64(t)
        assert float(G.a) - 1e-6 <= val <= float(G.b) + 1e-6, \
            f"t={t}: iv=[{float(G.a)},{float(G.b)}], float={val}"


def test_rigorous_max_conservative_plain_fejer():
    """For a=[], max g on [-1/2,1/2] = g(0) = T^2. Verified enclosure must
    contain T^2."""
    p = f1.F1Params(T=1.3, omega=2.0, a=[])
    lo, hi, n = rm.rigorous_max(
        lambda I: f1.g_iv(I, p),
        a=mpf("-0.5"), b=mpf("0.5"),
        rel_tol=1e-4, max_splits=500, precision_bits=100,
    )
    true_max = 1.3 ** 2
    assert float(lo) <= true_max + 1e-3
    assert float(hi) >= true_max - 1e-3


# -----------------------------------------------------------------------------
# F2 tests
# -----------------------------------------------------------------------------

def test_f2_plain_gaussian_ghat_zero():
    """For P=1, ghat(xi) = sqrt(pi/alpha) * exp(-pi^2 xi^2/alpha).
    ghat(0) = sqrt(pi/alpha)."""
    p = f2.F2Params(alpha=4.0, c=[1.0])
    v = float(f2.ghat_zero(p))
    assert abs(v - math.sqrt(math.pi / 4.0)) < 1e-12


def test_f2_plain_gaussian_pd():
    """A plain Gaussian is positive definite."""
    p = f2.F2Params(alpha=4.0, c=[1.0])
    ok, _ = f2.positive_definite_certificate(p)
    assert ok


def test_f2_g_iv_contains_value():
    """iv enclosure of g contains the float64 value."""
    p = f2.F2Params(alpha=3.0, c=[1.0, -0.5])
    for t in (0.0, 0.1, 0.3, -0.4):
        fv = math.exp(-3.0 * t ** 2) * (1.0 + (-0.5) * t ** 2)
        I = iv.mpf([mpf(t - 1e-8), mpf(t + 1e-8)])
        G = f2.g_iv(I, p)
        assert float(G.a) - 1e-6 <= fv <= float(G.b) + 1e-6


def test_f2_bad_pd_rejected():
    """c_0 < 0 makes Q(0) < 0, so not pd."""
    p = f2.F2Params(alpha=3.0, c=[-1.0])
    ok, _ = f2.positive_definite_certificate(p)
    assert not ok


# -----------------------------------------------------------------------------
# Signed weight L^sharp tests (rigor fix: old code used max(0, cos(pi|xi|)))
# -----------------------------------------------------------------------------

def test_weight_iv_positive_region():
    """weight_iv at xi = 0.3 should enclose cos(0.3 pi) ~ 0.588."""
    xi = iv.mpf([mpf("0.3"), mpf("0.3")])
    w = f1.weight_iv(xi)
    expected = math.cos(0.3 * math.pi)
    assert float(w.a) - 1e-12 <= expected <= float(w.b) + 1e-12
    assert expected > 0


def test_weight_iv_negative_region_between_half_and_one():
    """weight_iv at xi = 0.7 should enclose cos(0.7 pi) ~ -0.588 (NEGATIVE).
    This is the key rigor fix: the old weight clipped to 0 here."""
    xi = iv.mpf([mpf("0.7"), mpf("0.7")])
    w = f1.weight_iv(xi)
    expected = math.cos(0.7 * math.pi)
    assert expected < 0, "sanity: cos(0.7 pi) must be negative"
    assert float(w.a) - 1e-12 <= expected <= float(w.b) + 1e-12


def test_weight_iv_outside_unit_is_minus_one():
    """weight_iv at xi = 1.5 must enclose L^sharp = -1."""
    xi = iv.mpf([mpf("1.5"), mpf("1.5")])
    w = f1.weight_iv(xi)
    assert float(w.a) <= -1.0 + 1e-12
    assert float(w.b) >= -1.0 - 1e-12


def test_weight_iv_straddles_boundary_one():
    """An interval containing |xi| = 1 should enclose both pieces."""
    xi = iv.mpf([mpf("0.9"), mpf("1.1")])
    w = f1.weight_iv(xi)
    # Both endpoints give -1 (cos(pi) = -1) so the enclosure must include -1.
    assert float(w.a) <= -1.0 + 1e-9
    # Upper must be at least cos(0.9 pi) ~ -0.951
    assert float(w.b) >= math.cos(0.9 * math.pi) - 1e-9


def test_weight_iv_at_zero():
    """weight_iv at xi = 0 = cos(0) = 1."""
    xi = iv.mpf([mpf(0), mpf(0)])
    w = f1.weight_iv(xi)
    assert float(w.a) <= 1.0 + 1e-12
    assert float(w.b) >= 1.0 - 1e-12


def test_weight_iv_f2_delegates_to_f1():
    """F2's weight_iv delegates to F1."""
    for xi_val in ("0.3", "0.7", "1.5"):
        xi = iv.mpf([mpf(xi_val), mpf(xi_val)])
        w1 = f1.weight_iv(xi)
        w2 = f2.weight_iv(xi)
        assert float(w1.a) == float(w2.a)
        assert float(w1.b) == float(w2.b)


# -----------------------------------------------------------------------------
# End-to-end verify_f1 on plain Fejer (K=0): closed-form reference check.
#
# For g(t) = (sin(pi T t)/(pi t))^2 with T=1:
#   ghat(xi) = max(0, 1 - |xi|) on [-1, 1]
#   numerator = int_{-1}^{1} (1-|xi|) cos(pi|xi|) dxi = 4/pi^2 ~ 0.4053
#   (no |xi| > 1 contribution since ghat vanishes there)
#   int_{-1/2}^{1/2} (sin(pi t)/(pi t))^2 dt ~ 0.7737
#   ratio ~ 0.5238
# -----------------------------------------------------------------------------

def test_verify_f1_plain_fejer_reference():
    """verify_f1 with plain Fejer (K=0, T=1) must enclose the closed-form
    reference ratio ~0.5238."""
    p = f1.F1Params(T=1.0, omega=2.0, a=[])
    vb = ver.verify_f1(
        p, rel_tol=1e-6, n_subdiv=2048, precision_bits=120,
        denom_kind="int_g",
    )
    # PD trivially holds for plain Fejer; g >= 0 trivially (sinc^2).
    assert vb.pd_certified
    assert vb.g_nonneg_certified
    # Reference values (mpmath-computed):
    expected_num = 4.0 / math.pi ** 2           # ~ 0.4052847...
    expected_int_g = 0.77369500990281585        # int_{-1/2}^{1/2} sinc^2
    expected_ratio = expected_num / expected_int_g   # ~ 0.5238301...
    # Numerator enclosure contains 4/pi^2.
    assert float(vb.numerator_lo) <= expected_num + 1e-3
    assert float(vb.numerator_hi) >= expected_num - 1e-3
    # int_g enclosure contains the reference.
    assert float(vb.int_g_lo) <= expected_int_g + 1e-3
    assert float(vb.int_g_hi) >= expected_int_g - 1e-3
    # Ratio ball must contain the reference.
    assert float(vb.lb_low) <= expected_ratio + 1e-3
    assert float(vb.lb_high) >= expected_ratio - 1e-3
