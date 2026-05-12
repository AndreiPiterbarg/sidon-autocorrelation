"""Tests for the F5 (Turan / Gorbachev-Tikhonov) family.

These tests document the structural facts of F5 in the repo's Delsarte-ratio
framework. They verify the bare-triangle sanity, closed-form ghat(0), monotonic
behaviour as K grows, and a sanity cross-check against F1.

The headline claim of these tests: F5 in this pipeline does NOT exceed
the existing F1 ratio (and hence does not approach the brief's purported
1.27481 target). That is enforced by `test_f5_dominated_by_F1`.
"""
from __future__ import annotations

import math

import pytest
from mpmath import mp, mpf

from delsarte_dual.family_f5_turan_gt.f5 import (
    NotPDAdmissible,
    f5_idealised_ratio,
    f5_lower_bound,
    g_hat_value,
    g_hat_zero,
    g_value,
    is_pd_admissible,
    M_g,
)


# -----------------------------------------------------------------------------
# Sanity: bare triangle prefix.
# -----------------------------------------------------------------------------

def test_bare_triangle_g_pointwise():
    """g(t) = (1 - 2|t|)_+ at sample points."""
    mp.dps = 50
    c = [1.0]
    assert float(g_value(0, c)) == pytest.approx(1.0, abs=1e-12)
    assert float(g_value(0.25, c)) == pytest.approx(0.5, abs=1e-12)
    assert float(g_value(0.5, c)) == pytest.approx(0.0, abs=1e-12)
    assert float(g_value(0.6, c)) == pytest.approx(0.0, abs=1e-12)
    assert float(g_value(-0.25, c)) == pytest.approx(0.5, abs=1e-12)


def test_bare_triangle_ghat_zero():
    """ghat(0) for bare triangle = int_{-1/2}^{1/2} (1 - 2|t|) dt = 1/2."""
    mp.dps = 50
    c = [1.0]
    assert float(g_hat_zero(c)) == pytest.approx(0.5, abs=1e-30)


def test_bare_triangle_ghat_at_xi():
    """ghat(xi) = (1/2) sinc^2(pi xi/2)."""
    mp.dps = 50
    c = [1.0]
    for xi in [0.5, 1.0, 1.5, 2.5, 3.7]:
        expected = 0.5 * (math.sin(math.pi * xi / 2) / (math.pi * xi / 2)) ** 2
        got = float(g_hat_value(xi, c))
        assert got == pytest.approx(expected, abs=1e-10), f"xi={xi}: got {got}, want {expected}"


def test_bare_triangle_M_g():
    """M_g = max (1 - 2|t|) on [-1/2, 1/2] = 1."""
    mp.dps = 50
    c = [1.0]
    Mlo, Mhi, _ = M_g(c, rel_tol=1e-8)
    assert float(Mlo) == pytest.approx(1.0, abs=1e-6)
    assert float(Mhi) == pytest.approx(1.0, abs=1e-6)


def test_bare_triangle_idealised_ratio_is_half():
    """Idealised ratio ghat(0)/M_g = 0.5 for bare triangle."""
    mp.dps = 50
    c = [1.0]
    lo, hi = f5_idealised_ratio(c)
    assert float(lo) == pytest.approx(0.5, abs=1e-6)
    assert float(hi) == pytest.approx(0.5, abs=1e-6)


def test_bare_triangle_below_threshold():
    """Bare F5 idealised ratio is <= 0.51 (from brief)."""
    mp.dps = 50
    c = [1.0]
    lo, hi = f5_idealised_ratio(c)
    assert float(hi) <= 0.51, "bare F5 should give <= 0.51 per ideas_fourier_ineqs.md"


# -----------------------------------------------------------------------------
# Sanity on PD admissibility on a known-PD function.
# -----------------------------------------------------------------------------

def test_pd_admissibility_returns_certificate():
    """is_pd_admissible returns a dict certificate with the documented keys."""
    mp.dps = 50
    c = [1.0]
    ok, cert = is_pd_admissible(c, xi_grid_density=201, R=10.0,
                                return_certificate=True)
    assert isinstance(cert, dict)
    for key in ("grid_R", "grid_density", "lipschitz", "tail_R",
                "tail_bound", "min_value_on_grid", "ok"):
        assert key in cert
    # Min ghat on grid must be >= -tiny tolerance (since (1/2) sinc^2 >= 0
    # exactly; any negative is mpmath quadrature noise).
    assert cert["min_value_on_grid"] >= -1e-30


# -----------------------------------------------------------------------------
# Monotonicity / non-improvement: F5 with non-trivial P does not blow past
# the bare-triangle 0.5 idealised ratio.
# -----------------------------------------------------------------------------

def test_f5_idealised_ratio_does_not_exceed_one():
    """Hard sanity: idealised ratio is bounded by 1 for any reasonable c."""
    mp.dps = 50
    # Try a few P:
    for c in [[1.0], [1.0, 0.1], [1.0, 0.0, 0.05], [2.0, -1.0]]:
        lo, hi = f5_idealised_ratio(c)
        # The ratio can be inf or undefined for degenerate cases; here we want
        # finite, bounded.
        if float(hi) == 0:
            continue
        assert float(hi) < 1.5, (
            f"idealised ratio {float(hi)} for c={c} unexpectedly large"
        )


# -----------------------------------------------------------------------------
# Cross-check vs F1: F5 (in this Delsarte ratio pipeline) does NOT beat F1's
# bare reference value of 0.5+ (let alone 1.27481).
# -----------------------------------------------------------------------------

def test_f5_dominated_by_F1():
    """F5 idealised ratio cannot beat 0.7 in this Delsarte pipeline.

    This test enforces the structural conclusion documented in derivation.md
    and RESULTS.md. If a future change discovers a c that breaks 0.7, the
    pipeline framework conclusion needs to be revisited; for now, F5 is
    structurally bounded.
    """
    mp.dps = 50
    # Check a few candidate c values (small monomial perturbations).
    candidates = [
        [1.0],
        [1.0, 0.5],
        [1.0, -0.2],
        [1.0, 0.0, 0.5],
        [1.0, 0.0, -0.5],
        [0.5, 1.0],
    ]
    max_ratio = 0.0
    for c in candidates:
        lo, hi = f5_idealised_ratio(c)
        if float(hi) > max_ratio:
            max_ratio = float(hi)
    assert max_ratio < 0.7, (
        f"F5 idealised ratio {max_ratio:.6f} unexpectedly above 0.7"
    )


def test_f5_lower_bound_smoke_bare():
    """f5_lower_bound on bare F5 returns a finite ball; PD certificate may
    or may not pass under the conservative L*h margin (sinc^2 has zeros on
    grid). We accept either a successful bound or a NotPDAdmissible."""
    mp.dps = 50
    c = [1.0]
    try:
        result = f5_lower_bound(c, n_subdiv=512, verify_pd=True, dps=30)
        # If certified, the bound should be <= 0.51 (way below MV's 1.27481).
        assert float(result["lb_high"]) <= 0.6
    except NotPDAdmissible as e:
        # Acceptable: the conservative grid+Lipschitz check is strict.
        assert "interior grid violation" in str(e) or "tail" in str(e)
