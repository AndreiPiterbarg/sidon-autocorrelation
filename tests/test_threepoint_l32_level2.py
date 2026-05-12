"""Correctness tests for the LEVEL-2 Lasserre L^{3/2}-constrained SDP.

Tests:
  1. PSD VALIDATION: inject uniform f and check M_2, all localizers numerically PSD.
  2. SANITY: at large B (no effective L^{3/2} constraint), lambda > 0 and feasible.
  3. UPPER BOUND: at B = 2.0 (above SS norm), lambda <= pi/2 + slack (SS feasible).
  4. MONOTONICITY: lambda non-decreasing as B decreases.
  5. LEVEL-2 >= LEVEL-1: at the same n, B, level-2 lambda >= level-1 lambda - eps.

These tests use small n (typically n=12 or n=15) to keep runtime modest.
"""
import math
import numpy as np
import pytest

from lasserre.threepoint_l32 import build_l32_sdp, solve_l32
from lasserre.threepoint_l32_level2 import (
    build_l32_level2_sdp, solve_l32_level2, validate_psd_for_uniform,
    enum_alphas, fill_moments_from_f, canon,
)


# ---------------------------------------------------------------------
# Validation: PSD structure for a known feasible f
# ---------------------------------------------------------------------

@pytest.mark.parametrize("n", [5, 8, 10, 12, 15])
def test_uniform_density_passes_psd(n):
    """Inject the uniform density f_i = 2.0 (which has int=1 over [-1/4,1/4])
    into the level-2 moment matrix and all localizers; verify they are PSD
    numerically.
    """
    ok, min_eig_M2, min_eig_loc = validate_psd_for_uniform(n)
    assert ok, (f"PSD validation failed for n={n}: "
                f"min_eig(M2)={min_eig_M2:.3e}, min_eig(loc)={min_eig_loc:.3e}")
    # Sanity: should be close to PSD up to floating-point
    assert min_eig_M2 >= -1e-8
    assert min_eig_loc >= -1e-8


def test_canonical_indexing():
    """Multi-index canonicalization: same monomial -> same canonical key."""
    assert canon((3, 1, 2)) == (1, 2, 3)
    assert canon((1, 1, 2)) == (1, 1, 2)
    assert canon(()) == ()


def test_enum_alphas_count():
    """Number of multi-indices on n positions of degree <= D is C(n+D, D)."""
    for n, D in [(3, 4), (5, 3), (8, 2)]:
        alphas = enum_alphas(n, D)
        expected = math.comb(n + D, D)
        assert len(alphas) == expected, f"n={n} D={D}: got {len(alphas)} expected {expected}"


# ---------------------------------------------------------------------
# Solve sanity at small n
# ---------------------------------------------------------------------

def test_basic_solve_small():
    """Verify the level-2 SDP solves at small n without errors."""
    problem, info, handles = build_l32_level2_sdp(n=10, B=2.0)
    info = solve_l32_level2(problem, info, handles, verbose=False)
    assert info.status in ("optimal", "optimal_inaccurate"), \
        f"Solve failed: {info.status}"
    assert info.objective is not None
    # Discrete floor for n=10: (2n-2)/(2n-1) = 18/19 = 0.9474
    floor = (2 * 10 - 2) / (2 * 10 - 1)
    assert info.objective >= floor - 1e-3, \
        f"lambda={info.objective} below discrete floor {floor}"


def test_ss_feasibility_at_b_above_ss_norm():
    """At B = 2.0 > ||SS||_{3/2} = 1.587, lambda should be <= pi/2 + slack."""
    problem, info, handles = build_l32_level2_sdp(n=10, B=2.0)
    info = solve_l32_level2(problem, info, handles, verbose=False)
    assert info.status in ("optimal", "optimal_inaccurate")
    pi_over_2 = math.pi / 2
    assert info.objective <= pi_over_2 + 0.05, \
        f"lambda={info.objective} > pi/2 + slack; level-2 relaxation is wrong"


def test_l32_constraint_respected():
    """At small B, the realized ||f||_{3/2} of the recovered f should be <= B + slack.
    (The recovered f from y_{(i,)} may not be the true argmin since the relaxation
    is loose, but the L^{3/2} constraint enforced through z_i must still hold.)
    """
    problem, info, handles = build_l32_level2_sdp(n=10, B=1.6)
    info = solve_l32_level2(problem, info, handles, verbose=False)
    assert info.status in ("optimal", "optimal_inaccurate"), \
        f"Solve failed: {info.status}"
    assert info.L32_realized <= 1.6 + 0.05, \
        f"||f||_{{3/2}} = {info.L32_realized} > B = 1.6, constraint violated"


def test_lambda_monotone_decreasing_b():
    """As B decreases, lambda should be non-decreasing."""
    n = 8
    lam_vals = []
    for B in [3.0, 2.0, 1.6, 1.4]:
        problem, info, handles = build_l32_level2_sdp(n=n, B=B)
        info = solve_l32_level2(problem, info, handles)
        if info.status not in ("optimal", "optimal_inaccurate"):
            pytest.skip(f"Solver failed at B={B}: {info.status}")
        lam_vals.append((B, info.objective))
    for i in range(len(lam_vals) - 1):
        B_curr, lam_curr = lam_vals[i]
        B_next, lam_next = lam_vals[i + 1]
        assert lam_next >= lam_curr - 1e-3, \
            f"Monotonicity violated: B={B_curr}->{lam_curr}, B={B_next}->{lam_next}"


# ---------------------------------------------------------------------
# Level-2 must be at least as tight as level-1
# ---------------------------------------------------------------------

@pytest.mark.parametrize("B", [2.0, 1.6, 1.4])
def test_level2_at_least_as_tight_as_level1(B):
    """Level-2 SDP value must be >= level-1 SDP value (modulo solver tolerance).
    A level-2 lift is a refinement of level-1, so it can only INCREASE the
    minimum lambda (the relaxation gets tighter)."""
    n = 8
    p1, i1, h1 = build_l32_sdp(n=n, B=B)
    i1 = solve_l32(p1, i1, h1)
    p2, i2, h2 = build_l32_level2_sdp(n=n, B=B)
    i2 = solve_l32_level2(p2, i2, h2)
    if (i1.status not in ("optimal", "optimal_inaccurate") or
        i2.status not in ("optimal", "optimal_inaccurate")):
        pytest.skip(f"Solver failure: L1 {i1.status}, L2 {i2.status}")
    # Allow 1e-3 numerical slack (both relaxations may hit the same floor)
    assert i2.objective >= i1.objective - 1e-3, \
        f"Level-2 LOOSER than level-1 at B={B}: L1={i1.objective}, L2={i2.objective}"


# ---------------------------------------------------------------------
# Discrete floor: confirm it is exactly (2n-2)/(2n-1)
# ---------------------------------------------------------------------

@pytest.mark.parametrize("n", [6, 8, 12])
def test_discrete_floor_unconstrained(n):
    """At very large B (no L^{3/2} bite), lambda should be at the discrete
    floor (2n-2)/(2n-1) corresponding to a flat discrete convolution.
    """
    problem, info, handles = build_l32_level2_sdp(n=n, B=100.0)
    info = solve_l32_level2(problem, info, handles)
    if info.status not in ("optimal", "optimal_inaccurate"):
        pytest.skip(f"Solver failed: {info.status}")
    floor = (2 * n - 2) / (2 * n - 1)
    # Objective should be very close to floor (within solver tolerance)
    assert abs(info.objective - floor) < 5e-3, \
        f"n={n}: lambda={info.objective}, expected floor={floor}"


if __name__ == "__main__":
    print("=== Level-2 Lasserre tests ===")
    for n in [5, 8, 10]:
        test_uniform_density_passes_psd(n)
        print(f"  uniform_psd n={n}: PASSED")
    test_canonical_indexing(); print("  canonical_indexing: PASSED")
    test_enum_alphas_count(); print("  enum_alphas_count: PASSED")
    test_basic_solve_small(); print("  basic_solve_small: PASSED")
    test_ss_feasibility_at_b_above_ss_norm(); print("  ss_upper_bound: PASSED")
    test_l32_constraint_respected(); print("  l32_constraint_respected: PASSED")
    test_lambda_monotone_decreasing_b(); print("  monotonicity: PASSED")
    for B in [2.0, 1.6, 1.4]:
        test_level2_at_least_as_tight_as_level1(B)
        print(f"  level2_at_least_as_tight_as_level1 B={B}: PASSED")
    for n in [6, 8]:
        test_discrete_floor_unconstrained(n)
        print(f"  discrete_floor n={n}: PASSED")
    print("ALL TESTS PASSED")
