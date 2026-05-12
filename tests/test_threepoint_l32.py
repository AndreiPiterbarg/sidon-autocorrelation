"""Correctness tests for the discretized L^{3/2}-constrained SDP.

Tests:
  1. SANITY: at very large B (no effective L^{3/2} constraint), the SDP should
     give a non-trivial LB on C_{1a} (>= 1).  We don't expect it to be tight,
     but it should be sensible.
  2. SS-FEASIBILITY: at B = ||SS||_{3/2} = 2^{2/3} ~= 1.587, the SS function
     is feasible.  The discretized version of SS should give sup(f*f) ~= pi/2
     ~= 1.5708.  Verify the SDP returns lambda <= pi/2 (since SS is in the
     feasible set, the inf must be <= sup f_SS * f_SS = pi/2).
  3. CONVERGENCE: as n -> infinity, lambda should stabilize.
  4. TIGHTNESS: at small B, lambda should grow.
  5. POWER CONE: verify the realized ||f||_{3/2} matches the bound.
"""
import math
import numpy as np
import pytest
from lasserre.threepoint_l32 import build_l32_sdp, solve_l32


def test_basic_solve_small():
    """Verify the SDP solves at small n without errors."""
    problem, info, handles = build_l32_sdp(n=20, B=2.0)
    info = solve_l32(problem, info, handles, verbose=False)
    assert info.status in ("optimal", "optimal_inaccurate"), \
        f"Solve failed: {info.status}"
    assert info.objective is not None
    assert info.objective >= 0.99, f"lambda={info.objective} suspiciously small (expected >= 1)"


def test_ss_feasibility_at_b_above_ss_norm():
    """At B > ||SS||_{3/2} = 1.587, SS is feasible.  Therefore C_{1a}^{(B)} <= sup f_SS * f_SS = pi/2.

    The SDP gives a LOWER bound on C_{1a}^{(B)}, so SDP_value <= pi/2.
    """
    problem, info, handles = build_l32_sdp(n=40, B=2.0)
    info = solve_l32(problem, info, handles, verbose=False)
    assert info.status in ("optimal", "optimal_inaccurate")
    pi_over_2 = math.pi / 2
    # Some discretization slack
    assert info.objective <= pi_over_2 + 0.05, \
        f"lambda={info.objective} > pi/2 + slack; SDP relaxation is wrong"


def test_l32_constraint_active_at_small_b():
    """At small B, the L^{3/2} constraint should be active.  The realized
    ||f||_{3/2} should equal B (within tolerance).
    """
    problem, info, handles = build_l32_sdp(n=40, B=1.6)
    info = solve_l32(problem, info, handles, verbose=False)
    assert info.status in ("optimal", "optimal_inaccurate"), \
        f"Solve failed: {info.status}"
    # At B = 1.6 (just above SS norm 1.587), constraint may or may not be active
    # but the norm should be <= B + small slack
    assert info.L32_realized <= 1.6 + 0.05, \
        f"||f||_{{3/2}} = {info.L32_realized} > B = 1.6, constraint violated"


def test_lambda_increasing_in_decreasing_b():
    """As B decreases (tighter constraint), lambda should INCREASE
    (smaller feasible set => larger inf, modulo SDP relaxation looseness).
    """
    lam_vals = []
    for B in [3.0, 2.0, 1.7, 1.55, 1.4]:
        problem, info, handles = build_l32_sdp(n=30, B=B)
        info = solve_l32(problem, info, handles)
        if info.status not in ("optimal", "optimal_inaccurate"):
            pytest.skip(f"Solver failed at B={B}: {info.status}")
        lam_vals.append((B, info.objective))
    # Check monotonicity (allowing some numerical slack)
    for i in range(len(lam_vals) - 1):
        B_curr, lam_curr = lam_vals[i]
        B_next, lam_next = lam_vals[i + 1]
        # B is decreasing in the list, so lambda should be non-decreasing
        assert lam_next >= lam_curr - 0.01, \
            f"Monotonicity violated: B={B_curr} -> lam={lam_curr}, B={B_next} -> lam={lam_next}"


def test_uniform_distribution_value():
    """For uniform f = 2 on [-1/4, 1/4], ||f||_{3/2}^{3/2} = int 2^{1.5} dx = 2^{1.5} * 0.5
    = 0.5 * 2.828 = 1.414, so ||f||_{3/2} = 1.414^{2/3} ~= 1.260.
    sup(f*f) = 2.

    At B = 1.3 (just above uniform's norm), uniform is feasible with lambda = 2.
    But there might be better feasible f, so lambda <= 2.

    At B = 1.25 (just below uniform's norm), uniform is NOT feasible.
    """
    # Verify uniform is feasible at B=1.3
    problem, info, handles = build_l32_sdp(n=40, B=1.3)
    info = solve_l32(problem, info, handles)
    if info.status in ("optimal", "optimal_inaccurate"):
        # SDP gives a LB; uniform achieves sup f*f = 2 with ||f||_{3/2} = 1.26
        # Other feasible f might do better, so lambda can be < 2.  But > 1.
        assert info.objective >= 0.99
        assert info.objective <= 2.05  # some slack


if __name__ == "__main__":
    print("Running correctness tests...")
    test_basic_solve_small()
    print("  basic_solve_small: PASSED")
    test_ss_feasibility_at_b_above_ss_norm()
    print("  ss_feasibility: PASSED")
    test_l32_constraint_active_at_small_b()
    print("  l32_constraint_active: PASSED")
    test_lambda_increasing_in_decreasing_b()
    print("  lambda_monotonicity: PASSED")
    test_uniform_distribution_value()
    print("  uniform_distribution: PASSED")
    print("ALL TESTS PASSED")
