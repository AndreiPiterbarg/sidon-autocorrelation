"""Quick correctness checks for cs_refined_lp.

Confirms:
- CS test value matches existing repo's `compute_test_value_single`.
- Heuristic upper bound at small n matches rigorous b_{n,m}.

Run via:
    pytest tests/test_cs_refined_lp.py -q
"""
import os
import sys

import numpy as np

CS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "cloninger-steinerberger")
sys.path.insert(0, CS_DIR)


def test_cs_test_value_matches_existing():
    from cs_refined_lp import cs_test_value
    from test_values import compute_test_value_single

    rng = np.random.default_rng(0)
    for n_half in [2, 3, 4]:
        a = rng.dirichlet(np.ones(2 * n_half)) * (4 * n_half)
        v_new = cs_test_value(a)
        v_old = compute_test_value_single(a, n_half=n_half)
        assert abs(v_new - v_old) < 1e-9, (n_half, v_new, v_old)


def test_heuristic_close_to_rigorous_at_small_n():
    """At n=2, heuristic with enough restarts must match rigorous bound."""
    from cs_refined_lp import heuristic_an_star

    obj, _ = heuristic_an_star(2, n_restarts=32, n_iters=1500, seed=0)
    # rigorous: b_{2, large m} -> 1.103 (val(4))
    assert obj < 1.110, f"heuristic at n=2 too high: {obj}"
    assert obj > 1.099, f"heuristic at n=2 unexpectedly below val(4): {obj}"


def test_simplex_projection_preserves_mass_and_nonneg():
    from cs_refined_lp import project_simplex

    rng = np.random.default_rng(0)
    for _ in range(20):
        v = rng.normal(size=8)
        proj = project_simplex(v, total=4.0)
        assert proj.min() >= -1e-12
        assert abs(proj.sum() - 4.0) < 1e-9


if __name__ == "__main__":
    test_cs_test_value_matches_existing()
    print("OK: cs_test_value matches existing repo")
    test_heuristic_close_to_rigorous_at_small_n()
    print("OK: heuristic at n=2 matches rigorous within 0.01")
    test_simplex_projection_preserves_mass_and_nonneg()
    print("OK: simplex projection")
