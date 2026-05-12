import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simplex_window_dual.degree1 import (
    build_degree1_problem,
    build_multiplier_problem,
    degree1_identity_coefficients,
    evaluate_degree1_rhs_on_simplex,
    solve_multiplier_feasibility,
    solve_degree1_feasibility,
)
from simplex_window_dual.run_sweep import run_dimension_sweep
from simplex_window_dual.search import search_multiplier_grid
from simplex_window_dual.simplex_qp import solve_simplex_quadratic


def _grid_simplex_points(d: int, steps: int) -> np.ndarray:
    if d == 1:
        return np.array([[1.0]])
    points = []

    def rec(prefix, remaining, slots):
        if slots == 1:
            points.append(prefix + [remaining])
            return
        for value in range(remaining + 1):
            rec(prefix + [value], remaining - value, slots - 1)

    rec([], steps, d)
    return np.array(points, dtype=np.float64) / steps


def test_solve_simplex_quadratic_matches_grid_search_on_small_case():
    q = np.array(
        [
            [1.20, -0.10, 0.25, 0.00],
            [-0.10, 1.05, 0.15, 0.30],
            [0.25, 0.15, 1.40, -0.20],
            [0.00, 0.30, -0.20, 1.10],
        ],
        dtype=np.float64,
    )
    q = 0.5 * (q + q.T)

    result = solve_simplex_quadratic(q)
    grid = _grid_simplex_points(d=4, steps=40)
    grid_values = np.einsum("bi,ij,bj->b", grid, q, grid)

    assert abs(result.minimizer.sum() - 1.0) < 1e-10
    assert np.all(result.minimizer >= -1e-10)
    assert result.value <= float(grid_values.min()) + 5e-3


def test_solve_simplex_quadratic_handles_vertex_optimum():
    q = np.array(
        [
            [0.75, 2.0, 2.0],
            [2.0, 1.10, 2.0],
            [2.0, 2.0, 1.30],
        ],
        dtype=np.float64,
    )
    result = solve_simplex_quadratic(q)

    assert abs(result.value - 0.75) < 1e-12
    assert np.allclose(result.minimizer, np.array([1.0, 0.0, 0.0]))


def test_degree1_certificate_d4_has_expected_small_gap():
    problem = build_degree1_problem(4)
    yes = solve_degree1_feasibility(problem, 1.01)
    no = solve_degree1_feasibility(problem, 1.05)

    assert yes.success
    assert not no.success


def test_degree1_certificate_identity_residual_is_small():
    problem = build_degree1_problem(4)
    result = solve_degree1_feasibility(problem, 1.01)
    assert result.success

    residuals = degree1_identity_coefficients(problem, result)
    worst = max(abs(v) for v in residuals.values())
    assert worst < 1e-8

    rng = np.random.default_rng(123)
    for _ in range(20):
        x = rng.dirichlet(np.ones(problem.d))
        rhs = evaluate_degree1_rhs_on_simplex(problem, result, x)
        assert rhs >= -1e-9


def test_degree2_certificate_strengthens_d4_threshold():
    degree1 = solve_degree1_feasibility(build_degree1_problem(4), 1.05)
    degree2_problem = build_multiplier_problem(4, multiplier_degree=2)
    degree2 = solve_multiplier_feasibility(degree2_problem, 1.05)

    assert not degree1.success
    assert degree2.success

    residuals = degree1_identity_coefficients(degree2_problem, degree2)
    worst = max(abs(v) for v in residuals.values())
    assert worst < 1e-8


def test_monotone_search_stops_after_first_infeasible():
    outcome = search_multiplier_grid(
        d=4,
        alphas=[1.00, 1.05, 1.06],
        multiplier_degree=1,
        stop_on_first_infeasible=True,
    )

    assert outcome.best_result is not None
    assert abs(outcome.best_result.alpha - 1.0) < 1e-12
    assert outcome.attempted_alphas == [1.0, 1.05]
    assert abs(outcome.first_infeasible_alpha - 1.05) < 1e-12


def test_reflection_symmetry_reduction_preserves_feasibility():
    plain_problem = build_multiplier_problem(4, multiplier_degree=2)
    symmetric_problem = build_multiplier_problem(
        4,
        multiplier_degree=2,
        use_reflection_symmetry=True,
    )

    assert symmetric_problem.n_vars < plain_problem.n_vars

    plain_yes = solve_multiplier_feasibility(plain_problem, 1.05)
    symmetric_yes = solve_multiplier_feasibility(symmetric_problem, 1.05)
    plain_no = solve_multiplier_feasibility(plain_problem, 1.07)
    symmetric_no = solve_multiplier_feasibility(symmetric_problem, 1.07)

    assert plain_yes.success
    assert symmetric_yes.success
    assert not plain_no.success
    assert not symmetric_no.success

    residuals = degree1_identity_coefficients(symmetric_problem, symmetric_yes)
    assert max(abs(v) for v in residuals.values()) < 1e-8


def test_reflection_row_orbits_preserve_feasibility_and_reduce_rows():
    symmetric_problem = build_multiplier_problem(
        4,
        multiplier_degree=2,
        use_reflection_symmetry=True,
    )
    row_orbit_problem = build_multiplier_problem(
        4,
        multiplier_degree=2,
        use_reflection_symmetry=True,
        use_reflection_row_orbits=True,
    )

    assert len(row_orbit_problem.b_eq) < len(symmetric_problem.b_eq)
    assert row_orbit_problem.n_vars == symmetric_problem.n_vars

    symmetric_yes = solve_multiplier_feasibility(symmetric_problem, 1.05)
    row_orbit_yes = solve_multiplier_feasibility(row_orbit_problem, 1.05)
    symmetric_no = solve_multiplier_feasibility(symmetric_problem, 1.07)
    row_orbit_no = solve_multiplier_feasibility(row_orbit_problem, 1.07)

    assert symmetric_yes.success
    assert row_orbit_yes.success
    assert not symmetric_no.success
    assert not row_orbit_no.success

    residuals = degree1_identity_coefficients(row_orbit_problem, row_orbit_yes)
    assert max(abs(v) for v in residuals.values()) < 1e-8


def test_run_dimension_sweep_writes_resumable_jsonl(tmp_path):
    out_path = tmp_path / "simplex_sweep.jsonl"
    summary = run_dimension_sweep(
        d=4,
        alphas=[1.00, 1.05, 1.06],
        multiplier_degree=1,
        use_reflection_symmetry=True,
        use_reflection_row_orbits=True,
        stop_on_first_infeasible=True,
        jsonl_out=str(out_path),
        resume=False,
    )

    assert summary.best_feasible_alpha == 1.0
    assert summary.first_infeasible_alpha == 1.05
    assert tuple(summary.attempted_alphas) == (1.0, 1.05)

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert '"kind": "attempt"' in lines[0]
    assert '"kind": "summary"' in lines[-1]

    resumed = run_dimension_sweep(
        d=4,
        alphas=[1.00, 1.05, 1.06],
        multiplier_degree=1,
        use_reflection_symmetry=True,
        use_reflection_row_orbits=True,
        stop_on_first_infeasible=True,
        jsonl_out=str(out_path),
        resume=True,
    )
    assert resumed.best_feasible_alpha == summary.best_feasible_alpha
    assert resumed.first_infeasible_alpha == summary.first_infeasible_alpha
