"""Dual certificate experiments for the discrete window problem.

This package builds lower-bound certificates for

    val(d) = min_{mu in Delta_d} max_W mu^T M_W mu

using dual objects that live on the window side.

Two families are currently implemented:

1. Constant window mixtures.
2. Polynomial nonnegative window multipliers with a simplex equality multiplier.

Reflection symmetry and reflected row-orbit compression are both available for
the multiplier LP, along with a resumable sweep runner for larger experiments.
"""

from simplex_window_dual.core import (
    exact_degree_monomials,
    up_to_degree_monomials,
    window_quadratic_coefficients,
)
from simplex_window_dual.degree1 import (
    Degree1CertificateProblem,
    Degree1CertificateResult,
    MultiplierCertificateProblem,
    MultiplierCertificateResult,
    build_degree1_problem,
    build_multiplier_problem,
    degree1_identity_coefficients,
    degree1_rhs_polynomial,
    evaluate_degree1_rhs_on_simplex,
    solve_multiplier_feasibility,
    solve_degree1_feasibility,
)
from simplex_window_dual.simplex_qp import solve_simplex_quadratic
from simplex_window_dual.search import search_multiplier_grid
from simplex_window_dual.run_sweep import run_dimension_sweep

__all__ = [
    "Degree1CertificateProblem",
    "Degree1CertificateResult",
    "MultiplierCertificateProblem",
    "MultiplierCertificateResult",
    "build_degree1_problem",
    "build_multiplier_problem",
    "degree1_identity_coefficients",
    "degree1_rhs_polynomial",
    "exact_degree_monomials",
    "evaluate_degree1_rhs_on_simplex",
    "run_dimension_sweep",
    "search_multiplier_grid",
    "solve_degree1_feasibility",
    "solve_multiplier_feasibility",
    "solve_simplex_quadratic",
    "up_to_degree_monomials",
    "window_quadratic_coefficients",
]
