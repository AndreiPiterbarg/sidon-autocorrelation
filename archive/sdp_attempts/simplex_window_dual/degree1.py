"""Polynomial window-multiplier LP certificates on the simplex."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
from scipy import sparse as sp
from scipy.optimize import linprog

from simplex_window_dual.core import (
    Monomial,
    evaluate_polynomial,
    up_to_degree_monomials,
    window_quadratic_coefficients,
)


@dataclass(frozen=True)
class MultiplierCertificateProblem:
    d: int
    multiplier_degree: int
    use_reflection_symmetry: bool
    use_reflection_row_orbits: bool
    windows: Tuple[Tuple[int, int], ...]
    multiplier_monomials: Tuple[Monomial, ...]
    slack_monomials: Tuple[Monomial, ...]
    equality_monomials: Tuple[Monomial, ...]
    slack_full_to_reduced: Tuple[int, ...]
    equality_full_to_reduced: Tuple[int, ...]
    coefficient_row_full_to_reduced: Tuple[int, ...]
    coefficient_row_orbit_sizes: Tuple[int, ...]
    multiplier_slice: slice
    slack_slice: slice
    equality_slice: slice
    n_vars: int
    a_eq_base: sp.csc_matrix
    a_eq_alpha: sp.csc_matrix
    b_eq: np.ndarray
    bounds: Tuple[Tuple[float | None, float | None], ...]


@dataclass(frozen=True)
class MultiplierCertificateResult:
    success: bool
    alpha: float
    status: int
    message: str
    x: np.ndarray | None
    solver_method: str | None = None

    @property
    def constant_window_weights(self) -> np.ndarray | None:
        return None if self.x is None else self.x


def _reflect_monomial(mono: Monomial) -> Monomial:
    return tuple(reversed(mono))


def _reflect_window(window: Tuple[int, int], d: int) -> Tuple[int, int]:
    ell, s_lo = window
    return (ell, 2 * d - ell - s_lo)


def _build_orbit_map(reflected_indices: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    rep_to_reduced: Dict[int, int] = {}
    full_to_reduced = []
    for full_idx, reflected_idx in enumerate(reflected_indices):
        rep = min(full_idx, reflected_idx)
        reduced_idx = rep_to_reduced.get(rep)
        if reduced_idx is None:
            reduced_idx = len(rep_to_reduced)
            rep_to_reduced[rep] = reduced_idx
        full_to_reduced.append(reduced_idx)
    return tuple(full_to_reduced), len(rep_to_reduced)

@lru_cache(maxsize=None)
def build_multiplier_problem(
    d: int,
    multiplier_degree: int,
    use_reflection_symmetry: bool = False,
    use_reflection_row_orbits: bool = False,
) -> MultiplierCertificateProblem:
    """Build the fixed-structure LP for polynomial dual certificates.

    The feasibility problem at target alpha is

        sum_W Lambda_W(x) (f_W(x) - alpha)
          = N(x) + (1 - sum_i x_i) H(x)

    with:
      - each Lambda_W having nonnegative coefficients, degree <= multiplier_degree
      - N having nonnegative coefficients, degree <= multiplier_degree + 2
      - H free, degree <= multiplier_degree + 1
      - sum_W coeff(Lambda_W, 1) = 1 normalization

    For x in the simplex, the RHS reduces to N(x) >= 0. Therefore if the LP is
    feasible, max_W f_W(x) >= alpha for every simplex point x.
    """
    if use_reflection_row_orbits and not use_reflection_symmetry:
        raise ValueError(
            "reflection row orbits require use_reflection_symmetry=True"
        )

    windows, f_coeffs = window_quadratic_coefficients(d)
    multiplier_monomials = up_to_degree_monomials(d, multiplier_degree)
    slack_monomials = up_to_degree_monomials(d, multiplier_degree + 2)
    equality_monomials = up_to_degree_monomials(d, multiplier_degree + 1)
    idx_multiplier = {mono: idx for idx, mono in enumerate(multiplier_monomials)}
    idx_slack = {mono: idx for idx, mono in enumerate(slack_monomials)}
    idx_equality = {mono: idx for idx, mono in enumerate(equality_monomials)}

    n_windows = len(windows)
    if use_reflection_symmetry:
        window_to_idx = {window: idx for idx, window in enumerate(windows)}
        reflected_windows = tuple(
            window_to_idx[_reflect_window(window, d)] for window in windows
        )
        reflected_multiplier = tuple(
            idx_multiplier[_reflect_monomial(mono)] for mono in multiplier_monomials
        )
        reflected_slack = tuple(
            idx_slack[_reflect_monomial(mono)] for mono in slack_monomials
        )
        reflected_equality = tuple(
            idx_equality[_reflect_monomial(mono)] for mono in equality_monomials
        )

        multiplier_reflected_indices = []
        for w_idx in range(n_windows):
            for gamma_idx in range(len(multiplier_monomials)):
                multiplier_reflected_indices.append(
                    reflected_windows[w_idx] * len(multiplier_monomials)
                    + reflected_multiplier[gamma_idx]
                )
        multiplier_full_to_reduced, multiplier_var_count = _build_orbit_map(
            tuple(multiplier_reflected_indices)
        )
        slack_full_to_reduced, slack_var_count = _build_orbit_map(reflected_slack)
        equality_full_to_reduced, equality_var_count = _build_orbit_map(
            reflected_equality
        )
        if use_reflection_row_orbits:
            coefficient_row_full_to_reduced, coefficient_row_count = _build_orbit_map(
                reflected_slack
            )
        else:
            coefficient_row_full_to_reduced = tuple(range(len(slack_monomials)))
            coefficient_row_count = len(slack_monomials)
    else:
        multiplier_var_count = n_windows * len(multiplier_monomials)
        multiplier_full_to_reduced = tuple(range(multiplier_var_count))
        slack_full_to_reduced = tuple(range(len(slack_monomials)))
        slack_var_count = len(slack_monomials)
        equality_full_to_reduced = tuple(range(len(equality_monomials)))
        equality_var_count = len(equality_monomials)
        coefficient_row_full_to_reduced = tuple(range(len(slack_monomials)))
        coefficient_row_count = len(slack_monomials)

    row_orbit_sizes = [0] * coefficient_row_count
    for reduced_idx in coefficient_row_full_to_reduced:
        row_orbit_sizes[reduced_idx] += 1

    k = 0
    multiplier_slice = slice(k, k + multiplier_var_count)
    k += multiplier_var_count
    slack_slice = slice(k, k + slack_var_count)
    k += slack_var_count
    equality_slice = slice(k, k + equality_var_count)
    k += equality_var_count
    n_vars = k

    rows_base = []
    cols_base = []
    vals_base = []
    rows_alpha = []
    cols_alpha = []
    vals_alpha = []
    b_eq = np.zeros(coefficient_row_count + 1, dtype=np.float64)

    zero = tuple(0 for _ in range(d))
    multiplier_zero_idx = multiplier_monomials.index(zero)

    for w_idx, coeff in enumerate(f_coeffs):
        coeff_items = tuple(coeff.items())
        for gamma_idx, gamma in enumerate(multiplier_monomials):
            full_idx = w_idx * len(multiplier_monomials) + gamma_idx
            col_idx = multiplier_slice.start + multiplier_full_to_reduced[full_idx]
            row_idx = coefficient_row_full_to_reduced[idx_slack[gamma]]

            rows_alpha.append(row_idx)
            cols_alpha.append(col_idx)
            vals_alpha.append(-1.0)

            for beta, value in coeff_items:
                mono = tuple(g + b for g, b in zip(gamma, beta))
                rows_base.append(coefficient_row_full_to_reduced[idx_slack[mono]])
                cols_base.append(col_idx)
                vals_base.append(value)

    for row_idx, mono in enumerate(slack_monomials):
        # -N(x)
        rows_base.append(coefficient_row_full_to_reduced[row_idx])
        cols_base.append(slack_slice.start + slack_full_to_reduced[idx_slack[mono]])
        vals_base.append(-1.0)

    # -(1 - sum x_i) H(x) = -H(x) + sum_i x_i H(x)
    for eq_idx, mono in enumerate(equality_monomials):
        col_idx = equality_slice.start + equality_full_to_reduced[eq_idx]
        rows_base.append(coefficient_row_full_to_reduced[idx_slack[mono]])
        cols_base.append(col_idx)
        vals_base.append(-1.0)

        for i in range(d):
            lifted = list(mono)
            lifted[i] += 1
            lifted_mono = tuple(lifted)
            rows_base.append(coefficient_row_full_to_reduced[idx_slack[lifted_mono]])
            cols_base.append(col_idx)
            vals_base.append(1.0)

    # Normalization: sum_W coeff(Lambda_W, 1) = 1
    norm_row = coefficient_row_count
    for w_idx in range(n_windows):
        rows_base.append(norm_row)
        full_idx = w_idx * len(multiplier_monomials) + multiplier_zero_idx
        cols_base.append(multiplier_slice.start + multiplier_full_to_reduced[full_idx])
        vals_base.append(1.0)
    b_eq[norm_row] = 1.0

    a_eq_base = sp.csc_matrix(
        (vals_base, (rows_base, cols_base)),
        shape=(len(b_eq), n_vars),
        dtype=np.float64,
    )
    a_eq_alpha = sp.csc_matrix(
        (vals_alpha, (rows_alpha, cols_alpha)),
        shape=(len(b_eq), n_vars),
        dtype=np.float64,
    )

    bounds = (
        ((0.0, None),) * multiplier_var_count
        + ((0.0, None),) * slack_var_count
        + ((None, None),) * equality_var_count
    )

    return MultiplierCertificateProblem(
        d=d,
        multiplier_degree=multiplier_degree,
        use_reflection_symmetry=use_reflection_symmetry,
        use_reflection_row_orbits=use_reflection_row_orbits,
        windows=windows,
        multiplier_monomials=multiplier_monomials,
        slack_monomials=slack_monomials,
        equality_monomials=equality_monomials,
        slack_full_to_reduced=slack_full_to_reduced,
        equality_full_to_reduced=equality_full_to_reduced,
        coefficient_row_full_to_reduced=coefficient_row_full_to_reduced,
        coefficient_row_orbit_sizes=tuple(row_orbit_sizes),
        multiplier_slice=multiplier_slice,
        slack_slice=slack_slice,
        equality_slice=equality_slice,
        n_vars=n_vars,
        a_eq_base=a_eq_base,
        a_eq_alpha=a_eq_alpha,
        b_eq=b_eq,
        bounds=bounds,
    )


def solve_multiplier_feasibility(
    problem: MultiplierCertificateProblem,
    alpha: float,
) -> MultiplierCertificateResult:
    """Solve the polynomial-multiplier LP feasibility problem at fixed alpha."""
    a_eq = problem.a_eq_base + alpha * problem.a_eq_alpha
    objective = np.zeros(problem.n_vars, dtype=np.float64)
    methods = ("highs", "highs-ds", "highs-ipm")
    first_failure = None

    for method in methods:
        res = linprog(
            objective,
            A_eq=a_eq,
            b_eq=problem.b_eq,
            bounds=problem.bounds,
            method=method,
        )
        result = MultiplierCertificateResult(
            success=bool(res.success),
            alpha=alpha,
            status=int(res.status),
            message=str(res.message),
            x=None if not res.success else np.array(res.x, copy=True),
            solver_method=method,
        )
        if result.success:
            return result
        if first_failure is None:
            first_failure = result
        if result.status != 4:
            return result

    return first_failure


def degree1_identity_coefficients(
    problem: MultiplierCertificateProblem,
    result: MultiplierCertificateResult,
) -> Dict[Monomial, float]:
    """Return the coefficient residuals of the certificate identity.

    A successful certificate should return residual coefficients that are all
    very close to zero.
    """
    if result.x is None:
        raise ValueError("certificate has no primal vector")

    coeff_vector = (problem.a_eq_base + result.alpha * problem.a_eq_alpha) @ result.x
    coeff_vector -= problem.b_eq
    return {
        mono: float(
            coeff_vector[problem.coefficient_row_full_to_reduced[idx]]
            / problem.coefficient_row_orbit_sizes[
                problem.coefficient_row_full_to_reduced[idx]
            ]
        )
        for idx, mono in enumerate(problem.slack_monomials)
    }


def degree1_rhs_polynomial(
    problem: MultiplierCertificateProblem,
    result: MultiplierCertificateResult,
) -> Dict[Monomial, float]:
    """Return the nonnegative slack polynomial N(x)."""
    if result.x is None:
        raise ValueError("certificate has no primal vector")
    start = problem.slack_slice.start
    return {
        mono: float(result.x[start + problem.slack_full_to_reduced[idx]])
        for idx, mono in enumerate(problem.slack_monomials)
        if result.x[start + problem.slack_full_to_reduced[idx]] > 1e-12
    }


def evaluate_degree1_rhs_on_simplex(
    problem: MultiplierCertificateProblem,
    result: MultiplierCertificateResult,
    x: np.ndarray,
) -> float:
    """Evaluate the certified RHS N(x) on a simplex point."""
    return evaluate_polynomial(degree1_rhs_polynomial(problem, result), x)


Degree1CertificateProblem = MultiplierCertificateProblem
Degree1CertificateResult = MultiplierCertificateResult


def build_degree1_problem(d: int) -> Degree1CertificateProblem:
    return build_multiplier_problem(d, multiplier_degree=1)


def solve_degree1_feasibility(
    problem: Degree1CertificateProblem,
    alpha: float,
) -> Degree1CertificateResult:
    return solve_multiplier_feasibility(problem, alpha)
