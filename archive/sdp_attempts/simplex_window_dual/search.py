"""Search helpers for simplex-window dual certificates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from simplex_window_dual.degree1 import (
    MultiplierCertificateProblem,
    MultiplierCertificateResult,
    build_multiplier_problem,
    solve_multiplier_feasibility,
)


@dataclass(frozen=True)
class SearchOutcome:
    problem: MultiplierCertificateProblem
    best_result: MultiplierCertificateResult | None
    attempted_alphas: List[float]
    first_infeasible_alpha: float | None = None


def search_multiplier_grid(
    d: int,
    alphas: Iterable[float],
    multiplier_degree: int = 1,
    use_reflection_symmetry: bool = False,
    use_reflection_row_orbits: bool = False,
    stop_on_first_infeasible: bool = False,
) -> SearchOutcome:
    """Try a list of alpha values and keep the best feasible certificate.

    When ``stop_on_first_infeasible`` is true, the input alpha list must be
    nondecreasing. This is sound because feasibility is downward closed in
    alpha: if a certificate works at alpha_hi, then it also works at any
    alpha_lo <= alpha_hi by moving

        (alpha_hi - alpha_lo) * sum_W Lambda_W(x)

    into the nonnegative slack polynomial.
    """
    problem = build_multiplier_problem(
        d,
        multiplier_degree=multiplier_degree,
        use_reflection_symmetry=use_reflection_symmetry,
        use_reflection_row_orbits=use_reflection_row_orbits,
    )
    best = None
    attempted = []
    first_infeasible = None
    last_alpha = None
    for alpha in alphas:
        alpha = float(alpha)
        if (
            stop_on_first_infeasible
            and last_alpha is not None
            and alpha < last_alpha - 1e-12
        ):
            raise ValueError(
                "stop_on_first_infeasible requires a nondecreasing alpha sequence"
            )
        last_alpha = alpha
        attempted.append(alpha)
        result = solve_multiplier_feasibility(problem, alpha)
        if result.success:
            best = result
        elif stop_on_first_infeasible:
            first_infeasible = alpha
            break
    return SearchOutcome(
        problem=problem,
        best_result=best,
        attempted_alphas=attempted,
        first_infeasible_alpha=first_infeasible,
    )


def search_degree1_grid(
    d: int,
    alphas: Iterable[float],
    multiplier_degree: int = 1,
    use_reflection_symmetry: bool = False,
    use_reflection_row_orbits: bool = False,
    stop_on_first_infeasible: bool = False,
) -> SearchOutcome:
    """Backward-compatible alias for multiplier-certificate grid search."""
    return search_multiplier_grid(
        d=d,
        alphas=alphas,
        multiplier_degree=multiplier_degree,
        use_reflection_symmetry=use_reflection_symmetry,
        use_reflection_row_orbits=use_reflection_row_orbits,
        stop_on_first_infeasible=stop_on_first_infeasible,
    )
