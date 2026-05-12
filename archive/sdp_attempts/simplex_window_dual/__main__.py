"""CLI for simplex-window dual certificate search."""

from __future__ import annotations

import argparse

from simplex_window_dual.search import search_multiplier_grid


def _frange(start: float, stop: float, step: float):
    value = start
    while value <= stop + 1e-12:
        yield round(value, 10)
        value += step


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search simplex-window multiplier certificates"
    )
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--reflection-symmetry", action="store_true")
    parser.add_argument("--reflection-row-orbits", action="store_true")
    parser.add_argument("--alpha-start", type=float, default=1.00)
    parser.add_argument("--alpha-stop", type=float, default=1.10)
    parser.add_argument("--alpha-step", type=float, default=0.01)
    args = parser.parse_args()

    outcome = search_multiplier_grid(
        d=args.d,
        alphas=_frange(args.alpha_start, args.alpha_stop, args.alpha_step),
        multiplier_degree=args.degree,
        use_reflection_symmetry=args.reflection_symmetry,
        use_reflection_row_orbits=args.reflection_row_orbits,
        stop_on_first_infeasible=True,
    )
    best = outcome.best_result
    if best is None:
        print(
            f"No feasible degree-{args.degree} certificate found for d={args.d}."
        )
        return

    print(
        f"Best feasible alpha for d={args.d}, degree={args.degree}: "
        f"{best.alpha:.10f}"
    )


if __name__ == "__main__":
    main()
