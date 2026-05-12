"""Resumable sweep runner for simplex-window multiplier certificates."""

from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from simplex_window_dual.degree1 import build_multiplier_problem, solve_multiplier_feasibility


@dataclass(frozen=True)
class SweepAttemptRecord:
    kind: str
    d: int
    alpha: float
    multiplier_degree: int
    use_reflection_symmetry: bool
    use_reflection_row_orbits: bool
    success: bool
    status: int
    message: str
    solver_method: str | None
    solve_seconds: float
    n_vars: int
    n_eq_rows: int
    host: str


@dataclass(frozen=True)
class DimensionSweepSummary:
    d: int
    multiplier_degree: int
    use_reflection_symmetry: bool
    use_reflection_row_orbits: bool
    attempted_alphas: tuple[float, ...]
    best_feasible_alpha: float | None
    first_infeasible_alpha: float | None
    build_seconds: float
    n_vars: int
    n_eq_rows: int


def _frange(start: float, stop: float, step: float) -> list[float]:
    values = []
    value = start
    while value <= stop + 1e-12:
        values.append(round(value, 10))
        value += step
    return values


def _parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def _load_attempt_records(
    path: Path,
    *,
    d: int,
    multiplier_degree: int,
    use_reflection_symmetry: bool,
    use_reflection_row_orbits: bool,
) -> list[dict]:
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("kind") != "attempt":
                continue
            if payload.get("d") != d:
                continue
            if payload.get("multiplier_degree") != multiplier_degree:
                continue
            if payload.get("use_reflection_symmetry") != use_reflection_symmetry:
                continue
            if payload.get("use_reflection_row_orbits") != use_reflection_row_orbits:
                continue
            records.append(payload)
    return records


def run_dimension_sweep(
    *,
    d: int,
    alphas: Sequence[float],
    multiplier_degree: int,
    use_reflection_symmetry: bool,
    use_reflection_row_orbits: bool,
    stop_on_first_infeasible: bool = True,
    jsonl_out: str | None = None,
    resume: bool = False,
) -> DimensionSweepSummary:
    build_start = time.perf_counter()
    problem = build_multiplier_problem(
        d,
        multiplier_degree=multiplier_degree,
        use_reflection_symmetry=use_reflection_symmetry,
        use_reflection_row_orbits=use_reflection_row_orbits,
    )
    build_seconds = time.perf_counter() - build_start

    out_path = None if jsonl_out is None else Path(jsonl_out)
    resume_records = []
    if resume and out_path is not None:
        resume_records = _load_attempt_records(
            out_path,
            d=d,
            multiplier_degree=multiplier_degree,
            use_reflection_symmetry=use_reflection_symmetry,
            use_reflection_row_orbits=use_reflection_row_orbits,
        )

    attempted_lookup = {float(record["alpha"]): record for record in resume_records}
    attempted_alphas = [float(record["alpha"]) for record in resume_records]
    best_feasible_alpha = None
    first_infeasible_alpha = None

    for record in resume_records:
        alpha = float(record["alpha"])
        if record["success"]:
            best_feasible_alpha = alpha
        elif stop_on_first_infeasible:
            first_infeasible_alpha = alpha
            break

    if first_infeasible_alpha is None:
        for alpha in alphas:
            alpha = float(alpha)
            if alpha in attempted_lookup:
                continue

            solve_start = time.perf_counter()
            result = solve_multiplier_feasibility(problem, alpha)
            solve_seconds = time.perf_counter() - solve_start
            record = SweepAttemptRecord(
                kind="attempt",
                d=d,
                alpha=alpha,
                multiplier_degree=multiplier_degree,
                use_reflection_symmetry=use_reflection_symmetry,
                use_reflection_row_orbits=use_reflection_row_orbits,
                success=result.success,
                status=result.status,
                message=result.message,
                solver_method=result.solver_method,
                solve_seconds=solve_seconds,
                n_vars=problem.n_vars,
                n_eq_rows=len(problem.b_eq),
                host=socket.gethostname(),
            )
            attempted_lookup[alpha] = asdict(record)
            attempted_alphas.append(alpha)
            if out_path is not None:
                _append_jsonl(out_path, asdict(record))

            if result.success:
                best_feasible_alpha = alpha
            elif stop_on_first_infeasible:
                first_infeasible_alpha = alpha
                break

    summary = DimensionSweepSummary(
        d=d,
        multiplier_degree=multiplier_degree,
        use_reflection_symmetry=use_reflection_symmetry,
        use_reflection_row_orbits=use_reflection_row_orbits,
        attempted_alphas=tuple(attempted_alphas),
        best_feasible_alpha=best_feasible_alpha,
        first_infeasible_alpha=first_infeasible_alpha,
        build_seconds=build_seconds,
        n_vars=problem.n_vars,
        n_eq_rows=len(problem.b_eq),
    )

    if out_path is not None:
        _append_jsonl(
            out_path,
            {
                "kind": "summary",
                **asdict(summary),
            },
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a resumable simplex-window multiplier sweep"
    )
    parser.add_argument("--d-values", type=str, default="12,14,16")
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--reflection-symmetry", action="store_true")
    parser.add_argument("--reflection-row-orbits", action="store_true")
    parser.add_argument("--alpha-start", type=float, default=1.10)
    parser.add_argument("--alpha-stop", type=float, default=1.30)
    parser.add_argument("--alpha-step", type=float, default=0.01)
    parser.add_argument(
        "--jsonl-out",
        type=str,
        default="data/simplex_window_dual/prime_sweep.jsonl",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    d_values = _parse_int_list(args.d_values)
    alphas = _frange(args.alpha_start, args.alpha_stop, args.alpha_step)

    print(
        f"Sweeping d={d_values}, degree={args.degree}, "
        f"reflection_symmetry={args.reflection_symmetry}, "
        f"reflection_row_orbits={args.reflection_row_orbits}"
    )
    print(
        f"alpha grid: {alphas[0]:.4f} .. {alphas[-1]:.4f} "
        f"({len(alphas)} points)"
    )
    print(f"jsonl_out: {args.jsonl_out}")
    print(flush=True)

    summaries = []
    for d in d_values:
        print(f"[d={d}] building/problem setup...", flush=True)
        summary = run_dimension_sweep(
            d=d,
            alphas=alphas,
            multiplier_degree=args.degree,
            use_reflection_symmetry=args.reflection_symmetry,
            use_reflection_row_orbits=args.reflection_row_orbits,
            stop_on_first_infeasible=True,
            jsonl_out=args.jsonl_out,
            resume=args.resume,
        )
        summaries.append(summary)
        print(
            f"[d={d}] vars={summary.n_vars} eq_rows={summary.n_eq_rows} "
            f"build={summary.build_seconds:.2f}s "
            f"best={summary.best_feasible_alpha} "
            f"first_fail={summary.first_infeasible_alpha}",
            flush=True,
        )

    print("\nSummary")
    print("d  vars  eq_rows  best_feasible  first_infeasible")
    for summary in summaries:
        print(
            f"{summary.d:>2} {summary.n_vars:>6} {summary.n_eq_rows:>8} "
            f"{str(summary.best_feasible_alpha):>13} "
            f"{str(summary.first_infeasible_alpha):>16}"
        )


if __name__ == "__main__":
    main()
