"""Batch runner: sweep many (d, degree) cells in parallel on a CPU pod.

Writes one JSONL per cell to data/swd_batch/. Prints a summary at the end.
Installs scipy if missing.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _ensure_scipy() -> None:
    try:
        import scipy.optimize  # noqa: F401
    except ImportError:
        print("Installing scipy...", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "scipy"]
        )


_ensure_scipy()

from simplex_window_dual.run_sweep import run_dimension_sweep


CELLS: list[tuple[int, int, float, float]] = [
    # (d, degree, alpha_start, alpha_stop) — informative cells only
    (8, 4, 1.12, 1.30),
    (12, 3, 1.10, 1.30),
    (12, 4, 1.12, 1.30),
    (16, 2, 1.10, 1.30),
    (16, 3, 1.12, 1.30),
    (20, 2, 1.12, 1.30),
    (20, 3, 1.14, 1.30),
]


def _alphas(lo: float, hi: float, step: float = 0.01) -> list[float]:
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 6) for i in range(n)]


def _out_path(d: int, deg: int) -> Path:
    p = Path("data/swd_batch") / f"d{d}_r{deg}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def run_cell(cell: tuple[int, int, float, float]) -> dict:
    d, deg, lo, hi = cell
    out = _out_path(d, deg)
    t0 = time.perf_counter()
    try:
        summary = run_dimension_sweep(
            d=d,
            alphas=_alphas(lo, hi),
            multiplier_degree=deg,
            use_reflection_symmetry=True,
            use_reflection_row_orbits=True,
            stop_on_first_infeasible=True,
            jsonl_out=str(out),
            resume=True,
        )
        return {
            "d": d,
            "degree": deg,
            "best_feasible": summary.best_feasible_alpha,
            "first_infeasible": summary.first_infeasible_alpha,
            "n_vars": summary.n_vars,
            "n_eq_rows": summary.n_eq_rows,
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "out": str(out),
        }
    except Exception as exc:
        return {
            "d": d,
            "degree": deg,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }


def main() -> None:
    n_workers = min(12, len(CELLS))
    print(
        f"Batch simplex-window dual sweep: {len(CELLS)} cells, "
        f"{n_workers} parallel workers",
        flush=True,
    )
    for cell in CELLS:
        print(f"  planned: d={cell[0]} deg={cell[1]} alpha=[{cell[2]},{cell[3]}]", flush=True)

    results: list[dict] = []
    t_start = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(run_cell, c): c for c in CELLS}
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            results.append(r)
            print(
                f"[DONE] d={r['d']} deg={r['degree']} "
                f"best={r.get('best_feasible')} "
                f"first_fail={r.get('first_infeasible')} "
                f"vars={r.get('n_vars')} rows={r.get('n_eq_rows')} "
                f"t={r['elapsed_s']}s "
                f"err={r.get('error', '')}",
                flush=True,
            )

    total = time.perf_counter() - t_start
    print(f"\n=== ALL DONE in {total:.1f}s ===")

    results.sort(key=lambda r: (r["d"], r["degree"]))
    summary_path = Path("data/swd_batch") / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nSummary JSON: {summary_path}")

    print("\n d  deg  best_feasible   first_infeasible  vars    rows    time(s)")
    for r in results:
        print(
            f"{r['d']:>2}  {r['degree']:>3}  "
            f"{str(r.get('best_feasible')):>14}  "
            f"{str(r.get('first_infeasible')):>16}  "
            f"{str(r.get('n_vars', '-')):>6}  "
            f"{str(r.get('n_eq_rows', '-')):>6}  "
            f"{r['elapsed_s']:>7}"
        )


if __name__ == "__main__":
    main()
