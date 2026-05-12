"""End-to-end driver: build LP at (d, R), solve, report."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional
import json
import time

import numpy as np

from lasserre.polya_lp.build import (
    BuildOptions,
    BuildResult,
    build_handelman_lp,
    build_window_matrices,
)
from lasserre.polya_lp.solve import SolveResult, solve_lp
from lasserre.polya_lp.symmetry import (
    project_window_set_to_z2,
    project_window_set_to_z2_rescaled,
    z2_dim,
)


@dataclass
class RunRecord:
    d: int
    R: int
    use_z2: bool
    fixed_lambda: bool
    use_q_polynomial: bool
    n_vars: int
    n_eq: int
    n_nonzero_A: int
    n_windows_in_LP: int
    build_wall_s: float
    solve_wall_s: float
    status: str
    alpha: Optional[float]
    val_d_known: Optional[float]
    gap_to_known: Optional[float]
    notes: str = ""


VAL_D_KNOWN = {
    4: 1.102, 6: 1.171, 8: 1.205, 10: 1.241,
    12: 1.271, 14: 1.284, 16: 1.319,
    32: 1.336, 64: 1.384, 128: 1.420, 256: 1.448,
}


def run_one(
    d: int,
    R: int,
    use_z2: bool = True,
    fixed_lambda: Optional[np.ndarray] = None,
    use_q_polynomial: bool = True,
    solver: str = "highs",
    verbose: bool = False,
) -> tuple[RunRecord, BuildResult, SolveResult]:
    """Build and solve a single Handelman LP."""

    if verbose:
        print(f"\n=== Run d={d}, R={R}, z2={use_z2}, "
              f"fixed_lambda={fixed_lambda is not None}, q_poly={use_q_polynomial} ===")

    # Build window matrices
    t0 = time.time()
    _, M_mats = build_window_matrices(d)
    n_W_orig = len(M_mats)

    # Optional Z/2 symmetry projection (with rescaling so LP uses standard simplex)
    if use_z2:
        M_mats_eff, _counts = project_window_set_to_z2_rescaled(M_mats, d)
        d_eff = z2_dim(d)
        if verbose:
            print(f"  Z/2: d {d} -> d_eff {d_eff}, "
                  f"windows {n_W_orig} -> {len(M_mats_eff)} unique symmetric")
    else:
        M_mats_eff = M_mats
        d_eff = d
        if verbose:
            print(f"  No symmetry: d_eff={d}, windows={n_W_orig}")

    if fixed_lambda is not None and len(fixed_lambda) != len(M_mats_eff):
        # If user provided fixed lambda for original windows, project it
        raise ValueError(
            f"fixed_lambda length {len(fixed_lambda)} doesn't match "
            f"effective window count {len(M_mats_eff)}. "
            "Project lambda before passing.")

    # Build LP
    opts = BuildOptions(
        R=R,
        use_z2=use_z2,
        fixed_lambda=fixed_lambda,
        use_q_polynomial=use_q_polynomial,
        verbose=verbose,
    )
    build = build_handelman_lp(d_eff, M_mats_eff, opts)
    if verbose:
        print(f"  Build: n_vars={build.n_vars}, n_eq={build.A_eq.shape[0]}, "
              f"nnz(A)={build.n_nonzero_A}, wall={build.build_wall_s:.2f}s")

    # Solve
    sol = solve_lp(build, solver=solver, verbose=verbose)
    if verbose:
        print(f"  Solve: status={sol.status}, alpha={sol.alpha}, "
              f"wall={sol.wall_s:.2f}s, solver={sol.solver}")

    val_known = VAL_D_KNOWN.get(d)
    gap = (val_known - sol.alpha) if (val_known is not None and sol.alpha is not None) else None

    record = RunRecord(
        d=d, R=R, use_z2=use_z2,
        fixed_lambda=(fixed_lambda is not None),
        use_q_polynomial=use_q_polynomial,
        n_vars=build.n_vars,
        n_eq=build.A_eq.shape[0],
        n_nonzero_A=build.n_nonzero_A,
        n_windows_in_LP=len(M_mats_eff) if fixed_lambda is None else 0,
        build_wall_s=build.build_wall_s,
        solve_wall_s=sol.wall_s,
        status=sol.status,
        alpha=sol.alpha,
        val_d_known=val_known,
        gap_to_known=gap,
        notes="",
    )
    return record, build, sol


def run_sweep(
    d_list: List[int],
    R_list: List[int],
    use_z2: bool = True,
    use_q_polynomial: bool = True,
    solver: str = "highs",
    verbose: bool = True,
) -> List[RunRecord]:
    """Run a (d, R) grid sweep and return records."""
    records: List[RunRecord] = []
    for d in d_list:
        for R in R_list:
            try:
                rec, _, _ = run_one(
                    d=d, R=R,
                    use_z2=use_z2,
                    use_q_polynomial=use_q_polynomial,
                    solver=solver,
                    verbose=verbose,
                )
            except Exception as e:
                rec = RunRecord(
                    d=d, R=R, use_z2=use_z2,
                    fixed_lambda=False,
                    use_q_polynomial=use_q_polynomial,
                    n_vars=-1, n_eq=-1, n_nonzero_A=-1,
                    n_windows_in_LP=-1,
                    build_wall_s=0.0, solve_wall_s=0.0,
                    status="EXCEPTION",
                    alpha=None,
                    val_d_known=VAL_D_KNOWN.get(d),
                    gap_to_known=None,
                    notes=f"{type(e).__name__}: {str(e)[:200]}",
                )
            records.append(rec)
            if verbose:
                print(f"  >>> d={d} R={R} -> alpha={rec.alpha} "
                      f"(known val={rec.val_d_known}, gap={rec.gap_to_known}, "
                      f"build={rec.build_wall_s:.1f}s, solve={rec.solve_wall_s:.1f}s)")
    return records


def records_to_json(records: List[RunRecord], path: str) -> None:
    out = [asdict(r) for r in records]
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
