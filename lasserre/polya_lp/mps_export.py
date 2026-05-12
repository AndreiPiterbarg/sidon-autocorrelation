"""Export a Pólya/Handelman LP to MPS format.

The MPS file is consumable by ANY LP solver: Gurobi, MOSEK, HiGHS, COIN,
cuPDLP, cuOpt, OR-Tools PDLP. Writing this lets us run the same LP on
many backends (e.g., cloud H100 with cuOpt).

Format: standard MPS, free MPS variant (no column-position constraints).
"""
from __future__ import annotations
from typing import IO, List, Optional, Tuple, Iterable
import numpy as np
from scipy import sparse as sp


def _name_var(prefix: str, i: int) -> str:
    return f"{prefix}_{i}"


def write_mps(
    A_eq: sp.csr_matrix,
    b_eq: np.ndarray,
    c: np.ndarray,
    bounds: List[Tuple[Optional[float], Optional[float]]],
    path: str,
    name: str = "polya_lp",
    sense: str = "MIN",
):
    """Write the LP

        sense  c^T x
        s.t.   A_eq x = b_eq
               l_j <= x_j <= u_j   (None for ±inf)

    to file `path` in MPS format.

    Variables get auto-generated names x_0, x_1, ..., x_{n-1}.
    Rows get names r_0, r_1, ..., r_{m-1}; objective row "OBJ".
    """
    n = A_eq.shape[1]
    m = A_eq.shape[0]
    assert b_eq.shape[0] == m
    assert c.shape[0] == n
    assert len(bounds) == n
    sense = sense.upper()
    assert sense in ("MIN", "MAX")

    # MPS by columns: for each column, list (row, value) pairs.
    A_csc = A_eq.tocsc()

    with open(path, "w") as f:
        f.write(f"NAME          {name}\n")
        if sense == "MAX":
            f.write("OBJSENSE\n    MAX\n")

        # ROWS section
        f.write("ROWS\n")
        f.write(" N  OBJ\n")
        for i in range(m):
            f.write(f" E  r_{i}\n")

        # COLUMNS section: each variable's nonzeros
        f.write("COLUMNS\n")
        for j in range(n):
            col_name = _name_var("x", j)
            # Objective coefficient
            if c[j] != 0.0:
                f.write(f"    {col_name:10s}  OBJ        {c[j]:.16g}\n")
            # Constraint coefficients
            start, end = A_csc.indptr[j], A_csc.indptr[j + 1]
            for k in range(start, end):
                i = A_csc.indices[k]
                v = A_csc.data[k]
                if v != 0.0:
                    f.write(f"    {col_name:10s}  r_{i:<6d}   {v:.16g}\n")

        # RHS section
        f.write("RHS\n")
        for i in range(m):
            if b_eq[i] != 0.0:
                f.write(f"    RHS         r_{i:<6d}   {b_eq[i]:.16g}\n")

        # BOUNDS section
        f.write("BOUNDS\n")
        for j in range(n):
            col_name = _name_var("x", j)
            lo, hi = bounds[j]
            if lo is None and hi is None:
                # Free variable
                f.write(f" FR BND        {col_name}\n")
            elif lo is None:
                # Upper-bounded only
                f.write(f" MI BND        {col_name}\n")  # lower = -inf
                f.write(f" UP BND        {col_name}  {hi:.16g}\n")
            elif hi is None:
                # Lower-bounded only
                if lo != 0.0:
                    f.write(f" LO BND        {col_name}  {lo:.16g}\n")
                # else: default lower bound is 0, no need to specify
            else:
                # Both bounded
                if lo == hi:
                    f.write(f" FX BND        {col_name}  {lo:.16g}\n")
                else:
                    if lo != 0.0:
                        f.write(f" LO BND        {col_name}  {lo:.16g}\n")
                    f.write(f" UP BND        {col_name}  {hi:.16g}\n")

        f.write("ENDATA\n")


def write_buildresult_mps(build, path: str, sense: str = "MIN", name: str = "polya_lp"):
    """Write a BuildResult or MomentLPBuild to MPS."""
    write_mps(
        A_eq=build.A_eq,
        b_eq=build.b_eq,
        c=build.c,
        bounds=build.bounds,
        path=path,
        name=name,
        sense=sense,
    )
