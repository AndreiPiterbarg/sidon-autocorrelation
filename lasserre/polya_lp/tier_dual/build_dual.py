"""Construct the LP dual of the Sidon Handelman primal.

Primal (from lasserre/polya_lp/build.py, equality form):
   min  c^T x = -alpha
   s.t. A_eq x = b_eq
        x = [alpha (free), lambda (>=0), q (free), c (>=0)]
   where:
     polya rows (one per beta with |beta| <= R):
       sum_W lambda_W coeff_W(beta) - alpha delta_{beta=0}
         + q_beta - sum_j q_{beta-e_j} - c_beta = 0
     simplex row:
       sum_W lambda_W = 1

Variable types and KKT (LP duality):
  alpha   free           ->  (A^T y)_alpha = -y_{beta=0}     = c_alpha = -1
                              =>  y_{beta=0} = 1
  lambda  >= 0           ->  (A^T y)_W = sum_b y_b coeff_W(b) + y_simplex <= 0
  q_K     free           ->  (A^T y)_{qK} = y_K - sum_j y_{K+e_j} = 0
                              =>  y_K = sum_{j: K+e_j in monos_le_R} y_{K+e_j}
  c_beta  >= 0           ->  (A^T y)_{c_b} = -y_b <= 0
                              =>  y_beta >= 0

Dual objective: max b^T y = b_simplex * y_simplex + sum_b 0 * y_b = y_simplex.

So the DUAL LP is:
   min  -y_simplex                                  (objective)
   s.t. y_simplex                in R               (1 free)
        y_beta                    >= 0  for all b   (n_le_R bounded)
        y_{beta=0} = 1                               (1 equality)
        y_K - sum_{j: K+e_j in monos_le_R} y_{K+e_j} = 0
                                  for all K with |K| <= R-1   (n_q equalities)
        sum_b y_b coeff_W(b) + y_simplex <= 0
                                  for all W           (n_W inequalities)

Variable count: n_le_R + 1.
Equality count: n_q + 1   (= n_le_R - n_top_degree + 1, roughly).
Inequality count: n_W.

Number of FREE variables in dual: exactly 1.
This is the structural advantage that makes PDLP feasible.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import time

import numpy as np
from scipy import sparse as sp

from lasserre.polya_lp.poly import enum_monomials_le, index_map


# =====================================================================
# Window coefficient extraction (matches primal's coeff_W)
# =====================================================================

def _coeff_W_pairs(M_W: np.ndarray) -> List[Tuple[Tuple[int, ...], float]]:
    """Return list of (beta, coeff) where coeff = coeff_W(beta) and
    beta is a |beta|=2 multi-index. Only nonzero entries.

    For M_W symmetric:
      beta = 2*e_i        ->  M_W[i, i]
      beta = e_i + e_j    ->  2 * M_W[i, j]   (i < j)
    """
    d = M_W.shape[0]
    out: List[Tuple[Tuple[int, ...], float]] = []
    # Diagonal
    for i in range(d):
        v = float(M_W[i, i])
        if v != 0.0:
            beta = tuple(2 if k == i else 0 for k in range(d))
            out.append((beta, v))
    # Strict upper
    for i in range(d):
        for j in range(i + 1, d):
            v = 2.0 * float(M_W[i, j])
            if v != 0.0:
                beta = tuple(1 if (k == i or k == j) else 0 for k in range(d))
                out.append((beta, v))
    return out


# =====================================================================
# Dual LP build
# =====================================================================

@dataclass
class DualBuildResult:
    """LP standard form (min c^T x s.t. A_eq x = b_eq, A_ub x <= b_ub, x in [l,u]).

    Variable layout: x = [y_beta_0, y_beta_1, ..., y_beta_{n_le_R-1}, y_simplex]
    where the y_beta block is in the same order as monos_le_R.
    """
    A_eq: sp.csr_matrix
    b_eq: np.ndarray
    A_ub: sp.csr_matrix
    b_ub: np.ndarray
    c: np.ndarray
    bounds: List[Tuple[Optional[float], Optional[float]]]
    n_vars: int
    y_idx: slice                                # range of y_beta variables
    y_simplex_idx: int                          # index of y_simplex
    monos_le_R: List[Tuple[int, ...]]
    monos_le_Rm1: List[Tuple[int, ...]]
    beta_to_idx: dict
    n_W: int
    n_q_recursion_rows: int
    d: int
    R: int
    build_wall_s: float

    @property
    def n_eq(self) -> int:
        return self.A_eq.shape[0]

    @property
    def n_ub(self) -> int:
        return self.A_ub.shape[0]


def build_dual_lp(
    d: int,
    M_mats: Sequence[np.ndarray],
    R: int,
    verbose: bool = False,
) -> DualBuildResult:
    """Build the LP dual of the Sidon Handelman primal.

    Args:
        d        : ambient polynomial dimension (after Z/2 reduction if any).
        M_mats   : list of window matrices (after Z/2 projection if any).
        R        : Polya certificate degree.

    Returns:
        DualBuildResult ready for MOSEK / HiGHS / pdlp_robust.

    Soundness:
        By LP duality, optimum of (primal LP) = optimum of (dual LP).
        We solve   min -y_simplex   on the dual and report alpha_dual =
        +y_simplex_optimal = -obj. This MUST equal alpha_primal (the
        result of solving lasserre.polya_lp.build with the same d, M_mats, R).
    """
    t0 = time.time()
    n_W = len(M_mats)

    # Enumerate monomials
    monos_le_R = enum_monomials_le(d, R)
    n_le_R = len(monos_le_R)
    beta_to_idx = index_map(monos_le_R)

    monos_le_Rm1 = enum_monomials_le(d, R - 1) if R >= 1 else []
    n_q = len(monos_le_Rm1)

    # ----------------------------------------------------------------
    # Variable layout: x = [y_beta (n_le_R), y_simplex (1)]
    # ----------------------------------------------------------------
    y_simplex_idx = n_le_R
    n_vars = n_le_R + 1

    if verbose:
        print(f"  Dual LP layout: y_beta={n_le_R}, y_simplex=1, total={n_vars}",
              flush=True)
        print(f"  Equalities: 1 (y_0=1) + {n_q} (moment recursion)", flush=True)
        print(f"  Inequalities: {n_W} (window cuts)", flush=True)

    # ----------------------------------------------------------------
    # Equality block: A_eq x = b_eq
    #   Row 0 (y_0 = 1):
    #     coefficient +1 at index beta_to_idx[zero_beta]
    #     RHS = 1
    #
    #   Row 1..n_q (moment recursion, one per K with |K| <= R-1):
    #     y_K - sum_{j: K+e_j in monos_le_R} y_{K+e_j} = 0
    #     coefficient +1 at index beta_to_idx[K]
    #     coefficient -1 at index beta_to_idx[K + e_j]  for each valid j
    #     RHS = 0
    # ----------------------------------------------------------------
    eq_rows: List[int] = []
    eq_cols: List[int] = []
    eq_vals: List[float] = []
    eq_rhs: List[float] = []

    zero_beta = tuple([0] * d)
    if zero_beta not in beta_to_idx:
        raise ValueError("monos_le_R must contain the zero multi-index")

    # Row 0: y_0 = 1
    row_idx = 0
    eq_rows.append(row_idx)
    eq_cols.append(beta_to_idx[zero_beta])
    eq_vals.append(1.0)
    eq_rhs.append(1.0)
    row_idx += 1

    # Rows 1..n_q: moment recursion
    for K in monos_le_Rm1:
        # +1 at y_K
        if K not in beta_to_idx:
            continue  # safety
        eq_rows.append(row_idx)
        eq_cols.append(beta_to_idx[K])
        eq_vals.append(1.0)
        # -1 at y_{K+e_j} for each j with K+e_j in monos_le_R
        for j in range(d):
            shifted = list(K)
            shifted[j] += 1
            if sum(shifted) > R:
                continue
            shifted_t = tuple(shifted)
            j_idx = beta_to_idx.get(shifted_t)
            if j_idx is not None:
                eq_rows.append(row_idx)
                eq_cols.append(j_idx)
                eq_vals.append(-1.0)
        eq_rhs.append(0.0)
        row_idx += 1

    n_eq_rows = row_idx
    A_eq = sp.csr_matrix(
        (np.asarray(eq_vals, dtype=np.float64),
         (np.asarray(eq_rows, dtype=np.int64),
          np.asarray(eq_cols, dtype=np.int64))),
        shape=(n_eq_rows, n_vars),
    )
    b_eq = np.asarray(eq_rhs, dtype=np.float64)

    # ----------------------------------------------------------------
    # Inequality block: A_ub x <= b_ub
    #   For each W:  sum_b y_b coeff_W(b) + y_simplex <= 0
    #     coefficient coeff_W(b) at index beta_to_idx[b]   for each |b|=2 with coeff != 0
    #     coefficient +1 at y_simplex
    #     RHS = 0
    # ----------------------------------------------------------------
    ub_rows: List[int] = []
    ub_cols: List[int] = []
    ub_vals: List[float] = []

    for w, M_W in enumerate(M_mats):
        pairs = _coeff_W_pairs(np.asarray(M_W, dtype=np.float64))
        for beta, val in pairs:
            j = beta_to_idx.get(beta)
            if j is None:
                # If beta is |beta|=2 but somehow not in monos_le_R, skip.
                # Shouldn't happen since |beta|=2 <= R for R >= 2.
                continue
            ub_rows.append(w)
            ub_cols.append(j)
            ub_vals.append(val)
        # + y_simplex
        ub_rows.append(w)
        ub_cols.append(y_simplex_idx)
        ub_vals.append(1.0)

    A_ub = sp.csr_matrix(
        (np.asarray(ub_vals, dtype=np.float64),
         (np.asarray(ub_rows, dtype=np.int64),
          np.asarray(ub_cols, dtype=np.int64))),
        shape=(n_W, n_vars),
    )
    b_ub = np.zeros(n_W, dtype=np.float64)

    # ----------------------------------------------------------------
    # Objective and bounds
    #   We MINIMIZE -y_simplex (so c[y_simplex_idx] = -1, c rest = 0).
    #   y_beta in [0, +inf) for all beta.
    #   y_simplex free.
    # ----------------------------------------------------------------
    c_obj = np.zeros(n_vars, dtype=np.float64)
    c_obj[y_simplex_idx] = -1.0

    bounds: List[Tuple[Optional[float], Optional[float]]] = []
    for _ in range(n_le_R):
        bounds.append((0.0, None))
    bounds.append((None, None))   # y_simplex free

    return DualBuildResult(
        A_eq=A_eq, b_eq=b_eq,
        A_ub=A_ub, b_ub=b_ub,
        c=c_obj, bounds=bounds,
        n_vars=n_vars,
        y_idx=slice(0, n_le_R),
        y_simplex_idx=y_simplex_idx,
        monos_le_R=monos_le_R,
        monos_le_Rm1=monos_le_Rm1,
        beta_to_idx=beta_to_idx,
        n_W=n_W,
        n_q_recursion_rows=n_q,
        d=d, R=R,
        build_wall_s=time.time() - t0,
    )


# =====================================================================
# Pretty summary
# =====================================================================

def summarize(b: DualBuildResult) -> str:
    return (f"Dual LP at d={b.d} R={b.R}: "
            f"n_vars={b.n_vars} (1 free + {b.n_vars-1} bounded), "
            f"n_eq={b.n_eq} (1 + {b.n_q_recursion_rows} recursion), "
            f"n_ub={b.n_ub}, "
            f"nnz_eq={b.A_eq.nnz}, nnz_ub={b.A_ub.nnz}, "
            f"build={b.build_wall_s*1000:.1f}ms")
