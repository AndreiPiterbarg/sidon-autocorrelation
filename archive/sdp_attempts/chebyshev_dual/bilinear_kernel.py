"""Jacobi-Anger expansion of cos/sin(2 pi l x) in the Chebyshev basis of
y = 4x on [-1/4, 1/4], plus the rank-<=2 bilinear-kernel matrices A_l that
assemble into A(r) with c^T A(r) c approximating the Sidon primal-dual inner product

    integral integral f(x) f(y) p(x+y) dx dy,    p(t) = r_0 + 2 sum_l r_l cos(2 pi l t).

Math
====

With omega_l := pi l / 2, the classical Jacobi-Anger identity gives

    cos(omega y) = J_0(omega) T_0(y) + 2 sum_{k >= 1} (-1)^k J_{2k}(omega) T_{2k}(y),
    sin(omega y) = 2 sum_{k >= 0} (-1)^k J_{2k+1}(omega) T_{2k+1}(y).

Truncated to degree K = 2N:

    cos(omega y) ~= sum_{k=0}^{K} alpha_{l, k} T_k(y),
    sin(omega y) ~= sum_{k=0}^{K} beta_{l,  k} T_k(y),

with alpha_{l, 2m} = 2(-1)^m J_{2m}(omega_l) for m>=1, alpha_{l, 0} = J_0(omega_l),
alpha_{l, odd} = 0, and symmetrically beta_{l, 2m+1} = 2(-1)^m J_{2m+1}(omega_l),
beta_{l, even} = 0.

Since J_k(omega_l) at omega_l = pi l / 2 is generically irrational, we represent
alpha, beta as arb (midpoint-radius) balls; the cert path includes a rigorous
rounding error.

Bilinear form
-------------

Using C_l[f] := int f(x) cos(2 pi l x) dx ~= alpha_l^T c (with c_k the Chebyshev
moments of f), and similarly S_l[f] ~= beta_l^T c, and the factorization

    cos(2 pi l (x+y)) = cos(2 pi l x) cos(2 pi l y) - sin(2 pi l x) sin(2 pi l y),

we get

    integral integral f f p = r_0 c_0^2
        + 2 sum_{l=1}^{D} r_l [ (alpha_l^T c)^2 - (beta_l^T c)^2 ] + eps(f, r, N, D),

where eps(f, r, N, D) is the Jacobi-Anger truncation error.  In matrix form,

    c^T A(r) c  =  r_0 c_0^2 + 2 sum_{l=1}^{D} r_l [ (alpha_l^T c)^2 - (beta_l^T c)^2 ],
    A(r) = r_0 (e_0 e_0^T) + 2 sum_{l=1}^{D} r_l (alpha_l alpha_l^T - beta_l beta_l^T).

Each per-l basis matrix A_l has rank <= 2.  The assembled A(r) is symmetric, of
size (2N+1) x (2N+1), and linear in r.

Truncation bound
----------------

With C_l^trunc = sup_{y in [-1,1]} |cos tail| and S_l^trunc likewise for sin:

    C_l^trunc + S_l^trunc  <=  2 sum_{n >= 2N+1} |J_n(omega_l)|.

For n0 >= omega_l we use the rigorous bound |J_n(omega_l)| <= (omega_l/2)^n / n!
(Watson, Bessel functions, sec. 3.31), plus the geometric tail bound

    sum_{n >= n0} (omega_l/2)^n / n!  <=  2 (omega_l/2)^{n0} / n0!    when n0 >= omega_l.

The certified pipeline uses D <= 2N / pi roughly, which keeps truncation below
machine epsilon at prec >= 128 bits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from flint import arb, arb_mat, ctx


# ---------------------------------------------------------------------
# Jacobi-Anger coefficients
# ---------------------------------------------------------------------

def omega_of(l: int) -> arb:
    """omega_l = pi * l / 2, as an arb ball at the current working precision."""
    return arb.pi() * arb(l) / arb(2)


def jacobi_anger_chebyshev_coeffs(
    l: int, K: int
) -> Tuple[List[arb], List[arb]]:
    """Chebyshev coefficients for cos(omega_l y) and sin(omega_l y) on y in [-1,1].

    Returns (alpha_l, beta_l), each a list of length K+1.  For l = 0 alpha is
    (1, 0, 0, ...) and beta is all zero.

    Alpha is nonzero only at even k; beta only at odd k.  Uses arb.bessel_j at
    the current ctx.prec.
    """
    if K < 0:
        raise ValueError("K must be >= 0")
    alpha: List[arb] = [arb(0)] * (K + 1)
    beta: List[arb] = [arb(0)] * (K + 1)
    if l == 0:
        alpha[0] = arb(1)
        return alpha, beta
    w = omega_of(l)
    # alpha: J_0, 2(-1)^m J_{2m}
    alpha[0] = w.bessel_j(arb(0))
    m = 1
    while 2 * m <= K:
        J = w.bessel_j(arb(2 * m))
        sign = arb(-1) if m % 2 else arb(1)
        alpha[2 * m] = arb(2) * sign * J
        m += 1
    # beta: 2(-1)^m J_{2m+1}
    m = 0
    while 2 * m + 1 <= K:
        J = w.bessel_j(arb(2 * m + 1))
        sign = arb(-1) if m % 2 else arb(1)
        beta[2 * m + 1] = arb(2) * sign * J
        m += 1
    return alpha, beta


# ---------------------------------------------------------------------
# Truncation bound
# ---------------------------------------------------------------------

def _bessel_tail_L1(
    l: int,
    n_start: int,
    *,
    max_explicit: int = 400,
) -> arb:
    """Rigorous arb upper bound on  sum_{n >= n_start} |J_n(omega_l)|.

    Sums |J_n| in arb for n = n_start, ..., n_stop-1 where n_stop is the first
    integer >= max(omega_l, n_start) + 5, then adds the geometric tail bound
        sum_{n >= n_stop} (omega_l/2)^n / n!  <=  2 (omega_l/2)^{n_stop} / n_stop!
    valid because n_stop > omega_l.
    """
    if l == 0:
        return arb(0)
    w = omega_of(l)
    # Choose n_stop > omega_l so the (omega/2)^n/n! bound is monotone-decreasing.
    omega_upper = int(float(w.upper())) + 1
    n_stop = max(n_start + 30, omega_upper + 5)
    n_stop = min(n_stop, n_start + max_explicit)

    total = arb(0)
    for n in range(n_start, n_stop):
        total = total + abs(w.bessel_j(arb(n)))

    # Tail: sum_{n >= n_stop} (w/2)^n / n!  <=  2 * (w/2)^{n_stop} / n_stop!
    w_half = w / arb(2)
    term = arb(1)
    for i in range(1, n_stop + 1):
        term = term * w_half / arb(i)
    total = total + arb(2) * term
    return total


def jacobi_anger_truncation_infty(l: int, K: int) -> arb:
    """Arb upper bound on  sup_{y in [-1,1]} |R^(c)_l(y)| + |R^(s)_l(y)|,
    where R^(c)_l and R^(s)_l are the Chebyshev-series truncation residuals
    of cos(omega_l y) and sin(omega_l y) at degree K.

    Uses |T_k| <= 1 and 2 sum_{n >= K+1} |J_n(omega_l)|  (both parities).
    """
    if l == 0:
        return arb(0)
    # Cos tail: coefs at even k > K sum to 2 * sum_{m: 2m > K} |J_{2m}|.
    # Sin tail: coefs at odd  k > K sum to 2 * sum_{m: 2m+1 > K} |J_{2m+1}|.
    # Combined, sum over all n > K.
    return arb(2) * _bessel_tail_L1(l, K + 1)


def kernel_truncation_error_bound(
    r: Sequence,
    N: int,
    D: int,
) -> arb:
    """Rigorous arb upper bound on |eps(f, r, N, D)| for any f in A.

    From the derivation:  for each l in 1..D, the contribution to eps is
        2 r_l [ (alpha^T c)^2 - C_l^2  -  ((beta^T c)^2 - S_l^2) ]
        = -2 r_l [ E_l^(c) (C_l + alpha^T c)  -  E_l^(s) (S_l + beta^T c) ],
    where |E_l^(c)| <= C_l^trunc, |E_l^(s)| <= S_l^trunc, and |C_l|, |S_l|,
    |alpha^T c|, |beta^T c| <= 1 + C_l^trunc (resp. + S_l^trunc), because
    |C_l[f]|, |S_l[f]| <= int|f| = 1.

    Hence the uniform bound
        |eps| <= 2 sum_{l=1}^{D} |r_l| [ (2 + C_l^trunc) C_l^trunc
                                       + (2 + S_l^trunc) S_l^trunc ]
            <= 2 sum_{l=1}^{D} |r_l| (C_l^trunc + S_l^trunc) (2 + C_l^trunc + S_l^trunc).
    We use C_l^trunc + S_l^trunc <= jacobi_anger_truncation_infty(l, 2N).
    """
    if D < 1:
        return arb(0)
    K = 2 * N
    total = arb(0)
    for l in range(1, D + 1):
        rl = r[l] if isinstance(r[l], arb) else arb(r[l])
        tau = jacobi_anger_truncation_infty(l, K)
        total = total + arb(2) * abs(rl) * (arb(2) + tau) * tau
    return total


# ---------------------------------------------------------------------
# Tables and per-l basis matrices
# ---------------------------------------------------------------------

@dataclass
class JacobiAngerTables:
    """Precomputed Chebyshev-basis Jacobi-Anger tables at (N, D, prec).

    alphas[l], betas[l]: length-(2N+1) arb lists for l = 0, ..., D.
    A_basis[l]: (2N+1)x(2N+1) arb_mat basis for c^T A(r) c decomposition.
        A_basis[0] = e_0 e_0^T (unit in c_0^2 direction, coefficient r_0).
        A_basis[l] = 2 (alpha_l alpha_l^T - beta_l beta_l^T) for l >= 1.
    """
    N: int
    D: int
    prec: int
    alphas: List[List[arb]] = field(default_factory=list)
    betas: List[List[arb]] = field(default_factory=list)
    A_basis: List[arb_mat] = field(default_factory=list)

    @property
    def n(self) -> int:
        return 2 * self.N + 1


def build_jacobi_anger_tables(N: int, D: int, prec: int = 128) -> JacobiAngerTables:
    """Assemble Chebyshev coefficients and per-l basis matrices for the bilinear kernel.

    Uses the currently set ctx.prec for arb computations; documents the value
    taken.  Caller may increase ctx.prec before calling for tighter error balls.
    """
    old_prec = ctx.prec
    ctx.prec = max(prec, old_prec)
    try:
        alphas: List[List[arb]] = []
        betas: List[List[arb]] = []
        K = 2 * N
        for l in range(D + 1):
            a, b = jacobi_anger_chebyshev_coeffs(l, K)
            alphas.append(a)
            betas.append(b)

        n = 2 * N + 1
        A_basis: List[arb_mat] = []

        # A_0 = e_0 e_0^T
        A0 = arb_mat(n, n)
        A0[0, 0] = arb(1)
        A_basis.append(A0)

        # A_l = 2 (alpha alpha^T - beta beta^T) for l = 1, ..., D
        two = arb(2)
        for l in range(1, D + 1):
            Al = arb_mat(n, n)
            a = alphas[l]
            b = betas[l]
            for i in range(n):
                ai = a[i]
                bi = b[i]
                for j in range(n):
                    aj = a[j]
                    bj = b[j]
                    Al[i, j] = two * (ai * aj - bi * bj)
            A_basis.append(Al)

        return JacobiAngerTables(
            N=N, D=D, prec=ctx.prec,
            alphas=alphas, betas=betas, A_basis=A_basis,
        )
    finally:
        ctx.prec = old_prec


def build_kernel_matrix_A(
    r: Sequence,
    tables: JacobiAngerTables,
) -> arb_mat:
    """Assemble A(r) = sum_{l=0}^{D} r_l * A_basis[l] as an arb_mat.

    Accepts any sequence r of length D+1 where entries are arb / int / rational-convertible.
    """
    D = tables.D
    if len(r) != D + 1:
        raise ValueError(f"expected len(r) == D+1 = {D+1}, got {len(r)}")
    n = tables.n
    A = arb_mat(n, n)
    for l in range(D + 1):
        rl = r[l] if isinstance(r[l], arb) else arb(r[l])
        if rl == 0:
            continue
        A = A + tables.A_basis[l] * rl
    return A


# ---------------------------------------------------------------------
# Numeric helpers for tests
# ---------------------------------------------------------------------

def arb_mat_to_numpy_mid(M: arb_mat) -> "np.ndarray":
    """Midpoints of an arb_mat as float64 numpy."""
    import numpy as np
    r, c = M.nrows(), M.ncols()
    out = np.zeros((r, c), dtype=np.float64)
    for i in range(r):
        for j in range(c):
            out[i, j] = float(M[i, j].mid())
    return out


def evaluate_cheb_series_float(coefs: Sequence[arb], y: float) -> float:
    """Evaluate sum_{k=0}^{K} coefs[k] T_k(y) at a float y in [-1, 1] via Clenshaw-like
    recurrence (using arb midpoints).  Returns float."""
    K = len(coefs) - 1
    if K < 0:
        return 0.0
    if K == 0:
        return float(coefs[0].mid())
    # Standard T_k recurrence.
    T0 = 1.0
    T1 = y
    total = float(coefs[0].mid()) * T0 + float(coefs[1].mid()) * T1
    Tm1, Tc = T0, T1
    for k in range(2, K + 1):
        Tn = 2.0 * y * Tc - Tm1
        total += float(coefs[k].mid()) * Tn
        Tm1, Tc = Tc, Tn
    return total


__all__ = [
    "omega_of",
    "jacobi_anger_chebyshev_coeffs",
    "jacobi_anger_truncation_infty",
    "kernel_truncation_error_bound",
    "JacobiAngerTables",
    "build_jacobi_anger_tables",
    "build_kernel_matrix_A",
    "arb_mat_to_numpy_mid",
    "evaluate_cheb_series_float",
]
