"""Hausdorff moment PSD blocks in the standard Chebyshev basis.

We parametrize f in A = { f >= 0, supp f subset [-1/4, 1/4], int f = 1 }
by its Chebyshev moments on [-1/4, 1/4]:

    c_k = int_{-1/4}^{1/4} T_k(4x) f(x) dx,     k = 0, 1, ..., 2N,

where T_k is the standard Chebyshev polynomial of the first kind on [-1, 1].

Under the substitution y = 4x, the pushforward measure tilde_nu on [-1, 1]
has density tilde_f(y) = f(y/4) / 4 and satisfies

    c_k = int_{-1}^{1} T_k(y) d tilde_nu(y),
    tilde_m_k := int y^k d tilde_nu(y).

Forward change of basis (integer, lower triangular):
    T_k(y) = sum_{j=0}^{k} t[k, j] y^j         =>   c = T @ tilde_m.

Inverse change of basis (rational, lower triangular, dyadic denominators):
    y^k = sum_{j=0}^{k} u[k, j] T_j(y)          =>   tilde_m = U @ c,   U = T^{-1}.

Hausdorff support = [-1, 1] in y iff there exist nonneg PSD matrices H_1, H_2:
    H_1 = [tilde_m_{i+j}]_{i, j in 0..N}                   ((N+1) x (N+1))
    H_2 = [tilde_m_{i+j} - tilde_m_{i+j+2}]_{i, j in 0..N-1}  (N x N, localizer 1 - y^2)

Equivalently, f >= 0 on [-1/4, 1/4] iff H_1(c) >= 0 and H_2(c) >= 0, where
both H_i are linear in c_0, ..., c_{2N}.

(Spec note: the top-level prompt states both blocks are (N+1) x (N+1); that
is inconsistent with degree-2N moments under the (1 - y^2) localizer.  We
follow the canonical sizing H_1 ~ (N+1)x(N+1), H_2 ~ NxN.  The mass
normalization c_0 = 1 is enforced separately by the SDP.)

All arithmetic uses python-flint's fmpq / fmpq_mat for exactness; floats
are produced only for diagnostic PSD checks in tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from flint import fmpq, fmpq_mat, fmpz, fmpz_mat


# ---------------------------------------------------------------------
# 1-D Chebyshev <-> monomial change of basis
# ---------------------------------------------------------------------

def monomial_to_cheb_matrix(N: int) -> fmpz_mat:
    """Return the (N+1) x (N+1) integer matrix T with T[k, j] such that

        T_k(y) = sum_{j=0}^{k} T[k, j] y^j.

    Built from the recurrence T_{k+1}(y) = 2 y T_k(y) - T_{k-1}(y)
    with T_0 = 1, T_1 = y.  Applied dually,

        c_k = int T_k d tilde_nu = sum_j T[k, j] tilde_m_j,   i.e. c = T @ tilde_m.
    """
    if N < 0:
        raise ValueError("N must be >= 0")
    M = fmpz_mat(N + 1, N + 1)  # zero-initialized
    M[0, 0] = fmpz(1)
    if N >= 1:
        M[1, 1] = fmpz(1)
    for k in range(1, N):
        # T_{k+1, j} = 2 T_{k, j-1} - T_{k-1, j}
        for j in range(k + 2):
            left = M[k, j - 1] if j - 1 >= 0 else fmpz(0)
            right = M[k - 1, j] if j <= k - 1 else fmpz(0)
            M[k + 1, j] = fmpz(2) * left - right
    return M


def cheb_to_monomial_matrix(N: int) -> fmpq_mat:
    """Return the (N+1) x (N+1) rational matrix U with U[k, j] such that

        y^k = sum_{j=0}^{k} U[k, j] T_j(y).

    Equivalently U = T^{-1} for T from `monomial_to_cheb_matrix`.  Used to
    map Chebyshev moments c to monomial moments tilde_m via

        tilde_m = U @ c.

    Recurrence (derived from y T_j = (T_{j+1} + T_{j-1})/2 for j >= 1 and
    y T_0 = T_1):
        U[0, 0] = 1,  U[1, 1] = 1,
        U[k+1, 0] = U[k, 1] / 2,
        U[k+1, 1] = U[k, 0] + U[k, 2] / 2,
        U[k+1, m] = (U[k, m-1] + U[k, m+1]) / 2       for m >= 2.
    Denominators grow as 2^{k-1} for row k, kept exact by flint.
    """
    if N < 0:
        raise ValueError("N must be >= 0")
    U = fmpq_mat(N + 1, N + 1)
    U[0, 0] = fmpq(1)
    if N >= 1:
        U[1, 1] = fmpq(1)
    half = fmpq(1, 2)
    for k in range(1, N):
        # Row k+1 from row k.
        # m = 0
        U[k + 1, 0] = U[k, 1] * half
        # m = 1 (requires k >= 1, which we have)
        c_k0 = U[k, 0]
        c_k2 = U[k, 2] if 2 <= N else fmpq(0)
        U[k + 1, 1] = c_k0 + c_k2 * half
        # m >= 2
        for m in range(2, N + 1):
            left = U[k, m - 1]
            right = U[k, m + 1] if m + 1 <= N else fmpq(0)
            U[k + 1, m] = (left + right) * half
    return U


# ---------------------------------------------------------------------
# Hausdorff PSD blocks: linear maps c -> H_i(c) in exact rationals
# ---------------------------------------------------------------------

@dataclass
class HankelBlockSpec:
    """Linear map c -> H(c) for c in fmpq^{2N+1}, H in fmpq^{n x n}.

    Stored as a list of per-coordinate matrices `per_k[k]` such that

        H(c) = sum_{k=0}^{2N} c_k * per_k[k].

    `per_k[k]` is an (n x n) fmpq_mat with rational entries coming from
    the Chebyshev-to-monomial change of basis.  The SDP driver iterates
    `per_k` to assemble the linear PSD constraints; certification code
    calls `.apply(c_star)` on a rounded rational c_star.
    """
    per_k: List[fmpq_mat]
    name: str = ""

    @property
    def size(self) -> int:
        return self.per_k[0].nrows()

    @property
    def degree(self) -> int:
        return len(self.per_k) - 1

    def apply(self, c: Sequence) -> fmpq_mat:
        """Assemble H(c) at the given coefficient vector c.

        Accepts any sequence of fmpq-compatible scalars (fmpq, Fraction, int).
        """
        if len(c) != len(self.per_k):
            raise ValueError(
                f"{self.name or 'HankelBlockSpec'}: expected |c| = "
                f"{len(self.per_k)}, got {len(c)}")
        n = self.size
        H = fmpq_mat(n, n)
        for k, ck in enumerate(c):
            if ck == 0:
                continue
            scalar = ck if isinstance(ck, fmpq) else fmpq(ck)
            H = H + self.per_k[k] * scalar
        return H

    def apply_float(self, c: Sequence[float]) -> np.ndarray:
        """Float version of `apply` for numeric diagnostics."""
        n = self.size
        # Materialize per_k as float arrays once per call (small n).
        H = np.zeros((n, n), dtype=np.float64)
        for k, ck in enumerate(c):
            if ck == 0.0:
                continue
            Mk = fmpq_mat_to_numpy(self.per_k[k])
            H += float(ck) * Mk
        return H


def hausdorff_hankel_blocks_chebyshev(
    N: int,
    support: Optional[Tuple[Fraction, Fraction]] = None,
) -> Tuple[HankelBlockSpec, HankelBlockSpec]:
    """Return (H1_spec, H2_spec) for the Chebyshev-moment Hausdorff cone.

    Inputs:
      N       -- relaxation order.  H_1 is (N+1) x (N+1), H_2 is N x N.
                 The coefficient vector c has length 2N + 1.
      support -- Expected to be (-1/4, 1/4) (validated if given).
                 The Hausdorff conditions live on the scaled variable
                 y = 4x on [-1, 1]; the same c serves either parametrization.

    Returns two `HankelBlockSpec` objects:
      H1_spec:  H_1(c) = [tilde_m_{i+j}]_{i, j in 0..N}.
      H2_spec:  H_2(c) = [tilde_m_{i+j} - tilde_m_{i+j+2}]_{i, j in 0..N-1}
                representing the (1 - y^2) >= 0 localizer.

    Both specs are sparse-free linear maps in c (every coordinate can
    participate); they are exact over fmpq.
    """
    if N < 0:
        raise ValueError("N must be >= 0")
    if support is not None:
        a, b = support
        if (Fraction(a), Fraction(b)) != (Fraction(-1, 4), Fraction(1, 4)):
            raise NotImplementedError(
                f"support = {support}: only [-1/4, 1/4] is implemented. "
                f"For other supports, rescale externally via y = (2x - (a+b))/(b-a).")

    deg = 2 * N  # highest Chebyshev index we need
    U = cheb_to_monomial_matrix(deg)  # (2N+1) x (2N+1); tilde_m = U c

    # H1: per-k matrix A1_k has (A1_k)[i, j] = U[i + j, k] for i, j in 0..N.
    A1_list: List[fmpq_mat] = []
    for k in range(deg + 1):
        Mk = fmpq_mat(N + 1, N + 1)
        for i in range(N + 1):
            for j in range(N + 1):
                Mk[i, j] = U[i + j, k]
        A1_list.append(Mk)

    # H2: (A2_k)[i, j] = U[i + j, k] - U[i + j + 2, k] for i, j in 0..N-1.
    A2_list: List[fmpq_mat] = []
    # H2 has size N; for N == 0 it is empty, which makes the 1-y^2 localizer trivial.
    h2_size = N
    for k in range(deg + 1):
        Mk = fmpq_mat(h2_size, h2_size)
        for i in range(h2_size):
            for j in range(h2_size):
                Mk[i, j] = U[i + j, k] - U[i + j + 2, k]
        A2_list.append(Mk)

    return (
        HankelBlockSpec(per_k=A1_list, name="H1 (Chebyshev Hausdorff, full)"),
        HankelBlockSpec(per_k=A2_list, name="H2 (Chebyshev Hausdorff, 1-y^2 localizer)"),
    )


# ---------------------------------------------------------------------
# Numeric helpers (for tests and diagnostics; not in the cert path)
# ---------------------------------------------------------------------

def _fmpq_to_float(q: fmpq) -> float:
    """python-flint 0.8 does not implement float(fmpq) directly; go via p / q."""
    return float(int(q.p)) / float(int(q.q))


def fmpq_mat_to_numpy(M: fmpq_mat) -> np.ndarray:
    """Convert an fmpq_mat to float64 numpy (for eigenvalue / PSD diagnostics)."""
    r, c = M.nrows(), M.ncols()
    out = np.zeros((r, c), dtype=np.float64)
    for i in range(r):
        for j in range(c):
            out[i, j] = _fmpq_to_float(M[i, j])
    return out


def chebyshev_moments_of_density(
    f_func: Callable[[float], float],
    N: int,
    a: float = -0.25,
    b: float = 0.25,
    *,
    quad_kwargs: Optional[dict] = None,
) -> List[float]:
    """Compute c_k = int_a^b T_k(2(x - m)/(b-a)) f(x) dx numerically for
    k = 0 .. N, where m = (a + b)/2.  For the default [a, b] = [-1/4, 1/4]
    the argument reduces to T_k(4x).

    Returned list has length N + 1 and is intended for generating test
    moment vectors; it is not used in the cert path.
    """
    from scipy import integrate
    mid = 0.5 * (a + b)
    scale = 2.0 / (b - a)
    qk = dict(epsabs=1e-13, epsrel=1e-13, limit=200)
    if quad_kwargs:
        qk.update(quad_kwargs)

    def _T(k: int, y: float) -> float:
        if k == 0:
            return 1.0
        if k == 1:
            return y
        Tm1, T = 1.0, y
        for _ in range(2, k + 1):
            Tm1, T = T, 2.0 * y * T - Tm1
        return T

    out: List[float] = []
    for k in range(N + 1):
        val, _ = integrate.quad(
            lambda x, _k=k: _T(_k, scale * (x - mid)) * f_func(x), a, b, **qk
        )
        out.append(val)
    return out


__all__ = [
    "monomial_to_cheb_matrix",
    "cheb_to_monomial_matrix",
    "HankelBlockSpec",
    "hausdorff_hankel_blocks_chebyshev",
    "fmpq_mat_to_numpy",
    "chebyshev_moments_of_density",
]
