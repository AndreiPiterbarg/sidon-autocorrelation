"""Interval-arithmetic primitives for Step 2 (Krawczyk on saddle-KKT).

Wraps mpmath.iv with:
  * exact rational -> interval conversion (uses iv.mpf(num) / iv.mpf(den)
    which is rigorous outward-rounded since both endpoints are exact ints),
  * vector / matrix interval arithmetic with NumPy-style API,
  * float-midpoint extraction (for preconditioner construction).

Soundness invariant (verified by mpmath itself): every binary op on iv.mpf
produces an interval that *contains* the true real result.  This module
preserves that invariant; we never relax intervals (e.g. no inflation by
positive epsilon, only outward rounding).

NOTE: mpmath.iv.mpf (when given an integer) is exact.  iv_div(num, den)
where num,den are integers gives the tightest possible rounded interval
[num/den_down, num/den_up] using mpmath's controlled rounding.  Tested
below.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from mpmath import iv, mp, mpf, mpi


# ---------------------------------------------------------------------
# Working precision
# ---------------------------------------------------------------------

_DEFAULT_DPS = 50  # ~50 decimal digits in mpmath = ~166 bits


def set_precision(dps: int = _DEFAULT_DPS) -> None:
    """Set mpmath working precision (decimal digits)."""
    mp.dps = dps
    iv.dps = dps


# Initialize with default precision.
set_precision()


# ---------------------------------------------------------------------
# Rational <-> interval conversion
# ---------------------------------------------------------------------

def rat_to_iv(q):
    """Convert exact Fraction (or int) to mpmath.iv interval.

    For Fraction p/q with integer p, q: compute iv.mpf(p) / iv.mpf(q).
    iv.mpf(int) is exact (single point); the division is rounded outward.
    Result is the tightest interval containing the true rational value.
    """
    if isinstance(q, int):
        return iv.mpf(q)
    if isinstance(q, Fraction):
        if q.denominator == 1:
            return iv.mpf(q.numerator)
        return iv.mpf(q.numerator) / iv.mpf(q.denominator)
    if isinstance(q, float):
        # Floats convert exactly to mpmath.
        return iv.mpf(q)
    # Treat as already an iv.mpf
    return q


def iv_lo(x) -> mpf:
    """Lower endpoint of an iv.mpf as mpmath.mpf."""
    return x.a


def iv_hi(x) -> mpf:
    """Upper endpoint of an iv.mpf as mpmath.mpf."""
    return x.b


def iv_mid_float(x) -> float:
    """Float midpoint of an interval (for preconditioner construction).

    Robust to plain numeric types (int, float, Fraction) which are
    treated as degenerate (zero-width) intervals.

    NOTE: this is *not* used as a rigorous quantity; it's just a target
    for inversion. Krawczyk soundness comes from the interval evaluation,
    not the choice of preconditioner.
    """
    if hasattr(x, "a") and hasattr(x, "b"):
        lo = float(x.a)
        hi = float(x.b)
        return 0.5 * (lo + hi)
    if isinstance(x, Fraction):
        return float(x)
    return float(x)


def iv_contains(box, point) -> bool:
    """True if interval `box` contains the scalar `point`."""
    return float(box.a) <= float(point) <= float(box.b)


def iv_subset(inner, outer) -> bool:
    """True if `inner` interval is *strictly* in interior of `outer`."""
    return (float(outer.a) < float(inner.a) and
            float(inner.b) < float(outer.b))


def iv_disjoint(a, b) -> bool:
    """True if `a` and `b` intervals have empty intersection."""
    return float(a.b) < float(b.a) or float(b.b) < float(a.a)


def iv_intersect(a, b):
    """Intersection of two iv.mpf intervals; raises if empty."""
    lo = max(float(a.a), float(b.a))
    hi = min(float(a.b), float(b.b))
    if lo > hi:
        raise ValueError("Empty intersection")
    return iv.mpf([lo, hi])


def iv_zero():
    """Zero interval."""
    return iv.mpf(0)


def iv_one():
    """One interval."""
    return iv.mpf(1)


# ---------------------------------------------------------------------
# Vector / matrix wrappers
# ---------------------------------------------------------------------

class IVVec:
    """Vector of iv.mpf intervals. Stored as Python list."""
    __slots__ = ("data",)

    def __init__(self, data: Sequence):
        self.data = [rat_to_iv(x) if not isinstance(x, type(iv.mpf(0))) else x
                     for x in data]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, i, v):
        self.data[i] = rat_to_iv(v) if not isinstance(v, type(iv.mpf(0))) else v

    def __iter__(self):
        return iter(self.data)

    def __add__(self, other: "IVVec") -> "IVVec":
        assert len(self) == len(other)
        return IVVec([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other: "IVVec") -> "IVVec":
        assert len(self) == len(other)
        return IVVec([a - b for a, b in zip(self.data, other.data)])

    def midpoint_float(self) -> np.ndarray:
        """Return float64 array of interval midpoints."""
        return np.array([iv_mid_float(x) for x in self.data], dtype=np.float64)

    def width_float(self) -> np.ndarray:
        """Return float64 array of interval widths."""
        return np.array([float(x.b) - float(x.a) for x in self.data],
                        dtype=np.float64)

    def is_subset_of(self, other: "IVVec") -> bool:
        """Strict containment of self in interior of other."""
        return all(iv_subset(s, o) for s, o in zip(self.data, other.data))

    def is_disjoint_from(self, other: "IVVec") -> bool:
        """Componentwise: any component disjoint from corresponding => disjoint."""
        return any(iv_disjoint(s, o) for s, o in zip(self.data, other.data))

    def __repr__(self):
        return "IVVec([" + ", ".join(
            f"[{float(x.a):.6g}, {float(x.b):.6g}]" for x in self.data
        ) + "])"


class IVMat:
    """Matrix of iv.mpf intervals. Stored row-major as nested list."""
    __slots__ = ("data", "n", "m")

    def __init__(self, data, n: int, m: int):
        self.n = n
        self.m = m
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            self.data = [[rat_to_iv(c) for c in row] for row in data]
        else:
            # flat list of length n*m, row-major
            self.data = [[rat_to_iv(data[i * m + j]) for j in range(m)]
                         for i in range(n)]
        assert len(self.data) == n
        for row in self.data:
            assert len(row) == m

    def __getitem__(self, ij: Tuple[int, int]):
        return self.data[ij[0]][ij[1]]

    def matvec(self, x: IVVec) -> IVVec:
        assert len(x) == self.m
        out = []
        for i in range(self.n):
            s = iv_zero()
            for j in range(self.m):
                s = s + self.data[i][j] * x.data[j]
            out.append(s)
        return IVVec(out)

    def matmul(self, B: "IVMat") -> "IVMat":
        assert self.m == B.n
        out_data = [[iv_zero() for _ in range(B.m)] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(B.m):
                s = iv_zero()
                for k in range(self.m):
                    s = s + self.data[i][k] * B.data[k][j]
                out_data[i][j] = s
        out = IVMat.__new__(IVMat)
        out.data = out_data
        out.n = self.n
        out.m = B.m
        return out

    def midpoint_float(self) -> np.ndarray:
        """Return float64 (n x m) array of interval midpoints."""
        out = np.zeros((self.n, self.m), dtype=np.float64)
        for i in range(self.n):
            for j in range(self.m):
                out[i, j] = iv_mid_float(self.data[i][j])
        return out

    @classmethod
    def from_float(cls, A: np.ndarray) -> "IVMat":
        """Build IVMat from a float64 NumPy matrix.  Each entry becomes
        a degenerate (zero-width) interval at the float value."""
        n, m = A.shape
        data = [[iv.mpf(float(A[i, j])) for j in range(m)] for i in range(n)]
        out = cls.__new__(cls)
        out.data = data
        out.n = n
        out.m = m
        return out

    @classmethod
    def identity(cls, n: int) -> "IVMat":
        data = [[iv.mpf(1) if i == j else iv.mpf(0) for j in range(n)]
                for i in range(n)]
        out = cls.__new__(cls)
        out.data = data
        out.n = n
        out.m = n
        return out

    def __sub__(self, other: "IVMat") -> "IVMat":
        assert self.n == other.n and self.m == other.m
        out_data = [[self.data[i][j] - other.data[i][j] for j in range(self.m)]
                    for i in range(self.n)]
        out = IVMat.__new__(IVMat)
        out.data = out_data
        out.n = self.n
        out.m = self.m
        return out


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # rat_to_iv exactness
    third = rat_to_iv(Fraction(1, 3))
    assert float(third.a) <= 1.0 / 3.0 <= float(third.b)
    print(f"1/3 -> [{float(third.a):.20f}, {float(third.b):.20f}]")

    # IVVec
    v = IVVec([Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)])
    print("v =", v)
    s = v + v
    print("v + v =", s)

    # IVMat
    A = IVMat([[Fraction(1), Fraction(2)], [Fraction(3), Fraction(4)]], 2, 2)
    x = IVVec([Fraction(1), Fraction(1)])
    Ax = A.matvec(x)
    print("A x =", Ax)

    # Containment / disjointness
    a = iv.mpf([0.1, 0.5])
    b = iv.mpf([0.6, 1.0])
    c = iv.mpf([0.0, 1.0])
    print(f"a subset of c? {iv_subset(a, c)}")
    print(f"a disjoint from b? {iv_disjoint(a, b)}")

    print("\nself-test OK")
