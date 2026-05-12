"""Box representation for the (d-1)-simplex Delta_d.

A box B is an axis-aligned hyperrectangle [lo, hi] in R^d, INTERSECTED
with the closed simplex {mu : sum mu = 1, mu >= 0} and (optionally)
the single half-simplex symmetry cut {mu_0 <= mu_{d-1}} via
`Box.initial(..., sym_cuts=[(0, d-1)])`.

Each Box stores BOTH float64 numpy arrays (`lo`, `hi`) AND Python
shared-denominator integers (`lo_int`, `hi_int`) at denom 2**D_SHIFT
(D_SHIFT = 60). The integer form is the rigor-bearing representation:
all certified bounds (`bound_*_int_ge`) consume `lo_int/hi_int` and
operate in exact integer arithmetic. The float form drives the fast
batched bound evaluation (numpy BLAS) and is non-soundness-critical.

The BnB starts from the initial box [0, 1/2] x [0, 1]^{d-1} (with the
single sym cut) or [0, 1]^d (without). Splits bisect along an axis at
the integer midpoint `(lo_int + hi_int) >> 1`; `Box.split` raises
`RuntimeError` if the dyadic-2^60 grid is exhausted (which would
otherwise round silently and break soundness).

`to_fractions()` returns `Fraction(v, SCALE)` per endpoint — exact and
reproducible. It refuses if integer metadata is missing. `to_ints()`
returns `(lo_int, hi_int)`; the float-fallback rounds outward (floor
for `lo_int`, ceil for `hi_int`) to preserve box CONTAINMENT.

Simplex intersection check uses exact integer sums:
    B intersects Delta_d  <=>  sum(lo_int) <= 2^60  AND  sum(hi_int) >= 2^60.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import List, Optional, Tuple

import numpy as np


# Shared denominator for exact integer rigor arithmetic.
# Every endpoint we ever visit is a dyadic rational; the smallest
# representable positive value is 2^-D_SHIFT, so any tree depth up to
# D_SHIFT is safely representable.  D_SHIFT=60 is 18x the practical
# depth-limit implied by min_box_width = 1e-10 ≈ 2^-33, and products of
# two ints at scale 2^60 still fit in Python's arbitrary-precision int
# without slowdown.
D_SHIFT: int = 60
SCALE: int = 1 << D_SHIFT        # 2**60
SCALE2: int = SCALE * SCALE      # 2**120


class Box:
    """Box endpoints as float64 PLUS shared-denominator integers.

    Floats drive the fast-path bound evaluation (numpy BLAS). Integers
    drive the rigor replay: `lo_int[i]` is `lo[i] * 2**D_SHIFT` as a
    Python int (exact because every endpoint is a dyadic rational).
    Splits propagate both representations with O(1) work per split.

    The Fraction accessor `to_fractions()` is still available (used by
    the bound-eval helpers) but rigor-replay uses the int form directly.
    """

    __slots__ = ("lo", "hi", "lo_int", "hi_int")

    def __init__(
        self, lo: np.ndarray, hi: np.ndarray,
        lo_int: Optional[List[int]] = None,
        hi_int: Optional[List[int]] = None,
    ):
        self.lo = lo
        self.hi = hi
        self.lo_int = lo_int
        self.hi_int = hi_int

    @property
    def d(self) -> int:
        return self.lo.shape[0]

    @classmethod
    def initial(cls, d: int, sym_cuts: Optional[List[Tuple[int, int]]] = None) -> "Box":
        lo = np.zeros(d, dtype=np.float64)
        hi = np.ones(d, dtype=np.float64)
        lo_int = [0] * d
        hi_int = [SCALE] * d  # 1.0 == 2**60
        if sym_cuts:
            half = SCALE >> 1  # 2**59 == 0.5
            for i, _ in sym_cuts:
                # Enforce mu_i <= 0.5, which is implied by mu_i <= mu_{d-1-i}
                # and mu on the simplex (mu_i + mu_{d-1-i} <= 1).
                # Certifying this relaxation is stronger than certifying H_d,
                # so the resulting lower bound on val(d) remains sound.
                if hi[i] > 0.5:
                    hi[i] = 0.5
                    hi_int[i] = half
        return cls(lo, hi, lo_int, hi_int)

    def intersects_simplex(self) -> bool:
        """Exact check: does the box intersect {sum mu = 1, mu >= 0}?

        Uses integer sums at denominator 2**D_SHIFT so the comparison
        is rigorous even at deep splits where float64 sums of d dyadic
        rationals can lose up to a few bits.
        """
        if self.lo_int is not None and self.hi_int is not None:
            return sum(self.lo_int) <= SCALE and sum(self.hi_int) >= SCALE
        # Fallback to float when int metadata is absent (legacy Box
        # construction). Float-path remains sound at moderate depth.
        return float(self.lo.sum()) <= 1.0 and float(self.hi.sum()) >= 1.0

    def widest_axis(self) -> int:
        return int(np.argmax(self.hi - self.lo))

    def widest_splittable_axis(self) -> int:
        """Widest axis with integer width >= 2 (i.e. still splittable
        without exhausting the D_SHIFT-bit dyadic grid). Returns -1 if
        no axis is splittable.

        Falls back to `widest_axis()` if integer metadata is missing.
        """
        if self.lo_int is None or self.hi_int is None:
            return self.widest_axis()
        widths = self.hi - self.lo
        best_axis = -1
        best_width = -1.0
        for i in range(self.d):
            if (self.hi_int[i] - self.lo_int[i]) >= 2:
                w = float(widths[i])
                if w > best_width:
                    best_width = w
                    best_axis = i
        return best_axis

    def max_width(self) -> float:
        return float((self.hi - self.lo).max())

    def volume(self) -> float:
        """Hyperrectangle volume (product of axis widths) in R^d.

        NOTE: this is NOT vol(B ∩ Delta_d) — it ignores the simplex
        constraint and is therefore an UPPER bound on the simplex-feasible
        volume. Used in the parallel driver for ETA / progress display
        only; the soundness invariant is `in_flight == 0 AND not
        failed_event`, NOT `coverage_fraction == 1`.
        """
        return float(np.prod(self.hi - self.lo))

    def tighten_to_simplex(self) -> bool:
        """Shrink lo/hi using the simplex constraint sum(mu) = 1.

        For any mu on the simplex with lo <= mu <= hi,
            mu_i = 1 - sum_{j != i} mu_j
        lies in [1 - (hi_sum - hi_i), 1 - (lo_sum - lo_i)].
        So we can set
            new_hi[i] = min(hi[i], 1 - (lo_sum - lo[i]))
            new_lo[i] = max(lo[i], 1 - (hi_sum - hi[i])).
        The tightened box has the SAME simplex intersection as the
        original, so every bound on mu^T M_W mu computed over the
        tightened box is still a valid bound over the original box
        intersect simplex.

        Both float endpoints AND the exact integer endpoints are
        updated in lockstep so the rigor replay sees identical
        arithmetic.

        Returns True if any axis was tightened.
        """
        if self.lo_int is None or self.hi_int is None:
            return False
        lo_sum_i = sum(self.lo_int)
        hi_sum_i = sum(self.hi_int)
        tightened = False
        for i in range(len(self.lo_int)):
            # new_hi_int[i] = min(hi_int[i], SCALE - (lo_sum - lo_int[i]))
            bound_hi_i = SCALE - (lo_sum_i - self.lo_int[i])
            if bound_hi_i < self.hi_int[i]:
                self.hi_int[i] = bound_hi_i
                self.hi[i] = float(bound_hi_i) / SCALE
                # recompute hi_sum incrementally.
                hi_sum_i = sum(self.hi_int)
                tightened = True
            # new_lo_int[i] = max(lo_int[i], SCALE - (hi_sum - hi_int[i]))
            bound_lo_i = SCALE - (hi_sum_i - self.hi_int[i])
            if bound_lo_i > self.lo_int[i]:
                self.lo_int[i] = bound_lo_i
                self.lo[i] = float(bound_lo_i) / SCALE
                lo_sum_i = sum(self.lo_int)
                tightened = True
        return tightened

    def split(self, axis: int) -> Tuple["Box", "Box"]:
        mid = 0.5 * (self.lo[axis] + self.hi[axis])
        left_hi = self.hi.copy(); left_hi[axis] = mid
        right_lo = self.lo.copy(); right_lo[axis] = mid
        lo_i = self.lo_int; hi_i = self.hi_int
        if lo_i is not None and hi_i is not None:
            # Rigor guard: the integer midpoint (lo_i + hi_i) >> 1 is
            # exact iff (lo_i + hi_i) is even, i.e. we have not exhausted
            # the D_SHIFT-bit dyadic denominator. If this fails, the >> 1
            # would silently round and break the soundness of split.
            if ((lo_i[axis] + hi_i[axis]) & 1) != 0:
                raise RuntimeError(
                    "Box depth exceeded D_SHIFT bits — cannot split "
                    "dyadic box exactly."
                )
            mid_int = (lo_i[axis] + hi_i[axis]) >> 1
            left_hi_int = list(hi_i); left_hi_int[axis] = mid_int
            right_lo_int = list(lo_i); right_lo_int[axis] = mid_int
            left = Box(self.lo.copy(), left_hi, list(lo_i), left_hi_int)
            right = Box(right_lo, self.hi.copy(), right_lo_int, list(hi_i))
        else:
            left = Box(self.lo.copy(), left_hi)
            right = Box(right_lo, self.hi.copy())
        return left, right

    def to_fractions(self) -> Tuple[List[Fraction], List[Fraction]]:
        """Fraction endpoints (lazy). Converts from int form when
        available (fast), else from float64."""
        # Rigor hardening: refuse to silently fall back to the float
        # branch. Every Box used on the rigor path must carry integer
        # metadata so that to_fractions is exact and reproducible.
        if self.lo_int is None or self.hi_int is None:
            raise RuntimeError(
                "to_fractions called on a Box without integer metadata "
                "— rigor would be compromised"
            )
        lo_q = [Fraction(v, SCALE) for v in self.lo_int]
        hi_q = [Fraction(v, SCALE) for v in self.hi_int]
        return lo_q, hi_q

    def to_ints(self) -> Tuple[List[int], List[int]]:
        """Integer endpoints with denominator 2**D_SHIFT. Materialises
        from float if not cached (shouldn't happen in normal BnB use).

        SOUNDNESS: the float fallback uses outward rounding (floor for
        `lo_int`, ceil for `hi_int`) so the integer box CONTAINS the
        float box. Inward rounding (round-to-nearest) would silently
        produce an integer box that is a strict SUBSET of the float
        box — rigor would not transfer to bound_*_int_ge consumers.
        """
        if self.lo_int is None or self.hi_int is None:
            self.lo_int = [math.floor(float(x) * SCALE) for x in self.lo]
            self.hi_int = [math.ceil(float(x) * SCALE) for x in self.hi]
        return self.lo_int, self.hi_int

    def shape_summary(self) -> str:
        widths = self.hi - self.lo
        return f"max_w={float(widths.max()):.3e}  min_w={float(widths.min()):.3e}"

    def __repr__(self) -> str:
        return f"Box(d={self.d}, {self.shape_summary()})"
