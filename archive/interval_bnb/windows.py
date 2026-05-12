"""Window matrix metadata for interval BnB.

Delegates matrix construction to `lasserre.core.build_window_matrices`
(the single source of truth) and precomputes the data structures the
bound routines consume: for each window W we cache
    - pairs[W]:  list of (i, j) int index pairs with M_W[i,j] != 0
                 (ordered pairs; M_W is symmetric so every off-diagonal
                  entry appears twice)
    - scale[W]:  the common nonzero value of M_W (= 2d/ell)
    - scale_q:   exact Fraction version for rigorous replay.

No matrix multiplication is ever performed against the dense M_W by
the bound routines -- they iterate over `pairs` directly, which avoids
O(d^2) work per window on every evaluation.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Tuple

import numpy as np

# Allow this package to be imported without adjusting PYTHONPATH.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lasserre.core import build_window_matrices  # noqa: E402


@dataclass
class WindowMeta:
    ell: int
    s_lo: int
    pairs: Tuple[Tuple[int, int], ...]  # (i, j) with M_W[i,j] != 0, i<=j
    pairs_all: Tuple[Tuple[int, int], ...]  # all ordered pairs (both orderings)
    scale: float
    scale_q: Fraction

    @property
    def name(self) -> str:
        return f"W(ell={self.ell},s={self.s_lo})"


def build_windows(d: int) -> List[WindowMeta]:
    """Return the list of windows for dimension d."""
    windows, _M = build_window_matrices(d)
    out: List[WindowMeta] = []
    for (ell, s_lo) in windows:
        # Pair support: i + j in [s_lo, s_lo + ell - 2], 0 <= i, j <= d-1.
        pairs_all: List[Tuple[int, int]] = []
        pairs: List[Tuple[int, int]] = []
        for i in range(d):
            for j in range(d):
                s = i + j
                if s_lo <= s <= s_lo + ell - 2:
                    pairs_all.append((i, j))
                    if i <= j:
                        pairs.append((i, j))
        scale = 2.0 * d / ell
        scale_q = Fraction(2 * d, ell)
        out.append(WindowMeta(
            ell=ell, s_lo=s_lo,
            pairs=tuple(pairs), pairs_all=tuple(pairs_all),
            scale=scale, scale_q=scale_q,
        ))
    return out


def windows_by_symmetric_s(d: int, windows: List[WindowMeta]) -> List[int]:
    """Return indices of windows likely to be binding for a symmetric
    minimiser -- used to order window evaluation best-first, so the
    tightest window is tried first and cuts the box fast.

    Heuristic: central windows (s_lo near d-1) of modest length tend to
    be binding, matching the observed behaviour of the cascade. The full
    list is always searched; this only reorders.
    """
    def key(w: WindowMeta) -> float:
        center = w.s_lo + (w.ell - 2) / 2
        # Distance of window centre from conv-length midpoint (d-1).
        return abs(center - (d - 1)) + 0.1 * w.ell
    order = sorted(range(len(windows)), key=lambda k: key(windows[k]))
    return order
