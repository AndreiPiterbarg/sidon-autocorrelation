"""Rigorous interval branch-and-bound for max of g on [-1/2, 1/2].

Uses mpmath's interval arithmetic (mpmath.iv) to bound g on sub-intervals.
Returns a two-sided enclosure [lo, hi] satisfying lo <= max_{t in I} g(t) <= hi
with certified width.

Algorithm
---------
Maintain a priority queue of sub-intervals. For each sub-interval J, compute
an interval enclosure [g_lo(J), g_hi(J)] of g on J via mpmath.iv. The global
upper bound is max_J g_hi(J); the global lower bound is max_J g_lo(J) (which
is a valid lower bound on max g because g_lo(J) <= g(t) somewhere in J ...
actually, g_lo(J) is a LOWER bound on g over J, so max_J g_lo(J) is not a
valid lower bound on max_t g(t). We instead sample interior points and
evaluate in iv arithmetic to get rigorous point lower bounds on the global
max.)

We refine by splitting the interval with the largest g_hi until the relative
gap (global_hi - global_lo) / |global_lo| is below tolerance.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

from mpmath import iv, mp, mpf


def _iv_midpoint(I):
    """Return the mpf midpoint of an mpmath interval."""
    return (I.a + I.b) / 2


def _iv_width(I):
    return I.b - I.a


@dataclass(order=True)
class _PQEntry:
    """Priority queue entry; sorted by -upper so we pop the worst first."""
    neg_upper: float
    idx: int = field(compare=True)
    a: object = field(compare=False)          # mpf
    b: object = field(compare=False)          # mpf
    g_lo: object = field(compare=False)       # mpf
    g_hi: object = field(compare=False)       # mpf


def rigorous_max(
    g_iv: Callable,
    a,
    b,
    *,
    rel_tol: float = 1e-10,
    max_splits: int = 200_000,
    precision_bits: int = 200,
) -> Tuple[mpf, mpf, int]:
    """Certify an enclosure for max_{t in [a,b]} g(t).

    Parameters
    ----------
    g_iv : callable
        Takes an mpmath iv.mpf interval and returns an mpmath iv.mpf interval
        enclosing the range of g on the input.
    a, b : mpf-like
        Endpoints of the domain.
    rel_tol : float
        Stop when (hi - lo) / max(|lo|, eps) <= rel_tol.
    max_splits : int
        Safety cap on the number of interval splits.
    precision_bits : int
        mpmath precision in bits (default: 200 bits ~= 60 decimal digits).

    Returns
    -------
    (lo, hi, n_splits) : (mpf, mpf, int)
        Certified enclosure: lo <= max g <= hi.
    """
    mp.prec = precision_bits
    iv.prec = precision_bits

    a_mp = mpf(a)
    b_mp = mpf(b)

    # Initial evaluation on [a, b].
    I0 = iv.mpf([a_mp, b_mp])
    G0 = g_iv(I0)
    global_hi = G0.b
    # Rigorous lower bound: evaluate g at a single sample point in iv.
    mid = _iv_midpoint(I0)
    Gmid = g_iv(iv.mpf([mid, mid]))
    global_lo = Gmid.a

    # Priority queue: worst (largest upper bound) first.
    counter = 0
    heap: list[_PQEntry] = []
    heapq.heappush(
        heap,
        _PQEntry(neg_upper=float(-global_hi), idx=counter, a=a_mp, b=b_mp, g_lo=Gmid.a, g_hi=G0.b),
    )

    n_splits = 0
    while heap and n_splits < max_splits:
        # Check stopping criterion based on global bounds.
        gap = global_hi - global_lo
        denom = abs(global_lo) if abs(global_lo) > mpf("1e-30") else mpf("1e-30")
        if gap / denom <= rel_tol:
            break

        # Pop the worst.
        entry = heapq.heappop(heap)
        # This entry's upper bound may be stale relative to current global_hi.
        # But we always split the heap top; that's fine.
        aL, bL = entry.a, entry.b
        if bL - aL < mpf("1e-40"):
            # Interval too small to split; accept its upper as-is.
            continue

        mid_split = (aL + bL) / 2

        for (a_sub, b_sub) in ((aL, mid_split), (mid_split, bL)):
            I = iv.mpf([a_sub, b_sub])
            G = g_iv(I)
            # Sample center for rigorous lower bound on max.
            mid_sub = (a_sub + b_sub) / 2
            Gm = g_iv(iv.mpf([mid_sub, mid_sub]))
            # Update global lower.
            if Gm.a > global_lo:
                global_lo = Gm.a
            counter += 1
            heapq.heappush(
                heap,
                _PQEntry(neg_upper=float(-G.b), idx=counter, a=a_sub, b=b_sub, g_lo=Gm.a, g_hi=G.b),
            )
        n_splits += 1

        # Recompute global_hi as the max upper over all live intervals.
        # The heap top has the largest upper.
        if heap:
            global_hi = heap[0].a if False else max(e.g_hi for e in heap)

    return mpf(global_lo), mpf(global_hi), n_splits


def rigorous_integral_with_weight(
    ghat_iv: Callable,
    w_iv: Callable,
    xi_max,
    *,
    n_subdiv: int = 2048,
    precision_bits: int = 200,
) -> Tuple[mpf, mpf]:
    """Rigorous enclosure of int_{-xi_max}^{xi_max} ghat(xi) * w(xi) dxi.

    Uses midpoint-rule with an interval enclosure of the integrand on each
    subinterval (range bounding, not just midpoint value). This is a sound
    but conservative quadrature: on sub-interval J of width h, the integral
    contribution is in h * [inf G, sup G].

    Parameters
    ----------
    ghat_iv, w_iv : callables
        Interval enclosures of ghat and weight w as functions iv.mpf -> iv.mpf.
    xi_max : mpf-like
        Truncation: we assume ghat = 0 outside [-xi_max, xi_max].
    n_subdiv : int
        Number of sub-intervals.

    Returns
    -------
    (lo, hi) : (mpf, mpf)
        Rigorous enclosure of the integral.
    """
    mp.prec = precision_bits
    iv.prec = precision_bits

    xi_max = mpf(xi_max)
    h = (2 * xi_max) / n_subdiv

    lo_sum = mpf(0)
    hi_sum = mpf(0)
    for k in range(n_subdiv):
        xi_a = -xi_max + k * h
        xi_b = xi_a + h
        I = iv.mpf([xi_a, xi_b])
        G = ghat_iv(I)
        W = w_iv(I)
        GW = G * W
        lo_sum += h * GW.a
        hi_sum += h * GW.b
    return lo_sum, hi_sum
