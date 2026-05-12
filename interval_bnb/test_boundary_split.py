"""Tests for the boundary-concentrated split-axis heuristic.

The heuristic targets the d>=22 BnB stall mode in which many axes are
pinned at lo[i]<=1e-12 (the mu_i==0 face of the simplex). For such
boxes, splitting a "boundary" axis is wasted work — the LP gap is
dominated by (free_axis, free_axis) pair products, which only shrink
when FREE axes are split.

These tests verify:

  * Soundness: chosen axis is in [0, d) and `splittable[axis]` is True.
  * Detection: when >= boundary_axis_count_threshold axes are at
    lo<=1e-12, the heuristic prefers a free axis.
  * Fallback: when 0 axes are on boundary (or none meet threshold), the
    heuristic returns None so the caller's existing widest-axis path
    handles the box (no behavior change).
  * Disabled-by-default: with depth < boundary_split_depth_threshold
    (i.e. 999 default), the heuristic returns None.
  * Splittable mask: when the only free axes have int_width <= 1, the
    heuristic returns None (caller falls back to widest splittable).

This mirrors `test_lp_split.py`'s style of unit-testing the chooser
logic without having to spin up the full multiprocessing harness.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _boundary_choose_axis(
    lo: np.ndarray,
    hi: np.ndarray,
    splittable: np.ndarray,
    depth: int,
    boundary_split_depth_threshold: int,
    boundary_axis_count_threshold: int,
) -> int | None:
    """Mirror of the boundary-concentrated chooser logic in parallel.py.

    Returns the chosen axis (a free, splittable, widest axis), or None
    when the heuristic does not fire (depth gate not reached, too few
    boundary axes, or no free splittable axis).
    """
    if depth < boundary_split_depth_threshold:
        return None
    boundary_mask = lo <= 1e-12
    n_boundary = int(np.count_nonzero(boundary_mask))
    if n_boundary < boundary_axis_count_threshold:
        return None
    free_mask = ~boundary_mask
    free_splittable = free_mask & splittable
    if not free_splittable.any():
        return None
    widths = hi - lo
    ws_free = np.where(free_splittable, widths, -np.inf)
    cand = int(np.argmax(ws_free))
    if not np.isfinite(ws_free[cand]):
        return None
    return cand


def test_boundary_axis_detection():
    """At d=22 with 18 axes at lo=0 and 4 free axes, the heuristic
    identifies a free axis and picks the widest of them."""
    d = 22
    lo = np.zeros(d, dtype=np.float64)
    hi = np.full(d, 0.05, dtype=np.float64)
    # Mark 4 axes as "free" with positive lo and varied widths.
    free_axes = [3, 7, 12, 19]
    free_widths = [0.10, 0.20, 0.15, 0.25]  # axis 19 widest
    for ax, w in zip(free_axes, free_widths):
        lo[ax] = 0.01  # > 1e-12 → free
        hi[ax] = lo[ax] + w
    splittable = np.ones(d, dtype=bool)
    axis = _boundary_choose_axis(
        lo, hi, splittable,
        depth=40,
        boundary_split_depth_threshold=30,
        boundary_axis_count_threshold=d // 2,  # 11
    )
    assert axis is not None, "expected the heuristic to fire"
    assert 0 <= axis < d, f"axis {axis} out of range"
    assert splittable[axis], f"axis {axis} not splittable"
    assert lo[axis] > 1e-12, (
        f"chose boundary axis {axis} (lo={lo[axis]}); should be free"
    )
    assert axis == 19, f"expected widest free axis 19, got {axis}"
    # Sanity: 18 boundary axes, 4 free axes.
    assert int(np.count_nonzero(lo <= 1e-12)) == 18
    print(f"[T1 PASS] boundary detection at d=22 -> axis={axis} (widest free)")
    return True


def test_boundary_heuristic_falls_back():
    """When 0 axes are on the boundary, the heuristic returns None
    so the caller's existing widest-axis logic handles the box."""
    d = 22
    lo = np.full(d, 0.01, dtype=np.float64)  # all > 1e-12 → all free
    hi = lo + 0.05
    splittable = np.ones(d, dtype=bool)
    axis = _boundary_choose_axis(
        lo, hi, splittable,
        depth=40,
        boundary_split_depth_threshold=30,
        boundary_axis_count_threshold=d // 2,
    )
    assert axis is None, (
        f"heuristic should not fire with 0 boundary axes; got {axis}"
    )
    print("[T2 PASS] 0 boundary axes -> None (caller falls back)")
    return True


def test_boundary_heuristic_disabled_by_default():
    """With the default disabled threshold (999), the heuristic returns
    None regardless of how boundary-concentrated the box is."""
    d = 22
    lo = np.zeros(d, dtype=np.float64)
    hi = np.full(d, 0.05, dtype=np.float64)
    # Make every axis a boundary axis except one — extreme case.
    lo[5] = 0.02
    hi[5] = 0.10
    splittable = np.ones(d, dtype=bool)
    # Default threshold = 999; even depth=100 doesn't activate.
    axis = _boundary_choose_axis(
        lo, hi, splittable,
        depth=100,
        boundary_split_depth_threshold=999,
        boundary_axis_count_threshold=d // 2,
    )
    assert axis is None, (
        f"heuristic must be disabled at threshold 999; got {axis}"
    )
    print("[T3 PASS] default threshold 999 -> heuristic disabled")
    return True


def test_boundary_heuristic_respects_splittable_mask():
    """When the only free axis has int_width <= 1 (i.e. not splittable),
    the heuristic returns None so the caller falls back to widest
    splittable. No crash, no out-of-range axis."""
    d = 10
    lo = np.zeros(d, dtype=np.float64)
    hi = np.full(d, 0.05, dtype=np.float64)
    # One free axis, but it is NOT splittable (int_width <= 1).
    lo[4] = 0.01
    hi[4] = 0.02
    # Splittable: every boundary axis is splittable, but the free axis
    # (index 4) is NOT — simulates a deep box where the only free axis
    # has exhausted dyadic depth.
    splittable = np.ones(d, dtype=bool)
    splittable[4] = False
    axis = _boundary_choose_axis(
        lo, hi, splittable,
        depth=40,
        boundary_split_depth_threshold=30,
        boundary_axis_count_threshold=d // 2,  # 5
    )
    assert axis is None, (
        f"heuristic should return None when only free axis is "
        f"unsplittable; got {axis}"
    )
    # Sanity: there are >= 5 boundary axes (9 of them, in fact).
    assert int(np.count_nonzero(lo <= 1e-12)) == 9
    print("[T4 PASS] only free axis unsplittable -> None (caller falls back)")
    return True


def test_below_count_threshold_returns_none():
    """When depth >= threshold but boundary axis count < required, the
    heuristic returns None (so the caller's existing tier — LP-binding,
    PC-variance, widest — handles the box uninterrupted)."""
    d = 22
    lo = np.zeros(d, dtype=np.float64)
    hi = np.full(d, 0.05, dtype=np.float64)
    # Only 3 boundary axes; the other 19 are free.
    for ax in range(3, d):
        lo[ax] = 0.01
        hi[ax] = 0.06
    splittable = np.ones(d, dtype=bool)
    axis = _boundary_choose_axis(
        lo, hi, splittable,
        depth=40,
        boundary_split_depth_threshold=30,
        boundary_axis_count_threshold=d // 2,  # 11; we only have 3
    )
    assert axis is None, (
        f"heuristic should not fire with 3 < {d//2} boundary axes; got {axis}"
    )
    print(f"[T5 PASS] 3 < {d//2} boundary axes -> None (PC-variance handles)")
    return True


def test_boundary_axis_excluded_when_widest():
    """Sanity: the heuristic must NOT pick a wide boundary axis even
    when free axes are narrower. (This is the core failure mode the
    heuristic exists to prevent.)"""
    d = 22
    lo = np.zeros(d, dtype=np.float64)
    # 18 boundary axes, several with width 0.25 (the "wide" boundary
    # axes that the existing cascade would erroneously pick).
    hi = np.full(d, 0.25, dtype=np.float64)
    free_axes = [4, 11, 17, 21]
    for ax in free_axes:
        lo[ax] = 0.005
        hi[ax] = 0.012  # width 0.007, much narrower than 0.25
    splittable = np.ones(d, dtype=bool)
    axis = _boundary_choose_axis(
        lo, hi, splittable,
        depth=40,
        boundary_split_depth_threshold=30,
        boundary_axis_count_threshold=d // 2,
    )
    assert axis is not None, "expected heuristic to fire"
    assert axis in free_axes, (
        f"heuristic picked boundary axis {axis} (width 0.25); should be "
        f"a narrower free axis from {free_axes}"
    )
    assert lo[axis] > 1e-12
    print(f"[T6 PASS] wide boundary axes ignored -> picked free axis {axis}")
    return True


def main():
    print("=" * 60)
    print("Boundary-concentrated split-axis heuristic tests")
    print("=" * 60)
    results = []
    for name, fn in [
        ("test_boundary_axis_detection", test_boundary_axis_detection),
        ("test_boundary_heuristic_falls_back",
         test_boundary_heuristic_falls_back),
        ("test_boundary_heuristic_disabled_by_default",
         test_boundary_heuristic_disabled_by_default),
        ("test_boundary_heuristic_respects_splittable_mask",
         test_boundary_heuristic_respects_splittable_mask),
        ("test_below_count_threshold_returns_none",
         test_below_count_threshold_returns_none),
        ("test_boundary_axis_excluded_when_widest",
         test_boundary_axis_excluded_when_widest),
    ]:
        try:
            ok = fn()
            results.append((name, "PASS" if ok else "FAIL"))
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append((name, f"ERROR: {e}"))
    print()
    for name, status in results:
        print(f"  {name}: {status}")
    n_pass = sum(1 for _, s in results if s == "PASS")
    print(f"\n{n_pass}/{len(results)} passed")


if __name__ == "__main__":
    main()
