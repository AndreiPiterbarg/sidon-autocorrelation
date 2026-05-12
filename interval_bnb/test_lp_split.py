"""Tests for the LP-binding split-axis heuristic.

The heuristic picks the split axis based on which McCormick face is most
binding in the just-solved per-box epigraph LP. Soundness is automatic
(any axis split is sound); these tests verify:

  * Soundness: chosen axis is in [0, d) and `splittable[axis]` is True.
  * Determinism: same (ineqlin, widths) → same axis.
  * Sanity: when only one face has nonzero margin, the corresponding
    axis is chosen.
  * Integration: `bound_epigraph_int_ge_with_marginals` returns the
    same cert verdict as `bound_epigraph_int_ge` (no double-solve).
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from interval_bnb.bound_epigraph import (
    bound_epigraph_int_ge,
    bound_epigraph_int_ge_with_marginals,
    bound_epigraph_lp_float,
    lp_binding_axis_score,
)
from interval_bnb.box import SCALE as _SCALE
from interval_bnb.windows import build_windows


def _ints_from_floats(lo_f, hi_f):
    return ([int(round(x * _SCALE)) for x in lo_f],
            [int(round(x * _SCALE)) for x in hi_f])


def _choose_axis(ineqlin, widths, splittable, d):
    """Mirror of the chooser logic in parallel.py for unit testing."""
    score = lp_binding_axis_score(ineqlin, widths, d)
    score = np.where(splittable, score, -np.inf)
    cand = int(np.argmax(score))
    if np.isfinite(score[cand]) and score[cand] > 0:
        return cand
    return None


def test_score_returns_zero_for_none_marginals():
    d = 5
    widths = np.full(d, 0.1)
    out = lp_binding_axis_score(None, widths, d)
    assert out.shape == (d,)
    assert np.all(out == 0.0)
    print("[T1 PASS] None marginals -> all-zero score")
    return True


def test_score_shape_and_dtype():
    d = 4
    n_y = d * d
    # Just SW/NE/NW/SE blocks (4*n_y); no need for epigraph rows for the
    # axis score — only the first 4*n_y entries are read.
    rng = np.random.default_rng(1)
    ineqlin = rng.standard_normal(4 * n_y + 3)
    widths = np.array([0.1, 0.2, 0.3, 0.4])
    out = lp_binding_axis_score(ineqlin, widths, d)
    assert out.shape == (d,)
    assert out.dtype == np.float64
    assert np.all(out >= 0.0)  # |λ| · width is nonneg
    print(f"[T2 PASS] score shape ok: {out}")
    return True


def test_sanity_single_face():
    """When only one face dual is non-zero (face for pair (i*, j*)), the
    chooser picks an axis touching that pair (i.e. axis i* or j*)."""
    d = 4
    n_y = d * d
    ineqlin = np.zeros(4 * n_y + 5)
    # Set a single SW face dual at pair (i*, j*) = (2, 1).
    i_star, j_star = 2, 1
    ineqlin[i_star * d + j_star] = -3.0  # SW block, |λ| = 3
    widths = np.full(d, 1.0)  # uniform widths so the pair indices win.
    splittable = np.ones(d, dtype=bool)
    axis = _choose_axis(ineqlin, widths, splittable, d)
    assert axis in (i_star, j_star), (
        f"expected axis in ({i_star},{j_star}), got {axis}"
    )
    print(f"[T3 PASS] single-face dual at ({i_star},{j_star}) -> axis={axis}")
    return True


def test_sanity_diagonal_face():
    """A diagonal face λ at pair (i,i) only touches axis i."""
    d = 5
    n_y = d * d
    ineqlin = np.zeros(4 * n_y + 5)
    i_star = 3
    # Diagonal contributes to BOTH row i* and column i* — same axis.
    ineqlin[n_y + i_star * d + i_star] = -7.5  # NE block diagonal
    widths = np.full(d, 0.5)
    splittable = np.ones(d, dtype=bool)
    axis = _choose_axis(ineqlin, widths, splittable, d)
    assert axis == i_star, f"expected axis={i_star}, got {axis}"
    print(f"[T4 PASS] diagonal face at axis {i_star} -> axis={axis}")
    return True


def test_determinism():
    """Given fixed ineqlin and widths, the chosen axis is deterministic."""
    d = 6
    n_y = d * d
    rng = np.random.default_rng(42)
    ineqlin = rng.standard_normal(4 * n_y + 8)
    widths = rng.uniform(0.01, 0.5, size=d)
    splittable = np.ones(d, dtype=bool)
    a1 = _choose_axis(ineqlin, widths, splittable, d)
    a2 = _choose_axis(ineqlin, widths, splittable, d)
    a3 = _choose_axis(ineqlin.copy(), widths.copy(), splittable.copy(), d)
    assert a1 == a2 == a3, f"non-deterministic: {a1}, {a2}, {a3}"
    print(f"[T5 PASS] deterministic: axis={a1}")
    return True


def test_soundness_axis_in_range_and_splittable():
    """Axis must be in [0, d) and splittable[axis] must be True."""
    d = 7
    n_y = d * d
    rng = np.random.default_rng(7)
    ineqlin = np.abs(rng.standard_normal(4 * n_y + 10))  # all positive
    widths = rng.uniform(0.01, 0.5, size=d)
    # Mark some axes as not splittable.
    splittable = np.array([True, False, True, True, False, True, True])
    axis = _choose_axis(ineqlin, widths, splittable, d)
    assert axis is not None, "expected a valid axis"
    assert 0 <= axis < d, f"axis {axis} out of range"
    assert splittable[axis], f"axis {axis} not splittable"
    print(f"[T6 PASS] soundness: axis={axis}, splittable[axis]={splittable[axis]}")
    return True


def test_returns_none_when_all_zero():
    """All-zero ineqlin -> chooser returns None (caller falls back)."""
    d = 4
    ineqlin = np.zeros(4 * d * d + 3)
    widths = np.full(d, 0.1)
    splittable = np.ones(d, dtype=bool)
    axis = _choose_axis(ineqlin, widths, splittable, d)
    assert axis is None
    print("[T7 PASS] zero marginals -> None (caller falls back)")
    return True


def test_returns_none_when_no_splittable():
    """No splittable axis -> chooser returns None."""
    d = 4
    n_y = d * d
    ineqlin = np.abs(np.random.default_rng(0).standard_normal(4 * n_y + 5))
    widths = np.full(d, 0.1)
    splittable = np.zeros(d, dtype=bool)
    axis = _choose_axis(ineqlin, widths, splittable, d)
    assert axis is None
    print("[T8 PASS] no splittable axis -> None")
    return True


def test_marginals_helper_agrees_with_int_ge():
    """`bound_epigraph_int_ge_with_marginals` matches `bound_epigraph_int_ge`
    on the cert verdict, and returns ineqlin of length >= 4*d*d."""
    d = 6
    windows = build_windows(d)
    lo_f = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    hi_f = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
    lo_int, hi_int = _ints_from_floats(lo_f, hi_f)
    lp_val = bound_epigraph_lp_float(lo_f, hi_f, windows, d)

    target_low = Fraction.from_float(lp_val - 1e-4).limit_denominator(10**10)
    target_high = Fraction.from_float(lp_val + 1e-4).limit_denominator(10**10)

    cert_low_legacy = bound_epigraph_int_ge(
        lo_int, hi_int, windows, d,
        target_low.numerator, target_low.denominator,
    )
    cert_high_legacy = bound_epigraph_int_ge(
        lo_int, hi_int, windows, d,
        target_high.numerator, target_high.denominator,
    )
    cert_low_new, lp_val_new, ineqlin_new = (
        bound_epigraph_int_ge_with_marginals(
            lo_f, hi_f, windows, d, float(target_low),
        )
    )
    cert_high_new, _, _ = bound_epigraph_int_ge_with_marginals(
        lo_f, hi_f, windows, d, float(target_high),
    )
    assert cert_low_legacy == cert_low_new, (
        f"low cert mismatch: legacy={cert_low_legacy}, new={cert_low_new}"
    )
    assert cert_high_legacy == cert_high_new, (
        f"high cert mismatch: legacy={cert_high_legacy}, new={cert_high_new}"
    )
    assert ineqlin_new is not None
    assert len(ineqlin_new) >= 4 * d * d, (
        f"ineqlin too short: {len(ineqlin_new)} < {4*d*d}"
    )
    # Now exercise the chooser end-to-end on a real LP.
    widths = hi_f - lo_f
    splittable = np.ones(d, dtype=bool)
    axis = _choose_axis(ineqlin_new, widths, splittable, d)
    # Some axis (or None) is fine; just verify it's well-formed.
    if axis is not None:
        assert 0 <= axis < d
        assert splittable[axis]
    print(f"[T9 PASS] helper agrees with int_ge "
          f"(lp_val={lp_val_new:.6f}, axis={axis})")
    return True


def test_real_lp_axis_in_range():
    """Run the real epigraph LP at d=4 around mu_star and verify the
    chosen axis is in range and splittable."""
    d = 4
    windows = build_windows(d)
    mu_star = np.array([1/3, 1/6, 1/6, 1/3])
    radius = 1e-3
    lo_f = mu_star - radius
    hi_f = mu_star + radius
    _, lp_val, ineqlin = bound_epigraph_int_ge_with_marginals(
        lo_f, hi_f, windows, d, 1.05,
    )
    assert ineqlin is not None
    widths = hi_f - lo_f
    splittable = np.ones(d, dtype=bool)
    axis = _choose_axis(ineqlin, widths, splittable, d)
    if axis is not None:
        assert 0 <= axis < d
        assert splittable[axis]
        print(f"[T10 PASS] real LP at d=4: lp_val={lp_val:.6f}, axis={axis}")
    else:
        print(f"[T10 PASS] real LP at d=4: lp_val={lp_val:.6f}, "
              f"all-zero marginals -> fallback (None)")
    return True


def main():
    print("=" * 60)
    print("LP-binding split-axis heuristic tests")
    print("=" * 60)
    results = []
    for name, fn in [
        ("test_score_returns_zero_for_none_marginals",
         test_score_returns_zero_for_none_marginals),
        ("test_score_shape_and_dtype", test_score_shape_and_dtype),
        ("test_sanity_single_face", test_sanity_single_face),
        ("test_sanity_diagonal_face", test_sanity_diagonal_face),
        ("test_determinism", test_determinism),
        ("test_soundness_axis_in_range_and_splittable",
         test_soundness_axis_in_range_and_splittable),
        ("test_returns_none_when_all_zero",
         test_returns_none_when_all_zero),
        ("test_returns_none_when_no_splittable",
         test_returns_none_when_no_splittable),
        ("test_marginals_helper_agrees_with_int_ge",
         test_marginals_helper_agrees_with_int_ge),
        ("test_real_lp_axis_in_range", test_real_lp_axis_in_range),
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
