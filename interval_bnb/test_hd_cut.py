"""Tests for the H_d half-simplex symmetry cut.

The H_d cut is `mu_0 <= mu_{d-1}` (proper coordinate-coupled sigma
reduction). Soundness is proved in `interval_bnb/THEOREM.md` Lemma 3.4
and the pre-filter is implemented as `interval_bnb.symmetry.box_outside_hd`.

T1: synthetic small-d boxes — the pre-filter drops only the boxes
    whose interval projection is forced *strictly* outside H_d.
T2: pre-filter agrees with the integer-arithmetic definition on a
    randomized fuzz over dyadic-rational endpoints.
T3: serial BnB at d=4 with the H_d cut still proves val(4) >= 1.0
    (cheap correctness check; val(4) ~= 1.16, so threshold 1.0 is
    achievable in seconds).
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

from interval_bnb.box import SCALE, Box  # noqa: E402
from interval_bnb.symmetry import box_outside_hd, half_simplex_cuts  # noqa: E402


def _make_box(lo, hi):
    """Build a Box with integer metadata from float dyadic-rational
    endpoints. Each input must be representable as a dyadic at
    denominator SCALE = 2**60 (always true for short floats)."""
    d = len(lo)
    lo_f = np.array(lo, dtype=np.float64)
    hi_f = np.array(hi, dtype=np.float64)
    lo_int = [int(round(x * SCALE)) for x in lo_f]
    hi_int = [int(round(x * SCALE)) for x in hi_f]
    return Box(lo_f, hi_f, lo_int, hi_int)


def test_hd_cut_drops_strictly_outside():
    """T1a: a box with lo_int[0] > hi_int[d-1] is dropped."""
    # d=4, lo_int[0] = 0.5*SCALE, hi_int[3] = 0.25*SCALE → strictly outside
    B = _make_box([0.5, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.25])
    assert box_outside_hd(B) is True


def test_hd_cut_keeps_strictly_inside():
    """T1b: a box with hi_int[0] <= lo_int[d-1] is kept (interior of H_d)."""
    B = _make_box([0.0, 0.0, 0.0, 0.5], [0.25, 1.0, 1.0, 1.0])
    assert box_outside_hd(B) is False


def test_hd_cut_keeps_straddling_boundary():
    """T1c: a box that straddles {mu_0 = mu_{d-1}} is KEPT.
    lo_int[0] = 0 <= hi_int[d-1] = SCALE — the box may host a point of H_d.
    """
    B = _make_box([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0])
    assert box_outside_hd(B) is False


def test_hd_cut_keeps_equality_endpoint():
    """T1d: a box with lo_int[0] = hi_int[d-1] (touches boundary from
    outside) is KEPT — the inequality is non-strict."""
    B = _make_box([0.25, 0.0, 0.0, 0.0], [0.5, 1.0, 1.0, 0.25])
    # lo_int[0] = 0.25*SCALE, hi_int[3] = 0.25*SCALE → equal → keep
    assert box_outside_hd(B) is False


def test_hd_cut_initial_box_kept():
    """T1e: the initial box from Box.initial(d, sym_cuts) must be kept
    (it includes the diagonal mu_0 = mu_{d-1})."""
    for d in [2, 4, 6, 8]:
        B = Box.initial(d, half_simplex_cuts(d))
        assert box_outside_hd(B) is False, f"d={d} initial box dropped"


def test_hd_cut_d2_corner_box():
    """T1f: at d=2, H_2 = {mu_0 <= mu_1}. A box [0.6, 0.0]–[1.0, 0.5]
    is forced outside (lo_int[0] = 0.6*S > hi_int[1] = 0.5*S)."""
    B = _make_box([0.6, 0.0], [1.0, 0.5])
    assert box_outside_hd(B) is True


def test_hd_cut_fuzz_consistency():
    """T2: randomized fuzz — pre-filter result matches the integer
    definition on 200 random dyadic boxes."""
    rng = np.random.default_rng(42)
    d = 6
    n_strict_outside = 0
    n_kept = 0
    for _ in range(200):
        # Sample lo <= hi in [0, 1] with dyadic rationals at 2^-20.
        denom = 1 << 20
        lo_int20 = rng.integers(0, denom + 1, size=d)
        hi_offset = rng.integers(0, denom + 1, size=d)
        hi_int20 = np.minimum(lo_int20 + hi_offset, denom)
        # Convert to SCALE = 2^60 endpoints.
        shift = SCALE >> 20
        lo_int_full = [int(x) * shift for x in lo_int20]
        hi_int_full = [int(x) * shift for x in hi_int20]
        lo_f = np.array([float(x) / SCALE for x in lo_int_full], dtype=np.float64)
        hi_f = np.array([float(x) / SCALE for x in hi_int_full], dtype=np.float64)
        B = Box(lo_f, hi_f, lo_int_full, hi_int_full)

        expected = lo_int_full[0] > hi_int_full[d - 1]
        got = box_outside_hd(B)
        assert got == expected, f"mismatch on lo[0]={lo_int_full[0]} vs hi[d-1]={hi_int_full[d-1]}"
        if expected:
            n_strict_outside += 1
        else:
            n_kept += 1
    # Sanity: roughly half should be outside on a uniform fuzz.
    assert n_strict_outside > 20, f"only {n_strict_outside}/200 outside H_d"
    assert n_kept > 20, f"only {n_kept}/200 kept"


def test_hd_cut_d4_serial_bnb_certifies_1p0():
    """T3: serial BnB at d=4 with the H_d cut still certifies val(4) >= 1.0.
    val(4) ~= 1.16; target 1.0 should close in <30s on any machine.
    """
    from interval_bnb.bnb import branch_and_bound
    result = branch_and_bound(
        d=4,
        target_c="1.0",
        max_nodes=2_000_000,
        min_box_width=1e-10,
        use_symmetry=True,
        time_budget_s=60.0,
        verbose=False,
    )
    assert result.success, (
        f"d=4 target=1.0 BnB failed under H_d cut "
        f"(failing_lb={result.failing_lb}, "
        f"nodes={result.stats.nodes_processed})"
    )


if __name__ == "__main__":
    # Direct invocation runs each test and prints pass/fail.
    import traceback
    tests = [
        test_hd_cut_drops_strictly_outside,
        test_hd_cut_keeps_strictly_inside,
        test_hd_cut_keeps_straddling_boundary,
        test_hd_cut_keeps_equality_endpoint,
        test_hd_cut_initial_box_kept,
        test_hd_cut_d2_corner_box,
        test_hd_cut_fuzz_consistency,
        test_hd_cut_d4_serial_bnb_certifies_1p0,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except Exception:
            failed += 1
            print(f"[FAIL] {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(0 if failed == 0 else 1)
