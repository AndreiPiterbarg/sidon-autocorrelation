"""Spec §3 sanity rule: every admissible f must pass every filter.

Any filter that rejects any admissible f is mathematically invalid and
the corresponding code must be removed / proof revisited.

This test iterates the (currently 12-entry) admissible library at
N in {2, 3, 4, 5, 6} and asserts that each filter's verdict on the
library f is NOT ``REJECT``.  (``UNCLEAR`` is permitted -- that indicates
the filter boundary, which is correct behaviour when 2n > N forces the
use of a weak-bound [-1, 1] stand-in for missing moments.)
"""
from __future__ import annotations

import pytest

from flint import ctx

from delsarte_dual.grid_bound.admissible_f import build_library
from delsarte_dual.grid_bound.filters import (
    F1_magnitude, F2_moment_cs, F4_MO217, F7_bochner_f, F8_bochner_h,
    FilterVerdict,
)


_FILTERS = [
    ("F1_magnitude", F1_magnitude),
    ("F2_moment_cs", F2_moment_cs),
    ("F4_MO217",     F4_MO217),
    ("F7_bochner_f", F7_bochner_f),
    ("F8_bochner_h", F8_bochner_h),
]


@pytest.mark.parametrize("N", [2, 3, 4, 5, 6])
@pytest.mark.parametrize(
    "filter_name,filter_fn",
    _FILTERS,
    ids=[name for name, _ in _FILTERS],
)
def test_filter_accepts_every_admissible_f(filter_name, filter_fn, N):
    ctx.prec = 256
    lib = build_library()
    assert lib, "library is empty"
    for f in lib:
        ab = f.moments(N, prec_bits=256)
        verdict = filter_fn(ab, N)
        assert verdict != FilterVerdict.REJECT, (
            f"Filter {filter_name} REJECTS admissible f '{f.name}' at N={N}. "
            "The filter or its proof has a bug -- stop and investigate."
        )


def test_f7_strict_rejection_detects_nonpsd_toeplitz():
    """Rigour sanity: F7 must REJECT a (hand-built) non-PSD Toeplitz vector.

    Take z_1 = (a_1, b_1) = (1.1, 0): then |hat f(1)|^2 = 1.21 > 1 so
    T_f[1] = [[1, 1.1], [1.1, 1]] has det = 1 - 1.21 = -0.21 < 0 => F7 rejects.
    """
    from flint import arb
    ctx.prec = 128
    ab = [arb("1.1"), arb(0)]
    # At N=1, F7 reduces to 2x2 minor = 1 - z_1^2 = -0.21 < 0.
    assert F7_bochner_f(ab, 1) == FilterVerdict.REJECT


def test_f4_mo217_rejects_violating_point():
    """F4_MO217 must REJECT any (a_1, a_2) with a_2 > 2 a_1 - 1.

    E.g. a_1 = 0.5, a_2 = 0.2 gives 2*0.5 - 1 - 0.2 = -0.2 < 0  =>  REJECT.
    """
    from flint import arb
    ctx.prec = 128
    ab = [arb("0.5"), arb(0), arb("0.2"), arb(0)]   # (a_1, b_1, a_2, b_2)
    assert F4_MO217(ab, 2) == FilterVerdict.REJECT
