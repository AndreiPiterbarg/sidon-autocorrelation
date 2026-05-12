"""Apply simplex HC4-revise contraction to the 748 stuck d=10 boxes
and measure (a) shrinkage distribution, (b) prunes-for-free, (c) cert
rate before vs after through the cheap rigor-gate cascade.

Usage:
    python run_contract_d10.py
"""
from __future__ import annotations

import math
import sys
from fractions import Fraction
from typing import List, Tuple

import numpy as np

from contract_box import contract_box

from interval_bnb.box import SCALE, Box
from interval_bnb.bound_eval import (
    bound_autoconv_int_ge,
    bound_mccormick_ne_int_ge,
    bound_mccormick_sw_int_ge,
    bound_natural_int_ge,
)
from interval_bnb.windows import build_windows


D = 10
TARGET = Fraction(12, 10)
QUEUE_PATH = "stuck_d10_master_queue.npz"
OUT_PATH = "stuck_d10_contracted.npz"


def _box_from_floats(lo: np.ndarray, hi: np.ndarray) -> Box:
    """Build a Box with exact integer endpoints rounded OUTWARD so
    the integer box contains the float box."""
    lo_int = [math.floor(float(x) * SCALE) for x in lo]
    hi_int = [math.ceil(float(x) * SCALE) for x in hi]
    # Clamp to [0, SCALE] so we don't go below 0 or above 1.
    lo_int = [max(0, v) for v in lo_int]
    hi_int = [min(SCALE, v) for v in hi_int]
    return Box(np.array(lo, dtype=np.float64), np.array(hi, dtype=np.float64),
               lo_int=lo_int, hi_int=hi_int)


def cert_box(B: Box, windows, d: int, target_q: Fraction) -> bool:
    """Cheap rigor-gate cascade: certify if any window's bound clears target."""
    lo_int, hi_int = B.to_ints()
    tn, td = target_q.numerator, target_q.denominator
    for w in windows:
        if bound_natural_int_ge(lo_int, hi_int, w, tn, td):
            return True
        if bound_autoconv_int_ge(lo_int, hi_int, w, d, tn, td):
            return True
        if bound_mccormick_sw_int_ge(lo_int, hi_int, w, d, tn, td):
            return True
        if bound_mccormick_ne_int_ge(lo_int, hi_int, w, d, tn, td):
            return True
    return False


def main() -> int:
    data = np.load(QUEUE_PATH)
    LO = data["lo"].astype(np.float64)
    HI = data["hi"].astype(np.float64)
    N, d = LO.shape
    assert d == D, f"expected d={D}, got {d}"
    print(f"loaded {N} stuck boxes from {QUEUE_PATH}")

    windows = build_windows(D)
    print(f"built {len(windows)} windows for d={D}")

    # ---- BEFORE: cert rate over original boxes ----
    pre_cert = 0
    for k in range(N):
        B = _box_from_floats(LO[k], HI[k])
        if cert_box(B, windows, D, TARGET):
            pre_cert += 1
    pre_rate = pre_cert / N
    print(f"\n[BEFORE contraction]")
    print(f"  cert: {pre_cert} / {N}  =  {100 * pre_rate:.2f}%")

    # ---- CONTRACT each box ----
    LO_C = np.empty_like(LO)
    HI_C = np.empty_like(HI)
    empty_flags = np.zeros(N, dtype=bool)
    orig_widths = (HI - LO).max(axis=1)
    new_widths = np.empty(N, dtype=np.float64)
    iters_used = 0  # not tracked; using default 20

    for k in range(N):
        lo_new, hi_new, is_empty = contract_box(LO[k], HI[k], max_iters=20)
        LO_C[k] = lo_new
        HI_C[k] = hi_new
        empty_flags[k] = is_empty
        if is_empty:
            new_widths[k] = 0.0
        else:
            new_widths[k] = float((hi_new - lo_new).max())

    n_empty = int(empty_flags.sum())
    print(f"\n[CONTRACTION]")
    print(f"  boxes proven empty (prunable for free): {n_empty} / {N}")

    # Width-shrinkage ratios (only for non-empty boxes).
    nonempty = ~empty_flags
    ratios = new_widths[nonempty] / np.maximum(orig_widths[nonempty], 1e-300)
    p10 = float(np.percentile(ratios, 10))
    p50 = float(np.percentile(ratios, 50))
    p90 = float(np.percentile(ratios, 90))
    median = float(np.median(ratios))
    print(f"  width-shrinkage ratio  (contracted_max_w / original_max_w):")
    print(f"    p10 = {p10:.4f}    median = {median:.4f}    p90 = {p90:.4f}")
    print(f"    min = {float(ratios.min()):.4f}    max = {float(ratios.max()):.4f}")
    print(f"    mean = {float(ratios.mean()):.4f}")

    # Distribution buckets for color.
    buckets = [
        ("ratio == 0    (point box)",  ratios == 0.0),
        ("ratio < 0.25",                (ratios > 0.0) & (ratios < 0.25)),
        ("0.25 <= r < 0.5",             (ratios >= 0.25) & (ratios < 0.5)),
        ("0.5 <= r < 0.9",              (ratios >= 0.5) & (ratios < 0.9)),
        ("0.9 <= r < 1.0",              (ratios >= 0.9) & (ratios < 1.0)),
        ("ratio == 1    (no shrink)",   ratios == 1.0),
    ]
    print(f"  shrinkage histogram (over {len(ratios)} non-empty):")
    for label, mask in buckets:
        cnt = int(mask.sum())
        print(f"    {label:32s}  {cnt:5d}  ({100*cnt/len(ratios):5.1f}%)")

    # ---- AFTER: cert rate over contracted, non-empty boxes ----
    # We count empty boxes as "certified" (vacuously prunable).
    post_cert_empty = n_empty
    post_cert_rigor = 0
    for k in range(N):
        if empty_flags[k]:
            continue
        B = _box_from_floats(LO_C[k], HI_C[k])
        # Sanity: contracted box should still intersect simplex (else
        # contract_box would have flagged it empty).
        if not B.intersects_simplex():
            post_cert_empty += 1
            empty_flags[k] = True  # treat as prunable
            continue
        if cert_box(B, windows, D, TARGET):
            post_cert_rigor += 1

    post_cert_total = post_cert_empty + post_cert_rigor
    post_rate = post_cert_total / N
    print(f"\n[AFTER contraction]")
    print(f"  cert via empty: {post_cert_empty} / {N}")
    print(f"  cert via rigor: {post_cert_rigor} / {N}")
    print(f"  cert total:     {post_cert_total} / {N}  =  {100 * post_rate:.2f}%")
    print(f"\n[CHANGE]   "
          f"pre {100*pre_rate:.2f}%  ->  post {100*post_rate:.2f}%   "
          f"(+{100*(post_rate - pre_rate):.2f} pp)")
    if post_rate > 0.5:
        print(f"   >>> CLEARS 50% criticality threshold <<<")
    else:
        print(f"   does NOT clear 50% criticality (delta to 50%: "
              f"{100*(0.5 - post_rate):+.2f} pp)")

    # ---- Save contracted boxes ----
    np.savez_compressed(
        OUT_PATH,
        lo=LO_C, hi=HI_C,
        empty=empty_flags,
        orig_widths=orig_widths,
        new_widths=new_widths,
    )
    print(f"\nsaved -> {OUT_PATH}  (lo, hi, empty, orig_widths, new_widths)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
