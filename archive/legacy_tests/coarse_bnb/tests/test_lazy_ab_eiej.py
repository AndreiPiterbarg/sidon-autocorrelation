#!/usr/bin/env python
"""Regression test: lazy ab_eiej slicing must match eager materialisation.

At d=6 L=3 the full (n_loc, n_loc, d, d) array is small, so we can build
both eager and lazy precompute dicts and verify:

  * All per-(i,j) slices agree element-wise.
  * All per-window nonzero slices ab_eiej_idx[:,:,nz_i,nz_j] agree.
  * The lazy helpers return int64 arrays of the right shape.

Run:  python tests/test_lazy_ab_eiej.py
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

from lasserre_scalable import _precompute
from lasserre.core import (
    get_ab_eiej_slice, get_ab_eiej_ij,
    ab_eiej_slice_ij, ab_eiej_slice_batch,
)


def _check_ij_matches(d: int, order: int) -> None:
    print(f"[d={d} L={order}] building eager precompute...")
    P_eager = _precompute(d, order, verbose=False, lazy_ab_eiej=False)
    print(f"[d={d} L={order}] building lazy precompute...")
    P_lazy = _precompute(d, order, verbose=False, lazy_ab_eiej=True)

    assert P_eager['ab_eiej_idx'] is not None, \
        "eager precompute must materialise ab_eiej_idx"
    assert P_lazy['ab_eiej_idx'] is None, \
        "lazy precompute must NOT materialise ab_eiej_idx"

    eager_arr = P_eager['ab_eiej_idx']
    n_loc = P_eager['n_loc']
    assert eager_arr.shape == (n_loc, n_loc, d, d)

    # 1) Per-(i,j) scalar slice matches for every (i, j).
    ncheck = 0
    for i in range(d):
        for j in range(d):
            ref = eager_arr[:, :, i, j]
            got_lazy = get_ab_eiej_ij(P_lazy, i, j)
            got_eager = get_ab_eiej_ij(P_eager, i, j)
            if not np.array_equal(ref, got_lazy):
                raise AssertionError(
                    f"lazy (i,j)=({i},{j}) differs from eager reference")
            if not np.array_equal(ref, got_eager):
                raise AssertionError(
                    f"get_ab_eiej_ij on eager P differs from direct slice")
            ncheck += 1
    print(f"  OK: {ncheck} (i,j) slices match")

    # 2) Window-style batch slice matches for every nontrivial window.
    nz_total = 0
    for w in P_eager['nontrivial_windows']:
        Mw = P_eager['M_mats'][w]
        nz_i, nz_j = np.nonzero(Mw)
        if len(nz_i) == 0:
            continue
        ref = eager_arr[:, :, nz_i, nz_j]
        got_lazy = get_ab_eiej_slice(P_lazy, nz_i, nz_j)
        got_eager = get_ab_eiej_slice(P_eager, nz_i, nz_j)
        if not np.array_equal(ref, got_lazy):
            raise AssertionError(
                f"lazy window slice mismatch at w={w}")
        if not np.array_equal(ref, got_eager):
            raise AssertionError(
                f"eager passthrough mismatch at w={w}")
        nz_total += int(len(nz_i))
    print(f"  OK: {len(P_eager['nontrivial_windows'])} windows, "
          f"{nz_total} total (i,j) lookups match")

    # 3) Raw helper functions against eager reference.
    for (i, j) in [(0, 0), (0, d - 1), (d // 2, d // 2), (d - 1, d - 1)]:
        ref = eager_arr[:, :, i, j]
        got = ab_eiej_slice_ij(
            P_lazy['AB_loc_hash'], P_lazy['bases'],
            P_lazy['sorted_h'], P_lazy['sort_o'], i, j,
            prime=P_lazy.get('prime'))
        if not np.array_equal(ref, got):
            raise AssertionError(
                f"ab_eiej_slice_ij mismatch at ({i},{j})")
    print(f"  OK: raw ab_eiej_slice_ij agrees with eager array")

    # 4) Batched helper over a selection.
    idx_i = np.array([0, 1, d - 1, 2], dtype=np.int64)
    idx_j = np.array([d - 1, 0, 0, 3], dtype=np.int64)
    ref = eager_arr[:, :, idx_i, idx_j]
    got = ab_eiej_slice_batch(
        P_lazy['AB_loc_hash'], P_lazy['bases'],
        P_lazy['sorted_h'], P_lazy['sort_o'],
        idx_i, idx_j, prime=P_lazy.get('prime'))
    if not np.array_equal(ref, got):
        raise AssertionError("ab_eiej_slice_batch mismatch")
    print(f"  OK: raw ab_eiej_slice_batch agrees with eager array")


def main() -> int:
    # Smallest case that exercises order>=2 (so ab_eiej is built at all).
    _check_ij_matches(d=4, order=2)
    _check_ij_matches(d=6, order=2)
    _check_ij_matches(d=6, order=3)
    print("\nAll lazy-vs-eager ab_eiej checks passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
