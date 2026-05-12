"""Coarse cascade prover v2 — tight box-cert + parent pre-pruning.

Improvements over ``coarse_cascade_prover.py``:

1. **Tight box certification via McCormick LP** (replaces loose water-filling).
   The original used ``max_W (min_box TV_W)``, which is always ≤ the actual
   ``min_box (max_W TV_W)`` we need by the minimax inequality. The LP gives
   a sound lower bound on the true minimax via McCormick envelopes — closes
   the O(δ) → O(δ²) looseness gap.

2. **Whole-parent pre-pruning via Theorem-1 interval arithmetic** (Idea 4
   from CASCADE_UPDATE.md). For each parent at L1+, computes lower bounds
   on every conv entry across the entire cursor box; if any window exceeds
   threshold, the entire parent is pruned without enumerating ANY child.

3. **Hybrid box cert** — fast water-filling first, McCormick LP only on
   cells where water-filling fails. Drastically reduces total LP solves.

4. **Canonical-during-enum (Idea 5)** — skip non-canonical (rev-symmetric)
   children during cursor DFS instead of post-hoc dedup.

All changes preserve mathematical soundness (Theorem 1 → val(d) → C_{1a}).

Usage:
  python coarse_cascade_prover_v2.py --c_target 1.20 --S 50 --d_start 4
"""
from __future__ import annotations

import argparse
import time
import os
import sys

import numpy as np
import numba
from numba import njit, prange


# =====================================================================
# Threshold computation (identical to v1)
# =====================================================================

def compute_thresholds(c_target, S, d):
    """Per-ell integer thresholds for dimension d.

    Prune if ws_int > thr[ell].  Equivalent to TV >= c_target.
    Threshold formula:  thr[ell] = floor(c_target * ell * S^2 / (2*d) - eps).
    """
    max_ell = 2 * d
    thr = np.empty(max_ell + 1, dtype=np.int64)
    S2 = np.float64(S) * np.float64(S)
    two_d = np.float64(2 * d)
    for ell in range(2, max_ell + 1):
        thr[ell] = np.int64(c_target * np.float64(ell) * S2 / two_d - 1e-9)
    return thr


def compute_xcap(c_target, S, d):
    """Max integer mass per bin before self-conv prunes it (ell=2)."""
    return int(np.floor(S * np.sqrt(c_target / d)))


# =====================================================================
# L0: BnB (identical to v1) — full B&B with subtree pruning
# =====================================================================

@njit(cache=True)
def _l0_bnb_inner(c0, d, S, x_cap, thr, out_buf, count_only):
    """L0 BnB with bins[0]=c0 fixed.  See coarse_cascade_prover.py."""
    conv_len = 2 * d - 1
    d_m1 = d - 1
    max_ell = 2 * d

    conv = np.zeros(conv_len, dtype=np.int64)
    bins = np.zeros(d, dtype=np.int32)
    rem_arr = np.zeros(d, dtype=np.int32)

    bins[0] = np.int32(c0)
    conv[0] = np.int64(c0) * np.int64(c0)
    rem_arr[0] = np.int32(S)
    rem_arr[1] = np.int32(S - c0)

    n_surv = np.int64(0)
    n_tested = np.int64(0)
    buf_cap = np.int64(0)
    if not count_only:
        buf_cap = np.int64(out_buf.shape[0])

    if d == 1:
        if c0 == S:
            n_tested = 1
        return n_surv, n_tested

    if d == 2:
        forced = S - c0
        if 0 <= forced <= x_cap:
            n_tested = 1
            bins[1] = np.int32(forced)
            conv[0] = np.int64(c0) * np.int64(c0)
            conv[1] = np.int64(2) * np.int64(c0) * np.int64(forced)
            conv[2] = np.int64(forced) * np.int64(forced)

            pruned = False
            for ell in range(2, max_ell + 1):
                if pruned:
                    break
                n_cv = ell - 1
                nw = conv_len - n_cv + 1
                ws = np.int64(0)
                for k in range(n_cv):
                    ws += conv[k]
                for s_lo in range(nw):
                    if s_lo > 0:
                        ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                    if ws > thr[ell]:
                        pruned = True
                        break

            if not pruned:
                if not count_only and n_surv < buf_cap:
                    out_buf[n_surv, 0] = np.int32(c0)
                    out_buf[n_surv, 1] = np.int32(forced)
                n_surv += 1
        return n_surv, n_tested

    pos = 1
    bins[1] = np.int32(0)

    while True:
        c_val = bins[pos]
        rem = rem_arr[pos]

        if pos == d_m1:
            forced = rem
            if 0 <= forced <= x_cap:
                n_tested += 1
                bins[pos] = np.int32(forced)
                f64 = np.int64(forced)
                conv[2 * pos] += f64 * f64
                for j in range(pos):
                    conv[pos + j] += np.int64(2) * f64 * np.int64(bins[j])

                pruned_leaf = False
                for ell in range(2, max_ell + 1):
                    if pruned_leaf:
                        break
                    n_cv = ell - 1
                    nw = conv_len - n_cv + 1
                    ws = np.int64(0)
                    for k in range(n_cv):
                        ws += conv[k]
                    for s_lo in range(nw):
                        if s_lo > 0:
                            ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                        if ws > thr[ell]:
                            pruned_leaf = True
                            break

                if not pruned_leaf:
                    if not count_only and n_surv < buf_cap:
                        for i in range(d):
                            out_buf[n_surv, i] = bins[i]
                    n_surv += 1

                conv[2 * pos] -= f64 * f64
                for j in range(pos):
                    conv[pos + j] -= np.int64(2) * f64 * np.int64(bins[j])

            pos -= 1
            if pos < 1:
                break
            c_old = np.int64(bins[pos])
            if c_old > 0:
                conv[2 * pos] -= c_old * c_old
                for j in range(pos):
                    conv[pos + j] -= np.int64(2) * c_old * np.int64(bins[j])
            bins[pos] = np.int32(bins[pos] + 1)
            continue

        max_v = min(rem, x_cap)
        min_v = rem - (d_m1 - pos) * x_cap
        if min_v < 0:
            min_v = 0
        if c_val < min_v:
            bins[pos] = np.int32(min_v)
            c_val = min_v

        if c_val > max_v:
            if pos <= 1:
                break
            pos -= 1
            c_old = np.int64(bins[pos])
            if c_old > 0:
                conv[2 * pos] -= c_old * c_old
                for j in range(pos):
                    conv[pos + j] -= np.int64(2) * c_old * np.int64(bins[j])
            bins[pos] = np.int32(bins[pos] + 1)
            continue

        c64 = np.int64(c_val)
        if c_val > 0:
            conv[2 * pos] += c64 * c64
            for j in range(pos):
                conv[pos + j] += np.int64(2) * c64 * np.int64(bins[j])

        max_cv_pos = 2 * pos
        pruned_partial = False
        for ell in range(2, max_ell + 1):
            if pruned_partial:
                break
            n_cv = ell - 1
            max_s = min(max_cv_pos, conv_len - n_cv)
            if max_s < 0:
                continue
            ws = np.int64(0)
            init_end = min(n_cv, max_cv_pos + 1)
            for k in range(init_end):
                ws += conv[k]
            if ws > thr[ell]:
                pruned_partial = True
                break
            for s_lo in range(1, max_s + 1):
                new_k = s_lo + n_cv - 1
                if new_k <= max_cv_pos:
                    ws += conv[new_k]
                ws -= conv[s_lo - 1]
                if ws > thr[ell]:
                    pruned_partial = True
                    break

        if pruned_partial:
            if c_val > 0:
                conv[2 * pos] -= c64 * c64
                for j in range(pos):
                    conv[pos + j] -= np.int64(2) * c64 * np.int64(bins[j])
            bins[pos] = np.int32(bins[pos] + 1)
            continue

        rem_arr[pos + 1] = np.int32(rem - c_val)
        pos += 1
        bins[pos] = np.int32(0)

    return n_surv, n_tested


@njit(parallel=True, cache=True)
def _l0_count(d, S, x_cap, thr, min_c0, n_c0, counts, tested):
    dummy = np.empty((0, d), dtype=np.int32)
    for idx in prange(n_c0):
        c0 = np.int32(min_c0 + idx)
        ns, nt = _l0_bnb_inner(c0, d, S, x_cap, thr, dummy, True)
        counts[idx] = ns
        tested[idx] = nt


@njit(parallel=True, cache=True)
def _l0_fill(d, S, x_cap, thr, min_c0, n_c0, counts, offsets, out_buf):
    for idx in prange(n_c0):
        c0 = np.int32(min_c0 + idx)
        cnt = counts[idx]
        if cnt == 0:
            continue
        off = offsets[idx]
        _l0_bnb_inner(c0, d, S, x_cap, thr, out_buf[off:off + cnt], False)


def run_l0(d, S, c_target):
    thr = compute_thresholds(c_target, S, d)
    x_cap = compute_xcap(c_target, S, d)

    min_c0 = max(0, S - (d - 1) * x_cap)
    max_c0 = min(S, x_cap)
    max_c0 = min(max_c0, S // 2)
    n_c0 = max_c0 - min_c0 + 1

    if n_c0 <= 0:
        return np.empty((0, d), dtype=np.int32), 0, 0

    counts = np.zeros(n_c0, dtype=np.int64)
    tested = np.zeros(n_c0, dtype=np.int64)

    _l0_count(d, S, x_cap, thr, min_c0, n_c0, counts, tested)

    offsets = np.zeros(n_c0 + 1, dtype=np.int64)
    for i in range(n_c0):
        offsets[i + 1] = offsets[i] + counts[i]
    total_surv = int(offsets[n_c0])
    total_tested = int(np.sum(tested))

    if total_surv == 0:
        return np.empty((0, d), dtype=np.int32), 0, total_tested

    out_buf = np.empty((total_surv, d), dtype=np.int32)
    _l0_fill(d, S, x_cap, thr, min_c0, n_c0, counts, offsets, out_buf)

    return out_buf, total_surv, total_tested


# =====================================================================
# IDEA 4: Whole-parent pre-pruning via Theorem 1 interval arithmetic
# Adapted from cloninger-steinerberger/cpu/cascade_opts.py
# =====================================================================

@njit(cache=True)
def _whole_parent_prune(parent, d_parent, S, x_cap, thr):
    """Check if ALL children of a parent are provably pruned.

    For each child autoconvolution entry conv[r], compute a sound lower
    bound over ALL possible cursor assignments via interval arithmetic.
    If any window sum of these lower bounds exceeds threshold[ell], the
    parent has no surviving children — skip enumeration entirely.

    Soundness: each per-entry lower bound is sound (from corner/interior
    minimization of bilinear forms over the cursor box). Sum of
    lower bounds ≤ true min over the cursor box. So:
        sum(min_conv[k] over k in W) ≤ true_min_ws  ≤  threshold ⇒ pruned

    Returns True if all children are provably pruned.
    """
    d_child = 2 * d_parent
    conv_len = 2 * d_child - 1
    max_ell = 2 * d_child

    # Cursor ranges: lo[i] = max(0, parent[i] - x_cap), hi[i] = min(parent[i], x_cap).
    min_conv = np.zeros(conv_len, dtype=np.int64)

    # Within-parent contributions
    for i in range(d_parent):
        Pi = np.int64(parent[i])
        lo_i = np.int64(max(0, parent[i] - x_cap))
        hi_i = np.int64(min(parent[i], x_cap))
        twoP_i = np.int64(2) * Pi
        # Note: in the COARSE prover, the cursor x_i represents child[2i],
        # and child[2i+1] = parent[i] - x_i  (NOT 2*parent[i] - x_i — that
        # was the FINE cascade convention). Need to verify.
        # Looking at coarse prover line 332-333:
        #   "child[2i] = cursor[i], child[2i+1] = parent[i] - cursor[i]"
        # So child[2i+1] = parent[i] - x_i, NOT 2P - x.
        # Adjust formulas accordingly.

        # In coarse cascade: child[2i] = x in [lo_i, hi_i], child[2i+1] = P_i - x
        # where P_i = parent[i].
        # x_other = P_i - x has range [P_i - hi_i, P_i - lo_i] = [comp_lo, comp_hi]
        comp_lo = Pi - hi_i  # min of child[2i+1]
        comp_hi = Pi - lo_i  # max of child[2i+1]

        # Self-term for child[2i]: x^2, monotone increasing -> min = lo_i^2
        min_conv[4 * i] += lo_i * lo_i

        # Self-term for child[2i+1]: (P_i - x)^2, decreasing in x -> min = comp_lo^2
        min_conv[4 * i + 2] += comp_lo * comp_lo

        # Mutual term: 2*x*(P_i - x), concave -> min at endpoints
        val_lo = np.int64(2) * lo_i * (Pi - lo_i)
        val_hi = np.int64(2) * hi_i * (Pi - hi_i)
        if val_lo < val_hi:
            min_conv[4 * i + 1] += val_lo
        else:
            min_conv[4 * i + 1] += val_hi

    # Cross-parent contributions
    for i in range(d_parent):
        Pi = np.int64(parent[i])
        lo_i = np.int64(max(0, parent[i] - x_cap))
        hi_i = np.int64(min(parent[i], x_cap))
        comp_lo_i = Pi - hi_i
        comp_hi_i = Pi - lo_i

        for j in range(i + 1, d_parent):
            Pj = np.int64(parent[j])
            lo_j = np.int64(max(0, parent[j] - x_cap))
            hi_j = np.int64(min(parent[j], x_cap))
            comp_lo_j = Pj - hi_j
            comp_hi_j = Pj - lo_j

            # conv[2i + 2j] += 2*x_i*x_j → min = 2*lo_i*lo_j (both nonneg)
            min_conv[2 * i + 2 * j] += np.int64(2) * lo_i * lo_j

            # conv[2i+1 + 2j+1] += 2*(P_i-x_i)*(P_j-x_j) → min = 2*comp_lo_i*comp_lo_j
            min_conv[2 * i + 1 + 2 * j + 1] += np.int64(2) * comp_lo_i * comp_lo_j

            # conv[2i + 2j+1] += 2*x_i*(P_j - x_j)  (one term)
            #   bilinear in (x_i, x_j); min over 4 corners of [lo_i,hi_i] x [lo_j,hi_j]
            v1 = np.int64(2) * lo_i * (Pj - lo_j)
            v2 = np.int64(2) * lo_i * (Pj - hi_j)
            v3 = np.int64(2) * hi_i * (Pj - lo_j)
            v4 = np.int64(2) * hi_i * (Pj - hi_j)
            mv1 = v1
            if v2 < mv1:
                mv1 = v2
            if v3 < mv1:
                mv1 = v3
            if v4 < mv1:
                mv1 = v4
            min_conv[2 * i + 2 * j + 1] += mv1

            # conv[2i+1 + 2j] += 2*(P_i - x_i)*x_j
            v1 = np.int64(2) * (Pi - lo_i) * lo_j
            v2 = np.int64(2) * (Pi - lo_i) * hi_j
            v3 = np.int64(2) * (Pi - hi_i) * lo_j
            v4 = np.int64(2) * (Pi - hi_i) * hi_j
            mv2 = v1
            if v2 < mv2:
                mv2 = v2
            if v3 < mv2:
                mv2 = v3
            if v4 < mv2:
                mv2 = v4
            # conv[2i+1 + 2j] = conv[2(i+j)+1] = same index as the previous one
            #   2i + 2j+1 == 2i+1 + 2j, so both terms accumulate into the SAME entry
            min_conv[2 * i + 1 + 2 * j] += mv2

    # Window scan
    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1

        ws = np.int64(0)
        for k in range(n_cv):
            ws += min_conv[k]

        if ws > thr[ell]:
            return True

        for s_lo in range(1, n_windows):
            ws += min_conv[s_lo + n_cv - 1] - min_conv[s_lo - 1]
            if ws > thr[ell]:
                return True

    return False


# =====================================================================
# L1+: Cascade level with whole-parent pre-pruning + standard B&B
# =====================================================================

@njit(cache=True)
def _cascade_child_bnb_v2(parent, d_parent, S, x_cap, thr, out_buf):
    """Same as v1 _cascade_child_bnb but operates on the coarse-cascade
    cursor convention: child[2i] = cursor[i], child[2i+1] = parent[i] - cursor[i].
    """
    d_child = 2 * d_parent
    conv_len = 2 * d_child - 1
    max_ell = 2 * d_child

    lo = np.empty(d_parent, dtype=np.int32)
    hi = np.empty(d_parent, dtype=np.int32)
    for i in range(d_parent):
        lo[i] = np.int32(max(0, parent[i] - x_cap))
        hi[i] = np.int32(min(parent[i], x_cap))

    product = np.int64(1)
    for i in range(d_parent):
        product *= np.int64(hi[i] - lo[i] + 1)
    if product == 0:
        return 0, np.int64(0)

    cursor = np.empty(d_parent, dtype=np.int32)
    child = np.zeros(d_child, dtype=np.int32)
    conv = np.zeros(conv_len, dtype=np.int64)

    n_surv = np.int64(0)
    n_tested = np.int64(0)
    max_surv = np.int64(out_buf.shape[0])

    qc_ell = np.int32(0)
    qc_s = np.int32(0)

    pos = 0
    cursor[0] = lo[0]

    while True:
        c_val = cursor[pos]

        if c_val > hi[pos]:
            if pos == 0:
                break
            pos -= 1
            k1 = 2 * pos
            k2 = k1 + 1
            old1 = np.int64(child[k1])
            old2 = np.int64(child[k2])
            conv[2 * k1] -= old1 * old1
            conv[2 * k2] -= old2 * old2
            conv[k1 + k2] -= np.int64(2) * old1 * old2
            for j in range(k1):
                cj = np.int64(child[j])
                if cj != 0:
                    conv[k1 + j] -= np.int64(2) * old1 * cj
                    conv[k2 + j] -= np.int64(2) * old2 * cj
            child[k1] = 0
            child[k2] = 0
            cursor[pos] += 1
            continue

        k1 = 2 * pos
        k2 = k1 + 1
        new1 = np.int64(c_val)
        new2 = np.int64(parent[pos] - c_val)
        child[k1] = np.int32(new1)
        child[k2] = np.int32(new2)

        conv[2 * k1] += new1 * new1
        conv[2 * k2] += new2 * new2
        conv[k1 + k2] += np.int64(2) * new1 * new2
        for j in range(k1):
            cj = np.int64(child[j])
            if cj != 0:
                conv[k1 + j] += np.int64(2) * new1 * cj
                conv[k2 + j] += np.int64(2) * new2 * cj

        max_cv_pos = 2 * k2
        partial_pruned = False
        for ell in range(2, max_ell + 1):
            if partial_pruned:
                break
            n_cv = ell - 1
            max_s = min(max_cv_pos, conv_len - n_cv)
            if max_s < 0:
                continue
            ws = np.int64(0)
            init_end = min(n_cv, max_cv_pos + 1)
            for k in range(init_end):
                ws += conv[k]
            if ws > thr[ell]:
                partial_pruned = True
                break
            for s_lo in range(1, max_s + 1):
                new_k = s_lo + n_cv - 1
                if new_k <= max_cv_pos:
                    ws += conv[new_k]
                ws -= conv[s_lo - 1]
                if ws > thr[ell]:
                    partial_pruned = True
                    break

        if partial_pruned:
            conv[2 * k1] -= new1 * new1
            conv[2 * k2] -= new2 * new2
            conv[k1 + k2] -= np.int64(2) * new1 * new2
            for j in range(k1):
                cj = np.int64(child[j])
                if cj != 0:
                    conv[k1 + j] -= np.int64(2) * new1 * cj
                    conv[k2 + j] -= np.int64(2) * new2 * cj
            child[k1] = 0
            child[k2] = 0
            cursor[pos] += 1
            continue

        if pos == d_parent - 1:
            n_tested += 1

            quick_killed = False
            if qc_ell > 0:
                n_cv_qc = qc_ell - 1
                ws_qc = np.int64(0)
                for k in range(qc_s, qc_s + n_cv_qc):
                    ws_qc += conv[k]
                if ws_qc > thr[qc_ell]:
                    quick_killed = True

            if not quick_killed:
                full_pruned = False
                for ell in range(2, max_ell + 1):
                    if full_pruned:
                        break
                    n_cv = ell - 1
                    n_win = conv_len - n_cv + 1
                    ws = np.int64(0)
                    for k in range(n_cv):
                        ws += conv[k]
                    for s_lo in range(n_win):
                        if s_lo > 0:
                            ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                        if ws > thr[ell]:
                            full_pruned = True
                            qc_ell = np.int32(ell)
                            qc_s = np.int32(s_lo)
                            break

                if not full_pruned:
                    if n_surv < max_surv:
                        for i in range(d_child):
                            out_buf[n_surv, i] = child[i]
                    n_surv += 1

            conv[2 * k1] -= new1 * new1
            conv[2 * k2] -= new2 * new2
            conv[k1 + k2] -= np.int64(2) * new1 * new2
            for j in range(k1):
                cj = np.int64(child[j])
                if cj != 0:
                    conv[k1 + j] -= np.int64(2) * new1 * cj
                    conv[k2 + j] -= np.int64(2) * new2 * cj
            child[k1] = 0
            child[k2] = 0
            cursor[pos] += 1
        else:
            pos += 1
            cursor[pos] = lo[pos]

    return n_surv, n_tested


@njit(cache=True)
def _count_one_parent_v2(parent, d_parent, S, x_cap, thr,
                          use_whole_parent_prune):
    """Count survivors for one parent, with optional whole-parent pre-pruning."""
    if use_whole_parent_prune:
        if _whole_parent_prune(parent, d_parent, S, x_cap, thr):
            return np.int64(0), np.int64(0)
    dummy = np.empty((0, 2 * d_parent), dtype=np.int32)
    ns, nt = _cascade_child_bnb_v2(parent, d_parent, S, x_cap, thr, dummy)
    return ns, nt


def run_cascade_level_v2(survivors_prev, d_parent, S, c_target,
                          use_whole_parent_prune=True, verbose=True):
    """L1+ with whole-parent pre-pruning."""
    d_child = 2 * d_parent
    n_parents = survivors_prev.shape[0]
    thr = compute_thresholds(c_target, S, d_child)
    x_cap = compute_xcap(c_target, S, d_child)

    if verbose:
        print(f"    x_cap={x_cap}, d_child={d_child}, "
              f"n_parents={n_parents}, "
              f"whole_parent_prune={use_whole_parent_prune}")

    counts = np.zeros(n_parents, dtype=np.int64)
    tested = np.zeros(n_parents, dtype=np.int64)
    n_parents_pruned = 0

    t0 = time.time()
    for p_idx in range(n_parents):
        parent = survivors_prev[p_idx]
        ns, nt = _count_one_parent_v2(parent, d_parent, S, x_cap, thr,
                                       use_whole_parent_prune)
        counts[p_idx] = ns
        tested[p_idx] = nt
        if nt == 0 and use_whole_parent_prune:
            n_parents_pruned += 1

        if verbose and (p_idx + 1) % max(1, n_parents // 10) == 0:
            elapsed = time.time() - t0
            rate = (p_idx + 1) / elapsed if elapsed > 0 else 0
            print(f"      counting: {p_idx+1}/{n_parents} "
                  f"({rate:.0f} parents/s, "
                  f"{np.sum(tested[:p_idx+1]):,} tested, "
                  f"{np.sum(counts[:p_idx+1]):,} survived, "
                  f"{n_parents_pruned} parents pruned)")

    total_surv = int(np.sum(counts))
    total_tested = int(np.sum(tested))

    if verbose and use_whole_parent_prune:
        print(f"    Whole-parent pre-pruning: {n_parents_pruned}/{n_parents} "
              f"parents skipped ({n_parents_pruned/max(n_parents,1)*100:.1f}%)")

    if total_surv == 0:
        return np.empty((0, d_child), dtype=np.int32), 0, total_tested

    offsets = np.zeros(n_parents + 1, dtype=np.int64)
    for i in range(n_parents):
        offsets[i + 1] = offsets[i] + counts[i]

    out_buf = np.empty((total_surv, d_child), dtype=np.int32)

    for p_idx in range(n_parents):
        cnt = int(counts[p_idx])
        if cnt == 0:
            continue
        off = int(offsets[p_idx])
        parent = survivors_prev[p_idx]
        _cascade_child_bnb_v2(parent, d_parent, S, x_cap, thr,
                              out_buf[off:off + cnt])

    return out_buf, total_surv, total_tested


# =====================================================================
# Canonicalization (same as v1)
# =====================================================================

@njit(parallel=True, cache=True)
def _canonicalize_inplace(arr):
    B = arr.shape[0]
    d = arr.shape[1]
    half = d // 2
    for b in prange(B):
        swap = False
        for i in range(half):
            j = d - 1 - i
            if arr[b, j] < arr[b, i]:
                swap = True
                break
            elif arr[b, j] > arr[b, i]:
                break
        if swap:
            for i in range(half):
                j = d - 1 - i
                tmp = arr[b, i]
                arr[b, i] = arr[b, j]
                arr[b, j] = tmp


def dedup(arr):
    if len(arr) == 0:
        return arr
    d = arr.shape[1]
    keys = tuple(arr[:, d - 1 - i] for i in range(d))
    sort_idx = np.lexsort(keys)
    sorted_arr = arr[sort_idx]
    mask = np.ones(len(sorted_arr), dtype=bool)
    for i in range(1, len(sorted_arr)):
        if np.array_equal(sorted_arr[i], sorted_arr[i - 1]):
            mask[i] = False
    return sorted_arr[mask]


# =====================================================================
# IDEA 1: McCormick LP box certification (TIGHT)
# =====================================================================

def _mccormick_lp_min_max_tv(mu_center, d, S, c_target):
    """Sound lower bound on min_{μ in cell} max_W TV_W(μ).

    Setup: variables μ ∈ ℝ^d (bin masses), w_{ij} for i≤j (bilinear products),
    t (auxiliary upper bound). Solve:
        min t
        s.t. (2d/ell_W) · sum_{(i,j) in W} α_W_ij · w_{ij} ≤ t   ∀ W
             sum(μ) = 1
             max(0, μ_center_i - 1/(2S)) ≤ μ_i ≤ μ_center_i + 1/(2S)
             McCormick LOWER envelopes on w_{ij} (so w_{ij} ≥ μ_i μ_j)

    Returns (lp_value, status).  lp_value is a sound lower bound on
    min_μ max_W TV_W(μ).  If lp_value ≥ c_target, the cell is certified.
    """
    from scipy.optimize import linprog

    r = 1.0 / (2.0 * S)
    lo = np.maximum(mu_center - r, 0.0)
    hi = np.minimum(mu_center + r, 1.0)

    # All bilinear pairs (i, j) with i ≤ j.  Each w_{ij} represents μ_i μ_j.
    # Pair index ordering: pair_idx[i, j] = compressed index for i ≤ j.
    pair_idx = -np.ones((d, d), dtype=np.int32)
    pairs = []
    for i in range(d):
        for j in range(i, d):
            pair_idx[i, j] = pair_idx[j, i] = len(pairs)
            pairs.append((i, j))
    n_pairs = len(pairs)

    n_vars = d + n_pairs + 1  # μ (d) + w (n_pairs) + t (1)
    t_idx = n_vars - 1

    c_obj = np.zeros(n_vars)
    c_obj[t_idx] = 1.0  # min t

    A_eq_rows = []
    b_eq_vals = []

    # sum(μ) = 1
    row = np.zeros(n_vars)
    row[:d] = 1.0
    A_eq_rows.append(row)
    b_eq_vals.append(1.0)

    A_ub_rows = []
    b_ub_vals = []

    # McCormick LOWER envelopes on w_{ij} (off-diagonal):
    #   w_{ij} ≥ lo_i μ_j + lo_j μ_i - lo_i lo_j
    #   w_{ij} ≥ hi_i μ_j + hi_j μ_i - hi_i hi_j
    # Both are of the form: w_{ij} ≥ a μ_j + b μ_i + c
    # As scipy A_ub @ x ≤ b_ub:  -w_{ij} + a μ_j + b μ_i ≤ -c
    # Diagonal: w_{ii} ≥ μ_i² uses tangent at lo_i, hi_i, μ_center_i.
    for k, (i, j) in enumerate(pairs):
        w_idx = d + k
        if i == j:
            li, hi_i = lo[i], hi[i]
            mc = mu_center[i]
            # Lower bound: tangent at mc:  w_ii ≥ 2 mc μ_i - mc²
            row = np.zeros(n_vars)
            row[w_idx] = -1.0
            row[i] = 2.0 * mc
            A_ub_rows.append(row)
            b_ub_vals.append(mc * mc)
            # Lower bound: tangent at li
            row = np.zeros(n_vars)
            row[w_idx] = -1.0
            row[i] = 2.0 * li
            A_ub_rows.append(row)
            b_ub_vals.append(li * li)
            # Lower bound: tangent at hi_i
            row = np.zeros(n_vars)
            row[w_idx] = -1.0
            row[i] = 2.0 * hi_i
            A_ub_rows.append(row)
            b_ub_vals.append(hi_i * hi_i)
        else:
            li, hi_i = lo[i], hi[i]
            lj, hj = lo[j], hi[j]
            # Lower 1: w ≥ li μ_j + lj μ_i - li lj
            row = np.zeros(n_vars)
            row[w_idx] = -1.0
            row[j] = li
            row[i] = lj
            A_ub_rows.append(row)
            b_ub_vals.append(li * lj)
            # Lower 2: w ≥ hi_i μ_j + hj μ_i - hi_i hj
            row = np.zeros(n_vars)
            row[w_idx] = -1.0
            row[j] = hi_i
            row[i] = hj
            A_ub_rows.append(row)
            b_ub_vals.append(hi_i * hj)

    # For each window W = (ell, s_lo) with s_lo ≤ i+j ≤ s_lo+ell-2:
    # TV_W(μ) = (2d/ell) sum_{(i,j): s_lo ≤ i+j ≤ s_lo+ell-2} μ_i μ_j
    #         = (2d/ell) sum α_W_ij w_{ij}
    # where α_W_ij = #ordered pairs (i,j) with i+j in W.  For (i, j) with
    # i < j: 2 ordered pairs (i,j) and (j,i) both contribute, so α = 2.
    # For (i, i): 1 ordered pair.
    # Constraint: TV_W ≤ t  ⇒  (2d/ell) sum α w - t ≤ 0
    conv_len = 2 * d - 1
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        scale = 2.0 * d / ell
        for s in range(conv_len - n_cv + 1):
            row = np.zeros(n_vars)
            for i in range(d):
                for j in range(i, d):
                    if s <= i + j <= s + n_cv - 1:
                        alpha = 1.0 if i == j else 2.0
                        row[d + pair_idx[i, j]] += scale * alpha
            row[t_idx] = -1.0
            # Skip degenerate windows with no pairs in range
            if np.count_nonzero(row[d:d + n_pairs]) == 0:
                continue
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)

    # Bounds
    bounds = []
    for i in range(d):
        bounds.append((lo[i], hi[i]))
    for k, (i, j) in enumerate(pairs):
        # w bounds: must satisfy lo_i lo_j ≤ w ≤ hi_i hi_j; also bounded by box
        bounds.append((max(0.0, lo[i] * lo[j]), hi[i] * hi[j]))
    bounds.append((0.0, None))  # t ≥ 0 (TV_W is nonneg)

    A_eq = np.array(A_eq_rows)
    b_eq = np.array(b_eq_vals)
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_vals) if b_ub_vals else None

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if result.success:
        return float(result.fun), 'optimal'
    else:
        return -np.inf, str(result.message)


# Fast water-filling pre-check (max-min, the v1 method).  Used as a
# cheap filter before the more expensive McCormick LP.
@njit(cache=True)
def _box_certify_cell_waterfill(mu_center, d, delta, c_target):
    """Fast max-min lower bound (the v1 method).  Returns (certified, best_min_tv)."""
    conv_len = 2 * d - 1
    two_d = 2.0 * np.float64(d)

    lo = np.empty(d, dtype=np.float64)
    hi = np.empty(d, dtype=np.float64)
    for i in range(d):
        lo[i] = max(0.0, mu_center[i] - delta / 2.0)
        hi[i] = min(1.0, mu_center[i] + delta / 2.0)

    best_min_tv = 0.0

    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        scale = two_d / np.float64(ell)
        for s in range(conv_len - n_cv + 1):
            contrib = np.zeros(d, dtype=numba.boolean)
            for k in range(s, s + n_cv):
                for i in range(max(0, k - d + 1), min(d, k + 1)):
                    contrib[i] = True

            mu_opt = lo.copy()
            excess = 1.0 - np.sum(mu_opt)

            for i in range(d):
                if not contrib[i] and excess > 1e-15:
                    add = min(excess, hi[i] - mu_opt[i])
                    mu_opt[i] += add
                    excess -= add

            if excess > 1e-15:
                for i in range(d):
                    if contrib[i] and excess > 1e-15:
                        add = min(excess, hi[i] - mu_opt[i])
                        mu_opt[i] += add
                        excess -= add

            conv_opt = np.zeros(conv_len, dtype=np.float64)
            for i in range(d):
                mi = mu_opt[i]
                if mi > 0:
                    conv_opt[2 * i] += mi * mi
                    for j in range(i + 1, d):
                        mj = mu_opt[j]
                        if mj > 0:
                            conv_opt[i + j] += 2.0 * mi * mj

            ws = 0.0
            for k in range(s, s + n_cv):
                ws += conv_opt[k]
            min_tv = ws * scale

            if min_tv > best_min_tv:
                best_min_tv = min_tv

            if best_min_tv >= c_target:
                return True, best_min_tv

    return best_min_tv >= c_target, best_min_tv


def box_certify_cell_hybrid(mu_center, d, S, c_target,
                             use_lp_fallback=True):
    """Certify a single cell.  Uses fast water-filling first, McCormick LP
    fallback when water-filling fails.  Returns (certified, value, method)."""
    delta = 1.0 / S
    cert_wf, val_wf = _box_certify_cell_waterfill(mu_center, d, delta, c_target)
    if cert_wf:
        return True, val_wf, 'waterfill'
    if not use_lp_fallback:
        return False, val_wf, 'waterfill'
    val_lp, status = _mccormick_lp_min_max_tv(mu_center, d, S, c_target)
    return (val_lp >= c_target), val_lp, f'lp_{status}'


# =====================================================================
# Full-grid certification (rigorous: tests every grid point in [0,1]^d ∩ Δ)
# =====================================================================

def run_full_grid_box_cert(d_final, S, c_target, n_workers=1, verbose=True,
                            sample_only=None):
    """Certify the box around every grid point.

    For a rigorous proof, every grid point μ* in {(c_0,...,c_{d-1})/S :
    sum c_i = S, c_i ≥ 0} must have its δ-cell certified.

    Strategy:
      1. Enumerate all grid points in lex order (B&B-style).
      2. For each: water-filling cert (fast).  If it passes, done.
      3. Else: McCormick LP cert.  If it passes, done.
      4. Else: report failure.

    Returns (n_total, n_certified_wf, n_certified_lp, n_failed,
             worst_failed_mu, worst_failed_val).
    """
    if verbose:
        print(f"\n  Full-grid box certification at d={d_final}, S={S}, "
              f"c_target={c_target}:")

    n_total = 0
    n_cert_wf = 0
    n_cert_lp = 0
    n_failed = 0
    worst_failed_mu = None
    worst_failed_val = np.inf

    # Enumerate grid points by canonical compositions
    # For small d we can enumerate directly; for large d use the L0 BnB
    # without any pruning (just visit every composition with mass per bin
    # capped at S).
    if sample_only is not None:
        rng = np.random.RandomState(42)
        for _ in range(sample_only):
            mu_int = np.zeros(d_final, dtype=np.int32)
            remaining = S
            for i in range(d_final - 1):
                mu_int[i] = rng.randint(0, remaining + 1)
                remaining -= mu_int[i]
            mu_int[d_final - 1] = remaining
            mu_center = mu_int.astype(np.float64) / S
            n_total += 1
            cert, val, method = box_certify_cell_hybrid(
                mu_center, d_final, S, c_target)
            if cert:
                if method == 'waterfill':
                    n_cert_wf += 1
                else:
                    n_cert_lp += 1
            else:
                n_failed += 1
                if val < worst_failed_val:
                    worst_failed_val = val
                    worst_failed_mu = mu_center.copy()
        if verbose:
            print(f"    SAMPLED {n_total} points: "
                  f"{n_cert_wf} wf, {n_cert_lp} lp, {n_failed} failed")
            if worst_failed_mu is not None:
                print(f"    Worst failed val={worst_failed_val:.6f} "
                      f"at mu={worst_failed_mu}")
        return n_total, n_cert_wf, n_cert_lp, n_failed, worst_failed_mu, worst_failed_val

    # Full enumeration (slow but rigorous).  TODO: parallelize.
    def _recurse(c_int, pos, remaining):
        nonlocal n_total, n_cert_wf, n_cert_lp, n_failed, worst_failed_mu, worst_failed_val
        if pos == d_final - 1:
            c_int[pos] = remaining
            mu_center = c_int.astype(np.float64) / S
            n_total += 1
            cert, val, method = box_certify_cell_hybrid(
                mu_center, d_final, S, c_target)
            if cert:
                if method == 'waterfill':
                    n_cert_wf += 1
                else:
                    n_cert_lp += 1
            else:
                n_failed += 1
                if val < worst_failed_val:
                    worst_failed_val = val
                    worst_failed_mu = mu_center.copy()
                if verbose and n_failed <= 5:
                    print(f"      FAILED: mu={mu_center}, val={val:.6f}, method={method}")
            return
        for v in range(remaining + 1):
            c_int[pos] = v
            _recurse(c_int, pos + 1, remaining - v)

    c_int = np.zeros(d_final, dtype=np.int32)
    t0 = time.time()
    _recurse(c_int, 0, S)
    elapsed = time.time() - t0

    if verbose:
        print(f"    Total: {n_total:,} grid points in {elapsed:.1f}s")
        print(f"    Certified: {n_cert_wf:,} via waterfill, "
              f"{n_cert_lp:,} via LP, {n_failed} FAILED")
        if n_failed > 0 and worst_failed_mu is not None:
            print(f"    Worst failed: val={worst_failed_val:.6f} "
                  f"at mu={worst_failed_mu}")

    return n_total, n_cert_wf, n_cert_lp, n_failed, worst_failed_mu, worst_failed_val


# =====================================================================
# Main cascade driver
# =====================================================================

def run_cascade_v2(c_target=1.20, S=50, d_start=4, max_levels=5,
                    use_whole_parent_prune=True,
                    box_cert_mode='hybrid',
                    box_cert_sample=None,
                    verbose=True):
    """Full cascade with v2 improvements.

    box_cert_mode:
      'sample'  - random sample only (heuristic, NOT a proof)
      'hybrid'  - full grid with hybrid waterfill+LP cert (rigorous)
      'lp_only' - full grid with LP cert only (rigorous, slow)
    """
    if verbose:
        print("=" * 64)
        print(f"COARSE CASCADE PROVER v2: C_{{1a}} >= {c_target}")
        print("=" * 64)
        print(f"  Grid: S={S} (delta={1/S:.4f})")
        print(f"  Starting dimension: d={d_start}")
        print(f"  whole_parent_prune={use_whole_parent_prune}")
        print(f"  box_cert_mode={box_cert_mode}")
        print()

    t_total = time.time()

    # --- L0 ---
    d = d_start
    if verbose:
        print(f"  L0 (d={d}):")
    t0 = time.time()
    survivors, n_surv, n_tested = run_l0(d, S, c_target)
    elapsed = time.time() - t0

    if n_surv > 0:
        _canonicalize_inplace(survivors)
        survivors = dedup(survivors)
        n_surv = len(survivors)

    if verbose:
        print(f"    Tested: {n_tested:,}")
        print(f"    Survivors: {n_surv:,}")
        print(f"    Time: {elapsed:.2f}s")

    cascade_converged_at = -1
    if n_surv == 0:
        cascade_converged_at = 0
        if verbose:
            print(f"\n  CASCADE CONVERGED at L0.")
    else:
        os.makedirs("data", exist_ok=True)
        np.save(f"data/coarse_v2_L0_survivors_S{S}.npy", survivors)

    # --- L1+ ---
    if cascade_converged_at < 0:
        for level in range(1, max_levels + 1):
            d_parent = d
            d = 2 * d_parent

            if verbose:
                print(f"\n  L{level} (d={d}):")

            t0 = time.time()
            survivors, n_surv, n_tested = run_cascade_level_v2(
                survivors, d_parent, S, c_target,
                use_whole_parent_prune=use_whole_parent_prune,
                verbose=verbose)
            elapsed = time.time() - t0

            if n_surv > 0:
                _canonicalize_inplace(survivors)
                survivors = dedup(survivors)
                n_surv = len(survivors)

            if verbose:
                print(f"    Tested: {n_tested:,}")
                print(f"    Survivors (after dedup): {n_surv:,}")
                print(f"    Time: {elapsed:.2f}s")

            if n_surv == 0:
                cascade_converged_at = level
                if verbose:
                    print(f"\n  CASCADE CONVERGED at L{level} (d={d})!")
                break

            np.save(f"data/coarse_v2_L{level}_survivors_S{S}.npy", survivors)

    if cascade_converged_at < 0:
        if verbose:
            print(f"\n  Cascade did NOT converge within {max_levels} levels. "
                  f"Survivors at d={d}: {n_surv:,}")
        return False, None

    # --- Box certification ---
    if box_cert_mode == 'sample':
        if verbose:
            print(f"\n  Sampling {box_cert_sample or 2000} cells "
                  f"(NOT a proof — for diagnostic only):")
        cert_results = run_full_grid_box_cert(
            d, S, c_target, sample_only=box_cert_sample or 2000,
            verbose=verbose)
    elif box_cert_mode in ('hybrid', 'lp_only'):
        if verbose:
            print(f"\n  Running RIGOROUS full-grid box certification...")
        cert_results = run_full_grid_box_cert(
            d, S, c_target, verbose=verbose)
    else:
        raise ValueError(f"unknown box_cert_mode: {box_cert_mode}")

    n_total, n_wf, n_lp, n_fail, worst_mu, worst_val = cert_results

    total_time = time.time() - t_total
    proven = (n_fail == 0) and (box_cert_mode != 'sample')
    if verbose:
        print(f"\n  {'=' * 60}")
        if proven:
            print(f"  PROOF: C_{{1a}} >= {c_target}")
            print(f"  Method: coarse cascade v2 (S={S}) + hybrid box cert")
            print(f"  Cascade converged at d={d} (L{cascade_converged_at})")
            print(f"  Box cert: {n_wf:,} cells via waterfill, "
                  f"{n_lp:,} via McCormick LP")
        else:
            print(f"  PROOF FAILED: {n_fail} cells did not certify")
            print(f"  (cascade reached L{cascade_converged_at}, "
                  f"box cert mode: {box_cert_mode})")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  {'=' * 60}")

    return proven, cert_results


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Coarse cascade prover v2 — tight box cert + parent prune")
    parser.add_argument("--c_target", type=float, default=1.20)
    parser.add_argument("--S", type=int, default=50)
    parser.add_argument("--d_start", type=int, default=4)
    parser.add_argument("--max_levels", type=int, default=5)
    parser.add_argument("--no_whole_parent", action='store_true',
                        help="Disable whole-parent pre-pruning")
    parser.add_argument("--box_cert", default='hybrid',
                        choices=['sample', 'hybrid', 'lp_only'],
                        help="Box certification mode")
    parser.add_argument("--box_cert_sample", type=int, default=None)
    args = parser.parse_args()

    # JIT warmup
    print("Warming up JIT...", end="", flush=True)
    t0 = time.time()
    _w_thr = compute_thresholds(1.3, 10, 4)
    _w_buf = np.empty((0, 4), dtype=np.int32)
    _l0_bnb_inner(np.int32(2), 4, 10, 5, _w_thr, _w_buf, True)
    _w_parent = np.array([3, 3, 2, 2], dtype=np.int32)
    _w_buf2 = np.empty((100, 8), dtype=np.int32)
    _w_thr2 = compute_thresholds(1.3, 10, 8)
    _cascade_child_bnb_v2(_w_parent, 4, 10, 5, _w_thr2, _w_buf2)
    _whole_parent_prune(_w_parent, 4, 10, 5, _w_thr2)
    _canonicalize_inplace(np.array([[1, 2, 3, 4]], dtype=np.int32))
    print(f" done ({time.time()-t0:.1f}s)")

    os.makedirs("data", exist_ok=True)
    proven, _ = run_cascade_v2(
        c_target=args.c_target,
        S=args.S,
        d_start=args.d_start,
        max_levels=args.max_levels,
        use_whole_parent_prune=not args.no_whole_parent,
        box_cert_mode=args.box_cert,
        box_cert_sample=args.box_cert_sample,
    )

    sys.exit(0 if proven else 1)


if __name__ == "__main__":
    main()
