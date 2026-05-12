"""Hybrid coarse + fine-grid cascade box-certifier.

When the COARSE cascade (`run_cascade_coarse_v5.py`) cannot certify a cell
via N+O+Joint+Shor (because h_coarse = 1/(2S) is too large for the residual
TV-margin), we fall back to the FINE-GRID cascade restricted to the coarse
cell.  The fine grid uses h_fine = 1/(2*S_fine) where S_fine = 2*d*m_target,
which can be made arbitrarily small by raising m_target.

==========================================================================
MAPPING (coarse cell  →  fine sub-problem)
==========================================================================
COARSE cell (input):
    c_int ∈ Z^d, Σ c_int = S, μ_i ∈ [c_int[i]/S − 1/(2S), c_int[i]/S + 1/(2S)]
                                with Σμ = 1.

FINE grid (output):
    c ∈ Z^d, Σ c = S_fine := 2*d*m_target,  μ_i = c[i]/S_fine,
    each fine "cell" has half-width h_fine = 1/(2*S_fine).

Choose m_target so S_fine = K*S for some integer K ≥ 1, i.e.
    K = S_fine // S = 2*d*m_target // S.

For each coarse bin i, the coarse μ-range [c_int[i]/S − 1/(2S),
c_int[i]/S + 1/(2S)] corresponds to fine c[i] in
    [K*c_int[i] − K/2,  K*c_int[i] + K/2]
which is K+1 consecutive integers when K is even (centered on K*c_int[i])
or K integers when K is odd.

Coverage: the union of fine cells with c[i] in this range covers the
coarse cell exactly (every μ ∈ coarse cell has c[i] = round(μ*S_fine)).

==========================================================================
SOUNDNESS
==========================================================================
The fine cascade (F → FN → Q → QN → L) is independently sound: each fine
cell c is certified iff TV(μ) ≥ c_target on the WHOLE fine cell (not just
at the grid point), because the F threshold uses
    floor((c_target*m² + 1 + W_int/(2n)) * 4n*ell)
or with FN/Q/QN/L tightenings, which absorb the per-cell δ-error.

If EVERY fine cell intersecting the coarse cell is pruned by the chain,
the coarse cell is fully covered ⇒ coarse cell is rigorously certified.

==========================================================================
USAGE
==========================================================================
    from _hybrid_cascade import fine_grid_box_cert
    ok, info = fine_grid_box_cert(c_int, S, d, c_target, m_target=20)
"""
from __future__ import annotations
import os
import sys
import time
from math import gcd

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_THIS, 'cloninger-steinerberger', 'cpu'))

from run_cascade import _prune_dynamic
from post_filters import (apply_FN_filter,
                          apply_Q_filter,
                          apply_QN_filter,
                          apply_L_filter)


# =====================================================================
# Helpers: pick a clean (m_target, S_fine, K) and enumerate the cell
# =====================================================================

def pick_m_target(S, d, m_target_min):
    """Smallest m ≥ m_target_min such that 2*d*m is a multiple of S.

    This guarantees S_fine = 2*d*m  =  K * S for some integer K, so each
    coarse bin maps to exactly K+1 (K even) or K (K odd) fine integers.

    Period of m's for which 2*d*m % S == 0:  m_period = S / gcd(2*d, S).
    """
    base = S // gcd(2 * d, S)              # smallest m with 2*d*m % S == 0
    if base == 0:
        base = 1
    if m_target_min <= base:
        return base
    # Next multiple of base that's ≥ m_target_min.
    q = (m_target_min + base - 1) // base
    return q * base


def enumerate_fine_compositions_in_coarse_cell(c_int, S, d, m_target):
    """Enumerate all fine integer compositions c ∈ Z^d with
        Σc = 2*d*m_target,
        c[i] ∈ [K*c_int[i] − K/2, K*c_int[i] + K/2]   (covers coarse cell)
    where K = (2*d*m_target) // S.

    Returns (compositions int32 array (N, d), K).  Sum constraint enforced.
    """
    S_fine = 2 * d * m_target
    if S_fine % S != 0:
        raise ValueError(f"S_fine={S_fine} not multiple of S={S}; "
                         f"call pick_m_target first")
    K = S_fine // S

    # Per-bin range [lo_i, hi_i] inclusive, centered on K*c_int[i]
    # such that fine cell at c[i] covers μ_i ∈ [c[i]/S_fine − 1/(2 S_fine),
    # c[i]/S_fine + 1/(2 S_fine)] and the union covers the coarse interval
    # μ_i ∈ [c_int[i]/S − 1/(2S), c_int[i]/S + 1/(2S)].
    #
    # Fine cell centered at c[i]/S_fine = (K*c_int[i] + j)/(K*S) for some
    # j ∈ {-K/2, …, K/2}.  Clip to [0, S_fine] (non-negativity).
    K_half_lo = K // 2          # K/2 floor
    K_half_hi = K - K_half_lo   # K/2 ceil
    lo_arr = np.empty(d, dtype=np.int64)
    hi_arr = np.empty(d, dtype=np.int64)
    for i in range(d):
        center = int(K) * int(c_int[i])
        lo = center - K_half_lo
        hi = center + K_half_hi
        # Clip to [0, S_fine]
        if lo < 0:
            lo = 0
        if hi > S_fine:
            hi = S_fine
        if lo > hi:
            return np.empty((0, d), dtype=np.int32), K
        lo_arr[i] = lo
        hi_arr[i] = hi

    # Recursive enumeration of compositions with c_i ∈ [lo_i, hi_i] and
    # Σc = S_fine.  d is small (d ≤ 16 in practice), and the per-bin
    # range has width K (typically ≤ 4), so total ≤ (K+1)^d compositions
    # before sum filter — easily feasible.
    out = []
    cur = np.empty(d, dtype=np.int64)

    def _rec(i, remaining):
        if i == d - 1:
            if lo_arr[i] <= remaining <= hi_arr[i]:
                cur[i] = remaining
                out.append(cur.copy())
            return
        # Bound remaining by what later bins can absorb.
        # min total of bins i+1..d-1 is sum(lo_arr[i+1:])
        # max total is sum(hi_arr[i+1:])
        min_after = int(lo_arr[i + 1:].sum())
        max_after = int(hi_arr[i + 1:].sum())
        c_lo = max(int(lo_arr[i]), int(remaining - max_after))
        c_hi = min(int(hi_arr[i]), int(remaining - min_after))
        for v in range(c_lo, c_hi + 1):
            cur[i] = v
            _rec(i + 1, remaining - v)

    _rec(0, S_fine)
    if not out:
        return np.empty((0, d), dtype=np.int32), K
    arr = np.array(out, dtype=np.int32)
    return arr, K


# =====================================================================
# Main entry: fine_grid_box_cert
# =====================================================================

def fine_grid_box_cert(c_int, S, d, c_target, m_target=20,
                       use_FN=True, use_Q=True, use_QN=True, use_L=True,
                       n_workers=1, verbose=False):
    """Certify a coarse cell via fine-grid F → FN → Q → QN → L cascade.

    Parameters
    ----------
    c_int : array-like length d, integer.  Coarse cell center, Σ = S.
    S, d, c_target : coarse cascade parameters.
    m_target : minimum fine-grid m.  Final m may be larger so 2*d*m % S == 0.
    use_FN, use_Q, use_QN, use_L : enable post-filter stages.
    n_workers : worker count for Q/QN/L filters (1 = sequential, fast).

    Returns
    -------
    ok : bool — True iff every fine composition in the cell is pruned by
                some stage of the cascade (coarse cell rigorously certified).
    info : dict with timing, stage counts, m_target_used, K, n_fine_comps,
           survivors_at_each_stage, residual (if any).
    """
    c_int = np.asarray(c_int, dtype=np.int32).reshape(-1)
    if len(c_int) != d:
        raise ValueError(f"c_int length {len(c_int)} != d={d}")
    if int(c_int.sum()) != S:
        raise ValueError(f"sum(c_int)={c_int.sum()} != S={S}")

    info = {
        'd': int(d), 'S': int(S), 'c_target': float(c_target),
        'c_int': c_int.tolist(),
        'm_target_requested': int(m_target),
    }
    t0 = time.time()

    # Step 1: pick clean m_target
    m_used = pick_m_target(S, d, m_target)
    info['m_target_used'] = int(m_used)
    n_half = d / 2.0  # may be float for odd d
    info['n_half'] = float(n_half)
    info['S_fine'] = int(2 * d * m_used)

    # Step 2: enumerate fine compositions
    fine_comps, K = enumerate_fine_compositions_in_coarse_cell(
        c_int, S, d, m_used)
    info['K'] = int(K)
    info['n_fine_comps'] = int(len(fine_comps))
    if verbose:
        print(f"  [hybrid] coarse cell c={c_int.tolist()} S={S} d={d}", flush=True)
        print(f"  [hybrid] m_target={m_used}, K={K}, "
              f"n_fine_comps={len(fine_comps)}", flush=True)

    if len(fine_comps) == 0:
        info['ok'] = True
        info['reason'] = 'empty_cell_no_compositions'
        info['wall_sec'] = round(time.time() - t0, 4)
        return True, info

    # Step 3: F filter (numba parallel kernel)
    t_F = time.time()
    n_half_int = int(n_half) if float(n_half).is_integer() else n_half
    f_mask = _prune_dynamic(fine_comps, n_half_int, m_used, c_target,
                            use_flat_threshold=False, use_F=True)
    n_F_pruned = int((~f_mask).sum())
    surv = fine_comps[f_mask]
    info['n_F_pruned'] = n_F_pruned
    info['n_after_F'] = int(len(surv))
    info['time_F'] = round(time.time() - t_F, 4)
    if verbose:
        print(f"  [hybrid]  F: pruned {n_F_pruned}, surv {len(surv)}", flush=True)
    if len(surv) == 0:
        info['ok'] = True
        info['reason'] = 'closed_at_F'
        info['wall_sec'] = round(time.time() - t0, 4)
        return True, info

    # Step 4: FN filter
    if use_FN:
        t_FN = time.time()
        try:
            surv = apply_FN_filter(surv, n_half_int, m_used, c_target)
        except Exception as e:
            info['fn_error'] = str(e)
        info['n_after_FN'] = int(len(surv))
        info['time_FN'] = round(time.time() - t_FN, 4)
        if verbose:
            print(f"  [hybrid] FN: surv {len(surv)}", flush=True)
        if len(surv) == 0:
            info['ok'] = True
            info['reason'] = 'closed_at_FN'
            info['wall_sec'] = round(time.time() - t0, 4)
            return True, info

    # Step 5: Q filter
    if use_Q:
        t_Q = time.time()
        try:
            surv = apply_Q_filter(surv, n_half_int, m_used, c_target)
        except Exception as e:
            info['q_error'] = str(e)
        info['n_after_Q'] = int(len(surv))
        info['time_Q'] = round(time.time() - t_Q, 4)
        if verbose:
            print(f"  [hybrid]  Q: surv {len(surv)}", flush=True)
        if len(surv) == 0:
            info['ok'] = True
            info['reason'] = 'closed_at_Q'
            info['wall_sec'] = round(time.time() - t0, 4)
            return True, info

    # Step 6: QN filter
    if use_QN:
        t_QN = time.time()
        try:
            surv = apply_QN_filter(surv, n_half_int, m_used, c_target)
        except Exception as e:
            info['qn_error'] = str(e)
        info['n_after_QN'] = int(len(surv))
        info['time_QN'] = round(time.time() - t_QN, 4)
        if verbose:
            print(f"  [hybrid] QN: surv {len(surv)}", flush=True)
        if len(surv) == 0:
            info['ok'] = True
            info['reason'] = 'closed_at_QN'
            info['wall_sec'] = round(time.time() - t0, 4)
            return True, info

    # Step 7: L filter (Shor SDP)
    if use_L:
        t_L = time.time()
        try:
            surv = apply_L_filter(surv, n_half_int, m_used, c_target,
                                   solver='auto')
        except Exception as e:
            info['l_error'] = str(e)
        info['n_after_L'] = int(len(surv))
        info['time_L'] = round(time.time() - t_L, 4)
        if verbose:
            print(f"  [hybrid]  L: surv {len(surv)}", flush=True)
        if len(surv) == 0:
            info['ok'] = True
            info['reason'] = 'closed_at_L'
            info['wall_sec'] = round(time.time() - t0, 4)
            return True, info

    # If we reach here, some fine compositions are uncertified.
    info['ok'] = False
    info['reason'] = 'residual_after_FQNL'
    info['n_residual'] = int(len(surv))
    info['residual_examples'] = (surv[:5].tolist()
                                  if len(surv) > 0 else [])
    info['wall_sec'] = round(time.time() - t0, 4)
    return False, info


# =====================================================================
# Bench: test on synthetic L1 d=4 uncert cells
# =====================================================================

def _gen_synthetic_uncert_cells(n_cells, S=200, d=4, c_target=1.281,
                                  seed=0):
    """Generate synthetic d=4 coarse cells likely to be uncertified at
    c=1.281, S=200.

    L1 d=4 'uncert' means: TV(μ*) ≥ c_target at grid point, but the box-cert
    chain can't show TV(μ) ≥ c_target on the whole cell.  Such cells live
    near the boundary of the level set {μ : max_W TV_W(μ) = c_target}.

    Heuristic generator: sample compositions where μ* gives TV slightly
    above c_target (i.e., in the marginal regime).
    """
    rng = np.random.default_rng(seed)
    cells = []
    attempts = 0
    while len(cells) < n_cells and attempts < n_cells * 200:
        attempts += 1
        c = rng.integers(0, S + 1, size=d)
        c = c.astype(np.int32)
        s = int(c.sum())
        if s == 0:
            continue
        # Rescale to sum=S exactly
        c = (c * S // s).astype(np.int32)
        diff = S - int(c.sum())
        if diff != 0:
            # Distribute leftovers
            for _ in range(abs(diff)):
                idx = rng.integers(0, d)
                c[idx] += np.sign(diff)
        if int(c.sum()) != S:
            continue
        if (c < 0).any() or (c > S).any():
            continue

        # Quick TV check at grid: compute autoconv & pick best window
        conv = np.zeros(2 * d - 1, dtype=np.int64)
        for i in range(d):
            conv[2 * i] += int(c[i]) * int(c[i])
            for j in range(i + 1, d):
                conv[i + j] += 2 * int(c[i]) * int(c[j])
        S_d = float(S)
        S_sq = S_d * S_d
        d_d = float(d)
        max_ell = 2 * d
        best_tv = 0.0
        for ell in range(2, max_ell + 1):
            n_cv = ell - 1
            n_windows = (2 * d - 1) - n_cv + 1
            ws = sum(int(conv[k]) for k in range(n_cv))
            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += int(conv[s_lo + n_cv - 1]) - int(conv[s_lo - 1])
                tv = ws * 2.0 * d_d / (S_sq * ell)
                if tv > best_tv:
                    best_tv = tv
        # Want cells where best_tv is in the marginal range [c_target, c_target + 0.05]
        # (genuinely uncertain; near the level set).
        if c_target <= best_tv < c_target + 0.05:
            cells.append((c, best_tv))
    return cells


def _bench(n_cells=30, m_target=20, S=200, d=4, c_target=1.281,
           verbose=False):
    """Benchmark: try fine-grid cert on n_cells synthetic uncert cells."""
    print(f"\n{'='*72}")
    print(f"HYBRID CASCADE BENCH")
    print(f"  d={d}, S={S}, c_target={c_target}, m_target={m_target}")
    print(f"{'='*72}\n", flush=True)

    cells = _gen_synthetic_uncert_cells(n_cells, S=S, d=d,
                                          c_target=c_target, seed=0)
    print(f"Generated {len(cells)} synthetic uncert cells\n", flush=True)

    n_certified = 0
    n_failed = 0
    times = []
    stages = []

    for i, (c_int, tv0) in enumerate(cells):
        ok, info = fine_grid_box_cert(
            c_int, S, d, c_target, m_target=m_target, verbose=verbose)
        times.append(info['wall_sec'])
        stages.append(info.get('reason', '?'))
        if ok:
            n_certified += 1
        else:
            n_failed += 1
        print(f"  [{i+1:2d}/{len(cells)}] c={c_int.tolist()} tv0={tv0:.4f}  "
              f"-> {'OK' if ok else 'FAIL'} ({info['reason']})  "
              f"n_fine={info['n_fine_comps']}  m_used={info['m_target_used']}  "
              f"K={info['K']}  wall={info['wall_sec']:.3f}s",
              flush=True)

    print(f"\n{'='*72}")
    print(f"SUMMARY")
    print(f"{'='*72}")
    print(f"  n_cells          : {len(cells)}")
    print(f"  certified        : {n_certified}")
    print(f"  failed           : {n_failed}")
    if times:
        print(f"  wall avg         : {np.mean(times):.3f}s")
        print(f"  wall median      : {np.median(times):.3f}s")
        print(f"  wall p95         : {np.percentile(times, 95):.3f}s")
        print(f"  wall total       : {np.sum(times):.1f}s")
    # Stage breakdown
    from collections import Counter
    stage_ct = Counter(stages)
    print(f"  stage breakdown  : {dict(stage_ct)}")
    print(flush=True)

    return {
        'n_cells': len(cells),
        'n_certified': n_certified,
        'n_failed': n_failed,
        'wall_avg': float(np.mean(times)) if times else 0.0,
        'wall_median': float(np.median(times)) if times else 0.0,
        'wall_p95': float(np.percentile(times, 95)) if times else 0.0,
        'wall_total': float(np.sum(times)) if times else 0.0,
        'stages': dict(stage_ct),
        'config': {
            'd': d, 'S': S, 'c_target': c_target,
            'm_target': m_target,
        },
    }


if __name__ == '__main__':
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_cells', type=int, default=30)
    ap.add_argument('--m_target', type=int, default=20)
    ap.add_argument('--S', type=int, default=200)
    ap.add_argument('--d', type=int, default=4)
    ap.add_argument('--c_target', type=float, default=1.281)
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    res = _bench(n_cells=args.n_cells, m_target=args.m_target,
                  S=args.S, d=args.d, c_target=args.c_target,
                  verbose=args.verbose)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(res, f, indent=2, default=str)
        print(f"\nResults -> {args.out}")
