#!/usr/bin/env python
r"""Subdivision-based box certification.

==========================================================================
KEY INSIGHT
==========================================================================

val(d) = min_{mu on simplex} max_W TV_W(mu) > c_target means EVERY d-bin
mass distribution has some window with TV >= val(d).  No exceptions.

The cascade at d bins, grid resolution S, proves all grid points are pruned.
Box cert extends this to the continuum.  When box cert fails for a cell:

  margin ≈ 0.019,  cell_var ≈ 0.3  (cell_var >> margin)

Subdivision fixes this:
  1. Split the failing cell into sub-cells at resolution 2S.
  2. Each sub-cell center IS a valid d-bin distribution on the simplex.
  3. Therefore max_W TV_W(sub-center) >= val(d) > c_target.
  4. So margin > 0 at every sub-cell center.
  5. Sub-cell width is half the original, so cell_var halves.
  6. After ~k subdivisions, cell_var/2^k < margin, and cert passes.

Convergence is GUARANTEED because:
  - margin stays bounded below by val(d) - c_target > 0
  - cell_var -> 0 geometrically

==========================================================================
IMPLEMENTATION
==========================================================================

Two approaches:

A. GRID REFINEMENT: enumerate sub-grid-points at resolution r*S within
   the failing cell.  Each sub-cell has width 1/(2*r*S), so cell_var
   shrinks by factor r.  Clean, matches cascade structure.

B. ADAPTIVE BISECTION: split the cell along the coordinate with largest
   gradient contribution.  Binary tree, depth-first.  Efficient because
   most branches certify early.

We implement both and compare.
"""

import sys
import os
import math
import time
import itertools

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_this_dir)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, os.path.join(_root, 'cloninger-steinerberger'))

from compositions import generate_canonical_compositions_batched
from pruning import count_compositions


# =====================================================================
# Core TV computation
# =====================================================================

def tv_window_fast(mu, d, ell, s_lo):
    """TV_W(mu) for window (ell, s_lo)."""
    s_hi = s_lo + ell - 2
    total = 0.0
    for i in range(d):
        if mu[i] == 0:
            continue
        for j in range(d):
            if s_lo <= i + j <= s_hi:
                total += mu[i] * mu[j]
    return total * 2.0 * d / ell


def best_killing_window(mu, d, c_target):
    """Find window with highest TV. Returns (tv, ell, s_lo) or None."""
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.float64)
    for i in range(d):
        if mu[i] == 0:
            continue
        for j in range(d):
            conv[i + j] += mu[i] * mu[j]

    best_tv = 0.0
    best_ell = 0
    best_s = 0
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        inv_norm = 2.0 * d / ell
        ws = sum(conv[:n_cv])
        for s_lo in range(conv_len - n_cv + 1):
            if s_lo > 0:
                ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
            tv = ws * inv_norm
            if tv > best_tv:
                best_tv = tv
                best_ell = ell
                best_s = s_lo
    return best_tv, best_ell, best_s


# =====================================================================
# Single-window box cert on a general cell [lo, hi] ∩ simplex
# =====================================================================

def box_cert_cell(mu_center, cell_lo, cell_hi, d, c_target):
    """Box cert for a cell [cell_lo, cell_hi] ∩ simplex.

    mu_center: center of the cell (on simplex).
    cell_lo, cell_hi: per-coordinate bounds.

    Returns dict with certification result and diagnostics.
    """
    # Find best killing window at center
    best_tv, best_ell, best_s = best_killing_window(mu_center, d, c_target)
    margin = best_tv - c_target

    if margin < -1e-15:
        # Center itself not pruned — shouldn't happen if val(d) > c_target
        return {
            'certified': False,
            'margin': margin,
            'cell_var': 0.0,
            'quad_corr': 0.0,
            'net': margin,
            'ell': best_ell,
            's_lo': best_s,
        }

    s_lo = best_s
    s_hi = s_lo + best_ell - 2
    ell = best_ell
    scale_g = 4.0 * d / ell

    # Gradient at center
    grad = np.zeros(d, dtype=np.float64)
    for i in range(d):
        g = 0.0
        for j in range(d):
            if s_lo <= i + j <= s_hi:
                g += mu_center[j]
        grad[i] = g * scale_g

    # Cell variation: max |grad · delta| subject to
    #   |delta_i| <= half_width_i and sum delta_i = 0
    # Sorted-pairing bound (exact for uniform half-width + sum=0)
    half_width = (cell_hi - cell_lo) / 2.0

    # For non-uniform half-widths, use the general bound:
    # max grad·delta s.t. |delta_i| <= hw_i, sum delta_i = 0
    # = solve by sorting grad_i, then greedily assign delta_i = +hw_i
    # for top gradients and -hw_i for bottom, subject to sum=0.
    #
    # Simple upper bound: sum of top d/2 (grad_i * hw_i) - bottom d/2
    # Actually, general LP. For uniform hw: reduces to sorted-pairing.

    # For uniform half-width (standard case):
    hw = half_width[0] if np.allclose(half_width, half_width[0]) else None

    if hw is not None:
        grad_sorted = np.sort(grad)
        cell_var = 0.0
        for k in range(d // 2):
            cell_var += (grad_sorted[d - 1 - k] - grad_sorted[k]) * hw
    else:
        # Non-uniform: solve the LP max grad·delta s.t. |delta_i|<=hw_i, sum=0
        # Upper bound: sort by grad, assign +hw to top, -hw to bottom
        order = np.argsort(grad)
        # Greedy: +hw for highest grad, -hw for lowest, maintain sum balance
        delta = np.zeros(d)
        running_sum = 0.0
        # First pass: assign all to maximize grad·delta
        for i in range(d):
            delta[i] = half_width[i] if grad[i] >= 0 else -half_width[i]
        # Project to sum=0: remove excess
        excess = delta.sum()
        if abs(excess) > 1e-15:
            # Reduce the contribution of the least valuable coordinates
            if excess > 0:
                # Need to decrease some delta_i from +hw to something smaller
                order = np.argsort(grad)  # ascending
                for idx in order:
                    if excess <= 0:
                        break
                    room = delta[idx] - (-half_width[idx])
                    adjust = min(room, excess)
                    delta[idx] -= adjust
                    excess -= adjust
            else:
                order = np.argsort(-grad)  # descending
                for idx in order:
                    if excess >= 0:
                        break
                    room = half_width[idx] - delta[idx]
                    adjust = min(room, -excess)
                    delta[idx] += adjust
                    excess += adjust
        cell_var = abs(np.dot(grad, delta))

    # Quadratic correction
    n_pairs = 0
    for k in range(s_lo, s_lo + ell - 1):
        cnt = min(k + 1, d)
        if k > d - 1:
            cnt = min(cnt, 2 * d - 1 - k)
        n_pairs += cnt

    # Use max half-width for quad bound
    max_hw = half_width.max()
    quad_corr = (2.0 * d / ell) * n_pairs * max_hw * max_hw

    net = margin - cell_var - quad_corr

    return {
        'certified': net >= 0,
        'margin': margin,
        'cell_var': cell_var,
        'quad_corr': quad_corr,
        'net': net,
        'ell': ell,
        's_lo': s_lo,
    }


# =====================================================================
# Multi-window box cert (try all killing windows, pick best net)
# =====================================================================

def box_cert_cell_multiwindow(mu_center, cell_lo, cell_hi, d, c_target):
    """Try all killing windows, return best net."""
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.float64)
    for i in range(d):
        for j in range(d):
            conv[i + j] += mu_center[i] * mu_center[j]

    half_width = (cell_hi - cell_lo) / 2.0
    max_hw = half_width.max()
    # For uniform half-width (common case)
    hw = half_width[0] if np.allclose(half_width, half_width[0]) else None

    best_net = -1e30
    best_info = None

    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        scale_tv = 2.0 * d / ell
        scale_g = 4.0 * d / ell
        ws = sum(conv[:n_cv])

        for s_lo in range(conv_len - n_cv + 1):
            if s_lo > 0:
                ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
            tv = ws * scale_tv
            margin = tv - c_target
            if margin < 0:
                continue

            s_hi = s_lo + ell - 2

            # Gradient
            grad = np.zeros(d, dtype=np.float64)
            for i in range(d):
                g = 0.0
                for j in range(d):
                    if s_lo <= i + j <= s_hi:
                        g += mu_center[j]
                grad[i] = g * scale_g

            # cell_var
            if hw is not None:
                grad_sorted = np.sort(grad)
                cell_var = 0.0
                for k in range(d // 2):
                    cell_var += (grad_sorted[d-1-k] - grad_sorted[k]) * hw
            else:
                # Upper bound for non-uniform
                products = np.abs(grad) * half_width
                cell_var = 2.0 * np.sort(products)[d//2:].sum()

            # quad_corr
            n_pairs = 0
            for k in range(s_lo, s_lo + ell - 1):
                cnt = min(k + 1, d)
                if k > d - 1:
                    cnt = min(cnt, 2 * d - 1 - k)
                n_pairs += cnt
            quad_corr = scale_tv * n_pairs * max_hw * max_hw

            net = margin - cell_var - quad_corr
            if net > best_net:
                best_net = net
                best_info = {
                    'certified': net >= 0,
                    'margin': margin,
                    'cell_var': cell_var,
                    'quad_corr': quad_corr,
                    'net': net,
                    'ell': ell,
                    's_lo': s_lo,
                }

    if best_info is None:
        return {
            'certified': False, 'margin': 0, 'cell_var': 0,
            'quad_corr': 0, 'net': -1e30, 'ell': 0, 's_lo': 0,
        }
    return best_info


# =====================================================================
# Approach A: Grid refinement
# =====================================================================

def enumerate_subcells(k_parent, S_parent, refine):
    """Enumerate sub-grid-points at resolution S_parent * refine
    within the Voronoi cell of k_parent / S_parent.

    Each sub-grid-point j satisfies:
      |j_i / (refine*S_parent) - k_i / S_parent| <= 1/(2*S_parent)
      sum j_i = refine * S_parent
      j_i >= 0

    Equivalently: j_i in [refine*k_i - refine//2, refine*k_i + refine//2]
    and sum j_i = refine * S_parent.
    """
    d = len(k_parent)
    S_fine = S_parent * refine
    center_fine = k_parent * refine  # j_i ≈ refine * k_i

    # Range for each coordinate: +/- refine//2 from center
    # (This covers the original cell at the finer resolution)
    half = refine // 2
    lo = np.maximum(center_fine - half, 0).astype(np.int32)
    hi = (center_fine + half).astype(np.int32)

    # Generate all compositions in the box with sum = S_fine
    # Use recursive enumeration
    results = []
    _enumerate_recursive(lo, hi, d, S_fine, 0, np.zeros(d, dtype=np.int32),
                         results)
    return results, S_fine


def _enumerate_recursive(lo, hi, d, target_sum, idx, current, results):
    if idx == d - 1:
        val = target_sum - current[:idx].sum()
        if lo[idx] <= val <= hi[idx]:
            current[idx] = val
            results.append(current.copy())
        return

    remaining = d - idx - 1
    partial = int(current[:idx].sum())
    for v in range(lo[idx], hi[idx] + 1):
        current[idx] = v
        # Pruning: remaining sum must be achievable
        remaining_sum = target_sum - partial - v
        remaining_lo = int(lo[idx+1:].sum())
        remaining_hi = int(hi[idx+1:].sum())
        if remaining_sum < remaining_lo or remaining_sum > remaining_hi:
            continue
        _enumerate_recursive(lo, hi, d, target_sum, idx + 1, current, results)


def grid_refinement_cert(k_parent, d, S, c_target, max_refine=16,
                          use_multiwindow=True, verbose=False):
    """Certify a cell by grid refinement.

    Subdivide at resolution 2S, 4S, 8S, ... until all sub-cells pass.

    Returns dict with results.
    """
    refine = 2
    mu_parent = k_parent.astype(np.float64) / S

    # First check: does the original cell pass?
    eps = 1.0 / (2.0 * S)
    mu_lo = np.maximum(mu_parent - eps, 0.0)
    mu_hi = mu_parent + eps

    cert_fn = box_cert_cell_multiwindow if use_multiwindow else box_cert_cell
    result = cert_fn(mu_parent, mu_lo, mu_hi, d, c_target)
    if result['certified']:
        return {
            'certified': True,
            'refine_level': 0,
            'total_subcells': 1,
            'net': result['net'],
        }

    history = []

    while refine <= max_refine:
        subcells, S_fine = enumerate_subcells(k_parent, S, refine)
        n_subcells = len(subcells)

        eps_fine = 1.0 / (2.0 * S_fine)
        n_pass = 0
        n_fail = 0
        worst_net = 1e30

        for sc in subcells:
            mu_sc = sc.astype(np.float64) / S_fine
            sc_lo = np.maximum(mu_sc - eps_fine, 0.0)
            sc_hi = mu_sc + eps_fine
            r = cert_fn(mu_sc, sc_lo, sc_hi, d, c_target)
            if r['certified']:
                n_pass += 1
            else:
                n_fail += 1
            if r['net'] < worst_net:
                worst_net = r['net']

        history.append({
            'refine': refine,
            'S_fine': S_fine,
            'n_subcells': n_subcells,
            'n_pass': n_pass,
            'n_fail': n_fail,
            'worst_net': worst_net,
        })

        if verbose:
            print(f"    refine={refine}x (S={S_fine}): "
                  f"{n_subcells} subcells, "
                  f"{n_pass} pass, {n_fail} fail, "
                  f"worst_net={worst_net:.6f}")

        if n_fail == 0:
            return {
                'certified': True,
                'refine_level': refine,
                'total_subcells': n_subcells,
                'net': worst_net,
                'history': history,
            }

        refine *= 2

    return {
        'certified': False,
        'refine_level': refine // 2,
        'total_subcells': n_subcells,
        'net': worst_net,
        'history': history,
    }


# =====================================================================
# Approach B: Adaptive bisection
# =====================================================================

def adaptive_bisection_cert(mu_center, cell_lo, cell_hi, d, c_target,
                             max_depth=20, use_multiwindow=True):
    """Adaptively bisect failing cells until certified.

    Split along the coordinate whose gradient contribution to cell_var
    is largest.  Each split halves cell_var for that coordinate.

    Returns dict with results.
    """
    cert_fn = box_cert_cell_multiwindow if use_multiwindow else box_cert_cell

    stats = {'n_leaves': 0, 'n_certified': 0, 'max_depth': 0,
             'worst_net': 1e30, 'total_cells_visited': 0}

    def _recurse(center, lo, hi, depth):
        stats['total_cells_visited'] += 1

        r = cert_fn(center, lo, hi, d, c_target)

        if r['certified']:
            stats['n_leaves'] += 1
            stats['n_certified'] += 1
            if depth > stats['max_depth']:
                stats['max_depth'] = depth
            if r['net'] < stats['worst_net']:
                stats['worst_net'] = r['net']
            return True

        if depth >= max_depth:
            stats['n_leaves'] += 1
            if r['net'] < stats['worst_net']:
                stats['worst_net'] = r['net']
            return False

        # Find the coordinate to split: the one contributing most
        # to cell_var.
        # Recompute gradient for the best window
        best_tv, best_ell, best_s = best_killing_window(center, d, c_target)
        if best_tv < c_target:
            stats['n_leaves'] += 1
            stats['worst_net'] = min(stats['worst_net'], best_tv - c_target)
            return False

        s_lo = best_s
        s_hi = s_lo + best_ell - 2
        scale_g = 4.0 * d / best_ell
        grad = np.zeros(d)
        for i in range(d):
            g = 0.0
            for j in range(d):
                if s_lo <= i + j <= s_hi:
                    g += center[j]
            grad[i] = g * scale_g

        # Split axis: coordinate with widest range AND large gradient
        half_width = (hi - lo) / 2.0
        contribution = np.abs(grad) * half_width
        split_axis = np.argmax(contribution)

        # If all contributions are zero, try widest coordinate
        if contribution[split_axis] < 1e-15:
            split_axis = np.argmax(half_width)

        # Split
        mid = (lo[split_axis] + hi[split_axis]) / 2.0

        # Left sub-cell
        hi_left = hi.copy()
        hi_left[split_axis] = mid
        center_left = (lo + hi_left) / 2.0
        # Project center to simplex (adjust one coordinate)
        excess = center_left.sum() - 1.0
        if abs(excess) > 1e-15:
            # Distribute excess across coordinates, preferring non-split
            center_left -= excess / d
            center_left = np.maximum(center_left, lo)
            center_left = np.minimum(center_left, hi_left)
            # Renormalize
            center_left *= 1.0 / center_left.sum()

        # Right sub-cell
        lo_right = lo.copy()
        lo_right[split_axis] = mid
        center_right = (lo_right + hi) / 2.0
        excess = center_right.sum() - 1.0
        if abs(excess) > 1e-15:
            center_right -= excess / d
            center_right = np.maximum(center_right, lo_right)
            center_right = np.minimum(center_right, hi)
            center_right *= 1.0 / center_right.sum()

        ok_left = _recurse(center_left, lo, hi_left, depth + 1)
        ok_right = _recurse(center_right, lo_right, hi, depth + 1)

        return ok_left and ok_right

    certified = _recurse(mu_center, cell_lo, cell_hi, 0)
    stats['certified'] = certified
    return stats


# =====================================================================
# Pruning
# =====================================================================

def prune_coarse(batch_int, d, S, c_target):
    """Pure-Python coarse pruning. Returns survived mask + info."""
    B = batch_int.shape[0]
    survived = np.ones(B, dtype=bool)
    info = [None] * B

    for b in range(B):
        mu = batch_int[b].astype(np.float64) / S
        best_tv, best_ell, best_s = best_killing_window(mu, d, c_target)
        if best_tv >= c_target - 1e-12:
            survived[b] = False
            info[b] = (best_tv, best_ell, best_s, best_tv - c_target)

    return survived, info


# =====================================================================
# Main test
# =====================================================================

def run_test(d0, S, c_target, max_comps=50000, verbose=True):
    """Test subdivision box cert on coarse grid compositions."""
    n_total = count_compositions(d0, S)
    if verbose:
        print(f"\n{'='*70}")
        print(f"SUBDIVISION BOX CERTIFICATION TEST")
        print(f"  d0={d0}, S={S}, c_target={c_target}")
        print(f"  Total compositions: {n_total:,}")
        print(f"{'='*70}")

    # Generate and prune
    t0 = time.time()
    all_comps = []
    gen = generate_canonical_compositions_batched(d0, S, batch_size=100_000)
    for batch in gen:
        all_comps.append(batch)
        if sum(len(c) for c in all_comps) >= max_comps:
            break
    comps = np.vstack(all_comps)[:max_comps]

    survived, info = prune_coarse(comps, d0, S, c_target)
    n_pruned = int(np.sum(~survived))
    pruned_idx = np.where(~survived)[0]

    if verbose:
        print(f"  Pruned: {n_pruned:,} / {len(comps):,} in {time.time()-t0:.2f}s")

    if n_pruned == 0:
        print("  No pruned compositions. Try lower c_target.")
        return

    # Find hardest compositions (smallest margin)
    margins = np.array([info[i][3] for i in pruned_idx])
    order = np.argsort(margins)

    # Check which fail the old box cert
    n_old_pass = 0
    n_old_fail = 0
    failing_idx = []

    for idx in pruned_idx:
        k = comps[idx]
        mu = k.astype(np.float64) / S
        eps = 1.0 / (2.0 * S)
        mu_lo = np.maximum(mu - eps, 0.0)
        mu_hi = mu + eps
        r = box_cert_cell_multiwindow(mu, mu_lo, mu_hi, d0, c_target)
        if r['certified']:
            n_old_pass += 1
        else:
            n_old_fail += 1
            failing_idx.append(idx)

    if verbose:
        print(f"\n  Old box cert: {n_old_pass} pass, {n_old_fail} fail")

    if n_old_fail == 0:
        print("  All cells certified by old method. Nothing to subdivide.")
        return

    # Test subdivision on failing cells
    if verbose:
        print(f"\n  Testing subdivision on {len(failing_idx)} failing cells...")
        print(f"  Method A: Grid refinement (refine=2x, 4x, 8x, 16x)")

    # Sample at most 200 failing cells (most difficult first)
    fail_margins = np.array([info[i][3] for i in failing_idx])
    fail_order = np.argsort(fail_margins)
    test_idx = [failing_idx[fail_order[i]]
                for i in range(min(200, len(fail_order)))]

    t0 = time.time()
    n_subdiv_pass = 0
    n_subdiv_fail = 0
    total_subcells = 0
    max_refine_used = 0

    for count, idx in enumerate(test_idx):
        k = comps[idx]
        r = grid_refinement_cert(k, d0, S, c_target, max_refine=16,
                                  use_multiwindow=True,
                                  verbose=(count < 5 and verbose))
        if r['certified']:
            n_subdiv_pass += 1
            total_subcells += r['total_subcells']
            if r['refine_level'] > max_refine_used:
                max_refine_used = r['refine_level']
        else:
            n_subdiv_fail += 1

        if verbose and count < 5:
            status = "PASS" if r['certified'] else "FAIL"
            print(f"  [{count+1}] k={k} margin={info[idx][3]:.6f} "
                  f"-> {status} (refine={r['refine_level']}x, "
                  f"{r.get('total_subcells', '?')} subcells)")

    elapsed = time.time() - t0

    if verbose:
        print(f"\n{'='*70}")
        print(f"RESULTS ({len(test_idx)} failing cells tested, {elapsed:.2f}s)")
        print(f"{'='*70}")
        print(f"  Grid refinement:")
        print(f"    Certified: {n_subdiv_pass} / {len(test_idx)} "
              f"({100*n_subdiv_pass/len(test_idx):.1f}%)")
        print(f"    Failed:    {n_subdiv_fail}")
        print(f"    Max refinement needed: {max_refine_used}x "
              f"(S -> {S * max_refine_used})")
        if n_subdiv_pass > 0:
            print(f"    Avg subcells per cert: "
                  f"{total_subcells / n_subdiv_pass:.0f}")

    # Also test adaptive bisection on a few
    if verbose and len(test_idx) > 0:
        print(f"\n  Testing adaptive bisection on 10 hardest...")
        for count in range(min(10, len(test_idx))):
            idx = test_idx[count]
            k = comps[idx]
            mu = k.astype(np.float64) / S
            eps = 1.0 / (2.0 * S)
            mu_lo = np.maximum(mu - eps, 0.0)
            mu_hi = mu + eps

            r = adaptive_bisection_cert(mu, mu_lo, mu_hi, d0, c_target,
                                         max_depth=20, use_multiwindow=True)
            status = "PASS" if r['certified'] else "FAIL"
            print(f"    [{count+1}] k={k} margin={info[idx][3]:.6f} -> "
                  f"{status} (depth={r['max_depth']}, "
                  f"leaves={r['n_leaves']}, "
                  f"visited={r['total_cells_visited']})")

    return {
        'n_old_fail': n_old_fail,
        'n_subdiv_pass': n_subdiv_pass,
        'n_subdiv_fail': n_subdiv_fail,
        'max_refine': max_refine_used,
    }


# =====================================================================
# Entry point
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Test subdivision-based box certification")
    parser.add_argument('--d0', type=int, default=4)
    parser.add_argument('--S', type=int, default=20)
    parser.add_argument('--c_target', type=float, default=1.30)
    parser.add_argument('--max_comps', type=int, default=50000)
    args = parser.parse_args()

    run_test(args.d0, args.S, args.c_target, args.max_comps)


if __name__ == '__main__':
    main()
