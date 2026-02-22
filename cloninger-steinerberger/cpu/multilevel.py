"""Multi-level adaptive refinement solver.

Implements 2c from the improved algorithm: refine only high-sensitivity bins
of surviving parents, dramatically reducing the branching factor compared
to full uniform refinement.
"""
import sys
import os
import numpy as np
import time
import itertools
from numba import njit, prange

# Add both cpu/ and parent cloninger-steinerberger/ to path
_cpu_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_cpu_dir)
sys.path.insert(0, _cpu_dir)
sys.path.insert(0, _parent_dir)

from pruning import correction, asymmetry_threshold, count_compositions, asymmetry_prune_mask
from test_values import compute_test_values_batch, compute_test_value_single


def compute_sensitivities(parent_int, n_half, m):
    """Compute per-bin sensitivity for a parent composition.

    Sensitivity sigma_i measures how much bin i contributes to the
    best window test value.  High-sensitivity bins benefit most from
    refinement.

    For each window (ell, s_lo), the contribution of bin i is proportional
    to the sum of parent masses in the other bins that pair with i within
    that window.  We take the max over all windows.

    Parameters
    ----------
    parent_int : (d,) int array
        Parent composition in integer coordinates (sum = m).
    n_half : int
        Half the number of bins (d = 2 * n_half).
    m : int
        Grid resolution.

    Returns
    -------
    (d,) float64 array of per-bin sensitivities.
    """
    d = len(parent_int)
    conv_len = 2 * d - 1
    scale = 4.0 * n_half / m
    a = parent_int.astype(np.float64) * scale

    # Prefix sums of a for quick range-sum queries
    psum = np.zeros(d + 1, dtype=np.float64)
    for i in range(d):
        psum[i + 1] = psum[i] + a[i]

    sensitivities = np.zeros(d, dtype=np.float64)

    for ell in range(2, d + 1):
        n_cv = ell - 1
        inv_norm = 1.0 / (4.0 * n_half * ell)
        for s_lo in range(conv_len - n_cv + 1):
            s_hi = s_lo + n_cv - 1
            for i in range(d):
                # Valid j range for this (i, window): s_lo - i <= j <= s_hi - i
                j_lo = max(0, s_lo - i)
                j_hi = min(d - 1, s_hi - i)
                if j_lo > j_hi:
                    continue
                # Sum of a_j for j in [j_lo, j_hi]
                partner_mass = psum[j_hi + 1] - psum[j_lo]
                # Contribution: 2 * a_i * partner_mass / (4*n*ell)
                # (factor of 2 because a_i appears on both sides of the
                # bilinear form; for i in partner range, subtract a_i once)
                contrib = 2.0 * a[i] * partner_mass * inv_norm
                if contrib > sensitivities[i]:
                    sensitivities[i] = contrib

    return sensitivities


def select_bins_to_refine(sensitivities, unrefined_counts, K_force=2, top_fraction=0.5):
    """Select which bins to refine based on sensitivity and history.

    Parameters
    ----------
    sensitivities : (d,) float array
        Per-bin sensitivity from compute_sensitivities.
    unrefined_counts : (d,) int array
        How many consecutive levels each bin was NOT refined.
    K_force : int
        Force-refine bins unrefined for >= K_force levels.
    top_fraction : float
        Fraction of bins to select by sensitivity (0 to 1).

    Returns
    -------
    sorted list of bin indices to refine.
    """
    d = len(sensitivities)
    selected = set()

    # Force-refine bins that haven't been refined recently
    for i in range(d):
        if unrefined_counts[i] >= K_force:
            selected.add(i)

    # Select top fraction by sensitivity
    n_select = max(1, int(np.ceil(d * top_fraction)))
    ranked = np.argsort(-sensitivities)  # descending
    for k in range(min(n_select, d)):
        selected.add(int(ranked[k]))

    return sorted(selected)


@njit(parallel=True, cache=True)
def compute_masks_batch(survivors, n_half, m, unrefined_counts,
                        K_force, top_fraction):
    """Compute refine masks for all parents in parallel (Numba).

    Parameters
    ----------
    survivors : (N, d) int32 array
        Parent compositions.
    n_half : int
        Half the number of bins.
    m : int
        Grid resolution.
    unrefined_counts : (N, d) int32 array
        Per-bin unrefined level counts (parallel to survivors).
    K_force : int
        Force-refine bins unrefined >= K_force levels.
    top_fraction : float
        Fraction of bins to select by sensitivity.

    Returns
    -------
    (N, d) int8 array of masks (1 = refine, 0 = deterministic split).
    """
    num_parents = survivors.shape[0]
    d = survivors.shape[1]
    masks = np.zeros((num_parents, d), dtype=np.int8)
    conv_len = 2 * d - 1
    scale = 4.0 * n_half / m
    n_select = int(np.ceil(d * top_fraction))
    if n_select < 1:
        n_select = 1

    for p in prange(num_parents):
        # Compute scaled values and prefix sums
        a = np.empty(d, dtype=np.float64)
        for i in range(d):
            a[i] = survivors[p, i] * scale

        psum = np.zeros(d + 1, dtype=np.float64)
        for i in range(d):
            psum[i + 1] = psum[i] + a[i]

        # Compute per-bin sensitivities
        sens = np.zeros(d, dtype=np.float64)
        for ell in range(2, d + 1):
            n_cv = ell - 1
            inv_norm = 1.0 / (4.0 * n_half * ell)
            for s_lo in range(conv_len - n_cv + 1):
                s_hi = s_lo + n_cv - 1
                for i in range(d):
                    j_lo = s_lo - i
                    if j_lo < 0:
                        j_lo = 0
                    j_hi = s_hi - i
                    if j_hi > d - 1:
                        j_hi = d - 1
                    if j_lo > j_hi:
                        continue
                    partner_mass = psum[j_hi + 1] - psum[j_lo]
                    contrib = 2.0 * a[i] * partner_mass * inv_norm
                    if contrib > sens[i]:
                        sens[i] = contrib

        # Force-refine bins unrefined >= K_force levels
        for i in range(d):
            if unrefined_counts[p, i] >= K_force:
                masks[p, i] = 1

        # Top-k by sensitivity (partial selection sort, fine for small d)
        ranked = np.empty(d, dtype=np.int64)
        for i in range(d):
            ranked[i] = i
        for i in range(min(n_select, d)):
            for j in range(i + 1, d):
                if sens[ranked[j]] > sens[ranked[i]]:
                    tmp = ranked[i]
                    ranked[i] = ranked[j]
                    ranked[j] = tmp
        for k in range(min(n_select, d)):
            masks[p, ranked[k]] = 1

    return masks


def generate_children(parent_int, bins_to_refine, m):
    """Generate all children of a parent at the next refinement level.

    Each refined bin i with mass b_i is split into two sub-bins:
        c_{2i} in [0, b_i],  c_{2i+1} = b_i - c_{2i}
    Non-refined bins are split evenly:
        c_{2i} = b_i // 2,   c_{2i+1} = b_i - b_i // 2

    Parameters
    ----------
    parent_int : (d,) int array
        Parent composition in integer coordinates.
    bins_to_refine : list of int
        Indices of bins to refine.
    m : int
        Grid resolution (same m at all levels).

    Returns
    -------
    (N_children, 2*d) int32 array
    """
    d = len(parent_int)
    d_new = 2 * d
    refine_set = set(bins_to_refine)

    # Build per-bin choices
    # For refined bins: all splits [0..b_i]
    # For non-refined bins: single deterministic split
    per_bin_choices = []
    for i in range(d):
        b_i = int(parent_int[i])
        if i in refine_set:
            per_bin_choices.append(list(range(b_i + 1)))
        else:
            per_bin_choices.append([b_i // 2])

    # Total children = product of choice counts
    total = 1
    for choices in per_bin_choices:
        total *= len(choices)

    children = np.empty((total, d_new), dtype=np.int32)
    idx = 0
    for combo in itertools.product(*per_bin_choices):
        for i in range(d):
            children[idx, 2 * i] = combo[i]
            children[idx, 2 * i + 1] = int(parent_int[i]) - combo[i]
        idx += 1

    return children


def test_children_batch(children_int, n_half_new, m, c_target):
    """Test a batch of children at the new level.

    Parameters
    ----------
    children_int : (N, d_new) int32 array
        Children compositions in integer coordinates.
    n_half_new : int
        New n_half (= 2 * old n_half).
    m : int
        Grid resolution.
    c_target : float
        Target lower bound.

    Returns
    -------
    (survivors, stats) where survivors is (K, d_new) int32 array and
    stats is a dict with pruning counts.
    """
    if len(children_int) == 0:
        d_new = 2  # placeholder
        return np.empty((0, d_new), dtype=np.int32), {
            'n_tested': 0, 'n_asym': 0, 'n_test': 0, 'n_survived': 0
        }

    N, d_new = children_int.shape
    corr = correction(m)
    prune_target = corr + c_target

    # Asymmetry mask
    asym_mask = asymmetry_prune_mask(children_int, n_half_new, m, c_target)
    n_asym = int(np.sum(~asym_mask))

    # Test values for those passing asymmetry
    candidates = children_int[asym_mask]
    if len(candidates) > 0:
        test_vals = compute_test_values_batch(candidates, n_half_new, m,
                                               prune_target=prune_target)
        survived_mask = test_vals <= prune_target
        survivors = candidates[survived_mask]
        n_test = int(np.sum(~survived_mask))
    else:
        survivors = np.empty((0, d_new), dtype=np.int32)
        n_test = 0

    n_survived = len(survivors)
    stats = {
        'n_tested': N,
        'n_asym': n_asym,
        'n_test': n_test,
        'n_survived': n_survived,
    }
    return survivors, stats


def run_multi_level(n_start, n_max, m, c_target,
                    K_force=2, top_fraction=0.5, verbose=True):
    """Multi-level solver with adaptive refinement.

    Starts at n_half=n_start and refines survivors by doubling n_half
    each level.  Only high-sensitivity bins are refined, reducing the
    branching factor.

    Parameters
    ----------
    n_start : int
        Initial n_half.
    n_max : int
        Maximum n_half (stop refining beyond this).
    m : int
        Grid resolution (same at all levels).
    c_target : float
        Target lower bound to prove.
    K_force : int
        Force-refine bins unrefined for >= K_force levels.
    top_fraction : float
        Fraction of bins to refine by sensitivity.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys: proven, survivors, level_stats
    """
    from solvers import run_single_level

    if verbose:
        print(f"\n{'='*70}")
        print(f"MULTI-LEVEL ADAPTIVE REFINEMENT")
        print(f"  n_start={n_start}, n_max={n_max}, m={m}, c_target={c_target}")
        print(f"  K_force={K_force}, top_fraction={top_fraction}")
        print(f"{'='*70}")

    level_stats = []
    t_total = time.time()

    # Level 0: run single-level with survivor collection
    result0 = run_single_level(n_start, m, c_target,
                                collect_survivors=True, verbose=verbose)
    survivors = result0['survivors']
    n_survived_0 = result0['n_survivors']

    level_stats.append({
        'level': 0,
        'n_half': n_start,
        'd': 2 * n_start,
        'n_parents': 1,  # conceptual: whole grid
        'n_children_tested': result0['stats']['n_processed'],
        'n_survivors': n_survived_0,
        'elapsed': result0['stats']['elapsed'],
    })

    if n_survived_0 == 0:
        if verbose:
            print(f"\n  >>> PROVEN at level 0: C_{{1a}} >= {c_target:.6f} <<<")
        return {
            'proven': True,
            'survivors': np.empty((0, 2 * n_start), dtype=np.int32),
            'level_stats': level_stats,
        }

    if verbose:
        print(f"\n  Level 0: {n_survived_0} survivors")

    # Initialize unrefined counts (all zeros at start)
    current_n = n_start
    current_d = 2 * current_n

    # Track unrefined counts per survivor per bin
    # Key: tuple(survivor) -> np.array of counts
    unrefined_map = {}
    for i in range(len(survivors)):
        key = tuple(survivors[i])
        unrefined_map[key] = np.zeros(current_d, dtype=np.int32)

    while len(survivors) > 0 and current_n < n_max:
        next_n = 2 * current_n
        next_d = 2 * current_d
        t_level = time.time()

        if verbose:
            print(f"\n--- Level {len(level_stats)}: n_half {current_n} -> {next_n} "
                  f"(d={current_d} -> {next_d}) ---")
            print(f"  Parents to refine: {len(survivors)}")

        all_next_survivors = []
        total_children = 0
        total_cs_stats = {'n_tested': 0, 'n_asym': 0, 'n_test': 0, 'n_survived': 0}
        next_unrefined_map = {}

        for p_idx in range(len(survivors)):
            parent = survivors[p_idx]
            parent_key = tuple(parent)
            unref_counts = unrefined_map.get(parent_key,
                                              np.zeros(current_d, dtype=np.int32))

            # Compute sensitivities
            sensitivities = compute_sensitivities(parent, current_n, m)

            # Select bins to refine
            bins_to_refine = select_bins_to_refine(
                sensitivities, unref_counts, K_force, top_fraction)

            # Generate children
            children = generate_children(parent, bins_to_refine, m)
            total_children += len(children)

            # Test children
            child_survivors, stats = test_children_batch(
                children, next_n, m, c_target)

            for k in total_cs_stats:
                total_cs_stats[k] += stats[k]

            # Track unrefined counts for surviving children
            refine_set = set(bins_to_refine)
            for ci in range(len(child_survivors)):
                child_key = tuple(child_survivors[ci])
                # Build new unrefined counts for the child (2*d bins)
                new_unref = np.zeros(next_d, dtype=np.int32)
                for bi in range(current_d):
                    parent_bi = bi
                    if parent_bi in refine_set:
                        # This bin was refined; reset count for both sub-bins
                        new_unref[2 * bi] = 0
                        new_unref[2 * bi + 1] = 0
                    else:
                        # Not refined; increment count for both sub-bins
                        new_unref[2 * bi] = int(unref_counts[parent_bi]) + 1
                        new_unref[2 * bi + 1] = int(unref_counts[parent_bi]) + 1
                next_unrefined_map[child_key] = new_unref

            if len(child_survivors) > 0:
                all_next_survivors.append(child_survivors)

        # Deduplicate survivors
        if all_next_survivors:
            all_next = np.vstack(all_next_survivors)
            # Simple dedup via set of tuples
            seen = set()
            unique_indices = []
            for i in range(len(all_next)):
                key = tuple(all_next[i])
                if key not in seen:
                    seen.add(key)
                    unique_indices.append(i)
            survivors = all_next[unique_indices]
        else:
            survivors = np.empty((0, next_d), dtype=np.int32)

        elapsed_level = time.time() - t_level
        level_stats.append({
            'level': len(level_stats),
            'n_half': next_n,
            'd': next_d,
            'n_parents': total_cs_stats['n_tested'] // max(1, total_children) if total_children > 0 else 0,
            'n_children_tested': total_children,
            'n_survivors': len(survivors),
            'elapsed': elapsed_level,
            **total_cs_stats,
        })

        if verbose:
            print(f"  Children tested: {total_children:,}")
            print(f"  Asym pruned: {total_cs_stats['n_asym']:,}")
            print(f"  Test pruned: {total_cs_stats['n_test']:,}")
            print(f"  Survivors: {len(survivors):,}")
            print(f"  Time: {elapsed_level:.2f}s")

        current_n = next_n
        current_d = next_d
        unrefined_map = next_unrefined_map

    proven = len(survivors) == 0
    elapsed_total = time.time() - t_total

    if verbose:
        print(f"\n{'='*70}")
        if proven:
            print(f">>> PROVEN: C_{{1a}} >= {c_target:.6f} "
                  f"(multi-level, {elapsed_total:.1f}s total) <<<")
        else:
            print(f"NOT proven at target {c_target:.4f} "
                  f"({len(survivors)} survivors at n_half={current_n})")
        print(f"{'='*70}")

    return {
        'proven': proven,
        'survivors': survivors,
        'level_stats': level_stats,
    }


def run_multi_level_standard(n_start, n_max, m, c_target, verbose=True):
    """Multi-level solver with full (non-adaptive) refinement.

    Same as run_multi_level but refines ALL bins at every level.
    Serves as the baseline for comparison with adaptive refinement.
    """
    return run_multi_level(n_start, n_max, m, c_target,
                           K_force=0, top_fraction=1.0, verbose=verbose)
