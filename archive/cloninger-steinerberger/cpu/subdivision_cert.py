"""Subdivision-based box certification for coarse-grid cascade.

When the cascade converges at dimension d but box cert fails for some cells,
this module subdivides failing cells to finer grids until box cert passes.

PREREQUISITE: val(d) > c_target at the cascade's convergence dimension.
This guarantees every sub-cell center (any grid-point composition at d bins)
has max TV > c_target, so margin > 0 at every sub-cell. The subdivision
converges because cell_var halves each round while margin stays positive.

Algorithm:
  1. Start with failing cells at grid S (compositions where net < 0)
  2. For each failing cell k, generate sub-cells at grid 2S:
     k' = 2k + ε where ε_i ∈ {-1, 0, +1}, Σε_i = 0, k'_i ≥ 0
  3. Compute TV + box cert for each sub-cell at grid 2S
  4. Sub-cells with net ≥ 0 pass; collect failures for next round
  5. Repeat at 4S, 8S, ... until all pass

Sub-cell generation uses meet-in-the-middle: split d dimensions into two
halves, enumerate offsets for each half, match by sum. This is O(3^{d/2})
instead of O(3^d).
"""
import numpy as np
import numba
from numba import njit, prange
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_cascade import _build_pair_prefix


# =====================================================================
# Sub-cell generation (meet-in-the-middle)
# =====================================================================

def _enumerate_half_offsets(d_half, base_half):
    """Enumerate all offset vectors for d_half positions.

    Each ε_i ∈ {-1, 0, +1} subject to 2*base_i + ε_i ≥ 0.
    Returns array of shape (N, d_half) and sums array of shape (N,).
    """
    # Allowed offsets per position
    allowed = []
    for i in range(d_half):
        opts = [0, 1]  # 0 and +1 always allowed
        if base_half[i] >= 1:  # -1 only if 2*base >= 1 (base >= 1 at original S)
            opts.append(-1)
        allowed.append(sorted(opts))

    # Enumerate via Cartesian product
    from itertools import product
    offsets = []
    sums = []
    for combo in product(*allowed):
        arr = np.array(combo, dtype=np.int32)
        offsets.append(arr)
        sums.append(arr.sum())

    return np.array(offsets, dtype=np.int32), np.array(sums, dtype=np.int32)


def generate_subcells(parent_k, d):
    """Generate sub-cell compositions at 2S for a parent at S.

    parent_k: int array of shape (d,) with sum S.
    Returns: int array of shape (N, d) with sum 2S.
    Each row k' satisfies |k'_i - 2*parent_k_i| ≤ 1 and k'_i ≥ 0.

    Uses meet-in-the-middle: split d into two halves.
    """
    d_half = d // 2
    d_rest = d - d_half

    base = 2 * parent_k  # target center at 2S grid

    # Enumerate offsets for each half
    off_left, sums_left = _enumerate_half_offsets(d_half, parent_k[:d_half])
    off_right, sums_right = _enumerate_half_offsets(d_rest, parent_k[d_half:])

    # Group right offsets by sum for fast matching
    from collections import defaultdict
    right_by_sum = defaultdict(list)
    for idx, s in enumerate(sums_right):
        right_by_sum[int(s)].append(idx)

    # Match: sum_left + sum_right = 0
    results = []
    for i, s_left in enumerate(sums_left):
        target = -int(s_left)
        if target in right_by_sum:
            for j in right_by_sum[target]:
                # Combine: k' = base + [off_left[i], off_right[j]]
                k_prime = base.copy()
                k_prime[:d_half] += off_left[i]
                k_prime[d_half:] += off_right[j]
                results.append(k_prime)

    if not results:
        return np.empty((0, d), dtype=np.int32)
    return np.array(results, dtype=np.int32)


# =====================================================================
# Batch certification kernel (Numba-parallel)
# =====================================================================

@njit(parallel=True, cache=True)
def _certify_batch(batch_int, d, S, c_target, prefix_nk, prefix_mk):
    """Compute box cert net for each composition in batch.

    Returns:
        pruned: bool array — True if TV >= c_target for some window
        nets: float64 array — best (margin - cell_var - quad_corr) per comp
    """
    B = batch_int.shape[0]
    conv_len = 2 * d - 1
    S_d = np.float64(S)
    S_sq = S_d * S_d
    d_d = np.float64(d)
    inv_2d = 1.0 / (2.0 * d_d)
    inv_4S2 = 1.0 / (4.0 * S_sq)
    eps = 1e-9
    max_ell = 2 * d

    # Precompute thresholds
    thr_arr = np.empty(max_ell + 1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        thr_arr[ell] = np.int64(c_target * np.float64(ell) * S_sq * inv_2d - eps)

    pruned_out = np.zeros(B, dtype=numba.boolean)
    nets_out = np.full(B, -1e30, dtype=np.float64)

    for b in prange(B):
        # Autoconvolution
        conv = np.zeros(conv_len, dtype=np.int32)
        for i in range(d):
            ci = np.int32(batch_int[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int32(batch_int[b, j])
                    if cj != 0:
                        conv[i + j] += np.int32(2) * ci * cj

        # Window scan
        best_net = np.float64(-1e30)
        found = False
        grad_arr = np.empty(d, dtype=np.float64)

        for ell in range(2, max_ell + 1):
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += np.int64(conv[k])
            dyn_it = thr_arr[ell]
            ell_f = np.float64(ell)
            scale_g = 4.0 * d_d / ell_f

            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += np.int64(conv[s_lo + n_cv - 1]) - np.int64(
                        conv[s_lo - 1])
                if ws > dyn_it:
                    found = True
                    tv = np.float64(ws) * 2.0 * d_d / (S_sq * ell_f)
                    margin = tv - c_target

                    # Gradient
                    for i in range(d):
                        g = 0.0
                        for j in range(d):
                            if s_lo <= i + j <= s_lo + ell - 2:
                                g += np.float64(batch_int[b, j]) / S_d
                        grad_arr[i] = g * scale_g

                    # Sort (insertion sort)
                    for i in range(1, d):
                        key = grad_arr[i]
                        jj = i - 1
                        while jj >= 0 and grad_arr[jj] > key:
                            grad_arr[jj + 1] = grad_arr[jj]
                            jj -= 1
                        grad_arr[jj + 1] = key

                    # Cell var
                    cell_var = 0.0
                    for k in range(d // 2):
                        cell_var += grad_arr[d - 1 - k] - grad_arr[k]
                    cell_var /= (2.0 * S_d)

                    # Quad corr (v2 tight bound)
                    hi_idx = s_lo + ell - 1
                    N_W = prefix_nk[hi_idx] - prefix_nk[s_lo]
                    M_W = prefix_mk[hi_idx] - prefix_mk[s_lo]
                    cross_W = N_W - M_W
                    d_sq = np.int64(d) * np.int64(d)
                    compl_bound = d_sq - N_W
                    pb = min(cross_W, compl_bound)
                    qc = (2.0 * d_d / ell_f) * np.float64(
                        max(pb, np.int64(0))) * inv_4S2

                    net = margin - cell_var - qc
                    if net > best_net:
                        best_net = net

        pruned_out[b] = found
        if found:
            nets_out[b] = best_net

    return pruned_out, nets_out


# =====================================================================
# Main subdivision loop
# =====================================================================

def subdivide_and_certify(failing_comps, d, S_original, c_target,
                          max_rounds=6, verbose=True):
    """Subdivide failing cells until all pass box cert.

    Parameters
    ----------
    failing_comps : ndarray of shape (N, d), int32
        Compositions at grid S_original where box cert failed (net < 0)
        but grid-point pruning succeeded (TV >= c_target).
    d : int
        Number of bins (dimension).
    S_original : int
        Grid resolution of the failing compositions.
    c_target : float
        Target lower bound.
    max_rounds : int
        Maximum subdivision rounds.

    Returns
    -------
    dict with:
        certified: bool — True if all sub-cells eventually pass
        rounds: list of dicts with per-round statistics
        total_subcells: int — total sub-cells checked
        worst_net: float — worst net across all rounds
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"SUBDIVISION BOX CERTIFICATION")
        print(f"  d={d}, S_start={S_original}, c_target={c_target}")
        print(f"  Failing cells: {len(failing_comps):,}")
        print(f"  max_rounds={max_rounds}")
        print(f"{'='*60}", flush=True)

    prefix_nk, prefix_mk = _build_pair_prefix(d)

    current_failures = failing_comps.copy()
    current_S = S_original
    rounds = []
    total_subcells = 0

    for round_num in range(1, max_rounds + 1):
        new_S = 2 * current_S
        n_failures = len(current_failures)

        if n_failures == 0:
            if verbose:
                print(f"\n  All cells certified!", flush=True)
            break

        if verbose:
            print(f"\n  Round {round_num}: {n_failures:,} failing cells "
                  f"at S={current_S} -> subdivide to S={new_S}", flush=True)

        t0 = time.time()

        # Generate all sub-cells for all failing cells
        all_subcells = []
        for i in range(n_failures):
            subcells = generate_subcells(current_failures[i], d)
            all_subcells.append(subcells)
            if verbose and (i + 1) % 100 == 0:
                n_so_far = sum(len(s) for s in all_subcells)
                print(f"    Generated sub-cells for {i+1}/{n_failures} "
                      f"({n_so_far:,} total)", flush=True)

        if not all_subcells:
            break

        batch = np.vstack(all_subcells)
        n_subcells = len(batch)
        total_subcells += n_subcells

        if verbose:
            print(f"    Total sub-cells: {n_subcells:,} "
                  f"({n_subcells / n_failures:.0f} per cell)", flush=True)

        # Deduplicate (sub-cells from neighboring parents may overlap)
        batch = np.unique(batch, axis=0)
        n_unique = len(batch)
        if verbose and n_unique < n_subcells:
            print(f"    After dedup: {n_unique:,} unique", flush=True)

        # Certify all sub-cells
        pruned, nets = _certify_batch(batch, d, new_S, c_target,
                                      prefix_nk, prefix_mk)

        elapsed = time.time() - t0

        # Check results
        n_pruned = int(pruned.sum())
        n_not_pruned = n_unique - n_pruned

        if n_not_pruned > 0:
            # Some sub-cells have max TV < c_target — this shouldn't happen
            # if val(d) > c_target!
            print(f"    WARNING: {n_not_pruned} sub-cells NOT PRUNED "
                  f"(max TV < {c_target})!", flush=True)
            print(f"    This means val({d}) <= {c_target} — "
                  f"subdivision cannot succeed.", flush=True)
            return {
                'certified': False,
                'reason': f'{n_not_pruned} sub-cells have max TV < c_target',
                'rounds': rounds,
                'total_subcells': total_subcells,
            }

        # All pruned — check box cert
        min_net = float(nets[pruned].min()) if n_pruned > 0 else 1e30
        n_pass = int((nets >= 0).sum())
        n_fail = n_unique - n_pass
        failing_mask = pruned & (nets < 0)
        new_failures = batch[failing_mask]

        round_info = {
            'round': round_num,
            'S': new_S,
            'input_failures': n_failures,
            'subcells': n_unique,
            'all_pruned': n_not_pruned == 0,
            'box_pass': n_pass,
            'box_fail': n_fail,
            'min_net': min_net,
            'elapsed': elapsed,
        }
        rounds.append(round_info)

        if verbose:
            print(f"    {elapsed:.1f}s: all pruned, "
                  f"box cert: {n_pass:,} pass, {n_fail:,} fail, "
                  f"min_net={min_net:.6f}", flush=True)

        if n_fail == 0:
            if verbose:
                print(f"\n  *** ALL SUB-CELLS CERTIFIED at S={new_S}! ***",
                      flush=True)
            return {
                'certified': True,
                'rounds': rounds,
                'total_subcells': total_subcells,
                'final_S': new_S,
                'worst_net': min(r['min_net'] for r in rounds),
            }

        current_failures = new_failures
        current_S = new_S

    # Didn't converge
    if verbose:
        print(f"\n  Did not converge after {max_rounds} rounds. "
              f"{len(current_failures):,} cells still failing.", flush=True)

    return {
        'certified': False,
        'reason': f'did not converge after {max_rounds} rounds',
        'remaining_failures': len(current_failures),
        'rounds': rounds,
        'total_subcells': total_subcells,
    }


# =====================================================================
# End-to-end: cascade + subdivision
# =====================================================================

def prove_with_subdivision(d0, S, c_target, max_cascade_levels=8,
                           max_subdiv_rounds=6, verbose=True):
    """Run cascade, then subdivide failing cells for rigorous proof.

    Returns dict with proof status.
    """
    from run_cascade import run_cascade

    if verbose:
        print(f"\n{'='*60}")
        print(f"CASCADE + SUBDIVISION PROOF")
        print(f"  d0={d0}, S={S}, c_target={c_target}")
        print(f"{'='*60}", flush=True)

    t_total = time.time()
    os.makedirs('data', exist_ok=True)

    # Step 1: Run cascade
    info = run_cascade(
        n_half=d0 / 2.0, m=20, c_target=c_target,
        max_levels=max_cascade_levels, n_workers=None,
        verbose=verbose, output_dir='data',
        coarse_S=S, d0=d0)

    if 'proven_at' not in info:
        print(f"Cascade did not converge. Cannot proceed.", flush=True)
        return {'proven': False, 'reason': 'cascade_not_converged'}

    if info.get('box_certified', False):
        print(f"Cascade box cert already passes! No subdivision needed.",
              flush=True)
        return {'proven': True, 'method': 'cascade_only', 'info': info}

    # Step 2: Collect failing cells from each level
    # We need to find all compositions where pruning succeeded but box cert failed.
    # These are at the cascade's convergence level.
    conv_level = info['proven_at']  # e.g., 'L2'
    level_num = int(conv_level[1:])

    if verbose:
        print(f"\nCascade converged at {conv_level}. "
              f"Collecting failing cells...", flush=True)

    # The failing cells are at the convergence level's dimension
    # We need to re-run the final level to get per-composition nets
    # OR we can get them from the cascade's checkpoint

    # Reload the parent survivors from checkpoint
    d_conv = d0 * (2 ** level_num)

    # For now, re-run the final level's pruning to get per-composition data
    # Load the checkpoint of the level BEFORE convergence
    parent_level = level_num - 1
    parent_ckpt = f'data/checkpoint_L{parent_level}_survivors.npy'

    if not os.path.exists(parent_ckpt):
        print(f"Checkpoint {parent_ckpt} not found. "
              f"Re-run cascade with checkpoints.", flush=True)
        return {'proven': False, 'reason': 'no_checkpoint'}

    parents = np.load(parent_ckpt)
    d_parent = parents.shape[1]
    d_child = 2 * d_parent

    if verbose:
        print(f"  Convergence level: {conv_level} (d={d_child})")
        print(f"  Parents: {len(parents):,} at d={d_parent}")
        print(f"  Regenerating children to find failing cells...", flush=True)

    # Generate ALL children and find the ones with failing box cert
    from run_cascade import process_parent_coarse
    import math

    x_cap = int(math.floor(S * math.sqrt(c_target / d_child)))
    prefix_nk, prefix_mk = _build_pair_prefix(d_child)

    all_failing = []
    all_passing = 0
    total_children = 0

    for p_idx in range(len(parents)):
        parent = parents[p_idx]

        # Skip infeasible parents
        if np.any(parent > 2 * x_cap):
            continue

        survivors, n_ch, min_net = process_parent_coarse(
            parent, S, c_target, d_child)
        total_children += n_ch

        # All children should be pruned (0 survivors) at convergence level
        if len(survivors) > 0:
            print(f"  WARNING: {len(survivors)} survivors at parent {p_idx}!",
                  flush=True)
            continue

        # Re-certify this parent's children to find failing ones
        # process_parent_coarse already pruned them all, but we need per-child nets
        # Regenerate children and certify
        from run_cascade import _compute_bin_ranges_coarse
        result = _compute_bin_ranges_coarse(parent, S, c_target, d_child)
        if result is None:
            continue
        lo_arr, hi_arr, n_total = result

        # Generate children explicitly
        d_p = len(parent)
        children = []
        cursor = lo_arr.copy()
        while True:
            child = np.empty(d_child, dtype=np.int32)
            for i in range(d_p):
                child[2 * i] = cursor[i]
                child[2 * i + 1] = parent[i] - cursor[i]
            children.append(child.copy())

            # Advance odometer
            carry = d_p - 1
            while carry >= 0:
                cursor[carry] += 1
                if cursor[carry] <= hi_arr[carry]:
                    break
                cursor[carry] = lo_arr[carry]
                carry -= 1
            if carry < 0:
                break

        if not children:
            continue

        batch = np.array(children, dtype=np.int32)
        pruned, nets = _certify_batch(batch, d_child, S, c_target,
                                      prefix_nk, prefix_mk)

        n_pass = int((nets >= 0).sum())
        n_fail = len(batch) - n_pass
        all_passing += n_pass
        if n_fail > 0:
            failing = batch[nets < 0]
            all_failing.append(failing)

        if verbose and (p_idx + 1) % 100 == 0:
            n_f = sum(len(f) for f in all_failing)
            print(f"    [{p_idx+1}/{len(parents)}] "
                  f"{n_f:,} failing, {all_passing:,} passing", flush=True)

    if all_failing:
        failing_comps = np.vstack(all_failing)
        # Deduplicate
        failing_comps = np.unique(failing_comps, axis=0)
    else:
        failing_comps = np.empty((0, d_child), dtype=np.int32)

    n_failing = len(failing_comps)
    if verbose:
        print(f"\n  Total children: {total_children:,}")
        print(f"  Box cert pass: {all_passing:,}")
        print(f"  Box cert fail: {n_failing:,}", flush=True)

    if n_failing == 0:
        print(f"\n  All children pass box cert! (unexpected)", flush=True)
        return {'proven': True, 'method': 'cascade_only'}

    # Step 3: Subdivide failing cells
    result = subdivide_and_certify(
        failing_comps, d_child, S, c_target,
        max_rounds=max_subdiv_rounds, verbose=verbose)

    elapsed_total = time.time() - t_total

    if result['certified']:
        if verbose:
            print(f"\n{'='*60}")
            print(f"*** RIGOROUS PROOF: C_{{1a}} >= {c_target} ***")
            print(f"  Method: cascade (d0={d0}, S={S}) + subdivision")
            print(f"  Cascade converged at {conv_level} (d={d_child})")
            print(f"  Subdivision: {len(result['rounds'])} rounds "
                  f"to S={result['final_S']}")
            print(f"  Total sub-cells checked: "
                  f"{result['total_subcells']:,}")
            print(f"  Total time: {elapsed_total:.1f}s")
            print(f"{'='*60}", flush=True)
        return {
            'proven': True,
            'method': 'cascade_plus_subdivision',
            'cascade_info': info,
            'subdivision_info': result,
            'total_time': elapsed_total,
        }
    else:
        if verbose:
            print(f"\n  Subdivision did not converge: {result.get('reason')}")
            print(f"  Total time: {elapsed_total:.1f}s", flush=True)
        return {
            'proven': False,
            'reason': result.get('reason'),
            'subdivision_info': result,
            'total_time': elapsed_total,
        }


# =====================================================================
# CLI
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Cascade + subdivision proof for C_{1a} >= c_target')
    parser.add_argument('--d0', type=int, default=4)
    parser.add_argument('--S', type=int, default=30)
    parser.add_argument('--c_target', type=float, default=1.30)
    parser.add_argument('--max_cascade', type=int, default=8)
    parser.add_argument('--max_subdiv', type=int, default=6)
    args = parser.parse_args()

    result = prove_with_subdivision(
        d0=args.d0, S=args.S, c_target=args.c_target,
        max_cascade_levels=args.max_cascade,
        max_subdiv_rounds=args.max_subdiv)

    if result.get('proven'):
        print(f"\nSUCCESS: C_{{1a}} >= {args.c_target}")
    else:
        print(f"\nFAILED: {result.get('reason')}")


if __name__ == '__main__':
    main()
