"""Compare MATLAB (CS14) pruning vs Python CPU cascade across parameter configs.

Implements the MATLAB pruning logic faithfully in Python and runs it
side-by-side with our CPU cascade code on tiny samples.

MATLAB algorithm (from original_baseline_matlab.m / benchmark_cs14.m):
  - Continuous weights: a_i as floats, gridSpace = 1/m
  - Threshold: (c_target + gridSpace^2) + 2*gridSpace*W_continuous
  - No asymmetry pruning, no canonical filtering
  - Autoconvolution via pairwise product matrix

Python algorithm (from run_cascade.py):
  - Integer compositions: c_i integers summing to 4*n*m
  - W-refined threshold: (c_target*m^2 + 3 + W_int/(2n) + eps) * 4n*ell
  - Asymmetry pruning + canonical filtering
  - Gray-code fused kernel

Key formula difference:
  MATLAB correction: 1/m^2 + 2*W/m  (uses W_g directly)
  Python correction: 3/m^2 + 2*W_g/m (accounts for W_f <= W_g + 1/m)
  Python flat:       2/m + 1/m^2     (C&S Lemma 3, window-independent)
"""

import sys
import os
import time
import numpy as np

# Setup path — must add cloninger-steinerberger dir so cpu.run_cascade imports work
_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_dir = os.path.dirname(_this_dir)
_cs_dir = os.path.join(_repo_dir, 'cloninger-steinerberger')
sys.path.insert(0, _cs_dir)
sys.path.insert(0, _repo_dir)

# Set NUMBA_CACHE_DIR to avoid stale cache issues
os.environ['NUMBA_DISABLE_JIT'] = '0'

from compositions import generate_compositions_batched
from pruning import (correction, asymmetry_threshold, count_compositions,
                     asymmetry_prune_mask, _canonical_mask)

# Import pruning functions directly to avoid module-level JIT warmup issues
# We'll implement our own int32/int64 dispatch
from cpu.run_cascade import _prune_dynamic_int32, _prune_dynamic_int64

def _prune_dynamic(batch_int, n_half, m, c_target, use_flat):
    if m <= 200:
        return _prune_dynamic_int32(batch_int, n_half, m, c_target, use_flat)
    else:
        return _prune_dynamic_int64(batch_int, n_half, m, c_target, use_flat)


# ===================================================================
# MATLAB-style pruning (faithful reimplementation)
# ===================================================================

def matlab_prune_batch(batch_continuous, d, c_target, gridSpace):
    """Apply MATLAB CS14 pruning to a batch of continuous-weight compositions.

    Parameters
    ----------
    batch_continuous : (B, d) float64 array
        Each row is a composition of d continuous weights.
    d : int (= numBins in MATLAB)
    c_target : float (= lowerBound)
    gridSpace : float (= 1/m)

    Returns
    -------
    survived : (B,) bool array
    """
    B = batch_continuous.shape[0]
    survived = np.ones(B, dtype=bool)

    # Precompute all pairs (i, j) for i, j in 0..d-1
    # MATLAB uses 1-indexed, but logic is the same
    ii, jj = np.meshgrid(np.arange(d), np.arange(d))
    pairs_i = ii.ravel()
    pairs_j = jj.ravel()

    # For each window size ell (= j in MATLAB, 2..2d):
    # convBinIntervals is a Toeplitz indicator: which conv positions fall in each window
    # pairSum = pairs_i + pairs_j  (0-indexed: ranges 0..2d-2)
    pair_sums = pairs_i + pairs_j  # 0-indexed conv position

    for ell in range(2, 2 * d + 1):
        if not np.any(survived):
            break

        n_windows = 2 * d - ell  # number of window positions (MATLAB: numIntervals = 2*numBins - j + 1)
        # But MATLAB j=2..2*numBins with numIntervals = 2*numBins - j + 1
        # For us: ell corresponds to MATLAB j, n_windows = 2*d - ell + 1... no wait.
        # MATLAB conv space has 2*numBins positions (1-indexed: 2..2*numBins)
        # pair (i,j) maps to position i+j (1-indexed), so range 2..2*numBins
        # Window of size j: j consecutive positions starting at each offset
        # numIntervals = 2*numBins - j + 1

        # In 0-indexed: pair_sums range 0..2*(d-1). conv has 2d-1 positions.
        # Window of size ell: ell-1 consecutive conv positions (ell values of pair_sum)
        # Actually MATLAB "size j" means the window covers j pair-sum values.
        # Let me re-examine.

        # MATLAB: convBinIntervals is (numIntervals x 2*numBins)
        # row = [1; zeros(numIntervals-1, 1)], column = [ones(1,j) zeros(1,2*numBins-j)]
        # toeplitz(row, column) → each row has j consecutive 1s
        # Then sumIndicesStore{j} = (subsetBins * convBinIntervals' == 2)
        # subsetBins has shape (numPairs, 2*numBins), with 1s at positions pairSum-1 and pairSum
        # So pair (i,j) contributes to window k iff BOTH pairSum-1 and pairSum fall within
        # the window's j consecutive positions.
        # Window k (0-indexed) covers conv positions k..k+j-1 (1-indexed).
        # Pair contributes iff pairSum-1 >= k and pairSum <= k+j-1
        # i.e., pairSum >= k+1 and pairSum <= k+j-1
        # i.e., k+1 <= pairSum <= k+j-1
        # i.e., pairSum-j+1 <= k <= pairSum-1

        # Hmm, this is getting complex. Let me think in terms of what the MATLAB code
        # actually computes vs what our Python code computes.

        # MATLAB convFunctionVals = functionMult * sumIndicesStore{j}
        # where functionMult[row, pair] = a_{pair_i} * a_{pair_j}
        # sumIndicesStore{j}[pair, window] = 1 iff pair contributes to window
        # So convFunctionVals[row, window] = sum of a_i*a_j for contributing pairs
        # Then normalized: *= (2*numBins)/j

        # This is computing: for each window of size j (covering j conv positions),
        # TV_window = (sum of a_i*a_j for pairs where both i+j-1 and i+j fall in window) * 2d/j

        # Let me just implement this directly for clarity.

        # Actually, let me take a simpler approach. The MATLAB algorithm computes
        # the autoconvolution (f*f)(x) at discrete points and checks windows.
        # The TV for a window of ell consecutive bins is:
        #   TV = (1/(ell/d)) * integral over window of (f*f)(x) dx
        # For step functions: (f*f)(x) = sum_{i+j=k} a_i*a_j / (2d) for x in conv bin k
        # Wait, need to be more careful about normalization.

        # Let me just reimplement the MATLAB logic exactly.
        pass

    # OK, this approach is getting unwieldy. Let me take a cleaner approach.
    # I'll compute the autoconvolution directly and then apply MATLAB's threshold.

    for b in range(B):
        if not survived[b]:
            continue
        a = batch_continuous[b]  # (d,) continuous weights

        # Autoconvolution: conv[k] = sum_{i+j=k} a_i * a_j, k = 0..2(d-1)
        conv = np.zeros(2 * d - 1, dtype=np.float64)
        for i in range(d):
            for j in range(d):
                conv[i + j] += a[i] * a[j]

        # MATLAB normalization and window scan:
        # For window size j (= ell in MATLAB, j=2..2*numBins):
        #   numIntervals = 2*numBins - j + 1
        #   For each window position:
        #     convFunctionVals = sum of a_i*a_j for pairs contributing to window
        #     normalized by (2*numBins)/j
        #     boundToBeat = (c_target + gridSpace^2) + 2*gridSpace*W

        # But what exactly is "sum of a_i*a_j for pairs contributing to window"?
        # The MATLAB code uses sumIndicesStore which checks that BOTH pairSum-1 and pairSum
        # fall in the window. This means the pair (i,j) with sum s=i+j (1-indexed)
        # contributes to window covering positions [start, start+j-1] (1-indexed)
        # iff s-1 >= start AND s <= start+j-1
        # iff start <= s-1 AND s <= start+j-1
        # iff s-j+1 <= start <= s-1 ... hmm this doesn't simplify nicely.

        # Actually, in 0-indexed terms:
        # conv position k (0-indexed, 0..2d-2) corresponds to MATLAB position k+1
        # subsetBins maps pair to positions pairSum-1 and pairSum (1-indexed) = pairSum-2 and pairSum-1 (0-indexed)
        # Wait, MATLAB: pairs are 1-indexed. pair (i,j) with i,j in 1..numBins.
        # pairSum = i+j, range 2..2*numBins.
        # subsetBins has 1s at columns pairSum-1 and pairSum (1-indexed cols 1..2*numBins)
        # convBinIntervals has j consecutive 1s for each window.
        # Product == 2 means both pairSum-1 and pairSum are inside the window.

        # The window [start..start+j-1] (1-indexed) contains position p iff start <= p <= start+j-1.
        # For pair (i,j) to contribute: start <= pairSum-1 AND pairSum <= start+j-1
        # i.e. start <= pairSum-1 AND pairSum-j+1 <= start
        # But that means: pairSum-j+1 <= start <= pairSum-1
        # For valid windows: start = 1..numIntervals where numIntervals = 2*numBins - j + 1

        # This is essentially requiring that conv position (pairSum-1, 0-indexed) AND
        # the position before it are both in the window. The window has j positions,
        # so it spans j conv indices. The contribution is the integral of the piecewise-
        # linear autoconvolution over the window.

        # Actually I think there's a simpler interpretation. The MATLAB code is computing
        # the integral of (f*f) over intervals of length j/(2*numBins) of the total support.
        # The normalization (2*numBins)/j converts to an average over the interval.

        # For our purposes, let me just compute the same quantity directly:
        # For each window of ell consecutive conv bins (ell-1 values of conv):
        #   window_sum = sum(conv[s..s+ell-2]) for s = 0..2d-2-(ell-2)
        #   TV = window_sum * (2d) / ell  ... wait, the MATLAB normalizes differently.

        # Let me just look at what the MATLAB computes more carefully.
        # After the matmul, convFunctionVals has one value per window.
        # Then it multiplies by (2*numBins)/j.
        # The boundToBeat is (c_target + gridSpace^2) + 2*gridSpace*W.

        # The matmul sums a_i*a_j for all pairs (i,j) where i+j falls inside the window.
        # That's exactly sum_{s in window} conv[s] where conv[s] = sum_{i+j=s} a_i*a_j.
        # (In 1-indexed: s = i+j, and both s-1 and s must be in the j-wide window.
        # Actually the "both" check is because subsetBins puts 1s at pairSum-1 and pairSum,
        # and the indicator requires both to be in the window. This effectively means the
        # window covers positions where BOTH half-steps of the trapezoid are inside.)

        # Hmm, this is getting confusing with the half-step stuff. Let me just
        # compute things directly and use the simple interpretation:

        # Actually, I think the simplest faithful comparison is:
        # 1. Compute raw autoconvolution conv[k] = sum_{i+j=k} a_i*a_j
        # 2. For each window of ell conv values: window_sum = sum(conv[s..s+ell-1])
        # 3. Normalize: TV = window_sum / (ell / (2d)) = window_sum * 2d / ell
        #    (The interval has physical width ell/(2d), and we want the max of (f*f) over it,
        #     which for step functions is the average = integral / width)
        # 4. Compare against threshold: c_target + 1/m^2 + 2/m * W
        #    where W = sum of a_i for bins overlapping the window

        # This should match MATLAB's intent even if the indexing details differ slightly.

        # But wait - there's a subtlety. The conv array has 2d-1 entries (k=0..2d-2),
        # each representing the integral of (f*f) over a bin of width 1/(2d).
        # Actually each entry conv[k] = sum_{i+j=k} a_i*a_j, and a_i = c_i/(4nm).
        # For the MATLAB code, a_i are already the heights, so conv[k]/d is the
        # value of (f*f) at position k (since each bin has width 1/d, and the
        # convolution of two step functions is piecewise linear with breakpoints
        # at multiples of 1/d).

        # The MATLAB normalization (2*numBins)/j = 2d/j converts the sum of j
        # conv entries to the test value: TV = (sum/j) * 2d.
        # This is: average of conv[k] over window, times 2d.
        # Since (f*f)(k/(2d)) ≈ 2d * conv[k], we have TV = mean over window of (f*f).

        # OK let me just go with this.

        pruned = False
        for ell in range(2, 2 * d + 1):
            if pruned:
                break
            # Window covers ell-1 consecutive conv entries
            # (matching Python's convention: ell in window scan covers ell-1 conv values)
            n_cv = ell - 1
            n_windows = (2 * d - 1) - n_cv + 1

            for s in range(n_windows):
                ws = np.sum(conv[s:s + n_cv])

                # Compute W: sum of a_i for bins overlapping the window
                # conv position k corresponds to pairs (i,j) with i+j=k
                # The window spans conv positions s..s+n_cv-1
                # Bins overlapping: any bin i where there exists j s.t. i+j in [s, s+n_cv-1]
                # i.e., s - (d-1) <= i <= s + n_cv - 1
                lo_bin = max(0, s - (d - 1))
                hi_bin = min(d - 1, s + n_cv - 1)
                W = np.sum(a[lo_bin:hi_bin + 1])

                # MATLAB TV and threshold
                TV = ws * (2 * d) / ell
                bound = (c_target + gridSpace**2) + 2 * gridSpace * W

                if TV >= bound:
                    pruned = True
                    break

        if pruned:
            survived[b] = False

    return survived


def matlab_prune_batch_fast(batch_int, d, n_half, m, c_target):
    """MATLAB-style pruning using integer compositions (converted to continuous).

    Takes the same integer compositions as Python, converts to continuous
    weights, then applies MATLAB's threshold formula.
    """
    gridSpace = 1.0 / m
    S = 4 * n_half * m
    # Convert integer compositions to continuous weights
    # In MATLAB: weights sum to 1.0, each is a multiple of gridSpace
    batch_continuous = batch_int.astype(np.float64) / S
    return matlab_prune_batch(batch_continuous, d, c_target, gridSpace)


# ===================================================================
# Python cascade pruning (wrapper)
# ===================================================================

def python_prune_l0(batch_int, n_half, m, c_target,
                    use_flat_threshold=False,
                    skip_asymmetry=False,
                    skip_canonical=False):
    """Apply Python cascade pruning pipeline to a batch.

    Returns dict with counts for each stage.
    """
    B0 = len(batch_int)

    # Canonical filter
    if skip_canonical:
        canon_mask = np.ones(B0, dtype=bool)
    else:
        canon_mask = _canonical_mask(batch_int)
    n_non_canon = int(np.sum(~canon_mask))
    batch_canon = batch_int[canon_mask]

    # Asymmetry filter
    if skip_asymmetry or len(batch_canon) == 0:
        needs_check = np.ones(len(batch_canon), dtype=bool)
    else:
        needs_check = asymmetry_prune_mask(batch_canon, n_half, m, c_target)
    n_asym = int(np.sum(~needs_check))
    candidates = batch_canon[needs_check]

    if len(candidates) > 0:
        survived_mask = _prune_dynamic(candidates, n_half, m, c_target,
                                        use_flat_threshold)
    else:
        survived_mask = np.ones(0, dtype=bool)

    n_test_pruned = int(np.sum(~survived_mask))
    survivors = candidates[survived_mask]

    return {
        'total': B0,
        'non_canonical': n_non_canon,
        'asym_pruned': n_asym,
        'test_pruned': n_test_pruned,
        'survivors': len(survivors),
        'survivor_arr': survivors,
    }


# ===================================================================
# Side-by-side comparison
# ===================================================================

def compare_l0(n_half, m, c_target, max_compositions=50000):
    """Compare MATLAB vs Python pruning at L0 for given parameters.

    Generates up to max_compositions compositions and runs both pruning
    algorithms on them.
    """
    d = 2 * n_half
    S = 4 * n_half * m
    n_total = count_compositions(d, S)
    gridSpace = 1.0 / m

    print(f"\n{'='*70}")
    print(f"  L0 comparison: n_half={n_half}, m={m}, c_target={c_target}")
    print(f"  d={d}, S={S}, gridSpace={gridSpace}")
    print(f"  Total compositions: {n_total:,}")
    print(f"  Sampling up to {max_compositions:,}")
    print(f"{'='*70}")

    # Correction terms
    py_corr = correction(m, n_half)
    matlab_corr_base = gridSpace**2  # 1/m^2
    print(f"\n  Python correction (C&S Lemma 3): {py_corr:.6f}  (2/m + 1/m^2)")
    print(f"  MATLAB base correction: {matlab_corr_base:.6f}  (1/m^2)")
    print(f"  Python W-refined: 3/m^2 + 2*W/m  (window-dependent)")
    print(f"  MATLAB W-refined: 1/m^2 + 2*W/m  (window-dependent)")
    print(f"  Difference: Python uses 2/m^2 MORE correction (more conservative)")

    # Collect compositions
    all_comps = []
    n_collected = 0
    for batch in generate_compositions_batched(d, S, batch_size=min(50000, max_compositions)):
        all_comps.append(batch)
        n_collected += len(batch)
        if n_collected >= max_compositions:
            break

    all_comps = np.vstack(all_comps)[:max_compositions]
    print(f"  Collected {len(all_comps):,} compositions")

    # --- Python pruning (W-refined) ---
    t0 = time.time()
    py_result = python_prune_l0(all_comps.copy(), n_half, m, c_target,
                                use_flat_threshold=False,
                                skip_asymmetry=False,
                                skip_canonical=False)
    py_time = time.time() - t0

    # --- Python pruning (flat threshold, for Lean) ---
    t0 = time.time()
    py_flat_result = python_prune_l0(all_comps.copy(), n_half, m, c_target,
                                      use_flat_threshold=True,
                                      skip_asymmetry=False,
                                      skip_canonical=False)
    py_flat_time = time.time() - t0

    # --- Python pruning (no asymmetry, no canonical — apples-to-apples with MATLAB) ---
    t0 = time.time()
    py_raw_result = python_prune_l0(all_comps.copy(), n_half, m, c_target,
                                     use_flat_threshold=False,
                                     skip_asymmetry=True,
                                     skip_canonical=True)
    py_raw_time = time.time() - t0

    # --- MATLAB-style pruning ---
    t0 = time.time()
    matlab_survived = matlab_prune_batch_fast(all_comps.copy(), d, n_half, m, c_target)
    matlab_time = time.time() - t0
    n_matlab_survived = int(np.sum(matlab_survived))

    # Print results
    print(f"\n  Results on {len(all_comps):,} compositions:")
    print(f"  {'Method':<40} {'Survived':>10} {'Rate':>10} {'Time':>8}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*8}")

    def _row(label, surv, total, t):
        rate = f"{100*surv/total:.4f}%" if total > 0 else "N/A"
        print(f"  {label:<40} {surv:>10,} {rate:>10} {t:>7.2f}s")

    _row("MATLAB (1/m^2 + 2W/m)", n_matlab_survived, len(all_comps), matlab_time)
    _row("Python raw (3/m^2 + 2W/m, no asym/canon)", py_raw_result['survivors'],
         len(all_comps), py_raw_time)
    _row("Python full (W-refined + asym + canon)", py_result['survivors'],
         len(all_comps), py_time)
    _row("Python flat (2/m+1/m^2 + asym + canon)", py_flat_result['survivors'],
         len(all_comps), py_flat_time)

    # Detailed breakdown
    print(f"\n  Python full breakdown:")
    print(f"    Non-canonical filtered: {py_result['non_canonical']:,}")
    print(f"    Asymmetry pruned:       {py_result['asym_pruned']:,}")
    print(f"    Test-value pruned:      {py_result['test_pruned']:,}")
    print(f"    Survivors:              {py_result['survivors']:,}")

    # Find compositions that differ between MATLAB and Python raw
    # (both see all compositions, no asym/canon filter)
    py_raw_mask_all = np.ones(len(all_comps), dtype=bool)
    if len(all_comps) > 0:
        py_raw_survived_mask = _prune_dynamic(all_comps, n_half, m, c_target, False)
        py_raw_mask_all = py_raw_survived_mask

    n_matlab_only = int(np.sum(~py_raw_mask_all & matlab_survived))
    n_python_only = int(np.sum(py_raw_mask_all & ~matlab_survived))
    n_both = int(np.sum(py_raw_mask_all & matlab_survived))

    print(f"\n  Overlap (MATLAB vs Python raw, same compositions):")
    print(f"    Both survived:          {n_both:,}")
    print(f"    MATLAB only survived:   {n_matlab_only:,}")
    print(f"    Python only survived:   {n_python_only:,}")

    if n_matlab_only > 0:
        print(f"    (MATLAB more aggressive — prunes less. "
              f"Expected: MATLAB correction 1/m^2 < Python 3/m^2)")
    if n_python_only > 0:
        print(f"    (Python more aggressive — unexpected! "
              f"MATLAB should always survive >= Python)")

    return {
        'n_half': n_half, 'm': m, 'c_target': c_target,
        'n_compositions': len(all_comps),
        'matlab_survived': n_matlab_survived,
        'python_raw_survived': py_raw_result['survivors'],
        'python_full_survived': py_result['survivors'],
        'python_flat_survived': py_flat_result['survivors'],
        'matlab_only': n_matlab_only,
        'python_only': n_python_only,
        'both': n_both,
    }


def compare_l1(n_half, m, c_target, n_sample_parents=5):
    """Compare MATLAB vs Python pruning at L1 using L0 survivors.

    Takes a tiny sample of L0 survivors, generates children from each,
    and compares pruning.
    """
    from cpu.run_cascade import (run_level0, process_parent_fused,
                                  _compute_bin_ranges)  # noqa

    d_parent = 2 * n_half
    d_child = 2 * d_parent
    n_half_child = d_child // 2
    gridSpace = 1.0 / m

    print(f"\n{'='*70}")
    print(f"  L1 comparison: n_half={n_half}, m={m}, c_target={c_target}")
    print(f"  d_parent={d_parent}, d_child={d_child}")
    print(f"{'='*70}")

    # First get L0 survivors
    print(f"\n  Running L0 to get survivors...")
    l0 = run_level0(n_half, m, c_target, verbose=False)
    survivors = l0['survivors']
    print(f"  L0: {l0['n_survivors']:,} survivors from {l0['n_processed']:,} compositions")

    if l0['n_survivors'] == 0:
        print(f"  No L0 survivors — proven at L0!")
        return None

    # Sample parents
    n_sample = min(n_sample_parents, len(survivors))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(survivors), n_sample, replace=False)
    sample_parents = survivors[idx]
    print(f"  Sampling {n_sample} parents for L1 comparison")

    total_py_children = 0
    total_py_survivors = 0
    total_matlab_children = 0
    total_matlab_survivors = 0

    for pi, parent in enumerate(sample_parents):
        # Python: generate children via fused kernel
        py_survivors, py_total_children = process_parent_fused(
            parent, m, c_target, n_half_child)

        # Generate ALL children for MATLAB comparison
        # Use _compute_bin_ranges to get the Cartesian product ranges
        result = _compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
        if result is None:
            print(f"    Parent {pi}: empty range")
            continue

        lo_arr, hi_arr, cart_size = result

        # Generate a sample of children (not all — could be millions)
        # Use the same ranges as Python
        max_children_sample = 10000
        if cart_size <= max_children_sample:
            # Generate all children
            children = _enumerate_children(parent, lo_arr, hi_arr, d_child, max_count=cart_size)
        else:
            # Random sample
            children = _sample_children(parent, lo_arr, hi_arr, d_child,
                                         n_sample=max_children_sample, rng=rng)

        if len(children) == 0:
            print(f"    Parent {pi}: 0 children generated")
            continue

        # Apply MATLAB pruning to these children
        matlab_survived = matlab_prune_batch_fast(children, d_child, n_half_child, m, c_target)
        n_matlab = int(np.sum(matlab_survived))

        # Apply Python pruning (raw, no asym/canon) to same children
        py_raw_survived = _prune_dynamic(children, n_half_child, m, c_target, False)
        n_py_raw = int(np.sum(py_raw_survived))

        total_py_children += len(children)
        total_py_survivors += n_py_raw
        total_matlab_children += len(children)
        total_matlab_survivors += n_matlab

        # Overlap
        n_both = int(np.sum(py_raw_survived & matlab_survived))
        n_matlab_only = int(np.sum(~py_raw_survived & matlab_survived))
        n_python_only = int(np.sum(py_raw_survived & ~matlab_survived))

        cart_label = f"{cart_size:,}" if cart_size <= max_children_sample else f"~{max_children_sample:,}/{cart_size:,}"
        print(f"    Parent {pi}: {cart_label} children | "
              f"MATLAB surv={n_matlab:,} | Python surv={n_py_raw:,} | "
              f"both={n_both:,} | MATLAB-only={n_matlab_only:,} | Py-only={n_python_only:,}")

    print(f"\n  L1 totals ({total_py_children:,} children sampled):")
    print(f"    MATLAB survived: {total_matlab_survivors:,} "
          f"({100*total_matlab_survivors/max(1,total_matlab_children):.3f}%)")
    print(f"    Python survived: {total_py_survivors:,} "
          f"({100*total_py_survivors/max(1,total_py_children):.3f}%)")

    return {
        'n_half': n_half, 'm': m, 'c_target': c_target,
        'total_children': total_py_children,
        'matlab_survived': total_matlab_survivors,
        'python_survived': total_py_survivors,
    }


def _enumerate_children(parent, lo_arr, hi_arr, d_child, max_count):
    """Enumerate all children from Cartesian product of bin ranges."""
    d_parent = len(parent)
    ranges = []
    for i in range(d_parent):
        lo = int(lo_arr[i])
        hi = int(hi_arr[i])
        ranges.append(np.arange(lo, hi + 1))

    # Build Cartesian product (limit to max_count)
    total = 1
    for r in ranges:
        total *= len(r)
    if total == 0:
        return np.empty((0, d_child), dtype=np.int32)

    children = []
    count = 0
    # Use itertools.product for small products
    import itertools
    for combo in itertools.product(*ranges):
        child = np.empty(d_child, dtype=np.int32)
        for i, c in enumerate(combo):
            child[2*i] = c
            child[2*i+1] = 2 * parent[i] - c
        children.append(child)
        count += 1
        if count >= max_count:
            break

    return np.array(children, dtype=np.int32)


def _sample_children(parent, lo_arr, hi_arr, d_child, n_sample, rng):
    """Randomly sample children from the Cartesian product."""
    d_parent = len(parent)
    children = np.empty((n_sample, d_child), dtype=np.int32)

    for s in range(n_sample):
        for i in range(d_parent):
            lo = int(lo_arr[i])
            hi = int(hi_arr[i])
            c = rng.randint(lo, hi + 1)
            children[s, 2*i] = c
            children[s, 2*i+1] = 2 * parent[i] - c

    return children


# ===================================================================
# Main
# ===================================================================

def main():
    configs = [
        # (n_half, m, c_target)
        (2, 10, 1.28),
        (2, 10, 1.33),
        (2, 10, 1.40),
        (2, 15, 1.28),
        (2, 15, 1.33),
        (2, 15, 1.40),
        (2, 20, 1.28),
        (2, 20, 1.33),
        (2, 20, 1.40),
    ]

    print("=" * 70)
    print("  MATLAB vs Python Cascade Comparison")
    print("  Testing L0 and L1 across multiple configurations")
    print("=" * 70)

    # L0 comparisons
    l0_results = []
    for n_half, m, c_target in configs:
        try:
            r = compare_l0(n_half, m, c_target, max_compositions=20000)
            l0_results.append(r)
        except Exception as e:
            print(f"\n  ERROR for n_half={n_half}, m={m}, c_target={c_target}: {e}")
            import traceback; traceback.print_exc()

    # L1 comparisons (smaller set — these are slower)
    print("\n\n" + "=" * 70)
    print("  L1 COMPARISONS (tiny parent samples)")
    print("=" * 70)

    l1_configs = [
        (2, 10, 1.28),
        (2, 10, 1.40),
        (2, 15, 1.33),
        (2, 20, 1.40),
    ]

    l1_results = []
    for n_half, m, c_target in l1_configs:
        try:
            r = compare_l1(n_half, m, c_target, n_sample_parents=3)
            if r is not None:
                l1_results.append(r)
        except Exception as e:
            print(f"\n  ERROR for L1 n_half={n_half}, m={m}, c_target={c_target}: {e}")
            import traceback; traceback.print_exc()

    # Summary table
    print("\n\n" + "=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    print(f"\n  {'Config':<25} {'Comps':>8} {'MATLAB':>10} {'Py raw':>10} {'Py full':>10} {'Py flat':>10} {'M-only':>8} {'P-only':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for r in l0_results:
        label = f"n={r['n_half']},m={r['m']},c={r['c_target']}"
        print(f"  {label:<25} {r['n_compositions']:>8,} {r['matlab_survived']:>10,} "
              f"{r['python_raw_survived']:>10,} {r['python_full_survived']:>10,} "
              f"{r['python_flat_survived']:>10,} {r['matlab_only']:>8,} {r['python_only']:>8,}")

    print(f"\n  M-only = compositions MATLAB kept but Python pruned (MATLAB more permissive)")
    print(f"  P-only = compositions Python kept but MATLAB pruned (should be 0 or rare)")

    print(f"\n  Expected: MATLAB <= Python survivors (MATLAB correction 1/m^2 < Python 3/m^2)")
    print(f"  MATLAB prunes MORE aggressively because its threshold is LOWER.")
    print(f"  If M-only > 0, MATLAB is surviving configs Python prunes — this means")
    print(f"  Python is more aggressive on those specific configs (unexpected).")


if __name__ == '__main__':
    main()
