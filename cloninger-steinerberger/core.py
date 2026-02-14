"""Core routines for Cloninger-Steinerberger branch-and-prune algorithm.

Proves lower bounds on C_{1a} = inf ||f*f||_inf / ||f||_1^2
for nonneg f supported on [-1/4, 1/4].

Reference: Cloninger & Steinerberger, "On Suprema of Autoconvolutions
with an Application to Sidon Sets", Proc. AMS 145(8), 2017.

Algorithm:
  1. Partition [-1/4, 1/4] into d = 2*n_half equal bins.
  2. For any nonneg f, define masses a_i = 4*n_half * integral_{I_i} f.
     These satisfy a_i >= 0, sum a_i = 4*n_half.
  3. Key inequality (Lemma 1): for any f with these masses,
       ||f*f||_inf >= max_{ell,k} (1/(4*n*ell)) * sum_{k<=i+j<=k+ell-2} a_i a_j
     This holds because f_i*f_j has support contained in I_i + I_j, so
     summing products over pairs whose Minkowski sums fall inside a window
     gives a lower bound on the integral of f*f over that window.
  4. Discretize to grid B_{n,m} (a_i = multiples of 1/m).
     Lemma 3: C_{1a} >= b_{n,m} - 2/m - 1/m^2.
  5. If ALL grid points have test value > target + correction, bound is proven.

Conventions:
  n_half: the paper's 'n'. Bins: d = 2*n_half, indexed 0..d-1.
  Original paper indices: -n..-1, 0..n-1  ->  numpy: 0..d-1.
  Masses a_i in paper coordinates (sum = 4*n_half).
  Integer coordinates: c_i = m * a_i (non-neg integers, sum = 4*n_half*m).
"""
import numpy as np
from math import comb
import time


def correction(m):
    """Discretization error bound (Lemma 3): C_{1a} >= b_{n,m} - 2/m - 1/m^2."""
    return 2.0 / m + 1.0 / (m * m)


def asymmetry_threshold(c_target):
    """Minimum left-mass fraction for the asymmetry argument to give >= c_target.

    If left_mass_frac >= threshold (or right_mass_frac >= threshold),
    then ||f*f||_inf >= 2 * threshold^2 >= c_target.
    """
    return np.sqrt(c_target / 2.0)


def count_compositions(d, S):
    """Number of non-negative integer vectors of length d summing to S.
    Equals C(S + d - 1, d - 1).
    """
    return comb(S + d - 1, d - 1)


def _gen_compositions(d, S):
    """Recursively generate all compositions of S into d non-negative parts."""
    if d == 1:
        yield (S,)
        return
    for c in range(S + 1):
        for rest in _gen_compositions(d - 1, S - c):
            yield (c,) + rest


def _gen_compositions_inner3(R):
    """Generate all (c0,c1,c2) with c0+c1+c2 = R as numpy array. Fast."""
    rows = []
    for c0 in range(R + 1):
        n = R - c0 + 1
        c1 = np.arange(n, dtype=np.int32)
        c2 = (R - c0) - c1
        c0_col = np.full(n, c0, dtype=np.int32)
        rows.append(np.column_stack([c0_col, c1, c2]))
    return np.vstack(rows)


def generate_compositions_batched(d, S, batch_size=100000):
    """Yield batches of compositions as numpy int32 arrays.

    Uses vectorized inner dimensions for speed. For d>=4, the innermost
    3 dimensions are generated with numpy; outer dimensions use Python loops.
    """
    if d <= 3:
        if d == 1:
            yield np.array([[S]], dtype=np.int32)
            return
        if d == 2:
            c0 = np.arange(S + 1, dtype=np.int32)
            yield np.column_stack([c0, S - c0])
            return
        # d == 3
        yield _gen_compositions_inner3(S)
        return

    # d >= 4: iterate outer d-3 dims in Python, vectorize inner 3
    d_outer = d - 3
    buf = []
    buf_len = 0

    def _outer(depth, remaining, prefix):
        nonlocal buf, buf_len
        if depth == d_outer:
            inner = _gen_compositions_inner3(remaining)
            n = len(inner)
            if n == 0:
                return
            pre_cols = np.tile(np.array(prefix, dtype=np.int32), (n, 1))
            chunk = np.hstack([pre_cols, inner])
            buf.append(chunk)
            buf_len += n
            return
        for c in range(remaining + 1):
            _outer(depth + 1, remaining - c, prefix + [c])

    _outer(0, S, [])

    # Yield accumulated buffer in batch_size chunks
    if buf:
        all_data = np.vstack(buf)
        for i in range(0, len(all_data), batch_size):
            yield all_data[i:i + batch_size]


def compute_test_values_batch(batch_int, n_half, m):
    """Compute max test value for a batch of integer mass vectors.

    For each config, the test value is:
      max over all windows (ell, s_lo) of
        (1/(4*n_half*ell)) * sum_{s_lo <= k <= s_lo+ell-2} conv[k]
    where conv is the autoconvolution in a-coordinates (a_i = c_i/m).

    Parameters
    ----------
    batch_int : (B, d) int32 array
        Integer mass coordinates (c_i = m * a_i, sum = 4*n_half*m).
    n_half : int
        Paper's n.
    m : int
        Grid resolution.

    Returns
    -------
    (B,) float64 array of test values.
    """
    B, d = batch_int.shape
    batch = batch_int.astype(np.float64) / m  # Convert to a-coordinates

    # Autoconvolution: conv[k] = sum_{i+j=k} a[i]*a[j], k = 0..2d-2
    conv_len = 2 * d - 1
    conv = np.zeros((B, conv_len), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            conv[:, i + j] += batch[:, i] * batch[:, j]

    # Prefix sums for O(1) window queries
    cumconv = np.cumsum(conv, axis=1)

    test_vals = np.zeros(B, dtype=np.float64)

    # Check all valid windows:
    #   ell from 2 to d (larger ell approach trivial bound 1.0)
    #   window covers ell-1 consecutive conv values starting at s_lo
    #   physical window width = ell/(4*n_half)
    #   test = window_sum / (4*n_half*ell)
    for ell in range(2, d + 1):
        n_conv_vals = ell - 1  # Number of conv values in window
        for s_lo in range(conv_len - n_conv_vals + 1):
            s_hi = s_lo + n_conv_vals - 1
            ws = cumconv[:, s_hi].copy()
            if s_lo > 0:
                ws -= cumconv[:, s_lo - 1]
            tv = ws / (4.0 * n_half * ell)
            test_vals = np.maximum(test_vals, tv)

    return test_vals


def compute_test_value_single(a, n_half):
    """Compute test value for a single mass vector (a-coordinates, float)."""
    a = np.asarray(a, dtype=np.float64)
    batch = a.reshape(1, -1)
    m_dummy = 1  # a is already in a-coordinates
    # We need to pass integer coords, so multiply by a large m and round
    # Actually, just use the batch function with m=1 and batch = a as floats
    # Hack: pass batch directly as float and m=1
    d = len(a)
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.float64)
    for i in range(d):
        for j in range(d):
            conv[i + j] += a[i] * a[j]

    cumconv = np.cumsum(conv)
    best = 0.0
    for ell in range(2, d + 1):
        n_conv_vals = ell - 1
        for s_lo in range(conv_len - n_conv_vals + 1):
            s_hi = s_lo + n_conv_vals - 1
            ws = cumconv[s_hi]
            if s_lo > 0:
                ws -= cumconv[s_lo - 1]
            tv = ws / (4.0 * n_half * ell)
            if tv > best:
                best = tv
    return best


def asymmetry_prune_mask(batch_int, n_half, m, c_target):
    """Return boolean mask: True for configs NOT covered by asymmetry argument.

    The asymmetry argument: if left_mass_frac >= threshold or
    right_mass_frac >= threshold, then ||f*f||_inf >= c_target automatically.

    A grid point b with discrete left fraction p_b covers continuous functions
    whose left fraction can differ by up to 1/(4m) (from |a_i - b_i| <= 1/m
    across n bins, divided by total mass 4n). We can only prune b if ALL
    continuous functions it covers have extreme enough mass for asymmetry,
    i.e., p_b >= threshold + margin (or p_b <= 1 - threshold - margin).

    Returns True for configs that NEED test-value checking.
    """
    d = 2 * n_half
    total = float(4 * n_half * m)
    threshold = asymmetry_threshold(c_target)

    # Discretization margin: continuous left_frac can differ from discrete
    # by up to n_half * (1/m) / (4*n_half) = 1/(4m).
    margin = 1.0 / (4.0 * m)

    left = batch_int[:, :n_half].sum(axis=1).astype(np.float64)
    left_frac = left / total

    # Asymmetry safely covers: left_frac >= threshold + margin
    #                       or left_frac <= (1 - threshold) - margin
    # Need checking: everything else
    safe_threshold = threshold + margin
    needs_check = (left_frac > 1 - safe_threshold) & (left_frac < safe_threshold)
    return needs_check


def run_single_level(n_half, m, c_target, batch_size=100000, verbose=True):
    """Run the branch-and-prune at a single discretization level.

    Enumerates all grid points in B_{n,m}, applies asymmetry pruning and
    test-value pruning. If ALL grid points are pruned, the bound is proven.

    Parameters
    ----------
    n_half : int
        Paper's n. Number of bins = 2*n_half.
    m : int
        Grid resolution. Masses are multiples of 1/m.
    c_target : float
        Target lower bound for C_{1a}.
    batch_size : int
        Batch size for enumeration.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        proven : bool
        c_proven : float or None
        n_survivors : int
        min_test_val : float
        min_test_config : array or None
        stats : dict
    """
    d = 2 * n_half
    S = 4 * n_half * m
    n_total = count_compositions(d, S)
    corr = correction(m)
    prune_target = c_target + corr  # Test value must exceed this to prune

    asym_thresh = asymmetry_threshold(c_target)

    if verbose:
        print(f"Level n={n_half}, m={m}: d={d} bins, S={S}")
        print(f"  Grid points: {n_total:,}")
        print(f"  Correction: {corr:.6f}")
        print(f"  Target: C_{{1a}} >= {c_target:.4f}")
        print(f"  Prune threshold: test_val > {prune_target:.6f}")
        print(f"  Asymmetry threshold: left_frac >= {asym_thresh:.4f}")

    t0 = time.time()
    n_processed = 0
    n_pruned_asym = 0
    n_pruned_test = 0
    n_survived = 0
    min_test_val = float('inf')
    min_test_config = None
    survivor_configs = []

    for batch_int in generate_compositions_batched(d, S, batch_size):
        B = len(batch_int)

        # Asymmetry pruning
        needs_check = asymmetry_prune_mask(batch_int, n_half, m, c_target)
        n_asym_pruned_here = B - int(needs_check.sum())
        n_pruned_asym += n_asym_pruned_here

        if needs_check.sum() > 0:
            check_batch = batch_int[needs_check]

            # Test value pruning (with float64 safety margin to avoid
            # ruling out a config due to rounding error)
            tvs = compute_test_values_batch(check_batch, n_half, m)
            fp_margin = 1e-9
            pruned = tvs > prune_target + fp_margin
            n_pruned_test += int(pruned.sum())

            survived = ~pruned
            n_survived_here = int(survived.sum())
            if n_survived_here > 0:
                n_survived += n_survived_here
                batch_survivors = check_batch[survived]
                batch_tvs = tvs[survived]
                survivor_configs.append(batch_survivors)

                idx = np.argmin(batch_tvs)
                if batch_tvs[idx] < min_test_val:
                    min_test_val = float(batch_tvs[idx])
                    min_test_config = batch_survivors[idx].copy()

        n_processed += B
        if verbose and n_processed % (batch_size * 10) == 0:
            elapsed = time.time() - t0
            pct = 100.0 * n_processed / n_total
            rate = n_processed / elapsed if elapsed > 0 else 0
            eta = (n_total - n_processed) / rate if rate > 0 else 0
            print(f"  [{pct:5.1f}%] processed={n_processed:,} "
                  f"asym={n_pruned_asym:,} test={n_pruned_test:,} "
                  f"survive={n_survived:,} | {elapsed:.0f}s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    proven = n_survived == 0
    c_proven = c_target if proven else None

    if verbose:
        print(f"\n  Completed in {elapsed:.1f}s ({n_processed:,} configs)")
        print(f"  Asymmetry pruned: {n_pruned_asym:,} "
              f"({100*n_pruned_asym/max(1,n_total):.1f}%)")
        print(f"  Test pruned: {n_pruned_test:,} "
              f"({100*n_pruned_test/max(1,n_total):.1f}%)")
        print(f"  Survivors: {n_survived:,} "
              f"({100*n_survived/max(1,n_total):.1f}%)")
        if proven:
            print(f"  >>> PROVEN: C_{{1a}} >= {c_target:.6f} <<<")
        else:
            print(f"  NOT proven at target {c_target:.4f}")
            if min_test_config is not None:
                a_cfg = min_test_config.astype(np.float64) / m
                print(f"  Min test value: {min_test_val:.6f}")
                print(f"  Min config (a-coords): {a_cfg}")
                print(f"  Min config (mass frac): {a_cfg / (4*n_half)}")

    all_survivors = (np.vstack(survivor_configs) if survivor_configs
                     else np.empty((0, d), dtype=np.int32))

    return {
        'proven': proven,
        'c_proven': c_proven,
        'n_survivors': n_survived,
        'survivors': all_survivors,
        'min_test_val': min_test_val,
        'min_test_config': min_test_config,
        'stats': {
            'n_half': n_half, 'm': m, 'd': d, 'S': S,
            'n_total': n_total, 'n_processed': n_processed,
            'n_pruned_asym': n_pruned_asym,
            'n_pruned_test': n_pruned_test,
            'n_survived': n_survived,
            'elapsed': elapsed,
            'prune_target': prune_target,
            'correction': corr,
            'asym_threshold': asym_thresh,
        },
    }


def find_best_bound(n_half, m, lo=1.0, hi=1.5, tol=0.001,
                     batch_size=100000, verbose=True):
    """Binary search for the best provable bound at given (n, m).

    Finds the largest c_target such that run_single_level proves C_{1a} >= c_target.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Binary search: n={n_half}, m={m}, range=[{lo:.4f}, {hi:.4f}]")
        print(f"{'='*70}")

    best_proven = None

    while hi - lo > tol:
        mid = (lo + hi) / 2
        if verbose:
            print(f"\n--- Trying c_target = {mid:.6f} ---")
        result = run_single_level(n_half, m, mid,
                                   batch_size=batch_size, verbose=verbose)
        if result['proven']:
            lo = mid
            best_proven = mid
        else:
            hi = mid

    if verbose:
        print(f"\n{'='*70}")
        if best_proven is not None:
            print(f"Best proven bound: C_{{1a}} >= {best_proven:.6f}")
        else:
            print(f"Could not prove any bound in [{lo:.4f}, {hi:.4f}]")
        print(f"{'='*70}")

    return best_proven
