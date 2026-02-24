"""Benchmark pre-filter bounds against existing checks and true test values.

Evaluates four valid lower bounds (Fix 1-4 from docs/repairing_pre_filtering.md)
for the windowed autoconvolution test value, comparing them against:
  - True test values (exact autoconvolution)
  - Existing pre-checks in the GPU/CPU solvers
  - Conservative and dynamic thresholds

Usage:
    python cloninger-steinerberger/cpu/benchmark.py
"""

import sys
import os
import time
import numpy as np
from math import comb

# Path setup
_this_dir = os.path.dirname(os.path.abspath(__file__))
_cs_dir = os.path.dirname(_this_dir)
_archive_cpu = os.path.join(_cs_dir, 'archive', 'cpu')
sys.path.insert(0, _archive_cpu)
sys.path.insert(0, _cs_dir)

from compositions import generate_canonical_compositions_batched
from pruning import correction, count_compositions, asymmetry_threshold


# =====================================================================
# Utility functions
# =====================================================================

def collect_all_canonical(d, S):
    """Collect all canonical compositions into a single (N, d) int32 array."""
    batches = []
    for batch in generate_canonical_compositions_batched(d, S, batch_size=500000):
        batches.append(batch.copy())
    return np.vstack(batches)


def batch_autoconv(a, d):
    """Compute autoconvolutions for a batch of a-coordinate vectors.

    Parameters
    ----------
    a : (N, d) float64 array of a-coordinates
    d : int

    Returns
    -------
    (N, 2*d-1) float64 array of autoconvolution values
    """
    conv_len = 2 * d - 1
    conv = np.zeros((a.shape[0], conv_len), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            conv[:, i + j] += a[:, i] * a[:, j]
    return conv


def enumerate_windows(d):
    """List all valid windows (ell, s_lo) for dimension d.

    ell ranges from 2 to d, and for each ell, s_lo ranges over all
    valid starting positions.
    """
    conv_len = 2 * d - 1
    windows = []
    for ell in range(2, d + 1):
        n_cv = ell - 1
        for s_lo in range(conv_len - n_cv + 1):
            windows.append((ell, s_lo))
    return windows


def true_window_tvs(conv, d, n_half):
    """Compute true test value for every window from autoconvolution.

    Returns dict mapping (ell, s_lo) -> (N,) float64 array.
    """
    conv_len = 2 * d - 1
    prefix = np.cumsum(conv, axis=1)
    result = {}
    for ell in range(2, d + 1):
        n_cv = ell - 1
        inv_norm = 1.0 / (4.0 * n_half * ell)
        for s_lo in range(conv_len - n_cv + 1):
            s_hi = s_lo + n_cv - 1
            ws = prefix[:, s_hi].copy()
            if s_lo > 0:
                ws -= prefix[:, s_lo - 1]
            result[(ell, s_lo)] = ws * inv_norm
    return result


# =====================================================================
# Existing pre-checks (simulate what the GPU/CPU solvers do)
# =====================================================================

def check_asymmetry(comps, n_half, m, c_target):
    """Bool mask: True = pruned by asymmetry argument.

    Checks if left-half mass fraction is extreme enough that the
    asymmetry argument guarantees ||f*f||_inf >= c_target.
    """
    threshold = asymmetry_threshold(c_target)
    margin = 1.0 / (4.0 * m)
    safe = threshold + margin
    left_frac = comps[:, :n_half].sum(axis=1).astype(np.float64) / m
    return (left_frac >= safe) | (left_frac <= 1.0 - safe)


def check_existing_window_prechecks(comps, d, n_half, m, thresh):
    """Bool mask: True = pruned by existing window-based pre-checks.

    Simulates the cheap pre-checks applied before full autoconvolution:
      D=4: ell=4 block-sum (s_lo=0,4), max-element ell=2,
           ell=3 exact windows (s_lo=0,5)
      D=6: ell=6 block-sum (s_lo=0,6), max-element ell=2
    """
    scale = 4.0 * n_half / m
    N = len(comps)
    pruned = np.zeros(N, dtype=bool)
    half = d // 2

    # Block-sum ell=d: left-half and right-half squared
    norm_d = 4.0 * n_half * d
    left_a = comps[:, :half].sum(axis=1).astype(np.float64) * scale
    right_a = comps[:, half:].sum(axis=1).astype(np.float64) * scale
    pruned |= (left_a ** 2 / norm_d > thresh)
    pruned |= (right_a ** 2 / norm_d > thresh)

    # Max element ell=2
    max_a = comps.max(axis=1).astype(np.float64) * scale
    pruned |= (max_a ** 2 / (4.0 * n_half * 2) > thresh)

    # D=4 specific: ell=3 exact window values at s_lo=0 and s_lo=5
    # These are computed algebraically without full autoconvolution
    if d == 4:
        a = comps.astype(np.float64) * scale
        # s_lo=0, ell=3: conv[0]+conv[1] = a0^2 + 2*a0*a1 = a0*(a0+2*a1)
        w0 = a[:, 0] * (a[:, 0] + 2 * a[:, 1])
        pruned |= (w0 / (4.0 * n_half * 3) > thresh)
        # s_lo=5, ell=3: conv[5]+conv[6] = 2*a2*a3 + a3^2 = a3*(2*a2+a3)
        w5 = a[:, 3] * (2 * a[:, 2] + a[:, 3])
        pruned |= (w5 / (4.0 * n_half * 3) > thresh)

    return pruned


# =====================================================================
# Fix 2: Squared Interval Bound
# =====================================================================

def fix2_interval(s_lo, ell, d):
    """Compute Fix 2 interval [A_lo, A_hi] for window (ell, s_lo).

    A = {i : ceil(s_lo/2) <= i <= floor((s_lo+ell-2)/2)} clamped to [0, d-1].
    """
    s_hi = s_lo + ell - 2
    A_lo = (s_lo + 1) // 2     # ceil(s_lo / 2)
    A_hi = s_hi // 2           # floor(s_hi / 2)
    A_lo = max(0, min(A_lo, d - 1))
    A_hi = max(0, min(A_hi, d - 1))
    return A_lo, A_hi


def fix2_bound(a, d, n_half, ell, s_lo):
    """Fix 2 bound for window (ell, s_lo).

    Returns (N,) array: (sum_{i in A} a_i)^2 / (4*n*ell).
    """
    A_lo, A_hi = fix2_interval(s_lo, ell, d)
    if A_lo > A_hi:
        return np.zeros(a.shape[0], dtype=np.float64)
    sum_A = a[:, A_lo:A_hi + 1].sum(axis=1)
    return sum_A ** 2 / (4.0 * n_half * ell)


# =====================================================================
# Fix 1: Factored Rectangle Bound
# =====================================================================

def fix1_bound(prefix_a, d, n_half, ell, s_lo):
    """Fix 1 bound optimized over all interval pairs A, B.

    For each (a1, a2) defining A=[a1,a2], computes the widest valid
    B=[b1,b2] such that all pairs in AxB satisfy s_lo <= i+j <= s_hi.

    Parameters
    ----------
    prefix_a : (N, d+1) cumulative sum of a-coords, prefix_a[:,0]=0.
    """
    s_hi = s_lo + ell - 2
    inv_norm = 1.0 / (4.0 * n_half * ell)
    N = prefix_a.shape[0]
    best = np.zeros(N, dtype=np.float64)

    for a1 in range(d):
        for a2 in range(a1, d):
            # Widest valid B: b1 >= s_lo - a1, b2 <= s_hi - a2
            b1 = max(s_lo - a1, 0)
            b2 = min(s_hi - a2, d - 1)
            if b1 > b2:
                continue
            sum_A = prefix_a[:, a2 + 1] - prefix_a[:, a1]
            sum_B = prefix_a[:, b2 + 1] - prefix_a[:, b1]
            val = sum_A * sum_B * inv_norm
            np.maximum(best, val, out=best)

    return best


# =====================================================================
# Fix 3: Diagonal + Cauchy-Schwarz
# =====================================================================

def fix3_bound(a, d, n_half, ell, s_lo):
    """Fix 3 bound: (sum_{i in D} a_i)^2 / (4*n*ell*|D|).

    Same index set D as Fix 2's A, but divides by |D|.
    Always <= Fix 2.
    """
    A_lo, A_hi = fix2_interval(s_lo, ell, d)
    if A_lo > A_hi:
        return np.zeros(a.shape[0], dtype=np.float64)
    D_size = A_hi - A_lo + 1
    sum_D = a[:, A_lo:A_hi + 1].sum(axis=1)
    return sum_D ** 2 / (4.0 * n_half * ell * D_size)


# =====================================================================
# Fix 4: Partial Convolution (best single anti-diagonal per window)
# =====================================================================

def fix4_bound(conv, d, n_half, ell, s_lo):
    """Fix 4: max single conv[m] in window / (4*n*ell).

    Uses pre-computed autoconvolution.
    """
    s_hi = s_lo + ell - 2
    best_conv = conv[:, s_lo:s_hi + 1].max(axis=1)
    return best_conv / (4.0 * n_half * ell)


# =====================================================================
# Dynamic threshold computation
# =====================================================================

def dynamic_threshold(prefix_c, d, m, c_target, ell, s_lo):
    """Per-composition dynamic threshold for window (ell, s_lo).

    thresh = c_target + (1 + 2*W_int) / m^2 + fp_margin
    where W_int = sum of c[i] for contributing bins i.
    """
    inv_m_sq = 1.0 / (m * m)
    fp_margin = 1e-9
    d_minus_1 = d - 1
    lo_bin = max(0, s_lo - d_minus_1)
    hi_bin = min(d_minus_1, s_lo + ell - 2)
    W_int = prefix_c[:, hi_bin + 1] - prefix_c[:, lo_bin]
    return c_target + (1.0 + 2.0 * W_int) * inv_m_sq + fp_margin


# =====================================================================
# Main benchmark
# =====================================================================

def run_benchmark(n_half, m, c_target=1.28):
    d = 2 * n_half
    S = m
    scale = 4.0 * n_half / m
    corr = correction(m)
    thresh = c_target + corr
    fp_margin = 1e-9
    inv_m_sq = 1.0 / (m * m)
    conv_len = 2 * d - 1
    n_total = count_compositions(d, S)
    windows = enumerate_windows(d)
    n_windows = len(windows)

    print(f"\n{'='*70}")
    print(f"BENCHMARK: n_half={n_half}, m={m}, d={d}, c_target={c_target}")
    print(f"{'='*70}")
    print(f"  Total compositions: {n_total:,}")
    print(f"  Correction: {corr:.6f}")
    print(f"  Conservative threshold: {thresh:.6f}")
    print(f"  Number of windows: {n_windows}")

    # --- Generate canonical compositions ---
    t0 = time.time()
    all_comps = collect_all_canonical(d, S)
    N = len(all_comps)
    print(f"  Canonical compositions: {N:,} (generated in {time.time()-t0:.2f}s)")

    # --- Existing pre-checks ---
    asym = check_asymmetry(all_comps, n_half, m, c_target)
    n_asym = int(asym.sum())

    wpre = check_existing_window_prechecks(
        all_comps, d, n_half, m, thresh + fp_margin)
    n_wpre = int(wpre.sum())
    n_wpre_only = int((wpre & ~asym).sum())

    existing = asym | wpre
    n_existing = int(existing.sum())
    survives = ~existing
    n_surv = int(survives.sum())

    print(f"\n  EXISTING PRE-CHECK RESULTS:")
    print(f"    Asymmetry pruned:     {n_asym:>10,} ({100*n_asym/N:>5.1f}%)")
    print(f"    Window pre-check:     {n_wpre:>10,} ({100*n_wpre/N:>5.1f}%)")
    print(f"    Window-only (new):    {n_wpre_only:>10,} ({100*n_wpre_only/N:>5.1f}%)")
    print(f"    Total existing:       {n_existing:>10,} ({100*n_existing/N:>5.1f}%)")
    print(f"    Survive to autoconv:  {n_surv:>10,} ({100*n_surv/N:>5.1f}%)")

    if n_surv == 0:
        print("\n  All compositions pruned by existing checks!")
        _print_cost_analysis(d, n_half, n_windows)
        return

    # --- Compute autoconvolution for survivors ---
    survivors = all_comps[survives]
    a_surv = survivors.astype(np.float64) * scale

    t0 = time.time()
    conv = batch_autoconv(a_surv, d)
    t_ac = time.time() - t0
    print(f"\n  Autoconvolution: {n_surv:,} compositions in {t_ac:.2f}s")

    # True window test values and max
    tvs = true_window_tvs(conv, d, n_half)
    true_max = np.stack(list(tvs.values()), axis=1).max(axis=1)

    # Pruned by autoconvolution (conservative threshold)
    ac_pruned_cons = true_max > thresh + fp_margin
    n_ac_cons = int(ac_pruned_cons.sum())

    # Pruned by dynamic threshold
    prefix_c = np.zeros((n_surv, d + 1), dtype=np.int64)
    prefix_c[:, 1:] = np.cumsum(survivors, axis=1)

    ac_pruned_dyn = np.zeros(n_surv, dtype=bool)
    for ell, s_lo in windows:
        tv = tvs[(ell, s_lo)]
        dyn_t = dynamic_threshold(prefix_c, d, m, c_target, ell, s_lo)
        ac_pruned_dyn |= (tv > dyn_t)
    n_ac_dyn = int(ac_pruned_dyn.sum())

    print(f"    Pruned by autoconv (conservative): {n_ac_cons:>10,} "
          f"({100*n_ac_cons/n_surv:>5.1f}%)")
    print(f"    Pruned by autoconv (dynamic):      {n_ac_dyn:>10,} "
          f"({100*n_ac_dyn/n_surv:>5.1f}%)")
    print(f"    Final survivors (conservative):     {n_surv-n_ac_cons:>10,}")
    print(f"    Final survivors (dynamic):          {n_surv-n_ac_dyn:>10,}")

    # --- Compute all four bounds for survivors ---
    prefix_a = np.zeros((n_surv, d + 1), dtype=np.float64)
    prefix_a[:, 1:] = np.cumsum(a_surv, axis=1)

    print(f"\n  Computing bounds for {n_surv:,} compositions...")

    # Fix 2 (conservative + dynamic)
    t0 = time.time()
    f2_pruned_cons = np.zeros(n_surv, dtype=bool)
    f2_pruned_dyn = np.zeros(n_surv, dtype=bool)
    for ell, s_lo in windows:
        b = fix2_bound(a_surv, d, n_half, ell, s_lo)
        f2_pruned_cons |= (b > thresh + fp_margin)
        dyn_t = dynamic_threshold(prefix_c, d, m, c_target, ell, s_lo)
        f2_pruned_dyn |= (b > dyn_t)
    n_f2_cons = int(f2_pruned_cons.sum())
    n_f2_dyn = int(f2_pruned_dyn.sum())
    t_f2 = time.time() - t0

    # Fix 1 (conservative)
    t0 = time.time()
    f1_pruned_cons = np.zeros(n_surv, dtype=bool)
    for ell, s_lo in windows:
        b = fix1_bound(prefix_a, d, n_half, ell, s_lo)
        f1_pruned_cons |= (b > thresh + fp_margin)
    n_f1_cons = int(f1_pruned_cons.sum())
    t_f1 = time.time() - t0

    # Fix 3 (conservative)
    t0 = time.time()
    f3_pruned_cons = np.zeros(n_surv, dtype=bool)
    for ell, s_lo in windows:
        b = fix3_bound(a_surv, d, n_half, ell, s_lo)
        f3_pruned_cons |= (b > thresh + fp_margin)
    n_f3_cons = int(f3_pruned_cons.sum())
    t_f3 = time.time() - t0

    # Fix 4 (conservative)
    t0 = time.time()
    f4_pruned_cons = np.zeros(n_surv, dtype=bool)
    for ell, s_lo in windows:
        b = fix4_bound(conv, d, n_half, ell, s_lo)
        f4_pruned_cons |= (b > thresh + fp_margin)
    n_f4_cons = int(f4_pruned_cons.sum())
    t_f4 = time.time() - t0

    # --- Report pruning ---
    print(f"\n  NEW BOUND PRUNING (among {n_surv:,} pre-check survivors):")
    print(f"  {'Bound':<30} {'Pruned':>8} {'%surv':>7} {'Time':>7} {'Sound':>6}")
    print(f"  {'-'*62}")

    for name, mask, elapsed in [
        ('Fix 2 (conservative)', f2_pruned_cons, t_f2),
        ('Fix 2 (dynamic)', f2_pruned_dyn, t_f2),
        ('Fix 1 (conservative)', f1_pruned_cons, t_f1),
        ('Fix 3 (conservative)', f3_pruned_cons, t_f3),
        ('Fix 4 (conservative)', f4_pruned_cons, t_f4),
    ]:
        n_p = int(mask.sum())
        pct = 100 * n_p / max(1, n_surv)
        # Soundness: pruned by bound but NOT by autoconv = would be a bug
        not_by_ac = int((mask & ~ac_pruned_cons).sum())
        sound = "OK" if not_by_ac == 0 else f"!{not_by_ac}"
        print(f"  {name:<30} {n_p:>8,} {pct:>6.1f}% {elapsed:>6.2f}s {sound:>6}")

    print(f"\n  Reference: autoconv (conservative) prunes "
          f"{n_ac_cons:,}/{n_surv:,} ({100*n_ac_cons/max(1,n_surv):.1f}%)")
    print(f"  Reference: autoconv (dynamic) prunes "
          f"{n_ac_dyn:,}/{n_surv:,} ({100*n_ac_dyn/max(1,n_surv):.1f}%)")

    # --- Soundness verification ---
    print(f"\n  SOUNDNESS VERIFICATION:")
    violations = {'fix1': 0, 'fix2': 0, 'fix3': 0, 'fix4': 0}
    n_f2_gt_f1 = 0
    n_f3_gt_f2 = 0
    total_checks = 0
    eps = 1e-10

    for ell, s_lo in windows:
        tv = tvs[(ell, s_lo)]
        f1 = fix1_bound(prefix_a, d, n_half, ell, s_lo)
        f2 = fix2_bound(a_surv, d, n_half, ell, s_lo)
        f3 = fix3_bound(a_surv, d, n_half, ell, s_lo)
        f4 = fix4_bound(conv, d, n_half, ell, s_lo)

        violations['fix1'] += int(np.sum(f1 > tv + eps))
        violations['fix2'] += int(np.sum(f2 > tv + eps))
        violations['fix3'] += int(np.sum(f3 > tv + eps))
        violations['fix4'] += int(np.sum(f4 > tv + eps))
        n_f2_gt_f1 += int(np.sum(f2 > f1 + eps))
        n_f3_gt_f2 += int(np.sum(f3 > f2 + eps))
        total_checks += len(tv)

    for name, v in violations.items():
        s = "PASS" if v == 0 else f"FAIL ({v} violations!)"
        print(f"    {name} <= true_val: {s}")
    print(f"    fix2 <= fix1: "
          f"{'PASS' if n_f2_gt_f1 == 0 else f'FAIL ({n_f2_gt_f1})'}")
    print(f"    fix3 <= fix2: "
          f"{'PASS' if n_f3_gt_f2 == 0 else f'FAIL ({n_f3_gt_f2})'}")
    print(f"    ({total_checks:,} total checks)")

    # --- Tightness analysis ---
    print(f"\n  TIGHTNESS (bound / true_value, excluding zero-value windows):")
    all_ratios = {name: [] for name in ['fix1', 'fix2', 'fix3', 'fix4']}

    for ell, s_lo in windows:
        tv = tvs[(ell, s_lo)]
        nonzero = tv > 1e-15
        if not np.any(nonzero):
            continue

        f1 = fix1_bound(prefix_a, d, n_half, ell, s_lo)
        f2 = fix2_bound(a_surv, d, n_half, ell, s_lo)
        f3 = fix3_bound(a_surv, d, n_half, ell, s_lo)
        f4 = fix4_bound(conv, d, n_half, ell, s_lo)

        all_ratios['fix1'].append((f1[nonzero] / tv[nonzero]))
        all_ratios['fix2'].append((f2[nonzero] / tv[nonzero]))
        all_ratios['fix3'].append((f3[nonzero] / tv[nonzero]))
        all_ratios['fix4'].append((f4[nonzero] / tv[nonzero]))

    for name, ratio_list in all_ratios.items():
        if not ratio_list:
            print(f"    {name}: no data")
            continue
        r = np.concatenate(ratio_list)
        print(f"    {name}: mean={np.mean(r):.4f}  median={np.median(r):.4f}  "
              f"p10={np.percentile(r, 10):.4f}  p90={np.percentile(r, 90):.4f}  "
              f"max={np.max(r):.4f}")

    # --- Window coverage analysis ---
    _print_window_coverage(d, n_half, windows)

    # --- Per-window Fix 2 breakdown ---
    print(f"\n  PER-WINDOW FIX 2 PRUNING (conservative, among survivors):")
    any_printed = False
    for ell, s_lo in windows:
        b = fix2_bound(a_surv, d, n_half, ell, s_lo)
        n_p = int((b > thresh + fp_margin).sum())
        if n_p > 0:
            pct = 100 * n_p / max(1, n_surv)
            print(f"    (ell={ell}, s_lo={s_lo:>2}): {n_p:>8,} ({pct:>5.1f}%)")
            any_printed = True
    if not any_printed:
        print(f"    (none)")

    # --- Per-window Fix 1 breakdown (only windows not covered by Fix 2) ---
    print(f"\n  PER-WINDOW FIX 1 PRUNING (conservative, among survivors):")
    any_printed = False
    for ell, s_lo in windows:
        b = fix1_bound(prefix_a, d, n_half, ell, s_lo)
        b2 = fix2_bound(a_surv, d, n_half, ell, s_lo)
        # Count compositions pruned by Fix 1 but NOT by Fix 2 at this window
        n_f1_only = int(((b > thresh + fp_margin) & ~(b2 > thresh + fp_margin)).sum())
        if n_f1_only > 0:
            pct = 100 * n_f1_only / max(1, n_surv)
            print(f"    (ell={ell}, s_lo={s_lo:>2}): {n_f1_only:>8,} "
                  f"({pct:>5.1f}%) [Fix1-only]")
            any_printed = True
    if not any_printed:
        print(f"    (none beyond Fix 2)")

    # --- Cost analysis ---
    _print_cost_analysis(d, n_half, n_windows)


def _print_window_coverage(d, n_half, windows):
    """Print which windows are covered by existing pre-checks."""
    half = d // 2
    print(f"\n  WINDOW COVERAGE (existing pre-checks, d={d}):")
    print(f"  {'Window':<16} {'Fix2 A':>12} {'Existing check':>24}")
    print(f"  {'-'*55}")
    for ell, s_lo in windows:
        A_lo, A_hi = fix2_interval(s_lo, ell, d)
        a_str = f"[{A_lo},{A_hi}]" if A_lo <= A_hi else "(empty)"

        covered = "-"
        # Block-sum ell=d at extreme positions
        # s_lo=0 → Fix2 A covers left half [0, d//2-1]
        # s_lo=d → Fix2 A covers right half [d//2, d-1]
        if ell == d and s_lo == 0:
            covered = "block-sum LEFT"
        elif ell == d and s_lo == d:
            covered = "block-sum RIGHT"
        # Max element ell=2 (diagonal positions)
        if ell == 2 and A_lo == A_hi and A_lo <= d - 1:
            covered = f"max-elem (bin {A_lo})"
        if ell == 2 and A_lo > A_hi:
            covered = "empty (odd s_lo)"
        # D=4 specific: ell=3 exact windows
        if d == 4 and ell == 3 and s_lo == 0:
            covered = "exact window (L)"
        if d == 4 and ell == 3 and s_lo == 5:
            covered = "exact window (R)"

        print(f"  (ell={ell}, s_lo={s_lo:<2})   {a_str:>12}   {covered:>24}")


def _print_cost_analysis(d, n_half, n_windows):
    """Print computational cost analysis for each bound type."""
    conv_len = 2 * d - 1

    # Full autoconvolution cost (the expensive operation we're trying to avoid)
    ac_mults = d * d                    # multiplications for conv
    ac_prefix = conv_len - 1            # additions for prefix sum
    ac_scan = n_windows * 3             # subtract + divide + compare per window
    ac_total = ac_mults + ac_prefix + ac_scan

    # Fix 2: prefix sum + O(1) per window
    f2_prefix = d
    f2_per_win = 3                      # subtract + square + divide
    f2_total = f2_prefix + n_windows * f2_per_win

    # Fix 1: prefix sum + O(d^2) per window
    n_intervals = d * (d + 1) // 2      # number of (a1, a2) pairs
    f1_per_win = n_intervals * 4        # subtract + subtract + multiply + compare
    f1_total = d + n_windows * f1_per_win

    # Fix 3: same as Fix 2
    f3_total = f2_total

    # Fix 4: O(d) per anti-diagonal, 1 per window
    f4_per_win = d                      # d multiplications per conv value
    f4_total = n_windows * f4_per_win

    print(f"\n  COST ANALYSIS (d={d}, {n_windows} windows):")
    print(f"    Full autoconvolution: ~{ac_total} ops/composition")
    print(f"    Fix 2 (all windows):  ~{f2_total} ops "
          f"({f2_total/ac_total:.2f}x autoconv)")
    print(f"    Fix 1 (all windows):  ~{f1_total} ops "
          f"({f1_total/ac_total:.2f}x autoconv)")
    print(f"    Fix 3 (all windows):  ~{f3_total} ops "
          f"({f3_total/ac_total:.2f}x autoconv)")
    print(f"    Fix 4 (all windows):  ~{f4_total} ops "
          f"({f4_total/ac_total:.2f}x autoconv)")

    print(f"\n    Break-even prune rates:")
    for name, cost in [('Fix 2', f2_total), ('Fix 1', f1_total),
                       ('Fix 3', f3_total), ('Fix 4', f4_total)]:
        if ac_total > cost:
            be = cost / ac_total
            print(f"      {name}: >{be:.1%} of compositions must be pruned")
        else:
            print(f"      {name}: MORE expensive than full autoconv "
                  f"({cost/ac_total:.1f}x)")


if __name__ == '__main__':
    for n_half, m in [(2, 20), (3, 20), (3, 50)]:
        run_benchmark(n_half, m, c_target=1.28)
