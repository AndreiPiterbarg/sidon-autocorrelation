"""Detailed GPU kernel impact analysis for pruning proposals.

Measures:
  1. Per-parent arc consistency tightening statistics
  2. Projected children reduction at each level
  3. GPU shared memory budget with proposals
  4. Implementation cost in GPU warp operations
"""
import sys
import os
import math
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloninger-steinerberger', 'cpu'))

from pruning import correction
from run_cascade import _compute_bin_ranges


def make_threshold_table(m, c_target, d_child, n_half_child):
    inv_4n = 1.0 / (4.0 * n_half_child)
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    eps_margin = 1e-9 * m * m
    c_target_m2 = c_target * m * m
    max_ell = 2 * d_child
    m_plus_1 = m + 1
    table = np.empty((max_ell - 1) * m_plus_1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        ell_idx = ell - 2
        cs_base = (c_target_m2 + 3.0 + eps_margin) * ell * inv_4n
        w_scale = 2.0 * ell * inv_4n
        for w in range(m_plus_1):
            dyn_x = cs_base + w_scale * w
            table[ell_idx * m_plus_1 + w] = int(dyn_x * one_minus_4eps)
    return table


def tighten_ranges_fast(parent_int, m, c_target, n_half_child, lo_arr, hi_arr):
    """Fast arc consistency (edges only, 1 pass)."""
    d_parent = len(parent_int)
    d_child = 2 * d_parent
    conv_len = 2 * d_child - 1
    threshold_table = make_threshold_table(m, c_target, d_child, n_half_child)
    m_plus_1 = m + 1

    lo = lo_arr.copy()
    hi = hi_arr.copy()
    min_val = np.zeros(d_child, dtype=np.int64)
    max_val = np.zeros(d_child, dtype=np.int64)
    for p in range(d_parent):
        min_val[2*p] = lo[p]; min_val[2*p+1] = parent_int[p] - hi[p]
        max_val[2*p] = hi[p]; max_val[2*p+1] = parent_int[p] - lo[p]

    conv_min = np.zeros(conv_len, dtype=np.int64)
    for i in range(d_child):
        mi = int(min_val[i])
        if mi > 0:
            conv_min[2*i] += mi * mi
            for j in range(i+1, d_child):
                mj = int(min_val[j])
                if mj > 0:
                    conv_min[i+j] += 2 * mi * mj

    for p in range(d_parent):
        B_p = int(parent_int[p])
        k1, k2 = 2*p, 2*p+1
        old_ml, old_mh = int(min_val[k1]), int(min_val[k2])

        co = conv_min.copy()
        co[2*k1] -= old_ml * old_ml
        co[2*k2] -= old_mh * old_mh
        co[k1+k2] -= 2 * old_ml * old_mh
        for j in range(d_child):
            if j == k1 or j == k2: continue
            mj = int(min_val[j])
            if mj > 0:
                if old_ml > 0: co[k1+j] -= 2 * old_ml * mj
                if old_mh > 0: co[k2+j] -= 2 * old_mh * mj

        def check_v(v):
            v1, v2 = v, B_p - v
            cp = np.zeros(conv_len, dtype=np.int64)
            cp[2*k1] = v1*v1; cp[2*k2] = v2*v2; cp[k1+k2] = 2*v1*v2
            for j in range(d_child):
                if j == k1 or j == k2: continue
                mj = int(min_val[j])
                if mj > 0:
                    if v1 > 0: cp[k1+j] += 2*v1*mj
                    if v2 > 0: cp[k2+j] += 2*v2*mj
            for ell in range(2, 2*d_child+1):
                n_cv = ell - 1
                ell_idx = ell - 2
                for s_lo in range(conv_len - n_cv + 1):
                    ws = sum(int(co[k]) + int(cp[k])
                             for k in range(s_lo, s_lo + n_cv))
                    lo_bin = max(0, s_lo - (d_child-1))
                    hi_bin = min(d_child-1, s_lo + ell - 2)
                    W_max = min(sum(int(max_val[i])
                                    for i in range(lo_bin, hi_bin+1)), m)
                    if ws > threshold_table[ell_idx * m_plus_1 + W_max]:
                        return True
            return False

        for v in range(lo[p], hi[p]+1):
            if check_v(v): lo[p] = v + 1
            else: break
        for v in range(hi[p], lo[p]-1, -1):
            if check_v(v): hi[p] = v - 1
            else: break

    total = 1
    for i in range(d_parent):
        r = hi[i] - lo[i] + 1
        if r <= 0: return lo, hi, 0
        total *= r
    return lo, hi, total


def main():
    print("=" * 70)
    print("GPU KERNEL IMPACT ANALYSIS FOR PRUNING PROPOSALS")
    print("=" * 70)

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    m, c_target = 20, 1.40

    # ================================================================
    # Per-parent arc consistency analysis at each level
    # ================================================================
    for level, ckpt in [(0, 'checkpoint_L0_survivors.npy'),
                         (1, 'checkpoint_L1_survivors.npy'),
                         (2, 'checkpoint_L2_survivors.npy')]:
        path = os.path.join(data_dir, ckpt)
        if not os.path.exists(path):
            continue

        parents = np.load(path)
        d_parent = parents.shape[1]
        d_child = 2 * d_parent
        n_half_child = d_parent

        print(f"\n{'='*70}")
        print(f"L{level}->L{level+1}: d_parent={d_parent}, d_child={d_child}, "
              f"{len(parents)} parents")
        print(f"{'='*70}")

        max_p = min(100, len(parents))
        indices = np.linspace(0, len(parents)-1, max_p, dtype=int)

        orig_products = []
        ac_products = []
        per_pos_stats = []  # (positions_tightened, total_values_removed)
        t0 = time.time()

        for idx in indices:
            parent = parents[idx]
            result = _compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
            if result is None:
                continue
            lo_orig, hi_orig, tc_orig = result
            if tc_orig == 0:
                continue

            lo_t, hi_t, tc_t = tighten_ranges_fast(
                parent, m, c_target, n_half_child,
                lo_orig.copy(), hi_orig.copy())

            orig_products.append(tc_orig)
            ac_products.append(tc_t)

            n_pos_tight = 0
            n_vals_removed = 0
            for p in range(d_parent):
                r_orig = hi_orig[p] - lo_orig[p] + 1
                r_new = hi_t[p] - lo_t[p] + 1
                if r_new < r_orig:
                    n_pos_tight += 1
                    n_vals_removed += r_orig - r_new
            per_pos_stats.append((n_pos_tight, n_vals_removed))

        elapsed = time.time() - t0

        if orig_products:
            total_orig = sum(orig_products)
            total_ac = sum(ac_products)
            avg_orig = np.mean(orig_products)
            avg_ac = np.mean(ac_products)
            n_with_tightening = sum(1 for n, _ in per_pos_stats if n > 0)
            avg_pos_tight = np.mean([n for n, _ in per_pos_stats])
            avg_vals_removed = np.mean([v for _, v in per_pos_stats])

            print(f"  Parents sampled: {len(orig_products)}")
            print(f"  Average product (orig):  {avg_orig:.1f}")
            print(f"  Average product (AC):    {avg_ac:.1f}")
            print(f"  Total orig:              {total_orig:,}")
            print(f"  Total after AC:          {total_ac:,}")
            print(f"  AC reduction factor:     {total_orig/max(total_ac,1):.2f}x")
            print(f"  Parents with tightening: {n_with_tightening}/{len(orig_products)}")
            print(f"  Avg positions tightened:  {avg_pos_tight:.2f}")
            print(f"  Avg values removed:       {avg_vals_removed:.2f}")
            print(f"  Time: {elapsed:.1f}s ({elapsed/len(orig_products)*1000:.1f}ms/parent)")

            # Distribution
            ratios = [o/max(a,1) for o, a in zip(orig_products, ac_products)]
            print(f"\n  Per-parent AC reduction distribution:")
            print(f"    1.0x (no change): {sum(1 for r in ratios if r < 1.01)}/{len(ratios)}")
            for thresh in [1.5, 2.0, 3.0, 5.0, 10.0]:
                count = sum(1 for r in ratios if r >= thresh)
                if count > 0:
                    print(f"    >= {thresh:.1f}x: {count}/{len(ratios)}")

    # ================================================================
    # GPU shared memory budget analysis
    # ================================================================
    print(f"\n{'='*70}")
    print("GPU SHARED MEMORY BUDGET (with proposals)")
    print(f"{'='*70}")

    for d_child, label in [(8, "L0->L1"), (16, "L1->L2"),
                            (32, "L2->L3"), (64, "L3->L4")]:
        conv_len = 2 * d_child - 1
        ell_count = 2 * d_child - 1
        m_val = 20

        # Current kernel
        child_bytes = d_child * 4
        raw_conv_bytes = conv_len * 4
        prefix_c_bytes = (d_child + 1) * 8
        threshold_bytes = ell_count * (m_val + 1) * 8
        cursor_bytes = (d_child // 2) * 4
        gc_state_bytes = (d_child // 2) * 4 * 3  # gc_a, gc_dir, gc_focus
        partial_conv_bytes = conv_len * 4  # already allocated
        current_total = (child_bytes + raw_conv_bytes + prefix_c_bytes +
                         threshold_bytes + cursor_bytes + gc_state_bytes +
                         partial_conv_bytes)

        # Additional for proposals
        min_contrib_bytes = conv_len * 8  # int64
        min_contrib_prefix_bytes = conv_len * 8  # int64

        proposed_total = current_total + min_contrib_bytes + min_contrib_prefix_bytes

        h100_smem = 228 * 1024  # 228KB per SM
        blocks_current = h100_smem // current_total
        blocks_proposed = h100_smem // proposed_total

        print(f"\n  {label} (d_child={d_child}):")
        print(f"    Current kernel:   {current_total:>6} bytes ({blocks_current} blocks/SM)")
        print(f"    + min_contrib:    +{min_contrib_bytes:>5} bytes (Proposal 2)")
        print(f"    + mc_prefix:      +{min_contrib_prefix_bytes:>5} bytes (Proposal 4)")
        print(f"    Proposed total:   {proposed_total:>6} bytes ({blocks_proposed} blocks/SM)")

    # ================================================================
    # Warp operation cost of subtree check
    # ================================================================
    print(f"\n{'='*70}")
    print("WARP OPERATION COST PER SUBTREE CHECK")
    print(f"{'='*70}")

    for d_child in [32, 64]:
        d_parent = d_child // 2
        print(f"\n  d_child={d_child}:")

        # For a typical subtree check at level j=n_active/2
        n_active = d_parent // 2  # rough estimate
        j = n_active // 2
        fixed_len = j * 2  # rough

        # Cost components (in warp-parallel operations)
        partial_conv_ops = fixed_len * fixed_len // 32  # O(fixed_len^2/32)
        n_unfixed = d_parent - j
        min_contrib_ops = n_unfixed * fixed_len // 32  # cross terms
        min_contrib_self = n_unfixed  # self terms (1 per unfixed pos)
        unfixed_cross = n_unfixed * n_unfixed // 2 // 32

        # Window scan
        conv_len = 2 * d_child - 1
        ell_count = 2 * d_child - 1
        # Average windows checked before finding kill
        avg_windows = 20  # typically kill on first few ells
        scan_ops = avg_windows * (d_child // 32)  # prefix sum + query

        total_ops = (partial_conv_ops + min_contrib_ops + min_contrib_self +
                     unfixed_cross + scan_ops)

        print(f"    Typical j={j} (fixed_len={fixed_len}, n_unfixed={n_unfixed}):")
        print(f"      partial_conv:    ~{partial_conv_ops:>4} warp ops")
        print(f"      min_contrib:     ~{min_contrib_ops + min_contrib_self + unfixed_cross:>4} warp ops")
        print(f"      window scan:     ~{scan_ops:>4} warp ops")
        print(f"      TOTAL:           ~{total_ops:>4} warp ops per check")

        # Vs children saved
        subtree_size = 3 ** j  # rough average range of 3
        print(f"      Subtree size:    ~{subtree_size:>6} children")
        print(f"      Cost/benefit:    ~{total_ops/subtree_size:.4f} ops per child saved")

    # ================================================================
    # Implementation priority recommendation
    # ================================================================
    print(f"\n{'='*70}")
    print("IMPLEMENTATION PRIORITY FOR GPU KERNEL")
    print(f"{'='*70}")
    print("""
PRIORITY 1: Arc Consistency (Proposal 3)
  - Runs on CPU before GPU kernel launch
  - Zero GPU code changes needed
  - Reduces per-parent Cartesian product by tightening cursor ranges
  - Cost: O(d_parent * max_range * d_child * ell_count) per parent
  - At L2->L3 (d=16->32): ~0.5M ops per parent, <1ms
  - At L3->L4 (d=32->64): ~8M ops per parent, ~10ms

PRIORITY 2: Multi-Level Subtree Pruning (Proposal 1)
  - Change J_MIN=7 to check at ALL levels j >= 2
  - GPU: same code path, just run it more often
  - Each check costs O(fixed_len^2/32) warp ops
  - Benefit: skips product(range[0..j-1]) children per prune

PRIORITY 3: Min Unfixed Contribution (Proposal 2)
  - Add min_contrib array to shared memory (+1-2KB)
  - Compute guaranteed lower bounds from unfixed bins
  - Makes subtree checks much tighter (especially for high j)
  - Requires cross-term computation in shared memory

PRIORITY 4: Partial-Overlap Windows (Proposal 4)
  - Extend window scan beyond fixed prefix
  - Uses min_contrib for out-of-range windows
  - Most complex to implement on GPU (variable-size prefix sums)
  - Primarily benefits high-j checks (large subtrees)
""")


if __name__ == '__main__':
    main()
