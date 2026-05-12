"""Verify the NEW Lean axiom with pointwise (3+2W)/m^2 correction.

The new axiom (FinalResult.lean) requires:
  exists ell s_lo, test_value > 1.33 + (3 + 2*W_int) / m^2

The CPU cascade prunes when:
  ws > floor((c_target*m^2 + 3 + 2*W_int + eps) * ell/(4n) * (1-4*DBL_EPS))

At the killing window, test_value > c_target + (3+2*W_int)/m^2 (exactly).
So the CPU cascade output DIRECTLY validates the new axiom.

This script verifies this for all levels of the cascade.
"""

import sys, os, time
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _verify_new_axiom(batch_int, n_half, m, c_target):
    """For each composition, check CPU prunes it AND new axiom holds at killing window.

    New axiom threshold: test_value > c_target + (3 + 2*W_int) / m^2
    CPU threshold: ws > floor((c*m^2 + 3 + 2*W + eps)*ell/(4n)*(1-4eps))

    Returns:
      cpu_pruned[b]: 1 if CPU prunes
      new_axiom_ok[b]: 1 if test_value > c + (3+2W)/m^2 at killing window
    """
    B = batch_int.shape[0]
    d = batch_int.shape[1]
    conv_len = 2 * d - 1

    m_d = np.float64(m)
    m_sq = m_d * m_d
    n_d = np.float64(n_half)
    inv_4n = 1.0 / (4.0 * n_d)
    DBL_EPS = 2.220446049250313e-16
    one_minus_4eps = 1.0 - 4.0 * DBL_EPS
    eps_margin = 1e-9 * m_sq
    d_minus_1 = d - 1

    max_ell = 2 * d
    cs_corr_base = c_target * m_sq + 3.0 + eps_margin
    ct_base_ell_arr = np.empty(max_ell + 1, dtype=np.float64)
    w_scale_arr = np.empty(max_ell + 1, dtype=np.float64)
    for ell in range(2, max_ell + 1):
        ell_f = np.float64(ell)
        ct_base_ell_arr[ell] = cs_corr_base * ell_f * inv_4n
        w_scale_arr[ell] = 2.0 * ell_f * inv_4n

    cpu_pruned = np.zeros(B, dtype=np.int32)
    new_axiom_ok = np.zeros(B, dtype=np.int32)

    for b in prange(B):
        conv = np.zeros(conv_len, dtype=np.int32)
        for i in range(d):
            ci = np.int32(batch_int[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int32(batch_int[b, j])
                    if cj != 0:
                        conv[i + j] += np.int32(2) * ci * cj

        prefix_c = np.zeros(d + 1, dtype=np.int64)
        for i in range(d):
            prefix_c[i + 1] = prefix_c[i] + np.int64(batch_int[b, i])

        for ell in range(2, max_ell + 1):
            n_cv = ell - 1
            ct_base_ell = ct_base_ell_arr[ell]
            w_scale = w_scale_arr[ell]
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += np.int64(conv[k])
            found = False
            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += np.int64(conv[s_lo + n_cv - 1]) - np.int64(conv[s_lo - 1])
                lo_bin = s_lo - d_minus_1
                if lo_bin < 0:
                    lo_bin = 0
                hi_bin = s_lo + ell - 2
                if hi_bin > d_minus_1:
                    hi_bin = d_minus_1
                W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]
                dyn_x = ct_base_ell + w_scale * np.float64(W_int)
                dyn_it = np.int64(dyn_x * one_minus_4eps)
                if ws > dyn_it:
                    cpu_pruned[b] = 1
                    # Check new axiom: test_value > c + (3+2W)/m^2
                    # test_value = ws * 4n / (m^2 * ell)
                    tv = np.float64(ws) * 4.0 * n_d / (m_sq * np.float64(ell))
                    new_thresh = c_target + (3.0 + 2.0 * np.float64(W_int)) / m_sq
                    if tv > new_thresh:
                        new_axiom_ok[b] = 1
                    found = True
                    break
            if found:
                break

    return cpu_pruned, new_axiom_ok


def main():
    m, n_half, c_target = 35, 2, 1.33
    print(f"Verifying NEW axiom: TV > {c_target} + (3 + 2*W_int) / {m}^2")
    print(f"Parameters: m={m}, n_half={n_half}, c_target={c_target}")
    print(f"New correction: (3 + 2*W_int) / {m*m}")
    print()

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    cs_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(cs_dir, 'cloninger-steinerberger'))
    from compositions import generate_canonical_compositions_batched
    import itertools

    # L0
    d0 = 2 * n_half
    batches = []
    for batch in generate_canonical_compositions_batched(d0, m):
        batches.append(batch)
    comps = np.vstack(batches)
    print(f"L0: {len(comps):,} compositions, d={d0}")
    cpu_p, axiom_ok = _verify_new_axiom(comps.astype(np.int32), n_half, m, c_target)
    n_pruned = cpu_p.sum()
    n_ok = ((cpu_p == 1) & (axiom_ok == 1)).sum()
    n_fail = ((cpu_p == 1) & (axiom_ok == 0)).sum()
    print(f"  CPU pruned: {n_pruned:,}, Axiom OK: {n_ok:,}, FAIL: {n_fail:,}")
    if n_fail > 0:
        print(f"  >>> AXIOM FAILS for {n_fail} compositions! <<<")
    else:
        print(f"  >>> ALL OK <<<")

    # Higher levels
    for level in range(1, 4):
        prev_ckpt = os.path.join(data_dir, f'checkpoint_L{level-1}_survivors.npy')
        if not os.path.exists(prev_ckpt):
            print(f"\n  No checkpoint at L{level-1}, skipping")
            continue

        parents = np.load(prev_ckpt)
        d_parent = parents.shape[1]
        d_child = 2 * d_parent
        n_half_child = d_parent

        max_parents = min(50, len(parents))
        print(f"\nL{level}: children from {max_parents}/{len(parents):,} parents, d={d_child}")

        all_children = []
        for pi in range(max_parents):
            parent = parents[pi]
            ranges_list = [list(range(int(parent[k]) + 1)) for k in range(d_parent)]
            total = 1
            for r in ranges_list:
                total *= len(r)
            if total > 200_000:
                rng = np.random.default_rng(42 + pi)
                sz = min(50_000, total)
                ch = np.empty((sz, d_child), dtype=np.int32)
                for si in range(sz):
                    for k in range(d_parent):
                        c = rng.integers(0, int(parent[k]) + 1)
                        ch[si, 2*k] = c
                        ch[si, 2*k+1] = parent[k] - c
                all_children.append(ch)
            else:
                ch = np.empty((total, d_child), dtype=np.int32)
                for ci, combo in enumerate(itertools.product(*ranges_list)):
                    for k in range(d_parent):
                        ch[ci, 2*k] = combo[k]
                        ch[ci, 2*k+1] = parent[k] - combo[k]
                all_children.append(ch)
            if sum(len(c) for c in all_children) > 500_000:
                break

        if not all_children:
            continue
        all_ch = np.vstack(all_children)[:500_000]
        print(f"  Testing {len(all_ch):,} children")
        t0 = time.time()
        cpu_p, axiom_ok = _verify_new_axiom(all_ch.astype(np.int32), n_half_child, m, c_target)
        dt = time.time() - t0
        n_pruned = cpu_p.sum()
        n_ok = ((cpu_p == 1) & (axiom_ok == 1)).sum()
        n_fail = ((cpu_p == 1) & (axiom_ok == 0)).sum()
        print(f"  CPU pruned: {n_pruned:,}, Axiom OK: {n_ok:,}, FAIL: {n_fail:,} ({dt:.1f}s)")
        if n_fail > 0:
            print(f"  >>> AXIOM FAILS for {n_fail} compositions! <<<")
        else:
            print(f"  >>> ALL OK <<<")


if __name__ == '__main__':
    main()
