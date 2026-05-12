"""Coarse-grid N+O combined: spectral δ² (variant N) AND sparsity-aware
   one-sided LP (variant O).  Sound by construction (both are sound; using
   the smaller correction in each pair is sound).

For each (d, S, c_target):
  - baseline triangle: cell_var + quad_corr_BL
  - variant N:         cell_var + quad_corr_N      (quad_corr_N <= quad_corr_BL)
  - variant O:         cell_var_O + quad_corr_BL   (cell_var_O <= cell_var)
  - variant NO:        cell_var_O + quad_corr_N    (both tighter)

Reports survivor counts for each.  NO survivors <= O survivors AND <= N survivors,
since each tighter ingredient only helps.
"""
import os, sys, time, json
import numpy as np
import numba
from numba import njit, prange

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, _dir)

from compositions import generate_compositions_batched
from pruning import count_compositions

# Reuse helpers from the existing benches
from _coarse_O_bench import (_build_pair_prefix, _quad_corr, _one_sided_lp,
                              prune_coarse_baseline, prune_coarse_O)


def precompute_op_rest(d, max_ell):
    """For each (ell, s_lo), op_rest = ‖A_W − α·11ᵀ‖_op with α = N_W/d²."""
    conv_len = 2 * d - 1
    op_rest = np.zeros((max_ell + 1, conv_len), dtype=np.float64)
    n_pairs = np.zeros((max_ell + 1, conv_len), dtype=np.int64)
    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        if n_windows <= 0:
            continue
        for s_lo in range(n_windows):
            A = np.zeros((d, d), dtype=np.float64)
            for i in range(d):
                for j in range(d):
                    if s_lo <= i + j <= s_lo + ell - 2:
                        A[i, j] = 1.0
            n = int(A.sum())
            n_pairs[ell, s_lo] = n
            if n == 0:
                continue
            alpha = n / float(d * d)
            eigs = np.linalg.eigvalsh(A - alpha)
            op_rest[ell, s_lo] = float(np.abs(eigs).max())
    return op_rest, n_pairs


@njit(parallel=True, cache=True)
def prune_coarse_NO(batch_int, d, S, c_target, op_rest_d):
    """N+O combined pruner.

    quad_corr_N   = scale * min(cross_W, d²-N_W, op_rest*d) * h²
    cell_var_O    = LP with one-sided constraints for k_j=0
    """
    B = batch_int.shape[0]
    conv_len = 2 * d - 1
    survived = np.ones(B, dtype=numba.boolean)

    S_d = np.float64(S)
    S_sq = S_d * S_d
    d_d = np.float64(d)
    inv_2d = 1.0 / (2.0 * d_d)
    inv_4S2 = 1.0 / (4.0 * S_sq)
    eps = 1e-9
    max_ell = 2 * d
    h = 1.0 / (2.0 * S_d)

    thr_arr = np.empty(max_ell + 1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        thr_arr[ell] = np.int64(c_target * np.float64(ell) * S_sq * inv_2d - eps)

    prefix_nk, prefix_mk = _build_pair_prefix(d)

    for b in prange(B):
        conv = np.zeros(conv_len, dtype=np.int64)
        for i in range(d):
            ci = np.int64(batch_int[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int64(batch_int[b, j])
                    if cj != 0:
                        conv[i + j] += np.int64(2) * ci * cj

        pruned = False
        grad_arr = np.empty(d, dtype=np.float64)
        k_local = np.empty(d, dtype=np.int64)
        for j in range(d):
            k_local[j] = np.int64(batch_int[b, j])
        work_buf = np.empty(4 * d, dtype=np.int64)

        for ell in range(2, max_ell + 1):
            if pruned:
                break
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += conv[k]
            dyn_it = thr_arr[ell]
            ell_f = np.float64(ell)
            scale_g = 4.0 * d_d / ell_f

            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                if ws > dyn_it:
                    tv = np.float64(ws) * 2.0 * d_d / (S_sq * ell_f)
                    margin = tv - c_target

                    for i in range(d):
                        g = 0.0
                        for j in range(d):
                            kk = i + j
                            if s_lo <= kk <= s_lo + ell - 2:
                                g += np.float64(batch_int[b, j]) / S_d
                        grad_arr[i] = g * scale_g

                    # cell_var_O (one-sided LP)
                    cell_var_O = _one_sided_lp(grad_arr, k_local, d, h, work_buf)
                    # FP safety: clip to baseline cell_var
                    grad_sorted = np.empty(d, dtype=np.float64)
                    for i in range(d):
                        grad_sorted[i] = grad_arr[i]
                    for i in range(1, d):
                        key = grad_sorted[i]
                        jj = i - 1
                        while jj >= 0 and grad_sorted[jj] > key:
                            grad_sorted[jj + 1] = grad_sorted[jj]
                            jj -= 1
                        grad_sorted[jj + 1] = key
                    cell_var_BL = 0.0
                    for k in range(d // 2):
                        cell_var_BL += grad_sorted[d - 1 - k] - grad_sorted[k]
                    cell_var_BL *= h
                    if cell_var_O > cell_var_BL:
                        cell_var_O = cell_var_BL

                    # quad_corr_N: spectral pair_bound
                    hi = s_lo + ell - 1
                    N_W = prefix_nk[hi] - prefix_nk[s_lo]
                    M_W = prefix_mk[hi] - prefix_mk[s_lo]
                    cross_W = N_W - M_W
                    d_sq = np.int64(d) * np.int64(d)
                    compl_bound = d_sq - N_W
                    spec_bound = np.int64(np.ceil(op_rest_d[ell, s_lo]))
                    pair_bound = cross_W
                    if compl_bound < pair_bound:
                        pair_bound = compl_bound
                    if spec_bound < pair_bound:
                        pair_bound = spec_bound
                    if pair_bound <= 0:
                        qc = 0.0
                    else:
                        qc = (2.0 * d_d / ell_f
                              * np.float64(pair_bound) * inv_4S2)

                    if margin - cell_var_O - qc >= 0.0:
                        pruned = True
                        break

        if pruned:
            survived[b] = False

    return survived


@njit(parallel=True, cache=True)
def prune_coarse_N(batch_int, d, S, c_target, op_rest_d):
    """N alone: cell_var (baseline) + quad_corr_N (spectral)."""
    B = batch_int.shape[0]
    conv_len = 2 * d - 1
    survived = np.ones(B, dtype=numba.boolean)

    S_d = np.float64(S); S_sq = S_d * S_d
    d_d = np.float64(d); inv_2d = 1.0 / (2.0 * d_d)
    inv_4S2 = 1.0 / (4.0 * S_sq); eps = 1e-9
    max_ell = 2 * d; h = 1.0 / (2.0 * S_d)

    thr_arr = np.empty(max_ell + 1, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        thr_arr[ell] = np.int64(c_target * np.float64(ell) * S_sq * inv_2d - eps)

    prefix_nk, prefix_mk = _build_pair_prefix(d)

    for b in prange(B):
        conv = np.zeros(conv_len, dtype=np.int64)
        for i in range(d):
            ci = np.int64(batch_int[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int64(batch_int[b, j])
                    if cj != 0:
                        conv[i + j] += np.int64(2) * ci * cj

        pruned = False
        grad_arr = np.empty(d, dtype=np.float64)

        for ell in range(2, max_ell + 1):
            if pruned:
                break
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += conv[k]
            dyn_it = thr_arr[ell]
            ell_f = np.float64(ell)
            scale_g = 4.0 * d_d / ell_f

            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                if ws > dyn_it:
                    tv = np.float64(ws) * 2.0 * d_d / (S_sq * ell_f)
                    margin = tv - c_target

                    for i in range(d):
                        g = 0.0
                        for j in range(d):
                            kk = i + j
                            if s_lo <= kk <= s_lo + ell - 2:
                                g += np.float64(batch_int[b, j]) / S_d
                        grad_arr[i] = g * scale_g

                    for i in range(1, d):
                        key = grad_arr[i]
                        jj = i - 1
                        while jj >= 0 and grad_arr[jj] > key:
                            grad_arr[jj + 1] = grad_arr[jj]
                            jj -= 1
                        grad_arr[jj + 1] = key
                    cell_var = 0.0
                    for k in range(d // 2):
                        cell_var += grad_arr[d - 1 - k] - grad_arr[k]
                    cell_var *= h

                    hi = s_lo + ell - 1
                    N_W = prefix_nk[hi] - prefix_nk[s_lo]
                    M_W = prefix_mk[hi] - prefix_mk[s_lo]
                    cross_W = N_W - M_W
                    d_sq = np.int64(d) * np.int64(d)
                    compl_bound = d_sq - N_W
                    spec_bound = np.int64(np.ceil(op_rest_d[ell, s_lo]))
                    pair_bound = cross_W
                    if compl_bound < pair_bound:
                        pair_bound = compl_bound
                    if spec_bound < pair_bound:
                        pair_bound = spec_bound
                    if pair_bound <= 0:
                        qc = 0.0
                    else:
                        qc = (2.0 * d_d / ell_f
                              * np.float64(pair_bound) * inv_4S2)

                    if margin - cell_var - qc >= 0.0:
                        pruned = True
                        break

        if pruned:
            survived[b] = False

    return survived


def enum_compositions(d, S):
    if d == 1:
        yield (S,)
        return
    for v in range(S + 1):
        for rest in enum_compositions(d - 1, S - v):
            yield (v,) + rest


def run_one(d, S, c_target):
    print(f"\n=== d={d}, S={S}, c={c_target} ===", flush=True)
    op_rest, n_pairs_arr = precompute_op_rest(d, 2 * d)
    op_rest_d = op_rest * d

    batch = np.array(list(enum_compositions(d, S)), dtype=np.int32)
    print(f"  total comps: {len(batch):,}", flush=True)

    # Warm
    warm = batch[:1]
    _ = prune_coarse_baseline(warm, d, S, c_target)
    _ = prune_coarse_O(warm, d, S, c_target)
    _ = prune_coarse_N(warm, d, S, c_target, op_rest_d)
    _ = prune_coarse_NO(warm, d, S, c_target, op_rest_d)

    t0 = time.time()
    sBL = prune_coarse_baseline(batch, d, S, c_target)
    tBL = time.time() - t0

    t0 = time.time()
    sN = prune_coarse_N(batch, d, S, c_target, op_rest_d)
    tN = time.time() - t0

    t0 = time.time()
    sO = prune_coarse_O(batch, d, S, c_target)
    tO = time.time() - t0

    t0 = time.time()
    sNO = prune_coarse_NO(batch, d, S, c_target, op_rest_d)
    tNO = time.time() - t0

    nBL = int(sBL.sum())
    nN = int(sN.sum())
    nO = int(sO.sum())
    nNO = int(sNO.sum())

    extra_N = nBL - nN
    extra_O = nBL - nO
    extra_NO = nBL - nNO
    overlap = (extra_N + extra_O) - extra_NO  # cells closed by BOTH N and O
    n_only_N = extra_NO - extra_O  # cells closed by N alone in NO context
    n_only_O = extra_NO - extra_N

    pct = lambda x: 100.0 * x / max(1, nBL)
    print(f"  BL survivors: {nBL:,}  ({tBL:.2f}s)")
    print(f"   N survivors: {nN:,}   (-{extra_N:,} = {pct(extra_N):.2f}%, {tN:.2f}s)")
    print(f"   O survivors: {nO:,}   (-{extra_O:,} = {pct(extra_O):.2f}%, {tO:.2f}s)")
    print(f"  NO survivors: {nNO:,}  (-{extra_NO:,} = {pct(extra_NO):.2f}%, {tNO:.2f}s)")
    print(f"  Overlap (cells closed by BOTH N and O alone): {overlap:,}")
    print(f"  Strict gain N+O over max(N,O):   "
          f"{nNO - min(nN,nO):+,}  (i.e., NO closes {min(nN,nO) - nNO:,} more)")

    # Soundness sanity: NO ⊆ N AND NO ⊆ O (every cell pruned by NO is also pruned by N and O)
    NO_not_N = int(np.sum((~sNO) & sN))  # NO prunes, N doesn't — should be 0 if N tighter individually... actually NO uses tighter quad_corr_N AND tighter cell_var_O so it's strictly tighter than N alone (which uses cell_var). So NO_not_N could be positive.
    NO_not_O = int(np.sum((~sNO) & sO))  # NO prunes, O doesn't — same logic.
    N_not_BL = int(np.sum((~sN) & sBL))
    O_not_BL = int(np.sum((~sO) & sBL))
    print(f"  Sanity: N\\BL={N_not_BL:,}, O\\BL={O_not_BL:,}, "
          f"NO\\N={NO_not_N:,}, NO\\O={NO_not_O:,}")
    return {
        'd': d, 'S': S, 'c_target': c_target,
        'n_total': len(batch),
        'surv_BL': nBL, 'surv_N': nN, 'surv_O': nO, 'surv_NO': nNO,
        'extra_N': extra_N, 'extra_O': extra_O, 'extra_NO': extra_NO,
        'overlap_NO': overlap,
        'time_BL': tBL, 'time_N': tN, 'time_O': tO, 'time_NO': tNO,
    }


def main():
    configs = [
        (4, 20, 1.20),
        (4, 30, 1.25),
        (6, 15, 1.20),
        (6, 20, 1.25),
        (8, 12, 1.20),
        (8, 15, 1.20),
    ]
    results = []
    for d, S, c in configs:
        results.append(run_one(d, S, c))
    print("\n" + "=" * 70 + "\nSUMMARY:")
    for r in results:
        pct_NO = 100.0 * r['extra_NO'] / max(1, r['surv_BL'])
        pct_max_NO_strict = (r['extra_NO'] - max(r['extra_N'], r['extra_O'])) \
                            / max(1, r['surv_BL']) * 100.0
        print(f"  d={r['d']}, S={r['S']}, c={r['c_target']}: "
              f"NO -{r['extra_NO']:,} ({pct_NO:.2f}% of BL); "
              f"strict gain over max(N,O): {pct_max_NO_strict:+.2f}%")
    out = os.path.join(_dir, '_coarse_NO_bench_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=int)


if __name__ == '__main__':
    main()
