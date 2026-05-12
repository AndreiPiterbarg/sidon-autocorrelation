"""Parallelize per-window residual contributions across worker processes.

Each window W's contribution to the stationarity residual is independent:
    r_W[alpha] = t_test * adj_t(S_W)[alpha]  -  adj_qW(S_W)[alpha]

We compute these in parallel via multiprocessing. fmpq objects pickle;
fmpq_mat does not, so we pass S_W as a flat list of fmpq.

The main process sums all r_W + base contributions.
"""
from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

try:
    import flint  # type: ignore
    _HAS_FLINT = True
except ImportError:
    _HAS_FLINT = False


def _compute_window_contribution_worker(args):
    """Worker: compute `t_test * adj_t(S_W) - adj_qW(S_W)` for one window.

    Input (args):
        (w_idx, S_W_flat, nl, t_pick_list, ab_eiej_idx_w, M_W_support_nz,
         coeff_fmpq_num, coeff_fmpq_den, t_test_num, t_test_den, n_y)
    Output: (w_idx, list of (alpha, fmpq) for the contributions)
    """
    import flint  # re-import in worker
    (w_idx, S_flat_list, nl, t_pick_list, ab_eiej_w_flat,
     M_W_support_nz, coeff_num, coeff_den, t_test_num, t_test_den, n_y) = args

    coeff = flint.fmpq(coeff_num, coeff_den)
    t_test = flint.fmpq(t_test_num, t_test_den)

    # adj_t(S_W)[alpha] = sum_{(a,b): t_pick[ab]=alpha} S_W[a, b]
    accum_t = [flint.fmpq(0) for _ in range(n_y)]
    nsq = nl * nl
    for k in range(nsq):
        alpha = t_pick_list[k]
        if alpha < 0:
            continue
        accum_t[alpha] += S_flat_list[k]

    # adj_qW(S_W)[alpha] = coeff * sum over (a, b, i, j) of S_W[a, b] where
    # ab_eiej_idx[a, b, i, j] = alpha AND (i, j) in support(M_W).
    # ab_eiej_w_flat is a list of (ab_flat_idx, alpha) pairs (pre-computed).
    accum_q = [flint.fmpq(0) for _ in range(n_y)]
    for ab_idx, alpha in ab_eiej_w_flat:
        accum_q[alpha] += S_flat_list[ab_idx]

    # Combine: r_W[alpha] = t_test * accum_t[alpha] - coeff * accum_q[alpha]
    result = []
    for alpha in range(n_y):
        v = flint.fmpq(0)
        if accum_t[alpha] != 0:
            v += t_test * accum_t[alpha]
        if accum_q[alpha] != 0:
            v -= coeff * accum_q[alpha]
        if v != 0:
            result.append((alpha, v))
    return (w_idx, result)


def _precompute_ab_eiej_pairs(ab_eiej_idx: np.ndarray,
                               M_W_support: np.ndarray,
                               nl: int) -> List[Tuple[int, int]]:
    """Pre-compute (ab_flat_idx, alpha) pairs for one window's
    adj_qW loop. Runs in the MAIN process (can't pickle numpy slices cheaply).
    """
    nz_i, nz_j = np.nonzero(M_W_support)
    if len(nz_i) == 0:
        return []
    idx_slice = ab_eiej_idx[:, :, nz_i, nz_j]
    valid_mask = idx_slice >= 0
    if not valid_mask.any():
        return []
    # ab index (a*nl + b)
    ab_arr = (np.arange(nl)[:, None, None] * nl + np.arange(nl)[None, :, None])
    ab_arr = np.broadcast_to(ab_arr, idx_slice.shape)
    alpha_flat = idx_slice[valid_mask].ravel().tolist()
    ab_flat = ab_arr[valid_mask].ravel().tolist()
    return list(zip(ab_flat, alpha_flat))


def parallel_window_residuals(S_win_fmpq: dict, windows: List[Tuple[int, int]],
                               P: dict, t_test_fmpq,
                               d: int, n_workers: int = 8):
    """Compute window-localizing residual contributions in parallel.

    S_win_fmpq: dict {w_idx: fmpq_mat} for ACTIVE windows only.
    Returns: list of fmpq of length n_y, summed across all windows.
    """
    from multiprocessing import Pool

    n_y = P['n_y']
    n_loc = P['n_loc']
    ab_eiej_idx = P['ab_eiej_idx']
    t_pick_list = list(np.asarray(P['t_pick'], dtype=np.int64).tolist())
    M_mats = P['M_mats']

    t_num = int(t_test_fmpq.p)
    t_den = int(t_test_fmpq.q)

    # Build work items
    work_items = []
    for w_idx, S_fm in S_win_fmpq.items():
        ell, s_lo = windows[w_idx]
        coeff_num = 2 * d
        coeff_den = ell
        # Materialize S_fm as a flat list of fmpq (picklable)
        S_flat = [S_fm[a, b] for a in range(n_loc) for b in range(n_loc)]
        Mw_support = (M_mats[w_idx] != 0).astype(np.int64)
        ab_eiej_pairs = _precompute_ab_eiej_pairs(ab_eiej_idx, Mw_support, n_loc)
        work_items.append((
            w_idx, S_flat, n_loc, t_pick_list, ab_eiej_pairs,
            None, coeff_num, coeff_den, t_num, t_den, n_y,
        ))

    # Run parallel
    if n_workers <= 1 or len(work_items) <= 1:
        results = [_compute_window_contribution_worker(args) for args in work_items]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(_compute_window_contribution_worker, work_items)

    # Aggregate
    total = [flint.fmpq(0) for _ in range(n_y)]
    for w_idx, contrib in results:
        for alpha, v in contrib:
            total[alpha] += v
    return total
