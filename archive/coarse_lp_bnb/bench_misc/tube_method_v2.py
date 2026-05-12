"""Tube method v2 with per-cell adaptive Lipschitz pad and PSD-optimal alpha.

Key improvements over v1:
  (1) PSD-optimal alpha chosen via subgradient ascent on lambda_min(P H P).
      Gives max-possible PSD-ness, reducing tube false-inclusions.
  (2) Per-cell Lipschitz pad: pad = 2 * ||H z||_2 * h * sqrt(d) + max(0,lam_max) * d * h^2
      (depends on z = mu_c - mu*; sound but tighter than worst-case).
  (3) Numba-parallel filter at scale.
"""
import os
import sys
import time
import numpy as np
import numba
from numba import njit, prange


@njit(parallel=True, cache=True)
def _tube_filter_batch_v2(batch_int, mu_star, H, R_sq, d, S, h, lam_max_pos,
                            keep_mask):
    """Per-cell adaptive Lipschitz pad tube filter.

    For each cell at mu_c = batch[b]/S:
      z = mu_c - mu_star
      ||H z||_2 = sqrt(z^T H^2 z) — captures linear correction
      pad_cell = 2 * ||H z||_2 * h * sqrt(d) + lam_max_pos * d * h^2
      Cell is "in tube" iff z^T H z <= R_sq + pad_cell.

    This uses the ACTUAL z to compute the pad (not max-z worst case),
    giving a tighter inclusion criterion.
    """
    B = batch_int.shape[0]
    inv_S = 1.0 / float(S)
    sqrt_d = np.sqrt(float(d))
    quad_pad = lam_max_pos * float(d) * h * h

    for b in prange(B):
        # Compute z = mu_c - mu_star
        # Compute H z (as we go, accumulate z^T H z and z^T H^2 z)
        zHz = 0.0
        zH2z = 0.0
        # First pass: compute Hz vector
        for i in range(d):
            zi = float(batch_int[b, i]) * inv_S - mu_star[i]
            Hzi = 0.0
            for j in range(d):
                zj = float(batch_int[b, j]) * inv_S - mu_star[j]
                Hzi += H[i, j] * zj
            # Accumulate z^T H z
            zHz += zi * Hzi
            # Accumulate z^T H^2 z = (Hz)^T (Hz)
            zH2z += Hzi * Hzi
        # Per-cell linear pad
        Hz_norm = np.sqrt(zH2z)
        pad_cell = 2.0 * Hz_norm * h * sqrt_d + quad_pad
        keep_mask[b] = 1 if zHz <= R_sq + pad_cell else 0


def lipschitz_pad_global(H, h, d):
    """Worst-case (z-independent) cell-Lipschitz pad."""
    eigs = np.linalg.eigvalsh(H)
    lam_max_abs = max(abs(eigs[0]), abs(eigs[-1]))
    return lam_max_abs * float(d) * h * h


def lipschitz_pad_per_cell(z, H, h, d):
    """Per-cell Lipschitz pad for a single cell at mu_c with z = mu_c - mu*.

    pad = 2 * ||H z||_2 * h * sqrt(d) + max(0, lam_max(H)) * d * h^2
    (Linear correction + quadratic correction; both sound.)

    The pad is the maximum possible REDUCTION of z^T H z when moving from
    mu_c to any point in the cell box {mu_c + delta : |delta_i| <= h, sum=0}.
    """
    Hz = H @ z
    Hz_norm = np.linalg.norm(Hz)
    eigs = np.linalg.eigvalsh(H)
    lam_max_pos = max(0.0, eigs[-1])
    return 2.0 * Hz_norm * h * np.sqrt(d) + lam_max_pos * d * h * h


if __name__ == "__main__":
    # Quick test at d=8
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mu_star_optimal import (
        find_mu_star_parallel, compute_active_hessian_exact,
        find_alpha_max_psd, compute_v_kkt, _build_AW_local
    )
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "cloninger-steinerberger"))
    from compositions import generate_canonical_compositions_batched

    d = 8
    c_target = 1.18  # target below val(8)~1.21
    print(f"Testing tube v2 at d={d}, c={c_target}")

    val_d, mu_star = find_mu_star_parallel(d=d, n_restarts=50, n_workers=4)
    print(f"  val_d (NM UB) = {val_d:.6f}")
    H, alpha_kkt, active_idx, res_norm = compute_active_hessian_exact(
        mu_star, d, val_d)
    print(f"  KKT: residual={res_norm:.4f}")
    eigs = np.linalg.eigvalsh(H)
    print(f"  KKT-H eigvals: min={eigs[0]:.4f}, max={eigs[-1]:.4f}")

    # Build active_data for PSD optimization
    conv_len = 2 * d - 1
    tv_list = []
    for ell in range(2, 2 * d + 1):
        scale = 2.0 * d / float(ell)
        for s in range(conv_len - ell + 2):
            A = _build_AW_local(d, ell, s)
            tv_W = scale * float(mu_star @ A @ mu_star)
            tv_list.append((tv_W, ell, s, A, scale))
    tv_list.sort(key=lambda t: -t[0])
    active_data = [t for t in tv_list if t[0] >= tv_list[0][0] - 1e-3]

    alpha_psd, lam_min_psd = find_alpha_max_psd(active_data, d, n_iter=300)
    # Build H_psd from optimal alpha
    H_psd = np.zeros((d, d))
    for k, t in enumerate(active_data):
        H_psd += alpha_psd[k] * 2.0 * t[4] * t[3]
    eigs_psd = np.linalg.eigvalsh(H_psd)
    print(f"  PSD-optimal H eigvals on V: lam_min_V={lam_min_psd:.4f}, "
          f"lam_max={eigs_psd[-1]:.4f}")

    # Check v_kkt with the two alphas
    v_kkt_orig = compute_v_kkt(mu_star, alpha_kkt, active_idx, d)
    v_kkt_psd = sum(alpha_psd[k] * tv_list[i][0]
                    for k, i in enumerate(range(len(active_data))))
    print(f"  v_kkt (KKT alpha) = {v_kkt_orig:.6f}")
    print(f"  v_kkt (PSD alpha) = {v_kkt_psd:.6f}")
    print(f"  margin: KKT v_kkt - c = {v_kkt_orig - c_target:+.4f}, "
          f"PSD v_kkt - c = {v_kkt_psd - c_target:+.4f}")

    # Tube R^2
    R_sq_kkt = 2.0 * (v_kkt_orig - c_target)
    R_sq_psd = 2.0 * (v_kkt_psd - c_target)
    print(f"  R_sq (KKT)={R_sq_kkt:.4f}, R_sq (PSD)={R_sq_psd:.4f}")

    # Compare cell counts in tube using KKT-H vs PSD-H
    S = 16
    h = 1.0 / (2 * S)
    pad_kkt = lipschitz_pad_global(H, h, d)
    pad_psd = lipschitz_pad_global(H_psd, h, d)
    print(f"  global pad (KKT)={pad_kkt:.4f}, global pad (PSD)={pad_psd:.4f}")

    n_total = 0
    n_in_kkt = 0
    n_in_psd = 0
    for batch in generate_canonical_compositions_batched(d, S):
        n_total += batch.shape[0]
        # KKT version
        keep_kkt = np.zeros(batch.shape[0], dtype=np.int8)
        eigs_kkt_max_pos = max(0.0, eigs[-1])
        _tube_filter_batch_v2(batch, mu_star, H, R_sq_kkt, d, S, h,
                                eigs_kkt_max_pos, keep_kkt)
        n_in_kkt += int(np.sum(keep_kkt))
        # PSD version
        keep_psd = np.zeros(batch.shape[0], dtype=np.int8)
        eigs_psd_max_pos = max(0.0, eigs_psd[-1])
        _tube_filter_batch_v2(batch, mu_star, H_psd, R_sq_psd, d, S, h,
                                eigs_psd_max_pos, keep_psd)
        n_in_psd += int(np.sum(keep_psd))

    print(f"\n  At d={d} S={S}: total={n_total:,}")
    print(f"  Tube cells (KKT-H): {n_in_kkt:,} ({100*n_in_kkt/n_total:.2f}%)")
    print(f"  Tube cells (PSD-H): {n_in_psd:,} ({100*n_in_psd/n_total:.2f}%)")
