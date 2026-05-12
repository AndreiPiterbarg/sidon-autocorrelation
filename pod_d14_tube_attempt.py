"""Attempt c=1.28 at d=14 using TUBE filter to drastically reduce cell count.

Strategy:
  1. Find mu_star via Nelder-Mead multistart at d=14 (NO SDP).
  2. Compute KKT-weighted active Hessian H.
  3. Estimate v_kkt = sum alpha_W * TV_W(mu_star).
  4. Tube radius R^2 = 2*(v_kkt - c_target). If v_kkt < c, tube method useless.
  5. Iterate canonical compositions, keep ONLY those in tube + small Lipschitz pad.
  6. Run BADTR/CCTR/vertex on tube cells only.

This is a feasibility test — it estimates how many cells need fine treatment
at d=14 c=1.28, projecting tractability on the pod.
"""
import os
import sys
import time
import numpy as np
import numba

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "cloninger-steinerberger"))

numba.set_num_threads(64)

from coarse_cascade_prover import (
    find_mu_star_local,
    compute_active_hessian,
    _build_AW,
    _tube_filter_batch,
    _box_certify_batch_adaptive_v2,
    compute_qdrop_table,
    compute_window_eigen_table,
    compute_xcap,
)
from compositions import generate_canonical_compositions_batched


def attempt_c128_d14(S=21, n_restarts=200):
    d = 14
    c_target = 1.28
    print(f"\n=== c=1.28 attempt at d={d}, S={S} ===")
    print(f"Cell count target ~ C({d+S-1},{d-1})/2", flush=True)

    # Step 1: find mu_star via Nelder-Mead multistart
    t = time.perf_counter()
    val_d, mu_star = find_mu_star_local(d=d, n_restarts=n_restarts)
    print(f"  Nelder-Mead: val_d UB={val_d:.6f}, time={time.perf_counter()-t:.1f}s",
          flush=True)
    print(f"  max_mu={mu_star.max():.4f}, support={(mu_star > 1e-6).sum()}",
          flush=True)

    if val_d < c_target:
        print(f"  WARNING: Nelder-Mead val_d ({val_d:.4f}) < c_target (1.28)")
        print(f"  Either: val(d) really < 1.28 (proof impossible) OR Nelder-Mead suboptimal")
        # Continue anyway to see what happens

    # Step 2: KKT-weighted active Hessian
    t = time.perf_counter()
    H, alpha_star, active_idx, residual = compute_active_hessian(mu_star, d, val_d)
    print(f"  Active windows: {len(active_idx)}, alpha_star sum={alpha_star.sum():.4f}",
          flush=True)
    print(f"  KKT residual: {residual:.6f} (lower is better)", flush=True)
    eig = np.linalg.eigvalsh(H)
    print(f"  H eigenvalues: min={eig.min():.4f}, max={eig.max():.4f}", flush=True)

    # Step 3: v_kkt
    v_kkt = 0.0
    conv_len = 2 * d - 1
    k = 0
    for ell in range(2, 2*d+1):
        scale = 2.0 * d / float(ell)
        for s in range(conv_len - ell + 2):
            for (e_a, s_a) in active_idx:
                if e_a == ell and s_a == s:
                    A = _build_AW(d, ell, s)
                    tv_W = scale * float(mu_star @ A @ mu_star)
                    v_kkt += alpha_star[k] * tv_W
                    k += 1
                    break
    print(f"  v_kkt = {v_kkt:.6f} (margin to c=1.28: {v_kkt - c_target:+.6f})",
          flush=True)

    if v_kkt <= c_target:
        print(f"  Tube method INAPPLICABLE: v_kkt < c_target. Skip.")
        return

    # Step 4: tube radius
    R_sq = 2.0 * (v_kkt - c_target)
    L_H = float(eig.max())
    h = 1.0 / (2 * S)
    cell_lipschitz = abs(L_H) * h * h * d
    R_sq_eff = R_sq + cell_lipschitz
    print(f"  Tube R^2 = {R_sq:.6f}, cell-Lipschitz pad = {cell_lipschitz:.6f}",
          flush=True)
    print(f"  R_sq_eff = {R_sq_eff:.6f}", flush=True)

    # Step 5: filter compositions, count tube cells
    x_cap = compute_xcap(c_target, S, d)
    print(f"  x_cap = {x_cap}", flush=True)
    n_total = 0
    n_in_tube = 0
    n_skipped_xcap = 0
    t = time.perf_counter()
    for batch in generate_canonical_compositions_batched(d, S):
        # Filter x_cap winners
        keep_mask_xcap = np.all(batch <= x_cap, axis=1)
        n_skipped_xcap += int(np.sum(~keep_mask_xcap))
        batch = batch[keep_mask_xcap]
        if batch.shape[0] == 0:
            continue
        n_total += batch.shape[0]
        keep_mask = np.zeros(batch.shape[0], dtype=np.int8)
        _tube_filter_batch(batch, mu_star, H, R_sq_eff, d, S, keep_mask)
        n_in_tube += int(np.sum(keep_mask))
        # Periodic
        if n_total % 10_000_000 == 0:
            print(f"    [progress @ {time.perf_counter()-t:.0f}s] processed "
                  f"{n_total:,} cells, in tube: {n_in_tube:,} "
                  f"({100*n_in_tube/max(n_total,1):.4f}%)", flush=True)
        # Cap if too many
        if n_total > 200_000_000:
            print(f"    [cap at 200M cells] partial result", flush=True)
            break

    elapsed = time.perf_counter() - t
    print(f"\n  RESULT: processed {n_total:,} canonical cells in {elapsed:.1f}s",
          flush=True)
    print(f"          {n_in_tube:,} in tube ({100*n_in_tube/max(n_total,1):.6f}%)",
          flush=True)
    print(f"          {n_skipped_xcap:,} skipped by x_cap",
          flush=True)
    if n_in_tube > 0:
        # Estimate per-tube-cell cost via BADTR
        print(f"  At ~120us/tube-cell on 64 cores: tube cert time ~ "
              f"{n_in_tube * 120e-6 / 64:.1f}s", flush=True)


if __name__ == "__main__":
    attempt_c128_d14(S=21, n_restarts=200)
