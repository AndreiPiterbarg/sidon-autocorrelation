"""Focused d=14 tube measurement on pod.

Goals:
  (1) Find mu_star at d=14 via Nelder-Mead (no SDP).
  (2) Compute KKT-weighted active Hessian H, v_kkt.
  (3) Tube radius R^2 = 2*(v_kkt - c) for c=1.28.
  (4) At d=14 S=21 (~10^9 cells): filter compositions, count tube cells.
  (5) Project tube cell count to S=51 (where margin is tight enough to certify).
  (6) Estimate total proof time.

Outputs the answer to: "how far are we from c=1.28?"
"""
import os
import sys
import time
import numpy as np
import numba

numba.set_num_threads(64)
print(f"Numba threads: {numba.get_num_threads()}", flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "cloninger-steinerberger"))

from coarse_cascade_prover import (
    find_mu_star_local,
    compute_active_hessian,
    _tube_filter_batch,
    compute_xcap,
    _build_AW,
)
from compositions import generate_canonical_compositions_batched


def main():
    d = 14
    c_target = 1.28
    S_test = 21  # for tube count measurement (cell count manageable)
    print(f"\n=== d={d}, c_target={c_target}, S_test={S_test} ===\n", flush=True)

    # Step 1: find mu_star
    print("Step 1: Nelder-Mead multistart for mu_star (n_restarts=300)...",
          flush=True)
    t0 = time.perf_counter()
    val_d, mu_star = find_mu_star_local(d=d, n_restarts=300)
    print(f"  val_d UB = {val_d:.6f}, time={time.perf_counter()-t0:.1f}s", flush=True)
    print(f"  max(mu*) = {mu_star.max():.4f}, support = {(mu_star > 1e-6).sum()}",
          flush=True)
    print(f"  mu* = {mu_star}", flush=True)

    # Step 2: KKT active Hessian
    print(f"\nStep 2: KKT active Hessian...", flush=True)
    t0 = time.perf_counter()
    H, alpha_star, active_idx, residual = compute_active_hessian(
        mu_star, d, val_d, tol=1e-3)
    print(f"  active windows = {len(active_idx)}, alpha sum = {alpha_star.sum():.4f}",
          flush=True)
    print(f"  KKT residual = {residual:.6f}, time={time.perf_counter()-t0:.1f}s",
          flush=True)
    eig = np.linalg.eigvalsh(H)
    print(f"  H eigenvalues: min={eig.min():.4f}, max={eig.max():.4f}", flush=True)

    # Step 3: v_kkt and tube radius
    v_kkt = 0.0
    conv_len = 2 * d - 1
    k_idx = 0
    for ell in range(2, 2*d+1):
        scale = 2.0 * d / float(ell)
        for s in range(conv_len - ell + 2):
            for (e_a, s_a) in active_idx:
                if e_a == ell and s_a == s:
                    A = _build_AW(d, ell, s)
                    tv_W = scale * float(mu_star @ A @ mu_star)
                    v_kkt += alpha_star[k_idx] * tv_W
                    k_idx += 1
                    break
    print(f"\nStep 3: v_kkt = {v_kkt:.6f}", flush=True)
    print(f"  Margin to c=1.28: v_kkt - c = {v_kkt - c_target:+.6f}", flush=True)

    if v_kkt < c_target:
        print(f"\n  *** v_kkt < c_target. Tube method INAPPLICABLE.", flush=True)
        print(f"  Either: val(14) < 1.28 (proof IMPOSSIBLE) OR Nelder-Mead suboptimal.",
              flush=True)
        # Continue with v_kkt anyway to count cells where Lojasiewicz LB might hold
        R_sq = 0.001  # arbitrary small
    else:
        R_sq = 2.0 * (v_kkt - c_target)
    print(f"  Tube R^2 = {R_sq:.6f}", flush=True)

    # Step 4: tube filter at S_test
    print(f"\nStep 4: filter at d={d}, S={S_test} ...", flush=True)
    x_cap = compute_xcap(c_target, S_test, d)
    print(f"  x_cap = {x_cap}", flush=True)
    L_H = float(eig.max())
    h = 1.0 / (2 * S_test)
    cell_lipschitz = abs(L_H) * h * h * d  # cell radius effect
    R_sq_eff = R_sq + cell_lipschitz
    print(f"  cell-Lipschitz pad = {cell_lipschitz:.6f}", flush=True)
    print(f"  R_sq_eff = {R_sq_eff:.6f}", flush=True)

    n_total = 0
    n_in_tube = 0
    n_skipped_xcap = 0
    t0 = time.perf_counter()
    last_progress = t0
    n_processed_target = 200_000_000  # cap for sanity

    for batch in generate_canonical_compositions_batched(d, S_test):
        # x_cap filter
        if x_cap < S_test:
            keep_xc = np.all(batch <= x_cap, axis=1)
            n_skipped_xcap += int(np.sum(~keep_xc))
            batch = batch[keep_xc]
        if batch.shape[0] == 0:
            continue
        n_total += batch.shape[0]
        keep_mask = np.zeros(batch.shape[0], dtype=np.int8)
        _tube_filter_batch(batch, mu_star, H, R_sq_eff, d, S_test, keep_mask)
        n_in_tube += int(np.sum(keep_mask))

        now = time.perf_counter()
        if now - last_progress > 30.0:
            print(f"    [progress @ {now-t0:.0f}s] processed {n_total:,}, "
                  f"in tube {n_in_tube:,} ({100*n_in_tube/max(n_total,1):.4f}%)",
                  flush=True)
            last_progress = now
        if n_total > n_processed_target:
            print(f"    [cap reached]", flush=True)
            break

    elapsed = time.perf_counter() - t0
    pct_in = 100 * n_in_tube / max(n_total, 1)
    print(f"\n  RESULT at d={d} S={S_test}:", flush=True)
    print(f"    processed {n_total:,} cells in {elapsed:.1f}s "
          f"({n_total / max(elapsed, 1):.0f} cells/s)", flush=True)
    print(f"    in tube: {n_in_tube:,} ({pct_in:.6f}%)", flush=True)
    print(f"    skipped by x_cap: {n_skipped_xcap:,}", flush=True)

    # Step 5: project to S=51
    if n_total > 0 and n_in_tube > 0:
        from math import comb
        n_total_S21 = comb(S_test+d-1, d-1) // 2
        n_total_S51 = comb(51+d-1, d-1) // 2
        # Tube fraction is dimensionless (volume ratio); same at any S.
        # Cell count in tube at S = total_cells_at_S * tube_fraction.
        tube_frac = n_in_tube / n_total
        proj_S51 = int(n_total_S51 * tube_frac)
        print(f"\n  PROJECTION to d=14 S=51:", flush=True)
        print(f"    total canonical cells at S=51: {n_total_S51:,}", flush=True)
        print(f"    estimated tube cells at S=51: {proj_S51:,}", flush=True)
        print(f"    at 120us/tube-cell on 64 cores: tube cert = "
              f"{proj_S51 * 120e-6 / 64:.0f}s = "
              f"{proj_S51 * 120e-6 / 64 / 60:.1f}min", flush=True)
        print(f"    plus tube-filter cost: {n_total_S51 * 200 / 1e10 / 64:.0f}s "
              f"(filter at 1e10 ops/sec/core)", flush=True)


if __name__ == "__main__":
    main()
