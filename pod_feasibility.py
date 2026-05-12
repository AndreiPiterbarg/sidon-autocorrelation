"""Feasibility sweep for c=1.281 via cascade prover on pod.

Steps:
  A: Nelder-Mead multistart at d=14 (and d=16) — get val_d UB and mu*
  B: KKT active Hessian, compute v_kkt
  C: Sample cell vertex enum at varying S — find operational S
  D: Tube cell count at small S (where filter is fast) — project to operational S
  E: Decision matrix

Outputs feasibility verdict to stdout / log.
"""
import os
import sys
import time
import numpy as np
import numba
from math import comb

numba.set_num_threads(64)
print(f"Numba threads: {numba.get_num_threads()}", flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "cloninger-steinerberger"))

from coarse_cascade_prover import (
    find_mu_star_local,
    compute_active_hessian,
    _box_certify_cell_vertex,
    _box_certify_cell_badtr,
    _box_certify_cell_cctr,
    _tube_filter_batch,
    compute_qdrop_table,
    compute_window_eigen_table,
    compute_xcap,
    _build_AW,
)
from compositions import generate_canonical_compositions_batched

c_target = 1.281


def step_A_nelder_mead(d, n_restarts):
    print(f"\n{'='*70}", flush=True)
    print(f"  STEP A — Nelder-Mead at d={d} ({n_restarts} restarts)", flush=True)
    print('='*70, flush=True)
    t0 = time.perf_counter()
    val_d, mu_star = find_mu_star_local(d=d, n_restarts=n_restarts)
    elapsed = time.perf_counter() - t0
    print(f"  val_d_UB = {val_d:.6f}, time = {elapsed:.1f}s", flush=True)
    print(f"  max(mu*) = {mu_star.max():.4f}, support = {(mu_star > 1e-6).sum()}",
          flush=True)
    print(f"  mu* = {mu_star}", flush=True)
    return val_d, mu_star


def step_B_kkt(d, val_d, mu_star):
    print(f"\n  STEP B — KKT active Hessian and v_kkt at d={d}", flush=True)
    t0 = time.perf_counter()
    H, alpha_star, active_idx, residual = compute_active_hessian(
        mu_star, d, val_d, tol=1e-3)
    elapsed = time.perf_counter() - t0
    print(f"  active windows: {len(active_idx)}, KKT residual: {residual:.4f}, "
          f"time={elapsed:.1f}s", flush=True)
    eig = np.linalg.eigvalsh(H)
    print(f"  H eigvals: min={eig.min():.4f}, max={eig.max():.4f}", flush=True)

    # v_kkt
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

    print(f"  v_kkt = {v_kkt:.6f}", flush=True)
    print(f"  v_kkt - c_target ({c_target}) = {v_kkt - c_target:+.6f}",
          flush=True)
    return H, alpha_star, v_kkt, eig.max()


def step_C_sample_cells(d, mu_star, S_list):
    """For 3 canonical cells closest to mu*, compute cell-min via vertex enum at varying S.
    Find smallest S such that cell-min >= c_target on all of them."""
    print(f"\n  STEP C — Sample-cell vertex enum at d={d}", flush=True)
    V_table, lam_table, valid_mask = compute_window_eigen_table(d)

    # Find canonical compositions closest to mu_star at each S (manual rounding)
    results = {}
    for S in S_list:
        # Round mu_star to nearest grid; quantize counts
        cnt = np.round(mu_star * S).astype(int)
        diff = int(cnt.sum() - S)
        if diff != 0:
            # Adjust by adding/subtracting 1 to bins with biggest fractional residue
            frac = mu_star * S - cnt.astype(float)
            if diff > 0:
                idx = np.argsort(frac)[:diff]
                for i in idx: cnt[i] -= 1
            else:
                idx = np.argsort(frac)[diff:]
                for i in idx: cnt[i] += 1
        if (cnt < 0).any(): cnt = np.maximum(cnt, 0); cnt[np.argmax(cnt)] += S - cnt.sum()
        mu_c = cnt.astype(np.float64) / S
        # Check sum=1 within tol
        if abs(mu_c.sum() - 1.0) > 1e-6:
            print(f"    S={S}: mu_c sum = {mu_c.sum()}, skip", flush=True)
            results[S] = (None, None, None)
            continue

        delta_q = 1.0 / S
        t0 = time.perf_counter()
        cert_v, lb_v = _box_certify_cell_vertex(mu_c, d, delta_q, c_target)
        t_v = time.perf_counter() - t0
        cert_bd, lb_bd = _box_certify_cell_badtr(mu_c, d, delta_q, c_target,
                                                   V_table, lam_table, valid_mask)
        cert_cc, lb_cc = _box_certify_cell_cctr(mu_c, d, delta_q, c_target,
                                                  V_table, lam_table, valid_mask)
        best_lb = max(lb_v, lb_bd, lb_cc)
        ok = best_lb >= c_target
        print(f"    S={S}: vertex_lb={lb_v:.5f}, badtr_lb={lb_bd:.5f}, "
              f"cctr_lb={lb_cc:.5f}, best={best_lb:.5f}, certifies={ok}, "
              f"time_vertex={t_v*1000:.1f}ms", flush=True)
        results[S] = (cert_v, lb_v, lb_bd, lb_cc)
    return results


def step_D_tube_count(d, mu_star, H, v_kkt, S_test, max_eig):
    """Count tube cells at S_test, project to operational S."""
    print(f"\n  STEP D — Tube cell count at d={d}, S_test={S_test}", flush=True)
    if v_kkt <= c_target:
        print(f"  v_kkt <= c_target: tube method INAPPLICABLE", flush=True)
        return None
    R_sq = 2.0 * (v_kkt - c_target)
    h = 1.0 / (2 * S_test)
    cell_lipschitz = abs(max_eig) * h * h * d
    R_sq_eff = R_sq + cell_lipschitz
    print(f"  R_sq={R_sq:.6f}, cell-Lipschitz pad={cell_lipschitz:.6f}, "
          f"R_sq_eff={R_sq_eff:.6f}", flush=True)

    x_cap = compute_xcap(c_target, S_test, d)
    n_total = 0
    n_in_tube = 0
    n_processed_target = 50_000_000  # cap at 50M for speed
    t0 = time.perf_counter()
    last = t0
    for batch in generate_canonical_compositions_batched(d, S_test):
        if x_cap < S_test:
            keep = np.all(batch <= x_cap, axis=1)
            batch = batch[keep]
        if batch.shape[0] == 0: continue
        n_total += batch.shape[0]
        keep_mask = np.zeros(batch.shape[0], dtype=np.int8)
        _tube_filter_batch(batch, mu_star, H, R_sq_eff, d, S_test, keep_mask)
        n_in_tube += int(np.sum(keep_mask))
        now = time.perf_counter()
        if now - last > 30:
            print(f"    [{now-t0:.0f}s] processed {n_total:,}, in tube "
                  f"{n_in_tube:,} ({100*n_in_tube/n_total:.4f}%)", flush=True)
            last = now
        if n_total > n_processed_target: break
    elapsed = time.perf_counter() - t0
    pct = 100.0 * n_in_tube / max(n_total, 1)
    print(f"  Result: processed {n_total:,} in {elapsed:.1f}s, "
          f"{n_in_tube:,} tube ({pct:.4f}%)", flush=True)

    # Project to operational S = {51, 76, 101}
    print(f"  Projections (assuming tube fraction is dimensionless):", flush=True)
    for S_op in [51, 76, 101]:
        n_tot_op = comb(S_op + d - 1, d - 1) // 2
        n_tube_op = int(n_tot_op * pct / 100)
        time_120us = n_tube_op * 120e-6 / 64
        time_200us = n_tube_op * 200e-6 / 64
        print(f"    S={S_op}: total={n_tot_op:,}, tube={n_tube_op:,}, "
              f"BADTR @120us/cell parallel: {time_120us:.0f}s={time_120us/3600:.1f}h, "
              f"@200us: {time_200us:.0f}s={time_200us/3600:.1f}h", flush=True)
    return n_in_tube, n_total, pct


def step_E_decide(verdicts):
    print(f"\n{'='*70}", flush=True)
    print(f"  STEP E — DECISION MATRIX", flush=True)
    print('='*70, flush=True)
    print(f"  c_target = {c_target}", flush=True)
    for d, v in verdicts.items():
        if v is None:
            print(f"  d={d}: skipped (cannot answer)", flush=True)
            continue
        val_d, v_kkt, op_S, tube_proj = v
        if v_kkt < c_target:
            print(f"  d={d}: NOT FEASIBLE — v_kkt={v_kkt:.4f} < {c_target}",
                  flush=True)
            continue
        if op_S is None:
            print(f"  d={d}: NOT FEASIBLE — no S in tested range gives "
                  f"all-cell certification", flush=True)
            continue
        print(f"  d={d}: tentatively feasible. val_d_UB={val_d:.4f}, "
              f"v_kkt={v_kkt:.4f}, op S={op_S}", flush=True)
        print(f"        projected tube run time at S={op_S}: see Step D output",
              flush=True)


def main():
    print(f"FEASIBILITY SWEEP for c={c_target}", flush=True)
    print(f"=" * 70, flush=True)
    verdicts = {}
    for d in [14, 16]:
        try:
            val_d, mu_star = step_A_nelder_mead(d, n_restarts=200)
            if val_d < c_target:
                print(f"  CRITICAL: val_d_UB ({val_d:.4f}) < c_target ({c_target})",
                      flush=True)
                print(f"  proof at d={d} CANNOT succeed", flush=True)
                verdicts[d] = (val_d, val_d, None, None)
                continue
            H, alpha_star, v_kkt, max_eig = step_B_kkt(d, val_d, mu_star)
            if v_kkt < c_target:
                print(f"  v_kkt ({v_kkt:.4f}) < c_target. Tube unusable.",
                      flush=True)
                verdicts[d] = (val_d, v_kkt, None, None)
                continue
            # Step C — sample cells
            S_test_list = [51, 76, 101] if d <= 14 else [51, 76]
            cell_results = step_C_sample_cells(d, mu_star, S_test_list)
            # Pick smallest S where any of vertex/BADTR/CCTR certs at c_target
            op_S = None
            for S in S_test_list:
                if cell_results[S][0] is None: continue
                cert_v, lb_v, lb_bd, lb_cc = cell_results[S]
                if cert_v or lb_bd >= c_target or lb_cc >= c_target:
                    op_S = S
                    break
            print(f"  Operational S (smallest cert): {op_S}", flush=True)
            # Step D — tube count
            S_for_count = 21 if d == 14 else 18
            tube_proj = step_D_tube_count(d, mu_star, H, v_kkt, S_for_count, max_eig)
            verdicts[d] = (val_d, v_kkt, op_S, tube_proj)
        except Exception as e:
            print(f"  EXCEPTION at d={d}: {e}", flush=True)
            verdicts[d] = None
    step_E_decide(verdicts)
    print(f"\n{'='*70}\nFEASIBILITY SWEEP DONE\n{'='*70}", flush=True)


if __name__ == "__main__":
    main()
