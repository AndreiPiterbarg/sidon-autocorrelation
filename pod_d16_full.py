"""Full d=14 and d=16 feasibility evaluation with all improvements:
  (1) Parallel multistart Nelder-Mead (64 cores).
  (2) Exact KKT alpha via SLSQP.
  (3) PSD-optimal alpha (subgradient ascent on lambda_min^V).
  (4) Per-cell adaptive Lipschitz pad in tube filter.
  (5) Sample cell vertex enum at varying S.
"""
import os
import sys
import time
import numpy as np
import multiprocessing as mp
import numba

numba.set_num_threads(64)
print(f"Numba threads: {numba.get_num_threads()}", flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "cloninger-steinerberger"))

from mu_star_optimal import (
    find_mu_star_parallel, compute_active_hessian_exact,
    find_alpha_max_psd, compute_v_kkt, _build_AW_local,
)
from tube_method_v2 import _tube_filter_batch_v2, lipschitz_pad_global
from coarse_cascade_prover import (
    _box_certify_cell_vertex, _box_certify_cell_badtr, _box_certify_cell_cctr,
    compute_window_eigen_table, compute_xcap,
)
from compositions import generate_canonical_compositions_batched

c_target = 1.281


def evaluate_d(d, n_restarts):
    print(f"\n{'='*70}", flush=True)
    print(f"  EVALUATION at d={d} for c_target={c_target}", flush=True)
    print(f"{'='*70}", flush=True)

    # Step 1: parallel mu* extraction
    t0 = time.perf_counter()
    val_d, mu_star = find_mu_star_parallel(d=d, n_restarts=n_restarts, n_workers=64)
    print(f"  val_d UB = {val_d:.6f}, time = {time.perf_counter()-t0:.1f}s",
          flush=True)
    print(f"  mu* = {mu_star}", flush=True)

    if val_d < c_target:
        print(f"  STOP: val_d_UB ({val_d:.4f}) < c_target ({c_target}). Cannot prove "
              f"c={c_target} at d={d}.", flush=True)
        return

    # Step 2: KKT exact alpha
    print(f"\n  -- KKT alpha --", flush=True)
    t0 = time.perf_counter()
    H_kkt, alpha_kkt, active_idx, kkt_res = compute_active_hessian_exact(
        mu_star, d, val_d, tol_active=1e-3)
    print(f"  active windows: {len(active_idx)}, KKT residual: {kkt_res:.6f}",
          flush=True)
    eigs_kkt = np.linalg.eigvalsh(H_kkt)
    print(f"  H_kkt eigvals: min={eigs_kkt[0]:.4f}, max={eigs_kkt[-1]:.4f}, "
          f"time={time.perf_counter()-t0:.1f}s", flush=True)

    v_kkt_kkt = compute_v_kkt(mu_star, alpha_kkt, active_idx, d)
    print(f"  v_kkt (KKT) = {v_kkt_kkt:.6f}", flush=True)
    print(f"  margin to c: {v_kkt_kkt - c_target:+.6f}", flush=True)

    # Step 3: PSD-optimal alpha
    print(f"\n  -- PSD-optimal alpha --", flush=True)
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
    print(f"  active count: {len(active_data)}", flush=True)

    t0 = time.perf_counter()
    alpha_psd, lam_min_psd = find_alpha_max_psd(active_data, d, n_iter=500)
    H_psd = np.zeros((d, d))
    for k, t in enumerate(active_data):
        H_psd += alpha_psd[k] * 2.0 * t[4] * t[3]
    eigs_psd = np.linalg.eigvalsh(H_psd)
    v_kkt_psd = sum(alpha_psd[k] * active_data[k][0] for k in range(len(active_data)))
    print(f"  H_psd eigvals: min={eigs_psd[0]:.4f}, max={eigs_psd[-1]:.4f}, "
          f"time={time.perf_counter()-t0:.1f}s", flush=True)
    print(f"  v_kkt (PSD) = {v_kkt_psd:.6f}", flush=True)
    print(f"  margin to c: {v_kkt_psd - c_target:+.6f}", flush=True)

    if v_kkt_psd <= c_target:
        print(f"  STOP: PSD-alpha v_kkt ({v_kkt_psd:.4f}) <= c. Tube method "
              f"still doesn't apply.", flush=True)
        return

    # Step 4: sample cells at varying S — find smallest S that certifies
    print(f"\n  -- Sample cell vertex enum at d={d} --", flush=True)
    V_table, lam_table, valid_mask = compute_window_eigen_table(d)
    op_S = None
    sample_S_list = [76, 101, 151, 201, 301]
    for S in sample_S_list:
        # Quantize mu_star to canonical at S
        cnt = np.round(mu_star * S).astype(int)
        diff = int(cnt.sum() - S)
        if diff != 0:
            frac = mu_star * S - cnt.astype(float)
            if diff > 0:
                idx = np.argsort(frac)[:diff]
                for i in idx: cnt[i] -= 1
            else:
                idx = np.argsort(frac)[diff:]
                for i in idx: cnt[i] += 1
        cnt = np.maximum(cnt, 0)
        if cnt.sum() != S:
            cnt[np.argmax(cnt)] += S - cnt.sum()
        mu_c = cnt.astype(np.float64) / S
        if abs(mu_c.sum() - 1.0) > 1e-6:
            continue

        delta_q = 1.0 / S
        # We need vertex enum (most accurate). At d=16 vertex is expensive.
        if d <= 14:
            t = time.perf_counter()
            cert_v, lb_v = _box_certify_cell_vertex(mu_c, d, delta_q, c_target)
            t_v = time.perf_counter() - t
            cert_bd, lb_bd = _box_certify_cell_badtr(mu_c, d, delta_q, c_target,
                                                       V_table, lam_table, valid_mask)
            cert_cc, lb_cc = _box_certify_cell_cctr(mu_c, d, delta_q, c_target,
                                                      V_table, lam_table, valid_mask)
            best = max(lb_v, lb_bd, lb_cc)
            print(f"    S={S}: vertex_lb={lb_v:.5f}, badtr_lb={lb_bd:.5f}, "
                  f"cctr_lb={lb_cc:.5f}, best={best:.5f}, "
                  f"time_v={t_v:.1f}s", flush=True)
        else:
            # d=16: vertex enum has 2^15 vertices/cell (~600ms parallel). Limit total.
            t = time.perf_counter()
            cert_bd, lb_bd = _box_certify_cell_badtr(mu_c, d, delta_q, c_target,
                                                       V_table, lam_table, valid_mask)
            cert_cc, lb_cc = _box_certify_cell_cctr(mu_c, d, delta_q, c_target,
                                                      V_table, lam_table, valid_mask)
            t_bd = time.perf_counter() - t
            best = max(lb_bd, lb_cc)
            print(f"    S={S}: badtr_lb={lb_bd:.5f}, cctr_lb={lb_cc:.5f}, "
                  f"best={best:.5f}, time={t_bd:.2f}s", flush=True)
        if best >= c_target:
            op_S = S
            print(f"    >>> CERTIFIES at S={S}", flush=True)
            break

    if op_S is None:
        print(f"  Sample cells at d={d}: NONE of S in {sample_S_list} certifies.",
              flush=True)
        # Continue with tube count to see if it's salvageable

    # Step 5: tube count (use PSD-alpha) at small S, project to op_S
    print(f"\n  -- Tube count with PSD-alpha at d={d} --", flush=True)
    R_sq = 2.0 * (v_kkt_psd - c_target)
    print(f"  R_sq = {R_sq:.6f}", flush=True)
    S_count = 21 if d == 14 else 18
    h = 1.0 / (2 * S_count)
    lam_max_pos_psd = max(0.0, eigs_psd[-1])
    pad_global = lipschitz_pad_global(H_psd, h, d)
    print(f"  S_count={S_count}: global pad (PSD)={pad_global:.4f}", flush=True)

    n_total = 0
    n_in_tube = 0
    t0 = time.perf_counter()
    last_progress = t0
    for batch in generate_canonical_compositions_batched(d, S_count):
        n_total += batch.shape[0]
        keep_mask = np.zeros(batch.shape[0], dtype=np.int8)
        # Use PSD-H, R_sq from PSD-alpha (margin still v_kkt - c)
        # Per-cell pad: linear + quadratic
        _tube_filter_batch_v2(batch, mu_star, H_psd, R_sq, d, S_count, h,
                                lam_max_pos_psd, keep_mask)
        n_in_tube += int(np.sum(keep_mask))
        now = time.perf_counter()
        if now - last_progress > 30:
            print(f"    [{now-t0:.0f}s] processed {n_total:,}, in tube "
                  f"{n_in_tube:,} ({100*n_in_tube/n_total:.4f}%)", flush=True)
            last_progress = now
        if n_total > 50_000_000: break

    pct = 100.0 * n_in_tube / max(n_total, 1)
    print(f"\n  Result d={d} S={S_count}: {n_in_tube:,}/{n_total:,} = {pct:.4f}%",
          flush=True)
    print(f"  time = {time.perf_counter()-t0:.1f}s", flush=True)

    # Project to S=51, 76, 101
    from math import comb
    for S_op in [51, 76, 101]:
        n_tot_op = comb(S_op + d - 1, d - 1) // 2
        n_tube_op = int(n_tot_op * pct / 100)
        time_120us = n_tube_op * 120e-6 / 64
        print(f"    Projection S={S_op}: total={n_tot_op:,}, tube={n_tube_op:,}, "
              f"BADTR @120us/64core: {time_120us:.0f}s = {time_120us/3600:.1f}h",
              flush=True)


def main():
    print(f"FULL EVALUATION for c_target={c_target}", flush=True)
    for d in [14, 16]:
        try:
            evaluate_d(d, n_restarts=200)
        except Exception as e:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
