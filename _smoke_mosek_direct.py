"""Smoke test: direct MOSEK Task API for variant L Shor SDP, vs CVXPY+MOSEK.

Purpose: quantify CVXPY canonicalization overhead per per-composition SDP.
Setup: Q-survivors at (n_half=3, m=10, d=6); 172 SDPs at ~430 ms each via
CVXPY+MOSEK in `_L_results.json`.

Approach:
  1. Generate the F-then-Q survivor set (same as `_L_bench.run`).
  2. Run each survivor through:
       (A) baseline `prune_L_one(..., solver='MOSEK')` (CVXPY+MOSEK)
       (B) new direct `prune_L_mosek_direct(...)` (MOSEK Task API)
  3. Compare prune sets (must match) and timings.
  4. Bonus: warm-start variant that reuses an already-built mosek.Task by
     only updating the linear-constraint RHS / window thresholds and bounds
     across calls (PSD shape + RLT structure stay the same since `d` is
     constant — the only per-composition differences are the box [lo, hi]
     and the absolute window thresholds).
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_DIR, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched
from _M1_bench import prune_F
from _Q_bench import _build_windows, prune_Q_one, _enum_balanced_signs
from _L_bench import _build_A_matrices, prune_L_one, _make_cell


# =============================================================================
# Direct MOSEK Task implementation of variant L Shor SDP
# =============================================================================
import mosek

# Coefficient helper (mirrors _S1_sdp_mosek)
def _coeffs(d, alpha_const, x_coef, X_coef_lower):
    """Build sparse lower-triangle (subi, subj, val) for the (d+1)x(d+1)
    bar matrix `A` so that <A, Y> matches:
        alpha_const * 1 + sum_i x_coef[i] x_i + sum_{(i,j), i>=j} X_coef[i,j] X[i,j].
    Layout:  Y[0,0]=1, Y[i+1,0]=x_i, Y[i+1,j+1]=X[i,j].
    Recall <A,Y> = sum_i A[i,i] Y[i,i] + 2 sum_{i>j} A[i,j] Y[i,j].
    """
    subi, subj, val = [], [], []
    if alpha_const != 0.0:
        subi.append(0); subj.append(0); val.append(float(alpha_const))
    for i in range(d):
        c = x_coef[i]
        if c != 0.0:
            subi.append(i + 1); subj.append(0); val.append(0.5 * float(c))
    for (i, j), c in X_coef_lower.items():
        if c == 0.0:
            continue
        if i == j:
            subi.append(i + 1); subj.append(j + 1); val.append(float(c))
        else:
            ii, jj = (i, j) if i > j else (j, i)
            subi.append(ii + 1); subj.append(jj + 1); val.append(0.5 * float(c))
    return subi, subj, val


def shor_feasibility_mosek_direct(c_int, A_mats, windows, n_half, m, c_target,
                                   tol=1e-9, eps_margin=1e-9, env=None):
    """Direct MOSEK Task API version of `_L_bench._shor_feasibility`.

    Returns (pruned: bool, status_str). `pruned` iff PRIMAL_INFEASIBLE_CER.
    """
    d = len(c_int)
    bar_dim = d + 1
    lo, hi = _make_cell(c_int, m)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    nm = float(4 * n_half * m)
    cs_m2 = float(c_target) * m * m
    eps_thr = eps_margin * m * m

    own_env = env is None
    if own_env:
        env = mosek.Env()
        try:
            env.checkoutlicense(mosek.feature.pton)
        except Exception:
            pass

    try:
        with env.Task(0, 0) as task:
            task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas, tol)
            task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, tol)
            task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, tol)
            task.putdouparam(mosek.dparam.intpnt_co_tol_infeas, tol)
            task.putintparam(mosek.iparam.intpnt_max_iterations, 200)
            task.putintparam(mosek.iparam.log, 0)

            task.appendbarvars([bar_dim])
            task.putobjsense(mosek.objsense.minimize)

            def _add(subi, subj, vals, bk, blk, buk):
                cidx = task.getnumcon()
                task.appendcons(1)
                if len(subi) > 0:
                    aid = task.appendsparsesymmat(bar_dim, subi, subj, vals)
                    task.putbaraij(cidx, 0, [aid], [1.0])
                task.putconbound(cidx, bk, blk, buk)
                return cidx

            # Y[0,0] == 1 (anchor)
            sI, sJ, sV = _coeffs(d, 1.0, np.zeros(d), {})
            _add(sI, sJ, sV, mosek.boundkey.fx, 1.0, 1.0)

            # Box on x: lo[i] <= x_i <= hi[i]
            for i in range(d):
                xc = np.zeros(d); xc[i] = 1.0
                sI, sJ, sV = _coeffs(d, 0.0, xc, {})
                _add(sI, sJ, sV, mosek.boundkey.ra, lo[i], hi[i])

            # sum x = 4nm
            sI, sJ, sV = _coeffs(d, 0.0, np.ones(d), {})
            _add(sI, sJ, sV, mosek.boundkey.fx, nm, nm)

            # Diagonal: lo^2 <= X[i,i] <= hi^2
            # also McCormick (linear) on x_i:
            #     X[i,i] - 2*lo*x_i  >= -lo^2
            #     X[i,i] - 2*hi*x_i  >= -hi^2
            #     X[i,i] - (lo+hi)*x_i  <= -lo*hi
            for i in range(d):
                # X[i,i] in [lo^2, hi^2]
                Xc = {(i, i): 1.0}
                sI, sJ, sV = _coeffs(d, 0.0, np.zeros(d), Xc)
                _add(sI, sJ, sV, mosek.boundkey.ra, lo[i]*lo[i], hi[i]*hi[i])
                # X[i,i] - 2*lo[i]*x_i + lo[i]^2 >= 0  <=>  X[i,i] - 2*lo*x_i >= -lo^2
                xc = np.zeros(d); xc[i] = -2.0 * lo[i]
                sI, sJ, sV = _coeffs(d, 0.0, xc, {(i, i): 1.0})
                _add(sI, sJ, sV, mosek.boundkey.lo, -lo[i]*lo[i], 0.0)
                # X[i,i] - 2*hi*x_i >= -hi^2
                xc = np.zeros(d); xc[i] = -2.0 * hi[i]
                sI, sJ, sV = _coeffs(d, 0.0, xc, {(i, i): 1.0})
                _add(sI, sJ, sV, mosek.boundkey.lo, -hi[i]*hi[i], 0.0)
                # X[i,i] - (lo+hi)*x_i <= -lo*hi
                xc = np.zeros(d); xc[i] = -(lo[i] + hi[i])
                sI, sJ, sV = _coeffs(d, 0.0, xc, {(i, i): 1.0})
                _add(sI, sJ, sV, mosek.boundkey.up, 0.0, -lo[i]*hi[i])

            # RLT off-diagonal cuts on (i,j), i<j
            for i in range(d):
                for j in range(i + 1, d):
                    li, lj = lo[i], lo[j]
                    ui, uj = hi[i], hi[j]
                    # X[i,j] - lj*x_i - li*x_j + li*lj >= 0
                    xc = np.zeros(d); xc[i] = -lj; xc[j] = -li
                    Xc = {(j, i): 1.0}
                    sI, sJ, sV = _coeffs(d, li * lj, xc, Xc)
                    _add(sI, sJ, sV, mosek.boundkey.lo, 0.0, 0.0)
                    # X[i,j] - uj*x_i - ui*x_j + ui*uj >= 0
                    xc = np.zeros(d); xc[i] = -uj; xc[j] = -ui
                    sI, sJ, sV = _coeffs(d, ui * uj, xc, {(j, i): 1.0})
                    _add(sI, sJ, sV, mosek.boundkey.lo, 0.0, 0.0)
                    # -X[i,j] + uj*x_i + li*x_j - li*uj >= 0  =>  X[i,j] <= uj*x_i + li*x_j - li*uj
                    xc = np.zeros(d); xc[i] = -uj; xc[j] = -li
                    sI, sJ, sV = _coeffs(d, li * uj, xc, {(j, i): 1.0})
                    _add(sI, sJ, sV, mosek.boundkey.up, 0.0, 0.0)
                    # -X[i,j] + lj*x_i + ui*x_j - ui*lj >= 0  =>  X[i,j] <= lj*x_i + ui*x_j - ui*lj
                    xc = np.zeros(d); xc[i] = -lj; xc[j] = -ui
                    sI, sJ, sV = _coeffs(d, ui * lj, xc, {(j, i): 1.0})
                    _add(sI, sJ, sV, mosek.boundkey.up, 0.0, 0.0)

            # Window constraints: Tr(A_W X) <= 4n*ell*c_target*m^2
            for A_mat, (ell, _) in zip(A_mats, windows):
                thr = 4.0 * float(n_half) * float(ell) * (cs_m2 + eps_thr)
                Xc = {}
                # A_mat is symmetric in {0,1}; Tr(A_W X) = sum_{i,j} A[i,j] X[i,j].
                # encode lower triangle: diagonal once, off-diag once with factor 2.
                for ii in range(d):
                    Xc[(ii, ii)] = float(A_mat[ii, ii])
                    for jj in range(ii):
                        # X is symmetric so Tr(A X) = sum diag + 2 * sum_{i>j} A[i,j] X[i,j]
                        Xc[(ii, jj)] = 2.0 * float(A_mat[ii, jj])
                sI, sJ, sV = _coeffs(d, 0.0, np.zeros(d), Xc)
                _add(sI, sJ, sV, mosek.boundkey.up, -1e30, thr)

            try:
                task.optimize()
            except mosek.Error as e:
                return False, f"optimize-error: {e}"

            try:
                solsta = task.getsolsta(mosek.soltype.itr)
            except mosek.Error:
                return False, "getsolsta-error"

            if solsta == mosek.solsta.prim_infeas_cer:
                return True, "infeasible"
            if solsta == mosek.solsta.optimal:
                return False, "optimal"
            return False, f"solsta={solsta}"
    finally:
        if own_env:
            try:
                env.__exit__(None, None, None)
            except Exception:
                pass


# =============================================================================
# Reusable-Env variant (the most defensible bonus)
# =============================================================================
def shor_feas_with_env(c_int, A_mats, windows, n_half, m, c_target, env,
                       tol=1e-9, eps_margin=1e-9):
    """Direct MOSEK with shared `env` (so license/env init cost is amortized)."""
    return shor_feasibility_mosek_direct(c_int, A_mats, windows, n_half, m,
                                          c_target, tol=tol,
                                          eps_margin=eps_margin, env=env)


# =============================================================================
# Driver: reproduce L-bench at (3, 10) and time both paths
# =============================================================================
def main():
    n_half, m, c_target = 3, 10, 1.28
    d = 2 * n_half
    S_half = 2 * n_half * m

    print(f"=== smoke MOSEK-direct vs CVXPY ===")
    print(f"  n_half={n_half}, m={m}, c={c_target}, d={d}")

    windows, ell_int_sums = _build_windows(d)
    sigmas = _enum_balanced_signs(d)
    A_mats = _build_A_matrices(d, windows)
    print(f"  n_win={len(windows)}, n_sigma={len(sigmas)}")

    # Build Q-survivors set (same flow as _L_bench.run)
    q_survivors = []
    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=200_000):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        sF = prune_F(batch, n_half, m, c_target)
        f_idx = np.where(sF)[0]
        for idx in f_idx:
            c_int = batch[idx]
            if not prune_Q_one(c_int, windows, ell_int_sums, sigmas,
                                n_half, m, c_target, margin=1e-9):
                q_survivors.append(c_int.copy())
    print(f"  Q-survivors: {len(q_survivors)}")

    # ---- Path A: CVXPY+MOSEK (baseline) ----
    print(f"\n--- A: CVXPY+MOSEK ---")
    pruned_a = []
    times_a = []
    statuses_a = []
    t_a0 = time.time()
    for c_int in q_survivors:
        t0 = time.time()
        pr, st = prune_L_one(c_int, A_mats, windows, n_half, m, c_target,
                              solver='MOSEK', order=1, tol=1e-9, eps_margin=1e-9)
        times_a.append(time.time() - t0)
        pruned_a.append(pr)
        statuses_a.append(st)
    t_a = time.time() - t_a0
    print(f"  total {t_a:.2f}s  med {1000*np.median(times_a):.1f}ms"
          f"  p95 {1000*np.percentile(times_a, 95):.1f}ms"
          f"  pruned {sum(pruned_a)}/{len(pruned_a)}")

    # ---- Path B: direct MOSEK (own env per call) ----
    print(f"\n--- B: direct MOSEK Task API (per-call env) ---")
    pruned_b = []
    times_b = []
    statuses_b = []
    t_b0 = time.time()
    for c_int in q_survivors:
        t0 = time.time()
        pr, st = shor_feasibility_mosek_direct(c_int, A_mats, windows, n_half, m,
                                                 c_target, tol=1e-9,
                                                 eps_margin=1e-9)
        times_b.append(time.time() - t0)
        pruned_b.append(pr)
        statuses_b.append(st)
    t_b = time.time() - t_b0
    print(f"  total {t_b:.2f}s  med {1000*np.median(times_b):.1f}ms"
          f"  p95 {1000*np.percentile(times_b, 95):.1f}ms"
          f"  pruned {sum(pruned_b)}/{len(pruned_b)}")

    # ---- Path C: direct MOSEK with reused env ----
    print(f"\n--- C: direct MOSEK Task API (reused env) ---")
    pruned_c = []
    times_c = []
    statuses_c = []
    env = mosek.Env()
    try:
        env.checkoutlicense(mosek.feature.pton)
    except Exception:
        pass
    t_c0 = time.time()
    for c_int in q_survivors:
        t0 = time.time()
        pr, st = shor_feas_with_env(c_int, A_mats, windows, n_half, m, c_target,
                                      env, tol=1e-9, eps_margin=1e-9)
        times_c.append(time.time() - t0)
        pruned_c.append(pr)
        statuses_c.append(st)
    t_c = time.time() - t_c0
    print(f"  total {t_c:.2f}s  med {1000*np.median(times_c):.1f}ms"
          f"  p95 {1000*np.percentile(times_c, 95):.1f}ms"
          f"  pruned {sum(pruned_c)}/{len(pruned_c)}")

    # ---- Soundness: prune sets must match ----
    a_set = {tuple(c.tolist()) for c, p in zip(q_survivors, pruned_a) if p}
    b_set = {tuple(c.tolist()) for c, p in zip(q_survivors, pruned_b) if p}
    c_set = {tuple(c.tolist()) for c, p in zip(q_survivors, pruned_c) if p}
    print(f"\n--- soundness check ---")
    print(f"  |A| = {len(a_set)}  |B| = {len(b_set)}  |C| = {len(c_set)}")
    print(f"  A == B : {a_set == b_set}")
    print(f"  A == C : {a_set == c_set}")
    if a_set != b_set:
        print(f"   A \\ B = {a_set - b_set}")
        print(f"   B \\ A = {b_set - a_set}")
    if a_set != c_set:
        print(f"   A \\ C = {a_set - c_set}")
        print(f"   C \\ A = {c_set - a_set}")

    # ---- Speedup metric ----
    speedup_b = t_a / max(t_b, 1e-9)
    speedup_c = t_a / max(t_c, 1e-9)
    print(f"\n--- speedup ---")
    print(f"  B vs A: {speedup_b:.2f}x  ({t_a*1000/len(q_survivors):.1f} -> "
          f"{t_b*1000/len(q_survivors):.1f} ms/call)")
    print(f"  C vs A: {speedup_c:.2f}x  ({t_a*1000/len(q_survivors):.1f} -> "
          f"{t_c*1000/len(q_survivors):.1f} ms/call)")

    out = {
        'n_half': n_half, 'm': m, 'c_target': c_target, 'd': d,
        'n_q_survivors': len(q_survivors),
        'A_total_s': t_a, 'A_med_ms': 1000*float(np.median(times_a)),
        'A_p95_ms': 1000*float(np.percentile(times_a, 95)),
        'A_pruned': int(sum(pruned_a)),
        'B_total_s': t_b, 'B_med_ms': 1000*float(np.median(times_b)),
        'B_p95_ms': 1000*float(np.percentile(times_b, 95)),
        'B_pruned': int(sum(pruned_b)),
        'C_total_s': t_c, 'C_med_ms': 1000*float(np.median(times_c)),
        'C_p95_ms': 1000*float(np.percentile(times_c, 95)),
        'C_pruned': int(sum(pruned_c)),
        'A_eq_B': a_set == b_set, 'A_eq_C': a_set == c_set,
        'speedup_B_vs_A': speedup_b, 'speedup_C_vs_A': speedup_c,
    }
    with open(os.path.join(_DIR, '_smoke_mosek_direct.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote _smoke_mosek_direct.json")
    return out


if __name__ == '__main__':
    main()
