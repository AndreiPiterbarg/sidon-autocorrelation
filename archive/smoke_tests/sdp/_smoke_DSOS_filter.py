"""DSOS / SDSOS prefilter smoke test (Ahmadi-Majumdar 2014, arXiv:1306.0411).

CRITICAL CONE-DIRECTION CAVEAT:
  DD subset SDD subset PSD.  In *primal feasibility*, replacing PSD by the
  smaller cone DD (or SDD) gives a SMALLER feasibility region. Hence:
    DSOS-INFEASIBLE  is WEAKER than  PSD-INFEASIBLE
       (DSOS-pruned does NOT imply PSD-pruned -- UNSOUND for our use).
    PSD-INFEASIBLE   is STRONGER than DSOS-INFEASIBLE
       (L-pruned implies DSOS-pruned only if DD spans PSD which it doesn't).
  So a primal DSOS feasibility test is NOT a sound prefilter. The Ahmadi-
  Majumdar paper uses DSOS/SDSOS to OVER-approximate SoS in the *DUAL*
  (polynomial-positivity certificates), where DSOS-cert => SoS-cert.

Testing direction:
  Soundness check: every DSOS-prune must also be L-prune (would be required
  for safe prefilter use).  We expect this to FAIL.

Setting: n_half=4, m=10, c_target=1.28, d=8.  Q-survivors: 964 from
_L_results.json. L pruned 163/964 = 16.9%.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
from typing import Optional

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched
from _M1_bench import prune_F
from _Q_bench import _build_windows, prune_Q_one, _enum_balanced_signs
from _L_bench import _build_A_matrices, _make_cell, prune_L_one


# ----------------------------------------------------------------------
# DSOS feasibility (LP)
# ----------------------------------------------------------------------
def _dsos_feasibility(c_int, lo, hi, A_mats, windows, n_half, m, c_target,
                       solver='HIGHS', eps_margin=1e-9, verbose=False):
    """LP feasibility test: replace Y >> 0 with Y in DD cone.

    DD encoding (LP):
       For each i:  Y_ii >= sum_{j != i} t_ij
       For each i!=j:  t_ij >= Y_ij,  t_ij >= -Y_ij    (so t_ij >= |Y_ij|)
    where Y[0,0]=1, Y[0,1:]=x, Y[1:,0]=x, Y[1:,1:] = X.

    Otherwise constraints same as Shor: box, sum, RLT, window.

    INFEASIBLE => composition DSOS-pruned (looser, so weaker than L).
    """
    import cvxpy as cp

    d = len(c_int)
    nm = float(4 * n_half * m)
    cs_m2 = float(c_target) * m * m
    eps_thr = eps_margin * m * m
    N = d + 1   # size of Y

    x = cp.Variable(d)
    X = cp.Variable((d, d), symmetric=True)
    # Off-diagonal absolute-value variables (upper triangle of Y)
    # Y has indexing (0..N-1, 0..N-1).  Y[0,0]=1, Y[0,k+1] = x_k,
    # Y[k+1, j+1] = X[k, j].  We need t[a,b] >= |Y[a,b]| for a < b.
    n_off = N * (N - 1) // 2
    t_off = cp.Variable(n_off, nonneg=True)

    cons = []
    cons += [x >= lo, x <= hi]
    cons += [cp.sum(x) == nm]

    # Diagonal: x_i^2 ranges (same as Shor)
    for i in range(d):
        cons += [X[i, i] >= lo[i] * lo[i], X[i, i] <= hi[i] * hi[i]]
        cons += [X[i, i] >= 2.0 * lo[i] * x[i] - lo[i] * lo[i]]
        cons += [X[i, i] >= 2.0 * hi[i] * x[i] - hi[i] * hi[i]]
        cons += [X[i, i] <= (lo[i] + hi[i]) * x[i] - lo[i] * hi[i]]

    # RLT off-diagonal (same as Shor)
    for i in range(d):
        for j in range(i + 1, d):
            li, lj = lo[i], lo[j]
            ui, uj = hi[i], hi[j]
            cons += [X[i, j] >= lj * x[i] + li * x[j] - li * lj]
            cons += [X[i, j] >= uj * x[i] + ui * x[j] - ui * uj]
            cons += [X[i, j] <= ui * x[j] + lj * x[i] - ui * lj]
            cons += [X[i, j] <= uj * x[i] + li * x[j] - li * uj]

    # Window constraints
    for A_mat, (ell, _) in zip(A_mats, windows):
        thr = 4.0 * float(n_half) * float(ell) * (cs_m2 + eps_thr)
        cons += [cp.trace(A_mat @ X) <= thr]

    # DD encoding for Y. Index pairs (a, b) with a < b for upper-triangle.
    # off-diag index map
    def y_entry(a, b):
        # Y[a,b] in terms of x, X
        if a == 0 and b == 0:
            return cp.Constant(1.0)
        if a == 0:
            return x[b - 1]
        if b == 0:
            return x[a - 1]
        return X[a - 1, b - 1]

    # build t_off mapping
    pair_idx = {}
    k = 0
    for a in range(N):
        for b in range(a + 1, N):
            pair_idx[(a, b)] = k
            k += 1
    assert k == n_off

    # |Y[a,b]| <= t_off[idx]
    for (a, b), idx in pair_idx.items():
        ye = y_entry(a, b)
        cons += [ye <= t_off[idx], -ye <= t_off[idx]]

    # Diagonal dominance: Y[a,a] >= sum_{b != a} |Y[a,b]|
    # row sums of t_off: collect for each a all t_off[(min,max)] involving a
    for a in range(N):
        row_terms = []
        for b in range(N):
            if b == a:
                continue
            lo_p, hi_p = (a, b) if a < b else (b, a)
            row_terms.append(t_off[pair_idx[(lo_p, hi_p)]])
        if a == 0:
            # Y[0,0] = 1
            cons += [cp.sum(row_terms) <= 1.0]
        else:
            cons += [X[a - 1, a - 1] >= cp.sum(row_terms)]

    prob = cp.Problem(cp.Minimize(0), cons)
    try:
        if solver == 'HIGHS':
            prob.solve(solver='HIGHS', verbose=verbose)
        else:
            prob.solve(solver=solver, verbose=verbose)
    except Exception as e:
        return False, f'EXC:{type(e).__name__}'

    status = prob.status
    if status == 'infeasible':
        return True, status
    return False, status


# ----------------------------------------------------------------------
# SDSOS feasibility (SOCP)
# ----------------------------------------------------------------------
def _sdsos_feasibility(c_int, lo, hi, A_mats, windows, n_half, m, c_target,
                        solver='CLARABEL', eps_margin=1e-9, verbose=False):
    """SOCP feasibility test: replace Y >> 0 with Y in SDD cone.

    SDD = {Y : Y = sum_{i<j} M^{ij}, M^{ij} PSD on rows/cols (i,j) only}
    Equivalent to: there exist alpha_ij, beta_ij, gamma_ij with
       Y_ii = sum_{j != i} alpha_ij,
       Y_ij = gamma_ij                                for i != j (i<j),
       [[alpha_ij, gamma_ij],[gamma_ij, beta_ij]] PSD,    for each i<j,
       Y_jj = sum_{i != j} beta_ij                    (consistent splits).

    The 2x2 PSD constraint is a rotated SOC:
       gamma_ij^2 <= alpha_ij * beta_ij,   alpha_ij, beta_ij >= 0.

    INFEASIBLE => composition SDSOS-pruned (still looser than L,
    but tighter than DSOS).
    """
    import cvxpy as cp

    d = len(c_int)
    nm = float(4 * n_half * m)
    cs_m2 = float(c_target) * m * m
    eps_thr = eps_margin * m * m
    N = d + 1

    x = cp.Variable(d)
    X = cp.Variable((d, d), symmetric=True)

    # Per-pair (i<j) split variables: alpha_ij, beta_ij, gamma_ij
    # We store them as cvxpy expressions: gamma is the off-diag of Y, but we
    # split alpha (i-side diag), beta (j-side diag).
    pairs = []
    for a in range(N):
        for b in range(a + 1, N):
            pairs.append((a, b))
    n_pairs = len(pairs)
    alpha = cp.Variable(n_pairs, nonneg=True)   # row-i contribution to diag
    beta = cp.Variable(n_pairs, nonneg=True)    # row-j contribution to diag
    # gamma = Y[a,b] (no extra var; reuse x or X)

    pair_idx = {p: k for k, p in enumerate(pairs)}

    cons = []
    cons += [x >= lo, x <= hi]
    cons += [cp.sum(x) == nm]

    for i in range(d):
        cons += [X[i, i] >= lo[i] * lo[i], X[i, i] <= hi[i] * hi[i]]
        cons += [X[i, i] >= 2.0 * lo[i] * x[i] - lo[i] * lo[i]]
        cons += [X[i, i] >= 2.0 * hi[i] * x[i] - hi[i] * hi[i]]
        cons += [X[i, i] <= (lo[i] + hi[i]) * x[i] - lo[i] * hi[i]]

    for i in range(d):
        for j in range(i + 1, d):
            li, lj = lo[i], lo[j]
            ui, uj = hi[i], hi[j]
            cons += [X[i, j] >= lj * x[i] + li * x[j] - li * lj]
            cons += [X[i, j] >= uj * x[i] + ui * x[j] - ui * uj]
            cons += [X[i, j] <= ui * x[j] + lj * x[i] - ui * lj]
            cons += [X[i, j] <= uj * x[i] + li * x[j] - li * uj]

    for A_mat, (ell, _) in zip(A_mats, windows):
        thr = 4.0 * float(n_half) * float(ell) * (cs_m2 + eps_thr)
        cons += [cp.trace(A_mat @ X) <= thr]

    # Y entry helper
    def y_entry(a, b):
        if a == 0 and b == 0:
            return cp.Constant(1.0)
        if a == 0:
            return x[b - 1]
        if b == 0:
            return x[a - 1]
        return X[a - 1, b - 1]

    # 2x2 PSD constraints: [[alpha_p, gamma_p],[gamma_p, beta_p]] PSD
    # Use cvxpy 2x2 PSD; cvxpy will compile to SOCP.
    for k, (a, b) in enumerate(pairs):
        gamma_ab = y_entry(a, b)
        # 2x2 matrix
        Mab = cp.bmat([[cp.reshape(alpha[k], (1, 1), order='C'),
                          cp.reshape(gamma_ab, (1, 1), order='C')],
                         [cp.reshape(gamma_ab, (1, 1), order='C'),
                          cp.reshape(beta[k], (1, 1), order='C')]])
        cons += [Mab >> 0]

    # Diagonal split: Y_aa = sum over pairs that include 'a' of (alpha or beta)
    for a in range(N):
        contribs = []
        for k, (p, q) in enumerate(pairs):
            if p == a:
                contribs.append(alpha[k])
            elif q == a:
                contribs.append(beta[k])
        if a == 0:
            cons += [cp.sum(contribs) == 1.0]
        else:
            cons += [X[a - 1, a - 1] == cp.sum(contribs)]

    prob = cp.Problem(cp.Minimize(0), cons)
    try:
        prob.solve(solver=solver, verbose=verbose)
    except Exception as e:
        return False, f'EXC:{type(e).__name__}'

    status = prob.status
    if status == 'infeasible':
        return True, status
    return False, status


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def main():
    n_half, m, c_target = 4, 10, 1.28
    d = 2 * n_half
    S_full = 4 * n_half * m
    S_half = 2 * n_half * m

    print(f"=== DSOS/SDSOS smoke test, n_half={n_half}, m={m}, "
           f"c_target={c_target}, d={d} ===")

    windows, ell_int_sums = _build_windows(d)
    n_win = len(windows)
    sigmas = _enum_balanced_signs(d)
    A_mats = _build_A_matrices(d, windows)
    print(f"  n_win={n_win}, n_sigma={len(sigmas)}")

    # Warm
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = 2 * m
    prune_F(warm, n_half, m, c_target)

    # ----- Step 1: collect Q-survivors ------
    q_survivors = []
    n_processed = 0
    t0 = time.time()
    for half_batch in generate_compositions_batched(n_half, S_half,
                                                       batch_size=200_000):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        n_processed += len(batch)
        sF = prune_F(batch, n_half, m, c_target)
        f_idx = np.where(sF)[0]
        for idx in f_idx:
            c_int = batch[idx]
            if not prune_Q_one(c_int, windows, ell_int_sums, sigmas,
                                 n_half, m, c_target):
                q_survivors.append(c_int.copy())
    t_collect = time.time() - t0
    print(f"  processed={n_processed:,}, Q-survivors={len(q_survivors)} "
           f"(collected in {t_collect:.1f}s)")

    # ----- Step 2: Run DSOS, SDSOS, L on each Q-survivor ------
    # Cap to keep total runtime reasonable; still informative.
    MAX_TEST = min(len(q_survivors), 250)
    print(f"  Testing on first {MAX_TEST} Q-survivors")

    dsos_results = []   # list of (pruned, status, t)
    sdsos_results = []
    l_results = []

    n_dsos_pruned = 0
    n_sdsos_pruned = 0
    n_l_pruned = 0

    soundness_dsos_violations = []
    soundness_sdsos_violations = []

    for i in range(MAX_TEST):
        c_int = q_survivors[i]
        lo, hi = _make_cell(c_int, m)

        # DSOS (LP via HiGHS)
        t1 = time.time()
        d_pruned, d_status = _dsos_feasibility(c_int, lo, hi, A_mats, windows,
                                                  n_half, m, c_target,
                                                  solver='HIGHS')
        t_d = time.time() - t1
        dsos_results.append((bool(d_pruned), str(d_status), t_d))
        if d_pruned:
            n_dsos_pruned += 1

        # SDSOS (SOCP via CLARABEL)
        t1 = time.time()
        sd_pruned, sd_status = _sdsos_feasibility(c_int, lo, hi, A_mats, windows,
                                                     n_half, m, c_target,
                                                     solver='CLARABEL')
        t_sd = time.time() - t1
        sdsos_results.append((bool(sd_pruned), str(sd_status), t_sd))
        if sd_pruned:
            n_sdsos_pruned += 1

        # L (Shor SDP via MOSEK) — gold standard for comparison
        t1 = time.time()
        l_pruned, l_status = prune_L_one(c_int, A_mats, windows, n_half, m,
                                            c_target, solver='MOSEK', order=1)
        t_l = time.time() - t1
        l_results.append((bool(l_pruned), str(l_status), t_l))
        if l_pruned:
            n_l_pruned += 1

        # Soundness checks: DSOS/SDSOS pruned => L pruned
        if d_pruned and not l_pruned:
            soundness_dsos_violations.append(i)
        if sd_pruned and not l_pruned:
            soundness_sdsos_violations.append(i)

        if (i + 1) % 25 == 0:
            print(f"    [{i+1}/{MAX_TEST}] DSOS_pr={n_dsos_pruned} "
                   f"SDSOS_pr={n_sdsos_pruned} L_pr={n_l_pruned}")

    # ----- Step 3: Aggregate ------
    dsos_times = [r[2] for r in dsos_results]
    sdsos_times = [r[2] for r in sdsos_results]
    l_times = [r[2] for r in l_results]

    print(f"\n--- Summary on {MAX_TEST} Q-survivors ---")
    print(f"  DSOS pruned: {n_dsos_pruned} / {MAX_TEST} "
           f"({100*n_dsos_pruned/MAX_TEST:.2f}%)")
    print(f"  SDSOS pruned: {n_sdsos_pruned} / {MAX_TEST} "
           f"({100*n_sdsos_pruned/MAX_TEST:.2f}%)")
    print(f"  L pruned: {n_l_pruned} / {MAX_TEST} "
           f"({100*n_l_pruned/MAX_TEST:.2f}%)")
    print(f"  L's prunes caught by DSOS:  "
           f"{sum(1 for d, l in zip(dsos_results, l_results) if d[0] and l[0])} "
           f"/ {n_l_pruned}")
    print(f"  L's prunes caught by SDSOS: "
           f"{sum(1 for d, l in zip(sdsos_results, l_results) if d[0] and l[0])} "
           f"/ {n_l_pruned}")
    print(f"  DSOS  t med/p95/max: {1000*np.median(dsos_times):.1f} / "
           f"{1000*np.percentile(dsos_times, 95):.1f} / "
           f"{1000*np.max(dsos_times):.1f} ms")
    print(f"  SDSOS t med/p95/max: {1000*np.median(sdsos_times):.1f} / "
           f"{1000*np.percentile(sdsos_times, 95):.1f} / "
           f"{1000*np.max(sdsos_times):.1f} ms")
    print(f"  L     t med/p95/max: {1000*np.median(l_times):.1f} / "
           f"{1000*np.percentile(l_times, 95):.1f} / "
           f"{1000*np.max(l_times):.1f} ms")
    print(f"  DSOS soundness violations: {len(soundness_dsos_violations)}")
    print(f"  SDSOS soundness violations: {len(soundness_sdsos_violations)}")

    out = {
        'config': {'n_half': n_half, 'm': m, 'c_target': c_target, 'd': d},
        'n_q_survivors_total': len(q_survivors),
        'n_tested': MAX_TEST,
        'dsos_pruned': n_dsos_pruned,
        'sdsos_pruned': n_sdsos_pruned,
        'l_pruned': n_l_pruned,
        'dsos_caught_of_l': sum(1 for d, l in zip(dsos_results, l_results) if d[0] and l[0]),
        'sdsos_caught_of_l': sum(1 for d, l in zip(sdsos_results, l_results) if d[0] and l[0]),
        'dsos_times_ms': {
            'median': 1000 * float(np.median(dsos_times)),
            'p95': 1000 * float(np.percentile(dsos_times, 95)),
            'max': 1000 * float(np.max(dsos_times)),
            'mean': 1000 * float(np.mean(dsos_times)),
        },
        'sdsos_times_ms': {
            'median': 1000 * float(np.median(sdsos_times)),
            'p95': 1000 * float(np.percentile(sdsos_times, 95)),
            'max': 1000 * float(np.max(sdsos_times)),
            'mean': 1000 * float(np.mean(sdsos_times)),
        },
        'l_times_ms': {
            'median': 1000 * float(np.median(l_times)),
            'p95': 1000 * float(np.percentile(l_times, 95)),
            'max': 1000 * float(np.max(l_times)),
            'mean': 1000 * float(np.mean(l_times)),
        },
        'soundness_dsos_violations': len(soundness_dsos_violations),
        'soundness_sdsos_violations': len(soundness_sdsos_violations),
    }

    out_path = os.path.join(_dir, '_smoke_DSOS_filter.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")
    return out


if __name__ == '__main__':
    main()
