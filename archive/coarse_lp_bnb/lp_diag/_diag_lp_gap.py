"""Diagnose the LP gap in the per-box epigraph LP at d=20 near mu*.

Computes:
  1. Approximate mu* via projected-gradient on min_mu max_W TV_W(mu) (LSE smoothed).
  2. True max_W TV_W on a tight box around mu* (via random sampling + grid).
  3. LP value via _solve_epigraph_lp.
  4. Gap = true - LP.
  5. Identifies binding window, McCormick-face slacks, row-sum equality residuals.
"""
import os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from interval_bnb.windows import build_windows
from interval_bnb.bound_epigraph import _solve_epigraph_lp


def find_mu_star_quick(d, n_starts=200, n_iters=4000, beta=200.0, seed=0):
    """LSE-smoothed projected-gradient minimax min_mu max_W TV_W(mu)."""
    A_list, c_list = [], []
    for ell in range(2, 2 * d + 1):
        c_W = 2.0 * d / float(ell)
        for s in range(2 * d - ell + 1):
            A = np.zeros((d, d))
            for i in range(d):
                jl = max(s - i, 0); jh = min(s + ell - 2 - i, d - 1)
                for j in range(jl, jh + 1): A[i, j] = 1.0
            A = 0.5 * (A + A.T)
            A_list.append(A); c_list.append(c_W)
    A_stack = np.stack(A_list); c_W = np.array(c_list)

    def project(v):
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1
        idx = np.arange(1, n + 1)
        rho_arr = np.where(u - cssv / idx > 0)[0]
        rho = rho_arr[-1]
        theta = cssv[rho] / (rho + 1)
        return np.maximum(v - theta, 0.0)

    rng = np.random.RandomState(seed)
    best_mu, best_val = None, np.inf
    for k in range(n_starts):
        # symmetric warm start
        if k == 0:
            mu = np.ones(d) / d
        elif k < 5:
            x = np.linspace(0, 1, d) ** 2 + 0.01
            mu = x / x.sum()
            if k == 2: mu = mu[::-1]
        else:
            mu = rng.dirichlet(np.ones(d) * (0.3 + rng.rand() * 2))
        lr = 0.05
        for it in range(n_iters):
            Amu = A_stack @ mu
            tv = c_W * (Amu * mu[None, :]).sum(axis=1)
            # softmax
            m = tv.max()
            w = np.exp(beta * (tv - m)); w /= w.sum()
            grad = 2.0 * (w[:, None] * c_W[:, None] * Amu).sum(axis=0)
            mu = project(mu - lr * grad)
            if it % 1000 == 999: lr *= 0.5
        Amu = A_stack @ mu
        tv = c_W * (Amu * mu[None, :]).sum(axis=1)
        v = tv.max()
        if v < best_val:
            best_val, best_mu = v, mu.copy()
    return best_mu, best_val, A_stack, c_W


def true_max_tv_on_box(lo, hi, A_stack, c_W, n_samples=20000, seed=1):
    """Lower bound on true max over (mu in box ∩ simplex) of max_W TV_W."""
    rng = np.random.RandomState(seed)
    d = len(lo)
    best = -np.inf
    best_mu = None
    # Strategy: sample in box, project to simplex; or Dirichlet then clip.
    for _ in range(n_samples):
        x = rng.uniform(lo, hi)
        s = x.sum()
        if s <= 0: continue
        mu = x / s  # might violate box; reject
        if np.all(mu >= lo - 1e-12) and np.all(mu <= hi + 1e-12):
            Amu = A_stack @ mu
            tv = c_W * (Amu * mu).sum(axis=1)
            v = tv.max()
            if v > best:
                best = v; best_mu = mu.copy()
    return best, best_mu


def main():
    d = 20
    print(f"[1] Finding approximate mu* at d={d}...")
    t0 = time.time()
    mu_star, val_star, A_stack, c_W = find_mu_star_quick(d, n_starts=80, n_iters=3000)
    print(f"  mu* found in {time.time()-t0:.1f}s, val ≈ {val_star:.6f}")
    print(f"  mu* = {np.round(mu_star, 4)}")

    windows = build_windows(d)
    print(f"  #windows = {len(windows)}")

    # Build a tight box of width 1e-4 around mu*
    half_w = 5e-5
    lo = np.maximum(mu_star - half_w, 0.0)
    hi = np.minimum(mu_star + half_w, 1.0)
    # Note: this box may NOT contain a feasible simplex point unless we allow drift.
    # Adjust so that sum(lo) <= 1 <= sum(hi)
    print(f"\n[2] Box around mu*: width={2*half_w:.2e}, sum(lo)={lo.sum():.6f}, sum(hi)={hi.sum():.6f}")
    # If sum(lo) > 1 or sum(hi) < 1, widen sym
    if lo.sum() > 1 or hi.sum() < 1:
        # widen
        delta = 1.0 - lo.sum() if lo.sum() > 1 else 0.0
        # ensure simplex-feasible mu* is interior
        pass

    # Run LP
    print("\n[3] Solving LP...")
    t0 = time.time()
    lp_val, ineq_marg, eq_marg, lo_marg, up_marg = _solve_epigraph_lp(lo, hi, windows, d)
    print(f"  LP time: {time.time()-t0:.2f}s, lp_val = {lp_val:.8f}")

    # True value: at mu_star itself (which is in the box by construction)
    # Verify
    if np.all(mu_star >= lo) and np.all(mu_star <= hi):
        Amu = A_stack @ mu_star
        tv_at_star = c_W * (Amu * mu_star).sum(axis=1)
        true_val_lower = tv_at_star.max()
        print(f"  True value at mu* (in box): {true_val_lower:.8f}")
    else:
        print("  mu_star outside box (should not happen)")
        true_val_lower = val_star

    # Search for true max on box
    t0 = time.time()
    sample_max, sample_mu = true_max_tv_on_box(lo, hi, A_stack, c_W, n_samples=5000)
    print(f"  Sampled true_max on box: {sample_max:.8f} (t={time.time()-t0:.1f}s)")
    if sample_max > true_val_lower:
        true_val_lower = sample_max

    gap = true_val_lower - lp_val
    print(f"\n[4] LP GAP = true - lp = {true_val_lower:.8f} - {lp_val:.8f} = {gap:.6e}")

    # Identify binding window from epigraph row marginals
    # ineq_marg layout: [SW (n_y), NE (n_y), NW (n_y), SE (n_y), epigraph (n_W)]
    n_y = d * d
    n_W = len(windows)
    if ineq_marg is not None:
        epi_marg = ineq_marg[4 * n_y : 4 * n_y + n_W]
        # Most-active = most negative marginal (binding)
        order = np.argsort(epi_marg)
        print(f"\n[5] Top-5 binding windows (epigraph constraints):")
        for k in range(min(5, n_W)):
            kw = order[k]
            w = windows[kw]
            print(f"  W{kw} (ell={w.ell}, s={w.s_lo}, scale={w.scale:.4f}): epi_marg={epi_marg[kw]:.6e}")

    # Recover Y from LP: re-solve to get full primal soln. We need it for slack diagnosis.
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, coo_matrix
    # Re-build the LP and get full primal var values.
    # For shortcut, just parse from result; our wrapper drops it. Redo:
    lp_val2, primal = _solve_epigraph_lp_with_primal(lo, hi, windows, d)
    if primal is not None:
        Y = primal[:n_y].reshape(d, d)
        mu_lp = primal[n_y : n_y + d]
        z = primal[n_y + d]
        print(f"\n[6] LP primal: z = {z:.8f}, mu sum = {mu_lp.sum():.8f}")
        print(f"  mu_lp = {np.round(mu_lp, 5)}")

        # Row-sum equality residuals
        rowsum_res = Y.sum(axis=1) - mu_lp
        print(f"\n[7] Row-sum eq residual ||sum_j Y_ij - mu_i||_inf = {np.abs(rowsum_res).max():.3e}")

        # McCormick face slacks: for each (i,j),
        #   SW: Y_ij - (lo_j*mu_i + lo_i*mu_j - lo_i*lo_j) >= 0
        #   NE: Y_ij - (hi_j*mu_i + hi_i*mu_j - hi_i*hi_j) >= 0
        #   NW: (lo_j*mu_i + hi_i*mu_j - lo_j*hi_i) - Y_ij >= 0
        #   SE: (hi_j*mu_i + lo_i*mu_j - hi_j*lo_i) - Y_ij >= 0
        I, J = np.meshgrid(np.arange(d), np.arange(d), indexing='ij')
        SW = Y - (lo[J] * mu_lp[I] + lo[I] * mu_lp[J] - lo[I] * lo[J])
        NE = Y - (hi[J] * mu_lp[I] + hi[I] * mu_lp[J] - hi[I] * hi[J])
        NW = (lo[J] * mu_lp[I] + hi[I] * mu_lp[J] - lo[J] * hi[I]) - Y
        SE = (hi[J] * mu_lp[I] + lo[I] * mu_lp[J] - hi[J] * lo[I]) - Y
        print(f"\n[8] McCormick face slacks:")
        print(f"  SW: min={SW.min():.3e}  mean={SW.mean():.3e}")
        print(f"  NE: min={NE.min():.3e}  mean={NE.mean():.3e}")
        print(f"  NW: min={NW.min():.3e}  mean={NW.mean():.3e}")
        print(f"  SE: min={SE.min():.3e}  mean={SE.mean():.3e}")

        # Compute Y_lp - mu_lp_outer (the bilinear violation)
        Yconv = mu_lp[:, None] * mu_lp[None, :]
        Y_excess = Y - Yconv
        print(f"\n[9] Y_lp - mu_lp ⊗ mu_lp:")
        print(f"  ||Y - mu*mu^T||_F = {np.linalg.norm(Y - Yconv):.3e}")
        print(f"  max excess Y - mu*mu^T = {Y_excess.max():.3e}  at {np.unravel_index(Y_excess.argmax(), Y_excess.shape)}")
        print(f"  min excess (deficit) = {Y_excess.min():.3e}")

        # Symmetry: how non-symmetric is Y?
        Y_asym = (Y - Y.T)
        print(f"\n[10] Y symmetry: ||Y - Y^T||_inf = {np.abs(Y_asym).max():.3e}, ||Y - Y^T||_F = {np.linalg.norm(Y_asym):.3e}")

        # Window-by-window: TV computed at mu_lp vs LP z
        TVs_lp = c_W * (A_stack @ mu_lp * mu_lp).sum(axis=1)
        # And, what the LP "sees" as TV via Y:
        # TV_W(LP) = scale_W * sum_{(i,j) in S_W} Y_ij
        TVs_via_Y = np.zeros(n_W)
        for kw, w in enumerate(windows):
            s = 0.0
            for (i,j) in w.pairs_all:
                s += Y[i,j]
            TVs_via_Y[kw] = w.scale * s
        print(f"\n[11] At LP primal mu_lp: max TV = {TVs_lp.max():.6f} (vs LP z = {z:.6f})")
        print(f"     LP-Y-implied max TV = {TVs_via_Y.max():.6f}")
        print(f"     z - max_TV(mu_lp) = {z - TVs_lp.max():.6e}  (this is the McCormick lift slack)")

        # LP gap explanation: the LP returns z = max_W (scale_W * sum Y_ij over S_W).
        # If Y_ij > mu_i mu_j entry-wise on S_W, then z over-estimates TV; LP wants Y small
        # to minimize z, so it pushes Y down to McCormick LB. The bias of LB versus mu*mu^T
        # tells us the gap source.
        # Specifically: max_W TV_W(mu_lp) − z   should be ≤ 0  always  (Y >= conv-LB makes z large).
        # If Y < mu*mu^T, then z (computed from Y) UNDER-estimates true TV; that's the gap.

        # In which window is Y most-deficient?
        for kw, w in enumerate(windows):
            if kw == np.argmax(TVs_via_Y):
                print(f"\n  Binding window (max TV via Y): W{kw} (ell={w.ell}, s={w.s_lo}, scale={w.scale:.3f})")
                # Sum of Y over S_W vs sum of mu_lp ⊗ mu_lp over S_W
                sY = sum(Y[i,j] for (i,j) in w.pairs_all)
                sM = sum(mu_lp[i]*mu_lp[j] for (i,j) in w.pairs_all)
                print(f"  sum Y = {sY:.6f}, sum mu*mu^T = {sM:.6f}, Y-deficit = {sY - sM:.6e}")
                # Per-pair deficits
                deficits = []
                for (i,j) in w.pairs_all:
                    deficits.append((Y[i,j] - mu_lp[i]*mu_lp[j], i, j))
                deficits.sort()
                print(f"  Top-5 most deficient pairs (i,j): {deficits[:5]}")
                break


def _solve_epigraph_lp_with_primal(lo, hi, windows, d):
    """Same LP but returning the full primal solution vector."""
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, coo_matrix
    from interval_bnb.bound_epigraph import _cache_lp_structure
    n_y = d * d; n_mu = d; n_W = len(windows)
    n_vars = n_y + n_mu + 1; z_idx = n_y + n_mu
    pair_i, pair_j, rows_w, cols_w, scales_w = _cache_lp_structure(windows, d)
    lo = np.asarray(lo, dtype=np.float64); hi = np.asarray(hi, dtype=np.float64)

    n_pairs = n_y
    sw_rows = np.empty(3*n_pairs, dtype=np.int64); sw_cols = np.empty(3*n_pairs, dtype=np.int64); sw_data = np.empty(3*n_pairs, dtype=np.float64)
    sw_rows[:n_pairs] = np.arange(n_pairs); sw_cols[:n_pairs] = np.arange(n_pairs); sw_data[:n_pairs] = -1.0
    sw_rows[n_pairs:2*n_pairs] = np.arange(n_pairs); sw_cols[n_pairs:2*n_pairs] = n_y + pair_i; sw_data[n_pairs:2*n_pairs] = lo[pair_j]
    sw_rows[2*n_pairs:3*n_pairs] = np.arange(n_pairs); sw_cols[2*n_pairs:3*n_pairs] = n_y + pair_j; sw_data[2*n_pairs:3*n_pairs] = lo[pair_i]
    ne_rows = np.empty(3*n_pairs, dtype=np.int64); ne_cols = np.empty(3*n_pairs, dtype=np.int64); ne_data = np.empty(3*n_pairs, dtype=np.float64)
    ne_rows[:n_pairs] = n_pairs+np.arange(n_pairs); ne_cols[:n_pairs] = np.arange(n_pairs); ne_data[:n_pairs] = -1.0
    ne_rows[n_pairs:2*n_pairs] = n_pairs+np.arange(n_pairs); ne_cols[n_pairs:2*n_pairs] = n_y + pair_i; ne_data[n_pairs:2*n_pairs] = hi[pair_j]
    ne_rows[2*n_pairs:3*n_pairs] = n_pairs+np.arange(n_pairs); ne_cols[2*n_pairs:3*n_pairs] = n_y + pair_j; ne_data[2*n_pairs:3*n_pairs] = hi[pair_i]
    nw_rows = np.empty(3*n_pairs, dtype=np.int64); nw_cols = np.empty(3*n_pairs, dtype=np.int64); nw_data = np.empty(3*n_pairs, dtype=np.float64)
    nw_rows[:n_pairs] = 2*n_pairs+np.arange(n_pairs); nw_cols[:n_pairs] = np.arange(n_pairs); nw_data[:n_pairs] = +1.0
    nw_rows[n_pairs:2*n_pairs] = 2*n_pairs+np.arange(n_pairs); nw_cols[n_pairs:2*n_pairs] = n_y + pair_i; nw_data[n_pairs:2*n_pairs] = -lo[pair_j]
    nw_rows[2*n_pairs:3*n_pairs] = 2*n_pairs+np.arange(n_pairs); nw_cols[2*n_pairs:3*n_pairs] = n_y + pair_j; nw_data[2*n_pairs:3*n_pairs] = -hi[pair_i]
    se_rows = np.empty(3*n_pairs, dtype=np.int64); se_cols = np.empty(3*n_pairs, dtype=np.int64); se_data = np.empty(3*n_pairs, dtype=np.float64)
    se_rows[:n_pairs] = 3*n_pairs+np.arange(n_pairs); se_cols[:n_pairs] = np.arange(n_pairs); se_data[:n_pairs] = +1.0
    se_rows[n_pairs:2*n_pairs] = 3*n_pairs+np.arange(n_pairs); se_cols[n_pairs:2*n_pairs] = n_y + pair_i; se_data[n_pairs:2*n_pairs] = -hi[pair_j]
    se_rows[2*n_pairs:3*n_pairs] = 3*n_pairs+np.arange(n_pairs); se_cols[2*n_pairs:3*n_pairs] = n_y + pair_j; se_data[2*n_pairs:3*n_pairs] = -lo[pair_i]
    n_epi = len(rows_w)
    epi_rows = np.empty(n_epi+n_W, dtype=np.int64); epi_cols = np.empty(n_epi+n_W, dtype=np.int64); epi_data = np.empty(n_epi+n_W, dtype=np.float64)
    epi_rows[:n_epi] = 4*n_pairs + rows_w; epi_cols[:n_epi] = cols_w; epi_data[:n_epi] = scales_w
    epi_rows[n_epi:] = 4*n_pairs + np.arange(n_W); epi_cols[n_epi:] = z_idx; epi_data[n_epi:] = -1.0
    rows_all = np.concatenate([sw_rows, ne_rows, nw_rows, se_rows, epi_rows])
    cols_all = np.concatenate([sw_cols, ne_cols, nw_cols, se_cols, epi_cols])
    data_all = np.concatenate([sw_data, ne_data, nw_data, se_data, epi_data])
    n_ineq = 4*n_pairs + n_W
    A_ub = coo_matrix((data_all,(rows_all,cols_all)), shape=(n_ineq, n_vars)).tocsr()
    b_ub = np.empty(n_ineq, dtype=np.float64)
    b_ub[:n_pairs] = lo[pair_i]*lo[pair_j]; b_ub[n_pairs:2*n_pairs] = hi[pair_i]*hi[pair_j]
    b_ub[2*n_pairs:3*n_pairs] = -lo[pair_j]*hi[pair_i]; b_ub[3*n_pairs:4*n_pairs] = -hi[pair_j]*lo[pair_i]
    b_ub[4*n_pairs:] = 0.0
    eq_rows = []; eq_cols = []; eq_data = []
    eq_rows.extend([0]*d); eq_cols.extend([n_y+i for i in range(d)]); eq_data.extend([1.0]*d)
    for i in range(d):
        for j in range(d):
            eq_rows.append(1+i); eq_cols.append(i*d+j); eq_data.append(1.0)
        eq_rows.append(1+i); eq_cols.append(n_y+i); eq_data.append(-1.0)
    A_eq = csr_matrix((np.asarray(eq_data),(np.asarray(eq_rows),np.asarray(eq_cols))), shape=(1+d, n_vars))
    b_eq = np.zeros(1+d); b_eq[0] = 1.0
    bnds = [(0.0, None)] * n_y + [(float(lo[i]), float(hi[i])) for i in range(d)] + [(0.0, None)]
    c = np.zeros(n_vars); c[z_idx] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bnds, method="highs")
    if not res.success: return float("-inf"), None
    return float(res.fun), res.x


if __name__ == "__main__":
    main()
