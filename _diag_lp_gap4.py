"""Final test: do the RLT-2 / Shor-PSD additions close the gap on 5-wide-axes box?

We test:
  (a) RLT-symmetry Y_ij = Y_ji
  (b) Anchor cut μ - mu_star (translate)
  (c) Shor-PSD: [1 mu^T; mu Y] >= 0

We add (a)–(c) and re-solve, comparing to the baseline LP.

Hypothesis from prior tests: the gap of ~1e-2 in the 5-wide-axis case
arises because multiple Y_{i,j} variables can be set INDEPENDENTLY at the
intersection of their per-pair McCormick cells. Adding cross-pair PSD or
explicit RLT cuts forces them onto a common mu.
"""
import os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from interval_bnb.windows import build_windows
from interval_bnb.bound_epigraph import _solve_epigraph_lp, _cache_lp_structure
from _diag_lp_gap import _solve_epigraph_lp_with_primal


def solve_with_extras(lo, hi, windows, d, *, add_symmetry=False, add_shor=False):
    """Solve epigraph LP with optional extras.

    add_symmetry: enforce Y_ij = Y_ji (cuts var count basically)
    add_shor: add a CONIC PSD constraint [1 mu^T; mu Y] >= 0 (cvxpy)
    """
    if add_shor:
        return _solve_with_shor(lo, hi, windows, d, add_symmetry)

    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, coo_matrix
    n_y = d*d; n_mu = d; n_W = len(windows)
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
    eq_count = 1 + d
    if add_symmetry:
        # Y_ij = Y_ji => Y_ij - Y_ji = 0 for i < j
        for i in range(d):
            for j in range(i+1, d):
                eq_rows.append(eq_count); eq_cols.append(i*d + j); eq_data.append(1.0)
                eq_rows.append(eq_count); eq_cols.append(j*d + i); eq_data.append(-1.0)
                eq_count += 1
    A_eq = csr_matrix((np.asarray(eq_data),(np.asarray(eq_rows),np.asarray(eq_cols))), shape=(eq_count, n_vars))
    b_eq = np.zeros(eq_count); b_eq[0] = 1.0
    bnds = [(0.0, None)] * n_y + [(float(lo[i]), float(hi[i])) for i in range(d)] + [(0.0, None)]
    c = np.zeros(n_vars); c[z_idx] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bnds, method="highs")
    if not res.success: return float("-inf"), None
    return float(res.fun), res.x


def _solve_with_shor(lo, hi, windows, d, add_symmetry):
    """SDP version: add [1 mu^T; mu Y] >= 0 (Shor)."""
    import cvxpy as cp
    n_y = d*d; n_W = len(windows)
    Y = cp.Variable((d, d), symmetric=add_symmetry)  # if asymmetric, omit symmetric
    if not add_symmetry:
        Y = cp.Variable((d, d))
    mu = cp.Variable(d)
    z = cp.Variable()
    constraints = [
        Y >= 0,
        mu >= lo, mu <= hi,
        cp.sum(mu) == 1,
        z >= 0,
    ]
    # Row-sum
    for i in range(d):
        constraints.append(cp.sum(Y[i, :]) == mu[i])
    # McCormick
    for i in range(d):
        for j in range(d):
            constraints.append(Y[i,j] >= lo[j]*mu[i] + lo[i]*mu[j] - lo[i]*lo[j])
            constraints.append(Y[i,j] >= hi[j]*mu[i] + hi[i]*mu[j] - hi[i]*hi[j])
            constraints.append(Y[i,j] <= lo[j]*mu[i] + hi[i]*mu[j] - lo[j]*hi[i])
            constraints.append(Y[i,j] <= hi[j]*mu[i] + lo[i]*mu[j] - hi[j]*lo[i])
    # Shor PSD
    M = cp.bmat([[cp.reshape(cp.Constant(1.0), (1,1)), cp.reshape(mu, (1, d))],
                  [cp.reshape(mu, (d, 1)), Y]])
    constraints.append(M >> 0)
    # Epigraph
    for kw, w in enumerate(windows):
        s = sum(Y[i,j] for (i,j) in w.pairs_all)
        constraints.append(z >= w.scale * s)
    prob = cp.Problem(cp.Minimize(z), constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False)
        return float(prob.value), None
    except Exception as e:
        print(f"  SDP solve failed: {e}")
        return float("-inf"), None


def main():
    d = 20
    print("Building data...")
    from kkt_correct_mu_star import build_window_data
    A_stack, c_W = build_window_data(d)
    windows = build_windows(d)

    from _diag_lp_gap import find_mu_star_quick
    print("Finding mu_star...")
    mu_star, val_star, _, _ = find_mu_star_quick(d, n_starts=20, n_iters=2000)
    print(f"  val_star ≈ {val_star:.4f}")

    rng = np.random.RandomState(0)

    print("\n=== 5-wide-axis sliver boxes: compare LP variants ===")
    print(f"{'wide_w':>10s} {'baseline':>10s} {'+sym':>10s} {'+shor':>10s} {'true':>10s}")
    for wide_w in [1e-2, 1e-3, 1e-4]:
        half = np.full(d, 5e-10)
        wide_axes = rng.choice(d, 5, replace=False)
        half[wide_axes] = wide_w / 2
        lo = np.maximum(mu_star - half, 0.0)
        hi = np.minimum(mu_star + half, 1.0)
        # baseline
        v_base, _ = solve_with_extras(lo, hi, windows, d)
        # + symmetry
        v_sym, _ = solve_with_extras(lo, hi, windows, d, add_symmetry=True)
        # + shor (skip if cvxpy not available)
        try:
            v_shor, _ = solve_with_extras(lo, hi, windows, d, add_shor=True)
        except Exception as e:
            v_shor = float('nan'); print(f"  shor failed: {e}")
        # true min: use mu_star itself (in box by construction)
        Amu = A_stack @ mu_star
        true_at_star = (c_W * (Amu * mu_star).sum(axis=1)).max()
        print(f"{wide_w:>10.2e} {v_base:>10.5f} {v_sym:>10.5f} {v_shor:>10.5f} {true_at_star:>10.5f}")


if __name__ == "__main__":
    main()
