"""Test multiple methods for tightening per-cell bounds at d=2 borderline cells.

Methods:
  1. Order-1 Shor SDP (baseline, _shor_feasibility from _L_bench.py)
  2. Order-2 Lasserre SDP (_lasserre2_feasibility from _L_bench.py)
  3. Order-3 Lasserre (manual: monomials up to deg 6, M_3 PSD lift)
  4. Sherali-Adams hierarchy (RLT level-2: products of (x_i-lo)(hi-x_i) etc.)
  5. Direct numerical search (HEURISTIC): scipy.optimize.minimize_scalar over v
     for each cell; report exact min(max_W TV_W).

The TRUE val(cell) is computed exactly via fine grid (1D, easy), giving
ground truth for soundness validation.

Wall budget: < 30 minutes total.
"""
import sys, os, time, json
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

import numpy as np
from itertools import combinations_with_replacement
from _Q_bench import _build_windows
from _M1_bench import prune_F
from _L_bench import _build_A_matrices, _shor_feasibility, _lasserre2_feasibility, _make_cell

n_half = 1
m = 20
c_target = 1.281
d = 2
S = 4 * n_half * m  # = 80

windows, ell_int_sums = _build_windows(d)
A_mats = _build_A_matrices(d, windows)

# Get F-survivor canonical cells
all_comps = [[k, S - k] for k in range(0, S + 1)]
batch = np.array(all_comps, dtype=np.int32)
sF = prune_F(batch, n_half, m, c_target)
canonical = [c for c, s in zip(all_comps, sF) if s and c[0] <= c[1]]
# Keep "25 L0 cells" - exclude (40,40) duplicate which represents only itself
test_cells = [c for c in canonical if c[0] != c[1]]  # 25 cells
print(f"Testing on {len(test_cells)} L0 cells: {test_cells[0]} to {test_cells[-1]}")
print(f"  c_target = {c_target}, n_half = {n_half}, m = {m}, d = {d}\n")


# ===================================================================
# Truth: TRUE val(cell) via 1D grid (d=2 has only 1 DOF: x_0=v, x_1=80-v)
# ===================================================================
def true_val(c):
    """min over v in [c0-1, c0+1] of max_W m^2*TV_W(x), normalized by m^2."""
    c0 = c[0]
    v_lo = max(0.0, c0 - 1.0)
    v_hi = c0 + 1.0
    vs = np.linspace(v_lo, v_hi, 100001)
    best_val = np.inf
    best_v = None
    for v in vs:
        x = np.array([v, S - v])
        max_w = -np.inf
        for A_mat, (ell, _) in zip(A_mats, windows):
            val = float(x @ A_mat @ x) / (4.0 * n_half * ell)
            if val > max_w:
                max_w = val
        if max_w < best_val:
            best_val = max_w
            best_v = v
    return best_val / (m * m), best_v


# ===================================================================
# Method 1: Order-1 Shor SDP
# ===================================================================
def method_shor(c_int):
    lo, hi = _make_cell(c_int, m)
    t0 = time.time()
    pruned, status = _shor_feasibility(c_int, lo, hi, A_mats, windows,
                                         n_half, m, c_target,
                                         solver='MOSEK', tol=1e-9,
                                         eps_margin=1e-9)
    return pruned, status, time.time() - t0


# ===================================================================
# Method 2: Order-2 Lasserre
# ===================================================================
def method_lasserre2(c_int):
    lo, hi = _make_cell(c_int, m)
    t0 = time.time()
    pruned, status = _lasserre2_feasibility(c_int, lo, hi, windows,
                                              n_half, m, c_target,
                                              solver='MOSEK', tol=1e-9,
                                              eps_margin=1e-9)
    return pruned, status, time.time() - t0


# ===================================================================
# Method 3: Order-3 Lasserre (degree 6 monomials, M_3 size = C(d+3, 3))
# ===================================================================
def _build_monomials(d, deg):
    out = [()]
    for k in range(1, deg + 1):
        for comb in combinations_with_replacement(range(d), k):
            out.append(tuple(comb))
    return out


def _alpha_of(mon, d):
    a = [0] * d
    for v in mon:
        a[v] += 1
    return tuple(a)


def method_lasserre_order(c_int, order):
    """Order-r Lasserre feasibility: M_r PSD + r-1 localizing M_{r-1}."""
    import cvxpy as cp

    lo, hi = _make_cell(c_int, m)
    max_deg = 2 * order
    monos = _build_monomials(d, max_deg)
    alpha_to_idx = {}
    for mn in monos:
        a = _alpha_of(mn, d)
        if a not in alpha_to_idx:
            alpha_to_idx[a] = len(alpha_to_idx)
    n_y = len(alpha_to_idx)
    y = cp.Variable(n_y)

    basis = [mn for mn in monos if len(mn) <= order]
    alphas = [_alpha_of(mn, d) for mn in basis]
    B = len(basis)

    def add_a(a, b):
        return tuple(x + z for x, z in zip(a, b))

    M_rows = []
    for i in range(B):
        row = []
        for j in range(B):
            a = add_a(alphas[i], alphas[j])
            row.append(y[alpha_to_idx[a]])
        M_rows.append(row)
    M = cp.bmat(M_rows)

    basis_loc = [mn for mn in monos if len(mn) <= order - 1]
    alphas_loc = [_alpha_of(mn, d) for mn in basis_loc]
    B_loc = len(basis_loc)

    def loc_low(k, lo_k):
        e_k = [0] * d
        e_k[k] = 1
        e_k = tuple(e_k)
        rows = []
        for i in range(B_loc):
            rr = []
            for j in range(B_loc):
                base = add_a(alphas_loc[i], alphas_loc[j])
                a_plus = add_a(base, e_k)
                if sum(a_plus) > max_deg:
                    rr.append(cp.Constant(0.0))
                    continue
                rr.append(y[alpha_to_idx[a_plus]] - lo_k * y[alpha_to_idx[base]])
            rows.append(rr)
        return cp.bmat(rows)

    def loc_high(k, hi_k):
        e_k = [0] * d
        e_k[k] = 1
        e_k = tuple(e_k)
        rows = []
        for i in range(B_loc):
            rr = []
            for j in range(B_loc):
                base = add_a(alphas_loc[i], alphas_loc[j])
                a_plus = add_a(base, e_k)
                if sum(a_plus) > max_deg:
                    rr.append(cp.Constant(0.0))
                    continue
                rr.append(hi_k * y[alpha_to_idx[base]] - y[alpha_to_idx[a_plus]])
            rows.append(rr)
        return cp.bmat(rows)

    cons = [y[alpha_to_idx[(0,) * d]] == 1.0]
    cons += [y >= 0]
    cons += [M >> 0]

    nm = float(4 * n_half * m)
    e_idx = []
    for i in range(d):
        e = [0] * d
        e[i] = 1
        e_idx.append(alpha_to_idx[tuple(e)])
    cons += [cp.sum([y[k] for k in e_idx]) == nm]

    for k in range(d):
        cons += [loc_low(k, float(lo[k])) >> 0]
        cons += [loc_high(k, float(hi[k])) >> 0]

    cs_m2 = float(c_target) * m * m
    eps_thr = 1e-9 * m * m
    for (ell, s_lo) in windows:
        s_hi = s_lo + ell - 2
        terms = []
        for i in range(d):
            for j in range(d):
                if s_lo <= i + j <= s_hi:
                    a = [0] * d
                    a[i] += 1
                    a[j] += 1
                    terms.append(y[alpha_to_idx[tuple(a)]])
        thr = 4.0 * float(n_half) * float(ell) * (cs_m2 + eps_thr)
        cons += [cp.sum(terms) <= thr]

    prob = cp.Problem(cp.Minimize(0), cons)
    t0 = time.time()
    try:
        prob.solve(solver='MOSEK', verbose=False,
                    mosek_params={
                        'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-9,
                        'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-9,
                        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-9,
                    })
    except Exception as e:
        return False, f'EXC:{type(e).__name__}', time.time() - t0
    return prob.status == 'infeasible', prob.status, time.time() - t0


# ===================================================================
# Method 4: Sherali-Adams (RLT level-2)
# Multiply pairs of (x_i - lo_i), (hi_i - x_i), and sum constraint.
# ===================================================================
def method_sherali_adams(c_int):
    """SA level-2: introduce y_{ij} = x_i x_j with constraints derived from
    box * box products. Plus a level-3 piece: x_i x_j x_k via lifting.

    For d=2 this gives RLT cuts: (x_0-lo_0)(x_1-lo_1) >=0, (x_0-lo_0)(hi_0-x_0)>=0, etc.
    These are weaker than Lasserre (which adds PSD lift) but simpler.
    """
    import cvxpy as cp

    lo, hi = _make_cell(c_int, m)
    nm = float(4 * n_half * m)

    x = cp.Variable(d)
    X = cp.Variable((d, d), symmetric=True)
    # Level-3 var: y_ijk for i<=j<=k
    z = cp.Variable((d, d, d), symmetric=False)

    cons = []
    cons += [x >= lo, x <= hi]
    cons += [cp.sum(x) == nm]

    # Level-2 RLT cuts (without PSD)
    for i in range(d):
        cons += [X[i, i] >= 2 * lo[i] * x[i] - lo[i]**2]
        cons += [X[i, i] >= 2 * hi[i] * x[i] - hi[i]**2]
        cons += [X[i, i] <= (lo[i] + hi[i]) * x[i] - lo[i] * hi[i]]
        for j in range(i + 1, d):
            li, lj = lo[i], lo[j]
            ui, uj = hi[i], hi[j]
            cons += [X[i, j] >= lj * x[i] + li * x[j] - li * lj]
            cons += [X[i, j] >= uj * x[i] + ui * x[j] - ui * uj]
            cons += [X[i, j] <= ui * x[j] + lj * x[i] - ui * lj]
            cons += [X[i, j] <= uj * x[i] + li * x[j] - li * uj]

    # Sum constraint product: (sum x = nm) * (x_i - lo_i) >= 0 etc.
    # Sum_j X[j,i] = nm * x_i (this is a strong RLT cut)
    for i in range(d):
        cons += [cp.sum([X[j, i] for j in range(d)]) == nm * x[i]]

    # Window constraints
    cs_m2 = float(c_target) * m * m
    eps_thr = 1e-9 * m * m
    for A_mat, (ell, _) in zip(A_mats, windows):
        thr = 4.0 * float(n_half) * float(ell) * (cs_m2 + eps_thr)
        cons += [cp.trace(A_mat @ X) <= thr]

    # Level-3 RLT (a few key cuts)
    # (x_i - lo_i) * X[j,k] >= 0 etc.
    for i in range(d):
        for j in range(d):
            for k in range(d):
                # z[i,j,k] = x_i x_j x_k
                # symmetry: z[i,j,k] is symmetric in indices
                pass  # skipped for simplicity

    prob = cp.Problem(cp.Minimize(0), cons)
    t0 = time.time()
    try:
        prob.solve(solver='MOSEK', verbose=False)
    except Exception as e:
        return False, f'EXC:{type(e).__name__}', time.time() - t0
    return prob.status == 'infeasible', prob.status, time.time() - t0


# ===================================================================
# Method 5: Direct numerical search (HEURISTIC)
# ===================================================================
def method_direct(c_int):
    """1D scan of v in [c0-1, c0+1]; min over max_W m^2*TV_W.

    HEURISTIC: provides a witness if min < threshold (cell unprunable),
    or upper bound on val(cell). NOT a sound rigorous proof of pruning.
    """
    c0 = c_int[0]
    v_lo = max(0.0, c0 - 1.0)
    v_hi = c0 + 1.0
    n_grid = 10001
    vs = np.linspace(v_lo, v_hi, n_grid)
    t0 = time.time()
    best_val = np.inf
    best_v = None
    for v in vs:
        x = np.array([v, S - v])
        max_w = -np.inf
        for A_mat, (ell, _) in zip(A_mats, windows):
            val = float(x @ A_mat @ x) / (4.0 * n_half * ell)
            if val > max_w:
                max_w = val
        if max_w < best_val:
            best_val = max_w
            best_v = v
    elapsed = time.time() - t0
    val_normalized = best_val / (m * m)
    # Heuristic prunability: only if val_normalized > c_target
    return val_normalized > c_target, val_normalized, elapsed


# ===================================================================
# Run all methods
# ===================================================================
results = {
    'cells': [c for c in test_cells],
    'truth': [],
    'shor': {'pruned': 0, 'times': [], 'statuses': []},
    'lasserre2': {'pruned': 0, 'times': [], 'statuses': []},
    'lasserre3': {'pruned': 0, 'times': [], 'statuses': []},
    'sherali_adams': {'pruned': 0, 'times': [], 'statuses': []},
    'direct': {'pruned': 0, 'vals': [], 'times': []},
}

# Truth pre-computation
print("Computing TRUE val(cell) for each cell (1D grid sweep)...")
for c in test_cells:
    val, v_star = true_val(c)
    results['truth'].append({'c': c, 'val': float(val), 'v_star': float(v_star)})
    print(f"  c={c}: TRUE val(cell) = {val:.6f}, v*={v_star:.4f}, "
          f"genuinely_prunable={'YES' if val > c_target else 'NO'}")

n_truth_prunable = sum(1 for r in results['truth'] if r['val'] > c_target)
print(f"\n*** {n_truth_prunable} of {len(test_cells)} cells genuinely prunable (val > 1.281) ***\n")


print("=" * 70)
print("Method 1: Order-1 Shor SDP")
print("=" * 70)
for c in test_cells:
    pruned, status, t = method_shor(np.array(c, dtype=np.int32))
    results['shor']['times'].append(t)
    results['shor']['statuses'].append(status)
    if pruned:
        results['shor']['pruned'] += 1
        print(f"  c={c}: PRUNED ({status}) in {t*1000:.1f} ms")
    else:
        print(f"  c={c}: not pruned ({status}) in {t*1000:.1f} ms")
print(f"  TOTAL: {results['shor']['pruned']} of {len(test_cells)} pruned")
print(f"  Wall: {sum(results['shor']['times']):.2f}s, "
      f"med {1000*np.median(results['shor']['times']):.1f} ms/cell\n")

print("=" * 70)
print("Method 2: Order-2 Lasserre SDP")
print("=" * 70)
for c in test_cells:
    pruned, status, t = method_lasserre2(np.array(c, dtype=np.int32))
    results['lasserre2']['times'].append(t)
    results['lasserre2']['statuses'].append(status)
    if pruned:
        results['lasserre2']['pruned'] += 1
        print(f"  c={c}: PRUNED ({status}) in {t*1000:.1f} ms")
    else:
        print(f"  c={c}: not pruned ({status}) in {t*1000:.1f} ms")
print(f"  TOTAL: {results['lasserre2']['pruned']} of {len(test_cells)} pruned")
print(f"  Wall: {sum(results['lasserre2']['times']):.2f}s, "
      f"med {1000*np.median(results['lasserre2']['times']):.1f} ms/cell\n")

print("=" * 70)
print("Method 3: Order-3 Lasserre SDP")
print("=" * 70)
for c in test_cells:
    pruned, status, t = method_lasserre_order(np.array(c, dtype=np.int32), order=3)
    results['lasserre3']['times'].append(t)
    results['lasserre3']['statuses'].append(status)
    if pruned:
        results['lasserre3']['pruned'] += 1
        print(f"  c={c}: PRUNED ({status}) in {t*1000:.1f} ms")
    else:
        print(f"  c={c}: not pruned ({status}) in {t*1000:.1f} ms")
print(f"  TOTAL: {results['lasserre3']['pruned']} of {len(test_cells)} pruned")
print(f"  Wall: {sum(results['lasserre3']['times']):.2f}s, "
      f"med {1000*np.median(results['lasserre3']['times']):.1f} ms/cell\n")

print("=" * 70)
print("Method 4: Sherali-Adams")
print("=" * 70)
for c in test_cells:
    pruned, status, t = method_sherali_adams(np.array(c, dtype=np.int32))
    results['sherali_adams']['times'].append(t)
    results['sherali_adams']['statuses'].append(status)
    if pruned:
        results['sherali_adams']['pruned'] += 1
        print(f"  c={c}: PRUNED ({status}) in {t*1000:.1f} ms")
    else:
        print(f"  c={c}: not pruned ({status}) in {t*1000:.1f} ms")
print(f"  TOTAL: {results['sherali_adams']['pruned']} of {len(test_cells)} pruned")
print(f"  Wall: {sum(results['sherali_adams']['times']):.2f}s, "
      f"med {1000*np.median(results['sherali_adams']['times']):.1f} ms/cell\n")

print("=" * 70)
print("Method 5: Direct numerical search (HEURISTIC)")
print("=" * 70)
for c in test_cells:
    pruned, val, t = method_direct(np.array(c, dtype=np.int32))
    results['direct']['times'].append(t)
    results['direct']['vals'].append(val)
    if pruned:
        results['direct']['pruned'] += 1
        print(f"  c={c}: PRUNED-heuristic (val={val:.6f}) in {t*1000:.1f} ms")
    else:
        print(f"  c={c}: not pruned (val={val:.6f}) in {t*1000:.1f} ms")
print(f"  TOTAL: {results['direct']['pruned']} of {len(test_cells)} pruned (HEURISTIC)")
print(f"  Wall: {sum(results['direct']['times']):.2f}s, "
      f"med {1000*np.median(results['direct']['times']):.1f} ms/cell\n")

# Save
with open('_borderline_cells_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Cells tested: {len(test_cells)} (L0 cells at n_half=1, m=20, c_target=1.281)")
print(f"Genuinely prunable (truth val > 1.281): {n_truth_prunable}")
print(f"  Method 1 (Shor SDP, sound):       {results['shor']['pruned']}")
print(f"  Method 2 (Lasserre order-2, sound): {results['lasserre2']['pruned']}")
print(f"  Method 3 (Lasserre order-3, sound): {results['lasserre3']['pruned']}")
print(f"  Method 4 (Sherali-Adams, sound):   {results['sherali_adams']['pruned']}")
print(f"  Method 5 (Direct, heuristic):     {results['direct']['pruned']}")
print()
