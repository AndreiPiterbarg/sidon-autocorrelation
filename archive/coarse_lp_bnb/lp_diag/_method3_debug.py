"""Debug Method 3: why does it claim infeasible for c=[21,59] which has truth val=1.232?

If the order-3 Lasserre is sound, it CANNOT return infeasible for a cell where
a feasible point exists. So either the cvxpy reformulation has a bug, or there's
a numerical issue.
"""
import sys, os
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

import numpy as np
import cvxpy as cp
from itertools import combinations_with_replacement
from _Q_bench import _build_windows
from _L_bench import _build_A_matrices, _make_cell

n_half = 1
m = 20
c_target = 1.281
d = 2
S = 80

windows, ell_int_sums = _build_windows(d)
A_mats = _build_A_matrices(d, windows)
c_int = np.array([21, 59], dtype=np.int32)
lo, hi = _make_cell(c_int, m)
print(f"c={c_int.tolist()}, lo={lo}, hi={hi}")

# Verify there IS a feasible point: x=(22, 58), x sum=80
x_test = np.array([22.0, 58.0])
print(f"x_test = {x_test}")
print(f"  Within box? {(lo <= x_test).all() and (x_test <= hi).all()}")
print(f"  Sum = {x_test.sum()} (should be 80)")
for A_mat, (ell, s_lo) in zip(A_mats, windows):
    val = float(x_test @ A_mat @ x_test)
    thr = 4.0 * n_half * ell * (c_target * m * m)
    print(f"  ell={ell}, s_lo={s_lo}: x^T A x = {val}, thr = {thr}, OK = {val <= thr}")


# Check truth
print()
print("Method 3 in detail (manual reproduction):")

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


order = 3
max_deg = 2 * order  # = 6
monos = _build_monomials(d, max_deg)
alpha_to_idx = {}
for mn in monos:
    a = _alpha_of(mn, d)
    if a not in alpha_to_idx:
        alpha_to_idx[a] = len(alpha_to_idx)
n_y = len(alpha_to_idx)
print(f"  n_y (number of distinct moments) = {n_y}")

# This is order-3 Lasserre. M_3 size = C(d+order, order) = C(2+3, 3) = 10
basis = [mn for mn in monos if len(mn) <= order]
alphas = [_alpha_of(mn, d) for mn in basis]
print(f"  M_3 basis size = {len(basis)}")
B = len(basis)
y = cp.Variable(n_y)

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

basis_loc = [mn for mn in monos if len(mn) <= order - 1]  # deg <= 2
alphas_loc = [_alpha_of(mn, d) for mn in basis_loc]
B_loc = len(basis_loc)
print(f"  Localizing basis size = {B_loc}")

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

# Try with verbose to see what MOSEK says
print("\n  Trying solve with verbose=True:")
try:
    prob.solve(solver='MOSEK', verbose=True)
    print(f"  Status: {prob.status}")
except Exception as e:
    print(f"  Exception: {e}")

print(f"\nFinal status: {prob.status}")

# Inject a known feasible point and check if PSD constraint matrix is satisfied
# x = (22, 58) gives moments y_alpha = x_0^a0 * x_1^a1
print("\n--- Inject witness x=(22,58); check feasibility ---")
x_w = np.array([22.0, 58.0])
y_witness = np.zeros(n_y)
for alpha, idx in alpha_to_idx.items():
    y_witness[idx] = (x_w[0] ** alpha[0]) * (x_w[1] ** alpha[1])

# Check PSD-ness of M(y) at this y
M_val = np.zeros((B, B))
for i in range(B):
    for j in range(B):
        a = add_a(alphas[i], alphas[j])
        M_val[i, j] = y_witness[alpha_to_idx[a]]

eigs = np.linalg.eigvalsh(M_val)
print(f"  M(y_witness) eigenvalues range: [{eigs.min():.3e}, {eigs.max():.3e}]")
print(f"  Smallest eig: {eigs.min():.3e} (should be >= 0 for moment-of-Dirac)")

# Check window constraints
print("\n  Window constraints check:")
for (ell, s_lo) in windows:
    s_hi = s_lo + ell - 2
    val = 0
    for i in range(d):
        for j in range(d):
            if s_lo <= i + j <= s_hi:
                a = [0] * d
                a[i] += 1
                a[j] += 1
                val += y_witness[alpha_to_idx[tuple(a)]]
    thr = 4.0 * n_half * ell * (cs_m2 + eps_thr)
    print(f"    ell={ell}, s_lo={s_lo}: val={val}, thr={thr}, OK={val <= thr}")
