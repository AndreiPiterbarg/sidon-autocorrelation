"""Debug: extract pseudomoment values from solved SDP at d=4."""
import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, '../../')
import numpy as np
import cvxpy as cp
from lasserre.core import build_window_matrices
from probe import enum_monos, add_alpha

d = 4
windows, M_mats = build_window_matrices(d)

level = 2
max_deg = 2*level
monos_all = enum_monos(d, max_deg)
alpha_to_idx = {a: i for i, a in enumerate(monos_all)}
n_y = len(monos_all)
m = cp.Variable(n_y)

cons = [m[alpha_to_idx[tuple([0]*d)]] == 1]

basis_k = enum_monos(d, level)
Bk = len(basis_k)
Mk_rows = []
for i in range(Bk):
    row = []
    for j in range(Bk):
        a = add_alpha(basis_k[i], basis_k[j])
        row.append(m[alpha_to_idx[a]])
    Mk_rows.append(row)
Mk = cp.bmat(Mk_rows)
cons += [Mk >> 0]

# Equality: 1 - sum y_i^2 = 0
basis_lm1 = enum_monos(d, level - 1)
seen = set()
for ai in basis_lm1:
    for bi in basis_lm1:
        gamma = add_alpha(ai, bi)
        if gamma in seen: continue
        seen.add(gamma)
        if sum(gamma) + 2 > max_deg: continue
        terms = []
        for k in range(d):
            a2 = list(gamma); a2[k] += 2
            terms.append(m[alpha_to_idx[tuple(a2)]])
        cons += [m[alpha_to_idx[gamma]] - cp.sum(terms) == 0]

t = cp.Variable()
for M_W in M_mats:
    terms = []
    for i in range(d):
        for j in range(d):
            if M_W[i,j] == 0: continue
            a = [0]*d; a[i] += 2; a[j] += 2
            a_t = tuple(a)
            if a_t not in alpha_to_idx: continue
            terms.append(float(M_W[i,j]) * m[alpha_to_idx[a_t]])
    if terms:
        cons += [t >= cp.sum(terms)]

prob = cp.Problem(cp.Minimize(t), cons)
prob.solve(solver='MOSEK', verbose=False)
print(f'status={prob.status}, t={t.value:.6f}')

# Extract moment values
for alpha in [(0,0,0,0), (2,0,0,0), (0,2,0,0), (0,0,2,0), (0,0,0,2),
              (4,0,0,0), (0,4,0,0), (2,2,0,0), (1,1,0,0), (3,1,0,0), (1,3,0,0),
              (1,1,1,1), (2,1,1,0)]:
    idx = alpha_to_idx[alpha]
    print(f'  m_{alpha} = {m.value[idx]:.6f}')

# Check constraint sum_i m_{2 e_i} = 1
s = sum(m.value[alpha_to_idx[tuple(2 if i==k else 0 for i in range(d))]] for k in range(d))
print(f'sum m_{{2 e_i}} = {s:.6f}')
# 4 e_0
print(f'm_{{4 e_0}} = {m.value[alpha_to_idx[(4,0,0,0)]]:.6f}')

# Check first window's constraint
M_W = M_mats[0]
print(f'\nWindow 0 = {windows[0]}: M_W = ')
print(M_W)
# Compute sum_{ij} M_W[i,j] m_{2 e_i + 2 e_j}
s = 0.0
for i in range(d):
    for j in range(d):
        if M_W[i,j] != 0:
            a = [0]*d; a[i]+=2; a[j]+=2
            v = m.value[alpha_to_idx[tuple(a)]]
            s += M_W[i,j] * v
            print(f'  M_W[{i},{j}]={M_W[i,j]}, m_{tuple(a)}={v:.4f}, contrib={M_W[i,j]*v:.4f}')
print(f'  total = {s:.6f}, should be <= t = {t.value:.6f}')
