"""Independent reproduction of the matlab boundToBeat bug.

We verify the discretization_error_proof.md counterexample:
  d=4, m=10, c=(2,1,1,6), window (ell=2, s_0=6)
  At mu_3 = 0.51, the actual error TV_disc - TV_cont = 0.3996,
  while matlab's correction (Formula B) gives 0.13.

This means matlab over-prunes: it claims threshold of c + 0.13 suffices,
but actually the gap can be as large as ~0.4, so a composition can be
pruned by matlab even when the continuous TV_cont is below c_target.
"""
import numpy as np

# Parameters
n = 2
d = 2 * n
m = 10
c = np.array([2, 1, 1, 6], dtype=np.float64)
gridSpace = 1.0 / m

# Pre-image
# c=(2,1,1,6) requires: M(1) in [0.2, 0.3), M(2) in [0.3, 0.4), M(3) in [0.4, 0.5)
# Pick mu = (0.2, 0.1, 0.19, 0.51)
mu = np.array([0.20, 0.10, 0.19, 0.51])
print(f'mu = {mu}, sum = {mu.sum()}')
# Verify cumulative-floor maps mu to c
M = np.zeros(d + 1)
for i in range(d):
    M[i+1] = M[i] + mu[i]
D = np.floor(m * M).astype(int)
D[d] = m
c_check = np.diff(D)
print(f'cumulative-floor of mu => c = {c_check} (should be [2,1,1,6])')

# Window: ell=2, s_lo=6 means we sum sum_s sum_{i+j=s} for s in [s_lo, s_lo+ell-2] = [6,6]
# i.e. only s=6: (i,j) with i+j=6, i,j in [0,d-1]. That's only (3,3).
ell = 2
s_lo = 6
four_n = 4 * n

# TV_disc(c; ell=2, s_lo=6)
TV_disc = 0.0
for s in range(s_lo, s_lo + ell - 1):
    for i in range(d):
        j = s - i
        if 0 <= j < d:
            TV_disc += (four_n * c[i] / m) * (four_n * c[j] / m)
TV_disc /= (four_n * ell)
print(f'TV_disc(c; {ell}, {s_lo}) = {TV_disc}')

# TV_cont(f; ell=2, s_lo=6)
TV_cont = 0.0
for s in range(s_lo, s_lo + ell - 1):
    for i in range(d):
        j = s - i
        if 0 <= j < d:
            TV_cont += (four_n * mu[i]) * (four_n * mu[j])
TV_cont /= (four_n * ell)
print(f'TV_cont(f; {ell}, {s_lo}) = {TV_cont}')

actual_err = TV_disc - TV_cont
print(f'Actual error: TV_disc - TV_cont = {actual_err}')
print()

# Matlab Formula B:
# matlab boundToBeat = c_target + gridSpace^2 + 2*gridSpace*W
# where W = sum of contributing bin masses (matrix_tmp · binsContribute)
# Contributing bins for (i,j)=(3,3) at s=6: just bin 3.
W_matlab = mu[3]  # the matlab code sees mu_3 as the unique contributing bin mass
matlab_correction = gridSpace**2 + 2 * gridSpace * W_matlab
print(f'Matlab correction (Formula B): eps^2 + 2*eps*W')
print(f'  = (1/{m})^2 + 2*(1/{m})*{W_matlab} = {matlab_correction}')

# Paper Formula A:
# Δ = (4n/ell) * (2 W_mu / m + 1/m^2)
W_mu = mu[3]
paper_correction = (four_n / ell) * (2 * W_mu / m + 1 / m**2)
print(f'Paper correction (Formula A): (4n/ell)*(2*W_mu/m + 1/m^2)')
print(f'  = ({four_n}/{ell})*(2*{W_mu}/{m} + 1/{m}^2) = {paper_correction}')
print()

# Counterexample test
c_target = 1.3
matlab_threshold = c_target + matlab_correction
paper_threshold = c_target + paper_correction
print(f'For c_target = {c_target}:')
print(f'  Matlab pruning threshold: TV_disc >= {matlab_threshold}')
print(f'  Paper  pruning threshold: TV_disc >= {paper_threshold}')
print()
print(f'  TV_disc = {TV_disc}, so:')
print(f'  Matlab prunes? {TV_disc >= matlab_threshold}  (should NOT prune unsoundly)')
print(f'  Paper  prunes? {TV_disc >= paper_threshold}')
print()
print(f'  But TV_cont = {TV_cont} which is ALREADY < c_target = {c_target}!')
print(f'  Margin: c_target - TV_cont = {c_target - TV_cont}')
print()

if TV_disc >= matlab_threshold and TV_cont < c_target:
    print('*** MATLAB BUG CONFIRMED ***')
    print('Matlab prunes a composition where the continuous test value falls below c_target.')
    print('This is UNSOUND: the matlab cascade can certify a lower bound that does not actually hold.')
elif TV_disc >= matlab_threshold:
    print('Matlab prunes; TV_cont >= c_target so locally fine. But Formula B is provably wrong in worst case.')
else:
    print('In this specific case, matlab does not prune. But Formula B over-claims for other compositions.')
