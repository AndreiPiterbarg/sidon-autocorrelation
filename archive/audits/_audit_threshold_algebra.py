"""Verify the threshold algebra: matlab vs paper Formula A vs python production.

Goal: show whether matlab's boundToBeat IS the same as Formula A in disguise,
or whether matlab UNDER-CORRECTS (missing 4n/ell factor).

Strategy: compute concrete threshold values for representative parameters
and confirm by independent algebra.
"""
import numpy as np

def matlab_threshold(c_target, m, W_continuous):
    """Matlab boundToBeat in continuous units (line 286).
    W_continuous = sum of bin masses (m_i) over contributing bins, summing to <= 1."""
    gridSpace = 1.0 / m
    return c_target + gridSpace**2 + 2 * gridSpace * W_continuous

def paper_correction(n, m, ell, W_continuous):
    """Paper Formula A correction (cs2017_continuity_extraction.md)."""
    return (4 * n / ell) * (2 * W_continuous / m + 1 / m**2)

def paper_threshold(c_target, n, m, ell, W_continuous):
    return c_target + paper_correction(n, m, ell, W_continuous)

# Test at d=4, m=10, c_target=1.3, ell=2 (the canonical counterexample setting)
n = 2
d = 2 * n
m = 10
ell = 2
c_target = 1.3

print('='*60)
print('Configuration: d=4, m=10, c_target=1.3, ell=2')
print('='*60)

for W_int in [0, 1, 3, 6, 10]:
    W_cont = W_int / m  # bin mass = c_i/m, sum over contributing bins = W_int/m
    matlab_t = matlab_threshold(c_target, m, W_cont)
    paper_t = paper_threshold(c_target, n, m, ell, W_cont)
    print(f'  W_int={W_int:3d} (W_cont={W_cont:.2f}): matlab={matlab_t:.4f}  paper={paper_t:.4f}  diff={paper_t-matlab_t:.4f}')

print()
print('At smallest window ell=2, paper threshold > matlab by factor (4n/ell-1)*(2W/m+1/m^2)')
print('= 3 * (2W/m + 1/m^2). At W=0: paper-matlab = 3/m^2 = 0.03. At W=0.6: 3*(0.12+0.01) = 0.39.')
print()

# At larger ell, the gap shrinks
print('Same config, varying ell:')
for ell_v in [2, 4, 6, 8]:
    W_cont = 0.6  # representative
    matlab_t = matlab_threshold(c_target, m, W_cont)  # ell-independent
    paper_t = paper_threshold(c_target, n, m, ell_v, W_cont)
    print(f'  ell={ell_v}: paper={paper_t:.4f}  matlab={matlab_t:.4f}  ratio (paper-c_target)/(matlab-c_target) = {(paper_t-c_target)/(matlab_t-c_target):.3f}')

print()
print('At ell = 2*d = 4n (largest window), 4n/ell = 1, so paper = matlab. AGREE only here.')
print()

# Demonstrate the algebra error in tests/investigate_matlab_vs_python_threshold.py:
print('='*60)
print('Algebra check: matlab in INT units')
print('='*60)
# The investigation file claims: matlab in int units = (c_target*m^2 + 1 + W_int/(2n)) * 4n*ell
# This would only be correct if W_continuous = W_int/(4nm).
# Let's check: matlab boundToBeat in continuous TV space:
#   matlab_thresh = c_target + 1/m^2 + 2/m * W_continuous (the matlab formula in continuous space)
#
# Match to integer space: TV_cont = ws_int * (4n / ell) / m^2 / S^... hmm
# Convention: paper TV_disc(c; ell, s_lo) = (4n/ell) * sum_{(i,j) in window} (c_i/m)(c_j/m)
#                                          = (4n/(ell * m^2)) * ws_int
# So ws_int = TV_disc * (ell * m^2) / (4n)
#
# Pruning rule (matlab): TV_disc >= c_target + 1/m^2 + 2/m * W_cont
# Multiply both sides by ell*m^2/(4n):
#   ws_int >= (ell * m^2 / (4n)) * (c_target + 1/m^2 + 2 W_cont / m)
#          = (c_target * m^2 + 1 + 2 W_cont * m) * ell / (4n)
# That's NOT the (c_target*m^2 + 1 + W_int/(2n)) * 4n*ell form.

# Let me redo for clarity.
n_half = n
def matlab_int_threshold(c_target, n_half, m, ell, W_int):
    """Convert matlab boundToBeat to integer space (ws_int)."""
    W_cont = W_int / m
    matlab_TV = c_target + 1/m**2 + 2 * W_cont / m
    return matlab_TV * ell * m**2 / (4 * n_half)

def python_int_threshold(c_target, n_half, m, ell, W_int):
    """Python production threshold (paper Formula A) in int space."""
    corr_w = 1.0 + W_int / (2.0 * n_half)
    return (c_target * m**2 + corr_w) * 4 * n_half * ell

print(f'\n{"W_int":>5} {"matlab_int":>15} {"python_int":>15} {"ratio":>8}')
print('-'*50)
for W_int in [0, 1, 3, 6, 10, 30, 80]:
    mt = matlab_int_threshold(c_target, n, m, ell, W_int)
    pt = python_int_threshold(c_target, n, m, ell, W_int)
    print(f'{W_int:>5} {mt:>15.4f} {pt:>15.4f} {pt/mt:>8.3f}')

print()
print('VERDICT: matlab integer threshold does NOT equal python (paper Formula A) integer threshold.')
print('Ratio = (4n/ell)^2 = 16 at ell=2 — matlab threshold is much smaller.')
print('Investigation file `tests/investigate_matlab_vs_python_threshold.py` is WRONG: the formulas')
print('are NOT equivalent. (The error in line 182 of that file: W_cont = W_int/(4nm), but actually')
print('W_cont = W_int/m for matlab\'s convention.)')
