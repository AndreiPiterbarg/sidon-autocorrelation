"""Check if for the matlab-pruned composition, max(f*f) is actually < c_target.
If yes, matlab is provably unsound.
"""
import numpy as np

n = 2
d = 4
m = 10
c_target = 1.3

# Test mu candidates that map to c=(2,1,1,6)
# c=(2,1,1,6) requires M(1) in [0.2,0.3), M(2) in [0.3,0.4), M(3) in [0.4,0.5)
# Equivalently mu_0 in [0.2,0.3), mu_0+mu_1 in [0.3,0.4), mu_0+mu_1+mu_2 in [0.4,0.5)

def step_max_conv(mu, n):
    """Compute max((f_step * f_step)(t)) for the step function with bin masses mu.
    Convolution is piecewise linear with peaks at knots t_k = -1/2 + k/(2*n).
    Peak value at knot k: 4n * sum_{i+j=k-1} mu_i * mu_j"""
    d = 2 * n
    max_val = 0.0
    for k in range(0, 2 * d):
        peak = 0.0
        for i in range(d):
            j = k - 1 - i
            if 0 <= j < d:
                peak += mu[i] * mu[j]
        peak *= 4 * n
        max_val = max(max_val, peak)
    return max_val

# Sample many mus
candidates = []
for mu_0 in np.linspace(0.20, 0.299, 20):
    for s_01 in np.linspace(0.30 + 1e-6, 0.399, 20):
        mu_1 = s_01 - mu_0
        if mu_1 < 0:
            continue
        for s_012 in np.linspace(0.40 + 1e-6, 0.499, 20):
            mu_2 = s_012 - mu_0 - mu_1
            if mu_2 < 0:
                continue
            mu_3 = 1.0 - mu_0 - mu_1 - mu_2
            if mu_3 < 0:
                continue
            mu = np.array([mu_0, mu_1, mu_2, mu_3])
            # Verify cumulative floor gives c=(2,1,1,6)
            M = np.zeros(d + 1)
            for i in range(d):
                M[i+1] = M[i] + mu[i]
            D = np.floor(m * M).astype(int)
            D[d] = m
            c_check = np.diff(D)
            if not np.array_equal(c_check, [2, 1, 1, 6]):
                continue
            max_conv = step_max_conv(mu, n)
            candidates.append((mu, max_conv))

print(f'Found {len(candidates)} valid mus')
candidates.sort(key=lambda x: x[1])
print(f'Min max(f_step*f_step): {candidates[0][1]:.6f}')
print(f'  achieved at mu = {candidates[0][0]}')
print(f'Max max(f_step*f_step): {candidates[-1][1]:.6f}')

low_max = candidates[0][1]
if low_max < c_target:
    print(f'\n*** STEP-FUNCTION COUNTEREXAMPLE ***')
    print(f'There exists a step function f_step with bin masses mapping to c=(2,1,1,6)')
    print(f'and max(f_step*f_step) = {low_max:.6f} < c_target = {c_target}.')
    print(f'Matlab prunes this composition (claims no continuous f can violate).')
    print(f'But the step function IS continuous (piecewise constant), so matlab is UNSOUND.')
elif low_max >= c_target:
    print(f'\nNo step-function counterexample found (min max = {low_max} >= c_target).')
    print(f'Matlab might still be unsound for a non-step f, but step-functions don\'t')
    print(f'directly violate. The TV_cont vs c_target gap (TV_cont={1.04} < {c_target}) is the')
    print(f'ABSTRACT certificate-failure, but doesn\'t immediately give a numeric counterexample.')
