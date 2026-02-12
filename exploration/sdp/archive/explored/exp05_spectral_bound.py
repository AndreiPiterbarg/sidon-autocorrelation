"""Experiment 5: Spectral/eigenvalue bounds on V(P).

Different approach: use spectral properties of the convolution matrices A_k
to derive analytic or semi-analytic bounds.

Key identity: V(P) = min_{x in Delta_P} max_k 2P * x^T A_k x

For any probability distribution lambda on {0,...,2P-2}:
  V(P) >= min_{x in Delta_P} 2P * x^T Q(lambda) x
  where Q(lambda) = sum_k lambda_k A_k.

This is the Shor relaxation, giving 2P/(2P-1) for lambda = uniform.

BUT: We can get a TIGHTER bound by noting that for the OPTIMAL x*:
  max_k x*^T A_k x* is achieved at MULTIPLE k-values (generically).

Idea 1: Use the known primal solutions to identify which diagonals are
active at the optimum, then optimize lambda over those.

Idea 2: Compute V(P) bounds using the trace identity:
  sum_k x^T A_k x = (sum x_i)^2 = 1  (on the simplex)
  So we have 2P-1 quadratics summing to 1. By pigeonhole:
  max_k x^T A_k x >= 1/(2P-1), giving the Shor bound.

Idea 3: Use HIGHER MOMENTS of the distribution {x^T A_k x}_k.
  sum_k (x^T A_k x)^2 = x^T (sum_k A_k^2) x ... no, this is degree 4.
  Actually: sum_k (x^T A_k x)^2 = sum_{k} sum_{i+j=k,a+b=k} x_i x_j x_a x_b.

Let's compute this sum and see if it gives a useful bound.
"""
import numpy as np
from scipy.optimize import minimize
import time

def build_A(P):
    n_diags = 2 * P - 1
    A = []
    for k in range(n_diags):
        Ak = np.zeros((P, P))
        for i in range(max(0, k - P + 1), min(P, k + 1)):
            Ak[i, k - i] = 1
        A.append(Ak)
    return A


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


print("=" * 72)
print("EXP 5: Spectral analysis of convolution structure")
print("=" * 72)

# Part A: Analyze the structure of optimal primal solutions
print("\n--- Part A: Diagonal profile of primal optima ---")
for P in [5, 8, 10, 15]:
    A_mats = build_A(P)

    # Find good primal solution
    best_val = np.inf
    best_x = None
    for seed in range(50):
        rng = np.random.RandomState(seed)
        z0 = rng.randn(P) * 0.5
        for beta in [5, 20, 100, 500]:
            def obj_lse(z, b=beta):
                x = softmax(z)
                conv = np.convolve(x, x, mode='full')
                mx = np.max(conv)
                return 2*P*(mx + (1/b)*np.log(np.sum(np.exp(b*(conv-mx)))))
            res = minimize(obj_lse, z0, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-15})
            z0 = res.x
        def obj_exact(z):
            x = softmax(z)
            return 2*P*np.max(np.convolve(x, x, mode='full'))
        res = minimize(obj_exact, z0, method='L-BFGS-B',
                       options={'maxiter': 1000, 'ftol': 1e-15})
        if res.fun < best_val:
            best_val = res.fun
            best_x = softmax(res.x)

    # Diagonal values
    diag_vals = np.array([best_x @ A @ best_x for A in A_mats])
    max_diag = np.max(diag_vals)
    active = np.where(diag_vals > max_diag - 1e-6)[0]
    near_active = np.where(diag_vals > max_diag * 0.99)[0]
    print(f"\n  P={P}: V(P)={best_val:.6f}, max diag val={max_diag:.8f}")
    print(f"    Active diags (within 1e-6): {active} ({len(active)} of {2*P-1})")
    print(f"    Near-active (99%):          {near_active} ({len(near_active)} of {2*P-1})")
    print(f"    Top 5 diag values: {np.sort(diag_vals)[-5:][::-1]}")
    print(f"    x profile: min={best_x.min():.6f}, max={best_x.max():.6f}, "
          f"entropy={-np.sum(best_x*np.log(best_x+1e-30)):.4f}")


# Part B: Second moment bound
print("\n\n--- Part B: Second moment / Cauchy-Schwarz bound ---")
print("If sum_k q_k = 1 and sum_k q_k^2 >= S2, then max q_k >= S2.")
print("(Because max q_k >= sum q_k^2 / sum q_k = S2.)")
print()

for P in range(2, 16):
    A_mats = build_A(P)
    n_diags = 2 * P - 1

    # sum_k (x^T A_k x)^2 for x on simplex
    # Build the degree-4 tensor: T_{ijab} = sum_k A_k[i,j] * A_k[a,b]
    # = number of k such that i+j=k AND a+b=k = 1 if i+j=a+b, else 0
    # So sum_k (x^T A_k x)^2 = sum_{i+j=a+b} x_i x_j x_a x_b

    # Minimize this over the simplex to get tightest S2
    def second_moment(z):
        x = softmax(z)
        diags = np.array([x @ A @ x for A in A_mats])
        return np.sum(diags**2)

    best_s2 = np.inf
    for seed in range(20):
        z0 = np.random.RandomState(seed).randn(P) * 0.5
        res = minimize(second_moment, z0, method='L-BFGS-B',
                       options={'maxiter': 500})
        if res.fun < best_s2:
            best_s2 = res.fun

    shor = 2*P / (2*P-1)
    cs_bound = 2*P * best_s2  # max_k 2P*q_k >= 2P * S2
    # Actually: max q_k >= S2 (since sum q_k = 1, sum q_k^2 >= S2)
    # So max_k 2P*q_k >= 2P * S2
    # But S2 >= 1/(2P-1) always (by Cauchy-Schwarz on sum=1)
    # The question is whether minimizing S2 over the simplex gives S2 > 1/(2P-1)

    uniform_s2 = 1.0 / (2*P-1)
    improvement = best_s2 / uniform_s2

    print(f"  P={P:2d}: min S2 = {best_s2:.8f}, uniform S2 = {uniform_s2:.8f}, "
          f"ratio = {improvement:.6f}, CS bound = {cs_bound:.6f} vs Shor = {shor:.6f}")


# Part C: Higher moment bounds
print("\n\n--- Part C: Higher moment bounds ---")
print("max q_k >= (sum q_k^p)^{1/p} / (2P-1)^{1/p - 1}  for any p >= 1")

for P in [5, 8, 10]:
    A_mats = build_A(P)
    n_diags = 2 * P - 1
    shor = 2*P / (2*P-1)

    for p in [2, 3, 4, 6, 10]:
        def moment_p(z, p=p):
            x = softmax(z)
            diags = np.array([x @ A @ x for A in A_mats])
            return np.sum(diags**p)

        best_mp = np.inf
        for seed in range(20):
            z0 = np.random.RandomState(seed).randn(P) * 0.5
            res = minimize(moment_p, z0, method='L-BFGS-B',
                           options={'maxiter': 500})
            if res.fun < best_mp:
                best_mp = res.fun

        # max q_k >= (sum q_k^p)^(1/(p-1)) / (2P-1)^(1/(p-1))  ... hmm
        # Actually: max q_k >= (sum q_k^p / sum q_k)^(1/(p-1)) by Holder
        # = (sum q_k^p)^(1/(p-1)) since sum q_k = 1
        # No wait: sum q_k^p <= (max q_k)^{p-1} * sum q_k = (max q_k)^{p-1}
        # So max q_k >= (sum q_k^p)^{1/(p-1)}
        bound = 2*P * best_mp**(1/(p-1))
        print(f"  P={P}, p={p}: min sum(q^p)={best_mp:.8f}, "
              f"bound = 2P * S_p^(1/(p-1)) = {bound:.6f} (Shor={shor:.6f})")
