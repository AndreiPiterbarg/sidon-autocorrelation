"""
Compute the restricted C_{1a} over the doubly non-negative class:
    f >= 0, f_hat >= 0, supp f in [-1/2,1/2], ||f||_1 = 1
Equivalent reformulation: f = g * g_tilde, g >= 0, supp g in [0,1/2].
    ratio R(g) = ||g*g_tilde||_2^2 / ||g||_1^4
We minimize R(g) by projected gradient descent on a non-negative discrete g.

We use a fine grid g on [0, 1/2] with N points; convolutions are exact in the
discrete sense up to grid resolution.
"""
import numpy as np
from numpy.fft import rfft, irfft

def conv_full(a, b, dx):
    """Full continuous convolution (Riemann sum) via FFT."""
    n = len(a) + len(b) - 1
    # next pow2
    m = 1
    while m < n:
        m *= 2
    A = rfft(a, m)
    B = rfft(b, m)
    c = irfft(A*B, m)[:n] * dx
    return c

def ratio(g, dx):
    """R(g) = ||g*g_tilde||_2^2 / ||g||_1^4 with g real non-neg supported on [0,L]."""
    L1 = g.sum() * dx
    if L1 <= 0:
        return np.inf
    # g_tilde(x) = g(-x). Convolution g*g_tilde is autocorrelation A(t) = int g(s) g(s-t) ds
    # Discretely: autocorr = convolve(g, g[::-1]), output length 2N-1 covers t in [-L,L].
    A = conv_full(g, g[::-1], dx)
    f2 = (A*A).sum() * dx
    return f2 / (L1**4)

def grad_ratio(g, dx):
    """
    R(g) = N(g)/D(g), where N = ||A||_2^2, A = g*g_tilde, D = ||g||_1^4.
    Variation: dN = 2 <A, dA>; dA = (dg)*g_tilde + g*(d g_tilde)
              = 2 * (dg) * g_tilde  (symmetry, real g; cross terms equal in inner product with symmetric A)
    Actually for A(t) = int g(s) g(s-t) ds, dA(t)/dg(x) = g(x-t) + g(x+t).
    Then dN/dg(x) = 2 int A(t) (g(x-t)+g(x+t)) dt = 2 ( (A*g_tilde)(x) + (A*g)(x) ... )
    Since A is symmetric (A(t)=A(-t)) for real g, both terms equal: dN/dg(x) = 4 (A * g)(x)
    Wait carefully: int A(t) g(x-t) dt = (A*g)(x). int A(t) g(x+t) dt = (A * g_tilde)(x)
    Symmetry A(-t)=A(t) makes these equal => dN/dg = 4 (A*g)(x), restricted to support.
    dD/dg(x) = 4 L1^3.
    grad R = (dN * D - N * dD) / D^2 = (4(A*g) - 4 R L1^3 ) / D ? let's recompute:
       grad R = dN/D - N dD/D^2 = (1/D)[ 4(A*g)(x) - R * 4 L1^3 ]
              = (4/D)[ (A*g)(x) - R * L1^3 ]
       = (4 / L1^4)*((A*g)(x) - R L1^3)
       = (4/L1)*( (A*g)(x)/L1^3 - R )
    """
    L1 = g.sum() * dx
    A = conv_full(g, g[::-1], dx)        # length 2N-1, indices -N+1..N-1
    N_val = (A*A).sum() * dx
    R = N_val / L1**4
    # (A*g)(x): we want integral A(t) g(x-t) dt restricted to x in [0, L]
    # Implement as convolution conv_full(A, g, dx) of length (2N-1)+N-1 = 3N-2;
    # the t-integration variable for A indexes -L..L; result indices for output convolution
    # represent positions in [-L, 2L]. We need x in [0, L] which corresponds to indices N-1 .. 2N-2.
    Ag = conv_full(A, g, dx)
    N = len(g)
    Ag_on_support = Ag[N-1:2*N-1]
    grad = (4.0 / L1) * (Ag_on_support / L1**3 - R)
    return grad, R

def project_nonneg_normalize(g, target_L1, dx):
    """Project: clip to non-negative, then renormalize L1 to target."""
    g = np.maximum(g, 0.0)
    s = g.sum() * dx
    if s > 0:
        g *= target_L1 / s
    return g

def optimize(N=200, L=0.5, n_iter=5000, lr=0.05, seed=0, init='uniform'):
    """Projected gradient descent."""
    dx = L / N
    rng = np.random.default_rng(seed)
    if init == 'uniform':
        g = np.ones(N)
    elif init == 'random':
        g = 0.5 + rng.random(N)
    elif init == 'triangle':
        x = np.linspace(0, 1, N)
        g = 1.0 - np.abs(2*x - 1)
        g = np.maximum(g, 0)
    elif init == 'halfdomain':
        g = np.zeros(N)
        g[:N//2] = 1.0
    elif init == 'gauss':
        x = np.linspace(0, 1, N)
        g = np.exp(-20*(x-0.5)**2)
    g = project_nonneg_normalize(g, 1.0, dx)

    R_hist = []
    best_R = np.inf
    best_g = g.copy()
    for it in range(n_iter):
        grad, R = grad_ratio(g, dx)
        if R < best_R:
            best_R = R
            best_g = g.copy()
        R_hist.append(R)
        # gradient step
        g_new = g - lr * grad
        g_new = project_nonneg_normalize(g_new, 1.0, dx)
        # backtrack if step worsens
        gnext_R = ratio(g_new, dx)
        if gnext_R > R:
            # halve lr a few times
            for _ in range(20):
                lr_try = lr * 0.5
                g_try = project_nonneg_normalize(g - lr_try*grad, 1.0, dx)
                if ratio(g_try, dx) < R:
                    g_new = g_try
                    lr = lr_try
                    break
                lr = lr_try
        else:
            # try to grow
            lr = min(lr * 1.05, 1.0)
        g = g_new
    return best_R, best_g, R_hist

if __name__ == "__main__":
    print("=== Restricted C_{1a} over doubly nonneg (positive-definite) class ===\n")
    print("Reference scales:")
    print("  MV 2010 lower bound (unrestricted): 1.2748")
    print("  CS17 upper bound: 1.5029")
    print("  CS17 (purported) lower 1.2802 -- under audit\n")

    results = {}
    for init in ['uniform', 'triangle', 'halfdomain', 'gauss', 'random']:
        for N in [100, 200]:
            R, g_opt, hist = optimize(N=N, L=0.5, n_iter=3000, lr=0.02, seed=1, init=init)
            print(f"init={init:10s} N={N:4d}  best R = {R:.6f}   final R = {hist[-1]:.6f}")
            results[(init,N)] = R

    # Best overall
    best = min(results.values())
    print(f"\nBEST restricted C_{{1a}}^PD estimate: {best:.6f}")
    print(f"\nUnrestricted MV LB: 1.2748")
    if best > 1.2748:
        print("REFUTES MV: restricted UB > unrestricted LB.")
    else:
        print(f"Consistent with MV.  Gap to MV: {best - 1.2748:+.6f}")
    print(f"CS17 UB: 1.5029.  Gap to CS17 UB: {best - 1.5029:+.6f}")

    # Refine with longer run from best init
    best_init = min(results, key=results.get)
    print(f"\nRefining from best init {best_init} with N=400, 10k iter...")
    R_final, g_final, hist = optimize(N=400, L=0.5, n_iter=10000, lr=0.01, seed=1, init=best_init[0])
    print(f"Refined restricted C_{{1a}}^PD: {R_final:.6f}")

    # Save snapshot
    import json
    out = {
        "results_per_init": {f"{k[0]}_N{k[1]}": v for k,v in results.items()},
        "best_restricted_C1a_PD": float(best),
        "refined_N400_10k": float(R_final),
        "MV_LB": 1.2748,
        "CS17_UB": 1.5029,
    }
    with open(r"C:\Users\andre\OneDrive - PennO365\Desktop\compact_sidon\_master_pd_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved _master_pd_results.json")
