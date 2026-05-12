"""
Numerical Euler-Lagrange extremizer search for the restricted-Holder problem.

Maximize  J[f] = ||f*f||_2^2 / ||f*f||_inf  =:  c_emp(f)
subject to:
  (i)   ||f*f||_inf <= M_cap
  (ii)  int f = 1
  (iii) f >= 0
  (iv)  supp f subset [-1/4, 1/4]

If sup c_emp over the restricted class < log(16)/pi ~ 0.88254, Hyp_R holds.

Discretization
--------------
Grid: x_i = -1/4 + i*dx, i=0..N-1, dx = 0.5/(N-1).
G[k] := (f * f)(y_k), y_k = -1/2 + k*dx, k=0..2N-2; computed as
    G = fftconvolve(f, f) * dx.

Norms:
    ||g||_inf  = max_k G[k]
    ||g||_2^2  = sum_k G[k]^2 * dx
    ||g||_1    = sum_k G[k] * dx       (= 1 when int f = 1)

Gradients (continuous + discrete):
    dJ/df(x) = 4 (g * f_tilde)(x),  f_tilde(x) = f(-x)
Discretely  dJ/df[i] = 4 * dx^2 * fftconvolve(G, f[::-1])[i + N - 1].

Penalty P(f) = sum_k (G[k] - M_cap)_+^2 * dx; gradient:
    dP/df[i] = 4 * dx^2 * fftconvolve((G - M_cap)_+, f[::-1])[i + N - 1].

Constrained projection: project f >= 0 and rescale so int f * dx = 1.
"""
import numpy as np
from scipy.signal import fftconvolve

C_TARGET = float(np.log(16.0) / np.pi)   # 0.88254240061...


def autoconv(f, dx):
    return fftconvolve(f, f) * dx


def stats(f, dx):
    G = autoconv(f, dx)
    M = float(G.max())
    L2sq = float((G * G).sum() * dx)
    L1 = float(G.sum() * dx)
    return M, L2sq, L1, G


def project(f, dx):
    f = np.maximum(f, 0.0)
    s = float(f.sum() * dx)
    if s > 0:
        f = f / s
    return f


def project_simplex_pos(f, dx):
    """Project onto {f >= 0, sum(f)*dx = 1} (L2 projection by sort/threshold)."""
    # Eq Duchi et al.: project onto simplex with sum = 1/dx
    target = 1.0 / dx
    n = f.size
    u = np.sort(f)[::-1]
    cssv = np.cumsum(u) - target
    rho = np.argmax(u - cssv / (np.arange(n) + 1) <= 0) - 1
    if rho < 0:
        rho = n - 1
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(f - theta, 0.0)


def grad_J(f, dx, G=None):
    if G is None:
        G = autoconv(f, dx)
    full = fftconvolve(G, f[::-1])  # length 3N-2
    Nf = f.size
    return 4.0 * dx * dx * full[Nf - 1: 2*Nf - 1]


def grad_pen(f, dx, M_cap, G=None):
    if G is None:
        G = autoconv(f, dx)
    excess = np.maximum(G - M_cap, 0.0)
    full = fftconvolve(excess, f[::-1])
    Nf = f.size
    return 4.0 * dx * dx * full[Nf - 1: 2*Nf - 1]


def el_search(N=121, M_cap=1.378, n_steps=3000, lr=1e-3, seed=0,
              symmetric=False, n_restarts=8, alpha_pen=1.0,
              verbose=False, init_kind=None):
    """Projected gradient ascent on (J - alpha*P)."""
    rng = np.random.default_rng(seed)
    dx = 0.5 / (N - 1)
    x = np.linspace(-0.25, 0.25, N)

    best = (-np.inf, None, None)
    log = []

    for r in range(n_restarts):
        flavor = (r if init_kind is None else init_kind) % 7
        if flavor == 0:
            f = rng.uniform(0.5, 1.5, N)
        elif flavor == 1:
            sigma = 0.06 + 0.06 * rng.uniform()
            f = np.exp(-x*x / (2*sigma*sigma)) + 0.05*rng.uniform(size=N)
        elif flavor == 2:
            c1 = -0.18 + 0.04*rng.uniform()
            c2 =  0.10 + 0.10*rng.uniform()
            f = np.exp(-(x-c1)**2/0.005) + 0.7*np.exp(-(x-c2)**2/0.005)
        elif flavor == 3:
            f = np.ones(N) + 0.1*rng.uniform(size=N)
        elif flavor == 4:
            # 3 spikes
            f = np.zeros(N)
            ks = rng.choice(np.arange(5, N-5), size=3, replace=False)
            for kk in ks:
                f[kk-1:kk+2] = rng.uniform(0.5, 1.0)
        elif flavor == 5:
            # bathtub
            f = np.where(np.abs(x) > 0.20, 2.0, 0.5) + 0.05*rng.uniform(size=N)
        else:
            # half-indicator (asymmetric)
            f = np.where(x > 0.0, 1.5, 0.5)
        if symmetric:
            f = 0.5 * (f + f[::-1])
        f = project(f, dx)

        for it in range(n_steps):
            M, L2sq, L1, G = stats(f, dx)
            gJ = grad_J(f, dx, G)
            gP = grad_pen(f, dx, M_cap, G)
            grad = gJ - alpha_pen * gP
            # adaptive lr step (normalize by ||grad|| with min)
            gn = np.linalg.norm(grad)
            step = lr * (1.0 / max(gn, 1e-3))
            f = f + step * grad
            if symmetric:
                f = 0.5 * (f + f[::-1])
            f = project(f, dx)

        M, L2sq, L1, G = stats(f, dx)
        c_emp = L2sq / max(M, 1e-12)
        # Require constraint within tolerance
        feasible = (M <= M_cap * 1.001) and (M > 1.05)  # avoid trivial f=2*indicator with M~2
        if verbose:
            print(f"  restart {r}: c_emp={c_emp:.4f}  M={M:.4f}  L1={L1:.4f}  feas={feasible}")
        log.append((c_emp, M, feasible))
        if feasible and c_emp > best[0]:
            best = (c_emp, M, f.copy())
    return best, log


def el_residual(f, dx, M_cap):
    """Compute Lagrange multipliers (mu, lambda) and E-L residual.

    Continuous E-L (interior of supp):
        4 (f*f*f_tilde)(x) - 2 lambda (f_tilde * dnu)(x) - mu = 0,
    where dnu is the (probability) measure supported on {y : G(y) = M}.

    Discretely on supp(f) we solve for (mu, lambda) by least squares.
    """
    M, L2sq, L1, G = stats(f, dx)
    Nf = f.size
    # peak measure
    peak_thr = M - 1e-4 * max(M, 1.0)
    peak_mask = (G >= peak_thr).astype(float)
    s = float(peak_mask.sum() * dx)
    if s <= 0:
        return None
    peak_mask = peak_mask / s   # normalized as a discretized pdf
    # T1 := 4 (G * f_tilde) restricted to supp(f); units: dx multiplications
    T1full = fftconvolve(G, f[::-1])
    T1 = 4.0 * dx * T1full[Nf - 1: 2*Nf - 1]   # note: NOT dx^2, since this is (g*f_tilde)(x), continuous-style
    # T2 := 2 (peak_mask * f_tilde) restricted to supp(f)
    T2full = fftconvolve(peak_mask, f[::-1])
    T2 = 2.0 * dx * T2full[Nf - 1: 2*Nf - 1]
    supp = f > 1e-4 * f.max()
    if supp.sum() < 3:
        return None
    A = np.stack([np.ones(supp.sum()), -T2[supp]], axis=1)
    b = T1[supp]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    mu, lam = sol
    resid = T1 - lam * T2 - mu
    rel = float(np.linalg.norm(resid[supp]) / max(np.linalg.norm(b), 1e-12))
    return dict(mu=float(mu), lam=float(lam),
                resid_rel=rel,
                resid_max=float(np.abs(resid[supp]).max()),
                M=M, L2sq=L2sq, c_emp=L2sq/M, peak_supp_meas=s)


# ---- Sanity tests ----

def _sanity():
    print("Sanity: f = 2 * 1_{[-1/4,1/4]} (indicator).")
    N = 401
    dx = 0.5/(N-1)
    f = 2.0 * np.ones(N)
    M, L2sq, L1, G = stats(f, dx)
    print(f"  M={M:.5f} (exp 2.0)  L2sq={L2sq:.5f} (exp 4/3 = {4/3:.5f})  "
          f"L1={L1:.5f} (exp 1)  c_emp={L2sq/M:.5f} (exp 2/3={2/3:.5f})")

    print("\nSanity: f triangle on [-1/4,1/4].  c_emp = ?")
    x = np.linspace(-0.25, 0.25, N)
    f_t = np.maximum(0.25 - np.abs(x), 0.0)
    f_t = f_t / (f_t.sum()*dx)
    M, L2sq, L1, G = stats(f_t, dx)
    print(f"  M={M:.5f}  L2sq={L2sq:.5f}  L1={L1:.5f}  c_emp={L2sq/M:.5f}")

    # gradient finite-diff check on a smooth f
    print("\nSanity: grad_J finite-diff (without projection).")
    sigma = 0.05
    f0 = np.exp(-x*x/(2*sigma*sigma))
    f0 = f0 / (f0.sum()*dx)   # normalize to int=1
    g_an = grad_J(f0, dx)
    eps = 1e-5
    e = np.zeros(N); ig = N//3; e[ig] = 1.0
    M_p, L2_p, _, _ = stats(f0 + eps*e, dx)
    M_m, L2_m, _, _ = stats(f0 - eps*e, dx)
    g_num = (L2_p - L2_m)/(2*eps)
    print(f"  analytic grad[{ig}] = {g_an[ig]:.5f}   numerical = {g_num:.5f}   "
          f"ratio = {g_an[ig]/g_num if abs(g_num)>1e-12 else 'NA'}")


if __name__ == "__main__":
    print("=" * 70)
    print(f"Restricted-Holder E-L extremizer search; target c* = {C_TARGET:.6f}")
    print("=" * 70)
    _sanity()

    rounds = [
        ("Round 1: SYM, M_cap=1.378",      dict(N=121, M_cap=1.378, n_steps=3000, lr=2e-3,
                                                symmetric=True, n_restarts=14, alpha_pen=10.0, seed=42)),
        ("Round 2: GEN, M_cap=1.378",      dict(N=121, M_cap=1.378, n_steps=3000, lr=2e-3,
                                                symmetric=False, n_restarts=20, alpha_pen=10.0, seed=7)),
        ("Round 3: GEN, M_cap=1.500",      dict(N=121, M_cap=1.500, n_steps=3000, lr=2e-3,
                                                symmetric=False, n_restarts=14, alpha_pen=10.0, seed=99)),
        ("Round 4: GEN, M_cap=1.652 (BL)", dict(N=121, M_cap=1.652, n_steps=3000, lr=2e-3,
                                                symmetric=False, n_restarts=14, alpha_pen=10.0, seed=123)),
        ("Round 5: SYM N=201, M_cap=1.378",dict(N=201, M_cap=1.378, n_steps=4000, lr=1.5e-3,
                                                symmetric=True, n_restarts=10, alpha_pen=15.0, seed=2026)),
    ]

    summary = []
    for name, kw in rounds:
        print(f"\n[{name}]")
        (c, M, f), _log = el_search(verbose=False, **kw)
        if f is not None:
            info = el_residual(f, 0.5/(kw['N']-1), kw['M_cap'])
            print(f"  best c_emp = {c:.6f}  M = {M:.5f}  margin to c* = {C_TARGET - c:+.6f}")
            if info:
                print(f"  E-L residual rel L2 (on supp) = {info['resid_rel']:.3e}  "
                      f"mu={info['mu']:.3f}  lambda={info['lam']:.3f}")
            summary.append((name, c, M))
        else:
            print("  NO feasible (M <= M_cap*1.001) restart found.")
            summary.append((name, None, None))

    print("\n" + "=" * 70)
    print(f"SUMMARY: target c* = log(16)/pi = {C_TARGET:.6f}")
    print("=" * 70)
    for name, ce, M in summary:
        if ce is None:
            print(f"  {name}: <no feasible point>")
            continue
        flag = "OK (Hyp_R holds at f)" if ce < C_TARGET else "VIOLATES Hyp_R"
        print(f"  {name}: c_emp={ce:.6f}  M={M:.5f}  margin={C_TARGET-ce:+.6f}  {flag}")
