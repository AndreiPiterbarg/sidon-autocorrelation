"""
L-BFGS-B based extremizer search for the restricted-Holder problem.

Variables: f >= 0 on grid, ||f||_1 = 1 enforced, ||f*f||_inf <= M_cap.
Objective: maximize c_emp(f) = ||f*f||_2^2 / ||f*f||_inf  (subject to <= M_cap).

We reparametrize f = exp(theta) / (sum exp(theta)*dx) (positive + L1=1 free of
constraints), and use a smooth log-barrier on the M_cap constraint via a
softmax surrogate of ||g||_inf.  Then minimize  -c_emp(f) + alpha*max(0, M-M_cap)^2
"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve

C_TARGET = float(np.log(16.0)/np.pi)


def autoconv(f, dx):
    return fftconvolve(f, f) * dx


def f_from_theta(theta, dx):
    # avoid overflow
    th = theta - theta.max()
    f = np.exp(th)
    f = f / (f.sum()*dx)
    return f


def softmax_norm(G, beta):
    """Smooth approximation of ||G||_inf  =  (1/beta) log sum exp(beta G)"""
    Gm = G.max()
    return Gm + (1.0/beta) * np.log(np.exp(beta*(G - Gm)).sum())


def loss(theta, dx, M_cap, alpha, beta_smax, sym=False):
    if sym:
        # enforce symmetry by averaging
        theta = 0.5*(theta + theta[::-1])
    f = f_from_theta(theta, dx)
    G = autoconv(f, dx)
    M = float(G.max())
    Msoft = softmax_norm(G, beta_smax)
    L2sq = float((G*G).sum()*dx)
    # objective: maximize  L2sq / M     (true c_emp); penalty on Msoft > M_cap
    # use Msoft (smooth) for both the ratio and the penalty — gives smooth grad
    obj = -L2sq / Msoft
    pen = alpha * max(0.0, Msoft - M_cap)**2
    return obj + pen


def search_lbfgs(N=121, M_cap=1.378, sym=False, n_restarts=20, seed=0,
                 alphas=(50.0, 200.0, 1000.0), betas=(50.0, 100.0, 200.0),
                 verbose=False):
    rng = np.random.default_rng(seed)
    dx = 0.5/(N-1)
    x = np.linspace(-0.25, 0.25, N)
    best = (-np.inf, None, None)
    log = []
    for r in range(n_restarts):
        flavor = r % 8
        if flavor == 0:
            th0 = rng.normal(0, 1.0, N)
        elif flavor == 1:
            sigma = 0.05 + 0.05*rng.uniform()
            th0 = -x*x/(2*sigma*sigma) + 0.1*rng.normal(size=N)
        elif flavor == 2:
            c1 = -0.18 + 0.04*rng.uniform()
            c2 =  0.10 + 0.10*rng.uniform()
            th0 = np.log(np.exp(-(x-c1)**2/0.005) + 0.7*np.exp(-(x-c2)**2/0.005) + 1e-3)
        elif flavor == 3:
            th0 = 0.2*rng.normal(size=N)
        elif flavor == 4:
            # bathtub
            th0 = np.where(np.abs(x)>0.20, 0.5, -0.5) + 0.1*rng.normal(size=N)
        elif flavor == 5:
            # asymmetric step
            th0 = np.where(x>0, 0.5, -0.5) + 0.1*rng.normal(size=N)
        elif flavor == 6:
            # Sidon-like 3 spikes (smoothed)
            th0 = -10*np.ones(N)
            ks = sorted(rng.choice(np.arange(5, N-5), size=3, replace=False))
            for kk in ks:
                th0[kk-2:kk+3] = 0.5
        else:
            # try MV-like: many cosines
            th0 = np.zeros(N)
            for j in range(1, 8):
                th0 += rng.normal()*np.cos(2*np.pi*j*x)
        if sym:
            th0 = 0.5*(th0 + th0[::-1])
        # Continuation in alpha and beta
        for alpha in alphas:
            for beta in betas:
                res = minimize(loss, th0, args=(dx, M_cap, alpha, beta, sym),
                               method='L-BFGS-B', jac=None,
                               options=dict(maxiter=300, ftol=1e-10, gtol=1e-7))
                th0 = res.x  # warm-start next stage
        f = f_from_theta(th0, dx)
        if sym:
            f = 0.5*(f + f[::-1])
            f = f / (f.sum()*dx)
        G = autoconv(f, dx)
        M = float(G.max())
        L2sq = float((G*G).sum()*dx)
        c_emp = L2sq/max(M,1e-12)
        feasible = (M <= M_cap*1.005)
        log.append((c_emp, M, feasible))
        if verbose:
            print(f"  restart {r}: c_emp={c_emp:.4f}  M={M:.4f}  feas={feasible}")
        if feasible and c_emp > best[0]:
            best = (c_emp, M, f.copy())
    return best, log


def el_residual(f, dx):
    """Lagrange residual for E-L (continuous form):
        4 (f*f*f_tilde) - 2*lambda*(f_tilde * dnu) - mu = 0   on supp(f)
    """
    G = autoconv(f, dx)
    M = float(G.max())
    L2sq = float((G*G).sum()*dx)
    Nf = f.size
    # peak measure dnu (probability)
    peak_thr = M - 1e-3*max(M,1.0)
    pm = (G >= peak_thr).astype(float)
    s = float(pm.sum()*dx)
    if s <= 0: return None
    pm = pm/s
    # T1[x] = 4 (g*f_tilde)(x)
    T1full = fftconvolve(G, f[::-1])
    T1 = 4.0*dx * T1full[Nf-1:2*Nf-1]
    # T2[x] = 2 (peak * f_tilde)(x)
    T2full = fftconvolve(pm, f[::-1])
    T2 = 2.0*dx * T2full[Nf-1:2*Nf-1]
    supp = f > 1e-3*f.max()
    if supp.sum() < 3: return None
    A = np.stack([np.ones(supp.sum()), -T2[supp]], axis=1)
    b = T1[supp]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    mu, lam = sol
    resid = T1 - lam*T2 - mu
    return dict(mu=float(mu), lam=float(lam),
                resid_rel=float(np.linalg.norm(resid[supp])/max(np.linalg.norm(b),1e-12)),
                M=M, c_emp=L2sq/M, peak_supp=s)


if __name__ == "__main__":
    print("=" * 70)
    print(f"Restricted-Holder L-BFGS-B extremizer search; c* = {C_TARGET:.6f}")
    print("=" * 70)

    rounds = [
        ("SYM, M_cap=1.378",       dict(N=121, M_cap=1.378, sym=True,  n_restarts=24, seed=42)),
        ("GEN, M_cap=1.378",       dict(N=121, M_cap=1.378, sym=False, n_restarts=30, seed=7)),
        ("GEN, M_cap=1.500",       dict(N=121, M_cap=1.500, sym=False, n_restarts=24, seed=99)),
        ("GEN, M_cap=1.652 (BL)",  dict(N=121, M_cap=1.652, sym=False, n_restarts=24, seed=123)),
        ("SYM, M_cap=1.378 N=181", dict(N=181, M_cap=1.378, sym=True,  n_restarts=16, seed=2026)),
    ]
    summary = []
    for name, kw in rounds:
        print(f"\n[{name}]")
        (c, M, f), _log = search_lbfgs(verbose=True, **kw)
        if f is not None:
            info = el_residual(f, 0.5/(kw['N']-1))
            print(f"  >>> BEST c_emp={c:.6f}  M={M:.5f}  margin={C_TARGET-c:+.6f}")
            if info:
                print(f"      E-L resid rel L2 = {info['resid_rel']:.3e}  "
                      f"mu={info['mu']:.3f}  lambda={info['lam']:.3f}  peak_supp_meas={info['peak_supp']:.4f}")
            summary.append((name, c, M, info))
        else:
            print("  no feasible point.")
            summary.append((name, None, None, None))

    print("\n" + "="*70)
    print(f"FINAL SUMMARY: c* = {C_TARGET:.6f}")
    print("="*70)
    for name, c, M, info in summary:
        if c is None:
            print(f"  {name}: no feasible.")
            continue
        flag = "OK" if c < C_TARGET else "VIOLATES Hyp_R"
        print(f"  {name}: c_emp={c:.6f}  M={M:.5f}  margin={C_TARGET-c:+.6f}  {flag}")
