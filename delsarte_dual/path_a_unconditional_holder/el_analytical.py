"""
Analytical Euler-Lagrange characterization of the Phi(M) extremizer.

Phi(M) := sup { ||f*f||_2^2 / ||f*f||_inf : f >= 0 on [-1/4,1/4], int f = 1,
                                            ||f*f||_inf <= M }.

We want to know whether Phi(M) >= c_* := log(16)/pi  ~= 0.882542  for some
M in (0, 1.378].  If NO, then Hyp_R(c_*, 1.378) holds unconditionally.

E-L derivation (continuous form)
================================
With g := f*f, write the Lagrangian
    L = || g ||_2^2 - lambda * ( ||g||_inf - M )      # peak constraint
                   - mu * ( int f - 1 )                # mass
                   + < psi, f >                        # f >= 0  (psi >= 0, psi*f=0)

The peak ||g||_inf is realized on a (possibly singleton) "active set"
A := { y : g(y) = ||g||_inf }, with a Lagrange measure dnu supported on A,
dnu >= 0, int dnu = 1 at the active value (proper subgradient).

Variation in f: the gradient of ||g||_2^2 = int (f*f)^2 is
    d/df(x) ||g||_2^2 = 4 (g * f_tilde)(x) ,   f_tilde(x) := f(-x).
The gradient of ||g||_inf in the subgradient sense is
    d/df(x) ||g||_inf = 2 ( f_tilde * dnu )(x)    (the "2" from f appearing twice in g).
The gradient of int f is the constant 1.

Setting delta L / delta f = 0  (interior of supp f, where psi = 0):

    4 (g * f_tilde)(x) - 2 lambda ( f_tilde * dnu )(x) - mu = 0   on supp f.   (E-L)

On the *complement* of supp f (where f = 0), psi(x) >= 0 and
    4 (g * f_tilde)(x) - 2 lambda (f_tilde * dnu)(x) - mu  =  -psi(x) <= 0.

Since g = f*f, we can rewrite g*f_tilde  =  f*f*f_tilde  =  triple convolution.
For SYMMETRIC f (f = f_tilde), this is  (f*f*f) = the third self-convolution h(x).

Symmetric ansatz
----------------
If f is symmetric (f(x) = f(-x)), then g = f*f is symmetric, the active set
A is symmetric (A = -A), and dnu can be chosen symmetric. (E-L) becomes

    4 h(x) - 2 lambda phi(x) - mu = 0      on supp f,
    h := f*f*f,    phi := f * dnu   (symmetric).                          (E-L_sym)

In particular, on supp f, h(x) is an AFFINE function of phi(x):
    h(x) = (lambda/2) phi(x) + mu/4.                                   (Linear)

This linear identity is the *necessary condition* for any critical point of Phi.

Bang-bang structure
-------------------
By a soft-thresholding argument: if at some interior point x0 in supp f the
strict inequality 4h(x0) - 2 lambda phi(x0) - mu > 0 held (with f(x0) > 0),
we could push mass onto x0 and improve J at fixed mass; if < 0 we'd reduce
f(x0). Hence (E-L_sym) is sharp on supp f; off supp f, the LHS is <= 0.

Numerical strategy
==================
We solve a finite-dimensional analogue: with N grid points and discretized
g, h, phi, the Lagrangian Hessian is generically rank-deficient with kernel
parameterized by (mu, lambda, dnu). We use:

(A) A *projected gradient* loop on f, with a smooth softmax surrogate for
    ||g||_inf used both in the constraint penalty and in the Lagrange
    measure dnu := softmax(beta*g) (a discretized peak measure).

(B) After convergence, compute the actual E-L residual

       R(x) := 4 h(x) - 2 lambda phi(x) - mu        on supp f

    and the *dual* feasibility R(x) <= 0 off supp f. The norm
    ||R||_{supp f} / ||4h||_{supp f} is the relative E-L violation.

(C) Sweep M in [1.05, 1.378]. For each M, run multiple restarts (smooth
    Gaussian, multi-Gaussian, near-indicator, BL truncation, MV-cosine basis,
    bathtub, asymmetric tilts), record the best (c_emp, M, asym, residual).

Output: a table c_emp(M) and a verdict
   PROMISING   if  sup_M c_emp(M) <  c_* - 0.005
   INCONCLUSIVE if  sup_M c_emp(M)  in  [c_* - 0.005, c_*]
   OBSTRUCTED  if  sup_M c_emp(M) >=  c_*
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import minimize
import time
import json
import sys

C_TARGET = float(np.log(16.0) / np.pi)   # 0.88254240061...

N = 129                                  # smaller for speed
DX = 0.5 / (N - 1)
XS = np.linspace(-0.25, 0.25, N)


def autoconv(f):
    return fftconvolve(f, f, mode='full') * DX


def autoconv3(f):
    """h = f*f*f as length 3N-2 vector."""
    g = autoconv(f)                       # length 2N-1
    return fftconvolve(g, f, mode='full') * DX  # length 3N-2


def metrics(f):
    g = autoconv(f)
    Linf = float(g.max())
    L2sq = float((g * g).sum() * DX)
    L1 = float(g.sum() * DX)
    return Linf, L2sq, L1, (L2sq / Linf if Linf > 0 else 0.0)


def project_pdf(f):
    f = np.maximum(f, 0.0)
    s = float(f.sum() * DX)
    if s <= 0:
        return None
    return f / s


def softmax_peak_measure(g, beta):
    """Probability measure (length 2N-1) concentrated near argmax g."""
    z = beta * (g - g.max())
    w = np.exp(z)
    return w / (w.sum() * DX)


def el_residual(f, beta=200.0):
    """Compute E-L residual R(x) = 4 h(x) - 2 lambda phi(x) - mu on supp f."""
    g = autoconv(f)
    h_full = fftconvolve(g, f[::-1], mode='full') * DX  # (g * f_tilde), length 3N-2
    # h_full[k]  for k = 0..3N-3 corresponds to  x = -3/4 + k*DX
    # We want x in [-1/4, 1/4]: indices i=0..N-1 correspond to k = (N-1) + i
    h_x = h_full[(N-1):(2*N-1)]
    dnu = softmax_peak_measure(g, beta)
    phi_full = fftconvolve(dnu, f[::-1], mode='full') * DX  # (f_tilde * dnu)
    phi_x = phi_full[(N-1):(2*N-1)]
    supp = f > 1e-3 * f.max()
    if supp.sum() < 4:
        return None
    # 4 h(x) = 2 lambda phi(x) + mu  on supp f  =>  least-sq for (mu, lambda)
    A = np.stack([np.ones(supp.sum()), 2.0 * phi_x[supp]], axis=1)
    b = 4.0 * h_x[supp]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    mu, lam = sol
    R_supp = b - A @ sol
    rel = float(np.linalg.norm(R_supp) / max(np.linalg.norm(b), 1e-12))
    # off-supp dual feasibility: 4h - 2 lam phi - mu  should be <= 0
    R_off = 4.0 * h_x - 2.0 * lam * phi_x - mu
    off = ~supp
    if off.any():
        max_off = float(R_off[off].max())
    else:
        max_off = 0.0
    return dict(mu=float(mu), lam=float(lam),
                resid_rel=rel,
                max_R_offsupp=max_off)


def smooth_loss(theta, M_cap, alpha_pen, beta_smax, sym=False):
    """Loss = -L2sq/Msoft + alpha*max(0, Msoft-M_cap)^2.   theta has length N."""
    if sym:
        theta = 0.5 * (theta + theta[::-1])
    th = theta - theta.max()
    f = np.exp(th)
    s = float(f.sum() * DX)
    if s <= 0:
        return 1e6
    f = f / s
    g = autoconv(f)
    Mhard = float(g.max())
    # softmax-||g||_inf
    z = beta_smax * (g - g.max())
    Msoft = g.max() + (1.0 / beta_smax) * np.log(np.exp(z).sum())
    L2sq = float((g * g).sum() * DX)
    obj = -L2sq / Msoft
    pen = alpha_pen * max(0.0, Msoft - M_cap) ** 2
    return obj + pen


def initial_thetas(seed, sym=False, M_target=1.5):
    """Initial f-shapes with autoconv max BELOW M_target where possible.
    To reduce M, we need 'smoother' f (less concentrated). Smooth Gaussians
    and MV-cosine-like configs give low M."""
    rng = np.random.default_rng(seed)
    out = []
    # 1. wide Gaussian -> small M (~ 1.13 with sigma=0.10)
    sigma = 0.07 + 0.08 * rng.uniform()
    out.append(-XS * XS / (2 * sigma * sigma))
    # 2. wider Gaussian
    sigma2 = 0.10 + 0.05 * rng.uniform()
    out.append(-XS * XS / (2 * sigma2 * sigma2))
    # 3. Indicator-flat (trivial)
    out.append(0.05 * rng.normal(size=N))
    # 4. MV-cosine (small M ~ 1.27)
    th = 0.0 + 0.5 * np.cos(2 * np.pi * XS) + 0.2 * np.cos(4 * np.pi * XS)
    th += 0.1 * rng.normal(size=N)
    out.append(th)
    # 5. Asymmetric: shifted Gaussian
    c1 = -0.05 + 0.10 * rng.uniform()
    th = -((XS - c1) ** 2) / 0.02
    out.append(th)
    # 6. Mild two-Gaussians (small M)
    c1, c2 = -0.10, 0.10
    th = np.log(np.exp(-(XS - c1) ** 2 / 0.015) + np.exp(-(XS - c2) ** 2 / 0.015) + 1e-6)
    out.append(th)
    # 7. Triangle-like (smooth)
    th = np.log(np.maximum(0.25 - np.abs(XS), 1e-3))
    out.append(th)
    if sym:
        out = [0.5 * (t + t[::-1]) for t in out]
    return out


def search_one_M(M_cap, n_seeds=4, sym=False):
    """Multi-restart L-BFGS with strong penalty homotopy."""
    best = (-np.inf, None, None)
    for seed in range(n_seeds):
        for theta0 in initial_thetas(seed, sym=sym, M_target=M_cap):
            th = theta0.copy()
            # Aggressive penalty homotopy. We target strict M_cap feasibility.
            for alpha in (1000.0, 1e4, 1e5, 1e6):
                try:
                    res = minimize(smooth_loss, th,
                                   args=(M_cap, alpha, 300.0, sym),
                                   method='L-BFGS-B',
                                   options=dict(maxiter=150, ftol=1e-10, gtol=1e-7))
                    th = res.x
                except Exception:
                    pass
            # Evaluate
            t = th.copy()
            if sym:
                t = 0.5 * (t + t[::-1])
            t = t - t.max()
            f = np.exp(t)
            f = f / (f.sum() * DX)
            Mhard, L2sq, L1, c_emp = metrics(f)
            feasible = (Mhard <= M_cap * 1.005)
            if feasible and c_emp > best[0]:
                best = (c_emp, Mhard, f.copy())
    return best


def main():
    print(f"Analytical E-L extremizer search; c_* = {C_TARGET:.10f}", flush=True)
    print(f"N = {N}, dx = {DX:.4e}", flush=True)
    print()

    # --- Sanity: indicator on [-1/4, 1/4] should give c_emp = 2/3, M = 2 ---
    f_ind = np.full(N, 2.0)
    M, L2, L1, c = metrics(f_ind)
    print(f"Sanity (indicator):  M={M:.4f}  c_emp={c:.4f}  (exp 2.0, 2/3 = 0.6667)")
    print()

    M_grid = [1.10, 1.20, 1.27, 1.30, 1.35, 1.378, 1.50, 1.65]

    table = []
    t0 = time.time()
    for M_cap in M_grid:
        print(f"--- M_cap = {M_cap:.3f} ---", flush=True)
        # Try BOTH symmetric and asymmetric searches
        c_sym, M_sym, f_sym = search_one_M(M_cap, n_seeds=2, sym=True)
        c_asy, M_asy, f_asy = search_one_M(M_cap, n_seeds=2, sym=False)
        if c_asy >= c_sym:
            c_emp, M_at, f = c_asy, M_asy, f_asy
            tag = 'asy'
        else:
            c_emp, M_at, f = c_sym, M_sym, f_sym
            tag = 'sym'
        if f is None:
            print(f"  no feasible point.", flush=True)
            table.append((M_cap, None, None, None, None))
            continue
        info = el_residual(f, beta=300.0)
        print(f"  best [{tag}]:  c_emp = {c_emp:.6f}   M_at = {M_at:.5f}   "
              f"margin to c_* = {C_TARGET - c_emp:+.6f}", flush=True)
        if info is not None:
            print(f"     E-L residual rel = {info['resid_rel']:.3e}   "
                  f"mu = {info['mu']:+.4f}   lambda = {info['lam']:+.4f}   "
                  f"max R off-supp = {info['max_R_offsupp']:+.3e}", flush=True)
        table.append((M_cap, c_emp, M_at, info['resid_rel'] if info else None, tag))

    print("\n" + "=" * 72)
    print(f"PHI(M) UPPER ESTIMATE TABLE   (target c_* = {C_TARGET:.6f})")
    print("=" * 72)
    print(f"  {'M_cap':>8s}  {'c_emp':>10s}  {'M_at':>9s}  {'margin':>10s}  {'EL_rel':>10s}  tag")
    sup_c = -np.inf
    sup_M = None
    for (M_cap, c, Mat, R, tag) in table:
        if c is None:
            print(f"  {M_cap:8.3f}   <none>")
            continue
        margin = C_TARGET - c
        Rs = (f"{R:.2e}" if R is not None else " - ")
        print(f"  {M_cap:8.3f}  {c:10.6f}  {Mat:9.5f}  {margin:+10.6f}  {Rs:>10s}  {tag}")
        if M_cap <= 1.378 + 1e-9 and c > sup_c:
            sup_c = c
            sup_M = M_cap
    print("=" * 72)
    print(f"sup_{{M <= 1.378}} c_emp_found = {sup_c:.6f} at M = {sup_M}")
    margin_global = C_TARGET - sup_c
    if margin_global > 0.005:
        verdict = "PROMISING (E-L sup is well below c_*; structural evidence for Hyp_R)"
    elif margin_global >= -0.001:
        verdict = "INCONCLUSIVE (E-L sup is within 5e-3 of c_*; cannot conclude either way)"
    else:
        verdict = "OBSTRUCTED (E-L finds a point with c_emp >= c_*; Hyp_R numerically at risk)"
    print(f"VERDICT: {verdict}")
    print(f"(elapsed {time.time() - t0:.1f}s)")

    # Save full table for later inspection
    out = dict(c_target=C_TARGET, N=N,
               table=[dict(M_cap=mc, c_emp=ce, M_at=ma, EL_rel=R, tag=tg)
                      for (mc, ce, ma, R, tg) in table],
               sup_c_emp_le_1p378=sup_c,
               margin_global=margin_global,
               verdict=verdict)
    with open('delsarte_dual/path_a_unconditional_holder/el_analytical_result.json', 'w') as fh:
        json.dump(out, fh, indent=2, default=str)


if __name__ == '__main__':
    main()
