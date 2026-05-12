"""
Focused numerical scan of  c_emp(f) = ||g||_2^2 / ||g||_inf  (g = f*f) over
admissible f with M = ||g||_inf <= 1.378.  Goal: determine sup c_emp.

KEY ANALYTICAL FACT (path_a Attack 1 + Lemma 2.14):
   c_emp(f) = ||g||_2^2 / M  <=  (1 + mu(M)(K - 1)) / M,  K := ||f||_2^2.

For c_emp >= c_* = log(16)/pi at M=1.378 (mu(M)=0.333), need K >= 1.65.
   For SYMMETRIC f, K <= M, so K <= 1.378  -> c_emp_bound = 0.817 < c_*.
   For ASYMMETRIC f, K can exceed M; the question is by HOW MUCH.

So the structural question reduces to:
       SUP { K : f admissible, ||f*f||_inf <= 1.378 } = ?

If sup K <= K* := 1 + (c_* * M - 1)/mu(M) - eps for any eps > 0, we win.

We probe this by SEARCHING for asymmetric admissible f with M <= 1.378 and
LARGE K. Three strategies:
   (S1) Direct optimization of K subject to M <= 1.378.
   (S2) Empirical Sidon-spike sweep (the §5.2 construction).
   (S3) Cosine-coefficient family with phase asymmetry.

Then we report the empirically-attained sup K, and the corresponding c_emp.
"""
from __future__ import annotations

import json, time, warnings, sys
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

C_STAR = float(np.log(16.0) / np.pi)
M_TARGET = 1.378
ROOT = Path(__file__).resolve().parent

# ---- helpers ----
def mu_M(M):
    return M * np.sin(np.pi / M) / np.pi

def c_emp_bound(M, K):
    """Path A (†): c_emp <= (1 + mu(M)(K-1))/M."""
    return (1 + mu_M(M)*(K - 1)) / M

def K_threshold(M, c_target):
    """K such that the (†) bound saturates c_target."""
    return 1.0 + (c_target * M - 1.0) / mu_M(M)

def metrics(f, dx):
    g = fftconvolve(f, f) * dx
    M = float(g.max())
    L2sq = float((g*g).sum() * dx)
    L1 = float(g.sum() * dx)
    K = float((f*f).sum() * dx)
    return dict(M=M, c_emp=L2sq/max(M,1e-18), K=K, L2sq=L2sq, L1=L1)

# ---- (S1) Direct K-maximization ----
N = 401
DX = 0.5/(N-1)
XS = np.linspace(-0.25, 0.25, N)

def f_from_theta(theta):
    th = theta - theta.max()
    f = np.exp(th)
    return f / max(f.sum()*DX, 1e-12)

def loss_K_with_M_cap(theta, M_cap, alpha):
    f = f_from_theta(theta)
    g = fftconvolve(f, f) * DX
    M = float(g.max())
    K = float((f*f).sum() * DX)
    pen = alpha * max(0.0, M - M_cap)**2
    return -K + pen   # maximize K

def search_K_max(M_cap, n_restarts=24, sym=False, seed_base=0):
    rng = np.random.default_rng(seed_base)
    best_K, best_M, best_f, best_c = -np.inf, None, None, None
    for r in range(n_restarts):
        flavor = r % 8
        if flavor == 0:
            th0 = rng.normal(0, 1, N)
        elif flavor == 1:
            sigma = 0.05 + 0.05*rng.uniform()
            th0 = -XS*XS/(2*sigma*sigma) + 0.1*rng.normal(size=N)
        elif flavor == 2:
            c1 = -0.20 + 0.04*rng.uniform()
            th0 = -((XS-c1)**2)/0.005 + 0.1*rng.normal(size=N)
        elif flavor == 3:
            # 2-spike asymmetric
            c1, c2 = -0.20+0.05*rng.uniform(), 0.05+0.10*rng.uniform()
            th0 = np.log(np.exp(-(XS-c1)**2/0.003) + 0.7*np.exp(-(XS-c2)**2/0.003) + 1e-3)
        elif flavor == 4:
            # 3-spike (Sidon-style)
            cs = sorted(rng.uniform(-0.2, 0.2, 3))
            th0 = np.log(sum(np.exp(-(XS-c)**2/0.002) for c in cs) + 1e-3)
        elif flavor == 5:
            th0 = np.where(XS > 0, 0.5, -0.5) + 0.1*rng.normal(size=N)
        elif flavor == 6:
            # bathtub
            th0 = np.where(np.abs(XS) > 0.20, 0.5, -0.5) + 0.1*rng.normal(size=N)
        else:
            # cosine-bumpy
            th0 = np.zeros(N)
            for j in range(1, 6):
                th0 += rng.normal()*np.cos(2*np.pi*j*XS)
        if sym:
            th0 = 0.5*(th0 + th0[::-1])
        # Penalty homotopy
        for alpha in (10.0, 100.0, 1000.0, 1e4, 1e5):
            res = minimize(loss_K_with_M_cap, th0, args=(M_cap, alpha),
                           method='L-BFGS-B', options=dict(maxiter=200, ftol=1e-10, gtol=1e-7))
            th0 = res.x
        f = f_from_theta(th0)
        if sym:
            f = 0.5*(f + f[::-1]); f = f / max(f.sum()*DX, 1e-12)
        m = metrics(f, DX)
        feas = (m['M'] <= M_cap * 1.005)
        if feas and m['K'] > best_K:
            best_K, best_M, best_f, best_c = m['K'], m['M'], f.copy(), m['c_emp']
    return best_K, best_M, best_f, best_c

# ---- (S1') Direct c_emp maximization ----
def loss_cemp_with_M_cap(theta, M_cap, alpha):
    f = f_from_theta(theta)
    g = fftconvolve(f, f) * DX
    M = float(g.max())
    L2sq = float((g*g).sum() * DX)
    pen = alpha * max(0.0, M - M_cap)**2
    return -L2sq/max(M,1e-12) + pen   # maximize c_emp

def search_cemp_max(M_cap, n_restarts=24, sym=False, seed_base=0):
    rng = np.random.default_rng(seed_base)
    best_c, best_M, best_f, best_K = -np.inf, None, None, None
    for r in range(n_restarts):
        flavor = r % 8
        if flavor == 0:
            th0 = rng.normal(0, 1, N)
        elif flavor == 1:
            sigma = 0.05 + 0.05*rng.uniform()
            th0 = -XS*XS/(2*sigma*sigma) + 0.1*rng.normal(size=N)
        elif flavor == 2:
            c1 = -0.20 + 0.04*rng.uniform()
            th0 = -((XS-c1)**2)/0.005 + 0.1*rng.normal(size=N)
        elif flavor == 3:
            c1, c2 = -0.20+0.05*rng.uniform(), 0.05+0.10*rng.uniform()
            th0 = np.log(np.exp(-(XS-c1)**2/0.003) + 0.7*np.exp(-(XS-c2)**2/0.003) + 1e-3)
        elif flavor == 4:
            cs = sorted(rng.uniform(-0.2, 0.2, 3))
            th0 = np.log(sum(np.exp(-(XS-c)**2/0.002) for c in cs) + 1e-3)
        elif flavor == 5:
            th0 = np.where(XS > 0, 0.5, -0.5) + 0.1*rng.normal(size=N)
        elif flavor == 6:
            th0 = np.where(np.abs(XS) > 0.20, 0.5, -0.5) + 0.1*rng.normal(size=N)
        else:
            th0 = np.zeros(N)
            for j in range(1, 6):
                th0 += rng.normal()*np.cos(2*np.pi*j*XS)
        if sym:
            th0 = 0.5*(th0 + th0[::-1])
        for alpha in (10.0, 100.0, 1000.0, 1e4, 1e5):
            res = minimize(loss_cemp_with_M_cap, th0, args=(M_cap, alpha),
                           method='L-BFGS-B', options=dict(maxiter=200, ftol=1e-10, gtol=1e-7))
            th0 = res.x
        f = f_from_theta(th0)
        if sym:
            f = 0.5*(f + f[::-1]); f = f / max(f.sum()*DX, 1e-12)
        m = metrics(f, DX)
        feas = (m['M'] <= M_cap * 1.005)
        if feas and m['c_emp'] > best_c:
            best_c, best_M, best_f, best_K = m['c_emp'], m['M'], f.copy(), m['K']
    return best_c, best_M, best_f, best_K

# ---- (S2) Sidon-spike sweep ----
def sidon_spike(positions, weights, N=N, sigma=0.002):
    f = np.zeros(N)
    for p, w in zip(positions, weights):
        f += w * np.exp(-(XS - p)**2 / (2*sigma*sigma))
    if f.sum() > 0:
        f = f / (f.sum() * DX)
    return f

def search_sidon_spikes(M_cap):
    """k-spike construction with various positions."""
    best_K, best_M, best_f, best_c = -np.inf, None, None, None
    rng = np.random.default_rng(2026)
    # k=2
    for p1 in np.linspace(-0.24, -0.05, 10):
        for p2 in np.linspace(0.05, 0.24, 10):
            for w_ratio in [1.0, 0.5, 2.0, 0.3, 3.0]:
                f = sidon_spike([p1, p2], [1.0, w_ratio])
                m = metrics(f, DX)
                if m['M'] <= M_cap * 1.005 and m['K'] > best_K:
                    best_K, best_M, best_f, best_c = m['K'], m['M'], f.copy(), m['c_emp']
    # k=3
    for trial in range(150):
        ps = sorted(rng.uniform(-0.24, 0.24, 3))
        ws = rng.uniform(0.3, 1.5, 3)
        sigma_t = 0.001 + 0.005 * rng.uniform()
        f = sidon_spike(ps, ws, sigma=sigma_t)
        m = metrics(f, DX)
        if m['M'] <= M_cap * 1.005 and m['K'] > best_K:
            best_K, best_M, best_f, best_c = m['K'], m['M'], f.copy(), m['c_emp']
    # k=4
    for trial in range(150):
        ps = sorted(rng.uniform(-0.24, 0.24, 4))
        ws = rng.uniform(0.3, 1.5, 4)
        sigma_t = 0.001 + 0.004 * rng.uniform()
        f = sidon_spike(ps, ws, sigma=sigma_t)
        m = metrics(f, DX)
        if m['M'] <= M_cap * 1.005 and m['K'] > best_K:
            best_K, best_M, best_f, best_c = m['K'], m['M'], f.copy(), m['c_emp']
    # k=5
    for trial in range(150):
        ps = sorted(rng.uniform(-0.24, 0.24, 5))
        ws = rng.uniform(0.3, 1.5, 5)
        sigma_t = 0.001 + 0.003 * rng.uniform()
        f = sidon_spike(ps, ws, sigma=sigma_t)
        m = metrics(f, DX)
        if m['M'] <= M_cap * 1.005 and m['K'] > best_K:
            best_K, best_M, best_f, best_c = m['K'], m['M'], f.copy(), m['c_emp']
    return best_K, best_M, best_f, best_c

# ---- (E-L residual) ----
def el_residual(f):
    g = fftconvolve(f, f) * DX
    M = float(g.max())
    Nf = f.size
    h_full = fftconvolve(g, f[::-1]) * DX
    h_x = h_full[Nf-1: 2*Nf-1]
    peak_thr = M - 1e-4 * max(M, 1)
    pm = (g >= peak_thr).astype(float)
    s = float(pm.sum() * DX)
    if s <= 0: return None
    pm = pm / s
    phi_full = fftconvolve(pm, f[::-1]) * DX
    phi_x = phi_full[Nf-1: 2*Nf-1]
    supp = f > 1e-3 * f.max()
    if supp.sum() < 4: return None
    A = np.stack([np.ones(supp.sum()), 2.0 * phi_x[supp]], axis=1)
    b = 4.0 * h_x[supp]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    mu_l, lam = sol
    R = b - A @ sol
    rel = float(np.linalg.norm(R) / max(np.linalg.norm(b), 1e-12))
    return dict(mu=float(mu_l), lam=float(lam), resid_rel=rel)


def main():
    import builtins
    _print = builtins.print
    def print(*a, **kw):
        kw.setdefault('flush', True); _print(*a, **kw)
    builtins.print = print
    print("=" * 76)
    print(f"Hyp_R E-L extremizer scan  (focused)        c_* = {C_STAR:.10f}")
    print(f"M_TARGET = {M_TARGET}, mu(M_TARGET) = {mu_M(M_TARGET):.6f}")
    K_thr_at_target = K_threshold(M_TARGET, C_STAR)
    print(f"K threshold for c_emp = c_*  (at M=1.378):  K* = {K_thr_at_target:.4f}")
    print(f"For sym f, K <= M = {M_TARGET} < K* = {K_thr_at_target:.3f}: c_emp_bound <= {c_emp_bound(M_TARGET, M_TARGET):.5f}")
    print("=" * 76)

    M_grid = [1.30, 1.378]    # focus on relevant regime

    summary = {}
    t0 = time.time()
    for Mc in M_grid:
        print(f"\n=== M_cap = {Mc:.3f} ===")
        K_thr = K_threshold(Mc, C_STAR)
        print(f"  K-threshold (need K >= {K_thr:.4f} for c_emp >= c_*)")

        # S1: max K
        Ks, Ms, fs, cs = search_K_max(Mc, n_restarts=6, sym=False, seed_base=int(Mc*1000))
        if fs is not None:
            print(f"  [S1 max K, asy]: K = {Ks:.4f}  M={Ms:.4f}  c_emp={cs:.4f}")
            print(f"     => K/K_thr = {Ks/K_thr:.3f},  c_emp_bound at this K = {c_emp_bound(Ms, Ks):.4f}")
        Ks2, Ms2, fs2, cs2 = search_K_max(Mc, n_restarts=4, sym=True, seed_base=int(Mc*1000)+11)
        if fs2 is not None:
            print(f"  [S1 max K, sym]: K = {Ks2:.4f}  M={Ms2:.4f}  c_emp={cs2:.4f}")

        # S1': max c_emp
        cm, Mm, fm, Km = search_cemp_max(Mc, n_restarts=6, sym=False, seed_base=int(Mc*1000)+99)
        if fm is not None:
            print(f"  [S1' max c, asy]: c_emp = {cm:.4f}  M={Mm:.4f}  K={Km:.4f}")
            info = el_residual(fm)
            if info: print(f"     E-L resid rel = {info['resid_rel']:.3e}")
        cm2, Mm2, fm2, Km2 = search_cemp_max(Mc, n_restarts=4, sym=True, seed_base=int(Mc*1000)+199)
        if fm2 is not None:
            print(f"  [S1' max c, sym]: c_emp = {cm2:.4f}  M={Mm2:.4f}  K={Km2:.4f}")

        # S2: Sidon spikes
        Ks3, Ms3, fs3, cs3 = search_sidon_spikes(Mc)
        if fs3 is not None:
            print(f"  [S2 Sidon spikes]: K={Ks3:.4f}  M={Ms3:.4f}  c_emp={cs3:.4f}")

        # Best c_emp across all strategies
        all_c = [(cs, Ms, 'S1asy', Ks), (cs2, Ms2, 'S1sym', Ks2),
                 (cm, Mm, "S1'asy", Km), (cm2, Mm2, "S1'sym", Km2),
                 (cs3, Ms3, 'S2spikes', Ks3)]
        all_c = [t for t in all_c if t[0] is not None and t[1] is not None
                 and not (isinstance(t[0], float) and np.isnan(t[0]))]
        if all_c:
            best = max(all_c, key=lambda t: t[0])
            bk = best[3] if best[3] is not None else float('nan')
            bm = best[1] if best[1] is not None else float('nan')
            print(f"  >>> BEST c_emp at M_cap={Mc}:  {best[0]:.4f}  via [{best[2]}]  (M={bm:.4f}, K={bk:.4f})")
            summary[Mc] = dict(best_c=best[0], best_K=bk, best_M=bm, best_via=best[2],
                               margin_to_cstar=C_STAR - best[0], K_threshold=K_thr,
                               K_over_threshold=(bk/K_thr) if bk and K_thr else None)
        else:
            summary[Mc] = dict(best_c=None, best_K=None, K_threshold=K_thr)

    print("\n" + "=" * 76)
    print(f"SUMMARY  (target c_* = {C_STAR:.6f})")
    print("=" * 76)
    print(f"  {'M_cap':>7s}  {'best c':>8s}  {'margin':>9s}  {'best K':>8s}  {'K_thr':>8s}  {'K/K_thr':>8s}")
    sup_c = -np.inf
    for Mc in M_grid:
        s = summary[Mc]
        if s['best_c'] is None:
            print(f"  {Mc:7.3f}  {'--':>8s}")
            continue
        sup_c = max(sup_c, s['best_c'])
        bk = s.get('best_K') or float('nan')
        kov = s.get('K_over_threshold') or float('nan')
        print(f"  {Mc:7.3f}  {s['best_c']:8.4f}  {s['margin_to_cstar']:+9.4f}  {bk:8.4f}  {s['K_threshold']:8.4f}  {kov:8.4f}")

    margin = C_STAR - sup_c
    print()
    print(f"sup c_emp over M_cap <= 1.378: {sup_c:.4f}  ->  margin to c_* = {margin:+.4f}")
    if margin > 0.01:
        verdict = "PROMISING"
        msg = f"All numerical extremizers in restricted regime have c_emp <= {sup_c:.4f}, well below c_* = {C_STAR:.4f}."
    elif margin >= -0.001:
        verdict = "INCONCLUSIVE"
        msg = f"Numerical sup c_emp is within 0.01 of c_*; cannot conclude either way."
    else:
        verdict = "OBSTRUCTED"
        msg = "Numerical extremizer found with c_emp >= c_* in restricted regime; Hyp_R numerically at risk."
    print(f"VERDICT: {verdict}  -- {msg}")
    print(f"\n(elapsed {time.time()-t0:.1f}s)")

    out = dict(c_target=C_STAR, M_target=M_TARGET, summary=summary,
               sup_c_emp_le_1p378=sup_c, margin=margin, verdict=verdict)
    with open(ROOT / "el_extremiser_test4_result.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
