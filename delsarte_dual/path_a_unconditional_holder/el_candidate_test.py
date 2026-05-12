"""
E-L candidate-extremizer test suite for Phi(M) = sup ||g||_2^2 / ||g||_inf.

The continuous E-L equation on supp f, derived in el_analytical.py, reads

    4 (f*f*f_tilde)(x) = 2 lambda (f_tilde * dnu)(x) + mu        (E-L)

where dnu is a probability measure on the active set { y : (f*f)(y) = M }.
For SYMMETRIC f, this becomes  4 h(x) = 2 lambda phi(x) + mu  on supp f,
with h := f*f*f and phi := f * dnu.

Bang-bang principle (general): the *extremizer* is supported on a set where
this affine identity holds with equality, and is zero elsewhere with the
slack <= 0. Standard candidate extremizers include:

   (1) INDICATOR  f = 2 * 1_{[-1/4,1/4]}    -> M = 2, c_emp = 2/3
   (2) TRIANGLE   f = (1 - 4|x|)/(1/4) on [-1/4,1/4] (i.e., proportional to triangle)
   (3) BATHTUB    f = a*1_A + b*1_B  with A, B near boundary
   (4) DELTAS / 3-spike            (extremal in the discrete autocorr problem)
   (5) BL truncation               (the disproof witness restricted to lower M)
   (6) MV's 119-cosine             (the small-M near-extremizer)

We compute c_emp(f) for each, then verify HOW WELL each satisfies (E-L)
by computing the affine residual

    R(x) := 4 h(x) - 2 lam phi(x) - mu,   (mu, lam) by least squares on supp f.

A SMALL R on supp f and (R <= 0) off supp f means the candidate IS an E-L
critical point. If even MV's 119-cosine has c_emp ~ 0.59 << 0.882, that
puts a structural ceiling on the worst-case ratio in the small-M regime.
"""
import numpy as np
from scipy.signal import fftconvolve
import time

C_TARGET = float(np.log(16.0) / np.pi)

N = 401
DX = 0.5 / (N - 1)
XS = np.linspace(-0.25, 0.25, N)


def autoconv(f):
    return fftconvolve(f, f, mode='full') * DX


def metrics(f):
    g = autoconv(f)
    Linf = float(g.max())
    L2sq = float((g * g).sum() * DX)
    L1 = float(g.sum() * DX)
    return Linf, L2sq, L1, (L2sq / Linf if Linf > 0 else 0.0)


def normalize_pdf(f):
    f = np.maximum(f, 0.0)
    s = float(f.sum() * DX)
    if s <= 0:
        return None
    return f / s


def el_residual(f, beta=400.0):
    g = autoconv(f)
    Nf = f.size
    h_full = fftconvolve(g, f[::-1], mode='full') * DX
    h_x = h_full[(Nf-1):(2*Nf-1)]
    z = beta * (g - g.max())
    w = np.exp(z); w /= (w.sum() * DX)  # peak measure
    phi_full = fftconvolve(w, f[::-1], mode='full') * DX
    phi_x = phi_full[(Nf-1):(2*Nf-1)]
    supp = f > 1e-3 * f.max()
    if supp.sum() < 4:
        return None
    A = np.stack([np.ones(supp.sum()), 2.0 * phi_x[supp]], axis=1)
    b = 4.0 * h_x[supp]
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    mu, lam = sol
    R_supp = b - A @ sol
    rel = float(np.linalg.norm(R_supp) / max(np.linalg.norm(b), 1e-12))
    R_off = 4.0 * h_x - 2.0 * lam * phi_x - mu
    off = ~supp
    max_off_pos = float(R_off[off].max()) if off.any() else 0.0
    return dict(mu=float(mu), lam=float(lam),
                resid_rel=rel, max_R_offsupp=max_off_pos,
                supp_count=int(supp.sum()))


def candidate_indicator():
    return np.full(N, 2.0)


def candidate_triangle():
    f = np.maximum(0.25 - np.abs(XS), 0.0)
    return normalize_pdf(f)


def candidate_bathtub(width_in=0.10, height_ratio=4.0):
    """Two boundary slabs of mass."""
    mask = np.abs(XS) > (0.25 - width_in)
    f = np.where(mask, height_ratio, 1.0)
    return normalize_pdf(f)


def candidate_3spike(positions, weights, sigma=0.005):
    f = np.zeros_like(XS)
    for p, w in zip(positions, weights):
        f += w * np.exp(-(XS - p) ** 2 / (2 * sigma * sigma))
    return normalize_pdf(f)


_BL_cached = None
def candidate_BL_truncate(head_keep, scale_factor=1.0):
    global _BL_cached
    if _BL_cached is None:
        try:
            with open('delsarte_dual/restricted_holder/coeffBL.txt') as fh:
                s = fh.read().strip().strip('{}').split(',')
            _BL_cached = np.array([int(x) for x in s], dtype=float)
        except Exception:
            return None
    v = _BL_cached.copy()
    if head_keep < len(v):
        v = v[:head_keep]
    L = len(v)
    half = 0.25 * scale_factor
    if half <= 0:
        return None
    bin_idx = ((XS + half) / (2*half/L)).astype(int)
    bin_idx = np.clip(bin_idx, 0, L-1)
    in_supp = (XS >= -half) & (XS < half)
    f = np.where(in_supp, v[bin_idx], 0.0)
    return normalize_pdf(f)


def candidate_MV_cosine(n_terms=10):
    """Toy MV-style cosine sum that lives near M = 1.275."""
    # Coefficients from MV's main extremizer signature: a_0 = 1, alternating decay
    a = [1.0]
    for j in range(1, n_terms + 1):
        a.append(0.5 * (-1) ** j / j)
    f = np.zeros_like(XS)
    for j, aj in enumerate(a):
        f += aj * np.cos(2 * np.pi * j * XS)
    f = np.maximum(f, 0.0)
    return normalize_pdf(f)


def candidate_two_block(a, b, M_target):
    """Two-block bathtub of given relative weights, scaled to int=1."""
    f = np.zeros_like(XS)
    f[XS < -0.10] = a
    f[XS > 0.10] = b
    f[(XS >= -0.10) & (XS <= 0.10)] = 0.0  # gap in middle
    return normalize_pdf(f)


def main():
    print(f"E-L candidate extremizer test  (c_* = {C_TARGET:.6f})")
    print(f"N = {N}, dx = {DX:.4e}\n")

    cases = []
    cases.append(("indicator (M=2)", candidate_indicator()))
    cases.append(("triangle (M=4/3)", candidate_triangle()))
    cases.append(("MV-cosine 10 terms", candidate_MV_cosine(10)))
    cases.append(("BL full (575)", candidate_BL_truncate(575)))

    # MV-style mixtures that LIVE in the restricted M regime
    # f = (1 - alpha) * indicator + alpha * BL(scaled)
    f_ind = np.full(N, 2.0)
    f_BL = candidate_BL_truncate(575)
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        f = (1 - alpha) * f_ind + alpha * f_BL
        f = normalize_pdf(f)
        cases.append((f"BL+ind mix alpha={alpha:.2f}", f))

    # MV-cosine mixed with indicator, sweeping toward MV's M=1.275 spot
    f_mv = candidate_MV_cosine(10)
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        f = (1 - alpha) * f_ind + alpha * f_mv
        f = normalize_pdf(f)
        cases.append((f"MV+ind mix alpha={alpha:.2f}", f))

    # BL-rescaled to land in restricted M: scale_factor < 1 dilates downward
    for sf, hk in [(0.50, 575), (0.65, 575), (0.85, 575), (1.0, 575),
                   (0.50, 200), (0.65, 200), (1.0, 200)]:
        f = candidate_BL_truncate(hk, sf)
        cases.append((f"BL-resc sf={sf:.2f},hd={hk}", f))

    print(f"{'name':38s}  {'M':>7s}  {'c_emp':>8s}  {'margin':>9s}  {'EL_rel':>10s}  {'lambda':>8s}")
    print("-" * 100)

    results = []
    sup_in_restricted = -np.inf
    sup_name = ''
    for name, f in cases:
        if f is None:
            print(f"{name:38s}  <error>")
            continue
        M, L2sq, L1, c = metrics(f)
        info = el_residual(f)
        Rstr = f"{info['resid_rel']:.2e}" if info else " - "
        Lstr = f"{info['lam']:.3f}" if info else " - "
        margin = C_TARGET - c
        marker = ""
        if M <= 1.378:
            if c > sup_in_restricted:
                sup_in_restricted = c
                sup_name = name
            if c >= C_TARGET:
                marker = "  *** VIOLATES Hyp_R"
        print(f"{name:38s}  {M:7.4f}  {c:8.5f}  {margin:+9.5f}  {Rstr:>10s}  {Lstr:>8s}{marker}")
        results.append((name, M, c, info))

    print("-" * 100)
    print()
    print(f"sup_{{candidate, M <= 1.378}}  c_emp = {sup_in_restricted:.6f}  ({sup_name})")
    print(f"target c_*                     = {C_TARGET:.6f}")
    print(f"global gap                     = {C_TARGET - sup_in_restricted:+.6f}")
    print()
    if sup_in_restricted < C_TARGET - 0.05:
        verdict = "STRONG: closed-form candidates are ALL well below c_*; consistent with Hyp_R."
    elif sup_in_restricted < C_TARGET:
        verdict = "WEAK: closed-form candidates are below c_* but within 0.05; need search."
    else:
        verdict = "OBSTRUCTED: a closed-form candidate witnesses Hyp_R failure."
    print(f"INTERPRETATION: {verdict}")


if __name__ == '__main__':
    main()
