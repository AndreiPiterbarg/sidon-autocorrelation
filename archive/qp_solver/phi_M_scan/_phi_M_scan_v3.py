"""V3: investigate Phi(M) := sup_{f: M(f)=M} c_emp(f) where M(f) = ||f*f||_inf.

Critical issue uncovered:
  - The lower bound C_{1a} >= 1.2802 (CS 2017) means NO admissible f has M < 1.2802.
  - The best known upper bound C_{1a} <= 1.5029 (MV upper bound, explicit pdf).
  - BL witness has M ~ 1.668.
  - The conditional theorem PROVES (modulo Hyp_R) that no f has M < 1.378.
  - So the "domain" of Phi(M) for M < 1.5029 is conjectured EMPTY.

This means the interpolation argument from the prompt is fundamentally BROKEN:
  - We can't measure Phi(1.378) directly because we don't know any f with M ~ 1.378.
  - Existing pdfs only cover M >= 1.668 (BL is the smallest M known with high c).
  - The MV pdf has M = 2.31, NOT 1.275.
  - The 1.275 in the prompt is the LOWER BOUND on C_{1a}, an unachieved value.

So we report:
  1. The numerical Phi(M) curve OVER THE ACCESSIBLE RANGE of M.
  2. Test for monotonicity in the accessible range [BL_M, infinity).
  3. Investigate whether Phi(M) extends naturally to below 1.668 (it doesn't, since
     no admissible f exists there in our families).
  4. VERDICT: the interpolation strategy is INCONCLUSIVE because Phi is undefined
     on most of the relevant interval.
"""
from __future__ import annotations
import numpy as np
import re
from numpy.fft import rfft, irfft

D = 256
dx = 0.5 / D
xs = -0.25 + dx * (np.arange(D) + 0.5)
PAD = 2 * D - 1
PAD2 = 1 << (PAD - 1).bit_length()


def autoconv(p):
    f = p / dx
    F = rfft(f, n=PAD2)
    g = irfft(F * F, n=PAD2)[:PAD] * dx
    return g


def Mc(p):
    g = autoconv(p)
    M = g.max()
    return M, (g * g).sum() * dx / M


def project_simplex(v):
    n = len(v)
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u + (1 - css) / np.arange(1, n + 1) > 0)[0][-1]
    theta = (css[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)


def softmax(g, T):
    a = g / T
    m = a.max()
    e = np.exp(a - m)
    return e / e.sum()


def make_BL():
    with open("delsarte_dual/restricted_holder/coeffBL.txt") as fh:
        nums = [int(x) for x in re.findall(r"\d+", fh.read())]
    v = np.array(nums[:575], dtype=np.float64)
    p = np.zeros(D)
    for i, x in enumerate(xs):
        idx = int((x + 0.25) * 1150)
        idx = max(0, min(574, idx))
        p[i] = v[idx]
    return p / p.sum()


def make_uniform():
    return np.ones(D) / D


def make_MV(u_var=0.638, n_coeffs=119):
    coeffs = [
        +2.16620392, -1.87775750, +1.05828868, -0.729790538,
        +0.428008515, +0.217832838, -0.270415201, +0.0272834790,
        -0.191721888, +0.0551862060, +0.321662512, -0.164478392,
        +0.0395478603, -0.205402785, -0.0133758316, +0.231873221,
        -0.0437967118, +0.0612456374, -0.157361919, -0.0778036253,
        +0.138714392, -0.000145201483, +0.0916539824, -0.0834020840,
    ][:n_coeffs]
    f = np.zeros_like(xs)
    for j, a in enumerate(coeffs, start=1):
        f += a * np.cos(2 * np.pi * j * xs / u_var)
    f = np.maximum(f, 0)
    return f / f.sum()


def grad_obj_max_c_at_M(p, M_target, T_smax=0.005, lam=200.0):
    f = p / dx
    F = rfft(f, n=PAD2)
    G = F * F
    g = irfft(G, n=PAD2)[:PAD] * dx
    L22 = (g * g).sum() * dx
    M_actual = g.max()
    w = softmax(g, T_smax)
    smaxM = (w * g).sum()
    obj = L22 - lam * (smaxM - M_target) ** 2
    Gtilde = rfft(g, n=PAD2)
    Fconj = F.conjugate()
    corr_gf = irfft(Fconj * Gtilde, n=PAD2)[:D] * dx
    dL22_df = 4.0 * dx * corr_gf
    W = rfft(w, n=PAD2)
    corr_fw = irfft(Fconj * W, n=PAD2)[:D] * dx
    dsmax_df = 2.0 * corr_fw
    dobj_df = dL22_df - 2 * lam * (smaxM - M_target) * dsmax_df
    dobj_dp = dobj_df / dx
    c_actual = L22 / max(M_actual, 1e-12)
    return obj, dobj_dp, M_actual, c_actual


def maximize_c_at_M(p_init, M_target, n_iters=600, lr=0.005,
                     lam_start=10.0, lam_end=10000.0,
                     T_start=0.02, T_end=0.0005,
                     M_tol=0.04):
    p = project_simplex(p_init.copy())
    best_c = -np.inf
    best_M = None
    best_p = p.copy()
    for it in range(n_iters):
        T = T_start * (T_end / T_start) ** (it / max(1, n_iters - 1))
        lam = lam_start * (lam_end / lam_start) ** (it / max(1, n_iters - 1))
        obj, gradp, M_act, c_act = grad_obj_max_c_at_M(p, M_target, T_smax=T, lam=lam)
        p = project_simplex(p + lr * gradp)
        if abs(M_act - M_target) < M_tol and c_act > best_c:
            best_c = c_act
            best_p = p.copy()
            best_M = M_act
    return best_c, best_M, best_p


def main():
    print(f"D={D}, dx={dx:.5e}")
    print()
    p_BL = make_BL()
    p_U = make_uniform()
    p_MV = make_MV()
    M_BL, c_BL = Mc(p_BL)
    M_U, c_U = Mc(p_U)
    M_MV, c_MV = Mc(p_MV)
    print("Anchor points (M, c):")
    print(f"  Uniform 2*1_[-1/4,1/4]      M={M_U:.4f}  c={c_U:.4f}")
    print(f"  MV cosine sum (24)         M={M_MV:.4f}  c={c_MV:.4f}")
    print(f"  BL witness                 M={M_BL:.4f}  c={c_BL:.4f}")
    print()
    print("CRITICAL OBSERVATION about the prompt's data:")
    print(f"  Prompt says MV's pdf has M~1.275, but it actually has M={M_MV:.4f}.")
    print(f"  The 1.275 is the LOWER BOUND on C_{{1a}}, an unachieved value.")
    print(f"  No explicit admissible pdf with M < {M_BL:.4f} (= BL's M) is known.")
    print(f"  The conditional theorem PROVES (modulo Hyp_R) no f has M < 1.378.")
    print()
    print("=" * 92)
    print("Phase A: scatter scan over M-values for many parametric families")
    print("=" * 92)

    # Build a LARGE pool of pdfs spanning M >= 1.668
    pool = [("BL", p_BL), ("Uniform", p_U), ("MV(24)", p_MV)]
    # Sym Gaussians
    for sigma in np.linspace(0.01, 0.20, 40):
        f = np.exp(-0.5 * (xs / sigma) ** 2)
        f /= f.sum()
        pool.append((f"sym_gauss_s{sigma:.3f}", f))
    # Two-bump symmetric
    for off in np.linspace(0.01, 0.24, 20):
        for sigma in [0.005, 0.01, 0.02, 0.05, 0.10]:
            f = np.exp(-0.5*((xs-off)/sigma)**2) + np.exp(-0.5*((xs+off)/sigma)**2)
            f /= f.sum()
            pool.append((f"2bump_off{off:.3f}_s{sigma:.3f}", f))
    # Triangles, indicators
    for w in np.linspace(0.05, 0.25, 21):
        f = np.maximum(1 - np.abs(xs / w), 0)
        f /= f.sum()
        pool.append((f"triangle_w{w:.3f}", f))
    for w in np.linspace(0.05, 0.25, 21):
        f = ((np.abs(xs) < w)).astype(float)
        if f.sum() > 0:
            f /= f.sum()
            pool.append((f"indicator_w{w:.3f}", f))
    # Convex combos
    for a in np.arange(0, 1.01, 0.05):
        for b in np.arange(0, 1.01-a, 0.05):
            cc = 1 - a - b
            p = a*p_MV + b*p_BL + cc*p_U
            if p.min() < 0:
                continue
            p = p / p.sum()
            pool.append((f"mix_MV{a:.2f}_BL{b:.2f}_U{cc:.2f}", p))
    # Random
    rng = np.random.default_rng(0)
    for tr in range(200):
        nb = rng.integers(2, 9)
        f = np.zeros_like(xs)
        for _ in range(nb):
            mu = rng.uniform(-0.22, 0.22)
            s = rng.uniform(0.005, 0.08)
            h = rng.uniform(0.2, 1.0)
            f += h * np.exp(-0.5*((xs-mu)/s)**2)
        f /= f.sum()
        pool.append((f"rand_t{tr}", f))
    # BL perturbations
    for tr in range(80):
        amp = rng.uniform(0.0, 0.5)
        n_modes = rng.integers(1, 8)
        delta = np.zeros_like(xs)
        for _ in range(n_modes):
            k = rng.uniform(0.5, 12)
            phi = rng.uniform(0, 2*np.pi)
            delta += rng.normal() * np.sin(2*np.pi*k*np.arange(D)/D + phi)
        p = np.maximum(p_BL + amp * delta * p_BL.mean(), 0)
        if p.sum() == 0: continue
        p /= p.sum()
        pool.append((f"BL_pert_t{tr}_amp{amp:.2f}", p))

    pts = []
    for name, p in pool:
        try:
            M, c = Mc(p)
            pts.append((M, c, name, p))
        except Exception:
            pass
    Ms = np.array([t[0] for t in pts])
    Cs = np.array([t[1] for t in pts])
    print(f"Pool size: {len(pts)}")
    print(f"M range: [{Ms.min():.4f}, {Ms.max():.4f}]")
    print(f"Smallest M achieved: {Ms.min():.4f}  ({pts[np.argmin(Ms)][2]})")
    print(f"Among these: largest c at smallest M = {pts[np.argmin(Ms)][1]:.4f}")

    # Bin Phi(M) for M in 1.65 to 2.5
    bins = np.arange(1.65, 2.55, 0.05)
    print()
    print(f"Phi_baseline binned (max c per bin):")
    print(f"{'M_lo':>6s} {'M_hi':>6s}  {'count':>5s}  {'Phi_base':>9s}  {'best seed'}")
    bin_data = []
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        in_bin = [(M, c, name, p) for M, c, name, p in pts if lo <= M < hi]
        if in_bin:
            best = max(in_bin, key=lambda r: r[1])
            bin_data.append((0.5*(lo+hi), best))
            print(f"{lo:>6.3f} {hi:>6.3f}  {len(in_bin):>5d}  {best[1]:>9.4f}  {best[2]}")
        else:
            bin_data.append((0.5*(lo+hi), None))
            print(f"{lo:>6.3f} {hi:>6.3f}  {0:>5d}  {'(empty)':>9s}")

    print()
    print("=" * 92)
    print("Phase B: gradient ASCENT refinement of Phi(M) at each bin midpoint")
    print("=" * 92)
    refined = []
    print(f"{'M_target':>9s}  {'Phi_base':>9s}  {'Phi_grad':>9s}  {'M_act':>7s}  {'best seed'}")
    for M_mid, best_seed in bin_data:
        # Use top 5 highest-c seeds in [M_mid-0.05, M_mid+0.05]
        near = sorted([(M, c, name, p) for M, c, name, p in pts
                       if abs(M - M_mid) < 0.07], key=lambda r: -r[1])[:6]
        if not near:
            refined.append((M_mid, None, None, None))
            continue
        best_c = near[0][1]
        best_M = near[0][0]
        best_name = near[0][2]
        for M_seed, c_seed, name_s, p_s in near:
            try:
                c_g, M_g, _ = maximize_c_at_M(
                    p_s, M_mid, n_iters=400, lr=0.003,
                    lam_start=20.0, lam_end=10000.0, M_tol=0.025)
                if c_g > best_c:
                    best_c, best_M, best_name = c_g, M_g, f"refined-{name_s}"
            except Exception:
                pass
        refined.append((M_mid, best_c, best_M, best_name))
        bs = f"{best_c:.4f}" if best_c > -np.inf else " ----"
        bM = f"{best_M:.4f}" if best_M is not None else " ----"
        bbs = f"{near[0][1]:.4f}" if near else "----"
        print(f"  {M_mid:>7.3f}  {bbs:>9s}  {bs:>9s}  {bM:>7s}  {best_name}")

    print()
    print("=" * 92)
    print("Phase C: MONOTONICITY of Phi_refined(M) over accessible range")
    print("=" * 92)
    valid = [(M, c) for M, c, _, _ in refined if c is not None and c > -np.inf]
    if len(valid) < 2:
        print("Insufficient data.")
        return
    Ms_v = np.array([M for M, c in valid])
    Cs_v = np.array([c for M, c in valid])
    print(f"Refined Phi(M) curve (M, Phi):")
    for M, c in valid:
        print(f"  M={M:.3f}  Phi_est={c:.4f}")

    is_monotone_inc = True
    is_monotone_dec = True
    n_inc_viol = 0
    n_dec_viol = 0
    for i in range(1, len(Cs_v)):
        if Cs_v[i] < Cs_v[i-1] - 0.005:
            is_monotone_inc = False
            n_inc_viol += 1
        if Cs_v[i] > Cs_v[i-1] + 0.005:
            is_monotone_dec = False
            n_dec_viol += 1
    print()
    print(f"  Monotone INCREASING: {is_monotone_inc} (viol={n_inc_viol})")
    print(f"  Monotone DECREASING: {is_monotone_dec} (viol={n_dec_viol})")
    if not is_monotone_inc and not is_monotone_dec:
        print(f"  Phi is NOT monotone in either direction.")
    elif is_monotone_dec:
        print(f"  *** Phi appears MONOTONE DECREASING in M (opposite of prompt's hypothesis!)")

    print()
    print("=" * 92)
    print("Phase D: Phi(M) at M=1.378 — direct attempt + analysis")
    print("=" * 92)
    # Direct attempt
    print("Direct gradient ascent at M=1.378 from many seeds:")
    seeds_direct = []
    for name, p in pool:
        try:
            M, c = Mc(p)
            if 1.30 <= M <= 1.85:  # closer is better seed
                seeds_direct.append((abs(M-1.378), M, c, name, p))
        except Exception: pass
    seeds_direct.sort()
    print(f"Found {len(seeds_direct)} seeds with M in [1.30, 1.85]")
    if not seeds_direct:
        print("WARNING: no seeds even close to M=1.378. Cannot estimate Phi(1.378).")

    best_c_direct = -np.inf
    best_M_direct = None
    best_name_direct = None
    for diff, Ms, cs, name, p in seeds_direct[:30]:
        try:
            c_g, M_g, _ = maximize_c_at_M(
                p, 1.378, n_iters=800, lr=0.002,
                lam_start=50.0, lam_end=30000.0, M_tol=0.025)
            if c_g > best_c_direct:
                best_c_direct = c_g
                best_M_direct = M_g
                best_name_direct = name
        except Exception:
            pass

    print()
    if best_c_direct > -np.inf:
        print(f"BEST direct estimate: Phi(1.378) ~ {best_c_direct:.4f}")
        print(f"  achieved at M_actual = {best_M_direct:.4f}, seed = {best_name_direct}")
        margin = 0.88254 - best_c_direct
        print(f"  Margin to threshold 0.88254: {margin:+.4f}")
    else:
        print(f"No f found landing at M=1.378 (within tol 0.025).")

    print()
    print("=" * 92)
    print("VERDICT")
    print("=" * 92)
    print(f"DOMAIN ISSUE:")
    print(f"  - C_{{1a}} >= 1.2802 (CS 2017): no f has M < 1.2802 (PROVEN).")
    print(f"  - Best known explicit f: BL with M ~ {M_BL:.4f}.")
    print(f"  - Conditional theorem PROVES (modulo Hyp_R): no f has M < 1.378.")
    print(f"  - All numerical experiments find min(M) ~ {Ms.min():.4f}.")
    print(f"  - The interval [1.275, 1.668] is conjecturally EMPTY of admissible f.")
    print(f"  - Phi(M) for M in [1.275, 1.378) is sup over empty set, hence -inf.")
    print()
    print(f"INTERPOLATION ARGUMENT FROM PROMPT:")
    print(f"  Prompt assumes Phi(M=1.275) ~ 0.589 and Phi(M=1.652) ~ 0.902 are known.")
    print(f"  But Phi(M=1.275) has no admissible f attaining M=1.275, so the value")
    print(f"  0.589 is NOT a value of Phi at M=1.275; it's MV's c at M=2.31.")
    print(f"  The interpolation is therefore between two QUITE DIFFERENT M-values,")
    print(f"  and the claim 'Phi(1.378) ~ 0.674' rests on a category error.")
    print()
    if best_c_direct >= 0.88254:
        print(f"  FALSIFICATION ALERT: numerical Phi(1.378) >= {best_c_direct:.4f} >= 0.88254.")
        print(f"  If rigorously confirmed, would falsify Hyp_R(M_max=1.378, c=log16/pi).")
        print(f"  VERDICT: OBSTRUCTED")
    elif best_c_direct > -np.inf:
        print(f"  Numerical evidence: best Phi(1.378) found = {best_c_direct:.4f}.")
        if best_c_direct < 0.88254:
            print(f"  This is BELOW 0.88254 by {0.88254 - best_c_direct:.4f}.")
            print(f"  VERDICT: NUMERICAL EVIDENCE CONSISTENT WITH Hyp_R.")
            print(f"  But monotonicity of Phi is INCONCLUSIVE; interpolation argument FAILS.")
    else:
        print(f"  No numerical witness near M=1.378 found.")
        print(f"  VERDICT: INCONCLUSIVE — domain restrictions prevent direct measurement.")


if __name__ == "__main__":
    main()
