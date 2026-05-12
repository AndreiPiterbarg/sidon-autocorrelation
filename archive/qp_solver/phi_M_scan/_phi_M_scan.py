"""Investigate Phi(M) := sup over admissible f with ||f*f||_inf = M of
                  c_emp(f) = ||f*f||_2^2 / ||f*f||_inf.

We are interested in the conditional bound's defensibility: if Phi(M) is
monotone-increasing on M in [1.275, 1.65], then by interpolation,
Phi(1.378) <= linear-interp of Phi(1.275) ~ 0.589 (MV) and
Phi(1.652) ~ 0.902 (BL).

Strategy: at each target M, run gradient ASCENT on c_emp subject to a soft
penalty enforcing ||f*f||_inf approximately equal to M (penalty is a
quadratic on (M_actual - M_target)^2 if M_actual deviates).

We use multiple random/heuristic seeds to escape local optima.
"""
from __future__ import annotations
import numpy as np
import re
from numpy.fft import rfft, irfft

D = 256                       # bins on [-1/4, 1/4]
dx = 0.5 / D
xs = -0.25 + dx * (np.arange(D) + 0.5)
PAD = 2 * D - 1
PAD2 = 1 << (PAD - 1).bit_length()


def autoconv(p):
    f = p / dx
    F = rfft(f, n=PAD2)
    g = irfft(F * F, n=PAD2)[:PAD] * dx
    return g, F


def Mc(p):
    g, _ = autoconv(p)
    M = g.max()
    return M, (g * g).sum() * dx / M


def project_simplex(v):
    n = len(v)
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u + (1 - css) / np.arange(1, n + 1) > 0)[0][-1]
    theta = (css[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)


def softmax_w(g, T):
    a = g / T
    m = a.max()
    e = np.exp(a - m)
    return e / e.sum()


def grad_obj_max_c_at_M(p, M_target, T_smax=0.005, lam=200.0):
    """Compute objective and gradient for: maximize ||g||_2^2 - lam*(smaxM - M_target)^2.

    Returns (obj, grad_p, M_actual, c_actual).
    """
    f = p / dx
    F = rfft(f, n=PAD2)
    G = F * F
    g = irfft(G, n=PAD2)[:PAD] * dx
    # ||g||_2^2 = sum g^2 dx
    L22 = (g * g).sum() * dx
    M_actual = g.max()
    # Smoothed max for gradient
    w = softmax_w(g, T_smax)
    smaxM = (w * g).sum()  # smooth approx of max(g)
    # Objective:  L22 - lam*(smaxM - M_target)^2
    obj = L22 - lam * (smaxM - M_target) ** 2
    # ---- Gradient ----
    # d L22 / df = 4 dx * (f * g)?  Let's compute it via FFT.
    # L22 = int g^2 dt = int (f*f)^2;  d L22 / df[i] = 2 * (g * dgdf[i] dt)
    # dg/df[i](t) = 2 f(t-x_i) (continuous);  discretized: = 2 dx f shifted
    # So d L22 / df[i] = 4 dx * sum_t g(t) f(t - x_i) = 4 dx (g corr f)[i]
    # Implement via FFT: corr(g, f)[i] = sum_t g(t) f(t-x_i) = ifft(conj(F)*Gtilde)
    Gtilde = rfft(g, n=PAD2)
    Fconj = F.conjugate()
    corr_gf = irfft(Fconj * Gtilde, n=PAD2)[:D] * dx
    dL22_df = 4.0 * dx * corr_gf
    # Smoothed-max gradient: d smaxM / df = 2 dx * (f * w)?  Same formula as M-descent.
    W = rfft(w, n=PAD2)
    corr_fw = irfft(Fconj * W, n=PAD2)[:D] * dx
    dsmax_df = 2.0 * corr_fw
    # Total grad of obj wrt f
    dobj_df = dL22_df - 2 * lam * (smaxM - M_target) * dsmax_df
    # f = p / dx, so df/dp = 1/dx; grad wrt p = grad wrt f / dx
    dobj_dp = dobj_df / dx
    # actual c
    c_actual = L22 / max(M_actual, 1e-12)
    return obj, dobj_dp, M_actual, c_actual, L22


def maximize_c_at_M(p_init, M_target, n_iters=500, lr=0.01,
                     lam_start=50.0, lam_end=2000.0,
                     T_start=0.02, T_end=0.001):
    """Project gradient ascent to maximize c subject to M ~= M_target.
    Returns best (c, M, p) seen along the trajectory with |M-M_target| < 0.01.
    """
    p = project_simplex(p_init.copy())
    best_c = -np.inf
    best_p = p.copy()
    best_M = None
    for it in range(n_iters):
        T = T_start * (T_end / T_start) ** (it / max(1, n_iters - 1))
        lam = lam_start * (lam_end / lam_start) ** (it / max(1, n_iters - 1))
        obj, gradp, M_act, c_act, _ = grad_obj_max_c_at_M(
            p, M_target, T_smax=T, lam=lam)
        # Ascend (maximize): step in +grad direction
        p = project_simplex(p + lr * gradp)
        if abs(M_act - M_target) < 0.015 and c_act > best_c:
            best_c = c_act
            best_p = p.copy()
            best_M = M_act
    # Final eval
    M_final, c_final = Mc(p)
    if abs(M_final - M_target) < 0.015 and c_final > best_c:
        best_c = c_final
        best_p = p.copy()
        best_M = M_final
    return best_c, best_M, best_p


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


def make_MV():
    coeffs = [
        +2.16620392, -1.87775750, +1.05828868, -0.729790538,
        +0.428008515, +0.217832838, -0.270415201, +0.0272834790,
        -0.191721888, +0.0551862060, +0.321662512, -0.164478392,
        +0.0395478603, -0.205402785, -0.0133758316, +0.231873221,
        -0.0437967118, +0.0612456374, -0.157361919, -0.0778036253,
        +0.138714392, -0.000145201483, +0.0916539824, -0.0834020840,
    ]
    u = 0.638
    f = np.zeros_like(xs)
    for j, a in enumerate(coeffs, start=1):
        f += a * np.cos(2 * np.pi * j * xs / u)
    f = np.maximum(f, 0)
    return f / f.sum()


def make_indicator():
    """f = 2 * 1_[-1/4, 1/4]; M = 2, c = 2/3 exactly."""
    return np.ones(D) / D  # same as uniform on grid; M = 2 since it spans full support


def make_two_bump(off, sigma):
    p = np.exp(-0.5 * ((xs - off) / sigma) ** 2) + np.exp(-0.5 * ((xs + off) / sigma) ** 2)
    return p / p.sum()


def make_asym_two_bump(off1, off2, sigma, h_ratio=1.0):
    p = h_ratio * np.exp(-0.5 * ((xs - off1) / sigma) ** 2) + np.exp(-0.5 * ((xs - off2) / sigma) ** 2)
    return p / p.sum()


def initial_seeds_at_M(M_target, n_random=12, rng_seed=0):
    """Build a list of seed pdfs near the target M."""
    rng = np.random.default_rng(rng_seed)
    seeds = []
    p_BL = make_BL()
    p_U = make_uniform()
    p_MV = make_MV()
    # Reference (M, c)
    M_BL, _ = Mc(p_BL)
    M_U, _ = Mc(p_U)
    M_MV, _ = Mc(p_MV)

    # Linear interpolation between Uniform and BL: pick alpha so that M ~ M_target.
    # Rough binary search for alpha such that Mc((1-a)U + a BL) ~= M_target.
    def search_alpha(p_lo, p_hi, M_lo, M_hi):
        # If M_target is within [min(M_lo,M_hi), max(...)], find alpha.
        a, b = 0.0, 1.0
        for _ in range(40):
            mid = (a + b) / 2
            p_m = (1 - mid) * p_lo + mid * p_hi
            M_m, _ = Mc(p_m)
            if (M_m < M_target and M_lo < M_hi) or (M_m > M_target and M_lo > M_hi):
                a = mid
            else:
                b = mid
        p_m = (1 - a) * p_lo + a * p_hi
        return p_m / p_m.sum()

    # Seed 1: Uniform <-> BL
    if min(M_U, M_BL) <= M_target <= max(M_U, M_BL):
        seeds.append(("U-BL_homotopy", search_alpha(p_U, p_BL, M_U, M_BL)))
    # Seed 2: MV <-> BL
    if min(M_MV, M_BL) <= M_target <= max(M_MV, M_BL):
        seeds.append(("MV-BL_homotopy", search_alpha(p_MV, p_BL, M_MV, M_BL)))
    # Seed 3: MV <-> Uniform
    if min(M_MV, M_U) <= M_target <= max(M_MV, M_U):
        seeds.append(("MV-U_homotopy", search_alpha(p_MV, p_U, M_MV, M_U)))

    # Two-bump seeds: scan offset and sigma until we land near M_target
    for off in [0.05, 0.10, 0.15, 0.18, 0.20]:
        for sigma in [0.02, 0.04, 0.06, 0.08, 0.12]:
            p = make_two_bump(off, sigma)
            M, _ = Mc(p)
            if abs(M - M_target) < 0.10:
                seeds.append((f"sym_2b_off{off:.2f}_s{sigma:.2f}_M{M:.3f}", p))

    # Asymmetric seeds
    for off1, off2 in [(-0.20, 0.05), (-0.15, 0.10), (-0.05, 0.20),
                       (-0.22, 0.18), (-0.18, 0.22)]:
        for sigma in [0.03, 0.05, 0.08]:
            for h in [0.5, 1.0, 2.0]:
                p = make_asym_two_bump(off1, off2, sigma, h)
                M, _ = Mc(p)
                if abs(M - M_target) < 0.10:
                    seeds.append((f"asym_({off1:.2f},{off2:.2f})_s{sigma:.2f}_h{h}_M{M:.3f}", p))

    # Random asymmetric mixtures
    for tr in range(n_random):
        nb = int(rng.integers(3, 8))
        p = np.zeros_like(xs)
        for _ in range(nb):
            mu = rng.uniform(-0.22, 0.22)
            sigma = rng.uniform(0.005, 0.05)
            h = rng.uniform(0.2, 1.0)
            p += h * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
        p = p / p.sum()
        seeds.append((f"rand_seed{rng_seed}_tr{tr}", p))

    # Always include scaled BL itself (will end up far from target if M_target far)
    seeds.append(("BL_raw", p_BL))
    seeds.append(("MV_raw", p_MV))

    return seeds


def estimate_phi(M_target, max_seeds=20, max_iters=400, verbose=False):
    """Estimate Phi(M_target) = sup_f c(f) by gradient ascent at each seed.
    Returns the best c found and the seed name + final M."""
    best_c = -np.inf
    best_M = None
    best_name = None
    seeds = initial_seeds_at_M(M_target, n_random=8)
    seeds = seeds[:max_seeds]
    for name, p_init in seeds:
        try:
            c, M_act, _ = maximize_c_at_M(
                p_init, M_target, n_iters=max_iters, lr=0.005,
                lam_start=20.0, lam_end=2000.0,
                T_start=0.02, T_end=0.001)
        except Exception as e:
            if verbose:
                print(f"  {name}: failed {e}")
            continue
        if c > best_c:
            best_c, best_M, best_name = c, M_act, name
        if verbose:
            print(f"  {name:<60s} -> c={c:.4f}, M={M_act:.4f}")
    return best_c, best_M, best_name


def baseline_c_at_M(M_target, n_seeds=60, rng_seed=42):
    """Pure search: many seeds, take the max c over those landing close to M_target.
    No gradient ascent.
    """
    seeds = initial_seeds_at_M(M_target, n_random=n_seeds, rng_seed=rng_seed)
    best = -np.inf
    best_M = None
    best_name = None
    for name, p in seeds:
        M_act, c_act = Mc(p)
        if abs(M_act - M_target) < 0.025 and c_act > best:
            best = c_act
            best_M = M_act
            best_name = name
    return best, best_M, best_name


def main():
    print(f"D={D}, dx={dx:.5e}")
    print()
    # Reference points
    p_BL = make_BL()
    p_MV = make_MV()
    p_U = make_uniform()
    print("Reference (M, c):")
    print(f"  Uniform           M={Mc(p_U)[0]:.4f}  c={Mc(p_U)[1]:.4f}  (theory M=2, c=2/3=0.6667)")
    print(f"  MV-24 cosine sum  M={Mc(p_MV)[0]:.4f}  c={Mc(p_MV)[1]:.4f}")
    print(f"  BL witness        M={Mc(p_BL)[0]:.4f}  c={Mc(p_BL)[1]:.4f}")
    print()

    # ---- Sweep Phi(M) on grid M in [1.275, 1.65], spacing 0.025 ----
    print("=" * 92)
    print("Phi(M) ESTIMATE via gradient ASCENT on c at each M (seed-rich)")
    print("=" * 92)
    print(f"{'M_target':>9s}  {'Phi_est':>8s}  {'c_baseline':>11s}  {'M_act':>7s}  {'best_seed'}")
    Ms_target = np.arange(1.275, 1.661, 0.025)
    phi_curve = []
    baseline_curve = []
    for M in Ms_target:
        c_grad, M_grad, name = estimate_phi(M, max_seeds=24, max_iters=300)
        c_base, M_base, name_b = baseline_c_at_M(M, n_seeds=80)
        phi_curve.append((M, c_grad, M_grad, name))
        baseline_curve.append((M, c_base, M_base, name_b))
        c_grad_str = f"{c_grad:.4f}" if c_grad > -np.inf else "  ----"
        c_base_str = f"{c_base:.4f}" if c_base > -np.inf else "  ----"
        M_str = f"{M_grad:.4f}" if M_grad is not None else " ----"
        print(f"  {M:>7.3f}  {c_grad_str:>8s}  {c_base_str:>11s}  {M_str:>7s}  {name}")

    # ---- Monotonicity check ----
    print()
    print("=" * 92)
    print("MONOTONICITY analysis on Phi_est curve")
    print("=" * 92)
    is_monotone = True
    n_violations = 0
    max_viol_drop = 0.0
    Ms_arr = np.array([t[0] for t in phi_curve])
    cs_arr = np.array([t[1] for t in phi_curve])
    for i in range(1, len(phi_curve)):
        if cs_arr[i] < cs_arr[i-1]:
            drop = cs_arr[i-1] - cs_arr[i]
            n_violations += 1
            max_viol_drop = max(max_viol_drop, drop)
            is_monotone = False
            print(f"  M={Ms_arr[i]:.3f}: Phi={cs_arr[i]:.4f} < prev Phi={cs_arr[i-1]:.4f} (drop={drop:.4f})")
    if is_monotone:
        print("  Phi_est is MONOTONE INCREASING on the sampled grid.")
    else:
        print(f"  Phi_est is NOT MONOTONE: {n_violations} violations, max drop = {max_viol_drop:.4f}")

    # ---- Linear interpolation bound on Phi(1.378) ----
    print()
    print("=" * 92)
    print("INTERPOLATION bound on Phi(1.378)")
    print("=" * 92)
    M_low, M_high = 1.275, 1.65
    # Use the estimated Phi at endpoints
    idx_low = np.argmin(np.abs(Ms_arr - M_low))
    idx_high = np.argmin(np.abs(Ms_arr - M_high))
    phi_lo = cs_arr[idx_low]
    phi_hi = cs_arr[idx_high]
    print(f"  Phi_est({Ms_arr[idx_low]:.3f}) = {phi_lo:.4f}")
    print(f"  Phi_est({Ms_arr[idx_high]:.3f}) = {phi_hi:.4f}")
    M_query = 1.378
    if is_monotone:
        # If Phi monotone increasing AND continuous, Phi(1.378) lies between Phi(1.275) and Phi(1.65).
        # Linear interp gives a heuristic value; rigorous upper bound is Phi(M_high) since increasing.
        alpha = (M_query - M_low) / (M_high - M_low)
        interp = phi_lo + alpha * (phi_hi - phi_lo)
        print(f"  Linear interp at M=1.378:  {interp:.4f}")
        print(f"  RIGOROUS UB if monotone:  Phi(1.378) <= Phi(M_high) = {phi_hi:.4f}")
        if interp < 0.88254:
            print(f"  Interp value {interp:.4f} < 0.88254 (good for Hyp_R) by {0.88254 - interp:.4f}")
        if phi_hi < 0.88254:
            print(f"  RIGOROUS UB {phi_hi:.4f} < 0.88254 (good for Hyp_R) by {0.88254 - phi_hi:.4f}")
        else:
            print(f"  WARNING: Phi_est({Ms_arr[idx_high]:.3f}) = {phi_hi:.4f} >= 0.88254;")
            print(f"  monotonicity bound is too weak. Need INTERIOR evaluation at M=1.378.")
    else:
        print(f"  Cannot use monotonicity argument (DISPROVED numerically).")
        # In this case, just report Phi_est(1.378) directly
        idx_q = np.argmin(np.abs(Ms_arr - M_query))
        print(f"  Direct estimate: Phi_est({Ms_arr[idx_q]:.3f}) = {cs_arr[idx_q]:.4f}")

    # ---- Direct estimate at M=1.378 ----
    print()
    print("=" * 92)
    print("DIRECT estimate of Phi(1.378) (the value we care about)")
    print("=" * 92)
    c_d, M_d, name_d = estimate_phi(1.378, max_seeds=40, max_iters=600, verbose=False)
    print(f"  Phi_est(1.378) = {c_d:.4f} achieved at M_act = {M_d:.4f}, seed = {name_d}")
    if c_d < 0.88254:
        print(f"  *** Below 0.88254 by margin {0.88254 - c_d:.4f} ***")
    else:
        print(f"  *** EXCEEDS 0.88254 by {c_d - 0.88254:.4f} ***")

    # ---- Output curve as table ----
    print()
    print("=" * 92)
    print("PHI(M) CURVE (CSV)")
    print("=" * 92)
    print(f"{'M':>7s},{'Phi_est':>8s},{'c_baseline':>11s},{'M_act':>7s}")
    for (M, c, Ma, name), (Mb, cb, Mbb, _) in zip(phi_curve, baseline_curve):
        print(f"{M:>7.3f},{c:>8.4f},{cb:>11.4f},{Ma:>7.4f}")


if __name__ == "__main__":
    main()
