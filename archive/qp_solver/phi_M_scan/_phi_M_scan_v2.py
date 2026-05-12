"""Phi(M) := sup over admissible f with ||f*f||_inf = M of c_emp(f).

V2 Strategy:
  - Use both MV-class and BL-class as parametric "anchors".
  - Smooth interpolations between Uniform/MV/BL to produce a rich set of shapes.
  - Use gradient ASCENT on c_emp subject to a soft penalty on M.
  - For values of M difficult to reach, also report the convex-hull bound:
       Phi(M) >= max c over points (M_i, c_i) with M_i <= M (since c is scale-invariant).

  - We also need the SCALE-INVARIANCE observation: c(f) is invariant under
    f -> (1/lambda) f(x/lambda). Therefore c is naturally a property of "shape".
    To explore the (M, c) plane, we vary shape.

KEY: at a fixed shape g = f * f, scaling moves (M, c) along the constant-c line.
So the (M, c) plot, once we run ENOUGH shape varieties, traces out a horizontal
slice for each shape.
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
    """MV's full 119-coefficient cosine sum."""
    coeffs = [
        +2.16620392, -1.87775750, +1.05828868, -0.729790538,
        +0.428008515, +0.217832838, -0.270415201, +0.0272834790,
        -0.191721888, +0.0551862060, +0.321662512, -0.164478392,
        +0.0395478603, -0.205402785, -0.0133758316, +0.231873221,
        -0.0437967118, +0.0612456374, -0.157361919, -0.0778036253,
        +0.138714392, -0.000145201483, +0.0916539824, -0.0834020840,
        -0.101919986, +0.0594915025, -0.0119336618, +0.102155366,
        -0.0145929982, -0.0795205457, +0.00559733152, -0.0358987179,
        +0.0716132260, +0.0415425065, -0.0489180454, +0.00165425755,
        -0.0648251747, +0.0345951253, +0.0532122058, -0.0128435276,
        +0.0148814403, -0.0649404547, -0.00601344770, +0.0433784473,
        -0.000253362778, +0.0381674519, -0.0483816002, -0.0253878079,
        +0.0196933442, -0.00304861682, +0.0479203471, -0.0200930265,
        -0.0273895519, +0.00330183589, -0.0167380508, +0.0423917582,
        +0.00364690190, -0.0179916104, +0.0000731661649, -0.0299875575,
        +0.0271842526, +0.0141806855, -0.00601781076, +0.00586806100,
        -0.0332350597, +0.00923347466, +0.0147071722, -0.000742858080,
        +0.0163414270, -0.0287265671, -0.00164287280, +0.00802601605,
        -0.000762613027, +0.0218735533, -0.0178816282, -0.00658341101,
        +0.00267706547, -0.00625261247, +0.0224942824, -0.00810756022,
        -0.00568160823, +0.0000701871209, -0.0115294332, +0.0183608944,
        -0.00120567880, -0.00313147456, +0.00139083675, -0.0149312478,
        +0.0132106694, +0.00173474188, -0.000853469045, +0.00403211203,
        -0.0155352991, +0.00874711543, +0.00193998895, -0.0000271357322,
        +0.00613179585, -0.0141983972, +0.00584710551, +0.000922578333,
        -0.000216583469, +0.00707919829, -0.0118488582, +0.00439698322,
        -0.0000891346785, -0.000342086367, +0.00646355636, -0.00887555371,
        +0.00356799654, -0.000497335419, -0.000804560326, +0.00555076717,
        -0.00713560569, +0.00453679038, -0.00333261516, +0.00235463427,
        +0.000204023789, -0.00127746711, +0.000181247830,
    ]
    coeffs = coeffs[:n_coeffs]
    f = np.zeros_like(xs)
    for j, a in enumerate(coeffs, start=1):
        f += a * np.cos(2 * np.pi * j * xs / u_var)
    f = np.maximum(f, 0)
    return f / f.sum()


def grad_obj_max_c_at_M(p, M_target, T_smax=0.005, lam=200.0):
    """Maximize ||g||_2^2 - lam*(smaxM - M_target)^2."""
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
                     lam_start=10.0, lam_end=5000.0,
                     T_start=0.02, T_end=0.0005,
                     M_tol=0.04):
    """Project gradient ascent to max c subject to M ~ M_target.
    Returns best (c, M, p) seen with |M - M_target| < M_tol.
    """
    p = project_simplex(p_init.copy())
    best_c = -np.inf
    best_M = None
    best_p = p.copy()
    for it in range(n_iters):
        T = T_start * (T_end / T_start) ** (it / max(1, n_iters - 1))
        lam = lam_start * (lam_end / lam_start) ** (it / max(1, n_iters - 1))
        obj, gradp, M_act, c_act = grad_obj_max_c_at_M(p, M_target, T_smax=T, lam=lam)
        # Adaptive step
        p = project_simplex(p + lr * gradp)
        if abs(M_act - M_target) < M_tol and c_act > best_c:
            best_c = c_act
            best_p = p.copy()
            best_M = M_act
    M_f, c_f = Mc(p)
    if abs(M_f - M_target) < M_tol and c_f > best_c:
        best_c = c_f
        best_p = p.copy()
        best_M = M_f
    return best_c, best_M, best_p


def collect_seeds():
    """Build a large pool of (name, p, M, c) reference shapes."""
    p_BL = make_BL()
    p_U = make_uniform()
    seeds = [("BL", p_BL), ("Uniform", p_U)]
    # MV at various u_var
    for u in [0.55, 0.58, 0.60, 0.62, 0.638, 0.65, 0.68, 0.70, 0.72, 0.75]:
        for n in [12, 24, 48, 119]:
            try:
                p = make_MV(u_var=u, n_coeffs=n)
                seeds.append((f"MV_u{u:.3f}_n{n}", p))
            except Exception:
                pass
    # MV-BL homotopy
    p_MV = make_MV()
    for alpha in np.linspace(0, 1, 21):
        p = (1 - alpha) * p_MV + alpha * p_BL
        if p.min() < 0:
            continue
        p /= p.sum()
        seeds.append((f"MV-BL_a{alpha:.2f}", p))
    # MV-U homotopy
    for alpha in np.linspace(0, 1, 21):
        p = (1 - alpha) * p_MV + alpha * p_U
        if p.min() < 0:
            continue
        p /= p.sum()
        seeds.append((f"MV-U_a{alpha:.2f}", p))
    # BL-U
    for alpha in np.linspace(0, 1, 21):
        p = (1 - alpha) * p_BL + alpha * p_U
        if p.min() < 0:
            continue
        p /= p.sum()
        seeds.append((f"BL-U_a{alpha:.2f}", p))
    # 3-way mix
    for a in np.arange(0, 1.01, 0.1):
        for b in np.arange(0, 1.01 - a, 0.1):
            ccoef = 1 - a - b
            p = a * p_MV + b * p_BL + ccoef * p_U
            if p.min() < 0:
                continue
            p /= p.sum()
            seeds.append((f"mix_MV{a:.1f}_BL{b:.1f}_U{ccoef:.1f}", p))
    # MV with random perturbation (positive), sym + asym
    rng = np.random.default_rng(0)
    for trial in range(20):
        amp = rng.uniform(0.0, 0.8)
        n_modes = rng.integers(1, 8)
        delta = np.zeros_like(xs)
        for _ in range(n_modes):
            k = rng.uniform(0.5, 12)
            phi = rng.uniform(0, 2 * np.pi)
            delta += rng.normal() * np.sin(2 * np.pi * k * np.arange(D) / D + phi)
        p = np.maximum(p_MV + amp * delta * p_MV.mean(), 0)
        if p.sum() == 0:
            continue
        p /= p.sum()
        seeds.append((f"MV_pert_t{trial}_amp{amp:.2f}", p))
    # BL with random perturbation
    for trial in range(20):
        amp = rng.uniform(0.0, 0.5)
        n_modes = rng.integers(1, 8)
        delta = np.zeros_like(xs)
        for _ in range(n_modes):
            k = rng.uniform(0.5, 12)
            phi = rng.uniform(0, 2 * np.pi)
            delta += rng.normal() * np.sin(2 * np.pi * k * np.arange(D) / D + phi)
        p = np.maximum(p_BL + amp * delta * p_BL.mean(), 0)
        if p.sum() == 0:
            continue
        p /= p.sum()
        seeds.append((f"BL_pert_t{trial}_amp{amp:.2f}", p))
    return seeds


def main():
    print(f"D={D}, dx={dx:.5e}")
    print()
    # Reference points
    p_BL = make_BL()
    p_U = make_uniform()
    p_MV = make_MV()
    print("Reference (M, c):")
    for name, p in [("Uniform", p_U), ("MV(119)", p_MV), ("BL", p_BL)]:
        M, c = Mc(p)
        print(f"  {name:<20s}  M={M:.4f}  c={c:.4f}")

    # ---- Phase 1: Build a large catalogue of (M, c) baseline points ----
    print()
    print("=" * 92)
    print("Phase 1: Catalog of baseline (M, c) points from rich seed family")
    print("=" * 92)
    seeds = collect_seeds()
    pts = []
    for name, p in seeds:
        try:
            M, c = Mc(p)
        except Exception:
            continue
        pts.append((M, c, name))
    print(f"Total seeds: {len(seeds)}, evaluated pts: {len(pts)}")

    # Print sorted by M
    pts.sort()
    print(f"\nM range: [{pts[0][0]:.4f}, {pts[-1][0]:.4f}]")

    # Show baseline scatter restricted to M in [1.27, 1.7]
    print("\nBaseline points with M in [1.27, 1.7]:")
    for M, c, name in pts:
        if 1.27 <= M <= 1.70:
            print(f"  M={M:.4f}  c={c:.4f}  {name}")

    # Compute a "Phi_baseline(M)" by binning
    bins = np.arange(1.275, 1.701, 0.025)
    print("\nBinned Phi_baseline(M) (max c over seeds in each bin):")
    print(f"{'M_bin':>8s}  {'Phi_baseline':>13s}  {'n':>5s}  {'best seed'}")
    Phi_baseline = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i+1]
        in_bin = [(M, c, name) for M, c, name in pts if lo <= M < hi]
        if in_bin:
            best = max(in_bin, key=lambda r: r[1])
            Phi_baseline.append((0.5*(lo+hi), best[1], best[0], best[2]))
            print(f"  [{lo:.3f},{hi:.3f}]  {best[1]:>13.4f}  {len(in_bin):>5d}  {best[2]} (M={best[0]:.4f})")
        else:
            Phi_baseline.append((0.5*(lo+hi), -np.inf, None, None))
            print(f"  [{lo:.3f},{hi:.3f}]  {'(empty)':>13s}  {0:>5d}")

    # ---- Phase 2: For each non-empty bin, run gradient ASCENT seeded from best baseline ----
    print()
    print("=" * 92)
    print("Phase 2: Refine Phi(M) via gradient ASCENT from best baseline seed")
    print("=" * 92)
    print(f"{'M_target':>9s}  {'Phi_base':>9s}  {'Phi_grad':>9s}  {'M_act':>7s}")
    refined = []
    for M_target, c_base, M_base, name_base in Phi_baseline:
        # Find seed: take best baseline pt in [M_target - 0.05, M_target + 0.05]
        seed_pool = [(M, c, name) for M, c, name in pts
                     if abs(M - M_target) < 0.06]
        if not seed_pool:
            seed_pool = [(M, c, name) for M, c, name in pts]
        seed_pool.sort(key=lambda r: -r[1])  # take highest c first
        best_c, best_M = c_base, M_base
        best_name = name_base
        # Try top-5 highest-c seeds
        for M_i, c_i, name_i in seed_pool[:8]:
            try:
                # rebuild seed from name (we lost p reference); re-collect
                seeds_dict = {n: p for n, p in seeds}
                if name_i not in seeds_dict:
                    continue
                p_seed = seeds_dict[name_i]
                c_g, M_g, _ = maximize_c_at_M(
                    p_seed, M_target, n_iters=400, lr=0.003,
                    lam_start=20.0, lam_end=8000.0,
                    M_tol=0.025)
                if c_g > best_c:
                    best_c = c_g
                    best_M = M_g
                    best_name = f"refined-from-{name_i}"
            except Exception as e:
                continue
        refined.append((M_target, best_c, best_M, best_name))
        c_str = f"{best_c:.4f}" if best_c > -np.inf else "  ----"
        M_str = f"{best_M:.4f}" if best_M is not None else " ----"
        cb_str = f"{c_base:.4f}" if c_base > -np.inf else "  ----"
        print(f"  {M_target:>7.3f}  {cb_str:>9s}  {c_str:>9s}  {M_str:>7s}")

    # ---- Phase 3: Monotonicity analysis on the refined Phi ----
    print()
    print("=" * 92)
    print("Phase 3: MONOTONICITY of refined Phi(M)")
    print("=" * 92)
    valid = [(M, c) for M, c, _, _ in refined if c > -np.inf]
    if len(valid) < 2:
        print("Insufficient data points.")
        return
    Ms = np.array([M for M, c in valid])
    Cs = np.array([c for M, c in valid])
    is_monotone = True
    n_violations = 0
    max_drop = 0.0
    for i in range(1, len(Ms)):
        if Cs[i] < Cs[i-1] - 0.005:  # 0.005 tolerance
            n_violations += 1
            max_drop = max(max_drop, Cs[i-1] - Cs[i])
            is_monotone = False
            print(f"  Violation at M={Ms[i]:.3f}: Phi={Cs[i]:.4f} < prev Phi(M={Ms[i-1]:.3f})={Cs[i-1]:.4f} (drop={Cs[i-1]-Cs[i]:.4f})")
    if is_monotone:
        print(f"  Phi(M) is MONOTONE INCREASING on grid [{Ms[0]:.3f}, {Ms[-1]:.3f}] (within tol 0.005)")
    else:
        print(f"  Phi(M) NOT monotone: {n_violations} violations, max drop {max_drop:.4f}")

    # ---- Phase 4: Direct estimate at M=1.378 with high effort ----
    print()
    print("=" * 92)
    print("Phase 4: DIRECT high-effort estimate of Phi(1.378)")
    print("=" * 92)
    seeds_high_effort = []
    for M_t in [1.30, 1.34, 1.378, 1.42]:
        # find best 5 seeds whose M is near M_t
        near = [(M, c, name) for M, c, name in pts if abs(M - M_t) < 0.10]
        near.sort(key=lambda r: -r[1])
        for _, _, name in near[:5]:
            seeds_high_effort.append(name)
    seeds_high_effort = list(dict.fromkeys(seeds_high_effort))
    print(f"Trying {len(seeds_high_effort)} seeds for M=1.378")
    seeds_dict = {n: p for n, p in seeds}
    best_c, best_M, best_name = -np.inf, None, None
    for name in seeds_high_effort:
        if name not in seeds_dict:
            continue
        try:
            c_g, M_g, _ = maximize_c_at_M(
                seeds_dict[name], 1.378, n_iters=1000, lr=0.002,
                lam_start=50.0, lam_end=20000.0,
                M_tol=0.020)
            if c_g > best_c:
                best_c, best_M, best_name = c_g, M_g, name
        except Exception:
            continue
    print(f"Best Phi(1.378) ~= {best_c:.4f} achieved at M_act={best_M}, seed={best_name}")
    print()

    # ---- Phase 5: Interpolation summary ----
    print("=" * 92)
    print("Phase 5: INTERPOLATION bound on Phi(1.378)")
    print("=" * 92)
    # Find Phi at M~1.275 (MV regime) and Phi at M~1.65 (BL regime) from refined
    if len(Ms) > 0:
        # Interpolate
        from numpy import interp
        phi_interp_138 = float(interp(1.378, Ms, Cs))
        print(f"  Refined Phi grid: M in [{Ms[0]:.3f}, {Ms[-1]:.3f}]")
        print(f"  Linear-interp Phi(1.378) on refined grid = {phi_interp_138:.4f}")
        # Direct Phi(1.378)
        if best_c > -np.inf:
            print(f"  Direct (high-effort) Phi(1.378) >= {best_c:.4f}")
            margin = 0.88254 - best_c
            print(f"  Margin to threshold 0.88254 = {margin:+.4f}")

    # ---- Phase 6: VERDICT ----
    print()
    print("=" * 92)
    print("VERDICT")
    print("=" * 92)
    if is_monotone and best_c < 0.88254:
        print("PROMISING:")
        print(f"  Numerical evidence is consistent with Phi(M) monotone-increasing on [1.275, 1.65].")
        print(f"  Direct numerical Phi(1.378) ~= {best_c:.4f} < 0.88254.")
    elif not is_monotone:
        print("OBSTRUCTED (or further numerical effort needed):")
        print(f"  Phi(M) shows non-monotonic behavior on the sampled grid.")
        print(f"  This makes the interpolation argument INVALID.")
    elif best_c >= 0.88254:
        print(f"PROBLEMATIC: numerical Phi(1.378) >= 0.88254 (= {best_c:.4f}).")
        print(f"  This would FALSIFY Hyp_R if confirmed rigorously.")


if __name__ == "__main__":
    main()
