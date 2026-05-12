"""
Direct variational ascent on Phi(M) := sup { ||f*f||_2^2 / ||f*f||_inf : f admissible, ||f*f||_inf <= M }.

Strategy:
  1. K-step parameterization (atoms on uniform grid), simplex-constrained via softmax
     so f >= 0 and int f dx = 1 automatically.
  2. Augmented-Lagrangian: maximize c_emp + lam * (M_target - Linf)^2 (penalty for both directions
     to drive M -> M_target so that constraint binds at solution).
  3. Multiple restarts: BL-derived seeds, MV-cosine seeds, random asymmetric seeds.
  4. Inner solver: L-BFGS-B with relaxed convergence, many random restarts.

Output: phi_curve_focused.json + per-M best f arrays.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve

print = lambda *a, **k: __builtins__.print(*a, **k, flush=True)

ROOT = Path(__file__).resolve().parent
COEFF_FILE = ROOT.parent / "restricted_holder" / "coeffBL.txt"
import sys; sys.path.insert(0, str(ROOT.parent.parent))
from delsarte_dual.restricted_holder.conditional_bound import MV_COEFFS_119_STR

C_STAR = float(np.log(16.0) / np.pi)  # 0.88254240061060637

# ---- Grid ----
N = 256                         # f sampled on N points across [-1/4, 1/4]
DX = 0.5 / N
XS = np.linspace(-0.25 + DX/2, 0.25 - DX/2, N)


# ---- Core metrics ----
def conv_self(f):
    return fftconvolve(f, f, mode='full') * DX  # length 2N-1


def metrics(f):
    g = conv_self(f)
    Linf = float(np.max(g))
    L2sq = float(np.sum(g * g) * DX)
    L1 = float(np.sum(g) * DX)
    c_emp = L2sq / Linf if Linf > 1e-15 else 0.0
    return Linf, L2sq, L1, c_emp


# ---- Softmax simplex parameterization ----
def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def f_from_z(z):
    """K-dim parameter z; w = softmax(z) is K-simplex; f[i*block:(i+1)*block] = w[i] / (block*DX)."""
    K = len(z)
    w = softmax(z)
    if K == N:
        return w / DX
    block = N // K
    f = np.zeros(N)
    for i in range(K):
        s = i * block
        e = (i + 1) * block if i < K - 1 else N
        f[s:e] = w[i] / ((e - s) * DX)
    return f


# ---- Seed generators ----
_BL = None
def load_BL():
    global _BL
    if _BL is None:
        with open(COEFF_FILE) as fh:
            s = fh.read().strip().strip('{}').split(',')
        v = np.array([int(x) for x in s], dtype=float)
        v /= v.max()
        _BL = v
    return _BL.copy()


def seed_BL(K=N, scale=1.0, head_keep=None):
    """Resample BL onto K-grid (covering full [-1/4, 1/4] support)."""
    v = load_BL()
    if head_keep is not None:
        v[head_keep:] = 0.0
    L = len(v)
    half = 0.25 * scale
    xs_K = np.linspace(-0.25 + 0.25/K, 0.25 - 0.25/K, K)
    in_supp = (xs_K >= -half) & (xs_K < half)
    bin_idx = ((xs_K + half) / (2 * half / L)).astype(int)
    bin_idx = np.clip(bin_idx, 0, L - 1)
    w = np.zeros(K)
    w[in_supp] = v[bin_idx[in_supp]]
    w = np.maximum(w, 1e-8)
    w /= w.sum()
    return np.log(w)


def seed_BL_shifted(K=N, shift_frac=0.0, scale=1.0):
    """BL placed on a sub-window centered at shift_frac * available_offset."""
    v = load_BL()
    L = len(v)
    half_w = 0.25 * scale
    xs_K = np.linspace(-0.25 + 0.25/K, 0.25 - 0.25/K, K)
    center = shift_frac * (0.25 - half_w)
    in_supp = (xs_K >= center - half_w) & (xs_K < center + half_w)
    bin_idx = ((xs_K - (center - half_w)) / (2 * half_w / L)).astype(int)
    bin_idx = np.clip(bin_idx, 0, L - 1)
    w = np.zeros(K)
    w[in_supp] = v[bin_idx[in_supp]]
    w = np.maximum(w, 1e-8)
    w /= w.sum()
    return np.log(w)


def seed_MV_cosine(K=N):
    """Sample MV's 119-cosine extremizer onto K-grid."""
    a = [float(s) for s in MV_COEFFS_119_STR]
    u = 0.638
    xs_K = np.linspace(-0.25 + 0.25/K, 0.25 - 0.25/K, K)
    G = np.zeros(K)
    for j, aj in enumerate(a, start=1):
        G += aj * np.cos(2 * np.pi * j * xs_K / u)
    w = np.maximum(G, 1e-8)
    w /= w.sum()
    return np.log(w)


def seed_random_asymmetric(K, rng, scale=1.5):
    """Random asymmetric init."""
    return rng.normal(0, scale, K)


def seed_random_symmetric(K, rng, scale=1.5):
    """Random symmetric (mirrored) init."""
    half = (K + 1) // 2
    tmp = rng.normal(0, scale, half)
    return np.concatenate([tmp, tmp[:K - half][::-1]])


def seed_uniform(K):
    """Indicator on full support."""
    return np.zeros(K)


def seed_bimodal(K, mode_w_frac=0.1, gap_frac=0.6, h1=1.0, h2=1.0, asym=True):
    """Two-mode shape."""
    w = np.full(K, 1e-6)
    mw = max(2, int(mode_w_frac * K))
    s1 = 0
    e1 = mw
    s2 = K - mw
    e2 = K
    w[s1:e1] = h1
    w[s2:e2] = h2 * (1.0 if asym else 1.0)
    w /= w.sum()
    return np.log(w)


# ---- Inner objective: augmented Lagrangian ----
def neg_obj(z, M_target, lam):
    """Maximize c_emp; penalize Linf > M_target AND Linf < M_target * 0.95 (push to boundary)."""
    f = f_from_z(z)
    Linf, L2sq, _, c = metrics(f)
    pen_high = max(0.0, Linf - M_target)**2 * lam
    # mild push toward M_target (so constraint binds)
    return -c + pen_high


def maximize_c_emp(z0, M_target, max_iter_per_round=120, n_rounds=8, verbose=False):
    z = z0.copy()
    best = (-np.inf, None, None)
    lam = 30.0
    for r in range(n_rounds):
        try:
            res = minimize(
                neg_obj, z, args=(M_target, lam),
                method='L-BFGS-B',
                options={'maxiter': max_iter_per_round, 'ftol': 1e-9, 'gtol': 1e-7}
            )
            z = res.x
        except Exception:
            break
        f = f_from_z(z)
        Linf, _, _, c = metrics(f)
        # Accept solutions with Linf within 1% of target (or below)
        if Linf <= M_target * 1.01 and c > best[0]:
            best = (c, f.copy(), Linf)
        lam *= 2.5
        if verbose:
            print(f"    r={r}: lam={lam:.0f}, M={Linf:.5f}, c={c:.5f}, best_c={best[0]:.5f}")
    return best


# ---- Outer scan ----
def scan_phi_at_M(M_target, K_list=(64, 128, 256), n_random_restarts=30,
                   time_budget_s=120.0, verbose=False):
    rng = np.random.default_rng(20260503 + int(M_target * 1000))
    t_start = time.time()
    pool = []  # list of (c, f, M_actual, label)

    def try_z0(z0, label, K=None):
        if z0 is None:
            return
        c, f, M = maximize_c_emp(z0, M_target, max_iter_per_round=150, n_rounds=8)
        if f is not None:
            pool.append((c, f, M, label))
            if verbose:
                print(f"      {label}: c={c:.5f}, M={M:.5f}")

    # ---- Curated seeds ----
    for K in K_list:
        # MV cosine
        try_z0(seed_MV_cosine(K), f"mv_K{K}", K)
        if time.time() - t_start > time_budget_s:
            break
        # Indicator
        try_z0(seed_uniform(K), f"unif_K{K}", K)
        # BL full + scaled support
        for sf in [0.5, 0.7, 0.85, 1.0]:
            try_z0(seed_BL(K, scale=sf), f"bl_sf{sf}_K{K}", K)
            if time.time() - t_start > time_budget_s * 0.5:
                break
        # BL with truncated head
        for hk in [80, 150, 287]:
            try_z0(seed_BL(K, scale=1.0, head_keep=hk), f"bl_hk{hk}_K{K}", K)
        # BL shifted (asymmetric placement)
        for shift in [0.2, 0.4, 0.6, 0.8]:
            for sf in [0.6, 0.85]:
                try_z0(seed_BL_shifted(K, shift_frac=shift, scale=sf),
                       f"bl_sh{shift}_sf{sf}_K{K}", K)
        # Bimodal
        for mw in [0.05, 0.10, 0.15]:
            for h2 in [0.5, 1.0, 1.5]:
                try_z0(seed_bimodal(K, mode_w_frac=mw, h2=h2), f"bi_mw{mw}_h{h2}_K{K}", K)

    # ---- Random restarts ----
    for r in range(n_random_restarts):
        if time.time() - t_start > time_budget_s:
            break
        K = rng.choice(K_list)
        scale = rng.uniform(0.5, 2.5)
        if rng.random() < 0.5:
            z0 = seed_random_asymmetric(K, rng, scale)
        else:
            z0 = seed_random_symmetric(K, rng, scale)
        try_z0(z0, f"rand_K{K}_r{r}", K)

    if not pool:
        return None
    pool.sort(key=lambda x: -x[0])
    return pool[0], pool


# ---- Description helpers ----
def asymmetry(f):
    fr = f[::-1]
    n = float(np.sum(np.abs(f - fr)) * DX)
    d = float(np.sum(np.abs(f + fr)) * DX)
    return n / d if d > 0 else 0.0


def describe(f):
    Linf, L2sq, L1, c = metrics(f)
    a = asymmetry(f)
    K = float(np.sum(f * f) * DX)  # ||f||_2^2
    nz = f > 1e-3 * f.max()
    sup_lo = float(XS[np.where(nz)[0][0]]) if nz.any() else 0.0
    sup_hi = float(XS[np.where(nz)[0][-1]]) if nz.any() else 0.0
    return {
        'M': Linf, 'L2sq': L2sq, 'L1': L1, 'c_emp': c,
        'asymmetry': a,
        'support_lo': sup_lo, 'support_hi': sup_hi,
        'support_width': sup_hi - sup_lo,
        'peak_x': float(XS[np.argmax(f)]),
        'f_max': float(np.max(f)),
        'norm_f_sq_K': K,
        'K_over_M': K / Linf if Linf > 0 else 0.0,
    }


def main():
    print(f"c_star = log16/pi = {C_STAR:.10f}")
    print(f"Grid: N={N}, DX={DX:.4e}")
    print()
    M_grid = [1.27, 1.30, 1.33, 1.35, 1.378]
    out = {'c_star': C_STAR, 'N': N, 'M_grid': M_grid, 'results': {}}
    t0 = time.time()

    for M_target in M_grid:
        print('=' * 70)
        print(f"M_target = {M_target} (elapsed {time.time()-t0:.1f}s)")
        print('=' * 70)
        result = scan_phi_at_M(M_target, K_list=(64, 128, 256), n_random_restarts=20,
                                time_budget_s=180.0, verbose=False)
        if result is None:
            print(f"  *** NO FEASIBLE SAMPLE for M={M_target}")
            continue
        best, pool = result
        c, f, M, label = best
        d = describe(f)
        margin = C_STAR - c
        verdict = ('PROMISING' if margin > 0.01 else
                   ('NEAR-CRITICAL' if margin > -0.001 else 'DISPROVED'))
        print(f"\n  >> BEST at M={M_target}: c_emp={c:.6f}, M_actual={M:.6f}")
        print(f"     margin to c* = {margin:+.6f} -> {verdict}")
        print(f"     label = {label}")
        print(f"     STRUCT: asym={d['asymmetry']:.4f}, K=||f||_2^2={d['norm_f_sq_K']:.4f}, "
              f"K/M={d['K_over_M']:.3f}")
        print(f"     support [{d['support_lo']:.3f},{d['support_hi']:.3f}], "
              f"peak_x={d['peak_x']:.3f}, f_max={d['f_max']:.2f}")
        # top 5 results to see if ascent diversifies
        print(f"     Top 5 results in pool:")
        for c_p, _, M_p, lab_p in sorted(pool, key=lambda x: -x[0])[:5]:
            in_reg = "in" if M_p <= M_target * 1.005 else "out"
            print(f"       c={c_p:.5f}, M={M_p:.5f}, {in_reg}, {lab_p}")
        out['results'][f"{M_target}"] = {
            'c_emp': c, 'M_actual': M, 'label': label, 'description': d,
            'margin': margin, 'verdict': verdict,
            'top_5_pool': [
                {'c': c_p, 'M': M_p, 'label': lab_p}
                for c_p, _, M_p, lab_p in sorted(pool, key=lambda x: -x[0])[:5]
            ],
        }
        np.save(ROOT / f"phi_focused_f_M{M_target}.npy", f)

    print()
    print('=' * 70)
    print('PHI(M) CURVE SUMMARY')
    print('=' * 70)
    print(f"{'M':>8} {'phi_emp':>14} {'margin_to_c*':>14} {'verdict':<14}  K/M")
    for M_target in M_grid:
        key = f"{M_target}"
        if key not in out['results']:
            print(f"  M={M_target}: (no result)")
            continue
        r = out['results'][key]
        print(f"  {M_target:>6}  {r['c_emp']:>12.6f}  {r['margin']:>+14.6f}  "
              f"{r['verdict']:<14} {r['description']['K_over_M']:.3f}")

    out['c_star'] = C_STAR
    with open(ROOT / "phi_curve_focused.json", 'w') as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nSaved {ROOT / 'phi_curve_focused.json'} (total {time.time()-t0:.1f}s)")


if __name__ == '__main__':
    main()
