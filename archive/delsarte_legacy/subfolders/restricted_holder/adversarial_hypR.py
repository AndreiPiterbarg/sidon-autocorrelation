"""
Adversarial test of Hyp_R: try to find a counterexample with M <= 1.378.

Hyp_R says: for nonneg pdf f on [-1/4,1/4] with ||f*f||_inf <= 1.378,
  c_emp(f) := ||f*f||_2^2 / (||f*f||_inf * ||f*f||_1) <= 0.88254 (= log 16/pi).

KNOWN BENCHMARKS (M, c_emp):
  MV near-extremizer: (1.275, 0.589)
  Indicator on [-1/4,1/4]: (2.000, 0.667)
  Boyer-Li 575-step: (1.652, 0.902)
  Symmetric Gaussian on [-1/4,1/4]: M >> 2 (poor; concentrated)

Discrete representation: v = [v_0,...,v_{N-1}] nonneg step heights on N
contiguous cells. Rescaled to [-1/4, 1/4]:
  M = (2N) * max(v conv v) / (sum v)^2
  c_emp = sum(L^2) / ((sum v)^2 * max L)   (shape-invariant)
  ||g*g||_2^2 = c_emp * M.

OBSERVATION: The "natural" M of unimodal shapes (Gaussians, parabolas,
indicator, sech) is >= 2. Only NON-MONOTONIC oscillating shapes (BL,
MV-class) can achieve M < 2. So our search space must contain BL-like
multimodal shapes.

APPROACH:
  1. Normalize coefficients early (avoid float64 overflow).
  2. Design BL-INSPIRED multimodal seeds (cosine-poly-modulated indicators,
     squashed BL, BL with scaled heads/tails, BL+shift+convex combos, etc.).
  3. Project onto M = M_max via a one-parameter family that REDUCES M:
       v_alpha = alpha * v + (1-alpha) * indicator_window
     where the indicator_window has M = 2; convex combo with the small-M
     v drives M continuously from M(v) to 2. Bisect alpha to hit M_max.
  4. ALSO try the "lift" direction: scale BL down (=> divide coeffs around the
     peak by some r > 1 only) to LOWER M.
  5. Adversarial gradient ascent on c_emp under hard M-constraint.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

C_STAR = float(np.log(16.0) / np.pi)   # 0.88254240061060637
M_MAX  = 1.378

ROOT = Path(__file__).resolve().parent
COEFF_FILE = ROOT / "coeffBL.txt"


def load_BL_coeffs():
    txt = COEFF_FILE.read_text().strip().lstrip("{").rstrip("}")
    v = np.array([int(x.strip()) for x in txt.split(",") if x.strip()], dtype=np.float64)
    # Normalize to max ~ 1 for numerical stability (c_emp/M are scale-invariant)
    v /= v.max()
    return v


def Mc(v):
    """Return (M, c_emp) for nonneg sequence v rescaled to [-1/4, 1/4]."""
    v = np.asarray(v, dtype=np.float64)
    if np.any(v < 0):
        v = np.maximum(v, 0.0)
    N = len(v)
    S = v.sum()
    if S <= 0 or not np.isfinite(S):
        return float("nan"), float("nan")
    L = fftconvolve(v, v, mode="full")
    Mpeak = L.max()
    if Mpeak <= 0 or not np.isfinite(Mpeak):
        return float("nan"), float("nan")
    M = (2.0 * N) * Mpeak / (S * S)
    c_emp = (L * L).sum() / (S * S * Mpeak)
    return float(M), float(c_emp)


# ----------------------------------------------------------------------
# Shape transforms (non-overflowing)

def smooth_gauss(v, sigma):
    if sigma <= 0:
        return v.copy()
    half = max(1, int(np.ceil(4*sigma)))
    xs = np.arange(-half, half+1)
    w = np.exp(-0.5 * (xs/sigma)**2); w /= w.sum()
    return fftconvolve(v, w, mode="same")


def fft_truncate(v, keep):
    N = len(v)
    F = np.fft.rfft(v)
    F[keep+1:] = 0
    out = np.fft.irfft(F, n=N)
    return np.maximum(out, 0.0)


# ----------------------------------------------------------------------
# Projections to M = M_target

def project_via_indicator_blend(v, M_target, tol=1e-8, max_iter=80):
    """v_alpha = alpha * v + (1-alpha) * indicator(N).
    M(alpha=1) = M(v); M(alpha=0) = 2 (indicator). Monotone in alpha.
    If M(v) < M_target < 2, bisect on alpha in [0,1].
    """
    N = len(v)
    ind = np.ones(N) * (v.sum() / N)   # same total mass
    M_v, _ = Mc(v)
    M_i, _ = Mc(ind)  # = 2.0
    if not (min(M_v, M_i) - 1e-3 <= M_target <= max(M_v, M_i) + 1e-3):
        return None, None, None
    a, b = 0.0, 1.0
    Ma, Mb = M_i, M_v
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        v_m = m * v + (1 - m) * ind
        Mm, _ = Mc(v_m)
        if abs(Mm - M_target) < tol:
            return v_m, m, Mm
        # M decreasing in m if M_v < M_i (M(v)=1.65 < M(ind)=2)
        if Mm < M_target:
            # need more indicator (smaller m) -> b = m
            if Ma < Mb:
                b, Mb = m, Mm
            else:
                a, Ma = m, Mm
        else:
            if Ma < Mb:
                a, Ma = m, Mm
            else:
                b, Mb = m, Mm
    m = 0.5 * (a + b)
    v_m = m * v + (1 - m) * ind
    return v_m, m, Mc(v_m)[0]


def project_via_pad(v, M_target, max_pad_factor=20, tol=1e-8, max_iter=60):
    """Right-pad v with zeros: v_pad = [v, 0,...,0] of total length k*N.
    As k grows, the rescaled support shrinks the relative width of the
    nonzero portion, which TYPICALLY reduces M. Bisect k continuously by
    interpolating two adjacent integer pads.
    """
    # (We use integer pad-lengths here — discrete, simpler.)
    N0 = len(v)
    # Compute M for various pad lengths
    table = []
    for k in [1, 2, 3, 4, 6, 8, 12, 16, 20]:
        pad_len = N0 * k - N0
        v_pad = np.concatenate([v, np.zeros(pad_len)])
        M, c = Mc(v_pad)
        table.append((k, M, c, v_pad))
    table.sort(key=lambda t: t[1])  # by M
    # find adjacent pair bracketing M_target
    for i in range(len(table)-1):
        ka, Ma, ca, _ = table[i]
        kb, Mb, cb, _ = table[i+1]
        if (Ma <= M_target <= Mb) or (Mb <= M_target <= Ma):
            # pick whichever is closer
            if abs(Ma - M_target) < abs(Mb - M_target):
                return table[i][3], table[i][0], table[i][1]
            else:
                return table[i+1][3], table[i+1][0], table[i+1][1]
    return None, None, None


def project_via_window(v, M_target, tol=1e-8, max_iter=60):
    """Effective-support widening: pad both sides symmetrically with a small
    'baseline' constant b > 0. As b grows, mass redistributes toward uniform,
    reducing M. Bisect b in [0, max_b].
    """
    N = len(v)
    # b in [0, 2*v.max()] should bracket M -> 2
    lo, hi = 0.0, 5.0 * v.max()
    M_lo, _ = Mc(v + lo)
    M_hi, _ = Mc(v + hi)
    if not (min(M_lo, M_hi) - 1e-3 <= M_target <= max(M_lo, M_hi) + 1e-3):
        return None, None, None
    a, b = lo, hi
    Ma, Mb = M_lo, M_hi
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        Mm, _ = Mc(v + m)
        if abs(Mm - M_target) < tol:
            return v + m, m, Mm
        if Ma > Mb:
            if Mm > M_target:
                a, Ma = m, Mm
            else:
                b, Mb = m, Mm
        else:
            if Mm < M_target:
                a, Ma = m, Mm
            else:
                b, Mb = m, Mm
    m = 0.5 * (a + b)
    return v + m, m, Mc(v + m)[0]


def project_to_M(v, M_target):
    """Try all projections; return the one with highest c_emp."""
    candidates = []
    for fn in (project_via_indicator_blend, project_via_window, project_via_pad):
        res = fn(v, M_target)
        if res is None or res[0] is None:
            continue
        vp, _, M_actual = res
        if M_actual is None or not np.isfinite(M_actual):
            continue
        if abs(M_actual - M_target) > 5e-3:
            continue
        c = Mc(vp)[1]
        if not np.isnan(c):
            candidates.append((c, vp, fn.__name__))
    if not candidates:
        return None
    candidates.sort(key=lambda t: -t[0])
    return candidates[0][1]


# ----------------------------------------------------------------------
# BL-inspired multimodal seeds

def BL_template():
    return load_BL_coeffs()


def make_bimodal(N, gap_frac, head_w, tail_w, head_height=1.0, tail_height=0.5,
                  seed=None):
    """Two flat-ish blocks with a gap of zeros between, of total length N."""
    rng = np.random.default_rng(seed)
    head_len = int(head_w * N)
    tail_len = int(tail_w * N)
    gap_len = int(gap_frac * N)
    total = head_len + gap_len + tail_len
    if total > N:
        scale = N / total
        head_len = int(head_len * scale)
        gap_len = int(gap_len * scale)
        tail_len = N - head_len - gap_len
    head = head_height * (1 + 0.05 * rng.standard_normal(head_len))
    head = np.maximum(head, 0)
    tail = tail_height * (1 + 0.05 * rng.standard_normal(tail_len))
    tail = np.maximum(tail, 0)
    gap = np.zeros(gap_len)
    pad = np.zeros(N - len(head) - len(gap) - len(tail))
    v = np.concatenate([head, gap, tail, pad])
    return v[:N]


def make_multimodal(N, n_modes, mode_w, gap_w, heights, seed=None):
    """N cells; n_modes blocks of width mode_w, separated by gap_w zeros."""
    rng = np.random.default_rng(seed)
    v = []
    for i in range(n_modes):
        h = heights[i]
        block = h * (1 + 0.05 * rng.standard_normal(mode_w))
        block = np.maximum(block, 0)
        v.append(block)
        if i < n_modes - 1:
            v.append(np.zeros(gap_w))
    out = np.concatenate(v)
    if len(out) > N:
        return out[:N]
    return np.concatenate([out, np.zeros(N - len(out))])


def shaped_BL_variant(rng):
    """Random bimodal/multimodal/BL-inspired variant."""
    style = rng.integers(0, 8)
    if style == 0:
        # Pure BL
        return load_BL_coeffs()
    if style == 1:
        # BL + zero padding
        v = load_BL_coeffs()
        pad = int(rng.uniform(0.2, 4.0) * len(v))
        return np.concatenate([v, np.zeros(pad)])
    if style == 2:
        # BL with head amplified
        v = load_BL_coeffs().copy()
        k = int(rng.uniform(0.05, 0.20) * len(v))
        amp = rng.uniform(1.1, 2.5)
        v[:k] *= amp
        return v
    if style == 3:
        # BL with head squashed
        v = load_BL_coeffs().copy()
        k = int(rng.uniform(0.05, 0.20) * len(v))
        amp = rng.uniform(0.4, 0.9)
        v[:k] *= amp
        return v
    if style == 4:
        # bimodal
        N = int(rng.uniform(100, 600))
        return make_bimodal(N, gap_frac=rng.uniform(0.10, 0.40),
                              head_w=rng.uniform(0.05, 0.30),
                              tail_w=rng.uniform(0.30, 0.70),
                              head_height=rng.uniform(0.5, 2.0),
                              tail_height=rng.uniform(0.3, 1.2),
                              seed=int(rng.integers(0, 2**32-1)))
    if style == 5:
        # BL but smoothed
        sig = rng.uniform(0.3, 5.0)
        return smooth_gauss(load_BL_coeffs(), sig)
    if style == 6:
        # multimodal with random heights
        n = rng.integers(2, 6)
        N = int(rng.uniform(150, 700))
        mw = max(2, int(rng.uniform(0.05, 0.25) * N))
        gw = max(2, int(rng.uniform(0.05, 0.30) * N))
        heights = rng.uniform(0.2, 2.0, n)
        return make_multimodal(N, n, mw, gw, heights,
                               seed=int(rng.integers(0, 2**32-1)))
    # style == 7: BL FFT-truncated then re-amplified
    v = load_BL_coeffs()
    K = int(rng.integers(20, 287))
    return fft_truncate(v, K)


# ----------------------------------------------------------------------
# Adversarial search

def adversarial_step(v, rng, M_target, lr=0.04, n_perturb=24,
                      perturb_kind="gauss"):
    v_proj = project_to_M(v, M_target)
    if v_proj is None:
        return v, Mc(v)[1]
    base_c = Mc(v_proj)[1]
    best = (base_c, v_proj)
    Nv = len(v_proj)
    base_scale = lr * v_proj.mean()
    for _ in range(n_perturb):
        if perturb_kind == "gauss":
            d = rng.standard_normal(Nv)
        elif perturb_kind == "spike":
            d = np.zeros(Nv)
            i = rng.integers(0, Nv)
            d[i] = rng.choice([-1, 1])
        elif perturb_kind == "lowfreq":
            # smooth perturbation
            d = rng.standard_normal(Nv)
            d = smooth_gauss(d, rng.uniform(2, 10))
        elif perturb_kind == "block":
            d = np.zeros(Nv)
            i = rng.integers(0, Nv)
            j = min(Nv, i + rng.integers(2, 20))
            d[i:j] = rng.choice([-1, 1])
        else:
            d = rng.standard_normal(Nv)
        v_try = np.maximum(v_proj + base_scale * d, 0.0)
        v_try = project_to_M(v_try, M_target)
        if v_try is None:
            continue
        c_try = Mc(v_try)[1]
        if not np.isnan(c_try) and c_try > best[0]:
            best = (c_try, v_try)
    return best[1], best[0]


# ----------------------------------------------------------------------

def main():
    rng = np.random.default_rng(20260503)

    print("=" * 78)
    print("Adversarial Hyp_R search: counterexample with M <= 1.378, c_emp > 0.88254")
    print("=" * 78)
    print(f"c_* = log(16)/pi = {C_STAR:.10f}")
    print(f"M_max = {M_MAX}")

    v_BL = load_BL_coeffs()
    M_BL, c_BL = Mc(v_BL)
    print(f"\n[BL witness] N={len(v_BL)}, M={M_BL:.6f}, c_emp={c_BL:.6f}")
    assert abs(M_BL - 1.6520) < 1e-3 and abs(c_BL - 0.9016) < 1e-3, \
        f"BL mismatch M={M_BL}, c_emp={c_BL}"

    # State
    best_in_regime = (-1.0, None, "")
    best_global    = (c_BL, v_BL.copy(), "BL_raw")
    n_trials_total = 0
    pareto_pts = []

    def update(c, M, v, label, store=True):
        nonlocal best_in_regime, best_global
        if not np.isfinite(c) or not np.isfinite(M):
            return
        if store:
            pareto_pts.append((M, c, label))
        if c > best_global[0]:
            best_global = (c, v.copy() if v is not None else None, label)
        if M <= M_MAX + 1e-9 and c > best_in_regime[0]:
            best_in_regime = (c, v.copy() if v is not None else None, label)
        if M <= M_MAX + 1e-9 and c > C_STAR:
            print(f"  *** COUNTEREXAMPLE: {label}: M={M:.6f}, c_emp={c:.6f} ***")

    # ---- S0: Direct BL transformations (no projection) ----
    print(f"\n[S0] Direct BL-family transformations (natural M)")
    direct_variants = []
    direct_variants.append(("BL_raw", v_BL))
    for s in [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
        direct_variants.append((f"BL_smooth_s{s}", smooth_gauss(v_BL, s)))
    for K in [10, 25, 50, 100, 200, 287]:
        direct_variants.append((f"BL_fft{K}", fft_truncate(v_BL, K)))
    for r in [1.5, 2.0, 3.0, 5.0]:
        direct_variants.append((f"BL_pow{r}", v_BL ** r))
    # padded BL variants
    for k in [2, 3, 4, 6, 8]:
        direct_variants.append((f"BL_padx{k}", np.concatenate([v_BL, np.zeros(len(v_BL)*(k-1))])))
    # head-scaled BL
    for amp in [0.5, 0.7, 0.9, 1.1, 1.5, 2.0]:
        for kfrac in [0.05, 0.10, 0.20]:
            v = v_BL.copy()
            k = int(kfrac * len(v_BL))
            v[:k] *= amp
            direct_variants.append((f"BL_amp{amp}_k{kfrac}", v))
    for label, v in direct_variants:
        if v.sum() <= 0:
            continue
        M, c = Mc(v)
        n_trials_total += 1
        update(c, M, v, label)

    sub = [(M, c, l) for (M, c, l) in pareto_pts if 1.30 <= M <= 1.50]
    sub.sort(key=lambda t: -t[1])
    print(f"  Variants in M in [1.30,1.50] (top 12 by c_emp):")
    for (M, c, l) in sub[:12]:
        marker = "  <-- IN REGIME" if M <= M_MAX else ""
        print(f"    M={M:.5f}  c_emp={c:.5f}  {l}{marker}")
    sub = [(M, c, l) for (M, c, l) in pareto_pts if 1.50 <= M <= 1.70]
    sub.sort(key=lambda t: -t[1])
    print(f"  Variants in M in [1.50,1.70] (top 8):")
    for (M, c, l) in sub[:8]:
        print(f"    M={M:.5f}  c_emp={c:.5f}  {l}")

    # ---- S1: Projection of BL family to M = 1.378 ----
    print(f"\n[S1] Project BL family seeds to M = {M_MAX}")
    seeds_S1 = [v_BL] + [smooth_gauss(v_BL, s) for s in [0.5, 1.0, 2.0, 5.0]]
    for K in [10, 25, 50, 100, 200]:
        seeds_S1.append(fft_truncate(v_BL, K))
    for amp in [0.7, 0.9, 1.5, 2.0]:
        for kf in [0.05, 0.10, 0.20]:
            v = v_BL.copy()
            v[:int(kf*len(v))] *= amp
            seeds_S1.append(v)
    proj_results = []
    for i, seed in enumerate(seeds_S1):
        if seed.sum() <= 0: continue
        v_proj = project_to_M(seed, M_MAX)
        n_trials_total += 1
        if v_proj is None: continue
        M, c = Mc(v_proj)
        update(c, M, v_proj, f"proj_BL{i}")
        proj_results.append((c, M, v_proj, i))
    proj_results.sort(key=lambda t: -t[0])
    print(f"  Top 10 projections (c_emp, M):")
    for (c, M, _, i) in proj_results[:10]:
        marker = "  <-- IN REGIME" if M <= M_MAX else ""
        print(f"    c={c:.5f}  M={M:.5f}  seed#{i}{marker}")

    # ---- S2: Multimodal random shapes ----
    print(f"\n[S2] Random BL-style multimodal shapes (3000 samples)")
    for trial in range(3000):
        v = shaped_BL_variant(rng)
        if v.sum() <= 0: continue
        M, c = Mc(v)
        n_trials_total += 1
        update(c, M, v, f"multimode_t{trial}")
    sub = [(M, c, l) for (M, c, l) in pareto_pts if 1.30 <= M <= 1.50]
    sub.sort(key=lambda t: -t[1])
    print(f"  Random multimode in M in [1.30,1.50] (top 10):")
    for (M, c, l) in sub[:10]:
        marker = "  <-- IN REGIME" if M <= M_MAX else ""
        print(f"    M={M:.5f}  c_emp={c:.5f}  {l}{marker}")

    # ---- S3: Adversarial ascent on best 5 in-regime candidates ----
    print(f"\n[S3] Adversarial ascent on top in-regime candidates")
    in_regime_pts = sorted(
        [(c, M, l) for (M, c, l) in pareto_pts if M <= M_MAX + 1e-3],
        key=lambda t: -t[0])
    if in_regime_pts:
        # We need to recover the v's. Re-collect them from proj_results
        # plus the multimode trials (we don't store v's there). Just use BL projections.
        for rank, (c0, M0, _, _) in enumerate(proj_results[:5]):
            if M0 > M_MAX + 1e-3:
                continue
            v_cur = proj_results[rank][2].copy()
            t0 = time.time()
            for it in range(80):
                v_cur, c = adversarial_step(v_cur, rng, M_MAX,
                                              lr=0.05,
                                              perturb_kind=rng.choice(["gauss","lowfreq","block","spike"]),
                                              n_perturb=24)
                M_cur, _ = Mc(v_cur)
                n_trials_total += 24
                update(c, M_cur, v_cur, f"ascent_seed{rank}_it{it}")
                if time.time() - t0 > 60:
                    break
            print(f"  seed{rank}: c_emp {c0:.5f} -> {Mc(v_cur)[1]:.5f}")

    # ---- S4: Aggressive random restart ascent ----
    print(f"\n[S4] 100 random restarts (5-min budget)")
    t0 = time.time()
    n_rr = 0
    while time.time() - t0 < 300 and n_rr < 100:
        n_rr += 1
        v0 = shaped_BL_variant(rng)
        if v0.sum() <= 0: continue
        v_cur = project_to_M(v0, M_MAX)
        if v_cur is None: continue
        for it in range(20):
            v_cur, c = adversarial_step(v_cur, rng, M_MAX,
                                          lr=0.04,
                                          perturb_kind=rng.choice(["gauss","lowfreq","block"]),
                                          n_perturb=12)
            n_trials_total += 12
            if v_cur is None: break
        if v_cur is not None:
            M, c = Mc(v_cur)
            update(c, M, v_cur, f"rr{n_rr}")
    print(f"  ran {n_rr} restarts, best in regime now = {best_in_regime[0]:.5f}, "
          f"global best = {best_global[0]:.5f}")

    # ============ FINAL REPORT ============
    print("\n" + "=" * 78)
    print("FINAL REPORT")
    print("=" * 78)
    print(f"Total trials evaluated: ~{n_trials_total}")

    bins = [(1.27, 1.30), (1.30, 1.34), (1.34, 1.378), (1.378, 1.45),
            (1.45, 1.55), (1.55, 1.66)]
    print(f"\nPareto frontier (max c_emp per M-bin):")
    print(f"  {'M-range':<16} {'best c_emp':>10}  {'gap to c_*':>11}  source")
    for lo, hi in bins:
        sub = [(M, c, l) for (M, c, l) in pareto_pts if lo <= M < hi]
        if not sub:
            print(f"  {f'[{lo:.3f}, {hi:.3f})':<16} {'(none)':>10}")
            continue
        sub.sort(key=lambda t: -t[1])
        M, c, l = sub[0]
        marker = "  IN REGIME" if hi <= M_MAX + 1e-9 else ""
        print(f"  {f'[{lo:.3f}, {hi:.3f})':<16} {c:>10.5f}  {C_STAR-c:>+11.5f}  "
              f"{l[:50]}{marker}")

    if best_in_regime[1] is not None:
        c_in, v_in, label_in = best_in_regime
        M_in, _ = Mc(v_in)
        print(f"\nBEST IN RESTRICTED REGIME (M <= {M_MAX}):")
        print(f"  source = {label_in}")
        print(f"  M      = {M_in:.8f}")
        print(f"  c_emp  = {c_in:.8f}")
        print(f"  c_*    = {C_STAR:.8f}")
        gap = C_STAR - c_in
        print(f"  gap    = c_* - c_emp = {gap:+.6f}")
        if c_in > C_STAR:
            verdict = "DISPROVED"
        elif c_in > C_STAR - 0.02:
            verdict = "SUSPECT"
        else:
            verdict = "SURVIVED"
    else:
        verdict = "SURVIVED"

    print()
    print(f"VERDICT: {verdict}")
    print(f"BEST GLOBAL (any M): label={best_global[2]}, c_emp={best_global[0]:.6f}")

    out = {
        "verdict": verdict,
        "c_star": C_STAR,
        "M_max": M_MAX,
        "n_trials": n_trials_total,
        "best_in_regime": {
            "M": Mc(best_in_regime[1])[0] if best_in_regime[1] is not None else None,
            "c_emp": best_in_regime[0],
            "label": best_in_regime[2],
        },
        "best_global": {
            "c_emp": best_global[0],
            "label": best_global[2],
        },
        "pareto_frontier": [
            {"M_range": f"[{lo:.3f},{hi:.3f})",
             "best_c": max((c for (M, c, _) in pareto_pts if lo <= M < hi),
                            default=None)}
            for (lo, hi) in bins
        ],
    }
    out_path = ROOT / "adversarial_hypR_result.json"
    out_path.write_text(json.dumps(out, indent=2))
    if best_in_regime[1] is not None:
        np.savez(ROOT / "adversarial_hypR_witness.npz",
                  v=best_in_regime[1],
                  M=Mc(best_in_regime[1])[0],
                  c_emp=best_in_regime[0],
                  label=best_in_regime[2])
    return verdict


if __name__ == "__main__":
    main()
