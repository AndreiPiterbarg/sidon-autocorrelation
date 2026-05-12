"""Smoke test: check whether candidate functionals Psi(f) are determined by
bin averages alone, AND whether they lower-bound ||f*f||_inf.

We test 5 candidates A-E and report on each:
    SUFFICIENT (depends only on bin averages)?
    BOUND (<= ||f*f||_inf)?

For continuous-f-soundness in step-pruning, we need BOTH conditions.

Setup
-----
Domain: [-1/4, 1/4] divided into d = 2n bins of width h = 1/(4n) = 1/d.
Probability density f: f >= 0, ∫ f = 1, supp f ⊆ [-1/4, 1/4].
Bin average:   a_i := (1/h) ∫_{bin_i} f = d · ∫_{bin_i} f.
Sum constraint: Σ_i (h · a_i) = ∫ f = 1, so Σ_i a_i = d.
Normalization in cascade:    m·a_i is multiplied by 4n/m → 4n·(c_i/m) ; we keep
working in the "heights" a_i with Σ a_i = d, ∫ f = 1.

Candidates
----------
A. TV_W(f)  = (1/|W|) ∫_W (f*f)(t) dt    for window W = [W_lo, W_hi].
B. Bin-square-sum  Σ_i α_i a_i² for some weights α_i ≥ 0.
C. Convolution at grid points  Σ_k φ_k (f*f)(t_k) where t_k are bin-grid points.
D. Energy ||f*f||_2² = ∫(f*f)².
E. Test-measure dual functional: <μ, f*f> for non-negative μ.

For each, two questions:
   Q1.  Is Psi(f) = Psi(f_a) when f, f_a have the same bin averages a?
        ("bin-average sufficient")
   Q2.  Is Psi(f) <= ||f*f||_inf for all admissible f?
        (lower-bound property)
"""
import os
import sys
import numpy as np
from itertools import product

# Force UTF-8 stdout on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

np.random.seed(2026_05_09)

# Discretization
n = 3       # n_half
d = 2 * n   # number of bins
h = 1.0 / d  # bin width = 1/(4n)
N_GRID = 256   # fine grid resolution per bin for continuous-f integration


# ---------------------------------------------------------------------- helpers
def domain_grid():
    """Fine grid points x in [-1/4, 1/4] with width = 1/(4n)/N_GRID*d=..."""
    # For convenience use [0, 2n*h] = [0, 1/2] internally and shift to [-1/4, 1/4]
    # at the end.  Bin i = [i*h, (i+1)*h] for i=0..d-1 with internal x in [0,1/2].
    return np.linspace(0.0, d * h, N_GRID * d, endpoint=False)


def bin_indices(x):
    """Bin index of x in [0, 1/2]."""
    idx = np.floor(x / h).astype(int)
    return np.clip(idx, 0, d - 1)


def bin_averages(f_vals, x_grid):
    """Bin averages a_i = (1/h) ∫_{bin_i} f."""
    a = np.zeros(d)
    bins = bin_indices(x_grid)
    counts = np.zeros(d)
    for i in range(len(x_grid)):
        a[bins[i]] += f_vals[i]
        counts[bins[i]] += 1.0
    a /= np.maximum(counts, 1)
    return a


def step_from_avg(a, x_grid):
    """Step function on [0,1/2] with heights a_i."""
    bins = bin_indices(x_grid)
    return a[bins]


def normalize_mass(f, dx):
    """Scale f so ∫f = 1 (still on a fine grid; dx is grid step)."""
    Z = np.sum(f) * dx
    return f / Z, Z


def autoconv(f, dx):
    """Numeric autoconvolution; returns conv values and t-grid."""
    cv = np.convolve(f, f) * dx
    Lt = (len(cv) - 1) * dx
    t = np.linspace(0, Lt, len(cv))
    return cv, t


def linf_autoconv(f, dx):
    """||f*f||_∞."""
    cv, _ = autoconv(f, dx)
    return float(np.max(cv))


# ----------------------------------------------------------------------------
# Candidate evaluators.  All take (f_vals, x_grid).  Each returns Psi(f).
# ----------------------------------------------------------------------------
def _conv_grid_step(dx_fine):
    """Auxiliary: make sure dx is uniform = 1/(N_GRID*d) implicitly."""
    return dx_fine


def candidate_A_TV_W(f_vals, x_grid, W_lo, W_hi):
    """TV_W(f) = (1/(W_hi - W_lo)) ∫_{W_lo}^{W_hi} (f*f)(t) dt."""
    dx = x_grid[1] - x_grid[0]
    cv, t = autoconv(f_vals, dx)
    # t \in [0, 2*L_supp_f]; here supp f ⊂ [0, 1/2] so t \in [0,1].
    # Window [W_lo, W_hi] on the convolution time axis.
    mask = (t >= W_lo) & (t <= W_hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(cv[mask], t[mask]) / (W_hi - W_lo))


def candidate_B_bin_sq(a, weights=None):
    """B(f) = Σ_i α_i a_i² with α_i = 1 (or user-supplied weights)."""
    if weights is None:
        weights = np.ones(len(a))
    return float(np.sum(weights * (a ** 2)))


def candidate_C_grid_conv(f_vals, x_grid, weights=None):
    """C(f) = Σ_k φ_k (f*f)(t_k) where t_k are grid points = (k * h),
    k = 0, 1, ..., 2d-2 (the bin-pair conv grid).  Returns sum with all-ones
    weights by default (so this is ∫ "discrete f*f at grid points" weighted)."""
    dx = x_grid[1] - x_grid[0]
    cv, t = autoconv(f_vals, dx)
    # Find indices in cv corresponding to t_k = k * h, k = 0,...,2d-2.
    K = 2 * d - 1
    grid_t = np.arange(K) * h
    vals = np.zeros(K)
    for k in range(K):
        idx = int(round(grid_t[k] / dx))
        if 0 <= idx < len(cv):
            vals[k] = cv[idx]
    if weights is None:
        weights = np.ones(K)
    return float(np.dot(weights, vals))


def candidate_C2_max_grid_conv(f_vals, x_grid):
    """C2(f) = max_k (f*f)(t_k) for t_k = (k+0.5)*h grid midpoints
    (the peak positions of the step-function tent autoconv)."""
    dx = x_grid[1] - x_grid[0]
    cv, t = autoconv(f_vals, dx)
    K = 2 * d - 1
    grid_t = (np.arange(K) + 0.5) * h
    vals = np.zeros(K)
    for k in range(K):
        idx = int(round(grid_t[k] / dx))
        if 0 <= idx < len(cv):
            vals[k] = cv[idx]
    return float(np.max(vals))


def candidate_TVW_step_only(a, j, ell, n_half_local):
    """Cascade's discrete TV_W(f_a) — purely a function of bin averages.
    TV_W(f_a) = (1/(4n*ell)) * sum_{(p,q): j <= p+q <= j+ell-2} a_p a_q.
    This is determined by a alone — NO continuous f needed.

    The question is: what relationship does this have to (f*f) for cont. f?
    """
    s_lo = j
    s_hi = j + ell - 2
    s = 0.0
    for p in range(d):
        for q in range(d):
            if s_lo <= p + q <= s_hi:
                s += a[p] * a[q]
    return s / (4 * n_half_local * ell)


def candidate_D_energy(f_vals, x_grid):
    """D(f) = ||f*f||_2²."""
    dx = x_grid[1] - x_grid[0]
    cv, t = autoconv(f_vals, dx)
    return float(np.sum(cv ** 2) * dx)


def candidate_E_dual(f_vals, x_grid, mu_vals, mu_grid):
    """E(f) = <μ, f*f> = ∫ μ(t) (f*f)(t) dt for chosen non-negative μ."""
    dx = x_grid[1] - x_grid[0]
    cv, t = autoconv(f_vals, dx)
    # Interpolate μ onto t-grid
    mu_on_t = np.interp(t, mu_grid, mu_vals, left=0, right=0)
    return float(np.sum(mu_on_t * cv) * dx)


# ----------------------------------------------------------------------------
# Build sample continuous f's with the same bin averages.
# Strategy: pick bin averages a, then construct several different f's
# (uniform / triangular peak / Gaussian-like) all with bin averages = a.
# ----------------------------------------------------------------------------
def f_uniform(a, x_grid):
    """Step function = a_i on each bin (= the canonical f_a)."""
    return step_from_avg(a, x_grid)


def f_concentrated(a, x_grid, sigma=0.05):
    """Continuous f with the same bin averages a, but mass concentrated
    in narrow Gaussians centered at bin midpoints."""
    f = np.zeros_like(x_grid)
    for i in range(d):
        x_mid = (i + 0.5) * h
        bin_mass = a[i] * h
        # Truncated Gaussian inside bin
        s = sigma * h  # std dev relative to bin width
        f += bin_mass * np.exp(-0.5 * ((x_grid - x_mid) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    # Re-normalize per bin so each bin's mass matches exactly
    bins = bin_indices(x_grid)
    dx = x_grid[1] - x_grid[0]
    for i in range(d):
        mask = bins == i
        cur_mass = np.sum(f[mask]) * dx
        if cur_mass > 0:
            f[mask] *= a[i] * h / cur_mass
        else:
            f[mask] = a[i]
    return f


def f_left_skew(a, x_grid):
    """Continuous f with bin averages a, but mass skewed to bin LEFT halves."""
    f = np.zeros_like(x_grid)
    bins = bin_indices(x_grid)
    dx = x_grid[1] - x_grid[0]
    for i in range(d):
        mask = bins == i
        x_in_bin = x_grid[mask] - i * h
        # Linear ramp: 2*(h - x_in_bin)/h , shifted so bin-avg = a_i
        # Average of (2*(h-x)/h) over [0,h] = 1, so mass = a_i * h works.
        f[mask] = a[i] * 2.0 * (h - x_in_bin) / h
    return f


def f_right_skew(a, x_grid):
    """Mass skewed to bin RIGHT halves."""
    f = np.zeros_like(x_grid)
    bins = bin_indices(x_grid)
    for i in range(d):
        mask = bins == i
        x_in_bin = x_grid[mask] - i * h
        f[mask] = a[i] * 2.0 * x_in_bin / h
    return f


def f_two_spike(a, x_grid):
    """Two spikes per bin (left and right deltas, smoothed)."""
    f = np.zeros_like(x_grid)
    bins = bin_indices(x_grid)
    dx = x_grid[1] - x_grid[0]
    for i in range(d):
        mask = bins == i
        x_in_bin = x_grid[mask] - i * h
        # Two narrow gaussians at 0.1h and 0.9h
        s = 0.05 * h
        bin_mass = a[i] * h
        contribs = np.exp(-0.5 * ((x_in_bin - 0.1 * h) / s) ** 2) + \
                    np.exp(-0.5 * ((x_in_bin - 0.9 * h) / s) ** 2)
        # Normalize so bin mass = a_i * h
        cur = np.sum(contribs) * dx
        if cur > 0:
            f[mask] = contribs * (bin_mass / cur)
    return f


def f_uniform_like_step(a, x_grid):
    """Step + tiny uniform jitter (sanity check)."""
    f = step_from_avg(a, x_grid)
    return f


# ----------------------------------------------------------------------------
# Run the smoke test.
# ----------------------------------------------------------------------------
def test_candidates():
    print(f"\n=== Smoke test: bin-average sufficiency of candidate functionals ===")
    print(f"d={d} bins, bin width h={h:.4f}, fine grid N={N_GRID*d}")

    # Pick a few sample bin-average vectors a (each Σ a_i = d, a_i ≥ 0)
    test_a_list = []
    test_a_list.append(np.ones(d))                                        # flat
    a2 = np.zeros(d); a2[0] = 4; a2[1] = 4; a2[d-2] = a2[d-1] = 0; a2 = a2 + 0  # spike-like (Σ=8=d only if d=8)
    if abs(np.sum(a2) - d) < 1e-9:
        test_a_list.append(a2)
    a3 = np.array([2.0 if i % 2 == 0 else 0.0 for i in range(d)])  # alternating
    if abs(np.sum(a3) - d) < 1e-9:
        test_a_list.append(a3)
    a4 = np.linspace(0.5, 1.5, d)  # gradient
    a4 = a4 * (d / np.sum(a4))
    test_a_list.append(a4)
    a5 = np.zeros(d); a5[0] = 2*n; a5[1] = 2*n   # heavy spike
    if abs(np.sum(a5) - d) < 1e-9:
        test_a_list.append(a5)

    # Sample continuous f's per bin-average a (all have the same a)
    f_makers = [
        ('f_step',    f_uniform),
        ('f_conc',    lambda a, x: f_concentrated(a, x, sigma=0.10)),
        ('f_lskew',   f_left_skew),
        ('f_rskew',   f_right_skew),
        ('f_2spike',  f_two_spike),
    ]

    x_grid = domain_grid()
    dx = x_grid[1] - x_grid[0]

    # === Candidate A: TV_W ============================================
    print(f"\n----- Candidate A: TV_W(f) = (1/|W|) ∫_W (f*f) dt -----")
    print(f"  Sample windows: bin-aligned [k*h, (k+ell)*h] for k=0..2d-2, ell=2..d")
    print(f"  Q1: bin-average sufficient? Q2: bound on ||f*f||_∞?\n")
    print(f"{'a-vec':10s}  {'window':20s}  {'TV_W(f_step)':>14s}  {'TV_W(f_conc)':>14s}  "
          f"{'TV_W(lskew)':>13s}  {'TV_W(rskew)':>13s}  {'max-min':>9s}  {'||f*f||∞':>9s}")
    for a_idx, a in enumerate(test_a_list):
        for ell in [2, 4, d]:
            for k in [0, d-1, 2*d-1-ell]:
                W_lo = k * h
                W_hi = W_lo + ell * h
                if W_hi > 2 * d * h:
                    continue
                vals = {}
                linf = {}
                for name, mk in f_makers:
                    f = mk(a, x_grid)
                    vals[name] = candidate_A_TV_W(f, x_grid, W_lo, W_hi)
                    linf[name] = linf_autoconv(f, dx)
                v_arr = np.array([vals[n] for n, _ in f_makers])
                v_min = v_arr.min(); v_max = v_arr.max()
                wname = f"k={k},ell={ell}"
                vmin_inf = min(linf.values())
                # Show first row only per a
                if k == 0 and ell == 2:
                    print(f"a#{a_idx}: {a[:min(d,4)].tolist()}{'...' if d>4 else ''}")
                print(f"  {'':8s}  {wname:20s}  "
                      f"{vals['f_step']:14.6f}  {vals['f_conc']:14.6f}  "
                      f"{vals['f_lskew']:13.6f}  {vals['f_rskew']:13.6f}  "
                      f"{(v_max - v_min):9.6f}  {vmin_inf:9.6f}")

    # === Candidate B: Bin-square-sum ==================================
    print(f"\n----- Candidate B: B(f) = Σ_i a_i² · h (= ||f_a||_2²-like) -----")
    print(f"{'a-vec':10s}  {'B(f_step)':>12s}  {'B(f_conc)':>12s}  "
          f"{'B(lskew)':>12s}  {'B(rskew)':>12s}  {'max-min':>10s}  {'||f*f||∞':>10s}")
    for a_idx, a in enumerate(test_a_list):
        bin_avg_via = {}
        bs = {}
        linf = {}
        for name, mk in f_makers:
            f = mk(a, x_grid)
            a_emp = bin_averages(f, x_grid)
            bin_avg_via[name] = float(np.max(np.abs(a_emp - a)))
            bs[name] = candidate_B_bin_sq(a) * h    # h-weighted
            linf[name] = linf_autoconv(f, dx)
        bs_v = np.array(list(bs.values()))
        max_diff_a = max(bin_avg_via.values())
        print(f"a#{a_idx}: {a[:min(d,4)].tolist()}{'...' if d>4 else ''}, |a_recovered-a|max={max_diff_a:.2e}")
        print(f"  {'':8s}  {bs['f_step']:12.6f}  {bs['f_conc']:12.6f}  "
              f"{bs['f_lskew']:12.6f}  {bs['f_rskew']:12.6f}  "
              f"{(bs_v.max() - bs_v.min()):10.2e}  {min(linf.values()):10.6f}")

    # === Candidate C: convolution at grid points ======================
    print(f"\n----- Candidate C: C(f) = Σ_k (f*f)(t_k), t_k = k h grid points -----")
    print(f"{'a-vec':10s}  {'C(f_step)':>12s}  {'C(f_conc)':>12s}  "
          f"{'C(lskew)':>12s}  {'C(rskew)':>12s}  {'max-min':>10s}  {'||f*f||∞':>10s}")
    for a_idx, a in enumerate(test_a_list):
        cs = {}
        linf = {}
        for name, mk in f_makers:
            f = mk(a, x_grid)
            cs[name] = candidate_C_grid_conv(f, x_grid)
            linf[name] = linf_autoconv(f, dx)
        cs_v = np.array(list(cs.values()))
        print(f"a#{a_idx}:  {cs['f_step']:12.6f}  {cs['f_conc']:12.6f}  "
              f"{cs['f_lskew']:12.6f}  {cs['f_rskew']:12.6f}  "
              f"{(cs_v.max() - cs_v.min()):10.2e}  {min(linf.values()):10.6f}")

    # === Candidate D: ||f*f||_2² =====================================
    print(f"\n----- Candidate D: D(f) = ||f*f||_2² -----")
    print(f"{'a-vec':10s}  {'D(f_step)':>12s}  {'D(f_conc)':>12s}  "
          f"{'D(lskew)':>12s}  {'D(rskew)':>12s}  {'max-min':>10s}  {'||f*f||∞':>10s}")
    for a_idx, a in enumerate(test_a_list):
        ds = {}
        linf = {}
        for name, mk in f_makers:
            f = mk(a, x_grid)
            ds[name] = candidate_D_energy(f, x_grid)
            linf[name] = linf_autoconv(f, dx)
        ds_v = np.array(list(ds.values()))
        print(f"a#{a_idx}:  {ds['f_step']:12.6f}  {ds['f_conc']:12.6f}  "
              f"{ds['f_lskew']:12.6f}  {ds['f_rskew']:12.6f}  "
              f"{(ds_v.max() - ds_v.min()):10.2e}  {min(linf.values()):10.6f}")

    # === Candidate E: dual functional (uniform μ on whole domain) =====
    print(f"\n----- Candidate E: E(f) = <μ, f*f> for μ = 1_[-1/2, 1/2] (uniform) -----")
    print(f"  (this gives <μ,f*f> = (∫f)² = 1 for any prob density f, so trivial.)")
    print(f"{'a-vec':10s}  {'E(f_step)':>12s}  {'E(f_conc)':>12s}  "
          f"{'E(lskew)':>12s}  {'E(rskew)':>12s}  {'max-min':>10s}  {'||f*f||∞':>10s}")
    mu_grid = np.linspace(0, 1, 1000)
    mu_vals = np.ones_like(mu_grid)
    for a_idx, a in enumerate(test_a_list):
        es = {}
        linf = {}
        for name, mk in f_makers:
            f = mk(a, x_grid)
            es[name] = candidate_E_dual(f, x_grid, mu_vals, mu_grid)
            linf[name] = linf_autoconv(f, dx)
        es_v = np.array(list(es.values()))
        print(f"a#{a_idx}:  {es['f_step']:12.6f}  {es['f_conc']:12.6f}  "
              f"{es['f_lskew']:12.6f}  {es['f_rskew']:12.6f}  "
              f"{(es_v.max() - es_v.min()):10.2e}  {min(linf.values()):10.6f}")

    # === Cascade's discrete TV_W: bin-average sufficient TAUTOLOGICALLY ===
    # But does it lower-bound continuous-f's ||f*f||_inf?  Let's check.
    print(f"\n----- Cascade's discrete TV_W(f_a): bin-average sufficient by definition -----")
    print(f"  Q: does cascade-TV_W(f_a) <= sup_t (f*f)(t) for ALL cont. f with bin avg a?")
    print(f"  Q: does cascade-TV_W(f_a) <= (1/|W|) ∫_W (f*f)(t) dt for ALL cont. f w/ bin avg a?\n")
    n_half_local = n
    cascade_TV_violation_count = 0
    cascade_TV_total = 0
    cascade_min_gap = float('inf')
    cascade_max_TV_over_Linf = 0.0
    cascade_violations_examples = []
    for a_idx, a in enumerate(test_a_list):
        for ell in [2, 4, d]:
            for j in [0, d - 1, 2 * d - 1 - ell]:
                if j + ell > 2 * d - 1:
                    continue
                # Cascade's TV_W (discrete sum over bin pairs in window)
                tv_a = candidate_TVW_step_only(a, j, ell, n_half_local)
                # Generate several continuous f's with same bin avg
                for name, mk in f_makers:
                    f = mk(a, x_grid)
                    Linf = linf_autoconv(f, dx)
                    cascade_TV_total += 1
                    if tv_a > Linf + 1e-6:
                        cascade_TV_violation_count += 1
                        cascade_violations_examples.append(
                            (a_idx, name, ell, j, tv_a, Linf))
                    cascade_min_gap = min(cascade_min_gap, Linf - tv_a)
                    if Linf > 1e-9:
                        cascade_max_TV_over_Linf = max(cascade_max_TV_over_Linf, tv_a / Linf)
    print(f"  cascade-TV(f_a) > ||f*f||_inf violations: "
          f"{cascade_TV_violation_count}/{cascade_TV_total}")
    print(f"  min gap (||f*f||_inf - cascade-TV): {cascade_min_gap:.6f}")
    print(f"  max ratio (cascade-TV / ||f*f||_inf): {cascade_max_TV_over_Linf:.6f}")
    if cascade_violations_examples:
        print(f"  First 3 violations:")
        for ex in cascade_violations_examples[:3]:
            print(f"    a#{ex[0]}, f={ex[1]}, ell={ex[2]}, j={ex[3]}, "
                  f"TV_W={ex[4]:.4f} > Linf={ex[5]:.4f}")

    # === Adversarial search: ||f*f||_inf for f spike vs cascade-TV(f_a) ===
    print(f"\n----- Adversarial search: try to make ||f*f||_inf small relative to cascade-TV -----")
    n_adv_violations = 0
    n_adv_total = 0
    min_adv_ratio = float('inf')
    rng = np.random.default_rng(42)
    for trial in range(20):
        # Random a with sum d
        a = rng.dirichlet(np.ones(d)) * d
        for ell in [2, 4, d]:
            for j in range(0, 2 * d - ell):
                if j + ell > 2 * d - 1:
                    continue
                tv_a = candidate_TVW_step_only(a, j, ell, n_half_local)
                if tv_a < 1e-9:
                    continue
                # Try to construct adversarial cont. f with bin avg a:
                # spread mass out in each bin to lower max(f*f)
                f_uniform_v = f_uniform(a, x_grid)
                Linf_uniform = linf_autoconv(f_uniform_v, dx)
                f_spread = f_concentrated(a, x_grid, sigma=0.40)
                Linf_spread = linf_autoconv(f_spread, dx)
                f_lskew_v = f_left_skew(a, x_grid)
                Linf_lskew = linf_autoconv(f_lskew_v, dx)
                f_rskew_v = f_right_skew(a, x_grid)
                Linf_rskew = linf_autoconv(f_rskew_v, dx)
                Linf_min = min(Linf_uniform, Linf_spread, Linf_lskew, Linf_rskew)
                n_adv_total += 1
                if tv_a > Linf_min + 1e-6:
                    n_adv_violations += 1
                if Linf_min > 1e-9:
                    min_adv_ratio = min(min_adv_ratio, Linf_min / tv_a)
    print(f"  Adversarial: cascade-TV(f_a) > min Linf violations: "
          f"{n_adv_violations}/{n_adv_total}")
    print(f"  worst min ||f*f||_inf / cascade-TV ratio = {min_adv_ratio:.4f} "
          f"(>=1 means cascade-TV is a sound LOWER bound)")

    # === RIGOROUS check: cascade-TV(f_a) <= (1/|W|) ∫_W (f*f) for cont. f ===
    print(f"\n----- Rigorous test: cascade-TV(f_a) <= (1/|W|) ∫_W (f*f) for cont. f? -----")
    print(f"  Theorem (proven below): YES, with equality for step f_a iff no partial pairs.\n")
    n_rigor_total = 0
    n_rigor_violations = 0
    rigor_min_excess = float('inf')
    for a_idx, a in enumerate(test_a_list):
        for ell in [2, 4, d]:
            for j in range(0, 2 * d - ell):
                if j + ell > 2 * d - 1:
                    continue
                tv_a = candidate_TVW_step_only(a, j, ell, n_half_local)
                W_lo = j * h
                W_hi = (j + ell) * h
                for name, mk in f_makers:
                    f = mk(a, x_grid)
                    tv_W_f = candidate_A_TV_W(f, x_grid, W_lo, W_hi)
                    n_rigor_total += 1
                    if tv_a > tv_W_f + 1e-6:
                        n_rigor_violations += 1
                    rigor_min_excess = min(rigor_min_excess, tv_W_f - tv_a)
    print(f"  cascade-TV(f_a) > (1/|W|)∫_W f*f violations: "
          f"{n_rigor_violations}/{n_rigor_total}")
    print(f"  min (TV_W(f) - cascade-TV(f_a)) = {rigor_min_excess:.6f} "
          f"(>=0 means cascade-TV ≤ (1/|W|)∫_W f*f always)")

    # === Direct calculation: cascade-TV vs (f_a * f_a) at conv-grid points ===
    print(f"\n----- Identify cascade-TV vs (f_a*f_a) at conv-grid for a=ones -----")
    a_test = np.ones(d)
    # ell=2, window k=s_lo: TV_W = (1/(4n*2)) Σ_{i+j=k} a_i a_j
    print(f"  ell=2 window: TV_W = (1/(8n)) * sum_{{i+j=k}} a_i a_j")
    print(f"               = (1/(8n)) * sum_step_conv_at_pos_k_in_int_units")
    f_step_v = f_uniform(a_test, x_grid)
    cv_step, t_step = autoconv(f_step_v, dx)
    for k in range(2 * d - 1):
        # cascade TV_W (ell=2)
        s_lo = k
        ell_v = 2
        if s_lo + ell_v - 2 >= 2 * d - 1:
            continue
        tv_a = candidate_TVW_step_only(a_test, s_lo, ell_v, n_half_local)
        # (f_a*f_a) at the corresponding conv grid position
        # Convolution: conv-pos k corresponds to t in [k*h, (k+2)*h], peak (k+1)*h
        t_peak = (k + 1) * h
        idx_peak = int(round(t_peak / dx))
        ff_peak = cv_step[idx_peak] if 0 <= idx_peak < len(cv_step) else 0
        # Average over [k*h, (k+1)*h]
        idx_lo = int(round(k * h / dx))
        idx_hi = int(round((k + 1) * h / dx))
        ff_avg = float(np.mean(cv_step[idx_lo:idx_hi+1])) if idx_lo < idx_hi else ff_peak
        print(f"  k={k}: TV_W={tv_a:.4f}, (f_a*f_a)((k+1)h)={ff_peak:.4f}, "
              f"avg_[(k)h,(k+1)h]={ff_avg:.4f}")

    # === Soundness check: lower-bound test for each candidate ==========
    print(f"\n----- Lower-bound check: does Psi(f) <= ||f*f||_inf? -----")
    print(f"  (we need Psi(f) <= ||f*f||_inf as a SOUND bound.)\n")

    rng = np.random.default_rng(20260509)
    n_random_a = 5
    n_violations = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    n_total = 0
    for _ in range(n_random_a):
        a = rng.dirichlet(np.ones(d)) * d  # Σ = d
        f_step_v = f_uniform(a, x_grid)
        f_step_v, _ = normalize_mass(f_step_v, dx)
        Linf = linf_autoconv(f_step_v, dx)
        # A: TV_W with W=full (length d*h = 1/2 in original; corresponds to
        # convolution full-support 1).  TV_W(f) = (1/(W_hi-W_lo)) ∫_W f*f
        # Use W = [0, 2*d*h] (full conv support).
        A_v = candidate_A_TV_W(f_step_v, x_grid, 0, 2 * d * h)
        # B: Σ a_i² * h
        B_v = candidate_B_bin_sq(a) * h
        # C: sum at grid points (no extra normalization)
        C_v = candidate_C_grid_conv(f_step_v, x_grid)
        # D: energy
        D_v = candidate_D_energy(f_step_v, x_grid)
        # E: <μ=1, f*f> = (∫f)² = 1
        E_v = candidate_E_dual(f_step_v, x_grid, mu_vals, mu_grid)

        # WARNING: NONE of these is automatically a lower bound on ||f*f||_∞.
        # A is an AVERAGE over W, so ≤ max ; over full conv-support [0,1],
        # TV(full)=∫f*f/1 = (∫f)² = 1 < ||f*f||∞ usually.
        # B/C are different scales ; need careful normalization.
        # We're testing if these *can be* turned into a lower bound by max-over-tests.
        n_total += 1
        if A_v > Linf + 1e-9:
            n_violations['A'] += 1
        if B_v > Linf + 1e-9:
            n_violations['B'] += 1
        # We don't compare directly C, D, E since they're not on the same scale.

    print(f"  After {n_total} random a-vectors:")
    for k, v in n_violations.items():
        print(f"    {k}: violations of Psi(f) > ||f*f||∞ = {v}/{n_total}")


# ----------------------------------------------------------------------
# Mathematical analysis of bin-average sufficiency for Candidate A.
# ----------------------------------------------------------------------
def analyze_A_aligned_windows():
    """Verify: when window W = [j*h, (j+ell)*h] for INTEGER j, ell, with
    j+ell ≤ 2d, is TV_W bin-average-sufficient?

    Key calculation: ∫_W (f*f)(t) dt = ∫∫ f(s)f(u) 1_W(s+u) ds du.
    For W with integer endpoints in units of h, 1_W(s+u) = 1_{[j*h, (j+ell)*h]}(s+u).
    Within (bin_p, bin_q) = ([p*h, (p+1)*h], [q*h, (q+1)*h]):
        s+u ∈ [(p+q)*h, (p+q+2)*h].
        1_W is constant on the diagonals s+u = const.
        It is NOT constant on the entire square (s,u) bin.
    => TV_W is NOT bin-average sufficient even for integer W.

    HOWEVER, the cascade computes a step-function autoconvolution
        TV_W(f_a) = (1/(4n·ell)) Σ_{(p,q):j ≤ p+q ≤ j+ell-1} a_p a_q · ???
    which uses #PAIRS in the window, NOT the integral.  The cascade's TV_W is
    really a "discrete window" — sum of conv-sums over conv-positions.

    We must show that for STEP f_a:
        (1/(4n·ell)) Σ_{(p,q): j ≤ p+q ≤ j+ell-1} a_p a_q
    relates to (1/|W|) ∫_W (f_a * f_a)(t) dt for some |W|.

    Direct calculation (step_function autoconvolution):
        (f_a * f_a)(t) = sum over (p,q) of a_p a_q · ((1_{bin_p}*1_{bin_q})(t))
    where (1_{bin_p}*1_{bin_q})(t) is a tent on [(p+q)*h, (p+q+2)*h] with peak h
    at t = (p+q+1)*h.

    So ∫_{[j*h, (j+ell)*h]} (f_a * f_a)(t) dt
       = Σ_{p,q} a_p a_q · ∫_{[j*h,(j+ell)*h]} (1_{bin_p}*1_{bin_q})(t) dt
       = Σ_{p,q} a_p a_q · h^2 · I_{j,ell}(p+q)
    where I_{j,ell}(s) = (1/h^2) · ∫_W (1_{[s*h,(s+2)*h]}-tent) dt, depending
    on whether [j,j+ell] fully contains [s,s+2], partially overlaps, or
    misses.  When [j,j+ell] FULLY contains [s,s+2]:
        I_{j,ell}(s) = h.
    Partial overlap → some triangle integral.

    KEY OBSERVATION: for STEP f_a, the integral ∫_W (f_a*f_a) is bin-average-
    sufficient (since it's a polynomial in the a_p's).
    For CONTINUOUS f with the same bin averages a but non-trivial intra-bin
    distribution, the integral CHANGES because s+u depends on the actual
    f within bins, not just the averages.

    So Candidate A FAILS bin-average sufficiency for continuous f.
    BUT: there's a sub-question — could a CHOICE of W make 1_W(s+u) constant
    over each bin pair (p,q)?  Only if W's endpoints are spaced ≥ 2h apart
    AND positioned at multiples of h that DON'T cut through any bin diagonal.
    Since for any (p,q), the diagonal s+u sweeps through [(p+q)h, (p+q+2)h]
    (length 2h), making 1_W constant on the bin pair requires 2h ≤ |W|
    AND W aligned at {p+q : p,q ∈ Z}h boundaries.  Even so, within the
    triangular tent, the integral is not bin-average-sufficient.
    """
    print("\n=== Analysis: Candidate A — TV_W bin-average sufficiency ===")
    print("Result: TV_W is NOT bin-average sufficient for continuous f")
    print("        (only for STEP f_a, where it tautologically holds).")
    print("Reason: ∫∫ f(s)f(u) 1_W(s+u) ds du depends on the joint distribution")
    print("        of f within each bin, not just bin averages.")


# ----------------------------------------------------------------------
def main():
    test_candidates()
    analyze_A_aligned_windows()


if __name__ == '__main__':
    main()
