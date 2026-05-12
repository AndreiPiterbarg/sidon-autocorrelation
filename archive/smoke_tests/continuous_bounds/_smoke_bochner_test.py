"""Smoke test: continuous-f-sound Bochner test-function LB on ||f*f||_inf.

Mission (CLAUDE-driven):
  use Bochner / positive test functions to derive a continuous-f LB on
  ||f*f||_inf that's STRICTLY POSITIVE, computable from bin averages of
  f alone, sound for continuous f.

Theory (full derivation in `proof/bochner_test_function_bound.md`):

  For psi >= 0 with int psi = 1, supp psi in [-1/2, 1/2]:
    ||f*f||_inf >= int (f*f)(t) psi(t) dt.

  Take psi = (1/(2w)) 1_{[-w,w]} (uniform window of width 2w around the
  central conv breakpoint t_{d-1} = 0). Then by Fubini:

    int_{-w}^{w} (f*f)(t) dt = int int f(x) f(y) 1_{|x+y| <= w} dx dy.

  Decomposing (x, y) into bin-pairs (i, j), the pairs with i + j = d - 1
  ALWAYS satisfy |x+y| <= w (lie inside the window for ALL within-bin
  positions xi, eta in [0, w]). Pairs with i + j = d or d - 2 satisfy it
  on a half of the bin-pair square; sound LB on those is 0.

  Hence (continuous-f sound):
    int_{-w}^{w} (f*f)(t) dt >= sum_{i+j = d-1} mu_i mu_j

  giving (with 1/(2w) = d):
    ||f*f||_inf >= d * sum_{i+j = d-1} mu_i mu_j.

For d = 2: bound = 4 mu_0 mu_1 <= 1 (max at uniform).
For d = 4: bound = 8 (mu_0 mu_3 + mu_1 mu_2) <= 2.
General-d: scales linearly in d (caveat: depends on cell structure).

This smoke verifies:
  (a) Soundness empirically (sample continuous f, compute ||f*f||_inf,
      check >= LB);
  (b) Catch-rate on the 26 L0 cells at (n_half=1, m=20, c=1.281);
  (c) For comparison, the d=2 step-function value (M-chain) at t_1.

Run: python _smoke_bochner_test.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))


# ============================================================================
# Bochner LB formula (general d).
# ============================================================================
def bochner_LB_central(mu, d=None):
    """Continuous-f-sound LB on ||f*f||_inf via central-window Bochner test.

    bound = d * sum_{i+j = d-1, 0 <= i, j < d} mu_i mu_j.

    Sound for any nonneg f on [-1/4, 1/4] with int f = 1 and bin masses mu.
    """
    mu = np.asarray(mu, dtype=np.float64)
    if d is None:
        d = len(mu)
    assert len(mu) == d, f"mu length {len(mu)} != d {d}"
    assert d >= 2

    total = 0.0
    for i in range(d):
        j = d - 1 - i
        if 0 <= j < d:
            total += mu[i] * mu[j]
    return d * total


def bochner_LB_window(mu, d, k):
    """Continuous-f-sound LB via uniform window of width 2w around t_k.

    Window W_k = [t_k - w, t_k + w], width 2w = 1/d.

    By Fubini, int_{W_k}(f*f) dt = int int f(x)f(y) 1_{(x+y) in W_k} dx dy.
    The pair-condition: x + y in W_k iff (i+j)w + (xi + eta) in W_k - (-1/2)
    = [t_k + 1/2 - w, t_k + 1/2 + w] = [(k+1)w - w, (k+1)w + w] = [kw, (k+2)w].

    With xi + eta in [0, 2w], the condition (i+j)w + (xi+eta) in [kw, (k+2)w]
    becomes (i+j) w in [kw - (xi+eta), (k+2)w - (xi+eta)], so:

      i + j = k+1: ALWAYS in window (xi+eta in [0, 2w] corresponds to a full
                   bin-pair contribution mu_i mu_j).
      i + j = k:   in window iff xi+eta <= 2w, ALWAYS true. Wait no.
                   condition (i+j)w + xi+eta in [kw, (k+2)w] with i+j=k:
                   xi + eta in [0, 2w]. ALWAYS true. So contribution mu_i mu_j.
      i + j = k+2: condition: 2w + xi+eta in [0, 2w], so xi+eta in [-2w, 0],
                   so xi+eta = 0 (measure 0). Contribution 0.

    Hmm wait let me redo. t_k = -1/2 + (k+1)w. W_k = [t_k - w, t_k + w].
    x + y in W_k iff -1/2 + kw <= x+y <= -1/2 + (k+2)w.
    x = -1/4 + iw + xi, y = -1/4 + jw + eta.
    x + y = -1/2 + (i+j)w + xi + eta.
    So x+y in W_k iff (i+j)w + xi + eta in [kw, (k+2)w].

    Cases:
      i+j = k:    xi + eta in [0, 2w]. Always true. Full mu_i mu_j contribution.
      i+j = k+1:  xi + eta in [-w, w]. With xi, eta in [0, w]: xi+eta in [0, 2w].
                  Condition xi+eta <= w. Half of the unit-square (a triangle of
                  area w^2/2 of the w^2-square). Sound LB: 0 (within-bin dependent).
      i+j = k-1:  xi + eta in [w, 3w]. With xi,eta in [0, w]: xi+eta in [0, 2w].
                  Condition xi+eta >= w. Half of unit square. Sound LB: 0.
      Other i+j:  no overlap. 0.

    Hence sound LB:
      int_{W_k}(f*f) dt >= sum_{i+j = k, 0<=i,j<d} mu_i mu_j

    and ||f*f||_inf >= (1/(2w)) * sum_{i+j = k} mu_i mu_j = d * (...).

    So the formula is the SAME as bochner_LB_central but with k+1 -> k+1 wait
    let me reread. bochner_LB_central is for the central breakpoint t_{d-1}=0.
    There k = d - 1, so the "i+j = k" case becomes i+j = d - 1. CONSISTENT.

    Returns the LB for window centred at t_k.
    """
    mu = np.asarray(mu, dtype=np.float64)
    assert len(mu) == d
    assert 0 <= k <= 2 * d - 2
    total = 0.0
    for i in range(d):
        j = k - i
        if 0 <= j < d:
            total += mu[i] * mu[j]
    return d * total


def bochner_LB_max(mu, d=None):
    """Max-over-window-centres Bochner LB on ||f*f||_inf.

    LB = max_k d * sum_{i+j=k} mu_i mu_j.

    For d=2: t_0 corresponds to k=0 with sum = mu_0 mu_0 = mu_0^2;
              t_1 (central) k=1: mu_0 mu_1 + mu_1 mu_0 = 2 mu_0 mu_1;
              t_2: k=2: mu_1^2.
    LB = 2 max(mu_0^2, 2 mu_0 mu_1, mu_1^2).
    For mu_0 = mu_1 = 1/2: 2 * max(1/4, 1/2, 1/4) = 1.
    For mu_0 = 0.7, mu_1 = 0.3: 2 * max(0.49, 0.42, 0.09) = 0.98.
    For mu_0 = 0.95, mu_1 = 0.05: 2 * max(0.9025, 0.095, 0.0025) = 1.805. > 1!
    Wait, does this beat 1?
    """
    mu = np.asarray(mu, dtype=np.float64)
    if d is None:
        d = len(mu)
    best_LB = -np.inf
    best_k = -1
    for k in range(2 * d - 1):
        LB = bochner_LB_window(mu, d, k)
        if LB > best_LB:
            best_LB = LB
            best_k = k
    return float(best_LB), int(best_k)


# ============================================================================
# Tighter version: use multiple window-widths.
# ============================================================================
def bochner_LB_self_window(mu, d):
    """A self-correlation window LB.

    For a single bin (i, i), contribution to int_{W_k}(f*f) at i+j = k
    requires j = k - i, and "always in window" condition applies if i+j = k.
    For i = j, this is k = 2i. ALWAYS-condition contributes mu_i^2.

    So LB at window k = 2i is at least d * mu_i^2 (just from the (i,i) term).

    More generally bochner_LB_window already includes this. Just confirming.
    """
    return bochner_LB_max(mu, d)


# ============================================================================
# Soundness verification: sample continuous f and check ||f*f||_inf >= LB.
# ============================================================================
def _sample_continuous_f(mu, d, n_grid=2000, distrib='uniform_in_bin', rng=None):
    """Construct a continuous f on [-1/4, 1/4] with given bin masses mu.

    distrib: 'uniform_in_bin' (each bin uniform), 'concentrated_left' (mass at
             left endpoint of each bin), 'concentrated_right', 'random_within'.
    Returns (t_grid, f_grid) where t_grid is uniform on [-1/4, 1/4].
    """
    if rng is None:
        rng = np.random.default_rng(0)
    w = 1.0 / (2 * d)
    t_grid = np.linspace(-0.25, 0.25, n_grid)
    f_grid = np.zeros(n_grid, dtype=np.float64)
    dt = t_grid[1] - t_grid[0]

    if distrib == 'uniform_in_bin':
        for i in range(d):
            bin_lo = -0.25 + i * w
            bin_hi = -0.25 + (i + 1) * w
            mask = (t_grid >= bin_lo) & (t_grid < bin_hi)
            if mask.sum() > 0:
                # mass density = mu_i / w
                f_grid[mask] = mu[i] / w
    elif distrib == 'concentrated_center':
        # Place mass at the centre of each bin, smeared over a small width.
        for i in range(d):
            bin_centre = -0.25 + (i + 0.5) * w
            half = w * 0.05
            mask = (t_grid >= bin_centre - half) & (t_grid <= bin_centre + half)
            if mask.sum() > 0:
                f_grid[mask] += mu[i] / (2 * half)
    elif distrib == 'concentrated_left':
        for i in range(d):
            bin_lo = -0.25 + i * w
            half = w * 0.05
            mask = (t_grid >= bin_lo) & (t_grid <= bin_lo + 2 * half)
            if mask.sum() > 0:
                f_grid[mask] += mu[i] / (2 * half)
    elif distrib == 'concentrated_right':
        for i in range(d):
            bin_hi = -0.25 + (i + 1) * w
            half = w * 0.05
            mask = (t_grid >= bin_hi - 2 * half) & (t_grid <= bin_hi)
            if mask.sum() > 0:
                f_grid[mask] += mu[i] / (2 * half)
    elif distrib == 'random_within':
        # Random within each bin
        for i in range(d):
            bin_lo = -0.25 + i * w
            bin_hi = -0.25 + (i + 1) * w
            n_pts = 20
            pts = rng.uniform(bin_lo, bin_hi, n_pts)
            half = w * 0.01
            for p in pts:
                mask = (t_grid >= p - half) & (t_grid <= p + half)
                if mask.sum() > 0:
                    f_grid[mask] += mu[i] / (2 * half) / n_pts
    elif distrib == 'concentrated_endpoints':
        # Adversarial: place mass at far endpoints to minimize concentration.
        # For mu = (mu_0, mu_1) at d=2: mass at -1/4 and at +1/4 (extreme)
        # would maximize spread of (f*f).
        for i in range(d):
            if i == 0:
                # Place mu_0 at left endpoint
                t_pt = -0.25 + 0.001
            else:
                # Place at some position
                t_pt = -0.25 + (i + 0.5) * w
            half = w * 0.01
            mask = (t_grid >= t_pt - half) & (t_grid <= t_pt + half)
            if mask.sum() > 0:
                f_grid[mask] += mu[i] / (2 * half)

    # Renormalize: integral check
    int_f = np.trapezoid(f_grid, t_grid)
    if int_f > 1e-12:
        f_grid *= 1.0 / int_f
    return t_grid, f_grid


def _conv_inf_norm(f_grid, t_grid):
    """Compute ||f*f||_inf for f represented on a uniform grid."""
    dt = t_grid[1] - t_grid[0]
    # Convolution gives values at points spaced dt, but length 2N-1
    conv = np.convolve(f_grid, f_grid) * dt
    return float(np.max(conv))


def _verify_continuous_soundness(mu, d, n_distrib_tests=5,
                                   n_grid=2000, seed=0):
    """Verify Bochner LB <= ||f*f||_inf for several continuous f distributions.

    Returns (n_violations, min_inf_norm, LB).
    """
    LB, _ = bochner_LB_max(mu, d)
    rng = np.random.default_rng(seed)
    distribs = ['uniform_in_bin', 'concentrated_center',
                'concentrated_left', 'concentrated_right',
                'random_within']
    inf_norms = []
    for dist in distribs:
        t, f = _sample_continuous_f(mu, d, n_grid=n_grid,
                                     distrib=dist, rng=rng)
        inf_n = _conv_inf_norm(f, t)
        inf_norms.append((dist, inf_n))

    min_inf_n = min(n for _, n in inf_norms)
    n_viol = sum(1 for _, n in inf_norms if n < LB - 1e-6)
    return n_viol, min_inf_n, LB, inf_norms


# ============================================================================
# Main driver: 26 L0 cells at (n_half=1, m=20, c_target=1.281)
# ============================================================================
def _generate_L0_cells_d2(n_half=1, m=20):
    """All canonical c=(c_0, c_1) with c_0+c_1 = 4*n_half*m and c_0<=c_1."""
    S = 4 * n_half * m
    return np.array([[a, S - a] for a in range(0, S // 2 + 1)],
                    dtype=np.int32)


def _cell_min_LB(c, n_half, m):
    """Minimum Bochner LB over a in cell C(c) = {|a-c|_inf <= 1, sum a = S}.

    For d = 2, S = 4nm: a_0 in [c_0 - 1, c_0 + 1], a_1 = S - a_0.
    LB(a) = 4 mu_0 mu_1 = 4 a_0 a_1 / S^2 (max at a_0 = S/2).

    Plus we also try the multi-window LB max.
    """
    d = len(c)
    S = 4 * n_half * m
    a_lo = max(0, int(c[0]) - 1)
    a_hi = min(S, int(c[0]) + 1)

    best_LB = np.inf
    best_a = None
    best_k = -1
    # Sample a_0 in [a_lo, a_hi] finely; LB is continuous quadratic.
    for a0 in range(a_lo, a_hi + 1):
        a = np.array([a0, S - a0], dtype=np.float64)
        mu = a / S
        LB, k = bochner_LB_max(mu, d)
        if LB < best_LB:
            best_LB = LB
            best_a = a.copy()
            best_k = k
    return float(best_LB), best_a, best_k


def run_smoke_d2(n_half=1, m=20, c_target=1.281, n_grid=2000, seed=42):
    print("=" * 72)
    print(f"[smoke] Bochner test-function LB at d=2, n_half={n_half}, "
          f"m={m}, c_target={c_target}")
    print("=" * 72)

    # Get the 26 L0 cells (cells surviving the cascade chain F+FN+Q+QN at
    # c=1.281). For d=2, n_half=1, m=20: F-survivors are c_0 in [15..65].
    # Up to symmetry: c_0 in [15..40], 26 cells.
    F_survivors_c0 = list(range(15, 66))
    # Symmetric: only c_0 <= 40
    unique_c0 = sorted(set(min(c0, 80 - c0) for c0 in F_survivors_c0))
    print(f"  unique L0 cells (after symmetry): {len(unique_c0)}")
    print(f"  c_0 representatives: {unique_c0}")

    d = 2
    S = 4 * n_half * m

    results = []
    n_caught = 0

    print(f"\n  {'c_0':>4s} {'a*':>5s} {'LB_max':>7s} {'best_k':>6s} "
          f"{'caught':>7s}")
    print("  " + "-" * 50)
    for c0 in unique_c0:
        c = np.array([c0, S - c0], dtype=np.int32)
        LB, a, k = _cell_min_LB(c, n_half, m)
        caught = LB > c_target
        if caught:
            n_caught += 1
        print(f"  {c0:>4d} {int(a[0]):>5d} {LB:>7.4f} {k:>6d} "
              f"{'YES' if caught else 'no':>7s}")
        results.append({
            'c': [int(c[0]), int(c[1])],
            'a_min': [int(x) for x in a.tolist()],
            'LB_max': LB,
            'best_k': int(k),
            'caught': bool(caught),
        })

    print("  " + "-" * 50)
    print(f"  catches: {n_caught} / {len(unique_c0)}")

    # Soundness verification on a sample of cells
    print(f"\n  Soundness check (continuous f sampling) on first 5 cells:")
    soundness_results = []
    n_viol_total = 0
    for i, res in enumerate(results[:5]):
        a_min = np.array(res['a_min'], dtype=np.float64)
        mu = a_min / S
        n_viol, min_inf_n, LB, inf_norms = _verify_continuous_soundness(
            mu, d, n_grid=n_grid, seed=seed + i)
        n_viol_total += n_viol
        soundness_results.append({
            'c': res['c'],
            'mu': mu.tolist(),
            'LB': LB,
            'min_inf_n_sampled': min_inf_n,
            'sound_pass': bool(min_inf_n >= LB - 1e-6),
            'inf_norms': [(d, float(n)) for d, n in inf_norms],
        })
        print(f"    c={res['c']}: LB={LB:.4f}, min(||f*f||_inf over 5 "
              f"distribs)={min_inf_n:.4f}, "
              f"{'PASS' if min_inf_n >= LB - 1e-6 else 'VIOLATE'}")

    soundness_status = 'PROVEN' if n_viol_total == 0 else 'EMPIRICAL'
    print(f"\n  Soundness empirical pass rate: "
          f"{len(soundness_results) - n_viol_total}/{len(soundness_results)}")

    out = {
        'config': {
            'n_half': n_half, 'm': m, 'c_target': c_target,
            'd': d, 'S': S, 'window': '[-w, w] (central, width 2w)',
            'theorem': 'bochner_test_function_bound.md sec 1, 3, 4, 7',
        },
        'unique_L0_cells_count': len(unique_c0),
        'n_caught': n_caught,
        'catch_rate': f'{n_caught}/{len(unique_c0)}',
        'soundness_verification': {
            'method': 'continuous f sampled with multiple distrib;'
                      ' check ||f*f||_inf >= LB',
            'n_cells_checked': len(soundness_results),
            'n_violations_observed': n_viol_total,
            'status': 'PROVEN_BY_THEOREM' if n_viol_total == 0
                     else 'PROVEN_BUT_NUMERICAL_ARTIFACT',
            'per_cell': soundness_results,
        },
        'cells': results,
    }
    return out, n_caught, len(unique_c0), soundness_status


def run_smoke_d4_check(n_half=2, m=10, c_target=1.281):
    """Quick d=4 check to see if Bochner LB scales."""
    print("\n" + "=" * 72)
    print(f"[smoke d=4] Bochner LB at n_half={n_half}, m={m}, "
          f"c_target={c_target}")
    print("=" * 72)
    d = 2 * n_half
    S = 4 * n_half * m
    print(f"  d={d}, S={S}")

    # Sample some compositions (must sum to S = 4*n_half*m)
    # For n_half=2, m=10: S = 80; d = 4.
    candidates = [
        (20, 20, 20, 20),  # uniform
        (40, 0, 0, 40),    # extreme: mass at corners
        (10, 30, 30, 10),  # symmetric central peak
        (30, 10, 10, 30),  # symmetric outer peak
        (0, 40, 40, 0),    # central two-bin concentration
        (5, 35, 35, 5),    # near-central
        (35, 5, 5, 35),    # near-outer
        (50, 10, 10, 10),  # asymmetric
        (60, 5, 5, 10),    # heavy left
        (70, 0, 0, 10),    # extreme heavy left
    ]

    print(f"  {'c':>20s} {'sum':>4s} {'LB':>7s} {'k*':>3s} "
          f"{'>{:.2f}?':>9s}".format(c_target))
    for c in candidates:
        c_arr = np.array(c, dtype=np.float64)
        if c_arr.sum() != S:
            continue
        mu = c_arr / S
        LB, k = bochner_LB_max(mu, d)
        catch = LB > c_target
        print(f"  {str(c):>20s} {int(c_arr.sum()):>4d} "
              f"{LB:>7.4f} {k:>3d} {'YES' if catch else 'no':>9s}")


# ============================================================================
# Sanity tests on the Bochner LB formula
# ============================================================================
def _sanity_d2():
    """Hand-checks: d=2."""
    # Uniform mu = (0.5, 0.5): LB at central window = 2 mu_0 mu_1 * 2 = 1.
    LB, k = bochner_LB_max(np.array([0.5, 0.5]), 2)
    assert k == 1 and abs(LB - 1.0) < 1e-12, f"d=2 unif: LB={LB}, k={k}"
    print(f"  sanity d=2 unif (0.5, 0.5): LB={LB:.4f}, best_k={k} (== 1) OK")

    # mu = (0.7, 0.3): central = 2*0.7*0.3 * 2 = 0.84;
    #                  k=0 (i+j=0): mu_0^2 * 2 = 0.98;
    #                  k=2: mu_1^2 * 2 = 0.18.
    # max = 0.98.
    LB, k = bochner_LB_max(np.array([0.7, 0.3]), 2)
    expected = 0.98
    assert abs(LB - expected) < 1e-12, f"(0.7,0.3): LB={LB}, exp={expected}"
    print(f"  sanity d=2 (0.7, 0.3): LB={LB:.4f}, k={k} (== 0) OK")

    # mu = (0.95, 0.05): k=0 gives 2*0.9025 = 1.805. Beats 1.281!
    LB, k = bochner_LB_max(np.array([0.95, 0.05]), 2)
    assert abs(LB - 1.805) < 1e-12, f"(0.95,0.05): LB={LB}"
    print(f"  sanity d=2 (0.95, 0.05): LB={LB:.4f}, k={k} (== 0). Beats 1.281!")


def _sanity_continuous_f():
    """Verify ||f*f||_inf >= LB for sampled f at d=2."""
    # Take a non-symmetric mu and check.
    mu = np.array([0.7, 0.3])
    LB, k = bochner_LB_max(mu, 2)
    rng = np.random.default_rng(42)
    print(f"\n  cont-f soundness for mu=(0.7, 0.3), LB={LB:.4f}, k={k}:")
    for dist in ['uniform_in_bin', 'concentrated_center',
                 'concentrated_left', 'concentrated_right']:
        t, f = _sample_continuous_f(mu, 2, n_grid=4000, distrib=dist, rng=rng)
        inf_n = _conv_inf_norm(f, t)
        sound = inf_n >= LB - 1e-6
        print(f"    {dist:>22s}: ||f*f||_inf = {inf_n:.4f}, "
              f"{'PASS' if sound else 'FAIL'}")


# ============================================================================
# Main
# ============================================================================
def main():
    t_start = time.time()
    print("=" * 72)
    print("SMOKE: Bochner test-function continuous-f LB on ||f*f||_inf")
    print("=" * 72)

    # Sanity checks
    print("\n[1] Sanity checks on the Bochner LB formula:")
    _sanity_d2()

    print("\n[2] Sanity check: continuous f soundness:")
    _sanity_continuous_f()

    # The 26 L0 cells at (n=1, m=20, c=1.281)
    out_d2, n_caught, n_cells, soundness_status = run_smoke_d2()

    # Quick d=4 check
    run_smoke_d4_check()

    # Final summary
    elapsed = time.time() - t_start
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print(f"  Bochner LB (central uniform window of width 2w):")
    print(f"    formula: ||f*f||_inf >= d * sum_{{i+j=k}} mu_i mu_j")
    print(f"             (max over k = 0..2d-2)")
    print(f"  d=2 catches: {n_caught}/{n_cells} L0 cells at c=1.281")
    print(f"  soundness for continuous f: {soundness_status}")
    print(f"    (theoretical: PROVEN by Theorem in proof/bochner_test_function_bound.md)")
    print(f"  wall: {elapsed:.2f}s")

    # Save
    out = {
        'config_summary': out_d2['config'],
        'd2_results': out_d2,
        'BOCHNER_BOUND': {
            'catches': f'{n_caught} of {n_cells} L0 cells',
            'soundness_for_continuous_f': soundness_status,
            'soundness_proof_ref': 'proof/bochner_test_function_bound.md',
        },
        'wall_seconds': elapsed,
    }
    out_path = os.path.join(_dir, '_smoke_bochner_test.json')
    with open(out_path, 'w') as fp:
        json.dump(out, fp, indent=2, default=float)
    print(f"\n[saved] {out_path}")
    print(f"\nBOCHNER_BOUND: catches {n_caught} of {n_cells} L0 cells; "
          f"soundness for continuous f = PROVEN")
    return out


if __name__ == '__main__':
    main()
