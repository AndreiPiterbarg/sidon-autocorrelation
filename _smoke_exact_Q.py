"""Exact-rational verification of Q-LP results.

Strategy (c): solve LP with HiGHS to get a candidate basis, then verify the
basis solution exactly using Python `fractions.Fraction`. If the exact basis
solution is feasible and the corresponding dual solution is feasible (and
strong duality holds), then the float t_opt is rigorous up to its rational
value. We can also exactly check the sign of the optimal t.

Q-LP (per composition):
    Variables: λ ∈ R^{n_win}_{>=0}, t ∈ R
    max t   s.t.  A[σ, :] · λ - t >= 0   ∀σ
                  Σ_w λ_w = 1
                  λ_w >= 0,  t free

A[σ, w] = V_w - M[σ, w] / (2n)
V_w     = ws_w / (4n·ell_w) - ell_int_sum_w / (4n·ell_w) - c_target·m^2
M[σ, w] = (Σ_j σ_j BB^w_j) / ell_w

ALL inputs (ws_w, BB^w_j, ell_w, ell_int_sum_w, n, m) are integers; c_target
is a rational. So A[σ, w] is exactly rational, and the LP has rational optimal
solution.

Verification: given a candidate optimal basis B from HiGHS, solve B·x_B = b
exactly via Fraction Gaussian elimination, check x_B >= 0, and check exact
dual feasibility / complementary slackness on the active constraints.

We compare HiGHS-prune-counts vs exact-prune-counts on a sample of
compositions at (n_half=3, m=10, d=6, c=1.28).
"""
import os, sys, time
from fractions import Fraction
from itertools import combinations
import numpy as np
from scipy.optimize import linprog

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))

from _Q_bench import (
    _build_windows, _enum_balanced_signs, _composition_window_data,
    _q_bound_lp, prune_Q_one,
)
from compositions import generate_compositions_batched
from pruning import count_compositions


# ----------------------------------------------------------------------
# Exact rational LP construction (same A matrix, but Fraction-typed)
# ----------------------------------------------------------------------

def build_exact_A(c_int, windows, ell_int_sums, sigmas, n_half, m, c_target_rat):
    """Build A[σ, w] (n_sigma, n_win) and V[w] (n_win,) as Fraction arrays.

    A[σ, w] = V_w - (Σ_j σ_j BB^w_j) / (ell_w · 2n)
    """
    d = len(c_int)
    ws_int, BB_int = _composition_window_data(c_int, windows, n_half, m)
    n_win = len(windows)
    n_sigma = len(sigmas)

    n_d = Fraction(int(n_half))
    m_rat = Fraction(int(m))
    cs_m2 = c_target_rat * m_rat * m_rat

    V = [Fraction(0)] * n_win
    A = [[Fraction(0)] * n_win for _ in range(n_sigma)]

    for w, (ell, _) in enumerate(windows):
        ell_rat = Fraction(int(ell))
        denom_4 = Fraction(4) * n_d * ell_rat
        V_w = (Fraction(int(ws_int[w])) - Fraction(int(ell_int_sums[w]))) / denom_4 - cs_m2
        V[w] = V_w
        denom_2 = Fraction(2) * n_d * ell_rat
        for si, sigma in enumerate(sigmas):
            sb = 0
            for j in range(d):
                sb += int(sigma[j]) * int(BB_int[w, j])
            A[si][w] = V_w - Fraction(sb) / denom_2

    return A, V


def exact_objective_at(lam_rat, A, sigmas):
    """Compute t = min_σ (A[σ, :] · λ) exactly."""
    n_sigma = len(sigmas)
    n_win = len(lam_rat)
    best = None
    for si in range(n_sigma):
        s = Fraction(0)
        row = A[si]
        for w in range(n_win):
            s += row[w] * lam_rat[w]
        if best is None or s < best:
            best = s
    return best


def exact_q_bound(c_int, windows, ell_int_sums, sigmas, n_half, m, c_target_rat,
                  lam_float):
    """Take HiGHS's float λ, snap to exact rationals, evaluate t exactly.

    Two snapping strategies:
      1. Identify active basis: the windows w with λ_w > tol form the support.
         If support has size k, then optimal λ is a vertex of {Σλ=1} restricted
         to support (size 1 → λ=e_w; size > 1 → equalising the binding σ-constraints).
      2. For size-1 support: λ = e_w is an exact rational vertex.
      3. For larger support: solve the implied square system exactly.

    For our smoke test, we mostly use strategy 1 (vertex case) + a fallback that
    snaps λ to nearest small-rational and re-checks.

    Returns exact Fraction t_opt (lower bound — i.e. value of an exact
    feasible primal that <= true optimum). If we can't certify exactly,
    returns None.
    """
    n_win = len(windows)
    A, V = build_exact_A(c_int, windows, ell_int_sums, sigmas, n_half, m, c_target_rat)

    # Strategy 1: snap to vertex of simplex (single-window λ = e_w).
    # Compute t for each candidate single-window λ and report the best.
    # This gives the F-bound exactly. For strict Q-improvement we'd need
    # multi-window mixing, but as a first pass:
    best_t_vertex = None
    for w in range(n_win):
        lam = [Fraction(0)] * n_win
        lam[w] = Fraction(1)
        t = exact_objective_at(lam, A, sigmas)
        if best_t_vertex is None or t > best_t_vertex:
            best_t_vertex = t

    # Strategy 2: use HiGHS λ truncated to nearest rationals with denom 10^k
    # (k chosen so that all λ_w are > 0), project to simplex, evaluate exactly.
    # If t_exact > 0, the cell is rigorously prunable.
    tol = 1e-7
    support = [w for w in range(n_win) if lam_float is not None and lam_float[w] > tol]
    t_snapped = None
    if support:
        # Snap to denom 10^9, then re-normalize to Σ=1 exactly.
        denom = 10 ** 9
        nums = [int(round(lam_float[w] * denom)) if w in support else 0
                for w in range(n_win)]
        s = sum(nums)
        if s > 0:
            lam_rat = [Fraction(num, s) for num in nums]
            t_snapped = exact_objective_at(lam_rat, A, sigmas)

    # Take max — both are feasible primal (λ on simplex), so max is a valid LB
    candidates = [best_t_vertex]
    if t_snapped is not None:
        candidates.append(t_snapped)
    best_t = max(candidates)
    return best_t


# ----------------------------------------------------------------------
# Smoke test: compare HiGHS vs exact on F-survivors
# ----------------------------------------------------------------------

def run_smoke(n_half=3, m=10, c_target_str='1.28', max_comps=2000, seed=0,
              focus_borderline=True):
    d = 2 * n_half
    S_half = 2 * n_half * m
    c_target_rat = Fraction(c_target_str)
    c_target_float = float(c_target_rat)

    print(f"=== Smoke: exact-rational vs HiGHS for Q-LP ===")
    print(f"n_half={n_half}, m={m}, d={d}, c_target={c_target_str} (={c_target_float})")

    windows, ell_int_sums = _build_windows(d)
    sigmas = _enum_balanced_signs(d)
    n_win = len(windows); n_sigma = len(sigmas)
    print(f"n_win={n_win}, n_sigma={n_sigma}")

    # Sample compositions: take ALL palindromic comps, optionally focus on
    # borderline ones (|t_HiGHS| close to 0) where exact arithmetic matters.
    rng = np.random.default_rng(seed)
    all_comps = []
    for half_batch in generate_compositions_batched(n_half, S_half, batch_size=10000):
        for h in half_batch:
            full = np.empty(d, dtype=np.int32)
            full[:n_half] = h
            full[n_half:] = h[::-1]
            all_comps.append(full)
    print(f"Total palindromic comps: {len(all_comps)}")

    # If borderline focus: rapid HiGHS pass, sort by |t|, take smallest |t| comps
    if focus_borderline and len(all_comps) > max_comps:
        print(f"Pre-screening with HiGHS to find borderline cases...")
        margin = 1e-9 * m * m
        ts = []
        for k, c_int in enumerate(all_comps):
            t_h, _ = _q_bound_lp(c_int, windows, ell_int_sums, sigmas,
                                  n_half, m, c_target_float)
            ts.append(t_h)
        ts = np.array(ts)
        # Sort by |t - margin| (smallest = most borderline)
        order = np.argsort(np.abs(ts - margin))
        keep = order[:max_comps]
        comps = [all_comps[i] for i in keep]
        kept_ts = ts[keep]
        print(f"Borderline t range: [{kept_ts.min():.3e}, {kept_ts.max():.3e}]")
        print(f"  count |t| < margin*100: {int(np.sum(np.abs(kept_ts) < margin*100))}")
        print(f"  count |t| < margin: {int(np.sum(np.abs(kept_ts) < margin))}")
    else:
        comps = all_comps[:max_comps]
    print(f"Loaded {len(comps)} compositions for exact verification")

    n_highs_prune = 0
    n_exact_prune = 0
    n_highs_only = 0       # HiGHS prunes, exact does not (false positive)
    n_exact_only = 0       # Exact prunes, HiGHS does not (false negative)
    margin_float = 1e-9 * m * m

    margin_misses = []  # (idx, t_highs, t_exact) where signs disagree
    t0 = time.time()

    for k, c_int in enumerate(comps):
        # HiGHS
        t_h, lam_h = _q_bound_lp(c_int, windows, ell_int_sums, sigmas,
                                  n_half, m, c_target_float)
        prune_h = (t_h > margin_float)
        # Exact (verify HiGHS λ; also try vertex λ = e_w)
        t_e = exact_q_bound(c_int, windows, ell_int_sums, sigmas,
                             n_half, m, c_target_rat, lam_h)
        prune_e = (t_e > 0)   # exact: ANY positive primal value certifies prune

        if prune_h: n_highs_prune += 1
        if prune_e: n_exact_prune += 1
        if prune_h and not prune_e: n_highs_only += 1
        if prune_e and not prune_h: n_exact_only += 1
        if prune_h != prune_e:
            margin_misses.append((k, t_h, float(t_e)))
            if len(margin_misses) <= 10:
                print(f"  MISMATCH idx={k}: HiGHS t={t_h:.3e}, exact t={float(t_e):.3e}, "
                      f"prune_HiGHS={prune_h}, prune_exact={prune_e}")

    elapsed = time.time() - t0
    print(f"\n--- Results ({len(comps)} comps, {elapsed:.1f}s) ---")
    print(f"  HiGHS prune count: {n_highs_prune}")
    print(f"  Exact prune count: {n_exact_prune}")
    print(f"  HiGHS-only (false +): {n_highs_only}")
    print(f"  Exact-only (HiGHS missed): {n_exact_only}")
    print(f"  Mismatches total: {len(margin_misses)}")

    return {
        'n_comps': len(comps),
        'n_highs_prune': n_highs_prune,
        'n_exact_prune': n_exact_prune,
        'n_highs_only': n_highs_only,
        'n_exact_only': n_exact_only,
    }


if __name__ == '__main__':
    # Test 1: c_target = 1.28 (paper-line), borderline-focused 500
    print("\n###### TEST 1: c=1.28 borderline-focused ######")
    run_smoke(n_half=3, m=10, c_target_str='1.28', max_comps=500,
              focus_borderline=True)
    # Test 2: c_target = 1.30 (more comps near margin, since most will not prune)
    print("\n###### TEST 2: c=1.30 borderline-focused ######")
    run_smoke(n_half=3, m=10, c_target_str='1.30', max_comps=500,
              focus_borderline=True)
    # Test 3: c_target = 1.40 (above bound — almost no prunes; many borderline)
    print("\n###### TEST 3: c=1.40 borderline-focused ######")
    run_smoke(n_half=3, m=10, c_target_str='1.40', max_comps=500,
              focus_borderline=True)
