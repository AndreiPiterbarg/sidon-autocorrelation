#!/usr/bin/env python
r"""Minimax box certification via Sion's theorem.

==========================================================================
MATHEMATICAL FOUNDATION
==========================================================================

Problem: The cascade proves TV_W(mu*) >= c at every grid point mu*.
We need to extend this to ALL continuous mu in the Voronoi cell around mu*.

The continuous mu lives on the simplex {mu >= 0, sum mu_i = 1}.
The Voronoi cell around grid point k/S is:
    C = {mu : |mu_i - k_i/S| <= 1/(2S) for all i, sum mu_i = 1, mu_i >= 0}
     = box ∩ hyperplane(sum=1) ∩ nonneg

Critical constraint: sum delta_i = 0.  This is NOT optional — the simplex
constrains perturbations to a (d-1)-dimensional hyperplane.

Current method (broken at d>=16):
    For a single killing window W*, bound:
        min_{mu in C} TV_{W*}(mu) >= TV_{W*}(mu*) - cell_var - quad_corr
    The cell_var uses the sorted-pairing trick (exact for sum=0 + box constraint).
    The quad_corr uses |delta_i| <= eps per coordinate (ignoring sum=0).
    Total overestimation: cell_var >> margin, so certification fails.

New method (minimax duality via Sion's theorem):
    We need:  min_{mu in C} max_W TV_W(mu) >= c_target

    Sion's minimax theorem applies because:
      - C is compact and convex
      - The set of windows W is finite
      - TV_W(mu) is continuous in mu for each W

    Therefore:
      min_{mu in C} max_W TV_W(mu)  =  max_{lambda} min_{mu in C} sum_W lambda_W TV_W(mu)

    where lambda ranges over probability distributions on windows.

==========================================================================
FEASIBLE REGION: Box ∩ Simplex
==========================================================================

The cell C is the (d-1)-dimensional polytope:
    mu_i in [lo_i, hi_i],  sum mu_i = 1,  mu_i >= 0
where lo_i = max(0, k_i/S - 1/(2S)), hi_i = k_i/S + 1/(2S).

Vertices of C: each vertex has (d-1) coordinates at their box bounds,
with one "free" coordinate determined by sum=1.  Specifically:
    For each coordinate r in {0,...,d-1} and each assignment of the
    other d-1 coordinates to {lo_i, hi_i}:
        mu_r = 1 - sum_{i != r} mu_i
        If lo_r <= mu_r <= hi_r: this is a vertex.

Total candidate vertices: d * 2^{d-1}.  At d=16: 524,288.

==========================================================================
APPROACH
==========================================================================

For each pruned grid point mu*:
  1. Collect ALL killing windows {W : TV_W(mu*) >= c_target}.
  2. Enumerate vertices of C (or sample interior points).
  3. For each vertex, find the best killing window.
  4. Solve the LP dual: max_lambda min_{v in vertices} sum_W lambda_W TV_W(v).
     This gives a LOWER BOUND on the minimax at vertices.
  5. Since TV_W is not convex, the interior might be worse than vertices.
     To handle this: also sample random interior points of C and add them
     as constraints in the LP.

If the LP value >= c_target: all tested points are certified.
If also validated by dense interior sampling: high confidence.
Full rigor requires interval arithmetic (separate step).
"""

import sys
import os
import math
import time
import itertools

import numpy as np

# Setup imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_this_dir)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, os.path.join(_root, 'cloninger-steinerberger'))

from compositions import generate_canonical_compositions_batched
from pruning import count_compositions, asymmetry_threshold


# =====================================================================
# Core: compute TV_W for a continuous mass vector (float)
# =====================================================================

def tv_window(mu, d, ell, s_lo):
    """Compute TV_W(mu) for window (ell, s_lo).

    TV_W = (2d/ell) * sum_{s_lo <= i+j <= s_lo+ell-2} mu_i * mu_j
    """
    s_hi = s_lo + ell - 2
    total = 0.0
    for i in range(d):
        for j in range(d):
            if s_lo <= i + j <= s_hi:
                total += mu[i] * mu[j]
    return total * 2.0 * d / ell


def tv_max_over_windows(mu, d):
    """Compute max_W TV_W(mu) over all valid windows.

    Returns (best_tv, best_ell, best_s_lo).
    """
    conv_len = 2 * d - 1
    best_tv = 0.0
    best_ell = 0
    best_s = 0
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        for s_lo in range(conv_len - n_cv + 1):
            tv = tv_window(mu, d, ell, s_lo)
            if tv > best_tv:
                best_tv = tv
                best_ell = ell
                best_s = s_lo
    return best_tv, best_ell, best_s


def all_windows(d):
    """Return list of all valid (ell, s_lo) pairs."""
    conv_len = 2 * d - 1
    windows = []
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        for s_lo in range(conv_len - n_cv + 1):
            windows.append((ell, s_lo))
    return windows


# =====================================================================
# Enumerate vertices of the feasible cell (box ∩ simplex)
# =====================================================================

def cell_vertices(mu_lo, mu_hi, d, tol=1e-12):
    """Enumerate vertices of {mu : lo <= mu <= hi, sum mu = 1, mu >= 0}.

    Each vertex has d-1 coordinates at a box bound, and 1 coordinate
    determined by sum=1.

    Returns list of (d,) float arrays.
    """
    vertices = []

    # For each "free" coordinate r:
    for r in range(d):
        other = [i for i in range(d) if i != r]
        # For each assignment of the other d-1 coords to lo or hi:
        for bits in itertools.product([0, 1], repeat=d - 1):
            v = np.empty(d, dtype=np.float64)
            s = 0.0
            for idx, i in enumerate(other):
                v[i] = mu_hi[i] if bits[idx] else mu_lo[i]
                s += v[i]
            v[r] = 1.0 - s
            # Check feasibility
            if v[r] >= mu_lo[r] - tol and v[r] <= mu_hi[r] + tol:
                v[r] = np.clip(v[r], mu_lo[r], mu_hi[r])
                if v[r] >= -tol:
                    v[r] = max(v[r], 0.0)
                    vertices.append(v)

    # Deduplicate (vertices can appear multiple times)
    if not vertices:
        return []
    arr = np.array(vertices)
    # Round to avoid floating point duplicates
    rounded = np.round(arr, decimals=14)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    return [vertices[i] for i in sorted(unique_idx)]


def sample_cell_interior(mu_star_float, eps, d, n_samples, rng=None):
    """Sample random feasible points in box ∩ simplex.

    Uses rejection sampling: generate delta with sum=0, check box constraints.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    mu_lo = np.maximum(mu_star_float - eps, 0.0)
    mu_hi = mu_star_float + eps

    samples = []
    attempts = 0
    max_attempts = n_samples * 100

    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1
        # Generate random delta with sum=0
        delta = rng.uniform(-eps, eps, size=d)
        delta -= delta.mean()  # project to sum=0

        # Scale to stay within box
        mu = mu_star_float + delta
        # Check bounds
        if np.all(mu >= mu_lo - 1e-15) and np.all(mu <= mu_hi + 1e-15):
            mu = np.clip(mu, mu_lo, mu_hi)
            mu = np.maximum(mu, 0.0)
            # Renormalize to sum=1 (tiny adjustment)
            mu /= mu.sum()
            samples.append(mu)

    return samples


# =====================================================================
# Old method: single-window, triangle-inequality bound
# =====================================================================

def old_box_cert(mu_star, d, S, c_target):
    """Current box cert: margin - cell_var - quad_corr.

    Uses the exact same formulas as the production Numba kernel.

    Returns (best_net, best_ell, best_s_lo, margin, cell_var, quad_corr).
    """
    mu = mu_star.astype(np.float64) / S
    eps = 1.0 / (2.0 * S)

    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.float64)
    for i in range(d):
        for j in range(d):
            conv[i + j] += mu[i] * mu[j]

    best_net = -1e30
    best_info = None

    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        scale_tv = 2.0 * d / ell
        scale_g = 4.0 * d / ell

        for s_lo in range(conv_len - n_cv + 1):
            s_hi = s_lo + ell - 2
            ws = sum(conv[s_lo:s_lo + n_cv])
            tv = ws * scale_tv
            margin = tv - c_target

            if margin <= 0:
                continue

            # Gradient (matches production code)
            grad = np.zeros(d, dtype=np.float64)
            for i in range(d):
                g = 0.0
                for j in range(d):
                    if s_lo <= i + j <= s_hi:
                        g += mu[j]
                grad[i] = g * scale_g

            # Sorted-pairing cell_var (exact for sum=0 + box)
            grad_sorted = np.sort(grad)
            cell_var = 0.0
            for k in range(d // 2):
                cell_var += grad_sorted[d - 1 - k] - grad_sorted[k]
            cell_var /= (2.0 * S)

            # Quadratic correction
            n_pairs = 0
            for k in range(s_lo, s_lo + ell - 1):
                cnt = min(k + 1, d)
                if k > d - 1:
                    cnt = min(cnt, 2 * d - 1 - k)
                n_pairs += cnt
            quad_corr = scale_tv * n_pairs / (4.0 * S * S)

            net = margin - cell_var - quad_corr
            if net > best_net:
                best_net = net
                best_info = (net, ell, s_lo, margin, cell_var, quad_corr)

    if best_info is None:
        return (-1e30, 0, 0, 0.0, 0.0, 0.0)
    return best_info


# =====================================================================
# New method: multi-window LP (Sion dual) over cell vertices + samples
# =====================================================================

def minimax_box_cert_lp(mu_star, d, S, c_target, n_interior_samples=500):
    """Minimax certification via LP over window mixtures.

    Evaluates TV at vertices of box∩simplex + interior samples.
    Solves LP to find best window mixture.

    Returns dict with results.
    """
    eps = 1.0 / (2.0 * S)
    mu = mu_star.astype(np.float64) / S
    mu_lo = np.maximum(mu - eps, 0.0)
    mu_hi = mu + eps

    # Collect killing windows at center
    windows = all_windows(d)
    killing = []
    for ell, s_lo in windows:
        tv = tv_window(mu, d, ell, s_lo)
        if tv >= c_target - 1e-12:
            killing.append((ell, s_lo))

    if not killing:
        return {
            'certified': False,
            'lb': 0.0,
            'net': -c_target,
            'n_killing': 0,
            'n_test_points': 0,
            'method': 'no_killing_windows',
        }

    n_windows = len(killing)

    # Enumerate vertices of the cell
    verts = cell_vertices(mu_lo, mu_hi, d)

    # Also sample interior points
    rng = np.random.default_rng(42)
    interior = sample_cell_interior(mu, eps, d, n_interior_samples, rng)

    test_points = verts + interior
    n_points = len(test_points)

    if n_points == 0:
        return {
            'certified': False,
            'lb': 0.0,
            'net': -c_target,
            'n_killing': n_windows,
            'n_test_points': 0,
            'method': 'no_test_points',
        }

    # Build TV matrix: tv_matrix[i, w] = TV_{killing[w]}(test_points[i])
    tv_matrix = np.zeros((n_points, n_windows), dtype=np.float64)
    for pi, pt in enumerate(test_points):
        for wi, (ell, s_lo) in enumerate(killing):
            tv_matrix[pi, wi] = tv_window(pt, d, ell, s_lo)

    # --- Check single-window best (multi-window without mixing) ---
    # For each window, find its worst-case test point
    single_best_lb = -1e30
    single_best_w = -1
    for wi in range(n_windows):
        worst = tv_matrix[:, wi].min()
        if worst > single_best_lb:
            single_best_lb = worst
            single_best_w = wi

    # --- LP: find optimal window mixture ---
    lp_lb = single_best_lb  # fallback
    lp_lambda = None

    try:
        from scipy.optimize import linprog

        # Variables: [lambda_0, ..., lambda_{W-1}, z]
        # Maximize z  (minimize -z)
        c_obj = np.zeros(n_windows + 1)
        c_obj[-1] = -1.0

        # Constraints: z <= sum_w lambda_w tv[i,w] for all test points i
        # Rewrite: -sum_w lambda_w tv[i,w] + z <= 0
        A_ub = np.zeros((n_points, n_windows + 1))
        A_ub[:, :n_windows] = -tv_matrix
        A_ub[:, -1] = 1.0
        b_ub = np.zeros(n_points)

        # Equality: sum lambda = 1
        A_eq = np.zeros((1, n_windows + 1))
        A_eq[0, :n_windows] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0, None)] * n_windows + [(None, None)]

        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')

        if result.success:
            lp_lb = result.x[-1]
            lp_lambda = result.x[:n_windows]

    except ImportError:
        pass

    # --- Also check: does EVERY test point have SOME window >= c_target? ---
    all_covered = True
    uncovered_points = []
    for pi, pt in enumerate(test_points):
        max_tv = tv_matrix[pi, :].max()
        if max_tv < c_target:
            all_covered = False
            uncovered_points.append((pi, max_tv, pt))

    return {
        'certified_lp': lp_lb >= c_target,
        'certified_coverage': all_covered,
        'lb_lp': lp_lb,
        'net_lp': lp_lb - c_target,
        'lb_single_best': single_best_lb,
        'net_single_best': single_best_lb - c_target,
        'n_killing': n_windows,
        'n_test_points': n_points,
        'n_vertices': len(verts),
        'n_interior': len(interior),
        'n_uncovered': len(uncovered_points),
        'lambda': lp_lambda,
        'killing_windows': killing,
    }


# =====================================================================
# Pruning (pure Python, for test clarity)
# =====================================================================

def prune_coarse_python(batch_int, d, S, c_target):
    """Pure-Python coarse pruning. Returns (survived_mask, pruned_info)."""
    B = batch_int.shape[0]
    survived = np.ones(B, dtype=bool)
    pruned_info = [None] * B

    for b in range(B):
        mu = batch_int[b].astype(np.float64) / S
        conv_len = 2 * d - 1
        conv = np.zeros(conv_len, dtype=np.float64)
        for i in range(d):
            for j in range(d):
                conv[i + j] += mu[i] * mu[j]

        killers = []
        for ell in range(2, 2 * d + 1):
            n_cv = ell - 1
            for s_lo in range(conv_len - n_cv + 1):
                ws = sum(conv[s_lo:s_lo + n_cv])
                tv = ws * 2.0 * d / ell
                if tv >= c_target - 1e-12:
                    killers.append((tv, ell, s_lo, tv - c_target))

        if killers:
            survived[b] = False
            pruned_info[b] = killers

    return survived, pruned_info


# =====================================================================
# Unit tests
# =====================================================================

def test_cell_vertices():
    """Verify cell vertices are feasible and on the boundary."""
    print("\n[TEST] Cell vertex enumeration...")
    rng = np.random.default_rng(42)
    n_ok = 0

    for _ in range(100):
        d = rng.integers(3, 8)
        S = rng.integers(5, 20)
        k = rng.multinomial(S, np.ones(d) / d).astype(np.int32)
        mu = k.astype(np.float64) / S
        eps = 1.0 / (2.0 * S)
        mu_lo = np.maximum(mu - eps, 0.0)
        mu_hi = mu + eps

        verts = cell_vertices(mu_lo, mu_hi, d)
        for v in verts:
            # Check sum=1
            assert abs(v.sum() - 1.0) < 1e-10, \
                f"sum={v.sum()}, d={d}, S={S}"
            # Check bounds
            assert np.all(v >= mu_lo - 1e-10), f"below lo: {v} < {mu_lo}"
            assert np.all(v <= mu_hi + 1e-10), f"above hi: {v} > {mu_hi}"
            # Check nonneg
            assert np.all(v >= -1e-10), f"negative: {v}"
            # Check that d-1 coords are at bounds (vertex property)
            at_bound = np.sum(
                (np.abs(v - mu_lo) < 1e-10) | (np.abs(v - mu_hi) < 1e-10))
            assert at_bound >= d - 1, \
                f"only {at_bound} coords at bounds, need {d-1}"
        n_ok += 1

    print(f"  PASSED: {n_ok}/100 random tests.")
    return True


def test_tv_consistency():
    """Verify TV computation matches the known formula."""
    print("\n[TEST] TV computation consistency...")
    rng = np.random.default_rng(123)

    for _ in range(200):
        d = rng.integers(3, 8)
        mu = rng.dirichlet(np.ones(d))
        ell = rng.integers(2, 2 * d + 1)
        conv_len = 2 * d - 1
        n_cv = ell - 1
        s_lo = rng.integers(0, conv_len - n_cv + 1)

        # Method 1: direct sum
        tv1 = tv_window(mu, d, ell, s_lo)

        # Method 2: via autoconvolution array
        conv = np.zeros(conv_len, dtype=np.float64)
        for i in range(d):
            for j in range(d):
                conv[i + j] += mu[i] * mu[j]
        ws = sum(conv[s_lo:s_lo + n_cv])
        tv2 = ws * 2.0 * d / ell

        assert abs(tv1 - tv2) < 1e-12, f"tv mismatch: {tv1} vs {tv2}"

    print(f"  PASSED: 200 random tests.")
    return True


def test_old_method_soundness():
    """Check that old method's PASS is consistent with simplex-constrained
    evaluation at cell vertices.

    Old method says: TV_{W*}(mu) >= c_target for all mu in cell.
    We check this at all vertices of cell (necessary condition, since
    vertices are in the cell).
    """
    print("\n[TEST] Old method soundness at cell vertices...")
    rng = np.random.default_rng(456)
    n_tested = 0
    n_violations = 0

    for _ in range(200):
        d = rng.integers(3, 7)
        S = rng.integers(8, 25)
        c_target = 1.0 + rng.uniform(0, 0.3)
        k = rng.multinomial(S, np.ones(d) / d).astype(np.int32)

        mu = k.astype(np.float64) / S
        best_tv, _, _ = tv_max_over_windows(mu, d)
        if best_tv < c_target:
            continue

        old = old_box_cert(k, d, S, c_target)
        if old[0] < 0:
            continue  # old says FAIL, nothing to check

        n_tested += 1
        # Old says PASS with window (old[1], old[2]).
        # Check that this window gives >= c_target at all cell vertices.
        eps = 1.0 / (2.0 * S)
        mu_lo = np.maximum(mu - eps, 0.0)
        mu_hi = mu + eps
        verts = cell_vertices(mu_lo, mu_hi, d)

        ell_w, s_w = old[1], old[2]
        for v in verts:
            tv_v = tv_window(v, d, ell_w, s_w)
            if tv_v < c_target - 1e-9:
                n_violations += 1
                if n_violations <= 3:
                    print(f"    VIOLATION: k={k}, d={d}, S={S}, c={c_target:.3f}")
                    print(f"      Old net={old[0]:.6f}, window=({ell_w},{s_w})")
                    print(f"      vertex={np.array2string(v, precision=4)}")
                    print(f"      TV at vertex = {tv_v:.6f} < {c_target:.6f}")
                break

    if n_violations == 0:
        print(f"  PASSED: {n_tested} old-PASS compositions, "
              f"all vertices confirmed.")
    else:
        print(f"  FOUND {n_violations} violations out of {n_tested} tests.")
        print(f"  (This indicates the old method may be unsound for some configs.)")
    return True  # informational, not a hard failure


def test_lp_soundness():
    """Verify LP result is consistent with direct evaluation."""
    print("\n[TEST] LP soundness...")
    rng = np.random.default_rng(789)
    n_violations = 0
    n_tested = 0

    for _ in range(50):
        d = rng.integers(3, 6)
        S = rng.integers(8, 18)
        c_target = 1.0 + rng.uniform(0, 0.2)
        k = rng.multinomial(S, np.ones(d) / d).astype(np.int32)

        mu = k.astype(np.float64) / S
        best_tv, _, _ = tv_max_over_windows(mu, d)
        if best_tv < c_target:
            continue

        n_tested += 1
        result = minimax_box_cert_lp(k, d, S, c_target, n_interior_samples=200)

        # LP says the minimax lower bound is lb_lp.
        # Verify: at every test point, the mixed-window TV >= lb_lp (within tol).
        if result['lambda'] is not None and result['n_test_points'] > 0:
            lam = result['lambda']
            killing = result['killing_windows']

            eps = 1.0 / (2.0 * S)
            mu_lo = np.maximum(mu - eps, 0.0)
            mu_hi = mu + eps
            verts = cell_vertices(mu_lo, mu_hi, d)
            interior = sample_cell_interior(mu, eps, d, 200, rng)
            test_points = verts + interior

            for pt in test_points:
                mixed_tv = sum(lam[wi] * tv_window(pt, d, ell, s_lo)
                               for wi, (ell, s_lo) in enumerate(killing))
                if mixed_tv < result['lb_lp'] - 1e-8:
                    n_violations += 1
                    break

    if n_violations == 0:
        print(f"  PASSED: {n_tested} tests, LP bounds consistent.")
    else:
        print(f"  FAILED: {n_violations} violations!")
    return n_violations == 0


# =====================================================================
# Main comparison
# =====================================================================

def run_comparison(d0, S, c_target, max_compositions=50000, verbose=True):
    """Compare old single-window vs new minimax box certification."""
    n_total = count_compositions(d0, S)
    if verbose:
        print(f"\n{'='*70}")
        print(f"MINIMAX BOX CERTIFICATION COMPARISON")
        print(f"  d0={d0}, S={S}, c_target={c_target}")
        print(f"  Total compositions: {n_total:,}")
        print(f"  Cell half-width eps = 1/(2S) = {1/(2*S):.6f}")
        print(f"{'='*70}")

    # Generate compositions
    t0 = time.time()
    all_comps = []
    gen = generate_canonical_compositions_batched(d0, S, batch_size=100_000)
    for batch in gen:
        all_comps.append(batch)
        if sum(len(c) for c in all_comps) >= max_compositions:
            break
    comps = np.vstack(all_comps)[:max_compositions]
    if verbose:
        print(f"  Generated {len(comps):,} canonical compositions "
              f"in {time.time()-t0:.2f}s")

    # Prune
    t0 = time.time()
    survived, pruned_info = prune_coarse_python(comps, d0, S, c_target)
    n_pruned = int(np.sum(~survived))
    if verbose:
        print(f"  Pruned: {n_pruned:,} / {len(comps):,} "
              f"({100*n_pruned/len(comps):.1f}%) in {time.time()-t0:.2f}s")

    if n_pruned == 0:
        print("  No pruned compositions to certify. Try lower c_target.")
        return None

    pruned_idx = np.where(~survived)[0]

    # Select compositions to check (sample if too many)
    check_idx = pruned_idx
    if len(check_idx) > 2000:
        rng = np.random.default_rng(12345)
        margins = []
        for idx in pruned_idx:
            best_margin = max(info[3] for info in pruned_info[idx])
            margins.append(best_margin)
        margins = np.array(margins)
        order = np.argsort(margins)
        hard = pruned_idx[order[:1000]]
        rest = pruned_idx[order[1000:]]
        if len(rest) > 1000:
            rest = rng.choice(rest, 1000, replace=False)
        check_idx = np.concatenate([hard, rest])

    if verbose:
        print(f"\n  Checking {len(check_idx):,} pruned compositions...")

    # Run comparison
    t0 = time.time()
    n_old_pass = 0
    n_lp_pass = 0
    n_coverage_pass = 0
    worst_old = 1e30
    worst_lp = 1e30
    worst_single = 1e30

    improvement_examples = []

    for count, idx in enumerate(check_idx):
        mu_star = comps[idx]

        # Old method
        old = old_box_cert(mu_star, d0, S, c_target)
        old_net = old[0]
        if old_net >= 0:
            n_old_pass += 1
        if old_net < worst_old:
            worst_old = old_net

        # New: minimax LP
        new = minimax_box_cert_lp(mu_star, d0, S, c_target,
                                   n_interior_samples=300)
        if new['net_lp'] >= 0:
            n_lp_pass += 1
        if new.get('certified_coverage', False):
            n_coverage_pass += 1
        if new['net_lp'] < worst_lp:
            worst_lp = new['net_lp']
        if new['net_single_best'] < worst_single:
            worst_single = new['net_single_best']

        # Track cases where LP passes but old fails
        if new['net_lp'] >= 0 and old_net < 0:
            improvement_examples.append({
                'idx': idx,
                'mu_star': mu_star.copy(),
                'old_net': old_net,
                'lp_net': new['net_lp'],
                'n_killing': new['n_killing'],
            })

        if verbose and (count + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (count + 1) / elapsed
            print(f"    [{count+1}/{len(check_idx)}] "
                  f"{rate:.0f}/s, "
                  f"old_pass={n_old_pass} lp_pass={n_lp_pass} "
                  f"coverage_pass={n_coverage_pass}")

    elapsed = time.time() - t0
    n_checked = len(check_idx)

    if verbose:
        print(f"\n{'='*70}")
        print(f"RESULTS ({n_checked:,} compositions, {elapsed:.2f}s)")
        print(f"{'='*70}")
        print(f"\n  Old method (single-window, triangle inequality):")
        print(f"    Certified: {n_old_pass:,} / {n_checked:,} "
              f"({100*n_old_pass/n_checked:.1f}%)")
        print(f"    Worst net: {worst_old:.6f}")
        print(f"\n  Multi-window best single (no mixing):")
        print(f"    Worst net: {worst_single:.6f}")
        print(f"\n  LP method (Sion dual, window mixtures):")
        print(f"    Certified: {n_lp_pass:,} / {n_checked:,} "
              f"({100*n_lp_pass/n_checked:.1f}%)")
        print(f"    Worst net: {worst_lp:.6f}")
        print(f"\n  Coverage (every test point has some window >= c):")
        print(f"    Certified: {n_coverage_pass:,} / {n_checked:,} "
              f"({100*n_coverage_pass/n_checked:.1f}%)")

        if n_lp_pass > n_old_pass:
            print(f"\n  >>> LP certifies {n_lp_pass - n_old_pass:,} MORE "
                  f"than old method <<<")
        elif n_lp_pass == n_old_pass:
            print(f"\n  Both methods certify the same number.")
        else:
            print(f"\n  Old method certifies MORE than LP.")
            print(f"  (Old method may be UNSOUND — check with test_old_method_soundness)")

        if improvement_examples:
            print(f"\n  IMPROVEMENT EXAMPLES (LP passes, old fails):")
            for ex in improvement_examples[:5]:
                print(f"    k={ex['mu_star']}: old_net={ex['old_net']:.4f} "
                      f"-> lp_net={ex['lp_net']:.4f} "
                      f"({ex['n_killing']} killing windows)")

    return {
        'n_checked': n_checked,
        'n_old_pass': n_old_pass,
        'n_lp_pass': n_lp_pass,
        'n_coverage_pass': n_coverage_pass,
        'worst_old': worst_old,
        'worst_lp': worst_lp,
        'worst_single': worst_single,
        'improvement_examples': improvement_examples,
    }


# =====================================================================
# Detailed single-composition analysis
# =====================================================================

def analyze_one(mu_star, d, S, c_target):
    """Detailed analysis of one composition, comparing all methods."""
    mu = mu_star.astype(np.float64) / S
    eps = 1.0 / (2.0 * S)

    print(f"\n  Composition (int): {mu_star}")
    print(f"  Mass (float):      {np.array2string(mu, precision=4)}")
    print(f"  Sum: {mu.sum():.6f}")

    # Grid-point TV
    best_tv, best_ell, best_s = tv_max_over_windows(mu, d)
    print(f"\n  Grid-point max TV: {best_tv:.6f} (ell={best_ell}, s={best_s})")
    print(f"  Margin: {best_tv - c_target:.6f}")

    # Old method
    old = old_box_cert(mu_star, d, S, c_target)
    print(f"\n  OLD method (single-window):")
    print(f"    Window: ell={old[1]}, s={old[2]}")
    print(f"    margin={old[3]:.6f}, cell_var={old[4]:.6f}, "
          f"quad_corr={old[5]:.6f}")
    print(f"    net = {old[0]:.6f}  {'PASS' if old[0] >= 0 else 'FAIL'}")

    # LP method
    new = minimax_box_cert_lp(mu_star, d, S, c_target, n_interior_samples=500)
    print(f"\n  LP method (Sion dual):")
    print(f"    n_killing_windows: {new['n_killing']}")
    print(f"    n_test_points: {new['n_test_points']} "
          f"({new['n_vertices']} verts + {new['n_interior']} interior)")
    print(f"    LP lower bound: {new['lb_lp']:.6f}")
    print(f"    net = {new['net_lp']:.6f}  "
          f"{'PASS' if new['net_lp'] >= 0 else 'FAIL'}")
    print(f"    Coverage: {'PASS' if new['certified_coverage'] else 'FAIL'} "
          f"({new['n_uncovered']} uncovered points)")

    if new['lambda'] is not None:
        n_active = np.sum(new['lambda'] > 1e-6)
        print(f"    Active windows: {n_active}")
        top = np.argsort(new['lambda'])[::-1][:5]
        for idx in top:
            if new['lambda'][idx] > 1e-6 and idx < len(new['killing_windows']):
                ell, s_lo = new['killing_windows'][idx]
                print(f"      lambda={new['lambda'][idx]:.4f} -> "
                      f"(ell={ell}, s={s_lo})")

    # Check at all cell vertices with each method
    mu_lo = np.maximum(mu - eps, 0.0)
    mu_hi = mu + eps
    verts = cell_vertices(mu_lo, mu_hi, d)
    print(f"\n  Vertex analysis ({len(verts)} vertices):")

    worst_max_tv = 1e30
    worst_vert = None
    for v in verts:
        max_tv, w_ell, w_s = tv_max_over_windows(v, d)
        if max_tv < worst_max_tv:
            worst_max_tv = max_tv
            worst_vert = v.copy()

    print(f"    Worst vertex max_W TV: {worst_max_tv:.6f}")
    print(f"    net (worst vertex): {worst_max_tv - c_target:.6f}  "
          f"{'PASS' if worst_max_tv >= c_target else 'FAIL'}")
    if worst_vert is not None:
        print(f"    Worst vertex: {np.array2string(worst_vert, precision=4)}")
        print(f"    Worst vertex sum: {worst_vert.sum():.10f}")


# =====================================================================
# Entry point
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Test minimax box certification (Sion duality)")
    parser.add_argument('--d0', type=int, default=6,
                        help='Number of bins (default: 6)')
    parser.add_argument('--S', type=int, default=30,
                        help='Grid resolution (default: 30)')
    parser.add_argument('--c_target', type=float, default=1.20,
                        help='Target constant (default: 1.20)')
    parser.add_argument('--max_comps', type=int, default=50000,
                        help='Max compositions to process')
    parser.add_argument('--analyze', type=int, default=3,
                        help='Number of hard compositions to analyze in detail')
    parser.add_argument('--skip_tests', action='store_true',
                        help='Skip unit tests')
    args = parser.parse_args()

    print("=" * 70)
    print("MINIMAX BOX CERTIFICATION — SION'S THEOREM")
    print("=" * 70)

    if not args.skip_tests:
        ok1 = test_cell_vertices()
        ok2 = test_tv_consistency()
        ok3 = test_old_method_soundness()
        ok4 = test_lp_soundness()
        if not (ok1 and ok2 and ok4):
            print("\n*** UNIT TESTS FAILED — aborting ***")
            return
        print()

    result = run_comparison(args.d0, args.S, args.c_target,
                            max_compositions=args.max_comps)

    if result is None:
        return

    # Detailed analysis
    if args.analyze > 0:
        print(f"\n{'='*70}")
        print(f"DETAILED ANALYSIS OF {args.analyze} HARDEST COMPOSITIONS")
        print(f"{'='*70}")

        all_comps = []
        gen = generate_canonical_compositions_batched(
            args.d0, args.S, batch_size=100_000)
        for batch in gen:
            all_comps.append(batch)
            if sum(len(c) for c in all_comps) >= args.max_comps:
                break
        comps = np.vstack(all_comps)[:args.max_comps]

        survived, pruned_info = prune_coarse_python(
            comps, args.d0, args.S, args.c_target)
        pruned_idx = np.where(~survived)[0]

        if len(pruned_idx) > 0:
            margins = []
            for idx in pruned_idx:
                best_margin = max(info[3] for info in pruned_info[idx])
                margins.append(best_margin)
            order = np.argsort(margins)

            for i in range(min(args.analyze, len(order))):
                idx = pruned_idx[order[i]]
                print(f"\n  --- Hardest #{i+1} ---")
                analyze_one(comps[idx], args.d0, args.S, args.c_target)


if __name__ == '__main__':
    main()
