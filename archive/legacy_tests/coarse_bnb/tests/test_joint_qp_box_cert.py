#!/usr/bin/env python
r"""Test joint QP box certification vs. separate-bound method.

Mathematical setup
==================
The coarse cascade prover partitions [-1/4, 1/4] into d equal bins.
A mass distribution is a vector mu in R^d with mu_i >= 0, sum(mu) = 1.
The grid has spacing delta = 1/S, so grid points are integer compositions
c_i with sum = S, giving mu*_i = c_i / S.

For a window W = (ell, s_lo), the test value is:

    TV_W(mu) = (2d / ell) * sum_{s_lo <= i+j <= s_lo+ell-2} mu_i * mu_j

The cascade proves: for every grid point mu*, some window W has TV_W(mu*) >= c_target.

Box certification must extend this to ALL continuous mu in the Voronoi cell
around mu*:

    Cell(mu*) = { mu : |mu_i - mu*_i| <= 1/(2S), sum(mu_i) = 1, mu_i >= 0 }

CURRENT METHOD (separate bound):
    net = margin - cell_var - quad_corr
    where:
        margin = TV_W(mu*) - c_target
        cell_var = max_{delta in C} |grad(TV_W) . delta|    (first-order)
        quad_corr = max_{delta in C} |delta^T H delta / 2|  (second-order)
    The two maxima are computed INDEPENDENTLY over different worst-case deltas.
    This overestimates the drop because the worst delta for grad and for
    the Hessian point in different directions.

JOINT QP METHOD (this test):
    Compute: min_{mu in Cell(mu*)} TV_W(mu) directly.
    Since TV_W is quadratic in mu, this is a constrained QP.
    The QP finds the ACTUAL minimum, accounting for the joint effect of
    linear and quadratic terms.

    TV_W is NOT convex in general (the window's contribution matrix A_W
    can be indefinite), so we use global optimization (basin-hopping with
    multiple restarts) and verify with a McCormick LP lower bound.

McCormick LP relaxation
=======================
Replace each bilinear product mu_i * mu_j with a new variable w_{ij},
bounded by the exact convex envelope over the box [lo_i, hi_i] x [lo_j, hi_j]:

    w_{ij} >= lo_i*mu_j + lo_j*mu_i - lo_i*lo_j
    w_{ij} >= hi_i*mu_j + hi_j*mu_i - hi_i*hi_j
    w_{ij} <= hi_i*mu_j + lo_j*mu_i - hi_i*lo_j
    w_{ij} <= lo_i*mu_j + hi_j*mu_i - lo_i*hi_j

For squared terms w_{ii} = mu_i^2 on [lo_i, hi_i]:
    w_{ii} >= mu_i^2  (convex, so secant is upper bound, tangent is lower)
    w_{ii} >= 2*mu*_i * mu_i - mu*_i^2   (tangent at mu*_i, valid lower bound)
    w_{ii} <= (lo_i + hi_i)*mu_i - lo_i*hi_i   (secant, valid upper bound)

Then: min TV_W = min (2d/ell) * sum_{(i,j) in W} a_{ij} * w_{ij}
subject to McCormick constraints + sum(mu) = 1 + box bounds
is a LINEAR PROGRAM -- a rigorous lower bound on the true minimum.

The McCormick LP bound is RIGOROUS: any feasible point of the LP gives a
valid lower bound on TV_W over the cell. If LP_min >= c_target, the cell
is certified with mathematical certainty.
"""
import sys
import os
import time
import argparse

import numpy as np

# Ensure imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..',
                                'cloninger-steinerberger', 'cpu'))

from pruning import count_compositions, asymmetry_threshold
from compositions import generate_canonical_compositions_batched
from run_cascade import _build_pair_prefix


# =====================================================================
# TV_W computation (exact, for verification)
# =====================================================================

def compute_tv_w(mu, d, ell, s_lo):
    """Compute TV_W(mu) for window (ell, s_lo).

    TV_W = (2d/ell) * sum_{s_lo <= i+j <= s_lo+ell-2} mu_i * mu_j

    The sum is over ALL ordered pairs (i,j) with i+j in the window range.
    This includes both mu_i*mu_j and mu_j*mu_i for i != j, plus mu_i^2.
    """
    conv = np.zeros(2 * d - 1, dtype=np.float64)
    for i in range(d):
        for j in range(d):
            conv[i + j] += mu[i] * mu[j]
    # Window sum
    ws = 0.0
    for k in range(s_lo, s_lo + ell - 1):
        if 0 <= k < len(conv):
            ws += conv[k]
    return (2.0 * d / ell) * ws


def find_all_killing_windows(c_int, d, S, c_target):
    """Find ALL windows where TV_W(mu*) >= c_target.

    Returns list of (ell, s_lo, tv, margin) tuples, sorted by margin desc.
    """
    mu = c_int.astype(np.float64) / S
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.float64)
    for i in range(d):
        for j in range(d):
            conv[i + j] += mu[i] * mu[j]

    windows = []
    max_ell = 2 * d
    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        for s_lo in range(conv_len - n_cv + 1):
            ws = sum(conv[s_lo:s_lo + n_cv])
            tv = (2.0 * d / ell) * ws
            if tv >= c_target - 1e-12:
                windows.append((ell, s_lo, tv, tv - c_target))
    windows.sort(key=lambda x: -x[3])  # sort by margin descending
    return windows


# =====================================================================
# Current separate-bound box certification (reference implementation)
# =====================================================================

def box_cert_separate(c_int, d, S, c_target, ell, s_lo):
    """Current method: separate first-order and second-order bounds.

    Returns (margin, cell_var, quad_corr, net).
    """
    mu = c_int.astype(np.float64) / S
    r = 1.0 / (2.0 * S)  # cell half-width

    # TV at grid point
    tv = compute_tv_w(mu, d, ell, s_lo)
    margin = tv - c_target

    # Gradient: g_i = (4d/ell) * sum_{j: s_lo <= i+j <= s_lo+ell-2} mu_j
    grad = np.zeros(d, dtype=np.float64)
    for i in range(d):
        g = 0.0
        for j in range(d):
            if s_lo <= i + j <= s_lo + ell - 2:
                g += mu[j]
        grad[i] = (4.0 * d / ell) * g

    # cell_var with sum(delta)=0 constraint (current code's approach):
    # Sort gradient, pair top d/2 with bottom d/2
    g_sorted = np.sort(grad)
    cell_var = 0.0
    for k in range(d // 2):
        cell_var += g_sorted[d - 1 - k] - g_sorted[k]
    cell_var *= r

    # Quadratic correction (current code's approach using pair counts)
    conv_len = 2 * d - 1
    # N_W: number of ordered pairs (i,j) with s_lo <= i+j <= s_lo+ell-2
    # M_W: number of self-pairs i=j in window
    N_W = 0
    M_W = 0
    for k in range(s_lo, s_lo + ell - 1):
        if 0 <= k < conv_len:
            # n_k = number of pairs summing to k
            nk = min(k + 1, d, conv_len - k)
            N_W += nk
            # self-pair: k even and k//2 < d
            if k % 2 == 0 and k // 2 < d:
                M_W += 1
    cross_W = N_W - M_W
    compl_bound = d * d - N_W
    pb = min(cross_W, compl_bound)
    pb = max(pb, 0)
    quad_corr = (2.0 * d / ell) * pb * r * r

    net = margin - cell_var - quad_corr
    return margin, cell_var, quad_corr, net


# =====================================================================
# Naive separate bound (no sum(delta)=0 constraint)
# =====================================================================

def box_cert_naive(c_int, d, S, c_target, ell, s_lo):
    """Naive method: cell_var without sum(delta)=0 constraint.

    cell_var_naive = r * sum(|g_i|) -- worst case over unconstrained box.
    """
    mu = c_int.astype(np.float64) / S
    r = 1.0 / (2.0 * S)

    tv = compute_tv_w(mu, d, ell, s_lo)
    margin = tv - c_target

    grad = np.zeros(d, dtype=np.float64)
    for i in range(d):
        g = 0.0
        for j in range(d):
            if s_lo <= i + j <= s_lo + ell - 2:
                g += mu[j]
        grad[i] = (4.0 * d / ell) * g

    cell_var_naive = r * np.sum(np.abs(grad))
    return cell_var_naive


# =====================================================================
# Joint QP: actual minimum of TV_W over the Voronoi cell
# =====================================================================

def joint_qp_min(c_int, d, S, c_target, ell, s_lo, n_restarts=20):
    """Compute min TV_W(mu) over the Voronoi cell using scipy optimization.

    The cell is: { mu : |mu_i - mu*_i| <= 1/(2S), sum(mu) = 1, mu_i >= 0 }

    TV_W(mu) = (2d/ell) * sum_{(i,j): s_lo <= i+j <= s_lo+ell-2} mu_i mu_j

    Since TV_W is quadratic and possibly non-convex, we use multiple restarts
    of L-BFGS-B with the equality constraint handled via projection.

    Returns (min_tv, min_mu, n_evals).
    """
    from scipy.optimize import minimize, LinearConstraint

    mu_star = c_int.astype(np.float64) / S
    r = 1.0 / (2.0 * S)

    # Box bounds intersected with [0, inf)
    lo = np.maximum(mu_star - r, 0.0)
    hi = mu_star + r  # hi is always <= 1 since mu* <= 1

    # Build the window contribution matrix A_W
    # TV_W(mu) = (2d/ell) * mu^T A_W mu
    # where A_W[i,j] = 1 if s_lo <= i+j <= s_lo+ell-2
    A_W = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            if s_lo <= i + j <= s_lo + ell - 2:
                A_W[i, j] = 1.0
    scale = 2.0 * d / ell

    def tv_func(mu):
        return scale * mu @ A_W @ mu

    def tv_grad(mu):
        return scale * 2.0 * A_W @ mu

    # Use SLSQP with equality constraint sum(mu) = 1
    bounds = list(zip(lo, hi))
    eq_constraint = {'type': 'eq', 'fun': lambda mu: np.sum(mu) - 1.0,
                     'jac': lambda mu: np.ones(d)}

    best_val = np.inf
    best_mu = None
    n_evals = 0

    # Start points: grid center + random perturbations + box corners
    starts = [mu_star.copy()]

    rng = np.random.RandomState(42)
    for _ in range(n_restarts):
        delta = rng.uniform(-r, r, size=d)
        delta -= delta.mean()  # project onto sum=0
        mu_try = mu_star + delta
        mu_try = np.clip(mu_try, lo, hi)
        # Re-normalize to sum=1
        mu_try = mu_try / mu_try.sum()
        mu_try = np.clip(mu_try, lo, hi)
        starts.append(mu_try)

    # Also try corners that minimize the quadratic form heuristically
    # For each coordinate with large gradient, push to boundary
    grad_center = tv_grad(mu_star)
    # Push toward low-gradient directions
    idx_sorted = np.argsort(grad_center)
    for n_push in [d // 4, d // 2, 3 * d // 4]:
        mu_corner = mu_star.copy()
        for k in range(n_push):
            mu_corner[idx_sorted[k]] = lo[idx_sorted[k]]
        for k in range(n_push, d):
            mu_corner[idx_sorted[k]] = hi[idx_sorted[k]]
        # Re-project to sum=1
        excess = np.sum(mu_corner) - 1.0
        for k in range(d):
            adj = np.clip(mu_corner[k] - excess / d, lo[k], hi[k])
            excess -= (mu_corner[k] - adj)
            mu_corner[k] = adj
        starts.append(mu_corner)

    for mu0 in starts:
        try:
            res = minimize(tv_func, mu0, jac=tv_grad, method='SLSQP',
                           bounds=bounds, constraints=[eq_constraint],
                           options={'maxiter': 500, 'ftol': 1e-15})
            n_evals += res.nfev
            if res.fun < best_val:
                # Verify feasibility
                mu_res = res.x
                if (np.all(mu_res >= lo - 1e-10) and
                        np.all(mu_res <= hi + 1e-10) and
                        abs(np.sum(mu_res) - 1.0) < 1e-8):
                    best_val = res.fun
                    best_mu = mu_res.copy()
        except Exception:
            pass

    return best_val, best_mu, n_evals


# =====================================================================
# McCormick LP: rigorous lower bound
# =====================================================================

def mccormick_lp_bound(c_int, d, S, c_target, ell, s_lo):
    """Rigorous lower bound on min TV_W via McCormick LP relaxation.

    Replaces each bilinear term mu_i*mu_j with a variable w_ij constrained
    by the exact convex/concave envelopes over the box.

    Returns (lb, status) where lb is a rigorous lower bound on TV_W
    over the Voronoi cell, and status is the solver status string.
    """
    from scipy.optimize import linprog

    mu_star = c_int.astype(np.float64) / S
    r = 1.0 / (2.0 * S)
    lo = np.maximum(mu_star - r, 0.0)
    hi = mu_star + r
    scale = 2.0 * d / ell

    # Identify which (i,j) pairs contribute to the window
    # TV_W = scale * sum_{(i,j) in W_pairs} mu_i * mu_j
    # where W_pairs = {(i,j) : s_lo <= i+j <= s_lo+ell-2}
    W_pairs = []
    for i in range(d):
        for j in range(i, d):
            if s_lo <= i + j <= s_lo + ell - 2:
                W_pairs.append((i, j))

    n_pairs = len(W_pairs)
    if n_pairs == 0:
        return 0.0, "no_pairs"

    # Variables: mu[0..d-1], w[0..n_pairs-1]
    # w[k] represents mu_i * mu_j for the k-th pair
    n_vars = d + n_pairs

    # Objective: minimize scale * sum_k coeff_k * w_k
    c_obj = np.zeros(n_vars, dtype=np.float64)
    for k, (i, j) in enumerate(W_pairs):
        if i == j:
            c_obj[d + k] = scale  # mu_i^2 counted once in conv
        else:
            c_obj[d + k] = 2.0 * scale  # mu_i*mu_j + mu_j*mu_i

    # Bounds on mu
    bounds = []
    for i in range(d):
        bounds.append((lo[i], hi[i]))
    # Bounds on w (will be constrained by McCormick, but set wide initially)
    for k, (i, j) in enumerate(W_pairs):
        w_lo = lo[i] * lo[j]  # minimum product (all nonneg)
        w_hi = hi[i] * hi[j]  # maximum product
        bounds.append((w_lo, w_hi))

    # Equality constraint: sum(mu) = 1
    A_eq = np.zeros((1, n_vars), dtype=np.float64)
    for i in range(d):
        A_eq[0, i] = 1.0
    b_eq = np.array([1.0])

    # McCormick inequality constraints
    # For each pair (i,j), w_ij is constrained by:
    #   w >= lo_i * mu_j + lo_j * mu_i - lo_i * lo_j   (lower envelope 1)
    #   w >= hi_i * mu_j + hi_j * mu_i - hi_i * hi_j   (lower envelope 2)
    #   w <= hi_i * mu_j + lo_j * mu_i - hi_i * lo_j   (upper envelope 1)
    #   w <= lo_i * mu_j + hi_j * mu_i - lo_i * hi_j   (upper envelope 2)
    #
    # For diagonal (i=j):
    #   w >= 2*mu_star_i * mu_i - mu_star_i^2  (tangent lower bound)
    #   w <= (lo_i + hi_i) * mu_i - lo_i * hi_i  (secant upper bound)
    #   (Plus the general McCormick with lo_i=lo_j, hi_i=hi_j)

    A_ub_rows = []
    b_ub_vals = []

    for k, (i, j) in enumerate(W_pairs):
        w_idx = d + k
        li, hi_i = lo[i], hi[i]
        lj, hj = lo[j], hi[j]

        if i == j:
            # Diagonal: w_ii approximates mu_i^2
            # Lower bound: tangent at mu*_i: w >= 2*mu*_i * mu_i - mu*_i^2
            row = np.zeros(n_vars)
            row[w_idx] = -1.0
            row[i] = 2.0 * mu_star[i]
            A_ub_rows.append(row)
            b_ub_vals.append(mu_star[i] ** 2)

            # Lower bound: tangent at lo_i: w >= 2*lo_i * mu_i - lo_i^2
            row = np.zeros(n_vars)
            row[w_idx] = -1.0
            row[i] = 2.0 * li
            A_ub_rows.append(row)
            b_ub_vals.append(li ** 2)

            # Lower bound: tangent at hi_i: w >= 2*hi_i * mu_i - hi_i^2
            row = np.zeros(n_vars)
            row[w_idx] = -1.0
            row[i] = 2.0 * hi_i
            A_ub_rows.append(row)
            b_ub_vals.append(hi_i ** 2)

            # Upper bound: secant: w <= (lo_i + hi_i) * mu_i - lo_i * hi_i
            row = np.zeros(n_vars)
            row[w_idx] = 1.0
            row[i] = -(li + hi_i)
            A_ub_rows.append(row)
            b_ub_vals.append(-li * hi_i)

        else:
            # Off-diagonal McCormick envelopes

            # Lower 1: w >= lo_i * mu_j + lo_j * mu_i - lo_i * lo_j
            # => -w + lo_i * mu_j + lo_j * mu_i <= lo_i * lo_j
            row = np.zeros(n_vars)
            row[w_idx] = -1.0
            row[j] = li
            row[i] = lj
            A_ub_rows.append(row)
            b_ub_vals.append(li * lj)

            # Lower 2: w >= hi_i * mu_j + hi_j * mu_i - hi_i * hi_j
            # => -w + hi_i * mu_j + hi_j * mu_i <= hi_i * hi_j
            # Wait, this is the OTHER lower bound from the concave envelope.
            # Actually: w >= hi_i * mu_j + hj * mu_i - hi_i * hj
            row = np.zeros(n_vars)
            row[w_idx] = -1.0
            row[j] = hi_i
            row[i] = hj
            A_ub_rows.append(row)
            b_ub_vals.append(hi_i * hj)

            # Upper 1: w <= hi_i * mu_j + lo_j * mu_i - hi_i * lo_j
            # => w - hi_i * mu_j - lo_j * mu_i <= -hi_i * lo_j
            row = np.zeros(n_vars)
            row[w_idx] = 1.0
            row[j] = -hi_i
            row[i] = -lj
            A_ub_rows.append(row)
            b_ub_vals.append(-hi_i * lj)

            # Upper 2: w <= lo_i * mu_j + hi_j * mu_i - lo_i * hi_j
            # => w - lo_i * mu_j - hi_j * mu_i <= -lo_i * hi_j
            row = np.zeros(n_vars)
            row[w_idx] = 1.0
            row[j] = -li
            row[i] = -hj
            A_ub_rows.append(row)
            b_ub_vals.append(-li * hj)

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_vals) if b_ub_vals else None

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if result.success:
        return result.fun, "optimal"
    else:
        return -np.inf, result.message


# =====================================================================
# Verification helpers
# =====================================================================

def verify_tv_formula(c_int, d, S, ell, s_lo):
    """Cross-check TV computation between float and integer paths."""
    mu = c_int.astype(np.float64) / S

    # Float path
    tv_float = compute_tv_w(mu, d, ell, s_lo)

    # Integer path (matching cascade code)
    conv = np.zeros(2 * d - 1, dtype=np.int64)
    for i in range(d):
        ci = int(c_int[i])
        conv[2 * i] += ci * ci
        for j in range(i + 1, d):
            cj = int(c_int[j])
            conv[i + j] += 2 * ci * cj
    ws = sum(int(conv[k]) for k in range(s_lo, s_lo + ell - 1)
             if 0 <= k < len(conv))
    tv_int = (2.0 * d / ell) * ws / (S * S)

    assert abs(tv_float - tv_int) < 1e-10, \
        f"TV mismatch: float={tv_float}, int={tv_int}"
    return tv_float


def verify_joint_bound_ge_separate(margin, cell_var, quad_corr,
                                    qp_min, c_target):
    """The QP minimum should always be >= the separate lower bound.

    separate_lb = c_target + margin - cell_var - quad_corr
                = TV(mu*) - cell_var - quad_corr

    qp_min = actual minimum of TV over cell

    Must have: qp_min >= separate_lb (separate bound is looser).
    """
    separate_lb = c_target + margin - cell_var - quad_corr
    if qp_min < separate_lb - 1e-8:
        return False, f"QP min {qp_min:.8f} < separate lb {separate_lb:.8f}"
    return True, "OK"


def verify_mccormick_le_qp(lp_bound, qp_min):
    """McCormick LP bound must be <= QP minimum (LP is a relaxation)."""
    if lp_bound > qp_min + 1e-8:
        return False, f"LP {lp_bound:.8f} > QP {qp_min:.8f}"
    return True, "OK"


# =====================================================================
# Main analysis
# =====================================================================

def analyze_compositions(d, S, c_target, max_analyze=500, verbose=True):
    """Run full comparison on all pruned compositions at dimension d."""
    from pruning import asymmetry_threshold
    from compositions import generate_canonical_compositions_batched

    if verbose:
        print(f"\n{'='*72}")
        print(f"JOINT QP BOX CERTIFICATION TEST")
        print(f"  d={d}, S={S}, c_target={c_target}")
        n_total = count_compositions(d, S)
        print(f"  Total compositions: {n_total:,}")
        print(f"{'='*72}")

    threshold_a = asymmetry_threshold(c_target)

    # Collect pruned compositions with their killing windows
    pruned_data = []  # (c_int, ell, s_lo, margin)
    n_survived = 0
    n_processed = 0

    gen = generate_canonical_compositions_batched(d, S, batch_size=200_000)
    for batch in gen:
        n_processed += len(batch)
        # Asymmetry filter
        left_bins = d // 2
        left = batch[:, :left_bins].sum(axis=1).astype(np.float64)
        left_frac = left / float(S)
        asym_mask = (left_frac > 1 - threshold_a) & (left_frac < threshold_a)
        batch = batch[asym_mask]

        for b in range(len(batch)):
            c_int = batch[b]
            windows = find_all_killing_windows(c_int, d, S, c_target)
            if windows:
                # Take the best window (highest margin)
                ell, s_lo, tv, margin = windows[0]
                pruned_data.append((c_int.copy(), ell, s_lo, margin, windows))
            else:
                n_survived += 1

    if verbose:
        print(f"  Pruned: {len(pruned_data):,}, Survived: {n_survived:,}")
        if n_survived > 0:
            print(f"  WARNING: {n_survived} survivors -- cascade incomplete!")

    if not pruned_data:
        print("  No pruned compositions to analyze.")
        return {}

    # Sort by margin (ascending) to prioritize hard cases
    pruned_data.sort(key=lambda x: x[3])

    # Analyze up to max_analyze compositions (focus on hardest)
    n_analyze = min(len(pruned_data), max_analyze)
    if verbose:
        print(f"\n  Analyzing {n_analyze} compositions "
              f"(sorted by margin, hardest first)...")
        print(f"  {'idx':>5} {'margin':>9} {'cv_curr':>9} {'qc_curr':>9} "
              f"{'net_curr':>9} {'qp_min':>9} {'net_qp':>9} "
              f"{'lp_lb':>9} {'net_lp':>9} {'improve':>8}")
        print(f"  {'-'*90}")

    results = {
        'margins': [],
        'cell_vars': [],
        'quad_corrs': [],
        'nets_current': [],
        'qp_mins': [],
        'nets_qp': [],
        'lp_bounds': [],
        'nets_lp': [],
        'n_windows_tried': [],
        'verification_errors': [],
    }

    n_cert_current = 0
    n_cert_qp = 0
    n_cert_lp = 0
    n_cert_multiwindow_qp = 0
    n_cert_multiwindow_lp = 0

    for idx in range(n_analyze):
        c_int, best_ell, best_s_lo, best_margin, all_windows = pruned_data[idx]

        # --- Current separate bound (best single window) ---
        margin, cell_var, quad_corr, net_current = box_cert_separate(
            c_int, d, S, c_target, best_ell, best_s_lo)

        # Also check all windows for best net_current
        best_net_current = net_current
        best_win_for_current = (best_ell, best_s_lo)
        for ell_w, s_lo_w, tv_w, margin_w in all_windows[:20]:
            m_w, cv_w, qc_w, net_w = box_cert_separate(
                c_int, d, S, c_target, ell_w, s_lo_w)
            if net_w > best_net_current:
                best_net_current = net_w
                best_win_for_current = (ell_w, s_lo_w)
                margin, cell_var, quad_corr = m_w, cv_w, qc_w
        net_current = best_net_current

        # --- Joint QP (actual minimum over cell, best single window) ---
        # Try the best window from the current method
        qp_min, qp_mu, qp_evals = joint_qp_min(
            c_int, d, S, c_target,
            best_win_for_current[0], best_win_for_current[1],
            n_restarts=30)
        net_qp = qp_min - c_target

        # --- McCormick LP (rigorous lower bound, best single window) ---
        lp_bound, lp_status = mccormick_lp_bound(
            c_int, d, S, c_target,
            best_win_for_current[0], best_win_for_current[1])
        net_lp = lp_bound - c_target

        # --- Multi-window: try multiple windows with QP/LP ---
        best_qp_over_windows = qp_min
        best_lp_over_windows = lp_bound
        for ell_w, s_lo_w, tv_w, margin_w in all_windows[:10]:
            qp_w, _, _ = joint_qp_min(
                c_int, d, S, c_target, ell_w, s_lo_w, n_restarts=15)
            lp_w, _ = mccormick_lp_bound(c_int, d, S, c_target, ell_w, s_lo_w)
            if qp_w > best_qp_over_windows:
                best_qp_over_windows = qp_w
            if lp_w > best_lp_over_windows:
                best_lp_over_windows = lp_w

        # --- Verification ---
        # 1. TV formula cross-check
        verify_tv_formula(c_int, d, S, best_win_for_current[0],
                          best_win_for_current[1])

        # 2. QP >= separate bound
        ok, msg = verify_joint_bound_ge_separate(
            margin, cell_var, quad_corr, qp_min, c_target)
        if not ok:
            results['verification_errors'].append(
                f"idx={idx}: {msg}")

        # 3. LP <= QP
        ok, msg = verify_mccormick_le_qp(lp_bound, qp_min)
        if not ok:
            results['verification_errors'].append(
                f"idx={idx}: {msg}")

        # --- Tallies ---
        if net_current >= 0:
            n_cert_current += 1
        if net_qp >= -1e-9:
            n_cert_qp += 1
        if net_lp >= -1e-9:
            n_cert_lp += 1
        if best_qp_over_windows - c_target >= -1e-9:
            n_cert_multiwindow_qp += 1
        if best_lp_over_windows - c_target >= -1e-9:
            n_cert_multiwindow_lp += 1

        improve = (net_qp - net_current) if net_current < 0 else 0.0

        results['margins'].append(margin)
        results['cell_vars'].append(cell_var)
        results['quad_corrs'].append(quad_corr)
        results['nets_current'].append(net_current)
        results['qp_mins'].append(qp_min)
        results['nets_qp'].append(net_qp)
        results['lp_bounds'].append(lp_bound)
        results['nets_lp'].append(net_lp)
        results['n_windows_tried'].append(len(all_windows))

        if verbose and (idx < 20 or idx % max(1, n_analyze // 20) == 0):
            print(f"  {idx:>5} {margin:>9.5f} {cell_var:>9.5f} "
                  f"{quad_corr:>9.5f} {net_current:>+9.5f} "
                  f"{qp_min:>9.5f} {net_qp:>+9.5f} "
                  f"{lp_bound:>9.5f} {net_lp:>+9.5f} "
                  f"{improve:>+8.5f}")

    # --- Summary ---
    if verbose:
        print(f"\n{'='*72}")
        print(f"SUMMARY ({n_analyze} compositions analyzed)")
        print(f"{'='*72}")

        for name, arr in [('margin', results['margins']),
                           ('cell_var', results['cell_vars']),
                           ('quad_corr', results['quad_corrs']),
                           ('net_current', results['nets_current']),
                           ('net_qp', results['nets_qp']),
                           ('net_lp', results['nets_lp'])]:
            a = np.array(arr)
            print(f"  {name:>12}: min={np.min(a):>+9.5f}  "
                  f"p5={np.percentile(a,5):>+9.5f}  "
                  f"med={np.median(a):>+9.5f}  "
                  f"p95={np.percentile(a,95):>+9.5f}  "
                  f"max={np.max(a):>+9.5f}")

        print(f"\n  CERTIFICATION RATES (of {n_analyze} analyzed):")
        print(f"    Current separate bound:  {n_cert_current:>6} "
              f"({100*n_cert_current/n_analyze:.1f}%)")
        print(f"    Joint QP (single win):   {n_cert_qp:>6} "
              f"({100*n_cert_qp/n_analyze:.1f}%)")
        print(f"    McCormick LP (single):   {n_cert_lp:>6} "
              f"({100*n_cert_lp/n_analyze:.1f}%)")
        print(f"    Multi-win QP (top 10):   {n_cert_multiwindow_qp:>6} "
              f"({100*n_cert_multiwindow_qp/n_analyze:.1f}%)")
        print(f"    Multi-win LP (top 10):   {n_cert_multiwindow_lp:>6} "
              f"({100*n_cert_multiwindow_lp/n_analyze:.1f}%)")

        # Improvement statistics
        nets_c = np.array(results['nets_current'])
        nets_q = np.array(results['nets_qp'])
        improvement = nets_q - nets_c
        print(f"\n  QP vs CURRENT improvement:")
        print(f"    mean improvement:  {np.mean(improvement):>+.6f}")
        print(f"    max improvement:   {np.max(improvement):>+.6f}")
        print(f"    For failing cells (net_current < 0):")
        mask_fail = nets_c < 0
        if np.any(mask_fail):
            print(f"      n_failing:         {np.sum(mask_fail)}")
            print(f"      mean improvement:  {np.mean(improvement[mask_fail]):>+.6f}")
            print(f"      cells rescued:     "
                  f"{np.sum(nets_q[mask_fail] >= -1e-9)}")

        if results['verification_errors']:
            print(f"\n  VERIFICATION ERRORS ({len(results['verification_errors'])}):")
            for err in results['verification_errors'][:10]:
                print(f"    {err}")
        else:
            print(f"\n  All verification checks PASSED.")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test joint QP box certification vs separate bounds")
    parser.add_argument('--d', type=int, default=4,
                        help='Dimension (default: 4)')
    parser.add_argument('--S', type=int, default=20,
                        help='Grid resolution (compositions sum to S)')
    parser.add_argument('--c_target', type=float, default=1.30,
                        help='Target lower bound')
    parser.add_argument('--max_analyze', type=int, default=200,
                        help='Max compositions to analyze')
    parser.add_argument('--sweep', action='store_true',
                        help='Run sweep over multiple (d, S) configs')
    args = parser.parse_args()

    if args.sweep:
        configs = [
            (4, 10, 1.25),
            (4, 15, 1.28),
            (4, 20, 1.30),
            (6, 15, 1.25),
            (6, 20, 1.28),
            (6, 20, 1.30),
            (8, 15, 1.25),
            (8, 20, 1.28),
        ]
        for d, S, c in configs:
            n = count_compositions(d, S)
            if n > 5_000_000:
                print(f"\nSkipping d={d}, S={S}: {n:,} compositions (too many)")
                continue
            analyze_compositions(d, S, c, max_analyze=100)
    else:
        analyze_compositions(args.d, args.S, args.c_target,
                             max_analyze=args.max_analyze)


if __name__ == '__main__':
    main()
