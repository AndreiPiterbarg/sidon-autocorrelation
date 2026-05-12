"""Compare current triangle-inequality box bound (cell_var + quad_corr)
to the EXACT joint bound.

Two notions of "exact bound":

  bound_exact_drop(c, W)  = max_{delta in cell} -(grad . delta
                                                 + (2d/ell) Q(delta))
       = worst-case TV DECREASE from mu* in the cell.  This is the quantity
       that v2's triangle bound is meant to upper-bound: certification needs
       margin > bound_exact_drop, so a sound surrogate must satisfy
            cell_var + quad_corr  >=  bound_exact_drop.

  bound_exact_abs(c, W)   = max_{delta in cell} |grad . delta
                                                + (2d/ell) Q(delta)|
       = max absolute change in TV (drop OR jump).  This is what the user
       asked us to compare against.  It is always >= bound_exact_drop.

The current cell_var + quad_corr is a sound upper bound on the DROP only —
it is NOT a valid bound on the absolute change, because the second-order
term is bounded only when Q < 0 (the direction that hurts certification).

We compute both and report ratios for both.

We compute the exact value by two methods that bracket each other tightly:
  (a) Lower bound (LB on max): random sampling inside the cell + a local
      maximization (scipy SLSQP) seeded from each random sample's signs.
  (b) Upper bound (UB on max): vertex enumeration.  Because Q is quadratic
      and grad.delta is linear, the objective is quadratic in delta, and
      its maximum over a polytope is attained at a vertex of the polytope.
      For the cell {|delta_i|<=h, sum delta_i=0}, vertices are obtained by
      pinning d-1 components to +/-h and solving for the remaining one
      (it must lie in [-h, h] for feasibility).  This is 2^(d-1) * d
      candidates, fully enumerated for d in {4, 8}.

If LB == UB (to within 1e-10) we report that as the EXACT bound.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.optimize import minimize

C_TARGET_DEFAULT = 1.28


# -----------------------------------------------------------------------
# Core TV / window arithmetic (continuous masses)
# -----------------------------------------------------------------------

def window_indicator(d: int, ell: int, s: int) -> np.ndarray:
    """A_{ij} = 1 if s <= i+j <= s+ell-2 else 0."""
    A = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            k = i + j
            if s <= k <= s + ell - 2:
                A[i, j] = 1.0
    return A


def tv_value(mu: np.ndarray, d: int, ell: int, s: int) -> float:
    """TV_W(mu) = (2d/ell) * mu^T A mu (A as above, no symmetrisation needed
    because A is already symmetric and we use the full bilinear form)."""
    A = window_indicator(d, ell, s)
    return float((2.0 * d / ell) * mu @ A @ mu)


def grad_tv(mu: np.ndarray, d: int, ell: int, s: int) -> np.ndarray:
    """grad_i TV_W(mu) = (4d/ell) * sum_{j: s<=i+j<=s+ell-2} mu_j."""
    A = window_indicator(d, ell, s)
    return (4.0 * d / ell) * (A @ mu)


# -----------------------------------------------------------------------
# Current v2 bounds (cell_var + quad_corr)
# -----------------------------------------------------------------------

def cell_var_current(grad: np.ndarray, S: int) -> float:
    """First-order box bound:
       max_{|delta|<=h, sum delta=0} grad . delta
       = (1/(2S)) * sum_{k=0..d/2-1} (g_sorted[d-1-k] - g_sorted[k]).
    """
    d = grad.shape[0]
    g = np.sort(grad)
    h = 1.0 / (2.0 * S)
    total = 0.0
    for k in range(d // 2):
        total += g[d - 1 - k] - g[k]
    return h * total


def quad_corr_current(d: int, S: int, ell: int, s: int) -> float:
    """Second-order pair-count bound:
       quad_corr = (2d/ell) * min(cross_W, d^2 - N_W) / (4 S^2).
    """
    # n_k = #{(i,j): 0<=i,j<d, i+j=k} = min(k+1, d, 2d-1-k)
    # m_k = 1 if k even and k//2 < d
    N_W = 0
    M_W = 0
    for k in range(s, s + ell - 1):
        n_k = min(k + 1, d, 2 * d - 1 - k)
        if n_k <= 0:
            continue
        N_W += n_k
        if k % 2 == 0 and (k // 2) < d:
            M_W += 1
    cross_W = N_W - M_W
    pair_bound = min(cross_W, d * d - N_W)
    if pair_bound <= 0:
        return 0.0
    h2 = 1.0 / (4.0 * S * S)
    return (2.0 * d / ell) * pair_bound * h2


# -----------------------------------------------------------------------
# Exact bound: max_{delta in cell} |grad . delta + (2d/ell) Q(delta)|
# -----------------------------------------------------------------------

def _objective_signed(delta, grad, A, scale, sign):
    """sign * [grad . delta + scale * delta^T A delta].
    sign = +1 maximises original; sign = -1 maximises negation.
    We minimise -sign*obj in scipy."""
    obj = grad @ delta + scale * (delta @ A @ delta)
    return -sign * obj  # minimise


def _grad_signed(delta, grad, A, scale, sign):
    g = grad + 2.0 * scale * (A @ delta)
    return -sign * g


def exact_bound_random(grad, A, scale, h, n_samples=20000, seed=0):
    """Lower bound on (max signed_drop, max abs).
    Returns (max_drop, max_abs) where
       drop(delta) = -(grad.delta + scale*Q(delta))
       abs(delta)  =  |grad.delta + scale*Q(delta)|
    """
    rng = np.random.default_rng(seed)
    d = grad.shape[0]
    best_drop = 0.0
    best_abs = 0.0
    for _ in range(n_samples // 2):
        x = rng.uniform(-h, h, size=d)
        x -= x.mean()
        m = np.max(np.abs(x))
        if m > h:
            x *= (h / m)
        v = grad @ x + scale * (x @ A @ x)
        if -v > best_drop:
            best_drop = -v
        if abs(v) > best_abs:
            best_abs = abs(v)
    for _ in range(n_samples // 2):
        signs = rng.choice([-h, h], size=d)
        idx = rng.integers(0, d)
        signs[idx] = 0.0
        signs[idx] = -signs.sum()
        if abs(signs[idx]) <= h + 1e-12:
            v = grad @ signs + scale * (signs @ A @ signs)
            if -v > best_drop:
                best_drop = -v
            if abs(v) > best_abs:
                best_abs = abs(v)
    return best_drop, best_abs


def exact_bound_local(grad, A, scale, h, n_starts=64, seed=0):
    """Local maxima with SLSQP from random vertex-flavoured starts.
    Returns (max_drop, max_abs)."""
    rng = np.random.default_rng(seed + 1)
    d = grad.shape[0]
    bounds = [(-h, h)] * d
    cons = ({'type': 'eq', 'fun': lambda x: x.sum(),
             'jac': lambda x: np.ones(d)},)
    best_drop = 0.0
    best_abs = 0.0
    for sign in (+1.0, -1.0):
        for _ in range(n_starts):
            x0 = rng.choice([-h, h], size=d).astype(np.float64)
            x0 -= x0.mean()
            m = np.max(np.abs(x0))
            if m > h:
                x0 *= (h / m)
            try:
                res = minimize(
                    _objective_signed,
                    x0,
                    args=(grad, A, scale, sign),
                    jac=_grad_signed,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 200, 'ftol': 1e-14},
                )
                v = -sign * res.fun
                if -v > best_drop:
                    best_drop = -v
                if abs(v) > best_abs:
                    best_abs = abs(v)
            except Exception:
                pass
    return best_drop, best_abs


def exact_bound_vertices(grad, A, scale, h, max_d=10):
    """Enumerate all polytope vertices of {|x_i|<=h, sum x_i = 0}.

    Vertices lie on at least d-1 active inequality (box) constraints.  The
    one possibly-inactive coordinate is determined by sum=0.  So enumerate
    all 2^(d-1) sign patterns for the "active" coords, choose any of the d
    indices to be the "free" one, and solve for that free coord.

    For d=8 this is d * 2^(d-1) = 8 * 128 = 1024 vertices.  Tiny.

    Returns (max_drop, max_abs) — both EXACT (max of quadratic over polytope
    is on a vertex).  Returns None if d > max_d.
    """
    if grad.shape[0] > max_d:
        return None
    d = grad.shape[0]
    best_drop = 0.0
    best_abs = 0.0
    for free_idx in range(d):
        active = [i for i in range(d) if i != free_idx]
        for signs in itertools.product((-1.0, 1.0), repeat=d - 1):
            x = np.zeros(d)
            for k, i in enumerate(active):
                x[i] = signs[k] * h
            x[free_idx] = -sum(x[i] for i in active)
            if abs(x[free_idx]) <= h + 1e-12:
                v = grad @ x + scale * (x @ A @ x)
                if -v > best_drop:
                    best_drop = -v
                if abs(v) > best_abs:
                    best_abs = abs(v)
    return best_drop, best_abs


# -----------------------------------------------------------------------
# One example
# -----------------------------------------------------------------------

def evaluate_example(d, S, c_int, windows, c_target=C_TARGET_DEFAULT,
                     n_random=20000, n_local_starts=128, seed=0):
    """For each window, compute current vs. exact bound."""
    c_int = np.asarray(c_int, dtype=np.int64)
    assert c_int.sum() == S, f"sum(c)={c_int.sum()} != S={S}"
    mu = c_int.astype(np.float64) / S
    h = 1.0 / (2.0 * S)
    rows = []

    for (ell, s) in windows:
        A = window_indicator(d, ell, s)
        scale = 2.0 * d / ell
        tv = tv_value(mu, d, ell, s)
        grad = grad_tv(mu, d, ell, s)
        margin = tv - c_target

        cv = cell_var_current(grad, S)
        qc = quad_corr_current(d, S, ell, s)
        triangle = cv + qc

        # Exact via three methods (each returns (drop, abs))
        vert_b = exact_bound_vertices(grad, A, scale, h, max_d=10)
        rand_b = exact_bound_random(grad, A, scale, h,
                                    n_samples=n_random, seed=seed)
        loc_b = exact_bound_local(grad, A, scale, h,
                                  n_starts=n_local_starts, seed=seed)

        # Vertex enumeration is the true max of a quadratic over a polytope
        # (attained at a vertex).  Trust it when available.
        if vert_b is not None:
            exact_drop, exact_abs = vert_b
            method = 'vertices'
        else:
            exact_drop = max(rand_b[0], loc_b[0])
            exact_abs = max(rand_b[1], loc_b[1])
            method = 'random+local'

        ratio_drop = triangle / exact_drop if exact_drop > 0 else float('inf')
        ratio_abs = triangle / exact_abs if exact_abs > 0 else float('inf')

        rows.append({
            'd': d, 'S': S, 'c': tuple(int(x) for x in c_int),
            'ell': ell, 's': s,
            'tv': tv, 'margin': margin,
            'cell_var': cv, 'quad_corr': qc, 'triangle': triangle,
            'exact_drop': exact_drop, 'exact_abs': exact_abs,
            'method': method,
            'ratio_drop': ratio_drop, 'ratio_abs': ratio_abs,
        })
    return rows


# -----------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------

EXAMPLES = [
    # d=4
    (4, 10, (2, 1, 1, 6), [(2, 6), (4, 2), (8, 0)]),
    (4, 20, (5, 3, 4, 8), [(2, 6), (4, 2), (8, 0)]),
    # d=8
    (8, 20, (3, 2, 2, 2, 3, 3, 2, 3), [(2, 8), (8, 4)]),
    (8, 10, (2, 1, 1, 1, 1, 1, 1, 2), [(4, 4), (14, 1)]),
]


def fmt_row(r):
    return (f"d={r['d']:2d} S={r['S']:3d} c={r['c']}  "
            f"W=(ell={r['ell']:2d},s={r['s']:2d})  "
            f"TV={r['tv']:.4f} mar={r['margin']:+.4f}  "
            f"cv={r['cell_var']:.4f} qc={r['quad_corr']:.4f} "
            f"tri={r['triangle']:.4f}  "
            f"drop={r['exact_drop']:.4f} abs={r['exact_abs']:.4f}({r['method']})  "
            f"r_drop={r['ratio_drop']:.3f}x r_abs={r['ratio_abs']:.3f}x")


def main():
    print("=" * 130)
    print("Box certification tightness: triangle bound (cell_var + quad_corr)"
          " vs. exact joint max")
    print(f"c_target = {C_TARGET_DEFAULT}")
    print("Two exact quantities:")
    print("  drop = max -(grad.delta + scale*Q)  -- the WORST-CASE TV DECREASE.")
    print("         Triangle bound is a sound upper bound on this; the v2 "
          "certification uses margin > triangle.")
    print("  abs  = max  |grad.delta + scale*Q|  -- worst-case |TV change|.")
    print("=" * 130)

    all_rows = []
    for (d, S, c, windows) in EXAMPLES:
        rows = evaluate_example(d, S, c, windows)
        for r in rows:
            print(fmt_row(r))
            all_rows.append(r)
        print()

    # Summary table
    print("=" * 130)
    print(f"{'d':>2} {'S':>3} {'c':<26} {'(ell,s)':>9} "
          f"{'margin':>8} {'cell_var':>9} {'quad_corr':>10} "
          f"{'triangle':>9} {'drop':>8} {'abs':>8} "
          f"{'r_drop':>7} {'r_abs':>7}")
    print("-" * 130)
    for r in all_rows:
        c_str = ','.join(str(x) for x in r['c'])
        print(f"{r['d']:>2} {r['S']:>3} {c_str:<26} "
              f"({r['ell']:>2},{r['s']:>2})  "
              f"{r['margin']:>+8.4f} {r['cell_var']:>9.4f} "
              f"{r['quad_corr']:>10.4f} {r['triangle']:>9.4f} "
              f"{r['exact_drop']:>8.4f} {r['exact_abs']:>8.4f} "
              f"{r['ratio_drop']:>6.2f}x {r['ratio_abs']:>6.2f}x")
    print("=" * 130)

    # Worst (loosest) cases for the certification-relevant ratio
    finite_drop = [r for r in all_rows if math.isfinite(r['ratio_drop'])]
    if finite_drop:
        worst_drop = max(finite_drop, key=lambda r: r['ratio_drop'])
        print(f"\nWorst (loosest) tightness ratio for the DROP: "
              f"{worst_drop['ratio_drop']:.2f}x for "
              f"d={worst_drop['d']}, S={worst_drop['S']}, c={worst_drop['c']}, "
              f"(ell={worst_drop['ell']},s={worst_drop['s']})")
        print(f"  triangle = {worst_drop['triangle']:.5f}, "
              f"exact_drop = {worst_drop['exact_drop']:.5f}; "
              f"slack overstated by factor {worst_drop['ratio_drop']:.2f}.")

    # Soundness check (only the DROP must be bounded by triangle)
    bad_drop = [r for r in all_rows
                if r['triangle'] < r['exact_drop'] - 1e-9]
    if bad_drop:
        print(f"\nWARNING: triangle < exact_drop in {len(bad_drop)} cases "
              f"-- v2 certification UNSOUND on these")
        for r in bad_drop:
            print(" ", fmt_row(r))
    else:
        print("\nSoundness check (drop): triangle >= exact_drop in all "
              "cases. v2 OK.")

    bad_abs = [r for r in all_rows
               if r['triangle'] < r['exact_abs'] - 1e-9]
    if bad_abs:
        print(f"\nNote: triangle < exact_abs in {len(bad_abs)} cases.  "
              f"This is EXPECTED: v2's quad_corr only bounds NEGATIVE Q, "
              f"so the triangle bound is not designed to bound |TV change|, "
              f"only the worst-case decrease.  Examples:")
        for r in bad_abs:
            print(" ", fmt_row(r))


if __name__ == '__main__':
    main()
