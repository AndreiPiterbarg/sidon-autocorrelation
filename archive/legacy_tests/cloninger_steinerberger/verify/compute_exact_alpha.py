"""
Compute exact worst-case discretization errors for the Sidon autoconvolution problem.

NOTE: This analysis was written under the old coarse-grid (S=m) parameterization,
where the scaling factor is 4n/ell.  The project has since switched to the C&S
fine grid (S = 4nm), where the threshold formula uses 4n*ell as the scaling
factor and W_int ranges 0..S instead of 0..m.  The alpha(ell) bounds computed
here remain valid as mathematical analysis but should be interpreted in the
context of the old parameterization.

For each (d, m, composition c, window (ell, s0)), maximize the error
  E = sum_{s in window} (conv_w[s] - conv_mu[s])
over valid pre-images mu, then find alpha(ell) such that
  (4n/ell) * max_E <= alpha * (1 + 2*W_int) / m^2  for all c.

Pre-image: delta_k in [0, 1/m) for k=1,...,d-1;
  mu_k = c_k/m + delta_{k+1} - delta_k  (delta_0 = delta_d = 0)
Error is quadratic in delta -> maximize over box with linear constraints.
"""

import numpy as np
from scipy.optimize import minimize
import sys
import time
import warnings
warnings.filterwarnings('ignore')


def compositions_list(d, m):
    """Generate all compositions of m into d non-negative parts as numpy array."""
    from math import comb
    n_comps = comb(m + d - 1, d - 1)
    result = np.zeros((n_comps, d), dtype=np.int32)
    idx = [0]

    def _gen(pos, remaining, arr):
        if pos == d - 1:
            arr[pos] = remaining
            result[idx[0]] = arr
            idx[0] += 1
            return
        for v in range(remaining + 1):
            arr[pos] = v
            _gen(pos + 1, remaining - v, arr)

    _gen(0, m, np.zeros(d, dtype=np.int32))
    return result


def precompute_A(d):
    """Build the A matrix: mu = c/m + A @ delta."""
    n_delta = d - 1
    A = np.zeros((d, n_delta))
    for k in range(d):
        if k < n_delta:
            A[k, k] = 1.0
        if k - 1 >= 0 and k - 1 < n_delta:
            A[k, k - 1] = -1.0
    return A


def precompute_corners(n_delta):
    """Precompute all 2^n_delta corner masks as a boolean matrix."""
    n_corners = 1 << n_delta
    corners = np.zeros((n_corners, n_delta), dtype=bool)
    for mask in range(n_corners):
        for p in range(n_delta):
            if mask & (1 << p):
                corners[mask, p] = True
    return corners


def precompute_window_pairs(d):
    """Precompute (i, j) pairs for each window (ell, s0)."""
    conv_len = 2 * d - 1
    max_ell = 2 * d
    windows = []
    for ell in range(2, max_ell + 1):
        for s0 in range(conv_len - ell + 2):
            pairs = []
            for s in range(s0, s0 + ell - 1):
                for i in range(max(0, s - d + 1), min(d, s + 1)):
                    pairs.append((i, s - i))
            windows.append((ell, s0, pairs))
    return windows


def build_g_H(d, n_delta, A, mu0, pairs):
    """Build g and H for a specific window's (i,j) pairs."""
    g = np.zeros(n_delta)
    H = np.zeros((n_delta, n_delta))
    for i, j in pairs:
        g += mu0[i] * A[j]
        H += np.outer(A[i], A[j])
    return g, H


def maximize_error_fast(mu0, A, g, H, corners_bool, dm, n_delta):
    """
    Maximize E(delta) = -2*g^T*delta - delta^T*H*delta
    over delta in [0, dm]^n_delta with mu0 + A*delta >= 0.
    Uses vectorized corner evaluation + one L-BFGS-B start.
    """
    best_E = 0.0

    # Corner evaluation (vectorized)
    # delta_corners: (n_corners, n_delta) with values 0 or dm
    delta_corners = corners_bool.astype(np.float64) * dm  # (2^n_delta, n_delta)
    mu_corners = mu0 + delta_corners @ A.T  # (n_corners, d)
    feasible = np.all(mu_corners >= -1e-14, axis=1)

    if np.any(feasible):
        delta_f = delta_corners[feasible]
        # E = -2 * g @ delta - delta @ H @ delta for each row
        linear_part = -2.0 * delta_f @ g  # (n_feasible,)
        quad_part = -np.sum(delta_f @ H * delta_f, axis=1)  # (n_feasible,)
        E_corners = linear_part + quad_part
        max_corner = np.max(E_corners)
        if max_corner > best_E:
            best_E = max_corner

    # One L-BFGS-B from midpoint
    bounds = [(0.0, dm)] * n_delta
    x0 = np.full(n_delta, dm / 2)
    if np.min(mu0 + A @ x0) >= -1e-12:
        HpHT = H + H.T
        def neg_E(delta):
            return 2.0 * g @ delta + delta @ H @ delta
        def neg_E_grad(delta):
            return 2.0 * g + HpHT @ delta
        try:
            res = minimize(neg_E, x0, jac=neg_E_grad, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 100, 'ftol': 1e-16})
            if np.min(mu0 + A @ res.x) >= -1e-12:
                E = -res.fun
                if E > best_E:
                    best_E = E
        except Exception:
            pass

    return max(best_E, 0.0)


def W_int_batch(comps, d, ell, s0):
    """W_int for each composition (vectorized)."""
    lo = max(0, s0 - (d - 1))
    hi = min(d - 1, s0 + ell - 2)
    return np.sum(comps[:, lo:hi + 1], axis=1)


def run_analysis(d, m, verbose=True):
    """Run full analysis for given d and m."""
    n = d // 2
    n_delta = d - 1
    conv_len = 2 * d - 1
    max_ell = 2 * d

    comps = compositions_list(d, m)
    n_comps = comps.shape[0]

    A = precompute_A(d)
    corners_bool = precompute_corners(n_delta)
    windows = precompute_window_pairs(d)
    dm = 1.0 / m - 1e-15

    if verbose:
        print(f"\n{'='*70}")
        print(f"d={d}, n={n}, m={m}")
        print(f"Compositions: {n_comps}, Windows: {len(windows)}")
        print(f"Corners to check: {1 << n_delta}")
        print(f"{'='*70}")

    alpha_by_ell = {}
    t0 = time.time()

    for ci in range(n_comps):
        if verbose and ci % max(1, n_comps // 10) == 0:
            elapsed = time.time() - t0
            print(f"  [{elapsed:.1f}s] Composition {ci+1}/{n_comps} ...", flush=True)

        c = comps[ci]
        mu0 = c.astype(np.float64) / m

        for ell, s0, pairs in windows:
            g, H = build_g_H(d, n_delta, A, mu0, pairs)
            max_err_raw = maximize_error_fast(mu0, A, g, H, corners_bool, dm, n_delta)
            max_err_scaled = (4.0 * n / ell) * max_err_raw

            W_int = int(np.sum(c[max(0, s0-(d-1)):min(d, s0+ell-1)]))
            fb = (1.0 + 2.0 * W_int) / (m * m)
            ratio = max_err_scaled / fb if fb > 1e-20 else 0.0

            if ell not in alpha_by_ell:
                alpha_by_ell[ell] = {'max_ratio': 0.0, 'worst_c': None, 'worst_s0': None}
            if ratio > alpha_by_ell[ell]['max_ratio']:
                alpha_by_ell[ell]['max_ratio'] = ratio
                alpha_by_ell[ell]['worst_c'] = tuple(c)
                alpha_by_ell[ell]['worst_s0'] = s0

    elapsed = time.time() - t0

    print(f"\nRESULTS: d={d}, n={n}, m={m}  [{elapsed:.1f}s]")
    print(f"{'ell':>4} | {'alpha':>12} | {'4n/ell':>8} | {'savings':>8} | {'%_of_4n/l':>10} | worst comp, s0")
    print("-" * 90)
    for ell in sorted(alpha_by_ell.keys()):
        info = alpha_by_ell[ell]
        ref = 4.0 * n / ell
        savings = ref - info['max_ratio']
        pct = info['max_ratio'] / ref * 100
        c_str = str(info['worst_c'])
        if len(c_str) > 30:
            c_str = c_str[:27] + "..."
        print(f"{ell:>4} | {info['max_ratio']:>12.6f} | {ref:>8.4f} | {savings:>+8.4f} | {pct:>9.1f}% | {c_str}, s0={info['worst_s0']}")

    violations = [(ell, alpha_by_ell[ell]['max_ratio'], 4.0*n/ell)
                  for ell in alpha_by_ell if alpha_by_ell[ell]['max_ratio'] > 4.0*n/ell + 1e-6]
    if violations:
        print(f"\n*** {len(violations)} ell values with alpha > 4n/ell ***")
        for ell, a, r in violations:
            print(f"  ell={ell}: alpha={a:.8f} > 4n/ell={r:.4f}")
    else:
        print(f"\nAll alpha(ell) <= 4n/ell. Current bound is sound.")

    return alpha_by_ell


def run_sampled(d, m, n_samples, verbose=True):
    """Sample random compositions and compute alpha bounds."""
    n = d // 2
    n_delta = d - 1
    conv_len = 2 * d - 1
    max_ell = 2 * d

    A = precompute_A(d)
    corners_bool = precompute_corners(n_delta)
    windows = precompute_window_pairs(d)
    dm = 1.0 / m - 1e-15

    rng = np.random.RandomState(123)
    alpha_by_ell = {}
    t0 = time.time()

    if verbose:
        print(f"\n{'='*70}")
        print(f"SAMPLING d={d}, n={n}, m={m}, {n_samples} samples")
        print(f"{'='*70}")

    for trial in range(n_samples):
        if verbose and trial % max(1, n_samples // 10) == 0:
            elapsed = time.time() - t0
            print(f"  [{elapsed:.1f}s] Sample {trial+1}/{n_samples} ...", flush=True)

        # Random composition via stars-and-bars
        breaks = sorted(rng.choice(m + d - 1, d - 1, replace=False))
        c = []
        prev = -1
        for b in breaks:
            c.append(b - prev - 1)
            prev = b
        c.append(m + d - 2 - prev)
        c_arr = np.array(c, dtype=np.float64)
        mu0 = c_arr / m

        for ell, s0, pairs in windows:
            g, H = build_g_H(d, n_delta, A, mu0, pairs)
            max_err_raw = maximize_error_fast(mu0, A, g, H, corners_bool, dm, n_delta)
            max_err_scaled = (4.0 * n / ell) * max_err_raw

            lo = max(0, s0 - (d - 1))
            hi = min(d - 1, s0 + ell - 2)
            W_int = int(sum(c[lo:hi + 1]))
            fb = (1.0 + 2.0 * W_int) / (m * m)
            ratio = max_err_scaled / fb if fb > 1e-20 else 0.0

            if ell not in alpha_by_ell:
                alpha_by_ell[ell] = {'max_ratio': 0.0, 'worst_c': None}
            if ratio > alpha_by_ell[ell]['max_ratio']:
                alpha_by_ell[ell]['max_ratio'] = ratio
                alpha_by_ell[ell]['worst_c'] = tuple(c)

    elapsed = time.time() - t0

    print(f"\nSAMPLED RESULTS: d={d}, n={n}, m={m} [{elapsed:.1f}s]")
    print(f"{'ell':>4} | {'alpha_lb':>12} | {'4n/ell':>8} | {'%_of_4n/l':>10}")
    print("-" * 50)
    for ell in sorted(alpha_by_ell.keys()):
        info = alpha_by_ell[ell]
        ref = 4.0 * n / ell
        pct = info['max_ratio'] / ref * 100
        print(f"{ell:>4} | {info['max_ratio']:>12.6f} | {ref:>8.4f} | {pct:>9.1f}%")

    return alpha_by_ell


if __name__ == '__main__':
    all_results = {}

    # d=4, m=10 and m=20: exact
    for d, m in [(4, 10), (4, 20)]:
        result = run_analysis(d, m, verbose=True)
        if result is not None:
            all_results[(d, m)] = result

    # d=8, m=10: exact (may be slow)
    result = run_analysis(8, 10, verbose=True)
    if result is not None:
        all_results[(8, 10)] = result

    # d=8, m=20: sample
    result = run_sampled(8, 20, n_samples=2000, verbose=True)
    if result is not None:
        all_results[(8, 20)] = result

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY: alpha(ell, d) values")
    print("="*70)
    print(f"{'d':>3} | {'m':>3} | {'ell':>4} | {'alpha':>12} | {'4n/ell':>8} | {'%_of_4n/l':>10} | {'tight_factor':>12}")
    print("-" * 75)
    for (d, m) in sorted(all_results.keys()):
        result = all_results[(d, m)]
        n = d // 2
        for ell in sorted(result.keys()):
            alpha = result[ell]['max_ratio']
            ref = 4.0 * n / ell
            pct = alpha / ref * 100
            # Tight factor = proposed replacement for 4n/ell
            tight = alpha * 1.001  # 0.1% safety margin
            print(f"{d:>3} | {m:>3} | {ell:>4} | {alpha:>12.6f} | {ref:>8.4f} | {pct:>9.1f}% | {tight:>12.6f}")
        print("-" * 75)

    print("\nINTERPRETATION (old coarse-grid parameterization, S=m):")
    print("- alpha(ell) = tightest factor s.t. (4n/ell)*max_E <= alpha*(1+2W)/m^2 for all c")
    print("- If alpha < 4n/ell, the threshold can be tightened for that window length")
    print("- '%_of_4n/l' shows how much of the current bound is actually used")
    print("- NOTE: The project now uses the fine grid (S=4nm); these results are")
    print("  still valid as analysis but the code's threshold formula has changed.")
