"""Refined Cloninger-Steinerberger evaluator at high resolution.

Reference: Cloninger & Steinerberger, "On Suprema of Autoconvolutions
with an Application to Sidon Sets", arXiv:1403.7988v3.

Lemma 1 (CS): Let
    A_n = {a in (R+)^{2n} : sum_i a_i = 4n}
    a_n = min_{a in A_n} max_{2 <= ell <= 2n} max_{-n <= k <= n-ell}
              (1/(4n*ell)) sum_{k <= i+j <= k+ell-2} a_i a_j.
Then c = 2/sigma^2 >= a_n.

Lemma 3 (CS): With B_{n,m} the (1/m)-discretization,
    c >= b_{n,m} - 2/m - 1/m^2.

----------------------------------------------------------------------
SCOPE OF THIS MODULE
----------------------------------------------------------------------

The CS optimization is a *non-convex QP* on the simplex (each window is
a positive-semidefinite quadratic, but we minimize the maximum of them).
There is no LP / convex relaxation that yields a rigorous lower bound on
a_n^*: any tractable convex relaxation would give an upper bound.

This module therefore offers TWO complementary capabilities:

  1. A fast HEURISTIC upper bound on a_n^* at arbitrary n, using a
     projected-subgradient descent on the simplex.  This estimates
     "what is a_n^* really equal to?" — informational only, NOT a
     rigorous LB on c.  Useful for diagnosing whether the CS scheme
     is even capable of producing rigorous bounds above 1.2802 in
     principle.

  2. A wrapper around the existing rigorous branch-and-prune
     (`solvers.find_best_bound_direct`) for moderate (n, m) — this
     IS a rigorous LB on c via Lemma 3.  Costs grow exponentially in
     d = 2n, so realistic n is small (n_half <= 4 in this repo's
     hardware budget).

The combination tells us:
   - whether a_n^* > 1.2802 is plausible at any tractable n (heuristic);
   - what the certified LB at that (n,m) actually is (rigorous);
   - the gap between the two, indicating how much a_n^* improves with
     n vs how much room is left to certify.

----------------------------------------------------------------------
USAGE
----------------------------------------------------------------------

    python cs_refined_lp.py --heuristic --n 8 16 32 64 128 256 \
                             --restarts 256 --iters 5000

    python cs_refined_lp.py --rigorous --n_half 2 3 --m 20 40

The script prints both the heuristic value of a_n (a *plausibly tight*
estimate) and, where requested, the rigorous certified LB.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------
# CS test value: max_{ell, k} (1/(4n*ell)) sum_{k <= i+j <= k+ell-2} a_i a_j
# ---------------------------------------------------------------------

def cs_test_value(a: np.ndarray) -> float:
    """Compute max-window quadratic for vector a in A_n (length 2n).

    Returns max_{2<=ell<=4n} max_{0<=s_lo<=4n-ell} (1/(4n*ell))
            * sum_{s_lo <= i+j <= s_lo+ell-2 (shifted)} a_i a_j.

    This corresponds to CS's *proof* range (Lemma 1 proof, page 4):
    "Let 2 <= ell <= 4n and -2n <= k <= 2n - ell."  After the index
    shift k -> K = k + 2n, this is K in [0, 4n-ell], i.e. all windows
    of length ell-1 fitting in the conv range.

    The Lemma statement uses a narrower k range, but a delta at a corner
    of the simplex would make the narrow-range objective zero (vacuous).
    The proof actually establishes the wider-range bound; the existing
    repo's `compute_test_value_single` uses this wider range too.

    Conv positions s = I+J in [0, 4n-2] where I,J = i+n, j+n.
    """
    a = np.asarray(a, dtype=np.float64)
    two_n = a.shape[0]
    assert two_n % 2 == 0, "length must be 2n"
    n = two_n // 2

    conv = np.convolve(a, a)
    cumconv = np.concatenate([[0.0], np.cumsum(conv)])
    conv_len = conv.shape[0]  # = 4n - 1

    best = 0.0
    # ell in [2, 4n]; s_lo in [0, conv_len - (ell-1)]
    for ell in range(2, 4 * n + 1):
        n_cv = ell - 1
        if n_cv > conv_len:
            break
        inv_norm = 1.0 / (4.0 * n * ell)
        for s_lo in range(conv_len - n_cv + 1):
            s_hi = s_lo + n_cv - 1
            ws = cumconv[s_hi + 1] - cumconv[s_lo]
            tv = ws * inv_norm
            if tv > best:
                best = tv
    return best


def cs_test_value_full_range(a: np.ndarray) -> float:
    """Variant that ranges k over the full window-feasible domain
    (consistent with the existing repo's `compute_test_value_single`).

    The existing repo uses k in [0, conv_len - n_cv], which is a wider
    range than CS's [-n, n-ell].  Both give upper bounds on f*f's L^inf,
    but only CS's narrower range corresponds to the support [-1/2, 1/2].
    Returning both lets us audit the conventions.
    """
    a = np.asarray(a, dtype=np.float64)
    two_n = a.shape[0]
    n = two_n // 2
    conv = np.convolve(a, a)
    cumconv = np.concatenate([[0.0], np.cumsum(conv)])
    conv_len = conv.shape[0]
    best = 0.0
    for ell in range(2, two_n + 1):
        n_cv = ell - 1
        inv_norm = 1.0 / (4.0 * n * ell)
        for s_lo in range(0, conv_len - n_cv + 1):
            s_hi = s_lo + n_cv - 1
            ws = cumconv[s_hi + 1] - cumconv[s_lo]
            tv = ws * inv_norm
            if tv > best:
                best = tv
    return best


# ---------------------------------------------------------------------
# Heuristic: projected-subgradient descent on the simplex {sum a = 4n, a >= 0}
# ---------------------------------------------------------------------

def project_simplex(v: np.ndarray, total: float) -> np.ndarray:
    """Project v onto {x >= 0, sum x = total} (Euclidean)."""
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - total
    rho = np.argmax(u - cssv / (np.arange(n) + 1) <= 0) - 1
    if rho < 0:
        rho = n - 1
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)


def windows_iter(n: int):
    """Yield (ell, s_lo, s_hi, inv_norm) over the wider proof-range
    used by CS's proof and the existing repo's `compute_test_value_single`.

    ell in [2, 4n]; s_lo in [0, 4n-ell].  s_hi = s_lo + ell - 2.
    """
    two_n = 2 * n
    conv_len = 2 * two_n - 1  # = 4n - 1
    for ell in range(2, 4 * n + 1):
        n_cv = ell - 1
        if n_cv > conv_len:
            break
        inv_norm = 1.0 / (4.0 * n * ell)
        for s_lo in range(conv_len - n_cv + 1):
            s_hi = s_lo + n_cv - 1
            yield ell, s_lo, s_hi, inv_norm


def precompute_window_indices(n: int):
    """Return an array of shape (W, 4) with rows (ell, s_lo, s_hi, inv_norm)
    for the W active CS windows."""
    rows = []
    for ell, s_lo, s_hi, inv_norm in windows_iter(n):
        rows.append((ell, s_lo, s_hi, inv_norm))
    return np.array(rows, dtype=object)


def windowed_quadratics(a: np.ndarray, n: int):
    """For each CS window W=(ell, s_lo, s_hi), compute
        Q_W(a) = (1/(4n*ell)) sum_{s_lo <= I+J <= s_hi} a_I a_J.
    Returns array of shape (W,) and the corresponding (ell, s_lo, s_hi, inv_norm).
    """
    conv = np.convolve(a, a)
    cumconv = np.concatenate([[0.0], np.cumsum(conv)])
    rows = list(windows_iter(n))
    vals = np.empty(len(rows))
    for idx, (ell, s_lo, s_hi, inv_norm) in enumerate(rows):
        ws = cumconv[s_hi + 1] - cumconv[s_lo]
        vals[idx] = ws * inv_norm
    return vals, rows


def window_gradient(a: np.ndarray, ell: int, s_lo: int, s_hi: int,
                     inv_norm: float) -> np.ndarray:
    """Gradient of Q_W(a) = (1/(4n*ell)) sum_{s_lo<=I+J<=s_hi} a_I a_J w.r.t. a.

    grad_k Q_W(a) = (1/(4n*ell)) * 2 * sum_{J : s_lo <= k+J <= s_hi} a_J
                  = (2 * inv_norm) * sum_{J in [max(0, s_lo-k), min(2n-1, s_hi-k)]} a[J]
    """
    two_n = a.shape[0]
    g = np.zeros(two_n)
    for k in range(two_n):
        j_lo = max(0, s_lo - k)
        j_hi = min(two_n - 1, s_hi - k)
        if j_lo <= j_hi:
            g[k] = 2.0 * inv_norm * np.sum(a[j_lo:j_hi + 1])
    return g


def heuristic_an_star(n: int, n_restarts: int = 64, n_iters: int = 2000,
                       seed: int | None = None,
                       step_init: float = 0.5,
                       softmax_beta: float = 50.0,
                       tol: float = 1e-9,
                       verbose: bool = False) -> Tuple[float, np.ndarray]:
    """Heuristic upper bound on a_n^* via projected smoothed-max descent.

    Smoothing: replace max_W Q_W(a) by softmax_W Q_W(a) with growing beta.
    Subgradient: for the smoothed objective, gradient is a convex combo of
    window gradients.  Project onto simplex {sum a = 4n, a >= 0}.

    Returns (best a_n^* upper bound, best a vector found).

    NOTE: this is a HEURISTIC; the value returned is an *upper* bound on
    the true min, since any feasible a witnesses min <= max_W Q_W(a).
    Therefore it canNOT be used as a rigorous LB on c.  Its purpose is
    to estimate where a_n^* sits at large n.
    """
    rng = np.random.default_rng(seed)
    two_n = 2 * n
    total = float(4 * n)

    rows = list(windows_iter(n))
    n_win = len(rows)
    # Precompute window structure for gradient computation
    ells = np.array([r[0] for r in rows], dtype=np.int64)
    s_los = np.array([r[1] for r in rows], dtype=np.int64)
    s_his = np.array([r[2] for r in rows], dtype=np.int64)
    inv_norms = np.array([r[3] for r in rows], dtype=np.float64)

    best_obj = np.inf
    best_a = None

    for r in range(n_restarts):
        # Initialize: mix uniform + perturbed + corner-loaded starts
        if r == 0:
            a = np.full(two_n, total / two_n)
        elif r == 1:
            # Symmetric u-shape
            x = np.linspace(-1, 1, two_n)
            a = 1.0 + 0.5 * x * x
            a *= total / a.sum()
        elif r == 2:
            # B-spline-like centered (Matolcsi-Vinuesa-style optimizer)
            x = np.linspace(-1, 1, two_n)
            a = np.maximum(0.0, 1.0 - x * x) ** 0.5  # half-circle
            a *= total / a.sum()
        else:
            a = rng.dirichlet(np.ones(two_n)) * total

        # Run smoothed projected gradient with annealed beta
        beta = softmax_beta
        step = step_init
        prev_obj = np.inf
        for it in range(n_iters):
            # Compute Q_W(a) for all windows
            conv = np.convolve(a, a)
            cumconv = np.concatenate([[0.0], np.cumsum(conv)])
            qvals = (cumconv[s_his + 1] - cumconv[s_los]) * inv_norms

            # Smoothed max
            qmax = qvals.max()
            w = np.exp(beta * (qvals - qmax))
            w /= w.sum()
            obj_smooth = qmax + np.log(w.sum() * np.exp(-beta * 0.0)) / beta  # placeholder; use weighted
            obj_real = qmax  # the true objective

            # Gradient of smoothed max: sum_W w_W * grad Q_W
            # For each window, grad Q_W[k] = 2 * inv_norm * sum_{J in [s_lo-k, s_hi-k] cap [0, 2n-1]} a[J]
            # We compute this efficiently by building a "window mass" weighted accumulator.
            # Equivalently: grad sum_W w_W * Q_W = 2 * (P * a), where
            #   P[k, J] = sum_W w_W * inv_norm_W * 1{s_lo_W - k <= J <= s_hi_W - k, both in [0, 2n-1]}
            # That's expensive (O(n^4) per iter).  For moderate n (<= ~256) it's fine.
            # Cheaper: for each window, gradient contribution is 2*inv_norm*M_W*a where
            # M_W is the indicator matrix; sum_W w_W * M_W is a Toeplitz with structured rows.
            #
            # Faster: for each window W, the gradient is 2*inv_norm * (M_W @ a), and
            # M_W @ a at index k is the partial sum of a over [max(0,s_lo-k), min(2n-1,s_hi-k)].
            # Use prefix sums of a; vectorize over windows.
            #
            # Actually the SIMPLEST correct & fast formulation is this:
            # Let g[k] = sum_W w_W * 2 * inv_norm_W * (cumA[min(2n-1,s_hi-k)+1] - cumA[max(0,s_lo-k)]).
            # We compute g elementwise.
            # Vectorized gradient via "weight kernel" u(s):
            #   u(s) = sum_W w_W * inv_norm_W * 1{s_lo_W <= s <= s_hi_W}
            # supported on s in [0, 4n-2].  Then
            #   grad[k] = 2 * sum_{j=0}^{2n-1} a_j * u(k+j)
            # which is a correlation: grad = 2 * (u correlate a)[0..2n-1]
            # i.e. grad[k] = 2 * sum_j u[k+j] * a[j].
            conv_len = 2 * two_n - 1
            # Build u via prefix-sum over windows.
            # diff[s_lo] += w*inv_norm; diff[s_hi+1] -= w*inv_norm
            diff_arr = np.zeros(conv_len + 1)
            wts = w * inv_norms
            np.add.at(diff_arr, s_los, wts)
            np.add.at(diff_arr, s_his + 1, -wts)
            u = np.cumsum(diff_arr[:conv_len])
            # grad[k] = 2 * sum_{j=0}^{2n-1} a[j] * u[k+j]
            # = 2 * np.correlate(u, a, mode='valid')[k] for k=0..2n-1
            # since u has length 4n-1 and a has length 2n, valid output has length 2n.
            grad = 2.0 * np.correlate(u, a, mode='valid')

            # Projected gradient step
            a_new = a - step * grad
            a_new = project_simplex(a_new, total)

            # Diminishing step on plateau
            if obj_real >= prev_obj - tol:
                step *= 0.85
                beta *= 1.05
            prev_obj = obj_real
            a = a_new

            if step < 1e-7:
                break

        # Final exact objective at this iterate
        obj = cs_test_value(a)
        if obj < best_obj:
            best_obj = obj
            best_a = a.copy()
        if verbose:
            print(f"  restart {r:3d}: obj = {obj:.6f}, best = {best_obj:.6f}")

    return float(best_obj), best_a


# ---------------------------------------------------------------------
# Faster numpy-only heuristic using simulated annealing on the simplex
# (used for large n where projected gradient is too slow)
# ---------------------------------------------------------------------

def heuristic_an_star_sa(n: int, n_iters: int = 200000, seed: int | None = None,
                          temp_init: float = 0.05, temp_final: float = 1e-5,
                          step_size: float = 0.05, verbose: bool = False) -> Tuple[float, np.ndarray]:
    """Simulated annealing on the simplex with mass-conserving perturbations.

    Each step: pick (i,j), shift mass d (size step_size * 4n * temp) from j to i,
    keeping nonneg.  Accept by Metropolis.

    This is a HEURISTIC upper bound on a_n^*.
    """
    rng = np.random.default_rng(seed)
    two_n = 2 * n
    total = float(4 * n)

    a = np.full(two_n, total / two_n)
    obj = cs_test_value(a)
    best_obj = obj
    best_a = a.copy()

    log_t0 = np.log(temp_init)
    log_t1 = np.log(temp_final)
    for it in range(n_iters):
        t = np.exp(log_t0 + (log_t1 - log_t0) * it / max(1, n_iters - 1))
        # Choose two distinct indices
        i, j = rng.integers(0, two_n, size=2)
        if i == j:
            continue
        # Shift mass d from j to i
        d_max = a[j]
        d = rng.uniform(0, min(d_max, step_size * total))
        if d <= 0:
            continue
        a_new = a.copy()
        a_new[i] += d
        a_new[j] -= d
        obj_new = cs_test_value(a_new)
        # Minimization: accept if better, or by Metropolis prob exp(-(new-old)/t)
        if obj_new < obj or rng.random() < np.exp(-(obj_new - obj) / t):
            a = a_new
            obj = obj_new
            if obj < best_obj:
                best_obj = obj
                best_a = a.copy()
        if verbose and it % 10000 == 0:
            print(f"  iter {it:7d}: t={t:.5g}, obj={obj:.6f}, best={best_obj:.6f}")

    return float(best_obj), best_a


# ---------------------------------------------------------------------
# Discretized rigorous evaluation: wraps existing branch-and-prune
# ---------------------------------------------------------------------

def rigorous_lb(n_half: int, m: int, verbose: bool = True) -> dict:
    """Rigorous lower bound on c via Lemma 3 + branch-and-prune.

    Returns dict with: bound_on_c (= b_{n,m} - 2/m - 1/m^2),
                        b_nm (= raw min over B_{n,m}),
                        n_half, m, time.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from solvers import find_best_bound_direct
    t0 = time.time()
    res = find_best_bound_direct(n_half=n_half, m=m, verbose=verbose)
    el = time.time() - t0
    # find_best_bound_direct already subtracts the CS Lemma 3 correction
    # (corr = 2/m + 1/m^2), so `res` is the rigorous LB on c.
    bound_on_c = float(res)
    correction = 2.0 / m + 1.0 / (m * m)
    b_nm = bound_on_c + correction
    return {
        'n_half': n_half,
        'm': m,
        'bound_on_c': bound_on_c,
        'b_nm': b_nm,
        'correction': correction,
        'time_s': el,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Refined CS evaluator.")
    p.add_argument('--heuristic', action='store_true',
                    help='Compute heuristic upper bounds on a_n^*')
    p.add_argument('--rigorous', action='store_true',
                    help='Compute rigorous LB on c via branch-and-prune')
    p.add_argument('--n', type=int, nargs='*', default=[8, 16, 32, 64, 128],
                    help='Values of n (paper notation) for heuristic')
    p.add_argument('--n_half', type=int, nargs='*', default=[2, 3, 4],
                    help='Values of n_half = paper-n for rigorous')
    p.add_argument('--m', type=int, nargs='*', default=[20, 40],
                    help='Discretization parameters m for rigorous')
    p.add_argument('--restarts', type=int, default=64)
    p.add_argument('--iters', type=int, default=2000)
    p.add_argument('--sa_iters', type=int, default=80000,
                    help='SA iterations for large-n heuristic')
    p.add_argument('--sa_threshold', type=int, default=64,
                    help='Use SA instead of PGD when n > this')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--out', type=str, default=None,
                    help='Optional JSON output path')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    results = {'heuristic': [], 'rigorous': []}

    if args.heuristic:
        print("\n=== Heuristic upper bounds on a_n^* (NOT a rigorous LB on c) ===")
        for nv in args.n:
            t0 = time.time()
            if nv <= args.sa_threshold:
                obj, a = heuristic_an_star(
                    nv, n_restarts=args.restarts, n_iters=args.iters,
                    seed=args.seed, verbose=args.verbose)
                method = "PGD-softmax"
            else:
                obj, a = heuristic_an_star_sa(
                    nv, n_iters=args.sa_iters, seed=args.seed,
                    verbose=args.verbose)
                method = "SA"
            el = time.time() - t0
            row = {'n': nv, 'a_n_upper_heuristic': obj,
                   'method': method, 'time_s': el}
            results['heuristic'].append(row)
            print(f"  n={nv:4d} ({method:>15}): a_n* <= {obj:.6f}  ({el:.1f}s)")

    if args.rigorous:
        print("\n=== Rigorous LB on c via Lemma 3 (branch-and-prune) ===")
        for nh in args.n_half:
            for mv in args.m:
                try:
                    res = rigorous_lb(nh, mv, verbose=args.verbose)
                    results['rigorous'].append(res)
                    print(f"  n_half={nh}, m={mv:3d}: c >= {res['bound_on_c']:.6f}  "
                          f"(b_nm={res['b_nm']:.6f}, corr={res['correction']:.4f}, "
                          f"{res['time_s']:.1f}s)")
                except Exception as e:
                    print(f"  n_half={nh}, m={mv}: FAILED ({e})")

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.out}")

    # Summary: best rigorous LB and best heuristic upper bound
    print("\n=== Summary ===")
    if results['rigorous']:
        best_rig = max(r['bound_on_c'] for r in results['rigorous'])
        print(f"Best rigorous LB on c (this run): {best_rig:.6f}")
        if best_rig > 1.2802:
            print(f"  [!!] Exceeds 1.2802 by {best_rig - 1.2802:.6f}")
        else:
            print(f"  Below 1.2802 by {1.2802 - best_rig:.6f}")
    if results['heuristic']:
        # For each tested n, the heuristic gives an upper bound on a_n^*.
        # Lemma 1 says c >= a_n, so for the CS scheme to prove c > T at this n,
        # we'd need a_n^* > T.  If our heuristic UB <= T, then a_n^* <= T too
        # (because heuristic UB is an upper bound on a_n^*), so the CS scheme
        # at THIS n cannot exceed T -- but at larger n, a_n grows.
        # Report the maximum across n (since a_n is monotone non-decreasing).
        max_heu = max(r['a_n_upper_heuristic'] for r in results['heuristic'])
        n_max = max(r['n'] for r in results['heuristic'])
        print(f"Largest heuristic UB on a_n^* across tested n (n={n_max}): {max_heu:.6f}")
        print(f"  CS scheme at the LARGEST tested n can prove c > T only if a_n^* > T;")
        print(f"  a global heuristic UB <= 1.2802 at the largest n would mean CS@n is insufficient.")


if __name__ == '__main__':
    main()
