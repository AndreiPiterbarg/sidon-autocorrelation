"""Theorem 4 (atomic-nu dual) lower bound on C_{1a}.

REFERENCE: _master_compactness.md, Theorem 4. For any atomic measure
   nu = sum w_i delta_{t_i} with w_i >= 0, sum w_i = 1, t_i in [-1/2, 1/2],

     C_{1a} >= P(nu) := inf_{f in A} sum_i w_i (f*f)(t_i),

where A = {f >= 0, supp f subset [-1/4, 1/4], int f = 1}. The infimum is
attained by compactness (Theorem 1 of _master_compactness.md).

The outer supremum over (w, t) attains C_{1a} by strong duality
(min-max = max-min on a convex-compact set with continuous bilinear form).

==========
STRUCTURE OF THIS SCRIPT
==========

This script computes TWO quantities for many atomic-nu configurations:

(I)  PIECEWISE-CONSTANT INNER MINIMUM P_d_UB(nu): for a fixed bin-count d,
     min over piecewise-constant f (with bin masses mu) of
         G_d(mu, nu) = sum_i w_i * mu^T K_d(t_i) mu,
     where K_d(t)[i,j] = (1/w^2) * |B_i cap (t - B_j)|.
     Since pw-const f is in A (with the right normalization), this is an
     UPPER bound on P(nu). It SHOWS HOW HIGH we could ever certify by
     Theorem 4 with this discretization.

(II) BIN-PAIR LB on the SMEARED nu_eps (eps = bin width w):
     L_d_LB(mu, nu) = sum_i w_i * (1/(2*eps)) * [bin-LB of int_{W_i}(f*f)],
     a continuous-f-sound RIGOROUS lower bound on the smeared form.
     Then C_{1a} >= inf_f int (f*f) d(nu_eps) >= inf_mu L_d_LB(mu, nu).
     The smeared form approaches the atomic form as eps -> 0, but here
     eps = w is fixed.

If (II) is large (> 1.30), we have a breakthrough.

If (I) is small (i.e., the UB on P(nu) for any nu we tried is small), this
tells us Theorem 4 with finite atomic nu CANNOT push above that ceiling,
identifying an OBSTRUCTION.

The honest expected outcome: BOTH (I) and (II) will be small. Theorem 4 as
stated needs infinitely many atoms (the full active-set measure of the
extremizer), which we cannot enumerate without knowing the extremizer.

USAGE: python _theorem4_atomic_nu.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from fractions import Fraction
from typing import List, Tuple, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Bin kernel for piecewise-constant f
# ---------------------------------------------------------------------------

def bin_kernel_value(t: float, i: int, j: int, d: int) -> float:
    """(1/w^2) * |B_i cap (t - B_j)| for piecewise-constant f.

    B_i = [-1/4 + i*w, -1/4 + (i+1)*w], w = 1/(2d).
    Then (f*f)(t) = sum_{i,j} mu_i mu_j K(t)[i,j] / w^0 (with mu_i = int_{B_i} f).
    Actually since f = mu_i / w on B_i and (f*f)(t) = sum (mu_i/w)(mu_j/w) * overlap,
    K_d(t)[i,j] := overlap / w^2 so that (f*f)(t) = mu^T K_d(t) mu (mu is bin masses).
    """
    w = 1.0 / (2.0 * d)
    a_i = -0.25 + i * w
    # B_j = [a_j, a_j + w], so t - B_j = [t - a_j - w, t - a_j]
    a_j = -0.25 + j * w
    lo_R = t - a_j - w
    hi_R = t - a_j
    lo = max(a_i, lo_R)
    hi = min(a_i + w, hi_R)
    overlap = max(0.0, hi - lo)
    return overlap / (w * w)


def build_K_d_t(t: float, d: int) -> np.ndarray:
    """Build d x d matrix K_d(t)[i, j] for piecewise-const f."""
    K = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            K[i, j] = bin_kernel_value(t, i, j, d)
    return K


def build_Q_pwconst(nu_atoms: List[Tuple[float, float]], d: int) -> np.ndarray:
    """Q = sum_k w_k K_d(t_k); UB-direction matrix for (I)."""
    Q = np.zeros((d, d), dtype=np.float64)
    for w_k, t_k in nu_atoms:
        Q += w_k * build_K_d_t(t_k, d)
    return Q


def build_M_smeared_LB(nu_atoms: List[Tuple[float, int]], d: int) -> np.ndarray:
    """Smeared-LB matrix for (II): bin-aligned indices.

    M[i, j] = sum_k w_k * d * 1{i + j == k_idx_k}, symmetrized.
    """
    M = np.zeros((d, d), dtype=np.float64)
    for w_k, k_idx in nu_atoms:
        for i in range(d):
            j = k_idx - i
            if 0 <= j < d:
                M[i, j] += w_k * d
    return 0.5 * (M + M.T)


# ---------------------------------------------------------------------------
# QP min on simplex (Frank-Wolfe-style for PSD Q, simple and fast)
# ---------------------------------------------------------------------------

def qp_min_simplex_psd(Q: np.ndarray, n_restarts: int = 5,
                        max_iter: int = 200, tol: float = 1e-10,
                        seed: int = 0) -> Tuple[float, np.ndarray]:
    """Minimize mu^T Q mu over the simplex via projected gradient.

    For PSD Q, the global min on simplex is achieved either at the interior
    stationary point (if it's in the simplex) or on a face. We use
    multi-restart projected gradient. For small d this is robust.
    """
    d = Q.shape[0]
    rng = np.random.default_rng(seed)
    best_obj = np.inf
    best_mu = None

    # Restart 1: uniform start
    starts = [np.ones(d) / d]
    # Restart 2..k: vertex starts (one bin at a time)
    for i in range(min(d, n_restarts - 1)):
        v = np.zeros(d)
        v[i] = 1.0
        starts.append(v)
    # Add random Dirichlet starts
    while len(starts) < n_restarts:
        starts.append(rng.dirichlet(np.ones(d)))

    for mu0 in starts:
        mu = mu0.copy()
        # Spectral step size estimate
        L = max(np.linalg.eigvalsh(Q)[-1], 1e-12)
        step = 1.0 / L

        prev_obj = np.inf
        for it in range(max_iter):
            grad = 2.0 * Q @ mu
            mu_new = mu - step * grad
            mu_new = _project_to_simplex(mu_new)
            obj = float(mu_new @ Q @ mu_new)
            if abs(prev_obj - obj) < tol:
                mu = mu_new
                break
            mu = mu_new
            prev_obj = obj
        obj = float(mu @ Q @ mu)
        if obj < best_obj:
            best_obj = obj
            best_mu = mu.copy()

    # Also check pure vertices analytically (Q[i, i] for each i)
    for i in range(d):
        if Q[i, i] < best_obj:
            best_obj = Q[i, i]
            best_mu = np.zeros(d)
            best_mu[i] = 1.0

    return best_obj, best_mu


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u - cssv / np.arange(1, n + 1) > 0)[0]
    if len(rho) == 0:
        return np.ones(n) / n
    rho = rho[-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0)


# ---------------------------------------------------------------------------
# Outer search over (w, t) for the smeared-LB (II)
# ---------------------------------------------------------------------------

def search_aligned_atoms_II(d: int, k_atoms: int, n_grid_w: int = 11,
                              t_indices: Optional[List[int]] = None,
                              verbose: bool = False) -> dict:
    """Sweep bin-aligned atomic-nu for the smeared bin-LB (II)."""
    if t_indices is None:
        t_indices = list(range(0, 2 * d - 1))
    t_subsets = list(itertools.combinations(t_indices, k_atoms))

    K = n_grid_w - 1
    weight_compositions = []
    for parts in itertools.product(range(K + 1), repeat=k_atoms):
        if sum(parts) == K:
            weight_compositions.append(
                tuple(p / K if K > 0 else 1.0/k_atoms for p in parts))
    if not weight_compositions:
        weight_compositions = [tuple(1.0/k_atoms for _ in range(k_atoms))]

    best_LB = -np.inf
    best_cfg = None
    n_cfg = 0
    for t_sub in t_subsets:
        for w_tup in weight_compositions:
            n_cfg += 1
            nu = list(zip(w_tup, t_sub))
            M = build_M_smeared_LB(nu, d)
            obj, mu = qp_min_simplex_psd(M, n_restarts=4)
            if obj > best_LB:
                best_LB = obj
                best_cfg = {
                    'd': d, 'k_atoms': k_atoms,
                    't_indices': list(t_sub),
                    't_values': [(-0.5 + (ti + 1) * (1.0 / (2 * d)))
                                  for ti in t_sub],
                    'weights': list(w_tup),
                    'LB_II': obj,
                    'mu_opt': mu.tolist(),
                }
    return {'best_LB_II': best_LB, 'best_config_II': best_cfg,
            'n_configs': n_cfg}


def search_general_atoms_I(d: int, k_atoms: int, n_grid_t: int = 25,
                             n_grid_w: int = 11,
                             verbose: bool = False) -> dict:
    """Sweep general (non-aligned) atomic-nu for the piecewise-const UB (I).

    UB on P(nu): inf over piecewise-const f of mu^T Q mu, Q = sum w_k K_d(t_k).
    """
    t_grid = np.linspace(-0.5 + 1e-3, 0.5 - 1e-3, n_grid_t)
    t_subsets = list(itertools.combinations(range(n_grid_t), k_atoms))

    K = n_grid_w - 1
    weight_compositions = []
    for parts in itertools.product(range(K + 1), repeat=k_atoms):
        if sum(parts) == K:
            weight_compositions.append(
                tuple(p / K if K > 0 else 1.0/k_atoms for p in parts))
    if not weight_compositions:
        weight_compositions = [tuple(1.0/k_atoms for _ in range(k_atoms))]

    best_UB = -np.inf
    best_cfg = None
    n_cfg = 0
    # Limit subsets for speed
    if len(t_subsets) > 5000:
        # Sample random subsets
        rng = np.random.default_rng(42)
        idx = rng.choice(len(t_subsets), 5000, replace=False)
        t_subsets = [t_subsets[i] for i in idx]
    for t_sub in t_subsets:
        for w_tup in weight_compositions:
            n_cfg += 1
            t_vals = [t_grid[ti] for ti in t_sub]
            nu = list(zip(w_tup, t_vals))
            Q = build_Q_pwconst(nu, d)
            obj, mu = qp_min_simplex_psd(Q, n_restarts=3)
            if obj > best_UB:
                best_UB = obj
                best_cfg = {
                    'd': d, 'k_atoms': k_atoms,
                    't_values': t_vals,
                    'weights': list(w_tup),
                    'P_UB': obj,
                    'mu_opt': mu.tolist(),
                }
    return {'best_P_UB': best_UB, 'best_config_I': best_cfg,
            'n_configs': n_cfg}


# ---------------------------------------------------------------------------
# Diagnostic: extremizer-like distribution test
# ---------------------------------------------------------------------------

def diagnostic_extremizer_test(d: int = 20) -> dict:
    """For an MV-like f (arcsine on [-1/4, 1/4]), compute sum w_i (f*f)(t_i)
    for various nu, to see what M values arise.

    This gives a SANITY CHECK on the (I) upper bound at a non-trivial mu.
    """
    # MV-like: f = (1/pi) / sqrt(1/16 - x^2) (arcsine density on [-1/4, 1/4])
    # Bin masses:
    bin_masses = []
    w = 1.0 / (2 * d)
    for i in range(d):
        a = -0.25 + i * w
        b = a + w
        # int_a^b 1/(pi sqrt(1/16 - x^2)) dx = (2/pi) [arcsin(4 x)]_a^b
        # Note: avoid endpoints exactly
        eps = 1e-12
        a_c = min(max(a, -0.25 + eps), 0.25 - eps)
        b_c = min(max(b, -0.25 + eps), 0.25 - eps)
        mass = (2 / np.pi) * (np.arcsin(4 * b_c) - np.arcsin(4 * a_c))
        bin_masses.append(mass)
    mu_arcsine = np.array(bin_masses)
    mu_arcsine /= mu_arcsine.sum()

    # Uniform mu
    mu_unif = np.ones(d) / d

    # Quick check: peak (f*f)(t) at t=0 with arcsine vs uniform
    K_0 = build_K_d_t(0.0, d)
    arc_at_0 = float(mu_arcsine @ K_0 @ mu_arcsine)
    unif_at_0 = float(mu_unif @ K_0 @ mu_unif)
    return {
        'd': d,
        'arcsine_ff_at_0': arc_at_0,
        'uniform_ff_at_0': unif_at_0,
        'note': '(f*f)(0) at piecewise-const arcsine vs uniform',
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=" * 72, flush=True)
    print("THEOREM 4: atomic-nu dual lower bound on C_{1a}", flush=True)
    print("=" * 72, flush=True)
    print(flush=True)
    print("Computing TWO quantities for many atomic-nu configs:", flush=True)
    print("  (I)  P_d_UB(nu)  = inf over PIECEWISE-CONST f of "
          "sum w_i mu^T K_d(t_i) mu (UB on P(nu))", flush=True)
    print("  (II) L_d_LB(nu) = smeared bin-pair bilinear LB "
          "(rigorous LB on int (f*f) d(nu_eps))", flush=True)
    print(flush=True)
    print("Strong inequality chain: C_{1a} >= inf_f (smeared form) >= "
          "L_d_LB. And P_d_UB >= inf_{pw-const f} (...) >= ... yields no LB", flush=True)
    print("on C_{1a}; P_d_UB is informational only (shows ceiling of the", flush=True)
    print("piecewise-const restriction).", flush=True)
    print(flush=True)

    sys.stdout.flush()

    # --- Diagnostic: arcsine bin masses ---
    print("\n[diagnostic] arcsine bin masses test:", flush=True)
    for d in [6, 10, 16]:
        diag = diagnostic_extremizer_test(d)
        print(f"  d={d}: arcsine (f*f)(0) = {diag['arcsine_ff_at_0']:.4f}, "
              f"uniform = {diag['uniform_ff_at_0']:.4f}", flush=True)
    print(flush=True)
    sys.stdout.flush()

    results_II = {}
    results_I = {}

    # ---- Part (II): smeared rigorous LB ----
    print("=" * 72, flush=True)
    print("(II) SMEARED ATOMIC-NU RIGOROUS LB (BIN-ALIGNED)", flush=True)
    print("=" * 72, flush=True)
    sweep_II = [(4, [1, 2, 3, 4]), (6, [1, 2, 3, 4]),
                (8, [1, 2, 3]), (10, [1, 2])]
    for d, ks in sweep_II:
        print(f"\n--- d = {d} ---", flush=True)
        results_II[f'd={d}'] = {}
        for k in ks:
            n_grid_w = {1: 1, 2: 11, 3: 7, 4: 5}[k]
            t0 = time.time()
            res = search_aligned_atoms_II(d, k, n_grid_w=n_grid_w)
            el = time.time() - t0
            print(f"  k={k}: best LB_II = {res['best_LB_II']:.6f} "
                  f"({res['n_configs']} cfgs, {el:.1f}s)", flush=True)
            results_II[f'd={d}'][f'k={k}'] = res
            sys.stdout.flush()

    print(flush=True)

    # ---- Part (I): general-t UB ----
    print("=" * 72, flush=True)
    print("(I) PIECEWISE-CONST INNER MIN (UB ON P(nu)) -- INFORMATIONAL", flush=True)
    print("=" * 72, flush=True)
    for d in [6, 10]:
        print(f"\n--- d = {d} ---", flush=True)
        results_I[f'd={d}'] = {}
        for k in [1, 2, 3]:
            n_grid_t = {1: 41, 2: 21, 3: 13}[k]
            n_grid_w = {1: 1, 2: 11, 3: 5}[k]
            t0 = time.time()
            res = search_general_atoms_I(d, k, n_grid_t=n_grid_t,
                                          n_grid_w=n_grid_w)
            el = time.time() - t0
            print(f"  k={k}: best P_UB = {res['best_P_UB']:.6f} "
                  f"({res['n_configs']} cfgs, {el:.1f}s)", flush=True)
            results_I[f'd={d}'][f'k={k}'] = res
            sys.stdout.flush()

    # ---- Overall best ----
    print(flush=True)
    print("=" * 72, flush=True)
    print("OVERALL BEST", flush=True)
    print("=" * 72, flush=True)
    best_II = -np.inf
    best_II_cfg = None
    for d_key, d_res in results_II.items():
        for k_key, k_res in d_res.items():
            if k_res['best_LB_II'] > best_II:
                best_II = k_res['best_LB_II']
                best_II_cfg = k_res['best_config_II']
    print(f"\nBest (II) rigorous LB: {best_II:.6f}", flush=True)
    if best_II_cfg:
        print(f"  d = {best_II_cfg['d']}, k = {best_II_cfg['k_atoms']}", flush=True)
        print(f"  t_indices = {best_II_cfg['t_indices']}", flush=True)
        print(f"  t_values  = {[f'{t:+.4f}' for t in best_II_cfg['t_values']]}", flush=True)
        print(f"  weights   = {[f'{w:.4f}' for w in best_II_cfg['weights']]}", flush=True)
        print(f"  mu_opt    = {[f'{m:.4f}' for m in best_II_cfg['mu_opt']]}", flush=True)

    best_I = -np.inf
    best_I_cfg = None
    for d_key, d_res in results_I.items():
        for k_key, k_res in d_res.items():
            if k_res['best_P_UB'] > best_I:
                best_I = k_res['best_P_UB']
                best_I_cfg = k_res['best_config_I']
    print(f"\nBest (I) piecewise-const UB on P(nu): {best_I:.6f}", flush=True)
    if best_I_cfg:
        print(f"  d = {best_I_cfg['d']}, k = {best_I_cfg['k_atoms']}", flush=True)
        print(f"  t_values = {[f'{t:+.4f}' for t in best_I_cfg['t_values']]}", flush=True)
        print(f"  weights  = {[f'{w:.4f}' for w in best_I_cfg['weights']]}", flush=True)

    # ---- Targets ----
    print(flush=True)
    print("=" * 72, flush=True)
    print("COMPARISON TO TARGETS", flush=True)
    print("=" * 72, flush=True)
    targets = [
        ('MV',                1.2748),
        ('CS17 (unsound)',    1.2802),
        ('1.30 (threshold)',  1.3000),
        ('1.378 (Theor 4 target)', 1.3784),
        ('1.5029 (UB)',       1.5029),
    ]
    for name, M in targets:
        if best_II > M:
            print(f"  RIGOROUS LB {best_II:.4f} BEATS {name} ({M}) "
                  f"by +{best_II - M:.4f}", flush=True)
        else:
            print(f"  RIGOROUS LB {best_II:.4f} BELOW {name} ({M}) "
                  f"by -{M - best_II:.4f}", flush=True)

    # ---- Farkas rational rounding for the best (II) ----
    print(flush=True)
    print("=" * 72, flush=True)
    print("FARKAS RATIONAL ROUNDING (best II)", flush=True)
    print("=" * 72, flush=True)
    if best_II_cfg:
        d = best_II_cfg['d']
        t_idx = best_II_cfg['t_indices']
        weights = best_II_cfg['weights']
        mu_opt = best_II_cfg['mu_opt']

        # All t_values are exactly rational
        rat_t = [Fraction(ti + 1, 2 * d) - Fraction(1, 2) for ti in t_idx]
        rat_w = [Fraction(int(round(w * 10000)), 10000) for w in weights]
        S_w = sum(rat_w)
        rat_w = [w / S_w for w in rat_w]
        rat_mu = [Fraction(int(round(m * 10000)), 10000) for m in mu_opt]
        S_mu = sum(rat_mu)
        rat_mu = [m / S_mu for m in rat_mu]

        # Compute the form value exactly
        form_val = Fraction(0)
        for w_k, k_idx in zip(rat_w, t_idx):
            s_k = Fraction(0)
            for i in range(d):
                j = k_idx - i
                if 0 <= j < d:
                    s_k += rat_mu[i] * rat_mu[j]
            form_val += w_k * d * s_k

        form_val_frac = form_val.limit_denominator(10**10)
        form_val_float = float(form_val)

        print(f"  Rational t_values: {[str(t) for t in rat_t]}", flush=True)
        print(f"  Rational weights:  {[str(w) for w in rat_w]}", flush=True)
        print(f"  Rational mu_opt:   {[str(m) for m in rat_mu]}", flush=True)
        print(f"  Form value at rat_mu (= F(rat_mu)): {form_val_float:.10f}", flush=True)
        print(f"                                  = {form_val_frac}", flush=True)
        print(flush=True)
        print(f"  IMPORTANT INTERPRETATION:", flush=True)
        print(f"    F(rat_mu) = {form_val_float:.6f} is an UPPER bound on", flush=True)
        print(f"    the QP infimum inf_mu F(mu). The numerical LB is the", flush=True)
        print(f"    QP infimum (since mu_opt is its minimizer); for a", flush=True)
        print(f"    rigorous LB on inf_mu F we need an SDP dual cert.", flush=True)
        print(f"    For PSD M, the QP min = MM-eigenvalue, certifiable via", flush=True)
        print(f"    a Schur complement / dual multiplier.", flush=True)

    elapsed = time.time() - t_start
    print(f"\nTotal wall: {elapsed:.2f}s", flush=True)

    out = {
        'theorem_4': {
            'statement': 'C_{1a} >= P(nu) = inf_f sum w_i (f*f)(t_i) for atomic nu',
            'attainment': 'inf attained by compactness (Theorem 1, master_compactness.md)',
        },
        'method_II_rigorous': {
            'name': 'smeared atomic-nu bin-pair bilinear LB',
            'description': ('Replace delta_{t_k} by (1/2eps) 1_{[t_k - eps, t_k + eps]} '
                           'with eps = bin width; for continuous f, '
                           'sum w_i int_{W_i}(f*f) / (2*eps) is bounded below by '
                           'd * sum w_i * sum_{i+j = k_idx_i} mu_i mu_j (rigorous '
                           'continuous-f-sound bin-LB from _smoke_bochner_test.py).'),
            'rigor': 'CONTINUOUS-f-SOUND; rigorous LB on C_{1a}',
            'best_LB_II': best_II,
            'best_config_II': best_II_cfg,
        },
        'method_I_diagnostic': {
            'name': 'piecewise-constant inner-min UB on P(nu)',
            'description': ('inf over piecewise-const f of sum w_i (f*f)(t_i) '
                           '(quadratic form mu^T Q mu, Q = sum w_k K_d(t_k)). '
                           'This is an UPPER bound on the true P(nu), and so '
                           'shows how high Theorem 4 could possibly push by '
                           'this discretization. NOT a LB on C_{1a}.'),
            'best_UB': best_I,
            'best_config_I': best_I_cfg,
        },
        'results_II_per_d_k': {
            d_key: {k_key: {'best_LB_II': r['best_LB_II'],
                            'best_config_II': r['best_config_II']}
                    for k_key, r in d_res.items()}
            for d_key, d_res in results_II.items()
        },
        'results_I_per_d_k': {
            d_key: {k_key: {'best_P_UB': r['best_P_UB'],
                            'best_config_I': r['best_config_I']}
                    for k_key, r in d_res.items()}
            for d_key, d_res in results_I.items()
        },
        'wall_seconds': elapsed,
    }

    out_path = os.path.join(_HERE, '_theorem4_atomic_nu.json')
    with open(out_path, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f"\n[saved] {out_path}", flush=True)

    # --- Final verdict ---
    print(flush=True)
    print("=" * 72, flush=True)
    print("VERDICT", flush=True)
    print("=" * 72, flush=True)
    if best_II > 1.30:
        print(f"  BREAKTHROUGH: rigorous LB {best_II:.4f} > 1.30!", flush=True)
    elif best_II > 1.2802:
        print(f"  RIGOROUS BEATS UNSOUND CS17: {best_II:.4f} > 1.2802", flush=True)
    elif best_II > 1.2748:
        print(f"  Rigorous beats MV: {best_II:.4f} > 1.2748", flush=True)
    elif best_II > 1.0:
        print(f"  Rigorous {best_II:.4f}, above trivial but below MV", flush=True)
    elif best_II > 0:
        print(f"  Rigorous {best_II:.4f} -- positive but trivial-level", flush=True)
    else:
        print(f"  TRIVIAL/ZERO: LB = {best_II:.4f}", flush=True)

    return out


if __name__ == '__main__':
    main()
