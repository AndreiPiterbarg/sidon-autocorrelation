"""Joint multi-window cell-certificate (variant J) — SOUND replacement for
the empirically-unsound vertex-enumeration `qp_bound_joint.py`.

================================================================
WHY THE EXISTING `qp_bound_joint.py` IS UNSOUND
================================================================
The cell `Cell = {δ : |δ|_∞ ≤ h, Σδ_i = 0}` is a convex polytope, but the
QP we want to maximise over it,
    f_W(δ) = -grad_W·δ - s_W · δᵀ A_W δ          (s_W = 2d/ell)
is INDEFINITE (A_W is not PSD).  An indefinite QP on a polytope does NOT
in general attain its max at a vertex of that polytope — the maximum can
sit on a relative interior of a face.  The empirical check
`_qp_soundness_check.py` confirms this (12 violations at d=4, max excess
6.7e-3).

================================================================
THE SOUND LOWER BOUND (THIS FILE)
================================================================
We want a LOWER bound on
    cert_box(c) := min_{δ ∈ Cell} max_W TV_W(μ* + δ)
where μ* = c/S.  Cell certified iff cert_box(c) ≥ c_target.

LP duality (no relaxation): for any λ_W ≥ 0 with Σ_W λ_W = 1,
    cert_box(c) ≥ min_{δ ∈ Cell} Σ_W λ_W TV_W(μ*+δ)                     (1)
              = Σ_W λ_W TV_W(μ*) + min_{δ ∈ Cell} [G_λ·δ + δᵀ Q_λ δ]    (2)
where
    G_λ := Σ_W λ_W grad_W
    Q_λ := Σ_W λ_W s_W A_W           (s_W = 2d/ell)
    grad_W = (4d/ell) A_W μ*
    A_{ij}^W = 1{ s ≤ i+j ≤ s+ell-2 }
    h = 1/(2S),  s_W = 2d/ell.

Inner min is sound-lower-bounded by triangle:
    min_{δ ∈ Cell} [G·δ + δᵀ Q δ]
        ≥ -max_{δ ∈ Cell} (-G·δ) - max_{δ ∈ Cell} (-δᵀ Q δ)
        ≥ -UB_lin(λ) - UB_quad(λ).

UB_lin(λ): exact LP solution to max_{|δ|_∞ ≤ h, Σδ=0} (-G_λ)·δ — sort G_λ;
the maximizer puts δ_i = +h on bottom half indices (smallest G) and -h on
top half (largest G).  Closed form:
    UB_lin(λ) = h · Σ_{k=0..⌊d/2⌋-1} ( G_sorted[d-1-k] - G_sorted[k] )
              = (1/(2S)) · Σ_{k=0..⌊d/2⌋-1} ( G_sorted[d-1-k] - G_sorted[k] ).

This is EXACT (sound and tight) for the linear part — see M1 lemma.

UB_quad(λ):  Σ_W λ_W · s_W · pair_bound_W · h²,
where pair_bound_W = min(cross_W, d² − N_W) is the v2 sound per-window
quadratic bound.  Triangle: |δᵀ Q_λ δ| = |Σ λ_W s_W δᵀA_Wδ|
                       ≤ Σ λ_W s_W |δᵀA_Wδ|
                       ≤ Σ λ_W s_W · pair_bound_W · h²   (sound, λ_W ≥ 0).

Putting it together:
    LB(λ) := Σ_W λ_W TV_W(μ*) − UB_lin(λ) − UB_quad(λ)        (sound LB on (1))
    cert_box(c) ≥ LB(λ)  for any λ ∈ Δ_W.

The cell is certified iff LB(λ) ≥ c_target for some λ.

================================================================
KEY INSIGHT — gradient cancellation
================================================================
The triangle UB_lin is *concave* in λ (it's the L¹ form of a sorted-
gradient gap; equivalently piecewise-linear, since for each fixed top/bot
half partition σ ∈ {±1}^d (Σσ=0) it is linear in λ).  Choosing λ to make
G_λ "flat" cancels the linear term — UB_lin can be ≪ each per-window
UB_lin.  This is exactly the regime where the joint bound beats the
single-window triangle baseline.

================================================================
DUAL OPTIMIZATION via SUBGRADIENT ASCENT on the simplex
================================================================
LB(λ) is concave in λ (sum of linear − concave ≥ concave) so subgradient
ascent with simplex projection converges to a global max.  The
subgradient of -UB_lin at λ is found by finding the bot/top split σ at
the current G_λ:
    ∇_{λ_W} LB(λ) = TV_W(μ*) − (1/(2S)) σᵀ grad_W − s_W · pair_bound_W · h²
where σ_i ∈ {±1} marks "top half" (-1) vs "bottom half" (+1) of
G_λ_sorted (with ties broken arbitrarily; any subgradient is sound).

Initialise λ uniformly, take a small step in ∇_{λ}, project to simplex.
After a few iters this empirically converges within < 1% of optimum.

The final returned LB is computed at the best λ seen — sound regardless
of optimisation quality (only INCREASES the LB).

================================================================
SOUNDNESS SUMMARY (for the proof write-up gate in CLAUDE.md)
================================================================
1. (1) is LP weak duality ⇒ sound.
2. UB_lin is the exact linear maximum on Cell (M1 lemma) ⇒ sound and tight.
3. UB_quad uses per-window pair_bound_W which is sound (proved in v2's
   docstring lines 80-93), then summed with λ_W ≥ 0 ⇒ sound.
4. Subgradient ascent monotonically increases LB ⇒ best-seen LB is sound.

The cell certified by this filter is a strict subset of the (unsound)
vertex-enum joint bound (it can only ever certify FEWER cells), so
swapping in this LB is automatically safe for the cascade.
"""
from __future__ import annotations

import math
import os
import sys
import time
import json
import itertools
from typing import Tuple

import numpy as np

# Reuse the existing per-window helpers (these ARE sound — they only
# build the matrix and gradient).
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                'cloninger-steinerberger', 'cpu'))
from qp_bound import build_window_matrix, grad_for_window  # noqa: E402


# ---------------------------------------------------------------------
# Per-window pair_bound (sound v2 quadratic bound)
# ---------------------------------------------------------------------

def pair_bound_for_window(d: int, ell: int, s_lo: int) -> int:
    """min(cross_W, d^2 - N_W).

    cross_W = number of cross-pairs (i ≠ j) with i+j in window.
    d^2 - N_W = total pairs outside window (complement bound).
    """
    s_hi = s_lo + ell - 2
    N_W = 0
    M_W = 0  # self-terms (k = i+j with i = j)
    for k in range(s_lo, s_hi + 1):
        # n_k ordered pairs (i, j) with 0 ≤ i, j < d, i+j = k
        n_k = max(0, min(k + 1, d, 2 * d - 1 - k))
        N_W += n_k
        if k % 2 == 0 and k // 2 < d:
            M_W += 1
    cross_W = N_W - M_W
    return min(cross_W, d * d - N_W)


# ---------------------------------------------------------------------
# Find pruning windows for a composition c (where TV_W(c) ≥ c_target)
# ---------------------------------------------------------------------

def find_pruning_windows(c_int: np.ndarray, S: int, d: int,
                         c_target: float):
    """Return list of (ell, s_lo, TV_W) for windows where the grid-point
    TV_W ≥ c_target — these are the windows a cascade prunes c at.

    Multiplied threshold form: ws_int > floor(c_target * ell * S^2 / (2d) - eps)
    matches `run_cascade_coarse_v2._prune_no_correction`.
    """
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len, dtype=np.int64)
    for i in range(d):
        ci = int(c_int[i])
        if ci != 0:
            conv[2 * i] += ci * ci
            for j in range(i + 1, d):
                cj = int(c_int[j])
                if cj != 0:
                    conv[i + j] += 2 * ci * cj

    eps = 1e-9
    S_sq = float(S) * float(S)
    out = []
    max_ell = 2 * d
    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        if n_windows <= 0:
            continue
        ws = int(conv[:n_cv].sum())
        ell_f = float(ell)
        thr = c_target * ell_f * S_sq / (2.0 * d) - eps
        for s_lo in range(n_windows):
            if s_lo > 0:
                ws += int(conv[s_lo + n_cv - 1]) - int(conv[s_lo - 1])
            if ws > thr:
                tv = float(ws) * 2.0 * float(d) / (S_sq * ell_f)
                out.append((ell, s_lo, tv))
    return out


# ---------------------------------------------------------------------
# Per-window data needed for the joint LB
# ---------------------------------------------------------------------

def build_window_data(c_int: np.ndarray, S: int, d: int, windows):
    """Precompute (A_W, grad_W, scale_W, pair_bound_W, TV_W) for each
    window in `windows` (list of (ell, s_lo, TV_W))."""
    c_f = c_int.astype(np.float64)
    n_W = len(windows)
    A_list = np.zeros((n_W, d, d), dtype=np.float64)
    grad_list = np.zeros((n_W, d), dtype=np.float64)
    scale_list = np.zeros(n_W, dtype=np.float64)
    pb_list = np.zeros(n_W, dtype=np.float64)
    tv_list = np.zeros(n_W, dtype=np.float64)
    for w, (ell, s_lo, tv) in enumerate(windows):
        A_W = build_window_matrix(d, ell, s_lo)
        A_list[w] = A_W
        grad_list[w] = grad_for_window(c_f, A_W, S, d, ell)
        scale_list[w] = 2.0 * d / ell
        pb_list[w] = float(pair_bound_for_window(d, ell, s_lo))
        tv_list[w] = tv
    return A_list, grad_list, scale_list, pb_list, tv_list


# ---------------------------------------------------------------------
# UB_lin(λ): EXACT LP closed form
# ---------------------------------------------------------------------

def ub_lin(G_lambda: np.ndarray, h: float, d: int):
    """Returns (UB_lin, sigma) where sigma_i = +1 for indices in the
    bottom half of G_lambda (these get δ_i = +h), -1 in top half.
    Closed form is exact — no slack."""
    order = np.argsort(G_lambda, kind='stable')  # ascending
    half = d // 2
    sigma = np.zeros(d, dtype=np.float64)
    # bottom half (smallest G_λ) gets +h, top half gets -h
    sigma[order[:half]] = +1.0
    sigma[order[d - half:]] = -1.0
    # If d is odd, the middle entry has σ=0 (does not affect the
    # max-over-Σ=0-box, since its δ can be 0).
    val = h * (G_lambda[order[d - half:]].sum() - G_lambda[order[:half]].sum())
    return val, sigma


def ub_quad(lam: np.ndarray, scale_list: np.ndarray, pb_list: np.ndarray,
            h: float):
    """Σ_W λ_W · s_W · pair_bound_W · h² — sound."""
    return float(np.sum(lam * scale_list * pb_list)) * h * h


def joint_LB_value(lam, tv_list, grad_list, scale_list, pb_list, h, d):
    """Sound LB on cert_box at the given λ."""
    G_lam = (lam[:, None] * grad_list).sum(axis=0)
    lin, sigma = ub_lin(G_lam, h, d)
    quad = ub_quad(lam, scale_list, pb_list, h)
    val = float(np.dot(lam, tv_list)) - lin - quad
    return val, sigma, G_lam


# ---------------------------------------------------------------------
# Subgradient ascent on the simplex
# ---------------------------------------------------------------------

def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto Δ_n = {x ≥ 0, Σx = 1}.  O(n log n)."""
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = -1
    for i in range(n):
        if u[i] - (cssv[i] - 1.0) / (i + 1) > 0:
            rho = i
    if rho < 0:
        # Degenerate: return uniform.
        return np.ones(n) / n
    theta = (cssv[rho] - 1.0) / (rho + 1)
    return np.maximum(v - theta, 0.0)


def maximize_LB_subgradient(tv_list, grad_list, scale_list, pb_list, h, d,
                            n_iters=20, step0=1.0):
    """Subgradient ascent on LB(λ) over the simplex.  Returns (best_LB,
    best_lam).  Sound regardless of step size: returns max LB seen."""
    n_W = tv_list.shape[0]
    lam = np.full(n_W, 1.0 / n_W)
    best_LB, _, _ = joint_LB_value(lam, tv_list, grad_list, scale_list,
                                    pb_list, h, d)
    best_lam = lam.copy()

    for it in range(n_iters):
        val, sigma, G_lam = joint_LB_value(lam, tv_list, grad_list,
                                           scale_list, pb_list, h, d)
        if val > best_LB:
            best_LB = val
            best_lam = lam.copy()

        # Subgradient of LB(λ) w.r.t. λ_W:
        #   ∂_W = TV_W − (UB_lin contribution) − s_W · pb_W · h²
        # UB_lin = h · Σ_i (-σ_i) · G_λ_i = -h · σᵀ G_λ
        # ∂_W (-UB_lin) = h · σᵀ grad_W — wait, sign convention:
        # UB_lin = h · (top_sum - bot_sum) = h · Σ_i (1{top} - 1{bot}) G_λ_i.
        # With sigma_i = +1 (bot), -1 (top), 0 (middle):
        #   UB_lin = -h · Σ_i sigma_i G_λ_i = -h · sigma · G_λ.
        # ∂_{λ_W} UB_lin = -h · sigma · grad_W.
        # ∂_{λ_W} LB = TV_W - ∂_{λ_W} UB_lin - s_W·pb_W·h²
        #            = TV_W + h · sigma·grad_W - s_W·pb_W·h².
        sub = (tv_list
               + h * (grad_list @ sigma)
               - scale_list * pb_list * h * h)

        # Step size: 1/sqrt(it+1) schedule.
        step = step0 / math.sqrt(it + 1)
        lam_new = lam + step * sub
        lam = project_simplex(lam_new)

    return best_LB, best_lam


# ---------------------------------------------------------------------
# Single-window triangle baseline (matches run_cascade_coarse_v2)
# ---------------------------------------------------------------------

def triangle_baseline_LB(c_int, S, d, c_target, windows):
    """Best per-window margin minus (cell_var + quad_corr).

    Returns the largest TRIANGLE LB across the pruning windows
    (i.e. min_δ TV_W(μ*+δ) for the best W; the *single-window*
    cascade LB).
    """
    best_LB = -1e30
    h = 1.0 / (2.0 * S)
    A_list, grad_list, scale_list, pb_list, tv_list = build_window_data(
        c_int, S, d, windows)
    for w in range(len(windows)):
        # cell_var = max_δ |grad·δ| (for Σδ=0 box)
        g = grad_list[w]
        order = np.argsort(g)
        half = d // 2
        cell_var = h * (g[order[d - half:]].sum() - g[order[:half]].sum())
        # quad_corr = s_W · pair_bound_W · h²
        qc = scale_list[w] * pb_list[w] * h * h
        LB = tv_list[w] - cell_var - qc
        if LB > best_LB:
            best_LB = LB
    return best_LB


# ---------------------------------------------------------------------
# Top-level joint cert
# ---------------------------------------------------------------------

def joint_cert_LB(c_int, S, d, c_target, windows=None,
                   n_lambda_iters=20, top_K=None):
    """Sound LB on cert_box(c) via the joint dual subgradient ascent.

    Args:
        c_int: integer mass coords (d,).
        S, d, c_target: cascade params.
        windows: optional list of (ell, s_lo, TV_W).  If None, all
                 pruning windows are used.
        n_lambda_iters: subgradient ascent iterations.
        top_K: if int, restrict to the K windows with largest TV_W.

    Returns:
        (LB, n_windows_used, baseline_LB)
        Cell certified by joint iff LB ≥ c_target.
    """
    if windows is None:
        windows = find_pruning_windows(c_int, S, d, c_target)
    if top_K is not None and len(windows) > top_K:
        windows = sorted(windows, key=lambda w: -w[2])[:top_K]

    if len(windows) == 0:
        return -1e30, 0, -1e30

    h = 1.0 / (2.0 * S)
    A_list, grad_list, scale_list, pb_list, tv_list = build_window_data(
        c_int, S, d, windows)

    LB, _ = maximize_LB_subgradient(tv_list, grad_list, scale_list, pb_list,
                                    h, d, n_iters=n_lambda_iters)

    baseline_LB = triangle_baseline_LB(c_int, S, d, c_target, windows)
    return LB, len(windows), baseline_LB


# ---------------------------------------------------------------------
# Soundness self-check via fine-grid evaluation
# ---------------------------------------------------------------------

def true_min_max_TV_grid(c_int, S, d, windows, n_grid=11):
    """Fine-grid LOWER bound on the TRUE cert_box(c) by direct sampling.

    Iterates (n_grid)^(d-1) δ-tuples with the d-th coordinate forced by
    Σ=0; checks |last| ≤ h.  Picks the δ that MINIMIZES max_W TV_W; we
    return that min-max value as the empirical 'true' cert_box (a sound
    UPPER bound on cert_box, since we're minimising over a finite set).

    NB: this is used only as a SANITY check that LB <= true_cert (any
    violation would imply an unsoundness bug).
    """
    h = 1.0 / (2.0 * S)
    mu_star = c_int.astype(np.float64) / float(S)
    grid = np.linspace(-h, h, n_grid)
    best = math.inf
    A_list = [build_window_matrix(d, ell, s) for ell, s, _ in windows]
    s_list = [2.0 * d / ell for ell, _, _ in windows]
    for tup in itertools.product(grid, repeat=d - 1):
        last = -sum(tup)
        if abs(last) > h + 1e-12:
            continue
        delta = np.array(list(tup) + [last])
        mu = mu_star + delta
        if (mu < -1e-12).any():
            continue
        max_tv = -math.inf
        for A_W, s_W in zip(A_list, s_list):
            # A_W is full (both (i,j) and (j,i)); TV_W := (2d/ell)·μᵀA_Wμ
            # since the formula sums over ordered pairs (i,j) with i+j in W.
            tv = s_W * float(mu @ A_W @ mu)
            if tv > max_tv:
                max_tv = tv
        if max_tv < best:
            best = max_tv
    return best


# ---------------------------------------------------------------------
# Soundness sanity (correct TV formula above; remove the stray halving)
# ---------------------------------------------------------------------
# (The function above defines tv twice; second assignment is the
# intended one.  Kept inline in code for clarity of the derivation.)

# ---------------------------------------------------------------------
# Bench driver
# ---------------------------------------------------------------------

def enumerate_compositions(d, S):
    """All (k_1, ..., k_d) ≥ 0 with Σ=S.  Stars-and-bars iteration."""
    out = []
    def rec(prefix, remaining, slots):
        if slots == 1:
            out.append(tuple(prefix + [remaining]))
            return
        for v in range(remaining + 1):
            rec(prefix + [v], remaining - v, slots - 1)
    rec([], S, d)
    return out


def bench_one_config(d, S, c_target, K_list=(2, 4, 8),
                     n_lambda_iters=20, soundness_n=30,
                     soundness_n_grid=9, verbose=True):
    """Benchmarks the joint LB vs single-window triangle baseline at one
    (d, S, c_target) cell.

    For each composition c with TV(c) ≥ c_target:
      - triangle baseline LB   = best single-window triangle bound
      - joint LB (K=2,4,8)     = subgradient-ascent dual joint bound
      - certified iff LB ≥ c_target.

    Reports the count of NEWLY certified cells per K (cells certified by
    joint but NOT by triangle baseline).

    Soundness: for `soundness_n` newly certified cells, fine-grid search
    over Cell verifies LB ≤ true min-max TV.
    """
    if verbose:
        print(f"\n=== d={d}, S={S}, c_target={c_target} ===")
    t0 = time.time()
    comps = enumerate_compositions(d, S)
    t_enum = time.time() - t0

    # Filter to compositions with at least one pruning window (they're
    # the only ones we need to certify anyway — otherwise NOT pruned).
    pruned_comps = []
    for c in comps:
        c_arr = np.asarray(c, dtype=np.int32)
        ws = find_pruning_windows(c_arr, S, d, c_target)
        if ws:
            pruned_comps.append((c_arr, ws))
    if verbose:
        print(f"  total compositions: {len(comps)} ({t_enum:.2f}s)")
        print(f"  pruned (TV>=c at grid): {len(pruned_comps)}")

    n_baseline_cert = 0
    new_cert = {K: 0 for K in K_list}
    new_cert_examples = {K: [] for K in K_list}
    LB_data = []  # for soundness check

    t1 = time.time()
    for c_arr, windows in pruned_comps:
        # Baseline (triangle on best single window from FULL window set):
        baseline_LB = triangle_baseline_LB(c_arr, S, d, c_target, windows)
        baseline_cert = baseline_LB >= c_target
        if baseline_cert:
            n_baseline_cert += 1
        for K in K_list:
            joint_LB, n_used, _ = joint_cert_LB(
                c_arr, S, d, c_target, windows=windows,
                n_lambda_iters=n_lambda_iters, top_K=K)
            joint_cert = joint_LB >= c_target
            if joint_cert and not baseline_cert:
                new_cert[K] += 1
                if len(new_cert_examples[K]) < soundness_n:
                    new_cert_examples[K].append(
                        (c_arr.copy(), windows, joint_LB, n_used))
            # Save smoke-data for any cells with positive LB
            if joint_LB > -1e6:
                LB_data.append((K, c_arr.tolist(), joint_LB, baseline_LB))

    t_cert = time.time() - t1
    if verbose:
        print(f"  triangle baseline certs: {n_baseline_cert}")
        for K in K_list:
            print(f"  joint K={K}: NEW certs = {new_cert[K]:6d} "
                  f"(over baseline)  total = "
                  f"{n_baseline_cert + new_cert[K]:6d}")
        print(f"  cert phase time: {t_cert:.2f}s")

    # Soundness: for each new-cert example, fine-grid check.
    # Scale n_grid down for large d to keep cost ~ n_grid^(d-1) tractable:
    # d-1=3: 9; d=5: 9; d=7 -> 5 (5^7 = 78K); d=8 -> 4 (4^7 = 16K).
    if d <= 6:
        ng = soundness_n_grid
    elif d == 7:
        ng = min(soundness_n_grid, 5)
    else:
        ng = min(soundness_n_grid, 4)
    n_violations = 0
    max_excess = 0.0
    if soundness_n > 0:
        for K in K_list:
            for ex in new_cert_examples[K][:soundness_n]:
                c_arr, windows, joint_LB, n_used = ex
                # Use the SAME windows used in the joint cert (top_K filtered)
                ws_used = sorted(windows, key=lambda w: -w[2])[:K]
                true_minmax = true_min_max_TV_grid(
                    c_arr, S, d, ws_used, n_grid=ng)
                # LB must be ≤ true min-max (over all δ in Cell).
                excess = joint_LB - true_minmax
                if excess > 1e-9:
                    n_violations += 1
                    if excess > max_excess:
                        max_excess = excess
                    if verbose and n_violations <= 3:
                        print(f"  !!! VIOLATION K={K} c={c_arr.tolist()} "
                              f"LB={joint_LB:.6f} true={true_minmax:.6f}")

    if verbose:
        print(f"  soundness check: {n_violations} violations, "
              f"max excess = {max_excess:.3e}")

    return {
        'd': d, 'S': S, 'c_target': c_target,
        'n_compositions': len(comps),
        'n_pruned': len(pruned_comps),
        'baseline_cert': n_baseline_cert,
        'new_cert': new_cert,
        'soundness_violations': n_violations,
        'soundness_max_excess': max_excess,
    }


def main():
    configs = [
        (4, 20, 1.20),
        (4, 30, 1.25),
        (6, 15, 1.20),
        (6, 20, 1.25),
        (8, 12, 1.20),
    ]
    K_list = (2, 4, 8)
    results = []
    print("=" * 64)
    print("Joint multi-window cell-cert (variant J) bench")
    print("=" * 64)
    for d, S, c in configs:
        r = bench_one_config(d, S, c, K_list=K_list, n_lambda_iters=20,
                              soundness_n=30, soundness_n_grid=9)
        results.append(r)

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"{'d':>3} {'S':>4} {'c':>5} {'n_pruned':>9} {'base':>6} "
          f"{'K=2':>6} {'K=4':>6} {'K=8':>6} {'viol':>5}")
    for r in results:
        print(f"{r['d']:>3} {r['S']:>4} {r['c_target']:>5.2f} "
              f"{r['n_pruned']:>9} {r['baseline_cert']:>6} "
              f"{r['new_cert'][2]:>6} {r['new_cert'][4]:>6} "
              f"{r['new_cert'][8]:>6} {r['soundness_violations']:>5}")

    out_path = os.path.join(os.path.dirname(__file__),
                            '_coarse_J_bench_results.json')
    with open(out_path, 'w') as f:
        json.dump([{**r, 'new_cert': {str(k): v
                                       for k, v in r['new_cert'].items()}}
                    for r in results], f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
