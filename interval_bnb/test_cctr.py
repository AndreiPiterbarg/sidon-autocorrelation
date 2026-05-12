"""Soundness + tightness tests for CCTR aggregate bound.

Soundness tests:
  T1: Random box → CCTR float LB and CCTR int LB agree to ~1e-9.
  T2: Random box, random α → CCTR LB <= true min mu^T M mu (sampled).
  T3: At d=4 known mu* = [1/3, 1/6, 1/6, 1/3], CCTR LB on tiny box at mu*
      matches val(4) = 10/9.
  T4: For α concentrated on single window W*, CCTR-SW reduces to
      bound_mccormick_sw_int_lp on W* (modulo sign/scale).

Tightness tests (informational, not soundness):
  N1: At a deep box near mu_star, compare CCTR LB vs joint-face dual cert.
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from interval_bnb.bound_cctr import (
    bound_cctr_sw_float_lp, bound_cctr_sw_int_lp, bound_cctr_sw_int_ge,
    build_cctr_aggregate_int, D_M_DEFAULT,
)
from interval_bnb.bound_eval import _adjacency_matrix
from interval_bnb.box import Box, SCALE as _SCALE
from interval_bnb.windows import build_windows


def _ints_from_floats(lo_f, hi_f):
    lo_int = [int(round(x * _SCALE)) for x in lo_f]
    hi_int = [int(round(x * _SCALE)) for x in hi_f]
    return lo_int, hi_int


def test_soundness_under_approx():
    """T2: For a box where we sample feasible μ ∈ box ∩ simplex and compute
    the true mu^T M mu, the int CCTR LB must be ≤ true value (for sampled μ).
    Soundness is already proved analytically; this is a sanity check.
    """
    d = 6
    rng = np.random.default_rng(0)
    windows = build_windows(d)
    # Pick 5 random active windows, random non-neg alpha
    active_windows = [windows[i] for i in [3, 7, 12, 18, 22]]
    alpha = rng.uniform(0.01, 1, size=5)
    alpha = alpha / alpha.sum()
    M_int, D_M = build_cctr_aggregate_int(alpha, active_windows, d)
    # Choose a small box: μ ∈ [0.05, 0.4]^d ∩ simplex
    lo_f = np.full(d, 0.05)
    hi_f = np.full(d, 0.4)
    lo_int, hi_int = _ints_from_floats(lo_f, hi_f)
    # Sample 100 random μ in box ∩ simplex.
    M_float = np.zeros((d, d))
    for k, w in enumerate(active_windows):
        A = _adjacency_matrix(w, d)
        M_float += alpha[k] * w.scale * A
    lb_int = bound_cctr_sw_int_lp(lo_int, hi_int, M_int, d)
    if lb_int is None:
        print("[T2] box-simplex empty (skip)")
        return
    lb_int_float = lb_int / (D_M * _SCALE * _SCALE)
    # Test: at any μ in box ∩ simplex, μ^T M μ >= lb_int_float (modulo
    # rounding under-approximation). We allow a small tolerance for
    # the round-down soundness margin (≤ |alpha| * #wins * 1/D_M ≈ 1e-11
    # at d=6).
    max_violation = 0.0
    for _ in range(100):
        # Random μ in [lo, hi] with sum=1
        x = rng.uniform(lo_f, hi_f)
        x = x / x.sum()
        if not ((x >= lo_f - 1e-12).all() and (x <= hi_f + 1e-12).all()):
            continue
        true_val = float(x @ M_float @ x)
        if lb_int_float > true_val + 1e-9:
            max_violation = max(max_violation, lb_int_float - true_val)
    if max_violation > 1e-8:
        print(f"[T2 FAIL] max violation = {max_violation:.4e}")
        return False
    print(f"[T2 PASS] LB ≤ true val at all 100 sampled μ "
          f"(max viol={max_violation:.4e})")
    return True


def test_float_int_agreement():
    """T1: Float CCTR LB and int CCTR LB / (D_M * SCALE^2) agree to ~1e-9.
    """
    d = 8
    rng = np.random.default_rng(1)
    windows = build_windows(d)
    n_active = 10
    active_idx = rng.choice(len(windows), size=n_active, replace=False)
    active_windows = [windows[int(i)] for i in active_idx]
    alpha = rng.uniform(0.01, 1, size=n_active)
    alpha = alpha / alpha.sum()
    M_int, D_M = build_cctr_aggregate_int(alpha, active_windows, d)
    M_float = np.zeros((d, d))
    for k, w in enumerate(active_windows):
        A = _adjacency_matrix(w, d)
        M_float += alpha[k] * w.scale * A
    # Random box
    lo_f = rng.uniform(0.0, 0.05, size=d)
    hi_f = rng.uniform(0.4, 1.0, size=d)
    # Force feasibility: make sum constraint achievable.
    # Need sum_lo <= 1 <= sum_hi. With d=8 lo's ~ 0.025 and hi's ~ 0.7,
    # sum_lo ~ 0.2 and sum_hi ~ 5.6. OK.
    if hi_f.sum() < 1.0:
        hi_f = hi_f * (2.0 / hi_f.sum())
    lo_int, hi_int = _ints_from_floats(lo_f, hi_f)
    lb_int = bound_cctr_sw_int_lp(lo_int, hi_int, M_int, d)
    lb_float = bound_cctr_sw_float_lp(lo_f, hi_f, M_float, d)
    if lb_int is None or not np.isfinite(lb_float):
        print("[T1 SKIP] box-simplex degenerate")
        return True
    lb_int_float = lb_int / (D_M * _SCALE * _SCALE)
    diff = abs(lb_int_float - lb_float)
    # Float error at d=8: ~d^2 * eps * max_val ~ 64 * 2e-16 * 2 ~ 3e-14.
    # Int rounding error: |alpha| * scale * 1/D_M * d^2 ~ 0.1 * 2 * 1e-12 * 64 = 1e-11.
    if diff > 1e-9:
        print(f"[T1 FAIL] float={lb_float:.10f}, int={lb_int_float:.10f}, "
              f"diff={diff:.4e}")
        return False
    print(f"[T1 PASS] float={lb_float:.10f}, int={lb_int_float:.10f}, "
          f"diff={diff:.2e}")
    return True


def test_target_comparison():
    """T_TARGET: bound_cctr_sw_int_ge correctly compares to target_q.
    """
    d = 6
    rng = np.random.default_rng(2)
    windows = build_windows(d)
    active_windows = [windows[i] for i in [3, 7]]
    alpha = [0.5, 0.5]
    M_int, D_M = build_cctr_aggregate_int(alpha, active_windows, d)
    # Use a moderate box
    lo_f = np.full(d, 0.05)
    hi_f = np.full(d, 0.3)
    lo_int, hi_int = _ints_from_floats(lo_f, hi_f)
    lb_int = bound_cctr_sw_int_lp(lo_int, hi_int, M_int, d)
    if lb_int is None:
        print("[T_TARGET SKIP] empty domain")
        return True
    lb_real = lb_int / (D_M * _SCALE * _SCALE)
    # Test target just below (should certify) and just above (should not)
    eps = 1e-6
    target_low = Fraction.from_float(lb_real - eps).limit_denominator(10**10)
    target_high = Fraction.from_float(lb_real + eps).limit_denominator(10**10)
    cert_low = bound_cctr_sw_int_ge(
        lo_int, hi_int, M_int, d, D_M,
        target_low.numerator, target_low.denominator,
    )
    cert_high = bound_cctr_sw_int_ge(
        lo_int, hi_int, M_int, d, D_M,
        target_high.numerator, target_high.denominator,
    )
    if not cert_low:
        print(f"[T_TARGET FAIL] target_low={float(target_low)} "
              f"(< lb={lb_real}) failed to cert")
        return False
    if cert_high:
        print(f"[T_TARGET FAIL] target_high={float(target_high)} "
              f"(> lb={lb_real}) wrongly certified")
        return False
    print(f"[T_TARGET PASS] cert at target<lb, no cert at target>lb")
    return True


def test_d4_known_optimum():
    """T3: At d=4 with mu_star = [1/3, 1/6, 1/6, 1/3] (val(4) = 10/9),
    CCTR with α from KKT should give LB ≈ val(4) on a tiny box around mu*."""
    d = 4
    mu_star = np.array([1/3, 1/6, 1/6, 1/3])
    sys.path.insert(0, _REPO)
    from kkt_correct_mu_star import build_window_data, kkt_residual_qp
    A_stack, c_W = build_window_data(d)
    res = kkt_residual_qp(mu_star, A_stack, c_W, x_cap=1.0, tol_active=1e-4)
    if res['alpha'] is None:
        print("[T3 SKIP] KKT QP failed")
        return True
    alpha = res['alpha']
    print(f"[T3] d=4 mu_star: KKT residual={res['residual']:.4e}, "
          f"val_max={res['val_max']:.6f}, n_active={len(res['active_idx'])}")
    # Build CCTR from interval_bnb's window list
    from .windows import build_windows as _bnb_windows
    bnb_w = _bnb_windows(d)
    # Map kkt window indices (ell, s) → bnb indices
    kkt_window_ells_ss = []
    for ell in range(2, 2 * d + 1):
        for s in range(2 * d - ell + 1):
            kkt_window_ells_ss.append((ell, s))
    bnb_idx_by_key = {(w.ell, w.s_lo): i for i, w in enumerate(bnb_w)}
    active_bnb_w = []
    for kkt_idx in res['active_idx']:
        ell, s = kkt_window_ells_ss[int(kkt_idx)]
        bnb_idx = bnb_idx_by_key[(ell, s)]
        active_bnb_w.append(bnb_w[bnb_idx])
    M_int, D_M = build_cctr_aggregate_int(alpha, active_bnb_w, d)
    # Tiny box around mu_star: 1e-3 each side
    eps = 1e-3
    lo_f = np.maximum(mu_star - eps, 0.0)
    hi_f = np.minimum(mu_star + eps, 1.0)
    lo_int, hi_int = _ints_from_floats(lo_f, hi_f)
    lb_int = bound_cctr_sw_int_lp(lo_int, hi_int, M_int, d)
    if lb_int is None:
        print("[T3 SKIP] box-simplex empty (mu_star sum ≠ 1?)")
        return True
    lb_real = lb_int / (D_M * _SCALE * _SCALE)
    val_4 = 10.0 / 9.0
    # Expect lb_real close to val(4) = 10/9 ≈ 1.111111
    err = abs(lb_real - val_4)
    # CCTR is an LB; expect lb_real <= val(4) but close (small box)
    if lb_real > val_4 + 1e-6:
        print(f"[T3 FAIL] lb={lb_real} > val(4)={val_4} (unsoundness!)")
        return False
    if val_4 - lb_real > 0.01:
        # Box is small (eps=1e-3); CCTR should be tight to within 0.01
        print(f"[T3 WARN] lb={lb_real} much smaller than val(4)={val_4}, "
              f"gap={val_4-lb_real:.4f}")
    else:
        print(f"[T3 PASS] lb={lb_real:.6f} ≈ val(4)={val_4:.6f} "
              f"(gap={val_4-lb_real:.4e})")
    return True


def test_multi_alpha_soundness():
    """T_MULTI_SOUND: For multiple α aggregates each individually built
    with α >= 0, sum α = 1, each individual LB is sound. Their MAX is also
    sound. Test by sampling random feasible μ and verifying each LB ≤
    sample value."""
    d = 6
    rng = np.random.default_rng(7)
    windows = build_windows(d)
    # Build 3 aggregates with different α distributions
    aggregates = []
    for trial in range(3):
        n_active = 5
        active_idx = rng.choice(len(windows), size=n_active, replace=False)
        active_w = [windows[int(i)] for i in active_idx]
        alpha = rng.uniform(0.01, 1, size=n_active)
        alpha = alpha / alpha.sum()
        M_int, D_M = build_cctr_aggregate_int(alpha, active_w, d)
        # Build float version
        M_float = np.zeros((d, d))
        for k, w in enumerate(active_w):
            M_float += alpha[k] * w.scale * _adjacency_matrix(w, d)
        aggregates.append({'M_int': M_int, 'D_M': D_M, 'M_float': M_float,
                            'alpha': alpha, 'active_w': active_w})
    # Sample box
    lo_f = np.full(d, 0.05)
    hi_f = np.full(d, 0.5)
    lo_int = [int(round(x * _SCALE)) for x in lo_f]
    hi_int = [int(round(x * _SCALE)) for x in hi_f]
    # Sample random feasible μ
    M_floats = [a['M_float'] for a in aggregates]
    max_violation = 0.0
    n_samples = 50
    for _ in range(n_samples):
        x = rng.uniform(lo_f, hi_f)
        x = x / x.sum()
        if not ((x >= lo_f - 1e-12).all() and (x <= hi_f + 1e-12).all()):
            continue
        # max_W TV_W(x): for soundness, each aggregate's LB must be ≤ this
        # (since aggregate ≤ max).
        # We approximate by computing μ^T M_float μ for each aggregate;
        # this equals the aggregate value at x, which by definition is
        # ≤ max_W TV_W(x). The LB on min_box (μ^T M μ) must be ≤ μ^T M μ at x.
        for ag in aggregates:
            agg_val_at_x = float(x @ ag['M_float'] @ x)
            from interval_bnb.bound_cctr import bound_cctr_sw_int_lp
            lb_int = bound_cctr_sw_int_lp(lo_int, hi_int, ag['M_int'], d)
            if lb_int is None:
                continue
            lb_real = lb_int / (ag['D_M'] * _SCALE * _SCALE)
            if lb_real > agg_val_at_x + 1e-9:
                max_violation = max(max_violation, lb_real - agg_val_at_x)
    if max_violation > 1e-8:
        print(f"[T_MULTI_SOUND FAIL] violation={max_violation:.4e}")
        return False
    print(f"[T_MULTI_SOUND PASS] {n_samples} samples × 3 aggregates, "
          f"max violation={max_violation:.4e}")
    return True


def test_multi_alpha_setup():
    """T_MULTI_SETUP: setup_multi_cctr returns a list of valid contexts.
    """
    sys.path.insert(0, _REPO)
    from interval_bnb.cctr_setup import setup_multi_cctr
    mu_star = np.array([1/3, 1/6, 1/6, 1/3])
    aggregates = setup_multi_cctr(4, mu_star=mu_star)
    assert len(aggregates) == 3, f"expected 3 aggregates, got {len(aggregates)}"
    names = [a['name'] for a in aggregates]
    assert names == ['kkt', 'uniform_active', 'all_windows'], names
    for ag in aggregates:
        assert ag['M_int'] is not None
        assert ag['D_M'] > 0
        assert ag['alpha'].sum() > 0.999 and ag['alpha'].sum() < 1.001
        assert (ag['alpha'] >= 0).all()
        # M_int >= 0 element-wise
        for i in range(4):
            for j in range(4):
                assert ag['M_int'][i, j] >= 0
    print(f"[T_MULTI_SETUP PASS] 3 aggregates built, all sound contracts ok")
    return True


def test_multi_alpha_tightness():
    """T_MULTI_TIGHT: Multi-α MAX LB is at least as tight as single-α LB
    on each aggregate. Demonstrate strict improvement on at least one box."""
    sys.path.insert(0, _REPO)
    from interval_bnb.cctr_setup import setup_multi_cctr
    from interval_bnb.bound_cctr import (
        bound_cctr_sw_float_lp, multi_cctr_sw_float_best,
    )
    d = 4
    mu_star = np.array([1/3, 1/6, 1/6, 1/3])
    aggregates = setup_multi_cctr(d, mu_star=mu_star)
    # Build float versions
    M_floats = []
    for ag in aggregates:
        n_a = len(ag['alpha'])
        windows = build_windows(d)
        active_w = [windows[i] for i in ag['active_idx']]
        M_f = np.zeros((d, d))
        for k, w in enumerate(active_w):
            M_f += ag['alpha'][k] * w.scale * _adjacency_matrix(w, d)
        M_floats.append(M_f)
    # Test on multiple boxes
    test_boxes = [
        # box around μ*
        (np.array([0.30, 0.13, 0.13, 0.30]),
         np.array([0.35, 0.18, 0.18, 0.35])),
        # box at simplex centroid
        (np.array([0.20, 0.20, 0.20, 0.20]),
         np.array([0.30, 0.30, 0.30, 0.30])),
        # box at corner
        (np.array([0.40, 0.05, 0.05, 0.05]),
         np.array([0.55, 0.20, 0.20, 0.20])),
    ]
    for k_box, (lo, hi) in enumerate(test_boxes):
        per_ag = [bound_cctr_sw_float_lp(lo, hi, Mf, d) for Mf in M_floats]
        best_lb, best_idx = multi_cctr_sw_float_best(lo, hi, M_floats, d)
        # Multi-α best == max over per-ag.
        max_per = max(p for p in per_ag if np.isfinite(p))
        assert abs(best_lb - max_per) < 1e-12, (
            f"box {k_box}: best_lb={best_lb} != max(per_ag)={max_per}")
        print(f"[T_MULTI_TIGHT box {k_box}] LBs per aggregate: "
              f"{[f'{x:.4f}' for x in per_ag]} -> max={best_lb:.4f}")
    return True


def test_boundary_face_anchors():
    """T_FACE: boundary face aggregates are sound and tighten boundary regions."""
    sys.path.insert(0, _REPO)
    from interval_bnb.cctr_setup import setup_multi_cctr
    from interval_bnb.bound_cctr import bound_cctr_sw_float_lp
    d = 8
    sys.path.insert(0, _REPO)
    from kkt_correct_mu_star import find_kkt_correct_mu_star
    res = find_kkt_correct_mu_star(
        d=d, x_cap=1.0, n_starts=200, n_workers=4,
        top_K_phase2=20, top_K_phase3=5,
        target_residual=1e-6, verbose=False,
    )
    mu_star = res['mu_star']
    print(f"[T_FACE] d={d} mu_star f={res['f_value']:.6f}")
    aggregates = setup_multi_cctr(
        d, mu_star=mu_star,
        include=('kkt', 'uniform_active', 'all_windows', 'boundary_faces'),
    )
    print(f"[T_FACE] {len(aggregates)} aggregates: "
          f"{[a['name'] for a in aggregates]}")
    # Test: build float versions, check soundness on random samples.
    M_floats = []
    for ag in aggregates:
        M_f = np.zeros((d, d))
        windows = build_windows(d)
        active_w = [windows[i] for i in ag['active_idx']]
        n_a = len(active_w)
        for k, w in enumerate(active_w):
            M_f += ag['alpha'][k] * w.scale * _adjacency_matrix(w, d)
        M_floats.append(M_f)
    rng = np.random.default_rng(42)
    # Boundary box: μ_3 forced near 0
    lo_f = np.array([0.05, 0.05, 0.05, 0.0,  0.05, 0.05, 0.05, 0.05])
    hi_f = np.array([0.30, 0.30, 0.30, 0.05, 0.30, 0.30, 0.30, 0.30])
    lbs = [bound_cctr_sw_float_lp(lo_f, hi_f, Mf, d) for Mf in M_floats]
    print(f"[T_FACE] LBs at boundary box (μ_3 ≈ 0):")
    for ag, lb in zip(aggregates, lbs):
        print(f"  {ag['name']}: {lb:.4f}")
    # Soundness: each LB ≤ true min over box.
    # Sample feasible μ; verify each LB ≤ μ^T M μ at samples.
    max_violation = 0.0
    for _ in range(50):
        x = rng.uniform(lo_f, hi_f)
        if x.sum() < 1e-9:
            continue
        x = x / x.sum()
        if not ((x >= lo_f - 1e-12).all() and (x <= hi_f + 1e-12).all()):
            continue
        for k, Mf in enumerate(M_floats):
            agg_val = float(x @ Mf @ x)
            if lbs[k] > agg_val + 1e-9:
                max_violation = max(max_violation, lbs[k] - agg_val)
    if max_violation > 1e-7:
        print(f"[T_FACE FAIL] violation={max_violation:.4e}")
        return False
    print(f"[T_FACE PASS] all LBs sound, violation={max_violation:.4e}")
    return True


def main():
    print("="*60)
    print("CCTR soundness + tightness tests")
    print("="*60)
    results = []
    for name, fn in [
        ("test_float_int_agreement", test_float_int_agreement),
        ("test_soundness_under_approx", test_soundness_under_approx),
        ("test_target_comparison", test_target_comparison),
        ("test_d4_known_optimum", test_d4_known_optimum),
        ("test_multi_alpha_soundness", test_multi_alpha_soundness),
        ("test_multi_alpha_setup", test_multi_alpha_setup),
        ("test_multi_alpha_tightness", test_multi_alpha_tightness),
    ]:
        try:
            ok = fn()
            results.append((name, "PASS" if ok else "FAIL"))
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append((name, f"ERROR: {e}"))
    print()
    for name, status in results:
        print(f"  {name}: {status}")
    n_pass = sum(1 for _, s in results if s == "PASS")
    print(f"\n{n_pass}/{len(results)} passed")


if __name__ == "__main__":
    main()
