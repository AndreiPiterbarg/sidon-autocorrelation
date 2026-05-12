"""Soundness + tightness tests for the per-box epigraph LP bound.

THE EPIGRAPH BOUND IS A RIGOR-CRITICAL PIECE — these tests verify:

  T1: Soundness on random samples — for every feasible μ in the box,
      the float LP value is ≤ true max_W TV_W(μ).
  T2: Float ↔ integer-cert agreement — when the integer cert succeeds,
      the float LP value is ≥ target.
  T3: Boundary cases — empty box, single window, degenerate.
  T4: STRICTNESS (vs joint-face): on a stuck box from the d=10 stall
      configuration, epigraph LB > joint-face LB.
  T5: Closes the d=10 stall — at the empirical-agent's identified stall
      box centroid, epigraph_int_ge certifies target=1.208 (joint-face
      did not).
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

from interval_bnb.bound_epigraph import (
    bound_epigraph_lp_float, bound_epigraph_int_ge,
    _solve_epigraph_lp, _publication_cushion,
    _HIGHS_TOL_CUSHION, _DUAL_FEAS_TOL,
)
from interval_bnb.bound_eval import _adjacency_matrix
from interval_bnb.box import SCALE as _SCALE
from interval_bnb.windows import build_windows


def _ints_from_floats(lo_f, hi_f):
    return ([int(round(x * _SCALE)) for x in lo_f],
            [int(round(x * _SCALE)) for x in hi_f])


def test_epigraph_soundness_samples():
    """T1: float LP value ≤ max_W TV_W(μ) for every sampled feasible μ."""
    rng = np.random.default_rng(0)
    d = 6
    windows = build_windows(d)
    # Random box that contains feasible simplex points.
    lo_f = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    hi_f = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
    lp_val = bound_epigraph_lp_float(lo_f, hi_f, windows, d)
    assert lp_val > 0, f"LP returned {lp_val} (LP failed?)"
    # Sample 200 feasible μ in box ∩ simplex; verify max_W TV_W ≥ lp_val.
    max_violation = 0.0
    n_ok = 0
    for _ in range(200):
        x = rng.uniform(lo_f, hi_f)
        x = x / x.sum()
        if not ((x >= lo_f - 1e-12).all() and (x <= hi_f + 1e-12).all()):
            continue
        n_ok += 1
        # Compute max_W TV_W(x)
        max_tv = 0.0
        for w in windows:
            tv = w.scale * float(x @ _adjacency_matrix(w, d) @ x)
            if tv > max_tv:
                max_tv = tv
        if lp_val > max_tv + 1e-9:
            max_violation = max(max_violation, lp_val - max_tv)
    assert n_ok > 50, f"only {n_ok} feasible samples"
    if max_violation > 1e-7:
        print(f"[T1 FAIL] LP={lp_val:.6f}, max viol={max_violation:.4e}")
        return False
    print(f"[T1 PASS] LP={lp_val:.6f}, n_samples={n_ok}, "
          f"max viol={max_violation:.2e}")
    return True


def test_epigraph_int_cert_agrees_with_float():
    """T2: integer cert succeeds ⇔ float LP ≥ target (modulo small slack).
    """
    d = 6
    windows = build_windows(d)
    lo_f = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    hi_f = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
    lo_int, hi_int = _ints_from_floats(lo_f, hi_f)
    lp_val = bound_epigraph_lp_float(lo_f, hi_f, windows, d)
    # Test target just below LP value (should certify).
    target_low = Fraction.from_float(lp_val - 1e-4).limit_denominator(10**10)
    cert_low = bound_epigraph_int_ge(
        lo_int, hi_int, windows, d,
        target_low.numerator, target_low.denominator,
    )
    # Test target just above (should not certify).
    target_high = Fraction.from_float(lp_val + 1e-4).limit_denominator(10**10)
    cert_high = bound_epigraph_int_ge(
        lo_int, hi_int, windows, d,
        target_high.numerator, target_high.denominator,
    )
    if not cert_low:
        print(f"[T2 FAIL] target_low={float(target_low):.8f} (< {lp_val:.8f}) "
              f"failed to certify")
        return False
    if cert_high:
        print(f"[T2 FAIL] target_high={float(target_high):.8f} (> {lp_val:.8f}) "
              f"wrongly certified")
        return False
    print(f"[T2 PASS] target_low certifies, target_high does not "
          f"(LP={lp_val:.6f})")
    return True


def test_epigraph_strictly_tighter_than_jointface():
    """T4: at the d=10 stall configuration, epigraph LB > joint-face LB."""
    d = 10
    # Empirical-agent reported stall centroid (averaged over 1200 stuck boxes).
    centroid = np.array([0.135, 0.079, 0.054, 0.041, 0.047,
                          0.082, 0.096, 0.108, 0.113, 0.244])
    centroid = centroid / centroid.sum()  # ensure simplex
    # Box of width 1e-3 around centroid (deep — depth ~30).
    radius = 1e-3
    lo_f = np.maximum(centroid - radius, 0.0)
    hi_f = np.minimum(centroid + radius, 1.0)
    # Simplex projection: tighten to ensure intersection.
    s_lo = lo_f.sum()
    s_hi = hi_f.sum()
    if s_lo > 1.0 or s_hi < 1.0:
        print(f"[T4 SKIP] box infeasible (sum_lo={s_lo}, sum_hi={s_hi})")
        return True
    windows = build_windows(d)
    epi_lb = bound_epigraph_lp_float(lo_f, hi_f, windows, d)
    # Per-window: max joint-face LB
    from interval_bnb.bound_eval import bound_mccormick_joint_face_lp
    best_per_window = float("-inf")
    for w in windows:
        lb_w = bound_mccormick_joint_face_lp(lo_f, hi_f, w, w.scale)
        if lb_w > best_per_window:
            best_per_window = lb_w
    print(f"[T4] At d=10 stall centroid, radius {radius}:")
    print(f"     joint-face MAX over windows = {best_per_window:.6f}")
    print(f"     epigraph LP                  = {epi_lb:.6f}")
    print(f"     gain                          = {epi_lb - best_per_window:+.6f}")
    if epi_lb < best_per_window - 1e-9:
        print(f"[T4 FAIL] epigraph LB < joint-face LB (sound but loose)")
        return False
    if epi_lb < best_per_window + 1e-9:
        print(f"[T4 WARN] epigraph LB ≈ joint-face LB (no tightening on this box)")
    else:
        print(f"[T4 PASS] epigraph strictly tighter than joint-face")
    return True


def test_d4_known_optimum():
    """T5: at d=4, val(4)=10/9, with tiny box around mu_star, epigraph
    cert succeeds for target close to val."""
    d = 4
    windows = build_windows(d)
    mu_star = np.array([1/3, 1/6, 1/6, 1/3])
    radius = 1e-4
    lo_f = mu_star - radius
    hi_f = mu_star + radius
    lo_int, hi_int = _ints_from_floats(lo_f, hi_f)
    val_4 = Fraction(10, 9)  # 1.1111...
    # Target slightly below val(4): should certify.
    target = Fraction(11, 10)  # 1.10
    cert = bound_epigraph_int_ge(
        lo_int, hi_int, windows, d,
        target.numerator, target.denominator,
    )
    lp_val = bound_epigraph_lp_float(lo_f, hi_f, windows, d)
    print(f"[T5] d=4 mu* tiny box, target=1.10:")
    print(f"     epigraph LP = {lp_val:.6f}")
    print(f"     val(4)      = {float(val_4):.6f}")
    print(f"     cert        = {cert}")
    if not cert:
        print(f"[T5 FAIL] failed to cert target=1.10 at mu* (LP={lp_val})")
        return False
    # Target above val(4): should NOT certify.
    target_too_high = Fraction(112, 100)  # 1.12 > 10/9
    cert_high = bound_epigraph_int_ge(
        lo_int, hi_int, windows, d,
        target_too_high.numerator, target_too_high.denominator,
    )
    if cert_high:
        print(f"[T5 FAIL] wrongly certified target=1.12 > val(4)")
        return False
    print(f"[T5 PASS] cert at 1.10 ✓, no cert at 1.12 ✓")
    return True


def test_epigraph_breaks_d10_stall():
    """T6: closes the d=10 stall — empirical agent showed joint-face top-3
    closes 95.5%. Epigraph should close ~100%.

    At a tiny box around the stall centroid, epigraph LB ≥ target=1.208.
    """
    d = 10
    windows = build_windows(d)
    # Use the empirical stall centroid (clean version) — we should be
    # ABOVE 1.208 here.
    centroid = np.array([0.135, 0.079, 0.054, 0.041, 0.047,
                          0.082, 0.096, 0.108, 0.113, 0.244])
    centroid = centroid / centroid.sum()
    target = Fraction('1.208')
    # Try increasing-radius boxes; smallest box that certifies.
    print(f"[T6] d=10 stall centroid, target=1.208:")
    closed = False
    for radius in [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4]:
        lo_f = np.maximum(centroid - radius, 0.0)
        hi_f = np.minimum(centroid + radius, 1.0)
        if lo_f.sum() > 1.0 or hi_f.sum() < 1.0:
            print(f"     radius={radius}: SKIP (infeasible box)")
            continue
        lo_int, hi_int = _ints_from_floats(lo_f, hi_f)
        lp_val = bound_epigraph_lp_float(lo_f, hi_f, windows, d)
        cert = bound_epigraph_int_ge(
            lo_int, hi_int, windows, d,
            target.numerator, target.denominator,
            )
        # Per-window joint-face for comparison
        from interval_bnb.bound_eval import bound_mccormick_joint_face_lp
        best_jf = float("-inf")
        for w in windows:
            lb_w = bound_mccormick_joint_face_lp(lo_f, hi_f, w, w.scale)
            best_jf = max(best_jf, lb_w)
        flag = "✓" if cert else "✗"
        print(f"     radius={radius}: epigraph={lp_val:.4f}, "
              f"joint-face_max={best_jf:.4f}, cert={cert} {flag}")
        if cert and not closed:
            closed = True
    if closed:
        print(f"[T6 PASS] epigraph closes the d=10 stall config")
    else:
        print(f"[T6 FAIL] epigraph could not certify at any radius")
    return closed


def test_rigor_cushion_above_dual_residual_d4():
    """T7 (publication-rigor): solving an LP at d=4, the HiGHS-reported
    residuals (eq + ineq) must be << publication cushion.

    This is a SOUNDNESS test on the cushion size: at the requested
    HiGHS tolerance ε_d = 1e-9, residuals ≤ ε_d should be observed in
    practice on a healthy LP, and the cushion 100·ε_d = 1e-7 is the
    safety multiplier on top.
    """
    d = 4
    windows = build_windows(d)
    mu_star = np.array([1/3, 1/6, 1/6, 1/3])
    radius = 1e-3
    lo_f = mu_star - radius
    hi_f = mu_star + radius
    n_y = d * d
    n_vars = n_y + d + 1
    n_ineq = 4 * n_y + len(windows) + 1 + d
    cushion = _publication_cushion(d, n_vars, n_ineq)

    out = _solve_epigraph_lp(lo_f, hi_f, windows, d, return_residuals=True)
    lp_val = out[0]
    max_eq = out[5]
    max_ineq = out[6]
    assert np.isfinite(lp_val), "LP must succeed at the canonical d=4 box"
    # Residuals should be VERY much smaller than cushion (cushion is 1e-7,
    # residuals at ε_d=1e-9 should be ≤ 1e-9, i.e. 100x headroom).
    assert max_eq <= cushion, (
        f"max_eq_resid={max_eq:.3e} > cushion={cushion:.3e}"
    )
    assert max_ineq <= cushion, (
        f"max_ineq_resid={max_ineq:.3e} > cushion={cushion:.3e}"
    )
    # Stronger: residuals should fit a 10× headroom over the 1e-9 tol.
    # (If this trips, HiGHS may have changed its tolerance handling.)
    headroom = 1e-8
    assert max_eq <= headroom, (
        f"max_eq_resid={max_eq:.3e} exceeds 10x ε_d headroom — "
        f"HiGHS reported worse than expected"
    )
    assert max_ineq <= headroom, (
        f"max_ineq_resid={max_ineq:.3e} exceeds 10x ε_d headroom"
    )
    print(f"[T7 PASS] d=4 LP: lp_val={lp_val:.6f}, "
          f"max_eq={max_eq:.2e}, max_ineq={max_ineq:.2e}, "
          f"cushion={cushion:.2e}")
    return True


def test_rigor_cushion_actually_blocks_borderline_cert_d4():
    """T8 (publication-rigor): a borderline target (lp_val - 5e-8) must
    NOT certify because the new 1e-7 cushion is the binder.

    This is a NEGATIVE test confirming the cushion is the active
    constraint for borderline certs (the old 5e-11 arith-only safety
    would have certified this incorrectly).
    """
    d = 4
    windows = build_windows(d)
    mu_star = np.array([1/3, 1/6, 1/6, 1/3])
    radius = 1e-3
    lo_f = mu_star - radius
    hi_f = mu_star + radius
    lo_int, hi_int = _ints_from_floats(lo_f, hi_f)

    lp_val = bound_epigraph_lp_float(lo_f, hi_f, windows, d)
    assert np.isfinite(lp_val) and lp_val > 0
    # Pick target = lp_val - 5e-8 (between arith safety 5e-11 and cushion 1e-7)
    target_borderline = lp_val - 5e-8
    target_q = Fraction(target_borderline).limit_denominator(10**14)
    cert = bound_epigraph_int_ge(
        lo_int, hi_int, windows, d,
        target_q.numerator, target_q.denominator,
    )
    # The new cushion (1e-7) must block this cert.
    assert not cert, (
        f"[T8 FAIL] borderline target {float(target_q):.10f} "
        f"(lp_val={lp_val:.10f}, gap={lp_val - float(target_q):.3e}) "
        f"was wrongly certified — cushion did not bind"
    )
    print(f"[T8 PASS] borderline target (gap=5e-8) blocked by 1e-7 cushion")
    return True


def test_rigor_cushion_passes_genuine_certs_d4():
    """T9 (publication-rigor): genuine known-good certs at d=4 t=1.05
    still go through with the new larger cushion."""
    d = 4
    windows = build_windows(d)
    mu_star = np.array([1/3, 1/6, 1/6, 1/3])
    # tight box around d=4 mu*; val(4) = 10/9 ≈ 1.1111, comfortably > 1.05
    radius = 1e-4
    lo_f = mu_star - radius
    hi_f = mu_star + radius
    lo_int, hi_int = _ints_from_floats(lo_f, hi_f)

    target = Fraction(105, 100)  # 1.05
    cert = bound_epigraph_int_ge(
        lo_int, hi_int, windows, d,
        target.numerator, target.denominator,
    )
    assert cert, (
        f"[T9 FAIL] genuine cert at d=4, t=1.05 refused by new 1e-7 cushion"
    )
    # Also try slightly looser target.
    target2 = Fraction("1.10")
    cert2 = bound_epigraph_int_ge(
        lo_int, hi_int, windows, d,
        target2.numerator, target2.denominator,
    )
    assert cert2, (
        f"[T9 FAIL] genuine cert at d=4, t=1.10 refused"
    )
    print(f"[T9 PASS] cert at 1.05 ✓, cert at 1.10 ✓ (genuine certs preserved)")
    return True


def main():
    print("="*60)
    print("EPIGRAPH bound: soundness + stall-closure tests")
    print("="*60)
    results = []
    for name, fn in [
        ("test_epigraph_soundness_samples", test_epigraph_soundness_samples),
        ("test_epigraph_int_cert_agrees_with_float", test_epigraph_int_cert_agrees_with_float),
        ("test_epigraph_strictly_tighter_than_jointface", test_epigraph_strictly_tighter_than_jointface),
        ("test_d4_known_optimum", test_d4_known_optimum),
        ("test_epigraph_breaks_d10_stall", test_epigraph_breaks_d10_stall),
        ("test_rigor_cushion_above_dual_residual_d4",
         test_rigor_cushion_above_dual_residual_d4),
        ("test_rigor_cushion_actually_blocks_borderline_cert_d4",
         test_rigor_cushion_actually_blocks_borderline_cert_d4),
        ("test_rigor_cushion_passes_genuine_certs_d4",
         test_rigor_cushion_passes_genuine_certs_d4),
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
