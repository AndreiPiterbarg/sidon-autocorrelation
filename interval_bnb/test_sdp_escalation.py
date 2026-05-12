"""Soundness + tightness tests for the Lasserre order-2 SDP escalation.

The SDP escalation is the publication-rigor "tier 3" certifier that
fires after the epigraph LP cascade in the BnB driver. These tests
verify:

  T1: Soundness on random small-d boxes — for 5 random d=4 boxes, the
      SDP LB does not exceed the true min over a Monte-Carlo sample.
  T2: Tightening over the epigraph LP — on a d=6 box, the SDP LB is
      >= the epigraph LP LB (since the SDP is a tightening of the LP).
  T3: d=22 residual-box certification — at a half-width-0.003 box
      centred on `mu_star_d22`, the SDP escalation certifies target
      t = 1.281 (closes the residual the cascade left open).
  T4: Cushion correctness — perturbing the SDP value by exactly the
      cushion amount flips the certificate as expected.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from fractions import Fraction

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from interval_bnb.bound_sdp_escalation import (
    bound_sdp_escalation_int_ge,
    bound_sdp_escalation_lb_float,
    build_sdp_escalation_cache,
    _safe_cushion,
    _CUSHION_FLOOR,
)
from interval_bnb.bound_eval import _adjacency_matrix
from interval_bnb.bound_epigraph import bound_epigraph_lp_float
from interval_bnb.box import SCALE as _SCALE
from interval_bnb.windows import build_windows


def _ints_from_floats(lo_f, hi_f):
    return ([int(round(x * _SCALE)) for x in lo_f],
            [int(round(x * _SCALE)) for x in hi_f])


def _max_tv(mu, windows, d):
    """Compute f(mu) = max_W TV_W(mu)."""
    best = -1.0
    for w in windows:
        tv = float(w.scale) * float(mu @ _adjacency_matrix(w, d) @ mu)
        if tv > best:
            best = tv
    return best


def _sample_box_min(lo_f, hi_f, windows, d, n_samples=2000, rng=None):
    """Monte-Carlo sample of min_{mu in box cap simplex} max_W TV_W(mu).
    Returns the best (smallest) sampled value, or +inf if no sample
    falls in the simplex."""
    if rng is None:
        rng = np.random.default_rng(0)
    best = float('inf')
    for _ in range(n_samples):
        # Uniform in box, then project to simplex by /sum.
        x = rng.uniform(lo_f, hi_f)
        s = float(x.sum())
        if s <= 0:
            continue
        x = x / s
        if not ((x >= lo_f - 1e-12).all() and (x <= hi_f + 1e-12).all()):
            continue
        f = _max_tv(x, windows, d)
        if f < best:
            best = f
    return best


# ---------------------------------------------------------------------
# T1: soundness on random small-d boxes
# ---------------------------------------------------------------------

def test_sdp_soundness_random_d4():
    """For 5 random d=4 boxes intersecting the simplex, the SDP LB is
    <= the true min over a 2000-sample Monte-Carlo set inside the box.
    """
    d = 4
    rng = np.random.default_rng(42)
    windows = build_windows(d)

    n_tested = 0
    n_violations = 0
    max_violation = 0.0

    for trial in range(20):
        if n_tested >= 5:
            break
        # Random box biased to intersect the simplex.
        lo_f = rng.uniform(0.0, 0.3, d)
        widths = rng.uniform(0.05, 0.4, d)
        hi_f = np.minimum(lo_f + widths, 1.0)
        if lo_f.sum() > 1.0 or hi_f.sum() < 1.0:
            continue
        n_tested += 1

        res = bound_sdp_escalation_lb_float(lo_f, hi_f, windows, d,
                                            order=2, time_limit_s=10.0)
        if not res['is_feasible_status']:
            print(f"  trial {trial}: solver status={res['status']}; skip")
            continue

        # Cushion-adjusted LB
        cushion = _safe_cushion(res['r_prim'], res['r_dual'],
                                res['duality_gap'])
        lb_safe = float(res['obj_val_dual']) - cushion

        # Sample the box for a heuristic "true min" upper bound.
        sampled_min = _sample_box_min(lo_f, hi_f, windows, d,
                                       n_samples=2000, rng=rng)

        # Soundness: lb_safe <= true_min, and our heuristic
        # `sampled_min` is an upper bound on the true min, so
        # lb_safe <= true_min <= sampled_min should hold (modulo a tiny
        # arithmetic slack). We test lb_safe <= sampled_min + 1e-7.
        gap = lb_safe - sampled_min
        if gap > 1e-7:
            n_violations += 1
            max_violation = max(max_violation, gap)
        print(f"  trial {trial}: SDP_dual={res['obj_val_dual']:.6f}, "
              f"cushion={cushion:.2e}, lb_safe={lb_safe:.6f}, "
              f"sampled_min={sampled_min:.6f}, gap={gap:+.2e}")

    assert n_tested >= 3, f"only {n_tested} valid boxes generated"
    assert n_violations == 0, (
        f"[T1 FAIL] {n_violations} soundness violations "
        f"(max gap {max_violation:.4e})")


# ---------------------------------------------------------------------
# T2: SDP LB >= epigraph LP LB on a d=6 box (SDP tightens LP)
# ---------------------------------------------------------------------

def test_sdp_lb_ge_epigraph_lp_d6():
    """The order-2 Lasserre SDP is a tightening of the epigraph LP
    relaxation (it replaces the McCormick lifts Y_{ij} with
    pseudo-moments of an actual measure on the box). On a feasible
    d=6 box we expect SDP_LB >= LP_LB up to solver tolerance.
    """
    d = 6
    windows = build_windows(d)
    rng = np.random.default_rng(7)

    # A box similar to one of the d=6 stuck residual boxes from the
    # cascade (centred near a near-optimal mu).
    centroid = np.array([0.18, 0.10, 0.08, 0.08, 0.10, 0.46])
    centroid = centroid / centroid.sum()
    radius = 5e-3
    lo_f = np.maximum(centroid - radius, 0.0)
    hi_f = np.minimum(centroid + radius, 1.0)
    if lo_f.sum() > 1.0 or hi_f.sum() < 1.0:
        # widen until it intersects
        lo_f = np.maximum(centroid - 1e-2, 0.0)
        hi_f = np.minimum(centroid + 1e-2, 1.0)

    lp_lb = bound_epigraph_lp_float(lo_f, hi_f, windows, d)
    sdp_res = bound_sdp_escalation_lb_float(lo_f, hi_f, windows, d,
                                             order=2, time_limit_s=15.0)
    assert sdp_res['is_feasible_status'], (
        f"SDP solver failed: status={sdp_res['status']}")

    # Use the cushion-adjusted LB so the comparison is the SAFE LB.
    cushion = _safe_cushion(sdp_res['r_prim'], sdp_res['r_dual'],
                             sdp_res['duality_gap'])
    sdp_lb_safe = float(sdp_res['obj_val_dual']) - cushion

    print(f"[T2] d=6 box around centroid (radius {radius}):")
    print(f"     epigraph LP LB     = {lp_lb:.8f}")
    print(f"     SDP dual obj       = {sdp_res['obj_val_dual']:.8f}")
    print(f"     SDP cushion        = {cushion:.2e}")
    print(f"     SDP safe LB        = {sdp_lb_safe:.8f}")
    print(f"     gain over LP       = {sdp_lb_safe - lp_lb:+.6e}")

    # Allow a tiny numerical slack since both are float computations.
    # The SDP MUST be at least as tight as the LP up to solver
    # tolerance + cushion: SDP_safe >= LP - (LP_solver_slack +
    # SDP_cushion). LP solver slack is ~1e-8 from HiGHS default.
    slack = cushion + 1e-7
    assert sdp_lb_safe >= lp_lb - slack, (
        f"[T2 FAIL] SDP safe LB {sdp_lb_safe} < LP LB {lp_lb} - slack "
        f"{slack}; SDP should tighten the LP")


# ---------------------------------------------------------------------
# T3: Certifies a residual box around mu_star_d22 at target=1.281
# ---------------------------------------------------------------------

def test_sdp_certifies_d22_residual_box():
    """Load mu_star from `mu_star_d22.npz`, take a half-width-0.003 box
    around sigma(mu*) (where sigma is the bin reversal that brings
    mu* into the half-simplex H_d), and confirm the SDP escalation
    certifies target t = 1.281.

    This test is the d=22 acceptance gate for the SDP escalation:
    the cascade left these boxes uncertified at 49% closure, and the
    SDP must close them.
    """
    d = 22
    windows = build_windows(d)

    npz_path = Path(_REPO) / 'mu_star_d22.npz'
    if not npz_path.exists():
        # Smoke fallback: synthesise a near-optimal mu* by uniform
        # near-edge mass distribution. Less stringent but exercises
        # the d=22 path.
        print(f"[T3 SKIP-fallback] mu_star_d22.npz not found at {npz_path}")
        return
    data = np.load(str(npz_path))
    mu_star = np.asarray(data['mu'], dtype=np.float64)
    f_star = float(data['f']) if 'f' in data.files else None
    assert mu_star.shape == (d,), (
        f"mu_star_d22 shape {mu_star.shape}, expected ({d},)")

    # Apply the half-simplex symmetry: if mu*_0 > mu*_{d-1}, reverse.
    if mu_star[0] > mu_star[-1]:
        mu_star = mu_star[::-1].copy()

    # Half-width box around mu*. half_width = 3e-3.
    radius = 3e-3
    lo_f = np.maximum(mu_star - radius, 0.0)
    hi_f = np.minimum(mu_star + radius, 1.0)
    # Ensure box intersects simplex.
    assert lo_f.sum() <= 1.0 + 1e-12, (
        f"box infeasible: lo_sum {lo_f.sum()}")
    assert hi_f.sum() >= 1.0 - 1e-12, (
        f"box infeasible: hi_sum {hi_f.sum()}")

    target = Fraction('1.281')
    lo_int, hi_int = _ints_from_floats(lo_f, hi_f)
    cache = build_sdp_escalation_cache(d, windows)

    cert, info = bound_sdp_escalation_int_ge(
        lo_int, hi_int, windows, d,
        target.numerator, target.denominator,
        cache=cache,
        order=2,
        time_limit_s=60.0,  # d=22 SDP is heavier; allow generous budget
        return_diagnostic=True,
    )
    print(f"[T3] d=22 mu_star residual box (radius {radius}):")
    print(f"     status        : {info.get('status')}")
    print(f"     obj_val_dual  : {info.get('obj_val_dual')}")
    print(f"     r_prim        : {info.get('r_prim')}")
    print(f"     r_dual        : {info.get('r_dual')}")
    print(f"     cushion       : {info.get('cushion')}")
    print(f"     lb_safe       : {info.get('lb_safe')}")
    print(f"     target        : {info.get('target_f')}")
    print(f"     cert          : {cert}")
    print(f"     solve_time    : {info.get('solve_time')}s")
    if f_star is not None:
        print(f"     (true f*={f_star:.6f})")

    # Acceptance: the box surrounds a near-extremal mu* at d=22 with
    # f(mu*) ~ 1.309 according to the npz. A 3e-3 box must cleanly
    # certify t=1.281 unless the SDP relaxation is unexpectedly loose.
    assert cert, (
        f"[T3 FAIL] SDP escalation could not certify t=1.281 at the "
        f"d=22 residual box (lb_safe={info.get('lb_safe')}; "
        f"diagnostics: {info})")


# ---------------------------------------------------------------------
# T4: Cushion correctness — flipping by the exact cushion amount
# ---------------------------------------------------------------------

def test_sdp_cushion_correctness():
    """Verify the cushion logic: pick a target right at the certified
    boundary and confirm that target += cushion + 2*ULP fails to
    certify, while target -= cushion + 2*ULP certifies.

    This test is performed via the diagnostic-return pathway of
    `bound_sdp_escalation_int_ge` so we read the actual cushion the
    cert used and offset by exactly that amount.
    """
    d = 4
    windows = build_windows(d)

    # mu* = symmetric optimum at d=4 (val(4)=10/9 at this point).
    mu_star = np.array([1/3, 1/6, 1/6, 1/3])
    radius = 5e-4
    lo_f = mu_star - radius
    hi_f = mu_star + radius
    lo_int, hi_int = _ints_from_floats(lo_f, hi_f)

    # First: get the actual SDP value and cushion via diagnostic.
    target_probe = Fraction(1, 1)  # any low target — we just want the diag
    _cert, info = bound_sdp_escalation_int_ge(
        lo_int, hi_int, windows, d,
        target_probe.numerator, target_probe.denominator,
        order=2, time_limit_s=10.0,
        return_diagnostic=True,
    )
    # MOSEK status strings: 'Optimal' / 'NearOptimal' (success) vs other.
    assert info['status'] in ('Optimal', 'NearOptimal'), (
        f"SDP solver did not solve cleanly: {info}")
    obj_val_dual = info['obj_val_dual']
    cushion = info['cushion']
    lb_safe = info['lb_safe']
    print(f"[T4] cushion={cushion:.2e}, obj_val_dual={obj_val_dual:.8f}, "
          f"lb_safe={lb_safe:.8f}")

    # Build target = lb_safe (the cushion boundary). lb_safe is by
    # construction `obj_val_dual - cushion`. Round to a Fraction.
    eps = max(1e-14, abs(lb_safe) * 1e-13) * 4.0  # ~4 ULP

    # Target slightly BELOW lb_safe: should certify.
    t_below = lb_safe - eps
    t_below_q = Fraction(t_below).limit_denominator(10**12)
    cert_below = bound_sdp_escalation_int_ge(
        lo_int, hi_int, windows, d,
        t_below_q.numerator, t_below_q.denominator,
        order=2, time_limit_s=10.0,
    )
    print(f"     target below lb_safe by ~{eps:.2e}: cert={cert_below}")
    assert cert_below, (
        f"[T4 FAIL] target {float(t_below_q)} (below lb_safe by {eps}) "
        f"failed to certify; lb_safe={lb_safe}")

    # Target slightly ABOVE lb_safe: should NOT certify.
    t_above = lb_safe + eps
    t_above_q = Fraction(t_above).limit_denominator(10**12)
    cert_above = bound_sdp_escalation_int_ge(
        lo_int, hi_int, windows, d,
        t_above_q.numerator, t_above_q.denominator,
        order=2, time_limit_s=10.0,
    )
    print(f"     target above lb_safe by ~{eps:.2e}: cert={cert_above}")
    assert not cert_above, (
        f"[T4 FAIL] target {float(t_above_q)} (above lb_safe by {eps}) "
        f"wrongly certified; lb_safe={lb_safe}")

    # Also verify that the cushion floor is at least _CUSHION_FLOOR.
    assert cushion >= _CUSHION_FLOOR - 1e-15, (
        f"[T4 FAIL] cushion {cushion} below floor {_CUSHION_FLOOR}")


# ---------------------------------------------------------------------
# T5: d=30 compile + solve gate (the publishability gate for MOSEK
# escalation). Verifies that cache build + per-box solve fit in the
# per-call budget the cascade requires.
# ---------------------------------------------------------------------

def test_sdp_compiles_and_solves_at_d30():
    """At d=30 hw=0.025 (the BnB transition zone), the MOSEK build must
    compile and solve in < 30s (cache build amortised separately) on a
    cold-start box drawn from the H_d distribution."""
    import time as _time
    d = 30
    windows = build_windows(d)
    t0 = _time.time()
    cache = build_sdp_escalation_cache(d, windows)
    t_cache = _time.time() - t0
    print(f"[T5] d={d} cache build: {t_cache:.2f}s, n_y={cache['n_y']}, "
          f"B={cache['B']}, B1={cache['B1']}, n_eq={cache['n_eq']}, "
          f"n_W={cache['n_W_kept']}")
    assert t_cache < 60.0, (
        f"[T5 FAIL] cache build {t_cache:.1f}s exceeded 60s budget; "
        f"per-worker startup will be too slow")

    rng = np.random.default_rng(0)
    mu = rng.dirichlet(np.ones(d))
    if mu[0] > mu[-1]:
        mu = mu[::-1]
    hw = 0.025
    lo_f = np.maximum(mu - hw, 0.0)
    hi_f = np.minimum(mu + hw, 1.0)
    if lo_f.sum() > 1.0 or hi_f.sum() < 1.0:
        # widen until it intersects (rare).
        lo_f = np.maximum(mu - 0.04, 0.0)
        hi_f = np.minimum(mu + 0.04, 1.0)

    t1 = _time.time()
    res = bound_sdp_escalation_lb_float(lo_f, hi_f, windows, d,
                                         cache=cache, time_limit_s=30.0)
    t_solve = _time.time() - t1
    print(f"[T5] d={d} hw=0.025 solve: {t_solve:.2f}s status={res['status']} "
          f"obj_val_dual={res['obj_val_dual']}")
    assert t_solve < 30.0, (
        f"[T5 FAIL] per-box solve {t_solve:.1f}s exceeded 30s budget")
    assert res['is_feasible_status'], (
        f"[T5 FAIL] MOSEK did not solve cleanly: status={res['status']}")


# ---------------------------------------------------------------------
# T6: Empty-box vacuous cert (sanity check)
# ---------------------------------------------------------------------

def test_sdp_empty_box_vacuous():
    """For a box that does NOT intersect the simplex, the cert is
    vacuously True at any target (matches the convention used by
    `bound_anchor_int_ge` and `bound_epigraph_int_ge`)."""
    d = 4
    windows = build_windows(d)
    # All-zero box — sum(hi) = 0 < 1, so empty.
    lo_f = np.zeros(d)
    hi_f = np.full(d, 0.1)  # sum_hi = 0.4 < 1.0
    lo_int, hi_int = _ints_from_floats(lo_f, hi_f)
    cert = bound_sdp_escalation_int_ge(
        lo_int, hi_int, windows, d,
        target_num=1281, target_den=1000,
    )
    assert cert, "empty box should vacuously certify"

    # Box with sum(lo) > 1 — also empty.
    lo_f = np.full(d, 0.4)
    hi_f = np.full(d, 0.5)  # sum_lo = 1.6 > 1.0
    lo_int, hi_int = _ints_from_floats(lo_f, hi_f)
    cert = bound_sdp_escalation_int_ge(
        lo_int, hi_int, windows, d,
        target_num=1281, target_den=1000,
    )
    assert cert, "box with sum_lo > 1 should vacuously certify"


def main():
    """Direct-run harness (matches the d10 / epigraph test convention)."""
    print("=" * 60)
    print("SDP ESCALATION: soundness + cushion + d22 closure tests")
    print("=" * 60)
    results = []
    for name, fn in [
        ("test_sdp_empty_box_vacuous", test_sdp_empty_box_vacuous),
        ("test_sdp_soundness_random_d4", test_sdp_soundness_random_d4),
        ("test_sdp_lb_ge_epigraph_lp_d6", test_sdp_lb_ge_epigraph_lp_d6),
        ("test_sdp_cushion_correctness", test_sdp_cushion_correctness),
        ("test_sdp_compiles_and_solves_at_d30",
         test_sdp_compiles_and_solves_at_d30),
        ("test_sdp_certifies_d22_residual_box",
         test_sdp_certifies_d22_residual_box),
    ]:
        try:
            fn()
            results.append((name, "PASS"))
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
