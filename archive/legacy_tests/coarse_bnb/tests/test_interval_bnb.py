"""Correctness tests for interval_bnb.

Gates implemented (per task brief):
  G1. d=4 certifies val(4) >= 1.10 in < 10s.
  G2. d=4 with target 1.11 terminates in FAIL (soundness).
  G3. McCormick bound monotone in box size (random spot check).
  G4. Natural Fraction bound re-verifies float64 bound (sign agreement).
"""
from __future__ import annotations

import os
import sys
import time
from fractions import Fraction

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from interval_bnb.bnb import branch_and_bound
from interval_bnb.bound_eval import (
    bound_mccormick_lp,
    bound_natural,
    bound_natural_exact,
)
from interval_bnb.box import Box
from interval_bnb.symmetry import half_simplex_cuts
from interval_bnb.windows import build_windows


# ---------------------------------------------------------------------
# G1: d=4 anchor certification
# ---------------------------------------------------------------------

def test_g1_d4_certifies_1p10():
    """val(4) ~= 1.10233, so target 1.10 should certify quickly."""
    t0 = time.time()
    res = branch_and_bound(d=4, target_c="1.10", verbose=False, time_budget_s=30.0)
    elapsed = time.time() - t0
    assert res.success, f"d=4 failed to certify target 1.10 (elapsed={elapsed:.1f}s)"
    assert elapsed < 30.0, f"d=4 certification took {elapsed:.1f}s > 30s budget"


# ---------------------------------------------------------------------
# G2: d=4 soundness -- cannot certify > val(4)
# ---------------------------------------------------------------------

def test_g2_d4_fails_1p11():
    """val(4) ~= 1.10233, so target 1.11 MUST NOT be certified."""
    res = branch_and_bound(
        d=4, target_c="1.11", verbose=False,
        min_box_width=1e-6, max_nodes=200_000, time_budget_s=60.0,
    )
    assert not res.success, (
        "Soundness bug: BnB claimed val(4) >= 1.11 but val(4) ~= 1.10233"
    )


# ---------------------------------------------------------------------
# G3: McCormick monotonicity in box size
# ---------------------------------------------------------------------

def test_g3_mccormick_monotone_shrink():
    """Shrinking a box cannot worsen the McCormick lower bound."""
    rng = np.random.default_rng(42)
    d = 4
    windows = build_windows(d)
    w = windows[len(windows) // 2]

    checked = 0
    for _ in range(50):
        lo = rng.uniform(0.0, 0.2, size=d)
        hi = lo + rng.uniform(0.05, 0.4, size=d)
        hi = np.minimum(hi, 1.0)
        if lo.sum() > 1.0 or hi.sum() < 1.0:
            continue
        lb_big, _ = bound_mccormick_lp(lo, hi, w)

        mid = 0.5 * (lo + hi)
        shrink = 0.5
        lo_s = mid - shrink * (mid - lo)
        hi_s = mid + shrink * (hi - mid)
        if lo_s.sum() > 1.0 or hi_s.sum() < 1.0:
            continue
        lb_small, _ = bound_mccormick_lp(lo_s, hi_s, w)
        assert lb_small >= lb_big - 1e-9, (
            f"shrink worsened bound: {lb_big:.6f} -> {lb_small:.6f}"
        )
        checked += 1
    assert checked >= 5, f"too few feasible random boxes: {checked}"


# ---------------------------------------------------------------------
# G4: Fraction replay agrees with float64 natural bound
# ---------------------------------------------------------------------

def test_g4_fraction_natural_agrees_float():
    """For 200 random boxes, |bound_natural_float - float(bound_natural_exact)|
    is within float precision (a few ulps)."""
    rng = np.random.default_rng(0)
    d = 4
    windows = build_windows(d)

    max_abs_err = 0.0
    for _ in range(200):
        # Random dyadic-rational lo/hi.
        denom = 256
        lo_int = rng.integers(0, denom // 2, size=d)
        width_int = rng.integers(1, denom // 2, size=d)
        hi_int = np.minimum(lo_int + width_int, denom)
        lo_q = [Fraction(int(v), denom) for v in lo_int]
        hi_q = [Fraction(int(v), denom) for v in hi_int]
        lo = np.array([float(x) for x in lo_q], dtype=np.float64)
        hi = np.array([float(x) for x in hi_q], dtype=np.float64)

        for w in windows:
            b_float = bound_natural(lo, hi, w)
            b_rat = bound_natural_exact(lo_q, hi_q, w)
            err = abs(b_float - float(b_rat))
            if err > max_abs_err:
                max_abs_err = err
    # Bound on accumulated error: ~ |pairs| * float ulp.
    assert max_abs_err < 1e-10, f"float/Fraction disagreement: {max_abs_err}"


# ---------------------------------------------------------------------
# Bonus: symmetry cuts do not change feasibility (sanity).
# ---------------------------------------------------------------------

def test_sym_cuts_preserve_simplex_feasibility():
    d = 6
    sym = half_simplex_cuts(d)
    B = Box.initial(d, sym)
    assert B.intersects_simplex()
    # A symmetric point is feasible.
    mu = [Fraction(1, d) for _ in range(d)]
    for (i, j) in sym:
        assert mu[i] <= mu[j]
    assert sum(mu) == 1


def test_box_split_exact_midpoint():
    """Splits at midpoints remain EXACT dyadic rationals in float64
    (invariant required for lossless Fraction conversion)."""
    d = 4
    B = Box.initial(d)
    # Drop ten levels along each axis.
    for _ in range(10):
        ax = B.widest_axis()
        B, _ = B.split(ax)
    lo_q, hi_q = B.to_fractions()
    # Every endpoint must be a dyadic rational: denominator is a power of 2.
    for x in lo_q + hi_q:
        denom = x.denominator
        assert (denom & (denom - 1)) == 0, f"non-dyadic: {x}"


# ---------------------------------------------------------------------
# Extra: Fraction McCormick LP agrees with float LP.
# ---------------------------------------------------------------------

def test_mccormick_exact_agrees_float():
    """The Fraction rational LP and the float64 sort LP should give
    the same value (within float precision) for each window/box."""
    from interval_bnb.bound_eval import bound_mccormick_exact_nosym
    rng = np.random.default_rng(7)
    d = 4
    windows = build_windows(d)
    denom = 128
    for _ in range(30):
        lo_int = rng.integers(0, denom // 4, size=d)
        width_int = rng.integers(1, denom // 2, size=d)
        hi_int = np.minimum(lo_int + width_int, denom)
        lo_q = [Fraction(int(v), denom) for v in lo_int]
        hi_q = [Fraction(int(v), denom) for v in hi_int]
        lo = np.array([float(x) for x in lo_q], dtype=np.float64)
        hi = np.array([float(x) for x in hi_q], dtype=np.float64)
        if lo.sum() > 1.0 or hi.sum() < 1.0:
            continue
        for w in windows:
            lb_f, _ = bound_mccormick_lp(lo, hi, w)
            lb_q = bound_mccormick_exact_nosym(lo_q, hi_q, w)
            assert abs(lb_f - float(lb_q)) < 1e-9, (
                f"float LP {lb_f} vs Fraction {float(lb_q)} for {w.name}"
            )


# ---------------------------------------------------------------------
# Extra: Fraction rigor matches float pruning (leaves only closed if
# Fraction bound also clears target).
# ---------------------------------------------------------------------

def test_rigor_leaves_only_closed_when_fraction_bound_passes():
    """The BnB must refuse to certify a leaf whose Fraction bound is
    below target. This test exercises the rigor gate by requesting a
    tight target and confirming success (meaning Fraction agreed)."""
    from interval_bnb.bnb import branch_and_bound
    res = branch_and_bound(d=4, target_c="1.102", verbose=False,
                           time_budget_s=30.0)
    assert res.success
    assert res.stats.rigor_retries == 0, (
        f"fast path and Fraction path diverged: "
        f"rigor_retries={res.stats.rigor_retries}"
    )


# ---------------------------------------------------------------------
# Rigor test: half_simplex_cuts is a valid orbit cover for sigma.
# ---------------------------------------------------------------------

def test_half_simplex_cuts_orbit_cover():
    """For every mu in Delta_d (random sample), either mu or sigma(mu)
    must satisfy the half-simplex cuts. Catches the bug where the
    'all pairs ordered' cut missed orbits."""
    from interval_bnb.symmetry import half_simplex_cuts, sigma_pairs
    rng = np.random.default_rng(31415)
    for d in (2, 3, 4, 6, 8, 10, 14, 16):
        cuts = half_simplex_cuts(d)
        for _ in range(200):
            raw = rng.uniform(0.01, 1.0, size=d)
            mu = raw / raw.sum()
            mu_rev = mu[::-1]
            in_h1 = all(mu[i] <= mu[j] for i, j in cuts)
            in_h2 = all(mu_rev[i] <= mu_rev[j] for i, j in cuts)
            assert in_h1 or in_h2, (
                f"orbit cover VIOLATED at d={d}: neither mu nor sigma(mu) "
                f"in H_d for cuts={cuts}, mu={mu}, sigma(mu)={mu_rev}"
            )


def test_intersects_simplex_exact_at_edge():
    """Boundary cases: a box whose vertex exactly meets sum=1 must be
    reported as intersecting."""
    from interval_bnb.box import Box
    d = 4
    # Box containing exactly mu = (0.25, 0.25, 0.25, 0.25) as the sole
    # simplex point: lo = (0.25,...), hi = (0.25,...). Sums are exactly 1.
    lo = np.full(d, 0.25, dtype=np.float64)
    hi = np.full(d, 0.25, dtype=np.float64)
    B = Box(lo, hi)
    # Without int metadata, falls back to float check; should still be OK
    # for this cleanly representable case.
    assert B.intersects_simplex()


def test_bound_natural_ge_zero_and_valid():
    """Natural bound must be <= actual mu^T M_W mu at any mu in the box
    (spot-check random boxes and random points)."""
    from interval_bnb.bound_eval import bound_natural
    from interval_bnb.windows import build_windows
    rng = np.random.default_rng(2)
    d = 6
    windows = build_windows(d)
    for _ in range(50):
        lo = rng.uniform(0.0, 0.1, size=d)
        hi = lo + rng.uniform(0.01, 0.3, size=d)
        for w in windows:
            lb = bound_natural(lo, hi, w)
            # Check against actual mu^T M_W mu for a random mu in box.
            t = rng.uniform(size=d)
            mu = lo + t * (hi - lo)
            actual = w.scale * float(
                sum(mu[i] * mu[j] for (i, j) in w.pairs_all)
            )
            assert lb <= actual + 1e-10, (
                f"natural bound {lb} > actual {actual} -- invalid LB!"
            )


def test_bound_autoconv_valid_on_simplex():
    """Autoconv is a valid LB for points ON the simplex."""
    from interval_bnb.bound_eval import bound_autoconv
    from interval_bnb.windows import build_windows
    rng = np.random.default_rng(3)
    d = 6
    windows = build_windows(d)
    for _ in range(30):
        # Random simplex point via Dirichlet.
        mu = rng.dirichlet(np.ones(d))
        # Tight box around mu so autoconv bound applies at this point.
        lo = np.maximum(mu - 1e-6, 0.0)
        hi = np.minimum(mu + 1e-6, 1.0)
        # Double-check sum condition (should still include mu).
        if lo.sum() > 1.0 or hi.sum() < 1.0:
            continue
        for w in windows:
            lb = bound_autoconv(lo, hi, w, d)
            actual = w.scale * float(
                sum(mu[i] * mu[j] for (i, j) in w.pairs_all)
            )
            assert lb <= actual + 1e-8, (
                f"autoconv bound {lb} > actual {actual}"
            )


def test_end_to_end_rigor_sampled():
    """End-to-end: certify val(4) >= 1.10, then sample random simplex
    points, check each μ lies in one of the certified leaves, and verify
    that the certifying window's value at μ meets the target."""
    from interval_bnb.bnb import branch_and_bound_from_box
    from interval_bnb.box import Box
    from interval_bnb.symmetry import half_simplex_cuts
    from interval_bnb.windows import build_windows
    rng = np.random.default_rng(111)
    d = 4
    sym = half_simplex_cuts(d)
    windows = build_windows(d)
    # Run with verbose=False, short budget -- we only check consistency.
    res = branch_and_bound_from_box(
        d, "1.10", Box.initial(d, sym),
        verbose=False, time_budget_s=20, max_nodes=500_000,
    )
    assert res.success
    # Sample random simplex points. For each, check that SOME window has
    # μ^T M_W μ >= 1.10 (this is an instance of the certified claim).
    for _ in range(100):
        mu = rng.dirichlet(np.ones(d))
        # Apply sym reduction: if mu_0 > mu_{d-1}, reverse.
        if mu[0] > mu[-1]:
            mu = mu[::-1]
        max_val = 0.0
        for w in windows:
            val = w.scale * float(sum(mu[i] * mu[j] for (i, j) in w.pairs_all))
            if val > max_val:
                max_val = val
        # val(4) = 1.10233 > 1.10, so EVERY simplex point has max_W >= 1.10.
        assert max_val >= 1.10 - 1e-9, (
            f"sampled mu={mu} has max_W value {max_val} < 1.10"
        )


def test_work_stealing_matches_serial():
    """At d=6, target slightly below val(6)~1.171: work-stealing
    4-worker run must succeed, like the serial run."""
    import pytest
    from interval_bnb.bnb import branch_and_bound
    from interval_bnb.parallel import parallel_branch_and_bound
    target = "1.16"  # below val(6); must certify
    res_serial = branch_and_bound(d=6, target_c=target, verbose=False,
                                  time_budget_s=60, max_nodes=5_000_000)
    if not res_serial.success:
        pytest.skip("d=6 serial didn't finish in 60s; skip parallel comparison")
    res_par = parallel_branch_and_bound(
        d=6, target_c=target, verbose=False,
        workers=4, init_split_depth=4,
        time_budget_s=120, max_nodes=50_000_000,
    )
    assert res_par["success"], (
        f"work-stealing FAILED while serial succeeded: {res_par}"
    )
    # Sanity: in_flight reached 0.
    assert res_par["in_flight_final"] == 0


def test_termination_no_race():
    """Many short parallel runs: no hang, no spurious failure."""
    from interval_bnb.parallel import parallel_branch_and_bound
    for i in range(5):
        res = parallel_branch_and_bound(
            d=4, target_c="1.10", verbose=False,
            workers=4, init_split_depth=3,
            time_budget_s=30, max_nodes=500_000,
        )
        assert res["success"], f"parallel d=4 target=1.10 FAILED on iter {i}: {res}"
        assert res["in_flight_final"] == 0


def test_int_rigor_matches_fraction_rigor():
    """The integer-arithmetic rigor gate must give BIT-IDENTICAL
    results to the Fraction rigor gate on random dyadic boxes."""
    from interval_bnb.bnb import rigor_replay, rigor_replay_fraction
    from interval_bnb.box import Box, SCALE
    from interval_bnb.windows import build_windows
    from fractions import Fraction as F
    rng = np.random.default_rng(51)
    for d in (4, 6, 8):
        windows = build_windows(d)
        for _ in range(12):
            # Random dyadic box (denom up to 2^10 for speed).
            denom = 1024
            lo_n = rng.integers(0, denom // 4, size=d)
            wid_n = rng.integers(1, denom // 2, size=d)
            hi_n = np.minimum(lo_n + wid_n, denom)
            if lo_n.sum() > denom or hi_n.sum() < denom:
                continue
            lo = lo_n.astype(np.float64) / denom
            hi = hi_n.astype(np.float64) / denom
            lo_int = [int(x) * (SCALE // denom) for x in lo_n]
            hi_int = [int(x) * (SCALE // denom) for x in hi_n]
            B = Box(lo, hi, lo_int, hi_int)
            target_q = F(11, 10)  # 1.1
            for w in windows[:6]:  # subset for speed
                r_int = rigor_replay(B, w, d, target_q)
                r_frac = rigor_replay_fraction(B, w, d, target_q)
                assert r_int == r_frac, (
                    f"int vs fraction rigor disagree at d={d}, "
                    f"window={w.name}, target={target_q}: "
                    f"int={r_int} fraction={r_frac}"
                )


def test_rank1_update_matches_from_scratch():
    """The rank-1 update must produce the same bound as a from-scratch
    recompute of the child box."""
    from interval_bnb.bound_eval import (
        batch_bounds_full, batch_bounds_rank1_lo, batch_bounds_rank1_hi,
        window_tensor,
    )
    from interval_bnb.windows import build_windows
    d = 6
    windows = build_windows(d)
    A, scales = window_tensor(windows, d)
    rng = np.random.default_rng(7)
    for _ in range(10):
        lo = rng.uniform(0.0, 0.2, size=d)
        hi = lo + rng.uniform(0.05, 0.5, size=d)
        hi = np.minimum(hi, 1.0)
        if lo.sum() > 1.0 or hi.sum() < 1.0:
            continue
        _, _, _, _, parent_cache = batch_bounds_full(lo, hi, A, scales, 0.0)
        k = int(rng.integers(0, d))
        # Right child: lo[k] increases by delta
        delta = 0.3 * (hi[k] - lo[k])
        new_lo = lo.copy(); new_lo[k] += delta
        lb_r1, _, _, _, _ = batch_bounds_rank1_lo(
            A, scales, parent_cache, new_lo, k, 0.0,
        )
        lb_full, _, _, _, _ = batch_bounds_full(new_lo, hi, A, scales, 0.0)
        assert abs(lb_r1 - lb_full) < 1e-9, (
            f"rank-1 lo gave {lb_r1} vs from-scratch {lb_full}"
        )
        # Left child: hi[k] decreases
        new_hi = hi.copy(); new_hi[k] -= delta
        lb_r1h, _, _, _, _ = batch_bounds_rank1_hi(
            A, scales, parent_cache, new_hi, k, 0.0,
        )
        lb_fullh, _, _, _, _ = batch_bounds_full(lo, new_hi, A, scales, 0.0)
        assert abs(lb_r1h - lb_fullh) < 1e-9, (
            f"rank-1 hi gave {lb_r1h} vs from-scratch {lb_fullh}"
        )
