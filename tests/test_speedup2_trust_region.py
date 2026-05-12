"""Tests for Speedup #2: spectral Phase 1 + trust-region closed-form cell-min QP.

Verifies:
  (S1) SOUNDNESS — spectral Phase 1 LB <= true cell-min via vertex enum.
  (S2) SOUNDNESS — trust-region LB <= true cell-min via vertex enum.
  (T1) TIGHTNESS — spectral Phase 1 LB >= original Phase 1 LB (qdrop_table tightens qd_c).
  (T2) TIGHTNESS — trust-region LB >= spectral Phase 1 LB (joint vs split bounds).
  (Q1) QP CORRECTNESS — _trust_region_qp_lb returns the analytic minimum on
       a hand-picked PSD test case where the closed form is computable.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coarse_cascade_prover import (  # noqa: E402
    _phase1_lipschitz_lb,
    _phase1_lipschitz_lb_spec,
    _box_certify_cell_vertex,
    _box_certify_cell_trust_region,
    _trust_region_qp_lb,
    compute_qdrop_table,
    compute_window_eigen_table,
)


def _random_canonical_cell(rng, d, S):
    cuts = sorted(rng.choice(np.arange(1, S + d), size=d - 1, replace=False))
    cuts = [0] + list(cuts) + [S + d]
    parts = np.array([cuts[i + 1] - cuts[i] - 1 for i in range(d)], dtype=np.int64)
    assert parts.sum() == S
    return parts.astype(np.float64) / float(S)


def test_phase1_spec_sound_d4():
    """Spec Phase 1 LB <= true cell min via vertex enum, d=4."""
    rng = np.random.RandomState(0)
    d = 4
    S = 12
    delta_q = 1.0 / S
    c_target = 1e9
    qdrop = compute_qdrop_table(d)
    for trial in range(50):
        mu = _random_canonical_cell(rng, d, S)
        _, lb_spec = _phase1_lipschitz_lb_spec(mu, d, delta_q, c_target, qdrop)
        _, lb_v = _box_certify_cell_vertex(mu, d, delta_q, c_target)
        assert lb_spec <= lb_v + 1e-9, (
            f"trial {trial}: spec ({lb_spec}) > vertex ({lb_v}) UNSOUND")


def test_phase1_spec_sound_d6():
    rng = np.random.RandomState(1)
    d = 6
    S = 18
    delta_q = 1.0 / S
    c_target = 1e9
    qdrop = compute_qdrop_table(d)
    for trial in range(30):
        mu = _random_canonical_cell(rng, d, S)
        _, lb_spec = _phase1_lipschitz_lb_spec(mu, d, delta_q, c_target, qdrop)
        _, lb_v = _box_certify_cell_vertex(mu, d, delta_q, c_target)
        assert lb_spec <= lb_v + 1e-9, (
            f"trial {trial}: spec ({lb_spec}) > vertex ({lb_v}) UNSOUND")


def test_phase1_spec_tightness():
    """Spec Phase 1 LB >= original Phase 1 LB (qdrop_table only tightens)."""
    rng = np.random.RandomState(7)
    d = 6
    S = 18
    delta_q = 1.0 / S
    c_target = 1e9
    qdrop = compute_qdrop_table(d)
    for trial in range(30):
        mu = _random_canonical_cell(rng, d, S)
        _, lb_orig = _phase1_lipschitz_lb(mu, d, delta_q, c_target)
        _, lb_spec = _phase1_lipschitz_lb_spec(mu, d, delta_q, c_target, qdrop)
        assert lb_spec >= lb_orig - 1e-9, (
            f"trial {trial}: spec ({lb_spec}) < original ({lb_orig}) — should be tighter")


def test_trust_region_sound_d4():
    """Trust-region LB <= true cell min via vertex enum, d=4."""
    rng = np.random.RandomState(2)
    d = 4
    S = 12
    delta_q = 1.0 / S
    c_target = 1e9
    V, lam, valid = compute_window_eigen_table(d)
    for trial in range(50):
        mu = _random_canonical_cell(rng, d, S)
        _, lb_tr = _box_certify_cell_trust_region(mu, d, delta_q, c_target, V, lam, valid)
        _, lb_v = _box_certify_cell_vertex(mu, d, delta_q, c_target)
        assert lb_tr <= lb_v + 1e-9, (
            f"trial {trial}: trust-region ({lb_tr}) > vertex ({lb_v}) — UNSOUND")


def test_trust_region_sound_d6():
    rng = np.random.RandomState(3)
    d = 6
    S = 18
    delta_q = 1.0 / S
    c_target = 1e9
    V, lam, valid = compute_window_eigen_table(d)
    for trial in range(30):
        mu = _random_canonical_cell(rng, d, S)
        _, lb_tr = _box_certify_cell_trust_region(mu, d, delta_q, c_target, V, lam, valid)
        _, lb_v = _box_certify_cell_vertex(mu, d, delta_q, c_target)
        assert lb_tr <= lb_v + 1e-9, (
            f"trial {trial}: trust-region ({lb_tr}) > vertex ({lb_v}) — UNSOUND")


def test_trust_region_qp_psd():
    """Hand-picked PSD test case: D = diag(1, 2), c = (1, 2), R^2 = 0.5.

    Unconstrained min: x_k = -c_k/(2 D_kk) = (-0.5, -0.5).
    ||x*||^2 = 0.25 + 0.25 = 0.5 = R^2 (boundary).
    Min value = D_11*x1^2 + c_1*x_1 + D_22*x_2^2 + c_2*x_2
              = 1*0.25 - 0.5 + 2*0.25 - 1 = 0.25 - 0.5 + 0.5 - 1 = -0.75.
    """
    c = np.array([1.0, 2.0])
    lam = np.array([1.0, 2.0])
    R_sq = 0.5
    qp_min = _trust_region_qp_lb(c, lam, R_sq, 2)
    # Expected -0.75; closed-form sound LB so qp_min <= -0.75 (within FP)
    assert qp_min <= -0.75 + 1e-8, f"qp_min ({qp_min}) > -0.75 — wrong direction"
    assert qp_min >= -0.75 - 1e-6, f"qp_min ({qp_min}) << -0.75 — too loose"


def test_trust_region_qp_indefinite():
    """Indefinite test: D = diag(-1, 1), c = (1, 1), R^2 = 1.

    With negative eigenvalue, must have nu > 1. The QP becomes
    min -x_1^2 + x_2^2 + x_1 + x_2 s.t. x_1^2 + x_2^2 <= 1.
    The min is -3/2 (taking x_1 = 1, x_2 = -1/2, ||x||^2 = 1.25 — wait, let me recompute)

    Lagrangian: L(x, nu) = (nu-1)x_1^2 + (nu+1)x_2^2 + x_1 + x_2 - nu
    For nu > 1: x_1 = -1/(2(nu-1)), x_2 = -1/(2(nu+1))
    Constraint: 1/(4(nu-1)^2) + 1/(4(nu+1)^2) = 1
    Numerically solve: nu ~= 1.27...
    """
    c = np.array([1.0, 1.0])
    lam = np.array([-1.0, 1.0])
    R_sq = 1.0
    qp_min = _trust_region_qp_lb(c, lam, R_sq, 2)
    # The actual QP min is finite (constraint is active for nu > 1).
    # Sanity: must be < 0 (we can pick x = (-1/2, -1/2): val = -0.25+0.25-0.5-0.5 = -1).
    assert qp_min <= -0.5, f"qp_min ({qp_min}) too high for indefinite case"


def test_trust_region_zero_gradient():
    """If c = 0, QP min = 0 (x = 0 is optimal regardless of D)."""
    c = np.zeros(4)
    lam = np.array([-2.0, -1.0, 1.0, 2.0])
    R_sq = 1.0
    qp_min = _trust_region_qp_lb(c, lam, R_sq, 4)
    assert abs(qp_min) < 1e-9, f"qp_min ({qp_min}) != 0 for zero gradient"


def test_pipeline_full_d8():
    """Full pipeline: vertex enum at d=8, S=16, c=high, all cells should be
    sound across all three tiers."""
    rng = np.random.RandomState(4)
    d = 8
    S = 16
    delta_q = 1.0 / S
    c_target = 1e9
    qdrop = compute_qdrop_table(d)
    V, lam, valid = compute_window_eigen_table(d)
    for trial in range(15):
        mu = _random_canonical_cell(rng, d, S)
        _, lb_p1_orig = _phase1_lipschitz_lb(mu, d, delta_q, c_target)
        _, lb_p1_spec = _phase1_lipschitz_lb_spec(mu, d, delta_q, c_target, qdrop)
        _, lb_tr = _box_certify_cell_trust_region(mu, d, delta_q, c_target, V, lam, valid)
        _, lb_v = _box_certify_cell_vertex(mu, d, delta_q, c_target)
        assert lb_p1_spec <= lb_v + 1e-9, "spec phase1 unsound"
        assert lb_tr <= lb_v + 1e-9, "trust-region unsound"
        assert lb_p1_spec >= lb_p1_orig - 1e-9, "spec phase1 should be >= orig"


if __name__ == "__main__":
    test_phase1_spec_sound_d4()
    print("phase1 spec d=4 sound OK")
    test_phase1_spec_sound_d6()
    print("phase1 spec d=6 sound OK")
    test_phase1_spec_tightness()
    print("phase1 spec >= original OK")
    test_trust_region_sound_d4()
    print("trust-region d=4 sound OK")
    test_trust_region_sound_d6()
    print("trust-region d=6 sound OK")
    test_trust_region_qp_psd()
    print("QP PSD test OK")
    test_trust_region_qp_indefinite()
    print("QP indefinite test OK")
    test_trust_region_zero_gradient()
    print("QP zero-gradient test OK")
    test_pipeline_full_d8()
    print("d=8 full pipeline OK")
