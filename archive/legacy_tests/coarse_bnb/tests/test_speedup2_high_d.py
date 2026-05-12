"""High-d benchmarks for Speedup #2 (the regime that matters: d >= 12).

Quick (< 30s) tests on d in {12, 14} comparing v1 (original 2-tier) vs
v2 (4-tier with spectral Phase 1, trust-region QP). At d>=12 Phase 1 capture
drops below 100% on hard c-targets, so Tier 1b/2 actually fire.

We do NOT run the full cascade (intractable at d=12 with realistic S).
Instead we sample random canonical compositions and time the box-cert.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coarse_cascade_prover import (  # noqa: E402
    _box_certify_batch_adaptive,
    _box_certify_batch_adaptive_v2,
    _phase1_lipschitz_lb,
    _phase1_lipschitz_lb_spec,
    _box_certify_cell_trust_region,
    _box_certify_cell_vertex,
    compute_qdrop_table,
    compute_window_eigen_table,
)


def _random_canonical_batch(rng, d, S, B, x_cap=None):
    """Random canonical (sorted-asc not guaranteed) compositions of S into d parts."""
    out = np.zeros((B, d), dtype=np.int32)
    n_done = 0
    while n_done < B:
        cuts = sorted(rng.choice(np.arange(1, S + d), size=d - 1, replace=False))
        cuts = [0] + list(cuts) + [S + d]
        parts = np.array([cuts[i + 1] - cuts[i] - 1 for i in range(d)], dtype=np.int32)
        if x_cap is not None and (parts > x_cap).any():
            continue
        out[n_done] = parts
        n_done += 1
    return out


def test_soundness_d12():
    """At d=12, S=24, c high: every tier's LB <= true cell-min (via vertex enum).

    Vertex enum at d=12 is feasible for small batches (~1ms/cell).
    """
    rng = np.random.RandomState(0)
    d = 12
    S = 24
    delta_q = 1.0 / S
    c_target = 1e9  # disable shortcut
    qdrop = compute_qdrop_table(d)
    V, lam, valid = compute_window_eigen_table(d)

    n_check = 8  # vertex enum is slow at d=12 (~12ms/cell), keep small
    batch = _random_canonical_batch(rng, d, S, n_check)
    for b in range(n_check):
        mu = batch[b].astype(np.float64) / S
        _, lb_v = _box_certify_cell_vertex(mu, d, delta_q, c_target)
        _, lb_p1 = _phase1_lipschitz_lb(mu, d, delta_q, c_target)
        _, lb_p1s = _phase1_lipschitz_lb_spec(mu, d, delta_q, c_target, qdrop)
        _, lb_tr = _box_certify_cell_trust_region(mu, d, delta_q, c_target,
                                                    V, lam, valid)
        assert lb_p1 <= lb_v + 1e-8, f"trial {b}: P1 ({lb_p1}) > vertex ({lb_v})"
        assert lb_p1s <= lb_v + 1e-8, f"trial {b}: P1-spec ({lb_p1s}) > vertex ({lb_v})"
        assert lb_tr <= lb_v + 1e-8, f"trial {b}: TR ({lb_tr}) > vertex ({lb_v})"
        assert lb_p1s >= lb_p1 - 1e-9, f"trial {b}: spec ({lb_p1s}) < orig ({lb_p1})"


def test_capture_rate_d12_hard():
    """At d=12, S=24, c=1.25 (hard for Phase 1): measure how many cells each tier saves.

    The v2 dispatcher should reduce Tier 3 (vertex enum) calls.
    """
    rng = np.random.RandomState(1)
    d = 12
    S = 24
    c_target = 1.25
    qdrop = compute_qdrop_table(d)
    V, lam, valid = compute_window_eigen_table(d)

    B = 500
    batch = _random_canonical_batch(rng, d, S, B)

    # Warmup compile
    cert_out = np.zeros(10, dtype=np.int8)
    min_tv = np.zeros(10, dtype=np.float64)
    tier = np.zeros(10, dtype=np.int8)
    _box_certify_batch_adaptive(batch[:10], d, S, c_target, cert_out, min_tv, tier)
    _box_certify_batch_adaptive_v2(batch[:10], d, S, c_target,
                                    qdrop, V, lam, valid,
                                    cert_out, min_tv, tier)

    # Time and tier-count for v1
    cert1 = np.zeros(B, dtype=np.int8)
    mt1 = np.zeros(B, dtype=np.float64)
    tier1 = np.zeros(B, dtype=np.int8)
    t = time.perf_counter()
    _box_certify_batch_adaptive(batch, d, S, c_target, cert1, mt1, tier1)
    dt1 = time.perf_counter() - t

    # Time and tier-count for v2
    cert2 = np.zeros(B, dtype=np.int8)
    mt2 = np.zeros(B, dtype=np.float64)
    tier2 = np.zeros(B, dtype=np.int8)
    t = time.perf_counter()
    _box_certify_batch_adaptive_v2(batch, d, S, c_target,
                                    qdrop, V, lam, valid,
                                    cert2, mt2, tier2)
    dt2 = time.perf_counter() - t

    n1_t1 = int(np.sum(tier1 == 1))
    n1_t3 = int(np.sum(tier1 == 3))
    n2_t1 = int(np.sum(tier2 == 1))
    n2_t2 = int(np.sum(tier2 == 2))
    n2_t3 = int(np.sum(tier2 == 3))
    print(f"\n[d={d} S={S} c={c_target}]")
    print(f"  v1 (2-tier): T1={n1_t1}, T3={n1_t3}, time={dt1:.3f}s")
    print(f"  v2 (4-tier): T1={n2_t1}, T2={n2_t2}, T3={n2_t3}, time={dt2:.3f}s")
    print(f"  Tier 3 reduction: {n1_t3} -> {n2_t3} ({100*(n1_t3-n2_t3)/max(n1_t3,1):.1f}%)")
    print(f"  Cert agreement: v1={int(np.sum(cert1))}, v2={int(np.sum(cert2))}")

    # CORRECTNESS: cert decisions must agree (since both are sound LBs and
    # v2 strictly tighter, every v1-cert must also be v2-cert)
    for b in range(B):
        if cert1[b]:
            assert cert2[b], f"cell {b}: v1 cert but v2 not — soundness mismatch"


def test_capture_rate_d14_hard():
    """At d=14, S=20, c=1.28 (genuinely hard, near val(14)=1.284 ceiling)."""
    rng = np.random.RandomState(2)
    d = 14
    S = 20
    c_target = 1.28
    qdrop = compute_qdrop_table(d)
    V, lam, valid = compute_window_eigen_table(d)

    B = 200
    batch = _random_canonical_batch(rng, d, S, B)

    # Warmup
    cert_out = np.zeros(5, dtype=np.int8)
    min_tv = np.zeros(5, dtype=np.float64)
    tier = np.zeros(5, dtype=np.int8)
    _box_certify_batch_adaptive(batch[:5], d, S, c_target, cert_out, min_tv, tier)
    _box_certify_batch_adaptive_v2(batch[:5], d, S, c_target,
                                    qdrop, V, lam, valid,
                                    cert_out, min_tv, tier)

    # v1
    cert1 = np.zeros(B, dtype=np.int8)
    mt1 = np.zeros(B, dtype=np.float64)
    tier1 = np.zeros(B, dtype=np.int8)
    t = time.perf_counter()
    _box_certify_batch_adaptive(batch, d, S, c_target, cert1, mt1, tier1)
    dt1 = time.perf_counter() - t

    # v2
    cert2 = np.zeros(B, dtype=np.int8)
    mt2 = np.zeros(B, dtype=np.float64)
    tier2 = np.zeros(B, dtype=np.int8)
    t = time.perf_counter()
    _box_certify_batch_adaptive_v2(batch, d, S, c_target,
                                    qdrop, V, lam, valid,
                                    cert2, mt2, tier2)
    dt2 = time.perf_counter() - t

    n1_t1 = int(np.sum(tier1 == 1))
    n1_t3 = int(np.sum(tier1 == 3))
    n2_t1 = int(np.sum(tier2 == 1))
    n2_t2 = int(np.sum(tier2 == 2))
    n2_t3 = int(np.sum(tier2 == 3))
    print(f"\n[d={d} S={S} c={c_target}]")
    print(f"  v1 (2-tier): T1={n1_t1}, T3={n1_t3}, time={dt1:.3f}s")
    print(f"  v2 (4-tier): T1={n2_t1}, T2={n2_t2}, T3={n2_t3}, time={dt2:.3f}s")

    for b in range(B):
        if cert1[b]:
            assert cert2[b], f"cell {b}: v1 cert but v2 not — soundness mismatch"


if __name__ == "__main__":
    test_soundness_d12()
    print("d=12 soundness OK")
    test_capture_rate_d12_hard()
    test_capture_rate_d14_hard()
