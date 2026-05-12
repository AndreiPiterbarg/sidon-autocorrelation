"""Tests for Speedup #3: closed-form Shor + κ-shift region aggregation.

Runs at the regimes that matter (d=12, c near val(d) ceiling). Designed
to complete in roughly 60s.

The region primitive _region_certify_shor takes (lo, hi) per coordinate
and returns a sound LB on min over the polytope P = {mu_c + delta :
lo <= delta <= hi, sum delta = 0}. Applied to a single cell, it equals
the trust-region cell-cert. Applied to a region containing K cells, it
amortizes one QP across K cells.
"""
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coarse_cascade_prover import (  # noqa: E402
    _box_certify_batch_adaptive,
    _box_certify_batch_adaptive_v2,
    _box_certify_cell_vertex,
    _box_certify_cell_trust_region,
    _region_certify_shor,
    compute_qdrop_table,
    compute_window_eigen_table,
)


def _random_canonical_batch(rng, d, S, B, x_cap=None):
    out = np.zeros((B, d), dtype=np.int32)
    n_done = 0
    attempts = 0
    while n_done < B:
        cuts = sorted(rng.choice(np.arange(1, S + d), size=d - 1, replace=False))
        cuts = [0] + list(cuts) + [S + d]
        parts = np.array([cuts[i + 1] - cuts[i] - 1 for i in range(d)], dtype=np.int32)
        attempts += 1
        if x_cap is not None and (parts > x_cap).any():
            continue
        out[n_done] = parts
        n_done += 1
    return out


def _region_aggregation_certify(batch, d, S, c_target, V, lam, valid, qdrop, K=16):
    """Lex-sort batch, group into chunks of K, region-cert each group.

    Returns (n_region_cert, n_per_cell_cert, n_failed, t_region, t_per_cell).
    """
    h = 1.0 / (2.0 * S)
    sorted_idx = np.lexsort(batch.T)
    sorted_batch = batch[sorted_idx]
    B = len(sorted_batch)

    n_groups = (B + K - 1) // K
    n_region_cert = 0
    n_per_cell_cert = 0
    n_failed = 0
    failed_indices = []

    t_region_start = time.perf_counter()
    for g in range(n_groups):
        start = g * K
        end = min(start + K, B)
        group = sorted_batch[start:end].astype(np.float64) / S

        # Build region [mu_min - h, mu_max + h] for centers, then express
        # as box around mu_center = (max+min)/2 with delta box bounds
        mu_min = group.min(axis=0)
        mu_max = group.max(axis=0)
        mu_c = 0.5 * (mu_min + mu_max)
        # delta_min / max for cells in the group
        lo = mu_min - h - mu_c
        hi = mu_max + h - mu_c
        # Clip lo to keep mu_c + lo >= 0
        for i in range(d):
            if mu_c[i] + lo[i] < 0.0:
                lo[i] = -mu_c[i]

        cert, _ = _region_certify_shor(mu_c, lo, hi, d, c_target,
                                        V, lam, valid)
        if cert:
            n_region_cert += end - start
        else:
            failed_indices.extend(range(start, end))
    t_region = time.perf_counter() - t_region_start

    # Fallback to per-cell trust-region cert for failed regions
    t_per_cell_start = time.perf_counter()
    delta_q = 1.0 / S
    for idx in failed_indices:
        mu = sorted_batch[idx].astype(np.float64) / S
        cert, _ = _box_certify_cell_trust_region(mu, d, delta_q, c_target,
                                                   V, lam, valid)
        if cert:
            n_per_cell_cert += 1
        else:
            # Final fallback: vertex enum
            cert_v, _ = _box_certify_cell_vertex(mu, d, delta_q, c_target)
            if cert_v:
                n_per_cell_cert += 1
            else:
                n_failed += 1
    t_per_cell = time.perf_counter() - t_per_cell_start

    return n_region_cert, n_per_cell_cert, n_failed, t_region, t_per_cell


def test_speedup3_region_aggregation_d12_60s():
    """Real ~60s benchmark at d=12 with c-target near val(d) ceiling.

    Compares:
      v1: existing 2-tier (Phase1 + vertex)
      v2: new 4-tier (Phase1 + Phase1-spec + trust-region + vertex)
      region: lex-sorted groups of K cells, region cert + per-cell fallback

    Demonstrates the speedup of region aggregation when many cells live in
    'easy' regions (TV well above c_target).
    """
    rng = np.random.RandomState(42)
    d = 12
    S = 20
    c_target = 1.22  # closer to val(12)=1.271 — Phase 1 will work harder
    B = 50000  # ~60s for the v1 baseline at d=12

    qdrop = compute_qdrop_table(d)
    V, lam, valid = compute_window_eigen_table(d)
    x_cap = int(np.floor(S * np.sqrt(c_target / d)))
    batch = _random_canonical_batch(rng, d, S, B, x_cap=x_cap)

    # Warmup compile
    cert_w = np.zeros(10, dtype=np.int8)
    mt_w = np.zeros(10, dtype=np.float64)
    tier_w = np.zeros(10, dtype=np.int8)
    _box_certify_batch_adaptive(batch[:10], d, S, c_target, cert_w, mt_w, tier_w)
    _box_certify_batch_adaptive_v2(batch[:10], d, S, c_target,
                                    qdrop, V, lam, valid,
                                    cert_w, mt_w, tier_w)
    _region_aggregation_certify(batch[:32], d, S, c_target, V, lam, valid, qdrop, K=16)

    # v1 timing
    cert1 = np.zeros(B, dtype=np.int8)
    mt1 = np.zeros(B, dtype=np.float64)
    tier1 = np.zeros(B, dtype=np.int8)
    t = time.perf_counter()
    _box_certify_batch_adaptive(batch, d, S, c_target, cert1, mt1, tier1)
    t1 = time.perf_counter() - t

    # v2 timing
    cert2 = np.zeros(B, dtype=np.int8)
    mt2 = np.zeros(B, dtype=np.float64)
    tier2 = np.zeros(B, dtype=np.int8)
    t = time.perf_counter()
    _box_certify_batch_adaptive_v2(batch, d, S, c_target,
                                    qdrop, V, lam, valid,
                                    cert2, mt2, tier2)
    t2 = time.perf_counter() - t

    # Region aggregation (Speedup #3)
    n_region, n_perc, n_fail, t_reg, t_pc = _region_aggregation_certify(
        batch, d, S, c_target, V, lam, valid, qdrop, K=16)
    t3 = t_reg + t_pc
    n_total_cert3 = n_region + n_perc

    # Tier distribution for v2
    n_t1_v2 = int(np.sum(tier2 == 1))
    n_t2_v2 = int(np.sum(tier2 == 2))
    n_t3_v2 = int(np.sum(tier2 == 3))

    print(f"\n[d={d} S={S} c={c_target} B={B}, x_cap={x_cap}]")
    print(f"  v1 (2-tier):       certified={int(np.sum(cert1))}, time={t1:.2f}s")
    print(f"    tier dist: T1={int(np.sum(tier1==1))}, T3={int(np.sum(tier1==3))}")
    print(f"  v2 (4-tier):       certified={int(np.sum(cert2))}, time={t2:.2f}s")
    print(f"    tier dist: T1={n_t1_v2}, T2={n_t2_v2}, T3={n_t3_v2}")
    print(f"  Speedup #3 region: certified={n_total_cert3}, time={t3:.2f}s "
          f"(t_region={t_reg:.2f}s + t_perc={t_pc:.2f}s)")
    print(f"    region cert: {n_region}/{B} = {100*n_region/B:.1f}% of cells skipped per-cell")
    print(f"    per-cell fallback: {n_perc} cells (failed: {n_fail})")
    print(f"  v2 vs v1 speedup: {t1/t2:.2f}x")
    print(f"  region vs v2 speedup: {t2/t3:.2f}x")
    print(f"  region vs v1 speedup: {t1/t3:.2f}x")

    # Soundness check #1: any cell certified by v1 must be by v2 too
    # (v2 uses strictly tighter bounds across all tiers)
    n_v1_only = 0
    for b in range(B):
        if cert1[b] and not cert2[b]:
            n_v1_only += 1
    assert n_v1_only == 0, f"{n_v1_only} cells: v1 cert but not v2 — UNSOUND v2"

    # Soundness check #2: For a truly-failing test (c > val(d)) using a
    # smaller batch, region cert must also fail on cells where vertex fails
    # by a clear margin (vertex_lb < c - epsilon). We don't assert anything
    # here against the main batch (where region may legitimately certify
    # cells that vertex's max-min LB doesn't, since they're DIFFERENT
    # sound LBs on cell-min: vertex computes max_W min_cell TV_W while
    # region computes min over polytope of joint TV).


def test_speedup3_region_sound_above_val_d():
    """When c > val(d), some cells genuinely have cell-min < c. Region cert
    must fail those — verify by checking trivially-failing cells where the
    vertex max-min LB is much below c."""
    rng = np.random.RandomState(99)
    d = 12
    S = 18
    c_target = 1.40  # > val(12) = 1.271 — many cells truly fail
    delta_q = 1.0 / S
    V, lam, valid = compute_window_eigen_table(d)

    n_checked = 0
    n_violations = 0
    for trial in range(200):
        cuts = sorted(rng.choice(np.arange(1, S + d), size=d - 1, replace=False))
        cuts = [0] + list(cuts) + [S + d]
        parts = np.array([cuts[i + 1] - cuts[i] - 1 for i in range(d)],
                         dtype=np.float64)
        mu = parts / S
        cert_v, lb_v = _box_certify_cell_vertex(mu, d, delta_q, c_target)
        # Skip cells trivially passing on max-min (good cells)
        if cert_v:
            continue
        # If vertex_lb is well below c (margin > 0.05), the cell genuinely
        # cannot have TV-min >= c with our bound types. Region cert must
        # also fail.
        if lb_v < c_target - 0.05:
            h = delta_q / 2.0
            lo = np.array([max(-h, -mu[i]) for i in range(d)])
            hi = np.array([min(h, 1.0 - mu[i]) for i in range(d)])
            cert_r, lb_r = _region_certify_shor(mu, lo, hi, d, c_target,
                                                 V, lam, valid)
            n_checked += 1
            # Both must fail (sound bounds, both below c)
            if cert_r:
                n_violations += 1
        if n_checked >= 30:
            break
    assert n_violations == 0, (
        f"{n_violations}/{n_checked} cells where vertex_lb << c but "
        f"region cert succeeded — region UNSOUND")


if __name__ == "__main__":
    test_speedup3_region_aggregation_d12_60s()
    test_speedup3_region_sound_above_val_d()
    print("speedup3 sound-above-val-d OK")
