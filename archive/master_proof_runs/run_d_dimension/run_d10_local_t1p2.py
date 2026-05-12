"""d=10 t=1.2 BnB on local 16-core / 32 GB Windows box.

Smoke test for the low-K fast SDP escalation wiring (parallel.py now
calls bound_sdp_escalation_int_ge_fast with INTERVAL_BNB_SDP_K).

Background: the LP-only sweep at d=10 stalled at 99.08% coverage in
120s on target 1.200 (data/stall_scaling_results.json E3_d10_t1.200).
val(10) ≈ 1.241 → slack to 1.2 is ≈0.04, so most boxes close on the
cheap cascade; SDP escalation handles the residual boundary boxes
that the epigraph LP could not.

Pipeline thresholds tuned for d=10 (boxes shrink fast):
  TIGHTEN_DEPTH=4    (default)
  TOPK_JOINT_DEPTH=8 / TOPK_JOINT_K=3
  ANCHOR_DEPTH=12
  EPIGRAPH_DEPTH=12  (cheap LP after first ~12 splits)
  PC_DEPTH=14        (variance-weighted split)
  LP_SPLIT_DEPTH=14
  SDP_DEPTH=16       (after epi LP at 12 has had its chance)
  SDP_K=16           (top-16 windows → full PSD; rest scalar)
  SDP_FILTER=0.02    (only invoke SDP if lp_val within 0.02 of target)
  SDP_TIME_LIMIT_S=5

8 workers x ~500 MB SDP cache per worker ≈ 4 GB peak.
"""
import os
import sys
import time
import json

# Cascade thresholds for d=10. val(10)≈1.2249, slack=0.025 (TIGHT).
# Cert rate must clear 50% to drain the live-box pool — every tier matters.
os.environ['INTERVAL_BNB_TOPK_JOINT_DEPTH'] = '8'
os.environ['INTERVAL_BNB_TOPK_JOINT_K'] = '3'
os.environ['INTERVAL_BNB_ANCHOR_DEPTH'] = '12'         # mu*-anchor cut (now mu_star_d10.npz exists)
os.environ['INTERVAL_BNB_EPIGRAPH_DEPTH'] = '12'
os.environ['INTERVAL_BNB_EPIGRAPH_FILTER'] = '0.02'
os.environ['INTERVAL_BNB_PC_DEPTH'] = '14'
os.environ['INTERVAL_BNB_LP_SPLIT_DEPTH'] = '14'

# NEW: per-box centroid anchor cut around mu* — catches sub-1e-3-width
# boxes near the optimum. Cost ~10ms/box at d=10.
os.environ['INTERVAL_BNB_CENTROID_DEPTH'] = '16'

# NEW: boundary-axis split forces a free-axis split when many axes are
# pinned at lo<=1e-12. mu* at d=10 has only 2 zero entries → lower the
# count threshold from default d//2=5 to 3 so it actually triggers.
os.environ['INTERVAL_BNB_BOUNDARY_SPLIT_DEPTH'] = '14'
os.environ['INTERVAL_BNB_BOUNDARY_AXIS_COUNT'] = '3'

# Fast SDP escalation, fires on residual deep boxes.
# Bumped K to 32 (full PSD at d=10's ~100 windows is cheap, ~1 GB cache)
# and FILTER to 0.05 so SDP catches boxes with weaker LP values too.
os.environ['INTERVAL_BNB_SDP_DEPTH'] = '16'
os.environ['INTERVAL_BNB_SDP_K'] = '32'
os.environ['INTERVAL_BNB_SDP_FILTER'] = '0.05'
os.environ['INTERVAL_BNB_SDP_TIME_LIMIT_S'] = '5.0'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interval_bnb.parallel import parallel_branch_and_bound


def main():
    d = 10
    target = '1.2'
    workers = 8
    time_budget_s = 1800  # 30 min

    print("#" * 72, flush=True)
    print(f"# d={d} t={target} BnB local smoke (low-K fast SDP)", flush=True)
    print(f"# Workers: {workers}  budget: {time_budget_s}s ({time_budget_s/60:.0f} min)", flush=True)
    print(f"# SDP_DEPTH={os.environ['INTERVAL_BNB_SDP_DEPTH']} "
          f"SDP_K={os.environ['INTERVAL_BNB_SDP_K']} "
          f"SDP_FILTER={os.environ['INTERVAL_BNB_SDP_FILTER']}", flush=True)
    print("#" * 72, flush=True)

    t0 = time.time()
    r = parallel_branch_and_bound(
        d=d, target_c=target,
        workers=workers,
        init_split_depth=12,
        donate_threshold_floor=2,
        time_budget_s=time_budget_s,
        verbose=True,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 72, flush=True)
    print(f"FINAL RESULT", flush=True)
    print("=" * 72, flush=True)
    print(f"  success: {r['success']}", flush=True)
    print(f"  total nodes: {r['total_nodes']:,}", flush=True)
    print(f"  certified leaves: {r['total_leaves_certified']:,}", flush=True)
    print(f"  max depth: {r['max_depth']}", flush=True)
    print(f"  coverage: {100 * r['coverage_fraction']:.6f}%", flush=True)
    print(f"  in_flight final: {r['in_flight_final']}", flush=True)
    print(f"  elapsed: {elapsed:.1f}s = {elapsed/60:.2f}min", flush=True)
    print(f"  EPIGRAPH stats: {r.get('epi_stats', {})}", flush=True)
    print(f"  ANCHOR stats: {r.get('anchor_stats', {})}", flush=True)
    print(f"  CENTROID stats: {r.get('centroid_stats', {})}", flush=True)
    print(f"  SDP stats: {r.get('sdp_stats', {})}", flush=True)

    serializable = {k: v for k, v in r.items()
                    if isinstance(v, (int, float, str, bool, list, dict))}
    serializable['wall_time_s'] = elapsed
    out_path = 'd10_t1p2_result.json'
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
