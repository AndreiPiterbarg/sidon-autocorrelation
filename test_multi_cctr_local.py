"""Local test: compare single-α vs multi-α CCTR at d=8, d=10.

Uses local hardware (16 cores). Tests TIGHT margin targets to stress the
bound stack. Reports node count, coverage, time, CCTR cert breakdown.
"""
import os
import sys
import time
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interval_bnb.parallel import parallel_branch_and_bound


def run_one(d, target, workers, mu_star, enable_multi):
    label = 'multi-α' if enable_multi else 'single-α'
    kw = (
        {'enable_multi_cctr': True} if enable_multi
        else {'enable_cctr': True}
    )
    print(f"\n--- d={d} t={target} {label} ---")
    t0 = time.time()
    r = parallel_branch_and_bound(
        d=d, target_c=target, workers=workers,
        init_split_depth=14,
        donate_threshold_floor=4,
        time_budget_s=180,
        cctr_mu_star=mu_star,
        verbose=False,
        **kw,
    )
    elapsed = time.time() - t0
    s = r.get('cctr_stats', {})
    print(f"  {label}: success={r['success']}, "
          f"nodes={r['total_nodes']:,}, "
          f"cov={100*r['coverage_fraction']:.4f}%, "
          f"time={elapsed:.1f}s, "
          f"CCTR sw_ne/joint/rlt={s.get('sw_ne_certs',0)}/"
          f"{s.get('joint_certs',0)}/{s.get('rlt_certs',0)}, "
          f"in_flight={r.get('in_flight_final', 0)}")
    return r


def get_mu_star(d):
    """Try saved mu_star_d{d}.npz, else compute fresh."""
    fname = f"mu_star_d{d}.npz"
    for p in [".", os.path.dirname(os.path.abspath(__file__))]:
        path = os.path.join(p, fname)
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            keys = list(data.keys())
            for k in ('mu_star', 'mu'):
                if k in keys:
                    return np.asarray(data[k], dtype=np.float64)
    # Compute fresh
    print(f"computing fresh mu_star at d={d}...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from kkt_correct_mu_star import find_kkt_correct_mu_star
    res = find_kkt_correct_mu_star(
        d=d, x_cap=1.0, n_starts=200, n_workers=4,
        top_K_phase2=20, top_K_phase3=5, target_residual=1e-6,
        verbose=False,
    )
    return res['mu_star']


def main():
    workers = min(mp.cpu_count(), 16)
    print(f"Using {workers} workers")
    print(f"{'='*70}")
    print(f"Single-α vs Multi-α CCTR: d=8, d=10 with tight margin")
    print(f"{'='*70}")

    # d=8 mu* = optimum from kkt pipeline: val(8)=1.20060
    # target 1.184 (margin 0.017 — matching d=20 problem)
    mu_d8 = get_mu_star(8)
    if mu_d8 is None:
        print("Failed to get mu_star d=8")
        return

    # d=10 val=1.22492; target 1.208 (margin 0.017)
    mu_d10 = get_mu_star(10)
    if mu_d10 is None:
        print("Failed to get mu_star d=10")
        return

    for d, mu, target in [(8, mu_d8, '1.184'), (10, mu_d10, '1.208')]:
        print(f"\n{'='*70}\nd={d}, target={target}\n{'='*70}")
        # Run single-α first
        r_single = run_one(d, target, workers, mu, False)
        # Then multi-α
        r_multi = run_one(d, target, workers, mu, True)
        # Comparison
        delta_nodes = r_multi['total_nodes'] - r_single['total_nodes']
        delta_cov = (r_multi['coverage_fraction']
                      - r_single['coverage_fraction']) * 100
        print(f"\n  Δ(nodes) = {delta_nodes:+,}, Δ(coverage) = {delta_cov:+.6f}%")
        # Verdict
        if r_multi['success'] and not r_single['success']:
            print("  >>> MULTI-α WINS: solved when single-α failed")
        elif r_multi['success'] and r_single['success']:
            print("  >>> Both succeed; multi-α "
                  f"{'faster' if r_multi['total_nodes'] < r_single['total_nodes'] else 'slower'}")
        elif not r_multi['success'] and not r_single['success']:
            print(f"  >>> Both fail; multi-α covered "
                  f"{r_multi['coverage_fraction']*100:.4f}%, single-α "
                  f"{r_single['coverage_fraction']*100:.4f}%")


if __name__ == "__main__":
    main()
