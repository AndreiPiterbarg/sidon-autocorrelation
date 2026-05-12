#!/usr/bin/env python
r"""Benchmark: SCS-GPU (HSDE PyTorch) vs SCS-CPU on real Lasserre SDPs.

Builds the exact same problems from historical runs and compares wall time,
iteration count, and objective value.

Historical CPU baselines (from data/*.json and data/*.log):
  d=4  O2 bw=3:  68.8s   (scs_direct, n_y=70)
  d=4  O3 bw=3: 511.5s   (cvxpy_scs,  n_y=210)
  d=16 O2 bw=8:  ~2714s  (scs_cg,     n_y=4845, 1x153^2 + 16x17^2 PSD)
  d=32 O2 bw=16: 11.3s   (scs_cg per CG round, 1x561^2 + 32x33^2 PSD)

Usage:
    python tests/bench_gpu_vs_cpu.py
    python tests/bench_gpu_vs_cpu.py --configs small
    python tests/bench_gpu_vs_cpu.py --configs all
"""
import sys
import os
import time
import json
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =====================================================================
# Configuration
# =====================================================================

# (d, order, bandwidth) -- matches historical runs
CONFIGS_SMALL = [
    (4, 2, 3),   # n_y=70,   ~28 PSD cones, CPU baseline: 68.8s
    (4, 3, 3),   # n_y=210,  ~27 PSD cones, CPU baseline: 511.5s
]

CONFIGS_MEDIUM = [
    (8,  2, 4),  # bigger PSD cones
    (8,  3, 4),
]

CONFIGS_LARGE = [
    (16, 2, 8),  # 1x153^2 + 16x17^2, CPU baseline: 2714s total
    (16, 2, 15), # full bandwidth
]

CONFIGS_ALL = CONFIGS_SMALL + CONFIGS_MEDIUM + CONFIGS_LARGE

MAX_ITERS = 50000
EPS = 1e-6


# =====================================================================
# Historical baselines from data/*.json and data/*.log
# =====================================================================

BASELINES = {
    (4, 2, 3): {'time': 68.76, 'lb': 1.0000, 'solver': 'scs_direct',
                'source': 'data/result_d4_o2_bw3_scs.json'},
    (4, 3, 3): {'time': 511.45, 'lb': 1.0239, 'solver': 'cvxpy_scs',
                'source': 'data/result_d4_o3_bw3_scs.json'},
    (16, 2, 8): {'time': 2713.6, 'lb': 1.1329, 'solver': 'scs_cg (3 CG rounds)',
                 'source': 'data/l2_sweep.log'},
}


def build_problem(d, order, bw):
    """Build the Lasserre SDP problem data (A, b, c, cone)."""
    from lasserre_highd import _precompute_highd, _build_banded_cliques
    from run_scs_direct import build_base_problem

    print(f"  Building problem d={d} O{order} bw={bw}...", flush=True)
    t0 = time.time()
    cliques = _build_banded_cliques(d, bw)
    P = _precompute_highd(d, order, cliques, verbose=False)
    A, b, c, cone, meta = build_base_problem(P, add_upper_loc=True)
    dt = time.time() - t0
    print(f"  Built in {dt:.2f}s", flush=True)

    # Summarize cone structure
    psd_sizes = cone.get('s', [])
    from collections import Counter
    size_counts = Counter(psd_sizes)
    psd_summary = ", ".join(f"{cnt}x{sz}^2" for sz, cnt in sorted(size_counts.items(), reverse=True))

    print(f"  Dimensions: n={A.shape[1]}, m={A.shape[0]}, nnz={A.nnz:,}")
    print(f"  Cones: z={cone.get('z',0)}, l={cone.get('l',0)}, "
          f"PSD=[{psd_summary}]")

    return A, b, c, cone, meta


def bench_cpu(A, b, c, cone, meta):
    """Run SCS CPU (direct solver)."""
    import scs

    # Ensure int32 indices for SCS
    A_32 = A.copy()
    A_32.indices = A_32.indices.astype(np.int32)
    A_32.indptr = A_32.indptr.astype(np.int32)
    cone_32 = {}
    for k, v in cone.items():
        if isinstance(v, list):
            cone_32[k] = [int(x) for x in v]
        else:
            cone_32[k] = int(v)

    data = {'A': A_32, 'b': b.astype(np.float64), 'c': c.astype(np.float64)}
    t_col = meta['t_col']

    result = {}
    for mode_name, extra_kw in [('cpu_direct', {}),
                                 ('cpu_indirect', {'linear_solver': 'indirect'})]:
        try:
            solver = scs.SCS(data, cone_32, max_iters=MAX_ITERS,
                             eps_abs=EPS, eps_rel=EPS, verbose=False,
                             **extra_kw)
            t0 = time.time()
            sol = solver.solve()
            dt = time.time() - t0
            obj = float(sol['x'][t_col]) if sol['x'] is not None else None
            result[mode_name] = {
                'time': dt,
                'iters': sol['info']['iter'],
                'status': sol['info']['status'],
                'obj': obj,
                'setup_time': sol['info'].get('setup_time', 0) / 1000,
                'solve_time': sol['info'].get('solve_time', 0) / 1000,
            }
        except Exception as e:
            result[mode_name] = {'time': None, 'error': str(e)}

    return result


def bench_gpu(A, b, c, cone, meta):
    """Run SCS-GPU (our PyTorch HSDE implementation)."""
    from scs_gpu import scs_gpu_solve
    t_col = meta['t_col']

    try:
        # Warmup run (JIT, CUDA context)
        _ = scs_gpu_solve(A, b, c, cone, device='cuda',
                          max_iters=100, eps_abs=1e-2, eps_rel=1e-2,
                          verbose=False)

        t0 = time.time()
        sol = scs_gpu_solve(A, b, c, cone, device='cuda',
                            max_iters=MAX_ITERS, eps_abs=EPS, eps_rel=EPS,
                            verbose=False)
        dt = time.time() - t0
        obj = float(sol['x'][t_col]) if sol['x'] is not None else None
        return {
            'gpu_torch': {
                'time': dt,
                'iters': sol['info']['iter'],
                'status': sol['info']['status'],
                'obj': obj,
                'setup_time': sol['info'].get('setup_time', 0),
                'solve_time': sol['info'].get('solve_time', 0),
            }
        }
    except Exception as e:
        return {'gpu_torch': {'time': None, 'error': str(e)}}


def print_results(config, cpu_results, gpu_results, baseline):
    """Pretty-print comparison table."""
    d, order, bw = config
    print()
    print(f"  {'Solver':>15} | {'Time':>9} | {'Setup':>7} | {'Solve':>7} | "
          f"{'Iters':>6} | {'Obj':>10} | {'Status'}")
    print(f"  {'-'*15}-+-{'-'*9}-+-{'-'*7}-+-{'-'*7}-+-"
          f"{'-'*6}-+-{'-'*10}-+-{'-'*15}")

    all_results = {}
    all_results.update(cpu_results)
    all_results.update(gpu_results)

    for name, r in all_results.items():
        if r.get('time') is None:
            print(f"  {name:>15} | {'FAILED':>9} | {r.get('error','')}")
            continue
        setup = r.get('setup_time', 0)
        solve = r.get('solve_time', r['time'])
        obj_str = f"{r['obj']:.6f}" if r['obj'] is not None else "N/A"
        print(f"  {name:>15} | {r['time']:8.2f}s | {setup:6.2f}s | "
              f"{solve:6.2f}s | {r['iters']:>6} | {obj_str:>10} | "
              f"{r['status']}")

    if baseline:
        print(f"  {'BASELINE':>15} | {baseline['time']:8.2f}s | "
              f"{'---':>7} | {'---':>7} | {'---':>6} | "
              f"{baseline['lb']:10.4f} | {baseline['solver']}")

    # Speedup calculation
    cpu_best = None
    for name in ['cpu_direct', 'cpu_indirect']:
        r = all_results.get(name, {})
        if r.get('time') is not None:
            if cpu_best is None or r['time'] < cpu_best:
                cpu_best = r['time']

    gpu_r = all_results.get('gpu_torch', {})
    if cpu_best and gpu_r.get('time'):
        ratio = cpu_best / gpu_r['time']
        if ratio > 1:
            print(f"\n  => GPU is {ratio:.1f}x FASTER than best CPU")
        else:
            print(f"\n  => GPU is {1/ratio:.1f}x SLOWER than best CPU")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default='small',
                        choices=['small', 'medium', 'large', 'all'],
                        help='Which problem configs to run')
    parser.add_argument('--gpu-only', action='store_true',
                        help='Skip CPU benchmarks (use baselines for comparison)')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Skip GPU benchmarks')
    parser.add_argument('--save', default=None,
                        help='Save results to JSON file')
    args = parser.parse_args()

    configs = {
        'small': CONFIGS_SMALL,
        'medium': CONFIGS_SMALL + CONFIGS_MEDIUM,
        'large': CONFIGS_LARGE,
        'all': CONFIGS_ALL,
    }[args.configs]

    print("=" * 70)
    print("SCS GPU vs CPU Benchmark — Real Lasserre SDPs")
    print(f"Date: {datetime.now().isoformat()}")
    print(f"max_iters={MAX_ITERS}, eps={EPS}")
    print(f"Configs: {args.configs} ({len(configs)} problems)")
    print("=" * 70)

    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("GPU: NOT AVAILABLE")

    try:
        import scs
        print(f"SCS: {scs.__version__}")
    except ImportError:
        print("SCS: NOT INSTALLED")

    print()

    all_results = {}

    for d, order, bw in configs:
        key = f"d{d}_O{order}_bw{bw}"
        print("=" * 70)
        print(f"d={d}  order={order}  bandwidth={bw}")
        print("=" * 70)

        try:
            A, b, c, cone, meta = build_problem(d, order, bw)
        except Exception as e:
            print(f"  BUILD FAILED: {e}")
            continue

        cpu_results = {}
        gpu_results = {}

        if not args.gpu_only:
            print("\n  Running CPU benchmarks...")
            cpu_results = bench_cpu(A, b, c, cone, meta)

        if not args.cpu_only:
            print("  Running GPU benchmark...")
            gpu_results = bench_gpu(A, b, c, cone, meta)

        baseline = BASELINES.get((d, order, bw))
        print_results((d, order, bw), cpu_results, gpu_results, baseline)

        all_results[key] = {
            'config': {'d': d, 'order': order, 'bw': bw},
            'problem': {'n': A.shape[1], 'm': A.shape[0], 'nnz': A.nnz,
                        'cone': {k: v if not isinstance(v, list) else len(v)
                                 for k, v in cone.items()}},
            'cpu': cpu_results,
            'gpu': gpu_results,
            'baseline': baseline,
        }
        print()

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Config':<20} | {'CPU (best)':>10} | {'GPU':>10} | "
          f"{'Speedup':>8} | {'Baseline':>10}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}")

    for key, res in all_results.items():
        cpu_best = None
        for name in ['cpu_direct', 'cpu_indirect']:
            r = res['cpu'].get(name, {})
            if r.get('time') is not None:
                if cpu_best is None or r['time'] < cpu_best:
                    cpu_best = r['time']

        gpu_t = res['gpu'].get('gpu_torch', {}).get('time')
        bl_t = res['baseline']['time'] if res['baseline'] else None

        cpu_str = f"{cpu_best:.1f}s" if cpu_best else "N/A"
        gpu_str = f"{gpu_t:.1f}s" if gpu_t else "N/A"
        bl_str = f"{bl_t:.1f}s" if bl_t else "N/A"

        if cpu_best and gpu_t:
            ratio = cpu_best / gpu_t
            sp_str = f"{ratio:.1f}x" if ratio > 1 else f"1/{1/ratio:.1f}x"
        else:
            sp_str = "N/A"

        print(f"  {key:<20} | {cpu_str:>10} | {gpu_str:>10} | "
              f"{sp_str:>8} | {bl_str:>10}")

    if args.save:
        with open(args.save, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'max_iters': MAX_ITERS, 'eps': EPS,
                'results': all_results,
            }, f, indent=2, default=str)
        print(f"\nResults saved to {args.save}")


if __name__ == '__main__':
    main()
