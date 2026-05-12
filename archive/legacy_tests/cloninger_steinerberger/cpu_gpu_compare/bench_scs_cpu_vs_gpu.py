#!/usr/bin/env python
"""Benchmark SCS CPU vs GPU on a single SCS solve (no CG, no bisection).

Builds the base Lasserre SDP at a given (d, order, bw) and solves once
with SCS on CPU and (if available) GPU. Reports wall time and iterations.

Usage:
    python tests/bench_scs_cpu_vs_gpu.py
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_highd import _precompute_highd, _build_banded_cliques
from run_scs_direct import build_base_problem
import scs


CONFIGS = [
    (4, 2, 3),
    (4, 3, 3),
    (8, 2, 4),
    (8, 3, 4),
]

MAX_ITERS = 50000
EPS = 1e-6


def bench_one(d, order, bw):
    cliques = _build_banded_cliques(d, bw)
    P = _precompute_highd(d, order, cliques, verbose=False)
    A, b, c, cone, meta = build_base_problem(P, add_upper_loc=True)

    print(f"  Problem: {A.shape[0]} rows x {A.shape[1]} cols, "
          f"nnz={A.nnz}, PSD cones={cone['s']}")

    # SCS GPU build uses int32 — ensure ALL integers are int32
    A_32 = A.copy()
    A_32.indices = A_32.indices.astype(np.int32)
    A_32.indptr = A_32.indptr.astype(np.int32)
    # Cone sizes must also be plain Python int (not numpy int64)
    cone_32 = {}
    for k, v in cone.items():
        if isinstance(v, list):
            cone_32[k] = [int(x) for x in v]
        else:
            cone_32[k] = int(v)
    cone = cone_32
    data = {'A': A_32, 'b': b.astype(np.float64), 'c': c.astype(np.float64)}
    results = {}

    # Use high-level scs.SCS API with linear_solver kwarg (not boolean flags)
    modes = [
        ('cpu_direct', {}),
        ('cpu_indirect', {'linear_solver': 'indirect'}),
        ('gpu_cudss', {'linear_solver': 'cudss'}),
    ]
    for mode_name, extra_kw in modes:
        try:
            kw = dict(max_iters=MAX_ITERS, eps_abs=EPS, eps_rel=EPS,
                      verbose=False, **extra_kw)
            solver = scs.SCS(data, cone, **kw)
            t0 = time.time()
            sol = solver.solve()
            dt = time.time() - t0
            obj = float(sol['x'][meta['t_col']]) if sol['x'] is not None else None
            results[mode_name] = {
                'time': dt, 'iters': sol['info']['iter'],
                'status': sol['info']['status'], 'obj': obj,
            }
            label = f"{mode_name:>15}"
            print(f"  {label}: {dt:7.2f}s  {sol['info']['iter']:>6} iters  "
                  f"obj={obj:.6f}  status={sol['info']['status']}")
        except Exception as e:
            print(f"  {mode_name:>15}: FAILED ({e})")

    return results


def main():
    print(f"SCS {scs.__version__} — CPU vs GPU benchmark")
    print(f"max_iters={MAX_ITERS}, eps={EPS}")
    print()

    for d, order, bw in CONFIGS:
        print(f"d={d} O{order} bw={bw}:")
        bench_one(d, order, bw)
        print()


if __name__ == '__main__':
    main()
