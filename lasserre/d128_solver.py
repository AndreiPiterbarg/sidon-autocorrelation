"""d=128 sparse-clique Lasserre feasibility SDP solver.

Thin wrapper around ``lasserre.d64_solver.solve_sparse_farkas_at_t``.
Soundness, build, and dual-extraction logic are identical; only the
default parameters change. See ``lasserre/d64_solver.py`` and
``lasserre/d64_d128_plan.md`` for the design and pod requirements
(d=128 needs ≥ 1 TB RAM).
"""
from __future__ import annotations

from lasserre.d64_solver import solve_sparse_farkas_at_t, SparseSolveResult


def solve_d128(t_test: float = 1.281, *,
                order: int = 2,
                bandwidth: int = 16,
                n_threads: int = 0,
                verbose: bool = True) -> SparseSolveResult:
    """d=128 sparse SDP at fixed t_test. Defaults: order 2, b=16, all threads."""
    return solve_sparse_farkas_at_t(
        d=128, order=order, bandwidth=bandwidth, t_test=t_test,
        n_threads=n_threads, verbose=verbose,
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--order', type=int, default=2)
    ap.add_argument('--bandwidth', type=int, default=16)
    ap.add_argument('--t_test', type=float, default=1.281)
    ap.add_argument('--n_threads', type=int, default=0)
    args = ap.parse_args()

    res = solve_d128(
        t_test=args.t_test, order=args.order,
        bandwidth=args.bandwidth, n_threads=args.n_threads,
    )
    print(f"\n=== d=128 status: {res.status} ===")
    print(f"  build_time:  {res.build_time:.2f} s")
    print(f"  solver_time: {res.solver_time:.2f} s")
    print(f"  primal_obj:  {res.primal_obj}")
    print(f"  dual_obj:    {res.dual_obj}")
    print(f"  gap:         {res.gap}")


if __name__ == '__main__':
    main()
