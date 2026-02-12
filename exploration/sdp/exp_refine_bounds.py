"""Refine bounds at P=15-18 with fine tolerance.

The coarse tolerance (1e-3) used in exp_fast_lasserre.py may have loose bounds.
Here we refine the most important P values with tighter tolerance to get
the best possible bounds within the Lasserre Level-2 + simplex cuts framework.

Also: the P=19 bound (1.478872) > P=18 (1.475327) is anomalous — V(P) should
decrease with P. This likely means P=19 was cut short. We verify.
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from exploration.sdp.baseline_results import BASELINE, compare_with_baseline
from exploration.sdp.exp_fast_lasserre import build_lasserre_problem, solve_with_bracket


if __name__ == '__main__':
    print("=" * 80, flush=True)
    print("REFINEMENT: Tighter bounds at P=15-18", flush=True)
    print("=" * 80, flush=True)

    # Phase 1: Refine P=15 (biggest improvement over baseline: +0.004564)
    # Use fine tolerance (1e-6) with generous time
    for P in [15, 16, 17, 18]:
        print(f"\n{'='*60}", flush=True)
        print(f"P={P}: Building problem...", flush=True)
        pd = build_lasserre_problem(P, use_simplex_cuts=True)
        print(f"  d={pd['d']}, n_mom={pd['n_mom']}, setup={pd['setup_time']:.1f}s", flush=True)

        shor = 2 * P / (2 * P - 1)
        # Use known coarse results for tighter initial bracket
        coarse_bounds = {15: 1.490516, 16: 1.483826, 17: 1.481782, 18: 1.475327}
        eta_hi = coarse_bounds.get(P, 1.5) + 0.001
        eta_lo = shor

        # First: coarse search to narrow
        print(f"  Coarse search: [{eta_lo:.4f}, {eta_hi:.4f}]", flush=True)
        bound_coarse, t_coarse, n_coarse = solve_with_bracket(
            pd, eta_lo, eta_hi, eta_tol=1e-3, max_time=300)
        if bound_coarse is None:
            print(f"  FAILED at coarse level", flush=True)
            continue
        print(f"  Coarse: {bound_coarse:.6f} in {t_coarse:.1f}s ({n_coarse} iters)", flush=True)

        # Then: fine search within narrow bracket
        eta_lo_fine = max(shor, bound_coarse - 0.002)
        eta_hi_fine = bound_coarse + 0.001
        print(f"  Fine search: [{eta_lo_fine:.6f}, {eta_hi_fine:.6f}]", flush=True)

        bound_fine, t_fine, n_fine = solve_with_bracket(
            pd, eta_lo_fine, eta_hi_fine, eta_tol=1e-5, max_time=300)

        total_time = pd['setup_time'] + t_coarse + t_fine
        if bound_fine is not None:
            compare_with_baseline(P, bound_fine, total_time, 'Refined-L2+SC')
            print(f"  Fine: {bound_fine:.8f} in {t_fine:.1f}s ({n_fine} iters)", flush=True)
            print(f"  Total: {total_time:.1f}s", flush=True)
        else:
            print(f"  Fine refinement FAILED", flush=True)
            if bound_coarse is not None:
                compare_with_baseline(P, bound_coarse, total_time, 'Coarse-L2+SC')

    print("\n" + "=" * 80, flush=True)
    print("DONE", flush=True)
