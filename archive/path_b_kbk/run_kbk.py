"""Driver script for Path B KBK SDP -- closes Hyp_R if SDP value < target."""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "C:/Users/andre/OneDrive - PennO365/Desktop/compact_sidon")

from path_b_kbk.kbk_sdp import (
    HYP_R_C_STAR, mu_M, hyp_r_target, solve_kbk_sdp, report,
)

M_MAX = 1.378  # restricted-Hyp_R target


def baseline_no_kbk():
    """Baseline: no KBK constraint -- recovers Path A's bang-bang ~ 0.927."""
    print("=" * 70)
    print("BASELINE (no KBK, no phase-aware): Bochner + Schur lift only")
    print("=" * 70)
    res = solve_kbk_sdp(
        M=M_MAX, N=10, K_trunc=10,
        use_kbk=False, use_phase_aware_bochner=False, use_y_toeplitz=True,
        K_upper_bound=2.0,
    )
    print(report(res))
    print()


def with_phase_aware_bochner_no_kbk():
    print("=" * 70)
    print("WITH phase-aware Bochner only (still no KBK)")
    print("=" * 70)
    res = solve_kbk_sdp(
        M=M_MAX, N=10, K_trunc=10,
        use_kbk=False, use_phase_aware_bochner=True, use_y_toeplitz=False,
        K_upper_bound=2.0,
    )
    print(report(res))
    print()


def full_kbk_sweep():
    """The main run: full SDP with KBK localizing."""
    print("=" * 70)
    print("FULL KBK SDP -- closes Hyp_R if c_emp_bound < c_* = 0.88254")
    print("=" * 70)
    print(f"M_max = {M_MAX}")
    print(f"mu(M_max) = {mu_M(M_MAX):.6f}")
    print(f"Hyp_R c_* = log(16)/pi = {HYP_R_C_STAR:.6f}")
    print(f"Hyp_R needs sum y_n^2 <= {hyp_r_target(M_MAX):.6f}")
    print()

    # Sweep N (truncation level) and K_upper_bound (since K is unbounded; trying scenarios)
    for K_ub in [2.0, 3.0, 5.0, 10.0]:
        for N in [8, 12, 16]:
            for K_trunc in [6, 10]:
                if K_trunc >= N:
                    continue
                t0 = time.time()
                res = solve_kbk_sdp(
                    M=M_MAX, N=N, K_trunc=K_trunc,
                    use_kbk=True,
                    use_phase_aware_bochner=True,
                    use_y_toeplitz=True,
                    K_upper_bound=K_ub,
                )
                dt = time.time() - t0
                print(f"K_ub={K_ub}, N={N}, K_trunc={K_trunc}, time={dt:.1f}s")
                print(report(res))
                print()


if __name__ == "__main__":
    baseline_no_kbk()
    with_phase_aware_bochner_no_kbk()
    full_kbk_sweep()
