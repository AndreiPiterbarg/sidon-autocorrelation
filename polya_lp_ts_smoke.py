"""Smoke test: term-sparsity restriction must give SAME alpha as full LP.

This is the soundness check for the Newton-polytope restriction.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np

from lasserre.polya_lp.runner import run_one
from lasserre.polya_lp.build import (
    BuildOptions, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.solve import solve_lp
from lasserre.polya_lp.symmetry import (
    project_window_set_to_z2_rescaled, z2_dim,
)
from lasserre.polya_lp.term_sparsity import build_term_sparsity_support


def main():
    print("Soundness test: restricted LP must equal full LP", flush=True)
    print(f"{'d':>3} {'R':>3} {'full_alpha':>12} {'ts_alpha':>12} "
          f"{'diff':>10} {'full_n_eq':>10} {'ts_n_eq':>10} "
          f"{'reduction':>10} {'ok':>4}", flush=True)
    for d in (4, 6, 8):
        for R in (4, 6, 8):
            # Build M_mats with Z/2 reduction
            _, M_mats_orig = build_window_matrices(d)
            M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats_orig, d)
            d_eff = z2_dim(d)

            # Full LP
            opts_full = BuildOptions(R=R, use_z2=True, verbose=False)
            full = build_handelman_lp(d_eff, M_mats_eff, opts_full)
            sol_full = solve_lp(full)

            # Term-sparsity restricted LP
            ts = build_term_sparsity_support(M_mats_eff, R)
            opts_ts = BuildOptions(
                R=R, use_z2=True, verbose=False,
                restricted_Sigma_R=ts.Sigma_R,
                restricted_B_R=ts.B_R,
            )
            tsbuild = build_handelman_lp(d_eff, M_mats_eff, opts_ts)
            sol_ts = solve_lp(tsbuild)

            full_a = sol_full.alpha if sol_full.alpha is not None else float('nan')
            ts_a = sol_ts.alpha if sol_ts.alpha is not None else float('nan')
            diff = abs(full_a - ts_a) if (sol_full.alpha is not None and sol_ts.alpha is not None) else float('nan')
            ok = "Y" if diff < 1e-6 else "N"
            reduction = (full.A_eq.shape[0] / max(tsbuild.A_eq.shape[0], 1)
                         if tsbuild.A_eq.shape[0] else float('inf'))

            print(f"{d:>3} {R:>3} {full_a:>12.6f} {ts_a:>12.6f} "
                  f"{diff:>10.2e} {full.A_eq.shape[0]:>10} {tsbuild.A_eq.shape[0]:>10} "
                  f"{reduction:>10.2f} {ok:>4}", flush=True)


if __name__ == "__main__":
    main()
