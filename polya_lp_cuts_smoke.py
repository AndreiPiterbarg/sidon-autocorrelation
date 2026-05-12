"""Smoke test the Fourier cut suite at d=4..16.

Compares baseline LP (CS-window cuts only) to LP + Fourier cuts.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np

from lasserre.polya_lp.runner import VAL_D_KNOWN
from lasserre.polya_lp.build import (
    BuildOptions, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.solve import solve_lp
from lasserre.polya_lp.symmetry import (
    project_window_set_to_z2_rescaled, z2_dim, project_M_to_z2,
    rescale_for_standard_simplex,
)
from lasserre.polya_lp.sidon_cuts import (
    default_fourier_cut_suite,
    check_cut_validity_montecarlo,
    bin_centers,
)


def project_cut_matrices_to_z2(cut_matrices, d):
    """Apply Z/2 projection + rescale to each cut matrix."""
    out = []
    for M in cut_matrices:
        M_sym = project_M_to_z2(M)
        M_rescaled = rescale_for_standard_simplex(M_sym, d)
        out.append(M_rescaled)
    return out


def main():
    print("Sidon Fourier cuts smoke test", flush=True)

    for d in (4, 6, 8):
        print(f"\n=== d = {d} (val_known = {VAL_D_KNOWN.get(d)}) ===", flush=True)

        # Baseline: CS windows + Z/2
        _, M_mats_orig = build_window_matrices(d)
        M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats_orig, d)
        d_eff = z2_dim(d)

        # Fourier cut suite (computed at original d; project to Z/2)
        suite = default_fourier_cut_suite(d, n_t0=7, n_s=5)
        cut_mats_eff = project_cut_matrices_to_z2(suite.matrices, d)

        print(f"  d_eff={d_eff}, baseline windows: {len(M_mats_eff)}, "
              f"Fourier cuts: {len(cut_mats_eff)}", flush=True)

        # Validity check (Monte Carlo on a few cuts)
        check = check_cut_validity_montecarlo(
            cut_mats_eff[0], M_mats_eff, n_samples=2000)
        print(f"  validity check (cut[0]={suite.labels[0]}):", flush=True)
        print(f"    cut_max={check['cut_max']:.4f}, "
              f"win_max_max={check['win_max_max']:.4f}, "
              f"excess_max={check['excess_max']:.4f}, "
              f"n_excess>0={check['n_excess_positive']}/{check['n_samples']}", flush=True)

        for R in (4, 6, 8):
            # Baseline
            t0 = time.time()
            opts_base = BuildOptions(R=R, verbose=False)
            build_base = build_handelman_lp(d_eff, M_mats_eff, opts_base)
            sol_base = solve_lp(build_base)
            t_base = time.time() - t0

            # With Fourier cuts
            t0 = time.time()
            combined = list(M_mats_eff) + list(cut_mats_eff)
            opts_cut = BuildOptions(R=R, verbose=False)
            build_cut = build_handelman_lp(d_eff, combined, opts_cut)
            sol_cut = solve_lp(build_cut)
            t_cut = time.time() - t0

            base_a = sol_base.alpha if sol_base.alpha is not None else float('nan')
            cut_a = sol_cut.alpha if sol_cut.alpha is not None else float('nan')
            improvement = cut_a - base_a if (sol_base.alpha and sol_cut.alpha) else float('nan')

            print(f"  R={R}: baseline={base_a:.6f} (n_W={len(M_mats_eff)}, "
                  f"{t_base:.1f}s); +cuts={cut_a:.6f} "
                  f"(n_W={len(combined)}, {t_cut:.1f}s); "
                  f"improvement={improvement:+.6f}", flush=True)


if __name__ == "__main__":
    main()
