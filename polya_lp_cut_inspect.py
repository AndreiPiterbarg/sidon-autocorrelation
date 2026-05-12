"""Inspect which cuts the LP uses at the optimum, to find invalid ones."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from lasserre.polya_lp.build import (
    BuildOptions, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.solve import solve_lp
from lasserre.polya_lp.symmetry import (
    project_window_set_to_z2_rescaled, z2_dim,
    project_M_to_z2, rescale_for_standard_simplex,
)
from lasserre.polya_lp.sidon_cuts import (
    default_fourier_cut_suite, check_cut_validity_montecarlo,
)


def project_to_z2(M, d):
    return rescale_for_standard_simplex(project_M_to_z2(M), d)


def main():
    d = 4
    R = 8
    _, M_mats_orig = build_window_matrices(d)
    M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats_orig, d)
    d_eff = z2_dim(d)

    suite = default_fourier_cut_suite(d, n_t0=7, n_s=5)
    cut_mats_eff = [project_to_z2(M, d) for M in suite.matrices]

    combined = list(M_mats_eff) + list(cut_mats_eff)
    print(f"d={d} R={R}: {len(M_mats_eff)} baseline + {len(cut_mats_eff)} cuts = {len(combined)}", flush=True)

    sol = solve_lp(build_handelman_lp(d_eff, combined, BuildOptions(R=R)))
    print(f"alpha = {sol.alpha:.6f}", flush=True)
    if sol.x is None:
        return

    # Extract lambda
    build = build_handelman_lp(d_eff, combined, BuildOptions(R=R))
    lam = sol.x[build.lambda_idx]
    print(f"|lambda| = {len(lam)}, sum = {lam.sum():.4f}", flush=True)
    print("Top-15 active cuts (by lambda):", flush=True)
    sorted_idx = np.argsort(-lam)
    for k in sorted_idx[:15]:
        if lam[k] < 1e-6:
            break
        if k < len(M_mats_eff):
            label = f"baseline_window[{k}]"
        else:
            label = suite.labels[k - len(M_mats_eff)]
        # Check the cut value at uniform
        nu_unif = np.full(d_eff, 1.0/d_eff)
        cut_unif = float(nu_unif @ combined[k] @ nu_unif)
        max_baseline_unif = max(float(nu_unif @ M @ nu_unif) for M in M_mats_eff)
        # Check max value of the matrix
        max_entry = float(combined[k].max())
        ratio = cut_unif / max_baseline_unif
        print(f"  k={k} lambda={lam[k]:.5f} {label}: "
              f"max_entry={max_entry:.3f}, "
              f"@uniform={cut_unif:.3f} (ratio_to_max_W={ratio:.3f})", flush=True)


if __name__ == "__main__":
    main()
