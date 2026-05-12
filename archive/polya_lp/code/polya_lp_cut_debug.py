"""Debug: which Fourier cut makes alpha shoot up at d=4?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
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
    krein_poisson_family, cohn_elkies_family,
    squared_trig_family, gorbachev_tikhonov_family,
    cosine_pd_family,
)


def project_to_z2(M, d):
    return rescale_for_standard_simplex(project_M_to_z2(M), d)


def main():
    d = 4
    R = 4
    _, M_mats_orig = build_window_matrices(d)
    M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats_orig, d)
    d_eff = z2_dim(d)

    # Baseline
    sol_base = solve_lp(build_handelman_lp(d_eff, M_mats_eff, BuildOptions(R=R)))
    print(f"Baseline d={d} R={R}: alpha = {sol_base.alpha:.6f}", flush=True)

    # Try ONE cut at a time
    print("\n--- Single-cut tests ---", flush=True)
    test_families = [
        ("krein s=0.5 t0=0", krein_poisson_family(d, [0.5], [0.0])),
        ("krein s=0.1 t0=-0.4", krein_poisson_family(d, [0.1], [-0.4])),
        ("krein s=0.95 t0=0", krein_poisson_family(d, [0.95], [0.0])),
        ("CE-Fejer t0=0", cohn_elkies_family(d, [0.0])),
        ("Cosine [1, 0.5]", cosine_pd_family(d, [[1.0, 0.5]])),
        ("Cosine [1, 0.5, 0.25]", cosine_pd_family(d, [[1.0, 0.5, 0.25]])),
        ("trig sq [1, 0.5]", squared_trig_family(d, [[1.0, 0.5]], [0.0])),
        ("GT w=0.25 t0=0", gorbachev_tikhonov_family(d, [0.25], [0.0])),
    ]
    for name, fam in test_families:
        if not fam:
            print(f"  {name}: empty family", flush=True)
            continue
        M_orig, lbl = fam[0]
        M_proj = project_to_z2(M_orig, d)
        # Sanity: print diagonal and max value
        max_val = float(np.max(M_proj))
        print(f"  {name} -> shape={M_proj.shape}, max={max_val:.4f}, "
              f"trace={np.trace(M_proj):.4f}", flush=True)
        # Check uniform value
        nu = np.full(d_eff, 1.0/d_eff)
        cut_val = float(nu @ M_proj @ nu)
        # Compare to max baseline window at uniform
        max_W_at_uniform = max(float(nu @ M @ nu) for M in M_mats_eff)
        print(f"    cut at uniform nu={cut_val:.4f}, "
              f"max baseline at uniform={max_W_at_uniform:.4f}", flush=True)
        # Add to LP
        combined = list(M_mats_eff) + [M_proj]
        sol = solve_lp(build_handelman_lp(d_eff, combined, BuildOptions(R=R)))
        improvement = sol.alpha - sol_base.alpha
        print(f"    LP alpha with cut = {sol.alpha:.6f} "
              f"(baseline {sol_base.alpha:.6f}, delta {improvement:+.4f})", flush=True)


if __name__ == "__main__":
    main()
