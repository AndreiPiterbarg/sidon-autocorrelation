"""POC test of the moment-LP with Z/2 reduction.

Compare:
  - Baseline mu-only LP (Z/2 reduced; existing code path)
  - Moment LP (mu', nu_orbit) with Krein-Poisson cuts (rigorous via first
    moments, projected to Z/2)
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
    project_window_set_to_z2_rescaled, z2_dim,
)
from lasserre.polya_lp.moment_lp import (
    build_moment_lp, solve_moment_lp,
    krein_poisson_cut, KernelCut,
    project_kernel_cut_to_z2, z2_delta,
)


def main():
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    print(f"Moment LP POC at d={d}, R={R} (with Z/2 reduction)", flush=True)
    print(f"Target val(d) = {VAL_D_KNOWN.get(d)}", flush=True)

    if d % 2 != 0:
        raise SystemExit("d must be even for Z/2 reduction")
    d_eff = d // 2

    # Window matrices: original d, then project + rescale for Z/2
    _, M_mats_orig = build_window_matrices(d)
    M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats_orig, d)
    print(f"  d={d} -> d_eff={d_eff}, {len(M_mats_orig)} windows -> "
          f"{len(M_mats_eff)} unique symmetric", flush=True)

    # Krein-Poisson cuts: compute at original d, project to Z/2
    s_values = [0.3, 0.5, 0.7, 0.9]
    t0_values = [0.0, 0.1, -0.1, 0.2, -0.2]
    print(f"  Computing Krein-Poisson cuts: {len(s_values)} s x {len(t0_values)} t0 = "
          f"{len(s_values)*len(t0_values)} cuts", flush=True)
    cuts_eff = []
    for s in s_values:
        for t0 in t0_values:
            kc_full = krein_poisson_cut(d, s, t0)
            kc_eff = project_kernel_cut_to_z2(kc_full, d)
            cuts_eff.append(kc_eff)
    delta_eff = z2_delta(d)
    print(f"  delta_eff = {delta_eff:.6f}", flush=True)

    # BASELINE: mu-only LP at d_eff (Z/2 reduced)
    print("\n--- BASELINE (mu-only LP at d_eff with Z/2) ---", flush=True)
    base_opts = BuildOptions(R=R, use_z2=True, verbose=False)
    base_build = build_handelman_lp(d_eff, M_mats_eff, base_opts)
    base_sol = solve_lp(base_build)
    print(f"  n_eq={base_build.A_eq.shape[0]}, n_vars={base_build.n_vars}, "
          f"build={base_build.build_wall_s:.2f}s, solve={base_sol.wall_s:.2f}s, "
          f"alpha={base_sol.alpha:.6f}", flush=True)

    # MOMENT LP at d_eff
    print("\n--- MOMENT LP (mu', nu_orbit) at d_eff with Krein-Poisson cuts ---", flush=True)
    mom_build = build_moment_lp(d_eff, R, M_mats_eff, cuts_eff,
                                delta=delta_eff, verbose=True)
    print(f"  build wall: {mom_build.build_wall_s:.2f}s", flush=True)
    print(f"  n_eq={mom_build.A_eq.shape[0]}, n_vars={mom_build.n_vars}, "
          f"nnz(A)={mom_build.n_nonzero_A}", flush=True)
    print(f"  Variable layout: alpha=1, lambda={mom_build.n_lambda}, "
          f"q={mom_build.n_q}, c={mom_build.n_c}", flush=True)
    print("  Solving with MOSEK...", flush=True)
    mom_sol = solve_moment_lp(mom_build)
    print(f"  status={mom_sol.status}, alpha={mom_sol.alpha}, wall={mom_sol.wall_s:.2f}s",
          flush=True)

    # Sanity check
    C1a_upper = 1.5029
    if mom_sol.alpha is not None:
        if mom_sol.alpha <= C1a_upper + 1e-6:
            print(f"\n  SANITY OK: alpha={mom_sol.alpha:.6f} <= C_{{1a}} <= {C1a_upper}",
                  flush=True)
        else:
            print(f"\n  *** SANITY FAIL: alpha={mom_sol.alpha:.6f} > C_{{1a}} <= {C1a_upper} ***",
                  flush=True)

    # Compare
    if base_sol.alpha is not None and mom_sol.alpha is not None:
        print(f"\n  Baseline alpha = {base_sol.alpha:.6f}", flush=True)
        print(f"  Moment LP alpha = {mom_sol.alpha:.6f}", flush=True)
        print(f"  Improvement: {mom_sol.alpha - base_sol.alpha:+.6f}", flush=True)


if __name__ == "__main__":
    main()
