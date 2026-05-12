"""Smoke test: CG must converge to the same alpha as the full LP."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasserre.polya_lp.build import (
    BuildOptions, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.solve import solve_lp
from lasserre.polya_lp.symmetry import (
    project_window_set_to_z2_rescaled, z2_dim,
)
from lasserre.polya_lp.cutting_plane import solve_with_cg


def main():
    print("CG soundness smoke test", flush=True)
    print(f"{'d':>3} {'R':>3} {'full_alpha':>12} {'cg_alpha':>12} "
          f"{'diff':>10} {'cg_iters':>8} {'cg_n_eq':>8} {'full_n_eq':>8}", flush=True)
    for d in (4, 6, 8):
        for R in (4, 6, 8):
            _, M_mats_orig = build_window_matrices(d)
            M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats_orig, d)
            d_eff = z2_dim(d)

            # Full LP
            opts_full = BuildOptions(R=R, verbose=False)
            full = build_handelman_lp(d_eff, M_mats_eff, opts_full)
            sol_full = solve_lp(full)

            # CG
            cg = solve_with_cg(d_eff, M_mats_eff, R, max_iter=20, verbose=False)

            full_a = sol_full.alpha if sol_full.alpha else float('nan')
            cg_a = cg.final_alpha if cg.final_alpha else float('nan')
            diff = abs(full_a - cg_a)
            cg_n_eq = cg.iterations[-1].n_constraints if cg.iterations else 0
            print(f"{d:>3} {R:>3} {full_a:>12.6f} {cg_a:>12.6f} "
                  f"{diff:>10.2e} {len(cg.iterations):>8} "
                  f"{cg_n_eq:>8} {full.A_eq.shape[0]:>8}", flush=True)


if __name__ == "__main__":
    main()
