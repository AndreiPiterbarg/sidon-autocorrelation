"""Debug why every Tier-4 run falls back."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasserre.polya_lp.build import (
    BuildOptions, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim
from lasserre.polya_lp.tier4.coarse_solve import coarse_solve
from lasserre.polya_lp.tier4.active_set import extract_active_set, summarize_active_set
from lasserre.polya_lp.tier4.polish import polish_via_mosek, verify_active_set


def main():
    d, R = 8, 4
    _, M_mats = build_window_matrices(d)
    M_mats, _ = project_window_set_to_z2_rescaled(M_mats, d)
    d_eff = z2_dim(d)
    build = build_handelman_lp(d_eff, M_mats, BuildOptions(R=R, use_z2=True))
    print(f"FULL LP: n_eq={build.A_eq.shape[0]} n_vars={build.n_vars} "
          f"n_W={len(M_mats)}")

    coarse = coarse_solve(build, tol=1e-5, backend="highs_ipm")
    print(f"COARSE: alpha={coarse.alpha:.10f} converged={coarse.converged} "
          f"kkt={coarse.kkt:.2e}  wall={coarse.wall_s*1000:.1f}ms")

    act = extract_active_set(build, coarse)
    print(f"ACTIVE: {summarize_active_set(act)}")
    print(f"  active_lambda_idx[:10] = {act.active_lambda_idx[:10]}")

    polish = polish_via_mosek(M_mats, d_eff, R, act, tol=1e-9, verbose=False)
    print(f"POLISH: alpha={polish.alpha} converged={polish.converged} "
          f"status={polish.sol_polish.status if polish.sol_polish else None} "
          f"build={polish.wall_build_s*1000:.1f}ms solve={polish.wall_solve_s*1000:.1f}ms")
    if polish.sol_polish:
        sol = polish.sol_polish
        print(f"  solver={sol.solver}  raw_status={sol.raw_status}")
        if sol.y is not None:
            print(f"  dual y shape={sol.y.shape}")
        else:
            print(f"  dual y is None")

    verify = verify_active_set(M_mats, polish, act, tol=1e-7)
    print(f"VERIFY: max_v={verify.max_violation:.3e} n_viol={verify.n_violators}  "
          f"dropped={verify.dropped_indices.size}")


if __name__ == "__main__":
    main()
