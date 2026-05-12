"""POC test: solve our Pólya LP on RTX 3080 GPU via custom PDLP.

Compares:
 (a) Baseline mu-only LP (HiGHS / MOSEK ground truth)
 (b) Same LP solved on GPU via PDLP (Chambolle-Pock with restarts)

Tests at d=8 (small) up to d=64 (target). Reports KKT residual, obj, time.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import torch

from lasserre.polya_lp.runner import VAL_D_KNOWN
from lasserre.polya_lp.build import (
    BuildOptions, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.solve import solve_lp
from lasserre.polya_lp.symmetry import (
    project_window_set_to_z2_rescaled, z2_dim,
)
from lasserre.polya_lp.pdlp import build_gpu_lp, pdlp_solve, unscale_solution


def main():
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    use_z2 = "--no-z2" not in sys.argv

    print(f"Pólya LP via GPU PDLP: d={d}, R={R}, Z/2={use_z2}", flush=True)

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)",
              flush=True)
    else:
        print("  GPU NOT available — running on CPU", flush=True)

    # Build LP (with Z/2 reduction by default)
    _, M_mats_orig = build_window_matrices(d)
    if use_z2:
        M_mats, _ = project_window_set_to_z2_rescaled(M_mats_orig, d)
        d_eff = z2_dim(d)
    else:
        M_mats = M_mats_orig
        d_eff = d
    print(f"  d_eff={d_eff}, {len(M_mats)} window matrices", flush=True)

    print("\n--- BUILDING LP ---", flush=True)
    t0 = time.time()
    build = build_handelman_lp(d_eff, M_mats, BuildOptions(R=R, use_z2=use_z2))
    print(f"  n_eq={build.A_eq.shape[0]}, n_vars={build.n_vars}, "
          f"nnz={build.A_eq.nnz}, build={time.time()-t0:.2f}s", flush=True)

    # Ground truth via MOSEK / HiGHS
    print("\n--- GROUND TRUTH (HiGHS) ---", flush=True)
    t0 = time.time()
    sol_truth = solve_lp(build)
    truth_alpha = sol_truth.alpha
    print(f"  HiGHS alpha = {truth_alpha:.8f}, wall={time.time()-t0:.2f}s",
          flush=True)

    # GPU PDLP
    print("\n--- GPU PDLP ---", flush=True)
    t0 = time.time()
    lp, scaling = build_gpu_lp(build.A_eq, build.b_eq, build.c, build.bounds,
                               ruiz_iter=20, free_var_box=1e3)
    print(f"  Ruiz scaling D_r range [{scaling.D_r.min():.2e}, {scaling.D_r.max():.2e}]",
          flush=True)
    print(f"  Ruiz scaling D_c range [{scaling.D_c.min():.2e}, {scaling.D_c.max():.2e}]",
          flush=True)
    print(f"  GPU upload: {time.time()-t0:.2f}s", flush=True)
    print(f"  Tensors on device: {lp.device}", flush=True)
    if torch.cuda.is_available():
        print(f"  VRAM after upload: {torch.cuda.memory_allocated()/1e6:.1f} MB", flush=True)

    print("\n  Starting PDLP...", flush=True)
    result = pdlp_solve(
        lp,
        max_outer=300,
        max_inner=500,
        tol=1e-7,
        log_every=10,
    )

    # Unscale solution to original LP
    x_orig, y_orig = unscale_solution(result.x, result.y, scaling)
    # Recompute obj in original LP variables (we minimize c^T x where c was original)
    c_orig = torch.from_numpy(build.c).to(x_orig.device).to(x_orig.dtype)
    obj_orig = (c_orig * x_orig).sum().item()

    print("\n--- RESULTS ---", flush=True)
    pdlp_alpha = -obj_orig   # we minimize -alpha
    print(f"  HiGHS alpha   = {truth_alpha:.8f}", flush=True)
    print(f"  PDLP alpha    = {pdlp_alpha:.8f}", flush=True)
    print(f"  diff          = {abs(truth_alpha - pdlp_alpha):.2e}", flush=True)
    print(f"  PDLP KKT res  = {result.kkt:.2e}", flush=True)
    print(f"  primal res    = {result.primal_res:.2e}", flush=True)
    print(f"  dual res      = {result.dual_res:.2e}", flush=True)
    print(f"  gap           = {result.gap:.2e}", flush=True)
    print(f"  outer iters   = {result.n_outer}", flush=True)
    print(f"  inner iters   = {result.n_inner_total}", flush=True)
    print(f"  PDLP wall     = {result.wall_s:.1f}s", flush=True)
    print(f"  Converged     = {result.converged}", flush=True)


if __name__ == "__main__":
    main()
