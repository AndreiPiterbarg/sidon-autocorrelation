"""Run d=8 R=6 (no eliminate) repeatedly to see if vectorization caused a regression."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasserre.polya_lp.build import build_handelman_lp, BuildOptions, build_window_matrices
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim
from lasserre.polya_lp.solve import solve_lp


_, M_mats = build_window_matrices(8)
M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats, 8)
d_eff = z2_dim(8)

# 1) check matrix structure differs nothing between two builds
opts = BuildOptions(R=6, use_z2=True, eliminate_c_slacks=False, verbose=True)
b1 = build_handelman_lp(d_eff, M_mats_eff, opts)
print(f"build1: nnz={b1.n_nonzero_A}, n_eq={b1.A_eq.shape[0]}, n_vars={b1.n_vars}")

# 2) solve it
sol = solve_lp(b1, solver="mosek", verbose=True)
print(f"alpha={sol.alpha}  status={sol.status}  raw={sol.raw_status}")

# 3) try MOSEK with small perturbation
sol2 = solve_lp(b1, solver="mosek", verbose=False, tol=1e-8)
print(f"tol=1e-8: alpha={sol2.alpha}  status={sol2.status}")

# 4) also fall back to highspy
sol3 = solve_lp(b1, solver="highs", verbose=False)
print(f"highspy: alpha={sol3.alpha}  status={sol3.status}")
