"""Lasserre SDP solver entry points.

All solvers produce a dict with at least {'lb': float, 'd': int, 'order': int}.
The lb value satisfies lb <= val(d) (soundness guarantee).

Available solvers:
  solve_highd_sparse  — clique-restricted for d=64-128 (current focus)
  solve_cg            — constraint-generation with full moment set
  solve_enhanced      — CG + sparse/DSOS/BM modes
  solve_lasserre_fusion — monolithic full-model solver (small d only)
"""
import sys
import os

# Add tests/ to path for backward-compatible imports of solvers
# that haven't been fully migrated yet
_tests_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tests')
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)


# =====================================================================
# High-d sparse solver (current focus, fully migrated)
# =====================================================================

from lasserre.core import val_d_known  # noqa: E402

# Re-export the highd solver from its current location
# (tests/lasserre_highd.py, which now imports from the lasserre package)
from lasserre_highd import solve_highd_sparse  # noqa: E402


# =====================================================================
# CG solver (from lasserre_scalable.py)
# =====================================================================

def solve_cg(d, c_target, order=3, n_bisect=15,
             add_upper_loc=True, cg_rounds=5, cg_add_per_round=10,
             conv_tol=1e-7, verbose=True):
    """Constraint generation Lasserre solver.

    Uses full moment set (all C(d+2k,2k) moments). Exact Lasserre bound.
    Best for d <= 32.
    """
    from lasserre_scalable import solve_cg as _solve_cg
    return _solve_cg(d, c_target, order, n_bisect, add_upper_loc,
                     cg_rounds, cg_add_per_round, conv_tol, verbose)


def solve_lasserre_scalable(d, c_target, order=3, n_bisect=15,
                             mode='cg', add_upper_loc=True,
                             cg_rounds=5, cg_add_per_round=10,
                             verbose=True):
    """Unified entry point for scalable Lasserre hierarchy."""
    from lasserre_scalable import solve_lasserre_scalable as _solve
    return _solve(d, c_target, order, n_bisect, mode, add_upper_loc,
                  cg_rounds, cg_add_per_round, verbose)


# =====================================================================
# Enhanced solver (sparse/DSOS/BM modes)
# =====================================================================

def solve_enhanced(d, c_target, order=3, psd_mode='full',
                   search_mode='bisect', add_upper_loc=True,
                   max_cg_rounds=20, max_add_per_round=20,
                   n_bisect=15, sparse_bandwidth=8,
                   bm_rank=50, verbose=True):
    """Enhanced CG Lasserre with sparse/DSOS/BM modes.

    psd_mode='sparse' uses clique decomposition (Waki et al. 2006).
    Best for d=16-64 where full moment set fits but full PSD doesn't.
    """
    from lasserre_enhanced import solve_enhanced as _solve
    return _solve(d, c_target, order, psd_mode, search_mode, add_upper_loc,
                  max_cg_rounds, max_add_per_round, n_bisect,
                  sparse_bandwidth, bm_rank, verbose)


# =====================================================================
# Original monolithic solver (small d only)
# =====================================================================

def solve_lasserre_fusion(d, c_target, order=2, n_bisect=10, verbose=True):
    """Full Lasserre SDP via MOSEK Fusion. All windows, no CG.

    Builds all PSD window constraints upfront. Only feasible for small d
    (d <= 16 at L2, d <= 10 at L3).
    """
    from lasserre_fusion import solve_lasserre_fusion as _solve
    return _solve(d, c_target, order, n_bisect, verbose)
