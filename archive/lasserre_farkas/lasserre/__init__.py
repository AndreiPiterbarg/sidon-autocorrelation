"""Lasserre SDP hierarchy for lower bounds on val(d).

Package structure:
  core.py       — hash utils, monomials, windows, val_d_known
  precompute.py — _precompute, base constraints, window PSD
  cliques.py    — banded cliques, sparse PSD constraints
  solvers.py    — solve_cg, solve_enhanced, solve_highd_sparse
"""
from lasserre.core import (
    enum_monomials, build_window_matrices, collect_moments,
    val_d_known,
    _add_mi, _unit,
    _make_hash_bases, _hash_monos, _build_hash_table, _hash_lookup,
    _hash_add, MERSENNE_61,
)
from lasserre.precompute import (
    _precompute, _build_base_constraints, _add_psd_window,
    _eval_all_windows, _check_window_violations,
)
from lasserre.cliques import (
    _build_banded_cliques, _build_clique_basis,
    _add_sparse_moment_constraints, _add_sparse_localizing_constraints,
    _add_sparse_window_psd,
    _build_base_constraints_no_psd, _batch_check_violations,
)

# Solvers are imported lazily to avoid circular imports and heavy deps
# at package load time. Use: from lasserre.solvers import solve_highd_sparse
