"""Certified Lasserre SDP lower bounds on val(d).

Aggregated formulation: given lambda_W weights with sum=1, proves
    val(d) >= min_{mu in Delta_d} mu^T M_lambda mu   (M_lambda = sum lambda_W M_W)
via a Positivstellensatz certificate on the simplex.

Pipeline:
    build_sdp.build_sdp_data         -- standard SDP (c, A, b, F_blocks)
    bm_solver.solve_primal_dual      -- MOSEK / Clarabel / BM primal+dual
    dual_extract.extract_dual        -- approximate (lambda_A, {S_j})
    round_repair.round_and_repair    -- rational rounding + exact repair
    certify.certify_bound            -- end-to-end rigorous bound

Guarantee: if certify_bound succeeds, the returned rational lb_rig
satisfies lb_rig <= val(d) -- independently of any float64 computation.
"""

from certified_lasserre.build_sdp import build_sdp_data, SDPData
from certified_lasserre.bm_solver import solve_primal_dual
from certified_lasserre.dual_extract import extract_dual
from certified_lasserre.round_repair import round_and_repair
from certified_lasserre.certify import certify_bound

__all__ = [
    'build_sdp_data', 'SDPData',
    'solve_primal_dual',
    'extract_dual',
    'round_and_repair',
    'certify_bound',
]
