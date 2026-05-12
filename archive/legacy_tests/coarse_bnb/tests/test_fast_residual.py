"""Bit-exact equivalence of compute_residual_fast vs the reference fmpq loops.

The fast path uses fixed-denominator rounding + numpy int64 scatter-add;
the reference path uses the existing fmpq scatter loops in
farkas_certify._adj_qW_exact_fmpq, _adj_t_fmpq and
safe_certify_flint._adjoint_block_fmpq.  Feeding BOTH paths the same
rational inputs (same numerators, same denominator), the output fmpq
lists must be identical entry-by-entry.  This test also exercises the
int64 overflow guard on realistic duals.
"""
from __future__ import annotations
import os, sys

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

try:
    import flint  # noqa: F401
    _HAS_FLINT = True
except ImportError:
    _HAS_FLINT = False

requires_flint = pytest.mark.skipif(not _HAS_FLINT, reason="python-flint not installed")


@requires_flint
@pytest.mark.parametrize("d,order", [(3, 2), (4, 2), (4, 3), (5, 2)])
def test_fast_equals_reference_on_random_duals(d, order):
    """Feed the two paths the same (mu, S_base, S_win) in fmpq form and
    compare element-wise.  We use random rational duals — the Farkas
    identity doesn't matter for the residual-computation test."""
    from lasserre.precompute import _precompute
    from certified_lasserre.build_sdp import (
        _build_moment_block, _build_loc_blocks, _build_equality_constraints,
    )
    from certified_lasserre.fast_residual import (
        build_residual_precomp, compute_residual_fast, _reference_residual,
        round_mat_fixed_denom, round_vec_fixed_denom,
    )

    rng = np.random.default_rng(seed=1234 + d * 10 + order)
    P = _precompute(d, order, verbose=False)
    n_y = P['n_y']
    n_basis = P['n_basis']
    n_loc = P['n_loc']
    n_win = P['n_win']

    base_blocks = [_build_moment_block(P)] + _build_loc_blocks(P)
    A_csr, b_vec, _ = _build_equality_constraints(P)

    # Random dual (float) and consistent rounding via round_*_fixed_denom.
    D_mu, D_S, D_W = 10**6, 10**6, 10**6   # small denom for fast test
    mu_A_float = rng.standard_normal(A_csr.shape[0])
    S_mom_float = rng.standard_normal((n_basis, n_basis))
    S_mom_float = 0.5 * (S_mom_float + S_mom_float.T)
    S_loc_float = []
    for _ in range(d):
        S = rng.standard_normal((n_loc, n_loc))
        S_loc_float.append(0.5 * (S + S.T))
    # Only activate some windows so we also test the None / skipped path.
    active_mask = rng.uniform(size=n_win) < 0.5
    S_win_float = []
    for w in range(n_win):
        if active_mask[w]:
            S = rng.standard_normal((n_loc, n_loc))
            S_win_float.append(0.5 * (S + S.T))
        else:
            S_win_float.append(None)

    # Fixed-denominator int64 numerators
    mu_A_num = round_vec_fixed_denom(mu_A_float, D_mu)
    S_mom_num = round_mat_fixed_denom(S_mom_float, D_S)
    S_loc_num = [round_mat_fixed_denom(S, D_S) for S in S_loc_float]
    S_win_num = [round_mat_fixed_denom(S, D_W) if S is not None else None
                 for S in S_win_float]

    # Reference path requires the same rationals as fmpq objects.
    def _vec_num_to_fmpq_list(num, D):
        return [flint.fmpq(int(x), D) for x in num.tolist()]
    def _mat_num_to_fmpq_mat(num, D):
        n, m = num.shape
        flat = num.tolist()
        out = flint.fmpq_mat(n, m,
            [flint.fmpq(int(flat[i][j]), D) for i in range(n) for j in range(m)])
        return out

    mu_A_fmpq = _vec_num_to_fmpq_list(mu_A_num, D_mu)
    S_mom_fmpq = _mat_num_to_fmpq_mat(S_mom_num, D_S)
    S_loc_fmpq = [_mat_num_to_fmpq_mat(S, D_S) for S in S_loc_num]
    S_win_fmpq = [_mat_num_to_fmpq_mat(S, D_W) if S is not None else None
                  for S in S_win_num]
    base_S_fmpq = [S_mom_fmpq] + S_loc_fmpq

    t_test_fmpq = flint.fmpq(1099, 1000)  # 1.099

    # Build the precomp once
    pre = build_residual_precomp(P, base_blocks)

    # Fast residual
    r_fast = compute_residual_fast(
        pre=pre,
        A_csr=A_csr,
        mu_A_num=mu_A_num, D_mu=D_mu,
        base_S_num=[S_mom_num] + S_loc_num, D_S=D_S,
        win_S_num=S_win_num, D_W=D_W,
        t_test_fmpq=t_test_fmpq,
    )

    # Reference residual
    r_ref = _reference_residual(
        P=P, A_csr=A_csr, mu_A_fmpq=mu_A_fmpq,
        base_blocks=base_blocks, base_S_fmpq=base_S_fmpq,
        S_win_fmpq=S_win_fmpq, t_test_fmpq=t_test_fmpq,
        active_windows=np.where(active_mask)[0],
    )

    # Bit-exact equality.
    assert len(r_fast) == len(r_ref) == n_y
    mismatches = []
    for alpha in range(n_y):
        if r_fast[alpha] != r_ref[alpha]:
            mismatches.append((alpha,
                               (int(r_fast[alpha].p), int(r_fast[alpha].q)),
                               (int(r_ref[alpha].p), int(r_ref[alpha].q))))
        if len(mismatches) >= 5:
            break
    assert not mismatches, f"bit-mismatch at first {len(mismatches)} alphas: {mismatches[:3]}"


@requires_flint
def test_overflow_guard_triggers():
    """Force the int64 overflow guard by cranking D_S beyond what the
    accumulation can tolerate."""
    from lasserre.precompute import _precompute
    from certified_lasserre.build_sdp import (
        _build_moment_block, _build_loc_blocks, _build_equality_constraints,
    )
    from certified_lasserre.fast_residual import (
        build_residual_precomp, compute_residual_fast,
    )

    d, order = 4, 3
    P = _precompute(d, order, verbose=False)
    n_y = P['n_y']
    n_loc = P['n_loc']
    n_basis = P['n_basis']
    n_win = P['n_win']
    base_blocks = [_build_moment_block(P)] + _build_loc_blocks(P)
    A_csr, _, _ = _build_equality_constraints(P)
    pre = build_residual_precomp(P, base_blocks)

    # Set all S_win numerators to the int64 extreme to force overflow.
    big = int(2**62 // 2)
    S_win_num = [np.full((n_loc, n_loc), big, dtype=np.int64) for _ in range(n_win)]
    mu_A_num = np.zeros(A_csr.shape[0], dtype=np.int64)
    S_mom_num = np.zeros((n_basis, n_basis), dtype=np.int64)
    S_loc_num = [np.zeros((n_loc, n_loc), dtype=np.int64) for _ in range(d)]

    with pytest.raises(RuntimeError, match="int64 overflow"):
        compute_residual_fast(
            pre=pre,
            A_csr=A_csr,
            mu_A_num=mu_A_num, D_mu=1,
            base_S_num=[S_mom_num] + S_loc_num, D_S=1,
            win_S_num=S_win_num, D_W=1,
            t_test_fmpq=flint.fmpq(1, 1),
        )
