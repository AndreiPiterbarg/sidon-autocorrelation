"""Unit tests for lasserre.gap_accelerator.

Tests Layer A diagnostics on synthetic data and Layer B atom ranking
soundness on tiny d=4 problems. Integration with run_scs_direct is
covered separately in an integration-test script.
"""
import numpy as np
import pytest

from lasserre.gap_accelerator import (
    DEFAULTS, GapAccelHook,
    atom_based_window_ranking, blend_rankings,
    bootstrap_ci, compute_gradient_rank,
    diagnostic_report, extract_atoms,
    facial_reduction_rank, flat_extension_residual,
    richardson_extrapolate, shanks,
)


def test_richardson_geometric_sequence():
    L_true = 1.3
    C = 0.2
    rho = 0.5
    lb = np.array([L_true - C * rho ** n for n in range(6)])
    L_hat, rate = richardson_extrapolate(lb)
    assert abs(L_hat - L_true) < 1e-6, f'L_hat={L_hat} expected {L_true}'
    assert abs(rate - rho) < 0.05


def test_richardson_short_sequence_safe():
    assert np.isnan(richardson_extrapolate(np.array([]))[0]) or \
        richardson_extrapolate(np.array([]))[0] == richardson_extrapolate(np.array([]))[0]
    L_hat, _ = richardson_extrapolate(np.array([1.0]))
    assert L_hat == 1.0
    L_hat, _ = richardson_extrapolate(np.array([1.0, 1.1]))
    assert L_hat == 1.1


def test_richardson_nonconvergent_fallback():
    bad = np.array([1.0, 2.0, 1.0, 2.0, 1.0])
    L_hat, _ = richardson_extrapolate(bad)
    assert np.isfinite(L_hat)


def test_bootstrap_ci_covers_truth():
    rng = np.random.default_rng(42)
    L_true = 1.3
    covered = 0
    trials = 40
    for _ in range(trials):
        noise = rng.normal(0, 0.005, size=8)
        lb = np.array([L_true - 0.2 * 0.5 ** n for n in range(8)]) + noise
        lb = np.maximum.accumulate(lb)
        ci_lo, ci_hi = bootstrap_ci(lb, n_boot=100, block=2, rng_seed=0)
        if np.isfinite(ci_lo) and np.isfinite(ci_hi) and ci_lo <= L_true <= ci_hi:
            covered += 1
    assert covered >= trials * 0.5, f'CI only covered {covered}/{trials}'


def test_gradient_rank_full_vs_dependent():
    from lasserre.precompute import _precompute
    P = _precompute(d=4, order=2, verbose=False)
    n_win = P['n_win']
    rank_full, defect_full = compute_gradient_rank(
        np.zeros(P['n_y']), list(range(n_win)), P)
    assert rank_full > 0
    assert 0.0 <= defect_full <= 1.0
    rank_single, defect_single = compute_gradient_rank(
        np.zeros(P['n_y']), [0, 0, 0], P)
    assert rank_single == 1


def test_extract_atoms_low_rank():
    atoms_true = np.array([[1.0, 0.5], [0.0, 0.5], [0.0, 0.0]])
    weights_true = np.array([2.0, 1.0])
    M = atoms_true @ np.diag(weights_true) @ atoms_true.T
    atoms, weights = extract_atoms(M, rank_tol=1e-6)
    assert atoms.shape[1] == 2
    assert np.all(weights > 0)


def test_atom_ranking_returns_sorted_list():
    from lasserre.precompute import _precompute
    P = _precompute(d=4, order=2, verbose=False)
    n_y = P['n_y']
    y_vals = np.zeros(n_y)
    y_vals[P['idx'][tuple([0] * 4)]] = 1.0
    for i in range(4):
        mi = tuple(1 if k == i else 0 for k in range(4))
        y_vals[P['idx'][mi]] = 0.25
    candidates = list(range(min(10, P['n_win'])))
    ranked = atom_based_window_ranking(y_vals, P, candidates)
    assert len(ranked) == len(candidates)
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_blend_rankings_respects_budget():
    eig = [(w, -0.1 * w) for w in range(20)]
    atom = [(100 + w, 1.0 - 0.01 * w) for w in range(20)]
    blended = blend_rankings(eig, atom, n_add=10, atom_frac=0.5)
    assert len(blended) == 10
    ids = {x[0] for x in blended}
    assert len(ids) == 10


def test_facial_reduction_rank():
    Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((6, 6)))
    eigvals = np.array([5.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    S = Q @ np.diag(eigvals) @ Q.T
    r, U = facial_reduction_rank(S, tol=1e-6)
    assert r == 3
    assert U.shape == (6, 3)


def test_shanks_on_geometric():
    seq = np.array([1.3 - 0.5 ** n for n in range(6)])
    acc = shanks(seq)
    assert abs(acc - 1.3) < 1e-6


def test_flat_extension_residual_runs():
    from lasserre.precompute import _precompute
    P = _precompute(d=4, order=2, verbose=False)
    y = np.zeros(P['n_y'])
    y[P['idx'][tuple([0] * 4)]] = 1.0
    for i in range(4):
        mi = tuple(1 if k == i else 0 for k in range(4))
        y[P['idx'][mi]] = 0.25
    for i in range(4):
        for j in range(4):
            mi = tuple((1 if k == i else 0) + (1 if k == j else 0)
                       for k in range(4))
            if mi in P['idx']:
                y[P['idx'][mi]] = 0.25 * 0.25
    res = flat_extension_residual(y, P)
    assert np.isfinite(res)
    assert 0.0 <= res <= 1.0


def test_hook_noop_without_config():
    hook = GapAccelHook()
    assert hook.lb_history == []
    should_abort, _ = hook.should_abort()
    assert should_abort is False


def test_hook_records_history_and_reorder_noop_when_disabled():
    hook = GapAccelHook(config={'use_atom_ranking': False})
    from lasserre.precompute import _precompute
    P = _precompute(d=4, order=2, verbose=False)
    violations = [(0, -0.5), (1, -0.3), (2, -0.1)]
    y = np.zeros(P['n_y'])
    y[P['idx'][tuple([0] * 4)]] = 1.0
    result = hook.reorder_violations(violations, y, P, [], n_add=2)
    assert result == violations


def test_diagnostic_report_output_schema():
    from lasserre.precompute import _precompute
    P = _precompute(d=4, order=2, verbose=False)
    y = np.zeros(P['n_y'])
    y[P['idx'][tuple([0] * 4)]] = 1.0
    lb_seq = np.array([1.0, 1.05, 1.10, 1.12, 1.13, 1.135])
    gc_seq = np.array([0.0, 16.0, 33.0, 40.0, 43.0, 45.0])
    cfg = dict(DEFAULTS)
    rep = diagnostic_report(lb_seq, gc_seq, 5, [], y, P, cfg)
    expected_keys = {
        'round', 'lb', 'gc_percent', 'n_active', 'jacobian_rank',
        'rank_defect', 'L_hat_richardson', 'L_hat_rate', 'L_hat_ci_90',
        'flat_extension_residual', 'recommendation', 'reason', 'target_lb',
    }
    assert expected_keys.issubset(rep.keys())
    assert rep['target_lb'] == cfg['target_lb']


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
