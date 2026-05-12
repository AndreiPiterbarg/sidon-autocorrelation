#!/usr/bin/env python
"""Regression tests for the clique-decomposed SOS-dual builder.

Equivalence guarantees:
  1. bandwidth = d-1 recovers the monolithic bound exactly (modulo IPM tol).
  2. Bound is non-increasing as bandwidth shrinks (val_L^clique(b) ↓ in b).
  3. At bandwidth ≥ max window span, no window falls back to the full
     localising basis.

Integration:
  4. Full stack at d=6: monolithic(default) vs cliques+adaptive+graphpar
     agree to within IPM tolerance.

Each test runs real MOSEK bisections; no mocking.  The ``slow`` marker
lets the d ≥ 8 tests be opt-in (``pytest -m slow``).
"""
from __future__ import annotations

import io
import os
import sys
import contextlib

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

# Unit tests need only numpy + cliques.  MOSEK tests are skipped if
# MOSEK is unavailable (unlikely in this repo, but a safety net).
import numpy as np

from lasserre.cliques import _build_banded_cliques, _build_clique_basis

try:
    import mosek  # noqa: F401
    _HAS_MOSEK = True
except Exception:
    _HAS_MOSEK = False

# Tolerance for bisection bracket comparisons across runs.  At n_bisect=15
# the bracket width is ~0.5 / 2^15 ~ 1.5e-5, far below 5e-4 — so this
# accommodates IPM tolerance noise near the critical λ ≈ 0.5 boundary.
EQ_TOL = 5e-4


# =====================================================================
# Pure unit tests — no MOSEK required
# =====================================================================

def test_single_clique_bandwidth_matches_full_basis():
    """bandwidth = d-1 ⇒ single clique containing all bins."""
    d = 8
    cliques = _build_banded_cliques(d, d - 1)
    assert cliques == [[0, 1, 2, 3, 4, 5, 6, 7]]

    B = _build_clique_basis(cliques[0], order=3, d=d)
    # Row count = C(d + order, order) for full-support basis.
    from math import comb
    assert B.shape == (comb(d + 3, 3), d)
    # Every row has |row| ≤ order.
    assert (B.sum(axis=1) <= 3).all()
    # Sum of degrees equals 0 only for the zero monomial.
    assert (B.sum(axis=1) == 0).sum() == 1


def test_clique_basis_subset_of_full_basis():
    """A smaller-bandwidth clique's basis is a strict subset of the full basis."""
    d = 8
    order = 3
    cliques = _build_banded_cliques(d, bandwidth=3)  # cliques of size 4
    B0 = _build_clique_basis(cliques[0], order, d)
    # Rows in B0 must have support in {0,1,2,3} (the first clique).
    support_bins = set(cliques[0])
    for row in B0:
        nz = set(np.nonzero(row)[0].tolist())
        assert nz.issubset(support_bins)
    # B0 is strictly smaller than the full basis.
    from math import comb
    assert B0.shape[0] < comb(d + order, order)


# =====================================================================
# Helpers for driver comparison tests
# =====================================================================

def _run_monolithic(d, order, n_bisect, t_hi=None, tol=1e-6,
                    solve_form='dual'):
    from lasserre_mosek_dual import solve_mosek_dual
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = solve_mosek_dual(
            d, order,
            add_upper_loc=False,
            z2_full=False,
            n_bisect=n_bisect,
            t_hi=t_hi,
            primary_tol=tol,
            solve_form=solve_form,
            mosek_log=False,
            reuse_task=False,  # match clique path's rebuild-per-probe
            verbose=False)
    return r, buf.getvalue()


def _run_cliques(d, order, bandwidth, n_bisect, t_hi=None, tol=1e-6,
                 solve_form='dual'):
    from lasserre_mosek_dual_cliques import solve_mosek_dual_cliques
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = solve_mosek_dual_cliques(
            d, order,
            bandwidth=bandwidth,
            add_upper_loc=False,
            n_bisect=n_bisect,
            t_hi=t_hi,
            primary_tol=tol,
            solve_form=solve_form,
            mosek_log=False,
            verbose=False)
    return r, buf.getvalue()


def _bracket_mid(r):
    """Extract the bracket midpoint from a driver result dict."""
    if 'bracket_mid' in r and r['bracket_mid'] is not None:
        return float(r['bracket_mid'])
    lo = r.get('lo')
    hi = r.get('hi')
    if lo is not None and hi is not None:
        return 0.5 * (lo + hi)
    # Fall back to final history entry mid.
    if r.get('history'):
        hist = r['history']
        return float(hist[-1]['t'])
    return float(r['lb'])


# =====================================================================
# Parity tests — single clique reproduces the monolithic bound
# =====================================================================

@pytest.mark.skipif(not _HAS_MOSEK, reason="MOSEK not available")
def test_single_clique_reproduces_monolithic_d4():
    """At d=4, bandwidth=3: single clique = full basis ⇒ identical bound."""
    d, order, n_bisect = 4, 3, 10
    t_hi = 1.15
    r_mono, _ = _run_monolithic(d, order, n_bisect, t_hi=t_hi)
    r_cliq, _ = _run_cliques(d, order, bandwidth=d - 1,
                              n_bisect=n_bisect, t_hi=t_hi)

    m_mid = _bracket_mid(r_mono)
    c_mid = _bracket_mid(r_cliq)
    assert abs(m_mid - c_mid) < EQ_TOL, (
        f"d=4 single-clique bound diverges: mono={m_mid:.6f} "
        f"cliq={c_mid:.6f}")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_MOSEK, reason="MOSEK not available")
def test_single_clique_reproduces_monolithic_d6():
    """At d=6, bandwidth=5: single clique = full basis ⇒ identical bound."""
    d, order, n_bisect = 6, 3, 10
    t_hi = 1.22
    r_mono, _ = _run_monolithic(d, order, n_bisect, t_hi=t_hi)
    r_cliq, _ = _run_cliques(d, order, bandwidth=d - 1,
                              n_bisect=n_bisect, t_hi=t_hi)

    m_mid = _bracket_mid(r_mono)
    c_mid = _bracket_mid(r_cliq)
    assert abs(m_mid - c_mid) < EQ_TOL, (
        f"d=6 single-clique bound diverges: mono={m_mid:.6f} "
        f"cliq={c_mid:.6f}")


# =====================================================================
# Monotonicity in bandwidth
# =====================================================================

@pytest.mark.slow
@pytest.mark.skipif(not _HAS_MOSEK, reason="MOSEK not available")
def test_clique_lb_monotone_in_bandwidth_d6():
    """val_L^clique(b) must be non-increasing as b shrinks."""
    d, order, n_bisect = 6, 3, 8
    t_hi = 1.22
    bounds = {}
    for b in (5, 4, 3):
        r, _ = _run_cliques(d, order, bandwidth=b,
                             n_bisect=n_bisect, t_hi=t_hi)
        bounds[b] = _bracket_mid(r)

    assert bounds[5] + EQ_TOL >= bounds[4], (
        f"monotonicity violated: b=5 -> {bounds[5]:.6f}, "
        f"b=4 -> {bounds[4]:.6f}")
    assert bounds[4] + EQ_TOL >= bounds[3], (
        f"monotonicity violated: b=4 -> {bounds[4]:.6f}, "
        f"b=3 -> {bounds[3]:.6f}")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_MOSEK, reason="MOSEK not available")
def test_clique_lb_monotone_in_bandwidth_d8():
    d, order, n_bisect = 8, 3, 8
    t_hi = 1.26
    bounds = {}
    for b in (7, 6, 5, 4):
        r, _ = _run_cliques(d, order, bandwidth=b,
                             n_bisect=n_bisect, t_hi=t_hi)
        bounds[b] = _bracket_mid(r)

    # Each step down in b cannot increase the bound (up to IPM tol).
    for hi_b, lo_b in [(7, 6), (6, 5), (5, 4)]:
        assert bounds[hi_b] + EQ_TOL >= bounds[lo_b], (
            f"b={hi_b} -> {bounds[hi_b]:.6f} vs "
            f"b={lo_b} -> {bounds[lo_b]:.6f}")


# =====================================================================
# Window coverage — no fallback at large bandwidth
# =====================================================================

@pytest.mark.skipif(not _HAS_MOSEK, reason="MOSEK not available")
def test_window_coverage_at_large_bandwidth():
    """At bandwidth = d-1, the single clique covers every window."""
    from lasserre_scalable import _precompute
    from lasserre.dual_sdp_cliques import build_dual_task_cliques
    d, order = 6, 3
    P = _precompute(d, order, verbose=False, lazy_ab_eiej=False)

    env = mosek.Env()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        task, info = build_dual_task_cliques(
            P, t_val=1.20, env=env,
            bandwidth=d - 1,
            verbose=True, cache_for_reuse=False)
    assert info['n_fallback_windows'] == 0, (
        f"Expected no fallbacks at bandwidth=d-1, got "
        f"{info['n_fallback_windows']}")
    assert 'falling back' not in buf.getvalue(), buf.getvalue()
    try:
        task.__del__()
    except Exception:
        pass


# =====================================================================
# Integration test — full stack (cliques + adaptive tol + graphpar)
# =====================================================================

@pytest.mark.slow
@pytest.mark.skipif(not _HAS_MOSEK, reason="MOSEK not available")
def test_full_stack_d6_parity():
    """Monolithic (default tol, default ordering) vs cliques + adaptive +
    graphpar — the two end-to-end bounds must agree to within IPM tol.

    The graphpar reordering is already baked into _apply_task_params (so
    both drivers use it); the adaptive tol is in the clique driver's
    _probe loop.  This test therefore exercises:
        cliques(bandwidth=d-1) + adaptive tol + graphpar
    against
        monolithic + fixed tol=1e-6 + graphpar
    """
    d, order, n_bisect = 6, 3, 12
    t_hi = 1.22
    r_mono, _ = _run_monolithic(d, order, n_bisect, t_hi=t_hi, tol=1e-6)
    r_cliq, _ = _run_cliques(d, order, bandwidth=d - 1,
                              n_bisect=n_bisect, t_hi=t_hi, tol=1e-6)

    m_mid = _bracket_mid(r_mono)
    c_mid = _bracket_mid(r_cliq)
    assert abs(m_mid - c_mid) < EQ_TOL, (
        f"full-stack parity failed: mono={m_mid:.6f} "
        f"cliq={c_mid:.6f}")


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
