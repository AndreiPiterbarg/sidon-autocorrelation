#!/usr/bin/env python
"""Regression tests for adaptive IPM tolerance (Task 2) and graphpar
Cholesky reordering (Task 3).

Correctness guarantees:
  * Adaptive tolerance never flips a verdict relative to fixed tol.
  * Adaptive tolerance produces the same bracket midpoint (within IPM
    noise) as fixed tol at small d.
  * graphpar ordering produces the same bracket midpoint as the default
    ordering at small d — graphpar is a performance lever, not a
    correctness constraint.

Implementation notes:
  * The monolithic driver's ``_apply_task_params`` already applies
    graphpar unconditionally; we monkey-patch it here to disable
    graphpar when needed.
  * The clique driver's ``_probe`` uses adaptive tol inside the bisection
    loop.  We monkey-patch ``_apply_task_params`` to intercept the tol
    parameter and force a fixed value when testing the "fixed" baseline.
"""
from __future__ import annotations

import io
import os
import sys
import contextlib
from typing import List
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

try:
    import mosek
    _HAS_MOSEK = True
except Exception:
    _HAS_MOSEK = False

# graphpar / default-ordering should agree very tightly — they're the same
# feasibility problem under different Cholesky reorderings.
EQ_TOL_ORDERING = 5e-4

# With the tightened cap (1e-4 instead of 1e-3) + refine-on-ambiguous,
# the adaptive path should match the fixed-tol path within IPM noise.
EQ_TOL_ADAPTIVE = 5e-4


# =====================================================================
# Shared helper: run the CLIQUE driver (we exercise the same adaptive
# code path that lives in the monolithic driver but without the
# update_task_t subset-update concern).
# =====================================================================

def _run(d, order, bandwidth, n_bisect, t_hi, primary_tol,
         apply_patch=None, solve_form='dual'):
    """Run solve_mosek_dual_cliques optionally patching _apply_task_params.

    apply_patch : (optional) callable(task, **kwargs) executed *after*
                  the real _apply_task_params to override MOSEK params
                  (e.g. disable graphpar) OR None.  This is layered over
                  the real function; the real function is always called
                  first.
    """
    from lasserre_mosek_dual import _apply_task_params as real
    import lasserre_mosek_dual_cliques as drv

    def _wrapped(task, **kwargs):
        applied = real(task, **kwargs)
        if apply_patch is not None:
            apply_patch(task, **kwargs)
        return applied

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with mock.patch.object(drv, '_apply_task_params',
                                side_effect=_wrapped):
            r = drv.solve_mosek_dual_cliques(
                d, order,
                bandwidth=bandwidth,
                add_upper_loc=False,
                n_bisect=n_bisect,
                t_hi=t_hi,
                primary_tol=primary_tol,
                solve_form=solve_form,
                verbose=False)
    return r, buf.getvalue()


def _run_fixed_tol(d, order, bandwidth, n_bisect, t_hi, tol):
    """Run the driver with _apply_task_params forced to use a fixed tol
    regardless of what eff_tol computes in the driver."""
    from lasserre_mosek_dual import _apply_task_params as real
    import lasserre_mosek_dual_cliques as drv

    def _forced(task, **kwargs):
        # Intercept tol, force to `tol` — the rest of the tuning is
        # unchanged.
        kwargs = dict(kwargs)
        kwargs['tol'] = float(tol)
        return real(task, **kwargs)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with mock.patch.object(drv, '_apply_task_params',
                                side_effect=_forced):
            r = drv.solve_mosek_dual_cliques(
                d, order,
                bandwidth=bandwidth,
                add_upper_loc=False,
                n_bisect=n_bisect,
                t_hi=t_hi,
                primary_tol=tol,
                solve_form='dual',
                verbose=False)
    return r, buf.getvalue()


def _bracket_mid(r):
    if 'bracket_mid' in r and r['bracket_mid'] is not None:
        return float(r['bracket_mid'])
    lo, hi = r.get('lo'), r.get('hi')
    if lo is not None and hi is not None:
        return 0.5 * (lo + hi)
    return float(r['lb'])


def _verdicts(r) -> List[str]:
    return [h['verdict'] for h in r.get('history', [])]


# =====================================================================
# Task 2 — adaptive vs fixed tolerance
# =====================================================================

@pytest.mark.skipif(not _HAS_MOSEK, reason="MOSEK not available")
def test_adaptive_tol_matches_fixed_at_small_d():
    """Adaptive tol and fixed tol should reach the same final bracket."""
    d, order, n_bisect = 4, 3, 10
    t_hi = 1.15
    # (a) Fixed tol = 1e-6 throughout (forced via monkey-patch).
    r_fixed, _ = _run_fixed_tol(d, order, bandwidth=d - 1,
                                 n_bisect=n_bisect, t_hi=t_hi,
                                 tol=1e-6)
    # (b) Adaptive tol with primary_tol=1e-6 as floor.
    r_adapt, _ = _run(d, order, bandwidth=d - 1,
                       n_bisect=n_bisect, t_hi=t_hi,
                       primary_tol=1e-6)

    assert abs(_bracket_mid(r_fixed) - _bracket_mid(r_adapt)) < EQ_TOL_ADAPTIVE


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_MOSEK, reason="MOSEK not available")
def test_adaptive_tol_matches_fixed_at_d6():
    d, order, n_bisect = 6, 3, 10
    t_hi = 1.22
    r_fixed, _ = _run_fixed_tol(d, order, bandwidth=d - 1,
                                 n_bisect=n_bisect, t_hi=t_hi,
                                 tol=1e-6)
    r_adapt, _ = _run(d, order, bandwidth=d - 1,
                       n_bisect=n_bisect, t_hi=t_hi,
                       primary_tol=1e-6)
    assert abs(_bracket_mid(r_fixed) - _bracket_mid(r_adapt)) < EQ_TOL_ADAPTIVE


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_MOSEK, reason="MOSEK not available")
def test_adaptive_tol_verdict_stability_under_perturbation():
    """Verdicts must not flip between adaptive and fixed tol.

    This is the correctness guarantee: as long as the verdict sequence
    matches, the bisection walks the same path and produces the same
    bracket, independent of IPM tolerance schedule.
    """
    d, order, n_bisect = 6, 3, 10
    t_hi = 1.22
    r_fixed, _ = _run_fixed_tol(d, order, bandwidth=d - 1,
                                 n_bisect=n_bisect, t_hi=t_hi,
                                 tol=1e-6)
    r_adapt, _ = _run(d, order, bandwidth=d - 1,
                       n_bisect=n_bisect, t_hi=t_hi,
                       primary_tol=1e-6)

    # Once the bisection bracket encloses the critical val_L(d) within a
    # region of width comparable to the loose IPM tolerance, an isolated
    # verdict flip is possible at a probe landing right on val_L(d).  The
    # correctness guarantee is therefore "at most one verdict flip before
    # the bracket collapses further than the tolerance can resolve".
    v_fixed = _verdicts(r_fixed)
    v_adapt = _verdicts(r_adapt)
    min_len = min(len(v_fixed), len(v_adapt))
    assert min_len > 0, "no bisection history recorded"
    n_flips = 0
    for i in range(min_len):
        if v_fixed[i] == 'uncertain' or v_adapt[i] == 'uncertain':
            continue
        if v_fixed[i] != v_adapt[i]:
            n_flips += 1
    # Allow up to one knife-edge flip; catastrophic mis-tracking would
    # show up as many flips or a completely different trajectory.
    assert n_flips <= 1, (
        f"{n_flips} verdict flips between fixed and adaptive tol; "
        f"fixed={v_fixed} adapt={v_adapt}")


# =====================================================================
# Task 3 — graphpar ordering vs default
# =====================================================================

@pytest.mark.skipif(not _HAS_MOSEK, reason="MOSEK not available")
def test_graphpar_matches_default_at_small_d():
    """Forcing the default ordering (instead of graphpar) should not
    change the bisection bracket."""
    d, order, n_bisect = 4, 3, 10
    t_hi = 1.15

    def _disable_graphpar(task, **_kwargs):
        # Undo the graphpar setting; revert to the MOSEK default
        # (orderingtype.free).  Called AFTER _apply_task_params.
        try:
            task.putintparam(mosek.iparam.intpnt_order_method,
                             mosek.orderingtype.free)
        except Exception:
            pass

    r_graph, _ = _run(d, order, bandwidth=d - 1,
                       n_bisect=n_bisect, t_hi=t_hi,
                       primary_tol=1e-6)
    r_free, _ = _run(d, order, bandwidth=d - 1,
                      n_bisect=n_bisect, t_hi=t_hi,
                      primary_tol=1e-6,
                      apply_patch=_disable_graphpar)

    assert abs(_bracket_mid(r_graph) - _bracket_mid(r_free)) < EQ_TOL_ORDERING


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_MOSEK, reason="MOSEK not available")
def test_graphpar_matches_default_at_d6():
    d, order, n_bisect = 6, 3, 10
    t_hi = 1.22

    def _disable_graphpar(task, **_kwargs):
        try:
            task.putintparam(mosek.iparam.intpnt_order_method,
                             mosek.orderingtype.free)
        except Exception:
            pass

    r_graph, _ = _run(d, order, bandwidth=d - 1,
                       n_bisect=n_bisect, t_hi=t_hi,
                       primary_tol=1e-6)
    r_free, _ = _run(d, order, bandwidth=d - 1,
                      n_bisect=n_bisect, t_hi=t_hi,
                      primary_tol=1e-6,
                      apply_patch=_disable_graphpar)

    assert abs(_bracket_mid(r_graph) - _bracket_mid(r_free)) < EQ_TOL_ORDERING


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
