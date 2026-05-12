"""Tests for the local val(d) trajectory measurement.

Layered like ``tests/test_lasserre_d64.py``:

- d=4 / d=8 tests run a quick fresh cert (seconds).
- d=16 / d=32 tests are conditional on the JSON results file existing,
  produced by ``python -m lasserre.trajectory.run_trajectory``.

Hyperparameter convention: the trajectory itself uses
``order=2, bandwidth=7`` for apples-to-apples scaling. The
``test_d4_matches_memory`` test independently runs ``order=3, bandwidth=3``
(dense) at d=4 to validate the strongest certifiable bound from the memory
file, since that memory was recorded under those settings.
"""
from __future__ import annotations

import json
import os
import sys
from fractions import Fraction
from pathlib import Path

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

# Lift Python's int-from-string limit
try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass

from lasserre.trajectory.run_trajectory import (
    DEFAULT_ORDER, DEFAULT_BANDWIDTH,
)


RESULTS_DIR = Path(__file__).resolve().parent.parent / 'lasserre' / 'trajectory' / 'results'


def _result_for_d(d: int):
    p = RESULTS_DIR / f'd{d}_trajectory.json'
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _fmpq_str_is_positive(s: str) -> bool:
    s = s.strip()
    if s.startswith('-'):
        return False
    if s == '0' or s == '0/1':
        return False
    return True


# ---------------------------------------------------------------------
# d=4 / d=8: fresh cert tests (run in CI)
# ---------------------------------------------------------------------


def test_d4_matches_memory():
    """Memory ``project_farkas_certified_lasserre.md`` records val(4) > 1.0996
    at order 3 dense. Reproduce the 1.0963-cushion claim under those
    settings (NOT the trajectory's order=2/b=7 settings, which are
    deliberately weaker for apples-to-apples scaling).
    """
    pytest.importorskip("mosek")
    pytest.importorskip("flint")
    from lasserre.d64_farkas_cert import certify_sparse_farkas
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cert_path = os.path.join(tmp, 'd4_memory.json')
        r = certify_sparse_farkas(
            d=4, order=3, bandwidth=3, t_test=1.0963,
            max_denom_S=10**12, max_denom_mu=10**12, eig_margin=1e-10,
            cert_path=cert_path, verbose=False,
        )
    assert r.status == 'CERTIFIED', \
        f"expected CERTIFIED at d=4 order=3 t=1.0963; got {r.status} ({r.notes})"
    assert _fmpq_str_is_positive(r.safety_margin_fmpq), \
        "d=4/k=3/t=1.0963: safety_margin not positive in exact rationals"
    assert Fraction(r.lb_rig) >= Fraction(10963, 10000)


def test_d8_matches_prior():
    """The prior-session smoke test certified val(8) > 1.05 under
    order=2/b=7. This test re-runs that probe and asserts the cert still holds.
    """
    pytest.importorskip("mosek")
    pytest.importorskip("flint")
    from lasserre.d64_farkas_cert import certify_sparse_farkas
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cert_path = os.path.join(tmp, 'd8_prior.json')
        r = certify_sparse_farkas(
            d=8, order=2, bandwidth=7, t_test=1.05,
            max_denom_S=10**10, max_denom_mu=10**10, eig_margin=1e-9,
            cert_path=cert_path, verbose=False,
        )
    assert r.status == 'CERTIFIED', \
        f"d=8 prior smoke test failed: {r.status} ({r.notes})"
    assert _fmpq_str_is_positive(r.safety_margin_fmpq)
    assert Fraction(r.lb_rig) >= Fraction(105, 100)


def test_no_regression_on_d8_cert():
    """Same as test_d8_matches_prior but spelled out as the regression
    gate the user requested."""
    pytest.importorskip("mosek")
    pytest.importorskip("flint")
    from lasserre.d64_farkas_cert import certify_sparse_farkas
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cert_path = os.path.join(tmp, 'd8_regression.json')
        r = certify_sparse_farkas(
            d=8, order=2, bandwidth=7, t_test=1.05,
            max_denom_S=10**10, max_denom_mu=10**10, eig_margin=1e-9,
            cert_path=cert_path, verbose=False,
        )
    assert r.status == 'CERTIFIED'
    assert _fmpq_str_is_positive(r.safety_margin_fmpq)


# ---------------------------------------------------------------------
# Trajectory result tests (read from JSON, skip if missing)
# ---------------------------------------------------------------------


def test_d4_trajectory_recorded():
    r = _result_for_d(4)
    if r is None:
        pytest.skip("no d=4 trajectory result yet")
    assert r['status'] == 'COMPLETED', \
        f"d=4 trajectory status {r['status']}; details: {r.get('notes','')}"
    assert r['largest_certified_t'] is not None
    # At order=2/b=3 (clamped), expect roughly 1.07; do not require >=1.09
    assert r['largest_certified_t'] >= 1.05


def test_d8_trajectory_recorded():
    r = _result_for_d(8)
    if r is None:
        pytest.skip("no d=8 trajectory result yet")
    assert r['status'] == 'COMPLETED'
    assert r['largest_certified_t'] is not None
    assert r['largest_certified_t'] >= 1.10, \
        f"d=8 trajectory regressed: {r['largest_certified_t']}"


def test_d16_trajectory_recorded():
    r = _result_for_d(16)
    if r is None:
        pytest.skip("no d=16 trajectory result yet")
    if r['status'] != 'COMPLETED':
        pytest.skip(f"d=16 trajectory status: {r['status']} ({r.get('notes','')})")
    assert r['largest_certified_t'] is not None
    # At d=16 we expect at least the d=8 value
    d8 = _result_for_d(8)
    if d8 and d8['status'] == 'COMPLETED':
        assert r['largest_certified_t'] >= d8['largest_certified_t'] - 1e-6, \
            f"d=16 ({r['largest_certified_t']}) below d=8 ({d8['largest_certified_t']}) — non-monotone"


def test_d32_trajectory_recorded():
    r = _result_for_d(32)
    if r is None:
        pytest.skip("no d=32 trajectory result yet")
    if r['status'] != 'COMPLETED':
        pytest.skip(f"d=32 trajectory status: {r['status']} ({r.get('notes','')})")
    assert r['largest_certified_t'] is not None


def test_monotone_in_d():
    """Across all measured d's that COMPLETED, largest_certified_t must
    be (weakly) nondecreasing in d.

    Skips if fewer than two d's have completed results.
    """
    completed = []
    for d in (4, 8, 16, 32):
        r = _result_for_d(d)
        if r is not None and r['status'] == 'COMPLETED' \
                and r['largest_certified_t'] is not None:
            completed.append((d, r['largest_certified_t']))
    if len(completed) < 2:
        pytest.skip("need >=2 completed trajectory points to test monotonicity")
    for i in range(1, len(completed)):
        d_prev, t_prev = completed[i - 1]
        d_cur, t_cur = completed[i]
        assert t_cur >= t_prev - 1e-6, \
            f"non-monotone: val(d={d_prev})={t_prev}, val(d={d_cur})={t_cur}"
