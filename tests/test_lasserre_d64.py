"""Tests for the sparse-clique Farkas pipeline at d=64 (and d=128).

NOTE: the residual_l1_fmpq and safety_margin_fmpq fields can have
numerator/denominator with millions of digits when the rational rounding
exposes the float-to-fmpq blowup. Python 3.11+ defaults to a 4300-digit
limit on ``int(s)`` for huge strings; we raise it for the test session.

Two layers of test:

1. **Local sanity** at d=8 — runs in a few seconds, verifies the sparse
   clique decomposition is correct and the Farkas pipeline produces a
   rigorously certified safety_margin > 0 in exact rationals.

2. **Pod-grade gate** at d=64 — checks the persisted certificate file
   ``lasserre/certs/d64_cert.json`` exists and meets the acceptance
   criterion (lb_rig >= 1.281 in exact rationals, with positive
   safety_margin verifiable via mpmath at dps=80).

Pod-grade tests skip when no cert file is present so the suite remains
green on a developer laptop.
"""
from __future__ import annotations

import json
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

# Raise the int-from-string size limit (defaults to 4300 chars in 3.11+)
try:
    sys.set_int_max_str_digits(0)  # 0 = unlimited
except AttributeError:
    pass


def _fmpq_str_is_positive(s: str) -> bool:
    """Sign of a flint-fmpq string.

    flint stringifies an fmpq as either ``p`` (when denominator is 1) or
    ``p/q``; sign appears in the numerator. The denominator is always
    positive in flint's canonical form.
    """
    s = s.strip()
    if s.startswith('-'):
        return False
    if s == '0' or s == '0/1':
        return False
    return True


def _fmpq_str_split(s: str) -> tuple[str, str]:
    if '/' in s:
        p, q = s.split('/', 1)
    else:
        p, q = s, '1'
    return p, q

from lasserre.cliques import _build_banded_cliques
from lasserre.core import build_window_matrices


# ---------------------------------------------------------------------
# Layer 1: local sanity tests (run in CI on developer machines)
# ---------------------------------------------------------------------


def test_clique_decomposition_correct_d64():
    """The bandwidth-16 banded cliques cover every (i, j) pair with
    |i-j| <= 16 — so every window with active-bin spread <= 16 is
    coverable by at least one clique.

    Clique structure: I_c = {c, c+1, ..., c+16}, c = 0..d-17.
    """
    d = 64
    b = 16
    cliques = _build_banded_cliques(d, b)
    assert len(cliques) == d - b, f"expected {d - b} cliques, got {len(cliques)}"
    for c, cl in enumerate(cliques):
        assert cl == list(range(c, c + b + 1)), \
            f"clique {c} = {cl} != expected {list(range(c, c + b + 1))}"

    # Every i in 0..d-1 must appear in at least one clique
    seen = set()
    for cl in cliques:
        seen.update(cl)
    assert seen == set(range(d)), \
        f"cliques don't cover all bins: missing {set(range(d)) - seen}"

    # Every (i, j) with |i-j| <= b must be in some clique together
    for i in range(d):
        for j in range(d):
            if abs(i - j) <= b:
                covered = False
                for cl in cliques:
                    if i in cl and j in cl:
                        covered = True
                        break
                assert covered, f"(i={i}, j={j}, |i-j|={abs(i-j)}) not in any clique"


def test_window_clique_coverage_classification():
    """Every window matrix M_W with active bin-spread <= bandwidth should
    be classified as clique-coverable; wider ones fall back to full.

    This validates that ``_build_window_blocks`` does not silently drop
    a clique-coverable window into the (more expensive) full path.
    """
    d = 64
    b = 16
    cliques = _build_banded_cliques(d, b)
    windows, M_mats = build_window_matrices(d)

    coverable_count = 0
    full_count = 0
    for w_idx, Mw in enumerate(M_mats):
        nz_i, nz_j = np.nonzero(Mw)
        if len(nz_i) == 0:
            continue
        active = sorted(set(nz_i.tolist()) | set(nz_j.tolist()))
        spread = active[-1] - active[0]

        is_coverable = any(
            all(bb in set(cl) for bb in active) for cl in cliques)
        if spread <= b:
            assert is_coverable, \
                f"window {w_idx} has spread {spread} <= {b} but no clique covers it"
            coverable_count += 1
        else:
            assert not is_coverable, \
                f"window {w_idx} has spread {spread} > {b} but a clique covers it??"
            full_count += 1

    assert coverable_count > 0, "no clique-coverable windows found at d=64"
    assert full_count > 0, "no wide windows found at d=64 — sanity check broken"


@pytest.mark.parametrize('d, t_lo, t_hi', [
    (8, 1.0, 1.5),
])
def test_solver_returns_feasible_or_infeasible(d, t_lo, t_hi):
    """Sparse SDP at low t should be INFEASIBLE; at high t FEASIBLE."""
    pytest.importorskip("mosek")
    from lasserre.d64_solver import solve_sparse_farkas_at_t

    # Below val^{(k,b)}(d): infeasible
    r_lo = solve_sparse_farkas_at_t(
        d=d, order=2, bandwidth=d - 1, t_test=t_lo, verbose=False)
    assert r_lo.status == 'INFEASIBLE', \
        f"expected INFEASIBLE at t={t_lo}, got {r_lo.status}"

    # Above val(d): feasible
    r_hi = solve_sparse_farkas_at_t(
        d=d, order=2, bandwidth=d - 1, t_test=t_hi, verbose=False)
    assert r_hi.status == 'FEASIBLE', \
        f"expected FEASIBLE at t={t_hi}, got {r_hi.status}"


def test_farkas_safety_margin_positive_d8():
    """End-to-end smoke test: at d=8 / order 2 / b=7 / t=1.05, the
    Farkas pipeline must produce a positive safety margin in exact
    rationals.
    """
    pytest.importorskip("mosek")
    pytest.importorskip("flint")
    from lasserre.d64_farkas_cert import certify_sparse_farkas
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cert_path = os.path.join(tmp, 'd8_test.json')
        r = certify_sparse_farkas(
            d=8, order=2, bandwidth=7, t_test=1.05,
            max_denom_S=10**10, max_denom_mu=10**10, eig_margin=1e-9,
            cert_path=cert_path, verbose=False,
        )
        assert r.status == 'CERTIFIED', \
            f"expected CERTIFIED, got {r.status} ({r.notes})"
        # Safety margin must be strictly positive in exact rationals
        assert _fmpq_str_is_positive(r.safety_margin_fmpq), \
            "safety_margin not strictly positive in exact rationals"

        # Verify cert file has all expected fields
        with open(cert_path) as f:
            cert = json.load(f)
        assert cert['status'] == 'CERTIFIED'
        assert cert['d'] == 8
        assert cert['order'] == 2
        assert cert['bandwidth'] == 7
        assert Fraction(cert['lb_rig']) >= Fraction(105, 100)


def test_farkas_residual_consistency_dps80():
    """The mpmath dps-80 residual l1 string in the cert file must be
    consistent with the fmpq exact residual_l1.

    This catches arithmetic bugs in the residual accumulator.
    """
    pytest.importorskip("mosek")
    pytest.importorskip("flint")
    pytest.importorskip("mpmath")
    from lasserre.d64_farkas_cert import certify_sparse_farkas
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cert_path = os.path.join(tmp, 'd8_test_dps.json')
        certify_sparse_farkas(
            d=8, order=2, bandwidth=7, t_test=1.05,
            max_denom_S=10**10, max_denom_mu=10**10, eig_margin=1e-9,
            cert_path=cert_path, verbose=False,
        )
        with open(cert_path) as f:
            cert = json.load(f)

    # Cross-check: parse residual_l1_fmpq and compare to mpmath string
    import mpmath
    mpmath.mp.dps = 200  # extra cushion for the comparison
    p, q = _fmpq_str_split(cert['residual_l1_fmpq'])
    r_l1_mp = mpmath.mpf(int(p)) / mpmath.mpf(int(q))
    r_l1_str_mp = mpmath.mpf(cert['residual_l1_mpmath_dps80'])
    if abs(r_l1_mp) < mpmath.mpf('1e-100'):
        return  # below tolerance — trivially consistent
    rel_err = abs(r_l1_mp - r_l1_str_mp) / abs(r_l1_mp)
    assert float(rel_err) < 1e-70, \
        f"dps-80 residual mismatch: rel_err={float(rel_err)}"


# ---------------------------------------------------------------------
# Layer 2: pod-grade tests (skip if no cert file present)
# ---------------------------------------------------------------------

CERT_PATH_D64 = Path(__file__).resolve().parent.parent / 'lasserre' / 'certs' / 'd64_cert.json'
CERT_PATH_D128 = Path(__file__).resolve().parent.parent / 'lasserre' / 'certs' / 'd128_cert.json'


@pytest.mark.skipif(not CERT_PATH_D64.exists(),
                     reason="d=64 cert not yet produced (requires pod run)")
def test_val_d64_geq_1_281():
    """If a d=64 certificate has been produced, it must certify
    val^{(k,b)}(64) >= 1.281 — the project acceptance criterion.
    """
    with open(CERT_PATH_D64) as f:
        cert = json.load(f)
    assert cert['status'] == 'CERTIFIED', \
        f"d=64 cert status is {cert['status']}, not CERTIFIED"
    lb = Fraction(cert['lb_rig'])
    assert lb >= Fraction(1281, 1000), \
        f"d=64 lb_rig {cert['lb_rig']} < 1.281 (insufficient for unconditional improvement)"


@pytest.mark.skipif(not CERT_PATH_D64.exists(),
                     reason="d=64 cert not yet produced")
def test_d64_cert_safety_margin_positive_exact():
    """Verify the d=64 cert's safety margin is positive in exact
    rationals — this is the load-bearing soundness check.
    """
    with open(CERT_PATH_D64) as f:
        cert = json.load(f)
    assert _fmpq_str_is_positive(cert['safety_margin_fmpq']), \
        "d=64 cert safety_margin not strictly positive in exact rationals"


@pytest.mark.skipif(not CERT_PATH_D64.exists(),
                     reason="d=64 cert not yet produced")
def test_d64_cert_dps80_residual_below_safe_threshold():
    """At mpmath dps=80, the residual l1 must be less than mu_A[0] / (2k+1)
    — i.e., the safe-bound inequality holds at dps=80 with margin.
    """
    import mpmath
    mpmath.mp.dps = 200
    with open(CERT_PATH_D64) as f:
        cert = json.load(f)
    p_mu, q_mu = _fmpq_str_split(cert['mu0'])
    mu0 = mpmath.mpf(int(p_mu)) / mpmath.mpf(int(q_mu))
    r_l1 = mpmath.mpf(cert['residual_l1_mpmath_dps80'])
    bound = mpmath.mpf(cert['moment_l1_bound'])
    assert r_l1 * bound < mu0, \
        f"d=64 cert violates safe bound at dps=80: r_l1*{int(bound)}={r_l1*bound} >= mu0={mu0}"


@pytest.mark.skipif(not CERT_PATH_D128.exists(),
                     reason="d=128 cert not yet produced (corroborative)")
def test_val_d128_geq_1_281():
    """d=128 corroboration: if produced, must certify val^{(k,b)}(128) >= 1.281."""
    with open(CERT_PATH_D128) as f:
        cert = json.load(f)
    assert cert['status'] == 'CERTIFIED'
    lb = Fraction(cert['lb_rig'])
    assert lb >= Fraction(1281, 1000)
