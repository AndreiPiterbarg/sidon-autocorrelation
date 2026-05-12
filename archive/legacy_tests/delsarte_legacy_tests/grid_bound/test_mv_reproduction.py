"""Phase 1 acceptance: reproduce MV 1.2748 through the new rigorous pipeline.

Spec §6.2:  at N = 1 with MV parameters (delta=0.138, u=0.638, 119 coeffs)
``certify_bound`` returns M_cert in [1.2743, 1.2750].

This test is fast (~10s): it calls bisect_M_cert with a modest tolerance and
asserts the certified M_cert lies in the acceptance band; it also round-trips
the certificate through the independent verifier.
"""
from __future__ import annotations

import os
import tempfile

from flint import fmpq

from delsarte_dual.grid_bound.phi import PhiParams
from delsarte_dual.grid_bound.bisect import bisect_M_cert, emit_certificate
from delsarte_dual.grid_bound.certify import verify_certificate


def _fmpq_to_float(q: fmpq) -> float:
    return float(q.p) / float(q.q)


def test_reproduces_mv_1_2748_within_acceptance_band():
    """§6.2 acceptance band: M_cert in [1.2743, 1.2750]."""
    params = PhiParams.from_mv(n_cells_min_G=4096, prec_bits=256)
    bound = bisect_M_cert(
        params,
        M_lo_init=fmpq(127, 100),
        M_hi_init=fmpq(1276, 1000),
        tol_q=fmpq(1, 10**4),
        max_cells_per_M=100000,
        initial_splits=32,
        prec_bits=256,
        verbose=False,
    )
    M_cert = _fmpq_to_float(bound.M_cert_q)
    assert 1.2743 <= M_cert <= 1.2750, (
        f"M_cert = {M_cert} outside acceptance band [1.2743, 1.2750]"
    )


def test_certificate_round_trip_passes_independent_verifier():
    """The emitted certificate must survive the independent verifier."""
    params = PhiParams.from_mv(n_cells_min_G=4096, prec_bits=256)
    bound = bisect_M_cert(
        params,
        M_lo_init=fmpq(127, 100),
        M_hi_init=fmpq(1276, 1000),
        tol_q=fmpq(1, 10**4),
        max_cells_per_M=100000,
        initial_splits=32,
        prec_bits=256,
        verbose=False,
    )
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cert.json")
        emit_certificate(bound, path)
        result = verify_certificate(path)
    assert result.accepted, f"Independent verifier rejected: {result.messages}"
    assert result.M_cert_q == bound.M_cert_q


def test_below_mv_threshold_always_certifies():
    """At M = 1.27 the MV inequality is strictly violated by a wide margin;
    Phase 1 must certify forbidden quickly."""
    from delsarte_dual.grid_bound.cell_search import certify_phi_negative
    from flint import arb
    params = PhiParams.from_mv(n_cells_min_G=4096, prec_bits=256)
    res = certify_phi_negative(arb("1.27"), params, max_cells=10000)
    assert res.verdict == "CERTIFIED_FORBIDDEN", (
        f"M=1.27 should be forbidden, got {res.verdict}"
    )


def test_above_mv_ceiling_cannot_certify():
    """At M = 1.30 (well above MV's theoretical ceiling ~1.276), the MV
    argument is strictly insufficient and no reasonable cell budget
    certifies.  This is a sanity check on bisection bracket soundness."""
    from delsarte_dual.grid_bound.cell_search import certify_phi_negative
    from flint import arb
    params = PhiParams.from_mv(n_cells_min_G=4096, prec_bits=256)
    res = certify_phi_negative(arb("1.30"), params, max_cells=5000)
    assert res.verdict == "NOT_CERTIFIED", (
        f"M=1.30 should NOT certify via MV's argument, got {res.verdict}"
    )
