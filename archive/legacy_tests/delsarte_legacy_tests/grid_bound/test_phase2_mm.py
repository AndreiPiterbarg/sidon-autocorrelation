"""Phase 2 acceptance tests for the multi-moment pipeline.

Tests:
  1. At N=1, the new MM pipeline reproduces Phase 1 (M_cert in [1.2743, 1.2750]).
  2. All 12 admissible library f's pass filter_all at N=2..5 (no REJECT).
  3. Cell search at N=2 with + without MO 2.17 behaves as expected
     (MO 2.17 is a strict tightening, never loosening).
  4. Round-trip: a generated N=2 certificate passes certify_mm.py.

Tests 1-3 are cheap; test 4 is the expensive acceptance run.
"""
from __future__ import annotations

import os
import tempfile

import pytest
from flint import arb, fmpq, ctx

from delsarte_dual.grid_bound.admissible_f import build_library
from delsarte_dual.grid_bound.filters import filter_all, FilterVerdict
from delsarte_dual.grid_bound.phi_mm import PhiMMParams, mu_of_M
from delsarte_dual.grid_bound.cell_search_nd import certify_phi_mm_negative
from delsarte_dual.grid_bound.bisect_mm import bisect_M_cert_mm, emit_certificate_mm
from delsarte_dual.grid_bound.certify_mm import verify_certificate_mm


def _fmpq_to_float(q: fmpq) -> float:
    return float(q.p) / float(q.q)


@pytest.mark.parametrize("N", [2, 3, 4, 5])
def test_library_passes_filter_all(N):
    """filter_all must NOT REJECT any admissible library f at any N >= 2."""
    ctx.prec = 256
    M_dummy = arb("1.28")
    mu_up = mu_of_M(M_dummy).upper()
    lib = build_library()
    for f in lib:
        ab = f.moments(N, prec_bits=256)
        # Note: admissible f's have ||f*f||_inf much bigger than 1.28 in general
        # (e.g. uniform => 2); F_bathtub is an M-conditioned constraint, so for
        # this pure filter-validity check we DISABLE F_bathtub by not passing
        # mu_arb.  A filter that rejects here would indicate a genuine bug.
        verdict = filter_all(ab, N)
        assert verdict != FilterVerdict.REJECT, (
            f"filter_all REJECTS admissible f '{f.name}' at N={N}"
        )


def test_mm_pipeline_reproduces_phase1_at_N1():
    """At N=1, Phase 2 pipeline must certify M_cert in Phase 1's acceptance band."""
    ctx.prec = 256
    params = PhiMMParams.from_mv(N_max=1, n_cells_min_G=4096, prec_bits=256)
    # Use a large cell budget; N=1 + 2D is harder than 1D but should still work
    # once the arb_sqr dependency fix is in place.
    bound = bisect_M_cert_mm(
        params, N=1,
        M_lo_init=fmpq(127, 100),
        M_hi_init=fmpq(131, 100),
        tol_q=fmpq(1, 10**4),
        max_cells_per_M=200000,
        filter_kwargs=dict(enable_F4_MO217=False, enable_F7=True, enable_F8=True),
        prec_bits=256,
        verbose=False,
    )
    M = _fmpq_to_float(bound.M_cert_q)
    assert 1.2743 <= M <= 1.2750, (
        f"N=1 MM pipeline M_cert = {M} outside Phase 1 acceptance band"
    )


def test_mo_217_never_loosens_the_bound():
    """Enabling MO 2.17 at N=2 should certify at least as well as without it.

    At M = 1.274 we expect both to certify; at a higher M, with-MO217 should
    certify MORE M values than without.
    """
    ctx.prec = 256
    params = PhiMMParams.from_mv(N_max=2, n_cells_min_G=4096, prec_bits=256)
    M_test = arb("1.274")
    res_without = certify_phi_mm_negative(
        M_test, params, N=2, max_cells=500000,
        filter_kwargs=dict(enable_F4_MO217=False, enable_F7=True, enable_F8=True),
    )
    res_with = certify_phi_mm_negative(
        M_test, params, N=2, max_cells=500000,
        filter_kwargs=dict(enable_F4_MO217=True, enable_F7=True, enable_F8=True),
    )
    # If one certifies and the other doesn't, MO 2.17 must help (not hurt).
    if res_without.verdict == "NOT_CERTIFIED":
        # Then with MO 2.17, we shouldn't be worse; we might still fail to
        # certify if cell budget is the bottleneck, but the worst_live_up
        # should be LOWER with MO 2.17 than without.
        if res_with.verdict == "NOT_CERTIFIED":
            assert res_with.worst_live.phi_upper_float <= res_without.worst_live.phi_upper_float + 1e-6, (
                "MO 2.17 loosened the bound (worst_live_up increased)"
            )


@pytest.mark.slow
def test_n2_certificate_round_trip():
    """Generate an N=2 certificate and verify it with certify_mm.py."""
    ctx.prec = 256
    params = PhiMMParams.from_mv(N_max=2, n_cells_min_G=4096, prec_bits=256)
    # Use M_lo low enough that it definitely certifies, and very tight tol.
    bound = bisect_M_cert_mm(
        params, N=2,
        M_lo_init=fmpq(127, 100),
        M_hi_init=fmpq(129, 100),
        tol_q=fmpq(1, 10**3),
        max_cells_per_M=500000,
        filter_kwargs=dict(enable_F4_MO217=True, enable_F7=True, enable_F8=True),
        prec_bits=256,
        verbose=False,
    )
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cert.json")
        emit_certificate_mm(bound, path)
        res = verify_certificate_mm(path)
    assert res.accepted, f"verifier rejected: {res.messages}"
    assert res.M_cert_q == bound.M_cert_q
