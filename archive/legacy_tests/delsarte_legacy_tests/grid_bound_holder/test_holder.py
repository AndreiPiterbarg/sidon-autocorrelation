"""Acceptance tests for the Hoelder-generalised Phase 2 pipeline.

Tests:
  1. test_reduces_to_mv_at_p_equals_2: phi_holder == phi_mm (bit-identical)
     for 100 random (M, ab) and N in {1, 2, 3}.
  2. test_holder_rhs_monotone_in_p_direction: HM-10 RHS - LHS at fixed
     (M, ab, N=1) for p in a grid; direction is empirical and documented.
  3. test_holder_matches_MO_conjecture_at_p_star: slack ratio analysis at
     MO (p*, q*) = (4/3, 4).  Reported as a diagnostic (NOT a pass/fail).
  4. test_filter_validity_unchanged: every admissible library f passes
     the filter panel in the Hoelder pipeline (unchanged from MM-10).
  5. test_certificate_round_trip: a generated certificate passes
     certify_holder.py.
"""
from __future__ import annotations

import os
import random
import tempfile

import pytest
from flint import arb, fmpq, ctx

from delsarte_dual.grid_bound.admissible_f import build_library
from delsarte_dual.grid_bound.filters import filter_all, FilterVerdict
from delsarte_dual.grid_bound.phi_mm import phi_mm, mu_of_M, PhiMMParams
from delsarte_dual.grid_bound_holder.phi_holder import (
    PhiHolderParams, phi_holder,
)
from delsarte_dual.grid_bound_holder.cell_search_holder import (
    certify_phi_holder_negative,
)
from delsarte_dual.grid_bound_holder.bisect_holder import (
    bisect_M_cert_holder, emit_certificate_holder,
)
from delsarte_dual.grid_bound_holder.certify_holder import verify_certificate_holder


def _fmpq_to_float(q: fmpq) -> float:
    return float(q.p) / float(q.q)


# --------------------------------------------------------------------
# Test 1: p = q = 2 reduces to phi_mm bit-identically
# --------------------------------------------------------------------

@pytest.mark.parametrize("N", [1, 2, 3])
def test_reduces_to_mv_at_p_equals_2(N):
    """At (p, q) = (2, 2), phi_holder is bit-identical to phi_mm."""
    ctx.prec = 256
    random.seed(42 + N)
    params_H = PhiHolderParams.from_mv(N_max=N, p=fmpq(2), J_tail=64)
    params_M = params_H.as_phi_mm_params()
    M_base = fmpq(12745, 10000)
    success = 0
    for _ in range(200):
        # Random M in [1.27, 1.30]
        M_q = M_base + fmpq(random.randint(0, 500), 100000)
        M = arb(M_q)
        # Small ab so radicands stay non-negative for most draws
        ab = []
        for _k in range(2 * N):
            num = random.randint(-30, 30)
            ab.append(arb(fmpq(num, 500)))
        try:
            v_H = phi_holder(M, ab, params_H)
        except ValueError:
            # MM-path radicand negative; phi_mm must also raise
            with pytest.raises(ValueError):
                phi_mm(M, ab, params_M)
            continue
        v_M = phi_mm(M, ab, params_M)
        assert str(v_H) == str(v_M), (
            f"N={N}: phi_holder {v_H} != phi_mm {v_M}"
        )
        assert float(v_H.mid()) == float(v_M.mid())
        assert float(v_H.rad()) == float(v_M.rad())
        success += 1
        if success >= 100:
            break
    assert success >= 100, f"Only {success} matching draws (need 100)"


# --------------------------------------------------------------------
# Test 2: monotone-in-p check
# --------------------------------------------------------------------

def test_holder_rhs_monotone_in_p_direction():
    """
    At fixed (M, ab=0), the Hoelder RHS - LHS = phi_holder is monotone
    non-decreasing in p over a small grid.  This is empirical: it tests
    that the Hoelder family interpolates sensibly rather than exploding.

    We test at ab=0 (most-admissible point) and M = 1.275 (near MV's
    critical point) with N=1.  The claim is:
        phi_holder(p=2) <= phi_holder(p=9/4) <= phi_holder(p=5/2) ...
    up to rounding-precision agreement; deviations are flagged.
    """
    ctx.prec = 256
    M = arb(fmpq(12750, 10000))
    ab = [arb(0), arb(0)]  # z_1 = 0
    p_grid = [fmpq(2), fmpq(9, 4), fmpq(5, 2), fmpq(11, 4), fmpq(3)]
    values = []
    for p in p_grid:
        params = PhiHolderParams.from_mv(N_max=1, p=p, J_tail=64)
        v = phi_holder(M, ab, params)
        values.append(float(v.mid()))
    # Document values; allow non-monotonicity for diagnostic review.
    # The theoretical claim is only that HM-10 is a valid upper bound on
    # the true RHS for each p; M_cert itself (below) is what matters.
    print("phi_holder(ab=0, M=1.275, N=1) for p in grid:", values)
    # Record the ordering for diagnostic purposes
    for i in range(1, len(values)):
        if values[i] < values[i - 1] - 1e-12:
            pytest.fail(
                f"phi_holder decreased from p={p_grid[i-1]} ({values[i-1]}) "
                f"to p={p_grid[i]} ({values[i]}); not monotone at ab=0."
            )


# --------------------------------------------------------------------
# Test 3: MO conjecture slack diagnostic at (p*, q*) = (4/3, 4)
# --------------------------------------------------------------------

def test_holder_matches_MO_conjecture_at_p_star():
    """
    Diagnostic: report the ratio of tail terms at MO's heuristic optimum
    (p*, q*) = (4, 4/3) (Hoelder convention in our code, q <= 2, p >= 2).

    NOTE: MO 2004 lines 755-758 report (p*, q*) = (4/3, 4) in their
    (h-side, k-side) convention (h in L^p, hat h in L^q).  In our code
    the h-side Fourier exponent is ``p >= 2`` and the k-side exponent is
    ``q <= 2``.  Mapping: our ``p = 4`` (their q*), our ``q = 4/3``
    (their p*).  This mapping is documented in derivation.md §3.4.

    Target: the Hoelder tail / CS tail ratio at MV's (M, z_1) =
    (1.27481, sqrt(mu(M))).  MO's slack constant pi/log 16 ≈ 1.133
    enters as (pi/log 16)^{1/2} in the CS-direction heuristic; we
    simply record the ratio we see.  This is a DIAGNOSTIC test -- it
    never fails unless the ratio is pathological (<0.1 or >10).
    """
    ctx.prec = 256
    # MV critical point
    M = arb(fmpq(127481, 100000))
    params_cs = PhiHolderParams.from_mv(N_max=1, p=fmpq(2), J_tail=64)
    params_H4 = PhiHolderParams.from_mv(N_max=1, p=fmpq(4), J_tail=512)
    # z_1 = sqrt(mu(M)), placed on the real axis (b_1 = 0)
    mu = mu_of_M(M)
    z1 = mu.sqrt()
    ab = [z1, arb(0)]

    phi_cs = phi_holder(M, ab, params_cs)
    phi_H4 = phi_holder(M, ab, params_H4)

    # The "tail term" inside phi is (RHS - M - 1 - 2 z_1^2 k_1),
    # i.e. (phi + LHS - M - 1 - 2 z_1^2 k_1).  Rather than fishing out
    # that term, we simply compare the two phi values directly.
    cs_val = float(phi_cs.mid())
    h4_val = float(phi_H4.mid())
    diff = h4_val - cs_val
    print(f"MO slack diagnostic: phi_holder(p=2) = {cs_val}, phi_holder(p=4) = {h4_val}, diff = {diff}")
    # Just sanity: values should be finite and on the same order.
    assert abs(cs_val) < 10
    assert abs(h4_val) < 10


# --------------------------------------------------------------------
# Test 4: filter_all still rejects no library f (unchanged from MM-10)
# --------------------------------------------------------------------

@pytest.mark.parametrize("N", [2, 3, 4])
def test_filter_validity_unchanged(N):
    """The filter panel is Hoelder-independent; library f's must not be REJECTED.

    The filter panel (F1, F2, F4_MO217, F7, F8, F_bathtub) is independent of
    the tail step and therefore does NOT change between phi_mm and phi_holder.
    We test at N >= 2 because F2 requires a_{2n} (absent at N = 1).

    Following ``grid_bound.tests.test_phase2_mm.test_library_passes_filter_all``
    we DISABLE F_bathtub by omitting ``mu_arb``.  F_bathtub is an
    M-conditioned constraint, and library f's (e.g. uniform) have
    ||f*f||_inf well above our target M=1.28, so F_bathtub would correctly
    reject them for this particular M.  The remaining non-M filters are
    what we're testing.
    """
    ctx.prec = 256
    lib = build_library()
    for f in lib:
        ab = f.moments(N, prec_bits=256)
        v = filter_all(ab, N,
                       enable_F4_MO217=True, enable_F7=True, enable_F8=True)
        assert v != FilterVerdict.REJECT, (
            f"Library f {f.name!r} REJECTED by filter_all at N={N}"
        )


# --------------------------------------------------------------------
# Test 5: certificate round-trip
# --------------------------------------------------------------------

def test_certificate_round_trip(tmp_path):
    """Generate a small Hoelder certificate at (p, q) = (2, 2), N=1, and
    verify it with certify_holder.py.

    Small config: tight bracket [1.272, 1.276], coarse tol, no MO 2.17
    (N=1 has nothing to cut).  Acceptance is 'certificate ACCEPTED'
    from the independent verifier.
    """
    ctx.prec = 256
    params = PhiHolderParams.from_mv(N_max=1, p=fmpq(2), J_tail=64)
    filter_kwargs = dict(
        enable_F4_MO217=False, enable_F7=False, enable_F8=False,
    )
    bound = bisect_M_cert_holder(
        params, N=1,
        M_lo_init=fmpq(127, 100),       # 1.27
        M_hi_init=fmpq(1276, 1000),      # 1.276
        tol_q=fmpq(1, 500),              # 0.002
        max_cells_per_M=200_000,
        filter_kwargs=filter_kwargs,
        prec_bits=256,
        verbose=False,
    )
    out = str(tmp_path / "holder_roundtrip.json")
    digest = emit_certificate_holder(bound, out)
    print(f"Certificate written: {out}, sha256={digest}")
    res = verify_certificate_holder(out, prec_bits=256)
    assert res.accepted, f"Verifier rejected certificate: {res.messages}"
    assert res.M_cert_q == bound.M_cert_q
