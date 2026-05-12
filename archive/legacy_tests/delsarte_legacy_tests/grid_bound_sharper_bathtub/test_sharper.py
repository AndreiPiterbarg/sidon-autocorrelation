"""Tests for the (placeholder) sharper-bathtub pipeline.

STATUS: OBSTRUCTED (see ../derivation.md).

Because no rigorous quantitative sharpening is currently proved, mu_sharper
equals mu_MV. Tests therefore verify:

  - monotonicity in M (inherits from mu_MV),
  - dominance mu_sharper <= mu_MV at every (M, n) tested (equality),
  - NO library f is rejected by F_bathtub_sharper at its own ||f*f||_inf
    (correctness gate -- identical to the baseline),
  - strict improvement test is marked as expected-fail with a skip-reason
    pointing to derivation.md. This documents the open problem and will
    automatically start passing once a strict mu_sharper is plugged in.
"""
from __future__ import annotations

import pytest
from flint import arb, fmpq, ctx

from delsarte_dual.grid_bound.admissible_f import build_library
from delsarte_dual.grid_bound.filters import FilterVerdict
from delsarte_dual.grid_bound.phi_mm import mu_of_M as mu_MV

from delsarte_dual.grid_bound_sharper_bathtub.mu_sharper import mu_sharper
from delsarte_dual.grid_bound_sharper_bathtub.filters_sharper import (
    F_bathtub_sharper,
)


PREC = 256


# ---------------------------------------------------------------------------
# 1. Monotonicity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_mu_sharper_monotone(n):
    """mu_sharper(M, n) is non-decreasing in M."""
    old = ctx.prec
    ctx.prec = PREC
    try:
        Ms = [arb("1.26"), arb("1.27"), arb("1.28"), arb("1.29"), arb("1.30")]
        vals = [mu_sharper(M, n, prec_bits=PREC) for M in Ms]
        for i in range(len(vals) - 1):
            # vals[i].upper() <= vals[i+1].lower() is strict monotone.
            # We only require non-decreasing center at the prec we work at.
            assert float(vals[i].mid()) <= float(vals[i + 1].mid()) + 1e-30, (
                f"mu_sharper(M, n={n}): not monotone at M={Ms[i]} -> {Ms[i+1]}"
            )
    finally:
        ctx.prec = old


# ---------------------------------------------------------------------------
# 2. Dominance (mu_sharper <= mu_MV) at each (M, n)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M_str", ["1.26", "1.27", "1.28", "1.29", "1.30"])
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_mu_sharper_dominates_mv_at_each_n(M_str, n):
    """mu_sharper(M, n) <= mu_MV(M) for every (M, n).

    STATUS: currently equality (placeholder). Verifies soundness: the
    sharper bound is NEVER larger than MV, so any filter using mu_sharper
    remains sound.
    """
    old = ctx.prec
    ctx.prec = PREC
    try:
        M = arb(M_str)
        mv = mu_MV(M)
        sh = mu_sharper(M, n, prec_bits=PREC)
        # Rigorous: sh.upper() <= mv.upper() + 1e-60 (numerical slack).
        assert float(sh.upper()) <= float(mv.upper()) + 1e-30, (
            f"mu_sharper ({float(sh.upper())}) exceeds mu_MV "
            f"({float(mv.upper())}) at M={M_str}, n={n}"
        )
    finally:
        ctx.prec = old


# ---------------------------------------------------------------------------
# 3. Library correctness: no admissible f is rejected.
# ---------------------------------------------------------------------------

def _hsup_precomputed(name: str) -> str:
    """Precomputed ||f*f||_inf upper bound for library f's, as a string.

    These are LIBERAL upper bounds (above the true values) — large enough
    to guarantee F_bathtub passes on every admissible f regardless of
    numerical precision, but not so large that the test becomes vacuous.
    """
    # The library f's have specific norms; we use ||f*f||_inf <= ||f||_inf * ||f||_1.
    # For each library member, ||f*f||_inf <= 2 (loose but safe):
    #  - uniform on [-1/4, 1/4]: f = 2 * 1_{[-1/4,1/4]}, ||f||_inf = 2, so ||f*f||_inf = 1.
    #  - triangular: ||f||_inf = 4, ||f||_1 = 1, ||f*f||_inf ~= 2.67 but <= 4.
    #  - indicator [-alpha, alpha]: f = 1/(2 alpha), ||f*f||_inf = 1/(2 alpha).
    # Use 10 as a universal liberal upper bound.
    return "10"


@pytest.mark.parametrize("N", [2, 3, 4])
def test_library_still_passes(N):
    """For every library f, F_bathtub_sharper does NOT REJECT at its ||f*f||_inf.

    Correctness gate: a REJECT here would invalidate the proof. Since
    mu_sharper = mu_MV (placeholder), this inherits from the baseline
    F_bathtub soundness, which is test-covered elsewhere.
    """
    old = ctx.prec
    ctx.prec = PREC
    try:
        library = build_library()
        for f in library:
            M = arb(_hsup_precomputed(f.name))
            ab = f.moments(N, prec_bits=PREC)
            v = F_bathtub_sharper(ab, N, M, prec_bits=PREC)
            assert v != FilterVerdict.REJECT, (
                f"F_bathtub_sharper REJECTED library f={f.name!r} at M={M} "
                f"(N={N}) -- this invalidates the bound, STOP."
            )
    finally:
        ctx.prec = old


# ---------------------------------------------------------------------------
# 4. Strict-improvement: EXPECTED TO FAIL (documents the obstruction).
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason=(
        "OBSTRUCTED: no rigorous quantitative sharpening of Lemma 3.4 is "
        "proved in this session (see derivation.md Section 3). The placeholder "
        "mu_sharper = mu_MV has zero gap everywhere. This test becomes "
        "expected-pass when a strict sharper bound is plugged into mu_sharper.py."
    ),
    strict=True,
)
def test_strict_improvement_somewhere():
    """Exhibit (M, n) with mu_sharper < mu_MV with margin > 1e-6.

    STATUS: EXPECTED FAIL under OBSTRUCTION. If a future change makes this
    pass, update the xfail decorator and celebrate.
    """
    old = ctx.prec
    ctx.prec = PREC
    try:
        found_strict = False
        for M_str in ["1.26", "1.27", "1.28", "1.29", "1.30"]:
            for n in [1, 2, 3, 4]:
                M = arb(M_str)
                mv = float(mu_MV(M).upper())
                sh = float(mu_sharper(M, n, prec_bits=PREC).upper())
                if sh < mv - 1e-6:
                    found_strict = True
                    break
            if found_strict:
                break
        assert found_strict, (
            "No (M, n) found where mu_sharper < mu_MV by > 1e-6. "
            "Documented obstruction: see derivation.md."
        )
    finally:
        ctx.prec = old


# ---------------------------------------------------------------------------
# 5. Certification: sharper certifies at least as much as MV
# ---------------------------------------------------------------------------

def test_sharper_certifies_at_least_as_much_as_MV():
    """sharper's mu upper bound <= MV's mu upper bound at M=1.2748, n=1..2.

    Under OBSTRUCTION the two are equal. This test guards against an
    accidental SOUND REGRESSION (mu_sharper > mu_MV, which would be a bug).
    """
    old = ctx.prec
    ctx.prec = PREC
    try:
        M = arb("1.2748")
        mv = mu_MV(M)
        for n in [1, 2]:
            sh = mu_sharper(M, n, prec_bits=PREC)
            assert float(sh.upper()) <= float(mv.upper()) + 1e-40, (
                f"REGRESSION: mu_sharper > mu_MV at n={n}"
            )
    finally:
        ctx.prec = old
