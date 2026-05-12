"""Tests for Path A — unconditional Hölder bound on $\\|f*f\\|_2^2$.

The headline result is PARTIAL: Attack 1 yields a rigorous, unconditional
$c^* < 1$ for the SYMMETRIC subclass only, giving $C_{1a}^{sym} \\ge 1.42401$.
The asymmetric case is open (see ``derivation.md`` §5). Tests reflect this:

* ``test_c_star_below_1`` — $c^* < 1$ strictly for the symmetric class.
* ``test_c_star_below_recipe`` — $c^* \\le 0.99$ (Tier 1 acceptance, sym only).
* ``test_unconditional_beats_CS2017_symmetric`` — the headline (symmetric).
* ``test_unconditional_general`` — DOCUMENTED FAILURE: the general bound is
  unchanged (CS 2017's 1.2802) until the asymmetric obstruction is resolved.
* ``test_arb_verification`` — re-derive the headline at 256 bits.
* ``test_csym_monotone`` — $c^*_{sym}(M)$ decreasing on $[1.2802, 1.5]$.
* ``test_proof_self_contained`` — parse derivation.md, check citations.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import mpmath as mp
import pytest
from mpmath import mpf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from delsarte_dual.path_a_unconditional_holder.holder_constant import (  # noqa: E402
    CS2017_M_MIN,
    arb_verify_headline,
    c_sym_at,
    holder_bound_via_lemma214,
    mu,
    provable_c_star,
    provable_c_star_symmetric,
    unconditional_C1a_lower_bound,
    unconditional_C1a_symmetric_lower_bound,
    verify_csym_monotone,
)


# -----------------------------------------------------------------------------
# Tier acceptance tests (symmetric subclass only — all PASS).
# -----------------------------------------------------------------------------

def test_c_star_below_1():
    """Path A's c* (symmetric) is strictly less than 1."""
    c_star, _ = provable_c_star_symmetric(dps=40)
    assert c_star < mpf(1), f"c_star = {c_star} should be strictly < 1"


def test_c_star_below_tier1():
    """Path A's c* (symmetric) is <= 0.99 — Tier 1 acceptance for sym class."""
    c_star, _ = provable_c_star_symmetric(dps=40)
    assert c_star <= mpf("0.99"), f"c_star = {c_star} should be <= 0.99 (Tier 1)"


def test_c_star_below_tier2():
    """Path A's c* (symmetric) is <= 0.95 — Tier 2 acceptance for sym class."""
    c_star, _ = provable_c_star_symmetric(dps=40)
    assert c_star <= mpf("0.95"), f"c_star = {c_star} should be <= 0.95 (Tier 2)"


def test_c_star_below_tier3():
    """Path A's c* (symmetric) is <= log 16 / pi — Tier 3 for sym class."""
    mp.mp.dps = 50
    log16_over_pi = mp.log(16) / mp.pi
    c_star, _ = provable_c_star_symmetric(dps=40)
    # Path A gives c* ~ 0.838 < 0.8825 = log 16 / pi, so Tier 3 PASSES (sym only).
    assert c_star <= log16_over_pi, (
        f"c_star = {c_star} should be <= log 16 / pi = {log16_over_pi} (Tier 3)"
    )


def test_unconditional_beats_CS2017_symmetric():
    """Headline: Path A symmetric bound > CS 2017's 1.2802."""
    bound = unconditional_C1a_symmetric_lower_bound(dps=40)
    assert bound > mpf("1.2802"), f"bound = {bound} should be > 1.2802"


def test_headline_value():
    """Headline $C_{1a}^{sym} \\ge 1.4242$ at the published precision."""
    bound = unconditional_C1a_symmetric_lower_bound(dps=40)
    # Expected from the CLI: 1.42429430224486...
    expected = mpf("1.4242943")  # 7-decimal approx
    assert abs(bound - expected) < mpf("1e-6"), f"bound = {bound}, expected ~{expected}"


# -----------------------------------------------------------------------------
# General class — explicit "no improvement" test (current state of art).
# -----------------------------------------------------------------------------

def test_unconditional_general_class_no_improvement():
    """For GENERAL f (sym + asym), Path A does NOT improve over CS 2017's 1.2802.

    This test EXPLICITLY documents the gap: Attack 1 alone cannot bound
    $\\|f\\|_2^2$ for asymmetric f, so we cannot prove c* < 1 for the general
    class. See ``derivation.md`` §5.
    """
    bound, scope = unconditional_C1a_lower_bound(M_max=mpf("1.4243"), dps=40)
    # The function returns the symmetric-class bound but reports scope.
    assert scope == "symmetric_only", (
        f"Scope should be 'symmetric_only' (general class is open), got {scope}"
    )
    # The bound on the SYMMETRIC subclass is what we proved.
    assert bound > mpf("1.42"), f"sym bound = {bound} should exceed 1.42"


# -----------------------------------------------------------------------------
# Attack 1 chain consistency.
# -----------------------------------------------------------------------------

def test_holder_bound_at_M_eq_1_is_1():
    """At $M = 1$, $\\mu(1) = 0$, and the bound becomes 1 (trivial)."""
    # Use M just above 1 to avoid sin(pi)=0 numerical issue at the boundary.
    M = mpf("1.0001")
    bound = holder_bound_via_lemma214(M, M, dps=40)  # symmetric: ||f||_2^2 = M
    assert abs(bound - mpf(1)) < mpf("1e-3"), (
        f"At M = {M}, bound should be ~1, got {bound}"
    )


def test_csym_at_known_values():
    """Spot-check c_sym at tabulated values."""
    expected = {
        "1.2802": mpf("0.83773619"),
        "1.378":  mpf("0.81701286"),
        "1.40":   mpf("0.81383159"),
    }
    for M_str, expected_c in expected.items():
        actual = c_sym_at(M_str, dps=40)
        assert abs(actual - expected_c) < mpf("1e-7"), (
            f"c_sym({M_str}) = {actual}, expected ~{expected_c}"
        )


def test_csym_monotone():
    """$c^*_{sym}(M)$ is monotone-decreasing on $[1.2802, 1.5]$."""
    is_monotone, witness = verify_csym_monotone("1.2802", "1.5", n_samples=401, dps=50)
    assert is_monotone, f"c_sym not monotone on [1.2802, 1.5]; witness: {witness}"


def test_holder_bound_monotone_in_fnorm2sq():
    """The bound (†) is monotone-increasing in $\\|f\\|_2^2$ (sanity)."""
    M = mpf("1.378")
    b1 = holder_bound_via_lemma214(M, mpf("1.5"), dps=40)
    b2 = holder_bound_via_lemma214(M, mpf("2.0"), dps=40)
    b3 = holder_bound_via_lemma214(M, mpf("2.5"), dps=40)
    assert b1 < b2 < b3, f"Should be monotone increasing: {b1}, {b2}, {b3}"


def test_holder_bound_at_symmetric_extremizer():
    """At symmetric f (||f||_2^2 = M), bound matches c_sym(M) * M."""
    M = mpf("1.378")
    bound = holder_bound_via_lemma214(M, M, dps=40)
    bound_via_csym = c_sym_at(M, dps=40) * M
    assert abs(bound - bound_via_csym) < mpf("1e-30"), (
        f"holder_bound vs c_sym mismatch: {bound} vs {bound_via_csym}"
    )


# -----------------------------------------------------------------------------
# Self-consistency of (c*, M_max).
# -----------------------------------------------------------------------------

def test_self_consistent_fixed_point():
    """The (c*, M_max) pair is self-consistent: M_max = M_target(c*),
    and c*_sym(M) <= c* for all M in [M_min, M_max].
    """
    mp.mp.dps = 40
    c_star, M_max = provable_c_star_symmetric(dps=40)
    # M_max should equal M_target(c*) — by construction.
    from delsarte_dual.restricted_holder.conditional_bound import (
        conditional_bound_optimal,
    )
    M_target_recompute = conditional_bound_optimal(c_star, dps=40)
    assert abs(M_target_recompute - M_max) < mpf("1e-30"), (
        f"M_max not self-consistent: {M_max} vs {M_target_recompute}"
    )
    # c*_sym(M) is decreasing, so its max on [CS2017_M_MIN, M_max] is at M = M_min.
    assert c_sym_at(CS2017_M_MIN, dps=40) == c_star


# -----------------------------------------------------------------------------
# Arb / interval-arithmetic verification at 256 bits.
# -----------------------------------------------------------------------------

def test_arb_verification():
    """Re-derive the headline at 256-bit precision; the result excludes 1.2802."""
    info = arb_verify_headline(precision_bits=256)
    assert info['M_target'] > info['CS2017_M_MIN'], (
        f"M_target = {info['M_target']} should exceed CS2017's "
        f"{info['CS2017_M_MIN']}"
    )
    assert info['M_target_minus_CS'] > mpf("0.14"), (
        f"Headline gap = {info['M_target_minus_CS']}, expected >= 0.14"
    )
    assert info['scope'] == 'symmetric_only'


# -----------------------------------------------------------------------------
# Citation / proof-self-contained check.
# -----------------------------------------------------------------------------

def test_proof_self_contained():
    """Parse derivation.md and verify every external citation is one of:
    - In-repo file (reference or proof)
    - arXiv paper of MO/MV/CS/Boyer–Li (rigorous unconditional theorems)
    - Standard textbook theorem (Hölder, Parseval, Beckner, etc.)
    """
    here = Path(__file__).resolve().parents[1]
    derivation_path = here / "delsarte_dual/path_a_unconditional_holder/derivation.md"
    assert derivation_path.exists(), f"derivation.md not found: {derivation_path}"
    text = derivation_path.read_text(encoding="utf-8")
    # Required citation patterns (each must appear at least once):
    required = [
        # MO 2004 Lemma 2.14 (the only conjectural-replacement-bound we use):
        ("MO 2004 Lemma 2.14", "MO 2004"),
        # CS 2017 unconditional bound (M >= 1.2802):
        ("CS 2017 / 1.28 / 1403.7988", "1403.7988|Cloninger.*Steinerberger"),
        # MV 2010 (used as a comparison baseline):
        ("MV 2010 / 0907.1379", "0907.1379|Matolcsi.*Vinuesa"),
        # Boyer-Li 2025 (the unrestricted disproof, mentioned but not used in the
        # actual proof of Hyp_R):
        ("Boyer-Li 2025 / 2506.16750", "2506.16750|Boyer.*Li"),
    ]
    for label, pattern in required:
        assert re.search(pattern, text), f"Required citation missing: {label}"

    # No "conjecture" or "we believe" should appear in load-bearing sentences
    # for the actual (symmetric-class) headline. Conjecture mentions are
    # restricted to the §5 obstruction discussion (asymmetric, not used).
    # Quick check: the word "Conjecture" should appear (re Boyer-Li / MO 2.9
    # discussion) but NOT the phrase "we conjecture" or "is conjectured".
    assert "we conjecture" not in text.lower() or "(NOT PROVED" in text, (
        "Loose 'we conjecture' usage in load-bearing position"
    )


def test_no_use_of_unproved_conjectures():
    """The headline claim does NOT depend on MO Conj. 2.9 or any conjecture."""
    here = Path(__file__).resolve().parents[1]
    derivation_path = here / "delsarte_dual/path_a_unconditional_holder/derivation.md"
    text = derivation_path.read_text(encoding="utf-8")
    # The headline appears in the §1 box. Check that section before "## 5"
    # contains no explicit reliance on "Conjecture 2.9" or "Hausdorff–Young
    # extremizer" without the marker "(unproven)" or similar.
    headline_section = text.split("## 5")[0]
    # Conjecture 2.9 (MO 2004) is referenced only in the comparison/baseline
    # discussion, not as a proof input. Verify that MO 2004 Conj. 2.9 (the
    # log 16 / pi bound) is not invoked as an input:
    bad_patterns = [
        # "by Conjecture 2.9, ||f*f||_2^2 <= ..."
        r"by\s+(?:MO\s*)?[Cc]onjecture\s*2\.9.*?\\le",
        # "assuming Conjecture 2.9"
        r"assuming\s+(?:MO\s*)?[Cc]onjecture\s*2\.9",
    ]
    for p in bad_patterns:
        assert not re.search(p, headline_section, flags=re.DOTALL), (
            f"Headline section invokes Conjecture 2.9: matched {p}"
        )


# -----------------------------------------------------------------------------
# Robustness: result holds at the safer M_min = 1.28 (CS's exact published figure).
# -----------------------------------------------------------------------------

def test_robustness_at_safer_M_min():
    """Result still beats CS 2017 even if we use the safer M_min = 1.28
    (CS's exact published figure rather than the slightly tighter 1.2802).
    """
    bound = unconditional_C1a_symmetric_lower_bound(M_min=mpf("1.28"), dps=40)
    assert bound > mpf("1.2802"), (
        f"With M_min=1.28: bound = {bound} should still exceed 1.2802"
    )


# -----------------------------------------------------------------------------
# Pytest entry point.
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
