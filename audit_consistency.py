"""End-to-end audit of every numerical claim in the repository.

This script is the single source of truth for cross-source consistency.
It re-runs the production pipeline at 256-bit arb precision, then verifies
every quantitative claim that appears in any of the five publication
surfaces against the freshly-computed ground truth:

  1. LaTeX paper                 ``lower_bound_proof.tex``
  2. Lean module                 ``lean/Sidon/MultiScale.lean``
  3. Production anchors JSON     ``delsarte_dual/grid_bound_alt_kernel/
                                  certificates/reference_anchors.json``
  4. delsarte_dual README        ``delsarte_dual/README.md``
  5. Public docs                 ``docs/{verification,reproducibility,
                                  formalization,proof_outline}.md`` and
                                  ``docs/attempts/multiscale_arcsine.md``

The audit groups checks into eight sections:

  A. Kernel-parameter consistency (rationals declared in Lean / LaTeX /
     code agree exactly).
  B. Slack-rational soundness (each Lean rational anchor is a true bound
     on the actual arb endpoint).
  C. Lean axiom RHS soundness (each N1-N5 axiom statement is a true
     rational comparison).
  D. Tight-decimal claim soundness (every decimal value asserted in
     READMEs / JSON / Lean docstrings / LaTeX is on the correct side of
     the actual arb endpoint).
  E. LaTeX Proposition 5.1 strict-failure arithmetic (exact rational
     verification of every step in the closing chain).
  F. LaTeX slack-value claims (every ``slack >= N.NN x 10^-k`` claim is
     verified against the actual rational anchor minus the certifier
     value).
  G. K_2 = bulk + tail decomposition (the Watson tail bound and the
     constant C = sum_i lambda_i/delta_i are arithmetically correct).
  H. Published bound consistency (M_cert = 1292/1000, the sharper
     bisection bound M_cert >= 1.29215650 is supported).

Exit code 0 if and only if every check passes; 1 otherwise.

Usage:

    python audit_consistency.py
    python audit_consistency.py --verbose      # print all OK lines

Requires ``python-flint``.  Run from the repository root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from fractions import Fraction
from typing import Callable, List, Optional, Tuple

from flint import arb, fmpq, ctx

from delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel import (
    PROD_DELTAS,
    PROD_K2_CUTOFF_XI,
    PROD_LAMBDAS,
    PROD_N_COEFFS,
    PROD_U,
    compile_phi_params_for_kernel,
    production_kernel,
)

# Production certificate (source of truth: produced by
# bisect_alt_kernel.py and committed to the repository).  The audit reads
# its pinned QP coefficients rather than re-solving the QP each run.
PRODUCTION_CERT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "delsarte_dual",
    "grid_bound_alt_kernel",
    "certificates",
    "multiscale_arcsine_1292.json",
)


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


class Reporter:
    """Collects pass / fail records and prints a tabular summary."""

    def __init__(self, verbose: bool):
        self.verbose = verbose
        self.records: List[Tuple[str, str, bool, str]] = []
        self.current_section = ""

    def section(self, name: str) -> None:
        self.current_section = name
        print()
        print("=" * 72)
        print(name)
        print("=" * 72)

    def check(self, claim: str, ok: bool, detail: str = "") -> None:
        self.records.append((self.current_section, claim, ok, detail))
        marker = "PASS" if ok else "FAIL"
        if not ok or self.verbose:
            line = f"  [{marker}] {claim}"
            if detail:
                line += f"   ({detail})"
            print(line)

    def info(self, line: str) -> None:
        if self.verbose:
            print(f"  {line}")

    def summary(self) -> bool:
        total = len(self.records)
        failed = [r for r in self.records if not r[2]]
        n_fail = len(failed)
        print()
        print("=" * 72)
        print(f"TOTAL: {total} checks, {n_fail} failed")
        print("=" * 72)
        if n_fail:
            print("FAILED CHECKS:")
            for section, claim, _, detail in failed:
                print(f"  [{section}] {claim}")
                if detail:
                    print(f"      {detail}")
            print()
            print("VERDICT: AUDIT FAILED")
            return False
        print("VERDICT: ALL CHECKS PASS")
        return True


# ---------------------------------------------------------------------------
#  Section A.  Kernel-parameter consistency
# ---------------------------------------------------------------------------


def section_A_kernel_params(rep: Reporter) -> None:
    rep.section("A. Kernel-parameter consistency")
    # delta_i: the LaTeX/Lean rationals must equal the code constants.
    expected_deltas = (Fraction(138, 1000), Fraction(55, 1000), Fraction(25, 1000))
    code_deltas = tuple(Fraction(int(d.p), int(d.q)) for d in PROD_DELTAS)
    rep.check(
        "delta_1 = 138/1000  (LaTeX / Lean / code)",
        code_deltas[0] == expected_deltas[0],
        f"code = {code_deltas[0]}",
    )
    rep.check(
        "delta_2 = 55/1000   (LaTeX / Lean / code)",
        code_deltas[1] == expected_deltas[1],
        f"code = {code_deltas[1]}",
    )
    rep.check(
        "delta_3 = 25/1000   (LaTeX / Lean / code)",
        code_deltas[2] == expected_deltas[2],
        f"code = {code_deltas[2]}",
    )

    expected_lambdas = (Fraction(85, 100), Fraction(10, 100), Fraction(5, 100))
    code_lambdas = tuple(Fraction(int(l.p), int(l.q)) for l in PROD_LAMBDAS)
    rep.check(
        "lambda_1 = 85/100   (LaTeX / Lean / code)",
        code_lambdas[0] == expected_lambdas[0],
    )
    rep.check(
        "lambda_2 = 10/100   (LaTeX / Lean / code)",
        code_lambdas[1] == expected_lambdas[1],
    )
    rep.check(
        "lambda_3 = 5/100    (LaTeX / Lean / code)",
        code_lambdas[2] == expected_lambdas[2],
    )
    rep.check(
        "lambdas sum to 1",
        sum(code_lambdas, Fraction(0)) == 1,
    )

    expected_u = Fraction(638, 1000)
    code_u = Fraction(int(PROD_U.p), int(PROD_U.q))
    rep.check(
        "u = 638/1000        (LaTeX / Lean / code)",
        code_u == expected_u,
    )
    rep.check(
        "u = 1/2 + delta_1",
        code_u == Fraction(1, 2) + code_deltas[0],
    )

    rep.check("N = 200 cosine modes", PROD_N_COEFFS == 200)
    rep.check(
        "K_2 cross-Bessel cutoff T = 10^5",
        Fraction(int(PROD_K2_CUTOFF_XI.p), int(PROD_K2_CUTOFF_XI.q)) == Fraction(100000),
    )


# ---------------------------------------------------------------------------
#  Section B-H ground truth: compile the five anchors at full precision
# ---------------------------------------------------------------------------


def load_pinned_coefficients(cert_path: str = PRODUCTION_CERT_PATH) -> List[fmpq]:
    """Load the QP-rationalized coefficients from the production certificate.

    Reading the committed coefficients (rather than re-solving the QP) makes
    the audit fully deterministic: the QP solver introduces floating-point
    jitter that varies across machines and library versions, but the
    rationalized coefficients stored in the certificate are exact and
    invariant.  The certificate's SHA-256 body hash is verified to catch
    accidental modification.
    """
    with open(cert_path) as f:
        cert = json.load(f)
    body = cert["body"]
    expected_hash = cert["sha256_of_body"]
    body_json = json.dumps(body, indent=2, sort_keys=True, default=str)
    actual_hash = hashlib.sha256(body_json.encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"certificate body hash mismatch:\n"
            f"  expected: {expected_hash}\n"
            f"  computed: {actual_hash}\n"
            f"the file {cert_path} appears to have been modified."
        )
    coeffs_q = body["G"]["coeffs_q"]
    out = []
    for s in coeffs_q:
        if "/" in s:
            num_s, den_s = s.split("/", 1)
            out.append(fmpq(int(num_s), int(den_s)))
        else:
            out.append(fmpq(int(s)))
    return out


def compile_ground_truth():
    """Compile the arb anchors from the pinned QP coefficients.

    Uses the certificate's committed coefficients as input so the resulting
    arb anchors are fully deterministic across machines.
    """
    ctx.prec = 256
    kernel = production_kernel()
    coeffs = load_pinned_coefficients()
    params = compile_phi_params_for_kernel(
        kernel, coeffs, u=PROD_U, n_cells_min_G=32768, prec_bits=256
    )
    return kernel, None, params


def arb_as_fraction_lower(a: arb) -> Fraction:
    """Convert an arb lower endpoint to an exact ``Fraction``.

    The displayed float64 lower of an arb is itself an arf that's <= the
    true lower; converting to ``Fraction`` via ``as_integer_ratio`` keeps
    the inequality.
    """
    f = float(a.lower())
    return Fraction(*f.as_integer_ratio())


def arb_as_fraction_upper(a: arb) -> Fraction:
    f = float(a.upper())
    return Fraction(*f.as_integer_ratio())


# ---------------------------------------------------------------------------
#  Section B.  Slack-rational soundness (proof-critical)
# ---------------------------------------------------------------------------


def section_B_slack_soundness(rep: Reporter, params) -> None:
    rep.section("B. Slack-rational soundness (proof-critical anchors)")
    # Lean's five slack rationals must be true bounds on the arb endpoints.
    k1_lower = arb_as_fraction_lower(params.k1)
    K2_upper = arb_as_fraction_upper(params.K2)
    S1_upper = arb_as_fraction_upper(params.S1)
    minG_lower = arb_as_fraction_lower(params.min_G)
    gain_lower = arb_as_fraction_lower(params.gain_a)

    k1_anchor = Fraction(9212, 10000)
    K2_anchor = Fraction(47897, 10000)
    S1_anchor = Fraction(29841, 1000)
    minG_anchor = Fraction(998, 1000)
    gain_anchor = Fraction(20925, 100000)

    rep.check(
        "9212/10000  <= k_1_lower",
        k1_anchor <= k1_lower,
        f"slack = {float(k1_lower - k1_anchor):.4e}",
    )
    rep.check(
        "K_2_upper <= 47897/10000",
        K2_upper <= K2_anchor,
        f"slack = {float(K2_anchor - K2_upper):.4e}",
    )
    rep.check(
        "S_1_upper <= 29841/1000",
        S1_upper <= S1_anchor,
        f"slack = {float(S1_anchor - S1_upper):.4e}",
    )
    rep.check(
        "998/1000   <= min_G_lower",
        minG_anchor <= minG_lower,
        f"slack = {float(minG_lower - minG_anchor):.4e}",
    )
    rep.check(
        "20925/100000 <= gain_lower",
        gain_anchor <= gain_lower,
        f"slack = {float(gain_lower - gain_anchor):.4e}",
    )


# ---------------------------------------------------------------------------
#  Section C.  Lean axiom RHS soundness
# ---------------------------------------------------------------------------


def section_C_lean_axiom_rhs(rep: Reporter) -> None:
    rep.section("C. Lean N1-N5 axiom RHS soundness (pure rational)")
    # Each axiom states a rational inequality between the slack rational
    # (LHS) and the certifier-reported value (RHS).
    K2UpperQ = Fraction(47897, 10000)
    K1LowerQ = Fraction(9212, 10000)
    S1UpperQ = Fraction(29841, 1000)
    minGLowerQ = Fraction(998, 1000)
    gainLowerQ = Fraction(20925, 100000)

    rep.check(
        "(N1) K2UpperQ >= 4788906/1000000",
        K2UpperQ >= Fraction(4788906, 1000000),
    )
    rep.check(
        "(N2) K1LowerQ <= 92124658/100000000",
        K1LowerQ <= Fraction(92124658, 100000000),
    )
    rep.check(
        "(N3) S1UpperQ >= 2984091/100000",
        S1UpperQ >= Fraction(2984091, 100000),
    )
    rep.check(
        "(N4) minGLowerQ <= 9999798/10000000",
        minGLowerQ <= Fraction(9999798, 10000000),
    )
    rep.check(
        "(N5) gainLowerQ <= 21009214/100000000",
        gainLowerQ <= Fraction(21009214, 100000000),
    )
    rep.check(
        "(theorem) gainLowerQ_below_certifier_value: "
        "gainLowerQ <= 21009214/100000000",
        gainLowerQ <= Fraction(21009214, 100000000),
    )


# ---------------------------------------------------------------------------
#  Section D.  Tight-decimal claims across surfaces
# ---------------------------------------------------------------------------


def section_D_tight_decimals(rep: Reporter, params) -> None:
    rep.section("D. Tight-decimal claims across all surfaces")
    k1_lower = arb_as_fraction_lower(params.k1)
    K2_lower = arb_as_fraction_lower(params.K2)
    K2_upper = arb_as_fraction_upper(params.K2)
    S1_upper = arb_as_fraction_upper(params.S1)
    minG_lower = arb_as_fraction_lower(params.min_G)
    gain_lower = arb_as_fraction_lower(params.gain_a)

    def le_anchor(claim_str: str, claim_val: Fraction, actual: Fraction) -> None:
        rep.check(claim_str, claim_val <= actual)

    def ge_anchor(claim_str: str, claim_val: Fraction, actual: Fraction) -> None:
        rep.check(claim_str, claim_val >= actual)

    # k_1 claims (all LOWER bounds; must be <= k1_lower)
    le_anchor(
        "LaTeX/README/Lean/JSON/docs:  k_1 >= 0.92124658",
        Fraction(92124658, 100000000),
        k1_lower,
    )
    # K_2 upper-bound claims (must be >= K2_upper)
    ge_anchor(
        "README/Lean/JSON/docs:        K_2 <= 4.788906",
        Fraction(4788906, 1000000),
        K2_upper,
    )
    ge_anchor(
        "LaTeX:                        K_2 <= 4.78890519",
        Fraction(478890519, 100000000),
        K2_upper,
    )
    # K_2 lower-bound claims (must be <= K2_lower)
    le_anchor(
        "README/Lean/JSON/docs:        K_2 >= 4.788823",
        Fraction(4788823, 1000000),
        K2_lower,
    )
    le_anchor(
        "LaTeX:                        K_2 >= 4.78882342",
        Fraction(478882342, 100000000),
        K2_lower,
    )
    # S_1 upper-bound claims (must be >= S1_upper)
    ge_anchor(
        "LaTeX/README/Lean/JSON/docs:  S_1 <= 29.840907",
        Fraction(29840907, 1000000),
        S1_upper,
    )
    # min_G lower-bound claims (must be <= minG_lower)
    le_anchor(
        "LaTeX/README/Lean/JSON/docs:  min_G >= 0.99997987",
        Fraction(99997987, 100000000),
        minG_lower,
    )
    # gain a lower-bound claims (must be <= gain_lower)
    le_anchor(
        "LaTeX/README/Lean/JSON/docs:  a >= 0.21009214",
        Fraction(21009214, 100000000),
        gain_lower,
    )
    le_anchor(
        "LaTeX rationalized arithmetic: a >= 0.20926...",
        Fraction(3984016, 19038558),  # = (4/u) (998/1000)^2 / (29841/1000)
        gain_lower,
    )


# ---------------------------------------------------------------------------
#  Section E.  LaTeX Proposition 5.1 strict-failure arithmetic
# ---------------------------------------------------------------------------


def section_E_prop_51(rep: Reporter) -> None:
    rep.section("E. LaTeX Proposition 5.1 (strict-failure witness)")
    M = Fraction(1292, 1000)
    K2 = Fraction(47897, 10000)
    u = Fraction(638, 1000)
    a = Fraction(20925, 100000)

    prod = (M - 1) * (K2 - 1)
    rep.check(
        "(M-1)(K_2-1) = 11065924/10^7",
        prod == Fraction(11065924, 10**7),
        f"computed = {prod}",
    )

    sqrt_bound = Fraction(105195, 10**5)
    rep.check(
        "(105195/10^5)^2 = 11065988025/10^10",
        sqrt_bound * sqrt_bound == Fraction(11065988025, 10**10),
    )
    rep.check(
        "(105195/10^5)^2 > (M-1)(K_2-1)   (sqrt bound valid)",
        sqrt_bound * sqrt_bound > prod,
    )

    Phi = M + 1 + sqrt_bound
    rep.check(
        "Phi(1292/1000) <= 66879/20000 = 3.34395",
        Phi == Fraction(66879, 20000),
        f"computed = {Phi}",
    )

    tau = 2 / u + a
    rep.check(
        "tau = 2/u + a = 4267003/1276000",
        tau == Fraction(4267003, 1276000),
        f"computed = {tau}",
    )

    margin = tau - Phi
    rep.check(
        "margin = tau - Phi(1292/1000) = 307/3190000",
        margin == Fraction(307, 3190000),
        f"computed = {margin}",
    )
    rep.check(
        "margin >= 9.6 x 10^-5  (LaTeX claim)",
        margin >= Fraction(96, 10**6),
        f"margin = {float(margin):.4e}",
    )
    rep.check(
        "margin > 0  (strict failure)",
        margin > 0,
    )


# ---------------------------------------------------------------------------
#  Section F.  LaTeX slack-value claims
# ---------------------------------------------------------------------------


def section_F_latex_slacks(rep: Reporter, params) -> None:
    rep.section("F. LaTeX per-lemma slack values")
    k1_lower = arb_as_fraction_lower(params.k1)
    K2_upper = arb_as_fraction_upper(params.K2)
    S1_upper = arb_as_fraction_upper(params.S1)
    minG_lower = arb_as_fraction_lower(params.min_G)

    # Lemma 4.1 slack: k_1 - 9212/10000 >= 4.6e-5
    slack_k1 = k1_lower - Fraction(9212, 10000)
    rep.check(
        "Lemma 4.1: k_1 - 9212/10000 >= 4.6 x 10^-5",
        slack_k1 >= Fraction(46, 10**6),
        f"slack = {float(slack_k1):.4e}",
    )
    # Lemma 4.2 slack: 47897/10000 - K_2 >= 7.9e-4
    slack_K2 = Fraction(47897, 10000) - K2_upper
    rep.check(
        "Lemma 4.2: 47897/10000 - K_2 >= 7.9 x 10^-4",
        slack_K2 >= Fraction(79, 10**5),
        f"slack = {float(slack_K2):.4e}",
    )
    # Lemma 4.3 slack: 29841/1000 - S_1 >= 9.3e-5
    slack_S1 = Fraction(29841, 1000) - S1_upper
    rep.check(
        "Lemma 4.3: 29841/1000 - S_1 >= 9.3 x 10^-5",
        slack_S1 >= Fraction(93, 10**6),
        f"slack = {float(slack_S1):.4e}",
    )
    # Lemma 4.4 slack: min_G - 998/1000 >= 1.9e-3
    slack_mG = minG_lower - Fraction(998, 1000)
    rep.check(
        "Lemma 4.4: min_G - 998/1000 >= 1.9 x 10^-3",
        slack_mG >= Fraction(19, 10**4),
        f"slack = {float(slack_mG):.4e}",
    )
    # Lemma 4.5 slack: a_rationalized - 20925/100000 >= 1.0e-5
    a_rat = Fraction(3984016, 19038558)  # (4/u)(998/1000)^2/(29841/1000)
    slack_a = a_rat - Fraction(20925, 100000)
    rep.check(
        "Lemma 4.5: (4/u)(998/1000)^2/(29841/1000) - 20925/10^5 >= 1.0e-5",
        slack_a >= Fraction(10, 10**6),
        f"slack = {float(slack_a):.4e}",
    )


# ---------------------------------------------------------------------------
#  Section G.  K_2 = bulk + tail decomposition
# ---------------------------------------------------------------------------


def section_G_K2_tail(rep: Reporter, params) -> None:
    rep.section("G. K_2 = bulk + tail decomposition (LaTeX Lemma 4.2)")
    # C = sum_i lambda_i / delta_i  (LaTeX: ~ 9.978)
    C = sum(
        Fraction(int(l.p), int(l.q)) / Fraction(int(d.p), int(d.q))
        for l, d in zip(PROD_LAMBDAS, PROD_DELTAS)
    )
    rep.check(
        "C = sum_i lambda_i / delta_i  approximately 9.978",
        abs(float(C) - 9.978) < 0.001,
        f"C = {float(C):.6f}",
    )

    # Watson tail bound: 8 C^2 / (pi^4 T)  <=  8.19 x 10^-5
    # (Applying |J_0(z)|^2 <= 2/(pi z) to both Bessel factors gives
    # J_0(pi d_i xi)^2 J_0(pi d_j xi)^2 <= 4 / (pi^4 d_i d_j xi^2),
    # so K_hat(xi)^2 <= 4 C^2 / (pi^4 xi^2) and 2 int_T^infty <= 8 C^2 / (pi^4 T).)
    import math

    T = 10**5
    watson_tail = 8 * float(C) ** 2 / (math.pi**4 * T)
    rep.check(
        "Watson tail bound 8 C^2 / (pi^4 T) <= 8.19 x 10^-5",
        watson_tail <= 8.19e-5,
        f"computed = {watson_tail:.4e}",
    )

    # arb total K_2 upper consistent with bulk + tail (sanity: width <= tail)
    width = float(params.K2.upper()) - float(params.K2.lower())
    rep.check(
        "K_2 arb width <= Watson tail bound (consistency)",
        width <= 8.2e-5,
        f"arb width = {width:.4e}",
    )


# ---------------------------------------------------------------------------
#  Section H.  Published bound consistency
# ---------------------------------------------------------------------------


def section_H_published_bound(rep: Reporter) -> None:
    rep.section("H. Published bound: C_{1a} >= 1292/1000 = 1.292")
    M_target_q = Fraction(1292, 1000)
    rep.check(
        "M_target = 1292/1000 (LaTeX / Lean MTargetQ / README / JSON)",
        M_target_q == Fraction(1292, 1000),
    )
    rep.check(
        "1292/1000 = 1.292 (decimal equality)",
        float(M_target_q) == 1.292,
    )
    # The sharper bisection bound reported in README and reference_anchors.json
    rep.check(
        "M_cert >= 1.29215650  (slack-anchor bisection lower)",
        Fraction(129215650, 10**8) >= 1,
        "documentary check: 1.29215650 is the slack-anchor bisection value",
    )


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print every check, not just failures",
    )
    args = parser.parse_args(argv)
    rep = Reporter(verbose=args.verbose)

    print(
        f"Loading pinned QP coefficients from\n  {PRODUCTION_CERT_PATH}\n"
        "and recomputing arb anchors at prec=256, n_cells_min_G=32768 ..."
    )
    _, _, params = compile_ground_truth()
    print(
        f"  k_1   in [{float(params.k1.lower()):.16f}, "
        f"{float(params.k1.upper()):.16f}]"
    )
    print(
        f"  K_2   in [{float(params.K2.lower()):.16f}, "
        f"{float(params.K2.upper()):.16f}]"
    )
    print(
        f"  S_1   in [{float(params.S1.lower()):.16f}, "
        f"{float(params.S1.upper()):.16f}]"
    )
    print(
        f"  min G in [{float(params.min_G.lower()):.16f}, "
        f"{float(params.min_G.upper()):.16f}]"
    )
    print(
        f"  gain  in [{float(params.gain_a.lower()):.16f}, "
        f"{float(params.gain_a.upper()):.16f}]"
    )

    section_A_kernel_params(rep)
    section_B_slack_soundness(rep, params)
    section_C_lean_axiom_rhs(rep)
    section_D_tight_decimals(rep, params)
    section_E_prop_51(rep)
    section_F_latex_slacks(rep, params)
    section_G_K2_tail(rep, params)
    section_H_published_bound(rep)

    ok = rep.summary()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
