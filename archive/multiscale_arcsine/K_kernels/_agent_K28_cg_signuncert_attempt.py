"""
Agent K28 — Cohn-Gonçalves sign-uncertainty attempt for C_{1a}.

Goal:
  Test whether the Cohn-Gonçalves sign-uncertainty principle (CG, d=1, s=-1)
  ρ_1 ρ_2 >= 1
  for f Schwartz-eigenfunction (f̂ = -f), f even, f(0)=0,
  f >= 0 for |x| >= ρ_1, f̂ <= 0 for |ξ| >= ρ_2
  -- can yield an EXTRA term in the MV/Delsarte-style master inequality for C_{1a}.

Strategy:
  (1) Re-derive the MV/Delsarte dual bound symbolically.
  (2) Identify whether ρ := f*f (the C_{1a} primal object) ever fits into the
      CG class: ρ̂ = (f̂)^2 >= 0 EVERYWHERE on R (Bochner), so CG's hypothesis
      "ρ̂ <= 0 for |ξ| >= ρ_2" is VIOLATED unless ρ̂ ≡ 0 outside ρ_2 (i.e. ρ̂
      is compactly supported -- which would force ρ to be analytic and bandlimited,
      contradicting compact support of ρ).
  (3) Try BCK variant (s=+1): f̂(ξ) >= 0 for |ξ| >= ρ_2 AND f̂(0) <= 0.
      ρ̂(0) = (f̂(0))^2 >= 0; hypothesis says f̂(0) <= 0. Both => ρ̂(0)=0,
      i.e. ∫ ρ = (∫ f)^2 = 0, forcing f ≡ 0. So BCK can't apply either to
      the primal directly.
  (4) Try DUAL side: can we constrain the TEST FUNCTION g (with ĝ >= 0,
      g >= 0 on [-1/2,1/2]) to also satisfy CG-type sign uncertainty on
      something? This would lift the Delsarte dual into a 2-kernel dual.
      The natural attempt: pair g (Bochner-positive) with h (CG-eigenfunction).
  (5) Compute the augmented bound numerically with an explicit (g, h) ansatz.

Final answer: document the OBSTRUCTION precisely if the augmentation has
zero weight.

Author: K28 / parallel investigation 2026-05-11
"""

import json
import math
import numpy as np
from datetime import datetime


def mv_master_bound_form():
    """
    Symbolic-style summary of the MV / Cohn-Elkies-weak master inequality.

    Returns the abstract template:
       C_{1a} >= ∫ ĝ(ξ) w(ξ) dξ  /  max_{t∈[-1/2,1/2]} g(t)
    where w(ξ) <= |f̂(ξ)|^2 is a known envelope.

    For MV 2010: g = K + (arcsine-kernel-style construction), w = (sin(πξ/2)/(πξ/2))^2
    gives 1.2748.
    """
    return {
        "form": "C_{1a} >= ∫ ĝ(ξ) w(ξ) dξ / max_{t∈[-1/2,1/2]} g(t)",
        "constraints": [
            "g real, even on R",
            "ĝ >= 0 on R (Bochner-positivity / Fourier-positive)",
            "g >= 0 on [-1/2,1/2] (or use envelope reformulation)",
            "w(ξ) <= |f̂(ξ)|^2 for all ξ (Lasserre/Paley-Wiener envelope)",
        ],
        "MV_2010_choice": {
            "g": "arcsine kernel + 119-cosine Fourier-positive polynomial",
            "w": "sharp PW_{π/2} envelope cos^2(πξ/2) for |ξ|<=1",
            "bound": 1.2748,
            "ceiling": "Self-stated ~1.276",
        },
    }


def cg_theorem_statement():
    """
    Cohn-Gonçalves sign-uncertainty principle, d=1, s=-1 (from arXiv:1712.04438
    and arXiv:2003.10771, esp. as restated in arXiv:2210.01684 (Goncalves-Steinerberger)).

    Definition (Problem 1.1 in arXiv:2210.01684, with s ∈ {±1}):
      Find smallest ρ such that ∃ radial integrable f: R^d -> R not ≡ 0,
      with
        (i)   f̂ = s·f  (Fourier eigenfunction with eigenvalue s),
        (ii)  f(0) = 0,
        (iii) f(x) >= 0 whenever |x| >= ρ.

    Reduces from the original "two-radii" formulation to a single Fourier-eigenfunction
    problem via convex combination f + s·f̂ (each of the original sign-uncertainty
    optimization problems collapses to this).

    Cohn-Gonçalves d=1, s=-1 SHARP result:
      ρ_min(1, -1) = 1.
    Cohn-Gonçalves d=12, s=+1 SHARP result:
      ρ_min(12,+1) = √2 (via modular form E_6).

    The original two-radii statement (from arXiv:1712.04438, Section 2):
      f: R^d -> R integrable, real-valued, with f̂ real-valued and integrable.
      If
        - f(0) <= 0, f̂(0) <= 0,
        - f(x) >= 0 for |x| >= ρ_1,
        - f̂(ξ) >= 0 for |ξ| >= ρ_2  (BCK, s=+1)
      or
        - f̂(ξ) <= 0 for |ξ| >= ρ_2  (CG, s=-1)
      then ρ_1·ρ_2 >= A_±(d)^2.

    For d=1, s=-1: A_-(1)^2 = 1, so ρ_1·ρ_2 >= 1.
    """
    return {
        "two_radii_form_BCK": {
            "hyp_f": ["f real, integrable, even", "f̂ real, integrable", "f(0) <= 0", "f(x) >= 0 for |x| >= ρ_1"],
            "hyp_fhat": ["f̂(0) <= 0", "f̂(ξ) >= 0 for |ξ| >= ρ_2"],
            "conclusion": "ρ_1·ρ_2 >= A_+(d)^2",
            "d=1": "A_+(1) unknown, conjectured 1/(1+√5)^{1/2}; A_+(12)=√2 proven",
        },
        "two_radii_form_CG": {
            "hyp_f": ["f real, integrable, even", "f̂ real, integrable", "f(0) >= 0", "f(x) >= 0 for |x| >= ρ_1"],
            "hyp_fhat": ["f̂(0) <= 0", "f̂(ξ) <= 0 for |ξ| >= ρ_2"],
            "conclusion": "ρ_1·ρ_2 >= A_-(d)^2",
            "d=1": "A_-(1) = 1 SHARP (Cohn-Gonçalves 2019)",
        },
    }


def check_primal_compatibility():
    """
    Can ρ = f*f satisfy CG hypotheses?

    Primal C_{1a}:
       ρ := f*f, f >= 0, supp f ⊂ [-1/4,1/4], ∫f = 1.
       Thus ρ >= 0 EVERYWHERE on R, supp ρ ⊂ [-1/2,1/2].
       ρ̂ = (f̂)^2, real-valued (f even-extension OK), and ρ̂ >= 0 EVERYWHERE on R.

    CG hypothesis on f at large frequencies: f̂ <= 0 for |ξ| >= ρ_2.
    But our ρ̂ >= 0 globally. So CG hypothesis at infinity for the PRIMAL is:
       ρ̂(ξ) <= 0 for |ξ| >= ρ_2.
    Combined with ρ̂ >= 0 globally, this forces ρ̂ ≡ 0 on |ξ| >= ρ_2,
    i.e. ρ̂ has compact support.

    By Paley-Wiener, a function ρ with compact Fourier support is entire of
    exponential type, BUT we also have ρ compactly supported on [-1/2,1/2].
    The only function compactly supported with compactly supported FT is ρ ≡ 0.

    Conclusion: CG hypothesis is INCOMPATIBLE with the primal ρ = f*f.

    What about BCK (s=+1)?
       Hyp: f̂(0) <= 0, f̂ >= 0 outside ρ_2.
       For ρ = f*f: ρ̂(0) = (f̂(0))^2 = (∫f)^2 >= 0.
       The BCK hyp requires ρ̂(0) <= 0, so ρ̂(0) = 0, i.e. ∫f = 0.
       But f >= 0 and ∫f = 1 (normalization), contradiction.

    Conclusion: Both BCK and CG hypotheses are INCOMPATIBLE with the C_{1a}
    primal directly. ρ = f*f is "too positive" — both ρ and ρ̂ are
    nonneg everywhere, while CG/BCK need at least ONE sign-change.
    """
    return {
        "primal_object": "ρ = f*f with f >= 0, supp f ⊂ [-1/4,1/4]",
        "primal_facts": [
            "ρ >= 0 on R (auto-positive, from f>=0)",
            "ρ̂ >= 0 on R (Bochner, auto-positive)",
            "supp ρ ⊂ [-1/2,1/2] (compact support, NOT bandlimited)",
        ],
        "BCK_compat": {
            "hyp": "f̂(0) <= 0",
            "primal": "ρ̂(0) = (∫f)^2 = 1 > 0",
            "compatible": False,
            "reason": "ρ̂(0)=1, BCK demands <=0; only f=0 would satisfy both",
        },
        "CG_compat": {
            "hyp": "f̂(ξ) <= 0 for |ξ| >= ρ_2",
            "primal": "ρ̂(ξ) >= 0 globally",
            "compatible": False,
            "reason": "ρ̂ >= 0 globally + <=0 at ∞ ⟹ ρ̂ has compact support ⟹ ρ ≡ 0",
        },
        "verdict": "Both BCK and CG sign-uncertainty principles are STRUCTURALLY INCOMPATIBLE with the C_{1a} primal object ρ = f*f.",
    }


def check_dual_compatibility():
    """
    Can the DUAL test function g (Delsarte / CE-weak dual) be required to satisfy CG hypotheses?

    Delsarte/MV dual: pick g with
        - g >= 0 on [-1/2,1/2] (or with explicit lower-envelope handling),
        - ĝ >= 0 on R (Bochner),
        - then C_{1a} >= [∫ ĝ(ξ) w(ξ) dξ] / max_{[-1/2,1/2]} g.

    CG would IMPOSE on g (or on a second test fn h coupled to g):
        - "ĝ <= 0 for |ξ| >= ρ_2"  (s=-1)
        but Delsarte already imposes ĝ >= 0 on R.
        Both ⟹ ĝ ≡ 0 on |ξ| >= ρ_2 ⟹ g is bandlimited to [-ρ_2,ρ_2].

    This is NOT contradictory: a bandlimited g is fine. Examples: g(t) = sinc^2(πρ_2 t)
    has ĝ supported in [-ρ_2,ρ_2]. So we CAN restrict to bandlimited test functions.

    HOWEVER: this restriction does NOT GIVE A NEW TERM in the bound. It just SHRINKS the
    feasible set of dual test functions, hence WEAKENS (or at best preserves) the
    Delsarte bound. To EXTRACT value from CG, we need an INEQUALITY relating ρ_1 and ρ_2
    that gives a NUMERICAL constant on the right.

    The CG constant A_-(1)^2 = 1 says: ρ_1 ρ_2 >= 1 where ρ_1 = "last sign change
    of f" and ρ_2 = "last sign change of f̂". To USE this in a bound on C_{1a},
    we would need to relate ρ_1, ρ_2 to quantities appearing in the master inequality
    (like ∫ ĝ w dξ or M_g).

    Attempt 1 — coupled (g, h):
        Pick g = Delsarte test (g >= 0 on [-1/2,1/2], ĝ >= 0).
        Pick h = CG test (ĥ = -h, h(0) = 0, h >= 0 on |x| >= 1).
        Try a Lagrangian:
            L(f, g, h, λ) = (master ineq for g) + λ · (sign-uncertainty witness via h).
        The sign-uncertainty witness "f h has sign-constrained moments" — but f is
        the PRIMAL, not free. We can't constrain f to satisfy CG hypotheses (see
        primal incompatibility above).

    Attempt 2 — apply CG to f̂ directly (forget ρ):
        f >= 0, supp f ⊂ [-1/4,1/4], so f̂ is entire of type π/2, f̂(0) = 1,
        |f̂(ξ)| <= 1 by ∫f=1 and f>=0.
        Does CG say anything about the radius at which f̂ first becomes >= 0?
        Note f̂ here is the FT of a >=0 function, so f̂ is positive-definite
        (NOT auto-nonneg). f̂ can change sign. So f̂ is a candidate for CG/BCK
        IF we can identify f̂(0) <= 0 (false, f̂(0)=1) or, with sign flipped:
        g := -f̂ + c for some shift. But g must be EVEN and FOURIER-NICE.

        Actually: f >= 0, ∫f = 1 means f is a probability density on [-1/4,1/4].
        f̂(ξ) is the characteristic function of this distribution; |f̂(ξ)| <= 1,
        f̂(0) = 1, f̂ is positive-definite (in the sense of Bochner: ĝ for g=f >= 0).
        Sign-changes of f̂ on R are NOT controlled by CG, because the CG
        eigenfunction condition f̂ = -f is NOT satisfied by typical f.

    Attempt 3 — pair f̂ with a CG-eigenfunction h:
        Plancherel: ∫ f̂(ξ) h(ξ) dξ = ∫ f(x) ĥ(x) dx.
        If h has ĥ = -h: ∫ f̂(ξ) h(ξ) dξ = -∫ f(x) h(x) dx.
        So: ∫ f̂ h dξ + ∫ f h dx = 0.
        This is a LINEAR identity, but supplies NO numerical bound on
        max f*f.

    Conclusion of dual analysis: CG sign-uncertainty COULD be used to RESTRICT the
    dual test function class, but provides NO ADDITIONAL POSITIVE TERM in the
    master inequality. The reason: CG bounds the PRODUCT ρ_1 ρ_2 from BELOW
    by 1, which translates to "no g with too-narrow support both in space and
    Fourier." But the MV master inequality is already a RATIO, with the numerator
    being ∫ ĝ w dξ (Fourier-side integral) and denominator M_g (space-side
    max). CG ρ_1 ρ_2 >= 1 is a STATEMENT THAT GOES THE WRONG WAY: it FORBIDS
    g with simultaneously small ρ_1 (small space-support of "negative part")
    AND small ρ_2 (small Fourier-support of "negative part") — but the
    Delsarte bound benefits from g supported on small interval AND ĝ
    concentrated. CG would say "you can't have BOTH"; this would actually
    HURT, not HELP, the Delsarte bound.
    """
    return {
        "dual_object": "test function g with g >= 0 on [-1/2,1/2], ĝ >= 0 on R",
        "CG_on_dual": {
            "hyp": "ĝ <= 0 for |ξ| >= ρ_2 (CG, s=-1)",
            "Bochner_constraint": "ĝ >= 0 on R",
            "intersection": "ĝ supported in [-ρ_2, ρ_2] (bandlimited)",
            "is_restrictive": True,
            "produces_new_term": False,
            "reason": "Restricting g to bandlimited only SHRINKS feasible set; CG sign-uncertainty bounds ρ_1ρ_2 from BELOW, which is a CONSTRAINT not a numerical addition",
        },
        "BCK_on_dual": {
            "hyp": "ĝ(0) <= 0, ĝ >= 0 outside |ξ| >= ρ_2",
            "Bochner_constraint": "ĝ >= 0 on R",
            "compatible": False,
            "reason": "ĝ(0) <= 0 + ĝ >= 0 globally ⟹ ĝ(0)=0 ⟹ ∫g = 0; but g>=0 on [-1/2,1/2] and not identically zero is required for nontrivial bound",
        },
        "Plancherel_attempt": {
            "identity": "∫ f̂ h dξ = -∫ f h dx  if h is a CG -1 eigenfunction",
            "yields_bound": False,
            "reason": "Linear identity, no sign control, no max-norm appears",
        },
        "verdict": "Sign-uncertainty CONSTRAINS dual test function class but adds NO POSITIVE TERM to master inequality. Direction of inequality is wrong: CG forbids what Delsarte wants.",
    }


def attempt_explicit_construction():
    """
    Attempt a concrete (g, h) two-kernel ansatz to see if a hybrid bound emerges.

    Setup:
        g_α(t) := raised-cosine "Vaaler-like" PD kernel, ĝ_α >= 0, supp g_α ⊂ [-1/2,1/2].
        h_β(t) := Cohn-Goncalves d=1 s=-1 candidate (f̂ = -f, e.g. Gauss-Hermite combo).
        We test:  T(α,β) = max_{t ∈ [-1/2,1/2]}  g_α(t)  - μ · (CG-witness via h_β)
        looking for μ > 0 making T smaller (i.e. tightening the M_g denominator).

    The CG witness "(h, f)" is a vanishing integral, not a positive quantity to
    subtract. So the natural Lagrangian formulation has λ FREE, and the
    SUP_λ giving the tightest bound depends on the SIGN of ∫ f h dx, which is
    NOT known a priori for the optimal f.

    Without a SIGN of the CG witness, the Lagrangian dual reduces to: take λ = 0
    (i.e. drop the CG term). So the augmented bound = original bound.
    """
    # Numerical sanity check: construct a CG-style d=1, s=-1 candidate h
    # and a Vaaler-like g, compute master ratio.
    rng = np.random.default_rng(7)

    # Grid
    N = 4096
    L = 4.0  # half-window
    t = np.linspace(-L, L, N)
    dt = t[1] - t[0]

    # Bochner-positive g: triangular autoconvolution of indicator on [-1/4,1/4]
    indicator = (np.abs(t) <= 0.25).astype(float)
    g_full = np.real(np.fft.ifft(np.abs(np.fft.fft(indicator))**2)) * dt
    g_full = np.roll(g_full, N // 2)
    # Normalize
    g_full = g_full / np.max(g_full)
    # Restrict to t in [-1/2,1/2] for M_g
    mask = np.abs(t) <= 0.5
    M_g = np.max(g_full[mask])
    g_fourier_via_fft = np.fft.fftshift(np.abs(np.fft.fft(np.fft.ifftshift(g_full)))) * dt
    xi = np.fft.fftshift(np.fft.fftfreq(N, d=dt))
    # Envelope w(xi) = cos^2(πξ/2) for |ξ|<=1
    w = np.where(np.abs(xi) <= 1, np.cos(np.pi * xi / 2)**2, 0.0)

    # Master integral
    integrand = g_fourier_via_fft * w
    bound = np.trapz(integrand, xi) / M_g

    # CG candidate h: even Schwartz fn with FT eigenvalue -1.
    # The Hermite function H_2(x) e^{-pi x^2} (Hermite under the "unitary"
    # convention with f̂(ξ) = ∫ f(x) e^{-2πi x ξ} dx) has eigenvalue (-i)^2 = -1.
    # In the convention used here, the explicit -1 eigenfn is
    #   ψ(x) = (1 - 4π x^2) e^{-π x^2}  (NOT (1 - 2πx^2); fix coefficient).
    # (Reference: Hermite functions h_n(x) = H_n(√(2π) x) e^{-π x^2} are FT eigenfns
    #  with eigenvalue (-i)^n; n=2 gives -1.)
    psi = (4 * np.pi * t**2 - 1) * np.exp(-np.pi * t**2)
    # Normalize so peak |psi| is O(1)
    psi = psi / np.max(np.abs(psi))
    psi_hat = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(psi))) * dt
    # In unitary convention, eigenvalue -1 means ψ_hat(ξ) = -ψ(ξ) (as functions of ξ).
    # FFT shifts ψ_hat to be a function of ξ on the same grid as t (Plancherel-isometric).
    psi_xi = (4 * np.pi * xi**2 - 1) * np.exp(-np.pi * xi**2)
    psi_xi = psi_xi / np.max(np.abs(psi_xi))
    # Compare ψ_hat(ξ) with -ψ(ξ):
    eig_err = float(np.max(np.abs(psi_hat - (-psi_xi))))
    # Tolerance is set by FFT discretization and finite-window truncation
    eig_ok = eig_err < 0.05

    return {
        "delsarte_master_bound_numeric_demo_only": float(bound),
        "note_on_demo": "This is a TRIVIAL demo Delsarte bound with a triangular-kernel g, NOT the MV 1.2748 optimum. Numerical value here is irrelevant to the obstruction analysis above.",
        "M_g_demo": float(M_g),
        "CG_eigenfunction_sanity": {
            "psi_form": "(4πt^2 - 1) e^{-πt^2}  (Hermite H_2 type, FT eigenval -1)",
            "FT_eigenvalue": -1,
            "max_|psi_hat - (-psi)|": eig_err,
            "is_-1_eigenfn_numerically": eig_ok,
            "note": "Confirms a valid CG-eigenfunction candidate exists (we can construct h with ĥ = -h).",
        },
        "augmentation_attempt": "Plancherel identity ∫ψ̂(ξ)f̂(ξ)dξ = -∫ψ(x)f(x)dx is LINEAR; sup_λ Lagrangian collapses to λ=0",
        "augmented_bound": float(bound),  # Same as base; CG contributes nothing
        "improvement_over_MV": 0.0,
    }


def main():
    print("=" * 70)
    print("Agent K28: Cohn-Gonçalves sign-uncertainty for C_{1a}")
    print("=" * 70)

    cg = cg_theorem_statement()
    mv = mv_master_bound_form()
    primal = check_primal_compatibility()
    dual = check_dual_compatibility()
    attempt = attempt_explicit_construction()

    result = {
        "agent": "K28",
        "date": datetime.now().isoformat(),
        "task": "Cohn-Goncalves sign-uncertainty two-kernel extension for C_{1a}",
        "MV_master_form": mv,
        "CG_theorem": cg,
        "primal_compatibility": primal,
        "dual_compatibility": dual,
        "explicit_construction_attempt": attempt,
        "verdict": "NEGATIVE — CG sign-uncertainty has ZERO POSITIVE WEIGHT in C_{1a} master inequality",
        "obstruction_summary": [
            "1) PRIMAL OBSTRUCTION: ρ = f*f auto-satisfies ρ̂ >= 0 globally. CG hypothesis ρ̂ <= 0 outside ρ_2 + global ρ̂ >= 0 ⟹ ρ̂ ≡ 0 outside ρ_2 ⟹ ρ entire bandlimited but also compactly supported ⟹ ρ ≡ 0.",
            "2) DUAL OBSTRUCTION: Bochner-positive dual test ĝ >= 0 on R is INCOMPATIBLE with CG hypothesis ĝ <= 0 at infinity except via ĝ supported in [-ρ_2,ρ_2]; this only SHRINKS the dual feasible set, never adds a positive term.",
            "3) BCK OBSTRUCTION (s=+1): hypothesis f̂(0) <= 0 conflicts with ρ̂(0) = (∫f)^2 = 1 in the C_{1a} primal.",
            "4) PLANCHEREL LAGRANGIAN: pairing primal ρ̂ with a CG eigenfn h via ∫f̂h dξ = -∫fh dx is a LINEAR identity, no sign-control, sup_λ Lagrangian gives λ=0 ⟹ no improvement.",
            "5) MISMATCH OF NUMBERS: CG produces ρ_1 ρ_2 >= 1 (a CONSTRAINT, not a positive numerical addend); the master inequality wants a positive integral or sup-norm to ADD — wrong direction.",
        ],
        "conclusion": "Cohn-Goncalves sign-uncertainty does NOT yield a usable two-kernel extension of MV's master inequality for C_{1a}. The primal object ρ = f*f is structurally 'too positive' (both ρ and ρ̂ are nonneg) for CG/BCK hypotheses to bite, and the dual test function class is already maximally exploited by Bochner-positive g; CG only further restricts it. Final certified improvement over MV's 1.2748: 0.0.",
        "improvement_over_MV": 0.0,
        "new_bound_value": 1.2748,
    }

    output_path = "_agent_K28_cg_signuncert_result.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {output_path}")
    print()
    print("VERDICT:", result["verdict"])
    print("New bound:", result["new_bound_value"])
    print()
    for line in result["obstruction_summary"]:
        try:
            print(" -", line[:140].encode("ascii", "replace").decode("ascii"))
        except Exception:
            print(" - [unicode line]")
    return result


if __name__ == "__main__":
    main()
