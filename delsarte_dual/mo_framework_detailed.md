# The Martin–O'Bryant Framework: Foundation of the Matolcsi–Vinuesa Bound

**Purpose.** Pin down the *exact* Martin–O'Bryant (MO) framework that Matolcsi–Vinuesa (MV)
inherit, with explicit page/equation citations. This resolves the ambiguities flagged in
`delsarte_dual/mv_construction_detailed.md` §1 (the `0.5747/δ` constant) and §7 (items i–iv).

**Sources used.**
- MO 2004: *The symmetric subset problem in continuous Ramsey theory* (Martin, O'Bryant),
  arXiv:math/0410004, Experimental Mathematics **16** (2007) 145–165.
  Local extract: `delsarte_dual/mo_2004.txt`.
- MO 2009: *The supremum of autoconvolutions, with applications to additive number theory*,
  arXiv:0807.5121. Local extract: `delsarte_dual/mo_2009.txt`.
- MV 2010: Matolcsi–Vinuesa, *Improved bounds on the supremum of autoconvolutions*,
  arXiv:0907.1379. Local extract: `delsarte_dual/mv_paper.txt`.

All page/line numbers below refer to these extracted text files unless otherwise noted.

---

## A. Origin of the `0.5747/δ` constant — **RESOLVED**

### A.1 The kernel `K_β` is an autocorrelation, NOT the raw arcsine density

The kernel MV call "`K(x) = (1/δ) (2/π) / √(1 − 4(x/δ)²)`" is NOT what MO actually use.
MO 2009 (p. 6, eqs. (2.1)–(2.2), lines 349–366) defines

    β(x) := (2/π) / √(1 − 4x²)    if −½ < x < ½,   else 0           (MO 2009 eq. 2.1)

    K_β(x) := (1/δ) · (β ⋆ β)(x/δ),            where ⋆ = autocorrelation.

**Key observation.** `K_β` is the **autocorrelation** `β ⋆ β` rescaled to `[−δ,δ]`, NOT `β` itself.
MO 2009 p. 6 lines 358–359 literally states:

  > "simple calculus verifies that `‖K_β‖₁ = ‖β‖₁² = 1`. Since K_β is defined as an
  > autocorrelation, its Fourier transform K̃_β(j) is automatically real and nonnegative."

This is the reason K̃_β(j) ≥ 0 (a requirement of Lemma 3.3/3.4): the Fourier transform of
an autocorrelation is the squared modulus of a Fourier transform. MO 2009 (p. 6 eq. 2.2):

    K̃_β(j) = u |β̃(j)|² = (1/u) · J₀(π j δ / u)²                          (MO 2009 eq. 2.2)

### A.2 The `0.5747/δ` constant is a *literal* L²-norm-squared, computed via elliptic integrals

MO 2009 p. 6 lines 373–381 give the explicit closed form:

    ‖K_β‖₂² = (1/δ) · ‖β ⋆ β‖₂² < 0.5747 / δ.

The computation uses

    (β ⋆ β)(x) = (2/π) · (1/(2|x|)) · E(√(1 − 1/x²)),    for |x| < 1,

where `E(·)` is the **complete elliptic integral of the first kind** (MO 2009 p. 6 eq.
unnumbered, between eqs. 2.1 and 2.2). This is finite: `β ⋆ β` is bounded everywhere on
`(−1, 1)` (it has a removable logarithmic-type feature at the origin but is square-integrable).

**Therefore `‖K_β‖₂² = 0.5747/δ` is a TRUE L² norm, not a regularized Parseval surrogate.**
The earlier confusion in `mv_construction_detailed.md` §1 (that "K ∉ L²" because the
arcsine density has log divergence of its square) was based on misreading MV. MV do
use the *autocorrelation* of the arcsine, and `β ⋆ β ∈ L²`.

Side note: MO 2004 proved in a precursor (Corollary 2.8, p. 11 lines 676–711) that
*every* autocorrelation `b ⋆ b` with `b` supported on `(−½,½)` and `‖b‖₁ = 1` satisfies
`‖b ⋆ b‖₂² ≥ 0.5745752`. So MO's `K_β` achieves `0.5747`, which is within 0.0002 of the
provable lower limit over **all** symmetric-probability-autocorrelation kernels. This
explains MO 2009 p. 6 line 388: "our particular kernel K_β is at least nearly optimal."

### A.3 Period-`u` vs. period-1 normalisation reconciled

MO use two Fourier conventions simultaneously (MO 2009 p. 4 lines 205–240):

- **Period-1** (standard on `R`): `ĝ(ξ) = ∫ g(x) e^{−2πixξ} dx`. Write `ĝ(j)` for
  integer `j`. This is used in Lemmas 3.1–3.2 (upper bound).
- **Period-`u`** with `u = δ + ½`: `g̃(ξ) = (1/u) ∫ g(x) e^{−2πixξ/u} dx`. This is used
  in Lemmas 3.3–3.4 (lower bound via the second kernel G_{u,n}).

Under both conventions:

    K̂_β(j)  = |J₀(π j δ)|²            (period-1; MV p. 7 line 348)
    K̃_β(j) = (1/u) · |J₀(π j δ/u)|²   (period-u; MO 2009 eq. 2.2, MV p. 4 line 185)

The Parseval identity used in Lemma 3.2 (see §B below) is the period-1 identity
(MO 2009 Lemma 2.2, p. 6 lines 322–334):

    ‖K_β‖₂² = Σ_{r ∈ Z} |K̂_β(r)|².     (MO 2009 Lemma 2.2)

So the `0.5747/δ` is equal to `Σ_r |J₀(π r δ)|⁴` — a **convergent** sum of fourth powers
of Bessel values (since `|J₀(x)| ~ √(2/(πx))` for large `x`, so |J₀|⁴ ~ 4/(π²x²)).

---

## B. MO's exact Cauchy–Schwarz (Lemma 3.2 of MO 2009) — VERBATIM

**Lemma 3.2 of MO 2009 (p. 10, lines 609–664).** Let `f` be a square-integrable pdf
supported on `(−¼, ¼)`, and let `K` be a square-integrable pdf supported on `(−½, ½)`.
Then

    ∫_R (f·f)(x) K(x) dx ≤ 1 + (‖f·f‖_∞ − 1)^{1/2} · (‖K‖₂² − 1)^{1/2}.      (★)

**Proof structure (MO 2009 p. 10 lines 618–663).** In four steps:

1. **Parseval** (Lemma 2.2 of MO 2009): `∫ (f·f)(x) K(x) dx = Σ_r f̂(r) · f̂(r)̄ · K̂(r)
   = Σ_r |f̂(r)|² · K̂(r)`, then pull out r=0: `= 1 + Σ_{r≠0} |f̂(r)|² K̂(r)` since
   `f̂(0) = K̂(0) = 1`.

2. **Cauchy–Schwarz** on the mean-zero Fourier tail:

      Σ_{r≠0} |f̂(r)|² · K̂(r) ≤ (Σ_{r≠0} |f̂(r)|⁴)^{1/2} · (Σ_{r≠0} |K̂(r)|²)^{1/2}.

3. **Parseval again** for `f·f` (which is supported on `(−½,½)` and `L²`):
   `Σ_r |f̂(r)|⁴ = Σ_r |\widehat{f·f}(r)|² = ‖f·f‖_2²`, and
   `Σ_r |K̂(r)|² = ‖K‖_2²`. Subtract the `r=0` term (both equal `1`).

4. **Hölder** `‖f·f‖_2² ≤ ‖f·f‖_∞ · ‖f·f‖_1 = ‖f·f‖_∞ · 1 = ‖f·f‖_∞`
   (since `f` is a pdf implies `f·f` is a pdf hence `‖f·f‖_1 = 1`).

Combining: `Σ_{r≠0} |f̂(r)|² K̂(r) ≤ (‖f·f‖_∞ − 1)^{1/2} · (‖K‖₂² − 1)^{1/2}`, proving (★).

### B.1 Tightness analysis — slack identified by MO themselves

**Slack location:** the Hölder step (#4). It is sharp iff `f·f` is constant on its support
— which a nontrivial autoconvolution never is (`f·f` has a peak at 0 and decays). MO 2004
p. 12 lines 774–782 **explicitly flag this as slack** and formulate a conjecture:

  > MO 2004 Conjecture 2.9 (p. 12 lines 783–791): If `g` is a nonneg pdf supported on
  > `[−¼, ¼]` then `‖g·g‖_2² ≤ (π/log 16) · ‖g·g‖_∞ · ‖g·g‖_1`. If true, this would
  > improve MO's 1.262 to `σ(½) ≥ 0.6512`, i.e. the Sidon bound to `~ 1.3024`.

**The `π/log 16 ≈ 2.2662` improvement factor has NEVER been established**, but MO 2004
note (p. 12 lines 780–782) "we have not been able to realize any success with this
idea, although we believe Conjecture 2.9." So:

- **The C–S step itself is tight** (equality iff Fourier coefficients are proportional).
- **The subsequent Hölder step has up to ~44 % slack** in principle (if Conjecture 2.9
  holds, the `‖f·f‖_∞` in (★) could be replaced by `(π/log 16) · ‖f·f‖_∞ · ‖f·f‖_1`,
  tightening the inequality).
- This is **exactly the open direction** that could push the MV dual past 1.276 — not
  a refinement of G or K but of the `‖f·f‖_2² ≤ ‖f·f‖_∞` step. MV do not revisit this.

### B.2 Tighter Cauchy–Schwarz variants MO considered and rejected

MO 2004 p. 12 lines 753–769 explicitly considered:

- **Hausdorff–Young** instead of Parseval (`ĝ‖_q ≤ ‖g‖_p` with `p = 4/3`, `q = 4`).
  They note this leads to the same framework with the `‖K‖₂²` replaced by `‖K̂‖_{4/3}`,
  and explore optimal kernels for this variant in §2.7.
- **Beckner's sharpening of Hausdorff–Young** (MO 2004 p. 12 lines 760–769). They
  observe it gives `f·f‖_2² ≥ C(q) · ‖K̂‖_p^{−q}` with `C(q) = (q/2)(1 − q/2)^{q/2 − 1}`.
  They did not run numerical experiments to see if `q > 4` gives a larger bound — an
  **explicitly flagged untapped direction**.

---

## C. MO's 1.262 → MV's 1.2748 — the SOURCE of the gain

MO 2009 achieve `f·f‖_∞ > 1.262` (Theorem 1.1, p. 14 line 902). Let
`L := (1/2) · √(0.5747/δ − 1)` and `R := (2/u) + (2/u) · (min_{[0,¼]} G)² · (Σ_{Spec G}
G̃(j)²/K̃(j))^{−1}` (MO 2009 p. 14 lines 857, 878–882). The master inequality is

    ‖f·f‖_∞ + 2L · √(‖f·f‖_∞ − 1) + (1 − R) ≤ 0,   (quadratic in √(‖f·f‖_∞ − 1))

with MO's (`δ = 13/100`, `u = 63/100`, `n = 22`) giving `L = 0.9248`, `R = 3.20874`,
hence `‖f·f‖_∞ ≥ (L² + R − 2)^{1/2} − L)² + 1 > 1.262` (MO 2009 eq. (3.3), p. 14 line 902).

**MV's 1.2748 gain comes from THREE orthogonal improvements** (MV p. 4 lines 200–228,
p. 5 lines 268–305, p. 7 lines 342–353):

1. **Bigger G.** MO used the Selberg magic function `G_{63/100, 22}` (22 cosines). MV
   use an optimised 119-term cosine polynomial with free coefficients (`n = 119`,
   `δ = 0.138`), jumping from MO's `min G / Σ a²/J₀² = 0.03` to MV's `> 0.0713`.
   **This alone** (without the moment improvement) gives `f·f‖_∞ > 1.2743` (MV p. 4 line 228).
2. **Moment improvement** (`|f̂(1)| ≤ M sin(π/M)/π`). This is MV's Lemma 3.3 + Lemma 3.4
   (= MO 2004 Lemma 2.14, MO 2009 lemma not stated!) restricting the feasible range of
   `z₁ = |f̂(1)|`. The boundary value `z₁ = 0.50426` is substituted to yield `1.27481`.
   The gain `1.2743 → 1.27481` is **only 0.0005**.
3. **Slight δ, u tuning.** MO: `δ = 0.13`. MV: `δ = 0.138`.

So ~95 % of MV's improvement over MO is item (1) — a larger `n` and unconstrained
(non-Selberg) coefficients. The "moment constraint on `|f̂(1)|`" accounts for only ~0.05
of the improvement.

---

## D. The multi-frequency generalisation — already in MO 2004

### D.1 MO 2004 Lemma 2.14 gives the bound for ALL integer j

**MO 2004 Lemma 2.14 (p. 16, lines 1052–1077).** Let `f` be a pdf supported on `[−¼, ¼]`.
Then for every integer `j ≠ 0`,

    |f̂(j)|² ≤ ‖f·f‖_∞ · sin(π / ‖f·f‖_∞) / π.                        (MO 2004 eq. below 1055)

The proof (p. 16 lines 1058–1077) uses the symmetric-decreasing-rearrangement trick
(Lemma 2.13 of MO 2004, p. 15 lines 993–1035) applied to `h := (f·f)^{sdr}`; the key
step is that the maximum of the cosine integral over all pdfs of max ≤ M is achieved by
the indicator of `[−1/(2M), 1/(2M)]`.

**Consequently, MV's "open direction (iii)" (MV p. 7 lines 374–380, "extend Lemma 3.4
to `|f̂(n)|` for `n > 1`") is ALREADY SOLVED by MO 2004 Lemma 2.14, uniformly in `n`.**
The bound is *independent* of `n`:

    |f̂(2)|² ≤ M sin(π/M)/π,    |f̂(3)|² ≤ M sin(π/M)/π,    ...

For `M = 1.2748`, this is `< 0.5043²` for **every** `j ≠ 0`.

### D.2 MO 2004 Proposition 2.11 is the full multi-frequency Cauchy–Schwarz

**MO 2004 Proposition 2.11 (p. 13, lines 822–885).** Let `m ≥ 1`. Given `f` supported on
`[−¼, ¼]` and `K` with `K(x) = 1` on `[−¼, ¼]`, Σ_{j ≥ m}|K̂(j)|^{4/3} > 0. Set
`M := 1 − K̂(0) − 2 · Σ_{j=1}^{m−1} K̂(j) Re(f̂(j))`. Then

    ‖f·f‖_2² = Σ |f̂(j)|⁴ ≥ (M / ‖_m K̂‖_{4/3})⁴ + 2 Σ_{j=1}^{m−1} |f̂(j)|⁴.       (MO 2004 eq. (6))

Here `‖_m K̂‖_{4/3} := (Σ_{|j| ≥ m} |K̂(j)|^{4/3})^{3/4}`. Setting `m = 1` recovers
Proposition 2.7 (the plain C–S). Setting `m = 2` uses the information about `f̂(1)`.

**MO 2004 p. 14 lines 922–931 observes that *by itself* Prop 2.11 gives NO improvement
over the `m = 1` bound** — the minimum over choices of `f̂(1)` is achieved at the `m = 1`
value. The improvement *only* materialises when you **combine Prop 2.11 with an
independent upper bound on `f̂(j)`**. MO 2004 Corollary 2.12 is the example at `m = 2`:
it combines the kernel `K₆` with constraint `|f̂(1)| = x₁`, to get
`‖f·f‖_2² ≥ 1 + 2x₁⁴ + (1.53890 − 2.26425 x₁)⁴`.

### D.3 The higher-frequency moment constraint relating f̂(1), f̂(2)

**MO 2004 Lemma 2.17 (p. 22, lines 1438–1469).** For any pdf `f` supported on `[−¼, ¼]`,

    2 |f̂(1)|² − 1 ≤ Re f̂(2) ≤ 2 Re f̂(1) − 1.                              (MO 2004 Lemma 2.17)

**Proof idea (p. 22 lines 1443–1469).** Use trigonometric polynomials `L_b(x) = b cos(2πx)
− cos(4πx)` and `2 cos(2πx) − cos(4πx)` which dominate `0` and `1` respectively on `[−¼,¼]`,
then apply Parseval. This is the "Selberg trick" restricted to degree 2 — and it's
completely untouched by MV.

**Combining Lemmas 2.14 and 2.17** gives MO 2004 (p. 22 lines 1471–1479):
`max{|f̂(1)|, |f̂(2)|} ≥ 1/3`, whence `1/9 ≤ ‖f·f‖_∞ · sin(π/‖f·f‖_∞)/π`, yielding the
**crude** bound `‖f·f‖_∞ ≥ 1.11` (MO 2004 p. 22 line 1479).

### D.4 Clean generalisations of the MV `M sin(π/M)/π` bound

**Uniform bound (MO 2004 Lemma 2.14):**

    For all integer j ≠ 0,  |f̂(j)|² ≤ (M/π) sin(π/M),   M = ‖f·f‖_∞.     (NO j-dependence)

**Sharper j-dependent bounds: NOT IN MO or MV.** A sharper symmetric-decreasing-rearrangement
bound would be

    |f̂(j)|² ≤ M · ∫_{-1/(2M)}^{1/(2M)} cos(2π j x) · M dx = (M/(π j)) sin(π j/M),

but this is *weaker* than `(M/π) sin(π/M)` for `j > 1` (the integrand is oscillatory, not
monotone in `j`), and in fact *can be negative*, so one must take absolute values:
`|f̂(j)|² ≤ |(M/(π j)) sin(π j/M)|`. For `M = 1.276`, `π j/M > π` already at `j = 2`
(`π · 2/1.276 ≈ 4.925`), so `sin(π · 2/M)` is in the second oscillation regime and the
bound can be tight or vacuous depending on the quadrant. The **actually useful** inequality
is the MO 2004 Lemma 2.14 bound (independent of `j`) combined with the **MO 2004 Lemma 2.17**
chain `f̂(2) ≤ 2 Re f̂(1) − 1`, which is algebraic and gives nontrivial constraints.

**Proposed clean statement (NOT in either paper — CONJECTURAL).** For `f ≥ 0`, `‖f‖_1 = 1`,
`supp f ⊂ [−¼, ¼]`, `M = ‖f·f‖_∞`, and any integer `n ≥ 1`:

    |f̂(n)|² ≤ (M/π) · |sin(π n / M)| · (corrections for n ≥ 2).

The only **rigorous** uniform-in-`n` version is MO 2004 Lemma 2.14. MV's `n = 1`
specialisation is `|f̂(1)|² ≤ (M/π) sin(π/M)` because `π/M < π/2`, avoiding the
branching issue.

---

## E. Is the `≈ 1.276` MV "ceiling" rigorous?

**Short answer: NO, not as a formal theorem.** MV's `1.276` (MV p. 5 lines 232–261)
is only a heuristic derived from a *numerically solved discrete convex QP*, not a
proof. MV **explicitly say** (p. 5 line 258):

  > "all this could be done rigorously, but one needs to control the error arising from
  > the discretization, and the sheer documentation of it is simply not worth the
  > effort, in view of the minimal gain."

**Three caveats for the MV framework:**

1. MV's `1.276` is the sup of their argument for **fixed K = K_β** (the arcsine-autocorrelation
   kernel). If you change K (which MO 2004 §2.7 notes is an open search problem), you
   could in principle exceed `1.276`. MO 2004's Corollary 2.8 shows that `0.5747/δ` is
   nearly optimal (lower limit `0.5745`), so the gain from changing `K` is at most
   `≈ 0.0002/δ` at the master-inequality level — tiny.

2. The *Hölder slack* (Conjecture 2.9 of MO 2004): if `‖g·g‖_2² ≤ (π/log 16)‖g·g‖_∞`
   with `π/log 16 ≈ 2.266`, then the MV framework's `≈ 1.276` ceiling would rise to
   roughly `1.367`. **This is the single biggest potential gain in the framework, and
   it is untouched by MV.**

3. The *full Cauchy–Schwarz* (MO 2004 Proposition 2.11 with `m ≥ 3`): MV only pull out
   `|f̂(1)|`. Pulling out `|f̂(1)|` and `|f̂(2)|` simultaneously, subject to the Lemma
   2.17 linear constraint `f̂(2) ≤ 2 f̂(1) − 1` and both `|f̂(j)|² ≤ M sin(π/M)/π`,
   is an orthogonal tightening that MV do not attempt. **This is the cleanest
   publishable improvement of MV's 1.27481.**

**CS 2017's 1.28 is NOT attainable by MV's framework as formulated**, because (a) the
`1.276` heuristic ceiling is 0.004 short, and (b) no one has proved Conjecture 2.9.
CS 2017 uses a *completely different* (primal branch-and-prune) method, which is why
it succeeds where MV does not.

---

## F. Summary of MO's untapped slack (potential for lower-bound gains)

| Source              | Potential new lower bound | Status         |
| ------------------- | ------------------------- | -------------- |
| MO 2004 Conj. 2.9 (Hölder tightening) | ~1.367 | Open; conjectural |
| MO 2004 Prop. 2.11, m=3 (pull out f̂(2)) | 1.2748 (proven, **below** CS 2017) | Closed — see [proof/multifreq_mo217_proof.md](../proof/multifreq_mo217_proof.md) |
| MO 2004 Beckner-HY sharpening         | Unknown | Untested |
| Change of kernel K (away from β⋆β)    | ≤ 0.0002/δ extra | Negligible gain |
| MV `|f̂(1)|` moment (current use)      | 1.27481 | Achieved |
| Theoretical MV ceiling (fixed K_β)    | ~1.276 | Heuristic only |
| CS 2017 (different method)            | 1.28 | Proved |

**This direction is now closed.** The combination "MO 2004 Prop 2.11 at
m=3 + Lemma 2.17 + Lemma 2.14 + MV master inequality" was carried out
in [`proof/multifreq_mo217_proof.md`](../proof/multifreq_mo217_proof.md).
The optimum of the lifted QP is `M* = 1.2748376` — *strictly below* the
Cloninger–Steinerberger value `1.2802`. Theorems 1 and 2 of that file
prove the structural reason the lift cannot exceed `1.2802` even with
arbitrary tightenings: the active L2.17 constraint is met with equality
at the optimum, leaving no slack for further per-frequency moment
tightening within the MV dual ansatz. See the proof file for the full
argument and a diagnostic certificate at mpmath dps=80.

In short: the row **"MO 2004 Prop. 2.11, m=3 (pull out f̂(2))"** in the
table above should be read as "Closed — gives 1.2748, structurally
caps below 1.2802," not "Achievable but not yet done."

---

## G. Explicit citations for the reader returning to the PDFs

- `0.5747/δ` constant: MO 2009 p. 6 lines 373–375 (eq. 2.2 context), and the elliptic-integral
  formula on p. 6 lines 377–381.
- Lemma 3.2 (Cauchy–Schwarz upper bound): MO 2009 p. 10 lines 609–664.
- Lemma 3.3 (Parseval + Fourier decomposition): MO 2009 p. 11 lines 671–722.
- Lemma 3.4 (C–S dual for lower bound): MO 2009 p. 12–13 lines 750–824.
- `|f̂(j)|² ≤ M sin(π/M)/π` for ALL j: MO 2004 Lemma 2.14, p. 16 lines 1052–1077.
- Multi-frequency Prop 2.11: MO 2004 p. 13 lines 822–885.
- Lemma 2.17 (`f̂(2)` in terms of `f̂(1)`): MO 2004 p. 22 lines 1438–1469.
- Conjecture 2.9 (Hölder-slack conjecture): MO 2004 p. 12 lines 780–791.
- MV heuristic ceiling ~ 1.276: MV p. 5 lines 232–261.
- MV admission it is not rigorous: MV p. 5 line 258.
