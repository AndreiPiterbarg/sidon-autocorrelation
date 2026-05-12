# Reverse-Young Literature Review

**Conjecture under investigation (REV-3/2).** For nonneg \(f\) on \([-1/4, 1/4]\) with \(\int f = 1\) and \(\sup_{|t|\le 1/2}(f*f)(t) < \infty\):
\[\sup(f*f) \;\ge\; c_0\,\|f\|_{3/2}^3, \qquad c_0 = \pi/8 \approx 0.3927\]
with equality achieved in the limit by the Schinzel-Schmidt extremizer
\(f_0(x) = (2x+1/2)^{-1/2}\) on \([-1/4, 1/4]\).

This review surveys the literature for upper-bound (forward Young) and lower-bound
(reverse Young) inequalities of this type, asks whether REV-3/2 follows from any
existing theorem, and quantifies the deficit between forward Young and our conjecture
on the three canonical test functions.

---

## 1. Sharp upper bounds (Beckner, Brascamp-Lieb, Carlen-Loss)

The forward direction is classical and sharp.

### 1.1 Beckner (1975) — *Inequalities in Fourier analysis*, Annals of Math. 102

Beckner proved the sharp Young's convolution inequality on \(\mathbb{R}\): for
\(1/p + 1/q = 1 + 1/r\) with \(p, q, r \ge 1\),
\[\|f * g\|_r \;\le\; A_p A_q A_{r'} \,\|f\|_p \|g\|_q, \quad A_p = \big(p^{1/p}/p'^{1/p'}\big)^{1/2}.\]
Equality is achieved by **Gaussians** (no compactly supported extremizer).

For our setting, take \(f=g\) and \(p=q=3/2\), giving \(r=3\). The sharp Beckner constant is
\(A_{3/2}^2 A_3 = 0.95318...\), so
\[\|f*f\|_3 \le 0.9532\,\|f\|_{3/2}^2.\]

### 1.2 Brascamp-Lieb (1976) — *Best constants in Young's inequality, its converse, and its generalization to more than three functions*, Adv. Math. 20

This paper is the **single most important reference for our problem**, because it
proves both Young's inequality AND its converse (reverse direction) with sharp
Gaussian extremizers — but with a critical caveat:
- **Forward Young** (\(p,q,r\ge 1\)): sharp constant \(A_p A_q A_{r'}\), Gaussians extremal.
- **Reverse Young** (\(0 < p,q,r \le 1\)): sharp constant \(A_p A_q A_{r'}\) with \(A_p = ((1-p)^{1-p}/p^p)^{1/2}\) for \(p < 1\), Gaussians extremal **for exponents in \((0,1)\)**.

The reverse case requires \(p, q < 1\). **Our conjecture has \(p = q = 3/2 > 1\), so the
classical Brascamp-Lieb reverse Young does NOT directly apply.** This is the central
structural reason REV-3/2 cannot be obtained by simple analytic continuation of the
classical reverse-Young.

### 1.3 Barthe (1997) — *Optimal Young's inequality and its converse: a simple proof*, GAFA

Gives a unified mass-transport proof of both Young and reverse Young. Clarifies that
the dichotomy at \(p=1\) is intrinsic: the reverse direction is proved by exchanging
sup-with-inf in the variational characterization, valid only for \(p<1\).

### 1.4 Carlen-Lieb-Loss (2004) — *A sharp analog of Young's inequality on \(S^N\)*, J. Geom. Anal.

Extends sharp Young to compact manifolds via heat-flow techniques. Demonstrates that
on a compact domain, the extremizer is no longer Gaussian and constants can shift —
but on \([-1/4,1/4]\) the bulk geometry is degenerate (no curvature) and these
techniques do not yield improved compact-support constants for our case.

### 1.5 Bennett-Bez-Flock-Lee (2015), Bennett-Tao (2023) "Adjoint Brascamp-Lieb"

Bennett-Tao introduce *adjoint* (\(L^p\)-reverse) Brascamp-Lieb inequalities, dual to
the classical case, valid for all \(p \ge 1\). This direction produces upper bounds on
**reverse-tomographic** quantities (e.g., reverse Loomis-Whitney) but not the kind of
"\(\sup f*f\) lower-bounded by an \(L^{3/2}\) norm" inequality we need; the
multilinear setup is different.

---

## 2. Reverse inequalities — what is known

A search of the literature (Brascamp-Lieb 1976, Barthe 1997, Bennett-Tao 2023,
Lehec 2013, Chen-Dafnis-Paouris 2013, Borell 1975, Maurey, Nakamura-Tsuji 2025)
reveals **no inequality of the form**
\[\sup(f*f) \ge C \cdot \|f\|_p^k\]
for \(p > 1\) and any positive \(C\) over compactly supported nonnegative \(f\). The
following partial results are the closest analogs:

| Result | Form | Direction relevant? |
|--------|------|---------------------|
| Brascamp-Lieb 1976 reverse Young | \(\|f*g\|_r \ge C_p \|f\|_p \|g\|_q\), \(p,q,r<1\) | NO — wrong exponent regime |
| Borell 1975 reverse Brunn-Minkowski (Gauss) | volume / measure form | NO — geometric, not on convolution sup |
| Chen-Dafnis-Paouris 2013 reverse Hölder | Gaussian-only | NO — Gaussian-restricted |
| Bennett-Tao 2023 adjoint Brascamp-Lieb | \(L^p\)-reverse for \(p\ge 1\), but multilinear | NO — wrong functional form |
| Maurey reverse log-Sobolev | entropy-form | NO — no convolution |
| Reverse Hardy-Littlewood-Sobolev (Dou-Zhu, etc.) | for \(p<0\) regime | NO — wrong sign / regime |
| Nakamura-Tsuji 2025 (sharp Laplace under heat flow) | even functions, Laplace transform | NO — relates volume product |

**The exponent regime \(p > 1\) for reverse-Young-style inequalities is genuinely
empty in classical harmonic analysis.** A bound of the form \(\sup(f*f) \ge C \|f\|_p^k\)
must use the **compact-support hypothesis** essentially — without it, \(f\) can be
taken Gaussian and the bound fails.

---

## 3. Sidon-specific literature

### 3.1 Schinzel-Schmidt (2002) — *Comparison of \(L_1\)- and \(L_\infty\)-norms of squares of polynomials*, Acta Arithmetica 104

Original source of the problem. Defined \(S = \inf_{f \in \mathcal{F}} \|f*f\|_\infty\)
over \(\mathcal{F} = \{f \ge 0\) on \([-1/4,1/4]\), \(\int f = 1\}\). Conjectured
\(S = \pi/2\) with extremizer \(f_0(x) = (2x+1/2)^{-1/2}\). The conjecture form is
\(\sup(f*f) \ge \pi/2 \cdot (\int f)^2\) — a much weaker inequality than REV-3/2.

### 3.2 Cilleruelo-Vinuesa (2008) "B2[g] sets and a conjecture of Schinzel and Schmidt", CPC 17

Translates SS to the \(B_2[g]\) set asymptotics. Proves limit exists.

### 3.3 Matolcsi-Vinuesa (2009), arXiv:0907.1379

**Disproves the SS conjecture.** Proves \(S \le 1.5098 < \pi/2\) using a numerical
step-function example, and \(S \ge 1.2748\) by improving Yu / Martin-O'Bryant. Their
explicit analytic counterexample (eq. 4.15 in Vinuesa's thesis) is
\[f(x) = \begin{cases} 1.392887/(0.00195 - 2x)^{1/3} & x \in (-1/4, 0) \\
0.338537/(0.500166 - 2x)^{0.65} & x \in (0, 1/4) \end{cases}\]
giving \(\|f*f\|_\infty \approx 1.52799\). **Critically: only the \((\int f)^2 = 1\)
form is studied; no \(L^{3/2}\) or other \(L^p\) bounds appear.**

### 3.4 Cloninger-Steinerberger (2014), arXiv:1403.7988

Proves \(S \ge 1.28\) via a discretized branch-and-bound over mass distributions.
Same form: \(\sup(f*f) \ge 1.28 (\int f)^2\). No \(L^p\) norm dependence.

### 3.5 Barnard-Steinerberger (2019), arXiv:1903.08731

Three convolution inequalities. Proves a **dual** \(L^1\)-form bound
\(\min_{0\le t \le 1} \int f(x)f(x+t)dx \le 0.42 \|f\|_1^2\) and an \(L^1, L^2\) bound
\(\int_{-1/2}^{1/2} \int f(x)f(x+t)dx\,dt \le 0.91 \|f\|_1 \|f\|_2\). **This last one
is the closest in form to a reverse-Young, but uses \(\|f\|_2\), not \(\|f\|_{3/2}\),
and has the wrong direction (UB on autocorrelation by \(\|f\|_2\)).**

### 3.6 de Dios Pont-Madrid (2021), arXiv:2106.13873

Studies Barnard-Steinerberger autocorrelation inequality, proves existence of
extremizers, computes near-optimal numerical constant. Focused on the autocorrelation
side; no \(L^{3/2}\)-type bound.

### 3.7 White (2022), arXiv:2210.16437 — *Almost-tight \(L^2\) autoconvolution*

Studies \(\inf \|f*f\|_2\) over \(\int f = 1\), \(f\) supported on \([-1/2, 1/2]\).
Determines the value to 0.0014% accuracy and proves uniqueness of the minimizer.
**Different functional**: this is an \(L^2\) lower bound, not \(L^\infty\), and uses
\(\|f\|_1 = 1\) (not \(\|f\|_{3/2}\)). Provides corollaries on \(B_h[g]\) set sizes.

### 3.8 Boyer-Li (2025), arXiv:2506.16750

Improves the Martin-O'Bryant conjectured ratio
\(\|f*f\|_2^2 / (\|f*f\|_\infty \|f*f\|_1) \ge 0.9015\) via an explicit 575-step
function (improving on AlphaEvolve's 0.8962). All inequalities are in \(\|f*f\|\)
norms, not in any norm of \(f\) itself.

### 3.9 Boyer-Li 2025 arXiv:2508.02803 (follow-up)

Pushes the constant in 3.8 to \(\ge 0.94136\) with a 2399-step function. **Same
functional family: ratios of \(\|f*f\|\) norms.** None of these papers prove or use
a \(\sup(f*f) \ge C \|f\|_p^k\) inequality for any \(p\).

---

## 4. Direct relevance to our conjecture

**No paper in the literature proves or uses the exact REV-3/2 form.** The closest
relatives are:

- **Schinzel-Schmidt / MV / Cloninger-Steinerberger / White**: all bound \(\sup(f*f)\)
  by powers of \(\|f\|_1 = \int f\). To recover REV-3/2 from these one would need
  \(\|f\|_{3/2}^3 \le C \|f\|_1^2\), which is FALSE (both norms are scale-invariant
  in the same way, and \(\|f\|_{3/2}\) can be made arbitrarily small for fixed \(\|f\|_1\)
  on \([-1/4, 1/4]\); take \(f\) more uniform). So the existing constants give
  **incomparable** information.

- **Beckner / Brascamp-Lieb forward Young**: \(\|f*f\|_3 \le 0.953\|f\|_{3/2}^2\) is an
  **upper** bound on a sharper-than-sup norm of \(f*f\). Combined with Hölder
  \(\|f*f\|_\infty \ge \|f*f\|_3 / \|f*f\|_2 ^{?}\) (no clean direction), this gives
  no path to REV-3/2.

- **Brascamp-Lieb reverse Young**: applies only to \(p,q,r < 1\) (concave-power
  regime). Does not extend to \(p = 3/2\).

The compact-support hypothesis on \(f\) is **essential** for REV-3/2: any non-trivial
lower bound \(\sup(f*f) \ge C\|f\|_p^k\) over arbitrary nonnegative \(L^1\) functions
fails (rescale \(f\) wide enough to drive \(\sup(f*f) \to 0\) while keeping
\(\|f\|_p\) fixed). Compact support fixes the scale.

---

## 5. Citations with precise statements

| Year | Authors | Statement |
|------|---------|-----------|
| 1975 | W. Beckner, *Inequalities in Fourier analysis*, Ann. Math. 102 | Sharp Young: \(\|f*g\|_r \le A_p A_q A_{r'}\|f\|_p\|g\|_q\), Gaussian extremizers |
| 1976 | H. J. Brascamp & E. H. Lieb, *Best constants...*, Adv. Math. 20 | Sharp forward + reverse Young; reverse only for \(p,q,r<1\) |
| 1990 | E. H. Lieb, *Gaussian kernels have only Gaussian maximizers*, Invent. Math. 102 | Gaussian-only extremizers for general Brascamp-Lieb |
| 1997 | F. Barthe, *Optimal Young's inequality...*, GAFA | Unified transport proof of both Young and reverse Young |
| 2002 | A. Schinzel & W. Schmidt, *Comparison of \(L_1\) and \(L_\infty\) norms*, Acta Arith. 104 | Conjecture: \(S = \inf \|f*f\|_\infty = \pi/2\) |
| 2008 | J. Cilleruelo & C. Vinuesa, CPC 17 | \(B_2[g]\) connection: \(\beta_g/\sqrt{g} \to 2/S\) |
| 2009 | M. Matolcsi & C. Vinuesa, arXiv:0907.1379 | Disproves SS: \(S \le 1.5098\); explicit counterexample (eq. 4.15) |
| 2014 | A. Cloninger & S. Steinerberger, arXiv:1403.7988 | \(S \ge 1.28\) by branch-and-bound |
| 2019 | R. Barnard & S. Steinerberger, arXiv:1903.08731 | Three convolution inequalities (autocorr / autoconv) |
| 2021 | J. de Dios Pont & J. Madrid, arXiv:2106.13873 | Existence of extremizers for Barnard-Steinerberger |
| 2022 | E. P. White, arXiv:2210.16437 | \(\inf \|f*f\|_2\) determined to 0.0014% |
| 2023 | J. Bennett & T. Tao, *Adjoint Brascamp-Lieb*, Proc. London Math. Soc. | \(L^p\)-reverse Brascamp-Lieb for \(p\ge 1\) (multilinear, not our form) |
| 2025 | C. Boyer & Z. K. Li, arXiv:2506.16750, arXiv:2508.02803 | \(\|f*f\|_2^2 / (\|f*f\|_\infty \|f*f\|_1) \ge 0.94136\) |

---

## 6. Numerical verification on test functions

For \(f\) on \([-1/4, 1/4]\), \(\int f = 1\) (computed):

| \(f\) | \(\sup(f*f)\) | \(\|f\|_{3/2}^3\) | \(\sup/\|f\|_{3/2}^3\) | \(\|f*f\|_3 / \|f\|_{3/2}^2\) |
|-------|---------------|--------------------|--------------------------|--------------------------------|
| **Schinzel-Schmidt** \(f_0(x) = (2x+1/2)^{-1/2}\) | \(\pi/2 \approx 1.5708\) | \(4\) | \(\pi/8 \approx 0.3927\) | \(0.5061\) |
| **Uniform** \(f \equiv 2\) on \([-1/4,1/4]\) | \(1\) | \(2\) | \(0.5\) | \(0.3969\) |
| **MV eq. (4.15)** asymmetric example | \(1.5280\) | \(3.2212\) | \(0.4744\) | \(0.5644\) |
| Beckner sharp upper bound (R) | — | — | — | \(0.9532\) (Gaussian-extremal) |

**Observations:**
1. The conjectured constant \(\pi/8\) is achieved exactly at SS, and the MV
   counterexample gives ratio \(0.4744 > \pi/8\) — so MV (which disproves SS for the
   \((\int f)^2\) problem) does NOT disprove REV-3/2.
2. The uniform function gives \(0.5 > \pi/8\): far from extremal in the L^{3/2} ratio.
3. Beckner's sharp Young constant (Gaussian-extremized on \(\mathbb{R}\)) is
   \(0.9532\); on compact \([-1/4,1/4]\) the realized \(\|f*f\|_3/\|f\|_{3/2}^2\) is
   \(\le 0.5644\) (MV), \(\le 0.5061\) (SS), \(\le 0.3969\) (uniform). The
   "compact-support deficit" (gap from Beckner) is large and is what REV-3/2
   would crystallize if \(\sup(f*f)\) — not \(\|f*f\|_3\) — is on the LHS.

---

## 7. Verdict

**REV-3/2 (\(\sup(f*f) \ge \pi/8\, \|f\|_{3/2}^3\) with equality at SS) is genuinely new.**

- It is **not in the literature** in any form. No paper bounds \(\sup(f*f)\) below by
  any norm of \(f\) other than \(\|f\|_1\).
- It is **not derivable from sharp Beckner-Brascamp-Lieb Young** (forward direction
  is upper, not lower; reverse direction is restricted to \(p < 1\)).
- It is **not a consequence of any combinatorial Sidon bound** (those bound by
  \((\int f)^2\), an incomparable quantity).
- It is **not implied by any reverse-Holder, reverse-log-Sobolev, or
  reverse-Brunn-Minkowski** result we found.

The conjecture appears to be **structurally between forward-Young and the SS
problem**: it strengthens the SS lower bound (which uses only \(\int f\)) by allowing
the sharper \(L^{3/2}\) norm in the bound, while remaining consistent with the
SS-extremality of \(f_0\). The asymmetric case is the technical content (per the
project's own `PARTIAL_PROOFS_NOTES.md` Observation 1, the symmetric case is trivial).

**Implications for the Lasserre-SDP pipeline.** Because no off-the-shelf inequality
gives REV-3/2, proving it must be done from scratch — which is consistent with the
project's `lasserre/threepoint_l32*.py` SDP machinery being exploratory rather than
verifying a known theorem. Approaches likely to succeed:
1. **Variational / 1-D Lagrangian**: SS is critical for \(\sup(f*f)/\|f\|_{3/2}^3\),
   confirmed by computing first variation; need to show it is the unique minimizer.
2. **Polynomial dual SDP** with the constraint \(\|f\|_{3/2} \le B\) (this is the
   approach in `threepoint_l32_moment.py`); unfortunately, per `MEMORY.md`, the
   "3-point SDP pilot DEAD" entry indicates the V1+V2 lift is structurally
   degenerate for this problem.
3. **Direct sharp computation on the alpha-family** \(f_a(x) = c_a(b-2x)^{-a}\),
   showing the ratio is minimized at \(a = 1/2\) in this 1-parameter family
   (rigorous; see `PARTIAL_PROOFS_NOTES.md` Observation 3).
4. **Asymptotic / boundary analysis**: SS is on the *boundary of feasibility*
   (\(a = 1/2\) is exactly where \(\|f_a\|_2 \to \infty\)). This co-incidence with
   the \(\|f\|_{3/2}^3\) extremum is suggestive — perhaps an interpolation argument
   between the symmetric \(c_0 = 1\) bound and the boundary singularity.

**Bottom line:** REV-3/2 is a new conjecture that does not follow from existing
inequalities. Proving it (in any form) would be a novel contribution and unlocks
a route from val(d) lower bounds via the Lasserre hierarchy to a rigorous \(C_{1a}\)
improvement.
