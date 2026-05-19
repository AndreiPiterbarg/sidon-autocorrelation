# An autocorrelation constant related to Sidon sets — verification of the lower bound $C_{1a} \ge 1.292$

## Description of constant

$C_{1a}$ is the largest constant for which one has
$$
\max_{-1/2 \leq t \leq 1/2} \int_{\mathbb{R}} f(t-x) f(x)\,dx \;\geq\; C_{1a} \left(\int_{-1/4}^{1/4} f(x)\,dx\right)^{2}
$$
for all non-negative $f \colon \mathbb{R} \to \mathbb{R}$.
<a href="#1a-def">[1a-def]</a>

Equivalently, with $\mathcal{F}$ the family of non-negative $f \in L^{1}(\mathbb{R})$ supported in $(-1/4,1/4)$ with positive integral, $C_{1a} = \inf_{f \in \mathcal{F}} R(f)$, where $R(f) = \lVert f*f \rVert_{\infty} / (\int f)^{2}$. Throughout, $M$ denotes $R(f)$; a lower bound asserts $M \ge c$ for every $f \in \mathcal{F}$.
<a href="#PBV-def">[PBV-def]</a>

## Known upper bounds

Not affected by this work; the canonical page tracks them. Current best: $\approx 1.502862$ (AI-assisted, 2026). The bound proved here therefore leaves the gap $1.292 \le C_{1a} \le 1.502862$.
<a href="#1a-page">[1a-page]</a>

## Known lower bounds

Rows one to six are reproduced from the canonical page
<a href="#1a-lb">[1a-lb]</a>; the last row is established here.

| Bound | Reference | Comments |
| ----- | --------- | -------- |
| $1$ | Trivial | $f*f$ has integral $(\int f)^2$ over an interval of length $1$, so its maximum is at least $(\int f)^2$. |
| $1.182778$ | [[MO2004](#MO2004)] | |
| $1.262$ | [[MO2009](#MO2009)] | |
| $1.2748$ | [[MV2009](#MV2009)] | Best prior rigorous analytic bound; stated as $1.27481$ in the preprint. |
| $1.28$ | [[CS2017](#CS2017)] | Discrete branch-and-prune search. |
| $1.2802$ | [[XX2026](#XX2026)] | Unpublished, AI-assisted (Grok), unaudited. The canonical page attributes this to Xie (2026), *not* to Cloninger–Steinerberger. |
| $1.292$ | [[PBV2026](#PBV2026)] | This work. Rigorous; the Fourier-analytic identities invoked are MV 2010 Lemma 3.3 and MO 2009 Lemmas 2.1, 2.2, 3.2, 3.3, applied to a three-scale arcsine kernel. Machine-checked in Lean modulo those cited identities and two numerical inputs computed in `flint.arb`. |

## How the bound is established

**The dual framework.** A lower bound must hold for every admissible
$f$ simultaneously, so it cannot come from a single example. Following
Matolcsi–Vinuesa, one fixes an auxiliary pair: a *Bochner-admissible*
kernel $K$ — non-negative, even, with
$\mathrm{supp}(K) \subseteq [-\delta,\delta]$, $\int K = 1$, and
non-negative periodic Fourier coefficients
$\widetilde{K}(j) := \widehat{K}(j/u) \ge 0$ — and a non-negative
cosine multiplier $G$. Because $f$ is supported in $(-1/4,1/4)$, $f*f$
is supported in $(-1/2,1/2)$; taking period $u = 1/2+\delta$ converts
the constraint into a trigonometric inequality, and the pair $(K,G)$
produces, by duality, a single quadratic inequality satisfied by every
admissible $f$. Any valid pair yields some bound; a well-chosen pair
yields a strong one <a href="#PBV-master">[PBV-master]</a>.

**The $z_1$-free master inequality.** The sharp Matolcsi–Vinuesa
inequality also involves $z_1 = \lvert\widehat{f}(1)\rvert$, a Fourier
coefficient of the *unknown* extremiser, which cannot be controlled
analytically. A Cauchy–Schwarz step absorbs it, leaving
$$
M + 1 + \sqrt{(M-1)(K_2-1)} \;\ge\; \frac{2}{u} + a ,
$$
in which only the kernel/multiplier quantities
$K_2 = \lVert K\rVert_2^{2}$ and $a$ appear
<a href="#PBV-master">[PBV-master]</a>. The derivation invokes three
Fourier-analytic identities (period-$u$ Parseval on the torus, the
constant-plus-tail split for $\int (f\circ f)\,K$, and the lattice
$F$-bound $\sum_j \lvert\widehat{f}(j)\rvert^{4} \le \lVert f*f\rVert_{\infty}$),
all of which are statements or proof steps in MV 2010 and MO 2009 —
see <a href="#MV-primitives">[MV-primitives]</a> and
<a href="#MO-primitives">[MO-primitives]</a>. The present work uses
them verbatim, applied to the three-scale kernel below; no new
analytic content is introduced.

**The three-scale kernel.** Each arcsine kernel
$K_{\mathrm{arc}}(\delta;\cdot)$ is the autoconvolution of the arcsine
density of half-width $\delta$; its Fourier transform is
$J_0(\pi\delta\xi)^{2}\ge 0$, so any convex combination is
automatically Bochner-admissible. Matolcsi–Vinuesa used a single
arcsine kernel. Here $K = K_{\mathrm{ms}}$ is a convex combination of
**three**, with half-widths $(\delta_1,\delta_2,\delta_3) =
(138,55,25)/1000$, weights $(\lambda_1,\lambda_2,\lambda_3) =
(85,10,5)/100$, and period $u = 1/2+\delta_1 = 638/1000$; its periodic
coefficients are $\widetilde{K_{\mathrm{ms}}}(j) = \sum_i \lambda_i
J_0(\pi j\delta_i/u)^{2} \ge 0$. The multiplier $G$ is re-optimised
for this kernel as a $200$-term cosine sum
<a href="#PBV-kernel">[PBV-kernel]</a>. These parameters were chosen
by numerical search; the bound below is valid regardless of whether
they are globally optimal.

**The five certified functionals.** Each quantity below is enclosed
in `flint.arb` interval arithmetic at 256-bit precision and rounded
*outward* to an exact rational
<a href="#PBV-anchors">[PBV-anchors]</a>:

| Functional | Certified bound | Decimal |
| ---------- | --------------- | ------- |
| $k_1 = \widehat{K_{\mathrm{ms}}}(1)$ (kernel mass moment) | $\ge 9212/10000$ | $0.9212$ |
| $K_2 = \lVert K_{\mathrm{ms}}\rVert_2^{2}$ (kernel energy) | $\le 47897/10000$ | $4.7897$ |
| $S_1 = \sum_{j=1}^{200} a_j^{2}/\widetilde{K_{\mathrm{ms}}}(j)$ (multiplier denominator) | $\le 29841/1000$ | $29.841$ |
| $m_G = \min_{[0,1/4]} G$ (multiplier minimum) | $\ge 998/1000$ | $0.998$ |
| $a = (4/u)\,m_G^{2}/S_1$ (gain) | $\ge 20925/100000$ | $0.20925$ |

The $z_1$-free inequality consumes only $K_2$ and $a$; through
$a = (4/u)\,m_G^{2}/S_1$ it depends on $m_G$ and $S_1$. The remaining
quantity $k_1$ does not enter the rational headline — it is used only
by the sharper *refined* inequality that the Arb cell-search employs
to certify the tighter $M_{\mathrm{cert}}\approx 1.29232$
<a href="#PBV-fail">[PBV-fail]</a>.

**How each anchor is certified.** $K_2$ is split into a bulk integral
on $[0,T]$ with $T = 10^{5}$, computed by `flint.arb` adaptive
Gauss–Legendre quadrature (certified bulk
$\in [4.78882342, 4.78890519]$), and a tail past $T$ bounded
analytically by the Landau inequality
$\lvert J_0(z)\rvert^{2}\le 2/(\pi z)$, which gives tail
$\le 8.19\times 10^{-5}$ <a href="#Landau">[Landau]</a>; bulk plus
tail rounds outward to $47897/10000$. The minimum $m_G$ is certified
by partitioning $[0,1/4]$ into $32768$ closed cells and forming the
second-order Taylor enclosure of $G$ on each cell in arb interval
arithmetic; the minimum of the per-cell lower endpoints is
$\ge 0.99997987 > 998/1000$. $S_1$ and $k_1$ are evaluated as exact
rational sums in arb at radii below $10^{-70}$
<a href="#PBV-anchors">[PBV-anchors]</a>.

**Closing the inequality.** Set $\Phi(M) = M + 1 +
\sqrt{(M-1)(K_2-1)}$ and $\tau = 2/u + a$. The master inequality says
$\Phi(R(f)) \ge \tau$ for every admissible $f$. Since $\Phi$ is
continuous and strictly increasing in $M$, if $\Phi(M_0) < \tau$ at a
fixed rational $M_0$, then $R(f) > M_0$ for all $f$, so $C_{1a} \ge
M_0$ <a href="#PBV-inversion">[PBV-inversion]</a>. Take
$M_0 = 1292/1000$. From $u = 638/1000$ and $a \ge 20925/100000$,
$$
\tau \;=\; \frac{2}{u} + a \;\ge\; \frac{2000}{638} + \frac{20925}{100000} \;=\; \frac{4267003}{1276000} \;=\; 3.344046\ldots .
$$
From $K_2 \le 47897/10000$ and
$(M_0-1)(K_2-1) \le (292/1000)(37897/10000) = 11065924/10^{7}$, the
rational $105195/10^{5}$ majorises the square root because
$(105195/10^{5})^{2} = 11065988025/10^{10} \ge 11065924/10^{7}$, so
$$
\Phi(1292/1000) \;\le\; \frac{1292}{1000} + 1 + \frac{105195}{10^{5}} \;=\; \frac{66879}{20000} \;=\; 3.34395 .
$$
Hence the inequality strictly fails at $M_0$, with an exactly
rational margin
$$
\tau - \Phi(1292/1000) \;\ge\; \frac{4267003}{1276000} - \frac{66879}{20000} \;=\; \frac{307}{3190000} \;\ge\; 9.6\times10^{-5} \;>\; 0 ,
$$
yielding $C_{1a} \ge 1292/1000 = 1.292$ <a href="#PBV-fail">[PBV-fail]</a>.
The closing step is exact rational arithmetic, independent of any
floating-point computation.

**Mechanisation.** The analytic reduction — admissibility of
$K_{\mathrm{ms}}$, the master inequality, the quadratic inversion,
and the rational closing arithmetic — is mechanised in Lean 4 (15
modules, $\approx 8650$ lines, `mathlib v4.29.1`, no `sorry`). The
headline theorem `autoconvolution_ratio_ge_1292_1000` concludes
`autoconvolution_ratio f ≥ 1292/1000`, where `autoconvolution_ratio`
is the Lean definition of $R(f) = \lVert f*f\rVert_{\infty}/(\int
f)^{2}$ in `Sidon.Defs`. Its dependency closure consists of: Lean's
three logical axioms (`propext`, `Classical.choice`, `Quot.sound`);
two numerical user axioms recording the certifier's outputs for $K_2$
and $a$; and an analytic-primitives record `ExtremiserPrimitives f`
(slim variant `SchwartzAtomicResidual` for Schwartz $f$) whose fields
are Lean restatements of the cited MV/MO results
<a href="#MV-primitives">[MV-primitives]</a>
<a href="#MO-primitives">[MO-primitives]</a>. The slack-soundness
statements, the quadratic inversion, and the assembly are ordinary
Lean theorems <a href="#PBV-lean">[PBV-lean]</a>. The numerical
anchors are reproducible from a SHA-256-anchored certificate and are
cross-checked at 50 decimal digits by an independent `mpmath` script
that does not call `flint.arb` <a href="#PBV-cert">[PBV-cert]</a>.

## Scope of the claim

- **Analytic content: cited, not novel.** The headline takes an
  analytic-primitives record (`ExtremiserPrimitives f`, slim variant
  `SchwartzAtomicResidual` for Schwartz $f$) as a Lean hypothesis.
  Its fields are formal restatements of published, refereed results:
  the period-$u$ torus split (MO 2009 Lemma 2.1; MV 2010 Eq.(3) /
  Lemma 3.1(3)); the constant-plus-tail Parseval split for
  $\int(f\circ f)\,K$ (MO 2009 Lemma 3.2 proof; MV 2010 Lemma 3.3
  proof); and the lattice $F$-bound
  $\sum_j \lvert\widehat{f}(j)\rvert^{4} \le \lVert f*f\rVert_{\infty}$
  (MO 2009 Lemma 3.2 proof; MV 2010 Lemma 3.3 proof). MV 2010 itself
  invokes these by citation to MO 2009 — see
  <a href="#MV-primitives">[MV-primitives]</a> and
  <a href="#MO-primitives">[MO-primitives]</a>. No novel analytic
  assumption is introduced.
- **Numerical content: two `flint.arb` certificates.** Inside Lean,
  $K_2 \le 47897/10000$ and $a \ge 20925/100000$ are user axioms
  recording outputs of the Arb certifier; they are decidable
  inequalities, established by the trusted external computation, and
  cross-checked at 50 decimal digits by `mpmath` (structurally
  independent of `flint.arb`). They are not conjectural and are not
  stated in MV 2010.
- **Trust set.** Beyond Lean's standard axioms, the bound rests on
  trust in `flint.arb` (peer-reviewed, Johansson 2017
  <a href="#Flint">[Flint]</a>), the Python certifier driver
  `bisect_alt_kernel.py`, and the independent `mpmath` cross-check
  `audit3_mpmath.py`; the SHA-256 anchor pins the certificate
  artefact <a href="#PBV-cert">[PBV-cert]</a>. This is the standard
  computer-assisted-proof convention (Flyspeck, PFR, the
  Cohn–Elkies sphere-packing record): rigorous proof + trusted
  numerical oracle.

## Additional comments and links

- Canonical page: [`constants/1a.md`](https://teorth.github.io/optimizationproblems/constants/1a.html).
  This submission adds the $1.292$ row and changes no other row.
- Reproduce: `python -m delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel`
  (certificate); `python audit_consistency.py` (cross-surface audit);
  `cd lean && lake build`, then `lake env lean AxiomCheck.lean`
  (build and axiom inventory) <a href="#PBV-cert">[PBV-cert]</a>.

## References

- <a id="1a-page"></a>**[1a-page]** Tao, Terence (ed.). *An autocorrelation constant related to Sidon sets.* `teorth/optimizationproblems`, `constants/1a.md`. [Page](https://teorth.github.io/optimizationproblems/constants/1a.html)
	- <a id="1a-def"></a>**[1a-def]**
	  **loc:** `constants/1a.md`, "Description of constant".
	  **quote:** "$C_{1a}$ is the largest constant for which one has $\max_{-1/2 \leq t \leq 1/2} \int_{\mathbb{R}} f(t-x) f(x)\,dx \geq C_{1a} (\int_{-1/4}^{1/4} f(x)\,dx)^2$ for all non-negative $f \colon \mathbb{R} \to \mathbb{R}$."
	- <a id="1a-lb"></a>**[1a-lb]**
	  **loc:** `constants/1a.md`, "Known lower bounds" table.
	  **quote:** Rows, in order: $1$ (Trivial); $1.182778$ ([MO2004]); $1.262$ ([MO2009]); $1.2748$ ([MV2009]); $1.28$ ([CS2017]); $1.2802$ ([XX2026], "Unpublished improvement, Grok"). The page attributes $1.2802$ to [XX2026] and $1.28$ to [CS2017]; it does not attribute $1.2802$ to Cloninger–Steinerberger.

- <a id="PBV2026"></a>**[PBV2026]** Piterbarg, Andrei; Bajaj, Jai; Vincent, Derrick. *A New Lower Bound for the Supremum of Autoconvolutions.* Preprint, 2026. This repository: `lower_bound_proof.tex` / `.pdf`; Lean under `lean/Sidon/`.
	- <a id="PBV-def"></a>**[PBV-def]**
	  **loc:** `lower_bound_proof.tex`, Subsection "The Constant $C_{1a}$" (lines 279–288).
	  **quote:** "$C_{1a} = \inf_{f\in\mathcal{F}} R(f)$, $R(f) := \lVert f*f\rVert_{\infty}/(\int f)^2$", with $\mathcal{F}$ the non-negative $f\in L^1(\mathbb{R})$ such that $\mathrm{supp}(f) \subseteq (-1/4,1/4)$ and $\int f > 0$.
	- <a id="PBV-main"></a>**[PBV-main]**
	  **loc:** `lower_bound_proof.tex`, Theorem "Main result" (`thm:main`, lines 351–362).
	  **quote:** "Let $f:\mathbb{R}\to\mathbb{R}$ be nonnegative with $\mathrm{supp}(f)\subseteq(-1/4,1/4)$, $\int f>0$, and $\lVert f*f\rVert_{\infty}<\infty$. Then $\lVert f*f\rVert_{\infty}/(\int f)^2 \ge 1292/1000$. In particular $C_{1a}\ge 1292/1000=1.292$."
	- <a id="PBV-master"></a>**[PBV-master]**
	  **loc:** `lower_bound_proof.tex`, Theorem "Master inequality, $z_1$-free form" (`thm:mv-master`, lines 415–426); the $z_1$-absorption is in the proof (lines 428–460).
	  **quote:** "For $K$ Bochner-admissible at scale $\delta$, $u=1/2+\delta$, and $G$ admissible with constant $m_G$ (defining $S_1$, $a$): for every $f\in\mathcal{F}$ with $\int f>0$, $M + 1 + \sqrt{(M-1)(K_2-1)} \ge 2/u + a$." The sharp form additionally carries $k_1=\widehat{K}(1)$ and $z_1=\lvert\widehat{f}(1)\rvert$ and reduces to this form by Cauchy–Schwarz.
	- <a id="PBV-inversion"></a>**[PBV-inversion]**
	  **loc:** `lower_bound_proof.tex`, Lemma "Quadratic inversion" (`lem:inversion`, lines 462–473).
	  **quote:** "$\Phi(M) := M+1+\sqrt{(M-1)(K_2-1)}$ is continuous and strictly increasing on $[1,\infty)$ with $\Phi(1)=2$; for $\tau\ge 2$, every $f\in\mathcal{F}$ with $\Phi(R(f))\ge\tau$ satisfies $R(f)\ge M_*$, the unique solution of $\Phi(M_*)=\tau$. With $\tau=2/u+a$, $C_{1a}\ge M_*$."
	- <a id="PBV-kernel"></a>**[PBV-kernel]**
	  **loc:** `lower_bound_proof.tex`, Definition "Three-scale kernel" (`def:Kms`, lines 526–546) and Theorem "Admissibility" (`thm:admissibility`, lines 567–579).
	  **quote:** $(\delta_1,\delta_2,\delta_3)=(138,55,25)/1000$, $(\lambda_1,\lambda_2,\lambda_3)=(85,10,5)/100$, $u=1/2+\delta_1=638/1000$; $K_{\mathrm{ms}}=\sum_i \lambda_i K_{\mathrm{arc}}(\delta_i;\cdot)$ is Bochner-admissible with $\widetilde{K_{\mathrm{ms}}}(j)=\sum_i \lambda_i J_0(\pi j\delta_i/u)^2 \ge 0$; multiplier degree $N=200$.
	- <a id="PBV-anchors"></a>**[PBV-anchors]**
	  **loc:** `lower_bound_proof.tex`, Lemmas `lem:k1`, `lem:K2`, `lem:S1`, `lem:mG`, `lem:a` (lines 641–756) and table `tab:anchors`.
	  **quote:** $k_1 := \widehat{K_{\mathrm{ms}}}(1) \ge 9212/10000$; $K_2 := \lVert K_{\mathrm{ms}}\rVert_{2}^{2} \le 47897/10000$ (proof: bulk $[4.78882342,\,4.78890519]$ by arb adaptive Gauss–Legendre on $[0,10^{5}]$, tail $\le 8.19\times 10^{-5}$ by Landau); $S_1 = \sum_{j=1}^{200} a_j^{2}/\widetilde{K_{\mathrm{ms}}}(j) \le 29841/1000$; $m_G := \min_{[0,1/4]} G \ge 998/1000$ (32768-cell second-order Taylor B&B in arb); $a = (4/u)\,m_G^{2}/S_1 \ge 20925/100000$.
	- <a id="PBV-fail"></a>**[PBV-fail]**
	  **loc:** `lower_bound_proof.tex`, Proposition "Strict failure at the rational witness" (`prop:fail`, lines 819–858).
	  **quote:** "$\Phi(1292/1000) \le 66879/20000 = 3.34395 < 4267003/1276000 \le \tau$, with margin $\tau-\Phi(1292/1000) \ge 307/3190000 \ge 9.6\times10^{-5} > 0$." The Arb cell-search using the sharper refined inequality (which involves $k_1$) independently re-certifies $M_{\mathrm{cert}}\ge 1.29232$.
	- <a id="PBV-lean"></a>**[PBV-lean]**
	  **loc:** `lean/Sidon/MultiScale.lean`: axioms at lines 594 and 622; headline `autoconvolution_ratio_ge_1292_1000` at lines 1089–1098; slack-soundness theorems at lines 638–661; `autoconvolution_ratio` definition in `Sidon.Defs`. The Schwartz variant `autoconvolution_ratio_ge_1292_1000_schwartz` lives in `lean/Sidon/MultiScaleSchwartz.lean`; the slim variant `autoconvolution_ratio_ge_1292_1000_schwartz_residual` and the discharge of two of the five `SchwartzAtomic` fields live in `lean/Sidon/SchwartzAtomicDischarge.lean`. `mathlib v4.29.1`, commit `5e932f97dd25535344f80f9dd8da3aab83df0fe6`.
	  **quote:** The only kernel-specific user axioms are `K2_analytic_le_K2UpperQ` ($K_2 \le 47897/10000$) and `gain_analytic_ge_gainLowerQ` ($a \ge 20925/100000$). The general headline takes `(P : ExtremiserPrimitives f)` as a hypothesis and concludes `autoconvolution_ratio f ≥ 1292/1000`. The fields of `ExtremiserPrimitives` / `SchwartzAtomicResidual` that are not already discharged from existing infrastructure (`h_torus_split`, `h_split_eq2`, `h_F_lat`, and the supporting `S_cos_*` / `h_eq4_floor` data) are Lean restatements of MV 2010 Lemma 3.3 / MO 2009 Lemmas 2.1, 2.2, 3.2, 3.3 — see <a href="#MV-primitives">[MV-primitives]</a> and <a href="#MO-primitives">[MO-primitives]</a> for the verbatim source statements. The slack-soundness theorems are one-line `norm_num` checks.
	- <a id="PBV-cert"></a>**[PBV-cert]**
	  **loc:** `delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json`, `multiscale_arcsine_1292.json`; `audit3_mpmath.py`; `README.md`.
	  **quote:** `multiscale_arcsine_1292.json` has `sha256_of_body = 5fa9ae372b23d07f73f41d73c1740926115eb494b6ba3840551458ba8143a7c2` and `M_cert = 66167/51200` ($1.29232421875$). `reference_anchors.json` records the anchors ($k_1=0.9212465899364083$, $K_2\in[4.7888234212591545,\,4.7889051816332424]$, $S_1=29.8409064555132666$, $m_G=0.9999798743824747$, $a=0.2100921474866837$) and the kernel parameters (`deltas` $138/55/25\,/1000$, equivalently $69/500$, $11/200$, $1/40$; `lambdas` $85/10/5\,/100$, equivalently $17/20$, $1/10$, $1/20$; `u` $638/1000 = 319/500$; `n_coeffs` $200$; `prec_bits` $256$). `audit3_mpmath.py` recomputes $K_2$ and $a$ at 50 digits independently of `flint.arb`.

- <a id="MO2004"></a>**[MO2004]** Martin, Greg; O'Bryant, Kevin. *The symmetric subset problem in continuous Ramsey theory.* Exp. Math. **16** (2007), no. 2, 145–165. [arXiv:math/0410004](https://arxiv.org/abs/math/0410004)

- <a id="MO2009"></a>**[MO2009]** Martin, Greg; O'Bryant, Kevin. *The supremum of autoconvolutions, with applications to additive number theory.* Illinois J. Math. **53** (2009), no. 1, 219–235. [arXiv:0807.5121](https://arxiv.org/abs/0807.5121)
	- <a id="MO-primitives"></a>**[MO-primitives]** Verbatim source for the three Fourier-analytic primitives invoked by the present work and packaged in Lean as fields of `ExtremiserPrimitives` / `SchwartzAtomicResidual`.
	  **loc 1:** MO 2009 (arXiv:0807.5121v2), p. 5, Lemma 2.1 (period-$u$ Parseval — the *foundational* form of the torus split).
	  **quote 1:** "For $i\in\{1,2\}$, suppose that $g_i$ is a square-integrable function supported on $(-\alpha_i,\alpha_i)$. If $\alpha_1+\alpha_2\le u$, then $\int_{\mathbb{R}} g_1(x)\overline{g_2(x)}\,dx = u\sum_{r\in\mathbb{Z}} \widetilde{g_1}(r)\overline{\widetilde{g_2}(r)}.$"
	  **loc 2:** MO 2009, p. 6, Lemma 2.2 (1-periodic Parseval — used in the constant-plus-tail split).
	  **quote 2:** "If $g_1$ and $g_2$ are square-integrable functions supported on $(-\tfrac12,\tfrac12)$, then $\int_{\mathbb{R}} g_1(x)\overline{g_2(x)}\,dx = \sum_{r\in\mathbb{Z}} \widehat{g_1}(r)\overline{\widehat{g_2}(r)}$; in particular $\lVert g_1\rVert_2^2 = \sum_{r\in\mathbb{Z}}\lvert \widehat{g_1}(r)\rvert^2$."
	  **loc 3:** MO 2009, p. 10, proof of Lemma 3.2 (contains the constant-plus-tail split and the lattice $F$-bound *verbatim*).
	  **quote 3:** "$\int_{\mathbb{R}}(f\circ f(x))K(x)\,dx = \sum_{r\in\mathbb{Z}} \widehat{f\circ f}(r)\overline{\widehat{K}(r)} = \sum_{r\in\mathbb{Z}}\lvert\widehat{f}(r)\rvert^2\overline{\widehat{K}(r)} = 1 + \sum_{r\ne 0}\lvert\widehat{f}(r)\rvert^2\overline{\widehat{K}(r)}$ … $\sum_{r\in\mathbb{Z}}\lvert\widehat{f}(r)\rvert^4 = \sum_{r\in\mathbb{Z}}\lvert\widehat{f*f}(r)\rvert^2 = \lVert f*f\rVert_2^2 \le \lVert f*f\rVert_\infty$" (the final inequality uses $\lVert f*f\rVert_1 = 1$).
	  **loc 4:** MO 2009, p. 11, Lemma 3.3 (the period-$u$ torus split applied to $f*f + f\circ f$; this is the statement MV 2010 cites as their Eq.(3)).
	  **quote 4:** "Let $f$ be a square-integrable pdf supported on $(-\tfrac14,\tfrac14)$, and let $K$ be a pdf supported on $(-\delta,\delta)$. Then $\int_{\mathbb{R}}(f*f(x) + f\circ f(x))K(x)\,dx = \frac{2}{u} + 2u^2\sum_{j\ne 0}(\Re\widetilde{f}(j))^2\Re\widetilde{K}(j).$"

- <a id="MV2009"></a>**[MV2009]** Matolcsi, Máté; Vinuesa, Carlos. *Improved bounds on the supremum of autoconvolutions.* J. Math. Anal. Appl. **372** (2010), no. 2, 439–447. [arXiv:0907.1379](https://arxiv.org/abs/0907.1379) — source of the $\ge 1.27481$ bound and of the dual framework underlying <a href="#PBV-master">[PBV-master]</a>.
	- <a id="MV-primitives"></a>**[MV-primitives]** Verbatim source for the same three Fourier-analytic primitives, as MV 2010 consolidates and sharpens them. MV 2010 explicitly attributes them to MO 2009 (p. 3): *"Lemma 3.1. [Lemmas 3.1, 3.2, 3.3, 3.4 in [6]]"*, where [6] is MO 2009.
	  **loc 1:** MV 2010 (arXiv:0907.1379v2), p. 3, Lemma 3.1, Eq.(3) (period-$u$ torus split for $f*f + f\circ f$, cited from MO 2009 Lemma 3.3).
	  **quote 1:** "$\int (f*f(x) + f\circ f(x))K(x)\,dx = \frac{2}{u} + 2u^2 \sum_{j\ne 0}(\Re\widetilde{f}(j))^2 \widetilde{K}(j).$"
	  **loc 2:** MV 2010, pp. 5–6, Lemma 3.3 (sharpened master inequality; the proof contains the constant-plus-tail split and the lattice $F$-bound used in the $z_1$-free derivation).
	  **quote 2 (statement, Eq.(9)):** "Using the notation $z_1 = \lvert\widehat{f}(1)\rvert$ and $k_1 = \widehat{K}(1)$, $\int (f\circ f(x))K(x)\,dx \le 1 + 2 z_1^2 k_1 + \sqrt{\lVert f*f\rVert_\infty - 1 - 2 z_1^4}\sqrt{\lVert K\rVert_2^2 - 1 - 2 k_1^2}.$"
	  **quote 2 (proof excerpt, p. 6):** "$\int (f\circ f(x))K(x)\,dx = \sum_{j\in\mathbb{Z}} \widehat{f\circ f}(j)\widehat{K}(j) = 1 + 2 z_1^2 k_1 + \sum_{j\ne 0,\pm 1}\lvert\widehat{f}(j)\rvert^2 \widehat{K}(j) \le 1 + 2 z_1^2 k_1 + \sqrt{\sum_{j\ne 0,\pm 1}\lvert\widehat{f}(j)\rvert^4}\sqrt{\sum_{j\ne 0,\pm 1}\widehat{K}(j)^2} = 1 + 2 z_1^2 k_1 + \sqrt{\lVert f*f\rVert_2^2 - 1 - 2 z_1^4}\sqrt{\lVert K\rVert_2^2 - 1 - 2 k_1^2} \le 1 + 2 z_1^2 k_1 + \sqrt{\lVert f*f\rVert_\infty - 1 - 2 z_1^4}\sqrt{\lVert K\rVert_2^2 - 1 - 2 k_1^2}.$"
	  **loc 3:** MV 2010, p. 7, Eq.(10) (master inequality in the assembled form used by the present work).
	  **quote 3:** "$\frac{2}{u} + a \le \lVert f*f\rVert_\infty + 1 + 2 z_1^2 k_1 + \sqrt{\lVert f*f\rVert_\infty - 1 - 2 z_1^4}\sqrt{0.5747/\delta - 1 - 2 k_1^2}.$"

- <a id="CS2017"></a>**[CS2017]** Cloninger, Alexander; Steinerberger, Stefan. *On suprema of autoconvolutions with an application to Sidon sets.* Proc. Amer. Math. Soc. **145** (2017), no. 8, 3191–3200. [arXiv:1403.7988](https://arxiv.org/abs/1403.7988) — published $\ge 1.28$.

- <a id="XX2026"></a>**[XX2026]** Xie, Xinyuan. *Unpublished improvement to the lower bound for $C_{1a}$ (claiming $C_{1a} \ge 1.2802$).* 2026. Listed on the canonical page as "Unpublished improvement, Grok".

- <a id="Landau"></a>**[Landau]** Landau, L. J. *Bessel functions: monotonicity and bounds.* J. London Math. Soc. (2) **61** (2000), no. 1, 197–215. The bound $\lvert J_0(z)\rvert^{2}\le 2/(\pi z)$ is used to control the $K_2$ tail past $\xi=10^{5}$ (`lower_bound_proof.tex` line 672).

- <a id="Flint"></a>**[Flint]** Johansson, Fredrik. *Arb: efficient arbitrary-precision midpoint-radius interval arithmetic.* IEEE Trans. Comput. **66** (2017), no. 8, 1281–1292. [arblib.org](http://arblib.org/) — the interval-arithmetic library underlying every certified anchor.

## Contribution notes

Prepared with assistance from Claude (Anthropic). The quoted
statements are faithful renderings of the cited locations in this
repository, of the canonical `constants/1a.md`, and of the MV 2010 /
MO 2009 source papers (arXiv:0907.1379v2 / arXiv:0807.5121v2). Per
the repository's AI-use policy, the human contributor must verify the
primary-literature citations ([MO2004], [MO2009] including
[MO-primitives], [MV2009] including [MV-primitives], [CS2017],
[Landau], [Flint]) against the original sources before submission.
