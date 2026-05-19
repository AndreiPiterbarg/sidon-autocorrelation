# An autocorrelation constant related to Sidon sets — verification dossier for $C_{1a} \ge 1.292$

*This dossier is written in the verbatim-quotation style of
[`teorth/optimizationproblems/constants/77a.md`](https://teorth.github.io/optimizationproblems/constants/77a.html).
Every quantitative claim carries an inline anchor; the **References**
section gives, for each anchor, a **`loc:`** (the exact artefact and
line range or key) and a **`quote:`** (the text reproduced
character-for-character). The bound can therefore be checked by
opening each cited location and matching the quoted text — no source
code needs to be read or re-run. Every quote is reproduced either from
a file that ships in this repository or from the canonical
[`constants/1a.md`](https://teorth.github.io/optimizationproblems/constants/1a.html)
page; none is paraphrased.*

## Description of constant

$C_{1a}$ is the largest constant for which one has
$$
\max_{-1/2 \leq t \leq 1/2} \int_{\mathbb{R}} f(t-x) f(x)\ dx \geq C_{1a} \left(\int_{-1/4}^{1/4} f(x)\ dx\right)^2
$$
for all non-negative $f \colon \mathbb{R} \to \mathbb{R}$.
<a href="#1a-def">[1a-def]</a>

Equivalently, with $\mathcal{F}=\{f\in L^1(\mathbb{R}) : f\ge 0,\ \operatorname{supp}(f)\subseteq(-1/4,1/4),\ \int f>0\}$ and $R(f)=\lVert f*f\rVert_{L^\infty}/(\int f)^2$, one has $C_{1a}=\inf_{f\in\mathcal{F}}R(f)$.
<a href="#PBV-constant">[PBV-constant]</a>

## Known upper bounds

Upper bounds are not affected by this work; the canonical page tracks
them (currently down to $\approx 1.502862$, AI-assisted, 2026).
<a href="#1a-page">[1a-page]</a>

## Known lower bounds

The first six rows reproduce the canonical page verbatim
<a href="#1a-lb">[1a-lb]</a>; the final row is the contribution
verified by this dossier.

| Bound | Reference | Comments |
| ----- | --------- | -------- |
| $1$ | Trivial | <a href="#1a-def">[1a-def]</a> |
| $1.182778$ | [[MO2004](#MO2004)] | Canonical page. <a href="#1a-lb">[1a-lb]</a> |
| $1.262$ | [[MO2009](#MO2009)] | Canonical page. <a href="#1a-lb">[1a-lb]</a> |
| $1.2748$ | [[MV2009](#MV2009)] | Rigorous analytic bound (the preprint states it as $1.27481$). <a href="#1a-lb">[1a-lb]</a> <a href="#PBV-bounds">[PBV-bounds]</a> |
| $1.28$ | [[CS2017](#CS2017)] | Published; discrete branch-and-prune. <a href="#1a-lb">[1a-lb]</a> <a href="#PBV-bounds">[PBV-bounds]</a> |
| $1.2802$ | [[XX2026](#XX2026)] | Unpublished, AI-assisted (Grok), unaudited — *not* Cloninger–Steinerberger. <a href="#1a-lb">[1a-lb]</a> |
| $1.292$ | [[PBV2026](#PBV2026)] | **This work.** Computer-assisted; *conditional* on the Matolcsi–Vinuesa analytic primitives for the three-scale kernel together with two externally Arb-certified numerical inequalities. The algebraic chain and the quadratic inversion are machine-checked in Lean 4. <a href="#PBV-main">[PBV-main]</a> |

## How the $1.292$ lower bound is established

The bound is **conditional**. Its dependency structure is the
following; each link is reproduced verbatim under **References**.

1. **Master inequality (Matolcsi–Vinuesa).** Every $f\in\mathcal{F}$
   satisfies a quadratic inequality in five functionals
   $(k_1,K_2,S_1,m_G,a)$ of an auxiliary Bochner-admissible kernel $K$
   and a cosine multiplier $G$. The $z_1$-free form used here is
   stated at <a href="#PBV-master">[PBV-master]</a>; the underlying
   duality is Matolcsi–Vinuesa (2010)
   <a href="#MV2009-src">[MV2009-src]</a>.
2. **Three-scale kernel and admissibility.** $K_{\mathrm{ms}}$ is the
   convex combination of three arcsine kernels at half-widths
   $(138,55,25)/1000$ with weights $(85,10,5)/100$; it is
   Bochner-admissible. Parameters at
   <a href="#PBV-params">[PBV-params]</a>; admissibility at
   <a href="#PBV-admiss">[PBV-admiss]</a>.
3. **Five certified rational anchors.** $k_1\ge 9212/10000$,
   $K_2\le 47897/10000$, $S_1\le 29841/1000$, $m_G\ge 998/1000$, and
   $a\ge 20925/100000$ — each an exact rational obtained by enclosing
   the functional in `flint.arb` interval arithmetic at 256-bit
   precision and rounding outward. Stated at
   <a href="#PBV-k1">[PBV-k1]</a>, <a href="#PBV-K2">[PBV-K2]</a>,
   <a href="#PBV-S1">[PBV-S1]</a>, <a href="#PBV-mG">[PBV-mG]</a>,
   <a href="#PBV-a">[PBV-a]</a>. The $K_2$ tail estimate uses the
   Landau bound $\lvert J_0(z)\rvert^2\le 2/(\pi z)$
   <a href="#Landau-src">[Landau-src]</a>.
4. **Strict-failure step (exact rational arithmetic).** At
   $M=1292/1000$, $\Phi(M)\le 66879/20000=3.34395 < 4267003/1276000
   \le \tau$, with margin $\tau-\Phi(M)\ge 307/3190000\ge
   9.6\times10^{-5}$. Quoted at <a href="#PBV-fail">[PBV-fail]</a>.
5. **Machine-checked chain.** The two numerical inputs are isolated as
   the only kernel-specific user axioms, `K2_analytic_le_K2UpperQ` and
   `gain_analytic_ge_gainLowerQ`
   <a href="#Lean-axioms">[Lean-axioms]</a>. The slack-soundness
   lemmas, the quadratic inversion, the master chain, and the headline
   are Lean theorems <a href="#Lean-slack">[Lean-slack]</a>
   <a href="#Lean-headline">[Lean-headline]</a>. The general headline
   additionally takes the analytic admissibility bundle
   `ExtremiserPrimitives f` as a **hypothesis** — not an axiom
   <a href="#Lean-headline">[Lean-headline]</a>.
6. **Reproducible certificate.** The anchors and $M_{\mathrm{cert}}$
   are recorded in a SHA-256-anchored JSON
   <a href="#cert-sha">[cert-sha]</a>
   <a href="#cert-anchors">[cert-anchors]</a> and independently
   cross-checked at 50 decimal digits by an mpmath script that does
   not call `flint.arb` <a href="#mpmath-src">[mpmath-src]</a>.

## What is *not* claimed

1. **Not unconditional.** The general theorem is conditional on
   `ExtremiserPrimitives f` (the $L^1\cap L^2$ Plancherel plus
   period-$u$ Parseval bridge). This is genuine analytic content, not
   an interval-arithmetic engineering gap.
2. **Not facts from Matolcsi–Vinuesa.** The two numerical axioms are
   *new* inequalities for *this* three-scale kernel and *this*
   200-mode multiplier $G$. They are analogous only *in role* to the
   Mathematica citations in Matolcsi–Vinuesa (2010); they do not
   appear in that paper.
3. **`flint.arb` is trusted, not Lean-verified.** Arb is peer-reviewed
   (Johansson 2017) but is not re-verified inside Lean
   <a href="#Flint-src">[Flint-src]</a>.

## Additional comments and links

- Canonical tracking page: [`constants/1a.md`](https://teorth.github.io/optimizationproblems/constants/1a.html).
  This dossier supports an *update to an existing bound*: it adds the
  $1.292$ lower-bound row and changes no other row.
- Machine-checked development: 15 Lean modules, $\approx 8650$ lines,
  `mathlib` `v4.29.1`, zero `sorry`. Headline
  `Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000`.
- Reproduce the certificate:
  `python -m delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel`
  <a href="#cert-cmd">[cert-cmd]</a>. Consistency audit:
  `python audit_consistency.py` <a href="#audit-cmd">[audit-cmd]</a>.
  Lean build and axiom inventory: `cd lean && lake build` then
  `lake env lean AxiomCheck.lean` <a href="#lean-cmd">[lean-cmd]</a>.

## References

- <a id="1a-page"></a>**[1a-page]** Tao, Terence (ed.). *An autocorrelation constant related to Sidon sets.* `teorth/optimizationproblems`, file `constants/1a.md`. [Page](https://teorth.github.io/optimizationproblems/constants/1a.html)
	- <a id="1a-def"></a>**[1a-def]**
	  **loc:** `constants/1a.md`, section "Description of constant"; the "$1$ | Trivial" row appears under "Known lower bounds".
	  **quote:** "$C_{1a}$ is the largest constant for which one has $$\max_{-1/2 \leq t \leq 1/2} \int_{\mathbb{R}} f(t-x) f(x)\ dx \geq C_{1a} \left(\int_{-1/4}^{1/4} f(x)\ dx\right)^2$$ for all non-negative $f \colon \mathbb{R} \to \mathbb{R}$."
	- <a id="1a-lb"></a>**[1a-lb]**
	  **loc:** `constants/1a.md`, section "Known lower bounds" (table) and the matching reference entries.
	  **quote:** "| $1$ | Trivial | |" / "| $1.182778$ | [MO2004] | |" / "| $1.262$ | [MO2009] | |" / "| $1.2748$ | [MV2009] | |" / "| $1.28$ | [CS2017] | |" / "| $1.2802$ | [XX2026] | Unpublished improvement, Grok|"

- <a id="PBV2026"></a>**[PBV2026]** Piterbarg, Andrei; Bajaj, Jai; Vincent, Derrick. *A New Lower Bound for the Supremum of Autoconvolutions.* Preprint, 2026. This repository: `lower_bound_proof.tex` / `lower_bound_proof.pdf`; Lean development under `lean/Sidon/`.
	- <a id="PBV-constant"></a>**[PBV-constant]**
	  **loc:** `lower_bound_proof.tex:279-288` (Subsection "The Constant $C_{1a}$").
	  **quote:** "Let $\mathcal{F}=\{f\in L^1(\R) : f\ge 0,\ \supp(f)\subseteq(-1/4,1/4),\ \int f>0\}$.  The \emph{autoconvolution constant} is \[ C_{1a} \;=\; \inf_{f\in\mathcal{F}} R(f), \qquad R(f) \;:=\; \frac{\|f*f\|_{\Linf}}{\left(\int f\right)^2}. \]"
	- <a id="PBV-main"></a>**[PBV-main]**
	  **loc:** `lower_bound_proof.tex:351-362` (Theorem "Main result", label `thm:main`).
	  **quote:** "Let $f:\R\to\R$ be nonnegative with $\supp(f)\subseteq(-1/4,1/4)$, $\int f>0$, and $\|f*f\|_{\Linf}<\infty$.  Then \[ \frac{\|f*f\|_{\Linf}}{\left(\int f\right)^2} \;\ge\; \frac{1292}{1000}. \] In particular $C_{1a}\ge 1292/1000=1.292$."
	- <a id="PBV-bounds"></a>**[PBV-bounds]**
	  **loc:** `lower_bound_proof.tex:135-138` (Abstract).
	  **quote:** "This improves on the previously announced bound of $1.28$ due to Cloninger and Steinerberger~\cite{CS17} and on the rigorous analytic bound of $1.27481$ established by Matolcsi and Vinuesa~\cite{MV10}."
	- <a id="PBV-master"></a>**[PBV-master]**
	  **loc:** `lower_bound_proof.tex:415-426` (Theorem "Master inequality, $z_1$-free form", label `thm:mv-master`, equation `eq:master`).
	  **quote:** "Let $K$ be Bochner-admissible at scale $\delta$, $u=1/2+\delta$, and $G$ admissible of degree $N$ with constant $m_G$, defining $S_1$ and $a$ by~\eqref{eq:S1-a}.  Then for every $f\in\mathcal{F}$ with $\int f>0$, \begin{equation}\label{eq:master} M \;+\; 1 \;+\; \sqrt{(M-1)(K_2-1)} \;\ge\; \frac{2}{u} \;+\; a. \end{equation}"
	- <a id="PBV-admiss"></a>**[PBV-admiss]**
	  **loc:** `lower_bound_proof.tex:567-579` (Theorem "Admissibility", label `thm:admissibility`, equation `eq:Ktilde`).
	  **quote:** "The kernel $\Kms$ is Bochner-admissible at scale $\delta_1=138/1000<1/4$, with periodic coefficients \begin{equation}\label{eq:Ktilde} \widetilde{\Kms}(j) \;=\; \sum_{i=1}^{3}\lambda_i\, J_0\!\left(\frac{\pi j\delta_i}{u}\right)^{\!2} \;\ge\; 0, \qquad j\in\Z. \end{equation}"
	- <a id="PBV-params"></a>**[PBV-params]**
	  **loc:** `lower_bound_proof.tex:526-546` (Definition "Three-scale kernel", label `def:Kms`, equation `eq:anchors`).
	  **quote:** "$(\delta_1,\delta_2,\delta_3) \;=\; \bigl(\tfrac{138}{1000},\tfrac{55}{1000},\tfrac{25}{1000}\bigr), \qquad (\lambda_1,\lambda_2,\lambda_3) \;=\; \bigl(\tfrac{85}{100},\tfrac{10}{100},\tfrac{5}{100}\bigr),$ and set $u=1/2+\delta_1=638/1000$."
	- <a id="PBV-k1"></a>**[PBV-k1]**
	  **loc:** `lower_bound_proof.tex:641-648` (Lemma "Kernel-mass moment", label `lem:k1`).
	  **quote:** "$k_1 \;:=\; \widehat{\Kms}(1) \;=\; \sum_{i=1}^{3}\lambda_i J_0(\pi\delta_i)^2 \;\ge\; 9212/10000.$"
	- <a id="PBV-K2"></a>**[PBV-K2]**
	  **loc:** `lower_bound_proof.tex:657-663` (Lemma "Kernel-energy moment", label `lem:K2`); certified enclosure in the proof at `:684-686`.
	  **quote:** "$K_2 \;=\; \|\Kms\|_{L^2(\R)}^{2} \;\le\; 47897/10000.$" — proof: "$K_2\in[4.78882342,\,4.78890519]\subseteq[0,\,47897/10000]$, slack $\ge 7.9\times 10^{-4}$."
	- <a id="PBV-S1"></a>**[PBV-S1]**
	  **loc:** `lower_bound_proof.tex:689-696` (Lemma "Multiplier denominator", label `lem:S1`).
	  **quote:** "$S_1 \;=\; \sum_{j=1}^{200}\frac{a_j^{\,2}}{\widetilde{\Kms}(j)} \;\le\; 29841/1000.$"
	- <a id="PBV-mG"></a>**[PBV-mG]**
	  **loc:** `lower_bound_proof.tex:705-710` (Lemma "Pointwise lower bound of $G$", label `lem:mG`).
	  **quote:** "$m_G \;:=\; \min_{x\in[0,1/4]}G(x) \;\ge\; 998/1000.$"
	- <a id="PBV-a"></a>**[PBV-a]**
	  **loc:** `lower_bound_proof.tex:730-756` (Lemma "Gain functional", label `lem:a`, with proof).
	  **quote:** "$a \;=\; (4/u)\cdot m_G^{\,2}/S_1 \;\ge\; 20925/100000.$" — proof: "$a \;\ge\; \frac{4}{638/1000}\cdot\frac{(998/1000)^{2}}{29841/1000} \;=\; \frac{4000\cdot 998^{2}}{638\cdot 29841\cdot 10^{3}} \;=\; \frac{3984016000}{19038558000} \;=\; 0.20926\ldots,$ hence $a\ge 20925/100000$".
	- <a id="PBV-fail"></a>**[PBV-fail]**
	  **loc:** `lower_bound_proof.tex:819-858` (Proposition "Strict failure at the rational witness", label `prop:fail`, with proof).
	  **quote:** "$\Phi(1292/1000) \;\le\; 3.34395 \;<\; 3.344046\ldots \;\le\; \tau$, with margin $\tau-\Phi(1292/1000)\;\ge\; 9.6\times 10^{-5}$." — proof: "$\Phi(1292/1000) \;\le\; \frac{1292}{1000} + 1 + \frac{105195}{10^{5}} \;=\; \frac{66879}{20000} \;=\; 3.34395.$ … $\tau-\Phi(1292/1000) \;\ge\; \frac{4267003}{1276000} - \frac{66879}{20000} \;=\; \frac{307}{3190000} \;\ge\; 9.6\times 10^{-5}\;>\;0.$"

- <a id="PBV-Lean"></a>**[PBV-Lean]** The Lean 4 development, `lean/Sidon/MultiScale.lean` (headline module). `mathlib` `v4.29.1`, commit `5e932f97dd25535344f80f9dd8da3aab83df0fe6`.
	- <a id="Lean-axioms"></a>**[Lean-axioms]**
	  **loc:** `lean/Sidon/MultiScale.lean:594` and `:622` (the only kernel-specific user axioms), with the slack rationals defined at `:213` and `:240`.
	  **quote:** "axiom K2_analytic_le_K2UpperQ : K2_analytic ≤ (K2UpperQ : ℝ)" (`:594`); "axiom gain_analytic_ge_gainLowerQ : gain_analytic ≥ (gainLowerQ : ℝ)" (`:622`); "def K2UpperQ : ℚ := 47897 / 10000" (`:213`); "def gainLowerQ : ℚ := 20925 / 100000" (`:240`).
	- <a id="Lean-slack"></a>**[Lean-slack]**
	  **loc:** `lean/Sidon/MultiScale.lean:638-661` (five slack-soundness theorems, each closed by a one-line `norm_num`).
	  **quote:** "theorem K_two_upper_bound : (K2UpperQ : ℝ) ≥ (4788906 / 1000000 : ℝ) := by" (`:638`) … "theorem gain_lower_bound : (gainLowerQ : ℝ) ≤ (21009214 / 100000000 : ℝ) := by" (`:660`).
	- <a id="Lean-headline"></a>**[Lean-headline]**
	  **loc:** `lean/Sidon/MultiScale.lean:1089-1098` (theorem `autoconvolution_ratio_ge_1292_1000`).
	  **quote:** "theorem autoconvolution_ratio_ge_1292_1000 (f : ℝ → ℝ) (hf_nonneg : ∀ x, 0 ≤ f x) (hf_supp : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4)) (hf_int_pos : MeasureTheory.integral MeasureTheory.volume f > 0) (h_conv_fin : MeasureTheory.eLpNorm (MeasureTheory.convolution f f (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume) ⊤ MeasureTheory.volume ≠ ⊤) (P : ExtremiserPrimitives f) : autoconvolution_ratio f ≥ (1292 / 1000 : ℝ) := by"

- <a id="PBV-cert"></a>**[PBV-cert]** Reproducible certificate, `delsarte_dual/grid_bound_alt_kernel/certificates/`.
	- <a id="cert-sha"></a>**[cert-sha]**
	  **loc:** `delsarte_dual/grid_bound_alt_kernel/certificates/multiscale_arcsine_1292.json:2351`, key `sha256_of_body`.
	  **quote:** `"sha256_of_body": "5fa9ae372b23d07f73f41d73c1740926115eb494b6ba3840551458ba8143a7c2"`
	- <a id="cert-anchors"></a>**[cert-anchors]**
	  **loc:** `delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json:24-56` (key `anchors`); the production `M_cert` also at `multiscale_arcsine_1292.json:209-211` (key `body.M_cert`).
	  **quote:** `"k_1": { "value_mid": 0.9212465899364083, … }`; `"K_2": { "lower": 4.7888234212591545, "upper": 4.7889051816332424, … }`; `"S_1": { "upper": 29.8409064555132666, … }`; `"min_G": { … "lower": 0.9999798743824747, … }`; `"gain_a": { "lower": 0.2100921474866837, … }`; `"M_cert": { "rational": "1292/1000", "float": 1.292, "rigorous_lower_at_anchors": 1.2921564960222152, … }`; production file: `"float": 1.29232421875`, `"rational": "66167/51200"`.
	- <a id="cert-params"></a>**[cert-params]**
	  **loc:** `delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json:3-23` (keys `kernel`, `G`, `prec_bits`).
	  **quote:** `"deltas": [ "138/1000", "55/1000", "25/1000" ]`; `"lambdas": [ "85/100", "10/100", "5/100" ]`; `"u": "638/1000"`; `"n_coeffs": 200`; `"prec_bits": 256`.
	- <a id="mpmath-src"></a>**[mpmath-src]**
	  **loc:** `audit3_mpmath.py:1-7` (module docstring) and `:36` (precision setting); repository root.
	  **quote:** "Independent mpmath 50-digit verification of the two numerical axioms in lean/Sidon/MultiScale.lean. Recomputes K_2(K_ms) and gain = (4/u) * (min_G)^2 / S_1 using mpmath (completely independent of flint.arb) from the pinned QP coefficients in the production certificate." — and "mp.mp.dps = 50".
	- <a id="cert-cmd"></a>**[cert-cmd]**
	  **loc:** `delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json:2` (key `description`).
	  **quote:** "Running `python -m delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel` emits a self-contained certificate"
	- <a id="audit-cmd"></a>**[audit-cmd]**
	  **loc:** `README.md:36-38`.
	  **quote:** "```bash` / `python audit_consistency.py` / `````"
	- <a id="lean-cmd"></a>**[lean-cmd]**
	  **loc:** `README.md:63-66`.
	  **quote:** "cd lean && lake build" / "lake env lean AxiomCheck.lean    # print axiom inventory of the headline"

- <a id="MO2004"></a>**[MO2004]** Martin, Greg; O'Bryant, Kevin. *The symmetric subset problem in continuous Ramsey theory.* Exp. Math. **16** (2007), no. 2, 145–165. [arXiv:math/0410004](https://arxiv.org/abs/math/0410004)
	- <a id="MO2004-src"></a>**[MO2004-src]**
	  **loc:** `constants/1a.md`, "References" section.
	  **quote:** "[MO2004] Martin, Greg; O'Bryant, Kevin. The symmetric subset problem in continuous Ramsey theory. Exp. Math. 16, No. 2, 145-165 (2007). [arXiv:math/0410004](https://arxiv.org/abs/math/0410004)"

- <a id="MO2009"></a>**[MO2009]** Martin, Greg; O'Bryant, Kevin. *The supremum of autoconvolutions, with applications to additive number theory.* Illinois J. Math. **53** (2009), no. 1, 219–235. [arXiv:0807.5121](https://arxiv.org/abs/0807.5121)
	- <a id="MO2009-src"></a>**[MO2009-src]**
	  **loc:** Preprint bibliography, `lower_bound_proof.tex:1022-1025`, cite key `MO2009`.
	  **quote:** "Greg Martin and Kevin O'Bryant, \emph{The supremum of autoconvolutions, with applications to additive number theory}, Illinois J. Math. \textbf{53} (2009), no.~1, 219--235, arXiv:0807.5121."

- <a id="MV2009"></a>**[MV2009]** Matolcsi, Máté; Vinuesa, Carlos. *Improved bounds on the supremum of autoconvolutions.* J. Math. Anal. Appl. **372** (2010), no. 2, 439–447. [arXiv:0907.1379](https://arxiv.org/abs/0907.1379)
	- <a id="MV2009-src"></a>**[MV2009-src]**
	  **loc:** Preprint bibliography, `lower_bound_proof.tex:1027-1030`, cite key `MV10`. Primary source: arXiv:0907.1379; the master-inequality structure is restated verbatim at <a href="#PBV-master">[PBV-master]</a>.
	  **quote:** "M\'at\'e Matolcsi and Carlos Vinuesa, \emph{Improved bounds on the supremum of autoconvolutions}, J. Math. Anal. Appl. \textbf{372} (2010), no.~2, 439--447, arXiv:0907.1379."

- <a id="CS2017"></a>**[CS2017]** Cloninger, Alexander; Steinerberger, Stefan. *On suprema of autoconvolutions with an application to Sidon sets.* Proc. Amer. Math. Soc. **145** (2017), no. 8, 3191–3200. [arXiv:1403.7988](https://arxiv.org/abs/1403.7988)
	- <a id="CS2017-src"></a>**[CS2017-src]**
	  **loc:** Preprint bibliography, `lower_bound_proof.tex:1002-1005`, cite key `CS17`.
	  **quote:** "Alexander Cloninger and Stefan Steinerberger, \emph{On suprema of autoconvolutions with an application to Sidon sets}, Proc. Amer. Math. Soc.\ \textbf{145} (2017), no.~8, 3191--3200, arXiv:1403.7988."

- <a id="XX2026"></a>**[XX2026]** Xie, Xinyuan. *Unpublished improvement to the lower bound for $C_{1a}$ (claiming $C_{1a} \ge 1.2802$).* 2026.
	- <a id="XX2026-src"></a>**[XX2026-src]**
	  **loc:** `constants/1a.md`, "Known lower bounds" table row and "References" entry.
	  **quote:** "| $1.2802$ | [XX2026] | Unpublished improvement, Grok|" — reference (reproduced up to the trailing link, which is elided): "[XX2026] Xie, Xinyuan. Unpublished improvement to the lower bound for $C_{1a}$ (claiming $C_{1a} \ge 1.2802$). 2026. See [Grok chat](…)."

- <a id="Landau"></a>**[Landau]** Landau, L. J. *Bessel functions: monotonicity and bounds.* J. London Math. Soc. (2) **61** (2000), no. 1, 197–215.
	- <a id="Landau-src"></a>**[Landau-src]**
	  **loc:** `lower_bound_proof.tex:672` (uniform Bessel bound used in the $K_2$ tail estimate); preprint bibliography at `:1041-1043`, cite key `Landau`.
	  **quote:** "$|J_0(z)|^2\le 2/(\pi z)$~\cite{Landau}" — bibliography: "L.~J. Landau, \emph{Bessel functions: monotonicity and bounds}, J. London Math. Soc.\ (2) \textbf{61} (2000), no.~1, 197--215."

- <a id="Flint"></a>**[Flint]** Johansson, Fredrik. *Arb: efficient arbitrary-precision midpoint-radius interval arithmetic.* IEEE Trans. Comput. **66** (2017), no. 8, 1281–1292. [arblib.org](http://arblib.org/)
	- <a id="Flint-src"></a>**[Flint-src]**
	  **loc:** Preprint bibliography, `lower_bound_proof.tex:1012-1015`, cite key `Flint`.
	  **quote:** "Fredrik Johansson, \emph{Arb: efficient arbitrary-precision midpoint-radius interval arithmetic}, IEEE Trans. Comput. \textbf{66} (2017), no.~8, 1281--1292.  See \url{http://arblib.org/}."

## Contribution notes

Prepared with assistance from Claude (Anthropic). Every quotation was
extracted by exact text match from the cited in-repository artefact or
from the canonical `constants/1a.md`. The primary-literature
bibliographic entries — [MO2004], [MO2009], [MV2009], [CS2017],
[Landau], [Flint] — and the values attributed to them must be reviewed
and verified by the human contributor against the original sources
before submission, in accordance with the repository's AI-use policy.
