# An autocorrelation constant related to Sidon sets — verification of the lower bound $C_{1a} \ge 1.292$

## Description of constant

$C_{1a}$ is the largest constant for which one has
$$
\max_{-1/2 \leq t \leq 1/2} \int_{\mathbb{R}} f(t-x) f(x)\,dx \;\geq\; C_{1a} \left(\int_{-1/4}^{1/4} f(x)\,dx\right)^{2}
$$
for all non-negative $f \colon \mathbb{R} \to \mathbb{R}$.
<a href="#1a-def">[1a-def]</a>

Over the family $\mathcal{F}$ of non-negative $f \in L^{1}(\mathbb{R})$ supported in $(-1/4,1/4)$ with positive integral, this is $C_{1a} = \inf_{f \in \mathcal{F}} R(f)$, where $R(f) = \lVert f*f \rVert_{\infty} / (\int f)^{2}$.
<a href="#PBV-def">[PBV-def]</a>

## Known upper bounds

Not affected by this work; the canonical page tracks them. Current best: $\approx 1.502862$ (AI-assisted, 2026).
<a href="#1a-page">[1a-page]</a>

## Known lower bounds

Rows one to six are reproduced from the canonical page
<a href="#1a-lb">[1a-lb]</a>; the last row is established here.

| Bound | Reference | Comments |
| ----- | --------- | -------- |
| $1$ | Trivial | |
| $1.182778$ | [[MO2004](#MO2004)] | |
| $1.262$ | [[MO2009](#MO2009)] | |
| $1.2748$ | [[MV2009](#MV2009)] | Best prior rigorous analytic bound; stated as $1.27481$ in the preprint. |
| $1.28$ | [[CS2017](#CS2017)] | Discrete branch-and-prune search. |
| $1.2802$ | [[XX2026](#XX2026)] | Unpublished, AI-assisted (Grok), unaudited. The canonical page attributes this to Xie (2026), *not* to Cloninger–Steinerberger. |
| $1.292$ | [[PBV2026](#PBV2026)] | This work; computer-assisted, conditional. See below. |

## How the bound is established

The argument extends the Matolcsi–Vinuesa dual framework. For a
Bochner-admissible kernel $K$ (non-negative, with non-negative Fourier
coefficients) and a cosine multiplier $G$, every $f \in \mathcal{F}$
satisfies a quadratic master inequality in five real functionals
$(k_1, K_2, S_1, m_G, a)$ <a href="#PBV-master">[PBV-master]</a>.
Matolcsi–Vinuesa used a single arcsine kernel. Here $K$ is a convex
combination of three arcsine kernels — half-widths $(138,55,25)/1000$,
weights $(85,10,5)/100$, period $u = 638/1000$ — which remains
Bochner-admissible, and $G$ is re-optimised as a $200$-term cosine sum
<a href="#PBV-kernel">[PBV-kernel]</a>.

Each functional is enclosed in `flint.arb` interval arithmetic at
256-bit precision and rounded outward to an exact rational
<a href="#PBV-anchors">[PBV-anchors]</a>:

| Functional | Certified bound | Decimal |
| ---------- | --------------- | ------- |
| $k_1$ (kernel mass moment) | $\ge 9212/10000$ | $0.9212$ |
| $K_2$ (kernel energy) | $\le 47897/10000$ | $4.7897$ |
| $S_1$ (multiplier denominator) | $\le 29841/1000$ | $29.841$ |
| $m_G$ (multiplier minimum) | $\ge 998/1000$ | $0.998$ |
| $a = (4/u)\,m_G^{2}/S_1$ (gain) | $\ge 20925/100000$ | $0.20925$ |

Write $\Phi(M) = M + 1 + \sqrt{(M-1)(K_2-1)}$ and $\tau = 2/u + a$;
the master inequality forces $C_{1a} \ge M$ whenever $\Phi(M) < \tau$.
At $M = 1292/1000$ this holds with a strictly positive, exactly
rational margin,
$$
\tau - \Phi(1292/1000) \;\ge\; \frac{4267003}{1276000} - \frac{66879}{20000} \;=\; \frac{307}{3190000} \;\ge\; 9.6\times10^{-5},
$$
so $C_{1a} \ge 1292/1000 = 1.292$ <a href="#PBV-fail">[PBV-fail]</a>.
This step is exact rational arithmetic, independent of floating point.

The analytic reduction is mechanised in Lean 4 (15 modules, $\approx
8650$ lines, `mathlib v4.29.1`, no `sorry`). The headline theorem
reaches Lean's three logical axioms together with exactly two
numerical user axioms, which record the certifier's outputs for $K_2$
and $a$ <a href="#PBV-lean">[PBV-lean]</a>. The bound is reproducible
from a SHA-256-anchored certificate and is cross-checked independently
at 50 decimal digits <a href="#PBV-cert">[PBV-cert]</a>.

## Scope of the claim

- **Conditional.** The general Lean theorem takes the
  Matolcsi–Vinuesa analytic primitives for the three-scale kernel
  (`ExtremiserPrimitives f`) as a hypothesis. This is Fourier-analytic
  content — an $L^{1}\cap L^{2}$ Plancherel / periodisation bridge —
  not a numerical-engineering gap.
- **The two numerical axioms are new.** They concern this three-scale
  kernel and this $200$-mode $G$; they are analogous in role to the
  Mathematica citations in Matolcsi–Vinuesa but are not stated there.
- **Arb is trusted.** `flint.arb` is peer-reviewed (Johansson 2017)
  but is not itself re-verified inside Lean <a href="#Flint">[Flint]</a>.

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
	  **quote:** "$C_{1a} = \inf_{f\in\mathcal{F}} R(f)$, $R(f) := \lVert f*f\rVert_{\infty}/(\int f)^2$", with $\mathcal{F}$ the non-negative $f\in L^1(\mathbb{R})$ such that $\mathrm{supp}\, f \subseteq (-1/4,1/4)$ and $\int f > 0$.
	- <a id="PBV-main"></a>**[PBV-main]**
	  **loc:** `lower_bound_proof.tex`, Theorem "Main result" (`thm:main`, lines 351–362).
	  **quote:** "Let $f:\mathbb{R}\to\mathbb{R}$ be nonnegative with $\mathrm{supp}\, f\subseteq(-1/4,1/4)$, $\int f>0$, and $\lVert f*f\rVert_{\infty}<\infty$. Then $\lVert f*f\rVert_{\infty}/(\int f)^2 \ge 1292/1000$. In particular $C_{1a}\ge 1292/1000=1.292$."
	- <a id="PBV-master"></a>**[PBV-master]**
	  **loc:** `lower_bound_proof.tex`, Theorem "Master inequality, $z_1$-free form" (`thm:mv-master`, lines 415–426).
	  **quote:** "For $K$ Bochner-admissible at scale $\delta$, $u=1/2+\delta$, and $G$ admissible with constant $m_G$ (defining $S_1$, $a$): for every $f\in\mathcal{F}$ with $\int f>0$, $M + 1 + \sqrt{(M-1)(K_2-1)} \ge 2/u + a$."
	- <a id="PBV-kernel"></a>**[PBV-kernel]**
	  **loc:** `lower_bound_proof.tex`, Definition "Three-scale kernel" (`def:Kms`, lines 526–546) and Theorem "Admissibility" (`thm:admissibility`, lines 567–579).
	  **quote:** $(\delta_1,\delta_2,\delta_3)=(138,55,25)/1000$, $(\lambda_1,\lambda_2,\lambda_3)=(85,10,5)/100$, $u=1/2+\delta_1=638/1000$; $K_{\mathrm{ms}}=\sum_i \lambda_i K_{\mathrm{arc}}(\delta_i;\cdot)$ is Bochner-admissible with $\widetilde{K_{\mathrm{ms}}}(j)=\sum_i \lambda_i J_0(\pi j\delta_i/u)^2 \ge 0$; multiplier degree $N=200$.
	- <a id="PBV-anchors"></a>**[PBV-anchors]**
	  **loc:** `lower_bound_proof.tex`, Lemmas `lem:k1`, `lem:K2`, `lem:S1`, `lem:mG`, `lem:a` (lines 641–756) and table `tab:anchors`.
	  **quote:** $k_1 := \widehat{K_{\mathrm{ms}}}(1) \ge 9212/10000$; $K_2 := \lVert K_{\mathrm{ms}}\rVert_{2}^{2} \le 47897/10000$ (proof certifies $K_2\in[4.78882342,\,4.78890519]$); $S_1 = \sum_{j=1}^{200} a_j^2/\widetilde{K_{\mathrm{ms}}}(j) \le 29841/1000$; $m_G := \min_{[0,1/4]} G \ge 998/1000$; $a = (4/u)\,m_G^{2}/S_1 \ge 20925/100000$.
	- <a id="PBV-fail"></a>**[PBV-fail]**
	  **loc:** `lower_bound_proof.tex`, Proposition "Strict failure at the rational witness" (`prop:fail`, lines 819–858).
	  **quote:** "$\Phi(1292/1000) \le 66879/20000 = 3.34395 < 4267003/1276000 \le \tau$, with margin $\tau-\Phi(1292/1000) \ge 307/3190000 \ge 9.6\times10^{-5} > 0$." An Arb cell-search independently re-certifies the sharper $M_{\mathrm{cert}}\ge 1.29232$.
	- <a id="PBV-lean"></a>**[PBV-lean]**
	  **loc:** `lean/Sidon/MultiScale.lean`: axioms at lines 594 and 622; headline `autoconvolution_ratio_ge_1292_1000` at lines 1089–1098; slack-soundness theorems at lines 638–661. `mathlib v4.29.1`, commit `5e932f97dd25535344f80f9dd8da3aab83df0fe6`.
	  **quote:** The only kernel-specific user axioms are `K2_analytic_le_K2UpperQ` ($K_2 \le 47897/10000$) and `gain_analytic_ge_gainLowerQ` ($a \ge 20925/100000$). The headline takes `(P : ExtremiserPrimitives f)` as a hypothesis and concludes `autoconvolution_ratio f ≥ 1292/1000`; the slack-soundness theorems are one-line `norm_num` checks.
	- <a id="PBV-cert"></a>**[PBV-cert]**
	  **loc:** `delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json`, `multiscale_arcsine_1292.json`; `audit3_mpmath.py`; `README.md`.
	  **quote:** `multiscale_arcsine_1292.json` has `sha256_of_body = 5fa9ae372b23d07f73f41d73c1740926115eb494b6ba3840551458ba8143a7c2` and `M_cert = 66167/51200` ($1.29232421875$). `reference_anchors.json` records the anchors ($k_1=0.9212465899364083$, $K_2\in[4.7888234212591545,\,4.7889051816332424]$, $S_1=29.8409064555132666$, $m_G=0.9999798743824747$, $a=0.2100921474866837$) and parameters (`deltas` $138/55/25\,/1000$, `lambdas` $85/10/5\,/100$, `u` $638/1000$, `n_coeffs` $200$, `prec_bits` $256$). `audit3_mpmath.py` recomputes $K_2$ and $a$ at 50 digits independently of `flint.arb`.

- <a id="MO2004"></a>**[MO2004]** Martin, Greg; O'Bryant, Kevin. *The symmetric subset problem in continuous Ramsey theory.* Exp. Math. **16** (2007), no. 2, 145–165. [arXiv:math/0410004](https://arxiv.org/abs/math/0410004)

- <a id="MO2009"></a>**[MO2009]** Martin, Greg; O'Bryant, Kevin. *The supremum of autoconvolutions, with applications to additive number theory.* Illinois J. Math. **53** (2009), no. 1, 219–235. [arXiv:0807.5121](https://arxiv.org/abs/0807.5121)

- <a id="MV2009"></a>**[MV2009]** Matolcsi, Máté; Vinuesa, Carlos. *Improved bounds on the supremum of autoconvolutions.* J. Math. Anal. Appl. **372** (2010), no. 2, 439–447. [arXiv:0907.1379](https://arxiv.org/abs/0907.1379) — source of the $\ge 1.27481$ bound and of the dual framework underlying <a href="#PBV-master">[PBV-master]</a>.

- <a id="CS2017"></a>**[CS2017]** Cloninger, Alexander; Steinerberger, Stefan. *On suprema of autoconvolutions with an application to Sidon sets.* Proc. Amer. Math. Soc. **145** (2017), no. 8, 3191–3200. [arXiv:1403.7988](https://arxiv.org/abs/1403.7988) — published $\ge 1.28$.

- <a id="XX2026"></a>**[XX2026]** Xie, Xinyuan. *Unpublished improvement to the lower bound for $C_{1a}$ (claiming $C_{1a} \ge 1.2802$).* 2026. Listed on the canonical page as "Unpublished improvement, Grok".

- <a id="Landau"></a>**[Landau]** Landau, L. J. *Bessel functions: monotonicity and bounds.* J. London Math. Soc. (2) **61** (2000), no. 1, 197–215. The bound $\lvert J_0(z)\rvert^{2}\le 2/(\pi z)$ is used for the $K_2$ tail past $\xi=10^5$ (`lower_bound_proof.tex` line 672).

- <a id="Flint"></a>**[Flint]** Johansson, Fredrik. *Arb: efficient arbitrary-precision midpoint-radius interval arithmetic.* IEEE Trans. Comput. **66** (2017), no. 8, 1281–1292. [arblib.org](http://arblib.org/) — the interval-arithmetic library underlying every certified anchor.
