# Project Report: The Piterbarg-Bajaj-Vincent Bound

> A new lower bound on the autoconvolution constant
> $$ C_{1a} \;\ge\; \frac{1292}{1000} \;=\; 1.292, $$
> improving the previously announced $1.2802$ of Cloninger-Steinerberger
> (2017) and the rigorous analytic $1.27481$ of Matolcsi-Vinuesa
> (2010). The argument is closed by interval arithmetic in
> `flint.arb` at 256-bit precision and mechanized in Lean 4 with a
> single user axiom. The end-to-end audit (`audit_consistency.py`)
> reports **50/50 checks passing**.

| | |
|---|---|
| **Authors** | Andrei Piterbarg, Jai Bajaj, Derrick Vincent |
| **Manuscript** | [`lower_bound_proof.tex`](lower_bound_proof.tex) / [`lower_bound_proof.pdf`](lower_bound_proof.pdf) (9 pages, 841 lines TeX) |
| **Lean formalization** | [`lean/Sidon/`](lean/Sidon/) (594 lines across 4 files) |
| **Numerical certificate** | [`delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json`](delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json) |
| **Audit** | `python audit_consistency.py` &mdash; 50 checks, 0 failed |
| **First commit** | 2026-02-03 |
| **Latest commit** | 2026-05-12 |
| **Total commits** | 528 |

---

## 1. The Result

For every nonnegative $f \in L^1(\mathbb{R})$ supported on
$(-\tfrac14, \tfrac14)$ with $\int f > 0$ and
$\|f * f\|_{L^\infty} < \infty$,

$$ \frac{\|f * f\|_{L^\infty}}{\bigl(\int f\bigr)^2} \;\ge\; \frac{1292}{1000}. $$

Taking the infimum, $C_{1a} \ge 1292/1000 = 1.292$. Combined with the
upper bound $C_{1a} \le 1.5029$ from Georgiev-Gomez Serrano-Tao-Wagner
(AlphaEvolve, arXiv:2511.02864), the constant is now bracketed in
$[1.292, 1.5029]$. The improvement is

| | $C_{1a} \ge$ | Source |
|---|---|---|
| Erdős-Turán (1941) | $1$ | classical |
| Martin-O'Bryant (2009) | $1.262$ | arXiv:0807.5121 |
| Matolcsi-Vinuesa (2010) | $1.27481$ | arXiv:0907.1379 |
| Cloninger-Steinerberger (2017) | $1.2802$ | arXiv:1403.7988 |
| **This work (Piterbarg-Bajaj-Vincent)** | **$1.292$** | manuscript at root |

The lift over the prior published lower bound is
$1.292 - 1.2802 = 0.0118$; over the rigorous analytic
Matolcsi-Vinuesa baseline, $1.292 - 1.27481 = 0.01719$.

## 2. Method

The proof refines the Matolcsi-Vinuesa dual framework along four
axes:

1. **Three-scale arcsine kernel.** The single arcsine kernel
   $K_{\rm arc}(\delta_1; \cdot)$ used by MV is replaced by the convex
   combination
   $$ K_{\rm ms} = \sum_{i=1}^{3} \lambda_i\, K_{\rm arc}(\delta_i; \cdot), $$
   with
   $(\delta_1, \delta_2, \delta_3) = (138, 55, 25)/1000$ and
   $(\lambda_1, \lambda_2, \lambda_3) = (85, 10, 5)/100$. The smaller
   scales fill the gaps left by the first Bessel zero of
   $J_0(\pi \delta_1 \xi)^2$, lowering the dominant denominator $S_1$
   from $\approx 87.4$ (single-scale) to $\le 29.841$.

2. **200-mode cosine multiplier.** The trigonometric multiplier $G$ is
   re-optimized as a 200-cosine expansion (rather than the 119 modes
   of MV2010), solved by a convex QP minimizing
   $\sum_j a_j^2 / \widetilde{K_{\rm ms}}(j)$ subject to $G \ge 1$ on
   a 5001-point grid in $[0, 1/4]$. Coefficients are rationalized to
   $a_j \in \mathbb{Q}$ with denominator $10^8$.

3. **Rigorous interval arithmetic.** Every analytic functional
   entering the master inequality is bounded in `flint.arb` at
   256-bit precision and rounded outward to an exact rational.

4. **Quadratic strict-failure witness.** The $z_1$-free quadratic
   master inequality
   $$ \Phi(M) \;=\; M + 1 + \sqrt{(M-1)(K_2 - 1)} \;\ge\; \tau \;=\; \tfrac{2}{u} + a $$
   is closed by exhibiting a rational $M_0 = 1292/1000$ at which
   $\Phi(M_0) < \tau$ strictly. The certified upper bound on $K_2$
   makes $\Phi$ an over-estimate, so the strict failure forces
   $R(f) > M_0$ for every admissible $f$.

## 3. The Certified Anchors

The five real functionals certified in `flint.arb` at 256-bit
precision (all rationals are sourced from
[`reference_anchors.json`](delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json)):

| Functional | Direction | Certifier value | Rational slack | Lean theorem |
|---|---|---|---|---|
| $k_1 = \widehat{K_{\rm ms}}(1)$ | $\ge$ | $0.92124658993\ldots$ | $9212/10000$ | `k_one_lower_bound` |
| $K_2 = \|K_{\rm ms}\|_{L^2}^2$ | $\le$ | $\in [4.78882342, 4.78890519]$ | $47897/10000$ | `K_two_upper_bound` |
| $S_1 = \sum_j a_j^2 / \widetilde{K}(j)$ | $\le$ | $29.84090646\ldots$ | $29841/1000$ | `S_one_upper_bound` |
| $m_G = \min_{[0,1/4]} G$ | $\ge$ | $0.99997987\ldots$ | $998/1000$ | `min_G_lower_bound` |
| $a = (4/u)\, m_G^2 / S_1$ | $\ge$ | $0.21009214\ldots$ | $20925/100000$ | `gain_lower_bound` |

The **strict-failure margin** at the rational witness
$M_0 = 1292/1000$, $K_2 \le 47897/10000$ is exactly

$$ \tau - \Phi(M_0) \;=\; \frac{4267003}{1276000} - \frac{66879}{20000} \;=\; \frac{307}{3190000} \;\approx\; 9.624 \times 10^{-5}. $$

The certifier itself reports a tighter
$M_{\rm cert} \ge 66167/51200 \approx 1.29232422$ (production driver)
when the analytic anchors are coupled rather than rationalized
separately. The headline target $1292/1000$ is the looser rational
floor used in the published statement and in the Lean axiom.

## 4. Lean 4 Formalization

The analytic chain is mechanized in [`lean/Sidon/MultiScale.lean`](lean/Sidon/MultiScale.lean)
on top of Mathlib. The module compiles cleanly under `lake build
Sidon.MultiScale` with **zero `sorry` tactics**. Its dependency
closure reaches exactly **one user axiom**:

| Axiom | Statement |
|---|---|
| `MV_master_inequality_for_extremiser` | The 3-scale Matolcsi-Vinuesa master inequality on $\mathbb{R}/u\mathbb{Z}$, stated with the slack rationals `K2UpperQ = 47897/10000` and `gainLowerQ = 20925/100000` substituted for $K_2$ and $a$. |

Six dependent statements are Lean **theorems** (no `sorry`):

| Theorem | Role |
|---|---|
| `master_inequality_M_lower` | Quadratic inversion (Prop 5.1 of the paper); case analysis on $M \le 1$ vs $M > 1$ via `Real.sqrt` monotonicity. |
| `K_two_upper_bound` | The slack rational $K_2 \le 47897/10000$ dominates the certifier-reported $K_2 \le 4.788906$ (one-line `norm_num`). |
| `k_one_lower_bound` | Slack-soundness for $k_1$. |
| `S_one_upper_bound` | Slack-soundness for $S_1$. |
| `min_G_lower_bound` | Slack-soundness for $m_G$. |
| `gain_lower_bound` | Slack-soundness for $a$. |

The canonical headline theorem is

```lean
theorem Sidon.MultiScale.autoconvolution_ratio_ge_1292_1000
    (f : ℝ → ℝ)
    (hf_nonneg  : ∀ x, 0 ≤ f x)
    (hf_supp    : Function.support f ⊆ Set.Ioo (-(1/4 : ℝ)) (1/4))
    (hf_int_pos : MeasureTheory.integral MeasureTheory.volume f > 0)
    (h_conv_fin : MeasureTheory.eLpNorm
        (MeasureTheory.convolution f f
          (ContinuousLinearMap.mul ℝ ℝ) MeasureTheory.volume)
        ⊤ MeasureTheory.volume ≠ ⊤) :
    autoconvolution_ratio f ≥ (1292 : ℝ) / 1000
```

with decimal restatement `autoconvolution_ratio_ge_1_292` and the
flipped form `C1a_ge_1292` (`1292/1000 ≤ autoconvolution_ratio f`)
exported from the same namespace.

The slack-anchor substitution is monotonically sound: the master
inequality is increasing in $K_2 - 1$ and in $a$, so any true bound
on the analytic functionals transports to a valid bound for the
axiom. The five `norm_num`-checked slack-soundness theorems above
discharge that the rational slacks are on the correct side of the
certifier-reported decimals.

## 5. Repository Layout

```
compact_sidon/
├── lower_bound_proof.tex             # The 841-line manuscript
├── lower_bound_proof.pdf             # 9-page compiled output
├── audit_consistency.py              # 50-check cross-source audit
├── REPORT.md                         # This file
├── README.md                         # Project overview
│
├── lean/                             # Lean 4 formalization
│   ├── Sidon/MultiScale.lean         # Headline theorem + 1 axiom + 6 theorems
│   ├── Sidon/Defs.lean               # Shared definitions
│   ├── Sidon.lean                    # Top-level module entry
│   └── AxiomCheck.lean               # Prints axiom inventory
│
├── delsarte_dual/                    # The arb certifier
│   ├── grid_bound/                   # Single-scale MV machinery + certify.py verifier
│   ├── grid_bound_alt_kernel/        # Three-scale kernel, QP for G, bisect driver
│   │   └── certificates/
│   │       ├── reference_anchors.json    # Canonical 256-bit anchors
│   │       └── multiscale_arcsine_1292.json  # Fresh-run certificate
│   └── README.md
│
├── tests/                            # pytest suite
│   ├── grid_bound_alt_kernel/        # Kernel admissibility, Bochner positivity, QP
│   └── README.md
│
├── docs/                             # Public documentation
│   ├── proof_outline.md              # Mathematical summary (sections, key formulas)
│   ├── reproducibility.md            # Exact reproduction commands
│   ├── formalization.md              # Lean module description
│   ├── verification.md               # 14-task verification checklist
│   ├── presentation/                 # Slide deck (.pptx + figures)
│   └── attempts/                     # Historical attempt write-ups
│
└── archive/                          # Earlier exploration (cs-cascade,
                                      # Lasserre SDP, agent_experiments, etc.)
```

## 6. Reproducing the Result

### Compile the manuscript

```bash
pdflatex -interaction=nonstopmode lower_bound_proof.tex
```

(No external `.bib` file: the bibliography is inlined via
`thebibliography`.) Output: 9 pages, no overfull/underfull/undefined
warnings.

### Regenerate the numerical certificate

```bash
pip install python-flint cvxpy numpy
python -m delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel
```

Wall time $\approx 11\ \text{s}$ on a modern laptop at 256-bit
precision. The driver emits
`delsarte_dual/grid_bound_alt_kernel/certificates/multiscale_arcsine_1292.json`
with the five anchors, the bisection history, the terminal cell
list, and a SHA-256 body hash.

### Independent verification

```bash
python -m delsarte_dual.grid_bound.certify \
    delsarte_dual/grid_bound_alt_kernel/certificates/multiscale_arcsine_1292.json
```

`certify.py` is a stand-alone verifier that imports only
`python-flint` primitives. Exit code `0` iff the certificate body
hash matches, every anchor is recomputable in arb at the declared
precision, the terminal cells cover $[0, \mu(M_{\rm cert})]$
contiguously, and every cell has $\Phi < 0$ upper bound.

### Build the Lean formalization

```bash
cd lean && lake build Sidon.MultiScale
lake env lean AxiomCheck.lean
```

`AxiomCheck.lean` prints the axiom closure of the headline theorem.
Expected: Lean's three core axioms (`Classical.choice`, `propext`,
`Quot.sound`) plus exactly one user axiom
`Sidon.MultiScale.MV_master_inequality_for_extremiser`.

### Run the cross-source audit

```bash
python audit_consistency.py             # 50 checks, summary verdict
python audit_consistency.py --verbose   # print every individual check
```

Verifies that the numerical anchors in
[`reference_anchors.json`](delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json),
the slack rationals in
[`lean/Sidon/MultiScale.lean`](lean/Sidon/MultiScale.lean), the
decimal claims in `docs/{proof_outline,reproducibility,formalization,verification}.md`,
the headline-bound claims across READMEs, and the Proposition 5.1
arithmetic in
[`lower_bound_proof.tex`](lower_bound_proof.tex) are all mutually
consistent and on the correct side of the arb endpoints.

### Run the pytest suite

```bash
pytest tests/grid_bound_alt_kernel/
```

Covers kernel admissibility, Bochner positivity of $\widehat{K_{\rm ms}}$,
the QP solver convergence, and the single-scale baseline check
against the published Matolcsi-Vinuesa value $1.27481$.

## 7. Cross-Source Audit Framework

`audit_consistency.py` is the project's source of truth for
quantitative consistency. Each run re-derives the analytic anchors in
`flint.arb` at 256-bit precision and verifies every quantitative
claim against the freshly computed ground truth across eight
sections:

| Section | What it checks |
|---|---|
| A | Kernel-parameter consistency (rationals declared in Lean / LaTeX / code agree exactly). |
| B | Slack-rational soundness (every Lean rational anchor is a true bound on the arb endpoint). |
| C | Lean axiom RHS soundness (each of the five $\{k_1, K_2, S_1, m_G, a\}$ slack comparisons is rationally true). |
| D | Tight-decimal claim soundness (every decimal value asserted in READMEs / JSON / docstrings / LaTeX is on the correct side of the arb endpoint). |
| E | LaTeX Proposition 5.1 strict-failure arithmetic (exact rational verification of every step in the closing chain). |
| F | LaTeX per-lemma slack-value claims (e.g. "slack $\ge 9.3 \times 10^{-5}$"). |
| G | $K_2 = \text{bulk} + \text{tail}$ decomposition (Watson tail bound and the constant $C = \sum_i \lambda_i / \delta_i$). |
| H | Published bound consistency ($M_{\rm cert}$ production $\ge 1.29232422$; slack-anchor bisection $\ge 1.29215650$; headline $\ge 1292/1000$). |

**Current status: 50 / 50 checks pass, verdict `ALL CHECKS PASS`.**

## 8. Project History (Selected)

The repository carries a substantial earlier-exploration layer under
`archive/`, including:

- A multiscale branch-and-prune cascade extending the CS17 method,
  archived at `docs/attempts/cs_writeup_legacy/` (writeup) and
  `archive/cloninger-steinerberger/` (code).
- A Lasserre SDP hierarchy track with correlative sparsity for $d \in
  \{32, 64, 128\}$, archived at `docs/attempts/lasserre_writeup/` and
  `archive/coarse_lp_bnb/`.
- A two-scale arcsine kernel precursor that produced
  $C_{1a} \ge 1651/1280 \approx 1.28984$, documented in
  [`docs/attempts/multiscale_arcsine.md`](docs/attempts/multiscale_arcsine.md).
- Earlier Hölder, KBK, AlphaEvolve-dual, cohn-elkies, and minimum-overlap
  attempts under `archive/`.

Each historical attempt is preserved with its decision record (often
including the FLAG that closed it).

## 9. Trust Boundary

The published bound rests on the following components:

| Component | Trust |
|---|---|
| Lean 4 kernel | Foundational (assumed sound). |
| Mathlib (cited in the paper) | Community-verified library. |
| `MV_master_inequality_for_extremiser` axiom | One user axiom; encodes the 3-scale extension of Matolcsi-Vinuesa Lemma 3.1. Its proof is a verbatim repeat of the single-scale Fourier reduction with $J_0(\pi \delta \xi)^2$ replaced by $\sum_i \lambda_i J_0(\pi \delta_i \xi)^2$. The substitution preserves Bochner-admissibility (Theorem 3.2 of the paper). |
| `python-flint` / Arb library | Standard interval-arithmetic backend (Johansson 2017). |
| Numerical anchors | All five anchors are reproduced exactly by `bisect_alt_kernel.py` and independently re-verified by `grid_bound/certify.py`. |
| Rational slack substitution into the Lean axiom | Sound by monotonicity of the master inequality in $K_2 - 1$ and $a$; the five `norm_num`-decided slack-soundness theorems confirm the rationals lie on the correct side of the certifier's decimal output. |
| Final descent (`master_inequality_M_lower`) | Pure Lean theorem; case analysis on $M \le 1$ vs $M > 1$ via `Real.sqrt` monotonicity. No external dependency. |

No component is required beyond those listed.

## 10. Open Items

- The audit framework is comprehensive but only checks **consistency**;
  it does not verify the slack-anchor bisection at finer resolutions.
  The production driver's sharper bound $M_{\rm cert} \ge 1.29232422$
  could be used as the headline rational target ($1.292 \to 1.2923$)
  if the cost-benefit case for the tighter Lean target is worthwhile.
- The strict-failure margin $307/3190000 \approx 9.6 \times 10^{-5}$
  is comfortable but not large. Tightening the rational slacks on
  $S_1$ (currently $\le 29.841$ vs the certifier's $29.840907$)
  would widen it without recomputing.
- The `MV_master_inequality_for_extremiser` axiom remains
  external. Internalizing it as a Lean theorem (by porting the
  Fourier-reduction proof of Matolcsi-Vinuesa Lemma 3.1 to Lean)
  would eliminate the only user axiom in the closure.
- Two figures &mdash; a plot of the three-scale kernel and a plot of the
  periodic coefficients $j \mapsto \widetilde{K_{\rm ms}}(j)$
  &mdash; would substantially aid §3 of the manuscript and are not
  currently included.

## References

- [Manuscript: `lower_bound_proof.pdf`](lower_bound_proof.pdf)
- [Lean module: `lean/Sidon/MultiScale.lean`](lean/Sidon/MultiScale.lean)
- [Numerical certificate: `reference_anchors.json`](delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json)
- [Audit script: `audit_consistency.py`](audit_consistency.py)
- [Proof outline: `docs/proof_outline.md`](docs/proof_outline.md)
- [Reproducibility: `docs/reproducibility.md`](docs/reproducibility.md)
- [Formalization notes: `docs/formalization.md`](docs/formalization.md)
- [Verification checklist: `docs/verification.md`](docs/verification.md)

External:
- Cloninger, A., Steinerberger, S. *On suprema of autoconvolutions
  with an application to Sidon sets.* Proc. Amer. Math. Soc. 145
  (2017), 3191-3200, arXiv:1403.7988.
- Matolcsi, M., Vinuesa, C. *Improved bounds on the supremum of
  autoconvolutions.* J. Math. Anal. Appl. 372 (2010), 439-447,
  arXiv:0907.1379.
- Martin, G., O'Bryant, K. *The supremum of autoconvolutions, with
  applications to additive number theory.* Illinois J. Math. 53
  (2009), 219-235, arXiv:0807.5121.
- Johansson, F. *Arb: efficient arbitrary-precision midpoint-radius
  interval arithmetic.* IEEE Trans. Comput. 66(8) (2017), 1281-1292.
- de Moura, L., Ullrich, S. *The Lean 4 theorem prover and programming
  language.* CADE 28, LNCS 12699, Springer, 2021, 625-635.
- The mathlib community. *The Lean mathematical library.* CPP 2020,
  367-381.
- Georgiev, B., G&oacute;mez-Serrano, J., Tao, T., Wagner, A.Z.
  *Mathematical exploration and discovery at scale.* arXiv:2511.02864.
