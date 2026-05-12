# Improving the Bounds on the Supremum of Autoconvolutions

## The Piterbarg-Bajaj-Vincent Bound

This repository accompanies the preprint *Improving the Bounds on the
Supremum of Autoconvolutions* (Piterbarg, Bajaj, Vincent), in which we
propose the **Piterbarg-Bajaj-Vincent Bound** on the Sidon autocorrelation
constant:
$$C_{1a} \;\ge\; \frac{1292}{1000} \;=\; 1.292.$$
This improves on the previously announced lower bound of $1.2802$ due to
Cloninger and Steinerberger (2017) and on the rigorous analytic bound of
$1.27481$ established by Matolcsi and Vinuesa (2010). The argument builds
directly on the Matolcsi-Vinuesa dual framework: the single arcsine
kernel in their master inequality is replaced by a convex combination of
three arcsine kernels, and the cosine multiplier is re-optimized as a
$200$-mode expansion. The five resulting functionals are evaluated
rigorously in `flint.arb` interval arithmetic at 256-bit precision, and
the analytic chain is mechanized in Lean 4.

## Problem statement

Let $\mathcal{F} = \{ f \in L^1(\mathbb{R}) : f \ge 0,\ \operatorname{supp}(f)
\subseteq (-\tfrac14, \tfrac14),\ \int f > 0 \}$. The autoconvolution constant
is
$$C_{1a} \;=\; \inf_{f \in \mathcal{F}} \frac{\|f * f\|_{L^\infty}}{(\int f)^2}.$$
By homogeneity one may normalize $\int f = 1$, in which case
$C_{1a} = \inf_f \|f * f\|_\infty \ge 1$.

## Repository layout

| Path | Contents |
|------|----------|
| [`lower_bound_proof.pdf`](lower_bound_proof.pdf) / [`lower_bound_proof.tex`](lower_bound_proof.tex) | The paper. |
| [`audit_consistency.py`](audit_consistency.py) | End-to-end audit script: re-runs the production pipeline and verifies every numerical claim across the paper, the Lean module, the JSON anchors, the READMEs, and the docs. |
| [`delsarte_dual/grid_bound/`](delsarte_dual/grid_bound/) | The Matolcsi-Vinuesa machinery in `flint.arb`: cell-search certifier, master inequality, Taylor branch-and-bound for $\min G$, independent verifier. |
| [`delsarte_dual/grid_bound_alt_kernel/`](delsarte_dual/grid_bound_alt_kernel/) | The multi-scale arcsine kernel, the QP for $G$, and the certifier driver `bisect_alt_kernel.py`. |
| [`delsarte_dual/grid_bound_alt_kernel/certificates/`](delsarte_dual/grid_bound_alt_kernel/certificates/) | `reference_anchors.json` (canonical anchors); a fresh run emits `multiscale_arcsine_1292.json` here. |
| [`lean/Sidon/`](lean/Sidon/) | The Lean 4 formalization: `Defs.lean` (shared definitions) and `MultiScale.lean` (headline theorem and its sole user axiom). |
| [`tests/`](tests/) | `pytest` suite covering kernel admissibility, Bochner positivity, the QP solver, and the single-scale baseline. |
| [`docs/`](docs/) | Secondary documentation: proof outline, reproducibility, formalization notes, audit specification, attempts archive. |

## Running the full audit

The single command

```bash
python audit_consistency.py
```

re-runs the production pipeline at 256-bit `flint.arb` precision and
verifies every quantitative claim across all five publication surfaces
(LaTeX paper, Lean module, JSON anchors, READMEs, docs).  Pass
`--verbose` to print every individual check.  Exit code `0` iff every
claim is sound; `1` otherwise.

The eight check sections cover kernel-parameter consistency,
slack-rational soundness, Lean axiom RHS soundness, tight-decimal claims
across surfaces, the LaTeX Proposition 5.1 strict-failure arithmetic
(exact rational verification of `tau - Phi(1292/1000) = 307/3190000`),
per-lemma slack values, the `K_2 = bulk + tail` Watson-tail
decomposition, and the published bound.  Run this first whenever any
numerical value is changed.

For the focused unit-test suite (kernel admissibility, Bochner
positivity, QP convergence, single-scale baseline):

```bash
pytest tests/grid_bound_alt_kernel/
```

To rebuild the Lean formalisation:

```bash
cd lean && lake build Sidon.MultiScale
```

## Key documents in `docs/`

- [`docs/proof_outline.md`](docs/proof_outline.md) — mathematical summary of the proof (master inequality, three-scale kernel, certified anchors, strict-failure witness, lift mechanism).
- [`docs/reproducibility.md`](docs/reproducibility.md) — exact instructions to reproduce the certificate and build the Lean formalization.
- [`docs/formalization.md`](docs/formalization.md) — the Lean 4 module, the headline theorem, and the sole user axiom with its discharge method.
- [`docs/verification.md`](docs/verification.md) — public audit specification: fourteen independent checks.

## References

- Matolcsi, M., Vinuesa, C. *Improved bounds on the supremum of autoconvolutions.* J. Math. Anal. Appl. **372** (2010), 439-447. [arXiv:0907.1379](https://arxiv.org/abs/0907.1379).
- Cloninger, A., Steinerberger, S. *On suprema of autoconvolutions with an application to Sidon sets.* Proc. Amer. Math. Soc. **145** (2017), 3191-3200. [arXiv:1403.7988](https://arxiv.org/abs/1403.7988).
- Martin, G., O'Bryant, K. *The supremum of autoconvolutions, with applications to additive number theory.* Illinois J. Math. **53** (2009), 219-235. [arXiv:0807.5121](https://arxiv.org/abs/0807.5121).
- Johansson, F. *Arb: efficient arbitrary-precision midpoint-radius interval arithmetic.* IEEE Trans. Comput. **66** (2017), 1281-1292. [arblib.org](https://arblib.org/).
- Watson, G. N. *A Treatise on the Theory of Bessel Functions*, 2nd ed., Cambridge University Press, 1944.
- AlphaEvolve project. *AI-driven optimization of analytic constants arising in extremal combinatorics and additive number theory*, preprint, 2025. [arXiv:2511.02864](https://arxiv.org/abs/2511.02864).
