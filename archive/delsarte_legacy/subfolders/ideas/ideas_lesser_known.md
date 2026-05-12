# Lesser-Known Techniques for Autoconvolution Supremum Lower Bounds

Survey date: 2026-04-20. Budget: ~18 WebSearch/WebFetch calls. Goal: surface
obscure / non-SDP techniques that could push $C_{1a}$ above $1.2802$ for
$f\ge 0$ on $[-1/4,1/4]$ with $\int f = 1$ and target $\|f*f\|_\infty$.

## 100-word summary

Beyond the familiar MV / CS / White / Boyer-Li gradient-and-SDP cluster, two
genuinely distinct techniques appear underused. (A) White's Erdős-overlap
*Fourier-to-convex-program* transform (arXiv:2201.05704) converts an
$L^\infty$-style extremal problem on sums to a tractable convex program via
elementary Fourier analysis, closer to Delsarte/Beurling-Selberg than to SDP;
it has not been transcribed to the $f*f$ setting. (B) Barnard-Steinerberger /
Gonçalves-Oliveira-Steinerberger *dual-extremizer + Hausdorff-Young* framework
(arXiv:2106.13873, 2003.06962) relates autocorrelation constants to sharp
Fourier constants, yielding explicit perturbation necessary conditions.
Both bypass Lasserre entirely and could improve $C_{1a}$.

---

## Technique A. White's Fourier-to-convex transform (Erdős min-overlap)

**Source:** E. P. White, "Erdős' minimum overlap problem," arXiv:2201.05704
(Acta Arith. 208, 2023). Bound pushed to $M(n)/n > 0.379005$.

**Core idea.**

- The minimum-overlap problem is structurally an $L^\infty$ extremal
  problem on indicator autocorrelations.
- White uses *elementary Fourier analysis* (no SDP, no Lasserre) to translate
  the combinatorial $\min\max$ into a **convex program on Fourier
  coefficients** of a density. The constraints are reproducing-kernel-style
  inequalities on $\hat f$.

**Why it might help $C_{1a}$.** Our Lasserre pipeline encodes positivity of
$M_W$ in the *time* domain. White's dual lives on $\hat f$: nonnegativity of
$f$ is the hard constraint but $\|f*f\|_\infty = \|\hat f^2\|_\infty$-type
rewrites allow *Beurling-Selberg majorants of $\hat f^2$* to give
certified lower bounds on the peak of $f*f$ at a test point. This is exactly
the direction `ideas_cohn_elkies.md` gestures at but with a rigorous
combinatorial precedent.

**Action.** Port White's Lemma 2.1-2.3 (translation to convex LP over Fourier
coefficients of a bounded density) to our target
$\max_{|t|\le 1/2} (f*f)(t)$.

---

## Technique B. Dual-extremizer / Hausdorff-Young framework

**Sources.**
- S. Barnard, S. Steinerberger, arXiv:2106.13873 (classical inequalities for
  autocorrelations/autoconvolutions).
- F. Gonçalves, D. Oliveira, S. Steinerberger, "On optimal autocorrelation
  inequalities on the real line," arXiv:2003.06962 / CPAA 2020.
- K. Chen et al., "Extensions of autocorrelation inequalities,"
  arXiv:2001.02326.

**Core idea.**

- Construct a **suitable dual problem**: replace the primal
  $\inf_f \|f*f\|_\infty$ by a sup over *dual measures* $\mu$ such that
  $\int (f*f)\,d\mu \le 1$ and $\int f = 1$.
- Duality is justified functional-analytically (not via SDP).
- Sharp Hausdorff-Young, Hardy-Littlewood, and rearrangement inequalities
  control the dual, giving rigorous lower bounds on $C_{1a}$.
- Existence of extremizers is proved via concentration-compactness.
- Local *perturbation/variational* conditions (Chen et al., Lemma 2.x):
  extremizers satisfy a minimax equation $f(x) = c$ on support, yielding
  sharp necessary conditions that numerical minimizers should satisfy.

**Why it might help $C_{1a}$.**

- Gives **rigorous lower bounds** without discretization error: the dual
  measure $\mu$ is a *certificate* in the Lagrangian sense. Unlike SDP, no
  floating-point PSD verification is needed - certificate is a finite
  combination of deltas.
- Hausdorff-Young route: $\|f*f\|_\infty \ge \|f\|_2^2$ is weak, but using
  sharp HY on a *restricted* $f$ gives nontrivial bounds. Combined with the
  support constraint $[-1/4,1/4]$ via a Beurling-Selberg minorant of
  $\chi_{[-1/4,1/4]}$, this produces a clean analytic lower bound.
- Variational minimax condition gives an *a priori* structural constraint
  on any candidate minimizer - sharply narrowing the search space our
  cascade/Lasserre needs to certify.

**Action.**

1. Write the MV-type dual problem as a measure-valued LP on $[-1/2,1/2]$
   and check what our current $\mu$ in `lasserre/` actually certifies
   against the sharp HY lower bound.
2. Code up the Chen-Barnard-Steinerberger minimax necessary condition as a
   witness-check on our cascade output.

---

## Also-rans (noted but lower priority)

- **Yu's B_2[g] theorem** (Gang Yu, Illinois J. Math, *An upper bound for
  B_2[g] sets*): the autoconvolution sup lemma that Martin-O'Bryant sharpened
  to $0.631 \|f\|_1^2/I$. The proof is elementary analytic (Cauchy-Schwarz
  + pigeonhole on an explicit test function). Improving the test function
  gave the MV $1.28$ step; further improvement via higher-moment
  pigeonhole is unexplored but likely marginal (< $10^{-3}$).
- **Beurling-Selberg majorants/minorants of $\chi_{[-1/4,1/4]}$**
  (Vaaler 1985, Carneiro-Littmann 2013 Gaussian subordination): these are
  the natural dual witnesses once the problem is recast in Fourier.
- **Cilleruelo-Ruzsa-Vinuesa** (Adv. Math. 2010, arXiv:0909.5024):
  disproves Schinzel-Schmidt; their explicit counterexample is a
  *piecewise polynomial* built from arcsine-weighted indicators - the
  functional form is a concrete template for warm-starting our cascade.
- **Bi-Sidon / Ruzsa's problem** (Combinatorica 2025):
  new $L^\infty$-flavoured bounds but on difference sets, not obviously
  portable.
- **Gradient/Adam-with-noise** (Boyer-Li 2506.16750, 2508.02803,
  2602.07292): pure numerics, orthogonal to our rigor track.
- **AlphaEvolve** (Novikov et al. 2506.13131): LLM-guided local search;
  no technique insight beyond "good initializations matter."

## Sources

- https://arxiv.org/abs/2201.05704  (White, Erdős min-overlap)
- https://arxiv.org/abs/2106.13873  (Barnard-Steinerberger)
- https://arxiv.org/abs/2003.06962  (Gonçalves-Oliveira-Steinerberger)
- https://arxiv.org/abs/2001.02326  (Chen et al., perturbation)
- https://arxiv.org/abs/0907.1379   (Matolcsi-Vinuesa)
- https://arxiv.org/abs/2210.16437  (White, L^2 autoconvolution)
- https://arxiv.org/abs/0909.5024   (Cilleruelo-Ruzsa-Vinuesa)
- https://www.math.kent.edu/~yu/research/sidon.pdf  (Yu, B_2[g])
- https://personal.math.ubc.ca/~gerg/papers/downloads/SAAANT.pdf  (Martin-O'Bryant)
- https://arxiv.org/abs/2506.16750  (Boyer-Li)
- https://arxiv.org/abs/2508.02803  (Further improvements)
- https://arxiv.org/abs/2602.07292  (128 digits)
