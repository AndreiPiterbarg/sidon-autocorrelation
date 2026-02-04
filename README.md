# Improving the Upper Bound on the Sidon Autocorrelation Constant (C₁ₐ)

> **Current bounds:** 1.2802 ≤ C₁ₐ ≤ 1.5029
>
> **Goal:** Find a concrete function that pushes the upper bound below 1.5029.

## Problem Statement

For any nonnegative function $f : \mathbb{R} \to \mathbb{R}_{\geq 0}$ supported on $[-1/4, 1/4]$ with $\int f = 1$:

$$\max_{|t| \le 1/2} (f * f)(t) \;\geq\; C_{1a}$$

We solve the dual: find $f^*$ that **minimizes** the autoconvolution peak.

$$\min_{f \,\geq\, 0,\;\int f = 1} \;\max_{|t| \le 1/2} \int_{\mathbb{R}} f(x)\, f(t - x)\, dx$$

Any $f^*$ achieving a value below 1.5029 is an improved upper bound. The function itself is the proof — the result is machine-verifiable.

## Context

This constant appears in [Tao et al.'s optimization constants repository](https://teorth.github.io/optimizationproblems/constants/1a.html) and connects to the asymptotic size of Sidon sets in additive combinatorics. Recent progress on the upper bound has come from AI-driven search (AlphaEvolve → 1.503164, ThetaEvolve → 1.503133, TTT-Discover → 1.5029), but [Boyer–Li (2025)](https://arxiv.org/abs/2506.16750) showed that classical optimization (simulated annealing + gradient descent on step functions) can compete on related autoconvolution problems without LLMs or TPU clusters.

## Core Difficulty: Peak-Locking

The objective $\max_t (f*f)(t)$ creates **peak-locking** under gradient descent: whichever $t$ currently achieves the max gets reinforced by gradient updates, trapping the search in local basins. This is explicitly flagged as an open problem in [arXiv:2508.02803](https://arxiv.org/abs/2508.02803). Circumventing peak-locking is the central technical challenge.

## Research Plan

### Phase 1 — Literature Review

Read and extract the technical core from these papers. For each, identify: (a) how $f$ is parameterized, (b) the optimization method, (c) the structure of the resulting near-optimal function, (d) stated limitations.

| Paper | Why it matters |
|-------|---------------|
| [Matolcsi–Vinuesa (2010)](https://arxiv.org/abs/0907.1379) | Established the 1.5099 upper bound via explicit step functions. Need to understand the shape and symmetry of their extremal $f$. |
| [Cloninger–Steinerberger (2017)](https://arxiv.org/abs/1403.7988) | Finite reduction framework for the lower bound side. Bottleneck is runtime — relevant if we later attempt lower bound improvements. |
| [White (2022)](https://arxiv.org/abs/2210.16437) | Turned the **L² autoconvolution** variant into a **convex program** via Fourier analysis. Central question: does this convexification transfer to the L∞ objective? |
| [Boyer–Li (2025)](https://arxiv.org/abs/2506.16750) | Beat AlphaEvolve on the second autoconvolution inequality using coarse-to-fine gradient + simulated annealing with 575-interval step functions. Directly relevant pipeline. |
| [arXiv:2508.02803](https://arxiv.org/abs/2508.02803) | Documents the peak-locking phenomenon and why the L∞-numerator case specifically resists gradient search. |

**Of particular interest:** the approximate shapes of $f$ that produce the smallest constants. Understanding this structure (where mass concentrates, what symmetries appear, what $(f*f)$ looks like) determines whether smarter parameterizations are worth pursuing.

### Phase 2 — Piecewise Constant Baseline (v0.1)

Start with the simplest possible approach. This is what every successful method has used as a foundation.

**Parameterization:** Divide $[-1/4, 1/4]$ into $N$ bins with heights $h_i \geq 0$, normalized so total mass equals 1. **Do not hardcode uniform bins.** The code should accept arbitrary bin edges from the start — we will want to concentrate points in regions of fine structure once we see what the optimal $f$ looks like. Start with uniform spacing for simplicity, but the implementation must support non-uniform meshes without refactoring.

**Convolution:** The autoconvolution of a piecewise constant function on non-uniform bins is still closed-form: each bin-pair contributes a trapezoidal piece whose area depends on the two bin widths and heights. This is $O(N^2)$ via direct summation — no quadrature needed.

**Optimizer:** Use `scipy.optimize.dual_annealing` or `scipy.optimize.differential_evolution` (global search, handles non-convexity). Not plain `scipy.optimize.minimize` — the landscape is non-convex and L-BFGS-B alone will get trapped.

**Target:** $N = 50$ bins, uniform. Reproduce something close to the known 1.5099 bound. Visualize $f^*$ and $(f^* * f^*)(t)$.

- [ ] Implement piecewise constant $f$ with arbitrary bin edges and heights as variables
- [ ] Implement direct autoconvolution computation (closed-form, supports non-uniform bins)
- [ ] Minimize peak of autoconvolution using `dual_annealing` with box constraints $h_i \geq 0$
- [ ] Visualize: plot $f^*$, plot $(f^* * f^*)(t)$ on $[-1/2, 1/2]$, annotate peak value and location
- [ ] Compare result against known 1.5099 (Matolcsi–Vinuesa)

### Phase 3 — Scale and Coarse-to-Fine (v0.2)

Scale to higher resolution and adopt Boyer–Li's coarse-to-fine strategy.

**FFT convolution:** At $N > 200$, switch from direct $O(N^2)$ to FFT-based $O(N \log N)$. For uniform bins this is standard FFT with zero-padding. For non-uniform bins, evaluate $f$ on a fine uniform grid (this is $O(M)$ for piecewise constant — just a bin lookup) and then FFT on that grid. Alternatively, use the [Non-uniform FFT](https://en.wikipedia.org/wiki/Non-uniform_discrete_Fourier_transform) (Type 1: non-uniform spatial → uniform frequency) via `finufft` to compute $\hat{f}(\xi_k)$, then $|\hat{f}|^2$, then inverse FFT. Either way, non-uniform bins do not force $O(N^2)$.

**Adaptive mesh refinement:** After Phase 2, inspect $f^*$. Identify regions with sharp features (spikes, transitions). Rebuild the mesh with higher density there — e.g., 2–3× more bins near the spike, fewer in flat regions. Re-optimize on the refined mesh using the Phase 2 solution as initialization.

**Coarse-to-fine:** Optimize at $N = 50$ → upsample to $N = 200$ → refine → upsample to $N = 500$ → refine. This avoids the curse of starting a global search in 500 dimensions.

**Peak-locking diagnosis:** Run gradient-based refinement (L-BFGS-B) from the global-search solution. Track which $t$ achieves the max across iterations. If the peak location locks prematurely, we have empirical confirmation of peak-locking and motivation for Phase 4.

- [ ] Implement FFT-based autoconvolution (uniform bins: standard FFT; non-uniform bins: fine-grid evaluation + FFT or NUFFT via `finufft`)
- [ ] Implement adaptive mesh refinement: inspect $f^*$, concentrate bins in high-structure regions
- [ ] Implement coarse-to-fine pipeline: global search at low $N$ → upsample → local refinement at high $N$
- [ ] Benchmark: what bound does this achieve at $N = 200, 500, 1000$?
- [ ] Visualize peak-locking: plot $\arg\max_t (f*f)(t)$ across optimization iterations
- [ ] Compare result against current best 1.5029 (TTT-Discover)

### Phase 4 — Fourier Parameterization (v0.3)

This is the core technical bet to circumvent peak-locking.

**Key idea:** If $\hat{f}(\xi) = \int f(x) e^{-2\pi i \xi x} dx$, then $(f*f)(t)$ is the inverse Fourier transform of $|\hat{f}(\xi)|^2$. Parameterize $f$ via truncated Fourier coefficients rather than pointwise values. The hypothesis is that smooth spectral coefficients don't couple to a single spatial peak the way bin heights do, mitigating peak-locking.

**Constraints in Fourier space:**
- $\int f = 1$ becomes $\hat{f}(0) = 1$
- $f \geq 0$ is the hard constraint — options: penalty method ($\lambda \int \min(f, 0)^2$), projection after inverse FFT, or SDP relaxation via Bochner's theorem

**Precedent:** White (2022) used exactly this Fourier convexification for the $L^2$ autoconvolution variant with great success. The question is whether it extends to the $L^\infty$ objective.

- [ ] Parameterize $f(x) = \sum_{k=0}^{K} a_k \cos(2\pi k x / (1/2))$ with nonnegativity penalty
- [ ] Compare optimization landscape: same global search methods, Fourier params vs. bin heights
- [ ] Test hypothesis: does peak-locking diminish in Fourier parameterization?
- [ ] If yes, push resolution ($K = 50, 100, 200$)
- [ ] If no, try CMA-ES or population-based search in the step-function parameterization at high $N$

### Phase 5 — Verify and Submit

- [ ] Verify any candidate $f^*$ at high numerical precision (multiprecision arithmetic, e.g. `mpmath`)
- [ ] Cross-validate: compute $(f^* * f^*)(t)$ on a fine grid ($>10{,}000$ points) and confirm the peak
- [ ] If the bound improves: write up the function and method, submit to the [optimization constants repo](https://github.com/teorth/optimizationproblems)

## Success Criterion

Find an explicit $f^*$ (specified as a step function or truncated Fourier series) such that:

$$\max_{|t| \le 1/2} (f^* * f^*)(t) < 1.5029$$

One function. One verifiable number.

## References

- [Tao et al., Optimization Constants Repo](https://github.com/teorth/optimizationproblems)
- [Tao, Blog Post (Jan 2026)](https://terrytao.wordpress.com/2026/01/22/a-crowdsourced-repository-for-optimization-constants/)
- [Matolcsi–Vinuesa (2010), arXiv:0907.1379](https://arxiv.org/abs/0907.1379)
- [Cloninger–Steinerberger (2017), arXiv:1403.7988](https://arxiv.org/abs/1403.7988)
- [White (2022), arXiv:2210.16437](https://arxiv.org/abs/2210.16437)
- [Boyer–Li (2025), arXiv:2506.16750](https://arxiv.org/abs/2506.16750)
- [Jaech et al. (2025), arXiv:2508.02803](https://arxiv.org/abs/2508.02803)
- [AlphaEvolve, arXiv:2511.02864](https://arxiv.org/abs/2511.02864)
