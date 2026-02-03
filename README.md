# Improving the Upper Bound on the Sidon Autocorrelation Constant (C₁ₐ)

> **Current bounds:** 1.2802 ≤ C₁ₐ ≤ 1.5029
>
> **Goal:** Find a concrete function that pushes the upper bound below 1.5029.

## Problem Statement

For any nonnegative function $f : \mathbb{R} \to \mathbb{R}_{\geq 0}$ supported on $[-1/4, 1/4]$ with $\int f = 1$, we have:

$$\max_{|t| \le 1/2} (f * f)(t) \;\geq\; C_{1a}$$

We are solving the **dual optimization problem**: find $f^*$ that **minimizes** the autoconvolution peak.

$$\min_{f \,\geq\, 0,\;\int f = 1} \;\max_{|t| \le 1/2} \int_{\mathbb{R}} f(x)\, f(t - x)\, dx$$

Any $f^*$ achieving a value below 1.5029 improves the known upper bound. The function itself is the proof — the result is machine-verifiable.

## Context

This constant appears in [Tao et al.'s optimization constants repository](https://teorth.github.io/optimizationproblems/constants/1a.html) and connects to the asymptotic size of Sidon sets in additive combinatorics. Recent progress has come from AI-driven search (AlphaEvolve → 1.503164, ThetaEvolve → 1.503133, TTT-Discover → 1.5029), but [Boyer–Li (2025)](https://arxiv.org/abs/2506.16750) showed that classical optimization (simulated annealing + gradient descent) can compete on related autoconvolution problems.

## Core Difficulty

The objective $\max_t (f*f)(t)$ creates **peak-locking** under gradient descent: whichever $t$ currently achieves the max gets reinforced, trapping the search in local basins. This is explicitly flagged as open in [arXiv:2508.02803](https://arxiv.org/abs/2508.02803). The AI lab approaches sidestep this via evolutionary search at massive scale rather than solving the optimization structure directly.

## Research Plan

### Phase 1 — Literature and Landscape

Read and extract the technical core from these papers. The goal is to understand the structure of known near-optimal functions and the techniques that produced them.

| Paper | Why it matters |
|-------|---------------|
| [Matolcsi–Vinuesa (2010)](https://arxiv.org/abs/0907.1379) | Established the 1.5099 upper bound. Need to understand the shape and Fourier structure of their extremal step functions. |
| [Cloninger–Steinerberger (2017)](https://arxiv.org/abs/1403.7988) | Finite reduction framework for the lower bound. Key insight: the problem can be reduced to finitely many computational cases. Their stated bottleneck is runtime. |
| [White (2022)](https://arxiv.org/abs/2210.16437) | Turned the **L² autoconvolution** variant into a **convex program** via Fourier analysis. The central question is whether this convexification transfers to the L∞ objective. |
| [Boyer–Li (2025)](https://arxiv.org/abs/2506.16750) | Beat AlphaEvolve on the second autoconvolution inequality using coarse-to-fine gradient + simulated annealing with 575-interval step functions. No LLMs. Directly relevant pipeline. |
| [arXiv:2508.02803](https://arxiv.org/abs/2508.02803) | Concurrent work on gradient methods for autoconvolution. Documents the peak-locking phenomenon and why the L∞-numerator case resists gradient search. |

**Deliverable:** For each paper, extract (a) the parameterization of $f$ used, (b) the optimization method, (c) the structure of the resulting near-optimal function, (d) stated limitations.

### Phase 2 — Reproduce and Benchmark

Implement the baseline approaches and verify known results.

- [ ] Discretize $f$ as an $N$-bin step function on $[-1/4, 1/4]$, compute $(f*f)(t)$ via direct convolution
- [ ] Reproduce the Matolcsi–Vinuesa extremal function and verify the 1.5099 bound
- [ ] Implement Boyer–Li's coarse-to-fine pipeline (simulated annealing → gradient refinement) adapted to the $L^\infty$ objective
- [ ] Benchmark: what bound does naive gradient descent + random restarts achieve at $N = 50, 200, 500, 1000$?
- [ ] Profile the peak-locking phenomenon: visualize gradient flow and track which $t$ dominates across iterations

### Phase 3 — Fourier-Space Reformulation

This is the core technical bet. Parameterize $f$ via its Fourier coefficients and exploit:

- $(f * f)(t)$ is the inverse Fourier transform of $|\hat{f}(\xi)|^2$
- $\int f = 1$ becomes $\hat{f}(0) = 1$
- The nonnegativity constraint $f \geq 0$ is the hard part — it becomes a condition on the Fourier coefficients

Key questions to resolve:

- [ ] Can the $L^\infty$ peak objective be relaxed to something convex in Fourier space? (White did this for $L^2$ — investigate whether an analogous dual formulation exists here)
- [ ] Does Fourier parameterization mitigate peak-locking? (The hypothesis: smooth spectral coefficients don't couple to a single peak location the way pointwise values do)
- [ ] What's the right way to handle $f \geq 0$? Options: SDP relaxation via Bochner's theorem (autocovariance must be positive semidefinite), penalty methods, projection onto the nonneg cone after inverse FFT

### Phase 4 — Search and Improve

Based on what Phase 3 reveals, run the actual optimization.

- [ ] If Fourier convexification works: solve the convex program at increasing resolution
- [ ] If not: combine Fourier parameterization with global search (CMA-ES, basin-hopping, or population-based methods) to escape peak-locking
- [ ] Explore hybrid strategies: Fourier-space global search → real-space local refinement
- [ ] Try alternative parameterizations: splines, wavelets, mixtures of bump functions

### Phase 5 — Verify and Document

- [ ] Any candidate $f^*$ must be verified at high numerical precision (multiprecision arithmetic)
- [ ] Cross-validate: compute $(f^* * f^*)(t)$ on a fine grid of $t$ values and confirm the peak
- [ ] If the bound improves: write up the function, the method, and submit to the [optimization constants repo](https://github.com/teorth/optimizationproblems)

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