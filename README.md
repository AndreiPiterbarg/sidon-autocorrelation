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

## References

- [Tao et al., Optimization Constants Repo](https://github.com/teorth/optimizationproblems)
- [Tao, Blog Post (Jan 2026)](https://terrytao.wordpress.com/2026/01/22/a-crowdsourced-repository-for-optimization-constants/)
- [Matolcsi–Vinuesa (2010), arXiv:0907.1379](https://arxiv.org/abs/0907.1379)
- [Cloninger–Steinerberger (2017), arXiv:1403.7988](https://arxiv.org/abs/1403.7988)
- [White (2022), arXiv:2210.16437](https://arxiv.org/abs/2210.16437)
- [Boyer–Li (2025), arXiv:2506.16750](https://arxiv.org/abs/2506.16750)
- [Jaech et al. (2025), arXiv:2508.02803](https://arxiv.org/abs/2508.02803)
- [AlphaEvolve, arXiv:2511.02864](https://arxiv.org/abs/2511.02864)
