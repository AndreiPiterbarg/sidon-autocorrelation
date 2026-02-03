# CLAUDE.md

## Project Overview

This project aims to improve the upper bound on the Sidon autocorrelation constant C₁ₐ, currently bounded as 1.2802 ≤ C₁ₐ ≤ 1.5029. The goal is to find a concrete nonnegative function f supported on [-1/4, 1/4] with ∫f = 1 whose autoconvolution peak max_{|t|≤1/2} (f*f)(t) is strictly below 1.5029.

## Problem

Minimize the peak autoconvolution:

    min_{f ≥ 0, ∫f=1} max_{|t|≤1/2} ∫ f(x) f(t-x) dx

Any f* achieving a value below 1.5029 improves the known upper bound. The function itself is the proof — the result is machine-verifiable.

## Key Technical Challenge

**Peak-locking:** Under gradient descent, whichever t currently achieves the max gets reinforced, trapping the search in local basins. This is why AI lab approaches (AlphaEvolve, ThetaEvolve, TTT-Discover) used evolutionary search at scale rather than solving the optimization directly.

## Project Phases

1. **Literature & Landscape** — Read and extract technical core from key papers (Matolcsi-Vinuesa, Cloninger-Steinerberger, White, Boyer-Li, arXiv:2508.02803). For each: parameterization of f, optimization method, structure of near-optimal f, stated limitations.
2. **Reproduce & Benchmark** — Discretize f as N-bin step functions, reproduce known results (Matolcsi-Vinuesa 1.5099 bound), implement Boyer-Li coarse-to-fine pipeline, profile peak-locking.
3. **Fourier-Space Reformulation** — Parameterize f via Fourier coefficients. (f*f)(t) becomes inverse FT of |f̂(ξ)|². Investigate whether L∞ objective can be convexified in Fourier space (White did this for L²). Handle f ≥ 0 via SDP/Bochner's theorem or penalty methods.
4. **Search & Improve** — Run optimization: convex program if Fourier convexification works, otherwise combine Fourier parameterization with global search (CMA-ES, basin-hopping). Explore hybrid Fourier-global → real-space-local strategies.
5. **Verify & Document** — Verify candidates at high numerical precision (multiprecision). Cross-validate on fine grid of t values. Submit improvements to the optimization constants repo.

## Tech Stack

- Python (numerical computation)
- NumPy/SciPy for convolution, FFT, optimization
- Multiprecision arithmetic (mpmath) for verification

## Key Conventions

- f is parameterized as either an N-bin step function on [-1/4, 1/4] or a truncated Fourier series
- Autoconvolution (f*f)(t) is computed via direct convolution or FFT
- All candidate functions must satisfy: f ≥ 0, supported on [-1/4, 1/4], ∫f = 1
- Results are verified by evaluating (f*f)(t) on a fine grid and confirming the peak value

## Success Criterion

Find an explicit f* (step function or truncated Fourier series) such that max_{|t|≤1/2} (f*f*)(t) < 1.5029.

## Key References

- [Tao et al., Optimization Constants](https://teorth.github.io/optimizationproblems/constants/1a.html)
- [Matolcsi-Vinuesa (2010)](https://arxiv.org/abs/0907.1379) — 1.5099 upper bound
- [White (2022)](https://arxiv.org/abs/2210.16437) — L² convexification via Fourier
- [Boyer-Li (2025)](https://arxiv.org/abs/2506.16750) — SA + gradient on step functions, beat AlphaEvolve on related problem
- [arXiv:2508.02803](https://arxiv.org/abs/2508.02803) — Documents peak-locking phenomenon
