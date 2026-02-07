# CLAUDE.md — Sidon Autocorrelation Project

## Active Work

All work is in the `exploration/` folder. The repo has been cleaned — no legacy `src/`, `tests/`, or `run.py`.

## Repository Structure

```
exploration/
  logsumexp_optimizer.ipynb    # Best method: LSE continuation + Polyak (best: 1.5092)
  primal_optimizer.ipynb       # Polyak, L-BFGS-B, peak redistribution methods
  joint_edge_height_optimizer.ipynb  # Joint edge+height optimization (non-uniform grid)
  adaptive_grid_optimizer.ipynb     # Adaptive grid refinement (negative result)
  sdp_certification.ipynb     # Fourier + Spatial SDP (negative result, answers K3)
  lasserre_level2.ipynb       # Lasserre hierarchy level-2 SDP
  lower_bound_sdp.ipynb       # Dual lower bound via PSD/copositivity relaxation
  verifier.ipynb              # Independent verification of solutions
  results.md                  # Consolidated results, analysis, and key unknowns
  best_solutions.json         # Saved best solutions for warm-starting
  prompts/                    # LLM prompts used during exploration
```

## Project Overview

This project investigates the constant $C_{1a}$, defined as the infimum of $\|f*f\|_\infty / \|f\|_1^2$ over nonneg $f \in L^1[-1/4, 1/4]$. Current bounds: $C_{1a} \in [1.2802, 1.5029]$, gap $\approx 0.223$.

## Key Sources

- **[MV10]** Matolcsi & Vinuesa (2010): $C_{1a} \leq 1.5098$, step functions with $n=208$, LP iteration.
- **[AE25]** AlphaEvolve (Tao et al., 2025): $C_{1a} \leq 1.5032$, LLM-evolved cubic backtracking, $n=50$.
- **[TTT26]** TTT-Discover (Yuksekgonul et al., Jan 2026): $C_{1a} \leq 1.50286$, RL-at-test-time, 30,000 pieces.

## Fundamental Bottlenecks

1. **Non-convexity of $\|f*f\|_\infty$** (S1): Supremum of quadratic forms — non-convex. All methods find Clarke stationary points only.
2. **No structural characterization of extremizer** (S2): No necessary conditions beyond $f \geq 0$, $\int f = 1$.
3. **No dual certificate** (S3): All upper bound methods are purely primal. No dual witness.
4. **Peak-locking** (S4): Gradient methods reinforce the current argmax, trapping the optimizer.

## Current Assessment

Five independent paradigms converge to $C_{1a} \approx 1.50$. Upper bound likely within ~0.01 of optimal on uniform grids. ~98% of the gap is on the **lower bound side**.

## Key Unknowns

- **K1**: Is $C_{1a}$ truly $\approx 1.50$? (Open — weight of evidence says yes)
- **K2**: Boundary singularity at $\pm 1/4$? (Open — adaptive grid test was inconclusive)
- **K3**: SDP relaxation tight? (**Resolved — NO.** Shor+RLT gives $2P/(2P-1) \to 1$)
- **K4**: Moment/SOS convexification? (Open — most promising unexploited direction)
- **K5**: Fourier kernel ceiling? (Open — estimated ~1.276, essentially saturated)

## Recommended Attack Priorities

1. **Fourier-domain SDP** (sine+cosine basis + Fejér-Riesz): highest risk/reward
2. **Non-uniform grid experiments**: free-knot splines, geometric grids near $\pm 1/4$
3. **Euler-Lagrange analysis**: necessary conditions for extremizer
4. **Warm-start polishing**: LP/prox-linear from [AE25]/[TTT26] solutions
