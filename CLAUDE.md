# CLAUDE.md — Sidon Autocorrelation Project

## Active Work

We are **only working on notebooks in the `exploration/` folder**. The rest of the repo (`src/`, `run.py`, `tests/`, etc.) is ancient artifact code from earlier experimentation. It may still contain useful reference implementations, but it is **not** the focus of current work — do not modify it unless explicitly asked.

## Project Overview

This project investigates the constant $C_{1a}$, defined as the infimum of $\|f*f\|_\infty / \|f\|_1^2$ over nonneg $f \in L^1[-1/4, 1/4]$. Current bounds: $C_{1a} \in [1.2802, 1.5029]$, gap $\approx 0.223$.

## Key Sources

- **[MV10]** Matolcsi & Vinuesa (2010): $C_{1a} \leq 1.5098$, step functions with $n=208$, LP iteration.
- **[AE25]** AlphaEvolve (Tao et al., 2025): $C_{1a} \leq 1.5032$, LLM-evolved cubic backtracking, $n=50$.
- **[TTT26]** TTT-Discover (Yuksekgonul et al., Jan 2026): $C_{1a} \leq 1.50286$, RL-at-test-time, 30,000 pieces.
- **[KB]** Computational notebook: ~8 optimization methods (Polyak, L-BFGS, NTD, prox-linear, LP, SDP, TV surrogate, Gaussian mixture) on the discretized problem up to $P=1000$.

## Fundamental Bottlenecks

1. **Non-convexity of $\|f*f\|_\infty$** (S1): The objective is a supremum of quadratic forms — intrinsically non-convex. All methods converge to Clarke stationary points, not guaranteed global minima.
2. **No structural characterization of the extremizer** (S2): No necessary conditions on the extremal $f$ beyond $f \geq 0$, $\int f = 1$. No conjectured closed form.
3. **No dual certificate / global optimality gap** (S3): All upper bound methods are purely primal. No dual witness bounds the gap.
4. **Peak-locking** (S4): Gradient methods reinforce the current $\arg\max_t(f*f)(t)$, trapping the optimizer.

## Current Assessment

Five independent optimization paradigms all converge to $C_{1a} \approx 1.50$. The upper bound is likely within $\sim 0.01$ of optimal for step functions on uniform grids. The overwhelmingly dominant source of slack ($\sim 98\%$ of the gap) is on the **lower bound side**.

## Recommended Attack Priorities (for upper bound improvement)

1. **SDP certification at moderate $P$ (50-200)**: Bound the local-vs-global gap. If SDP bound $\geq 1.49$, confirms $C_{1a} \approx 1.50$.
2. **Non-uniform grid experiments**: Test geometrically-refined grids near $\pm 1/4$ to check for boundary singularity in the extremizer.
3. **Fourier-domain SDP formulation**: Attempt Fejer-Riesz / moment-based convexification — highest risk, highest reward.
4. **Euler-Lagrange analysis**: Derive necessary conditions for the extremizer to constrain search space.
5. **Warm-start polishing**: Run LP iteration or prox-linear from best known solutions ([AE25]/[TTT26]).

## Key Unknowns

- **K1**: Is $C_{1a}$ truly $\approx 1.50$ or could it be much lower?
- **K2**: Does the extremizer have a boundary singularity at $\pm 1/4$?
- **K3**: Is the SDP relaxation tight at small $P$?
- **K4**: Can the $L^\infty$ problem be convexified via moment/SOS formulation?
- **K5**: What is the exact ceiling of the Fourier kernel lower bound method?

## Unexploited High-Signal Ideas

- SDP relaxation of polynomial optimization (suggested independently by [MV10], [AE25], [KB])
- Dual-informed primal search (suggested by [MV10], [AE25], [TTT26] — never pursued)
- Fourier-domain reformulation for the upper bound (only used for lower bounds so far)
- Adaptive / non-uniform discretization (never explored; all published work uses uniform grids)
- $L^p$ smooth approximation to $L^\infty$ (controlled error, smoother landscape)
