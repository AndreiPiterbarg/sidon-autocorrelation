# Week 4 Report: LogSumExp Continuation for Minimizing Peak Autoconvolution

## Problem Statement

### What are you optimizing?

We seek to minimize the **peak autoconvolution** of a nonnegative function $f$ supported on $[-1/4, 1/4]$ with $\int f = 1$. Formally, we are computing upper bounds on the constant $C_{1a}$, defined as:

$$C_{1a} = \inf_{f \geq 0,\; \int f = 1} \max_{|t| \leq 1/2} (f * f)(t)$$

where $(f * f)(t) = \int f(x) f(t-x)\,dx$ is the autoconvolution. We discretize $f$ as a step function on $P$ equal-width bins, reducing this to a finite-dimensional optimization over the probability simplex: $\min_{x \in \Delta^{P-1}} \max_k c_k$, where $c_k = 2P \sum_{i+j=k} x_i x_j$ are the discrete autoconvolution coefficients.

### Why does this problem matter?

This problem appears as one of the unsolved mathematical optimization problems in [Davis et al.'s optimization constants repository](https://github.com/teorth/optimizationproblems) as an open problem (this was one of the project ideas Professor Damek Davis suggested). The current bounds are $1.2802 \leq C_{1a} \leq 1.5029$, a gap of ~0.22. Recent AI-driven methods (AlphaEvolve, TTT-Discover) have pushed the upper bound, but classical optimization has shown it can compete without LLMs or TPU clusters.

### How will you measure success

The current best upper bound is $1.5029$, achieved by researchers at Stanford University, Nvidia, and Together AI. The dream goal is to reduce this upper bound and hence produce a new tightest bound. A more realistic goal is to beat Professor Davis's best upper bound ($1.50972$).|

### What are your constraints?

- **Feasibility:** $x \geq 0$ and $\sum_i x_i = 1$ (probability simplex).
- **Computational:** All optimization runs on CPU only. The $O(P^2)$ autoconvolution cost limits scalability beyond $P \approx 1000$.
- **Non-convexity:** The objective $\max_k c_k$ is non-smooth and non-convex (supremum of quadratic forms), so only local optima are guaranteed.

### What data do you need?

No external data is required. The problem is purely computational. Inputs are the discretization level $P$ and algorithm hyperparameters (beta schedule, iteration counts, number of restarts).

### What could go wrong?

- **Peak-locking:** Gradient methods reinforce the current argmax location, trapping the optimizer in suboptimal basins.
- **Beta schedule sensitivity:** The LogSumExp continuation path depends heavily on the schedule. Too aggressive causes basin-trapping; too gentle wastes compute.
- **Local optima:** Multiple restarts mitigate this but cannot guarantee the global minimum. The problem has no known convex reformulation.

---

## Technical Approach

### Mathematical formulation

**Objective function:** $\min_{x \in \Delta^{P-1}} \max_k c_k$ where $c_k = 2P \sum_{i+j=k} x_i x_j$.

**Smooth surrogate:** Replace the non-smooth $\max$ with the LogSumExp approximation:

$$\text{LSE}_\beta(c) = \frac{1}{\beta} \log \sum_k \exp(\beta \, c_k)$$

which satisfies $\max_k c_k \leq \text{LSE}_\beta(c) \leq \max_k c_k + \frac{\log n}{\beta}$. The gradient distributes across near-peak positions via softmax weights, breaking peak-locking.

**Constraints:** Nonnegativity and normalization are enforced by projecting onto the probability simplex after each gradient step (Duchi et al., 2008).

### Algorithm choice and justification

We use a **two-phase hybrid** approach:

1. **Phase 1 -- LSE Nesterov Continuation:** Nesterov accelerated gradient descent with Armijo line search on the smooth $\text{LSE}_\beta$ surrogate. A 15-stage beta schedule ($\beta \in \{1, 2, 4, \ldots, 2000\}$) gradually sharpens the approximation from smooth to near-exact. This finds a good basin without peak-locking.

2. **Phase 2 -- Polyak Subgradient Polish:** The Polyak subgradient method on the true (non-smooth) $\max_k c_k$ objective refines within the basin found by Phase 1, using adaptive step sizes $\alpha_t = (f(x_t) - f^*_{\text{target}}) / \|g_t\|^2$.

Multiple random restarts (30 per run) are parallelized across CPU cores via `joblib`.

### Implementation strategy

All inner-loop math is compiled to machine code via **Numba JIT** (`@nb.njit`) for ~50--100x speedup over pure Python. Key compiled functions: simplex projection, autoconvolution coefficients, LogSumExp/softmax, LSE objective and gradient, and Armijo line search. Parallelism across restarts uses `joblib.Parallel`.

### Validation methods

- **Gradient verification:** Analytical JIT gradient checked against central finite differences (max error < $10^{-4}$).
- **Known analytical test cases:** Uniform distribution (peak = 1.0), Dirac-like concentration (peak = 2P), simplex constraint satisfaction.
- **Exact vs. fast evaluation:** $O(P^2)$ breakpoint evaluation cross-checked against discrete autoconvolution coefficients.

### Resource requirements

- CPU-only execution (no GPU required).
- Memory: < 100 KB peak for a single hybrid restart at P=200.
- Time: scales roughly as $O(P^2 \times \text{iters} \times \text{restarts})$; a full sweep over P=50,100,200,500 with 30 restarts completes in minutes on a modern multi-core CPU.

---

## Initial Results

### Evidence the implementation works

All validation tests pass:
- **Gradient check:** Analytical gradient matches finite differences to $< 10^{-4}$ relative error.
- **Uniform distribution test:** Correctly returns peak = 1.0 at $t = 0$.
- **Dirac concentration test:** Correctly returns peak = $2P$ for a single-bin delta.
- **Simplex constraints:** All optimized solutions satisfy $\sum x_i = 1$ and $x_i \geq 0$ to machine precision.
- **Exact vs. fast agreement:** Breakpoint-exact and discrete coefficient evaluations agree to < 1% relative error.

### Performance metrics

| P | Peak Autoconvolution | Time (s) | Literature Best |
|---|---------------------|----------|----------------|
| 50 | ~1.520 | ~5 | 1.5029 |
| 100 | ~1.515 | ~15 | 1.5029 |
| 200 | ~1.512 | ~45 | 1.5029 |
| 500 | ~1.510 | ~180 | 1.5029 |

The optimizer consistently finds solutions improving with increasing $P$, showing convergence toward the literature values. The Polyak polish phase improves over the LSE phase by ~0.001--0.005 on average across restarts.

### Current limitations

1. **Gap to state-of-the-art:** Our best results at P=500 are still ~0.007 above the literature best (1.5029), which was obtained with 30,000 pieces and specialized AI-driven search.
2. **Uniform grid only:** We discretize on a uniform grid, which may not efficiently capture boundary singularities near $\pm 1/4$.
3. **Scalability:** The $O(P^2)$ per-iteration cost makes P > 1000 expensive without FFT-based convolution.
4. **Restart variance:** Standard deviation across restarts can be significant (0.001--0.01), indicating sensitivity to initialization.

### Resource usage

- Per-operation timings (P=200): autoconvolution ~50 $\mu$s, LSE gradient ~100 $\mu$s, exact evaluation ~50 ms.
- Memory footprint: < 100 KB peak per restart at P=200.
- Full sweep (P=50,100,200,500, 30 restarts each): a few minutes total on a multi-core CPU.

### Unexpected challenges

- **Beta schedule tuning** required significant manual experimentation. Small changes in the schedule can shift results by 0.005+.
- **Numba compilation overhead** on first call (~2--5s) needs explicit warm-up to avoid polluting timing measurements.
- **Numerical stability** of LogSumExp at large $\beta$ (> $10^4$) required careful max-subtraction in the implementation.

---

## Next Steps

### Immediate improvements needed

1. **Scale up P:** Run at P=750 and P=1000 with more restarts and extended iterations, warm-starting from lower-P solutions.
2. **Warm-start cascade:** Upsample solutions from low P to high P (interpolate simplex weights), then polish. This is far more efficient than cold starts at high P.
3. **Adaptive beta schedule:** Replace the fixed geometric schedule with one that adapts based on gradient norm or objective improvement rate.

### Technical challenges to address

- **Non-uniform grids:** Free-knot splines or geometric grids concentrated near $\pm 1/4$ to capture potential boundary singularities, which uniform grids miss.
- **FFT-based convolution:** Replace the $O(P^2)$ direct convolution with $O(P \log P)$ FFT to enable P > 1000.
- **Better initialization:** Test Gaussian, bimodal, symmetric, and boundary-heavy initializations beyond random Dirichlet draws to improve restart coverage.

### Questions needing help

- Is the true extremizer expected to have boundary singularities at $\pm 1/4$? Adaptive grid tests have been inconclusive.
- Can moment/SOS or Fourier-domain SDP methods provide a tighter lower bound to narrow the gap from the current 1.2802?
- What is the optimal beta continuation schedule -- is there a principled way to choose it (e.g., via homotopy theory)?

### Alternative approaches to try

- **Fourier-domain SDP** (sine+cosine basis + Fejer-Riesz): Highest risk/reward approach for dual certificates.
- **Euler-Lagrange analysis:** Derive necessary conditions for the extremizer to guide the discretization.
- **Joint edge+height optimization:** Optimize bin edges and heights simultaneously (non-uniform grid) instead of fixing a uniform grid.

### What I have learned so far

- The LogSumExp continuation effectively breaks peak-locking by distributing gradients across near-peak positions via softmax.
- The two-phase hybrid (smooth basin-finding + non-smooth polishing) consistently outperforms either phase alone.
- Numba JIT is essential for performance -- the compiled inner loops are 50--100x faster than pure NumPy.
- Five independent optimization paradigms converge to $C_{1a} \approx 1.50$, giving strong evidence that the upper bound is close to optimal on uniform grids. The remaining gap (~0.22) is overwhelmingly on the lower bound side.
