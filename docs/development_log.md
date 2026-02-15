# Development Log

## Week 1-2: Problem Setup and Initial Exploration

### Problem formulation
Identified the Sidon autocorrelation constant $C_{1a}$ from [Tao et al.'s optimization constants repository](https://github.com/teorth/optimizationproblems). The constant is defined as:

$$C_{1a} = \inf_{f \geq 0,\; \int f = 1} \max_{|t| \leq 1/2} (f * f)(t)$$

Current bounds: $1.2802 \leq C_{1a} \leq 1.5029$.

### Initial approach
Discretized $f$ as a step function on $P$ equal-width bins, reducing the problem to minimizing the peak autoconvolution over the probability simplex. Implemented and tested several optimization methods:

- **L-BFGS-B** with softmax reparametrization: peak ~1.58 at P=10. Poor performance.
- **Basin hopping + L-BFGS**: peak ~1.575 at P=10. Slightly better but still far from literature.
- **Polyak subgradient method**: peak ~1.516 at P=150. Significant improvement via Numba JIT parallelization (180 restarts). First competitive result.
- **Peak redistribution hybrid**: peak ~1.508 at P=1000. Combined Polyak with heuristic mass redistribution.

### Key decision: Numba JIT
All inner-loop math compiled to machine code via Numba `@njit` for 50-100x speedup over pure NumPy. This made large-scale multi-restart optimization feasible on CPU.

---

## Week 3: LogSumExp Continuation (Breakthrough)

### Insight
The non-smooth $\max_k c_k$ objective causes **peak-locking** -- gradient methods reinforce the current argmax, trapping the optimizer. Replaced $\max$ with the smooth LogSumExp approximation:

$$\text{LSE}_\beta(c) = \frac{1}{\beta} \log \sum_k \exp(\beta c_k)$$

which distributes gradients across near-peak coefficients via softmax weights.

### Two-phase hybrid
1. **Phase 1 -- LSE Nesterov Continuation:** 15-stage beta schedule ($\beta \in \{1, 2, ..., 2000\}$), Nesterov accelerated gradient descent with Armijo line search.
2. **Phase 2 -- Polyak Subgradient Polish:** Refines within the basin found by Phase 1.

### Results
| P | LSE Hybrid | Polyak Only | Improvement |
|---|-----------|-------------|-------------|
| 50 | 1.5218 | 1.5242 | -0.003 |
| 100 | 1.5154 | 1.5202 | -0.006 |
| 200 | **1.5092** | 1.5199 | -0.008 |

This established our **project best of 1.5092** at P=200.

### Negative results
- **Adaptive grid refinement**: Failed badly -- peak went from 1.51 to 1.54. Root cause: point-value interpolation (np.interp) destroyed solution quality when transferring between grids. Width ratios reached 114:1.
- **SDP relaxation**: Shor+RLT gives $2P/(2P-1) \to 1$, useless as a lower bound.
- **Lasserre level-2 (initial SCS runs)**: Tight at P=2-4 but appeared loose at P>=5.

---

## Week 4: Scaling Up and Cloud Compute

### Cloud pipeline (Modal)
Deployed the LSE hybrid optimizer to Modal cloud compute (32-core containers). Ran a 9-round pipeline:
1. Round 1: Strategy tournament at P=200 (12 initialization strategies, 80 restarts each)
2. Rounds 2-8: Progressive upsampling through P=300, 500, 750, 1000, 1500
3. Round 9: Cross-pollination (blending solutions across P values)

### Cloud results
| P | Best Peak | Strategy |
|---|-----------|----------|
| 200 | 1.5097 | cosine_shaped |
| 500 | 1.5069 | warm_perturb |
| 1000 | 1.5057 | warm_perturb |
| 1500 | **1.5055** | warm_perturb |

**New project best: 1.5055** at P=1500.

### Key findings
- **Warm-start dominates**: At high P, only warm-starting from a known good solution finds the best basin. Cold random starts land in bad local minima.
- **Diminishing returns**: P=1000->1500 gained only 0.00015. Extrapolation suggests the method converges to ~1.5055, not 1.5029.
- **Cross-pollination failed**: Blending solutions destroys fine-grained structure.
- **Solutions are asymmetric**: All solutions are 70%+ asymmetric, suggesting the optimizer is finding symmetry-broken local minima.

### Additional explorations
- **Joint edge+height optimization**: Optimized bin edges and heights simultaneously (non-uniform grid). Reached ~1.51 at P=200 -- promising but needs more development.
- **Free-knot alternating optimization**: Alternated between height and edge optimization. Reached ~1.509 but did not beat the cloud results.

### Fundamental bottleneck
Five independent optimization paradigms converge to $C_{1a} \approx 1.50$. The remaining gap to 1.5029 is likely a *basin* problem, not a *resolution* problem. The most promising directions for further progress are structural (SOS/moment methods, Fourier-domain SDP) rather than better optimization on uniform grids.

---

## Lessons Learned

1. **LogSumExp continuation is highly effective** for non-smooth minimax problems. It breaks peak-locking by distributing gradients across near-peak positions.
2. **Numba JIT is essential** for CPU-bound optimization. 50-100x speedup over pure NumPy.
3. **Warm-starting matters more than initialization strategy** at high P. The landscape has exponentially many local minima, and cold starts get trapped.
4. **Grid interpolation is dangerous** for step functions. Mass-conserving transfer (overlap integrals) is the correct approach; point-value interpolation destroys solutions.
5. **Non-convexity is the fundamental barrier.** All methods find only Clarke stationary points. ~98% of the gap between upper and lower bounds is on the lower bound side.

---

## Postscript (February 2026): Lower-Bound Clarification

A CLARABEL-first re-run of level-2 Lasserre (see `prev_attempts/lasserre2_sweep.py`) produced much stronger **discretized** bounds for the $P$-bin problem than earlier SCS-heavy runs.

However, these values are bounds on $\eta_P$ (the discretized minimax value), not directly on continuous $C_{1a}$.

Formally:
- $C_{1a} = \inf_{f \in \mathcal{F}} M(f)$ (continuous feasible class),
- $\eta_P = \inf_{f \in \mathcal{S}_P} M(f)$ with $\mathcal{S}_P \subset \mathcal{F}$,
- level-2 Lasserre computes $LB_P \le \eta_P$.

So high $LB_P$ does **not** by itself imply $C_{1a} \ge LB_P$.
To convert this pipeline into continuous lower bounds, we need a transfer estimate
$\eta_P - C_{1a} \le \varepsilon_P$, giving $C_{1a} \ge LB_P - \varepsilon_P$.
