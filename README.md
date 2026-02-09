# Improving the Upper Bound on the Sidon Autocorrelation Constant (C_{1a})

> **Current bounds:** 1.2802 <= C_{1a} <= 1.5029
>
> **Our best result:** C_{1a} <= 1.5055 (P=1500, LogSumExp hybrid + cloud compute)

## Problem Statement

For any nonnegative function $f : \mathbb{R} \to \mathbb{R}_{\geq 0}$ supported on $[-1/4, 1/4]$ with $\int f = 1$:

$$\max_{|t| \le 1/2} (f * f)(t) \;\geq\; C_{1a}$$

We solve the dual: find $f^*$ that **minimizes** the autoconvolution peak.

$$\min_{f \,\geq\, 0,\;\int f = 1} \;\max_{|t| \le 1/2} \int_{\mathbb{R}} f(x)\, f(t - x)\, dx$$

Any $f^*$ achieving a value below 1.5029 is an improved upper bound. The function itself is the proof -- the result is machine-verifiable.

## Context

This constant appears in [Tao et al.'s optimization constants repository](https://teorth.github.io/optimizationproblems/constants/1a.html) and connects to the asymptotic size of Sidon sets in additive combinatorics. Recent progress on the upper bound has come from AI-driven search (AlphaEvolve -> 1.503164, ThetaEvolve -> 1.503133, TTT-Discover -> 1.5029), but [Boyer-Li (2025)](https://arxiv.org/abs/2506.16750) showed that classical optimization (simulated annealing + gradient descent on step functions) can compete on related autoconvolution problems without LLMs or TPU clusters.

## Method

Our approach is a **two-phase hybrid optimizer**:

1. **Phase 1 -- LogSumExp Continuation:** Replace the non-smooth max with a smooth LogSumExp surrogate. Schedule beta from 1 to 3000 across ~21 stages, using Nesterov accelerated gradient descent with Armijo line search. This breaks peak-locking by distributing gradients via softmax weights.

2. **Phase 2 -- Polyak Subgradient Polish:** Refine within the basin found by Phase 1 using adaptive Polyak step sizes on the true (non-smooth) objective.

All inner-loop math is compiled to machine code via **Numba JIT** for 50-100x speedup. Multiple restarts are parallelized via `joblib`.

## Results

| P (bins) | Peak Autoconvolution | Method |
|----------|---------------------|--------|
| 200 | 1.5092 | LSE hybrid (local) |
| 500 | 1.5069 | Cloud pipeline |
| 1000 | 1.5057 | Cloud pipeline |
| 1500 | **1.5055** | Cloud pipeline |

## Repository Structure

```
sidon-autocorrelation/
├── README.md                          # This file
├── report.md                          # Week 4 report
├── notebooks/
│   └── week4_implementation.ipynb     # Main working implementation
├── prev_attempts/                     # All previous exploration notebooks
│   ├── logsumexp_optimizer.ipynb      # Best method: LSE + Polyak hybrid
│   ├── primal_optimizer.ipynb         # Polyak, L-BFGS-B, peak redistribution
│   ├── joint_edge_height_optimizer.ipynb  # Joint edge+height optimization
│   ├── free_knot_alternating.ipynb    # Free-knot spline optimization
│   ├── adaptive_grid_optimizer.ipynb  # Adaptive grid (negative result)
│   ├── sdp_certification.ipynb        # SDP relaxation (negative result)
│   ├── lasserre_level2.ipynb          # Lasserre hierarchy SDP
│   ├── lower_bound_sdp.ipynb          # Dual lower bound relaxation
│   ├── verifier.ipynb                 # Independent solution verification
│   ├── sidon_core.py                  # Core optimization library
│   ├── sidon_cloud.py                 # Modal cloud compute pipeline
│   ├── results.md                     # Consolidated results & analysis
│   └── best_solutions.json            # Saved solutions for warm-starting
├── tests/
│   └── test_basic.py                  # Basic validation tests
├── docs/
│   ├── development_log.md             # Progress & decisions log
│   └── llm_exploration/
│       └── week4_log.md               # LLM conversation logs
├── cloud_results/                     # Raw output from Modal cloud runs
└── prompts_IGNORE/                    # LLM prompts used during exploration
```

## Running Tests

```bash
pip install numpy numba pytest joblib
pytest tests/ -v
```

## References

- [Tao et al., Optimization Constants Repo](https://github.com/teorth/optimizationproblems)
- [Tao, Blog Post (Jan 2026)](https://terrytao.wordpress.com/2026/01/22/a-crowdsourced-repository-for-optimization-constants/)
- [Matolcsi-Vinuesa (2010), arXiv:0907.1379](https://arxiv.org/abs/0907.1379)
- [Cloninger-Steinerberger (2017), arXiv:1403.7988](https://arxiv.org/abs/1403.7988)
- [White (2022), arXiv:2210.16437](https://arxiv.org/abs/2210.16437)
- [Boyer-Li (2025), arXiv:2506.16750](https://arxiv.org/abs/2506.16750)
- [Jaech et al. (2025), arXiv:2508.02803](https://arxiv.org/abs/2508.02803)
- [AlphaEvolve, arXiv:2511.02864](https://arxiv.org/abs/2511.02864)
