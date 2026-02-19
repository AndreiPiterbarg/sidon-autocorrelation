# Improving the Lower Bound on the Sidon Autocorrelation Constant ($C_{1a}$)

> **Current bounds:** $1.2802 \leq C_{1a} \leq 1.5029$
>
> **Goal:** Push the lower bound above 1.2802 by improving the Cloninger-Steinerberger algorithm.
>
> **Current focus:** GPU kernel optimization targeting NVIDIA A100 cloud GPUs — the algorithm is complete; all effort is on making it run as fast as possible.

## Problem Statement

For any nonnegative function $f : \mathbb{R} \to \mathbb{R}_{\geq 0}$ supported on $[-1/4, 1/4]$ with $\int f = 1$:

$$\max_{|t| \le 1/2} (f * f)(t) \;\geq\; C_{1a}$$

The **upper bound** comes from exhibiting explicit functions with small autoconvolution peaks. The **lower bound** requires proving that *no* function can do better than a given threshold — a fundamentally harder task.

The best known lower bound is $C_{1a} \geq 1.2802$, established by [Cloninger & Steinerberger (2017)](https://arxiv.org/abs/1403.7988) via an exhaustive branch-and-prune algorithm. We aim to improve this.

## Approach

We build on the Cloninger-Steinerberger branch-and-prune algorithm:

1. **Partition** $[-1/4, 1/4]$ into $d = 2n$ equal bins and track the mass $a_i = \int_{I_i} f$ on each bin.
2. **Lower bound** $\|f*f\|_\infty$ by the maximum over windowed sums of $a_i a_j$ products (Lemma 1 of [CS17]): for any window of $\ell$ consecutive bins, $\|f*f\|_\infty \geq \frac{1}{4n\ell} \sum a_i a_j$.
3. **Discretize** masses to a grid $B_{n,m}$ (multiples of $1/m$), with correction $C_{1a} \geq b_{n,m} - 2/m - 1/m^2$.
4. **Prune** all grid points via:
   - **Asymmetry argument**: if too much mass is on one side, the bound holds automatically.
   - **Test-value computation**: check all windows; if the best lower bound exceeds the target, the point is pruned.
5. If ALL grid points are pruned, the lower bound is **rigorously proven**.

The original paper reached $n = 24$ (48 bins), $m = 50$, using ~20,000 CPU hours on Yale HPC with GPU acceleration. Our current work focuses on optimized CUDA kernels for NVIDIA A100 cloud GPUs to push to larger parameter regimes.

## Repository Structure

```
sidon-autocorrelation/
├── cloninger-steinerberger/    # Active: branch-and-prune implementation
│   ├── core.py                 # Re-exports all public API
│   ├── compositions.py         # Batched composition generators (Numba JIT)
│   ├── test_values.py          # Autoconvolution + windowed test-value computation
│   ├── pruning.py              # Correction terms, asymmetry threshold, canonical mask
│   ├── solvers.py              # Solvers + fused GPU/Numba kernels
│   └── run.py                  # CLI entry point for running proofs
├── tests/                      # Test suite
│   ├── test_basics.py
│   ├── test_compositions.py
│   ├── test_autoconvolution.py
│   ├── test_integration.py
│   └── test_heavy.py           # Production-scale benchmarks
├── exploration/                # Archived upper-bound notebooks (read-only)
├── cloud/                      # Archived cloud infra (read-only)
├── data/                       # Checkpoints and results
├── docs/                       # Documentation
└── requirements.txt
```

## Running

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt

# Run the branch-and-prune demo
python cloninger-steinerberger/run.py

# Run tests
pytest tests/ -v
```

## Prime Intellect Deployment (A100 + uv)

For full pod provisioning + on-pod execution steps, see:

- `docs/prime_intellect_runbook.md`

Default GPU recommendation in that runbook:

- `A100 (SXM4)` by default
- `A100 (PCIE)` fallback when availability is better

## Context

This constant appears in [Tao et al.'s optimization constants repository](https://teorth.github.io/optimizationproblems/constants/1a.html) and connects to the asymptotic size of Sidon sets in additive combinatorics. While recent progress on the *upper* bound has come from AI-driven search (AlphaEvolve, TTT-Discover), the *lower* bound has not been improved since 2017. Closing the gap from either direction is valuable.

## References

- [Cloninger & Steinerberger (2017), arXiv:1403.7988](https://arxiv.org/abs/1403.7988) — **Primary reference** for the lower bound algorithm
- [Tao et al., Optimization Constants Repo](https://github.com/teorth/optimizationproblems)
- [Matolcsi & Vinuesa (2010), arXiv:0907.1379](https://arxiv.org/abs/0907.1379)
- [White (2022), arXiv:2210.16437](https://arxiv.org/abs/2210.16437)
- [Boyer & Li (2025), arXiv:2506.16750](https://arxiv.org/abs/2506.16750)
- [AlphaEvolve, arXiv:2511.02864](https://arxiv.org/abs/2511.02864)
