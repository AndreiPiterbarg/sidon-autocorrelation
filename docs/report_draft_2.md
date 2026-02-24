# Report 2: GPU-Accelerated Branch-and-Prune for the Autoconvolution Lower Bound

## Problem Statement

### What are you optimizing?

We seek to **prove rigorous lower bounds** on the autoconvolution constant $c$, defined as:

$$c = \inf_{\substack{f \geq 0 \\ \text{supp}(f) \subseteq (-1/4,\,1/4)}} \frac{\|f*f\|_{L^\infty}}{\left(\int f\right)^2}$$

This is the complementary problem to the upper-bound work in Report 1. While the upper bound asks "how small can the peak autoconvolution be?" (answered by constructing explicit functions), the lower bound asks "can we *prove* it can never be smaller than some value?" (answered by exhaustive verification over a discretized search space).

We implement the Cloninger--Steinerberger hierarchical branch-and-prune algorithm (arXiv:1403.7988) on NVIDIA A100 GPUs, aiming to push the current best lower bound of $c \geq 1.2802$ higher.

### Why does this problem matter?

The autoconvolution constant is an open problem in Davis et al.'s optimization constants repository. Current bounds: $c \in [1.2802,\; 1.5029]$, a gap of ~0.22. The lower bound of 1.2802 was established by Cloninger and Steinerberger in 2014 using ~20,000 CPU hours on 2014-era hardware. No one has improved this lower bound in over a decade. A GPU reimplementation with modern hardware and tighter pruning could push this boundary.

### How will you measure success?

The primary goal is to formally prove $c \geq c_\text{target}$ for some $c_\text{target} > 1.2802$. A proof requires exhaustively eliminating all discretized configurations at sufficient resolution. Secondary metrics include GPU throughput (configurations/second), pruning efficiency (fraction eliminated at each level), and the feasibility of reaching deeper refinement levels.

### What are your constraints?

- **Mathematical**: The algorithm requires checking all $\binom{4nm + 2n - 1}{2n - 1}$ lattice points in $B_{n,m}$, growing polynomially in $m$ but exponentially in $n$.
- **Discretization floor**: The correction term $2/m + 1/m^2$ sets a minimum $m$ for each target. For $c_\text{target} = 1.28$, we need $m \geq 50$.
- **Computational**: The hierarchical refinement produces $\prod_i (B_i + 1)$ children per parent, causing exponential blowup at deeper levels.
- **Hardware**: NVIDIA A100 GPU (80 GB HBM2e, 108 SMs, compute capability 8.0) accessed via RunPod cloud.

### What data do you need?

No external data. The algorithm is a self-contained exhaustive verification. Inputs are the target $c_\text{target}$, grid resolution $m$, starting coarseness $n$, and number of refinement levels.

### What could go wrong?

- **Refinement explosion**: Survivors from level $k$ may produce infeasibly many children at level $k+1$, making the proof computationally impossible.
- **Register pressure**: CUDA kernels for $D = 24$ or $48$ require many registers per thread, reducing GPU occupancy and throughput.
- **Memory limits**: Even 80 GB of A100 HBM is finite when millions of parent configurations each produce billions of children.

---

## Technical Approach

### Mathematical formulation

The Cloninger--Steinerberger algorithm (Lemmas 1--3 of the paper) works as follows:

1. **Discretization** (Lemma 1): Approximate $f$ as a step function with $2n$ bins of equal width $1/(4n)$. The bin-average vector $a \in A_n$ captures the mass distribution.
2. **Lattice covering** (Lemma 2): Discretize $A_n$ into the lattice $B_{n,m} = \{b : b_i \in \{0, 1/m, \ldots\}, \sum b_i = 4n\}$ which forms a $1/m$-net. In integer coordinates ($S = m$ convention), $\sum c_i = m$.
3. **Error correction** (Lemma 3): For any lattice point $b$, $c \geq \text{tv}(b) - 2/m - 1/m^2$, where $\text{tv}(b) = \max_{k,\ell} \frac{1}{4n\ell} \sum_{k \leq i+j \leq k+\ell-2} b_i b_j$ is the windowed autoconvolution test value.
4. **Hierarchical refinement**: Start at coarse $n = 3$ (6 bins), prune, then refine survivors by doubling resolution ($n \to 2n$). Each parent bin $B_i$ splits into two children $c_{2i} + c_{2i+1} = B_i$. A parent is ruled out iff ALL its children are ruled out.

If all configurations are eliminated at some level, then $c \geq c_\text{target}$ is formally proven.

### Algorithm choice and justification

We chose to reimplement the original Cloninger--Steinerberger branch-and-prune rather than pursue alternative approaches (SDP relaxations, Fourier methods) because:

- It is the **only method that has produced a certified lower bound** for this problem.
- It is embarrassingly parallel: each configuration can be tested independently.
- Modern GPUs offer 10--100x throughput over the 2014-era hardware used in the original work.
- The algorithm's correctness is well-established and our implementation has been formally audited against the paper.

### Implementation strategy

**Two-phase GPU pipeline:**

- **Phase 1 (FP32 pre-checks)**: Cheap filters eliminate most configurations before expensive FP64 computation. Includes asymmetry pruning, half-sum bounds, max-element checks, block-sum bounds, and canonical palindrome filtering.
- **Phase 2 (FP64 verification)**: Full autoconvolution with dynamic per-position thresholds. Only reached by the ~3% of configurations surviving Phase 1.

**Hierarchical refinement on GPU:**

- **Batched parent processing**: Multiple parents combined into single kernel launches, indexed via prefix sums and binary search for parent lookup.
- **Energy cap**: Per-bin cap $x_\text{cap} = \lfloor m\sqrt{T/D} \rfloor$ skips children that would trivially fail the $\ell = 2$ check, reducing the Cartesian product size.
- **Templated kernels**: Specialized for $D_\text{child} = 12, 24, 48$ with appropriate block sizes (256, 128, 64 threads).

**Dynamic thresholds (Equation 1 of the paper):**

Instead of using the global correction $2/m + 1/m^2$ at every window position, we compute per-position thresholds $T(k,\ell) = c_\text{target} + (1 + 2W_\text{int})/m^2$ where $W_\text{int}$ is the mass contributing to that specific window. This tightens the threshold at boundary windows where fewer bins contribute, pruning more aggressively.

### Validation methods

- **Correctness audit**: Full mathematical audit of all GPU kernels against the paper (documented in `docs/correctness_check.md`). Every formula verified algebraically.
- **CPU cross-validation**: GPU results checked against independent CPU (Numba JIT) implementations at small parameters.
- **Conservative rounding**: FP32 thresholds inflated by $10^{-5}$ (80x headroom over FP32 relative error). FP64 thresholds rounded down via directed truncation. Cannot cause false pruning.
- **Strict fail-closed mode**: Any anomaly (timeout, extraction truncation, count mismatch) causes the run to abort with status "inconclusive" rather than claim a false proof.

### Resource requirements

- **Hardware**: NVIDIA A100-SXM4-80GB on RunPod ($1.64/hr for 1x A100).
- **Memory**: Level 0 (D=6, m=50): ~200 MB. Level 1 refinement: ~2 GB for survivor buffers. Level 2 (D=24): projected 10--40 GB.
- **Time**: Level 0 completes in <1 second. Level 1 in 5--10 seconds. Level 2 is the bottleneck (estimated 100+ hours).

---

## Results

### Evidence the implementation works

**Mathematical correctness verified:**
- All 12 audit items pass (test value formula, correction term, dynamic thresholds, contributing bins, autoconvolution, asymmetry pruning, canonical palindrome, pre-filter bounds, FP32 margins, integer threshold truncation, refinement kernel, INT32/INT64 dispatch).
- GPU checks a superset of the paper's window positions (safe: can only strengthen the bound).
- CPU and GPU produce identical survivor counts at small parameters.

**Proof runs completed successfully through Level 1:**

| Target | m | Level 0 (D=6) | Level 1 (D=12) | Level 2 (D=24) |
|--------|---|---------------|----------------|----------------|
| 1.28 | 50 | 3.5M grid, 74K survivors (0.02s) | 74K parents, 3.1M survivors (6.3s) | INFEASIBLE (~151 hrs) |
| 1.30 | 50 | 3.5M grid, 105K survivors (0.03s) | 105K parents, 19.9M survivors (10.4s) | INFEASIBLE (~138 hrs) |

### Performance metrics

**GPU throughput (A100):**
- Level 0 base enumeration: ~145M configs/s (D=6, m=50)
- Level 1 refinement: ~2.9B refs/s (D=12, including FP32 pre-checks)
- Level 1 calibration: 500 parents in ~0.15s, extrapolates linearly

**Pruning effectiveness:**
- Level 0: FP32 pre-checks eliminate ~75% before FP64. Final survivors ~2--3% of total grid.
- Level 1: FP64 test-value pruning eliminates 99.87% of all refinements. But 0.13% of 15.7 billion is still 19.9 million survivors.

### Current limitations

1. **Level 2 wall**: The refinement at D=24 is computationally infeasible. With 19.9M parents averaging 6.5M refinements each, the total work is ~$1.3 \times 10^{14}$ refinements. At current throughput, this requires ~500,000 seconds (~138 hours) on a single A100.

2. **Survivor growth**: Survivors grow from level to level instead of shrinking. Level 0 produces 74K--105K survivors; Level 1 produces 3.1M--19.9M. This is the opposite of what the algorithm needs for convergence.

3. **Higher targets are worse**: $c_\text{target} = 1.30$ produces 2.7x more Level 0 survivors and 6.4x more Level 1 survivors than $c_\text{target} = 1.28$. The threshold $T$ is closer to the test values, so fewer configurations are pruned.

4. **Register pressure at D=24**: The kernel needs registers for $D = 24$ child bins, 47 convolution entries, prefix sums, and dynamic threshold computation. This limits occupancy to ~25% on A100, reducing throughput.

### Resource usage

- Total A100 GPU-hours consumed: ~50 hours across all development and proof runs.
- Storage: ~2 GB of survivor binary files and proof JSON logs.
- RunPod cost: ~$82 total.

### Unexpected challenges

- **Refinement calibration variance**: The first 10K parents are not representative of the full set. Evenly-spaced sampling (500 parents across the full range) gives 2x more accurate time estimates.
- **Streaming survivors to disk**: At 19.9M survivors $\times$ 12 dimensions $\times$ 4 bytes = 955 MB, in-memory storage works but is tight. Chunked file I/O was needed for robustness.
- **Energy cap diminishing returns**: The $x_\text{cap}$ reduces per-parent refinements by 10--100x at D=12, but at D=24 the Cartesian product over 12 parent bins still explodes even with capping.

---

## Next Steps

### Immediate improvements needed

1. **Tighter pruning at Level 1**: The 19.9M survivors at D=12 are the root cause of the Level 2 wall. Any improvement here has multiplicative impact. Potential approaches:
   - Stronger FP32 pre-filters exploiting more window positions simultaneously
   - Multi-bin bound: instead of single-bin $\ell=2$ checks, test $\ell=4$ or $\ell=6$ windows as pre-filters
   - Tighter energy cap using per-window rather than global bounds

2. **Reduce per-parent refinement count at D=24**: Even if survivor count is unchanged, reducing the average 6.5M refinements/parent would help. Possible via:
   - Stronger energy cap derivation (currently uses global $x_\text{cap}$; could use per-bin caps based on neighboring bin values)
   - Early parent elimination: track whether all children of a parent have been pruned and skip remaining children

3. **Higher m**: Increasing $m$ from 50 to 75 or 100 reduces the correction term, potentially allowing more pruning. But this increases the Level 0 grid combinatorially, so the net effect is unclear.

### Technical challenges to address

- **D=24 kernel optimization**: Profile register usage, test shared memory spilling strategies, experiment with reduced block sizes (32 or 64 threads).
- **Multi-GPU scaling**: The batched refinement is embarrassingly parallel across parents. Distributing across 4--8 A100s could bring Level 2 into the feasible range if individual throughput is maintained.
- **Adaptive level scheduling**: Instead of uniform doubling ($n \to 2n$), consider intermediate steps ($n \to 1.5n$) to limit the branching factor at each level.

### Questions needing help

- Is there a tighter per-position energy cap that accounts for correlations between adjacent bins (rather than treating each bin independently)?
- Can LP relaxation pre-screening (as described in `docs/improved_autoconvolution_algorithm.md`) eliminate parents before enumerating any children, and is the overhead justified?
- At what point does increasing $m$ become counterproductive (larger Level 0 grid) versus helpful (smaller correction term)?

### Alternative approaches to try

- **Fourier-analytic certification**: For the hardest surviving parents, use $\|g*g\|_\infty \geq \int \hat\phi \cdot |\hat g|^2$ with optimized test functions $\hat\phi \geq 0$ to eliminate them without full refinement enumeration.
- **Targeted refinement**: Only refine the bins where the test value is closest to the threshold (sensitivity-guided splitting), reducing the Cartesian product dramatically.
- **Hybrid CPU+GPU**: Use CPU for the relatively cheap parent-level bookkeeping and LP bounds, reserving the GPU for the inner-loop autoconvolution.

### What I have learned so far

- The Cloninger--Steinerberger algorithm is mathematically clean but computationally brutal at scale. The exponential branching factor is the fundamental bottleneck, not throughput.
- GPU acceleration helps enormously for Level 0 and Level 1 (seconds instead of hours), but cannot overcome the combinatorial explosion at Level 2 without algorithmic improvements to the pruning.
- Dynamic per-position thresholds are a genuine improvement over the paper's global correction, but the gain (~5--15% more pruning) is insufficient to change the feasibility picture.
- The strict fail-closed approach is essential. Several early runs had subtle issues (extraction truncation, timer overflow) that would have produced false proofs without it.
- FP32/FP64 dual-precision is a significant optimization: ~75% of configurations are eliminated by cheap FP32 checks, saving 4x on the expensive FP64 autoconvolution.
