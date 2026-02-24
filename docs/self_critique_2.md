# GPU Branch-and-Prune: Self-Critique

## The Problem

We seek to prove $c \geq c_\text{target}$ for the autoconvolution constant $c = \inf \|f*f\|_\infty / (\int f)^2$ over nonneg $f$ supported on $(-1/4, 1/4)$. Current bounds: $c \in [1.2802,\; 1.5029]$. The Cloninger--Steinerberger hierarchical branch-and-prune algorithm (arXiv:1403.7988) is the only known method for certified lower bounds. We reimplemented it in CUDA for NVIDIA A100 GPUs.

## The Approach (GPU Reimplementation)

A faithful translation of the CS14 algorithm to GPU:

1. **Level 0**: Enumerate all compositions in $B_{n,m}$ (D=6 bins, $m=50$, ~3.5M points). Fused CUDA kernel does composition generation + FP32 pre-checks + FP64 autoconvolution + dynamic windowed threshold in one pass. Extract survivors.
2. **Level 1 (refinement)**: Each surviving parent's 6 bins split into 12. Batched GPU kernel processes millions of parents, prefix-sum indexing maps flat thread indices to parent/child pairs. Energy cap ($x_\text{cap}$) limits per-bin split range.
3. **Level 2+**: Continue doubling ($D=24, 48, \ldots$) until all configurations eliminated or budget exhausted.

Key optimizations: FP32/FP64 dual precision, canonical palindrome reduction (2x), dynamic per-position thresholds (Eq. 1), energy cap ($x_\text{cap}$), batched multi-parent kernel launches, streaming survivor extraction to disk.

**Result**: Level 0 and Level 1 complete in seconds on A100. Level 2 is infeasible (~138--151 hours estimated for a single A100).

## What Worked

**GPU throughput at Levels 0 and 1.** The base enumeration (D=6, 3.5M configs) runs in 0.024s -- roughly 145M configs/s. Level 1 refinement processes 15.7B child configurations in 5--10s (~2.9B refs/s). What took the original MATLAB implementation hours now takes seconds.

**FP32/FP64 dual-precision pipeline.** The FP32 pre-checks (asymmetry, half-sum, max-element, block-sum) eliminate ~75% of configurations before the expensive FP64 autoconvolution. This is a clean 3--4x speedup over FP64-only.

**Dynamic per-position thresholds.** Using $T(k,\ell) = c_\text{target} + (1+2W_\text{int})/m^2$ instead of the global $T = c_\text{target} + 2/m + 1/m^2$ tightens thresholds at boundary windows. This prunes an additional 5--15% of configurations that would survive with static thresholds.

**Strict fail-closed correctness.** Every anomaly (timeout, extraction overflow, count mismatch) aborts with "inconclusive" rather than claiming a proof. This caught real issues: extraction truncation in early runs, timer overflows on Windows TDR, and inconsistent survivor counts from race conditions. The mathematical audit of all GPU kernels against the paper found zero correctness bugs.

**Batched multi-parent refinement.** Instead of launching one kernel per parent (108K kernel launches), the batched approach packs all parents into a single launch with prefix-sum indexing. This eliminates kernel launch overhead and improves L2 cache reuse on the A100 (40 MB L2).

## What Did Not Work

**The fundamental scaling wall at Level 2.** This is the central failure. The numbers tell the story:

| Level | D | Parents | Survivors | Time (A100) |
|-------|---|---------|-----------|-------------|
| 0 | 6 | 3.5M (grid) | 74K--105K | 0.02s |
| 1 | 12 | 74K--105K | 3.1M--19.9M | 5--10s |
| 2 | 24 | 3.1M--19.9M | ??? | ~138--151 hrs (est.) |

Survivors **grow** from level to level. Level 0 produces ~74K survivors (for $c_\text{target}=1.28$); Level 1 produces ~3.1M. This is the wrong direction. For the proof to complete, survivors must eventually reach zero. Instead, the branching factor dominates the pruning rate.

**The branching factor is the bottleneck, not throughput.** Each parent at Level 1 has 6 bins averaging $B_i \approx 8.3$ (since $\sum B_i = 50$). With energy cap, each bin produces $\sim 2 \cdot x_\text{cap} + 1 \approx 13$ children. The Cartesian product: $13^6 \approx 4.8$M refinements per parent. With 74K parents, that's ~350 billion refinements. At 2.9B refs/s, Level 1 takes ~120s. But at Level 2, parents have 12 bins each averaging $B_i \approx 4.2$, producing ~$9^{12} \approx 2.8$B refinements per parent. With 3.1M parents, the total is ~$8.7 \times 10^{15}$ -- completely infeasible regardless of GPU throughput.

**Increasing GPU throughput does not solve an exponential problem.** Going from 1 A100 to 8 A100s reduces Level 2 from 138 hours to ~17 hours. Going to 128 A100s brings it to ~1 hour. But Level 3 (D=48) would then take $10^{10}$ hours. The exponential growth of the Cartesian product means hardware scaling hits a wall one level later.

**Higher targets make everything worse.** At $c_\text{target} = 1.30$ vs $1.28$, Level 0 survivors increase 1.4x (74K to 105K) and Level 1 survivors increase 6.4x (3.1M to 19.9M). The threshold $T = c_\text{target} + 0.04$ gets closer to test values, so fewer configurations are pruned. Pushing above 1.28 requires either much more compute or fundamentally better pruning.

**Energy cap has diminishing returns at higher D.** The cap $x_\text{cap} = \lfloor m\sqrt{T/D} \rfloor$ helps at D=12 ($x_\text{cap} \approx 16$, reducing splits from 50 to ~33 per bin). At D=24, $x_\text{cap} \approx 12$ is already close to the typical bin value, so the cap provides less reduction. And the number of bins doubles, so the Cartesian product still explodes.

**Pruning improvements are linear; the problem is exponential.** Dynamic thresholds, better pre-filters, and energy caps each improve pruning by 5--30%. But the combinatorial explosion grows by factors of $10^6$--$10^9$ per level. No constant-factor pruning improvement can overcome this.

## The Core Assumption That Was Wrong

**The assumption**: GPU throughput gains (100--1000x over CPU) would be sufficient to push the branch-and-prune algorithm to higher $c_\text{target}$ values.

**The reality**: The algorithm's computational cost grows doubly exponentially with the refinement depth (exponential branching at each level, and the number of levels needed grows with $c_\text{target}$). GPU acceleration compresses Levels 0 and 1 from hours to seconds but does not change the fundamental feasibility of Level 2+. The original paper's result of $c \geq 1.28$ was not limited by hardware speed -- it was limited by the algorithm's combinatorial structure at the parameter regime ($n=12, m=50$) where Level 2 becomes necessary.

To meaningfully improve the lower bound beyond 1.28, we need one of:
1. **Algorithmic pruning breakthroughs** that reduce survivors by orders of magnitude (not percentages). LP dual-bound pre-screening or Fourier-analytic certificates could potentially eliminate parents without enumerating children.
2. **Non-uniform refinement** that only splits the bins most responsible for borderline test values, reducing the Cartesian product from $\prod_{i=1}^{D} (B_i+1)$ to $\prod_{i \in S} (B_i+1)$ for a small subset $S$.
3. **Qualitatively different mathematical frameworks** that avoid exhaustive enumeration entirely (e.g., SDP/SOS certificates, or analytic arguments about the structure of near-optimal functions).

The GPU reimplementation was not wasted work -- it confirmed the algorithm's correctness independently, demonstrated the precise scalability wall, and produced the infrastructure needed if algorithmic breakthroughs are found. But the path from "fast GPU kernels" to "better lower bound" requires crossing an exponential gap that hardware alone cannot bridge.
