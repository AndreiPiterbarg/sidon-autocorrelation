# Curriculum Learning: Self-Critique

## The Problem

We seek the constant $C_{1a} = \inf \|f * f\|_\infty / \|f\|_1^2$ over nonnegative $f \in L^1[-1/4, 1/4]$. Current bounds: $C_{1a} \in [1.2802, 1.5029]$. We approximate $f$ as a step function with $P$ equal-width bins on $[-1/4, 1/4]$ and minimize the peak of its autoconvolution.

## The Cloud Solution (Baseline)

A 9-round tournament pipeline on Modal (32-core containers):

1. **Strategy sweep** at P=200: 12 initialization strategies, 80 restarts each. Top 6 advance.
2. **Progressive upsampling**: P=200 -> 300 -> 500 -> 750 -> 1000 -> 1500, warm-starting from the best solution at each level.
3. **Cross-pollination**: Blend best solutions across P values, re-optimize.

Each restart runs LSE continuation (smooth approximation to max via LogSumExp, annealed from beta=1 to beta=3000) followed by Polyak subgradient polish (200K-500K iterations on the non-smooth objective).

**Result**: 1.5055 at P=1500.

## What Curriculum Learning Tried

Hypothesis: massive exploration at low P (where restarts are cheap) would discover fundamentally better basins that, when upsampled, would beat the cloud's results at high P.

**Stage 1 -- Massive exploration.** 6 strategies x 5,000 restarts = 30,000 total restarts at each of P=30, 40, 50. Keep top-10 diverse solutions (L2 distance threshold, scaled by P) at each P.

**Stage 2 -- Upsample cascade.** Take the 10 diverse P=50 solutions and cascade upward: P=50 -> 100 -> 200 -> 500 -> 1000. At each level, upsample each candidate via height interpolation, then run 1 direct polish + 50 warm perturbations. Keep top-3 diverse for the next level.

Total compute: 90,000 restarts at low P + ~2,500 restarts across the cascade. Run on Modal with the same LSE+Polyak optimizer.

## Results

| P | Curriculum | Cloud | Delta |
|---|-----------|-------|-------|
| 30 | 1.5244 | -- | -- |
| 40 | 1.5204 | -- | -- |
| 50 | 1.5171 | -- | -- |
| 100 | 1.5105 | -- | -- |
| 200 | **1.5096** | 1.5097 | **-0.0001** |
| 500 | 1.5074 | 1.5069 | +0.0005 |
| 1000 | 1.5065 | 1.5057 | +0.0008 |
| 1500 | -- | **1.5055** | -- |

Curriculum wins at P=200 by a negligible margin. Cloud wins everywhere else. Curriculum global best (1.5065) is 0.001 worse than cloud global best (1.5055).

## What Worked

**Monotone improvement across the cascade.** The progression 1.524 -> 1.517 -> 1.511 -> 1.510 -> 1.507 -> 1.507 is smooth and stable. The upsample + polish mechanics are sound.

**Marginal win at P=200.** The 30,000 restarts at P=50 found a seed that, when upsampled to P=200, slightly beat the cloud's P=200 result. This confirms that more restarts at low P do help locally.

## What Did Not Work

**Diversity filtering was ineffective.** At P=50, the 10 "diverse" solutions cluster within a 0.003 range (1.517-1.520). The L2 distance threshold in simplex-weight space does not distinguish functionally different basins. Solutions that look different in weight space produce nearly identical autoconvolution profiles.

**Warm-start cascading gets trapped.** Upsampling a P=50 solution to P=1000 locks the optimizer into the same local basin. The cloud's fresh random restarts at P=750/1000/1500 found genuinely different (and better) basins that no amount of upsampling from P=50 could reach. The curriculum approach has zero probability of finding these basins because it only explores neighborhoods of its P=50 seeds.

**Massive restarts hit diminishing returns.** Going from 80 restarts (cloud) to 5,000 restarts (curriculum) at P=50 improved the best from ~1.520 to 1.517 -- a gain of 0.003. Going to 30,000 total restarts (6 strategies) added nothing further. The landscape at P=50 simply does not have meaningfully better basins to find.

**Missing P=1500.** The cascade stopped at P=1000. The cloud's best came from P=1500, a regime the curriculum never reached. Even if it had, the basin-trapping problem would likely persist.

**The core assumption was wrong.** The hypothesis -- that better low-P exploration transfers to better high-P results -- is falsified. The optimization landscape changes qualitatively as P increases. Good basins at P=50 do not correspond to good basins at P=1000. Fresh random exploration at the target P beats warm-started cascading.
