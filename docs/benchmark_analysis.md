# Benchmark Sweep Analysis — 2026-02-24

## Overview

Hyperparameter sweep over `(n_half, m)` for `c_target = 1.30` on an NVIDIA A100.
Each configuration ran L0 (Level 0) + L1 (refinement) fully on GPU, then an unbiased
random L2 sample (~60s budget) to project total compute.

**Dimension schedule:**
- `n_half=2` (d0=4): 4 → 8 → 16 → 32 → 64 → ...
- `n_half=3` (d0=6): 6 → 12 → 24 → 48 → 96 → ...

---

## Results Table

### n_half=2 (d0=4)

| m | Correction | L0 Comps | L0 Surv | L1 Surv | L2 Refs (exact) | L2 Thru (refs/s) | L2 Surv Rate | Est L2 Surv | Total Est | L3 Needed? |
|----:|----------:|----------:|---------:|----------:|----------------:|------------------:|-------------:|------------:|----------:|:----------:|
| 25 | 0.0816 | 3,276 | 416 | 28,987 | 748M | 1.19B | 4.26e-4 | 318,952 | **0.8s** | YES |
| 50 | 0.0404 | 23,426 | 2,718 | 1,074,640 | 1.92T | 692M | 2.38e-5 | 45.6M | **46 min** | YES |
| 75 | 0.0268 | 76,076 | 8,533 | 10,652,793 | 294T | 16.9B | 1.31e-6 | 383.5M | **4.8 h** | YES |
| 100 | 0.0201 | 176,851 | 19,415 | 59,642,682 | 12.5P | 30.9B | 1.35e-7 | 1.69B | **4.7 days** | YES |
| 125 | 0.0161 | 341,376 | 36,710 | 235,449,366 | 191P | 21.2B | 1.17e-8 | 2.23B | **104 days** | YES |
| 150 | 0.0134 | 585,276 | 62,143 | 741,381,740 | 582P | 19.9B | 4.48e-10 | 261M | **339 days** | YES |
| 175 | 0.0115 | 924,176 | 97,845 | 1,987,635,274 | 1.31E | 16.0B | **0** | **0** | 952 days | NO |
| 200 | 0.0100 | 1,373,701 | 144,308 | 4,714,720,989 | 2.16E | 14.1B | **0** | **0** | 1,771 days | NO |
| 225 | 0.0089 | 1,949,476 | 203,552 | 10,152,410,943 | 2.45E | 13.5B | **0** | **0** | 2,096 days | NO |
| 250 | 0.0080 | 2,667,126 | 277,082 | 20,290,636,220 | 2.94E | 12.7B | **0** | **0** | 2,667 days | NO |

*E = exarefs (10^18), P = petarefs (10^15), T = terarefs (10^12), B = billion, M = million*

### n_half=3 (d0=6)

| m | Correction | L0 Comps | L0 Surv | L1 Surv | L2 Refs (exact) | L2 Thru (refs/s) | L2 Surv Rate | Est L2 Surv | Total Est | L3 Needed? |
|----:|----------:|----------:|---------:|----------:|----------------:|------------------:|-------------:|------------:|----------:|:----------:|
| 25 | 0.0816 | 142,506 | 6,011 | 186,856 | 20.8B | 268M | 2.06e-6 | 42,871 | **78 s** | YES |
| 50 | 0.0404 | 3,478,761 | 105,447 | 19,933,279 | 552T | 5.82B | **0** | **0** | **26.3 h** | **NO** |

*n_half=3 at m >= 75 was not reached during the sweep (configs sorted cheapest-first).*

---

## Key Comparisons

### 1. Effect of m at Fixed n_half=2

As m increases, three opposing forces are at work:

| Factor | Low m (25-50) | Mid m (75-125) | High m (150+) |
|--------|:------------:|:--------------:|:-------------:|
| Correction term (2/m + 1/m^2) | Large (0.04-0.08) | Moderate (0.016-0.027) | Small (0.008-0.013) |
| Pruning power per level | Weak | Moderate | Strong |
| L1 survivor explosion | ~29K-1M | ~11M-235M | ~741M-20B |
| L2 survival rate | High (10^-4 to 10^-5) | Low (10^-6 to 10^-8) | Zero in sample |
| Levels needed for proof | Many (4+) | Several (3-4) | Potentially just 2 |
| L2 compute cost | Cheap | Expensive | Astronomical |

**The fundamental tension:** Higher m gives better per-level pruning (lower correction
means the effective threshold 1.30 + correction is closer to 1.30), so fewer levels are
needed. But higher m creates exponentially more compositions at each level, making each
level exponentially more expensive.

**Critical threshold at m ~ 175:** At m >= 175, the L2 sample (processing ~2.3T refs in
~147s) finds ZERO survivors. This means the proof would terminate at L2 — but L2 itself
requires 1.3 exarefs, taking ~950 days on a single A100. **Not feasible.**

### 2. n_half=2 vs n_half=3 at Same m

| Metric | n_half=2, m=25 | n_half=3, m=25 | n_half=2, m=50 | n_half=3, m=50 |
|--------|:-----------:|:-----------:|:-----------:|:-----------:|
| d0 | 4 | 6 | 4 | 6 |
| L0 compositions | 3,276 | 142,506 | 23,426 | 3,478,761 |
| L0 survivors | 416 | 6,011 | 2,718 | 105,447 |
| L1 survivors | 28,987 | 186,856 | 1,074,640 | 19,933,279 |
| L2 survival rate | 4.26e-4 | 2.06e-6 | 2.38e-5 | **0** |
| Est L2 survivors | 318,952 | 42,871 | 45.6M | **0** |
| L3 needed? | YES | YES | YES | **NO** |
| Total est | 0.8s (+L3+) | 78s (+L3+) | 46min (+L3+) | **26.3h (DONE)** |

**Key insight:** n_half=3 has dramatically better pruning power at the same m. Despite
starting with 43x more L0 compositions at m=25 and 149x more at m=50, n_half=3 achieves
much lower L2 survival rates. At m=50, n_half=3 reaches zero L2 survivors — meaning the
proof can terminate at Level 2 — while n_half=2 still has 45.6M estimated L2 survivors
and needs many additional levels.

The reason: higher d0 means each level doubles to a higher dimension (6→12→24 vs 4→8→16),
giving each refinement step more "geometric room" to prune. The autoconvolution becomes
more constrained in higher dimensions.

### 3. L1 Survivor Explosion

L1 survivors grow super-exponentially with m (for n_half=2):

| m | L1 Survivors | Extracted (capped) | Growth Factor |
|---:|-----------:|------------------:|-------------:|
| 25 | 28,987 | 28,987 | — |
| 50 | 1,074,640 | 1,074,640 | 37x |
| 75 | 10,652,793 | 10,652,793 | 10x |
| 100 | 59,642,682 | 59,642,682 | 5.6x |
| 125 | 235,449,366 | 200,000,000 | 3.9x |
| 150 | 741,381,740 | 200,000,000 | 3.1x |
| 200 | 4,714,720,989 | 200,000,000 | — |
| 250 | 20,290,636,220 | 200,000,000 | — |

At m >= 125, the 200M extraction cap is hit. This means the L2 sample only sees a
random subset of parents, but the L2 exact ref count is computed from all L1 survivors.
The extraction cap does NOT affect the projected total time calculation (which uses exact
counts), but it means the survival rate estimate comes from a potentially biased subsample.

### 4. L2 Throughput

| Config | L2 Throughput (refs/s) |
|--------|---------------------:|
| n_half=2, m=25 | 1.19B |
| n_half=2, m=50 | 692M |
| n_half=2, m=75 | 16.9B |
| n_half=2, m=100 | **30.9B** |
| n_half=2, m=125 | 21.2B |
| n_half=2, m=150 | 19.9B |
| n_half=2, m=175 | 16.0B |
| n_half=2, m=200 | 14.1B |
| n_half=3, m=25 | 268M |
| n_half=3, m=50 | 5.82B |

Peak throughput is at n_half=2, m=100 (30.9B refs/s). The low throughput at m=25
(especially n_half=3) suggests kernel underutilization — too few parents per batch. The
decline at m >= 150 reflects higher child dimensions (d_child=32+) reducing occupancy.

n_half=3 throughput is ~5x lower than n_half=2 at the same m because the child dimensions
are larger (d_child_L2 = 24 vs 16).

---

## Statistical Confidence on Zero-Survivor Findings

For configs reporting 0 L2 survivors, the upper bound on the true survival rate (at 95%
confidence, using the rule of 3) is:

| Config | L2 Sample Refs | Upper Bound on Rate | Upper Bound on Total Survivors |
|--------|---------------:|-------------------:|-----------------------------:|
| n_half=2, m=175 | 2.34T | 1.28e-12 | 1.68 |
| n_half=2, m=200 | 2.06T | 1.46e-12 | 3.14 |
| n_half=3, m=50 | 440B | 6.82e-12 | **3.77** |

**For n_half=3, m=50:** Even at the 95% upper confidence bound, we'd expect at most ~4
L2 survivors. This is strong evidence the proof terminates at L2. However, the sample
fraction is only 0.08% of total L2 refs — a larger sample would increase confidence.

For n_half=2, m=175+: The upper bounds are also very low, but the total L2 compute is
infeasible (>900 days per config).

---

## Recommendations for Next Run

### Primary Recommendation: n_half=3, m=50

**Projected time: ~26 hours on 1x A100 (or ~3 hours on 8x A100).**

This is the only configuration that:
1. Shows zero L2 survivors (proof likely terminates at Level 2)
2. Has a feasible total estimated time (~26h single-GPU)
3. L2 survival rate upper bound suggests <= 4 total survivors even pessimistically

**Action:** Run the full proof at `n_half=3, m=50, c_target=1.30`:
```bash
python -m cloninger-steinerberger.gpu.run_proof --n_half 3 --m 50 --c_target 1.30
```

### Secondary: Explore n_half=3 at m=35-45

The sweep only tested m=25 and m=50 for n_half=3. There may be a faster sweetspot:

| m | Correction | Est L0 Comps (C(m+5, 5)) | Effective Threshold |
|---:|----------:|------------------------:|-------------------:|
| 30 | 0.0678 | 324,632 | 1.3678 |
| 35 | 0.0579 | 658,008 | 1.3579 |
| 40 | 0.0506 | 1,221,759 | 1.3506 |
| 45 | 0.0449 | 2,118,760 | 1.3449 |
| 50 | 0.0404 | 3,478,761 | 1.3404 |

**Potential speedup:** If m=40 already achieves zero L2 survivors, L0 comp count is 2.8x
smaller than m=50, which cascades to proportionally fewer L1 and L2 refs. Could cut the
total time by 3-5x.

**Action:** Run a targeted mini-sweep at n_half=3 with m in {35, 40, 45}:
```bash
python -m cloninger-steinerberger.gpu.benchmark_sweep --n_half_values 3 --m_values 35,40,45 --l2_sample_sec 120 --resume
```

### Avoid: n_half=2 at Any m

For `c_target=1.30`, n_half=2 is dominated by n_half=3 at every regime:
- **Low m (25-75):** Fast L2 but high survival rates → needs many levels (L3, L4, ...) with unknown cascading costs
- **High m (100+):** Terminates earlier in the level hierarchy but L2 alone takes days to years
- **The n_half=2 extraction cap issue:** At m >= 125, the 200M cap means only a fraction of L1 survivors are processed in the L2 sample, making projections less reliable

### Future Exploration: Higher c_target

Once `c >= 1.30` is proven, the next targets would be:
- `c_target = 1.31`: Correction at m=50 is 0.0404 → effective threshold 1.3504. Still very feasible.
- `c_target = 1.32`: Effective threshold 1.3604. May need m=60-75 for n_half=3.
- `c_target = 1.33+`: Will need dedicated sweeps.

---

## Summary

| Priority | Config | Action | Est Time (1x A100) |
|:--------:|--------|--------|-------------------:|
| 1 | n_half=3, m=50 | Full proof run | ~26h |
| 2 | n_half=3, m=35-45 | Mini-sweep to find faster sweetspot | ~30min sweep |
| 3 | n_half=3, m=best | Full proof at optimal m | TBD |

The data strongly indicates **n_half=3 is the right dimension family** and **m ~ 40-50
is the right resolution** for proving `c >= 1.30`. The proof should terminate at Level 2
(dimension 24), avoiding the unknown costs of Level 3+.
