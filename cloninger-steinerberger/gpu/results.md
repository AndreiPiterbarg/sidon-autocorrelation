# GPU Benchmark Results — 2026-02-15

## Hardware

- **GPU**: NVIDIA GeForce RTX 3080 Laptop GPU (CC 8.6, 8192 MB HBM)
- **CPU**: Laptop CPU with Numba JIT (baseline from `run_1hour_20260215_092522.json`)

## d=4 (n_half=2) — Head-to-head comparison

| m | Configs | CPU time | CPU rate | GPU time | GPU rate | Speedup |
|---|---------|----------|----------|----------|----------|---------|
| 50 | 10.8M | 0.008s | 1.4B/s | 0.008s | 1.4B/s | ~1x |
| 100 | 86.0M | 0.057s | 1.5B/s | 0.052s | 1.6B/s | 1.1x |
| 200 | 685.2M | 0.374s | 1.8B/s | 0.327s | 2.1B/s | 1.1x |
| 400 | 5.5B | 2.84s | 1.9B/s | 1.84s | 3.0B/s | **1.5x** |
| 700 | 29.3B | 15.0s | 2.0B/s | 7.83s | 3.7B/s | **1.9x** |
| 1000 | 85.4B | 43.7s | 2.0B/s | 24.5s | 3.5B/s | **1.8x** |
| 1500 | 288.1B | 140.2s | 2.1B/s | 80.4s | 3.6B/s | **1.7x** |
| 2000 | 682.9B | 312.1s | 2.2B/s | 196.1s | 3.5B/s | **1.6x** |
| 3000 | 2.3T | 1136.7s | 2.0B/s | 705.3s | 3.3B/s | **1.6x** |

## d=6 (n_half=3) — Head-to-head comparison

| m | Configs | CPU time | CPU rate | GPU time | GPU rate | Speedup |
|---|---------|----------|----------|----------|----------|---------|
| 5 | 8.3M | 0.008s | 1.0B/s | 0.008s | 1.0B/s | ~1x |
| 10 | 234.5M | 0.252s | 930M/s | 0.236s | 983M/s | 1.1x |
| 16 | 2.3B | 2.58s | 910M/s | 2.29s | 1.0B/s | 1.1x |
| 20 | 7.1B | 7.75s | 911M/s | 5.32s | 1.3B/s | **1.5x** |
| 25 | 21.3B | 23.5s | 906M/s | 15.5s | 1.4B/s | **1.5x** |
| 30 | 52.5B | 57.7s | 910M/s | 51.9s | 1.0B/s | 1.1x |
| 40 | 219.1B | 239.3s | 915M/s | 399.3s | 549M/s | **0.6x (SLOWER)** |
| 50 | 664.4B | 613.3s | 1.1B/s | killed | **TOO SLOW** | — |

## d=8 (n_half=4) — CPU only (GPU has no d=8 kernel)

| m | Configs | CPU time | CPU rate | Bound |
|---|---------|----------|----------|-------|
| 2 | 15.4M | 0.567s | 27.1M/s | 0.1667 |
| 3 | 202.9M | 8.51s | 23.9M/s | 0.4592 |
| 5 | 5.8B | 287.6s | 20.3M/s | 0.7692 |

## Best bounds achieved

| Backend | Best bound | Config | Total time |
|---------|-----------|--------|------------|
| CPU | **1.1252** | n=3, m=50 | 2892s (48 min) |
| GPU | **1.1157** | n=3, m=40 | ~1076s (killed before m=50 finished) |

## Diagnosis: Why the GPU is only 1.5-1.9x faster

Three critical performance bugs in `kernels.cu`:

### 1. Two-phase DRAM materialization (the killer)

The kernel uses a Phase 1 / Phase 2 architecture:
- Phase 1: generates compositions, applies cheap pruning, writes survivors to DRAM
- Phase 2: re-reads survivors from DRAM, computes full autoconvolution

For `find_min` mode, the pruning threshold is so high that **nearly all compositions
survive Phase 1**. This means the GPU writes and re-reads trillions of int32 values
through DRAM. Each survivor costs 12 bytes written + 12 bytes read = 24 bytes of DRAM
traffic for D=4.

The CPU Numba kernel is **single-pass fused**: generate composition, compute test value,
reduce — all in CPU registers with **zero DRAM traffic per composition**. The GPU's DRAM
roundtrip completely negates its compute advantage.

### 2. INT64 arithmetic on a consumer GPU (64x penalty)

Phase 2 uses `long long` (INT64) for the autoconvolution:
```c
long long conv[CONV_LEN];
conv[i + j] += 2LL * ci * (long long)c[j];
```

On the RTX 3080 (GA102 / Ampere consumer), **INT64 throughput is 1/64th of INT32**.
But for D=4 m=3000: max autoconvolution value = S^2 = 576,000,000, which fits easily
in INT32 (max 2,147,483,647). The kernel pays 64x compute cost for no reason.

This applies for all practical m values up to ~5800 (where S^2 exceeds INT32 range).

### 3. FP64 in Phase 2 window scan (64x penalty)

The window scan normalizes with `double`:
```c
double inv_m_sq = 1.0 / ((double)m * (double)m);
double tv = (double)ws * inv_norm;
```

On consumer Ampere GPUs, **FP64 throughput is 1/64th of FP32** (only 2 FP64 units per
SM vs 128 FP32 cores). The combination of INT64 + FP64 means Phase 2 runs at roughly
**1/64th of the GPU's actual capability**.

### Summary of performance loss

| Factor | Penalty | Explanation |
|--------|---------|-------------|
| DRAM materialization | ~5-10x | Write+read survivors vs CPU's register-only |
| INT64 on consumer GPU | ~64x | Phase 2 autoconvolution |
| FP64 on consumer GPU | ~64x | Phase 2 window scan |
| Combined effective | ~50-100x slowdown vs theoretical peak |

The RTX 3080 has 10.4 TOPS INT32. At ~30 INT32 ops per composition, theoretical
throughput is ~350B compositions/s. Actual: 3.3B/s = **0.9% of peak**.

