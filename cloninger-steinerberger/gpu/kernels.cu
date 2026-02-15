/*
 * GPU kernels for Cloninger-Steinerberger branch-and-prune algorithm.
 *
 * Two-phase pipeline:
 *   Phase 1: Generate compositions + all cheap pruning + atomic compaction
 *   Phase 2: Integer autoconvolution + window max + block reduction
 *
 * Templated on D (number of bins). Instantiated for D=4, 6.
 *
 * Phase 1 parallelism:
 *   D=4: one thread per (c0, c1) pair, inner c2 loop
 *   D=6: one thread per (c0, c1, c2) triple, inner (c3, c4) loop
 *
 * Phase 2: one thread per survivor, uniform work (zero warp divergence).
 *   Uses integer autoconvolution (INT32 multiply -> INT64 accumulate)
 *   for consumer GPU performance (1:64 FP64:FP32 ratio).
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

/* ================================================================
 * Error checking macro
 * ================================================================ */
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

/* ================================================================
 * Device helpers
 * ================================================================ */

/* AtomicMin for double using CAS idiom.
 * For positive FP64 values, integer bit ordering matches FP64 ordering,
 * so compare-and-swap on the bit pattern works correctly. */
__device__ double atomicMinDouble(double* addr, double val) {
    unsigned long long int* addr_ull = (unsigned long long int*)addr;
    unsigned long long int old = *addr_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) <= val)
            return __longlong_as_double(assumed);
        old = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

/* Binary search: find largest i such that arr[i] <= val.
 * arr must be sorted ascending, arr[0] = 0. */
__device__ int binary_search_le(const long long* arr, int n, long long val) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = lo + (hi - lo + 1) / 2;
        if (arr[mid] <= val) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}


/* ================================================================
 * Phase 1 kernel: D=4
 *
 * One thread per (c0, c1) pair. Inner c2 loop.
 * Pruning: pair-sum, asymmetry, ell=3 right-window, canonical, ell=2 max.
 * Survivors written via per-survivor atomicAdd.
 * ================================================================ */
__global__ void phase1_d4(
    int S,
    int n_half,
    float inv_m_f,
    float thresh_f,       /* pruning threshold, FP32-inflated */
    float margin_f,       /* 1/(4*m) for asymmetry */
    float asym_limit_f,   /* find_min: init_min_eff; prove: c_target (inflated) */
    const long long* __restrict__ prefix_sums,  /* cumulative (c0,c1) pair counts */
    const int* __restrict__ c0_order,           /* zigzag c0 values */
    int n_c0,             /* number of c0 values (S/2 + 1) */
    long long total_pairs,
    int* __restrict__ survivor_buf,     /* 3 int32 per survivor: c0,c1,c2 */
    unsigned int* __restrict__ survivor_count,
    int max_survivors
) {
    long long flat_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    float norm_ell_d = 4.0f * n_half * 4;  /* D=4 */
    float norm_ell3 = 4.0f * n_half * 3;
    float inv_ell2 = 1.0f / (4.0f * n_half * 2);
    float total_mass_f = (float)S;
    float m_sq_f = 1.0f / (inv_m_f * inv_m_f);

    for (long long work_idx = flat_idx; work_idx < total_pairs; work_idx += stride) {
        /* Map flat index to (c0, c1) via prefix sum table */
        int c0_idx = binary_search_le(prefix_sums, n_c0, work_idx);
        int c0 = c0_order[c0_idx];
        int c1 = (int)(work_idx - prefix_sums[c0_idx]);

        int r0 = S - c0;
        int c1_max = r0 - c0;  /* canonical: c1 <= S - 2*c0 */
        if (c1 > c1_max) continue;

        int r1 = r0 - c1;
        int c2_max = r1 - c0;  /* canonical: c3 = r1-c2 >= c0 */

        /* M4: ell=4 pair-sum bound */
        float pair_left = (float)(c0 + c1);
        float pair_right = (float)r1;
        if (pair_left * pair_left * inv_m_f * inv_m_f / norm_ell_d > thresh_f)
            continue;
        if (pair_right * pair_right * inv_m_f * inv_m_f / norm_ell_d > thresh_f)
            continue;

        /* Asymmetry check at (c0, c1) level */
        float left_frac = pair_left / total_mass_f;
        float dom = (left_frac > 0.5f) ? left_frac : (1.0f - left_frac);
        float asym_base = dom - margin_f;
        if (asym_base < 0.0f) asym_base = 0.0f;
        float asym_val = 2.0f * asym_base * asym_base;
        if (asym_val >= asym_limit_f) continue;

        /* M4: ell=3 right-window c2 lower bound (conservative for FP32) */
        float r1_sq = (float)r1 * (float)r1;
        float ell3_cutoff_sq = r1_sq - thresh_f * m_sq_f * norm_ell3;
        int c2_start = 0;
        if (ell3_cutoff_sq > 0.0f) {
            /* No +1: conservative for FP32 rounding */
            c2_start = (int)(sqrtf(ell3_cutoff_sq));
        }
        if (c2_start < 0) c2_start = 0;

        /* Inner c2 loop */
        for (int c2 = c2_start; c2 <= c2_max; c2++) {
            int c3 = r1 - c2;

            /* Canonical palindrome check: if c0==c3, need c1<=c2 */
            if (c0 == c3 && c1 > c2) continue;

            /* ell=2 max element bound (skip c0: it's the smallest in canonical) */
            int max_c = c3;
            if (c1 > max_c) max_c = c1;
            if (c2 > max_c) max_c = c2;
            float max_a = max_c * inv_m_f;
            if (max_a * max_a * inv_ell2 > thresh_f) continue;

            /* Survivor: write to compacted buffer */
            unsigned int idx = atomicAdd(survivor_count, 1u);
            if (idx < (unsigned int)max_survivors) {
                survivor_buf[idx * 3 + 0] = c0;
                survivor_buf[idx * 3 + 1] = c1;
                survivor_buf[idx * 3 + 2] = c2;
            }
        }
    }
}


/* ================================================================
 * Phase 1 kernel: D=6
 *
 * One thread per (c0, c1, c2) triple. Inner (c3, c4) loop.
 * ================================================================ */
__global__ void phase1_d6(
    int S,
    int n_half,
    float inv_m_f,
    float thresh_f,
    float margin_f,
    float asym_limit_f,
    const long long* __restrict__ prefix_sums,  /* cumulative triple counts per c0 */
    const int* __restrict__ c0_order,
    int n_c0,
    long long total_triples,
    int* __restrict__ survivor_buf,     /* 5 int32 per survivor */
    unsigned int* __restrict__ survivor_count,
    int max_survivors
) {
    long long flat_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    float norm_ell_d = 4.0f * n_half * 6;  /* D=6 */
    float inv_ell2 = 1.0f / (4.0f * n_half * 2);
    float total_mass_f = (float)S;

    for (long long work_idx = flat_idx; work_idx < total_triples; work_idx += stride) {
        /* Map flat index to (c0, c1, c2) */
        int c0_idx = binary_search_le(prefix_sums, n_c0, work_idx);
        int c0 = c0_order[c0_idx];
        long long local_idx = work_idx - prefix_sums[c0_idx];

        /* Within c0: (c1, c2) in triangle with side R = S - 2*c0 */
        int R = S - 2 * c0;
        if (R < 0) continue;

        /* Map local_idx to (c1, c2) using quadratic formula.
         * cum(c1) = c1*(2R - c1 + 3)/2
         * Solve: c1^2 - (2R+3)*c1 + 2*local_idx <= 0 */
        float Rf = (float)(2 * R + 3);
        float disc = Rf * Rf - 8.0f * (float)local_idx;
        if (disc < 0.0f) disc = 0.0f;
        int c1 = (int)((Rf - sqrtf(disc)) * 0.5f);

        /* Adjust for FP32 rounding */
        long long R_ll = (long long)R;
        while (c1 > 0) {
            long long cum = (long long)c1 * (2LL * R_ll - (long long)c1 + 3LL) / 2LL;
            if (cum <= local_idx) break;
            c1--;
        }
        while (c1 < R) {
            long long cum_next = (long long)(c1 + 1) * (2LL * R_ll - (long long)c1 + 2LL) / 2LL;
            if (cum_next > local_idx) break;
            c1++;
        }

        long long cum_c1 = (long long)c1 * (2LL * R_ll - (long long)c1 + 3LL) / 2LL;
        int c2 = (int)(local_idx - cum_c1);
        if (c1 < 0 || c1 > R || c2 < 0 || c2 > R - c1) continue;

        int r2 = S - c0 - c1 - c2;

        /* M4: ell=6 three-bin sum bound */
        float left3 = (float)(c0 + c1 + c2);
        float right3 = (float)r2;
        if (left3 * left3 * inv_m_f * inv_m_f / norm_ell_d > thresh_f) continue;
        if (right3 * right3 * inv_m_f * inv_m_f / norm_ell_d > thresh_f) continue;

        /* Asymmetry at half-point (d/2 = 3 components) */
        float left_frac = left3 / total_mass_f;
        float dom = (left_frac > 0.5f) ? left_frac : (1.0f - left_frac);
        float asym_base = dom - margin_f;
        if (asym_base < 0.0f) asym_base = 0.0f;
        float asym_val = 2.0f * asym_base * asym_base;
        if (asym_val >= asym_limit_f) continue;

        /* Inner (c3, c4) loop */
        for (int c3 = 0; c3 <= r2; c3++) {
            int r3 = r2 - c3;
            int c4_max = r3 - c0;  /* canonical: c5 = r3-c4 >= c0 */
            if (c4_max < 0) break;

            for (int c4 = 0; c4 <= c4_max; c4++) {
                int c5 = r3 - c4;

                /* Canonical: if c0==c5, check (c1,c2) vs (c4,c3) */
                if (c0 == c5) {
                    if (c1 > c4) continue;
                    if (c1 == c4 && c2 > c3) continue;
                }

                /* ell=2 max element bound */
                int max_c = c5;
                if (c1 > max_c) max_c = c1;
                if (c2 > max_c) max_c = c2;
                if (c3 > max_c) max_c = c3;
                if (c4 > max_c) max_c = c4;
                float max_a = max_c * inv_m_f;
                if (max_a * max_a * inv_ell2 > thresh_f) continue;

                /* Survivor */
                unsigned int idx = atomicAdd(survivor_count, 1u);
                if (idx < (unsigned int)max_survivors) {
                    survivor_buf[idx * 5 + 0] = c0;
                    survivor_buf[idx * 5 + 1] = c1;
                    survivor_buf[idx * 5 + 2] = c2;
                    survivor_buf[idx * 5 + 3] = c3;
                    survivor_buf[idx * 5 + 4] = c4;
                }
            }
        }
    }
}


/* ================================================================
 * Phase 2 kernel: find_min mode (templated on D)
 *
 * One thread per survivor. Integer autoconvolution.
 * Block reduction for minimum effective value.
 * Per-block output: (min_val, min_config).
 * ================================================================ */

#define PHASE2_BLOCK_SIZE 256

template <int D>
__global__ void phase2_find_min(
    int S,
    int n_half,
    int m,
    double corr,
    double margin,      /* 1/(4*m) */
    const int* __restrict__ survivor_buf,   /* (D-1) int32 per survivor */
    unsigned int num_survivors,
    double* __restrict__ block_min_vals,     /* [num_blocks] */
    int* __restrict__ block_min_configs      /* [num_blocks * D] */
) {
    unsigned int tid = (unsigned int)blockIdx.x * blockDim.x + threadIdx.x;

    constexpr int HALF_D = D / 2;
    constexpr int CONV_LEN = 2 * D - 1;
    constexpr int STORED = D - 1;

    double my_eff = 1e30;
    int my_cfg[D];
    #pragma unroll
    for (int i = 0; i < D; i++) my_cfg[i] = 0;

    if (tid < num_survivors) {
        /* Load survivor tuple */
        int c[D];
        int csum = 0;
        #pragma unroll
        for (int i = 0; i < STORED; i++) {
            c[i] = survivor_buf[(unsigned long long)tid * STORED + i];
            csum += c[i];
        }
        c[D - 1] = S - csum;

        /* Integer autoconvolution: c[i]*c[j] in INT32->INT64 */
        long long conv[CONV_LEN];
        #pragma unroll
        for (int k = 0; k < CONV_LEN; k++) conv[k] = 0;

        #pragma unroll
        for (int i = 0; i < D; i++) {
            long long ci = (long long)c[i];
            conv[2 * i] += ci * ci;
            #pragma unroll
            for (int j = i + 1; j < D; j++) {
                conv[i + j] += 2LL * ci * (long long)c[j];
            }
        }

        /* Integer prefix sums */
        #pragma unroll
        for (int k = 1; k < CONV_LEN; k++) {
            conv[k] += conv[k - 1];
        }

        /* Window max: convert to FP64 only at normalization step */
        double inv_m_sq = 1.0 / ((double)m * (double)m);
        double best = 0.0;

        #pragma unroll
        for (int ell = D; ell >= 2; ell--) {
            int n_cv = ell - 1;
            double inv_norm = inv_m_sq / (4.0 * n_half * ell);
            int n_windows = 2 * D - ell + 1;
            for (int s_lo = 0; s_lo < n_windows; s_lo++) {
                int s_hi = s_lo + n_cv - 1;
                long long ws = conv[s_hi];
                if (s_lo > 0) ws -= conv[s_lo - 1];
                double tv = (double)ws * inv_norm;
                if (tv > best) best = tv;
            }
        }

        /* Effective value = max(test_val - corr, asym_bound) */
        double eff = best - corr;

        /* Asymmetry bound */
        double total_mass = (double)S;
        double left_sum = 0.0;
        #pragma unroll
        for (int i = 0; i < HALF_D; i++) left_sum += (double)c[i];
        double left_frac = left_sum / total_mass;
        double dom_d = (left_frac > 0.5) ? left_frac : (1.0 - left_frac);
        double asym_base_d = dom_d - margin;
        if (asym_base_d < 0.0) asym_base_d = 0.0;
        double asym_val_d = 2.0 * asym_base_d * asym_base_d;
        if (asym_val_d > eff) eff = asym_val_d;

        my_eff = eff;
        #pragma unroll
        for (int i = 0; i < D; i++) my_cfg[i] = c[i];
    }

    /* Block-level tree reduction for minimum */
    __shared__ double s_vals[PHASE2_BLOCK_SIZE];
    __shared__ int s_cfgs[PHASE2_BLOCK_SIZE * D];

    int lane = threadIdx.x;
    s_vals[lane] = my_eff;
    #pragma unroll
    for (int i = 0; i < D; i++) s_cfgs[lane * D + i] = my_cfg[i];
    __syncthreads();

    for (int s = PHASE2_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (lane < s && s_vals[lane + s] < s_vals[lane]) {
            s_vals[lane] = s_vals[lane + s];
            #pragma unroll
            for (int i = 0; i < D; i++)
                s_cfgs[lane * D + i] = s_cfgs[(lane + s) * D + i];
        }
        __syncthreads();
    }

    /* Write per-block result */
    if (lane == 0) {
        block_min_vals[blockIdx.x] = s_vals[0];
        #pragma unroll
        for (int i = 0; i < D; i++)
            block_min_configs[blockIdx.x * D + i] = s_cfgs[i];
    }
}


/* ================================================================
 * Phase 2 kernel: prove_target mode (templated on D)
 *
 * Same autoconvolution, but checks against a threshold.
 * Per-block output: counts (asym_pruned, test_pruned, survived)
 *                   and min test value among survivors.
 * ================================================================ */
template <int D>
__global__ void phase2_prove_target(
    int S,
    int n_half,
    int m,
    double corr,
    double margin,
    double c_target,       /* asymmetry comparison target */
    double thresh,         /* prune_target + fp_margin (FP64 exact) */
    const int* __restrict__ survivor_buf,
    unsigned int num_survivors,
    /* Per-block outputs */
    long long* __restrict__ block_counts,    /* [num_blocks * 3]: asym, test, surv */
    double* __restrict__ block_min_tv,       /* [num_blocks] */
    int* __restrict__ block_min_configs      /* [num_blocks * D] */
) {
    unsigned int tid = (unsigned int)blockIdx.x * blockDim.x + threadIdx.x;

    constexpr int HALF_D = D / 2;
    constexpr int CONV_LEN = 2 * D - 1;
    constexpr int STORED = D - 1;

    int is_asym = 0, is_test = 0, is_surv = 0;
    double my_tv = 1e30;
    int my_cfg[D];
    #pragma unroll
    for (int i = 0; i < D; i++) my_cfg[i] = 0;

    if (tid < num_survivors) {
        int c[D];
        int csum = 0;
        #pragma unroll
        for (int i = 0; i < STORED; i++) {
            c[i] = survivor_buf[(unsigned long long)tid * STORED + i];
            csum += c[i];
        }
        c[D - 1] = S - csum;

        /* Asymmetry check (FP64) */
        double total_mass = (double)S;
        double left_sum = 0.0;
        #pragma unroll
        for (int i = 0; i < HALF_D; i++) left_sum += (double)c[i];
        double left_frac = left_sum / total_mass;
        double dom_d = (left_frac > 0.5) ? left_frac : (1.0 - left_frac);
        double asym_base_d = dom_d - margin;
        if (asym_base_d < 0.0) asym_base_d = 0.0;
        double asym_val_d = 2.0 * asym_base_d * asym_base_d;

        if (asym_val_d >= c_target) {
            is_asym = 1;
        } else {
            /* Full autoconvolution (integer) */
            long long conv[CONV_LEN];
            #pragma unroll
            for (int k = 0; k < CONV_LEN; k++) conv[k] = 0;

            #pragma unroll
            for (int i = 0; i < D; i++) {
                long long ci = (long long)c[i];
                conv[2 * i] += ci * ci;
                #pragma unroll
                for (int j = i + 1; j < D; j++) {
                    conv[i + j] += 2LL * ci * (long long)c[j];
                }
            }

            #pragma unroll
            for (int k = 1; k < CONV_LEN; k++) conv[k] += conv[k - 1];

            double inv_m_sq = 1.0 / ((double)m * (double)m);
            double best = 0.0;

            #pragma unroll
            for (int ell = D; ell >= 2; ell--) {
                int n_cv = ell - 1;
                double inv_norm = inv_m_sq / (4.0 * n_half * ell);
                int n_windows = 2 * D - ell + 1;
                for (int s_lo = 0; s_lo < n_windows; s_lo++) {
                    int s_hi = s_lo + n_cv - 1;
                    long long ws = conv[s_hi];
                    if (s_lo > 0) ws -= conv[s_lo - 1];
                    double tv = (double)ws * inv_norm;
                    if (tv > best) {
                        best = tv;
                        if (best > thresh) break;
                    }
                }
                if (best > thresh) break;
            }

            if (best > thresh) {
                is_test = 1;
            } else {
                is_surv = 1;
                my_tv = best;
                #pragma unroll
                for (int i = 0; i < D; i++) my_cfg[i] = c[i];
            }
        }
    }

    /* Block-level reduction: sum counts, min test value among survivors */
    __shared__ int s_asym, s_test, s_surv;
    __shared__ double s_min_tv[PHASE2_BLOCK_SIZE];
    __shared__ int s_min_cfg[PHASE2_BLOCK_SIZE * D];

    int lane = threadIdx.x;
    if (lane == 0) { s_asym = 0; s_test = 0; s_surv = 0; }
    __syncthreads();

    if (is_asym) atomicAdd(&s_asym, 1);
    if (is_test) atomicAdd(&s_test, 1);
    if (is_surv) atomicAdd(&s_surv, 1);

    s_min_tv[lane] = my_tv;
    #pragma unroll
    for (int i = 0; i < D; i++) s_min_cfg[lane * D + i] = my_cfg[i];
    __syncthreads();

    /* Tree reduction for min test value */
    for (int s = PHASE2_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (lane < s && s_min_tv[lane + s] < s_min_tv[lane]) {
            s_min_tv[lane] = s_min_tv[lane + s];
            #pragma unroll
            for (int i = 0; i < D; i++)
                s_min_cfg[lane * D + i] = s_min_cfg[(lane + s) * D + i];
        }
        __syncthreads();
    }

    if (lane == 0) {
        block_counts[blockIdx.x * 3 + 0] = (long long)s_asym;
        block_counts[blockIdx.x * 3 + 1] = (long long)s_test;
        block_counts[blockIdx.x * 3 + 2] = (long long)s_surv;
        block_min_tv[blockIdx.x] = s_min_tv[0];
        #pragma unroll
        for (int i = 0; i < D; i++)
            block_min_configs[blockIdx.x * D + i] = s_min_cfg[i];
    }
}


/* ================================================================
 * Host implementation: find_best_bound_direct for D=4
 * Uses chunked processing to handle arbitrary m values.
 * ================================================================ */
static int find_best_bound_direct_d4(
    int S, int n_half, int m,
    double init_min_eff,
    double* result_min_eff,
    int* result_min_config
) {
    const int D = 4;
    const int STORED = D - 1;  /* 3 */
    double corr = 2.0 / m + 1.0 / ((double)m * m);
    double margin = 1.0 / (4.0 * m);
    float inv_m_f = 1.0f / (float)m;

    /* FP32-inflated threshold for Phase 1 (conservative: never falsely prune) */
    float thresh_f = (float)(init_min_eff + corr) * (1.0f + 1e-5f);
    float margin_f = (float)margin;
    float asym_limit_f = (float)init_min_eff * (1.0f + 1e-5f);

    int half_S = S / 2;
    int n_c0 = half_S + 1;

    /* Build interleaved (zigzag) c0 ordering */
    int* h_c0_order = (int*)malloc(n_c0 * sizeof(int));
    {
        int lo = 0, hi = half_S, idx = 0;
        while (lo <= hi) {
            h_c0_order[idx++] = lo;
            if (lo < hi) h_c0_order[idx++] = hi;
            lo++; hi--;
        }
    }

    /* Compute per-c0 counts:
     * - pairs: number of c1 values (for Phase 1 work distribution)
     * - configs: number of (c1,c2) tuples (upper bound on survivors) */
    long long* h_per_c0_pairs = (long long*)malloc(n_c0 * sizeof(long long));
    long long* h_per_c0_configs = (long long*)malloc(n_c0 * sizeof(long long));
    for (int i = 0; i < n_c0; i++) {
        int c0 = h_c0_order[i];
        long long R = (long long)(S - 2 * c0);
        h_per_c0_pairs[i] = R + 1;            /* c1 in [0, S-2*c0] */
        h_per_c0_configs[i] = (R + 1) * (R + 2) / 2;  /* upper bound on survivors */
    }

    /* Determine max buffer size based on GPU memory (use 70% of free) */
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    long long max_buffer = (long long)((double)free_mem * 0.7) / (STORED * (int)sizeof(int));
    if (max_buffer > (1LL << 30)) max_buffer = 1LL << 30;
    if (max_buffer < 10000) max_buffer = 10000;

    /* Persistent GPU buffers for survivor buffer + counter */
    int* d_survivor_buf = NULL;
    unsigned int* d_survivor_count = NULL;
    int actual_buf_size = 0;

    /* Process c0 values in chunks that fit in the buffer */
    double best_eff = init_min_eff;
    int best_config[D];
    for (int i = 0; i < D; i++) best_config[i] = 0;
    int found_better = 0;

    int chunk_start = 0;
    while (chunk_start < n_c0) {
        /* Group c0 values until total configs would exceed buffer */
        int chunk_end = chunk_start;
        long long chunk_configs = 0;
        long long chunk_pairs = 0;
        while (chunk_end < n_c0) {
            if (chunk_configs + h_per_c0_configs[chunk_end] > max_buffer
                && chunk_end > chunk_start) break;
            chunk_configs += h_per_c0_configs[chunk_end];
            chunk_pairs += h_per_c0_pairs[chunk_end];
            chunk_end++;
        }
        int chunk_n_c0 = chunk_end - chunk_start;
        int max_survivors = (int)fmin((double)chunk_configs, (double)((1LL << 30) - 1));

        /* Build prefix sums and c0 order for this chunk */
        long long* h_chunk_prefix = (long long*)malloc((chunk_n_c0 + 1) * sizeof(long long));
        int* h_chunk_c0 = (int*)malloc(chunk_n_c0 * sizeof(int));
        h_chunk_prefix[0] = 0;
        for (int i = 0; i < chunk_n_c0; i++) {
            h_chunk_c0[i] = h_c0_order[chunk_start + i];
            h_chunk_prefix[i + 1] = h_chunk_prefix[i] + h_per_c0_pairs[chunk_start + i];
        }
        long long chunk_total_pairs = h_chunk_prefix[chunk_n_c0];

        /* Allocate/reallocate GPU buffers */
        if (max_survivors > actual_buf_size) {
            if (d_survivor_buf) cudaFree(d_survivor_buf);
            CUDA_CHECK(cudaMalloc(&d_survivor_buf,
                (long long)max_survivors * STORED * sizeof(int)));
            actual_buf_size = max_survivors;
        }
        if (!d_survivor_count) {
            CUDA_CHECK(cudaMalloc(&d_survivor_count, sizeof(unsigned int)));
        }
        CUDA_CHECK(cudaMemset(d_survivor_count, 0, sizeof(unsigned int)));

        /* Upload chunk tables */
        long long* d_prefix = NULL;
        int* d_c0_order = NULL;
        CUDA_CHECK(cudaMalloc(&d_prefix, (chunk_n_c0 + 1) * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_c0_order, chunk_n_c0 * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_prefix, h_chunk_prefix,
            (chunk_n_c0 + 1) * sizeof(long long), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_c0_order, h_chunk_c0,
            chunk_n_c0 * sizeof(int), cudaMemcpyHostToDevice));

        /* Launch Phase 1 for this chunk */
        int block_size = 256;
        int grid_size = (int)fmin(
            (double)(chunk_total_pairs + block_size - 1) / block_size, 65535.0);

        phase1_d4<<<grid_size, block_size>>>(
            S, n_half, inv_m_f, thresh_f, margin_f, asym_limit_f,
            d_prefix, d_c0_order, chunk_n_c0, chunk_total_pairs,
            d_survivor_buf, d_survivor_count, max_survivors);
        CUDA_CHECK(cudaGetLastError());

        /* Read survivor count */
        unsigned int h_survivor_count = 0;
        CUDA_CHECK(cudaMemcpy(&h_survivor_count, d_survivor_count,
            sizeof(unsigned int), cudaMemcpyDeviceToHost));
        if (h_survivor_count > (unsigned int)max_survivors)
            h_survivor_count = (unsigned int)max_survivors;

        /* Phase 2 */
        if (h_survivor_count > 0) {
            int p2_block = PHASE2_BLOCK_SIZE;
            int p2_grid = (h_survivor_count + p2_block - 1) / p2_block;

            double* d_block_min_vals = NULL;
            int* d_block_min_configs = NULL;
            CUDA_CHECK(cudaMalloc(&d_block_min_vals, p2_grid * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_block_min_configs, p2_grid * D * sizeof(int)));

            phase2_find_min<D><<<p2_grid, p2_block>>>(
                S, n_half, m, corr, margin,
                d_survivor_buf, h_survivor_count,
                d_block_min_vals, d_block_min_configs);
            CUDA_CHECK(cudaGetLastError());

            double* h_block_vals = (double*)malloc(p2_grid * sizeof(double));
            int* h_block_cfgs = (int*)malloc(p2_grid * D * sizeof(int));
            CUDA_CHECK(cudaMemcpy(h_block_vals, d_block_min_vals,
                p2_grid * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_block_cfgs, d_block_min_configs,
                p2_grid * D * sizeof(int), cudaMemcpyDeviceToHost));

            for (int b = 0; b < p2_grid; b++) {
                if (h_block_vals[b] < best_eff) {
                    best_eff = h_block_vals[b];
                    for (int i = 0; i < D; i++)
                        best_config[i] = h_block_cfgs[b * D + i];
                    found_better = 1;
                }
            }

            free(h_block_vals);
            free(h_block_cfgs);
            cudaFree(d_block_min_vals);
            cudaFree(d_block_min_configs);
        }

        cudaFree(d_prefix);
        cudaFree(d_c0_order);
        free(h_chunk_prefix);
        free(h_chunk_c0);
        chunk_start = chunk_end;
    }

    *result_min_eff = best_eff;
    if (found_better) {
        for (int i = 0; i < D; i++) result_min_config[i] = best_config[i];
    }

    if (d_survivor_buf) cudaFree(d_survivor_buf);
    if (d_survivor_count) cudaFree(d_survivor_count);
    free(h_c0_order);
    free(h_per_c0_pairs);
    free(h_per_c0_configs);

    return 0;
}


/* ================================================================
 * Host implementation: find_best_bound_direct for D=6
 * Uses chunked processing.
 * ================================================================ */
static int find_best_bound_direct_d6(
    int S, int n_half, int m,
    double init_min_eff,
    double* result_min_eff,
    int* result_min_config
) {
    const int D = 6;
    const int STORED = D - 1;  /* 5 */
    double corr = 2.0 / m + 1.0 / ((double)m * m);
    double margin = 1.0 / (4.0 * m);
    float inv_m_f = 1.0f / (float)m;

    float thresh_f = (float)(init_min_eff + corr) * (1.0f + 1e-5f);
    float margin_f = (float)margin;
    float asym_limit_f = (float)init_min_eff * (1.0f + 1e-5f);

    int half_S = S / 2;
    int n_c0 = half_S + 1;

    /* Build zigzag c0 ordering */
    int* h_c0_order = (int*)malloc(n_c0 * sizeof(int));
    {
        int lo = 0, hi = half_S, idx = 0;
        while (lo <= hi) {
            h_c0_order[idx++] = lo;
            if (lo < hi) h_c0_order[idx++] = hi;
            lo++; hi--;
        }
    }

    /* Per-c0 counts:
     * triples: (c1,c2) pairs per c0 = (R+1)*(R+2)/2 (for Phase 1 work)
     * configs: total (c1,c2,c3,c4) tuples per c0 (upper bound on survivors)
     * For D=6, inner loop is (c3,c4) for each (c1,c2). Survivors per (c0,c1,c2)
     * triple is at most (r2+1)*(r2+2)/2 where r2=S-c0-c1-c2.
     * Total survivors per c0 is bounded by sum over (c1,c2) of (r2+1)(r2+2)/2.
     * This is expensive to compute exactly; use the triangle count as buffer size
     * since Phase 1 filters (c0,c1,c2) triples and for each triple, the inner
     * loop generates at most O(S^2) survivors. For safety, compute exact count. */
    long long* h_per_c0_triples = (long long*)malloc(n_c0 * sizeof(long long));
    long long* h_per_c0_configs = (long long*)malloc(n_c0 * sizeof(long long));
    for (int i = 0; i < n_c0; i++) {
        int c0 = h_c0_order[i];
        long long R = (long long)(S - 2 * c0);
        h_per_c0_triples[i] = (R + 1) * (R + 2) / 2;
        /* Upper bound on survivors: for each (c1,c2) triple, the inner (c3,c4)
         * loop generates at most (r2-c0+1)*(r2-c0+2)/2 configs where r2=S-c0-c1-c2.
         * Sum over all (c1,c2): bounded by (R+1)(R+2)(R+3)(R+4)/24 for large R.
         * Use a simpler upper bound: triples * (R/3 + 1) */
        /* Upper bound on survivors: C(R+4, 4) = total (c1,c2,c3,c4) tuples
         * with sum <= R and all >= 0. This is exact for the unpruned case. */
        h_per_c0_configs[i] = (R + 4) * (R + 3) * (R + 2) * (R + 1) / 24;
    }

    /* Max buffer from GPU memory */
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    long long max_buffer = (long long)((double)free_mem * 0.7) / (STORED * (int)sizeof(int));
    if (max_buffer > (1LL << 30)) max_buffer = 1LL << 30;
    if (max_buffer < 10000) max_buffer = 10000;

    int* d_survivor_buf = NULL;
    unsigned int* d_survivor_count = NULL;
    int actual_buf_size = 0;

    double best_eff = init_min_eff;
    int best_config[D];
    for (int i = 0; i < D; i++) best_config[i] = 0;
    int found_better = 0;

    int chunk_start = 0;
    while (chunk_start < n_c0) {
        int chunk_end = chunk_start;
        long long chunk_configs = 0;
        long long chunk_triples = 0;
        while (chunk_end < n_c0) {
            if (chunk_configs + h_per_c0_configs[chunk_end] > max_buffer
                && chunk_end > chunk_start) break;
            chunk_configs += h_per_c0_configs[chunk_end];
            chunk_triples += h_per_c0_triples[chunk_end];
            chunk_end++;
        }
        int chunk_n_c0 = chunk_end - chunk_start;
        int max_survivors = (int)fmin((double)chunk_configs, (double)((1LL << 30) - 1));

        long long* h_chunk_prefix = (long long*)malloc((chunk_n_c0 + 1) * sizeof(long long));
        int* h_chunk_c0 = (int*)malloc(chunk_n_c0 * sizeof(int));
        h_chunk_prefix[0] = 0;
        for (int i = 0; i < chunk_n_c0; i++) {
            h_chunk_c0[i] = h_c0_order[chunk_start + i];
            h_chunk_prefix[i + 1] = h_chunk_prefix[i] + h_per_c0_triples[chunk_start + i];
        }
        long long chunk_total_triples = h_chunk_prefix[chunk_n_c0];

        if (max_survivors > actual_buf_size) {
            if (d_survivor_buf) cudaFree(d_survivor_buf);
            CUDA_CHECK(cudaMalloc(&d_survivor_buf,
                (long long)max_survivors * STORED * sizeof(int)));
            actual_buf_size = max_survivors;
        }
        if (!d_survivor_count) {
            CUDA_CHECK(cudaMalloc(&d_survivor_count, sizeof(unsigned int)));
        }
        CUDA_CHECK(cudaMemset(d_survivor_count, 0, sizeof(unsigned int)));

        long long* d_prefix = NULL;
        int* d_c0_order = NULL;
        CUDA_CHECK(cudaMalloc(&d_prefix, (chunk_n_c0 + 1) * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_c0_order, chunk_n_c0 * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_prefix, h_chunk_prefix,
            (chunk_n_c0 + 1) * sizeof(long long), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_c0_order, h_chunk_c0,
            chunk_n_c0 * sizeof(int), cudaMemcpyHostToDevice));

        int block_size = 256;
        int grid_size = (int)fmin(
            (double)(chunk_total_triples + block_size - 1) / block_size, 65535.0);

        phase1_d6<<<grid_size, block_size>>>(
            S, n_half, inv_m_f, thresh_f, margin_f, asym_limit_f,
            d_prefix, d_c0_order, chunk_n_c0, chunk_total_triples,
            d_survivor_buf, d_survivor_count, max_survivors);
        CUDA_CHECK(cudaGetLastError());

        unsigned int h_survivor_count = 0;
        CUDA_CHECK(cudaMemcpy(&h_survivor_count, d_survivor_count,
            sizeof(unsigned int), cudaMemcpyDeviceToHost));
        if (h_survivor_count > (unsigned int)max_survivors)
            h_survivor_count = (unsigned int)max_survivors;

        if (h_survivor_count > 0) {
            int p2_block = PHASE2_BLOCK_SIZE;
            int p2_grid = (h_survivor_count + p2_block - 1) / p2_block;

            double* d_block_min_vals = NULL;
            int* d_block_min_configs = NULL;
            CUDA_CHECK(cudaMalloc(&d_block_min_vals, p2_grid * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_block_min_configs, p2_grid * D * sizeof(int)));

            phase2_find_min<D><<<p2_grid, p2_block>>>(
                S, n_half, m, corr, margin,
                d_survivor_buf, h_survivor_count,
                d_block_min_vals, d_block_min_configs);
            CUDA_CHECK(cudaGetLastError());

            double* h_block_vals = (double*)malloc(p2_grid * sizeof(double));
            int* h_block_cfgs = (int*)malloc(p2_grid * D * sizeof(int));
            CUDA_CHECK(cudaMemcpy(h_block_vals, d_block_min_vals,
                p2_grid * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_block_cfgs, d_block_min_configs,
                p2_grid * D * sizeof(int), cudaMemcpyDeviceToHost));

            for (int b = 0; b < p2_grid; b++) {
                if (h_block_vals[b] < best_eff) {
                    best_eff = h_block_vals[b];
                    for (int i = 0; i < D; i++)
                        best_config[i] = h_block_cfgs[b * D + i];
                    found_better = 1;
                }
            }

            free(h_block_vals);
            free(h_block_cfgs);
            cudaFree(d_block_min_vals);
            cudaFree(d_block_min_configs);
        }

        cudaFree(d_prefix);
        cudaFree(d_c0_order);
        free(h_chunk_prefix);
        free(h_chunk_c0);
        chunk_start = chunk_end;
    }

    *result_min_eff = best_eff;
    if (found_better) {
        for (int i = 0; i < D; i++) result_min_config[i] = best_config[i];
    }

    if (d_survivor_buf) cudaFree(d_survivor_buf);
    if (d_survivor_count) cudaFree(d_survivor_count);
    free(h_c0_order);
    free(h_per_c0_triples);
    free(h_per_c0_configs);

    return 0;
}


/* ================================================================
 * Host implementation: run_single_level for D=4
 * Uses chunked processing.
 * ================================================================ */
static int run_single_level_d4(
    int S, int n_half, int m,
    double c_target,
    long long* out_n_pruned_asym,
    long long* out_n_pruned_test,
    long long* out_n_survivors,
    double* out_min_test_val,
    int* out_min_test_config
) {
    const int D = 4;
    const int STORED = D - 1;
    double corr = 2.0 / m + 1.0 / ((double)m * m);
    double prune_target = c_target + corr;
    double margin = 1.0 / (4.0 * m);
    double fp_margin = 1e-9;
    double thresh = prune_target + fp_margin;
    float inv_m_f = 1.0f / (float)m;

    float thresh_f = (float)thresh * (1.0f + 1e-5f);
    float margin_f = (float)margin;
    float asym_limit_f = (float)c_target * (1.0f + 1e-5f);

    int half_S = S / 2;
    int n_c0 = half_S + 1;

    int* h_c0_order = (int*)malloc(n_c0 * sizeof(int));
    {
        int lo = 0, hi = half_S, idx = 0;
        while (lo <= hi) {
            h_c0_order[idx++] = lo;
            if (lo < hi) h_c0_order[idx++] = hi;
            lo++; hi--;
        }
    }

    long long* h_per_c0_pairs = (long long*)malloc(n_c0 * sizeof(long long));
    long long* h_per_c0_configs = (long long*)malloc(n_c0 * sizeof(long long));
    for (int i = 0; i < n_c0; i++) {
        int c0 = h_c0_order[i];
        long long R = (long long)(S - 2 * c0);
        h_per_c0_pairs[i] = R + 1;
        h_per_c0_configs[i] = (R + 1) * (R + 2) / 2;
    }

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    long long max_buffer = (long long)((double)free_mem * 0.7) / (STORED * (int)sizeof(int));
    if (max_buffer > (1LL << 30)) max_buffer = 1LL << 30;
    if (max_buffer < 10000) max_buffer = 10000;

    int* d_survivor_buf = NULL;
    unsigned int* d_survivor_count = NULL;
    int actual_buf_size = 0;

    long long total_asym = 0, total_test = 0, total_surv = 0;
    double min_tv = 1e30;
    int min_cfg[D];
    for (int i = 0; i < D; i++) min_cfg[i] = 0;

    int chunk_start = 0;
    while (chunk_start < n_c0) {
        int chunk_end = chunk_start;
        long long chunk_configs = 0;
        long long chunk_pairs = 0;
        while (chunk_end < n_c0) {
            if (chunk_configs + h_per_c0_configs[chunk_end] > max_buffer
                && chunk_end > chunk_start) break;
            chunk_configs += h_per_c0_configs[chunk_end];
            chunk_pairs += h_per_c0_pairs[chunk_end];
            chunk_end++;
        }
        int chunk_n_c0 = chunk_end - chunk_start;
        int max_survivors = (int)fmin((double)chunk_configs, (double)((1LL << 30) - 1));

        long long* h_chunk_prefix = (long long*)malloc((chunk_n_c0 + 1) * sizeof(long long));
        int* h_chunk_c0 = (int*)malloc(chunk_n_c0 * sizeof(int));
        h_chunk_prefix[0] = 0;
        for (int i = 0; i < chunk_n_c0; i++) {
            h_chunk_c0[i] = h_c0_order[chunk_start + i];
            h_chunk_prefix[i + 1] = h_chunk_prefix[i] + h_per_c0_pairs[chunk_start + i];
        }
        long long chunk_total_pairs = h_chunk_prefix[chunk_n_c0];

        if (max_survivors > actual_buf_size) {
            if (d_survivor_buf) cudaFree(d_survivor_buf);
            CUDA_CHECK(cudaMalloc(&d_survivor_buf,
                (long long)max_survivors * STORED * sizeof(int)));
            actual_buf_size = max_survivors;
        }
        if (!d_survivor_count) {
            CUDA_CHECK(cudaMalloc(&d_survivor_count, sizeof(unsigned int)));
        }
        CUDA_CHECK(cudaMemset(d_survivor_count, 0, sizeof(unsigned int)));

        long long* d_prefix = NULL;
        int* d_c0_order = NULL;
        CUDA_CHECK(cudaMalloc(&d_prefix, (chunk_n_c0 + 1) * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_c0_order, chunk_n_c0 * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_prefix, h_chunk_prefix,
            (chunk_n_c0 + 1) * sizeof(long long), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_c0_order, h_chunk_c0,
            chunk_n_c0 * sizeof(int), cudaMemcpyHostToDevice));

        int block_size = 256;
        int grid_size = (int)fmin(
            (double)(chunk_total_pairs + block_size - 1) / block_size, 65535.0);

        phase1_d4<<<grid_size, block_size>>>(
            S, n_half, inv_m_f, thresh_f, margin_f, asym_limit_f,
            d_prefix, d_c0_order, chunk_n_c0, chunk_total_pairs,
            d_survivor_buf, d_survivor_count, max_survivors);
        CUDA_CHECK(cudaGetLastError());

        unsigned int h_survivor_count = 0;
        CUDA_CHECK(cudaMemcpy(&h_survivor_count, d_survivor_count,
            sizeof(unsigned int), cudaMemcpyDeviceToHost));
        if (h_survivor_count > (unsigned int)max_survivors)
            h_survivor_count = (unsigned int)max_survivors;

        if (h_survivor_count > 0) {
            int p2_block = PHASE2_BLOCK_SIZE;
            int p2_grid = (h_survivor_count + p2_block - 1) / p2_block;

            long long* d_block_counts = NULL;
            double* d_block_min_tv = NULL;
            int* d_block_min_configs = NULL;
            CUDA_CHECK(cudaMalloc(&d_block_counts, (long long)p2_grid * 3 * sizeof(long long)));
            CUDA_CHECK(cudaMalloc(&d_block_min_tv, p2_grid * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_block_min_configs, p2_grid * D * sizeof(int)));

            phase2_prove_target<D><<<p2_grid, p2_block>>>(
                S, n_half, m, corr, margin, c_target, thresh,
                d_survivor_buf, h_survivor_count,
                d_block_counts, d_block_min_tv, d_block_min_configs);
            CUDA_CHECK(cudaGetLastError());

            long long* h_counts = (long long*)malloc(p2_grid * 3 * sizeof(long long));
            double* h_min_tvs = (double*)malloc(p2_grid * sizeof(double));
            int* h_min_cfgs = (int*)malloc(p2_grid * D * sizeof(int));
            CUDA_CHECK(cudaMemcpy(h_counts, d_block_counts,
                p2_grid * 3 * sizeof(long long), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_min_tvs, d_block_min_tv,
                p2_grid * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_min_cfgs, d_block_min_configs,
                p2_grid * D * sizeof(int), cudaMemcpyDeviceToHost));

            for (int b = 0; b < p2_grid; b++) {
                total_asym += h_counts[b * 3 + 0];
                total_test += h_counts[b * 3 + 1];
                total_surv += h_counts[b * 3 + 2];
                if (h_min_tvs[b] < min_tv) {
                    min_tv = h_min_tvs[b];
                    for (int i = 0; i < D; i++)
                        min_cfg[i] = h_min_cfgs[b * D + i];
                }
            }

            free(h_counts);
            free(h_min_tvs);
            free(h_min_cfgs);
            cudaFree(d_block_counts);
            cudaFree(d_block_min_tv);
            cudaFree(d_block_min_configs);
        }

        cudaFree(d_prefix);
        cudaFree(d_c0_order);
        free(h_chunk_prefix);
        free(h_chunk_c0);
        chunk_start = chunk_end;
    }

    *out_n_pruned_asym = total_asym;
    *out_n_pruned_test = total_test;
    *out_n_survivors = total_surv;
    *out_min_test_val = min_tv;
    for (int i = 0; i < D; i++) out_min_test_config[i] = min_cfg[i];

    if (d_survivor_buf) cudaFree(d_survivor_buf);
    if (d_survivor_count) cudaFree(d_survivor_count);
    free(h_c0_order);
    free(h_per_c0_pairs);
    free(h_per_c0_configs);

    return 0;
}


/* ================================================================
 * Host implementation: run_single_level for D=6
 * Uses chunked processing.
 * ================================================================ */
static int run_single_level_d6(
    int S, int n_half, int m,
    double c_target,
    long long* out_n_pruned_asym,
    long long* out_n_pruned_test,
    long long* out_n_survivors,
    double* out_min_test_val,
    int* out_min_test_config
) {
    const int D = 6;
    const int STORED = D - 1;
    double corr = 2.0 / m + 1.0 / ((double)m * m);
    double prune_target = c_target + corr;
    double margin = 1.0 / (4.0 * m);
    double fp_margin = 1e-9;
    double thresh = prune_target + fp_margin;
    float inv_m_f = 1.0f / (float)m;

    float thresh_f = (float)thresh * (1.0f + 1e-5f);
    float margin_f = (float)margin;
    float asym_limit_f = (float)c_target * (1.0f + 1e-5f);

    int half_S = S / 2;
    int n_c0 = half_S + 1;

    int* h_c0_order = (int*)malloc(n_c0 * sizeof(int));
    {
        int lo = 0, hi = half_S, idx = 0;
        while (lo <= hi) {
            h_c0_order[idx++] = lo;
            if (lo < hi) h_c0_order[idx++] = hi;
            lo++; hi--;
        }
    }

    long long* h_per_c0_triples = (long long*)malloc(n_c0 * sizeof(long long));
    long long* h_per_c0_configs = (long long*)malloc(n_c0 * sizeof(long long));
    for (int i = 0; i < n_c0; i++) {
        int c0 = h_c0_order[i];
        long long R = (long long)(S - 2 * c0);
        h_per_c0_triples[i] = (R + 1) * (R + 2) / 2;
        /* Upper bound on survivors: C(R+4, 4) = total (c1,c2,c3,c4) tuples
         * with sum <= R and all >= 0. This is exact for the unpruned case. */
        h_per_c0_configs[i] = (R + 4) * (R + 3) * (R + 2) * (R + 1) / 24;
    }

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    long long max_buffer = (long long)((double)free_mem * 0.7) / (STORED * (int)sizeof(int));
    if (max_buffer > (1LL << 30)) max_buffer = 1LL << 30;
    if (max_buffer < 10000) max_buffer = 10000;

    int* d_survivor_buf = NULL;
    unsigned int* d_survivor_count = NULL;
    int actual_buf_size = 0;

    long long total_asym = 0, total_test = 0, total_surv = 0;
    double min_tv = 1e30;
    int min_cfg[D];
    for (int i = 0; i < D; i++) min_cfg[i] = 0;

    int chunk_start = 0;
    while (chunk_start < n_c0) {
        int chunk_end = chunk_start;
        long long chunk_configs = 0;
        long long chunk_triples = 0;
        while (chunk_end < n_c0) {
            if (chunk_configs + h_per_c0_configs[chunk_end] > max_buffer
                && chunk_end > chunk_start) break;
            chunk_configs += h_per_c0_configs[chunk_end];
            chunk_triples += h_per_c0_triples[chunk_end];
            chunk_end++;
        }
        int chunk_n_c0 = chunk_end - chunk_start;
        int max_survivors = (int)fmin((double)chunk_configs, (double)((1LL << 30) - 1));

        long long* h_chunk_prefix = (long long*)malloc((chunk_n_c0 + 1) * sizeof(long long));
        int* h_chunk_c0 = (int*)malloc(chunk_n_c0 * sizeof(int));
        h_chunk_prefix[0] = 0;
        for (int i = 0; i < chunk_n_c0; i++) {
            h_chunk_c0[i] = h_c0_order[chunk_start + i];
            h_chunk_prefix[i + 1] = h_chunk_prefix[i] + h_per_c0_triples[chunk_start + i];
        }
        long long chunk_total_triples = h_chunk_prefix[chunk_n_c0];

        if (max_survivors > actual_buf_size) {
            if (d_survivor_buf) cudaFree(d_survivor_buf);
            CUDA_CHECK(cudaMalloc(&d_survivor_buf,
                (long long)max_survivors * STORED * sizeof(int)));
            actual_buf_size = max_survivors;
        }
        if (!d_survivor_count) {
            CUDA_CHECK(cudaMalloc(&d_survivor_count, sizeof(unsigned int)));
        }
        CUDA_CHECK(cudaMemset(d_survivor_count, 0, sizeof(unsigned int)));

        long long* d_prefix = NULL;
        int* d_c0_order = NULL;
        CUDA_CHECK(cudaMalloc(&d_prefix, (chunk_n_c0 + 1) * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_c0_order, chunk_n_c0 * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_prefix, h_chunk_prefix,
            (chunk_n_c0 + 1) * sizeof(long long), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_c0_order, h_chunk_c0,
            chunk_n_c0 * sizeof(int), cudaMemcpyHostToDevice));

        int block_size = 256;
        int grid_size = (int)fmin(
            (double)(chunk_total_triples + block_size - 1) / block_size, 65535.0);

        phase1_d6<<<grid_size, block_size>>>(
            S, n_half, inv_m_f, thresh_f, margin_f, asym_limit_f,
            d_prefix, d_c0_order, chunk_n_c0, chunk_total_triples,
            d_survivor_buf, d_survivor_count, max_survivors);
        CUDA_CHECK(cudaGetLastError());

        unsigned int h_survivor_count = 0;
        CUDA_CHECK(cudaMemcpy(&h_survivor_count, d_survivor_count,
            sizeof(unsigned int), cudaMemcpyDeviceToHost));
        if (h_survivor_count > (unsigned int)max_survivors)
            h_survivor_count = (unsigned int)max_survivors;

        if (h_survivor_count > 0) {
            int p2_block = PHASE2_BLOCK_SIZE;
            int p2_grid = (h_survivor_count + p2_block - 1) / p2_block;

            long long* d_block_counts = NULL;
            double* d_block_min_tv = NULL;
            int* d_block_min_configs = NULL;
            CUDA_CHECK(cudaMalloc(&d_block_counts, (long long)p2_grid * 3 * sizeof(long long)));
            CUDA_CHECK(cudaMalloc(&d_block_min_tv, p2_grid * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_block_min_configs, p2_grid * D * sizeof(int)));

            phase2_prove_target<D><<<p2_grid, p2_block>>>(
                S, n_half, m, corr, margin, c_target, thresh,
                d_survivor_buf, h_survivor_count,
                d_block_counts, d_block_min_tv, d_block_min_configs);
            CUDA_CHECK(cudaGetLastError());

            long long* h_counts = (long long*)malloc(p2_grid * 3 * sizeof(long long));
            double* h_min_tvs = (double*)malloc(p2_grid * sizeof(double));
            int* h_min_cfgs = (int*)malloc(p2_grid * D * sizeof(int));
            CUDA_CHECK(cudaMemcpy(h_counts, d_block_counts,
                p2_grid * 3 * sizeof(long long), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_min_tvs, d_block_min_tv,
                p2_grid * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_min_cfgs, d_block_min_configs,
                p2_grid * D * sizeof(int), cudaMemcpyDeviceToHost));

            for (int b = 0; b < p2_grid; b++) {
                total_asym += h_counts[b * 3 + 0];
                total_test += h_counts[b * 3 + 1];
                total_surv += h_counts[b * 3 + 2];
                if (h_min_tvs[b] < min_tv) {
                    min_tv = h_min_tvs[b];
                    for (int i = 0; i < D; i++)
                        min_cfg[i] = h_min_cfgs[b * D + i];
                }
            }

            free(h_counts);
            free(h_min_tvs);
            free(h_min_cfgs);
            cudaFree(d_block_counts);
            cudaFree(d_block_min_tv);
            cudaFree(d_block_min_configs);
        }

        cudaFree(d_prefix);
        cudaFree(d_c0_order);
        free(h_chunk_prefix);
        free(h_chunk_c0);
        chunk_start = chunk_end;
    }

    *out_n_pruned_asym = total_asym;
    *out_n_pruned_test = total_test;
    *out_n_survivors = total_surv;
    *out_min_test_val = min_tv;
    for (int i = 0; i < D; i++) out_min_test_config[i] = min_cfg[i];

    if (d_survivor_buf) cudaFree(d_survivor_buf);
    if (d_survivor_count) cudaFree(d_survivor_count);
    free(h_c0_order);
    free(h_per_c0_triples);
    free(h_per_c0_configs);

    return 0;
}


/* ================================================================
 * C dispatch functions (extern "C")
 * ================================================================ */

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern "C" {

EXPORT int gpu_check_cuda() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) return 0;
    return 1;
}

EXPORT int gpu_get_device_name(char* buf, int buf_len) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return -1;
    snprintf(buf, buf_len, "%s (CC %d.%d, %d MB)",
             prop.name, prop.major, prop.minor,
             (int)(prop.totalGlobalMem / (1024 * 1024)));
    return 0;
}

EXPORT int gpu_find_best_bound_direct(
    int d, int S, int n_half, int m,
    double init_min_eff,
    double* result_min_eff,
    int* result_min_config
) {
    switch (d) {
        case 4: return find_best_bound_direct_d4(S, n_half, m,
                    init_min_eff, result_min_eff, result_min_config);
        case 6: return find_best_bound_direct_d6(S, n_half, m,
                    init_min_eff, result_min_eff, result_min_config);
        default:
            fprintf(stderr, "GPU: d=%d not supported (only d=4,6)\n", d);
            return -2;
    }
}

EXPORT int gpu_run_single_level(
    int d, int S, int n_half, int m,
    double c_target,
    long long* n_pruned_asym,
    long long* n_pruned_test,
    long long* n_survivors,
    double* min_test_val,
    int* min_test_config
) {
    switch (d) {
        case 4: return run_single_level_d4(S, n_half, m, c_target,
                    n_pruned_asym, n_pruned_test, n_survivors,
                    min_test_val, min_test_config);
        case 6: return run_single_level_d6(S, n_half, m, c_target,
                    n_pruned_asym, n_pruned_test, n_survivors,
                    min_test_val, min_test_config);
        default:
            fprintf(stderr, "GPU: d=%d not supported (only d=4,6)\n", d);
            return -2;
    }
}

}  /* extern "C" */
