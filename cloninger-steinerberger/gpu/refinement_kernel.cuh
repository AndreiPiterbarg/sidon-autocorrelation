#pragma once
/*
 * Refinement kernel for hierarchical multi-level branch-and-prune.
 *
 * Given a parent configuration B[0..d_parent-1] at level n, generates
 * all child configurations at level 2n via Cartesian product refinement:
 *   c_child[2i]   + c_child[2i+1] = B[i]   for each i  (S=m convention)
 *   c_child[2i]   in [max(0, B[i]-x_cap), min(B[i], x_cap)]
 *
 * Index mapping is simple iterated divmod (no simplex enumeration needed).
 *
 * Template-specialized for D_CHILD = 12 (unrolled), with generic loops
 * for D_CHILD = 24 and 48.
 *
 * Depends on: device_helpers.cuh (CUDA_CHECK, shfl_down_double)
 */

#define REFINE_BLOCK_SIZE_12  256
#define REFINE_BLOCK_SIZE_24  128
#define REFINE_BLOCK_SIZE_48   64
#define MAX_D_CHILD            48
#define MAX_BATCH_PARENTS  10000000  /* max parents per batched kernel launch */


/* ================================================================
 * Refinement kernel: process children of a single parent
 *
 * Each thread processes one child configuration from the Cartesian
 * product defined by the parent's bin values.
 * ================================================================ */
template <int D_CHILD>
__global__ void __launch_bounds__(
    (D_CHILD <= 12) ? REFINE_BLOCK_SIZE_12 :
    (D_CHILD <= 24) ? REFINE_BLOCK_SIZE_24 : REFINE_BLOCK_SIZE_48)
refine_prove_target(
    const int* __restrict__ parent_B,     /* [d_parent] parent bin values */
    const int* __restrict__ splits,       /* [d_parent] effective split count per bin (with x_cap) */
    int d_parent,
    int S_child,
    int n_half_child,
    int m,
    double c_target,
    double thresh,
    float inv_m_f,
    float thresh_f,
    float margin_f,
    float asym_limit_f,
    const long long* __restrict__ int_thresh,  /* [D_CHILD - 1] per-ell */
    long long work_offset,
    long long work_count,
    /* Outputs */
    long long* __restrict__ block_counts,       /* [num_blocks * 3]: asym, test, surv */
    double*    __restrict__ block_min_tv,
    int*       __restrict__ block_min_configs,   /* [num_blocks * D_CHILD] */
    int*       __restrict__ survivor_buf,        /* [max_survivors * D_CHILD] or NULL */
    int*       __restrict__ survivor_count,      /* atomic counter or NULL */
    int        max_survivors,
    int        x_cap                             /* single-bin energy cap */
) {
    constexpr int CONV_LEN = 2 * D_CHILD - 1;
    constexpr int D_PARENT = D_CHILD / 2;

    /* Load parent config into shared memory.
     * With S=m + energy cap: splits are effective counts. */
    __shared__ int s_parent_B[MAX_D_CHILD / 2];
    __shared__ int s_splits[MAX_D_CHILD / 2];
    if (threadIdx.x < D_PARENT) {
        s_parent_B[threadIdx.x] = parent_B[threadIdx.x];
        s_splits[threadIdx.x] = splits[threadIdx.x];
    }
    __syncthreads();

    long long flat_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    int my_asym = 0, my_test = 0, my_surv = 0;
    double my_tv = 1e30;
    int my_cfg[D_CHILD];
    for (int i = 0; i < D_CHILD; i++) my_cfg[i] = 0;

    /* FP32 scale factor: converts integer c to a-coordinates.
     * With S=m convention: a_j = (4n/m) * c_j. */
    float scale_f = 4.0f * n_half_child * inv_m_f;
    float total_mass_f = (float)S_child;
    float norm_ell_d = 4.0f * n_half_child * D_CHILD;
    float inv_ell2 = 1.0f / (4.0f * n_half_child * 2);

    double inv_S = 1.0 / (double)S_child;
    double margin = 1.0 / (4.0 * m);

    /* Precompute per-ell FP64 normalization factors.
     * With S=m: test_val = 4n_child/(m^2*ell) * sum(c_i*c_j). */
    double inv_norm_arr[D_CHILD - 1];
    for (int ell = 2; ell <= D_CHILD; ell++)
        inv_norm_arr[ell - 2] = (4.0 * n_half_child) / ((double)m * (double)m * (double)ell);

    /* Load integer thresholds into registers */
    long long local_thresh[D_CHILD - 1];
    for (int i = 0; i < D_CHILD - 1; i++)
        local_thresh[i] = int_thresh[i];

    for (long long work_idx = flat_idx; work_idx < work_count; work_idx += stride) {
        long long global_idx = work_offset + work_idx;

        /* === Index mapping: divmod decomposition into child config ===
         * With S=m + energy cap: c_even in [lo, lo+s-1] where
         * lo = max(0, B[i]-x_cap), s = effective split count. */
        int c[D_CHILD];
        {
            long long temp = global_idx;
            for (int i = 0; i < D_PARENT; i++) {
                int s = s_splits[i];
                int lo = (s_parent_B[i] > x_cap) ? (s_parent_B[i] - x_cap) : 0;
                int c_even = lo + (int)(temp % (long long)s);
                temp /= (long long)s;
                c[2*i]     = c_even;
                c[2*i + 1] = s_parent_B[i] - c_even;
            }
        }

        /* === 1. Canonical palindrome check === */
        {
            int skip = 0;
            for (int i = 0; i < D_CHILD / 2; i++) {
                if (c[i] < c[D_CHILD - 1 - i]) break;
                if (c[i] > c[D_CHILD - 1 - i]) { skip = 1; break; }
            }
            if (skip) continue;
        }

        /* === 2. FP32 asymmetry pre-check === */
        {
            float left_sum = 0.0f;
            for (int i = 0; i < D_CHILD / 2; i++) left_sum += (float)c[i];
            float left_frac = left_sum / total_mass_f;
            float dom = (left_frac > 0.5f) ? left_frac : (1.0f - left_frac);
            float asym_base = dom - margin_f;
            if (asym_base < 0.0f) asym_base = 0.0f;
            float asym_val = 2.0f * asym_base * asym_base;
            if (asym_val >= asym_limit_f) continue;
        }

        /* === 3. FP32 half-sum bound (ell=D_CHILD window) === */
        {
            float left_sum = 0.0f;
            for (int i = 0; i < D_CHILD / 2; i++) left_sum += (float)c[i];
            if (left_sum * left_sum * scale_f * scale_f / norm_ell_d > thresh_f) continue;
            float right_sum = total_mass_f - left_sum;
            if (right_sum * right_sum * scale_f * scale_f / norm_ell_d > thresh_f) continue;
        }

        /* === 4. FP32 ell=2 max element bound === */
        {
            int max_c = c[0];
            for (int i = 1; i < D_CHILD; i++) {
                if (c[i] > max_c) max_c = c[i];
            }
            float max_a = max_c * scale_f;
            if (max_a * max_a * inv_ell2 > thresh_f) continue;
        }

        /* === 5. FP64 asymmetry === */
        double asym_val_d;
        {
            double left_sum_d = 0.0;
            for (int i = 0; i < D_CHILD / 2; i++) left_sum_d += (double)c[i];
            double left_frac_d = left_sum_d * inv_S;
            double dom_d = (left_frac_d > 0.5) ? left_frac_d : (1.0 - left_frac_d);
            double asym_base_d = dom_d - margin;
            if (asym_base_d < 0.0) asym_base_d = 0.0;
            asym_val_d = 2.0 * asym_base_d * asym_base_d;
            if (asym_val_d >= c_target) { my_asym++; continue; }
        }

        /* === 6. Autoconvolution + integer threshold window scan === */
        long long conv[CONV_LEN];
        for (int k = 0; k < CONV_LEN; k++) conv[k] = 0;
        for (int i = 0; i < D_CHILD; i++) {
            conv[2*i] += (long long)c[i] * c[i];
            for (int j = i+1; j < D_CHILD; j++)
                conv[i+j] += 2LL * c[i] * c[j];
        }
        /* Prefix sum */
        for (int k = 1; k < CONV_LEN; k++) conv[k] += conv[k-1];

        /* Window scan with integer thresholds */
        int pruned = 0;
        for (int ell = 2; ell <= D_CHILD; ell++) {
            int n_cv = ell - 1;
            long long it = local_thresh[ell - 2];
            int n_windows = CONV_LEN - n_cv + 1;
            for (int s_lo = 0; s_lo < n_windows; s_lo++) {
                int s_hi = s_lo + n_cv - 1;
                long long ws = conv[s_hi];
                if (s_lo > 0) ws -= conv[s_lo - 1];
                if (ws > it) { pruned = 1; break; }
            }
            if (pruned) break;
        }

        if (pruned) {
            my_test++;
        } else {
            /* Survivor: compute FP64 test value for reporting */
            double best = 0.0;
            for (int ell = 2; ell <= D_CHILD; ell++) {
                int n_cv = ell - 1;
                double inv_norm = inv_norm_arr[ell - 2];
                int n_windows = CONV_LEN - n_cv + 1;
                for (int s_lo = 0; s_lo < n_windows; s_lo++) {
                    int s_hi = s_lo + n_cv - 1;
                    long long ws = conv[s_hi];
                    if (s_lo > 0) ws -= conv[s_lo - 1];
                    double tv = (double)ws * inv_norm;
                    if (tv > best) best = tv;
                }
            }
            my_surv++;
            if (survivor_buf != NULL) {
                /* Guard: skip atomicAdd once buffer is full to prevent
                 * int overflow when billions of configs survive */
                if (*survivor_count < max_survivors) {
                    int slot = atomicAdd(survivor_count, 1);
                    if (slot < max_survivors) {
                        for (int i = 0; i < D_CHILD; i++)
                            survivor_buf[slot * D_CHILD + i] = c[i];
                    }
                }
            }
            if (best < my_tv) {
                my_tv = best;
                for (int i = 0; i < D_CHILD; i++) my_cfg[i] = c[i];
            }
        }
    }

    /* === Per-block reduction: sum counts + min test value === */
    constexpr int BLOCK_SZ = (D_CHILD <= 12) ? REFINE_BLOCK_SIZE_12 :
                             (D_CHILD <= 24) ? REFINE_BLOCK_SIZE_24 :
                                                REFINE_BLOCK_SIZE_48;

    __shared__ int s_asym[BLOCK_SZ];
    __shared__ int s_test[BLOCK_SZ];
    __shared__ int s_surv[BLOCK_SZ];
    __shared__ double s_min_tv[BLOCK_SZ];
    __shared__ int s_cfgs[BLOCK_SZ * D_CHILD];

    int lane = threadIdx.x;
    s_asym[lane] = my_asym;
    s_test[lane] = my_test;
    s_surv[lane] = my_surv;
    s_min_tv[lane] = my_tv;
    for (int i = 0; i < D_CHILD; i++)
        s_cfgs[lane * D_CHILD + i] = my_cfg[i];
    __syncthreads();

    /* Cross-warp tree reduction */
    for (int s = BLOCK_SZ / 2; s > 16; s >>= 1) {
        if (lane < s) {
            s_asym[lane] += s_asym[lane + s];
            s_test[lane] += s_test[lane + s];
            s_surv[lane] += s_surv[lane + s];
            if (s_min_tv[lane + s] < s_min_tv[lane]) {
                s_min_tv[lane] = s_min_tv[lane + s];
                for (int i = 0; i < D_CHILD; i++)
                    s_cfgs[lane * D_CHILD + i] = s_cfgs[(lane + s) * D_CHILD + i];
            }
        }
        __syncthreads();
    }

    /* Intra-warp reduction (last 32 threads) */
    if (lane < 32) {
        int a = s_asym[lane], t = s_test[lane], sv = s_surv[lane];
        double val = s_min_tv[lane];

        for (int s = 16; s > 0; s >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, s);
            t += __shfl_down_sync(0xFFFFFFFF, t, s);
            sv += __shfl_down_sync(0xFFFFFFFF, sv, s);

            double other_val = shfl_down_double(0xFFFFFFFF, val, s);
            if (other_val < val) {
                val = other_val;
                int src_lane = lane + s;
                if (src_lane < 32) {
                    for (int i = 0; i < D_CHILD; i++)
                        s_cfgs[lane * D_CHILD + i] = s_cfgs[src_lane * D_CHILD + i];
                }
            }
            __syncwarp(0xFFFFFFFF);
        }

        if (lane == 0) {
            block_counts[blockIdx.x * 3 + 0] = (long long)a;
            block_counts[blockIdx.x * 3 + 1] = (long long)t;
            block_counts[blockIdx.x * 3 + 2] = (long long)sv;
            block_min_tv[blockIdx.x] = val;
            for (int i = 0; i < D_CHILD; i++)
                block_min_configs[blockIdx.x * D_CHILD + i] = s_cfgs[i];
        }
    }
}


/* ================================================================
 * Batched refinement kernel: processes children of MULTIPLE parents
 * in a single kernel launch.
 *
 * Each thread determines its parent via binary_search_le on prefix
 * sums of cumulative refinement counts.  Parent configs are read
 * from global memory (cached in L2, A100 has 40 MB).
 *
 * This eliminates per-parent kernel launch + cudaMemcpy overhead,
 * converting millions of tiny launches into a handful of large ones.
 * ================================================================ */
template <int D_CHILD>
__global__ void __launch_bounds__(
    (D_CHILD <= 12) ? REFINE_BLOCK_SIZE_12 :
    (D_CHILD <= 24) ? REFINE_BLOCK_SIZE_24 : REFINE_BLOCK_SIZE_48)
refine_prove_target_batched(
    const int* __restrict__ all_parents,       /* [n_batch * D_PARENT] parent configs */
    const long long* __restrict__ batch_prefix, /* [n_batch + 1] prefix sums, [0]=0 */
    int n_batch,
    int S_child,
    int n_half_child,
    int m,
    double c_target,
    double thresh,
    float inv_m_f,
    float thresh_f,
    float margin_f,
    float asym_limit_f,
    const long long* __restrict__ int_thresh,  /* [D_CHILD - 1] per-ell */
    long long total_work,
    /* Outputs */
    long long* __restrict__ block_counts,       /* [num_blocks * 3]: asym, test, surv */
    double*    __restrict__ block_min_tv,
    int*       __restrict__ block_min_configs,   /* [num_blocks * D_CHILD] */
    int*       __restrict__ survivor_buf,        /* [max_survivors * D_CHILD] or NULL */
    int*       __restrict__ survivor_count,      /* atomic counter or NULL */
    int        max_survivors,
    int        x_cap                             /* single-bin energy cap */
) {
    constexpr int CONV_LEN = 2 * D_CHILD - 1;
    constexpr int D_PARENT = D_CHILD / 2;

    long long flat_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    int my_asym = 0, my_test = 0, my_surv = 0;
    double my_tv = 1e30;
    int my_cfg[D_CHILD];
    for (int i = 0; i < D_CHILD; i++) my_cfg[i] = 0;

    /* FP32 scale factor: converts integer c to a-coordinates.
     * With S=m convention: a_j = (4n/m) * c_j. */
    float scale_f = 4.0f * n_half_child * inv_m_f;
    float total_mass_f = (float)S_child;
    float norm_ell_d = 4.0f * n_half_child * D_CHILD;
    float inv_ell2 = 1.0f / (4.0f * n_half_child * 2);

    double inv_S = 1.0 / (double)S_child;
    double margin = 1.0 / (4.0 * m);

    /* Precompute per-ell FP64 normalization factors.
     * With S=m: test_val = 4n_child/(m^2*ell) * sum(c_i*c_j). */
    double inv_norm_arr[D_CHILD - 1];
    for (int ell = 2; ell <= D_CHILD; ell++)
        inv_norm_arr[ell - 2] = (4.0 * n_half_child) / ((double)m * (double)m * (double)ell);

    /* Load integer thresholds into registers */
    long long local_thresh[D_CHILD - 1];
    for (int i = 0; i < D_CHILD - 1; i++)
        local_thresh[i] = int_thresh[i];

    for (long long work_idx = flat_idx; work_idx < total_work; work_idx += stride) {
        /* === Parent lookup via binary search on prefix sums === */
        int pidx = binary_search_le(batch_prefix, n_batch, work_idx);
        long long child_idx = work_idx - batch_prefix[pidx];
        const int* pB = all_parents + pidx * D_PARENT;

        /* === Index mapping: divmod decomposition into child config ===
         * With S=m + energy cap: c_even in [lo, lo+eff_count-1]. */
        int c[D_CHILD];
        {
            long long temp = child_idx;
            for (int i = 0; i < D_PARENT; i++) {
                int lo = (pB[i] > x_cap) ? (pB[i] - x_cap) : 0;
                int hi = (pB[i] < x_cap) ? pB[i] : x_cap;
                int s = hi - lo + 1;  /* effective split count with energy cap */
                int c_even = lo + (int)(temp % (long long)s);
                temp /= (long long)s;
                c[2*i]     = c_even;
                c[2*i + 1] = pB[i] - c_even;
            }
        }

        /* === 1. Canonical palindrome check === */
        {
            int skip = 0;
            for (int i = 0; i < D_CHILD / 2; i++) {
                if (c[i] < c[D_CHILD - 1 - i]) break;
                if (c[i] > c[D_CHILD - 1 - i]) { skip = 1; break; }
            }
            if (skip) continue;
        }

        /* === 2. FP32 asymmetry pre-check === */
        {
            float left_sum = 0.0f;
            for (int i = 0; i < D_CHILD / 2; i++) left_sum += (float)c[i];
            float left_frac = left_sum / total_mass_f;
            float dom = (left_frac > 0.5f) ? left_frac : (1.0f - left_frac);
            float asym_base = dom - margin_f;
            if (asym_base < 0.0f) asym_base = 0.0f;
            float asym_val = 2.0f * asym_base * asym_base;
            if (asym_val >= asym_limit_f) continue;
        }

        /* === 3. FP32 half-sum bound (ell=D_CHILD window) === */
        {
            float left_sum = 0.0f;
            for (int i = 0; i < D_CHILD / 2; i++) left_sum += (float)c[i];
            if (left_sum * left_sum * scale_f * scale_f / norm_ell_d > thresh_f) continue;
            float right_sum = total_mass_f - left_sum;
            if (right_sum * right_sum * scale_f * scale_f / norm_ell_d > thresh_f) continue;
        }

        /* === 4. FP32 ell=2 max element bound === */
        {
            int max_c = c[0];
            for (int i = 1; i < D_CHILD; i++) {
                if (c[i] > max_c) max_c = c[i];
            }
            float max_a = max_c * scale_f;
            if (max_a * max_a * inv_ell2 > thresh_f) continue;
        }

        /* === 5. FP64 asymmetry === */
        double asym_val_d;
        {
            double left_sum_d = 0.0;
            for (int i = 0; i < D_CHILD / 2; i++) left_sum_d += (double)c[i];
            double left_frac_d = left_sum_d * inv_S;
            double dom_d = (left_frac_d > 0.5) ? left_frac_d : (1.0 - left_frac_d);
            double asym_base_d = dom_d - margin;
            if (asym_base_d < 0.0) asym_base_d = 0.0;
            asym_val_d = 2.0 * asym_base_d * asym_base_d;
            if (asym_val_d >= c_target) { my_asym++; continue; }
        }

        /* === 6. Autoconvolution + integer threshold window scan === */
        long long conv[CONV_LEN];
        for (int k = 0; k < CONV_LEN; k++) conv[k] = 0;
        for (int i = 0; i < D_CHILD; i++) {
            conv[2*i] += (long long)c[i] * c[i];
            for (int j = i+1; j < D_CHILD; j++)
                conv[i+j] += 2LL * c[i] * c[j];
        }
        /* Prefix sum */
        for (int k = 1; k < CONV_LEN; k++) conv[k] += conv[k-1];

        /* Window scan with integer thresholds */
        int pruned = 0;
        for (int ell = 2; ell <= D_CHILD; ell++) {
            int n_cv = ell - 1;
            long long it = local_thresh[ell - 2];
            int n_windows = CONV_LEN - n_cv + 1;
            for (int s_lo = 0; s_lo < n_windows; s_lo++) {
                int s_hi = s_lo + n_cv - 1;
                long long ws = conv[s_hi];
                if (s_lo > 0) ws -= conv[s_lo - 1];
                if (ws > it) { pruned = 1; break; }
            }
            if (pruned) break;
        }

        if (pruned) {
            my_test++;
        } else {
            /* Survivor: compute FP64 test value for reporting */
            double best = 0.0;
            for (int ell = 2; ell <= D_CHILD; ell++) {
                int n_cv = ell - 1;
                double inv_norm = inv_norm_arr[ell - 2];
                int n_windows = CONV_LEN - n_cv + 1;
                for (int s_lo = 0; s_lo < n_windows; s_lo++) {
                    int s_hi = s_lo + n_cv - 1;
                    long long ws = conv[s_hi];
                    if (s_lo > 0) ws -= conv[s_lo - 1];
                    double tv = (double)ws * inv_norm;
                    if (tv > best) best = tv;
                }
            }
            my_surv++;
            if (survivor_buf != NULL) {
                /* Guard: skip atomicAdd once buffer is full to prevent
                 * int overflow when billions of configs survive */
                if (*survivor_count < max_survivors) {
                    int slot = atomicAdd(survivor_count, 1);
                    if (slot < max_survivors) {
                        for (int i = 0; i < D_CHILD; i++)
                            survivor_buf[slot * D_CHILD + i] = c[i];
                    }
                }
            }
            if (best < my_tv) {
                my_tv = best;
                for (int i = 0; i < D_CHILD; i++) my_cfg[i] = c[i];
            }
        }
    }

    /* === Per-block reduction: sum counts + min test value === */
    constexpr int BLOCK_SZ = (D_CHILD <= 12) ? REFINE_BLOCK_SIZE_12 :
                             (D_CHILD <= 24) ? REFINE_BLOCK_SIZE_24 :
                                                REFINE_BLOCK_SIZE_48;

    __shared__ int s_asym[BLOCK_SZ];
    __shared__ int s_test[BLOCK_SZ];
    __shared__ int s_surv[BLOCK_SZ];
    __shared__ double s_min_tv[BLOCK_SZ];
    __shared__ int s_cfgs[BLOCK_SZ * D_CHILD];

    int lane = threadIdx.x;
    s_asym[lane] = my_asym;
    s_test[lane] = my_test;
    s_surv[lane] = my_surv;
    s_min_tv[lane] = my_tv;
    for (int i = 0; i < D_CHILD; i++)
        s_cfgs[lane * D_CHILD + i] = my_cfg[i];
    __syncthreads();

    /* Cross-warp tree reduction */
    for (int s = BLOCK_SZ / 2; s > 16; s >>= 1) {
        if (lane < s) {
            s_asym[lane] += s_asym[lane + s];
            s_test[lane] += s_test[lane + s];
            s_surv[lane] += s_surv[lane + s];
            if (s_min_tv[lane + s] < s_min_tv[lane]) {
                s_min_tv[lane] = s_min_tv[lane + s];
                for (int i = 0; i < D_CHILD; i++)
                    s_cfgs[lane * D_CHILD + i] = s_cfgs[(lane + s) * D_CHILD + i];
            }
        }
        __syncthreads();
    }

    /* Intra-warp reduction (last 32 threads) */
    if (lane < 32) {
        int a = s_asym[lane], t = s_test[lane], sv = s_surv[lane];
        double val = s_min_tv[lane];

        for (int s = 16; s > 0; s >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, s);
            t += __shfl_down_sync(0xFFFFFFFF, t, s);
            sv += __shfl_down_sync(0xFFFFFFFF, sv, s);

            double other_val = shfl_down_double(0xFFFFFFFF, val, s);
            if (other_val < val) {
                val = other_val;
                int src_lane = lane + s;
                if (src_lane < 32) {
                    for (int i = 0; i < D_CHILD; i++)
                        s_cfgs[lane * D_CHILD + i] = s_cfgs[src_lane * D_CHILD + i];
                }
            }
            __syncwarp(0xFFFFFFFF);
        }

        if (lane == 0) {
            block_counts[blockIdx.x * 3 + 0] = (long long)a;
            block_counts[blockIdx.x * 3 + 1] = (long long)t;
            block_counts[blockIdx.x * 3 + 2] = (long long)sv;
            block_min_tv[blockIdx.x] = val;
            for (int i = 0; i < D_CHILD; i++)
                block_min_configs[blockIdx.x * D_CHILD + i] = s_cfgs[i];
        }
    }
}
