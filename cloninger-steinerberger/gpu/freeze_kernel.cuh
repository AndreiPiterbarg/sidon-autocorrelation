#pragma once
/*
 * Batched refinement kernel WITH per-thread freeze check.
 *
 * Splits D_PARENT bins into K_OUTER "outer" bins (enumerated across
 * threads via grid-stride loop) and K_INNER "inner" bins (freeze-
 * checked per thread, enumerated serially only if not frozen).
 *
 * For each outer config, the thread:
 *   1. Sets inner bins to flat-split values
 *   2. Computes full autoconvolution at outer+flat_inner
 *   3. Computes window scan to find min|gap| across all windows
 *   4. Computes per-inner-bin perturbation bounds
 *   5. If total inner perturbation < min|gap|: FROZEN — process only
 *      the flat-split inner config through the pruning cascade
 *   6. Otherwise: enumerate all inner configs serially
 *
 * The bin_order array maps logical positions to physical parent bin
 * indices. Outer bins = bin_order[0..K_OUTER-1], inner bins =
 * bin_order[K_OUTER..D_PARENT-1]. The host sorts bins so that inner
 * bins have the smallest B[i] values (most likely to freeze).
 *
 * Template parameters:
 *   D_CHILD: child dimension (2 * D_PARENT)
 *   K_OUTER: number of outer bins (enumerated across threads)
 *   USE_INT64: use int64 for convolution accumulators
 *
 * Depends on: device_helpers.cuh, refinement_kernel.cuh (block size defs)
 */

#include <type_traits>

template <int D_CHILD, int K_OUTER, bool USE_INT64 = true>
__global__ void __launch_bounds__(
    (D_CHILD <= 12) ? REFINE_BLOCK_SIZE_12 :
    (D_CHILD <= 24) ? REFINE_BLOCK_SIZE_24 : REFINE_BLOCK_SIZE_48, 2)
refine_prove_target_freeze(
    const int* __restrict__ all_parents,       /* [n_batch * D_PARENT] parent configs */
    const long long* __restrict__ batch_prefix, /* [n_batch + 1] prefix sums of OUTER work */
    const int* __restrict__ all_lo,            /* [n_batch * D_PARENT] per-parent lo values */
    const int* __restrict__ all_bin_order,      /* [n_batch * D_PARENT] bin ordering per parent */
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
    long long total_outer_work,
    /* Outputs */
    long long* __restrict__ block_counts,       /* [num_blocks * 3]: asym, test, surv */
    double*    __restrict__ block_min_tv,
    int*       __restrict__ block_min_configs,   /* [num_blocks * D_CHILD] */
    int*       __restrict__ survivor_buf,        /* [max_survivors * D_CHILD] or NULL */
    int*       __restrict__ survivor_count,      /* atomic counter or NULL */
    int        max_survivors,
    int        x_cap
) {
    constexpr int CONV_LEN = 2 * D_CHILD - 1;
    constexpr int D_PARENT = D_CHILD / 2;
    constexpr int K_INNER = D_PARENT - K_OUTER;
    using conv_t = typename std::conditional<USE_INT64, long long, int>::type;

    long long flat_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    int my_asym = 0, my_test = 0, my_surv = 0;
    double my_tv = 1e30;
    int my_cfg[D_CHILD];
    #pragma unroll
    for (int i = 0; i < D_CHILD; i++) my_cfg[i] = 0;

    float scale_f = 4.0f * n_half_child * inv_m_f;
    float total_mass_f = (float)S_child;
    float norm_ell_d = 4.0f * n_half_child * D_CHILD;
    float inv_ell2 = 1.0f / (4.0f * n_half_child * 2);

    double inv_S = 1.0 / (double)S_child;
    double margin = 1.0 / (4.0 * m);

    /* inv_norm_arr in shared memory: saves 46 regs/thread for D=24,
     * only accessed in the rare survivor path (<5% of threads). */
    __shared__ double s_inv_norm[2 * D_CHILD - 1];
    if (threadIdx.x == 0) {
        for (int ell = 2; ell <= 2 * D_CHILD; ell++)
            s_inv_norm[ell - 2] = (4.0 * n_half_child) / ((double)m * (double)m * (double)ell);
    }
    __syncthreads();

    double dyn_base = c_target * (double)m * (double)m + 1.0
                    + 1e-9 * (double)m * (double)m;
    double inv_4n = 1.0 / (4.0 * (double)n_half_child);

    for (long long outer_work_idx = flat_idx; outer_work_idx < total_outer_work;
         outer_work_idx += stride) {

        /* === Parent lookup via binary search on prefix sums === */
        int pidx = binary_search_le(batch_prefix, n_batch, outer_work_idx);
        long long outer_idx = outer_work_idx - batch_prefix[pidx];
        const int* pB = all_parents + pidx * D_PARENT;
        const int* pLo = all_lo + pidx * D_PARENT;
        const int* bin_order = all_bin_order + pidx * D_PARENT;

        /* === Decode outer bins from outer_idx === */
        int c[D_CHILD];
        {
            long long temp = outer_idx;
            for (int k = 0; k < K_OUTER; k++) {
                int bi = bin_order[k];  /* physical parent bin index */
                int lo = pLo[bi];
                int hi = (pB[bi] < x_cap) ? pB[bi] : x_cap;
                if (hi < lo) hi = lo;
                int s = hi - lo + 1;
                int c_even = lo + (int)(temp % (long long)s);
                temp /= (long long)s;
                c[2*bi]     = c_even;
                c[2*bi + 1] = pB[bi] - c_even;
            }
        }

        /* === Set inner bins to flat-split === */
        for (int k = K_OUTER; k < D_PARENT; k++) {
            int bi = bin_order[k];
            c[2*bi]     = pB[bi] / 2;
            c[2*bi + 1] = pB[bi] - pB[bi] / 2;
        }

        /* === Compute full autoconvolution of outer+flat_inner === */
        int prefix_c[D_CHILD + 1];
        prefix_c[0] = 0;
        for (int i = 0; i < D_CHILD; i++)
            prefix_c[i + 1] = prefix_c[i] + c[i];

        conv_t conv_flat[CONV_LEN];
        for (int kk = 0; kk < CONV_LEN; kk++) conv_flat[kk] = 0;
        for (int i = 0; i < D_CHILD; i++) {
            conv_flat[2*i] += (conv_t)c[i] * c[i];
            for (int j = i+1; j < D_CHILD; j++)
                conv_flat[i+j] += (conv_t)2 * c[i] * c[j];
        }

        /* Prefix sum of conv_flat for window scan */
        conv_t conv_ps[CONV_LEN];
        conv_ps[0] = conv_flat[0];
        for (int kk = 1; kk < CONV_LEN; kk++)
            conv_ps[kk] = conv_ps[kk-1] + conv_flat[kk];

        /* === Per-window P1 freeze check ===
         * Freeze if ANY window has gap > per-window perturbation.
         * If one window is guaranteed to prune all inner configs,
         * the parent is fully eliminated without inner enumeration. */
        int inner_delta_maxes[D_PARENT];
        for (int k = K_OUTER; k < D_PARENT; k++) {
            int bi = bin_order[k];
            int lo_bi = pLo[bi];
            int hi_bi = (pB[bi] < x_cap) ? pB[bi] : x_cap;
            if (hi_bi < lo_bi) hi_bi = lo_bi;
            int flat_val = pB[bi] / 2;
            int delta_max = flat_val - lo_bi;
            if (hi_bi - flat_val > delta_max) delta_max = hi_bi - flat_val;
            inner_delta_maxes[k - K_OUTER] = delta_max;
        }

        conv_t cross_pert = 0;
        for (int a = 0; a < K_INNER; a++) {
            if (inner_delta_maxes[a] <= 0) continue;
            for (int b = a + 1; b < K_INNER; b++) {
                if (inner_delta_maxes[b] <= 0) continue;
                cross_pert += (conv_t)4 * (conv_t)inner_delta_maxes[a]
                                         * (conv_t)inner_delta_maxes[b];
            }
        }

        int frozen = 0;
        for (int ell = 2; ell <= 2 * D_CHILD && !frozen; ell++) {
            int n_cv = ell - 1;
            double dyn_base_ell = dyn_base * (double)ell * inv_4n;
            double two_ell_inv_4n = 2.0 * (double)ell * inv_4n;
            int n_windows = CONV_LEN - n_cv + 1;
            for (int w = 0; w < n_windows && !frozen; w++) {
                /* Zigzag: check boundary positions first */
                int s_lo = (w & 1) ? (n_windows - 1 - w / 2) : (w / 2);
                int s_hi = s_lo + n_cv - 1;
                conv_t ws = conv_ps[s_hi];
                if (s_lo > 0) ws -= conv_ps[s_lo - 1];
                int lo_bin = (s_lo > D_CHILD - 1) ? s_lo - (D_CHILD - 1) : 0;
                int hi_bin = (s_lo + ell - 2 < D_CHILD - 1) ? s_lo + ell - 2 : D_CHILD - 1;
                int W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin];
                double dyn_x = dyn_base_ell + two_ell_inv_4n * (double)W_int;
                conv_t dyn_it = (conv_t)((long long)(dyn_x * (1.0 - 4.0 * DBL_EPSILON)));
                conv_t gap = ws - dyn_it;
                if (gap <= 0) continue;

                /* Window exceeds threshold — check if perturbation can close it */
                conv_t window_pert = 0;
                for (int k = K_OUTER; k < D_PARENT; k++) {
                    int dm = inner_delta_maxes[k - K_OUTER];
                    if (dm <= 0) continue;
                    int bi = bin_order[k];
                    int u_lo = s_lo - 2*bi;
                    int u_hi = s_lo + ell - 2 - 2*bi;
                    int v_lo = s_lo - 2*bi - 1;
                    int v_hi = s_lo + ell - 3 - 2*bi;
                    int u_lo_c = (u_lo < 0) ? 0 : u_lo;
                    int u_hi_c = (u_hi > D_CHILD - 1) ? D_CHILD - 1 : u_hi;
                    conv_t sum_u = (u_hi_c >= u_lo_c) ?
                        (conv_t)(prefix_c[u_hi_c + 1] - prefix_c[u_lo_c]) : 0;
                    int v_lo_c = (v_lo < 0) ? 0 : v_lo;
                    int v_hi_c = (v_hi > D_CHILD - 1) ? D_CHILD - 1 : v_hi;
                    conv_t sum_v = (v_hi_c >= v_lo_c) ?
                        (conv_t)(prefix_c[v_hi_c + 1] - prefix_c[v_lo_c]) : 0;
                    conv_t p1_abs = (conv_t)2 * ((sum_u >= sum_v) ?
                        (sum_u - sum_v) : (sum_v - sum_u));
                    window_pert += p1_abs * (conv_t)dm
                                 + (conv_t)2 * (conv_t)dm * (conv_t)dm;
                    window_pert += (conv_t)dm;
                }
                window_pert += cross_pert;
                if (gap > window_pert) frozen = 1;
            }
        }

        /* === Freeze decision === */
        int inner_work = 1;
        for (int k = K_OUTER; k < D_PARENT; k++) {
            int bi = bin_order[k];
            int lo_bi = pLo[bi];
            int hi_bi = (pB[bi] < x_cap) ? pB[bi] : x_cap;
            if (hi_bi < lo_bi) hi_bi = lo_bi;
            inner_work *= (hi_bi - lo_bi + 1);
        }
        int n_inner_configs = frozen ? 1 : inner_work;

        /* === Process inner configs === */
        for (int inner_idx = 0; inner_idx < n_inner_configs; inner_idx++) {

            if (!frozen) {
                /* Decode inner bins from inner_idx */
                int temp_inner = inner_idx;
                for (int k = K_OUTER; k < D_PARENT; k++) {
                    int bi = bin_order[k];
                    int lo_bi = pLo[bi];
                    int hi_bi = (pB[bi] < x_cap) ? pB[bi] : x_cap;
                    if (hi_bi < lo_bi) hi_bi = lo_bi;
                    int s = hi_bi - lo_bi + 1;
                    int c_even = lo_bi + (temp_inner % s);
                    temp_inner /= s;
                    c[2*bi]     = c_even;
                    c[2*bi + 1] = pB[bi] - c_even;
                }
            }
            /* else: inner bins already at flat-split values from above */

            /* === Full pruning cascade (same as standard batched kernel) === */

            /* 1. (Canonical filter removed — applied host-side after extraction) */

            /* 2+3. FP32 asymmetry + half-sum bound */
            {
                float left_sum = 0.0f;
                #pragma unroll
                for (int i = 0; i < D_CHILD / 2; i++) left_sum += (float)c[i];
                float left_frac = left_sum / total_mass_f;
                float dom = (left_frac > 0.5f) ? left_frac : (1.0f - left_frac);
                float asym_base = dom - margin_f;
                if (asym_base < 0.0f) asym_base = 0.0f;
                float asym_val = 2.0f * asym_base * asym_base;
                if (asym_val >= asym_limit_f) continue;
                if (left_sum * left_sum * scale_f * scale_f / norm_ell_d > thresh_f) continue;
                float right_sum = total_mass_f - left_sum;
                if (right_sum * right_sum * scale_f * scale_f / norm_ell_d > thresh_f) continue;
            }

            /* 4. FP32 ell=2 max element + two-max */
            {
                int max_c = c[0], max2_c = 0;
                #pragma unroll
                for (int i = 1; i < D_CHILD; i++) {
                    if (c[i] >= max_c) { max2_c = max_c; max_c = c[i]; }
                    else if (c[i] > max2_c) max2_c = c[i];
                }
                float max_a = max_c * scale_f;
                if (max_a * max_a * inv_ell2 > thresh_f) continue;
                if (2.0f * max_a * (max2_c * scale_f) * inv_ell2 > thresh_f) continue;
            }

            /* 4b. Block-sum bounds at ell=4 */
            {
                int block_pruned = 0;
                conv_t dyn_it4 = (conv_t)((long long)(
                    (dyn_base + 2.0 * (double)S_child) * 4.0 * inv_4n
                    * (1.0 - 4.0 * DBL_EPSILON)));
                #pragma unroll
                for (int i = 0; i < D_CHILD - 1; i++) {
                    conv_t bs = (conv_t)(c[i] + c[i+1]);
                    if (bs * bs > dyn_it4) { block_pruned = 1; break; }
                }
                if (block_pruned) { my_test++; continue; }
            }

            /* 5. FP64 asymmetry */
            double asym_val_d;
            {
                double left_sum_d = 0.0;
                #pragma unroll
                for (int i = 0; i < D_CHILD / 2; i++) left_sum_d += (double)c[i];
                double left_frac_d = left_sum_d * inv_S;
                double dom_d = (left_frac_d > 0.5) ? left_frac_d : (1.0 - left_frac_d);
                double asym_base_d = dom_d - margin;
                if (asym_base_d < 0.0) asym_base_d = 0.0;
                asym_val_d = 2.0 * asym_base_d * asym_base_d;
                if (asym_val_d >= c_target) { my_asym++; continue; }
            }

            /* 5b. Center convolution pre-check (ell=2) */
            {
                conv_t center = 0;
                #pragma unroll
                for (int i = 0; i < D_CHILD / 2; i++)
                    center += (conv_t)2 * c[i] * c[D_CHILD - 1 - i];
                conv_t dyn_it2_min = (conv_t)((long long)(
                    (dyn_base + 2.0 * (double)S_child) * 2.0 * inv_4n
                    * (1.0 - 4.0 * DBL_EPSILON)));
                if (center > dyn_it2_min) { my_test++; continue; }
            }

            /* 6. Autoconvolution with inline multi-ell early exit */
            int prefix_ci[D_CHILD + 1];
            prefix_ci[0] = 0;
            #pragma unroll
            for (int i = 0; i < D_CHILD; i++)
                prefix_ci[i + 1] = prefix_ci[i] + c[i];

            /* Conservative thresholds for multi-ell early abort (W_int=S) */
            conv_t dyn_it2 = (conv_t)((long long)(
                (dyn_base + 2.0 * (double)S_child) * 2.0 * inv_4n
                * (1.0 - 4.0 * DBL_EPSILON)));
            conv_t dyn_it3 = (conv_t)((long long)(
                (dyn_base + 2.0 * (double)S_child) * 3.0 * inv_4n
                * (1.0 - 4.0 * DBL_EPSILON)));
            conv_t dyn_it4_cons = (conv_t)((long long)(
                (dyn_base + 2.0 * (double)S_child) * 4.0 * inv_4n
                * (1.0 - 4.0 * DBL_EPSILON)));

            conv_t conv[CONV_LEN];
            #pragma unroll
            for (int kk = 0; kk < CONV_LEN; kk++) conv[kk] = 0;
            int early_exit = 0;
            for (int i = 0; i < D_CHILD; i++) {
                conv[2*i] += (conv_t)c[i] * c[i];
                #pragma unroll
                for (int j = i+1; j < D_CHILD; j++)
                    conv[i+j] += (conv_t)2 * c[i] * c[j];
                /* ell=2 checks on completed conv values */
                if (conv[2*i] > dyn_it2) { early_exit = 1; break; }
                if (i > 0 && conv[2*i - 1] > dyn_it2) { early_exit = 1; break; }
                /* ell=3 checks: window = sum of 2 consecutive raw conv values */
                if (i > 0) {
                    conv_t w3 = conv[2*i] + conv[2*i - 1];
                    if (w3 > dyn_it3) { early_exit = 1; break; }
                }
                if (i >= 2) {
                    conv_t w3b = conv[2*i - 1] + conv[2*i - 2];
                    if (w3b > dyn_it3) { early_exit = 1; break; }
                }
                /* ell=4 check: window = sum of 3 consecutive raw conv values */
                if (i >= 2) {
                    conv_t w4 = conv[2*i] + conv[2*i - 1] + conv[2*i - 2];
                    if (w4 > dyn_it4_cons) { early_exit = 1; break; }
                }
            }
            if (early_exit) { my_test++; continue; }

            /* Prefix sum */
            #pragma unroll
            for (int kk = 1; kk < CONV_LEN; kk++) conv[kk] += conv[kk-1];

            /* Window scan with per-position dynamic thresholds (zigzag order) */
            int pruned = 0;
            for (int ell = 2; ell <= 2 * D_CHILD; ell++) {
                int n_cv = ell - 1;
                double dyn_base_ell2 = dyn_base * (double)ell * inv_4n;
                double two_ell_inv_4n2 = 2.0 * (double)ell * inv_4n;
                int n_windows = CONV_LEN - n_cv + 1;
                for (int w = 0; w < n_windows; w++) {
                    /* Zigzag: check boundary positions first */
                    int s_lo = (w & 1) ? (n_windows - 1 - w / 2) : (w / 2);
                    int s_hi = s_lo + n_cv - 1;
                    conv_t ws = conv[s_hi];
                    if (s_lo > 0) ws -= conv[s_lo - 1];
                    int lo_bin = (s_lo > D_CHILD - 1) ? s_lo - (D_CHILD - 1) : 0;
                    int hi_bin = (s_lo + ell - 2 < D_CHILD - 1) ? s_lo + ell - 2 : D_CHILD - 1;
                    int W_int = prefix_ci[hi_bin + 1] - prefix_ci[lo_bin];
                    double dyn_x = dyn_base_ell2 + two_ell_inv_4n2 * (double)W_int;
                    conv_t dyn_it = (conv_t)((long long)(dyn_x * (1.0 - 4.0 * DBL_EPSILON)));
                    if (ws > dyn_it) { pruned = 1; break; }
                }
                if (pruned) break;
            }

            if (pruned) {
                my_test++;
            } else {
                /* Survivor */
                double best = 0.0;
                for (int ell = 2; ell <= 2 * D_CHILD; ell++) {
                    int n_cv = ell - 1;
                    double inv_norm = s_inv_norm[ell - 2];
                    int n_windows = CONV_LEN - n_cv + 1;
                    for (int s_lo = 0; s_lo < n_windows; s_lo++) {
                        int s_hi = s_lo + n_cv - 1;
                        conv_t ws = conv[s_hi];
                        if (s_lo > 0) ws -= conv[s_lo - 1];
                        double tv = (double)ws * inv_norm;
                        if (tv > best) best = tv;
                    }
                }
                my_surv++;
                if (survivor_buf != NULL) {
                    if (*survivor_count < max_survivors) {
                        int slot = atomicAdd(survivor_count, 1);
                        if (slot < max_survivors) {
                            long long base2 = (long long)slot * D_CHILD;
                            #pragma unroll
                            for (int i = 0; i < D_CHILD; i++)
                                survivor_buf[base2 + i] = c[i];
                        }
                    }
                }
                if (best < my_tv) {
                    my_tv = best;
                    #pragma unroll
                    for (int i = 0; i < D_CHILD; i++) my_cfg[i] = c[i];
                }
            }
        } /* end inner loop */
    } /* end outer grid-stride loop */

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
    #pragma unroll
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
                #pragma unroll
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

        #pragma unroll
        for (int s = 16; s > 0; s >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, s);
            t += __shfl_down_sync(0xFFFFFFFF, t, s);
            sv += __shfl_down_sync(0xFFFFFFFF, sv, s);

            double other_val = shfl_down_double(0xFFFFFFFF, val, s);
            if (other_val < val) {
                val = other_val;
                int src_lane = lane + s;
                if (src_lane < 32) {
                    #pragma unroll
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
            #pragma unroll
            for (int i = 0; i < D_CHILD; i++)
                block_min_configs[blockIdx.x * D_CHILD + i] = s_cfgs[i];
        }
    }
}
