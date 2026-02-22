#pragma once
#include <type_traits>
/*
 * Fused single-pass GPU kernels: composition generation + pruning + autoconvolution.
 *
 * Eliminates all DRAM intermediate traffic by fusing Phase 1 and Phase 2
 * into a single kernel. Each thread generates compositions from a flat index,
 * applies cheap FP32 pruning, then computes INT64 autoconvolution in registers.
 *
 * Contents:
 *   - map_triangle:          flat index -> (c1, c2) in triangle (D=4 helper)
 *   - map_tetrahedron:       flat index -> (c1, c2, c3) in tetrahedron (D=6 helper)
 *   - fused_find_min<D>:     Find minimum effective value across all compositions
 *   - fused_prove_target<D>: Prove target bound by checking all compositions
 *
 * Depends on: device_helpers.cuh (binary_search_le, shfl_down_double, FUSED_BLOCK_SIZE)
 */


/* ================================================================
 * Flat-index-to-triangle mapping (used by D=4)
 *
 * Maps local_idx in [0, (R+1)(R+2)/2) to (c1, c2) where:
 *   c1 >= 0, c2 >= 0, c1 + c2 <= R
 * Row c1 has (R - c1 + 1) entries. Cumulative: c1*(2R - c1 + 3)/2
 * ================================================================ */
__device__ __forceinline__ void map_triangle(
    long long local_idx, int R,
    int* c1_out, int* c2_out, int* valid
) {
    float Rf = (float)(2 * R + 3);
    float disc = Rf * Rf - 8.0f * (float)local_idx;
    if (disc < 0.0f) disc = 0.0f;
    int c1 = (int)((Rf - sqrtf(disc)) * 0.5f);

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

    if (c1 < 0 || c1 > R || c2 < 0 || c2 > R - c1) {
        *valid = 0;
        return;
    }
    *c1_out = c1;
    *c2_out = c2;
    *valid = 1;
}


/* ================================================================
 * Flat-index-to-tetrahedron mapping (used by D=6)
 *
 * Maps local_idx in [0, (R+1)(R+2)(R+3)/6) to (c1, c2, c3) where:
 *   c1, c2, c3 >= 0, c1 + c2 + c3 <= R
 * Layer c1 has triangle count (R-c1+1)(R-c1+2)/2.
 * Cumulative: (R+1)(R+2)(R+3)/6 - (R-c1+1)(R-c1+2)(R-c1+3)/6
 * ================================================================ */
__device__ __forceinline__ void map_tetrahedron(
    long long local_idx, int R,
    int* c1_out, int* c2_out, int* c3_out, int* valid
) {
    long long R_ll = (long long)R;
    long long total_tet = (R_ll + 1) * (R_ll + 2) * (R_ll + 3) / 6;
    long long remaining = total_tet - local_idx;

    /* Cubic root inversion for c1 */
    float P_approx = cbrtf(6.0f * (float)remaining);
    int c1 = (int)(R_ll + 1) - (int)(P_approx + 1.5f);
    if (c1 < 0) c1 = 0;

    while (c1 > 0) {
        long long P = R_ll - (long long)c1 + 1;
        long long cum = total_tet - P * (P + 1) * (P + 2) / 6;
        if (cum <= local_idx) break;
        c1--;
    }
    while (c1 <= R) {
        long long P = R_ll - (long long)c1;
        long long cum_next = total_tet - P * (P + 1) * (P + 2) / 6;
        if (cum_next > local_idx) break;
        c1++;
    }
    if (c1 < 0 || c1 > R) { *valid = 0; return; }

    long long P = R_ll - (long long)c1 + 1;
    long long cum_c1 = total_tet - P * (P + 1) * (P + 2) / 6;
    long long local_idx_2 = local_idx - cum_c1;

    /* Within layer c1: triangle of (c2, c3) with c2 + c3 <= R - c1 */
    int R2 = R - c1;
    int c2, c3;
    int tri_valid;
    map_triangle(local_idx_2, R2, &c2, &c3, &tri_valid);
    if (!tri_valid) { *valid = 0; return; }

    *c1_out = c1;
    *c2_out = c2;
    *c3_out = c3;
    *valid = 1;
}


/* ================================================================
 * Fused find_min kernel (templated on D)
 *
 * One thread per (c0,c1,c2) triple [D=4] or (c0,c1,c2,c3) quad [D=6].
 * Grid-stride loop over flat indices. Per-block tree reduction.
 * Zero intermediate DRAM traffic.
 * ================================================================ */
template <int D, bool USE_INT64 = true>
__global__ void __launch_bounds__(FUSED_BLOCK_SIZE)
fused_find_min(
    int S,
    int n_half,
    int m,
    double corr,
    double margin,
    double init_min_eff,
    float inv_m_f,
    float thresh_f,
    float margin_f,
    float asym_limit_f,
    const long long* __restrict__ prefix_sums,
    const int* __restrict__ c0_order,
    int n_c0,
    long long work_start,
    long long work_end,
    double* __restrict__ block_min_vals,
    int* __restrict__ block_min_configs
) {
    constexpr int CONV_LEN = 2 * D - 1;
    using conv_t = typename std::conditional<USE_INT64, long long, int>::type;

    long long flat_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    double my_eff = init_min_eff;
    int my_cfg[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) my_cfg[i] = 0;

    /* FP32 scale factor: converts integer c to a-coordinates.
     * With S=m convention: a_j = (4n/m) * c_j, so scale_f = 4n/m.
     * norm_ell_* are denominators in a-space: test_val = a^2 / (4n*ell). */
    float scale_f = 4.0f * n_half * inv_m_f;   /* c -> a conversion */
    float norm_ell_d = 4.0f * n_half * D;
    float norm_ell_4 = 4.0f * n_half * 4;
    float inv_ell2 = 1.0f / (4.0f * n_half * 2);
    float total_mass_f = (float)S;

    /* Hoisted FP64 invariants.
     * With S=m: test_val = 4n/(m^2*ell) * sum(c_i*c_j), so
     * inv_norm = 4n / (m^2 * ell). */
    double inv_S = 1.0 / (double)S;
    double inv_norm_arr[D - 1];
    #pragma unroll
    for (int ell = 2; ell <= D; ell++)
        inv_norm_arr[ell - 2] = (4.0 * n_half) / ((double)m * (double)m * (double)ell);

    /* Early-exit cutoff: once best >= cutoff, eff >= my_eff so skip */
    double cutoff = my_eff + corr;

    for (long long work_idx = work_start + flat_idx; work_idx < work_end; work_idx += stride) {
        /* Decode c0 from flat index */
        int c0_idx = binary_search_le(prefix_sums, n_c0, work_idx);
        int c0 = c0_order[c0_idx];
        long long local_idx = work_idx - prefix_sums[c0_idx];
        int R = S - 2 * c0;
        if (R < 0) continue;

        if (D == 4) {
            /* D=4: map to (c1, c2), c3 = remainder */
            int c1, c2, tri_valid;
            map_triangle(local_idx, R, &c1, &c2, &tri_valid);
            if (!tri_valid) continue;
            int c3 = S - c0 - c1 - c2;

            /* 1. Canonical */
            if (c3 < c0) continue;
            if (c0 == c3 && c1 > c2) continue;

            /* 2. FP32 asymmetry */
            float pair_left = (float)(c0 + c1);
            float left_frac = pair_left / total_mass_f;
            float dom = (left_frac > 0.5f) ? left_frac : (1.0f - left_frac);
            float asym_base = dom - margin_f;
            if (asym_base < 0.0f) asym_base = 0.0f;
            float asym_val = 2.0f * asym_base * asym_base;
            if (asym_val >= asym_limit_f) continue;

            /* 3. Pair-sum bound (ell=D) */
            if (pair_left * pair_left * scale_f * scale_f / norm_ell_d > thresh_f) continue;
            float pair_right = (float)(c2 + c3);
            if (pair_right * pair_right * scale_f * scale_f / norm_ell_d > thresh_f) continue;

            /* 3b. Block-sum at ell=4 (interior pair {c1,c2}) */
            {
                float s_f = (float)(c1 + c2);
                if (s_f * s_f * scale_f * scale_f / norm_ell_4 > thresh_f) continue;
            }

            /* 4. ell=2 max element + two-max enhanced */
            int max_c = c3, max2_c = c0;
            if (c1 >= max_c) { max2_c = max_c; max_c = c1; } else if (c1 > max2_c) max2_c = c1;
            if (c2 >= max_c) { max2_c = max_c; max_c = c2; } else if (c2 > max2_c) max2_c = c2;
            float max_a = max_c * scale_f;
            if (max_a * max_a * inv_ell2 > thresh_f) continue;
            if (2.0f * max_a * (max2_c * scale_f) * inv_ell2 > thresh_f) continue;

            /* === Autoconvolution (INT64 or INT32 based on template) === */
            int c[4] = {c0, c1, c2, c3};
            conv_t conv[CONV_LEN];
            #pragma unroll
            for (int k = 0; k < CONV_LEN; k++) conv[k] = 0;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                conv[2*i] += (conv_t)c[i] * c[i];
                #pragma unroll
                for (int j = i+1; j < 4; j++)
                    conv[i+j] += (conv_t)2 * c[i] * c[j];
            }
            #pragma unroll
            for (int k = 1; k < CONV_LEN; k++) conv[k] += conv[k-1];

            /* Window max (FP64) with early exit */
            double best = 0.0;
            #pragma unroll
            for (int ell = 2; ell <= 4; ell++) {
                int n_cv = ell - 1;
                double inv_norm = inv_norm_arr[ell - 2];
                int n_windows = CONV_LEN - n_cv + 1;
                for (int s_lo = 0; s_lo < n_windows; s_lo++) {
                    int s_hi = s_lo + n_cv - 1;
                    conv_t ws = conv[s_hi];
                    if (s_lo > 0) ws -= conv[s_lo - 1];
                    double tv = (double)ws * inv_norm;
                    if (tv > best) best = tv;
                }
                if (best >= cutoff) break;
            }

            if (best >= cutoff) continue;

            double eff = best - corr;

            /* FP64 asymmetry bound */
            double left_sum_d = (double)c0 + (double)c1;
            double left_frac_d = left_sum_d * inv_S;
            double dom_d = (left_frac_d > 0.5) ? left_frac_d : (1.0 - left_frac_d);
            double asym_base_d = dom_d - margin;
            if (asym_base_d < 0.0) asym_base_d = 0.0;
            double asym_val_d = 2.0 * asym_base_d * asym_base_d;
            if (asym_val_d > eff) eff = asym_val_d;

            if (eff < my_eff) {
                my_eff = eff;
                my_cfg[0] = c0; my_cfg[1] = c1;
                my_cfg[2] = c2; my_cfg[3] = c3;
                cutoff = my_eff + corr;
            }
        }

        if (D == 6) {
            /* D=6: map to (c1, c2, c3), inner c4 loop */
            int c1, c2, c3, tet_valid;
            map_tetrahedron(local_idx, R, &c1, &c2, &c3, &tet_valid);
            if (!tet_valid) continue;

            int r3 = S - c0 - c1 - c2 - c3;

            /* FP32 pre-checks at (c0,c1,c2,c3) level */
            float left3 = (float)(c0 + c1 + c2);
            float right3 = (float)S - left3;
            if (left3 * left3 * scale_f * scale_f / norm_ell_d > thresh_f) continue;
            if (right3 * right3 * scale_f * scale_f / norm_ell_d > thresh_f) continue;

            float left_frac = left3 / total_mass_f;
            float dom = (left_frac > 0.5f) ? left_frac : (1.0f - left_frac);
            float asym_base = dom - margin_f;
            if (asym_base < 0.0f) asym_base = 0.0f;
            float asym_val = 2.0f * asym_base * asym_base;
            if (asym_val >= asym_limit_f) continue;

            /* Block-sum bounds at ell=4 (adjacent pairs, outer loop) */
            {
                float s_f;
                s_f = (float)(c0 + c1);
                if (s_f * s_f * scale_f * scale_f / norm_ell_4 > thresh_f) continue;
                s_f = (float)(c1 + c2);
                if (s_f * s_f * scale_f * scale_f / norm_ell_4 > thresh_f) continue;
                s_f = (float)(c2 + c3);
                if (s_f * s_f * scale_f * scale_f / norm_ell_4 > thresh_f) continue;
            }
            /* Block-sum at ell=6 (interior triple {c1,c2,c3}) */
            {
                float s_f = (float)(c1 + c2 + c3);
                if (s_f * s_f * scale_f * scale_f / norm_ell_d > thresh_f) continue;
            }

            /* FP64 asymmetry (constant across c4 loop) */
            double left_sum_d = (double)c0 + (double)c1 + (double)c2;
            double left_frac_d = left_sum_d * inv_S;
            double dom_d = (left_frac_d > 0.5) ? left_frac_d : (1.0 - left_frac_d);
            double asym_base_d = dom_d - margin;
            if (asym_base_d < 0.0) asym_base_d = 0.0;
            double asym_val_d = 2.0 * asym_base_d * asym_base_d;

            /* Inner c4 loop */
            int c4_max = r3 - c0;
            if (c4_max < 0) continue;

            /* === INCREMENTAL CONVOLUTION: precompute base from (c0,c1,c2,c3) ===
             * conv[0..3] depends only on (c0,c1,c2,c3) — constant across c4 loop.
             * conv[4..10] updated incrementally each iteration.
             * Reduces inner-loop work from 21 mults to 11 mults per iteration.
             */
            conv_t conv[11];
            conv[0] = (conv_t)c0 * c0;
            conv[1] = conv[0] + (conv_t)2 * c0 * c1;
            conv[2] = conv[1] + (conv_t)c1 * c1 + (conv_t)2 * c0 * c2;
            conv[3] = conv[2] + (conv_t)2 * ((conv_t)c0 * c3 + (conv_t)c1 * c2);
            conv_t br4 = (conv_t)c2 * c2 + (conv_t)2 * c1 * c3;
            conv_t br5 = (conv_t)2 * c2 * c3;
            conv_t br6 = (conv_t)c3 * c3;
            conv_t two_c0 = (conv_t)2 * c0;
            conv_t two_c1 = (conv_t)2 * c1;
            conv_t two_c2 = (conv_t)2 * c2;
            conv_t two_c3 = (conv_t)2 * c3;

            for (int c4 = 0; c4 <= c4_max; c4++) {
                int c5 = r3 - c4;

                /* Canonical palindrome */
                if (c0 == c5) {
                    if (c1 > c4) continue;
                    if (c1 == c4 && c2 > c3) continue;
                }

                /* ell=2 max element + two-max enhanced */
                int max_c = c5, max2_c = 0;
                if (c1 >= max_c) { max2_c = max_c; max_c = c1; } else if (c1 > max2_c) max2_c = c1;
                if (c2 >= max_c) { max2_c = max_c; max_c = c2; } else if (c2 > max2_c) max2_c = c2;
                if (c3 >= max_c) { max2_c = max_c; max_c = c3; } else if (c3 > max2_c) max2_c = c3;
                if (c4 >= max_c) { max2_c = max_c; max_c = c4; } else if (c4 > max2_c) max2_c = c4;
                if (c0 > max2_c) max2_c = c0;
                float max_a = max_c * scale_f;
                if (max_a * max_a * inv_ell2 > thresh_f) continue;
                /* Two-max cross-term at ell=2 */
                if (2.0f * max_a * (max2_c * scale_f) * inv_ell2 > thresh_f) continue;

                /* Block-sum bounds at ell=4 (pairs involving c4/c5) */
                {
                    float s_f;
                    s_f = (float)(c3 + c4);
                    if (s_f * s_f * scale_f * scale_f / norm_ell_4 > thresh_f) continue;
                    s_f = (float)(c4 + c5);
                    if (s_f * s_f * scale_f * scale_f / norm_ell_4 > thresh_f) continue;
                }
                /* Block-sum at ell=6 (interior triple {c2,c3,c4}) */
                {
                    float s_f = (float)(c2 + c3 + c4);
                    if (s_f * s_f * scale_f * scale_f / norm_ell_d > thresh_f) continue;
                }
                /* Central conv bound: conv[5] = 2*(c0*c5 + c1*c4 + c2*c3) at ell=2 */
                {
                    float cc = 2.0f * ((float)c0 * (float)c5 + (float)c1 * (float)c4 + (float)c2 * (float)c3);
                    if (cc * scale_f * scale_f * inv_ell2 > thresh_f) continue;
                }

                /* Incremental autoconvolution (prefix-summed):
                 * conv[0..3] already set (constant base).
                 * conv[4..10] = base + cross-terms from (c4, c5). */
                conv[4] = conv[3] + br4 + two_c0 * (conv_t)c4;
                conv[5] = conv[4] + br5 + two_c1 * (conv_t)c4 + two_c0 * (conv_t)c5;
                conv[6] = conv[5] + br6 + two_c2 * (conv_t)c4 + two_c1 * (conv_t)c5;
                conv[7] = conv[6] + two_c3 * (conv_t)c4 + two_c2 * (conv_t)c5;
                conv[8] = conv[7] + (conv_t)c4 * (conv_t)c4 + two_c3 * (conv_t)c5;
                conv[9] = conv[8] + (conv_t)2 * (conv_t)c4 * (conv_t)c5;
                conv[10] = conv[9] + (conv_t)c5 * (conv_t)c5;

                /* Window max (FP64) with early exit */
                double best = 0.0;
                int d6_skip = 0;
                #pragma unroll
                for (int ell = 2; ell <= 6; ell++) {
                    int n_cv = ell - 1;
                    double inv_norm = inv_norm_arr[ell - 2];
                    int n_win = 11 - n_cv + 1;
                    for (int s_lo = 0; s_lo < n_win; s_lo++) {
                        int s_hi = s_lo + n_cv - 1;
                        conv_t ws = conv[s_hi];
                        if (s_lo > 0) ws -= conv[s_lo - 1];
                        double tv = (double)ws * inv_norm;
                        if (tv > best) best = tv;
                    }
                    if (best >= cutoff) { d6_skip = 1; break; }
                }

                if (d6_skip) continue;

                double eff = best - corr;
                if (asym_val_d > eff) eff = asym_val_d;

                if (eff < my_eff) {
                    my_eff = eff;
                    my_cfg[0] = c0; my_cfg[1] = c1;
                    my_cfg[2] = c2; my_cfg[3] = c3;
                    my_cfg[4] = c4; my_cfg[5] = c5;
                    cutoff = my_eff + corr;
                }
            }
        }
    }

    /* === Per-block tree reduction for minimum === */
    __shared__ double s_vals[FUSED_BLOCK_SIZE];
    __shared__ int s_cfgs[FUSED_BLOCK_SIZE * D];

    int lane = threadIdx.x;
    s_vals[lane] = my_eff;
    #pragma unroll
    for (int i = 0; i < D; i++) s_cfgs[lane * D + i] = my_cfg[i];
    __syncthreads();

    for (int s = FUSED_BLOCK_SIZE / 2; s > 16; s >>= 1) {
        if (lane < s && s_vals[lane + s] < s_vals[lane]) {
            s_vals[lane] = s_vals[lane + s];
            #pragma unroll
            for (int i = 0; i < D; i++)
                s_cfgs[lane * D + i] = s_cfgs[(lane + s) * D + i];
        }
        __syncthreads();
    }

    if (lane < 32) {
        double val = s_vals[lane];
        int cfg[D];
        #pragma unroll
        for (int i = 0; i < D; i++) cfg[i] = s_cfgs[lane * D + i];

        #pragma unroll
        for (int s = 16; s > 0; s >>= 1) {
            double other_val = shfl_down_double(0xFFFFFFFF, val, s);
            int other_cfg[D];
            #pragma unroll
            for (int i = 0; i < D; i++)
                other_cfg[i] = __shfl_down_sync(0xFFFFFFFF, cfg[i], s);
            if (other_val < val) {
                val = other_val;
                #pragma unroll
                for (int i = 0; i < D; i++) cfg[i] = other_cfg[i];
            }
        }

        if (lane == 0) {
            s_vals[0] = val;
            #pragma unroll
            for (int i = 0; i < D; i++) s_cfgs[i] = cfg[i];
        }
    }

    if (lane == 0) {
        block_min_vals[blockIdx.x] = s_vals[0];
        #pragma unroll
        for (int i = 0; i < D; i++)
            block_min_configs[blockIdx.x * D + i] = s_cfgs[i];
    }
}


/* ================================================================
 * Fused prove_target kernel (templated on D)
 *
 * Same composition generation + pruning as fused_find_min.
 * Uses integer thresholds for zero-FP64 inner loop.
 * Per-block output: counts (asym, test, surv) + min test value.
 * ================================================================ */
template <int D, bool USE_INT64 = true>
__global__ void __launch_bounds__(FUSED_BLOCK_SIZE)
fused_prove_target(
    int S,
    int n_half,
    int m,
    double margin,
    double c_target,
    double thresh,
    float inv_m_f,
    float thresh_f,
    float margin_f,
    float asym_limit_f,
    const long long* __restrict__ prefix_sums,
    const int* __restrict__ c0_order,
    int n_c0,
    long long work_start,
    long long work_end,
    long long* __restrict__ block_counts, /* [num_blocks * 4] */
    double* __restrict__ block_min_tv,
    int* __restrict__ block_min_configs,
    /* Survivor extraction (pass NULL + NULL + 0 to disable) */
    int*  __restrict__ survivor_buf,     /* [max_survivors * D] or NULL */
    int*  __restrict__ survivor_count,   /* atomic counter or NULL */
    int   max_survivors                  /* 0 = disabled */
) {
    constexpr int CONV_LEN = 2 * D - 1;
    using conv_t = typename std::conditional<USE_INT64, long long, int>::type;

    long long flat_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    int my_fp32 = 0, my_asym = 0, my_test = 0, my_surv = 0;
    double my_tv = 1e30;
    int my_cfg[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) my_cfg[i] = 0;

    /* FP32 scale factor: converts integer c to a-coordinates.
     * With S=m convention: a_j = (4n/m) * c_j, so scale_f = 4n/m. */
    float scale_f = 4.0f * n_half * inv_m_f;
    float norm_ell_d = 4.0f * n_half * D;
    float norm_ell_4 = 4.0f * n_half * 4;
    float inv_ell2 = 1.0f / (4.0f * n_half * 2);
    float total_mass_f = (float)S;

    /* Hoisted FP64 invariants.
     * With S=m: test_val = 4n/(m^2*ell) * sum(c_i*c_j). */
    double inv_S = 1.0 / (double)S;
    double inv_norm_arr[D - 1];
    #pragma unroll
    for (int ell = 2; ell <= D; ell++)
        inv_norm_arr[ell - 2] = (4.0 * n_half) / ((double)m * (double)m * (double)ell);

    /* Compute fixed integer thresholds per-thread (conservative, for pre-checks).
     * The main window scan uses tighter per-window-size dynamic thresholds,
     * but the cheap integer pre-checks still use the conservative fixed threshold. */
    conv_t local_thresh[D - 1];
    #pragma unroll
    for (int i = 0; i < D - 1; i++) {
        double x = thresh * (double)m * (double)m * (double)(i + 2) / (4.0 * (double)n_half);
        local_thresh[i] = (conv_t)((long long)(x * (1.0 - 4.0 * DBL_EPSILON)));
    }
    /* Dynamic correction constants (Lemma 2, per-window-size).
     * MATLAB: boundToBeat = c_target + delta^2 + 2*delta*W_j
     * In integer space: dyn_it(ell) = (dyn_base + 2*W_int) * ell / (4*n_half)
     * where W_int = sum of c[ell/2 .. ell-1] (contributing bins per Lemma 2).
     * dyn_base includes c_target*m^2 + 1 + fp_margin*m^2. */
    double dyn_base = c_target * (double)m * (double)m + 1.0
                    + 1e-9 * (double)m * (double)m;
    double inv_4n = 1.0 / (4.0 * (double)n_half);

    for (long long work_idx = work_start + flat_idx; work_idx < work_end; work_idx += stride) {
        int c0_idx = binary_search_le(prefix_sums, n_c0, work_idx);
        int c0 = c0_order[c0_idx];
        long long local_idx = work_idx - prefix_sums[c0_idx];
        int R = S - 2 * c0;
        if (R < 0) continue;

        if (D == 4) {
            int c1, c2, tri_valid;
            map_triangle(local_idx, R, &c1, &c2, &tri_valid);
            if (!tri_valid) continue;
            int c3 = S - c0 - c1 - c2;

            /* 1. Canonical */
            if (c3 < c0) continue;
            if (c0 == c3 && c1 > c2) continue;

            /* 2. FP32 asymmetry */
            float pair_left = (float)(c0 + c1);
            float left_frac = pair_left / total_mass_f;
            float dom = (left_frac > 0.5f) ? left_frac : (1.0f - left_frac);
            float asym_base = dom - margin_f;
            if (asym_base < 0.0f) asym_base = 0.0f;
            float asym_val = 2.0f * asym_base * asym_base;
            if (asym_val >= asym_limit_f) { my_fp32++; continue; }

            /* 3. Pair-sum bound */
            if (pair_left * pair_left * scale_f * scale_f / norm_ell_d > thresh_f) { my_fp32++; continue; }
            float pair_right = (float)(c2 + c3);
            if (pair_right * pair_right * scale_f * scale_f / norm_ell_d > thresh_f) { my_fp32++; continue; }

            /* 3b. Block-sum at ell=4 (interior pair {c1,c2}, integer) */
            {
                conv_t bs = (conv_t)(c1 + c2);
                if (bs * bs > local_thresh[2]) { my_fp32++; continue; }
            }

            /* 4. ell=2 max element + two-max enhanced */
            int max_c = c3, max2_c = c0;
            if (c1 >= max_c) { max2_c = max_c; max_c = c1; } else if (c1 > max2_c) max2_c = c1;
            if (c2 >= max_c) { max2_c = max_c; max_c = c2; } else if (c2 > max2_c) max2_c = c2;
            float max_a = max_c * scale_f;
            if (max_a * max_a * inv_ell2 > thresh_f) { my_fp32++; continue; }
            /* Two-max cross-term at ell=2 (integer) */
            if ((conv_t)2 * max_c * max2_c > local_thresh[0]) { my_fp32++; continue; }

            /* 5. FP64 asymmetry -> asym_pruned */
            double left_sum_d = (double)c0 + (double)c1;
            double left_frac_d = left_sum_d * inv_S;
            double dom_d = (left_frac_d > 0.5) ? left_frac_d : (1.0 - left_frac_d);
            double asym_base_d = dom_d - margin;
            if (asym_base_d < 0.0) asym_base_d = 0.0;
            double asym_val_d = 2.0 * asym_base_d * asym_base_d;
            if (asym_val_d >= c_target) { my_asym++; continue; }

            /* 6. Autoconvolution + integer threshold */
            int c[4] = {c0, c1, c2, c3};
            conv_t conv[CONV_LEN];
            #pragma unroll
            for (int k = 0; k < CONV_LEN; k++) conv[k] = 0;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                conv[2*i] += (conv_t)c[i] * c[i];
                #pragma unroll
                for (int j = i+1; j < 4; j++)
                    conv[i+j] += (conv_t)2 * c[i] * c[j];
            }
            #pragma unroll
            for (int k = 1; k < CONV_LEN; k++) conv[k] += conv[k-1];

            /* Prefix sums of c[] for per-position dynamic threshold */
            int prefix_c[5];
            prefix_c[0] = 0;
            prefix_c[1] = c0;
            prefix_c[2] = c0 + c1;
            prefix_c[3] = c0 + c1 + c2;
            prefix_c[4] = S;

            int pruned = 0;
            #pragma unroll
            for (int ell = 2; ell <= 4; ell++) {
                int n_cv = ell - 1;
                /* Precompute per-ell constants for dynamic threshold */
                double dyn_base_ell = dyn_base * (double)ell * inv_4n;
                double two_ell_inv_4n = 2.0 * (double)ell * inv_4n;
                int n_windows = CONV_LEN - n_cv + 1;
                for (int s_lo = 0; s_lo < n_windows; s_lo++) {
                    int s_hi = s_lo + n_cv - 1;
                    conv_t ws = conv[s_hi];
                    if (s_lo > 0) ws -= conv[s_lo - 1];
                    /* Per-window-POSITION dynamic threshold (Lemma 2):
                     * W_int = mass of bins contributing to this window position.
                     * Bin i contributes iff max(0,s_lo-3) <= i <= min(3,s_lo+ell-2). */
                    int lo_bin = (s_lo > 3) ? s_lo - 3 : 0;
                    int hi_bin = (s_lo + ell - 2 < 3) ? s_lo + ell - 2 : 3;
                    int W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin];
                    double dyn_x = dyn_base_ell + two_ell_inv_4n * (double)W_int;
                    conv_t dyn_it = (conv_t)((long long)(dyn_x
                                 * (1.0 - 4.0 * DBL_EPSILON)));
                    if (ws > dyn_it) { pruned = 1; break; }
                }
                if (pruned) break;
            }

            if (pruned) {
                my_test++;
            } else {
                /* Survivor: compute FP64 test value for reporting */
                double best = 0.0;
                #pragma unroll
                for (int ell = 2; ell <= 4; ell++) {
                    int n_cv = ell - 1;
                    double inv_norm = inv_norm_arr[ell - 2];
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
                    int slot = atomicAdd(survivor_count, 1);
                    if (slot < max_survivors) {
                        long long off = (long long)slot * 4;
                        survivor_buf[off + 0] = c0;
                        survivor_buf[off + 1] = c1;
                        survivor_buf[off + 2] = c2;
                        survivor_buf[off + 3] = c3;
                    }
                }
                if (best < my_tv) {
                    my_tv = best;
                    my_cfg[0] = c0; my_cfg[1] = c1;
                    my_cfg[2] = c2; my_cfg[3] = c3;
                }
            }
        }

        if (D == 6) {
            int c1, c2, c3, tet_valid;
            map_tetrahedron(local_idx, R, &c1, &c2, &c3, &tet_valid);
            if (!tet_valid) continue;

            int r3 = S - c0 - c1 - c2 - c3;

            /* FP32 pre-checks */
            float left3 = (float)(c0 + c1 + c2);
            float right3 = (float)S - left3;
            if (left3 * left3 * scale_f * scale_f / norm_ell_d > thresh_f) { my_fp32++; continue; }
            if (right3 * right3 * scale_f * scale_f / norm_ell_d > thresh_f) { my_fp32++; continue; }

            float left_frac = left3 / total_mass_f;
            float dom = (left_frac > 0.5f) ? left_frac : (1.0f - left_frac);
            float asym_base = dom - margin_f;
            if (asym_base < 0.0f) asym_base = 0.0f;
            float asym_val = 2.0f * asym_base * asym_base;
            if (asym_val >= asym_limit_f) { my_fp32++; continue; }

            /* Block-sum bounds at ell=4 (adjacent pairs, outer loop, integer) */
            {
                conv_t bs;
                bs = (conv_t)(c0 + c1);
                if (bs * bs > local_thresh[2]) { my_fp32++; continue; }
                bs = (conv_t)(c1 + c2);
                if (bs * bs > local_thresh[2]) { my_fp32++; continue; }
                bs = (conv_t)(c2 + c3);
                if (bs * bs > local_thresh[2]) { my_fp32++; continue; }
            }
            /* Block-sum at ell=6 (interior triple {c1,c2,c3}, integer) */
            {
                conv_t bs = (conv_t)(c1 + c2 + c3);
                if (bs * bs > local_thresh[4]) { my_fp32++; continue; }
            }

            /* FP64 asymmetry (constant across c4 loop) */
            double left_sum_d = (double)c0 + (double)c1 + (double)c2;
            double left_frac_d = left_sum_d * inv_S;
            double dom_d = (left_frac_d > 0.5) ? left_frac_d : (1.0 - left_frac_d);
            double asym_base_d = dom_d - margin;
            if (asym_base_d < 0.0) asym_base_d = 0.0;
            double asym_val_d = 2.0 * asym_base_d * asym_base_d;

            int c4_max = r3 - c0;
            if (c4_max < 0) continue;

            /* === INCREMENTAL CONVOLUTION: precompute base from (c0,c1,c2,c3) ===
             * The autoconvolution of (c0,..,c5) decomposes into:
             *   conv[0..3]: depends only on (c0,c1,c2,c3) — constant across c4 loop
             *   conv[4..10]: base terms + cross-terms involving (c4, c5)
             * This reduces inner-loop work from 21 mults to 11 mults per iteration.
             */
            conv_t conv[11];
            conv[0] = (conv_t)c0 * c0;
            conv[1] = conv[0] + (conv_t)2 * c0 * c1;
            conv[2] = conv[1] + (conv_t)c1 * c1 + (conv_t)2 * c0 * c2;
            conv[3] = conv[2] + (conv_t)2 * ((conv_t)c0 * c3 + (conv_t)c1 * c2);
            conv_t br4 = (conv_t)c2 * c2 + (conv_t)2 * c1 * c3;
            conv_t br5 = (conv_t)2 * c2 * c3;
            conv_t br6 = (conv_t)c3 * c3;
            conv_t two_c0 = (conv_t)2 * c0;
            conv_t two_c1 = (conv_t)2 * c1;
            conv_t two_c2 = (conv_t)2 * c2;
            conv_t two_c3 = (conv_t)2 * c3;

            /* Prefix sums of c[0..3] for dynamic correction (constant across c4) */
            int prefix_c[7];
            prefix_c[0] = 0;
            prefix_c[1] = c0;
            prefix_c[2] = c0 + c1;
            prefix_c[3] = c0 + c1 + c2;
            prefix_c[4] = c0 + c1 + c2 + c3;

            for (int c4 = 0; c4 <= c4_max; c4++) {
                int c5 = r3 - c4;

                /* Canonical palindrome */
                if (c0 == c5) {
                    if (c1 > c4) continue;
                    if (c1 == c4 && c2 > c3) continue;
                }

                /* ell=2 max element + two-max enhanced */
                int max_c = c5, max2_c = 0;
                if (c1 >= max_c) { max2_c = max_c; max_c = c1; } else if (c1 > max2_c) max2_c = c1;
                if (c2 >= max_c) { max2_c = max_c; max_c = c2; } else if (c2 > max2_c) max2_c = c2;
                if (c3 >= max_c) { max2_c = max_c; max_c = c3; } else if (c3 > max2_c) max2_c = c3;
                if (c4 >= max_c) { max2_c = max_c; max_c = c4; } else if (c4 > max2_c) max2_c = c4;
                if (c0 > max2_c) max2_c = c0;
                float max_a = max_c * scale_f;
                if (max_a * max_a * inv_ell2 > thresh_f) { my_fp32++; continue; }
                /* Two-max cross-term at ell=2 (integer) */
                if ((conv_t)2 * max_c * max2_c > local_thresh[0]) { my_fp32++; continue; }

                /* Block-sum bounds at ell=4 (pairs involving c4/c5, integer) */
                {
                    conv_t bs;
                    bs = (conv_t)(c3 + c4);
                    if (bs * bs > local_thresh[2]) { my_fp32++; continue; }
                    bs = (conv_t)(c4 + c5);
                    if (bs * bs > local_thresh[2]) { my_fp32++; continue; }
                }
                /* Block-sum at ell=6 (interior triple {c2,c3,c4}, integer) */
                {
                    conv_t bs = (conv_t)(c2 + c3 + c4);
                    if (bs * bs > local_thresh[4]) { my_fp32++; continue; }
                }
                /* Central conv bound: conv[5] = 2*(c0*c5 + c1*c4 + c2*c3) at ell=2 */
                {
                    conv_t cc = (conv_t)2 * ((conv_t)c0 * c5 + (conv_t)c1 * c4 + (conv_t)c2 * c3);
                    if (cc > local_thresh[0]) { my_fp32++; continue; }
                }

                /* FP64 asymmetry */
                if (asym_val_d >= c_target) { my_asym++; continue; }

                /* Incremental autoconvolution (prefix-summed):
                 * conv[0..3] already set (constant base).
                 * conv[4..10] = base + cross-terms from (c4, c5). */
                conv[4] = conv[3] + br4 + two_c0 * (conv_t)c4;
                conv[5] = conv[4] + br5 + two_c1 * (conv_t)c4 + two_c0 * (conv_t)c5;
                conv[6] = conv[5] + br6 + two_c2 * (conv_t)c4 + two_c1 * (conv_t)c5;
                conv[7] = conv[6] + two_c3 * (conv_t)c4 + two_c2 * (conv_t)c5;
                conv[8] = conv[7] + (conv_t)c4 * (conv_t)c4 + two_c3 * (conv_t)c5;
                conv[9] = conv[8] + (conv_t)2 * (conv_t)c4 * (conv_t)c5;
                conv[10] = conv[9] + (conv_t)c5 * (conv_t)c5;

                /* Complete prefix sums for this (c4, c5) */
                prefix_c[5] = prefix_c[4] + c4;
                prefix_c[6] = S;

                int pruned = 0;
                #pragma unroll
                for (int ell = 2; ell <= 6; ell++) {
                    int n_cv = ell - 1;
                    /* Precompute per-ell constants for dynamic threshold */
                    double dyn_base_ell = dyn_base * (double)ell * inv_4n;
                    double two_ell_inv_4n = 2.0 * (double)ell * inv_4n;
                    int n_win = 11 - n_cv + 1;
                    for (int s_lo = 0; s_lo < n_win; s_lo++) {
                        int s_hi = s_lo + n_cv - 1;
                        conv_t ws = conv[s_hi];
                        if (s_lo > 0) ws -= conv[s_lo - 1];
                        /* Per-window-POSITION dynamic threshold (Lemma 2):
                         * W_int = mass of bins contributing to this window position.
                         * Bin i contributes iff max(0,s_lo-5) <= i <= min(5,s_lo+ell-2). */
                        int lo_bin = (s_lo > 5) ? s_lo - 5 : 0;
                        int hi_bin = (s_lo + ell - 2 < 5) ? s_lo + ell - 2 : 5;
                        int W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin];
                        double dyn_x = dyn_base_ell + two_ell_inv_4n * (double)W_int;
                        conv_t dyn_it = (conv_t)((long long)(dyn_x
                                     * (1.0 - 4.0 * DBL_EPSILON)));
                        if (ws > dyn_it) { pruned = 1; break; }
                    }
                    if (pruned) break;
                }

                if (pruned) {
                    my_test++;
                } else {
                    double best = 0.0;
                    #pragma unroll
                    for (int ell = 2; ell <= 6; ell++) {
                        int n_cv = ell - 1;
                        double inv_norm = inv_norm_arr[ell - 2];
                        int n_win = 11 - n_cv + 1;
                        for (int s_lo = 0; s_lo < n_win; s_lo++) {
                            int s_hi = s_lo + n_cv - 1;
                            conv_t ws = conv[s_hi];
                            if (s_lo > 0) ws -= conv[s_lo - 1];
                            double tv = (double)ws * inv_norm;
                            if (tv > best) best = tv;
                        }
                    }
                    my_surv++;
                    if (survivor_buf != NULL) {
                        int slot = atomicAdd(survivor_count, 1);
                        if (slot < max_survivors) {
                            long long off = (long long)slot * 6;
                            survivor_buf[off + 0] = c0;
                            survivor_buf[off + 1] = c1;
                            survivor_buf[off + 2] = c2;
                            survivor_buf[off + 3] = c3;
                            survivor_buf[off + 4] = c4;
                            survivor_buf[off + 5] = c5;
                        }
                    }
                    if (best < my_tv) {
                        my_tv = best;
                        my_cfg[0] = c0; my_cfg[1] = c1;
                        my_cfg[2] = c2; my_cfg[3] = c3;
                        my_cfg[4] = c4; my_cfg[5] = c5;
                    }
                }
            }
        }
    }

    /* === Per-block reduction: sum counts + min test value === */
    __shared__ int s_fp32[FUSED_BLOCK_SIZE];
    __shared__ int s_asym[FUSED_BLOCK_SIZE];
    __shared__ int s_test[FUSED_BLOCK_SIZE];
    __shared__ int s_surv[FUSED_BLOCK_SIZE];
    __shared__ double s_min_tv[FUSED_BLOCK_SIZE];
    __shared__ int s_cfgs[FUSED_BLOCK_SIZE * D];

    int lane = threadIdx.x;
    s_fp32[lane] = my_fp32;
    s_asym[lane] = my_asym;
    s_test[lane] = my_test;
    s_surv[lane] = my_surv;
    s_min_tv[lane] = my_tv;
    #pragma unroll
    for (int i = 0; i < D; i++) s_cfgs[lane * D + i] = my_cfg[i];
    __syncthreads();

    /* Cross-warp tree reduction */
    for (int s = FUSED_BLOCK_SIZE / 2; s > 16; s >>= 1) {
        if (lane < s) {
            s_fp32[lane] += s_fp32[lane + s];
            s_asym[lane] += s_asym[lane + s];
            s_test[lane] += s_test[lane + s];
            s_surv[lane] += s_surv[lane + s];
            if (s_min_tv[lane + s] < s_min_tv[lane]) {
                s_min_tv[lane] = s_min_tv[lane + s];
                #pragma unroll
                for (int i = 0; i < D; i++)
                    s_cfgs[lane * D + i] = s_cfgs[(lane + s) * D + i];
            }
        }
        __syncthreads();
    }

    /* Intra-warp reduction */
    if (lane < 32) {
        int fp = s_fp32[lane], a = s_asym[lane];
        int t = s_test[lane], sv = s_surv[lane];
        double val = s_min_tv[lane];
        int cfg[D];
        #pragma unroll
        for (int i = 0; i < D; i++) cfg[i] = s_cfgs[lane * D + i];

        #pragma unroll
        for (int s = 16; s > 0; s >>= 1) {
            fp += __shfl_down_sync(0xFFFFFFFF, fp, s);
            a += __shfl_down_sync(0xFFFFFFFF, a, s);
            t += __shfl_down_sync(0xFFFFFFFF, t, s);
            sv += __shfl_down_sync(0xFFFFFFFF, sv, s);

            double other_val = shfl_down_double(0xFFFFFFFF, val, s);
            int other_cfg[D];
            #pragma unroll
            for (int i = 0; i < D; i++)
                other_cfg[i] = __shfl_down_sync(0xFFFFFFFF, cfg[i], s);
            if (other_val < val) {
                val = other_val;
                #pragma unroll
                for (int i = 0; i < D; i++) cfg[i] = other_cfg[i];
            }
        }

        if (lane == 0) {
            block_counts[blockIdx.x * 4 + 0] = (long long)fp;
            block_counts[blockIdx.x * 4 + 1] = (long long)a;
            block_counts[blockIdx.x * 4 + 2] = (long long)t;
            block_counts[blockIdx.x * 4 + 3] = (long long)sv;
            block_min_tv[blockIdx.x] = val;
            #pragma unroll
            for (int i = 0; i < D; i++)
                block_min_configs[blockIdx.x * D + i] = cfg[i];
        }
    }
}
