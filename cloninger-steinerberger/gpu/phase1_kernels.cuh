#pragma once
/*
 * Fused single-pass GPU kernels: composition generation + pruning + autoconvolution.
 *
 * Eliminates all DRAM intermediate traffic by fusing Phase 1 and Phase 2
 * into a single kernel. Each thread generates compositions from a flat index,
 * applies cheap FP32 pruning, then computes INT32 autoconvolution in registers.
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
template <int D>
__global__ void __launch_bounds__(FUSED_BLOCK_SIZE)
fused_find_min(
    int S,
    int n_half,
    int m,
    double corr,
    double margin,
    float inv_m_f,
    float thresh_f,
    float margin_f,
    float asym_limit_f,
    const long long* __restrict__ prefix_sums,
    const int* __restrict__ c0_order,
    int n_c0,
    long long total_work,
    double* __restrict__ block_min_vals,
    int* __restrict__ block_min_configs
) {
    constexpr int HALF_D = D / 2;
    constexpr int CONV_LEN = 2 * D - 1;

    long long flat_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    double my_eff = 1e30;
    int my_cfg[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) my_cfg[i] = 0;

    float norm_ell_d = 4.0f * n_half * D;
    float inv_ell2 = 1.0f / (4.0f * n_half * 2);
    float total_mass_f = (float)S;

    for (long long work_idx = flat_idx; work_idx < total_work; work_idx += stride) {
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
            if (pair_left * pair_left * inv_m_f * inv_m_f / norm_ell_d > thresh_f) continue;
            float pair_right = (float)(c2 + c3);
            if (pair_right * pair_right * inv_m_f * inv_m_f / norm_ell_d > thresh_f) continue;

            /* 4. ell=2 max element */
            int max_c = c3;
            if (c1 > max_c) max_c = c1;
            if (c2 > max_c) max_c = c2;
            float max_a = max_c * inv_m_f;
            if (max_a * max_a * inv_ell2 > thresh_f) continue;

            /* === INT32 autoconvolution === */
            int c[4] = {c0, c1, c2, c3};
            int conv[CONV_LEN];
            #pragma unroll
            for (int k = 0; k < CONV_LEN; k++) conv[k] = 0;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                conv[2*i] += c[i] * c[i];
                #pragma unroll
                for (int j = i+1; j < 4; j++)
                    conv[i+j] += 2 * c[i] * c[j];
            }
            #pragma unroll
            for (int k = 1; k < CONV_LEN; k++) conv[k] += conv[k-1];

            /* Window max (FP64) */
            double inv_m_sq = 1.0 / ((double)m * (double)m);
            double best = 0.0;
            #pragma unroll
            for (int ell = 4; ell >= 2; ell--) {
                int n_cv = ell - 1;
                double inv_norm = inv_m_sq / (4.0 * n_half * ell);
                int n_windows = CONV_LEN - n_cv + 1;
                for (int s_lo = 0; s_lo < n_windows; s_lo++) {
                    int s_hi = s_lo + n_cv - 1;
                    int ws = conv[s_hi];
                    if (s_lo > 0) ws -= conv[s_lo - 1];
                    double tv = (double)ws * inv_norm;
                    if (tv > best) best = tv;
                }
            }

            double eff = best - corr;

            /* FP64 asymmetry bound */
            double left_sum_d = (double)c0 + (double)c1;
            double left_frac_d = left_sum_d / (double)S;
            double dom_d = (left_frac_d > 0.5) ? left_frac_d : (1.0 - left_frac_d);
            double asym_base_d = dom_d - margin;
            if (asym_base_d < 0.0) asym_base_d = 0.0;
            double asym_val_d = 2.0 * asym_base_d * asym_base_d;
            if (asym_val_d > eff) eff = asym_val_d;

            if (eff < my_eff) {
                my_eff = eff;
                my_cfg[0] = c0; my_cfg[1] = c1;
                my_cfg[2] = c2; my_cfg[3] = c3;
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
            if (left3 * left3 * inv_m_f * inv_m_f / norm_ell_d > thresh_f) continue;
            if (right3 * right3 * inv_m_f * inv_m_f / norm_ell_d > thresh_f) continue;

            float left_frac = left3 / total_mass_f;
            float dom = (left_frac > 0.5f) ? left_frac : (1.0f - left_frac);
            float asym_base = dom - margin_f;
            if (asym_base < 0.0f) asym_base = 0.0f;
            float asym_val = 2.0f * asym_base * asym_base;
            if (asym_val >= asym_limit_f) continue;

            /* FP64 asymmetry (constant across c4 loop) */
            double left_sum_d = (double)c0 + (double)c1 + (double)c2;
            double left_frac_d = left_sum_d / (double)S;
            double dom_d = (left_frac_d > 0.5) ? left_frac_d : (1.0 - left_frac_d);
            double asym_base_d = dom_d - margin;
            if (asym_base_d < 0.0) asym_base_d = 0.0;
            double asym_val_d = 2.0 * asym_base_d * asym_base_d;

            double inv_m_sq = 1.0 / ((double)m * (double)m);

            /* Inner c4 loop */
            int c4_max = r3 - c0;
            if (c4_max < 0) continue;

            for (int c4 = 0; c4 <= c4_max; c4++) {
                int c5 = r3 - c4;

                /* Canonical palindrome */
                if (c0 == c5) {
                    if (c1 > c4) continue;
                    if (c1 == c4 && c2 > c3) continue;
                }

                /* ell=2 max element */
                int max_c = c5;
                if (c1 > max_c) max_c = c1;
                if (c2 > max_c) max_c = c2;
                if (c3 > max_c) max_c = c3;
                if (c4 > max_c) max_c = c4;
                float max_a = max_c * inv_m_f;
                if (max_a * max_a * inv_ell2 > thresh_f) continue;

                /* INT32 autoconvolution */
                int cc[6] = {c0, c1, c2, c3, c4, c5};
                int conv[11]; /* CONV_LEN for D=6 */
                #pragma unroll
                for (int k = 0; k < 11; k++) conv[k] = 0;
                #pragma unroll
                for (int i = 0; i < 6; i++) {
                    conv[2*i] += cc[i] * cc[i];
                    #pragma unroll
                    for (int j = i+1; j < 6; j++)
                        conv[i+j] += 2 * cc[i] * cc[j];
                }
                #pragma unroll
                for (int k = 1; k < 11; k++) conv[k] += conv[k-1];

                /* Window max (FP64) */
                double best = 0.0;
                #pragma unroll
                for (int ell = 6; ell >= 2; ell--) {
                    int n_cv = ell - 1;
                    double inv_norm = inv_m_sq / (4.0 * n_half * ell);
                    int n_win = 11 - n_cv + 1;
                    for (int s_lo = 0; s_lo < n_win; s_lo++) {
                        int s_hi = s_lo + n_cv - 1;
                        int ws = conv[s_hi];
                        if (s_lo > 0) ws -= conv[s_lo - 1];
                        double tv = (double)ws * inv_norm;
                        if (tv > best) best = tv;
                    }
                }

                double eff = best - corr;
                if (asym_val_d > eff) eff = asym_val_d;

                if (eff < my_eff) {
                    my_eff = eff;
                    my_cfg[0] = c0; my_cfg[1] = c1;
                    my_cfg[2] = c2; my_cfg[3] = c3;
                    my_cfg[4] = c4; my_cfg[5] = c5;
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
template <int D>
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
    long long total_work,
    const int* __restrict__ int_thresh,   /* [D-1] per-ell integer thresholds */
    long long* __restrict__ block_counts, /* [num_blocks * 3] */
    double* __restrict__ block_min_tv,
    int* __restrict__ block_min_configs
) {
    constexpr int HALF_D = D / 2;
    constexpr int CONV_LEN = 2 * D - 1;

    long long flat_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    int my_asym = 0, my_test = 0, my_surv = 0;
    double my_tv = 1e30;
    int my_cfg[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) my_cfg[i] = 0;

    float norm_ell_d = 4.0f * n_half * D;
    float inv_ell2 = 1.0f / (4.0f * n_half * 2);
    float total_mass_f = (float)S;

    for (long long work_idx = flat_idx; work_idx < total_work; work_idx += stride) {
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
            if (asym_val >= asym_limit_f) continue;

            /* 3. Pair-sum bound */
            if (pair_left * pair_left * inv_m_f * inv_m_f / norm_ell_d > thresh_f) continue;
            float pair_right = (float)(c2 + c3);
            if (pair_right * pair_right * inv_m_f * inv_m_f / norm_ell_d > thresh_f) continue;

            /* 4. ell=2 max element */
            int max_c = c3;
            if (c1 > max_c) max_c = c1;
            if (c2 > max_c) max_c = c2;
            float max_a = max_c * inv_m_f;
            if (max_a * max_a * inv_ell2 > thresh_f) continue;

            /* 5. FP64 asymmetry -> asym_pruned */
            double left_sum_d = (double)c0 + (double)c1;
            double left_frac_d = left_sum_d / (double)S;
            double dom_d = (left_frac_d > 0.5) ? left_frac_d : (1.0 - left_frac_d);
            double asym_base_d = dom_d - margin;
            if (asym_base_d < 0.0) asym_base_d = 0.0;
            double asym_val_d = 2.0 * asym_base_d * asym_base_d;
            if (asym_val_d >= c_target) { my_asym++; continue; }

            /* 6. INT32 autoconvolution + integer threshold */
            int c[4] = {c0, c1, c2, c3};
            int conv[CONV_LEN];
            #pragma unroll
            for (int k = 0; k < CONV_LEN; k++) conv[k] = 0;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                conv[2*i] += c[i] * c[i];
                #pragma unroll
                for (int j = i+1; j < 4; j++)
                    conv[i+j] += 2 * c[i] * c[j];
            }
            #pragma unroll
            for (int k = 1; k < CONV_LEN; k++) conv[k] += conv[k-1];

            int pruned = 0;
            #pragma unroll
            for (int ell = 4; ell >= 2; ell--) {
                int n_cv = ell - 1;
                int it = int_thresh[ell - 2];
                int n_windows = CONV_LEN - n_cv + 1;
                for (int s_lo = 0; s_lo < n_windows; s_lo++) {
                    int s_hi = s_lo + n_cv - 1;
                    int ws = conv[s_hi];
                    if (s_lo > 0) ws -= conv[s_lo - 1];
                    if (ws > it) { pruned = 1; break; }
                }
                if (pruned) break;
            }

            if (pruned) {
                my_test++;
            } else {
                /* Survivor: compute FP64 test value for reporting */
                double inv_m_sq = 1.0 / ((double)m * (double)m);
                double best = 0.0;
                #pragma unroll
                for (int ell = 4; ell >= 2; ell--) {
                    int n_cv = ell - 1;
                    double inv_norm = inv_m_sq / (4.0 * n_half * ell);
                    int n_windows = CONV_LEN - n_cv + 1;
                    for (int s_lo = 0; s_lo < n_windows; s_lo++) {
                        int s_hi = s_lo + n_cv - 1;
                        int ws = conv[s_hi];
                        if (s_lo > 0) ws -= conv[s_lo - 1];
                        double tv = (double)ws * inv_norm;
                        if (tv > best) best = tv;
                    }
                }
                my_surv++;
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
            if (left3 * left3 * inv_m_f * inv_m_f / norm_ell_d > thresh_f) continue;
            if (right3 * right3 * inv_m_f * inv_m_f / norm_ell_d > thresh_f) continue;

            float left_frac = left3 / total_mass_f;
            float dom = (left_frac > 0.5f) ? left_frac : (1.0f - left_frac);
            float asym_base = dom - margin_f;
            if (asym_base < 0.0f) asym_base = 0.0f;
            float asym_val = 2.0f * asym_base * asym_base;
            if (asym_val >= asym_limit_f) continue;

            /* FP64 asymmetry (constant across c4 loop) */
            double left_sum_d = (double)c0 + (double)c1 + (double)c2;
            double left_frac_d = left_sum_d / (double)S;
            double dom_d = (left_frac_d > 0.5) ? left_frac_d : (1.0 - left_frac_d);
            double asym_base_d = dom_d - margin;
            if (asym_base_d < 0.0) asym_base_d = 0.0;
            double asym_val_d = 2.0 * asym_base_d * asym_base_d;

            int c4_max = r3 - c0;
            if (c4_max < 0) continue;

            double inv_m_sq = 1.0 / ((double)m * (double)m);

            for (int c4 = 0; c4 <= c4_max; c4++) {
                int c5 = r3 - c4;

                /* Canonical palindrome */
                if (c0 == c5) {
                    if (c1 > c4) continue;
                    if (c1 == c4 && c2 > c3) continue;
                }

                /* ell=2 max element */
                int max_c = c5;
                if (c1 > max_c) max_c = c1;
                if (c2 > max_c) max_c = c2;
                if (c3 > max_c) max_c = c3;
                if (c4 > max_c) max_c = c4;
                float max_a = max_c * inv_m_f;
                if (max_a * max_a * inv_ell2 > thresh_f) continue;

                /* FP64 asymmetry */
                if (asym_val_d >= c_target) { my_asym++; continue; }

                /* INT32 autoconvolution */
                int cc[6] = {c0, c1, c2, c3, c4, c5};
                int conv[11];
                #pragma unroll
                for (int k = 0; k < 11; k++) conv[k] = 0;
                #pragma unroll
                for (int i = 0; i < 6; i++) {
                    conv[2*i] += cc[i] * cc[i];
                    #pragma unroll
                    for (int j = i+1; j < 6; j++)
                        conv[i+j] += 2 * cc[i] * cc[j];
                }
                #pragma unroll
                for (int k = 1; k < 11; k++) conv[k] += conv[k-1];

                int pruned = 0;
                #pragma unroll
                for (int ell = 6; ell >= 2; ell--) {
                    int n_cv = ell - 1;
                    int it = int_thresh[ell - 2];
                    int n_win = 11 - n_cv + 1;
                    for (int s_lo = 0; s_lo < n_win; s_lo++) {
                        int s_hi = s_lo + n_cv - 1;
                        int ws = conv[s_hi];
                        if (s_lo > 0) ws -= conv[s_lo - 1];
                        if (ws > it) { pruned = 1; break; }
                    }
                    if (pruned) break;
                }

                if (pruned) {
                    my_test++;
                } else {
                    double best = 0.0;
                    #pragma unroll
                    for (int ell = 6; ell >= 2; ell--) {
                        int n_cv = ell - 1;
                        double inv_norm = inv_m_sq / (4.0 * n_half * ell);
                        int n_win = 11 - n_cv + 1;
                        for (int s_lo = 0; s_lo < n_win; s_lo++) {
                            int s_hi = s_lo + n_cv - 1;
                            int ws = conv[s_hi];
                            if (s_lo > 0) ws -= conv[s_lo - 1];
                            double tv = (double)ws * inv_norm;
                            if (tv > best) best = tv;
                        }
                    }
                    my_surv++;
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
    __shared__ int s_asym[FUSED_BLOCK_SIZE];
    __shared__ int s_test[FUSED_BLOCK_SIZE];
    __shared__ int s_surv[FUSED_BLOCK_SIZE];
    __shared__ double s_min_tv[FUSED_BLOCK_SIZE];
    __shared__ int s_cfgs[FUSED_BLOCK_SIZE * D];

    int lane = threadIdx.x;
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
        int a = s_asym[lane], t = s_test[lane], sv = s_surv[lane];
        double val = s_min_tv[lane];
        int cfg[D];
        #pragma unroll
        for (int i = 0; i < D; i++) cfg[i] = s_cfgs[lane * D + i];

        #pragma unroll
        for (int s = 16; s > 0; s >>= 1) {
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
            block_counts[blockIdx.x * 3 + 0] = (long long)a;
            block_counts[blockIdx.x * 3 + 1] = (long long)t;
            block_counts[blockIdx.x * 3 + 2] = (long long)sv;
            block_min_tv[blockIdx.x] = val;
            #pragma unroll
            for (int i = 0; i < D; i++)
                block_min_configs[blockIdx.x * D + i] = cfg[i];
        }
    }
}
