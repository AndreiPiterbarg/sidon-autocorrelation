#pragma once
/*
 * Host orchestration for run_single_level (D=4 and D=6).
 *
 * Fused single-pass pipeline with integer thresholds:
 *   1. Build zigzag c0 ordering
 *   2. Compute per-c0 triangle/tetrahedron counts
 *   3. Precompute per-ell integer thresholds (zero FP64 in inner loop)
 *   4. Launch fused_prove_target kernel (zero intermediate DRAM)
 *   5. Host-side aggregation of per-block counts + min test value
 *
 * Depends on: device_helpers.cuh (CUDA_CHECK, FUSED_BLOCK_SIZE)
 *             phase1_kernels.cuh (fused_prove_target)
 */


/* ================================================================
 * Host implementation: run_single_level for D=4
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

    /* INT32 overflow guard */
    if ((long long)S * S > 2000000000LL) {
        fprintf(stderr, "GPU: S=%d too large for INT32 (S^2 > 2B)\n", S);
        return -3;
    }

    double corr = 2.0 / m + 1.0 / ((double)m * m);
    double prune_target = c_target + corr;
    double margin = 1.0 / (4.0 * m);
    double fp_margin = 1e-9;
    double thresh = prune_target + fp_margin;
    float inv_m_f = 1.0f / (float)m;

    float thresh_f = (float)thresh * (1.0f + 1e-5f);
    float margin_f = (float)margin;
    float asym_limit_f = (float)c_target * (1.0f + 1e-5f);

    /* Precompute per-ell integer thresholds: int_thresh[ell-2] = floor(thresh * m^2 * 4 * n_half * ell) */
    int h_int_thresh[D - 1];
    for (int ell = 2; ell <= D; ell++) {
        double x = thresh * (double)m * (double)m * 4.0 * (double)n_half * (double)ell;
        h_int_thresh[ell - 2] = (int)x;  /* floor for positive x */
    }

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

    /* Compute prefix sums of triangle counts per c0 */
    long long* h_prefix = (long long*)malloc((n_c0 + 1) * sizeof(long long));
    h_prefix[0] = 0;
    for (int i = 0; i < n_c0; i++) {
        int c0 = h_c0_order[i];
        long long R = (long long)(S - 2 * c0);
        h_prefix[i + 1] = h_prefix[i] + (R + 1) * (R + 2) / 2;
    }
    long long total_work = h_prefix[n_c0];

    /* Upload to GPU */
    long long* d_prefix = NULL;
    int* d_c0_order = NULL;
    int* d_int_thresh = NULL;
    CUDA_CHECK(cudaMalloc(&d_prefix, (n_c0 + 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_c0_order, n_c0 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_int_thresh, (D - 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_prefix, h_prefix,
        (n_c0 + 1) * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c0_order, h_c0_order,
        n_c0 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_int_thresh, h_int_thresh,
        (D - 1) * sizeof(int), cudaMemcpyHostToDevice));

    int block_size = FUSED_BLOCK_SIZE;
    int grid_size = (int)fmin(
        (double)(total_work + block_size - 1) / block_size, 65535.0);
    if (grid_size < 1) grid_size = 1;

    /* Per-block output buffers */
    long long* d_block_counts = NULL;
    double* d_block_min_tv = NULL;
    int* d_block_min_configs = NULL;
    CUDA_CHECK(cudaMalloc(&d_block_counts, (long long)grid_size * 3 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_block_min_tv, grid_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_min_configs, grid_size * D * sizeof(int)));

    fused_prove_target<D><<<grid_size, block_size>>>(
        S, n_half, m, margin, c_target, thresh,
        inv_m_f, thresh_f, margin_f, asym_limit_f,
        d_prefix, d_c0_order, n_c0, total_work,
        d_int_thresh,
        d_block_counts, d_block_min_tv, d_block_min_configs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Read and reduce per-block results */
    long long* h_counts = (long long*)malloc(grid_size * 3 * sizeof(long long));
    double* h_min_tvs = (double*)malloc(grid_size * sizeof(double));
    int* h_min_cfgs = (int*)malloc(grid_size * D * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_counts, d_block_counts,
        grid_size * 3 * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_min_tvs, d_block_min_tv,
        grid_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_min_cfgs, d_block_min_configs,
        grid_size * D * sizeof(int), cudaMemcpyDeviceToHost));

    long long total_asym = 0, total_test = 0, total_surv = 0;
    double min_tv = 1e30;
    int min_cfg[D];
    for (int i = 0; i < D; i++) min_cfg[i] = 0;

    for (int b = 0; b < grid_size; b++) {
        total_asym += h_counts[b * 3 + 0];
        total_test += h_counts[b * 3 + 1];
        total_surv += h_counts[b * 3 + 2];
        if (h_min_tvs[b] < min_tv) {
            min_tv = h_min_tvs[b];
            for (int i = 0; i < D; i++)
                min_cfg[i] = h_min_cfgs[b * D + i];
        }
    }

    *out_n_pruned_asym = total_asym;
    *out_n_pruned_test = total_test;
    *out_n_survivors = total_surv;
    *out_min_test_val = min_tv;
    for (int i = 0; i < D; i++) out_min_test_config[i] = min_cfg[i];

    cudaFree(d_prefix);
    cudaFree(d_c0_order);
    cudaFree(d_int_thresh);
    cudaFree(d_block_counts);
    cudaFree(d_block_min_tv);
    cudaFree(d_block_min_configs);
    free(h_c0_order);
    free(h_prefix);
    free(h_counts);
    free(h_min_tvs);
    free(h_min_cfgs);

    return 0;
}


/* ================================================================
 * Host implementation: run_single_level for D=6
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

    /* INT32 overflow guard */
    if ((long long)S * S > 2000000000LL) {
        fprintf(stderr, "GPU: S=%d too large for INT32 (S^2 > 2B)\n", S);
        return -3;
    }

    double corr = 2.0 / m + 1.0 / ((double)m * m);
    double prune_target = c_target + corr;
    double margin = 1.0 / (4.0 * m);
    double fp_margin = 1e-9;
    double thresh = prune_target + fp_margin;
    float inv_m_f = 1.0f / (float)m;

    float thresh_f = (float)thresh * (1.0f + 1e-5f);
    float margin_f = (float)margin;
    float asym_limit_f = (float)c_target * (1.0f + 1e-5f);

    /* Precompute per-ell integer thresholds */
    int h_int_thresh[D - 1];
    for (int ell = 2; ell <= D; ell++) {
        double x = thresh * (double)m * (double)m * 4.0 * (double)n_half * (double)ell;
        h_int_thresh[ell - 2] = (int)x;
    }

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

    /* Compute prefix sums of tetrahedral counts per c0 */
    long long* h_prefix = (long long*)malloc((n_c0 + 1) * sizeof(long long));
    h_prefix[0] = 0;
    for (int i = 0; i < n_c0; i++) {
        int c0 = h_c0_order[i];
        long long R = (long long)(S - 2 * c0);
        h_prefix[i + 1] = h_prefix[i] + (R + 1) * (R + 2) * (R + 3) / 6;
    }
    long long total_work = h_prefix[n_c0];

    /* Upload to GPU */
    long long* d_prefix = NULL;
    int* d_c0_order = NULL;
    int* d_int_thresh = NULL;
    CUDA_CHECK(cudaMalloc(&d_prefix, (n_c0 + 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_c0_order, n_c0 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_int_thresh, (D - 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_prefix, h_prefix,
        (n_c0 + 1) * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c0_order, h_c0_order,
        n_c0 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_int_thresh, h_int_thresh,
        (D - 1) * sizeof(int), cudaMemcpyHostToDevice));

    int block_size = FUSED_BLOCK_SIZE;
    int grid_size = (int)fmin(
        (double)(total_work + block_size - 1) / block_size, 65535.0);
    if (grid_size < 1) grid_size = 1;

    long long* d_block_counts = NULL;
    double* d_block_min_tv = NULL;
    int* d_block_min_configs = NULL;
    CUDA_CHECK(cudaMalloc(&d_block_counts, (long long)grid_size * 3 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_block_min_tv, grid_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_min_configs, grid_size * D * sizeof(int)));

    fused_prove_target<D><<<grid_size, block_size>>>(
        S, n_half, m, margin, c_target, thresh,
        inv_m_f, thresh_f, margin_f, asym_limit_f,
        d_prefix, d_c0_order, n_c0, total_work,
        d_int_thresh,
        d_block_counts, d_block_min_tv, d_block_min_configs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    long long* h_counts = (long long*)malloc(grid_size * 3 * sizeof(long long));
    double* h_min_tvs = (double*)malloc(grid_size * sizeof(double));
    int* h_min_cfgs = (int*)malloc(grid_size * D * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_counts, d_block_counts,
        grid_size * 3 * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_min_tvs, d_block_min_tv,
        grid_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_min_cfgs, d_block_min_configs,
        grid_size * D * sizeof(int), cudaMemcpyDeviceToHost));

    long long total_asym = 0, total_test = 0, total_surv = 0;
    double min_tv = 1e30;
    int min_cfg[D];
    for (int i = 0; i < D; i++) min_cfg[i] = 0;

    for (int b = 0; b < grid_size; b++) {
        total_asym += h_counts[b * 3 + 0];
        total_test += h_counts[b * 3 + 1];
        total_surv += h_counts[b * 3 + 2];
        if (h_min_tvs[b] < min_tv) {
            min_tv = h_min_tvs[b];
            for (int i = 0; i < D; i++)
                min_cfg[i] = h_min_cfgs[b * D + i];
        }
    }

    *out_n_pruned_asym = total_asym;
    *out_n_pruned_test = total_test;
    *out_n_survivors = total_surv;
    *out_min_test_val = min_tv;
    for (int i = 0; i < D; i++) out_min_test_config[i] = min_cfg[i];

    cudaFree(d_prefix);
    cudaFree(d_c0_order);
    cudaFree(d_int_thresh);
    cudaFree(d_block_counts);
    cudaFree(d_block_min_tv);
    cudaFree(d_block_min_configs);
    free(h_c0_order);
    free(h_prefix);
    free(h_counts);
    free(h_min_tvs);
    free(h_min_cfgs);

    return 0;
}
