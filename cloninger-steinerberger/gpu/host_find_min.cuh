#pragma once
/*
 * Host orchestration for find_best_bound_direct (D=4 and D=6).
 *
 * Fused single-pass pipeline:
 *   1. Build zigzag c0 ordering
 *   2. Compute per-c0 triangle/tetrahedron counts (upper bounds on work)
 *   3. Build prefix sums, upload to GPU
 *   4. Launch fused_find_min kernel (zero intermediate DRAM)
 *   5. Host-side final reduction of per-block results
 *
 * Depends on: device_helpers.cuh (CUDA_CHECK, FUSED_BLOCK_SIZE)
 *             phase1_kernels.cuh (fused_find_min)
 */


/* ================================================================
 * Host implementation: find_best_bound_direct for D=4
 * ================================================================ */
static int find_best_bound_direct_d4(
    int S, int n_half, int m,
    double init_min_eff,
    double* result_min_eff,
    int* result_min_config
) {
    const int D = 4;

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
    CUDA_CHECK(cudaMalloc(&d_prefix, (n_c0 + 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_c0_order, n_c0 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_prefix, h_prefix,
        (n_c0 + 1) * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c0_order, h_c0_order,
        n_c0 * sizeof(int), cudaMemcpyHostToDevice));

    /* Kernel launch config */
    int block_size = FUSED_BLOCK_SIZE;
    int grid_size = (int)fmin(
        (double)(total_work + block_size - 1) / block_size, 108.0 * 32);
    if (grid_size < 1) grid_size = 1;

    /* Per-block output buffers */
    double* d_block_min_vals = NULL;
    int* d_block_min_configs = NULL;
    CUDA_CHECK(cudaMalloc(&d_block_min_vals, grid_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_min_configs, grid_size * D * sizeof(int)));

    /* Dispatch INT32 or INT64 based on S*S overflow check */
    bool use_int64 = ((long long)S * S > 2000000000LL);
    if (use_int64) {
        fused_find_min<D, true><<<grid_size, block_size>>>(
            S, n_half, m, corr, margin, init_min_eff,
            inv_m_f, thresh_f, margin_f, asym_limit_f,
            d_prefix, d_c0_order, n_c0, 0LL, total_work,
            d_block_min_vals, d_block_min_configs);
    } else {
        fused_find_min<D, false><<<grid_size, block_size>>>(
            S, n_half, m, corr, margin, init_min_eff,
            inv_m_f, thresh_f, margin_f, asym_limit_f,
            d_prefix, d_c0_order, n_c0, 0LL, total_work,
            d_block_min_vals, d_block_min_configs);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Read and reduce per-block results */
    double* h_block_vals = (double*)malloc(grid_size * sizeof(double));
    int* h_block_cfgs = (int*)malloc(grid_size * D * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_block_vals, d_block_min_vals,
        grid_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_block_cfgs, d_block_min_configs,
        grid_size * D * sizeof(int), cudaMemcpyDeviceToHost));

    double best_eff = init_min_eff;
    int best_config[D];
    for (int i = 0; i < D; i++) best_config[i] = 0;
    int found_better = 0;

    for (int b = 0; b < grid_size; b++) {
        if (h_block_vals[b] < best_eff) {
            best_eff = h_block_vals[b];
            for (int i = 0; i < D; i++)
                best_config[i] = h_block_cfgs[b * D + i];
            found_better = 1;
        }
    }

    *result_min_eff = best_eff;
    if (found_better) {
        for (int i = 0; i < D; i++) result_min_config[i] = best_config[i];
    }

    cudaFree(d_prefix);
    cudaFree(d_c0_order);
    cudaFree(d_block_min_vals);
    cudaFree(d_block_min_configs);
    free(h_c0_order);
    free(h_prefix);
    free(h_block_vals);
    free(h_block_cfgs);

    return 0;
}


/* ================================================================
 * Host implementation: find_best_bound_direct for D=6
 * ================================================================ */
static int find_best_bound_direct_d6(
    int S, int n_half, int m,
    double init_min_eff,
    double* result_min_eff,
    int* result_min_config
) {
    const int D = 6;

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
    CUDA_CHECK(cudaMalloc(&d_prefix, (n_c0 + 1) * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_c0_order, n_c0 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_prefix, h_prefix,
        (n_c0 + 1) * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c0_order, h_c0_order,
        n_c0 * sizeof(int), cudaMemcpyHostToDevice));

    int block_size = FUSED_BLOCK_SIZE;
    int grid_size = (int)fmin(
        (double)(total_work + block_size - 1) / block_size, 108.0 * 32);
    if (grid_size < 1) grid_size = 1;

    double* d_block_min_vals = NULL;
    int* d_block_min_configs = NULL;
    CUDA_CHECK(cudaMalloc(&d_block_min_vals, grid_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_min_configs, grid_size * D * sizeof(int)));

    /* Dispatch INT32 or INT64 based on S*S overflow check */
    bool use_int64 = ((long long)S * S > 2000000000LL);
    if (use_int64) {
        fused_find_min<D, true><<<grid_size, block_size>>>(
            S, n_half, m, corr, margin, init_min_eff,
            inv_m_f, thresh_f, margin_f, asym_limit_f,
            d_prefix, d_c0_order, n_c0, 0LL, total_work,
            d_block_min_vals, d_block_min_configs);
    } else {
        fused_find_min<D, false><<<grid_size, block_size>>>(
            S, n_half, m, corr, margin, init_min_eff,
            inv_m_f, thresh_f, margin_f, asym_limit_f,
            d_prefix, d_c0_order, n_c0, 0LL, total_work,
            d_block_min_vals, d_block_min_configs);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double* h_block_vals = (double*)malloc(grid_size * sizeof(double));
    int* h_block_cfgs = (int*)malloc(grid_size * D * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_block_vals, d_block_min_vals,
        grid_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_block_cfgs, d_block_min_configs,
        grid_size * D * sizeof(int), cudaMemcpyDeviceToHost));

    double best_eff = init_min_eff;
    int best_config[D];
    for (int i = 0; i < D; i++) best_config[i] = 0;
    int found_better = 0;

    for (int b = 0; b < grid_size; b++) {
        if (h_block_vals[b] < best_eff) {
            best_eff = h_block_vals[b];
            for (int i = 0; i < D; i++)
                best_config[i] = h_block_cfgs[b * D + i];
            found_better = 1;
        }
    }

    *result_min_eff = best_eff;
    if (found_better) {
        for (int i = 0; i < D; i++) result_min_config[i] = best_config[i];
    }

    cudaFree(d_prefix);
    cudaFree(d_c0_order);
    cudaFree(d_block_min_vals);
    cudaFree(d_block_min_configs);
    free(h_c0_order);
    free(h_prefix);
    free(h_block_vals);
    free(h_block_cfgs);

    return 0;
}
