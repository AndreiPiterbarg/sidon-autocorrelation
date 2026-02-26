#pragma once
/*
 * Host orchestration for run_single_level (D=4 and D=6).
 *
 * Fused single-pass pipeline with dynamic per-window-size thresholds:
 *   1. Build zigzag c0 ordering
 *   2. Compute per-c0 triangle/tetrahedron counts
 *   3. Launch fused_prove_target kernel (dynamic thresholds computed per-thread)
 *   4. Host-side aggregation of per-block counts + min test value
 *
 * Depends on: device_helpers.cuh (CUDA_CHECK, FUSED_BLOCK_SIZE)
 *             phase1_kernels.cuh (fused_prove_target)
 *             host_find_min.cuh  (count_per_c0<D>)
 */


/* ================================================================
 * Unified template: run_single_level for D=4 or D=6,
 * with optional survivor extraction (when survivor params are non-NULL).
 * ================================================================ */
template <int D>
static int run_single_level_impl(
    int S, int n_half, int m,
    double c_target,
    long long* out_n_fp32_skipped,
    long long* out_n_pruned_asym,
    long long* out_n_pruned_test,
    long long* out_n_survivors,
    double* out_min_test_val,
    int* out_min_test_config,
    int* out_survivor_configs,
    int* out_n_extracted,
    int max_survivors
) {
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

    /* Compute prefix sums of per-c0 counts */
    long long* h_prefix = (long long*)malloc((n_c0 + 1) * sizeof(long long));
    h_prefix[0] = 0;
    for (int i = 0; i < n_c0; i++) {
        int c0 = h_c0_order[i];
        long long R = (long long)(S - 2 * c0);
        h_prefix[i + 1] = h_prefix[i] + count_per_c0<D>(R);
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

    /* Per-block output buffers */
    long long* d_block_counts = NULL;
    double* d_block_min_tv = NULL;
    int* d_block_min_configs = NULL;
    CUDA_CHECK(cudaMalloc(&d_block_counts, (long long)grid_size * 4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_block_min_tv, grid_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_min_configs, grid_size * D * sizeof(int)));

    /* Survivor extraction buffers (only when extracting) */
    bool extracting = (out_survivor_configs != NULL);
    int* d_survivor_buf = NULL;
    int* d_survivor_count = NULL;
    if (extracting) {
        CUDA_CHECK(cudaMalloc(&d_survivor_buf, (long long)max_survivors * D * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_survivor_count, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_survivor_count, 0, sizeof(int)));
    }

    /* Dispatch INT32 or INT64 based on S*S overflow check */
    bool use_int64 = ((long long)S * S > 2000000000LL);
    if (use_int64) {
        fused_prove_target<D, true><<<grid_size, block_size>>>(
            S, n_half, m, margin, c_target, thresh,
            inv_m_f, thresh_f, margin_f, asym_limit_f,
            d_prefix, d_c0_order, n_c0, 0LL, total_work,
            d_block_counts, d_block_min_tv, d_block_min_configs,
            d_survivor_buf, d_survivor_count,
            extracting ? max_survivors : 0);
    } else {
        fused_prove_target<D, false><<<grid_size, block_size>>>(
            S, n_half, m, margin, c_target, thresh,
            inv_m_f, thresh_f, margin_f, asym_limit_f,
            d_prefix, d_c0_order, n_c0, 0LL, total_work,
            d_block_counts, d_block_min_tv, d_block_min_configs,
            d_survivor_buf, d_survivor_count,
            extracting ? max_survivors : 0);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Read survivor data if extracting */
    if (extracting) {
        int h_survivor_count = 0;
        CUDA_CHECK(cudaMemcpy(&h_survivor_count, d_survivor_count, sizeof(int), cudaMemcpyDeviceToHost));
        int n_ext = h_survivor_count;
        if (n_ext > max_survivors) n_ext = max_survivors;
        if (n_ext > 0) {
            CUDA_CHECK(cudaMemcpy(out_survivor_configs, d_survivor_buf,
                (long long)n_ext * D * sizeof(int), cudaMemcpyDeviceToHost));
        }
        *out_n_extracted = n_ext;
    }

    /* Read and reduce per-block results */
    long long* h_counts = (long long*)malloc(grid_size * 4 * sizeof(long long));
    double* h_min_tvs = (double*)malloc(grid_size * sizeof(double));
    int* h_min_cfgs = (int*)malloc(grid_size * D * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_counts, d_block_counts,
        grid_size * 4 * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_min_tvs, d_block_min_tv,
        grid_size * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_min_cfgs, d_block_min_configs,
        grid_size * D * sizeof(int), cudaMemcpyDeviceToHost));

    long long total_fp32 = 0, total_asym = 0, total_test = 0, total_surv = 0;
    double min_tv = 1e30;
    int min_cfg[D];
    for (int i = 0; i < D; i++) min_cfg[i] = 0;

    for (int b = 0; b < grid_size; b++) {
        total_fp32 += h_counts[b * 4 + 0];
        total_asym += h_counts[b * 4 + 1];
        total_test += h_counts[b * 4 + 2];
        total_surv += h_counts[b * 4 + 3];
        if (h_min_tvs[b] < min_tv) {
            min_tv = h_min_tvs[b];
            for (int i = 0; i < D; i++)
                min_cfg[i] = h_min_cfgs[b * D + i];
        }
    }

    *out_n_fp32_skipped = total_fp32;
    *out_n_pruned_asym = total_asym;
    *out_n_pruned_test = total_test;
    *out_n_survivors = total_surv;
    *out_min_test_val = min_tv;
    for (int i = 0; i < D; i++) out_min_test_config[i] = min_cfg[i];

    cudaFree(d_prefix);
    cudaFree(d_c0_order);
    cudaFree(d_block_counts);
    cudaFree(d_block_min_tv);
    cudaFree(d_block_min_configs);
    if (extracting) {
        cudaFree(d_survivor_buf);
        cudaFree(d_survivor_count);
    }
    free(h_c0_order);
    free(h_prefix);
    free(h_counts);
    free(h_min_tvs);
    free(h_min_cfgs);

    return 0;
}


/* Thin wrappers preserving exact signatures called by dispatch.cuh */

static int run_single_level_d4(
    int S, int n_half, int m, double c_target,
    long long* out_n_fp32_skipped, long long* out_n_pruned_asym,
    long long* out_n_pruned_test, long long* out_n_survivors,
    double* out_min_test_val, int* out_min_test_config
) {
    return run_single_level_impl<4>(S, n_half, m, c_target,
        out_n_fp32_skipped, out_n_pruned_asym, out_n_pruned_test, out_n_survivors,
        out_min_test_val, out_min_test_config, NULL, NULL, 0);
}

static int run_single_level_d4_extract(
    int S, int n_half, int m, double c_target,
    long long* out_n_fp32_skipped, long long* out_n_pruned_asym,
    long long* out_n_pruned_test, long long* out_n_survivors,
    double* out_min_test_val, int* out_min_test_config,
    int* out_survivor_configs, int* out_n_extracted, int max_survivors
) {
    return run_single_level_impl<4>(S, n_half, m, c_target,
        out_n_fp32_skipped, out_n_pruned_asym, out_n_pruned_test, out_n_survivors,
        out_min_test_val, out_min_test_config,
        out_survivor_configs, out_n_extracted, max_survivors);
}

static int run_single_level_d6(
    int S, int n_half, int m, double c_target,
    long long* out_n_fp32_skipped, long long* out_n_pruned_asym,
    long long* out_n_pruned_test, long long* out_n_survivors,
    double* out_min_test_val, int* out_min_test_config
) {
    return run_single_level_impl<6>(S, n_half, m, c_target,
        out_n_fp32_skipped, out_n_pruned_asym, out_n_pruned_test, out_n_survivors,
        out_min_test_val, out_min_test_config, NULL, NULL, 0);
}

static int run_single_level_d6_extract(
    int S, int n_half, int m, double c_target,
    long long* out_n_fp32_skipped, long long* out_n_pruned_asym,
    long long* out_n_pruned_test, long long* out_n_survivors,
    double* out_min_test_val, int* out_min_test_config,
    int* out_survivor_configs, int* out_n_extracted, int max_survivors
) {
    return run_single_level_impl<6>(S, n_half, m, c_target,
        out_n_fp32_skipped, out_n_pruned_asym, out_n_pruned_test, out_n_survivors,
        out_min_test_val, out_min_test_config,
        out_survivor_configs, out_n_extracted, max_survivors);
}


/* ================================================================
 * Templated helper: chunked streaming extraction for D=4 or D=6
 *
 * Splits the composition space into work-range chunks. For each
 * chunk, launches the kernel on [chunk_start, chunk_end), copies
 * survivors to a host staging buffer, and appends to a binary file
 * on disk. This bounds GPU and host memory regardless of total
 * survivor count.
 *
 * Binary file format: packed int32[D] per survivor, no header.
 * Total file size = n_extracted * D * sizeof(int).
 * ================================================================ */
template <int D>
static int run_single_level_extract_streamed_impl(
    int S, int n_half, int m,
    double c_target,
    long long* out_n_fp32_skipped,
    long long* out_n_pruned_asym,
    long long* out_n_pruned_test,
    long long* out_n_survivors,
    double* out_min_test_val,
    int* out_min_test_config,
    const char* survivor_file_path,
    long long* out_n_extracted,
    long long target_survivors
) {
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

    long long* h_prefix = (long long*)malloc((n_c0 + 1) * sizeof(long long));
    h_prefix[0] = 0;
    for (int i = 0; i < n_c0; i++) {
        int c0 = h_c0_order[i];
        long long R = (long long)(S - 2 * c0);
        h_prefix[i + 1] = h_prefix[i] + count_per_c0<D>(R);
    }
    long long total_work = h_prefix[n_c0];

    /* Upload constant data to GPU */
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

    /* Per-block output buffers (reused across chunks) */
    long long* d_block_counts = NULL;
    double* d_block_min_tv = NULL;
    int* d_block_min_configs = NULL;
    CUDA_CHECK(cudaMalloc(&d_block_counts, (long long)grid_size * 4 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_block_min_tv, grid_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_min_configs, grid_size * D * sizeof(int)));

    /* Determine survivor buffer size from available GPU memory */
    size_t gpu_free = 0, gpu_total = 0;
    cudaMemGetInfo(&gpu_free, &gpu_total);
    long long per_survivor_bytes = (long long)D * (long long)sizeof(int);
    /* Use 50% of remaining free GPU for survivor buffer.
     * Cap at 1.5B survivors (36 GB for D=6) — fits A100 80GB with headroom.
     * int32 atomicAdd limits absolute max to 2^31-1 ≈ 2.1B. */
    long long max_buf_survivors = (long long)(gpu_free * 0.5) / per_survivor_bytes;
    if (max_buf_survivors > 1500000000LL) max_buf_survivors = 1500000000LL;
    if (max_buf_survivors < 1000000LL) max_buf_survivors = 1000000LL;

    int* d_survivor_buf = NULL;
    int* d_survivor_count = NULL;
    CUDA_CHECK(cudaMalloc(&d_survivor_buf,
        (size_t)(max_buf_survivors * per_survivor_bytes)));
    CUDA_CHECK(cudaMalloc(&d_survivor_count, sizeof(int)));

    /* Host staging buffer: 100M survivors or match GPU buffer, whichever smaller.
     * 100M * 24 bytes = 2.4 GB — fits easily in 117 GB host RAM. */
    long long host_staging_cap = max_buf_survivors;
    if (host_staging_cap > 100000000LL) host_staging_cap = 100000000LL;
    int* h_staging = (int*)malloc((size_t)(host_staging_cap * per_survivor_bytes));
    if (!h_staging) {
        fprintf(stderr, "STREAMED: failed to allocate host staging buffer "
                "(%lld MB)\n",
                host_staging_cap * per_survivor_bytes / (1024 * 1024));
        fflush(stderr);
        host_staging_cap = 10000000LL;
        h_staging = (int*)malloc((size_t)(host_staging_cap * per_survivor_bytes));
        if (!h_staging) {
            fprintf(stderr, "STREAMED: even 10M staging buffer failed, aborting\n");
            fflush(stderr);
            cudaFree(d_prefix); cudaFree(d_c0_order);
            cudaFree(d_block_counts); cudaFree(d_block_min_tv);
            cudaFree(d_block_min_configs);
            cudaFree(d_survivor_buf); cudaFree(d_survivor_count);
            free(h_c0_order); free(h_prefix);
            return -1;
        }
    }

    /* Host-side block reduction buffers (reused across chunks) */
    long long* h_counts = (long long*)malloc(
        (size_t)grid_size * 4 * sizeof(long long));
    double* h_min_tvs = (double*)malloc((size_t)grid_size * sizeof(double));
    int* h_min_cfgs = (int*)malloc((size_t)grid_size * D * sizeof(int));

    /* Open output file */
    FILE* outfile = fopen(survivor_file_path, "wb");
    if (!outfile) {
        fprintf(stderr, "STREAMED: failed to open output file: %s\n",
                survivor_file_path);
        fflush(stderr);
        cudaFree(d_prefix); cudaFree(d_c0_order);
        cudaFree(d_block_counts); cudaFree(d_block_min_tv);
        cudaFree(d_block_min_configs);
        cudaFree(d_survivor_buf); cudaFree(d_survivor_count);
        free(h_c0_order); free(h_prefix); free(h_staging);
        free(h_counts); free(h_min_tvs); free(h_min_cfgs);
        return -1;
    }

    fprintf(stderr, "STREAMED: D=%d, S=%d, n_half=%d, m=%d, c_target=%.4f\n",
            D, S, n_half, m, c_target);
    fprintf(stderr, "STREAMED: total_work=%lld, GPU free=%.1f GB\n",
            total_work, gpu_free / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "STREAMED: GPU survivor buffer: %lld survivors (%.1f GB)\n",
            max_buf_survivors,
            max_buf_survivors * per_survivor_bytes / (1024.0*1024.0*1024.0));
    fprintf(stderr, "STREAMED: host staging: %lld survivors (%.1f MB)\n",
            host_staging_cap,
            host_staging_cap * per_survivor_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "STREAMED: output file: %s\n", survivor_file_path);
    fflush(stderr);

    bool use_int64 = ((long long)S * S > 2000000000LL);

    /* Chunking: start with 4 equal chunks (GPU buffer is large on A100),
     * adapt after first chunk based on observed survivor rate. */
    long long chunk_size = total_work / 4;
    if (chunk_size < 1000000LL) chunk_size = total_work;
    if (chunk_size > total_work) chunk_size = total_work;

    long long total_fp32 = 0, total_asym = 0, total_test = 0, total_surv = 0;
    long long total_extracted = 0;
    double min_tv = 1e30;
    int min_cfg[D];
    for (int i = 0; i < D; i++) min_cfg[i] = 0;
    long long total_disk_bytes = 0;
    double max_observed_rate = 0.0;
    int chunk_idx = 0;
    long long chunk_start = 0;

    while (chunk_start < total_work) {
        long long chunk_end = chunk_start + chunk_size;
        if (chunk_end > total_work) chunk_end = total_work;

        CUDA_CHECK(cudaMemset(d_survivor_count, 0, sizeof(int)));

        cudaEvent_t ev_start, ev_stop;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_stop);
        cudaEventRecord(ev_start);

        if (use_int64) {
            fused_prove_target<D, true><<<grid_size, block_size>>>(
                S, n_half, m, margin, c_target, thresh,
                inv_m_f, thresh_f, margin_f, asym_limit_f,
                d_prefix, d_c0_order, n_c0, chunk_start, chunk_end,
                d_block_counts, d_block_min_tv, d_block_min_configs,
                d_survivor_buf, d_survivor_count,
                (int)max_buf_survivors);
        } else {
            fused_prove_target<D, false><<<grid_size, block_size>>>(
                S, n_half, m, margin, c_target, thresh,
                inv_m_f, thresh_f, margin_f, asym_limit_f,
                d_prefix, d_c0_order, n_c0, chunk_start, chunk_end,
                d_block_counts, d_block_min_tv, d_block_min_configs,
                d_survivor_buf, d_survivor_count,
                (int)max_buf_survivors);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        float kernel_ms = 0;
        cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);

        int h_surv_count = 0;
        CUDA_CHECK(cudaMemcpy(&h_surv_count, d_survivor_count,
            sizeof(int), cudaMemcpyDeviceToHost));

        /* Buffer overflow — update rate estimate and resize chunk */
        if (h_surv_count > (int)max_buf_survivors) {
            double overflow_rate = (double)h_surv_count / (double)(chunk_end - chunk_start);
            if (overflow_rate > max_observed_rate) max_observed_rate = overflow_rate;
            long long target = (long long)(max_buf_survivors * 0.5);
            chunk_size = (long long)(target / max_observed_rate);
            fprintf(stderr, "STREAMED: chunk %d OVERFLOW: %d > %lld buf. "
                    "rate=%.6f new_chunk=%lld\n", chunk_idx, h_surv_count,
                    max_buf_survivors, overflow_rate, chunk_size);
            fflush(stderr);
            if (chunk_size < 1) {
                fprintf(stderr, "STREAMED: cannot subdivide further.\n");
                fflush(stderr);
                fclose(outfile);
                cudaFree(d_prefix); cudaFree(d_c0_order);
                cudaFree(d_block_counts); cudaFree(d_block_min_tv);
                cudaFree(d_block_min_configs);
                cudaFree(d_survivor_buf); cudaFree(d_survivor_count);
                free(h_c0_order); free(h_prefix); free(h_staging);
                free(h_counts); free(h_min_tvs); free(h_min_cfgs);
                return -3;
            }
            continue;
        }

        /* Aggregate per-block counts */
        CUDA_CHECK(cudaMemcpy(h_counts, d_block_counts,
            grid_size * 4 * sizeof(long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_min_tvs, d_block_min_tv,
            grid_size * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_min_cfgs, d_block_min_configs,
            grid_size * D * sizeof(int), cudaMemcpyDeviceToHost));

        long long chunk_fp32 = 0, chunk_asym = 0, chunk_test = 0, chunk_surv = 0;
        for (int b = 0; b < grid_size; b++) {
            chunk_fp32 += h_counts[b * 4 + 0];
            chunk_asym += h_counts[b * 4 + 1];
            chunk_test += h_counts[b * 4 + 2];
            chunk_surv += h_counts[b * 4 + 3];
            if (h_min_tvs[b] < min_tv) {
                min_tv = h_min_tvs[b];
                for (int i = 0; i < D; i++)
                    min_cfg[i] = h_min_cfgs[b * D + i];
            }
        }
        total_fp32 += chunk_fp32;
        total_asym += chunk_asym;
        total_test += chunk_test;
        total_surv += chunk_surv;

        /* D->H copy + disk write in batches */
        long long to_write = h_surv_count;
        long long written = 0;
        int disk_error = 0;
        while (written < to_write) {
            long long batch = to_write - written;
            if (batch > host_staging_cap) batch = host_staging_cap;
            CUDA_CHECK(cudaMemcpy(h_staging,
                d_survivor_buf + written * D,
                (size_t)(batch * per_survivor_bytes),
                cudaMemcpyDeviceToHost));
            size_t bw = fwrite(h_staging, 1,
                (size_t)(batch * per_survivor_bytes), outfile);
            if ((long long)bw != batch * per_survivor_bytes) {
                fprintf(stderr, "STREAMED: disk write FAILED chunk %d: "
                        "wrote %zu/%lld bytes. Aborting extraction.\n",
                        chunk_idx, bw, batch * per_survivor_bytes);
                fflush(stderr);
                fflush(outfile);
                total_disk_bytes += (long long)bw;
                total_extracted += written + (long long)(bw / per_survivor_bytes);
                disk_error = 1;
                break;
            }
            total_disk_bytes += (long long)bw;
            written += batch;
        }
        if (disk_error) {
            /* Set outputs with partial data, cleanup, return -4 */
            *out_n_fp32_skipped = total_fp32;
            *out_n_pruned_asym = total_asym;
            *out_n_pruned_test = total_test;
            *out_n_survivors = total_surv;
            *out_min_test_val = min_tv;
            for (int i = 0; i < D; i++) out_min_test_config[i] = min_cfg[i];
            *out_n_extracted = total_extracted;
            fclose(outfile);
            cudaFree(d_prefix); cudaFree(d_c0_order);
            cudaFree(d_block_counts); cudaFree(d_block_min_tv);
            cudaFree(d_block_min_configs);
            cudaFree(d_survivor_buf); cudaFree(d_survivor_count);
            free(h_c0_order); free(h_prefix); free(h_staging);
            free(h_counts); free(h_min_tvs); free(h_min_cfgs);
            return -4;
        }
        total_extracted += to_write;

        double buf_util = (max_buf_survivors > 0)
            ? 100.0 * to_write / max_buf_survivors : 0.0;
        fprintf(stderr, "STREAMED: chunk %d [%lld,%lld) %.1fs surv=%d "
                "cum=%lld util=%.0f%% disk=%.1fMB\n",
                chunk_idx, chunk_start, chunk_end, kernel_ms / 1000.0,
                h_surv_count, total_extracted, buf_util,
                total_disk_bytes / (1024.0 * 1024.0));
        fflush(stderr);

        /* Early termination if we've reached the target survivor count */
        if (target_survivors > 0 && total_extracted >= target_survivors) {
            fprintf(stderr, "STREAMED: target_survivors=%lld reached (extracted=%lld), "
                    "stopping early\n", target_survivors, total_extracted);
            fflush(stderr);
            chunk_start = total_work;  /* exit the while loop */
            chunk_idx++;
            break;
        }

        /* Continuous adaptive sizing: track rolling max rate, re-adapt every chunk */
        if (to_write > 0) {
            double this_rate = (double)to_write / (double)(chunk_end - chunk_start);
            if (this_rate > max_observed_rate) max_observed_rate = this_rate;
        }
        if (max_observed_rate > 0) {
            long long target = (long long)(max_buf_survivors * 0.5);
            long long new_cs = (long long)(target / max_observed_rate);
            if (new_cs < 100000LL) new_cs = 100000LL;
            if (new_cs > total_work) new_cs = total_work;
            if (chunk_idx == 0) {
                fprintf(stderr, "STREAMED: adaptive: rate=%.6f new_chunk=%lld\n",
                        max_observed_rate, new_cs);
                fflush(stderr);
            }
            chunk_size = new_cs;
        } else if (chunk_idx == 0) {
            chunk_size = chunk_end - chunk_start;
            if (chunk_size < total_work / 4) chunk_size = total_work / 4;
        }

        chunk_start = chunk_end;
        chunk_idx++;
    }

    fclose(outfile);
    fprintf(stderr, "STREAMED: DONE %d chunks, %lld survivors, %.1f MB\n",
            chunk_idx, total_extracted,
            total_disk_bytes / (1024.0 * 1024.0));
    fflush(stderr);

    *out_n_fp32_skipped = total_fp32;
    *out_n_pruned_asym = total_asym;
    *out_n_pruned_test = total_test;
    *out_n_survivors = total_surv;
    *out_min_test_val = min_tv;
    for (int i = 0; i < D; i++) out_min_test_config[i] = min_cfg[i];
    *out_n_extracted = total_extracted;

    cudaFree(d_prefix);
    cudaFree(d_c0_order);
    cudaFree(d_block_counts);
    cudaFree(d_block_min_tv);
    cudaFree(d_block_min_configs);
    cudaFree(d_survivor_buf);
    cudaFree(d_survivor_count);
    free(h_c0_order);
    free(h_prefix);
    free(h_staging);
    free(h_counts);
    free(h_min_tvs);
    free(h_min_cfgs);

    return 0;
}
