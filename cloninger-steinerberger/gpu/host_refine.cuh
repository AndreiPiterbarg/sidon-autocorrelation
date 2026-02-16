#pragma once
/*
 * Host orchestration for hierarchical refinement.
 *
 * Batched approach: groups many small parents into a single kernel
 * launch using prefix sums + binary_search_le (same pattern as
 * the Level 0 kernel in phase1_kernels.cuh).  This eliminates
 * per-parent cudaMemcpy + cudaDeviceSynchronize overhead.
 *
 * For very large parents (single parent N > REFINE_CHUNK_SIZE),
 * falls back to the existing chunked single-parent approach.
 *
 * Template-instantiated for D_CHILD = 12, 24, 48.
 *
 * Depends on: device_helpers.cuh (CUDA_CHECK)
 *             refinement_kernel.cuh (refine_prove_target, refine_prove_target_batched)
 */

#include <algorithm>
#include <ctime>
#include <cstring>

/* Chunk size per kernel launch (~2B — most parents fit in one launch) */
#define REFINE_CHUNK_SIZE 2000000000LL

/* Default max survivors across all parents */
#define REFINE_MAX_SURVIVORS 10000000


/* ================================================================
 * Helper: compute total refinement count for a parent
 * ================================================================ */
static long long compute_refinement_count(const int* parent_B, int d_parent) {
    long long N = 1;
    for (int i = 0; i < d_parent; i++) {
        N *= (2 * (long long)parent_B[i] + 1);
        if (N < 0) return -1;  /* overflow */
    }
    return N;
}


/* ================================================================
 * Helper: reduce per-block GPU outputs into running totals
 * ================================================================ */
template <int D_CHILD>
static void reduce_block_results(
    const long long* h_counts, const double* h_min_tvs, const int* h_min_cfgs,
    int grid_size,
    long long& total_asym, long long& total_test, long long& total_surv,
    double& global_min_tv, int* global_min_cfg)
{
    for (int b = 0; b < grid_size; b++) {
        total_asym += h_counts[b * 3 + 0];
        total_test += h_counts[b * 3 + 1];
        total_surv += h_counts[b * 3 + 2];
        if (h_min_tvs[b] < global_min_tv) {
            global_min_tv = h_min_tvs[b];
            for (int i = 0; i < D_CHILD; i++)
                global_min_cfg[i] = h_min_cfgs[b * D_CHILD + i];
        }
    }
}


/* ================================================================
 * Host implementation: refine parents (templated on D_CHILD)
 *
 * parent_configs: [num_parents * d_parent] flattened array of parent bin values
 * Returns 0 on success, 1 on timeout, -1 on error.
 * ================================================================ */
template <int D_CHILD>
static int refine_parents_impl(
    const int* parent_configs,
    int num_parents,
    int d_parent,
    int m,
    double c_target,
    /* Outputs */
    long long* out_total_asym,
    long long* out_total_test,
    long long* out_total_survivors,
    double* out_min_test_val,
    int* out_min_test_config,    /* [D_CHILD] */
    int* out_survivor_configs,   /* [max_survivors * D_CHILD] or NULL */
    int* out_n_extracted,
    int max_survivors,
    double time_budget_sec       /* 0 = no limit */
) {
    if (d_parent != D_CHILD / 2) {
        fprintf(stderr, "refine_parents: d_parent=%d but D_CHILD=%d (expected d_parent=%d)\n",
                d_parent, D_CHILD, D_CHILD / 2);
        return -1;
    }

    /* Compute child-level parameters */
    int S_parent = 0;
    for (int i = 0; i < d_parent; i++) S_parent += parent_configs[i]; /* use first parent to get S */
    int S_child = 2 * S_parent;
    int n_half_parent = d_parent / 2;
    int n_half_child = 2 * n_half_parent;

    double corr = 2.0 / m + 1.0 / ((double)m * m);
    double prune_target = c_target + corr;
    double fp_margin = 1e-9;
    double thresh = prune_target + fp_margin;
    double margin = 1.0 / (4.0 * m);
    float inv_m_f = 1.0f / (float)m;
    float thresh_f = (float)thresh * (1.0f + 1e-5f);
    float margin_f = (float)margin;
    float asym_limit_f = (float)c_target * (1.0f + 1e-5f);

    /* Precompute per-ell integer thresholds for child level */
    long long h_int_thresh[D_CHILD - 1];
    for (int ell = 2; ell <= D_CHILD; ell++) {
        double x = thresh * (double)m * (double)m * 4.0 * (double)n_half_child * (double)ell;
        h_int_thresh[ell - 2] = (long long)(x * (1.0 - 4.0 * DBL_EPSILON));
    }

    /* Sort parents by refinement count ascending — O(n log n) */
    long long* ref_counts = (long long*)malloc(num_parents * sizeof(long long));
    int* parent_order = (int*)malloc(num_parents * sizeof(int));
    for (int p = 0; p < num_parents; p++) {
        ref_counts[p] = compute_refinement_count(parent_configs + p * d_parent, d_parent);
        parent_order[p] = p;
    }
    std::sort(parent_order, parent_order + num_parents,
        [ref_counts](int a, int b) { return ref_counts[a] < ref_counts[b]; });

    /* ============================================================
     * GPU buffer allocation
     * ============================================================ */
    constexpr int block_size = (D_CHILD <= 12) ? REFINE_BLOCK_SIZE_12 :
                               (D_CHILD <= 24) ? REFINE_BLOCK_SIZE_24 :
                                                  REFINE_BLOCK_SIZE_48;
    int max_grid_size = 108 * 32;  /* A100: 108 SMs * 32 blocks/SM */

    /* Integer thresholds (shared by all kernels) */
    long long* d_int_thresh = NULL;
    CUDA_CHECK(cudaMalloc(&d_int_thresh, (D_CHILD - 1) * sizeof(long long)));
    CUDA_CHECK(cudaMemcpy(d_int_thresh, h_int_thresh,
        (D_CHILD - 1) * sizeof(long long), cudaMemcpyHostToDevice));

    /* Per-block output buffers (reused across all launches) */
    long long* d_block_counts = NULL;
    double* d_block_min_tv = NULL;
    int* d_block_min_configs = NULL;
    CUDA_CHECK(cudaMalloc(&d_block_counts, (long long)max_grid_size * 3 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_block_min_tv, max_grid_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_min_configs, (long long)max_grid_size * D_CHILD * sizeof(int)));

    /* Batched kernel buffers: parent configs + prefix sums */
    int max_batch_parents = MAX_BATCH_PARENTS;
    int* d_batch_parents = NULL;
    long long* d_batch_prefix = NULL;
    CUDA_CHECK(cudaMalloc(&d_batch_parents,
        (long long)max_batch_parents * d_parent * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_batch_prefix,
        ((long long)max_batch_parents + 1) * sizeof(long long)));

    /* Host staging buffers for building batches */
    int* h_batch_parents = (int*)malloc(
        (long long)max_batch_parents * d_parent * sizeof(int));
    long long* h_batch_prefix = (long long*)malloc(
        ((long long)max_batch_parents + 1) * sizeof(long long));

    /* Single-parent buffers (for large-parent fallback) */
    int* d_parent_B = NULL;
    int* d_splits = NULL;
    CUDA_CHECK(cudaMalloc(&d_parent_B, d_parent * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_splits, d_parent * sizeof(int)));

    /* Dynamic survivor buffer: query actual remaining GPU memory */
    int* d_survivor_buf = NULL;
    int* d_survivor_count = NULL;
    int do_extract = (out_survivor_configs != NULL && max_survivors > 0);
    if (do_extract) {
        size_t gpu_free = 0, gpu_total = 0;
        cudaMemGetInfo(&gpu_free, &gpu_total);
        long long bytes_per_surv = (long long)D_CHILD * sizeof(int);
        /* Use 90% of remaining GPU memory for survivor buffer */
        long long max_from_gpu = (long long)((double)gpu_free * 0.9) / bytes_per_surv;
        if (max_from_gpu < (long long)max_survivors) {
            fprintf(stderr, "refine_parents: GPU memory limits survivors to %lld "
                    "(requested %d, %.1f GB free)\n",
                    max_from_gpu, max_survivors,
                    (double)gpu_free / (1024.0 * 1024.0 * 1024.0));
            if (max_from_gpu < 1) max_from_gpu = 1;
            max_survivors = (int)max_from_gpu;
        }
        CUDA_CHECK(cudaMalloc(&d_survivor_buf, (long long)max_survivors * D_CHILD * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_survivor_count, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_survivor_count, 0, sizeof(int)));
    }

    /* Pre-allocate host reduction buffers (reused across all chunks) */
    long long* h_counts = (long long*)malloc((long long)max_grid_size * 3 * sizeof(long long));
    double* h_min_tvs = (double*)malloc(max_grid_size * sizeof(double));
    int* h_min_cfgs = (int*)malloc((long long)max_grid_size * D_CHILD * sizeof(int));

    /* Aggregate results */
    long long total_asym = 0, total_test = 0, total_surv = 0;
    double global_min_tv = 1e30;
    int global_min_cfg[D_CHILD];
    for (int i = 0; i < D_CHILD; i++) global_min_cfg[i] = 0;

    clock_t start_time = clock();
    clock_t last_log_time = start_time;
    long long cumulative_refs = 0;
    int timed_out = 0;

    /* ============================================================
     * Main processing loop: batched for small parents,
     * chunked fallback for large parents
     * ============================================================ */
    int p_idx = 0;
    while (p_idx < num_parents && !timed_out) {
        int p = parent_order[p_idx];
        long long N = ref_counts[p];

        if (N > REFINE_CHUNK_SIZE) {
            /* ====================================================
             * LARGE PARENT: chunked single-parent approach
             * (one parent at a time, split into REFINE_CHUNK_SIZE
             *  chunks with separate kernel launches)
             * ==================================================== */
            const int* pB = parent_configs + p * d_parent;

            /* Verify parent sum */
            int this_S = 0;
            for (int i = 0; i < d_parent; i++) this_S += pB[i];
            if (this_S != S_parent) {
                fprintf(stderr, "refine_parents: parent %d sum=%d != expected %d\n",
                        p, this_S, S_parent);
                p_idx++;
                continue;
            }

            int h_splits[MAX_D_CHILD / 2];
            for (int i = 0; i < d_parent; i++)
                h_splits[i] = 2 * pB[i] + 1;

            CUDA_CHECK(cudaMemcpy(d_parent_B, pB,
                d_parent * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_splits, h_splits,
                d_parent * sizeof(int), cudaMemcpyHostToDevice));

            long long processed = 0;
            while (processed < N) {
                long long chunk = N - processed;
                if (chunk > REFINE_CHUNK_SIZE) chunk = REFINE_CHUNK_SIZE;

                int grid_size = (int)fmin(
                    (double)(chunk + block_size - 1) / block_size, (double)max_grid_size);
                if (grid_size < 1) grid_size = 1;

                refine_prove_target<D_CHILD><<<grid_size, block_size>>>(
                    d_parent_B, d_splits, d_parent,
                    S_child, n_half_child, m,
                    c_target, thresh,
                    inv_m_f, thresh_f, margin_f, asym_limit_f,
                    d_int_thresh,
                    processed, chunk,
                    d_block_counts, d_block_min_tv, d_block_min_configs,
                    do_extract ? d_survivor_buf : NULL,
                    do_extract ? d_survivor_count : NULL,
                    do_extract ? max_survivors : 0);

                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());

                CUDA_CHECK(cudaMemcpy(h_counts, d_block_counts,
                    grid_size * 3 * sizeof(long long), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_min_tvs, d_block_min_tv,
                    grid_size * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_min_cfgs, d_block_min_configs,
                    grid_size * D_CHILD * sizeof(int), cudaMemcpyDeviceToHost));

                reduce_block_results<D_CHILD>(h_counts, h_min_tvs, h_min_cfgs,
                    grid_size, total_asym, total_test, total_surv,
                    global_min_tv, global_min_cfg);

                processed += chunk;

                if (time_budget_sec > 0 && processed < N) {
                    double elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
                    if (elapsed > time_budget_sec) {
                        timed_out = 1;
                        break;
                    }
                }
            }

            cumulative_refs += N;
            p_idx++;

        } else {
            /* ====================================================
             * SMALL PARENTS: batched approach
             * Group consecutive sorted parents into one kernel launch.
             * Binary search in kernel maps each thread to its parent.
             * ==================================================== */
            int batch_count = 0;
            long long batch_work = 0;
            h_batch_prefix[0] = 0;

            while (p_idx + batch_count < num_parents &&
                   batch_count < max_batch_parents) {
                int pp = parent_order[p_idx + batch_count];
                long long nn = ref_counts[pp];
                if (nn > REFINE_CHUNK_SIZE) break;  /* hit a large parent */
                if (batch_work + nn > REFINE_CHUNK_SIZE) break;  /* batch full */

                /* Copy parent config into batch staging buffer */
                memcpy(h_batch_parents + (long long)batch_count * d_parent,
                       parent_configs + pp * d_parent,
                       d_parent * sizeof(int));

                batch_work += nn;
                batch_count++;
                h_batch_prefix[batch_count] = batch_work;
            }

            if (batch_count == 0) {
                /* Edge case: single parent that somehow wasn't caught above */
                p_idx++;
                continue;
            }

            /* Upload batch data to GPU */
            CUDA_CHECK(cudaMemcpy(d_batch_parents, h_batch_parents,
                (long long)batch_count * d_parent * sizeof(int),
                cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_batch_prefix, h_batch_prefix,
                (batch_count + 1) * sizeof(long long),
                cudaMemcpyHostToDevice));

            /* Launch batched kernel */
            int grid_size = (int)fmin(
                (double)(batch_work + block_size - 1) / block_size,
                (double)max_grid_size);
            if (grid_size < 1) grid_size = 1;

            refine_prove_target_batched<D_CHILD><<<grid_size, block_size>>>(
                d_batch_parents, d_batch_prefix, batch_count,
                S_child, n_half_child, m,
                c_target, thresh,
                inv_m_f, thresh_f, margin_f, asym_limit_f,
                d_int_thresh, batch_work,
                d_block_counts, d_block_min_tv, d_block_min_configs,
                do_extract ? d_survivor_buf : NULL,
                do_extract ? d_survivor_count : NULL,
                do_extract ? max_survivors : 0);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            /* Reduce per-block results */
            CUDA_CHECK(cudaMemcpy(h_counts, d_block_counts,
                grid_size * 3 * sizeof(long long), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_min_tvs, d_block_min_tv,
                grid_size * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_min_cfgs, d_block_min_configs,
                grid_size * D_CHILD * sizeof(int), cudaMemcpyDeviceToHost));

            reduce_block_results<D_CHILD>(h_counts, h_min_tvs, h_min_cfgs,
                grid_size, total_asym, total_test, total_surv,
                global_min_tv, global_min_cfg);

            cumulative_refs += batch_work;
            p_idx += batch_count;
        }

        if (timed_out) break;

        /* Progress logging: every 10 seconds or on last parent */
        double elapsed_s = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        double since_log = (double)(clock() - last_log_time) / CLOCKS_PER_SEC;
        if (since_log >= 10.0 || p_idx >= num_parents) {
            double rate = (elapsed_s > 0) ? (double)cumulative_refs / elapsed_s : 0;
            fprintf(stderr, "refine: %d/%d parents | %.2e refs | %lld surv | "
                    "%.2e refs/s | %.0fs\n",
                    p_idx, num_parents, (double)cumulative_refs,
                    (long long)total_surv, rate, elapsed_s);
            fflush(stderr);
            last_log_time = clock();
        }

        /* Check time budget */
        if (time_budget_sec > 0) {
            double elapsed_s2 = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            if (elapsed_s2 > time_budget_sec) {
                fprintf(stderr, "refine_parents: time budget exceeded after %d/%d parents\n",
                        p_idx, num_parents);
                timed_out = 1;
            }
        }
    }

    /* ============================================================
     * Read survivor extraction results
     * ============================================================ */
    int n_ext = 0;
    if (do_extract) {
        int h_surv_count = 0;
        CUDA_CHECK(cudaMemcpy(&h_surv_count, d_survivor_count, sizeof(int), cudaMemcpyDeviceToHost));
        n_ext = h_surv_count;
        if (n_ext > max_survivors) n_ext = max_survivors;
        if (n_ext > 0) {
            CUDA_CHECK(cudaMemcpy(out_survivor_configs, d_survivor_buf,
                (long long)n_ext * D_CHILD * sizeof(int), cudaMemcpyDeviceToHost));
        }
    }

    /* Write outputs */
    *out_total_asym = total_asym;
    *out_total_test = total_test;
    *out_total_survivors = total_surv;
    *out_min_test_val = global_min_tv;
    for (int i = 0; i < D_CHILD; i++) out_min_test_config[i] = global_min_cfg[i];
    if (out_n_extracted) *out_n_extracted = n_ext;

    /* Cleanup */
    cudaFree(d_int_thresh);
    cudaFree(d_block_counts);
    cudaFree(d_block_min_tv);
    cudaFree(d_block_min_configs);
    cudaFree(d_batch_parents);
    cudaFree(d_batch_prefix);
    cudaFree(d_parent_B);
    cudaFree(d_splits);
    if (d_survivor_buf) cudaFree(d_survivor_buf);
    if (d_survivor_count) cudaFree(d_survivor_count);
    free(h_batch_parents);
    free(h_batch_prefix);
    free(h_counts);
    free(h_min_tvs);
    free(h_min_cfgs);
    free(ref_counts);
    free(parent_order);

    return timed_out ? 1 : 0;  /* 0 = complete, 1 = timed out */
}


/* ================================================================
 * Concrete instantiations
 * ================================================================ */
static int refine_parents_d12(
    const int* parent_configs, int num_parents, int d_parent, int m, double c_target,
    long long* out_asym, long long* out_test, long long* out_surv,
    double* out_min_tv, int* out_min_cfg,
    int* out_surv_configs, int* out_n_ext, int max_surv, double time_budget)
{
    return refine_parents_impl<12>(parent_configs, num_parents, d_parent, m, c_target,
        out_asym, out_test, out_surv, out_min_tv, out_min_cfg,
        out_surv_configs, out_n_ext, max_surv, time_budget);
}

static int refine_parents_d24(
    const int* parent_configs, int num_parents, int d_parent, int m, double c_target,
    long long* out_asym, long long* out_test, long long* out_surv,
    double* out_min_tv, int* out_min_cfg,
    int* out_surv_configs, int* out_n_ext, int max_surv, double time_budget)
{
    return refine_parents_impl<24>(parent_configs, num_parents, d_parent, m, c_target,
        out_asym, out_test, out_surv, out_min_tv, out_min_cfg,
        out_surv_configs, out_n_ext, max_surv, time_budget);
}

static int refine_parents_d48(
    const int* parent_configs, int num_parents, int d_parent, int m, double c_target,
    long long* out_asym, long long* out_test, long long* out_surv,
    double* out_min_tv, int* out_min_cfg,
    int* out_surv_configs, int* out_n_ext, int max_surv, double time_budget)
{
    return refine_parents_impl<48>(parent_configs, num_parents, d_parent, m, c_target,
        out_asym, out_test, out_surv, out_min_tv, out_min_cfg,
        out_surv_configs, out_n_ext, max_surv, time_budget);
}
