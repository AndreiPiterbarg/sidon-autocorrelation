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
 * Template-instantiated for D_CHILD = 8, 12, 16, 24, 32, 48.
 *
 * Depends on: device_helpers.cuh (CUDA_CHECK)
 *             refinement_kernel.cuh (refine_prove_target, refine_prove_target_batched)
 */

#include <algorithm>
#include <ctime>
#include <cstring>
#include <cub/cub.cuh>

/* Chunk size per kernel launch (~2B — most parents fit in one launch) */
#define REFINE_CHUNK_SIZE 2000000000LL

/* Default max survivors across all parents */
#define REFINE_MAX_SURVIVORS 10000000


/* ================================================================
 * Helper: compute total refinement count for a parent
 * ================================================================ */
/* With S=m convention: c[2i]+c[2i+1]=B[i], split count = B[i]+1.
 * x_cap limits each sub-bin to <= x_cap (ell=2 energy cap).
 * If x_cap < 0, no cap is applied. */
static long long compute_refinement_count(const int* parent_B, int d_parent, int x_cap) {
    long long N = 1;
    for (int i = 0; i < d_parent; i++) {
        int eff_count;
        if (x_cap >= 0 && parent_B[i] > 0) {
            /* c_even in [max(0, B[i]-x_cap), min(B[i], x_cap)] */
            int lo = (parent_B[i] > x_cap) ? (parent_B[i] - x_cap) : 0;
            int hi = (parent_B[i] < x_cap) ? parent_B[i] : x_cap;
            eff_count = hi - lo + 1;
            if (eff_count <= 0) return 0;  /* parent fully prunable */
        } else {
            eff_count = parent_B[i] + 1;
        }
        N *= (long long)eff_count;
        if (N < 0) return -1;  /* overflow */
    }
    return N;
}

/* Variant with per-parent lo values (supports frozen bins where lo=hi). */
static long long compute_refinement_count_with_lo(const int* parent_B, const int* lo_arr,
                                                   int d_parent, int x_cap) {
    long long N = 1;
    for (int i = 0; i < d_parent; i++) {
        int lo = lo_arr[i];
        int hi = (parent_B[i] < x_cap) ? parent_B[i] : x_cap;
        if (hi < lo) hi = lo;
        int eff_count = hi - lo + 1;
        if (eff_count <= 0) return 0;
        N *= (long long)eff_count;
        if (N < 0) return -1;
    }
    return N;
}

/* Compute default lo values for a parent (from x_cap). */
static void compute_default_lo(const int* parent_B, int d_parent, int x_cap, int* lo_out) {
    for (int i = 0; i < d_parent; i++) {
        lo_out[i] = (x_cap >= 0 && parent_B[i] > x_cap) ? (parent_B[i] - x_cap) : 0;
    }
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
 * CPU pre-filter: analyze parent freezability before GPU dispatch
 *
 * For each parent, builds the flat-split child, computes autoconvolution
 * and window scan, then checks if the universal multi-bin perturbation
 * bound is small enough to resolve the parent without GPU.
 *
 * Returns:
 *   0 = needs GPU (gap too small relative to perturbation, or
 *       all windows below threshold — need to enumerate actual survivors)
 *   1 = fully eliminated on CPU (all windows exceed threshold even
 *       with worst-case perturbation — all children provably pruned)
 * ================================================================ */
template <int D_CHILD>
static int analyze_parent_cpu(
    const int* parent_B,    /* [D_PARENT] parent bin values */
    int d_parent,
    int m,
    double c_target,
    int x_cap,
    /* Output: flat-split child config (valid when return != 0) */
    int* flat_child,        /* [D_CHILD] */
    double* flat_test_val   /* best test value of flat-split child */
) {
    constexpr int D_PARENT = D_CHILD / 2;
    constexpr int CONV_LEN = 2 * D_CHILD - 1;

    if (d_parent != D_PARENT) return 0;

    int S_child = 0;
    for (int i = 0; i < D_PARENT; i++) S_child += parent_B[i];
    int n_half_child = D_CHILD / 2;  /* n_half at child level = d_parent */

    /* Build flat-split child: c[2i] = B[i]/2, c[2i+1] = B[i] - B[i]/2 */
    int c[D_CHILD];
    for (int i = 0; i < D_PARENT; i++) {
        c[2*i]     = parent_B[i] / 2;
        c[2*i + 1] = parent_B[i] - parent_B[i] / 2;
    }
    for (int i = 0; i < D_CHILD; i++) flat_child[i] = c[i];

    /* Compute full autoconvolution of flat-split child */
    long long conv[CONV_LEN];
    for (int k = 0; k < CONV_LEN; k++) conv[k] = 0;
    for (int i = 0; i < D_CHILD; i++) {
        conv[2*i] += (long long)c[i] * c[i];
        for (int j = i+1; j < D_CHILD; j++)
            conv[i+j] += (long long)2 * c[i] * c[j];
    }

    /* Prefix sum of conv */
    for (int k = 1; k < CONV_LEN; k++) conv[k] += conv[k-1];

    /* Prefix sum of c for W_int computation */
    int prefix_c[D_CHILD + 1];
    prefix_c[0] = 0;
    for (int i = 0; i < D_CHILD; i++)
        prefix_c[i + 1] = prefix_c[i] + c[i];

    /* Dynamic threshold constants */
    double dyn_base = c_target * (double)m * (double)m + 1.0
                    + 1e-9 * (double)m * (double)m;
    double inv_4n = 1.0 / (4.0 * (double)n_half_child);

    /* Per-bin delta_max values for perturbation bound.
     * delta_max_a = max(B[a]/2 - lo_a, hi_a - B[a]/2) where
     * lo_a = max(0, B[a]-x_cap), hi_a = min(B[a], x_cap). */
    int delta_maxes[D_PARENT];
    for (int a = 0; a < D_PARENT; a++) {
        int lo_a = (parent_B[a] > x_cap) ? (parent_B[a] - x_cap) : 0;
        int hi_a = (parent_B[a] < x_cap) ? parent_B[a] : x_cap;
        int flat_val = parent_B[a] / 2;
        int delta_max = flat_val - lo_a;
        if (hi_a - flat_val > delta_max) delta_max = hi_a - flat_val;
        delta_maxes[a] = delta_max;
    }

    /* Cross-terms between pairs of bins (universal, kept as-is) */
    long long cross_pert = 0;
    for (int a = 0; a < D_PARENT; a++) {
        if (delta_maxes[a] <= 0) continue;
        for (int b = a + 1; b < D_PARENT; b++) {
            if (delta_maxes[b] <= 0) continue;
            cross_pert += 4 * (long long)delta_maxes[a] * (long long)delta_maxes[b];
        }
    }

    /* Window scan + per-window freeze check.
     * For each window, track best test value for reporting.
     * Also check: if ANY window has gap > per-window perturbation,
     * all inner configs will be pruned by that window → parent eliminated. */
    double best_tv = 0.0;
    int frozen = 0;

    for (int ell = 2; ell <= 2 * D_CHILD; ell++) {
        int n_cv = ell - 1;
        double dyn_base_ell = dyn_base * (double)ell * inv_4n;
        double two_ell_inv_4n = 2.0 * (double)ell * inv_4n;
        double inv_norm = (4.0 * n_half_child) / ((double)m * (double)m * (double)ell);
        int n_windows = CONV_LEN - n_cv + 1;

        for (int s_lo = 0; s_lo < n_windows; s_lo++) {
            int s_hi = s_lo + n_cv - 1;
            long long ws = conv[s_hi];
            if (s_lo > 0) ws -= conv[s_lo - 1];

            /* Track test value for reporting */
            double tv = (double)ws * inv_norm;
            if (tv > best_tv) best_tv = tv;

            if (frozen) continue;  /* already found a freezable window */

            /* Per-window dynamic threshold */
            int lo_bin = (s_lo > D_CHILD - 1) ? s_lo - (D_CHILD - 1) : 0;
            int hi_bin = (s_lo + ell - 2 < D_CHILD - 1) ? s_lo + ell - 2 : D_CHILD - 1;
            int W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin];
            double dyn_x = dyn_base_ell + two_ell_inv_4n * (double)W_int;
            long long dyn_it = (long long)(dyn_x * (1.0 - 4.0 * DBL_EPSILON));

            long long gap = ws - dyn_it;
            if (gap <= 0) continue;

            /* Window exceeds threshold — check if perturbation can close it */
            long long window_pert = 0;
            for (int a = 0; a < D_PARENT; a++) {
                int dm = delta_maxes[a];
                if (dm <= 0) continue;
                int u_lo = s_lo - 2*a;
                int u_hi = s_lo + ell - 2 - 2*a;
                int v_lo = s_lo - 2*a - 1;
                int v_hi = s_lo + ell - 3 - 2*a;
                int u_lo_c = (u_lo < 0) ? 0 : u_lo;
                int u_hi_c = (u_hi > D_CHILD - 1) ? D_CHILD - 1 : u_hi;
                long long sum_u = (u_hi_c >= u_lo_c) ?
                    (long long)(prefix_c[u_hi_c + 1] - prefix_c[u_lo_c]) : 0;
                int v_lo_c = (v_lo < 0) ? 0 : v_lo;
                int v_hi_c = (v_hi > D_CHILD - 1) ? D_CHILD - 1 : v_hi;
                long long sum_v = (v_hi_c >= v_lo_c) ?
                    (long long)(prefix_c[v_hi_c + 1] - prefix_c[v_lo_c]) : 0;
                long long p1_abs = 2 * ((sum_u >= sum_v) ?
                    (sum_u - sum_v) : (sum_v - sum_u));
                window_pert += p1_abs * (long long)dm
                             + (long long)2 * (long long)dm * (long long)dm;
                window_pert += (long long)dm;
            }
            window_pert += cross_pert;
            if (gap > window_pert) frozen = 1;
        }
    }

    *flat_test_val = best_tv;

    if (frozen) {
        return 1;  /* fully eliminated */
    }

    return 0;  /* needs GPU */
}


/* ================================================================
 * Helper: compute bin ordering for freeze kernel.
 * Outer bins (first K_OUTER) = largest B[i] values.
 * Inner bins (remaining) = smallest B[i] values (most likely to freeze).
 * Returns the bin_order array and K_OUTER.
 * ================================================================ */
template <int D_CHILD>
static int compute_freeze_bin_order(
    const int* parent_B, int d_parent, int x_cap,
    int* bin_order  /* [d_parent] output: bin indices sorted outer-first */
) {
    constexpr int D_PARENT = D_CHILD / 2;

    /* Sort bin indices by effective split count descending.
     * Outer = largest effective split count (most children, hardest to freeze).
     * Inner = smallest effective split count (most likely to freeze). */
    int eff_counts[D_PARENT];
    for (int i = 0; i < D_PARENT; i++) {
        bin_order[i] = i;
        int lo = (parent_B[i] > x_cap) ? (parent_B[i] - x_cap) : 0;
        int hi = (parent_B[i] < x_cap) ? parent_B[i] : x_cap;
        eff_counts[i] = hi - lo + 1;
        if (eff_counts[i] < 1) eff_counts[i] = 1;
    }
    /* Simple insertion sort (D_PARENT is small, typically 6) */
    for (int i = 1; i < D_PARENT; i++) {
        int key = bin_order[i];
        int key_eff = eff_counts[key];
        int j = i - 1;
        while (j >= 0 && eff_counts[bin_order[j]] < key_eff) {
            bin_order[j + 1] = bin_order[j];
            j--;
        }
        bin_order[j + 1] = key;
    }

    /* Choose K_OUTER: default to D_PARENT/2 (e.g., 3 for D_PARENT=6).
     * For D_PARENT=6 at m=100, K_OUTER=3 is the sweet spot. */
    int k_outer = D_PARENT / 2;
    if (k_outer < 1) k_outer = 1;
    if (k_outer > D_PARENT - 1) k_outer = D_PARENT - 1;

    return k_outer;
}

/* Compute outer-only work count (product of outer bins' effective splits). */
static long long compute_outer_work(
    const int* parent_B, const int* lo_arr, const int* bin_order,
    int k_outer, int d_parent, int x_cap)
{
    long long N = 1;
    for (int k = 0; k < k_outer; k++) {
        int bi = bin_order[k];
        int lo = lo_arr[bi];
        int hi = (parent_B[bi] < x_cap) ? parent_B[bi] : x_cap;
        if (hi < lo) hi = lo;
        int s = hi - lo + 1;
        if (s <= 0) return 0;
        N *= (long long)s;
        if (N < 0) return -1;
    }
    return N;
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
    double time_budget_sec,      /* 0 = no limit */
    int no_freeze                /* 1 = disable freeze kernel, use original batched kernel */
) {
    constexpr int D_PARENT = D_CHILD / 2;

    if (d_parent != D_PARENT) {
        fprintf(stderr, "refine_parents: d_parent=%d but D_CHILD=%d (expected d_parent=%d)\n",
                d_parent, D_CHILD, D_PARENT);
        return -1;
    }

    /* Compute child-level parameters.
     * With S=m convention, total integer mass is always m for all levels.
     * S_child = S_parent = m (mass stays constant; n_half doubles). */
    int S_parent = 0;
    for (int i = 0; i < d_parent; i++) S_parent += parent_configs[i]; /* use first parent to get S */
    int S_child = S_parent;  /* S=m stays constant across refinement levels */
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

    /* Single-bin energy cap: any sub-bin c_i > x_cap is guaranteed to be
     * pruned by the ell=2 max-element check. Skip generating such children.
     * x_cap = floor(m * sqrt(thresh / D_CHILD)), derived from:
     *   (c_i * 4n/m)^2 / (4*n*2) > thresh  =>  c_i > m*sqrt(thresh/d_child) */
    int x_cap = (int)floor((double)m * sqrt(thresh / (double)D_CHILD));
    if (x_cap > m) x_cap = m;
    if (x_cap < 0) x_cap = 0;

    /* int32/int64 dispatch: use int32 when S^2 fits comfortably in int32.
     * For m=50, S=50, S^2=2500 — easily fits. Saves 47 regs for D=24. */
    bool use_int64 = ((long long)S_child * S_child > 2000000000LL);

    /* ============================================================
     * CPU pre-filter: analyze each parent before GPU dispatch.
     * Parents resolved on CPU are marked with ref_counts = 0 and
     * skipped during GPU processing.
     * ============================================================ */
    long long* ref_counts = (long long*)malloc(num_parents * sizeof(long long));
    int* parent_order = (int*)malloc(num_parents * sizeof(int));
    long long cpu_eliminated = 0;
    long long total_refs_saved = 0;

    /* Aggregate results (CPU pre-filter contributes to these) */
    long long total_asym = 0, total_test = 0, total_surv = 0;
    double global_min_tv = 1e30;
    int global_min_cfg[D_CHILD];
    for (int i = 0; i < D_CHILD; i++) global_min_cfg[i] = 0;

    int do_extract = (out_survivor_configs != NULL && max_survivors > 0);

    for (int p = 0; p < num_parents; p++) {
        const int* pB = parent_configs + p * d_parent;
        long long N = compute_refinement_count(pB, d_parent, x_cap);
        ref_counts[p] = N;
        parent_order[p] = p;

        /* Try CPU pre-filter */
        int flat_child[D_CHILD];
        double flat_tv;
        int cpu_result = analyze_parent_cpu<D_CHILD>(
            pB, d_parent, m, c_target, x_cap, flat_child, &flat_tv);

        if (cpu_result == 1) {
            /* Fully eliminated on CPU: all children would be pruned
             * by the window scan (total perturbation < min gap, all above). */
            ref_counts[p] = 0;  /* skip GPU */
            total_refs_saved += N;
            cpu_eliminated++;
            total_test += N;  /* count as test-pruned */
        }
    }

    if (cpu_eliminated > 0) {
        fprintf(stderr, "refine: CPU pre-filter: %lld eliminated (saved %.2e refs)\n",
                (long long)cpu_eliminated, (double)total_refs_saved);
        fflush(stderr);
    }

    /* Sort parents by refinement count ascending — O(n log n) */
    std::sort(parent_order, parent_order + num_parents,
        [ref_counts](int a, int b) { return ref_counts[a] < ref_counts[b]; });

    /* Find first parent with ref_counts > 0 (skip CPU-resolved parents) */
    int first_gpu_parent = 0;
    while (first_gpu_parent < num_parents && ref_counts[parent_order[first_gpu_parent]] == 0)
        first_gpu_parent++;

    /* ============================================================
     * GPU buffer allocation
     * ============================================================ */
    constexpr int block_size = (D_CHILD <= 12) ? REFINE_BLOCK_SIZE_12 :
                               (D_CHILD <= 24) ? REFINE_BLOCK_SIZE_24 :
                                                  REFINE_BLOCK_SIZE_48;
    int max_grid_size = 108 * 32;  /* A100: 108 SMs * 32 blocks/SM */

    /* Per-block output buffers (reused across all launches) */
    long long* d_block_counts = NULL;
    double* d_block_min_tv = NULL;
    int* d_block_min_configs = NULL;
    CUDA_CHECK(cudaMalloc(&d_block_counts, (long long)max_grid_size * 3 * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_block_min_tv, max_grid_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_min_configs, (long long)max_grid_size * D_CHILD * sizeof(int)));

    /* Batched kernel buffers: double-buffered for CUDA stream overlap.
     * Buffer 0 and 1 alternate: while kernel runs on buf[cur], we can
     * upload the next batch to buf[1-cur] on a different stream. */
    int max_batch_parents = MAX_BATCH_PARENTS;
    int* d_batch_parents[2] = {NULL, NULL};
    long long* d_batch_prefix[2] = {NULL, NULL};
    int* d_batch_lo[2] = {NULL, NULL};        /* per-parent lo arrays */
    int* d_batch_bin_order[2] = {NULL, NULL}; /* per-parent bin ordering for freeze kernel */
    for (int buf = 0; buf < 2; buf++) {
        CUDA_CHECK(cudaMalloc(&d_batch_parents[buf],
            (long long)max_batch_parents * d_parent * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_batch_prefix[buf],
            ((long long)max_batch_parents + 1) * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_batch_lo[buf],
            (long long)max_batch_parents * d_parent * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_batch_bin_order[buf],
            (long long)max_batch_parents * d_parent * sizeof(int)));
    }

    /* Host staging buffers for building batches (double-buffered) */
    int* h_batch_parents[2];
    long long* h_batch_prefix[2];
    int* h_batch_lo[2];
    int* h_batch_bin_order[2];
    for (int buf = 0; buf < 2; buf++) {
        h_batch_parents[buf] = (int*)malloc(
            (long long)max_batch_parents * d_parent * sizeof(int));
        h_batch_prefix[buf] = (long long*)malloc(
            ((long long)max_batch_parents + 1) * sizeof(long long));
        h_batch_lo[buf] = (int*)malloc(
            (long long)max_batch_parents * d_parent * sizeof(int));
        h_batch_bin_order[buf] = (int*)malloc(
            (long long)max_batch_parents * d_parent * sizeof(int));
    }

    /* ============================================================
     * Two-stage refinement buffers
     *
     * Step A writes non-frozen configs to intermediate buffers.
     * CUB prefix sum computes load-balancing offsets.
     * Step B processes non-frozen configs with full pruning.
     * ============================================================ */
    /* Two-stage refinement: per-window P1 bounds enable meaningful
     * freeze rates (60-90% expected), making the two-stage split
     * worthwhile. Previously disabled when universal P1 bounds
     * exceeded min_abs_gap by 30-160x (0% freeze rate). */
    int use_two_stage = 1;

    int max_nf_entries = 50000000;  /* 50M entries */
    /* Adjust based on available GPU memory: 20% of free / 28 bytes per entry */
    {
        size_t gpu_free_nf, gpu_total_nf;
        cudaMemGetInfo(&gpu_free_nf, &gpu_total_nf);
        long long nf_bytes_per = 4 + 8 + 8 + 8;  /* parent_idx + outer_idx + inner_work + prefix */
        long long max_from_mem = (long long)((double)gpu_free_nf * 0.2) / nf_bytes_per;
        if (max_from_mem < (long long)max_nf_entries)
            max_nf_entries = (int)max_from_mem;
        if (max_nf_entries < 1000) max_nf_entries = 1000;  /* minimum viable */
    }

    int*       d_nf_parent_idx    = NULL;
    long long* d_nf_outer_idx     = NULL;
    long long* d_nf_inner_work    = NULL;  /* long long for CUB ExclusiveSum compat */
    long long* d_nf_prefix        = NULL;  /* [max_nf_entries + 1] */
    long long* d_block_frozen_test = NULL;
    int*       d_nf_count         = NULL;
    int*       d_nf_overflow      = NULL;
    void*      d_cub_temp         = NULL;
    size_t     cub_temp_bytes     = 0;
    long long* h_block_frozen_test = NULL;

    if (use_two_stage) {
        CUDA_CHECK(cudaMalloc(&d_nf_parent_idx, (long long)max_nf_entries * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_nf_outer_idx,  (long long)max_nf_entries * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_nf_inner_work, ((long long)max_nf_entries + 1) * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_nf_prefix,     ((long long)max_nf_entries + 1) * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_block_frozen_test, (long long)max_grid_size * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_nf_count,    sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_nf_overflow, sizeof(int)));

        /* CUB temporary storage for prefix sum (dry run to query size) */
        cub::DeviceScan::ExclusiveSum(NULL, cub_temp_bytes,
            (long long*)NULL, (long long*)NULL, max_nf_entries + 1);
        CUDA_CHECK(cudaMalloc(&d_cub_temp, cub_temp_bytes));

        h_block_frozen_test = (long long*)malloc((long long)max_grid_size * sizeof(long long));

        fprintf(stderr, "refine: two-stage buffers allocated (max_nf=%d, %.1f MB)\n",
                max_nf_entries,
                (double)((long long)max_nf_entries * 28 + cub_temp_bytes) / (1024.0 * 1024.0));
        fflush(stderr);
    }

    /* CUDA streams for overlapped execution */
    cudaStream_t streams[2];
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));

    /* Single-parent buffers (for large-parent fallback) */
    int* d_parent_B = NULL;
    int* d_splits = NULL;
    CUDA_CHECK(cudaMalloc(&d_parent_B, d_parent * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_splits, d_parent * sizeof(int)));

    /* Dynamic survivor buffer: query actual remaining GPU memory */
    int* d_survivor_buf = NULL;
    int* d_survivor_count = NULL;
    /* do_extract already declared in CPU pre-filter section */
    if (do_extract) {
        size_t gpu_free = 0, gpu_total = 0;
        cudaMemGetInfo(&gpu_free, &gpu_total);
        long long bytes_per_surv = (long long)D_CHILD * sizeof(int);
        /* Use at most 50% of remaining GPU memory for survivor buffer.
         * The kernel needs headroom for execution (stack, local memory,
         * CUDA runtime bookkeeping).  Previous 90% caused illegal memory
         * access errors when the kernel ran out of memory. */
        long long max_from_gpu = (long long)((double)gpu_free * 0.5) / bytes_per_surv;
        /* Hard cap: 200M survivors (far more than any realistic run needs) */
        if (max_from_gpu > 200000000LL) max_from_gpu = 200000000LL;
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

    /* Aggregate results initialized in CPU pre-filter section above */

    clock_t start_time = clock();
    clock_t last_log_time = start_time;
    long long cumulative_refs = 0;
    int timed_out = 0;

    /* ============================================================
     * Main processing loop: batched for small parents,
     * chunked fallback for large parents
     * ============================================================ */
    int p_idx = first_gpu_parent;  /* skip CPU-resolved parents (ref_counts=0) */
    while (p_idx < num_parents && !timed_out) {
        int p = parent_order[p_idx];
        long long N = ref_counts[p];

        if (N <= 0) {
            /* CPU-resolved parent (should not happen after first_gpu_parent, but guard) */
            p_idx++;
            continue;
        }

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

            /* With S=m + energy cap: effective split count per component */
            int h_splits[MAX_D_CHILD / 2];
            for (int i = 0; i < d_parent; i++) {
                int lo = (pB[i] > x_cap) ? (pB[i] - x_cap) : 0;
                int hi = (pB[i] < x_cap) ? pB[i] : x_cap;
                h_splits[i] = hi - lo + 1;
            }

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

                #define LAUNCH_REFINE_SINGLE(USE64) \
                    refine_prove_target<D_CHILD, USE64><<<grid_size, block_size>>>( \
                        d_parent_B, d_splits, d_parent, \
                        S_child, n_half_child, m, \
                        c_target, thresh, \
                        inv_m_f, thresh_f, margin_f, asym_limit_f, \
                        processed, chunk, \
                        d_block_counts, d_block_min_tv, d_block_min_configs, \
                        do_extract ? d_survivor_buf : NULL, \
                        do_extract ? d_survivor_count : NULL, \
                        do_extract ? max_survivors : 0, \
                        x_cap)
                if (use_int64) { LAUNCH_REFINE_SINGLE(true); }
                else           { LAUNCH_REFINE_SINGLE(false); }
                #undef LAUNCH_REFINE_SINGLE

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
             * SMALL PARENTS: batched approach.
             *
             * Two paths:
             *   use_two_stage=1: Step A (freeze check) + CUB prefix
             *     sum + Step B (inner enumeration). Sequential per
             *     batch. Eliminates warp divergence.
             *   use_two_stage=0: Original double-buffered pipeline
             *     with monolithic freeze or batched kernel.
             * ==================================================== */
            constexpr int FREEZE_K_OUTER = D_PARENT / 2 > 0 ? D_PARENT / 2 : 1;
            int use_freeze_kernel = (!no_freeze && D_PARENT >= 4);

          if (use_two_stage) {
            /* ====================================================
             * TWO-STAGE REFINEMENT PATH
             * Sequential batch processing (no double buffering).
             * Step A: uniform work, outputs non-frozen buffer.
             * Step B: processes non-frozen configs only.
             * ==================================================== */
            double observed_freeze_rate = 0.9;  /* conservative initial estimate */
            long long total_outer_processed = 0;
            long long total_nf_observed = 0;
            int build_p_idx = p_idx;

            while (build_p_idx < num_parents && !timed_out) {
                /* Adaptive batch work limit: allow more outer configs
                 * when freeze rate is high (fewer non-frozen outputs). */
                long long adaptive_cap = (long long)max_nf_entries;
                if (observed_freeze_rate > 0.5 && total_outer_processed > 0) {
                    double nf_rate = 1.0 - observed_freeze_rate;
                    if (nf_rate < 0.01) nf_rate = 0.01;  /* floor at 1% */
                    adaptive_cap = (long long)((double)max_nf_entries / nf_rate);
                    if (adaptive_cap > 250000000LL) adaptive_cap = 250000000LL;
                }

                /* --- Build batch into buf[0] --- */
                int batch_count = 0;
                long long batch_work = 0;
                h_batch_prefix[0][0] = 0;

                while (build_p_idx + batch_count < num_parents &&
                       batch_count < max_batch_parents) {
                    int pp = parent_order[build_p_idx + batch_count];
                    long long nn = ref_counts[pp];
                    if (nn <= 0) { build_p_idx++; continue; }
                    if (nn > REFINE_CHUNK_SIZE) break;

                    const int* pB = parent_configs + pp * d_parent;

                    int* bo = h_batch_bin_order[0] + (long long)batch_count * d_parent;
                    compute_freeze_bin_order<D_CHILD>(pB, d_parent, x_cap, bo);

                    int* lo_arr = h_batch_lo[0] + (long long)batch_count * d_parent;
                    compute_default_lo(pB, d_parent, x_cap, lo_arr);

                    long long work_for_prefix = compute_outer_work(
                        pB, lo_arr, bo, FREEZE_K_OUTER, d_parent, x_cap);
                    if (work_for_prefix <= 0) work_for_prefix = 1;

                    if (batch_work + work_for_prefix > adaptive_cap) break;

                    memcpy(h_batch_parents[0] + (long long)batch_count * d_parent,
                           pB, d_parent * sizeof(int));

                    batch_work += work_for_prefix;
                    batch_count++;
                    h_batch_prefix[0][batch_count] = batch_work;
                }

                if (batch_count == 0) break;
                build_p_idx += batch_count;

                /* --- Upload batch (synchronous, no double buffering) --- */
                CUDA_CHECK(cudaMemcpy(d_batch_parents[0],
                    h_batch_parents[0],
                    (long long)batch_count * d_parent * sizeof(int),
                    cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_batch_prefix[0],
                    h_batch_prefix[0],
                    (batch_count + 1) * sizeof(long long),
                    cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_batch_lo[0],
                    h_batch_lo[0],
                    (long long)batch_count * d_parent * sizeof(int),
                    cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_batch_bin_order[0],
                    h_batch_bin_order[0],
                    (long long)batch_count * d_parent * sizeof(int),
                    cudaMemcpyHostToDevice));

                /* --- Reset two-stage counters --- */
                CUDA_CHECK(cudaMemset(d_nf_count, 0, sizeof(int)));
                CUDA_CHECK(cudaMemset(d_nf_overflow, 0, sizeof(int)));

                /* --- STEP A: freeze check (uniform work per thread) --- */
                int grid_a = (int)fmin(
                    (double)(batch_work + block_size - 1) / block_size,
                    (double)max_grid_size);
                if (grid_a < 1) grid_a = 1;

                #define LAUNCH_STEP_A(USE64) \
                    freeze_check_kernel<D_CHILD, FREEZE_K_OUTER, USE64> \
                        <<<grid_a, block_size>>>( \
                        d_batch_parents[0], d_batch_prefix[0], \
                        d_batch_lo[0], d_batch_bin_order[0], \
                        batch_count, \
                        S_child, n_half_child, m, c_target, x_cap, \
                        batch_work, \
                        d_nf_parent_idx, d_nf_outer_idx, d_nf_inner_work, \
                        d_nf_count, d_nf_overflow, max_nf_entries, \
                        d_block_frozen_test)
                if (use_int64) { LAUNCH_STEP_A(true); }
                else           { LAUNCH_STEP_A(false); }
                #undef LAUNCH_STEP_A

                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());

                /* --- Read Step A results --- */
                int h_nf_count = 0, h_overflow = 0;
                CUDA_CHECK(cudaMemcpy(&h_nf_count, d_nf_count,
                    sizeof(int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(&h_overflow, d_nf_overflow,
                    sizeof(int), cudaMemcpyDeviceToHost));

                /* Sum Step A frozen stats (per-block reduction) */
                CUDA_CHECK(cudaMemcpy(h_block_frozen_test, d_block_frozen_test,
                    (long long)grid_a * sizeof(long long), cudaMemcpyDeviceToHost));
                long long batch_frozen_test = 0;
                for (int b = 0; b < grid_a; b++)
                    batch_frozen_test += h_block_frozen_test[b];

                if (h_overflow) {
                    /* ================================================
                     * OVERFLOW: non-frozen buffer too small.
                     * Fall back to monolithic freeze kernel for this
                     * entire batch. Correctness guarantee: the freeze
                     * kernel is known correct.
                     * ================================================ */
                    fprintf(stderr, "refine: two-stage overflow (nf_count=%d > max=%d), "
                            "falling back to freeze kernel\n",
                            h_nf_count, max_nf_entries);
                    fflush(stderr);

                    /* Discard Step A frozen stats (partial/incomplete).
                     * Re-process entire batch with monolithic kernel. */
                    int grid_f = grid_a;

                    #define LAUNCH_FREEZE_FALLBACK(USE64) \
                        refine_prove_target_freeze<D_CHILD, FREEZE_K_OUTER, USE64> \
                            <<<grid_f, block_size>>>( \
                            d_batch_parents[0], d_batch_prefix[0], \
                            d_batch_lo[0], d_batch_bin_order[0], \
                            batch_count, \
                            S_child, n_half_child, m, \
                            c_target, thresh, \
                            inv_m_f, thresh_f, margin_f, asym_limit_f, \
                            batch_work, \
                            d_block_counts, d_block_min_tv, d_block_min_configs, \
                            do_extract ? d_survivor_buf : NULL, \
                            do_extract ? d_survivor_count : NULL, \
                            do_extract ? max_survivors : 0, \
                            x_cap)
                    if (use_int64) { LAUNCH_FREEZE_FALLBACK(true); }
                    else           { LAUNCH_FREEZE_FALLBACK(false); }
                    #undef LAUNCH_FREEZE_FALLBACK

                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());

                    CUDA_CHECK(cudaMemcpy(h_counts, d_block_counts,
                        grid_f * 3 * sizeof(long long), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(h_min_tvs, d_block_min_tv,
                        grid_f * sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(h_min_cfgs, d_block_min_configs,
                        grid_f * D_CHILD * sizeof(int), cudaMemcpyDeviceToHost));

                    reduce_block_results<D_CHILD>(h_counts, h_min_tvs, h_min_cfgs,
                        grid_f, total_asym, total_test, total_surv,
                        global_min_tv, global_min_cfg);

                    /* Reduce adaptive cap for next batch */
                    observed_freeze_rate *= 0.5;

                } else {
                    /* ================================================
                     * Normal two-stage path: Step A succeeded.
                     * Frozen configs already counted. Now process
                     * non-frozen configs with Step B.
                     * ================================================ */
                    total_test += batch_frozen_test;

                    /* Update freeze rate estimate */
                    total_outer_processed += batch_work;
                    total_nf_observed += h_nf_count;
                    if (total_outer_processed > 0)
                        observed_freeze_rate = 1.0 -
                            (double)total_nf_observed / (double)total_outer_processed;

                    if (h_nf_count > 0) {
                        /* --- CUB prefix sum on d_nf_inner_work --- */
                        cub::DeviceScan::ExclusiveSum(
                            d_cub_temp, cub_temp_bytes,
                            d_nf_inner_work, d_nf_prefix,
                            h_nf_count + 1);

                        /* Read only the total: nf_prefix[h_nf_count] */
                        long long total_inner_work = 0;
                        CUDA_CHECK(cudaMemcpy(&total_inner_work,
                            d_nf_prefix + h_nf_count,
                            sizeof(long long), cudaMemcpyDeviceToHost));

                        if (total_inner_work > 0) {
                            /* --- STEP B: inner enumeration --- */
                            int grid_b = (int)fmin(
                                (double)(total_inner_work + block_size - 1) / block_size,
                                (double)max_grid_size);
                            if (grid_b < 1) grid_b = 1;

                            #define LAUNCH_STEP_B(USE64) \
                                inner_enumerate_kernel<D_CHILD, FREEZE_K_OUTER, USE64> \
                                    <<<grid_b, block_size>>>( \
                                    d_batch_parents[0], \
                                    d_batch_lo[0], d_batch_bin_order[0], \
                                    batch_count, \
                                    S_child, n_half_child, m, \
                                    c_target, thresh, \
                                    inv_m_f, thresh_f, margin_f, asym_limit_f, \
                                    x_cap, \
                                    d_nf_parent_idx, d_nf_outer_idx, \
                                    d_nf_prefix, \
                                    h_nf_count, total_inner_work, \
                                    d_block_counts, d_block_min_tv, \
                                    d_block_min_configs, \
                                    do_extract ? d_survivor_buf : NULL, \
                                    do_extract ? d_survivor_count : NULL, \
                                    do_extract ? max_survivors : 0)
                            if (use_int64) { LAUNCH_STEP_B(true); }
                            else           { LAUNCH_STEP_B(false); }
                            #undef LAUNCH_STEP_B

                            CUDA_CHECK(cudaGetLastError());
                            CUDA_CHECK(cudaDeviceSynchronize());

                            /* Reduce Step B results */
                            CUDA_CHECK(cudaMemcpy(h_counts, d_block_counts,
                                grid_b * 3 * sizeof(long long),
                                cudaMemcpyDeviceToHost));
                            CUDA_CHECK(cudaMemcpy(h_min_tvs, d_block_min_tv,
                                grid_b * sizeof(double),
                                cudaMemcpyDeviceToHost));
                            CUDA_CHECK(cudaMemcpy(h_min_cfgs, d_block_min_configs,
                                grid_b * D_CHILD * sizeof(int),
                                cudaMemcpyDeviceToHost));

                            reduce_block_results<D_CHILD>(h_counts, h_min_tvs,
                                h_min_cfgs, grid_b,
                                total_asym, total_test, total_surv,
                                global_min_tv, global_min_cfg);
                        }
                    }

                    fprintf(stderr, "  two-stage batch: %d parents, %.2e outer, "
                            "%d nf (%.1f%% frozen), frozen_test=%lld\n",
                            batch_count, (double)batch_work,
                            h_nf_count,
                            (batch_work > 0) ?
                                100.0 * (1.0 - (double)h_nf_count / (double)batch_work)
                                : 0.0,
                            (long long)batch_frozen_test);
                    fflush(stderr);
                }

                cumulative_refs += batch_work;
                p_idx += batch_count;

                /* Time budget check */
                if (time_budget_sec > 0) {
                    double elapsed_check = (double)(clock() - start_time) / CLOCKS_PER_SEC;
                    if (elapsed_check > time_budget_sec) timed_out = 1;
                }
            }

          } else {
            /* ====================================================
             * ORIGINAL DOUBLE-BUFFERED BATCH PROCESSING
             * (used when use_two_stage=0, i.e. no_freeze or D<8)
             * ==================================================== */
            int cur_buf = 0;
            int have_pending = 0;
            int pending_grid = 0;
            long long pending_work = 0;
            int pending_count = 0;
            int build_p_idx = p_idx;

            while (build_p_idx < num_parents && !timed_out) {
                /* --- Build next batch into buf[cur_buf] --- */
                int batch_count = 0;
                long long batch_work = 0;
                h_batch_prefix[cur_buf][0] = 0;

                while (build_p_idx + batch_count < num_parents &&
                       batch_count < max_batch_parents) {
                    int pp = parent_order[build_p_idx + batch_count];
                    long long nn = ref_counts[pp];
                    if (nn <= 0) { build_p_idx++; continue; }
                    if (nn > REFINE_CHUNK_SIZE) break;

                    const int* pB = parent_configs + pp * d_parent;

                    long long work_for_prefix;
                    if (use_freeze_kernel) {
                        int* bo = h_batch_bin_order[cur_buf] + (long long)batch_count * d_parent;
                        compute_freeze_bin_order<D_CHILD>(pB, d_parent, x_cap, bo);

                        int* lo_arr = h_batch_lo[cur_buf] + (long long)batch_count * d_parent;
                        compute_default_lo(pB, d_parent, x_cap, lo_arr);

                        work_for_prefix = compute_outer_work(
                            pB, lo_arr, bo, FREEZE_K_OUTER, d_parent, x_cap);
                        if (work_for_prefix <= 0) work_for_prefix = 1;
                    } else {
                        work_for_prefix = nn;
                    }

                    if (batch_work + work_for_prefix > REFINE_CHUNK_SIZE) break;

                    memcpy(h_batch_parents[cur_buf] + (long long)batch_count * d_parent,
                           pB, d_parent * sizeof(int));

                    if (!use_freeze_kernel) {
                        compute_default_lo(pB, d_parent, x_cap,
                            h_batch_lo[cur_buf] + (long long)batch_count * d_parent);
                    }

                    batch_work += work_for_prefix;
                    batch_count++;
                    h_batch_prefix[cur_buf][batch_count] = batch_work;
                }

                if (batch_count == 0) break;
                build_p_idx += batch_count;

                /* --- Wait for previous kernel if one is in flight --- */
                if (have_pending) {
                    int prev_buf = 1 - cur_buf;
                    CUDA_CHECK(cudaStreamSynchronize(streams[prev_buf]));

                    CUDA_CHECK(cudaMemcpy(h_counts, d_block_counts,
                        pending_grid * 3 * sizeof(long long), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(h_min_tvs, d_block_min_tv,
                        pending_grid * sizeof(double), cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(h_min_cfgs, d_block_min_configs,
                        pending_grid * D_CHILD * sizeof(int), cudaMemcpyDeviceToHost));

                    reduce_block_results<D_CHILD>(h_counts, h_min_tvs, h_min_cfgs,
                        pending_grid, total_asym, total_test, total_surv,
                        global_min_tv, global_min_cfg);

                    cumulative_refs += pending_work;
                    p_idx += pending_count;
                    have_pending = 0;
                }

                /* --- Upload + launch --- */
                CUDA_CHECK(cudaMemcpyAsync(d_batch_parents[cur_buf],
                    h_batch_parents[cur_buf],
                    (long long)batch_count * d_parent * sizeof(int),
                    cudaMemcpyHostToDevice, streams[cur_buf]));
                CUDA_CHECK(cudaMemcpyAsync(d_batch_prefix[cur_buf],
                    h_batch_prefix[cur_buf],
                    (batch_count + 1) * sizeof(long long),
                    cudaMemcpyHostToDevice, streams[cur_buf]));
                CUDA_CHECK(cudaMemcpyAsync(d_batch_lo[cur_buf],
                    h_batch_lo[cur_buf],
                    (long long)batch_count * d_parent * sizeof(int),
                    cudaMemcpyHostToDevice, streams[cur_buf]));
                if (use_freeze_kernel) {
                    CUDA_CHECK(cudaMemcpyAsync(d_batch_bin_order[cur_buf],
                        h_batch_bin_order[cur_buf],
                        (long long)batch_count * d_parent * sizeof(int),
                        cudaMemcpyHostToDevice, streams[cur_buf]));
                }

                int grid_size = (int)fmin(
                    (double)(batch_work + block_size - 1) / block_size,
                    (double)max_grid_size);
                if (grid_size < 1) grid_size = 1;

                if (use_freeze_kernel) {
                    #define LAUNCH_FREEZE_BATCH(USE64) \
                        refine_prove_target_freeze<D_CHILD, FREEZE_K_OUTER, USE64> \
                            <<<grid_size, block_size, 0, streams[cur_buf]>>>( \
                            d_batch_parents[cur_buf], d_batch_prefix[cur_buf], \
                            d_batch_lo[cur_buf], d_batch_bin_order[cur_buf], \
                            batch_count, \
                            S_child, n_half_child, m, \
                            c_target, thresh, \
                            inv_m_f, thresh_f, margin_f, asym_limit_f, \
                            batch_work, \
                            d_block_counts, d_block_min_tv, d_block_min_configs, \
                            do_extract ? d_survivor_buf : NULL, \
                            do_extract ? d_survivor_count : NULL, \
                            do_extract ? max_survivors : 0, \
                            x_cap)
                    if (use_int64) { LAUNCH_FREEZE_BATCH(true); }
                    else           { LAUNCH_FREEZE_BATCH(false); }
                    #undef LAUNCH_FREEZE_BATCH
                } else {
                    #define LAUNCH_REFINE_BATCH(USE64) \
                        refine_prove_target_batched<D_CHILD, USE64> \
                            <<<grid_size, block_size, 0, streams[cur_buf]>>>( \
                            d_batch_parents[cur_buf], d_batch_prefix[cur_buf], \
                            d_batch_lo[cur_buf], \
                            batch_count, \
                            S_child, n_half_child, m, \
                            c_target, thresh, \
                            inv_m_f, thresh_f, margin_f, asym_limit_f, \
                            batch_work, \
                            d_block_counts, d_block_min_tv, d_block_min_configs, \
                            do_extract ? d_survivor_buf : NULL, \
                            do_extract ? d_survivor_count : NULL, \
                            do_extract ? max_survivors : 0, \
                            x_cap)
                    if (use_int64) { LAUNCH_REFINE_BATCH(true); }
                    else           { LAUNCH_REFINE_BATCH(false); }
                    #undef LAUNCH_REFINE_BATCH
                }

                CUDA_CHECK(cudaGetLastError());

                have_pending = 1;
                pending_grid = grid_size;
                pending_work = batch_work;
                pending_count = batch_count;
                cur_buf = 1 - cur_buf;

                if (time_budget_sec > 0) {
                    double elapsed_check = (double)(clock() - start_time) / CLOCKS_PER_SEC;
                    if (elapsed_check > time_budget_sec) timed_out = 1;
                }
            }

            /* --- Drain the last pending kernel --- */
            if (have_pending) {
                int prev_buf = 1 - cur_buf;
                CUDA_CHECK(cudaStreamSynchronize(streams[prev_buf]));

                CUDA_CHECK(cudaMemcpy(h_counts, d_block_counts,
                    pending_grid * 3 * sizeof(long long), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_min_tvs, d_block_min_tv,
                    pending_grid * sizeof(double), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_min_cfgs, d_block_min_configs,
                    pending_grid * D_CHILD * sizeof(int), cudaMemcpyDeviceToHost));

                reduce_block_results<D_CHILD>(h_counts, h_min_tvs, h_min_cfgs,
                    pending_grid, total_asym, total_test, total_surv,
                    global_min_tv, global_min_cfg);

                cumulative_refs += pending_work;
                p_idx += pending_count;
            }
          } /* end use_two_stage / non-two-stage */
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
     * Read survivor extraction results from GPU
     * ============================================================ */
    int n_ext = 0;
    if (do_extract && d_survivor_count != NULL) {
        int h_surv_count = 0;
        CUDA_CHECK(cudaMemcpy(&h_surv_count, d_survivor_count, sizeof(int), cudaMemcpyDeviceToHost));
        n_ext = h_surv_count;
        if (n_ext > max_survivors) n_ext = max_survivors;
        if (n_ext > 0) {
            CUDA_CHECK(cudaMemcpy(
                out_survivor_configs,
                d_survivor_buf,
                (long long)n_ext * D_CHILD * sizeof(int),
                cudaMemcpyDeviceToHost));
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
    cudaFree(d_block_counts);
    cudaFree(d_block_min_tv);
    cudaFree(d_block_min_configs);
    for (int buf = 0; buf < 2; buf++) {
        cudaFree(d_batch_parents[buf]);
        cudaFree(d_batch_prefix[buf]);
        cudaFree(d_batch_lo[buf]);
        cudaFree(d_batch_bin_order[buf]);
        free(h_batch_parents[buf]);
        free(h_batch_prefix[buf]);
        free(h_batch_lo[buf]);
        free(h_batch_bin_order[buf]);
    }
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaFree(d_parent_B);
    cudaFree(d_splits);
    if (d_survivor_buf) cudaFree(d_survivor_buf);
    if (d_survivor_count) cudaFree(d_survivor_count);
    /* Two-stage buffer cleanup */
    if (d_nf_parent_idx) cudaFree(d_nf_parent_idx);
    if (d_nf_outer_idx) cudaFree(d_nf_outer_idx);
    if (d_nf_inner_work) cudaFree(d_nf_inner_work);
    if (d_nf_prefix) cudaFree(d_nf_prefix);
    if (d_block_frozen_test) cudaFree(d_block_frozen_test);
    if (d_nf_count) cudaFree(d_nf_count);
    if (d_nf_overflow) cudaFree(d_nf_overflow);
    if (d_cub_temp) cudaFree(d_cub_temp);
    if (h_block_frozen_test) free(h_block_frozen_test);
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
static int refine_parents_d8(
    const int* parent_configs, int num_parents, int d_parent, int m, double c_target,
    long long* out_asym, long long* out_test, long long* out_surv,
    double* out_min_tv, int* out_min_cfg,
    int* out_surv_configs, int* out_n_ext, int max_surv, double time_budget,
    int no_freeze)
{
    return refine_parents_impl<8>(parent_configs, num_parents, d_parent, m, c_target,
        out_asym, out_test, out_surv, out_min_tv, out_min_cfg,
        out_surv_configs, out_n_ext, max_surv, time_budget, no_freeze);
}

static int refine_parents_d12(
    const int* parent_configs, int num_parents, int d_parent, int m, double c_target,
    long long* out_asym, long long* out_test, long long* out_surv,
    double* out_min_tv, int* out_min_cfg,
    int* out_surv_configs, int* out_n_ext, int max_surv, double time_budget,
    int no_freeze)
{
    return refine_parents_impl<12>(parent_configs, num_parents, d_parent, m, c_target,
        out_asym, out_test, out_surv, out_min_tv, out_min_cfg,
        out_surv_configs, out_n_ext, max_surv, time_budget, no_freeze);
}

static int refine_parents_d16(
    const int* parent_configs, int num_parents, int d_parent, int m, double c_target,
    long long* out_asym, long long* out_test, long long* out_surv,
    double* out_min_tv, int* out_min_cfg,
    int* out_surv_configs, int* out_n_ext, int max_surv, double time_budget,
    int no_freeze)
{
    return refine_parents_impl<16>(parent_configs, num_parents, d_parent, m, c_target,
        out_asym, out_test, out_surv, out_min_tv, out_min_cfg,
        out_surv_configs, out_n_ext, max_surv, time_budget, no_freeze);
}

static int refine_parents_d24(
    const int* parent_configs, int num_parents, int d_parent, int m, double c_target,
    long long* out_asym, long long* out_test, long long* out_surv,
    double* out_min_tv, int* out_min_cfg,
    int* out_surv_configs, int* out_n_ext, int max_surv, double time_budget,
    int no_freeze)
{
    return refine_parents_impl<24>(parent_configs, num_parents, d_parent, m, c_target,
        out_asym, out_test, out_surv, out_min_tv, out_min_cfg,
        out_surv_configs, out_n_ext, max_surv, time_budget, no_freeze);
}

static int refine_parents_d32(
    const int* parent_configs, int num_parents, int d_parent, int m, double c_target,
    long long* out_asym, long long* out_test, long long* out_surv,
    double* out_min_tv, int* out_min_cfg,
    int* out_surv_configs, int* out_n_ext, int max_surv, double time_budget,
    int no_freeze)
{
    return refine_parents_impl<32>(parent_configs, num_parents, d_parent, m, c_target,
        out_asym, out_test, out_surv, out_min_tv, out_min_cfg,
        out_surv_configs, out_n_ext, max_surv, time_budget, no_freeze);
}

static int refine_parents_d48(
    const int* parent_configs, int num_parents, int d_parent, int m, double c_target,
    long long* out_asym, long long* out_test, long long* out_surv,
    double* out_min_tv, int* out_min_cfg,
    int* out_surv_configs, int* out_n_ext, int max_surv, double time_budget,
    int no_freeze)
{
    return refine_parents_impl<48>(parent_configs, num_parents, d_parent, m, c_target,
        out_asym, out_test, out_surv, out_min_tv, out_min_cfg,
        out_surv_configs, out_n_ext, max_surv, time_budget, no_freeze);
}
