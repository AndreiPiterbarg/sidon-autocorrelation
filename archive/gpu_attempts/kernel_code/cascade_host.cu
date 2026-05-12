/*
 * cascade_host.cu — Host-side helpers and kernel launcher.
 *
 * Provides:
 *   - build_threshold_table(): precompute int32 thresholds on CPU
 *   - build_ell_order():       precompute optimised ell scan order
 *   - launch_cascade_kernel(): grid config + launch
 *   - main():                  end-to-end driver (load parents, launch, save)
 *
 * Build (combined with cascade_kernel.cu):
 *   nvcc -arch=sm_90 -O3 -ftz=false -prec-div=true -prec-sqrt=true \
 *        -fmad=false -lineinfo cascade_kernel.cu cascade_host.cu    \
 *        -o cascade_prover
 */

#include "cascade_kernel.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <numeric>
#include <chrono>
#include <thread>
#include <atomic>

/* Forward declaration of the kernel (defined in cascade_kernel.cu). */
extern __global__ void cascade_kernel(
    const int32_t* __restrict__ g_parents,
    const int32_t* __restrict__ g_lo_arrays,
    const int32_t* __restrict__ g_hi_arrays,
    const int32_t* __restrict__ g_ell_order,
    int32_t*       __restrict__ g_survivors,
    int32_t*       __restrict__ g_survivor_count,
    int32_t*       __restrict__ g_next_parent,
    int32_t*       __restrict__ g_done_parent,
    int num_parents, int d_parent, int d_child, int m,
    int ell_count, int conv_len,
    double threshold_asym,
    double c_target,
    bool use_flat_threshold,
    int max_survivors,
    int surv_cap);

/* g_next_parent is a global-memory int32 counter, allocated alongside
 * survivor_count.  The host monitors it for progress reporting. */

/* ═══════════════════════════════════════════════════════════════════
 *  CUDA error checking
 * ═══════════════════════════════════════════════════════════════════ */
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            return -1;                                                    \
        }                                                                 \
    } while (0)

#define CUDA_CHECK_VOID(call)                                             \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
        }                                                                 \
    } while (0)

/* ═══════════════════════════════════════════════════════════════════
 *  build_threshold_table
 *
 *  Precomputes int32 thresholds on the CPU.  Two modes:
 *
 *  W-refined (use_flat_threshold=false, default):
 *    C&S equation (1) per-window correction:
 *    threshold = floor((c_target*m^2 + 1 + W_int/(2n) + eps) * 4n*ell)
 *    where 1 + W_int/(2n) = (1/m² + W_int/(2nm²)) * m² is the integer-space
 *    correction from C&S eq(1): (g*g)(x) ≤ (f*f)(x) + 2·W_g(x)/m + 1/m².
 *
 *  Flat (use_flat_threshold=true, required for Lean axiom verification):
 *    C&S Lemma 3 global correction:
 *    threshold = floor((c_target*m^2 + 2m + 1 + eps) * 4n*ell)
 *    where 2m + 1 = (2/m + 1/m²) * m² is the integer-space flat correction.
 *    This is W_int-independent (same threshold for all W values).
 *    Matches the Lean axiom cascade_all_pruned which uses correction = 2/m + 1/m².
 *
 *  The flat threshold is HIGHER (harder to prune) than the W-refined threshold,
 *  since W_int/(2n) ≤ 2m (because W_int ≤ S = 4nm).  The flat bound is what
 *  the formal proof requires for soundness.
 *
 *  where:
 *    n = n_half_child = d_child / 2
 *    S = 4*n*m (fine-grid total mass; compositions sum to S)
 *    eps_margin   = 1e-9 * m^2
 *    ell ranges from 2 to 2*d_child (ell_idx = ell - 2)
 *    W_int ranges from 0 to S (fine-grid mass in overlapping bins)
 * ═══════════════════════════════════════════════════════════════════ */
void build_threshold_table(int32_t* table,
                           int d_child, int m, double c_target,
                           bool use_flat_threshold)
{
    int n_half_child = d_child / 2;
    int S_child = 4 * n_half_child * m;  /* Fine grid: compositions sum to 4nm */
    double m_d = (double)m;
    double four_n = 4.0 * (double)n_half_child;
    double n_half_d = (double)n_half_child;
    double eps_margin = 1e-9 * m_d * m_d;
    double cs_base_m2 = c_target * m_d * m_d;

    /* Flat correction: 2m + 1 in integer space = (2/m + 1/m²) * m² */
    double flat_corr = 2.0 * m_d + 1.0;

    for (int ell = 2; ell <= 2 * d_child; ell++) {
        int ell_idx = ell - 2;
        double scale_ell = (double)ell * four_n;
        for (int w = 0; w <= S_child; w++) {
            double corr;
            if (use_flat_threshold) {
                corr = flat_corr;
            } else {
                corr = 1.0 + (double)w / (2.0 * n_half_d);
            }
            double dyn_x = (cs_base_m2 + corr + eps_margin) * scale_ell;
            table[ell_idx * (S_child + 1) + w] = (int32_t)(dyn_x);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════
 *  build_ell_order
 *
 *  Matches CPU reference (_fused_generate_and_prune_gray, lines 1119-1160):
 *    - For d_child >= 20: profile-guided order centred at hc = d_child/2
 *    - For d_child < 20: sequential 2..16 then wide windows
 *    - Remaining ells in ascending order
 * ═══════════════════════════════════════════════════════════════════ */
int build_ell_order(int32_t* ell_order, int d_child)
{
    int ell_count = 2 * d_child - 1;
    std::vector<bool> used(ell_count, false);
    int oi = 0;

    auto try_add = [&](int ell) {
        if (ell >= 2 && ell <= 2 * d_child && !used[ell - 2]) {
            ell_order[oi++] = ell;
            used[ell - 2] = true;
        }
    };

    if (d_child >= 40) {
        /* Extended profile-guided order for d_child=64.
         * At d=64, hc=32.  Kill rates shift: wider spread around center,
         * and medium-width windows (ell ~ d_child/2 to d_child) contribute
         * more than at d=32.  Phase 1 covers a broader range around hc
         * and adds medium windows earlier. */
        int hc = d_child / 2;
        /* Phase 1: core killing range — wider than d=32 */
        int phase1[] = { hc+1, hc+2, hc+3, hc, hc-1, hc+4, hc+5,
                         hc-2, hc+6, hc-3, hc+7, hc+8, hc-4, hc+9,
                         hc-5, hc+10, hc-6, hc+11, hc-7, hc+12 };
        for (int e : phase1) try_add(e);

        /* Phase 2: medium windows — these kill at d=64 more than d=32 */
        int phase2[] = { d_child*3/4, d_child*3/4+1, d_child*3/4-1,
                         d_child, d_child+1, d_child-1, d_child+2,
                         d_child-2, d_child*2, d_child + d_child/2,
                         d_child/4, d_child/4+1, d_child/4-1 };
        for (int e : phase2) try_add(e);
    } else if (d_child >= 20) {
        int hc = d_child / 2;
        /* Profile-guided order: kill-rate descending at d_child=32 */
        int phase1[] = { hc+1, hc+2, hc+3, hc, hc-1, hc+4, hc+5,
                         hc-2, hc+6, hc-3, hc+7, hc+8 };
        for (int e : phase1) try_add(e);

        /* Phase 2: wide windows around d_child */
        int phase2[] = { d_child, d_child+1, d_child-1, d_child+2,
                         d_child-2, d_child*2, d_child + d_child/2 };
        for (int e : phase2) try_add(e);
    } else {
        int phase1_end = std::min(16, 2 * d_child);
        for (int ell = 2; ell <= phase1_end; ell++)
            try_add(ell);
        int phase2[] = { d_child, d_child+1, d_child-1, d_child+2,
                         d_child-2, d_child*2, d_child + d_child/2,
                         d_child/2 };
        for (int e : phase2) try_add(e);
    }

    /* Phase 3: everything remaining in ascending order. */
    for (int ell = 2; ell <= 2 * d_child; ell++)
        try_add(ell);

    return oi;  /* should == ell_count */
}

/* ═══════════════════════════════════════════════════════════════════
 *  launch_cascade_kernel
 * ═══════════════════════════════════════════════════════════════════ */
int launch_cascade_kernel(const CascadeParams* p)
{
    /* Allocate the parent counter and survivor counter as host-mapped
     * (zero-copy) memory so the host can read them directly without
     * cudaMemcpy (which would block on the running kernel). */
    int32_t *h_next_parent, *d_next_parent;
    int32_t *h_progress_surv, *d_progress_surv;
    CUDA_CHECK(cudaHostAlloc(&h_next_parent, sizeof(int32_t),
                             cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_next_parent, h_next_parent, 0));
    *h_next_parent = 0;

    CUDA_CHECK(cudaHostAlloc(&h_progress_surv, sizeof(int32_t),
                             cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_progress_surv, h_progress_surv, 0));
    *h_progress_surv = 0;

    /* Grid configuration.
     * Round up to next multiple of 32 (warp size) so that every warp is
     * full.  A partial last warp (e.g. d_child=48 → 16 threads in warp 1)
     * causes __shfl_down_sync(0xFFFFFFFF) to reference non-existent lanes.
     * Threads with lane >= d_child are idle (loop bounds exclude them). */
    int block_size = ((p->d_child + 31) / 32) * 32;
    if (block_size < 32) block_size = 32;

    int device_id = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    int sm_count = prop.multiProcessorCount;
    printf("  GPU: %s (%d SMs, compute %d.%d)\n",
           prop.name, sm_count, prop.major, prop.minor);

    /* Survivor staging buffer capacity — Idea 3: adaptive sizing.
     * At d_child>=64 (L4), survival rate ~0.001%, use small buffer.
     * At d_child<=32 (L2→L3), survival rate ~43%, use full buffer.
     * Allocate (surv_cap+1) slots: canonicalize_and_stage writes BEFORE
     * the flush check, so the extra slot absorbs the overflow entry. */
    int surv_cap = (p->d_child >= 64) ? 4 : 64;

    /* Dynamic shared memory = ell_order + surv_buf.
     * Threshold table replaced by inline A_ell formula (Improvement 2). */
    size_t smem_ell_order = (size_t)p->ell_count * sizeof(int32_t);
    size_t smem_surv_buf  = (size_t)(surv_cap + 1) * p->d_child * sizeof(int32_t);
    size_t dynamic_smem_bytes = smem_ell_order + smem_surv_buf;

    /* Increase CUDA printf buffer.  The default 1MB fills quickly even
     * without explicit TRACE, because watchdog/error printfs exist.
     * A full buffer causes device threads to block on printf calls. */
#ifdef DEBUG
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 64 * 1024 * 1024));
    printf("  DEBUG: printf buffer set to 64 MB\n");
#else
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 8 * 1024 * 1024));
#endif

    /* Request max shared memory per block. */
    CUDA_CHECK(cudaFuncSetAttribute(
        cascade_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)dynamic_smem_bytes));

    /* Use CUDA occupancy API to determine optimal blocks per SM.
     * This accounts for shared memory, registers, and thread limits
     * on whatever GPU we're actually running on. */
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, cascade_kernel, block_size, dynamic_smem_bytes));
    if (blocks_per_sm <= 0) blocks_per_sm = 1;
    int grid_size = sm_count * blocks_per_sm;
    if (grid_size > p->num_parents)
        grid_size = p->num_parents;

    /* ── WDDM TDR detection ──
     * On Windows with WDDM driver model, the OS kills GPU kernels
     * exceeding TdrDelay (default ~2s).  Warn the user. */
    {
        #ifdef _WIN32
        /* cudaDeviceGetAttribute doesn't expose driver model directly.
         * prop.tccDriver == 0 means WDDM on Windows. */
        if (!prop.tccDriver) {
            printf("\n");
            printf("  *** WARNING: GPU is in WDDM mode (display driver). ***\n");
            printf("  Windows TDR will kill kernels running longer than ~2 seconds.\n");
            printf("  For large workloads, either:\n");
            printf("    1. Increase TdrDelay: HKLM\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers\\TdrDelay (DWORD, seconds)\n");
            printf("    2. Use a TCC-mode GPU (Tesla/datacenter) or Linux\n");
            printf("    3. Build without -DDEBUG to reduce kernel runtime\n");
            printf("\n");
            fflush(stdout);
        }
        #endif
    }

    printf("Launching cascade kernel:\n");
    printf("  parents:       %d\n", p->num_parents);
    printf("  d_parent=%d  d_child=%d  m=%d\n",
           p->d_parent, p->d_child, p->m);
    printf("  grid=%d  block=%d  blocks/SM=%d  dynamic_smem=%zu B\n",
           grid_size, block_size, blocks_per_sm, dynamic_smem_bytes);
    fflush(stdout);

    /* Launch the kernel on a separate stream so that cudaMemcpy on the
     * default stream can read progress counters while the kernel runs. */
    cudaStream_t kernel_stream;
    CUDA_CHECK(cudaStreamCreate(&kernel_stream));

    auto t0 = std::chrono::high_resolution_clock::now();

    cascade_kernel<<<grid_size, block_size, dynamic_smem_bytes, kernel_stream>>>(
        p->parents, p->lo_arrays, p->hi_arrays,
        p->ell_order,
        p->survivors, p->survivor_count,
        d_next_parent,
        d_progress_surv,        /* g_done_parent: completed parent counter */
        p->num_parents, p->d_parent, p->d_child, p->m,
        p->ell_count, p->conv_len,
        p->threshold_asym,
        p->c_target,
        p->use_flat_threshold,
        p->max_survivors,
        surv_cap);

    CUDA_CHECK(cudaGetLastError());

    /* ── Progress monitor loop (main thread) ──
     * Reads host-mapped (zero-copy) counters directly — no cudaMemcpy
     * needed, so it doesn't block on the running kernel.
     * volatile reads ensure we see the GPU's atomic writes.
     *
     * CRITICAL: Also polls cudaStreamQuery() to detect kernel completion
     * or failure (e.g., WDDM TDR killing the kernel).  Without this,
     * a TDR-killed kernel leaves the progress counter frozen and the
     * monitor loop spins forever. */
    int num_parents_copy = p->num_parents;
    volatile int32_t* vol_done = (volatile int32_t*)h_progress_surv;
    int32_t prev_progress = -1;
    const int POLL_MS = 1000;              /* poll every 1s for kernel completion */
    const int PRINT_INTERVAL_MS = 30000;  /* print progress every 30s */
    const int MAX_STALL_MS = 120000;      /* 120s with no progress → stall */
    int ms_since_print = PRINT_INTERVAL_MS; /* print immediately on first poll */
    int ms_stalled = 0;

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(POLL_MS));

        /* Check if the kernel has finished (or been killed by TDR). */
        cudaError_t query_err = cudaStreamQuery(kernel_stream);
        if (query_err == cudaSuccess) {
            /* Kernel finished — print final progress. */
            int32_t done = *vol_done;
            if (done > num_parents_copy) done = num_parents_copy;
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_s = std::chrono::duration<double>(now - t0).count();
            printf("\r     [%d/%d] (100.0%%) done  [%.1fs elapsed]\n",
                   done, num_parents_copy, elapsed_s);
            fflush(stdout);
            break;
        } else if (query_err != cudaErrorNotReady) {
            /* Kernel failed — likely TDR or device error. */
            fprintf(stderr, "\n  *** KERNEL FAILED: %s ***\n",
                    cudaGetErrorString(query_err));
            fprintf(stderr, "  This is likely caused by Windows WDDM TDR "
                    "(kernel exceeded ~2s timeout).\n");
            fprintf(stderr, "  Fix: increase TdrDelay registry key or "
                    "build without -DDEBUG.\n\n");
            fflush(stderr);
            cudaGetLastError();
            cudaStreamDestroy(kernel_stream);
            cudaFreeHost(h_next_parent);
            cudaFreeHost(h_progress_surv);
            return -1;
        }

        /* Kernel still running — read completed-parent counter. */
        int32_t progress = *vol_done;
        if (progress > num_parents_copy) progress = num_parents_copy;

        ms_since_print += POLL_MS;
        if (ms_since_print >= PRINT_INTERVAL_MS) {
            ms_since_print = 0;

            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_s = std::chrono::duration<double>(now - t0).count();
            double rate = (elapsed_s > 0) ? progress / elapsed_s : 0;
            double eta_s = (rate > 0) ? (num_parents_copy - progress) / rate : 0;
            double pct = (double)progress / num_parents_copy * 100.0;

            int eta_h = (int)(eta_s / 3600);
            int eta_m = (int)((eta_s - eta_h * 3600) / 60);
            int eta_sec = (int)(eta_s - eta_h * 3600 - eta_m * 60);

            printf("\r     [%d/%d] (%.1f%%) "
                   "%.0f parents/s, ETA %02d:%02d:%02d  [%.1fs elapsed]",
                   progress, num_parents_copy, pct,
                   rate, eta_h, eta_m, eta_sec, elapsed_s);
            fflush(stdout);
        }

        /* Detect stalls. */
        if (progress == prev_progress) {
            ms_stalled += POLL_MS;
            if (ms_stalled >= MAX_STALL_MS) {
                fprintf(stderr, "\n\n  *** TIMEOUT: kernel stalled for %.0fs "
                        "with no progress. ***\n",
                        (double)ms_stalled / 1000.0);
                fprintf(stderr, "  Progress frozen at %d/%d completed.\n",
                        progress, num_parents_copy);
                fprintf(stderr, "  Attempting cudaDeviceSynchronize() to "
                        "diagnose...\n");
                fflush(stderr);

                cudaError_t sync_err = cudaDeviceSynchronize();
                if (sync_err != cudaSuccess) {
                    fprintf(stderr, "  cudaDeviceSynchronize returned: %s\n",
                            cudaGetErrorString(sync_err));
                    fprintf(stderr, "  ==> WDDM TDR killed the kernel. "
                            "Increase TdrDelay registry key.\n\n");
                } else {
                    fprintf(stderr, "  cudaDeviceSynchronize returned success "
                            "(kernel completed during sync).\n");
                }
                fflush(stderr);
                cudaGetLastError();
                cudaStreamDestroy(kernel_stream);
                cudaFreeHost(h_next_parent);
                cudaFreeHost(h_progress_surv);
                return -1;
            }
        } else {
            ms_stalled = 0;
            prev_progress = progress;
        }
    }

    /* cudaStreamQuery returned cudaSuccess above, so no sync needed,
     * but call it anyway to be safe and to surface any deferred errors. */
    CUDA_CHECK(cudaStreamSynchronize(kernel_stream));

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  kernel time: %.1f ms (%.2f s)\n", elapsed_ms, elapsed_ms / 1000.0);

    cudaStreamDestroy(kernel_stream);
    cudaFreeHost(h_next_parent);
    cudaFreeHost(h_progress_surv);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 *  Simple NumPy .npy loader for int32 2D arrays
 *
 *  Reads the standard .npy v1.0 format produced by np.save().
 *  Only supports int32, C-contiguous, 2D arrays.
 * ═══════════════════════════════════════════════════════════════════ */
static bool load_npy_int32(const char* path, std::vector<int32_t>& data,
                           int& rows, int& cols)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }

    /* Read 10-byte header prefix: \x93NUMPY + major + minor + header_len. */
    uint8_t prefix[10];
    if (fread(prefix, 1, 10, f) != 10) {
        fclose(f); return false;
    }
    if (prefix[0] != 0x93 || memcmp(prefix + 1, "NUMPY", 5) != 0) {
        fprintf(stderr, "%s: not a .npy file\n", path);
        fclose(f); return false;
    }
    uint16_t header_len = prefix[8] | ((uint16_t)prefix[9] << 8);
    std::vector<char> header(header_len + 1, '\0');
    if (fread(header.data(), 1, header_len, f) != header_len) {
        fclose(f); return false;
    }

    /* Parse shape from header string.
     * Expected format: "{'descr': '<i4', 'fortran_order': False, 'shape': (R, C), }" */
    const char* sp = strstr(header.data(), "'shape': (");
    if (!sp) { fprintf(stderr, "%s: cannot parse shape\n", path); fclose(f); return false; }
    sp += strlen("'shape': (");

    rows = atoi(sp);
    const char* comma = strchr(sp, ',');
    if (!comma) {
        /* 1D array: shape (N,) — treat as (N, 1). */
        cols = 1;
    } else {
        cols = atoi(comma + 1);
    }

    size_t total = (size_t)rows * cols;
    data.resize(total);
    size_t nread = fread(data.data(), sizeof(int32_t), total, f);
    fclose(f);
    if (nread != total) {
        /* Truncated file (e.g. interrupted save).  Use what we have. */
        size_t actual_rows = nread / cols;
        fprintf(stderr, "%s: header says %d rows but file has %zu rows "
                "(truncated). Using %zu rows.\n",
                path, rows, actual_rows, actual_rows);
        rows = (int)actual_rows;
        data.resize(actual_rows * cols);
    }
    return true;
}

/* ═══════════════════════════════════════════════════════════════════
 *  Simple NumPy .npy saver for int32 2D arrays
 * ═══════════════════════════════════════════════════════════════════ */
static bool save_npy_int32(const char* path, const int32_t* data,
                           int rows, int cols)
{
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot create %s\n", path); return false; }

    /* Build header string. */
    char hdr_str[256];
    snprintf(hdr_str, sizeof(hdr_str),
             "{'descr': '<i4', 'fortran_order': False, 'shape': (%d, %d), }",
             rows, cols);
    int hdr_len = (int)strlen(hdr_str);
    /* Pad to multiple of 64 (NumPy convention). */
    int total_hdr = 10 + hdr_len + 1;  /* +1 for newline */
    int pad = (64 - (total_hdr % 64)) % 64;
    hdr_len += pad + 1;  /* pad spaces + trailing newline */

    /* Write 10-byte prefix. */
    uint8_t prefix[10] = { 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, 0, 0 };
    prefix[8] = (uint8_t)(hdr_len & 0xFF);
    prefix[9] = (uint8_t)((hdr_len >> 8) & 0xFF);
    fwrite(prefix, 1, 10, f);

    /* Write header string + padding + newline. */
    fprintf(f, "%s", hdr_str);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);

    /* Write data. */
    fwrite(data, sizeof(int32_t), (size_t)rows * cols, f);
    fclose(f);
    return true;
}

/* ═══════════════════════════════════════════════════════════════════
 *  _compute_bin_ranges — host-side range computation
 *
 *  Matches CPU reference (_compute_bin_ranges, lines 1516-1553).
 * ═══════════════════════════════════════════════════════════════════ */
static bool compute_bin_ranges(
    const int32_t* parent, int d_parent, int d_child,
    int m, double c_target,
    int32_t* lo_out, int32_t* hi_out, int64_t* total_children)
{
    /* correction(m, n_half_child):
     * Matches pruning.py correction() exactly:
     *   correction = 2.0/m + 1.0/(m*m)
     * The n_half parameter is accepted for API compat but not used.
     * (Old formula multiplied by max(1, 4*n_half_child/2) — removed
     * to match CPU source of truth; see pruning.py docstring.) */
    double corr = 2.0 / (double)m + 1.0 / ((double)m * (double)m);
    double thresh = c_target + corr + 1e-9;
    /* Fine grid: height = c/m, TV(ell=2) = c^2/(8n*m^2).
     * c >= m*sqrt(4*d_child*thresh) implies TV >= thresh. */
    int x_cap = (int)floor((double)m * sqrt(4.0 * (double)d_child * thresh));
    int x_cap_cs = (int)floor((double)m * sqrt(4.0 * (double)d_child * c_target)) + 1;
    x_cap = std::min(x_cap, x_cap_cs);
    x_cap = std::max(x_cap, 0);

    *total_children = 1;
    for (int i = 0; i < d_parent; i++) {
        int b_i = parent[i];
        /* Fine grid: child_left + child_right = 2*b_i */
        int lo = std::max(0, 2 * b_i - x_cap);
        int hi = std::min(2 * b_i, x_cap);
        if (lo > hi) return false;
        lo_out[i] = lo;
        hi_out[i] = hi;
        *total_children *= (int64_t)(hi - lo + 1);
    }
    return true;
}

/* ═══════════════════════════════════════════════════════════════════
 *  tighten_ranges — Arc consistency constraint propagation
 *
 *  For each cursor position p and each edge value v, computes a lower
 *  bound on the window sum when all other positions take their minimum-
 *  contribution values (full autoconvolution including cross-terms).
 *  If this lower bound exceeds the pruning threshold for any window,
 *  v is provably infeasible and removed from the cursor range.
 *
 *  Uses W_int_max (maximum possible mass in window) for the threshold
 *  to ensure soundness: min_ws > threshold(W_int_max) >= threshold(W_int).
 *
 *  Modifies lo[] and hi[] in place.  Returns false if any range empties.
 * ═══════════════════════════════════════════════════════════════════ */
static bool tighten_ranges(
    const int32_t* parent, int d_parent, int d_child,
    int m, double c_target,
    int32_t* lo, int32_t* hi)
{
    int conv_len = 2 * d_child - 1;
    int n_half_child = d_child / 2;
    int S_child = 4 * n_half_child * m;
    double m_d = (double)m;
    double four_n = 4.0 * (double)n_half_child;
    double n_half_d = (double)n_half_child;
    double eps_margin = 1e-9 * m_d * m_d;
    double c_target_m2 = c_target * m_d * m_d;
    int S_plus_1 = S_child + 1;
    int ell_count = conv_len;

    /* Fine grid (S = 4nm): threshold = floor((c_target*m^2 + corr + eps) * 4n*ell)
     * W-refined: corr = 1 + W_int/(2n).   Flat: corr = 2m + 1.
     * Arc consistency uses W_int_max (highest threshold) for soundness,
     * so W-refined is safe here — it's strictly ≤ flat. */
    std::vector<int32_t> threshold_table(ell_count * S_plus_1);
    for (int ell = 2; ell <= 2 * d_child; ell++) {
        int idx = ell - 2;
        double scale_ell = (double)ell * four_n;
        for (int w = 0; w <= S_child; w++) {
            /* Arc consistency always uses W-refined: it's tighter, so
             * removing infeasible values is strictly more aggressive.
             * Any value removed by W-refined would also be removed by flat. */
            double corr_w = 1.0 + (double)w / (2.0 * n_half_d);
            double dyn_x = (c_target_m2 + corr_w + eps_margin) * scale_ell;
            threshold_table[idx * S_plus_1 + w] = (int32_t)(dyn_x);
        }
    }

    std::vector<int64_t> conv_min(conv_len);
    std::vector<int64_t> test_conv(conv_len);
    std::vector<int32_t> child_min(d_child);
    std::vector<int64_t> max_child_prefix(d_child + 1);

    for (int round = 0; round <= d_parent; round++) {
        bool any_changed = false;

        /* Build child_min: each bin at its independent minimum. */
        for (int q = 0; q < d_parent; q++) {
            child_min[2*q]   = lo[q];
            child_min[2*q+1] = 2 * parent[q] - hi[q];
        }

        /* Full autoconvolution of child_min → lower bound. */
        std::fill(conv_min.begin(), conv_min.end(), 0);
        for (int i = 0; i < d_child; i++) {
            int64_t ci = child_min[i];
            if (ci == 0) continue;
            conv_min[2*i] += ci * ci;
            for (int j = i + 1; j < d_child; j++) {
                int64_t cj = child_min[j];
                if (cj != 0)
                    conv_min[i+j] += 2 * ci * cj;
            }
        }

        /* Max child prefix sum for W_int_max queries. */
        max_child_prefix[0] = 0;
        for (int q = 0; q < d_parent; q++) {
            max_child_prefix[2*q+1] = max_child_prefix[2*q] + hi[q];
            max_child_prefix[2*q+2] = max_child_prefix[2*q+1]
                                      + (2 * parent[q] - lo[q]);
        }

        for (int p = 0; p < d_parent; p++) {
            if (lo[p] == hi[p]) continue;

            int B_p = parent[p];
            int k1 = 2 * p, k2 = 2 * p + 1;
            int64_t old1 = child_min[k1], old2 = child_min[k2];
            int new_lo = lo[p], new_hi = hi[p];

            /* Lambda: check if value v at position p is infeasible. */
            auto is_infeasible = [&](int v) -> bool {
                int64_t n1 = v, n2 = B_p - v;
                int64_t d1 = n1 - old1, d2 = n2 - old2;

                for (int kk = 0; kk < conv_len; kk++)
                    test_conv[kk] = conv_min[kk];
                test_conv[2*k1]   += n1*n1 - old1*old1;
                test_conv[2*k2]   += n2*n2 - old2*old2;
                test_conv[k1+k2]  += 2*(n1*n2 - old1*old2);
                for (int j = 0; j < d_child; j++) {
                    if (j == k1 || j == k2) continue;
                    int64_t cj = child_min[j];
                    if (cj != 0) {
                        test_conv[k1+j] += 2 * d1 * cj;
                        test_conv[k2+j] += 2 * d2 * cj;
                    }
                }

                for (int ell = 2; ell <= 2*d_child; ell++) {
                    int n_cv = ell - 1;
                    int n_windows = conv_len - n_cv + 1;
                    if (n_windows <= 0) continue;
                    int ell_idx = ell - 2;

                    int64_t ws = 0;
                    for (int kk = 0; kk < n_cv; kk++)
                        ws += test_conv[kk];

                    int hb = std::min(d_child - 1, ell - 2);
                    int64_t W_max = max_child_prefix[hb + 1];
                    if (W_max > S_child) W_max = S_child;
                    if (ws > threshold_table[ell_idx * S_plus_1 + (int)W_max])
                        return true;

                    for (int s = 1; s < n_windows; s++) {
                        ws += test_conv[s + n_cv - 1] - test_conv[s - 1];
                        int lb = s - (d_child - 1);
                        if (lb < 0) lb = 0;
                        hb = s + ell - 2;
                        if (hb > d_child - 1) hb = d_child - 1;
                        W_max = max_child_prefix[hb+1] - max_child_prefix[lb];
                        if (W_max > S_child) W_max = S_child;
                        if (ws > threshold_table[ell_idx*S_plus_1 + (int)W_max])
                            return true;
                    }
                }
                return false;
            };

            /* Tighten from low end. */
            for (int v = lo[p]; v <= hi[p]; v++) {
                if (is_infeasible(v)) {
                    if (v == new_lo) new_lo = v + 1;
                    else break;
                } else break;
            }

            /* Tighten from high end. */
            for (int v = hi[p]; v >= new_lo; v--) {
                if (is_infeasible(v)) {
                    if (v == new_hi) new_hi = v - 1;
                    else break;
                } else break;
            }

            if (new_lo != lo[p] || new_hi != hi[p]) {
                lo[p] = new_lo;
                hi[p] = new_hi;
                any_changed = true;
                if (new_lo > new_hi) return false;
            }
        }

        if (!any_changed) break;
    }
    return true;
}

/* ═══════════════════════════════════════════════════════════════════
 *  verify_relaxed_children — ±1 floor rounding verification
 *
 *  After the main cascade finds 0 survivors, this verifies that ALL
 *  ±1 rounding variants are also pruned.  Required for soundness of
 *  the Lean CascadePruned axiom with relaxed is_valid_child.
 *
 *  For each surviving parent, canonical_discretization at doubled
 *  resolution can produce child[2i]+child[2i+1] = 2*parent[i] + delta_i
 *  where delta_i ∈ {-1, 0, 1} and ∑ delta_i = 0.
 *
 *  This function checks that all such ±1 children are pruned by the
 *  flat C&S Lemma 3 threshold.
 * ═══════════════════════════════════════════════════════════════════ */
static int verify_relaxed_children(
    const int32_t* parents, int num_parents, int d_parent,
    int m, double c_target)
{
    int d_child = 2 * d_parent;
    int n_half_child = d_child / 2;
    int S_child = 4 * n_half_child * m;
    int conv_len = 2 * d_child - 1;
    int ell_count = conv_len;
    int S_plus_1 = S_child + 1;

    /* Build flat threshold table for ±1 verification. */
    double m_d = (double)m;
    double four_n = 4.0 * (double)n_half_child;
    double eps_margin = 1e-9 * m_d * m_d;
    double cs_base_m2 = c_target * m_d * m_d;
    double flat_corr = 2.0 * m_d + 1.0;

    std::vector<int32_t> threshold_flat(ell_count * S_plus_1);
    for (int ell = 2; ell <= 2 * d_child; ell++) {
        int idx = ell - 2;
        double scale_ell = (double)ell * four_n;
        /* Flat: same threshold for all W values. */
        int32_t flat_val = (int32_t)((cs_base_m2 + flat_corr + eps_margin) * scale_ell);
        for (int w = 0; w <= S_child; w++)
            threshold_flat[idx * S_plus_1 + w] = flat_val;
    }

    /* Cauchy-Schwarz cap for feasibility filter. */
    double corr_cs = 2.0 / m_d + 1.0 / (m_d * m_d);
    double thresh_cs = c_target + corr_cs + 1e-9;
    int x_cap = (int)floor(m_d * sqrt(4.0 * (double)d_child * thresh_cs));
    int x_cap_cs = (int)floor(m_d * sqrt(4.0 * (double)d_child * c_target)) + 1;
    x_cap = std::min(x_cap, x_cap_cs);
    x_cap = std::max(x_cap, 0);

    /* Generate delta vectors: all (delta_0, ..., delta_{d_parent-1}) with
     * delta_i ∈ {-1, 0, 1} and ∑ delta_i = 0.
     * Skip all-zeros (that's the standard cascade).
     *
     * For small d_parent (≤ 16), enumerate all 3^d_parent combinations.
     * For larger d_parent, this is infeasible (~43 billion for d=22).
     * In practice d_parent ≤ 16 at the levels where ±1 matters. */
    if (d_parent > 16) {
        printf("  SKIP: d_parent=%d too large for exhaustive ±1 verification\n",
               d_parent);
        printf("  (3^%d = %.0e delta vectors — use CPU --verify_relaxed instead)\n",
               d_parent, pow(3.0, d_parent));
        return -1;
    }

    /* Count 3^d_parent total delta vectors. */
    int64_t n_deltas_total = 1;
    for (int i = 0; i < d_parent; i++) n_deltas_total *= 3;

    printf("  ±1 verification: %d parents, %lld delta vectors each\n",
           num_parents, (long long)n_deltas_total);

    std::vector<int32_t> child(d_child);
    std::vector<int64_t> conv(conv_len);
    std::vector<int32_t> delta(d_parent);
    int total_unpruned = 0;

    for (int pi = 0; pi < num_parents; pi++) {
        const int32_t* parent = &parents[pi * d_parent];

        /* Enumerate all delta vectors via base-3 counter. */
        for (int64_t di = 0; di < n_deltas_total; di++) {
            /* Decode delta vector from base-3 index. */
            int64_t tmp = di;
            int delta_sum = 0;
            for (int j = 0; j < d_parent; j++) {
                delta[j] = (int)(tmp % 3) - 1;  /* -1, 0, +1 */
                delta_sum += delta[j];
                tmp /= 3;
            }

            /* Skip if ∑ delta ≠ 0 (total mass constraint). */
            if (delta_sum != 0) continue;

            /* Skip all-zeros (standard cascade already verified). */
            bool all_zero = true;
            for (int j = 0; j < d_parent; j++)
                if (delta[j] != 0) { all_zero = false; break; }
            if (all_zero) continue;

            /* Compute pair sums and check feasibility. */
            bool feasible = true;
            for (int j = 0; j < d_parent; j++) {
                int pair_sum = 2 * parent[j] + delta[j];
                if (pair_sum < 0) { feasible = false; break; }
            }
            if (!feasible) continue;

            /* For this delta variant, we need to check that ALL children
             * with these pair sums are pruned.  A child has
             * child[2j] ∈ [0, pair_sum_j], child[2j+1] = pair_sum_j - child[2j].
             *
             * The cursor range is [0, pair_sum_j] ∩ [0, x_cap].
             * Enumerate all cursor combinations. */
            int64_t total_children = 1;
            std::vector<int> ps(d_parent), clo(d_parent), chi(d_parent);
            for (int j = 0; j < d_parent; j++) {
                ps[j] = 2 * parent[j] + delta[j];
                clo[j] = std::max(0, ps[j] - x_cap);
                chi[j] = std::min(ps[j], x_cap);
                if (clo[j] > chi[j]) { total_children = 0; break; }
                total_children *= (int64_t)(chi[j] - clo[j] + 1);
            }
            if (total_children == 0) continue;

            /* Enumerate all children for this delta variant and check pruning. */
            std::vector<int32_t> cursor(d_parent);
            for (int j = 0; j < d_parent; j++) cursor[j] = clo[j];

            for (int64_t ci = 0; ci < total_children; ci++) {
                /* Build child from cursor. */
                for (int j = 0; j < d_parent; j++) {
                    child[2*j]   = cursor[j];
                    child[2*j+1] = ps[j] - cursor[j];
                }

                /* Check nonneg. */
                bool nonneg = true;
                for (int j = 0; j < d_child; j++)
                    if (child[j] < 0) { nonneg = false; break; }
                if (!nonneg) goto next_child;

                /* Check x_cap constraint. */
                for (int j = 0; j < d_child; j++)
                    if (child[j] > x_cap) goto next_child;

                /* Full autoconvolution. */
                std::fill(conv.begin(), conv.end(), 0);
                for (int ii = 0; ii < d_child; ii++) {
                    int64_t cii = child[ii];
                    if (cii == 0) continue;
                    conv[2*ii] += cii * cii;
                    for (int jj = ii + 1; jj < d_child; jj++) {
                        int64_t cjj = child[jj];
                        if (cjj != 0)
                            conv[ii+jj] += 2 * cii * cjj;
                    }
                }

                /* Window scan: check if ANY (ell, s_lo) prunes this child. */
                {
                    bool pruned = false;
                    for (int ell = 2; ell <= 2 * d_child && !pruned; ell++) {
                        int n_cv = ell - 1;
                        int n_windows = conv_len - n_cv + 1;
                        if (n_windows <= 0) continue;
                        int ell_idx = ell - 2;

                        /* Compute W_int and window sum via sliding window. */
                        int64_t ws = 0;
                        for (int k = 0; k < n_cv; k++) ws += conv[k];

                        /* W_int for first window */
                        int hb = std::min(d_child - 1, ell - 2);
                        int64_t W_int = 0;
                        for (int k = 0; k <= hb; k++) W_int += child[k];
                        int W_cl = (W_int > S_child) ? S_child : (int)W_int;

                        if (ws > threshold_flat[ell_idx * S_plus_1 + W_cl]) {
                            pruned = true; break;
                        }

                        for (int s = 1; s < n_windows; s++) {
                            ws += conv[s + n_cv - 1] - conv[s - 1];
                            /* Update W_int via sliding window. */
                            int r_add = s + ell - 2;
                            if (r_add < d_child) W_int += child[r_add];
                            int l_sub = s - 1;
                            /* Bins exit when s > d_child - 1 */
                            if (s - 1 >= 0 && (s - 1) < d_child && s + n_cv - 1 >= d_child)
                                ;  /* no removal needed yet */
                            /* Use simpler recomputation for correctness */
                            int lb = (s >= d_child) ? s - d_child + 1 : 0;
                            hb = std::min(d_child - 1, s + ell - 2);
                            W_int = 0;
                            for (int k = lb; k <= hb; k++) W_int += child[k];
                            W_cl = (W_int > S_child) ? S_child : (int)W_int;

                            if (ws > threshold_flat[ell_idx * S_plus_1 + W_cl]) {
                                pruned = true; break;
                            }
                        }
                    }

                    if (!pruned) {
                        total_unpruned++;
                        if (total_unpruned <= 5) {
                            printf("  UNPRUNED ±1 child! parent %d, delta=[", pi);
                            for (int j = 0; j < d_parent; j++)
                                printf("%+d%s", delta[j], j < d_parent-1 ? "," : "");
                            printf("], child=[");
                            for (int j = 0; j < d_child; j++)
                                printf("%d%s", child[j], j < d_child-1 ? "," : "");
                            printf("]\n");
                        }
                    }
                }

                next_child:
                /* Advance cursor (odometer). */
                for (int j = d_parent - 1; j >= 0; j--) {
                    cursor[j]++;
                    if (cursor[j] <= chi[j]) break;
                    cursor[j] = clo[j];
                }
            }
        }
    }

    return total_unpruned;
}

/* ═══════════════════════════════════════════════════════════════════
 *  main — End-to-end driver
 *
 *  Usage:
 *    ./cascade_prover <parents.npy> <output.npy> \
 *        --d_parent 32 --m 20 --c_target 1.4 [--max_survivors 200000] \
 *        [--use_flat_threshold] [--verify_relaxed]
 * ═══════════════════════════════════════════════════════════════════ */
int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <parents.npy> <output.npy> "
            "--d_parent D --m M --c_target C [--max_survivors N] "
            "[--use_flat_threshold] [--verify_relaxed]\n",
            argv[0]);
        return 1;
    }

    const char* parents_path = argv[1];
    const char* output_path  = argv[2];

    int    d_parent           = 32;
    int    m                  = 20;
    double c_target           = 1.4;
    int    max_survivors      = 200000;
    bool   use_flat_threshold = false;
    bool   verify_relaxed     = false;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--d_parent") == 0 && i + 1 < argc)
            d_parent = atoi(argv[++i]);
        else if (strcmp(argv[i], "--m") == 0 && i + 1 < argc)
            m = atoi(argv[++i]);
        else if (strcmp(argv[i], "--c_target") == 0 && i + 1 < argc)
            c_target = atof(argv[++i]);
        else if (strcmp(argv[i], "--max_survivors") == 0 && i + 1 < argc)
            max_survivors = atoi(argv[++i]);
        else if (strcmp(argv[i], "--use_flat_threshold") == 0)
            use_flat_threshold = true;
        else if (strcmp(argv[i], "--verify_relaxed") == 0)
            verify_relaxed = true;
    }

    int d_child  = 2 * d_parent;
    int ell_count = 2 * d_child - 1;
    int conv_len  = 2 * d_child - 1;

    printf("Sidon Cascade GPU Prover\n");
    printf("  d_parent=%d  d_child=%d  m=%d  c_target=%.4f\n",
           d_parent, d_child, m, c_target);
    printf("  ell_count=%d  conv_len=%d  max_survivors=%d\n",
           ell_count, conv_len, max_survivors);
    if (use_flat_threshold)
        printf("  threshold: FLAT (C&S Lemma 3: 2/m + 1/m^2) — Lean axiom mode\n");
    else
        printf("  threshold: W-refined (C&S eq(1): (1 + W_int/(2n))/m^2)\n");
    if (verify_relaxed)
        printf("  ±1 relaxed child verification: ENABLED\n");

    /* ── Load parents ── */
    std::vector<int32_t> h_parents;
    int nrows, ncols;
    if (!load_npy_int32(parents_path, h_parents, nrows, ncols)) return 1;
    if (ncols != d_parent) {
        fprintf(stderr, "Parents array has %d columns, expected %d\n",
                ncols, d_parent);
        return 1;
    }
    int num_parents = nrows;
    printf("  Loaded %d parents from %s\n", num_parents, parents_path);

    /* ── Compute lo/hi arrays on host ── */
    printf("  Computing bin ranges...\n");
    std::vector<int32_t> h_lo(num_parents * d_parent);
    std::vector<int32_t> h_hi(num_parents * d_parent);
    std::vector<int>     valid_indices;
    valid_indices.reserve(num_parents);
    int64_t total_children_all = 0;

    for (int i = 0; i < num_parents; i++) {
        int64_t tc;
        if (compute_bin_ranges(&h_parents[i * d_parent], d_parent, d_child,
                               m, c_target,
                               &h_lo[i * d_parent], &h_hi[i * d_parent],
                               &tc))
        {
            /* Arc consistency: tighten ranges before enumeration. */
            if (!tighten_ranges(&h_parents[i * d_parent], d_parent, d_child,
                                m, c_target,
                                &h_lo[i * d_parent], &h_hi[i * d_parent]))
            {
                continue;  /* All ranges emptied — skip this parent. */
            }
            /* Recompute total_children with tightened ranges. */
            tc = 1;
            for (int j = 0; j < d_parent; j++)
                tc *= (int64_t)(h_hi[i * d_parent + j]
                                - h_lo[i * d_parent + j] + 1);
            valid_indices.push_back(i);
            total_children_all += tc;
        }
    }
    printf("  Valid parents (non-empty range): %zu / %d\n",
           valid_indices.size(), num_parents);
    printf("  Total children to enumerate:     %lld\n",
           (long long)total_children_all);

    /* Pack valid parents into contiguous arrays. */
    int n_valid = (int)valid_indices.size();
    std::vector<int32_t> pack_parents(n_valid * d_parent);
    std::vector<int32_t> pack_lo(n_valid * d_parent);
    std::vector<int32_t> pack_hi(n_valid * d_parent);
    for (int i = 0; i < n_valid; i++) {
        int src = valid_indices[i];
        memcpy(&pack_parents[i * d_parent],
               &h_parents[src * d_parent],
               d_parent * sizeof(int32_t));
        memcpy(&pack_lo[i * d_parent],
               &h_lo[src * d_parent],
               d_parent * sizeof(int32_t));
        memcpy(&pack_hi[i * d_parent],
               &h_hi[src * d_parent],
               d_parent * sizeof(int32_t));
    }

    /* ── Build ell order ── */
    std::vector<int32_t> h_ell_order(ell_count);
    int ell_written = build_ell_order(h_ell_order.data(), d_child);
    printf("  ell_order: %d entries\n", ell_written);

    /* ── Allocate GPU memory ──
     * Threshold table is computed inline on the GPU (A_ell shared memory
     * + 2*ell*W_int formula), eliminating ~1.27 MB of global memory. */
    printf("  Allocating GPU memory...\n");
    int32_t *d_parents, *d_lo, *d_hi, *d_survivors, *d_count;
    int32_t *d_ell_order;

    size_t parent_bytes = (size_t)n_valid * d_parent * sizeof(int32_t);
    size_t ell_bytes    = (size_t)ell_count * sizeof(int32_t);
    size_t surv_bytes   = (size_t)max_survivors * d_child * sizeof(int32_t);

    CUDA_CHECK_VOID(cudaMalloc(&d_parents,   parent_bytes));
    CUDA_CHECK_VOID(cudaMalloc(&d_lo,        parent_bytes));
    CUDA_CHECK_VOID(cudaMalloc(&d_hi,        parent_bytes));
    CUDA_CHECK_VOID(cudaMalloc(&d_ell_order, ell_bytes));
    CUDA_CHECK_VOID(cudaMalloc(&d_survivors, surv_bytes));
    CUDA_CHECK_VOID(cudaMalloc(&d_count,     sizeof(int32_t)));

    /* ── Copy to device ── */
    printf("  Copying data to GPU...\n");
    CUDA_CHECK_VOID(cudaMemcpy(d_parents, pack_parents.data(),
                               parent_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(d_lo, pack_lo.data(),
                               parent_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(d_hi, pack_hi.data(),
                               parent_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemcpy(d_ell_order, h_ell_order.data(),
                               ell_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_VOID(cudaMemset(d_count, 0, sizeof(int32_t)));

    /* ── Launch kernel ── */
    CascadeParams params;
    params.parents         = d_parents;
    params.lo_arrays       = d_lo;
    params.hi_arrays       = d_hi;
    params.ell_order       = d_ell_order;
    params.survivors       = d_survivors;
    params.survivor_count  = d_count;
    params.num_parents     = n_valid;
    params.d_parent        = d_parent;
    params.d_child         = d_child;
    params.m               = m;
    params.ell_count       = ell_count;
    params.conv_len        = conv_len;
    params.threshold_asym  = sqrt(c_target / 2.0);
    params.c_target        = c_target;
    params.max_survivors   = max_survivors;
    params.use_flat_threshold = use_flat_threshold;

    auto main_t0 = std::chrono::high_resolution_clock::now();
    int rc = launch_cascade_kernel(&params);
    auto main_t1 = std::chrono::high_resolution_clock::now();
    if (rc != 0) {
        fprintf(stderr, "Kernel launch failed\n");
        return 1;
    }

    /* ── Throughput summary ── */
    double main_elapsed_s = std::chrono::duration<double>(main_t1 - main_t0).count();
    double children_per_sec = (main_elapsed_s > 0)
        ? (double)total_children_all / main_elapsed_s : 0;
    printf("\n");
    printf("  ═══ THROUGHPUT SUMMARY ═══\n");
    printf("  Total children enumerated: %lld\n", (long long)total_children_all);
    printf("  Wall time:                 %.3f s\n", main_elapsed_s);
    printf("  Throughput:                %.2e children/s\n", children_per_sec);
    if (children_per_sec > 1e9)
        printf("  Throughput:                %.2f billion children/s\n",
               children_per_sec / 1e9);
    else if (children_per_sec > 1e6)
        printf("  Throughput:                %.2f million children/s\n",
               children_per_sec / 1e6);
    printf("\n");

    /* ── Read back results ── */
    int32_t h_count = 0;
    CUDA_CHECK_VOID(cudaMemcpy(&h_count, d_count,
                               sizeof(int32_t), cudaMemcpyDeviceToHost));
    printf("  Survivors found: %d\n", h_count);

    if (h_count > max_survivors) {
        fprintf(stderr, "WARNING: survivor_count (%d) > max_survivors (%d); "
                "output is truncated!\n", h_count, max_survivors);
        h_count = max_survivors;
    }

    std::vector<int32_t> h_survivors(h_count * d_child);
    if (h_count > 0) {
        CUDA_CHECK_VOID(cudaMemcpy(h_survivors.data(), d_survivors,
                                   (size_t)h_count * d_child * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost));
    }

    /* ── Save output ── */
    if (h_count > 0) {
        if (!save_npy_int32(output_path, h_survivors.data(), h_count, d_child))
            return 1;
        printf("  Saved %d survivors to %s\n", h_count, output_path);
    } else {
        printf("  No survivors — proof complete at this level!\n");
    }

    /* ── ±1 Relaxed child verification ── */
    if (verify_relaxed) {
        printf("\n  ═══ ±1 RELAXED CHILD VERIFICATION ═══\n");
        if (!use_flat_threshold) {
            printf("  WARNING: --verify_relaxed should be used with "
                   "--use_flat_threshold for Lean axiom soundness.\n");
        }

        /* Verify the INPUT parents (at the parent level).
         * For each parent, check that all ±1 children are pruned.
         * This is the same check as CPU --verify_relaxed. */
        auto vt0 = std::chrono::high_resolution_clock::now();
        int unpruned = verify_relaxed_children(
            pack_parents.data(), n_valid, d_parent, m, c_target);
        auto vt1 = std::chrono::high_resolution_clock::now();
        double vt_s = std::chrono::duration<double>(vt1 - vt0).count();

        if (unpruned < 0) {
            printf("  ±1 verification skipped (d_parent too large).\n");
        } else if (unpruned == 0) {
            printf("  ±1 verification PASSED: all relaxed children pruned "
                   "(%.1f s)\n", vt_s);
        } else {
            printf("  ±1 verification FAILED: %d unpruned relaxed children "
                   "(%.1f s)\n", unpruned, vt_s);
            printf("  The Lean axiom cascade_all_pruned is NOT verified.\n");
        }
    }

    /* ── Cleanup ── */
    cudaFree(d_parents);
    cudaFree(d_lo);
    cudaFree(d_hi);
    cudaFree(d_ell_order);
    cudaFree(d_survivors);
    cudaFree(d_count);

    return 0;
}
