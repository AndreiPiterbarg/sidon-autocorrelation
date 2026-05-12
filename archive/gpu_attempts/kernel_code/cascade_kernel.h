/*
 * cascade_kernel.h — Shared declarations for the Sidon cascade prover GPU kernel.
 *
 * This kernel computes a rigorous mathematical proof.  Every design choice
 * prioritises soundness: no false prunes, complete enumeration, exact
 * integer arithmetic throughout.
 *
 * Build:
 *   nvcc -arch=sm_90 -O3 -ftz=false -prec-div=true -prec-sqrt=true \
 *        -fmad=false -lineinfo cascade_kernel.cu cascade_host.cu    \
 *        -o cascade_prover
 *
 *   Add -DTRACE for enumeration trace, -DDEBUG for device printf.
 */

#ifndef CASCADE_KERNEL_H
#define CASCADE_KERNEL_H

#include <cstdint>

/* ───────────────────── compile-time constants ───────────────────── */

/* Maximum supported dimensions.
 * Sized for n_half=3 cascade through L5: d_parent=96, d_child=192.
 * Static shared memory ~24 KB per block (fits 5+ blocks/SM on H100/B200). */
#define MAX_D_PARENT  128
#define MAX_D_CHILD   256
#define MAX_CONV_LEN  511          /* 2*MAX_D_CHILD - 1           */
#define MAX_ELL_COUNT 511          /* 2*MAX_D_CHILD - 1           */
#define WARP_SIZE     32

/* Survivor staging-buffer capacity per block (shared memory). */
#define SURV_CAP      64

/* Subtree pruning: check at multiple J boundaries for more pruning.
 * At d_child=64, the partial conv signal is weaker (thresholds scale with
 * 1/n_half_child), so we need larger fixed prefixes (smaller J) to trigger.
 * Multi-level: check at J=3, 5, 7 — each fires when gc_j equals that value. */
#define J_LEVELS      3
#define J_CHECK_0     3
#define J_CHECK_1     5
#define J_CHECK_2     7

/* H100 SM count (sm_90). */
#define SM_COUNT      132

/* ───────────────────── kernel launch parameters ─────────────────── */

struct CascadeParams {
    const int32_t*  parents;          /* [num_parents x d_parent]         */
    const int32_t*  lo_arrays;        /* [num_parents x d_parent]         */
    const int32_t*  hi_arrays;        /* [num_parents x d_parent]         */
    const int32_t*  ell_order;        /* [ell_count]                      */
    int32_t*        survivors;        /* [max_survivors x d_child]        */
    int32_t*        survivor_count;   /* scalar, global atomic            */
    int             num_parents;
    int             d_parent;
    int             d_child;
    int             m;
    int             ell_count;        /* 2*d_child - 1                    */
    int             conv_len;         /* 2*d_child - 1                    */
    double          threshold_asym;   /* sqrt(c_target / 2.0)             */
    double          c_target;         /* target constant (e.g. 1.4)       */
    int             max_survivors;
    bool            use_flat_threshold; /* true = Lean axiom mode          */
};

/* ───────────────────── host API ─────────────────────────────────── */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Precompute the int32 threshold table on the host.
 *   table:  output, size ell_count * (S_child+1), row-major [ell_idx][W_int].
 *   S_child = 4 * n_half_child * m (fine-grid total mass).
 *   W_int ranges from 0 to S_child (fine-grid mass in overlapping bins).
 *   ell_count = 2*d_child - 1.
 *   ell_idx = ell - 2  (ell ranges from 2 to 2*d_child).
 *   use_flat_threshold: when true, uses C&S Lemma 3 flat correction (2/m + 1/m²)
 *     for all W_int values.  Required for Lean axiom verification.
 *     When false, uses the tighter W-refined correction (1 + W_int/(2n))/m².
 */
void build_threshold_table(int32_t* table,
                           int d_child, int m, double c_target,
                           bool use_flat_threshold);

/*
 * Precompute the ell scanning order on the host.
 *   ell_order:  output, size ell_count.
 *   Returns the number of entries written (== ell_count).
 */
int build_ell_order(int32_t* ell_order, int d_child);

/*
 * Launch the cascade kernel on the current CUDA device.
 *   Returns 0 on success, nonzero on CUDA error.
 */
int launch_cascade_kernel(const CascadeParams* params);

#ifdef __cplusplus
}
#endif

#endif /* CASCADE_KERNEL_H */
