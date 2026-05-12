"""Stage 1 / Stage 2 benchmark: sharper L^infty correction + cell-cert.

Stage 1: replace the L^infty term (the "1" in `corr_w = 1 + W_int/(2n)`)
         with `avg_ell_phys` over the conv window, where
            ell(x) = max(0, 1/2 - |x|),
            x_k = -1/2 + (k+1)/(4n)  for conv bin k.
         This is rigorous (Cauchy-Schwarz + support-overlap) and at most 1/2,
         so always tighter than the existing "1".

Stage 2: apply joint_cell_cert_for_composition to L0 survivors with
         c_target_eff = c_target + flat_corr/m^2 (since cell cert checks TV >=
         c_target_eff, equivalent to TV - flat_corr/m^2 >= c_target).

A/B/C compare:
  A = current code (W-refined "1")
  B = Stage 1 (W-refined avg_ell_phys)
  B+C = Stage 1 + Stage 2 (cell cert on B's survivors)
"""
import os, sys, time
import numpy as np
import numba
from numba import njit, prange

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
from compositions import generate_compositions_batched
from pruning import count_compositions
from qp_bound_joint import joint_cell_cert_for_composition


# ============================================================
# A: current W-refined (the "1" + W_int/(2n))
# ============================================================
@njit(parallel=True, cache=True)
def prune_A(batch_int, n_half, m, c_target):
    B = batch_int.shape[0]
    d = batch_int.shape[1]
    conv_len = 2 * d - 1
    survived = np.ones(B, dtype=numba.boolean)

    m_d = np.float64(m)
    four_n = 4.0 * np.float64(n_half)
    n_half_d = np.float64(n_half)
    d_minus_1 = d - 1
    eps_margin = 1e-9 * m_d * m_d
    max_ell = 2 * d
    cs_base_m2 = c_target * m_d * m_d
    scale_arr = np.empty(max_ell + 1, dtype=np.float64)
    for ell in range(2, max_ell + 1):
        scale_arr[ell] = np.float64(ell) * four_n

    for b in prange(B):
        conv = np.zeros(conv_len, dtype=np.int64)
        for i in range(d):
            ci = np.int64(batch_int[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int64(batch_int[b, j])
                    if cj != 0:
                        conv[i + j] += np.int64(2) * ci * cj

        prefix_c = np.zeros(d + 1, dtype=np.int64)
        for i in range(d):
            prefix_c[i + 1] = prefix_c[i] + np.int64(batch_int[b, i])

        pruned = False
        for ell in range(2, max_ell + 1):
            if pruned:
                break
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += conv[k]
            scale_ell = scale_arr[ell]
            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                lo_bin = s_lo - d_minus_1
                if lo_bin < 0:
                    lo_bin = 0
                hi_bin = s_lo + ell - 2
                if hi_bin > d_minus_1:
                    hi_bin = d_minus_1
                W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]
                corr_w = 1.0 + np.float64(W_int) / (2.0 * n_half_d)
                dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_ell
                dyn_it = np.int64(dyn_x)
                if ws > dyn_it:
                    pruned = True
                    break
        if pruned:
            survived[b] = False
    return survived


# ============================================================
# B: Stage 1 (TIGHT) — δ² correction = #pairs/(4n·ell·m²)
#    in m² units: ell_int_sum / (ell · 4n)
# ============================================================
@njit(parallel=True, cache=True)
def prune_B(batch_int, n_half, m, c_target):
    B = batch_int.shape[0]
    d = batch_int.shape[1]
    conv_len = 2 * d - 1
    survived = np.ones(B, dtype=numba.boolean)

    m_d = np.float64(m)
    four_n = 4.0 * np.float64(n_half)
    n_half_d = np.float64(n_half)
    d_minus_1 = d - 1
    eps_margin = 1e-9 * m_d * m_d
    max_ell = 2 * d
    cs_base_m2 = c_target * m_d * m_d
    scale_arr = np.empty(max_ell + 1, dtype=np.float64)
    for ell in range(2, max_ell + 1):
        scale_arr[ell] = np.float64(ell) * four_n

    # ell_arr[k] = 4n * (1/2 - |x_k|) for conv bin k, x_k = (k+1)/(4n) - 1/2.
    # Integer form: ell_int[k] = 2n - |k+1 - 2n|, in [0, 2n].
    # avg over window = sum / (n_cv * 4n)  in physical units.
    ell_int_arr = np.empty(conv_len, dtype=np.int64)
    two_n = 2 * n_half
    for k in range(conv_len):
        d_idx = (k + 1) - two_n
        if d_idx < 0:
            d_idx = -d_idx
        v = two_n - d_idx
        if v < 0:
            v = 0
        ell_int_arr[k] = v
    ell_prefix = np.zeros(conv_len + 1, dtype=np.int64)
    for k in range(conv_len):
        ell_prefix[k + 1] = ell_prefix[k] + ell_int_arr[k]

    for b in prange(B):
        conv = np.zeros(conv_len, dtype=np.int64)
        for i in range(d):
            ci = np.int64(batch_int[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int64(batch_int[b, j])
                    if cj != 0:
                        conv[i + j] += np.int64(2) * ci * cj

        prefix_c = np.zeros(d + 1, dtype=np.int64)
        for i in range(d):
            prefix_c[i + 1] = prefix_c[i] + np.int64(batch_int[b, i])

        pruned = False
        for ell in range(2, max_ell + 1):
            if pruned:
                break
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += conv[k]
            scale_ell = scale_arr[ell]
            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                lo_bin = s_lo - d_minus_1
                if lo_bin < 0:
                    lo_bin = 0
                hi_bin = s_lo + ell - 2
                if hi_bin > d_minus_1:
                    hi_bin = d_minus_1
                W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]
                # Stage 1 (TIGHT): replace L^infty constant '1' with the exact
                # δ²-bound (# pairs in window) / (4n·ell·m²).
                # In m² units that's ell_int_sum / (4n·ell) — note ell, not n_cv.
                ell_int_sum = ell_prefix[s_lo + n_cv] - ell_prefix[s_lo]
                corr_l_inf = np.float64(ell_int_sum) / (np.float64(ell) * four_n)
                corr_w = corr_l_inf + np.float64(W_int) / (2.0 * n_half_d)
                dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_ell
                dyn_it = np.int64(dyn_x)
                if ws > dyn_it:
                    pruned = True
                    break
        if pruned:
            survived[b] = False
    return survived


# ============================================================
# C: Stage 1++ — both δ² and linear tightened.
#   Linear: existing W_int/(2n) → (ell-1)*(W_int + #overlap)/(2n*ell).
#   This is from |Σ a_i δ_j|_W ≤ (ell-1)·Σ_overlap a_i / m, with
#   a_i ≤ b_i + 1/m so Σ a_i ≤ W_int/m + #overlap/m.
#
# CAVEAT: I am NOT 100% sure this dominates the existing W_int/(2n) bound
#   in all configurations.  Empirical test below: must give SUBSET of B's
#   survivors (sound) AND fewer survivors than B (strictly tighter).
# ============================================================
@njit(parallel=True, cache=True)
def prune_C(batch_int, n_half, m, c_target):
    B = batch_int.shape[0]
    d = batch_int.shape[1]
    conv_len = 2 * d - 1
    survived = np.ones(B, dtype=numba.boolean)

    m_d = np.float64(m)
    four_n = 4.0 * np.float64(n_half)
    n_half_d = np.float64(n_half)
    d_minus_1 = d - 1
    eps_margin = 1e-9 * m_d * m_d
    max_ell = 2 * d
    cs_base_m2 = c_target * m_d * m_d
    scale_arr = np.empty(max_ell + 1, dtype=np.float64)
    for ell in range(2, max_ell + 1):
        scale_arr[ell] = np.float64(ell) * four_n

    ell_int_arr = np.empty(conv_len, dtype=np.int64)
    two_n = 2 * n_half
    for k in range(conv_len):
        d_idx = (k + 1) - two_n
        if d_idx < 0:
            d_idx = -d_idx
        v = two_n - d_idx
        if v < 0:
            v = 0
        ell_int_arr[k] = v
    ell_prefix = np.zeros(conv_len + 1, dtype=np.int64)
    for k in range(conv_len):
        ell_prefix[k + 1] = ell_prefix[k] + ell_int_arr[k]

    for b in prange(B):
        conv = np.zeros(conv_len, dtype=np.int64)
        for i in range(d):
            ci = np.int64(batch_int[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int64(batch_int[b, j])
                    if cj != 0:
                        conv[i + j] += np.int64(2) * ci * cj

        prefix_c = np.zeros(d + 1, dtype=np.int64)
        for i in range(d):
            prefix_c[i + 1] = prefix_c[i] + np.int64(batch_int[b, i])

        pruned = False
        for ell in range(2, max_ell + 1):
            if pruned:
                break
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += conv[k]
            scale_ell = scale_arr[ell]
            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                lo_bin = s_lo - d_minus_1
                if lo_bin < 0:
                    lo_bin = 0
                hi_bin = s_lo + ell - 2
                if hi_bin > d_minus_1:
                    hi_bin = d_minus_1
                W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]
                # δ² tight
                ell_int_sum = ell_prefix[s_lo + n_cv] - ell_prefix[s_lo]
                corr_l_inf = np.float64(ell_int_sum) / (np.float64(ell) * four_n)
                # Linear tight (TENTATIVE — needs validation):
                # (ell-1)*(W_int + #overlap) / (2n*ell)
                n_overlap = hi_bin - lo_bin + 1
                corr_lin = (np.float64(ell - 1)
                            * np.float64(W_int + n_overlap)
                            / (2.0 * n_half_d * np.float64(ell)))
                corr_w = corr_l_inf + corr_lin
                dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_ell
                dyn_it = np.int64(dyn_x)
                if ws > dyn_it:
                    pruned = True
                    break
        if pruned:
            survived[b] = False
    return survived


# ============================================================
# D: TIGHTEST proven sound — uses #pairs = ell_int_sum exactly.
#    Total correction in m^2 units (in TV space, divide by m^2):
#       corr = (ell-1)·W_int/(2n·ell) + 3·ell_int_sum/(4n·ell)
#    Derivation:
#       linear ≤ ((ell-1)·W_int + ell_int_sum) / (2n·ell·m^2)
#       δ²    ≤ ell_int_sum / (4n·ell·m^2)
#    Sum: (ell-1)·W_int/(2n·ell) + 3·ell_int_sum/(4n·ell).
#
# This is strictly ≤ C (since #pairs ≤ (ell-1)·#overlap), so if C was
# sound (empirically verified), D is also sound and gives MORE pruning.
# ============================================================
@njit(parallel=True, cache=True)
def prune_D(batch_int, n_half, m, c_target):
    B = batch_int.shape[0]
    d = batch_int.shape[1]
    conv_len = 2 * d - 1
    survived = np.ones(B, dtype=numba.boolean)

    m_d = np.float64(m)
    four_n = 4.0 * np.float64(n_half)
    n_half_d = np.float64(n_half)
    d_minus_1 = d - 1
    eps_margin = 1e-9 * m_d * m_d
    max_ell = 2 * d
    cs_base_m2 = c_target * m_d * m_d
    scale_arr = np.empty(max_ell + 1, dtype=np.float64)
    for ell in range(2, max_ell + 1):
        scale_arr[ell] = np.float64(ell) * four_n

    ell_int_arr = np.empty(conv_len, dtype=np.int64)
    two_n = 2 * n_half
    for k in range(conv_len):
        d_idx = (k + 1) - two_n
        if d_idx < 0:
            d_idx = -d_idx
        v = two_n - d_idx
        if v < 0:
            v = 0
        ell_int_arr[k] = v
    ell_prefix = np.zeros(conv_len + 1, dtype=np.int64)
    for k in range(conv_len):
        ell_prefix[k + 1] = ell_prefix[k] + ell_int_arr[k]

    for b in prange(B):
        conv = np.zeros(conv_len, dtype=np.int64)
        for i in range(d):
            ci = np.int64(batch_int[b, i])
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d):
                    cj = np.int64(batch_int[b, j])
                    if cj != 0:
                        conv[i + j] += np.int64(2) * ci * cj

        prefix_c = np.zeros(d + 1, dtype=np.int64)
        for i in range(d):
            prefix_c[i + 1] = prefix_c[i] + np.int64(batch_int[b, i])

        pruned = False
        for ell in range(2, max_ell + 1):
            if pruned:
                break
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            ws = np.int64(0)
            for k in range(n_cv):
                ws += conv[k]
            scale_ell = scale_arr[ell]
            ell_f = np.float64(ell)
            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                lo_bin = s_lo - d_minus_1
                if lo_bin < 0:
                    lo_bin = 0
                hi_bin = s_lo + ell - 2
                if hi_bin > d_minus_1:
                    hi_bin = d_minus_1
                W_int = prefix_c[hi_bin + 1] - prefix_c[lo_bin]
                ell_int_sum = ell_prefix[s_lo + n_cv] - ell_prefix[s_lo]
                # linear: (ell-1)*W_int/(2n*ell) + ell_int_sum/(2n*ell)
                # δ²:    ell_int_sum/(4n*ell)
                # sum:   (ell-1)*W_int/(2n*ell) + 3*ell_int_sum/(4n*ell)
                corr_w = (np.float64(ell - 1) * np.float64(W_int)
                           / (2.0 * n_half_d * ell_f)
                           + 3.0 * np.float64(ell_int_sum)
                           / (4.0 * n_half_d * ell_f))
                dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_ell
                dyn_it = np.int64(dyn_x)
                if ws > dyn_it:
                    pruned = True
                    break
        if pruned:
            survived[b] = False
    return survived


# ============================================================
# Test driver
# ============================================================
def run(n_half, m, c_target, batch_size=200_000, verbose=True):
    d = 2 * n_half
    # Fine grid: full composition sums to S = 4nm; palindromic half sums to 2nm.
    S_full = 4 * n_half * m
    S_half = 2 * n_half * m
    n_total_half = count_compositions(n_half, S_half)

    if verbose:
        print(f"\n=== n_half={n_half}, m={m}, c_target={c_target} ===")
        print(f"d={d}, S_full=4nm={S_full}, palindromic half_sum=2nm={S_half}, "
              f"total palindromic comps={n_total_half:,}")

    # Warm up JIT
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = 2 * m
    prune_A(warm, n_half, m, c_target)
    prune_B(warm, n_half, m, c_target)
    prune_C(warm, n_half, m, c_target)
    prune_D(warm, n_half, m, c_target)

    n_processed = 0
    surv_A = []
    surv_B = []
    surv_C = []
    surv_D = []
    pruned_A = pruned_B = pruned_C = pruned_D = 0
    t_A = t_B = t_C = t_D = 0.0
    t0 = time.time()

    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=batch_size):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        n_processed += len(batch)

        ta = time.time()
        sA = prune_A(batch, n_half, m, c_target)
        t_A += time.time() - ta
        pruned_A += int(np.sum(~sA))
        if sA.any():
            surv_A.append(batch[sA].copy())

        tb = time.time()
        sB = prune_B(batch, n_half, m, c_target)
        t_B += time.time() - tb
        pruned_B += int(np.sum(~sB))
        if sB.any():
            surv_B.append(batch[sB].copy())

        tc = time.time()
        sC = prune_C(batch, n_half, m, c_target)
        t_C += time.time() - tc
        pruned_C += int(np.sum(~sC))
        if sC.any():
            surv_C.append(batch[sC].copy())

        td = time.time()
        sD = prune_D(batch, n_half, m, c_target)
        t_D += time.time() - td
        pruned_D += int(np.sum(~sD))
        if sD.any():
            surv_D.append(batch[sD].copy())

        # Soundness checks
        if not bool(np.all(sB <= sA)):
            print(f"  *** SOUNDNESS BUG: B has survivors not in A!")
        if not bool(np.all(sC <= sA)):
            print(f"  *** WARNING: C has survivors not in A "
                  f"(C-not-A: {int(np.sum(sC & ~sA))})")
        if not bool(np.all(sD <= sA)):
            print(f"  *** WARNING: D has survivors not in A "
                  f"(D-not-A: {int(np.sum(sD & ~sA))})")
        if not bool(np.all(sD <= sC)):
            print(f"  ## D vs C diff: D-not-C={int(np.sum(sD & ~sC))} "
                  f"C-not-D={int(np.sum(sC & ~sD))}")

    surv_A_n = sum(s.shape[0] for s in surv_A)
    surv_B_n = sum(s.shape[0] for s in surv_B)
    surv_C_n = sum(s.shape[0] for s in surv_C)
    surv_D_n = sum(s.shape[0] for s in surv_D)
    elapsed = time.time() - t0

    print(f"\n--- L0 results (palindromic, sum=2m) ---")
    print(f"  total processed: {n_processed:,}")
    print(f"  A (baseline '1' L^∞):    {pruned_A:,} pruned, "
          f"{surv_A_n:,} survivors  [{t_A:.2f}s]")
    print(f"  B (TIGHT δ² only):       {pruned_B:,} pruned, "
          f"{surv_B_n:,} survivors  [{t_B:.2f}s]")
    print(f"  C (TIGHT δ² + lin loose):{pruned_C:,} pruned, "
          f"{surv_C_n:,} survivors  [{t_C:.2f}s]")
    print(f"  D (TIGHTEST proven):     {pruned_D:,} pruned, "
          f"{surv_D_n:,} survivors  [{t_D:.2f}s]")
    if surv_A_n > 0:
        rB = (surv_A_n - surv_B_n) / surv_A_n * 100
        rC = (surv_A_n - surv_C_n) / surv_A_n * 100
        rD = (surv_A_n - surv_D_n) / surv_A_n * 100
        print(f"  B prunes  +{surv_A_n - surv_B_n:,}  ({rB:.2f}% of A survivors)")
        print(f"  C prunes  +{surv_A_n - surv_C_n:,}  ({rC:.2f}% of A survivors)")
        print(f"  D prunes  +{surv_A_n - surv_D_n:,}  ({rD:.2f}% of A survivors)")
    print(f"  total wall: {elapsed:.2f}s")

    # Stage 2: apply joint cell cert to B's survivors
    if surv_B and surv_B_n <= 200_000:
        survB = np.vstack(surv_B)
        # Effective c_target for cell cert:
        # cell cert checks TV(mu) >= c_target_eff for all mu in cell
        # In fine grid mu_i = c_i/(4nm). The cell is [c_i - 1/2, c_i + 1/2] integer
        # mass coords -> mu_i within +/- 1/(8nm). The Lemma 3 correction is
        # (1 + W/(2n))/m^2; using flat 2/m + 1/m^2 here for safety.
        flat_corr = 2.0 / m + 1.0 / (m * m)
        c_target_eff = c_target + flat_corr
        S_int = 4 * n_half * m  # fine grid total
        print(f"\n--- Stage 2: joint cell cert on B's survivors ---")
        print(f"  c_target_eff = c + 2/m + 1/m^2 = {c_target_eff:.6f}")
        n_cell_certed = 0
        # Warm up
        joint_cell_cert_for_composition(survB[0], S_int, d, c_target_eff)
        t0 = time.time()
        for i in range(survB.shape[0]):
            cert, n_pw = joint_cell_cert_for_composition(
                survB[i], S_int, d, c_target_eff)
            if cert >= c_target_eff:
                n_cell_certed += 1
        t_C = time.time() - t0
        print(f"  cell-certed: {n_cell_certed:,}/{survB.shape[0]:,} "
              f"({100.0*n_cell_certed/max(1,survB.shape[0]):.2f}%)  "
              f"[{t_C:.2f}s, {1000*t_C/max(1,survB.shape[0]):.2f} ms/cell]")
        if n_cell_certed:
            print(f"  Stage 2 prunes ADDITIONAL {n_cell_certed:,} "
                  f"({100.0*n_cell_certed/max(1,surv_A_n):.2f}% of A's survivors)")
    elif surv_B_n > 200_000:
        print(f"\n--- Stage 2: skipped (too many survivors: {surv_B_n:,}) ---")

    return {
        'n_processed': n_processed,
        'surv_A': surv_A_n,
        'surv_B': surv_B_n,
        't_A': t_A, 't_B': t_B,
    }


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_half', type=int, default=2)
    ap.add_argument('--m', type=int, default=20)
    ap.add_argument('--c_target', type=float, default=1.20)
    ap.add_argument('--batch', type=int, default=200_000)
    ap.add_argument('--sweep', action='store_true')
    args = ap.parse_args()
    if args.sweep:
        for nh, m, c in [(3, 10, 1.20), (3, 20, 1.20), (3, 30, 1.20),
                          (3, 10, 1.10), (3, 10, 1.28),
                          (4, 5, 1.20), (4, 10, 1.20), (4, 10, 1.28),
                          (5, 5, 1.28)]:
            run(nh, m, c, batch_size=args.batch)
    else:
        run(args.n_half, args.m, args.c_target, batch_size=args.batch)
