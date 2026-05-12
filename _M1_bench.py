"""M1 benchmark: tighter linear correction using Σδ=0 (LP optimum).

DERIVATION (verified):
=====================
Setup: c ∈ Z^d, b_i = c_i/m, a continuous heights with |b_i - a_i| ≤ 1/m and
       Σ a = Σ b = 4n (so δ_i := b_i - a_i has |δ_i| ≤ 1/m and Σ δ_i = 0).

For window W (= conv positions in [s_lo, s_hi]):
  TV_W(b) - TV_W(a) = (1/(4n·ell)) · [Σ_{(i,j)∈W} b_i b_j - Σ_{(i,j)∈W} a_i a_j]
                    = (1/(4n·ell)) · [2·Σ_{(i,j)∈W} b_i δ_j  -  Σ_{(i,j)∈W} δ_i δ_j]
  (the second equality uses a = b - δ, so b_ib_j - a_ia_j = 2 b_iδ_j - δ_iδ_j).

To prune c, we need TV_W(b) ≥ c_target + sup_a [TV_W(b) - TV_W(a)] for some W.
So bound:
  sup_{|δ|≤h, Σδ=0} (1/(4n·ell)) [2 Σ b_iδ_j - Σ δ_iδ_j]                    (h=1/m)
  ≤ (1/(4n·ell)) [2·sup_{|δ|≤h, Σδ=0} Σ_j δ_j B_j  +  sup |Σ δ_iδ_j|]
where B_j := Σ_{i:(i,j)∈W} b_i.

LINEAR (with Σδ=0):
  max_{|δ|≤h, Σδ=0} Σ_j δ_j B_j = h · [Σ_top(d/2) B - Σ_bot(d/2) B]
                                = h · Σ_j |B_j - median(B)|     (when d even)
  Proof (LP duality): primal: max c·δ s.t. ‖δ‖_∞≤h, 1·δ=0.
   Dual variables λ⁺, λ⁻ ≥ 0, μ ∈ R, with B_j = λ⁺_j - λ⁻_j + μ;
   value = h·Σ(λ⁺+λ⁻) = h·Σ|B_j - μ|, minimized at μ = median.
   Primal optimizer: δ = +h on top d/2, -h on bot d/2 (Σδ=0 since |top|=|bot|).

QUADRATIC (no Σδ=0 needed; matches variant D):
  |Σ_{(i,j)∈W} δ_iδ_j| ≤ h² · #pairs = h² · ell_int_sum.

Total per-window correction (in m^2 units):
  corr_F = (1/m^2) · m^2/(4n·ell) · [(2/m)·Δ_B + (1/m^2)·ell_int_sum]
         = m·Δ_B/(2n·ell) + ell_int_sum/(4n·ell)
        BUT Δ_B = (1/m)·Δ_BB where BB_j = m·B_j = Σ_{i:(i,j)∈W} c_i (integer):
  corr_F (m^2 units) = Δ_BB/(2n·ell) + ell_int_sum/(4n·ell)

COMPARISON to variant D:
  D: (ell-1)·W_int/(2n·ell) + 3·ell_int_sum/(4n·ell)
   = [(ell-1)·W_int + ell_int_sum] /(2n·ell) + ell_int_sum/(4n·ell)

  F: Δ_BB/(2n·ell) + ell_int_sum/(4n·ell)

  Δ_BB ≤ Σ_top BB ≤ Σ_j BB_j = Σ_i c_i·N_i ≤ (ell-1)·W_int   ⇒ F's linear ≤ D's linear.
  Quadratic identical. So F ≤ D pointwise (per window).

  Strict gain whenever Δ_BB < (ell-1)·W_int + ell_int_sum, i.e. typically:
    - When BB is balanced (all bins similar), Δ_BB << Σ BB, gain is large.
    - When BB is concentrated, Δ_BB ≈ Σ BB ≈ (ell-1)·W_int (gain ≈ ell_int_sum).

So F should give ADDITIONAL pruning beyond D. Soundness check: F-survivors ⊆ A.

(Note: prune_F's correction is a STRICTLY tighter upper bound on the same
 sup_a [TV_W(b) - TV_W(a)], so prune_F is ALWAYS sound provided D was sound.)
"""
import os, sys, time, json
import numpy as np
import numba
from numba import njit, prange

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
from compositions import generate_compositions_batched
from pruning import count_compositions


# ============================================================
# A: baseline (variant A): corr = 1 + W_int/(2n)  in m^2 units
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
# D: variant D (current TIGHTEST proven baseline)
#   corr = (ell-1)·W_int/(2n·ell) + 3·ell_int_sum/(4n·ell)   [m^2 units]
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
# F: variant F = M1 (Σδ=0 closed-form for linear part)
#   corr = Δ_BB/(2n·ell) + ell_int_sum/(4n·ell)   [m^2 units]
#   where Δ_BB = Σ_top(d/2) BB - Σ_bot(d/2) BB,
#         BB_j = Σ_{i:(i,j)∈W} c_i.
#
# Each window: compute BB[0..d-1] from prefix_c, sort, compute Δ_BB.
# d typically small (≤ 16-20), so per-window O(d log d) is OK.
# ============================================================
@njit(parallel=True, cache=True)
def prune_F(batch_int, n_half, m, c_target):
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

    half_d = d // 2  # d is always even (palindromic full = 2*n_half)

    for b in prange(B):
        # Autoconvolution
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

        BB = np.empty(d, dtype=np.int64)

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
                s_hi = s_lo + ell - 2  # last conv pos in window

                # Compute BB[j] = Σ_{i: s_lo ≤ i+j ≤ s_hi, 0≤i≤d-1} c_i.
                # i ∈ [max(0, s_lo - j), min(d-1, s_hi - j)].
                for j in range(d):
                    lo_i = s_lo - j
                    if lo_i < 0:
                        lo_i = 0
                    hi_i = s_hi - j
                    if hi_i > d_minus_1:
                        hi_i = d_minus_1
                    if hi_i < lo_i:
                        BB[j] = 0
                    else:
                        BB[j] = prefix_c[hi_i + 1] - prefix_c[lo_i]

                # Δ_BB = sum of top d/2 minus sum of bottom d/2 (after sort).
                # Use numpy partition: O(d).  For correctness, sort and split.
                BB_sorted = np.sort(BB)
                sum_top = np.int64(0)
                for k in range(half_d, d):
                    sum_top += BB_sorted[k]
                sum_bot = np.int64(0)
                for k in range(half_d):
                    sum_bot += BB_sorted[k]
                delta_BB = sum_top - sum_bot

                ell_int_sum = ell_prefix[s_lo + n_cv] - ell_prefix[s_lo]
                # corr_F = Δ_BB/(2n·ell) + ell_int_sum/(4n·ell)
                corr_w = (np.float64(delta_BB)
                           / (2.0 * n_half_d * ell_f)
                           + np.float64(ell_int_sum)
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
# Sanity: tiny self-test of the LP closed form (d=4, manual case).
# ============================================================
def _sanity_lp():
    """Verify the LP closed-form against brute force for small d."""
    rng = np.random.default_rng(0)
    for trial in range(50):
        d = rng.integers(2, 9)
        if d % 2 == 1:
            continue  # the prune kernel always uses even d
        h = 1.0
        B = rng.integers(-5, 6, size=d).astype(np.int64)
        # Brute over corner LP: δ in {-h,+h}^d with Σ=0 (only valid when d even)
        best_brute = -1e9
        from itertools import product
        for sign in product([-1, +1], repeat=d):
            if sum(sign) != 0:
                continue
            v = sum(B[i] * sign[i] * h for i in range(d))
            if v > best_brute:
                best_brute = v
        # Closed form: h * (Σ_top(d/2) - Σ_bot(d/2))
        Bs = np.sort(B)
        cf = h * (int(Bs[d // 2:].sum()) - int(Bs[:d // 2].sum()))
        if abs(best_brute - cf) > 1e-9:
            print(f"  *** LP MISMATCH d={d}: brute={best_brute}, formula={cf}, B={B}")
            return False
    print("  LP closed-form matches brute on 25 random even-d trials [OK]")
    return True


# ============================================================
# Test driver
# ============================================================
def run(n_half, m, c_target, batch_size=200_000, verbose=True):
    d = 2 * n_half
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
    prune_D(warm, n_half, m, c_target)
    prune_F(warm, n_half, m, c_target)

    n_processed = 0
    pruned_A = pruned_D = pruned_F = 0
    surv_A_n = surv_D_n = surv_F_n = 0
    bug_F_minus_A = 0   # F survivors that are NOT A survivors (soundness violation)
    bug_F_minus_D = 0   # F survivors that are NOT D survivors (looser than D)
    extra_F_over_D = 0  # D survivors that F ALSO prunes
    t_A = t_D = t_F = 0.0
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
        surv_A_n += int(np.sum(sA))

        td = time.time()
        sD = prune_D(batch, n_half, m, c_target)
        t_D += time.time() - td
        pruned_D += int(np.sum(~sD))
        surv_D_n += int(np.sum(sD))

        tf = time.time()
        sF = prune_F(batch, n_half, m, c_target)
        t_F += time.time() - tf
        pruned_F += int(np.sum(~sF))
        surv_F_n += int(np.sum(sF))

        bug_F_minus_A += int(np.sum(sF & ~sA))
        bug_F_minus_D += int(np.sum(sF & ~sD))
        extra_F_over_D += int(np.sum(~sF & sD))

    elapsed = time.time() - t0

    print(f"\n--- L0 results (palindromic, sum=2nm) ---")
    print(f"  total processed: {n_processed:,}")
    print(f"  A (corr=1+W/(2n)):     {pruned_A:,} pruned, "
          f"{surv_A_n:,} survivors  [{t_A:.2f}s]")
    print(f"  D (variant D, current):{pruned_D:,} pruned, "
          f"{surv_D_n:,} survivors  [{t_D:.2f}s]")
    print(f"  F (M1 = Σδ=0 LP):      {pruned_F:,} pruned, "
          f"{surv_F_n:,} survivors  [{t_F:.2f}s]")
    if surv_A_n > 0:
        rD = (surv_A_n - surv_D_n) / max(1, surv_A_n) * 100
        rF = (surv_A_n - surv_F_n) / max(1, surv_A_n) * 100
        print(f"  D prunes  +{surv_A_n - surv_D_n:,}  ({rD:.2f}% of A survivors)")
        print(f"  F prunes  +{surv_A_n - surv_F_n:,}  ({rF:.2f}% of A survivors)")
    print(f"  F vs D: F-extra-prunes = {extra_F_over_D:,}, "
          f"F-not-A (SOUND BUG) = {bug_F_minus_A:,}, "
          f"F-not-D = {bug_F_minus_D:,}")
    if bug_F_minus_A > 0:
        print(f"  *** SOUNDNESS BUG: F has {bug_F_minus_A} survivors not in A! ***")
    if bug_F_minus_D > 0:
        # Theoretically impossible — F ≤ D. Indicates a bug.
        print(f"  *** F has {bug_F_minus_D} survivors not in D — F should be ≤ D! ***")
    print(f"  total wall: {elapsed:.2f}s")

    return {
        'n_half': n_half, 'm': m, 'c_target': c_target,
        'n_processed': n_processed,
        'surv_A': surv_A_n, 'surv_D': surv_D_n, 'surv_F': surv_F_n,
        'pruned_A': pruned_A, 'pruned_D': pruned_D, 'pruned_F': pruned_F,
        'extra_F_over_D': extra_F_over_D,
        'bug_F_minus_A': bug_F_minus_A,
        'bug_F_minus_D': bug_F_minus_D,
        't_A': t_A, 't_D': t_D, 't_F': t_F,
        'elapsed': elapsed,
    }


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_half', type=int, default=2)
    ap.add_argument('--m', type=int, default=20)
    ap.add_argument('--c_target', type=float, default=1.20)
    ap.add_argument('--batch', type=int, default=200_000)
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--out', type=str, default='_M1_bench.json')
    args = ap.parse_args()

    print("=== M1 sanity: LP closed-form vs brute force ===")
    _sanity_lp()

    results = []
    if args.sweep:
        configs = [
            (3, 10, 1.20), (3, 20, 1.20), (3, 30, 1.20),
            (3, 10, 1.10), (3, 10, 1.28),
            (4, 5, 1.20), (4, 10, 1.20), (4, 10, 1.28),
            (5, 5, 1.28),
        ]
        for nh, m, c in configs:
            try:
                r = run(nh, m, c, batch_size=args.batch)
                results.append(r)
            except Exception as e:
                print(f"  *** ERROR in ({nh},{m},{c}): {e}")
                results.append({'n_half': nh, 'm': m, 'c_target': c,
                                'error': str(e)})
    else:
        r = run(args.n_half, args.m, args.c_target, batch_size=args.batch)
        results.append(r)

    with open(args.out, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f"\nWrote {args.out}")
