"""Variant FN: F's tight LP linear + N's restricted-spectrum δ² bound.

==============================================================================
DERIVATION (sound):
==============================================================================
Setup (from _M1_bench.py:prune_F):
  c integer composition; b_i = c_i/m; a continuous heights with |b_i - a_i|
  ≤ 1/m, Σ a = Σ b. Define δ = b - a, so |δ_i| ≤ h := 1/m and Σ δ_i = 0.

Per window W (set of conv positions in [s_lo, s_lo+ell-2]):
  TV_W(b) - TV_W(a) = (1/(4n·ell)) · [2·Σ_{(i,j)∈W} b_i δ_j - Σ_{(i,j)∈W} δ_i δ_j]
                    = (1/(4n·ell)) · [2·δ^T B - δ^T A_W δ]
where B_j := Σ_{i:(i,j)∈W} b_i and A_W is the indicator matrix of W
(symmetric, since W is symmetric in (i,j)).

LINEAR (already TIGHT in F): max_{|δ|≤h, Σδ=0} 2·δ^T B = 2h·Δ_BB/m
                            where Δ_BB = sort+extremes split of B (in m units).

QUADRATIC (the LOOSE part in F): bound |δ^T A_W δ|.
  F uses: |δ^T A_W δ| ≤ Σ_{(i,j)∈W} |δ_i δ_j| ≤ #pairs · h² = ell_int_sum · h².

  N's improvement (restricted-spectrum):
    Since Σ δ = 0, ANY α ∈ R gives δ^T(A_W - α·11ᵀ)δ = δ^T A_W δ
    (because δ^T(α·11ᵀ)δ = α·(Σδ)² = 0).
    Hence |δ^T A_W δ| ≤ ‖A_W - α·11ᵀ‖_op · ‖δ‖₂²
                       ≤ op_restricted · d · h²    (‖δ‖₂² ≤ d·h²).
    Choose α = (Σ A_W)/d² = n_pairs/d² to minimize ‖A_W - α·11ᵀ‖_op
    (this projects out the all-ones eigenvector that dominates ‖A_W‖_op).

VARIANT FN: combine F's tight linear with N's tight δ²:
  corr_FN (m² units) = Δ_BB/(2n·ell) + min(op_restricted·d, ell_int_sum) / (4n·ell)

  The min ensures FN ≤ F per-window (sound regression: FN cannot be looser).

==============================================================================
SOUNDNESS PROOF (formal, for ANY δ with Σδ=0, |δ|_∞ ≤ h):
==============================================================================
Let A := A_W (real symmetric, n×n with n=d), and α := (Σ A)/d² scalar.
Let M := A - α·11ᵀ.  M is real symmetric (since 11ᵀ is symmetric & A is).
Let λ_max := op_restricted := max_i |λ_i(M)|.

For any δ with Σδ=0: 1ᵀδ = 0, so 11ᵀδ = 0, so δ^T(α·11ᵀ)δ = α·(1ᵀδ)² = 0.
Therefore δ^T A δ = δ^T M δ.

By the spectral theorem (M symmetric):
  |δ^T M δ| ≤ λ_max(|M|) · ‖δ‖₂² = op_restricted · ‖δ‖₂².

Under |δ|_∞ ≤ h:  ‖δ‖₂² = Σ δ_i² ≤ d · h².

Combining: |δ^T A_W δ| ≤ op_restricted · d · h².  ∎

This bound holds for ALL δ in the feasible set, hence variant N (and FN, since
FN takes min with F's bound which is already sound) is rigorously sound.

==============================================================================
COMPARISON to variant F:
==============================================================================
  F's δ²:  ell_int_sum · h²
  FN's δ²: min(op_restricted · d, ell_int_sum) · h²

  We always have op_restricted · d ≤ Σ_i |λ_i(M)| · 1 (loose) and ‖M‖_op
  is bounded by Frobenius: ‖M‖_op ≤ ‖M‖_F.  ‖M‖_F² = Σ A²_ij - 2α·Σ A_ij/d²·d²
                                            = n_pairs - n_pairs²/d² ≤ n_pairs.
  But this only gives op_rest ≤ √n_pairs, so op_rest·d ≤ d·√n_pairs which
  is NOT smaller than n_pairs in general.

  EMPIRICALLY (per _push_bench.py): op_restricted·d < ell_int_sum for many
  windows, especially "long" ones where the indicator pattern is rich.

==============================================================================
NOTE on push 3 (joint LP-extreme): the QP-at-LP-vertex approach is NOT proven
sound for indefinite A_W (since max |2L−Q| at vertices ≤ max over feasible),
so it is omitted from the sound kernel below.
==============================================================================
"""
import os, sys, time, json
import numpy as np
import numba
from numba import njit, prange

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, _dir)

from compositions import generate_compositions_batched
from pruning import count_compositions
from _M1_bench import prune_F
from _N_bench import precompute_op_norm_restricted


# ============================================================
# Soundness self-test: for random δ with Σδ=0 and |δ|_∞ ≤ h,
# verify  |δ^T A_W δ| ≤ op_restricted · d · h²  in many trials.
# ============================================================
def _sanity_spectral(d=8, n_trials=300, seed=0):
    """Random δ with Σδ=0, |δ|_∞≤1; verify bound holds for various A_W."""
    rng = np.random.default_rng(seed)
    conv_len = 2 * d - 1
    max_ell = 2 * d
    op_rest, n_pairs_arr = precompute_op_norm_restricted(d, max_ell, conv_len)

    n_violations = 0
    n_checked = 0
    n_strict_better = 0  # cases where op_rest*d < n_pairs (FN tighter than F)
    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        for s_lo in range(n_windows):
            n_pairs = int(n_pairs_arr[ell, s_lo])
            if n_pairs == 0:
                continue
            # Build A_W
            A = np.zeros((d, d), dtype=np.float64)
            for i in range(d):
                for j in range(d):
                    if s_lo <= i + j <= s_lo + ell - 2:
                        A[i, j] = 1.0
            op_d_h2 = op_rest[ell, s_lo] * d  # h=1 for sanity
            if op_d_h2 < n_pairs - 1e-9:
                n_strict_better += 1
            for _ in range(n_trials):
                # Random δ in [-1, 1]^d, then center to get Σδ=0
                delta = rng.uniform(-1.0, 1.0, size=d)
                delta = delta - delta.mean()
                # Rescale so |δ|_∞ ≤ 1 (h=1 for sanity)
                if np.abs(delta).max() > 1.0:
                    delta = delta / np.abs(delta).max()
                qf = float(delta @ A @ delta)
                bound = op_rest[ell, s_lo] * (delta @ delta)  # ‖δ‖₂² ≤ d·h²
                if abs(qf) > bound + 1e-9:
                    n_violations += 1
                    if n_violations <= 5:
                        print(f"   *** VIOLATION ell={ell} s_lo={s_lo}: "
                              f"|qf|={abs(qf):.4f} > bound={bound:.4f}")
                n_checked += 1
    print(f"  sanity (d={d}): {n_checked:,} random δ tested, "
          f"{n_violations} violations, {n_strict_better} (ell,s_lo) windows "
          f"where op_rest·d < n_pairs (FN strictly tighter than F).")
    return n_violations == 0


# ============================================================
# variant FN kernel (sound: min(op_rest*d, ell_int_sum) for δ²)
# ============================================================
@njit(parallel=True, cache=True)
def prune_FN(batch_int, n_half, m, c_target, ell_prefix, op_rest_d_arr):
    """Variant FN: F's tight LP linear + N's restricted-spectrum δ².

    Per-window correction (m² units):
      Δ_BB/(2n·ell) + min(op_rest·d, ell_int_sum) / (4n·ell)

    Soundness: both bounds on the δ² term are individually sound; the min
    is therefore sound and ≤ F's δ² bound.
    """
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

    half_d = d // 2

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
                s_hi = s_lo + ell - 2

                # BB[j] for this window
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

                # Δ_BB: sort+extremes
                BB_sorted = np.sort(BB)
                sum_top = np.int64(0)
                for k in range(half_d, d):
                    sum_top += BB_sorted[k]
                sum_bot = np.int64(0)
                for k in range(half_d):
                    sum_bot += BB_sorted[k]
                delta_BB = sum_top - sum_bot

                ell_int_sum = ell_prefix[s_lo + n_cv] - ell_prefix[s_lo]

                # δ² term: min of F's bound (ell_int_sum) and N's (op_rest*d)
                op_d = op_rest_d_arr[ell, s_lo]  # = op_rest * d, precomputed
                ell_int_f = np.float64(ell_int_sum)
                if op_d < ell_int_f:
                    delta_sq = op_d
                else:
                    delta_sq = ell_int_f

                corr_w = (np.float64(delta_BB) / (2.0 * n_half_d * ell_f)
                          + delta_sq / (4.0 * n_half_d * ell_f))
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
    S_full = 4 * n_half * m
    S_half = 2 * n_half * m
    n_total = count_compositions(n_half, S_half)

    if verbose:
        print(f"\n=== n_half={n_half}, m={m}, c_target={c_target} ===")
        print(f"     d={d}, palindromic comps={n_total:,}")

    conv_len = 2 * d - 1
    max_ell = 2 * d

    # Build ell_prefix (matches _M1_bench.prune_F's construction)
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

    print(f"     precomputing op-norm restricted...", end=' ', flush=True)
    t_pre = time.time()
    op_rest, n_pairs_arr = precompute_op_norm_restricted(d, max_ell, conv_len)
    op_rest_d = op_rest * d
    print(f"[{time.time()-t_pre:.2f}s]")

    # Sanity: confirm n_pairs_arr == ell_int_sum at each (ell, s_lo)
    mismatch = 0
    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        for s_lo in range(n_windows):
            ellsum = int(ell_prefix[s_lo + n_cv] - ell_prefix[s_lo])
            np_val = int(n_pairs_arr[ell, s_lo])
            if ellsum != np_val:
                mismatch += 1
    if mismatch:
        print(f"     *** ell_int_sum vs n_pairs mismatch in {mismatch} windows!")
    else:
        print(f"     ell_int_sum vs n_pairs: identical (consistency OK)")

    # How often is op_rest*d strictly < n_pairs?  (FN strictly tighter than F)
    n_strict = 0
    n_total_w = 0
    sum_ratio = 0.0
    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        for s_lo in range(n_windows):
            np_val = int(n_pairs_arr[ell, s_lo])
            if np_val == 0:
                continue
            n_total_w += 1
            if op_rest_d[ell, s_lo] < np_val - 1e-9:
                n_strict += 1
                sum_ratio += op_rest_d[ell, s_lo] / np_val
    avg_ratio = sum_ratio / max(1, n_strict)
    print(f"     spectral δ² strictly tighter in {n_strict}/{n_total_w} "
          f"windows (avg ratio when tighter: {avg_ratio:.4f})")

    # Warm JIT
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = 2 * m
    prune_F(warm, n_half, m, c_target)
    prune_FN(warm, n_half, m, c_target, ell_prefix, op_rest_d)

    n_proc = 0
    surv_F = 0
    surv_FN = 0
    n_violations = 0  # FN-survivors not in F (would be a soundness bug)
    extra_FN_over_F = 0
    t_F = t_FN = 0.0
    t0 = time.time()

    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=batch_size):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        n_proc += len(batch)

        ta = time.time()
        sF = prune_F(batch, n_half, m, c_target)
        t_F += time.time() - ta

        tb = time.time()
        sFN = prune_FN(batch, n_half, m, c_target, ell_prefix, op_rest_d)
        t_FN += time.time() - tb

        surv_F += int(sF.sum())
        surv_FN += int(sFN.sum())
        n_violations += int(np.sum(sFN & ~sF))
        extra_FN_over_F += int(np.sum(~sFN & sF))

    elapsed = time.time() - t0
    pct_extra = 100.0 * extra_FN_over_F / max(1, surv_F)

    print(f"\n     === Survivor counts ===")
    print(f"     F:  {surv_F:,} survivors  [{t_F:.2f}s]")
    print(f"     FN: {surv_FN:,} survivors  [{t_FN:.2f}s]")
    print(f"     FN prunes ADDITIONAL {extra_FN_over_F:,} = "
          f"{pct_extra:.3f}% of F-survivors")
    if n_violations > 0:
        print(f"     *** SOUNDNESS BUG: {n_violations} FN survivors not in F ***")
    print(f"     wall: {elapsed:.2f}s")

    return {
        'n_half': n_half, 'm': m, 'c_target': c_target,
        'n_processed': n_proc,
        'surv_F': surv_F, 'surv_FN': surv_FN,
        'extra_FN_over_F': extra_FN_over_F, 'pct_extra': pct_extra,
        'n_soundness_violations': n_violations,
        'n_windows': n_total_w, 'n_windows_FN_tighter': n_strict,
        'avg_ratio_when_tighter': avg_ratio,
        't_F': t_F, 't_FN': t_FN, 'elapsed': elapsed,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--sanity', action='store_true',
                     help='Run spectral sanity (slow); skip benchmark.')
    ap.add_argument('--out', type=str, default='_FN_bench.json')
    ap.add_argument('--quick', action='store_true',
                     help='Quick subset (skip largest configs).')
    args = ap.parse_args()

    print("=== FN bench: sanity check (spectral bound on random δ) ===")
    sane = _sanity_spectral(d=8, n_trials=200)
    if not sane:
        print("  *** sanity failed; aborting ***")
        return
    if args.sanity:
        return

    if args.quick:
        configs = [
            (3, 10, 1.28),
            (3, 20, 1.28),
            (4, 10, 1.28),
            (5, 5, 1.28),
        ]
    else:
        configs = [
            (3, 10, 1.28),
            (3, 20, 1.28),
            (4, 10, 1.28),
            (5, 5, 1.28),
            (6, 5, 1.28),
        ]

    results = []
    for nh, m, c in configs:
        try:
            r = run(nh, m, c)
            results.append(r)
        except Exception as e:
            print(f"  *** ERROR ({nh},{m},{c}): {e}")
            results.append({'n_half': nh, 'm': m, 'c_target': c,
                            'error': str(e)})

    out_path = os.path.join(_dir, args.out)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"\n{'='*70}\nSummary:")
    for r in results:
        if 'error' in r:
            print(f"  ({r['n_half']},{r['m']},{r['c_target']}): ERROR")
            continue
        print(f"  (n={r['n_half']},m={r['m']},c={r['c_target']}): "
              f"F={r['surv_F']:,} → FN={r['surv_FN']:,}  "
              f"(+{r['pct_extra']:.3f}% extra prune, "
              f"{r['n_soundness_violations']} violations)")


if __name__ == '__main__':
    main()
