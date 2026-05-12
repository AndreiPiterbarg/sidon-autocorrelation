"""Three pushes beyond variant F.  Empirical-only, no formal proofs.

F (current best, m² units):
   corr_F = Δ_BB/(2n·ell) + ell_int_sum/(4n·ell)
   linear: tight LP (sort+extremes) under {|δ|≤h, Σδ=0}
   δ²:     #pairs · h²  (loose: doesn't exploit Σδ=0 or A_W structure)

Push J (sound): a≥0 ⇒ δ_j ∈ [-h, 0] for empty bins (c_j=0).  Re-solve LP
                with these per-bin constraints.  Tighter linear when many
                empty bins.

Push N (sound): bound δ²-term via restricted-spectrum
                |δ^T A_W δ| = |δ^T (A_W − α·11ᵀ) δ|  (since Σδ=0)
                ≤ ‖A_W − α·11ᵀ‖_op · ‖δ‖₂²  ≤ ‖A_W − α·11ᵀ‖_op · d · h²
                where α = (Σ A_W)/d² makes A_W−α·11ᵀ row-sum-0 and removes
                the dominant Perron eigenvector.

Push H (heuristic, NOT PROVEN SOUND): evaluate |2L(δ)−Q(δ)| at LP-vertices
        (sort+extremes), take max.  If empirically << F, F is loose.

For each variant, evaluate the BOUND (not just survivor count) over a
sample of compositions; report ratio vs F.
"""
import os, sys, time, json
import numpy as np
from itertools import combinations
from scipy.optimize import linprog

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, _dir)

from compositions import generate_compositions_batched
from pruning import count_compositions
from _M1_bench import prune_F


# ─────────── Window structure utilities ───────────
def build_A_W(d, s_lo, ell):
    """A_W[i][j] = 1 if i+j ∈ [s_lo, s_lo+ell-2]."""
    A = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            k = i + j
            if s_lo <= k <= s_lo + ell - 2:
                A[i, j] = 1.0
    return A


def precompute_windows(d, conv_len, max_ell):
    """For each (ell, s_lo), return:
       - n_pairs (int): Σ A_W
       - op_norm_restricted (float): ‖A_W − (n_pairs/d²)·11ᵀ‖_op
       - A_W (matrix, for vertex enumeration)
    """
    out = {}
    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        if n_windows <= 0:
            continue
        for s_lo in range(n_windows):
            A_W = build_A_W(d, s_lo, ell)
            n_pairs = int(A_W.sum())
            if n_pairs == 0:
                op = 0.0
            else:
                alpha = n_pairs / (d * d)
                A_rest = A_W - alpha
                eigs = np.linalg.eigvalsh(A_rest)
                op = float(np.abs(eigs).max())
            out[(ell, s_lo)] = {
                'A_W': A_W,
                'n_pairs': n_pairs,
                'op_norm_restricted': op,
            }
    return out


# ─────────── Bound computations per (composition, window) ───────────
def F_bound(c, ell, s_lo, n, m, win_info):
    """corr_F (m² units) for one (composition, window)."""
    d = len(c)
    h = 1.0  # in units of 1/m, so actual h_phys = 1/m, but we'll add 1/m² scaling at end
    A_W = win_info['A_W']
    BB = A_W @ c.astype(np.float64)  # BB_j = Σ_{i:(i,j)∈W} c_i (integer)
    # Sort BB descending, top d/2 vs bot d/2
    BB_sorted = np.sort(BB)[::-1]
    half = d // 2
    Delta_BB = BB_sorted[:half].sum() - BB_sorted[half:].sum()
    n_pairs = win_info['n_pairs']
    # corr in m² units:
    return Delta_BB / (2 * n * ell) + n_pairs / (4 * n * ell)


def J_bound(c, ell, s_lo, n, m, win_info):
    """LP with a≥0 narrowing for empty bins.  m² units.

    Linear part: solve max Σ B_j·δ_j s.t.
      - δ_j ∈ [-1, 1] if c_j ≥ 1 (in units of 1/m)
      - δ_j ∈ [-1, 0] if c_j = 0
      - Σ δ_j = 0
    Use scipy.linprog (slow but correct).

    Quadratic part: same as F (n_pairs).
    """
    d = len(c)
    A_W = win_info['A_W']
    B = A_W @ c.astype(np.float64) / m  # B_j in units of c/m
    # We solve in units of δ' = δ·m so δ' ∈ [-1, 1] or [-1, 0]
    # and m·B = BB. Objective: max Σ BB_j · δ'_j → linprog minimizes -obj.
    BB = A_W @ c.astype(np.float64)  # integer
    bounds = []
    for j in range(d):
        if c[j] == 0:
            bounds.append((-1.0, 0.0))
        else:
            bounds.append((-1.0, 1.0))
    A_eq = np.ones((1, d))
    b_eq = np.array([0.0])
    obj = -BB
    res = linprog(obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if not res.success:
        return F_bound(c, ell, s_lo, n, m, win_info)  # fallback
    Delta_BB_J = -res.fun  # max Σ BB·δ'

    # Linear in m² units: 2·h_phys·m·Delta_BB_J / (4n·ell·m²) · m² = Delta_BB_J·m·h_phys / (2n·ell·m²)
    # Wait, let me redo: TV-correction-linear ≤ 2·h·m·Delta_BB_J /(4n·ell·m²)
    # But h_phys = 1/m, so 2·(1/m)·Delta_BB_J/(4n·ell·m²) = Delta_BB_J/(2n·ell·m³)???
    # Hmm, units are confusing.  Let me restart from scratch.
    #
    # In F: TV-correction-linear ≤ (1/(4n·ell)) · 2·‖δ‖_∞ · max Σ B_j δ_j
    #       = (1/(4n·ell)) · 2·(1/m) · (1/m)·Delta_BB = Delta_BB/(2n·ell·m²)
    # m² units: corr_w_lin = Delta_BB/(2n·ell)  (matches F's formula)
    #
    # For J: same, with Delta_BB_J in place of Delta_BB.
    n_pairs = win_info['n_pairs']
    return Delta_BB_J / (2 * n * ell) + n_pairs / (4 * n * ell)


def N_bound(c, ell, s_lo, n, m, win_info):
    """F's linear + restricted-spectrum δ² bound.

    |Q(δ)| ≤ ‖A_W − α·11ᵀ‖_op · ‖δ‖₂² ≤ op_restricted · d · h².
    In m² units: op_restricted · d / (4n·ell).

    Take MIN with F's δ² bound (n_pairs / (4n·ell)) for soundness.
    """
    d = len(c)
    A_W = win_info['A_W']
    BB = A_W @ c.astype(np.float64)
    BB_sorted = np.sort(BB)[::-1]
    half = d // 2
    Delta_BB = BB_sorted[:half].sum() - BB_sorted[half:].sum()

    op_rest = win_info['op_norm_restricted']
    n_pairs = win_info['n_pairs']
    delta_sq_bound = min(n_pairs, op_rest * d)  # in numerator

    return Delta_BB / (2 * n * ell) + delta_sq_bound / (4 * n * ell)


def H_vertex_max(c, ell, s_lo, n, m, win_info):
    """NOT PROVEN SOUND.  Eval |2L(δ) - Q(δ)| at LP-vertices, take max.
    Vertices: sign patterns σ ∈ {-1,+1}^d with Σσ=0, i.e., d/2 +1's.

    Units: b_i = c_i/m, δ_j = σ_j/m, so 2·Σ b_iδ_j over W = 2 σ·BB / m²,
    and Σ δ_iδ_j over W = σ^T A_W σ / m².  In m² units (× m²):
      |2L − Q| · m² = |2 σ·BB − σ^T A_W σ| / (4n·ell).
    """
    d = len(c)
    A_W = win_info['A_W']
    BB = A_W @ c.astype(np.float64)
    half = d // 2
    best = 0.0
    for top in combinations(range(d), half):
        sigma = np.full(d, -1.0)
        for j in top:
            sigma[j] = +1.0
        L_int = float(np.dot(sigma, BB))         # = Σ σ_j BB_j
        Q_int = float(sigma @ A_W @ sigma)       # = σ^T A_W σ
        v = abs(2.0 * L_int - Q_int) / (4 * n * ell)
        if v > best:
            best = v
    return best


# ─────────── Test driver ───────────
def run(n_half, m, c_target, max_comps=2000, sample_stride=1, verbose=True):
    d = 2 * n_half
    S_full = 4 * n_half * m
    S_half = 2 * n_half * m
    n_total_half = count_compositions(n_half, S_half)
    if verbose:
        print(f"\n=== n_half={n_half}, m={m}, c_target={c_target} ===")
        print(f"     d={d}, S_full={S_full}, palindromic comps={n_total_half:,}")

    # Get F-survivors first (these are the hard cases for further pushes)
    surv = []
    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=200_000):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        s = prune_F(batch, n_half, m, c_target)
        if s.any():
            surv.append(batch[s].copy())
    surv = np.vstack(surv) if surv else np.empty((0, d), dtype=np.int32)
    if verbose:
        print(f"     F-survivors: {len(surv):,}")
    if len(surv) == 0:
        return {'n_F_survivors': 0}

    # Precompute window structure
    conv_len = 2 * d - 1
    max_ell = 2 * d
    win_info = precompute_windows(d, conv_len, max_ell)

    # Sample F-survivors for the bound comparison
    n_sample = min(len(surv), max_comps)
    rng = np.random.default_rng(0)
    if len(surv) > n_sample:
        idx = rng.choice(len(surv), n_sample, replace=False)
        sample = surv[idx]
    else:
        sample = surv
    if verbose:
        print(f"     Sample size: {len(sample):,} F-survivors")

    # For each F-survivor, find its TIGHTEST window (one closest to triggering
    # the F-bound).  This is where pushes matter most.
    # We compute: for each (composition, window), corr_F, corr_J, corr_N, H_vertex.
    # Track averages and max-tightness ratios.
    F_to_J_ratios = []
    F_to_N_ratios = []
    F_to_H_ratios = []
    n_J_better = n_N_better = n_H_better = 0
    n_J_strict = n_N_strict = n_H_strict = 0
    t0 = time.time()

    for idx_c in range(len(sample)):
        c = sample[idx_c]
        # Find window with max ws / threshold ratio (most binding)
        # For simplicity, evaluate all windows.
        for (ell, s_lo), info in win_info.items():
            if info['n_pairs'] == 0:
                continue
            f_b = F_bound(c, ell, s_lo, n_half, m, info)
            if f_b == 0:
                continue
            j_b = J_bound(c, ell, s_lo, n_half, m, info)
            n_b = N_bound(c, ell, s_lo, n_half, m, info)
            # H is heuristic, we evaluate but don't use for soundness
            if d <= 8:  # too slow at d=10+
                h_b = H_vertex_max(c, ell, s_lo, n_half, m, info)
            else:
                h_b = f_b

            F_to_J_ratios.append(j_b / f_b if f_b > 0 else 1.0)
            F_to_N_ratios.append(n_b / f_b if f_b > 0 else 1.0)
            F_to_H_ratios.append(h_b / f_b if f_b > 0 else 1.0)
            if j_b < f_b: n_J_better += 1
            if n_b < f_b: n_N_better += 1
            if h_b < f_b: n_H_better += 1
            if j_b < f_b - 1e-9: n_J_strict += 1
            if n_b < f_b - 1e-9: n_N_strict += 1
            if h_b < f_b - 1e-9: n_H_strict += 1

        if (idx_c + 1) % 50 == 0 and verbose:
            print(f"     [{idx_c+1}/{len(sample)}]  "
                  f"J<F: {n_J_strict}/{len(F_to_J_ratios):,}  "
                  f"N<F: {n_N_strict}/{len(F_to_N_ratios):,}  "
                  f"H<F: {n_H_strict}/{len(F_to_H_ratios):,}  "
                  f"[{time.time()-t0:.1f}s]", flush=True)

    n_eval = len(F_to_J_ratios)
    if n_eval == 0:
        return {'n_F_survivors': len(surv), 'n_evaluated': 0}

    out = {
        'n_F_survivors': len(surv),
        'n_sample': len(sample),
        'n_evaluated_window_comp_pairs': n_eval,
        'mean_J/F': float(np.mean(F_to_J_ratios)),
        'min_J/F': float(np.min(F_to_J_ratios)),
        'mean_N/F': float(np.mean(F_to_N_ratios)),
        'min_N/F': float(np.min(F_to_N_ratios)),
        'mean_H/F': float(np.mean(F_to_H_ratios)),
        'min_H/F': float(np.min(F_to_H_ratios)),
        'n_J_strict_better': n_J_strict,
        'n_N_strict_better': n_N_strict,
        'n_H_strict_better': n_H_strict,
        'wall_sec': time.time() - t0,
    }
    if verbose:
        print(f"\n     === Bound comparison vs F (smaller = tighter) ===")
        print(f"     Variant J (a≥0 LP):       mean ratio {out['mean_J/F']:.4f},  "
              f"strict tighter: {n_J_strict:,}/{n_eval:,}")
        print(f"     Variant N (op-norm δ²):   mean ratio {out['mean_N/F']:.4f},  "
              f"strict tighter: {n_N_strict:,}/{n_eval:,}")
        if d <= 8:
            print(f"     Variant H (LP vertex max, NOT SOUND):  "
                  f"mean ratio {out['mean_H/F']:.4f},  "
                  f"strict tighter: {n_H_strict:,}/{n_eval:,}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_comps', type=int, default=500)
    args = ap.parse_args()

    configs = [
        (3, 10, 1.20),
        (3, 10, 1.28),
        (4, 10, 1.20),
        (4, 10, 1.28),
        (5, 5, 1.28),
    ]
    results = []
    for nh, m, c in configs:
        res = run(nh, m, c, max_comps=args.max_comps)
        res['n_half'] = nh; res['m'] = m; res['c_target'] = c
        results.append(res)

    out = os.path.join(_dir, '_push_bench_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}\n")
    print(f"\n{'='*70}\nSummary:")
    for r in results:
        print(f"  (n={r['n_half']}, m={r['m']}, c={r['c_target']}): "
              f"J/F={r.get('mean_J/F', 1.0):.3f}  "
              f"N/F={r.get('mean_N/F', 1.0):.3f}  "
              f"H/F={r.get('mean_H/F', 1.0):.3f}")


if __name__ == '__main__':
    main()
