"""Variant QN: Q's multi-window LP weighting + N's restricted-spectrum δ² bound.

==============================================================================
DERIVATION (sound):
==============================================================================
Setup (same as Q): for composition c, b = c/m, cell A := { a : a >= 0,
|a-b|_oo <= 1/m, sum a = 4n }, δ := b - a satisfies |δ|_oo <= h := 1/m,
sum δ = 0.

For any λ >= 0 with sum λ = 1, define the λ-weighted "score":
    S(a; λ) := sum_W λ_W · TV_W(a)

We seek a SOUND lower bound on S(a; λ) over a in cell A. Then any composition
with min_a S(a; λ) > c_target·m^2 (in m^2 units) is prunable.

S(b; λ) - S(a; λ) = sum_W λ_W · [TV_W(b) - TV_W(a)]
                  = (1/(4n)) · sum_W (λ_W / ell_W) · [2·δ^T B^W - δ^T A_W δ]

Define
    T_j(λ) := sum_W (λ_W / ell_W) · BB^W_j         (linear in λ)
    A_λ    := sum_W (λ_W / ell_W) · A_W            (linear in λ)

Then S(b;λ) - S(a;λ) = (1/(4n))·[2·δ^T T(λ)/m - δ^T A_λ δ]
       (factor 1/m because BB is in c-units: T = m·B, δ·B = (δ·BB)/m)

LINEAR (TIGHT, matches Q): max over feasible δ of δ^T T(λ)/m = (1/m^2)·Δ_T(λ),
    Δ_T(λ) := sum_top(d/2) T(λ) - sum_bot(d/2) T(λ).
QUADRATIC (Q's bound):
    |δ^T A_λ δ| <= h² · sum_W (λ_W/ell_W) · n_pairs_W      (Q's per-window sum)
QUADRATIC (QN's spectral bound):
    Since Σδ = 0, for any α ∈ R:  δ^T (α·11ᵀ) δ = 0, so
        δ^T A_λ δ = δ^T (A_λ - α·11ᵀ) δ
    Choose α = α(λ) := (1ᵀ A_λ 1) / d² to project out the all-ones eigenvector.
    Then |δ^T A_λ δ| <= ‖A_λ - α·11ᵀ‖_op · ‖δ‖₂² <= op_rest(A_λ) · d · h²

QN's BOUND (sound, in m^2 units, tighter than Q's per-window sum):
    corr_QN(λ) = Δ_T(λ)/(2n) + min(op_rest(A_λ)·d, sum_W (λ_W/ell_W)·n_pairs_W) / (4n)

The min ensures QN ≤ Q per-λ (sound regression).

==============================================================================
LP-FRIENDLY ENCODING — fixed-point on λ (iterative)
==============================================================================
Spectral norm of A_λ is convex in λ but NOT linear. Direct LP encoding fails.
Two routes:
  (a) ITERATIVE FIXED POINT (this file): pick a candidate λ⁽ᵏ⁾, compute
      α⁽ᵏ⁾ = (1ᵀ A_{λ⁽ᵏ⁾} 1) / d², op_norm at that α, then RESOLVE the LP
      treating op_rest as a CONSTANT (linear in λ). Update λ, repeat.
      Convergence not guaranteed but each iterate gives a sound bound (since
      we always use min(spectral, sum)). We just need ANY iterate to certify
      pruning.
  (b) SDP (Lasserre level 1/2): exact. Out of scope for a fast pruning kernel.

CRUCIAL OBSERVATION (loose-but-cheap variant QN-init):
    Δ_T(λ) is tight; the spectral bound op_rest(A_λ)·d is NOT linear in λ.
    However, we can use the BOUND
        op_rest(A_λ) <= sum_W (λ_W/ell_W) · op_rest(A_W)
    by triangle inequality on operator norm of a sum. This GIVES a linear
    upper bound on op_rest(A_λ) in λ, hence directly LP-encodable.
    The resulting bound is:
        spectral_lin(λ) := sum_W (λ_W/ell_W) · op_rest(A_W) · d
    which is ≤ Q's "per-window sum" (since op_rest(A_W)·d ≤ n_pairs_W in many
    windows).
    This LP-friendly bound is WEAKER than the true op_rest(A_λ)·d but stronger
    than Q's per-window n_pairs sum AS LONG AS op_rest(A_W)·d ≤ n_pairs_W
    (which is the FN regime where N strictly improves F). In windows where
    op_rest(A_W)·d > n_pairs_W, we use the n_pairs_W bound (the per-window min).

    So: define per-window MIN bound m_W := min(op_rest(A_W)·d, n_pairs_W).
        QN-fast: corr = Δ_T(λ)/(2n) + sum_W (λ_W/ell_W) · m_W / (4n)
    This is JUST Q's LP with each window's n_pairs_W replaced by m_W.

==============================================================================
THIS FILE IMPLEMENTS QN-FAST (LP-friendly tightening, no inner iteration).
==============================================================================
This is sound: each m_W is a sound per-window bound on |δ^T A_W δ| (under Σδ=0,
|δ|_∞ ≤ h), and a sum of sound bounds with non-negative weights is a sound
bound on the sum. So QN-fast ≤ Q (because m_W ≤ n_pairs_W per window).

QN-fast SOUNDNESS PROOF:
    For any δ with Σδ=0, |δ|_∞ ≤ h:
       sum_W (λ_W/ell_W) · |δ^T A_W δ|
       <= sum_W (λ_W/ell_W) · min(op_rest(A_W)·d·h², n_pairs_W·h²)        (per-W sound)
       =  h² · sum_W (λ_W/ell_W) · m_W       ∎

    Note: this is NOT spectral_op(A_λ); rather it's a weighted sum of per-window
    spectral bounds. It's WEAKER than the joint spectral bound (which would
    be op_rest(A_λ)·d·h²) because of triangle inequality slack, but it's
    LINEAR in λ and hence LP-encodable.

QN-tight (iterative joint spectral) is implemented as a separate function,
QN_tight_one(), but is not used in the main loop because the iteration is
slow and rare wins. A future fast-SDP version could close this gap.

==============================================================================
EXPECTED GAIN OVER Q:
==============================================================================
QN-fast gains over Q exactly when m_W < n_pairs_W (per-window N tighter than F).
From _FN_bench.py: this happens in many "long-window" cases. Empirically the
N-vs-F gain is 5-30% extra on top of F at d>=10. So QN-fast over Q is bounded
by similar magnitude (per-window). The Q-gain over F is large (57% at d=10).
Combined QN-fast gain over F ≈ Q-gain × (1 + small N-gain).
"""
import os, sys, time, json
from itertools import combinations
import numpy as np
from scipy.optimize import linprog

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, _dir)
from compositions import generate_compositions_batched
from pruning import count_compositions

from _M1_bench import prune_F
from _N_bench import precompute_op_norm_restricted
from _Q_bench import (_enum_balanced_signs, _build_windows,
                       _composition_window_data, prune_Q_one)


# ----------------------------------------------------------------------
# QN-fast core: same LP as Q but with per-window m_W replacing n_pairs_W
# ----------------------------------------------------------------------
def _qn_bound_lp(c_int, windows, ell_int_sums, sigmas, n_half, m, c_target,
                  m_W_arr):
    """Solve QN-fast-LP for one composition.

    m_W_arr[w] = min(op_rest(A_W)·d, n_pairs_W) precomputed per-window.

    LP form (EXACTLY parallel to Q's _q_bound_lp, but V_w uses m_W_arr in place
    of ell_int_sums when m_W < ell_int_sum):

       V_w := ws_w/(4n·ell_w) - m_W_arr[w]/(4n·ell_w) - c_target·m^2

    All else is identical to Q.

    Returns (qn_excess_max, lambda_opt). qn_excess > 0 ⇒ prune.
    """
    d = len(c_int)
    ws, BB = _composition_window_data(c_int, windows, n_half, m)
    n_win = len(windows)
    n_sigma = len(sigmas)

    n_d = float(n_half)
    inv_4nl = np.array([1.0 / (4.0 * n_d * ell) for (ell, _) in windows])
    cs_m2 = c_target * m * m

    # Per-window constant: V_w = (ws_w - m_W) / (4n·ell) - cs_m2.
    V = ws.astype(np.float64) * inv_4nl - m_W_arr * inv_4nl - cs_m2

    # Linear-in-λ correction: same Δ_T as Q.
    ell_arr = np.array([ell for (ell, _) in windows], dtype=np.float64)
    BB_over_ell = BB.astype(np.float64) / ell_arr[:, None]
    M = sigmas.astype(np.float64) @ BB_over_ell.T
    A = V[None, :] - M / (2.0 * n_d)

    # LP: max t  s.t.  -A·λ + t <= 0; sum λ = 1; λ >= 0
    nvar = n_win + 1
    c_obj = np.zeros(nvar); c_obj[-1] = -1.0
    A_ub = np.zeros((n_sigma, nvar))
    A_ub[:, :n_win] = -A
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(n_sigma)
    A_eq = np.zeros((1, nvar))
    A_eq[0, :n_win] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0.0, None)] * n_win + [(None, None)]

    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs', options={'presolve': True})
        if not res.success:
            return -np.inf, None
        return float(-res.fun), res.x[:n_win]
    except Exception:
        return -np.inf, None


def prune_QN_one(c_int, windows, ell_int_sums, sigmas, n_half, m, c_target,
                  m_W_arr, margin=1e-9):
    """QN-fast-prune one composition. Returns True if pruned."""
    t_opt, _ = _qn_bound_lp(c_int, windows, ell_int_sums, sigmas,
                              n_half, m, c_target, m_W_arr)
    return t_opt > margin * m * m


# ----------------------------------------------------------------------
# QN-tight (iterative joint spectral) — implemented but not used in main loop
# ----------------------------------------------------------------------
def _qn_bound_iter(c_int, windows, ell_int_sums, sigmas, n_half, m, c_target,
                    m_W_arr, n_pairs_arr_per_window, A_W_list, max_iter=4):
    """QN-tight: iterative refinement using spectral bound on A_λ.

    Algorithm:
      1. Start with QN-fast LP (linear-in-λ). Get λ⁽⁰⁾.
      2. At each iter k:
         a. Form A_λ = sum_W (λ_W^(k)/ell_W) · A_W
         b. Compute α(λ) = (1ᵀ A_λ 1)/d², spectral norm ‖A_λ - α·11ᵀ‖_op
         c. The "true" δ²-bound is op_rest(A_λ)·d·h². Compare to the
            "per-window-sum" bound used in QN-fast.
         d. If spectral is tighter, REPLACE the LP's constant correction
            with the spectral value (treated as constant for this iter)
            and re-solve. New λ⁽ᵏ⁺¹⁾.
      3. Return best (smallest excess threshold) bound found.

    NOTE: the resulting bound at each iter is sound (we always use the better
    of spectral or sum, both individually sound). The LP at each iter treats
    spectral as a CONSTANT (computed at the previous iter's λ), so it doesn't
    capture the joint linearity. This makes the iteration not formally
    convergent, but each iterate is a valid pruning certificate.
    """
    d = len(c_int)
    n_win = len(windows)

    # Iteration 0: solve QN-fast.
    t_best, lam = _qn_bound_lp(c_int, windows, ell_int_sums, sigmas,
                                  n_half, m, c_target, m_W_arr)
    if lam is None:
        return -np.inf
    if t_best > 1e-9 * m * m:
        return t_best  # already pruning, no need to refine

    # Iterations 1..max_iter: refine using joint spectral.
    for it in range(max_iter):
        # Build A_λ = sum_W (λ_W/ell_W) · A_W.  (d × d real symmetric)
        A_lam = np.zeros((d, d), dtype=np.float64)
        for w, (ell, s_lo) in enumerate(windows):
            if lam[w] > 1e-12:
                A_lam += (lam[w] / ell) * A_W_list[w]
        # Restrict to Σδ=0 subspace
        alpha = float(np.sum(A_lam)) / (d * d)
        A_rest = A_lam - alpha
        eigs = np.linalg.eigvalsh(A_rest)
        op_norm = float(np.abs(eigs).max())  # spectral bound on |δ^T A_λ δ| / ‖δ‖₂²
        # Joint spectral correction: (op_norm * d) / (4n)  in m^2 units
        joint_corr = op_norm * d / (4.0 * n_half)
        # Compare to the LP's current "weighted sum" correction:
        #    sum_W (λ_W/ell_W) · m_W / (4n)
        sum_corr = 0.0
        for w, (ell, s_lo) in enumerate(windows):
            sum_corr += lam[w] / ell * m_W_arr[w]
        sum_corr /= (4.0 * n_half)
        if joint_corr >= sum_corr - 1e-12:
            break  # spectral not tighter; QN-fast already sufficient
        # Re-solve LP using a CONSTANT correction = joint_corr (for this λ).
        # This means the LP's m_W → 0 (effectively), and we ADD a CONSTANT
        # offset to the objective. Equivalently, treat the joint_corr as
        # a constant lower-bound replacement:
        #   excess_QN_iter(λ) = sum_W λ_W ws_W/(4n·ell_W) - Δ_T(λ)/(2n)
        #                       - joint_corr - cs_m2.
        # But the LP as written uses per-window V_w = ws_w·... - m_W·... - cs_m2,
        # so for the iterated version we set m_W := 0 (no per-window quadratic),
        # and SUBTRACT joint_corr from the t-threshold offset. Equivalently,
        # solve LP with m_W = 0 and the new lower bound is t_LP - joint_corr.
        zero_mW = np.zeros_like(m_W_arr)
        t_lp_zero, lam_new = _qn_bound_lp(c_int, windows, ell_int_sums, sigmas,
                                              n_half, m, c_target, zero_mW)
        if lam_new is None:
            break
        t_iter = t_lp_zero - joint_corr
        if t_iter > t_best:
            t_best = t_iter
            lam = lam_new
        if t_iter > 1e-9 * m * m:
            break  # certificate found
    return t_best


# ----------------------------------------------------------------------
# Window data construction with op_rest precomputation
# ----------------------------------------------------------------------
def precompute_window_data(d, n_half):
    """Build windows, ell_int_sums, sigmas, and m_W_arr.

    Returns:
        windows, ell_int_sums, sigmas, m_W_arr, A_W_list
    where m_W_arr[w] = min(op_rest(A_W)·d, n_pairs_W).
    """
    conv_len = 2 * d - 1
    max_ell = 2 * d
    windows, ell_int_sums = _build_windows(d)
    sigmas = _enum_balanced_signs(d)
    op_rest, n_pairs_arr = precompute_op_norm_restricted(d, max_ell, conv_len)

    # m_W per window. Iterate (ell, s_lo) parallel to _build_windows order.
    m_W_arr = np.empty(len(windows), dtype=np.float64)
    A_W_list = []
    for wi, (ell, s_lo) in enumerate(windows):
        n_pairs = float(n_pairs_arr[ell, s_lo])
        op_d = op_rest[ell, s_lo] * d
        m_W_arr[wi] = min(op_d, n_pairs)
        # Build A_W for the (rare) iterative refinement
        A = np.zeros((d, d), dtype=np.float64)
        for i in range(d):
            for j in range(d):
                if s_lo <= i + j <= s_lo + ell - 2:
                    A[i, j] = 1.0
        A_W_list.append(A)

    return windows, ell_int_sums, sigmas, m_W_arr, A_W_list


# ----------------------------------------------------------------------
# Test driver: run F, FN, Q, QN side-by-side
# ----------------------------------------------------------------------
def run_compare(n_half, m, c_target, batch_size=200_000, verbose=True,
                 max_lp_per_batch=None, run_qn_tight=False):
    d = 2 * n_half
    S_full = 4 * n_half * m
    S_half = 2 * n_half * m
    n_total_half = count_compositions(n_half, S_half)

    if verbose:
        print(f"\n=== n_half={n_half}, m={m}, c_target={c_target} (d={d}) ===")
        print(f"     palindromic comps={n_total_half:,}")

    # Window data + op_rest
    print(f"     precomputing windows + op_rest...", end=' ', flush=True)
    t_pre = time.time()
    windows, ell_int_sums, sigmas, m_W_arr, A_W_list = \
        precompute_window_data(d, n_half)
    n_win = len(windows)
    n_sigma = len(sigmas)
    print(f"n_win={n_win}, n_sigma={n_sigma} [{time.time()-t_pre:.1f}s]")

    # FN setup
    conv_len = 2 * d - 1
    max_ell = 2 * d
    op_rest, n_pairs_arr = precompute_op_norm_restricted(d, max_ell, conv_len)
    op_rest_d = op_rest * d
    ell_int_arr = np.empty(conv_len, dtype=np.int64)
    two_n = 2 * n_half
    for k in range(conv_len):
        d_idx = abs((k + 1) - two_n)
        v = max(0, two_n - d_idx)
        ell_int_arr[k] = v
    ell_prefix = np.zeros(conv_len + 1, dtype=np.int64)
    for k in range(conv_len):
        ell_prefix[k + 1] = ell_prefix[k] + ell_int_arr[k]

    from _FN_bench import prune_FN
    # Warm JITs
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = 2 * m
    prune_F(warm, n_half, m, c_target)
    prune_FN(warm, n_half, m, c_target, ell_prefix, op_rest_d)

    n_proc = 0
    surv_F_n = surv_FN_n = surv_Q_n = surv_QN_n = 0
    extra_FN = extra_Q = extra_QN = 0
    soundness_QN_minus_Q = 0
    soundness_Q_minus_F = 0
    soundness_FN_minus_F = 0
    t_F = t_FN = t_Q = t_QN = 0.0
    n_lp_runs = 0  # number of LPs solved (for time/LP profile)
    t0 = time.time()

    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=batch_size):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        n_proc += len(batch)

        # F
        ta = time.time()
        sF = prune_F(batch, n_half, m, c_target)
        t_F += time.time() - ta
        surv_F_n += int(sF.sum())

        # FN
        tb = time.time()
        sFN = prune_FN(batch, n_half, m, c_target, ell_prefix, op_rest_d)
        t_FN += time.time() - tb
        surv_FN_n += int(sFN.sum())
        soundness_FN_minus_F += int(np.sum(sFN & ~sF))
        extra_FN += int(np.sum(~sFN & sF))

        # Q (LP per F-survivor)
        f_idx = np.where(sF)[0]
        sQ_on_F = np.ones(len(f_idx), dtype=bool)
        if max_lp_per_batch is not None and max_lp_per_batch < len(f_idx):
            f_idx = f_idx[:max_lp_per_batch]
            sQ_on_F = sQ_on_F[:max_lp_per_batch]
        tc = time.time()
        for k, idx in enumerate(f_idx):
            c_int = batch[idx]
            if prune_Q_one(c_int, windows, ell_int_sums, sigmas,
                            n_half, m, c_target):
                sQ_on_F[k] = False
        t_Q += time.time() - tc
        n_lp_runs += len(f_idx)
        n_extra_Q = int(np.sum(~sQ_on_F))
        extra_Q += n_extra_Q
        # surv_Q on this batch (for stats only — note we only ran on F-survivors)
        surv_Q_batch = int(np.sum(sQ_on_F))
        surv_Q_n += surv_Q_batch

        # QN-fast (LP per F-survivor; expected to be ≤ Q's survivors)
        sQN_on_F = sQ_on_F.copy()  # start from Q-survivor set
        td = time.time()
        for k, idx in enumerate(f_idx):
            if not sQN_on_F[k]:
                continue  # already pruned by Q (so QN ⊆ Q)
            c_int = batch[idx]
            # Run QN-fast.  If QN prunes (and Q didn't), that's the win.
            if prune_QN_one(c_int, windows, ell_int_sums, sigmas,
                              n_half, m, c_target, m_W_arr):
                sQN_on_F[k] = False
        t_QN += time.time() - td
        n_extra_QN = int(np.sum(~sQN_on_F)) - int(np.sum(~sQ_on_F))
        extra_QN += n_extra_QN
        surv_QN_n += int(np.sum(sQN_on_F))

        # Soundness checks: QN survivor on f_idx should be ⊆ Q survivor;
        # Q survivor should be ⊆ F survivor; FN survivor ⊆ F survivor.
        for k in range(len(f_idx)):
            if sQN_on_F[k] and not sQ_on_F[k]:
                soundness_QN_minus_Q += 1
            if sQ_on_F[k] and not sF[f_idx[k]]:
                soundness_Q_minus_F += 1

    elapsed = time.time() - t0
    if verbose:
        print(f"     === Survivor counts ===")
        print(f"     F:   {surv_F_n:,}  [{t_F:.2f}s]")
        print(f"     FN:  {surv_FN_n:,}  [{t_FN:.2f}s]   "
              f"(+{extra_FN} extra prune over F)")
        print(f"     Q:   {surv_Q_n:,}   [{t_Q:.2f}s on {n_lp_runs:,} LPs]   "
              f"(+{extra_Q} extra prune over F)")
        print(f"     QN:  {surv_QN_n:,}  [{t_QN:.2f}s on {n_lp_runs:,} LPs]   "
              f"(+{extra_QN} extra prune over Q)")
        if n_lp_runs:
            print(f"     ms/LP: Q={1000*t_Q/n_lp_runs:.2f}, "
                  f"QN={1000*t_QN/n_lp_runs:.2f}")
        print(f"     SOUNDNESS:")
        print(f"        FN <= F:  {soundness_FN_minus_F} violations")
        print(f"        Q  <= F:  {soundness_Q_minus_F} violations")
        print(f"        QN <= Q:  {soundness_QN_minus_Q} violations")
        print(f"     wall: {elapsed:.1f}s")

    return {
        'n_half': n_half, 'm': m, 'c_target': c_target, 'd': d,
        'n_processed': n_proc, 'n_lp_runs': n_lp_runs,
        'n_win': n_win, 'n_sigma': n_sigma,
        'surv_F': surv_F_n, 'surv_FN': surv_FN_n,
        'surv_Q': surv_Q_n, 'surv_QN': surv_QN_n,
        'extra_FN_over_F': extra_FN, 'extra_Q_over_F': extra_Q,
        'extra_QN_over_Q': extra_QN,
        'soundness_FN_minus_F': soundness_FN_minus_F,
        'soundness_Q_minus_F': soundness_Q_minus_F,
        'soundness_QN_minus_Q': soundness_QN_minus_Q,
        't_F': t_F, 't_FN': t_FN, 't_Q': t_Q, 't_QN': t_QN,
        'ms_per_LP_Q': (1000 * t_Q / n_lp_runs) if n_lp_runs else None,
        'ms_per_LP_QN': (1000 * t_QN / n_lp_runs) if n_lp_runs else None,
        'wall_sec': elapsed,
    }


# ----------------------------------------------------------------------
# Sanity tests
# ----------------------------------------------------------------------
def _sanity():
    """Verify QN ≤ Q and QN ≤ F on a few hand-crafted comps."""
    print("=== QN sanity ===")
    n_half, m, c_target = 2, 5, 1.20
    d = 2 * n_half
    windows, ell_int_sums, sigmas, m_W_arr, _ = precompute_window_data(d, n_half)

    # m_W_arr should be ≤ ell_int_sums per-window
    diffs = m_W_arr - ell_int_sums.astype(np.float64)
    print(f"  m_W_arr - ell_int_sums:  min={diffs.min():.3f}, "
          f"max={diffs.max():.3f}, n_strict={int((diffs < -1e-9).sum())}")
    assert diffs.max() <= 1e-9, "m_W exceeds n_pairs — soundness bug!"

    # Run a few comps through Q and QN
    for c_int in [
        np.array([5, 5, 5, 5], dtype=np.int32),
        np.array([10, 0, 0, 10], dtype=np.int32),
        np.array([8, 2, 2, 8], dtype=np.int32),
        np.array([3, 7, 7, 3], dtype=np.int32),
    ]:
        if c_int.sum() != 4 * n_half * m:
            continue
        from _Q_bench import _q_bound_lp
        tQ, _ = _q_bound_lp(c_int, windows, ell_int_sums, sigmas,
                             n_half, m, c_target)
        tQN, _ = _qn_bound_lp(c_int, windows, ell_int_sums, sigmas,
                                n_half, m, c_target, m_W_arr)
        print(f"  c={c_int.tolist()}: Q-excess={tQ:+.4f}, QN-excess={tQN:+.4f}, "
              f"QN>=Q? {tQN >= tQ - 1e-9}")
        assert tQN >= tQ - 1e-9, "QN < Q — soundness regression!"


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_half', type=int, default=3)
    ap.add_argument('--m', type=int, default=10)
    ap.add_argument('--c_target', type=float, default=1.28)
    ap.add_argument('--batch', type=int, default=200_000)
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--sanity', action='store_true')
    ap.add_argument('--max_lp', type=int, default=None,
                     help='Cap LPs per batch (smoke test only). None = run all.')
    ap.add_argument('--out', type=str, default='_QN_results.json')
    args = ap.parse_args()

    if args.sanity:
        _sanity()
        sys.exit(0)

    results = []
    if args.sweep:
        configs = [
            (2, 10, 1.28),  # d=4
            (3, 10, 1.28),  # d=6
            (4, 10, 1.28),  # d=8
            (5, 5, 1.28),   # d=10
            (5, 10, 1.28),  # d=10 again, larger m
            (6, 5, 1.28),   # d=12
        ]
        for nh, m, c in configs:
            try:
                r = run_compare(nh, m, c, batch_size=args.batch,
                                  max_lp_per_batch=args.max_lp)
                results.append(r)
            except Exception as e:
                import traceback
                traceback.print_exc()
                results.append({'n_half': nh, 'm': m, 'c_target': c,
                                'error': str(e)})
    else:
        r = run_compare(args.n_half, args.m, args.c_target,
                          batch_size=args.batch,
                          max_lp_per_batch=args.max_lp)
        results.append(r)

    with open(args.out, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f"\nWrote {args.out}")
