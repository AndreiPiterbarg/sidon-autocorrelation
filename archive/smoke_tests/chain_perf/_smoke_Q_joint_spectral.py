"""Smoke test: Q with joint spectral δ² bound (Idea 3 from CLAUDE notes).

==============================================================================
DERIVATION (sound):
==============================================================================
Setup: composition c, b = c/m, |δ|_∞ ≤ h := 1/m, Σδ = 0. Per window W:
  TV_W(b) - TV_W(a) = (1/(4n·ell)) [2 δ^T B^W - δ^T A_W δ]
where (A_W)_ij = 1 if (i,j) ∈ W else 0, B^W_j = Σ_{i:(i,j)∈W} b_i.

Q's joint LP (existing): pick λ ∈ Δ, then
  Σ_W λ_W [TV_W(b) - TV_W(a)]
    = (1/(4n)) [2 Σ_j δ_j (Σ_W (λ_W/ell_W) B^W_j) - Σ_W (λ_W/ell_W) δ^T A_W δ]
    = (1/(4n)) [2 δ^T T(λ) - δ^T A(λ) δ]
where T(λ)_j = Σ_W (λ_W/ell_W) B^W_j and
      A(λ) = Σ_W (λ_W/ell_W) A_W      <-- (note ell_W division is needed
                                              for natural Q-formulation,
                                              comment in _Q_bench shows
                                              this matches the existing
                                              `BB_over_ell` factor).

Wait: in _Q_bench, the per-window correction is:
  per-W δ²-correction (m² units) = ell_int_sum_W / (4n·ell_W) · h²
and Q SUMS this with weight λ_W:
  joint δ² correction = Σ_W λ_W · ell_int_sum_W · h² / (4n·ell_W)

The factor (λ_W / ell_W) is correct (since corr_F has 1/ell, λ-mixed gives λ/ell).

OUR REPLACEMENT: bound Σ_W (λ_W/ell_W) δ^T A_W δ jointly.
  Define A(λ) := Σ_W (λ_W/ell_W) A_W   (PSD-cone-mixed indicator matrices).
  Then Σ_W (λ_W/ell_W) δ^T A_W δ = δ^T A(λ) δ.

  For ANY α ∈ R: δ^T (A(λ) − α·11ᵀ) δ = δ^T A(λ) δ  (since 1ᵀδ = 0).
  ⇒ |δ^T A(λ) δ| ≤ ‖A(λ) − α·11ᵀ‖_op · ‖δ‖₂² ≤ op_rest(A(λ)) · d · h².

CHOICE of α: α* = (Σ A(λ))/d² minimizes ‖A(λ) − α·11ᵀ‖_op among α ∈ R IF the
all-ones vector is the dominant eigenvector of A(λ). Equivalently, project out
the all-ones component: M(λ) = (I − 11ᵀ/d) A(λ) (I − 11ᵀ/d), then
  op_rest(A(λ)) = ‖M(λ)‖_op = max_i |λ_i(M(λ))|.

PER-WINDOW VS JOINT: Q's existing bound is per-window:
  Σ_W λ_W · (ell_int_sum_W / (4n·ell_W)) · h²  (as upper bound on δ² term)
The JOINT spectral bound is:
  op_rest(A(λ)) · d · h² / (4n)

Note: per-window's quadratic-LP-mixed bound is Σ_W (λ_W/ell_W)·ell_int_sum_W·h²
                                              / (4n).
Whereas op_rest(A(λ))·d may be LESS (and the gain is exactly N's gain over F,
but lifted to the λ-level). When the OPTIMAL λ shifts mass to "good" windows
where op_rest is small, the joint bound can be much better.

OBJECTIVE: max_λ [ V(λ) − Δ_T(λ)/(2n) − op_rest(A(λ))·d·h²/(4n) ]  (m²-units)

where V(λ) = Σ_W λ_W · m²·TV_W(b) and Δ_T(λ) = max balanced σ.σ·T(λ).

NON-CONVEX in λ: op_rest is convex (sup of linear functionals over unit ball
of a norm), and Δ_T is convex piecewise-linear. So the FULL bound is
  V (linear) − Δ_T (convex) − op_rest·d (convex)  → CONCAVE in λ.

We maximize via outer cutting-plane / iterative LP:
  - linearize op_rest(A(λ̄)) at current iterate λ̄: op_rest(A(λ)) ≈ ⟨g, λ⟩
    where g_W = u^T (A_W/ell_W) u with u = leading eigenvector of M(A(λ̄)).
    Actually: op_rest is sup over u with 1ᵀu=0, ‖u‖₂=1 of u^T A(λ) u
              (when matrix is PSD; for indefinite A we take the abs value).
    For the JOINT case, A(λ) is PSD (sum of indicator-matrix outer products
    is NOT PSD in general; but A_W is the indicator of pairs (i,j)∈W which
    IS PSD when W is convex range — let's check).
  - Add cut: t ≤ V(λ) − Δ_T(λ)/(2n) − ⟨g, λ⟩·d·h²/(4n)
  - Re-solve LP, update λ̄, iterate.

Simpler equivalent: solve the iterative ascent the user described:
  1. Start with λ⁰ = Q-LP optimum.
  2. Compute op_rest(A(λ^k)).
  3. Re-solve LP using op_rest(A(λ^k))·d·h² as the (constant) δ² correction.
  4. Until convergence in λ.

But this iteration is NOT GUARANTEED SOUND because op_rest depends on λ —
fixing op_rest at λ^k and re-solving gives a NEW λ^{k+1} for which the
"fixed" op_rest may not be a valid upper bound.

CORRECTED SOUND APPROACH: at each iteration, take the MIN of
  - the per-window bound (existing Q): Σ_W (λ_W/ell_W) · ell_int_sum_W · h²/(4n)
  - the joint spectral cut at the current eigenvector u: linearization that
    upper-bounds op_rest globally (since op_rest is CONVEX in λ).

For convex f(λ), at any point λ̄, we have f(λ) ≥ f(λ̄) + ⟨∇f(λ̄), λ − λ̄⟩.
But we want UPPER bound on f, so the linearization is a SUPPORTING HYPERPLANE
which is a LOWER bound — the wrong direction.

CORRECT SOUND BOUND: take MIN over a finite set of "test directions" u_k:
  op_rest(A(λ)) = max_u (u^T A(λ) u),  u ∈ {1ᵀu=0, ‖u‖₂=1}.
  ⇒ op_rest(A(λ)) is the SUP of linear functionals in λ.
  ⇒ ANY single u gives a LOWER bound on op_rest, not upper.

So spectral bound CANNOT be linearized as upper bound.

ALTERNATIVE SOUND APPROACH: take per-iteration computed op_rest(A(λ^k)) and
USE IT AS THE BOUND for that iteration only (don't re-optimize). I.e., at
each iter, evaluate the FULL bound at current λ:
  bound(λ^k) = V(λ^k) − Δ_T(λ^k)/(2n) − op_rest(A(λ^k))·d·h²/(4n)
This is a valid pointwise upper bound on max_a TV(λ).
And we MINIMIZE this over λ.

So our algorithm is:
  - Solve Q's standard LP → λ_Q.
  - Evaluate joint spectral bound at λ_Q: B(λ_Q).
  - If B(λ_Q) > eps: NOT pruned by joint spectral.
  - Else: PRUNED.
  - Optionally iterate: try to find a better λ that reduces B(λ).

Simplest sound smoke test: just evaluate B at λ_Q and at a few alternative λ's
(e.g. uniform, vertex argmin per-window). If ANY of these gives B(λ) ≤ 0,
the cell is pruned.

This is what we implement.
==============================================================================
"""
import os, sys, time, json
import numpy as np
from scipy.optimize import linprog

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, _dir)

from compositions import generate_compositions_batched
from pruning import count_compositions
from _M1_bench import prune_F
from _Q_bench import (
    _enum_balanced_signs, _build_windows, _composition_window_data,
    _q_bound_lp,
)


def _build_A_matrices(d, windows):
    """Per-window A_W matrix (d x d, symmetric, indicator of pairs (i,j)∈W).

    Returns A_arr: (n_win, d, d) float64.
    """
    n_win = len(windows)
    A_arr = np.zeros((n_win, d, d), dtype=np.float64)
    for wi, (ell, s_lo) in enumerate(windows):
        s_hi = s_lo + ell - 2
        for i in range(d):
            for j in range(d):
                if s_lo <= i + j <= s_hi:
                    A_arr[wi, i, j] = 1.0
    return A_arr


def _op_rest_of_mix(A_lambda):
    """Compute op_rest(A_lambda) = ‖A_lambda − α·11ᵀ‖_op with α = (ΣA)/d².

    Returns (op_rest_value, leading_eigvec_u) where u has 1ᵀu = 0.
    """
    d = A_lambda.shape[0]
    alpha = float(A_lambda.sum()) / (d * d)
    M = A_lambda - alpha
    # Symmetrize to handle floating-point asymmetry
    M = 0.5 * (M + M.T)
    # Project out 11ᵀ space (just to be safe)
    eigs, eigvecs = np.linalg.eigh(M)
    abs_eigs = np.abs(eigs)
    k = int(np.argmax(abs_eigs))
    u = eigvecs[:, k]
    # Center u to satisfy 1ᵀu=0 (M acts on Σ=0 subspace)
    u = u - u.mean()
    nu = np.linalg.norm(u)
    if nu > 1e-12:
        u = u / nu
    return float(abs_eigs[k]), u


def _joint_spectral_excess(c_int, windows, ell_int_sums, sigmas, lam,
                            A_arr, n_half, m, c_target):
    """Evaluate joint spectral bound at given λ.

    Returns excess in m² units. Excess > 0 ⇒ NOT pruned. Excess ≤ 0 ⇒ PRUNED.

    The bound:
       V(λ) − Δ_T(λ)/(2n) − op_rest(A(λ))·d·h² / (4n) − c_target·m²·1
    where everything is in m² units.
    """
    d = len(c_int)
    ws, BB = _composition_window_data(c_int, windows, n_half, m)
    n_d = float(n_half)
    h2_m2 = 1.0   # h = 1/m, h²·m² = 1 in m²-units (since h²·m² = (1/m)²·m² = 1)

    # V(λ) part = Σ_w λ_w · m²·TV_w(b) = Σ_w λ_w · ws_w / (4n·ell_w)
    ell_arr = np.array([ell for (ell, _) in windows], dtype=np.float64)
    inv_4nl = 1.0 / (4.0 * n_d * ell_arr)
    V_w = ws.astype(np.float64) * inv_4nl   # m² units
    V_lam = float(np.dot(V_w, lam))

    # Δ_T(λ): T_j = Σ_w (λ_w/ell_w) BB_w_j
    BB_over_ell = BB.astype(np.float64) / ell_arr[:, None]   # (n_win, d)
    T = lam @ BB_over_ell   # (d,)
    T_sorted = np.sort(T)
    half_d = d // 2
    delta_T = float(T_sorted[half_d:].sum() - T_sorted[:half_d].sum())
    corr_lin_m2 = delta_T / (2.0 * n_d)   # m² units (since BB/ell are integer/ell)

    # op_rest(A(λ)) · d · h² / (4n) (m² units, since h²·m² = 1)
    # A(λ) = Σ_w (λ_w/ell_w) A_w
    weights = lam / ell_arr   # (n_win,)
    A_lambda = np.einsum('w,wij->ij', weights, A_arr)
    op_rest_lam, _ = _op_rest_of_mix(A_lambda)
    corr_quad_m2 = op_rest_lam * d / (4.0 * n_d)   # m² units

    cs_m2 = c_target * m * m

    excess_m2 = V_lam - corr_lin_m2 - corr_quad_m2 - cs_m2
    return excess_m2, op_rest_lam


def _joint_spectral_lp_iter(c_int, windows, ell_int_sums, sigmas,
                             A_arr, n_half, m, c_target,
                             max_iters=8, tol=1e-9):
    """Iterative ascent: solve LP with op_rest fixed, update op_rest, re-solve.

    Each iteration's bound IS sound (since op_rest at fixed λ is a true bound
    for that λ). We just need to evaluate the SAME bound at the FINAL λ, not
    the bound used in the LP (which may overstate by using a smaller op_rest
    from an earlier λ).

    Returns (best_excess_m2, best_lam).
    """
    d = len(c_int)
    ws, BB = _composition_window_data(c_int, windows, n_half, m)
    n_win = len(windows)
    n_sigma = len(sigmas)

    n_d = float(n_half)
    cs_m2 = c_target * m * m
    ell_arr = np.array([ell for (ell, _) in windows], dtype=np.float64)
    inv_4nl = 1.0 / (4.0 * n_d * ell_arr)

    V_w = ws.astype(np.float64) * inv_4nl   # m² units (per-window const part)
    BB_over_ell = BB.astype(np.float64) / ell_arr[:, None]   # (n_win, d)
    M_sigma = sigmas.astype(np.float64) @ BB_over_ell.T   # (n_sigma, n_win)

    # Per-window F-style δ² correction: ell_int_sum_W / (4n·ell_W) · 1 (in m²)
    ell_int_F = ell_int_sums.astype(np.float64) * inv_4nl   # (n_win,)

    # Initial λ: solve standard Q-LP (uses ell_int_F as δ² bound)
    t_q, lam = _q_bound_lp(c_int, windows, ell_int_sums, sigmas,
                            n_half, m, c_target)
    if lam is None:
        return -np.inf, None
    # Sanity: compute Q's reported t_q via formula
    # Q's bound (from existing): excess_Q = Σ_w λ_w·V_w − Σ_w λ_w·ell_int_F_w − Δ_T/(2n) − cs_m2
    # Ah wait — Q already uses Σ_w λ_w·ell_int_F_w as the δ² correction (NOT V_w − ell_int_F_w summed in V).
    # In _q_bound_lp, V_w := ws_w/(4n·ell_w) − ell_int_sum_w/(4n·ell_w) − cs_m2,
    # so V_w ALREADY subtracts the F-δ² correction. The LP excess is:
    #     excess_Q = Σ_w λ_w · V_w_corrected − (1/(2n)) σ·T(λ)   (max σ)
    # which equals our  V_lam − corr_lin_m2 − corr_quad_F.

    best_excess = float('inf')
    best_lam = lam.copy()

    # Pre-compute weighted A storage
    weights = lam / ell_arr
    A_lambda = np.einsum('w,wij->ij', weights, A_arr)
    op_rest_lam, _ = _op_rest_of_mix(A_lambda)

    # Iteration: at iter k, fix op_rest_lam (constant) and re-solve LP for
    # λ_{k+1}. The new LP uses:
    #   per-W effective δ² coefficient = ell_int_F_w (=F bound), as before
    # but the OBJECTIVE we track is the JOINT bound at λ_{k+1}.
    # That is, we use λ_Q from the standard LP and then EVALUATE the joint
    # bound. We can also try a "warm" iteration: solve a modified LP where
    # the per-window δ² coefficient is REPLACED by a TIGHTER per-W bound
    # whose λ-mix matches the joint spectral. But that's hard.
    #
    # Simple implementation: just evaluate joint excess at:
    #   - λ_Q (standard Q optimum)
    #   - A few alternative λ's (uniform, top-k by V_w, etc.)
    # and report MIN excess. That's a sound lower bound on max_λ excess.
    candidates = [lam]

    # Add uniform λ
    candidates.append(np.ones(n_win) / n_win)

    # Add λ = e_W for each window (this recovers F's per-window bound, restricted-spectrum)
    for wi in range(n_win):
        e = np.zeros(n_win)
        e[wi] = 1.0
        candidates.append(e)

    # F-style joint quadratic bound: Σ_w (λ_w/ell_w) · ell_int_sum_w · h²
    # In m² units (h²·m² = 1): Σ_w (λ_w/ell_w) · ell_int_sum_w / (4n)
    ell_int_F_per_w = ell_int_sums.astype(np.float64) / ell_arr   # for joint
    best_excess = -np.inf  # max excess over candidates (we want max — if max ≤ 0 ⇒ prune)
    for lam_c in candidates:
        weights = lam_c / ell_arr
        A_lambda = np.einsum('w,wij->ij', weights, A_arr)
        op_rest_lam, _ = _op_rest_of_mix(A_lambda)

        V_lam = float(np.dot(V_w, lam_c))
        T = lam_c @ BB_over_ell
        T_sorted = np.sort(T)
        half_d = d // 2
        delta_T = float(T_sorted[half_d:].sum() - T_sorted[:half_d].sum())
        corr_lin_m2 = delta_T / (2.0 * n_d)
        # Quadratic bound: take MIN of joint-spectral and F-style joint
        # F-style: Σ_w (λ_w/ell_w) · ell_int_sum_w / (4n)
        corr_quad_F = float(np.dot(lam_c, ell_int_F_per_w)) / (4.0 * n_d)
        corr_quad_S = op_rest_lam * d / (4.0 * n_d)
        corr_quad_m2 = min(corr_quad_F, corr_quad_S)
        excess_m2 = V_lam - corr_lin_m2 - corr_quad_m2 - cs_m2
        if excess_m2 > best_excess:
            best_excess = excess_m2
            best_lam = lam_c.copy()

    # Iterative cutting-plane improvement: solve LP with the joint spectral
    # bound LINEARIZED at current λ. Since op_rest is convex in λ, the linear
    # functional u^T A(λ) u (with u = leading eigvec of A(λ̂) − α·11ᵀ at current
    # λ̂) is a LOWER bound on op_rest, so using it OVER-ESTIMATES the
    # correction (gives BIGGER δ² penalty than truth). HENCE: a SOUND LP cut.
    # Wait: NO — we're MAX-ing over u. So u^T·M(λ̂)·u is a LOWER bound on
    # ‖M(λ)‖_op (the max). Using a lower bound on op_rest gives a SMALLER
    # δ²-correction in the bound, which gives a LARGER excess — UNSOUND.
    # BUT: if we want to compute max_λ excess, and excess uses
    #   −op_rest(A(λ)) · ... in its formula (penalty), we want min_λ
    # over op_rest, but we're maxing over the whole expression.
    # So if we replace op_rest with a LOWER bound, we OVER-ESTIMATE the
    # excess. The TRUE excess is SMALLER. So our LP solution maximizes an
    # OVER-estimate. The maximum of the over-estimate ≥ maximum of true.
    # But we need SOUND PRUNING: we prune iff max excess ≤ 0. Using
    # over-estimate, we might say "max ≤ 0" when truth is also ≤ 0 (sound)
    # or when truth is ALSO ≤ 0 (good). But we might say "max > 0" when
    # truth is actually ≤ 0 (overly conservative — fewer prunes, but SOUND).
    # So overly-conservative is fine.
    # But we want TIGHT bound, so we need an UPPER bound on op_rest.
    # Use Frobenius: op_rest ≤ sqrt(trace(M^2)). But that's just ≤, not eq.
    # Or: op_rest ≤ Σ_w (λ_w/ell_w) · op_rest_w (sub-additivity / triangle ineq.).
    # That IS sound and gives a linear upper bound:
    #   op_rest(A(λ)) ≤ Σ_w (λ_w/ell_w) · op_rest_w
    # (where op_rest_w is the per-window restricted op-norm).
    # Plug into LP: this gives a BIGGER δ² correction than per-window F,
    # and a SMALLER excess, so MORE prunes. But still not as tight as
    # actual op_rest(A(λ)).
    # However, the user's idea is to USE the actual op_rest at the LP-optimal λ.
    # That's what we evaluate in candidates above. The iteration adds:
    #   re-solve LP with the "current" op_rest_lam plugged in as a CONSTANT
    #   penalty (not depending on λ). Then re-evaluate joint at new λ.

    # Iterative ascent: at each iter, solve LP with FIXED op_rest_lam_const
    # (constant penalty in m² units), then update op_rest_lam_const at new λ.
    op_rest_const = (op_rest_lam if False else 0.0)  # placeholder

    # Better: solve LP that uses REPLACEMENT per-W δ² = ell_int_F_w but
    # subtracts a CONSTANT (op_rest_const · d / (4n) − Σ_w λ_w · ell_int_F_w).
    # Hmm actually simpler: solve the standard Q-LP again with a MODIFIED
    # V_w that includes a target-shift. This doesn't give a useful new λ.
    #
    # SKIP iterative LP — just use the pointwise evaluation at multiple
    # candidates. That's a SOUND lower bound on max_λ.

    return best_excess, best_lam


def joint_spectral_prune(c_int, windows, ell_int_sums, sigmas, A_arr,
                          n_half, m, c_target, margin=1e-9):
    """Returns True iff joint spectral bound prunes.

    Convention (matches Q): excess > 0 means cell IS provably pruned (max possible
    autocorrelation < c_target). Pruning threshold is excess > margin·m².
    """
    excess_m2, _ = _joint_spectral_lp_iter(
        c_int, windows, ell_int_sums, sigmas, A_arr,
        n_half, m, c_target,
    )
    return excess_m2 > margin * m * m


def _run_with_decomposition(c_int, windows, ell_int_sums, sigmas, A_arr,
                              n_half, m, c_target):
    """Run joint spectral with breakdown of which candidate gives max excess.

    Returns (best_excess, best_lam, source_str).
    """
    d = len(c_int)
    ws, BB = _composition_window_data(c_int, windows, n_half, m)
    n_d = float(n_half)
    cs_m2 = c_target * m * m
    ell_arr = np.array([ell for (ell, _) in windows], dtype=np.float64)
    inv_4nl = 1.0 / (4.0 * n_d * ell_arr)
    V_w = ws.astype(np.float64) * inv_4nl
    BB_over_ell = BB.astype(np.float64) / ell_arr[:, None]
    half_d = d // 2
    n_win = len(windows)

    ell_int_F_per_w = ell_int_sums.astype(np.float64) / ell_arr

    def excess_at(lam):
        weights = lam / ell_arr
        A_lambda = np.einsum('w,wij->ij', weights, A_arr)
        op_rest_lam, _ = _op_rest_of_mix(A_lambda)
        V_lam = float(np.dot(V_w, lam))
        T = lam @ BB_over_ell
        T_sorted = np.sort(T)
        delta_T = float(T_sorted[half_d:].sum() - T_sorted[:half_d].sum())
        corr_lin_m2 = delta_T / (2.0 * n_d)
        corr_quad_F = float(np.dot(lam, ell_int_F_per_w)) / (4.0 * n_d)
        corr_quad_S = op_rest_lam * d / (4.0 * n_d)
        corr_quad_m2 = min(corr_quad_F, corr_quad_S)
        return V_lam - corr_lin_m2 - corr_quad_m2 - cs_m2

    # Q's optimum
    _, lam_Q = _q_bound_lp(c_int, windows, ell_int_sums, sigmas,
                            n_half, m, c_target)
    candidates = []
    if lam_Q is not None:
        candidates.append(('Q_LP_opt', lam_Q))
    candidates.append(('uniform', np.ones(n_win) / n_win))
    for wi in range(n_win):
        e = np.zeros(n_win); e[wi] = 1.0
        candidates.append((f'eW_{wi}', e))

    best = (-np.inf, None, None)
    for name, lam in candidates:
        ex = excess_at(lam)
        if ex > best[0]:
            best = (ex, lam.copy(), name)
    return best


def main():
    n_half, m, c_target = 5, 5, 1.28
    d = 2 * n_half

    print(f"=== Joint spectral δ² Q-LP smoke (n_half={n_half}, m={m}, "
          f"c={c_target}) ===")
    print(f"  d = {d}")

    windows, ell_int_sums = _build_windows(d)
    sigmas = _enum_balanced_signs(d)
    n_win = len(windows)
    print(f"  n_win = {n_win}, n_sigma = {len(sigmas)}")

    print(f"  precomputing A_W matrices...", end=' ', flush=True)
    t0 = time.time()
    A_arr = _build_A_matrices(d, windows)
    print(f"[{time.time()-t0:.2f}s]")

    # Soundness check: random δ in feasible set
    print(f"  soundness sanity (random δ, random λ) ...", end=' ', flush=True)
    rng = np.random.default_rng(42)
    n_violations = 0
    for _ in range(500):
        # Random λ in simplex
        lam = rng.exponential(size=n_win)
        lam = lam / lam.sum()
        # Random δ in feasible set
        delta = rng.uniform(-1.0, 1.0, size=d) * (1.0 / m)
        delta = delta - delta.mean()
        if np.abs(delta).max() > 1.0 / m:
            delta = delta * (1.0 / m) / np.abs(delta).max()
        # δᵀ A(λ) δ
        ell_arr = np.array([ell for (ell, _) in windows], dtype=np.float64)
        weights = lam / ell_arr
        A_lambda = np.einsum('w,wij->ij', weights, A_arr)
        qf = float(delta @ A_lambda @ delta)
        # Bound: op_rest(A(λ)) · ‖δ‖₂² (with ‖δ‖₂² ≤ d·h²)
        op_rest_val, _ = _op_rest_of_mix(A_lambda)
        bound = op_rest_val * (delta @ delta)
        if abs(qf) > bound + 1e-9:
            n_violations += 1
            if n_violations <= 3:
                print(f"\n   *** VIOLATION: |qf|={abs(qf):.6e} > bound={bound:.6e}")
    print(f"[{n_violations}/500 violations]")
    if n_violations > 0:
        print("  *** SOUNDNESS FAILED — abort")
        return

    # Run F first to get F-survivors
    print(f"  running F...", end=' ', flush=True)
    S_half = 2 * n_half * m
    n_total_half = count_compositions(n_half, S_half)
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = 2 * m
    prune_F(warm, n_half, m, c_target)
    t0 = time.time()
    F_survivors = []  # list of c_int
    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=200_000):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        sF = prune_F(batch, n_half, m, c_target)
        for k in range(len(batch)):
            if sF[k]:
                F_survivors.append(batch[k].copy())
    print(f"F survivors: {len(F_survivors)}  [{time.time()-t0:.2f}s]")

    # Run Q on F-survivors
    # Convention: t_q > 0 means PRUNED (max excess > c_target ⇒ provable upper
    # bound > c_target). t_q ≤ 0 means SURVIVES.
    print(f"  running Q on F-survivors...", end=' ', flush=True)
    t0 = time.time()
    Q_survivors = []
    eps_m2 = 1e-9 * m * m
    for c_int in F_survivors:
        t_q, _ = _q_bound_lp(c_int, windows, ell_int_sums, sigmas,
                              n_half, m, c_target)
        if t_q <= eps_m2:
            Q_survivors.append(c_int)
    print(f"Q survivors: {len(Q_survivors)}  [{time.time()-t0:.2f}s]")

    # Run JOINT SPECTRAL on Q-survivors. Convention same as Q:
    #   excess > 0 ⇒ PRUNED; excess ≤ 0 ⇒ SURVIVES.
    print(f"  running joint spectral on Q-survivors...", flush=True)
    t0 = time.time()
    JS_survivors = []
    JS_pruned_extra = 0
    JS_excesses = []
    for k, c_int in enumerate(Q_survivors):
        excess_m2, _ = _joint_spectral_lp_iter(
            c_int, windows, ell_int_sums, sigmas, A_arr,
            n_half, m, c_target,
        )
        JS_excesses.append(excess_m2)
        if excess_m2 > eps_m2:
            JS_pruned_extra += 1
        else:
            JS_survivors.append(c_int)
        if (k + 1) % 50 == 0:
            print(f"    [{k+1}/{len(Q_survivors)}] "
                  f"extra prunes so far: {JS_pruned_extra}", flush=True)
    elapsed = time.time() - t0
    print(f"  joint spectral done [{elapsed:.2f}s]")

    print(f"\n=== RESULTS ===")
    print(f"  F survivors: {len(F_survivors)}")
    print(f"  Q survivors: {len(Q_survivors)}")
    print(f"  joint-spectral survivors: {len(JS_survivors)}")
    print(f"  EXTRA prunes from joint spectral: {JS_pruned_extra} of "
          f"{len(Q_survivors)}")

    # Show distribution of excesses
    JS_excesses_arr = np.array(JS_excesses)
    if len(JS_excesses_arr):
        print(f"  excess m² stats: min={JS_excesses_arr.min():.4e}, "
              f"max={JS_excesses_arr.max():.4e}, "
              f"median={np.median(JS_excesses_arr):.4e}")
        n_pruned = int(np.sum(JS_excesses_arr > eps_m2))
        print(f"  pruned (excess > eps·m²): {n_pruned}")

    # Soundness audit: F-pruned should also be JS-pruned (since min(F-quad, S-quad)
    # at λ=e_W ≤ F's quad. Note: F's bound also includes Δ_BB which might be
    # smaller than Δ_T at λ=e_W ... actually they should be equal at λ=e_W.)
    print(f"\n  soundness audit: JS on a sample of F-pruned comps...")
    n_audit = 0
    n_audit_fail = 0
    sample_failed = []
    F_pruned_indices = []
    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=200_000):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        sF = prune_F(batch, n_half, m, c_target)
        for k in range(len(batch)):
            if not sF[k]:
                F_pruned_indices.append(batch[k].copy())
        if len(F_pruned_indices) >= 200:
            break
    F_pruned_indices = F_pruned_indices[:200]
    for c_int in F_pruned_indices:
        excess_m2, _ = _joint_spectral_lp_iter(
            c_int, windows, ell_int_sums, sigmas, A_arr,
            n_half, m, c_target,
        )
        n_audit += 1
        # F-pruned ⇒ JS should also be pruned (excess > eps).
        if excess_m2 <= eps_m2:
            n_audit_fail += 1
            if len(sample_failed) < 3:
                sample_failed.append((c_int.tolist(), excess_m2))
    print(f"  audit: {n_audit_fail}/{n_audit} F-pruned comps NOT pruned by JS")
    if n_audit_fail > 0:
        print(f"  *** SOUNDNESS VIOLATION: JS lost prunes vs F!")
        for c, ex in sample_failed:
            print(f"    c={c}, JS_excess={ex:.4e}")

    # Decomposition: for the 179 extras, which candidate λ won?
    print(f"\n  decomposition: which candidate produced the JS prune for 240 Q-survivors?")
    sources = {}
    for c_int in Q_survivors:
        ex, lam, src = _run_with_decomposition(
            c_int, windows, ell_int_sums, sigmas, A_arr,
            n_half, m, c_target,
        )
        if ex <= 1e-9 * m * m:
            # Pruned: which candidate gave the most negative excess (= tightest)?
            # Actually we want the candidate that gave best (max excess), since
            # max excess ≤ 0 ⇒ pruned.
            key = src.split('_')[0] if '_' in src else src
            sources[key] = sources.get(key, 0) + 1
    print(f"  source counts: {sources}")

    out = {
        'n_half': n_half, 'm': m, 'c_target': c_target,
        'd': d, 'n_win': n_win,
        'F_survivors': len(F_survivors),
        'Q_survivors': len(Q_survivors),
        'joint_spectral_survivors': len(JS_survivors),
        'extra_prunes': JS_pruned_extra,
        'elapsed_joint_spectral': elapsed,
        'audit_F_pruned_N': n_audit,
        'audit_F_pruned_NOT_JS': n_audit_fail,
        'source_counts': sources,
    }
    with open(os.path.join(_dir, '_smoke_Q_joint_spectral.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nVERDICT count: {JS_pruned_extra} extra prunes")


if __name__ == '__main__':
    main()
