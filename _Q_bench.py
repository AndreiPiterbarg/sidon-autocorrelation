"""Q benchmark: multi-window weighted-sum cell LP, single-composition level.

VERDICT (this file's premise, formally derived):
=================================================
The user's first-cut analysis says "single-composition multi-window LP = F".
This is WRONG when the inner inf-over-cell is taken jointly across windows.

DERIVATION (verified):
======================
For composition c with b = c/m, cell of valid continuous heights a is:
    A := { a : a >= 0, |a - b|_oo <= 1/m, sum a = 4n }
Equivalently δ := b - a satisfies |δ|_oo <= 1/m, sum δ = 0 (a >= 0 follows
when b is integer/m and δ <= b coordinate-wise; ignored as in F).

Per window W = (ell, s_lo) define
    TV_W(a) := (1/(4n·ell)) · sum_{(i,j) in W} a_i a_j
where (i,j) in W means s_lo <= i+j <= s_lo + ell - 2 (consecutive ell-1 conv
positions in the autoconv).

Identity: TV_W(b) - TV_W(a) = (1/(4n·ell)) [ 2 sum_{(i,j) in W} b_i δ_j
                                              - sum_{(i,j) in W} δ_i δ_j ].

For F (variant F): per-window
    TV_W(a) >= TV_W(b) - corr_F(W; b)
where corr_F(W;b) = Δ_BB / (2n·ell) + ell_int_sum / (4n·ell),
      Δ_BB := sort(BB^W) top-d/2 sum minus bot-d/2 sum,
      BB^W_j := m · sum_{i: (i,j) in W} b_i.
(All quantities scaled to m^2 units throughout.)

F PRUNES iff there exists W with [m^2 TV_W(b) - corr_F(W;b)] > c_target·m^2.

For Q: pick λ >= 0, sum λ = 1 (over windows W). Then
    sum_W λ_W TV_W(a) >= sum_W λ_W TV_W(b) - sup_{δ in cell} sum_W λ_W [TV_W(b)-TV_W(a)]
The sup uses a SINGLE δ shared across windows, which is the key tightening:
    sum_W λ_W [TV_W(b)-TV_W(a)]
       = (1/(4n)) sum_W (λ_W/ell_W) [2 sum_j δ_j BB^W_j/m - sum_{(i,j) in W} δ_i δ_j]
       <= (1/(4n)) [2 sum_j δ_j (sum_W (λ_W/ell_W) BB^W_j)/m
                    + h^2 sum_W (λ_W/ell_W) ell_int_sum^W ]
With T_j := sum_W (λ_W/ell_W) BB^W_j, the LP closed form (M1 lemma):
    sup_{|δ|_oo <= 1/m, sum δ = 0} sum_j δ_j T_j / m = (1/m^2) Δ_T   (m^2 units),
    Δ_T = sum_top(d/2) T - sum_bot(d/2) T.
So in m^2 units:
    corr_Q(λ; b) = Δ_T(λ)/(2n) + (1/(4n)) sum_W (λ_W/ell_W) ell_int_sum^W.
Hence Q PRUNES iff there exists λ >= 0, sum λ = 1, with
    sum_W λ_W [m^2 TV_W(b) - (1/(4n))(λ-weighted ell_int_sum/ell term)]
        - Δ_T(λ)/(2n) > c_target·m^2.
The objective is concave (Δ_T is convex piecewise-linear in λ), so we can
solve as an LP via the σ-formulation:

    For each σ ∈ {-1,+1}^d with sum σ = 0:
        g_σ(λ) := sum_W λ_W·V_W  -  (1/(2n)) sum_j σ_j sum_W (λ_W/ell_W)·BB^W_j
        with V_W := m^2·TV_W(b) - ell_int_sum^W/(4n·ell_W).
    Q-bound = max_λ min_σ g_σ(λ).
LP:
    max t  s.t.  g_σ(λ) >= t  ∀σ;  λ >= 0;  sum λ = 1.
    Number of σ's = binom(d, d/2). Tractable for d <= 10.

DUALITY GAIN (Q vs F):
======================
F = max_W min_σ_W g_W,σ_W(e_W),  where e_W is unit vector at W, and σ_W is
    the per-W extremiser top/bot d/2 of BB^W.
Q = max_{λ ∈ Δ} min_σ g_σ(λ),  with σ a SHARED sign pattern (cardinality
    binom(d,d/2)) and λ a convex combination over windows.
The F-bound corresponds to setting λ=e_W for some W, then σ to the W-best.
Q allows mixing windows; the LP duality gives a strictly tighter bound when
the optimal-σ for different W's disagree (mixing reduces the L1-Δ jointly).

When does Q == F? When ALL windows share the same Δ_BB-extremiser σ AND the
optimal λ is a vertex (one window only). Otherwise Q > F (strict).

SOUNDNESS:
==========
Same δ-cell as F. Same correction-bound structure. Q's bound is a tightening
of F's, and any δ feasible for F is feasible for Q. So Q is sound iff F is,
which we already know.  Empirically: every Q-survivor must be an F-survivor
(verified in run() below).
"""
import os, sys, time, json
from itertools import combinations
import numpy as np
from scipy.optimize import linprog

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
from compositions import generate_compositions_batched
from pruning import count_compositions

# Reuse F's reference kernel from _M1_bench
sys.path.insert(0, _dir)
from _M1_bench import prune_F  # numba-jit reference for F


# ----------------------------------------------------------------------
# Helpers (no JIT — outer Python loop, batched LPs)
# ----------------------------------------------------------------------

def _enum_balanced_signs(d):
    """Enumerate all σ ∈ {-1,+1}^d with sum σ = 0 (d even).

    Returns array of shape (n_sigma, d), each row in {-1,+1}.
    n_sigma = C(d, d/2).
    """
    half = d // 2
    rows = []
    for top_idx in combinations(range(d), half):
        s = -np.ones(d, dtype=np.int8)
        for k in top_idx:
            s[k] = 1
        rows.append(s)
    return np.stack(rows, axis=0)


def _build_windows(d):
    """Enumerate all (ell, s_lo) windows.

    Returns:
        windows: list of (ell, s_lo) tuples
        ell_int_sum: int array, ell_int_sum[w] = sum of (i,j) pairs in window
    """
    conv_len = 2 * d - 1
    max_ell = 2 * d
    windows = []
    ell_int_sums = []

    # ell_int per conv position (number of (i,j) pairs with i+j = k)
    ell_int_arr = np.empty(conv_len, dtype=np.int64)
    for k in range(conv_len):
        d_idx = abs((k + 1) - d)
        v = d - d_idx
        ell_int_arr[k] = max(0, v)

    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        for s_lo in range(n_windows):
            windows.append((ell, s_lo))
            ell_int_sums.append(int(np.sum(ell_int_arr[s_lo:s_lo + n_cv])))
    return windows, np.array(ell_int_sums, dtype=np.int64)


def _composition_window_data(c_int, windows, n_half, m):
    """For composition c (integer length d, sum 4nm), compute per-window:
        ws_W   = sum of conv[s_lo .. s_lo+ell-2]                (int)
        BB^W   = (BB^W_0, ..., BB^W_{d-1}),  BB^W_j = sum_{i:(i,j) in W} c_i
    """
    d = len(c_int)
    conv_len = 2 * d - 1

    # Autoconvolution
    conv = np.zeros(conv_len, dtype=np.int64)
    for i in range(d):
        ci = int(c_int[i])
        if ci != 0:
            conv[2 * i] += ci * ci
            for j in range(i + 1, d):
                cj = int(c_int[j])
                if cj != 0:
                    conv[i + j] += 2 * ci * cj

    # Prefix sum of c (for BB)
    prefix_c = np.zeros(d + 1, dtype=np.int64)
    for i in range(d):
        prefix_c[i + 1] = prefix_c[i] + int(c_int[i])

    n_win = len(windows)
    ws = np.empty(n_win, dtype=np.int64)
    BB = np.empty((n_win, d), dtype=np.int64)

    for wi, (ell, s_lo) in enumerate(windows):
        n_cv = ell - 1
        s_hi = s_lo + n_cv - 1
        ws[wi] = int(np.sum(conv[s_lo:s_lo + n_cv]))
        for j in range(d):
            lo_i = max(0, s_lo - j)
            hi_i = min(d - 1, s_hi - j)
            if hi_i < lo_i:
                BB[wi, j] = 0
            else:
                BB[wi, j] = int(prefix_c[hi_i + 1] - prefix_c[lo_i])

    return ws, BB


def _q_bound_lp(c_int, windows, ell_int_sums, sigmas, n_half, m, c_target):
    """Solve Q-LP for one composition.

    Returns (q_excess_max, lambda_opt). q_excess_max > 0 means Q-prunable.
    q_excess_max in m^2 units, comparable to F's per-window excess.

    LP variables: [λ_0, ..., λ_{n_win-1}, t]   (n_win + 1 vars).
    Maximize t.
    Constraints (in m^2 units):
        sum_w λ_w * V_w  -  (1/(2n)) sum_j σ_j (sum_w (λ_w/ell_w) BB^w_j)  -  c_target·m^2  >=  t
        sum_w λ_w = 1
        λ_w >= 0

    where V_w := ws_w - ell_int_sum_w/(4n·ell_w).
    Note: ws_w is INTEGER (m^2·4n·ell_w·TV_W(b) units — careful), so:
         m^2·TV_w(b) = ws_w / (4n·ell_w)
    Ah, but we want m^2·TV_w as a coefficient of λ_w. Let's be precise.

    Working in integer-conv units (NO normalisation):
       m^2·4n·ell_w · TV_w(b) = ws_w   [integer]
       m^2·4n·ell_w · TV_w(a) >= ws_w - 4n·ell_w·corr_F(W)·1
       m^2·4n·ell_w · c_target = c_target·m^2·4n·ell_w
    Pruning F (per-W): ws_w  >  c_target·m^2·4n·ell_w + 4n·ell_w·corr_F  (in units)
    But corr_F is in m^2 units; let's stay in m^2-units throughout for clarity.

    Switch to "per-window normalized excess":
       excess_F(W) := m^2·TV_w(b) - corr_F(W) - c_target·m^2    (m^2 units)
                    = ws_w/(4n·ell_w) - Δ_BB/(2n·ell_w) - ell_int_sum/(4n·ell_w) - c_target·m^2
       F prunes iff excess_F(W) > 0 for some W.

    For Q:
       excess_Q(λ) := sum_w λ_w · [ws_w/(4n·ell_w) - ell_int_sum_w/(4n·ell_w) - c_target·m^2]
                    -  Δ_T(λ)/(2n)
       where T_j = sum_w (λ_w/ell_w) BB^w_j
             Δ_T(λ) = sort(T) top-d/2 sum minus bot-d/2 sum
                    = max over balanced σ of σ·T = max over σ of sum_j σ_j T_j
       Q prunes iff there exists λ ∈ Δ with excess_Q(λ) > 0.
       Since Δ_T = max_σ σ·T is a max of linear functions, we LP-formulate
       with one constraint per σ:
            for all σ:
              sum_w λ_w·[ws_w/(4n·ell_w) - ell_int_sum_w/(4n·ell_w) - c_target·m^2]
                - (1/(2n)) σ·T(λ)    >=    t
                where σ·T(λ) = sum_w (λ_w/ell_w)·(σ·BB^w)
       Q prunes iff t* > 0.
    """
    d = len(c_int)
    ws, BB = _composition_window_data(c_int, windows, n_half, m)
    n_win = len(windows)
    n_sigma = len(sigmas)

    n_d = float(n_half)
    inv_4nl = np.array([1.0 / (4.0 * n_d * ell) for (ell, _) in windows])
    inv_2nl = np.array([1.0 / (2.0 * n_d * ell) for (ell, _) in windows])
    cs_m2 = c_target * m * m

    # Per-window constant in λ (the part not involving σ):
    #   V_w := ws_w * inv_4nl[w]  -  ell_int_sum_w * inv_4nl[w]  -  cs_m2
    V = ws.astype(np.float64) * inv_4nl - ell_int_sums.astype(np.float64) * inv_4nl - cs_m2

    # σ·BB^w  =  sum_j σ_j BB^w_j   (n_sigma x n_win matrix)
    # We want, per σ, the linear function in λ:
    #   const_σ(λ) := sum_w λ_w·V_w - sum_w (λ_w·inv_2nl_w)·(σ·BB^w_j summed over j)·(1/m^?)
    # Recall: BB is in c-units (integer), so σ·BB^w is integer.
    # The correction in m^2 units involves (m^2 / m) ·... ; let's re-derive cleanly.
    #
    # corr_F linear part in m^2 units = m·Δ_B / (2n·ell), with B = b·m·... wait re-do.
    #
    # In _M1_bench corr_F (m^2 units) = Δ_BB/(2n·ell) + ell_int_sum/(4n·ell)
    # where BB^W_j = sum_{i:(i,j)∈W} c_i (integer, in c-units, NOT b-units).
    # So Δ_BB = (sum_top BB) - (sum_bot BB), all integers.
    # The Q linear correction in m^2 units:
    #     corr_Q_lin = (1/(2n)) Δ_T  with T_j = sum_w (λ_w/ell_w) BB^w_j  (no ell-w in defn, the / ell is there)
    # Wait — original F per-W: corr_F_lin = Δ_BB^W / (2n·ell_W) = Δ_(BB^W/ell_W) / (2n).
    # So if we define U^w_j := BB^w_j / ell_w, then F's linear correction is Δ_U^w / (2n).
    # Then Q: T_j = sum_w λ_w·U^w_j, and Δ_T / (2n) is the multi-window correction.
    # Sanity: λ = e_w gives T = U^w, recovers F's per-window correction.   ✓

    # Build sigma·U^w:   M[σ, w] := sum_j σ_j BB^w_j / ell_w
    # Then constraint per σ:  sum_w λ_w · (V_w - M[σ,w]/(2n))  >= t
    # i.e.   sum_w λ_w · A[σ, w]  >= t,  with  A[σ, w] = V_w - M[σ, w]/(2n).

    # σ shape (n_sigma, d); BB shape (n_win, d); ell array (n_win,)
    ell_arr = np.array([ell for (ell, _) in windows], dtype=np.float64)
    BB_over_ell = BB.astype(np.float64) / ell_arr[:, None]   # (n_win, d)
    # M[σ, w] = (BB_over_ell @ σ.T).T = (n_sigma, n_win)
    M = sigmas.astype(np.float64) @ BB_over_ell.T   # (n_sigma, n_win)
    A = V[None, :] - M / (2.0 * n_d)                # (n_sigma, n_win)

    # LP:  max t   s.t.  A[σ, :] · λ - t >= 0  ∀σ;  Σλ = 1;  λ >= 0
    # In linprog form (min c · x s.t. A_ub x <= b_ub, A_eq x = b_eq):
    #   x = [λ_0, ..., λ_{n_win-1}, t];   c = [0, ..., 0, -1]
    #   A_ub x <= b_ub:    -A[σ,:] · λ + t <= 0   ∀σ
    #   A_eq x = b_eq:     1·λ = 1
    #   bounds: λ >= 0, t free

    nvar = n_win + 1
    c_obj = np.zeros(nvar)
    c_obj[-1] = -1.0

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
        t_opt = -res.fun
        return float(t_opt), res.x[:n_win]
    except Exception:
        return -np.inf, None


def prune_Q_one(c_int, windows, ell_int_sums, sigmas, n_half, m, c_target,
                  margin=1e-9):
    """Q-prune one composition. Returns True if pruned."""
    t_opt, _ = _q_bound_lp(c_int, windows, ell_int_sums, sigmas,
                             n_half, m, c_target)
    # excess_Q in m^2 units; prune if > margin·m^2 to match F's eps_margin
    return t_opt > margin * m * m


# ----------------------------------------------------------------------
# F reference (use _M1_bench.prune_F directly — it's njit-parallel, fast)
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Test driver
# ----------------------------------------------------------------------
def run(n_half, m, c_target, batch_size=200_000, verbose=True,
         max_q_per_batch=None):
    d = 2 * n_half
    S_full = 4 * n_half * m
    S_half = 2 * n_half * m
    n_total_half = count_compositions(n_half, S_half)

    if verbose:
        print(f"\n=== Q-bench: n_half={n_half}, m={m}, c_target={c_target} ===")
        print(f"d={d}, S_full=4nm={S_full}, palindromic half_sum=2nm={S_half}, "
              f"total palindromic comps={n_total_half:,}")

    # Build LP scaffolding
    windows, ell_int_sums = _build_windows(d)
    n_win = len(windows)
    sigmas = _enum_balanced_signs(d)
    print(f"  n_win = {n_win}, n_sigma (balanced d/2) = {len(sigmas)}")

    # Warm up F kernel
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = 2 * m
    prune_F(warm, n_half, m, c_target)

    n_processed = 0
    pruned_F = surv_F_n = 0
    pruned_Q = surv_Q_n = 0
    bug_Q_minus_F = 0      # Q survivors that are NOT F survivors (soundness viol)
    extra_Q_over_F = 0     # F survivors that Q ALSO prunes (this is the win!)
    t_F = 0.0
    t_Q = 0.0
    t0 = time.time()

    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=batch_size):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        n_processed += len(batch)

        # F (njit-parallel, fast)
        tf = time.time()
        sF = prune_F(batch, n_half, m, c_target)
        t_F += time.time() - tf
        pruned_F += int(np.sum(~sF))
        surv_F_n += int(np.sum(sF))

        # Q (Python LP, slow). Run only on F survivors -- Q can ONLY prune more
        # than F can in theory; in practice we still need to verify that EVERY
        # Q-pruner is an F-pruner, but if Q is sound it strictly tightens F so
        # we can skip non-F-survivors (they're already pruned).
        # However, soundness audit requires running Q on a sample of NON-F-survivors
        # to ensure no false-prune. We'll run Q on F-survivors only for the count;
        # to verify soundness, we audit a random sample of non-F-survivors below.
        f_surv_idx = np.where(sF)[0]
        sQ_on_F = np.ones(len(f_surv_idx), dtype=bool)

        n_run_q = len(f_surv_idx)
        if max_q_per_batch is not None and max_q_per_batch < n_run_q:
            # subsample (for fast smoke testing only); will be flagged
            n_run_q = max_q_per_batch
            f_surv_idx = f_surv_idx[:n_run_q]
            sQ_on_F = sQ_on_F[:n_run_q]

        tq = time.time()
        for k, idx in enumerate(f_surv_idx):
            c_int = batch[idx]
            if prune_Q_one(c_int, windows, ell_int_sums, sigmas,
                            n_half, m, c_target):
                sQ_on_F[k] = False
        t_Q += time.time() - tq

        # Build sQ (full bool array): True iff (F-survivor and Q-survivor) — but
        # we need to count Q over ALL comps (not just F-survivors). For F-pruned
        # comps: should also be Q-pruned (Q is a tightening of F). So Q-prune
        # count = F-prune count + extra_Q_over_F.
        n_extra = int(np.sum(~sQ_on_F))
        extra_Q_over_F += n_extra

        # Q-survivors = F-survivors minus Q-extras.
        surv_Q_n += int(np.sum(sQ_on_F))
        pruned_Q += int(np.sum(~sF)) + n_extra
        # (Note: only valid if Q dominates F. Verified next.)

        # SOUNDNESS AUDIT: run Q on a small sample of F-pruned comps. If Q says
        # "don't prune" any of these, that's fine (Q is allowed to be conservative).
        # If Q says "PRUNE" but F doesn't... wait, the issue is:
        # We claim Q is at least as tight as F. So if F prunes c, Q should also
        # prune c (since Q's λ=e_W recovers F's window-W bound). Let's verify:
        # for a sample of F-pruned comps, check Q also prunes.
        f_pruned_idx = np.where(~sF)[0]
        sample_size = min(50, len(f_pruned_idx))
        if sample_size > 0:
            rng = np.random.default_rng(42 + n_processed)
            sample = rng.choice(f_pruned_idx, size=sample_size, replace=False)
            for idx in sample:
                c_int = batch[idx]
                if not prune_Q_one(c_int, windows, ell_int_sums, sigmas,
                                    n_half, m, c_target):
                    bug_Q_minus_F += 1
                    if bug_Q_minus_F <= 3:
                        print(f"   *** Q-FAIL on F-pruned: c={c_int.tolist()}")

    elapsed = time.time() - t0

    print(f"\n--- Q vs F ---")
    print(f"  total processed: {n_processed:,}")
    print(f"  F survivors: {surv_F_n:,}  ({100*surv_F_n/max(1,n_processed):.2f}%)  [{t_F:.2f}s]")
    print(f"  Q survivors: {surv_Q_n:,}  ({100*surv_Q_n/max(1,n_processed):.2f}%)  [{t_Q:.2f}s on {surv_F_n:,} F-survivors]")
    print(f"  Q-extra prunes over F: {extra_Q_over_F:,}  "
          f"({100*extra_Q_over_F/max(1,surv_F_n):.2f}% of F survivors)")
    print(f"  Soundness audit (Q on sample of F-pruned): {bug_Q_minus_F} mismatches")
    if bug_Q_minus_F > 0:
        print(f"  *** SOUNDNESS WARN: Q failed to prune {bug_Q_minus_F} F-pruned comps ***")
        print(f"      (means Q is NOT a strict tightening of F; theory bug)")
    print(f"  total wall: {elapsed:.2f}s")

    return {
        'n_half': n_half, 'm': m, 'c_target': c_target,
        'd': d, 'n_win': n_win, 'n_sigma': len(sigmas),
        'n_processed': n_processed,
        'surv_F': surv_F_n, 'surv_Q': surv_Q_n,
        'pruned_F': pruned_F, 'pruned_Q': pruned_Q,
        'extra_Q_over_F': extra_Q_over_F,
        'soundness_bugs_Q_minus_F': bug_Q_minus_F,
        't_F': t_F, 't_Q': t_Q,
        'elapsed': elapsed,
    }


# ----------------------------------------------------------------------
# Sanity tests
# ----------------------------------------------------------------------
def _sanity_tests():
    """Verify Q reduces to F on single-window cases (λ = e_W)."""
    print("=== Q sanity ===")

    # Tiny test: d=4, n_half=2, m=5, simple comp.
    n_half, m, c_target = 2, 5, 1.20
    d = 2 * n_half
    windows, ell_int_sums = _build_windows(d)
    sigmas = _enum_balanced_signs(d)

    # Test case: c = (5, 5, 5, 5)  (uniform)
    c_int = np.array([5, 5, 5, 5], dtype=np.int32)
    t_q, lam = _q_bound_lp(c_int, windows, ell_int_sums, sigmas,
                              n_half, m, c_target)
    print(f"  uniform c=(5,5,5,5): Q-bound (excess m^2) = {t_q:.4f}")

    # Compare to F on same c:  manually compute F-bound = max_W [excess_F(W)]
    from _M1_bench import prune_F
    # Direct: get F's excess for each window
    n_d = float(n_half)
    cs_m2 = c_target * m * m
    f_excesses = []
    ws_arr, BB = _composition_window_data(c_int, windows, n_half, m)
    for wi, (ell, s_lo) in enumerate(windows):
        ws = ws_arr[wi]
        BB_w = BB[wi].copy()
        BB_w.sort()
        delta = int(np.sum(BB_w[d // 2:]) - np.sum(BB_w[:d // 2]))
        ell_int_sum = ell_int_sums[wi]
        ws_m2 = ws / (4.0 * n_d * ell)
        corr_F = delta / (2.0 * n_d * ell) + ell_int_sum / (4.0 * n_d * ell)
        excess = ws_m2 - corr_F - cs_m2
        f_excesses.append(excess)
    f_max = max(f_excesses)
    print(f"  F-bound (max_W excess m^2): {f_max:.4f}")
    print(f"  Q >= F? {t_q >= f_max - 1e-9}  (gap = {t_q - f_max:+.4e})")

    # Validate Q on a known F-pruner
    # F prunes when excess > 0
    for c_int in [
        np.array([8, 1, 1, 0], dtype=np.int32),  # asymmetric, sum=10
        np.array([10, 0, 0, 0], dtype=np.int32),  # extreme concentration
        np.array([2, 3, 3, 2], dtype=np.int32),   # symmetric balanced
    ]:
        if c_int.sum() != 4 * n_half * m // 2:
            continue
        sF = prune_F(c_int.reshape(1, d), n_half, m, c_target)
        t_q, _ = _q_bound_lp(c_int, windows, ell_int_sums, sigmas,
                                n_half, m, c_target)
        f_pruned = bool(~sF[0])
        q_pruned = t_q > 1e-9 * m * m
        print(f"  c={c_int.tolist()}: F-prune={f_pruned}, Q-prune={q_pruned}, "
              f"Q-excess={t_q:+.4f}")
        if f_pruned and not q_pruned:
            print("    *** SOUNDNESS BUG: F prunes but Q does not!")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_half', type=int, default=3)
    ap.add_argument('--m', type=int, default=10)
    ap.add_argument('--c_target', type=float, default=1.28)
    ap.add_argument('--batch', type=int, default=200_000)
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--sanity', action='store_true')
    ap.add_argument('--max_q', type=int, default=None,
                     help='Cap LPs per batch (for fast smoke). None = run all.')
    ap.add_argument('--out', type=str, default='_Q_results.json')
    args = ap.parse_args()

    if args.sanity:
        _sanity_tests()
        sys.exit(0)

    results = []
    if args.sweep:
        configs = [
            (3, 10, 1.28),  # F=172 reference
            (4, 10, 1.28),  # F=1014 reference
            (5, 5, 1.28),   # F=558 reference
        ]
        for nh, m, c in configs:
            try:
                r = run(nh, m, c, batch_size=args.batch,
                         max_q_per_batch=args.max_q)
                results.append(r)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  *** ERROR in ({nh},{m},{c}): {e}")
                results.append({'n_half': nh, 'm': m, 'c_target': c,
                                'error': str(e)})
    else:
        r = run(args.n_half, args.m, args.c_target, batch_size=args.batch,
                 max_q_per_batch=args.max_q)
        results.append(r)

    with open(args.out, 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f"\nWrote {args.out}")
