"""Pre-screen tests for Q's multi-window LP.

GOAL: Avoid running full Q-LP (HiGHS via scipy linprog) for compositions where
a cheap O(n_win) check at a particular vertex λ already proves Q-prunable
(t_opt > 0). The key observation:

    Q-LP value t_opt = max_λ min_σ g_σ(λ).

So for ANY fixed λ in the simplex, min_σ g_σ(λ) is a lower bound on t_opt. If we
can pick a λ for which this lower bound is positive, the composition is
Q-prunable WITHOUT running HiGHS.

Two prescreens:
  (a) Q-quick-uniform: λ = uniform over all windows (1/n_win each).
      min_σ g_σ(λ) closed form: A_uniform = (1/n_win) * sum_w A[:, w] over windows
      then min_σ g_σ = -|sum_σ| / (2n) ... actually it's:
          excess_Q(λ_unif) = sum_w (1/n_win)·V_w - Δ_T(λ_unif)/(2n)
      where T_j = (1/n_win) * sum_w (BB^w_j / ell_w).
      This is just a sort + closed form, microseconds.

  (b) Q-quick-Fbest: λ = e_{w*} where w* maximizes single-window F-bound.
      But F's per-window cert with σ = (top-d/2 of BB^w) is exactly F's
      excess_F(W). Since F is run before Q already, this gives nothing new.
      INSTEAD, we test the LP-vertex λ = e_w averaged over the TOP-K F-windows
      with NORMALIZED weights — this corresponds to "uniform over the windows
      that are nearly-tight in F".

      Per derivation: take W_topK = {top-K windows by F-excess}. Set λ = uniform
      over W_topK. Compute closed-form g_σ at that λ. If > 0, prune.

SOUNDNESS:
==========
Both prescreens give a LOWER BOUND on Q-LP value (since min_σ g_σ(λ) <= max_λ
min_σ g_σ(λ)). So prescreen-prune ⇒ Q-prune. Subset of Q's prunes.
"""
import os, sys, time, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched
from pruning import count_compositions

from _M1_bench import prune_F
from _Q_bench import (_build_windows, _enum_balanced_signs,
                       _composition_window_data, _q_bound_lp)


# ----------------------------------------------------------------------
# Closed-form excess at fixed λ.
# Given λ vector over windows, returns t_lambda := min_σ g_σ(λ),
# i.e. lower bound on Q-LP value.
# ----------------------------------------------------------------------
def _build_static(d, windows, ell_int_sums, n_half, m, c_target):
    """Build static (c-independent) arrays for prescreens. Reuse across all
    F-survivors of a sweep."""
    ell_arr = np.array([ell for (ell, _) in windows], dtype=np.float64)
    n_d = float(n_half)
    cs_m2 = c_target * m * m
    inv_4nl = 1.0 / (4.0 * n_d * ell_arr)
    inv_2n = 1.0 / (2.0 * n_d)
    half_d = d // 2
    return {
        'ell_arr': ell_arr,
        'inv_4nl': inv_4nl,
        'inv_2n': inv_2n,
        'cs_m2': cs_m2,
        'half_d': half_d,
        'n_d': n_d,
        'ell_int_sums_f': ell_int_sums.astype(np.float64),
    }


def _ws_BB(c_int, windows):
    """Compute ws and BB given c. Hot path; avoid the helper to share static."""
    return _composition_window_data(c_int, windows, None, None)


def _excess_at_lambda_pre(ws, BB, lam, static):
    """Closed-form excess at λ given precomputed ws, BB. No re-derivation.
    Returns excess (m^2 units).
    """
    inv_4nl = static['inv_4nl']
    ell_arr = static['ell_arr']
    cs_m2 = static['cs_m2']
    inv_2n = static['inv_2n']
    half_d = static['half_d']

    V = ws.astype(np.float64) * inv_4nl - static['ell_int_sums_f'] * inv_4nl - cs_m2
    # T_j = sum_w lam_w · BB[w,j] / ell_w
    T = (lam / ell_arr) @ BB.astype(np.float64)  # (d,)
    T_sorted = np.sort(T)
    delta_T = float(T_sorted[half_d:].sum() - T_sorted[:half_d].sum())
    return float((lam * V).sum()) - delta_T * inv_2n


# ----------------------------------------------------------------------
# Prescreen (a): uniform λ over ALL windows (precomputed ws/BB).
# ----------------------------------------------------------------------
def prescreen_uniform_pre(ws, BB, static, lam_unif, margin):
    e = _excess_at_lambda_pre(ws, BB, lam_unif, static)
    return e > margin * static['cs_m2'] / float(static['cs_m2']) * (margin * static['cs_m2'] / max(static['cs_m2'], 1)) and False, e
    # NOTE: incorrect comparison expression above; use simple form below.


def prescreen_uniform_simple(ws, BB, static, lam_unif, m, margin):
    e = _excess_at_lambda_pre(ws, BB, lam_unif, static)
    return e > margin * m * m, e


# ----------------------------------------------------------------------
# Prescreen (b): uniform λ over top-K F-windows (by F-excess).
# ----------------------------------------------------------------------
def _f_excess_per_window_pre(ws, BB, static):
    """Compute F's per-window excess given precomputed ws, BB."""
    inv_4nl = static['inv_4nl']
    ell_arr = static['ell_arr']
    cs_m2 = static['cs_m2']
    half_d = static['half_d']

    BB_sorted = np.sort(BB, axis=1)
    delta_BB = BB_sorted[:, half_d:].sum(axis=1) - BB_sorted[:, :half_d].sum(axis=1)
    excess = (ws.astype(np.float64) * inv_4nl
              - delta_BB.astype(np.float64) / (2.0 * static['n_d'] * ell_arr)
              - static['ell_int_sums_f'] * inv_4nl
              - cs_m2)
    return excess


def prescreen_topK_pre(ws, BB, static, K, m, margin):
    n_win = ws.shape[0]
    f_exc = _f_excess_per_window_pre(ws, BB, static)
    k_eff = min(K, n_win)
    top_idx = np.argpartition(-f_exc, k_eff - 1)[:k_eff]
    lam = np.zeros(n_win)
    lam[top_idx] = 1.0 / k_eff
    e = _excess_at_lambda_pre(ws, BB, lam, static)
    return e > margin * m * m, e


# ----------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------
def run_test(n_half, m, c_target, batch_size=200_000, K_topk=8):
    d = 2 * n_half
    S_half = 2 * n_half * m
    print(f"\n=== Prescreen test: n_half={n_half}, m={m}, "
          f"c_target={c_target}, d={d} ===")

    windows, ell_int_sums = _build_windows(d)
    n_win = len(windows)
    sigmas = _enum_balanced_signs(d)
    print(f"  n_win={n_win}, n_sigma={len(sigmas)}")

    static = _build_static(d, windows, ell_int_sums, n_half, m, c_target)
    lam_unif = np.full(n_win, 1.0 / n_win)

    # JIT warmup
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = 2 * m
    prune_F(warm, n_half, m, c_target)

    # Collect F-survivors
    f_surv = []
    n_processed = 0
    t0 = time.time()
    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=batch_size):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        n_processed += len(batch)
        sF = prune_F(batch, n_half, m, c_target)
        f_idx = np.where(sF)[0]
        for idx in f_idx:
            f_surv.append(batch[idx].copy())
    n_F = len(f_surv)
    print(f"  Processed {n_processed:,}; F-survivors: {n_F}")
    if n_F == 0:
        return {'n_half': n_half, 'm': m, 'c_target': c_target, 'd': d,
                'n_F': 0, 'note': 'no F-survivors'}

    # Run prescreens + full Q on each F-survivor; record decisions and times
    t_pre_unif = []
    t_pre_topk = []
    t_ws_bb = []  # cost of computing ws/BB (shared by both prescreens)
    t_full_q = []
    pre_unif_decisions = np.zeros(n_F, dtype=bool)
    pre_topk_decisions = np.zeros(n_F, dtype=bool)
    full_q_decisions = np.zeros(n_F, dtype=bool)
    margin = 1e-9

    for i, c_int in enumerate(f_surv):
        # Compute ws / BB once (shared by both prescreens)
        ts = time.perf_counter()
        ws, BB = _composition_window_data(c_int, windows, n_half, m)
        t_ws_bb.append(time.perf_counter() - ts)

        # Prescreen (a)
        ts = time.perf_counter()
        prune_unif, _ = prescreen_uniform_simple(ws, BB, static, lam_unif,
                                                    m, margin)
        t_pre_unif.append(time.perf_counter() - ts)
        pre_unif_decisions[i] = prune_unif

        # Prescreen (b)
        ts = time.perf_counter()
        prune_topk, _ = prescreen_topK_pre(ws, BB, static, K_topk, m, margin)
        t_pre_topk.append(time.perf_counter() - ts)
        pre_topk_decisions[i] = prune_topk

        # Full Q LP (recomputes its own ws/BB internally; that's how the
        # current pipeline runs, so we measure it as-is)
        ts = time.perf_counter()
        t_opt, _ = _q_bound_lp(c_int, windows, ell_int_sums, sigmas,
                                 n_half, m, c_target)
        t_full_q.append(time.perf_counter() - ts)
        full_q_decisions[i] = t_opt > margin * m * m

    t_pre_unif = np.array(t_pre_unif)
    t_pre_topk = np.array(t_pre_topk)
    t_ws_bb = np.array(t_ws_bb)
    t_full_q = np.array(t_full_q)

    # Counts
    n_full_prune = int(full_q_decisions.sum())
    n_unif_prune = int(pre_unif_decisions.sum())
    n_topk_prune = int(pre_topk_decisions.sum())

    # Soundness: prescreen_prune must imply full_q_prune
    soundness_unif_violations = int(np.sum(pre_unif_decisions & ~full_q_decisions))
    soundness_topk_violations = int(np.sum(pre_topk_decisions & ~full_q_decisions))

    # Coverage: how many full-Q-prunes does each prescreen catch
    cover_unif = int(np.sum(pre_unif_decisions & full_q_decisions))
    cover_topk = int(np.sum(pre_topk_decisions & full_q_decisions))

    # Combined prescreens (any one fires)
    pre_any = pre_unif_decisions | pre_topk_decisions
    n_any_prune = int(pre_any.sum())
    cover_any = int(np.sum(pre_any & full_q_decisions))
    soundness_any_violations = int(np.sum(pre_any & ~full_q_decisions))

    # Time savings: prescreen-pruned compositions skip full Q LP
    avg_unif_us = float(t_pre_unif.mean() * 1e6)
    avg_topk_us = float(t_pre_topk.mean() * 1e6)
    avg_ws_bb_us = float(t_ws_bb.mean() * 1e6)
    avg_q_ms = float(t_full_q.mean() * 1e3)

    # Total time with pre-screen vs without
    t_no_prescreen = float(t_full_q.sum())
    # With prescreen (any fires): pay ws/BB + both prescreens for all,
    # then full Q only for !any
    t_with_prescreen = float(t_ws_bb.sum() + t_pre_unif.sum() + t_pre_topk.sum()
                                + t_full_q[~pre_any].sum())

    # Also report fraction of Q-LPs avoided
    if n_full_prune > 0:
        pct_avoid = 100.0 * n_any_prune / n_F
    else:
        pct_avoid = 0.0

    results = {
        'n_half': n_half, 'm': m, 'c_target': c_target, 'd': d,
        'n_win': n_win, 'n_F': n_F,
        'full_q_prunes': n_full_prune,
        'q_survivors': n_F - n_full_prune,
        'prescreen_unif': {
            'prunes': n_unif_prune,
            'cover_of_full_q': cover_unif,
            'soundness_violations': soundness_unif_violations,
            'avg_us': avg_unif_us,
        },
        'prescreen_topk': {
            'K': K_topk,
            'prunes': n_topk_prune,
            'cover_of_full_q': cover_topk,
            'soundness_violations': soundness_topk_violations,
            'avg_us': avg_topk_us,
        },
        'ws_bb_avg_us': avg_ws_bb_us,
        'prescreen_any': {
            'prunes': n_any_prune,
            'cover_of_full_q': cover_any,
            'soundness_violations': soundness_any_violations,
        },
        'full_q_avg_ms': avg_q_ms,
        't_no_prescreen_s': t_no_prescreen,
        't_with_prescreen_s': t_with_prescreen,
        'speedup': t_no_prescreen / max(t_with_prescreen, 1e-12),
        'lps_avoided_pct': pct_avoid,
    }

    print(f"  Full Q LPs prune: {n_full_prune}/{n_F}  "
          f"({100*n_full_prune/n_F:.1f}%)")
    print(f"  ws/BB compute (shared) : avg {avg_ws_bb_us:.1f} us")
    print(f"  Prescreen (a) uniform : prunes {n_unif_prune}, "
          f"cover {cover_unif}/{n_full_prune}, soundness viol {soundness_unif_violations}, "
          f"avg {avg_unif_us:.1f} us")
    print(f"  Prescreen (b) top-{K_topk}     : prunes {n_topk_prune}, "
          f"cover {cover_topk}/{n_full_prune}, soundness viol {soundness_topk_violations}, "
          f"avg {avg_topk_us:.1f} us")
    print(f"  Prescreen (any)        : prunes {n_any_prune}, "
          f"cover {cover_any}/{n_full_prune}, soundness viol {soundness_any_violations}")
    print(f"  Full Q LP avg          : {avg_q_ms:.2f} ms")
    print(f"  Time without prescreen : {t_no_prescreen:.2f} s")
    print(f"  Time with prescreen    : {t_with_prescreen:.2f} s "
          f"(speedup {results['speedup']:.2f}x)")
    print(f"  LPs avoided            : {pct_avoid:.1f}% "
          f"({n_any_prune}/{n_F})")
    print(f"  Total wall:            {time.time()-t0:.1f} s")
    return results


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_half', type=int, default=4)
    ap.add_argument('--m', type=int, default=10)
    ap.add_argument('--c_target', type=float, default=1.28)
    ap.add_argument('--K', type=int, default=8,
                     help='Top-K windows for prescreen (b).')
    ap.add_argument('--out', type=str, default='_smoke_prescreen.json')
    ap.add_argument('--multi', action='store_true',
                     help='Run multiple K values for prescreen (b)')
    args = ap.parse_args()

    all_results = []
    if args.multi:
        for K in [1, 4, 8, 16, 32]:
            r = run_test(args.n_half, args.m, args.c_target, K_topk=K)
            all_results.append(r)
    else:
        r = run_test(args.n_half, args.m, args.c_target, K_topk=args.K)
        all_results.append(r)

    with open(args.out, 'w') as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\nWrote {args.out}")
