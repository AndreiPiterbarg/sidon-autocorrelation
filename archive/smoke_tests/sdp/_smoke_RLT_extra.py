"""Smoke test: extended RLT cuts on Shor SDP at d=6.

Compares baseline order-1 Shor (current `_L_bench._shor_feasibility`) vs
an extended variant with extra sound RLT-style cuts.

Setup: n_half=3 (d=6), m=10, c_target=1.28, palindromic compositions.

Filter chain: F -> Q -> [L_baseline | L_extra] (tested on Q-survivors).

Extra cuts considered (must be SOUND, i.e. valid for any (x, X = x x^T) feasible):

  (E1) Sum-of-products identity:  sum_i X[i,i] + 2 * sum_{i<j} X[i,j] = (sum_i x_i)^2 = S^2
       This is an EQUALITY implied by sum x = S (trace identity for X = x x^T)
       AND the RLT lift -- but the SDP only has the lift loosely PSD-relaxed,
       so the equality on X is NOT automatic -- adding it AS AN EQUALITY
       constraint is sound and tightens.

  (E2) Diagonal lower bound from sum:  sum_i X[i,i] >= S^2 / d  (Cauchy-Schwarz on uniform).
       Sound via Jensen / CS:  S^2 = (sum x_i)^2 <= d * sum x_i^2 = d * sum X_ii.

  (E3) Pairwise McCormick refinement using SUM info:
       For any pair (i,j), x_i + x_j is bounded:
           lo_i + lo_j  <=  x_i + x_j  <=  hi_i + hi_j
       AND  x_i + x_j  =  S - sum_{k!=i,j} x_k  in [S - sum_k!=ij hi_k, S - sum_k!=ij lo_k].
       Combine these for tighter bounds on x_i + x_j, then derive
           X[i,i] + X[j,j] + 2*X[i,j] = (x_i+x_j)^2  <=  upper(x_i+x_j)^2
           (via secant)
       But (x_i+x_j)^2 is convex; replace with secant linearization
       between best-known lo/hi for x_i+x_j.

  (E4) (NEW) 3rd-order RLT on a specific TRIPLE (i,j,k):
       Lift Y_ijk and add the cut from product of 3 lower-bound constraints:
           (x_i - lo_i)(x_j - lo_j)(x_k - lo_k) >= 0
       Linearize:  Y_ijk - lo_i*X_jk - lo_j*X_ik - lo_k*X_ij
                   + lo_i lo_j x_k + lo_i lo_k x_j + lo_j lo_k x_i
                   - lo_i lo_j lo_k >= 0
       This requires lifting a single triple monomial Y_ijk, with PSD coupling
       implicit (we only add the LINEAR cuts, not full M_2 PSD).
       SOUND: any (x, X = xx^T, Y = xxx) extends the feasible set.

For SOUNDNESS, we add (E1), (E2), (E3), (E4) only via inequalities valid for
ALL (x, X = x x^T) feasible.  No false-prune introduced.

Output: prune count and time for baseline vs extra. Verdict at end.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np
from itertools import combinations

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))

from compositions import generate_compositions_batched
from pruning import count_compositions
from _M1_bench import prune_F
from _Q_bench import _build_windows, prune_Q_one, _enum_balanced_signs
from _L_bench import _build_A_matrices, _shor_feasibility, _make_cell


# ----------------------------------------------------------------------
# Extended Shor SDP feasibility with extra cuts
# ----------------------------------------------------------------------
def _shor_feasibility_extra(c_int, lo, hi, A_mats, windows, n_half, m, c_target,
                              solver='MOSEK', tol=1e-9, eps_margin=1e-9,
                              triple_cuts: int = 4, verbose=False):
    """Extended Shor SDP with E1+E2+E3+E4 extra cuts.

    triple_cuts: number of (i,j,k) consecutive triples to add 3rd-order RLT for.
    """
    import cvxpy as cp

    d = len(c_int)
    nm = float(4 * n_half * m)
    cs_m2 = float(c_target) * m * m
    eps_thr = eps_margin * m * m
    S = nm  # sum mass

    x = cp.Variable(d)
    X = cp.Variable((d, d), symmetric=True)

    ones11 = np.ones((1, 1))
    Y = cp.bmat([[ones11, cp.reshape(x, (1, d), order='C')],
                  [cp.reshape(x, (d, 1), order='C'), X]])

    cons = [Y >> 0]
    cons += [x >= lo, x <= hi]
    cons += [cp.sum(x) == nm]

    # Baseline diagonal McCormick + range
    for i in range(d):
        cons += [X[i, i] >= lo[i] * lo[i], X[i, i] <= hi[i] * hi[i]]
        cons += [X[i, i] >= 2.0 * lo[i] * x[i] - lo[i] * lo[i]]
        cons += [X[i, i] >= 2.0 * hi[i] * x[i] - hi[i] * hi[i]]
        cons += [X[i, i] <= (lo[i] + hi[i]) * x[i] - lo[i] * hi[i]]

    # Baseline RLT off-diagonal
    for i in range(d):
        for j in range(i + 1, d):
            li, lj = lo[i], lo[j]
            ui, uj = hi[i], hi[j]
            cons += [X[i, j] >= lj * x[i] + li * x[j] - li * lj]
            cons += [X[i, j] >= uj * x[i] + ui * x[j] - ui * uj]
            cons += [X[i, j] <= ui * x[j] + lj * x[i] - ui * lj]
            cons += [X[i, j] <= uj * x[i] + li * x[j] - li * uj]

    # ----- E1: trace identity (sum X) = S^2 -----
    # Sum_i X[i,i] + 2*sum_{i<j} X[i,j] == S^2
    sum_diag = cp.sum([X[i, i] for i in range(d)])
    sum_off = 0
    for i in range(d):
        for j in range(i + 1, d):
            sum_off = sum_off + X[i, j]
    cons += [sum_diag + 2.0 * sum_off == S * S]

    # ----- E2: Cauchy-Schwarz lower bound on diagonal sum -----
    # S^2 <= d * sum_i X[i,i]   <=>   sum_i X[i,i] >= S^2 / d
    cons += [sum_diag >= (S * S) / float(d)]

    # ----- E3: Tighter pairwise sum-of-squares secant cut -----
    # For each pair (i,j), bound s_ij = x_i + x_j in [Llo_ij, Lhi_ij] using
    # both pair box AND mass-conservation residual.  s^2 is convex:
    #   chord (UPPER): s^2 <= (Llo + Lhi)*s - Llo*Lhi   over s in [Llo, Lhi]
    #   tangent (LOWER): s^2 >= 2*s_0*s - s_0^2   for any s_0
    # We add ONLY the upper chord (the lower bound from M_1 PSD via
    # X_ii + X_jj + 2X_ij >= (x_i + x_j)^2  is already encoded by Schur).
    sum_lo = float(np.sum(lo))
    sum_hi = float(np.sum(hi))
    for i in range(d):
        for j in range(i + 1, d):
            other_lo = sum_lo - lo[i] - lo[j]
            other_hi = sum_hi - hi[i] - hi[j]
            mass_lo = S - other_hi
            mass_hi = S - other_lo
            box_lo = lo[i] + lo[j]
            box_hi = hi[i] + hi[j]
            Llo = max(box_lo, mass_lo)
            Lhi = min(box_hi, mass_hi)
            if Lhi <= Llo:
                continue
            # Upper chord: SOUND  (s^2 <= (Llo+Lhi)*s - Llo*Lhi  for s in [Llo, Lhi])
            cons += [X[i, i] + X[j, j] + 2.0 * X[i, j]
                       <= (Lhi + Llo) * (x[i] + x[j]) - Lhi * Llo]

    # ----- E4: 3rd-order RLT cuts on selected triples -----
    # For each chosen triple (i,j,k), lift Y_ijk and add lo-product cut + hi cuts.
    # Y_ijk should be a scalar that respects:
    #    (x_i - lo_i)(x_j - lo_j)(x_k - lo_k) >= 0  (positivity from box)
    # and analogous with hi:
    #    (hi_i - x_i)(hi_j - x_j)(hi_k - x_k) >= 0
    #    (x_i - lo_i)(x_j - lo_j)(hi_k - x_k) >= 0   -> sign +
    #    (x_i - lo_i)(hi_j - x_j)(x_k - lo_k) >= 0   -> sign +
    #    (hi_i - x_i)(x_j - lo_j)(x_k - lo_k) >= 0   -> sign +
    # We only add a few selective ones to avoid blowup.
    if triple_cuts > 0:
        # Pick consecutive index triples (most relevant for our autoconv windows).
        triples = []
        # Consecutive triples: (0,1,2), (1,2,3), (2,3,4), (3,4,5)
        for i0 in range(d - 2):
            triples.append((i0, i0 + 1, i0 + 2))
        # Optional: palindrome-symmetric pair (extreme indices)
        if d >= 4 and triple_cuts > len(triples):
            triples.append((0, 1, d - 1))
            triples.append((0, d - 2, d - 1))
        triples = triples[:triple_cuts]

        for (i, j, k) in triples:
            # Lift Y_ijk as a scalar variable
            Y_var = cp.Variable()
            li, lj, lk = lo[i], lo[j], lo[k]
            ui, uj, uk = hi[i], hi[j], hi[k]

            # Bound on Y from box (only valid if all bounds positive,
            # which lo>=0 holds here):
            cons += [Y_var >= 0]   # all monomials nonneg since x >= 0

            # All 8 sign patterns: (x_i - lo_i)/(hi_i - x_i), etc., product nonneg
            # Linearize:   prod_3 = Y_ijk + (lin terms in X, x, const)
            # Cut 1: (x_i - lo)(x_j - lo)(x_k - lo) >= 0
            #     = Y - lo_k X_ij - lo_j X_ik - lo_i X_jk
            #         + lo_j lo_k x_i + lo_i lo_k x_j + lo_i lo_j x_k
            #         - lo_i lo_j lo_k >= 0
            cons += [Y_var
                       - lk * X[i, j] - lj * X[i, k] - li * X[j, k]
                       + lj * lk * x[i] + li * lk * x[j] + li * lj * x[k]
                       - li * lj * lk
                       >= 0]
            # Cut 2: (hi_i - x_i)(hi_j - x_j)(hi_k - x_k) >= 0
            #   Expand:  ui*uj*uk - ui*uj*x_k - ui*uk*x_j - uj*uk*x_i
            #            + ui*X_jk + uj*X_ik + uk*X_ij - Y_ijk >= 0
            cons += [ui * uj * uk
                       - ui * uj * x[k] - ui * uk * x[j] - uj * uk * x[i]
                       + ui * X[j, k] + uj * X[i, k] + uk * X[i, j]
                       - Y_var >= 0]
            # Cut 3: (x_i - li)(x_j - lj)(uk - x_k) >= 0
            #   Expand:
            #      uk * (x_i x_j - lj x_i - li x_j + li lj)
            #    - x_k * (x_i x_j - lj x_i - li x_j + li lj)
            #    = uk X_ij - uk lj x_i - uk li x_j + uk li lj
            #      - Y + lj X_ik + li X_jk - li lj x_k
            cons += [uk * X[i, j] - uk * lj * x[i] - uk * li * x[j] + uk * li * lj
                       - Y_var + lj * X[i, k] + li * X[j, k] - li * lj * x[k]
                       >= 0]
            # Cut 4: (x_i - li)(uj - x_j)(x_k - lk) >= 0
            #   = uj X_ik - uj lk x_i - li uj x_k + li uj lk
            #     - Y + lk X_ij + li X_jk - li lk x_j
            cons += [uj * X[i, k] - uj * lk * x[i] - li * uj * x[k] + li * uj * lk
                       - Y_var + lk * X[i, j] + li * X[j, k] - li * lk * x[j]
                       >= 0]
            # Cut 5: (ui - x_i)(x_j - lj)(x_k - lk) >= 0
            #   = ui X_jk - ui lk x_j - ui lj x_k + ui lj lk
            #     - Y + lk X_ij + lj X_ik - lj lk x_i
            cons += [ui * X[j, k] - ui * lk * x[j] - ui * lj * x[k] + ui * lj * lk
                       - Y_var + lk * X[i, j] + lj * X[i, k] - lj * lk * x[i]
                       >= 0]

    # Window constraints (same as baseline)
    for A_mat, (ell, _) in zip(A_mats, windows):
        thr = 4.0 * float(n_half) * float(ell) * (cs_m2 + eps_thr)
        cons += [cp.trace(A_mat @ X) <= thr]

    prob = cp.Problem(cp.Minimize(0), cons)
    try:
        if solver == 'MOSEK':
            prob.solve(solver='MOSEK', verbose=verbose,
                        mosek_params={
                            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': tol,
                            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': tol,
                            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol,
                        })
        else:
            prob.solve(solver=solver, verbose=verbose)
    except Exception as e:
        return False, f'EXC:{type(e).__name__}'

    return prob.status == 'infeasible', prob.status


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def main():
    n_half = 3
    m = 10
    c_target = 1.28
    d = 2 * n_half
    S_full = 4 * n_half * m
    S_half = 2 * n_half * m

    print(f"=== Smoke RLT Extra: n_half={n_half}, m={m}, c={c_target}, d={d}, S={S_full}")
    n_total_half = count_compositions(n_half, S_half)
    print(f"    Palindromic comps: {n_total_half}")

    # Build LP/SDP scaffolding
    windows, ell_int_sums = _build_windows(d)
    n_win = len(windows)
    sigmas = _enum_balanced_signs(d)
    A_mats = _build_A_matrices(d, windows)
    print(f"    n_win={n_win}, n_sigma={len(sigmas)}")

    t0 = time.time()
    # Stage F
    n_total = 0
    f_survivors = []
    for half_batch in generate_compositions_batched(n_half, S_half, batch_size=200_000):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        n_total += len(batch)
        sF = prune_F(batch, n_half, m, c_target)
        for k in range(len(batch)):
            if sF[k]:
                f_survivors.append(batch[k].copy())
    print(f"    F survivors: {len(f_survivors)} / {n_total}  ({time.time()-t0:.2f}s)")

    # Stage Q
    t1 = time.time()
    q_survivors = []
    for c_int in f_survivors:
        if not prune_Q_one(c_int, windows, ell_int_sums, sigmas,
                            n_half, m, c_target):
            q_survivors.append(c_int)
    print(f"    Q survivors: {len(q_survivors)} / {len(f_survivors)}  ({time.time()-t1:.2f}s)")

    if len(q_survivors) == 0:
        print("    No Q survivors -- nothing to test.")
        print("VERDICT: NO_GAIN")
        return

    # Cap survivors to keep wall time bounded
    MAX_TEST = 100
    if len(q_survivors) > MAX_TEST:
        print(f"    Capping at {MAX_TEST} for testing...")
        q_survivors = q_survivors[:MAX_TEST]

    # Stage L: baseline
    t2 = time.time()
    n_baseline_pruned = 0
    n_baseline_status = {}
    for c_int in q_survivors:
        lo, hi = _make_cell(c_int, m)
        pruned, status = _shor_feasibility(c_int, lo, hi, A_mats, windows,
                                              n_half, m, c_target,
                                              solver='MOSEK', tol=1e-9,
                                              eps_margin=1e-9)
        if pruned:
            n_baseline_pruned += 1
        n_baseline_status[status] = n_baseline_status.get(status, 0) + 1
    t_baseline = time.time() - t2
    print(f"    Baseline L (Shor order-1):   pruned {n_baseline_pruned}/{len(q_survivors)}"
          f"  in {t_baseline:.2f}s  status={n_baseline_status}")

    # Stage L: extra cuts (with 4 triple cuts)
    t3 = time.time()
    n_extra_pruned = 0
    n_extra_status = {}
    for c_int in q_survivors:
        lo, hi = _make_cell(c_int, m)
        pruned, status = _shor_feasibility_extra(c_int, lo, hi, A_mats, windows,
                                                    n_half, m, c_target,
                                                    solver='MOSEK', tol=1e-9,
                                                    eps_margin=1e-9,
                                                    triple_cuts=4)
        if pruned:
            n_extra_pruned += 1
        n_extra_status[status] = n_extra_status.get(status, 0) + 1
    t_extra = time.time() - t3
    print(f"    Extended L (Shor + E1+E2+E3+E4): pruned {n_extra_pruned}/{len(q_survivors)}"
          f"  in {t_extra:.2f}s  status={n_extra_status}")

    delta = n_extra_pruned - n_baseline_pruned
    print(f"\n    DELTA: extended pruned {delta} EXTRA over baseline")
    print(f"    Time ratio: {t_extra/max(t_baseline,1e-9):.2f}x")

    out = {
        'n_half': n_half, 'm': m, 'c_target': c_target,
        'n_total': n_total, 'n_F': len(f_survivors), 'n_Q': len(q_survivors),
        'n_baseline_L_prune': n_baseline_pruned,
        'n_extra_L_prune': n_extra_pruned,
        'delta': delta,
        't_baseline': t_baseline, 't_extra': t_extra,
    }
    with open('_smoke_RLT_extra_results.json', 'w') as f:
        json.dump(out, f, indent=2)

    if delta > 0:
        print(f"\nVERDICT: EXTRA_RLT_PRUNES_{delta}_EXTRA")
    else:
        print(f"\nVERDICT: NO_GAIN")


if __name__ == '__main__':
    main()
