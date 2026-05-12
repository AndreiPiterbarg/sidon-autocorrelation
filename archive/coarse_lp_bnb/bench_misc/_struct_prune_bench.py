"""Structural pruning audit for the Cloninger-Steinerberger cascade.

GOAL
====
Find SOUND structural prunes (beyond reversal canonicalization + asymmetry +
energy cap) that can reduce L0 survivor count for the C&S cascade.

VERDICT (read this first)
=========================
After auditing the candidate principles in the task brief against the
published literature (C&S 2017 arXiv:1403.7988, Matolcsi-Vinuesa 2010
arXiv:0907.1379, the in-repo audit `delsarte_dual/ideas_rearrangement.md`,
and the cascade's own proof artefact `proof/cs-proof/lower_bound_proof.tex`),
ALL FIVE candidate principles in the brief are UNSOUND for the LOWER-bound
problem on C_{1a}.  The bench below quantifies how much pruning each WOULD
yield IF used, but flags every one of them as UNSOUND.

(1) Even-extremizer / palindromic c -- UNSOUND.
    The Schinzel-Schmidt conjecture stated that the extremizer of
    inf_f sup_t (f*f)(t) is symmetric.  Matolcsi-Vinuesa 2010
    (arXiv:0907.1379) DISPROVED this conjecture by explicit construction
    of an asymmetric f beating any symmetric candidate.  The current
    upper bound 1.5029 traces back to such asymmetric constructions.
    Riesz rearrangement (Lieb-Loss Thm 3.7) gives only
        sup_t (f^* * f^*)(t) >= sup_t (f*f)(t),
    i.e. symmetric-decreasing rearrangement INCREASES the autoconv sup.
    For an inf_f, this means restricting to symmetric f gives an UPPER
    bound on C_{1a}, NOT a lower bound.  The symmetric-decreasing class
    has answer 2 (uniform achieves it), far outside [1.2802, 1.5029].
    See `delsarte_dual/ideas_rearrangement.md` (verdict line 4) and
    `delsarte_dual/path_a_unconditional_holder/derivation.md` Attack 4
    (line 276): "rearrangement runs the wrong direction."

(2) Unimodality of c -- UNSOUND, same reason.
    Unimodal symmetric f is a strict subclass of symmetric-decreasing
    (after centering), which already gives the wrong direction.

(3) Nonneg-Fourier (Boyer-Li 2025 style) -- NOT a structural prune.
    Boyer-Li 2025 (arXiv:2506.16750) construct an explicit asymmetric g
    with ||g*g||_inf ~= 1.652, refuting MO's restricted form Hyp_R(1.51).
    This produces specific counterexamples but no general structural
    Fourier-positivity restriction on the extremizer.

(4) Convex support boundary -- ALREADY HANDLED implicitly.
    The cascade enumerates compositions on a FIXED support [-1/4, 1/4].
    Compositions with leading/trailing zero bins effectively shrink
    support; they are admissible and already counted.

(5) Mass concentration bounds -- ALREADY USED (asymmetry filter).
    The asymmetry threshold (Theorem `lower_bound_proof.tex:thm:asymmetry`)
    already prunes c with left_frac >= sqrt(c_target/2) ~= 0.8.

WHAT IS RIGOROUSLY ALREADY USED IN THE CASCADE
==============================================
- Reversal canonicalization c <= rev(c)  (factor ~2 reduction)
  [proof: Theorem `lower_bound_proof.tex:thm:reversal`]
- Asymmetry pruning left_frac < sqrt(c_target/2) and > 1-sqrt(c_target/2)
  [proof: Theorem `lower_bound_proof.tex:thm:asymmetry`]
- Energy cap c_j <= floor(m * sqrt(c_target / d_child))  (single-bin)
  [proof: Proposition `lower_bound_proof.tex:subsec:energy-cap`]
- Per-window dynamic threshold via local W_int  (variant A)
- Variant D: (ell-1)*W_int/(2n*ell) + 3*ell_int_sum/(4n*ell) correction
- Variant F: LP-tight Sigma-delta=0 correction Δ_BB/(2n*ell)
  [proof: M1 derivation in `_M1_bench.py` lines 1-52]

WHAT THIS BENCH MEASURES
========================
For the variant F survivor set at small (n_half, m, c_target), we count
how many pass each candidate "structural" filter.  This quantifies the
upper bound on potential gain IF a sound version of each filter were
ever discovered.  Since none is sound today, none is claimed as a valid
prune; the numbers are reported only to inform future research about
where the redundancy lies.

USAGE
=====
    py _struct_prune_bench.py --n_half 3 --m 10 --c_target 1.28
    py _struct_prune_bench.py --all      # the three brief-mandated configs

NOTE: at (5,5,1.28) the FULL fine-grid (S=4nm=100, d=10) has C(109,9)~=
4.3e12 compositions which is intractable.  The bench therefore enumerates
only canonical asymmetry-feasible compositions and uses an outer cap on
total enumeration via a hard time budget; if the budget is exceeded,
the bench reports the partial result.
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import numba
from numba import njit, prange

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_compositions_batched
from pruning import (
    asymmetry_prune_mask,
    asymmetry_threshold,
    count_compositions,
    _canonical_mask,
)


# ----------------------------------------------------------------------
# Variant F prune kernel (copied from _M1_bench.py for self-containment)
# ----------------------------------------------------------------------
@njit(parallel=True, cache=True)
def prune_F(batch_int, n_half, m, c_target):
    """Variant F (LP-tight Σδ=0 correction) — rigorous baseline."""
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

    half_d = d // 2

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

                BB_sorted = np.sort(BB)
                sum_top = np.int64(0)
                for k in range(half_d, d):
                    sum_top += BB_sorted[k]
                sum_bot = np.int64(0)
                for k in range(half_d):
                    sum_bot += BB_sorted[k]
                delta_BB = sum_top - sum_bot

                ell_int_sum = ell_prefix[s_lo + n_cv] - ell_prefix[s_lo]
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


# ----------------------------------------------------------------------
# Candidate (UNSOUND) structural filters — measured for academic interest
# ----------------------------------------------------------------------
@njit(cache=True)
def is_palindrome(c):
    d = len(c)
    for i in range(d // 2):
        if c[i] != c[d - 1 - i]:
            return False
    return True


@njit(cache=True)
def is_unimodal(c):
    """Strictly: c is unimodal iff it is non-decreasing up to a peak then
    non-increasing.
    """
    d = len(c)
    k = 0
    while k + 1 < d and c[k] <= c[k + 1]:
        k += 1
    for i in range(k, d - 1):
        if c[i] < c[i + 1]:
            return False
    return True


@njit(cache=True)
def is_unimodal_centered(c):
    """Stricter: peak in middle third."""
    d = len(c)
    if not is_unimodal(c):
        return False
    peak = 0
    pv = c[0]
    for i in range(1, d):
        if c[i] > pv:
            pv = c[i]
            peak = i
    return d // 3 <= peak <= 2 * d // 3


@njit(cache=True)
def is_no_interior_zeros(c):
    """No zero bins strictly between first and last nonzero bin."""
    d = len(c)
    lo = -1
    hi = -1
    for i in range(d):
        if c[i] > 0:
            if lo < 0:
                lo = i
            hi = i
    if lo < 0:
        return True
    for i in range(lo, hi + 1):
        if c[i] == 0:
            return False
    return True


@njit(cache=True)
def is_concentration_band(c, total):
    """Mass within middle 20% of bins is in [20%, 80%] of total."""
    d = len(c)
    lo = (2 * d) // 5  # 40%
    hi = (3 * d) // 5  # 60%
    s = np.int64(0)
    for i in range(lo, hi):
        s += c[i]
    frac = float(s) / float(total)
    return 0.20 <= frac <= 0.80


@njit(cache=True)
def has_nonneg_dft(c):
    """DFT-positivity: discrete cosine coefficients all non-negative."""
    d = len(c)
    tol = 1e-9
    for k in range(1, (d // 2) + 1):
        total = 0.0
        for n in range(d):
            total += c[n] * np.cos(2.0 * np.pi * k * n / d)
        if total < -tol:
            return False
    return True


@njit(parallel=True, cache=True)
def apply_filter(batch_int, total_per_row, filter_id):
    """filter_id: 0=palindromic, 1=unimodal, 2=unimodal-centered,
       3=no interior zeros, 4=concentration band, 5=nonneg-DFT.
    """
    B = batch_int.shape[0]
    keep = np.ones(B, dtype=numba.boolean)
    for b in prange(B):
        c = batch_int[b]
        if filter_id == 0:
            if not is_palindrome(c):
                keep[b] = False
        elif filter_id == 1:
            if not is_unimodal(c):
                keep[b] = False
        elif filter_id == 2:
            if not is_unimodal_centered(c):
                keep[b] = False
        elif filter_id == 3:
            if not is_no_interior_zeros(c):
                keep[b] = False
        elif filter_id == 4:
            if not is_concentration_band(c, total_per_row[b]):
                keep[b] = False
        elif filter_id == 5:
            if not has_nonneg_dft(c):
                keep[b] = False
    return keep


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
FILTERS = [
    ('palindromic',           0, 'UNSOUND',
     'MV-2010 disproved Schinzel-Schmidt; Riesz wrong direction.'),
    ('unimodal',              1, 'UNSOUND',
     'Subclass of symmetric-decreasing; sym-dec bound = 2.'),
    ('unimodal-centered',     2, 'UNSOUND',
     'Strict subclass of unimodal — same obstruction.'),
    ('no-interior-zeros',     3, 'CONJECTURAL',
     'No published theorem; gaps possible (Cantor-like).'),
    ('mass-concentration',    4, 'PARTIALLY USED',
     'Asymmetry filter already exploits the rigorous part.'),
    ('nonneg-DFT',            5, 'CONJECTURAL',
     'No published Fourier-positivity theorem for our extremizer.'),
]


def run_bench(n_half, m, c_target, batch_size=200_000, time_budget=180.0,
              verbose=True):
    """Run the bench at one config.

    A hard time budget of `time_budget` seconds is imposed on the
    enumeration loop.  If exceeded, the bench reports partial counts and
    flags the result.
    """
    d = 2 * n_half
    S = 4 * n_half * m
    n_total = count_compositions(d, S)
    if verbose:
        print(f"\n=== struct prune bench: n_half={n_half}, m={m}, "
              f"c_target={c_target}, d={d}, S={S} ===")
        print(f"  total compositions = {n_total:,}")

    # Warm up JIT
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = S
    sys.stdout.write("  [JIT warming...] ")
    sys.stdout.flush()
    prune_F(warm, n_half, m, c_target)
    apply_filter(warm, np.array([S], dtype=np.int64), 0)
    print("done.")
    sys.stdout.flush()

    n_processed = 0
    n_canonical = 0
    n_asym_feas = 0
    n_F_surv = 0
    F_survivors = []
    F_survivor_totals = []
    t0 = time.time()
    last_report = t0
    timed_out = False

    for batch in generate_compositions_batched(d, S, batch_size=batch_size):
        n_processed += len(batch)

        canon = _canonical_mask(batch)
        batch_c = batch[canon]
        n_canonical += len(batch_c)
        if len(batch_c) == 0:
            now = time.time()
            if now - last_report > 5.0:
                print(f"    [{n_processed:,} processed, {now-t0:.1f}s]")
                sys.stdout.flush()
                last_report = now
            if now - t0 > time_budget:
                timed_out = True
                break
            continue

        asym = asymmetry_prune_mask(batch_c, n_half, m, c_target)
        batch_ca = batch_c[asym]
        n_asym_feas += len(batch_ca)
        if len(batch_ca) == 0:
            now = time.time()
            if now - last_report > 5.0:
                print(f"    [{n_processed:,} processed, "
                      f"{n_canonical:,} canon, {n_asym_feas:,} asym, "
                      f"{now-t0:.1f}s]")
                sys.stdout.flush()
                last_report = now
            if now - t0 > time_budget:
                timed_out = True
                break
            continue

        sF = prune_F(batch_ca, n_half, m, c_target)
        survivors = batch_ca[sF]
        n_F_surv += len(survivors)
        if len(survivors) > 0:
            F_survivors.append(survivors.copy())
            F_survivor_totals.append(
                np.full(len(survivors), S, dtype=np.int64))

        now = time.time()
        if now - last_report > 5.0:
            print(f"    [{n_processed:,} processed, "
                  f"{n_canonical:,} canon, {n_asym_feas:,} asym, "
                  f"{n_F_surv:,} F-surv, {now-t0:.1f}s]")
            sys.stdout.flush()
            last_report = now
        if now - t0 > time_budget:
            timed_out = True
            break

    elapsed_F = time.time() - t0

    if F_survivors:
        F_survivors_arr = np.vstack(F_survivors)
        F_survivor_totals_arr = np.concatenate(F_survivor_totals)
    else:
        F_survivors_arr = np.empty((0, d), dtype=np.int32)
        F_survivor_totals_arr = np.empty(0, dtype=np.int64)

    if verbose:
        print(f"\n  Pipeline (rigorous baseline):")
        if timed_out:
            print(f"    *** TIMED OUT after {time_budget:.1f}s "
                  f"(processed {n_processed:,} / {n_total:,}) ***")
        print(f"    canonical (c<=rev(c)):         {n_canonical:,}")
        print(f"    asymmetry-feasible:            {n_asym_feas:,}")
        print(f"    variant F survivors:           {n_F_surv:,}")
        print(f"    elapsed: {elapsed_F:.2f}s")
        sys.stdout.flush()

    results = {
        'config': {'n_half': n_half, 'm': m, 'c_target': c_target,
                   'd': d, 'S': S},
        'n_total': n_total,
        'n_processed': n_processed,
        'timed_out': timed_out,
        'n_canonical': n_canonical,
        'n_asym_feas': n_asym_feas,
        'n_F_surv': n_F_surv,
        'elapsed_F': elapsed_F,
        'filters': [],
    }

    if n_F_surv == 0:
        if verbose:
            print(f"\n  (no F survivors -- nothing to filter)")
        return results

    if verbose:
        print(f"\n  Candidate structural filters applied to F survivors "
              f"(n={n_F_surv:,}):")
        print(f"  {'name':<20s} {'remain':>10s} {'removed':>10s} "
              f"{'soundness':>16s}  {'note'}")
        print(f"  {'-'*20:<20s} {'-'*10:>10s} {'-'*10:>10s} "
              f"{'-'*16:>16s}  {'-'*40}")
        sys.stdout.flush()

    for name, fid, soundness, note in FILTERS:
        keep = apply_filter(F_survivors_arr, F_survivor_totals_arr, fid)
        n_remain = int(np.sum(keep))
        n_removed = n_F_surv - n_remain
        results['filters'].append({
            'name': name,
            'n_remain': n_remain,
            'n_removed': n_removed,
            'soundness': soundness,
            'note': note,
        })
        if verbose:
            print(f"  {name:<20s} {n_remain:>10,d} {n_removed:>10,d} "
                  f"{soundness:>16s}  {note}")
            sys.stdout.flush()

    if verbose:
        print(f"\n  RIGOROUS verdict at this config:")
        print(f"    Cascade L0 (canonical+asym+F) survivors = {n_F_surv:,}.")
        print(f"    None of the 6 listed structural filters above is rigorously")
        print(f"    sound for the lower-bound problem on C_{{1a}}.")
        print(f"    Any 'gain' from an UNSOUND filter is illusory: it would")
        print(f"    miss real extremizers (MV-2010-style asymmetric f).")
        sys.stdout.flush()

    return results


def run_palindromic_bench(n_half, m, c_target, batch_size=200_000,
                            verbose=True):
    """Enumerate ONLY palindromic compositions (M1-style baseline).

    For c=(c_0, ..., c_{2n_half-1}) palindromic, the half-vector
    (c_0, ..., c_{n_half-1}) is unconstrained except sum = 2*n_half*m.
    So we enumerate compositions of length n_half summing to 2*n_half*m
    and mirror.  Then apply asymmetry + F filter.
    """
    d = 2 * n_half
    S_full = 4 * n_half * m
    S_half = 2 * n_half * m
    n_total = count_compositions(n_half, S_half)
    if verbose:
        print(f"\n=== palindromic bench: n_half={n_half}, m={m}, "
              f"c_target={c_target}, d={d}, S_full={S_full}, "
              f"S_half={S_half} ===")
        print(f"  total palindromic = {n_total:,}")

    # Warm up JIT
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = S_full
    prune_F(warm, n_half, m, c_target)
    apply_filter(warm, np.array([S_full], dtype=np.int64), 0)

    n_processed = 0
    n_asym_feas = 0
    n_F_surv = 0
    F_survivors = []
    F_survivor_totals = []
    t0 = time.time()

    for half_batch in generate_compositions_batched(n_half, S_half,
                                                      batch_size=batch_size):
        batch = np.empty((len(half_batch), d), dtype=np.int32)
        batch[:, :n_half] = half_batch
        batch[:, n_half:] = half_batch[:, ::-1]
        n_processed += len(batch)

        asym = asymmetry_prune_mask(batch, n_half, m, c_target)
        batch_a = batch[asym]
        n_asym_feas += len(batch_a)
        if len(batch_a) == 0:
            continue

        sF = prune_F(batch_a, n_half, m, c_target)
        survivors = batch_a[sF]
        n_F_surv += len(survivors)
        if len(survivors) > 0:
            F_survivors.append(survivors.copy())
            F_survivor_totals.append(
                np.full(len(survivors), S_full, dtype=np.int64))

    elapsed = time.time() - t0

    if F_survivors:
        F_survivors_arr = np.vstack(F_survivors)
        F_survivor_totals_arr = np.concatenate(F_survivor_totals)
    else:
        F_survivors_arr = np.empty((0, d), dtype=np.int32)
        F_survivor_totals_arr = np.empty(0, dtype=np.int64)

    if verbose:
        print(f"  total palindromic processed: {n_processed:,}")
        print(f"  asymmetry-feasible: {n_asym_feas:,}")
        print(f"  variant F survivors: {n_F_surv:,}")
        print(f"  elapsed: {elapsed:.2f}s")

    results = {
        'mode': 'palindromic',
        'config': {'n_half': n_half, 'm': m, 'c_target': c_target,
                   'd': d, 'S_full': S_full, 'S_half': S_half},
        'n_total_palindromic': n_total,
        'n_processed': n_processed,
        'n_asym_feas': n_asym_feas,
        'n_F_surv': n_F_surv,
        'elapsed_F': elapsed,
        'filters': [],
    }

    if n_F_surv == 0:
        return results

    if verbose:
        print(f"\n  Filters applied to {n_F_surv:,} palindromic F-survivors:")
        print(f"  {'name':<20s} {'remain':>10s} {'removed':>10s} "
              f"{'soundness':>16s}")

    for name, fid, soundness, note in FILTERS:
        keep = apply_filter(F_survivors_arr, F_survivor_totals_arr, fid)
        n_remain = int(np.sum(keep))
        n_removed = n_F_surv - n_remain
        results['filters'].append({
            'name': name,
            'n_remain': n_remain,
            'n_removed': n_removed,
            'soundness': soundness,
            'note': note,
        })
        if verbose:
            print(f"  {name:<20s} {n_remain:>10,d} {n_removed:>10,d} "
                  f"{soundness:>16s}")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_half', type=int, default=3)
    ap.add_argument('--m', type=int, default=10)
    ap.add_argument('--c_target', type=float, default=1.28)
    ap.add_argument('--batch', type=int, default=200_000)
    ap.add_argument('--time_budget', type=float, default=180.0)
    ap.add_argument('--out', type=str, default='_struct_prune_bench.json')
    ap.add_argument('--all', action='store_true',
                    help='Run all three test configs.')
    ap.add_argument('--palindromic_only', action='store_true',
                    help='Only enumerate palindromic compositions '
                         '(tractable at larger d).')
    args = ap.parse_args()

    if args.all:
        configs = [(3, 10, 1.28), (4, 10, 1.28), (5, 5, 1.28)]
    else:
        configs = [(args.n_half, args.m, args.c_target)]

    results_all = []
    for n_half, m, c_target in configs:
        if args.palindromic_only:
            r = run_palindromic_bench(n_half, m, c_target,
                                       batch_size=args.batch)
        else:
            r = run_bench(n_half, m, c_target, batch_size=args.batch,
                          time_budget=args.time_budget)
        results_all.append(r)

    with open(args.out, 'w') as f:
        json.dump(results_all, f, indent=2, default=int)

    print(f"\n\nResults written to {args.out}")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
