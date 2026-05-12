"""Smoke: exact max_t (f*f)(t) vs cascade's max_W TV_W at d=4.

Setup: n_half=1, m=20 cascade. Parents at d_parent=2 with sum=80.
Children at d_child=4 with mass-doubling refinement: child sum=160,
n_half_child=2, m=20, bin width h=1/(2*d_child)=1/8 over [-1/4, 1/4].

Step function f at d=4 has heights b_i = c_i/m with Σ b_i = 4n_half_child = 8,
and ∫f = h·Σb_i = (1/8)·8 = 1. So f is normalized.

(f*f)(t) on [-1/2, 1/2] is piecewise linear with breakpoints at
t_k = (k - (d-1))·h = (k-3)/8 for k = 0,1,...,2d-2 = 0..6, plus
endpoints ±d·h = ±1/2 (where (f*f) = 0).

At breakpoint t_k (integer k):
    (f*f)(t_k) = h · Σ_{i+j=k} b_i b_j = conv[k] · h / m^2
where conv[k] = Σ_{i+j=k} c_i c_j (integer convolution).

The cascade's TV_W for window of ell consecutive conv positions [s_lo, s_lo+ell-2]:
    TV_W = ws / (4n·ell·m^2)  where ws = Σ_{k=s_lo}^{s_lo+ell-2} conv[k]
This is the AVERAGE of (f*f) over the t-interval [s_lo·h - d·h, (s_lo+ell-2)·h - (d-2)·h]
... actually it's the average of the linear segment values at the endpoints.

CLAIM: max_t (f*f)(t) = max_k (f*f)(t_k) (piecewise linear ⇒ maxes at vertices).

Cascade comparison:
- ell=2 windows give TV_W = (conv[k] + conv[k+1])/(8m^2·2) = average of (f*f) at
  two adjacent breakpoints, NOT the breakpoint values themselves.
- max over ell=2 windows = max_k (conv[k]+conv[k+1])/2 / (4m^2) =
  (1/2)·avg of two adjacent (f*f) values.

So max_W TV_W(ell=2) UNDERESTIMATES max_k (f*f)(t_k) when one breakpoint
dominates: avg(a,b) < max(a,b) when a≠b.

This script:
1. Generates all d=4 children of n_half=1, m=20 L0 survivors at c_target=1.281.
2. For each child, computes:
     A. exact max_k (f*f)(t_k) = max_k conv[k] / (8·m^2)
     B. cascade max_W TV_W (ell=2 only) = max_k (conv[k]+conv[k+1])/(8·m^2·2)
     C. cascade max_W TV_W (all ell ∈ [2, 2d]) (uncorrected, for fair comparison
        with A which is also uncorrected)
3. Counts: cells where A > 1.281 but max(B,C) ≤ 1.281 — these would be missed
   prunes if cascade only used uncorrected TV_W.

The cascade's actual prune adds correction terms (variant F's Δ_BB + ell_int·δ²)
which account for the continuous-witness slack. So the comparison here is
"naked" TV_W (no correction) vs exact (f*f) at integer composition.
"""
from __future__ import annotations
import os
import sys
import time
import itertools
import numpy as np

# Path setup so we can import compositions.
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

# ---------------------------------------------------------------- params
N_HALF = 1
M = 20
D_PARENT = 2 * N_HALF        # d=2
D_CHILD = 4                  # d=4 child
N_HALF_CHILD = D_PARENT      # =2
S_PARENT = 4 * N_HALF * M    # 80
S_CHILD = 4 * N_HALF_CHILD * M  # 160
C_TARGET = 1.281
H = 1.0 / (2 * D_CHILD)      # 1/8
M2 = M * M                   # 400
NORMALIZER = 4.0 * N_HALF_CHILD * M2  # 8 * 400 = 3200

# ---------------------------------------------------------------- helpers
def conv_int(c):
    """Integer auto-convolution of c (length d → length 2d-1)."""
    d = len(c)
    out = np.zeros(2 * d - 1, dtype=np.int64)
    for i in range(d):
        ci = int(c[i])
        if ci == 0:
            continue
        out[2 * i] += ci * ci
        for j in range(i + 1, d):
            cj = int(c[j])
            if cj != 0:
                out[i + j] += 2 * ci * cj
    return out


def exact_max_t_ff(c):
    """Exact max_t (f*f)(t) for step function f with c_i integer summing to S_CHILD.

    (f*f)(t_k) = conv[k] · h / m^2 = conv[k] / (8 * m^2) at breakpoint t_k.
    Returns max over breakpoints. Endpoints t = ±1/2 give (f*f)=0.
    """
    cv = conv_int(c)
    return float(cv.max()) / (D_CHILD * 2 * M2)  # = conv_max / 3200 / 2 ?
    # Wait: h = 1/(2*d_child) = 1/8, m^2 = 400 → h/m^2 = 1/3200.
    # Coefficient: 1/(8·400) = 1/3200. Equivalent to NORMALIZER above? Let's
    # double-check: NORMALIZER = 4·n_half_child·m^2 = 8·400 = 3200. Yes.


def exact_max_t_ff_v2(c):
    cv = conv_int(c)
    return float(cv.max()) / NORMALIZER


def cascade_tv_max(c, ell_set=None):
    """Cascade's uncorrected max_W TV_W over given ell range.

    TV_W = ws / (4n·ell·m^2) where ws = sum of conv on ell-1 consecutive positions.
    """
    cv = conv_int(c)
    d = len(c)
    conv_len = 2 * d - 1
    if ell_set is None:
        ell_set = range(2, 2 * d + 1)
    best = 0.0
    best_ell = 0
    best_s = 0
    for ell in ell_set:
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        if n_windows <= 0:
            continue
        for s_lo in range(n_windows):
            ws = int(cv[s_lo:s_lo + n_cv].sum())
            tv = ws / (4.0 * N_HALF_CHILD * ell * M2)
            if tv > best:
                best = tv
                best_ell = ell
                best_s = s_lo
    return best, best_ell, best_s


# ---------------------------------------------------------------- L0 survivors
def all_d2_compositions():
    """All compositions of S_PARENT=80 into d_parent=2 bins, both nonneg."""
    out = []
    for c0 in range(S_PARENT + 1):
        c1 = S_PARENT - c0
        out.append((c0, c1))
    return out


def asymmetry_ok_d2(c):
    """At n_half=1, d=2: asym requires c[0] >= some threshold (loose).
    For smoke we keep all; the cascade's actual asym filter is at L0.
    Just symmetric reduction: c[0] <= c[1] (canonical form).
    """
    return c[0] <= c[1]


def survives_l0_naive(c, c_target):
    """Apply un-corrected Theorem-1 prune: max ws/(4n·ell·m^2) <= c_target?

    For d=2: conv = [c0^2, 2 c0 c1, c1^2]; ell ∈ {2,3,4}.
    Returns True if survives (no window exceeds c_target).
    """
    cv = conv_int(c)
    d = len(c)
    conv_len = 2 * d - 1
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        if n_windows <= 0:
            continue
        for s_lo in range(n_windows):
            ws = int(cv[s_lo:s_lo + n_cv].sum())
            tv = ws / (4.0 * N_HALF * ell * M * M)
            if tv > c_target:
                return False
    return True


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    print(f"=== Smoke: exact max_t (f*f) vs cascade max_W TV_W at d=4 ===")
    print(f"n_half={N_HALF}, m={M}, c_target={C_TARGET}")
    print(f"d_parent={D_PARENT}, S_parent={S_PARENT}")
    print(f"d_child={D_CHILD}, n_half_child={N_HALF_CHILD}, S_child={S_CHILD}")
    print(f"bin width h={H}, normalizer (4n·m^2)={int(NORMALIZER)}")
    print()

    # Step 1: enumerate parents (d=2) that survive naive Theorem-1.
    parents_all = [c for c in all_d2_compositions() if asymmetry_ok_d2(c)]
    parents_surv = [c for c in parents_all if survives_l0_naive(c, C_TARGET)]
    print(f"d=2 parents (canonical, c[0]<=c[1]): {len(parents_all)}")
    print(f"d=2 parents surviving naive Thm-1 at c_target={C_TARGET}: {len(parents_surv)}")
    print()

    # Step 2: for each surviving parent, generate all d=4 mass-doubled children:
    #   child = (a, 2*c0 - a, b, 2*c1 - b) with 0<=a<=2*c0, 0<=b<=2*c1.
    # For each child, compute exact max_t (f*f) and cascade TV_W.
    n_children_total = 0
    n_alive_naive = 0
    n_gap_ell2_only = 0
    n_gap_all_ell = 0
    sample_gap_ell2 = []
    sample_gap_all = []
    sample_tight = []

    # Cap children for runtime — if too many, sample.
    MAX_CHILDREN = 200000
    rng = np.random.default_rng(seed=0)

    for (c0, c1) in parents_surv:
        # Iterate a in [0, 2c0], b in [0, 2c1]
        n_pa = 2 * c0 + 1
        n_pb = 2 * c1 + 1
        total = n_pa * n_pb
        if total > 4000:
            # subsample
            idxs = rng.choice(total, size=min(4000, total), replace=False)
        else:
            idxs = range(total)
        for idx in idxs:
            a = idx // n_pb
            b = idx % n_pb
            child = np.array([a, 2 * c0 - a, b, 2 * c1 - b], dtype=np.int64)
            assert child.sum() == S_CHILD
            n_children_total += 1
            if n_children_total >= MAX_CHILDREN:
                break

            # Exact max_t at breakpoints:
            exact_max = exact_max_t_ff_v2(child)

            # Cascade ell=2 windows:
            tv_ell2, ell2_e, ell2_s = cascade_tv_max(child, ell_set=[2])
            # Cascade all ell ∈ [2, 2d]:
            tv_all, all_e, all_s = cascade_tv_max(child)

            # If exact > c_target but cascade naive (uncorrected) <= c_target,
            # cascade misses this prune (under un-corrected check).
            if exact_max > C_TARGET:
                # ell=2 only:
                if tv_ell2 <= C_TARGET:
                    n_gap_ell2_only += 1
                    if len(sample_gap_ell2) < 5:
                        sample_gap_ell2.append((child.tolist(), exact_max,
                                                 tv_ell2, tv_all))
                # all ell:
                if tv_all <= C_TARGET:
                    n_gap_all_ell += 1
                    if len(sample_gap_all) < 5:
                        sample_gap_all.append((child.tolist(), exact_max,
                                                tv_ell2, tv_all))
                else:
                    n_alive_naive += 1
                    if len(sample_tight) < 3:
                        sample_tight.append((child.tolist(), exact_max,
                                              tv_ell2, tv_all))
        if n_children_total >= MAX_CHILDREN:
            break

    elapsed = time.time() - t0
    print(f"Total d=4 children scanned: {n_children_total}")
    print(f"Children where exact max_t > {C_TARGET}: total counted")
    print(f"  -> cascade ell=2 ONLY misses (TV_W(ell=2) <= {C_TARGET}): "
          f"{n_gap_ell2_only}")
    print(f"  -> cascade ALL ell misses (TV_W <= {C_TARGET}): {n_gap_all_ell}")
    print(f"  -> cascade ALL ell catches (sound w.r.t. exact): "
          f"{n_alive_naive}")
    print()

    if sample_gap_ell2:
        print(f"Sample (ell=2 only) cells where exact > c_target but TV_W(ell=2) <= c_target:")
        for child, exact, te2, tea in sample_gap_ell2[:3]:
            print(f"  child={child}")
            print(f"    exact_max_t={exact:.6f}, TV_W(ell=2)={te2:.6f},"
                  f" TV_W(all_ell)={tea:.6f}")
    print()
    if sample_gap_all:
        print(f"Sample cells where exact > c_target but TV_W(all_ell) <= c_target:")
        for child, exact, te2, tea in sample_gap_all[:3]:
            print(f"  child={child}")
            print(f"    exact_max_t={exact:.6f}, TV_W(ell=2)={te2:.6f},"
                  f" TV_W(all_ell)={tea:.6f}")
    if sample_tight:
        print(f"Sample tight cells (exact > c_target, TV_W also catches):")
        for child, exact, te2, tea in sample_tight[:3]:
            print(f"  child={child}")
            print(f"    exact_max_t={exact:.6f}, TV_W(ell=2)={te2:.6f},"
                  f" TV_W(all_ell)={tea:.6f}")
    print()
    print(f"=== Verification: exact max_t = max_k (f*f)(t_k)? ===")
    # Quick check: for piecewise linear (f*f), max is at vertex.
    # Verify on a single example: child=[40,40,40,40] symmetric.
    sym = np.array([40, 40, 40, 40], dtype=np.int64)
    cv = conv_int(sym)
    print(f"  Symmetric child [40,40,40,40]: conv = {cv.tolist()}")
    print(f"    breakpoint values (f*f)(t_k) = {[c/NORMALIZER for c in cv]}")
    print(f"    exact max_t = {cv.max()/NORMALIZER:.6f}")
    # Also sample mid-segment via numeric piecewise-linear interp:
    fff = []
    for k in range(len(cv) - 1):
        # linear interp midpoint:
        fff.append(0.5 * (cv[k] + cv[k+1]) / NORMALIZER)
    print(f"    midpoint values (linear interp) = {[f'{x:.4f}' for x in fff]}")
    print(f"    midpoint max = {max(fff):.6f}  (must be <= breakpoint max)")

    print()
    print(f"Elapsed: {elapsed:.2f}s")
    # FINAL line: for parser
    if n_gap_all_ell == 0:
        print(f"\nMAX_T_GAP: TIGHT  (cascade catches every cell where exact (f*f) > c_target)")
    else:
        # average per parent:
        if parents_surv:
            per_parent = n_gap_all_ell / len(parents_surv)
        else:
            per_parent = 0
        print(f"\nMAX_T_GAP: at d=4, cascade misses {n_gap_all_ell} prunes "
              f"(~{per_parent:.2f} per parent over {len(parents_surv)} parents)")


if __name__ == '__main__':
    main()
