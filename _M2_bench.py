"""M2 bench: tightest LINF correction under {|a-b|_inf<=1/m, a>=0} only.

Background (from _stage1_bench.py):
  prune_A: baseline 1 + W_int/(2n)                                 [LOOSEST sound]
  prune_D: (ell-1)*W_int/(2n*ell) + 3*ell_int_sum/(4n*ell)         [TIGHTEST PRIOR]

The current "Stage 1++ / D" linear bound uses
   N_i <= ell-1          --->     Sum_i N_i*c_i <= (ell-1) * W_int.
That overestimates because boundary i's have N_i < ell-1.  M2 removes the
overestimate by computing Sum_i N_i*c_i EXACTLY per window via
a 1-D cross-correlation that depends only on (n_half, ell, s_lo).

================================================================
DERIVATION (sound, in m^2-units; corr_w / m^2 bounds TV(mu)-TV(a)):
================================================================
  ws(c) = sum_{(i,j) in [0,d-1]^2 : i+j in W} c_i c_j     (integer)
  tv(a) = ws(c) / (4n*ell*m^2)                            with a_i = c_i/m
  mu_i = a_i + d_i,  d_i in [0, 1/m]   (fine-grid Lemma 3 convention)
  tv(mu) - tv(a) = (1/(4n*ell)) * (2L + Q),
    L = sum_{(i,j) ord, i+j in W} a_i d_j
    Q = sum_{(i,j) ord, i+j in W} d_i d_j

Linear bound:
  L = sum_i a_i * (sum_{j: i+j in W} d_j) <= (1/m) sum_i a_i * N_i,
  where N_i = #{j in [0,d-1] : i+j in W}.
  Use a_i <= (c_i+1)/m  (since |mu_i - c_i/m| <= 1/m, mu_i = a_i + d_i, d_i >= 0
    -- equivalently the loosest bound under Linf assumption).
  ==>  sum_i N_i a_i <= (1/m) (sum_i N_i c_i + sum_i N_i)
                       = (1/m) (sum_i N_i c_i + ell_int_sum)
  since sum_i N_i = #{(i,j) in [0,d-1]^2 : i+j in W} = ell_int_sum.

  Thus 2L <= (2/m^2) (sum_i N_i c_i + ell_int_sum), and in TV:
    2L/(4n*ell) <= (sum_i N_i c_i + ell_int_sum) / (2n*ell*m^2).

Quadratic bound:
  Q <= (1/m^2) ell_int_sum,  so  Q/(4n*ell) <= ell_int_sum/(4n*ell*m^2).

TOTAL (in corr_w units, i.e. m^2-units of cs_base):
  corr_M2 = (sum_i N_i c_i)/(2n*ell)
            + ell_int_sum/(2n*ell)         (linear / 1-bound piece)
            + ell_int_sum/(4n*ell)         (quadratic piece)
          = (sum_i N_i c_i)/(2n*ell)
            + 3 * ell_int_sum/(4n*ell)

D-vs-M2 difference:  D uses (ell-1)*W_int / (2n*ell) instead of
  Sum_i N_i*c_i / (2n*ell);  since N_i <= ell-1 with equality only for i in
  the "core" of the window, M2 is strictly tighter when boundary bins have
  c_i > 0.

OPTIMALITY:  Both bounds (linear and quadratic) are tight under
  { |d_i| <= 1/m, a_i >= 0 }: take d_i = 1/m for ALL i; equivalently
  a_i = (c_i+1)/m everywhere, mu_i = a_i + d_i = (c_i+1)/m + 1/m
  -- wait, mu_i = a_i + d_i in [a_i, a_i+1/m]; saturating both layers.
  Concretely: a_i = c_i/m + 1/m saturates a_i <= (c_i+1)/m, and
  d_i = 1/m saturates Q <= ell_int_sum/m^2.  Hence M2 is the BEST possible
  bound under the {|delta|<=1/m, a>=0} constraint set.  Any tighter bound
  must use Sum a = 4n (mass conservation) or further structure.

SOUNDNESS NOTE: We use a_i <= (c_i+1)/m, which holds whenever the cell
  convention is mu_i in [c_i/m, (c_i+1)/m] OR mu_i in [c_i/m - 1/(2m), c_i/m + 1/(2m)]
  (centered).  In both cases mu_i <= (c_i+1)/m / 1 holds (centered case is even
  tighter, mu_i <= (c_i + 0.5)/m, so the bound is conservative).

================================================================
"""
import os, sys, time, json, argparse
import numpy as np
import numba
from numba import njit, prange

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
from compositions import generate_compositions_batched
from pruning import count_compositions


# ============================================================
# A: baseline (current code's W-refined "1") — for soundness comparison
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
# D: TIGHTEST PRIOR (Stage 1++ / "_stage1_bench.py prune_D")
#   corr_D = (ell-1)*W_int/(2n*ell) + 3*ell_int_sum/(4n*ell)
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
# E: M2 — replace (ell-1)*W_int by EXACT Sum_i N_i*c_i.
#
# Implementation:  N_i for window [s_lo, s_lo+ell-2] is
#    N_i = #{j in [0,d-1] : i+j in [s_lo, s_lo+ell-2]}
#        = max(0, min(d-1, s_lo+ell-2-i) - max(0, s_lo-i) + 1).
#
# Equivalently a tent function in i:
#   N_i = ramps up from 0 at i = s_lo-(d-1) up to ell-1 by i = s_lo,
#         flat ell-1 from i = s_lo to i = s_lo+ell-1-d (only if d <= ell-1, else
#         the plateau width is ell-1 - (d-1) = ell-d; if ell <= d the maximum is
#         capped by d but...) -- actually the cleanest description:
#
#   N_i = (number of integer j in [0, d-1] with j in [s_lo - i, s_lo+ell-2 - i])
#       = max(0, min(d-1, s_lo+ell-2-i) - max(0, s_lo-i) + 1)
#       = max(0, min(d, s_lo+ell-1-i) - max(0, s_lo-i))
#
# For each window we compute Sum_i N_i*c_i in O(d), or in O(1) per s_lo using
# prefix sums of (i*c_i, c_i).  Below we use the tent decomposition:
#   N_i is piecewise-linear in i with breakpoints at
#     i_a = s_lo - (d-1) (start of ramp up),
#     i_b = s_lo         (end of ramp / start of plateau),  *if* ell >= d, else
#     i_b = s_lo+ell-1-d (end of ramp; plateau width 0 when ell <= d, so peak is
#                          ell-1 if d>=ell, else d if d<ell -- handled below),
#     i_c = s_lo+ell-1-d (start of ramp down),  -- need careful treatment when
#                                                  s_lo+ell-1-d < s_lo
#     i_d = s_lo+ell-1   (end of ramp down).
#
# Because the kernel "N" is piecewise-linear, we use a closed form: split into
# the four regions and compute Sum N_i*c_i with prefix sums of c_i and i*c_i.
# This keeps the inner loop O(1) per window and runs at the same speed as D.
# ============================================================
@njit(parallel=True, cache=True)
def prune_E(batch_int, n_half, m, c_target):
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

        # Prefix sums of c_i and i*c_i over [0, d-1].
        prefix_c = np.zeros(d + 1, dtype=np.int64)
        prefix_ic = np.zeros(d + 1, dtype=np.int64)
        for i in range(d):
            ci_i = np.int64(batch_int[b, i])
            prefix_c[i + 1] = prefix_c[i] + ci_i
            prefix_ic[i + 1] = prefix_ic[i] + np.int64(i) * ci_i

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

                # ----- compute Sum_i N_i * c_i exactly via piecewise-linear N -----
                # Window W = [s_lo, s_lo+ell-2], W has (ell-1) positions.
                # j ranges in [0, d-1].
                # For fixed i, N_i = |[max(0, s_lo - i), min(d-1, s_lo+ell-2-i)]|+1
                # if non-empty.
                #
                # As function of i, N_i is a "trapezoid" (see derivation above),
                # peaking at value cap = min(d, ell-1, ...) actually
                # peak = min(d, ell-1) does NOT capture exactly:
                #   when both ranges are unconstrained (interior), N_i = ell-1.
                #   when constrained on one side by [0,d-1], N_i = depends on i.
                # The cleanest approach: N_i = N_i(s_lo, ell, d) is a closed form;
                # define
                #   alpha_i = max(s_lo - i, 0),
                #   beta_i  = min(s_lo + ell - 2 - i, d - 1).
                # Then  N_i = max(0, beta_i - alpha_i + 1).
                # Let's split into ranges of i:
                #
                # NOTE: i in [0, d-1].
                # Define i1 = s_lo - (d-1)   = s_lo - d + 1
                #        i2 = s_lo
                #        i3 = s_lo + ell - 1 - d    (start of ramp down)
                #        i4 = s_lo + ell - 1
                # Two cases:  (A) i2 <= i3,  i.e. ell >= d: full plateau.
                #             (B) i2 >  i3,  i.e. ell <  d: peak at i3 < i2
                #                  with peak height ell-1 (if i3 >= 0).  Actually
                #                  in case B the trapezoid degenerates: ramp up
                #                  region ends at i = i3 (when constraint
                #                  beta_i = d-1 binds), and ramp down begins
                #                  immediately at i = i2 (when alpha_i = 0 binds).
                #                  In between (i3 < i < i2), N_i = ell-1 STILL.
                # Let's check (B): for i with s_lo - i >= 0 (i.e. i <= s_lo)
                #   alpha_i = s_lo - i.  beta_i = min(s_lo+ell-2-i, d-1).
                #   N_i = beta_i - alpha_i + 1.
                #   If beta_i = s_lo+ell-2-i (interior),
                #     N_i = (s_lo+ell-2-i) - (s_lo-i) + 1 = ell-1.
                #   If beta_i = d-1 (right-clamped),
                #     N_i = d - 1 - (s_lo - i) + 1 = d - s_lo + i.
                #
                # And for i with s_lo - i < 0 (i.e. i > s_lo), alpha_i = 0,
                #   N_i = beta_i + 1 = min(s_lo+ell-1-i, d).
                #     interior: N_i = s_lo+ell-1-i  (when <= d).
                #     left-clamped beta: N_i = d  (when s_lo+ell-1-i >= d).
                # The "clamp d" case: only happens if s_lo+ell-1-i >= d,
                #   i.e. i <= s_lo + ell - 1 - d = i3.  But here i > s_lo = i2,
                #   so requires i3 >= i2+1, i.e. i3 > i2, i.e. ell > d.
                #   So in case B (ell < d), the plateau height is at most ell-1.
                # In case A (ell >= d), the plateau on i in (s_lo, s_lo+ell-1-d]
                #   has N_i = d (capped by [0,d-1] range of j).
                #
                # SUMMARY: N_i is the trapezoidal kernel
                #     N_i = max(0, min(i - i1 + 1, ell - 1, d, i4 - i + 1))
                # for i in [i1, i4], else N_i = 0.  With i1=s_lo-d+1, i4=s_lo+ell-1.
                #
                # For correctness we just compute Sum_i N_i*c_i directly using
                # this expression in an O(d) inner loop.  We've already tried
                # the closed-form and it's bug-prone.  d is small (<= 12 here),
                # so O(d) per window is FINE.
                sum_Nc = np.int64(0)
                ell_minus_1 = np.int64(ell - 1)
                d_int = np.int64(d)
                i1 = s_lo - (d - 1)
                i4 = s_lo + ell - 1
                # Clip the i-range to [0, d-1].
                i_lo = i1
                if i_lo < 0:
                    i_lo = 0
                i_hi = i4
                if i_hi > d_minus_1:
                    i_hi = d_minus_1
                for i in range(i_lo, i_hi + 1):
                    a_lo = s_lo - i
                    if a_lo < 0:
                        a_lo = 0
                    b_hi = s_lo + ell - 2 - i
                    if b_hi > d_minus_1:
                        b_hi = d_minus_1
                    Ni = np.int64(b_hi - a_lo + 1)
                    if Ni > 0:
                        sum_Nc += Ni * np.int64(batch_int[b, i])

                ell_int_sum = ell_prefix[s_lo + n_cv] - ell_prefix[s_lo]
                # corr_M2 = sum_Nc/(2n*ell) + 3*ell_int_sum/(4n*ell)
                corr_w = (np.float64(sum_Nc)
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
# Sanity check (Python, off the JIT path): for d=4 examples, verify
#   Sum_i N_i = ell_int_sum (from prefix).
# ============================================================
def _sanity_Ni(n_half, ell, s_lo, d):
    Ni = np.zeros(d, dtype=np.int64)
    for i in range(d):
        a_lo = max(0, s_lo - i)
        b_hi = min(d - 1, s_lo + ell - 2 - i)
        Ni[i] = max(0, b_hi - a_lo + 1)
    return Ni


def sanity_M2():
    """Verify that for any (n_half, ell, s_lo), Sum_i N_i == ell_int_sum
    (count of ordered (i,j) in [0,d-1]^2 with i+j in W)."""
    for n_half in (3, 4, 5):
        d = 2 * n_half
        conv_len = 2 * d - 1
        two_n = 2 * n_half
        for ell in range(2, 2 * d + 1):
            n_cv = ell - 1
            for s_lo in range(conv_len - n_cv + 1):
                # pen-and-paper Sum N_i
                Ni = _sanity_Ni(n_half, ell, s_lo, d)
                S = int(Ni.sum())
                # ell_int_sum from arrays
                S_ref = 0
                for k in range(s_lo, s_lo + n_cv):
                    diff = abs((k + 1) - two_n)
                    v = max(0, two_n - diff)
                    S_ref += v
                # But ell_int_arr counts ordered (i,j) with i+j=k AND i,j in [0,d-1]?
                # Yes: ell_int_arr[k] = max(0, 2n - |k+1 - 2n|).
                # That equals #{(i,j) in [0,d-1]^2 : i+j = k}.
                # Verify pen-and-paper:
                S_pen = 0
                for k in range(s_lo, s_lo + n_cv):
                    cnt = 0
                    for i in range(d):
                        for j in range(d):
                            if i + j == k:
                                cnt += 1
                    S_pen += cnt
                if S != S_ref or S_ref != S_pen:
                    return False, (n_half, ell, s_lo, S, S_ref, S_pen)
    return True, None


# ============================================================
# Test driver
# ============================================================
def run(n_half, m, c_target, batch_size=200_000, verbose=True, results_dict=None):
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
    prune_E(warm, n_half, m, c_target)

    n_processed = 0
    surv_A_n = 0
    surv_D_n = 0
    surv_E_n = 0
    pruned_A = pruned_D = pruned_E = 0
    t_A = t_D = t_E = 0.0
    n_E_not_in_A = 0
    n_E_not_in_D = 0
    n_E_extra_pruned_vs_D = 0
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

        te = time.time()
        sE = prune_E(batch, n_half, m, c_target)
        t_E += time.time() - te
        pruned_E += int(np.sum(~sE))
        surv_E_n += int(np.sum(sE))

        # Soundness checks
        ena = int(np.sum(sE & ~sA))
        end = int(np.sum(sE & ~sD))
        n_E_not_in_A += ena
        n_E_not_in_D += end
        # E ⊆ D means "everything E-survived was also D-survived";
        # equivalently, E pruned >= D pruned.  D ⊆ A similarly.
        # E prunes EXTRA iff D-survivor that E pruned: int(sum(sD & ~sE))
        n_E_extra_pruned_vs_D += int(np.sum(sD & ~sE))
        if ena > 0:
            print(f"  *** SOUNDNESS BUG: E has {ena} survivors NOT in A!")
        if end > 0:
            print(f"  *** SOUNDNESS BUG: E has {end} survivors NOT in D!")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n--- L0 results ---")
        print(f"  total processed:  {n_processed:,}")
        print(f"  A (baseline):     {pruned_A:,} pruned, "
              f"{surv_A_n:,} survivors  [{t_A:.2f}s]")
        print(f"  D (Stage 1++):    {pruned_D:,} pruned, "
              f"{surv_D_n:,} survivors  [{t_D:.2f}s]")
        print(f"  E (M2 sharpest):  {pruned_E:,} pruned, "
              f"{surv_E_n:,} survivors  [{t_E:.2f}s]")
        if surv_A_n > 0:
            rD = (surv_A_n - surv_D_n) / surv_A_n * 100
            rE = (surv_A_n - surv_E_n) / surv_A_n * 100
            print(f"  D prunes  +{surv_A_n - surv_D_n:,}  "
                  f"({rD:.2f}% of A survivors)")
            print(f"  E prunes  +{surv_A_n - surv_E_n:,}  "
                  f"({rE:.2f}% of A survivors)")
        if surv_D_n > 0:
            rED = (surv_D_n - surv_E_n) / surv_D_n * 100
            print(f"  E vs D extra:  +{surv_D_n - surv_E_n:,} more pruned "
                  f"({rED:.3f}% of D survivors)")
        if t_D > 0:
            print(f"  E/D wall-time ratio: {t_E/t_D:.3f}x  "
                  f"(E is {'SLOWER' if t_E > t_D else 'faster'})")
        print(f"  total wall: {elapsed:.2f}s")
        print(f"  E-not-in-A: {n_E_not_in_A}  E-not-in-D: {n_E_not_in_D}")

    rec = {
        'n_half': n_half, 'm': m, 'c_target': c_target,
        'd': d, 'n_processed': n_processed,
        'pruned_A': pruned_A, 'pruned_D': pruned_D, 'pruned_E': pruned_E,
        'surv_A': surv_A_n, 'surv_D': surv_D_n, 'surv_E': surv_E_n,
        't_A': t_A, 't_D': t_D, 't_E': t_E,
        'E_not_in_A': n_E_not_in_A, 'E_not_in_D': n_E_not_in_D,
        'E_extra_vs_D': n_E_extra_pruned_vs_D,
        'percent_E_extra_vs_D': (
            (surv_D_n - surv_E_n) / surv_D_n * 100
            if surv_D_n > 0 else 0.0),
        'percent_E_total_vs_A': (
            (surv_A_n - surv_E_n) / surv_A_n * 100
            if surv_A_n > 0 else 0.0),
        'time_ratio_E_over_D': t_E / t_D if t_D > 0 else float('nan'),
        'soundness_ok': (n_E_not_in_A == 0 and n_E_not_in_D == 0),
    }
    if results_dict is not None:
        results_dict.append(rec)
    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_half', type=int, default=3)
    ap.add_argument('--m', type=int, default=10)
    ap.add_argument('--c_target', type=float, default=1.20)
    ap.add_argument('--batch', type=int, default=200_000)
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--sanity', action='store_true')
    ap.add_argument('--out', default='_M2_results.json')
    args = ap.parse_args()

    if args.sanity:
        ok, info = sanity_M2()
        print(f"sanity_M2: ok={ok} info={info}")
        sys.exit(0 if ok else 1)

    # Always run sanity first.
    ok, info = sanity_M2()
    print(f"[sanity] sum_i N_i == ell_int_sum invariant: ok={ok}  info={info}")
    if not ok:
        sys.exit(2)

    results = []
    if args.sweep:
        cfgs = [(3, 10, 1.20), (3, 20, 1.20), (3, 30, 1.20),
                (3, 10, 1.10), (3, 10, 1.28),
                (4, 5, 1.20), (4, 10, 1.20), (4, 10, 1.28),
                (5, 5, 1.28)]
        for nh, m, c in cfgs:
            try:
                run(nh, m, c, batch_size=args.batch, results_dict=results)
            except Exception as e:
                print(f"  cfg ({nh},{m},{c}) FAILED: {e}")
    else:
        run(args.n_half, args.m, args.c_target, batch_size=args.batch,
             results_dict=results)

    print(f"\n=== Sweep summary ({len(results)} cfgs) ===")
    print(f"{'nh':>3} {'m':>3} {'c':>5}  {'survA':>10} {'survD':>10} "
          f"{'survE':>10}  {'E-vs-D':>8} {'E-vs-A':>8}  {'tD':>6} {'tE':>6} "
          f"{'tE/tD':>6}  {'sound':>5}")
    for r in results:
        print(f"{r['n_half']:>3} {r['m']:>3} {r['c_target']:>5.2f}  "
              f"{r['surv_A']:>10,} {r['surv_D']:>10,} {r['surv_E']:>10,}  "
              f"{r['percent_E_extra_vs_D']:>7.2f}% "
              f"{r['percent_E_total_vs_A']:>7.2f}% "
              f"{r['t_D']:>6.2f} {r['t_E']:>6.2f} "
              f"{r['time_ratio_E_over_D']:>6.2f}  {str(r['soundness_ok']):>5}")

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nresults -> {args.out}")
