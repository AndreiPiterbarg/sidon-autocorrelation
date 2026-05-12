"""LP Relaxation via McCormick Envelopes for parent-level pruning.

Tests whether an LP relaxation of the quadratic feasibility problem can
determine at the parent level that NO child survives, without enumerating
the full Cartesian product of cursor ranges.

The autoconvolution window sum is quadratic in cursor variables v_p.
McCormick envelopes linearize bilinear terms v_p * v_q, and secant/chord
envelopes linearize squared terms v_p^2, yielding an LP.  If the LP is
infeasible, no child can survive -- a sound (conservative) parent-level prune.

Three key tightening improvements over the naive formulation:
  1. Variable W_int threshold: instead of using W_int_max (loosest threshold),
     express W_int as a linear function of v_p and absorb it into the LHS.
  2. Dense tangent cuts for u_p = v_p^2: add O(range) tangent lines instead
     of just 2, reducing the secant-chord gap from O(range^2) to O(1).
  3. Piecewise McCormick refinement: split variable ranges and add McCormick
     constraints for sub-rectangles.
"""

import sys, os, time, math
import numpy as np
from scipy.optimize import linprog

# -- Project imports --
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_this_dir)
_cs_dir = os.path.join(_project_dir, 'cloninger-steinerberger')
sys.path.insert(0, _cs_dir)

from cpu.run_cascade import (
    _compute_bin_ranges, _tighten_ranges, _fused_generate_and_prune_gray
)


# =====================================================================
# Core LP builder
# =====================================================================

def _add(d, key, val):
    """Accumulate val into dict d at key."""
    if key in d:
        d[key] += val
    else:
        d[key] = val


def build_lp(parent_int, lo_arr, hi_arr, m, c_target, n_half_child,
             use_flat_threshold=False, n_tangent_cuts=0,
             use_variable_threshold=False):
    """Build LP relaxation of the child feasibility problem.

    Variables:
      v_p   (p = 0..d_parent-1)  : cursor values, box-bounded [lo_p, hi_p]
      u_p   (p = 0..d_parent-1)  : relaxation of v_p^2
      w_pq  (p < q)              : relaxation of v_p * v_q

    Tightening options (all sound -- they only cut infeasible LP regions):
      n_tangent_cuts: add tangent lines u >= 2a*v - a^2 at n equally-spaced
          interior integer points.  Reduces the secant-chord gap from
          O(range^2/4) to O(range^2/(n+1)^2).
      use_variable_threshold: instead of W_int_max (worst-case), express
          W_int as a linear function of v_p, yielding:
            ws(v) - 2*ell*W_int(v) <= (c*m^2 + 1 + eps) * 4n*ell

    Returns (c_obj, A_ub, b_ub, bounds) for scipy.optimize.linprog.
    """
    d_parent = len(parent_int)
    d_child = 2 * d_parent
    conv_len = 2 * d_child - 1

    # --- Variable layout ---
    n_pairs = d_parent * (d_parent - 1) // 2
    n_vars = 2 * d_parent + n_pairs

    pair_idx = {}
    idx = 2 * d_parent
    for p in range(d_parent):
        for q in range(p + 1, d_parent):
            pair_idx[(p, q)] = idx
            idx += 1

    B = np.array(parent_int, dtype=np.int64)
    lo = np.array(lo_arr, dtype=np.int64)
    hi = np.array(hi_arr, dtype=np.int64)

    # --- Express each conv[k] as linear in LP variables + constant ---
    conv_coeffs = [None] * conv_len
    conv_const = np.zeros(conv_len, dtype=np.float64)

    for k in range(conv_len):
        coeffs = {}
        const = 0.0
        for i in range(d_child):
            j = k - i
            if j < 0 or j >= d_child or j < i:
                continue
            weight = 1 if i == j else 2
            p_i, p_j = i // 2, j // 2
            i_even, j_even = (i % 2 == 0), (j % 2 == 0)

            if i_even and j_even:
                # v_p * v_q
                if p_i == p_j:
                    _add(coeffs, d_parent + p_i, weight)
                else:
                    pp, qq = (min(p_i, p_j), max(p_i, p_j))
                    _add(coeffs, pair_idx[(pp, qq)], weight)

            elif i_even and not j_even:
                # v_p * (2*B_q - v_q) = 2*B_q*v_p - v_p*v_q
                if p_i == p_j:
                    _add(coeffs, p_i, weight * 2 * int(B[p_i]))
                    _add(coeffs, d_parent + p_i, -weight)
                else:
                    _add(coeffs, p_i, weight * 2 * int(B[p_j]))
                    pp, qq = (min(p_i, p_j), max(p_i, p_j))
                    _add(coeffs, pair_idx[(pp, qq)], -weight)

            elif not i_even and j_even:
                # (2*B_p - v_p) * v_q = 2*B_p*v_q - v_p*v_q
                if p_i == p_j:
                    _add(coeffs, p_i, weight * 2 * int(B[p_i]))
                    _add(coeffs, d_parent + p_i, -weight)
                else:
                    _add(coeffs, p_j, weight * 2 * int(B[p_i]))
                    pp, qq = (min(p_i, p_j), max(p_i, p_j))
                    _add(coeffs, pair_idx[(pp, qq)], -weight)

            else:
                # (2*B_p - v_p)(2*B_q - v_q) = 4BpBq - 2Bp*vq - 2Bq*vp + vp*vq
                if p_i == p_j:
                    const += weight * 4 * int(B[p_i]) * int(B[p_i])
                    _add(coeffs, p_i, -weight * 4 * int(B[p_i]))
                    _add(coeffs, d_parent + p_i, weight)
                else:
                    const += weight * 4 * int(B[p_i]) * int(B[p_j])
                    _add(coeffs, p_j, -weight * 2 * int(B[p_i]))
                    _add(coeffs, p_i, -weight * 2 * int(B[p_j]))
                    pp, qq = (min(p_i, p_j), max(p_i, p_j))
                    _add(coeffs, pair_idx[(pp, qq)], weight)

        conv_coeffs[k] = coeffs
        conv_const[k] = const

    # --- Threshold constants ---
    m_d = float(m)
    four_n = 4.0 * float(n_half_child)
    n_half_d = float(n_half_child)
    eps_margin = 1e-9 * m_d * m_d
    cs_base_m2 = c_target * m_d * m_d
    flat_corr = 2.0 * m_d + 1.0
    S_child = int(4 * n_half_child * m)

    # Max child prefix for W_int_max
    max_child_prefix = np.zeros(d_child + 1, dtype=np.int64)
    for q in range(d_parent):
        max_child_prefix[2 * q + 1] = max_child_prefix[2 * q] + int(hi[q])
        max_child_prefix[2 * q + 2] = (max_child_prefix[2 * q + 1]
                                        + int(2 * B[q] - lo[q]))

    # --- W_int as linear function of v (for variable-threshold mode) ---
    def _w_int_linear(lo_bin, hi_bin):
        """Return (coeffs_dict, constant) for W_int as linear fn of v_p.
        W_int = sum child[i] for i in [lo_bin, hi_bin].
        child[2p] = v_p, child[2p+1] = 2*B_p - v_p.
        """
        wc = {}
        wconst = 0.0
        for i in range(lo_bin, hi_bin + 1):
            p = i // 2
            if i % 2 == 0:
                _add(wc, p, 1.0)
            else:
                wconst += 2.0 * float(B[p])
                _add(wc, p, -1.0)
        return wc, wconst

    # --- Build window constraints ---
    A_rows = []
    b_rows = []

    for ell in range(2, 2 * d_child + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        if n_windows <= 0:
            continue
        scale_ell = float(ell) * four_n

        for s_lo in range(n_windows):
            lo_bin = max(0, s_lo - (d_child - 1))
            hi_bin = min(d_child - 1, s_lo + ell - 2)

            # Sum conv linear expressions over window
            win_coeffs = {}
            win_const = 0.0
            for k in range(s_lo, s_lo + n_cv):
                for var_idx, coeff in conv_coeffs[k].items():
                    _add(win_coeffs, var_idx, coeff)
                win_const += conv_const[k]

            if use_variable_threshold and not use_flat_threshold:
                # IMPROVEMENT 1: Absorb W_int into LHS.
                # Survival: ws <= (c*m^2 + 1 + W_int/(2n) + eps) * 4n*ell
                # Rearrange: ws - 2*ell*W_int <= (c*m^2 + 1 + eps) * 4n*ell
                #
                # Sound: continuous RHS >= floor(RHS), so LP infeasible
                # implies no child satisfies the floor version either.
                wc, wconst_w = _w_int_linear(lo_bin, hi_bin)
                for var_idx, wval in wc.items():
                    _add(win_coeffs, var_idx, -2.0 * float(ell) * wval)
                win_const -= 2.0 * float(ell) * wconst_w
                threshold = (cs_base_m2 + 1.0 + eps_margin) * scale_ell
            else:
                # Original: use W_int_max for threshold (conservative)
                W_int_max = int(max_child_prefix[hi_bin + 1]
                                - max_child_prefix[lo_bin])
                if W_int_max > S_child:
                    W_int_max = S_child
                if use_flat_threshold:
                    dyn_x = (cs_base_m2 + flat_corr + eps_margin) * scale_ell
                else:
                    corr_w = 1.0 + float(W_int_max) / (2.0 * n_half_d)
                    dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_ell
                threshold = math.floor(dyn_x)

            # Pre-screen
            ub = _estimate_window_upper_bound(
                win_coeffs, win_const, lo, hi, d_parent, n_pairs, pair_idx)
            if ub < 0.8 * threshold:
                continue

            row = np.zeros(n_vars, dtype=np.float64)
            for var_idx, coeff in win_coeffs.items():
                row[var_idx] = coeff
            A_rows.append(row)
            b_rows.append(threshold - win_const)

    # --- McCormick envelope constraints ---
    def _add_mccormick_4(p, q, w_idx, lp_v, hp_v, lq_v, hq_v):
        """Add 4 McCormick constraints for w_pq on [lp,hp] x [lq,hq]."""
        # w >= lp*vq + vp*lq - lp*lq
        row = np.zeros(n_vars, dtype=np.float64)
        row[w_idx] = -1; row[p] = lq_v; row[q] = lp_v
        A_rows.append(row); b_rows.append(float(lp_v * lq_v))
        # w >= hp*vq + vp*hq - hp*hq
        row = np.zeros(n_vars, dtype=np.float64)
        row[w_idx] = -1; row[p] = hq_v; row[q] = hp_v
        A_rows.append(row); b_rows.append(float(hp_v * hq_v))
        # w <= hp*vq + vp*lq - hp*lq
        row = np.zeros(n_vars, dtype=np.float64)
        row[w_idx] = 1; row[p] = -lq_v; row[q] = -hp_v
        A_rows.append(row); b_rows.append(float(-hp_v * lq_v))
        # w <= lp*vq + vp*hq - lp*hq
        row = np.zeros(n_vars, dtype=np.float64)
        row[w_idx] = 1; row[p] = -hq_v; row[q] = -lp_v
        A_rows.append(row); b_rows.append(float(-lp_v * hq_v))

    for (p, q), w_idx in pair_idx.items():
        _add_mccormick_4(p, q, w_idx,
                         int(lo[p]), int(hi[p]), int(lo[q]), int(hi[q]))
        # NOTE: Piecewise McCormick splits (sub-rectangle constraints) are
        # UNSOUND for a single w_pq variable because they assume v_p is
        # restricted to a sub-interval. They require indicator variables
        # (MILP) or separate w variables per sub-rectangle. Removed.

    # --- Secant/chord envelope for u_p = v_p^2 ---
    for p in range(d_parent):
        lp_v, hp_v = int(lo[p]), int(hi[p])
        u_idx = d_parent + p

        # Chord: u <= (lo+hi)*v - lo*hi
        row = np.zeros(n_vars, dtype=np.float64)
        row[u_idx] = 1
        row[p] = -(lp_v + hp_v)
        A_rows.append(row)
        b_rows.append(float(-lp_v * hp_v))

        # Tangent at endpoints: u >= 2*a*v - a^2
        for a in [lp_v, hp_v]:
            row = np.zeros(n_vars, dtype=np.float64)
            row[u_idx] = -1
            row[p] = 2 * a
            A_rows.append(row)
            b_rows.append(float(a * a))

        # IMPROVEMENT 2: Dense interior tangent cuts
        if n_tangent_cuts > 0 and hp_v - lp_v > 2:
            n_cuts = min(n_tangent_cuts, hp_v - lp_v - 1)
            for ci in range(1, n_cuts + 1):
                a = lp_v + ci * (hp_v - lp_v) // (n_cuts + 1)
                if a == lp_v or a == hp_v:
                    continue
                row = np.zeros(n_vars, dtype=np.float64)
                row[u_idx] = -1
                row[p] = 2 * a
                A_rows.append(row)
                b_rows.append(float(a * a))

    # --- Bounds ---
    bounds = []
    for p in range(d_parent):
        bounds.append((float(lo[p]), float(hi[p])))
    for p in range(d_parent):
        lp_v, hp_v = float(lo[p]), float(hi[p])
        bounds.append((lp_v * lp_v, hp_v * hp_v))
    for (p, q) in pair_idx:
        lp_v, hp_v = float(lo[p]), float(hi[p])
        lq_v, hq_v = float(lo[q]), float(hi[q])
        products = [lp_v*lq_v, lp_v*hq_v, hp_v*lq_v, hp_v*hq_v]
        bounds.append((min(products), max(products)))

    if len(A_rows) == 0:
        return np.zeros(n_vars), np.zeros((0, n_vars)), np.zeros(0), bounds

    A_ub = np.array(A_rows, dtype=np.float64)
    b_ub = np.array(b_rows, dtype=np.float64)
    c_obj = np.zeros(n_vars, dtype=np.float64)

    return c_obj, A_ub, b_ub, bounds


def _estimate_window_upper_bound(win_coeffs, win_const, lo, hi,
                                 d_parent, n_pairs, pair_idx):
    """Quick upper bound on window sum using variable bounds."""
    ub = win_const
    for var_idx, coeff in win_coeffs.items():
        if var_idx < d_parent:
            p = var_idx
            if coeff >= 0:
                ub += coeff * float(hi[p])
            else:
                ub += coeff * float(lo[p])
        elif var_idx < 2 * d_parent:
            p = var_idx - d_parent
            lp_v, hp_v = float(lo[p]), float(hi[p])
            if coeff >= 0:
                ub += coeff * hp_v * hp_v
            else:
                ub += coeff * lp_v * lp_v
        else:
            for (p, q), idx in pair_idx.items():
                if idx == var_idx:
                    lp_v, hp_v = float(lo[p]), float(hi[p])
                    lq_v, hq_v = float(lo[q]), float(hi[q])
                    products = [lp_v*lq_v, lp_v*hq_v, hp_v*lq_v, hp_v*hq_v]
                    if coeff >= 0:
                        ub += coeff * max(products)
                    else:
                        ub += coeff * min(products)
                    break
    return ub


def lp_parent_infeasible(parent_int, lo_arr, hi_arr, m, c_target, n_half_child,
                         use_flat_threshold=False, **lp_kwargs):
    """Solve LP relaxation. Returns True if infeasible (no child can survive)."""
    c_obj, A_ub, b_ub, bounds = build_lp(
        parent_int, lo_arr, hi_arr, m, c_target, n_half_child,
        use_flat_threshold, **lp_kwargs
    )
    if A_ub.shape[0] == 0:
        return False

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True, 'time_limit': 5.0})
    return result.status == 2  # 2 = infeasible


# =====================================================================
# Exhaustive brute-force for small parents
# =====================================================================

def exhaustive_check(parent_int, lo_arr, hi_arr, m, c_target, n_half_child,
                     use_flat_threshold=False):
    """Brute-force enumerate all children and check each against thresholds.

    Returns (survivors, total_tested).
    Only feasible for very small d_parent (2-3).
    """
    d_parent = len(parent_int)
    d_child = 2 * d_parent
    conv_len = 2 * d_child - 1
    m_d = float(m)
    four_n = 4.0 * float(n_half_child)
    n_half_d = float(n_half_child)
    eps_margin = 1e-9 * m_d * m_d
    cs_base_m2 = c_target * m_d * m_d
    flat_corr = 2.0 * m_d + 1.0
    S_child = int(4 * n_half_child * m)

    def compute_threshold(ell, W_int):
        scale_ell = float(ell) * four_n
        if use_flat_threshold:
            dyn_x = (cs_base_m2 + flat_corr + eps_margin) * scale_ell
        else:
            corr_w = 1.0 + float(W_int) / (2.0 * n_half_d)
            dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_ell
        return math.floor(dyn_x)

    from itertools import product as cart_product
    ranges = [range(int(lo_arr[p]), int(hi_arr[p]) + 1) for p in range(d_parent)]
    survivors = []
    total = 0

    for cursors in cart_product(*ranges):
        total += 1
        child = np.zeros(d_child, dtype=np.int64)
        for p, v in enumerate(cursors):
            child[2 * p] = v
            child[2 * p + 1] = 2 * int(parent_int[p]) - v

        conv = np.zeros(conv_len, dtype=np.int64)
        for i in range(d_child):
            if child[i] == 0:
                continue
            conv[2 * i] += child[i] * child[i]
            for j in range(i + 1, d_child):
                if child[j] == 0:
                    continue
                conv[i + j] += 2 * child[i] * child[j]

        prefix_c = np.zeros(d_child + 1, dtype=np.int64)
        for i in range(d_child):
            prefix_c[i + 1] = prefix_c[i] + child[i]

        pruned = False
        for ell in range(2, 2 * d_child + 1):
            if pruned:
                break
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            if n_windows <= 0:
                continue
            ws = sum(conv[:n_cv])
            for s_lo in range(n_windows):
                if s_lo > 0:
                    ws += conv[s_lo + n_cv - 1] - conv[s_lo - 1]
                lo_bin = max(0, s_lo - (d_child - 1))
                hi_bin = min(d_child - 1, s_lo + ell - 2)
                W_int = int(prefix_c[hi_bin + 1] - prefix_c[lo_bin])
                if W_int > S_child:
                    W_int = S_child
                thresh = compute_threshold(ell, W_int)
                if ws > thresh:
                    pruned = True
                    break

        if not pruned:
            survivors.append(cursors)

    return survivors, total


# =====================================================================
# Main test / benchmark
# =====================================================================

def run_sanity_check():
    """Sanity check on synthetic small parents (d_parent=2, d_child=4)."""
    print("=" * 70)
    print("SANITY CHECK: d_parent=2, d_child=4")
    print("=" * 70)

    m = 20
    c_target = 1.35
    n_half_child = 2

    test_parents = [
        np.array([20, 60], dtype=np.int32),
        np.array([40, 40], dtype=np.int32),
        np.array([35, 45], dtype=np.int32),
        np.array([10, 70], dtype=np.int32),
        np.array([30, 50], dtype=np.int32),
    ]

    for idx, parent in enumerate(test_parents):
        print(f"\n--- Parent {idx}: {parent} (sum={sum(parent)}) ---")
        result = _compute_bin_ranges(parent, m, c_target, 4, n_half_child)
        if result is None:
            print("  _compute_bin_ranges returned None")
            continue
        lo_arr, hi_arr, total = result
        lo_t, hi_t = lo_arr.copy(), hi_arr.copy()
        total_t = _tighten_ranges(parent, lo_t, hi_t, m, c_target, n_half_child)
        print(f"  Tightened: lo={lo_t}, hi={hi_t}, total={total_t}")
        if total_t == 0:
            print("  All pruned by arc consistency")
            continue

        survivors, n_tested = exhaustive_check(parent, lo_t, hi_t, m, c_target, n_half_child)
        print(f"  Exhaustive: {n_tested} tested, {len(survivors)} survived")

        # Test all LP configurations
        for label, kwargs in [
            ("baseline", {}),
            ("var_thresh", {"use_variable_threshold": True}),
            ("tangent50", {"n_tangent_cuts": 50}),
            ("vt+t50", {"use_variable_threshold": True, "n_tangent_cuts": 50}),
        ]:
            t0 = time.perf_counter()
            infeasible = lp_parent_infeasible(
                parent, lo_t, hi_t, m, c_target, n_half_child, **kwargs)
            dt = time.perf_counter() - t0
            tag = "INFEASIBLE" if infeasible else "FEASIBLE"
            correct = ""
            if len(survivors) > 0 and infeasible:
                correct = " *** SOUNDNESS VIOLATION ***"
            elif len(survivors) == 0 and infeasible:
                correct = " (correct)"
            elif len(survivors) == 0 and not infeasible:
                correct = " (LP loose)"
            print(f"  LP [{label:12s}]: {tag} ({dt*1000:.1f}ms){correct}")


def run_tightening_comparison():
    """Compare LP tightening improvements on real L0 data."""
    print("\n" + "=" * 70)
    print("TIGHTENING COMPARISON: L0->L1 (d_parent=4, d_child=8)")
    print("=" * 70)

    m = 20
    c_target = 1.35
    data_path = os.path.join(_project_dir, 'data', 'checkpoint_L0_survivors.npy')
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    parents = np.load(data_path)
    d_parent = parents.shape[1]
    d_child = 2 * d_parent
    n_half_child = d_child // 2

    N = min(2000, len(parents))

    configs = [
        ("baseline", {}),
        ("var_thresh", {"use_variable_threshold": True}),
        ("tang_50", {"n_tangent_cuts": 50}),
        ("vt+t50", {"use_variable_threshold": True, "n_tangent_cuts": 50}),
    ]

    results = {label: {"killed": 0, "feasible": 0, "times": []}
               for label, _ in configs}

    n_arc_survived = 0

    t_start = time.perf_counter()
    for i in range(N):
        parent = parents[i]
        result = _compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
        if result is None:
            continue
        lo_arr, hi_arr, _ = result
        lo_t, hi_t = lo_arr.copy(), hi_arr.copy()
        total_t = _tighten_ranges(parent, lo_t, hi_t, m, c_target, n_half_child)
        if total_t == 0:
            continue
        n_arc_survived += 1

        for label, kwargs in configs:
            t0 = time.perf_counter()
            infeasible = lp_parent_infeasible(
                parent, lo_t, hi_t, m, c_target, n_half_child, **kwargs)
            dt = time.perf_counter() - t0
            results[label]["times"].append(dt)
            if infeasible:
                results[label]["killed"] += 1
            else:
                results[label]["feasible"] += 1

        if (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  [{i+1}/{N}] arc_survived={n_arc_survived}, "
                  f"elapsed={elapsed:.0f}s")

    print(f"\nArc-consistency survivors: {n_arc_survived} / {N}")
    print(f"\n{'Config':<16s} {'Killed':>8s} {'Kill%':>8s} {'Median ms':>10s} {'P95 ms':>10s}")
    print("-" * 56)
    for label, _ in configs:
        r = results[label]
        killed = r["killed"]
        pct = 100 * killed / max(1, n_arc_survived)
        if r["times"]:
            tarr = np.array(r["times"])
            med = np.median(tarr) * 1000
            p95 = np.percentile(tarr, 95) * 1000
        else:
            med = p95 = 0
        print(f"{label:<16s} {killed:>8d} {pct:>7.1f}% {med:>10.1f} {p95:>10.1f}")

    return results


def run_l1_tightening():
    """Test LP tightening on L1 shard data (L1->L2, d_parent=8)."""
    print("\n" + "=" * 70)
    print("L1->L2 TIGHTENING (d_parent=8, d_child=16)")
    print("=" * 70)

    m = 20
    c_target = 1.35
    shard_path = os.path.join(_project_dir, 'data', '_shards_L1', 'shard_0000.npy')
    if not os.path.exists(shard_path):
        print(f"L1 shard not found: {shard_path}")
        return

    l1 = np.load(shard_path)
    d_parent = l1.shape[1]
    d_child = 2 * d_parent
    n_half_child = d_child // 2

    N = min(200, len(l1))

    configs = [
        ("baseline", {}),
        ("vt+t50", {"use_variable_threshold": True, "n_tangent_cuts": 50}),
    ]

    results = {label: {"killed": 0, "feasible": 0, "times": []}
               for label, _ in configs}
    n_arc_survived = 0

    for i in range(N):
        parent = l1[i]
        result = _compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
        if result is None:
            continue
        lo_arr, hi_arr, _ = result
        lo_t, hi_t = lo_arr.copy(), hi_arr.copy()
        total_t = _tighten_ranges(parent, lo_t, hi_t, m, c_target, n_half_child)
        if total_t == 0:
            continue
        n_arc_survived += 1

        for label, kwargs in configs:
            t0 = time.perf_counter()
            infeasible = lp_parent_infeasible(
                parent, lo_t, hi_t, m, c_target, n_half_child, **kwargs)
            dt = time.perf_counter() - t0
            results[label]["times"].append(dt)
            if infeasible:
                results[label]["killed"] += 1
            else:
                results[label]["feasible"] += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N}] arc_survived={n_arc_survived}")

    print(f"\nArc-consistency survivors: {n_arc_survived} / {N}")
    print(f"\n{'Config':<16s} {'Killed':>8s} {'Kill%':>8s} {'Median ms':>10s}")
    print("-" * 46)
    for label, _ in configs:
        r = results[label]
        killed = r["killed"]
        pct = 100 * killed / max(1, n_arc_survived)
        med = np.median(r["times"]) * 1000 if r["times"] else 0
        print(f"{label:<16s} {killed:>8d} {pct:>7.1f}% {med:>10.1f}")


def run_corner_analysis():
    """Diagnose LP gap by evaluating actual quadratic at LP solution."""
    print("\n" + "=" * 70)
    print("CORNER AND GAP ANALYSIS")
    print("=" * 70)

    m = 20
    c_target = 1.35
    data_path = os.path.join(_project_dir, 'data', 'checkpoint_L0_survivors.npy')
    if not os.path.exists(data_path):
        return

    parents = np.load(data_path)
    d_parent = parents.shape[1]
    d_child = 2 * d_parent
    n_half_child = d_child // 2
    conv_len = 2 * d_child - 1

    m_d = float(m)
    four_n = 4.0 * float(n_half_child)
    eps_margin = 1e-9 * m_d * m_d
    cs_base_m2 = c_target * m_d * m_d
    S_child = int(4 * n_half_child * m)

    for pi in range(3):
        parent = parents[pi]
        result = _compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
        if result is None:
            continue
        lo_arr, hi_arr, _ = result
        lo_t, hi_t = lo_arr.copy(), hi_arr.copy()
        _tighten_ranges(parent, lo_t, hi_t, m, c_target, n_half_child)

        print(f"\nParent {pi}: {parent}")
        print(f"  Ranges: lo={lo_t}, hi={hi_t}")
        print(f"  Widths: {[int(hi_t[j])-int(lo_t[j])+1 for j in range(d_parent)]}")

        # Check all 2^d_parent corners
        n_pruned = 0
        for mask in range(1 << d_parent):
            child = np.zeros(d_child, dtype=np.int64)
            for q in range(d_parent):
                v = lo_t[q] if (mask >> q) & 1 == 0 else hi_t[q]
                child[2*q] = v
                child[2*q+1] = 2*parent[q] - v

            conv = np.zeros(conv_len, dtype=np.int64)
            for i in range(d_child):
                conv[2*i] += child[i]**2
                for j in range(i+1, d_child):
                    conv[i+j] += 2*child[i]*child[j]

            prefix_c = np.zeros(d_child+1, dtype=np.int64)
            for i in range(d_child):
                prefix_c[i+1] = prefix_c[i] + child[i]

            pruned = False
            for ell in range(2, 2*d_child+1):
                if pruned:
                    break
                n_cv = ell - 1
                n_win = conv_len - n_cv + 1
                if n_win <= 0:
                    continue
                ws = int(sum(conv[:n_cv]))
                for s_lo in range(n_win):
                    if s_lo > 0:
                        ws += int(conv[s_lo+n_cv-1]) - int(conv[s_lo-1])
                    lo_bin = max(0, s_lo-(d_child-1))
                    hi_bin = min(d_child-1, s_lo+ell-2)
                    W_int = int(prefix_c[hi_bin+1] - prefix_c[lo_bin])
                    if W_int > S_child:
                        W_int = S_child
                    corr_w = 1.0 + float(W_int)/(2.0*float(n_half_child))
                    thresh = int((cs_base_m2 + corr_w + eps_margin) * float(ell)*four_n)
                    if ws > thresh:
                        pruned = True
                        break
            if pruned:
                n_pruned += 1

        print(f"  Corners pruned: {n_pruned}/{1 << d_parent}")
        all_pruned = (n_pruned == (1 << d_parent))

        # LP results
        for label, kwargs in [
            ("baseline", {}),
            ("vt+t50", {"use_variable_threshold": True, "n_tangent_cuts": 50}),
        ]:
            infeasible = lp_parent_infeasible(
                parent, lo_t, hi_t, m, c_target, n_half_child, **kwargs)
            print(f"  LP [{label}]: {'INFEASIBLE' if infeasible else 'FEASIBLE'}"
                  f" (corners_all_pruned={all_pruned})")

        # Examine LP solution gap
        c_obj, A_ub, b_ub, bounds = build_lp(
            parent, lo_t, hi_t, m, c_target, n_half_child,
            use_variable_threshold=True, n_tangent_cuts=50)
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            v_sol = res.x[:d_parent]
            u_sol = res.x[d_parent:2*d_parent]
            print(f"  LP solution v = {np.round(v_sol, 1)}")
            print(f"  u vs v^2 gap = {np.round(u_sol - v_sol**2, 1)}")


def run_multilevel_cascade():
    """Test LP at each cascade level using actual checkpoint/shard data.

    Uses real cascade survivors at each level:
      - L0: data/checkpoint_L0_survivors.npy (d=4 parents for L0->L1)
      - L1: data/_shards_L1/shard_0000.npy (d=8 parents for L1->L2)
    For higher levels without shard data, run kernel on a small sample of
    L1 parents to get L2 survivors, then continue.
    """
    print("\n" + "=" * 70)
    print("MULTI-LEVEL CASCADE LP TEST")
    print("  d0=2, m=20, c_target=1.35")
    print("  Using actual cascade survivors at each level")
    print("=" * 70)

    m = 20
    c_target = 1.35

    # Gather data sources at each level
    data_sources = []

    # L0 survivors -> parents for L0->L1
    l0_path = os.path.join(_project_dir, 'data', 'checkpoint_L0_survivors.npy')
    if os.path.exists(l0_path):
        data_sources.append(("L0->L1", np.load(l0_path)))

    # L1 shard -> parents for L1->L2
    l1_path = os.path.join(_project_dir, 'data', '_shards_L1', 'shard_0000.npy')
    if os.path.exists(l1_path):
        data_sources.append(("L1->L2", np.load(l1_path)))

    if not data_sources:
        print("No data files found!")
        return

    # For L2->L3: generate L2 survivors by running kernel on L1 parents
    # with small enough child counts
    if os.path.exists(l1_path):
        print("\nGenerating L2 survivors from L1 shard (for L2->L3 test)...")
        l1_parents = np.load(l1_path)
        d_parent_l1 = l1_parents.shape[1]  # 8
        d_child_l1 = 2 * d_parent_l1       # 16
        n_half_child_l1 = d_child_l1 // 2  # 8

        l2_survivors = []
        MAX_KERNEL = 500_000
        n_tried = 0
        for i in range(min(50000, len(l1_parents))):
            if len(l2_survivors) >= 500:
                break
            parent = l1_parents[i]
            result = _compute_bin_ranges(parent, m, c_target, d_child_l1, n_half_child_l1)
            if result is None:
                continue
            lo_arr, hi_arr, _ = result
            lo_t, hi_t = lo_arr.copy(), hi_arr.copy()
            total_t = _tighten_ranges(parent, lo_t, hi_t, m, c_target, n_half_child_l1)
            if total_t == 0 or total_t > MAX_KERNEL:
                continue
            n_tried += 1
            out_buf = np.empty((min(int(total_t), 5000), d_child_l1), dtype=np.int32)
            n_surv, _ = _fused_generate_and_prune_gray(
                parent, n_half_child_l1, m, c_target, lo_t, hi_t, out_buf)
            if n_surv > 0:
                take = min(n_surv, 20)
                for si in range(take):
                    if len(l2_survivors) < 500:
                        l2_survivors.append(out_buf[si].copy())
            if n_tried % 100 == 0:
                print(f"  Tried {n_tried} L1 parents, collected {len(l2_survivors)} L2 survivors")

        if l2_survivors:
            l2_arr = np.array(l2_survivors, dtype=np.int32)
            data_sources.append(("L2->L3", l2_arr))
            print(f"  Collected {len(l2_survivors)} L2 survivors at d={l2_arr.shape[1]}")

            # Generate L3 survivors from L2
            print("\nGenerating L3 survivors from L2 (for L3->L4 test)...")
            d_parent_l2 = l2_arr.shape[1]   # 16
            d_child_l2 = 2 * d_parent_l2     # 32
            n_half_child_l2 = d_child_l2 // 2 # 16

            l3_survivors = []
            n_tried = 0
            for i in range(len(l2_arr)):
                if len(l3_survivors) >= 200:
                    break
                parent = l2_arr[i]
                result = _compute_bin_ranges(parent, m, c_target, d_child_l2, n_half_child_l2)
                if result is None:
                    continue
                lo_arr, hi_arr, _ = result
                lo_t, hi_t = lo_arr.copy(), hi_arr.copy()
                total_t = _tighten_ranges(parent, lo_t, hi_t, m, c_target, n_half_child_l2)
                if total_t == 0 or total_t > MAX_KERNEL:
                    continue
                n_tried += 1
                est_s = int(total_t) / 7_000_000
                if est_s > 30:
                    continue
                out_buf = np.empty((min(int(total_t), 5000), d_child_l2), dtype=np.int32)
                n_surv, _ = _fused_generate_and_prune_gray(
                    parent, n_half_child_l2, m, c_target, lo_t, hi_t, out_buf)
                if n_surv > 0:
                    take = min(n_surv, 10)
                    for si in range(take):
                        if len(l3_survivors) < 200:
                            l3_survivors.append(out_buf[si].copy())
                if n_tried % 20 == 0:
                    print(f"  Tried {n_tried} L2 parents, collected {len(l3_survivors)} L3 survivors")

            if l3_survivors:
                l3_arr = np.array(l3_survivors, dtype=np.int32)
                data_sources.append(("L3->L4", l3_arr))
                print(f"  Collected {len(l3_survivors)} L3 survivors at d={l3_arr.shape[1]}")
        else:
            print("  No L2 survivors found (all L1 parents too large)")

    # Now test LP at each level
    for label, parents in data_sources:
        d_parent = parents.shape[1]
        d_child = 2 * d_parent
        n_half_child = d_child // 2

        N = min(500, len(parents))

        print(f"\n{'='*60}")
        print(f"{label}: d_parent={d_parent}, d_child={d_child}, testing {N} parents")
        print(f"{'='*60}")

        n_arc_pruned = 0
        n_arc_survived = 0
        n_lp_killed_base = 0
        n_lp_killed_tight = 0
        lp_times_base = []
        lp_times_tight = []
        children_saved_tight = 0

        for i in range(N):
            parent = parents[i]
            result = _compute_bin_ranges(parent, m, c_target, d_child, n_half_child)
            if result is None:
                n_arc_pruned += 1
                continue
            lo_arr, hi_arr, _ = result
            lo_t, hi_t = lo_arr.copy(), hi_arr.copy()
            total_t = _tighten_ranges(parent, lo_t, hi_t, m, c_target, n_half_child)
            if total_t == 0:
                n_arc_pruned += 1
                continue
            n_arc_survived += 1

            t0 = time.perf_counter()
            inf_base = lp_parent_infeasible(
                parent, lo_t, hi_t, m, c_target, n_half_child)
            lp_times_base.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            inf_tight = lp_parent_infeasible(
                parent, lo_t, hi_t, m, c_target, n_half_child,
                use_variable_threshold=True, n_tangent_cuts=50)
            lp_times_tight.append(time.perf_counter() - t0)

            if inf_base:
                n_lp_killed_base += 1
            if inf_tight:
                n_lp_killed_tight += 1
                children_saved_tight += int(total_t)

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{N}] arc={n_arc_survived}, "
                      f"base_kill={n_lp_killed_base}, tight_kill={n_lp_killed_tight}")

        print(f"\n  RESULTS:")
        print(f"    Total parents:       {N}")
        print(f"    Arc-pruned:          {n_arc_pruned} ({100*n_arc_pruned/N:.1f}%)")
        print(f"    Arc-survived:        {n_arc_survived} ({100*n_arc_survived/N:.1f}%)")
        print(f"    LP baseline kill:    {n_lp_killed_base} "
              f"({100*n_lp_killed_base/max(1,n_arc_survived):.1f}%)")
        print(f"    LP tightened kill:   {n_lp_killed_tight} "
              f"({100*n_lp_killed_tight/max(1,n_arc_survived):.1f}%)")
        print(f"    Children saved:      {children_saved_tight:,.0f}")
        if lp_times_base:
            tb = np.array(lp_times_base)
            tt = np.array(lp_times_tight)
            print(f"    LP base time:    med={np.median(tb)*1000:.1f}ms, "
                  f"p95={np.percentile(tb,95)*1000:.1f}ms")
            print(f"    LP tight time:   med={np.median(tt)*1000:.1f}ms, "
                  f"p95={np.percentile(tt,95)*1000:.1f}ms")


if __name__ == '__main__':
    run_sanity_check()
    run_corner_analysis()
    run_multilevel_cascade()
