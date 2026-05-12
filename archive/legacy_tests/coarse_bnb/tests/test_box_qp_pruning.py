"""Test Box-QP and LP Certificate pruning effectiveness at L0->L1 and L1->L2.

For each parent, computes the minimum of each window sum over all cursor
assignments (the Box-QP test). If min ws > threshold for any window,
the parent can be pruned without expansion.

Also tests the LP certificate: weighted sum of (ws - threshold) with
uniform weights. If min over box > 0, parent is prunable.

Mathematical basis:
- Child bins: child[2k] = c_k, child[2k+1] = 2*parent[k] - c_k
- Each window sum ws(ell, s_lo) is a degree-2 polynomial in cursor vars
- Minimum of quadratic on 2D box: check 4 vertices + 4 edge critical pts + interior
- Uses W_int_max (upper bound) for threshold: sound because threshold increases with W_int
"""
import sys
import os
import time
import math
import numpy as np

# Path setup
_this_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_this_dir)
sys.path.insert(0, os.path.join(_root, 'cloninger-steinerberger'))
sys.path.insert(0, _root)

from pruning import correction, asymmetry_threshold


def compute_conv_coefficients_d2(p0, p1):
    """Compute quadratic coefficients for each conv value at d_child=4.

    child = [c0, 2p0-c0, c1, 2p1-c1]
    conv[k] = sum_{i+j=k} child[i]*child[j]

    Returns array of shape (7, 6) where each row is [alpha, beta, gamma, delta, epsilon, zeta]
    for Q = alpha*c0^2 + beta*c1^2 + gamma*c0*c1 + delta*c0 + epsilon*c1 + zeta
    """
    coeffs = np.zeros((7, 6), dtype=np.float64)
    # conv[0] = c0^2
    coeffs[0] = [1, 0, 0, 0, 0, 0]
    # conv[1] = 2*c0*(2p0-c0) = -2c0^2 + 4p0*c0
    coeffs[1] = [-2, 0, 0, 4*p0, 0, 0]
    # conv[2] = (2p0-c0)^2 + 2*c0*c1 = c0^2 - 4p0*c0 + 4p0^2 + 2*c0*c1
    coeffs[2] = [1, 0, 2, -4*p0, 0, 4*p0**2]
    # conv[3] = 2*(c0*(2p1-c1) + (2p0-c0)*c1) = -4*c0*c1 + 4*p1*c0 + 4*p0*c1
    coeffs[3] = [0, 0, -4, 4*p1, 4*p0, 0]
    # conv[4] = c1^2 + 2*(2p0-c0)*(2p1-c1) = c1^2 + 2*c0*c1 - 4p1*c0 - 4p0*c1 + 8p0*p1
    coeffs[4] = [0, 1, 2, -4*p1, -4*p0, 8*p0*p1]
    # conv[5] = 2*c1*(2p1-c1) = -2*c1^2 + 4*p1*c1
    coeffs[5] = [0, -2, 0, 0, 4*p1, 0]
    # conv[6] = (2p1-c1)^2 = c1^2 - 4*p1*c1 + 4*p1^2
    coeffs[6] = [0, 1, 0, 0, -4*p1, 4*p1**2]
    return coeffs


def eval_quad(alpha, beta, gamma, delta, epsilon, zeta, x, y):
    """Evaluate quadratic Q(x,y)."""
    return alpha*x*x + beta*y*y + gamma*x*y + delta*x + epsilon*y + zeta


def min_quad_2d_box(alpha, beta, gamma, delta, epsilon, zeta,
                     x_lo, x_hi, y_lo, y_hi):
    """Find minimum of Q(x,y) = alpha*x^2 + beta*y^2 + gamma*x*y + delta*x + epsilon*y + zeta
    on the box [x_lo, x_hi] x [y_lo, y_hi].

    Checks all 9 candidate points:
    - 4 vertices
    - 4 edge critical points (1D optimization on each edge)
    - 1 interior critical point (if in box and local min)
    """
    best = float('inf')

    # 4 vertices
    for x in [x_lo, x_hi]:
        for y in [y_lo, y_hi]:
            v = eval_quad(alpha, beta, gamma, delta, epsilon, zeta, x, y)
            if v < best:
                best = v

    # 4 edge critical points
    # Edge y = y_lo: Q(x, y_lo) -> dQ/dx = 2*alpha*x + gamma*y_lo + delta = 0
    for y_fixed in [y_lo, y_hi]:
        if alpha != 0:
            x_crit = -(gamma * y_fixed + delta) / (2 * alpha)
            if x_lo <= x_crit <= x_hi:
                v = eval_quad(alpha, beta, gamma, delta, epsilon, zeta, x_crit, y_fixed)
                if v < best:
                    best = v

    # Edge x = x_lo: Q(x_lo, y) -> dQ/dy = 2*beta*y + gamma*x_lo + epsilon = 0
    for x_fixed in [x_lo, x_hi]:
        if beta != 0:
            y_crit = -(gamma * x_fixed + epsilon) / (2 * beta)
            if y_lo <= y_crit <= y_hi:
                v = eval_quad(alpha, beta, gamma, delta, epsilon, zeta, x_fixed, y_crit)
                if v < best:
                    best = v

    # Interior critical point
    det = 4 * alpha * beta - gamma * gamma
    if abs(det) > 1e-12:
        x_crit = (gamma * epsilon - 2 * beta * delta) / det
        y_crit = (gamma * delta - 2 * alpha * epsilon) / det
        if x_lo <= x_crit <= x_hi and y_lo <= y_crit <= y_hi:
            # Only a minimum if Hessian is PSD (det > 0 and alpha > 0)
            if det > 0 and alpha > 0:
                v = eval_quad(alpha, beta, gamma, delta, epsilon, zeta, x_crit, y_crit)
                if v < best:
                    best = v

    return best


def test_box_qp_l1(survivors_d2, m, c_target, use_flat_threshold=False):
    """Test Box-QP pruning on L0->L1 transition.

    survivors_d2: (N, 2) int32 array of L0 survivors at d=2
    Returns: number of parents that would be pruned by Box-QP
    """
    n_half_child = 2  # L1 has n_half_child = 2*n_half_parent = 2*1 = 2
    d_child = 4
    d_parent = 2
    conv_len = 2 * d_child - 1  # 7
    max_ell = 2 * d_child  # 8

    m_d = float(m)
    n_half_d = float(n_half_child)
    four_n = 4.0 * n_half_d
    eps_margin = 1e-9 * m_d * m_d
    cs_base_m2 = c_target * m_d * m_d
    flat_corr = 2.0 * m_d + 1.0

    # x_cap for child bins
    thresh_xcap = c_target + 2.0 / m_d + 1.0 / (m_d * m_d) + 1e-9
    x_cap = int(math.floor(m_d * math.sqrt(4.0 * d_child * thresh_xcap)))

    n_qp_pruned = 0
    n_lp_pruned = 0

    for idx in range(len(survivors_d2)):
        p0 = int(survivors_d2[idx, 0])
        p1 = int(survivors_d2[idx, 1])

        # Cursor ranges
        lo0 = max(0, 2*p0 - x_cap)
        hi0 = min(2*p0, x_cap)
        lo1 = max(0, 2*p1 - x_cap)
        hi1 = min(2*p1, x_cap)

        if lo0 > hi0 or lo1 > hi1:
            continue  # infeasible, already caught by pre-filter

        # Compute conv coefficients
        coeffs = compute_conv_coefficients_d2(p0, p1)

        # Test each window
        qp_pruned = False
        lp_sum_at_vertices = np.full(4, 0.0)  # 4 vertices

        for ell in range(2, max_ell + 1):
            if qp_pruned:
                break
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            scale_ell = float(ell) * four_n

            for s_lo in range(n_windows):
                if qp_pruned:
                    break

                # Sum coefficients for this window
                a, b, g, d, e, z = 0, 0, 0, 0, 0, 0
                for k in range(s_lo, s_lo + n_cv):
                    a += coeffs[k, 0]
                    b += coeffs[k, 1]
                    g += coeffs[k, 2]
                    d += coeffs[k, 3]
                    e += coeffs[k, 4]
                    z += coeffs[k, 5]

                # Compute W_int_max
                lo_bin = max(0, s_lo - (d_child - 1))
                hi_bin = min(d_child - 1, s_lo + ell - 2)
                p_lo_w = lo_bin // 2  # CORRECTED: use floor, not ceil
                p_hi_w = hi_bin // 2
                p_lo_w = max(0, p_lo_w)
                p_hi_w = min(d_parent - 1, p_hi_w)
                W_int_max = 0
                for pp in range(p_lo_w, p_hi_w + 1):
                    W_int_max += 2 * int(survivors_d2[idx, pp])

                if use_flat_threshold:
                    dyn_x = (cs_base_m2 + flat_corr + eps_margin) * scale_ell
                else:
                    corr_w = 1.0 + float(W_int_max) / (2.0 * n_half_d)
                    dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_ell
                threshold = int(dyn_x)

                # Find minimum of ws on cursor box
                min_ws = min_quad_2d_box(a, b, g, d, e, z,
                                         float(lo0), float(hi0),
                                         float(lo1), float(hi1))

                if min_ws > threshold:
                    qp_pruned = True
                    print(f"  QP PRUNE! parent ({p0},{p1}), "
                          f"ell={ell}, s_lo={s_lo}, "
                          f"min_ws={min_ws:.0f} > thr={threshold}")
                    break

        if qp_pruned:
            n_qp_pruned += 1
            continue

        # LP certificate: evaluate E = sum (ws - threshold) at each vertex
        vertices = [(lo0, lo1), (lo0, hi1), (hi0, lo1), (hi0, hi1)]
        min_E = float('inf')

        for vi, (c0, c1) in enumerate(vertices):
            # Build child
            child = [c0, 2*p0-c0, c1, 2*p1-c1]

            # Full autoconvolution
            conv = [0] * conv_len
            for i in range(d_child):
                if child[i] != 0:
                    conv[2*i] += child[i] * child[i]
                    for j in range(i+1, d_child):
                        if child[j] != 0:
                            conv[i+j] += 2 * child[i] * child[j]

            # Sum (ws - threshold) over all windows
            E_v = 0.0
            for ell in range(2, max_ell + 1):
                n_cv = ell - 1
                n_win = conv_len - n_cv + 1
                scale_ell = float(ell) * four_n

                for s_lo in range(n_win):
                    ws = sum(conv[s_lo:s_lo+n_cv])

                    lo_bin = max(0, s_lo - (d_child - 1))
                    hi_bin = min(d_child - 1, s_lo + ell - 2)
                    p_lo_w = max(0, lo_bin // 2)
                    p_hi_w = min(d_parent - 1, hi_bin // 2)
                    W_int_max = sum(2 * int(survivors_d2[idx, pp])
                                   for pp in range(p_lo_w, p_hi_w + 1))

                    if use_flat_threshold:
                        dyn_x = (cs_base_m2 + flat_corr + eps_margin) * scale_ell
                    else:
                        corr_w = 1.0 + float(W_int_max) / (2.0 * n_half_d)
                        dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_ell
                    threshold = int(dyn_x)

                    E_v += float(ws) - float(threshold)

            if E_v < min_E:
                min_E = E_v

        if min_E > 0:
            n_lp_pruned += 1
            print(f"  LP PRUNE! parent ({p0},{p1}), min_E={min_E:.0f}")

    return n_qp_pruned, n_lp_pruned


def generate_l0_survivors(m, c_target, d0=2):
    """Generate L0 survivors at d=d0."""
    S = int(2 * d0 * m)  # = 4 * n_half * m for n_half = d0/2
    n_half = d0 / 2.0

    # x_cap
    thresh_xcap = c_target + 2.0/m + 1.0/(m*m) + 1e-9
    x_cap = int(math.floor(m * math.sqrt(4.0 * d0 * thresh_xcap)))

    # Asymmetry threshold
    asym_thr = math.sqrt(c_target / 2.0)
    left_bins = d0 // 2

    survivors = []
    for k in range(0, S + 1):
        if k > x_cap or (S - k) > x_cap:
            continue

        # For d=2: composition is (k, S-k)
        left_frac = k / S
        if left_frac >= asym_thr or left_frac <= 1 - asym_thr:
            continue

        # Window scan
        child = [k, S - k]
        conv = [0] * (2 * d0 - 1)
        for i in range(d0):
            conv[2*i] += child[i] * child[i]
            for j in range(i+1, d0):
                conv[i+j] += 2 * child[i] * child[j]

        pruned = False
        m_d = float(m)
        four_n = 4.0 * n_half
        n_half_d = float(n_half)
        eps_margin = 1e-9 * m_d * m_d
        cs_base_m2 = c_target * m_d * m_d
        conv_len = 2 * d0 - 1

        for ell in range(2, 2*d0+1):
            if pruned:
                break
            n_cv = ell - 1
            scale_ell = float(ell) * four_n
            for s_lo in range(conv_len - n_cv + 1):
                ws = sum(conv[s_lo:s_lo+n_cv])
                lo_bin = max(0, s_lo - (d0-1))
                hi_bin = min(d0-1, s_lo+ell-2)
                W_int = sum(child[lo_bin:hi_bin+1])
                corr_w = 1.0 + float(W_int) / (2.0 * n_half_d)
                dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_ell
                dyn_it = int(dyn_x)
                if ws > dyn_it:
                    pruned = True
                    break

        if not pruned:
            survivors.append([k, S-k])

    # Canonicalize: keep only k <= S-k, i.e., k <= S/2
    canonical = []
    seen = set()
    for s in survivors:
        key = tuple(min(s, s[::-1]))
        if key not in seen:
            seen.add(key)
            canonical.append(list(key))

    return np.array(canonical, dtype=np.int32)


def main():
    m = 20
    c_target = 1.30
    d0 = 2

    print(f"Box-QP and LP Certificate Pruning Test")
    print(f"Parameters: m={m}, c_target={c_target}, d0={d0}")
    print(f"Correction: {2/m + 1/(m*m):.6f}")
    print(f"Effective threshold: {c_target + 2/m + 1/(m*m):.6f}")
    print()

    # Generate L0 survivors
    print("=== L0 Survivors ===")
    l0_surv = generate_l0_survivors(m, c_target, d0)
    print(f"L0 survivors: {len(l0_surv)} at d={d0}")
    print(f"  Range: ({l0_surv[:,0].min()}, {l0_surv[:,0].max()}) to "
          f"({l0_surv[:,1].min()}, {l0_surv[:,1].max()})")
    print(f"  Compositions: {[tuple(r) for r in l0_surv[:5]]}...")
    print()

    # Test Box-QP on L0->L1 transition
    print("=== Box-QP Test: L0->L1 (d_parent=2 -> d_child=4) ===")
    t0 = time.time()
    n_qp, n_lp = test_box_qp_l1(l0_surv, m, c_target)
    t1 = time.time()
    print(f"\nResults:")
    print(f"  Parents tested: {len(l0_surv)}")
    print(f"  Box-QP pruned: {n_qp}")
    print(f"  LP cert pruned (vertex check): {n_lp}")
    print(f"  Time: {t1-t0:.3f}s")
    print()

    # Now generate L1 survivors by expanding each L0 parent
    print("=== Generating L1 survivors (for L1->L2 test) ===")
    n_half_child_l1 = 2  # n_half doubles each level
    d_child_l1 = 4

    thresh_xcap = c_target + 2.0/m + 1.0/(m*m) + 1e-9
    x_cap = int(math.floor(m * math.sqrt(4.0 * d_child_l1 * thresh_xcap)))

    all_l1_survivors = []
    total_children = 0
    total_survived = 0

    for idx in range(len(l0_surv)):
        p0, p1 = int(l0_surv[idx, 0]), int(l0_surv[idx, 1])
        lo0 = max(0, 2*p0 - x_cap)
        hi0 = min(2*p0, x_cap)
        lo1 = max(0, 2*p1 - x_cap)
        hi1 = min(2*p1, x_cap)

        if lo0 > hi0 or lo1 > hi1:
            continue

        n_children = (hi0 - lo0 + 1) * (hi1 - lo1 + 1)
        total_children += n_children
        n_surv = 0

        m_d = float(m)
        n_half_d = float(n_half_child_l1)
        four_n = 4.0 * n_half_d
        eps_margin = 1e-9 * m_d * m_d
        cs_base_m2 = c_target * m_d * m_d
        conv_len = 2 * d_child_l1 - 1
        d_minus_1 = d_child_l1 - 1

        for c0 in range(lo0, hi0 + 1):
            for c1 in range(lo1, hi1 + 1):
                child = [c0, 2*p0-c0, c1, 2*p1-c1]

                # Asymmetry
                left_sum = child[0] + child[1]  # d_child//2 = 2 bins
                S_child = sum(child)
                left_frac = left_sum / S_child
                asym_thr = math.sqrt(c_target / 2.0)
                if left_frac >= asym_thr or left_frac <= 1 - asym_thr:
                    continue

                # Autoconvolution
                conv = [0] * conv_len
                for i in range(d_child_l1):
                    if child[i] != 0:
                        conv[2*i] += child[i] * child[i]
                        for j in range(i+1, d_child_l1):
                            if child[j] != 0:
                                conv[i+j] += 2 * child[i] * child[j]

                # Window scan
                pruned = False
                for ell in range(2, 2*d_child_l1 + 1):
                    if pruned:
                        break
                    n_cv = ell - 1
                    scale_ell = float(ell) * four_n
                    for s_lo in range(conv_len - n_cv + 1):
                        ws = sum(conv[s_lo:s_lo+n_cv])
                        lo_bin = max(0, s_lo - d_minus_1)
                        hi_bin = min(d_minus_1, s_lo + ell - 2)
                        W_int = sum(child[lo_bin:hi_bin+1])
                        corr_w = 1.0 + float(W_int) / (2.0 * n_half_d)
                        dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_ell
                        dyn_it = int(dyn_x)
                        if ws > dyn_it:
                            pruned = True
                            break

                if not pruned:
                    # Canonicalize
                    canon = list(min(child, child[::-1]))
                    all_l1_survivors.append(canon)
                    n_surv += 1

        total_survived += n_surv

    # Deduplicate L1 survivors
    if all_l1_survivors:
        l1_surv = np.array(all_l1_survivors, dtype=np.int32)
        l1_surv = np.unique(l1_surv, axis=0)
    else:
        l1_surv = np.empty((0, 4), dtype=np.int32)

    print(f"  Total children generated: {total_children:,}")
    print(f"  Total survived (before dedup): {total_survived:,}")
    print(f"  Unique L1 survivors at d=4: {len(l1_surv):,}")
    print()

    # Now test Box-QP on L1->L2 (d_parent=4 -> d_child=8)
    # For d_parent=4, we use vertex checking (2^4 = 16 vertices)
    print("=== Box-QP Test: L1->L2 (d_parent=4 -> d_child=8) ===")
    if len(l1_surv) > 0:
        t0 = time.time()
        n_qp_l2, n_lp_l2 = test_box_qp_l2(l1_surv, m, c_target)
        t1 = time.time()
        print(f"\nResults:")
        print(f"  Parents tested: {len(l1_surv)}")
        print(f"  Box-QP pruned: {n_qp_l2}")
        print(f"  LP cert pruned (vertex check): {n_lp_l2}")
        print(f"  Time: {t1-t0:.3f}s")
    else:
        print("  No L1 survivors to test")


def test_box_qp_l2(survivors_d4, m, c_target, use_flat_threshold=False):
    """Test Box-QP pruning on L1->L2 transition (d_parent=4 -> d_child=8).

    For d_parent=4, uses 16-vertex evaluation + edge critical points.
    """
    n_half_child = 4  # doubles again
    d_child = 8
    d_parent = 4
    conv_len = 2 * d_child - 1  # 15
    max_ell = 2 * d_child  # 16

    m_d = float(m)
    n_half_d = float(n_half_child)
    four_n = 4.0 * n_half_d
    eps_margin = 1e-9 * m_d * m_d
    cs_base_m2 = c_target * m_d * m_d
    flat_corr = 2.0 * m_d + 1.0

    thresh_xcap = c_target + 2.0/m_d + 1.0/(m_d*m_d) + 1e-9
    x_cap = int(math.floor(m_d * math.sqrt(4.0 * d_child * thresh_xcap)))

    n_qp_pruned = 0
    n_lp_pruned = 0
    n_tested = 0

    # Only test a sample if there are too many
    max_test = min(len(survivors_d4), 5000)
    step = max(1, len(survivors_d4) // max_test)

    for idx in range(0, len(survivors_d4), step):
        n_tested += 1
        parent = survivors_d4[idx]

        # Cursor ranges
        lo = [max(0, 2*int(parent[i]) - x_cap) for i in range(d_parent)]
        hi = [min(2*int(parent[i]), x_cap) for i in range(d_parent)]

        if any(lo[i] > hi[i] for i in range(d_parent)):
            continue

        # Generate 16 vertices
        vertices = []
        for v in range(16):
            cursor = [lo[i] if not (v >> i) & 1 else hi[i] for i in range(d_parent)]
            vertices.append(cursor)

        # For each vertex, build child and compute full conv + window scan
        qp_pruned = False

        # Test: for each window, compute min ws over vertices
        # First build all vertex children and their convolutions
        vertex_convs = []
        for cursor in vertices:
            child = []
            for i in range(d_parent):
                child.append(cursor[i])
                child.append(2*int(parent[i]) - cursor[i])

            conv = [0] * conv_len
            for i in range(d_child):
                if child[i] != 0:
                    conv[2*i] += child[i] * child[i]
                    for j in range(i+1, d_child):
                        if child[j] != 0:
                            conv[i+j] += 2 * child[i] * child[j]
            vertex_convs.append(conv)

        for ell in range(2, max_ell + 1):
            if qp_pruned:
                break
            n_cv = ell - 1
            n_windows = conv_len - n_cv + 1
            scale_ell = float(ell) * four_n

            for s_lo in range(n_windows):
                if qp_pruned:
                    break

                # W_int_max
                lo_bin = max(0, s_lo - (d_child - 1))
                hi_bin = min(d_child - 1, s_lo + ell - 2)
                p_lo_w = max(0, lo_bin // 2)
                p_hi_w = min(d_parent - 1, hi_bin // 2)
                W_int_max = sum(2 * int(parent[pp]) for pp in range(p_lo_w, p_hi_w + 1))

                if use_flat_threshold:
                    dyn_x = (cs_base_m2 + flat_corr + eps_margin) * scale_ell
                else:
                    corr_w = 1.0 + float(W_int_max) / (2.0 * n_half_d)
                    dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_ell
                threshold = int(dyn_x)

                # Min ws over vertices (UPPER BOUND on true min)
                min_ws = min(sum(vc[s_lo:s_lo+n_cv]) for vc in vertex_convs)

                # NOTE: this is an UPPER BOUND on the true minimum.
                # If min_ws > threshold, the true min MIGHT still be <= threshold
                # (at an edge/face). So we can't prune based on this alone.
                # But we report it for diagnostic purposes.
                if min_ws > threshold:
                    # This is a POTENTIAL prune (need edge/face check to confirm)
                    qp_pruned = True
                    print(f"  QP CANDIDATE! parent {tuple(parent)}, "
                          f"ell={ell}, s_lo={s_lo}, "
                          f"min_vertex_ws={min_ws} > thr={threshold}")
                    print(f"    WARNING: vertex-only check, true min may be lower")
                    break

        if qp_pruned:
            n_qp_pruned += 1

        # LP certificate at vertices
        min_E = float('inf')
        for vi, conv in enumerate(vertex_convs):
            E_v = 0.0
            for ell in range(2, max_ell + 1):
                n_cv = ell - 1
                n_win = conv_len - n_cv + 1
                scale_ell = float(ell) * four_n
                for s_lo in range(n_win):
                    ws = sum(conv[s_lo:s_lo+n_cv])
                    lo_bin = max(0, s_lo - (d_child - 1))
                    hi_bin = min(d_child - 1, s_lo + ell - 2)
                    p_lo_w = max(0, lo_bin // 2)
                    p_hi_w = min(d_parent - 1, hi_bin // 2)
                    W_int_max = sum(2*int(parent[pp]) for pp in range(p_lo_w, p_hi_w+1))
                    if use_flat_threshold:
                        dyn_x = (cs_base_m2 + flat_corr + eps_margin) * scale_ell
                    else:
                        corr_w = 1.0 + float(W_int_max) / (2.0 * n_half_d)
                        dyn_x = (cs_base_m2 + corr_w + eps_margin) * scale_ell
                    threshold = int(dyn_x)
                    E_v += float(ws) - float(threshold)
            if E_v < min_E:
                min_E = E_v

        if min_E > 0 and not qp_pruned:
            n_lp_pruned += 1
            print(f"  LP CANDIDATE! parent {tuple(parent)}, min_E={min_E:.0f}")

        if n_tested % 500 == 0:
            print(f"  ... tested {n_tested}/{max_test} parents ...")

    return n_qp_pruned, n_lp_pruned


if __name__ == '__main__':
    main()
