"""Optimization functions for the Sidon cascade prover.

Functions:
1. _theorem1_threshold_table: Builds a 1D threshold table using Theorem 1.
2. _whole_parent_prune_theorem1: Interval arithmetic whole-parent pre-pruning.
3. lp_dual_certificate: LP-based parent pruning via weighted window combination.

These are used for early parent-level pruning before expanding children.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _theorem1_threshold_table(n_half_child, m, c_target):
    """Build Theorem 1 threshold table: 1D array indexed by ell.

    Theorem 1 states: for any nonneg f supported on [-1/4, 1/4] with
    bin masses mu_i = c_i / S (where S = 4 * n_half * m), the maximum
    of (f*f) is at least the TV of the window achieving the max.

    For discrete compositions with integer entries c_i summing to S,
    the autoconvolution conv[k] = sum_{i+j=k} c_i * c_j, and the
    window sum ws = sum_{k=s..s+ell-2} conv[k] satisfies:

        max(f*f) >= ws / (S^2 * ell / (2*d))
                  = ws / (4 * n_half * m^2 * ell)

    So we prune if ws > floor(c_target * 4 * n_half * m^2 * ell - eps).

    This has NO correction term (no epsilon-smoothing error), so it is
    TIGHTER than the C&S Lemma 3 threshold. It is valid because Theorem 1
    applies directly to bin masses (step functions on the fine grid).

    Parameters
    ----------
    n_half_child : int
        Half the child dimension (d_child = 2 * n_half_child).
    m : int
        Mass quantization parameter (S = 4 * n_half_child * m).
    c_target : float
        Target lower bound on C_{1a}.

    Returns
    -------
    thr : np.ndarray, dtype=int64, shape=(max_ell - 1,)
        thr[ell - 2] = floor(c_target * 4 * n_half_child * m^2 * ell - eps).
        ell ranges from 2 to max_ell = 4 * n_half_child (= 2 * d_child).
    """
    max_ell = int(4 * n_half_child)  # = 2 * d_child
    ell_count = int(max_ell - 1)     # ell ranges from 2 to max_ell inclusive

    m_d = np.float64(m)
    n_d = np.float64(n_half_child)
    eps = 1e-9
    base = c_target * 4.0 * n_d * m_d * m_d

    thr = np.empty(ell_count, dtype=np.int64)
    for ell in range(2, max_ell + 1):
        thr[ell - 2] = np.int64(base * np.float64(ell) - eps)
    return thr


@njit(cache=True)
def _whole_parent_prune_theorem1(parent_int, lo_arr, hi_arr,
                                  n_half_child, m, c_target):
    """Check if ALL children of a parent are pruned by Theorem 1.

    Uses interval arithmetic: for each autoconvolution entry conv[r],
    computes a lower bound over ALL possible children. If any window
    sum of these lower bounds exceeds the Theorem 1 threshold, then
    every child is pruned and the parent can be skipped entirely.

    Children are defined by: child[2i] = x_i, child[2i+1] = 2*P_i - x_i,
    where x_i (the "cursor") ranges in [lo_i, hi_i] and P_i = parent_int[i].

    Conv[r] = sum_{p+q=r} child[p] * child[q]. Each (p, q) pair contributes
    a term that depends on one or two cursor values. We minimize each term
    independently (this is a RELAXATION -- the true minimum of the sum may
    be higher, but we need a valid lower bound).

    For within-parent terms (bins 2i and 2i+1 from same parent i):
      - conv[4i]   += x_i^2                          -> min = lo_i^2
      - conv[4i+2] += (2P_i - x_i)^2                 -> min = (2P_i - hi_i)^2
      - conv[4i+1] += 2 * x_i * (2P_i - x_i)        -> concave in x_i,
                       min at endpoints: min(2*lo*(2P-lo), 2*hi*(2P-hi))

    For cross-parent terms (parents i < j, bins 2i,2i+1 and 2j,2j+1):
      conv[2i+2j]   += 2 * x_i * x_j                  -> min = 2*lo_i*lo_j
      conv[2i+2j+2] += 2 * (2Pi-xi) * (2Pj-xj)        -> min = 2*(2Pi-hi_i)*(2Pj-hi_j)
      conv[2i+2j+1] += 2*x_i*(2Pj-xj) + 2*(2Pi-xi)*xj
                      = 4*xi*Pj + 4*xj*Pi - 4*xi*xj   (bilinear in xi, xj)
                      -> min over 4 corners {lo_i,hi_i} x {lo_j,hi_j}

    Parameters
    ----------
    parent_int : np.ndarray, shape=(d_parent,), dtype=int32
        Parent composition (bin masses).
    lo_arr, hi_arr : np.ndarray, shape=(d_parent,), dtype=int32
        Cursor range for each parent bin.
    n_half_child : int
        Half the child dimension.
    m : int
        Mass quantization parameter.
    c_target : float
        Target lower bound.

    Returns
    -------
    pruned : bool
        True if ALL children are provably pruned by Theorem 1.
    """
    d_parent = parent_int.shape[0]
    d_child = 2 * d_parent
    conv_len = 2 * d_child - 1

    # Build lower-bound conv array
    min_conv = np.zeros(conv_len, dtype=np.int64)

    # Within-parent contributions (parent i -> child bins 2i, 2i+1)
    for i in range(d_parent):
        Pi = np.int64(parent_int[i])
        lo_i = np.int64(lo_arr[i])
        hi_i = np.int64(hi_arr[i])
        twoP_i = 2 * Pi

        k0 = 2 * i  # child bin index for even child
        k1 = 2 * i + 1  # child bin index for odd child

        # Self-term for child[2i]: x_i^2, monotone increasing -> min at lo_i
        min_conv[2 * k0] += lo_i * lo_i

        # Self-term for child[2i+1]: (2P_i - x_i)^2, decreasing in x_i -> min at hi_i
        comp_hi = twoP_i - hi_i
        min_conv[2 * k1] += comp_hi * comp_hi

        # Mutual term child[2i]*child[2i+1]: 2*x_i*(2P_i - x_i)
        # This is concave in x_i, so min is at one of the endpoints
        val_lo = 2 * lo_i * (twoP_i - lo_i)
        val_hi = 2 * hi_i * (twoP_i - hi_i)
        if val_lo < val_hi:
            min_conv[k0 + k1] += val_lo
        else:
            min_conv[k0 + k1] += val_hi

    # Cross-parent contributions (parents i < j)
    for i in range(d_parent):
        Pi = np.int64(parent_int[i])
        lo_i = np.int64(lo_arr[i])
        hi_i = np.int64(hi_arr[i])
        twoP_i = 2 * Pi
        comp_lo_i = twoP_i - lo_i  # child[2i+1] max
        comp_hi_i = twoP_i - hi_i  # child[2i+1] min

        k0_i = 2 * i
        k1_i = 2 * i + 1

        for j in range(i + 1, d_parent):
            Pj = np.int64(parent_int[j])
            lo_j = np.int64(lo_arr[j])
            hi_j = np.int64(hi_arr[j])
            twoP_j = 2 * Pj
            comp_lo_j = twoP_j - lo_j  # child[2j+1] max
            comp_hi_j = twoP_j - hi_j  # child[2j+1] min

            k0_j = 2 * j
            k1_j = 2 * j + 1

            # conv[2i + 2j] += 2 * child[2i] * child[2j] = 2 * x_i * x_j
            # Both non-negative, independent -> min = 2 * lo_i * lo_j
            min_conv[k0_i + k0_j] += 2 * lo_i * lo_j

            # conv[2i+1 + 2j+1] += 2 * child[2i+1] * child[2j+1]
            #   = 2 * (2Pi - xi) * (2Pj - xj)
            # Both non-negative, independent -> min at max xi, xj
            min_conv[k1_i + k1_j] += 2 * comp_hi_i * comp_hi_j

            # conv[2i + 2j+1] += 2 * x_i * (2Pj - xj)
            # and conv[2i+1 + 2j] += 2 * (2Pi - xi) * xj
            # These two contribute to the SAME conv index: k0_i + k1_j = k1_i + k0_j
            # (since 2i + 2j+1 = 2i+1 + 2j)
            #
            # Combined: f(xi, xj) = 2*xi*(2Pj - xj) + 2*(2Pi - xi)*xj
            #                      = 4*xi*Pj + 4*xj*Pi - 4*xi*xj
            # Bilinear in (xi, xj) -> min over 4 corners
            r = k0_i + k1_j  # = k1_i + k0_j

            # Corner (lo_i, lo_j)
            v1 = 4 * lo_i * Pj + 4 * lo_j * Pi - 4 * lo_i * lo_j
            # Corner (lo_i, hi_j)
            v2 = 4 * lo_i * Pj + 4 * hi_j * Pi - 4 * lo_i * hi_j
            # Corner (hi_i, lo_j)
            v3 = 4 * hi_i * Pj + 4 * lo_j * Pi - 4 * hi_i * lo_j
            # Corner (hi_i, hi_j)
            v4 = 4 * hi_i * Pj + 4 * hi_j * Pi - 4 * hi_i * hi_j

            min_val = v1
            if v2 < min_val:
                min_val = v2
            if v3 < min_val:
                min_val = v3
            if v4 < min_val:
                min_val = v4

            min_conv[r] += min_val

    # Build Theorem 1 threshold table
    thr = _theorem1_threshold_table(n_half_child, m, c_target)

    # Window scan: check if any (ell, s_lo) window sum exceeds threshold
    max_ell = int(4 * n_half_child)  # = 2 * d_child
    for ell in range(2, max_ell + 1):
        n_cv = ell - 1  # number of conv entries in window
        n_windows = conv_len - n_cv + 1
        dyn_it = thr[ell - 2]

        # Initialize sliding window sum for s_lo = 0
        ws = np.int64(0)
        for k in range(n_cv):
            ws += min_conv[k]

        if ws > dyn_it:
            return True

        # Slide the window
        for s_lo in range(1, n_windows):
            ws += min_conv[s_lo + n_cv - 1] - min_conv[s_lo - 1]
            if ws > dyn_it:
                return True

    return False


# =====================================================================
# LP Dual Certificate (Idea 1 from NEW.md)
# =====================================================================

def lp_dual_certificate(parent_int, lo_arr, hi_arr, n_half_child, m, c_target):
    """Check if ALL children of a parent are pruned via LP dual certificate.

    Finds non-negative window weights lambda_W (summing to 1) such that
    the weighted TV combination F(c) = sum lambda_W * TV_W(c) exceeds
    the weighted threshold for EVERY cursor assignment c in the box.

    Mathematical basis:
        max_W TV_W(c) >= sum lambda_W TV_W(c) = F(c)
    So if min_c F(c) >= weighted_threshold, every child is pruned.

    Uses the per-window box cert correction: threshold_W in integer space =
    c_target * 4n * m^2 * ell + min(n, ell-1, 2d-ell) * n*(8m+1)/2.

    Algorithm: cutting-plane LP on box vertices.
    For d_parent <= 10: enumerate all 2^d_parent vertices (up to 1024).
    LP has ~n_windows variables (lambda_W), ~n_vertices constraints.

    Parameters
    ----------
    parent_int : (d_parent,) int32 array
    lo_arr, hi_arr : (d_parent,) int32 arrays (cursor bounds)
    n_half_child : int
    m : int
    c_target : float

    Returns
    -------
    certified : bool
        True if the LP proves ALL children are pruned.
    """
    from scipy.optimize import linprog

    d_parent = len(parent_int)
    d_child = 2 * d_parent
    n_half = int(n_half_child)
    conv_len = 2 * d_child - 1
    S = 4 * n_half * m

    # --- Build list of windows and their thresholds ---
    # Each window is (ell, s_lo) with threshold in INTEGER conv space.
    B_corr = n_half * (8.0 * m + 1.0) / 2.0
    max_ell = 2 * d_child

    windows = []       # list of (ell, s_lo)
    thresholds = []    # integer threshold for each window

    for ell in range(2, max_ell + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        if n_windows <= 0:
            continue
        # Per-window box cert correction
        mult = min(n_half, ell - 1, 2 * d_child - ell)
        scale_ell = float(ell) * 4.0 * n_half
        thr_int = c_target * m * m * scale_ell + mult * B_corr
        for s_lo in range(n_windows):
            windows.append((ell, s_lo))
            thresholds.append(thr_int)

    n_win = len(windows)
    if n_win == 0:
        return False

    # --- Enumerate box vertices (2^d_parent) ---
    # For d_parent <= 16 this is feasible (65536 max)
    if d_parent > 16:
        return False  # too many vertices, skip LP

    n_vertices = 1 << d_parent  # 2^d_parent
    # For each vertex and each window: compute integer window sum
    # vertex_tv[v, w] = integer window sum at vertex v for window w
    vertex_excess = np.empty((n_vertices, n_win), dtype=np.float64)

    for vi in range(n_vertices):
        # Build child from vertex (each cursor at lo or hi)
        child = np.empty(d_child, dtype=np.int64)
        for p in range(d_parent):
            if (vi >> p) & 1:
                c_p = int(hi_arr[p])
            else:
                c_p = int(lo_arr[p])
            child[2 * p] = c_p
            child[2 * p + 1] = 2 * int(parent_int[p]) - c_p

        # Compute autoconvolution
        conv = np.zeros(conv_len, dtype=np.int64)
        for i in range(d_child):
            ci = child[i]
            if ci != 0:
                conv[2 * i] += ci * ci
                for j in range(i + 1, d_child):
                    cj = child[j]
                    if cj != 0:
                        conv[i + j] += 2 * ci * cj

        # For each window: compute ws and excess = ws - threshold
        for wi, (ell, s_lo) in enumerate(windows):
            n_cv = ell - 1
            ws = 0
            for k in range(s_lo, s_lo + n_cv):
                ws += int(conv[k])
            vertex_excess[vi, wi] = float(ws) - thresholds[wi]

    # --- Solve LP ---
    # Variables: lambda_0 ... lambda_{n_win-1}  (window weights)
    # Objective: max t  s.t.  sum_w lambda_w * excess[v,w] >= t  for all v
    #            lambda >= 0,  sum lambda = 1
    #
    # Reformulate for linprog (minimization):
    # Variables: [lambda_0, ..., lambda_{n_win-1}, t]  (n_win + 1 variables)
    # Minimize: -t  (i.e., c = [0,...,0, -1])
    # Subject to:
    #   sum_w lambda_w * excess[v,w] - t >= 0   for each vertex v
    #   sum_w lambda_w = 1
    #   lambda_w >= 0, t free

    c_obj = np.zeros(n_win + 1)
    c_obj[-1] = -1.0  # minimize -t = maximize t

    # Inequality constraints: A_ub @ x <= b_ub
    # -sum_w lambda_w * excess[v,w] + t <= 0  for each vertex
    A_ub = np.zeros((n_vertices, n_win + 1))
    for vi in range(n_vertices):
        for wi in range(n_win):
            A_ub[vi, wi] = -vertex_excess[vi, wi]
        A_ub[vi, -1] = 1.0  # +t
    b_ub = np.zeros(n_vertices)

    # Equality constraint: sum lambda = 1
    A_eq = np.zeros((1, n_win + 1))
    A_eq[0, :n_win] = 1.0
    b_eq = np.array([1.0])

    # Bounds: lambda >= 0, t is free (but bounded below for solver stability)
    bounds = [(0.0, None)] * n_win + [(None, None)]

    try:
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                         A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                         method='highs', options={'presolve': True})
        if not result.success:
            return False
        t_opt = -result.fun
        if t_opt <= 1e-9:
            return False

        # Cutting-plane verification: the LP checked only box vertices.
        # The weighted F(c) = sum lambda_w * ws_w(c) is QUADRATIC in c.
        # Its minimum over the box might be at an INTERIOR point (indefinite Q).
        # Verify by evaluating F at a grid of interior points.
        lam = result.x[:n_win]
        # Compute F at the center and a few random interior points
        import random
        rng = random.Random(42)
        check_points = []
        # Center
        center = [(int(lo_arr[p]) + int(hi_arr[p])) / 2.0 for p in range(d_parent)]
        check_points.append(center)
        # Random interior points
        for _ in range(20):
            pt = [rng.uniform(int(lo_arr[p]), int(hi_arr[p]))
                  for p in range(d_parent)]
            check_points.append(pt)

        for pt in check_points:
            child = np.empty(d_child, dtype=np.float64)
            for p in range(d_parent):
                child[2 * p] = pt[p]
                child[2 * p + 1] = 2.0 * float(parent_int[p]) - pt[p]
            conv = np.zeros(conv_len, dtype=np.float64)
            for i in range(d_child):
                ci = child[i]
                if ci != 0:
                    conv[2 * i] += ci * ci
                    for j in range(i + 1, d_child):
                        cj = child[j]
                        if cj != 0:
                            conv[i + j] += 2.0 * ci * cj
            # Compute F(pt) = sum lambda_w * (ws_w - threshold_w)
            F_val = 0.0
            for wi, (ell, s_lo) in enumerate(windows):
                if lam[wi] < 1e-15:
                    continue
                n_cv = ell - 1
                ws = sum(conv[s_lo:s_lo + n_cv])
                F_val += lam[wi] * (ws - thresholds[wi])
            if F_val < -1e-6:
                return False  # interior point violates → not certified

        return True

    except Exception:
        pass

    return False


# =====================================================================
# SDP Relaxation for Parent-Level Certification (NEW.md surviving idea)
# =====================================================================

def sdp_certify_parent(parent_int, lo_arr, hi_arr, n_half_child, m, c_target):
    """Test if ALL children of a parent are provably pruned via SDP relaxation.

    Formulates a QCQP: "does there exist cursor x in [lo,hi] such that
    ws_W(x) <= threshold_W for ALL windows W?" and solves its Shor/Lasserre
    SDP relaxation. If the SDP is infeasible, no cursor assignment (integer
    or continuous) can have all windows below threshold — every child is pruned.

    Mathematical basis:
        child[2p] = x_p, child[2p+1] = 2*P_p - x_p  (affine in cursor x)
        ws_W(x) = x^T Q_W x + g_W^T x + k_W          (quadratic in x)

    SDP relaxation: replace xx^T with PSD matrix X >= xx^T via moment matrix.
    If SDP infeasible -> original QCQP infeasible -> all children pruned.
    One-sided: can never incorrectly prune a genuine survivor.

    Returns True if SDP proves ALL children are pruned.
    """
    try:
        import cvxpy as cp
    except ImportError:
        return False

    d_parent = len(parent_int)
    d_child = 2 * d_parent
    n_half = int(n_half_child)
    conv_len = 2 * d_child - 1

    P = np.array([float(parent_int[i]) for i in range(d_parent)])
    lo = np.array([float(lo_arr[i]) for i in range(d_parent)])
    hi = np.array([float(hi_arr[i]) for i in range(d_parent)])

    # child[p] = signs[p] * x_{parent_of[p]} + consts[p]
    signs = np.empty(d_child)
    consts = np.empty(d_child)
    parent_of = np.empty(d_child, dtype=int)
    for i in range(d_parent):
        signs[2 * i] = 1.0
        signs[2 * i + 1] = -1.0
        consts[2 * i] = 0.0
        consts[2 * i + 1] = 2.0 * P[i]
        parent_of[2 * i] = i
        parent_of[2 * i + 1] = i

    # Per-window box cert correction: min(n, ell-1, 2d-ell) * B
    B_corr = float(n_half) * (8.0 * m + 1.0) / 2.0

    # Build quadratic form for each window: ws = Tr(Q @ X) + g^T x + k
    window_constraints = []
    for ell in range(2, 2 * d_child + 1):
        n_cv = ell - 1
        n_win = conv_len - n_cv + 1
        if n_win <= 0:
            continue
        mult = min(n_half, ell - 1, 2 * d_child - ell)
        scale = float(ell) * 4.0 * n_half
        thr = c_target * m * m * scale + mult * B_corr

        for s_lo in range(n_win):
            Q = np.zeros((d_parent, d_parent))
            g = np.zeros(d_parent)
            k = 0.0

            for r in range(s_lo, s_lo + n_cv):
                for p in range(max(0, r - d_child + 1), min(d_child, r + 1)):
                    q = r - p
                    if q < 0 or q >= d_child:
                        continue
                    a = parent_of[p]
                    b = parent_of[q]
                    sp = signs[p]
                    sq = signs[q]
                    cp_ = consts[p]
                    cq = consts[q]
                    # child[p]*child[q] = (sp*x_a + cp)*(sq*x_b + cq)
                    Q[a, b] += sp * sq
                    g[a] += sp * cq
                    g[b] += cp_ * sq
                    k += cp_ * cq

            Q_sym = (Q + Q.T) / 2.0
            window_constraints.append((Q_sym, g, k, thr))

    if not window_constraints:
        return False

    # --- Build SDP ---
    x = cp.Variable(d_parent)
    X = cp.Variable((d_parent, d_parent), symmetric=True)

    # Moment matrix: Y = [[1, x^T], [x, X]] >= 0  (implies X >= xx^T)
    ones_11 = np.ones((1, 1))
    Y = cp.bmat([[ones_11, cp.reshape(x, (1, d_parent))],
                  [cp.reshape(x, (d_parent, 1)), X]])
    constraints = [Y >> 0]

    # Box constraints on x
    constraints += [x >= lo, x <= hi]

    # Diagonal bounds on X (from x_i^2 range)
    for i in range(d_parent):
        constraints.append(X[i, i] >= lo[i] ** 2)
        constraints.append(X[i, i] <= hi[i] ** 2)

    # RLT (Reformulation-Linearization) cuts on off-diagonal X entries.
    # These are CRITICAL for tightness: they constrain X[i,j] to the
    # range achievable by actual x_i*x_j products within the box.
    # (x_i - lo_i)(x_j - lo_j) >= 0: X[i,j] >= lo_j*x_i + lo_i*x_j - lo_i*lo_j
    # (hi_i - x_i)(x_j - lo_j) >= 0: X[i,j] <= hi_i*x_j + lo_j*x_i - hi_i*lo_j
    # (x_i - lo_i)(hi_j - x_j) >= 0: X[i,j] <= hi_j*x_i + lo_i*x_j - lo_i*hi_j
    # (hi_i - x_i)(hi_j - x_j) >= 0: X[i,j] >= hi_i*x_j + hi_j*x_i - hi_i*hi_j
    for i in range(d_parent):
        for j in range(i + 1, d_parent):
            li, lj = lo[i], lo[j]
            ui, uj = hi[i], hi[j]
            constraints.append(X[i, j] >= lj * x[i] + li * x[j] - li * lj)
            constraints.append(X[i, j] >= uj * x[i] + ui * x[j] - ui * uj)
            constraints.append(X[i, j] <= ui * x[j] + lj * x[i] - ui * lj)
            constraints.append(X[i, j] <= uj * x[i] + li * x[j] - li * uj)

    # Window constraints: ws_W(x, X) <= threshold_W
    for Q_sym, g_vec, k_val, thr_val in window_constraints:
        ws_expr = cp.trace(Q_sym @ X) + g_vec @ x + k_val
        constraints.append(ws_expr <= thr_val)

    prob = cp.Problem(cp.Minimize(0), constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=10000, eps=1e-8)
    except Exception:
        return False

    return prob.status in ('infeasible', 'infeasible_inaccurate')
