"""Adaptive branch-and-bound exact cell box-cert.

The cascade's per-cell certificate at depth 0 (variant N+O, run_cascade_coarse_v3)
fails on a fraction of "hard" compositions. For those cells, we recursively
subdivide and re-bound. This file implements:

  cell_cert_bnb(c_int, S, d, c_target, max_depth) -> bool

which returns True iff the cell {mu = mu* + delta : |delta_i|<=h, Σdelta=0,
mu_i>=0} can be certified TV_W(mu) > c_target for some W, by recursive
subdivision.

============================================================================
SOUNDNESS
============================================================================
At each leaf, we use a sound LOWER BOUND on min_{delta in cell} TV_W(mu*+delta)
for each window W.  Anchoring at c' := projection of c_box onto {sum delta = 0}
(c' = c_box - mean(c_box)*1), let eps = delta - c'.  The Σ-constraint becomes
Σ eps = 0.  Algebraic identity (A_W symmetric):

  TV_W(mu* + delta) = TV_W(mu* + c')
                    + (4d/ell) * (A_W (mu*+c')) . eps
                    + (2d/ell) * eps^T A_W eps.

Sound bounds over the eps-cell:
  |grad_c' . eps| <= L_W (LP closed-form max over [lo-c', hi-c'] cap mu>=0,
                            Σeps=0)
  |eps^T A_W eps| <= sum_{(i,j) in pairs(W)} M_i M_j, where
                       M_i := max(|lo_i - c'_i|, |hi_i - c'_i|).

Net:  bound_W = TV_W(mu*+c') - c_target - L_W - (2d/ell) Σ M_i M_j
The cell is W-certified if bound_W >= 0; the cell is certified if some W
gives bound_W >= 0.

OR-aggregate: cell = sub_cell_1 U sub_cell_2  (split axis i*: hi_i := mid in
sub_cell_1, lo_i := mid in sub_cell_2).  If both sub-cells are certified at
depth+1, the parent is certified (union of certified pieces).

CONVERGENCE: as depth grows, the per-axis half-widths shrink along splitted
axes; for each leaf, M_i and L_W -> 0 along those axes, and the bound
approaches TV_W(mu*+c) - c_target for c = the leaf's center (a feasible point
in the leaf's interior).  Hence depth -> infty exhausts every cell whose true
minimum exceeds c_target.

NOTE: the entry-wise quadratic bound is NOT optimal — splitting one axis at
a time gives bound shrinkage proportional to 2 * (#-axes-split).  To halve
the bound uniformly, BnB needs depth ~ d.  This explains the slow convergence
observed in benchmarks (BnB much weaker than Shor SDP at d=4..8 with
max_depth=4).

============================================================================
USAGE
============================================================================
  python _coarse_BnB_bench.py
"""
import os, sys, time, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, _dir)

from compositions import generate_compositions_batched
from pruning import count_compositions

# Reuse pre-compute and Numba kernels from existing benches
from _coarse_NO_bench import (precompute_op_rest, prune_coarse_NO,
                              prune_coarse_baseline)


# =====================================================================
# Window enumeration and per-window precompute
# =====================================================================

def _enumerate_windows(d):
    """All windows W = (ell, s_lo) with ell in [2, 2d], s_lo such that
    window covers ell-1 conv positions starting at s_lo, s_lo+ell-2 < 2d-1.
    Returns list of (ell, s_lo)."""
    out = []
    conv_len = 2 * d - 1
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        n_windows = conv_len - n_cv + 1
        for s_lo in range(n_windows):
            out.append((ell, s_lo))
    return out


def _build_A(W, d):
    """Indicator A_W[i,j] = 1 iff s_lo <= i+j <= s_lo+ell-2 (ordered pairs)."""
    ell, s_lo = W
    s_hi = s_lo + ell - 2
    A = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            if s_lo <= i + j <= s_hi:
                A[i, j] = 1.0
    return A


def _tv_at(mu, A, d, ell):
    """TV_W(mu) = (2d/ell) * sum_{(i,j) in W} mu_i mu_j."""
    return (2.0 * d / ell) * float(mu @ A @ mu)


# =====================================================================
# Per-axis bound at a leaf cell
# =====================================================================

def _box_lp_max(grad, target_sum, lo_eps, hi_eps, d):
    """Solve  max grad . eps  s.t.  lo_eps <= eps <= hi_eps, Σ eps = target_sum.
    Closed-form O(d log d) via Lagrangian sweep over λ.
    Returns max value, or -inf if infeasible.
    """
    # Feasibility
    s_lo = float(lo_eps.sum())
    s_hi = float(hi_eps.sum())
    if target_sum < s_lo - 1e-12 or target_sum > s_hi + 1e-12:
        return -np.inf
    # Edge case: equality at boundary
    if abs(target_sum - s_lo) < 1e-12:
        return float(np.dot(grad, lo_eps))
    if abs(target_sum - s_hi) < 1e-12:
        return float(np.dot(grad, hi_eps))

    # Dual: minimize over λ of Σ_i max((grad_i - λ) * eps_i for eps_i in [lo_i, hi_i]) + λ * target_sum.
    # = Σ_i max((grad_i - λ) * lo_i, (grad_i - λ) * hi_i) + λ * target_sum
    # = Σ_i (grad_i - λ) * (hi_i if grad_i > λ else lo_i) + λ * target_sum
    # We sweep λ over candidate values: λ in {grad_i}. At each, evaluate primal.
    # The active-set sort: sort grad descending; for each break point, the indices
    # with grad_i > λ contribute hi_i, others lo_i. Sum eps = Σ_{i: g_i > λ} hi_i
    # + Σ_{i: g_i < λ} lo_i + (one "boundary" index that may sit somewhere in [lo, hi]
    # to satisfy Σeps = target).
    sorted_idx = np.argsort(-grad)  # descending grad

    # Compute cumulative: for prefix of size k (indices i with grad >= grad[sorted_idx[k-1]]),
    # eps_i = hi_i; for the rest, eps_i = lo_i. Sum = sum_{k=0..K-1} hi[idx[k]] + sum_{k=K..d-1} lo[idx[k]].
    # This is monotonically nondecreasing in K (since hi >= lo).
    # We want sum = target_sum. Find K* such that cumulative crosses target.
    cum = float(lo_eps.sum())
    # cum at K=0 = sum lo. At K=d = sum hi. We want equality at some K* with
    # one boundary index 0 <= eps_b <= hi_b - lo_b.
    if cum > target_sum + 1e-12:
        return -np.inf
    if abs(cum - target_sum) < 1e-12:
        return float(np.dot(grad, lo_eps))
    K_star = -1
    for k in range(d):
        i = sorted_idx[k]
        cum += hi_eps[i] - lo_eps[i]
        if cum >= target_sum - 1e-12:
            K_star = k
            break
    if K_star < 0:
        return -np.inf
    # K_star is the boundary index. Indices [0..K_star-1] have eps = hi.
    # Index K_star has eps = lo + (target - prev_cum) within [lo, hi].
    # Indices [K_star+1..d-1] have eps = lo.
    eps = lo_eps.copy()
    for k in range(K_star):
        i = sorted_idx[k]
        eps[i] = hi_eps[i]
    # Boundary
    if K_star >= 0:
        cum_before = float(lo_eps.sum())
        for k in range(K_star):
            i = sorted_idx[k]
            cum_before += hi_eps[i] - lo_eps[i]
        i_b = sorted_idx[K_star]
        eps_b = lo_eps[i_b] + (target_sum - cum_before)
        # Clamp to [lo, hi] (numerical robustness)
        eps_b = max(lo_eps[i_b], min(hi_eps[i_b], eps_b))
        eps[i_b] = eps_b
    val = float(np.dot(grad, eps))
    return val


def _lin_corr_lp(grad, target_sum, lo_eps_in, hi_eps_in, d):
    """Linear correction over eps-cell:

      max_{eps} | grad . eps |  s.t.  lo_eps_in <= eps <= hi_eps_in,
                                       Σ eps = target_sum.

    Sound LP, closed-form via _box_lp_max.
    """
    lo_eps = lo_eps_in
    hi_eps = hi_eps_in
    target = float(target_sum)

    if np.any(lo_eps > hi_eps + 1e-12):
        return 0.0

    val_pos = _box_lp_max(grad, target, lo_eps, hi_eps, d)
    val_neg = _box_lp_max(-grad, target, lo_eps, hi_eps, d)
    best = max(0.0, val_pos, val_neg)
    return best


def _quad_corr_box(W_pairs, r_box, d, ell):
    """Quadratic correction in TV-units:
      (2d/ell) * sum_{(i,j) in W_pairs} r_i r_j.

    This sound bound uses |eps^T A_W eps| <= sum_{(i,j) in pairs} |eps_i eps_j|
    <= sum r_i r_j (where pairs = {(i,j) : A_W[i,j]=1}).
    """
    s = 0.0
    for (i, j) in W_pairs:
        s += r_box[i] * r_box[j]
    return (2.0 * d / ell) * s




# =====================================================================
# Sound bound at a leaf cell (returns net = margin - corr per window;
# cell certified iff max_W net_W >= 0).
# =====================================================================

def _cell_bound(c_int, lo, hi, S, d, c_target, A_list, pairs_list, ell_list):
    """Per-axis sound bound at a single cell.

    Returns (max_net, best_W_idx, grad_at_center).
    cell certified iff max_net >= 0.
    """
    h = 1.0 / (2.0 * S)
    mu_star = c_int.astype(np.float64) / S
    c_box = (lo + hi) / 2.0
    r_box = (hi - lo) / 2.0
    mu_center = mu_star + c_box

    if np.any(mu_center < -1e-12):
        # Cell intersects the negative half-space — but we should have ensured
        # mu_i + delta_i >= 0 by clamping lo. Proceed cautiously: this means
        # the cell has shrunk and bound is over the constrained box.
        pass

    best_net = -np.inf
    best_W = -1
    best_grad = None
    n_W = len(A_list)

    # Anchor c' = projection of per-axis center c_box onto {Σ = 0}, then
    # eps = delta - c'. eps bounds depend on (lo, hi) translated by c'.
    # Σ eps = Σ delta - Σ c' = 0 - 0 = 0.
    c_prime = c_box - float(np.mean(c_box))
    mu_anchor = mu_star + c_prime    # this satisfies Σ = 1 (a valid prob. dist
                                     # if mu_anchor >= 0 — may violate box bds).

    # eps-cell bounds: eps_i = delta_i - c'_i, delta_i in [lo_i, hi_i],
    # delta_i + mu*_i >= 0  -> eps_i >= -mu*_i - c'_i.
    lo_eps_box = lo - c_prime  # delta_i >= lo_i means eps_i >= lo_i - c'_i
    hi_eps_box = hi - c_prime
    lo_eps = np.maximum(lo_eps_box, -mu_star - c_prime)
    hi_eps = hi_eps_box.copy()
    target_sum = 0.0   # since Σ c' = 0

    # eps-cell empty? -> cell is empty -> trivially certified.
    if np.any(lo_eps > hi_eps + 1e-12):
        return np.inf, 0, np.zeros(d)
    s_lo = float(lo_eps.sum())
    s_hi = float(hi_eps.sum())
    if target_sum < s_lo - 1e-12 or target_sum > s_hi + 1e-12:
        return np.inf, 0, np.zeros(d)

    # |eps_i| <= max(|lo_eps_i|, |hi_eps_i|) for the entry-wise quad bound.
    M_eps = np.maximum(np.abs(lo_eps), np.abs(hi_eps))

    for w_idx in range(n_W):
        A_W = A_list[w_idx]
        pairs = pairs_list[w_idx]
        ell = ell_list[w_idx]

        # TV at mu_anchor (Σ = 1, may have entries outside [lo, hi] but this
        # only matters for the bound's anchor — it's used algebraically).
        tv_c = (2.0 * d / ell) * float(mu_anchor @ A_W @ mu_anchor)
        margin = tv_c - c_target

        # Algebraic identity:
        #  TV_W(mu* + delta) = tv_c + grad_c . eps + (2d/ell) eps^T A_W eps
        # where grad_c = (4d/ell) A_W mu_anchor.
        grad = (4.0 * d / ell) * (A_W @ mu_anchor)
        # Linear correction: max |grad . eps| over eps-cell.
        lin = _lin_corr_lp(grad, target_sum, lo_eps, hi_eps, d)

        # Quadratic correction: (2d/ell) |eps^T A_W eps|, bounded entry-wise:
        #   |eps^T A_W eps| <= sum_{(i,j) in pairs} M_i M_j.
        quad = (2.0 * d / ell) * sum(M_eps[i] * M_eps[j] for (i, j) in pairs)

        net = margin - lin - quad
        if net > best_net:
            best_net = net
            best_W = w_idx
            best_grad = grad

    return best_net, best_W, best_grad


# =====================================================================
# BnB driver
# =====================================================================

def cell_cert_bnb(c_int, S, d, c_target, max_depth=4,
                   A_list=None, pairs_list=None, ell_list=None,
                   collect_stats=False):
    """Sound BnB cert: returns True iff the cell can be certified at depth
    <= max_depth.

    Cell representation: per-axis [lo[i], hi[i]] for delta = mu - mu_star.
    Initial: lo[i] = max(-h, -mu_star[i]) = max(-1/(2S), -k_i/S),
             hi[i] = +1/(2S).
    (Variant O is automatic via clamping lo.)

    If `collect_stats=True`, also returns depth_required (max depth at any
    cert-leaf, so that cum-cert at depth >= depth_required), leaves count,
    splits count.
    """
    if A_list is None:
        windows = _enumerate_windows(d)
        A_list = [_build_A(W, d) for W in windows]
        pairs_list = [[(i, j) for i in range(d) for j in range(d)
                       if A_list[k][i, j] > 0]
                      for k in range(len(windows))]
        ell_list = [W[0] for W in windows]

    h = 1.0 / (2.0 * S)
    mu_star = c_int.astype(np.float64) / S
    lo0 = np.maximum(-h, -mu_star)
    hi0 = np.full(d, h)

    # Initial bound: depth 0
    net0, _, _ = _cell_bound(c_int, lo0, hi0, S, d, c_target,
                               A_list, pairs_list, ell_list)
    stats = {'leaves': 0, 'splits': 0, 'depth_required': 0}
    if net0 >= 0.0:
        stats['leaves'] = 1
        if collect_stats:
            return True, 0, stats
        return True

    # Stack-based BnB. Each entry: (lo, hi, depth)
    stack = [(lo0, hi0, 0)]
    depth_required = 0

    while stack:
        lo, hi, depth = stack.pop()
        net, best_W, best_grad = _cell_bound(c_int, lo, hi, S, d, c_target,
                                                A_list, pairs_list, ell_list)
        stats['leaves'] += 1
        if net >= 0.0:
            if depth > depth_required:
                depth_required = depth
            continue

        if depth >= max_depth:
            stats['depth_required'] = depth_required
            if collect_stats:
                return False, depth_required, stats
            return False

        # Subdivide along axis with largest contribution to bound failure.
        r_box = (hi - lo) / 2.0
        A_W = A_list[best_W]
        score = np.abs(best_grad) * r_box + (A_W @ r_box) * r_box * 2.0
        score[r_box <= 1e-15] = -1.0
        i_star = int(np.argmax(score))
        if r_box[i_star] <= 1e-15:
            stats['depth_required'] = depth_required
            if collect_stats:
                return False, depth_required, stats
            return False

        mid = (lo[i_star] + hi[i_star]) / 2.0
        stats['splits'] += 1

        lo_2 = lo.copy()
        hi_2 = hi.copy()
        lo_2[i_star] = mid
        lo_1 = lo.copy()
        hi_1 = hi.copy()
        hi_1[i_star] = mid

        # Push both (both must cert for parent to cert).
        stack.append((lo_2, hi_2, depth + 1))
        stack.append((lo_1, hi_1, depth + 1))

    stats['depth_required'] = depth_required
    if collect_stats:
        return True, depth_required, stats
    return True


# =====================================================================
# Bench driver
# =====================================================================

def enum_compositions(d, S):
    if d == 1:
        yield (S,)
        return
    for v in range(S + 1):
        for rest in enum_compositions(d - 1, S - v):
            yield (v,) + rest


def _shor_cert(c_int, S, d, c_target, A_list, ell_list, tol=1e-9, eps_margin=1e-9):
    """Shor SDP feasibility test on the coarse cell.

    Cell: lo = max(0, c-0.5), hi = c+0.5 (in x = S*mu units), Σx = S, x >= 0.
    Window thr: Tr(A_W X) <= c_target * ell * S² / (2d).

    Returns True iff SDP infeasible (cell certified).
    """
    import cvxpy as cp

    c_arr = np.asarray(c_int, dtype=np.float64)
    lo = np.maximum(0.0, c_arr - 0.5)
    hi = c_arr + 0.5

    x = cp.Variable(d)
    X = cp.Variable((d, d), symmetric=True)
    ones11 = np.ones((1, 1))
    Y = cp.bmat([[ones11, cp.reshape(x, (1, d), order='C')],
                  [cp.reshape(x, (d, 1), order='C'), X]])
    cons = [Y >> 0]
    cons += [x >= lo, x <= hi]
    cons += [cp.sum(x) == float(S)]
    for i in range(d):
        cons += [X[i, i] >= lo[i] * lo[i], X[i, i] <= hi[i] * hi[i]]
        cons += [X[i, i] >= 2.0 * lo[i] * x[i] - lo[i] * lo[i]]
        cons += [X[i, i] >= 2.0 * hi[i] * x[i] - hi[i] * hi[i]]
        cons += [X[i, i] <= (lo[i] + hi[i]) * x[i] - lo[i] * hi[i]]
    for i in range(d):
        for j in range(i + 1, d):
            li, lj = lo[i], lo[j]
            ui, uj = hi[i], hi[j]
            cons += [X[i, j] >= lj * x[i] + li * x[j] - li * lj]
            cons += [X[i, j] >= uj * x[i] + ui * x[j] - ui * uj]
            cons += [X[i, j] <= ui * x[j] + lj * x[i] - ui * lj]
            cons += [X[i, j] <= uj * x[i] + li * x[j] - li * uj]
    # Window thr in x-units: x^T A_W x = S^2 * mu^T A_W mu = S^2 * (ell/(2d)) TV_W(mu)
    # TV_W <= c_target ⇒ mu^T A_W mu <= c_target * ell / (2d)
    # ⇒ x^T A_W x <= c_target * ell * S^2 / (2d)
    thr_eps = eps_margin * S * S
    for w_idx, A_W in enumerate(A_list):
        ell = ell_list[w_idx]
        thr = c_target * ell * S * S / (2.0 * d) + thr_eps
        cons += [cp.trace(A_W @ X) <= thr]

    prob = cp.Problem(cp.Minimize(0), cons)
    try:
        prob.solve(solver='MOSEK', verbose=False,
                    mosek_params={
                        'MSK_DPAR_INTPNT_CO_TOL_PFEAS': tol,
                        'MSK_DPAR_INTPNT_CO_TOL_DFEAS': tol,
                        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': tol,
                    })
    except Exception as e:
        return False, f'EXC:{type(e).__name__}'
    return prob.status == 'infeasible', prob.status


def run_one(d, S, c_target, max_depth=4, run_shor=True, max_shor=200,
             max_bnb=400, verbose=True):
    if verbose:
        print(f"\n=== d={d}, S={S}, c={c_target}, max_depth={max_depth} ===",
              flush=True)

    # 1) Enumerate compositions and N+O prune (warm via _coarse_NO_bench).
    op_rest, _ = precompute_op_rest(d, 2 * d)
    op_rest_d = op_rest * d
    batch = np.array(list(enum_compositions(d, S)), dtype=np.int32)
    n_total = len(batch)
    if verbose:
        print(f"  total comps: {n_total:,}", flush=True)

    warm = batch[:1]
    _ = prune_coarse_baseline(warm, d, S, c_target)
    _ = prune_coarse_NO(warm, d, S, c_target, op_rest_d)

    t0 = time.time()
    s_BL = prune_coarse_baseline(batch, d, S, c_target)
    t_BL = time.time() - t0
    t0 = time.time()
    s_NO = prune_coarse_NO(batch, d, S, c_target, op_rest_d)
    t_NO = time.time() - t0
    n_BL = int(s_BL.sum())
    n_NO = int(s_NO.sum())

    if verbose:
        print(f"  BL survivors: {n_BL:,}  ({t_BL:.3f}s)")
        print(f"  NO survivors: {n_NO:,}  ({t_NO:.3f}s)")

    # 2) For each NO-survivor, run BnB at increasing depths.
    survivor_idx = np.where(s_NO)[0]
    n_surv = len(survivor_idx)

    # Cap to manageable size for Shor/BnB comparison.
    if n_surv > max_bnb:
        if verbose:
            print(f"  (capping BnB to first {max_bnb} of {n_surv} NO-survivors)")
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(n_surv, size=max_bnb, replace=False)
        sample_idx = sorted(sample_idx)
        survivor_idx = survivor_idx[sample_idx]
        n_surv = len(survivor_idx)

    # Build window structures once
    windows = _enumerate_windows(d)
    A_list = [_build_A(W, d) for W in windows]
    pairs_list = [[(i, j) for i in range(d) for j in range(d)
                   if A_list[k][i, j] > 0]
                  for k in range(len(windows))]
    ell_list = [W[0] for W in windows]
    if verbose:
        print(f"  n_windows = {len(windows)}; running BnB on {n_surv} survivors")

    # For each cell, run BnB at max_depth ONCE; record min depth required to cert
    # (or -1 if uncert).  Cumulative cert at depth dd = #cells with required_depth <= dd.
    cert_required_depth = []   # list of min depth at which cert; -1 if uncert.
    cell_times = []
    cell_leaves = []
    cell_splits = []
    bnb_uncert = []
    cert_results = []   # (ci, depth_required) for shor comparison

    t0_total = time.time()
    for surv_i, ci in enumerate(survivor_idx):
        c_int = batch[ci]
        t1 = time.time()
        ok, depth_req, stats = cell_cert_bnb(c_int, S, d, c_target,
                                                max_depth=max_depth,
                                                A_list=A_list,
                                                pairs_list=pairs_list,
                                                ell_list=ell_list,
                                                collect_stats=True)
        dt = time.time() - t1
        cell_times.append(dt)
        cell_leaves.append(stats['leaves'])
        cell_splits.append(stats['splits'])
        if ok:
            cert_required_depth.append(depth_req)
            cert_results.append((int(ci), int(depth_req)))
        else:
            cert_required_depth.append(-1)
            bnb_uncert.append(int(ci))
            cert_results.append((int(ci), -1))
    t_bnb_total = time.time() - t0_total

    cert_required_arr = np.array(cert_required_depth)
    cell_times_arr = np.array(cell_times)
    cum_cert = {}
    for dd in range(max_depth + 1):
        cum_cert[dd] = int(np.sum((cert_required_arr >= 0) & (cert_required_arr <= dd)))
    cert_rate = {dd: cum_cert[dd] / max(1, n_surv) for dd in range(max_depth + 1)}

    # Per-depth median time (over all cells, since each cell uses the same call)
    time_total_per_depth = float(cell_times_arr.sum())
    time_med = float(np.median(cell_times_arr)) if len(cell_times_arr) else 0.0
    time_avg = float(np.mean(cell_times_arr)) if len(cell_times_arr) else 0.0

    if verbose:
        print(f"  BnB: cert rate at depth (cumulative):")
        for dd in range(max_depth + 1):
            print(f"     depth<={dd}: {cum_cert[dd]:>5}/{n_surv} "
                  f"({100.0*cert_rate[dd]:.1f}%)")
        print(f"  BnB per-cell time: avg {1000*time_avg:.1f}ms, "
              f"median {1000*time_med:.1f}ms; "
              f"avg leaves={float(np.mean(cell_leaves)):.1f}, "
              f"splits={float(np.mean(cell_splits)):.1f}")
        print(f"  BnB total elapsed: {t_bnb_total:.2f}s; "
              f"final uncert: {len(bnb_uncert)}")

    # 3) Shor SDP comparison on a sample.
    shor_results = None
    if run_shor and n_surv > 0:
        n_shor = min(max_shor, n_surv)
        shor_idx = survivor_idx[:n_shor]
        if verbose:
            print(f"  Running Shor SDP on first {n_shor} survivors...",
                  flush=True)
        t_shor_total = 0.0
        n_shor_pruned = 0
        n_shor_unsolved = 0
        shor_outcomes = []   # (idx, pruned, status, t)
        bnb_outcomes_for_shor = {}   # idx -> certified depth
        for r in cert_results[:n_shor]:
            bnb_outcomes_for_shor[r[0]] = r[1]
        for ci in shor_idx:
            c_int = batch[ci]
            t1 = time.time()
            try:
                pruned, status = _shor_cert(c_int, S, d, c_target, A_list,
                                              ell_list)
            except Exception as e:
                pruned = False
                status = f'EXC:{type(e).__name__}'
            dt = time.time() - t1
            t_shor_total += dt
            if pruned:
                n_shor_pruned += 1
            if status not in ('infeasible', 'optimal'):
                n_shor_unsolved += 1
            shor_outcomes.append((int(ci), bool(pruned), str(status), float(dt)))

        # Compare Shor vs BnB:
        n_both = 0
        n_shor_only = 0
        n_bnb_only = 0
        n_neither = 0
        for ci, pr_shor, status, _ in shor_outcomes:
            bnb_cert = bnb_outcomes_for_shor.get(ci, -1) >= 0
            if pr_shor and bnb_cert:
                n_both += 1
            elif pr_shor and not bnb_cert:
                n_shor_only += 1
            elif not pr_shor and bnb_cert:
                n_bnb_only += 1
            else:
                n_neither += 1

        if verbose:
            print(f"  Shor: {n_shor_pruned}/{n_shor} cert "
                  f"({100.0*n_shor_pruned/n_shor:.1f}%); "
                  f"avg {1000*t_shor_total/n_shor:.0f}ms/cell; total {t_shor_total:.1f}s")
            print(f"  Comparison (Shor vs BnB up to depth {max_depth}):")
            print(f"    Both cert: {n_both}, Shor-only: {n_shor_only}, "
                  f"BnB-only: {n_bnb_only}, neither: {n_neither}")

        shor_results = {
            'n_shor': n_shor,
            'n_shor_cert': n_shor_pruned,
            'n_shor_unsolved': n_shor_unsolved,
            'time_total': t_shor_total,
            'time_avg': t_shor_total / max(1, n_shor),
            'compare_both': n_both,
            'compare_shor_only': n_shor_only,
            'compare_bnb_only': n_bnb_only,
            'compare_neither': n_neither,
        }

    return {
        'd': d, 'S': S, 'c_target': c_target, 'max_depth': max_depth,
        'n_total': n_total, 'n_BL': n_BL, 'n_NO': n_NO,
        'n_bnb_input': n_surv,
        'cum_cert': cum_cert,
        'cert_rate': cert_rate,
        'time_avg': time_avg,
        'time_med': time_med,
        'time_total': time_total_per_depth,
        'time_bnb_total': t_bnb_total,
        'avg_leaves': float(np.mean(cell_leaves)) if cell_leaves else 0.0,
        'avg_splits': float(np.mean(cell_splits)) if cell_splits else 0.0,
        'n_bnb_uncert': len(bnb_uncert),
        'shor': shor_results,
    }


def main():
    configs = [
        (4, 20, 1.20),
        (6, 15, 1.20),
        (8, 12, 1.20),
    ]
    results = []
    for d, S, c in configs:
        # Cap Shor to ≤100, BnB to ≤200 for tractable wall-clock.
        max_shor = 100 if d <= 6 else 50
        max_bnb = 300 if d <= 6 else 150
        r = run_one(d, S, c, max_depth=4, run_shor=True,
                    max_shor=max_shor, max_bnb=max_bnb, verbose=True)
        results.append(r)

    print("\n" + "=" * 70 + "\nSUMMARY:")
    for r in results:
        n = r['n_bnb_input']
        cum = r['cum_cert']
        line = (f"  d={r['d']}, S={r['S']}: NO-survivors={n}; "
                f"BnB cum cert at depth ")
        for dd in range(r['max_depth'] + 1):
            pct = 100.0 * cum[dd] / max(1, n)
            line += f"{dd}: {cum[dd]} ({pct:.0f}%); "
        print(line)
        if r['shor'] is not None:
            s = r['shor']
            print(f"    Shor: {s['n_shor_cert']}/{s['n_shor']} "
                  f"(avg {1000*s['time_avg']:.0f}ms); "
                  f"both={s['compare_both']}, Shor-only={s['compare_shor_only']}, "
                  f"BnB-only={s['compare_bnb_only']}, neither={s['compare_neither']}")

    out = os.path.join(_dir, '_coarse_BnB_bench_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=lambda o: int(o)
                  if isinstance(o, np.integer) else float(o))
    print(f"\nResults written to {out}")


if __name__ == '__main__':
    main()
