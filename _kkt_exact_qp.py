"""KKT-augmented exhaustive critical-point enumeration for the per-window QP.

Problem
-------
For window W = (ell, s_lo) with grad and A_W := A_W^T (symmetric, 0/1) and
scale := 2d/ell, h := 1/(2S), find

    f*(W) = max_{delta in Cell} f(delta),
            f(delta) := -grad . delta - scale * delta^T A_W delta,
    Cell  = { delta in R^d : |delta_i| <= h, sum_i delta_i = 0 }.

The Hessian of f is H = -2*scale*A_W, which is INDEFINITE for typical W.
=> vertex enumeration is INSUFFICIENT for an indefinite quadratic. (Concave
QPs admit a vertex maximum, but our objective is generic indefinite.)

Mathematical fact (a standard KKT/Karush–John consequence): every local
maximum of a quadratic over a polytope is a critical point of f restricted
to *some* face F of the polytope. Hence

    f*(W) = max over all faces F of (max of f restricted to F that lies in F).

A face of Cell is determined by:
  - a subset I_+ of indices fixed at +h,
  - a disjoint subset I_- of indices fixed at -h,
  - the remaining "free" indices J = [d] \ (I_+ U I_-),
  - the affine constraint sum_J delta_j = -h*(|I_+|-|I_-|),
  - the box-interior constraint -h < delta_j < h for all j in J
    (or equality at the boundary, but those collapse onto a smaller face).

For a 0-dim face (|J|=1 or |J|=0 with feasible sum) we get a vertex.
For a face of free-dim k = |J| - 1 (one constraint removed by sum=0),
the restricted quadratic has Hessian H_JJ = -2*scale*A_W^{JJ}, projected
onto the affine subspace {1^T delta_J = -fixed_sum}.

Lagrangian: L(delta_J, mu) = -grad_J . delta_J - scale * delta_J^T A_W^{JJ} delta_J
                            + mu * (1^T delta_J + fixed_sum).
KKT: H_JJ delta_J = -grad_J + mu * 1,   1^T delta_J = -fixed_sum.

Stack as a (k+1) x (k+1) linear system

   [ H_JJ   1 ] [ delta_J ]   [ -grad_J     ]
   [ 1^T    0 ] [   mu    ] = [ -fixed_sum  ]

Solve (numpy.linalg.solve). The critical point is feasible (interior of box)
iff |delta_J,j| <= h for all j; if so, evaluate f and include in the running
max. (If on the boundary, it is on a smaller face which we will also visit.)

Total face count: sum_{|I|=0..d} C(d, |I|) * 2^{|I|} = 3^d. For d=8: 6561.

Soundness
---------
This enumeration is exhaustive over all polytope faces; the global max of f
on Cell is attained on some face's interior (or boundary, which collapses
to a sub-face). Hence f*(W) is the supremum over the candidate values,
which we take as the bound. Vertex enum is the |J|=1 sub-case, so this
is provably tighter than vertex enum and SOUND (no additional rounding).
"""
from __future__ import annotations

import itertools
import numpy as np


# ---------------------------------------------------------------------------
# Single face KKT solve
# ---------------------------------------------------------------------------

def _face_critical_points(grad, A_W, scale, h, d, fixed_idxs, fixed_signs):
    """Solve KKT system for one face.

    Parameters
    ----------
    grad, A_W, scale, h, d : as in qp_bound.qp_bound_vertex.
    fixed_idxs   : tuple of indices fixed at +/-h.
    fixed_signs  : tuple of signs in {-1, +1}; fixed[i] = signs[i]*h.

    Returns
    -------
    (feasible, val) where:
      - feasible = True iff the critical point lies in [-h, h]^k_free.
      - val = f(delta) at the critical point (if feasible) else -inf.
    """
    fixed = set(fixed_idxs)
    free_idxs = np.array([i for i in range(d) if i not in fixed], dtype=np.int64)
    k = len(free_idxs)

    # Vertex face: |J| <= 1. Handle separately (no Lagrangian needed).
    if k == 0:
        # All coordinates fixed. delta determined entirely.
        # Feasible iff sum(fixed_signs)*h == 0 ⟹ #(+) == #(-). Then f computed.
        s = sum(fixed_signs)
        if s != 0:
            return False, -np.inf
        delta = np.zeros(d)
        for idx, sg in zip(fixed_idxs, fixed_signs):
            delta[idx] = sg * h
        val = -grad @ delta - scale * delta @ A_W @ delta
        return True, float(val)
    if k == 1:
        # One free coord, determined by sum=0.
        free = int(free_idxs[0])
        fixed_sum = sum(fixed_signs) * h
        free_val = -fixed_sum
        if abs(free_val) > h + 1e-12:
            return False, -np.inf
        # Clamp tiny FP drift
        free_val = max(-h, min(h, free_val))
        delta = np.zeros(d)
        delta[free] = free_val
        for idx, sg in zip(fixed_idxs, fixed_signs):
            delta[idx] = sg * h
        val = -grad @ delta - scale * delta @ A_W @ delta
        return True, float(val)

    # General case: k >= 2 free coords with sum(free) = -fixed_sum.
    fixed_sum_h = sum(fixed_signs) * h  # = sum_{i in I_+ U I_-} delta_i
    g_J = grad[free_idxs]
    H_JJ = -2.0 * scale * A_W[np.ix_(free_idxs, free_idxs)]

    # Stack (k+1)x(k+1) KKT system. Note: we are MAXIMIZING f, so we want
    #     ∇_J f = 0,  i.e.  -grad_J - 2*scale*A_W^{JJ}*delta_J + mu*1 = 0,
    # plus 1^T delta_J = -fixed_sum_h.
    # Equivalently:  H_JJ delta_J - mu * 1 = grad_J  (writing -grad_J = -grad_J
    # and bringing mu*1 to RHS as +mu*1 means rearranging signs carefully).
    #
    # Original objective f. Setting partials w.r.t. delta_J:
    #   ∂f/∂delta_J = -grad_J - 2*scale*A_W^{JJ} delta_J = -grad_J + (1/2)*H_JJ * delta_J ... wait
    # Let H_JJ_obj := H_JJ = -2*scale*A_W^{JJ}, the Hessian of f w.r.t. delta_J.
    # Then ∇_J f = -grad_J + H_JJ * delta_J. (Check: f = -g.d - s d^T A d,
    # ∂/∂d_i = -g_i - 2s (Ad)_i = -g_i + (H d)_i with H = -2s A.) ✓
    # Stationary: -grad_J + H_JJ delta_J + mu * 1 = 0  with Lagrangian
    #   L = f + mu*(1^T delta_J + fixed_sum_h), ∇_J L = ∇_J f + mu*1 = 0.
    # i.e.  H_JJ delta_J + mu * 1 = grad_J.
    # Constraint: 1^T delta_J = -fixed_sum_h.
    #
    # KKT system:  [ H_JJ   1 ] [ delta_J ]   [ grad_J        ]
    #              [ 1^T    0 ] [   mu    ] = [ -fixed_sum_h  ]
    M = np.zeros((k + 1, k + 1), dtype=np.float64)
    M[:k, :k] = H_JJ
    M[:k, k] = 1.0
    M[k, :k] = 1.0
    rhs = np.zeros(k + 1, dtype=np.float64)
    rhs[:k] = g_J
    rhs[k] = -fixed_sum_h

    try:
        sol = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        # Singular ⇒ infinitely many critical points (e.g., flat direction).
        # Vertices on this face's boundary will catch any maximizers; skip.
        return False, -np.inf
    delta_J = sol[:k]
    # Feasibility: |delta_J| <= h. Use tight tolerance — boundary cases will be
    # caught by sub-faces.
    tol = 1e-12 + 1e-10 * h
    if np.any(np.abs(delta_J) > h + tol):
        return False, -np.inf
    delta_J = np.clip(delta_J, -h, h)

    # Build full delta vector and evaluate f.
    delta = np.zeros(d)
    delta[free_idxs] = delta_J
    for idx, sg in zip(fixed_idxs, fixed_signs):
        delta[idx] = sg * h
    val = -grad @ delta - scale * delta @ A_W @ delta
    return True, float(val)


# ---------------------------------------------------------------------------
# Top-level: enumerate all faces
# ---------------------------------------------------------------------------

def qp_bound_kkt(grad, A_W, scale, h, d):
    """Provably exact max of f over Cell via KKT face enumeration.

    Returns max(f(0)=0, max over all face critical points).

    Total work: 3^d face solves; each is O(d^3) linear system.
    For d=8: 6561 * 8^3 = 3.4M ops. d=12: 530K * 12^3 = 0.9G ops.
    """
    grad = np.ascontiguousarray(grad, dtype=np.float64)
    A_W = np.ascontiguousarray(A_W, dtype=np.float64)
    best = 0.0  # f(0) = 0 is always feasible.

    # Iterate over all subsets I = I_+ U I_-, by choosing |I|=t.
    indices = list(range(d))
    for t in range(0, d + 1):
        for fixed_idxs in itertools.combinations(indices, t):
            for sign_mask in range(1 << t):
                fixed_signs = tuple(
                    +1 if (sign_mask >> j) & 1 else -1 for j in range(t)
                )
                feas, val = _face_critical_points(
                    grad, A_W, scale, h, d, fixed_idxs, fixed_signs
                )
                if feas and val > best:
                    best = val
    return best


# ---------------------------------------------------------------------------
# Convenience wrapper using qp_bound infrastructure
# ---------------------------------------------------------------------------

def qp_bound_kkt_for_composition(c_int, S, d, ell, s_lo):
    """Compute KKT-exact bound for composition c_int, window (ell, s_lo)."""
    import os, sys
    _dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
    from qp_bound import build_window_matrix, grad_for_window
    A_W = build_window_matrix(d, ell, s_lo)
    grad = grad_for_window(np.asarray(c_int, dtype=np.float64), A_W, S, d, ell)
    h = 1.0 / (2.0 * S)
    scale = 2.0 * d / ell
    return qp_bound_kkt(grad, A_W, scale, h, d)


if __name__ == '__main__':
    # Tiny self-test: d=3, deterministic.
    import os, sys
    _dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
    from qp_bound import build_window_matrix, grad_for_window, qp_bound_vertex

    np.random.seed(0)
    d = 3
    S = 10
    c = np.array([3, 4, 3], dtype=np.float64)
    ell = 3
    s_lo = 1
    A_W = build_window_matrix(d, ell, s_lo)
    grad = grad_for_window(c, A_W, S, d, ell)
    h = 1.0 / (2.0 * S)
    scale = 2.0 * d / ell
    v = qp_bound_vertex(grad, A_W, scale, h, d)
    k = qp_bound_kkt(grad, A_W, scale, h, d)
    print(f"d={d}, S={S}: vertex={v:.6e}, KKT={k:.6e}, KKT >= vertex: {k >= v - 1e-12}")
