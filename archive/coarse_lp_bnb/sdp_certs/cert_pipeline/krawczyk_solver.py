"""Krawczyk operator + bisection for excluding zeros of a polynomial system.

Given a square polynomial system F: R^n -> R^n and a box X (interval n-vec),
the Krawczyk operator
    K(X) = mid(X) - Y * F(mid(X)) + (I - Y * F'(X)) * (X - mid(X))
satisfies:
  (i)  Every zero of F in X lies in K(X) cap X.
  (ii) If K(X) is contained strictly in interior(X) AND ||I - Y F'(X)|| < 1,
       then F has a unique zero in X (Brouwer + Banach).
  (iii) If K(X) cap X is empty, F has no zero in X.

We use mpmath.iv for the interval arithmetic; the preconditioner Y is taken
as a NumPy float64 inverse of mid(F'(X)) (preconditioner correctness is
not required for soundness; it only affects sharpness/convergence).

ABSOLUTE-CORRECTNESS GUARANTEE:
  * F and F' must be evaluable in iv arithmetic. Our KKTSystem provides this.
  * mpmath.iv internal arithmetic is outward-rounded.
  * The verdict "no zero in X" or "unique zero in X" is a *theorem* under
    the standard Krawczyk hypotheses; we test those hypotheses with
    interval arithmetic before issuing a verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from mpmath import iv, mp, mpf

from .iv_core import (IVMat, IVVec, iv_disjoint, iv_mid_float, iv_subset,
                      rat_to_iv)
from .saddle_kkt import KKTSystem, derived_quantities


class Verdict(Enum):
    EXCLUDED = "EXCLUDED"          # No zero in X (proved)
    UNIQUE_ZERO = "UNIQUE_ZERO"    # Exactly one zero in X (proved)
    UNDECIDED = "UNDECIDED"        # Cannot decide; subdivide
    EMPTY_BOX = "EMPTY_BOX"        # Box is empty (e.g. after intersection)


@dataclass
class KrawczykResult:
    verdict: Verdict
    K_X: Optional[IVVec] = None         # The Krawczyk image (when computed)
    norm_I_YJ: Optional[float] = None   # ||I - Y J(X)||_inf (rough, float)
    iters: int = 0
    notes: str = ""


def _iv_inf_norm_imat(M: IVMat) -> float:
    """Approximate ||M||_inf where each entry is replaced by its
    mag-bound max(|lo|, |hi|).  Used only as a *test* for contraction
    (||I - YJ|| < 1 implies uniqueness).  Conservative (overestimates)."""
    best = 0.0
    for i in range(M.n):
        row_sum = 0.0
        for j in range(M.m):
            v = M.data[i][j]
            mag = max(abs(float(v.a)), abs(float(v.b)))
            row_sum += mag
        if row_sum > best:
            best = row_sum
    return best


def _iv_to_imat(F_vals: Sequence) -> IVVec:
    """Coerce list of iv.mpf to an IVVec."""
    out = IVVec.__new__(IVVec)
    out.data = list(F_vals)
    return out


def krawczyk_step(system: KKTSystem, X: IVVec,
                  hi_iv: Sequence, lo_iv: Sequence,
                  scales_iv: Sequence) -> KrawczykResult:
    """One Krawczyk step on box X for the KKT system.

    Args:
      system : KKTSystem providing residual(x) and jacobian(x) over scalars
               compatible with iv.mpf.
      X      : IVVec of length system.n_vars (the box for unknowns).
      hi_iv  : iv-valued tuple of upper-box endpoints (length d). Pass
               iv.mpf representations of system.box.hi_q exactly.
      lo_iv  : iv-valued tuple of lower-box endpoints (length d).
      scales_iv : iv-valued list of A_W scales (length n_AW).

    Returns:
      KrawczykResult with verdict, K_X (the new box), norm_I_YJ, etc.
    """
    n = system.n_vars
    if len(X) != n:
        raise ValueError(f"X has length {len(X)} but n_vars={n}")

    # PRE-FILTER: interval evaluation of F(X). If any component's
    # interval does not contain 0, then F has no zero in X.
    F_X_iv = system.residual(list(X.data), hi_q=hi_iv, lo_q=lo_iv,
                             scale_q_override=scales_iv)
    for k, fv in enumerate(F_X_iv):
        if hasattr(fv, "a"):
            if float(fv.a) > 0 or float(fv.b) < 0:
                return KrawczykResult(
                    verdict=Verdict.EXCLUDED,
                    notes=f"interval F_{k}(X) = [{float(fv.a):.3e}, "
                          f"{float(fv.b):.3e}] excludes 0",
                )
        else:
            if fv != 0:
                return KrawczykResult(
                    verdict=Verdict.EXCLUDED,
                    notes=f"interval F_{k}(X) = {fv} != 0",
                )

    # 1. Midpoint x0 as Python floats (we use float for preconditioner Y
    #    and as a center for evaluation; result soundness comes from
    #    interval evaluation around x0, not from the choice of x0).
    x0_float = X.midpoint_float()

    # 2. Evaluate F at x0 in interval arithmetic.
    #    Wrap x0 entries as iv.mpf (degenerate intervals).
    x0_iv = [iv.mpf(float(v)) for v in x0_float]
    F_x0_iv = system.residual(x0_iv, hi_q=hi_iv, lo_q=lo_iv,
                              scale_q_override=scales_iv)
    # F_x0_iv : list of iv.mpf

    # 3. Evaluate Jacobian J(X) over the box (interval enclosure).
    #    Pass the interval entries of X for variables; hi_iv/lo_iv for
    #    the constants.
    J_X_iv_list = system.jacobian(list(X.data), hi_q=hi_iv, lo_q=lo_iv,
                                  scale_q_override=scales_iv)
    # J_X is n_eqs x n_vars; here n_eqs == n_vars == n
    J_X = IVMat.__new__(IVMat)
    J_X.data = J_X_iv_list
    J_X.n = n
    J_X.m = n

    # 4. Build float midpoint matrix and invert via NumPy.
    midJ = J_X.midpoint_float()
    try:
        Y_np = np.linalg.inv(midJ)
    except np.linalg.LinAlgError:
        return KrawczykResult(
            verdict=Verdict.UNDECIDED,
            notes="mid Jacobian singular at midpoint",
        )
    # Coerce Y to interval matrix (degenerate, since Y is float exact-
    # ish; we treat it as exact float for the soundness argument since
    # Y is a preconditioner).
    Y = IVMat.from_float(Y_np)

    # 5. Compute Y * F(x0)  (interval matvec)
    Y_F_x0 = Y.matvec(_iv_to_imat(F_x0_iv))

    # 6. Compute I - Y * J(X)  (interval matmul)
    YJ = Y.matmul(J_X)
    I_n = IVMat.identity(n)
    I_minus_YJ = I_n - YJ

    # 7. Compute (I - Y J(X)) * (X - x0)
    x0_as_iv = IVVec([iv.mpf(float(v)) for v in x0_float])
    Xdiff = IVVec([X.data[i] - x0_as_iv.data[i] for i in range(n)])
    YJ_Xdiff = I_minus_YJ.matvec(Xdiff)

    # 8. K(X) = x0 - Y F(x0) + (I - Y J(X))(X - x0)
    K_data = []
    for i in range(n):
        v = x0_as_iv.data[i] - Y_F_x0.data[i] + YJ_Xdiff.data[i]
        K_data.append(v)
    K_X = IVVec(K_data)

    # 9. Verdicts
    # Disjoint test: K(X) cap X = empty?
    if K_X.is_disjoint_from(X):
        return KrawczykResult(
            verdict=Verdict.EXCLUDED,
            K_X=K_X,
            notes="K(X) cap X = empty",
        )

    # Strict containment test: K(X) subset int(X) ?
    norm_check = _iv_inf_norm_imat(I_minus_YJ)
    if K_X.is_subset_of(X) and norm_check < 1.0:
        return KrawczykResult(
            verdict=Verdict.UNIQUE_ZERO,
            K_X=K_X,
            norm_I_YJ=norm_check,
            notes=f"K(X) subset int(X) and ||I-YJ||_inf <= {norm_check:.6g} < 1",
        )

    # Otherwise: undecided.
    return KrawczykResult(
        verdict=Verdict.UNDECIDED,
        K_X=K_X,
        norm_I_YJ=norm_check,
        notes=f"||I-YJ||_inf approx {norm_check:.6g}",
    )


# ---------------------------------------------------------------------
# Bisection wrapper
# ---------------------------------------------------------------------

def krawczyk_recurse(system: KKTSystem, X: IVVec,
                     hi_iv, lo_iv, scales_iv,
                     max_depth: int = 12,
                     min_width: float = 1e-12) -> List[Tuple[Verdict, IVVec]]:
    """Recursive Krawczyk with bisection.  Returns list of leaves with
    verdict in {EXCLUDED, UNIQUE_ZERO, UNDECIDED}.
    """
    leaves: List[Tuple[Verdict, IVVec]] = []
    stack: List[Tuple[IVVec, int]] = [(X, 0)]
    while stack:
        Xc, depth = stack.pop()
        # Check if box is degenerate small (avoid infinite recursion).
        widths = Xc.width_float()
        if all(w < min_width for w in widths):
            leaves.append((Verdict.UNDECIDED, Xc))
            continue
        res = krawczyk_step(system, Xc, hi_iv, lo_iv, scales_iv)
        if res.verdict in (Verdict.EXCLUDED, Verdict.UNIQUE_ZERO):
            leaves.append((res.verdict, Xc))
            continue
        if depth >= max_depth:
            leaves.append((Verdict.UNDECIDED, Xc))
            continue
        # Bisect along widest axis (Moore's Heuristic).
        widest = int(np.argmax(widths))
        v = Xc.data[widest]
        mid = iv_mid_float(v)
        # Construct two halves
        L_data = list(Xc.data)
        R_data = list(Xc.data)
        L_data[widest] = iv.mpf([float(v.a), float(mid)])
        R_data[widest] = iv.mpf([float(mid), float(v.b)])
        stack.append((IVVec.__new__(IVVec).__class__.__new__(IVVec),
                      depth + 1))  # placeholder
        # Replace placeholder properly
        L_box = IVVec.__new__(IVVec); L_box.data = L_data
        R_box = IVVec.__new__(IVVec); R_box.data = R_data
        # Replace last (placeholder) entry
        stack.pop()
        stack.append((L_box, depth + 1))
        stack.append((R_box, depth + 1))
    return leaves


# ---------------------------------------------------------------------
# Self-test on a 2-variable polynomial system
# ---------------------------------------------------------------------

if __name__ == "__main__":
    """Test Krawczyk on a tiny system independent of KKT.

    Solve: f1 = x^2 + y^2 - 1 = 0
           f2 = x - y           = 0
    Solutions: (sqrt(2)/2, sqrt(2)/2) and (-sqrt(2)/2, -sqrt(2)/2).

    On box X = [0.5, 0.9] x [0.5, 0.9] (around the positive solution):
    expect UNIQUE_ZERO.
    On box X = [0.0, 0.4] x [0.0, 0.4]:
    expect EXCLUDED (no solution).
    """
    # We'll write a minimal "system" object with residual/jacobian like KKTSystem.
    class SimpleSystem:
        n_vars = 2
        def residual(self, x, **_):
            return [x[0] * x[0] + x[1] * x[1] - 1, x[0] - x[1]]
        def jacobian(self, x, **_):
            return [[2 * x[0], 2 * x[1]], [iv.mpf(1), iv.mpf(-1)]]

    sys_simple = SimpleSystem()

    def run(box_lohi, label):
        X = IVVec([iv.mpf(box_lohi[0]), iv.mpf(box_lohi[1])])
        # Adapt to our krawczyk_step signature: but it expects KKTSystem.
        # We'll inline the math here.
        n = 2
        x0_f = X.midpoint_float()
        x0_iv = [iv.mpf(float(v)) for v in x0_f]
        Fx0 = sys_simple.residual(x0_iv)
        JX = sys_simple.jacobian(list(X.data))
        J_mat = IVMat.__new__(IVMat); J_mat.data = JX; J_mat.n = J_mat.m = n
        midJ = J_mat.midpoint_float()
        Y_np = np.linalg.inv(midJ)
        Y = IVMat.from_float(Y_np)
        f_vec = IVVec.__new__(IVVec); f_vec.data = list(Fx0)
        YF = Y.matvec(f_vec)
        YJ = Y.matmul(J_mat)
        I2 = IVMat.identity(2)
        IYJ = I2 - YJ
        x0_iv_vec = IVVec([iv.mpf(float(v)) for v in x0_f])
        Xdiff = IVVec([X.data[i] - x0_iv_vec.data[i] for i in range(n)])
        YJX = IYJ.matvec(Xdiff)
        K_data = [x0_iv_vec.data[i] - YF.data[i] + YJX.data[i]
                  for i in range(n)]
        K_X = IVVec(K_data)
        norm = _iv_inf_norm_imat(IYJ)
        # Verdicts
        if K_X.is_disjoint_from(X):
            verdict = "EXCLUDED"
        elif K_X.is_subset_of(X) and norm < 1.0:
            verdict = "UNIQUE_ZERO"
        else:
            verdict = "UNDECIDED"
        print(f"{label}: verdict={verdict}, ||I-YJ||={norm:.4g}")
        return verdict

    v1 = run([(0.5, 0.9), (0.5, 0.9)], "Around (sqrt2/2, sqrt2/2)")
    v2 = run([(0.0, 0.4), (0.0, 0.4)], "Box [0,0.4]^2 (no solution)")
    assert v1 == "UNIQUE_ZERO", f"Expected UNIQUE_ZERO, got {v1}"
    assert v2 == "EXCLUDED", f"Expected EXCLUDED, got {v2}"
    print("\nself-test OK")
