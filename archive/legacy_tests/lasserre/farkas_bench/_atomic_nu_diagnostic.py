"""Diagnostic: compare three SDP relaxations at the same (d, order):

  (A) L^scalar_k(d) = min t  s.t.  t >= f_W(y) for all W, y in K_k        (LP)
  (B) atomic-nu scalar with lam from joint_bisect                          (what I built)
  (C) L^strong_k(d) = joint_bisect (min-max with matrix-valued window PSD)

By minimax on the linear game max_lam min_y Sum lam_W f_W(y), (A) is the
true upper envelope of (B) over lam.  If (A) = (C), the matrix-PSD
constraints are NOT strictly stronger than their (0,0) scalar versions,
and my 'scalar is weak' claim is wrong.  If (A) < (C), the claim holds.

Also shows whether (B) at joint_bisect's lam matches (A) -- if it does,
my lam choice is optimal; if not, CG on scalar atomic-nu can still
improve.
"""
from __future__ import annotations
import os, sys
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lasserre.precompute import _precompute
from certified_lasserre.atomic_nu_sdp import (
    solve_atomic_nu_sdp, seed_from_joint_bisect, list_windows,
)


def solve_scalar_minmax(d: int, order: int, verbose: bool = False) -> dict:
    """Direct LP: min t s.t. t >= f_W(y) for every W, y in K_k.

    f_W(y) = Sum_{i,j} M_W[i,j] * y_{e_i+e_j}.

    Uses the exact same K_k (base PSD + consistency) as build_sdp_data,
    so any gap to atomic-nu is purely due to lam choice, and any gap to
    joint_bisect is purely due to scalar-vs-matrix PSD constraints.
    """
    import mosek
    from mosek.fusion import Model, Domain, Expr, Matrix, ObjectiveSense, SolutionStatus

    P = _precompute(d, order, verbose=False)
    n_y = P['n_y']
    n_basis = P['n_basis']
    n_loc = P['n_loc']
    n_win = P['n_win']
    idx = P['idx']
    moment_pick = P['moment_pick']
    loc_picks = P['loc_picks']
    M_mats = P['M_mats']

    with Model('scalar_minmax') as M:
        y = M.variable('y', n_y, Domain.unbounded())
        t = M.variable('t', 1, Domain.unbounded())

        # y_0 = 1
        zero = tuple(0 for _ in range(d))
        M.constraint('y0', y.index(idx[zero]), Domain.equalsTo(1.0))

        # Consistency
        c_r, c_c, c_v = [], [], []
        n_c = 0
        for r in range(len(P['consist_mono'])):
            ai = int(P['consist_idx'][r])
            if ai < 0:
                continue
            child = P['consist_ei_idx'][r]
            has = False
            for ci in range(d):
                ch = int(child[ci])
                if ch >= 0:
                    c_r.append(n_c); c_c.append(ch); c_v.append(1.0)
                    has = True
            if not has:
                continue
            c_r.append(n_c); c_c.append(ai); c_v.append(-1.0)
            n_c += 1
        if n_c > 0:
            Ac = Matrix.sparse(n_c, n_y, c_r, c_c, c_v)
            M.constraint('cons', Expr.mul(Ac, y), Domain.equalsTo([0.0]*n_c))

        # Moment PSD
        mom = Expr.reshape(y.pick(moment_pick), n_basis, n_basis)
        M.constraint('mom', mom, Domain.inPSDCone(n_basis))

        # mu_i localizing PSD
        if order >= 2:
            for i_var in range(d):
                lm = Expr.reshape(y.pick(list(loc_picks[i_var])), n_loc, n_loc)
                M.constraint(f'loc_{i_var}', lm, Domain.inPSDCone(n_loc))

        # Scalar constraints: t >= f_W(y) for every W
        # f_W(y) = Sum_{i,j: M_W[i,j] != 0} M_W[i,j] * y_{e_i+e_j}
        idx_ij = P['idx_ij']  # d x d matrix of moment indices
        w_cons = []
        for w in range(n_win):
            Mw = M_mats[w]
            nz_i, nz_j = np.nonzero(Mw)
            if len(nz_i) == 0:
                # Empty window: t >= 0
                c = M.constraint(f'w_{w}', t, Domain.greaterThan(0.0))
                w_cons.append(c); continue
            mi = idx_ij[nz_i, nz_j]
            valid = mi >= 0
            if not np.any(valid):
                c = M.constraint(f'w_{w}', t, Domain.greaterThan(0.0))
                w_cons.append(c); continue
            cols = mi[valid].astype(int).tolist()
            vals = Mw[nz_i[valid], nz_j[valid]].tolist()
            # t - sum vals * y[cols] >= 0
            fW_expr = Expr.dot(vals, y.pick(cols))
            c = M.constraint(f'w_{w}', Expr.sub(t, fW_expr), Domain.greaterThan(0.0))
            w_cons.append(c)

        # Objective: min t
        M.objective(ObjectiveSense.Minimize, t)
        if verbose:
            import sys as _s
            M.setLogHandler(_s.stdout)
        M.setSolverParam('numThreads', 4)
        M.setSolverParam('intpntCoTolPfeas', 1e-10)
        M.setSolverParam('intpntCoTolDfeas', 1e-10)
        M.setSolverParam('intpntCoTolRelGap', 1e-10)
        M.solve()
        status = str(M.getPrimalSolutionStatus())
        t_val = float(t.level()[0])
        y_val = np.array(y.level(), dtype=np.float64)
        # Duals on the t >= f_W(y) constraints: these ARE the optimal atomic-nu
        # weights (LP-minimax dual: t_opt = max_lam min_y Sum lam_W f_W(y)).
        lam_dual = np.array([float(c.dual()[0]) for c in w_cons], dtype=np.float64)
        lam_dual = np.maximum(lam_dual, 0.0)
        s = lam_dual.sum()
        if s > 0:
            lam_dual = lam_dual / s

    return {'t': t_val, 'y': y_val, 'status': status, 'lam_scalar': lam_dual}


if __name__ == "__main__":
    print(f"{'d':>3} {'k':>3} | {'scalar-LP (A)':>15} {'atomic-nu@JB (B)':>18} "
          f"{'atomic-nu@scalar* (D)':>23} {'joint_bisect (C)':>18} | "
          f"{'A-B':>9} {'D-A':>9} {'C-A':>9}")
    print("-" * 125)
    cases = [(4, 2), (4, 3), (6, 2), (6, 3), (8, 2), (10, 2)]
    for d, order in cases:
        try:
            # (C) joint_bisect
            lam_jb, t_hi_jb = seed_from_joint_bisect(d=d, order=order, tol=1e-6)
            # (B) atomic-nu scalar at lam from joint_bisect
            rB = solve_atomic_nu_sdp(lam_jb, d=d, order=order, solver='auto',
                                      compute_window_values=False)
            # (A) scalar min-max LP directly
            rA = solve_scalar_minmax(d, order)
            t_A = rA['t']
            # (D) atomic-nu scalar at the OPTIMAL scalar-LP lam
            lam_star = rA['lam_scalar']
            rD = solve_atomic_nu_sdp(lam_star, d=d, order=order, solver='auto',
                                      compute_window_values=False)
            t_B = rB.lb_numerical
            t_C = t_hi_jb
            t_D = rD.lb_numerical
            print(f"{d:>3} {order:>3} | {t_A:>15.6f} {t_B:>18.6f} "
                  f"{t_D:>23.6f} {t_C:>18.6f} | "
                  f"{t_A - t_B:>+9.4f} {t_D - t_A:>+9.4f} {t_C - t_A:>+9.4f}")
        except Exception as e:
            print(f"{d:>3} {order:>3} | FAILED: {type(e).__name__}: {e}")
