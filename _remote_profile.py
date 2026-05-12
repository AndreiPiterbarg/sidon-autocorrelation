"""Profile each phase: cache build, fmat build, model build, solve."""
import sys
import time
import numpy as np

sys.path.insert(0, '.')
sys.stdout.reconfigure(line_buffering=True)
from interval_bnb.windows import build_windows
from interval_bnb.bound_sdp_escalation import (
    build_sdp_escalation_cache, _get_or_build_fusion_matrices, _import_fusion,
)


def profile(d, hw, verbose=True, tol=1e-9):
    print(f"\n=== d={d} hw={hw} tol={tol} ===")
    t0 = time.time()
    windows = build_windows(d)
    t_win = time.time() - t0
    print(f"  build_windows: {t_win:.3f}s, |W|={len(windows)}")

    t0 = time.time()
    cache = build_sdp_escalation_cache(d, windows)
    t_cache = time.time() - t0
    print(f"  build_cache:   {t_cache:.3f}s, n_y={cache['n_y']}, B={cache['B']}, B1={cache['B1']}, n_eq={cache['n_eq']}, n_W={cache['n_W_kept']}")

    t0 = time.time()
    fmats = _get_or_build_fusion_matrices(cache)
    t_fmat = time.time() - t0
    print(f"  fusion mats:   {t_fmat:.3f}s")

    rng = np.random.default_rng(0)
    mu = rng.dirichlet(np.ones(d))
    if mu[0] > mu[-1]:
        mu = mu[::-1]
    lo = np.maximum(mu - hw, 0.0)
    hi = np.minimum(mu + hw, 1.0)
    if lo.sum() > 1.0 or hi.sum() < 1.0:
        lo = np.maximum(mu - 0.04, 0.0)
        hi = np.minimum(mu + 0.04, 1.0)

    F = _import_fusion()
    Model, Domain, Expr = F['Model'], F['Domain'], F['Expr']
    ObjectiveSense = F['ObjectiveSense']
    AccSolutionStatus = F['AccSolutionStatus']

    t0 = time.time()
    M = Model(f"profile_d{d}")
    if verbose:
        M.setLogHandler(sys.stdout)
    M.setSolverParam('intpntCoTolPfeas', tol)
    M.setSolverParam('intpntCoTolDfeas', tol)
    M.setSolverParam('intpntCoTolRelGap', tol)
    M.setSolverParam('numThreads', 1)
    M.acceptedSolutionStatus(AccSolutionStatus.Anything)

    n_y = cache['n_y']
    B = cache['B']
    B1 = cache['B1']
    n_W = cache['n_W_kept']
    y = M.variable('y', n_y, Domain.greaterThan(0.0))
    u = M.variable('u', 1, Domain.unbounded())

    M.constraint('eq', Expr.mul(fmats['F_eq'], y),
                 Domain.equalsTo(fmats['eq_b_list']))
    n_M_svec = B * (B + 1) // 2
    M.constraint('mom_M', Expr.mul(fmats['F_M'], y),
                 Domain.inSVecPSDCone(n_M_svec))
    F_loc_base = fmats['F_loc_base']
    F_loc_a_plus_list = fmats['F_loc_a_plus']
    n_L_svec = B1 * (B1 + 1) // 2
    F_base_y = Expr.mul(F_loc_base, y)
    for i_var in range(d):
        F_a_plus_i_y = Expr.mul(F_loc_a_plus_list[i_var], y)
        L_flat = Expr.sub(F_a_plus_i_y, Expr.mul(float(lo[i_var]), F_base_y))
        M.constraint(f'L{i_var}', L_flat, Domain.inSVecPSDCone(n_L_svec))
        U_flat = Expr.sub(Expr.mul(float(hi[i_var]), F_base_y), F_a_plus_i_y)
        M.constraint(f'U{i_var}', U_flat, Domain.inSVecPSDCone(n_L_svec))
    if n_W > 0 and fmats['F_W'] is not None:
        u_repeat = Expr.repeat(u, int(n_W), 0)
        By_expr = Expr.mul(fmats['F_W'], y)
        M.constraint('win', Expr.sub(u_repeat, By_expr),
                     Domain.greaterThan(0.0))
    M.objective(ObjectiveSense.Minimize, u)
    t_build = time.time() - t0
    print(f"  build_model:   {t_build:.3f}s")

    t0 = time.time()
    M.solve()
    t_solve = time.time() - t0
    pst = str(M.getPrimalSolutionStatus()).split('.')[-1]
    dst = str(M.getDualSolutionStatus()).split('.')[-1]
    try:
        pobj = float(M.primalObjValue())
    except Exception:
        pobj = float('nan')
    try:
        dobj = float(M.dualObjValue())
    except Exception:
        dobj = float('nan')
    try:
        iters = int(M.getSolverIntInfo('intpntIter'))
    except Exception:
        iters = -1
    print(f"  SOLVE:         {t_solve:.3f}s status={pst}/{dst} pobj={pobj} dobj={dobj} iters={iters}")
    M.dispose()


if __name__ == '__main__':
    import os
    only = os.environ.get('PROFILE_ONLY')
    tol = float(os.environ.get('PROFILE_TOL', '1e-9'))
    verbose = os.environ.get('PROFILE_VERBOSE', '0') == '1'
    if only:
        profile(int(only), 0.025, verbose=verbose, tol=tol)
    else:
        for d in (10, 16, 22, 30):
            try:
                profile(d, 0.025, verbose=verbose, tol=tol)
            except Exception as e:
                print(f"d={d} ERROR: {type(e).__name__}: {e}")
