#!/usr/bin/env python3
r"""
Benchmark SDP solvers for Lasserre hierarchy — scaling study.

Runs order=2 at increasing d to find which solvers scale, then projects
feasibility for the full L3 (order=3, d=16) problem.

Full L3 target: d=16, order=3 → 969×969 moment matrix, 74K variables,
496 windows × 153×153 PSD. Most solvers will choke — the point of this
benchmark is to find which ones DON'T.

Solvers tested:
  1. CVXPY + SCS       (first-order ADMM, open source)
  2. CVXPY + Clarabel  (interior-point, open source)
  3. CVXPY + MOSEK     (interior-point, commercial)
  4. MOSEK Fusion API  (direct, no CVXPY overhead)
  5. CVXPY + COPT      (interior-point, commercial)

Usage:
  python tests/benchmark_sdp_solvers.py
  python tests/benchmark_sdp_solvers.py --max_d 16 --orders 2,3
"""
import argparse
import time
import sys
import json
import numpy as np
from scipy import sparse


# ── Shared helpers ───────────────────────────────────────────────────────────

def enum_monomials(d, max_deg):
    result = []
    def gen(pos, remaining, current):
        if pos == d:
            result.append(tuple(current))
            return
        for v in range(remaining + 1):
            current.append(v)
            gen(pos + 1, remaining - v, current)
            current.pop()
    gen(0, max_deg, [])
    return result


def build_window_matrices(d):
    conv_len = 2 * d - 1
    windows = [(ell, s) for ell in range(2, 2 * d + 1)
               for s in range(conv_len - ell + 2)]
    ii, jj = np.meshgrid(np.arange(d), np.arange(d), indexing='ij')
    sums = ii + jj
    M_mats = []
    for ell, s_lo in windows:
        mask = (sums >= s_lo) & (sums <= s_lo + ell - 2)
        M_mats.append((2.0 * d / ell) * mask.astype(np.float64))
    return windows, M_mats


def _add_mi(a, b, d):
    return tuple(a[i] + b[i] for i in range(d))

def _unit(d, i):
    return tuple(1 if k == i else 0 for k in range(d))


def _collect_all_moments(d, order, basis, loc_basis, consist_mono):
    mono_set = set()
    for a in basis:
        for b in basis:
            mono_set.add(_add_mi(a, b, d))
    for m in enum_monomials(d, 2 * order):
        mono_set.add(m)
    if order >= 2:
        units = [_unit(d, i) for i in range(d)]
        for a in loc_basis:
            for b in loc_basis:
                ab = _add_mi(a, b, d)
                mono_set.add(ab)
                for ei in units:
                    mono_set.add(_add_mi(ab, ei, d))
                    for ej in units:
                        mono_set.add(_add_mi(ab, _add_mi(ei, ej, d), d))
    for alpha in consist_mono:
        mono_set.add(alpha)
        for i in range(d):
            mono_set.add(_add_mi(alpha, _unit(d, i), d))
    mono_list = sorted(mono_set)
    idx = {m: i for i, m in enumerate(mono_list)}
    return mono_list, idx


def _build_perm_sparse(index_map, n_rows, n_y):
    flat_size = index_map.size
    rows = np.arange(flat_size)
    cols = index_map.ravel()
    vals = np.ones(flat_size, dtype=np.float64)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(flat_size, n_y))


def precompute_sdp_data(d, order=2):
    t0 = time.time()
    windows, M_mats = build_window_matrices(d)
    n_win = len(windows)

    basis = enum_monomials(d, order)
    n_basis = len(basis)
    loc_basis = enum_monomials(d, order - 1) if order >= 2 else []
    n_loc = len(loc_basis)
    consist_mono = enum_monomials(d, 2 * order - 1)

    mono_list, idx = _collect_all_moments(d, order, basis, loc_basis, consist_mono)
    n_y = len(mono_list)

    # Moment consistency sparse matrix
    c_rows, c_cols, c_vals = [], [], []
    row_count = 0
    for alpha in consist_mono:
        if alpha not in idx:
            continue
        for i in range(d):
            aei = _add_mi(alpha, _unit(d, i), d)
            if aei in idx:
                c_rows.append(row_count)
                c_cols.append(idx[aei])
                c_vals.append(1.0)
        c_rows.append(row_count)
        c_cols.append(idx[alpha])
        c_vals.append(-1.0)
        row_count += 1

    A_consist = None
    if row_count > 0:
        A_consist = sparse.csr_matrix(
            (c_vals, (c_rows, c_cols)), shape=(row_count, n_y))

    M_indices = np.zeros((n_basis, n_basis), dtype=int)
    for i in range(n_basis):
        for j in range(n_basis):
            M_indices[i, j] = idx[_add_mi(basis[i], basis[j], d)]
    P_moment = _build_perm_sparse(M_indices, n_basis, n_y)

    units = [_unit(d, i) for i in range(d)]
    loc_data = []
    if order >= 2:
        for i_var in range(d):
            ei = units[i_var]
            L_indices = np.zeros((n_loc, n_loc), dtype=int)
            L_valid = np.ones((n_loc, n_loc), dtype=np.float64)
            for a in range(n_loc):
                for b in range(n_loc):
                    mi = _add_mi(_add_mi(loc_basis[a], loc_basis[b], d), ei, d)
                    if mi in idx:
                        L_indices[a, b] = idx[mi]
                    else:
                        L_valid[a, b] = 0.0
            flat_size = n_loc * n_loc
            r = np.arange(flat_size)
            c = L_indices.ravel()
            v = L_valid.ravel()
            P_loc = sparse.csr_matrix((v, (r, c)), shape=(flat_size, n_y))
            loc_data.append(P_loc)

    t_indices = np.zeros((n_loc, n_loc), dtype=int)
    for a in range(n_loc):
        for b in range(n_loc):
            t_indices[a, b] = idx[_add_mi(loc_basis[a], loc_basis[b], d)]
    T_perm = _build_perm_sparse(t_indices, n_loc, n_y)

    ab_eiej_idx = np.full((n_loc, n_loc, d, d), -1, dtype=np.int32)
    for a in range(n_loc):
        for b in range(n_loc):
            ab = _add_mi(loc_basis[a], loc_basis[b], d)
            for i in range(d):
                for j in range(d):
                    mi = _add_mi(ab, _add_mi(units[i], units[j], d), d)
                    if mi in idx:
                        ab_eiej_idx[a, b, i, j] = idx[mi]

    ab_flat = (np.arange(n_loc).reshape(-1, 1) * n_loc
               + np.arange(n_loc).reshape(1, -1))

    flat_size_w = n_loc * n_loc
    Cw_list = []
    for w in range(n_win):
        Mw = M_mats[w]
        nz_i, nz_j = np.nonzero(Mw)
        if len(nz_i) == 0:
            Cw_list.append(None)
        else:
            y_all = ab_eiej_idx[:, :, nz_i, nz_j]
            valid_all = y_all >= 0
            ab_exp = np.broadcast_to(ab_flat[:, :, np.newaxis], y_all.shape)
            mw_vals = Mw[nz_i, nz_j]
            mw_exp = np.broadcast_to(mw_vals[np.newaxis, np.newaxis, :], y_all.shape)
            if np.any(valid_all):
                Cw = sparse.csr_matrix(
                    (mw_exp[valid_all].ravel().astype(np.float64),
                     (ab_exp[valid_all].ravel(), y_all[valid_all].ravel())),
                    shape=(flat_size_w, n_y))
            else:
                Cw = sparse.csr_matrix((flat_size_w, n_y))
            Cw_list.append(Cw)

    elapsed = time.time() - t0

    return {
        'n_y': n_y, 'n_basis': n_basis, 'n_loc': n_loc, 'n_win': n_win,
        'idx': idx, 'd': d, 'order': order,
        'A_consist': A_consist, 'P_moment': P_moment, 'loc_data': loc_data,
        'T_perm': T_perm, 'Cw_list': Cw_list, 'M_mats': M_mats,
        'precompute_time': elapsed,
    }


# ── CVXPY-based solvers ─────────────────────────────────────────────────────

def build_cvxpy_problem(data):
    import cvxpy as cp
    n_y = data['n_y']
    n_basis = data['n_basis']
    n_loc = data['n_loc']
    d = data['d']

    y = cp.Variable(n_y)
    t_param = cp.Parameter(nonneg=True)
    constraints = [y >= 0]

    zero = tuple(0 for _ in range(d))
    constraints.append(y[data['idx'][zero]] == 1)

    if data['A_consist'] is not None:
        constraints.append(data['A_consist'] @ y == 0)

    P_moment = data['P_moment']
    M_expr = cp.reshape(P_moment @ y, (n_basis, n_basis), order='C')
    constraints.append(M_expr >> 0)

    for P_loc in data['loc_data']:
        Li_expr = cp.reshape(P_loc @ y, (n_loc, n_loc), order='C')
        constraints.append(Li_expr >> 0)

    T_perm = data['T_perm']
    for w in range(data['n_win']):
        Cw = data['Cw_list'][w]
        if Cw is None:
            Lw_flat = t_param * (T_perm @ y)
        else:
            Lw_flat = t_param * (T_perm @ y) - Cw @ y
        Lw_expr = cp.reshape(Lw_flat, (n_loc, n_loc), order='C')
        constraints.append(Lw_expr >> 0)

    prob = cp.Problem(cp.Minimize(0), constraints)
    return prob, t_param


def solve_cvxpy(data, solver_name, solver_const, solver_opts,
                n_bisect=6, timeout=300):
    import cvxpy as cp

    t_build = time.time()
    prob, t_param = build_cvxpy_problem(data)
    build_time = time.time() - t_build

    solve_times = []
    t_total_start = time.time()

    def check(t_val):
        if time.time() - t_total_start > timeout:
            raise TimeoutError("solver timeout")
        t_param.value = t_val
        t0 = time.time()
        try:
            prob.solve(solver=solver_const, warm_start=True, **solver_opts)
            ok = prob.status in ['optimal', 'optimal_inaccurate']
        except Exception:
            ok = False
        solve_times.append(time.time() - t0)
        return ok

    lo, hi = 0.5, 5.0
    try:
        while not check(hi):
            hi *= 2
            if hi > 100:
                return {'solver': solver_name, 'error': 'cannot bracket',
                        'build_time': build_time}

        for step in range(n_bisect):
            mid = (lo + hi) / 2
            if check(mid):
                hi = mid
            else:
                lo = mid
    except TimeoutError:
        return {'solver': solver_name, 'error': f'timeout ({timeout}s)',
                'build_time': build_time,
                'partial_solves': len(solve_times),
                'avg_solve': np.mean(solve_times) if solve_times else 0}

    total_time = time.time() - t_total_start

    return {
        'solver': solver_name,
        'lb': lo,
        'build_time': build_time,
        'solve_times': solve_times,
        'avg_solve': np.mean(solve_times),
        'median_solve': np.median(solve_times),
        'total_time': total_time,
        'n_solves': len(solve_times),
    }


# ── MOSEK Fusion direct API ─────────────────────────────────────────────────

def solve_mosek_fusion(data, n_bisect=6, timeout=300):
    from mosek.fusion import Model, Domain, Expr, ObjectiveSense, Matrix
    from mosek.fusion import SolutionStatus

    n_y = data['n_y']
    n_basis = data['n_basis']
    n_loc = data['n_loc']
    d = data['d']
    idx = data['idx']
    zero = tuple(0 for _ in range(d))

    t_build = time.time()

    M = Model("lasserre")

    y = M.variable("y", n_y, Domain.greaterThan(0.0))
    M.constraint("y0", y.index(idx[zero]), Domain.equalsTo(1.0))

    A = data['A_consist']
    if A is not None:
        A_coo = A.tocoo()
        A_mosek = Matrix.sparse(A.shape[0], A.shape[1],
                                A_coo.row.tolist(), A_coo.col.tolist(),
                                A_coo.data.tolist())
        M.constraint("consist", Expr.mul(A_mosek, y), Domain.equalsTo(0.0))

    P = data['P_moment']
    P_coo = P.tocoo()
    P_mosek = Matrix.sparse(P.shape[0], P.shape[1],
                            P_coo.row.tolist(), P_coo.col.tolist(),
                            P_coo.data.tolist())
    mom_flat = Expr.mul(P_mosek, y)
    mom_mat = Expr.reshape(mom_flat, n_basis, n_basis)
    M.constraint("moment_psd", mom_mat, Domain.inPSDCone(n_basis))

    for i_var, P_loc in enumerate(data['loc_data']):
        P_coo = P_loc.tocoo()
        P_m = Matrix.sparse(P_loc.shape[0], P_loc.shape[1],
                            P_coo.row.tolist(), P_coo.col.tolist(),
                            P_coo.data.tolist())
        loc_flat = Expr.mul(P_m, y)
        loc_mat = Expr.reshape(loc_flat, n_loc, n_loc)
        M.constraint(f"loc_{i_var}", loc_mat, Domain.inPSDCone(n_loc))

    t_par = M.parameter("t")
    T = data['T_perm']
    T_coo = T.tocoo()
    T_mosek = Matrix.sparse(T.shape[0], T.shape[1],
                            T_coo.row.tolist(), T_coo.col.tolist(),
                            T_coo.data.tolist())
    T_y = Expr.mul(T_mosek, y)

    for w in range(data['n_win']):
        Cw = data['Cw_list'][w]
        if Cw is None:
            Lw_flat = Expr.mul(t_par, T_y)
        else:
            Cw_coo = Cw.tocoo()
            Cw_mosek = Matrix.sparse(Cw.shape[0], Cw.shape[1],
                                     Cw_coo.row.tolist(), Cw_coo.col.tolist(),
                                     Cw_coo.data.tolist())
            Cw_y = Expr.mul(Cw_mosek, y)
            Lw_flat = Expr.sub(Expr.mul(t_par, T_y), Cw_y)
        Lw_mat = Expr.reshape(Lw_flat, n_loc, n_loc)
        M.constraint(f"win_{w}", Lw_mat, Domain.inPSDCone(n_loc))

    M.objective(ObjectiveSense.Minimize, Expr.constTerm(0.0))
    build_time = time.time() - t_build

    solve_times = []
    t_total_start = time.time()

    def check(t_val):
        if time.time() - t_total_start > timeout:
            raise TimeoutError("solver timeout")
        t_par.setValue(t_val)
        t0 = time.time()
        try:
            M.solve()
            status = M.getPrimalSolutionStatus()
            ok = status in (SolutionStatus.Optimal, SolutionStatus.NearOptimal)
        except Exception:
            ok = False
        solve_times.append(time.time() - t0)
        return ok

    try:
        lo, hi = 0.5, 5.0
        while not check(hi):
            hi *= 2
            if hi > 100:
                M.dispose()
                return {'solver': 'MOSEK Fusion', 'error': 'cannot bracket',
                        'build_time': build_time}

        for step in range(n_bisect):
            mid = (lo + hi) / 2
            if check(mid):
                hi = mid
            else:
                lo = mid
    except TimeoutError:
        M.dispose()
        return {'solver': 'MOSEK Fusion', 'error': f'timeout ({timeout}s)',
                'build_time': build_time,
                'partial_solves': len(solve_times),
                'avg_solve': np.mean(solve_times) if solve_times else 0}

    total_time = time.time() - t_total_start
    M.dispose()

    return {
        'solver': 'MOSEK Fusion (direct)',
        'lb': lo,
        'build_time': build_time,
        'solve_times': solve_times,
        'avg_solve': np.mean(solve_times),
        'median_solve': np.median(solve_times),
        'total_time': total_time,
        'n_solves': len(solve_times),
    }


# ── Print helpers ────────────────────────────────────────────────────────────

def print_result(r):
    if 'error' in r:
        extra = ""
        if 'avg_solve' in r and r.get('partial_solves', 0) > 0:
            extra = f" (partial: {r['partial_solves']} solves, avg={r['avg_solve']:.3f}s)"
        print(f"  ERROR: {r['error']}{extra}")
    else:
        print(f"  Build: {r['build_time']:.3f}s")
        print(f"  Avg solve: {r['avg_solve']:.3f}s  "
              f"Median: {r['median_solve']:.3f}s")
        print(f"  Total: {r['total_time']:.3f}s  ({r['n_solves']} solves)")
        print(f"  Lower bound: {r['lb']:.10f}")


def print_summary(all_results):
    print(f"\n{'='*80}")
    print("FULL SUMMARY — ALL CONFIGURATIONS")
    print(f"{'='*80}")
    print(f"{'Config':<20} {'Solver':<25} {'Build':>7} {'AvgSlv':>8} "
          f"{'Total':>8} {'LB':>12}")
    print(f"{'-'*20} {'-'*25} {'-'*7} {'-'*8} {'-'*8} {'-'*12}")

    for cfg_label, results in all_results:
        for r in results:
            name = r['solver']
            if 'error' in r:
                err = r['error'][:15]
                build = f"{r.get('build_time', 0):.1f}s" if 'build_time' in r else "--"
                print(f"{cfg_label:<20} {name:<25} {build:>7} {'--':>8} "
                      f"{'--':>8} {err:>12}")
            else:
                print(f"{cfg_label:<20} {name:<25} {r['build_time']:>6.1f}s "
                      f"{r['avg_solve']:>7.3f}s {r['total_time']:>7.1f}s "
                      f"{r['lb']:>12.8f}")


# ── Main benchmark runner ───────────────────────────────────────────────────

def run_single_config(d, order, c_target, timeout_per_solver=300, n_bisect=6):
    """Run all solvers for one (d, order) config."""
    cfg_label = f"d={d},ord={order}"
    print(f"\n{'='*70}")
    print(f"CONFIG: d={d}, order={order}, c_target={c_target}")
    print(f"{'='*70}")

    print("Precomputing SDP data...", flush=True)
    try:
        data = precompute_sdp_data(d, order=order)
    except Exception as e:
        print(f"  FAILED to precompute: {e}")
        return cfg_label, [{'solver': 'ALL', 'error': f'precompute: {e}'}]

    print(f"  n_y={data['n_y']}, moment_mat={data['n_basis']}x{data['n_basis']}, "
          f"loc_mat={data['n_loc']}x{data['n_loc']}, windows={data['n_win']}")
    print(f"  Total PSD constraints: {1 + d + data['n_win']}")
    print(f"  Precompute: {data['precompute_time']:.2f}s")
    print()

    results = []
    timeout = timeout_per_solver

    # 1. SCS (skip for large problems — single solve takes >20min at d=16)
    print(f"{'─'*50}")
    print("1. CVXPY + SCS")
    if data['n_y'] > 1000:
        print("  SKIPPED: too slow at this scale (>20min per solve at d=16)")
        results.append({'solver': 'CVXPY+SCS', 'error': 'skipped (too slow)'})
    else:
        try:
            import cvxpy as cp
            r = solve_cvxpy(data, "CVXPY+SCS", cp.SCS,
                            {'verbose': False, 'max_iters': 50000,
                             'eps_abs': 1e-7, 'eps_rel': 1e-7},
                            n_bisect=n_bisect, timeout=min(timeout, 120))
            results.append(r)
            print_result(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({'solver': 'CVXPY+SCS', 'error': str(e)})
    print()

    # 2. Clarabel
    print(f"{'─'*50}")
    print("2. CVXPY + Clarabel")
    try:
        import cvxpy as cp
        r = solve_cvxpy(data, "CVXPY+Clarabel", cp.CLARABEL,
                        {'verbose': False},
                        n_bisect=n_bisect, timeout=timeout)
        results.append(r)
        print_result(r)
    except Exception as e:
        print(f"  FAILED: {e}")
        results.append({'solver': 'CVXPY+Clarabel', 'error': str(e)})
    print()

    # 3. MOSEK via CVXPY
    print(f"{'─'*50}")
    print("3. CVXPY + MOSEK")
    try:
        import mosek
        mosek.Env().checkoutlicense(mosek.feature.pton)
        import cvxpy as cp
        r = solve_cvxpy(data, "CVXPY+MOSEK", cp.MOSEK,
                        {'verbose': False},
                        n_bisect=n_bisect, timeout=timeout)
        results.append(r)
        print_result(r)
    except ImportError:
        print("  SKIPPED: mosek not installed")
        results.append({'solver': 'CVXPY+MOSEK', 'error': 'not installed'})
    except mosek.Error:
        print("  SKIPPED: no MOSEK license")
        results.append({'solver': 'CVXPY+MOSEK', 'error': 'no license'})
    except Exception as e:
        print(f"  FAILED: {e}")
        results.append({'solver': 'CVXPY+MOSEK', 'error': str(e)})
    print()

    # 4. MOSEK Fusion direct
    print(f"{'─'*50}")
    print("4. MOSEK Fusion (direct)")
    try:
        import mosek as _mosek
        _mosek.Env().checkoutlicense(_mosek.feature.pton)
        from mosek.fusion import Model  # noqa: F401
        r = solve_mosek_fusion(data, n_bisect=n_bisect, timeout=timeout)
        results.append(r)
        print_result(r)
    except ImportError:
        print("  SKIPPED: mosek.fusion not installed")
        results.append({'solver': 'MOSEK Fusion (direct)', 'error': 'not installed'})
    except _mosek.Error:
        print("  SKIPPED: no MOSEK license")
        results.append({'solver': 'MOSEK Fusion (direct)', 'error': 'no license'})
    except Exception as e:
        print(f"  FAILED: {e}")
        results.append({'solver': 'MOSEK Fusion (direct)', 'error': str(e)})
    print()

    # 5. COPT
    print(f"{'─'*50}")
    print("5. CVXPY + COPT")
    try:
        import coptpy  # noqa: F401
        import cvxpy as cp
        r = solve_cvxpy(data, "CVXPY+COPT", cp.COPT,
                        {'verbose': False},
                        n_bisect=n_bisect, timeout=timeout)
        results.append(r)
        print_result(r)
    except ImportError:
        print("  SKIPPED: coptpy not installed")
        results.append({'solver': 'CVXPY+COPT', 'error': 'not installed'})
    except Exception as e:
        print(f"  FAILED: {e}")
        results.append({'solver': 'CVXPY+COPT', 'error': str(e)})
    print()

    return cfg_label, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c_target', type=float, default=1.10)
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout per solver in seconds (default: 300)')
    parser.add_argument('--n_bisect', type=int, default=6)
    args = parser.parse_args()

    print(f"SDP SOLVER SCALING BENCHMARK")
    print(f"c_target={args.c_target}, timeout={args.timeout}s/solver, "
          f"bisect={args.n_bisect} steps\n")

    # Scaling study: small → large
    configs = [
        # (d, order) — increasing difficulty
        (4,  2),   # tiny:   basis=15,  loc=5,   ~30 windows
        (8,  2),   # small:  basis=45,  loc=9,   ~105 windows
        (16, 2),   # medium: basis=153, loc=17,  ~496 windows
        (4,  3),   # L3 tiny:  basis=35,  loc=15,  ~30 windows
        (8,  3),   # L3 small: basis=165, loc=45, ~105 windows
        # (16, 3), # L3 full: basis=969, loc=153, ~496 windows — only if above works
    ]

    all_results = []
    for d, order in configs:
        cfg_label, results = run_single_config(
            d, order, args.c_target,
            timeout_per_solver=args.timeout,
            n_bisect=args.n_bisect)
        all_results.append((cfg_label, results))

    # Check if any solver survived d=8,order=3 — if so, try d=16,order=3
    last_cfg, last_results = all_results[-1]
    survivors = [r for r in last_results if 'error' not in r]
    if survivors:
        fastest = min(survivors, key=lambda r: r['total_time'])
        print(f"\n*** Fastest solver at {last_cfg}: {fastest['solver']} "
              f"({fastest['total_time']:.1f}s)")
        print(f"*** Attempting L3 full (d=16, order=3)...")
        # Increase timeout for the big one
        cfg_label, results = run_single_config(
            16, 3, args.c_target,
            timeout_per_solver=args.timeout * 4,
            n_bisect=args.n_bisect)
        all_results.append((cfg_label, results))
    else:
        print(f"\n*** No solver completed {last_cfg} — skipping d=16,order=3")

    print_summary(all_results)

    # Scaling analysis
    print(f"\n{'='*80}")
    print("SCALING ANALYSIS")
    print(f"{'='*80}")
    solver_times = {}
    for cfg_label, results in all_results:
        for r in results:
            name = r['solver']
            if name not in solver_times:
                solver_times[name] = []
            if 'error' not in r:
                solver_times[name].append((cfg_label, r['total_time']))
            else:
                solver_times[name].append((cfg_label, None))

    for name, times in solver_times.items():
        completed = [(c, t) for c, t in times if t is not None]
        failed = [(c, t) for c, t in times if t is None]
        if completed:
            series = ", ".join(f"{c}={t:.1f}s" for c, t in completed)
            print(f"  {name}: {series}")
            if len(completed) >= 2:
                t1, t2 = completed[-2][1], completed[-1][1]
                ratio = t2 / t1 if t1 > 0 else float('inf')
                print(f"    Scaling ratio (last two): {ratio:.1f}x")
        if failed:
            print(f"    Failed: {', '.join(c for c, _ in failed)}")

    print("\nDone.")


if __name__ == '__main__':
    main()
