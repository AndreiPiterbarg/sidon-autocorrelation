"""Core utilities extracted from lasserre_level2_v2.ipynb.

Functions:
- StepFunction, peak_autoconv_exact: step function evaluation and autoconvolution
- monomial_basis, combine, reflect_index, canonical_index, build_symmetry_map: basis/symmetry
- solve_lasserre_2_v2: Level-2 Lasserre SDP solver
- solve_primal_improved: Primal upper bound solver
- diagnose_solution, print_diagnostics: Solution diagnostics
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from itertools import combinations_with_replacement
import time
import warnings as _warnings


# --- Solver selection ---
if 'MOSEK' in cp.installed_solvers():
    PRIMARY_SOLVER = 'MOSEK'
else:
    PRIMARY_SOLVER = 'CLARABEL'


# --- StepFunction and autoconvolution ---

class StepFunction:
    """A piecewise-constant function on [-1/4, 1/4] defined by bin edges and heights."""

    def __init__(self, edges, heights):
        self.edges = np.asarray(edges, dtype=float)
        self.heights = np.asarray(heights, dtype=float)

    @classmethod
    def from_heights(cls, edges, heights):
        return cls(edges=edges, heights=heights)

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x)
        for k in range(len(self.heights)):
            mask = (x >= self.edges[k]) & (x < self.edges[k + 1])
            result[mask] = self.heights[k]
        return result


def _autoconv_direct(sf, t_values):
    """Compute autoconvolution (f*f)(t) at given t values by direct integration."""
    t_values = np.asarray(t_values, dtype=float)
    edges = sf.edges
    h = sf.heights
    N = len(h)
    a = edges[:-1]
    b = edges[1:]
    hh = h[:, None] * h[None, :]
    T = len(t_values)
    result = np.empty(T, dtype=float)
    batch_size = 1000
    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        t_batch = t_values[start:end]
        lo = np.maximum(a[:, None, None], t_batch[None, None, :] - b[None, :, None])
        hi = np.minimum(b[:, None, None], t_batch[None, None, :] - a[None, :, None])
        overlap = np.maximum(0.0, hi - lo)
        result[start:end] = (hh[:, :, None] * overlap).sum(axis=(0, 1))
    return result


def peak_autoconv_exact(sf):
    """Compute the exact peak of the autoconvolution at all breakpoints."""
    edges = sf.edges
    bp = (edges[:, None] + edges[None, :]).ravel()
    bp = np.unique(bp)
    bp = bp[(bp >= -0.5) & (bp <= 0.5)]
    conv = _autoconv_direct(sf, bp)
    return float(np.max(conv))


# --- Monomial basis and symmetry ---

def monomial_basis(P, max_deg):
    """Enumerate monomials of degree <= max_deg in P variables."""
    basis = [()]
    for deg in range(1, max_deg + 1):
        for combo in combinations_with_replacement(range(P), deg):
            basis.append(combo)
    return basis


def combine(*multi_indices):
    """Combine multi-indices by concatenation and sorting."""
    return tuple(sorted(sum(multi_indices, ())))


def reflect_index(multi_idx, P):
    """Reflect a multi-index under x_i <-> x_{P-1-i}."""
    return tuple(sorted(P - 1 - i for i in multi_idx))


def canonical_index(multi_idx, P):
    """Canonical representative under reflection symmetry."""
    reflected = reflect_index(multi_idx, P)
    return min(multi_idx, reflected)


def build_symmetry_map(moments_list, P):
    """Build a mapping from each moment to its canonical representative."""
    equiv_classes = {}
    for m in moments_list:
        c = canonical_index(m, P)
        if c not in equiv_classes:
            equiv_classes[c] = []
        if m not in equiv_classes[c]:
            equiv_classes[c].append(m)

    canon_moments = sorted(equiv_classes.keys(), key=lambda m: (len(m), m))
    canon_idx = {m: i for i, m in enumerate(canon_moments)}
    full_map = {}
    for cm, members in equiv_classes.items():
        idx = canon_idx[cm]
        for m in members:
            full_map[m] = idx
    return canon_moments, canon_idx, full_map, equiv_classes


# --- Level-2 Lasserre SDP solver ---

def solve_lasserre_2_v2(P, solver=None, verbose=False, eta_tol=1e-6,
                        primal_hint=None, use_symmetry=True,
                        use_product_constraints=True):
    """Level-2 Lasserre SDP relaxation for discretized C_1a."""
    t0 = time.time()

    basis_2 = monomial_basis(P, 2)
    d = len(basis_2)
    basis_1 = monomial_basis(P, 1)
    loc_d = len(basis_1)
    all_beta = monomial_basis(P, 3)

    # Collect all unique moments needed (degree <= 4)
    moments_set = set()
    for i in range(d):
        for j in range(i, d):
            moments_set.add(combine(basis_2[i], basis_2[j]))
    for k in range(P):
        for a in range(loc_d):
            for b in range(a, loc_d):
                moments_set.add(combine((k,), basis_1[a], basis_1[b]))
    for beta in all_beta:
        moments_set.add(beta)
        for i in range(P):
            moments_set.add(combine((i,), beta))
    for k_conv in range(2 * P - 1):
        for a in range(loc_d):
            for b in range(a, loc_d):
                moments_set.add(combine(basis_1[a], basis_1[b]))
                for i in range(P):
                    j_val = k_conv - i
                    if 0 <= j_val < P:
                        moments_set.add(combine(basis_1[a], basis_1[b], (i,), (j_val,)))
    if use_product_constraints:
        for i in range(P):
            for j in range(i + 1, P):
                for a in range(loc_d):
                    for b in range(a, loc_d):
                        moments_set.add(combine((i,), (j,), basis_1[a], basis_1[b]))

    moments_list = sorted(moments_set, key=lambda m: (len(m), m))

    if use_symmetry:
        canon_moments, canon_idx_map, full_map, equiv_classes = \
            build_symmetry_map(moments_list, P)
        n_mom = len(canon_moments)
        moment_idx = full_map
        for m, idx in canon_idx_map.items():
            moment_idx[m] = idx
    else:
        n_mom = len(moments_list)
        moment_idx = {m: idx for idx, m in enumerate(moments_list)}

    print(f"  P={P}: d={d}, loc_d={loc_d}, n_moments={n_mom}"
          f"{' (symmetry-reduced)' if use_symmetry else ''}"
          f"{' +products' if use_product_constraints else ''}")

    # Pre-compute indicator matrices
    B_M = {}
    for i in range(d):
        for j in range(i, d):
            mu = combine(basis_2[i], basis_2[j])
            idx = moment_idx[mu]
            if idx not in B_M:
                B_M[idx] = np.zeros((d, d))
            B_M[idx][i, j] += 1
            if i != j:
                B_M[idx][j, i] += 1

    if use_symmetry:
        nonneg_indices = list(range((P + 1) // 2))
    else:
        nonneg_indices = list(range(P))

    B_Locs = []
    for k in nonneg_indices:
        B_L = {}
        for a in range(loc_d):
            for b in range(a, loc_d):
                mu = combine((k,), basis_1[a], basis_1[b])
                idx = moment_idx[mu]
                if idx not in B_L:
                    B_L[idx] = np.zeros((loc_d, loc_d))
                B_L[idx][a, b] += 1
                if a != b:
                    B_L[idx][b, a] += 1
        B_Locs.append(B_L)

    B_prod_Locs = []
    if use_product_constraints:
        for i in range(P):
            for j in range(i + 1, P):
                if use_symmetry:
                    i2, j2 = P - 1 - j, P - 1 - i
                    if (i2, j2) < (i, j):
                        continue
                B_pL = {}
                for a in range(loc_d):
                    for b in range(a, loc_d):
                        mu = combine((i,), (j,), basis_1[a], basis_1[b])
                        idx = moment_idx[mu]
                        if idx not in B_pL:
                            B_pL[idx] = np.zeros((loc_d, loc_d))
                        B_pL[idx][a, b] += 1
                        if a != b:
                            B_pL[idx][b, a] += 1
                B_prod_Locs.append(B_pL)

    simplex_data = []
    for beta in all_beta:
        lhs_indices = [moment_idx[combine((i,), beta)] for i in range(P)]
        rhs_idx = moment_idx[beta]
        simplex_data.append((lhs_indices, rhs_idx))

    B_M1 = {}
    for a in range(loc_d):
        for b in range(a, loc_d):
            mu = combine(basis_1[a], basis_1[b])
            idx = moment_idx[mu]
            if idx not in B_M1:
                B_M1[idx] = np.zeros((loc_d, loc_d))
            B_M1[idx][a, b] += 1
            if a != b:
                B_M1[idx][b, a] += 1

    if use_symmetry:
        conv_indices = list(range(P))
    else:
        conv_indices = list(range(2 * P - 1))

    B_pks = []
    for k_conv in conv_indices:
        B_pk = {}
        for a in range(loc_d):
            for b in range(a, loc_d):
                for i in range(P):
                    j_val = k_conv - i
                    if 0 <= j_val < P:
                        mu = combine(basis_1[a], basis_1[b], (i,), (j_val,))
                        idx = moment_idx[mu]
                        if idx not in B_pk:
                            B_pk[idx] = np.zeros((loc_d, loc_d))
                        B_pk[idx][a, b] += 1
                        if a != b:
                            B_pk[idx][b, a] += 1
        B_pks.append(B_pk)

    # Build CVXPY problem
    y = cp.Variable(n_mom)
    eta_param = cp.Parameter(nonneg=True)
    constraints = []

    M_expr = sum(y[idx] * mat for idx, mat in B_M.items())
    constraints.append(M_expr >> 0)
    constraints.append(y[moment_idx[()]] == 1)

    for B_L in B_Locs:
        L_k = sum(y[idx] * mat for idx, mat in B_L.items())
        constraints.append(L_k >> 0)

    for B_pL in B_prod_Locs:
        L_ij = sum(y[idx] * mat for idx, mat in B_pL.items())
        constraints.append(L_ij >> 0)

    for lhs_indices, rhs_idx in simplex_data:
        constraints.append(sum(y[i] for i in lhs_indices) == y[rhs_idx])

    M1_expr = sum(y[idx] * mat for idx, mat in B_M1.items())
    for B_pk in B_pks:
        pk_expr = sum(y[idx] * mat for idx, mat in B_pk.items())
        L_gk = eta_param * M1_expr - 2 * P * pk_expr
        constraints.append(L_gk >> 0)

    prob = cp.Problem(cp.Minimize(0), constraints)

    if solver is not None:
        solver_list = [solver]
    else:
        solver_list = [PRIMARY_SOLVER]
        if PRIMARY_SOLVER == 'MOSEK' and 'CLARABEL' in cp.installed_solvers():
            solver_list.append('CLARABEL')
        elif PRIMARY_SOLVER != 'MOSEK' and 'SCS' in cp.installed_solvers():
            solver_list.append('SCS')

    def try_solve(tight=True):
        for s in solver_list:
            try:
                kwargs = {'verbose': False}
                if s == 'MOSEK':
                    if tight:
                        kwargs['mosek_params'] = {
                            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-10,
                            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-10,
                            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-10,
                            'MSK_DPAR_INTPNT_CO_TOL_MU_RED': 1e-12,
                        }
                    else:
                        kwargs['mosek_params'] = {
                            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-7,
                            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-7,
                            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-7,
                        }
                elif s == 'SCS':
                    kwargs.update({'max_iters': 10000, 'eps': 1e-7})
                with _warnings.catch_warnings():
                    _warnings.filterwarnings('ignore',
                                             message='Solution may be inaccurate')
                    prob.solve(solver=s, warm_start=True, **kwargs)
                return s, prob.status
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                continue
        return None, 'solver_error'

    # Binary search on eta
    shor_bound = 2 * P / (2 * P - 1)
    eta_lo = shor_bound
    eta_hi = (primal_hint + 0.001) if primal_hint is not None else 2.0

    best_y = None
    used_solver = None
    n_iter = 0

    eta_param.value = eta_hi
    used_solver, status = try_solve()
    if status not in ('optimal', 'optimal_inaccurate'):
        eta_hi = 2.0 * P
        eta_param.value = eta_hi
        used_solver, status = try_solve()
        if status not in ('optimal', 'optimal_inaccurate'):
            print(f"  WARNING: even eta={eta_hi} is infeasible ({status})")
            return {'status': 'infeasible', 'bound': None,
                    'time': time.time() - t0, 'P': P}
    best_y = y.value.copy()

    # Phase 1: Coarse binary search
    eta_tol_coarse = max(eta_tol, 1e-3)
    while eta_hi - eta_lo > eta_tol_coarse:
        eta_mid = (eta_lo + eta_hi) / 2
        eta_param.value = eta_mid
        _, status = try_solve()
        n_iter += 1
        if status in ('optimal', 'optimal_inaccurate'):
            eta_hi = eta_mid
            best_y = y.value.copy()
        else:
            if eta_mid > shor_bound + 0.01:
                _, status2 = try_solve(tight=False)
                if status2 in ('optimal', 'optimal_inaccurate'):
                    eta_hi = eta_mid
                    best_y = y.value.copy()
                    continue
            eta_lo = eta_mid

    # Phase 2: Fine binary search
    if eta_tol < eta_tol_coarse:
        while eta_hi - eta_lo > eta_tol:
            eta_mid = (eta_lo + eta_hi) / 2
            eta_param.value = eta_mid
            _, status = try_solve()
            n_iter += 1
            if status in ('optimal', 'optimal_inaccurate'):
                eta_hi = eta_mid
                best_y = y.value.copy()
            else:
                eta_lo = eta_mid

    solve_time = time.time() - t0
    n_prod = len(B_prod_Locs)
    print(f"  Binary search: {n_iter} iters, eta* in [{eta_lo:.8f}, {eta_hi:.8f}], "
          f"{solve_time:.1f}s"
          f"{f', {n_prod} product constraints' if use_product_constraints else ''}")

    result = {
        'status': 'optimal', 'time': solve_time,
        'solver': used_solver, 'd': d, 'n_moments': n_mom,
        'bound': eta_hi, 'n_iter': n_iter, 'P': P,
        'eta_lo': eta_lo, 'eta_hi': eta_hi,
        'use_symmetry': use_symmetry,
        'use_product_constraints': use_product_constraints,
        'n_product_constraints': n_prod,
    }

    if best_y is not None:
        first_mom = np.array([best_y[moment_idx[(i,)]] for i in range(P)])
        Y_mat = np.zeros((P, P))
        for i in range(P):
            for j in range(P):
                Y_mat[i, j] = best_y[moment_idx[tuple(sorted((i, j)))]]
        M_val = sum(best_y[idx] * mat for idx, mat in B_M.items())
        M_eigvals = np.linalg.eigvalsh(M_val)
        M_rank = int(np.sum(M_eigvals > 1e-6 * max(M_eigvals.max(), 1e-12)))
        M1_val = M_val[:loc_d, :loc_d]
        M1_eigvals = np.linalg.eigvalsh(M1_val)
        M1_rank = int(np.sum(M1_eigvals > 1e-6 * max(M1_eigvals.max(), 1e-12)))

        result.update({
            'first_moments': first_mom, 'Y': Y_mat,
            'M': M_val, 'M_rank': M_rank, 'M_eigvals': M_eigvals,
            'M1_rank': M1_rank, 'M1_eigvals': M1_eigvals,
            'best_y': best_y, 'moment_idx': moment_idx,
            'B_M': B_M, 'B_Locs': B_Locs, 'B_M1': B_M1, 'B_pks': B_pks,
            'B_prod_Locs': B_prod_Locs,
            'basis_1': basis_1, 'loc_d': loc_d,
        })

    return result


# --- Primal solver ---

def solve_primal_improved(P, n_restarts=100, seeds=None, warm_start_x=None):
    """Primal upper bound via LSE continuation + L-BFGS-B."""
    if seeds is None:
        seeds = list(range(20))

    def softmax(z):
        z = z - np.max(z)
        e = np.exp(z)
        return e / np.sum(e)

    def objective_lse(z, beta):
        x = softmax(z)
        conv = np.convolve(x, x, mode='full')
        c_max = np.max(conv)
        return 2 * P * (c_max + (1.0 / beta) * np.log(
            np.sum(np.exp(beta * (conv - c_max)))))

    def objective_exact(z):
        x = softmax(z)
        conv = np.convolve(x, x, mode='full')
        return 2 * P * np.max(conv)

    best_val = np.inf
    best_x = None
    restarts_per_seed = max(1, n_restarts // len(seeds))
    beta_schedule = [5, 20, 100, 500, 2000]

    for seed in seeds:
        rng = np.random.RandomState(seed)
        for _ in range(restarts_per_seed):
            z0 = rng.randn(P) * 0.5
            for beta in beta_schedule:
                res = minimize(lambda z, b=beta: objective_lse(z, b), z0,
                               method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
                z0 = res.x
            res = minimize(objective_exact, z0, method='L-BFGS-B',
                           options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = softmax(res.x)

    structured_starts = [np.zeros(P), np.linspace(-1, 1, P), np.linspace(1, -1, P)]

    if warm_start_x is not None:
        x_clip = np.maximum(warm_start_x, 1e-10)
        z_warm = np.log(x_clip)
        z_warm -= z_warm.mean()
        structured_starts.append(z_warm)
        rng = np.random.RandomState(9999)
        for _ in range(10):
            structured_starts.append(z_warm + rng.randn(P) * 0.1)

    for z0 in structured_starts:
        for beta in beta_schedule:
            res = minimize(lambda z, b=beta: objective_lse(z, b), z0,
                           method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
            z0 = res.x
        res = minimize(objective_exact, z0, method='L-BFGS-B',
                       options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12})
        if res.fun < best_val:
            best_val = res.fun
            best_x = softmax(res.x)

    w = 1.0 / (2 * P)
    edges = np.linspace(-0.25, 0.25, P + 1)
    heights = best_x / w
    sf = StepFunction(edges=edges, heights=heights)
    val_exact = peak_autoconv_exact(sf)
    return val_exact, best_x


# --- Diagnostics ---

def diagnose_solution(res):
    """Compute diagnostic information from a Lasserre-2 solution."""
    diag = {'P': res['P']}

    if res.get('M') is None:
        diag['error'] = 'No moment matrix available'
        return diag

    M = res['M']
    M_eigvals = res['M_eigvals']
    d = res['d']

    diag['M2_rank'] = res['M_rank']
    diag['M2_min_eigval'] = float(M_eigvals[0])
    diag['M2_max_eigval'] = float(M_eigvals[-1])
    diag['M2_cond'] = float(M_eigvals[-1] / max(abs(M_eigvals[0]), 1e-15))
    diag['M2_dim'] = d
    diag['M1_rank'] = res.get('M1_rank', '?')
    M1_eigvals = res.get('M1_eigvals')
    if M1_eigvals is not None:
        diag['M1_min_eigval'] = float(M1_eigvals[0])
    if res.get('M1_rank') is not None:
        diag['flat_extension'] = (res['M_rank'] == res['M1_rank'])
    else:
        diag['flat_extension'] = None

    first_mom = res.get('first_moments')
    if first_mom is not None:
        diag['moment_sum'] = float(np.sum(first_mom))
        diag['moment_sum_error'] = abs(float(np.sum(first_mom)) - 1.0)
        diag['min_first_moment'] = float(np.min(first_mom))
        diag['max_first_moment'] = float(np.max(first_mom))

    return diag


def print_diagnostics(diag, P=None):
    """Pretty-print diagnostics."""
    P = P or diag.get('P', '?')
    print(f"\n  --- Diagnostics for P={P} ---")

    if 'error' in diag:
        print(f"  ERROR: {diag['error']}")
        return

    print(f"  M_2: rank={diag['M2_rank']}/{diag['M2_dim']}, "
          f"min_eig={diag['M2_min_eigval']:.2e}, "
          f"cond={diag['M2_cond']:.2e}")

    if 'M1_rank' in diag:
        print(f"  M_1: rank={diag['M1_rank']}, "
              f"min_eig={diag.get('M1_min_eigval', '?'):.2e}")

    if diag.get('flat_extension') is not None:
        status = 'YES (TIGHT)' if diag['flat_extension'] else 'NO'
        print(f"  Flat extension: {status}")

    if 'moment_sum' in diag:
        print(f"  Moment sum: {diag['moment_sum']:.10f} "
              f"(error={diag['moment_sum_error']:.2e})")
