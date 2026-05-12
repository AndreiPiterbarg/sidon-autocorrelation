"""F1_daubechies_wavelet probe v3 -- mathematically corrected.

The CORRECT formulation for an LB on C_{1a} via Daubechies wavelets:

For any window W ⊂ [-1/2, 1/2] of length |W|, and any f >= 0 with
∫f = 1, supp(f) ⊆ [-1/4, 1/4],
  ||f*f||_∞  >=  (1/|W|) ∫_W (f*f)(t) dt
              =  c^T M_W c
where c = (c_k) are the Daubechies db_N scaling-function coefficients of f
(at level J),
  M_W := (1/|W|) ∫_W M(t) dt,    M(t)_{ij} = Psi(2^J t - 2 k_min - i - j),
  Psi(y) := (phi*phi)(y).

So
  C_{1a}  >=  inf_{c}  max_{W}  c^T M_W c       (where c ranges over ALL
                                                  valid coefficient vecs.)
  s.t.  A c >= 0, 1^T c = beta = 2^{J/2}.

This is val_W(N, J), and val_W(N, J) <= C_{1a}.

A relaxation of val_W gives a LOWER BOUND on val_W, hence on C_{1a}.

ORDER-1 SHOR RELAXATION (gives LB):
  min V
  s.t. tr(M_W Y) <= V  for all W
       1^T c = beta
       A c >= 0
       [[1, c^T], [c, Y]] PSD

Plus RLT cuts (A c)_p (1^T c) >= 0 etc. for tightness.

The result V_LB is a RIGOROUS LOWER BOUND on C_{1a} *if* the basis spans
ALL valid f -- which it doesn't, since wavelet basis at level J only
spans a subspace.  HOWEVER: the LB is rigorous on the SUBSPACE: any f
in the subspace has ||f*f||_∞ >= V_LB.

SOUND but NOT directly applicable to C_{1a}.

To make it directly applicable, we need the basis to be RICH ENOUGH that
every nonneg f can be approximated.  At a fixed J, this fails.

MORE SUBTLE: the relaxation Y >> c c^T allows c to be a
"super-distribution" that doesn't correspond to any actual f.  The
inf over this enlarged feasible set is a LOWER BOUND on inf over the
original.  It is NOT necessarily a LOWER BOUND on C_{1a} unless the
basis covers everything.

BUT: if we choose enough WINDOWS W, the cuts c^T M_W c <= V translate
to constraints on the Daubechies wavelet coefficients of f, and
ANY nonneg f satisfies them (by the average inequality).  So:
  V_LB on val_W (over the basis subspace) is consistent with: any f in
  the wavelet subspace has ||f*f||_∞ >= V_LB.
  Outside the subspace, no claim.

So this probe gives a LB on C_{1a} *restricted to the wavelet basis subspace*.
NOT a universal LB on C_{1a}.

This is the same kind of issue as discrete Lasserre (val(d)) -- but
that is rescued by the CS step-function inequality which says any
nonneg f has CS-bin masses, so val(d) <= C_{1a}.

For wavelets, we'd need: every nonneg f has well-defined Daubechies
wavelet coefficients at level J satisfying the same constraints.
This is true: any f can be PROJECTED onto the wavelet subspace and
the projection coefficients satisfy A c = f_projected (some nonneg).
But the projection's autoconvolution differs from f's autoconvolution.

Specifically: f = f_J + g_J where f_J = projection at level J.  Then
  (f*f) = (f_J * f_J) + 2 (f_J * g_J) + (g_J * g_J).
And max_t (f*f)(t) is NOT bounded below by max_t (f_J * f_J)(t) in
general.  So a LB on val_W (basis) does NOT imply LB on C_{1a}.

CONCLUSION OF MATHEMATICAL ANALYSIS: Daubechies wavelet basis cannot
directly give an LB on C_{1a} via the moment / SDP framework, because:
  1. Restricting f to the basis only gives an UPPER BOUND on C_{1a}
     (not LB).
  2. Even if we relax the QP to a Shor SDP, the relaxation gives an LB
     only on val_W(basis), not on C_{1a}.
  3. Unlike the bin-mass approach (CS Lemma 1), there is no
     "windowed test-value inequality" connecting wavelet coefficients
     to a UNIVERSAL lower bound across all nonneg f.

THIS IS THE CRITICAL FINDING OF THE PROBE.

For COMPLETENESS, this probe v3 still computes the wavelet-Galerkin
val_W(N, J) on the SUBSPACE (= UPPER BOUND on C_{1a}, useful for UB
side), and also runs the Shor LB on val_W (on the subspace) for
comparison.
"""
import os, sys, json, time
import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import minimize
import pywt

START = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, 'run.log')
RES_PATH = os.path.join(HERE, 'results.json')


def log(msg, append=True):
    elapsed = time.time() - START
    line = f'[{elapsed:7.2f}s] {msg}'
    print(line, flush=True)
    mode = 'a' if append else 'w'
    with open(LOG_PATH, mode, encoding='utf-8') as f:
        f.write(line + '\n')


def make_scaling_function(N, levels=12):
    w = pywt.Wavelet(f'db{N}')
    phi, psi, x = w.wavefun(level=levels)
    return np.asarray(x), np.asarray(phi)


def autoconv_of_scaling(N, levels=12):
    x, phi = make_scaling_function(N, levels=levels)
    dx = x[1] - x[0]
    psi_auto = fftconvolve(phi, phi) * dx
    y = np.arange(len(psi_auto)) * dx
    return y, psi_auto


def build_M_t_func(N, J, levels=12):
    """Returns a function M(t) -> KxK Hankel matrix from cached Psi."""
    y_psi, psi_auto = autoconv_of_scaling(N, levels=levels)
    dx = y_psi[1] - y_psi[0]
    twoJ = 2 ** J
    k_min = int(np.ceil(-twoJ / 4.0))
    k_max = int(np.floor(twoJ / 4.0)) - (2 * N - 1)
    if k_max < k_min:
        return None, None, None, None
    K = k_max - k_min + 1

    def Psi_at(arg_arr):
        out = np.zeros_like(arg_arr, dtype=float)
        idx = arg_arr / dx
        i0 = np.floor(idx).astype(int)
        frac = idx - i0
        valid = (i0 >= 0) & (i0 < len(psi_auto) - 1)
        out[valid] = (1 - frac[valid]) * psi_auto[i0[valid]] + frac[valid] * psi_auto[i0[valid] + 1]
        return out

    def M_at_t(t):
        s_grid = np.arange(2 * K - 1)
        args = twoJ * t - 2 * k_min - s_grid
        psi_vals = Psi_at(args)
        ii = np.arange(K)
        return psi_vals[ii[:, None] + ii[None, :]]

    return M_at_t, K, k_min, k_max


def build_problem_v3(N, J, T_grid_n=401, P_pos_factor=20):
    log(f'Building (v3): N={N} J={J}')
    M_at_t, K, k_min, k_max = build_M_t_func(N, J)
    if K is None:
        raise ValueError(f'No valid k range for N={N}, J={J}')
    log(f'K={K}, k range [{k_min}, {k_max}]')

    T = np.linspace(-0.5, 0.5, T_grid_n)
    M_list = [M_at_t(t) for t in T]

    # Window matrices: for each window of L consecutive t values, compute
    # M_W = average over the window. Use only a few key window sizes.
    # Use windows centred on the most active t region near 0.
    M_W_list = []  # list of (W_label, M_W)
    twoJ = 2 ** J
    # The CS windows correspond to ell in [2, ..., d-1] in monomial Lasserre.
    # For wavelet: windows in t-space of length ell * (1/(2K)) -- mimicking bins.
    # Here, choose windows of varying length and position.
    window_lengths = [3, 5, 9, 17, 33]  # in t-grid points (odd for centring)
    for L in window_lengths:
        if L > T_grid_n:
            continue
        # slide windows of length L
        for start in range(0, T_grid_n - L + 1, max(1, L // 2)):
            M_avg = np.mean(M_list[start:start + L], axis=0)
            M_W_list.append((f'L{L}_s{start}', M_avg))
    # Also: pointwise constraints for fine resolution
    pointwise = [(f't{i}', M_list[i]) for i in range(0, T_grid_n, 4)]

    # Combined window set:
    all_W = M_W_list + pointwise
    log(f'  windows: {len(M_W_list)} averaged + {len(pointwise)} pointwise = {len(all_W)} total')

    # Positivity constraints
    x_phi, phi = make_scaling_function(N, levels=12)
    dx = x_phi[1] - x_phi[0]
    P_pos = max(P_pos_factor * K, 200)
    x_pos = np.linspace(-0.25, 0.25, P_pos)
    A = np.zeros((P_pos, K))
    ks = np.arange(k_min, k_max + 1)
    for j_idx, k in enumerate(ks):
        arg = twoJ * x_pos - k
        idx = (arg - x_phi[0]) / dx
        i0 = np.floor(idx).astype(int)
        frac = idx - i0
        valid = (i0 >= 0) & (i0 < len(phi) - 1)
        vals = np.zeros(P_pos)
        vals[valid] = (1 - frac[valid]) * phi[i0[valid]] + frac[valid] * phi[i0[valid] + 1]
        A[:, j_idx] = (twoJ ** 0.5) * vals

    int_target = twoJ ** 0.5

    return {
        'N': N, 'J': J, 'K': K, 'P': P_pos,
        'ks': ks, 'T': T,
        'M_list': M_list,
        'all_W': all_W,
        'A_pos': A,
        'int_target': int_target,
    }


def primal_minmax_QP_scipy(prob, n_starts=8, max_iter=400):
    """Solve the PRIMAL min-max QP via repeated SLSQP from random starts.
    Returns the minimum value found = upper bound on val_W = upper bound
    on C_{1a}|_subspace.
    """
    K = prob['K']
    M_list = prob['M_list']
    A_pos = prob['A_pos']
    int_target = prob['int_target']
    rng = np.random.default_rng(0)

    # objective: max_t c^T M(t) c
    def obj(c):
        vals = [c @ M @ c for M in M_list]
        return max(vals)

    # smooth approx via log-sum-exp
    def smooth_obj(c, beta=200):
        vals = np.array([c @ M @ c for M in M_list])
        m = vals.max()
        return m + (1 / beta) * np.log(np.exp(beta * (vals - m)).sum())

    def grad_smooth(c, beta=200):
        vals = np.array([c @ M @ c for M in M_list])
        w = np.exp(beta * (vals - vals.max()))
        w /= w.sum()
        # d/dc max ~ sum_t w_t * 2 M_t c
        g = np.zeros(K)
        for t_idx, M in enumerate(M_list):
            g += w[t_idx] * 2 * (M @ c)
        return g

    constraints = [
        {'type': 'eq', 'fun': lambda c: np.sum(c) - int_target},
        {'type': 'ineq', 'fun': lambda c: A_pos @ c},
    ]
    best_val = np.inf
    best_c = None
    for s in range(n_starts):
        # warm start: bell shape
        x_pos = np.linspace(-0.25, 0.25, K)
        if s == 0:
            f0 = np.maximum(1 - (4 * x_pos) ** 2, 1e-4)
        elif s == 1:
            f0 = np.maximum(np.cos(np.pi * 2 * x_pos), 1e-4)
        elif s == 2:
            f0 = 1 + 0 * x_pos
        else:
            f0 = np.maximum(0, rng.uniform(0, 2, K))
        c0, *_ = np.linalg.lstsq(A_pos, A_pos @ f0, rcond=None)
        # rescale to satisfy ∫f = 1
        c0 = c0 * (int_target / max(1e-9, np.sum(c0)))
        try:
            res = minimize(smooth_obj, c0, jac=grad_smooth,
                           method='SLSQP', constraints=constraints,
                           options={'maxiter': max_iter, 'ftol': 1e-9})
            if res.success or res.fun < best_val:
                v = obj(res.x)
                if v < best_val:
                    best_val = v
                    best_c = res.x.copy()
        except Exception as e:
            pass
    return {'V_upper_bound': float(best_val), 'c': best_c.tolist() if best_c is not None else None}


def shor_LB_with_RLT(prob, max_W=200):
    """Order-1 Shor SDP:
      min V
      s.t. tr(M_W Y) <= V for top max_W windows (largest c^T M_W c at primal opt)
           1^T c = beta, A c >= 0
           [[1, c'], [c, Y]] PSD
           RLT cuts: (Ac) (1^T c) and Sum-row constraints
    Gives a LOWER BOUND on val_W on the subspace (NOT on C_{1a} universally).
    """
    import cvxpy as cp
    K = prob['K']
    A_pos = prob['A_pos']
    int_target = prob['int_target']

    log('  Setting up Shor SDP w/ RLT cuts.')
    c = cp.Variable(K)
    Y = cp.Variable((K, K), symmetric=True)
    V = cp.Variable()

    Z = cp.bmat([
        [cp.reshape(cp.Constant(1.0), (1, 1), order='C'),
         cp.reshape(c, (1, K), order='C')],
        [cp.reshape(c, (K, 1), order='C'), Y]
    ])
    cons = [Z >> 0, cp.sum(c) == int_target, A_pos @ c >= 0]
    # RLT-1: (1^T c) (A c) = sum over k,l of A_{p,l} Y_{l,k} = beta * (A c)_p
    cons.append(A_pos @ Y @ np.ones(K) == int_target * (A_pos @ c))
    # RLT-2: (1^T c)^2 = beta^2
    cons.append(np.ones(K) @ Y @ np.ones(K) == int_target ** 2)

    # Use ALL windowed constraints (most informative)
    Ws = prob['all_W']
    if len(Ws) > max_W:
        # subsample but keep extremes
        idx = np.linspace(0, len(Ws) - 1, max_W, dtype=int)
        Ws = [Ws[i] for i in idx]
    log(f'  using {len(Ws)} windowed constraints.')
    for label, M_W in Ws:
        cons.append(cp.trace(M_W @ Y) <= V)

    prob_cp = cp.Problem(cp.Minimize(V), cons)
    try:
        prob_cp.solve(solver='CLARABEL', verbose=False, max_iter=20000)
        if prob_cp.status not in ('optimal', 'optimal_inaccurate'):
            raise RuntimeError(f'CLARABEL: {prob_cp.status}')
    except Exception as e:
        log(f'  CLARABEL failed ({e}); trying SCS')
        prob_cp.solve(solver='SCS', verbose=False, max_iters=5000)

    log(f'  status: {prob_cp.status}, V_LB = {prob_cp.value}')
    return {
        'status': prob_cp.status,
        'V_LB': float(prob_cp.value) if prob_cp.value is not None else None,
        'c': c.value.tolist() if c.value is not None else None,
    }


def main():
    log('Daubechies wavelet probe v3', append=False)
    log('============================')

    cs_baseline = 1.2802
    lasserre_baselines = {10: 1.231, 12: 1.271, 14: 1.284, 16: 1.319, 18: 1.319,
                          20: 1.319, 22: 1.319, 24: 1.319, 26: 1.319}

    configs = [
        (4, 5),  # K=10
        (4, 6),  # K=26
        (8, 6),  # K=18
    ]

    experiments = []
    best_LB = -np.inf
    best_UB = np.inf

    for (N, J) in configs:
        log('')
        log(f'=== N={N}, J={J} (db{N} at resolution {J}) ===')
        try:
            t0 = time.time()
            prob = build_problem_v3(N, J, T_grid_n=201)
            t_build = time.time() - t0

            t0 = time.time()
            primal = primal_minmax_QP_scipy(prob, n_starts=4, max_iter=200)
            UB = primal['V_upper_bound']
            t_primal = time.time() - t0

            t0 = time.time()
            dual = shor_LB_with_RLT(prob, max_W=150)
            LB = dual['V_LB']
            t_dual = time.time() - t0

            log(f'  primal UB on val_W (subspace): {UB:.4f}')
            log(f'  Shor LB on val_W (subspace):   {LB}')
            log(f'  build={t_build:.2f}s primal={t_primal:.2f}s dual={t_dual:.2f}s')

            entry = {
                'N': N, 'J': J, 'K': prob['K'],
                'primal_upper_bound': UB,
                'shor_lower_bound': LB,
                'build_time_sec': t_build,
                'primal_time_sec': t_primal,
                'dual_time_sec': t_dual,
                'shor_status': dual['status'],
                'note': 'val_W is on subspace; LB does NOT prove anything about C_{1a}.',
            }
            experiments.append(entry)
            if LB is not None and not np.isinf(LB) and LB > best_LB:
                best_LB = LB
            if UB is not None and not np.isinf(UB) and UB < best_UB:
                best_UB = UB
        except Exception as e:
            import traceback
            log(f'!!! ERROR: {type(e).__name__}: {e}')
            log(traceback.format_exc())
            experiments.append({'N': N, 'J': J, 'error': str(e)})

    # -- Summary and verdict --
    log('')
    log('=== Summary ===')
    log(f'best primal UB on val_W: {best_UB:.4f} (UB on C_{{1a}}|subspace; smaller subspace -> larger UB)')
    log(f'best Shor LB on val_W:   {best_LB}    (LB on val_W|subspace, NOT on C_{{1a}})')

    # The wavelet val_W is on the SUBSPACE. As J->inf, it converges to
    # C_{1a} from above. Compare to current UB 1.5029 (rigorous).
    # primal UB < 1.5029 at small J would suggest f exists with low max-autocorr
    #   - useful information for UPPER BOUND research, but UB rigour requires
    #     specific construction in Daubechies, not "computed by gradient descent".

    # MATHEMATICAL CONCLUSION:
    # Wavelet basis cannot directly give an LB on C_{1a} via SDP relaxation,
    # because (a) restricting f to the basis gives val_W >= C_{1a} (UB direction),
    # and (b) the relaxation Y >> c c^T allows c outside the basis-feasible
    # set, so V_LB is not bounded by anything related to C_{1a}.

    verdict_short = (
        'Daubechies wavelet basis is mathematically the WRONG direction for an LB '
        'on C_{1a}: restricting f to a finite-dim wavelet subspace gives val_W >= C_{1a} '
        '(an UPPER bound, smaller subspace = larger inf), and Shor relaxation of val_W '
        'gives only a LB on val_W itself, not on C_{1a}.  No analog of CS Lemma 1 '
        '(windowed test-value inequality) exists for wavelet coefficients that would '
        'connect the LB to C_{1a} universally.'
    )

    verdict_long = (
        f'The probe set up the autoconvolution (f*f)(t) = c^T M(t) c in the Daubechies '
        f'db_N scaling-function basis at level J, with M(t) Hankel from the autoconvolution '
        f'Psi(y) = (phi*phi)(y).  Three configurations were tested: (N=4,J=5,K=10), '
        f'(N=4,J=6,K=26), (N=8,J=6,K=18). For each: primal min-max QP (SLSQP from random '
        f'starts) gave an UPPER BOUND on val_W := inf_{{c valid}} max_t c^T M(t) c, and an '
        f'order-1 Shor SDP with RLT cuts gave a LOWER BOUND on val_W.  Best UB = '
        f'{best_UB:.4f}, best LB = {best_LB}.\n\n'
        f'CRITICAL MATHEMATICAL OBSERVATION: val_W is the inf over the wavelet subspace, '
        f'so val_W(N, J) >= C_{{1a}}.  Hence any LB on val_W is NOT a LB on C_{{1a}}.  '
        f'For the bin-mass Lasserre track in lasserre/, the analogous quantity val(d) '
        f'is rescued by the CS windowed inequality: ||f*f||_inf >= mu^T M_W mu for any '
        f'window W, where mu are bin masses of ANY nonneg f -- this gives val(d) <= C_{{1a}}.  '
        f'No such universal inequality is available for Daubechies wavelet coefficients '
        f'because the projection of f onto a wavelet subspace can have arbitrarily '
        f'different autoconvolution from f itself (when the high-frequency tail '
        f'g_J = f - f_J is nontrivial).\n\n'
        f'DEAD-END.  The wavelet basis is suitable for UB construction (find a smooth '
        f'f with low max-autocorrelation), not for LB certification.  The current '
        f'monomial Lasserre track is mathematically the right framework for LB.'
    )

    promising = False
    final = {
        'agent': 'F1_daubechies_wavelet',
        'approach': 'Daubechies scaling-function basis + Shor SDP relaxation of inf_c max_t c^T M(t) c',
        'math_correct': True,
        'best_lb_obtained': float(best_LB) if (best_LB is not None and not np.isinf(best_LB)) else None,
        'vs_1_2802': 'unknown_irrelevant',
        'vs_lasserre_baseline': lasserre_baselines.get(10, 1.231),
        'promising': promising,
        'verdict_short': verdict_short,
        'verdict_long': verdict_long,
        'next_steps_if_promising': [],
        'compute_time_sec': time.time() - START,
        'files_created': ['probe.py', 'probe_v2.py', 'probe_v3.py', 'run.log',
                          'results.json', 'analysis.md'],
        'experiments': experiments,
        'extra_metrics': {
            'best_primal_val_W_UB_on_subspace': float(best_UB) if (best_UB is not None and not np.isinf(best_UB)) else None,
            'best_shor_LB_on_val_W_subspace': float(best_LB) if (best_LB is not None and not np.isinf(best_LB)) else None,
            'critical_note': (
                'val_W is on the WAVELET SUBSPACE only. val_W(N,J) >= C_{1a} (UB '
                'direction). Shor LB on val_W is NOT a LB on C_{1a}. No analog '
                'of CS Lemma 1 windowed inequality exists for wavelet coefficients, '
                'so this approach cannot produce a universal LB on C_{1a}.'
            ),
        },
    }

    with open(RES_PATH, 'w', encoding='utf-8') as fh:
        json.dump(final, fh, indent=2)
    log('')
    log(f'wrote {RES_PATH}')
    log(f'verdict: {verdict_short}')


if __name__ == '__main__':
    main()
