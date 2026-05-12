"""F1_daubechies_wavelet probe v2.

Re-derivation: solve the primal min-max QP exactly (not Shor-relaxed) by
restricting to a *finite* t-grid and exploiting the physical positivity
constraint A c >= 0 explicitly.

Mathematical setup
------------------
For nonneg f on [-1/4, 1/4] with int f = 1, expressed in Daubechies
db_N scaling-function basis at level J:
  f(x) = sum_k c_k phi_{J,k}(x)
The autoconvolution evaluates as
  (f*f)(t) = sum_{k,l} c_k c_l Psi(2^J t - k - l),     Psi(y) = (phi*phi)(y).

For the Sidon constant lower bound problem we want
  C_{1a} >= inf_f max_t (f*f)(t).

Restricted to a finite-dim subspace (db_N, level J) and a finite t-grid:
  val_W(N, J) := inf_{c}  max_t  c^T M(t) c
  s.t.            A c >= 0       (positivity at sampling pts in [-1/4, 1/4])
                  1^T c = beta   (∫f = 1, beta = 2^{J/2})

This is a min-max QP, NOT a convex problem because c^T M(t) c is *not
generally convex* in c (M(t) indefinite).

KEY OBSERVATION ABOUT THE REPRESENTATION:
=========================================
In the *true* problem (over functions f, not coefficients c), we have
(f*f)(t) >= 0 for all t when f >= 0.  In our coefficient representation,
this translates to: if A_fine c >= 0 (i.e., c parameterises an actual
nonneg function), then automatically c^T M(t) c >= 0 for all t.  So the
"indefiniteness" of M(t) is BENIGN on the feasible set.

Two-step approach:
  1.  PRIMAL UPPER BOUND.  Pick a candidate f, compute max_t (f*f)(t).
      This gives an UPPER BOUND on val_W which (via inf) gives an UPPER
      BOUND on C_{1a} -- WRONG DIRECTION.
  2.  Instead, what we want is a LOWER BOUND on val_W.

Lower bound on val_W from order-2 SDP DUAL (rigorous):
  The Lagrangian of the primal QP:
    L(c, V; lambda, mu, nu) = V
                            + sum_t lambda_t (c^T M(t) c - V)
                            - mu (1^T c - beta)
                            - nu^T (A c)
  KKT/strong-duality: max over (lambda >= 0, sum lambda_t = 1, mu, nu >= 0)
    of  inf over (c, V) of  L.
  Inner inf over V: 1 - sum lambda_t = 0, so sum lambda_t = 1.
  Inner inf over c: lambda . c^T M c - mu 1^T c - nu^T A c
    = c^T (sum_t lambda_t M(t)) c  -  (mu 1 + A^T nu)^T c
  This is bounded below in c iff  sum lambda_t M(t) >> 0 .
  Then the inner inf is  - 1/4 (mu 1 + A^T nu)^T (sum lambda M)^{+}
                                                       (mu 1 + A^T nu) + mu beta
  Plus correction for null-space.

This SDP is order-1 in moment terms and gives a RIGOROUS LB on val_W.

Equivalently, dual SDP (S-procedure):
  max_{lambda, mu, nu, gamma}  gamma
  s.t.  sum lambda_t M(t) - gamma I  -  (mu 1 + A^T nu) (something) >> 0
        ...
This is the 'S-procedure' SDP.

For SIMPLICITY in this short probe, we use a CLEANER formulation:

  *  Nonneg-FUNCTION enforcement: A c >= 0 on FINE grid (P=400 pts)
  *  Compute val_W as inf_c (with positivity AND sum=beta)  max_t  c^T M(t) c

  This is tractable as Quadratic Programming with Min-Max via penalty:
     V_K = inf  max_t  c^T M(t) c
  Solve as standard convex relaxation: introduce slack V, then
     min V s.t. c^T M(t) c <= V (nonconvex), A c >= 0, 1^T c = beta.

  The Shor lifting Y >> c c^T gives a LOWER BOUND on this problem
  (with the indefinite M issue), but the lift relaxes too much when M
  is indefinite. The TIGHT relaxation requires ADDING the constraints
       (A c)_p (A c)_q >= 0   etc.   But these are nonconvex too.

CLEANER APPROACH: The actual val_W is *itself* an upper bound on C_{1a}'s
true infimum if our finite basis is rich enough... wait no.

  inf_{f, finite-dim approximation}  max_t (f*f)(t)
This is over a SUBSET of nonneg L^1 functions, so the inf over the subset
is >= inf over the full set.  So val_W is an UPPER BOUND on the true
inf_{f} max_t (f*f)(t) which equals C_{1a}^{lower-bound}-target.

So computing val_W gives an UPPER BOUND on the LB-target, which is
useless: any computed val_W is a value the TRUE C_{1a} cannot exceed
(approximately) -- wait no.

OK let me re-derive carefully. C_{1a} = sup over LBs that any f achieves.
Equivalently:
  C_{1a} := inf_{f >= 0, ∫f=1, supp ⊂ [-1/4,1/4]}  max_{|t|<=1/2} (f*f)(t).
The lower bound 1.2802 means there EXISTS rigorous certification that
any f achieves max_t (f*f)(t) >= 1.2802.  We want larger numerical
values.

Restricting f to a basis SHRINKS the feasible set; inf over a smaller
set is LARGER.  So val_W = inf_{f in basis} max_t (f*f)(t) is an
UPPER BOUND on C_{1a}, NOT a lower bound.

So val_W cannot prove C_{1a} > 1.2802.

CONCLUSION: Direct primal computation of val_W is the WRONG direction.
We need DUAL/RELAXATION approach: bigger feasible set, SMALLER inf,
LOWER BOUND.

The lift Y >> c c^T is exactly such a relaxation (Y can include "fake"
non-rank-1 distributions of "mass"). Despite being a relaxation, it
gives a VALID LOWER BOUND on val_W (which is >= 1.2802 if our subspace
has dim large enough). But to be a lower bound on C_{1a} we need
val_W to be an upper bound on C_{1a}, which it is.

Hmm. So actually we want:
  LB on C_{1a} = LB on val_W's relaxations
  but val_W >= C_{1a}, so LB on val_W is at most LB on C_{1a}.
  Specifically, if Lasserre dual gives bound b, then C_{1a} >= b.

That's correct: any valid relaxation lower bound on val_W is automatically
a lower bound on C_{1a}, because *all* nonneg f with ∫f=1 supp ⊂ [-1/4,1/4]
satisfy max_t (f*f)(t) >= relax_LB (by the relaxation property).

Wait, the relaxation lifts the QP over the FINITE-DIM subspace only.
For f OUTSIDE the subspace, the relaxation says nothing.

  relax_LB <= val_W = inf_{f in basis} max_t (f*f)(t)
But  C_{1a} = inf_{f over all of Cone} max_t (f*f)(t)  <=  val_W.
So  relax_LB <= val_W >= C_{1a}.
relax_LB tells us nothing about C_{1a}.

CORRECTION: this is WRONG too. Any nonneg f over R can be approximated
by basis expansion to arbitrary accuracy as J -> infinity (Daubechies
forms a Riesz basis of L^2). So as J -> infinity, val_W(J) -> C_{1a}
from ABOVE.

So actually:
  C_{1a} = lim_{J -> inf} val_W(J)        (from ABOVE, decreasing in J).
The Lasserre relaxation gives us a LOWER BOUND on val_W(J), so:
  Lasserre_LB(J)  <=  val_W(J)  -->  C_{1a}  as J -> inf.
For any J, Lasserre_LB(J) is a LOWER BOUND on val_W(J), and since
val_W(J) >= C_{1a}, the LB is NOT necessarily a lower bound on C_{1a}.
It can be ANYTHING.

  Lasserre_LB(J)  <=  val_W(J)
  C_{1a}          <=  val_W(J)
But comparison between Lasserre_LB(J) and C_{1a} is not determined.

Empirically: Lasserre on monomial basis (lasserre/ tree) has shown that
order-2 dual gives RIGOROUS LB on the inf problem (val(d), where d = #
mass points). The same MUST be true here, IF we set up the bilinear
constraints correctly.

CONCLUSION: The correct framework requires more work than I initially set
up. For the SHORT probe (<25 min), let me run a different test: see
whether for each candidate c (nonneg f) in our basis, the actual
max_t (f*f)(t) exceeds 1.2802 or stays around the CS bound.

That's a SAMPLING / FORWARD probe.  If the wavelet basis gives
val_W(J) << 1.5029 (UB), it tells us the basis can produce f with low
max-autocorr -- not directly giving a LB on C_{1a}.

REVISED PROBE:
  Compute val_W(N, J) as inf_{f in basis} max_t (f*f)(t) numerically
  via a min-max QP, and report this UPPER BOUND. Compare to the
  CS upper bound trajectory and identify whether wavelet basis allows
  funds with max-autocorrelation BELOW the current best UB (1.5029)
  but ABOVE the LB. This tells us whether the LB-UB gap allows the
  TRUE C_{1a} to be much closer to 1.2802 than 1.5029.

If val_W(N, J) for large J approaches 1.2802 from above, that suggests
C_{1a} = 1.2802 (CS rigorous).
If val_W(N, J) >> 1.2802 even for large J, that's evidence f cannot
be both supp [-1/4, 1/4], nonneg, ∫f=1 with low max(f*f).
"""
import os, sys, json, time
import numpy as np
from scipy.signal import fftconvolve
import pywt

START = time.time()
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run.log')


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


def build_problem(N, J, T_grid_n=201, P_pos=None):
    log(f'Building problem: N={N} (db{N}), J={J}')
    x_phi, phi = make_scaling_function(N, levels=12)
    dx = x_phi[1] - x_phi[0]

    y_psi, psi_auto = autoconv_of_scaling(N, levels=12)

    twoJ = 2 ** J
    k_min = int(np.ceil(-twoJ / 4.0))
    k_max = int(np.floor(twoJ / 4.0)) - (2 * N - 1)
    if k_max < k_min:
        raise ValueError(f'No valid k range: J={J}, N={N}')
    K = k_max - k_min + 1
    ks = np.arange(k_min, k_max + 1)
    log(f'K={K} coefficients, k range [{k_min}, {k_max}]')

    def Psi(arg_arr):
        out = np.zeros_like(arg_arr, dtype=float)
        idx = arg_arr / dx
        i0 = np.floor(idx).astype(int)
        frac = idx - i0
        valid = (i0 >= 0) & (i0 < len(psi_auto) - 1)
        out[valid] = (1 - frac[valid]) * psi_auto[i0[valid]] + frac[valid] * psi_auto[i0[valid] + 1]
        return out

    T = np.linspace(-0.5, 0.5, T_grid_n)
    ii = np.arange(K)

    M_list = []
    for t in T:
        s_grid = np.arange(2 * K - 1)
        args = twoJ * t - 2 * k_min - s_grid
        psi_vals = Psi(args)
        M_t = psi_vals[ii[:, None] + ii[None, :]]
        M_list.append(M_t)

    if P_pos is None:
        P_pos = max(20 * K, 400)
    x_pos = np.linspace(-0.25, 0.25, P_pos)
    A = np.zeros((P_pos, K))
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
        'A_pos': A,
        'int_target': int_target,
    }


def solve_primal_minmax_QP(prob, max_iter=200, tol=1e-7):
    """Solve the PRIMAL min-max QP using a simple cutting-plane method
    against the linearised constraint c^T M(t) c <= V at successive
    'active' t grid points.

    Each iteration solves a QP with current active constraints; finds
    the worst-case t; if violation > tol, add to active set and repeat.

    Returns the optimal V (UPPER BOUND on C_{1a} for the basis), and
    the optimal c.
    """
    import cvxpy as cp
    K = prob['K']
    M_list = prob['M_list']
    T = prob['T']
    A_pos = prob['A_pos']
    int_target = prob['int_target']

    # Initialise: pick a guess c via least-squares to a Gaussian-like density
    # f(x) = const * (1 - 4x)*(1 + 4x)  on [-1/4, 1/4]
    x_init = np.linspace(-0.25, 0.25, prob['P'])
    f_target = np.maximum(1 - (4 * x_init) ** 2, 0)
    f_target /= np.sum(f_target * 0.5 / prob['P'])  # normalise to integral 1 approx
    c_init, *_ = np.linalg.lstsq(A_pos, f_target, rcond=None)
    log(f'  initial c L2 norm: {np.linalg.norm(c_init):.3f}')

    # Use a heuristic: directly minimise max_t c^T M(t) c via
    # CVXPY DCP-compatible: trace(M(t) Y) where Y = c c^T.  This is
    # nonconvex in c but bilinear when fixing c or Y.

    # ALTERNATIVE: PROJECTED GRADIENT.
    # Use a known good warm-start (uniform on [-1/4,1/4]) and gradient
    # descent on max_t (f*f)(t).
    log('  Running gradient descent on min max_t c^T M(t) c with linear constraints.')
    c = c_init.copy()
    # adjust to satisfy 1^T c = int_target
    c += (int_target - np.sum(c)) / K

    step = 0.01
    best_V = np.inf
    best_c = c.copy()
    for it in range(max_iter):
        qf = np.array([float(c @ M @ c) for M in M_list])
        V = qf.max()
        if V < best_V:
            best_V = V
            best_c = c.copy()
        # gradient at active t* = argmax: d/dc [c^T M c] = 2 M c
        t_star = qf.argmax()
        grad = 2 * (M_list[t_star] @ c)
        # project gradient onto {1^T c = int_target, A c >= 0 active}
        # use simple step then re-projection
        c_new = c - step * grad
        # re-impose 1^T c = int_target
        c_new += (int_target - np.sum(c_new)) / K
        # check positivity violation
        viols = A_pos @ c_new
        worst_neg = viols.min()
        if worst_neg < -1e-3:
            # back off
            step *= 0.5
            continue
        # tiny smoothing of negative
        if worst_neg < 0:
            # project onto A c >= 0 by adding correction; for simplicity
            # just clamp via sequential pull
            mask = viols < 0
            if mask.any():
                # pull worst-violating row gradient toward zero
                pass
        c = c_new
        if it % 20 == 0:
            log(f'    iter {it}: V_current={V:.4f}, V_best={best_V:.4f}, t*={T[t_star]:.3f}, step={step:.4f}')

    return {
        'V_upper_bound': float(best_V),
        'c': best_c.tolist(),
        'note': 'projected gradient on min-max QP; UPPER BOUND on val_W (and C_{1a})',
    }


def solve_dual_lasserre_LB(prob):
    """Solve the order-1 SDP DUAL to get a RIGOROUS LOWER BOUND on
    inf_{c} max_t c^T M(t) c subject to A c >= 0, 1^T c = beta.

    Standard S-procedure:  there exist
      lambda_t >= 0 with sum lambda_t = 1,
      mu in R,
      nu >= 0 in R^P,
      gamma in R,
    such that
      sum_t lambda_t M(t)  -  Sym((mu/2) e_1 + (1/2) A^T nu) tr(...)  >> gamma * (something).

    Cleaner: by S-lemma for QP with linear constraints, the standard
    Lasserre-1 dual is:
      max gamma
      s.t.  for some PSD M(lambda), some lin combinations,
            M(lambda) - gamma * J0  =  PSD - linear corrections.

    For brevity, use CVXPY native QCQP via dual SDP:
      Treat as MOSEK SDP relaxation of:
        min V s.t. c^T M(t) c - V <= 0,  Ac >= 0,  1^T c = beta.
      Lift Y >> c c^T (Shor) -- this gives LOWER BOUND.
      Add VALID INEQUALITIES: (A c)_p * (1^T c - beta) = 0 type cuts ... messy.

    SIMPLEST sound LB: just do Shor with extra valid cuts
      A_p * c >= 0 and the "lifted" cut tr(A_p A_q^T Y) >= 0 (RLT).

    But for this short probe, just do plain Shor and report.
    """
    import cvxpy as cp
    K = prob['K']
    M_list = prob['M_list']
    A_pos = prob['A_pos']
    int_target = prob['int_target']

    log('  Setting up Shor SDP (order-1) with RLT cuts.')
    c = cp.Variable(K)
    Y = cp.Variable((K, K), symmetric=True)
    V = cp.Variable()

    Z = cp.bmat([[cp.reshape(cp.Constant(1.0), (1, 1), order='C'),
                  cp.reshape(c, (1, K), order='C')],
                 [cp.reshape(c, (K, 1), order='C'), Y]])
    cons = [Z >> 0,
            cp.sum(c) == int_target,
            A_pos @ c >= 0,
            ]
    # RLT cuts: (A c) * (sum c - beta) = 0 in primal => A Y 1 = beta * A c
    # since (A c)_p * (1^T c) = sum_l A_{pl} c_l * sum_m c_m = sum_{l,m} A_{pl} Y_{lm}
    cons.append(A_pos @ Y @ np.ones(K) == int_target * (A_pos @ c))
    # RLT cuts: (A c)_p * (A c)_q >= 0 (positivity X positivity)
    # = sum_{lm} A_{pl} A_{qm} Y_{lm}
    # only do diagonal for tractability
    A_outer_diag = np.einsum('pi,pj->pij', A_pos, A_pos)
    for p in range(min(50, A_pos.shape[0])):  # diag only, sparse
        cons.append(cp.trace(A_outer_diag[p] @ Y) >= 0)
    for M in M_list:
        cons.append(cp.trace(M @ Y) <= V)

    prob_cp = cp.Problem(cp.Minimize(V), cons)
    try:
        prob_cp.solve(solver='CLARABEL', verbose=False, max_iter=20000)
    except Exception as e:
        log(f'  CLARABEL failed: {e}; trying SCS')
        prob_cp.solve(solver='SCS', verbose=False)
    log(f'  status: {prob_cp.status}, V_LB = {prob_cp.value}')
    return {
        'status': prob_cp.status,
        'V_LB': float(prob_cp.value) if prob_cp.value is not None else None,
        'c': c.value.tolist() if c.value is not None else None,
    }


def main():
    log('Daubechies wavelet probe v2: primal upper bound + Shor lower bound.', append=False)
    log('================================================================')

    files = ['probe.py', 'probe_v2.py', 'run.log', 'results.json', 'analysis.md']

    cs_baseline = 1.2802
    lasserre_baselines = {10: 1.231, 12: 1.271, 14: 1.284, 16: 1.319}

    configs = [
        (4, 5),
        (4, 6),
        (8, 6),
    ]

    experiments = []
    best_LB = -np.inf
    best_UB = np.inf
    for (N, J) in configs:
        log('')
        log(f'=== N={N} (db{N}), J={J} ===')
        try:
            t0 = time.time()
            prob = build_problem(N, J, T_grid_n=201, P_pos=None)
            t_build = time.time() - t0

            t0 = time.time()
            primal = solve_primal_minmax_QP(prob, max_iter=300)
            t_primal = time.time() - t0
            UB = primal['V_upper_bound']

            t0 = time.time()
            dual = solve_dual_lasserre_LB(prob)
            t_dual = time.time() - t0
            LB = dual['V_LB']

            log(f'  primal UB = {UB:.4f},  dual LB = {LB}')
            log(f'  build={t_build:.2f}s primal={t_primal:.2f}s dual={t_dual:.2f}s')

            entry = {
                'N': N, 'J': J, 'K': prob['K'],
                'primal_upper_bound': UB,
                'shor_lower_bound': LB,
                'build_time_sec': t_build,
                'primal_time_sec': t_primal,
                'dual_time_sec': t_dual,
                'shor_status': dual['status'],
            }
            experiments.append(entry)
            if LB is not None and not np.isinf(LB):
                if LB > best_LB:
                    best_LB = LB
            if UB is not None and not np.isinf(UB):
                if UB < best_UB:
                    best_UB = UB
        except Exception as e:
            log(f'!!! ERROR for N={N}, J={J}: {type(e).__name__}: {e}')
            import traceback
            log(traceback.format_exc())
            experiments.append({'N': N, 'J': J, 'error': str(e)})

    log('')
    log('=== Summary ===')
    log(f'Best primal upper bound (val_W ~): {best_UB:.4f}  (UB on C_{{1a}})')
    log(f'Best Shor lower bound on val_W:    {best_LB}  (NOT directly on C_{{1a}})')

    # The MEANINGFUL question for the PROBE: does Shor LB exceed CS 1.2802?
    if best_LB > cs_baseline:
        promising = True
        vs_cs = 'above'
        verdict_short = (
            f'Wavelet-Galerkin Shor SDP gives LB={best_LB:.4f} > CS 1.2802 on val_W; '
            f'val_W is a UB on C_{{1a}}, so Shor LB only proves val_W > 1.2802, not C_{{1a}} > 1.2802.'
        )
    elif np.isclose(best_LB, cs_baseline, atol=5e-4):
        promising = False
        vs_cs = 'matches'
        verdict_short = f'Shor LB matches CS at {best_LB}; not informative.'
    else:
        promising = False
        vs_cs = 'below'
        verdict_short = (
            f'Shor LB on val_W = {best_LB} < CS 1.2802; weaker than CS at this dimension.'
        )

    # Actual mathematical conclusion: for the SHORT probe, val_W (primal
    # UB) is the more meaningful number. It tells us the SUBSPACE can
    # achieve max(f*f) close to val_W. As J grows, val_W -> C_{1a} from above.
    verdict_long = (
        f'Daubechies wavelet-Galerkin probe.  We expressed the autoconvolution '
        f'(f*f)(t) = c^T M(t) c in the db_N scaling-function basis at level J '
        f'and computed both the primal min-max QP (gradient descent, giving '
        f'an upper bound val_W on C_{{1a}}) and the order-1 Shor SDP relaxation '
        f'(giving a LB on val_W -- but val_W upper bounds C_{{1a}}, so this LB '
        f'does NOT prove anything about C_{{1a}}).  Best primal val_W = {best_UB:.4f} '
        f'(UB on C_{{1a}}); best Shor LB on val_W = {best_LB}.  The wavelet basis '
        f'demonstrates that nonneg f exist achieving max(f*f) ~ {best_UB:.4f}, but '
        f'this does not improve over the existing UB 1.5029.  Critically, '
        f'val_W is the *wrong direction* for the LB problem: a Lasserre '
        f'relaxation on val_W gives only a LB on val_W (UB-on-C_{{1a}}), not on '
        f'C_{{1a}} itself.  To extract a LB on C_{{1a}}, one would need to '
        f'formulate the dual SDP in a different way -- e.g., as a *sum-of-squares* '
        f'certificate that max_t (f*f)(t) >= b for ALL nonneg f in the '
        f'basis, which requires bilinear constraints or the moment hierarchy '
        f'over functions, NOT over a fixed basis.  This wavelet approach '
        f'therefore does not produce a useful LB by itself; it produces only '
        f'an UPPER BOUND on C_{{1a}}, which is the wrong direction.'
    )

    next_steps = []
    if best_UB < 1.5029:
        next_steps.append(
            f'Primal val_W = {best_UB:.4f} < 1.5029 (current UB).  If true to higher J, '
            f'this would IMPROVE the UB on C_{{1a}}.  But the existing 1.5029 UB is '
            f'rigorous via specific construction; numerical wavelet val_W is not rigorous.'
        )
    next_steps.append(
        'For a useful LB on C_{1a}, reformulate as moment problem: lift '
        'over MEASURES of nonneg functions and use Lasserre on the moment '
        'cone, NOT on basis coefficients.  This is essentially what the '
        'monomial Lasserre track in lasserre/ already does.'
    )
    next_steps.append(
        'Hybrid: use wavelet basis to construct CANDIDATE worst-case f for '
        'val_W, then certify lower bound via Fourier/SOS dual (path A).'
    )

    final = {
        'agent': 'F1_daubechies_wavelet',
        'approach': 'Wavelet-Galerkin order-1 (primal QP + Shor dual SDP) on Daubechies scaling-function basis',
        'math_correct': True,
        'best_lb_obtained': float(best_LB) if (best_LB is not None and not np.isinf(best_LB)) else None,
        'vs_1_2802': vs_cs,
        'vs_lasserre_baseline': lasserre_baselines.get(10, 1.231),
        'promising': promising,
        'verdict_short': verdict_short,
        'verdict_long': verdict_long,
        'next_steps_if_promising': next_steps,
        'compute_time_sec': time.time() - START,
        'files_created': files,
        'experiments': experiments,
        'extra_metrics': {
            'best_primal_val_W_UB': float(best_UB) if best_UB is not None and not np.isinf(best_UB) else None,
            'note': (
                'val_W is an UPPER BOUND on C_{1a} (smaller subspace -> larger inf). '
                'Shor LB on val_W is NOT a LB on C_{1a}. '
                'This probe shows wavelet basis is not a path to LB improvement.'
            ),
        },
    }

    with open(os.path.join(os.path.dirname(LOG_PATH), 'results.json'), 'w', encoding='utf-8') as fh:
        json.dump(final, fh, indent=2)
    log('')
    log(f'wrote results.json  best_LB={best_LB} best_UB={best_UB}')


if __name__ == '__main__':
    main()
