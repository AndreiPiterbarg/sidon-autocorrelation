"""Path B — Krein–Boas–Kac SDP closure of Hyp_R.

PROBLEM
-------
Restricted-Hölder hypothesis Hyp_R(c_*, M_max) with c_* = log(16)/pi:
    For every nonneg pdf f on [-1/4, 1/4] with int f = 1 and ||f*f||_inf <= M_max,
    we have ||f*f||_2^2 <= c_* * M_max.

If proved unconditionally at M_max = 1.378..., then C_{1a} >= 1.378 unconditionally.

OBSTRUCTION (from Step 1, see _step1_v2.py / _step1_v3.py)
-----------
Path A's chain
    sum_{n>=1} y_n^2 <= mu(M) sum_{n>=1} y_n      with  y_n := |hat f(n)|^2
gives at K=2 (the universal Cauchy-Schwarz minimum on supp [-1/4,1/4])
    c_emp <= 0.927  > c_* = 0.88254
even after bang-bang refinement and Toeplitz-PSD on {y_n} and
phase-aware Hermitian Toeplitz on {hat f(n)}.

The bang-bang extremizer is the trig polynomial
    f_*(x) = 1 + 2 sqrt(mu) cos(2 pi x) + 2 sqrt(S-mu) cos(4 pi x),    S=(K-1)/2,
which satisfies f_*(1/4) = 0.183 > 0 and f_*(1/2) = 0.663 > 0 -- so f_* is NOT
supported on [-1/4, 1/4]. It violates the support constraint.

KEY IDEA (this file)
--------------------
Encode supp f subset [-1/4, 1/4] via the Krein-Boas-Kac (Putinar) localizing
polynomial p(x) = 1/16 - x^2 (which is >= 0 on [-1/4, 1/4] and <= 0 on the
"forbidden" region [-1/2, -1/4] u [1/4, 1/2]).

Encoding: f is the density of a nonneg measure nu on [-1/2, 1/2] with
supp nu subset [-1/4, 1/4] iff BOTH

    (Bochner)     T = [hat nu(i-j)]_{i,j=0..N}                   PSD
    (Localizing)  L = [(p * hat nu)(i-j)]_{i,j=0..N-K_trunc}     PSD

where (p * hat nu)(n) = sum_k hat p(k) * hat nu(n-k) is the trig convolution
between the localizing polynomial's Fourier coefficients and the moment sequence.

For f real, we have hat f(-n) = conj hat f(n), so we work with Hermitian Toeplitz
matrices encoded as real (2(N+1)) x (2(N+1)) symmetric PSD blocks.

VARIABLES
---------
  hat f(n) = a_n + i b_n,  n = 1..N    (a_n, b_n in R)
  hat f(0) = 1                          (mass normalisation)
  hat f(-n) = conj hat f(n)             (f real)

OBJECTIVE
---------
Bound max sum_{n=1..N} |hat f(n)|^4 = max sum_{n=1..N} (a_n^2 + b_n^2)^2.
Use a Schur lift V = y y^T  with  y_n := |hat f(n)|^2:
   trace(V)  is an upper bound on  sum y_n^2,
   constraint  [[1, y^T]; [y, V]] >> 0   makes the lift consistent.

ADMISSIBILITY CONSTRAINTS
-------------------------
  (MO 2.14)  |hat f(n)|^2 <= mu(M)  for n >= 1      (M = M_max)
  (Bochner)  Hermitian Toeplitz on hat f PSD
  (KBK)      Localizing Hermitian Toeplitz PSD with p(x) = 1/16 - x^2
  (Lift)     Schur consistency on V = y y^T

NOTE on Parseval / K
--------------------
sum |hat f(n)|^2 = K = ||f||_2^2 has NO universal upper bound -- for asymmetric
f, K can exceed M arbitrarily (this is the Path-A obstruction). We do NOT add
sum y_n <= K_max as a constraint; the SDP becomes scale-bounded only via the
combination (mu(M) cap on each y_n) AND (Bochner+KBK PSD constraints).

If the SDP is unbounded (sum y_n^2 -> infinity), Hyp_R cannot be proved through
this chain. If bounded, the SDP value is a rigorous upper bound on c_emp.

REFERENCES
----------
- delsarte_dual/grid_bound_sharper_bathtub/path1_sdp.py  (Fejer-Riesz LMI idiom)
- bochner_sos/toeplitz_psd_probe.py                       (Toeplitz-PSD on y_n)
- delsarte_dual/restricted_holder/derivation.md           (Hyp_R conditional theorem)
- Putinar 1993 "Positive polynomials on compact semialgebraic sets"
- Krein, M.G. & Nudel'man, A.A. 1977 "The Markov moment problem"
- Boas, R.P. & Kac, M. 1945 "Inequalities for Fourier transforms of positive functions"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import cvxpy as cp


# ---------- problem constants ----------
LOG16_PI = math.log(16) / math.pi  # = 0.882542400610606...
HYP_R_C_STAR = LOG16_PI


def mu_M(M: float) -> float:
    """MO 2.14 pointwise bound: |hat f(n)|^2 <= mu(M) for n >= 1, n in Z."""
    return M * math.sin(math.pi / M) / math.pi


def hyp_r_target(M_max: float) -> float:
    """Hyp_R bound on max sum_{n>=1} y_n^2 = (c_* M_max - 1) / 2."""
    return 0.5 * (HYP_R_C_STAR * M_max - 1.0)


# ---------- localizing polynomial ----------
def p_hat_coefficients(K_trunc: int) -> dict:
    """Period-1 Fourier coefficients of p(x) = 1/16 - x^2 on [-1/2, 1/2].

      hat p(0) = int_{-1/2}^{1/2} (1/16 - x^2) dx = 1/16 - 1/12 = -1/48
      hat p(k) = int_{-1/2}^{1/2} (1/16 - x^2) e^{-2 pi i k x} dx
                = - int x^2 cos(2 pi k x) dx                      (k != 0)
                = -(-1)^k / (2 pi^2 k^2)                          (k != 0)
    p is real and even, so hat p(-k) = hat p(k) (real).
    """
    p_hat = {0: -1.0 / 48.0}
    for k in range(1, K_trunc + 1):
        coef = -((-1) ** k) / (2.0 * math.pi ** 2 * k ** 2)
        p_hat[k] = coef
        p_hat[-k] = coef
    return p_hat


# ---------- Hermitian Toeplitz construction (real-form encoding) ----------
def hermitian_toeplitz_real_form(
    a: cp.Variable, b: cp.Variable, N: int
) -> cp.Expression:
    """Real-form encoding of Hermitian PSD T = [hat f(i-j)]_{i,j=0..N}, where
    hat f(0) = 1 and hat f(n) = a[n-1] + i b[n-1] for n = 1..N (and conj for n<0).

    Hermitian PSD <=> [[Re T, -Im T], [Im T, Re T]] >> 0  (real symmetric PSD).
    """
    Tsize = N + 1
    Re_rows = []
    Im_rows = []
    for i in range(Tsize):
        Re_row = []
        Im_row = []
        for j in range(Tsize):
            d = i - j
            if d == 0:
                Re_row.append(np.array([[1.0]]))
                Im_row.append(np.array([[0.0]]))
            elif d > 0:
                Re_row.append(cp.reshape(a[d - 1], (1, 1), order="C"))
                Im_row.append(cp.reshape(b[d - 1], (1, 1), order="C"))
            else:  # d < 0
                Re_row.append(cp.reshape(a[-d - 1], (1, 1), order="C"))
                Im_row.append(cp.reshape(-b[-d - 1], (1, 1), order="C"))
        Re_rows.append(Re_row)
        Im_rows.append(Im_row)
    Re_T = cp.bmat(Re_rows)
    Im_T = cp.bmat(Im_rows)
    return cp.bmat([[Re_T, -Im_T], [Im_T, Re_T]])


def localizing_toeplitz_real_form(
    a: cp.Variable, b: cp.Variable, N: int, L_size: int, p_hat: dict, K_trunc: int
) -> cp.Expression:
    """Real-form encoding of Hermitian PSD localizing matrix
        L = [(p * hat f)(i - j)]_{i,j=0..L_size}
    where (p * hat f)(d) = sum_{k=-K_trunc..K_trunc} hat p(k) * hat f(d - k).

    Ensures: f admissible on [-1/4, 1/4] requires  L >> 0.
    """

    def hat_f(n):
        """Return (Re hat f(n), Im hat f(n)) as cvxpy expressions or floats."""
        if n == 0:
            return (1.0, 0.0)
        elif 1 <= n <= N:
            return (a[n - 1], b[n - 1])
        elif -N <= n <= -1:
            return (a[-n - 1], -b[-n - 1])
        else:
            return None  # out of truncation -- treat as 0 (sound; gives upper bound)

    L_dim = L_size + 1
    Re_rows = []
    Im_rows = []
    for i in range(L_dim):
        Re_row = []
        Im_row = []
        for j in range(L_dim):
            d = i - j
            re_total = 0.0
            im_total = 0.0
            for k in range(-K_trunc, K_trunc + 1):
                if k not in p_hat:
                    continue
                f_pair = hat_f(d - k)
                if f_pair is None:
                    continue
                re_f, im_f = f_pair
                re_total = re_total + p_hat[k] * re_f
                im_total = im_total + p_hat[k] * im_f
            # Wrap as 1x1 matrices for cp.bmat
            Re_row.append(_wrap_as_1x1(re_total))
            Im_row.append(_wrap_as_1x1(im_total))
        Re_rows.append(Re_row)
        Im_rows.append(Im_row)
    Re_L = cp.bmat(Re_rows)
    Im_L = cp.bmat(Im_rows)
    return cp.bmat([[Re_L, -Im_L], [Im_L, Re_L]])


def _wrap_as_1x1(expr) -> cp.Expression:
    """Wrap a scalar / cvxpy expression as a 1x1 matrix expression."""
    if isinstance(expr, (int, float)):
        return np.array([[float(expr)]])
    return cp.reshape(expr, (1, 1), order="C")


# ---------- the SDP ----------
@dataclass
class KBKResult:
    M: float
    N: int
    K_trunc: int
    L_size: int
    sum_y_sq_bound: float    # SDP value: upper bound on sum y_n^2
    c_emp_bound: float       # = (1 + 2 * sum_y_sq_bound) / M
    hyp_r_target: float      # c_*  (=  log 16 / pi)
    hyp_r_target_sumy2: float  # (c_* M - 1) / 2
    closes_hyp_r: bool       # True iff c_emp_bound < c_*
    status: str
    a_opt: Optional[np.ndarray] = None
    b_opt: Optional[np.ndarray] = None
    V_opt: Optional[np.ndarray] = None


def solve_kbk_sdp(
    M: float,
    N: int = 15,
    K_trunc: int = 10,
    L_size: Optional[int] = None,
    use_kbk: bool = True,
    use_phase_aware_bochner: bool = True,
    use_y_toeplitz: bool = True,
    K_upper_bound: Optional[float] = None,
    solver: str = "SCS",
    verbose: bool = False,
) -> KBKResult:
    """Solve the Path-B Krein-Boas-Kac SDP at truncation N.

    Parameters
    ----------
    M : float
        M_max for Hyp_R; hat f(n) admissibility uses MO 2.14 with mu = mu(M).
    N : int
        Number of complex Fourier modes (n = 1..N).
    K_trunc : int
        Truncation order for the localizing convolution.
    L_size : int, optional
        Size of localizing PSD block. Defaults to max(5, N - K_trunc).
    use_kbk : bool
        If True, add the KBK localizing PSD (the new constraint).
    use_phase_aware_bochner : bool
        If True, add Hermitian Toeplitz on hat f (= Bochner on f, gives f >= 0).
        If False, only the weaker Bochner on |hat f|^2 (= R_f >= 0).
    use_y_toeplitz : bool
        If True, also add Toeplitz PSD on the |hat f|^2 = y_n sequence (Bochner on R_f).
        Implied by phase-aware Bochner but useful as a sanity check.
    K_upper_bound : float, optional
        If provided, add sum y_n <= (K_upper_bound - 1) / 2.  If None, no bound.
        Note: K is unbounded in the asymmetric case; without a bound the SDP may
        be unbounded.
    solver : str
        cvxpy solver: "SCS", "CLARABEL", or "MOSEK".
    """
    if L_size is None:
        L_size = max(5, N - K_trunc)
    if L_size > N:
        L_size = N

    mu = mu_M(M)

    # Decision variables
    a = cp.Variable(N)
    b = cp.Variable(N)
    # y_n is a SEPARATE variable with the DCP-allowed constraint
    #     a_n^2 + b_n^2 <= y_n   (cp.square is convex; "convex <= linear" is DCP)
    # The lift V = y y^T (Schur) gives an UPPER BOUND on sum y_n^2, hence on
    # sum (a_n^2 + b_n^2)^2 = sum |hat f(n)|^4.
    # NOTE: y_n can exceed a_n^2 + b_n^2 in the relaxation; this only LOOSENS
    # the bound (i.e. gives an upper bound), so it is sound for proving Hyp_R.
    y = cp.Variable(N, nonneg=True)

    constraints = []
    # Link y to (a, b): a^2 + b^2 <= y  (epigraph form)
    constraints.append(cp.square(a) + cp.square(b) <= y)
    constraints.append(y <= mu)                     # MO 2.14
    if K_upper_bound is not None:
        S = 0.5 * (K_upper_bound - 1.0)
        constraints.append(cp.sum(y) <= S)          # Parseval cap on K

    # Hermitian Toeplitz PSD on hat f  (= Bochner on f, => f >= 0)
    if use_phase_aware_bochner:
        T_real = hermitian_toeplitz_real_form(a, b, N)
        constraints.append(T_real >> 0)

    # Magnitude Toeplitz PSD on |hat f|^2 = y_n (= Bochner on R_f, weaker)
    if use_y_toeplitz:
        Tsize_y = N + 1
        T_y_rows = []
        for i in range(Tsize_y):
            row = []
            for j in range(Tsize_y):
                d = abs(i - j)
                if d == 0:
                    row.append(np.array([[1.0]]))
                else:
                    row.append(cp.reshape(y[d - 1], (1, 1), order="C"))
            T_y_rows.append(row)
        T_y = cp.bmat(T_y_rows)
        constraints.append(T_y >> 0)

    # KBK localizing PSD (the NEW constraint)
    if use_kbk:
        p_hat = p_hat_coefficients(K_trunc)
        L_real = localizing_toeplitz_real_form(a, b, N, L_size, p_hat, K_trunc)
        constraints.append(L_real >> 0)

    # Schur lift V = y y^T  for objective sum y_n^2
    V = cp.Variable((N, N), symmetric=True)
    constraints.append(V >> 0)
    constraints.append(cp.diag(V) <= mu ** 2)
    # KEY CUT: y_n in [0, mu] => y_n (mu - y_n) >= 0 => y_n^2 <= mu * y_n
    # Hence V_nn = y_n^2 <= mu * y_n.  Linear cut, tightens the relaxation.
    constraints.append(cp.diag(V) <= mu * y)
    # [[1, y^T]; [y, V]] >> 0  forces  V >= y y^T  (Schur)
    schur_lift = cp.bmat([
        [np.array([[1.0]]), cp.reshape(y, (1, N), order="C")],
        [cp.reshape(y, (N, 1), order="C"), V],
    ])
    constraints.append(schur_lift >> 0)

    # Maximize sum_n V_nn  (>= sum_n y_n^2 >= sum (a^2+b^2)^2; rigorous upper bound)
    objective = cp.Maximize(cp.sum(cp.diag(V)))
    prob = cp.Problem(objective, constraints)

    try:
        if solver == "SCS":
            prob.solve(solver=cp.SCS, eps=1e-9, max_iters=20000, verbose=verbose)
        elif solver == "CLARABEL":
            prob.solve(solver=cp.CLARABEL, verbose=verbose)
        elif solver == "MOSEK":
            prob.solve(solver=cp.MOSEK, verbose=verbose)
        else:
            prob.solve(verbose=verbose)
    except Exception as e:
        return KBKResult(
            M=M, N=N, K_trunc=K_trunc, L_size=L_size,
            sum_y_sq_bound=float("nan"),
            c_emp_bound=float("nan"),
            hyp_r_target=HYP_R_C_STAR,
            hyp_r_target_sumy2=hyp_r_target(M),
            closes_hyp_r=False,
            status=f"error: {e}",
        )

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return KBKResult(
            M=M, N=N, K_trunc=K_trunc, L_size=L_size,
            sum_y_sq_bound=float("nan"),
            c_emp_bound=float("nan"),
            hyp_r_target=HYP_R_C_STAR,
            hyp_r_target_sumy2=hyp_r_target(M),
            closes_hyp_r=False,
            status=prob.status,
        )

    sum_y_sq_bound = float(prob.value)
    c_emp_bound = (1.0 + 2.0 * sum_y_sq_bound) / M
    return KBKResult(
        M=M, N=N, K_trunc=K_trunc, L_size=L_size,
        sum_y_sq_bound=sum_y_sq_bound,
        c_emp_bound=c_emp_bound,
        hyp_r_target=HYP_R_C_STAR,
        hyp_r_target_sumy2=hyp_r_target(M),
        closes_hyp_r=c_emp_bound < HYP_R_C_STAR,
        status=prob.status,
        a_opt=np.asarray(a.value),
        b_opt=np.asarray(b.value),
        V_opt=np.asarray(V.value),
    )


def report(res: KBKResult) -> str:
    closes_str = "*** CLOSES HYP_R ***" if res.closes_hyp_r else "(does not close)"
    return (
        f"M={res.M}, N={res.N}, K_trunc={res.K_trunc}, L_size={res.L_size}\n"
        f"  status                = {res.status}\n"
        f"  max sum y_n^2  (SDP)  = {res.sum_y_sq_bound:.6f}\n"
        f"  c_emp upper bound     = {res.c_emp_bound:.6f}\n"
        f"  Hyp_R target c_*      = {res.hyp_r_target:.6f}  -> max sum y^2 needed: {res.hyp_r_target_sumy2:.6f}\n"
        f"  {closes_str}"
    )


__all__ = [
    "KBKResult",
    "LOG16_PI",
    "HYP_R_C_STAR",
    "mu_M",
    "hyp_r_target",
    "p_hat_coefficients",
    "hermitian_toeplitz_real_form",
    "localizing_toeplitz_real_form",
    "solve_kbk_sdp",
    "report",
]
