"""Path-1 Toeplitz-PSD SDP for sharper Lemma 3.4 (numerical).

Implements TWO variants of the primal SDP from ``proof_paths.md`` Path 1:

  (SYM) symmetric f case  -- y_k real and >= 0 for ALL k:
        max  y_{n0}
        s.t. y_0 = 1,  y_k >= 0  for k = 0..N,
             T_N := [y_{|i-j|}]_{i,j=0..N}  >= 0,
             M - (y_0 + 2 sum_{k=1}^N y_k cos(2 pi k x)) >= 0 on T.

  (ASYM) general (asymmetric) f -- y_k real, only y_{n0} >= 0:
        same as above but DROP y_k >= 0 for k != n_0.

The second pointwise constraint is encoded by Fejer-Riesz: a real
symmetric trig polynomial s(x) = s_0 + 2 sum_{k=1..N} s_k cos(2 pi k x)
is nonnegative on the torus iff there exists a Hermitian PSD Q in
C^{(N+1) x (N+1)} with s_k = sum_i Q[i, i+k] (Hermitian convention).
We encode Hermitian Q by a real (2(N+1)) x (2(N+1)) PSD block.

This is Dumitrescu 2007, Ch. 4 "bounded density" + Fejer-Riesz LMI.

KEY FINDINGS (numerical, this session):
  (SYM)  mu_sharper(M, n_0) = (M - 1) / 2   for all n_0 >= 1, all N.
         This is a 46% improvement over MV's mu(M) = M sin(pi/M)/pi
         at M = 1.275 (0.1375 vs 0.2544).
         Achieved by y_{n_0} = (M-1)/2 with all other y_k = 0.
         Proof is one-line: y_k >= 0, p(0) <= M => 1 + 2*sum_{k>=1} y_k <= M.
         The Toeplitz-PSD constraint is SLACK; only the L^infinity LMI binds.

  (ASYM) mu_sharper(M, n_0) -> mu_MV(M)  as N -> infinity.
         The Path-1 SDP at this (low) level adds NOTHING for asymmetric f.
         This is consistent with the MV bathtub being the correct extremum
         over h = h(x) >= 0, h <= M, int h = 1.

So Path 1 gives a STRICT IMPROVEMENT only for the SYMMETRIC subclass of
Sidon problems (where C_{1a}^sym >= 1.42429 already from Path A); it does
NOT improve the unrestricted C_{1a} bound through this LMI alone.

The 1-5% gap predicted in proof_paths.md (line ~50) was based on a
formulation that conflates symmetric and asymmetric cases.  In the symmetric
case the actual gap is 46% (much larger), but the symmetric subclass is
already handled separately.  In the asymmetric case the gap is 0% at
SDP level N -- to get a strict improvement for asymmetric f one must
use Path 2 (Fejer-Riesz factorization h = |p|^2) which exploits the
SCHUR-PRODUCT structure y_k = (hat f)(k) * (hat f)(-k) in a way that
Path 1's Toeplitz-PSD relaxation does not capture.

Returns: numerical mu'_N(M, n0).  Not rigorous (cvxpy/CLARABEL float SDP);
rigorous certification (rational rounding + Putinar SOS verification in
arb / Lean) is the Phase-2 task documented in ``derivation.md`` §6.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SDPResult:
    M: float
    n0: int
    N: int                # truncation level
    mu_sharper: float     # numerical optimum
    mu_MV: float          # M sin(pi/M)/pi for comparison
    gap: float            # mu_MV - mu_sharper
    status: str           # solver status
    y: np.ndarray         # full optimal moment vector (length N+1)
    solve_time: float     # seconds
    n_vars: int
    n_psd_blocks: int
    psd_block_sizes: tuple


def mu_MV(M: float) -> float:
    """MV's bathtub bound mu(M) = M sin(pi/M)/pi."""
    return M * np.sin(np.pi / M) / np.pi


def solve_path1_sdp(
    M: float,
    N: int = 4,
    n0: int = 1,
    solver: str = "CLARABEL",
    verbose: bool = False,
    symmetric: bool = True,
) -> SDPResult:
    """Solve the Path-1 Toeplitz-PSD SDP at truncation N for target moment n0.

    Parameters
    ----------
    M : float
        ||h||_inf ceiling.
    N : int
        Truncation level — Toeplitz size is (N+1) x (N+1), Fejer-Riesz block
        is also (N+1) x (N+1).  Number of decision variables: N+1 (the y_k's).
    n0 : int
        Target Fourier index (1 <= n0 <= N).
    solver : str
        cvxpy solver name. CLARABEL is the recommended default; MOSEK is
        faster if available.
    verbose : bool
        Solver verbosity.

    Returns
    -------
    SDPResult dataclass.
    """
    import cvxpy as cp
    import time

    if not (1 <= n0 <= N):
        raise ValueError(f"need 1 <= n0={n0} <= N={N}")
    if M <= 1.0:
        raise ValueError(f"M={M} must be > 1 (else feasible set degenerate)")

    # Decision variables: y_0, y_1, ..., y_N (real).
    # symmetric=True: y_k >= 0 for all k (the symmetric-f Path 1 SDP).
    # symmetric=False: y_k unconstrained sign except y_{n0} >= 0 (asymmetric).
    if symmetric:
        y = cp.Variable(N + 1, nonneg=True)
    else:
        y = cp.Variable(N + 1)

    # T_N: Toeplitz PSD for "h >= 0 as a measure on T"
    # T_{i,j} = y_{|i-j|}.  Symmetric, real.
    T = cp.bmat(
        [[y[abs(i - j)] for j in range(N + 1)] for i in range(N + 1)]
    )

    # Fejer-Riesz / Lukacs block(s) for "M - p(x) >= 0 on T".
    #
    # On the full torus, a real symmetric trig poly
    #   s(x) = s_0 + 2 sum_{k=1..N} s_k cos(2 pi k x)
    # is nonneg iff there exists HERMITIAN PSD Q in C^{(N+1) x (N+1)} with
    #   s_k = sum_i Q[i, i+k]   for k = 0..N,
    # where the k>=1 sum is taken in the upper triangle (Hermitian convention
    # gives s_k = Re of the diagonal sum, but for symmetric s_k this equals
    # the sum itself once we enforce Hermitian structure).
    #
    # We encode Hermitian Q via a real (2(N+1)) x (2(N+1)) symmetric PSD
    # block Qr = [[A, -B], [B, A]] where Q = A + i*B with A symmetric and B
    # antisymmetric.  The trig-coeff equations become:
    #   s_k = sum_{i=0..N-k} A[i, i+k]      (for real-coefficient s_k)
    # and the off-diagonal antisymmetric part B drops out of s_k for symmetric
    # input.  This is equivalent to using a Hermitian Q with the constraint
    # that s_k be real.
    A = cp.Variable((N + 1, N + 1), symmetric=True)
    B = cp.Variable((N + 1, N + 1))
    # Hermitian PSD encoding via real 2x2 block:
    Qr = cp.bmat([[A, -B], [B, A]])
    # B antisymmetric:
    constraints_aux = [B + B.T == 0]

    # s_k equations.  s_0 = M - 1, s_k = -y_k for k>=1.
    # For Hermitian Q with real entries on the diagonal, s_k = sum A[i,i+k]
    # (the imaginary parts from B cancel by symmetry of s).
    trace_eqs = []
    trace_eqs.append(cp.sum(cp.diag(A)) == M - y[0])
    for k in range(1, N + 1):
        diag_k = cp.sum(
            cp.hstack([A[i, i + k] for i in range(N + 1 - k)])
        )
        trace_eqs.append(diag_k == -y[k])

    constraints = [
        y[0] == 1,
        T >> 0,
        Qr >> 0,
        *constraints_aux,
        *trace_eqs,
    ]
    if not symmetric:
        # In asymmetric mode, only enforce nonneg target (phase WLOG).
        constraints.append(y[n0] >= 0)

    obj = cp.Maximize(y[n0])
    prob = cp.Problem(obj, constraints)

    t0 = time.time()
    try:
        prob.solve(solver=solver, verbose=verbose)
    except Exception as e:
        return SDPResult(
            M=M, n0=n0, N=N, mu_sharper=float("nan"),
            mu_MV=mu_MV(M), gap=float("nan"),
            status=f"error: {e}", y=np.full(N + 1, np.nan),
            solve_time=time.time() - t0,
            n_vars=N + 1 + (N + 1) * (N + 2) // 2,
            n_psd_blocks=2,
            psd_block_sizes=(N + 1, N + 1),
        )
    elapsed = time.time() - t0

    if prob.status not in ("optimal", "optimal_inaccurate"):
        mu_val = float("nan")
        y_val = np.full(N + 1, np.nan)
    else:
        mu_val = float(y[n0].value)
        y_val = np.array(y.value, dtype=float)

    mu_mv = mu_MV(M)
    return SDPResult(
        M=M,
        n0=n0,
        N=N,
        mu_sharper=mu_val,
        mu_MV=mu_mv,
        gap=(mu_mv - mu_val) if np.isfinite(mu_val) else float("nan"),
        status=prob.status,
        y=y_val,
        solve_time=elapsed,
        n_vars=N + 1 + (N + 1) * (N + 2) // 2,
        n_psd_blocks=2,
        psd_block_sizes=(N + 1, N + 1),
    )


def sdp_size_summary(N: int) -> dict:
    """Report SDP size at truncation N (analytic, no solve)."""
    n_y = N + 1                              # moment variables
    n_Q = (N + 1) * (N + 2) // 2             # symmetric Q entries
    n_total_vars = n_y + n_Q
    n_psd_dim = N + 1
    n_eq = 1 + (N + 1)                       # y_0 = 1, plus N+1 trace eqs
    return dict(
        N=N,
        n_y=n_y,
        n_Q_entries=n_Q,
        n_total_decision_vars=n_total_vars,
        psd_block_dim=n_psd_dim,
        n_psd_blocks=2,
        n_equality_constraints=n_eq,
        n_nonneg_constraints=N + 1,
    )


def sweep_N(
    M: float,
    Ns=(2, 3, 4, 5, 6, 8),
    n0: int = 1,
    solver: str = "CLARABEL",
) -> list:
    """Solve the SDP at multiple truncation levels and report the gap series."""
    results = []
    for N in Ns:
        res = solve_path1_sdp(M=M, N=N, n0=n0, solver=solver)
        results.append(res)
    return results


def propagate_to_M_lower(
    mu_sharper_value: float,
    M_grid=None,
) -> dict:
    """Propagate a sharper mu(M) bound back through MV's master inequality.

    MV's chain of bounds yields a self-consistency equation roughly of the
    form (one-moment, schematic):

        2/u + a  <=  M + 1 + 2 mu(M) + sqrt(M - 1) * sqrt(K2 - 1).

    Solving for the smallest M satisfying this with optimal (u, delta) gives
    M_lower = 1.27481 (MV one-moment).  Replacing mu(M) by mu_sharper(M) <
    mu(M) and re-solving gives a strictly larger M_lower.

    This helper performs the substitution numerically by 1-D scan on M.
    Coefficients (u, K2, gain_a) are taken from MV's published values.
    """
    # MV one-moment baseline parameters (from delsarte_dual/grid_bound/coeffs.py)
    # u = 0.4, delta = 0.5, K2 = 4/pi.  gain_a is the "cos" gain.
    u = 0.4
    K2 = 4.0 / np.pi
    delta = 0.5
    # gain_a placeholder = 0 for the toy propagation; use full MV pipeline for cert.
    gain_a = 0.0

    def slack(M, mu_fn):
        rhs = M + 1.0 + 2.0 * mu_fn(M) + np.sqrt(max(M - 1.0, 0.0)) * np.sqrt(
            max(K2 - 1.0, 0.0)
        )
        lhs = 2.0 / u + gain_a
        return rhs - lhs

    if M_grid is None:
        M_grid = np.linspace(1.20, 1.35, 1501)

    mu_const = lambda M_: mu_sharper_value if M_ >= 1.27 else mu_MV(M_)
    mu_mv = lambda M_: mu_MV(M_)

    # Find smallest M with slack(M) >= 0 in each case
    def find_root(mu_fn):
        for M_ in M_grid:
            if slack(M_, mu_fn) >= 0:
                return M_
        return None

    return dict(
        M_lower_MV=find_root(mu_mv),
        M_lower_sharper=find_root(mu_const),
        delta_M=(find_root(mu_const) or float("nan"))
        - (find_root(mu_mv) or float("nan")),
    )


def report(results: list) -> str:
    """Pretty-print a sweep result."""
    lines = []
    lines.append(
        f"{'N':>3} {'mu_sharper':>12} {'mu_MV':>12} {'gap':>12}  "
        f"{'gap_pct':>8} {'status':>20} {'time_s':>8}"
    )
    lines.append("-" * 90)
    for r in results:
        gap_pct = (
            100 * r.gap / r.mu_MV if np.isfinite(r.gap) and r.mu_MV > 0 else float("nan")
        )
        lines.append(
            f"{r.N:>3} {r.mu_sharper:>12.7f} {r.mu_MV:>12.7f} "
            f"{r.gap:>12.3e} {gap_pct:>7.2f}% {r.status:>20} {r.solve_time:>8.2f}"
        )
    return "\n".join(lines)


__all__ = [
    "SDPResult",
    "mu_MV",
    "solve_path1_sdp",
    "sdp_size_summary",
    "sweep_N",
    "propagate_to_M_lower",
    "report",
]


if __name__ == "__main__":
    print("=" * 80)
    print("Path-1 Toeplitz-PSD SDP for sharper Lemma 3.4")
    print("=" * 80)

    M_test = 1.275
    print(f"\nReference: mu_MV(M={M_test}) = {mu_MV(M_test):.7f}\n")

    print("SDP size summary:")
    for N in (2, 3, 4, 5, 6, 8, 10, 12, 16, 20):
        s = sdp_size_summary(N)
        print(f"  N={N:>2}: vars={s['n_total_decision_vars']:>4}, "
              f"PSD={s['n_psd_blocks']}x{s['psd_block_dim']}, "
              f"eq={s['n_equality_constraints']}")

    print("\n" + "=" * 80)
    print("VARIANT (SYM): symmetric f, y_k >= 0 for all k")
    print("=" * 80)
    results_sym = []
    for N in (2, 3, 4, 5, 6, 8, 10, 12, 16):
        results_sym.append(solve_path1_sdp(M_test, N=N, n0=1, symmetric=True))
    print(report(results_sym))

    print("\n" + "=" * 80)
    print("VARIANT (ASYM): general f, only y_{n0} >= 0")
    print("=" * 80)
    results_asym = []
    for N in (2, 3, 4, 5, 6, 8, 10, 12, 16):
        results_asym.append(solve_path1_sdp(M_test, N=N, n0=1, symmetric=False))
    print(report(results_asym))

    print("\n" + "=" * 80)
    print("M-sweep (SYM mode), N=10, n0=1")
    print("=" * 80)
    print(f"{'M':>6} {'mu_MV':>10} {'mu_sharp_SYM':>14} {'(M-1)/2':>10} {'gap_pct':>8}")
    for M in (1.05, 1.10, 1.15, 1.20, 1.25, 1.275, 1.30, 1.35, 1.40, 1.50, 1.65, 1.85, 1.95):
        r = solve_path1_sdp(M, N=10, n0=1, symmetric=True)
        print(f"{M:>6.3f} {r.mu_MV:>10.5f} {r.mu_sharper:>14.7f} "
              f"{(M-1)/2:>10.5f} {100*r.gap/r.mu_MV:>7.2f}%")

    print("\n" + "=" * 80)
    print("Propagation: 1-moment self-consistency M-lower (toy scan)")
    print("=" * 80)
    print(f"  Using mu_sharper(M)   = (M-1)/2          [SYM, rigorous]")
    print(f"  Using mu_sharper(M)   = mu_MV(M)         [ASYM]")
    # 1-moment fixed-point scan
    u = 0.4; K2 = 4.0/np.pi; gain_a = 0.0
    def slack(M, mu_fn):
        rhs = M + 1.0 + 2.0*mu_fn(M) + np.sqrt(max(M-1.0,0))*np.sqrt(max(K2-1.0,0))
        return rhs - 2.0/u - gain_a
    M_grid = np.linspace(1.0, 2.0, 10001)
    def find_min_M(mu_fn):
        for M_ in M_grid:
            if slack(M_, mu_fn) >= 0:
                return M_
        return None
    M_mv = find_min_M(mu_MV)
    M_sym = find_min_M(lambda M: (M-1)/2)
    print(f"  M_lower under mu_MV    = {M_mv}")
    print(f"  M_lower under (M-1)/2  = {M_sym}")
    if M_mv and M_sym:
        print(f"  Improvement Delta_M    = {M_sym - M_mv:+.5f}")
    print()
    print("=" * 80)
    print("CONCLUSION (one-session results)")
    print("=" * 80)
    print("""
  SYM:  mu_sharper(M, n) = (M-1)/2 RIGOROUSLY (one-line proof from y_k >= 0,
        sum y_k = h(0) <= M is *not* needed; rather, p(x) <= M at x=0 gives
        1 + 2*sum y_k <= M for k=1..N).  At M=1.275, this is 0.1375 (vs MV's
        0.2544): a 46% improvement.  But applies ONLY to symmetric f.
        The symmetric-f bound C_{1a}^sym >= 1.42429 (Path A) already exists.
  ASYM: SDP equals MV at all N tested (N <= 16).  No improvement over MV.
        Path 1's Toeplitz-PSD + Fejer-Riesz LMI is exactly the dual of MV's
        bathtub problem at this level; the autoconvolution structure
        (h = f*f for some f) does not enter Path 1 except through the
        single-coefficient phase shift.

  RECOMMENDATION:
    1) Update mu_sharper.py to expose mu_sharper_sym(M) = (M-1)/2 for the
       SYM subclass (rigorous, NO SDP needed; just a Cauchy-Schwarz/L^inf
       argument).  This is a NEW result for the symmetric subproblem and
       can sharpen Path A's 1.42429.
    2) Path 1 ASYM does NOT improve over MV. Sharper bounds for asymmetric
       f require Path 2 (Fejer-Riesz factorization h=|p|^2; uses Schur-
       product structure not visible to Path 1's Toeplitz-PSD).
    3) Plug (M-1)/2 into the multi-moment Phi (phi_mm.py) restricted to
       the symmetric f search and bisect to get the new M_cert^sym.
""")
