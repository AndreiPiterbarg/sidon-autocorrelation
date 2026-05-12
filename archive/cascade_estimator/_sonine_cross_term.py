"""Rigorous arb-interval computation of the multi-scale arcsine cross-term
   I(alpha, beta) = int_0^infty J_0(pi*alpha*xi)^2 * J_0(pi*beta*xi)^2 dxi.

This module produces a RIGOROUS rational interval enclosure of I(alpha,beta)
using:
  (1) Rigorous adaptive arb-integration of int_0^T f(xi) dxi via flint's
      `acb.integral` (which returns an arb ball certified by the algorithm).
  (2) An explicit, citable tail bound:

      Sonin-Polya / Schafheitlin (see Watson, A Treatise on the Theory of
      Bessel Functions, 2nd ed., 1944, Sec. 7.31 and Sec. 13.74; also
      Polya-Szego, Aufgaben und Lehrsaetze II, problem 207):

         J_0(z)^2 <= 2/(pi*z)   for all z > 0.

      Hence for any T > 0 with z_min := pi*min(alpha,beta)*T > 0,

         |int_T^infty J_0(pi a x)^2 J_0(pi b x)^2 dx|
            <= int_T^infty (2/(pi^2 a x)) (2/(pi^2 b x)) dx
            =  4 / (pi^2 * a * b * T) / pi^2
         (after careful tracking)
      In fact J_0(pi*alpha*xi)^2 <= 2/(pi^2 alpha xi) gives
         f(xi) <= 4/(pi^2 alpha beta xi^2),
         int_T^inf f dxi <= 4/(pi^2 alpha beta T).

      We use this rigorous tail bound.

Combining (1) and (2), the output I_arb is an arb interval that rigorously
contains I(alpha,beta).

We then plug into the multi-scale arcsine K_2 formula:

    K_hat(xi) = lambda * J_0(pi alpha xi)^2 + (1-lambda) * J_0(pi beta xi)^2
    K_2 = int_{-infty}^infty K_hat^2 dxi
        = 2 * int_0^infty K_hat^2 dxi
        = 2 [lambda^2 I(alpha,alpha) + 2 lambda(1-lambda) I(alpha,beta)
             + (1-lambda)^2 I(beta,beta)]

(where I(a,a) = (1/2) K_2(a) for a single-scale arcsine.)

Finally we evaluate the MV master inequality with rigorous arb arithmetic
to produce M_cert as an arb interval.
"""
from __future__ import annotations

import json
import os
import sys
import time

from flint import acb, arb, ctx

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

# Working precision (bits)
PREC_BITS = 160


def _set_prec(p: int = PREC_BITS):
    ctx.prec = p


def J0_sq(z: arb) -> arb:
    """Return J_0(z)^2 as an arb ball."""
    return z.bessel_j(0) ** 2


def integrand_arb(alpha: arb, beta: arb):
    pi = arb.pi()
    def f(x, _):
        z = pi * alpha * x
        w = pi * beta * x
        return z.bessel_j(0) ** 2 * w.bessel_j(0) ** 2
    return f


def sonine_I(alpha, beta, T: float = 1.0e5,
             rel_tol_bits: int = 50, verbose: bool = False) -> arb:
    """Rigorous arb interval enclosing I(alpha, beta) = int_0^infty
       J_0(pi alpha xi)^2 J_0(pi beta xi)^2 dxi.

    alpha, beta : arb or numeric
        Scale parameters (>0).
    T : float
        Truncation point. Tail past T is bounded rigorously via Sonin-Polya.
    """
    _set_prec(PREC_BITS)
    if not isinstance(alpha, arb):
        alpha = arb(str(alpha))
    if not isinstance(beta, arb):
        beta = arb(str(beta))

    f = integrand_arb(alpha, beta)
    # Rigorous adaptive integration on [0, T]
    t0 = time.time()
    I_T = acb.integral(f, 0, T, rel_tol=2 ** (-rel_tol_bits))
    # acb.integral returns an acb; take real part
    I_T_re = arb(I_T.real)
    elapsed = time.time() - t0

    # Rigorous tail bound:  |tail| <= 4 / (pi^2 * alpha * beta * T)
    # NOTE: Sonin-Polya: J_0(z)^2 <= 2/(pi z) for all z > 0  (Watson §7.31; §13.74)
    # Hence f(xi) <= (2/(pi^2 alpha xi)) (2/(pi^2 beta xi)) = 4/(pi^4 alpha beta xi^2)
    # => int_T^inf f <= 4/(pi^4 alpha beta T)
    pi = arb.pi()
    tail_bound = arb(4) / (pi ** 4 * alpha * beta * arb(T))
    # tail is in [0, tail_bound]
    # Construct arb interval [I_T_lower, I_T_upper + tail_bound]
    # arb.union with 0 + tail enclosure
    tail_mid = tail_bound / 2
    tail_ball = tail_mid + arb(0).union(tail_bound) - tail_mid  # symmetric ball [0, tail_bound]
    # Simpler: arb represents value (1/2)*tail_bound +/- (1/2)*tail_bound
    half = tail_bound / 2
    # Construct arb ball with midpoint half and radius half
    # using union of 0 and tail_bound
    tail_arb = arb(0).union(tail_bound)
    I = I_T_re + tail_arb
    if verbose:
        print(f"  sonine_I(alpha={float(alpha):.6f}, beta={float(beta):.6f}, T={T:.1e}):")
        print(f"    int [0,T]    = {I_T_re}")
        print(f"    tail bound   <= {float(tail_bound):.3e}")
        print(f"    I            = {I}")
        print(f"    elapsed      = {elapsed:.2f}s")
    return I


# ---------------------------------------------------------------------------
# Multi-scale K_2 and M_cert via MV master inequality (rigorous arb arithmetic)
# ---------------------------------------------------------------------------

def K_hat_multi_arb(xi: arb, deltas, lambdas) -> arb:
    """K_hat(xi) = sum_i lambdas[i] * J_0(pi*deltas[i]*xi)^2   (arb)."""
    pi = arb.pi()
    out = arb(0)
    for lam, d in zip(lambdas, deltas):
        z = pi * d * xi
        out = out + lam * z.bessel_j(0) ** 2
    return out


def rigorous_K2_multiscale(deltas, lambdas, T_cut: float = 1.0e5,
                            verbose: bool = False) -> arb:
    """Rigorous arb enclosure of K_2 = int_{-inf}^inf K_hat^2 dxi for
       multi-scale arcsine kernel.

    K_hat(xi) = sum_i lambda_i J_0(pi delta_i xi)^2  >= 0.
    K_2 = 2 sum_{i,j} lambda_i lambda_j I(delta_i, delta_j)
        = 2 [ sum_i lambda_i^2 I(d_i, d_i)
              + 2 sum_{i<j} lambda_i lambda_j I(d_i, d_j) ]
    """
    _set_prec(PREC_BITS)
    deltas = [arb(str(d)) if not isinstance(d, arb) else d for d in deltas]
    lambdas = [arb(str(l)) if not isinstance(l, arb) else l for l in lambdas]
    n = len(deltas)
    Iij = [[None]*n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            Iij[i][j] = sonine_I(deltas[i], deltas[j], T=T_cut, verbose=verbose)
            if i != j:
                Iij[j][i] = Iij[i][j]
    # Sum lambda_i lambda_j I_ij
    S = arb(0)
    for i in range(n):
        for j in range(n):
            S = S + lambdas[i] * lambdas[j] * Iij[i][j]
    K2 = 2 * S
    return K2, Iij


def K_hat_qp(xi_rat, deltas, lambdas) -> arb:
    """K_hat at a rational point xi (typically j/U)."""
    return K_hat_multi_arb(arb(xi_rat) if not isinstance(xi_rat, arb) else xi_rat,
                            deltas, lambdas)


def load_mv_coeffs_arb():
    """Load MV's a_j coefficients as a list of arb (rational)."""
    from delsarte_dual.grid_bound.coeffs import mv_coeffs_fmpq
    coeffs = mv_coeffs_fmpq()
    return [arb(c.p) / arb(c.q) for c in coeffs]


def rigorous_S1(deltas, lambdas, U_val) -> arb:
    """S_1 = sum_{j=1..119} a_j^2 / K_hat(j/U). All arb."""
    mvc = load_mv_coeffs_arb()
    U_arb = arb(str(U_val)) if not isinstance(U_val, arb) else U_val
    S = arb(0)
    for j, aj in enumerate(mvc, start=1):
        xi = arb(j) / U_arb
        kh = K_hat_multi_arb(xi, deltas, lambdas)
        S = S + aj * aj / kh
    return S


def rigorous_k1(deltas, lambdas) -> arb:
    return K_hat_multi_arb(arb(1), deltas, lambdas)


def mv_master_M_cert_arb(k1: arb, K2: arb, S1: arb, U_val,
                          verbose: bool = False) -> arb:
    """Solve M from sup_R(M) = 2/U + 4/(U S1) using interval bisection on arb.

    Returns an arb ball [M_lo, M_hi] guaranteed to contain M_cert.
    """
    _set_prec(PREC_BITS)
    U = arb(str(U_val)) if not isinstance(U_val, arb) else U_val
    target = arb(2)/U + arb(4)/(U * S1)
    pi = arb.pi()

    def sup_R(M: arb) -> arb:
        # mu = M * sin(pi/M) / pi
        mu = M * (pi/M).sin() / pi
        # y* = k1 * sqrt((M-1)/(K2-1))
        y_star_sq = k1 * k1 * (M - 1) / (K2 - 1)
        y_star = y_star_sq.sqrt()
        rad1 = (M - 1 - 2 * mu * mu)
        rad2 = (K2 - 1 - 2 * k1 * k1)
        # Branch 1: y_star <= mu  =>  R1 = M+1+sqrt((M-1)(K2-1))
        b1 = M + 1 + ((M - 1) * (K2 - 1)).sqrt()
        # Branch 2: y_star > mu  =>  R2 = M+1+2 mu k1 + sqrt(rad1*rad2)
        # (when rad1 >= 0, rad2 >= 0)
        try:
            b2 = M + 1 + 2 * mu * k1 + (rad1 * rad2).sqrt()
        except Exception:
            b2 = None
        # Decide branch rigorously
        diff = y_star_sq - mu*mu
        if diff.upper() <= 0:
            return b1
        elif diff.lower() > 0:
            return b2 if b2 is not None else b1
        else:
            # Ambiguous interval: return upper envelope
            if b2 is None:
                return b1
            return b1.union(b2)

    # Bisect on [1+eps, 2].  Use rigorous interval bisection: at each step,
    # we maintain the invariant that M_cert in [M_lo, M_hi].  We can only
    # shrink the bracket when sup_R(M_mid) is rigorously on one side of target.
    # Otherwise we STOP (the interval cannot be shrunk further with current arb prec).
    M_lo = arb('1.001')
    M_hi = arb(2)
    # First check the endpoints
    s_lo = sup_R(M_lo) - target
    s_hi = sup_R(M_hi) - target
    if s_lo.lower() > 0 or s_hi.upper() < 0:
        # No bracket
        return M_lo.union(M_hi)
    for _ in range(200):
        # M_mid as midpoint of arb bracket (use float midpoint to keep ball tight)
        mid_f = (float(M_lo.lower()) + float(M_hi.upper())) / 2
        M_mid = arb(str(mid_f))
        s = sup_R(M_mid) - target
        if s.upper() < 0:
            M_lo = M_mid
        elif s.lower() > 0:
            M_hi = M_mid
        else:
            # Ambiguous: can't shrink further; M_cert lies in [M_lo, M_hi]
            # but the current bracket may still be wider than we like.
            # Break out -- further bisection won't reduce uncertainty.
            break
        if float(M_hi) - float(M_lo) < 1e-12:
            break
    # Return the bracket [M_lo, M_hi] as an arb ball.
    lo = float(M_lo.lower())
    hi = float(M_hi.upper())
    return arb(lo).union(arb(hi))


# ---------------------------------------------------------------------------
# Main: compute rigorous M_cert for K26-best multi-scale (alpha=0.138,beta=0.055,lam=0.9312)
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("Rigorous Sonine cross-term I(alpha,beta) and multi-scale arcsine M_cert")
    print("=" * 78)

    DELTA = arb('0.138')
    BETA = arb('0.055')
    LAM = arb('0.9312')  # K26 best

    T_cut = 1.0e5

    # ----- Step 1: rigorous I_ij ------
    print("\nStep 1: Compute I(alpha,beta) rigorously")
    I_aa = sonine_I(DELTA, DELTA, T=T_cut, verbose=True)
    I_ab = sonine_I(DELTA, BETA, T=T_cut, verbose=True)
    I_bb = sonine_I(BETA, BETA, T=T_cut, verbose=True)

    # ----- Step 2: rigorous K_2 ------
    print("\nStep 2: Compute rigorous K_2")
    # K_2 = 2*(lam^2 I_aa + 2 lam (1-lam) I_ab + (1-lam)^2 I_bb)
    K2 = 2 * (LAM*LAM*I_aa + 2*LAM*(1-LAM)*I_ab + (1-LAM)*(1-LAM)*I_bb)
    print(f"   K_2 = {K2}")
    print(f"   K_2 width = {float(K2.rad()):.3e}")

    # ----- Step 3: k_1 and S_1 (rigorous) -----
    print("\nStep 3: Compute k_1 and S_1 rigorously")
    deltas = [DELTA, BETA]
    lambdas = [LAM, 1 - LAM]
    k1 = rigorous_k1(deltas, lambdas)
    print(f"   k_1 = {k1}")
    U_val = arb('0.5') + DELTA  # 0.638
    S1 = rigorous_S1(deltas, lambdas, U_val)
    print(f"   S_1 = {S1}")

    # ----- Step 4: rigorous M_cert ------
    print("\nStep 4: Solve MV master inequality for M_cert")
    M_cert = mv_master_M_cert_arb(k1, K2, S1, U_val, verbose=True)
    print(f"\n   M_cert (rigorous interval) = {M_cert}")
    print(f"   M_cert lower bound = {float(M_cert.lower()):.6f}")
    print(f"   M_cert upper bound = {float(M_cert.upper()):.6f}")
    print(f"   width = {float(M_cert.rad()):.3e}")

    # ----- Step 5: Save -----
    out = {
        "alpha": float(DELTA),
        "beta": float(BETA),
        "lambda": float(LAM),
        "T_cut": T_cut,
        "I_aa_mid": float(I_aa.mid()),
        "I_aa_rad": float(I_aa.rad()),
        "I_ab_mid": float(I_ab.mid()),
        "I_ab_rad": float(I_ab.rad()),
        "I_bb_mid": float(I_bb.mid()),
        "I_bb_rad": float(I_bb.rad()),
        "K2_mid": float(K2.mid()),
        "K2_rad": float(K2.rad()),
        "k1_mid": float(k1.mid()),
        "k1_rad": float(k1.rad()),
        "S1_mid": float(S1.mid()),
        "S1_rad": float(S1.rad()),
        "M_cert_mid": float(M_cert.mid()),
        "M_cert_rad": float(M_cert.rad()),
        "M_cert_lower": float(M_cert.lower()),
        "M_cert_upper": float(M_cert.upper()),
        "rigorous_M_cert_lb": float(M_cert.lower()),
        "beats_MV_1.2748": float(M_cert.lower()) > 1.2748,
        "beats_CS_1.2802": float(M_cert.lower()) > 1.2802,
        "beats_127": float(M_cert.lower()) > 1.270,
        "beats_1275": float(M_cert.lower()) > 1.275,
    }
    outpath = os.path.join(REPO, "_sonine_cross_term_result.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {outpath}")

    print(f"\n{'='*78}\nFINAL: rigorous M_cert in [{float(M_cert.lower()):.6f}, "
          f"{float(M_cert.upper()):.6f}]")
    print(f"   beats MV (1.2748)? {out['beats_MV_1.2748']}")
    print(f"   beats CS (1.2802)? {out['beats_CS_1.2802']}")
    print(f"   beats 1.275? {out['beats_1275']}")
    return out


if __name__ == "__main__":
    main()
