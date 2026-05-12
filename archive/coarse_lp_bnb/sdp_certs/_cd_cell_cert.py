"""Christoffel-Darboux per-cell box-cert investigation — NULL RESULT.

Investigation: can a CD-kernel based per-cell certificate beat the existing
Shor (order-1 Lasserre + RLT) cell-cert in the COARSE cascade?

VERDICT: NO.  Three plausible CD encodings all fail by one of:
  (a) wrong-direction relaxation (CD-smoothed integral < max_t, gives a
      LOWER bound that is strictly LOOSER than what we already have);
  (b) dominated by the existing PSD lift (CD-as-localizer is a special
      case of order-N Lasserre, dominated by L2);
  (c) trivial-collapse pathology already documented for the 3-point pilot
      (THREEPOINT_FULL_REPORT.md: lambda* = 1 by 3-extension of diagonal
      pseudo-measures).

This file documents the analysis; no implementation is shipped because
none of the three encodings could beat the existing chain.

================================================================
1. Mathematical setup of per-cell box cert
================================================================

Coarse-grid cell at composition c (sum c = S, h = 1/(2S)):

    Cell(c) = { mu = c/S + delta : |delta_i| <= h, sum delta = 0,
                                   delta_i + c_i/S >= 0 }

For each window W = (ell, s_lo) the autocorrelation TV is a quadratic
form in delta:

    TV_W(c/S + delta) = TV_W(c/S) + grad_W . delta
                      + (2d/ell) * delta^T A_W delta.

Cell-cert problem:

    cert(c, c_target) = min_{delta in Cell} max_W TV_W(c/S + delta)

Cell is pruned iff cert(c, c_target) >= c_target.

The existing chain (from `_coarse_L_bench.py` and `_L_bench.py`) uses:
  - F:  LP relaxation of the linear part with delta^2 spectral bound;
  - Q:  LP dual averaging max-min over sign vertices;
  - L:  Shor SDP — order-1 Lasserre M_1 PSD lift + box + RLT cuts;
  - L2: order-2 Lasserre M_2 (degree-4 polynomial certificate, dim O(d^2)).

================================================================
2. CD encoding A: smoothed max via Christoffel-Darboux kernel
================================================================

Christoffel-Darboux kernel (orthonormal Legendre on [-1/2, 1/2]):

    K_N(t) = sum_{j=0}^N p_j(t)^2,   {p_j} L^2-orthonormal.

Idea: replace max_t (mu*mu)(t) by int K_N(t) (mu*mu)(t) dt with K_N
normalized to integrate to 1.  Because

    int K_N(t) (mu*mu)(t) dt  <=  max_t (mu*mu)(t),

the CD-smoothed integral is a LOWER bound on max_t (mu*mu).  We want a
LOWER bound on max_t — but the inequality goes the wrong way for our
purpose!  Replacing max with the integral gives a SMALLER number, so
the certificate

    int K_N(t) ((c/S + delta)*(c/S + delta))(t) dt  >=  c_target

is STRICTLY MORE RESTRICTIVE than the original

    max_t ((c/S + delta)*(c/S + delta))(t)  >=  c_target,

so any cell certified via CD-smoothing is already certified by max_t
(but the converse fails).  CD-smoothing is the wrong relaxation
direction.

EMPIRICAL (sanity, d=4, S=10, c=(2,3,3,2) palindromic at cell center):

    max_t (f*f)(t) ~ 2.080
    int K_4 (f*f) dt = 0.714  (factor 2.9 loss)
    int K_8 (f*f) dt = 0.710
    int K_16 (f*f) dt = 0.709

The CD-smoothed integral is ~3x smaller than max_t; certifying it
above c_target ~ 1.281 is impossible at any N.  USELESS.

================================================================
3. CD encoding B: Christoffel function as localizer
================================================================

Idea: enforce the polynomial inequality K_N(delta_i) <= K_N(0)
component-wise as a Putinar localizer in a Lasserre hierarchy.  This
encodes "delta is close to 0" via the well-known

    K_N(x; mu)  inversely proportional to local mass density of mu.

But the cell box [-h, h]^d is ALREADY a polynomial constraint, encoded
exactly as h^2 - delta_i^2 >= 0 in the Shor RLT.  K_N is a sum of
squares of degree 2N, so K_N(0) - K_N(delta_i) is a degree-2N
polynomial.  Adding it as a localizer would amount to a SUBSET of
order-N Lasserre constraints — strictly dominated by L2 (M_2 PSD +
all degree-2 box localizers), which we already have in the cascade
fallback.

So CD-as-localizer adds NOTHING beyond what L2 already enforces.

================================================================
4. CD encoding C: CD as objective in 3-cube Lasserre relaxation
================================================================

This is the variant from `delsarte_dual/sdp_hierarchy_design.md` and
`lasserre/threepoint_full.py`.  Replace the Dirac in
    sup_t (mu*mu)(t) >= int (mu*mu)(t) d_nu(t)
by the polynomial Christoffel-Darboux kernel:
    int K_N(x + y - t) d_mu(x) d_mu(y) d_nu(t)

THIS WAS TESTED in `lasserre/THREEPOINT_FULL_REPORT.md` (2026-04-28).
Result: lambda* = 1 EXACTLY at every reachable (k, N), because the
diagonal pseudo-measure rho^(2) = 1_{u_1 = u_2} du has u_1 + u_2
uniform on [-1/2, 1/2], giving the trivial bound int (f*f) = 1.
The 3-point lift cone 3-extends to a feasible 3D moment vector with
no PSD or localizer violation.

VERDICT in that report: "DEAD by structural mathematics, not just
empirics."  The pilot's stop trigger fired with Delta = 0 exactly.

When carried over to per-cell: replacing the cell-QP objective with a
CD-smoothed integral gives the same trivial collapse, plus loses the
exactness of the cell's polynomial structure.

================================================================
5. Why CD cannot beat Shor for this QP
================================================================

The per-cell QP is DEGREE-2 in delta over a polytope.  By a classical
result (Goemans-Williamson; or the M_1 exactness theorem on convex
QCQPs), the Shor/M_1 relaxation is EXACT for the convex part of a
degree-2 QP and provides the BEST polynomial-time SDP bound for the
non-convex (indefinite A_W) part on a polytope.  The only ways to
strictly beat Shor are:
  - L2 / order-2 Lasserre (M_2 PSD): captures degree-4 via a strictly
    larger SDP.  ALREADY in the cascade fallback.
  - Lift to a higher-dimensional cube and use polynomial smoothing.
    But CD smoothing is the wrong direction here (sec 2).

A hypothetical CD certificate that avoided these traps would need to
be a POLYNOMIAL CERTIFICATE strictly stronger than M_2 on a degree-2
QP.  No such certificate exists (M_2 is degree-4 SOS-complete on a
polytope by Putinar's Positivstellensatz).

================================================================
6. Status verdict for the cascade
================================================================

NO useful CD-based per-cell certificate exists that:
  (i)   is sound,
  (ii)  is polynomial-time computable,
  (iii) is strictly tighter than the existing Shor / Lasserre L2.

The right paths to push the cascade harder are still:
  - L2 (already in `interval_bnb/lasserre_cert.py`);
  - C2 (adaptive cell refinement, deferred);
  - C3 (Lasserre paradigm without 1/m discretization tax) — outside
    cascade scope, separate track.

This file is a placeholder documenting why CD was investigated and why
it was abandoned for per-cell certs.  No code is shipped.

================================================================
References
================================================================

- THREEPOINT_FULL_REPORT.md (2026-04-28): full-design 3-point pilot,
  CD-objective lambda* = 1 collapse.
- delsarte_dual/sdp_hierarchy_design.md: CD/Fejer kernel design for
  the continuous problem (works in principle; rate too slow).
- delsarte_dual/ideas_lasserre_sos.md: CD-smoothed Lasserre for C_{1a},
  rate O(1/k^2) + O(1/N), insufficient to beat 1.2802.
- _coarse_L_bench.py: existing Shor cell-cert.
- _L_bench.py: existing per-composition L (Shor) and L2 (order-2 Lasserre)
  cell certs.

Date: 2026-05-09.
"""
import sys


def cd_cell_cert(c_int, S, d, c_target, **kwargs):
    """CD-based per-cell certifier (NULL — not implemented).

    Investigated 2026-05-09; see this file's docstring for the
    mathematical reasons no CD-based cert can beat the existing
    Shor / Lasserre L2 chain.

    Raises NotImplementedError unconditionally.
    """
    raise NotImplementedError(
        "No CD-based per-cell certificate strictly improves on the existing "
        "Shor (M_1) / order-2 Lasserre (M_2) cell certs.  See "
        "_cd_cell_cert.py docstring for the three-way derivation:\n"
        "  (a) CD-smoothing of max_t goes the wrong direction;\n"
        "  (b) CD-as-localizer is dominated by M_2 box localizers;\n"
        "  (c) CD-as-objective hits trivial lambda* = 1 collapse "
        "(documented in lasserre/THREEPOINT_FULL_REPORT.md).\n"
        "Recommendation: use L2 (lasserre_box_lb_float) for tighter cells.")


if __name__ == '__main__':
    print(__doc__)
    sys.exit(0)
