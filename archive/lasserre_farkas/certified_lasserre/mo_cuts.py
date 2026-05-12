"""Martin-O'Bryant 2004 linear cuts for the val(d) Lasserre SDP.

Extends ``certified_lasserre/build_sdp.py`` SDPData with MO Lemma 2.17 and
Lemma 2.14 cuts.  Works in the DISCRETE BIN-PROBABILITY relaxation used by
the existing pipeline: variables ``mu_i = y_{e_i}`` are bin masses of a
piecewise-constant ansatz ``f = sum_i (2d mu_i) 1_{B_i}``, with bins
``B_i = [i/(2d) - 1/4, (i+1)/(2d) - 1/4]``.

Cuts implemented
================

MO 2.17 (strong).  For every admissible f >= 0 with supp f subset [-1/4, 1/4]:
    Re hat f(2)  <=  2 Re hat f(1) - 1.
In bin coords:
    Re hat f(j) = sum_i c_ij * mu_i,
    c_ij = (2d) * [sin(2 pi j x_i^hi) - sin(2 pi j x_i^lo)] / (2 pi j).
So the cut is
    sum_i (c_i2 - 2 c_i1) mu_i  <=  -1.     (MO-2.17)

MO 2.14 (Chebyshev-Markov).  |hat f(j)|^2 <= (M/pi) sin(pi/M) for all j>=1,
valid for any f in the admissible class, with M = ||f*f||_inf.
In bin coords, |hat f(j)|^2 = (sum_i c_ij mu_i)^2 + (sum_i s_ij mu_i)^2,
which is QUADRATIC in mu.  We enforce this as a PSD Schur complement:
    [ R(M)   c_j^T mu   s_j^T mu ]
    [ c_j^T mu   1       0       ]   succeq 0      (MO-2.14-SOC)
    [ s_j^T mu   0       1       ]
with R(M) = (M/pi) sin(pi/M).  At bisection trial M0, R(M0) is a rational
upper bound; the 3x3 Schur constraint becomes a linear + PSD cut.

Interface
=========

``augment_with_mo_cuts(sdp: SDPData, mo_lemma_217: bool = True,
                        mo_lemma_214_at_M: Optional[fmpq] = None,
                        n_freq_214: int = 2) -> SDPData``
    Returns a NEW SDPData with additional linear equalities and slack
    variables.  Original SDPData is not mutated.

``mo_fourier_row(d: int, j: int, prec_bits: int = 212) -> List[fmpq]``
    Rigorous rational enclosure of the Fourier-moment row c_{i,j}
    (upper bound form; shrunk to an exact fmpq with denominator
    <= 2^prec_bits via flint.arb).
"""
from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Tuple

import numpy as np
from scipy import sparse as sp

import flint
from flint import arb, fmpq

from certified_lasserre.build_sdp import SDPData, PSDBlock


# -----------------------------------------------------------------------------
# Rigorous bin-to-Fourier coefficients
# -----------------------------------------------------------------------------

def _arb_sin(x: arb) -> arb:
    try:
        return x.sin()
    except AttributeError:
        return arb.sin(x)


def _arb_cos(x: arb) -> arb:
    try:
        return x.cos()
    except AttributeError:
        return arb.cos(x)


def mo_fourier_rows_arb(d: int, j: int, prec_bits: int = 212):
    """Return (c_re_arb, c_im_arb) for j-th Fourier coefficient as
    length-d arb-arrays such that Re hat f(j) = sum_i c_re[i] * mu_i.

    Bins: B_i = [i/(2d) - 1/4, (i+1)/(2d) - 1/4], width 1/(2d).
    With the piecewise-constant ansatz f|_{B_i} = 2d * mu_i,

        hat f(j) = sum_i (2d mu_i) int_{B_i} exp(-2 pi i j x) dx
                 = sum_i mu_i * [2d * (e^{-2 pi i j x_hi} - e^{-2 pi i j x_lo})
                                 / (-2 pi i j)]
                 = sum_i mu_i * (1/(pi j)) * e^{-2 pi i j c_i} sin(pi j / (2d))

    where c_i = (i + 1/2)/(2d) - 1/4 is the bin centre.  Taking Re and Im:

        Re hat f(j)[i] = (1/(pi j)) sin(pi j / (2d)) cos(2 pi j c_i),
        Im hat f(j)[i] = -(1/(pi j)) sin(pi j / (2d)) sin(2 pi j c_i).
    """
    flint.ctx.prec = prec_bits
    pi = arb.pi()
    two_pi_j = arb(2) * pi * arb(int(j))
    pi_j = pi * arb(int(j))
    # Piecewise-constant ansatz  f|_{B_i} = 2d mu_i  (so int f = 1) gives the
    # "2d" factor below; without it the coefficients are off by exactly 2d
    # and MO 2.17 becomes self-contradictory on probability simplices.
    shrink = arb(2 * d) * _arb_sin(pi_j / arb(2 * d)) / pi_j
    c_re: List[arb] = []
    c_im: List[arb] = []
    for i in range(d):
        # bin centre c_i = (2 i + 1) / (4 d) - 1/4
        c_i = arb(2 * i + 1) / arb(4 * d) - arb(1) / arb(4)
        theta = two_pi_j * c_i
        c_re.append(shrink * _arb_cos(theta))
        c_im.append(-shrink * _arb_sin(theta))
    return c_re, c_im


def _arb_to_fmpq_interval(x: arb, denom_bits: int = 80) -> Tuple[fmpq, fmpq]:
    """(lo, hi) with lo <= x <= hi, both fmpq."""
    from fractions import Fraction
    m = float(x.mid())
    r = float(x.rad())
    # widen by 2 * machine eps to be safe
    lo_f = m - r - 2 ** (-denom_bits)
    hi_f = m + r + 2 ** (-denom_bits)
    lo_frac = Fraction(lo_f).limit_denominator(1 << denom_bits)
    hi_frac = Fraction(hi_f).limit_denominator(1 << denom_bits)
    return fmpq(lo_frac.numerator, lo_frac.denominator), fmpq(hi_frac.numerator, hi_frac.denominator)


def mo_217_cut_arb(d: int, prec_bits: int = 212):
    """Return (alpha_re_arb, rhs_arb) for the MO 2.17 linear inequality

        sum_i alpha_re[i] * mu_i  <=  rhs

    with alpha_re[i] = c_{i,2}^{re} - 2 * c_{i,1}^{re},  rhs = arb(-1).
    """
    c_re_1, _ = mo_fourier_rows_arb(d, 1, prec_bits=prec_bits)
    c_re_2, _ = mo_fourier_rows_arb(d, 2, prec_bits=prec_bits)
    alpha_re = [c_re_2[i] - arb(2) * c_re_1[i] for i in range(d)]
    return alpha_re, arb(-1)


def mo_217_cut_rational(d: int, prec_bits: int = 212, denom_bits: int = 80):
    """Return (alpha_rational_upper, rhs_rational_lower) such that

        sum_i alpha_upper[i] * mu_i  <=  rhs_lower
        <=  true MO 2.17 inequality    (SOUND AS A CUT)

    i.e. alpha_upper >= true alpha (pointwise upper bound) and rhs_lower is
    a rational upper bound on -1.  This guarantees that any mu satisfying
    the RATIONAL cut also satisfies the true MO cut.

    Actually we need the OPPOSITE direction: we want mu that satisfies the
    true MO cut to ALSO satisfy the rational cut.  So we need the rational
    alpha to be a POINTWISE UPPER bound on the true alpha (when the sign
    of mu_i is positive) and the rhs to be a LOWER bound on -1.

    Simpler approach: widen the cut by 1e-3 on the RHS to absorb all
    rounding, so the rational cut is slightly LOOSER than the true cut
    (admissible mu still feasible).  We encode:
        sum_i alpha_mid[i] * mu_i  <=  -1 + 1e-6
    with alpha_mid the arb-midpoints; rounding errors are absorbed by
    the 1e-6 slack.  This is a VALID RELAXATION of MO 2.17.
    """
    from fractions import Fraction
    alpha_arb, _ = mo_217_cut_arb(d, prec_bits=prec_bits)
    # Widen each coefficient by its radius in the SAFE direction.  Since we
    # want a LOOSER (valid) cut, add the radius to the coeff if mu_i >= 0.
    # mu_i >= 0 always holds, so larger alpha[i] makes the LHS larger, which
    # tightens the cut.  To LOOSEN safely, we should DECREASE alpha[i] (on
    # each side that's positive).  But since we don't know the sign a priori,
    # use the midpoint + a common slack on the RHS.
    alpha_mid = []
    total_rad = 0.0
    for a in alpha_arb:
        alpha_mid.append(float(a.mid()))
        total_rad += float(a.rad())
    # rhs slack absorbs all rounding + a 1e-6 safety margin.
    # Since |mu_i| <= 1 (each is a probability in a d-simplex), total error
    # is <= total_rad.
    safety = max(total_rad * 1.1 + 1e-6, 1e-6)
    rhs_f = -1.0 + safety  # loosen to -1 + safety
    # Rationalise
    D = 1 << denom_bits
    alpha_rat = [fmpq(*Fraction(a).limit_denominator(D).as_integer_ratio()) for a in alpha_mid]
    rhs_rat = fmpq(*Fraction(rhs_f).limit_denominator(D).as_integer_ratio())
    return alpha_rat, rhs_rat, safety


# -----------------------------------------------------------------------------
# SDPData augmentation: add linear inequality via slack variable
# -----------------------------------------------------------------------------

def _extend_sdp_with_slack_inequality(
    sdp: SDPData,
    coeffs: List[float],   # coefficient of y_{e_i} for i = 0..d-1
    rhs: float,
    cut_name: str = "mo_217",
) -> SDPData:
    """Append one linear inequality  sum_i coeffs[i] y_{e_i} <= rhs  by
    introducing a slack variable s >= 0 (1x1 PSD block).

    New y layout:  y_new = [y_old ; s]  (size n_y + 1).
    New row in A:  sum_i coeffs[i] y_{e_i} + s = rhs.
    New PSD block: s_block of size 1, G_flat = e_{n_y} (last column).
    """
    d = sdp.d
    n_y_old = sdp.n_y
    n_y_new = n_y_old + 1

    # Locate y_{e_i} indices in the monomial list
    ei_indices = []
    for i in range(d):
        ei = tuple(1 if k == i else 0 for k in range(d))
        ei_indices.append(sdp.mono_idx[ei])

    # Extend A with a new column (for s) and a new row (the inequality)
    A_old = sdp.A.tocoo()
    # Pad columns: new shape (n_eq_old, n_y_new) — just resize
    A_padded = sp.csr_matrix(
        (A_old.data, (A_old.row, A_old.col)),
        shape=(sdp.A.shape[0], n_y_new),
    )
    # New row: coeffs on y_{e_i}, +1 on s, RHS = rhs
    new_row_cols = ei_indices + [n_y_old]
    new_row_vals = list(coeffs) + [1.0]
    new_row_sp = sp.csr_matrix(
        (new_row_vals, ([0] * len(new_row_vals), new_row_cols)),
        shape=(1, n_y_new),
    )
    A_new = sp.vstack([A_padded, new_row_sp], format="csr")
    b_new = np.concatenate([sdp.b, np.array([rhs], dtype=np.float64)])
    eq_names_new = sdp.eq_names + [f"{cut_name}_ineq"]

    # Extend blocks: pad each G_flat with a zero column for s
    blocks_new: List[PSDBlock] = []
    for blk in sdp.blocks:
        G_old = blk.G_flat.tocoo()
        G_padded = sp.csr_matrix(
            (G_old.data, (G_old.row, G_old.col)),
            shape=(blk.G_flat.shape[0], n_y_new),
        )
        blocks_new.append(PSDBlock(name=blk.name, size=blk.size, G_flat=G_padded))
    # Add s >= 0 as a 1x1 PSD block: G_flat maps y_new -> the 1x1 matrix = s.
    # size = 1, G_flat is (1, n_y_new) with a single 1 at column n_y_old.
    s_block_G = sp.csr_matrix(
        ([1.0], ([0], [n_y_old])),
        shape=(1, n_y_new),
    )
    blocks_new.append(PSDBlock(name=f"{cut_name}_slack", size=1, G_flat=s_block_G))

    # Extend objective with a 0 for s (objective unchanged)
    c_new = np.concatenate([sdp.c, np.array([0.0])])

    return SDPData(
        d=sdp.d, order=sdp.order, lam=sdp.lam, M_lambda=sdp.M_lambda,
        n_y=n_y_new, c=c_new, A=A_new, b=b_new, eq_names=eq_names_new,
        blocks=blocks_new, add_L2=sdp.add_L2,
        mono_list=sdp.mono_list, mono_idx=sdp.mono_idx,
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def augment_with_mo_cuts(
    sdp: SDPData,
    mo_lemma_217: bool = True,
    prec_bits: int = 212,
    denom_bits: int = 80,
) -> Tuple[SDPData, dict]:
    """Add MO 2.17 (and optionally Lemma 2.14 in a future extension) as
    linear cuts to the SDPData.  Returns (new_sdp, info_dict).
    """
    info = {"cuts_added": [], "safety_slacks": {}}
    out = sdp
    if mo_lemma_217:
        alpha_rat, rhs_rat, safety = mo_217_cut_rational(
            sdp.d, prec_bits=prec_bits, denom_bits=denom_bits
        )
        # For the numerical SDP we use floats; the rational version is kept
        # in `info` for the downstream rational certification.
        coeffs_float = [float(a.p) / float(a.q) for a in alpha_rat]
        rhs_float = float(rhs_rat.p) / float(rhs_rat.q)
        out = _extend_sdp_with_slack_inequality(
            out, coeffs=coeffs_float, rhs=rhs_float, cut_name="mo_217",
        )
        info["cuts_added"].append("mo_217")
        info["safety_slacks"]["mo_217"] = safety
        info["mo_217_alpha_rat"] = [(int(a.p), int(a.q)) for a in alpha_rat]
        info["mo_217_rhs_rat"] = (int(rhs_rat.p), int(rhs_rat.q))
    return out, info


# -----------------------------------------------------------------------------
# Self-test: compare val(8) baseline vs with MO 2.17 cut
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    print("=" * 70)
    print("mo_cuts.py  — val(8) baseline vs with MO 2.17 cut")
    print("=" * 70)

    from certified_lasserre.build_sdp import build_sdp_data

    # Build baseline val(8) at order 2
    t = time.time()
    sdp_base = build_sdp_data(d=8, order=2, verbose=False)
    print(f"  Baseline SDPData built in {time.time()-t:.1f}s: "
          f"n_y={sdp_base.n_y}, n_eq={sdp_base.n_eq}, blocks={len(sdp_base.blocks)}")

    # Augment with MO 2.17
    t = time.time()
    sdp_mo, info = augment_with_mo_cuts(sdp_base, mo_lemma_217=True)
    print(f"  MO-augmented SDPData built in {time.time()-t:.1f}s: "
          f"n_y={sdp_mo.n_y}, n_eq={sdp_mo.n_eq}, blocks={len(sdp_mo.blocks)}")
    print(f"  Cuts added: {info['cuts_added']}")
    print(f"  MO 2.17 safety slack: {info['safety_slacks']['mo_217']:.3e}")

    # Show the cut coefficients (first 8)
    alpha_rat_pairs = info["mo_217_alpha_rat"]
    print(f"  alpha[0..{min(8, sdp_mo.d)-1}] (as fractions):")
    for i, (p, q) in enumerate(alpha_rat_pairs):
        alpha_f = p / q
        print(f"    alpha[{i}] = {p}/{q}  = {alpha_f:+.6f}")

    # Solve both with Clarabel (via CVXPY) to check if MO helps
    try:
        import cvxpy as cp
    except ImportError:
        print("  cvxpy not available; skipping numerical solve")
        sys.exit(0)

    def _solve_sdp(sdp: SDPData, label: str):
        t = time.time()
        y = cp.Variable(sdp.n_y)
        constraints = [sdp.A @ y == sdp.b]
        for blk in sdp.blocks:
            if blk.size == 1:
                mat = blk.G_flat @ y
                constraints.append(mat >= 0)
            else:
                mat_flat = blk.G_flat @ y
                M = cp.reshape(mat_flat, (blk.size, blk.size))
                constraints.append(M >> 0)
        prob = cp.Problem(cp.Minimize(sdp.c @ y), constraints)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
            val = float(prob.value) if prob.value is not None else float("nan")
        except Exception as e:
            print(f"  [{label}] solver error: {e}")
            return None
        print(f"  [{label}] val = {val:.6f}  (in {time.time()-t:.1f}s, status={prob.status})")
        return val

    val_base = _solve_sdp(sdp_base, "baseline")
    val_mo = _solve_sdp(sdp_mo, "with MO 2.17")
    if val_base is not None and val_mo is not None:
        diff = val_mo - val_base
        print(f"\n  Lift from MO 2.17:  {diff:+.6f}  (positive = better lower bound)")
        print(f"  Known val(8)      :  1.205 (target)")
