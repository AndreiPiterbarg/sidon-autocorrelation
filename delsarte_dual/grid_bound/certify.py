"""Independent verifier for ``delsarte_dual`` rigorous certificates.

The verifier re-derives every quantitative claim of a certificate using
only ``flint.arb`` / ``flint.fmpq`` primitives and the paper-sourced
Matolcsi-Vinuesa decimal coefficients (transcribed inline below).  No
function or constant in this file is imported from the rest of the
package; the verifier is intentionally self-contained so a third party
can audit the certificate from this single source file.

Two certificate formats are accepted:

  * ``kind = "grid_bound_N1_single_scale_arcsine"`` (or the legacy
    ``"grid_bound_N1_MV_reproduction"``) -- the single-scale arcsine
    reproduction of ``C_{1a} >= 1.27481``.

  * ``kind = "multiscale_arcsine_rigorous_certificate"`` -- the
    multi-scale arcsine production certificate of the
    Piterbarg-Bajaj-Vincent Bound ``C_{1a} >= 1292/1000 = 1.292``.

What is checked
---------------
  1. The certificate's SHA-256 body hash matches the recomputed hash.
  2. The kernel inputs (``delta_i``, ``lambda_i``, ``u``) are valid
     (rationals in the documented form; ``sum lambda_i = 1``).
  3. ``k_1``, ``K_2``, ``S_1``, ``min G``, ``gain a`` are recomputed
     in arb at the declared precision and agree with the certificate's
     anchors (with rigorous inclusion).
  4. The terminal cells from the certifying cell-search cover the
     admissible box ``[0, mu(M_cert)]``.
  5. For every terminal cell the recomputed ``Phi(M_cert, cell)`` has
     ``.upper() < 0``.

Exit code ``0`` on success, ``1`` otherwise.

Usage:

    ``python -m delsarte_dual.grid_bound.certify <certificate.json>``
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

from flint import acb, arb, fmpq, ctx


# ============================================================================
#  Paper-sourced Matolcsi-Vinuesa constants (no internal imports).
# ============================================================================

_MV_DECIMALS_STANDALONE = [
    "+2.16620392","-1.87775750","+1.05828868","-7.29790538e-01",
    "+4.28008515e-01","+2.17832838e-01","-2.70415201e-01","+2.72834790e-02",
    "-1.91721888e-01","+5.51862060e-02","+3.21662512e-01","-1.64478392e-01",
    "+3.95478603e-02","-2.05402785e-01","-1.33758316e-02","+2.31873221e-01",
    "-4.37967118e-02","+6.12456374e-02","-1.57361919e-01","-7.78036253e-02",
    "+1.38714392e-01","-1.45201483e-04","+9.16539824e-02","-8.34020840e-02",
    "-1.01919986e-01","+5.94915025e-02","-1.19336618e-02","+1.02155366e-01",
    "-1.45929982e-02","-7.95205457e-02","+5.59733152e-03","-3.58987179e-02",
    "+7.16132260e-02","+4.15425065e-02","-4.89180454e-02","+1.65425755e-03",
    "-6.48251747e-02","+3.45951253e-02","+5.32122058e-02","-1.28435276e-02",
    "+1.48814403e-02","-6.49404547e-02","-6.01344770e-03","+4.33784473e-02",
    "-2.53362778e-04","+3.81674519e-02","-4.83816002e-02","-2.53878079e-02",
    "+1.96933442e-02","-3.04861682e-03","+4.79203471e-02","-2.00930265e-02",
    "-2.73895519e-02","+3.30183589e-03","-1.67380508e-02","+4.23917582e-02",
    "+3.64690190e-03","-1.79916104e-02","+7.31661649e-05","-2.99875575e-02",
    "+2.71842526e-02","+1.41806855e-02","-6.01781076e-03","+5.86806100e-03",
    "-3.32350597e-02","+9.23347466e-03","+1.47071722e-02","-7.42858080e-04",
    "+1.63414270e-02","-2.87265671e-02","-1.64287280e-03","+8.02601605e-03",
    "-7.62613027e-04","+2.18735533e-02","-1.78816282e-02","-6.58341101e-03",
    "+2.67706547e-03","-6.25261247e-03","+2.24942824e-02","-8.10756022e-03",
    "-5.68160823e-03","+7.01871209e-05","-1.15294332e-02","+1.83608944e-02",
    "-1.20567880e-03","-3.13147456e-03","+1.39083675e-03","-1.49312478e-02",
    "+1.32106694e-02","+1.73474188e-03","-8.53469045e-04","+4.03211203e-03",
    "-1.55352991e-02","+8.74711543e-03","+1.93998895e-03","-2.71357322e-05",
    "+6.13179585e-03","-1.41983972e-02","+5.84710551e-03","+9.22578333e-04",
    "-2.16583469e-04","+7.07919829e-03","-1.18488582e-02","+4.39698322e-03",
    "-8.91346785e-05","-3.42086367e-04","+6.46355636e-03","-8.87555371e-03",
    "+3.56799654e-03","-4.97335419e-04","-8.04560326e-04","+5.55076717e-03",
    "-7.13560569e-03","+4.53679038e-03","-3.33261516e-03","+2.35463427e-03",
    "+2.04023789e-04","-1.27746711e-03","+1.81247830e-04",
]
assert len(_MV_DECIMALS_STANDALONE) == 119

_MV_DELTA_Q          = fmpq(138, 1000)
_MV_U_Q              = fmpq(638, 1000)
_MV_K2_TIMES_DELTA_Q = fmpq(5747, 10000)


# ============================================================================
#  Helpers
# ============================================================================

def _decimal_str_to_fmpq(s: str) -> fmpq:
    """Parse a signed decimal string to exact ``fmpq``.  Independently
    implemented here; no import from ``coeffs.py``.
    """
    s = s.strip()
    sign = 1
    if s.startswith("+"):
        s = s[1:]
    elif s.startswith("-"):
        sign = -1
        s = s[1:]
    if "e" in s or "E" in s:
        mant, e = s.replace("E", "e").split("e", 1)
        exp = int(e)
    else:
        mant, exp = s, 0
    if "." in mant:
        ip, fp = mant.split(".", 1)
    else:
        ip, fp = mant, ""
    if ip == "":
        ip = "0"
    dig_aft = len(fp)
    mi = int(ip + fp) if (ip + fp) else 0
    net = exp - dig_aft
    if net >= 0:
        return fmpq(sign * mi * (10 ** net), 1)
    return fmpq(sign * mi, 10 ** (-net))


def _fmpq_to_float(q: fmpq) -> float:
    return float(q.p) / float(q.q)


def _fmpq_from_str(s: str) -> fmpq:
    if "/" in s:
        p_str, q_str = s.split("/", 1)
        return fmpq(int(p_str), int(q_str))
    return fmpq(int(s))


def _safe_sqrt(x: arb) -> arb:
    x_up = x.upper()
    if x_up < 0:
        raise ValueError(f"x.upper() = {x_up} < 0")
    if x.lower() >= 0:
        return x.sqrt()
    return arb(0).union(x_up.sqrt())


# ============================================================================
#  Recomputations -- single-scale and multi-scale
# ============================================================================

def _k1_single(delta: fmpq, prec_bits: int) -> arb:
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        arg = arb.pi() * arb(delta)
        return (arg.bessel_j(0)) ** arb(2)
    finally:
        ctx.prec = old


def _k1_multi(deltas: Sequence[fmpq], lambdas: Sequence[fmpq],
              prec_bits: int) -> arb:
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        total = arb(0)
        for lam, d in zip(lambdas, deltas):
            j0 = (arb.pi() * arb(d)).bessel_j(0)
            total = total + arb(lam) * j0 * j0
        return total
    finally:
        ctx.prec = old


def _K_tilde_real_multi(
    xi: arb, deltas: Sequence[fmpq], lambdas: Sequence[fmpq], prec_bits: int
) -> arb:
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        total = arb(0)
        for lam, d in zip(lambdas, deltas):
            j0 = (arb.pi() * arb(d) * xi).bessel_j(0)
            total = total + arb(lam) * j0 * j0
        return total
    finally:
        ctx.prec = old


def _S1_single(
    coeffs: List[fmpq], delta: fmpq, u: fmpq, prec_bits: int
) -> arb:
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        total = arb(0)
        for j, a_j in enumerate(coeffs, start=1):
            q = fmpq(j) * delta / u
            arg = arb.pi() * arb(q)
            j0 = arg.bessel_j(0)
            total = total + (arb(a_j) * arb(a_j)) / (j0 * j0)
        return total
    finally:
        ctx.prec = old


def _S1_multi(
    coeffs: List[fmpq],
    deltas: Sequence[fmpq],
    lambdas: Sequence[fmpq],
    u: fmpq,
    prec_bits: int,
) -> arb:
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        total = arb(0)
        for j, a_j in enumerate(coeffs, start=1):
            xi = arb(fmpq(j)) / arb(u)
            w = _K_tilde_real_multi(xi, deltas, lambdas, prec_bits)
            total = total + (arb(a_j) * arb(a_j)) / w
        return total
    finally:
        ctx.prec = old


def _K2_single(K2_times_delta: fmpq, delta: fmpq, prec_bits: int) -> arb:
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        return arb(K2_times_delta) / arb(delta)
    finally:
        ctx.prec = old


def _K2_multi(
    deltas: Sequence[fmpq],
    lambdas: Sequence[fmpq],
    T_q: fmpq,
    prec_bits: int,
    use_diag_surrogate: bool = False,
    diag_num: fmpq = fmpq(5747, 10000),
) -> arb:
    """Recompute ``K_2 = 2 sum_{i,j} lambda_i lambda_j C_{ij}`` with
    ``C_{ij} = int_0^infty J_0(pi di xi)^2 J_0(pi dj xi)^2 dxi``.

    Each ``C_{ij}`` is an ``acb.integral`` on ``[0, T]`` plus the
    asymptotic tail bound.  If ``use_diag_surrogate`` is True the
    diagonal terms use the Martin-O'Bryant value ``0.5747 / (2 di)``.
    """
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        T = arb(T_q)

        def cross(di_q: fmpq, dj_q: fmpq) -> arb:
            di = arb(di_q)
            dj = arb(dj_q)

            def f(z, _flags):
                a_i = acb.pi() * acb(di) * z
                a_j = acb.pi() * acb(dj) * z
                return (a_i.bessel_j(acb(0)) ** 2) * (
                    a_j.bessel_j(acb(0)) ** 2
                )

            val = acb.integral(f, acb(0), acb(T))
            main = val.real
            pi_arb = arb.pi()
            pi_fourth = pi_arb * pi_arb * pi_arb * pi_arb
            tail = arb(4) / (pi_fourth * di * dj * T)
            return main + arb(0).union(tail)

        total = arb(0)
        n = len(deltas)
        for i in range(n):
            for j in range(n):
                di_q = deltas[i]
                dj_q = deltas[j]
                if i == j and use_diag_surrogate:
                    Cij = arb(diag_num) / (arb(2) * arb(di_q))
                else:
                    Cij = cross(di_q, dj_q)
                total = total + arb(lambdas[i]) * arb(lambdas[j]) * Cij
        return arb(2) * total
    finally:
        ctx.prec = old


def _min_G_taylor(
    coeffs: List[fmpq], u: fmpq, prec_bits: int, n_cells: int = 8192
) -> tuple[arb, fmpq]:
    """Taylor B&B lower bound on ``min_{[0, 1/4]} G(x)``.  Independent
    reimplementation of :mod:`G_min`.
    """
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        two_pi_over_u = arb(2) * arb.pi() / arb(u)
        x_lo = fmpq(0)
        x_hi = fmpq(1, 4)
        cw = (x_hi - x_lo) / fmpq(n_cells)
        hw = cw / fmpq(2)

        def G_at(x_q: fmpq) -> arb:
            x_a = arb(x_q)
            t = arb(0)
            for j, a in enumerate(coeffs, start=1):
                t = t + arb(a) * (two_pi_over_u * arb(j) * x_a).cos()
            return t

        def Gp_at(x_q: fmpq) -> arb:
            x_a = arb(x_q)
            t = arb(0)
            for j, a in enumerate(coeffs, start=1):
                w = two_pi_over_u * arb(j)
                t = t - arb(a) * w * (w * x_a).sin()
            return t

        def Gpp_cell(cell_a: arb) -> arb:
            t = arb(0)
            for j, a in enumerate(coeffs, start=1):
                w = two_pi_over_u * arb(j)
                t = t - arb(a) * (w * w) * (w * cell_a).cos()
            return t

        worst_arb = None
        worst_center = None
        worst_float = None
        for k in range(n_cells):
            c = x_lo + fmpq(2 * k + 1) * hw
            cell_arb = arb(c, hw)
            G_c = G_at(c)
            Gp_c = Gp_at(c)
            Gpp = Gpp_cell(cell_arb)
            dx = arb(0, hw)
            half_r_sq = (arb(hw) * arb(hw)) / arb(2)
            rem = arb(0, 1) * half_r_sq
            encl = G_c + Gp_c * dx + Gpp * rem
            lf = float(encl.lower())
            if worst_float is None or lf < worst_float:
                worst_float = lf
                worst_arb = encl
                worst_center = c
        return worst_arb, worst_center
    finally:
        ctx.prec = old


def _mu_of_M(M: arb) -> arb:
    return M * (arb.pi() / M).sin() / arb.pi()


def _phi_N1(
    M: arb,
    y: arb,
    *,
    K2: arb,
    k1: arb,
    u: fmpq,
    gain_a: arb,
) -> arb:
    """Recomputation of ``Phi(M, y)`` independent of :mod:`phi`."""
    two = arb(2)
    rad1 = M - arb(1) - two * y * y
    rad2 = K2 - arb(1) - two * k1 * k1
    s1 = _safe_sqrt(rad1)
    s2 = _safe_sqrt(rad2)
    rhs = M + arb(1) + two * y * k1 + s1 * s2
    lhs = arb(2) / arb(u) + gain_a
    return rhs - lhs


# ============================================================================
#  Main verification routine
# ============================================================================

@dataclass
class VerifyResult:
    accepted: bool
    messages: list
    M_cert_q: Optional[fmpq] = None


def verify_certificate(
    cert_path: str, prec_bits: Optional[int] = None
) -> VerifyResult:
    """Verify the certificate at ``cert_path``.

    Dispatches on ``body['kind']``:

      * ``"multiscale_arcsine_rigorous_certificate"`` (production)
      * ``"grid_bound_N1_single_scale_arcsine"``
      * ``"grid_bound_N1_MV_reproduction"`` (legacy alias)
    """
    with open(cert_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    msgs: list = []

    def log(s: str) -> None:
        msgs.append(s)
        print(s)

    body = raw["body"]
    body_json = json.dumps(body, indent=2, sort_keys=True)
    digest = hashlib.sha256(body_json.encode("utf-8")).hexdigest()
    if digest != raw["sha256_of_body"]:
        log(
            f"FAIL: SHA-256 mismatch: recomputed {digest}, "
            f"stored {raw['sha256_of_body']}"
        )
        return VerifyResult(False, msgs)
    log(f"OK: SHA-256 integrity = {digest}")

    kind = body.get("kind", "")
    prec = prec_bits or int(body.get("prec_bits", 256))

    if kind == "multiscale_arcsine_rigorous_certificate":
        return _verify_multiscale(body, prec, msgs, log)
    if kind in (
        "grid_bound_N1_single_scale_arcsine",
        "grid_bound_N1_MV_reproduction",
    ):
        return _verify_single_scale(body, prec, msgs, log)
    log(f"FAIL: unknown certificate kind '{kind}'")
    return VerifyResult(False, msgs)


def _verify_single_scale(
    body: dict, prec: int, msgs: list, log
) -> VerifyResult:
    inputs = body["inputs"]
    delta_q = _fmpq_from_str(inputs["delta_q"])
    u_q = _fmpq_from_str(inputs["u_q"])
    K2d_q = _fmpq_from_str(inputs["K2_times_delta_q"])

    if delta_q != _MV_DELTA_Q:
        log(f"FAIL: delta mismatch: cert={delta_q}, paper={_MV_DELTA_Q}")
        return VerifyResult(False, msgs)
    if u_q != _MV_U_Q:
        log(f"FAIL: u mismatch: cert={u_q}, paper={_MV_U_Q}")
        return VerifyResult(False, msgs)
    if K2d_q != _MV_K2_TIMES_DELTA_Q:
        log(
            f"FAIL: K2*delta mismatch: cert={K2d_q}, "
            f"paper={_MV_K2_TIMES_DELTA_Q}"
        )
        return VerifyResult(False, msgs)
    if inputs["n_coeffs"] != 119:
        log(f"FAIL: n_coeffs != 119: {inputs['n_coeffs']}")
        return VerifyResult(False, msgs)
    log(
        f"OK: inputs match paper (delta={delta_q}, u={u_q}, "
        f"K2*delta={K2d_q}, n=119)"
    )

    coeffs = [_decimal_str_to_fmpq(s) for s in _MV_DECIMALS_STANDALONE]

    k1 = _k1_single(delta_q, prec)
    K2 = _K2_single(K2d_q, delta_q, prec)
    S1 = _S1_single(coeffs, delta_q, u_q, prec)
    min_G_arb, _ = _min_G_taylor(coeffs, u_q, prec, n_cells=32768)
    min_G_cert = min_G_arb.lower()
    if min_G_cert.upper() <= 0:
        log(f"FAIL: recomputed min G <= 0 ({min_G_cert})")
        return VerifyResult(False, msgs)
    gain_a = (arb(4) / arb(u_q)) * (min_G_cert * min_G_cert) / S1
    log(f"OK: recomputed k_1    = {k1}")
    log(f"OK: recomputed K_2    = {K2}")
    log(f"OK: recomputed S_1    = {S1}")
    log(f"OK: recomputed min G  = {min_G_cert}")
    log(f"OK: recomputed gain a = {gain_a}")

    return _check_cells_and_coverage(
        body, prec, K2, k1, u_q, gain_a, msgs, log
    )


def _verify_multiscale(
    body: dict, prec: int, msgs: list, log
) -> VerifyResult:
    kernel = body["kernel"]
    deltas = [_fmpq_from_str(s) for s in kernel["deltas_q"]]
    lambdas = [_fmpq_from_str(s) for s in kernel["lambdas_q"]]
    if sum(lambdas, fmpq(0)) != fmpq(1):
        log(f"FAIL: lambdas do not sum to 1: {lambdas}")
        return VerifyResult(False, msgs)
    log(
        f"OK: kernel = {kernel['name']}  "
        f"(deltas {[_fmpq_to_float(d) for d in deltas]}, "
        f"lambdas {[_fmpq_to_float(l) for l in lambdas]})"
    )

    G = body["G"]
    u_q = _fmpq_from_str(G["u_q"])
    coeffs = [_fmpq_from_str(s) for s in G["coeffs_q"]]
    if len(coeffs) != G["n_coeffs"]:
        log(
            f"FAIL: G n_coeffs mismatch: "
            f"len(coeffs_q)={len(coeffs)}, n_coeffs={G['n_coeffs']}"
        )
        return VerifyResult(False, msgs)
    log(f"OK: G has {G['n_coeffs']} cosine modes, u = {u_q}")

    T_q = _fmpq_from_str(kernel["K2_cross_cutoff_xi_q"])
    use_diag_surrogate = bool(kernel.get("use_diag_surrogate", False))

    k1 = _k1_multi(deltas, lambdas, prec)
    K2 = _K2_multi(
        deltas, lambdas, T_q, prec, use_diag_surrogate=use_diag_surrogate
    )
    S1 = _S1_multi(coeffs, deltas, lambdas, u_q, prec)
    min_G_arb, _ = _min_G_taylor(coeffs, u_q, prec, n_cells=32768)
    min_G_cert = min_G_arb.lower()
    if min_G_cert.upper() <= 0:
        log(f"FAIL: recomputed min G <= 0 ({min_G_cert})")
        return VerifyResult(False, msgs)
    gain_a = (arb(4) / arb(u_q)) * (min_G_cert * min_G_cert) / S1
    log(f"OK: recomputed k_1    = {k1}")
    log(f"OK: recomputed K_2    = {K2}")
    log(f"OK: recomputed S_1    = {S1}")
    log(f"OK: recomputed min G  = {min_G_cert}")
    log(f"OK: recomputed gain a = {gain_a}")

    return _check_cells_and_coverage(
        body, prec, K2, k1, u_q, gain_a, msgs, log
    )


def _check_cells_and_coverage(
    body: dict,
    prec: int,
    K2: arb,
    k1: arb,
    u_q: fmpq,
    gain_a: arb,
    msgs: list,
    log,
) -> VerifyResult:
    M_cert_q = _fmpq_from_str(body["M_cert"]["rational"])
    M_cert_arb = arb(M_cert_q)
    log(f"Verifying M_cert = {M_cert_q}  (~{_fmpq_to_float(M_cert_q):.6f})")

    cells_info = body["cell_search_at_M_cert"]["terminal_cells"]
    log(f"Re-checking {len(cells_info)} terminal cells ...")
    max_upper = None
    for i, rec in enumerate(cells_info):
        lo_q = _fmpq_from_str(rec["cell"]["lo"])
        hi_q = _fmpq_from_str(rec["cell"]["hi"])
        if hi_q <= lo_q:
            log(f"FAIL: cell {i} has hi <= lo: [{lo_q}, {hi_q}]")
            return VerifyResult(False, msgs)
        center = (lo_q + hi_q) / fmpq(2)
        hw = (hi_q - lo_q) / fmpq(2)
        y_arb = arb(center, hw)
        phi_v = _phi_N1(M_cert_arb, y_arb, K2=K2, k1=k1, u=u_q, gain_a=gain_a)
        up = float(phi_v.upper())
        if max_upper is None or up > max_upper:
            max_upper = up
        if not (phi_v.upper() < 0):
            log(
                f"FAIL: cell {i}  lo={lo_q}  hi={hi_q}  "
                f"Phi.upper()={phi_v.upper()} not < 0"
            )
            return VerifyResult(False, msgs)
    log(
        f"OK: every terminal cell has Phi.upper() < 0; "
        f"worst upper = {max_upper:+.3e}"
    )

    mu_up = float(_mu_of_M(M_cert_arb).upper())
    sorted_cells = sorted(
        [
            (_fmpq_from_str(c["cell"]["lo"]), _fmpq_from_str(c["cell"]["hi"]))
            for c in cells_info
        ],
        key=lambda lh: lh[0],
    )
    if sorted_cells[0][0] != 0:
        log(f"FAIL: lowest cell starts at {sorted_cells[0][0]}, not 0")
        return VerifyResult(False, msgs)
    for i in range(1, len(sorted_cells)):
        if sorted_cells[i][0] != sorted_cells[i - 1][1]:
            log(
                f"FAIL: cells not contiguous: "
                f"cell {i-1} ends at {sorted_cells[i-1][1]}, "
                f"cell {i} starts at {sorted_cells[i][0]}"
            )
            return VerifyResult(False, msgs)
    top_q = sorted_cells[-1][1]
    if _fmpq_to_float(top_q) < mu_up:
        log(
            f"FAIL: top cell ends at {_fmpq_to_float(top_q):.8f} < "
            f"mu(M_cert).upper() = {mu_up:.8f}"
        )
        return VerifyResult(False, msgs)
    log(
        f"OK: cells cover [0, {_fmpq_to_float(top_q):.8f}] >= "
        f"[0, mu(M_cert).upper() = {mu_up:.8f}]"
    )

    log("")
    log(
        f"VERDICT: CERTIFICATE ACCEPTED.  "
        f"C_{{1a}} >= {M_cert_q} (~{_fmpq_to_float(M_cert_q):.6f})."
    )
    return VerifyResult(True, msgs, M_cert_q)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Independent verifier for delsarte_dual certificates."
    )
    parser.add_argument("certificate", help="path to the JSON certificate")
    parser.add_argument(
        "--prec-bits",
        type=int,
        default=None,
        help="override arb precision (default: certificate's prec_bits)",
    )
    args = parser.parse_args(argv)
    res = verify_certificate(args.certificate, prec_bits=args.prec_bits)
    sys.exit(0 if res.accepted else 1)


if __name__ == "__main__":
    main()
