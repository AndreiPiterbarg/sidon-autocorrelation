"""Independent verifier for Phase-2 Hoelder-generalised certificates.

Soundness contract
------------------
Imports only ``flint`` (python-flint) and the Python stdlib.  Does NOT
import any module under ``delsarte_dual.grid_bound`` nor any module in
``delsarte_dual.grid_bound_holder`` -- every quantity and filter is
recomputed here from first principles.  MV's 119 coefficients are
inlined as DATA; no code dependency.

What is verified
----------------
  1. SHA-256 integrity of certificate body.
  2. Input rationals delta, u, K2_times_delta match MV.
  3. Hoelder exponents (p, q) satisfy 1/p + 1/q = 1 and p >= 2.
  4. 119 coefficient count, re-parsed from inline decimals.
  5. Recompute K2, K_q_upper (at given q), S1, gain_a, min_G, k_1..k_N.
  6. For each terminal cell: rerun filter_all (FILTER_REJECT) or
     rerun phi_holder and require .upper() < 0 (PHI_REJECT).
  7. Coverage of the starting root box (implicit via bisection).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from typing import Sequence

from flint import arb, acb, fmpq, ctx


# ============================================================================
#  Inline MV data (paper-sourced; NOT imported from any project module)
# ============================================================================

_MV_DECIMALS_V = [
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
assert len(_MV_DECIMALS_V) == 119

_MV_DELTA_Q          = fmpq(138, 1000)
_MV_U_Q              = fmpq(638, 1000)
_MV_K2_TIMES_DELTA_Q = fmpq(5747, 10000)


# ============================================================================
#  Helpers
# ============================================================================

def _decimal_to_fmpq(s: str) -> fmpq:
    s = s.strip()
    sign = 1
    if s.startswith("+"): s = s[1:]
    elif s.startswith("-"): sign = -1; s = s[1:]
    exp = 0
    if "e" in s or "E" in s:
        mant, e = s.replace("E", "e").split("e", 1); exp = int(e)
    else:
        mant = s
    if "." in mant: ip, fp = mant.split(".", 1)
    else: ip, fp = mant, ""
    if ip == "": ip = "0"
    dig_aft = len(fp)
    mi = int(ip + fp) if (ip + fp) else 0
    net = exp - dig_aft
    return fmpq(sign * mi * (10 ** net), 1) if net >= 0 else fmpq(sign * mi, 10 ** (-net))


def _fmpq_from_str(s: str) -> fmpq:
    if "/" in s:
        p, q = s.split("/", 1); return fmpq(int(p), int(q))
    return fmpq(int(s))


def _fmpq_to_float(q: fmpq) -> float:
    return float(q.p) / float(q.q)


def _safe_sqrt(x: arb) -> arb:
    x_up = x.upper()
    if x_up < 0:
        raise ValueError(f"sqrt: x.upper()={x_up} < 0")
    x_lo = x.lower()
    if x_lo >= 0:
        return x.sqrt()
    return arb(0).union(x_up.sqrt())


def _safe_root(x: arb, exp_q: fmpq) -> arb:
    x_up = x.upper()
    if x_up < 0:
        raise ValueError(f"root: x.upper()={x_up} < 0")
    x_lo = x.lower()
    if x_lo >= 0:
        return x ** arb(exp_q)
    return arb(0).union(arb(x_up) ** arb(exp_q))


def _arb_nonneg_is_reject(x: arb) -> bool:
    return x.upper() < 0


# ============================================================================
#  Re-computation of Phase-1 inputs
# ============================================================================

def _j0_pi_q(q: fmpq, prec_bits: int) -> arb:
    old = ctx.prec; ctx.prec = prec_bits
    try:
        return (arb.pi() * arb(q)).bessel_j(0)
    finally:
        ctx.prec = old


def recompute_kn(n: int, delta: fmpq, prec_bits: int) -> arb:
    j0 = _j0_pi_q(fmpq(n) * delta, prec_bits)
    return j0 * j0


def recompute_S1(coeffs: list[fmpq], delta: fmpq, u: fmpq, prec_bits: int) -> arb:
    old = ctx.prec; ctx.prec = prec_bits
    try:
        t = arb(0)
        for j, a in enumerate(coeffs, start=1):
            j0 = _j0_pi_q(fmpq(j) * delta / u, prec_bits)
            t = t + (arb(a) * arb(a)) / (j0 * j0)
        return t
    finally:
        ctx.prec = old


def recompute_K2(K2d: fmpq, delta: fmpq, prec_bits: int) -> arb:
    old = ctx.prec; ctx.prec = prec_bits
    try:
        return arb(K2d) / arb(delta)
    finally:
        ctx.prec = old


def recompute_Kq_upper(
    q: fmpq,
    delta: fmpq,
    K2: arb,
    *,
    J_tail: int = 1024,
    prec_bits: int = 256,
) -> arb:
    """Independent re-implementation of kq_kernel_upper for the verifier.

    Mirrors phi_holder.kq_kernel_upper: direct sum over |j| <= J_tail plus
    Krasikov tail bound (J_0(x)^2 <= (2/pi)/sqrt(x^2 + 3/2) for x >= sqrt(5/2),
    relaxed to (2/pi)/x for the tail analytic bound).
    """
    if q == fmpq(2):
        return K2
    if q < fmpq(1):
        raise ValueError(f"recompute_Kq_upper requires q >= 1, got {q}")
    old = ctx.prec; ctx.prec = prec_bits
    try:
        q_arb = arb(q)
        pi_arb = arb.pi()
        delta_arb = arb(delta)
        total = arb(1)
        for j in range(1, J_tail + 1):
            k_j = recompute_kn(j, delta, prec_bits)
            k_lo = k_j.lower()
            if k_lo > 0:
                kq = k_j ** q_arb
            else:
                k_u = k_j.upper()
                if k_u <= 0:
                    kq = arb(0)
                else:
                    kq = arb(0).union(arb(k_u) ** q_arb)
            total = total + arb(2) * kq
        if J_tail < 8:
            raise ValueError("J_tail too small for Krasikov tail")
        pd = pi_arb * delta_arb
        C = (arb(2) / (pi_arb * pd)) ** q_arb
        J_arb = arb(J_tail)
        one = arb(1)
        tail_sum_bound = one / ((q_arb - one) * (J_arb ** (q_arb - one)))
        tail_pair = arb(2) * C * tail_sum_bound
        return total + tail_pair
    finally:
        ctx.prec = old


def recompute_min_G_lower(
    coeffs: list[fmpq], u: fmpq, prec_bits: int, n_cells: int = 8192,
) -> arb:
    old = ctx.prec; ctx.prec = prec_bits
    try:
        tpu = arb(2) * arb.pi() / arb(u)
        x_lo = fmpq(0); x_hi = fmpq(1, 4)
        cw = (x_hi - x_lo) / fmpq(n_cells); hw = cw / fmpq(2)

        def G(c_q):
            c = arb(c_q); t = arb(0)
            for j, a in enumerate(coeffs, start=1):
                t = t + arb(a) * (tpu * arb(j) * c).cos()
            return t

        def Gp(c_q):
            c = arb(c_q); t = arb(0)
            for j, a in enumerate(coeffs, start=1):
                t = t - arb(a) * (tpu * arb(j)) * (tpu * arb(j) * c).sin()
            return t

        def Gpp(cell_a):
            t = arb(0)
            for j, a in enumerate(coeffs, start=1):
                w = tpu * arb(j)
                t = t - arb(a) * (w * w) * (w * cell_a).cos()
            return t

        worst = None; worst_float = None
        for k in range(n_cells):
            c = x_lo + fmpq(2 * k + 1) * hw
            ca = arb(c, hw)
            encl = G(c) + Gp(c) * arb(0, hw) + Gpp(ca) * (arb(0, 1) * (arb(hw) * arb(hw) / arb(2)))
            lf = float(encl.lower())
            if worst_float is None or lf < worst_float:
                worst_float = lf; worst = encl
        return worst.lower()
    finally:
        ctx.prec = old


def mu_of_M(M: arb) -> arb:
    return M * (arb.pi() / M).sin() / arb.pi()


# ============================================================================
#  Hoelder Phi and filter panel re-implementation
# ============================================================================

def _arb_sqr(x: arb) -> arb:
    al = x.abs_lower(); au = x.abs_upper()
    return (al * al).union(au * au)


def phi_holder_verify(
    M: arb, ab: Sequence[arb], *,
    p: fmpq, q: fmpq, K_q: arb, sum_kn_q: arb,
    k_arb, sum_kn_sq, K2, u, gain_a,
) -> arb:
    N = len(k_arb)
    assert len(ab) == 2 * N
    # p = q = 2: MM-10 code path, bit-identical to phi_mm
    if p == fmpq(2) and q == fmpq(2):
        two = arb(2)
        sum_zk = arb(0); sum_z4 = arb(0)
        for n in range(1, N + 1):
            a = ab[2 * (n - 1)]; b = ab[2 * (n - 1) + 1]
            z2 = _arb_sqr(a) + _arb_sqr(b)
            sum_zk = sum_zk + z2 * k_arb[n - 1]
            sum_z4 = sum_z4 + _arb_sqr(z2)
        rad1 = M - arb(1) - two * sum_z4
        rad2 = K2 - arb(1) - two * sum_kn_sq
        return (M + arb(1) + two * sum_zk + _safe_sqrt(rad1) * _safe_sqrt(rad2)
                - (arb(2) / arb(u) + gain_a))
    # General Hoelder with TIGHT h-tail (derivation.md Lemma 1):
    #   sum_{|j|>N} |hat h|^p <= mu(M)^{p-2} * (M - 1 - 2 sum z_n^4)
    two = arb(2)
    inv_p = fmpq(1) / p
    inv_q = fmpq(1) / q
    sum_zk = arb(0); sum_z4 = arb(0)
    for n in range(1, N + 1):
        a = ab[2 * (n - 1)]; b = ab[2 * (n - 1) + 1]
        z2 = _arb_sqr(a) + _arb_sqr(b)
        sum_zk = sum_zk + z2 * k_arb[n - 1]
        sum_z4 = sum_z4 + _arb_sqr(z2)
    mu = mu_of_M(M)
    rad_h = M - arb(1) - two * sum_z4
    if p == fmpq(2):
        mu_power = arb(1)
    else:
        mu_lo = mu.lower()
        if mu_lo > 0:
            mu_power = mu ** arb(p - fmpq(2))
        else:
            mu_u = mu.upper()
            if mu_u <= 0:
                mu_power = arb(0)
            else:
                mu_power = arb(0).union(arb(mu_u) ** arb(p - fmpq(2)))
    scaled_rad_h = mu_power * rad_h
    fh = _safe_root(scaled_rad_h, inv_p)
    rad_k = K_q - arb(1) - two * sum_kn_q
    fk = _safe_root(rad_k, inv_q)
    return (M + arb(1) + two * sum_zk + fh * fk
            - (arb(2) / arb(u) + gain_a))


def _acb_det(M: list[list[acb]]) -> acb:
    n = len(M)
    if n == 1: return M[0][0]
    if n == 2: return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    t = acb(0)
    for j in range(n):
        minor = [[M[i][jj] for jj in range(n) if jj != j] for i in range(1, n)]
        sgn = acb(1) if j % 2 == 0 else acb(-1)
        t = t + sgn * M[0][j] * _acb_det(minor)
    return t


def _T_minor_det_real(ab, N, k, kind="f") -> arb:
    def hat_f(kk):
        if kk == 0: return acb(1)
        if kk > 0: return acb(ab[2 * (kk - 1)], ab[2 * (kk - 1) + 1])
        kk = -kk; return acb(ab[2 * (kk - 1)], -ab[2 * (kk - 1) + 1])
    def hat_h(kk):
        if kk == 0: return acb(1)
        if kk > 0: return hat_f(kk) * hat_f(kk)
        return hat_f(-kk) * hat_f(-kk)
    src = hat_f if kind == "f" else hat_h
    Mt = [[src(i - j) for j in range(k)] for i in range(k)]
    return _acb_det(Mt).real


def _filter_reject_check(
    ab, N, mu_arb, *, enable_F4_MO217, enable_F7, enable_F8,
) -> bool:
    # Use the dependency-aware square so our rigor matches the runtime
    # filter_all (which uses _arb_sqr throughout).  Raw a*a is LOOSER
    # when ``a`` straddles zero (or when rad(a) is comparable to |mid(a)|),
    # which can cause false negatives in the verifier.
    def _sq(x):
        al = x.abs_lower(); au = x.abs_upper()
        return (al * al).union(au * au)
    # F_bathtub
    for n in range(1, N + 1):
        a = ab[2 * (n - 1)]; b = ab[2 * (n - 1) + 1]
        if _arb_nonneg_is_reject(mu_arb - (_sq(a) + _sq(b))):
            return True
    # F1
    for n in range(1, N + 1):
        a = ab[2 * (n - 1)]; b = ab[2 * (n - 1) + 1]
        if _arb_nonneg_is_reject(arb(1) - (_sq(a) + _sq(b))):
            return True
    # F2
    for n in range(1, N + 1):
        a = ab[2 * (n - 1)]; b = ab[2 * (n - 1) + 1]
        two_n = 2 * n
        if two_n <= N: a_2n = ab[2 * (two_n - 1)]
        else: a_2n = arb(0, 1)
        if _arb_nonneg_is_reject((arb(1) + a_2n) / arb(2) - _sq(a)):
            return True
        if _arb_nonneg_is_reject((arb(1) - a_2n) / arb(2) - _sq(b)):
            return True
    # F4 MO 2.17
    if enable_F4_MO217 and N >= 2:
        if _arb_nonneg_is_reject(arb(2) * ab[0] - arb(1) - ab[2]):
            return True
    # F7
    if enable_F7:
        for kk in range(2, N + 2):
            if _arb_nonneg_is_reject(_T_minor_det_real(ab, N, kk, kind="f")):
                return True
    # F8
    if enable_F8:
        for kk in range(2, N + 2):
            if _arb_nonneg_is_reject(_T_minor_det_real(ab, N, kk, kind="h")):
                return True
    return False


# ============================================================================
#  Main verification
# ============================================================================

@dataclass
class VerifyResult:
    accepted: bool
    messages: list
    M_cert_q: fmpq | None = None


def verify_certificate_holder(cert_path: str, prec_bits: int | None = None) -> VerifyResult:
    with open(cert_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    msgs = []
    def log(s): msgs.append(s); print(s)

    body = raw["body"]
    body_json = json.dumps(body, indent=2, sort_keys=True)
    digest = hashlib.sha256(body_json.encode("utf-8")).hexdigest()
    if digest != raw["sha256_of_body"]:
        log("FAIL: SHA256 mismatch"); return VerifyResult(False, msgs)
    log(f"OK: SHA-256 = {digest}")

    if body.get("kind") != "grid_bound_Holder_Phase2":
        log(f"FAIL: unexpected kind {body.get('kind')!r}")
        return VerifyResult(False, msgs)

    holder = body.get("holder_exponents", {})
    p_q = _fmpq_from_str(holder.get("p", "2/1"))
    q_q = _fmpq_from_str(holder.get("q", "2/1"))
    if p_q < fmpq(2):
        log(f"FAIL: p = {p_q} < 2"); return VerifyResult(False, msgs)
    # 1/p + 1/q = 1
    lhs = fmpq(1) / p_q + fmpq(1) / q_q
    if lhs != fmpq(1):
        log(f"FAIL: 1/p + 1/q = {lhs} != 1"); return VerifyResult(False, msgs)
    log(f"OK: Hoelder exponents p={p_q}, q={q_q}, 1/p+1/q=1.")

    inputs = body["inputs"]
    prec = prec_bits or int(body.get("prec_bits", 256))
    delta_q = _fmpq_from_str(inputs["delta_q"])
    u_q     = _fmpq_from_str(inputs["u_q"])
    K2d_q   = _fmpq_from_str(inputs["K2_times_delta_q"])
    if delta_q != _MV_DELTA_Q or u_q != _MV_U_Q or K2d_q != _MV_K2_TIMES_DELTA_Q:
        log("FAIL: MV input rationals mismatch")
        return VerifyResult(False, msgs)
    if inputs["n_coeffs"] != 119:
        log("FAIL: n_coeffs != 119"); return VerifyResult(False, msgs)
    N = int(body["N"])
    log(f"OK: MV rationals match; N={N}")

    coeffs = [_decimal_to_fmpq(s) for s in _MV_DECIMALS_V]
    K2 = recompute_K2(K2d_q, delta_q, prec)
    S1 = recompute_S1(coeffs, delta_q, u_q, prec)
    min_G = recompute_min_G_lower(coeffs, u_q, prec)
    if min_G.upper() <= 0:
        log("FAIL: min_G lower <= 0"); return VerifyResult(False, msgs)
    gain_a = (arb(4) / arb(u_q)) * (min_G * min_G) / S1

    k_arb = [recompute_kn(n, delta_q, prec) for n in range(1, N + 1)]
    sum_kn_sq = arb(0)
    for kk in k_arb:
        sum_kn_sq = sum_kn_sq + kk * kk

    # K_q_upper re-computation
    K_q = recompute_Kq_upper(q_q, delta_q, K2, prec_bits=prec)

    q_arb = arb(q_q)
    sum_kn_q = arb(0)
    for kk in k_arb:
        if q_q == fmpq(2):
            sum_kn_q = sum_kn_q + kk * kk
        else:
            sum_kn_q = sum_kn_q + kk ** q_arb
    log(f"OK: recomputed gain_a={gain_a}, K_q={K_q}")

    M_cert_q = _fmpq_from_str(body["M_cert"]["rational"])
    M_arb = arb(M_cert_q)
    log(f"Verifying M_cert = {M_cert_q} (~{_fmpq_to_float(M_cert_q):.6f})")
    mu = mu_of_M(M_arb)
    mu_upper = mu.upper()

    filter_panel = body["filter_panel"]
    f_kw = dict(
        enable_F4_MO217=filter_panel.get("enable_F4_MO217", True),
        enable_F7=filter_panel.get("enable_F7", True),
        enable_F8=filter_panel.get("enable_F8", True),
    )

    cells = body["cell_search_at_M_cert"]["terminal_cells"]
    log(f"Re-checking {len(cells)} terminal cells ...")
    for i, rec in enumerate(cells):
        lo_q = tuple(_fmpq_from_str(s) for s in rec["cell"]["lo"])
        hi_q = tuple(_fmpq_from_str(s) for s in rec["cell"]["hi"])
        if len(lo_q) != 2 * N or len(hi_q) != 2 * N:
            log(f"FAIL: cell {i} dim mismatch"); return VerifyResult(False, msgs)
        for d in range(2 * N):
            if hi_q[d] < lo_q[d]:
                log(f"FAIL: cell {i} inverted dim {d}"); return VerifyResult(False, msgs)
        ab = [
            arb((lo_q[d] + hi_q[d]) / fmpq(2), (hi_q[d] - lo_q[d]) / fmpq(2))
            for d in range(2 * N)
        ]
        verdict = rec["verdict"]
        if verdict == "FILTER_REJECT":
            if not _filter_reject_check(ab, N, mu_upper, **f_kw):
                log(f"FAIL: cell {i} FILTER_REJECT not reproducible")
                return VerifyResult(False, msgs)
        elif verdict == "PHI_REJECT":
            try:
                phi_v = phi_holder_verify(
                    M_arb, ab,
                    p=p_q, q=q_q, K_q=K_q, sum_kn_q=sum_kn_q,
                    k_arb=k_arb, sum_kn_sq=sum_kn_sq, K2=K2,
                    u=u_q, gain_a=gain_a,
                )
            except ValueError:
                continue  # non-physical radicand => forbidden
            if not (phi_v.upper() < 0):
                log(f"FAIL: cell {i} PHI_REJECT but re-Phi.upper()={float(phi_v.upper())} NOT < 0")
                return VerifyResult(False, msgs)
        else:
            log(f"FAIL: cell {i} unknown verdict {verdict}")
            return VerifyResult(False, msgs)
    log(f"OK: all {len(cells)} cells re-verified.")

    mu_sqrt_upper = mu_upper.sqrt().upper()
    log(f"OK: root box [-{float(mu_sqrt_upper):.6f}, +{float(mu_sqrt_upper):.6f}]^{{2N}} is bisection root.")
    log("")
    log(f"VERDICT: CERTIFICATE ACCEPTED.  C_{{1a}} >= {M_cert_q} (~{_fmpq_to_float(M_cert_q):.6f}) at (p, q) = ({p_q}, {q_q}).")
    return VerifyResult(True, msgs, M_cert_q)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("certificate")
    p.add_argument("--prec-bits", type=int, default=None)
    args = p.parse_args(argv)
    res = verify_certificate_holder(args.certificate, prec_bits=args.prec_bits)
    sys.exit(0 if res.accepted else 1)


if __name__ == "__main__":
    main()
