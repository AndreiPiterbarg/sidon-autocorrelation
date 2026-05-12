"""Production driver: certify the Piterbarg-Bajaj-Vincent Bound
``C_{1a} >= 1292/1000 = 1.292``.

Composes the package modules into a single end-to-end pipeline:

  1. Build the 3-scale arcsine kernel
     ``K = sum_{i=1}^{3} lambda_i K_arc(delta_i; .)``
     with the writeup parameters

         ``(delta_1, delta_2, delta_3) = (138/1000, 55/1000, 25/1000)``
         ``(lambda_1, lambda_2, lambda_3) = (85/100, 10/100, 5/100)``.

  2. Solve the quadratic programme

         ``min  sum_{j=1}^{N} a_j^2 / hat K(j/u)``
         ``s.t. sum_{j=1}^{N} a_j cos(2 pi j x / u) >= 1  for x in [0, 1/4]``

     for the re-optimised ``N = 200`` cosine multiplier
     ``G(x) = sum_j a_j cos(2 pi j x / u)``; round to ``fmpq``.

  3. Compile the five rigorous arb anchors
     ``(k_1, K_2, S_1, min_G, gain_a)`` from the kernel and the rounded
     coefficients (``compile_phi_params_for_kernel``).

  4. Bisect on ``M`` in ``[M_lo_init, M_hi_init]`` using the cell-search
     certifier ``CERTIFIED_FORBIDDEN`` predicate from
     ``grid_bound.cell_search`` to find the largest ``M`` for which
     ``Phi(M, y) < 0`` over the admissible box ``y in [0, mu(M)]``.

  5. Emit a JSON certificate with the rational ``M_cert``, all input
     parameters, the QP coefficients, the per-anchor arb enclosures, and
     the terminal cells of the certifying cell search.

Run with

    ``python -m delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel``

The default settings reproduce the writeup's ``M_cert >= 1292/1000``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional

from flint import arb, fmpq, ctx

from delsarte_dual.grid_bound.bisect import (
    bisect_M_cert,
    fmpq_to_float,
    fmpq_to_str,
)
from delsarte_dual.grid_bound.cell_search import certify_phi_negative
from delsarte_dual.grid_bound.G_min import min_G_lower_bound
from delsarte_dual.grid_bound.phi import PhiParams

from .kernels import Kernel, MultiScaleArcsineKernel
from .optimize_G import solve_qp_for_kernel


# -----------------------------------------------------------------------------
#  Writeup parameters (3-scale arcsine, N = 200)
# -----------------------------------------------------------------------------

#: Three arcsine half-widths (writeup Section 1.3).
PROD_DELTAS: tuple = (fmpq(138, 1000), fmpq(55, 1000), fmpq(25, 1000))

#: Mixture weights, summing to one.
PROD_LAMBDAS: tuple = (fmpq(85, 100), fmpq(10, 100), fmpq(5, 100))

#: Period parameter ``u = 1/2 + delta_1 = 638/1000``.
PROD_U: fmpq = fmpq(638, 1000)

#: Number of cosine modes in ``G``.
PROD_N_COEFFS: int = 200

#: Cutoff in the cross-Bessel ``K_2`` integral; the tail past ``T`` is
#: bounded by ``4 / (pi^2 delta_i delta_j T)`` per cross pair.
PROD_K2_CUTOFF_XI: fmpq = fmpq(10**5)


def production_kernel() -> MultiScaleArcsineKernel:
    """Return the 3-scale arcsine kernel of the writeup."""
    return MultiScaleArcsineKernel(
        deltas=list(PROD_DELTAS),
        lambdas=list(PROD_LAMBDAS),
        K2_cross_cutoff_xi=PROD_K2_CUTOFF_XI,
        use_diag_surrogate=False,
    )


# -----------------------------------------------------------------------------
#  Compile Phi parameters from a kernel + a list of fmpq G-coefficients
# -----------------------------------------------------------------------------


def compile_phi_params_for_kernel(
    kernel: Kernel,
    coeffs: List[fmpq],
    u: fmpq = PROD_U,
    n_cells_min_G: int = 32768,
    prec_bits: int = 256,
) -> PhiParams:
    """Build a ``PhiParams`` object using kernel-specific anchors.

    The G-coefficients are the rounded ``a_j`` values of
    ``G(x) = sum_{j=1}^{N} a_j cos(2 pi j x / u)``.  All five anchors are
    computed in arb at ``prec_bits``:

      * ``k_1   = hat K(1)``                          (``kernel.K_tilde(1)``),
      * ``K_2   = ||K||_2^2``                         (``kernel.K_norm_sq``),
      * ``S_1   = sum a_j^2 / hat K(j/u)``,
      * ``min_G = rigorous lower bound on min_{[0, 1/4]} G``  (Taylor B&B),
      * ``gain_a = (4/u) * min_G^2 / S_1``.
    """
    delta = kernel.supp_halfwidth
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        K2_arb = kernel.K_norm_sq(prec_bits=prec_bits)
        k1_arb = kernel.K_tilde(1, prec_bits=prec_bits)

        S1 = arb(0)
        for j, a_j in enumerate(coeffs, start=1):
            xi = arb(fmpq(j)) / arb(u)
            w_j = kernel.K_tilde_real(xi, prec_bits=prec_bits)
            if w_j.lower() <= 0:
                a_j_f = float(a_j.p) / float(a_j.q)
                if a_j_f != 0.0:
                    raise ValueError(
                        f"{kernel.name}: Bochner violated at j={j} "
                        f"(hat K(j/u) = {w_j}) but coefficient a_{j} != 0"
                    )
                continue
            S1 = S1 + (arb(a_j) * arb(a_j)) / w_j

        min_G_encl, min_G_center = min_G_lower_bound(
            coeffs, u, n_cells=n_cells_min_G, prec_bits=prec_bits
        )
        min_G_lower = min_G_encl.lower()
        if min_G_lower.upper() <= 0:
            raise ValueError(
                f"{kernel.name}: certified min G lower bound is "
                f"non-positive ({min_G_lower})"
            )
        gain_a = (arb(4) / arb(u)) * (min_G_lower * min_G_lower) / S1

        return PhiParams(
            delta=delta,
            u=u,
            K2=K2_arb,
            k1=k1_arb,
            gain_a=gain_a,
            min_G=min_G_lower,
            S1=S1,
            n_coeffs=len(coeffs),
            min_G_center=min_G_center,
        )
    finally:
        ctx.prec = old


# -----------------------------------------------------------------------------
#  Run a single kernel end-to-end (admissibility -> QP -> Phi -> bisection)
# -----------------------------------------------------------------------------


@dataclass
class CertificationResult:
    """End-to-end result for one kernel."""

    kernel_name: str
    admissible: bool
    K_norm_sq_float: float
    k_1_float: float
    S_1_float: Optional[float]
    min_G_float: Optional[float]
    gain_a_float: Optional[float]
    M_cert_float: Optional[float]
    M_cert_q: Optional[str]
    bisect_history: Optional[list]
    terminal_cells: Optional[list]
    note: str
    wall_time_sec: float
    params: Optional[PhiParams] = None
    coeffs: Optional[List[fmpq]] = None


def run_single_kernel(
    kernel: Kernel,
    *,
    u: fmpq = PROD_U,
    n_coeffs: int = PROD_N_COEFFS,
    n_grid_qp: int = 5001,
    n_cells_min_G: int = 32768,
    M_lo_init: fmpq = fmpq(127, 100),
    M_hi_init: fmpq = fmpq(130, 100),
    tol_q: fmpq = fmpq(1, 10**4),
    max_cells_per_M: int = 50000,
    initial_splits: int = 32,
    bochner_max: int = 100,
    prec_bits: int = 256,
    verbose: bool = True,
) -> CertificationResult:
    """Run admissibility check, QP, Phi compilation and bisection for one kernel.

    On any failure (Bochner violation, QP non-convergence, infeasible lower
    bracket, ...) the returned ``CertificationResult`` records the reason
    in ``note`` and leaves the optional fields as ``None``.
    """
    t0 = time.time()

    # (1) Bochner admissibility on the first ``bochner_max`` frequencies.
    bochner_ok = True
    note_parts: list = []
    for j in range(1, bochner_max + 1):
        try:
            v = kernel.K_tilde(j, prec_bits=prec_bits)
        except Exception as exc:  # pragma: no cover - defensive
            bochner_ok = False
            note_parts.append(f"K_tilde({j}) raised: {exc}")
            break
        if v.lower() < 0:
            bochner_ok = False
            note_parts.append(f"Bochner fails at j={j}: tilde K(j) = {v}")
            break

    k1_float = float(kernel.K_tilde(1, prec_bits=prec_bits).mid())
    try:
        K2_float = float(kernel.K_norm_sq(prec_bits=prec_bits).mid())
    except Exception as exc:  # pragma: no cover - defensive
        K2_float = float("nan")
        note_parts.append(f"K_norm_sq raised: {exc}")

    def _fail(extra: str = "") -> CertificationResult:
        msg = " | ".join([*note_parts, extra]) if extra else " | ".join(note_parts)
        return CertificationResult(
            kernel_name=kernel.name,
            admissible=bochner_ok,
            K_norm_sq_float=K2_float,
            k_1_float=k1_float,
            S_1_float=None,
            min_G_float=None,
            gain_a_float=None,
            M_cert_float=None,
            M_cert_q=None,
            bisect_history=None,
            terminal_cells=None,
            note=msg or "failure",
            wall_time_sec=time.time() - t0,
        )

    if not bochner_ok:
        return _fail("Bochner violation")

    # (2) QP re-optimisation of G against the kernel-specific weights.
    try:
        qp = solve_qp_for_kernel(
            kernel, n=n_coeffs, u=u, n_grid=n_grid_qp, verbose=verbose
        )
    except Exception as exc:
        return _fail(f"QP solve failed: {exc}")

    # (3) Compile Phi parameters.
    try:
        params = compile_phi_params_for_kernel(
            kernel,
            qp.a_opt_fmpq,
            u=u,
            n_cells_min_G=n_cells_min_G,
            prec_bits=prec_bits,
        )
    except Exception as exc:
        return _fail(f"PhiParams compile failed: {exc}")

    S1_float = float(params.S1.mid())
    min_G_float = float(params.min_G.mid())
    gain_float = float(params.gain_a.mid())
    if verbose:
        print(
            f"  {kernel.name}: "
            f"k_1={k1_float:.6f}  K_2={K2_float:.4f}  "
            f"S_1={S1_float:.4f}  min G={min_G_float:.6f}  "
            f"gain a={gain_float:.6f}"
        )

    # (4) Bisect on M.  First confirm the lower bracket is certifiable;
    # retreat to a smaller value if it is not.
    try:
        check_lo = certify_phi_negative(
            arb(M_lo_init),
            params,
            max_cells=max_cells_per_M,
            initial_splits=initial_splits,
            prec_bits=prec_bits,
        )
        if check_lo.verdict != "CERTIFIED_FORBIDDEN":
            # Coarse retreat ladder for weaker kernels; the production
            # kernel certifies at the default M_lo_init = 1.27 immediately.
            for M_retry in (
                fmpq(125, 100),
                fmpq(120, 100),
                fmpq(115, 100),
                fmpq(110, 100),
            ):
                check2 = certify_phi_negative(
                    arb(M_retry),
                    params,
                    max_cells=max_cells_per_M,
                    initial_splits=initial_splits,
                    prec_bits=prec_bits,
                )
                if check2.verdict == "CERTIFIED_FORBIDDEN":
                    M_lo_init = M_retry
                    break
            else:
                return _fail("no certifiable lower bracket in [1.10, 1.27]")
    except Exception as exc:
        return _fail(f"lower bracket check raised: {exc}")

    try:
        bound = bisect_M_cert(
            params,
            M_lo_init=M_lo_init,
            M_hi_init=M_hi_init,
            tol_q=tol_q,
            max_cells_per_M=max_cells_per_M,
            initial_splits=initial_splits,
            prec_bits=prec_bits,
            verbose=False,
        )
    except Exception as exc:
        return _fail(f"bisect raised: {exc}")

    terminal_cells = [
        {
            "cell": cr.cell.to_dict(),
            "phi_upper_float": cr.phi_upper_float,
            "phi_arb": cr.phi_arb_str,
        }
        for cr in bound.cell_search.terminal_cells
    ]

    return CertificationResult(
        kernel_name=kernel.name,
        admissible=True,
        K_norm_sq_float=K2_float,
        k_1_float=k1_float,
        S_1_float=S1_float,
        min_G_float=min_G_float,
        gain_a_float=gain_float,
        M_cert_float=fmpq_to_float(bound.M_cert_q),
        M_cert_q=fmpq_to_str(bound.M_cert_q),
        bisect_history=bound.bisection_history,
        terminal_cells=terminal_cells,
        note="ok",
        wall_time_sec=time.time() - t0,
        params=params,
        coeffs=qp.a_opt_fmpq,
    )


# -----------------------------------------------------------------------------
#  JSON certificate
# -----------------------------------------------------------------------------


def _arb_endpoints(a: arb) -> dict:
    return {
        "repr": str(a),
        "mid_float": float(a.mid()),
        "lower_float": float(a.lower()),
        "upper_float": float(a.upper()),
    }


def emit_certificate(
    result: CertificationResult, kernel: MultiScaleArcsineKernel, filepath: str
) -> str:
    """Write a JSON certificate for ``result``; return its SHA-256 hex."""
    if result.M_cert_q is None or result.params is None or result.coeffs is None:
        raise ValueError("cannot emit certificate for a failed run")

    p = result.params
    body = {
        "format_version": 2,
        "kind": "multiscale_arcsine_rigorous_certificate",
        "description": (
            "Rigorous lower bound on the Sidon autocorrelation constant "
            "C_{1a} via the Matolcsi-Vinuesa master inequality with a "
            "multi-scale arcsine kernel and a QP-reoptimised cosine "
            "multiplier G."
        ),
        "kernel": {
            "name": kernel.name,
            "deltas_q": [fmpq_to_str(d) for d in kernel.deltas],
            "lambdas_q": [fmpq_to_str(l) for l in kernel.lambdas],
            "supp_halfwidth_q": fmpq_to_str(kernel.supp_halfwidth),
            "K2_cross_cutoff_xi_q": fmpq_to_str(kernel.K2_cross_cutoff_xi),
            "use_diag_surrogate": kernel.use_diag_surrogate,
        },
        "G": {
            "n_coeffs": p.n_coeffs,
            "u_q": fmpq_to_str(p.u),
            "coeffs_q": [fmpq_to_str(c) for c in result.coeffs],
        },
        "anchors": {
            "k_1": _arb_endpoints(p.k1),
            "K_2": _arb_endpoints(p.K2),
            "S_1": _arb_endpoints(p.S1),
            "min_G": _arb_endpoints(p.min_G),
            "gain_a": _arb_endpoints(p.gain_a),
            "min_G_cell_center_q": fmpq_to_str(p.min_G_center),
        },
        "M_cert": {
            "rational": result.M_cert_q,
            "float": result.M_cert_float,
        },
        "cell_search_at_M_cert": {
            "n_terminal_cells": len(result.terminal_cells),
            "terminal_cells": result.terminal_cells,
        },
        "bisect_history": result.bisect_history,
        "prec_bits": 256,
        "wall_time_sec": result.wall_time_sec,
    }
    body_json = json.dumps(body, indent=2, sort_keys=True, default=str)
    digest = hashlib.sha256(body_json.encode("utf-8")).hexdigest()
    final = {"sha256_of_body": digest, "body": body}
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, sort_keys=True, default=str)
    return digest


# -----------------------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> CertificationResult:
    parser = argparse.ArgumentParser(
        description=(
            "Certify the Piterbarg-Bajaj-Vincent Bound "
            "C_{1a} >= 1.292 via a 3-scale arcsine kernel and a "
            "re-optimised 200-cosine multiplier."
        )
    )
    parser.add_argument(
        "--n-coeffs",
        type=int,
        default=PROD_N_COEFFS,
        help="number of cosine modes in G",
    )
    parser.add_argument(
        "--prec-bits", type=int, default=256, help="arb precision in bits"
    )
    parser.add_argument(
        "--tol",
        type=str,
        default="1/10000",
        help="bisection tolerance on M as p/q",
    )
    parser.add_argument(
        "--M-lo",
        type=str,
        default="127/100",
        help="initial lower bracket on M",
    )
    parser.add_argument(
        "--M-hi",
        type=str,
        default="130/100",
        help="initial upper bracket on M",
    )
    parser.add_argument(
        "--max-cells-per-M",
        type=int,
        default=50000,
        help="cell budget per Phi-certification call",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="delsarte_dual/grid_bound_alt_kernel/certificates/"
        "multiscale_arcsine_1292.json",
        help="output certificate path",
    )
    args = parser.parse_args(argv)

    def parse_q(s: str) -> fmpq:
        if "/" in s:
            p, q = s.split("/", 1)
            return fmpq(int(p), int(q))
        return fmpq(int(s))

    kernel = production_kernel()
    result = run_single_kernel(
        kernel,
        n_coeffs=args.n_coeffs,
        prec_bits=args.prec_bits,
        tol_q=parse_q(args.tol),
        M_lo_init=parse_q(args.M_lo),
        M_hi_init=parse_q(args.M_hi),
        max_cells_per_M=args.max_cells_per_M,
        verbose=True,
    )

    print()
    print("=" * 72)
    if result.M_cert_q is not None:
        print(
            f"Certified M_cert = {result.M_cert_q} "
            f"(~{result.M_cert_float:.6f})"
        )
        print(f"  wall time: {result.wall_time_sec:.1f}s")
        digest = emit_certificate(result, kernel, args.out)
        print(f"Certificate: {args.out}")
        print(f"SHA-256:     {digest}")
    else:
        print(f"FAILED: {result.note}")
    print("=" * 72)
    return result


if __name__ == "__main__":
    main()
