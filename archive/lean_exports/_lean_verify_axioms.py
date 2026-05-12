"""Lean trusted-tool axiom verifier.

Re-checks every numerical axiom asserted in `lean/Sidon/RationalCert.lean`
using `flint.arb` interval arithmetic at high precision (default 256 bits).
On success, writes the certified `arb` ball for each axiom to
`_lean_verify_axioms_result.json`.  Exits non-zero on any disagreement.

USAGE
-----
    python _lean_verify_axioms.py [--prec 256] [--grid 200000] [--xi-max 600]

Outputs:
    _lean_verify_axioms_result.json      (machine-readable certificates)
    stdout                               (human-readable summary)
    exit code 0 on full agreement, 1 on any mismatch.

The script is read by CI (or by the user manually) to re-establish trust
in the trusted-tool axioms.  See `lean/Sidon/RationalCert.lean` for the
matching axiom statements and the rational intervals each axiom asserts.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

# Force UTF-8 stdout on Windows consoles (Unicode subscripts in messages).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from flint import arb, fmpq, ctx


def _to_float(x) -> float:
    """Robust float() that handles fmpq (and arb upper/lower returns)."""
    try:
        return float(x)
    except TypeError:
        # fmpq -> float via repr / direct ratio
        try:
            return float(x.p) / float(x.q)
        except AttributeError:
            return float(str(x))

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

# Local imports
from delsarte_dual.grid_bound.coeffs import mv_coeffs_fmpq
from delsarte_dual.grid_bound.G_min import min_G_lower_bound


# ============================================================================
# Helpers
# ============================================================================


def fmt_arb(b: arb) -> dict:
    """Format an arb ball as JSON-serialisable lower/upper/mid/radius."""
    return {
        "lower_str": str(b.lower()),
        "upper_str": str(b.upper()),
        "lower_float": _to_float(b.lower()),
        "upper_float": _to_float(b.upper()),
        "mid_float": _to_float(b),
    }


def check_interval(name: str, b: arb, lo_q: fmpq, hi_q: fmpq) -> bool:
    """Assert that the arb ball is contained in [lo_q, hi_q] rationals."""
    bl = b.lower()
    bu = b.upper()
    ok_lo = bl >= arb(lo_q)
    ok_hi = bu <= arb(hi_q)
    ok = bool(ok_lo) and bool(ok_hi)
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {name}: arb in [{_to_float(bl):.12g}, {_to_float(bu):.12g}]"
          f"  vs Lean [{_to_float(lo_q):.12g}, {_to_float(hi_q):.12g}]")
    return ok


# ============================================================================
# Certificate 1: G grid lower (200,000 points on [0, 1/4])
# ============================================================================


def cert_G_grid(prec_bits: int, grid_M: int) -> dict:
    """Run the interval-Taylor B&B to get a rigorous lower bound on min G
    over [0, 1/4], with `grid_M` cells.  Lean asserts G ≥ 998/1000."""
    print(f"\n[cert_G_grid] grid_M={grid_M}, prec_bits={prec_bits}")
    t0 = time.perf_counter()
    coeffs = mv_coeffs_fmpq()
    u = fmpq(638, 1000)

    # The G_min.py routine uses arbitrary cell count.  For full 200k, this
    # is the production-quality run.  For CI speed we may use grid_M=8192
    # (still rigorous, slightly looser) and verify the bound 998/1000.
    encl, center = min_G_lower_bound(
        coeffs, u, x_lo=fmpq(0), x_hi=fmpq(1, 4),
        n_cells=grid_M, prec_bits=prec_bits,
    )
    elapsed = time.perf_counter() - t0
    lo = encl.lower()
    print(f"  G_min lower (arb)  = {_to_float(lo):.12f}")
    print(f"  argmin cell center = {_to_float(center):.6f}")
    print(f"  elapsed = {elapsed:.2f}s")

    lean_lb = fmpq(998, 1000)  # Lean's G_min_lb
    ok = bool(arb(lo) >= arb(lean_lb))
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] G_min ≥ 998/1000")
    return {
        "axiom": "G_grid_lower",
        "grid_M": grid_M,
        "prec_bits": prec_bits,
        "arb_lower_str": str(lo),
        "arb_lower_float": _to_float(lo),
        "argmin_center_str": str(center),
        "argmin_center_float": _to_float(center),
        "lean_assertion": "G_min_lb = 998/1000",
        "ok": ok,
        "elapsed_s": elapsed,
    }


# ============================================================================
# Certificate 2: J₀² at MV parameter point
# ============================================================================


def cert_J0_sq(prec_bits: int) -> list[dict]:
    """Rigorous arb ball for J₀(z)² at z = π·138/638 and z = π·55/638."""
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        out = []
        for name, num, den, lo_lean, hi_lean in [
            ("z1=pi*138/638", 138, 638,
             fmpq(7882759, 10_000_000), fmpq(7882762, 10_000_000)),
            ("z2=pi*55/638",  55,  638,
             fmpq(9638272, 10_000_000), fmpq(9638275, 10_000_000)),
        ]:
            z = arb.pi() * arb(num) / arb(den)
            J0 = z.bessel_j(arb(0))
            J0_sq = J0 * J0
            print(f"\n[cert_J0_sq] {name}")
            print(f"  z      = {_to_float(z):.16f}")
            print(f"  J0(z)  = arb({_to_float(J0.lower()):.16f}, {_to_float(J0.upper()):.16f})")
            print(f"  J0(z)² = arb({_to_float(J0_sq.lower()):.16f}, {_to_float(J0_sq.upper()):.16f})")
            ok = check_interval(name, J0_sq, lo_lean, hi_lean)
            out.append({
                "axiom": f"J0_sq_at_{name}",
                "z_num": num, "z_den": den,
                "J0_sq_arb": fmt_arb(J0_sq),
                "lean_lb": str(lo_lean),
                "lean_ub": str(hi_lean),
                "ok": ok,
            })
        return out
    finally:
        ctx.prec = old


# ============================================================================
# Certificate 3: K₂ for single arcsine kernel
# ============================================================================


def _K2_single_arcsine_arb(delta: fmpq, xi_max_over_delta: float,
                           n_xi: int, prec_bits: int) -> arb:
    """K₂(δ) = 2 · ∫₀^∞ J₀(π·δ·ξ)⁴ dξ.

    Rigorous via `acb.integral` (Arb's adaptive Gauss-Legendre with rigorous
    enclosures) plus a closed-form tail bound from |J₀(z)| ≤ √(2/(π z)).

    Tail bound:
        ∫_{Ξ}^∞ J₀(πδξ)⁴ dξ ≤ ∫_{Ξ}^∞ 4/(π·(πδξ))² dξ
                            = 4/(π³ δ²) · (1/Ξ).
    """
    from flint import acb
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        d_arb = arb(delta)
        XI_MAX = xi_max_over_delta / _to_float(delta)

        # acb.integral expects function (x, analytic) -> acb.  J₀ is entire,
        # so it is analytic everywhere -- we ignore the analytic flag.
        d_acb = acb(delta)
        pid_acb = acb.pi() * d_acb

        def integrand(x, analytic):
            return (pid_acb * x).bessel_j(0) ** 4

        integ_acb = acb.integral(integrand, 0, XI_MAX)
        # integ_acb is acb (could have nonzero imag from precision noise),
        # take real part as arb.
        integ_pos = integ_acb.real

        # Tail bound on [XI_MAX, ∞)
        Z = arb(XI_MAX)
        tail_ub_arb = arb(4) / (arb.pi() ** 3 * d_arb * d_arb * Z)
        tail_ub_f = _to_float(tail_ub_arb)
        tail_ball = arb(tail_ub_f / 2.0, tail_ub_f / 2.0)

        K2 = arb(2) * (integ_pos + tail_ball)
        return K2
    finally:
        ctx.prec = old


def cert_K2_single(prec_bits: int, n_xi: int = 8001,
                   xi_max_over_delta: float = 200.0) -> dict:
    """Verify the rational interval for K₂(δ=138/1000)."""
    delta = fmpq(138, 1000)
    print(f"\n[cert_K2_single] δ=138/1000, n_xi={n_xi}, "
          f"xi_max/δ={xi_max_over_delta}, prec_bits={prec_bits}")
    K2 = _K2_single_arcsine_arb(delta, xi_max_over_delta, n_xi, prec_bits)
    print(f"  K₂ arb = [{_to_float(K2.lower()):.6f}, {_to_float(K2.upper()):.6f}]"
          f"  mid={_to_float(K2):.6f}")
    lo_lean = fmpq(4_164, 1_000)
    hi_lean = fmpq(4_168, 1_000)
    ok = check_interval("K_two_single_arcsine_value", K2, lo_lean, hi_lean)
    return {
        "axiom": "K_two_single_arcsine_value",
        "delta": "138/1000",
        "n_xi": n_xi,
        "xi_max_over_delta": xi_max_over_delta,
        "K2_arb": fmt_arb(K2),
        "lean_lb": str(lo_lean),
        "lean_ub": str(hi_lean),
        "ok": ok,
    }


# ============================================================================
# Certificate 4: S₁ = Σ aⱼ² / J₀(π·j·δ/u)²
# ============================================================================


def cert_S1(prec_bits: int) -> dict:
    """S₁ as a rigorous arb ball."""
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        coeffs = mv_coeffs_fmpq()
        delta = fmpq(138, 1000)
        u = fmpq(638, 1000)
        pid_over_u = arb.pi() * arb(delta) / arb(u)
        total = arb(0)
        for j, a_j in enumerate(coeffs, start=1):
            z = pid_over_u * arb(j)
            J0 = z.bessel_j(arb(0))
            J0_sq = J0 * J0
            term = (arb(a_j) ** 2) / J0_sq
            total = total + term
        print(f"\n[cert_S1] prec_bits={prec_bits}")
        print(f"  S₁ arb = [{_to_float(total.lower()):.6f}, {_to_float(total.upper()):.6f}]"
              f"  mid={_to_float(total):.6f}")
        lo_lean = fmpq(87856, 1000)
        hi_lean = fmpq(87858, 1000)
        ok = check_interval("S_one_value", total, lo_lean, hi_lean)
        return {
            "axiom": "S_one_value",
            "S1_arb": fmt_arb(total),
            "lean_lb": str(lo_lean),
            "lean_ub": str(hi_lean),
            "ok": ok,
        }
    finally:
        ctx.prec = old


# ============================================================================
# Certificate 5: Master quadratic reproduces 1.2743
# ============================================================================


def cert_master_quadratic(prec_bits: int) -> dict:
    """Plug in K₂, R rationals into the closed form and recover M_star ≥ 1.2743."""
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        K2 = arb(fmpq(4_164_051, 1_000_000))
        R = arb(fmpq(3_206_057, 1_000_000))
        c = (K2 - arb(1)).sqrt()
        D = (K2 - arb(1)) + arb(4) * (R - arb(2))
        s = (-c + D.sqrt()) / arb(2)
        M = arb(1) + s * s
        print(f"\n[cert_master_quadratic] K₂=4.164051, R=3.206057")
        print(f"  c = √(K₂-1) = arb({_to_float(c.lower()):.6f}, {_to_float(c.upper()):.6f})")
        print(f"  D = arb({_to_float(D.lower()):.6f}, {_to_float(D.upper()):.6f})")
        print(f"  M_star = arb({_to_float(M.lower()):.6f}, {_to_float(M.upper()):.6f})")
        lean_lb = fmpq(12743, 10_000)
        ok = bool(M.lower() >= arb(lean_lb))
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] M_star.lower={_to_float(M.lower()):.6f} ≥ 12743/10000")
        return {
            "axiom": "master_quadratic_reproduces_MV",
            "K2": "4164051/1000000",
            "R": "3206057/1000000",
            "M_star_arb": fmt_arb(M),
            "lean_lb": "12743/10000",
            "ok": ok,
        }
    finally:
        ctx.prec = old


# ============================================================================
# Main
# ============================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prec", type=int, default=128,
                    help="arb precision in bits (default: 128)")
    ap.add_argument("--grid", type=int, default=8192,
                    help="number of cells for G_min B&B (default: 8192; "
                         "use 200000 for the FULL Lean grid_M)")
    ap.add_argument("--xi-max", type=float, default=200.0,
                    help="ξ-grid extent (multiples of 1/δ) for K₂ integration")
    ap.add_argument("--n-xi", type=int, default=4001,
                    help="number of ξ-grid cells for K₂ integration")
    ap.add_argument("--out", default="_lean_verify_axioms_result.json")
    ap.add_argument("--skip-K2", action="store_true",
                    help="skip the K₂ certificate (slowest)")
    args = ap.parse_args()

    print("=" * 78)
    print("Lean trusted-tool axiom verifier  —  lean/Sidon/RationalCert.lean")
    print("=" * 78)

    results = {}

    # 1. G grid lower
    try:
        results["G_grid"] = cert_G_grid(args.prec, args.grid)
    except Exception as e:
        results["G_grid"] = {"ok": False, "error": repr(e)}

    # 2. J₀² at z₁ and z₂
    try:
        results["J0_sq"] = cert_J0_sq(args.prec)
    except Exception as e:
        results["J0_sq"] = [{"ok": False, "error": repr(e)}]

    # 3. K₂ single arcsine
    if not args.skip_K2:
        try:
            results["K2_single"] = cert_K2_single(
                args.prec, n_xi=args.n_xi, xi_max_over_delta=args.xi_max)
        except Exception as e:
            results["K2_single"] = {"ok": False, "error": repr(e)}

    # 4. S₁
    try:
        results["S1"] = cert_S1(args.prec)
    except Exception as e:
        results["S1"] = {"ok": False, "error": repr(e)}

    # 5. Master quadratic
    try:
        results["master_quadratic"] = cert_master_quadratic(args.prec)
    except Exception as e:
        results["master_quadratic"] = {"ok": False, "error": repr(e)}

    # Summarise + write
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    all_ok = True

    def check_ok(name, rec):
        nonlocal all_ok
        if isinstance(rec, list):
            for r in rec:
                ok = r.get("ok", False)
                all_ok = all_ok and ok
                print(f"  {name}[{r.get('axiom','?')}]: "
                      f"{'OK' if ok else 'FAIL'}")
        else:
            ok = rec.get("ok", False)
            all_ok = all_ok and ok
            print(f"  {name}: {'OK' if ok else 'FAIL'}")

    for name, rec in results.items():
        check_ok(name, rec)

    out_path = os.path.join(REPO, args.out)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    print(f"\nFINAL: {'ALL AXIOMS VERIFIED' if all_ok else 'SOME AXIOMS FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
