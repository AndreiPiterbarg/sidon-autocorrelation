"""Normalize v3 reoptimized G so that min_G >= 1.0 rigorously.

Background
----------
The v3 reoptimized G (saved in ``_cohn_elkies_128_greopt_coeffs.json``) has

    min_{x in [0, 1/4]} G(x) = 0.998887...   (rigorous LB, 200001-pt grid).

This is BELOW 1.  Depending on the reading of MV's master inequality the
``(min G)^2`` factor either (a) needs only positivity, or (b) is wired into a
normalization convention that wants min G >= 1.  Either way the **gain
parameter**

    a = (4/u) * (min G)^2 / S_1,    S_1 = sum_j a_j^2 / |J_0(pi j delta/u)|^2

is **invariant under scaling**:  if we replace G(x) by c*G(x) with c>0, then
both (min G)^2 and S_1 scale by c^2 and ``a`` is unchanged.  So we can freely
rescale G so that the rigorous lower bound on min G is >= 1, without changing
the certified ``M_*``.

Strategy
--------
1.  Load the v3 G coefficients (rational, denominator 5e11 or 1e12).
2.  Compute a rigorous LOWER bound ``m_lo`` on min G via the existing
    ``min_G_lower_bound`` of ``_cohn_elkies_128_v3``.
3.  Pick a rational ``c_q`` with ``c_q <= m_lo`` (use the arb lower
    endpoint truncated downward to a rational at denominator 1e15).
4.  Define a_j_new = a_j_old / c_q  (exact rational division).
5.  Renormalize the resulting rationals to a common denominator
    (1e15) so the JSON stays neat.
6.  Verify rigorously that the new ``min_G_new >= 1.0`` and that the
    certified ``M_*`` is unchanged.
7.  Save the result to ``_cohn_elkies_128_v3_normalized.py`` /
    ``_cohn_elkies_128_v3_normalized.json``.

Usage
-----
    python _v3_g_normalize.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

import flint
from flint import arb

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "delsarte_dual"))

# Reuse the v3 routines.
import _cohn_elkies_128_v3 as v3


SRC_JSON = _HERE / "_cohn_elkies_128_greopt_coeffs.json"
OUT_JSON = _HERE / "_cohn_elkies_128_v3_normalized.json"
OUT_PY = _HERE / "_cohn_elkies_128_v3_normalized.py"

NEW_DEN = 10 ** 15          # common denominator after rescaling


def safe_rational_lower(arb_val: arb, denom: int = NEW_DEN) -> Fraction:
    """Return a rational ``c_q`` with ``c_q <= arb_val`` (rigorously).

    We use ``arb_val.lower()`` (a Python float strictly below the true value)
    and floor it to a multiple of 1/denom.  Then we also subtract one ulp
    (1/denom) for paranoia.
    """
    lo_f = float(arb_val.lower())
    # floor lo_f to multiple of 1/denom
    num = math.floor(lo_f * denom) - 1   # -1 ulp paranoia
    return Fraction(num, denom)


def rescale_coeffs(coeffs_q, c_q: Fraction, new_den: int = NEW_DEN):
    """Return (coeffs_new_q, list_of_num_den_pairs).

    coeffs_new_q[j] = coeffs_q[j] / c_q  (exact rationals).
    For the JSON dump we then rewrite each at a common denominator
    new_den, using FLOOR rounding for the numerator so the resulting
    rational is <= coeffs_new_q (which keeps min_G >= 1).

    Actually we need the *new G evaluations* still >= 1, but rounding all
    numerators downward does NOT preserve a uniform sign condition (cosine
    can be negative).  Safer: keep the exact rational divisions and pick a
    common denominator that holds them exactly.
    """
    new = [a / c_q for a in coeffs_q]
    # Find lcm of denominators
    from math import lcm
    den = 1
    for q in new:
        den = lcm(den, q.denominator)
    pairs = [(q.numerator * (den // q.denominator), den) for q in new]
    return new, pairs, den


def main() -> dict:
    print("=" * 76)
    print("Normalize v3 G so that min_G >= 1.0 rigorously")
    print("=" * 76)

    # ---------------- 1.  Load v3 coefficients ----------------
    coeffs_q, d1_q, d2_q, l1_q = v3.load_coeffs_from_json(SRC_JSON)
    print(f"\nLoaded {len(coeffs_q)} coefficients from {SRC_JSON.name}")
    print(f"  delta_1 = {float(d1_q):.4f}  delta_2 = {float(d2_q):.4f}  "
          f"lambda_1 = {float(l1_q):.4f}")

    # ---------------- 2.  Compute rigorous min_G_LB at higher precision ----
    print("\n[Rigorous min_G_LB] re-evaluating v3 G at multiple grid sizes")
    grids = [(200001, 256), (400001, 256), (800001, 512)]
    m_lo_best = None
    for n_grid, prec in grids:
        v3.configure_precision(prec)
        coeffs_arb = [v3.Q_arb(q) for q in coeffs_q]
        coeffs_float = [float(q) for q in coeffs_q]
        t0 = time.time()
        m_lo, lip_rem = v3.min_G_lower_bound(coeffs_arb, coeffs_float, n_grid=n_grid)
        el = time.time() - t0
        m_lo_f = float(m_lo.lower())
        lip_f = float(lip_rem.upper())
        print(f"  grid={n_grid:>7d}  prec={prec:>4d}  "
              f"min_G_LB={m_lo_f:.10f}  lip_rem<={lip_f:.3e}  ({el:.1f}s)")
        if m_lo_best is None or m_lo_f > float(m_lo_best.lower()):
            m_lo_best = m_lo

    print(f"\nBest rigorous min_G_LB = {float(m_lo_best.lower()):.10f}")

    # ---------------- 3.  Pick rational c_q <= m_lo / TARGET ----------------
    # We want min_G_new = min_G_old / c_q to satisfy min_G_new >= TARGET.
    # So we pick c_q <= m_lo / TARGET.   With TARGET = 1.0001 we get a margin.
    TARGET = Fraction(10001, 10000)  # 1.0001 exactly as rational
    target_arb = arb(TARGET.numerator) / arb(TARGET.denominator)
    c_q = safe_rational_lower(m_lo_best / target_arb, denom=NEW_DEN)
    c_f = float(c_q)
    print(f"\nTarget min_G_new >= {float(TARGET):.6f}")
    print(f"Picked rational divisor c_q = {c_q.numerator}/{c_q.denominator}")
    print(f"  c_q = {c_f:.15f}")
    assert c_q > 0
    # Rigorous check: c_q * TARGET <= m_lo_best
    c_arb = arb(c_q.numerator) / arb(c_q.denominator)
    margin = m_lo_best - c_arb * target_arb
    assert float(margin.lower()) > 0, (
        f"safety: c_q*TARGET={float(c_arb*target_arb)} not provably <= "
        f"m_lo_LB={float(m_lo_best.lower())}")
    print(f"  arb-certified  m_lo - c_q*{float(TARGET):.4f} >= "
          f"{float(margin.lower()):.3e}  OK")

    # ---------------- 4.  Rescale coefficients ----------------
    new_q, pairs, common_den = rescale_coeffs(coeffs_q, c_q, new_den=NEW_DEN)
    print(f"\nRescaled {len(new_q)} coefficients.  Common denominator = {common_den}")

    # ---------------- 5.  Re-run rigorous certification ----------------
    #  Use the SAME grid (800001 / prec 512) that we used to pick c_q, so the
    #  rigorous min_G_new LB is computed at matching tightness.
    print("\n[Re-certify with normalized G at matching grid=800001, prec=512]")
    v3.configure_precision(512)
    new_arb = [v3.Q_arb(q) for q in new_q]
    new_float = [float(q) for q in new_q]
    m_lo_new, lip_rem_new = v3.min_G_lower_bound(new_arb, new_float, n_grid=800001)
    m_lo_new_f = float(m_lo_new.lower())
    print(f"  min_G_norm_LB (grid 800001, prec 512) = {m_lo_new_f:.10f}  "
          f"(lip_rem<={float(lip_rem_new.upper()):.3e})")

    # And run the full certify (lower-grid) for M_cert (it does not matter:
    # the gain `a` uses (min_G_LB)^2; we will pass the high-grid m_lo down).
    print("\n[Run certify_combined with normalized G (grid 200001, xi_max 1e5)]")
    r_norm = v3.certify_combined(d2_q, l1_q, new_q,
                                 xi_max=10**5, prec_bits=256,
                                 arb_grid=200001, verbose=True)

    # ---------------- 6.  Also re-run baseline (un-normalized) for comparison ---
    print("\n[Re-certify with ORIGINAL G  (for comparison)]")
    r_orig = v3.certify_combined(d2_q, l1_q, coeffs_q,
                                 xi_max=10**5, prec_bits=256,
                                 arb_grid=200001, verbose=True)

    # ---------------- 7.  Sanity: min_G_new >= 1.0 ----------------
    ok = m_lo_new_f >= 1.0
    print(f"\nmin_G(normalized)_LB (grid 800001/prec 512) = {m_lo_new_f:.10f}  "
          f"=> >= 1.0 ?  {'YES' if ok else 'NO'}")
    if not ok:
        print(f"  shortfall = {1.0 - m_lo_new_f:.3e}")
    # Save these high-grid numbers for output
    r_norm["min_G_lower_highgrid"] = m_lo_new_f
    r_norm["min_G_lip_rem_highgrid"] = float(lip_rem_new.upper())

    # ---------------- 8.  Sanity: M_cert preserved ----------------
    print(f"\nM_cert ORIGINAL    = {r_orig['M_cert_lower']:.8f}")
    print(f"M_cert NORMALIZED  = {r_norm['M_cert_lower']:.8f}")
    print(f"|delta|            = {abs(r_orig['M_cert_lower'] - r_norm['M_cert_lower']):.3e}")

    # ---------------- 9.  Save JSON ----------------
    out = {
        "source_json": SRC_JSON.name,
        "delta_1": [d1_q.numerator, d1_q.denominator],
        "delta_2": [d2_q.numerator, d2_q.denominator],
        "lambda_1": [l1_q.numerator, l1_q.denominator],
        # u = 1/2 + delta_1 = 1/2 + 69/500 = 319/500 = 0.638
        "u": [319, 500],
        "n_modes": len(new_q),
        "common_denominator": common_den,
        "coeffs_num_den": pairs,
        "rescale_divisor": [c_q.numerator, c_q.denominator],
        "rescale_divisor_float": c_f,
        "min_G_LB_before_rescale": float(m_lo_best.lower()),
        "min_G_LB_after_rescale":  r_norm["min_G_lower"],
        "min_G_LB_after_rescale_highgrid": m_lo_new_f,
        "min_G_LB_after_rescale_highgrid_lip_rem": float(lip_rem_new.upper()),
        "min_G_grid_used_for_certificate": 800001,
        "min_G_prec_used_for_certificate": 512,
        "M_cert_before_rescale":   r_orig["M_cert_lower"],
        "M_cert_after_rescale":    r_norm["M_cert_lower"],
        "S_1_upper_before_rescale": r_orig["S_1_upper"],
        "S_1_upper_after_rescale":  r_norm["S_1_upper"],
        "a_gain_before_rescale":   r_orig["a_gain_lower"],
        "a_gain_after_rescale":    r_norm["a_gain_lower"],
        "K_2_upper":               r_norm["K_2_upper"],
        "k_1_mid":                 r_norm["k_1_mid"],
        "min_G_after_rescale_ok":  ok,
        "rigorous_M_cert_lower_after":  r_norm["M_cert_lower"],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_JSON}")

    # ---------------- 10.  Save a small Python wrapper -------------
    py_text = (
        '"""v3 G coefficients, rescaled so that min G >= 1.0 rigorously.\n'
        '\n'
        'Generated by _v3_g_normalize.py from _cohn_elkies_128_greopt_coeffs.json.\n'
        '\n'
        'See _cohn_elkies_128_v3_normalized.json for the rigorous certificates.\n'
        '"""\n\n'
        'from fractions import Fraction\n\n'
        f'DELTA_1 = Fraction({d1_q.numerator}, {d1_q.denominator})\n'
        f'DELTA_2 = Fraction({d2_q.numerator}, {d2_q.denominator})\n'
        f'LAMBDA_1 = Fraction({l1_q.numerator}, {l1_q.denominator})\n'
        f'U = Fraction(319, 500)   # = 1/2 + delta_1 = 0.638\n'
        f'N_MODES = {len(new_q)}\n'
        f'COMMON_DENOMINATOR = {common_den}\n\n'
        f'RESCALE_DIVISOR = Fraction({c_q.numerator}, {c_q.denominator})\n\n'
        f'MIN_G_LB_GRID_200001 = {r_norm["min_G_lower"]!r}\n'
        f'MIN_G_LB_GRID_800001_PREC_512 = {m_lo_new_f!r}   # rigorous certificate\n'
        f'M_CERT_LOWER = {r_norm["M_cert_lower"]!r}\n'
        f'S_1_UPPER = {r_norm["S_1_upper"]!r}\n'
        f'K_2_UPPER = {r_norm["K_2_upper"]!r}\n\n'
        f'COEFFS_NUM_DEN = [\n'
    )
    for n, d in pairs:
        py_text += f"    ({n}, {d}),\n"
    py_text += "]\n\n"
    py_text += (
        "COEFFS = [Fraction(n, d) for n, d in COEFFS_NUM_DEN]\n"
    )
    with open(OUT_PY, "w") as f:
        f.write(py_text)
    print(f"Wrote {OUT_PY}")
    return out


if __name__ == "__main__":
    main()
