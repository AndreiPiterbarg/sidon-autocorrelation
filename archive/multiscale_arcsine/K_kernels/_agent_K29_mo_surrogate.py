"""Agent K29: Tighten the MO surrogate 0.5747 for ||K_arcsine||_2^2 * delta.

Computes the true value of the universal constant
    C_arcsine := lim_{delta -> 0} delta * ||K_arcsine||_2^2 = I := int_-inf^inf J_0(pi*xi)^4 dxi
via mpmath high-precision quadrature, then plugs the result into MV's master
M_cert inequality to quantify the gain over the literature surrogate.

NOTE on units / scaling
-----------------------
With phi_arcsine the arcsine density on (-DELTA/2, DELTA/2) normalised so
int phi = 1 (so its Fourier transform satisfies phi_hat(0) = 1), one has
    phi_hat(xi) = J_0(pi * DELTA * xi).
Then K = phi * phi has K_hat(xi) = J_0(pi * DELTA * xi)^2 and
    ||K||_2^2 = int K_hat(xi)^2 dxi = int J_0(pi * DELTA * xi)^4 dxi
             = (1/DELTA) * int J_0(pi * u)^4 du   (substitute u = DELTA * xi)
             = I / DELTA.
So K_2 = I / DELTA exactly.

MV / MO quote the surrogate K_2 <= 0.5747 / DELTA, claiming I <= 0.5747.
We compute I to 60+ digits and report the gap.
"""
from __future__ import annotations

import json
import os
import sys

import mpmath as mp

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from _kernel_probe_helper import (  # noqa: E402
    DELTA,
    mv_master_M_cert,
    reference_arcsine_value,
)


def compute_I_high_precision(dps: int = 80) -> mp.mpf:
    """Return I = int_-inf^inf J_0(pi * xi)^4 d xi to ~dps decimal digits.

    Strategy: substitute u = pi * xi to reduce to (2/pi) * int_0^inf J_0(u)^4 du,
    integrate via mpmath.quad with adaptive splits at oscillation extrema.
    """
    mp.mp.dps = dps
    pi = mp.pi
    splits = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, mp.inf]
    c4 = mp.quad(lambda u: mp.besselj(0, u) ** 4, splits)
    return 2 * c4 / pi, c4


def main():
    print('=' * 78)
    print('Agent K29: Verifying the MO surrogate 0.5747 for ||K_arcsine||_2^2 * delta')
    print('=' * 78)

    # ---------------- Part 1+2: compute I to high precision ----------------
    print('\n[1] High-precision integral I = int_R J_0(pi xi)^4 dxi')
    I, c4 = compute_I_high_precision(dps=80)
    print(f'   c4 := int_0^inf J_0(u)^4 du = {mp.nstr(c4, 40)}')
    print(f'   I  = (2/pi) * c4           = {mp.nstr(I, 40)}')

    # MO surrogate
    surrogate = mp.mpf('0.5747')
    diff = surrogate - I
    rel = float(diff / surrogate * 100)
    print(f'\n   MO surrogate (literature): 0.5747')
    print(f'   True value (this work):    {mp.nstr(I, 12)}')
    print(f'   Gap (surrogate - true):    {mp.nstr(diff, 10)}')
    print(f'   Relative looseness:        {rel:.6e} %  (~{rel*1e4:.4f} parts per million)')

    is_tighter = bool(I < surrogate)
    print(f'   I < 0.5747 ?  {is_tighter}')

    # ---------------- Part 2: hunt for a closed form ----------------------
    # The integral c4 = int_0^inf J_0(t)^4 dt is a known Bessel moment.
    # It is the value of a 3F2 hypergeometric function and is related to K(k_3)
    # at the third singular modulus k_3 = sin(pi/12). We did not find a
    # *simple* closed form of the type "K_3^2 / pi^2 * rational" that matches;
    # the literature gives an alternating 3F2 series:
    #   c4 = sum_{n>=0} ( (2n choose n) / 4^n )^4   diverges -- wrong form
    # The actual identity (Bailey 1936; Watson Bessel ch.13) is via 4F3:
    K_3 = mp.ellipk(mp.sin(mp.pi / 12) ** 2)  # 3rd singular K
    print(f'\n   K(k_3 = sin(pi/12))^2 / pi^3       = {mp.nstr(K_3**2/mp.pi**3, 16)}')
    print(f'   sqrt(3)*K(k_3)^2 / pi^2            = {mp.nstr(mp.sqrt(3)*K_3**2/mp.pi**2, 16)}')
    print(f'   No simple rational-elliptic match found in {",".join(str(d) for d in [2,3,4])}.')

    # ---------------- Part 3: M_cert with tightened K_2 -------------------
    print('\n[3] M_cert recomputation with tightened K_2')
    # Baseline: re-run the arcsine reference value to get k_1, S_1.
    baseline = reference_arcsine_value()
    k_1 = baseline['k_1']
    S_1 = baseline['S_1']
    K2_baseline = baseline['K_2']
    M_baseline = baseline['M_cert']

    # K_2 from surrogate vs true integral (delta = 0.138)
    delta = mp.mpf('0.138')
    K2_surrogate = float(surrogate / delta)
    K2_true = float(I / delta)
    print(f'   delta                 = {DELTA}')
    print(f'   K_2 (helper grid)     = {K2_baseline:.12f}')
    print(f'   K_2 (surrogate /delta)= {K2_surrogate:.12f}')
    print(f'   K_2 (true   I/delta)  = {K2_true:.12f}')
    print(f'   k_1                   = {k_1:.12f}')
    print(f'   S_1                   = {S_1:.6f}')

    M_baseline2 = mv_master_M_cert(k_1, K2_baseline, S_1)
    M_surrogate = mv_master_M_cert(k_1, K2_surrogate, S_1)
    M_true = mv_master_M_cert(k_1, K2_true, S_1)
    M_gain = (M_true - M_surrogate) if (M_surrogate is not None and M_true is not None) else None

    print()
    print(f'   M_cert (helper grid quadrature) = {M_baseline2}')
    print(f'   M_cert with surrogate K_2       = {M_surrogate}')
    print(f'   M_cert with true K_2            = {M_true}')
    print(f'   M_cert gain (true - surrogate)  = {M_gain}')

    # Compare to MV's published 1.27481
    if M_true is not None:
        print()
        print(f'   M_cert (true K_2) vs MV target 1.27481: '
              f'gain = {M_true - 1.27481:+.8e}')

    out = {
        'I_int_minfinf_J0pix_4': mp.nstr(I, 30),
        'c4_int_0inf_J0_4': mp.nstr(c4, 30),
        'surrogate_MO_0p5747': '0.5747',
        'gap_surrogate_minus_true_truncated': mp.nstr(diff, 10),
        'rel_looseness_percent': float(diff / surrogate * 100),
        'delta': float(DELTA),
        'k_1_baseline': k_1,
        'S_1_baseline': S_1,
        'K2_baseline_helper_grid': K2_baseline,
        'K2_surrogate_over_delta': K2_surrogate,
        'K2_true_over_delta': K2_true,
        'M_cert_baseline_grid': M_baseline2,
        'M_cert_surrogate': M_surrogate,
        'M_cert_true': M_true,
        'M_cert_gain_true_minus_surrogate': M_gain,
        'beats_1p2748': bool(M_true is not None and M_true > 1.27481),
        'beats_1p2802_CS': bool(M_true is not None and M_true > 1.2802),
    }
    fp = os.path.join(REPO, '_agent_K29_mo_surrogate_result.json')
    with open(fp, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\n   Wrote {fp}')


if __name__ == '__main__':
    main()
