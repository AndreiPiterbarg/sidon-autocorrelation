"""Quick empirical exploration: how high can M(g) get over a parametric
family of g, using the Shor relaxation of the inner inf?

This is a smoke test for the Path B framework.  If the best M(g) we
find here is far below CS 2017's 1.2802, that signals our Shor
relaxation is too loose and we need Lasserre level 2.

Output: ranked list of g's by M(g).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bochner_sos.g_candidates import (g_central_bump, g_constant, g_cosine,
                                      g_pwc, g_pwl)
from bochner_sos.m_g_eval import M_g


def explore(d: int = 80):
    print(f"=== Path B viability exploration (d={d}, Shor SDP) ===")
    print()
    results = []

    # 1. constant
    res = M_g(g_constant(1.0), d=d, force_method='convex')
    results.append(("g = 1 (constant)", res['M'], res['int_g'], res['is_psd']))

    # 2. central bump c0 + c1*(1-2|t|), various
    for c0, c1 in [(1.0, 0.0), (1.0, 0.25), (1.0, 0.5), (1.0, 1.0),
                   (0.5, 1.0), (0.25, 1.0), (0.1, 1.0)]:
        res = M_g(g_central_bump(c0, c1), d=d, force_method='convex')
        results.append((f"central bump c0={c0}, c1={c1}", res['M'],
                        res['int_g'], res['is_psd']))

    # 3. piecewise-constant: try heights = (1, h1, h2, ..., 1) with h symmetric
    for prof_name, profile in [
        ("pwc [1,1,1]", [1.0, 1.0, 1.0]),
        ("pwc [1,2,1]", [1.0, 2.0, 1.0]),
        ("pwc [1,1.5,1.5,1]", [1.0, 1.5, 1.5, 1.0]),
        ("pwc [0.5,1,1,0.5]", [0.5, 1.0, 1.0, 0.5]),  # peak in middle
        ("pwc [1,2,3,2,1]", [1.0, 2.0, 3.0, 2.0, 1.0]),
        ("pwc [2,1,0.5,1,2]", [2.0, 1.0, 0.5, 1.0, 2.0]),  # boundary-heavy
        ("pwc [1,3,1]", [1.0, 3.0, 1.0]),  # narrow central spike
    ]:
        res = M_g(g_pwc(profile), d=d, force_method='convex')
        results.append((prof_name, res['M'], res['int_g'], res['is_psd']))

    # 4. cosines
    for label, coeffs in [
        ("cos c=[1, -0.5]", [1.0, -0.5]),
        ("cos c=[1, 0.5]", [1.0, 0.5]),
        ("cos c=[1, -0.5, 0.1]", [1.0, -0.5, 0.1]),
        ("cos c=[1, -0.3, 0.3]", [1.0, -0.3, 0.3]),
        ("cos c=[1, -0.2, 0.2, -0.1]", [1.0, -0.2, 0.2, -0.1]),
    ]:
        res = M_g(g_cosine(coeffs), d=d, force_method='convex')
        results.append((label, res['M'], res['int_g'], res['is_psd']))

    # 5. piecewise-linear: heights at break points
    for label, heights in [
        ("pwl [1,1,1]", [1.0, 1.0, 1.0]),
        ("pwl [1,2,1]", [1.0, 2.0, 1.0]),
        ("pwl [1,1,2,1,1]", [1.0, 1.0, 2.0, 1.0, 1.0]),
        ("pwl [2,1,2]", [2.0, 1.0, 2.0]),  # boundary peaks
    ]:
        res = M_g(g_pwl(heights), d=d, force_method='convex')
        results.append((label, res['M'], res['int_g'], res['is_psd']))

    # Sort by M descending
    results.sort(key=lambda r: -r[1])

    print(f"{'Rank':>4} {'M':>10} {'int_g':>10} {'PSD':>5}  Description")
    print("-" * 80)
    for rank, (name, M, int_g, is_psd) in enumerate(results, start=1):
        psd_str = "Y" if is_psd else "n"
        print(f"{rank:>4} {M:>10.6f} {int_g:>10.4f} {psd_str:>5}  {name}")

    print()
    print(f"Best M(g) found: {results[0][1]:.6f} from `{results[0][0]}`")
    print(f"Compare: CS 2017 proved M >= 1.2802 with their step-function dual.")
    if results[0][1] < 1.0:
        print()
        print("All M values <= 1. This indicates the Shor relaxation is too loose")
        print("on these g shapes; the TRUE inf_f is likely higher but Shor")
        print("under-bounds it. To match CS 2017 we need either")
        print("(a) Lasserre level-2 relaxation (tighter inner bound), OR")
        print("(b) Outer SDP that jointly optimizes g and inner relaxation.")
        print("This is the heavy lift documented in README.md Stage 2.")
    elif results[0][1] >= 1.28:
        print()
        print("FOUND: M(g) >= 1.28 with simple parametric g. Path B viable.")
    else:
        print()
        print(f"Best M = {results[0][1]:.4f}. Some uplift over baseline 1.0,")
        print("but still below CS 2017's 1.2802. Need outer optimization of g.")


if __name__ == "__main__":
    explore(d=80)
