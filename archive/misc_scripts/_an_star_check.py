"""Check a_n^* (heuristic upper bound on cascade discrete optimum) at various n.

a_n^* := min over a in simplex of max_W TV_W(a).
Cloninger-Steinerberger: C_{1a} >= a_n^* (LIMIT as n -> infinity).
At finite n, a_n^* is an UPPER bound on C_{1a} via the cascade.

If a_n^* < 1.281 at any tractable n, the cascade CANNOT prove c=1.281.
Use the existing project's heuristic_an_star.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger'))
from cs_refined_lp import heuristic_an_star

print(f"a_n^* = min_a max_W TV_W(a)  --  cascade discrete optimum")
print(f"Target: prove a_n^* >= 1.281")
print(f"=" * 60)

results = {}
for n in [4, 8, 16, 32, 64, 128]:
    t0 = time.time()
    val, a = heuristic_an_star(n, n_restarts=64, n_iters=2000, seed=42)
    wall = time.time() - t0
    print(f"  n={n:>3} (d={2*n:>4}): a_n^* <= {val:.6f}  "
          f"vs 1.281 [{'BELOW' if val < 1.281 else 'above'}]  "
          f"vs 1.2802 [{'BELOW' if val < 1.2802 else 'above'}]  "
          f"({wall:.0f}s)")
    results[n] = float(val)

print(f"=" * 60)
print(f"\nInterpretation:")
print(f"  a_n^* is an UPPER bound on the cascade's discrete minimum at scale n.")
print(f"  If a_n^* < 1.281 at any n, the cascade at that n CANNOT prove c=1.281.")
print(f"  As n -> infinity, a_n^* -> C_{{1a}} (the true Sidon constant).")
