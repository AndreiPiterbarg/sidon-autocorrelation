"""Verify F at L0 matches M1 reference."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cloninger-steinerberger', 'cpu'))
from run_cascade import run_level0

# M1 reference: at (n=3, m=10, c=1.28), F leaves 172 / 1891.
# At (n=4, m=10, c=1.28), F leaves 1014 / 91881.
# At (n=5, m=5, c=1.28), F leaves 558 / 316251.
configs = [(3, 10, 1.28, 172), (4, 10, 1.28, 1014), (5, 5, 1.28, 558)]

for n, m, c, expected_F in configs:
    r_W = run_level0(n, m, c, verbose=False, use_F=False)
    r_F = run_level0(n, m, c, verbose=False, use_F=True)
    sW = r_W['n_survivors']
    sF = r_F['n_survivors']
    print(f"(n={n}, m={m}, c={c}): W={sW}  F={sF}  (M1 ref F={expected_F})  "
          f"{'PASS' if sF == expected_F else 'MISMATCH'}")
