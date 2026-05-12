"""
Pilot run: compute Q2 = I_R(a,b) * max(a,b) at small grid to test conjecture.
Use lower dps=20 to keep each call <30s.  We'll bump precision for the verified
non-trivial cases later.
"""
import time, json
from mpmath import mp, mpf, besselj, quadosc, pi as mp_pi, inf, nstr

mp.dps = 20

def compute_Q2(a, b):
    a = mpf(a); b = mpf(b)
    f = lambda t: besselj(0, a*t)**2 * besselj(0, b*t)**2
    period = mp_pi / max(a, b)
    Ih = quadosc(f, [0, inf], period=period)
    IR = (mpf(2)/mp_pi) * Ih
    Q2 = IR * max(a, b)
    return Ih, IR, Q2

grid = [
    (1, 0.99),
    (1, 0.5),
    (1, 0.326),
    (1, 0.1),
    (1, 0.01),
    (0.138, 0.045),
    (0.138, 0.07),
    (0.138, 0.10),
    (0.138, 0.13),
    (1, 1),
    (0.138, 0.138),
    (0.5, 0.5),
]

rows = []
print(f"dps={mp.dps}")
print(f"{'a':>8} {'b':>8}  {'Q2=I_R*max':>30}  {'time':>6}")
for a, b in grid:
    t0 = time.time()
    Ih, IR, Q2 = compute_Q2(a, b)
    dt = time.time() - t0
    print(f"{float(a):>8.4f} {float(b):>8.4f}  {nstr(Q2, 25):>30}  {dt:>6.1f}")
    rows.append({"a": str(a), "b": str(b), "I_half": str(Ih),
                 "I_R": str(IR), "Q2": str(Q2)})

with open("_n3_pilot.json", "w") as f:
    json.dump({"dps": mp.dps, "rows": rows}, f, indent=2)
print("Wrote _n3_pilot.json")
