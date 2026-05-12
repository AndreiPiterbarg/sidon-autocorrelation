"""
N3: High-precision numerical verification of the conjecture
  I_R(a, b) := integral_{-inf}^{inf} J_0(pi*a*xi)^2 J_0(pi*b*xi)^2 dxi = C / max(a, b)
where C is conjectured to be the MO constant ~ 0.5747.

Substitution t = pi*xi: I_R(a, b) = (2/pi) * I_half(a, b)  where
  I_half(a, b) := integral_0^inf J_0(a t)^2 J_0(b t)^2 dt.

Strategy at 50 dps:
- J_0(a t)^2 J_0(b t)^2 ~ (1/(pi^2 t^2 sqrt(ab))) * (1+cos(2at-pi/2))(1+cos(2bt-pi/2))/(...)
  i.e. the integrand decays like 1/t^2 for large t, with oscillating corrections.
- Split: integral_0^T  + integral_T^inf, with T = 200 / min(a, b) say.
- For the finite part [0, T] we use quadgl (Gauss-Legendre) with subdivision.
- For the tail we use quadosc which handles oscillating decay.

To make this tractable at dps=50 over many pairs, reduce subdivision overhead
by lowering dps to 30 (still well-above standard "20 digits = certificate").
"""

import json
import time
from mpmath import mp, mpf, besselj, quadosc, quad, mpc, pi as mp_pi, inf as mp_inf, nstr

# 30 digits is plenty for a numerical certificate; 50 is overkill and OOMs.
mp.dps = 30


def integrand(a, b):
    a = mpf(a); b = mpf(b)
    return lambda t: besselj(0, a*t)**2 * besselj(0, b*t)**2


def I_half(a, b):
    """Compute I_half(a,b) = integral_0^inf J_0(at)^2 J_0(bt)^2 dt."""
    a = mpf(a); b = mpf(b)
    f = integrand(a, b)
    # Split at a "transition" T where the asymptotic series becomes accurate.
    # Smallest period from both factors: pi / max(a, b).
    period_min = mp_pi / max(a, b)
    # Use ~ 30 periods of the faster oscillation as a transition.
    T = 30 * period_min
    # Finite part: subdivide into pieces of size period_min, integrate each with quad.
    n_pieces = 30
    edges = [k * T / n_pieces for k in range(n_pieces + 1)]
    head = mpf(0)
    for k in range(n_pieces):
        head += quad(f, [edges[k], edges[k+1]])
    # Tail: oscillatory, use quadosc with period equal to pi / max(a, b).
    period_tail = mp_pi / max(a, b)
    tail = quadosc(f, [T, mp_inf], period=period_tail)
    return head + tail


def verify_one(a, b):
    a = mpf(a); b = mpf(b)
    Ih = I_half(a, b)
    IR = (mpf(2) / mp_pi) * Ih
    m = max(a, b)
    Q1 = Ih * m
    Q2 = IR * m
    return Ih, IR, Q1, Q2


def fmt(x, d=20):
    return nstr(x, d, strip_zeros=False)


grid_main = [
    (1, 0.99),
    (1, 0.5),
    (1, 0.326),
    (1, 0.1),
    (1, 0.01),
]

grid_scaling = [
    (0.138, 0.045),
    (0.138, 0.07),
    (0.138, 0.10),
    (0.138, 0.13),
]

grid_diagonal = [
    (1, 1),
    (0.138, 0.138),
    (0.5, 0.5),
]

grid_residual = [(1, mpf(k) / mpf(20)) for k in range(1, 21)]  # r = 0.05, ..., 1.00


def run_grid(label, grid):
    print(f"\n=== {label} ===")
    print(f"{'a':>10} {'b':>10}  {'I_half*max':>30}  {'I_R*max (Q2)':>30}  {'time(s)':>8}")
    rows = []
    for a, b in grid:
        a = mpf(a); b = mpf(b)
        t0 = time.time()
        Ih, IR, Q1, Q2 = verify_one(a, b)
        dt = time.time() - t0
        print(f"{fmt(a, 8):>10} {fmt(b, 8):>10}  {fmt(Q1, 26):>30}  {fmt(Q2, 26):>30}  {dt:>8.2f}")
        rows.append({
            "a": str(a), "b": str(b),
            "I_half": str(Ih), "I_R": str(IR),
            "Q1": str(Q1), "Q2": str(Q2),
        })
    return rows


def main():
    print(f"mpmath dps = {mp.dps}")
    out = {"dps": mp.dps, "results": {}}

    out["results"]["main"]     = run_grid("main grid (a=1)", grid_main)
    out["results"]["scaling"]  = run_grid("scaling (a=0.138)", grid_scaling)
    out["results"]["diagonal"] = run_grid("diagonal (a=b)", grid_diagonal)

    # ---- residual function f(r) = r * I_R(1, r) / I_R(1, 1) ----
    print("\n=== residual f(r) = r * I_R(1, r) / I_R(1, 1) ===")
    IR_diag = (mpf(2) / mp_pi) * I_half(mpf(1), mpf(1))
    print(f"I_R(1,1) = {fmt(IR_diag, 30)}   (this is our 'C_diag')")
    res = []
    for (a, b) in grid_residual:
        a = mpf(a); b = mpf(b)
        Ih = I_half(a, b)
        IR = (mpf(2) / mp_pi) * Ih
        ratio = IR / IR_diag                       # = 1 iff conjecture (since max(1,r)=1)
        f_r   = b * ratio                          # r * I_R(1,r)/I_R(1,1)
        print(f"r = {fmt(b, 8):>10}   I_R(1,r) = {fmt(IR, 22):>26}   "
              f"ratio = {fmt(ratio, 18):>22}   f(r) = {fmt(f_r, 18):>22}")
        res.append({"r": str(b), "I_R_1_r": str(IR),
                    "ratio_to_diag": str(ratio), "f_r": str(f_r)})
    out["results"]["residual"] = res
    out["C_diag_estimate"] = str(IR_diag)

    # ---- summary ----
    print("\n=== summary ===")
    all_Q2 = []
    for key in ("main", "scaling", "diagonal"):
        for r in out["results"][key]:
            all_Q2.append((r["a"], r["b"], mpf(r["Q2"])))
    Q2_vals = [v for _, _, v in all_Q2]
    print(f"Conjecture: Q2 := I_R(a,b)*max(a,b) is constant ~ 0.5747.")
    print(f"min Q2  = {fmt(min(Q2_vals), 30)}")
    print(f"max Q2  = {fmt(max(Q2_vals), 30)}")
    print(f"spread  = {fmt(max(Q2_vals) - min(Q2_vals), 30)}")
    out["Q2_min"] = str(min(Q2_vals))
    out["Q2_max"] = str(max(Q2_vals))
    out["Q2_spread"] = str(max(Q2_vals) - min(Q2_vals))

    with open("_n3_bessel_conjecture.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote _n3_bessel_conjecture.json")


if __name__ == "__main__":
    main()
