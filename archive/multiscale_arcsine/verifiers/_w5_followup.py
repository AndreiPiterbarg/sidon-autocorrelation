"""
W5 follow-up: contextualize the FLAG against the V1 baseline.

V1 (2-scale): min w_j ~ 3.3e-5
W5 (3-scale): min w_j = 2.08e-4 at j=147

Check:
- How does W5's min compare to V1's 3.3e-5? (~6.3x larger -> better)
- Is the min sharp (single isolated near-zero) or broad?
- Distance from the Bessel zeros: closest j*delta1/u to a Bessel zero?
- Estimate S_1 contribution upper bound w/ all w_j > 2.08e-4
"""
import mpmath as mp
import json

mp.mp.dps = 30

delta1 = mp.mpf("0.138")
delta2 = mp.mpf("0.055")
delta3 = mp.mpf("0.025")
c1 = mp.mpf("0.85")
c2 = mp.mpf("0.10")
c3 = mp.mpf("0.05")
u = mp.mpf("0.638")
pi = mp.pi

def K_hat(xi):
    return (c1 * mp.besselj(0, pi * delta1 * xi) ** 2
            + c2 * mp.besselj(0, pi * delta2 * xi) ** 2
            + c3 * mp.besselj(0, pi * delta3 * xi) ** 2)

# Bessel J_0 zeros (first 10): 2.4048, 5.5201, 8.6537, 11.7915, 14.9309, 18.0711, 21.2116, 24.3525, 27.4935, 30.6346
# At j=147: xi = 147/0.638 = 230.407
# pi*delta1*xi = pi*0.138*230.407 = 99.91 -> near zero of J_0
# Zeros of J_0 are ~ (n - 0.25)*pi, so 99.91/pi = 31.81 -> close to 31.75 = (32 - 0.25), so n=32

j_star = 147
xi_star = mp.mpf(j_star) / u
arg1 = pi * delta1 * xi_star
arg2 = pi * delta2 * xi_star
arg3 = pi * delta3 * xi_star

print(f"j*=147 analysis:")
print(f"  xi*           = {float(xi_star):.6f}")
print(f"  pi*d1*xi      = {float(arg1):.6f}  (J_0 zeros: 2.40, 5.52, ..., n*pi - pi/4)")
print(f"  pi*d2*xi      = {float(arg2):.6f}")
print(f"  pi*d3*xi      = {float(arg3):.6f}")
print(f"  J_0(arg1)     = {float(mp.besselj(0, arg1)):.6e}")
print(f"  J_0(arg2)     = {float(mp.besselj(0, arg2)):.6e}")
print(f"  J_0(arg3)     = {float(mp.besselj(0, arg3)):.6e}")
print()

# Find nearest J_0 zero
arg1_over_pi = arg1 / pi
print(f"  arg1/pi       = {float(arg1_over_pi):.6f}")
print(f"  -> nearest 'n - 0.25' giving J_0 zero at n*pi-pi/4")
n_guess = int(round(float(arg1_over_pi) + 0.25))
zero_n = mp.besseljzero(0, n_guess)
print(f"  n_guess       = {n_guess}, J_0 zero #{n_guess} = {float(zero_n):.6f}")
print(f"  |arg1 - zero| = {float(abs(arg1 - zero_n)):.6e}")
print()

# Bessel zeros bound check: for j in 1..200, is the J_0 component the dominant near-zero?
# Recall c1*J_0(arg1)^2 dominates magnitude. Even if J_0(arg1)~0, the c2,c3 terms give lift.

# Lift contributions at j=147:
val1 = c1 * mp.besselj(0, arg1)**2
val2 = c2 * mp.besselj(0, arg2)**2
val3 = c3 * mp.besselj(0, arg3)**2
total = val1 + val2 + val3
print(f"  Components at j=147:")
print(f"    c1*J_0(d1*pi*xi)^2 = {float(val1):.6e}")
print(f"    c2*J_0(d2*pi*xi)^2 = {float(val2):.6e}")
print(f"    c3*J_0(d3*pi*xi)^2 = {float(val3):.6e}")
print(f"    sum                = {float(total):.6e}")
print()

# How many j have w_j between 1e-4 and 1e-3?
between = []
above_1em3 = []
for j in range(1, 201):
    xi = mp.mpf(j) / u
    w = K_hat(xi)
    if mp.mpf("1e-4") < w < mp.mpf("1e-3"):
        between.append((j, float(w)))
    if w >= mp.mpf("1e-3"):
        above_1em3.append((j, float(w)))

print(f"j with 1e-4 < w_j < 1e-3: {len(between)}")
print(f"j with w_j >= 1e-3:       {len(above_1em3)}")
print(f"j with w_j < 1e-4:        {200 - len(between) - len(above_1em3)}")
print()

# Compare to V1 baseline ~3.3e-5
v1_baseline = mp.mpf("3.3e-5")
w5_min = mp.mpf("2.081738e-4")
print(f"V1 (2-scale) min w_j:   ~{float(v1_baseline):.3e}")
print(f"W5 (3-scale) min w_j:    {float(w5_min):.3e}")
print(f"Improvement factor:      {float(w5_min/v1_baseline):.2f}x")
print()

# Sanity: is the W5 kernel's min still 'soft FLAG' or 'hard FLAG'?
# The 1e-3 cutoff is met by 194/200 = 97% of frequencies.
# V1 was below 1e-4 → "soft FLAG"; W5 is above 1e-4 everywhere → improved.

out = {
    "j_star": j_star,
    "min_w_j": float(w5_min),
    "v1_baseline_approx": float(v1_baseline),
    "improvement_vs_v1": float(w5_min / v1_baseline),
    "num_below_1e-3": 6,
    "num_below_1e-4": 0,
    "num_in_1e-4_to_1e-3": len(between),
    "components_at_jstar": {
        "c1J0sq": float(val1),
        "c2J0sq": float(val2),
        "c3J0sq": float(val3),
    },
    "bessel_zero_distance_arg1": float(abs(arg1 - zero_n)),
}
with open("_w5_followup.json", "w") as f:
    json.dump(out, f, indent=2)

print("Wrote _w5_followup.json")
