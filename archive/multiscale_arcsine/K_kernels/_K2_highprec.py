"""
High-precision computation of K_2 = integral_R |K_hat(xi)|^2 dxi
for 2-scale arcsine: K_hat(xi) = lam1*J0(pi*d1*xi)^2 + lam2*J0(pi*d2*xi)^2
where lam1=0.85, lam2=0.15, d1=0.138, d2=0.045.

|K_hat|^2 = lam1^2 J0(pi d1 xi)^4 + 2 lam1 lam2 J0(pi d1 xi)^2 J0(pi d2 xi)^2
            + lam2^2 J0(pi d2 xi)^4.

Integrating over R (i.e. 2 * integral over [0,inf)).

Known closed forms (Bessel moments):
  c_4 := int_0^inf J0(t)^4 dt = (gamma(1/4)^4) / (4 pi^3) ??? Let's compute it directly.
  Actually: int_0^inf J0(t)^4 dt = K(1/2)/pi by Bailey-Borwein-Broadhurst (Glasser/Montaldi).
  Numerically c_4 = 0.902729...
  By scaling: int_0^inf J0(a t)^4 dt = c_4 / a.

Cross integral: int_0^inf J0(a t)^2 J0(b t)^2 dt, a != b.
This has a hypergeometric form. Watson 13.46, also BBBG 2008.
We compute it by mpmath.quad with high precision and oscillatory subdivision.
"""
import mpmath as mp

mp.mp.dps = 50  # 50 decimal digits

pi = mp.pi
J0 = lambda x: mp.besselj(0, x)

lam1 = mp.mpf('0.85')
lam2 = mp.mpf('0.15')
d1   = mp.mpf('0.138')
d2   = mp.mpf('0.045')

# ============================================================
# 1) High-accuracy single-scale integral: int_0^inf J0(t)^4 dt
# ============================================================
print("=" * 70)
print("Step 1: c4 = int_0^inf J0(t)^4 dt")
print("=" * 70)

# Use mpmath.quadosc which is designed for oscillatory integrands.
# J0(t)^4 oscillates with period roughly pi (since J0 has period 2pi but ^4 doubles freq).
# Period of J0(t)^4 oscillation ~ pi.
c4 = mp.quadosc(lambda t: J0(t)**4, [0, mp.inf], period=mp.pi)
print(f"c4 (quadosc, dps=50)             = {mp.nstr(c4, 25)}")

# Cross-check: known closed form c4 = gamma(1/4)^4 / (4 pi^3) ? Let me check numerically.
candidate = mp.gamma(mp.mpf(1)/4)**4 / (4 * pi**3)
print(f"gamma(1/4)^4 / (4 pi^3)           = {mp.nstr(candidate, 25)}")
# Bailey/Borwein result: int_0^inf J0(t)^4 dt does NOT equal gamma(1/4)^4/(4pi^3) directly.
# Actually I think it's K(1/2)/pi where K is complete elliptic of 1st kind.
K_half = mp.ellipk(mp.mpf('0.5'))  # mpmath uses parameter m, K(m)
print(f"ellipk(1/2)/pi (modulus k^2=1/2)  = {mp.nstr(K_half / pi, 25)}")
# This should match c4 (Glasser-Montaldi 1994).

# ============================================================
# 2) Same-scale integrals via scaling
# ============================================================
print()
print("=" * 70)
print("Step 2: same-scale ints via scaling, int J0(a t)^4 dt = c4 / a")
print("=" * 70)

I11 = c4 / (pi * d1)   # int_0^inf J0(pi d1 xi)^4 dxi, substitute t=pi d1 xi
I22 = c4 / (pi * d2)
print(f"int_0^inf J0(pi d1 xi)^4 dxi      = {mp.nstr(I11, 25)}")
print(f"int_0^inf J0(pi d2 xi)^4 dxi      = {mp.nstr(I22, 25)}")

# ============================================================
# 3) Cross integral: int_0^inf J0(pi d1 xi)^2 J0(pi d2 xi)^2 dxi
# ============================================================
print()
print("=" * 70)
print("Step 3: cross int_0^inf J0(pi d1 xi)^2 J0(pi d2 xi)^2 dxi")
print("=" * 70)

# Method A: mpmath quadosc with period = min(pi/(pi d1), pi/(pi d2)) ~ 1/d1 ~ 7.25
# The integrand oscillates with two frequencies pi*d1 and pi*d2 (squared, so double).
# Effective period = 2 / d1 (the slower oscillation is squared, period ~ 1/d1).
# Slower scale: d2 = 0.045, so longest period ~ 1/0.045 = 22.2.
period_long = mp.mpf(1) / d2  # most conservative
period_short = mp.mpf(1) / d1

# Try multiple periods
print("Attempting quadosc with various period guesses...")
mp.mp.dps = 50
results = []
for p_choice, label in [(period_long, "1/d2=22.22"), (period_short, "1/d1=7.25"),
                         ((period_long + period_short) / 2, "avg"),
                         (mp.pi / (pi * d1), "pi/(pi d1)"),
                         (mp.pi / (pi * d2), "pi/(pi d2)")]:
    try:
        val = mp.quadosc(lambda xi: J0(pi*d1*xi)**2 * J0(pi*d2*xi)**2,
                          [0, mp.inf], period=p_choice)
        results.append((label, val))
        print(f"  quadosc, period={label:20s}  -> {mp.nstr(val, 20)}")
    except Exception as e:
        print(f"  quadosc, period={label:20s}  FAILED: {e}")

# Method B: split [0, R] adaptive + tail Bessel asymptotic.
# For large xi, J0(z) ~ sqrt(2/(pi z)) cos(z - pi/4), so
# J0(pi d1 xi)^2 J0(pi d2 xi)^2 ~ (4 / (pi^2 d1 d2 xi^2)) cos^2(pi d1 xi - pi/4) cos^2(pi d2 xi - pi/4)
# decays like 1/xi^2 -- integrable.

print()
print("Method B: finite [0,R] + tail estimate")
R = mp.mpf('1000')
mp.mp.dps = 40
I_finite = mp.quad(lambda xi: J0(pi*d1*xi)**2 * J0(pi*d2*xi)**2,
                    [0, R], maxdegree=12)
print(f"int_0^{R} (cross)                = {mp.nstr(I_finite, 20)}")

# Tail bound: |J0(pi d1 xi)^2 J0(pi d2 xi)^2| <= (2/(pi^2 d1 d2 xi^2))^2 ... wait
# |J0(z)| <= 1 always, and for z > 1, |J0(z)| <= sqrt(2/(pi z)) (this is exact bound).
# So |J0(pi d1 xi)^2| <= 2/(pi^2 d1 xi)  for pi d1 xi > 1.
# Similarly for d2. Product: 4 / (pi^4 d1 d2 xi^2).
# int_R^inf 4 / (pi^4 d1 d2 xi^2) dxi = 4 / (pi^4 d1 d2 R).
tail_bound = 4 / (pi**4 * d1 * d2 * R)
print(f"Tail bound for xi > {R}            <= {mp.nstr(tail_bound, 10)}")

# Method C: very large finite cutoff
print()
print("Method C: extending cutoff R")
mp.mp.dps = 40
for R_try in [mp.mpf('500'), mp.mpf('2000'), mp.mpf('10000')]:
    try:
        val = mp.quad(lambda xi: J0(pi*d1*xi)**2 * J0(pi*d2*xi)**2,
                       [0, R_try], maxdegree=14)
        print(f"  R={float(R_try):>8.0f}  ->  {mp.nstr(val, 18)}")
    except Exception as e:
        print(f"  R={float(R_try):>8.0f}  FAILED: {e}")
