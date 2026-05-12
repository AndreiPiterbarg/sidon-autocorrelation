"""More candidate forms — including ones that approach 1 as f -> infinity."""
import numpy as np
from scipy.optimize import curve_fit

f_data = np.array([0.000, 0.017, 0.034, 0.085])
p_data = np.array([0.094, 0.389, 0.522, 0.626])


def sigmoid_to_1(f, c, s):
    """Sigmoid forced to plateau at 1.0 at f→∞, lower bound at 0."""
    return 1.0 / (1 + np.exp(-(f - c) / s))

def weibull(f, k, lam):
    """Weibull CDF: p = 1 - exp(-(f/lam)^k). Approaches 1 at infinity."""
    return 1 - np.exp(-(f / lam)**k)

def gompertz(f, eta, b):
    """Gompertz: p = exp(-eta*exp(-b*f)). Approaches 1."""
    return np.exp(-eta * np.exp(-b * f))

def hill_to_1(f, K_half, n):
    """Hill with plateau forced to 1.0 (3-param can plateau lower)."""
    return f**n / (K_half**n + f**n)

def hill_free(f, a, b, K_half, n):
    return a + (b - a) * f**n / (K_half**n + f**n)

def exp_sat_free(f, a, b, beta):
    return b - (b - a) * np.exp(-beta * f)


print("=" * 80)
print("FIXED-PLATEAU FITS (force p→1 as f→∞)")
print("=" * 80)

for name, func, p0, bounds in [
    ('sigmoid_to_1',   sigmoid_to_1,   [0.05, 0.05],         ([0, 0.001], [1, 1])),
    ('weibull',        weibull,        [1.0, 0.05],          ([0.1, 0.001], [10, 10])),
    ('gompertz',       gompertz,       [3.0, 30],            ([0.1, 0.1], [1000, 1000])),
    ('hill_to_1',      hill_to_1,      [0.05, 1.5],          ([0.001, 0.1], [10, 100])),
]:
    try:
        popt, _ = curve_fit(func, f_data, p_data, p0=p0, bounds=bounds, maxfev=20000)
        pred = func(f_data, *popt)
        rmse = np.sqrt(np.mean((p_data - pred)**2))
        # Predict at high K
        f_test = np.array([0.169, 0.338, 0.529, 1.0])  # K=160, 320, 500, 946
        p_test = func(f_test, *popt)
        print(f"\n{name}: params = {[f'{x:.4f}' for x in popt]}  RMSE={rmse:.4f}")
        print(f"  fit: {[f'{x:.3f}' for x in pred]}  vs data {[f'{x:.3f}' for x in p_data]}")
        print(f"  residuals: {[f'{x:+.4f}' for x in (p_data - pred)]}")
        print(f"  predict at K=160 (f=0.169): p={p_test[0]:.4f}")
        print(f"  predict at K=320 (f=0.338): p={p_test[1]:.4f}")
        print(f"  predict at K=500 (f=0.529): p={p_test[2]:.4f}")
        print(f"  predict at K=946 (f=1.000): p={p_test[3]:.4f}")
        # Solve for K to hit 0.998
        fs = np.linspace(0, 100, 1000000)
        ps = func(fs, *popt)
        idx = np.argmax(ps >= 0.998)
        if ps[idx] >= 0.998:
            f_req = fs[idx]
            print(f"  K required for p≥0.998: K={int(np.ceil(f_req*946))}  (f={f_req:.3f})")
        else:
            print(f"  K required for p≥0.998: NEVER REACHED (max p = {ps[-1]:.4f})")
    except Exception as e:
        print(f"{name}: failed: {e}")


print("\n" + "=" * 80)
print("FREE-PLATEAU FITS (let plateau be a free param)")
print("=" * 80)
for name, func, p0, bounds in [
    ('hill_free',  hill_free,    [0.09, 0.7, 0.05, 1.5],    ([0.05, 0.5, 0.001, 0.1], [0.15, 1.5, 1, 5])),
    ('exp_sat',    exp_sat_free, [0.09, 0.7, 30],            ([0.05, 0.5, 0.1], [0.15, 1.5, 200])),
]:
    try:
        popt, _ = curve_fit(func, f_data, p_data, p0=p0, bounds=bounds, maxfev=20000)
        pred = func(f_data, *popt)
        rmse = np.sqrt(np.mean((p_data - pred)**2))
        f_test = np.array([0.169, 0.338, 0.529, 1.0])
        p_test = func(f_test, *popt)
        print(f"\n{name}: params = {[f'{x:.4f}' for x in popt]}  RMSE={rmse:.4f}")
        print(f"  predict at K=160: p={p_test[0]:.4f}")
        print(f"  predict at K=320: p={p_test[1]:.4f}")
        print(f"  predict at K=500: p={p_test[2]:.4f}")
        print(f"  predict at K=946: p={p_test[3]:.4f}")
    except Exception as e:
        print(f"{name}: failed: {e}")


print("\n" + "=" * 80)
print("THE QUESTION:  Is the close-rate plateau really < 1?")
print("=" * 80)
print("""
With only 4 data points all at f ≤ 0.085, we can't distinguish:
  (A) curves that plateau LOW (~67%) — extrapolated predictions: K=full closes 67%
  (B) curves that plateau at 1.0 but rise slowly past f=0.1

Both fit the data nearly perfectly. The DIFFERENCE in K=full prediction:
  curve A says: K=full closes 67% (residual 33%)
  curve B says: K=full closes ~95-99% (residual 1-5%)

The curve shape past f=0.1 is the question. We have NO d=22 data above f=0.085.

BEST EVIDENCE: d=10 K-sweep data. At f=0.084: p=0.92. At f=0.126: p=1.0.
So at d=10, the curve continues to rise sharply past f=0.1 to reach 1.0.

If d=22 has the same shape (just rescaled), curve B is correct.
If d=22 has a structurally different (lower) plateau at this depth, curve A is correct.

ONE MORE DATA POINT AT K=160 OR K=320 WOULD DECIDE THIS.

Currently the cascade is collecting K=160 data — wait for it.
""")
