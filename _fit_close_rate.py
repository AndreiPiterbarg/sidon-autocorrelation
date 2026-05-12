"""Fit candidate close-rate-vs-K-coverage curves to d=22 data and pick the best."""
import numpy as np
from scipy.optimize import curve_fit

# Direct close rate on LP-fails at d=22, computed via chain rule from
# orchestrator + current run:
#   K=0   (f=0.000): 9.4% direct (orchestrator iter 6: 39/416)
#   K=16  (f=0.017): 9.4% + 90.6% * 32.6% = 38.94%
#   K=32  (f=0.034): 38.94% + 61.06% * 21.7% = 52.19%
#   K=80  (f=0.085): 9.4% + 90.6% * 58.7% = 62.59% (current run)
# d=22, |W|=946

f_data = np.array([0.000, 0.017, 0.034, 0.085])
p_data = np.array([0.094, 0.389, 0.522, 0.626])

# Also d=10 data for comparison (separate fit)
f_d10 = np.array([0.000, 0.021, 0.042, 0.063, 0.084, 0.126, 0.168])
p_d10 = np.array([0.020, 0.680, 0.880, 0.940, 0.920, 1.000, 1.000])
# (K=16 for d=10 had 92% in original sweep, 94% in cross-val — use mean 0.93)
p_d10[4] = 0.93


def sigmoid(f, a, b, c, s):
    return a + (b - a) / (1 + np.exp(-(f - c) / s))

def exponential_saturation(f, a, b, beta):
    """p = b - (b-a)*exp(-beta*f). Approaches b as f->inf."""
    return b - (b - a) * np.exp(-beta * f)

def hill(f, a, b, K_half, n):
    return a + (b - a) * f**n / (K_half**n + f**n)

def power(f, a, b, beta):
    """p = a + (b-a) * (1 - (1/(1+f))**beta). Different saturation shape."""
    return a + (b - a) * (1 - 1.0 / (1 + f)**beta)


def fit_and_score(func, p0, bounds=None, name=''):
    try:
        if bounds is not None:
            popt, _ = curve_fit(func, f_data, p_data, p0=p0,
                                 bounds=bounds, maxfev=20000)
        else:
            popt, _ = curve_fit(func, f_data, p_data, p0=p0, maxfev=20000)
        pred = func(f_data, *popt)
        ss_res = np.sum((p_data - pred)**2)
        ss_tot = np.sum((p_data - p_data.mean())**2)
        r2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(ss_res / len(p_data))
        return popt, r2, rmse, pred
    except Exception as e:
        return None, None, None, str(e)


print("d=22 close-rate fits:")
print("=" * 80)

# Sigmoid
popt_s, r2_s, rmse_s, pred_s = fit_and_score(
    sigmoid, p0=[0.09, 1.0, 0.05, 0.05],
    bounds=([0.05, 0.5, -1, 0.001], [0.15, 1.5, 1, 1]),
    name='sigmoid')
print(f"\nSIGMOID  p = a + (b-a)/(1+exp(-(f-c)/s))")
print(f"  a={popt_s[0]:.4f} b={popt_s[1]:.4f} c={popt_s[2]:.4f} s={popt_s[3]:.4f}")
print(f"  R²={r2_s:.6f}  RMSE={rmse_s:.4f}")
print(f"  predictions: {[f'{x:.3f}' for x in pred_s]}")
print(f"  residuals:   {[f'{x:+.3f}' for x in (p_data - pred_s)]}")

# Exponential saturation
popt_e, r2_e, rmse_e, pred_e = fit_and_score(
    exponential_saturation, p0=[0.09, 1.0, 5.0],
    bounds=([0.05, 0.5, 0.1], [0.15, 1.5, 100]),
    name='exp')
print(f"\nEXPONENTIAL SAT  p = b - (b-a)*exp(-beta*f)")
print(f"  a={popt_e[0]:.4f} b={popt_e[1]:.4f} beta={popt_e[2]:.4f}")
print(f"  R²={r2_e:.6f}  RMSE={rmse_e:.4f}")
print(f"  predictions: {[f'{x:.3f}' for x in pred_e]}")
print(f"  residuals:   {[f'{x:+.3f}' for x in (p_data - pred_e)]}")

# Hill
popt_h, r2_h, rmse_h, pred_h = fit_and_score(
    hill, p0=[0.09, 1.0, 0.05, 1.5],
    bounds=([0.05, 0.5, 0.001, 0.5], [0.15, 1.5, 1, 5]),
    name='hill')
print(f"\nHILL  p = a + (b-a)*f^n/(K^n+f^n)")
print(f"  a={popt_h[0]:.4f} b={popt_h[1]:.4f} K_half={popt_h[2]:.4f} n={popt_h[3]:.4f}")
print(f"  R²={r2_h:.6f}  RMSE={rmse_h:.4f}")
print(f"  predictions: {[f'{x:.3f}' for x in pred_h]}")
print(f"  residuals:   {[f'{x:+.3f}' for x in (p_data - pred_h)]}")

# Power
popt_p, r2_p, rmse_p, pred_p = fit_and_score(
    power, p0=[0.09, 1.0, 5.0],
    bounds=([0.05, 0.5, 0.1], [0.15, 1.5, 50]),
    name='power')
print(f"\nPOWER  p = a + (b-a)*(1 - 1/(1+f)^beta)")
print(f"  a={popt_p[0]:.4f} b={popt_p[1]:.4f} beta={popt_p[2]:.4f}")
print(f"  R²={r2_p:.6f}  RMSE={rmse_p:.4f}")
print(f"  predictions: {[f'{x:.3f}' for x in pred_p]}")
print(f"  residuals:   {[f'{x:+.3f}' for x in (p_data - pred_p)]}")

# Choose best by R²
fits = [
    ('sigmoid', sigmoid, popt_s, r2_s),
    ('exp_sat', exponential_saturation, popt_e, r2_e),
    ('hill', hill, popt_h, r2_h),
    ('power', power, popt_p, r2_p),
]
fits.sort(key=lambda x: -x[3])
print("\n" + "=" * 80)
print(f"BEST FIT by R²: {fits[0][0]}  (R²={fits[0][3]:.6f})")
best_name, best_func, best_popt = fits[0][0], fits[0][1], fits[0][2]

# Solve: at what coverage f does p(f) = threshold?
print("\n" + "=" * 80)
print("CONVERGENCE THRESHOLDS (using best fit):")
print(f"  split_depth: S    required p=1-1/S    required f       required K (f*946)")
for d_split, S in [(9, 500), (10, 1000), (11, 2000), (12, 4000), (13, 8000)]:
    target_p = 1 - 1.0/S
    # Numerical solve: search f from 0 to 1
    fs = np.linspace(0.0, 1.0, 100000)
    ps = best_func(fs, *best_popt)
    idx = np.argmin(np.abs(ps - target_p))
    if ps[idx] < target_p:
        # Find first f where p exceeds
        idx2 = np.argmax(ps >= target_p)
        if ps[idx2] >= target_p:
            f_req = fs[idx2]
            K_req = int(np.ceil(f_req * 946))
            print(f"  split_depth={d_split:2d}, S={S:5d}: p≥{target_p:.5f} → "
                  f"f≥{f_req:.4f} → K≥{K_req}")
        else:
            print(f"  split_depth={d_split:2d}, S={S:5d}: p≥{target_p:.5f} → "
                  f"NEVER REACHED (asymptote = {best_popt[1]:.3f})")
    else:
        f_req = fs[idx]
        K_req = int(np.ceil(f_req * 946))
        print(f"  split_depth={d_split:2d}, S={S:5d}: p≥{target_p:.5f} → "
              f"f≥{f_req:.4f} → K≥{K_req}")

# Predict close rate at K=160, 320, 425, 500, 800, 946
print("\n" + "=" * 80)
print("PREDICTED CLOSE RATES at higher K (best fit):")
print(f"  K (f=K/946)   predicted p   residual at S=500   residual at S=4000")
for K in [80, 160, 200, 250, 300, 320, 400, 500, 700, 946]:
    f = K / 946
    p_pred = best_func(np.array([f]), *best_popt)[0]
    res_500 = max(0, 1 - p_pred) * 500
    res_4000 = max(0, 1 - p_pred) * 4000
    print(f"  K={K:>4} (f={f:.3f})   p={p_pred:.4f}     {res_500:7.1f}              {res_4000:7.1f}")
