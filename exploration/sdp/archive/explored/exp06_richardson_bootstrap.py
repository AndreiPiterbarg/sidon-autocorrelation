"""Experiment 6: Improved Richardson extrapolation with bootstrap CIs.

Use all known V(P) bounds (both UB and LB sequences) with:
1. Multiple model families (polynomial in 1/P, log corrections, etc.)
2. Bootstrap confidence intervals
3. Cross-validation model selection
4. Bayesian model averaging
"""
import numpy as np
from scipy.optimize import curve_fit, minimize
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# All known bounds
ub_data = {
    2: 1.777778, 3: 1.706667, 4: 1.644465, 5: 1.633817,
    6: 1.600883, 7: 1.591746, 8: 1.580150, 9: 1.578073,
    10: 1.566436, 11: 1.562873, 12: 1.559773, 13: 1.559581,
    14: 1.552608, 15: 1.550623,
}

lb_data = {
    2: 1.777778, 3: 1.706667, 4: 1.644465, 5: 1.632653,
    6: 1.585612, 7: 1.581746, 8: 1.548319, 9: 1.545626,
    10: 1.524823, 11: 1.520012, 12: 1.507730, 13: 1.503642,
    14: 1.493577, 15: 1.490516, 16: 1.483826, 17: 1.481782,
    18: 1.475327,
}


# Model families
def model_1overP(P, c, a):
    return c + a / P

def model_1overP2(P, c, a, b):
    return c + a / P + b / P**2

def model_1overP3(P, c, a, b, d):
    return c + a / P + b / P**2 + d / P**3

def model_log(P, c, a, b):
    return c + a / P + b * np.log(P) / P**2

def model_sqrt(P, c, a, b):
    return c + a / np.sqrt(P) + b / P

def model_power(P, c, a, alpha):
    return c + a / P**alpha

MODELS = [
    ('1/P',       model_1overP,  [1.5, 1.0],       2),
    ('1/P+1/P²',  model_1overP2, [1.5, 1.0, 1.0],  3),
    ('1/P+1/P³',  model_1overP3, [1.5, 1.0, 1.0, 1.0], 4),
    ('1/P+logP/P²', model_log,  [1.5, 1.0, 1.0],  3),
    ('a/sqrt(P)+b/P', model_sqrt, [1.5, 1.0, 1.0], 3),
    ('a/P^alpha', model_power,  [1.5, 1.0, 0.7],   3),
]


def fit_and_evaluate(data, P_min, models=MODELS):
    """Fit all models and return results."""
    Ps = np.array([p for p in sorted(data) if p >= P_min], dtype=float)
    vals = np.array([data[int(p)] for p in Ps])
    n = len(Ps)
    results = []
    for name, model, p0, n_params in models:
        if n <= n_params:
            continue
        try:
            popt, pcov = curve_fit(model, Ps, vals, p0=p0, maxfev=50000)
            resid = vals - model(Ps, *popt)
            rmse = np.sqrt(np.mean(resid**2))
            aic = n * np.log(np.mean(resid**2)) + 2 * n_params
            bic = n * np.log(np.mean(resid**2)) + n_params * np.log(n)
            perr = np.sqrt(np.diag(pcov))
            results.append({
                'name': name, 'c_inf': popt[0], 'err': perr[0],
                'rmse': rmse, 'aic': aic, 'bic': bic, 'params': popt,
                'n_params': n_params, 'model': model
            })
        except Exception:
            pass
    return results


def bootstrap_ci(data, P_min, n_boot=2000, models=MODELS):
    """Bootstrap confidence intervals for C_inf."""
    Ps = np.array([p for p in sorted(data) if p >= P_min], dtype=float)
    vals = np.array([data[int(p)] for p in Ps])
    n = len(Ps)

    all_estimates = {name: [] for name, _, _, _ in models}
    rng = np.random.RandomState(42)

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        Ps_b = Ps[idx]
        vals_b = vals[idx]
        for name, model, p0, n_params in models:
            if n <= n_params:
                continue
            try:
                popt, _ = curve_fit(model, Ps_b, vals_b, p0=p0, maxfev=10000)
                if 1.0 < popt[0] < 2.0:  # sanity check
                    all_estimates[name].append(popt[0])
            except Exception:
                pass

    ci_results = {}
    for name, ests in all_estimates.items():
        if len(ests) > 100:
            ests = np.array(ests)
            ci_results[name] = {
                'mean': np.mean(ests),
                'median': np.median(ests),
                'ci_95': (np.percentile(ests, 2.5), np.percentile(ests, 97.5)),
                'ci_80': (np.percentile(ests, 10), np.percentile(ests, 90)),
                'std': np.std(ests),
                'n_valid': len(ests),
            }
    return ci_results


def leave_one_out_cv(data, P_min, models=MODELS):
    """Leave-one-out cross-validation for model selection."""
    Ps = np.array([p for p in sorted(data) if p >= P_min], dtype=float)
    vals = np.array([data[int(p)] for p in Ps])
    n = len(Ps)

    cv_scores = {}
    for name, model, p0, n_params in models:
        if n <= n_params + 1:
            continue
        errors = []
        for i in range(n):
            Ps_train = np.delete(Ps, i)
            vals_train = np.delete(vals, i)
            try:
                popt, _ = curve_fit(model, Ps_train, vals_train, p0=p0, maxfev=10000)
                pred = model(Ps[i], *popt)
                errors.append((pred - vals[i])**2)
            except Exception:
                errors.append(1.0)  # penalty
        cv_scores[name] = np.mean(errors)
    return cv_scores


print("=" * 80)
print("EXP 6: Improved Richardson Extrapolation with Bootstrap CIs")
print("=" * 80)

for label, data in [("Upper Bounds (primal)", ub_data), ("Lower Bounds (Lasserre)", lb_data)]:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    for P_min in [5, 7, 10]:
        Ps_avail = [p for p in sorted(data) if p >= P_min]
        if len(Ps_avail) < 4:
            continue

        print(f"\n  --- P >= {P_min} ({len(Ps_avail)} data points) ---")

        # Fit models
        results = fit_and_evaluate(data, P_min)
        results.sort(key=lambda r: r['bic'])
        print(f"  {'Model':>18} | {'C_inf':>8} | {'±err':>8} | {'RMSE':>10} | {'BIC':>8}")
        print(f"  {'-'*60}")
        for r in results:
            print(f"  {r['name']:>18} | {r['c_inf']:>8.5f} | {r['err']:>8.5f} | {r['rmse']:>10.7f} | {r['bic']:>8.2f}")

        # Bootstrap CIs
        ci = bootstrap_ci(data, P_min)
        print(f"\n  Bootstrap CIs (2000 resamples):")
        for name, c in sorted(ci.items(), key=lambda x: x[1].get('std', 99)):
            print(f"    {name:>18}: {c['median']:.5f} "
                  f"[{c['ci_95'][0]:.5f}, {c['ci_95'][1]:.5f}] (95% CI), "
                  f"std={c['std']:.5f}, n={c['n_valid']}")

        # LOO-CV
        cv = leave_one_out_cv(data, P_min)
        if cv:
            best_cv = min(cv, key=cv.get)
            print(f"\n  LOO-CV scores (lower = better):")
            for name, score in sorted(cv.items(), key=lambda x: x[1]):
                marker = " <-- BEST" if name == best_cv else ""
                print(f"    {name:>18}: {score:.2e}{marker}")


# Bayesian model averaging
print(f"\n{'='*60}")
print(f"  Bayesian Model Averaging")
print(f"{'='*60}")

for label, data in [("UB", ub_data), ("LB", lb_data)]:
    for P_min in [5, 8]:
        results = fit_and_evaluate(data, P_min)
        if not results:
            continue
        # BIC-based weights
        bics = np.array([r['bic'] for r in results])
        delta_bic = bics - np.min(bics)
        weights = np.exp(-delta_bic / 2)
        weights /= weights.sum()

        c_infs = np.array([r['c_inf'] for r in results])
        bma_mean = np.sum(weights * c_infs)
        bma_var = np.sum(weights * (c_infs - bma_mean)**2) + np.sum(weights * np.array([r['err']**2 for r in results]))
        bma_std = np.sqrt(bma_var)

        print(f"\n  {label} (P>={P_min}): BMA estimate = {bma_mean:.5f} ± {bma_std:.5f}")
        for r, w in zip(results, weights):
            print(f"    {r['name']:>18}: w={w:.3f}, c_inf={r['c_inf']:.5f}")


# Final sandwich
print(f"\n{'='*60}")
print(f"  FINAL SANDWICH ESTIMATES")
print(f"{'='*60}")
print(f"  Known rigorous bounds: C_1a in [1.2802, 1.5029]")
print()

for P_min in [5, 7, 10]:
    ub_ci = bootstrap_ci(ub_data, P_min)
    lb_ci = bootstrap_ci(lb_data, P_min)

    # Use BIC-best model for each
    ub_res = fit_and_evaluate(ub_data, P_min)
    lb_res = fit_and_evaluate(lb_data, P_min)
    if ub_res and lb_res:
        ub_best = min(ub_res, key=lambda r: r['bic'])
        lb_best = min(lb_res, key=lambda r: r['bic'])

        # Conservative: use widest CIs
        ub_lo = min(c['ci_95'][0] for c in ub_ci.values()) if ub_ci else ub_best['c_inf']
        ub_hi = max(c['ci_95'][1] for c in ub_ci.values()) if ub_ci else ub_best['c_inf']
        lb_lo = min(c['ci_95'][0] for c in lb_ci.values()) if lb_ci else lb_best['c_inf']
        lb_hi = max(c['ci_95'][1] for c in lb_ci.values()) if lb_ci else lb_best['c_inf']

        mid_ub = ub_best['c_inf']
        mid_lb = lb_best['c_inf']
        midpoint = (mid_ub + mid_lb) / 2

        print(f"  P >= {P_min}:")
        print(f"    UB extrapolation: {mid_ub:.5f} (model: {ub_best['name']}) [{ub_lo:.5f}, {ub_hi:.5f}]")
        print(f"    LB extrapolation: {mid_lb:.5f} (model: {lb_best['name']}) [{lb_lo:.5f}, {lb_hi:.5f}]")
        print(f"    Sandwich midpoint: {midpoint:.5f}")
        print(f"    Conservative CI: [{lb_lo:.5f}, {ub_hi:.5f}]")
        print()
