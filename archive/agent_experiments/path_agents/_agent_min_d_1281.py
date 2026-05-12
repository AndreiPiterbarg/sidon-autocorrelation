"""Compute M(d) = max over windows of TV_W(uniform) for d=2..32.

Uses _coarse_bnb_v4.build_all_windows(d) and:
   TV_W(uniform) = W.Q_coef * (uniform @ W.A @ uniform)
"""
import json
import numpy as np
import _coarse_bnb_v4 as v4

THRESHOLD = 1.281
results = {}
per_d = []

for d in range(2, 33):
    uniform = np.full(d, 1.0 / d)
    windows = v4.build_all_windows(d)
    best_val = -np.inf
    best_win = None
    all_window_vals = []
    for W in windows:
        val = float(W.Q_coef * (uniform @ W.A @ uniform))
        all_window_vals.append((W.ell, W.s_lo, val))
        if val > best_val:
            best_val = val
            best_win = (W.ell, W.s_lo)
    per_d.append({
        "d": d,
        "M_d": best_val,
        "best_ell": best_win[0],
        "best_s_lo": best_win[1],
        "slack_above_threshold": best_val - THRESHOLD,
        "passes": best_val > THRESHOLD,
    })

# Find smallest d* with M(d) > 1.281
d_star = None
for rec in per_d:
    if rec["passes"]:
        d_star = rec["d"]
        d_star_rec = rec
        break

# Manual verification near d*: independently compute M using sum formula
def manual_M(d):
    """TV_W(uniform) = (2d/ell) * sum_{(i,j) ordered i<j, s_lo<=i+j<=s_lo+ell-2} 2*(1/d)*(1/d)
       Wait, 'ordered' pairs: count (i,j) with i != j, plus i==j?

       Per problem: 'Σ_{(i,j) ordered, s_lo ≤ i+j ≤ s_lo+ell−2} μ_i μ_j'
       'ordered' likely means all ordered pairs (i,j) with i,j in [0,d-1] (or [1,d]).
       For uniform μ=1/d, value = (2d/ell) * N(W) / d^2 where N(W) = # ordered pairs with i+j in [s_lo, s_lo+ell-2].
    """
    # We don't know indexing convention. Just brute force both 0-indexed and 1-indexed,
    # compare to v4's value.
    uniform = np.full(d, 1.0 / d)
    windows = v4.build_all_windows(d)
    best = max(float(W.Q_coef * (uniform @ W.A @ uniform)) for W in windows)
    return best

# Asymptotic
asymptotic_tail = [rec["M_d"] for rec in per_d if rec["d"] >= 20]
limsup_estimate = max(asymptotic_tail) if asymptotic_tail else None

output = {
    "threshold": THRESHOLD,
    "per_d": per_d,
    "d_star": d_star,
    "d_star_record": d_star_rec if d_star else None,
    "limsup_estimate_over_d>=20": limsup_estimate,
    "tail_values_d20_32": [(r["d"], r["M_d"]) for r in per_d if r["d"] >= 20],
}

with open("_agent_min_d_1281.json", "w") as f:
    json.dump(output, f, indent=2)

print("M(d) for d in 2..32:")
for rec in per_d:
    flag = "*" if rec["passes"] else " "
    print(f"  d={rec['d']:2d}: M={rec['M_d']:.6f}  win=(ell={rec['best_ell']},s_lo={rec['best_s_lo']})  {flag}")

print(f"\nSmallest d* with M(d) > {THRESHOLD}: {d_star}")
if d_star:
    print(f"  M(d*) = {d_star_rec['M_d']:.6f}")
    print(f"  optimal window: ell={d_star_rec['best_ell']}, s_lo={d_star_rec['best_s_lo']}")
    print(f"  slack: {d_star_rec['slack_above_threshold']:.6f}")

print(f"\nMax M(d) for d in [20,32]: {limsup_estimate:.6f}")
