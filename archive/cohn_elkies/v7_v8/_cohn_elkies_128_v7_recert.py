"""Recertify the best v7 4-scale config at XI_MAX = 1e5 with full arb pipeline,
then update _cohn_elkies_128_v7_results.json with the result and finalize.
"""

from __future__ import annotations

import json
import sys
import time
from fractions import Fraction
from pathlib import Path

_HERE = Path(__file__).parent

import importlib.util
spec = importlib.util.spec_from_file_location("_cohn_elkies_128_v7",
                                              _HERE / "_cohn_elkies_128_v7.py")
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)

DELTA1_Q = v7.DELTA1_Q
_Q = v7._Q
_norm_lambdas = v7._norm_lambdas
certify_with_reopt_NG = v7.certify_with_reopt_NG

# Best config from the sweep
d_best = [0.138, 0.055, 0.030, 0.015]
l_best = [0.85, 0.10, 0.03, 0.02]
n_g_best = 500

print("=" * 78)
print("v7 final recertification @ XI_MAX = 1e5")
print(f"  deltas  = {d_best}")
print(f"  lambdas = {l_best}")
print(f"  N_G     = {n_g_best}")
print("=" * 78)

d_q_best = [DELTA1_Q] + [_Q(d, 10**6) for d in d_best[1:]]
l_q_best = _norm_lambdas(list(l_best))

XI_MAX = 100000
t0 = time.time()
r = certify_with_reopt_NG(d_q_best, l_q_best, xi_max=XI_MAX,
                          n_modes=n_g_best, verbose=True)
el = time.time() - t0

r["tag"] = (f"4sc NG={n_g_best} d=({','.join(f'{d:.3f}' for d in d_best)}) "
            f"l=({','.join(f'{l:.2f}' for l in l_best)}) @xi={XI_MAX}")
r["N_G"] = n_g_best
r["source"] = "rerun_hires"
print(f"\nTotal time: {el:.1f}s")
print(f"M_cert_lower @ XI_MAX={XI_MAX}: {r['M_cert_lower']:.8f}")

# Merge into results JSON
res_path = _HERE / "_cohn_elkies_128_v7_results.json"
if res_path.exists():
    with open(res_path) as f:
        out = json.load(f)
else:
    out = {"runs": []}

out["runs"].append(r)
# Best overall
best = max(out["runs"], key=lambda x: x["M_cert_lower"])
out["best_overall"] = best
out["status"] = "final"
out["configuration"] = {
    "delta_1": 0.138, "u": 0.638, "prec_bits": 256,
    "XI_MAX_sweep": 10000, "XI_MAX_best": 100000,
    "N_G_values": [200, 500],
}
out["baselines"] = {
    "v4_3sc_N200": 1.29216,
    "v5_4sc_N119": 1.29136,
    "cs17_paper": 1.28020,
    "MV_numerical": 1.27428,
    "MV_arcsine_empirical_ceiling": 1.2924,
}
out["note"] = (
    "v7 finalization: 4-scale rows 1-41 are parsed from the partial log of the "
    "original (interrupted) v7 sweep (rigorous arb pipeline at XI_MAX=10000). "
    "5-scale and 6-scale rows + best-config XI_MAX=1e5 recert are reruns."
)

with open(res_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\nWrote {res_path}")
print(f"BEST overall: {best['tag']}  M_cert = {best['M_cert_lower']:.8f}")
print(f"  vs 3-scale v4 ref (1.29216): {best['M_cert_lower'] - 1.29216:+.6f}")
print(f"  vs CS17 paper (1.28020):     {best['M_cert_lower'] - 1.28020:+.6f}")
