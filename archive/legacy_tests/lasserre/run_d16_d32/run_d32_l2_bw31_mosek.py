#!/usr/bin/env python
"""MOSEK Lasserre L2 at d=32 bw=31: rigor-oriented run.

Target: certify val(32) >= 1.285. Uses constraint-generation over a
clique-decomposed PSD (Waki-style correlative sparsity, chordal
bandwidth = 31 → single clique of all 32 bins, no structural drop
vs full L2 at this bandwidth).

Why MOSEK, not SCS/ADMM: MOSEK is interior-point, delivering
primal-dual pairs with ~1e-8 KKT residual. Jansson safe-dual-bound
rounding (safe_certify) then loses only O(1e-6) in the rigor step,
vs O(1e-2) for first-order SCS. With target headroom 0.103 we want
the tighter solver.

Writes: data/d32_l2_bw31_mosek.log, data/d32_l2_bw31_mosek.json.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasserre_enhanced import solve_enhanced, val_d_known

D = 32
ORDER = 2
BW = 31
C_TARGET = 1.285

print(f"=== d=32 L2 bw=31 MOSEK run | target >= {C_TARGET} ===", flush=True)
print(f"val_d_known[32] = {val_d_known.get(32)}", flush=True)
print(f"UTC: {datetime.now(timezone.utc).isoformat()}", flush=True)

t0 = time.time()
r = solve_enhanced(
    d=D,
    c_target=C_TARGET,
    order=ORDER,
    psd_mode='sparse',
    sparse_bandwidth=BW,
    search_mode='bisect',
    add_upper_loc=True,
    max_cg_rounds=6,
    max_add_per_round=50,
    n_bisect=15,
    verbose=True,
)
elapsed = time.time() - t0

out = {
    'd': D,
    'order': ORDER,
    'bw': BW,
    'c_target': C_TARGET,
    'elapsed_s': elapsed,
    'result': r,
}

# Strip non-JSON types
def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    try:
        json.dumps(x)
        return x
    except TypeError:
        return str(x)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'data', 'd32_l2_bw31_mosek.json')
with open(out_path, 'w') as fh:
    json.dump(_clean(out), fh, indent=2)

print(f"\n=== DONE elapsed={elapsed:.1f}s ===", flush=True)
print(f"Result: {r.get('lb')}  (target {C_TARGET})", flush=True)
print(f"Wrote {out_path}", flush=True)
