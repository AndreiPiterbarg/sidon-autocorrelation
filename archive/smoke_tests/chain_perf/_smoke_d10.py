"""Smoke test: parallel BnB at d=10 t=1.215.

Exercises the boundary-concentrated split heuristic by running the
parallel driver (which calls _worker_main, where the heuristic lives).
"""
import os
import sys
import time
from fractions import Fraction

sys.path.insert(0, ".")

from interval_bnb.parallel import parallel_branch_and_bound

t0 = time.time()
res = parallel_branch_and_bound(
    d=10,
    target_c=Fraction("1.215"),
    workers=2,
    init_split_depth=8,
    time_budget_s=120.0,
    verbose=False,
)
elapsed = time.time() - t0

success = res.get("success", False) if isinstance(res, dict) else getattr(res, "success", False)
print(
    "BOUNDARY_DEPTH=", os.environ.get("INTERVAL_BNB_BOUNDARY_SPLIT_DEPTH", "999"),
    " BOUNDARY_COUNT=", os.environ.get("INTERVAL_BNB_BOUNDARY_AXIS_COUNT", "default"),
)
print("SUCCESS=", success, " elapsed=%.1fs" % elapsed)
sys.exit(0 if success else 1)
