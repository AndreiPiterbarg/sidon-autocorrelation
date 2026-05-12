"""d=4 anchor / soundness run (gates G1 and G2).

val(4) ~= 1.10233:
  * Target 1.10 must SUCCEED in < 10s (G1).
  * Target 1.11 must FAIL (G2, soundness).
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from interval_bnb.bnb import branch_and_bound  # noqa: E402


def main():
    for target, expect_ok in [("1.10", True), ("1.102", True),
                              ("1.10233", True), ("1.11", False)]:
        t0 = time.time()
        res = branch_and_bound(
            d=4, target_c=target, verbose=False,
            time_budget_s=30.0, method="combined",
            max_nodes=500_000, min_box_width=1e-10,
        )
        ok = "OK" if res.success == expect_ok else "BUG"
        print(f"[d=4 target={target}] success={res.success} "
              f"(expected={expect_ok}) {ok}  "
              f"t={time.time()-t0:.2f}s nodes={res.stats.nodes_processed}")


if __name__ == "__main__":
    main()
