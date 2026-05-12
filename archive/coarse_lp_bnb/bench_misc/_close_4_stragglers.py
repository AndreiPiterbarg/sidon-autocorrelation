"""Close the 4 remaining open cells from d=8, S=16, c=1.25 with deeper BnB."""
import os, sys, time, logging
logging.getLogger('cvxpy').setLevel(logging.ERROR)
import numpy as np
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
import _coarse_bnb_v4 as v4

d, S, c_target = 8, 16, 1.25
windows = v4.build_all_windows(d)
v4.get_sdp_template(d)
v4.get_joint_template(d, 4)

stragglers = [
    [4, 1, 0, 0, 2, 2, 2, 5],
    [4, 1, 0, 0, 3, 1, 2, 5],
    [4, 1, 1, 0, 2, 2, 1, 5],
    [4, 1, 1, 0, 2, 2, 2, 4],
]

print(f"\n=== Closing {len(stragglers)} stragglers at d=8, S=16, c=1.25 ===", flush=True)
for c_list in stragglers:
    c = np.array(c_list, dtype=np.float64)
    print(f"\n  c={c_list}", flush=True)
    for md in (3, 6, 10, 15):
        t0 = time.time()
        r = v4.certify_composition(c, S, d, c_target,
                                      windows=windows, max_depth=md)
        elapsed = time.time() - t0
        print(f"    max_depth={md:>2}: cert={r.certified} tier={r.tier_used} "
              f"bound={r.bound:.4f} depth={r.depth_used} sub_cells={r.n_subcells} "
              f"[{elapsed:.1f}s]", flush=True)
        if r.certified:
            break
