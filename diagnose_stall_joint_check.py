"""Quick: of the harvested STUCK boxes, how many would close under
joint-face LP on their winning window? (Float LP, not int rigor)."""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interval_bnb.bound_eval import bound_mccormick_joint_face_lp
from interval_bnb.windows import build_windows


def main():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "stall_diag_d10_t1208.json")
    with open(p) as f:
        data = json.load(f)
    target = 1.208
    d = 10
    windows = build_windows(d)
    win_index = {(w.ell, w.s_lo): i for i, w in enumerate(windows)}

    n_total = len(data["stuck"])
    n_joint_closes = 0
    n_top3_closes = 0
    delta_joint = []
    delta_top3 = []
    for r in data["stuck"]:
        lo = np.array(r["lo"])
        hi = np.array(r["hi"])
        wi = win_index[(r["win_ell"], r["win_s"])]
        w = windows[wi]
        lb = bound_mccormick_joint_face_lp(lo, hi, w, w.scale)
        delta_joint.append(lb - r["lb_fast"])
        if lb >= target:
            n_joint_closes += 1
        # Try top-3 alternate windows by LB.
        best_top = lb
        for ell, s, _ in r["top3_lb"]:
            if (ell, s) == (r["win_ell"], r["win_s"]):
                continue
            wi2 = win_index.get((ell, s))
            if wi2 is None:
                continue
            lb2 = bound_mccormick_joint_face_lp(lo, hi, windows[wi2], windows[wi2].scale)
            if lb2 > best_top:
                best_top = lb2
        delta_top3.append(best_top - r["lb_fast"])
        if best_top >= target:
            n_top3_closes += 1
    delta_joint = np.array(delta_joint)
    delta_top3 = np.array(delta_top3)
    print(f"Total stuck boxes: {n_total}")
    print(f"Joint-face LP on winning window closes: {n_joint_closes}/{n_total} "
          f"({100*n_joint_closes/n_total:.1f}%)")
    print(f"  delta_joint stats: median={np.median(delta_joint):.5f} "
          f"max={delta_joint.max():.5f} mean={delta_joint.mean():.5f}")
    print(f"Joint-face LP top-3 windows closes: {n_top3_closes}/{n_total} "
          f"({100*n_top3_closes/n_total:.1f}%)")
    print(f"  delta_top3 stats: median={np.median(delta_top3):.5f} "
          f"max={delta_top3.max():.5f}")


if __name__ == "__main__":
    main()
