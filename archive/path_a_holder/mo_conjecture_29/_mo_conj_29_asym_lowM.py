"""Asymmetric search: M can be < 2 only for asymmetric f.
We try BL-shrink (perturb BL toward smaller M while tracking c_*),
and random asymmetric step functions with bias toward low M.
"""
from __future__ import annotations
import math, sys, re
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
from pathlib import Path
import numpy as np

C_STAR = math.log(16) / math.pi
ROOT = Path(__file__).resolve().parent
COEFF_PATH = ROOT / "delsarte_dual" / "restricted_holder" / "coeffBL.txt"

def load_bl():
    raw = COEFF_PATH.read_text().strip()
    inside = raw.strip().lstrip("{").rstrip("}")
    nums = [int(x) for x in re.split(r"[,\s]+", inside) if x]
    return np.asarray(nums, dtype=np.float64)

def evaluate_v(v, dx):
    """v is K step heights, total support has K cells."""
    s = float(np.sum(v) * dx)
    if s <= 0:
        return None
    v = v / s
    g = np.convolve(v, v) * dx
    Minf = float(np.max(g))
    L1 = float(np.sum(g) * dx)
    L2 = float(np.sum(g*g) * dx)
    return Minf, L1, L2, L2 / (Minf * L1)

def main():
    print(f"Threshold log(16)/pi = {C_STAR:.6f}")

    v_bl = load_bl()
    K = len(v_bl)
    # cell width on [-1/4,1/4]: 0.5/K
    dx = 0.5 / K
    r = evaluate_v(v_bl, dx);
    print(f"BL on its native grid (K={K} cells):  M={r[0]:.5f}  c_*={r[3]:.6f}  (margin {r[3]-C_STAR:+.4f})")

    # ---- 1. Hill-climb v: minimize M subject to keeping c_* high, OR
    #          maximize c_* subject to M <= 1.378.
    # We test the latter, starting from BL but with M-penalty.
    print("\nGradient-free climb: maximize c_* + lambda*max(0, M - M_max).")
    rng = np.random.default_rng(0)

    for M_max in [1.378, 1.51, 1.65]:
        v = v_bl.copy()
        v += rng.uniform(-1.0, 1.0, K)*1000  # perturb
        v = np.maximum(v, 0)
        best = None
        c_best = -1
        for step in range(8000):
            i = rng.integers(0, K)
            delta = rng.normal(0, 5000)
            v_new = v.copy()
            v_new[i] = max(0.0, v_new[i] + delta)
            r = evaluate_v(v_new, dx)
            if r is None: continue
            Minf, L1, L2, c = r
            # acceptance: stay below M_max, maximize c
            if Minf <= M_max:
                # accept if c improves OR we're not yet in feasible region
                if c > c_best or c_best < 0:
                    v = v_new; c_best = c
                    if c > C_STAR:
                        print(f"  step {step}: VIOLATOR M={Minf:.4f} c={c:.6f}")
        if c_best > 0:
            r = evaluate_v(v, dx)
            print(f"  M_max={M_max:.3f}:  best c_* = {c_best:.6f}  at M = {r[0]:.5f}  margin = {c_best-C_STAR:+.4f}")
        else:
            print(f"  M_max={M_max:.3f}:  no feasible found")

    # ---- 2. Sequential subset: BL scaled and clipped at top ----
    # Take BL, scale all cells, clip M down by reducing peak heights
    print("\nBL with cells near argmax clipped (try to reduce M while preserving structure):")
    v = v_bl.copy()
    L = np.convolve(v, v)
    peak_j = int(np.argmax(L))
    # cells contributing most to peak L_j are those n with v_n * v_{j-n} large
    contrib = v * v[peak_j - np.arange(K)] if peak_j >= K-1 else v[max(0,peak_j-K+1):min(K,peak_j+1)]
    # simpler: scale globally down
    for shrink in [0.99, 0.95, 0.90, 0.80, 0.50]:
        v2 = v_bl.copy()
        # uniform shrink of all coeffs preserves c_* (scale-invariant up to normalization)
        # so this won't help. Instead, shrink only top contributing cell.
        top_idx = np.argsort(v_bl)[-int(K*0.1):]
        v2[top_idx] *= shrink
        r = evaluate_v(v2, dx)
        if r:
            print(f"  shrink_top10%={shrink}:  M={r[0]:.5f}  c_*={r[3]:.6f}  margin={r[3]-C_STAR:+.4f}")

    # ---- 3. Boyer-Li shrunk: convex mix BL <-> 0 (cannot reduce M ratio).
    # The fundamental issue is that c_*(f) and M depend differently on rescaling.

if __name__ == "__main__":
    main()
