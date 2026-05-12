"""Random nonneg step-function search:
For each draw of v in R_+^K, compute f = sum v_i 1_{cell i} (cells partition [-1/4,1/4]),
normalize to int=1, compute (M, c_*). Filter on M <= 1.51 (Hyp_R restricted class)
and report max c_*.

If max c_* found is far below log 16/pi = 0.8825, this is strong empirical
evidence that Hyp_R holds.

We also include a directed local-search: starting from the worst random sample,
perturb cells and seek to maximize c_* while keeping M <= 1.51.
"""
from __future__ import annotations
import math, sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np

C_STAR = math.log(16) / math.pi

K_CELLS = 50              # number of cells partitioning [-1/4,1/4]
N_RANDOM = 4000
SEED = 42
M_MAX = 1.51              # restricted-class bound from existing Hyp_R analysis

rng = np.random.default_rng(SEED)

# x-grid (subgrid per cell for convolution accuracy)
SUBPIX = 8
GRID_N = K_CELLS * SUBPIX
x = np.linspace(-0.25, 0.25, GRID_N+1)
dx = x[1] - x[0]

def f_from_cells(v: np.ndarray) -> np.ndarray:
    """expand cell heights v (length K_CELLS) into per-pixel array."""
    # each cell -> SUBPIX pixels
    f = np.repeat(v, SUBPIX)
    # pad to GRID_N+1
    if len(f) < len(x):
        f = np.concatenate([f, [f[-1]]])
    return f

def quantities(v: np.ndarray):
    f = f_from_cells(v)
    s = float(np.sum(f) * dx)
    if s <= 0:
        return None
    f = f / s
    g = np.convolve(f, f) * dx
    Minf = float(np.max(g))
    L1 = float(np.sum(g) * dx)
    L2 = float(np.sum(g*g) * dx)
    c = L2 / (Minf * L1)
    return Minf, L1, L2, c, f

def main():
    print("=" * 90)
    print(f"Random nonneg step-function search ({N_RANDOM} draws, K_CELLS={K_CELLS})")
    print(f"Filter: M = ||f*f||_inf <= {M_MAX:.4f}")
    print(f"Target: c_* = ||f*f||_2^2/(M*1) approaching log(16)/pi = {C_STAR:.6f}")
    print("=" * 90)

    best = (-1.0, None, None)  # c_*, v, M
    accepted = 0
    Ms = []
    cs = []

    for trial in range(N_RANDOM):
        # different scale distributions
        if trial % 5 == 0:
            v = rng.exponential(scale=1.0, size=K_CELLS)
        elif trial % 5 == 1:
            v = rng.uniform(0, 1.0, size=K_CELLS)
        elif trial % 5 == 2:
            v = rng.uniform(0, 1.0, size=K_CELLS) ** 4  # heavy on small
        elif trial % 5 == 3:
            # support a random subset
            mask = rng.uniform(0, 1, size=K_CELLS) < 0.4
            v = mask.astype(float) * rng.uniform(0.1, 2.0, size=K_CELLS)
        else:
            # peaked at random location with random width
            mu = rng.integers(0, K_CELLS)
            sig = rng.integers(2, K_CELLS//2 + 1)
            v = np.exp(-((np.arange(K_CELLS) - mu)**2) / (2*sig**2))

        if np.sum(v) <= 0:
            continue
        q = quantities(v)
        if q is None:
            continue
        Minf, L1, L2, c, f = q
        if Minf <= M_MAX:
            accepted += 1
            Ms.append(Minf); cs.append(c)
            if c > best[0]:
                best = (c, v.copy(), Minf)

    print(f"\nAccepted (M <= {M_MAX}):  {accepted} / {N_RANDOM}")
    if accepted > 0:
        print(f"  M range: [{min(Ms):.4f}, {max(Ms):.4f}]")
        print(f"  c_* range: [{min(cs):.5f}, {max(cs):.5f}]")
        print(f"  BEST c_* = {best[0]:.6f}  at M = {best[2]:.4f}")
        print(f"  margin to log(16)/pi: {best[0] - C_STAR:+.5f}")
        if best[0] > C_STAR:
            print("  *** VIOLATOR FOUND ***")

    # Hill-climb from the best
    if best[1] is not None:
        v = best[1].copy()
        c_best = best[0]
        for step in range(2000):
            i = rng.integers(0, K_CELLS)
            delta = rng.normal(0, 0.05)
            v_new = v.copy()
            v_new[i] = max(0.0, v_new[i] + delta)
            q = quantities(v_new)
            if q is None: continue
            Minf, L1, L2, c, f = q
            if Minf <= M_MAX and c > c_best:
                v = v_new
                c_best = c
        print(f"\nAfter 2000 random-coord hill-climb steps from random-best:")
        Minf, L1, L2, c, f = quantities(v)
        print(f"  c_* = {c_best:.6f}  M = {Minf:.4f}  margin = {c_best - C_STAR:+.5f}")
        if c_best > C_STAR:
            print("  *** POST-CLIMB VIOLATOR ***")

    # also test: among M <= 1.378
    print("\n-- Stricter Hyp_R: M <= 1.378 --")
    accepted2 = [(M, c) for (M, c) in zip(Ms, cs) if M <= 1.378]
    if accepted2:
        print(f"  count = {len(accepted2)}, max c_* = {max(c for M,c in accepted2):.6f}")
    else:
        print("  count = 0 (M <= 1.378 is hard to hit randomly with K=50 cells)")

if __name__ == "__main__":
    main()
