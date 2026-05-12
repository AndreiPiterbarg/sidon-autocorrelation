"""Refine Blaschke zeros of hat g for the rescaled Boyer-Li witness.

For g a real step function on [-1/4,1/4] of width h = 1/1150 in 575 cells,
g*g has Fourier transform (hat g)^2. Hyp_R-style "Blaschke zeros" are zeros
of hat(g*g) in the upper half-plane (counted with multiplicity 2). They
coincide with zeros of hat g.

We use a coarser secondary refinement via Muller's method on a complex grid.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
COEFF_PATH = ROOT / "delsarte_dual" / "restricted_holder" / "coeffBL.txt"


def load_coeffs() -> np.ndarray:
    raw = COEFF_PATH.read_text().strip().lstrip("{").rstrip("}")
    nums = [int(x) for x in re.split(r"[,\s]+", raw) if x]
    return np.asarray(nums, dtype=np.int64)


v = load_coeffs()
N = len(v)
S = int(v.sum())
n_arr = np.arange(N, dtype=np.float64)
n_centered = n_arr - (N - 1) / 2.0


def hatg(z: complex) -> complex:
    if z == 0:
        return 1.0 + 0.0j
    alpha = 2.0 * np.pi * z / 1150.0
    sinc_half = np.sin(alpha / 2.0) / (alpha / 2.0)
    phases = np.exp(-1j * alpha * n_centered)
    return (1.0 / S) * sinc_half * np.sum(v.astype(np.float64) * phases)


def muller(z0: complex, z1: complex, z2: complex, max_iter: int = 60, tol: float = 1e-12):
    f0, f1, f2 = hatg(z0), hatg(z1), hatg(z2)
    for _ in range(max_iter):
        h0 = z1 - z0
        h1 = z2 - z1
        if h0 == 0 or h1 == 0:
            return None
        d0 = (f1 - f0) / h0
        d1 = (f2 - f1) / h1
        if h1 + h0 == 0:
            return None
        a = (d1 - d0) / (h1 + h0)
        b = a * h1 + d1
        D = (b * b - 4.0 * a * f2) ** 0.5
        if abs(b - D) > abs(b + D):
            denom = b - D
        else:
            denom = b + D
        if denom == 0:
            return None
        z3 = z2 - 2.0 * f2 / denom
        if abs(z3 - z2) < tol:
            return z3
        z0, z1, z2 = z1, z2, z3
        f0, f1, f2 = f1, f2, hatg(z3)
    return z2


def main() -> None:
    # Full UHP search box: |xi| <= 50, 0 < eta <= 8
    XI_LO, XI_HI = -50.0, 50.0
    ETA_LO, ETA_HI = 0.02, 8.0
    nx, ny = 801, 161
    xs = np.linspace(XI_LO, XI_HI, nx)
    ys = np.linspace(ETA_LO, ETA_HI, ny)

    # Compute |hat g| on grid (vectorized)
    print(f"# scanning {nx} x {ny} grid in UHP...")
    grid_abs = np.empty((ny, nx))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            grid_abs[i, j] = abs(hatg(complex(x, y)))

    # Find seed minima
    seeds = []
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            v0 = grid_abs[i, j]
            if v0 > 0.01:
                continue
            if (
                v0 < grid_abs[i - 1, j]
                and v0 < grid_abs[i + 1, j]
                and v0 < grid_abs[i, j - 1]
                and v0 < grid_abs[i, j + 1]
            ):
                seeds.append((v0, complex(xs[j], ys[i])))
    seeds.sort(key=lambda t: t[0])
    print(f"# seed minima: {len(seeds)}")

    # Refine each seed
    refined = []
    for v0, z in seeds:
        z_ref = muller(z + 0.01, z - 0.005j, z, tol=1e-14)
        if z_ref is None:
            continue
        if abs(z_ref.imag) < 0.001:
            continue
        if abs(hatg(z_ref)) < 1e-8:
            # de-dup (within 0.05)
            dup = False
            for zz in refined:
                if abs(zz - z_ref) < 0.05:
                    dup = True
                    break
            if not dup:
                refined.append(z_ref)

    refined.sort(key=lambda z: (z.real, z.imag))
    print(f"# refined unique zeros in UHP (eta>0.001, |hat g|<1e-8): {len(refined)}")
    for z in refined:
        print(f"   z = {z.real:+10.5f} + i {z.imag:9.5f}    |hat g| = {abs(hatg(z)):.2e}")

    # Statistics
    if refined:
        etas = [z.imag for z in refined]
        xis = [z.real for z in refined]
        print(f"\nzeros count: {len(refined)}")
        print(f"  imag range: [{min(etas):.4f}, {max(etas):.4f}]")
        print(f"  |real| range: [{min(abs(x) for x in xis):.3f}, {max(abs(x) for x in xis):.3f}]")
        print(f"  mean imag: {np.mean(etas):.4f}")


if __name__ == "__main__":
    main()
