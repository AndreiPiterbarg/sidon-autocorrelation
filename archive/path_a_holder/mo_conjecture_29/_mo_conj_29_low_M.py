"""Search for c_*(f) over LOW-M densities, including BL-witness contractions.

Strategy:
  1. Symmetric mixtures of indicator+arcsine: vary mixing parameter.
     We try to push down M; what is the achievable c_* range with M <= 1.378?
  2. BL-witness contraction: f_t := (1-t)*BL + t*uniform.
     Track c_*(f_t) vs M(f_t) as t -> 1, hoping for low-M sample with high c_*.
  3. Smoothed BL by convolving with narrow tent: reduces M, computes c_*.

This is a *forensic* probe for whether ANY low-M f attains c_* > 0.85.
"""
from __future__ import annotations
import math, re, sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
from pathlib import Path
import numpy as np

C_STAR = math.log(16) / math.pi
ROOT = Path(__file__).resolve().parent
COEFF_PATH = ROOT / "delsarte_dual" / "restricted_holder" / "coeffBL.txt"

def load_bl_coeffs():
    raw = COEFF_PATH.read_text().strip()
    inside = raw.strip().lstrip("{").rstrip("}")
    nums = [int(x) for x in re.split(r"[,\s]+", inside) if x]
    return np.asarray(nums, dtype=np.int64)

def make_bl_grid(N_pts=4000):
    """Resample the 575-step BL witness onto a uniform grid on [-1/4,1/4]."""
    v = load_bl_coeffs()
    Nsteps = len(v)        # 575
    S = int(v.sum())
    # g(x) = (1150 / S) * f_0(1150*x + 575/2)  on [-1/4,1/4]
    # On grid x: which cell index?
    x = np.linspace(-0.25, 0.25, N_pts+1)
    s = 1150.0 * x + 575.0/2.0
    idx = np.clip(np.floor(s).astype(int), 0, Nsteps-1)
    g_vals = (1150.0 / S) * v[idx].astype(np.float64)
    return x, g_vals

def autoconv(f, dx):
    return np.convolve(f, f) * dx

def qrr(f, dx, normalize=False):
    if normalize:
        s = float(np.sum(f) * dx)
        if s > 0:
            f = f / s
    g = autoconv(f, dx)
    Minf = float(np.max(g))
    L1 = float(np.sum(g) * dx)
    L2 = float(np.sum(g*g) * dx)
    return Minf, L1, L2, L2 / (Minf * L1)

def main():
    print(f"# Threshold log(16)/pi = {C_STAR:.10f}")
    print()
    # ---------- baseline BL witness on grid ----------
    x, f_bl = make_bl_grid(8000)
    dx = x[1] - x[0]
    Minf, L1, L2, c = qrr(f_bl, dx)
    print(f"BL witness on grid:                    M={Minf:.4f}  c_*={c:.6f}")
    print()

    # ---------- 1. mixture: (1-a)*BL + a*uniform indicator, a in [0,1] ----------
    print("Convex mix (1-a)*BL + a*uniform on [-1/4,1/4]:")
    f_unif = np.ones_like(x) * 2.0
    out = []
    for a in np.linspace(0, 1, 21):
        f = (1-a)*f_bl + a*f_unif
        # both integrate to 1 already
        Minf, L1, L2, c = qrr(f, dx, normalize=True)
        out.append((a, Minf, c))
        print(f"  a={a:.3f}  M={Minf:.4f}  c_*={c:.5f}  (margin to log16/pi = {c-C_STAR:+.4f})")

    # ---------- 2. BL convolved with narrow tent (smoothing) ----------
    print("\nSmoothed BL via convolution with tent of width w (reduces M):")
    for w in [0.005, 0.010, 0.020, 0.040, 0.080, 0.120]:
        # narrow tent kernel
        x_k = x
        ker = np.clip(1 - np.abs(x_k)/w, 0, None)
        ker /= np.sum(ker) * dx
        f_sm = np.convolve(f_bl, ker, mode='same') * dx
        # support may grow outside [-1/4,1/4]; clip
        mask = np.abs(x) <= 0.25
        f_sm = np.where(mask, f_sm, 0.0)
        Minf, L1, L2, c = qrr(f_sm, dx, normalize=True)
        print(f"  w={w:.3f}  M={Minf:.4f}  c_*={c:.6f}  (margin = {c-C_STAR:+.4f})")

    # ---------- 3. BL dilated/contracted (rescale support, breaks ||f||_1=1) ----------
    # Compress BL into [-r, r] for r < 1/4 - investigates "concentrated BL"
    print("\nBL compressed into [-r,r] for r <= 1/4 (concentration test):")
    for r in [0.05, 0.10, 0.15, 0.20, 0.24, 0.25]:
        x_c = np.linspace(-r, r, len(x))
        # map original x in [-1/4,1/4] to x_c via linear scaling
        # f_compressed(x) = (1/4)/r * f_bl(x*1/4/r)
        x_orig = x_c * (0.25 / r)
        # interpolate
        f_c_vals = np.interp(x_orig, x, f_bl) * (0.25 / r)
        # embed back onto same grid (zero outside [-r,r])
        f_emb = np.zeros_like(x)
        mask = np.abs(x) <= r
        if mask.sum() > 0:
            f_emb[mask] = np.interp(x[mask], x_c, f_c_vals)
        Minf, L1, L2, c = qrr(f_emb, dx, normalize=True)
        print(f"  r={r:.3f}  M={Minf:.4f}  c_*={c:.6f}")

    # ---------- 4. symmetric components (BL symmetrized) ----------
    print("\nBL symmetrized: f_sym(x) = (BL(x) + BL(-x))/2:")
    f_sym = (f_bl + f_bl[::-1]) / 2
    Minf, L1, L2, c = qrr(f_sym, dx, normalize=True)
    print(f"  symmetrized BL:   M={Minf:.4f}  c_*={c:.6f}")
    f_antisym_amp = (f_bl - f_bl[::-1]) / 2
    print(f"  ||asym part||_inf = {np.max(np.abs(f_antisym_amp)):.4f}  (relative asymmetry indicator)")

if __name__ == "__main__":
    main()
