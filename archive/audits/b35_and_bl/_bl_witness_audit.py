"""Audit of Boyer-Li 2025 (arXiv:2506.16750) witness g on [-1/4, 1/4].

We use the verbatim integer coefficients v_n (n=0..574) from
delsarte_dual/restricted_holder/coeffBL.txt. The step function
f_0 = sum_n v_n * 1_{[n,n+1)} on [0, 575] gives autoconvolution via
discrete linear convolution L = v * v at integer lags. The rescaled
witness g on [-1/4, 1/4] is g(x) = (1150/S) f_0(1150 x + 575/2),
S = sum_n v_n.

For g (a step function on [-1/4, 1/4] with 575 cells of width 1/1150):
  ||g||_2^2  = (1/S^2) * 1150 * sum_n v_n^2
  (g*g)(t)   piecewise linear on grid spacing 1/1150
  ||g*g||_inf = (1150/S^2) * max_j L_j         (L_j = (v*v)_j)
  ||g*g||_2^2 = (1/(1150 S^4)) * (4/3 * sum_j L_j^2 - 1/3 * sum_j L_j L_{j+1})
              [from the L^2 norm of a piecewise-linear interpolant]
  hat g(xi)  = (1/S) * (sin(pi xi/1150)/(pi xi/1150)) * sum_n v_n e^{-i pi xi (2n-574)/1150}
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
COEFF_PATH = ROOT / "delsarte_dual" / "restricted_holder" / "coeffBL.txt"


def load_coeffs() -> np.ndarray:
    raw = COEFF_PATH.read_text().strip()
    inside = raw.strip().lstrip("{").rstrip("}")
    nums = [int(x) for x in re.split(r"[,\s]+", inside) if x]
    arr = np.asarray(nums, dtype=object)  # python ints, exact
    return arr


def main() -> None:
    v = load_coeffs()
    N = len(v)
    print(f"# coefficients N = {N}")
    assert N == 575, f"expected 575 coeffs, got {N}"

    S = int(sum(int(x) for x in v))
    print(f"S = sum v_n = {S}")

    # ----- discrete autoconvolution L = v * v ------
    # length 2N-1 = 1149
    v_int = np.array([int(x) for x in v], dtype=np.int64)
    # use Python ints to avoid overflow (max coeff ~1.5e5, N=575)
    # max term ~ N * (1.5e5)^2 ~ 1.3e13 < 2^63 ~ 9.2e18. Safe in int64.
    # but cumulative sums ~ 1.3e16 also safe.
    L = np.convolve(v_int, v_int).astype(np.int64)
    Lmax = int(L.max())
    jmax = int(np.argmax(L))
    print(f"max_j L_j = L_{jmax} = {Lmax}")

    # ----- ||g*g||_inf -----
    gg_inf = 1150.0 * Lmax / (S ** 2)
    print(f"||g*g||_inf  M = {gg_inf!r}  ~ {gg_inf:.15f}")

    # ----- ||g||_2^2 = K -----
    sum_v2 = int(np.sum(v_int.astype(object) * v_int.astype(object)))
    K = 1150.0 * sum_v2 / (S ** 2)
    print(f"||g||_2^2     K = {K!r}  ~ {K:.15f}")
    print(f"K - M (gap)     = {K - gg_inf:.15f}")

    # ----- ||g*g||_2^2 -----
    # g*g is piecewise linear on the grid t_j = (j - (N-1))/1150 (j=0..2N-2)
    # with node values  V_j = (1/S^2) * L_j * (need scale check).
    # On [-1/4, 1/4] dilation: g(x) = (1150/S) * f_0(1150 x + 575/2),
    # so (g*g)(t) = (1150/S^2) * (f_0 * f_0)(1150 t + 575).
    # For step f_0 with unit cells, (f_0*f_0)(s) is piecewise linear with
    # node values L_j at integer s = j+1 (j=0,...,2N-2), zero outside.
    # Therefore (g*g)(t) is piecewise linear with node values
    #   V_j := (1150/S^2) * L_j  at  t_j := (j+1 - N + 1/2)/1150,  spacing h=1/1150.
    # (We don't need t_j locations for the L^2 norm.)
    # L^2 norm of piecewise linear w/ nodes V_j at uniform spacing h:
    #   ||p||_2^2 = h * sum_j (V_j^2 / 3 + V_j V_{j+1} / 3 + V_{j+1}^2 / 3)
    #             = (h/3) * (sum_j V_j^2 + sum_j V_j V_{j+1} + sum_j V_{j+1}^2)
    # = (h/3) * (2*sum V_j^2 - V_0^2 - V_{end}^2 + sum_j V_j V_{j+1})
    # We'll compute via scaled L:
    h = 1.0 / 1150.0
    Lf = L.astype(np.float64)
    sum_Lj2 = float(np.sum(Lf * Lf))
    sum_LjLj1 = float(np.sum(Lf[:-1] * Lf[1:]))
    end_sq = float(Lf[0] ** 2 + Lf[-1] ** 2)
    # In V-units: V_j = c L_j with c = 1150/S^2
    c_scale = 1150.0 / (S ** 2)
    integral_in_L_units = (h / 3.0) * (2.0 * sum_Lj2 - end_sq + sum_LjLj1)
    gg_l22 = (c_scale ** 2) * integral_in_L_units
    print(f"||g*g||_2^2     = {gg_l22:.15f}")
    print(f"||g*g||_2^2 / M = {gg_l22 / gg_inf:.15f}    (this is c-param)")

    # asymmetry check
    # f_0 step function values are v_n on [n, n+1). After translate by -575/2,
    # the function f_1(x) = f_0(x + 575/2). f_1 symmetric iff v[n] == v[N-1-n].
    asym = max(abs(int(v_int[i]) - int(v_int[N - 1 - i])) for i in range(N // 2))
    asym_rel = asym / max(abs(int(x)) for x in v_int)
    print(f"max |v_n - v_{{N-1-n}}| = {asym}   (rel to max coeff: {asym_rel:.6f})")
    print("-> witness is " + ("SYMMETRIC" if asym == 0 else "ASYMMETRIC"))

    # quick visual: first/last 5 coeffs
    print("first 5:", v_int[:5])
    print("last  5:", v_int[-5:])

    # ----- hat g on grid [-50, 50] -----
    # hat g(xi) = int g(x) e^{-i 2 pi xi x} dx
    # g(x) = (1150/S) f_0(1150 x + 575/2)
    # f_0(s) = sum_n v_n 1_{[n,n+1)}(s)
    # change vars s = 1150 x + 575/2, dx = ds/1150
    # hat g(xi) = (1/S) int_0^{575} f_0(s) exp(-i 2 pi xi (s - 575/2)/1150) ds
    #          = (1/S) sum_n v_n int_n^{n+1} exp(-i 2 pi xi (s - 575/2)/1150) ds
    # Let alpha = 2 pi xi / 1150. Then
    #   int_n^{n+1} e^{-i alpha (s - 575/2)} ds
    #   = e^{-i alpha (n + 1/2 - 575/2)} * (2 sin(alpha/2)/alpha)   if alpha!=0
    # so hat g(xi) = (1/S) * sinc(alpha/(2 pi)) * sum_n v_n e^{-i alpha (n - (N-1)/2)}
    # where sinc(t) := sin(pi t)/(pi t).
    def hatg(xi: float) -> complex:
        if xi == 0:
            return 1.0 + 0.0j
        alpha = 2.0 * np.pi * xi / 1150.0
        # sin(alpha/2)/(alpha/2)
        sinc_half = np.sin(alpha / 2.0) / (alpha / 2.0)
        # sum v_n exp(-i alpha (n - (N-1)/2))
        n = np.arange(N)
        phases = np.exp(-1j * alpha * (n - (N - 1) / 2.0))
        s = np.sum(v_int.astype(np.float64) * phases)
        return (1.0 / S) * sinc_half * s

    xis = np.linspace(-50.0, 50.0, 1001)
    hat_vals = np.array([hatg(float(x)) for x in xis])
    abs_hat = np.abs(hat_vals)
    print("\n--- hat g on [-50, 50] (1001 pts) ---")
    print(f"|hat g(0)| = {abs_hat[500]:.6f}  (should be 1)")
    print(f"max |hat g| on grid = {abs_hat.max():.6f} at xi = {xis[abs_hat.argmax()]:.3f}")
    print(f"min |hat g| on grid = {abs_hat.min():.6e} at xi = {xis[abs_hat.argmin()]:.3f}")
    # symmetry diagnostic in Fourier:
    # for symmetric g, hat g real-valued. For asymmetric, imag part nonzero.
    print(f"max |Im hat g(xi)| over xis = {np.max(np.abs(hat_vals.imag)):.6f}")
    print(f"max |Re hat g(xi) - Re hat g(-xi)| = {np.max(np.abs(hat_vals.real - hat_vals.real[::-1])):.3e}")
    print(f"max |Im hat g(xi) + Im hat g(-xi)| = {np.max(np.abs(hat_vals.imag + hat_vals.imag[::-1])):.3e}")
    # Save to JSON-ish text for downstream
    out = ROOT / "bl_witness_audit.json"
    import json
    json.dump(
        {
            "K": K,
            "M": gg_inf,
            "K_minus_M": K - gg_inf,
            "gg_l22": gg_l22,
            "c_param_gg_l22_over_M": gg_l22 / gg_inf,
            "asymmetric_max_diff": asym,
            "S": S,
            "N": N,
            "Lmax": Lmax,
            "jmax": jmax,
            "hat_g_grid_xi": xis.tolist(),
            "hat_g_grid_re": hat_vals.real.tolist(),
            "hat_g_grid_im": hat_vals.imag.tolist(),
            "hat_g_grid_abs": abs_hat.tolist(),
        },
        open(out, "w"),
    )
    print(f"\nwrote {out}")

    # ----- find sign changes / approximate real zeros of |hat g| --------
    # Real zeros of hat g (real part for asymmetric => need both real & imag = 0).
    # Use: zeros of |hat g|^2 on a fine grid, then refine.
    fine = np.linspace(-50.0, 50.0, 50001)
    fine_vals = np.array([hatg(float(x)) for x in fine])
    fine_abs = np.abs(fine_vals)
    # Local minima below threshold
    minima_idx = []
    thresh = 1e-3
    for i in range(1, len(fine) - 1):
        if fine_abs[i] < fine_abs[i - 1] and fine_abs[i] < fine_abs[i + 1] and fine_abs[i] < thresh:
            minima_idx.append(i)
    print(f"\n# near-zeros of |hat g| on [-50,50] (local mins below {thresh}): {len(minima_idx)}")
    for idx in minima_idx[:30]:
        print(f"   xi ~ {fine[idx]:+8.4f}   |hat g| = {fine_abs[idx]:.3e}")
    if len(minima_idx) > 30:
        print(f"   ... and {len(minima_idx) - 30} more")

    # ----- complex zeros (Blaschke / "off the real axis") --------
    # We probe hat g(xi + i eta) for eta in (0, eta_max] on a coarse grid;
    # locate complex zeros via 2D mesh + secondary refinement.
    print("\n--- Searching complex zeros of hat g in upper half-plane ---")
    # restrict search to bounded box
    XI_LO, XI_HI = -50.0, 50.0
    ETA_LO, ETA_HI = 0.05, 5.0
    nx, ny = 401, 41
    xs = np.linspace(XI_LO, XI_HI, nx)
    ys = np.linspace(ETA_LO, ETA_HI, ny)
    grid_abs = np.empty((ny, nx))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            z = complex(x, y)
            # Re-use the formula with xi -> z (complex)
            alpha = 2.0 * np.pi * z / 1150.0
            sinc_half = np.sin(alpha / 2.0) / (alpha / 2.0)
            n = np.arange(N)
            phases = np.exp(-1j * alpha * (n - (N - 1) / 2.0))
            s = np.sum(v_int.astype(np.float64) * phases)
            val = (1.0 / S) * sinc_half * s
            grid_abs[i, j] = abs(val)
    # find local minima below threshold
    cz_thresh = 0.5  # log-scale thresh; we report the deepest minima
    deepest = []
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            v0 = grid_abs[i, j]
            if (
                v0 < grid_abs[i - 1, j]
                and v0 < grid_abs[i + 1, j]
                and v0 < grid_abs[i, j - 1]
                and v0 < grid_abs[i, j + 1]
            ):
                deepest.append((v0, xs[j], ys[i]))
    deepest.sort()
    print(f"found {len(deepest)} local minima of |hat g(z)| in box xi=[-50,50], eta=[0.05,5]")
    print("Top 12 deepest minima (likely complex zeros / Blaschke zeros in UHP):")
    for v0, x0, y0 in deepest[:12]:
        print(f"   z ~ {x0:+8.3f} + i {y0:6.3f}    |hat g(z)| = {v0:.3e}")
    print("\nNote: Blaschke zeros = zeros of hat(g*g) in UHP, but hat(g*g) = (hat g)^2,")
    print("so they coincide with zeros of hat g (each counted with multiplicity 2).")


if __name__ == "__main__":
    main()
