"""V6 x-space verification of 2-scale arcsine kernel.

K_hat(xi) = 0.85*J0(pi*delta1*xi)^2 + 0.15*J0(pi*delta2*xi)^2
where delta1 = 0.138, delta2 = 0.045.

K(x) = inv-FT of K_hat = 0.85*K_arc(x; delta1) + 0.15*K_arc(x; delta2)
where K_arc(x; delta) = phi_arc(.; delta) * phi_arc(.; delta) and
phi_arc(x; delta) = (2/(pi*delta)) * 1/sqrt(1 - (2x/delta)^2) on (-delta/2, delta/2).

K_arc(x; delta) has supp [-delta, delta] with log-singularity at 0.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from scipy.integrate import quad
from scipy.special import j0


DELTA1 = 0.138
DELTA2 = 0.045
W1 = 0.85
W2 = 0.15


def K_hat(xi: np.ndarray | float) -> np.ndarray | float:
    """Fourier transform of K."""
    return W1 * j0(math.pi * DELTA1 * xi) ** 2 + W2 * j0(math.pi * DELTA2 * xi) ** 2


def K_arc(x: float, delta: float) -> float:
    """Analytic auto-convolution of arcsine on [-delta/2, delta/2].

    Closed form: For |x| < delta,
      K_arc(x; delta) = (2 / (pi^2 * delta^2)) * integral with arcsine kernel.

    We use the known identity: the Fourier transform of phi_arc(.; delta) is
    J_0(pi * delta * xi). So K_arc_hat(xi) = J_0(pi*delta*xi)^2.

    A direct convolution form: K_arc(x; delta) =
      int_{max(-d/2, x-d/2)}^{min(d/2, x+d/2)} phi_arc(t; delta) phi_arc(x-t; delta) dt
    for |x| <= delta, else 0.

    For numerics we compute it via inverse Fourier:
      K_arc(x; delta) = 2 * int_0^infty J_0(pi*delta*xi)^2 cos(2*pi*xi*x) dxi.

    But for clarity / to handle the log-spike, we will compute K(x) globally
    via numerical inverse Fourier of the full K_hat below.
    """
    raise NotImplementedError  # Use inverse-FT routine instead.


def K_via_invFT(x: float, xi_max: float = 800.0) -> float:
    """K(x) = 2 * int_0^xi_max K_hat(xi) cos(2*pi*xi*x) dxi.

    K_hat decays like 1/(pi*delta*xi) so K_hat^2 like 1/xi, so K is not
    absolutely integrable -- but K is a probability density so the inv-FT
    exists as an improper integral. We use scipy.quad with oscillatory weight
    via 'weight=cos' option, which handles oscillatory integrands.
    """
    if abs(x) < 1e-14:
        # Singular at 0 (logarithmic); split integral
        def integrand(xi):
            return K_hat(xi)
        val, _ = quad(integrand, 0.0, xi_max, limit=500)
        return 2.0 * val
    # Use oscillatory weighting:  w(xi) = cos(omega*xi), omega = 2*pi*x
    def integrand(xi):
        return K_hat(xi)
    val, _ = quad(
        integrand,
        0.0,
        xi_max,
        weight="cos",
        wvar=2.0 * math.pi * x,
        limit=200,
    )
    return 2.0 * val


def main():
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "_V6_xspace_verify.json",
    )

    # Step 1: K(x) on fine grid in [-0.20, 0.20]
    xs = np.linspace(-0.20, 0.20, 801)  # 0.0005 spacing
    Kx = np.zeros_like(xs)
    for i, x in enumerate(xs):
        Kx[i] = K_via_invFT(float(x))

    # Step 2: K(x) >= 0 everywhere?
    min_K = float(Kx.min())
    argmin = int(np.argmin(Kx))
    x_min = float(xs[argmin])
    nonneg = min_K >= -1e-6  # numerical tolerance
    n_neg = int(np.sum(Kx < -1e-8))

    # Step 3: supp(K) subset [-0.138, 0.138] ?
    outside = np.abs(xs) > DELTA1 + 1e-9
    K_outside = Kx[outside]
    max_leak = float(np.max(np.abs(K_outside))) if K_outside.size > 0 else 0.0
    # Where is the worst leak?
    if K_outside.size > 0:
        leak_idx = int(np.argmax(np.abs(K_outside)))
        leak_x = float(xs[outside][leak_idx])
    else:
        leak_x = None

    # Step 4: integral K(x) dx -- should equal K_hat(0) = 1
    integral_K_trap = float(np.trapz(Kx, xs))

    # Better: use scipy.quad on the inverse FT (handle log spike at 0)
    # Use symmetric integral with split at small epsilon
    def K_for_quad(x):
        return K_via_invFT(float(x))

    try:
        eps = 1e-4
        val_left, _ = quad(K_for_quad, -0.20, -eps, limit=100)
        val_right, _ = quad(K_for_quad, eps, 0.20, limit=100)
        # Near 0, use trapezoid on fine sub-grid:
        x_near = np.linspace(-eps, eps, 41)
        K_near = np.array([K_for_quad(float(xx)) for xx in x_near])
        val_near = float(np.trapz(K_near, x_near))
        integral_K_quad = val_left + val_near + val_right
    except Exception as e:
        integral_K_quad = None
        print(f"quad failed: {e}", file=sys.stderr)

    # Step 6: K_hat(j) >= 0 for j=1..200 (trivial since sum of squares)
    js = np.arange(1, 201)
    Khat_at_j = K_hat(js.astype(float))
    min_Khat = float(Khat_at_j.min())
    j_min = int(js[np.argmin(Khat_at_j)])

    # Step 7: K_2 = int K(x)^2 dx
    # Subtract two log-spikes analytically near 0:
    # Near 0, K(x) ~ W1*A1*log(1/|x|) + W2*A2*log(1/|x|) + smooth
    # where for arcsine auto-convolution, K_arc(x; delta) ~ (4/(pi^2 * delta^2))
    # * log(delta / |x|) as x -> 0 (leading log behavior).
    # Coefficient of log(1/|x|) near 0: c = 4/(pi^2 * delta^2) for each scale.
    c1 = 4.0 / (math.pi ** 2 * DELTA1 ** 2)
    c2 = 4.0 / (math.pi ** 2 * DELTA2 ** 2)
    # Total log coefficient (combined into the kernel value):
    c_log = W1 * c1 + W2 * c2

    # Direct K_2 via fine grid trapezoid (may underestimate the log spike):
    K2_trap = float(np.trapz(Kx ** 2, xs))

    # Better K_2 via quad with singularity at 0:
    def K2_integrand(x):
        return K_via_invFT(float(x)) ** 2

    try:
        eps2 = 5e-5
        # Outside small neighborhood:
        v_left, _ = quad(K2_integrand, -DELTA1, -eps2, limit=200)
        v_right, _ = quad(K2_integrand, eps2, DELTA1, limit=200)
        # Inside small neighborhood, do trapezoidal with finer subdivision:
        x_inner = np.geomspace(eps2, 1e-2, 60)
        x_inner_full = np.concatenate([-x_inner[::-1], [0.0], x_inner])
        # Skip x=0 in integration; use symmetric points:
        K_inner = np.array(
            [K_via_invFT(float(xx)) for xx in x_inner_full if abs(xx) > 1e-12]
        )
        x_inner_eval = np.array([xx for xx in x_inner_full if abs(xx) > 1e-12])
        # Sort:
        order = np.argsort(x_inner_eval)
        x_inner_eval = x_inner_eval[order]
        K_inner = K_inner[order]
        v_inner = float(np.trapz(K_inner ** 2, x_inner_eval))
        K2_quad = v_left + v_inner + v_right
    except Exception as e:
        K2_quad = None
        print(f"K2 quad failed: {e}", file=sys.stderr)

    # Parseval check value (target from prompt):
    K2_parseval_target = 4.358

    # Summary
    result = {
        "delta1": DELTA1,
        "delta2": DELTA2,
        "w1": W1,
        "w2": W2,
        "grid_n": int(xs.size),
        "grid_min": float(xs[0]),
        "grid_max": float(xs[-1]),
        "min_K": min_K,
        "argmin_x": x_min,
        "n_grid_neg": n_neg,
        "nonneg_within_1e-6": bool(nonneg),
        "max_leak_outside_delta1": max_leak,
        "leak_x": leak_x,
        "integral_K_trapz_grid": integral_K_trap,
        "integral_K_quad_split": integral_K_quad,
        "Khat_min_on_int_1to200": min_Khat,
        "Khat_argmin_j": j_min,
        "c_log_coefficient_K_sing_at_0": c_log,
        "K2_trap_grid": K2_trap,
        "K2_quad_with_singularity": K2_quad,
        "K2_parseval_target_from_prompt": K2_parseval_target,
        "K2_diff_to_parseval": (
            None if K2_quad is None else K2_quad - K2_parseval_target
        ),
    }

    # Also save sample values of K(x) at notable points
    notable = {}
    for x in [0.0, 0.001, 0.005, 0.01, 0.045, 0.046, 0.10, 0.138, 0.139, 0.15, 0.20]:
        notable[f"K({x:.4f})"] = K_via_invFT(x)
    result["K_sample_values"] = notable

    # Plot K(x)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        # Clip enormous spike for plotting:
        Kx_plot = np.clip(Kx, 0, np.percentile(Kx, 99.5))
        ax.plot(xs, Kx_plot, "b-", lw=1)
        ax.axvline(DELTA1, color="r", ls="--", alpha=0.5, label=f"+/- delta1 = {DELTA1}")
        ax.axvline(-DELTA1, color="r", ls="--", alpha=0.5)
        ax.axvline(DELTA2, color="g", ls="--", alpha=0.5, label=f"+/- delta2 = {DELTA2}")
        ax.axvline(-DELTA2, color="g", ls="--", alpha=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("K(x)  (clipped at 99.5 pct for visibility)")
        ax.set_title("2-scale arcsine kernel K(x); double log-spike at 0")
        ax.legend()
        ax.grid(alpha=0.3)
        plot_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "_V6_xspace_verify_plot.png",
        )
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        result["plot_path"] = plot_path
    except Exception as e:
        print(f"plot failed: {e}", file=sys.stderr)
        result["plot_path"] = None

    with open(out_path, "w") as fp:
        json.dump(result, fp, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
