"""Family F4: Matolcsi-Vinuesa arcsine-autoconvolution kernel x trig poly.

SCAFFOLD ONLY. The exact mathematical construction is being confirmed by
reading the MV paper (arXiv:0907.1379); this file will be completed AFTER
the formulas are extracted verbatim from the paper. Until that is done,
imports from this module will raise NotImplementedError.

Planned structure (subject to confirmation from MV paper)
---------------------------------------------------------

Building blocks:
    beta(x)   = (2/pi) / sqrt(1 - 4 x^2)    on (-1/2, 1/2)  [arcsine density]
    K(x)      = (1/delta) * (beta * beta)(x/delta)          [kernel at scale delta]
    G(x)      = sum_{j=1}^n a_j cos(2 pi j x / u)           [trig polynomial]

Test function: combination of K and G to be confirmed from MV paper
(multiplication / convolution / some integral kernel B(x, y)).

Fourier transforms we will need (closed forms):
    beta_hat(xi) = J_0(pi xi)                   [Bessel J_0; standard identity]
    K_hat(xi)    = (beta_hat(delta xi))^2 = J_0(pi delta xi)^2
    G_hat(xi)    = (1/2) sum_j a_j [delta_{j/u} + delta_{-j/u}]  [line spectrum]

Positive-definiteness:
    K is autoconvolution of a nonneg density, so K >= 0 and K_hat >= 0.
    G >= 0 on R requires a cosine-polynomial nonnegativity constraint.
    (Or, if G ghat is needed nonneg instead, use Toeplitz-PSD Bochner.)

Numerical coefficients a_j: 119 values from MV Appendix table (Mathematica QP
output). We will transcribe them when the paper is read.

Pipeline integration:
    - g_iv(t_iv, params)    : interval arithmetic of test function
    - ghat_iv(xi_iv, ...)   : interval arithmetic of its FT
    - weight_iv(xi_iv)      : reuse family_f1_selberg.weight_iv (L^sharp)
    - positive_definite_certificate
    - integral_g_iv / ghat_support / ghat_zero
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class F4Params:
    """Matolcsi-Vinuesa-style parameters.

    Attributes
    ----------
    delta : float
        Scale of the arcsine autoconvolution kernel. MV use 0.138.
    u : float
        Period parameter of the trig polynomial. MV use 0.638.
    a : Sequence[float]
        Cosine coefficients a_1, ..., a_n. MV use n=119 tabulated values.
    """
    delta: float
    u: float
    a: Sequence[float] = field(default_factory=list)

    def n(self) -> int:
        return len(self.a)


def g_iv(t_iv, params: F4Params):
    raise NotImplementedError(
        "Family F4 is a scaffold pending MV paper re-read. "
        "See delsarte_dual/mv_construction_detailed.md once written."
    )


def ghat_iv(xi_iv, params: F4Params):
    raise NotImplementedError("F4 scaffold pending.")


def integral_g_iv(params: F4Params, n_subdiv: int = 1024):
    raise NotImplementedError("F4 scaffold pending.")


def ghat_support(params: F4Params) -> float:
    raise NotImplementedError("F4 scaffold pending.")


def ghat_zero(params: F4Params):
    raise NotImplementedError("F4 scaffold pending.")


def positive_definite_certificate(params: F4Params):
    raise NotImplementedError("F4 scaffold pending.")


def weight_iv(xi_iv):
    """Reuse the F1 sharp signed weight L^sharp."""
    from .family_f1_selberg import weight_iv as f1_weight
    return f1_weight(xi_iv)
