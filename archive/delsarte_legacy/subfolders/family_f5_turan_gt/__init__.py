"""F5: Turan / Gorbachev-Tikhonov triangle-prefix family for the C_{1a} dual.

g(t) = (1 - 2|t|)_+ * P(t^2),   P(u) = sum c_k u^k,   P >= 0 on [0, 1/4].

See `derivation.md` for the full derivation and the verdict (F5 is dominated
by F1 in the existing Delsarte ratio pipeline). This module is provided for
completeness and reproducibility of the negative result.
"""

from .f5 import (
    F5Params,
    NotPDAdmissible,
    f5_idealised_ratio,
    f5_lower_bound,
    g_hat_value,
    g_hat_zero,
    g_iv,
    g_value,
    is_pd_admissible,
    M_g,
    weight_iv,
)

__all__ = [
    "F5Params",
    "NotPDAdmissible",
    "f5_idealised_ratio",
    "f5_lower_bound",
    "g_hat_value",
    "g_hat_zero",
    "g_iv",
    "g_value",
    "is_pd_admissible",
    "M_g",
    "weight_iv",
]
