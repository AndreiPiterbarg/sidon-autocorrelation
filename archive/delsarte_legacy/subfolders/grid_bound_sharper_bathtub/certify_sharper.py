"""Certificate verification for the sharper-bathtub pipeline.

STATUS: OBSTRUCTED. No sharper certificates are written in this session.
This module re-exports the baseline certificate verifier for pipeline
completeness; once a rigorous mu_sharper is proved, sharper-labelled
certificates can be verified identically.
"""
from __future__ import annotations

from delsarte_dual.grid_bound.certify_mm import verify_certificate_mm


def verify_sharper_certificate(path: str) -> dict:
    """Verify a certificate file produced by ``bisect_sharper``.

    Currently identical to the baseline verifier; the file schema is
    unchanged.
    """
    return verify_certificate_mm(path)


__all__ = ["verify_sharper_certificate"]
