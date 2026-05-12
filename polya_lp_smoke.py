"""Smoke test the Handelman LP at small d to verify correctness."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from lasserre.polya_lp.runner import run_one
from lasserre.polya_lp.verify import (
    verify_certificate_symbolic,
    verify_certificate_montecarlo,
)
from lasserre.polya_lp.build import build_window_matrices
from lasserre.polya_lp.symmetry import project_window_set_to_z2


def smoke():
    print("=" * 70)
    print("SMOKE TEST: Handelman LP at d=4, R=4 (variable lambda, Z/2)")
    print("=" * 70)
    rec, build, sol = run_one(d=4, R=4, use_z2=True, verbose=True)
    if sol.alpha is None:
        print("FAIL: solver did not return alpha")
        return False
    print(f"\nResult: alpha = {sol.alpha:.6f}, val(4)_known = 1.102")

    # Verify (use the same rescaled matrices the LP was built from)
    from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled
    _, M_mats = build_window_matrices(4)
    M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats, 4)
    sym = verify_certificate_symbolic(build, M_mats_eff, sol.alpha, sol.x)
    print(f"Symbolic verify: {sym}")
    mc = verify_certificate_montecarlo(build, M_mats_eff, sol.alpha, sol.x,
                                        n_samples=5000)
    print(f"Monte-Carlo verify: {mc}")

    # The certificate must be sound: max_coeff_residual ~0 and min_margin >= -tol.
    if not sym["passed"]:
        print("FAIL: symbolic certificate identity violated")
        return False
    if mc["min_margin"] < -1e-7:
        print("FAIL: Monte-Carlo found mu in Delta with p_lambda(mu) < alpha")
        return False
    print("\nPASS")
    return True


if __name__ == "__main__":
    ok = smoke()
    sys.exit(0 if ok else 1)
