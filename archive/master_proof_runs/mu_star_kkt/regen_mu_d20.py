"""Regenerate mu_star_d20.npz with the proper KKT-correct mu* (residual 1e-10).
Save mu_star, alpha (for diagnostic), residual to npz.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kkt_correct_mu_star import find_kkt_correct_mu_star


def main():
    d = 20
    print(f"Regenerating mu_star at d={d} with full pipeline (Phase 1+2+subgrad)")
    t0 = time.time()
    result = find_kkt_correct_mu_star(
        d=d, x_cap=1.0, n_starts=6400, n_workers=64,
        top_K_phase2=100, top_K_phase3=20,
        target_residual=1e-6, verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")
    if result['mu_star'] is None:
        print("FAILED")
        return
    print(f"f(mu*) = {result['f_value']:.10f}")
    print(f"residual = {result['residual']:.6e}")
    print(f"active = {len(result['active_idx'])}")
    np.savez('mu_star_d20.npz',
             mu_star=result['mu_star'],
             alpha_star=result['alpha_star'],
             active_idx=np.array(result['active_idx']),
             f_value=result['f_value'],
             residual=result['residual'])
    print("Saved mu_star_d20.npz")


if __name__ == "__main__":
    main()
