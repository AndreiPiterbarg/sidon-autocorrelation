"""End-to-end Tier-4 validation: tier4_solve vs monolithic MOSEK 1e-9.

For each (d, R) in the grid:
  1. Run tier4_solve and capture (alpha_rigorous, alpha_polish, wall, breakdown).
  2. Run monolithic MOSEK 1e-9 and capture (alpha_truth, wall_truth).
  3. Sanity: |alpha_polish - alpha_truth| < 1e-7 (Tier 4 must reproduce truth).
  4. Sanity: alpha_rigorous <= alpha_truth (Jansson is a valid lower bound).
  5. Report wall speedup tier4 / monolithic.

Writes _tier4_e2e_results.json.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasserre.polya_lp.tier4.driver import tier4_solve, monolithic_solve


def run_one(d, R, verbose=False):
    print(f"\n=== d={d} R={R} ===", flush=True)
    # Monolithic baseline
    sol_mon, wall_mon = monolithic_solve(d, R, use_z2=True, tol=1e-9)
    print(f"  MONOLITHIC MOSEK 1e-9: alpha={sol_mon.alpha:.10f}  "
          f"wall={wall_mon*1000:.1f}ms", flush=True)

    # Tier 4
    t0 = time.time()
    res = tier4_solve(d, R, use_z2=True, coarse_backend="highs_ipm",
                      coarse_tol=1e-5, polish_tol=1e-9,
                      verbose=verbose)
    wall_t4 = time.time() - t0
    print(f"  TIER 4              : alpha_rig ={res.alpha_rigorous:.10f}  "
          f"alpha_pol={res.alpha_polish:.10f}  wall={wall_t4*1000:.1f}ms", flush=True)
    print(f"    breakdown: setup {res.wall_setup_s*1000:.1f}ms  "
          f"coarse {res.wall_coarse_s*1000:.1f}ms  "
          f"active {res.wall_active_s*1000:.1f}ms  "
          f"polish_build {res.wall_polish_build_s*1000:.1f}ms  "
          f"polish_solve {res.wall_polish_solve_s*1000:.1f}ms  "
          f"verify {res.wall_verify_s*1000:.1f}ms  "
          f"rigorize {res.wall_rigorize_s*1000:.1f}ms", flush=True)
    print(f"    active: lambda {res.n_W_active}/{res.n_W_full}  "
          f"cbeta {res.n_cbeta_active}/{res.n_cbeta_full}  "
          f"verify_max_v={res.verify_max_violation:.3e}  "
          f"fallback={res.fell_back_to_full}", flush=True)

    diff_polish = abs(res.alpha_polish - sol_mon.alpha)
    rig_below = (res.alpha_rigorous <= sol_mon.alpha + 1e-9)
    speedup = wall_mon / wall_t4
    print(f"    diff(polish-truth)={diff_polish:.2e}  "
          f"rig<=truth:{rig_below}  speedup={speedup:.2f}x", flush=True)

    return dict(
        d=d, R=R,
        alpha_truth=sol_mon.alpha,
        alpha_polish=res.alpha_polish,
        alpha_rigorous=res.alpha_rigorous,
        alpha_coarse=res.alpha_coarse,
        wall_monolithic=wall_mon,
        wall_tier4=wall_t4,
        speedup=speedup,
        diff_polish_vs_truth=diff_polish,
        rig_le_truth=rig_below,
        n_W_full=res.n_W_full,
        n_W_active=res.n_W_active,
        n_cbeta_full=res.n_cbeta_full,
        n_cbeta_active=res.n_cbeta_active,
        verify_max_v=res.verify_max_violation,
        fell_back=res.fell_back_to_full,
        coarse_kkt=res.coarse_kkt,
        epsilon_shift=res.epsilon_shift,
        breakdown=dict(
            setup_ms=res.wall_setup_s * 1000,
            coarse_ms=res.wall_coarse_s * 1000,
            active_ms=res.wall_active_s * 1000,
            polish_build_ms=res.wall_polish_build_s * 1000,
            polish_solve_ms=res.wall_polish_solve_s * 1000,
            verify_ms=res.wall_verify_s * 1000,
            rigorize_ms=res.wall_rigorize_s * 1000,
        ),
    )


if __name__ == "__main__":
    grid = [
        (8, 4), (8, 6), (8, 8), (8, 10), (8, 12),
        (10, 4), (10, 6), (10, 8),
        (12, 4), (12, 6), (12, 8),
    ]
    rows = []
    for d, R in grid:
        try:
            rows.append(run_one(d, R, verbose=False))
        except Exception as e:
            import traceback
            traceback.print_exc()
            rows.append(dict(d=d, R=R, error=str(e)))

    print("\n--- SUMMARY ---", flush=True)
    print(f"{'d':>3s} {'R':>3s} {'alpha_truth':>12s} {'a_polish':>12s} {'a_rig':>12s} "
          f"{'diff':>8s} {'wall_T4':>8s} {'wall_mon':>8s} {'speedup':>8s} "
          f"{'lam_act':>8s} {'cb_act':>8s}", flush=True)
    for r in rows:
        if "error" in r:
            print(f"{r['d']:>3d} {r['R']:>3d}  ERROR: {r['error'][:60]}", flush=True)
            continue
        print(f"{r['d']:>3d} {r['R']:>3d} {r['alpha_truth']:>12.8f} "
              f"{r['alpha_polish']:>12.8f} {r['alpha_rigorous']:>12.8f} "
              f"{r['diff_polish_vs_truth']:>8.1e} "
              f"{r['wall_tier4']*1000:>7.1f}ms {r['wall_monolithic']*1000:>7.1f}ms "
              f"{r['speedup']:>7.2f}x "
              f"{r['n_W_active']}/{r['n_W_full']} "
              f"{r['n_cbeta_active']}/{r['n_cbeta_full']}",
              flush=True)

    with open("_tier4_e2e_results.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print("\nWrote _tier4_e2e_results.json", flush=True)
