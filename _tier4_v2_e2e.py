"""End-to-end Tier 4 v2 (Tier 3+4 composed) validation.

For each (d, R):
  1. Run monolithic MOSEK 1e-9 baseline.
  2. Run tier4_solve_v2 with mosek_simplex coarse backend.
  3. Verify alpha_polish matches monolithic to <= 1e-7.
  4. Verify alpha_rigorous <= alpha_truth (Jansson is sound LB).
  5. Report wall-time speedup and per-stage breakdown.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasserre.polya_lp.tier4.driver_v2 import tier4_solve_v2, monolithic_solve


def run_one(d, R, verbose=False, **kw):
    print(f"\n=== d={d} R={R} ===", flush=True)
    sol_mon, wall_mon = monolithic_solve(d, R, use_z2=True, tol=1e-9)
    print(f"  MONOLITHIC MOSEK 1e-9: alpha={sol_mon.alpha:.10f}  "
          f"wall={wall_mon*1000:.1f}ms", flush=True)

    t0 = time.time()
    res = tier4_solve_v2(
        d, R, use_z2=True,
        coarse_backend=kw.get("coarse_backend", "mosek_simplex"),
        coarse_tol=kw.get("coarse_tol", 1e-6),
        polish_tol=1e-9,
        cg_violator_tol=kw.get("cg_violator_tol", 1e-7),
        polish_violator_tol=1e-9,
        rigorize_prec=64,
        max_cg_iter=30, max_recovery_iter=5,
        verbose=verbose,
    )
    wall_t4 = time.time() - t0

    diff_polish = (abs(res.alpha_polish - sol_mon.alpha)
                    if res.alpha_polish is not None else float("inf"))
    rig_le_truth = (res.alpha_rigorous is not None
                    and res.alpha_rigorous <= sol_mon.alpha + 1e-9)
    speedup = wall_mon / wall_t4

    print(f"  TIER 4 v2           : alpha_rig ={res.alpha_rigorous:.10f}  "
          f"alpha_pol={res.alpha_polish:.10f}  wall={wall_t4*1000:.1f}ms",
          flush=True)
    print(f"    breakdown: setup {res.wall_setup_s*1000:.1f}ms  "
          f"cg {res.wall_cg_total_s*1000:.1f}ms  "
          f"active {res.wall_active_s*1000:.1f}ms  "
          f"polish_total {res.wall_polish_total_s*1000:.1f}ms  "
          f"verify {res.wall_verify_total_s*1000:.1f}ms  "
          f"rigorize {res.wall_rigorize_s*1000:.1f}ms",
          flush=True)
    print(f"    Sigma: seed={res.n_sigma_seed} -> final={res.n_sigma_final}/"
          f"{res.n_sigma_full}  cg_iter={res.n_cg_iter}  recovery={res.n_recovery_iter}",
          flush=True)
    print(f"    lambda: active {res.n_W_active_final}/{res.n_W_full}",
          flush=True)
    print(f"    primal_violation_max={res.primal_max_violation:.2e}  "
          f"dual_violation_max={res.dual_max_violation:.2e}  "
          f"eps_total={res.epsilon_total:.2e}  matvec_err={res.matvec_err_bound:.2e}",
          flush=True)
    print(f"    diff(polish vs truth)={diff_polish:.2e}  "
          f"rig<=truth:{rig_le_truth}  speedup={speedup:.2f}x  "
          f"fallback={res.fell_back_to_full}",
          flush=True)
    return dict(
        d=d, R=R,
        alpha_truth=sol_mon.alpha,
        alpha_polish=res.alpha_polish,
        alpha_rigorous=res.alpha_rigorous,
        diff_polish_vs_truth=diff_polish,
        rig_le_truth=rig_le_truth,
        wall_mon_ms=wall_mon * 1000, wall_t4_ms=wall_t4 * 1000,
        speedup=speedup,
        n_W_full=res.n_W_full, n_W_active=res.n_W_active_final,
        n_sigma_seed=res.n_sigma_seed,
        n_sigma_full=res.n_sigma_full,
        n_sigma_final=res.n_sigma_final,
        n_cg_iter=res.n_cg_iter,
        n_recovery=res.n_recovery_iter,
        primal_max_v=res.primal_max_violation,
        dual_max_v=res.dual_max_violation,
        epsilon_total=res.epsilon_total,
        matvec_err=res.matvec_err_bound,
        fell_back=res.fell_back_to_full,
        coarse_backend=res.coarse_backend,
        breakdown_ms=dict(
            setup=res.wall_setup_s * 1000,
            cg=res.wall_cg_total_s * 1000,
            active=res.wall_active_s * 1000,
            polish=res.wall_polish_total_s * 1000,
            verify=res.wall_verify_total_s * 1000,
            rigorize=res.wall_rigorize_s * 1000,
        ),
    )


if __name__ == "__main__":
    grid = [
        (8, 4), (8, 6), (8, 8), (8, 10), (8, 12),
        (10, 4), (10, 6), (10, 8),
        (12, 4), (12, 6), (12, 8),
        (14, 6), (14, 8),
        (16, 4), (16, 6),
    ]
    rows = []
    verbose = ("-v" in sys.argv) or ("--verbose" in sys.argv)
    for d, R in grid:
        try:
            rows.append(run_one(d, R, verbose=verbose))
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append(dict(d=d, R=R, error=str(e)))

    print("\n--- SUMMARY ---", flush=True)
    print(f"{'d':>3s} {'R':>3s} {'a_truth':>10s} {'a_pol':>10s} "
          f"{'diff':>9s} {'wall_T4':>8s} {'wall_mon':>8s} {'speedup':>8s} "
          f"{'lam':>9s} {'Sigma':>13s} {'cg':>3s} {'rec':>3s} "
          f"{'eps':>8s}", flush=True)
    for r in rows:
        if "error" in r:
            print(f"{r['d']:>3d} {r['R']:>3d}  ERROR: {r['error'][:60]}", flush=True)
            continue
        print(f"{r['d']:>3d} {r['R']:>3d} {r['alpha_truth']:>10.6f} "
              f"{r['alpha_polish']:>10.6f} {r['diff_polish_vs_truth']:>9.1e} "
              f"{r['wall_t4_ms']:>7.1f}ms {r['wall_mon_ms']:>7.1f}ms "
              f"{r['speedup']:>7.2f}x  {r['n_W_active']:>3d}/{r['n_W_full']:<4d} "
              f"{r['n_sigma_final']:>4d}/{r['n_sigma_full']:<6d} "
              f"{r['n_cg_iter']:>3d} {r['n_recovery']:>3d} "
              f"{r['epsilon_total']:>8.1e}",
              flush=True)

    with open("_tier4_v2_e2e_results.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print("\nWrote _tier4_v2_e2e_results.json", flush=True)
