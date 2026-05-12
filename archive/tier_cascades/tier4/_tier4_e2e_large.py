"""Tier-4 on larger instances where the speedup should emerge.

The local 8 GB GPU and laptop CPU limit how big we can go, but the
trend should be visible by d=14, R=8 or d=16, R=6.

Also benchmarks an OPTIMIZED rigorize at lower mpmath precision (prec=80
instead of 200), since prec=200 was 92ms on d=12 R=8 — well beyond the
~1e-9 noise from MOSEK.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasserre.polya_lp.tier4.driver import tier4_solve, monolithic_solve


def run_one(d, R, rigorize_prec=80):
    print(f"\n=== d={d} R={R} (rigorize_prec={rigorize_prec}) ===", flush=True)

    sol_mon, wall_mon = monolithic_solve(d, R, use_z2=True, tol=1e-9)
    print(f"  MONOLITHIC: alpha={sol_mon.alpha:.10f}  wall={wall_mon*1000:.1f}ms",
          flush=True)

    t0 = time.time()
    res = tier4_solve(
        d, R, use_z2=True,
        coarse_backend="highs_ipm", coarse_tol=1e-5,
        polish_tol=1e-9,
        rigorize_prec=rigorize_prec,
        verbose=False,
    )
    wall_t4 = time.time() - t0

    print(f"  TIER 4    : alpha_rig={res.alpha_rigorous:.10f}  "
          f"alpha_pol={res.alpha_polish:.10f}  wall={wall_t4*1000:.1f}ms",
          flush=True)
    print(f"    setup={res.wall_setup_s*1000:.1f}ms  "
          f"coarse={res.wall_coarse_s*1000:.1f}ms  "
          f"polish_build={res.wall_polish_build_s*1000:.1f}ms  "
          f"polish_solve={res.wall_polish_solve_s*1000:.1f}ms  "
          f"verify={res.wall_verify_s*1000:.1f}ms  "
          f"rigorize={res.wall_rigorize_s*1000:.1f}ms",
          flush=True)
    print(f"    active: lambda {res.n_W_active}/{res.n_W_full}  "
          f"cbeta {res.n_cbeta_active}/{res.n_cbeta_full}  "
          f"verify_max_v={res.verify_max_violation:.2e}",
          flush=True)
    speedup = wall_mon / wall_t4
    diff = abs(res.alpha_polish - sol_mon.alpha)
    rig_ok = res.alpha_rigorous is not None and res.alpha_rigorous <= sol_mon.alpha + 1e-9
    print(f"    diff(polish-truth)={diff:.2e}  rig<=truth:{rig_ok}  speedup={speedup:.2f}x",
          flush=True)
    return dict(
        d=d, R=R, prec=rigorize_prec,
        alpha_truth=sol_mon.alpha, alpha_polish=res.alpha_polish,
        alpha_rigorous=res.alpha_rigorous, diff=diff,
        wall_mon_ms=wall_mon * 1000, wall_t4_ms=wall_t4 * 1000,
        speedup=speedup,
        n_W_full=res.n_W_full, n_W_active=res.n_W_active,
        breakdown=dict(
            setup=res.wall_setup_s * 1000,
            coarse=res.wall_coarse_s * 1000,
            polish_build=res.wall_polish_build_s * 1000,
            polish_solve=res.wall_polish_solve_s * 1000,
            verify=res.wall_verify_s * 1000,
            rigorize=res.wall_rigorize_s * 1000,
        ),
    )


if __name__ == "__main__":
    grid = [
        (12, 8), (12, 10),
        (14, 6), (14, 8),
        (16, 4), (16, 6), (16, 8),
    ]
    rows = []
    for d, R in grid:
        try:
            rows.append(run_one(d, R, rigorize_prec=80))
        except Exception as e:
            import traceback
            traceback.print_exc()
            rows.append(dict(d=d, R=R, error=str(e)))

    print("\n--- SUMMARY ---", flush=True)
    print(f"{'d':>3s} {'R':>3s} {'a_truth':>10s} {'a_pol':>10s} {'a_rig':>10s} "
          f"{'wall_T4':>9s} {'wall_mon':>9s} {'speedup':>8s} {'lam_act':>8s} "
          f"{'coarse':>7s} {'polish':>7s} {'rig':>6s}", flush=True)
    for r in rows:
        if "error" in r:
            print(f"{r['d']:>3d} {r['R']:>3d}  ERROR: {r['error'][:60]}", flush=True)
            continue
        bd = r['breakdown']
        print(f"{r['d']:>3d} {r['R']:>3d} {r['alpha_truth']:>10.6f} "
              f"{r['alpha_polish']:>10.6f} {r['alpha_rigorous']:>10.6f} "
              f"{r['wall_t4_ms']:>8.1f}ms {r['wall_mon_ms']:>8.1f}ms "
              f"{r['speedup']:>7.2f}x {r['n_W_active']:>3d}/{r['n_W_full']:<4d} "
              f"{bd['coarse']:>6.0f}ms {bd['polish_solve']:>6.0f}ms "
              f"{bd['rigorize']:>5.0f}ms", flush=True)

    with open("_tier4_e2e_large_results.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print("\nWrote _tier4_e2e_large_results.json", flush=True)
