#!/bin/bash
# Full autonomous Farkas sweep across d=4..16 on the pod.
# Results go to /workspace/sidon-autocorrelation/data/farkas_results/
#
# Strategy: for each d, do Farkas-CG bisection on t to find the tightest certifiable bound.
# Uses CG to keep MOSEK fast, flint fmpq for fast rational residual.

cd /workspace/sidon-autocorrelation
mkdir -p data/farkas_results
exec > data/farkas_results/sweep.log 2>&1

echo "=========================================="
echo "FARKAS SWEEP STARTED: $(date)"
echo "=========================================="

for d in 4 6 8 10 12 14 16; do
    echo ""
    echo "#############################################"
    echo "# d=$d, order=3"
    echo "#############################################"
    date

    # Bracket per d (t_lo <= t_hi; want to certify val(d) >= t_lo, and approach t_hi which is val_L3(d))
    case $d in
        4) T_LO=1.05; T_HI=1.10 ;;
        6) T_LO=1.10; T_HI=1.17 ;;
        8) T_LO=1.15; T_HI=1.20 ;;
        10) T_LO=1.20; T_HI=1.24 ;;
        12) T_LO=1.24; T_HI=1.27 ;;
        14) T_LO=1.25; T_HI=1.28 ;;
        16) T_LO=1.27; T_HI=1.31 ;;
    esac

    python3.11 -u -c "
import sys, time, json
sys.path.insert(0, '.')
try:
    sys.set_int_max_str_digits(10**7)
except AttributeError:
    pass

from certified_lasserre.farkas_cg import farkas_certify_cg

d = $d
order = 3
t_lo = $T_LO
t_hi = $T_HI
tol = 1e-4

print(f'=== bisect d={d}, order={order}, bracket=[{t_lo}, {t_hi}] ===', flush=True)
best_lb = None
best_result = None
probe_lo, probe_hi = t_lo, t_hi

for step in range(18):
    t_try = 0.5 * (probe_lo + probe_hi)
    print(f'\n[bisect {step+1}/18] probe t={t_try:.8f} (bracket=[{probe_lo:.6f}, {probe_hi:.6f}])', flush=True)
    t0 = time.time()
    try:
        res, _ = farkas_certify_cg(
            d=d, order=order, t_test=t_try,
            max_cg_iters=40, n_add_per_iter=15,
            max_denom_S=10**12, max_denom_mu=10**12,
            eig_margin=1e-10, nthreads=16,
            verbose=True,
        )
        print(f'  -> {res.status}  mu0={res.mu0_float:.3e}  '
              f'||r||_1={res.residual_l1_float:.3e}  '
              f'margin={res.safety_margin_float:+.3e}  '
              f'time={time.time()-t0:.1f}s', flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'  -> ERROR: {type(e).__name__}: {e}', flush=True)
        break

    if res.status == 'CERTIFIED':
        best_result = res
        best_lb = res.lb_rig
        probe_lo = t_try
    else:
        probe_hi = t_try
    if probe_hi - probe_lo < tol:
        break

if best_result:
    print(f'\nBEST CERTIFIED LB d={d}: {best_result.lb_rig_decimal}', flush=True)
    out = {
        'd': d, 'order': order,
        'lb_rig_decimal': best_result.lb_rig_decimal,
        'lb_rig_num_den': [int(best_lb.numerator), int(best_lb.denominator)],
        'mu0_float': best_result.mu0_float,
        'residual_l1_float': best_result.residual_l1_float,
        'safety_margin_float': best_result.safety_margin_float,
        'notes': best_result.notes,
    }
    import json as J
    with open(f'data/farkas_results/d{d}_o3.json', 'w') as f:
        J.dump(out, f, indent=2)
    print(f'  written to data/farkas_results/d{d}_o3.json', flush=True)
else:
    print(f'\nNO CERTIFIED LB found for d={d}', flush=True)
    sys.exit(2)  # signal bash to stop sweep
"

    rc=$?
    if [ $rc -eq 2 ]; then
        echo "d=$d could not be certified — stopping sweep"
        break
    fi
    echo "d=$d finished at $(date)"
done

echo ""
echo "=========================================="
echo "SWEEP FINISHED: $(date)"
echo "=========================================="
ls -la data/farkas_results/
