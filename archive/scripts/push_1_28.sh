#!/bin/bash
# Aggressive push to certify val(d) > 1.2802 (beat Boyer-Li).
# Tries successively smaller t at a single d until something certifies.
#
# Usage:  push_1_28.sh  <d>  <t0,t1,t2,...>
# Example: push_1_28.sh 14 1.2830,1.2820,1.2810,1.2805,1.2803

cd /workspace/sidon-autocorrelation
mkdir -p data/farkas_push
d=${1:-14}
t_list=${2:-1.2830,1.2820,1.2810,1.2805,1.2803}

LOG=data/farkas_push/d${d}.log
exec > "$LOG" 2>&1

echo "======================================"
echo "PUSH TO 1.28+ at d=$d  started $(date)"
echo "t_list = $t_list"
echo "======================================"

IFS=',' read -ra T_ARRAY <<< "$t_list"
for t in "${T_ARRAY[@]}"; do
    echo ""
    echo "######################"
    echo "# d=$d t_test=$t"
    echo "######################"
    date
    python3.11 -u -c "
import sys, time, json
sys.path.insert(0, '.')
try:
    sys.set_int_max_str_digits(10**7)
except AttributeError:
    pass
from certified_lasserre.farkas_cg import farkas_certify_cg

d = $d
t_test = $t
t0 = time.time()
res, _ = farkas_certify_cg(
    d=d, order=3, t_test=t_test,
    max_cg_iters=50, n_add_per_iter=25,
    max_denom_S=10**12, max_denom_mu=10**12,
    eig_margin=1e-10, nthreads=16,
    verbose=True,
)
print(f'\n=== d={d} t={t_test}: {res.status}  mu0={res.mu0_float:.3e}  '
      f'||r||_1={res.residual_l1_float:.3e}  '
      f'margin={res.safety_margin_float:+.3e}  '
      f'time={time.time()-t0:.1f}s ===', flush=True)
if res.status == 'CERTIFIED':
    import json as J
    with open(f'data/farkas_push/d{d}_t{t_test:.5f}.json', 'w') as f:
        J.dump({
            'd': d, 't_test': t_test, 'order': 3,
            'lb_rig_decimal': res.lb_rig_decimal,
            'lb_rig_num_den': [int(res.lb_rig.numerator), int(res.lb_rig.denominator)],
            'mu0_float': res.mu0_float,
            'residual_l1_float': res.residual_l1_float,
            'safety_margin_float': res.safety_margin_float,
            'notes': res.notes,
        }, f, indent=2)
    print(f'*** CERTIFIED d={d} t={t_test} ***  val({d}) > {res.lb_rig_decimal}', flush=True)
    sys.exit(0)
else:
    print(f'  not certified at t={t_test}, trying smaller', flush=True)
    sys.exit(3)
"
    rc=$?
    if [ $rc -eq 0 ]; then
        echo ""
        echo "SUCCESS at d=$d t=$t — stopping push."
        break
    fi
done

echo ""
echo "=== FINISHED d=$d push at $(date) ==="
ls -la data/farkas_push/
