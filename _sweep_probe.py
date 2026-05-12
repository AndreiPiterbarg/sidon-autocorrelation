
import sys, time, json, os
sys.path.insert(0, r"C:/Users/andre/OneDrive - PennO365/Desktop/compact_sidon")
from lasserre.polya_lp.runner import run_one
d = int(sys.argv[1]); R = int(sys.argv[2])
t0 = time.time()
try:
    rec, _, sol = run_one(d=d, R=R, use_z2=True, solver='mosek', verbose=False)
    wall = time.time() - t0
    out = dict(d=d, R=R, alpha=rec.alpha, wall=wall,
               n_eq=rec.n_eq, n_vars=rec.n_vars, nnz=rec.n_nonzero_A,
               build_s=rec.build_wall_s, solve_s=rec.solve_wall_s,
               status='OK')
except MemoryError as e:
    out = dict(d=d, R=R, status='OOM', error='MemoryError',
               wall=time.time()-t0)
except Exception as e:
    msg = str(e)[:300]
    status = 'OOM' if 'large memory' in msg or 'space' in msg.lower() else 'ERROR'
    out = dict(d=d, R=R, status=status, error=msg,
               wall=time.time()-t0)
print('RESULT:', json.dumps(out, default=str), flush=True)
