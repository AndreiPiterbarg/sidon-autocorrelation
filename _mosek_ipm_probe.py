
import sys, time
sys.path.insert(0, r'C:/Users/andre/OneDrive - PennO365/Desktop/compact_sidon')
from lasserre.polya_lp.runner import run_one
import json

d = int(sys.argv[1]); R = int(sys.argv[2])
t0 = time.time()
rec, _, _ = run_one(d=d, R=R, use_z2=True, solver='mosek', verbose=False)
wall = time.time() - t0
out = dict(d=d, R=R, alpha=rec.alpha, wall=wall, n_eq=rec.n_eq, n_vars=rec.n_vars,
           build_s=rec.build_wall_s, solve_s=rec.solve_wall_s)
print('RESULT:', json.dumps(out, default=str), flush=True)
