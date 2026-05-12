"""Debug SCS GPU/indirect/cuDSS module loading."""
import scs
import scs._scs_cudss as cudss_mod
import scs._scs_indirect as indirect_mod
import numpy as np
from scipy import sparse as sp
import inspect

print(f"SCS {scs.__version__}")
print(f"int size: {scs.__sizeof_int__}")
print(f"float size: {scs.__sizeof_float__}")

# Check module selection logic
print("\n=== _select_scs_module source ===")
print(inspect.getsource(scs._select_scs_module))

# Test with various matrix sizes
for n in [2, 5, 10]:
    m = 2 * n
    A = sp.random(m, n, density=0.5, format="csc", dtype=np.float64)
    A.indices = A.indices.astype(np.int32)
    A.indptr = A.indptr.astype(np.int32)
    b = np.random.randn(m)
    c = np.random.randn(n)
    data = {"A": A, "b": b, "c": c}
    cone = {"l": m}

    print(f"\n=== n={n}, m={m} ===")

    for name, mod in [("direct", scs._scs_direct),
                       ("indirect", indirect_mod),
                       ("cudss", cudss_mod)]:
        try:
            solver = mod.SCS(data, cone, verbose=False, max_iters=100,
                             eps_abs=1e-5, eps_rel=1e-5)
            sol = solver.solve()
            print(f"  {name}: {sol['info']['status']}, "
                  f"obj={sol['info']['pobj']:.4f}, "
                  f"iters={sol['info']['iter']}")
        except Exception as e:
            print(f"  {name}: FAIL - {type(e).__name__}: {e}")

# Test with PSD cone
print("\n=== PSD cone test ===")
# 2x2 PSD: vec = [X00, sqrt(2)*X10, X11] (3 entries)
# min x1 s.t. [[x1, 0], [0, 1]] >> 0 => x1 >= 0
A_psd = sp.csc_matrix(
    (np.array([-1.0, -1.0], dtype=np.float64),
     np.array([0, 2], dtype=np.int32),
     np.array([0, 1, 1, 2], dtype=np.int32)),
    shape=(4, 2))  # 1 zero cone row + 3 PSD rows, 2 vars
b_psd = np.array([1.0, 0, 0, 0])  # x2 = 1, PSD slack = 0
c_psd = np.array([1.0, 0.0])  # min x1
data_psd = {"A": A_psd, "b": b_psd, "c": c_psd}
cone_psd = {"z": 1, "s": [2]}

for name, mod in [("direct", scs._scs_direct),
                   ("indirect", indirect_mod),
                   ("cudss", cudss_mod)]:
    try:
        solver = mod.SCS(data_psd, cone_psd, verbose=False, max_iters=1000,
                         eps_abs=1e-7, eps_rel=1e-7)
        sol = solver.solve()
        print(f"  {name}: {sol['info']['status']}, "
              f"x={sol['x']}, iters={sol['info']['iter']}")
    except Exception as e:
        print(f"  {name}: FAIL - {type(e).__name__}: {e}")
