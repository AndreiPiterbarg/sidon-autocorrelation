"""Try T=100000 at dps=30 for C_11 to get expected runtime/value."""
import time
import mpmath as mp

mp.mp.dps = 30
DELTA = [mp.mpf(138) / 1000, mp.mpf(55) / 1000, mp.mpf(25) / 1000]


def integrand(i, j, xi):
    a = mp.pi * DELTA[i] * xi
    b = mp.pi * DELTA[j] * xi
    return mp.besselj(0, a) ** 2 * mp.besselj(0, b) ** 2


def compute_Cij(i, j, T):
    a_max = max(DELTA[i], DELTA[j])
    val = mp.mpf(0)
    X_inner = mp.mpf(50)
    val += mp.quad(lambda xi: integrand(i, j, xi), [0, X_inner])
    k_start = int(mp.ceil(X_inner * a_max))
    k_end = int(mp.floor(T * a_max))
    CHUNK = 1000
    prev = X_inner
    for k in range(k_start, k_end + 1, CHUNK):
        nxt = mp.mpf(min(k + CHUNK, k_end)) / a_max
        if nxt <= prev:
            continue
        val += mp.quad(lambda xi: integrand(i, j, xi), [prev, nxt])
        prev = nxt
    if prev < T:
        val += mp.quad(lambda xi: integrand(i, j, xi), [prev, T])
    return 2 * val


t = time.time()
v = compute_Cij(0, 0, mp.mpf(100000))
print(f"C_11 (T=1e5, dps=30) = {mp.nstr(v, 14)}  ({time.time()-t:.1f} s)")

t = time.time()
v = compute_Cij(2, 2, mp.mpf(100000))
print(f"C_33 (T=1e5, dps=30) = {mp.nstr(v, 14)}  ({time.time()-t:.1f} s)")
print("ref C_33 ~ 22.98780")
