"""Quick smoke bench for v2 vs v3 vs v4 on pod with VERBOSE per-step logging.
Runs ONE tiny config (d0=2, S=30, c=1.20) without SDP, so it finishes in seconds.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                  'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, os.path.dirname(__file__))

print(f"[{time.strftime('%H:%M:%S')}] start, importing...", flush=True)
from run_cascade_coarse_v2 import run_cascade as rv2
print(f"[{time.strftime('%H:%M:%S')}] v2 imported", flush=True)
from run_cascade_coarse_v3 import run_cascade as rv3
print(f"[{time.strftime('%H:%M:%S')}] v3 imported", flush=True)
from run_cascade_coarse_v4 import run_cascade as rv4
print(f"[{time.strftime('%H:%M:%S')}] v4 imported", flush=True)

d0, S, c = 2, 30, 1.20

print(f"\n=== d0={d0}, S={S}, c={c}, use_sdp=False ===", flush=True)

t0=time.time()
print(f"[{time.strftime('%H:%M:%S')}] running v2...", flush=True)
r2 = rv2(d0=d0, S=S, c_target=c, max_levels=3, n_workers=1, verbose=False)
t2=time.time()-t0
print(f"[{time.strftime('%H:%M:%S')}] v2 done in {t2:.1f}s: pa={r2.get('proven_at')} L0={r2['l0']['survivors']} levels={[(lv['level'],lv['survivors']) for lv in r2['levels']]}", flush=True)

t0=time.time()
print(f"[{time.strftime('%H:%M:%S')}] running v3...", flush=True)
r3 = rv3(d0=d0, S=S, c_target=c, max_levels=3, n_workers=1, verbose=False)
t3=time.time()-t0
print(f"[{time.strftime('%H:%M:%S')}] v3 done in {t3:.1f}s: pa={r3.get('proven_at')} L0={r3['l0']['survivors']} levels={[(lv['level'],lv['survivors']) for lv in r3['levels']]}", flush=True)

t0=time.time()
print(f"[{time.strftime('%H:%M:%S')}] running v4 (use_sdp=False)...", flush=True)
r4 = rv4(d0=d0, S=S, c_target=c, max_levels=3, n_workers=1, verbose=False, use_joint=True, use_sdp=False)
t4=time.time()-t0
print(f"[{time.strftime('%H:%M:%S')}] v4 done in {t4:.1f}s: pa={r4.get('proven_at')} L0={r4['l0']['survivors']} levels={[(lv['level'],lv['survivors']) for lv in r4['levels']]}", flush=True)
L0_v4 = r4['l0']
print(f"      L0 v4 layer counts: NO_cert={L0_v4.get('n_certified_NO',0)} J_cert={L0_v4.get('n_certified_J',0)} L_cert={L0_v4.get('n_certified_L',0)} hard={L0_v4.get('n_uncertified',0)}", flush=True)

print(f"\n[{time.strftime('%H:%M:%S')}] DONE", flush=True)
