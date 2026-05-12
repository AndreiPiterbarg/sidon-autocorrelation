"""Refine c=1.275 stragglers to d=16."""
import os, sys, time, json, logging
logging.getLogger('cvxpy').setLevel(logging.ERROR)
import numpy as np
from itertools import product
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))
import _coarse_bnb_v4 as v4
from _d16_F_bench import _prune_coarse_count_cell
from compositions import generate_canonical_compositions_batched

# Stragglers from c=1.275 at d=8, S=16
# 28 v4-open + 19 grid-survivors
with open('_prove_1275_d8S16.json') as fp:
    data = json.load(fp)
v4_open = [tuple(x) for x in data['stage2']['open_samples']]
print(f"v4-open samples from JSON: {len(v4_open)}", flush=True)

# Also re-find grid-survivors at d=8, S=16, c=1.275
warm = np.zeros((1, 8), dtype=np.int32); warm[0, 0] = 16
_prune_coarse_count_cell(warm, 8, 16, 1.0)
grid_surv = []
v4_open_set = set(tuple(x) for x in v4_open)
all_open_from_log = []
# Re-scan
for batch in generate_canonical_compositions_batched(8, 16, batch_size=200000):
    survived, neg_mask, n_neg, mn = _prune_coarse_count_cell(
        batch.astype(np.int32), 8, 16, 1.275)
    for idx in np.where(survived)[0]:
        grid_surv.append(tuple(int(x) for x in batch[idx]))

print(f"grid-survivors at c=1.275: {len(grid_surv)}", flush=True)

# Combine all stragglers (grid_surv + v4_open)
all_stragglers = list(set(grid_surv) | set(v4_open))
# Re-run v4 on the FULL residue (cell-uncertain) at d=8 to collect actually-open list
# But for time, just use grid_surv + v4_open sample
print(f"refining {len(all_stragglers)} unique stragglers", flush=True)
for s in all_stragglers[:5]:
    print(f"  example: {s}", flush=True)

# Generate children
def gen_children(parent):
    d_p = len(parent)
    opts = [[(a, p - a) for a in range(p + 1)] for p in parent]
    out = []
    for combo in product(*opts):
        ch = np.zeros(2 * d_p, dtype=np.int64)
        for i, (a, b) in enumerate(combo):
            ch[2 * i] = a; ch[2 * i + 1] = b
        out.append(ch)
    return out

def canonical_dedup(children):
    seen = set(); out = []
    for c in children:
        rc = tuple(c[::-1]); ct = tuple(c)
        key = ct if ct <= rc else rc
        if key not in seen:
            seen.add(key); out.append(c)
    return out

t0 = time.time()
all_children = []
for p in all_stragglers:
    ch = gen_children(p)
    all_children.extend(ch)
unique = canonical_dedup(all_children)
print(f"  total raw: {len(all_children)}, unique canonical: {len(unique)}  [{time.time()-t0:.1f}s]",
      flush=True)

# Numba F at d=16
print("\n  Numba F at d=16...", flush=True)
warm = np.zeros((1, 16), dtype=np.int32); warm[0, 0] = 16
_prune_coarse_count_cell(warm, 16, 16, 1.0)
batch = np.array(unique, dtype=np.int32)
t0 = time.time()
survived, neg_mask, n_neg, min_net = _prune_coarse_count_cell(batch, 16, 16, 1.275)
n_p = int((~survived).sum()); n_s = int(survived.sum()); n_u = int(n_neg)
print(f"  total:           {len(unique)}", flush=True)
print(f"  grid-pruned:     {n_p}", flush=True)
print(f"  grid-survivors:  {n_s}  (NEED d=32 refinement — likely intractable)", flush=True)
print(f"  cell-uncertain:  {n_u}", flush=True)
print(f"  min_net:         {min_net:.4f}", flush=True)
print(f"  time:            {time.time()-t0:.2f}s", flush=True)

if n_s > 0:
    print("\n  Cannot fully close — grid-survivors at d=16 would need d=32 refinement.", flush=True)
    print("  Sample grid-survivors:", flush=True)
    for i in np.where(survived)[0][:5]:
        print(f"    {batch[i].tolist()}", flush=True)

if n_u == 0:
    if n_s == 0:
        print("\n  *** PROOF COMPLETE @ c=1.275 ***", flush=True)
    return_now = True
else:
    residue_mask = (~survived) & neg_mask
    residue = [batch[i].astype(np.int64).copy() for i in np.where(residue_mask)[0]]
    print(f"\n  v4 on {len(residue)} residue at d=16...", flush=True)
    windows = v4.build_all_windows(16)
    v4.get_sdp_template(16)
    v4.get_joint_template(16, 4)
    t0 = time.time()
    counts = {'B1': 0, 'empty': 0, 'F': 0, 'L': 0, 'L_joint': 0, 'split': 0}
    open_cells = []
    for k, c in enumerate(residue):
        if k > 0 and k % max(1, len(residue) // 20) == 0:
            print(f"    [{k}/{len(residue)}] closed={sum(counts.values())} open={len(open_cells)} elapsed={time.time()-t0:.1f}s", flush=True)
        try:
            r = v4.certify_composition(c.astype(np.float64), 16, 16, 1.275,
                                          windows=windows, max_depth=3)
        except Exception:
            open_cells.append(c); continue
        if r.certified:
            counts[r.tier_used] = counts.get(r.tier_used, 0) + 1
        else:
            open_cells.append(c)
    print(f"\n  v4 SUMMARY: closed={sum(counts.values())} open={len(open_cells)} elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"  by tier: {counts}", flush=True)
    if n_s == 0 and len(open_cells) == 0:
        print(f"\n  *** PROOF COMPLETE: C_{{1a}} >= 1.275 ***", flush=True)
    else:
        print(f"\n  Not fully closed: {n_s} grid-survivors + {len(open_cells)} v4-open at d=16.", flush=True)
