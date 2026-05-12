"""Verify `cascade_all_pruned_p` at (n_half=2, m=20, c_target=32/25):
every composition of S=160 into d=4 nonneg bins must be P-pruned.

Total: C(160+3, 3) = 762,376 compositions.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'cloninger-steinerberger', 'cpu'))
from compositions import generate_compositions_batched
from _M1_bench import prune_P

n_half, m = 2, 20
d = 2 * n_half  # 4
S = 4 * n_half * m  # 160
c_target = 32 / 25  # 1.28

# JIT warmup
warm = np.array([[S, 0, 0, 0]], dtype=np.int32)
_ = prune_P(warm, n_half, m, c_target)

t0 = time.time()
total = 0
n_pruned = 0
n_survived = 0
survivor_examples = []
for batch in generate_compositions_batched(d, S, batch_size=200_000):
    batch = batch.astype(np.int32)
    survived = prune_P(batch, n_half, m, c_target)
    total += len(batch)
    n_p = int((~survived).sum())
    n_s = int(survived.sum())
    n_pruned += n_p
    n_survived += n_s
    if n_s > 0 and len(survivor_examples) < 5:
        survivor_indices = np.where(survived)[0][:5]
        for idx in survivor_indices:
            survivor_examples.append(tuple(int(x) for x in batch[idx]))
elapsed = time.time() - t0

print(f"\n=== cascade_all_pruned_p verification at (n={n_half}, m={m}, c={c_target}) ===")
print(f"d={d}, S=4nm={S}, c_target={c_target}={32}/{25}")
print(f"  total compositions:     {total:,}")
print(f"  pruned by P:            {n_pruned:,}  ({100*n_pruned/total:.4f}%)")
print(f"  survivors:              {n_survived:,}  ({100*n_survived/total:.4f}%)")
print(f"  elapsed:                {elapsed:.2f}s")
if n_survived == 0:
    print(f"\n  *** AXIOM VERIFIED ***")
    print(f"  Every composition is P-pruned at L0 (n=2, m=20, c=32/25).")
else:
    print(f"\n  *** AXIOM NOT VERIFIED ***")
    print(f"  Survivor examples (first 5):")
    for s in survivor_examples:
        print(f"    {s}")
