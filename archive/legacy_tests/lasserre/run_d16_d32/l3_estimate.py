"""Estimate L3 survivors for m=15, c_target=1.35, d0=2."""
import sys, os, time, math
import numpy as np

sys.path.insert(0, os.path.join('.', 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join('.', 'cloninger-steinerberger', 'cpu'))
from run_cascade import run_level0, process_parent_fused
from pruning import correction

m, n_half, c_target = 15, 1, 1.35

# L0 exhaustive
res = run_level0(n_half, m, c_target, verbose=False)
l0_surv = res['survivors']
print(f'L0: {len(l0_surv)} survivors (exact)', flush=True)

# L1 exhaustive
all_l1 = []
for parent in l0_surv:
    surv_i, _ = process_parent_fused(parent, m, c_target, 2)
    all_l1.append(surv_i)
all_l1 = np.vstack(all_l1)
print(f'L1: {len(all_l1):,} survivors (exact)', flush=True)

# L2: sample 10 random parents, collect survivors
rng = np.random.default_rng(77777)
l2_sample_idx = rng.choice(len(all_l1), 10, replace=False)
l2_sample = all_l1[l2_sample_idx]

l2_surv_list = []
l2_total_ch = 0
l2_total_surv = 0
for i, parent in enumerate(l2_sample):
    surv_i, n_ch = process_parent_fused(parent, m, c_target, 4)
    l2_total_ch += n_ch
    l2_total_surv += len(surv_i)
    if len(surv_i) > 0:
        # Only keep up to 500 survivors per parent to keep memory manageable
        l2_surv_list.append(surv_i[:500])
    print(f'  L2 [{i+1}/10]: {n_ch:,} ch -> {len(surv_i):,} surv', flush=True)

avg_surv_per = l2_total_surv / 10
est_l2_total = avg_surv_per * len(all_l1)
print(f'\nL2: {l2_total_surv:,} surv / {l2_total_ch:,} ch = {l2_total_surv/l2_total_ch*100:.2f}%', flush=True)
print(f'Est total L2 survivors: {est_l2_total:,.0f}', flush=True)

# Collect L2 survivors for L3 analysis
if not l2_surv_list:
    print('No L2 survivors, stopping.')
    sys.exit(0)

l2_collected = np.vstack(l2_surv_list)
print(f'\nCollected {len(l2_collected):,} L2 survivors for L3 estimation', flush=True)

# L3 children count estimate using Python ints
n_half_l3 = 8
d_child_l3 = 16
c_l3 = correction(m, n_half_l3)
thresh_l3 = c_target + c_l3 + 1e-9
x_cap = int(math.floor(m * math.sqrt(4 * d_child_l3 * thresh_l3)))
x_cap_cs = int(math.floor(m * math.sqrt(4 * d_child_l3 * c_target))) + 1
x_cap = min(x_cap, x_cap_cs)
print(f'\nL3: d_child={d_child_l3}, correction={c_l3:.6f}, x_cap={x_cap}', flush=True)

counts = []
for row in l2_collected:
    c = 1
    for val in row:
        lo = max(0, 2*int(val) - x_cap)
        hi = min(2*int(val), x_cap)
        eff = max(hi - lo + 1, 0)
        c *= eff
    counts.append(c)

counts_sorted = sorted(counts)
n = len(counts_sorted)
print(f'L3 children per L2 parent (from {n} collected):')
print(f'  min:    {counts_sorted[0]:,}')
print(f'  p10:    {counts_sorted[n//10]:,}')
print(f'  p25:    {counts_sorted[n//4]:,}')
print(f'  median: {counts_sorted[n//2]:,}')
print(f'  p75:    {counts_sorted[3*n//4]:,}')
print(f'  p90:    {counts_sorted[9*n//10]:,}')
print(f'  max:    {counts_sorted[-1]:,}')

under_500m = sum(1 for c in counts if c <= 500_000_000)
under_100m = sum(1 for c in counts if c <= 100_000_000)
under_10m = sum(1 for c in counts if c <= 10_000_000)
print(f'  under 10M:  {under_10m}/{n}')
print(f'  under 100M: {under_100m}/{n}')
print(f'  under 500M: {under_500m}/{n}')
sys.stdout.flush()

# Sort by count, try to process the 5 lightest
order = sorted(range(len(counts)), key=lambda i: counts[i])
n_try = 0
l3_total_ch = 0
l3_total_surv = 0
for rank in range(min(10, len(order))):
    idx = order[rank]
    expected = counts[idx]
    if expected > 200_000_000:
        print(f'  L3 parent rank={rank}: {expected:,} children - skipping (>200M)', flush=True)
        continue
    parent = l2_collected[idx]
    t0 = time.time()
    surv_i, n_ch = process_parent_fused(parent, m, c_target, n_half_l3)
    elapsed = time.time() - t0
    l3_total_ch += n_ch
    l3_total_surv += len(surv_i)
    n_try += 1
    rate = len(surv_i)/n_ch*100 if n_ch > 0 else 0
    print(f'  L3 parent rank={rank}: {n_ch:,} ch -> {len(surv_i):,} surv ({rate:.2f}%) [{elapsed:.1f}s]', flush=True)

if l3_total_ch > 0:
    l3_rate = l3_total_surv / l3_total_ch
    median_ch = counts_sorted[n//2]
    est_l3_children = float(median_ch) * est_l2_total
    est_l3_survivors = l3_rate * est_l3_children
    print(f'\nL3 ESTIMATE ({n_try} parents processed):')
    print(f'  survival rate:        {l3_rate*100:.4f}%')
    print(f'  median ch/parent:     {median_ch:,}')
    print(f'  est total L3 children:  {est_l3_children:.3e}')
    print(f'  est total L3 survivors: {est_l3_survivors:.3e}')
else:
    print('\nNo L3 parents were processable.')
