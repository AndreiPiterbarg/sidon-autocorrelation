import json, math
from functools import reduce
from operator import mul

d = json.load(open('_local_d8S16_dep6.json'))
samples = d['stage2']['open_samples']
N = len(samples)
print(f"N samples = {N}")

uniform = [2]*8

palin = 0
maxminusmin = []
nzeros = []
l2_uniform = []
prodchildren = []

for s in samples:
    c = s['c']
    assert sum(c) == 16 and len(c) == 8
    # palindrome
    if c == c[::-1]:
        palin += 1
    maxminusmin.append(max(c) - min(c))
    nzeros.append(sum(1 for x in c if x == 0))
    l2_uniform.append(math.sqrt(sum((ci - 2)**2 for ci in c)))
    prodchildren.append(reduce(mul, [ci+1 for ci in c], 1))

print(f"Palindromic: {palin}/{N} = {100*palin/N:.1f}%")
print(f"max-min: min={min(maxminusmin)} max={max(maxminusmin)} mean={sum(maxminusmin)/N:.2f}")
from collections import Counter
print(f"max-min distribution: {sorted(Counter(maxminusmin).items())}")
print(f"# zeros distribution: {sorted(Counter(nzeros).items())}")
print(f"L2 to uniform: min={min(l2_uniform):.3f} max={max(l2_uniform):.3f} mean={sum(l2_uniform)/N:.3f}")
print(f"Pi(c_i+1): min={min(prodchildren)} max={max(prodchildren)} mean={sum(prodchildren)/N:.1f} median={sorted(prodchildren)[N//2]}")

# Top 5 hardest by largest |bound|
ranked = sorted(samples, key=lambda s: -abs(s.get('bound', 0)))[:5]
print("\n--- Top 5 hardest by |bound| ---")
hardest_children = []
for s in ranked:
    c = s['c']
    pc = reduce(mul, [ci+1 for ci in c], 1)
    print(f"c={c}  bound={s['bound']:.4f}  tier={s['tier']}  depth={s['depth']}  Pi(c+1)={pc}")
    hardest_children.append({'c': c, 'bound': s['bound'], 'tier': s['tier'], 'children_count': pc})

# Distribution of bounds and tiers
print("\n--- Tier counts ---")
print(sorted(Counter(s['tier'] for s in samples).items()))
print("--- Depth distribution ---")
print(sorted(Counter(s['depth'] for s in samples).items()))

avg_children = sum(prodchildren) / N
total_estimate = 1434 * avg_children
print(f"\nAvg Pi(c_i+1) = {avg_children:.2f}")
print(f"Total child count estimate = 1434 * {avg_children:.2f} = {total_estimate:.0f}")

out = {
    'n_samples': N,
    'palindromic_pct': 100*palin/N,
    'maxminusmin_mean': sum(maxminusmin)/N,
    'maxminusmin_dist': dict(Counter(maxminusmin)),
    'nzeros_dist': dict(Counter(nzeros)),
    'l2_uniform_mean': sum(l2_uniform)/N,
    'l2_uniform_min': min(l2_uniform),
    'l2_uniform_max': max(l2_uniform),
    'prodchildren_min': min(prodchildren),
    'prodchildren_max': max(prodchildren),
    'prodchildren_mean': avg_children,
    'prodchildren_median': sorted(prodchildren)[N//2],
    'open_count_full': 1434,
    'total_child_estimate': total_estimate,
    'hardest5': hardest_children,
    'tier_dist': dict(Counter(s['tier'] for s in samples)),
    'depth_dist': dict(Counter(s['depth'] for s in samples)),
}
json.dump(out, open('_agent_d8_open_analysis.json','w'), indent=2)
print("\nSaved _agent_d8_open_analysis.json")
