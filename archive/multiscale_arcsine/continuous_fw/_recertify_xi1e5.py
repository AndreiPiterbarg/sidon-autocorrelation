"""Re-run rigorous arb cert at xi_max=10^5 on the polished ν from
_continuous_nu_fw_N100_final.json."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))
_p = REPO
for _ in range(5):
    if (_p / "delsarte_dual").is_dir():
        sys.path.insert(0, str(_p))
        break
    _p = _p.parent

from _continuous_nu_fw import rigorous_verify

d = json.load(open(REPO / "_continuous_nu_fw_N100_final.json"))
atoms = d["pruned_atoms_for_rigor"]
print(f"Re-certifying {len(atoms)} atoms at xi_max=100000...")
r = rigorous_verify(atoms, xi_max=100_000, verbose=True)
print(f"\n  Rigorous M_cert >= {r['M_cert_lower']:.8f}")
print(f"  K_2 in [{r['K_2_lower']:.6f}, {r['K_2_upper']:.6f}]")
print(f"  K_2 tail upper: {r['K_2_tail_upper']:.4e}")
print(f"  S_1 <= {r['S_1_upper']:.6f}")
print(f"  min_G >= {r['min_G_lower']:.6f}")
print(f"  a_gain >= {r['a_gain_lower']:.6f}")

out = {**d, "rigorous_xi1e5": r}
with open(REPO / "_continuous_nu_fw_N100_final_xi1e5.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print(f"\nWrote {REPO / '_continuous_nu_fw_N100_final_xi1e5.json'}")
