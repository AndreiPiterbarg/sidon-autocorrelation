# path_a_holder/

Path-A Hölder-inequality lane for $C_{1a}$. **Status: DEAD** —
asymmetric obstruction unresolved (see
`project_path_a_unconditional_status.md` in MEMORY); 7 attacks tried,
no improvement on CS17's 1.2802. Restricted-Hölder gives only a
conditional 1.37842 under Hyp_R (`project_restricted_holder.md`).

## Subfolders

- **hausdorff/** — 2D Hausdorff moment probes (v2–v5, test, simple_check).
- **l3/** — $L^3$ estimates: scan, rescue tests, true-rescue search,
  $d{=}4,S{=}20$ snapshot.
- **asymmetric_search/** — asymmetric-counterexample search v1/v2/v3
  with run logs and v3 result JSON.
- **mo_conjecture_29/** — MO Conjecture 2.9 attempts: direct, low-$M$,
  smooth-low-$M$, search-low-$M$, asym-low-$M$, findings.
- **krein_markov/** — Krein-Markov probe v1/v2.
- **misc/** — `_mo214_augmented_sdp.py` (MO 2.14 augmented SDP),
  `_option_A_search_counterexample.py` (Option-A counterexample search).

## Notes

No external entry points; all scripts are exploratory probes kept for
audit. See referenced MEMORY entries for narrative status.
