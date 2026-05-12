# coarse_bnb/

Coarse branch-and-bound and cascade-pruning attempts (archived;
superseded by the v5/v6 cascade — see project memory).

- `tests/` — `test_*.py` for pruning ideas, QP/box certs, subdivision,
  bivariate / trust-region, etc.
- `prove_sweep/` — `prove_*` and the matching `*sweep*` / `deep_*` /
  `feasibility_sweep` drivers.
- `param_studies/` — parameter analysis / convergence-frontier work.
- `coarse_drivers/` — the `coarse_*` gridpoint / l0 / max-rigorous
  drivers (1.10 / 1.30 sweeps).
- `cascade_misc/` — runtime/cloud estimators, quick smoke drivers,
  experiment-grade scripts.
