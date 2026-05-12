# archive/multiscale_arcsine

Scratch from the multi-scale arcsine kernel attack line. This was the
**working line** that produced the rigorous lower bound
`C_{1a} >= 1651/1280 = 1.28984` (+0.0150 over MV's 1.2742).

The productionised pipeline lives in `delsarte_dual/grid_bound_alt_kernel/`.
The rigorous Lean certificate lives in `lean/Sidon/`. Files here are
parallel agent runs, probes, and dead-ends — keep for provenance.

## Layout

- **`M_attacks/`** (32 files) — Multi-scale variants `_M1` ... `_M15`:
  3-component mixes, n-component, arc+Chebyshev, arc+Askey, mm-multiscale,
  continuous-density, free-delta, alternating, higher-nqp, arb lifts,
  frequency-tuned, arc+bspline, scale-ratio, Chebyshev composition.
- **`K_kernels/`** (36 files) — Single-kernel probes `_agent_K20` ... `_agent_K32`:
  Wendland, Kaiser-Bessel, cosine-power, B-spline-gap, Tukey, two-param Beta,
  K26 multiscale-arcsine + convergence, poly-rooted phi, CG sign-uncertainty,
  MO surrogate, direct K-hat, Selberg extremal, SDP K-hat opt. Also
  `_K2_highprec`, `_K26_full_sweep_reopt`, `_K26_push_deeper`, `_K26_reopt_G`.
- **`master_k26/`** (33 files) — `_master_k26_*`: the master K26 sweep —
  alt-master, continuous, finer-scan, hybrid v1..v5, large-N, n-scale,
  pilot, recompute-pure, reopt-full, z1, plus caches (`*.npz`) and logs.
- **`continuous_fw/`** (15 files) — Continuous measure Frank-Wolfe and
  atomic-nu (`_continuous_nu_fw*`, `_recertify_xi1e5*`,
  `_theorem4_atomic_nu*`, `_n3_*`).
- **`verifiers/`** (27 files) — Audits and verifiers: `_V1/V2/V6/V7`,
  `_w2/w5/w9/w13` smoke + Bochner + qp + cross-Bessel checks,
  `verify_w13_*` (v3/v4/final), `_f7_multiscale_audit`,
  `_cross_validate_v3`, `_convergence_probe`.
- **`N_S_rigorous/`** (10 files) — Rigorous-lift companions:
  `_N1_rigorous_cross_bessel`, `_N4_multiscale_bspline`, `_S_sweep_bench`,
  `_cohn_elkies_128_v5`, `_kernel_probe_helper`, `_L_n2m20_c128.json`,
  `_multiscale_arcsine_rigorous.json`.

## Context

See `MEMORY.md` projects: `multiscale_arcsine_lead`, `multiscale_reopt_G`,
`multiscale_RIGOROUS_lift`, `multiscale_audit_v1`, `K26_audit_correction`,
`K26_DEAD` (retracted), `master_synth_2026_05_11`.

Before: 157 entries (155 files + `__pycache__/` + empty `continuous_nu_fw_work/`).
After: 153 files in 6 subfolders + this README.
