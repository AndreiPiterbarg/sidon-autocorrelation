# agent_experiments/

Per-agent scratch from research sub-agents that explored Sidon $C_{1a}$ lower-bound
strategies. Files keep their `_agent_*` / `_remote_*` prefixes so the originating
research-agent is identifiable.

## Layout

- `path_agents/` — Path-A style agents (a/b/c/d, plus d8/d16 dimension probes,
  `_agent_min_d_*`, `_agent_pod_run_synthesis`). 19 files.
  - `a` = moment couplings
  - `b` = smoothed Path A
  - `c` = HypR hard-constrained (with run logs + scatter plot)
  - `d` = 2D Hausdorff 4-loc
  - `d8` / `d16` = dimension-specific feasibility/open-cell analyses
  - `min_d` = minimum-d certifying 1.281
  - `pod_run_synthesis` = pod-run aggregator output

- `agent9_audits/` — Agent-9 edge audit / eps-margin / extreme-S / kernel-lean
  random probes. 5 files.

- `remote_probes/` — `_remote_*.py` benchmark / correctness / diag / sweep /
  smoke / validation scripts run on remote pods. 19 files.

- `_novel_agents/` — Distinct novel-method agents, each in its own subdir
  (`probe.py`, `results.json`, `run.log`):
  - C1 Fisher info, C2 optimal transport, C3 copositive, C4 Hardy-Littlewood,
    C5 Stein method
  - F1 Daubechies wavelet, F2 heat kernel, F3 Wigner phase-space,
    F4 free probability, F5 de Branges
  - IMPL_lp_holder = LP-Hölder 4-point implementation

## Notes

These are exploratory probes; most are dead-ends documented in the project
memory under `project_*` notes. None of these scripts is in the rigorous proof
chain (see `lean/` for the verified bound).
