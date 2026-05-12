# cascade_estimator

Cascade throughput estimation, pruning prototypes, and ad-hoc cell-level probes
used while tuning the F to FN to Q to QN to L cascade chain.

## Files

- `_cascade_estimate.py` + `_cascade_*.log` — main estimator script and run logs
  for n=1/2, m=20 chain configurations (`F`, `FQL`, `FQL1` variants on `c128`).
- `_pruning_proto.{py,log,json}` — pruning prototype: explores tighter
  per-stage closure rules; JSON has per-cell prune stats.
- `_sdp_hardcell_probe.{py,json}` — single-cell SDP probe on cascade-residue
  hard cells (post-QN survivors at d=10).
- `_sonine_cross_term.{py,json}` — Sonine cross-term identity probe used in
  multi-scale arcsine analysis (cross-Bessel coefficient bounds).
- `_route_c_holder_chain.py`, `_route_c_white_transfer.py` — Route C probes
  (Holder chain, White-2022 L^2 transfer); kept here as they were drafted
  alongside cascade tuning. Both negative results.

## Subdirs (run snapshots)

- `cascade_est_n1m20_c1281/summary.json` — n=1, m=20, c=1281 estimate snapshot.
- `coarse_cascade_estimate_20260509_130255/summary.json` — coarse cascade
  estimate, 2026-05-09 run.

## Status

All exploratory; no rigorous bound depends on these files. Historical
narrative writeups were archived prior to publication.
