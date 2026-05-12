# GPU/CUDA SCS for Lasserre SDPs (H100) (DEAD)

Custom CUDA kernel design for the discrete cascade prover targeting NVIDIA H100 SXM 80 GB (132 SMs, HBM3 3.35 TB/s). Engineering work complete and validated against CPU reference; route closed because the Lasserre/cascade plateau itself stalled below the multi-scale arcsine result.

## Goal

Accelerate L3 (d=32) and L4 (d=64) cascade levels (which on 32-core CPU Numba require ~16 h and ~3 days respectively), preserving rigorous integer arithmetic so a single missed survivor invalidates the proof.

## Architecture (consolidated from 12 design notes)

| Element | Choice | Rationale |
|---------|--------|-----------|
| Parent dispatch | Persistent blocks + global atomic counter | True work-stealing under 1000:1 child-count variance; cheap (~1 atomic per ~50K children) |
| Inner enumeration | Serial Gray code within a block | Incremental O(d) conv updates, quick-check temporal locality, subtree pruning all preserved |
| Intra-child work | Warp-parallel (32 / 64 threads) | Conv update + window scan account for 35% + 45% of time; warp parallelism speeds these without breaking serial structure |
| Arithmetic | int32 conv, int32 window sums, int64 threshold compare from precomputed table | Zero FP on hot path; soundness-by-construction |
| Compiler flags | `-fmad=false --ftz=false --prec-div=true --prec-sqrt=true` | Eliminate FMA rounding drift in any residual FP |
| Memory layout | Templated shared-mem sizes per d; threshold table in L2 cache (too large for smem at d=64) | ~14-22 KB shared/block → 7-10 blocks/SM |

Per-cell wall-clock budget at d=64: ~157 cycles weighted average (85% QC-hit, 15% full window-scan). At L3->L4 with c_target=1.4: measured 65 parents/s on 1x H100 (target was 1000+); 700M parents projected to ~125 days on 1 GPU, ~2 days on 64 GPUs at ~$12K. 10 verified optimizations identified (template shared mem, SURV_CAP reduction, hierarchical window scan etc.) for a projected 3-5x speedup, not pursued.

## Validation outcome

CPU/GPU survivor sets match bit-exactly at L0-L2 (small-d exact match); random-parent spot-checks at L3/L4 confirmed `GPU_survivors ⊇ CPU_survivors` after canonicalization. The kernel is correct.

## Why the route closed

1. **Lasserre plateau:** val(d=8) = 1.205 is a structural ceiling; bypass needs d >= 14. At d=16 the Pólya gap projection ([coarse_lp_bnb.md](coarse_lp_bnb.md)) still needs `R ~ 25` with ~24M LP variables — H100 alone cannot bridge that without a fundamentally different relaxation. User constraint forbids large SDP / B&B / Lasserre routes.
2. **Survivor blow-up:** Raw GPU output is 58x larger than unique survivors at L1->L2 and ~200,000x larger at L3->L4 (no on-GPU dedup); a chunked-processing wrapper is necessary but not implemented.
3. **Multi-scale arcsine route delivers more:** Rigorous C_{1a} >= 1.292 ([multiscale_arcsine.md](multiscale_arcsine.md)) without any GPU compute, formalised in Lean 4.

## Final disposition

Architecture, kernel design, memory layout, optimization shortlist, and PROOF_PLAN.md are retained as the contingency plan if d=16 cascade B&B is reattempted. No CUDA code remains in the active tree.

## References

- [../proof_outline.md](../proof_outline.md), [../formalization.md](../formalization.md)
- [cascade_estimator.md](cascade_estimator.md), [coarse_lp_bnb.md](coarse_lp_bnb.md), [lasserre.md](lasserre.md)
