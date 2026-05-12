"""Publication-grade certification pipeline for val(d) >= target.

Goal: prove that `val(d) := min_{μ ∈ Δ_d} max_W μ^T M_W μ >= target` for
specified (d, target). Combined with Theorem 1.1 of `interval_bnb/THEOREM.md`
(`val(d) <= C_{1a}` for all d), this yields `C_{1a} >= target`.

The pipeline is **iterative cascade-BnB + SDP escalation**:

    while uncertified_volume > 0:
        run BnB cascade until in_flight stuck → instant-dump every box
        run SDP on every dumped box (K=0 then K=16 fallback)
        re-inject any SDP-failed box (split + re-enqueue)

Each iteration shrinks the uncertified set monotonically. Termination
is guaranteed because the cascade box-width threshold is strictly above 0
and SDP either certs or splits, so every box is eventually closed.

SOUNDNESS GUARANTEES (publication-grade):

  S1. Every initial box is accounted for in the persistent journal at
      every moment. State transitions are append-only.
  S2. SIGINT delivers an INSTANT dump (workers do NOT continue cascade
      processing) — every in_flight box at trigger time is captured.
      No "graceful processing" closures that would inflate the cascade
      cert count without on-disk evidence.
  S3. Box identity is by canonical (lo_int, hi_int) hash. Re-encountered
      boxes are deduplicated by hash; soundness is unaffected.
  S4. Cascade certs are recorded by VOLUME (sum of `Box.volume()` over
      all boxes the cascade closed). The BnB master prints periodic
      `[par]` lines with `cert=N` and `closed_volume=V`; we capture both.
  S5. SDP certs are recorded per-box with method (K=0 / K=16 / Z/2),
      MOSEK status, dual-Farkas λ*, residuals, peak RSS, wall time.
  S6. Volume balance: at end of pipeline, we verify
          initial_volume == sum(cascade_cert_volume across iters)
                          + sum(sdp_cert_volume across iters)
      with float tolerance 1e-9 × initial_volume.

PIPELINE FILE LAYOUT:

  runs/
    {tag}/
      config.json            # full env + seeds + git rev + library versions
      git_state.diff         # git rev-parse HEAD + diff
      initial_partition.npz  # the n_starter initial boxes (lo, hi as int)
      journal.jsonl          # append-only state-transition log
      iter_001/
        bnb.log              # BnB master stdout (full)
        bnb_metrics.jsonl    # parsed [par] log lines (timestamped)
        bnb_summary.json     # trigger reason, in_flight at trigger,
                             # cert_at_trigger, closed_vol_at_trigger
        dumps/
          worker_w*.npz      # per-worker local_stack at SIGINT
          master_queue.npz   # shared queue contents at SIGINT
          dump_errors.txt    # any worker that failed to dump (sentinel)
        sdp/
          pool_progress.log  # per-result line: box_idx, K, verdict, time
          per_box/{hash}.json # per-box SDP result (full diagnostic)
          summary.json       # cert/fail counts, fallback usage, total vol
        iter_summary.json    # iter rollup: cascade_vol, sdp_vol, fail_vol
      audit_final.json       # soundness verifier output (volume balance)
      certificate.json       # the publishable cert (if successful)

The directory is fully introspectable post-mortem. Append-only logs
(journal, bnb_metrics) survive any orchestrator crash. SDP per-box
results are individual JSON files keyed by box-hash so we never lose
data on partial-batch failures.
"""

__all__ = []
