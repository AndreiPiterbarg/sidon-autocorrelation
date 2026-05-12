# helpers

Python utilities glue between cascade kernel runs.

- `generate_l0.py` — emit the level-0 (root) cell list for a given (d, S).
- `split_parents.py` — split a parent-cell file into N shards for
  multi-pod runs.
- `merge_survivors.py` — merge per-shard survivor files back into one
  list after a multi-pod run.
- `run_chunked.py` — Python driver that chunks a level into batches sized
  to GPU memory, calls the prover binary per chunk, then merges output.

CPU-only; no CUDA dependency.
