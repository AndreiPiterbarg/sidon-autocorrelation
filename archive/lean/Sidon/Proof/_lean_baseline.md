## Lean Baseline (pre-flight, 2026-05-09)

### Toolchain & deps
- Lean: `leanprover/lean4:v4.24.0` (per `lean/lean-toolchain`).
- Lake: `Lake 5.0.0-src+797c613` (matches Lean 4.24.0).
- Mathlib pin: rev `f897ebcf72cd16f89ab4577d0c826cd14afaafc7` (`lakefile.lean` + `lake-manifest.json`).
- elan toolchains installed: 4.24.0, 4.28.0, 4.29.0, 4.29.1; project pins 4.24.0.

### Build status
- `lake build Sidon.Proof.PostFilterF` -> Build completed successfully (7356 jobs).
- Wall: 4m 15s (full incremental replay, mostly cached oleans).
- Errors: 0. Warnings: 4 (unused vars + a `ring` -> `ring_nf` Try-this in `StepFunction.lean:161`).
- Existing oleans cover all `Sidon/Proof/*.lean` modules.

### Lakefile targets
- `Sidon` (default, root `Sidon.lean`), `SidonMonolithic` (legacy), `LasserreAudit`, `IntervalBnB`, `SOSDual`.
- Project options: `autoImplicit := false`; per-file: `maxHeartbeats 8000000`, `maxRecDepth 4000`.

### Existing definitions (already in code)
From `Sidon/Defs.lean`:
- `autoconvolution_ratio (f : R -> R) : R` -- ||f*f||_inf / (int f)^2 (the C_{1a} target).
- `discrete_autoconvolution {d} (a : Fin d -> R) (k : N) : R` = `conv[k]` (sum_{i+j=k} a_i a_j).
- `test_value (n m) (c : Fin (2n) -> N) (l s_lo) : R` -- TV on fine grid (window sum / (4n l)).
- `bin_masses (f) (n) : Fin (2n) -> R` -- per-bin integral.
- `canonical_discretization (f) (n m) : Fin (2n) -> N` -- floor-rounding of cumulative mass.
- `canonical_cumulative_distribution (f n m k) : N`.
- `is_valid_child` and inductive `CascadePruned m c_target correction n_half c`.
- `convolution_nonneg` lemma.

From `Sidon/Proof/StepFunction.lean`:
- `step_function (n m) (c : Fin (2n) -> N) : R -> R` -- piecewise-const a_i = c_i/m on bin i.
- `step_function_nonneg`, `step_function_support`, `step_function_integrable`.
- `integral_step_function` (= 1 when sum c_i = 4nm).
- `convolution_at_grid_point`: (g*g)(y_k) = (1/(4n m^2)) * conv_int[k] at y_k = -1/2 + (k+1) delta.
- `test_value_continuous` variant (heights = 4n * bin_masses).

From `Sidon/Proof/Foundational.lean`: F1-F15 chain on `canonical_discretization` (telescope, monotone, sum = 4nm, etc.).

### Notes for upcoming M-chain file
- Bin index `bin_i`, breakpoints `t_k`, conv array `conv[k]` are NOT defined as named entities; they live inline inside `step_function` (uses `floor((x + 1/4)/delta)`) and `discrete_autoconvolution k`. Grid point `y_k = -1/2 + (k+1)*delta` appears only in `convolution_at_grid_point`.
- `M(f)` (max of f*f over |t| <= 1/2) is NOT a named def; the bound is stated via `autoconvolution_ratio` (eLpNorm top of f*f).
- `C_1a` is the constant 32/25 referenced as the target of `autoconvolution_ratio_ge_32_25` in `FinalResult.lean`.
- File sizes (LOC): PostFilterF 953, PostFilterQ 1003, PostFilterL 509, TightDiscretization 1236, StepFunction 397, Defs 137.

BUILD_STATUS: clean | with 0 errors

Definitions ALREADY available:
- `autoconvolution_ratio`, `discrete_autoconvolution`, `test_value`, `test_value_continuous`
- `bin_masses`, `canonical_discretization`, `canonical_cumulative_distribution`
- `step_function` (+ nonneg/support/integrable/integral lemmas)
- `convolution_at_grid_point`, `convolution_nonneg`
- `is_valid_child`, `CascadePruned` (inductive)
- F1-F15 foundational chain (telescope, monotone, sum = 4nm)
