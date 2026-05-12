import Lake
open Lake DSL

package sidon where
  leanOptions := #[⟨`autoImplicit, false⟩]

/-- Default build target: the multi-scale arcsine proof of `C_{1a} ≥ 1.292`,
    rooted at `lean/Sidon.lean` and transitively importing `Sidon.Defs` and
    `Sidon.MultiScale`. -/
@[default_target]
lean_lib Sidon where
  srcDir := "."
  roots := #[`Sidon]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "f897ebcf72cd16f89ab4577d0c826cd14afaafc7"
