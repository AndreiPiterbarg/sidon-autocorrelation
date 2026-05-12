import Mathlib

set_option linter.mathlibStandardSet false
set_option autoImplicit false
set_option relaxedAutoImplicit false

noncomputable section

namespace Lasserre

inductive EnhancedPSDMode where
  | full
  | sparse
  | dsos
  | bm
  deriving DecidableEq

inductive SearchMode where
  | direct
  | bisect
  | secant
  deriving DecidableEq

structure BaseConfig where
  d : Nat
  order : Nat
  bandwidth : Nat
  addUpperLoc : Bool := true

structure HighDConfig where
  d : Nat
  order : Nat
  bandwidth : Nat
  addUpperLoc : Bool := true
  maxCGRounds : Nat := 0
  maxAddPerRound : Nat := 0
  nBisect : Nat := 0

structure EnhancedConfig where
  d : Nat
  order : Nat
  bandwidth : Nat
  addUpperLoc : Bool := true
  psdMode : EnhancedPSDMode := .full
  searchMode : SearchMode := .bisect
  maxCGRounds : Nat := 0
  maxAddPerRound : Nat := 0
  nBisect : Nat := 0

def HighDConfig.base (cfg : HighDConfig) : BaseConfig :=
  { d := cfg.d
    order := cfg.order
    bandwidth := cfg.bandwidth
    addUpperLoc := cfg.addUpperLoc }

def EnhancedConfig.base (cfg : EnhancedConfig) : BaseConfig :=
  { d := cfg.d
    order := cfg.order
    bandwidth := cfg.bandwidth
    addUpperLoc := cfg.addUpperLoc }

structure Window where
  ell : Nat
  sLo : Nat

structure Clique where
  bins : Finset Nat

structure SolverOutput where
  lowerBound : Real
  activeWindows : Finset Nat := {}
  nMoments : Nat := 0

structure VerifiedCertificate where
  primalWitnessChecked : Prop
  dualOrInfeasibilityChecked : Prop
  residualBoundsChecked : Prop
  spectralBoundsChecked : Prop

def currentRecord : Real := (6401 : Real) / 5000

def provesNewBest (out : SolverOutput) : Prop :=
  currentRecord < out.lowerBound

axiom valD : Nat -> Real
axiom C1a : Real

axiom CoreEncodingMatchesCode : BaseConfig -> Prop
axiom ReducedMomentSetMatchesCode : BaseConfig -> Prop
axiom CliqueSystemMatchesCode : BaseConfig -> Prop
axiom ConsistencySystemMatchesCode : BaseConfig -> Prop
axiom WindowSystemMatchesCode : BaseConfig -> Prop
axiom ConstraintGenerationMatchesCode : BaseConfig -> Prop
axiom SearchWorkflowMatchesCode : BaseConfig -> Prop
axiom NumericalCertificationMatchesCode : BaseConfig -> Prop

axiom RestrictedRelaxationSound : BaseConfig -> Prop
axiom ConvergedWithoutMissedViolations : BaseConfig -> SolverOutput -> Prop
axiom VerifiedLowerBound : BaseConfig -> SolverOutput -> Prop
axiom DiscreteBridgeEstablished : BaseConfig -> Prop
axiom ContinuousBridgeEstablished : BaseConfig -> Prop

axiom SolverOutputMatchesHighDCode : HighDConfig -> SolverOutput -> Prop
axiom SolverOutputMatchesEnhancedCode : EnhancedConfig -> SolverOutput -> Prop
axiom SolverOutputMatchesCGCode : BaseConfig -> SolverOutput -> Prop
axiom SolverOutputMatchesFusionCode : BaseConfig -> SolverOutput -> Prop

end Lasserre

end
