"""Core domain-agnostic Freq-HRL interfaces and utilities."""

from .action_decoders import (
    CausalSmoothstepMacroPlan,
    CausalZeroDCMacroProjector,
)
from .causal_joint_frequency_projector import (
    BUDGET_MODES,
    CausalJointFrequencyProjector,
)
from .causal_receding_horizon_joint_projector import (
    AffineQuadraticBallProjector,
    CausalRecedingHorizonJointProjector,
)
from .causal_terminal_reserve_projector import CausalTerminalReserveProjector
from .diagnostics import FrequencyDiagnostics, binned_mutual_information
from .leakage import (
    ActionEffectOperator,
    CausalLeakageRewardShaper,
    CausalLowFrequencyEffectProjector,
    CausalRollingBandTracker,
    CumulativeActionEffectOperator,
    LeakageRegularizer,
    evaluate_rms_leakage_budget,
)
from .phase0 import (
    PHASE0_REQUIRED_FIELDS,
    PHASE0_SCHEMA_VERSION,
    Phase0TraceLogger,
    load_phase0_records,
    validate_phase0_record_schema,
)
from .promotion_gate import CausalPromotionGate
from .receding_horizon_responsibility import (
    CausalRecedingHorizonResponsibilityPlanner,
    future_rolling_mean_system,
)
from .reward import RewardAttributionAccumulator
from .responsibility_gauge import (
    CausalAuditOptimalMacroGaugeFixer,
    CausalAuditAlignedGaugeFixer,
    CausalFeasibilityNormalizedAuditProjectionFixer,
    CausalGaugeFixer,
    CausalMacroHoldAuditGaugeFixer,
    CausalSmoothMacroGaugeFixer,
    CausalStreamingAuditProjectionFixer,
    canonical_responsibility_trace,
)
from .router import FrequencyRouter
from .shared_core_audit import audit_shared_training_core
from .spec import (
    FrozenFreqHRLSpec,
    default_spec,
    validate_claim_freeze,
    validate_frequency_features,
    validate_lower_policy_state,
    validate_shared_core_paths,
    validate_upper_policy_state,
)
from .stream_adapter import BinnedExogenousStreamAdapter, MultiEntityBinnedStream
from .types import ExogenousBin, FrequencyFeatures, PromotionSignal

__all__ = [
    "ActionEffectOperator",
    "AffineQuadraticBallProjector",
    "BinnedExogenousStreamAdapter",
    "BUDGET_MODES",
    "CausalJointFrequencyProjector",
    "CausalRecedingHorizonJointProjector",
    "CausalTerminalReserveProjector",
    "CausalPromotionGate",
    "CausalRecedingHorizonResponsibilityPlanner",
    "CausalLeakageRewardShaper",
    "CausalGaugeFixer",
    "CausalAuditOptimalMacroGaugeFixer",
    "CausalAuditAlignedGaugeFixer",
    "CausalFeasibilityNormalizedAuditProjectionFixer",
    "CausalMacroHoldAuditGaugeFixer",
    "CausalSmoothMacroGaugeFixer",
    "CausalStreamingAuditProjectionFixer",
    "CausalLowFrequencyEffectProjector",
    "CausalRollingBandTracker",
    "CausalSmoothstepMacroPlan",
    "CausalZeroDCMacroProjector",
    "CumulativeActionEffectOperator",
    "ExogenousBin",
    "FrequencyDiagnostics",
    "FrequencyFeatures",
    "FrequencyRouter",
    "FrozenFreqHRLSpec",
    "LeakageRegularizer",
    "MultiEntityBinnedStream",
    "PHASE0_REQUIRED_FIELDS",
    "PHASE0_SCHEMA_VERSION",
    "Phase0TraceLogger",
    "PromotionSignal",
    "RewardAttributionAccumulator",
    "audit_shared_training_core",
    "binned_mutual_information",
    "canonical_responsibility_trace",
    "default_spec",
    "evaluate_rms_leakage_budget",
    "future_rolling_mean_system",
    "load_phase0_records",
    "validate_claim_freeze",
    "validate_frequency_features",
    "validate_lower_policy_state",
    "validate_phase0_record_schema",
    "validate_shared_core_paths",
    "validate_upper_policy_state",
]
