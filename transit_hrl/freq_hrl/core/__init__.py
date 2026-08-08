"""Core domain-agnostic Freq-HRL interfaces and utilities."""

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
from .reward import RewardAttributionAccumulator
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
    "BinnedExogenousStreamAdapter",
    "CausalPromotionGate",
    "CausalLeakageRewardShaper",
    "CausalLowFrequencyEffectProjector",
    "CausalRollingBandTracker",
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
    "default_spec",
    "evaluate_rms_leakage_budget",
    "load_phase0_records",
    "validate_claim_freeze",
    "validate_frequency_features",
    "validate_lower_policy_state",
    "validate_phase0_record_schema",
    "validate_shared_core_paths",
    "validate_upper_policy_state",
]
