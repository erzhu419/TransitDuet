"""Core domain-agnostic Freq-HRL interfaces and utilities."""

from .diagnostics import FrequencyDiagnostics, binned_mutual_information
from .leakage import (
    ActionEffectOperator,
    CausalLeakageRewardShaper,
    CausalLowFrequencyEffectProjector,
    CumulativeActionEffectOperator,
    LeakageRegularizer,
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
from .stream_adapter import BinnedExogenousStreamAdapter, MultiEntityBinnedStream
from .types import ExogenousBin, FrequencyFeatures, PromotionSignal

__all__ = [
    "ActionEffectOperator",
    "BinnedExogenousStreamAdapter",
    "CausalPromotionGate",
    "CausalLeakageRewardShaper",
    "CausalLowFrequencyEffectProjector",
    "CumulativeActionEffectOperator",
    "ExogenousBin",
    "FrequencyDiagnostics",
    "FrequencyFeatures",
    "FrequencyRouter",
    "LeakageRegularizer",
    "MultiEntityBinnedStream",
    "PHASE0_REQUIRED_FIELDS",
    "PHASE0_SCHEMA_VERSION",
    "Phase0TraceLogger",
    "PromotionSignal",
    "RewardAttributionAccumulator",
    "binned_mutual_information",
    "load_phase0_records",
    "validate_phase0_record_schema",
]
