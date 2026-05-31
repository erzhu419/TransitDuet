"""General Frequency-Separated HRL core.

This package is intentionally independent from the existing FreqDuet transit
runner.  Domain code should adapt its own environment state/action schema to
these interfaces instead of importing transit-specific simulator objects here.
"""

from .core.leakage import CausalLeakageRewardShaper, LeakageRegularizer
from .core.phase0 import Phase0TraceLogger
from .core.promotion_gate import CausalPromotionGate
from .core.reward import RewardAttributionAccumulator
from .core.router import FrequencyRouter
from .core.stream_adapter import BinnedExogenousStreamAdapter
from .encoders.causal_ema import CausalEMAEncoder
from .encoders.causal_fourier import CausalFourierEncoder

__all__ = [
    "BinnedExogenousStreamAdapter",
    "CausalEMAEncoder",
    "CausalFourierEncoder",
    "CausalLeakageRewardShaper",
    "CausalPromotionGate",
    "FrequencyRouter",
    "LeakageRegularizer",
    "Phase0TraceLogger",
    "RewardAttributionAccumulator",
]
