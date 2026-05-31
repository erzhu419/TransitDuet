"""RL trainers for domain-agnostic Freq-HRL experiments."""

from .dual_actor_critic import (
    DualActorCriticPPO,
    DualPPOConfig,
    TrajectoryBatch,
)

__all__ = [
    "DualActorCriticPPO",
    "DualPPOConfig",
    "TrajectoryBatch",
]
