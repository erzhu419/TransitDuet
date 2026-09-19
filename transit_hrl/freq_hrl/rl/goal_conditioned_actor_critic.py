"""Lean goal-conditioned facade over the validated asynchronous SMDP PPO core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
from torch import nn

from .smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig


GOAL_CONDITIONED_TRAINER_CONTRACT = (
    "upper_state_space_goal_lower_actuator_action_independent_smdp_ppo_v1"
)


@dataclass(frozen=True)
class GoalConditionedPPOConfig:
    """Configuration with goal and actuator spaces kept semantically distinct."""

    upper_state_dim: int
    lower_state_dim: int
    goal_dim: int
    action_dim: int
    hidden_dim: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    max_grad_norm: float = 1.0
    epochs: int = 4
    minibatch_size: int = 512
    init_log_std: float = -0.7
    device: str = "cpu"

    def __post_init__(self) -> None:
        dimensions = (
            self.upper_state_dim,
            self.lower_state_dim,
            self.goal_dim,
            self.action_dim,
            self.hidden_dim,
        )
        if any(int(value) < 1 for value in dimensions):
            raise ValueError("goal-conditioned dimensions must be positive")
        if not np.isfinite(float(self.learning_rate)) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        if not 0.0 < float(self.gamma) <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 <= float(self.gae_lambda) <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if not 0.0 < float(self.clip_ratio) < 1.0:
            raise ValueError("clip_ratio must be in (0, 1)")


class GoalConditionedActorCriticPPO(FrequencySeparatedActorCriticPPO):
    """Upper plans a goal; lower alone emits the physical actuator action.

    The inherited implementation supplies the already-tested asynchronous
    upper/lower trajectory accounting.  Every action-spectrum constraint,
    projection, promotion stream, and leakage penalty remains disabled.
    """

    def __init__(self, config: GoalConditionedPPOConfig) -> None:
        self.goal_config = config
        super().__init__(SMDPPPOConfig(
            upper_state_dim=int(config.upper_state_dim),
            lower_state_dim=int(config.lower_state_dim),
            upper_action_dim=int(config.goal_dim),
            lower_action_dim=int(config.action_dim),
            hidden_dim=int(config.hidden_dim),
            upper_learning_rate=float(config.learning_rate),
            lower_learning_rate=float(config.learning_rate),
            gamma=float(config.gamma),
            gae_lambda=float(config.gae_lambda),
            clip_ratio=float(config.clip_ratio),
            value_coef=float(config.value_coef),
            entropy_coef=float(config.entropy_coef),
            max_grad_norm=float(config.max_grad_norm),
            epochs=int(config.epochs),
            minibatch_size=int(config.minibatch_size),
            init_log_std=float(config.init_log_std),
            upper_cost_critic=False,
            lower_cost_critic=False,
            upper_dual_lr=0.0,
            lower_dual_lr=0.0,
            upper_projection_consistency_coef=0.0,
            lower_projection_consistency_coef=0.0,
            upper_deployment_frequency_dual_lr=0.0,
            lower_deployment_frequency_dual_lr=0.0,
            promotion_state_dim=0,
            hf_state_dim=0,
            device=str(config.device),
        ))

    def plan_goal(
        self,
        state: np.ndarray,
        *,
        sample: bool,
    ) -> dict[str, np.ndarray | float]:
        return self.act_upper(state, sample=sample)

    def act_conditioned(
        self,
        state: np.ndarray,
        *,
        sample: bool,
    ) -> dict[str, np.ndarray | float]:
        return self.act_lower(state, sample=sample)

    def mainline_contract(self) -> dict[str, Any]:
        return {
            "contract": GOAL_CONDITIONED_TRAINER_CONTRACT,
            "upper_output": "state_space_goal",
            "lower_output": "physical_actuator_action",
            "projector": "disabled",
            "promotion": "disabled",
            "leakage_loss": "disabled",
            "responsibility_gauge": "disabled",
            "upper_decision_rate": "semi_markov_macro_interval",
            "lower_decision_rate": "primitive_environment_step",
        }

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        for module in (
            self.upper_actor,
            self.lower_actor,
            self.upper_value,
            self.lower_value,
        ):
            yield from module.parameters()

    @property
    def trainable_parameter_count(self) -> int:
        return int(sum(
            parameter.numel()
            for parameter in self.trainable_parameters()
            if parameter.requires_grad
        ))
