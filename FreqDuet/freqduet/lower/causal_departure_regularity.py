"""Causal soft regularity cost for stop-level holding actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def causal_two_sided_holding_target_s(
    *,
    forward_headway_s: float | None,
    follower_departure_gap_s: float | None,
    action_cap_s: float,
) -> float | None:
    """Return the clipped holding action that balances two causal gaps.

    Holding the current bus by ``a`` seconds changes the local pair to
    ``(forward + a, follower - a)``. The midpoint action therefore minimizes
    the unconstrained symmetric quadratic loss used by the two-sided
    regularity objective. Invalid evidence fails closed instead of encoding a
    synthetic recommendation.
    """
    cap = float(action_cap_s)
    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError("action_cap_s must be finite and positive")

    values = (forward_headway_s, follower_departure_gap_s)
    if any(value is None for value in values):
        return None
    forward, follower = (float(value) for value in values)
    if not (np.isfinite(forward) and np.isfinite(follower)):
        return None
    if forward < 0.0 or follower < 0.0:
        return None
    return float(np.clip(0.5 * (follower - forward), 0.0, cap))


def causal_two_sided_action_excess_cost(
    *,
    action_s: float,
    target_action_s: float,
    target_headway_s: float,
    cost_cap: float,
) -> float:
    """Return the action-dependent term of the two-sided quadratic loss."""
    action = float(action_s)
    target_action = float(target_action_s)
    target_headway = float(target_headway_s)
    cap = float(cost_cap)
    if not all(np.isfinite(value) for value in (
            action, target_action, target_headway, cap)):
        raise ValueError("regularity action cost inputs must be finite")
    if action < 0.0 or target_action < 0.0:
        raise ValueError("regularity actions must be non-negative")
    if target_headway <= 0.0 or cap <= 0.0:
        raise ValueError("target headway and cost cap must be positive")
    return float(min(
        ((action - target_action) / target_headway) ** 2,
        cap,
    ))


def causal_two_sided_zero_hold_regret_cost(
    *,
    action_s: float,
    target_action_s: float,
    target_headway_s: float,
    cost_cap: float,
) -> float:
    """Return positive regularity regret relative to taking no hold action.

    Actions that are at least as close to the causal balancing target as zero
    holding have no constraint cost. This leaves their passenger and dispatch
    tradeoff to the reward critic while penalizing actions that make the local
    two-sided regularity term worse than no intervention.
    """
    action_cost = causal_two_sided_action_excess_cost(
        action_s=action_s,
        target_action_s=target_action_s,
        target_headway_s=target_headway_s,
        cost_cap=cost_cap,
    )
    zero_hold_cost = causal_two_sided_action_excess_cost(
        action_s=0.0,
        target_action_s=target_action_s,
        target_headway_s=target_headway_s,
        cost_cap=cost_cap,
    )
    return float(max(action_cost - zero_hold_cost, 0.0))


def holding_action_efficiency_gate(
    *,
    action_s: float,
    action_scale_s: float,
    penalty: float,
) -> float:
    """Discount regularity gain by the holding time used to obtain it."""
    action = float(action_s)
    scale = float(action_scale_s)
    weight = float(penalty)
    if not all(np.isfinite(value) for value in (action, scale, weight)):
        raise ValueError("holding efficiency inputs must be finite")
    if action < 0.0 or scale <= 0.0 or weight < 0.0:
        raise ValueError(
            "holding efficiency requires action >= 0, scale > 0, penalty >= 0")
    action_fraction = float(np.clip(action / scale, 0.0, 1.0))
    return float(1.0 / (1.0 + weight * action_fraction))


def fleet_utilization_pressure(
    *,
    utilization: float,
    pressure_start: float,
    pressure_full: float,
    exponent: float = 1.0,
) -> float:
    """Return smooth fleet pressure from causal in-service utilization."""
    value = float(utilization)
    start = float(pressure_start)
    full = float(pressure_full)
    power = float(exponent)
    if not all(np.isfinite(x) for x in (value, start, full, power)):
        raise ValueError("fleet utilization pressure inputs must be finite")
    if value < 0.0 or not 0.0 <= start < full or power <= 0.0:
        raise ValueError(
            "fleet utilization pressure requires utilization >= 0, "
            "0 <= start < full, exponent > 0")
    normalized = float(np.clip((value - start) / (full - start), 0.0, 1.0))
    return float(normalized ** power)


def holding_fleet_efficiency_gate(
    *,
    action_s: float,
    action_scale_s: float,
    penalty: float,
    fleet_pressure: float,
) -> float:
    """Discount holding gain only in proportion to current fleet pressure."""
    pressure = float(fleet_pressure)
    if not np.isfinite(pressure) or not 0.0 <= pressure <= 1.0:
        raise ValueError("fleet pressure must lie in [0, 1]")
    return holding_action_efficiency_gate(
        action_s=action_s,
        action_scale_s=action_scale_s,
        penalty=float(penalty) * pressure,
    )


def holding_target_pressure(
    *,
    target_action_s: float,
    action_scale_s: float,
    exponent: float,
) -> float:
    """Return a dimensionless causal target-magnitude pressure."""
    target = float(target_action_s)
    scale = float(action_scale_s)
    power = float(exponent)
    if not all(np.isfinite(value) for value in (target, scale, power)):
        raise ValueError("target pressure inputs must be finite")
    if target < 0.0 or scale <= 0.0 or power < 0.0:
        raise ValueError(
            "target pressure requires target >= 0, scale > 0, exponent >= 0")
    target_fraction = float(np.clip(target / scale, 0.0, 1.0))
    return float(target_fraction ** power) if power > 0.0 else 1.0


def holding_target_preserving_fleet_efficiency_gate(
    *,
    target_action_s: float,
    action_scale_s: float,
    penalty: float,
    fleet_pressure: float,
    target_pressure_exponent: float,
) -> float:
    """Discount a state's gain without changing its action-bin ordering."""
    weight = float(penalty)
    pressure = float(fleet_pressure)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("opportunity-cost penalty must be finite and non-negative")
    if not np.isfinite(pressure) or not 0.0 <= pressure <= 1.0:
        raise ValueError("fleet pressure must lie in [0, 1]")
    target_pressure = holding_target_pressure(
        target_action_s=target_action_s,
        action_scale_s=action_scale_s,
        exponent=target_pressure_exponent,
    )
    return float(1.0 / (1.0 + weight * pressure * target_pressure))


@dataclass(frozen=True)
class DepartureRegularityContext:
    """Immutable action-time evidence used when the transition settles."""

    forward_headway_s: float | None
    target_headway_s: float | None
    evidence_source: str
    evidence_valid: bool
    follower_departure_gap_s: float | None
    follower_evidence_source: str
    follower_evidence_valid: bool


@dataclass(frozen=True)
class DepartureRegularityResult:
    cost: float
    reward_adjustment: float
    predicted_headway_s: float
    predicted_follower_gap_s: float
    normalized_deviation: float
    normalized_overshoot: float
    baseline_loss: float
    post_action_loss: float
    evidence_valid: bool
    follower_evidence_valid: bool


class CausalDepartureRegularityCost:
    """Shape predicted post-hold regularity without changing the action.

    The context is captured after mandatory service, immediately before the
    holding action is returned to the environment. Holding delays the current
    departure, so ``forward_headway + action`` is the causal one-step departure
    headway implied by that action. The frozen context prevents a later bus
    state from leaking into transition settlement.
    """

    EVIDENCE_MODE = "pre_action_departure_v6"
    EVIDENCE_SOURCE = "matched_departure_event"
    CMDP_ABSOLUTE = "cmdp_absolute"
    FORWARD_INCREMENTAL_REWARD = "forward_incremental_reward"
    AVL_TWO_SIDED_INCREMENTAL_REWARD = (
        "avl_two_sided_incremental_reward")

    def __init__(
        self,
        *,
        enabled: bool = False,
        cost_weight: float = 0.0,
        reward_weight: float = 0.0,
        tolerance_fraction: float = 0.0,
        cost_cap: float = 1.0,
        objective_mode: str = CMDP_ABSOLUTE,
        evidence_mode: str = EVIDENCE_MODE,
    ) -> None:
        self.enabled = bool(enabled)
        self.cost_weight = float(cost_weight)
        self.reward_weight = float(reward_weight)
        self.tolerance_fraction = float(tolerance_fraction)
        self.cost_cap = float(cost_cap)
        self.objective_mode = str(objective_mode).strip().lower()
        self.evidence_mode = str(evidence_mode).strip().lower()
        if self.evidence_mode != self.EVIDENCE_MODE:
            raise ValueError(
                "causal departure regularity requires "
                "evidence_mode=pre_action_departure_v6")
        if not np.isfinite(self.cost_weight) or self.cost_weight < 0.0:
            raise ValueError("cost_weight must be finite and non-negative")
        if not np.isfinite(self.reward_weight) or self.reward_weight < 0.0:
            raise ValueError("reward_weight must be finite and non-negative")
        allowed_modes = {
            self.CMDP_ABSOLUTE,
            self.FORWARD_INCREMENTAL_REWARD,
            self.AVL_TWO_SIDED_INCREMENTAL_REWARD,
        }
        if self.objective_mode not in allowed_modes:
            raise ValueError("unknown causal departure regularity objective")
        if self.enabled:
            if (self.objective_mode == self.CMDP_ABSOLUTE
                    and self.cost_weight <= 0.0):
                raise ValueError(
                    "enabled cmdp_absolute regularity requires cost_weight > 0")
            if (self.objective_mode != self.CMDP_ABSOLUTE
                    and self.reward_weight <= 0.0):
                raise ValueError(
                    "enabled incremental regularity requires reward_weight > 0")
        if not np.isfinite(self.tolerance_fraction) or not (
                0.0 <= self.tolerance_fraction < 1.0):
            raise ValueError("tolerance_fraction must lie in [0, 1)")
        if not np.isfinite(self.cost_cap) or self.cost_cap <= 0.0:
            raise ValueError("cost_cap must be finite and positive")

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any] | None
    ) -> "CausalDepartureRegularityCost":
        cfg = dict(config or {})
        return cls(
            enabled=cfg.get("enable", False),
            cost_weight=cfg.get("cost_weight", 0.0),
            reward_weight=cfg.get("reward_weight", 0.0),
            tolerance_fraction=cfg.get("tolerance_fraction", 0.0),
            cost_cap=cfg.get("cost_cap", 1.0),
            objective_mode=cfg.get("objective_mode", cls.CMDP_ABSOLUTE),
            evidence_mode=cfg.get("evidence_mode", cls.EVIDENCE_MODE),
        )

    def capture(
        self,
        *,
        forward_headway_s: float | None,
        target_headway_s: float | None,
        evidence_source: str | None,
        follower_departure_gap_s: float | None = None,
        follower_evidence_source: str | None = None,
    ) -> DepartureRegularityContext:
        source = str(evidence_source or "unavailable").strip().lower()
        forward = self._optional_finite_nonnegative(forward_headway_s)
        target = self._optional_finite_positive(target_headway_s)
        valid = (
            source == self.EVIDENCE_SOURCE
            and forward is not None
            and target is not None
        )
        follower_source = str(
            follower_evidence_source or "unavailable").strip().lower()
        follower_gap = self._optional_finite_nonnegative(
            follower_departure_gap_s)
        follower_valid = (
            follower_source.startswith("same_time_avl_")
            and follower_gap is not None
        )
        return DepartureRegularityContext(
            forward_headway_s=forward,
            target_headway_s=target,
            evidence_source=source,
            evidence_valid=valid,
            follower_departure_gap_s=follower_gap,
            follower_evidence_source=follower_source,
            follower_evidence_valid=follower_valid,
        )

    def evaluate(
        self,
        context: DepartureRegularityContext | None,
        action_s: float,
    ) -> DepartureRegularityResult:
        if not self.enabled or context is None or not context.evidence_valid:
            return self._zero_result(False, False)

        forward = context.forward_headway_s
        target = context.target_headway_s
        if forward is None or target is None:
            return self._zero_result(False, context.follower_evidence_valid)
        action = max(float(action_s), 0.0)
        if not np.isfinite(action):
            raise ValueError("holding action must be finite")
        predicted = forward + action
        follower_gap = context.follower_departure_gap_s
        if self.objective_mode == self.AVL_TWO_SIDED_INCREMENTAL_REWARD:
            if follower_gap is None or not context.follower_evidence_valid:
                return self._zero_result(True, False)
            predicted_follower = max(float(follower_gap) - action, 0.0)
            baseline_loss = self._two_sided_loss(
                forward, float(follower_gap), target)
            post_loss = self._two_sided_loss(
                predicted, predicted_follower, target)
            deviation = 0.5 * (
                abs(predicted - target) / target
                + abs(predicted_follower - target) / target)
            overshoot = max(
                predicted - target,
                target - predicted_follower,
                0.0,
            ) / target
        else:
            predicted_follower = 0.0
            baseline_loss = self._one_sided_loss(forward, target)
            post_loss = self._one_sided_loss(predicted, target)
            deviation = abs(predicted - target) / target
            overshoot = max(predicted - target, 0.0) / target

        if self.objective_mode == self.CMDP_ABSOLUTE:
            raw_cost = min(post_loss, self.cost_cap)
            cost = float(self.cost_weight * raw_cost)
            reward_adjustment = 0.0
        else:
            improvement = float(np.clip(
                baseline_loss - post_loss,
                -self.cost_cap,
                self.cost_cap,
            ))
            cost = 0.0
            reward_adjustment = float(self.reward_weight * improvement)
        return DepartureRegularityResult(
            cost=cost,
            reward_adjustment=reward_adjustment,
            predicted_headway_s=float(predicted),
            predicted_follower_gap_s=float(predicted_follower),
            normalized_deviation=float(deviation),
            normalized_overshoot=float(overshoot),
            baseline_loss=float(baseline_loss),
            post_action_loss=float(post_loss),
            evidence_valid=True,
            follower_evidence_valid=context.follower_evidence_valid,
        )

    def _one_sided_loss(self, headway_s: float, target_s: float) -> float:
        deviation = abs(float(headway_s) - float(target_s)) / float(target_s)
        excess = max(deviation - self.tolerance_fraction, 0.0)
        return float(excess * excess)

    def _two_sided_loss(
        self, forward_s: float, follower_s: float, target_s: float
    ) -> float:
        return 0.5 * (
            self._one_sided_loss(forward_s, target_s)
            + self._one_sided_loss(follower_s, target_s)
        )

    @staticmethod
    def _zero_result(
        evidence_valid: bool, follower_evidence_valid: bool
    ) -> DepartureRegularityResult:
        return DepartureRegularityResult(
            cost=0.0,
            reward_adjustment=0.0,
            predicted_headway_s=0.0,
            predicted_follower_gap_s=0.0,
            normalized_deviation=0.0,
            normalized_overshoot=0.0,
            baseline_loss=0.0,
            post_action_loss=0.0,
            evidence_valid=bool(evidence_valid),
            follower_evidence_valid=bool(follower_evidence_valid),
        )

    @staticmethod
    def _optional_finite_nonnegative(value: float | None) -> float | None:
        if value is None:
            return None
        result = float(value)
        return result if np.isfinite(result) and result >= 0.0 else None

    @staticmethod
    def _optional_finite_positive(value: float | None) -> float | None:
        if value is None:
            return None
        result = float(value)
        return result if np.isfinite(result) and result > 0.0 else None
