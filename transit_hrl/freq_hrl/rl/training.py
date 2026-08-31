"""Domain-agnostic training loops for dual-level Freq-HRL policies."""

from __future__ import annotations

import copy
from typing import Any, Callable, Iterable

import numpy as np
import torch

from .dual_actor_critic import DualActorCriticPPO, TrajectoryBatch
from .checkpoint_selection import (
    RobustValidationCheckpointSelector,
    StateAlignedLexicographicCheckpointSelector,
)
from .joint_actor_critic import (
    JointActorCriticPPO,
    JointTrajectoryBatch,
    concat_joint_batches,
)
from .smdp_actor_critic import (
    FrequencySeparatedActorCriticPPO,
    HierarchicalTrajectoryBatch,
    concat_hierarchical_batches,
)

RolloutFn = Callable[[DualActorCriticPPO, int, bool], tuple[TrajectoryBatch | None, dict[str, Any]]]
ObjectiveFn = Callable[[dict[str, Any]], float]
CheckpointScoreFn = Callable[[list[dict[str, Any]]], float]
CheckpointRankFn = Callable[[list[dict[str, Any]]], tuple[float, ...]]
CheckpointDiagnosticsFn = Callable[
    [list[dict[str, Any]]], dict[str, Any]
]
SummaryFn = Callable[[list[dict[str, Any]]], dict[str, Any]]
SMDPRolloutFn = Callable[
    [FrequencySeparatedActorCriticPPO, int, bool],
    tuple[HierarchicalTrajectoryBatch | None, dict[str, Any]],
]
SMDPReferenceRolloutFn = Callable[
    [FrequencySeparatedActorCriticPPO, int],
    HierarchicalTrajectoryBatch,
]
SMDPClosedLoopGuardFn = Callable[
    [FrequencySeparatedActorCriticPPO], dict[str, Any]
]
TrainingSeedFn = Callable[[int, int], int]
JointRolloutFn = Callable[
    [JointActorCriticPPO, int, bool],
    tuple[JointTrajectoryBatch | None, dict[str, Any]],
]
PROJECTION_CONSISTENCY_SCHEDULES = (
    "constant",
    "delayed_linear",
)


def projection_consistency_schedule_scale(
    *,
    iteration: int,
    total_iterations: int,
    schedule: str,
    warmup_fraction: float,
    ramp_fraction: float,
) -> float:
    """Return the projection-consistency multiplier for one PPO update."""

    if str(schedule) not in PROJECTION_CONSISTENCY_SCHEDULES:
        raise ValueError("unknown projection-consistency training schedule")
    if int(total_iterations) < 1:
        raise ValueError("total_iterations must be positive")
    if int(iteration) < 0 or int(iteration) >= int(total_iterations):
        raise ValueError("iteration must index the configured training run")
    warmup = float(warmup_fraction)
    ramp = float(ramp_fraction)
    if (
        not np.isfinite(warmup)
        or not np.isfinite(ramp)
        or not 0.0 <= warmup <= 1.0
        or not 0.0 <= ramp <= 1.0
        or warmup + ramp > 1.0
    ):
        raise ValueError(
            "projection-consistency warmup and ramp fractions must be finite, "
            "non-negative, and sum to at most one"
        )
    if str(schedule) == "constant":
        return 1.0
    if ramp <= 0.0:
        raise ValueError(
            "delayed-linear projection consistency requires a positive ramp"
        )
    progress = float(int(iteration) + 1) / float(int(total_iterations))
    if progress <= warmup:
        return 0.0
    if progress >= warmup + ramp:
        return 1.0
    return float((progress - warmup) / ramp)


def _validated_closed_loop_guard_snapshot(
    payload: dict[str, Any],
    *,
    restoration_filter: bool = False,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("closed-loop guard snapshot must be a mapping")
    rank = tuple(float(value) for value in payload.get("rank", ()))
    if not rank or not np.all(np.isfinite(rank)):
        raise ValueError("closed-loop guard rank must be finite and non-empty")
    normalized = dict(payload)
    normalized["rank"] = rank
    for key in (
        "path_count",
        "constraint_count",
        "reward_violation_count",
        "frequency_violation_count",
    ):
        value = payload.get(key)
        if (
            isinstance(value, bool)
            or value is None
            or int(value) != value
            or int(value) < 0
        ):
            raise ValueError(f"closed-loop guard {key} must be a non-negative integer")
        normalized[key] = int(value)
    if normalized["path_count"] < 1 or normalized["constraint_count"] < 1:
        raise ValueError("closed-loop guard registry must be non-empty")
    contract = str(payload.get("contract", "")).strip()
    if not contract:
        raise ValueError("closed-loop guard contract must be non-empty")
    normalized["contract"] = contract
    if restoration_filter:
        for key in (
            "frequency_violation_merit",
            "worst_frequency_violation",
        ):
            value = float(payload.get(key, float("nan")))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"closed-loop guard {key} must be finite and non-negative"
                )
            normalized[key] = value
        merit = float(normalized["frequency_violation_merit"])
        worst = float(normalized["worst_frequency_violation"])
        frequency_count = int(normalized["frequency_violation_count"])
        tolerance = 1e-10
        merit_tolerance = tolerance * tolerance
        worst_squared = worst * worst
        merit_roundoff = (
            8.0 * np.finfo(np.float64).eps
            * max(merit, worst_squared, merit_tolerance)
        )
        if worst_squared > merit + merit_tolerance + merit_roundoff:
            raise ValueError(
                "closed-loop guard worst frequency violation exceeds its "
                "aggregate merit"
            )
        if frequency_count == 0 and (
            worst > tolerance
            or merit > (
                normalized["constraint_count"] * merit_tolerance
                + merit_roundoff
            )
        ):
            raise ValueError(
                "a frequency-feasible guard snapshot has positive continuous "
                "frequency violations"
            )
        if frequency_count > 0 and (
            worst <= tolerance or merit <= merit_tolerance
        ):
            raise ValueError(
                "an infeasible guard snapshot must have positive continuous "
                "frequency violations"
            )
    return normalized


def _lexicographic_rank_not_worse(
    candidate: tuple[float, ...],
    baseline: tuple[float, ...],
    *,
    tolerance: float = 1e-10,
) -> bool:
    if len(candidate) != len(baseline):
        raise ValueError("closed-loop guard rank dimensions changed")
    for candidate_value, baseline_value in zip(candidate, baseline, strict=True):
        if candidate_value > baseline_value + tolerance:
            return True
        if candidate_value < baseline_value - tolerance:
            return False
    return True


def _closed_loop_guard_assessment(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    restoration_filter: bool = False,
    trial_fraction: float = 1.0,
    restoration_min_reduction: float = 0.0,
    restoration_funnel_limit: float | None = None,
) -> tuple[bool, list[str]]:
    registry_keys = ("contract", "path_count", "constraint_count")
    if any(candidate[key] != baseline[key] for key in registry_keys):
        raise ValueError("closed-loop guard registry changed during training")
    fraction = float(trial_fraction)
    min_reduction = float(restoration_min_reduction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("closed-loop guard trial fraction must be in [0, 1]")
    if not 0.0 <= min_reduction < 1.0:
        raise ValueError(
            "closed-loop restoration minimum reduction must be in [0, 1)"
        )
    reasons: list[str] = []
    if int(candidate["reward_violation_count"]) != 0:
        reasons.append("reward_floor_violation")
    candidate_count = int(candidate["frequency_violation_count"])
    baseline_count = int(baseline["frequency_violation_count"])
    if not restoration_filter:
        if candidate_count > baseline_count:
            reasons.append("frequency_violation_count_increase")
        if not _lexicographic_rank_not_worse(
            tuple(candidate["rank"]), tuple(baseline["rank"])
        ):
            reasons.append("lexicographic_rank_worse")
        return not reasons, reasons

    if baseline_count == 0:
        if candidate_count != 0:
            reasons.append("maintenance_frequency_violation")
        return not reasons, reasons

    if restoration_funnel_limit is None:
        raise ValueError("closed-loop restoration requires a funnel limit")
    funnel_limit = float(restoration_funnel_limit)
    if not np.isfinite(funnel_limit) or funnel_limit < 0.0:
        raise ValueError(
            "closed-loop restoration funnel limit must be finite and "
            "non-negative"
        )
    baseline_merit = float(baseline["frequency_violation_merit"])
    candidate_merit = float(candidate["frequency_violation_merit"])
    required_merit = baseline_merit * (
        1.0 - min_reduction * fraction
    )
    if candidate_merit > required_merit + 1e-12:
        reasons.append("restoration_merit_not_reduced")
    if (
        float(candidate["worst_frequency_violation"])
        > funnel_limit + 1e-12
    ):
        reasons.append("restoration_funnel_exceeded")
    return not reasons, reasons


def _closed_loop_guard_accepts(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    **kwargs: Any,
) -> bool:
    accepted, _ = _closed_loop_guard_assessment(
        candidate, baseline, **kwargs
    )
    return bool(accepted)


def _closed_loop_guard_trial_record(
    snapshot: dict[str, Any],
    *,
    fraction: float,
    accepted: bool,
    rejection_reasons: list[str],
) -> dict[str, Any]:
    return {
        "fraction": float(fraction),
        "accepted": bool(accepted),
        "rejection_reasons": list(rejection_reasons),
        "reward_violation_count": int(snapshot["reward_violation_count"]),
        "frequency_violation_count": int(
            snapshot["frequency_violation_count"]
        ),
        "rank": list(snapshot["rank"]),
        "frequency_violation_merit": float(
            snapshot.get("frequency_violation_merit", 0.0)
        ),
        "worst_frequency_violation": float(
            snapshot.get("worst_frequency_violation", 0.0)
        ),
    }


def _actor_state_rms_difference(
    before_state: dict[str, Any],
    after_state: dict[str, Any],
) -> float:
    squared_sum = 0.0
    count = 0
    for actor_name in ("upper_actor", "lower_actor"):
        before_actor = before_state[actor_name]
        after_actor = after_state[actor_name]
        if set(before_actor) != set(after_actor):
            raise ValueError("closed-loop actor state registry changed")
        for key, before in before_actor.items():
            after = after_actor[key]
            if torch.is_floating_point(before):
                delta = after.detach().double() - before.detach().double()
                squared_sum += float(torch.sum(delta * delta).cpu().item())
                count += int(delta.numel())
    return float(np.sqrt(squared_sum / max(count, 1)))


def _install_closed_loop_actor_fraction(
    model: FrequencySeparatedActorCriticPPO,
    *,
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    fraction: float,
) -> None:
    value = float(fraction)
    if not 0.0 <= value <= 1.0:
        raise ValueError("closed-loop actor fraction must be in [0, 1]")
    # Preserve the just-trained critics and duals, then replace only actors.
    model.load_state_dict(copy.deepcopy(after_state))
    for actor_name, optimizer_name in (
        ("upper_actor", "upper_actor_optimizer"),
        ("lower_actor", "lower_actor_optimizer"),
    ):
        before_actor = before_state[actor_name]
        after_actor = after_state[actor_name]
        blended: dict[str, torch.Tensor] = {}
        for key, before in before_actor.items():
            after = after_actor[key]
            blended[key] = (
                before + value * (after - before)
                if torch.is_floating_point(before)
                else (after if value == 1.0 else before)
            )
        getattr(model, actor_name).load_state_dict(blended)
        optimizer_state = (
            after_state[optimizer_name]
            if value == 1.0 else before_state[optimizer_name]
        )
        getattr(model, optimizer_name).load_state_dict(
            copy.deepcopy(optimizer_state)
        )
    model.reset_recurrent_inference()


def _apply_closed_loop_actor_guard(
    model: FrequencySeparatedActorCriticPPO,
    *,
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    before_snapshot: dict[str, Any],
    evaluate_fn: SMDPClosedLoopGuardFn,
    max_backtracks: int,
    restoration_filter: bool = False,
    restoration_min_reduction: float = 0.0,
    restoration_funnel_limit: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prefix = "deployment_frequency_closed_loop_guard_"
    before = _validated_closed_loop_guard_snapshot(
        before_snapshot,
        restoration_filter=restoration_filter,
    )
    full = _validated_closed_loop_guard_snapshot(
        evaluate_fn(model),
        restoration_filter=restoration_filter,
    )
    full_actor_rms = _actor_state_rms_difference(before_state, after_state)
    selected = full
    selected_fraction = 1.0
    backtracks = 0
    accepted, reasons = _closed_loop_guard_assessment(
        full,
        before,
        restoration_filter=restoration_filter,
        trial_fraction=1.0,
        restoration_min_reduction=restoration_min_reduction,
        restoration_funnel_limit=restoration_funnel_limit,
    )
    trial_trace: list[dict[str, Any]] = [
        _closed_loop_guard_trial_record(
            full,
            fraction=1.0,
            accepted=accepted,
            rejection_reasons=reasons,
        )
    ]
    evaluations = 1
    if not accepted:
        for backtrack in range(1, int(max_backtracks) + 1):
            fraction = 0.5 ** backtrack
            _install_closed_loop_actor_fraction(
                model,
                before_state=before_state,
                after_state=after_state,
                fraction=fraction,
            )
            candidate = _validated_closed_loop_guard_snapshot(
                evaluate_fn(model),
                restoration_filter=restoration_filter,
            )
            evaluations += 1
            backtracks = backtrack
            candidate_accepted, candidate_reasons = (
                _closed_loop_guard_assessment(
                    candidate,
                    before,
                    restoration_filter=restoration_filter,
                    trial_fraction=fraction,
                    restoration_min_reduction=restoration_min_reduction,
                    restoration_funnel_limit=restoration_funnel_limit,
                )
            )
            trial_trace.append(_closed_loop_guard_trial_record(
                candidate,
                fraction=fraction,
                accepted=candidate_accepted,
                rejection_reasons=candidate_reasons,
            ))
            if candidate_accepted:
                selected = candidate
                selected_fraction = float(fraction)
                accepted = True
                break
    if not accepted:
        _install_closed_loop_actor_fraction(
            model,
            before_state=before_state,
            after_state=after_state,
            fraction=0.0,
        )
        selected = _validated_closed_loop_guard_snapshot(
            evaluate_fn(model),
            restoration_filter=restoration_filter,
        )
        evaluations += 1
        rollback_accepted, rollback_reasons = _closed_loop_guard_assessment(
            selected,
            before,
            restoration_filter=restoration_filter,
            trial_fraction=0.0,
            restoration_min_reduction=restoration_min_reduction,
            restoration_funnel_limit=restoration_funnel_limit,
        )
        trial_trace.append(_closed_loop_guard_trial_record(
            selected,
            fraction=0.0,
            accepted=rollback_accepted,
            rejection_reasons=rollback_reasons,
        ))
        if not rollback_accepted:
            raise RuntimeError(
                "closed-loop actor rollback did not restore the guard rank"
            )
        selected_fraction = 0.0
    installed_state = copy.deepcopy(model.state_dict())
    final_actor_rms = _actor_state_rms_difference(
        before_state, installed_state
    )
    effective_update = bool(
        selected_fraction > 0.0 and final_actor_rms > 1e-12
    )
    return {
        f"{prefix}enabled": 1.0,
        f"{prefix}attempted": 1.0,
        f"{prefix}accepted": float(effective_update),
        f"{prefix}backtracks": float(backtracks),
        f"{prefix}step_fraction": float(selected_fraction),
        f"{prefix}evaluation_count": float(evaluations),
        f"{prefix}full_actor_rms": float(full_actor_rms),
        f"{prefix}final_actor_rms": float(final_actor_rms),
        f"{prefix}optimizer_restored": float(selected_fraction < 1.0),
        f"{prefix}contract": str(before["contract"]),
        f"{prefix}path_count": float(before["path_count"]),
        f"{prefix}constraint_count": float(before["constraint_count"]),
        f"{prefix}rank_before": list(before["rank"]),
        f"{prefix}rank_full_step": list(full["rank"]),
        f"{prefix}rank_after": list(selected["rank"]),
        f"{prefix}full_step_reward_violation_count": float(
            full["reward_violation_count"]
        ),
        f"{prefix}reward_violation_count": float(
            selected["reward_violation_count"]
        ),
        f"{prefix}full_step_frequency_violation_count": float(
            full["frequency_violation_count"]
        ),
        f"{prefix}frequency_violation_count": float(
            selected["frequency_violation_count"]
        ),
        f"{prefix}restoration_filter_enabled": float(restoration_filter),
        f"{prefix}restoration_phase_before": (
            "restoration"
            if int(before["frequency_violation_count"]) > 0
            else "maintenance"
        ),
        f"{prefix}restoration_phase_after": (
            "restoration"
            if int(selected["frequency_violation_count"]) > 0
            else "maintenance"
        ),
        f"{prefix}restoration_funnel_limit": float(
            restoration_funnel_limit or 0.0
        ),
        f"{prefix}restoration_merit_before": float(
            before.get("frequency_violation_merit", 0.0)
        ),
        f"{prefix}restoration_merit_full_step": float(
            full.get("frequency_violation_merit", 0.0)
        ),
        f"{prefix}restoration_merit_after": float(
            selected.get("frequency_violation_merit", 0.0)
        ),
        f"{prefix}worst_frequency_violation_before": float(
            before.get("worst_frequency_violation", 0.0)
        ),
        f"{prefix}worst_frequency_violation_full_step": float(
            full.get("worst_frequency_violation", 0.0)
        ),
        f"{prefix}worst_frequency_violation_after": float(
            selected.get("worst_frequency_violation", 0.0)
        ),
        f"{prefix}trial_trace": trial_trace,
    }, selected


def _disabled_closed_loop_guard_metrics() -> dict[str, Any]:
    prefix = "deployment_frequency_closed_loop_guard_"
    return {
        f"{prefix}enabled": 0.0,
        f"{prefix}attempted": 0.0,
        f"{prefix}accepted": 0.0,
        f"{prefix}backtracks": 0.0,
        f"{prefix}step_fraction": 1.0,
        f"{prefix}evaluation_count": 0.0,
        f"{prefix}full_actor_rms": 0.0,
        f"{prefix}final_actor_rms": 0.0,
        f"{prefix}optimizer_restored": 0.0,
        f"{prefix}contract": "disabled",
        f"{prefix}path_count": 0.0,
        f"{prefix}constraint_count": 0.0,
        f"{prefix}rank_before": [],
        f"{prefix}rank_full_step": [],
        f"{prefix}rank_after": [],
        f"{prefix}full_step_reward_violation_count": 0.0,
        f"{prefix}reward_violation_count": 0.0,
        f"{prefix}full_step_frequency_violation_count": 0.0,
        f"{prefix}frequency_violation_count": 0.0,
        f"{prefix}restoration_filter_enabled": 0.0,
        f"{prefix}restoration_phase_before": "disabled",
        f"{prefix}restoration_phase_after": "disabled",
        f"{prefix}restoration_funnel_limit": 0.0,
        f"{prefix}restoration_merit_before": 0.0,
        f"{prefix}restoration_merit_full_step": 0.0,
        f"{prefix}restoration_merit_after": 0.0,
        f"{prefix}worst_frequency_violation_before": 0.0,
        f"{prefix}worst_frequency_violation_full_step": 0.0,
        f"{prefix}worst_frequency_violation_after": 0.0,
        f"{prefix}trial_trace": [],
    }


def _iteration_rollout_seeds(
    seed_roots: list[int],
    iteration: int,
    training_seed_fn: TrainingSeedFn | None,
) -> list[int]:
    if training_seed_fn is None:
        return [int(seed) for seed in seed_roots]
    return [int(training_seed_fn(int(seed), int(iteration))) for seed in seed_roots]


def concat_batches(batches: Iterable[TrajectoryBatch]) -> TrajectoryBatch:
    items = list(batches)
    if not items:
        raise ValueError("at least one trajectory batch is required")
    return TrajectoryBatch(
        upper_state=np.concatenate([b.upper_state for b in items], axis=0),
        lower_state=np.concatenate([b.lower_state for b in items], axis=0),
        upper_action=np.concatenate([b.upper_action for b in items], axis=0),
        lower_action=np.concatenate([b.lower_action for b in items], axis=0),
        reward=np.concatenate([b.reward for b in items], axis=0),
        done=np.concatenate([b.done for b in items], axis=0),
        old_upper_logp=np.concatenate([b.old_upper_logp for b in items], axis=0),
        old_lower_logp=np.concatenate([b.old_lower_logp for b in items], axis=0),
        old_upper_value=np.concatenate([b.old_upper_value for b in items], axis=0),
        old_lower_value=np.concatenate([b.old_lower_value for b in items], axis=0),
        constraint=(
            np.concatenate([np.asarray(b.constraint, dtype=np.float32).reshape(-1) for b in items], axis=0)
            if all(b.constraint is not None for b in items) else None
        ),
    )


def summarize_numeric_rows(rows: list[dict[str, Any]], keys: list[str] | None = None) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    if keys is None:
        keys = [
            key for key, value in rows[0].items()
            if key != "seed" and isinstance(value, (int, float, np.integer, np.floating))
        ]
    summary = {
        f"{key}_mean": float(np.mean([float(row[key]) for row in rows]))
        for key in keys
        if key in rows[0]
    }
    summary["n"] = len(rows)
    return summary


def apply_replay_updates(
    model: DualActorCriticPPO,
    batch: TrajectoryBatch | None,
    updates: list[dict[str, Any]] | None = None,
    *,
    episode: int = 0,
    replay_updates: int = 1,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply PPO updates to an already-collected domain rollout batch.

    Native simulators often own the episode loop because they need to install
    policy proxies into existing control code.  This helper keeps the learning
    update itself in the shared Freq-HRL RL core: domains may collect
    `TrajectoryBatch` objects differently, but PPO replay updates are recorded
    through one implementation.
    """
    if batch is None:
        return {}
    row_metadata = dict(metadata or {})
    latest: dict[str, Any] = {}
    for replay_idx in range(max(1, int(replay_updates))):
        latest = model.update(batch)
        if updates is not None:
            updates.append({
                "episode": int(episode),
                "replay_update": int(replay_idx),
                **row_metadata,
                **latest,
            })
    return latest


def _sampled_summary(rows: list[dict[str, Any]], objective_fn: ObjectiveFn) -> dict[str, float]:
    out = {"sampled_objective": float(np.mean([objective_fn(row) for row in rows])) if rows else 0.0}
    if rows and "sharpe" in rows[0]:
        out["sampled_sharpe"] = float(np.mean([float(row["sharpe"]) for row in rows]))
    if rows and "reward_mean" in rows[0]:
        out["sampled_reward_mean"] = float(np.mean([float(row["reward_mean"]) for row in rows]))
    if rows and "LowerActionRouterStrength" in rows[0]:
        out["sampled_lower_action_router_strength"] = float(np.mean([
            float(row["LowerActionRouterStrength"]) for row in rows
        ]))
    for key in (
        "episode_length",
        "rollout_segment_count",
        "natural_episode_count",
        "trace_boundary_count",
        "mdp_terminal_count",
        "bootstrap_boundary_count",
        "transition_budget_exact",
    ):
        if rows and key in rows[0]:
            out[f"sampled_{key}_mean"] = float(np.mean([
                float(row[key]) for row in rows
            ]))
    return out


def _checkpoint_evaluation_due(
    iteration: int,
    *,
    total_iterations: int,
    interval: int,
) -> bool:
    if isinstance(interval, bool) or int(interval) < 1:
        raise ValueError("checkpoint evaluation interval must be positive")
    return bool(
        (int(iteration) + 1) % int(interval) == 0
        or int(iteration) == int(total_iterations) - 1
    )


def train_dual_ppo(
    model: DualActorCriticPPO,
    train_seeds: list[int],
    eval_seeds: list[int],
    iterations: int,
    rollout_fn: RolloutFn,
    objective_fn: ObjectiveFn,
    summary_fn: SummaryFn = summarize_numeric_rows,
    *,
    selection_seeds: list[int] | None = None,
    training_seed_fn: TrainingSeedFn | None = None,
    policy: str = "ppo_dual_actor_critic",
    trainer: str = "shared_dual_level_ppo",
    domain: str = "generic",
    metadata: dict[str, Any] | None = None,
    checkpoint_smoothing_window: int = 1,
    checkpoint_min_delta: float = 0.0,
    checkpoint_evaluation_interval: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]], DualActorCriticPPO]:
    """Train a dual-level PPO model through a domain-supplied rollout adapter."""
    metadata = dict(metadata or {})
    _checkpoint_evaluation_due(
        0, total_iterations=1, interval=checkpoint_evaluation_interval
    )
    selection_seed_list = list(selection_seeds or train_seeds)
    initial_rows = [
        rollout_fn(model, int(seed), False)[1] for seed in selection_seed_list
    ]
    initial_validation_score = float(np.mean([
        objective_fn(row) for row in initial_rows
    ]))
    selector = RobustValidationCheckpointSelector(
        initial_score=initial_validation_score,
        initial_state=model.state_dict(),
        smoothing_window=checkpoint_smoothing_window,
        min_delta=checkpoint_min_delta,
    )
    history: list[dict[str, Any]] = [{
        "iteration": -1,
        "score": initial_validation_score,
        **selector.initial_history_fields(),
        "sampled_objective": 0.0,
        **summary_fn(initial_rows),
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "constraint_loss": 0.0,
        "constraint_mean": 0.0,
        "constraint_lambda": float(model.constraint_lambda),
    }]

    total_iterations = max(1, int(iterations))
    for iteration in range(total_iterations):
        batches = []
        sampled_rows = []
        rollout_seeds = _iteration_rollout_seeds(
            train_seeds, iteration, training_seed_fn
        )
        for seed in rollout_seeds:
            batch, row = rollout_fn(model, int(seed), True)
            if batch is not None:
                batches.append(batch)
            sampled_rows.append(row)
        metrics = model.update(concat_batches(batches))
        evaluate_checkpoint = _checkpoint_evaluation_due(
            iteration,
            total_iterations=total_iterations,
            interval=checkpoint_evaluation_interval,
        )
        eval_rows = (
            [rollout_fn(model, int(seed), False)[1] for seed in selection_seed_list]
            if evaluate_checkpoint else []
        )
        score = (
            float(np.mean([objective_fn(row) for row in eval_rows]))
            if evaluate_checkpoint else None
        )
        checkpoint_fields = (
            selector.consider(
                score=float(score), state=model.state_dict(), iteration=iteration
            )
            if evaluate_checkpoint else {
                "checkpoint_selection_score": float(selector.best_score),
                "checkpoint_selection_eligible": False,
                "checkpoint_selected": False,
            }
        )
        history.append({
            "iteration": int(iteration),
            "training_rollout_seeds": rollout_seeds,
            "score": score,
            "checkpoint_evaluation_performed": evaluate_checkpoint,
            **checkpoint_fields,
            **_sampled_summary(sampled_rows, objective_fn),
            **(summary_fn(eval_rows) if evaluate_checkpoint else {}),
            **metrics,
        })

    model.load_state_dict(selector.best_state)
    heldout_rows = [rollout_fn(model, int(seed), False)[1] for seed in eval_seeds]
    payload = {
        "policy": policy,
        "trainer": trainer,
        "domain": domain,
        "train_seeds": list(train_seeds),
        "rollout_seed_roots": list(train_seeds),
        "selection_seeds": selection_seed_list,
        "eval_seeds": list(eval_seeds),
        "iterations": int(iterations),
        "best_score": float(selector.best_score),
        "initial_validation_score": initial_validation_score,
        "validation_learning_gain": float(
            selector.best_score - initial_validation_score
        ),
        "selected_checkpoint_iteration": int(selector.selected_iteration),
        "config": model.config.__dict__,
        "history": history,
        "summary": summary_fn(heldout_rows),
        **metadata,
        **selector.metadata(total_iterations=max(1, int(iterations))),
        "checkpoint_evaluation_interval": int(checkpoint_evaluation_interval),
    }
    return payload, heldout_rows, model


def train_joint_ppo(
    model: JointActorCriticPPO,
    train_seeds: list[int],
    eval_seeds: list[int],
    iterations: int,
    rollout_fn: JointRolloutFn,
    objective_fn: ObjectiveFn,
    summary_fn: SummaryFn = summarize_numeric_rows,
    *,
    selection_seeds: list[int] | None = None,
    training_seed_fn: TrainingSeedFn | None = None,
    policy: str = "flat_ppo",
    domain: str = "generic",
    metadata: dict[str, Any] | None = None,
    checkpoint_smoothing_window: int = 1,
    checkpoint_min_delta: float = 0.0,
    checkpoint_evaluation_interval: int = 1,
    checkpoint_score_fn: CheckpointScoreFn | None = None,
    checkpoint_score_contract: str = "mean_objective",
) -> tuple[dict[str, Any], list[dict[str, Any]], JointActorCriticPPO]:
    """Train a standard flat PPO with one joint action and one task return."""

    metadata = dict(metadata or {})
    _checkpoint_evaluation_due(
        0, total_iterations=1, interval=checkpoint_evaluation_interval
    )
    selection_seed_list = list(selection_seeds or train_seeds)
    if not str(checkpoint_score_contract).strip():
        raise ValueError("checkpoint_score_contract must be non-empty")

    def validation_score(rows: list[dict[str, Any]]) -> float:
        score = (
            float(checkpoint_score_fn(rows))
            if checkpoint_score_fn is not None
            else float(np.mean([objective_fn(row) for row in rows]))
        )
        if not np.isfinite(score):
            raise ValueError("checkpoint score must be finite")
        return score

    initial_rows = [
        rollout_fn(model, int(seed), False)[1] for seed in selection_seed_list
    ]
    initial_validation_score = validation_score(initial_rows)
    selector = RobustValidationCheckpointSelector(
        initial_score=initial_validation_score,
        initial_state=model.state_dict(),
        smoothing_window=checkpoint_smoothing_window,
        min_delta=checkpoint_min_delta,
    )
    history: list[dict[str, Any]] = [{
        "iteration": -1,
        "score": initial_validation_score,
        **selector.initial_history_fields(),
        "sampled_objective": 0.0,
        **summary_fn(initial_rows),
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "actor_optimizer_steps": 0.0,
        "value_optimizer_steps": 0.0,
    }]

    total_iterations = max(1, int(iterations))
    for iteration in range(total_iterations):
        batches: list[JointTrajectoryBatch] = []
        sampled_rows: list[dict[str, Any]] = []
        rollout_seeds = _iteration_rollout_seeds(
            train_seeds, iteration, training_seed_fn
        )
        for seed in rollout_seeds:
            batch, row = rollout_fn(model, int(seed), True)
            if batch is not None:
                batches.append(batch)
            sampled_rows.append(row)
        if not batches:
            raise RuntimeError("sampled rollouts did not produce a joint trajectory")
        metrics = model.update(concat_joint_batches(batches))
        evaluate_checkpoint = _checkpoint_evaluation_due(
            iteration,
            total_iterations=total_iterations,
            interval=checkpoint_evaluation_interval,
        )
        eval_rows = (
            [rollout_fn(model, int(seed), False)[1] for seed in selection_seed_list]
            if evaluate_checkpoint else []
        )
        score = (
            validation_score(eval_rows)
            if evaluate_checkpoint else None
        )
        checkpoint_fields = (
            selector.consider(
                score=float(score), state=model.state_dict(), iteration=iteration
            )
            if evaluate_checkpoint else {
                "checkpoint_selection_score": float(selector.best_score),
                "checkpoint_selection_eligible": False,
                "checkpoint_selected": False,
            }
        )
        history.append({
            "iteration": int(iteration),
            "training_rollout_seeds": rollout_seeds,
            "score": score,
            "checkpoint_evaluation_performed": evaluate_checkpoint,
            **checkpoint_fields,
            **_sampled_summary(sampled_rows, objective_fn),
            **(summary_fn(eval_rows) if evaluate_checkpoint else {}),
            **metrics,
        })

    model.load_state_dict(selector.best_state)
    heldout_rows = [rollout_fn(model, int(seed), False)[1] for seed in eval_seeds]
    actor_optimizer_steps = int(sum(
        float(row.get("actor_optimizer_steps", 0.0)) for row in history
    ))
    critic_optimizer_steps = int(sum(
        float(row.get("value_optimizer_steps", 0.0)) for row in history
    ))
    payload = {
        "policy": policy,
        "trainer": "canonical_joint_flat_ppo_v1",
        "domain": domain,
        "train_seeds": list(train_seeds),
        "rollout_seed_roots": list(train_seeds),
        "selection_seeds": selection_seed_list,
        "eval_seeds": list(eval_seeds),
        "iterations": int(iterations),
        "best_score": float(selector.best_score),
        "initial_validation_score": initial_validation_score,
        "validation_learning_gain": float(
            selector.best_score - initial_validation_score
        ),
        "selected_checkpoint_iteration": int(selector.selected_iteration),
        "config": model.config.__dict__,
        "history": history,
        "summary": summary_fn(heldout_rows),
        "actor_optimizer_steps_train": actor_optimizer_steps,
        "critic_optimizer_steps_train": critic_optimizer_steps,
        "temperature_optimizer_steps_train": 0,
        "gradient_updates_train": actor_optimizer_steps + critic_optimizer_steps,
        "trajectory_contract": {
            "decision_rate": "one joint action per primitive environment step",
            "policy_ratio": "one joint diagonal-Gaussian PPO ratio",
            "credit": "one task-return GAE shared by all action coordinates",
            "critic": "one state-value function",
        },
        **metadata,
        **selector.metadata(total_iterations=max(1, int(iterations))),
        "checkpoint_evaluation_interval": int(checkpoint_evaluation_interval),
        "checkpoint_score_contract": str(checkpoint_score_contract),
    }
    return payload, heldout_rows, model


def apply_smdp_updates(
    model: FrequencySeparatedActorCriticPPO,
    batch: HierarchicalTrajectoryBatch | None,
    updates: list[dict[str, Any]] | None = None,
    *,
    episode: int = 0,
    replay_updates: int = 1,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply independent upper/lower PPO updates to an SMDP rollout."""
    if batch is None:
        return {}
    row_metadata = dict(metadata or {})
    latest: dict[str, Any] = {}
    for replay_idx in range(max(1, int(replay_updates))):
        latest = model.update(batch)
        if updates is not None:
            updates.append({
                "episode": int(episode),
                "replay_update": int(replay_idx),
                **row_metadata,
                **latest,
            })
    return latest


def train_frequency_separated_ppo(
    model: FrequencySeparatedActorCriticPPO,
    train_seeds: list[int],
    eval_seeds: list[int],
    iterations: int,
    rollout_fn: SMDPRolloutFn,
    objective_fn: ObjectiveFn,
    summary_fn: SummaryFn = summarize_numeric_rows,
    *,
    selection_seeds: list[int] | None = None,
    training_seed_fn: TrainingSeedFn | None = None,
    policy: str = "freq_hrl_smdp_ppo",
    domain: str = "generic",
    metadata: dict[str, Any] | None = None,
    checkpoint_smoothing_window: int = 1,
    checkpoint_min_delta: float = 0.0,
    checkpoint_minimum_iteration: int = -1,
    checkpoint_evaluation_interval: int = 1,
    checkpoint_score_fn: CheckpointScoreFn | None = None,
    checkpoint_score_contract: str = "mean_objective",
    checkpoint_rank_fn: CheckpointRankFn | None = None,
    checkpoint_rank_names: tuple[str, ...] = (),
    checkpoint_rank_contract: str = "disabled",
    checkpoint_diagnostics_fn: CheckpointDiagnosticsFn | None = None,
    deployment_frequency_reference_rollout_fn: (
        SMDPReferenceRolloutFn | None
    ) = None,
    deployment_frequency_reference_seeds: list[int] | None = None,
    deployment_frequency_closed_loop_guard_fn: (
        SMDPClosedLoopGuardFn | None
    ) = None,
    projection_consistency_training_schedule: str = "constant",
    projection_consistency_warmup_fraction: float = 0.0,
    projection_consistency_ramp_fraction: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]], FrequencySeparatedActorCriticPPO]:
    """Train Freq-HRL with one upper transition per macro interval."""
    metadata = dict(metadata or {})
    _checkpoint_evaluation_due(
        0, total_iterations=1, interval=checkpoint_evaluation_interval
    )
    has_hf_stream = int(getattr(model.config, "hf_state_dim", 0)) > 0
    has_promotion_stream = int(getattr(model.config, "promotion_state_dim", 0)) > 0
    if has_hf_stream and has_promotion_stream:
        policy_ratio_contract = (
            "independent upper, lower, HF tactical, and promotion PPO ratios"
        )
    elif has_hf_stream:
        policy_ratio_contract = "independent upper, lower, and HF tactical PPO ratios"
    elif has_promotion_stream:
        policy_ratio_contract = "independent upper, lower, and promotion PPO ratios"
    else:
        policy_ratio_contract = "independent upper and lower PPO ratios"
    selection_seed_list = list(selection_seeds or train_seeds)
    if not str(checkpoint_score_contract).strip():
        raise ValueError("checkpoint_score_contract must be non-empty")
    rank_names = tuple(map(str, checkpoint_rank_names))
    if (checkpoint_rank_fn is None) != (not rank_names):
        raise ValueError(
            "checkpoint rank function and rank names must be configured together"
        )
    if checkpoint_rank_fn is not None and not str(
        checkpoint_rank_contract
    ).strip():
        raise ValueError("checkpoint rank contract must be non-empty")
    if checkpoint_rank_fn is not None and (
        int(checkpoint_smoothing_window) != 1
        or float(checkpoint_min_delta) != 0.0
    ):
        raise ValueError(
            "state-aligned checkpoint ranking requires smoothing_window=1 "
            "and min_delta=0"
        )

    def validation_score(rows: list[dict[str, Any]]) -> float:
        score = (
            float(checkpoint_score_fn(rows))
            if checkpoint_score_fn is not None
            else float(np.mean([objective_fn(row) for row in rows]))
        )
        if not np.isfinite(score):
            raise ValueError("checkpoint score must be finite")
        return score

    def validation_rank(
        rows: list[dict[str, Any]],
    ) -> tuple[float, ...] | None:
        if checkpoint_rank_fn is None:
            return None
        rank = tuple(float(value) for value in checkpoint_rank_fn(rows))
        if len(rank) != len(rank_names) or not np.all(np.isfinite(rank)):
            raise ValueError("checkpoint rank must be finite and aligned")
        return rank

    total_iterations = max(1, int(iterations))
    projection_consistency_target_upper = float(
        model.config.upper_projection_consistency_coef
    )
    projection_consistency_target_lower = float(
        model.config.lower_projection_consistency_coef
    )
    projection_consistency_schedule_scale(
        iteration=0,
        total_iterations=total_iterations,
        schedule=projection_consistency_training_schedule,
        warmup_fraction=projection_consistency_warmup_fraction,
        ramp_fraction=projection_consistency_ramp_fraction,
    )
    if int(checkpoint_minimum_iteration) >= total_iterations:
        raise ValueError(
            "checkpoint minimum iteration must be below total iterations"
        )
    anchor_state_replay_enabled = bool(
        model.config.deployment_frequency_anchor_state_replay
    )
    anchor_state_replay_seeds: list[int] = []
    anchor_state_replay_batch: HierarchicalTrajectoryBatch | None = None
    explicit_reference_seeds = (
        None
        if deployment_frequency_reference_seeds is None
        else list(deployment_frequency_reference_seeds)
    )
    if explicit_reference_seeds is not None and (
        not explicit_reference_seeds
        or any(
            isinstance(seed, bool) or int(seed) != seed or int(seed) < 0
            for seed in explicit_reference_seeds
        )
        or len(set(map(int, explicit_reference_seeds)))
        != len(explicit_reference_seeds)
    ):
        raise ValueError(
            "deployment-frequency reference seeds must be unique "
            "non-negative integers"
        )
    if anchor_state_replay_enabled:
        if deployment_frequency_reference_rollout_fn is None:
            raise ValueError(
                "anchor-state replay requires an explicit deterministic "
                "reference rollout function"
            )
        anchor_state_replay_seeds = (
            list(map(int, explicit_reference_seeds))
            if explicit_reference_seeds is not None
            else _iteration_rollout_seeds(
                train_seeds, 0, training_seed_fn
            )
        )
        reference_batches: list[HierarchicalTrajectoryBatch] = []
        for seed in anchor_state_replay_seeds:
            reference_batch = deployment_frequency_reference_rollout_fn(
                model, int(seed)
            )
            if not isinstance(reference_batch, HierarchicalTrajectoryBatch):
                raise RuntimeError(
                    "anchor-state replay rollout did not produce an SMDP "
                    "trajectory"
                )
            reference_batches.append(reference_batch)
        anchor_state_replay_batch = concat_hierarchical_batches(
            reference_batches
        )
    elif explicit_reference_seeds is not None:
        raise ValueError(
            "explicit deployment-frequency reference seeds require "
            "anchor-state replay"
        )
    closed_loop_guard_enabled = bool(
        model.config.deployment_frequency_closed_loop_trust_region
    )
    restoration_filter_enabled = bool(
        model.config.
        deployment_frequency_closed_loop_restoration_filter
    )
    if closed_loop_guard_enabled and deployment_frequency_closed_loop_guard_fn is None:
        raise ValueError(
            "closed-loop trust region requires an explicit deterministic "
            "guard evaluation function"
        )
    if not closed_loop_guard_enabled and deployment_frequency_closed_loop_guard_fn is not None:
        raise ValueError(
            "closed-loop guard evaluation cannot be supplied while the "
            "trust region is disabled"
        )
    initial_closed_loop_guard_snapshot = (
        _validated_closed_loop_guard_snapshot(
            deployment_frequency_closed_loop_guard_fn(model),
            restoration_filter=restoration_filter_enabled,
        )
        if deployment_frequency_closed_loop_guard_fn is not None else None
    )
    restoration_funnel_limit = (
        0.0
        if initial_closed_loop_guard_snapshot is None
        else float(initial_closed_loop_guard_snapshot.get(
            "worst_frequency_violation", 0.0
        )) * float(
            model.config.
            deployment_frequency_closed_loop_restoration_funnel_multiplier
        )
    )
    current_closed_loop_guard_snapshot = initial_closed_loop_guard_snapshot
    closed_loop_guard_evaluation_count = int(closed_loop_guard_enabled)
    if initial_closed_loop_guard_snapshot is None:
        initial_closed_loop_guard_metrics = _disabled_closed_loop_guard_metrics()
    else:
        guard_prefix = "deployment_frequency_closed_loop_guard_"
        initial_closed_loop_guard_metrics = {
            f"{guard_prefix}enabled": 1.0,
            f"{guard_prefix}attempted": 0.0,
            f"{guard_prefix}accepted": 0.0,
            f"{guard_prefix}backtracks": 0.0,
            f"{guard_prefix}step_fraction": 0.0,
            f"{guard_prefix}evaluation_count": 1.0,
            f"{guard_prefix}full_actor_rms": 0.0,
            f"{guard_prefix}final_actor_rms": 0.0,
            f"{guard_prefix}optimizer_restored": 0.0,
            f"{guard_prefix}contract": str(
                initial_closed_loop_guard_snapshot["contract"]
            ),
            f"{guard_prefix}path_count": float(
                initial_closed_loop_guard_snapshot["path_count"]
            ),
            f"{guard_prefix}constraint_count": float(
                initial_closed_loop_guard_snapshot["constraint_count"]
            ),
            f"{guard_prefix}rank_before": list(
                initial_closed_loop_guard_snapshot["rank"]
            ),
            f"{guard_prefix}rank_full_step": list(
                initial_closed_loop_guard_snapshot["rank"]
            ),
            f"{guard_prefix}rank_after": list(
                initial_closed_loop_guard_snapshot["rank"]
            ),
            f"{guard_prefix}full_step_reward_violation_count": float(
                initial_closed_loop_guard_snapshot["reward_violation_count"]
            ),
            f"{guard_prefix}reward_violation_count": float(
                initial_closed_loop_guard_snapshot["reward_violation_count"]
            ),
            f"{guard_prefix}full_step_frequency_violation_count": float(
                initial_closed_loop_guard_snapshot["frequency_violation_count"]
            ),
            f"{guard_prefix}frequency_violation_count": float(
                initial_closed_loop_guard_snapshot["frequency_violation_count"]
            ),
            f"{guard_prefix}restoration_filter_enabled": float(
                restoration_filter_enabled
            ),
            f"{guard_prefix}restoration_phase_before": (
                "restoration"
                if int(initial_closed_loop_guard_snapshot[
                    "frequency_violation_count"
                ]) > 0 else "maintenance"
            ),
            f"{guard_prefix}restoration_phase_after": (
                "restoration"
                if int(initial_closed_loop_guard_snapshot[
                    "frequency_violation_count"
                ]) > 0 else "maintenance"
            ),
            f"{guard_prefix}restoration_funnel_limit": float(
                restoration_funnel_limit
            ),
            f"{guard_prefix}restoration_merit_before": float(
                initial_closed_loop_guard_snapshot.get(
                    "frequency_violation_merit", 0.0
                )
            ),
            f"{guard_prefix}restoration_merit_full_step": float(
                initial_closed_loop_guard_snapshot.get(
                    "frequency_violation_merit", 0.0
                )
            ),
            f"{guard_prefix}restoration_merit_after": float(
                initial_closed_loop_guard_snapshot.get(
                    "frequency_violation_merit", 0.0
                )
            ),
            f"{guard_prefix}worst_frequency_violation_before": float(
                initial_closed_loop_guard_snapshot.get(
                    "worst_frequency_violation", 0.0
                )
            ),
            f"{guard_prefix}worst_frequency_violation_full_step": float(
                initial_closed_loop_guard_snapshot.get(
                    "worst_frequency_violation", 0.0
                )
            ),
            f"{guard_prefix}worst_frequency_violation_after": float(
                initial_closed_loop_guard_snapshot.get(
                    "worst_frequency_violation", 0.0
                )
            ),
            f"{guard_prefix}trial_trace": [],
        }
    initial_rows = [
        rollout_fn(model, int(seed), False)[1] for seed in selection_seed_list
    ]
    initial_validation_score = validation_score(initial_rows)
    initial_validation_rank = validation_rank(initial_rows)
    initial_checkpoint_diagnostics = (
        checkpoint_diagnostics_fn(initial_rows)
        if checkpoint_diagnostics_fn is not None else None
    )
    if (
        initial_checkpoint_diagnostics is not None
        and not isinstance(initial_checkpoint_diagnostics, dict)
    ):
        raise ValueError("checkpoint diagnostics must be a mapping")
    selector = (
        StateAlignedLexicographicCheckpointSelector(
            initial_score=initial_validation_score,
            initial_rank=initial_validation_rank,
            rank_names=rank_names,
            initial_state=model.state_dict(),
            minimum_eligible_iteration=checkpoint_minimum_iteration,
        )
        if initial_validation_rank is not None
        else RobustValidationCheckpointSelector(
            initial_score=initial_validation_score,
            initial_state=model.state_dict(),
            smoothing_window=checkpoint_smoothing_window,
            min_delta=checkpoint_min_delta,
            minimum_eligible_iteration=checkpoint_minimum_iteration,
        )
    )
    history: list[dict[str, Any]] = [{
        "iteration": -1,
        "score": initial_validation_score,
        **selector.initial_history_fields(),
        "sampled_objective": 0.0,
        **summary_fn(initial_rows),
        "upper_policy_loss": 0.0,
        "upper_value_loss": 0.0,
        "lower_policy_loss": 0.0,
        "lower_value_loss": 0.0,
        "upper_actor_anchor_kl": 0.0,
        "upper_actor_anchor_loss": 0.0,
        "lower_actor_anchor_kl": 0.0,
        "lower_actor_anchor_loss": 0.0,
        "projection_consistency_schedule_scale": (
            1.0
            if str(projection_consistency_training_schedule) == "constant"
            else 0.0
        ),
        "upper_projection_consistency_effective_coef": (
            projection_consistency_target_upper
            if str(projection_consistency_training_schedule) == "constant"
            else 0.0
        ),
        "lower_projection_consistency_effective_coef": (
            projection_consistency_target_lower
            if str(projection_consistency_training_schedule) == "constant"
            else 0.0
        ),
        "constraint_mean": 0.0,
        "constraint_lambda": float(model.constraint_lambda),
        "upper_constraint_mean": 0.0,
        "upper_constraint_lambda": float(model.upper_constraint_lambda),
        "lower_constraint_mean": 0.0,
        "lower_constraint_lambda": float(model.constraint_lambda),
        "upper_actor_optimizer_steps": 0.0,
        "upper_value_optimizer_steps": 0.0,
        "upper_cost_value_optimizer_steps": 0.0,
        "lower_actor_optimizer_steps": 0.0,
        "lower_value_optimizer_steps": 0.0,
        "lower_cost_value_optimizer_steps": 0.0,
        "hf_actor_optimizer_steps": 0.0,
        "hf_value_optimizer_steps": 0.0,
        "hf_cost_value_optimizer_steps": 0.0,
        "promotion_actor_optimizer_steps": 0.0,
        "promotion_value_optimizer_steps": 0.0,
        "promotion_cost_value_optimizer_steps": 0.0,
        "deployment_frequency_anchor_state_replay_enabled": float(
            anchor_state_replay_enabled
        ),
        "deployment_frequency_anchor_state_replay_path_count": float(
            len(anchor_state_replay_seeds)
        ),
        **initial_closed_loop_guard_metrics,
        **(
            {
                "checkpoint_selection_diagnostics": (
                    initial_checkpoint_diagnostics
                )
            }
            if initial_checkpoint_diagnostics is not None else {}
        ),
    }]

    for iteration in range(total_iterations):
        projection_consistency_scale = projection_consistency_schedule_scale(
            iteration=iteration,
            total_iterations=total_iterations,
            schedule=projection_consistency_training_schedule,
            warmup_fraction=projection_consistency_warmup_fraction,
            ramp_fraction=projection_consistency_ramp_fraction,
        )
        model.config.upper_projection_consistency_coef = float(
            projection_consistency_target_upper * projection_consistency_scale
        )
        model.config.lower_projection_consistency_coef = float(
            projection_consistency_target_lower * projection_consistency_scale
        )
        batches: list[HierarchicalTrajectoryBatch] = []
        sampled_rows = []
        rollout_seeds = _iteration_rollout_seeds(
            train_seeds, iteration, training_seed_fn
        )
        for seed in rollout_seeds:
            batch, row = rollout_fn(model, int(seed), True)
            if batch is not None:
                batches.append(batch)
            sampled_rows.append(row)
        if not batches:
            raise RuntimeError("sampled rollouts did not produce an SMDP trajectory")
        before_update_state = (
            copy.deepcopy(model.state_dict())
            if closed_loop_guard_enabled else None
        )
        restoration_mode = bool(
            restoration_filter_enabled
            and current_closed_loop_guard_snapshot is not None
            and int(current_closed_loop_guard_snapshot[
                "frequency_violation_count"
            ]) > 0
        )
        metrics = model.update(
            concat_hierarchical_batches(batches),
            deployment_frequency_reference_batch=(
                anchor_state_replay_batch
            ),
            deployment_frequency_restoration_mode=restoration_mode,
        )
        if closed_loop_guard_enabled:
            if (
                before_update_state is None
                or current_closed_loop_guard_snapshot is None
                or deployment_frequency_closed_loop_guard_fn is None
            ):
                raise RuntimeError("closed-loop guard transaction was not initialized")
            after_update_state = copy.deepcopy(model.state_dict())
            guard_metrics, current_closed_loop_guard_snapshot = (
                _apply_closed_loop_actor_guard(
                    model,
                    before_state=before_update_state,
                    after_state=after_update_state,
                    before_snapshot=current_closed_loop_guard_snapshot,
                    evaluate_fn=deployment_frequency_closed_loop_guard_fn,
                    max_backtracks=(
                        model.config.
                        deployment_frequency_closed_loop_trust_region_backtracks
                    ),
                    restoration_filter=restoration_filter_enabled,
                    restoration_min_reduction=float(
                        model.config.
                        deployment_frequency_closed_loop_restoration_min_reduction
                    ),
                    restoration_funnel_limit=restoration_funnel_limit,
                )
            )
            closed_loop_guard_evaluation_count += int(
                guard_metrics[
                    "deployment_frequency_closed_loop_guard_evaluation_count"
                ]
            )
        else:
            guard_metrics = _disabled_closed_loop_guard_metrics()
        metrics.update(guard_metrics)
        metrics.update({
            "projection_consistency_schedule_scale": float(
                projection_consistency_scale
            ),
            "upper_projection_consistency_effective_coef": float(
                model.config.upper_projection_consistency_coef
            ),
            "lower_projection_consistency_effective_coef": float(
                model.config.lower_projection_consistency_coef
            ),
        })
        evaluate_checkpoint = _checkpoint_evaluation_due(
            iteration,
            total_iterations=total_iterations,
            interval=checkpoint_evaluation_interval,
        )
        eval_rows = (
            [rollout_fn(model, int(seed), False)[1] for seed in selection_seed_list]
            if evaluate_checkpoint else []
        )
        score = (
            validation_score(eval_rows)
            if evaluate_checkpoint else None
        )
        rank = validation_rank(eval_rows) if evaluate_checkpoint else None
        checkpoint_diagnostics = (
            checkpoint_diagnostics_fn(eval_rows)
            if evaluate_checkpoint and checkpoint_diagnostics_fn is not None
            else None
        )
        if (
            checkpoint_diagnostics is not None
            and not isinstance(checkpoint_diagnostics, dict)
        ):
            raise ValueError("checkpoint diagnostics must be a mapping")
        checkpoint_fields = (
            (
                selector.consider(
                    score=float(score),
                    rank=rank,
                    state=model.state_dict(),
                    iteration=iteration,
                )
                if rank is not None
                else selector.consider(
                    score=float(score),
                    state=model.state_dict(),
                    iteration=iteration,
                )
            )
            if evaluate_checkpoint else {
                "checkpoint_selection_score": float(selector.best_score),
                "checkpoint_selection_eligible": False,
                "checkpoint_selected": False,
            }
        )
        history.append({
            "iteration": int(iteration),
            "training_rollout_seeds": rollout_seeds,
            "score": score,
            "checkpoint_evaluation_performed": evaluate_checkpoint,
            **checkpoint_fields,
            **_sampled_summary(sampled_rows, objective_fn),
            **(summary_fn(eval_rows) if evaluate_checkpoint else {}),
            **metrics,
            **(
                {"checkpoint_selection_diagnostics": checkpoint_diagnostics}
                if checkpoint_diagnostics is not None else {}
            ),
        })

    if not selector.has_eligible_selection:
        raise RuntimeError("checkpoint selector produced no eligible checkpoint")
    model.load_state_dict(selector.best_state)
    selected_closed_loop_guard_snapshot = (
        _validated_closed_loop_guard_snapshot(
            deployment_frequency_closed_loop_guard_fn(model),
            restoration_filter=restoration_filter_enabled,
        )
        if deployment_frequency_closed_loop_guard_fn is not None else None
    )
    closed_loop_guard_evaluation_count += int(closed_loop_guard_enabled)
    if (
        selected_closed_loop_guard_snapshot is not None
        and initial_closed_loop_guard_snapshot is not None
        and not _closed_loop_guard_accepts(
            selected_closed_loop_guard_snapshot,
            initial_closed_loop_guard_snapshot,
            restoration_filter=restoration_filter_enabled,
            restoration_min_reduction=0.0,
            restoration_funnel_limit=restoration_funnel_limit,
        )
    ):
        raise RuntimeError(
            "selected checkpoint violates the initial closed-loop guard"
        )
    heldout_rows = [rollout_fn(model, int(seed), False)[1] for seed in eval_seeds]
    actor_optimizer_steps = int(sum(
        float(row.get("upper_actor_optimizer_steps", 0.0))
        + float(row.get("lower_actor_optimizer_steps", 0.0))
        + float(row.get("hf_actor_optimizer_steps", 0.0))
        + float(row.get("promotion_actor_optimizer_steps", 0.0))
        for row in history
    ))
    critic_optimizer_steps = int(sum(
        float(row.get("upper_value_optimizer_steps", 0.0))
        + float(row.get("lower_value_optimizer_steps", 0.0))
        + float(row.get("upper_cost_value_optimizer_steps", 0.0))
        + float(row.get("lower_cost_value_optimizer_steps", 0.0))
        + float(row.get("hf_value_optimizer_steps", 0.0))
        + float(row.get("hf_cost_value_optimizer_steps", 0.0))
        + float(row.get("promotion_value_optimizer_steps", 0.0))
        for row in history
    ))

    def projection_guard_training_summary(level: str) -> dict[str, float]:
        attempted_key = f"{level}_projection_guard_attempted"
        attempted_rows = [
            row for row in history
            if float(row.get(attempted_key, 0.0)) > 0.0
        ]
        attempted_mass = float(sum(
            float(row.get(attempted_key, 0.0)) for row in attempted_rows
        ))
        accepted_mass = float(sum(
            float(row.get(f"{level}_projection_guard_accepted", 0.0))
            for row in attempted_rows
        ))
        return {
            "active_iteration_count": float(len(attempted_rows)),
            "attempted_mass": attempted_mass,
            "accepted_mass": accepted_mass,
            "acceptance_rate": (
                accepted_mass / attempted_mass
                if attempted_mass > 0.0 else 0.0
            ),
            "reward_loss_delta_max": max(
                (
                    float(row.get(
                        f"{level}_projection_guard_reward_loss_delta",
                        0.0,
                    ))
                    for row in attempted_rows
                ),
                default=0.0,
            ),
            "native_constraint_loss_delta_max": max(
                (
                    float(row.get(
                        f"{level}_projection_guard_native_constraint_loss_delta",
                        0.0,
                    ))
                    for row in attempted_rows
                ),
                default=0.0,
            ),
            "consistency_loss_delta_mean": (
                float(np.mean([
                    float(row.get(
                        f"{level}_projection_guard_consistency_loss_delta",
                        0.0,
                    ))
                    for row in attempted_rows
                ]))
                if attempted_rows else 0.0
            ),
            "gradient_conflict_rate": (
                float(np.mean([
                    float(row.get(
                        f"{level}_projection_gradient_conflict", 0.0
                    ))
                    for row in attempted_rows
                ]))
                if attempted_rows else 0.0
            ),
        }

    payload = {
        "policy": policy,
        "trainer": "frequency_separated_smdp_ppo_v2",
        "domain": domain,
        "train_seeds": list(train_seeds),
        "rollout_seed_roots": list(train_seeds),
        "selection_seeds": selection_seed_list,
        "eval_seeds": list(eval_seeds),
        "iterations": int(iterations),
        "projection_consistency_guard_training": {
            level: projection_guard_training_summary(level)
            for level in ("upper", "lower")
        },
        "projection_consistency_training_schedule": str(
            projection_consistency_training_schedule
        ),
        "projection_consistency_warmup_fraction": float(
            projection_consistency_warmup_fraction
        ),
        "projection_consistency_ramp_fraction": float(
            projection_consistency_ramp_fraction
        ),
        "upper_projection_consistency_target_coef": float(
            projection_consistency_target_upper
        ),
        "lower_projection_consistency_target_coef": float(
            projection_consistency_target_lower
        ),
        "deployment_frequency_anchor_state_replay_enabled": (
            anchor_state_replay_enabled
        ),
        "deployment_frequency_anchor_state_replay_seeds": list(
            anchor_state_replay_seeds
        ),
        "deployment_frequency_anchor_state_replay_path_count": len(
            anchor_state_replay_seeds
        ),
        "deployment_frequency_anchor_state_replay_seed_source": (
            "explicit"
            if explicit_reference_seeds is not None
            else (
                "iteration_zero_training_paths"
                if anchor_state_replay_enabled else "disabled"
            )
        ),
        "deployment_frequency_anchor_state_replay_contract": (
            "deterministic_frozen_anchor_deployment_trajectory_v1"
            if anchor_state_replay_enabled else "disabled"
        ),
        "deployment_frequency_anchor_state_replay_upper_transitions": (
            0
            if anchor_state_replay_batch is None
            else anchor_state_replay_batch.upper.size
        ),
        "deployment_frequency_anchor_state_replay_lower_transitions": (
            0
            if anchor_state_replay_batch is None
            else anchor_state_replay_batch.lower.size
        ),
        "deployment_frequency_closed_loop_guard_enabled": (
            closed_loop_guard_enabled
        ),
        "deployment_frequency_closed_loop_guard_contract": (
            "disabled"
            if initial_closed_loop_guard_snapshot is None
            else str(initial_closed_loop_guard_snapshot["contract"])
        ),
        "deployment_frequency_closed_loop_guard_path_count": (
            0
            if initial_closed_loop_guard_snapshot is None
            else int(initial_closed_loop_guard_snapshot["path_count"])
        ),
        "deployment_frequency_closed_loop_guard_constraint_count": (
            0
            if initial_closed_loop_guard_snapshot is None
            else int(initial_closed_loop_guard_snapshot["constraint_count"])
        ),
        "deployment_frequency_closed_loop_restoration_filter_enabled": (
            restoration_filter_enabled
        ),
        "deployment_frequency_restoration_freeze_reward_actor_enabled": bool(
            model.config.
            deployment_frequency_restoration_freeze_reward_actor
        ),
        "deployment_frequency_closed_loop_restoration_min_reduction": float(
            model.config.
            deployment_frequency_closed_loop_restoration_min_reduction
        ),
        "deployment_frequency_closed_loop_restoration_funnel_multiplier": float(
            model.config.
            deployment_frequency_closed_loop_restoration_funnel_multiplier
        ),
        "deployment_frequency_closed_loop_restoration_funnel_limit": float(
            restoration_funnel_limit
        ),
        "deployment_frequency_closed_loop_guard_initial_rank": (
            []
            if initial_closed_loop_guard_snapshot is None
            else list(initial_closed_loop_guard_snapshot["rank"])
        ),
        "deployment_frequency_closed_loop_guard_initial_reward_violation_count": (
            0
            if initial_closed_loop_guard_snapshot is None
            else int(initial_closed_loop_guard_snapshot[
                "reward_violation_count"
            ])
        ),
        "deployment_frequency_closed_loop_guard_initial_frequency_violation_count": (
            0
            if initial_closed_loop_guard_snapshot is None
            else int(initial_closed_loop_guard_snapshot[
                "frequency_violation_count"
            ])
        ),
        "deployment_frequency_closed_loop_guard_initial_frequency_violation_merit": (
            0.0
            if initial_closed_loop_guard_snapshot is None
            else float(initial_closed_loop_guard_snapshot.get(
                "frequency_violation_merit", 0.0
            ))
        ),
        "deployment_frequency_closed_loop_guard_initial_worst_frequency_violation": (
            0.0
            if initial_closed_loop_guard_snapshot is None
            else float(initial_closed_loop_guard_snapshot.get(
                "worst_frequency_violation", 0.0
            ))
        ),
        "deployment_frequency_closed_loop_guard_training_final_rank": (
            []
            if current_closed_loop_guard_snapshot is None
            else list(current_closed_loop_guard_snapshot["rank"])
        ),
        "deployment_frequency_closed_loop_guard_training_final_reward_violation_count": (
            0
            if current_closed_loop_guard_snapshot is None
            else int(current_closed_loop_guard_snapshot[
                "reward_violation_count"
            ])
        ),
        "deployment_frequency_closed_loop_guard_training_final_frequency_violation_count": (
            0
            if current_closed_loop_guard_snapshot is None
            else int(current_closed_loop_guard_snapshot[
                "frequency_violation_count"
            ])
        ),
        "deployment_frequency_closed_loop_guard_training_final_frequency_violation_merit": (
            0.0
            if current_closed_loop_guard_snapshot is None
            else float(current_closed_loop_guard_snapshot.get(
                "frequency_violation_merit", 0.0
            ))
        ),
        "deployment_frequency_closed_loop_guard_training_final_worst_frequency_violation": (
            0.0
            if current_closed_loop_guard_snapshot is None
            else float(current_closed_loop_guard_snapshot.get(
                "worst_frequency_violation", 0.0
            ))
        ),
        "deployment_frequency_closed_loop_guard_selected_rank": (
            []
            if selected_closed_loop_guard_snapshot is None
            else list(selected_closed_loop_guard_snapshot["rank"])
        ),
        "deployment_frequency_closed_loop_guard_selected_reward_violation_count": (
            0
            if selected_closed_loop_guard_snapshot is None
            else int(selected_closed_loop_guard_snapshot[
                "reward_violation_count"
            ])
        ),
        "deployment_frequency_closed_loop_guard_selected_frequency_violation_count": (
            0
            if selected_closed_loop_guard_snapshot is None
            else int(selected_closed_loop_guard_snapshot[
                "frequency_violation_count"
            ])
        ),
        "deployment_frequency_closed_loop_guard_selected_frequency_violation_merit": (
            0.0
            if selected_closed_loop_guard_snapshot is None
            else float(selected_closed_loop_guard_snapshot.get(
                "frequency_violation_merit", 0.0
            ))
        ),
        "deployment_frequency_closed_loop_guard_selected_worst_frequency_violation": (
            0.0
            if selected_closed_loop_guard_snapshot is None
            else float(selected_closed_loop_guard_snapshot.get(
                "worst_frequency_violation", 0.0
            ))
        ),
        "deployment_frequency_closed_loop_guard_evaluation_count": int(
            closed_loop_guard_evaluation_count
        ),
        "deployment_frequency_closed_loop_guard_effective_update_count": int(
            sum(
                float(row.get(
                    "deployment_frequency_closed_loop_guard_accepted", 0.0
                ))
                for row in history
            )
        ),
        "best_score": float(selector.best_score),
        "initial_validation_score": initial_validation_score,
        "validation_learning_gain": float(
            selector.best_score - initial_validation_score
        ),
        "selected_checkpoint_iteration": int(selector.selected_iteration),
        "config": model.config.__dict__,
        "history": history,
        "summary": summary_fn(heldout_rows),
        "actor_optimizer_steps_train": actor_optimizer_steps,
        "critic_optimizer_steps_train": critic_optimizer_steps,
        "temperature_optimizer_steps_train": 0,
        "gradient_updates_train": actor_optimizer_steps + critic_optimizer_steps,
        "trajectory_contract": {
            "upper": "one transition per macro action with gamma^duration bootstrap",
            "lower": "one transition per primitive control action",
            "hf": (
                "one independent tactical transition per primitive step with "
                "a dedicated marginal HF reward"
                if int(getattr(model.config, "hf_state_dim", 0)) > 0
                else "disabled"
            ),
            "promotion": (
                "one sparse Bernoulli transition per eligible replan probe; "
                "reward and gamma duration extend until the next gate decision"
                if int(getattr(model.config, "promotion_state_dim", 0)) > 0
                else "disabled"
            ),
            "policy_ratios": (
                policy_ratio_contract
            ),
        },
        **metadata,
        **selector.metadata(total_iterations=max(1, int(iterations))),
        "checkpoint_evaluation_interval": int(checkpoint_evaluation_interval),
        "checkpoint_score_contract": str(checkpoint_score_contract),
        "checkpoint_rank_contract": (
            str(checkpoint_rank_contract)
            if checkpoint_rank_fn is not None else "disabled"
        ),
    }
    if checkpoint_diagnostics_fn is not None:
        selected_diagnostics = next(
            (
                row["checkpoint_selection_diagnostics"]
                for row in history
                if int(row["iteration"]) == int(selector.selected_iteration)
            ),
            None,
        )
        if selected_diagnostics is None:
            raise RuntimeError(
                "selected checkpoint diagnostics were not retained"
            )
        payload["selected_checkpoint_diagnostics"] = selected_diagnostics
    return payload, heldout_rows, model
