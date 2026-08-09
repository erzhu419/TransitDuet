"""Domain-agnostic training loops for dual-level Freq-HRL policies."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np

from .dual_actor_critic import DualActorCriticPPO, TrajectoryBatch
from .checkpoint_selection import RobustValidationCheckpointSelector
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
SummaryFn = Callable[[list[dict[str, Any]]], dict[str, Any]]
SMDPRolloutFn = Callable[
    [FrequencySeparatedActorCriticPPO, int, bool],
    tuple[HierarchicalTrajectoryBatch | None, dict[str, Any]],
]
TrainingSeedFn = Callable[[int, int], int]
JointRolloutFn = Callable[
    [JointActorCriticPPO, int, bool],
    tuple[JointTrajectoryBatch | None, dict[str, Any]],
]


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

    def validation_score(rows: list[dict[str, Any]]) -> float:
        score = (
            float(checkpoint_score_fn(rows))
            if checkpoint_score_fn is not None
            else float(np.mean([objective_fn(row) for row in rows]))
        )
        if not np.isfinite(score):
            raise ValueError("checkpoint score must be finite")
        return score

    total_iterations = max(1, int(iterations))
    if int(checkpoint_minimum_iteration) >= total_iterations:
        raise ValueError(
            "checkpoint minimum iteration must be below total iterations"
        )
    initial_rows = [
        rollout_fn(model, int(seed), False)[1] for seed in selection_seed_list
    ]
    initial_validation_score = validation_score(initial_rows)
    selector = RobustValidationCheckpointSelector(
        initial_score=initial_validation_score,
        initial_state=model.state_dict(),
        smoothing_window=checkpoint_smoothing_window,
        min_delta=checkpoint_min_delta,
        minimum_eligible_iteration=checkpoint_minimum_iteration,
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
    }]

    for iteration in range(total_iterations):
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
        metrics = model.update(concat_hierarchical_batches(batches))
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

    if not selector.has_eligible_selection:
        raise RuntimeError("checkpoint selector produced no eligible checkpoint")
    model.load_state_dict(selector.best_state)
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
    payload = {
        "policy": policy,
        "trainer": "frequency_separated_smdp_ppo_v2",
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
    }
    return payload, heldout_rows, model
