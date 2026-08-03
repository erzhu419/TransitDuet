"""Domain-agnostic training loops for dual-level Freq-HRL policies."""

from __future__ import annotations

import copy
from typing import Any, Callable, Iterable

import numpy as np

from .dual_actor_critic import DualActorCriticPPO, TrajectoryBatch
from .smdp_actor_critic import (
    FrequencySeparatedActorCriticPPO,
    HierarchicalTrajectoryBatch,
    concat_hierarchical_batches,
)

RolloutFn = Callable[[DualActorCriticPPO, int, bool], tuple[TrajectoryBatch | None, dict[str, Any]]]
ObjectiveFn = Callable[[dict[str, Any]], float]
SummaryFn = Callable[[list[dict[str, Any]]], dict[str, Any]]
SMDPRolloutFn = Callable[
    [FrequencySeparatedActorCriticPPO, int, bool],
    tuple[HierarchicalTrajectoryBatch | None, dict[str, Any]],
]


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
    return out


def train_dual_ppo(
    model: DualActorCriticPPO,
    train_seeds: list[int],
    eval_seeds: list[int],
    iterations: int,
    rollout_fn: RolloutFn,
    objective_fn: ObjectiveFn,
    summary_fn: SummaryFn = summarize_numeric_rows,
    *,
    policy: str = "ppo_dual_actor_critic",
    trainer: str = "shared_dual_level_ppo",
    domain: str = "generic",
    metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], DualActorCriticPPO]:
    """Train a dual-level PPO model through a domain-supplied rollout adapter."""
    metadata = dict(metadata or {})
    best_state = copy.deepcopy(model.state_dict())
    initial_rows = [rollout_fn(model, int(seed), False)[1] for seed in train_seeds]
    best_score = float(np.mean([objective_fn(row) for row in initial_rows]))
    history: list[dict[str, Any]] = [{
        "iteration": -1,
        "score": best_score,
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

    for iteration in range(max(1, int(iterations))):
        batches = []
        sampled_rows = []
        for seed in train_seeds:
            batch, row = rollout_fn(model, int(seed), True)
            if batch is not None:
                batches.append(batch)
            sampled_rows.append(row)
        metrics = model.update(concat_batches(batches))
        eval_rows = [rollout_fn(model, int(seed), False)[1] for seed in train_seeds]
        score = float(np.mean([objective_fn(row) for row in eval_rows]))
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
        history.append({
            "iteration": int(iteration),
            "score": score,
            **_sampled_summary(sampled_rows, objective_fn),
            **summary_fn(eval_rows),
            **metrics,
        })

    model.load_state_dict(best_state)
    heldout_rows = [rollout_fn(model, int(seed), False)[1] for seed in eval_seeds]
    payload = {
        "policy": policy,
        "trainer": trainer,
        "domain": domain,
        "train_seeds": list(train_seeds),
        "eval_seeds": list(eval_seeds),
        "iterations": int(iterations),
        "best_score": float(best_score),
        "config": model.config.__dict__,
        "history": history,
        "summary": summary_fn(heldout_rows),
        **metadata,
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
    policy: str = "freq_hrl_smdp_ppo",
    domain: str = "generic",
    metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], FrequencySeparatedActorCriticPPO]:
    """Train Freq-HRL with one upper transition per macro interval."""
    metadata = dict(metadata or {})
    best_state = copy.deepcopy(model.state_dict())
    initial_rows = [rollout_fn(model, int(seed), False)[1] for seed in train_seeds]
    best_score = float(np.mean([objective_fn(row) for row in initial_rows]))
    history: list[dict[str, Any]] = [{
        "iteration": -1,
        "score": best_score,
        "sampled_objective": 0.0,
        **summary_fn(initial_rows),
        "upper_policy_loss": 0.0,
        "upper_value_loss": 0.0,
        "lower_policy_loss": 0.0,
        "lower_value_loss": 0.0,
        "constraint_mean": 0.0,
        "constraint_lambda": float(model.constraint_lambda),
        "upper_actor_optimizer_steps": 0.0,
        "upper_value_optimizer_steps": 0.0,
        "upper_cost_value_optimizer_steps": 0.0,
        "lower_actor_optimizer_steps": 0.0,
        "lower_value_optimizer_steps": 0.0,
        "lower_cost_value_optimizer_steps": 0.0,
    }]

    for iteration in range(max(1, int(iterations))):
        batches: list[HierarchicalTrajectoryBatch] = []
        sampled_rows = []
        for seed in train_seeds:
            batch, row = rollout_fn(model, int(seed), True)
            if batch is not None:
                batches.append(batch)
            sampled_rows.append(row)
        if not batches:
            raise RuntimeError("sampled rollouts did not produce an SMDP trajectory")
        metrics = model.update(concat_hierarchical_batches(batches))
        eval_rows = [rollout_fn(model, int(seed), False)[1] for seed in train_seeds]
        score = float(np.mean([objective_fn(row) for row in eval_rows]))
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
        history.append({
            "iteration": int(iteration),
            "score": score,
            **_sampled_summary(sampled_rows, objective_fn),
            **summary_fn(eval_rows),
            **metrics,
        })

    model.load_state_dict(best_state)
    heldout_rows = [rollout_fn(model, int(seed), False)[1] for seed in eval_seeds]
    actor_optimizer_steps = int(sum(
        float(row.get("upper_actor_optimizer_steps", 0.0))
        + float(row.get("lower_actor_optimizer_steps", 0.0))
        for row in history
    ))
    critic_optimizer_steps = int(sum(
        float(row.get("upper_value_optimizer_steps", 0.0))
        + float(row.get("lower_value_optimizer_steps", 0.0))
        + float(row.get("upper_cost_value_optimizer_steps", 0.0))
        + float(row.get("lower_cost_value_optimizer_steps", 0.0))
        for row in history
    ))
    payload = {
        "policy": policy,
        "trainer": "frequency_separated_smdp_ppo_v2",
        "domain": domain,
        "train_seeds": list(train_seeds),
        "eval_seeds": list(eval_seeds),
        "iterations": int(iterations),
        "best_score": float(best_score),
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
            "policy_ratios": "independent upper and lower PPO ratios",
        },
        **metadata,
    }
    return payload, heldout_rows, model
