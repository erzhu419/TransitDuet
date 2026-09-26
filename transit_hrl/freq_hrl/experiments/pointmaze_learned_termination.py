"""On-policy, fixed-budget learned termination for the Stage-9 PointMaze task."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.domains.mujoco import RelativeSubgoalAdapter
from freq_hrl.rl.dual_actor_critic import BernoulliActor, ValueNet

from .pointmaze_budgeted_trigger import (
    POINTMAZE_BUDGETED_TRIGGER_PROTOCOL_VERSION,
    balanced_jitter_schedule,
    build_parser as build_stage9_parser,
)
from .pointmaze_goal_validation import (
    POINTMAZE_LOWER_ACTION_COST,
    _json_ready,
    pointmaze_goal_bounds,
    squash_box_action,
)
from .pointmaze_plan_validity_branching import (
    PointMazeRegimeFeatureBuilder,
    _causal_plan_features,
    _task_options,
    _validate_branch_seed_roles,
)
from .pointmaze_plan_value_qualification import (
    _make_task,
    train_pointmaze_plan_value_cell,
)


PROTOCOL_VERSION = "pointmaze_learned_termination_stage10_v1"
ALGORITHM_PATH = "on_policy_bernoulli_termination_fixed_upper_call_budget"
FEATURE_DIM = 38  # 37 causal plan features and the within-bin offset.


class TerminationPPO:
    def __init__(
        self,
        *,
        seed: int,
        hidden_dim: int = 64,
        learning_rate: float = 3e-4,
    ) -> None:
        torch.manual_seed(int(seed) + 10_003)
        self.actor = BernoulliActor(FEATURE_DIM, hidden_dim, init_logit=-1.0)
        self.critic = ValueNet(FEATURE_DIM, hidden_dim)
        self.optimizer = torch.optim.Adam(
            [*self.actor.parameters(), *self.critic.parameters()],
            lr=float(learning_rate),
        )
        self.rng = np.random.default_rng(int(seed) + 10_019)
        self.mean = np.zeros(FEATURE_DIM, dtype=np.float32)
        self.scale = np.ones(FEATURE_DIM, dtype=np.float32)
        self.normalized = False
        self.optimizer_steps = 0

    def set_normalizer(self, trajectories: Iterable[list[dict[str, Any]]]) -> None:
        states = np.asarray(
            [record["state"] for episode in trajectories for record in episode],
            dtype=np.float32,
        )
        if states.ndim != 2 or states.shape[1] != FEATURE_DIM:
            raise ValueError("termination warm-up has no valid decision states")
        self.mean = states.mean(axis=0)
        self.scale = np.maximum(states.std(axis=0), 0.1)
        self.normalized = True

    def act(self, raw_state: np.ndarray, *, sample: bool) -> dict[str, Any]:
        state = (
            (raw_state - self.mean) / self.scale
            if self.normalized else raw_state
        )
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32).view(1, -1)
            action, logp = self.actor(x, sample=sample)
            value = self.critic(x)
        return {
            "state": state.astype(np.float32),
            "action": float(action.item()),
            "logp": float(logp.item()),
            "value": float(value.item()),
        }

    def update(self, trajectories: Iterable[list[dict[str, Any]]]) -> dict[str, float]:
        records: list[dict[str, Any]] = []
        for episode in trajectories:
            advantage = 0.0
            next_value = 0.0
            for record in reversed(episode):
                delta = record["reward"] + next_value - record["value"]
                advantage = delta + 0.95 * advantage
                records.append({
                    **record,
                    "advantage": advantage,
                    "return_target": advantage + record["value"],
                })
                next_value = record["value"]
        if not records:
            raise ValueError("termination PPO has no decisions")
        states = torch.as_tensor(np.stack([r["state"] for r in records]))
        actions = torch.as_tensor(
            [[r["action"]] for r in records], dtype=torch.float32,
        )
        old_logp = torch.as_tensor(
            [r["logp"] for r in records], dtype=torch.float32,
        )
        advantages = torch.as_tensor(
            [r["advantage"] for r in records], dtype=torch.float32,
        )
        targets = torch.as_tensor(
            [r["return_target"] for r in records], dtype=torch.float32,
        )
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-6
        )
        last_loss = 0.0
        for _ in range(4):
            for indices in np.array_split(
                self.rng.permutation(len(records)),
                max(1, int(np.ceil(len(records) / 256))),
            ):
                batch = torch.as_tensor(indices, dtype=torch.long)
                logp, entropy = self.actor.log_prob_entropy(
                    states[batch], actions[batch]
                )
                ratio = torch.exp(logp - old_logp[batch])
                surrogate = torch.minimum(
                    ratio * advantages[batch],
                    torch.clamp(ratio, 0.8, 1.2) * advantages[batch],
                )
                value_loss = torch.mean(
                    (self.critic(states[batch]) - targets[batch]) ** 2
                )
                loss = -surrogate.mean() + 0.5 * value_loss - 0.01 * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [*self.actor.parameters(), *self.critic.parameters()], 1.0
                )
                self.optimizer.step()
                self.optimizer_steps += 1
                last_loss = float(loss.item())
        return {"loss": last_loss, "decision_count": float(len(records))}


def termination_state(
    *,
    observation: Any,
    history: PointMazeRegimeFeatureBuilder,
    subgoal: np.ndarray,
    plan_age_steps: int,
    offset: int,
    max_offset_steps: int,
    time_scale: PhysicalTimeScaleContract,
) -> np.ndarray:
    names, values, _ = _causal_plan_features(
        observation=observation,
        feature_builder=history,
        subgoal=subgoal,
        plan_age_steps=plan_age_steps,
        time_scale=time_scale,
    )
    if any("regime" in name or "oracle" in name for name in names):
        raise ValueError("termination policy received privileged context")
    state = np.asarray([*values, offset / max_offset_steps], dtype=np.float32)
    if state.shape != (FEATURE_DIM,) or not np.all(np.isfinite(state)):
        raise ValueError("termination state contract changed")
    return state


def rollout_learned_termination(
    controller: Any,
    policy: TerminationPPO,
    *,
    seed: int,
    sample: bool,
    env_id: str,
    horizon: int,
    time_scale: PhysicalTimeScaleContract,
    maximum_subgoal_delta: float,
    max_offset_steps: int,
    check_stride_steps: int,
    task_options: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    period = time_scale.upper_period_steps
    if (
        horizon % period or max_offset_steps >= period
        or max_offset_steps % check_stride_steps
    ):
        raise ValueError("termination timing contract changed")
    task = _make_task(env_id=env_id, seed=seed, horizon=horizon, **task_options)
    try:
        observation = task.reset()
        world_low, world_high = pointmaze_goal_bounds(task.environment)
        adapter = RelativeSubgoalAdapter(
            maximum_delta=np.full(
                observation.achieved_goal.size, maximum_subgoal_delta,
                dtype=np.float32,
            ),
            world_low=world_low,
            world_high=world_high,
            action_cost=POINTMAZE_LOWER_ACTION_COST,
        )
        history = PointMazeRegimeFeatureBuilder(
            time_scale=time_scale,
            task_dim=int(observation.task_measurement.size),
        )
        history.reset(observation)
        controller.reset_recurrent_inference()
        achieved_before = observation.achieved_goal.copy()
        subgoal = achieved_before.copy()
        last_plan_step = -1
        planned_bin = -1
        decisions: list[int] = []
        trajectory: list[dict[str, Any]] = []
        current_record: dict[str, Any] | None = None
        segment_reward = 0.0
        episode_return = 0.0
        squared_error_sum = 0.0
        early_calls = 0
        score_checks = 0
        for step in range(horizon):
            bin_index = step // period
            offset = step - bin_index * period
            plan_now = step == 0
            if step > 0 and bin_index != planned_bin:
                if offset <= max_offset_steps and offset % check_stride_steps == 0:
                    if offset == max_offset_steps:
                        plan_now = True
                    else:
                        if current_record is not None:
                            current_record["reward"] = segment_reward
                            trajectory.append(current_record)
                        segment_reward = 0.0
                        action = policy.act(termination_state(
                            observation=observation,
                            history=history,
                            subgoal=subgoal,
                            plan_age_steps=step - last_plan_step,
                            offset=offset,
                            max_offset_steps=max_offset_steps,
                            time_scale=time_scale,
                        ), sample=sample)
                        current_record = {**action, "reward": 0.0}
                        score_checks += 1
                        plan_now = bool(action["action"])
                        if plan_now:
                            early_calls += 1
            if plan_now:
                upper_state = history.upper_state(
                    observation, oracle_context=None
                )
                output = controller.plan_goal(upper_state, sample=False)
                subgoal = adapter.decode(
                    np.asarray(output["action"], dtype=np.float32),
                    achieved_before,
                )
                decisions.append(step)
                planned_bin = bin_index
                last_plan_step = step
            lower_state = history.lower_state(observation, subgoal=subgoal)
            output = controller.act_conditioned(lower_state, sample=False)
            action = squash_box_action(
                np.asarray(output["action"], dtype=np.float32),
                task.action_low,
                task.action_high,
            )
            next_observation, reward, terminated, truncated, info = task.step(
                action
            )
            if (terminated or truncated) and step + 1 != horizon:
                raise RuntimeError("termination episode ended early")
            error = float(info["tracking_distance"]) ** 2 * time_scale.dt_seconds
            squared_error_sum += error
            segment_reward -= error
            episode_return += float(reward)
            achieved_before = next_observation.achieved_goal.copy()
            observation = next_observation
            history.update(observation)
        if current_record is not None:
            current_record["reward"] = segment_reward
            trajectory.append(current_record)
        durations = np.diff([*decisions, horizon])
        if (
            len(decisions) != horizon // period
            or [step // period for step in decisions]
            != list(range(horizon // period))
            or min(durations) < period - max_offset_steps
            or max(durations) > period + max_offset_steps
        ):
            raise RuntimeError("termination policy changed the upper-call budget")
        return ({
            "seed": int(seed),
            "mode": "learned_termination_ppo",
            "episode_length": int(horizon),
            "episode_return": episode_return,
            "tracking_squared_error_integral": squared_error_sum,
            "upper_decision_count": len(decisions),
            "decision_steps": decisions,
            "option_duration_steps_min": int(min(durations)),
            "option_duration_steps_max": int(max(durations)),
            "trigger_score_checks": score_checks,
            "trigger_early_calls": early_calls,
            "planner_called_only_on_decision": True,
            "has_privileged_regime_input": False,
            "protocol_valid": True,
        }, trajectory)
    finally:
        task.environment.close()


def train_cell(args: argparse.Namespace) -> dict[str, Any]:
    training, selection, branch_fit, trigger_eval = _validate_branch_seed_roles(
        train_seeds=args.train_seeds,
        selection_seeds=args.selection_seeds,
        branch_fit_seeds=args.branch_fit_seeds,
        branch_eval_seeds=args.trigger_eval_seeds,
    )
    task_options = _task_options(args)
    time_scale = PhysicalTimeScaleContract(
        dt_seconds=0.01,
        upper_period_seconds=args.upper_period_seconds,
        history_seconds=args.history_seconds,
        fast_period_seconds=args.fast_period_seconds,
    )

    def schedule(seed: int) -> tuple[int, ...]:
        return balanced_jitter_schedule(
            seed=seed,
            horizon=args.horizon,
            period_steps=time_scale.upper_period_steps,
            max_offset_steps=args.max_offset_steps,
        )

    controller_payload, controller = train_pointmaze_plan_value_cell(
        method="hrl_regime_history",
        env_id=args.env_id,
        train_seeds=training,
        selection_seeds=selection,
        eval_seeds=(*branch_fit, *trigger_eval),
        iterations=args.iterations,
        horizon=args.horizon,
        optimizer_seed=args.optimizer_seed,
        upper_period_seconds=args.upper_period_seconds,
        history_seconds=args.history_seconds,
        fast_period_seconds=args.fast_period_seconds,
        maximum_subgoal_delta=args.maximum_subgoal_delta,
        reference_hidden_dim=args.reference_hidden_dim,
        learning_rate=args.learning_rate,
        checkpoint_evaluation_interval=args.checkpoint_evaluation_interval,
        waypoint_perturbation=0.25,
        event_window_seconds=1.0,
        task_options=task_options,
        diagnostic_schedules=("fixed",),
        training_decision_steps_fn=schedule,
        training_schedule_name="balanced_jitter",
    )
    policy = TerminationPPO(
        seed=args.optimizer_seed,
        hidden_dim=args.termination_hidden_dim,
        learning_rate=args.termination_learning_rate,
    )
    initial_weights = torch.cat([
        parameter.detach().flatten() for parameter in policy.actor.parameters()
    ])
    rollouts = 0
    warm_trajectories = []
    for seed in branch_fit:
        _, trajectory = rollout_learned_termination(
            controller, policy, seed=seed, sample=True,
            env_id=args.env_id, horizon=args.horizon, time_scale=time_scale,
            maximum_subgoal_delta=args.maximum_subgoal_delta,
            max_offset_steps=args.max_offset_steps,
            check_stride_steps=args.check_stride_steps,
            task_options=task_options,
        )
        warm_trajectories.append(trajectory)
        rollouts += 1
    policy.set_normalizer(warm_trajectories)
    validation_history = []
    best_ise = float("inf")
    best_actor = None
    interval = max(1, args.termination_iterations // 4)
    for iteration in range(1, args.termination_iterations + 1):
        trajectories = []
        for seed in branch_fit:
            _, trajectory = rollout_learned_termination(
                controller, policy, seed=seed, sample=True,
                env_id=args.env_id, horizon=args.horizon, time_scale=time_scale,
                maximum_subgoal_delta=args.maximum_subgoal_delta,
                max_offset_steps=args.max_offset_steps,
                check_stride_steps=args.check_stride_steps,
                task_options=task_options,
            )
            trajectories.append(trajectory)
            rollouts += 1
        update = policy.update(trajectories)
        if iteration % interval == 0 or iteration == args.termination_iterations:
            rows = []
            for seed in selection:
                row, _ = rollout_learned_termination(
                    controller, policy, seed=seed, sample=False,
                    env_id=args.env_id, horizon=args.horizon, time_scale=time_scale,
                    maximum_subgoal_delta=args.maximum_subgoal_delta,
                    max_offset_steps=args.max_offset_steps,
                    check_stride_steps=args.check_stride_steps,
                    task_options=task_options,
                )
                rows.append(row)
                rollouts += 1
            mean_ise = float(np.mean([
                row["tracking_squared_error_integral"] for row in rows
            ]))
            validation_history.append({
                "iteration": iteration,
                "mean_tracking_ise": mean_ise,
                "mean_early_calls": float(np.mean([
                    row["trigger_early_calls"] for row in rows
                ])),
                "last_update_loss": update["loss"],
            })
            if mean_ise < best_ise:
                best_ise = mean_ise
                best_actor = copy.deepcopy(policy.actor.state_dict())
                best_iteration = iteration
    if best_actor is None:
        raise RuntimeError("termination actor has no selected checkpoint")
    weight_change = float(torch.linalg.vector_norm(
        torch.cat([
            parameter.detach().flatten() for parameter in policy.actor.parameters()
        ]) - initial_weights
    ).item())
    policy.actor.load_state_dict(best_actor)
    evaluation = [
        rollout_learned_termination(
            controller, policy, seed=seed, sample=False,
            env_id=args.env_id, horizon=args.horizon, time_scale=time_scale,
            maximum_subgoal_delta=args.maximum_subgoal_delta,
            max_offset_steps=args.max_offset_steps,
            check_stride_steps=args.check_stride_steps,
            task_options=task_options,
        )[0]
        for seed in trigger_eval
    ]
    fixed = [
        row for row in controller_payload["evaluation_rows"]
        if int(row["seed"]) in set(trigger_eval)
    ]
    return {
        "optimizer_seed": args.optimizer_seed,
        "protocol_version": PROTOCOL_VERSION,
        "algorithm_path": ALGORITHM_PATH,
        "train_seeds": list(training),
        "selection_seeds": list(selection),
        "branch_fit_seeds": list(branch_fit),
        "trigger_eval_seeds": list(trigger_eval),
        "controller_selected_iteration": controller_payload[
            "selected_checkpoint_iteration"
        ],
        "controller_gradient_updates_train": controller_payload[
            "gradient_updates_train"
        ],
        "controller_parameter_count": controller_payload["capacity"][
            "actual_parameter_count"
        ],
        "runtime_versions": controller_payload["runtime_versions"],
        "termination_actor_parameter_count": sum(
            parameter.numel() for parameter in policy.actor.parameters()
        ),
        "termination_critic_parameter_count": sum(
            parameter.numel() for parameter in policy.critic.parameters()
        ),
        "termination_optimizer_steps": policy.optimizer_steps,
        "termination_actor_weight_change_norm": weight_change,
        "termination_training_iterations": args.termination_iterations,
        "termination_rollouts_before_eval": rollouts,
        "termination_extra_primitive_steps": rollouts * args.horizon,
        "termination_selected_iteration": best_iteration,
        "termination_validation_history": validation_history,
        "termination_state_feature_count": FEATURE_DIM,
        "termination_state_contract": "causal_plan_features_plus_offset_fraction",
        "fixed_replay_rows": fixed,
        "learned_termination_evaluation_rows": evaluation,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = build_stage9_parser()
    parser.description = __doc__
    parser.add_argument("--termination-iterations", type=int, default=20)
    parser.add_argument("--termination-hidden-dim", type=int, default=64)
    parser.add_argument("--termination-learning-rate", type=float, default=3e-4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.termination_iterations < 1:
        raise ValueError("termination training requires an update")
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "algorithm_path": ALGORITHM_PATH,
        "source_controller_protocol": POINTMAZE_BUDGETED_TRIGGER_PROTOCOL_VERSION,
        "evidence_role": "learned_termination_development",
        "optimizer_seed": args.optimizer_seed,
        "environment_id": args.env_id,
        "iterations": args.iterations,
        "horizon": args.horizon,
        "upper_period_seconds": args.upper_period_seconds,
        "history_seconds": args.history_seconds,
        "fast_period_seconds": args.fast_period_seconds,
        "maximum_subgoal_delta": args.maximum_subgoal_delta,
        "reference_hidden_dim": args.reference_hidden_dim,
        "learning_rate": args.learning_rate,
        "checkpoint_evaluation_interval": args.checkpoint_evaluation_interval,
        "max_offset_steps": args.max_offset_steps,
        "check_stride_steps": args.check_stride_steps,
        "termination_iterations": args.termination_iterations,
        "termination_hidden_dim": args.termination_hidden_dim,
        "termination_learning_rate": args.termination_learning_rate,
        "train_seeds": args.train_seeds,
        "selection_seeds": args.selection_seeds,
        "branch_fit_seeds": args.branch_fit_seeds,
        "trigger_eval_seeds": args.trigger_eval_seeds,
        "task_options": _task_options(args),
        "training_objective": "negative_tracking_squared_error_integral",
        "termination_training": "on_policy_ppo_frozen_controller",
        "planning_budget": "exactly_one_upper_call_per_50_step_bin",
    }
    output = {
        "status": "dry_run" if args.dry_run else "complete",
        "protocol": protocol,
        "cells": [] if args.dry_run else [train_cell(args)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_ready(output), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
