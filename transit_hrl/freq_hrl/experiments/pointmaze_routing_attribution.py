"""Stage-4 PointMaze attribution of selective frequency routing inside HRL."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from freq_hrl.core import CausalHaarMultiscaleEncoder, PhysicalTimeScaleContract
from freq_hrl.domains.mujoco import (
    environment_timing,
    goal_environment_contract,
    parse_goal_observation,
)
from freq_hrl.rl import (
    GoalConditionedActorCriticPPO,
    GoalConditionedPPOConfig,
    flat_actor_critic_parameter_count,
    matched_hierarchical_hidden_dim,
    summarize_numeric_rows,
    train_frequency_separated_ppo,
)

from .pointmaze_goal_validation import (
    DEFAULT_ENV_ID,
    POINTMAZE_CHECKPOINT_RANK_CONTRACT,
    POINTMAZE_CHECKPOINT_RANK_NAMES,
    POINTMAZE_CONTINUING_TASK,
    compact_pointmaze_history,
    make_pointmaze_environment,
    pointmaze_checkpoint_rank,
    pointmaze_runtime_versions,
)
from .pointmaze_multiscale_validation import (
    POINTMAZE_STRESS_SPECS,
    _json_ready,
    _stress_for_episode,
    _training_seed,
    _validate_seed_roles,
    pointmaze_multiscale_dimensions,
    rollout_hrl_pointmaze_multiscale,
)


POINTMAZE_ROUTING_PROTOCOL_VERSION = "pointmaze_frequency_routing_stage4_v1"
POINTMAZE_ROUTING_ALGORITHM_PATH = "pointmaze_frequency_routing_attribution"
POINTMAZE_ROUTING_METHODS = (
    "hrl_history",
    "hrl_causal_filter",
    "hrl_multiscale_all",
    "hrl_multiscale_routed",
    "hrl_multiscale_swapped",
)
POINTMAZE_ROUTING_SCENARIOS = (
    "clean",
    "fast_observation_noise",
    "slow_drift_fast_action",
)
POINTMAZE_ROUTING_REPRESENTATIONS = {
    "hrl_history": "history",
    "hrl_causal_filter": "filtered",
    "hrl_multiscale_all": "multiscale_all",
    "hrl_multiscale_routed": "multiscale_routed",
    "hrl_multiscale_swapped": "multiscale_swapped",
}


def routing_representation(method: str) -> str:
    try:
        return POINTMAZE_ROUTING_REPRESENTATIONS[str(method)]
    except KeyError as exc:
        raise ValueError(f"unknown PointMaze routing method: {method}") from exc


def build_pointmaze_routing_model(
    *,
    method: str,
    dimensions: dict[str, Any],
    reference_hidden_dim: int,
    learning_rate: float,
    optimizer_seed: int,
) -> tuple[GoalConditionedActorCriticPPO, dict[str, Any]]:
    representation = routing_representation(method)
    reference = dimensions["history"]
    selected = dimensions[representation]
    target_parameters = flat_actor_critic_parameter_count(
        reference.flat,
        reference.action,
        int(reference_hidden_dim),
    )
    hidden_dim, expected, ratio = matched_hierarchical_hidden_dim(
        target_parameter_count=target_parameters,
        upper_state_dim=selected.upper,
        lower_state_dim=selected.lower,
        goal_dim=selected.goal,
        action_dim=selected.action,
    )
    torch.manual_seed(int(optimizer_seed))
    np.random.seed(int(optimizer_seed) % (2**32 - 1))
    model = GoalConditionedActorCriticPPO(GoalConditionedPPOConfig(
        upper_state_dim=selected.upper,
        lower_state_dim=selected.lower,
        goal_dim=selected.goal,
        action_dim=selected.action,
        hidden_dim=hidden_dim,
        learning_rate=float(learning_rate),
        epochs=4,
        minibatch_size=1024,
        init_log_std=-0.7,
    ))
    actual = model.trainable_parameter_count
    if actual != expected:
        raise RuntimeError("PointMaze routing parameter accounting changed")
    return model, {
        "reference_parameter_budget": target_parameters,
        "actual_parameter_count": actual,
        "parameter_budget_ratio": ratio,
        "hidden_dim": hidden_dim,
        **model.mainline_contract(),
    }


def train_pointmaze_routing_cell(
    *,
    method: str,
    scenario: str,
    env_id: str,
    train_seeds: Iterable[int],
    selection_seeds: Iterable[int],
    eval_seeds: Iterable[int],
    iterations: int,
    horizon: int,
    optimizer_seed: int,
    upper_period_seconds: float,
    history_seconds: float,
    fast_period_seconds: float,
    maximum_subgoal_delta: float,
    reference_hidden_dim: int = 128,
    learning_rate: float = 3e-4,
    checkpoint_evaluation_interval: int = 96,
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    representation = routing_representation(method)
    if str(scenario) not in POINTMAZE_ROUTING_SCENARIOS:
        raise ValueError(f"unknown PointMaze routing scenario: {scenario}")
    training, selection, evaluation = _validate_seed_roles(
        train_seeds, selection_seeds, eval_seeds
    )
    probe = make_pointmaze_environment(env_id=env_id, horizon=horizon)
    try:
        observation, _ = probe.reset(seed=0)
        contract = goal_environment_contract(probe)
        timing = environment_timing(probe)
        parsed = parse_goal_observation(observation)
        stress_probe = _stress_for_episode(
            scenario=scenario,
            seed=0,
            timing=timing,
            observation=observation,
            action_dim=int(np.prod(probe.action_space.shape)),
            horizon=horizon,
        )
    finally:
        probe.close()
    time_scale = PhysicalTimeScaleContract(
        dt_seconds=float(contract["env_dt_seconds"]),
        upper_period_seconds=float(upper_period_seconds),
        history_seconds=float(history_seconds),
        fast_period_seconds=float(fast_period_seconds),
    )
    if time_scale.upper_period_steps >= int(horizon):
        raise ValueError("PointMaze upper period must be shorter than the horizon")
    representations = tuple(dict.fromkeys(
        ("history", *POINTMAZE_ROUTING_REPRESENTATIONS.values())
    ))
    dimensions = pointmaze_multiscale_dimensions(
        env_id=env_id,
        horizon=horizon,
        time_scale=time_scale,
        representations=representations,
    )
    model, capacity = build_pointmaze_routing_model(
        method=method,
        dimensions=dimensions,
        reference_hidden_dim=reference_hidden_dim,
        learning_rate=learning_rate,
        optimizer_seed=optimizer_seed,
    )
    parameter_budget = int(capacity["reference_parameter_budget"])
    encoder = CausalHaarMultiscaleEncoder(
        feature_dim=int(parsed.physical.size),
        time_scale=time_scale,
    )
    common_metadata = {
        "protocol_version": POINTMAZE_ROUTING_PROTOCOL_VERSION,
        "algorithm_path": POINTMAZE_ROUTING_ALGORITHM_PATH,
        "environment_id": str(env_id),
        "scenario": str(scenario),
        "optimizer_seed": int(optimizer_seed),
        "representation": representation,
        "hierarchical": True,
        "reward_type": "dense",
        "continuing_task": POINTMAZE_CONTINUING_TASK,
        "reset_target": False,
        "task_reward_contract": "fixed_horizon_exp_negative_goal_distance_v1",
        "success_is_primary_endpoint": True,
        "checkpoint_objective": POINTMAZE_CHECKPOINT_RANK_CONTRACT,
        "goal_semantics": "upper_relative_xy_waypoint_lower_physical_acceleration",
        "lower_final_goal_visibility": "hidden; lower receives only waypoint error",
        "history_information_contract": (
            "same_fixed_trailing_actor_visible_samples_with_raw_filter_or_haar"
        ),
        "current_physical_feedback_contract": (
            "both_hierarchy_levels_retain_full_current_actor_visible_physical_state"
        ),
        "routing_attribution_contract": {
            "history": "raw_history_to_both_levels",
            "filtered": "causal_filtered_history_to_both_levels",
            "multiscale_all": "all_haar_bands_to_both_levels",
            "multiscale_routed": "slow_mid_upper_and_mid_high_lower",
            "multiscale_swapped": "mid_high_upper_and_slow_mid_lower",
        },
        "stress_observability_contract": (
            "actor_never_receives_measurement_or_action_disturbance_truth"
        ),
        "stress": stress_probe.metadata(),
        "feature_dimensions": {
            key: asdict(value) for key, value in dimensions.items()
        },
        "band_layout": encoder.band_layout(),
        "time_scale": time_scale.metadata(),
        "capacity": capacity,
        "environment_contract": contract,
        "runtime_versions": pointmaze_runtime_versions(),
        "maximum_subgoal_delta": float(maximum_subgoal_delta),
        "lower_intrinsic_reward_contract": (
            "actor_visible_waypoint_progress_minus_requested_action_cost_v1"
        ),
        "lower_credit_boundary_contract": "waypoint_change_or_episode_end_v1",
        "projector": "disabled",
        "promotion": "disabled",
        "leakage_loss": "disabled",
        "responsibility_gauge": "disabled",
    }
    seed_fn = lambda root, iteration: _training_seed(
        optimizer_seed=optimizer_seed,
        rollout_root=root,
        iteration=iteration,
    )
    payload, rows, trained = train_frequency_separated_ppo(
        model=model,
        train_seeds=training,
        selection_seeds=selection,
        eval_seeds=evaluation,
        iterations=int(iterations),
        rollout_fn=lambda policy, seed, sample: rollout_hrl_pointmaze_multiscale(
            policy,
            method=str(method),
            scenario=str(scenario),
            env_id=str(env_id),
            seed=seed,
            horizon=int(horizon),
            sample=sample,
            parameter_budget=parameter_budget,
            time_scale=time_scale,
            maximum_subgoal_delta=maximum_subgoal_delta,
            representation_override=representation,
            protocol_version=POINTMAZE_ROUTING_PROTOCOL_VERSION,
            algorithm_path=POINTMAZE_ROUTING_ALGORITHM_PATH,
        ),
        objective_fn=lambda row: float(row["episode_return"]),
        summary_fn=summarize_numeric_rows,
        training_seed_fn=seed_fn,
        policy=str(method),
        domain="pointmaze_frequency_routing_attribution",
        metadata=common_metadata,
        checkpoint_score_contract="mean_dense_episode_return",
        checkpoint_rank_fn=pointmaze_checkpoint_rank,
        checkpoint_rank_names=POINTMAZE_CHECKPOINT_RANK_NAMES,
        checkpoint_rank_contract=POINTMAZE_CHECKPOINT_RANK_CONTRACT,
        checkpoint_minimum_iteration=0,
        checkpoint_evaluation_interval=int(checkpoint_evaluation_interval),
    )
    payload["training_core"] = payload["trainer"]
    payload["trainer"] = "goal_conditioned_smdp_ppo_v1"
    payload["trajectory_contract"]["lower"] = (
        "one primitive transition with GAE terminated at waypoint change or episode end"
    )
    for row in rows:
        row["training_replicate_seed"] = int(optimizer_seed)
    payload["history"] = compact_pointmaze_history(payload["history"])
    payload["history_schema"] = "pointmaze_compact_training_history_v1"
    payload["optimizer_seed"] = int(optimizer_seed)
    payload["evaluation_rows"] = rows
    return payload, rows, trained


def resolved_pointmaze_routing_protocol(
    *,
    methods: Iterable[str],
    scenarios: Iterable[str],
    env_id: str,
    iterations: int,
    horizon: int,
    optimizer_seed: int,
    upper_period_seconds: float,
    history_seconds: float,
    fast_period_seconds: float,
    maximum_subgoal_delta: float,
    reference_hidden_dim: int,
    learning_rate: float,
    checkpoint_evaluation_interval: int,
    train_seeds: Iterable[int],
    selection_seeds: Iterable[int],
    eval_seeds: Iterable[int],
) -> dict[str, Any]:
    training, selection, evaluation = _validate_seed_roles(
        train_seeds, selection_seeds, eval_seeds
    )
    method_names = list(map(str, methods))
    scenario_names = list(map(str, scenarios))
    if not method_names or any(name not in POINTMAZE_ROUTING_METHODS for name in method_names):
        raise ValueError("PointMaze routing protocol has an invalid method set")
    if not scenario_names or any(name not in POINTMAZE_ROUTING_SCENARIOS for name in scenario_names):
        raise ValueError("PointMaze routing protocol has an invalid scenario set")
    return {
        "protocol_version": POINTMAZE_ROUTING_PROTOCOL_VERSION,
        "algorithm_path": POINTMAZE_ROUTING_ALGORITHM_PATH,
        "methods": method_names,
        "representations": {
            method: routing_representation(method) for method in method_names
        },
        "scenarios": scenario_names,
        "primary_stress_scenarios": [
            "fast_observation_noise",
            "slow_drift_fast_action",
        ],
        "primary_endpoint": "success_rate",
        "supportive_endpoints": ["episode_return", "final_goal_distance"],
        "clean_noninferiority_margin_success": 0.10,
        "environment_id": str(env_id),
        "reward_type": "dense",
        "continuing_task": POINTMAZE_CONTINUING_TASK,
        "reset_target": False,
        "iterations": int(iterations),
        "horizon": int(horizon),
        "optimizer_seed": int(optimizer_seed),
        "upper_period_seconds": float(upper_period_seconds),
        "history_seconds": float(history_seconds),
        "fast_period_seconds": float(fast_period_seconds),
        "maximum_subgoal_delta": float(maximum_subgoal_delta),
        "reference_hidden_dim": int(reference_hidden_dim),
        "learning_rate": float(learning_rate),
        "checkpoint_evaluation_interval": int(checkpoint_evaluation_interval),
        "stress_specs": {
            name: asdict(POINTMAZE_STRESS_SPECS[name]) for name in scenario_names
        },
        "train_seeds": training,
        "selection_seeds": selection,
        "eval_seeds": evaluation,
        "claim_gate": {
            "routed_vs_all_success": "positive_in_both_primary_stresses",
            "routed_vs_swapped_success": "positive_in_both_primary_stresses",
            "observation_routed_vs_filter_success": "positive",
            "clean_routed_vs_all_success": "noninferior_margin_0.10",
        },
        "disabled_legacy_mechanisms": [
            "action_spectrum_projector",
            "promotion",
            "leakage_loss",
            "responsibility_gauge",
            "projection_consistency",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the PointMaze Stage-4 frequency-routing attribution."
    )
    parser.add_argument("--methods", nargs="+", choices=POINTMAZE_ROUTING_METHODS, default=list(POINTMAZE_ROUTING_METHODS))
    parser.add_argument("--scenarios", nargs="+", choices=POINTMAZE_ROUTING_SCENARIOS, default=list(POINTMAZE_ROUTING_SCENARIOS))
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--iterations", type=int, default=768)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--optimizer-seed", type=int, default=104007)
    parser.add_argument("--upper-period-seconds", type=float, default=0.25)
    parser.add_argument("--history-seconds", type=float, default=0.32)
    parser.add_argument("--fast-period-seconds", type=float, default=0.04)
    parser.add_argument("--maximum-subgoal-delta", type=float, default=0.75)
    parser.add_argument("--reference-hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-evaluation-interval", type=int, default=96)
    parser.add_argument("--train-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--selection-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--eval-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = resolved_pointmaze_routing_protocol(
        methods=args.methods,
        scenarios=args.scenarios,
        env_id=args.env_id,
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
        train_seeds=args.train_seeds,
        selection_seeds=args.selection_seeds,
        eval_seeds=args.eval_seeds,
    )
    output: dict[str, Any] = {
        "protocol": protocol,
        "status": "dry_run" if args.dry_run else "complete",
        "cells": [],
    }
    if not args.dry_run:
        for scenario in args.scenarios:
            for method in args.methods:
                payload, _, _ = train_pointmaze_routing_cell(
                    method=method,
                    scenario=scenario,
                    env_id=args.env_id,
                    train_seeds=args.train_seeds,
                    selection_seeds=args.selection_seeds,
                    eval_seeds=args.eval_seeds,
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
                )
                output["cells"].append(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_ready(output), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
