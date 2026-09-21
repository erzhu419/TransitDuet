"""Equal-shape frequency attribution on the separate PointMaze external stream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from freq_hrl.core import CausalHaarMultiscaleEncoder, PhysicalTimeScaleContract
from freq_hrl.domains.mujoco import goal_environment_contract
from freq_hrl.rl import (
    summarize_numeric_rows,
    train_frequency_separated_ppo,
    train_joint_ppo,
)

from .pointmaze_exogenous_validation import (
    DEFAULT_FORCE_PERIOD_SECONDS,
    DEFAULT_FORCE_RMS,
    DEFAULT_POINTMAZE_EXOGENOUS_CHECKPOINT_RANK_MODE,
    DEFAULT_TARGET_SPEED,
    POINTMAZE_EXOGENOUS_HISTORY_FIELDS,
    PointMazeExogenousDimensions,
    build_pointmaze_exogenous_model,
    pointmaze_exogenous_checkpoint_rank,
    pointmaze_exogenous_checkpoint_rank_spec,
    pointmaze_exogenous_dimensions,
    rollout_flat_pointmaze_exogenous,
    rollout_hrl_pointmaze_exogenous,
)
from .pointmaze_goal_validation import (
    DEFAULT_ENV_ID,
    _json_ready,
    _training_seed,
    _validate_seed_roles,
    make_pointmaze_environment,
    pointmaze_goal_bounds,
    pointmaze_runtime_versions,
)


POINTMAZE_EXOGENOUS_ROUTING_PROTOCOL_VERSION = (
    "pointmaze_exogenous_frequency_routing_stage6_v1"
)
POINTMAZE_EXOGENOUS_ROUTING_ALGORITHM_PATH = (
    "external_multiscale_goal_conditioned_hrl_attribution"
)
POINTMAZE_EXOGENOUS_ROUTING_METHOD_SPECS = {
    "flat_exogenous_history": ("flat", "history"),
    "flat_exogenous_filtered": ("flat", "filtered"),
    "flat_exogenous_multiscale_all": ("flat", "multiscale_all"),
    "hrl_exogenous_history": ("hrl", "history"),
    "hrl_exogenous_filtered": ("hrl", "filtered"),
    "hrl_exogenous_multiscale_all": ("hrl", "multiscale_all"),
    "hrl_exogenous_multiscale_routed": (
        "hrl", "multiscale_routed_masked"
    ),
    "hrl_exogenous_multiscale_swapped": (
        "hrl", "multiscale_swapped_masked"
    ),
}
POINTMAZE_EXOGENOUS_ROUTING_METHODS = tuple(
    POINTMAZE_EXOGENOUS_ROUTING_METHOD_SPECS
)
POINTMAZE_EXOGENOUS_ROUTING_CONTRACT = {
    "history": "raw_external_history_to_policy",
    "filtered": "causal_low_pass_external_history_to_policy",
    "multiscale_all": "all_external_haar_bands_to_policy",
    "multiscale_routed_masked": (
        "slow_mid_external_bands_to_upper_mid_high_to_lower"
    ),
    "multiscale_swapped_masked": (
        "mid_high_external_bands_to_upper_slow_mid_to_lower"
    ),
}
POINTMAZE_EXOGENOUS_ROUTING_SHAPE_CONTRACT = (
    "fixed_134d_states_and_identical_architecture_initialization_within_level_type"
)


def exogenous_routing_method_spec(method: str) -> tuple[str, str]:
    try:
        return POINTMAZE_EXOGENOUS_ROUTING_METHOD_SPECS[str(method)]
    except KeyError as exc:
        raise ValueError(f"unknown external routing method: {method}") from exc


def compact_exogenous_routing_history(
    history: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items()
         if key in POINTMAZE_EXOGENOUS_HISTORY_FIELDS}
        for row in history
    ]


def build_pointmaze_exogenous_routing_model(
    *,
    method: str,
    dimensions: PointMazeExogenousDimensions,
    reference_hidden_dim: int,
    learning_rate: float,
    optimizer_seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Build one architecture-matched model independent of representation."""

    architecture, _ = exogenous_routing_method_spec(method)
    base_method = (
        "flat_exogenous_history"
        if architecture == "flat" else "hrl_exogenous_history"
    )
    return build_pointmaze_exogenous_model(
        method=base_method,
        dimensions=dimensions,
        reference_hidden_dim=reference_hidden_dim,
        learning_rate=learning_rate,
        optimizer_seed=optimizer_seed,
    )


def train_pointmaze_exogenous_routing_cell(
    *,
    method: str,
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
    target_speed: float = DEFAULT_TARGET_SPEED,
    force_rms: float = DEFAULT_FORCE_RMS,
    force_period_seconds: tuple[float, float] = DEFAULT_FORCE_PERIOD_SECONDS,
    reference_hidden_dim: int = 128,
    learning_rate: float = 3e-4,
    checkpoint_evaluation_interval: int = 96,
    checkpoint_rank_mode: str = (
        DEFAULT_POINTMAZE_EXOGENOUS_CHECKPOINT_RANK_MODE
    ),
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    architecture, representation = exogenous_routing_method_spec(method)
    training, selection, evaluation = _validate_seed_roles(
        train_seeds, selection_seeds, eval_seeds
    )
    probe = make_pointmaze_environment(env_id=env_id, horizon=horizon)
    try:
        contract = goal_environment_contract(probe)
        world_low, world_high = pointmaze_goal_bounds(probe)
    finally:
        probe.close()
    time_scale = PhysicalTimeScaleContract(
        dt_seconds=float(contract["env_dt_seconds"]),
        upper_period_seconds=float(upper_period_seconds),
        history_seconds=float(history_seconds),
        fast_period_seconds=float(fast_period_seconds),
    )
    dimensions = pointmaze_exogenous_dimensions(
        env_id=env_id,
        horizon=horizon,
        time_scale=time_scale,
        target_speed=target_speed,
        force_rms=force_rms,
        force_period_seconds=force_period_seconds,
        representation=representation,
    )
    model, capacity = build_pointmaze_exogenous_routing_model(
        method=method,
        dimensions=dimensions,
        reference_hidden_dim=reference_hidden_dim,
        learning_rate=learning_rate,
        optimizer_seed=optimizer_seed,
    )
    checkpoint_rank_names, checkpoint_rank_contract = (
        pointmaze_exogenous_checkpoint_rank_spec(checkpoint_rank_mode)
    )
    encoder = CausalHaarMultiscaleEncoder(
        feature_dim=dimensions.task,
        time_scale=time_scale,
    )
    parameter_budget = int(capacity["reference_parameter_budget"])
    common_metadata = {
        "protocol_version": POINTMAZE_EXOGENOUS_ROUTING_PROTOCOL_VERSION,
        "algorithm_path": POINTMAZE_EXOGENOUS_ROUTING_ALGORITHM_PATH,
        "environment_id": str(env_id),
        "optimizer_seed": int(optimizer_seed),
        "architecture": architecture,
        "representation": representation,
        "routing_attribution_contract": dict(
            POINTMAZE_EXOGENOUS_ROUTING_CONTRACT
        ),
        "routing_shape_contract": POINTMAZE_EXOGENOUS_ROUTING_SHAPE_CONTRACT,
        "task_reward_contract": (
            "fixed_horizon_exp_negative_dynamic_target_distance_v1"
        ),
        "primary_endpoint": "tracking_success_rate",
        "checkpoint_objective": checkpoint_rank_contract,
        "checkpoint_rank_mode": str(checkpoint_rank_mode),
        "state_contract": (
            "current_physical_z_plus_routed_separate_external_x_history_v1"
        ),
        "external_stream_contract": (
            "action_independent_slow_route_target_plus_fast_measured_force_v1"
        ),
        "history_information_contract": (
            "same_fixed_causal_external_samples_raw_filter_or_orthonormal_haar"
        ),
        "goal_semantics": (
            "flat_physical_action_or_upper_relative_xy_waypoint_lower_action"
        ),
        "lower_final_target_visibility": (
            "no_explicit_target_error_external_representation_only"
            if architecture == "hrl" else "not_applicable"
        ),
        "projector": "disabled",
        "promotion": "disabled",
        "leakage_loss": "disabled",
        "responsibility_gauge": "disabled",
        "dimensions": dimensions.__dict__,
        "band_layout": encoder.band_layout(),
        "capacity": capacity,
        "environment_contract": contract,
        "runtime_versions": pointmaze_runtime_versions(),
        "world_low": world_low.tolist(),
        "world_high": world_high.tolist(),
        "time_scale": time_scale.metadata(),
        "maximum_subgoal_delta": float(maximum_subgoal_delta),
        "target_speed_world_per_second": float(target_speed),
        "force_rms_per_axis": float(force_rms),
        "force_period_seconds": list(map(float, force_period_seconds)),
        "lower_intrinsic_reward_contract": (
            "waypoint_distance_progress_minus_action_cost_v1"
            if architecture == "hrl" else "not_applicable"
        ),
        "lower_credit_boundary_contract": (
            "waypoint_change_or_episode_end_v1"
            if architecture == "hrl" else "not_applicable"
        ),
    }
    common_rollout = {
        "env_id": str(env_id),
        "horizon": int(horizon),
        "parameter_budget": parameter_budget,
        "time_scale": time_scale,
        "target_speed": float(target_speed),
        "force_rms": float(force_rms),
        "force_period_seconds": tuple(map(float, force_period_seconds)),
        "method": str(method),
        "representation": representation,
        "protocol_version": POINTMAZE_EXOGENOUS_ROUTING_PROTOCOL_VERSION,
        "algorithm_path": POINTMAZE_EXOGENOUS_ROUTING_ALGORITHM_PATH,
    }
    if architecture == "flat":
        rollout_fn = lambda policy, seed, sample: rollout_flat_pointmaze_exogenous(
            policy, seed=seed, sample=sample, **common_rollout
        )
    else:
        rollout_fn = lambda policy, seed, sample: rollout_hrl_pointmaze_exogenous(
            policy,
            seed=seed,
            sample=sample,
            maximum_subgoal_delta=maximum_subgoal_delta,
            **common_rollout,
        )
    untrained_rows = [
        rollout_fn(model, int(seed), False)[1] for seed in evaluation
    ]
    for row in untrained_rows:
        row["training_replicate_seed"] = int(optimizer_seed)
    seed_fn = lambda root, iteration: _training_seed(
        optimizer_seed=optimizer_seed,
        rollout_root=root,
        iteration=iteration,
    )
    trainer_kwargs = {
        "model": model,
        "train_seeds": training,
        "selection_seeds": selection,
        "eval_seeds": evaluation,
        "iterations": int(iterations),
        "rollout_fn": rollout_fn,
        "objective_fn": lambda row: float(row["episode_return"]),
        "summary_fn": summarize_numeric_rows,
        "training_seed_fn": seed_fn,
        "policy": str(method),
        "domain": "pointmaze_exogenous_frequency_routing_attribution",
        "metadata": common_metadata,
        "checkpoint_score_contract": "mean_dense_episode_return",
        "checkpoint_rank_fn": lambda rows: pointmaze_exogenous_checkpoint_rank(
            rows, mode=checkpoint_rank_mode
        ),
        "checkpoint_rank_names": checkpoint_rank_names,
        "checkpoint_rank_contract": checkpoint_rank_contract,
        "checkpoint_minimum_iteration": 0,
        "checkpoint_evaluation_interval": int(checkpoint_evaluation_interval),
    }
    if architecture == "flat":
        payload, rows, trained = train_joint_ppo(**trainer_kwargs)
    else:
        payload, rows, trained = train_frequency_separated_ppo(**trainer_kwargs)
        payload["training_core"] = payload["trainer"]
        payload["trainer"] = "goal_conditioned_smdp_ppo_v1"
        payload["trajectory_contract"]["lower"] = (
            "one primitive transition with GAE terminated at waypoint change "
            "or episode end"
        )
    for row in rows:
        row["training_replicate_seed"] = int(optimizer_seed)
    payload["history"] = compact_exogenous_routing_history(payload["history"])
    payload["history_schema"] = (
        "pointmaze_exogenous_routing_compact_training_history_v1"
    )
    payload["optimizer_seed"] = int(optimizer_seed)
    payload["untrained_evaluation_rows"] = untrained_rows
    payload["evaluation_rows"] = rows
    return payload, rows, trained


def resolved_pointmaze_exogenous_routing_protocol(
    *,
    methods: Iterable[str],
    env_id: str,
    iterations: int,
    horizon: int,
    optimizer_seed: int,
    upper_period_seconds: float,
    history_seconds: float,
    fast_period_seconds: float,
    maximum_subgoal_delta: float,
    target_speed: float,
    force_rms: float,
    force_period_seconds: tuple[float, float],
    reference_hidden_dim: int,
    learning_rate: float,
    checkpoint_evaluation_interval: int,
    checkpoint_rank_mode: str,
    train_seeds: Iterable[int],
    selection_seeds: Iterable[int],
    eval_seeds: Iterable[int],
) -> dict[str, Any]:
    training, selection, evaluation = _validate_seed_roles(
        train_seeds, selection_seeds, eval_seeds
    )
    names = list(map(str, methods))
    if not names or any(
        name not in POINTMAZE_EXOGENOUS_ROUTING_METHOD_SPECS for name in names
    ):
        raise ValueError("external routing protocol has an unknown method")
    _, checkpoint_rank_contract = pointmaze_exogenous_checkpoint_rank_spec(
        checkpoint_rank_mode
    )
    return {
        "protocol_version": POINTMAZE_EXOGENOUS_ROUTING_PROTOCOL_VERSION,
        "algorithm_path": POINTMAZE_EXOGENOUS_ROUTING_ALGORITHM_PATH,
        "methods": names,
        "method_specs": {
            name: list(exogenous_routing_method_spec(name)) for name in names
        },
        "routing_attribution_contract": dict(
            POINTMAZE_EXOGENOUS_ROUTING_CONTRACT
        ),
        "routing_shape_contract": POINTMAZE_EXOGENOUS_ROUTING_SHAPE_CONTRACT,
        "environment_id": str(env_id),
        "iterations": int(iterations),
        "horizon": int(horizon),
        "optimizer_seed": int(optimizer_seed),
        "upper_period_seconds": float(upper_period_seconds),
        "history_seconds": float(history_seconds),
        "fast_period_seconds": float(fast_period_seconds),
        "maximum_subgoal_delta": float(maximum_subgoal_delta),
        "target_speed_world_per_second": float(target_speed),
        "force_rms_per_axis": float(force_rms),
        "force_period_seconds": list(map(float, force_period_seconds)),
        "reference_hidden_dim": int(reference_hidden_dim),
        "learning_rate": float(learning_rate),
        "checkpoint_evaluation_interval": int(checkpoint_evaluation_interval),
        "checkpoint_rank_mode": str(checkpoint_rank_mode),
        "checkpoint_rank_contract": checkpoint_rank_contract,
        "primary_endpoint": "tracking_success_rate",
        "train_seeds": training,
        "selection_seeds": selection,
        "eval_seeds": evaluation,
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
        description="Run external-stream PointMaze frequency attribution."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=POINTMAZE_EXOGENOUS_ROUTING_METHODS,
        default=list(POINTMAZE_EXOGENOUS_ROUTING_METHODS),
    )
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--iterations", type=int, default=768)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--optimizer-seed", type=int, default=184007)
    parser.add_argument("--upper-period-seconds", type=float, default=0.25)
    parser.add_argument("--history-seconds", type=float, default=0.32)
    parser.add_argument("--fast-period-seconds", type=float, default=0.04)
    parser.add_argument("--maximum-subgoal-delta", type=float, default=0.75)
    parser.add_argument("--target-speed", type=float, default=DEFAULT_TARGET_SPEED)
    parser.add_argument("--force-rms", type=float, default=DEFAULT_FORCE_RMS)
    parser.add_argument(
        "--force-period-seconds",
        nargs=2,
        type=float,
        default=list(DEFAULT_FORCE_PERIOD_SECONDS),
    )
    parser.add_argument("--reference-hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-evaluation-interval", type=int, default=96)
    parser.add_argument(
        "--checkpoint-rank-mode",
        choices=("success_then_return", "return_then_success"),
        default=DEFAULT_POINTMAZE_EXOGENOUS_CHECKPOINT_RANK_MODE,
    )
    parser.add_argument("--train-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--selection-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--eval-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    force_periods = tuple(map(float, args.force_period_seconds))
    protocol = resolved_pointmaze_exogenous_routing_protocol(
        methods=args.methods,
        env_id=args.env_id,
        iterations=args.iterations,
        horizon=args.horizon,
        optimizer_seed=args.optimizer_seed,
        upper_period_seconds=args.upper_period_seconds,
        history_seconds=args.history_seconds,
        fast_period_seconds=args.fast_period_seconds,
        maximum_subgoal_delta=args.maximum_subgoal_delta,
        target_speed=args.target_speed,
        force_rms=args.force_rms,
        force_period_seconds=force_periods,
        reference_hidden_dim=args.reference_hidden_dim,
        learning_rate=args.learning_rate,
        checkpoint_evaluation_interval=args.checkpoint_evaluation_interval,
        checkpoint_rank_mode=args.checkpoint_rank_mode,
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
        for method in args.methods:
            payload, _, _ = train_pointmaze_exogenous_routing_cell(
                method=method,
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
                target_speed=args.target_speed,
                force_rms=args.force_rms,
                force_period_seconds=force_periods,
                reference_hidden_dim=args.reference_hidden_dim,
                learning_rate=args.learning_rate,
                checkpoint_evaluation_interval=(
                    args.checkpoint_evaluation_interval
                ),
                checkpoint_rank_mode=args.checkpoint_rank_mode,
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
