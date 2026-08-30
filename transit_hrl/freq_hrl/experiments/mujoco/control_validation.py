"""Shared-core Freq-HRL validation on Gymnasium MuJoCo control tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

from freq_hrl.core import (
    CausalRollingBandTracker,
    CausalSmoothstepMacroPlan,
    LeakageRegularizer,
    evaluate_rms_leakage_budget,
)
from freq_hrl.domains.mujoco import (
    CausalBandDecomposer,
    CausalLowerActionRouter,
    CausalResponsibilityTransfer,
    DISTURBANCE_MODES,
    LOWER_ACTION_ROUTER_MODES,
    RESPONSIBILITY_MODES,
    action_from_unit_box,
    deterministic_actuation_disturbance,
    lower_action_router_contract,
)
from freq_hrl.experiments.reproducibility import (
    derive_seed,
    training_rollout_seed,
    validate_evaluation_seed_roles,
    validate_unique_seeds,
    verify_current_freq_hrl_source_identity,
)
from freq_hrl.rl import (
    DEPLOYMENT_FREQUENCY_PROJECTION_OBJECTIVES,
    FrequencySeparatedActorCriticPPO,
    HierarchicalRolloutBuilder,
    HierarchicalTrajectoryBatch,
    JointActorCriticPPO,
    JointPPOConfig,
    JointTrajectoryBatch,
    SMDPPPOConfig,
    summarize_numeric_rows,
    train_frequency_separated_ppo,
    train_joint_ppo,
)


MUJOCO_CONTROL_PROTOCOL_VERSION = (
    "freq_hrl_mujoco_shared_core_v14_15_closed_loop_restoration_filter"
)
MUJOCO_CONTROL_PROTOCOL_VERSION_V14_16 = (
    "freq_hrl_mujoco_shared_core_v14_16_crossed_pathwise_restoration"
)
MUJOCO_CONTROL_PROTOCOL_VERSION_V14_17 = (
    "freq_hrl_mujoco_shared_core_v14_17_native_pd_cvar"
)
MUJOCO_CONTROL_PROTOCOL_VERSION_V16_2 = (
    "freq_hrl_mujoco_shared_core_v16_2_macro_hold_gauge"
)
MUJOCO_CONTROL_PROTOCOL_VERSION_V17 = (
    "freq_hrl_mujoco_shared_core_v17_zero_dc_plan"
)
MUJOCO_CONTROL_PROTOCOL_VERSION_V17_1 = (
    "freq_hrl_mujoco_shared_core_v17_1_headroom_homotopy_promotion"
)
MUJOCO_CONTROL_PROTOCOL_VERSION_V17_2 = (
    "freq_hrl_mujoco_shared_core_v17_2_smooth_macro_gauge"
)
MUJOCO_CONTROL_PROTOCOL_VERSION_V17_3 = (
    "freq_hrl_mujoco_shared_core_v17_3_audit_optimal_macro_gauge"
)
MUJOCO_CONTROL_PROTOCOL_VERSIONS = (
    MUJOCO_CONTROL_PROTOCOL_VERSION,
    MUJOCO_CONTROL_PROTOCOL_VERSION_V14_16,
    MUJOCO_CONTROL_PROTOCOL_VERSION_V14_17,
    MUJOCO_CONTROL_PROTOCOL_VERSION_V16_2,
    MUJOCO_CONTROL_PROTOCOL_VERSION_V17,
    MUJOCO_CONTROL_PROTOCOL_VERSION_V17_1,
    MUJOCO_CONTROL_PROTOCOL_VERSION_V17_2,
    MUJOCO_CONTROL_PROTOCOL_VERSION_V17_3,
)
MUJOCO_CONTROL_PROTOCOL_SELECTIONS = (
    "auto",
    *MUJOCO_CONTROL_PROTOCOL_VERSIONS,
)
METHODS = (
    "freq_hrl",
    "freq_hrl_safe_selector",
    "freq_hrl_no_leakage",
    "generic_hrl",
    "flat_ppo",
)
UPPER_ACTION_DECODER_MODES = (
    "hold",
    "causal_smoothstep_plan",
)
DEFAULT_ENV_IDS = ("HalfCheetah-v5", "Hopper-v5", "Walker2d-v5")
DEFAULT_TRAIN_SEEDS = (31013, 31019, 31033, 31039)
DEFAULT_SELECTION_SEEDS = (32003, 32009, 32027, 32029)
CLOSED_LOOP_RISK_MODES = (
    "legacy",
    "mode_mean",
    "pathwise_all",
    "mode_cvar",
)


def _resolve_closed_loop_risk_mode(
    *, pathwise_robust: bool, risk_mode: str
) -> str:
    mode = str(risk_mode)
    if mode not in CLOSED_LOOP_RISK_MODES:
        raise ValueError("unknown closed-loop risk mode")
    if mode == "legacy":
        return "pathwise_all" if pathwise_robust else "mode_mean"
    if pathwise_robust and mode != "pathwise_all":
        raise ValueError(
            "pathwise_robust conflicts with the explicit closed-loop risk mode"
        )
    return mode


def _empirical_upper_tail_cvar(
    values: Iterable[float], *, alpha: float
) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("CVaR values must be finite and non-empty")
    tail_count = max(1, int(np.ceil((1.0 - float(alpha)) * array.size)))
    return float(np.mean(np.partition(array, array.size - tail_count)[-tail_count:]))


def deployment_frequency_constraint_contract(
    *,
    requested: bool,
    groupwise: bool,
    anchor_state_replay: bool,
    ppo_trust_region: bool,
    closed_loop_trust_region: bool = False,
    closed_loop_restoration_filter: bool = False,
    projection_objective: str = "worst_group",
    projection_cvar_alpha: float = 0.5,
    restoration_freeze_reward_actor: bool = False,
    pathwise_robust: bool = False,
    closed_loop_risk_mode: str = "legacy",
    closed_loop_cvar_alpha: float = 0.5,
) -> str:
    if not requested:
        return "disabled"
    objective = str(projection_objective)
    if objective not in DEPLOYMENT_FREQUENCY_PROJECTION_OBJECTIVES:
        raise ValueError("unknown deployment-frequency projection objective")
    resolved_risk_mode = _resolve_closed_loop_risk_mode(
        pathwise_robust=pathwise_robust,
        risk_mode=closed_loop_risk_mode,
    )
    for label, alpha in (
        ("projection", projection_cvar_alpha),
        ("closed-loop", closed_loop_cvar_alpha),
    ):
        if not np.isfinite(float(alpha)) or not 0.0 <= float(alpha) < 1.0:
            raise ValueError(f"{label} CVaR alpha must be in [0, 1)")
    if (
        objective != "worst_group"
        or restoration_freeze_reward_actor
        or pathwise_robust
        or str(closed_loop_risk_mode) != "legacy"
    ):
        mechanisms = [
            "episode_reset",
            "groupwise" if groupwise else "pooled",
            objective,
            "frozen_anchor_state_replay"
            if anchor_state_replay else "current_state_only",
            "ppo_trust_region"
            if ppo_trust_region else "unfiltered_ppo_step",
            "crossed_closed_loop_filter"
            if closed_loop_trust_region else "no_closed_loop_filter",
            "restoration_filter"
            if closed_loop_restoration_filter else "no_restoration_filter",
            "frozen_reward_actor_during_restoration"
            if restoration_freeze_reward_actor
            else "joint_reward_actor_during_restoration",
        ]
        new_risk_contract = (
            objective == "violation_cvar"
            or resolved_risk_mode == "mode_cvar"
        )
        if objective == "violation_cvar":
            mechanisms.append(
                f"projection_cvar_alpha_{float(projection_cvar_alpha):.6g}"
            )
        mechanisms.extend([
            {
                "pathwise_all": "individual_path_constraints",
                "mode_mean": "mode_mean_constraints",
                "mode_cvar": (
                    "mode_cvar_constraints_alpha_"
                    f"{float(closed_loop_cvar_alpha):.6g}"
                ),
            }[resolved_risk_mode],
            "v10" if new_risk_contract else "v9",
        ])
        return "_".join(mechanisms)
    if not groupwise:
        return (
            "episode_reset_differentiable_actor_mean_tanh_upper_hold_hpf8_"
            "lower_lpf32_anchor_relative_target_with_absolute_floor_and_"
            "dimensionless_iterative_cumulative_reward_budget_projection_v4"
        )
    base = (
        "episode_reset_groupwise_worst_differentiable_actor_mean_tanh_upper_"
        "hold_hpf8_lower_lpf32_per_group_anchor_relative_target_with_"
        "absolute_floor"
    )
    if closed_loop_trust_region:
        inner_mechanisms = ""
        if anchor_state_replay:
            inner_mechanisms += "frozen_anchor_state_replay_"
        if ppo_trust_region:
            inner_mechanisms += "ppo_trust_region_"
        closed_loop_contract = (
            base
            + "_"
            + inner_mechanisms
            + "independent_crossed_closed_loop_reward_floor_and_five_"
            "frequency_endpoint_joint_actor_backtracking_v7"
        )
        if closed_loop_restoration_filter:
            return (
                closed_loop_contract
                + "_infeasible_start_merit_restoration_and_feasible_"
                "maintenance_filter_v8"
            )
        return closed_loop_contract
    if anchor_state_replay and ppo_trust_region:
        return (
            "episode_reset_candidate_and_frozen_anchor_state_replay_"
            "groupwise_worst_differentiable_actor_mean_tanh_upper_hold_hpf8_"
            "lower_lpf32_per_group_anchor_relative_target_with_absolute_"
            "floor_ppo_trust_region_and_iterative_per_group_cumulative_"
            "reward_budget_projection_v6"
        )
    if anchor_state_replay:
        return (
            base
            + "_frozen_anchor_state_replay_iterative_per_group_cumulative_"
            "reward_budget_projection_v6"
        )
    if ppo_trust_region:
        return (
            base
            + "_ppo_trust_region_and_iterative_per_group_cumulative_reward_"
            "budget_projection_v6"
        )
    return (
        base
        + "_iterative_per_group_cumulative_reward_budget_projection_v5"
    )
DEFAULT_SAFETY_SELECTION_SEEDS = (
    32503, 32507, 32531, 32533, 32537, 32561, 32563, 32569,
)
DEFAULT_EVAL_SEEDS = (33013, 33023, 33029, 33037, 33049)
DEFAULT_TRAINING_DISTURBANCE_MODES = (
    "standard",
    "low_frequency",
    "high_frequency",
    "mixed",
)
SAFE_SELECTOR_BASELINE_BRANCH = "no_leakage"
SAFE_SELECTOR_BRANCHES = (
    SAFE_SELECTOR_BASELINE_BRANCH,
    "responsibility_guarded_adam_projection",
    "behavior_guarded_adam_projection",
    "behavior_guarded_upper_hf",
    "behavior_scalarized_upper_hf",
)
SAFE_SELECTOR_REWARD_MARGIN_FRACTION = 0.02
SAFE_SELECTOR_MIN_DRIFT_REDUCTION_FRACTION = 0.10
SAFE_SELECTOR_MIN_RAW_DRIFT_REDUCTION_FRACTION = 0.10
SAFE_SELECTOR_MAX_UPPER_HF_RMS = 0.10
SAFE_SELECTOR_CONFIDENCE = 0.90
SAFE_SELECTOR_BOOTSTRAP_DRAWS = 4096
RESPONSIBILITY_TRANSFER_ALPHA = 0.04
DEFAULT_LOWER_ROUTER_ALPHA = 0.10
DEFAULT_LOWER_ROUTER_STRENGTH = 1.0
LOWER_ACTION_ROUTER_TRAINING_SCHEDULES = (
    "constant",
    "delayed_linear",
    "delayed_cosine",
)
CAUSAL_LOWER_ACTION_ROUTER_MODES = {
    "causal_ema_high_pass",
    "causal_ema_conservative_transfer",
    "causal_joint_band_projection",
    "causal_total_action_gauge",
    "causal_audit_aligned_gauge",
    "causal_macro_hold_audit_gauge",
    "causal_smooth_macro_gauge",
    "causal_audit_optimal_macro_gauge",
    "causal_macro_zero_dc",
    "causal_macro_zero_dc_headroom",
}
FUNCTION_PRESERVING_LOWER_ACTION_ROUTER_MODES = {
    "causal_ema_conservative_transfer",
    "causal_joint_band_projection",
    "causal_total_action_gauge",
    "causal_audit_aligned_gauge",
    "causal_macro_hold_audit_gauge",
    "causal_smooth_macro_gauge",
    "causal_audit_optimal_macro_gauge",
}
LEAKAGE_CONSTRAINT_SCOPES = (
    "responsibility",
    "joint_behavior",
    "joint_behavior_latent",
)
LEAKAGE_COST_MODES = (
    "ratio_excess_squared",
    "power_excess",
)
UPPER_CONSTRAINT_MODES = (
    "disabled",
    "static_reward_penalty",
    "primal_dual",
)
CHECKPOINT_SELECTION_MODES = (
    "assigned_condition",
    "crossed_conditions",
)
CHECKPOINT_SCORE_MODES = (
    "mean_reward",
    "behavior_robust",
    "latent_behavior_robust",
    "latent_behavior_feasibility_first",
    "paired_relative_frequency_feasibility_first",
)
DEFAULT_UPPER_HF_RMS_BUDGET = 0.10
DEFAULT_UPPER_HF_PENALTY_COEF = 2.0
DEFAULT_UPPER_DUAL_LR = 0.1
DEFAULT_LOWER_DUAL_LR = 0.1
DEFAULT_CHECKPOINT_CONSTRAINT_PENALTY = 10.0


def _gym() -> Any:
    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Gymnasium with the MuJoCo extra is required for this validation"
        ) from exc
    return gym


def _make_env(env_id: str, *, episode_horizon: int) -> Any:
    if int(episode_horizon) < 1:
        raise ValueError("MuJoCo episode_horizon must be positive")
    env = _gym().make(
        str(env_id),
        render_mode=None,
        max_episode_steps=int(episode_horizon),
    )
    if len(env.observation_space.shape or ()) != 1:
        env.close()
        raise ValueError("MuJoCo validation requires a vector observation space")
    if len(env.action_space.shape or ()) != 1:
        env.close()
        raise ValueError("MuJoCo validation requires a vector Box action space")
    return env


def environment_dimensions(
    env_id: str,
    *,
    episode_horizon: int = 1000,
) -> tuple[int, int]:
    env = _make_env(env_id, episode_horizon=episode_horizon)
    try:
        return int(env.observation_space.shape[0]), int(env.action_space.shape[0])
    finally:
        env.close()


def _module_parameter_count(model: Any) -> int:
    parameters: dict[int, torch.nn.Parameter] = {}
    for value in vars(model).values():
        if isinstance(value, torch.nn.Module):
            for parameter in value.parameters():
                parameters[id(parameter)] = parameter
    if not parameters:
        raise ValueError("model exposes no trainable torch modules")
    return int(sum(parameter.numel() for parameter in parameters.values()))


def _model_parameter_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    seen: set[int] = set()
    for attribute, value in sorted(vars(model).items()):
        if not isinstance(value, torch.nn.Module):
            continue
        for name, parameter in sorted(value.named_parameters()):
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            array = parameter.detach().cpu().contiguous().numpy()
            digest.update(str(attribute).encode("utf-8") + b"\0")
            digest.update(str(name).encode("utf-8") + b"\0")
            digest.update(str(array.dtype).encode("ascii") + b"\0")
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    if not seen:
        raise ValueError("model exposes no parameters to hash")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_paired_mujoco_checkpoint(
    model: FrequencySeparatedActorCriticPPO,
    *,
    checkpoint_path: Path,
    summary_path: Path,
    env_id: str,
    optimizer_seed: int,
    expected_code_revision: str,
    expected_source_manifest_sha256: str,
    expected_method: str = "freq_hrl_no_leakage",
    expected_router_mode: str = "direct",
    expected_router_strength: float = 0.0,
    expected_router_observe_strength: bool = True,
    expected_responsibility_mode: str = "causal_lf_transfer",
    expected_protocol_version: str = MUJOCO_CONTROL_PROTOCOL_VERSION,
    reset_upper_deployment_frequency_lambda: float | None = None,
    reset_lower_deployment_frequency_lambda: float | None = None,
) -> dict[str, Any]:
    """Load and audit a matched continuation checkpoint."""

    checkpoint_file = Path(checkpoint_path)
    summary_file = Path(summary_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(checkpoint_file)
    if not summary_file.is_file():
        raise FileNotFoundError(summary_file)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("paired checkpoint summary must be a JSON object")
    router_strength = float(expected_router_strength)
    if not np.isfinite(router_strength) or not 0.0 <= router_strength <= 1.0:
        raise ValueError("paired checkpoint router strength must be in [0, 1]")
    file_sha256 = _file_sha256(checkpoint_file)
    expected_summary = {
        "protocol_version": str(expected_protocol_version),
        "method": str(expected_method),
        "environment": str(env_id),
        "optimizer_seed": int(optimizer_seed),
        "code_revision": str(expected_code_revision),
        "source_manifest_sha256": str(expected_source_manifest_sha256),
        "lower_action_router_mode": str(expected_router_mode),
        "lower_action_router_strength": router_strength,
        "lower_action_router_observe_strength": bool(
            expected_router_observe_strength
        ),
        "responsibility_mode": str(expected_responsibility_mode),
    }
    mismatches = {
        key: {"expected": value, "observed": summary.get(key)}
        for key, value in expected_summary.items()
        if summary.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "paired checkpoint summary contract mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if summary.get("checkpoint_file_sha256") != file_sha256:
        raise ValueError("paired checkpoint serialized SHA-256 mismatch")
    frozen_parameter_sha256 = str(
        summary.get("frozen_parameter_sha256", "")
    )
    if len(frozen_parameter_sha256) != 64:
        raise ValueError("paired checkpoint parameter SHA-256 is invalid")

    checkpoint = torch.load(
        checkpoint_file,
        map_location=model.device,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model_state_dict"), dict
    ):
        raise ValueError("paired checkpoint payload is invalid")
    checkpoint_contract = {
        "protocol_version": str(expected_protocol_version),
        "method": str(expected_method),
        "environment": str(env_id),
        "optimizer_seed": int(optimizer_seed),
        "frozen_parameter_sha256": frozen_parameter_sha256,
        "code_revision": str(expected_code_revision),
        "source_manifest_sha256": str(expected_source_manifest_sha256),
        "lower_action_router_mode": str(expected_router_mode),
        "lower_action_router_strength": router_strength,
        "lower_action_router_observe_strength": bool(
            expected_router_observe_strength
        ),
        "responsibility_mode": str(expected_responsibility_mode),
    }
    checkpoint_mismatches = {
        key: {"expected": value, "observed": checkpoint.get(key)}
        for key, value in checkpoint_contract.items()
        if checkpoint.get(key) != value
    }
    if checkpoint_mismatches:
        raise ValueError(
            "paired checkpoint payload contract mismatch: "
            + json.dumps(checkpoint_mismatches, sort_keys=True)
        )
    saved_config = dict(checkpoint["model_state_dict"].get("config") or {})
    architecture_keys = (
        "upper_state_dim",
        "lower_state_dim",
        "upper_action_dim",
        "lower_action_dim",
        "hidden_dim",
        "state_encoder",
    )
    architecture_mismatches = {
        key: {
            "expected": getattr(model.config, key),
            "observed": saved_config.get(key),
        }
        for key in architecture_keys
        if saved_config.get(key) != getattr(model.config, key)
    }
    if architecture_mismatches:
        raise ValueError(
            "paired checkpoint architecture mismatch: "
            + json.dumps(architecture_mismatches, sort_keys=True)
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    loaded_parameter_sha256 = _model_parameter_sha256(model)
    if loaded_parameter_sha256 != frozen_parameter_sha256:
        raise ValueError("paired checkpoint parameter SHA-256 mismatch")
    loaded_upper_deployment_lambda = float(
        model.upper_deployment_frequency_lambda
    )
    loaded_lower_deployment_lambda = float(
        model.lower_deployment_frequency_lambda
    )
    reset_values = {
        "upper": reset_upper_deployment_frequency_lambda,
        "lower": reset_lower_deployment_frequency_lambda,
    }
    for level, value in reset_values.items():
        if value is None:
            continue
        maximum = float(getattr(
            model.config, f"{level}_deployment_frequency_max_lambda"
        ))
        numeric = float(value)
        if not np.isfinite(numeric) or not 0.0 <= numeric <= maximum:
            raise ValueError(
                f"paired {level} deployment-frequency lambda reset must "
                "be finite and within its configured maximum"
            )
        setattr(model, f"{level}_deployment_frequency_lambda", numeric)
    return {
        "enabled": True,
        "checkpoint_path": str(checkpoint_file),
        "summary_path": str(summary_file),
        "checkpoint_file_sha256": file_sha256,
        "checkpoint_parameter_sha256": loaded_parameter_sha256,
        "checkpoint_selected_iteration": int(
            summary["selected_checkpoint_iteration"]
        ),
        "checkpoint_optimizer_seed": int(optimizer_seed),
        "checkpoint_environment": str(env_id),
        "checkpoint_protocol_version": str(expected_protocol_version),
        "checkpoint_router_mode": str(expected_router_mode),
        "checkpoint_router_strength": router_strength,
        "checkpoint_router_observe_strength": bool(
            expected_router_observe_strength
        ),
        "checkpoint_responsibility_mode": str(expected_responsibility_mode),
        "loaded_upper_deployment_frequency_lambda": (
            loaded_upper_deployment_lambda
        ),
        "loaded_lower_deployment_frequency_lambda": (
            loaded_lower_deployment_lambda
        ),
        "reset_upper_deployment_frequency_lambda": (
            float(reset_upper_deployment_frequency_lambda)
            if reset_upper_deployment_frequency_lambda is not None
            else loaded_upper_deployment_lambda
        ),
        "reset_lower_deployment_frequency_lambda": (
            float(reset_lower_deployment_frequency_lambda)
            if reset_lower_deployment_frequency_lambda is not None
            else loaded_lower_deployment_lambda
        ),
        "deployment_frequency_state_reset_contract": (
            "new_constraint_duals_optionally_reset_to_registered_"
            "continuation_initial_values_after_actor_critic_optimizer_load_v2"
        ),
    }


@torch.no_grad()
def _value_prediction(
    value_net: torch.nn.Module,
    state: np.ndarray,
    *,
    device: torch.device,
) -> float:
    tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).view(1, -1)
    return float(value_net(tensor).item())


def _with_explicit_bootstrap(
    level: Any,
    *,
    boundary_next_values: list[float],
    boundary_terminals: list[float],
    boundary_next_cost_values: list[float] | None = None,
    cost_values: list[float] | None = None,
) -> Any:
    done = np.asarray(level.done, dtype=np.float32).reshape(-1)
    values = np.asarray(level.old_value, dtype=np.float32).reshape(-1)
    boundary_indices = np.flatnonzero(done > 0.5)
    if (
        boundary_indices.size != len(boundary_next_values)
        or boundary_indices.size != len(boundary_terminals)
    ):
        raise RuntimeError("MuJoCo bootstrap boundaries do not match trajectory")
    next_values = np.zeros_like(values)
    if values.size > 1:
        next_values[:-1] = values[1:]
    terminals = np.zeros_like(values)
    for index, next_value, terminal in zip(
        boundary_indices,
        boundary_next_values,
        boundary_terminals,
    ):
        next_values[int(index)] = float(next_value)
        terminals[int(index)] = float(terminal)
    next_cost_values = None
    if boundary_next_cost_values is not None:
        if (
            cost_values is None
            or len(cost_values) != values.size
            or boundary_indices.size != len(boundary_next_cost_values)
        ):
            raise RuntimeError(
                "MuJoCo cost bootstrap boundaries do not match trajectory"
            )
        next_cost_values = np.zeros_like(values)
        if values.size > 1:
            next_cost_values[:-1] = np.asarray(
                cost_values[1:], dtype=np.float32
            )
        for index, next_cost_value in zip(
            boundary_indices, boundary_next_cost_values
        ):
            next_cost_values[int(index)] = float(next_cost_value)
    replacement = {
        "next_value": next_values,
        "terminal": terminals,
    }
    if hasattr(level, "next_cost_value"):
        replacement["next_cost_value"] = next_cost_values
    elif next_cost_values is not None:
        raise TypeError("trajectory type does not support cost bootstrap")
    return replace(level, **replacement)


def _mlp_count(in_dim: int, out_dim: int, hidden_dim: int) -> int:
    hidden = int(hidden_dim)
    if hidden <= 0:
        return int(in_dim * out_dim + out_dim)
    return int(
        in_dim * hidden + hidden
        + hidden * hidden + hidden
        + hidden * out_dim + out_dim
    )


def _flat_ppo_parameter_count(
    state_dim: int,
    action_dim: int,
    hidden_dim: int,
) -> int:
    return int(
        _mlp_count(state_dim, action_dim, hidden_dim)
        + action_dim
        + _mlp_count(state_dim, 1, hidden_dim)
    )


def capacity_matched_flat_hidden_dim(
    *,
    target_parameter_count: int,
    state_dim: int,
    action_dim: int,
    maximum_hidden_dim: int = 512,
) -> tuple[int, int, float]:
    if int(target_parameter_count) <= 0:
        raise ValueError("target_parameter_count must be positive")
    candidates = [
        (
            abs(_flat_ppo_parameter_count(state_dim, action_dim, hidden) - int(
                target_parameter_count
            )),
            hidden,
            _flat_ppo_parameter_count(state_dim, action_dim, hidden),
        )
        for hidden in range(1, int(maximum_hidden_dim) + 1)
    ]
    _, hidden, actual = min(candidates)
    return hidden, actual, float(actual / int(target_parameter_count))


def mujoco_policy_state_dim(
    observation_dim: int,
    action_dim: int,
    *,
    observe_router_strength: bool = False,
    lower_action_router_mode: str = "direct",
) -> int:
    if int(observation_dim) < 1 or int(action_dim) < 1:
        raise ValueError("MuJoCo observation and action dimensions must be positive")
    mode = str(lower_action_router_mode)
    if mode not in LOWER_ACTION_ROUTER_MODES:
        raise ValueError("unknown MuJoCo lower-action router mode")
    filter_context_count = (
        3 if mode == "causal_audit_optimal_macro_gauge" else 2
    )
    router_scalar_count = (
        1 if mode == "causal_audit_optimal_macro_gauge" else 0
    )
    return (
        int(observation_dim)
        + (3 + filter_context_count) * int(action_dim)
        + router_scalar_count
        + int(bool(observe_router_strength))
    )


def lower_action_router_training_strength(
    *,
    iteration: int,
    total_iterations: int,
    target_strength: float,
    schedule: str,
    warmup_fraction: float,
    ramp_fraction: float,
) -> float:
    if int(total_iterations) < 1:
        raise ValueError("router schedule requires a positive iteration count")
    if not 0 <= int(iteration) < int(total_iterations):
        raise ValueError("router schedule iteration is out of range")
    target = float(target_strength)
    warmup = float(warmup_fraction)
    ramp = float(ramp_fraction)
    if not np.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("router target strength must be in [0, 1]")
    if (
        not np.isfinite(warmup)
        or not np.isfinite(ramp)
        or not 0.0 <= warmup <= 1.0
        or not 0.0 <= ramp <= 1.0
        or warmup + ramp > 1.0
    ):
        raise ValueError("router warmup and ramp fractions are invalid")
    mode = str(schedule)
    if mode not in LOWER_ACTION_ROUTER_TRAINING_SCHEDULES:
        raise ValueError("unknown lower-action router training schedule")
    if mode == "constant" or target == 0.0:
        return target
    if ramp <= 0.0:
        raise ValueError("a delayed router schedule requires a positive ramp")
    progress = float(int(iteration) + 1) / float(int(total_iterations))
    phase = float(np.clip((progress - warmup) / ramp, 0.0, 1.0))
    if mode == "delayed_cosine":
        phase = 0.5 - 0.5 * float(np.cos(np.pi * phase))
    return float(target * phase)


def _feature_state(
    observation: np.ndarray,
    bands: dict[str, np.ndarray],
    action_context: np.ndarray,
    *,
    filter_contexts: tuple[np.ndarray, ...] | None = None,
    router_scalar_contexts: tuple[float, ...] = (),
    router_strength_context: float | None = None,
    frequency_routing: bool,
    level: str,
) -> np.ndarray:
    endogenous = np.asarray(observation, dtype=np.float32).reshape(-1)
    context = np.asarray(action_context, dtype=np.float32).reshape(-1)
    if filter_contexts is None:
        filters = (np.zeros_like(context), np.zeros_like(context))
    else:
        filters = tuple(
            np.asarray(value, dtype=np.float32).reshape(-1)
            for value in filter_contexts
        )
        if len(filters) < 1 or any(
            value.shape != context.shape for value in filters
        ):
            raise ValueError("MuJoCo filter contexts must match the action context")
    if frequency_routing and level == "upper":
        exogenous = (bands["slow"], bands["mid"])
    elif frequency_routing and level == "lower":
        exogenous = (bands["mid"], bands["high"])
    else:
        exogenous = (bands["raw"], bands["delta"])
    strength = (
        ()
        if router_strength_context is None
        else (np.asarray([router_strength_context], dtype=np.float32),)
    )
    router_scalars = tuple(
        np.asarray([float(value)], dtype=np.float32)
        for value in router_scalar_contexts
    )
    if any(not np.all(np.isfinite(value)) for value in router_scalars):
        raise ValueError("MuJoCo router scalar contexts must be finite")
    pieces = (
        endogenous,
        *exogenous,
        context,
        *filters,
        *router_scalars,
        *strength,
    )
    return np.concatenate(pieces).astype(np.float32, copy=False)


def _leakage_constraint_cost(
    budget_info: dict[str, float | bool],
    *,
    mode: str,
) -> float:
    if str(mode) == "ratio_excess_squared":
        return float(budget_info["budget_excess_squared"])
    if str(mode) == "power_excess":
        return float(budget_info["power_excess"])
    raise ValueError(f"unknown leakage constraint cost mode: {mode}")


def _trace_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _episode_row(
    *,
    seed: int,
    env_id: str,
    disturbance_mode: str,
    rewards: list[float],
    executed_actions: list[np.ndarray],
    upper_actions: list[np.ndarray],
    lower_actions: list[np.ndarray],
    upper_policy_actions: list[np.ndarray],
    upper_promotion_actions: list[np.ndarray],
    upper_promotion_contexts: list[np.ndarray],
    raw_lower_actions: list[np.ndarray],
    latent_lower_actions: list[np.ndarray],
    lower_router_removed_actions: list[np.ndarray],
    lower_router_reconstruction_errors: list[np.ndarray],
    lower_router_clip_values: list[float],
    lower_router_audit_alphas: list[float],
    lower_router_audit_imbalances: list[float],
    lower_router_macro_projection_rates: list[float],
    lower_router_macro_debt_rms_values: list[float],
    lower_router_macro_completion_errors: list[float],
    lower_router_headroom_clip_values: list[float],
    responsibility_transfers: list[np.ndarray],
    requested_transfers: list[np.ndarray],
    transfer_saturation_values: list[float],
    reconstruction_errors: list[np.ndarray],
    forward_rewards: list[float],
    control_rewards: list[float],
    upper_decisions: int,
    upper_transitions: int,
    lower_transitions: int,
    method: str,
    segment_returns: list[float],
    natural_episode_returns: list[float],
    boundary_terminals: list[float],
    transition_budget: int,
    lower_lf_rms_values: list[float],
    lower_lf_budget_excesses: list[float],
    raw_lower_lf_rms_values: list[float],
    raw_lower_lf_budget_excesses: list[float],
    latent_lower_lf_rms_values: list[float],
    latent_lower_lf_budget_excesses: list[float],
    lower_constraint_costs: list[float],
    lower_lf_rms_budget: float,
    leakage_constraint_scope: str,
    upper_transition_delta_rms_values: list[float],
    upper_hf_rms_values: list[float],
    upper_hf_budget_excesses: list[float],
    upper_hf_penalties: list[float],
    upper_constraint_costs: list[float],
    upper_constraint_mode: str,
    upper_hf_rms_budget: float,
    upper_hf_penalty_coef: float,
    leakage_cost_mode: str,
    lower_action_router_mode: str,
    lower_action_router_strength: float,
    upper_promotion_gain: float,
) -> dict[str, Any]:
    executed = np.asarray(executed_actions, dtype=np.float64)
    upper = np.asarray(upper_actions, dtype=np.float64)
    lower = np.asarray(lower_actions, dtype=np.float64)
    upper_policy = np.asarray(upper_policy_actions, dtype=np.float64)
    upper_promotion = np.asarray(upper_promotion_actions, dtype=np.float64)
    upper_promotion_context = np.asarray(
        upper_promotion_contexts, dtype=np.float64
    )
    raw_lower = np.asarray(raw_lower_actions, dtype=np.float64)
    latent_lower = np.asarray(latent_lower_actions, dtype=np.float64)
    router_removed = np.asarray(
        lower_router_removed_actions, dtype=np.float64
    )
    router_reconstruction = np.asarray(
        lower_router_reconstruction_errors, dtype=np.float64
    )
    transfers = np.asarray(responsibility_transfers, dtype=np.float64)
    requested = np.asarray(requested_transfers, dtype=np.float64)
    reconstruction = np.asarray(reconstruction_errors, dtype=np.float64)
    leakage = LeakageRegularizer(
        upper_hf_window=8,
        lower_lf_window=32,
    ).compute(upper, lower)
    raw_leakage = LeakageRegularizer(
        upper_hf_window=8,
        lower_lf_window=32,
    ).compute(upper_policy, raw_lower)
    latent_leakage = LeakageRegularizer(
        upper_hf_window=8,
        lower_lf_window=32,
    ).compute(upper_policy, latent_lower)
    transfer_leakage = LeakageRegularizer(
        upper_hf_window=8,
        lower_lf_window=32,
    ).compute(np.zeros_like(transfers), transfers)
    upper_energy = float(np.mean(np.square(upper)))
    lower_energy = float(np.mean(np.square(lower)))
    responsibility_energy = upper_energy + lower_energy
    smoothness = (
        float(np.mean(np.square(np.diff(executed, axis=0))))
        if executed.shape[0] > 1 else 0.0
    )
    additive_action = upper + lower
    additive_clip_excess = np.maximum(
        np.abs(additive_action) - 1.0, 0.0
    )
    return {
        "seed": int(seed),
        "environment": str(env_id),
        "disturbance_mode": str(disturbance_mode),
        "method": str(method),
        "episode_return": float(np.sum(rewards)),
        "reward_mean": float(np.mean(rewards)),
        "RewardTraceSHA256": _trace_sha256(np.asarray(
            rewards, dtype=np.float64
        )),
        "ExecutedActionTraceSHA256": _trace_sha256(executed),
        "LatentPolicyTraceSHA256": _trace_sha256(
            upper_policy, latent_lower
        ),
        "episode_length": len(rewards),
        "rollout_segment_count": len(segment_returns),
        "rollout_segment_return_mean": float(np.mean(segment_returns)),
        "natural_episode_count": len(natural_episode_returns),
        "natural_episode_return_sum": float(np.sum(natural_episode_returns)),
        "trace_boundary_count": len(boundary_terminals),
        "mdp_terminal_count": int(np.sum(boundary_terminals)),
        "bootstrap_boundary_count": int(
            len(boundary_terminals) - np.sum(boundary_terminals)
        ),
        "transition_budget_exact": float(len(rewards) == int(transition_budget)),
        "forward_reward_sum": float(np.sum(forward_rewards)),
        "control_reward_sum": float(np.sum(control_rewards)),
        "action_energy": float(np.mean(np.square(executed))),
        "action_smoothness": smoothness,
        "UpperActionRMS": float(np.sqrt(upper_energy)),
        "LowerActionRMS": float(np.sqrt(lower_energy)),
        "UpperPolicyActionRMS": float(np.sqrt(np.mean(np.square(upper_policy)))),
        "UpperPromotionRMS": float(np.sqrt(
            np.mean(np.square(upper_promotion))
        )),
        "UpperPromotionActivationRate": float(np.mean(
            np.abs(upper_promotion) > 1e-12
        )),
        "UpperPromotionContextRMS": float(np.sqrt(
            np.mean(np.square(upper_promotion_context))
        )),
        "UpperPromotionGain": float(upper_promotion_gain),
        "RawLowerActionRMS": float(np.sqrt(np.mean(np.square(raw_lower)))),
        "LatentLowerActionRMS": float(np.sqrt(
            np.mean(np.square(latent_lower))
        )),
        "LowerRouterRemovedRMS": float(np.sqrt(
            np.mean(np.square(router_removed))
        )),
            "LowerRouterUpperTransferRMS": float(
                np.sqrt(np.mean(np.square(router_removed)))
                if str(lower_action_router_mode)
                in FUNCTION_PRESERVING_LOWER_ACTION_ROUTER_MODES
                else 0.0
            ),
            "LowerRouterFunctionPreserving": float(
                str(lower_action_router_mode)
                in FUNCTION_PRESERVING_LOWER_ACTION_ROUTER_MODES
            ),
        "LowerRouterActionReconstructionRMS": float(
            np.sqrt(np.mean(np.square(router_reconstruction)))
        ),
        "LowerRouterClipRate": float(np.mean(lower_router_clip_values)),
        "LowerRouterAuditAlphaMean": float(np.mean(
            lower_router_audit_alphas
        )),
        "LowerRouterAuditAlphaFinal": float(lower_router_audit_alphas[-1]),
        "LowerRouterAuditBandImbalanceMean": float(np.mean(
            lower_router_audit_imbalances
        )),
        "LowerRouterMacroProjectionRate": float(np.mean(
            lower_router_macro_projection_rates
        )),
        "LowerRouterMacroDebtRMSMean": float(np.mean(
            lower_router_macro_debt_rms_values
        )),
        "LowerRouterMacroCompletionErrorMax": float(np.max(
            lower_router_macro_completion_errors
        )),
        "LowerRouterHeadroomClipRate": float(np.mean(
            lower_router_headroom_clip_values
        )),
        "EffectiveToLatentLowerEnergyRatio": float(
            np.mean(np.square(raw_lower))
            / max(float(np.mean(np.square(latent_lower))), 1e-12)
        ),
        "UpperActionEnergyShare": float(
            upper_energy / responsibility_energy
            if responsibility_energy > 0.0 else 0.0
        ),
        "AdditiveActionClipRate": float(np.mean(
            additive_clip_excess > 1e-7
        )),
        "AdditiveActionClipExcessMax": float(np.max(
            additive_clip_excess
        )),
        "AdditiveActionClipExcessRMS": float(np.sqrt(np.mean(
            np.square(additive_clip_excess)
        ))),
        "UpperHFPower": float(leakage["UpperHFPower"]),
        "UpperHFPowerAbs": float(leakage["UpperHFPowerAbs"]),
        "LatentUpperHFPower": float(latent_leakage["UpperHFPower"]),
        "LatentUpperHFPowerAbs": float(
            latent_leakage["UpperHFPowerAbs"]
        ),
        "LowerLFDrift": float(leakage["LowerLFDrift"]),
        "LowerLFDriftAbs": float(leakage["LowerLFDriftAbs"]),
        "RawLowerLFDrift": float(raw_leakage["LowerLFDrift"]),
        "RawLowerLFDriftAbs": float(raw_leakage["LowerLFDriftAbs"]),
        "LatentLowerLFDrift": float(latent_leakage["LowerLFDrift"]),
        "LatentLowerLFDriftAbs": float(latent_leakage["LowerLFDriftAbs"]),
        "TransferredLFDriftAbs": float(transfer_leakage["LowerLFDriftAbs"]),
        "ResponsibilityTransferRMS": float(
            np.sqrt(np.mean(np.square(transfers)))
        ),
        "RequestedResponsibilityTransferRMS": float(
            np.sqrt(np.mean(np.square(requested)))
        ),
        "ResponsibilityTransferActivationRate": float(
            np.mean(np.abs(transfers) > 1e-12)
        ),
        "ResponsibilityTransferHeadroomSaturationRate": float(
            np.mean(transfer_saturation_values)
        ),
        "ResponsibilityReconstructionRMS": float(
            np.sqrt(np.mean(np.square(reconstruction)))
        ),
        "LowerContributionOutOfUnitRate": float(np.mean(np.abs(lower) > 1.0)),
        "LowerLFRmsOnlineMean": float(np.mean(lower_lf_rms_values)),
        "LowerLFPowerOnlineMean": float(np.mean(np.square(
            lower_lf_rms_values
        ))),
        "LowerLFBudgetExcessMean": float(np.mean(lower_lf_budget_excesses)),
        "LowerLFBudgetViolationRate": float(np.mean(
            np.asarray(lower_lf_budget_excesses, dtype=np.float64) > 0.0
        )),
        "RawLowerLFRmsOnlineMean": float(np.mean(raw_lower_lf_rms_values)),
        "RawLowerLFPowerOnlineMean": float(np.mean(np.square(
            raw_lower_lf_rms_values
        ))),
        "RawLowerLFBudgetExcessMean": float(np.mean(
            raw_lower_lf_budget_excesses
        )),
        "RawLowerLFBudgetViolationRate": float(np.mean(
            np.asarray(raw_lower_lf_budget_excesses, dtype=np.float64) > 0.0
        )),
        "LatentLowerLFRmsOnlineMean": float(np.mean(
            latent_lower_lf_rms_values
        )),
        "LatentLowerLFPowerOnlineMean": float(np.mean(np.square(
            latent_lower_lf_rms_values
        ))),
        "LatentLowerLFBudgetExcessMean": float(np.mean(
            latent_lower_lf_budget_excesses
        )),
        "LatentLowerLFBudgetViolationRate": float(np.mean(
            np.asarray(latent_lower_lf_budget_excesses, dtype=np.float64) > 0.0
        )),
        "LowerConstraintCostMean": float(np.mean(lower_constraint_costs)),
        "LowerConstraintCostMax": float(np.max(lower_constraint_costs)),
        "LowerLFRmsBudget": float(lower_lf_rms_budget),
        "LeakageConstraintScope": str(leakage_constraint_scope),
        "LeakageConstraintCostMode": str(leakage_cost_mode),
        "LowerActionRouterMode": str(lower_action_router_mode),
        "LowerActionRouterStrength": float(lower_action_router_strength),
        "UpperTransitionDeltaRMSMean": float(np.mean(
            upper_transition_delta_rms_values
        )),
        "UpperTransitionDeltaRMSMax": float(np.max(
            upper_transition_delta_rms_values
        )),
        "UpperHFRmsOnlineMean": float(np.mean(upper_hf_rms_values)),
        "UpperHFPowerOnlineMean": float(np.mean(np.square(
            upper_hf_rms_values
        ))),
        "UpperHFBudgetExcessMean": float(np.mean(
            upper_hf_budget_excesses
        )),
        "UpperHFBudgetViolationRate": float(np.mean(
            np.asarray(upper_hf_budget_excesses, dtype=np.float64) > 0.0
        )),
        "UpperHFPenaltyMean": float(np.mean(
            upper_hf_penalties
        )),
        "UpperHFPenaltyTotal": float(np.sum(
            upper_hf_penalties
        )),
        "UpperConstraintCostMean": float(np.mean(
            upper_constraint_costs
        )),
        "UpperConstraintCostMax": float(np.max(
            upper_constraint_costs
        )),
        "UpperConstraintMode": str(upper_constraint_mode),
        "UpperHFRMSBudget": float(upper_hf_rms_budget),
        "UpperHFPenaltyCoef": float(upper_hf_penalty_coef),
        "upper_decision_count": int(upper_decisions),
        "upper_transition_count": int(upper_transitions),
        "lower_transition_count": int(lower_transitions),
        "protocol_valid": float(
            lower_transitions == len(rewards)
            and upper_transitions == upper_decisions
            and (
                method == "flat_ppo"
                or 0 < upper_transitions < lower_transitions
            )
            and np.all(np.isfinite(reconstruction))
            and np.all(np.isfinite(router_reconstruction))
            and (
                str(lower_action_router_mode)
                not in FUNCTION_PRESERVING_LOWER_ACTION_ROUTER_MODES
                or (
                    float(np.max(np.abs(router_reconstruction))) <= 1e-7
                    and float(np.max(np.abs(reconstruction))) <= 1e-7
                )
            )
        ),
    }


def rollout_hierarchical(
    model: FrequencySeparatedActorCriticPPO,
    *,
    seed: int,
    env_id: str,
    disturbance_mode: str,
    steps: int,
    upper_period: int,
    frequency_routing: bool,
    leakage_constraint: bool,
    sample: bool,
    collect_trajectory: bool = False,
    upper_action_scale: float = 1.0,
    lower_action_scale: float = 1.0,
    upper_action_decoder_mode: str = "hold",
    lower_lf_alpha: float = 0.04,
    lower_lf_rms_budget: float = 0.05,
    leakage_constraint_scope: str = "responsibility",
    upper_hf_rms_budget: float = DEFAULT_UPPER_HF_RMS_BUDGET,
    upper_hf_penalty_coef: float = 0.0,
    upper_constraint_mode: str = "static_reward_penalty",
    responsibility_mode: str = "additive",
    leakage_cost_mode: str = "ratio_excess_squared",
    lower_action_router_mode: str = "direct",
    lower_action_router_alpha: float = DEFAULT_LOWER_ROUTER_ALPHA,
    lower_action_router_strength: float = DEFAULT_LOWER_ROUTER_STRENGTH,
    lower_action_router_observe_strength: bool = False,
    upper_promotion_gain: float = 0.0,
    method: str = "freq_hrl",
    episode_horizon: int = 1000,
) -> tuple[HierarchicalTrajectoryBatch | None, dict[str, Any]]:
    if str(leakage_constraint_scope) not in LEAKAGE_CONSTRAINT_SCOPES:
        raise ValueError("unknown MuJoCo leakage constraint scope")
    if str(upper_constraint_mode) not in UPPER_CONSTRAINT_MODES:
        raise ValueError("unknown MuJoCo upper constraint mode")
    if str(leakage_cost_mode) not in LEAKAGE_COST_MODES:
        raise ValueError("unknown MuJoCo leakage constraint cost mode")
    if str(lower_action_router_mode) not in LOWER_ACTION_ROUTER_MODES:
        raise ValueError("unknown MuJoCo lower-action router mode")
    if str(upper_action_decoder_mode) not in UPPER_ACTION_DECODER_MODES:
        raise ValueError("unknown MuJoCo upper-action decoder mode")
    if not 0.0 < float(lower_action_router_alpha) <= 1.0:
        raise ValueError("MuJoCo lower-action router alpha must be in (0, 1]")
    if not 0.0 <= float(lower_action_router_strength) <= 1.0:
        raise ValueError(
            "MuJoCo lower-action router strength must be in [0, 1]"
        )
    if (
        not np.isfinite(float(upper_promotion_gain))
        or not 0.0 <= float(upper_promotion_gain) <= 1.0
    ):
        raise ValueError("MuJoCo upper promotion gain must be in [0, 1]")
    headroom_zero_dc_router = (
        str(lower_action_router_mode) == "causal_macro_zero_dc_headroom"
    )
    smooth_macro_gauge_router = (
        str(lower_action_router_mode) == "causal_smooth_macro_gauge"
    )
    audit_optimal_macro_gauge_router = (
        str(lower_action_router_mode)
        == "causal_audit_optimal_macro_gauge"
    )
    if headroom_zero_dc_router and (
        str(upper_action_decoder_mode) != "causal_smoothstep_plan"
    ):
        raise ValueError(
            "headroom zero-DC routing requires a frozen smooth upper plan"
        )
    if headroom_zero_dc_router and str(responsibility_mode) != "additive":
        raise ValueError(
            "headroom zero-DC routing requires additive responsibility"
        )
    if smooth_macro_gauge_router and (
        str(upper_action_decoder_mode) != "causal_smoothstep_plan"
    ):
        raise ValueError(
            "smooth macro gauge routing requires a frozen smooth upper plan"
        )
    if smooth_macro_gauge_router and str(responsibility_mode) != "additive":
        raise ValueError(
            "smooth macro gauge routing requires additive responsibility"
        )
    if audit_optimal_macro_gauge_router and (
        str(upper_action_decoder_mode) != "causal_smoothstep_plan"
    ):
        raise ValueError(
            "audit-optimal macro gauge routing requires a frozen smooth upper plan"
        )
    if (
        audit_optimal_macro_gauge_router
        and str(responsibility_mode) != "additive"
    ):
        raise ValueError(
            "audit-optimal macro gauge routing requires additive responsibility"
        )
    if float(upper_promotion_gain) > 0.0 and not headroom_zero_dc_router:
        raise ValueError(
            "upper promotion requires headroom zero-DC lower routing"
        )
    effective_upper_promotion_gain = (
        float(upper_promotion_gain) * float(lower_action_router_strength)
    )
    router_strength_context = (
        float(lower_action_router_strength)
        if bool(lower_action_router_observe_strength)
        else None
    )
    if (
        str(upper_constraint_mode) == "primal_dual"
        and model.upper_cost_value is None
    ):
        raise ValueError("primal-dual upper control requires an upper cost critic")
    if (
        not np.isfinite(float(upper_hf_penalty_coef))
        or float(upper_hf_penalty_coef) < 0.0
    ):
        raise ValueError("upper-HF penalty coefficient must be non-negative")
    if (
        not np.isfinite(float(upper_hf_rms_budget))
        or float(upper_hf_rms_budget) <= 0.0
    ):
        raise ValueError("upper-HF RMS budget must be positive")
    transition_budget = int(steps) if sample else int(episode_horizon)
    env = _make_env(env_id, episode_horizon=episode_horizon)
    try:
        observation, _ = env.reset(seed=int(seed))
        model.reset_recurrent_inference()
        decomposer = CausalBandDecomposer()
        action_dim = int(env.action_space.shape[0])
        previous_action = np.zeros(action_dim, dtype=np.float32)
        upper_anchor = np.zeros(action_dim, dtype=np.float32)
        upper_policy_action = np.zeros(action_dim, dtype=np.float32)
        upper_promotion_action = np.zeros(action_dim, dtype=np.float32)
        upper_promotion_context = np.zeros(action_dim, dtype=np.float32)
        latent_responsibility_lf = np.zeros(action_dim, dtype=np.float64)
        latent_lower_context = np.zeros(action_dim, dtype=np.float32)
        previous_upper_anchor = np.zeros(action_dim, dtype=np.float32)
        has_previous_upper_anchor = False
        current_requested_transfer = np.zeros(action_dim, dtype=np.float32)
        responsibility = CausalResponsibilityTransfer(
            mode=str(responsibility_mode), alpha=float(lower_lf_alpha)
        )
        responsibility.reset(action_dim)
        lower_router = CausalLowerActionRouter(
            mode=str(lower_action_router_mode),
            alpha=float(lower_action_router_alpha),
            strength=float(lower_action_router_strength),
            upper_rms_budget=float(upper_hf_rms_budget),
            lower_rms_budget=float(lower_lf_rms_budget),
            macro_steps=int(upper_period),
        )
        lower_router.reset(action_dim)
        upper_plan = CausalSmoothstepMacroPlan(macro_steps=int(upper_period))
        upper_plan.reset(action_dim)
        responsibility_lf_tracker = CausalRollingBandTracker(window=32)
        responsibility_lf_tracker.reset(action_dim)
        raw_lower_lf_tracker = CausalRollingBandTracker(window=32)
        raw_lower_lf_tracker.reset(action_dim)
        latent_lower_lf_tracker = CausalRollingBandTracker(window=32)
        latent_lower_lf_tracker.reset(action_dim)
        upper_hf_tracker = CausalRollingBandTracker(window=8)
        upper_hf_tracker.reset(action_dim)
        latent_upper_hf_tracker = CausalRollingBandTracker(window=8)
        latent_upper_hf_tracker.reset(action_dim)
        function_preserving_router = bool(
            str(lower_action_router_mode)
            in FUNCTION_PRESERVING_LOWER_ACTION_ROUTER_MODES
        )

        def actor_filter_contexts(
            level: str,
        ) -> tuple[np.ndarray, ...]:
            if audit_optimal_macro_gauge_router:
                action_blocks, _ = lower_router.policy_context
                return action_blocks
            if headroom_zero_dc_router:
                strength = float(lower_action_router_strength)
                baseline = (
                    latent_responsibility_lf,
                    latent_lower_context,
                )
                if str(level) == "upper":
                    target = (
                        lower_router.promotion_context,
                        responsibility.raw_lower_lf,
                    )
                else:
                    target = (
                        responsibility.raw_lower_lf,
                        lower_router.context,
                    )
                if strength == 0.0:
                    return baseline
                if strength == 1.0:
                    return target
                return tuple(
                    (1.0 - strength) * left + strength * right
                    for left, right in zip(baseline, target)
                )
            if function_preserving_router:
                context = lower_router.context
                return context, context
            return responsibility.raw_lower_lf, lower_router.context

        def cost_filter_contexts() -> tuple[np.ndarray, ...]:
            if audit_optimal_macro_gauge_router:
                action_blocks, _ = lower_router.policy_context
                return action_blocks
            if function_preserving_router:
                context = latent_lower_lf_tracker.low
                return context, context
            return raw_lower_lf_tracker.low, responsibility_lf_tracker.low

        def router_scalar_contexts() -> tuple[float, ...]:
            if audit_optimal_macro_gauge_router:
                _, scalars = lower_router.policy_context
                return scalars
            return ()
        builder = HierarchicalRolloutBuilder(gamma=float(model.config.gamma))
        rewards: list[float] = []
        executed_actions: list[np.ndarray] = []
        upper_actions: list[np.ndarray] = []
        lower_actions: list[np.ndarray] = []
        upper_policy_actions: list[np.ndarray] = []
        upper_promotion_actions: list[np.ndarray] = []
        upper_promotion_contexts: list[np.ndarray] = []
        raw_lower_actions: list[np.ndarray] = []
        latent_lower_actions: list[np.ndarray] = []
        lower_router_removed_actions: list[np.ndarray] = []
        lower_router_reconstruction_errors: list[np.ndarray] = []
        lower_router_clip_values: list[float] = []
        lower_router_audit_alphas: list[float] = []
        lower_router_audit_imbalances: list[float] = []
        lower_router_macro_projection_rates: list[float] = []
        lower_router_macro_debt_rms_values: list[float] = []
        lower_router_macro_completion_errors: list[float] = []
        lower_router_headroom_clip_values: list[float] = []
        responsibility_transfers: list[np.ndarray] = []
        requested_transfers: list[np.ndarray] = []
        transfer_saturation_values: list[float] = []
        reconstruction_errors: list[np.ndarray] = []
        forward_rewards: list[float] = []
        control_rewards: list[float] = []
        upper_decisions = 0
        segment_returns: list[float] = []
        natural_episode_returns: list[float] = []
        boundary_upper_next_values: list[float] = []
        boundary_upper_next_cost_values: list[float] = []
        boundary_lower_next_values: list[float] = []
        boundary_lower_next_cost_values: list[float] = []
        boundary_terminals: list[float] = []
        lower_lf_rms_values: list[float] = []
        lower_lf_budget_excesses: list[float] = []
        raw_lower_lf_rms_values: list[float] = []
        raw_lower_lf_budget_excesses: list[float] = []
        latent_lower_lf_rms_values: list[float] = []
        latent_lower_lf_budget_excesses: list[float] = []
        lower_constraint_costs: list[float] = []
        upper_transition_delta_rms_values: list[float] = []
        upper_hf_rms_values: list[float] = []
        upper_hf_budget_excesses: list[float] = []
        upper_hf_penalties: list[float] = []
        upper_constraint_costs: list[float] = []
        upper_cost_values: list[float] = []
        lower_cost_values: list[float] = []
        current_episode_return = 0.0
        episode_index = 0
        episode_seed = int(seed)
        episode_step = 0
        reset_exogenous = True
        steps_since_upper = int(upper_period)
        require_upper = True

        for step in range(transition_budget):
            upper_decision_now = False
            disturbance = deterministic_actuation_disturbance(
                mode=disturbance_mode,
                step=episode_step,
                action_dim=action_dim,
                seed=episode_seed,
                horizon=int(episode_horizon),
            )
            bands = (
                decomposer.reset(disturbance)
                if reset_exogenous
                else decomposer.update(disturbance)
            )
            reset_exogenous = False
            if require_upper or steps_since_upper >= int(upper_period):
                upper_decision_now = True
                upper_promotion_context = lower_router.promotion_context
                upper_state = _feature_state(
                    observation,
                    bands,
                    previous_action,
                    filter_contexts=actor_filter_contexts("upper"),
                    router_scalar_contexts=router_scalar_contexts(),
                    router_strength_context=router_strength_context,
                    frequency_routing=frequency_routing,
                    level="upper",
                )
                upper_out = model.act_upper(upper_state, sample=sample)
                upper_cost_values.append(float(upper_out["cost_value"]))
                upper_raw = np.asarray(upper_out["action"], dtype=np.float32)
                upper_policy_action = (
                    float(upper_action_scale) * np.tanh(upper_raw)
                )
                promoted_upper_policy_action = np.clip(
                    upper_policy_action
                    + effective_upper_promotion_gain
                    * upper_promotion_context,
                    -float(upper_action_scale),
                    float(upper_action_scale),
                ).astype(np.float32, copy=False)
                upper_promotion_action = np.asarray(
                    promoted_upper_policy_action - upper_policy_action,
                    dtype=np.float32,
                )
                upper_assignment = responsibility.begin_macro(
                    promoted_upper_policy_action
                )
                upper_target = np.asarray(
                    upper_assignment["upper_responsibility"],
                    dtype=np.float32,
                )
                upper_anchor = (
                    upper_plan.activate(upper_target)
                    if str(upper_action_decoder_mode)
                    == "causal_smoothstep_plan"
                    else upper_target.copy()
                )
                upper_transition_power = (
                    float(np.mean(np.square(
                        np.asarray(upper_target, dtype=np.float64)
                        - np.asarray(previous_upper_anchor, dtype=np.float64)
                    )))
                    if has_previous_upper_anchor else 0.0
                )
                upper_transition_delta_rms_values.append(float(
                    np.sqrt(upper_transition_power)
                ))
                previous_upper_anchor = upper_target.copy()
                has_previous_upper_anchor = True
                current_requested_transfer = np.asarray(
                    upper_assignment["requested_transfer"],
                    dtype=np.float32,
                )
                transfer_saturation_values.append(float(
                    upper_assignment["headroom_saturation_rate"]
                ))
                builder.begin_upper(
                    state=upper_state,
                    action=upper_raw,
                    logp=float(upper_out["logp"]),
                    value=float(upper_out["value"]),
                )
                upper_decisions += 1
                steps_since_upper = 0
                require_upper = False
            elif str(upper_action_decoder_mode) == "causal_smoothstep_plan":
                upper_anchor = upper_plan.advance()

            lower_state = _feature_state(
                observation,
                bands,
                (
                    upper_anchor
                    if str(upper_action_decoder_mode)
                    == "causal_smoothstep_plan"
                    else upper_policy_action
                ),
                filter_contexts=actor_filter_contexts("lower"),
                router_scalar_contexts=router_scalar_contexts(),
                router_strength_context=router_strength_context,
                frequency_routing=frequency_routing,
                level="lower",
            )
            lower_cost_state = _feature_state(
                observation,
                bands,
                upper_anchor,
                filter_contexts=cost_filter_contexts(),
                router_scalar_contexts=router_scalar_contexts(),
                router_strength_context=router_strength_context,
                frequency_routing=frequency_routing,
                level="lower",
            )
            lower_out = model.act_lower(
                lower_state,
                sample=sample,
                cost_state=lower_cost_state,
            )
            lower_raw = np.asarray(lower_out["action"], dtype=np.float32)
            lower_cost_values.append(float(lower_out["cost_value"]))
            latent_lower_residual = (
                float(lower_action_scale) * np.tanh(lower_raw)
            )
            routed_lower = lower_router.route(
                latent_lower_residual,
                upper_action=(
                    upper_anchor
                    if str(lower_action_router_mode)
                    in {
                        "causal_joint_band_projection",
                        "causal_total_action_gauge",
                        "causal_audit_aligned_gauge",
                        "causal_macro_hold_audit_gauge",
                        "causal_smooth_macro_gauge",
                        "causal_audit_optimal_macro_gauge",
                        "causal_macro_zero_dc_headroom",
                    }
                    else None
                ),
                future_upper_actions=(
                    upper_plan.future_values()
                    if headroom_zero_dc_router else None
                ),
                action_limit=float(lower_action_scale),
                upper_action_limit=float(upper_action_scale),
                macro_boundary=upper_decision_now,
            )
            raw_lower_residual = np.asarray(
                routed_lower["effective"], dtype=np.float32
            )
            router_upper_transfer = np.asarray(
                routed_lower["upper_transfer"], dtype=np.float32
            )
            responsibility_split = responsibility.split_lower(
                raw_lower_residual
            )
            latent_responsibility_lf += float(lower_lf_alpha) * (
                latent_lower_residual - latent_responsibility_lf
            )
            latent_lower_context = latent_lower_residual.astype(
                np.float32, copy=True
            )
            lower_residual = np.asarray(
                responsibility_split["lower_responsibility"],
                dtype=np.float32,
            )
            effective_upper_action = np.asarray(
                upper_anchor + router_upper_transfer,
                dtype=np.float32,
            )
            canonical_lower_action = (
                latent_lower_residual
                if str(lower_action_router_mode)
                in FUNCTION_PRESERVING_LOWER_ACTION_ROUTER_MODES
                else raw_lower_residual
            )
            # Use the pre-split canonical sum once. In conservative mode this
            # is exactly the direct-policy action, while the reported upper and
            # lower responsibilities reconstruct it independently.
            execution_upper_action = (
                upper_anchor
                if str(upper_action_decoder_mode) == "causal_smoothstep_plan"
                else upper_policy_action
            )
            nominal = np.clip(
                execution_upper_action + canonical_lower_action, -1.0, 1.0
            )
            executed = np.clip(nominal + disturbance, -1.0, 1.0)
            env_action = action_from_unit_box(
                executed, env.action_space.low, env.action_space.high
            )
            next_observation, reward, terminated, truncated, info = env.step(
                env_action
            )
            natural_done = bool(terminated or truncated)
            budget_done = step == transition_budget - 1
            done = bool(natural_done or budget_done)
            responsibility_bands = responsibility_lf_tracker.update(
                lower_residual
            )
            raw_lower_bands = raw_lower_lf_tracker.update(raw_lower_residual)
            latent_lower_bands = latent_lower_lf_tracker.update(
                latent_lower_residual
            )
            lower_budget = evaluate_rms_leakage_budget(
                float(np.mean(np.square(responsibility_bands["low"]))),
                float(lower_lf_rms_budget),
            )
            raw_lower_budget = evaluate_rms_leakage_budget(
                float(np.mean(np.square(raw_lower_bands["low"]))),
                float(lower_lf_rms_budget),
            )
            latent_lower_budget = evaluate_rms_leakage_budget(
                float(np.mean(np.square(latent_lower_bands["low"]))),
                float(lower_lf_rms_budget),
            )
            upper_bands = upper_hf_tracker.update(effective_upper_action)
            upper_budget = evaluate_rms_leakage_budget(
                float(np.mean(np.square(upper_bands["high"]))),
                float(upper_hf_rms_budget),
            )
            latent_upper_bands = latent_upper_hf_tracker.update(
                upper_policy_action
            )
            latent_upper_budget = evaluate_rms_leakage_budget(
                float(np.mean(np.square(latent_upper_bands["high"]))),
                float(upper_hf_rms_budget),
            )
            upper_effective_cost = _leakage_constraint_cost(
                upper_budget, mode=str(leakage_cost_mode)
            )
            upper_latent_cost = _leakage_constraint_cost(
                latent_upper_budget, mode=str(leakage_cost_mode)
            )
            upper_endpoint_cost = (
                max(upper_effective_cost, upper_latent_cost)
                if str(leakage_constraint_scope)
                == "joint_behavior_latent"
                else upper_effective_cost
            )
            upper_hf_penalty = (
                float(upper_hf_penalty_coef)
                * upper_endpoint_cost
                if str(upper_constraint_mode) == "static_reward_penalty"
                else 0.0
            )
            upper_constraint_cost = (
                upper_endpoint_cost
                if str(upper_constraint_mode) == "primal_dual"
                else 0.0
            )
            lower_lf_rms_values.append(float(lower_budget["rms"]))
            lower_lf_budget_excesses.append(float(lower_budget["budget_excess"]))
            raw_lower_lf_rms_values.append(float(raw_lower_budget["rms"]))
            raw_lower_lf_budget_excesses.append(float(
                raw_lower_budget["budget_excess"]
            ))
            latent_lower_lf_rms_values.append(float(
                latent_lower_budget["rms"]
            ))
            latent_lower_lf_budget_excesses.append(float(
                latent_lower_budget["budget_excess"]
            ))
            responsibility_cost = _leakage_constraint_cost(
                lower_budget, mode=str(leakage_cost_mode)
            )
            raw_behavior_cost = _leakage_constraint_cost(
                raw_lower_budget, mode=str(leakage_cost_mode)
            )
            lower_cost = 0.0
            if leakage_constraint:
                if str(leakage_constraint_scope) == "responsibility":
                    lower_cost = responsibility_cost
                elif str(leakage_constraint_scope) == "joint_behavior":
                    lower_cost = max(
                        responsibility_cost, raw_behavior_cost
                    )
                else:
                    latent_behavior_cost = _leakage_constraint_cost(
                        latent_lower_budget, mode=str(leakage_cost_mode)
                    )
                    lower_cost = max(
                        responsibility_cost,
                        raw_behavior_cost,
                        latent_behavior_cost,
                    )
            lower_constraint_costs.append(float(lower_cost))
            upper_hf_rms_values.append(float(upper_budget["rms"]))
            upper_hf_budget_excesses.append(float(
                upper_budget["budget_excess"]
            ))
            upper_hf_penalties.append(float(upper_hf_penalty))
            upper_constraint_costs.append(float(upper_constraint_cost))
            builder.add_lower(
                state=lower_state,
                action=lower_raw,
                logp=float(lower_out["logp"]),
                value=float(lower_out["value"]),
                reward=float(reward),
                upper_reward=float(reward) - upper_hf_penalty,
                upper_cost=upper_constraint_cost,
                cost_state=lower_cost_state,
                cost=float(lower_cost),
                done=done,
            )
            rewards.append(float(reward))
            current_episode_return += float(reward)
            executed_actions.append(executed.copy())
            upper_actions.append(effective_upper_action.copy())
            lower_actions.append(lower_residual.copy())
            upper_policy_actions.append(upper_policy_action.copy())
            upper_promotion_actions.append(upper_promotion_action.copy())
            upper_promotion_contexts.append(upper_promotion_context.copy())
            raw_lower_actions.append(raw_lower_residual.copy())
            latent_lower_actions.append(latent_lower_residual.copy())
            lower_router_removed_actions.append(np.asarray(
                routed_lower["removed_low_frequency"], dtype=np.float32
            ))
            lower_router_reconstruction_errors.append(np.asarray(
                routed_lower["transfer_reconstruction_error"],
                dtype=np.float64,
            ))
            lower_router_clip_values.append(float(routed_lower["clip_rate"]))
            lower_router_audit_alphas.append(float(
                routed_lower["audit_alpha_after"]
            ))
            lower_router_audit_imbalances.append(float(
                routed_lower["audit_normalized_band_imbalance"]
            ))
            lower_router_macro_projection_rates.append(float(
                routed_lower["macro_projection_rate"]
            ))
            lower_router_macro_debt_rms_values.append(float(
                routed_lower["macro_debt_rms"]
            ))
            lower_router_macro_completion_errors.append(float(
                routed_lower["macro_completion_error_rms"]
            ))
            lower_router_headroom_clip_values.append(float(
                routed_lower["headroom_clip_rate"]
            ))
            responsibility_transfers.append(np.asarray(
                responsibility.effective_transfer + router_upper_transfer,
                dtype=np.float32,
            ))
            requested_transfers.append(np.asarray(
                current_requested_transfer + router_upper_transfer,
                dtype=np.float32,
            ))
            reconstruction_errors.append(
                np.asarray(effective_upper_action, dtype=np.float64)
                + np.asarray(lower_residual, dtype=np.float64)
                - np.asarray(execution_upper_action, dtype=np.float64)
                - np.asarray(canonical_lower_action, dtype=np.float64)
            )
            forward_rewards.append(float(info.get("reward_forward", 0.0)))
            control_rewards.append(float(info.get("reward_ctrl", 0.0)))
            previous_action = executed.astype(np.float32, copy=True)
            steps_since_upper += 1
            if done:
                terminal = float(bool(terminated))
                upper_next_value = 0.0
                upper_next_cost_value = 0.0
                lower_next_value = 0.0
                lower_next_cost_value = 0.0
                if sample and not bool(terminated):
                    next_disturbance = deterministic_actuation_disturbance(
                        mode=disturbance_mode,
                        step=episode_step + 1,
                        action_dim=action_dim,
                        seed=episode_seed,
                        horizon=int(episode_horizon),
                    )
                    next_bands = decomposer.update(next_disturbance)
                    next_upper_state = _feature_state(
                        next_observation,
                        next_bands,
                        executed,
                        filter_contexts=actor_filter_contexts("upper"),
                        router_scalar_contexts=router_scalar_contexts(),
                        router_strength_context=router_strength_context,
                        frequency_routing=frequency_routing,
                        level="upper",
                    )
                    next_upper = model.act_upper(
                        next_upper_state,
                        sample=False,
                    )
                    upper_next_value = float(next_upper["value"])
                    upper_next_cost_value = float(next_upper["cost_value"])
                    next_upper_policy_action = upper_policy_action
                    next_upper_anchor = upper_anchor
                    if steps_since_upper >= int(upper_period):
                        next_upper_policy = float(upper_action_scale) * np.tanh(
                            np.asarray(next_upper["action"], dtype=np.float32)
                        )
                        next_upper_policy_action = next_upper_policy
                        next_promoted_upper_policy = np.clip(
                            next_upper_policy
                            + effective_upper_promotion_gain
                            * lower_router.promotion_context,
                            -float(upper_action_scale),
                            float(upper_action_scale),
                        )
                        next_upper_anchor = np.asarray(
                            responsibility.preview_upper(
                                next_promoted_upper_policy
                            )["upper_responsibility"],
                            dtype=np.float32,
                        )
                    next_execution_upper = (
                        (
                            upper_plan.target
                            if steps_since_upper >= int(upper_period)
                            else upper_plan.peek_advance()
                        )
                        if str(upper_action_decoder_mode)
                        == "causal_smoothstep_plan"
                        else next_upper_anchor
                    )
                    next_lower_state = _feature_state(
                        next_observation,
                        next_bands,
                        (
                            next_execution_upper
                            if str(upper_action_decoder_mode)
                            == "causal_smoothstep_plan"
                            else next_upper_policy_action
                        ),
                        filter_contexts=actor_filter_contexts("lower"),
                        router_scalar_contexts=router_scalar_contexts(),
                        router_strength_context=router_strength_context,
                        frequency_routing=frequency_routing,
                        level="lower",
                    )
                    lower_next_value = _value_prediction(
                        model.lower_value,
                        next_lower_state,
                        device=model.device,
                    )
                    next_lower_cost_state = _feature_state(
                        next_observation,
                        next_bands,
                        next_execution_upper,
                        filter_contexts=cost_filter_contexts(),
                        router_scalar_contexts=router_scalar_contexts(),
                        router_strength_context=router_strength_context,
                        frequency_routing=frequency_routing,
                        level="lower",
                    )
                    lower_next_cost_value = _value_prediction(
                        model.lower_cost_value,
                        next_lower_cost_state,
                        device=model.device,
                    )
                boundary_upper_next_values.append(upper_next_value)
                boundary_upper_next_cost_values.append(
                    upper_next_cost_value
                )
                boundary_lower_next_values.append(lower_next_value)
                boundary_lower_next_cost_values.append(
                    lower_next_cost_value
                )
                boundary_terminals.append(terminal)
                segment_returns.append(current_episode_return)
                if natural_done:
                    natural_episode_returns.append(current_episode_return)
                current_episode_return = 0.0
                if not sample or budget_done:
                    break
                episode_index += 1
                episode_seed = derive_seed(
                    "freq_hrl_mujoco_episode_reset_v1",
                    int(seed),
                    int(episode_index),
                )
                observation, _ = env.reset(seed=episode_seed)
                model.reset_recurrent_inference()
                previous_action = np.zeros(action_dim, dtype=np.float32)
                upper_anchor = np.zeros(action_dim, dtype=np.float32)
                upper_policy_action = np.zeros(action_dim, dtype=np.float32)
                upper_promotion_action = np.zeros(
                    action_dim, dtype=np.float32
                )
                upper_promotion_context = np.zeros(
                    action_dim, dtype=np.float32
                )
                latent_responsibility_lf = np.zeros(
                    action_dim, dtype=np.float64
                )
                latent_lower_context = np.zeros(
                    action_dim, dtype=np.float32
                )
                previous_upper_anchor = np.zeros(
                    action_dim, dtype=np.float32
                )
                has_previous_upper_anchor = False
                current_requested_transfer = np.zeros(
                    action_dim, dtype=np.float32
                )
                responsibility.reset(action_dim)
                lower_router.reset(action_dim)
                upper_plan.reset(action_dim)
                responsibility_lf_tracker.reset(action_dim)
                raw_lower_lf_tracker.reset(action_dim)
                latent_lower_lf_tracker.reset(action_dim)
                upper_hf_tracker.reset(action_dim)
                latent_upper_hf_tracker.reset(action_dim)
                episode_step = 0
                reset_exogenous = True
                steps_since_upper = int(upper_period)
                require_upper = True
            else:
                observation = next_observation
                episode_step += 1

        builder.finish(terminal=True)
        trajectory = builder.build()
        if sample:
            trajectory = replace(
                trajectory,
                upper=_with_explicit_bootstrap(
                    trajectory.upper,
                    boundary_next_values=boundary_upper_next_values,
                    boundary_terminals=boundary_terminals,
                    boundary_next_cost_values=(
                        boundary_upper_next_cost_values
                        if str(upper_constraint_mode) == "primal_dual"
                        else None
                    ),
                    cost_values=(
                        upper_cost_values
                        if str(upper_constraint_mode) == "primal_dual"
                        else None
                    ),
                ),
                lower=_with_explicit_bootstrap(
                    trajectory.lower,
                    boundary_next_values=boundary_lower_next_values,
                    boundary_terminals=boundary_terminals,
                    boundary_next_cost_values=(
                        boundary_lower_next_cost_values
                        if leakage_constraint else None
                    ),
                    cost_values=(lower_cost_values if leakage_constraint else None),
                ),
            )
        row = _episode_row(
            seed=seed,
            env_id=env_id,
            disturbance_mode=disturbance_mode,
            rewards=rewards,
            executed_actions=executed_actions,
            upper_actions=upper_actions,
            lower_actions=lower_actions,
            upper_policy_actions=upper_policy_actions,
            upper_promotion_actions=upper_promotion_actions,
            upper_promotion_contexts=upper_promotion_contexts,
            raw_lower_actions=raw_lower_actions,
            latent_lower_actions=latent_lower_actions,
            lower_router_removed_actions=lower_router_removed_actions,
            lower_router_reconstruction_errors=(
                lower_router_reconstruction_errors
            ),
            lower_router_clip_values=lower_router_clip_values,
            lower_router_audit_alphas=lower_router_audit_alphas,
            lower_router_audit_imbalances=lower_router_audit_imbalances,
            lower_router_macro_projection_rates=(
                lower_router_macro_projection_rates
            ),
            lower_router_macro_debt_rms_values=(
                lower_router_macro_debt_rms_values
            ),
            lower_router_macro_completion_errors=(
                lower_router_macro_completion_errors
            ),
            lower_router_headroom_clip_values=(
                lower_router_headroom_clip_values
            ),
            responsibility_transfers=responsibility_transfers,
            requested_transfers=requested_transfers,
            transfer_saturation_values=transfer_saturation_values,
            reconstruction_errors=reconstruction_errors,
            forward_rewards=forward_rewards,
            control_rewards=control_rewards,
            upper_decisions=upper_decisions,
            upper_transitions=trajectory.upper.size,
            lower_transitions=trajectory.lower.size,
            method=method,
            segment_returns=segment_returns,
            natural_episode_returns=natural_episode_returns,
            boundary_terminals=boundary_terminals,
            transition_budget=transition_budget,
            lower_lf_rms_values=lower_lf_rms_values,
            lower_lf_budget_excesses=lower_lf_budget_excesses,
            raw_lower_lf_rms_values=raw_lower_lf_rms_values,
            raw_lower_lf_budget_excesses=raw_lower_lf_budget_excesses,
            latent_lower_lf_rms_values=latent_lower_lf_rms_values,
            latent_lower_lf_budget_excesses=(
                latent_lower_lf_budget_excesses
            ),
            lower_constraint_costs=lower_constraint_costs,
            lower_lf_rms_budget=lower_lf_rms_budget,
            leakage_constraint_scope=(
                str(leakage_constraint_scope)
                if leakage_constraint else "disabled"
            ),
            upper_transition_delta_rms_values=(
                upper_transition_delta_rms_values
            ),
            upper_hf_rms_values=upper_hf_rms_values,
            upper_hf_budget_excesses=upper_hf_budget_excesses,
            upper_hf_penalties=upper_hf_penalties,
            upper_constraint_costs=upper_constraint_costs,
            upper_constraint_mode=str(upper_constraint_mode),
            upper_hf_rms_budget=upper_hf_rms_budget,
            upper_hf_penalty_coef=upper_hf_penalty_coef,
            leakage_cost_mode=str(leakage_cost_mode),
            lower_action_router_mode=str(lower_action_router_mode),
            lower_action_router_strength=float(
                lower_action_router_strength
            ),
            upper_promotion_gain=effective_upper_promotion_gain,
        )
        return (
            trajectory if sample or collect_trajectory else None
        ), row
    finally:
        env.close()


def rollout_flat(
    model: JointActorCriticPPO,
    *,
    seed: int,
    env_id: str,
    disturbance_mode: str,
    steps: int,
    sample: bool,
    episode_horizon: int = 1000,
    lower_lf_rms_budget: float = 0.05,
) -> tuple[JointTrajectoryBatch | None, dict[str, Any]]:
    transition_budget = int(steps) if sample else int(episode_horizon)
    env = _make_env(env_id, episode_horizon=episode_horizon)
    try:
        observation, _ = env.reset(seed=int(seed))
        model.reset_recurrent_inference()
        decomposer = CausalBandDecomposer()
        action_dim = int(env.action_space.shape[0])
        previous_action = np.zeros(action_dim, dtype=np.float32)
        data: dict[str, list[Any]] = {
            key: []
            for key in ("state", "action", "reward", "done", "old_logp", "old_value")
        }
        rewards: list[float] = []
        executed_actions: list[np.ndarray] = []
        nominal_actions: list[np.ndarray] = []
        forward_rewards: list[float] = []
        control_rewards: list[float] = []
        segment_returns: list[float] = []
        natural_episode_returns: list[float] = []
        boundary_next_values: list[float] = []
        boundary_terminals: list[float] = []
        lower_lf = np.zeros(action_dim, dtype=np.float64)
        lower_lf_rms_values: list[float] = []
        lower_lf_budget_excesses: list[float] = []
        current_episode_return = 0.0
        episode_index = 0
        episode_seed = int(seed)
        episode_step = 0
        reset_exogenous = True

        for step in range(transition_budget):
            disturbance = deterministic_actuation_disturbance(
                mode=disturbance_mode,
                step=episode_step,
                action_dim=action_dim,
                seed=episode_seed,
                horizon=int(episode_horizon),
            )
            bands = (
                decomposer.reset(disturbance)
                if reset_exogenous
                else decomposer.update(disturbance)
            )
            reset_exogenous = False
            state = _feature_state(
                observation,
                bands,
                previous_action,
                filter_contexts=(lower_lf, lower_lf),
                frequency_routing=False,
                level="lower",
            )
            output = model.act(state, sample=sample)
            raw_action = np.asarray(output["action"], dtype=np.float32)
            nominal = np.tanh(raw_action)
            executed = np.clip(nominal + disturbance, -1.0, 1.0)
            env_action = action_from_unit_box(
                executed, env.action_space.low, env.action_space.high
            )
            next_observation, reward, terminated, truncated, info = env.step(
                env_action
            )
            natural_done = bool(terminated or truncated)
            budget_done = step == transition_budget - 1
            done = bool(natural_done or budget_done)
            data["state"].append(state)
            data["action"].append(raw_action)
            data["reward"].append(float(reward))
            data["done"].append(float(done))
            data["old_logp"].append(float(output["logp"]))
            data["old_value"].append(float(output["value"]))
            rewards.append(float(reward))
            current_episode_return += float(reward)
            executed_actions.append(executed.copy())
            nominal_actions.append(nominal.copy())
            forward_rewards.append(float(info.get("reward_forward", 0.0)))
            control_rewards.append(float(info.get("reward_ctrl", 0.0)))
            previous_action = executed.astype(np.float32, copy=True)
            lower_lf += 0.04 * (
                np.asarray(executed, dtype=np.float64) - lower_lf
            )
            lower_budget = evaluate_rms_leakage_budget(
                float(np.mean(np.square(lower_lf))),
                float(lower_lf_rms_budget),
            )
            lower_lf_rms_values.append(float(lower_budget["rms"]))
            lower_lf_budget_excesses.append(float(lower_budget["budget_excess"]))
            if done:
                terminal = float(bool(terminated))
                next_value = 0.0
                if sample and not bool(terminated):
                    next_disturbance = deterministic_actuation_disturbance(
                        mode=disturbance_mode,
                        step=episode_step + 1,
                        action_dim=action_dim,
                        seed=episode_seed,
                        horizon=int(episode_horizon),
                    )
                    next_bands = decomposer.update(next_disturbance)
                    next_state = _feature_state(
                        next_observation,
                        next_bands,
                        executed,
                        filter_contexts=(lower_lf, lower_lf),
                        frequency_routing=False,
                        level="lower",
                    )
                    next_value = _value_prediction(
                        model.value,
                        next_state,
                        device=model.device,
                    )
                boundary_next_values.append(next_value)
                boundary_terminals.append(terminal)
                segment_returns.append(current_episode_return)
                if natural_done:
                    natural_episode_returns.append(current_episode_return)
                current_episode_return = 0.0
                if not sample or budget_done:
                    break
                episode_index += 1
                episode_seed = derive_seed(
                    "freq_hrl_mujoco_episode_reset_v1",
                    int(seed),
                    int(episode_index),
                )
                observation, _ = env.reset(seed=episode_seed)
                model.reset_recurrent_inference()
                previous_action = np.zeros(action_dim, dtype=np.float32)
                lower_lf = np.zeros(action_dim, dtype=np.float64)
                episode_step = 0
                reset_exogenous = True
            else:
                observation = next_observation
                episode_step += 1

        batch = JointTrajectoryBatch(
            state=np.asarray(data["state"], dtype=np.float32),
            action=np.asarray(data["action"], dtype=np.float32),
            reward=np.asarray(data["reward"], dtype=np.float32),
            done=np.asarray(data["done"], dtype=np.float32),
            old_logp=np.asarray(data["old_logp"], dtype=np.float32),
            old_value=np.asarray(data["old_value"], dtype=np.float32),
        )
        if sample:
            batch = _with_explicit_bootstrap(
                batch,
                boundary_next_values=boundary_next_values,
                boundary_terminals=boundary_terminals,
            )
        zeros = [np.zeros(action_dim, dtype=np.float32) for _ in rewards]
        row = _episode_row(
            seed=seed,
            env_id=env_id,
            disturbance_mode=disturbance_mode,
            rewards=rewards,
            executed_actions=executed_actions,
            upper_actions=zeros,
            lower_actions=executed_actions,
            upper_policy_actions=zeros,
            upper_promotion_actions=zeros,
            upper_promotion_contexts=zeros,
            raw_lower_actions=nominal_actions,
            latent_lower_actions=nominal_actions,
            lower_router_removed_actions=zeros,
            lower_router_reconstruction_errors=zeros,
            lower_router_clip_values=[0.0 for _ in rewards],
            lower_router_audit_alphas=[0.0 for _ in rewards],
            lower_router_audit_imbalances=[0.0 for _ in rewards],
            lower_router_macro_projection_rates=[0.0 for _ in rewards],
            lower_router_macro_debt_rms_values=[0.0 for _ in rewards],
            lower_router_macro_completion_errors=[0.0 for _ in rewards],
            lower_router_headroom_clip_values=[0.0 for _ in rewards],
            responsibility_transfers=zeros,
            requested_transfers=zeros,
            transfer_saturation_values=[0.0 for _ in rewards],
            reconstruction_errors=zeros,
            forward_rewards=forward_rewards,
            control_rewards=control_rewards,
            upper_decisions=len(rewards),
            upper_transitions=len(rewards),
            lower_transitions=len(rewards),
            method="flat_ppo",
            segment_returns=segment_returns,
            natural_episode_returns=natural_episode_returns,
            boundary_terminals=boundary_terminals,
            transition_budget=transition_budget,
            lower_lf_rms_values=lower_lf_rms_values,
            lower_lf_budget_excesses=lower_lf_budget_excesses,
            raw_lower_lf_rms_values=lower_lf_rms_values,
            raw_lower_lf_budget_excesses=lower_lf_budget_excesses,
            latent_lower_lf_rms_values=lower_lf_rms_values,
            latent_lower_lf_budget_excesses=lower_lf_budget_excesses,
            lower_constraint_costs=[0.0 for _ in rewards],
            lower_lf_rms_budget=lower_lf_rms_budget,
            leakage_constraint_scope="disabled",
            upper_transition_delta_rms_values=[0.0 for _ in rewards],
            upper_hf_rms_values=[0.0 for _ in rewards],
            upper_hf_budget_excesses=[0.0 for _ in rewards],
            upper_hf_penalties=[0.0 for _ in rewards],
            upper_constraint_costs=[0.0 for _ in rewards],
            upper_constraint_mode="disabled",
            upper_hf_rms_budget=DEFAULT_UPPER_HF_RMS_BUDGET,
            upper_hf_penalty_coef=0.0,
            leakage_cost_mode="ratio_excess_squared",
            lower_action_router_mode="direct",
            lower_action_router_strength=1.0,
            upper_promotion_gain=0.0,
        )
        return (batch if sample else None), row
    finally:
        env.close()


SUMMARY_KEYS = [
    "episode_return",
    "reward_mean",
    "episode_length",
    "rollout_segment_count",
    "rollout_segment_return_mean",
    "natural_episode_count",
    "natural_episode_return_sum",
    "trace_boundary_count",
    "mdp_terminal_count",
    "bootstrap_boundary_count",
    "transition_budget_exact",
    "forward_reward_sum",
    "control_reward_sum",
    "action_energy",
    "action_smoothness",
    "UpperActionRMS",
    "LowerActionRMS",
    "UpperPolicyActionRMS",
    "UpperPromotionRMS",
    "UpperPromotionActivationRate",
    "UpperPromotionContextRMS",
    "UpperPromotionGain",
    "RawLowerActionRMS",
    "LatentLowerActionRMS",
    "LowerRouterRemovedRMS",
    "LowerRouterUpperTransferRMS",
    "LowerRouterFunctionPreserving",
    "LowerRouterActionReconstructionRMS",
    "LowerRouterClipRate",
    "LowerRouterAuditAlphaMean",
    "LowerRouterAuditAlphaFinal",
    "LowerRouterAuditBandImbalanceMean",
    "LowerRouterMacroProjectionRate",
    "LowerRouterMacroDebtRMSMean",
    "LowerRouterMacroCompletionErrorMax",
    "LowerRouterHeadroomClipRate",
    "LowerActionRouterStrength",
    "EffectiveToLatentLowerEnergyRatio",
    "UpperActionEnergyShare",
    "AdditiveActionClipRate",
    "AdditiveActionClipExcessMax",
    "AdditiveActionClipExcessRMS",
    "UpperHFPower",
    "UpperHFPowerAbs",
    "LatentUpperHFPower",
    "LatentUpperHFPowerAbs",
    "LowerLFDrift",
    "LowerLFDriftAbs",
    "RawLowerLFDrift",
    "RawLowerLFDriftAbs",
    "LatentLowerLFDrift",
    "LatentLowerLFDriftAbs",
    "TransferredLFDriftAbs",
    "ResponsibilityTransferRMS",
    "RequestedResponsibilityTransferRMS",
    "ResponsibilityTransferActivationRate",
    "ResponsibilityTransferHeadroomSaturationRate",
    "ResponsibilityReconstructionRMS",
    "LowerContributionOutOfUnitRate",
    "LowerLFRmsOnlineMean",
    "LowerLFPowerOnlineMean",
    "LowerLFBudgetExcessMean",
    "LowerLFBudgetViolationRate",
    "RawLowerLFRmsOnlineMean",
    "RawLowerLFPowerOnlineMean",
    "RawLowerLFBudgetExcessMean",
    "RawLowerLFBudgetViolationRate",
    "LatentLowerLFRmsOnlineMean",
    "LatentLowerLFPowerOnlineMean",
    "LatentLowerLFBudgetExcessMean",
    "LatentLowerLFBudgetViolationRate",
    "LowerConstraintCostMean",
    "LowerConstraintCostMax",
    "LowerLFRmsBudget",
    "UpperTransitionDeltaRMSMean",
    "UpperTransitionDeltaRMSMax",
    "UpperHFRmsOnlineMean",
    "UpperHFPowerOnlineMean",
    "UpperHFBudgetExcessMean",
    "UpperHFBudgetViolationRate",
    "UpperHFPenaltyMean",
    "UpperHFPenaltyTotal",
    "UpperConstraintCostMean",
    "UpperConstraintCostMax",
    "UpperHFRMSBudget",
    "UpperHFPenaltyCoef",
    "upper_decision_count",
    "upper_transition_count",
    "lower_transition_count",
    "protocol_valid",
]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return summarize_numeric_rows(rows, keys=SUMMARY_KEYS)


def _one_sided_bootstrap_bounds(
    values: np.ndarray,
    *,
    confidence: float,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("safe-selector bootstrap values must be finite")
    if not 0.5 < float(confidence) < 1.0:
        raise ValueError("safe-selector confidence must be in (0.5, 1)")
    if int(draws) < 100:
        raise ValueError("safe-selector bootstrap requires at least 100 draws")
    if array.size == 1 or np.all(array == array[0]):
        value = float(np.mean(array))
        return value, value
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(
        0,
        array.size,
        size=(int(draws), array.size),
    )
    means = np.mean(array[indices], axis=1)
    alpha = 1.0 - float(confidence)
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def select_safe_mujoco_branch(
    branch_rows: dict[str, list[dict[str, Any]]],
    *,
    bootstrap_seed: int,
    reward_margin_fraction: float = SAFE_SELECTOR_REWARD_MARGIN_FRACTION,
    minimum_drift_reduction_fraction: float = (
        SAFE_SELECTOR_MIN_DRIFT_REDUCTION_FRACTION
    ),
    minimum_raw_drift_reduction_fraction: float = (
        SAFE_SELECTOR_MIN_RAW_DRIFT_REDUCTION_FRACTION
    ),
    maximum_upper_hf_rms: float = SAFE_SELECTOR_MAX_UPPER_HF_RMS,
    confidence: float = SAFE_SELECTOR_CONFIDENCE,
    bootstrap_draws: int = SAFE_SELECTOR_BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Choose a constrained branch only when paired safety paths support it."""
    if set(branch_rows) != set(SAFE_SELECTOR_BRANCHES):
        raise ValueError("safe-selector branch registry is incomplete")
    if not 0.0 <= float(reward_margin_fraction) < 1.0:
        raise ValueError("safe-selector reward margin must be in [0, 1)")
    if not 0.0 < float(minimum_drift_reduction_fraction) < 1.0:
        raise ValueError("safe-selector drift reduction must be in (0, 1)")
    if not 0.0 < float(minimum_raw_drift_reduction_fraction) < 1.0:
        raise ValueError("safe-selector raw drift reduction must be in (0, 1)")
    if (
        not np.isfinite(float(maximum_upper_hf_rms))
        or float(maximum_upper_hf_rms) <= 0.0
    ):
        raise ValueError("safe-selector upper-HF RMS budget must be positive")

    def indexed(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
        output: dict[tuple[str, int], dict[str, Any]] = {}
        for row in rows:
            key = (str(row["disturbance_mode"]), int(row["seed"]))
            if key in output:
                raise ValueError("safe-selector paths must be unique")
            output[key] = row
        if not output:
            raise ValueError("safe-selector requires at least one path")
        return output

    indexed_rows = {branch: indexed(rows) for branch, rows in branch_rows.items()}
    baseline_index = indexed_rows[SAFE_SELECTOR_BASELINE_BRANCH]
    path_keys = tuple(sorted(baseline_index))
    if any(set(rows) != set(path_keys) for rows in indexed_rows.values()):
        raise ValueError("safe-selector branches must use paired paths")
    baseline_reward = np.asarray([
        float(baseline_index[key]["episode_return"]) for key in path_keys
    ])
    baseline_drift = np.asarray([
        float(baseline_index[key]["LowerLFDriftAbs"]) for key in path_keys
    ])
    baseline_raw_drift = np.asarray([
        float(baseline_index[key]["RawLowerLFDriftAbs"])
        for key in path_keys
    ])
    baseline_upper_hf_power = np.asarray([
        float(baseline_index[key]["UpperHFPowerAbs"])
        for key in path_keys
    ])
    baseline_reward_mean = float(np.mean(baseline_reward))
    baseline_drift_mean = float(np.mean(baseline_drift))
    baseline_raw_drift_mean = float(np.mean(baseline_raw_drift))
    baseline_upper_hf_rms = float(np.sqrt(np.mean(
        baseline_upper_hf_power
    )))
    independent_seeds = tuple(sorted({seed for _, seed in path_keys}))
    modes_by_seed = {
        seed: tuple(sorted(
            mode for mode, item_seed in path_keys if item_seed == seed
        ))
        for seed in independent_seeds
    }
    if len(set(modes_by_seed.values())) != 1:
        raise ValueError(
            "safe-selector disturbance modes must be balanced within seed"
        )
    reward_margin = float(reward_margin_fraction) * max(
        abs(baseline_reward_mean), 1.0
    )
    required_drift_reduction = (
        float(minimum_drift_reduction_fraction) * baseline_drift_mean
    )
    required_raw_drift_reduction = (
        float(minimum_raw_drift_reduction_fraction)
        * baseline_raw_drift_mean
    )
    diagnostics: dict[str, dict[str, Any]] = {
        SAFE_SELECTOR_BASELINE_BRANCH: {
            "path_count": len(path_keys),
            "independent_seed_count": len(independent_seeds),
            "episode_return_mean": baseline_reward_mean,
            "LowerLFDriftAbs_mean": baseline_drift_mean,
            "RawLowerLFDriftAbs_mean": baseline_raw_drift_mean,
            "UpperHFRMS": baseline_upper_hf_rms,
            "reward_difference_mean": 0.0,
            "reward_difference_one_sided_lower": 0.0,
            "reward_difference_one_sided_upper": 0.0,
            "drift_difference_mean": 0.0,
            "drift_difference_one_sided_lower": 0.0,
            "drift_difference_one_sided_upper": 0.0,
            "raw_drift_difference_mean": 0.0,
            "raw_drift_difference_one_sided_lower": 0.0,
            "raw_drift_difference_one_sided_upper": 0.0,
            "upper_hf_rms_one_sided_upper": baseline_upper_hf_rms,
            "reward_noninferiority_supported": True,
            "minimum_drift_reduction_supported": False,
            "minimum_raw_drift_reduction_supported": False,
            "upper_hf_budget_supported": bool(
                baseline_upper_hf_rms <= float(maximum_upper_hf_rms)
            ),
            "minimum_normalized_safety_slack": 0.0,
            "feasible": True,
        }
    }
    feasible_candidates: list[str] = []
    for branch in SAFE_SELECTOR_BRANCHES:
        if branch == SAFE_SELECTOR_BASELINE_BRANCH:
            continue
        rows = indexed_rows[branch]
        reward = np.asarray([
            float(rows[key]["episode_return"]) for key in path_keys
        ])
        drift = np.asarray([
            float(rows[key]["LowerLFDriftAbs"]) for key in path_keys
        ])
        raw_drift = np.asarray([
            float(rows[key]["RawLowerLFDriftAbs"]) for key in path_keys
        ])
        upper_hf_power = np.asarray([
            float(rows[key]["UpperHFPowerAbs"]) for key in path_keys
        ])
        reward_difference = reward - baseline_reward
        drift_difference = drift - baseline_drift
        raw_drift_difference = raw_drift - baseline_raw_drift
        reward_difference_by_seed = np.asarray([
            float(np.mean([
                float(rows[(mode, seed)]["episode_return"])
                - float(baseline_index[(mode, seed)]["episode_return"])
                for mode in modes_by_seed[seed]
            ]))
            for seed in independent_seeds
        ])
        drift_difference_by_seed = np.asarray([
            float(np.mean([
                float(rows[(mode, seed)]["LowerLFDriftAbs"])
                - float(baseline_index[(mode, seed)]["LowerLFDriftAbs"])
                for mode in modes_by_seed[seed]
            ]))
            for seed in independent_seeds
        ])
        raw_drift_difference_by_seed = np.asarray([
            float(np.mean([
                float(rows[(mode, seed)]["RawLowerLFDriftAbs"])
                - float(baseline_index[(mode, seed)]["RawLowerLFDriftAbs"])
                for mode in modes_by_seed[seed]
            ]))
            for seed in independent_seeds
        ])
        upper_hf_power_by_seed = np.asarray([
            float(np.mean([
                float(rows[(mode, seed)]["UpperHFPowerAbs"])
                for mode in modes_by_seed[seed]
            ]))
            for seed in independent_seeds
        ])
        reward_lower, reward_upper = _one_sided_bootstrap_bounds(
            reward_difference_by_seed,
            confidence=confidence,
            draws=bootstrap_draws,
            seed=derive_seed(
                "mujoco_safe_selector_reward_bootstrap_v1",
                int(bootstrap_seed),
                branch,
            ),
        )
        drift_lower, drift_upper = _one_sided_bootstrap_bounds(
            drift_difference_by_seed,
            confidence=confidence,
            draws=bootstrap_draws,
            seed=derive_seed(
                "mujoco_safe_selector_drift_bootstrap_v1",
                int(bootstrap_seed),
                branch,
            ),
        )
        raw_drift_lower, raw_drift_upper = _one_sided_bootstrap_bounds(
            raw_drift_difference_by_seed,
            confidence=confidence,
            draws=bootstrap_draws,
            seed=derive_seed(
                "mujoco_safe_selector_raw_drift_bootstrap_v2",
                int(bootstrap_seed),
                branch,
            ),
        )
        _, upper_hf_power_upper = _one_sided_bootstrap_bounds(
            upper_hf_power_by_seed,
            confidence=confidence,
            draws=bootstrap_draws,
            seed=derive_seed(
                "mujoco_safe_selector_upper_hf_bootstrap_v2",
                int(bootstrap_seed),
                branch,
            ),
        )
        upper_hf_rms = float(np.sqrt(np.mean(upper_hf_power)))
        upper_hf_rms_upper = float(np.sqrt(max(
            upper_hf_power_upper, 0.0
        )))
        reward_supported = reward_lower >= -reward_margin
        drift_supported = (
            baseline_drift_mean > np.finfo(np.float64).eps
            and drift_upper <= -required_drift_reduction
        )
        raw_drift_supported = (
            baseline_raw_drift_mean > np.finfo(np.float64).eps
            and raw_drift_upper <= -required_raw_drift_reduction
        )
        upper_hf_supported = (
            upper_hf_rms_upper <= float(maximum_upper_hf_rms)
        )
        normalized_slacks = (
            (reward_lower + reward_margin) / max(reward_margin, 1e-12),
            (-required_drift_reduction - drift_upper)
            / max(required_drift_reduction, 1e-12),
            (-required_raw_drift_reduction - raw_drift_upper)
            / max(required_raw_drift_reduction, 1e-12),
            (float(maximum_upper_hf_rms) - upper_hf_rms_upper)
            / float(maximum_upper_hf_rms),
        )
        minimum_safety_slack = float(min(normalized_slacks))
        feasible = bool(
            reward_supported
            and drift_supported
            and raw_drift_supported
            and upper_hf_supported
        )
        diagnostics[branch] = {
            "path_count": len(path_keys),
            "independent_seed_count": len(independent_seeds),
            "episode_return_mean": float(np.mean(reward)),
            "LowerLFDriftAbs_mean": float(np.mean(drift)),
            "RawLowerLFDriftAbs_mean": float(np.mean(raw_drift)),
            "UpperHFRMS": upper_hf_rms,
            "reward_difference_mean": float(np.mean(reward_difference)),
            "reward_difference_one_sided_lower": reward_lower,
            "reward_difference_one_sided_upper": reward_upper,
            "drift_difference_mean": float(np.mean(drift_difference)),
            "drift_difference_one_sided_lower": drift_lower,
            "drift_difference_one_sided_upper": drift_upper,
            "raw_drift_difference_mean": float(np.mean(
                raw_drift_difference
            )),
            "raw_drift_difference_one_sided_lower": raw_drift_lower,
            "raw_drift_difference_one_sided_upper": raw_drift_upper,
            "upper_hf_rms_one_sided_upper": upper_hf_rms_upper,
            "reward_noninferiority_supported": bool(reward_supported),
            "minimum_drift_reduction_supported": bool(drift_supported),
            "minimum_raw_drift_reduction_supported": bool(
                raw_drift_supported
            ),
            "upper_hf_budget_supported": bool(upper_hf_supported),
            "minimum_normalized_safety_slack": minimum_safety_slack,
            "feasible": feasible,
        }
        if feasible:
            feasible_candidates.append(branch)

    selected_branch = (
        min(
            feasible_candidates,
            key=lambda branch: (
                -float(diagnostics[branch][
                    "minimum_normalized_safety_slack"
                ]),
                -float(diagnostics[branch][
                    "reward_difference_one_sided_lower"
                ]),
                branch,
            ),
        )
        if feasible_candidates else SAFE_SELECTOR_BASELINE_BRANCH
    )
    return {
        "protocol": (
            "paired_seed_cluster_bootstrap_reward_responsibility_raw_and_"
            "upper_hf_gate_v2"
        ),
        "inference_unit": "independent_safety_selection_seed",
        "selected_branch": selected_branch,
        "selection_status": (
            "constrained_branch_selected"
            if feasible_candidates else "fallback_to_no_leakage"
        ),
        "baseline_branch": SAFE_SELECTOR_BASELINE_BRANCH,
        "candidate_branches": [
            branch for branch in SAFE_SELECTOR_BRANCHES
            if branch != SAFE_SELECTOR_BASELINE_BRANCH
        ],
        "paired_path_count": len(path_keys),
        "independent_seed_count": len(independent_seeds),
        "confidence": float(confidence),
        "bootstrap_draws": int(bootstrap_draws),
        "reward_margin_fraction": float(reward_margin_fraction),
        "reward_noninferiority_margin": reward_margin,
        "minimum_drift_reduction_fraction": float(
            minimum_drift_reduction_fraction
        ),
        "required_absolute_drift_reduction": required_drift_reduction,
        "minimum_raw_drift_reduction_fraction": float(
            minimum_raw_drift_reduction_fraction
        ),
        "required_absolute_raw_drift_reduction": (
            required_raw_drift_reduction
        ),
        "maximum_upper_hf_rms": float(maximum_upper_hf_rms),
        "branch_diagnostics": diagnostics,
    }


def _hierarchical_model(
    *,
    state_dim: int,
    action_dim: int,
    hidden_dim: int,
    learning_rate: float,
    leakage_constraint: bool,
    upper_cost_critic: bool = False,
    upper_constraint: bool = False,
    upper_dual_lr: float = DEFAULT_UPPER_DUAL_LR,
    lower_dual_lr: float = DEFAULT_LOWER_DUAL_LR,
    upper_constraint_update_mode: str = "reward_guarded_adam_projection",
    lower_constraint_update_mode: str = "reward_guarded_adam_projection",
    upper_actor_anchor_coef: float = 0.0,
    lower_actor_anchor_coef: float = 0.0,
    actor_anchor_zero_state_indices: tuple[int, ...] = (),
    upper_deployment_frequency_dual_lr: float = 0.0,
    lower_deployment_frequency_dual_lr: float = 0.0,
    upper_deployment_frequency_lambda_init: float = 0.0,
    lower_deployment_frequency_lambda_init: float = 0.0,
    upper_deployment_frequency_step_scale: float = 1.0,
    lower_deployment_frequency_step_scale: float = 1.0,
    upper_deployment_frequency_max_projection_steps: int = 1,
    lower_deployment_frequency_max_projection_steps: int = 1,
    upper_deployment_frequency_reward_tolerance: float = 1e-8,
    lower_deployment_frequency_reward_tolerance: float = 1e-8,
    upper_deployment_frequency_target_tolerance: float = 0.0,
    lower_deployment_frequency_target_tolerance: float = 0.0,
    upper_deployment_frequency_rms_budget: float = 0.0,
    lower_deployment_frequency_rms_budget: float = 0.0,
    upper_deployment_frequency_reference_reduction_fraction: float = 0.0,
    lower_deployment_frequency_reference_reduction_fraction: float = 0.0,
    upper_deployment_frequency_action_scale: float = 1.0,
    lower_deployment_frequency_action_scale: float = 1.0,
    deployment_frequency_groupwise_robust: bool = False,
    deployment_frequency_anchor_state_replay: bool = False,
    deployment_frequency_projection_objective: str = "worst_group",
    deployment_frequency_projection_cvar_alpha: float = 0.5,
    deployment_frequency_restoration_freeze_reward_actor: bool = False,
    deployment_frequency_ppo_trust_region: bool = False,
    deployment_frequency_ppo_trust_region_backtracks: int = 8,
    deployment_frequency_closed_loop_trust_region: bool = False,
    deployment_frequency_closed_loop_trust_region_backtracks: int = 8,
    deployment_frequency_closed_loop_restoration_filter: bool = False,
    deployment_frequency_closed_loop_restoration_min_reduction: float = 1e-4,
    deployment_frequency_closed_loop_restoration_funnel_multiplier: float = 3.0,
    constraint_dual_normalization: str = "none",
    constraint_dual_scale_ema_beta: float = 0.95,
    constraint_dual_scale_floor: float = 1e-6,
) -> FrequencySeparatedActorCriticPPO:
    return FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
        upper_state_dim=state_dim,
        lower_state_dim=state_dim,
        upper_action_dim=action_dim,
        lower_action_dim=action_dim,
        hidden_dim=int(hidden_dim),
        upper_learning_rate=float(learning_rate),
        lower_learning_rate=float(learning_rate),
        epochs=4,
        minibatch_size=512,
        init_log_std=-0.7,
        deployment_action_transform="tanh",
        upper_deployment_frequency_rms_budget=float(
            upper_deployment_frequency_rms_budget
        ),
        upper_deployment_frequency_reference_reduction_fraction=float(
            upper_deployment_frequency_reference_reduction_fraction
        ),
        upper_deployment_frequency_window=8,
        upper_deployment_frequency_action_scale=float(
            upper_deployment_frequency_action_scale
        ),
        upper_deployment_frequency_dual_lr=float(
            upper_deployment_frequency_dual_lr
        ),
        upper_deployment_frequency_lambda_init=float(
            upper_deployment_frequency_lambda_init
        ),
        upper_deployment_frequency_max_lambda=20.0,
        upper_deployment_frequency_step_scale=float(
            upper_deployment_frequency_step_scale
        ),
        upper_deployment_frequency_max_projection_steps=int(
            upper_deployment_frequency_max_projection_steps
        ),
        upper_deployment_frequency_reward_tolerance=float(
            upper_deployment_frequency_reward_tolerance
        ),
        upper_deployment_frequency_target_tolerance=float(
            upper_deployment_frequency_target_tolerance
        ),
        lower_deployment_frequency_rms_budget=float(
            lower_deployment_frequency_rms_budget
        ),
        lower_deployment_frequency_reference_reduction_fraction=float(
            lower_deployment_frequency_reference_reduction_fraction
        ),
        lower_deployment_frequency_window=32,
        lower_deployment_frequency_action_scale=float(
            lower_deployment_frequency_action_scale
        ),
        lower_deployment_frequency_dual_lr=float(
            lower_deployment_frequency_dual_lr
        ),
        lower_deployment_frequency_lambda_init=float(
            lower_deployment_frequency_lambda_init
        ),
        lower_deployment_frequency_max_lambda=20.0,
        lower_deployment_frequency_step_scale=float(
            lower_deployment_frequency_step_scale
        ),
        lower_deployment_frequency_max_projection_steps=int(
            lower_deployment_frequency_max_projection_steps
        ),
        lower_deployment_frequency_reward_tolerance=float(
            lower_deployment_frequency_reward_tolerance
        ),
        lower_deployment_frequency_target_tolerance=float(
            lower_deployment_frequency_target_tolerance
        ),
        deployment_frequency_groupwise_robust=bool(
            deployment_frequency_groupwise_robust
        ),
        deployment_frequency_anchor_state_replay=bool(
            deployment_frequency_anchor_state_replay
        ),
        deployment_frequency_projection_objective=str(
            deployment_frequency_projection_objective
        ),
        deployment_frequency_projection_cvar_alpha=float(
            deployment_frequency_projection_cvar_alpha
        ),
        deployment_frequency_restoration_freeze_reward_actor=bool(
            deployment_frequency_restoration_freeze_reward_actor
        ),
        deployment_frequency_ppo_trust_region=bool(
            deployment_frequency_ppo_trust_region
        ),
        deployment_frequency_ppo_trust_region_backtracks=int(
            deployment_frequency_ppo_trust_region_backtracks
        ),
        deployment_frequency_closed_loop_trust_region=bool(
            deployment_frequency_closed_loop_trust_region
        ),
        deployment_frequency_closed_loop_trust_region_backtracks=int(
            deployment_frequency_closed_loop_trust_region_backtracks
        ),
        deployment_frequency_closed_loop_restoration_filter=bool(
            deployment_frequency_closed_loop_restoration_filter
        ),
        deployment_frequency_closed_loop_restoration_min_reduction=float(
            deployment_frequency_closed_loop_restoration_min_reduction
        ),
        deployment_frequency_closed_loop_restoration_funnel_multiplier=float(
            deployment_frequency_closed_loop_restoration_funnel_multiplier
        ),
        upper_actor_anchor_coef=float(upper_actor_anchor_coef),
        lower_actor_anchor_coef=float(lower_actor_anchor_coef),
        actor_anchor_zero_state_indices=tuple(
            map(int, actor_anchor_zero_state_indices)
        ),
        upper_cost_critic=bool(upper_cost_critic),
        upper_lambda_init=0.0,
        upper_cost_target=0.0,
        upper_dual_lr=(
            float(upper_dual_lr) if upper_constraint else 0.0
        ),
        upper_max_lambda=20.0,
        upper_cost_activation_threshold=1e-6,
        upper_zero_init_cost_value=True,
        upper_skip_inactive_cost_value_update=True,
        upper_constraint_update_mode=str(upper_constraint_update_mode),
        upper_constraint_step_scale=1.0,
        upper_constraint_max_backtracks=8,
        upper_constraint_reward_tolerance=1e-8,
        lower_lambda_init=0.0,
        lower_cost_target=0.0,
        lower_dual_lr=(
            float(lower_dual_lr) if leakage_constraint else 0.0
        ),
        lower_max_lambda=20.0,
        lower_cost_activation_threshold=1e-6,
        lower_zero_init_cost_value=True,
        lower_skip_inactive_cost_value_update=True,
        lower_constraint_update_mode=str(lower_constraint_update_mode),
        lower_constraint_step_scale=1.0,
        lower_constraint_max_backtracks=8,
        lower_constraint_reward_tolerance=1e-8,
        constraint_dual_normalization=str(
            constraint_dual_normalization
        ),
        constraint_dual_scale_ema_beta=float(
            constraint_dual_scale_ema_beta
        ),
        constraint_dual_scale_floor=float(constraint_dual_scale_floor),
    ))


def _validated_disturbance_modes(
    modes: Iterable[str],
    *,
    role: str,
) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(map(str, modes)))
    if not values or not set(values).issubset(DISTURBANCE_MODES):
        raise ValueError(f"MuJoCo {role} disturbance registry is invalid")
    return values


def _assign_seed_modes(
    seeds: Iterable[int],
    modes: tuple[str, ...],
) -> dict[int, str]:
    return {
        int(seed): modes[index % len(modes)]
        for index, seed in enumerate(seeds)
    }


def crossed_checkpoint_selection_paths(
    selection_roots: Iterable[int],
    modes: Iterable[str],
    *,
    env_id: str,
) -> tuple[list[int], dict[int, str]]:
    """Expand independent validation roots across every disturbance mode."""

    roots = validate_unique_seeds(
        selection_roots, role="mujoco_checkpoint_selection_roots"
    )
    condition_registry = _validated_disturbance_modes(
        modes, role="checkpoint selection"
    )
    path_seeds: list[int] = []
    assignments: dict[int, str] = {}
    for root in roots:
        for mode in condition_registry:
            path_seed = derive_seed(
                "mujoco_checkpoint_selection_crossed_v1",
                str(env_id),
                int(root),
                str(mode),
            )
            if path_seed in assignments:
                raise RuntimeError(
                    "crossed MuJoCo checkpoint path seeds collided"
                )
            path_seeds.append(path_seed)
            assignments[path_seed] = str(mode)
    return path_seeds, assignments


def crossed_deployment_frequency_guard_paths(
    guard_roots: Iterable[int],
    modes: Iterable[str],
    *,
    env_id: str,
) -> tuple[list[int], dict[int, str]]:
    """Expand independent trust-region roots across every disturbance mode."""

    roots = validate_unique_seeds(
        guard_roots, role="mujoco_deployment_frequency_guard_roots"
    )
    condition_registry = _validated_disturbance_modes(
        modes, role="deployment frequency closed-loop guard"
    )
    path_seeds: list[int] = []
    assignments: dict[int, str] = {}
    for root in roots:
        for mode in condition_registry:
            path_seed = derive_seed(
                "mujoco_deployment_frequency_closed_loop_guard_crossed_v1",
                str(env_id),
                int(root),
                str(mode),
            )
            if path_seed in assignments:
                raise RuntimeError(
                    "crossed MuJoCo closed-loop guard path seeds collided"
                )
            path_seeds.append(path_seed)
            assignments[path_seed] = str(mode)
    return path_seeds, assignments


def crossed_deployment_frequency_reference_paths(
    reference_roots: Iterable[int],
    modes: Iterable[str],
    *,
    env_id: str,
) -> tuple[list[int], dict[int, str]]:
    """Expand frozen constraint-state roots across every disturbance mode."""

    roots = validate_unique_seeds(
        reference_roots,
        role="mujoco_deployment_frequency_reference_roots",
    )
    condition_registry = _validated_disturbance_modes(
        modes, role="deployment frequency frozen-state reference"
    )
    path_seeds: list[int] = []
    assignments: dict[int, str] = {}
    for root in roots:
        for mode in condition_registry:
            path_seed = derive_seed(
                "mujoco_deployment_frequency_reference_crossed_v1",
                str(env_id),
                int(root),
                str(mode),
            )
            if path_seed in assignments:
                raise RuntimeError(
                    "crossed MuJoCo frequency-reference path seeds collided"
                )
            path_seeds.append(path_seed)
            assignments[path_seed] = str(mode)
    return path_seeds, assignments


def behavior_robust_checkpoint_diagnostics(
    rows: list[dict[str, Any]],
    *,
    expected_modes: Iterable[str],
    lower_lf_rms_budget: float,
    upper_hf_rms_budget: float,
    constraint_penalty: float,
    include_latent: bool = False,
) -> dict[str, Any]:
    """Score worst-condition reward while penalizing endpoint violations."""

    modes = tuple(dict.fromkeys(map(str, expected_modes)))
    if not rows or not modes:
        raise ValueError("behavior-robust checkpoint scoring needs rows and modes")
    if (
        not np.isfinite(float(lower_lf_rms_budget))
        or float(lower_lf_rms_budget) <= 0.0
        or not np.isfinite(float(upper_hf_rms_budget))
        or float(upper_hf_rms_budget) <= 0.0
    ):
        raise ValueError("checkpoint leakage budgets must be positive")
    if (
        not np.isfinite(float(constraint_penalty))
        or float(constraint_penalty) < 0.0
    ):
        raise ValueError("checkpoint constraint penalty must be non-negative")
    grouped = {
        mode: [
            row for row in rows
            if str(row.get("disturbance_mode")) == mode
        ]
        for mode in modes
    }
    missing = [mode for mode, values in grouped.items() if not values]
    unexpected = sorted({
        str(row.get("disturbance_mode")) for row in rows
    } - set(modes))
    if missing or unexpected:
        raise ValueError(
            "checkpoint rows do not match the expected disturbance registry: "
            f"missing={missing}, unexpected={unexpected}"
        )

    by_mode: dict[str, dict[str, float]] = {}
    for mode, values in grouped.items():
        metric_names = [
            "reward_mean",
            "LowerLFDriftAbs",
            "RawLowerLFDriftAbs",
            "UpperHFPowerAbs",
        ]
        if bool(include_latent):
            metric_names.extend((
                "LatentLowerLFDriftAbs",
                "LatentUpperHFPowerAbs",
            ))
        metrics = {
            key: np.asarray([float(row[key]) for row in values])
            for key in metric_names
        }
        if any(not np.all(np.isfinite(metric)) for metric in metrics.values()):
            raise ValueError("checkpoint metrics must be finite")
        lower_rms = float(np.sqrt(max(
            float(np.mean(metrics["LowerLFDriftAbs"])), 0.0
        )))
        raw_lower_rms = float(np.sqrt(max(
            float(np.mean(metrics["RawLowerLFDriftAbs"])), 0.0
        )))
        upper_hf_rms = float(np.sqrt(max(
            float(np.mean(metrics["UpperHFPowerAbs"])), 0.0
        )))
        diagnostics = {
            "reward_mean": float(np.mean(metrics["reward_mean"])),
            "lower_lf_rms": lower_rms,
            "raw_lower_lf_rms": raw_lower_rms,
            "upper_hf_rms": upper_hf_rms,
            "lower_violation": max(
                0.0, lower_rms / float(lower_lf_rms_budget) - 1.0
            ),
            "raw_lower_violation": max(
                0.0, raw_lower_rms / float(lower_lf_rms_budget) - 1.0
            ),
            "upper_hf_violation": max(
                0.0, upper_hf_rms / float(upper_hf_rms_budget) - 1.0
            ),
        }
        if bool(include_latent):
            latent_lower_rms = float(np.sqrt(max(
                float(np.mean(metrics["LatentLowerLFDriftAbs"])), 0.0
            )))
            latent_upper_hf_rms = float(np.sqrt(max(
                float(np.mean(metrics["LatentUpperHFPowerAbs"])), 0.0
            )))
            diagnostics.update({
                "latent_lower_lf_rms": latent_lower_rms,
                "latent_upper_hf_rms": latent_upper_hf_rms,
                "latent_lower_violation": max(
                    0.0,
                    latent_lower_rms / float(lower_lf_rms_budget) - 1.0,
                ),
                "latent_upper_hf_violation": max(
                    0.0,
                    latent_upper_hf_rms / float(upper_hf_rms_budget) - 1.0,
                ),
            })
        by_mode[mode] = diagnostics
    worst_reward = min(item["reward_mean"] for item in by_mode.values())
    reward_scale = max(
        1.0,
        max(abs(item["reward_mean"]) for item in by_mode.values()),
    )
    violation_keys = [
        "lower_violation",
        "raw_lower_violation",
        "upper_hf_violation",
    ]
    if bool(include_latent):
        violation_keys.extend((
            "latent_lower_violation",
            "latent_upper_hf_violation",
        ))
    worst_violations = {
        key: max(item[key] for item in by_mode.values())
        for key in violation_keys
    }
    normalized_penalty = float(sum(
        value * value for value in worst_violations.values()
    ))
    score = (
        float(worst_reward)
        - float(constraint_penalty) * reward_scale * normalized_penalty
    )
    return {
        "score": score,
        "worst_condition_reward_mean": float(worst_reward),
        "reward_scale": reward_scale,
        "normalized_constraint_penalty": normalized_penalty,
        "worst_normalized_violations": worst_violations,
        "by_disturbance": by_mode,
    }


def latent_behavior_feasibility_rank(
    rows: list[dict[str, Any]],
    *,
    expected_modes: Iterable[str],
    lower_lf_rms_budget: float,
    upper_hf_rms_budget: float,
) -> tuple[float, float, float]:
    """Rank one checkpoint by worst endpoint feasibility, then reward."""

    diagnostics = behavior_robust_checkpoint_diagnostics(
        rows,
        expected_modes=expected_modes,
        lower_lf_rms_budget=lower_lf_rms_budget,
        upper_hf_rms_budget=upper_hf_rms_budget,
        constraint_penalty=1.0,
        include_latent=True,
    )
    violations = np.asarray(
        list(diagnostics["worst_normalized_violations"].values()),
        dtype=np.float64,
    )
    if violations.size != 5 or not np.all(np.isfinite(violations)):
        raise ValueError("latent checkpoint endpoint registry is invalid")
    return (
        -float(np.max(violations)),
        -float(np.sum(np.square(violations))),
        float(diagnostics["worst_condition_reward_mean"]),
    )


def paired_relative_frequency_feasibility_diagnostics(
    rows: list[dict[str, Any]],
    *,
    baseline_rows: list[dict[str, Any]],
    expected_modes: Iterable[str],
    lower_reduction_fraction: float,
    upper_reduction_fraction: float,
    lower_power_floor: float,
    upper_power_floor: float,
    reward_noninferiority_margin_fraction: float = 0.02,
    pathwise_robust: bool = False,
    risk_mode: str = "legacy",
    cvar_alpha: float = 0.5,
) -> dict[str, Any]:
    """Audit paired reward and frequency feasibility by mode and endpoint."""

    modes = tuple(dict.fromkeys(map(str, expected_modes)))
    if not rows or not baseline_rows or not modes:
        raise ValueError("paired-relative checkpoint ranking needs paired rows")
    candidate_keys = {
        (str(row.get("disturbance_mode")), int(row.get("seed")))
        for row in rows
    }
    baseline_keys = {
        (str(row.get("disturbance_mode")), int(row.get("seed")))
        for row in baseline_rows
    }
    if (
        len(candidate_keys) != len(rows)
        or len(baseline_keys) != len(baseline_rows)
        or candidate_keys != baseline_keys
    ):
        raise ValueError(
            "paired-relative checkpoint rows must use identical unique paths"
        )
    for label, fraction in (
        ("lower", lower_reduction_fraction),
        ("upper", upper_reduction_fraction),
    ):
        if not np.isfinite(float(fraction)) or not 0.0 <= float(fraction) < 1.0:
            raise ValueError(
                f"paired-relative {label} reduction must be in [0, 1)"
            )
    for label, floor in (
        ("lower", lower_power_floor),
        ("upper", upper_power_floor),
    ):
        if not np.isfinite(float(floor)) or float(floor) <= 0.0:
            raise ValueError(
                f"paired-relative {label} power floor must be positive"
            )
    margin_fraction = float(reward_noninferiority_margin_fraction)
    if not np.isfinite(margin_fraction) or margin_fraction < 0.0:
        raise ValueError("paired-relative reward margin must be non-negative")
    if not isinstance(pathwise_robust, bool):
        raise TypeError("paired-relative pathwise_robust must be boolean")
    resolved_risk_mode = _resolve_closed_loop_risk_mode(
        pathwise_robust=pathwise_robust, risk_mode=risk_mode
    )
    if (
        not np.isfinite(float(cvar_alpha))
        or not 0.0 <= float(cvar_alpha) < 1.0
    ):
        raise ValueError("paired-relative CVaR alpha must be in [0, 1)")

    metrics = (
        "reward_mean",
        "LowerLFDriftAbs",
        "RawLowerLFDriftAbs",
        "LatentLowerLFDriftAbs",
        "UpperHFPowerAbs",
        "LatentUpperHFPowerAbs",
    )

    def metric_row(row: dict[str, Any]) -> dict[str, float]:
        values = {
            metric: float(row[metric]) for metric in metrics
        }
        if not np.all(np.isfinite(list(values.values()))):
            raise ValueError("paired-relative checkpoint metrics must be finite")
        return values

    def mode_means(source: list[dict[str, Any]], mode: str) -> dict[str, float]:
        selected = [
            row for row in source
            if str(row.get("disturbance_mode")) == mode
        ]
        if not selected:
            raise ValueError(
                f"paired-relative checkpoint rows omit mode {mode}"
            )
        values = {
            metric: np.asarray(
                [float(row[metric]) for row in selected], dtype=np.float64
            )
            for metric in metrics
        }
        if any(not np.all(np.isfinite(value)) for value in values.values()):
            raise ValueError("paired-relative checkpoint metrics must be finite")
        return {metric: float(np.mean(value)) for metric, value in values.items()}

    candidate_index = {
        (str(row["disturbance_mode"]), int(row["seed"])): row
        for row in rows
    }
    baseline_index = {
        (str(row["disturbance_mode"]), int(row["seed"])): row
        for row in baseline_rows
    }
    comparison_groups: list[
        tuple[str, int | None, dict[str, float], dict[str, float]]
    ] = []
    if resolved_risk_mode in {"pathwise_all", "mode_cvar"}:
        for mode in modes:
            mode_keys = sorted(
                key for key in candidate_keys if key[0] == mode
            )
            if not mode_keys:
                raise ValueError(
                    f"paired-relative checkpoint rows omit mode {mode}"
                )
            comparison_groups.extend(
                (
                    mode,
                    int(seed),
                    metric_row(baseline_index[mode, seed]),
                    metric_row(candidate_index[mode, seed]),
                )
                for _, seed in mode_keys
            )
    else:
        comparison_groups.extend(
            (
                mode,
                None,
                mode_means(baseline_rows, mode),
                mode_means(rows, mode),
            )
            for mode in modes
        )

    violations: list[float] = []
    reward_slacks: list[float] = []
    constraints: list[dict[str, Any]] = []
    for mode, path_seed, baseline, candidate in comparison_groups:
        reward_scale = max(abs(baseline["reward_mean"]), 1.0)
        reward_floor = (
            baseline["reward_mean"] - margin_fraction * reward_scale
        )
        reward_violation = max(
            (reward_floor - candidate["reward_mean"]) / reward_scale,
            0.0,
        )
        violations.append(reward_violation)
        reward_slack = (
            (candidate["reward_mean"] - reward_floor) / reward_scale
        )
        reward_slacks.append(reward_slack)
        reward_constraint = {
            "disturbance_mode": mode,
            "endpoint": "reward_mean",
            "direction": "minimum",
            "baseline": baseline["reward_mean"],
            "candidate": candidate["reward_mean"],
            "target": reward_floor,
            "normalized_violation": reward_violation,
            "normalized_slack": reward_slack,
        }
        if path_seed is not None:
            reward_constraint["seed"] = int(path_seed)
        constraints.append(reward_constraint)
        for metric in (
            "LowerLFDriftAbs",
            "RawLowerLFDriftAbs",
            "LatentLowerLFDriftAbs",
        ):
            target = max(
                (1.0 - float(lower_reduction_fraction)) * baseline[metric],
                float(lower_power_floor),
            )
            violation = max(candidate[metric] / target - 1.0, 0.0)
            violations.append(violation)
            constraint = {
                "disturbance_mode": mode,
                "endpoint": metric,
                "direction": "maximum",
                "baseline": baseline[metric],
                "candidate": candidate[metric],
                "target": target,
                "normalized_violation": violation,
                "normalized_slack": 1.0 - candidate[metric] / target,
            }
            if path_seed is not None:
                constraint["seed"] = int(path_seed)
            constraints.append(constraint)
        for metric in ("UpperHFPowerAbs", "LatentUpperHFPowerAbs"):
            target = max(
                (1.0 - float(upper_reduction_fraction)) * baseline[metric],
                float(upper_power_floor),
            )
            violation = max(candidate[metric] / target - 1.0, 0.0)
            violations.append(violation)
            constraint = {
                "disturbance_mode": mode,
                "endpoint": metric,
                "direction": "maximum",
                "baseline": baseline[metric],
                "candidate": candidate[metric],
                "target": target,
                "normalized_violation": violation,
                "normalized_slack": 1.0 - candidate[metric] / target,
            }
            if path_seed is not None:
                constraint["seed"] = int(path_seed)
            constraints.append(constraint)
    if resolved_risk_mode == "mode_cvar":
        path_constraints = constraints
        constraints = []
        violations = []
        reward_slacks = []
        for mode in modes:
            for metric in metrics:
                selected = [
                    item for item in path_constraints
                    if (
                        str(item["disturbance_mode"]) == mode
                        and str(item["endpoint"]) == metric
                    )
                ]
                if not selected:
                    raise ValueError(
                        "paired-relative CVaR aggregation omitted an endpoint"
                    )
                signed_excesses = [
                    -float(item["normalized_slack"]) for item in selected
                ]
                cvar = _empirical_upper_tail_cvar(
                    signed_excesses, alpha=float(cvar_alpha)
                )
                violation = max(cvar, 0.0)
                aggregate = {
                    "disturbance_mode": mode,
                    "endpoint": metric,
                    "direction": str(selected[0]["direction"]),
                    "baseline": float(np.mean([
                        float(item["baseline"]) for item in selected
                    ])),
                    "candidate": float(np.mean([
                        float(item["candidate"]) for item in selected
                    ])),
                    "target": float(np.mean([
                        float(item["target"]) for item in selected
                    ])),
                    "normalized_violation": violation,
                    "normalized_slack": -cvar,
                    "normalized_signed_excess_cvar": cvar,
                    "cvar_alpha": float(cvar_alpha),
                    "path_count": len(selected),
                    "path_signed_normalized_excesses": signed_excesses,
                }
                constraints.append(aggregate)
                violations.append(violation)
                if metric == "reward_mean":
                    reward_slacks.append(-cvar)
    values = np.asarray(violations, dtype=np.float64)
    comparison_group_count = (
        len(modes)
        if resolved_risk_mode == "mode_cvar"
        else len(comparison_groups)
    )
    if (
        values.size != comparison_group_count * 6
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("paired-relative checkpoint violation registry is invalid")
    worst = max(
        constraints,
        key=lambda item: (
            float(item["normalized_violation"]),
            str(item["disturbance_mode"]),
            str(item["endpoint"]),
        ),
    )
    return {
        "rank": (
            -float(np.max(values)),
            -float(np.sum(np.square(values))),
            float(min(reward_slacks)),
        ),
        "constraint_count": len(constraints),
        "comparison_group_count": comparison_group_count,
        "aggregation": {
            "pathwise_all": "pathwise",
            "mode_mean": "disturbance_mode_mean",
            "mode_cvar": "disturbance_mode_cvar",
        }[resolved_risk_mode],
        "risk_mode": resolved_risk_mode,
        "cvar_alpha": float(cvar_alpha),
        "constraints": constraints,
        "worst_constraint": worst,
    }


def paired_relative_frequency_feasibility_rank(
    rows: list[dict[str, Any]],
    *,
    baseline_rows: list[dict[str, Any]],
    expected_modes: Iterable[str],
    lower_reduction_fraction: float,
    upper_reduction_fraction: float,
    lower_power_floor: float,
    upper_power_floor: float,
    reward_noninferiority_margin_fraction: float = 0.02,
    pathwise_robust: bool = False,
    risk_mode: str = "legacy",
    cvar_alpha: float = 0.5,
) -> tuple[float, float, float]:
    """Rank against a paired checkpoint on the same selection paths."""

    diagnostics = paired_relative_frequency_feasibility_diagnostics(
        rows,
        baseline_rows=baseline_rows,
        expected_modes=expected_modes,
        lower_reduction_fraction=lower_reduction_fraction,
        upper_reduction_fraction=upper_reduction_fraction,
        lower_power_floor=lower_power_floor,
        upper_power_floor=upper_power_floor,
        reward_noninferiority_margin_fraction=(
            reward_noninferiority_margin_fraction
        ),
        pathwise_robust=pathwise_robust,
        risk_mode=risk_mode,
        cvar_alpha=cvar_alpha,
    )
    return tuple(map(float, diagnostics["rank"]))


def paired_closed_loop_guard_snapshot(
    rows: list[dict[str, Any]],
    *,
    baseline_rows: list[dict[str, Any]],
    expected_modes: Iterable[str],
    lower_reduction_fraction: float,
    upper_reduction_fraction: float,
    lower_power_floor: float,
    upper_power_floor: float,
    reward_noninferiority_margin_fraction: float = 0.02,
    pathwise_robust: bool = False,
    risk_mode: str = "legacy",
    cvar_alpha: float = 0.5,
) -> dict[str, Any]:
    """Reduce actual paired rollouts to the generic actor-guard contract."""

    diagnostics = paired_relative_frequency_feasibility_diagnostics(
        rows,
        baseline_rows=baseline_rows,
        expected_modes=expected_modes,
        lower_reduction_fraction=lower_reduction_fraction,
        upper_reduction_fraction=upper_reduction_fraction,
        lower_power_floor=lower_power_floor,
        upper_power_floor=upper_power_floor,
        reward_noninferiority_margin_fraction=(
            reward_noninferiority_margin_fraction
        ),
        pathwise_robust=pathwise_robust,
        risk_mode=risk_mode,
        cvar_alpha=cvar_alpha,
    )
    violation_tolerance = 1e-10
    reward_violations = sum(
        float(item["normalized_violation"]) > violation_tolerance
        for item in diagnostics["constraints"]
        if str(item["endpoint"]) == "reward_mean"
    )
    frequency_violations = sum(
        float(item["normalized_violation"]) > violation_tolerance
        for item in diagnostics["constraints"]
        if str(item["endpoint"]) != "reward_mean"
    )
    frequency_values = np.asarray([
        float(item["normalized_violation"])
        for item in diagnostics["constraints"]
        if str(item["endpoint"]) != "reward_mean"
    ], dtype=np.float64)
    if frequency_values.size < 1 or not np.all(np.isfinite(frequency_values)):
        raise ValueError(
            "closed-loop guard frequency violation registry is invalid"
        )
    return {
        "contract": (
            (
                "paired_frozen_anchor_actual_closed_loop_mode_cvar_reward_"
                "floor_and_five_frequency_endpoints_with_restoration_"
                f"merit_alpha_{float(cvar_alpha):.6g}_v4"
            )
            if diagnostics["risk_mode"] == "mode_cvar" else (
            (
                "paired_frozen_anchor_actual_closed_loop_pathwise_reward_"
                "floor_and_five_frequency_endpoints_with_restoration_"
                "merit_v3"
            )
            if diagnostics["risk_mode"] == "pathwise_all" else (
                "paired_frozen_anchor_actual_closed_loop_reward_floor_and_"
                "five_frequency_endpoints_with_restoration_merit_v2"
            )
            )
        ),
        "rank": tuple(map(float, diagnostics["rank"])),
        "risk_mode": str(diagnostics["risk_mode"]),
        "cvar_alpha": float(diagnostics["cvar_alpha"]),
        "path_count": len(rows),
        "constraint_count": int(diagnostics["constraint_count"]),
        "reward_violation_count": int(reward_violations),
        "frequency_violation_count": int(frequency_violations),
        "frequency_violation_merit": float(np.sum(
            np.square(frequency_values)
        )),
        "worst_frequency_violation": float(np.max(frequency_values)),
        "worst_constraint": dict(diagnostics["worst_constraint"]),
    }


def train_mujoco_method(
    *,
    method: str,
    env_id: str,
    disturbance_mode: str,
    train_seeds: Iterable[int],
    selection_seeds: Iterable[int],
    eval_seeds: Iterable[int],
    safety_selection_seeds: Iterable[int] | None = None,
    deployment_frequency_closed_loop_guard_seeds: (
        Iterable[int] | None
    ) = None,
    deployment_frequency_anchor_state_replay_seeds: (
        Iterable[int] | None
    ) = None,
    steps: int,
    iterations: int,
    optimizer_seed: int,
    episode_horizon: int = 1000,
    upper_period: int = 16,
    hidden_dim: int = 64,
    learning_rate: float = 3e-4,
    lower_lf_rms_budget: float = 0.05,
    checkpoint_smoothing_window: int = 8,
    checkpoint_min_delta: float = 1e-3,
    checkpoint_minimum_iteration: int = -1,
    checkpoint_evaluation_interval: int = 4,
    training_disturbance_modes: Iterable[str] | None = None,
    evaluation_disturbance_modes: Iterable[str] | None = None,
    upper_action_scale: float = 1.0,
    lower_action_scale: float = 1.0,
    upper_action_decoder_mode: str = "hold",
    upper_promotion_gain: float = 0.0,
    responsibility_mode: str = "additive",
    leakage_constraint_scope: str = "joint_behavior",
    upper_hf_rms_budget: float = DEFAULT_UPPER_HF_RMS_BUDGET,
    upper_hf_penalty_coef: float = DEFAULT_UPPER_HF_PENALTY_COEF,
    upper_constraint_mode: str = "static_reward_penalty",
    upper_dual_lr: float = DEFAULT_UPPER_DUAL_LR,
    lower_dual_lr: float = DEFAULT_LOWER_DUAL_LR,
    upper_deployment_frequency_dual_lr: float = 0.0,
    lower_deployment_frequency_dual_lr: float = 0.0,
    upper_deployment_frequency_lambda_init: float = 0.0,
    lower_deployment_frequency_lambda_init: float = 0.0,
    upper_deployment_frequency_step_scale: float = 1.0,
    lower_deployment_frequency_step_scale: float = 1.0,
    upper_deployment_frequency_max_projection_steps: int = 1,
    lower_deployment_frequency_max_projection_steps: int = 1,
    upper_deployment_frequency_reward_tolerance: float = 1e-8,
    lower_deployment_frequency_reward_tolerance: float = 1e-8,
    upper_deployment_frequency_target_tolerance: float = 0.0,
    lower_deployment_frequency_target_tolerance: float = 0.0,
    upper_deployment_frequency_rms_budget: float = 0.0,
    lower_deployment_frequency_rms_budget: float = 0.0,
    upper_deployment_frequency_reference_reduction_fraction: float = 0.0,
    lower_deployment_frequency_reference_reduction_fraction: float = 0.0,
    deployment_frequency_groupwise_robust: bool = False,
    deployment_frequency_anchor_state_replay: bool = False,
    deployment_frequency_projection_objective: str = "worst_group",
    deployment_frequency_projection_cvar_alpha: float = 0.5,
    deployment_frequency_restoration_freeze_reward_actor: bool = False,
    deployment_frequency_pathwise_robust: bool = False,
    deployment_frequency_closed_loop_risk_mode: str = "legacy",
    deployment_frequency_closed_loop_cvar_alpha: float = 0.5,
    deployment_frequency_ppo_trust_region: bool = False,
    deployment_frequency_ppo_trust_region_backtracks: int = 8,
    deployment_frequency_closed_loop_trust_region: bool = False,
    deployment_frequency_closed_loop_trust_region_backtracks: int = 8,
    deployment_frequency_closed_loop_restoration_filter: bool = False,
    deployment_frequency_closed_loop_restoration_min_reduction: float = 1e-4,
    deployment_frequency_closed_loop_restoration_funnel_multiplier: float = 3.0,
    upper_constraint_update_mode: str = "reward_guarded_adam_projection",
    lower_constraint_update_mode: str = "reward_guarded_adam_projection",
    constraint_dual_normalization: str = "none",
    constraint_dual_scale_ema_beta: float = 0.95,
    constraint_dual_scale_floor: float = 1e-6,
    leakage_cost_mode: str = "ratio_excess_squared",
    lower_action_router_mode: str = "direct",
    lower_action_router_alpha: float = DEFAULT_LOWER_ROUTER_ALPHA,
    lower_action_router_strength: float = DEFAULT_LOWER_ROUTER_STRENGTH,
    lower_action_router_training_schedule: str = "constant",
    lower_action_router_warmup_fraction: float = 0.0,
    lower_action_router_ramp_fraction: float = 0.0,
    lower_action_router_observe_strength: bool = False,
    initial_checkpoint_path: Path | None = None,
    initial_checkpoint_summary_path: Path | None = None,
    initial_checkpoint_router_mode: str = "direct",
    initial_checkpoint_router_strength: float = 0.0,
    upper_actor_anchor_coef: float = 0.0,
    lower_actor_anchor_coef: float = 0.0,
    checkpoint_selection_mode: str = "assigned_condition",
    checkpoint_score_mode: str = "mean_reward",
    checkpoint_constraint_penalty: float = (
        DEFAULT_CHECKPOINT_CONSTRAINT_PENALTY
    ),
    code_revision: str = "",
    expected_source_manifest_sha256: str = "",
    control_protocol_version: str = "auto",
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    name = str(method)
    if name not in METHODS:
        raise ValueError(f"unknown MuJoCo method: {name}")
    if str(upper_action_decoder_mode) not in UPPER_ACTION_DECODER_MODES:
        raise ValueError("unknown MuJoCo upper-action decoder mode")
    if not isinstance(deployment_frequency_closed_loop_trust_region, bool):
        raise ValueError(
            "MuJoCo deployment-frequency closed-loop trust-region flag "
            "must be boolean"
        )
    if not isinstance(
        deployment_frequency_closed_loop_restoration_filter, bool
    ):
        raise ValueError(
            "MuJoCo deployment-frequency closed-loop restoration-filter "
            "flag must be boolean"
        )
    if (
        deployment_frequency_closed_loop_restoration_filter
        and not deployment_frequency_closed_loop_trust_region
    ):
        raise ValueError(
            "MuJoCo closed-loop restoration filtering requires the "
            "closed-loop trust region"
        )
    closed_loop_guard_seed_values = (
        None
        if deployment_frequency_closed_loop_guard_seeds is None
        else tuple(map(int, deployment_frequency_closed_loop_guard_seeds))
    )
    anchor_state_replay_seed_values = (
        None
        if deployment_frequency_anchor_state_replay_seeds is None
        else tuple(map(int, deployment_frequency_anchor_state_replay_seeds))
    )
    roots = validate_unique_seeds(train_seeds, role="mujoco_train_seeds")
    selection, evaluation = validate_evaluation_seed_roles(
        selection_seeds, eval_seeds
    )
    safety_selection = (
        validate_unique_seeds(
            DEFAULT_SAFETY_SELECTION_SEEDS
            if safety_selection_seeds is None else safety_selection_seeds,
            role="mujoco_safety_selection_seeds",
        )
        if name == "freq_hrl_safe_selector" else []
    )
    closed_loop_guard_roots = (
        validate_unique_seeds(
            closed_loop_guard_seed_values or (),
            role="mujoco_deployment_frequency_closed_loop_guard_seeds",
        )
        if deployment_frequency_closed_loop_trust_region else []
    )
    anchor_state_replay_roots = (
        validate_unique_seeds(
            anchor_state_replay_seed_values,
            role="mujoco_deployment_frequency_anchor_state_replay_seeds",
        )
        if (
            deployment_frequency_anchor_state_replay
            and anchor_state_replay_seed_values is not None
        ) else []
    )
    if (
        deployment_frequency_closed_loop_trust_region
        and not closed_loop_guard_roots
    ):
        raise ValueError(
            "closed-loop trust region requires independent guard seeds"
        )
    if (
        not deployment_frequency_closed_loop_trust_region
        and closed_loop_guard_seed_values
    ):
        raise ValueError(
            "closed-loop guard seeds cannot be supplied while the trust "
            "region is disabled"
        )
    if (
        not deployment_frequency_anchor_state_replay
        and anchor_state_replay_seed_values
    ):
        raise ValueError(
            "anchor-state replay seeds cannot be supplied while replay is "
            "disabled"
        )
    if (
        str(deployment_frequency_projection_objective)
        not in DEPLOYMENT_FREQUENCY_PROJECTION_OBJECTIVES
    ):
        raise ValueError(
            "unknown MuJoCo deployment-frequency projection objective"
        )
    resolved_closed_loop_risk_mode = _resolve_closed_loop_risk_mode(
        pathwise_robust=deployment_frequency_pathwise_robust,
        risk_mode=deployment_frequency_closed_loop_risk_mode,
    )
    for label, alpha in (
        ("projection", deployment_frequency_projection_cvar_alpha),
        ("closed-loop", deployment_frequency_closed_loop_cvar_alpha),
    ):
        if not np.isfinite(float(alpha)) or not 0.0 <= float(alpha) < 1.0:
            raise ValueError(f"MuJoCo {label} CVaR alpha must be in [0, 1)")
    if str(constraint_dual_normalization) not in {"none", "ema_abs"}:
        raise ValueError("unknown MuJoCo constraint dual normalization")
    if (
        not np.isfinite(float(constraint_dual_scale_ema_beta))
        or not 0.0 <= float(constraint_dual_scale_ema_beta) < 1.0
    ):
        raise ValueError("MuJoCo constraint dual EMA beta must be in [0, 1)")
    if (
        not np.isfinite(float(constraint_dual_scale_floor))
        or float(constraint_dual_scale_floor) <= 0.0
    ):
        raise ValueError("MuJoCo constraint dual scale floor must be positive")
    for feature_name, enabled in (
        (
            "deployment-frequency restoration reward-actor freeze",
            deployment_frequency_restoration_freeze_reward_actor,
        ),
        (
            "deployment-frequency pathwise robustness",
            deployment_frequency_pathwise_robust,
        ),
    ):
        if not isinstance(enabled, bool):
            raise TypeError(f"MuJoCo {feature_name} flag must be boolean")
    if (
        deployment_frequency_restoration_freeze_reward_actor
        and not deployment_frequency_closed_loop_restoration_filter
    ):
        raise ValueError(
            "MuJoCo restoration reward-actor freeze requires the "
            "restoration filter"
        )
    if (
        deployment_frequency_pathwise_robust
        and not deployment_frequency_groupwise_robust
    ):
        raise ValueError(
            "MuJoCo pathwise frequency robustness requires groupwise "
            "constraints"
        )
    if (
        not np.isfinite(float(upper_promotion_gain))
        or not 0.0 <= float(upper_promotion_gain) <= 1.0
    ):
        raise ValueError("MuJoCo upper promotion gain must be in [0, 1]")
    if (
        str(lower_action_router_mode) == "causal_macro_zero_dc_headroom"
        and str(upper_action_decoder_mode) != "causal_smoothstep_plan"
    ):
        raise ValueError(
            "headroom zero-DC routing requires a frozen smooth upper plan"
        )
    if (
        str(lower_action_router_mode) == "causal_macro_zero_dc_headroom"
        and str(responsibility_mode) != "additive"
    ):
        raise ValueError(
            "headroom zero-DC routing requires additive responsibility"
        )
    if (
        str(lower_action_router_mode) == "causal_smooth_macro_gauge"
        and str(upper_action_decoder_mode) != "causal_smoothstep_plan"
    ):
        raise ValueError(
            "smooth macro gauge routing requires a frozen smooth upper plan"
        )
    if (
        str(lower_action_router_mode) == "causal_smooth_macro_gauge"
        and str(responsibility_mode) != "additive"
    ):
        raise ValueError(
            "smooth macro gauge routing requires additive responsibility"
        )
    if (
        str(lower_action_router_mode) == "causal_audit_optimal_macro_gauge"
        and str(upper_action_decoder_mode) != "causal_smoothstep_plan"
    ):
        raise ValueError(
            "audit-optimal macro gauge routing requires a frozen smooth upper plan"
        )
    if (
        str(lower_action_router_mode) == "causal_audit_optimal_macro_gauge"
        and str(responsibility_mode) != "additive"
    ):
        raise ValueError(
            "audit-optimal macro gauge routing requires additive responsibility"
        )
    if (
        float(upper_promotion_gain) > 0.0
        and str(lower_action_router_mode)
        != "causal_macro_zero_dc_headroom"
    ):
        raise ValueError(
            "upper promotion requires headroom zero-DC lower routing"
        )
    inferred_protocol_version = (
        MUJOCO_CONTROL_PROTOCOL_VERSION_V17_3
        if str(lower_action_router_mode)
        == "causal_audit_optimal_macro_gauge"
        else MUJOCO_CONTROL_PROTOCOL_VERSION_V17_2
        if str(lower_action_router_mode) == "causal_smooth_macro_gauge"
        else
        MUJOCO_CONTROL_PROTOCOL_VERSION_V17_1
        if str(lower_action_router_mode) == "causal_macro_zero_dc_headroom"
        else MUJOCO_CONTROL_PROTOCOL_VERSION_V17
        if (
            str(upper_action_decoder_mode) == "causal_smoothstep_plan"
            or str(lower_action_router_mode) == "causal_macro_zero_dc"
        )
        else MUJOCO_CONTROL_PROTOCOL_VERSION_V16_2
        if str(lower_action_router_mode) == "causal_macro_hold_audit_gauge"
        else MUJOCO_CONTROL_PROTOCOL_VERSION_V14_17
        if (
            str(deployment_frequency_projection_objective)
            == "violation_cvar"
            or resolved_closed_loop_risk_mode == "mode_cvar"
            or str(constraint_dual_normalization) != "none"
        )
        else MUJOCO_CONTROL_PROTOCOL_VERSION_V14_16
        if (
            anchor_state_replay_roots
            or str(deployment_frequency_projection_objective)
            != "worst_group"
            or deployment_frequency_restoration_freeze_reward_actor
            or deployment_frequency_pathwise_robust
        )
        else MUJOCO_CONTROL_PROTOCOL_VERSION
    )
    selected_protocol_version = str(control_protocol_version)
    if selected_protocol_version not in MUJOCO_CONTROL_PROTOCOL_SELECTIONS:
        raise ValueError("unknown MuJoCo control protocol version")
    if (
        inferred_protocol_version == MUJOCO_CONTROL_PROTOCOL_VERSION_V17_3
        and selected_protocol_version
        not in {"auto", MUJOCO_CONTROL_PROTOCOL_VERSION_V17_3}
    ):
        raise ValueError("v17.3 mechanisms cannot use an earlier protocol label")
    if (
        selected_protocol_version == MUJOCO_CONTROL_PROTOCOL_VERSION_V17_3
        and inferred_protocol_version != MUJOCO_CONTROL_PROTOCOL_VERSION_V17_3
    ):
        raise ValueError(
            "the v17.3 protocol label requires audit-optimal macro gauge routing"
        )
    if (
        inferred_protocol_version == MUJOCO_CONTROL_PROTOCOL_VERSION_V17_2
        and selected_protocol_version
        not in {"auto", MUJOCO_CONTROL_PROTOCOL_VERSION_V17_2}
    ):
        raise ValueError("v17.2 mechanisms cannot use an earlier protocol label")
    if (
        selected_protocol_version == MUJOCO_CONTROL_PROTOCOL_VERSION_V17_2
        and inferred_protocol_version != MUJOCO_CONTROL_PROTOCOL_VERSION_V17_2
    ):
        raise ValueError(
            "the v17.2 protocol label requires smooth macro gauge routing"
        )
    if (
        inferred_protocol_version == MUJOCO_CONTROL_PROTOCOL_VERSION_V17_1
        and selected_protocol_version
        not in {"auto", MUJOCO_CONTROL_PROTOCOL_VERSION_V17_1}
    ):
        raise ValueError("v17.1 mechanisms cannot use an earlier protocol label")
    if (
        selected_protocol_version == MUJOCO_CONTROL_PROTOCOL_VERSION_V17_1
        and inferred_protocol_version != MUJOCO_CONTROL_PROTOCOL_VERSION_V17_1
    ):
        raise ValueError(
            "the v17.1 protocol label requires headroom zero-DC routing"
        )
    if (
        inferred_protocol_version == MUJOCO_CONTROL_PROTOCOL_VERSION_V17
        and selected_protocol_version
        not in {"auto", MUJOCO_CONTROL_PROTOCOL_VERSION_V17}
    ):
        raise ValueError("v17 mechanisms cannot use an earlier protocol label")
    if (
        inferred_protocol_version == MUJOCO_CONTROL_PROTOCOL_VERSION_V16_2
        and selected_protocol_version
        not in {"auto", MUJOCO_CONTROL_PROTOCOL_VERSION_V16_2}
    ):
        raise ValueError("v16.2 mechanisms cannot use an earlier protocol label")
    if (
        inferred_protocol_version in {
            MUJOCO_CONTROL_PROTOCOL_VERSION_V14_16,
            MUJOCO_CONTROL_PROTOCOL_VERSION_V14_17,
        }
        and selected_protocol_version == MUJOCO_CONTROL_PROTOCOL_VERSION
    ):
        raise ValueError(
            "v14.16 restoration mechanisms cannot be labeled as v14.15"
        )
    if (
        inferred_protocol_version == MUJOCO_CONTROL_PROTOCOL_VERSION_V14_17
        and selected_protocol_version == MUJOCO_CONTROL_PROTOCOL_VERSION_V14_16
    ):
        raise ValueError("v14.17 mechanisms cannot be labeled as v14.16")
    effective_protocol_version = (
        inferred_protocol_version
        if selected_protocol_version == "auto"
        else selected_protocol_version
    )
    seed_roles = {
        "training": set(roots),
        "checkpoint_selection": set(selection),
        "safety_selection": set(safety_selection),
        "anchor_state_replay": set(anchor_state_replay_roots),
        "closed_loop_guard": set(closed_loop_guard_roots),
        "heldout_test": set(evaluation),
    }
    role_names = list(seed_roles)
    for index, left in enumerate(role_names):
        for right in role_names[index + 1:]:
            overlap = sorted(seed_roles[left] & seed_roles[right])
            if overlap:
                raise ValueError(
                    f"MuJoCo seed roles {left} and {right} overlap: {overlap}"
                )
    if int(upper_period) < 2:
        raise ValueError("MuJoCo upper_period must be at least two")
    if (
        isinstance(checkpoint_minimum_iteration, bool)
        or int(checkpoint_minimum_iteration) < -1
        or int(checkpoint_minimum_iteration) >= max(1, int(iterations))
    ):
        raise ValueError(
            "MuJoCo checkpoint minimum iteration must be in [-1, iterations)"
        )
    source_identity = verify_current_freq_hrl_source_identity(
        code_revision=str(code_revision),
        expected_source_manifest_sha256=str(expected_source_manifest_sha256),
    )
    if int(steps) < 1 or int(episode_horizon) < 1:
        raise ValueError("MuJoCo steps and episode_horizon must be positive")
    if not 0.0 <= float(upper_action_scale) <= 1.0:
        raise ValueError("MuJoCo upper_action_scale must be in [0, 1]")
    if not 0.0 < float(lower_action_scale) <= 1.0:
        raise ValueError("MuJoCo lower_action_scale must be in (0, 1]")
    if str(responsibility_mode) not in RESPONSIBILITY_MODES:
        raise ValueError("unknown MuJoCo responsibility mode")
    if str(leakage_constraint_scope) not in LEAKAGE_CONSTRAINT_SCOPES:
        raise ValueError("unknown MuJoCo leakage constraint scope")
    if (
        not np.isfinite(float(upper_hf_rms_budget))
        or float(upper_hf_rms_budget) <= 0.0
    ):
        raise ValueError("MuJoCo upper-HF RMS budget must be positive")
    if (
        not np.isfinite(float(upper_hf_penalty_coef))
        or float(upper_hf_penalty_coef) < 0.0
    ):
        raise ValueError("MuJoCo upper-HF penalty must be non-negative")
    if str(upper_constraint_mode) not in UPPER_CONSTRAINT_MODES:
        raise ValueError("unknown MuJoCo upper constraint mode")
    if (
        not np.isfinite(float(upper_dual_lr))
        or float(upper_dual_lr) < 0.0
    ):
        raise ValueError("MuJoCo upper dual learning rate must be non-negative")
    if (
        not np.isfinite(float(lower_dual_lr))
        or float(lower_dual_lr) < 0.0
    ):
        raise ValueError("MuJoCo lower dual learning rate must be non-negative")
    deployment_values = {
        "upper dual learning rate": upper_deployment_frequency_dual_lr,
        "lower dual learning rate": lower_deployment_frequency_dual_lr,
        "upper lambda init": upper_deployment_frequency_lambda_init,
        "lower lambda init": lower_deployment_frequency_lambda_init,
    }
    for label, value in deployment_values.items():
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(
                f"MuJoCo deployment-frequency {label} must be non-negative"
            )
    for label, value in (
        ("upper step scale", upper_deployment_frequency_step_scale),
        ("lower step scale", lower_deployment_frequency_step_scale),
    ):
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(
                f"MuJoCo deployment-frequency {label} must be positive"
            )
    for label, value in (
        (
            "upper maximum projection steps",
            upper_deployment_frequency_max_projection_steps,
        ),
        (
            "lower maximum projection steps",
            lower_deployment_frequency_max_projection_steps,
        ),
    ):
        if (
            isinstance(value, bool)
            or int(value) != value
            or int(value) < 1
        ):
            raise ValueError(
                f"MuJoCo deployment-frequency {label} must be a positive integer"
            )
    for label, value in (
        (
            "upper reward tolerance",
            upper_deployment_frequency_reward_tolerance,
        ),
        (
            "lower reward tolerance",
            lower_deployment_frequency_reward_tolerance,
        ),
        (
            "upper target tolerance",
            upper_deployment_frequency_target_tolerance,
        ),
        (
            "lower target tolerance",
            lower_deployment_frequency_target_tolerance,
        ),
    ):
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(
                f"MuJoCo deployment-frequency {label} must be non-negative"
            )
    deployment_frequency_level_active = {
        "upper": (
            float(upper_deployment_frequency_dual_lr) > 0.0
            or float(upper_deployment_frequency_lambda_init) > 0.0
        ),
        "lower": (
            float(lower_deployment_frequency_dual_lr) > 0.0
            or float(lower_deployment_frequency_lambda_init) > 0.0
        ),
    }
    deployment_frequency_reference_reductions = {
        "upper": float(
            upper_deployment_frequency_reference_reduction_fraction
        ),
        "lower": float(
            lower_deployment_frequency_reference_reduction_fraction
        ),
    }
    for level, fraction in deployment_frequency_reference_reductions.items():
        if not np.isfinite(fraction) or not 0.0 <= fraction < 1.0:
            raise ValueError(
                f"MuJoCo deployment-frequency {level} reference reduction "
                "must be in [0, 1)"
            )
        if fraction > 0.0 and not deployment_frequency_level_active[level]:
            raise ValueError(
                f"MuJoCo deployment-frequency {level} reference reduction "
                "requires an active level constraint"
            )
    deployment_frequency_requested = any(
        deployment_frequency_level_active.values()
    )
    if deployment_frequency_requested and name != "freq_hrl":
        raise ValueError(
            "deployment-frequency constraints require method=freq_hrl"
        )
    if not isinstance(deployment_frequency_groupwise_robust, bool):
        raise ValueError(
            "MuJoCo deployment-frequency groupwise robust flag must be boolean"
        )
    if (
        deployment_frequency_groupwise_robust
        and not deployment_frequency_requested
    ):
        raise ValueError(
            "groupwise robust deployment frequency requires an active constraint"
        )
    for feature_name, enabled in (
        (
            "deployment-frequency anchor-state replay",
            deployment_frequency_anchor_state_replay,
        ),
        (
            "deployment-frequency PPO trust region",
            deployment_frequency_ppo_trust_region,
        ),
        (
            "deployment-frequency closed-loop trust region",
            deployment_frequency_closed_loop_trust_region,
        ),
        ):
        if not isinstance(enabled, bool):
            raise ValueError(
                f"MuJoCo {feature_name} flag must be boolean"
            )
        if enabled and not deployment_frequency_groupwise_robust:
            raise ValueError(
                f"MuJoCo {feature_name} requires groupwise robust constraints"
            )
    if (
        isinstance(deployment_frequency_ppo_trust_region_backtracks, bool)
        or int(deployment_frequency_ppo_trust_region_backtracks)
        != deployment_frequency_ppo_trust_region_backtracks
        or int(deployment_frequency_ppo_trust_region_backtracks) < 1
    ):
        raise ValueError(
            "MuJoCo deployment-frequency PPO trust-region backtracks must "
            "be a positive integer"
        )
    if (
        isinstance(
            deployment_frequency_closed_loop_trust_region_backtracks, bool
        )
        or int(deployment_frequency_closed_loop_trust_region_backtracks)
        != deployment_frequency_closed_loop_trust_region_backtracks
        or int(deployment_frequency_closed_loop_trust_region_backtracks) < 1
    ):
        raise ValueError(
            "MuJoCo deployment-frequency closed-loop trust-region "
            "backtracks must be a positive integer"
        )
    restoration_min_reduction = float(
        deployment_frequency_closed_loop_restoration_min_reduction
    )
    if (
        not np.isfinite(restoration_min_reduction)
        or not 0.0 < restoration_min_reduction < 1.0
    ):
        raise ValueError(
            "MuJoCo closed-loop restoration minimum reduction must be "
            "finite and in (0, 1)"
        )
    restoration_funnel_multiplier = float(
        deployment_frequency_closed_loop_restoration_funnel_multiplier
    )
    if (
        not np.isfinite(restoration_funnel_multiplier)
        or restoration_funnel_multiplier < 1.0
    ):
        raise ValueError(
            "MuJoCo closed-loop restoration funnel multiplier must be "
            "finite and at least one"
        )
    for level, value in (
        ("upper", upper_deployment_frequency_rms_budget),
        ("lower", lower_deployment_frequency_rms_budget),
    ):
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(
                f"MuJoCo deployment-frequency {level} RMS budget must be "
                "non-negative"
            )
        if deployment_frequency_level_active[level] and float(value) <= 0.0:
            raise ValueError(
                f"an active MuJoCo deployment-frequency {level} RMS budget "
                "must be positive"
            )
    for level, coefficient in (
        ("upper", upper_actor_anchor_coef),
        ("lower", lower_actor_anchor_coef),
    ):
        if not np.isfinite(float(coefficient)) or float(coefficient) < 0.0:
            raise ValueError(
                f"MuJoCo {level} actor anchor coefficient must be non-negative"
            )
    checkpoint_paths_present = (
        initial_checkpoint_path is not None,
        initial_checkpoint_summary_path is not None,
    )
    if checkpoint_paths_present[0] != checkpoint_paths_present[1]:
        raise ValueError(
            "paired continuation requires both checkpoint and summary paths"
        )
    paired_continuation = all(checkpoint_paths_present)
    if (
        deployment_frequency_anchor_state_replay
        or deployment_frequency_ppo_trust_region
        or deployment_frequency_closed_loop_trust_region
    ) and not paired_continuation:
        raise ValueError(
            "MuJoCo deployment-frequency replay and trust regions require "
            "a paired frozen-anchor continuation"
        )
    if (
        any(
            value > 0.0
            for value in deployment_frequency_reference_reductions.values()
        )
        and not paired_continuation
    ):
        raise ValueError(
            "a relative deployment-frequency target requires a paired "
            "checkpoint"
        )
    if (
        float(upper_actor_anchor_coef) > 0.0
        or float(lower_actor_anchor_coef) > 0.0
    ) and not paired_continuation:
        raise ValueError("an actor anchor requires a paired checkpoint")
    if paired_continuation and name not in {
        "freq_hrl",
        "freq_hrl_no_leakage",
    }:
        raise ValueError(
            "paired MuJoCo continuation requires a Freq-HRL policy"
        )
    if str(leakage_cost_mode) not in LEAKAGE_COST_MODES:
        raise ValueError("unknown MuJoCo leakage constraint cost mode")
    if str(lower_action_router_mode) not in LOWER_ACTION_ROUTER_MODES:
        raise ValueError("unknown MuJoCo lower-action router mode")
    if str(initial_checkpoint_router_mode) not in LOWER_ACTION_ROUTER_MODES:
        raise ValueError("unknown initial-checkpoint lower-action router mode")
    if (
        not np.isfinite(float(initial_checkpoint_router_strength))
        or not 0.0 <= float(initial_checkpoint_router_strength) <= 1.0
    ):
        raise ValueError(
            "initial-checkpoint lower-action router strength must be in [0, 1]"
        )
    if not 0.0 < float(lower_action_router_alpha) <= 1.0:
        raise ValueError("MuJoCo lower-action router alpha must be in (0, 1]")
    if not 0.0 <= float(lower_action_router_strength) <= 1.0:
        raise ValueError(
            "MuJoCo lower-action router strength must be in [0, 1]"
        )
    if (
        str(lower_action_router_training_schedule)
        not in LOWER_ACTION_ROUTER_TRAINING_SCHEDULES
    ):
        raise ValueError("unknown lower-action router training schedule")
    if str(lower_action_router_mode) == "causal_macro_zero_dc" and (
        float(lower_action_router_strength) != 1.0
        or str(lower_action_router_training_schedule) != "constant"
        or float(lower_action_router_warmup_fraction) != 0.0
        or float(lower_action_router_ramp_fraction) != 0.0
    ):
        raise ValueError(
            "zero-DC lower routing requires a full-strength constant schedule"
        )
    if (
        str(lower_action_router_mode) == "causal_macro_zero_dc_headroom"
        and float(lower_action_router_strength) != 1.0
    ):
        raise ValueError(
            "headroom zero-DC training requires a full-strength target"
        )
    if (
        str(upper_action_decoder_mode) != "hold"
        and not name.startswith("freq_hrl")
    ):
        raise ValueError("the smooth upper plan requires a Freq-HRL policy")
    if (
        not np.isfinite(float(lower_action_router_warmup_fraction))
        or not np.isfinite(float(lower_action_router_ramp_fraction))
        or not 0.0 <= float(lower_action_router_warmup_fraction) <= 1.0
        or not 0.0 <= float(lower_action_router_ramp_fraction) <= 1.0
        or (
            float(lower_action_router_warmup_fraction)
            + float(lower_action_router_ramp_fraction)
            > 1.0
        )
    ):
        raise ValueError("invalid lower-action router schedule fractions")
    if str(checkpoint_selection_mode) not in CHECKPOINT_SELECTION_MODES:
        raise ValueError("unknown MuJoCo checkpoint selection mode")
    if str(checkpoint_score_mode) not in CHECKPOINT_SCORE_MODES:
        raise ValueError("unknown MuJoCo checkpoint score mode")
    if (
        name == "flat_ppo"
        and str(checkpoint_score_mode) in {
            "latent_behavior_feasibility_first",
            "paired_relative_frequency_feasibility_first",
        }
    ):
        raise ValueError(
            "latent behavior feasibility ranking requires hierarchical metrics"
        )
    if (
        str(checkpoint_score_mode)
        == "paired_relative_frequency_feasibility_first"
        and not paired_continuation
    ):
        raise ValueError(
            "paired-relative checkpoint ranking requires a paired continuation"
        )
    if (
        not np.isfinite(float(checkpoint_constraint_penalty))
        or float(checkpoint_constraint_penalty) < 0.0
    ):
        raise ValueError(
            "MuJoCo checkpoint constraint penalty must be non-negative"
        )
    if name == "flat_ppo" and str(responsibility_mode) != "additive":
        raise ValueError("flat PPO cannot use hierarchical responsibility transfer")
    if str(lower_constraint_update_mode) not in {
        "scalarized",
        "reward_guarded_projection",
        "reward_guarded_adam_projection",
    }:
        raise ValueError("unknown lower constraint update mode")
    if str(upper_constraint_update_mode) not in {
        "scalarized",
        "reward_guarded_projection",
        "reward_guarded_adam_projection",
    }:
        raise ValueError("unknown upper constraint update mode")
    observation_dim, action_dim = environment_dimensions(
        env_id,
        episode_horizon=episode_horizon,
    )
    effective_lower_action_router_mode = (
        str(lower_action_router_mode)
        if name.startswith("freq_hrl") else "direct"
    )
    effective_lower_action_router_strength = (
        float(lower_action_router_strength)
        if effective_lower_action_router_mode in CAUSAL_LOWER_ACTION_ROUTER_MODES
        else 0.0
    )
    selected_upper_deployment_frequency_dual_lr = (
        float(upper_deployment_frequency_dual_lr)
        if name == "freq_hrl" else 0.0
    )
    selected_lower_deployment_frequency_dual_lr = (
        float(lower_deployment_frequency_dual_lr)
        if name == "freq_hrl" else 0.0
    )
    selected_upper_deployment_frequency_lambda_init = (
        float(upper_deployment_frequency_lambda_init)
        if name == "freq_hrl" else 0.0
    )
    selected_lower_deployment_frequency_lambda_init = (
        float(lower_deployment_frequency_lambda_init)
        if name == "freq_hrl" else 0.0
    )
    selected_upper_deployment_frequency_reference_reduction = (
        float(upper_deployment_frequency_reference_reduction_fraction)
        if name == "freq_hrl" else 0.0
    )
    selected_lower_deployment_frequency_reference_reduction = (
        float(lower_deployment_frequency_reference_reduction_fraction)
        if name == "freq_hrl" else 0.0
    )
    effective_router_training_schedule = (
        str(lower_action_router_training_schedule)
        if effective_lower_action_router_mode in CAUSAL_LOWER_ACTION_ROUTER_MODES
        else "constant"
    )
    effective_router_observe_strength = bool(
        lower_action_router_observe_strength
    )
    if (
        effective_lower_action_router_mode not in CAUSAL_LOWER_ACTION_ROUTER_MODES
        and str(lower_action_router_training_schedule) != "constant"
    ):
        raise ValueError("a direct lower-action router cannot use a curriculum")
    if (
        effective_router_training_schedule != "constant"
        and not effective_router_observe_strength
        and effective_lower_action_router_mode
        not in {
            "causal_smooth_macro_gauge",
            "causal_audit_optimal_macro_gauge",
        }
    ):
        raise ValueError(
            "a router curriculum must expose its strength in policy state"
        )
    if name == "flat_ppo" and effective_router_observe_strength:
        raise ValueError("flat PPO cannot observe hierarchical router strength")
    if name == "flat_ppo" and int(checkpoint_minimum_iteration) != -1:
        raise ValueError(
            "flat PPO does not implement checkpoint minimum iteration"
        )
    if (
        paired_continuation
        and not effective_router_observe_strength
        and effective_lower_action_router_mode
        not in FUNCTION_PRESERVING_LOWER_ACTION_ROUTER_MODES
    ):
        raise ValueError(
            "non-conservative paired continuation requires an observed "
            "router-strength context"
        )
    router_training_strengths_by_iteration = [
        lower_action_router_training_strength(
            iteration=iteration,
            total_iterations=max(1, int(iterations)),
            target_strength=effective_lower_action_router_strength,
            schedule=effective_router_training_schedule,
            warmup_fraction=lower_action_router_warmup_fraction,
            ramp_fraction=lower_action_router_ramp_fraction,
        )
        for iteration in range(max(1, int(iterations)))
    ]
    state_dim = mujoco_policy_state_dim(
        observation_dim,
        action_dim,
        observe_router_strength=effective_router_observe_strength,
        lower_action_router_mode=effective_lower_action_router_mode,
    )
    actor_anchor_zero_state_indices = (
        (int(state_dim) - 1,)
        if paired_continuation and effective_router_observe_strength
        else ()
    )
    torch.manual_seed(int(optimizer_seed))
    np.random.seed(int(optimizer_seed))

    training_modes = _validated_disturbance_modes(
        [str(disturbance_mode)]
        if training_disturbance_modes is None
        else training_disturbance_modes,
        role="training",
    )
    if len(training_modes) > 1 and (
        len(roots) < len(training_modes)
        or (
            str(checkpoint_selection_mode) == "assigned_condition"
            and len(selection) < len(training_modes)
        )
    ):
        raise ValueError(
            "multi-condition MuJoCo training requires at least one train and "
            "assigned selection seed per disturbance mode"
        )
    domain_seed_key = f"mujoco:{env_id}:multi_condition_v1"
    seed_modes: dict[int, str] = {}
    train_root_modes = _assign_seed_modes(roots, training_modes)
    if str(checkpoint_selection_mode) == "crossed_conditions":
        selection_rollout_seeds, selection_seed_modes = (
            crossed_checkpoint_selection_paths(
                selection,
                training_modes,
                env_id=env_id,
            )
        )
    else:
        selection_rollout_seeds = list(selection)
        selection_seed_modes = _assign_seed_modes(
            selection_rollout_seeds, training_modes
        )
    if deployment_frequency_closed_loop_trust_region:
        (
            closed_loop_guard_rollout_seeds,
            closed_loop_guard_seed_modes,
        ) = crossed_deployment_frequency_guard_paths(
            closed_loop_guard_roots,
            training_modes,
            env_id=env_id,
        )
    else:
        closed_loop_guard_rollout_seeds = []
        closed_loop_guard_seed_modes = {}
    if anchor_state_replay_roots:
        (
            anchor_state_replay_rollout_seeds,
            anchor_state_replay_seed_modes,
        ) = crossed_deployment_frequency_reference_paths(
            anchor_state_replay_roots,
            training_modes,
            env_id=env_id,
        )
    else:
        anchor_state_replay_rollout_seeds = []
        anchor_state_replay_seed_modes = {}
    evaluation_seed_modes = _assign_seed_modes(evaluation, training_modes)

    def register_seed_mode(seed: int, mode: str) -> None:
        previous = seed_modes.get(int(seed))
        if previous is not None and previous != str(mode):
            raise ValueError(
                f"MuJoCo seed {int(seed)} maps to conflicting conditions"
            )
        seed_modes[int(seed)] = str(mode)

    derived_training_seeds: set[int] = set()
    training_router_strength_by_seed: dict[int, float] = {}
    for iteration in range(max(1, int(iterations))):
        for root in roots:
            derived = training_rollout_seed(
                int(optimizer_seed), root, iteration, domain=domain_seed_key
            )
            derived_training_seeds.add(int(derived))
            training_router_strength_by_seed[int(derived)] = float(
                router_training_strengths_by_iteration[iteration]
            )
            register_seed_mode(derived, train_root_modes[int(root)])
    for seed, mode in selection_seed_modes.items():
        register_seed_mode(seed, mode)
    for seed, mode in closed_loop_guard_seed_modes.items():
        register_seed_mode(seed, mode)
    for seed, mode in anchor_state_replay_seed_modes.items():
        register_seed_mode(seed, mode)
    for seed, mode in evaluation_seed_modes.items():
        register_seed_mode(seed, mode)
    protected_seed_roles = (
        set(roots)
        | set(selection)
        | set(safety_selection)
        | set(closed_loop_guard_roots)
        | set(anchor_state_replay_roots)
        | set(evaluation)
    )
    crossed_collisions = sorted(
        set(selection_rollout_seeds)
        & (protected_seed_roles | derived_training_seeds)
    )
    if (
        str(checkpoint_selection_mode) == "crossed_conditions"
        and crossed_collisions
    ):
        raise RuntimeError(
            "crossed checkpoint paths overlap a seed-role root: "
            f"{crossed_collisions}"
        )
    closed_loop_guard_collisions = sorted(
        set(closed_loop_guard_rollout_seeds)
        & (
            protected_seed_roles
            | derived_training_seeds
            | set(selection_rollout_seeds)
        )
    )
    if closed_loop_guard_collisions:
        raise RuntimeError(
            "crossed closed-loop guard paths overlap another seed role: "
            f"{closed_loop_guard_collisions}"
        )
    anchor_state_replay_collisions = sorted(
        set(anchor_state_replay_rollout_seeds)
        & (
            protected_seed_roles
            | derived_training_seeds
            | set(selection_rollout_seeds)
            | set(closed_loop_guard_rollout_seeds)
        )
    )
    if anchor_state_replay_collisions:
        raise RuntimeError(
            "crossed anchor-state replay paths overlap another seed role: "
            f"{anchor_state_replay_collisions}"
        )

    def assigned_mode(seed: int) -> str:
        try:
            return seed_modes[int(seed)]
        except KeyError as exc:
            raise KeyError(
                f"MuJoCo rollout seed {int(seed)} has no registered condition"
            ) from exc

    def router_strength_for_seed(seed: int) -> float:
        return float(training_router_strength_by_seed.get(
            int(seed), effective_lower_action_router_strength
        ))

    checkpoint_score_fn = None
    checkpoint_rank_fn = None
    checkpoint_diagnostics_fn = None
    checkpoint_rank_names: tuple[str, ...] = ()
    checkpoint_rank_contract = "disabled"
    checkpoint_score_contract = "mean_reward_mean_v1"
    paired_relative_baseline_rows: list[dict[str, Any]] = []
    paired_relative_baseline_parameter_sha256 = ""
    closed_loop_guard_baseline_rows: list[dict[str, Any]] = []
    closed_loop_guard_baseline_parameter_sha256 = ""
    if str(checkpoint_score_mode) in {
        "behavior_robust",
        "latent_behavior_robust",
        "latent_behavior_feasibility_first",
    }:
        include_latent_checkpoint_cost = (
            str(checkpoint_score_mode) in {
                "latent_behavior_robust",
                "latent_behavior_feasibility_first",
            }
        )
        checkpoint_score_fn = lambda rows: float(
            behavior_robust_checkpoint_diagnostics(
                rows,
                expected_modes=training_modes,
                lower_lf_rms_budget=lower_lf_rms_budget,
                upper_hf_rms_budget=upper_hf_rms_budget,
                constraint_penalty=checkpoint_constraint_penalty,
                include_latent=include_latent_checkpoint_cost,
            )["score"]
        )
        checkpoint_score_contract = (
            "worst_condition_reward_minus_scale_normalized_squared_"
            + (
                "lower_raw_lower_latent_lower_upper_and_latent_upper_"
                "budget_violations_v2"
                if include_latent_checkpoint_cost
                else "lower_raw_lower_and_upper_hf_budget_violations_v1"
            )
        )
        if str(checkpoint_score_mode) == "latent_behavior_feasibility_first":
            checkpoint_rank_fn = lambda rows: latent_behavior_feasibility_rank(
                rows,
                expected_modes=training_modes,
                lower_lf_rms_budget=lower_lf_rms_budget,
                upper_hf_rms_budget=upper_hf_rms_budget,
            )
            checkpoint_rank_names = (
                "negative_worst_endpoint_violation",
                "negative_endpoint_violation_l2",
                "worst_condition_reward_mean",
            )
            checkpoint_rank_contract = (
                "state_aligned_minimax_latent_behavior_feasibility_then_"
                "worst_condition_reward_v1"
            )

    upper_cost_critic_for_capacity = (
        str(upper_constraint_mode) == "primal_dual"
    )
    reference = _hierarchical_model(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        leakage_constraint=True,
        upper_cost_critic=upper_cost_critic_for_capacity,
        upper_constraint=False,
        upper_dual_lr=upper_dual_lr,
        lower_dual_lr=lower_dual_lr,
        upper_constraint_update_mode=upper_constraint_update_mode,
        lower_constraint_update_mode=lower_constraint_update_mode,
        constraint_dual_normalization=constraint_dual_normalization,
        constraint_dual_scale_ema_beta=constraint_dual_scale_ema_beta,
        constraint_dual_scale_floor=constraint_dual_scale_floor,
    )
    target_parameters = _module_parameter_count(reference)
    selected_leakage_constraint = name == "freq_hrl"
    selected_constraint_update_mode = str(lower_constraint_update_mode)
    selected_constraint_scope = (
        str(leakage_constraint_scope)
        if selected_leakage_constraint else "disabled"
    )
    selected_upper_hf_penalty_coef = (
        float(upper_hf_penalty_coef)
        if (
            selected_leakage_constraint
            and str(upper_constraint_mode) == "static_reward_penalty"
        ) else 0.0
    )
    selected_upper_constraint_mode = (
        str(upper_constraint_mode)
        if selected_leakage_constraint else "disabled"
    )
    paired_checkpoint_metadata: dict[str, Any] = {"enabled": False}
    if name == "freq_hrl_safe_selector":
        branch_specs = {
            SAFE_SELECTOR_BASELINE_BRANCH: {
                "leakage_constraint": False,
                "constraint_update_mode": "reward_guarded_adam_projection",
                "constraint_scope": "responsibility",
                "upper_hf_penalty_coef": 0.0,
                "upper_constraint_mode": "disabled",
            },
            "responsibility_guarded_adam_projection": {
                "leakage_constraint": True,
                "constraint_update_mode": "reward_guarded_adam_projection",
                "constraint_scope": "responsibility",
                "upper_hf_penalty_coef": 0.0,
                "upper_constraint_mode": "disabled",
            },
            "behavior_guarded_adam_projection": {
                "leakage_constraint": True,
                "constraint_update_mode": "reward_guarded_adam_projection",
                "constraint_scope": "joint_behavior",
                "upper_hf_penalty_coef": 0.0,
                "upper_constraint_mode": "disabled",
            },
            "behavior_guarded_upper_hf": {
                "leakage_constraint": True,
                "constraint_update_mode": "reward_guarded_adam_projection",
                "constraint_scope": "joint_behavior",
                "upper_hf_penalty_coef": float(
                    upper_hf_penalty_coef
                    if str(upper_constraint_mode) == "static_reward_penalty"
                    else 0.0
                ),
                "upper_constraint_mode": str(upper_constraint_mode),
            },
            "behavior_scalarized_upper_hf": {
                "leakage_constraint": True,
                "constraint_update_mode": "scalarized",
                "constraint_scope": "joint_behavior",
                "upper_hf_penalty_coef": float(
                    upper_hf_penalty_coef
                    if str(upper_constraint_mode) == "static_reward_penalty"
                    else 0.0
                ),
                "upper_constraint_mode": str(upper_constraint_mode),
            },
        }
        branch_models: dict[str, FrequencySeparatedActorCriticPPO] = {}
        branch_payloads: dict[str, dict[str, Any]] = {}
        branch_selection_rows: dict[str, list[dict[str, Any]]] = {}
        branch_training_summaries: dict[str, dict[str, Any]] = {}
        initial_hashes: set[str] = set()
        for branch in SAFE_SELECTOR_BRANCHES:
            spec = branch_specs[branch]
            branch_leakage = bool(spec["leakage_constraint"])
            branch_update_mode = str(spec["constraint_update_mode"])
            branch_constraint_scope = str(spec["constraint_scope"])
            branch_upper_penalty_coef = float(
                spec["upper_hf_penalty_coef"]
            )
            branch_upper_constraint_mode = str(
                spec["upper_constraint_mode"]
            )
            torch.manual_seed(int(optimizer_seed))
            np.random.seed(int(optimizer_seed))
            branch_model = _hierarchical_model(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                leakage_constraint=branch_leakage,
                upper_cost_critic=upper_cost_critic_for_capacity,
                upper_constraint=(
                    branch_upper_constraint_mode == "primal_dual"
                ),
                upper_dual_lr=upper_dual_lr,
                lower_dual_lr=lower_dual_lr,
                upper_constraint_update_mode=upper_constraint_update_mode,
                lower_constraint_update_mode=branch_update_mode,
                constraint_dual_normalization=constraint_dual_normalization,
                constraint_dual_scale_ema_beta=constraint_dual_scale_ema_beta,
                constraint_dual_scale_floor=constraint_dual_scale_floor,
            )
            initial_hash = _model_parameter_sha256(branch_model)
            initial_hashes.add(initial_hash)
            def branch_rollout(
                policy: FrequencySeparatedActorCriticPPO,
                seed: int,
                sample: bool,
                *,
                leakage: bool = branch_leakage,
                scope: str = branch_constraint_scope,
                upper_penalty: float = branch_upper_penalty_coef,
                upper_mode: str = branch_upper_constraint_mode,
            ) -> tuple[HierarchicalTrajectoryBatch | None, dict[str, Any]]:
                return rollout_hierarchical(
                    policy,
                    seed=seed,
                    env_id=env_id,
                    disturbance_mode=assigned_mode(seed),
                    steps=steps,
                    upper_period=upper_period,
                    frequency_routing=True,
                    leakage_constraint=leakage,
                    lower_lf_rms_budget=lower_lf_rms_budget,
                    leakage_constraint_scope=scope,
                    upper_hf_rms_budget=upper_hf_rms_budget,
                    upper_hf_penalty_coef=upper_penalty,
                    upper_constraint_mode=upper_mode,
                    lower_lf_alpha=RESPONSIBILITY_TRANSFER_ALPHA,
                    upper_action_scale=upper_action_scale,
                    lower_action_scale=lower_action_scale,
                    responsibility_mode=responsibility_mode,
                    leakage_cost_mode=leakage_cost_mode,
                    lower_action_router_mode=effective_lower_action_router_mode,
                    lower_action_router_alpha=lower_action_router_alpha,
                    lower_action_router_strength=(
                        router_strength_for_seed(seed)
                    ),
                    lower_action_router_observe_strength=(
                        effective_router_observe_strength
                    ),
                    upper_action_decoder_mode=upper_action_decoder_mode,
                    upper_promotion_gain=upper_promotion_gain,
                    sample=sample,
                    method=name,
                    episode_horizon=episode_horizon,
                )
            branch_payload, _, branch_model = train_frequency_separated_ppo(
                model=branch_model,
                train_seeds=roots,
                selection_seeds=selection_rollout_seeds,
                eval_seeds=[],
                iterations=iterations,
                training_seed_fn=lambda root, iteration: training_rollout_seed(
                    int(optimizer_seed), root, iteration,
                    domain=domain_seed_key,
                ),
                rollout_fn=branch_rollout,
                objective_fn=lambda row: float(row["reward_mean"]),
                summary_fn=summarize,
                policy=f"{name}:{branch}",
                domain="mujoco",
                checkpoint_smoothing_window=checkpoint_smoothing_window,
                checkpoint_min_delta=checkpoint_min_delta,
                checkpoint_minimum_iteration=checkpoint_minimum_iteration,
                checkpoint_evaluation_interval=checkpoint_evaluation_interval,
                checkpoint_score_fn=checkpoint_score_fn,
                checkpoint_score_contract=checkpoint_score_contract,
                checkpoint_rank_fn=checkpoint_rank_fn,
                checkpoint_rank_names=checkpoint_rank_names,
                checkpoint_rank_contract=checkpoint_rank_contract,
            )
            safety_rows: list[dict[str, Any]] = []
            for safety_mode in training_modes:
                for safety_seed in safety_selection:
                    row = rollout_hierarchical(
                        branch_model,
                        seed=int(safety_seed),
                        env_id=env_id,
                        disturbance_mode=safety_mode,
                        steps=steps,
                        upper_period=upper_period,
                        frequency_routing=True,
                        leakage_constraint=branch_leakage,
                        lower_lf_rms_budget=lower_lf_rms_budget,
                        leakage_constraint_scope=branch_constraint_scope,
                        upper_hf_rms_budget=upper_hf_rms_budget,
                        upper_hf_penalty_coef=(
                            branch_upper_penalty_coef
                        ),
                        upper_constraint_mode=(
                            branch_upper_constraint_mode
                        ),
                        lower_lf_alpha=RESPONSIBILITY_TRANSFER_ALPHA,
                        upper_action_scale=upper_action_scale,
                        lower_action_scale=lower_action_scale,
                        responsibility_mode=responsibility_mode,
                        leakage_cost_mode=leakage_cost_mode,
                        lower_action_router_mode=(
                            effective_lower_action_router_mode
                        ),
                        lower_action_router_alpha=lower_action_router_alpha,
                        lower_action_router_strength=(
                            effective_lower_action_router_strength
                        ),
                        lower_action_router_observe_strength=(
                            effective_router_observe_strength
                        ),
                        upper_action_decoder_mode=upper_action_decoder_mode,
                        upper_promotion_gain=upper_promotion_gain,
                        sample=False,
                        method=name,
                        episode_horizon=episode_horizon,
                    )[1]
                    row["safe_selector_branch"] = branch
                    safety_rows.append(row)
            branch_models[branch] = branch_model
            branch_payloads[branch] = branch_payload
            branch_selection_rows[branch] = safety_rows
            branch_training_summaries[branch] = {
                "leakage_constraint_enabled": branch_leakage,
                "constraint_update_mode": branch_update_mode,
                "constraint_scope": (
                    branch_constraint_scope
                    if branch_leakage else "disabled"
                ),
                "upper_hf_penalty_coef": (
                    branch_upper_penalty_coef
                ),
                "upper_constraint_mode": branch_upper_constraint_mode,
                "initial_parameter_sha256": initial_hash,
                "selected_parameter_sha256": _model_parameter_sha256(
                    branch_model
                ),
                "selected_checkpoint_iteration": int(
                    branch_payload["selected_checkpoint_iteration"]
                ),
                "validation_learning_gain": float(
                    branch_payload["validation_learning_gain"]
                ),
                "actor_optimizer_steps_train": int(
                    branch_payload["actor_optimizer_steps_train"]
                ),
                "critic_optimizer_steps_train": int(
                    branch_payload["critic_optimizer_steps_train"]
                ),
                "safety_selection_summary": summarize(safety_rows),
            }
        if len(initial_hashes) != 1:
            raise RuntimeError(
                "safe-selector branches did not share one initialization"
            )
        selector = select_safe_mujoco_branch(
            branch_selection_rows,
            bootstrap_seed=derive_seed(
                "mujoco_safe_selector_v1",
                str(env_id),
                int(optimizer_seed),
            ),
        )
        selected_branch = str(selector["selected_branch"])
        selected_spec = branch_specs[selected_branch]
        model = branch_models[selected_branch]
        payload = branch_payloads[selected_branch]
        rows = []
        selected_leakage_constraint = bool(
            selected_spec["leakage_constraint"]
        )
        selected_constraint_update_mode = (
            str(selected_spec["constraint_update_mode"])
            if selected_leakage_constraint else "disabled"
        )
        selected_constraint_scope = (
            str(selected_spec["constraint_scope"])
            if selected_leakage_constraint else "disabled"
        )
        selected_upper_hf_penalty_coef = float(
            selected_spec["upper_hf_penalty_coef"]
        )
        selected_upper_constraint_mode = str(
            selected_spec["upper_constraint_mode"]
        )
        selected_actor_steps = int(payload["actor_optimizer_steps_train"])
        selected_critic_steps = int(payload["critic_optimizer_steps_train"])
        payload.update({
            "policy": name,
            "branch_training_eval_seeds": [],
            "eval_seeds": list(evaluation),
            "safe_selector": selector,
            "safe_selector_branch_training": branch_training_summaries,
            "safe_selector_initialization_sha256": next(iter(initial_hashes)),
            "safe_selector_selection_seeds": list(safety_selection),
            "safe_selector_selection_disturbance_modes": list(training_modes),
            "safe_selector_behavioral_gate_contract": (
                "reward_floor_responsibility_lf_raw_lower_lf_and_upper_hf_v2"
            ),
            "selected_branch_actor_optimizer_steps_train": selected_actor_steps,
            "selected_branch_critic_optimizer_steps_train": selected_critic_steps,
            "actor_optimizer_steps_train": int(sum(
                item["actor_optimizer_steps_train"]
                for item in branch_training_summaries.values()
            )),
            "critic_optimizer_steps_train": int(sum(
                item["critic_optimizer_steps_train"]
                for item in branch_training_summaries.values()
            )),
            "safe_selector_training_compute_multiplier": len(
                SAFE_SELECTOR_BRANCHES
            ),
            "heldout_test_access_status": "not_loaded_during_branch_selection",
        })
        payload["gradient_updates_train"] = int(
            payload["actor_optimizer_steps_train"]
            + payload["critic_optimizer_steps_train"]
        )
    elif name == "flat_ppo":
        flat_hidden, _, _ = capacity_matched_flat_hidden_dim(
            target_parameter_count=target_parameters,
            state_dim=state_dim,
            action_dim=action_dim,
        )
        torch.manual_seed(int(optimizer_seed))
        model = JointActorCriticPPO(JointPPOConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=flat_hidden,
            learning_rate=float(learning_rate),
            epochs=4,
            minibatch_size=512,
            init_log_std=-0.7,
        ))
        rollout: Callable[..., tuple[Any, dict[str, Any]]] = lambda policy, seed, sample: rollout_flat(
            policy,
            seed=seed,
            env_id=env_id,
            disturbance_mode=assigned_mode(seed),
            steps=steps,
            sample=sample,
            episode_horizon=episode_horizon,
            lower_lf_rms_budget=lower_lf_rms_budget,
        )
        payload, rows, model = train_joint_ppo(
            model=model,
            train_seeds=roots,
            selection_seeds=selection_rollout_seeds,
            eval_seeds=[],
            iterations=iterations,
            training_seed_fn=lambda root, iteration: training_rollout_seed(
                int(optimizer_seed), root, iteration,
                domain=domain_seed_key,
            ),
            rollout_fn=rollout,
            objective_fn=lambda row: float(row["reward_mean"]),
            summary_fn=summarize,
            policy=name,
            domain="mujoco",
            checkpoint_smoothing_window=checkpoint_smoothing_window,
            checkpoint_min_delta=checkpoint_min_delta,
            checkpoint_evaluation_interval=checkpoint_evaluation_interval,
            checkpoint_score_fn=checkpoint_score_fn,
            checkpoint_score_contract=checkpoint_score_contract,
        )
    else:
        frequency_routing = name != "generic_hrl"
        leakage_constraint = name == "freq_hrl"
        method_constraint_scope = (
            str(leakage_constraint_scope)
            if leakage_constraint else "responsibility"
        )
        method_upper_penalty_coef = (
            float(upper_hf_penalty_coef)
            if (
                leakage_constraint
                and str(upper_constraint_mode) == "static_reward_penalty"
            ) else 0.0
        )
        method_upper_constraint_mode = (
            str(upper_constraint_mode)
            if leakage_constraint else "disabled"
        )
        torch.manual_seed(int(optimizer_seed))
        model = _hierarchical_model(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            leakage_constraint=leakage_constraint,
            upper_cost_critic=upper_cost_critic_for_capacity,
            upper_constraint=(
                method_upper_constraint_mode == "primal_dual"
            ),
            upper_dual_lr=upper_dual_lr,
            lower_dual_lr=lower_dual_lr,
            upper_constraint_update_mode=upper_constraint_update_mode,
            lower_constraint_update_mode=lower_constraint_update_mode,
            upper_actor_anchor_coef=upper_actor_anchor_coef,
            lower_actor_anchor_coef=lower_actor_anchor_coef,
            actor_anchor_zero_state_indices=(
                actor_anchor_zero_state_indices
            ),
            upper_deployment_frequency_dual_lr=(
                selected_upper_deployment_frequency_dual_lr
            ),
            lower_deployment_frequency_dual_lr=(
                selected_lower_deployment_frequency_dual_lr
            ),
            upper_deployment_frequency_lambda_init=(
                selected_upper_deployment_frequency_lambda_init
            ),
            lower_deployment_frequency_lambda_init=(
                selected_lower_deployment_frequency_lambda_init
            ),
            upper_deployment_frequency_step_scale=float(
                upper_deployment_frequency_step_scale
            ),
            lower_deployment_frequency_step_scale=float(
                lower_deployment_frequency_step_scale
            ),
            upper_deployment_frequency_max_projection_steps=int(
                upper_deployment_frequency_max_projection_steps
            ),
            lower_deployment_frequency_max_projection_steps=int(
                lower_deployment_frequency_max_projection_steps
            ),
            upper_deployment_frequency_reward_tolerance=float(
                upper_deployment_frequency_reward_tolerance
            ),
            lower_deployment_frequency_reward_tolerance=float(
                lower_deployment_frequency_reward_tolerance
            ),
            upper_deployment_frequency_target_tolerance=float(
                upper_deployment_frequency_target_tolerance
            ),
            lower_deployment_frequency_target_tolerance=float(
                lower_deployment_frequency_target_tolerance
            ),
            upper_deployment_frequency_rms_budget=float(
                upper_deployment_frequency_rms_budget
            ),
            lower_deployment_frequency_rms_budget=float(
                lower_deployment_frequency_rms_budget
            ),
            upper_deployment_frequency_reference_reduction_fraction=(
                selected_upper_deployment_frequency_reference_reduction
            ),
            lower_deployment_frequency_reference_reduction_fraction=(
                selected_lower_deployment_frequency_reference_reduction
            ),
            upper_deployment_frequency_action_scale=float(
                upper_action_scale
            ),
            lower_deployment_frequency_action_scale=float(
                lower_action_scale
            ),
            deployment_frequency_groupwise_robust=bool(
                deployment_frequency_groupwise_robust
            ),
            deployment_frequency_anchor_state_replay=bool(
                deployment_frequency_anchor_state_replay
            ),
            deployment_frequency_projection_objective=str(
                deployment_frequency_projection_objective
            ),
            deployment_frequency_projection_cvar_alpha=float(
                deployment_frequency_projection_cvar_alpha
            ),
            deployment_frequency_restoration_freeze_reward_actor=bool(
                deployment_frequency_restoration_freeze_reward_actor
            ),
            deployment_frequency_ppo_trust_region=bool(
                deployment_frequency_ppo_trust_region
            ),
            deployment_frequency_ppo_trust_region_backtracks=int(
                deployment_frequency_ppo_trust_region_backtracks
            ),
            deployment_frequency_closed_loop_trust_region=bool(
                deployment_frequency_closed_loop_trust_region
            ),
            deployment_frequency_closed_loop_trust_region_backtracks=int(
                deployment_frequency_closed_loop_trust_region_backtracks
            ),
            deployment_frequency_closed_loop_restoration_filter=bool(
                deployment_frequency_closed_loop_restoration_filter
            ),
            deployment_frequency_closed_loop_restoration_min_reduction=float(
                deployment_frequency_closed_loop_restoration_min_reduction
            ),
            deployment_frequency_closed_loop_restoration_funnel_multiplier=float(
                deployment_frequency_closed_loop_restoration_funnel_multiplier
            ),
            constraint_dual_normalization=str(
                constraint_dual_normalization
            ),
            constraint_dual_scale_ema_beta=float(
                constraint_dual_scale_ema_beta
            ),
            constraint_dual_scale_floor=float(
                constraint_dual_scale_floor
            ),
        )
        if paired_continuation:
            paired_checkpoint_metadata = load_paired_mujoco_checkpoint(
                model,
                checkpoint_path=Path(initial_checkpoint_path),
                summary_path=Path(initial_checkpoint_summary_path),
                env_id=env_id,
                optimizer_seed=optimizer_seed,
                expected_code_revision=source_identity["code_revision"],
                expected_source_manifest_sha256=(
                    source_identity["source_manifest_sha256"]
                ),
                expected_method=name,
                expected_router_mode=str(initial_checkpoint_router_mode),
                expected_router_strength=float(
                    initial_checkpoint_router_strength
                ),
                expected_router_observe_strength=bool(
                    effective_router_observe_strength
                ),
                expected_responsibility_mode=str(responsibility_mode),
                expected_protocol_version=effective_protocol_version,
                reset_upper_deployment_frequency_lambda=(
                    selected_upper_deployment_frequency_lambda_init
                ),
                reset_lower_deployment_frequency_lambda=(
                    selected_lower_deployment_frequency_lambda_init
                ),
            )
            model.capture_actor_anchor()
        def rollout(
            policy: FrequencySeparatedActorCriticPPO,
            seed: int,
            sample: bool,
            *,
            collect_trajectory: bool = False,
        ) -> tuple[HierarchicalTrajectoryBatch | None, dict[str, Any]]:
            return rollout_hierarchical(
                policy,
                seed=seed,
                env_id=env_id,
                disturbance_mode=assigned_mode(seed),
                steps=steps,
                upper_period=upper_period,
                frequency_routing=frequency_routing,
                leakage_constraint=leakage_constraint,
                lower_lf_rms_budget=lower_lf_rms_budget,
                leakage_constraint_scope=method_constraint_scope,
                upper_hf_rms_budget=upper_hf_rms_budget,
                upper_hf_penalty_coef=method_upper_penalty_coef,
                upper_constraint_mode=method_upper_constraint_mode,
                lower_lf_alpha=RESPONSIBILITY_TRANSFER_ALPHA,
                upper_action_scale=upper_action_scale,
                lower_action_scale=lower_action_scale,
                responsibility_mode=responsibility_mode,
                leakage_cost_mode=leakage_cost_mode,
                lower_action_router_mode=(
                    effective_lower_action_router_mode
                ),
                lower_action_router_alpha=lower_action_router_alpha,
                lower_action_router_strength=(
                    router_strength_for_seed(seed)
                ),
                lower_action_router_observe_strength=(
                    effective_router_observe_strength
                ),
                upper_action_decoder_mode=upper_action_decoder_mode,
                upper_promotion_gain=upper_promotion_gain,
                sample=sample,
                collect_trajectory=collect_trajectory,
                method=name,
                episode_horizon=episode_horizon,
            )
        closed_loop_guard_fn = None
        if deployment_frequency_closed_loop_trust_region:
            closed_loop_guard_baseline_parameter_sha256 = (
                _model_parameter_sha256(model)
            )
            closed_loop_guard_baseline_rows = [
                rollout(model, int(seed), False)[1]
                for seed in closed_loop_guard_rollout_seeds
            ]
            if (
                closed_loop_guard_baseline_parameter_sha256
                != _model_parameter_sha256(model)
            ):
                raise RuntimeError(
                    "closed-loop guard baseline evaluation mutated the model"
                )

            def closed_loop_guard_fn(
                policy: FrequencySeparatedActorCriticPPO,
            ) -> dict[str, Any]:
                parameter_sha256 = _model_parameter_sha256(policy)
                guard_rows = [
                    rollout(policy, int(seed), False)[1]
                    for seed in closed_loop_guard_rollout_seeds
                ]
                if parameter_sha256 != _model_parameter_sha256(policy):
                    raise RuntimeError(
                        "closed-loop guard evaluation mutated the model"
                    )
                snapshot = paired_closed_loop_guard_snapshot(
                    guard_rows,
                    baseline_rows=closed_loop_guard_baseline_rows,
                    expected_modes=training_modes,
                    lower_reduction_fraction=(
                        selected_lower_deployment_frequency_reference_reduction
                    ),
                    upper_reduction_fraction=(
                        selected_upper_deployment_frequency_reference_reduction
                    ),
                    lower_power_floor=(
                        float(lower_deployment_frequency_rms_budget) ** 2
                    ),
                    upper_power_floor=(
                        float(upper_deployment_frequency_rms_budget) ** 2
                    ),
                    pathwise_robust=bool(
                        deployment_frequency_pathwise_robust
                    ),
                    risk_mode=str(
                        deployment_frequency_closed_loop_risk_mode
                    ),
                    cvar_alpha=float(
                        deployment_frequency_closed_loop_cvar_alpha
                    ),
                )
                snapshot["parameter_sha256"] = parameter_sha256
                return snapshot
        if (
            str(checkpoint_score_mode)
            == "paired_relative_frequency_feasibility_first"
        ):
            baseline_parameter_sha256 = _model_parameter_sha256(model)
            paired_relative_baseline_parameter_sha256 = (
                baseline_parameter_sha256
            )
            paired_relative_baseline_rows = [
                rollout(model, int(seed), False)[1]
                for seed in selection_rollout_seeds
            ]
            if baseline_parameter_sha256 != _model_parameter_sha256(model):
                raise RuntimeError(
                    "paired checkpoint baseline evaluation mutated the model"
                )
            checkpoint_rank_fn = lambda rows: (
                paired_relative_frequency_feasibility_rank(
                    rows,
                    baseline_rows=paired_relative_baseline_rows,
                    expected_modes=training_modes,
                    lower_reduction_fraction=(
                        selected_lower_deployment_frequency_reference_reduction
                    ),
                    upper_reduction_fraction=(
                        selected_upper_deployment_frequency_reference_reduction
                    ),
                    lower_power_floor=(
                        float(lower_deployment_frequency_rms_budget) ** 2
                    ),
                    upper_power_floor=(
                        float(upper_deployment_frequency_rms_budget) ** 2
                    ),
                    pathwise_robust=bool(
                        deployment_frequency_pathwise_robust
                    ),
                    risk_mode=str(
                        deployment_frequency_closed_loop_risk_mode
                    ),
                    cvar_alpha=float(
                        deployment_frequency_closed_loop_cvar_alpha
                    ),
                )
            )
            checkpoint_diagnostics_fn = lambda rows: (
                paired_relative_frequency_feasibility_diagnostics(
                    rows,
                    baseline_rows=paired_relative_baseline_rows,
                    expected_modes=training_modes,
                    lower_reduction_fraction=(
                        selected_lower_deployment_frequency_reference_reduction
                    ),
                    upper_reduction_fraction=(
                        selected_upper_deployment_frequency_reference_reduction
                    ),
                    lower_power_floor=(
                        float(lower_deployment_frequency_rms_budget) ** 2
                    ),
                    upper_power_floor=(
                        float(upper_deployment_frequency_rms_budget) ** 2
                    ),
                    pathwise_robust=bool(
                        deployment_frequency_pathwise_robust
                    ),
                    risk_mode=str(
                        deployment_frequency_closed_loop_risk_mode
                    ),
                    cvar_alpha=float(
                        deployment_frequency_closed_loop_cvar_alpha
                    ),
                )
            )
            checkpoint_rank_names = (
                "negative_worst_paired_relative_violation",
                "negative_paired_relative_violation_l2",
                "worst_reward_floor_slack",
            )
            checkpoint_rank_contract = (
                (
                    "state_aligned_paired_selection_mode_cvar_reward_floor_"
                    "and_five_frequency_endpoint_relative_feasibility_"
                    f"alpha_{float(deployment_frequency_closed_loop_cvar_alpha):.6g}_v3"
                )
                if resolved_closed_loop_risk_mode == "mode_cvar" else (
                (
                    "state_aligned_paired_selection_individual_path_reward_"
                    "floor_and_five_frequency_endpoint_relative_"
                    "feasibility_v2"
                )
                if resolved_closed_loop_risk_mode == "pathwise_all" else (
                    "state_aligned_paired_selection_path_reward_floor_and_"
                    "five_frequency_endpoint_relative_feasibility_v1"
                )
                )
            )
            checkpoint_score_contract = (
                "paired_checkpoint_selection_risk_relative_feasibility_v2"
                if resolved_closed_loop_risk_mode == "mode_cvar"
                else "paired_checkpoint_selection_path_relative_feasibility_v1"
            )
        payload, rows, model = train_frequency_separated_ppo(
            model=model,
            train_seeds=roots,
            selection_seeds=selection_rollout_seeds,
            eval_seeds=[],
            iterations=iterations,
            training_seed_fn=lambda root, iteration: training_rollout_seed(
                int(optimizer_seed), root, iteration,
                domain=domain_seed_key,
            ),
            rollout_fn=rollout,
            objective_fn=lambda row: float(row["reward_mean"]),
            summary_fn=summarize,
            policy=name,
            domain="mujoco",
            checkpoint_smoothing_window=checkpoint_smoothing_window,
            checkpoint_min_delta=checkpoint_min_delta,
            checkpoint_minimum_iteration=checkpoint_minimum_iteration,
            checkpoint_evaluation_interval=checkpoint_evaluation_interval,
            checkpoint_score_fn=checkpoint_score_fn,
            checkpoint_score_contract=checkpoint_score_contract,
            checkpoint_rank_fn=checkpoint_rank_fn,
            checkpoint_rank_names=checkpoint_rank_names,
            checkpoint_rank_contract=checkpoint_rank_contract,
            checkpoint_diagnostics_fn=checkpoint_diagnostics_fn,
            deployment_frequency_reference_rollout_fn=(
                (
                    lambda policy, seed: rollout(
                        policy,
                        seed,
                        False,
                        collect_trajectory=True,
                    )[0]
                )
                if deployment_frequency_anchor_state_replay else None
            ),
            deployment_frequency_reference_seeds=(
                anchor_state_replay_rollout_seeds or None
            ),
            deployment_frequency_closed_loop_guard_fn=closed_loop_guard_fn,
        )

    actual_parameters = _module_parameter_count(model)
    evaluation_modes = _validated_disturbance_modes(
        [str(disturbance_mode)]
        if evaluation_disturbance_modes is None
        else evaluation_disturbance_modes,
        role="evaluation",
    )
    checkpoint_hash = _model_parameter_sha256(model)
    evaluation_rows: list[dict[str, Any]] = []
    for evaluation_mode in evaluation_modes:
        for evaluation_seed in evaluation:
            if name == "flat_ppo":
                row = rollout_flat(
                    model,
                    seed=int(evaluation_seed),
                    env_id=env_id,
                    disturbance_mode=evaluation_mode,
                    steps=steps,
                    sample=False,
                    episode_horizon=episode_horizon,
                    lower_lf_rms_budget=lower_lf_rms_budget,
                )[1]
            else:
                row = rollout_hierarchical(
                    model,
                    seed=int(evaluation_seed),
                    env_id=env_id,
                    disturbance_mode=evaluation_mode,
                    steps=steps,
                    upper_period=upper_period,
                    frequency_routing=name != "generic_hrl",
                    leakage_constraint=selected_leakage_constraint,
                    lower_lf_rms_budget=lower_lf_rms_budget,
                    leakage_constraint_scope=(
                        selected_constraint_scope
                        if selected_leakage_constraint
                        else "responsibility"
                    ),
                    upper_hf_rms_budget=upper_hf_rms_budget,
                    upper_hf_penalty_coef=(
                        selected_upper_hf_penalty_coef
                    ),
                    upper_constraint_mode=(
                        selected_upper_constraint_mode
                    ),
                    lower_lf_alpha=RESPONSIBILITY_TRANSFER_ALPHA,
                    upper_action_scale=upper_action_scale,
                    lower_action_scale=lower_action_scale,
                    responsibility_mode=responsibility_mode,
                    leakage_cost_mode=leakage_cost_mode,
                    lower_action_router_mode=effective_lower_action_router_mode,
                    lower_action_router_alpha=lower_action_router_alpha,
                    lower_action_router_strength=(
                        effective_lower_action_router_strength
                    ),
                    lower_action_router_observe_strength=(
                        effective_router_observe_strength
                    ),
                    upper_action_decoder_mode=upper_action_decoder_mode,
                    upper_promotion_gain=upper_promotion_gain,
                    sample=False,
                    method=name,
                    episode_horizon=episode_horizon,
                )[1]
            row.update({
                "training_replicate_seed": int(optimizer_seed),
                "evaluation_role": "heldout_test",
                "protocol_version": effective_protocol_version,
                "parameter_count": actual_parameters,
                "training_disturbance_mode": "multi_condition",
                "training_disturbance_modes": "|".join(training_modes),
                "responsibility_mode": str(responsibility_mode),
            })
            evaluation_rows.append(row)
    if checkpoint_hash != _model_parameter_sha256(model):
        raise RuntimeError("MuJoCo held-out evaluation mutated the checkpoint")
    payload["summary"] = summarize(evaluation_rows)
    payload["eval_seeds"] = list(evaluation)
    payload.update({
        "protocol_version": effective_protocol_version,
        "protocol_version_selection": selected_protocol_version,
        "method": name,
        "optimizer_seed": int(optimizer_seed),
        "environment": str(env_id),
        "disturbance_mode": str(disturbance_mode),
        "training_disturbance_modes": list(training_modes),
        "training_root_condition_assignment": {
            str(seed): mode for seed, mode in train_root_modes.items()
        },
        "checkpoint_selection_mode": str(checkpoint_selection_mode),
        "checkpoint_selection_seed_roots": list(selection),
        "checkpoint_selection_path_count": len(selection_rollout_seeds),
        "selection_seed_condition_assignment": {
            str(seed): mode for seed, mode in selection_seed_modes.items()
        },
        "checkpoint_score_mode": str(checkpoint_score_mode),
        "checkpoint_constraint_penalty": float(
            checkpoint_constraint_penalty
        ),
        "paired_relative_checkpoint_baseline": {
            "enabled": bool(paired_relative_baseline_rows),
            "row_count": len(paired_relative_baseline_rows),
            "parameter_sha256": paired_relative_baseline_parameter_sha256,
            "selection_paths": [
                {
                    "disturbance_mode": str(row["disturbance_mode"]),
                    "seed": int(row["seed"]),
                }
                for row in paired_relative_baseline_rows
            ],
            "summary": (
                summarize(paired_relative_baseline_rows)
                if paired_relative_baseline_rows else {}
            ),
            "heldout_rows_used": 0,
        },
        "deployment_frequency_closed_loop_guard_seed_roots": list(
            closed_loop_guard_roots
        ),
        "deployment_frequency_anchor_state_replay_seed_roots": list(
            anchor_state_replay_roots
        ),
        "deployment_frequency_anchor_state_replay_crossed_path_count": len(
            anchor_state_replay_rollout_seeds
        ),
        "deployment_frequency_anchor_state_replay_condition_assignment": {
            str(seed): mode
            for seed, mode in anchor_state_replay_seed_modes.items()
        },
        "deployment_frequency_closed_loop_guard_path_count": len(
            closed_loop_guard_rollout_seeds
        ),
        "deployment_frequency_closed_loop_guard_condition_assignment": {
            str(seed): mode
            for seed, mode in closed_loop_guard_seed_modes.items()
        },
        "deployment_frequency_closed_loop_guard_baseline": {
            "enabled": bool(closed_loop_guard_baseline_rows),
            "row_count": len(closed_loop_guard_baseline_rows),
            "parameter_sha256": (
                closed_loop_guard_baseline_parameter_sha256
            ),
            "paths": [
                {
                    "disturbance_mode": str(row["disturbance_mode"]),
                    "seed": int(row["seed"]),
                }
                for row in closed_loop_guard_baseline_rows
            ],
            "summary": (
                summarize(closed_loop_guard_baseline_rows)
                if closed_loop_guard_baseline_rows else {}
            ),
            "checkpoint_selection_rows_used": 0,
            "heldout_rows_used": 0,
        },
        "core_evaluation_seed_condition_assignment": {
            str(seed): mode for seed, mode in evaluation_seed_modes.items()
        },
        "evaluation_disturbance_modes": list(evaluation_modes),
        "evaluation_row_count": len(evaluation_rows),
        "steps": int(steps),
        "training_transition_budget_per_path": int(steps),
        "evaluation_episode_horizon": int(episode_horizon),
        "bootstrap_contract": (
            "explicit_reward_and_cost_next_value_with_separate_trace_boundary_"
            "and_mdp_terminal"
        ),
        "upper_period": int(upper_period),
        "frequency_routing_enabled": name.startswith("freq_hrl"),
        "leakage_constraint_enabled": selected_leakage_constraint,
        "leakage_constraint_scope": selected_constraint_scope,
        "leakage_constraint_cost_mode": str(leakage_cost_mode),
        "leakage_cost_contract": (
            "max_endpoint_aligned_32_step_causal_responsibility_effective_"
            "and_latent_lower_lf_"
            f"{str(leakage_cost_mode)}_v5"
            if selected_constraint_scope == "joint_behavior_latent"
            else (
                "max_endpoint_aligned_32_step_causal_responsibility_and_"
                f"effective_lower_lf_{str(leakage_cost_mode)}_v4"
                if selected_constraint_scope == "joint_behavior"
                else (
                    "endpoint_aligned_32_step_causal_responsibility_lf_"
                    f"{str(leakage_cost_mode)}_v3"
                    if selected_constraint_scope == "responsibility"
                    else "disabled"
                )
            )
        ),
        "lower_lf_rms_budget": float(lower_lf_rms_budget),
        "upper_hf_rms_budget": float(upper_hf_rms_budget),
        "upper_hf_penalty_coef": float(
            selected_upper_hf_penalty_coef
        ),
        "upper_constraint_mode": selected_upper_constraint_mode,
        "upper_dual_lr": (
            float(upper_dual_lr)
            if selected_upper_constraint_mode == "primal_dual"
            else 0.0
        ),
        "lower_dual_lr": (
            float(lower_dual_lr) if selected_leakage_constraint else 0.0
        ),
        "constraint_dual_normalization": (
            str(constraint_dual_normalization)
            if selected_leakage_constraint else "disabled"
        ),
        "constraint_dual_scale_ema_beta": float(
            constraint_dual_scale_ema_beta
        ),
        "constraint_dual_scale_floor": float(
            constraint_dual_scale_floor
        ),
        "upper_constraint_lambda_final": float(
            getattr(model, "upper_constraint_lambda", 0.0)
        ),
        "lower_constraint_lambda_final": float(
            getattr(model, "constraint_lambda", 0.0)
        ),
        "upper_constraint_violation_scale_final": float(
            getattr(model, "upper_constraint_violation_scale", 0.0)
        ),
        "lower_constraint_violation_scale_final": float(
            getattr(model, "lower_constraint_violation_scale", 0.0)
        ),
        "deployment_frequency_constraint_enabled": bool(
            deployment_frequency_requested and name == "freq_hrl"
        ),
        "upper_deployment_frequency_constraint_enabled": bool(
            deployment_frequency_level_active["upper"]
            and name == "freq_hrl"
        ),
        "lower_deployment_frequency_constraint_enabled": bool(
            deployment_frequency_level_active["lower"]
            and name == "freq_hrl"
        ),
        "deployment_frequency_action_source": (
            "deterministic_squashed_actor_mean"
            if deployment_frequency_requested and name == "freq_hrl"
            else "disabled"
        ),
        "upper_deployment_frequency_dual_lr": (
            selected_upper_deployment_frequency_dual_lr
        ),
        "lower_deployment_frequency_dual_lr": (
            selected_lower_deployment_frequency_dual_lr
        ),
        "upper_deployment_frequency_lambda_init": (
            selected_upper_deployment_frequency_lambda_init
        ),
        "lower_deployment_frequency_lambda_init": (
            selected_lower_deployment_frequency_lambda_init
        ),
        "upper_deployment_frequency_lambda_final": float(
            getattr(model, "upper_deployment_frequency_lambda", 0.0)
        ),
        "lower_deployment_frequency_lambda_final": float(
            getattr(model, "lower_deployment_frequency_lambda", 0.0)
        ),
        "upper_deployment_frequency_step_scale": float(
            upper_deployment_frequency_step_scale
        ),
        "lower_deployment_frequency_step_scale": float(
            lower_deployment_frequency_step_scale
        ),
        "upper_deployment_frequency_max_projection_steps": int(
            upper_deployment_frequency_max_projection_steps
        ),
        "lower_deployment_frequency_max_projection_steps": int(
            lower_deployment_frequency_max_projection_steps
        ),
        "upper_deployment_frequency_reward_tolerance": float(
            upper_deployment_frequency_reward_tolerance
        ),
        "lower_deployment_frequency_reward_tolerance": float(
            lower_deployment_frequency_reward_tolerance
        ),
        "upper_deployment_frequency_target_tolerance": float(
            upper_deployment_frequency_target_tolerance
        ),
        "lower_deployment_frequency_target_tolerance": float(
            lower_deployment_frequency_target_tolerance
        ),
        "upper_deployment_frequency_rms_budget": float(
            upper_deployment_frequency_rms_budget
        ),
        "lower_deployment_frequency_rms_budget": float(
            lower_deployment_frequency_rms_budget
        ),
        "upper_deployment_frequency_reference_reduction_fraction": (
            selected_upper_deployment_frequency_reference_reduction
        ),
        "lower_deployment_frequency_reference_reduction_fraction": (
            selected_lower_deployment_frequency_reference_reduction
        ),
        "deployment_frequency_groupwise_robust": bool(
            deployment_frequency_groupwise_robust
            and deployment_frequency_requested
            and name == "freq_hrl"
        ),
        "deployment_frequency_anchor_state_replay": bool(
            deployment_frequency_anchor_state_replay
            and deployment_frequency_requested
            and name == "freq_hrl"
        ),
        "deployment_frequency_projection_objective": str(
            deployment_frequency_projection_objective
        ),
        "deployment_frequency_projection_cvar_alpha": float(
            deployment_frequency_projection_cvar_alpha
        ),
        "deployment_frequency_restoration_freeze_reward_actor": bool(
            deployment_frequency_restoration_freeze_reward_actor
            and deployment_frequency_requested
            and name == "freq_hrl"
        ),
        "deployment_frequency_pathwise_robust": bool(
            deployment_frequency_pathwise_robust
            and deployment_frequency_requested
            and name == "freq_hrl"
        ),
        "deployment_frequency_closed_loop_risk_mode": (
            resolved_closed_loop_risk_mode
            if deployment_frequency_requested and name == "freq_hrl"
            else "disabled"
        ),
        "deployment_frequency_closed_loop_cvar_alpha": float(
            deployment_frequency_closed_loop_cvar_alpha
        ),
        "deployment_frequency_ppo_trust_region": bool(
            deployment_frequency_ppo_trust_region
            and deployment_frequency_requested
            and name == "freq_hrl"
        ),
        "deployment_frequency_ppo_trust_region_backtracks": int(
            deployment_frequency_ppo_trust_region_backtracks
        ),
        "deployment_frequency_closed_loop_trust_region": bool(
            deployment_frequency_closed_loop_trust_region
            and deployment_frequency_requested
            and name == "freq_hrl"
        ),
        "deployment_frequency_closed_loop_trust_region_backtracks": int(
            deployment_frequency_closed_loop_trust_region_backtracks
        ),
        "deployment_frequency_closed_loop_restoration_filter": bool(
            deployment_frequency_closed_loop_restoration_filter
            and deployment_frequency_requested
            and name == "freq_hrl"
        ),
        "deployment_frequency_closed_loop_restoration_min_reduction": float(
            deployment_frequency_closed_loop_restoration_min_reduction
        ),
        "deployment_frequency_closed_loop_restoration_funnel_multiplier": float(
            deployment_frequency_closed_loop_restoration_funnel_multiplier
        ),
        "deployment_frequency_constraint_contract": (
            deployment_frequency_constraint_contract(
                requested=(
                    deployment_frequency_requested and name == "freq_hrl"
                ),
                groupwise=deployment_frequency_groupwise_robust,
                anchor_state_replay=(
                    deployment_frequency_anchor_state_replay
                ),
                ppo_trust_region=deployment_frequency_ppo_trust_region,
                closed_loop_trust_region=(
                    deployment_frequency_closed_loop_trust_region
                ),
                closed_loop_restoration_filter=(
                    deployment_frequency_closed_loop_restoration_filter
                ),
                projection_objective=(
                    deployment_frequency_projection_objective
                ),
                projection_cvar_alpha=(
                    deployment_frequency_projection_cvar_alpha
                ),
                restoration_freeze_reward_actor=(
                    deployment_frequency_restoration_freeze_reward_actor
                ),
                pathwise_robust=deployment_frequency_pathwise_robust,
                closed_loop_risk_mode=(
                    deployment_frequency_closed_loop_risk_mode
                ),
                closed_loop_cvar_alpha=(
                    deployment_frequency_closed_loop_cvar_alpha
                ),
            )
        ),
        "upper_constraint_update_mode": (
            str(upper_constraint_update_mode)
            if selected_upper_constraint_mode == "primal_dual"
            else "disabled"
        ),
        "upper_objective_contract": (
            "raw_environment_reward_with_separate_primal_dual_endpoint_"
            + (
                "aligned_effective_and_latent_upper_high_pass_constraint_v2"
                if selected_constraint_scope == "joint_behavior_latent"
                else "aligned_upper_high_pass_constraint_v1"
            )
            if selected_upper_constraint_mode == "primal_dual"
            else (
                "raw_environment_reward_minus_endpoint_aligned_causal_upper_"
                "high_pass_budget_penalty_v2"
                if selected_upper_hf_penalty_coef > 0.0
                else "raw_environment_reward"
            )
        ),
        "reported_return_contract": "unshaped_environment_return",
        "diagnostic_alignment_contract": (
            "online_32_step_lower_low_pass_and_8_step_upper_high_pass_"
            "match_reported_LeakageRegularizer_windows_v1"
        ),
        "upper_action_scale": float(upper_action_scale),
        "lower_action_scale": float(lower_action_scale),
        "upper_action_decoder_mode": str(upper_action_decoder_mode),
        "upper_action_decoder_contract": (
            "boundary_sampled_endpoint_exact_c1_smoothstep_primitive_"
            "execution_v2"
            if str(upper_action_decoder_mode) == "causal_smoothstep_plan"
            else "macro_target_zero_order_hold_v1"
        ),
        "upper_promotion_gain": float(upper_promotion_gain),
        "upper_promotion_contract": (
            "previous_macro_causal_latent_lower_mean_promoted_into_clipped_"
            "next_upper_target_v1"
            if float(upper_promotion_gain) > 0.0 else "disabled"
        ),
        "lower_action_router_mode": effective_lower_action_router_mode,
        "lower_action_router_alpha": float(lower_action_router_alpha),
        "lower_action_router_strength": float(
            effective_lower_action_router_strength
        ),
        "lower_action_router_training_schedule": (
            effective_router_training_schedule
        ),
        "lower_action_router_warmup_fraction": float(
            lower_action_router_warmup_fraction
        ),
        "lower_action_router_ramp_fraction": float(
            lower_action_router_ramp_fraction
        ),
        "lower_action_router_observe_strength": bool(
            effective_router_observe_strength
        ),
        "paired_checkpoint_continuation": paired_checkpoint_metadata,
        "upper_actor_anchor_coef": float(upper_actor_anchor_coef),
        "lower_actor_anchor_coef": float(lower_actor_anchor_coef),
        "actor_anchor_zero_state_indices": list(
            actor_anchor_zero_state_indices
        ),
        "actor_anchor_contract": (
            (
                "frozen_matched_conservative_policy_same_state_analytic_"
                "gaussian_kl_v2"
                if effective_lower_action_router_mode
                in FUNCTION_PRESERVING_LOWER_ACTION_ROUTER_MODES
                else
                "frozen_matched_direct_policy_at_zero_router_context_"
                "analytic_gaussian_kl_v1"
            )
            if paired_continuation else "disabled"
        ),
        "upper_actor_anchor_parameter_rms": (
            model.actor_anchor_parameter_rms("upper")
            if paired_continuation else 0.0
        ),
        "lower_actor_anchor_parameter_rms": (
            model.actor_anchor_parameter_rms("lower")
            if paired_continuation else 0.0
        ),
        "lower_action_router_training_strengths_by_iteration": [
            float(value) for value in router_training_strengths_by_iteration
        ],
        "lower_action_router_schedule_contract": (
            "strength_independent_total_action_state_allows_unobserved_"
            "training_homotopy_with_frozen_target_evaluation_v2"
            if effective_lower_action_router_mode in {
                "causal_smooth_macro_gauge",
                "causal_audit_optimal_macro_gauge",
            }
            else "curriculum_applies_only_to_sampled_training_rollouts_while_"
            "checkpoint_selection_and_heldout_use_frozen_target_v1"
        ),
        "lower_action_router_contract": lower_action_router_contract(
            effective_lower_action_router_mode
        ),
        "lower_action_router_function_preserving": bool(
            effective_lower_action_router_mode
            in FUNCTION_PRESERVING_LOWER_ACTION_ROUTER_MODES
        ),
        "lower_action_router_diagnostic_contract": (
            "latent_and_effective_lower_actions_both_reported_v1"
        ),
        "lower_action_headroom_contract": (
            "audit_optimal_requested_upper_projected_to_exact_joint_component_"
            "feasibility_interval_each_step_v1"
            if effective_lower_action_router_mode
            == "causal_audit_optimal_macro_gauge"
            else
            "smooth_canonical_upper_projected_to_exact_joint_component_"
            "feasibility_interval_each_step_v1"
            if effective_lower_action_router_mode
            == "causal_smooth_macro_gauge"
            else
            "frozen_upper_macro_suffix_reserves_per_step_total_action_"
            "headroom_before_environment_disturbance_v1"
            if effective_lower_action_router_mode
            == "causal_macro_zero_dc_headroom" else "disabled"
        ),
        "responsibility_mode": str(responsibility_mode),
        "responsibility_transfer_alpha": RESPONSIBILITY_TRANSFER_ALPHA,
        "upper_to_lower_action_capacity_ratio": float(
            upper_action_scale / lower_action_scale
        ),
        "role_capacity_status": (
            "symmetric"
            if np.isclose(upper_action_scale, lower_action_scale)
            else (
                "upper_limited"
                if upper_action_scale < lower_action_scale
                else "lower_limited"
            )
        ),
        "action_capacity_contract": (
            "upper_anchor_and_lower_residual_unit_box_scales_reported_"
            "separately_v1"
        ),
        "responsibility_transfer_contract": (
            "causal_prior_lower_lf_to_next_upper_exact_nominal_"
            "reconstruction_v1"
            if str(responsibility_mode) == "causal_lf_transfer"
            else "additive_responsibility_control_v1"
        ),
        "policy_filter_state_contract": (
            "causal_total_low_pass_current_audit_plan_terminal_target_and_"
            "normalized_macro_phase_independent_of_gauge_strength_v1"
            if effective_lower_action_router_mode
            == "causal_audit_optimal_macro_gauge"
            else
            "causal_total_action_low_pass_context_independent_of_gauge_"
            "strength_v1"
            if effective_lower_action_router_mode
            == "causal_smooth_macro_gauge"
            else
            "causal_previous_macro_lower_mean_for_upper_replanning_and_"
            "running_raw_lower_lf_with_zero_dc_debt_for_lower_control_v1"
            if effective_lower_action_router_mode
            == "causal_macro_zero_dc_headroom"
            else "conservative_latent_ema_context_independent_of_transfer_"
            "strength_v4"
            if effective_lower_action_router_mode
            in FUNCTION_PRESERVING_LOWER_ACTION_ROUTER_MODES
            else (
                "canonical_raw_lf_observed_lower_router_state_and_strength_v3"
                if effective_router_observe_strength
                else "canonical_raw_lf_and_observed_lower_router_state_v2"
            )
        ),
        "lower_cost_state_contract": (
            "conservative_latent_32_step_lf_context_with_latent_effective_"
            "and_responsibility_endpoint_cost_v5"
            if (
                effective_lower_action_router_mode
                in FUNCTION_PRESERVING_LOWER_ACTION_ROUTER_MODES
                and selected_constraint_scope == "joint_behavior_latent"
            )
            else (
                "conservative_latent_32_step_lf_context_independent_of_"
                "transfer_strength_v4"
                if effective_lower_action_router_mode
                in FUNCTION_PRESERVING_LOWER_ACTION_ROUTER_MODES
                else
                "causal_responsibility_anchor_32_step_raw_and_responsibility_"
                "rolling_lf_cost_critic_only_v3"
            )
        ),
        "upper_cost_state_contract": (
            "same_causal_upper_policy_state_with_8_step_"
            + (
                "effective_and_latent_upper_high_pass_endpoint_cost_critic_v2"
                if selected_constraint_scope == "joint_behavior_latent"
                else "upper_high_pass_endpoint_cost_critic_v1"
            )
            if upper_cost_critic_for_capacity else "disabled"
        ),
        "lower_constraint_update_mode": selected_constraint_update_mode,
        "exogenous_observation_contract": (
            "current_causal_actuation_disturbance_decomposed_separately_from_"
            "raw_endogenous_state"
        ),
        "temporal_hierarchy_enabled": name != "flat_ppo",
        "capacity_target_parameter_count": target_parameters,
        "capacity_actual_parameter_count": actual_parameters,
        "capacity_ratio": float(actual_parameters / target_parameters),
        "source_identity_status": source_identity["source_identity_status"],
        "code_revision": source_identity["code_revision"],
        "source_manifest_sha256": source_identity["source_manifest_sha256"],
        "heldout_test_access_status": (
            "loaded_once_after_safe_branch_selection"
            if name == "freq_hrl_safe_selector"
            else "loaded_once_after_checkpoint_selection"
        ),
        "heldout_evaluation_pass_count": 1,
        "frozen_parameter_sha256": checkpoint_hash,
        "frozen_checkpoint_sha256": checkpoint_hash,
        "evaluation_summary_by_disturbance": {
            mode: summarize([
                row for row in evaluation_rows
                if str(row["disturbance_mode"]) == mode
            ])
            for mode in evaluation_modes
        },
    })
    return payload, evaluation_rows, model


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_cell(
    output_dir: Path,
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    model: Any,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "protocol_version": payload.get(
            "protocol_version", MUJOCO_CONTROL_PROTOCOL_VERSION
        ),
        "method": payload["method"],
        "environment": payload["environment"],
        "disturbance_mode": payload["disturbance_mode"],
        "optimizer_seed": payload["optimizer_seed"],
        "code_revision": payload["code_revision"],
        "source_manifest_sha256": payload["source_manifest_sha256"],
        "frozen_parameter_sha256": payload["frozen_parameter_sha256"],
        "frozen_checkpoint_sha256": payload["frozen_checkpoint_sha256"],
        "lower_action_router_mode": payload["lower_action_router_mode"],
        "lower_action_router_strength": payload[
            "lower_action_router_strength"
        ],
        "lower_action_router_observe_strength": payload[
            "lower_action_router_observe_strength"
        ],
        "upper_promotion_gain": float(payload.get("upper_promotion_gain", 0.0)),
        "responsibility_mode": payload["responsibility_mode"],
    }, checkpoint_path)
    payload["checkpoint_file_sha256"] = _file_sha256(checkpoint_path)
    payload["checkpoint_integrity_contract"] = (
        "independent_parameter_and_serialized_file_sha256_v1"
    )
    (output / "training_history.json").write_text(
        json.dumps(payload["history"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "evaluation_rows.csv", rows)
    summary = {key: value for key, value in payload.items() if key != "history"}
    (output / "cell_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--env-id", choices=DEFAULT_ENV_IDS, required=True)
    parser.add_argument(
        "--disturbance-mode",
        choices=DISTURBANCE_MODES,
        default="standard",
    )
    parser.add_argument(
        "--training-disturbance-modes",
        nargs="+",
        choices=DISTURBANCE_MODES,
        default=list(DEFAULT_TRAINING_DISTURBANCE_MODES),
    )
    parser.add_argument(
        "--evaluation-disturbance-modes",
        nargs="+",
        choices=DISTURBANCE_MODES,
        default=list(DISTURBANCE_MODES),
    )
    parser.add_argument("--train-seeds", type=int, nargs="+", default=list(DEFAULT_TRAIN_SEEDS))
    parser.add_argument("--selection-seeds", type=int, nargs="+", default=list(DEFAULT_SELECTION_SEEDS))
    parser.add_argument(
        "--safety-selection-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SAFETY_SELECTION_SEEDS),
    )
    parser.add_argument(
        "--deployment-frequency-closed-loop-guard-seeds",
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "--deployment-frequency-anchor-state-replay-seeds",
        type=int,
        nargs="+",
    )
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(DEFAULT_EVAL_SEEDS))
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--episode-horizon", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--upper-period", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lower-lf-rms-budget", type=float, default=0.05)
    parser.add_argument("--upper-action-scale", type=float, default=1.0)
    parser.add_argument("--lower-action-scale", type=float, default=1.0)
    parser.add_argument(
        "--upper-action-decoder-mode",
        choices=UPPER_ACTION_DECODER_MODES,
        default="hold",
    )
    parser.add_argument("--upper-promotion-gain", type=float, default=0.0)
    parser.add_argument(
        "--responsibility-mode",
        choices=RESPONSIBILITY_MODES,
        default="additive",
    )
    parser.add_argument(
        "--lower-action-router-mode",
        choices=LOWER_ACTION_ROUTER_MODES,
        default="direct",
    )
    parser.add_argument(
        "--lower-action-router-alpha",
        type=float,
        default=DEFAULT_LOWER_ROUTER_ALPHA,
    )
    parser.add_argument(
        "--lower-action-router-strength",
        type=float,
        default=DEFAULT_LOWER_ROUTER_STRENGTH,
    )
    parser.add_argument(
        "--lower-action-router-training-schedule",
        choices=LOWER_ACTION_ROUTER_TRAINING_SCHEDULES,
        default="constant",
    )
    parser.add_argument(
        "--lower-action-router-warmup-fraction",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--lower-action-router-ramp-fraction",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--lower-action-router-observe-strength",
        action="store_true",
    )
    parser.add_argument("--initial-checkpoint-path", type=Path)
    parser.add_argument("--initial-checkpoint-summary-path", type=Path)
    parser.add_argument(
        "--initial-checkpoint-router-mode",
        choices=LOWER_ACTION_ROUTER_MODES,
        default="direct",
    )
    parser.add_argument(
        "--initial-checkpoint-router-strength",
        type=float,
        default=0.0,
    )
    parser.add_argument("--upper-actor-anchor-coef", type=float, default=0.0)
    parser.add_argument("--lower-actor-anchor-coef", type=float, default=0.0)
    parser.add_argument(
        "--leakage-constraint-scope",
        choices=LEAKAGE_CONSTRAINT_SCOPES,
        default="joint_behavior",
    )
    parser.add_argument(
        "--upper-hf-rms-budget",
        type=float,
        default=DEFAULT_UPPER_HF_RMS_BUDGET,
    )
    parser.add_argument(
        "--upper-hf-penalty-coef",
        type=float,
        default=DEFAULT_UPPER_HF_PENALTY_COEF,
    )
    parser.add_argument(
        "--upper-constraint-mode",
        choices=UPPER_CONSTRAINT_MODES,
        default="static_reward_penalty",
    )
    parser.add_argument(
        "--upper-dual-lr",
        type=float,
        default=DEFAULT_UPPER_DUAL_LR,
    )
    parser.add_argument(
        "--lower-dual-lr",
        type=float,
        default=DEFAULT_LOWER_DUAL_LR,
    )
    parser.add_argument(
        "--constraint-dual-normalization",
        choices=("none", "ema_abs"),
        default="none",
    )
    parser.add_argument(
        "--constraint-dual-scale-ema-beta", type=float, default=0.95
    )
    parser.add_argument(
        "--constraint-dual-scale-floor", type=float, default=1e-6
    )
    parser.add_argument(
        "--upper-deployment-frequency-dual-lr", type=float, default=0.0
    )
    parser.add_argument(
        "--lower-deployment-frequency-dual-lr", type=float, default=0.0
    )
    parser.add_argument(
        "--upper-deployment-frequency-lambda-init", type=float, default=0.0
    )
    parser.add_argument(
        "--lower-deployment-frequency-lambda-init", type=float, default=0.0
    )
    parser.add_argument(
        "--upper-deployment-frequency-step-scale", type=float, default=1.0
    )
    parser.add_argument(
        "--lower-deployment-frequency-step-scale", type=float, default=1.0
    )
    parser.add_argument(
        "--upper-deployment-frequency-max-projection-steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--lower-deployment-frequency-max-projection-steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--upper-deployment-frequency-reward-tolerance",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--lower-deployment-frequency-reward-tolerance",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--upper-deployment-frequency-target-tolerance",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--lower-deployment-frequency-target-tolerance",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--upper-deployment-frequency-rms-budget", type=float, default=0.0
    )
    parser.add_argument(
        "--lower-deployment-frequency-rms-budget", type=float, default=0.0
    )
    parser.add_argument(
        "--upper-deployment-frequency-reference-reduction-fraction",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--lower-deployment-frequency-reference-reduction-fraction",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--deployment-frequency-groupwise-robust",
        action="store_true",
    )
    parser.add_argument(
        "--deployment-frequency-anchor-state-replay",
        action="store_true",
    )
    parser.add_argument(
        "--deployment-frequency-projection-objective",
        choices=DEPLOYMENT_FREQUENCY_PROJECTION_OBJECTIVES,
        default="worst_group",
    )
    parser.add_argument(
        "--deployment-frequency-projection-cvar-alpha",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--deployment-frequency-restoration-freeze-reward-actor",
        action="store_true",
    )
    parser.add_argument(
        "--deployment-frequency-pathwise-robust",
        action="store_true",
    )
    parser.add_argument(
        "--deployment-frequency-closed-loop-risk-mode",
        choices=CLOSED_LOOP_RISK_MODES,
        default="legacy",
    )
    parser.add_argument(
        "--deployment-frequency-closed-loop-cvar-alpha",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--deployment-frequency-ppo-trust-region",
        action="store_true",
    )
    parser.add_argument(
        "--deployment-frequency-ppo-trust-region-backtracks",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--deployment-frequency-closed-loop-trust-region",
        action="store_true",
    )
    parser.add_argument(
        "--deployment-frequency-closed-loop-trust-region-backtracks",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--deployment-frequency-closed-loop-restoration-filter",
        action="store_true",
    )
    parser.add_argument(
        "--deployment-frequency-closed-loop-restoration-min-reduction",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--deployment-frequency-closed-loop-restoration-funnel-multiplier",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--leakage-cost-mode",
        choices=LEAKAGE_COST_MODES,
        default="ratio_excess_squared",
    )
    parser.add_argument(
        "--upper-constraint-update-mode",
        choices=(
            "scalarized",
            "reward_guarded_projection",
            "reward_guarded_adam_projection",
        ),
        default="reward_guarded_adam_projection",
    )
    parser.add_argument(
        "--lower-constraint-update-mode",
        choices=(
            "scalarized",
            "reward_guarded_projection",
            "reward_guarded_adam_projection",
        ),
        default="reward_guarded_adam_projection",
    )
    parser.add_argument("--checkpoint-smoothing-window", type=int, default=8)
    parser.add_argument("--checkpoint-min-delta", type=float, default=1e-3)
    parser.add_argument(
        "--checkpoint-minimum-iteration", type=int, default=-1
    )
    parser.add_argument("--checkpoint-evaluation-interval", type=int, default=4)
    parser.add_argument(
        "--checkpoint-selection-mode",
        choices=CHECKPOINT_SELECTION_MODES,
        default="assigned_condition",
    )
    parser.add_argument(
        "--checkpoint-score-mode",
        choices=CHECKPOINT_SCORE_MODES,
        default="mean_reward",
    )
    parser.add_argument(
        "--checkpoint-constraint-penalty",
        type=float,
        default=DEFAULT_CHECKPOINT_CONSTRAINT_PENALTY,
    )
    parser.add_argument("--code-revision", default="")
    parser.add_argument("--source-manifest-sha256", default="")
    parser.add_argument(
        "--control-protocol-version",
        choices=MUJOCO_CONTROL_PROTOCOL_SELECTIONS,
        default="auto",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload, rows, model = train_mujoco_method(
        method=args.method,
        env_id=args.env_id,
        disturbance_mode=args.disturbance_mode,
        train_seeds=args.train_seeds,
        selection_seeds=args.selection_seeds,
        safety_selection_seeds=args.safety_selection_seeds,
        deployment_frequency_closed_loop_guard_seeds=(
            args.deployment_frequency_closed_loop_guard_seeds
        ),
        deployment_frequency_anchor_state_replay_seeds=(
            args.deployment_frequency_anchor_state_replay_seeds
        ),
        eval_seeds=args.eval_seeds,
        steps=args.steps,
        iterations=args.iterations,
        optimizer_seed=args.optimizer_seed,
        episode_horizon=args.episode_horizon,
        upper_period=args.upper_period,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        lower_lf_rms_budget=args.lower_lf_rms_budget,
        checkpoint_smoothing_window=args.checkpoint_smoothing_window,
        checkpoint_min_delta=args.checkpoint_min_delta,
        checkpoint_minimum_iteration=args.checkpoint_minimum_iteration,
        checkpoint_evaluation_interval=args.checkpoint_evaluation_interval,
        training_disturbance_modes=args.training_disturbance_modes,
        evaluation_disturbance_modes=args.evaluation_disturbance_modes,
        upper_action_scale=args.upper_action_scale,
        lower_action_scale=args.lower_action_scale,
        upper_action_decoder_mode=args.upper_action_decoder_mode,
        upper_promotion_gain=args.upper_promotion_gain,
        responsibility_mode=args.responsibility_mode,
        lower_action_router_mode=args.lower_action_router_mode,
        lower_action_router_alpha=args.lower_action_router_alpha,
        lower_action_router_strength=args.lower_action_router_strength,
        lower_action_router_training_schedule=(
            args.lower_action_router_training_schedule
        ),
        lower_action_router_warmup_fraction=(
            args.lower_action_router_warmup_fraction
        ),
        lower_action_router_ramp_fraction=(
            args.lower_action_router_ramp_fraction
        ),
        lower_action_router_observe_strength=(
            args.lower_action_router_observe_strength
        ),
        initial_checkpoint_path=args.initial_checkpoint_path,
        initial_checkpoint_summary_path=(
            args.initial_checkpoint_summary_path
        ),
        initial_checkpoint_router_mode=args.initial_checkpoint_router_mode,
        initial_checkpoint_router_strength=(
            args.initial_checkpoint_router_strength
        ),
        upper_actor_anchor_coef=args.upper_actor_anchor_coef,
        lower_actor_anchor_coef=args.lower_actor_anchor_coef,
        leakage_constraint_scope=args.leakage_constraint_scope,
        leakage_cost_mode=args.leakage_cost_mode,
        upper_hf_rms_budget=args.upper_hf_rms_budget,
        upper_hf_penalty_coef=args.upper_hf_penalty_coef,
        upper_constraint_mode=args.upper_constraint_mode,
        upper_dual_lr=args.upper_dual_lr,
        lower_dual_lr=args.lower_dual_lr,
        constraint_dual_normalization=args.constraint_dual_normalization,
        constraint_dual_scale_ema_beta=(
            args.constraint_dual_scale_ema_beta
        ),
        constraint_dual_scale_floor=args.constraint_dual_scale_floor,
        upper_deployment_frequency_dual_lr=(
            args.upper_deployment_frequency_dual_lr
        ),
        lower_deployment_frequency_dual_lr=(
            args.lower_deployment_frequency_dual_lr
        ),
        upper_deployment_frequency_lambda_init=(
            args.upper_deployment_frequency_lambda_init
        ),
        lower_deployment_frequency_lambda_init=(
            args.lower_deployment_frequency_lambda_init
        ),
        upper_deployment_frequency_step_scale=(
            args.upper_deployment_frequency_step_scale
        ),
        lower_deployment_frequency_step_scale=(
            args.lower_deployment_frequency_step_scale
        ),
        upper_deployment_frequency_max_projection_steps=(
            args.upper_deployment_frequency_max_projection_steps
        ),
        lower_deployment_frequency_max_projection_steps=(
            args.lower_deployment_frequency_max_projection_steps
        ),
        upper_deployment_frequency_reward_tolerance=(
            args.upper_deployment_frequency_reward_tolerance
        ),
        lower_deployment_frequency_reward_tolerance=(
            args.lower_deployment_frequency_reward_tolerance
        ),
        upper_deployment_frequency_target_tolerance=(
            args.upper_deployment_frequency_target_tolerance
        ),
        lower_deployment_frequency_target_tolerance=(
            args.lower_deployment_frequency_target_tolerance
        ),
        upper_deployment_frequency_rms_budget=(
            args.upper_deployment_frequency_rms_budget
        ),
        lower_deployment_frequency_rms_budget=(
            args.lower_deployment_frequency_rms_budget
        ),
        upper_deployment_frequency_reference_reduction_fraction=(
            args.upper_deployment_frequency_reference_reduction_fraction
        ),
        lower_deployment_frequency_reference_reduction_fraction=(
            args.lower_deployment_frequency_reference_reduction_fraction
        ),
        deployment_frequency_groupwise_robust=(
            args.deployment_frequency_groupwise_robust
        ),
        deployment_frequency_anchor_state_replay=(
            args.deployment_frequency_anchor_state_replay
        ),
        deployment_frequency_projection_objective=(
            args.deployment_frequency_projection_objective
        ),
        deployment_frequency_projection_cvar_alpha=(
            args.deployment_frequency_projection_cvar_alpha
        ),
        deployment_frequency_restoration_freeze_reward_actor=(
            args.deployment_frequency_restoration_freeze_reward_actor
        ),
        deployment_frequency_pathwise_robust=(
            args.deployment_frequency_pathwise_robust
        ),
        deployment_frequency_closed_loop_risk_mode=(
            args.deployment_frequency_closed_loop_risk_mode
        ),
        deployment_frequency_closed_loop_cvar_alpha=(
            args.deployment_frequency_closed_loop_cvar_alpha
        ),
        deployment_frequency_ppo_trust_region=(
            args.deployment_frequency_ppo_trust_region
        ),
        deployment_frequency_ppo_trust_region_backtracks=(
            args.deployment_frequency_ppo_trust_region_backtracks
        ),
        deployment_frequency_closed_loop_trust_region=(
            args.deployment_frequency_closed_loop_trust_region
        ),
        deployment_frequency_closed_loop_trust_region_backtracks=(
            args.deployment_frequency_closed_loop_trust_region_backtracks
        ),
        deployment_frequency_closed_loop_restoration_filter=(
            args.deployment_frequency_closed_loop_restoration_filter
        ),
        deployment_frequency_closed_loop_restoration_min_reduction=(
            args.deployment_frequency_closed_loop_restoration_min_reduction
        ),
        deployment_frequency_closed_loop_restoration_funnel_multiplier=(
            args.
            deployment_frequency_closed_loop_restoration_funnel_multiplier
        ),
        upper_constraint_update_mode=args.upper_constraint_update_mode,
        lower_constraint_update_mode=args.lower_constraint_update_mode,
        checkpoint_selection_mode=args.checkpoint_selection_mode,
        checkpoint_score_mode=args.checkpoint_score_mode,
        checkpoint_constraint_penalty=(
            args.checkpoint_constraint_penalty
        ),
        code_revision=args.code_revision,
        expected_source_manifest_sha256=args.source_manifest_sha256,
        control_protocol_version=args.control_protocol_version,
    )
    write_cell(args.output_dir, payload, rows, model)
    print(
        f"mujoco_control_cell status=valid method={args.method} "
        f"env={args.env_id} output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
