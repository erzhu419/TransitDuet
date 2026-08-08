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

from freq_hrl.core import LeakageRegularizer, evaluate_rms_leakage_budget
from freq_hrl.domains.mujoco import (
    CausalBandDecomposer,
    DISTURBANCE_MODES,
    action_from_unit_box,
    deterministic_actuation_disturbance,
)
from freq_hrl.experiments.reproducibility import (
    derive_seed,
    training_rollout_seed,
    validate_evaluation_seed_roles,
    validate_unique_seeds,
    verify_current_freq_hrl_source_identity,
)
from freq_hrl.rl import (
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
    "freq_hrl_mujoco_shared_core_v9_role_capacity_and_safe_selector"
)
METHODS = (
    "freq_hrl",
    "freq_hrl_safe_selector",
    "freq_hrl_no_leakage",
    "generic_hrl",
    "flat_ppo",
)
DEFAULT_ENV_IDS = ("HalfCheetah-v5", "Hopper-v5", "Walker2d-v5")
DEFAULT_TRAIN_SEEDS = (31013, 31019, 31033, 31039)
DEFAULT_SELECTION_SEEDS = (32003, 32009, 32027, 32029)
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
    "reward_guarded_adam_projection",
    "scalarized",
)
SAFE_SELECTOR_REWARD_MARGIN_FRACTION = 0.02
SAFE_SELECTOR_MIN_DRIFT_REDUCTION_FRACTION = 0.10
SAFE_SELECTOR_CONFIDENCE = 0.90
SAFE_SELECTOR_BOOTSTRAP_DRAWS = 4096


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


def mujoco_policy_state_dim(observation_dim: int, action_dim: int) -> int:
    if int(observation_dim) < 1 or int(action_dim) < 1:
        raise ValueError("MuJoCo observation and action dimensions must be positive")
    return int(observation_dim) + 3 * int(action_dim)


def _feature_state(
    observation: np.ndarray,
    bands: dict[str, np.ndarray],
    action_context: np.ndarray,
    *,
    frequency_routing: bool,
    level: str,
) -> np.ndarray:
    endogenous = np.asarray(observation, dtype=np.float32).reshape(-1)
    context = np.asarray(action_context, dtype=np.float32).reshape(-1)
    if frequency_routing and level == "upper":
        exogenous = (bands["slow"], bands["mid"])
    elif frequency_routing and level == "lower":
        exogenous = (bands["mid"], bands["high"])
    else:
        exogenous = (bands["raw"], bands["delta"])
    pieces = (endogenous, *exogenous, context)
    return np.concatenate(pieces).astype(np.float32, copy=False)


def _episode_row(
    *,
    seed: int,
    env_id: str,
    disturbance_mode: str,
    rewards: list[float],
    executed_actions: list[np.ndarray],
    upper_actions: list[np.ndarray],
    lower_actions: list[np.ndarray],
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
    lower_lf_rms_budget: float,
) -> dict[str, Any]:
    executed = np.asarray(executed_actions, dtype=np.float64)
    upper = np.asarray(upper_actions, dtype=np.float64)
    lower = np.asarray(lower_actions, dtype=np.float64)
    leakage = LeakageRegularizer(
        upper_hf_window=8,
        lower_lf_window=32,
    ).compute(upper, lower)
    upper_energy = float(np.mean(np.square(upper)))
    lower_energy = float(np.mean(np.square(lower)))
    responsibility_energy = upper_energy + lower_energy
    smoothness = (
        float(np.mean(np.square(np.diff(executed, axis=0))))
        if executed.shape[0] > 1 else 0.0
    )
    return {
        "seed": int(seed),
        "environment": str(env_id),
        "disturbance_mode": str(disturbance_mode),
        "method": str(method),
        "episode_return": float(np.sum(rewards)),
        "reward_mean": float(np.mean(rewards)),
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
        "UpperActionEnergyShare": float(
            upper_energy / responsibility_energy
            if responsibility_energy > 0.0 else 0.0
        ),
        "AdditiveActionClipRate": float(np.mean(
            np.abs(upper + lower) > 1.0
        )),
        "UpperHFPower": float(leakage["UpperHFPower"]),
        "UpperHFPowerAbs": float(leakage["UpperHFPowerAbs"]),
        "LowerLFDrift": float(leakage["LowerLFDrift"]),
        "LowerLFDriftAbs": float(leakage["LowerLFDriftAbs"]),
        "LowerLFRmsOnlineMean": float(np.mean(lower_lf_rms_values)),
        "LowerLFBudgetExcessMean": float(np.mean(lower_lf_budget_excesses)),
        "LowerLFBudgetViolationRate": float(np.mean(
            np.asarray(lower_lf_budget_excesses, dtype=np.float64) > 0.0
        )),
        "LowerLFRmsBudget": float(lower_lf_rms_budget),
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
    upper_action_scale: float = 1.0,
    lower_action_scale: float = 1.0,
    lower_lf_alpha: float = 0.04,
    lower_lf_rms_budget: float = 0.05,
    method: str = "freq_hrl",
    episode_horizon: int = 1000,
) -> tuple[HierarchicalTrajectoryBatch | None, dict[str, Any]]:
    transition_budget = int(steps) if sample else int(episode_horizon)
    env = _make_env(env_id, episode_horizon=episode_horizon)
    try:
        observation, _ = env.reset(seed=int(seed))
        model.reset_recurrent_inference()
        decomposer = CausalBandDecomposer()
        action_dim = int(env.action_space.shape[0])
        previous_action = np.zeros(action_dim, dtype=np.float32)
        upper_anchor = np.zeros(action_dim, dtype=np.float32)
        lower_lf = np.zeros(action_dim, dtype=np.float64)
        builder = HierarchicalRolloutBuilder(gamma=float(model.config.gamma))
        rewards: list[float] = []
        executed_actions: list[np.ndarray] = []
        upper_actions: list[np.ndarray] = []
        lower_actions: list[np.ndarray] = []
        forward_rewards: list[float] = []
        control_rewards: list[float] = []
        upper_decisions = 0
        segment_returns: list[float] = []
        natural_episode_returns: list[float] = []
        boundary_upper_next_values: list[float] = []
        boundary_lower_next_values: list[float] = []
        boundary_lower_next_cost_values: list[float] = []
        boundary_terminals: list[float] = []
        lower_lf_rms_values: list[float] = []
        lower_lf_budget_excesses: list[float] = []
        lower_cost_values: list[float] = []
        current_episode_return = 0.0
        episode_index = 0
        episode_seed = int(seed)
        episode_step = 0
        reset_exogenous = True
        steps_since_upper = int(upper_period)
        require_upper = True

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
            if require_upper or steps_since_upper >= int(upper_period):
                upper_state = _feature_state(
                    observation,
                    bands,
                    previous_action,
                    frequency_routing=frequency_routing,
                    level="upper",
                )
                upper_out = model.act_upper(upper_state, sample=sample)
                upper_raw = np.asarray(upper_out["action"], dtype=np.float32)
                upper_anchor = float(upper_action_scale) * np.tanh(upper_raw)
                builder.begin_upper(
                    state=upper_state,
                    action=upper_raw,
                    logp=float(upper_out["logp"]),
                    value=float(upper_out["value"]),
                )
                upper_decisions += 1
                steps_since_upper = 0
                require_upper = False

            lower_state = _feature_state(
                observation,
                bands,
                upper_anchor,
                frequency_routing=frequency_routing,
                level="lower",
            )
            lower_out = model.act_lower(lower_state, sample=sample)
            lower_raw = np.asarray(lower_out["action"], dtype=np.float32)
            lower_cost_values.append(float(lower_out["cost_value"]))
            lower_residual = float(lower_action_scale) * np.tanh(lower_raw)
            nominal = np.clip(upper_anchor + lower_residual, -1.0, 1.0)
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
            lower_lf += float(lower_lf_alpha) * (
                np.asarray(lower_residual, dtype=np.float64) - lower_lf
            )
            lower_budget = evaluate_rms_leakage_budget(
                float(np.mean(np.square(lower_lf))),
                float(lower_lf_rms_budget),
            )
            lower_lf_rms_values.append(float(lower_budget["rms"]))
            lower_lf_budget_excesses.append(float(lower_budget["budget_excess"]))
            lower_cost = (
                float(lower_budget["budget_excess_squared"])
                if leakage_constraint else 0.0
            )
            builder.add_lower(
                state=lower_state,
                action=lower_raw,
                logp=float(lower_out["logp"]),
                value=float(lower_out["value"]),
                reward=float(reward),
                upper_reward=float(reward),
                cost=float(lower_cost),
                done=done,
            )
            rewards.append(float(reward))
            current_episode_return += float(reward)
            executed_actions.append(executed.copy())
            upper_actions.append(upper_anchor.copy())
            lower_actions.append(lower_residual.copy())
            forward_rewards.append(float(info.get("reward_forward", 0.0)))
            control_rewards.append(float(info.get("reward_ctrl", 0.0)))
            previous_action = executed.astype(np.float32, copy=True)
            steps_since_upper += 1
            if done:
                terminal = float(bool(terminated))
                upper_next_value = 0.0
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
                        frequency_routing=frequency_routing,
                        level="upper",
                    )
                    next_upper = model.act_upper(
                        next_upper_state,
                        sample=False,
                    )
                    upper_next_value = float(next_upper["value"])
                    next_upper_anchor = upper_anchor
                    if steps_since_upper >= int(upper_period):
                        next_upper_anchor = float(upper_action_scale) * np.tanh(
                            np.asarray(next_upper["action"], dtype=np.float32)
                        )
                    next_lower_state = _feature_state(
                        next_observation,
                        next_bands,
                        next_upper_anchor,
                        frequency_routing=frequency_routing,
                        level="lower",
                    )
                    lower_next_value = _value_prediction(
                        model.lower_value,
                        next_lower_state,
                        device=model.device,
                    )
                    lower_next_cost_value = _value_prediction(
                        model.lower_cost_value,
                        next_lower_state,
                        device=model.device,
                    )
                boundary_upper_next_values.append(upper_next_value)
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
                lower_lf = np.zeros(action_dim, dtype=np.float64)
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
            lower_lf_rms_budget=lower_lf_rms_budget,
        )
        return (trajectory if sample else None), row
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
            lower_lf_rms_budget=lower_lf_rms_budget,
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
    "UpperActionEnergyShare",
    "AdditiveActionClipRate",
    "UpperHFPower",
    "UpperHFPowerAbs",
    "LowerLFDrift",
    "LowerLFDriftAbs",
    "LowerLFRmsOnlineMean",
    "LowerLFBudgetExcessMean",
    "LowerLFBudgetViolationRate",
    "LowerLFRmsBudget",
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
    baseline_reward_mean = float(np.mean(baseline_reward))
    baseline_drift_mean = float(np.mean(baseline_drift))
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
    diagnostics: dict[str, dict[str, Any]] = {
        SAFE_SELECTOR_BASELINE_BRANCH: {
            "path_count": len(path_keys),
            "independent_seed_count": len(independent_seeds),
            "episode_return_mean": baseline_reward_mean,
            "LowerLFDriftAbs_mean": baseline_drift_mean,
            "reward_difference_mean": 0.0,
            "reward_difference_one_sided_lower": 0.0,
            "reward_difference_one_sided_upper": 0.0,
            "drift_difference_mean": 0.0,
            "drift_difference_one_sided_lower": 0.0,
            "drift_difference_one_sided_upper": 0.0,
            "reward_noninferiority_supported": True,
            "minimum_drift_reduction_supported": False,
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
        reward_difference = reward - baseline_reward
        drift_difference = drift - baseline_drift
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
        reward_supported = reward_lower >= -reward_margin
        drift_supported = (
            baseline_drift_mean > np.finfo(np.float64).eps
            and drift_upper <= -required_drift_reduction
        )
        feasible = bool(reward_supported and drift_supported)
        diagnostics[branch] = {
            "path_count": len(path_keys),
            "independent_seed_count": len(independent_seeds),
            "episode_return_mean": float(np.mean(reward)),
            "LowerLFDriftAbs_mean": float(np.mean(drift)),
            "reward_difference_mean": float(np.mean(reward_difference)),
            "reward_difference_one_sided_lower": reward_lower,
            "reward_difference_one_sided_upper": reward_upper,
            "drift_difference_mean": float(np.mean(drift_difference)),
            "drift_difference_one_sided_lower": drift_lower,
            "drift_difference_one_sided_upper": drift_upper,
            "reward_noninferiority_supported": bool(reward_supported),
            "minimum_drift_reduction_supported": bool(drift_supported),
            "feasible": feasible,
        }
        if feasible:
            feasible_candidates.append(branch)

    selected_branch = (
        min(
            feasible_candidates,
            key=lambda branch: (
                float(diagnostics[branch][
                    "drift_difference_one_sided_upper"
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
            "paired_seed_cluster_bootstrap_reward_floor_and_drift_reduction_v1"
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
        "branch_diagnostics": diagnostics,
    }


def _hierarchical_model(
    *,
    state_dim: int,
    action_dim: int,
    hidden_dim: int,
    learning_rate: float,
    leakage_constraint: bool,
    lower_constraint_update_mode: str = "reward_guarded_adam_projection",
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
        lower_lambda_init=0.0,
        lower_cost_target=0.0,
        lower_dual_lr=0.1 if leakage_constraint else 0.0,
        lower_max_lambda=20.0,
        lower_cost_activation_threshold=1e-6,
        lower_zero_init_cost_value=True,
        lower_skip_inactive_cost_value_update=True,
        lower_constraint_update_mode=str(lower_constraint_update_mode),
        lower_constraint_step_scale=1.0,
        lower_constraint_max_backtracks=8,
        lower_constraint_reward_tolerance=1e-8,
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


def train_mujoco_method(
    *,
    method: str,
    env_id: str,
    disturbance_mode: str,
    train_seeds: Iterable[int],
    selection_seeds: Iterable[int],
    eval_seeds: Iterable[int],
    safety_selection_seeds: Iterable[int] | None = None,
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
    checkpoint_evaluation_interval: int = 4,
    training_disturbance_modes: Iterable[str] | None = None,
    evaluation_disturbance_modes: Iterable[str] | None = None,
    upper_action_scale: float = 1.0,
    lower_action_scale: float = 1.0,
    lower_constraint_update_mode: str = "reward_guarded_adam_projection",
    code_revision: str = "",
    expected_source_manifest_sha256: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    name = str(method)
    if name not in METHODS:
        raise ValueError(f"unknown MuJoCo method: {name}")
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
    seed_roles = {
        "training": set(roots),
        "checkpoint_selection": set(selection),
        "safety_selection": set(safety_selection),
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
    if str(lower_constraint_update_mode) not in {
        "scalarized",
        "reward_guarded_projection",
        "reward_guarded_adam_projection",
    }:
        raise ValueError("unknown lower constraint update mode")
    observation_dim, action_dim = environment_dimensions(
        env_id,
        episode_horizon=episode_horizon,
    )
    state_dim = mujoco_policy_state_dim(observation_dim, action_dim)
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
        or len(selection) < len(training_modes)
    ):
        raise ValueError(
            "multi-condition MuJoCo training requires at least one train and "
            "selection seed per disturbance mode"
        )
    domain_seed_key = f"mujoco:{env_id}:multi_condition_v1"
    seed_modes: dict[int, str] = {}
    train_root_modes = _assign_seed_modes(roots, training_modes)
    selection_seed_modes = _assign_seed_modes(selection, training_modes)
    evaluation_seed_modes = _assign_seed_modes(evaluation, training_modes)

    def register_seed_mode(seed: int, mode: str) -> None:
        previous = seed_modes.get(int(seed))
        if previous is not None and previous != str(mode):
            raise ValueError(
                f"MuJoCo seed {int(seed)} maps to conflicting conditions"
            )
        seed_modes[int(seed)] = str(mode)

    for iteration in range(max(1, int(iterations))):
        for root in roots:
            derived = training_rollout_seed(
                int(optimizer_seed), root, iteration, domain=domain_seed_key
            )
            register_seed_mode(derived, train_root_modes[int(root)])
    for seed, mode in selection_seed_modes.items():
        register_seed_mode(seed, mode)
    for seed, mode in evaluation_seed_modes.items():
        register_seed_mode(seed, mode)

    def assigned_mode(seed: int) -> str:
        try:
            return seed_modes[int(seed)]
        except KeyError as exc:
            raise KeyError(
                f"MuJoCo rollout seed {int(seed)} has no registered condition"
            ) from exc

    reference = _hierarchical_model(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        leakage_constraint=True,
        lower_constraint_update_mode=lower_constraint_update_mode,
    )
    target_parameters = _module_parameter_count(reference)
    selected_leakage_constraint = name == "freq_hrl"
    selected_constraint_update_mode = str(lower_constraint_update_mode)
    if name == "freq_hrl_safe_selector":
        branch_specs = {
            SAFE_SELECTOR_BASELINE_BRANCH: {
                "leakage_constraint": False,
                "constraint_update_mode": "reward_guarded_adam_projection",
            },
            "reward_guarded_adam_projection": {
                "leakage_constraint": True,
                "constraint_update_mode": "reward_guarded_adam_projection",
            },
            "scalarized": {
                "leakage_constraint": True,
                "constraint_update_mode": "scalarized",
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
            torch.manual_seed(int(optimizer_seed))
            np.random.seed(int(optimizer_seed))
            branch_model = _hierarchical_model(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                leakage_constraint=branch_leakage,
                lower_constraint_update_mode=branch_update_mode,
            )
            initial_hash = _model_parameter_sha256(branch_model)
            initial_hashes.add(initial_hash)
            branch_rollout = lambda policy, seed, sample, leakage=branch_leakage: rollout_hierarchical(
                policy,
                seed=seed,
                env_id=env_id,
                disturbance_mode=assigned_mode(seed),
                steps=steps,
                upper_period=upper_period,
                frequency_routing=True,
                leakage_constraint=leakage,
                lower_lf_rms_budget=lower_lf_rms_budget,
                upper_action_scale=upper_action_scale,
                lower_action_scale=lower_action_scale,
                sample=sample,
                method=name,
                episode_horizon=episode_horizon,
            )
            branch_payload, _, branch_model = train_frequency_separated_ppo(
                model=branch_model,
                train_seeds=roots,
                selection_seeds=selection,
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
                checkpoint_evaluation_interval=checkpoint_evaluation_interval,
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
                        upper_action_scale=upper_action_scale,
                        lower_action_scale=lower_action_scale,
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
            selection_seeds=selection,
            eval_seeds=evaluation,
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
        )
    else:
        frequency_routing = name != "generic_hrl"
        leakage_constraint = name == "freq_hrl"
        torch.manual_seed(int(optimizer_seed))
        model = _hierarchical_model(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            leakage_constraint=leakage_constraint,
            lower_constraint_update_mode=lower_constraint_update_mode,
        )
        rollout = lambda policy, seed, sample: rollout_hierarchical(
            policy,
            seed=seed,
            env_id=env_id,
            disturbance_mode=assigned_mode(seed),
            steps=steps,
            upper_period=upper_period,
            frequency_routing=frequency_routing,
            leakage_constraint=leakage_constraint,
            lower_lf_rms_budget=lower_lf_rms_budget,
            upper_action_scale=upper_action_scale,
            lower_action_scale=lower_action_scale,
            sample=sample,
            method=name,
            episode_horizon=episode_horizon,
        )
        payload, rows, model = train_frequency_separated_ppo(
            model=model,
            train_seeds=roots,
            selection_seeds=selection,
            eval_seeds=evaluation,
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
                    upper_action_scale=upper_action_scale,
                    lower_action_scale=lower_action_scale,
                    sample=False,
                    method=name,
                    episode_horizon=episode_horizon,
                )[1]
            row.update({
                "training_replicate_seed": int(optimizer_seed),
                "evaluation_role": "heldout_test",
                "protocol_version": MUJOCO_CONTROL_PROTOCOL_VERSION,
                "parameter_count": actual_parameters,
                "training_disturbance_mode": "multi_condition",
                "training_disturbance_modes": "|".join(training_modes),
            })
            evaluation_rows.append(row)
    if checkpoint_hash != _model_parameter_sha256(model):
        raise RuntimeError("MuJoCo held-out evaluation mutated the checkpoint")
    if name == "freq_hrl_safe_selector":
        payload["summary"] = summarize(evaluation_rows)
    payload.update({
        "protocol_version": MUJOCO_CONTROL_PROTOCOL_VERSION,
        "method": name,
        "environment": str(env_id),
        "disturbance_mode": str(disturbance_mode),
        "training_disturbance_modes": list(training_modes),
        "training_root_condition_assignment": {
            str(seed): mode for seed, mode in train_root_modes.items()
        },
        "selection_seed_condition_assignment": {
            str(seed): mode for seed, mode in selection_seed_modes.items()
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
        "leakage_cost_contract": "causal_lf_rms_budget_excess_squared_v1",
        "lower_lf_rms_budget": float(lower_lf_rms_budget),
        "upper_action_scale": float(upper_action_scale),
        "lower_action_scale": float(lower_action_scale),
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
            else "loaded_after_checkpoint_selection"
        ),
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
        "protocol_version": MUJOCO_CONTROL_PROTOCOL_VERSION,
        "method": payload["method"],
        "environment": payload["environment"],
        "disturbance_mode": payload["disturbance_mode"],
        "frozen_parameter_sha256": payload["frozen_parameter_sha256"],
        "frozen_checkpoint_sha256": payload["frozen_checkpoint_sha256"],
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
    parser.add_argument("--checkpoint-evaluation-interval", type=int, default=4)
    parser.add_argument("--code-revision", default="")
    parser.add_argument("--source-manifest-sha256", default="")
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
        checkpoint_evaluation_interval=args.checkpoint_evaluation_interval,
        training_disturbance_modes=args.training_disturbance_modes,
        evaluation_disturbance_modes=args.evaluation_disturbance_modes,
        upper_action_scale=args.upper_action_scale,
        lower_action_scale=args.lower_action_scale,
        lower_constraint_update_mode=args.lower_constraint_update_mode,
        code_revision=args.code_revision,
        expected_source_manifest_sha256=args.source_manifest_sha256,
    )
    write_cell(args.output_dir, payload, rows, model)
    print(
        f"mujoco_control_cell status=valid method={args.method} "
        f"env={args.env_id} output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
