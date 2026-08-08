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

from freq_hrl.core import LeakageRegularizer
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


MUJOCO_CONTROL_PROTOCOL_VERSION = "freq_hrl_mujoco_shared_core_v3"
METHODS = (
    "freq_hrl",
    "freq_hrl_no_leakage",
    "generic_hrl",
    "flat_ppo",
)
DEFAULT_ENV_IDS = ("HalfCheetah-v5", "Hopper-v5", "Walker2d-v5")
DEFAULT_TRAIN_SEEDS = (31013, 31019, 31033)
DEFAULT_SELECTION_SEEDS = (32003, 32009, 32027)
DEFAULT_EVAL_SEEDS = (33013, 33023, 33029, 33037, 33049)


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
    return replace(
        level,
        next_value=next_values,
        terminal=terminals,
    )


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


def _feature_state(
    bands: dict[str, np.ndarray],
    action_context: np.ndarray,
    *,
    frequency_routing: bool,
    level: str,
) -> np.ndarray:
    context = np.asarray(action_context, dtype=np.float32).reshape(-1)
    if frequency_routing and level == "upper":
        pieces = (bands["slow"], bands["mid"], context)
    elif frequency_routing and level == "lower":
        pieces = (bands["mid"], bands["high"], context)
    else:
        pieces = (bands["raw"], bands["delta"], context)
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
) -> dict[str, Any]:
    executed = np.asarray(executed_actions, dtype=np.float64)
    upper = np.asarray(upper_actions, dtype=np.float64)
    lower = np.asarray(lower_actions, dtype=np.float64)
    leakage = LeakageRegularizer(
        upper_hf_window=8,
        lower_lf_window=32,
    ).compute(upper, lower)
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
        "UpperHFPower": float(leakage["UpperHFPower"]),
        "UpperHFPowerAbs": float(leakage["UpperHFPowerAbs"]),
        "LowerLFDrift": float(leakage["LowerLFDrift"]),
        "LowerLFDriftAbs": float(leakage["LowerLFDriftAbs"]),
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
    upper_action_scale: float = 0.70,
    lower_action_scale: float = 0.35,
    lower_lf_alpha: float = 0.04,
    lower_lf_budget: float = 0.015,
    method: str = "freq_hrl",
    episode_horizon: int = 1000,
) -> tuple[HierarchicalTrajectoryBatch | None, dict[str, Any]]:
    transition_budget = int(steps) if sample else int(episode_horizon)
    env = _make_env(env_id, episode_horizon=episode_horizon)
    try:
        observation, _ = env.reset(seed=int(seed))
        model.reset_recurrent_inference()
        decomposer = CausalBandDecomposer()
        bands = decomposer.reset(observation)
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
        boundary_terminals: list[float] = []
        current_episode_return = 0.0
        episode_index = 0
        steps_since_upper = int(upper_period)
        require_upper = True

        for step in range(transition_budget):
            if require_upper or steps_since_upper >= int(upper_period):
                upper_state = _feature_state(
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
                bands,
                upper_anchor,
                frequency_routing=frequency_routing,
                level="lower",
            )
            lower_out = model.act_lower(lower_state, sample=sample)
            lower_raw = np.asarray(lower_out["action"], dtype=np.float32)
            lower_residual = float(lower_action_scale) * np.tanh(lower_raw)
            nominal = np.clip(upper_anchor + lower_residual, -1.0, 1.0)
            disturbance = deterministic_actuation_disturbance(
                mode=disturbance_mode,
                step=step,
                action_dim=action_dim,
                seed=int(seed),
                horizon=int(episode_horizon),
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
            lower_lf += float(lower_lf_alpha) * (
                np.asarray(lower_residual, dtype=np.float64) - lower_lf
            )
            lower_cost = (
                max(float(np.mean(np.square(lower_lf))) - float(lower_lf_budget), 0.0)
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
                if sample and not bool(terminated):
                    next_bands = decomposer.update(next_observation)
                    next_upper_state = _feature_state(
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
                boundary_upper_next_values.append(upper_next_value)
                boundary_lower_next_values.append(lower_next_value)
                boundary_terminals.append(terminal)
                segment_returns.append(current_episode_return)
                if natural_done:
                    natural_episode_returns.append(current_episode_return)
                current_episode_return = 0.0
                if not sample or budget_done:
                    break
                episode_index += 1
                observation, _ = env.reset(seed=derive_seed(
                    "freq_hrl_mujoco_episode_reset_v1",
                    int(seed),
                    int(episode_index),
                ))
                model.reset_recurrent_inference()
                bands = decomposer.reset(observation)
                previous_action = np.zeros(action_dim, dtype=np.float32)
                upper_anchor = np.zeros(action_dim, dtype=np.float32)
                lower_lf = np.zeros(action_dim, dtype=np.float64)
                steps_since_upper = int(upper_period)
                require_upper = True
            else:
                bands = decomposer.update(next_observation)

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
) -> tuple[JointTrajectoryBatch | None, dict[str, Any]]:
    transition_budget = int(steps) if sample else int(episode_horizon)
    env = _make_env(env_id, episode_horizon=episode_horizon)
    try:
        observation, _ = env.reset(seed=int(seed))
        model.reset_recurrent_inference()
        decomposer = CausalBandDecomposer()
        bands = decomposer.reset(observation)
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
        current_episode_return = 0.0
        episode_index = 0

        for step in range(transition_budget):
            state = _feature_state(
                bands,
                previous_action,
                frequency_routing=False,
                level="lower",
            )
            output = model.act(state, sample=sample)
            raw_action = np.asarray(output["action"], dtype=np.float32)
            nominal = np.tanh(raw_action)
            disturbance = deterministic_actuation_disturbance(
                mode=disturbance_mode,
                step=step,
                action_dim=action_dim,
                seed=int(seed),
                horizon=int(episode_horizon),
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
            if done:
                terminal = float(bool(terminated))
                next_value = 0.0
                if sample and not bool(terminated):
                    next_bands = decomposer.update(next_observation)
                    next_state = _feature_state(
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
                observation, _ = env.reset(seed=derive_seed(
                    "freq_hrl_mujoco_episode_reset_v1",
                    int(seed),
                    int(episode_index),
                ))
                model.reset_recurrent_inference()
                bands = decomposer.reset(observation)
                previous_action = np.zeros(action_dim, dtype=np.float32)
            else:
                bands = decomposer.update(next_observation)

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
    "UpperHFPower",
    "UpperHFPowerAbs",
    "LowerLFDrift",
    "LowerLFDriftAbs",
    "upper_decision_count",
    "upper_transition_count",
    "lower_transition_count",
    "protocol_valid",
]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return summarize_numeric_rows(rows, keys=SUMMARY_KEYS)


def _hierarchical_model(
    *,
    state_dim: int,
    action_dim: int,
    hidden_dim: int,
    learning_rate: float,
    leakage_constraint: bool,
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
        lower_lambda_init=0.05 if leakage_constraint else 0.0,
        lower_cost_target=0.0,
        lower_dual_lr=1e-3 if leakage_constraint else 0.0,
        lower_max_lambda=20.0,
    ))


def train_mujoco_method(
    *,
    method: str,
    env_id: str,
    disturbance_mode: str,
    train_seeds: Iterable[int],
    selection_seeds: Iterable[int],
    eval_seeds: Iterable[int],
    steps: int,
    iterations: int,
    optimizer_seed: int,
    episode_horizon: int = 1000,
    upper_period: int = 16,
    hidden_dim: int = 64,
    learning_rate: float = 3e-4,
    checkpoint_smoothing_window: int = 8,
    checkpoint_min_delta: float = 1e-3,
    checkpoint_evaluation_interval: int = 4,
    evaluation_disturbance_modes: Iterable[str] | None = None,
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
    roles = (set(roots), set(selection), set(evaluation))
    if roles[0] & roles[1] or roles[0] & roles[2]:
        raise ValueError("MuJoCo training and evaluation seed roles overlap")
    if int(upper_period) < 2:
        raise ValueError("MuJoCo upper_period must be at least two")
    source_identity = verify_current_freq_hrl_source_identity(
        code_revision=str(code_revision),
        expected_source_manifest_sha256=str(expected_source_manifest_sha256),
    )
    if int(steps) < 1 or int(episode_horizon) < 1:
        raise ValueError("MuJoCo steps and episode_horizon must be positive")
    observation_dim, action_dim = environment_dimensions(
        env_id,
        episode_horizon=episode_horizon,
    )
    state_dim = 2 * observation_dim + action_dim
    torch.manual_seed(int(optimizer_seed))
    np.random.seed(int(optimizer_seed))

    reference = _hierarchical_model(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        leakage_constraint=True,
    )
    target_parameters = _module_parameter_count(reference)
    if name == "flat_ppo":
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
            disturbance_mode=disturbance_mode,
            steps=steps,
            sample=sample,
            episode_horizon=episode_horizon,
        )
        payload, rows, model = train_joint_ppo(
            model=model,
            train_seeds=roots,
            selection_seeds=selection,
            eval_seeds=evaluation,
            iterations=iterations,
            training_seed_fn=lambda root, iteration: training_rollout_seed(
                int(optimizer_seed), root, iteration,
                domain=f"mujoco:{env_id}:{disturbance_mode}",
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
        )
        rollout = lambda policy, seed, sample: rollout_hierarchical(
            policy,
            seed=seed,
            env_id=env_id,
            disturbance_mode=disturbance_mode,
            steps=steps,
            upper_period=upper_period,
            frequency_routing=frequency_routing,
            leakage_constraint=leakage_constraint,
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
                domain=f"mujoco:{env_id}:{disturbance_mode}",
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
    evaluation_modes = tuple(dict.fromkeys(
        [str(disturbance_mode)]
        if evaluation_disturbance_modes is None
        else map(str, evaluation_disturbance_modes)
    ))
    if not evaluation_modes or not set(evaluation_modes).issubset(
        DISTURBANCE_MODES
    ):
        raise ValueError("MuJoCo evaluation disturbance registry is invalid")
    checkpoint_hash = _model_parameter_sha256(model)
    primary_rows = {
        int(row["seed"]): row for row in rows
    }
    evaluation_rows: list[dict[str, Any]] = []
    for evaluation_mode in evaluation_modes:
        for evaluation_seed in evaluation:
            if (
                evaluation_mode == str(disturbance_mode)
                and int(evaluation_seed) in primary_rows
            ):
                row = dict(primary_rows[int(evaluation_seed)])
            elif name == "flat_ppo":
                row = rollout_flat(
                    model,
                    seed=int(evaluation_seed),
                    env_id=env_id,
                    disturbance_mode=evaluation_mode,
                    steps=steps,
                    sample=False,
                    episode_horizon=episode_horizon,
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
                    leakage_constraint=name == "freq_hrl",
                    sample=False,
                    method=name,
                    episode_horizon=episode_horizon,
                )[1]
            row.update({
                "training_replicate_seed": int(optimizer_seed),
                "evaluation_role": "heldout_test",
                "protocol_version": MUJOCO_CONTROL_PROTOCOL_VERSION,
                "parameter_count": actual_parameters,
                "training_disturbance_mode": str(disturbance_mode),
            })
            evaluation_rows.append(row)
    if checkpoint_hash != _model_parameter_sha256(model):
        raise RuntimeError("MuJoCo held-out evaluation mutated the checkpoint")
    payload.update({
        "protocol_version": MUJOCO_CONTROL_PROTOCOL_VERSION,
        "method": name,
        "environment": str(env_id),
        "disturbance_mode": str(disturbance_mode),
        "evaluation_disturbance_modes": list(evaluation_modes),
        "evaluation_row_count": len(evaluation_rows),
        "steps": int(steps),
        "training_transition_budget_per_path": int(steps),
        "evaluation_episode_horizon": int(episode_horizon),
        "bootstrap_contract": (
            "explicit_next_value_with_separate_trace_boundary_and_mdp_terminal"
        ),
        "upper_period": int(upper_period),
        "frequency_routing_enabled": name.startswith("freq_hrl"),
        "leakage_constraint_enabled": name == "freq_hrl",
        "temporal_hierarchy_enabled": name != "flat_ppo",
        "capacity_target_parameter_count": target_parameters,
        "capacity_actual_parameter_count": actual_parameters,
        "capacity_ratio": float(actual_parameters / target_parameters),
        "source_identity_status": source_identity["source_identity_status"],
        "code_revision": source_identity["code_revision"],
        "source_manifest_sha256": source_identity["source_manifest_sha256"],
        "heldout_test_access_status": "loaded_after_checkpoint_selection",
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
    summary = {key: value for key, value in payload.items() if key != "history"}
    (output / "cell_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "training_history.json").write_text(
        json.dumps(payload["history"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "evaluation_rows.csv", rows)
    torch.save({
        "model_state_dict": model.state_dict(),
        "protocol_version": MUJOCO_CONTROL_PROTOCOL_VERSION,
        "method": payload["method"],
        "environment": payload["environment"],
        "disturbance_mode": payload["disturbance_mode"],
        "frozen_checkpoint_sha256": payload["frozen_checkpoint_sha256"],
    }, output / "checkpoint.pt")


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
        "--evaluation-disturbance-modes",
        nargs="+",
        choices=DISTURBANCE_MODES,
        default=list(DISTURBANCE_MODES),
    )
    parser.add_argument("--train-seeds", type=int, nargs="+", default=list(DEFAULT_TRAIN_SEEDS))
    parser.add_argument("--selection-seeds", type=int, nargs="+", default=list(DEFAULT_SELECTION_SEEDS))
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(DEFAULT_EVAL_SEEDS))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--episode-horizon", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--optimizer-seed", type=int, required=True)
    parser.add_argument("--upper-period", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
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
        eval_seeds=args.eval_seeds,
        steps=args.steps,
        iterations=args.iterations,
        optimizer_seed=args.optimizer_seed,
        episode_horizon=args.episode_horizon,
        upper_period=args.upper_period,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        checkpoint_smoothing_window=args.checkpoint_smoothing_window,
        checkpoint_min_delta=args.checkpoint_min_delta,
        checkpoint_evaluation_interval=args.checkpoint_evaluation_interval,
        evaluation_disturbance_modes=args.evaluation_disturbance_modes,
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
