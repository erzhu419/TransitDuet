#!/usr/bin/env python3
"""Probe occupancy-aware MuJoCo restoration with action-cost critics."""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
import json
import multiprocessing
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn import functional as F

from freq_hrl.experiments.mujoco.control_validation import (
    _model_parameter_sha256,
    rollout_hierarchical,
)
from freq_hrl.experiments.reproducibility import derive_seed
from freq_hrl.rl.action_cost_critic import (
    ActionCostCritic,
    discounted_smdp_cost_returns,
    transform_latent_action,
)
from freq_hrl.rl.restoration_portfolio import (
    fold_guarded_restoration_eligibility as fold_guarded_design_eligibility,
    paired_trace_invariance_diagnostics,
    restoration_snapshot_eligible,
    select_guarded_restoration_portfolio,
)
from freq_hrl.rl.deployment_frequency import deterministic_actor_action
from freq_hrl.rl.smdp_actor_critic import (
    HierarchicalTrajectoryBatch,
    LevelTrajectoryBatch,
    concat_hierarchical_batches,
)
from scripts import mujoco_v14_17_native_pd_cvar_screen_spec as v14_17_spec
from scripts.probe_mujoco_radial_restoration import (
    _actor_output_head,
    _evaluate_rows,
    _load_model,
    _v14_17_anchor_profile,
)
from scripts.probe_mujoco_zeroth_order_actor_restoration import (
    _paths_for_roots,
    _snapshot_fn,
)


PROBE_VERSION = "mujoco_action_cost_critic_restoration_probe_v1"


def actor_mean_parameter_vector(model: Any) -> np.ndarray:
    values = []
    for actor in (model.upper_actor, model.lower_actor):
        for name, parameter in actor.named_parameters():
            if name == "log_std" or name.endswith(".log_std"):
                continue
            values.append(parameter.detach().cpu().numpy().reshape(-1))
    if not values:
        raise RuntimeError("actor mean networks contain no parameters")
    return np.concatenate(values).astype(np.float64, copy=True)


def actor_output_bias_vector(model: Any) -> np.ndarray:
    values = []
    for actor in (model.upper_actor, model.lower_actor):
        bias = _actor_output_head(actor).bias
        if bias is None:
            raise TypeError("actor output-bias update requires output biases")
        values.append(bias.detach().cpu().numpy().reshape(-1))
    return np.concatenate(values).astype(np.float64, copy=True)


def apply_actor_mean_parameter_delta(model: Any, delta: np.ndarray) -> None:
    values = np.asarray(delta, dtype=np.float64).reshape(-1)
    expected = actor_mean_parameter_vector(model).size
    if values.size != expected or not np.all(np.isfinite(values)):
        raise ValueError("actor mean-parameter delta must be finite and aligned")
    offset = 0
    with torch.no_grad():
        for actor in (model.upper_actor, model.lower_actor):
            for name, parameter in actor.named_parameters():
                if name == "log_std" or name.endswith(".log_std"):
                    continue
                count = int(parameter.numel())
                update = torch.as_tensor(
                    values[offset:offset + count].reshape(parameter.shape),
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
                parameter.add_(update)
                offset += count
    if offset != values.size:
        raise RuntimeError("actor mean-parameter delta application was incomplete")


def apply_actor_output_bias_delta(model: Any, delta: np.ndarray) -> None:
    values = np.asarray(delta, dtype=np.float64).reshape(-1)
    expected = actor_output_bias_vector(model).size
    if values.size != expected or not np.all(np.isfinite(values)):
        raise ValueError("actor output-bias delta must be finite and aligned")
    offset = 0
    with torch.no_grad():
        for actor in (model.upper_actor, model.lower_actor):
            bias = _actor_output_head(actor).bias
            if bias is None:
                raise TypeError("actor output-bias update requires output biases")
            count = int(bias.numel())
            bias.add_(torch.as_tensor(
                values[offset:offset + count].reshape(bias.shape),
                dtype=bias.dtype,
                device=bias.device,
            ))
            offset += count
    if offset != values.size:
        raise RuntimeError("actor output-bias delta application was incomplete")


def actor_update_parameter_vector(model: Any, *, scope: str) -> np.ndarray:
    if str(scope) == "full_mean":
        return actor_mean_parameter_vector(model)
    if str(scope) == "output_bias":
        return actor_output_bias_vector(model)
    raise ValueError(f"unknown actor update scope: {scope}")


def apply_actor_update_delta(
    model: Any, delta: np.ndarray, *, scope: str
) -> None:
    if str(scope) == "full_mean":
        apply_actor_mean_parameter_delta(model, delta)
        return
    if str(scope) == "output_bias":
        apply_actor_output_bias_delta(model, delta)
        return
    raise ValueError(f"unknown actor update scope: {scope}")


def apply_actor_output_bias_intervention(
    model: Any,
    *,
    upper_bias: np.ndarray | None = None,
    lower_bias: np.ndarray | None = None,
) -> None:
    """Apply a temporary additive intervention to actor mean outputs."""

    for level, values in (("upper", upper_bias), ("lower", lower_bias)):
        if values is None:
            continue
        actor = getattr(model, f"{level}_actor")
        head = _actor_output_head(actor)
        if head.bias is None:
            raise TypeError("paired actor intervention requires output biases")
        bias = np.asarray(values, dtype=np.float64).reshape(-1)
        if bias.size != int(head.out_features) or not np.all(np.isfinite(bias)):
            raise ValueError(f"{level} actor intervention bias is misaligned")
        with torch.no_grad():
            head.bias.add_(torch.as_tensor(
                bias, dtype=head.bias.dtype, device=head.bias.device
            ))


def _actor_update_parameters(
    actor: torch.nn.Module, *, scope: str
) -> list[torch.nn.Parameter]:
    if str(scope) == "full_mean":
        return [
            parameter
            for name, parameter in actor.named_parameters()
            if name != "log_std" and not name.endswith(".log_std")
        ]
    if str(scope) == "output_bias":
        bias = _actor_output_head(actor).bias
        if bias is None:
            raise TypeError("actor output-bias update requires output biases")
        return [bias]
    raise ValueError(f"unknown actor update scope: {scope}")


def _rollout_path(
    model: Any,
    *,
    summary: dict[str, Any],
    path: dict[str, Any],
    sample: bool,
    collect_trajectory: bool,
    episode_horizon: int,
    leakage_cost_mode: str,
) -> tuple[HierarchicalTrajectoryBatch | None, dict[str, Any]]:
    return rollout_hierarchical(
        model,
        seed=int(path["seed"]),
        env_id=str(summary["environment"]),
        disturbance_mode=str(path["disturbance_mode"]),
        steps=int(summary["steps"]),
        upper_period=int(summary["upper_period"]),
        frequency_routing=True,
        leakage_constraint=True,
        lower_lf_rms_budget=float(summary["lower_lf_rms_budget"]),
        leakage_constraint_scope=str(summary["leakage_constraint_scope"]),
        upper_hf_rms_budget=float(summary["upper_hf_rms_budget"]),
        upper_hf_penalty_coef=float(summary["upper_hf_penalty_coef"]),
        upper_constraint_mode=str(summary["upper_constraint_mode"]),
        lower_lf_alpha=0.04,
        upper_action_scale=float(summary["upper_action_scale"]),
        lower_action_scale=float(summary["lower_action_scale"]),
        responsibility_mode=str(summary["responsibility_mode"]),
        leakage_cost_mode=str(leakage_cost_mode),
        lower_action_router_mode=str(summary["lower_action_router_mode"]),
        lower_action_router_alpha=float(summary["lower_action_router_alpha"]),
        lower_action_router_strength=float(summary["lower_action_router_strength"]),
        lower_action_router_observe_strength=bool(
            summary["lower_action_router_observe_strength"]
        ),
        sample=bool(sample),
        collect_trajectory=bool(collect_trajectory),
        method="freq_hrl",
        episode_horizon=int(episode_horizon),
    )


def _trajectory_job(job: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    policy_seed = int(job["policy_seed"])
    torch.manual_seed(policy_seed)
    np.random.seed(policy_seed % (2**32 - 1))
    checkpoint = torch.load(
        job["checkpoint_path"], map_location="cpu", weights_only=False
    )
    model = _load_model(checkpoint)
    source_hash = _model_parameter_sha256(model)
    intervention = dict(job.get("intervention") or {})
    if intervention:
        apply_actor_output_bias_intervention(
            model,
            upper_bias=(
                None if intervention.get("upper_bias") is None
                else np.asarray(intervention["upper_bias"], dtype=np.float64)
            ),
            lower_bias=(
                None if intervention.get("lower_bias") is None
                else np.asarray(intervention["lower_bias"], dtype=np.float64)
            ),
        )
    rollout_hash = _model_parameter_sha256(model)
    batch, row = _rollout_path(
        model,
        summary=job["summary"],
        path=job["path"],
        sample=bool(job["sample"]),
        collect_trajectory=True,
        episode_horizon=int(job["episode_horizon"]),
        leakage_cost_mode=str(job["leakage_cost_mode"]),
    )
    if batch is None:
        raise RuntimeError("trajectory worker did not collect a trajectory")
    return {
        "batch": batch,
        "row": row,
        "parameter_sha256": source_hash,
        "rollout_parameter_sha256": rollout_hash,
        "intervention_variant": str(
            intervention.get("variant", "stochastic_policy")
        ),
    }


def _actor_delta_path_job(job: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    checkpoint = torch.load(
        job["checkpoint_path"], map_location="cpu", weights_only=False
    )
    model = _load_model(checkpoint)
    apply_actor_update_delta(
        model,
        np.asarray(job["delta"], dtype=np.float64),
        scope=str(job.get("actor_update_scope", "full_mean")),
    )
    router_strength = float(job.get(
        "router_strength", job["summary"]["lower_action_router_strength"]
    ))
    row = _evaluate_rows(
        model,
        summary=job["summary"],
        paths=[job["path"]],
        episode_horizon=int(job["episode_horizon"]),
        leakage_cost_mode=str(job["leakage_cost_mode"]),
        router_strength=router_strength,
    )[0]
    return {
        "candidate_index": int(job["candidate_index"]),
        "path_index": int(job["path_index"]),
        "parameter_sha256": _model_parameter_sha256(model),
        "router_strength": router_strength,
        "row": row,
    }


def _parallel_map(
    worker: Any,
    jobs: list[dict[str, Any]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    if int(workers) < 1:
        raise ValueError("parallel worker count must be positive")
    if int(workers) == 1:
        return [worker(job) for job in jobs]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(int(workers), len(jobs)),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        return list(executor.map(worker, jobs))


def paired_output_bias_interventions(
    model: Any,
    *,
    seed: int,
    bias_rms: float,
    direction_scheme: str = "random_rademacher",
    direction_index: int = 0,
    hadamard_order: int = 0,
) -> list[dict[str, Any]]:
    """Create control and level-isolated antithetic actor interventions."""

    amplitude = float(bias_rms)
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("paired actor intervention RMS must be positive")
    upper_dim = int(_actor_output_head(model.upper_actor).out_features)
    lower_dim = int(_actor_output_head(model.lower_actor).out_features)
    scheme = str(direction_scheme)
    if scheme == "random_rademacher":
        rng = np.random.default_rng(int(seed))
        upper = amplitude * rng.choice((-1.0, 1.0), size=upper_dim)
        lower = amplitude * rng.choice((-1.0, 1.0), size=lower_dim)
    elif scheme == "balanced_hadamard":
        upper = amplitude * balanced_hadamard_direction(
            upper_dim, index=direction_index, order=hadamard_order
        )
        lower = amplitude * balanced_hadamard_direction(
            lower_dim, index=direction_index, order=hadamard_order
        )
    else:
        raise ValueError("unknown paired actor intervention direction scheme")
    return [
        {
            "variant": "control", "upper_bias": None, "lower_bias": None,
            "direction_scheme": scheme, "direction_index": int(direction_index),
        },
        {
            "variant": "upper_plus", "upper_bias": upper, "lower_bias": None,
            "direction_scheme": scheme, "direction_index": int(direction_index),
        },
        {
            "variant": "upper_minus", "upper_bias": -upper, "lower_bias": None,
            "direction_scheme": scheme, "direction_index": int(direction_index),
        },
        {
            "variant": "lower_plus", "upper_bias": None, "lower_bias": lower,
            "direction_scheme": scheme, "direction_index": int(direction_index),
        },
        {
            "variant": "lower_minus", "upper_bias": None, "lower_bias": -lower,
            "direction_scheme": scheme, "direction_index": int(direction_index),
        },
    ]


def balanced_hadamard_direction(
    dimension: int, *, index: int, order: int
) -> np.ndarray:
    """Return one row from a balanced orthogonal intervention design."""

    width = int(order)
    if width < 1 or width & (width - 1):
        raise ValueError("Hadamard order must be a positive power of two")
    if int(dimension) < 1 or int(dimension) > width:
        raise ValueError("Hadamard order must cover the actor output dimension")
    if int(index) < 0 or int(index) >= width:
        raise ValueError("Hadamard direction index is outside the design")
    matrix = np.ones((1, 1), dtype=np.float64)
    while matrix.shape[0] < width:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix[int(index), :int(dimension)].copy()


@dataclass
class _CriticFit:
    model: ActionCostCritic
    state_mean: np.ndarray
    state_scale: np.ndarray
    action_mean: np.ndarray
    action_scale: np.ndarray
    target_mean: float
    target_scale: float
    final_train_loss: float


def _level_arrays(
    batches: Iterable[HierarchicalTrajectoryBatch],
    *,
    level: str,
    gamma: float,
    action_transform: str,
    action_scale: float,
    max_return_decisions: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    merged = concat_hierarchical_batches(list(batches))
    batch = merged.upper if str(level) == "upper" else merged.lower
    if batch.cost is None:
        raise RuntimeError(f"{level} trajectory lacks native cost labels")
    state = np.asarray(
        batch.state if batch.cost_state is None else batch.cost_state,
        dtype=np.float32,
    )
    latent_action = torch.as_tensor(batch.action, dtype=torch.float32)
    action = transform_latent_action(
        latent_action,
        transform=str(action_transform),
        scale=float(action_scale),
    ).numpy()
    target = discounted_smdp_cost_returns(
        batch.cost,
        batch.duration,
        batch.done,
        gamma=float(gamma),
        max_decisions=max_return_decisions,
    )
    return state, action.astype(np.float32), target


def _fit_critic(
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    seed: int,
    hidden_dim: int,
    epochs: int,
    minibatch_size: int,
    learning_rate: float,
) -> _CriticFit:
    state, action, target = train
    if state.shape[0] != action.shape[0] or state.shape[0] != target.size:
        raise ValueError("action-cost training arrays are misaligned")
    state_mean = np.mean(state, axis=0, dtype=np.float64).astype(np.float32)
    state_scale = np.maximum(
        np.std(state, axis=0, dtype=np.float64), 1e-3
    ).astype(np.float32)
    action_mean = np.mean(action, axis=0, dtype=np.float64).astype(np.float32)
    action_scale = np.maximum(
        np.std(action, axis=0, dtype=np.float64), 1e-3
    ).astype(np.float32)
    target_mean = float(np.mean(target, dtype=np.float64))
    target_scale = max(float(np.std(target, dtype=np.float64)), 1e-6)
    state_t = torch.as_tensor(
        (state - state_mean) / state_scale, dtype=torch.float32
    )
    action_t = torch.as_tensor(
        (action - action_mean) / action_scale, dtype=torch.float32
    )
    target_t = torch.as_tensor(
        (target - target_mean) / target_scale, dtype=torch.float32
    )
    torch.manual_seed(int(seed))
    model = ActionCostCritic(
        int(state.shape[1]), int(action.shape[1]), int(hidden_dim)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    rng = np.random.default_rng(int(seed))
    bootstrap = rng.integers(0, target.size, size=target.size)
    final_loss = 0.0
    width = max(1, min(int(minibatch_size), target.size))
    model.train()
    for _ in range(int(epochs)):
        order = rng.permutation(bootstrap)
        losses = []
        for start in range(0, order.size, width):
            index = torch.as_tensor(order[start:start + width], dtype=torch.long)
            prediction = model(state_t[index], action_t[index])
            loss = F.smooth_l1_loss(prediction, target_t[index])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        final_loss = float(np.mean(losses))
    model.eval()
    return _CriticFit(
        model=model,
        state_mean=state_mean,
        state_scale=state_scale,
        action_mean=action_mean,
        action_scale=action_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        final_train_loss=final_loss,
    )


def _predict_normalized(
    fit: _CriticFit,
    state: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    state_mean = torch.as_tensor(
        fit.state_mean, dtype=state.dtype, device=state.device
    )
    state_scale = torch.as_tensor(
        fit.state_scale, dtype=state.dtype, device=state.device
    )
    action_mean = torch.as_tensor(
        fit.action_mean, dtype=action.dtype, device=action.device
    )
    action_scale = torch.as_tensor(
        fit.action_scale, dtype=action.dtype, device=action.device
    )
    return fit.model(
        (state - state_mean) / state_scale,
        (action - action_mean) / action_scale,
    )


def _critic_metrics(
    fits: list[_CriticFit],
    holdout: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    permutation_seed: int,
) -> dict[str, Any]:
    state, action, target = holdout
    state_t = torch.as_tensor(state, dtype=torch.float32)
    action_t = torch.as_tensor(action, dtype=torch.float32)
    predictions = []
    with torch.no_grad():
        for fit in fits:
            normalized = _predict_normalized(fit, state_t, action_t)
            predictions.append(
                normalized.numpy() * fit.target_scale + fit.target_mean
            )
    matrix = np.stack(predictions, axis=0)
    ensemble = np.mean(matrix, axis=0)
    permutation = np.random.default_rng(int(permutation_seed)).permutation(
        target.size
    )
    if target.size > 1 and np.array_equal(
        permutation, np.arange(target.size)
    ):
        permutation = np.roll(permutation, 1)
    permuted_action_t = action_t[torch.as_tensor(permutation, dtype=torch.long)]
    permuted_predictions = []
    with torch.no_grad():
        for fit in fits:
            normalized = _predict_normalized(
                fit, state_t, permuted_action_t
            )
            permuted_predictions.append(
                normalized.numpy() * fit.target_scale + fit.target_mean
            )
    permuted_ensemble = np.mean(
        np.stack(permuted_predictions, axis=0), axis=0
    )
    ensemble_mse = float(np.mean(np.square(target - ensemble)))
    permuted_mse = float(
        np.mean(np.square(target - permuted_ensemble))
    )
    denominator = float(np.sum(np.square(target - np.mean(target))))

    def r2(prediction: np.ndarray) -> float:
        if denominator <= 1e-12:
            return -1e6
        return float(
            1.0 - np.sum(np.square(target - prediction)) / denominator
        )

    target_std = float(np.std(target))
    prediction_std = float(np.std(ensemble))
    pearson = (
        float(np.corrcoef(target, ensemble)[0, 1])
        if target_std > 1e-12 and prediction_std > 1e-12 else 0.0
    )
    return {
        "train_transition_count": None,
        "holdout_transition_count": int(target.size),
        "target_mean": float(np.mean(target)),
        "target_std": target_std,
        "target_positive_fraction": float(np.mean(target > 0.0)),
        "individual_holdout_r2": [r2(row) for row in matrix],
        "ensemble_holdout_r2": r2(ensemble),
        "ensemble_holdout_normalized_rmse": float(
            np.sqrt(np.mean(np.square(target - ensemble)))
            / max(target_std, 1e-6)
        ),
        "ensemble_holdout_pearson": pearson,
        "ensemble_holdout_mse": ensemble_mse,
        "permuted_action_holdout_mse": permuted_mse,
        "action_permutation_mse_increase_fraction": float(
            (permuted_mse - ensemble_mse) / max(ensemble_mse, 1e-12)
        ),
        "action_permutation_seed": int(permutation_seed),
        "final_train_losses": [fit.final_train_loss for fit in fits],
    }


def _cost_target_mean(
    batch: HierarchicalTrajectoryBatch,
    *,
    level: str,
    gamma: float,
    max_return_decisions: int | None,
) -> float:
    level_batch = batch.upper if str(level) == "upper" else batch.lower
    if level_batch.cost is None:
        raise RuntimeError(f"{level} trajectory lacks native cost labels")
    target = discounted_smdp_cost_returns(
        level_batch.cost,
        level_batch.duration,
        level_batch.done,
        gamma=float(gamma),
        max_decisions=max_return_decisions,
    )
    return float(np.mean(target, dtype=np.float64))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if not np.isfinite(denominator) or denominator <= 1e-12:
        raise RuntimeError("paired finite-difference direction is degenerate")
    return float(np.dot(left, right) / denominator)


def _paired_finite_difference_estimate(
    results: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    *,
    level: str,
    gamma: float,
    max_return_decisions: int | None,
    estimator: str = "coordinate_median_spsa",
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    if len(results) != len(jobs):
        raise ValueError("paired intervention results and jobs are misaligned")
    groups: dict[tuple[int, str], dict[str, tuple[Any, Any]]] = {}
    for result, job in zip(results, jobs, strict=True):
        path = job["path"]
        intervention = dict(job.get("intervention") or {})
        variant = str(result["intervention_variant"])
        key = (int(path["seed"]), str(path["disturbance_mode"]))
        groups.setdefault(key, {})[variant] = (result["batch"], intervention)
    gradients = []
    direction_rows = []
    directional_derivatives = []
    modes = []
    required = {"control", f"{level}_plus", f"{level}_minus"}
    for (_, mode), variants in sorted(groups.items()):
        if not required.issubset(variants):
            raise RuntimeError("paired intervention group is incomplete")
        plus_batch, plus_intervention = variants[f"{level}_plus"]
        minus_batch, minus_intervention = variants[f"{level}_minus"]
        plus_bias = np.asarray(
            plus_intervention[f"{level}_bias"], dtype=np.float64
        ).reshape(-1)
        minus_bias = np.asarray(
            minus_intervention[f"{level}_bias"], dtype=np.float64
        ).reshape(-1)
        if not np.allclose(plus_bias, -minus_bias, rtol=0.0, atol=1e-12):
            raise RuntimeError("paired actor interventions are not antithetic")
        amplitude = float(np.sqrt(np.mean(np.square(plus_bias))))
        if not np.isfinite(amplitude) or amplitude <= 0.0:
            raise RuntimeError("paired actor intervention has zero amplitude")
        plus_cost = _cost_target_mean(
            plus_batch,
            level=level,
            gamma=gamma,
            max_return_decisions=max_return_decisions,
        )
        minus_cost = _cost_target_mean(
            minus_batch,
            level=level,
            gamma=gamma,
            max_return_decisions=max_return_decisions,
        )
        unit_direction = plus_bias / amplitude
        directional_derivative = (
            plus_cost - minus_cost
        ) / (2.0 * amplitude)
        gradient = directional_derivative * unit_direction
        if not np.all(np.isfinite(gradient)):
            raise RuntimeError("paired finite-difference gradient is non-finite")
        gradients.append(gradient)
        direction_rows.append(unit_direction)
        directional_derivatives.append(directional_derivative)
        modes.append(str(mode))
    matrix = np.stack(gradients, axis=0)
    design = np.stack(direction_rows, axis=0)
    response = np.asarray(directional_derivatives, dtype=np.float64)
    modes_array = np.asarray(modes)
    estimator_name = str(estimator)
    design_metrics: dict[str, Any] = {}
    if estimator_name == "coordinate_median_spsa":
        direction = np.median(matrix, axis=0)
        per_mode = {
            mode: np.median(matrix[modes_array == mode], axis=0)
            for mode in sorted(set(modes))
        }
    elif estimator_name == "orthogonal_least_squares":
        direction, _, global_rank, _ = np.linalg.lstsq(
            design, response, rcond=None
        )
        per_mode = {}
        per_mode_rank = {}
        per_mode_condition = {}
        for mode in sorted(set(modes)):
            selected = modes_array == mode
            mode_design = design[selected]
            mode_response = response[selected]
            estimate, _, rank, _ = np.linalg.lstsq(
                mode_design, mode_response, rcond=None
            )
            per_mode[mode] = estimate
            per_mode_rank[mode] = int(rank)
            per_mode_condition[mode] = float(np.linalg.cond(mode_design))
        parameter_count = int(design.shape[1])
        if int(global_rank) != parameter_count or any(
            rank != parameter_count for rank in per_mode_rank.values()
        ):
            raise RuntimeError("paired orthogonal direction design is rank deficient")
        residual = response - design @ direction
        design_metrics = {
            "global_design_rank": int(global_rank),
            "global_design_condition": float(np.linalg.cond(design)),
            "global_directional_residual_rms": float(
                np.sqrt(np.mean(np.square(residual)))
            ),
            "per_mode_design_rank": per_mode_rank,
            "per_mode_design_condition": per_mode_condition,
        }
    else:
        raise ValueError("unknown paired finite-difference estimator")
    rms = float(np.sqrt(np.mean(np.square(direction))))
    if not np.isfinite(rms) or rms <= 1e-12:
        raise RuntimeError("robust paired finite-difference direction is degenerate")
    norms = np.linalg.norm(matrix, axis=1)
    return direction / rms, per_mode, {
        "estimator": estimator_name,
        "path_count": int(matrix.shape[0]),
        "parameter_count": int(matrix.shape[1]),
        "path_gradient_norm_median": float(np.median(norms)),
        "path_gradient_norm_maximum": float(np.max(norms)),
        "coordinate_gradient_iqr_mean": float(np.mean(
            np.quantile(matrix, 0.75, axis=0)
            - np.quantile(matrix, 0.25, axis=0)
        )),
        "disturbance_modes": sorted(set(modes)),
        **design_metrics,
    }


def _subsample_indices(size: int, limit: int, *, seed: int) -> np.ndarray:
    if int(size) <= int(limit):
        return np.arange(int(size), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(int(size), size=int(limit), replace=False))


def _level_actor_directions(
    model: Any,
    *,
    level: str,
    fits: list[_CriticFit],
    design_batch: LevelTrajectoryBatch,
    state_limit: int,
    sample_seed: int,
    actor_update_scope: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    actor = getattr(model, f"{level}_actor")
    parameters = _actor_update_parameters(
        actor, scope=str(actor_update_scope)
    )
    indices = _subsample_indices(
        design_batch.size, int(state_limit), seed=int(sample_seed)
    )
    state = torch.as_tensor(
        np.asarray(design_batch.state)[indices], dtype=torch.float32
    )
    cost_source = (
        design_batch.state
        if design_batch.cost_state is None else design_batch.cost_state
    )
    cost_state = torch.as_tensor(
        np.asarray(cost_source)[indices], dtype=torch.float32
    )
    scale = float(model.config.__dict__[f"{level}_deployment_frequency_action_scale"])
    directions = []
    losses = []
    for fit in fits:
        deployed_action = deterministic_actor_action(
            actor,
            state,
            transform=str(model.config.deployment_action_transform),
            scale=scale,
        )
        loss = torch.mean(_predict_normalized(fit, cost_state, deployed_action))
        gradients = torch.autograd.grad(
            loss, parameters, allow_unused=True
        )
        pieces = [
            (
                torch.zeros_like(parameter)
                if gradient is None else gradient
            ).detach().cpu().numpy().reshape(-1)
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ]
        vector = np.concatenate(pieces).astype(np.float64)
        rms = float(np.sqrt(np.mean(np.square(vector))))
        if not np.isfinite(rms) or rms <= 1e-12:
            raise RuntimeError(f"{level} action-cost actor gradient is degenerate")
        directions.append(vector / rms)
        losses.append(float(loss.detach().item()))
    cosines = []
    for left in range(len(directions)):
        for right in range(left + 1, len(directions)):
            cosine = float(
                np.dot(directions[left], directions[right])
                / (
                    np.linalg.norm(directions[left])
                    * np.linalg.norm(directions[right])
                )
            )
            cosines.append(cosine)
    direction = np.mean(np.stack(directions, axis=0), axis=0)
    rms = float(np.sqrt(np.mean(np.square(direction))))
    if not np.isfinite(rms) or rms <= 1e-12:
        raise RuntimeError(f"{level} ensemble actor gradient is degenerate")
    return direction / rms, {
        "state_count": int(indices.size),
        "parameter_count": int(direction.size),
        "ensemble_normalized_costs": losses,
        "pairwise_gradient_cosines": cosines,
        "median_gradient_cosine": float(np.median(cosines)),
        "minimum_gradient_cosine": float(np.min(cosines)),
    }


def _evaluate_deltas(
    *,
    checkpoint_path: Path,
    summary: dict[str, Any],
    paths: list[dict[str, Any]],
    deltas: list[np.ndarray],
    workers: int,
    episode_horizon: int,
    leakage_cost_mode: str,
    actor_update_scope: str,
    router_strengths: list[float] | None = None,
) -> list[dict[str, Any]]:
    baseline_router_strength = float(summary["lower_action_router_strength"])
    strengths = (
        [baseline_router_strength] * len(deltas)
        if router_strengths is None else list(map(float, router_strengths))
    )
    if len(strengths) != len(deltas) or not np.all(np.isfinite(strengths)):
        raise ValueError("actor deltas and router strengths must be aligned")
    jobs = [
        {
            "checkpoint_path": str(checkpoint_path.resolve()),
            "summary": summary,
            "path": path,
            "delta": delta,
            "candidate_index": candidate_index,
            "path_index": path_index,
            "episode_horizon": int(episode_horizon),
            "leakage_cost_mode": str(leakage_cost_mode),
            "actor_update_scope": str(actor_update_scope),
            "router_strength": strengths[candidate_index],
        }
        for candidate_index, delta in enumerate(deltas)
        for path_index, path in enumerate(paths)
    ]
    results = _parallel_map(_actor_delta_path_job, jobs, workers=workers)
    grouped = []
    for candidate_index in range(len(deltas)):
        selected = [
            result for result in results
            if result["candidate_index"] == candidate_index
        ]
        selected.sort(key=lambda result: result["path_index"])
        hashes = {result["parameter_sha256"] for result in selected}
        routed = {float(result["router_strength"]) for result in selected}
        if (
            len(selected) != len(paths)
            or len(hashes) != 1
            or routed != {strengths[candidate_index]}
        ):
            raise RuntimeError("parallel actor candidate evaluation is incomplete")
        grouped.append({
            "parameter_sha256": hashes.pop(),
            "router_strength": routed.pop(),
            "rows": [result["row"] for result in selected],
        })
    return grouped


def build_design_fold_contracts(
    baseline_rows: list[dict[str, Any]],
    fold_slices: list[slice],
    snapshot_factory: Any,
) -> tuple[list[Any], list[dict[str, Any]]]:
    snapshot_functions = []
    baselines = []
    for selected in fold_slices:
        fold_rows = baseline_rows[selected]
        if not fold_rows:
            raise ValueError("design folds must contain baseline paths")
        snapshot = snapshot_factory(fold_rows)
        snapshot_functions.append(snapshot)
        baselines.append(snapshot(fold_rows))
    return snapshot_functions, baselines


def run_probe(
    *,
    checkpoint_path: Path,
    summary_path: Path,
    output_path: Path,
    critic_train_roots: tuple[int, ...],
    critic_holdout_roots: tuple[int, ...],
    design_roots: tuple[int, ...],
    validation_roots: tuple[int, ...],
    critic_seeds: tuple[int, ...],
    critic_hidden_dim: int,
    critic_epochs: int,
    critic_minibatch_size: int,
    critic_learning_rate: float,
    critic_minimum_holdout_r2: float,
    critic_minimum_action_permutation_mse_increase: float,
    minimum_gradient_median_cosine: float,
    actor_state_limit: int,
    actor_step_rms_values: tuple[float, ...],
    minimum_reduction: float,
    funnel_multiplier: float,
    workers: int,
    risk_mode: str,
    cvar_alpha: float,
    episode_horizon: int,
    leakage_cost_mode: str,
    probe_version: str = PROBE_VERSION,
    upper_cost_return_horizon_decisions: int = 0,
    lower_cost_return_horizon_decisions: int = 0,
    critic_collection_mode: str = "stochastic_policy",
    critic_intervention_bias_rms: float = 0.0,
    actor_update_scope: str = "full_mean",
    actor_direction_source: str = "critic_gradient",
    minimum_paired_holdout_cosine: float = 0.0,
    critic_intervention_direction_scheme: str = "random_rademacher",
    critic_intervention_hadamard_order: int = 0,
    paired_direction_estimator: str = "coordinate_median_spsa",
    router_strength_values: tuple[float, ...] = (),
    design_fold_count: int = 1,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    raw_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary, _ = _v14_17_anchor_profile(raw_summary)
    baseline_model = _load_model(checkpoint)
    baseline_hash = _model_parameter_sha256(baseline_model)
    if baseline_hash != str(checkpoint["frozen_parameter_sha256"]):
        raise RuntimeError("action-cost probe did not reconstruct the frozen actor")
    role_roots = (
        critic_train_roots,
        critic_holdout_roots,
        design_roots,
        validation_roots,
    )
    flattened_roots = [root for role in role_roots for root in role]
    if len(flattened_roots) != len(set(flattened_roots)):
        raise ValueError("action-cost probe root roles overlap")
    router_registry = tuple(map(float, router_strength_values))
    if (
        len(router_registry) != len(set(router_registry))
        or any(not np.isfinite(value) or value < 0.0 or value > 1.0
               for value in router_registry)
    ):
        raise ValueError("router strength registry must be unique and in [0, 1]")
    fold_count = int(design_fold_count)
    if (
        fold_count < 1
        or fold_count > len(design_roots)
        or len(design_roots) % fold_count
    ):
        raise ValueError("design roots must divide evenly into nonempty folds")
    return_horizons = {
        "upper": int(upper_cost_return_horizon_decisions),
        "lower": int(lower_cost_return_horizon_decisions),
    }
    if any(value < 0 for value in return_horizons.values()):
        raise ValueError("action-cost return horizons cannot be negative")
    collection_mode = str(critic_collection_mode)
    if collection_mode not in {"stochastic_policy", "paired_output_bias"}:
        raise ValueError("unknown action-cost critic collection mode")
    intervention_rms = float(critic_intervention_bias_rms)
    if collection_mode == "stochastic_policy" and intervention_rms != 0.0:
        raise ValueError("stochastic critic collection cannot use interventions")
    if collection_mode == "paired_output_bias" and (
        not np.isfinite(intervention_rms) or intervention_rms <= 0.0
    ):
        raise ValueError("paired critic collection requires a positive bias RMS")
    update_scope = str(actor_update_scope)
    if update_scope not in {"full_mean", "output_bias"}:
        raise ValueError("unknown action-cost actor update scope")
    direction_source = str(actor_direction_source)
    if direction_source not in {"critic_gradient", "paired_finite_difference"}:
        raise ValueError("unknown action-cost actor direction source")
    if direction_source == "paired_finite_difference" and (
        collection_mode != "paired_output_bias" or update_scope != "output_bias"
    ):
        raise ValueError(
            "paired finite differences require paired collection and bias updates"
        )
    intervention_direction_scheme = str(critic_intervention_direction_scheme)
    if intervention_direction_scheme not in {
        "random_rademacher", "balanced_hadamard",
    }:
        raise ValueError("unknown paired intervention direction scheme")
    hadamard_order = int(critic_intervention_hadamard_order)
    if intervention_direction_scheme == "random_rademacher":
        if hadamard_order != 0:
            raise ValueError("random interventions cannot declare a Hadamard order")
    else:
        if collection_mode != "paired_output_bias":
            raise ValueError("balanced interventions require paired collection")
        dimensions = (
            int(_actor_output_head(baseline_model.upper_actor).out_features),
            int(_actor_output_head(baseline_model.lower_actor).out_features),
        )
        if (
            hadamard_order < max(dimensions)
            or hadamard_order & (hadamard_order - 1)
        ):
            raise ValueError("Hadamard order must be a power of two covering both actors")
    direction_estimator = str(paired_direction_estimator)
    if direction_estimator not in {
        "coordinate_median_spsa", "orthogonal_least_squares",
    }:
        raise ValueError("unknown paired finite-difference estimator")
    if direction_estimator == "orthogonal_least_squares" and (
        direction_source != "paired_finite_difference"
        or intervention_direction_scheme != "balanced_hadamard"
    ):
        raise ValueError(
            "orthogonal least squares requires paired Hadamard directions"
        )
    train_paths = _paths_for_roots(summary["environment"], critic_train_roots)
    holdout_paths = _paths_for_roots(
        summary["environment"], critic_holdout_roots
    )
    design_paths = _paths_for_roots(summary["environment"], design_roots)
    validation_paths = _paths_for_roots(
        summary["environment"], validation_roots
    )

    def trajectory_jobs(
        paths: list[dict[str, Any]],
        *,
        sample: bool,
        role: str,
        paired_interventions: bool = False,
    ) -> list[dict[str, Any]]:
        jobs = []
        mode_occurrences: dict[str, int] = {}
        for path in paths:
            mode = str(path["disturbance_mode"])
            direction_index = mode_occurrences.get(mode, 0)
            mode_occurrences[mode] = direction_index + 1
            if intervention_direction_scheme == "balanced_hadamard":
                direction_index %= hadamard_order
            intervention_seed = derive_seed(
                "mujoco_paired_output_bias_intervention_v1",
                str(summary["environment"]),
                str(role),
                int(path["seed"]),
                str(path["disturbance_mode"]),
            )
            interventions = (
                paired_output_bias_interventions(
                    baseline_model,
                    seed=intervention_seed,
                    bias_rms=intervention_rms,
                    direction_scheme=intervention_direction_scheme,
                    direction_index=direction_index,
                    hadamard_order=hadamard_order,
                )
                if paired_interventions else [None]
            )
            for intervention in interventions:
                variant = (
                    "stochastic_policy" if intervention is None
                    else str(intervention["variant"])
                )
                jobs.append({
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "summary": summary,
                    "path": path,
                    "sample": bool(sample),
                    "intervention": intervention,
                    "policy_seed": derive_seed(
                        "mujoco_action_cost_policy_sampling_v2",
                        str(summary["environment"]),
                        str(role),
                        int(path["seed"]),
                        str(variant),
                    ),
                    "episode_horizon": int(episode_horizon),
                    "leakage_cost_mode": str(leakage_cost_mode),
                })
        return jobs

    paired_collection = collection_mode == "paired_output_bias"
    train_jobs = trajectory_jobs(
        train_paths,
        sample=not paired_collection,
        role="critic_train",
        paired_interventions=paired_collection,
    )
    holdout_jobs = trajectory_jobs(
        holdout_paths,
        sample=not paired_collection,
        role="critic_holdout",
        paired_interventions=paired_collection,
    )
    critic_results = _parallel_map(
        _trajectory_job,
        train_jobs + holdout_jobs,
        workers=workers,
    )
    train_count = len(train_jobs)
    train_batches = [result["batch"] for result in critic_results[:train_count]]
    holdout_batches = [result["batch"] for result in critic_results[train_count:]]
    if any(result["parameter_sha256"] != baseline_hash for result in critic_results):
        raise RuntimeError("critic trajectory collection mutated the frozen actor")
    if paired_collection:
        if any(
            (
                result["rollout_parameter_sha256"] == baseline_hash
            ) != (result["intervention_variant"] == "control")
            for result in critic_results
        ):
            raise RuntimeError("paired actor interventions were not isolated")
    elif any(
        result["rollout_parameter_sha256"] != baseline_hash
        for result in critic_results
    ):
        raise RuntimeError("stochastic critic collection changed actor parameters")

    design_results = _parallel_map(
        _trajectory_job,
        trajectory_jobs(design_paths, sample=False, role="actor_design"),
        workers=workers,
    )
    if any(result["parameter_sha256"] != baseline_hash for result in design_results):
        raise RuntimeError("design trajectory collection mutated the frozen actor")
    design_batches = [result["batch"] for result in design_results]
    design_baseline_rows = [result["row"] for result in design_results]
    design_snapshot = _snapshot_fn(
        summary,
        design_baseline_rows,
        risk_mode=risk_mode,
        cvar_alpha=cvar_alpha,
    )
    design_baseline = design_snapshot(design_baseline_rows)
    paths_per_root = len(design_paths) // len(design_roots)
    roots_per_fold = len(design_roots) // fold_count
    design_fold_slices = [
        slice(
            fold * roots_per_fold * paths_per_root,
            (fold + 1) * roots_per_fold * paths_per_root,
        )
        for fold in range(fold_count)
    ]
    if sum(
        selected.stop - selected.start for selected in design_fold_slices
    ) != len(design_paths):
        raise RuntimeError("design fold construction did not cover every path")
    design_fold_snapshot_functions, design_fold_baselines = (
        build_design_fold_contracts(
            design_baseline_rows,
            design_fold_slices,
            lambda fold_rows: _snapshot_fn(
                summary,
                fold_rows,
                risk_mode=risk_mode,
                cvar_alpha=cvar_alpha,
            ),
        )
    )

    fits_by_level: dict[str, list[_CriticFit]] = {}
    critic_metrics: dict[str, dict[str, Any]] = {}
    for level in ("upper", "lower"):
        train_arrays = _level_arrays(
            train_batches,
            level=level,
            gamma=float(baseline_model.config.gamma),
            action_transform=str(
                baseline_model.config.deployment_action_transform
            ),
            action_scale=float(getattr(
                baseline_model.config,
                f"{level}_deployment_frequency_action_scale",
            )),
            max_return_decisions=(
                None if return_horizons[level] == 0
                else return_horizons[level]
            ),
        )
        holdout_arrays = _level_arrays(
            holdout_batches,
            level=level,
            gamma=float(baseline_model.config.gamma),
            action_transform=str(
                baseline_model.config.deployment_action_transform
            ),
            action_scale=float(getattr(
                baseline_model.config,
                f"{level}_deployment_frequency_action_scale",
            )),
            max_return_decisions=(
                None if return_horizons[level] == 0
                else return_horizons[level]
            ),
        )
        fits = [
            _fit_critic(
                train_arrays,
                seed=int(seed),
                hidden_dim=critic_hidden_dim,
                epochs=critic_epochs,
                minibatch_size=critic_minibatch_size,
                learning_rate=critic_learning_rate,
            )
            for seed in critic_seeds
        ]
        metrics = _critic_metrics(
            fits,
            holdout_arrays,
            permutation_seed=derive_seed(
                "mujoco_v14_22_holdout_action_permutation_v1",
                str(summary["environment"]),
                str(level),
            ),
        )
        metrics["train_transition_count"] = int(train_arrays[2].size)
        fits_by_level[level] = fits
        critic_metrics[level] = metrics

    gradient_metrics: dict[str, Any] = {}
    directions = []
    gradient_error = None
    try:
        if direction_source == "critic_gradient":
            merged_design = concat_hierarchical_batches(design_batches)
            for level, batch in (
                ("upper", merged_design.upper),
                ("lower", merged_design.lower),
            ):
                direction, metrics = _level_actor_directions(
                    baseline_model,
                    level=level,
                    fits=fits_by_level[level],
                    design_batch=batch,
                    state_limit=actor_state_limit,
                    sample_seed=derive_seed(
                        "mujoco_v14_22_actor_state_sample_v1",
                        str(summary["environment"]),
                        str(level),
                    ),
                    actor_update_scope=update_scope,
                )
                directions.append(direction)
                gradient_metrics[level] = metrics
        else:
            for level in ("upper", "lower"):
                horizon = (
                    None if return_horizons[level] == 0
                    else return_horizons[level]
                )
                train_direction, train_modes, train_metrics = (
                    _paired_finite_difference_estimate(
                        critic_results[:train_count],
                        train_jobs,
                        level=level,
                        gamma=float(baseline_model.config.gamma),
                        max_return_decisions=horizon,
                        estimator=direction_estimator,
                    )
                )
                holdout_direction, holdout_modes, holdout_metrics = (
                    _paired_finite_difference_estimate(
                        critic_results[train_count:],
                        holdout_jobs,
                        level=level,
                        gamma=float(baseline_model.config.gamma),
                        max_return_decisions=horizon,
                        estimator=direction_estimator,
                    )
                )
                if set(train_modes) != set(holdout_modes):
                    raise RuntimeError(
                        "paired finite-difference disturbance modes are misaligned"
                    )
                mode_cosines = {
                    mode: _cosine(train_modes[mode], holdout_modes[mode])
                    for mode in sorted(train_modes)
                }
                directions.append(train_direction)
                gradient_metrics[level] = {
                    "train": train_metrics,
                    "holdout": holdout_metrics,
                    "holdout_direction_cosine": _cosine(
                        train_direction, holdout_direction
                    ),
                    "holdout_mode_direction_cosines": mode_cosines,
                    "minimum_holdout_mode_direction_cosine": float(
                        min(mode_cosines.values())
                    ),
                }
    except RuntimeError as exc:
        gradient_error = str(exc)
    critic_gate = bool(
        gradient_error is None
        and all(
            float(critic_metrics[level]["ensemble_holdout_r2"])
            > float(critic_minimum_holdout_r2)
            for level in ("upper", "lower")
        )
        and all(
            float(critic_metrics[level][
                "action_permutation_mse_increase_fraction"
            ]) > float(critic_minimum_action_permutation_mse_increase)
            for level in ("upper", "lower")
        )
        and (
            all(
                float(gradient_metrics[level]["median_gradient_cosine"])
                > float(minimum_gradient_median_cosine)
                for level in ("upper", "lower")
            )
            if direction_source == "critic_gradient"
            else all(
                float(gradient_metrics[level]["holdout_direction_cosine"])
                > float(minimum_paired_holdout_cosine)
                and float(gradient_metrics[level][
                    "minimum_holdout_mode_direction_cosine"
                ]) > float(minimum_paired_holdout_cosine)
                for level in ("upper", "lower")
            )
        )
    )
    candidates = []
    eligible: list[dict[str, Any]] = []
    selected = None
    joint_direction = None
    baseline_router_strength = float(summary["lower_action_router_strength"])
    candidate_specs: list[dict[str, Any]] = []
    if critic_gate:
        joint_direction = np.concatenate(directions)
        direction_rms = float(np.sqrt(np.mean(np.square(joint_direction))))
        joint_direction /= direction_rms
        for step_rms in actor_step_rms_values:
            candidate_specs.append({
                "source": (
                    str(direction_source)
                    if direction_source == "critic_gradient"
                    else f"{direction_source}:{direction_estimator}"
                ),
                "step_rms": float(step_rms),
                "router_strength": baseline_router_strength,
                "delta": -float(step_rms) * joint_direction,
            })
    zero_delta = np.zeros(
        actor_update_parameter_vector(baseline_model, scope=update_scope).size,
        dtype=np.float64,
    )
    for router_strength in router_registry:
        candidate_specs.append({
            "source": "function_preserving_router_adapter",
            "step_rms": 0.0,
            "router_strength": float(router_strength),
            "delta": zero_delta.copy(),
        })
    if candidate_specs:
        evaluated = _evaluate_deltas(
            checkpoint_path=checkpoint_path,
            summary=summary,
            paths=design_paths,
            deltas=[specification["delta"] for specification in candidate_specs],
            workers=workers,
            episode_horizon=episode_horizon,
            leakage_cost_mode=leakage_cost_mode,
            actor_update_scope=update_scope,
            router_strengths=[
                specification["router_strength"]
                for specification in candidate_specs
            ],
        )
        for specification, result in zip(
            candidate_specs, evaluated, strict=True
        ):
            snapshot = design_snapshot(result["rows"])
            fold_snapshots = [
                fold_snapshot(result["rows"][selected_slice])
                for fold_snapshot, selected_slice in zip(
                    design_fold_snapshot_functions,
                    design_fold_slices,
                    strict=True,
                )
            ]
            requires_trace_invariance = bool(
                specification["source"]
                == "function_preserving_router_adapter"
            )
            candidate = {
                "source": str(specification["source"]),
                "step_rms": float(specification["step_rms"]),
                "router_strength": float(result["router_strength"]),
                "parameter_sha256": result["parameter_sha256"],
                "snapshot": snapshot,
                "fold_snapshots": fold_snapshots,
                "requires_trace_invariance": requires_trace_invariance,
                "trace_invariance": (
                    paired_trace_invariance_diagnostics(
                        result["rows"], design_baseline_rows
                    )
                    if requires_trace_invariance else None
                ),
                "selection_priority": [
                    float(specification["step_rms"]),
                    float(result["router_strength"]),
                ],
                "_delta": specification["delta"],
            }
            candidates.append(candidate)
        decision = select_guarded_restoration_portfolio(
            candidates,
            baseline=design_baseline,
            fold_baselines=design_fold_baselines,
            minimum_reduction=minimum_reduction,
            funnel_multiplier=funnel_multiplier,
        )
        for index, candidate in enumerate(candidates):
            candidate["design_fold_snapshots"] = candidate.pop(
                "fold_snapshots"
            )
            candidate["design_fold_eligible"] = list(
                decision.fold_eligibility[index]
            )
            candidate["trace_invariance_eligible"] = bool(
                decision.trace_invariance_eligibility[index]
            )
            candidate["design_eligible"] = bool(
                decision.design_eligibility[index]
            )
        eligible = [candidates[index] for index in decision.eligible_indices]
        selected = (
            None
            if decision.selected_index is None
            else candidates[decision.selected_index]
        )

    validation_baseline = None
    validation_candidate = None
    validation_trace_invariance = None
    validation_supported = False
    if selected is not None:
        validation_results = _evaluate_deltas(
            checkpoint_path=checkpoint_path,
            summary=summary,
            paths=validation_paths,
            deltas=[
                np.zeros_like(np.asarray(selected["_delta"])),
                np.asarray(selected["_delta"]),
            ],
            workers=workers,
            episode_horizon=episode_horizon,
            leakage_cost_mode=leakage_cost_mode,
            actor_update_scope=update_scope,
            router_strengths=[
                baseline_router_strength,
                float(selected["router_strength"]),
            ],
        )
        if validation_results[0]["parameter_sha256"] != baseline_hash:
            raise RuntimeError("validation baseline changed the frozen actor")
        if validation_results[1]["parameter_sha256"] != str(
            selected["parameter_sha256"]
        ):
            raise RuntimeError("validation actor delta was not reconstructed")
        if float(validation_results[1]["router_strength"]) != float(
            selected["router_strength"]
        ):
            raise RuntimeError("validation router strength was not reconstructed")
        validation_baseline_rows = validation_results[0]["rows"]
        validation_snapshot = _snapshot_fn(
            summary,
            validation_baseline_rows,
            risk_mode=risk_mode,
            cvar_alpha=cvar_alpha,
        )
        validation_baseline = validation_snapshot(validation_baseline_rows)
        validation_candidate = validation_snapshot(
            validation_results[1]["rows"]
        )
        if bool(selected["requires_trace_invariance"]):
            validation_trace_invariance = paired_trace_invariance_diagnostics(
                validation_results[1]["rows"],
                validation_baseline_rows,
            )
        validation_supported = restoration_snapshot_eligible(
            validation_candidate,
            validation_baseline,
            minimum_reduction=minimum_reduction,
            funnel_multiplier=funnel_multiplier,
        ) and bool(
            not selected["requires_trace_invariance"]
            or validation_trace_invariance["all_traces_invariant"]
        )

    public_candidates = [
        {key: value for key, value in candidate.items() if key != "_delta"}
        for candidate in candidates
    ]
    public_selected = None
    if selected is not None:
        selected_index = next(
            index for index, candidate in enumerate(candidates)
            if candidate is selected
        )
        public_selected = public_candidates[selected_index]
    payload = {
        "probe_version": str(probe_version),
        "checkpoint": str(checkpoint_path),
        "summary": str(summary_path),
        "environment": str(summary["environment"]),
        "optimizer_seed": int(summary["optimizer_seed"]),
        "baseline_parameter_sha256": baseline_hash,
        "actor_mean_parameter_count": int(
            actor_mean_parameter_vector(baseline_model).size
        ),
        "actor_update_scope": update_scope,
        "actor_direction_source": direction_source,
        "minimum_paired_holdout_cosine": float(
            minimum_paired_holdout_cosine
        ),
        "baseline_router_strength": baseline_router_strength,
        "router_strength_values": list(router_registry),
        "design_fold_count": fold_count,
        "actor_update_parameter_count": int(
            actor_update_parameter_vector(
                baseline_model, scope=update_scope
            ).size
        ),
        "critic_train_roots": list(critic_train_roots),
        "critic_holdout_roots": list(critic_holdout_roots),
        "design_roots": list(design_roots),
        "validation_roots": list(validation_roots),
        "critic_train_base_path_count": len(train_paths),
        "critic_holdout_base_path_count": len(holdout_paths),
        "critic_train_path_count": len(train_jobs),
        "critic_holdout_path_count": len(holdout_jobs),
        "design_path_count": len(design_paths),
        "validation_path_count": len(validation_paths),
        "critic_seeds": list(critic_seeds),
        "critic_hidden_dim": int(critic_hidden_dim),
        "critic_epochs": int(critic_epochs),
        "critic_minibatch_size": int(critic_minibatch_size),
        "critic_learning_rate": float(critic_learning_rate),
        "critic_collection_mode": collection_mode,
        "critic_intervention_bias_rms": intervention_rms,
        "critic_intervention_direction_scheme": intervention_direction_scheme,
        "critic_intervention_hadamard_order": hadamard_order,
        "paired_direction_estimator": direction_estimator,
        "critic_intervention_variants": sorted({
            result["intervention_variant"] for result in critic_results
        }),
        "cost_return_horizon_decisions": return_horizons,
        "critic_minimum_holdout_r2": float(critic_minimum_holdout_r2),
        "critic_minimum_action_permutation_mse_increase": float(
            critic_minimum_action_permutation_mse_increase
        ),
        "minimum_gradient_median_cosine": float(
            minimum_gradient_median_cosine
        ),
        "actor_state_limit": int(actor_state_limit),
        "actor_step_rms_values": list(actor_step_rms_values),
        "minimum_reduction": float(minimum_reduction),
        "funnel_multiplier": float(funnel_multiplier),
        "workers": int(workers),
        "risk_mode": str(risk_mode),
        "cvar_alpha": float(cvar_alpha),
        "critic_metrics": critic_metrics,
        "gradient_metrics": gradient_metrics,
        "gradient_error": gradient_error,
        "critic_gate_pass": bool(critic_gate),
        "design_baseline": design_baseline,
        "design_fold_baselines": design_fold_baselines,
        "candidate_count": len(candidates),
        "design_eligible_candidate_count": len(eligible),
        "selected_design_candidate": public_selected,
        "validation_baseline": validation_baseline,
        "validation_candidate": validation_candidate,
        "validation_trace_invariance": validation_trace_invariance,
        "validation_supported": bool(validation_supported),
        "candidates": public_candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _integers(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)) or any(item < 0 for item in values):
        raise ValueError("integer registry must be unique and non-negative")
    return values


def _positive_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)) or any(
        not np.isfinite(item) or item <= 0.0 for item in values
    ):
        raise ValueError("float registry must be unique, positive, and finite")
    return values


def _unit_interval_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(values) != len(set(values)) or any(
        not np.isfinite(item) or item < 0.0 or item > 1.0
        for item in values
    ):
        raise ValueError("router strengths must be unique, finite, and in [0, 1]")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--critic-train-roots", required=True)
    parser.add_argument("--critic-holdout-roots", required=True)
    parser.add_argument("--design-roots", required=True)
    parser.add_argument("--validation-roots", required=True)
    parser.add_argument("--critic-seeds", required=True)
    parser.add_argument("--critic-hidden-dim", type=int, required=True)
    parser.add_argument("--critic-epochs", type=int, required=True)
    parser.add_argument("--critic-minibatch-size", type=int, required=True)
    parser.add_argument("--critic-learning-rate", type=float, required=True)
    parser.add_argument("--critic-minimum-holdout-r2", type=float, required=True)
    parser.add_argument(
        "--critic-minimum-action-permutation-mse-increase",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--minimum-gradient-median-cosine", type=float, required=True
    )
    parser.add_argument("--actor-state-limit", type=int, required=True)
    parser.add_argument("--actor-step-rms-values", required=True)
    parser.add_argument("--minimum-reduction", type=float, required=True)
    parser.add_argument("--funnel-multiplier", type=float, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--risk-mode", required=True)
    parser.add_argument("--cvar-alpha", type=float, required=True)
    parser.add_argument("--episode-horizon", type=int, required=True)
    parser.add_argument("--leakage-cost-mode", required=True)
    parser.add_argument("--probe-version", default=PROBE_VERSION)
    parser.add_argument(
        "--upper-cost-return-horizon-decisions", type=int, default=0
    )
    parser.add_argument(
        "--lower-cost-return-horizon-decisions", type=int, default=0
    )
    parser.add_argument(
        "--critic-collection-mode",
        choices=("stochastic_policy", "paired_output_bias"),
        default="stochastic_policy",
    )
    parser.add_argument(
        "--critic-intervention-bias-rms", type=float, default=0.0
    )
    parser.add_argument(
        "--critic-intervention-direction-scheme",
        choices=("random_rademacher", "balanced_hadamard"),
        default="random_rademacher",
    )
    parser.add_argument(
        "--critic-intervention-hadamard-order", type=int, default=0
    )
    parser.add_argument(
        "--actor-update-scope",
        choices=("full_mean", "output_bias"),
        default="full_mean",
    )
    parser.add_argument(
        "--actor-direction-source",
        choices=("critic_gradient", "paired_finite_difference"),
        default="critic_gradient",
    )
    parser.add_argument(
        "--minimum-paired-holdout-cosine", type=float, default=0.0
    )
    parser.add_argument(
        "--paired-direction-estimator",
        choices=("coordinate_median_spsa", "orthogonal_least_squares"),
        default="coordinate_median_spsa",
    )
    parser.add_argument("--router-strength-values", default="")
    parser.add_argument("--design-fold-count", type=int, default=1)
    args = parser.parse_args()
    payload = run_probe(
        checkpoint_path=args.checkpoint,
        summary_path=args.summary,
        output_path=args.output,
        critic_train_roots=_integers(args.critic_train_roots),
        critic_holdout_roots=_integers(args.critic_holdout_roots),
        design_roots=_integers(args.design_roots),
        validation_roots=_integers(args.validation_roots),
        critic_seeds=_integers(args.critic_seeds),
        critic_hidden_dim=args.critic_hidden_dim,
        critic_epochs=args.critic_epochs,
        critic_minibatch_size=args.critic_minibatch_size,
        critic_learning_rate=args.critic_learning_rate,
        critic_minimum_holdout_r2=args.critic_minimum_holdout_r2,
        critic_minimum_action_permutation_mse_increase=(
            args.critic_minimum_action_permutation_mse_increase
        ),
        minimum_gradient_median_cosine=args.minimum_gradient_median_cosine,
        actor_state_limit=args.actor_state_limit,
        actor_step_rms_values=_positive_floats(args.actor_step_rms_values),
        minimum_reduction=args.minimum_reduction,
        funnel_multiplier=args.funnel_multiplier,
        workers=args.workers,
        risk_mode=args.risk_mode,
        cvar_alpha=args.cvar_alpha,
        episode_horizon=args.episode_horizon,
        leakage_cost_mode=args.leakage_cost_mode,
        probe_version=args.probe_version,
        upper_cost_return_horizon_decisions=(
            args.upper_cost_return_horizon_decisions
        ),
        lower_cost_return_horizon_decisions=(
            args.lower_cost_return_horizon_decisions
        ),
        critic_collection_mode=args.critic_collection_mode,
        critic_intervention_bias_rms=args.critic_intervention_bias_rms,
        actor_update_scope=args.actor_update_scope,
        actor_direction_source=args.actor_direction_source,
        minimum_paired_holdout_cosine=args.minimum_paired_holdout_cosine,
        critic_intervention_direction_scheme=(
            args.critic_intervention_direction_scheme
        ),
        critic_intervention_hadamard_order=(
            args.critic_intervention_hadamard_order
        ),
        paired_direction_estimator=args.paired_direction_estimator,
        router_strength_values=_unit_interval_floats(
            args.router_strength_values
        ),
        design_fold_count=args.design_fold_count,
    )
    print(json.dumps({
        "output": str(args.output),
        "critic_gate_pass": payload["critic_gate_pass"],
        "design_eligible_candidate_count": payload[
            "design_eligible_candidate_count"
        ],
        "validation_supported": payload["validation_supported"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
