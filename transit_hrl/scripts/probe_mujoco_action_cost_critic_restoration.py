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
from freq_hrl.rl.deployment_frequency import deterministic_actor_action
from freq_hrl.rl.smdp_actor_critic import (
    HierarchicalTrajectoryBatch,
    LevelTrajectoryBatch,
    concat_hierarchical_batches,
)
from scripts import mujoco_v14_17_native_pd_cvar_screen_spec as v14_17_spec
from scripts.probe_mujoco_radial_restoration import (
    _evaluate_rows,
    _load_model,
    _v14_17_anchor_profile,
)
from scripts.probe_mujoco_zeroth_order_actor_restoration import (
    _eligible,
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


def _actor_mean_parameters(actor: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [
        parameter
        for name, parameter in actor.named_parameters()
        if name != "log_std" and not name.endswith(".log_std")
    ]


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
        "parameter_sha256": _model_parameter_sha256(model),
    }


def _actor_delta_path_job(job: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    checkpoint = torch.load(
        job["checkpoint_path"], map_location="cpu", weights_only=False
    )
    model = _load_model(checkpoint)
    apply_actor_mean_parameter_delta(
        model, np.asarray(job["delta"], dtype=np.float64)
    )
    row = _evaluate_rows(
        model,
        summary=job["summary"],
        paths=[job["path"]],
        episode_horizon=int(job["episode_horizon"]),
        leakage_cost_mode=str(job["leakage_cost_mode"]),
        router_strength=float(job["summary"]["lower_action_router_strength"]),
    )[0]
    return {
        "candidate_index": int(job["candidate_index"]),
        "path_index": int(job["path_index"]),
        "parameter_sha256": _model_parameter_sha256(model),
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
) -> tuple[np.ndarray, dict[str, Any]]:
    actor = getattr(model, f"{level}_actor")
    parameters = _actor_mean_parameters(actor)
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
) -> list[dict[str, Any]]:
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
        if len(selected) != len(paths) or len(hashes) != 1:
            raise RuntimeError("parallel actor candidate evaluation is incomplete")
        grouped.append({
            "parameter_sha256": hashes.pop(),
            "rows": [result["row"] for result in selected],
        })
    return grouped


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
    train_paths = _paths_for_roots(summary["environment"], critic_train_roots)
    holdout_paths = _paths_for_roots(
        summary["environment"], critic_holdout_roots
    )
    design_paths = _paths_for_roots(summary["environment"], design_roots)
    validation_paths = _paths_for_roots(
        summary["environment"], validation_roots
    )

    def trajectory_jobs(
        paths: list[dict[str, Any]], *, sample: bool, role: str
    ) -> list[dict[str, Any]]:
        return [{
            "checkpoint_path": str(checkpoint_path.resolve()),
            "summary": summary,
            "path": path,
            "sample": bool(sample),
            "policy_seed": derive_seed(
                "mujoco_v14_22_policy_sampling_v1",
                str(summary["environment"]),
                str(role),
                int(path["seed"]),
            ),
            "episode_horizon": int(episode_horizon),
            "leakage_cost_mode": str(leakage_cost_mode),
        } for path in paths]

    critic_results = _parallel_map(
        _trajectory_job,
        trajectory_jobs(train_paths, sample=True, role="critic_train")
        + trajectory_jobs(holdout_paths, sample=True, role="critic_holdout"),
        workers=workers,
    )
    train_count = len(train_paths)
    train_batches = [result["batch"] for result in critic_results[:train_count]]
    holdout_batches = [result["batch"] for result in critic_results[train_count:]]
    if any(result["parameter_sha256"] != baseline_hash for result in critic_results):
        raise RuntimeError("critic trajectory collection mutated the frozen actor")

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

    merged_design = concat_hierarchical_batches(design_batches)
    gradient_metrics: dict[str, Any] = {}
    directions = []
    gradient_error = None
    try:
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
            )
            directions.append(direction)
            gradient_metrics[level] = metrics
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
        and all(
            float(gradient_metrics[level]["median_gradient_cosine"])
            > float(minimum_gradient_median_cosine)
            for level in ("upper", "lower")
        )
    )
    candidates = []
    eligible: list[dict[str, Any]] = []
    selected = None
    joint_direction = None
    if critic_gate:
        joint_direction = np.concatenate(directions)
        direction_rms = float(np.sqrt(np.mean(np.square(joint_direction))))
        joint_direction /= direction_rms
        deltas = [
            -float(step_rms) * joint_direction
            for step_rms in actor_step_rms_values
        ]
        evaluated = _evaluate_deltas(
            checkpoint_path=checkpoint_path,
            summary=summary,
            paths=design_paths,
            deltas=deltas,
            workers=workers,
            episode_horizon=episode_horizon,
            leakage_cost_mode=leakage_cost_mode,
        )
        for step_rms, delta, result in zip(
            actor_step_rms_values, deltas, evaluated, strict=True
        ):
            snapshot = design_snapshot(result["rows"])
            candidate = {
                "source": "ensemble_action_cost_gradient",
                "step_rms": float(step_rms),
                "parameter_sha256": result["parameter_sha256"],
                "snapshot": snapshot,
                "_delta": delta,
            }
            candidate["design_eligible"] = _eligible(
                snapshot,
                design_baseline,
                minimum_reduction=minimum_reduction,
                funnel_multiplier=funnel_multiplier,
            )
            candidates.append(candidate)
        eligible = [
            candidate for candidate in candidates
            if candidate["design_eligible"]
        ]
        eligible.sort(key=lambda candidate: (
            float(candidate["snapshot"]["reward_violation_count"]),
            float(candidate["snapshot"]["frequency_violation_merit"]),
            float(candidate["snapshot"]["worst_frequency_violation"]),
            float(candidate["step_rms"]),
        ))
        selected = eligible[0] if eligible else None

    validation_baseline = None
    validation_candidate = None
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
        )
        if validation_results[0]["parameter_sha256"] != baseline_hash:
            raise RuntimeError("validation baseline changed the frozen actor")
        if validation_results[1]["parameter_sha256"] != str(
            selected["parameter_sha256"]
        ):
            raise RuntimeError("validation actor delta was not reconstructed")
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
        validation_supported = _eligible(
            validation_candidate,
            validation_baseline,
            minimum_reduction=minimum_reduction,
            funnel_multiplier=funnel_multiplier,
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
        "probe_version": PROBE_VERSION,
        "checkpoint": str(checkpoint_path),
        "summary": str(summary_path),
        "environment": str(summary["environment"]),
        "optimizer_seed": int(summary["optimizer_seed"]),
        "baseline_parameter_sha256": baseline_hash,
        "actor_mean_parameter_count": int(
            actor_mean_parameter_vector(baseline_model).size
        ),
        "critic_train_roots": list(critic_train_roots),
        "critic_holdout_roots": list(critic_holdout_roots),
        "design_roots": list(design_roots),
        "validation_roots": list(validation_roots),
        "critic_train_path_count": len(train_paths),
        "critic_holdout_path_count": len(holdout_paths),
        "design_path_count": len(design_paths),
        "validation_path_count": len(validation_paths),
        "critic_seeds": list(critic_seeds),
        "critic_hidden_dim": int(critic_hidden_dim),
        "critic_epochs": int(critic_epochs),
        "critic_minibatch_size": int(critic_minibatch_size),
        "critic_learning_rate": float(critic_learning_rate),
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
        "candidate_count": len(candidates),
        "design_eligible_candidate_count": len(eligible),
        "selected_design_candidate": public_selected,
        "validation_baseline": validation_baseline,
        "validation_candidate": validation_candidate,
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
