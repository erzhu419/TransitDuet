#!/usr/bin/env python3
"""Run the frozen v18.2 grouped state-conditioned actor screen."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.mujoco import (  # noqa: E402
    full_horizon_responsibility_oracle as oracle,
)
from freq_hrl.experiments.mujoco.state_conditioned_actor import (  # noqa: E402
    apply_state_conditioned_actor,
    fit_state_conditioned_actor,
)
from freq_hrl.experiments.reproducibility import (  # noqa: E402
    derive_seed,
    verify_current_freq_hrl_source_identity,
)
from scripts import mujoco_v18_1_state_actor_dataset_spec as v18_1  # noqa: E402
from scripts import (  # noqa: E402
    mujoco_v18_2_state_conditioned_actor_spec as spec,
)
from scripts.export_mujoco_v18_1_state_actor_dataset_path import (  # noqa: E402
    validate_state_trace,
)
from scripts.train_mujoco_v17_8_causal_fir import (  # noqa: E402
    load_reused_panel,
)


def state_artifact_path(
    state_root: Path, environment: str, mode: str, seed: int
) -> Path:
    return (
        Path(state_root) / str(environment) / str(mode)
        / f"seed_{int(seed)}" / "actor_state_path.npz"
    )


def target_artifact_path(target_root: Path, row: dict[str, Any]) -> Path:
    return (
        Path(target_root) / str(row["environment"])
        / str(row["disturbance_mode"])
        / f"seed_{int(row['evaluation_seed'])}"
        / "nearest_feasible_target.npz"
    )


def load_state_actor_panel(
    state_root: Path,
    reference_root: Path,
    target_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_rows = load_reused_panel(Path(reference_root))
    panel: list[dict[str, Any]] = []
    expected_state_keys = set(v18_1.TRACE_KEYS)
    for source in reference_rows:
        row = dict(source)
        path = state_artifact_path(
            state_root,
            str(row["environment"]),
            str(row["disturbance_mode"]),
            int(row["evaluation_seed"]),
        )
        if not path.is_file():
            raise FileNotFoundError(f"missing v18.1 state path: {path}")
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != expected_state_keys:
                raise ValueError(f"invalid v18.1 state keys: {path}")
            state_arrays = {
                key: np.asarray(arrays[key]).copy()
                for key in arrays.files
            }
        trace_summary = validate_state_trace(
            str(row["environment"]),
            {
                "total_action": state_arrays["total_action"],
                "upper_action": state_arrays["baseline_upper_action"],
                "lower_action": state_arrays["baseline_lower_action"],
                "executed_action": state_arrays["executed_action"],
            },
            {
                "observation": state_arrays["observation"],
                "lower_policy_state": state_arrays["lower_policy_state"],
                "disturbance": state_arrays["disturbance"],
                "upper_policy_action": state_arrays["upper_policy_action"],
                "latent_lower_action": state_arrays["latent_lower_action"],
                "upper_decision": state_arrays["upper_decision"],
                "episode_step": state_arrays["episode_step"],
            },
        )
        total = np.asarray(row["total_action"], dtype=np.float64)
        state_total = np.asarray(
            state_arrays["total_action"], dtype=np.float64
        )
        state_upper = np.asarray(
            state_arrays["baseline_upper_action"], dtype=np.float64
        )
        if (
            state_total.shape != total.shape
            or float(np.max(np.abs(state_total - total)))
            > spec.STATE_TRACE_ALIGNMENT_TOLERANCE
            or float(np.max(np.abs(
                state_upper
                - np.asarray(row["baseline_upper_action"], dtype=np.float64)
            )))
            > spec.STATE_TRACE_ALIGNMENT_TOLERANCE
        ):
            raise ValueError(f"v18.1 state path does not align to v17.8: {path}")
        row.update({
            "lower_policy_state": np.asarray(
                state_arrays["lower_policy_state"], dtype=np.float64
            ),
            "state_trace_summary": trace_summary,
        })
        target_path = target_artifact_path(target_root, row)
        actor_floor = not bool(row["oracle_joint_feasible"])
        if actor_floor:
            if not target_path.is_file():
                raise FileNotFoundError(
                    f"missing v17.12 actor target: {target_path}"
                )
            with np.load(target_path, allow_pickle=False) as target_arrays:
                expected_target_keys = {
                    "reference_total_action",
                    "target_upper_action",
                    "target_lower_action",
                    "target_total_action",
                    "total_action_correction",
                }
                if set(target_arrays.files) != expected_target_keys:
                    raise ValueError(f"invalid v17.12 target keys: {target_path}")
                reference_total = np.asarray(
                    target_arrays["reference_total_action"], dtype=np.float64
                ).copy()
                target_upper = np.asarray(
                    target_arrays["target_upper_action"], dtype=np.float64
                ).copy()
                target_lower = np.asarray(
                    target_arrays["target_lower_action"], dtype=np.float64
                ).copy()
                target_total = np.asarray(
                    target_arrays["target_total_action"], dtype=np.float64
                ).copy()
                target_correction = np.asarray(
                    target_arrays["total_action_correction"], dtype=np.float64
                ).copy()
            if any(
                value.shape != total.shape or not np.all(np.isfinite(value))
                for value in (
                    reference_total,
                    target_upper,
                    target_lower,
                    target_total,
                    target_correction,
                )
            ):
                raise ValueError(f"invalid v17.12 target arrays: {target_path}")
            if (
                np.max(np.abs(reference_total - total))
                > spec.TARGET_ALIGNMENT_TOLERANCE
                or np.max(np.abs(target_upper + target_lower - target_total))
                > spec.TARGET_ALIGNMENT_TOLERANCE
                or np.max(np.abs(target_total - total - target_correction))
                > spec.TARGET_ALIGNMENT_TOLERANCE
            ):
                raise ValueError(f"misaligned v17.12 target: {target_path}")
        else:
            if target_path.exists():
                raise ValueError(
                    f"unexpected target on reference-feasible path: {target_path}"
                )
            target_correction = np.zeros_like(total)
        target_executed = (
            np.clip(
                total + target_correction,
                -spec.EXECUTED_ACTION_LIMIT,
                spec.EXECUTED_ACTION_LIMIT,
            )
            - np.clip(
                total,
                -spec.EXECUTED_ACTION_LIMIT,
                spec.EXECUTED_ACTION_LIMIT,
            )
        )
        row.update({
            "actor_floor": actor_floor,
            "target_total_correction": target_correction,
            "target_correction_rms": _rms(target_correction),
            "target_executed_correction_rms": _rms(target_executed),
        })
        panel.append(row)
    floor = [row for row in panel if row["actor_floor"]]
    floor_by_seed = {
        str(seed): sum(
            int(row["evaluation_seed"]) == int(seed) for row in floor
        )
        for seed in spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT_BY_SEED
    }
    expected_floor_by_seed = {
        str(seed): int(count)
        for seed, count in spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT_BY_SEED.items()
    }
    if (
        len(panel) != spec.EXPECTED_PATH_COUNT
        or len(floor) != spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT
        or any(
            row["environment"] != spec.EXPECTED_ACTOR_FLOOR_ENVIRONMENT
            for row in floor
        )
        or floor_by_seed != expected_floor_by_seed
    ):
        raise RuntimeError("v18.2 state/target panel count mismatch")
    validation = {
        "path_count": len(panel),
        "reference_feasible_path_count": sum(
            bool(row["oracle_joint_feasible"]) for row in panel
        ),
        "actor_floor_path_count": len(floor),
        "actor_floor_path_count_by_seed": floor_by_seed,
        "state_path_count_by_environment": {
            environment: sum(
                row["environment"] == environment for row in panel
            )
            for environment in spec.ENVIRONMENTS
        },
        "trajectory_step_count_by_environment": {
            environment: sum(
                int(row["state_trace_summary"]["trajectory_length"])
                for row in panel if row["environment"] == environment
            )
            for environment in spec.ENVIRONMENTS
        },
        "target_labels_used_only_as_training_outputs": True,
    }
    return panel, validation


def candidate_id(
    proposal_window: int,
    hidden_dim: int,
    actor_floor_path_weight: float,
    correction_abs_limit: float,
) -> str:
    return (
        f"state_mlp_w{int(proposal_window)}_h{int(hidden_dim)}_"
        f"floorw{float(actor_floor_path_weight):g}_"
        f"cap{float(correction_abs_limit):.3f}"
    )


def candidate_configs() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": candidate_id(window, hidden, weight, limit),
            "proposal_window": int(window),
            "hidden_dim": int(hidden),
            "hidden_layers": int(spec.HIDDEN_LAYERS),
            "actor_floor_path_weight": float(weight),
            "correction_abs_limit": float(limit),
        }
        for window in spec.PROPOSAL_WINDOWS
        for hidden in spec.HIDDEN_DIMS
        for weight in spec.ACTOR_FLOOR_PATH_WEIGHTS
        for limit in spec.CORRECTION_ABS_LIMITS
    ]


def grouped_fold_rows(
    environment_rows: list[dict[str, Any]], held_seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fit = [
        row for row in environment_rows
        if int(row["evaluation_seed"]) != int(held_seed)
    ]
    held = [
        row for row in environment_rows
        if int(row["evaluation_seed"]) == int(held_seed)
    ]
    if (
        len(fit)
        != (len(spec.REUSED_SELECTION_SEEDS) - 1)
        * len(spec.DISTURBANCE_MODES)
        or len(held) != len(spec.DISTURBANCE_MODES)
    ):
        raise RuntimeError("v18.2 grouped fold construction mismatch")
    return fit, held


def zero_state_actor_model(
    row: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    state_dim = int(np.asarray(row["lower_policy_state"]).shape[1])
    action_dim = int(np.asarray(row["total_action"]).shape[1])
    input_dim = state_dim + int(config["proposal_window"]) * 2 * action_dim
    hidden = int(config["hidden_dim"])
    dimensions = [input_dim]
    dimensions.extend([hidden] * int(config["hidden_layers"]))
    dimensions.append(action_dim)
    return {
        "model_type": "causal_state_residual_mlp_v1",
        "proposal_window": int(config["proposal_window"]),
        "state_dimension": state_dim,
        "action_dimension": action_dim,
        "input_dimension": input_dim,
        "hidden_dim": hidden,
        "hidden_layers": int(config["hidden_layers"]),
        "correction_abs_limit": float(config["correction_abs_limit"]),
        "feature_mean": np.zeros(input_dim, dtype=np.float64),
        "feature_scale": np.ones(input_dim, dtype=np.float64),
        "layers": [
            {
                "weight": np.zeros((target, source), dtype=np.float64),
                "bias": np.zeros(target, dtype=np.float64),
            }
            for source, target in zip(
                dimensions[:-1], dimensions[1:], strict=True
            )
        ],
        "fit_path_count": 0,
        "fit_total_path_weight": 0.0,
        "training_loss": 0.0,
        "training_epochs": 0,
        "training_seed": 0,
    }


def fit_environment_model(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    held_seed: int | str,
) -> dict[str, Any]:
    if not any(bool(row["actor_floor"]) for row in rows):
        return zero_state_actor_model(rows[0], config)
    training_seed = derive_seed(
        "mujoco_v18_2_state_actor_fit_v1",
        config["candidate_id"],
        rows[0]["environment"],
        held_seed,
    )
    return fit_state_conditioned_actor(
        [row["lower_policy_state"] for row in rows],
        [row["total_action"] for row in rows],
        [row["baseline_upper_action"] for row in rows],
        [row["target_total_correction"] for row in rows],
        [
            float(config["actor_floor_path_weight"])
            if row["actor_floor"] else 1.0
            for row in rows
        ],
        proposal_window=int(config["proposal_window"]),
        hidden_dim=int(config["hidden_dim"]),
        hidden_layers=int(config["hidden_layers"]),
        correction_abs_limit=float(config["correction_abs_limit"]),
        learning_rate=spec.LEARNING_RATE,
        weight_decay=spec.WEIGHT_DECAY,
        epochs=spec.TRAINING_EPOCHS,
        random_seed=training_seed,
        feature_scale_floor=spec.FEATURE_SCALE_FLOOR,
    )


def prediction_metrics(
    row: dict[str, Any],
    applied: dict[str, np.ndarray],
    *,
    correction_abs_limit: float,
) -> tuple[dict[str, Any], np.ndarray]:
    total = np.asarray(row["total_action"], dtype=np.float64)
    corrected = np.asarray(applied["corrected_total"], dtype=np.float64)
    correction = np.asarray(applied["correction"], dtype=np.float64)
    executed = np.asarray(applied["executed_correction"], dtype=np.float64)
    target = np.asarray(row["target_total_correction"], dtype=np.float64)
    raw = np.asarray(applied["raw_correction"], dtype=np.float64)
    limit = float(correction_abs_limit)
    valid = bool(
        np.all(np.isfinite(corrected))
        and np.all(np.isfinite(correction))
        and np.all(np.isfinite(raw))
        and np.max(np.abs(corrected))
        <= spec.COMPONENT_SUM_LIMIT + spec.CORRECTION_BOUND_TOLERANCE
        and np.max(np.abs(correction))
        <= limit + spec.CORRECTION_BOUND_TOLERANCE
        and np.max(np.abs(raw))
        <= limit + spec.CORRECTION_BOUND_TOLERANCE
    )
    return ({
        "valid": valid,
        "correction_rms": _rms(correction),
        "correction_abs_max": float(np.max(np.abs(correction))),
        "executed_correction_rms": _rms(executed),
        "target_squared_error": float(np.sum(np.square(correction - target))),
        "target_squared_energy": float(np.sum(np.square(target))),
    }, corrected)


def solve_corrected_oracle(corrected: np.ndarray) -> dict[str, Any]:
    return oracle.solve_full_horizon_responsibility_oracle(
        corrected,
        upper_rms_budget=spec.UPPER_RMS_BUDGET,
        lower_rms_budget=spec.LOWER_RMS_BUDGET,
        upper_action_limit=spec.UPPER_ACTION_LIMIT,
        lower_action_limit=spec.LOWER_ACTION_LIMIT,
        upper_window=spec.UPPER_WINDOW,
        lower_window=spec.LOWER_WINDOW,
        power_tolerance=spec.POWER_TOLERANCE,
    ).summary()


def evaluate_candidate(
    config: dict[str, Any], panel: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    torch.set_num_threads(1)
    rows: list[dict[str, Any]] = []
    for environment in spec.ENVIRONMENTS:
        environment_rows = [
            row for row in panel if row["environment"] == environment
        ]
        for held_seed in spec.REUSED_SELECTION_SEEDS:
            fit_rows, held_rows = grouped_fold_rows(
                environment_rows, int(held_seed)
            )
            model = fit_environment_model(
                fit_rows, config, held_seed=int(held_seed)
            )
            for row in held_rows:
                applied = apply_state_conditioned_actor(
                    row["lower_policy_state"],
                    row["total_action"],
                    row["baseline_upper_action"],
                    model,
                    component_sum_limit=spec.COMPONENT_SUM_LIMIT,
                    executed_action_limit=spec.EXECUTED_ACTION_LIMIT,
                )
                metrics, corrected = prediction_metrics(
                    row,
                    applied,
                    correction_abs_limit=float(
                        config["correction_abs_limit"]
                    ),
                )
                if metrics["correction_rms"] <= 1e-15:
                    oracle_summary = {
                        "joint_feasible": bool(row["oracle_joint_feasible"]),
                        "upper_power": float(row["oracle_upper_power"]),
                        "lower_power": float(row["oracle_lower_power"]),
                        "status": "unchanged_reference_oracle_reused",
                    }
                else:
                    oracle_summary = solve_corrected_oracle(corrected)
                rows.append({
                    "candidate_id": str(config["candidate_id"]),
                    "environment": str(environment),
                    "disturbance_mode": str(row["disturbance_mode"]),
                    "evaluation_seed": int(held_seed),
                    "actor_floor": bool(row["actor_floor"]),
                    "reference_joint_feasible": bool(
                        row["oracle_joint_feasible"]
                    ),
                    **metrics,
                    "corrected_joint_feasible": bool(
                        oracle_summary["joint_feasible"]
                    ),
                    "corrected_upper_power": float(
                        oracle_summary["upper_power"]
                    ),
                    "corrected_lower_power": float(
                        oracle_summary["lower_power"]
                    ),
                    "oracle_status": str(oracle_summary["status"]),
                })
    if len(rows) != spec.EXPECTED_PATH_COUNT:
        raise RuntimeError("v18.2 candidate OOF row count mismatch")
    return summarize_candidate(config, rows), rows


def summarize_candidate(
    config: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    floor = [row for row in rows if row["actor_floor"]]
    reference = [row for row in rows if row["reference_joint_feasible"]]
    target_energy = sum(float(row["target_squared_energy"]) for row in floor)
    target_error = sum(float(row["target_squared_error"]) for row in floor)
    by_environment = {}
    for environment in spec.ENVIRONMENTS:
        selected = [row for row in rows if row["environment"] == environment]
        selected_floor = [row for row in selected if row["actor_floor"]]
        selected_reference = [
            row for row in selected if row["reference_joint_feasible"]
        ]
        by_environment[environment] = {
            "path_count": len(selected),
            "corrected_joint_feasible_path_count": sum(
                bool(row["corrected_joint_feasible"]) for row in selected
            ),
            "actor_floor_path_count": len(selected_floor),
            "actor_floor_recovered_path_count": sum(
                bool(row["corrected_joint_feasible"])
                for row in selected_floor
            ),
            "reference_feasible_path_count": len(selected_reference),
            "reference_feasible_preserved_path_count": sum(
                bool(row["corrected_joint_feasible"])
                for row in selected_reference
            ),
        }
    floor_by_seed = {}
    for seed in sorted({int(row["evaluation_seed"]) for row in floor}):
        selected = [row for row in floor if row["evaluation_seed"] == seed]
        floor_by_seed[str(seed)] = {
            "path_count": len(selected),
            "recovered_path_count": sum(
                bool(row["corrected_joint_feasible"]) for row in selected
            ),
        }
    return {
        **config,
        "path_count": len(rows),
        "valid_path_count": sum(bool(row["valid"]) for row in rows),
        "corrected_joint_feasible_path_count": sum(
            bool(row["corrected_joint_feasible"]) for row in rows
        ),
        "reference_feasible_path_count": len(reference),
        "reference_feasible_preserved_path_count": sum(
            bool(row["corrected_joint_feasible"]) for row in reference
        ),
        "actor_floor_path_count": len(floor),
        "actor_floor_recovered_path_count": sum(
            bool(row["corrected_joint_feasible"]) for row in floor
        ),
        "actor_floor_executed_nonzero_path_count": sum(
            float(row["executed_correction_rms"])
            >= spec.EXECUTED_CORRECTION_RMS_MIN_GATE
            for row in floor
        ),
        "actor_floor_target_normalized_mse": float(
            target_error / target_energy
        ),
        "reference_feasible_correction_rms_mean": float(np.mean([
            row["correction_rms"] for row in reference
        ])),
        "reference_feasible_correction_rms_maximum": float(np.max([
            row["correction_rms"] for row in reference
        ])),
        "correction_abs_maximum": float(np.max([
            row["correction_abs_max"] for row in rows
        ])),
        "actor_floor_by_seed": floor_by_seed,
        "by_environment": by_environment,
    }


def advancement_gate(summary: dict[str, Any]) -> dict[str, bool]:
    return {
        "expected_path_count": bool(
            summary["path_count"] == spec.EXPECTED_PATH_COUNT
        ),
        "expected_reference_feasible_path_count": bool(
            summary["reference_feasible_path_count"]
            == spec.EXPECTED_REFERENCE_FEASIBLE_PATH_COUNT
        ),
        "expected_actor_floor_path_count": bool(
            summary["actor_floor_path_count"]
            == spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT
        ),
        "all_paths_valid": bool(
            summary["valid_path_count"] == spec.EXPECTED_PATH_COUNT
        ),
        "all_reference_feasible_paths_preserved": bool(
            summary["reference_feasible_preserved_path_count"]
            == spec.EXPECTED_REFERENCE_FEASIBLE_PATH_COUNT
        ),
        "all_actor_floor_paths_recovered": bool(
            summary["actor_floor_recovered_path_count"]
            == spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT
        ),
        "all_actor_floor_seed_groups_recovered": bool(all(
            row["recovered_path_count"] == row["path_count"]
            for row in summary["actor_floor_by_seed"].values()
        )),
        "actor_floor_target_fidelity_gate": bool(
            summary["actor_floor_target_normalized_mse"]
            <= spec.TARGET_NORMALIZED_MSE_GATE
        ),
        "reference_feasible_trust_region_gate": bool(
            summary["reference_feasible_correction_rms_maximum"]
            <= spec.REFERENCE_FEASIBLE_CORRECTION_RMS_MAX_GATE
        ),
        "all_actor_floor_paths_change_executed_action": bool(
            summary["actor_floor_executed_nonzero_path_count"]
            == spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT
        ),
    }


def selection_key(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(summary["corrected_joint_feasible_path_count"]),
        -int(summary["actor_floor_recovered_path_count"]),
        -int(summary["reference_feasible_preserved_path_count"]),
        -sum(
            row["recovered_path_count"] == row["path_count"]
            for row in summary["actor_floor_by_seed"].values()
        ),
        float(summary["actor_floor_target_normalized_mse"]),
        float(summary["reference_feasible_correction_rms_mean"]),
        float(summary["correction_abs_maximum"]),
        str(summary["candidate_id"]),
    )


def fit_final_models(
    panel: list[dict[str, Any]], selected: dict[str, Any]
) -> dict[str, Any]:
    models = {}
    for environment in spec.ENVIRONMENTS:
        rows = [row for row in panel if row["environment"] == environment]
        model = fit_environment_model(rows, selected, held_seed="all")
        models[environment] = model_to_json(model)
    return models


def model_to_json(model: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in model.items()
        if key not in {"feature_mean", "feature_scale", "layers"}
    }
    result["feature_mean"] = np.asarray(
        model["feature_mean"], dtype=np.float64
    ).tolist()
    result["feature_scale"] = np.asarray(
        model["feature_scale"], dtype=np.float64
    ).tolist()
    result["layers"] = [
        {
            "weight": np.asarray(
                layer["weight"], dtype=np.float64
            ).tolist(),
            "bias": np.asarray(layer["bias"], dtype=np.float64).tolist(),
        }
        for layer in model["layers"]
    ]
    return result


def run_selection(
    state_root: Path,
    reference_root: Path,
    target_root: Path,
    *,
    workers: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    torch.set_num_threads(1)
    source_identity = verify_current_freq_hrl_source_identity(
        code_revision=spec.FROZEN_CORE_REVISION,
        expected_source_manifest_sha256=spec.FROZEN_SOURCE_MANIFEST_SHA256,
        require_verified=True,
    )
    panel, dataset_validation = load_state_actor_panel(
        state_root, reference_root, target_root
    )
    configs = candidate_configs()
    if len(configs) != spec.EXPECTED_CANDIDATE_COUNT:
        raise RuntimeError("v18.2 candidate grid count drifted")
    worker_count = int(workers)
    if worker_count < 1:
        raise ValueError("v18.2 worker count must be positive")
    results: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    if worker_count == 1:
        for config in configs:
            result = evaluate_candidate(config, panel)
            results[str(config["candidate_id"])] = result
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(evaluate_candidate, config, panel): str(
                    config["candidate_id"]
                )
                for config in configs
            }
            for index, future in enumerate(as_completed(futures), start=1):
                candidate = futures[future]
                results[candidate] = future.result()
                print(
                    f"v18.2 exact candidate {index}/{len(configs)} "
                    f"complete={candidate}",
                    flush=True,
                )
    summaries = sorted(
        (value[0] for value in results.values()),
        key=lambda row: str(row["candidate_id"]),
    )
    selected = min(summaries, key=selection_key)
    selected_rows = results[str(selected["candidate_id"])][1]
    gate = advancement_gate(selected)
    floor = [row for row in panel if row["actor_floor"]]
    target_nonzero_count = sum(
        row["target_executed_correction_rms"]
        >= spec.EXECUTED_CORRECTION_RMS_MIN_GATE
        for row in floor
    )
    gate["all_actor_floor_targets_change_executed_action"] = bool(
        target_nonzero_count == spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT
    )
    advances = bool(all(gate.values()))
    status = (
        "state_conditioned_actor_authorizes_fresh_closed_loop_validation"
        if advances
        else "state_conditioned_actor_stops_before_fresh_path_access"
    )
    summary = {
        "status": status,
        "evidence_role": spec.EVIDENCE_ROLE,
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "frozen_core_revision": spec.FROZEN_CORE_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "source_identity": source_identity,
        "state_root": str(state_root),
        "reference_root": str(reference_root),
        "target_root": str(target_root),
        "selection_contract": spec.SELECTION_CONTRACT,
        "dataset_validation": dataset_validation,
        "candidate_count": len(configs),
        "full_oracle_candidate_count": len(summaries),
        "worker_count": worker_count,
        "target_audit": {
            "actor_floor_path_count": len(floor),
            "post_clipping_nonzero_target_path_count": target_nonzero_count,
            "target_correction_rms_mean": float(np.mean([
                row["target_correction_rms"] for row in floor
            ])),
            "target_executed_correction_rms_mean": float(np.mean([
                row["target_executed_correction_rms"] for row in floor
            ])),
        },
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate": selected,
        "advancement_gate": gate,
        "fresh_path_access_allowed": advances,
        "fresh_validation_paths_accessed": False,
        "support_gate": False,
        "selected_out_of_fold_rows": selected_rows,
        "full_oracle_candidate_summaries": summaries,
        "runtime_seconds": float(time.perf_counter() - started),
        "claim_boundary": spec.SELECTION_CONTRACT["claim_boundary"],
    }
    model = {
        "status": "v18_2_selected_state_conditioned_actor_fitted",
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "frozen_core_revision": spec.FROZEN_CORE_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "selected_candidate_id": selected["candidate_id"],
        "candidate": {
            key: selected[key]
            for key in (
                "candidate_id",
                "proposal_window",
                "hidden_dim",
                "hidden_layers",
                "actor_floor_path_weight",
                "correction_abs_limit",
            )
        },
        "component_sum_limit": spec.COMPONENT_SUM_LIMIT,
        "executed_action_limit": spec.EXECUTED_ACTION_LIMIT,
        "fresh_validation_eligible": advances,
        "environment_models": fit_final_models(panel, selected),
        "claim_boundary": spec.SELECTION_CONTRACT["claim_boundary"],
    }
    return summary, model


def _rms(value: Any) -> float:
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    summary, model = run_selection(
        args.state_root,
        args.reference_root,
        args.target_root,
        workers=args.workers,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "selected_model.json").write_text(
        json.dumps(model, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"DONE v18.2 state actor status={summary['status']} "
        f"selected={summary['selected_candidate_id']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
