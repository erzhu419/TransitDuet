#!/usr/bin/env python3
"""Analyze the frozen MuJoCo v14.11 iterative-projection screen."""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.reproducibility import derive_seed  # noqa: E402
from scripts import mujoco_v14_11_iterative_projection_screen_spec as spec  # noqa: E402


ANALYSIS_VERSION = "mujoco_v14_11_iterative_projection_screen_analysis_v1"
METRICS = (
    "episode_return",
    "LowerLFDriftAbs",
    "RawLowerLFDriftAbs",
    "LatentLowerLFDriftAbs",
    "RawLowerActionRMS",
    "LowerRouterClipRate",
    "LowerRouterUpperTransferRMS",
    "LowerRouterFunctionPreserving",
    "LowerRouterActionReconstructionRMS",
    "LowerActionRouterStrength",
    "UpperHFPowerAbs",
    "LatentUpperHFPowerAbs",
    "ResponsibilityReconstructionRMS",
)
TRACE_KEYS = (
    "RewardTraceSHA256",
    "ExecutedActionTraceSHA256",
    "LatentPolicyTraceSHA256",
)
MAX_METRICS = {
    "ResponsibilityReconstructionRMS",
    "LowerRouterActionReconstructionRMS",
}


def _cell_dir(
    run_dir: Path,
    *,
    environment: str,
    arm: str,
    seed: int,
) -> Path:
    return run_dir / "cells" / environment / arm / f"replicate_{int(seed)}"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = (
        len(spec.DEVELOPMENT_EVALUATION_SEEDS)
        * len(spec.EVALUATION_DISTURBANCE_MODES)
    )
    if len(rows) != expected:
        raise ValueError(
            f"v14.11 cell has {len(rows)} rows; expected {expected}: {path}"
        )
    keys = [(str(row["disturbance_mode"]), int(row["seed"])) for row in rows]
    expected_keys = {
        (mode, int(seed))
        for mode in spec.EVALUATION_DISTURBANCE_MODES
        for seed in spec.DEVELOPMENT_EVALUATION_SEEDS
    }
    if len(set(keys)) != len(keys) or set(keys) != expected_keys:
        raise ValueError(f"v14.11 evaluation path registry mismatch: {path}")
    return rows


def _finite_mean(rows: list[dict[str, str]], metric: str) -> float:
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite v14.11 metric: {metric}")
    return float(np.mean(values))


def _bounds(
    values: np.ndarray,
    *,
    confidence: float,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("v14.11 bootstrap requires finite optimizer replicates")
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, array.size, size=(int(draws), array.size))
    means = np.mean(array[indices], axis=1)
    alpha = 1.0 - float(confidence)
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


@functools.lru_cache(maxsize=None)
def _actor_vectors(path: str) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if not isinstance(model, dict):
        raise ValueError(f"invalid v14.11 checkpoint payload: {path}")
    vectors: list[np.ndarray] = []
    for actor_name in ("upper_actor", "lower_actor"):
        actor = model.get(actor_name)
        if not isinstance(actor, dict) or not actor:
            raise ValueError(f"missing v14.11 checkpoint actor {actor_name}: {path}")
        values = []
        for name in sorted(actor):
            tensor = actor[name]
            if not torch.is_tensor(tensor):
                raise ValueError(
                    f"non-tensor v14.11 actor parameter {actor_name}.{name}"
                )
            values.append(
                tensor.detach().cpu().to(torch.float64).reshape(-1).numpy()
            )
        vectors.append(np.concatenate(values))
    return vectors[0], vectors[1]


def _actor_rms_difference(candidate: Path, comparator: Path) -> dict[str, float]:
    candidate_upper, candidate_lower = _actor_vectors(str(candidate.resolve()))
    comparator_upper, comparator_lower = _actor_vectors(
        str(comparator.resolve())
    )
    if (
        candidate_upper.shape != comparator_upper.shape
        or candidate_lower.shape != comparator_lower.shape
    ):
        raise ValueError("v14.11 paired actor checkpoint shapes differ")
    upper_difference = candidate_upper - comparator_upper
    lower_difference = candidate_lower - comparator_lower
    combined_difference = np.concatenate((
        upper_difference, lower_difference,
    ))
    return {
        "upper": float(np.sqrt(np.mean(np.square(upper_difference)))),
        "lower": float(np.sqrt(np.mean(np.square(lower_difference)))),
        "combined": float(np.sqrt(np.mean(np.square(combined_difference)))),
    }


def _input_sha256(run_dir: Path) -> str:
    digest = hashlib.sha256()
    files = [
        run_dir / "preregistration.json",
        run_dir / "merged" / "run_scoped_result_sync.json",
        run_dir / "merged" / "cell_manifest.json",
    ]
    for environment in spec.ENVIRONMENTS:
        for seed in spec.OPTIMIZER_SEEDS:
            anchor = run_dir / "anchors" / environment / f"replicate_{seed}"
            files.extend((
                anchor / "cell_summary.json",
                anchor / "training_history.json",
                anchor / "evaluation_rows.csv",
                anchor / "checkpoint.pt",
            ))
        for arm in spec.ARMS:
            for seed in spec.OPTIMIZER_SEEDS:
                cell = _cell_dir(
                    run_dir, environment=environment, arm=arm, seed=seed
                )
                files.extend((
                    cell / "cell_summary.json",
                    cell / "training_history.json",
                    cell / "evaluation_rows.csv",
                    cell / "checkpoint.pt",
                ))
    for path in sorted(files):
        relative = str(path.relative_to(run_dir)).encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def load_replicates(
    run_dir: Path,
) -> dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]:
    preregistration = json.loads(
        (run_dir / "preregistration.json").read_text(encoding="utf-8")
    )
    if (
        preregistration.get("status")
        != "frozen_before_v14_11_iterative_projection_outcome_access"
        or preregistration.get("development_protocol_version")
        != spec.DEVELOPMENT_PROTOCOL_VERSION
        or preregistration.get("frozen_algorithm_revision")
        != spec.FROZEN_ALGORITHM_REVISION
        or preregistration.get("frozen_source_manifest_sha256")
        != spec.FROZEN_SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("v14.11 preregistration identity mismatch")
    manifest = json.loads(
        (run_dir / "merged" / "cell_manifest.json").read_text(encoding="utf-8")
    )
    expected_cells = (
        len(spec.ENVIRONMENTS)
        * (len(spec.ARMS) + 1)
        * len(spec.OPTIMIZER_SEEDS)
    )
    if (
        manifest.get("status") != "development_screen_complete_unanalyzed"
        or int(manifest.get("cell_count", -1)) != expected_cells
    ):
        raise ValueError("v14.11 analysis requires a complete merged manifest")

    output: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    for environment in spec.ENVIRONMENTS:
        output[environment] = {}
        for arm, arm_spec in spec.ARMS.items():
            by_mode = {
                mode: {metric: [] for metric in (*METRICS, *TRACE_KEYS)}
                for mode in spec.EVALUATION_DISTURBANCE_MODES
            }
            expected_strength = float(arm_spec["lower_action_router_strength"])
            for seed in spec.OPTIMIZER_SEEDS:
                rows = _read_rows(
                    _cell_dir(
                        run_dir,
                        environment=environment,
                        arm=arm,
                        seed=seed,
                    ) / "evaluation_rows.csv"
                )
                observed_strengths = np.asarray([
                    float(row["LowerActionRouterStrength"]) for row in rows
                ])
                if not np.allclose(
                    observed_strengths, expected_strength, rtol=0.0, atol=1e-12
                ):
                    raise ValueError(
                        f"v14.11 router strength mismatch: {environment}/{arm}/{seed}"
                    )
                for mode in spec.EVALUATION_DISTURBANCE_MODES:
                    mode_rows = [
                        row for row in rows if row["disturbance_mode"] == mode
                    ]
                    mode_rows.sort(key=lambda row: int(row["seed"]))
                    for metric in METRICS:
                        value = (
                            max(float(row[metric]) for row in mode_rows)
                            if metric in MAX_METRICS
                            else _finite_mean(mode_rows, metric)
                        )
                        by_mode[mode][metric].append(value)
                    for trace in TRACE_KEYS:
                        values = tuple(str(row[trace]) for row in mode_rows)
                        if any(len(value) != 64 for value in values):
                            raise ValueError(f"invalid v14.11 trace hash: {trace}")
                        by_mode[mode][trace].append(values)
            output[environment][arm] = {
                mode: {
                    metric: np.asarray(
                        values,
                        dtype=str if metric in TRACE_KEYS else np.float64,
                    )
                    for metric, values in collected.items()
                }
                for mode, collected in by_mode.items()
            }
    return output


def load_training_diagnostics(run_dir: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for arm in spec.ARMS:
        arm_spec = spec.ARMS[arm]
        by_environment: dict[str, dict[str, Any]] = {}
        all_trained: list[float] = []
        all_upper_lambda: list[float] = []
        all_lower_lambda: list[float] = []
        all_projection_pass: list[float] = []
        for environment in spec.ENVIRONMENTS:
            trained: list[float] = []
            upper_lambda: list[float] = []
            lower_lambda: list[float] = []
            projection_pass: list[float] = []
            projection_by_level = {
                level: {
                    "attempted_updates": 0,
                    "accepted_updates": 0,
                    "attempted_steps": 0,
                    "accepted_steps": 0,
                    "multistep_updates": 0,
                    "reward_budget_violations": 0,
                    "reductions": [],
                    "reduction_fractions": [],
                }
                for level in ("upper", "lower")
            }
            hashes: dict[str, str] = {}
            selected_iterations: dict[str, int] = {}
            for seed in spec.OPTIMIZER_SEEDS:
                cell = _cell_dir(
                    run_dir, environment=environment, arm=arm, seed=seed
                )
                summary = json.loads(
                    (cell / "cell_summary.json").read_text(encoding="utf-8")
                )
                history = json.loads(
                    (cell / "training_history.json").read_text(encoding="utf-8")
                )
                if not isinstance(history, list) or not history:
                    raise ValueError("v14.11 training history is empty")
                selected = int(summary.get("selected_checkpoint_iteration", -2))
                selected_iterations[str(seed)] = selected
                trained.append(float(
                    selected >= spec.ANALYSIS_TRAINED_CHECKPOINT_MINIMUM_ITERATION
                ))
                parameter_hash = str(summary.get("frozen_parameter_sha256", ""))
                if len(parameter_hash) != 64:
                    raise ValueError("v14.11 selected parameter hash is invalid")
                hashes[str(seed)] = parameter_hash
                upper_lambda.append(float(summary.get(
                    "upper_deployment_frequency_lambda_final", 0.0
                )))
                lower_lambda.append(float(summary.get(
                    "lower_deployment_frequency_lambda_final", 0.0
                )))
                seed_pass = True
                seed_accepted_steps = 0
                seed_multistep_updates = 0
                for level in ("upper", "lower"):
                    prefix = f"{level}_deployment_frequency"
                    active = bool(
                        float(arm_spec[f"{prefix}_dual_lr"]) > 0.0
                    )
                    diagnostic_rows = [
                        row for row in history
                        if float(row.get(f"{prefix}_enabled", 0.0)) > 0.5
                    ]
                    violating_rows = [
                        row for row in diagnostic_rows
                        if float(row.get(
                            f"{prefix}_projection_target_reached_before", 0.0
                        )) < 0.5
                    ]
                    attempted_rows = [
                        row for row in history
                        if float(row.get(
                            f"{prefix}_projection_steps_attempted",
                            row.get(f"{prefix}_guard_attempted", 0.0),
                        )) > 0.5
                    ]
                    accepted_rows = [
                        row for row in attempted_rows
                        if float(row.get(
                            f"{prefix}_projection_steps_accepted",
                            row.get(f"{prefix}_guard_accepted", 0.0),
                        )) > 0.5
                    ]
                    attempted_steps = int(round(sum(float(row.get(
                        f"{prefix}_projection_steps_attempted",
                        row.get(f"{prefix}_guard_attempted", 0.0),
                    )) for row in attempted_rows)))
                    accepted_steps = int(round(sum(float(row.get(
                        f"{prefix}_projection_steps_accepted",
                        row.get(f"{prefix}_guard_accepted", 0.0),
                    )) for row in accepted_rows)))
                    multistep_updates = sum(
                        float(row.get(
                            f"{prefix}_projection_steps_accepted",
                            row.get(f"{prefix}_guard_accepted", 0.0),
                        )) >= spec.MINIMUM_ITERATIVE_ACCEPTED_STEPS
                        for row in accepted_rows
                    )
                    reductions = [
                        float(row[f"{prefix}_power_before"])
                        - float(row[f"{prefix}_power_after"])
                        for row in accepted_rows
                    ]
                    reduction_fractions = [
                        reduction / max(
                            float(row[f"{prefix}_power_before"]), 1e-12
                        )
                        for row, reduction in zip(
                            accepted_rows, reductions, strict=True
                        )
                    ]
                    reward_tolerance = float(
                        arm_spec[f"{prefix}_reward_tolerance"]
                    )
                    reward_budget_violations = sum(
                        float(row.get(
                            f"{prefix}_guard_reward_loss_delta", 0.0
                        )) > reward_tolerance + 1e-8
                        for row in diagnostic_rows
                    )
                    config_match = bool(diagnostic_rows) and all(
                        int(round(float(row.get(
                            f"{prefix}_projection_steps_requested", -1.0
                        )))) == int(arm_spec[
                            f"{prefix}_max_projection_steps"
                        ])
                        and abs(float(row.get(
                            f"{prefix}_projection_reward_tolerance",
                            float("nan"),
                        )) - reward_tolerance) <= 1e-15
                        for row in diagnostic_rows
                    )
                    feasibility_or_correction = bool(
                        diagnostic_rows
                        and (
                            not violating_rows
                            or (
                                attempted_rows
                                and accepted_rows
                                and float(np.mean(reductions)) > 0.0
                            )
                        )
                    )
                    values = projection_by_level[level]
                    values["attempted_updates"] += len(attempted_rows)
                    values["accepted_updates"] += len(accepted_rows)
                    values["attempted_steps"] += attempted_steps
                    values["accepted_steps"] += accepted_steps
                    values["multistep_updates"] += multistep_updates
                    values["reward_budget_violations"] += (
                        reward_budget_violations
                    )
                    projection_by_level[level]["reductions"].extend(reductions)
                    projection_by_level[level]["reduction_fractions"].extend(
                        reduction_fractions
                    )
                    if active:
                        seed_pass = bool(
                            seed_pass
                            and feasibility_or_correction
                            and config_match
                            and reward_budget_violations
                            <= spec.MAXIMUM_PROJECTION_REWARD_BUDGET_VIOLATIONS
                        )
                        seed_accepted_steps += accepted_steps
                        seed_multistep_updates += multistep_updates
                if arm in spec.ITERATIVE_ARMS:
                    seed_pass = bool(
                        seed_pass
                        and seed_accepted_steps
                        >= spec.MINIMUM_ITERATIVE_ACCEPTED_STEPS
                        and seed_multistep_updates
                        >= spec.MINIMUM_ITERATIVE_MULTISTEP_UPDATES
                    )
                projection_pass.append(float(seed_pass))
            by_environment[environment] = {
                "trained_checkpoint_fraction": float(np.mean(trained)),
                "initial_checkpoint_fallback_count": int(sum(
                    selected < 0 for selected in selected_iterations.values()
                )),
                "upper_deployment_frequency_lambda_final_mean": float(
                    np.mean(upper_lambda)
                ),
                "lower_deployment_frequency_lambda_final_mean": float(
                    np.mean(lower_lambda)
                ),
                "deployment_projection_mechanism_fraction": float(
                    np.mean(projection_pass)
                ),
                "deployment_projection_by_level": {
                    level: {
                        "active": bool(float(arm_spec[
                            f"{level}_deployment_frequency_dual_lr"
                        ]) > 0.0),
                        "guard_attempted_update_count": int(
                            values["attempted_updates"]
                        ),
                        "guard_accepted_update_count": int(
                            values["accepted_updates"]
                        ),
                        "projection_steps_attempted": int(
                            values["attempted_steps"]
                        ),
                        "projection_steps_accepted": int(
                            values["accepted_steps"]
                        ),
                        "multistep_update_count": int(
                            values["multistep_updates"]
                        ),
                        "reward_budget_violation_count": int(
                            values["reward_budget_violations"]
                        ),
                        "accepted_power_reduction_mean": float(
                            np.mean(values["reductions"])
                            if values["reductions"] else 0.0
                        ),
                        "accepted_power_reduction_fraction_mean": float(
                            np.mean(values["reduction_fractions"])
                            if values["reduction_fractions"] else 0.0
                        ),
                    }
                    for level, values in projection_by_level.items()
                },
                "selected_checkpoint_iteration": selected_iterations,
                "selected_parameter_sha256": hashes,
            }
            all_trained.extend(trained)
            all_upper_lambda.extend(upper_lambda)
            all_lower_lambda.extend(lower_lambda)
            all_projection_pass.extend(projection_pass)
        output[arm] = {
            "by_environment": by_environment,
            "minimum_environment_trained_checkpoint_fraction": float(min(
                row["trained_checkpoint_fraction"]
                for row in by_environment.values()
            )),
            "minimum_environment_deployment_projection_mechanism_fraction": (
                float(min(
                    row["deployment_projection_mechanism_fraction"]
                    for row in by_environment.values()
                ))
            ),
            "upper_deployment_frequency_lambda_final_mean": float(
                np.mean(all_upper_lambda)
            ),
            "lower_deployment_frequency_lambda_final_mean": float(
                np.mean(all_lower_lambda)
            ),
            "deployment_projection_mechanism_fraction": float(
                np.mean(all_projection_pass)
            ),
            "accepted_power_reduction_fraction_mean": float(np.mean([
                level_status["accepted_power_reduction_fraction_mean"]
                for environment_status in by_environment.values()
                for level_status in environment_status[
                    "deployment_projection_by_level"
                ].values()
                if level_status["active"]
            ]) if any(
                level_status["active"]
                for environment_status in by_environment.values()
                for level_status in environment_status[
                    "deployment_projection_by_level"
                ].values()
            ) else 0.0),
        }
    return output


def _paired_actor_statistics(
    run_dir: Path,
    *,
    arm: str,
    comparator: str,
) -> dict[str, Any]:
    combined_all: list[float] = []
    upper_all: list[float] = []
    lower_all: list[float] = []
    by_environment: dict[str, dict[str, float]] = {}
    for environment in spec.ENVIRONMENTS:
        combined: list[float] = []
        upper: list[float] = []
        lower: list[float] = []
        for seed in spec.OPTIMIZER_SEEDS:
            difference = _actor_rms_difference(
                _cell_dir(
                    run_dir, environment=environment, arm=arm, seed=seed
                ) / "checkpoint.pt",
                _cell_dir(
                    run_dir,
                    environment=environment,
                    arm=comparator,
                    seed=seed,
                ) / "checkpoint.pt",
            )
            combined.append(difference["combined"])
            upper.append(difference["upper"])
            lower.append(difference["lower"])
        by_environment[environment] = {
            "combined_mean": float(np.mean(combined)),
            "combined_minimum": float(min(combined)),
            "combined_maximum": float(max(combined)),
            "upper_mean": float(np.mean(upper)),
            "lower_mean": float(np.mean(lower)),
        }
        combined_all.extend(combined)
        upper_all.extend(upper)
        lower_all.extend(lower)
    return {
        "comparator": comparator,
        "by_environment": by_environment,
        "combined_mean": float(np.mean(combined_all)),
        "combined_minimum": float(min(combined_all)),
        "combined_maximum": float(max(combined_all)),
        "upper_mean": float(np.mean(upper_all)),
        "lower_mean": float(np.mean(lower_all)),
    }


def _reduction_gate(
    *,
    baseline: np.ndarray,
    candidate: np.ndarray,
    reduction_fraction: float,
    materiality_floor: float,
    seed_parts: tuple[object, ...],
) -> dict[str, Any]:
    baseline_values = np.asarray(baseline, dtype=np.float64)
    candidate_values = np.asarray(candidate, dtype=np.float64)
    baseline_mean = float(np.mean(baseline_values))
    difference = candidate_values - baseline_values
    _, difference_upper = _bounds(
        difference,
        confidence=spec.SELECTION_CONFIDENCE,
        draws=spec.BOOTSTRAP_DRAWS,
        seed=derive_seed(ANALYSIS_VERSION, *seed_parts, "difference"),
    )
    material = baseline_mean > float(materiality_floor)
    if material:
        required = float(reduction_fraction) * baseline_mean
        passed = difference_upper <= -required
        candidate_upper = float("nan")
        slack = (-required - difference_upper) / max(
            required, float(materiality_floor)
        )
        gate_type = "strict_relative_improvement"
    else:
        _, candidate_upper = _bounds(
            candidate_values,
            confidence=spec.SELECTION_CONFIDENCE,
            draws=spec.BOOTSTRAP_DRAWS,
            seed=derive_seed(
                ANALYSIS_VERSION, *seed_parts, "absolute_floor"
            ),
        )
        required = 0.0
        passed = candidate_upper <= float(materiality_floor)
        slack = (
            float(materiality_floor) - candidate_upper
        ) / float(materiality_floor)
        gate_type = "absolute_floor_noninferiority"
    return {
        "baseline_mean": baseline_mean,
        "candidate_mean": float(np.mean(candidate_values)),
        "difference_mean": float(np.mean(difference)),
        "difference_one_sided_upper": float(difference_upper),
        "candidate_one_sided_upper": float(candidate_upper),
        "required_reduction": float(required),
        "materiality_floor": float(materiality_floor),
        "baseline_material": bool(material),
        "gate_type": gate_type,
        "pass": bool(passed),
        "normalized_slack": float(slack),
    }


def _flatten_gate(prefix: str, gate: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in gate.items()}


def analyze(run_dir: Path) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    replicates = load_replicates(run)
    training = load_training_diagnostics(run)
    legacy_projection_reduction = float(
        training[spec.LEGACY_ONE_STEP_ARM][
            "accepted_power_reduction_fraction_mean"
        ]
    )
    rows: list[dict[str, Any]] = []
    arm_status: dict[str, dict[str, Any]] = {}

    for arm in spec.EVALUATED_ARMS:
        is_calibration = arm == spec.CALIBRATION_ARM
        comparator = (
            spec.BASE_CONTROL_ARM
            if is_calibration else spec.MATCHED_COMPARATOR_ARM
        )
        role = "calibration" if is_calibration else "learned"
        condition_rows: list[dict[str, Any]] = []
        for environment in spec.ENVIRONMENTS:
            for mode in spec.EVALUATION_DISTURBANCE_MODES:
                baseline = replicates[environment][comparator][mode]
                candidate = replicates[environment][arm][mode]
                return_difference = (
                    candidate["episode_return"] - baseline["episode_return"]
                )
                baseline_return = float(np.mean(baseline["episode_return"]))
                reward_margin = (
                    spec.RETURN_NONINFERIORITY_MARGIN_FRACTION
                    * max(abs(baseline_return), 1.0)
                )
                reward_lower, reward_upper = _bounds(
                    return_difference,
                    confidence=spec.SELECTION_CONFIDENCE,
                    draws=spec.BOOTSTRAP_DRAWS,
                    seed=derive_seed(
                        ANALYSIS_VERSION, arm, environment, mode, "return"
                    ),
                )
                absolute_return_difference_max = float(np.max(np.abs(
                    return_difference
                )))
                exact_return_pass = bool(
                    absolute_return_difference_max
                    <= spec.MAXIMUM_ABSOLUTE_RETURN_DIFFERENCE
                )
                trace_matches = {
                    trace: bool(np.array_equal(candidate[trace], baseline[trace]))
                    for trace in TRACE_KEYS
                }
                exact_trace_pass = bool(all(trace_matches.values()))
                reward_noninferiority_pass = bool(
                    reward_lower >= -reward_margin
                )
                calibration_pathwise_pass = bool(
                    exact_return_pass and exact_trace_pass
                )
                strict_reward_improvement_pass = bool(reward_lower > 0.0)

                responsibility = _reduction_gate(
                    baseline=baseline["LowerLFDriftAbs"],
                    candidate=candidate["LowerLFDriftAbs"],
                    reduction_fraction=(
                        spec.MINIMUM_PROJECTION_LOWER_REDUCTION_FRACTION
                        if is_calibration
                        else spec.MINIMUM_RESPONSIBILITY_DRIFT_REDUCTION_FRACTION
                    ),
                    materiality_floor=spec.LOWER_DRIFT_MATERIALITY_FLOOR,
                    seed_parts=(arm, environment, mode, "responsibility"),
                )
                raw_lower = _reduction_gate(
                    baseline=baseline["RawLowerLFDriftAbs"],
                    candidate=candidate["RawLowerLFDriftAbs"],
                    reduction_fraction=(
                        spec.MINIMUM_PROJECTION_LOWER_REDUCTION_FRACTION
                        if is_calibration
                        else spec.MINIMUM_RAW_LOWER_DRIFT_REDUCTION_FRACTION
                    ),
                    materiality_floor=spec.LOWER_DRIFT_MATERIALITY_FLOOR,
                    seed_parts=(arm, environment, mode, "raw_lower"),
                )
                latent_lower = _reduction_gate(
                    baseline=baseline["LatentLowerLFDriftAbs"],
                    candidate=candidate["LatentLowerLFDriftAbs"],
                    reduction_fraction=(
                        0.0 if is_calibration
                        else spec.MINIMUM_LATENT_LOWER_DRIFT_REDUCTION_FRACTION
                    ),
                    materiality_floor=spec.LOWER_DRIFT_MATERIALITY_FLOOR,
                    seed_parts=(arm, environment, mode, "latent_lower"),
                )
                effective_upper = _reduction_gate(
                    baseline=baseline["UpperHFPowerAbs"],
                    candidate=candidate["UpperHFPowerAbs"],
                    reduction_fraction=(
                        spec.MINIMUM_PROJECTION_UPPER_REDUCTION_FRACTION
                        if is_calibration else 0.0
                    ),
                    materiality_floor=spec.UPPER_HF_MATERIALITY_FLOOR,
                    seed_parts=(arm, environment, mode, "effective_upper"),
                )
                latent_upper = _reduction_gate(
                    baseline=baseline["LatentUpperHFPowerAbs"],
                    candidate=candidate["LatentUpperHFPowerAbs"],
                    reduction_fraction=(
                        0.0 if is_calibration
                        else spec.MINIMUM_LATENT_UPPER_HF_REDUCTION_FRACTION
                    ),
                    materiality_floor=spec.UPPER_HF_MATERIALITY_FLOOR,
                    seed_parts=(arm, environment, mode, "latent_upper"),
                )

                _, effective_upper_power_bound = _bounds(
                    candidate["UpperHFPowerAbs"],
                    confidence=spec.SELECTION_CONFIDENCE,
                    draws=spec.BOOTSTRAP_DRAWS,
                    seed=derive_seed(
                        ANALYSIS_VERSION, arm, environment, mode,
                        "effective_upper_budget",
                    ),
                )
                _, latent_upper_power_bound = _bounds(
                    candidate["LatentUpperHFPowerAbs"],
                    confidence=spec.SELECTION_CONFIDENCE,
                    draws=spec.BOOTSTRAP_DRAWS,
                    seed=derive_seed(
                        ANALYSIS_VERSION, arm, environment, mode,
                        "latent_upper_budget",
                    ),
                )
                effective_upper_rms_bound = float(np.sqrt(max(
                    effective_upper_power_bound, 0.0
                )))
                latent_upper_rms_bound = float(np.sqrt(max(
                    latent_upper_power_bound, 0.0
                )))
                effective_upper_budget_pass = bool(
                    effective_upper_rms_bound <= spec.UPPER_HF_REPORTING_GATE
                )
                latent_upper_budget_pass = bool(
                    latent_upper_rms_bound
                    <= spec.LATENT_UPPER_HF_REPORTING_GATE
                )

                activity_margin = (
                    candidate["RawLowerActionRMS"]
                    - spec.MINIMUM_EFFECTIVE_LOWER_ACTION_RMS_FRACTION
                    * baseline["RawLowerActionRMS"]
                )
                activity_lower, _ = _bounds(
                    activity_margin,
                    confidence=spec.SELECTION_CONFIDENCE,
                    draws=spec.BOOTSTRAP_DRAWS,
                    seed=derive_seed(
                        ANALYSIS_VERSION, arm, environment, mode, "activity"
                    ),
                )
                activity_pass = bool(activity_lower >= 0.0)
                _, router_clip_upper = _bounds(
                    candidate["LowerRouterClipRate"],
                    confidence=spec.SELECTION_CONFIDENCE,
                    draws=spec.BOOTSTRAP_DRAWS,
                    seed=derive_seed(
                        ANALYSIS_VERSION, arm, environment, mode, "clip"
                    ),
                )
                router_clip_pass = bool(
                    router_clip_upper <= spec.MAXIMUM_ROUTER_CLIP_RATE
                )
                reconstruction_max = float(np.max(
                    candidate["ResponsibilityReconstructionRMS"]
                ))
                router_reconstruction_max = float(np.max(
                    candidate["LowerRouterActionReconstructionRMS"]
                ))
                reconstruction_pass = bool(
                    reconstruction_max <= spec.MAXIMUM_RECONSTRUCTION_RMS
                    and router_reconstruction_max
                    <= spec.MAXIMUM_RECONSTRUCTION_RMS
                )
                function_preserving_pass = bool(np.allclose(
                    candidate["LowerRouterFunctionPreserving"],
                    1.0,
                    rtol=0.0,
                    atol=0.0,
                ))
                if is_calibration:
                    frequency_pass = bool(
                        responsibility["pass"]
                        and raw_lower["pass"]
                        and effective_upper["pass"]
                    )
                    reward_pass = calibration_pathwise_pass
                else:
                    frequency_pass = bool(
                        responsibility["pass"]
                        and raw_lower["pass"]
                        and latent_lower["pass"]
                        and latent_upper["pass"]
                    )
                    reward_pass = reward_noninferiority_pass
                condition_gate_pass = bool(
                    reward_pass
                    and frequency_pass
                    and effective_upper_budget_pass
                    and latent_upper_budget_pass
                    and activity_pass
                    and router_clip_pass
                    and reconstruction_pass
                    and function_preserving_pass
                )
                frequency_slacks = (
                    responsibility["normalized_slack"],
                    raw_lower["normalized_slack"],
                    effective_upper["normalized_slack"],
                ) if is_calibration else (
                    responsibility["normalized_slack"],
                    raw_lower["normalized_slack"],
                    latent_lower["normalized_slack"],
                    latent_upper["normalized_slack"],
                )
                slacks = (
                    (reward_lower + reward_margin) / max(reward_margin, 1e-12),
                    *frequency_slacks,
                    (
                        spec.UPPER_HF_REPORTING_GATE
                        - effective_upper_rms_bound
                    ) / spec.UPPER_HF_REPORTING_GATE,
                    (
                        spec.LATENT_UPPER_HF_REPORTING_GATE
                        - latent_upper_rms_bound
                    ) / spec.LATENT_UPPER_HF_REPORTING_GATE,
                    activity_lower / max(
                        float(np.mean(baseline["RawLowerActionRMS"])), 1e-12
                    ),
                    (
                        spec.MAXIMUM_ROUTER_CLIP_RATE - router_clip_upper
                    ) / spec.MAXIMUM_ROUTER_CLIP_RATE,
                )
                row = {
                    "environment": environment,
                    "disturbance_mode": mode,
                    "arm": arm,
                    "arm_role": role,
                    "comparator_arm": comparator,
                    "upper_dual_lr": float(spec.ARMS[arm]["upper_dual_lr"]),
                    "lower_dual_lr": float(spec.ARMS[arm]["lower_dual_lr"]),
                    "upper_deployment_frequency_dual_lr": float(
                        spec.ARMS[arm]["upper_deployment_frequency_dual_lr"]
                    ),
                    "lower_deployment_frequency_dual_lr": float(
                        spec.ARMS[arm]["lower_deployment_frequency_dual_lr"]
                    ),
                    "checkpoint_score_mode": spec.ARMS[arm][
                        "checkpoint_score_mode"
                    ],
                    "replicate_count": len(spec.OPTIMIZER_SEEDS),
                    "baseline_return_mean": baseline_return,
                    "candidate_return_mean": float(np.mean(
                        candidate["episode_return"]
                    )),
                    "return_difference_mean": float(np.mean(return_difference)),
                    "return_difference_one_sided_lower": reward_lower,
                    "return_difference_one_sided_upper": reward_upper,
                    "absolute_return_difference_max": (
                        absolute_return_difference_max
                    ),
                    "exact_return_pass": exact_return_pass,
                    **{
                        f"{trace}_match": matched
                        for trace, matched in trace_matches.items()
                    },
                    "exact_trace_pass": exact_trace_pass,
                    "calibration_pathwise_identity_pass": (
                        calibration_pathwise_pass
                    ),
                    "executed_action_trace_changed": bool(
                        not trace_matches["ExecutedActionTraceSHA256"]
                    ),
                    "latent_policy_trace_changed": bool(
                        not trace_matches["LatentPolicyTraceSHA256"]
                    ),
                    "reward_noninferiority_margin": reward_margin,
                    "reward_noninferiority_pass": reward_noninferiority_pass,
                    "strict_reward_improvement_pass": (
                        strict_reward_improvement_pass
                    ),
                    **_flatten_gate("responsibility_drift", responsibility),
                    **_flatten_gate("raw_lower_drift", raw_lower),
                    **_flatten_gate("latent_lower_drift", latent_lower),
                    **_flatten_gate("effective_upper_hf", effective_upper),
                    **_flatten_gate("latent_upper_hf", latent_upper),
                    "effective_upper_hf_rms": float(np.sqrt(np.mean(
                        candidate["UpperHFPowerAbs"]
                    ))),
                    "effective_upper_hf_rms_one_sided_upper": (
                        effective_upper_rms_bound
                    ),
                    "effective_upper_hf_budget_pass": (
                        effective_upper_budget_pass
                    ),
                    "latent_upper_hf_rms": float(np.sqrt(np.mean(
                        candidate["LatentUpperHFPowerAbs"]
                    ))),
                    "latent_upper_hf_rms_one_sided_upper": (
                        latent_upper_rms_bound
                    ),
                    "latent_upper_hf_budget_pass": latent_upper_budget_pass,
                    "effective_lower_action_rms_mean": float(np.mean(
                        candidate["RawLowerActionRMS"]
                    )),
                    "effective_lower_activity_margin_one_sided_lower": (
                        activity_lower
                    ),
                    "effective_lower_activity_pass": activity_pass,
                    "router_clip_rate_mean": float(np.mean(
                        candidate["LowerRouterClipRate"]
                    )),
                    "router_clip_rate_one_sided_upper": router_clip_upper,
                    "router_clip_pass": router_clip_pass,
                    "reconstruction_rms_max": reconstruction_max,
                    "router_reconstruction_rms_max": router_reconstruction_max,
                    "reconstruction_integrity_pass": reconstruction_pass,
                    "function_preserving_pass": function_preserving_pass,
                    "upper_transfer_rms_mean": float(np.mean(
                        candidate["LowerRouterUpperTransferRMS"]
                    )),
                    "frequency_gate_pass": frequency_pass,
                    "minimum_normalized_safety_slack": float(min(slacks)),
                    "condition_gate_pass": condition_gate_pass,
                }
                rows.append(row)
                condition_rows.append(row)

        actor_statistics = _paired_actor_statistics(
            run, arm=arm, comparator=comparator
        )
        training_status = training[arm]
        minimum_trained_fraction = float(
            training_status["minimum_environment_trained_checkpoint_fraction"]
        )
        trained_pass = bool(
            minimum_trained_fraction
            >= spec.MINIMUM_TRAINED_CHECKPOINT_FRACTION
        )
        deployment_projection_pass = bool(
            is_calibration
            or training_status[
                "minimum_environment_deployment_projection_mechanism_fraction"
            ] >= 1.0
        )
        iterative_candidate = bool(arm in spec.ITERATIVE_ARMS)
        iterative_reduction_gain = float(
            training_status["accepted_power_reduction_fraction_mean"]
            - legacy_projection_reduction
        )
        iterative_reduction_gain_pass = bool(
            is_calibration
            or (
                iterative_candidate
                and iterative_reduction_gain + 1e-12
                >= spec.MINIMUM_ITERATIVE_REDUCTION_GAIN_OVER_ONE_STEP
            )
        )
        all_conditions = bool(all(
            row["condition_gate_pass"] for row in condition_rows
        ))
        calibration_pathwise_pass = bool(all(
            row["calibration_pathwise_identity_pass"]
            for row in condition_rows
        ))
        calibration_actor_pass = bool(
            actor_statistics["combined_maximum"]
            <= spec.MAXIMUM_CALIBRATION_ACTOR_RMS
        )
        learned_actor_pass = bool(
            actor_statistics["combined_minimum"]
            >= spec.MINIMUM_LEARNED_PARAMETER_RMS
        )
        changed_action_conditions = int(sum(
            bool(row["executed_action_trace_changed"])
            for row in condition_rows
        ))
        changed_action_environments = len({
            str(row["environment"])
            for row in condition_rows
            if bool(row["executed_action_trace_changed"])
        })
        changed_latent_conditions = int(sum(
            bool(row["latent_policy_trace_changed"])
            for row in condition_rows
        ))
        learned_behavior_pass = bool(
            changed_action_conditions
            >= spec.MINIMUM_CHANGED_ACTION_TRACE_CONDITIONS
            and changed_latent_conditions
            >= spec.MINIMUM_CHANGED_ACTION_TRACE_CONDITIONS
            and changed_action_environments
            >= spec.MINIMUM_CHANGED_ACTION_TRACE_ENVIRONMENTS
        )
        strict_reward_improvements = int(sum(
            bool(row["strict_reward_improvement_pass"])
            for row in condition_rows
        ))
        strict_reward_pass = bool(
            strict_reward_improvements
            >= spec.MINIMUM_STRICT_REWARD_IMPROVEMENT_CONDITIONS
        )
        calibration_validation_pass = bool(
            is_calibration
            and all_conditions
            and trained_pass
            and calibration_pathwise_pass
            and calibration_actor_pass
        )
        development_selection_pass = bool(
            not is_calibration
            and all_conditions
            and trained_pass
            and learned_actor_pass
            and learned_behavior_pass
            and deployment_projection_pass
            and iterative_reduction_gain_pass
            and strict_reward_pass
        )
        global_slacks = [
            (
                minimum_trained_fraction
                - spec.MINIMUM_TRAINED_CHECKPOINT_FRACTION
            ) / spec.MINIMUM_TRAINED_CHECKPOINT_FRACTION,
        ]
        if is_calibration:
            global_slacks.append((
                spec.MAXIMUM_CALIBRATION_ACTOR_RMS
                - actor_statistics["combined_maximum"]
            ) / max(spec.MAXIMUM_CALIBRATION_ACTOR_RMS, 1e-12))
        else:
            global_slacks.extend((
                (
                    actor_statistics["combined_minimum"]
                    - spec.MINIMUM_LEARNED_PARAMETER_RMS
                ) / spec.MINIMUM_LEARNED_PARAMETER_RMS,
                (
                    changed_action_conditions
                    - spec.MINIMUM_CHANGED_ACTION_TRACE_CONDITIONS
                ) / max(spec.MINIMUM_CHANGED_ACTION_TRACE_CONDITIONS, 1),
                (
                    changed_action_environments
                    - spec.MINIMUM_CHANGED_ACTION_TRACE_ENVIRONMENTS
                ) / max(spec.MINIMUM_CHANGED_ACTION_TRACE_ENVIRONMENTS, 1),
                (
                    strict_reward_improvements
                    - spec.MINIMUM_STRICT_REWARD_IMPROVEMENT_CONDITIONS
                ) / max(
                    spec.MINIMUM_STRICT_REWARD_IMPROVEMENT_CONDITIONS, 1
                ),
                (
                    training_status[
                        "minimum_environment_deployment_projection_mechanism_fraction"
                    ] - 1.0
                ),
                (
                    iterative_reduction_gain
                    - spec.MINIMUM_ITERATIVE_REDUCTION_GAIN_OVER_ONE_STEP
                ) / spec.MINIMUM_ITERATIVE_REDUCTION_GAIN_OVER_ONE_STEP,
            ))
        arm_status[arm] = {
            "arm_role": role,
            "comparator_arm": comparator,
            "all_environment_condition_gates_pass": all_conditions,
            "passed_gate_count": int(sum(
                bool(row["condition_gate_pass"]) for row in condition_rows
            )),
            "total_gate_count": len(condition_rows),
            "trained_checkpoint_gate_pass": trained_pass,
            "minimum_environment_trained_checkpoint_fraction": (
                minimum_trained_fraction
            ),
            "calibration_pathwise_identity_gate_pass": (
                calibration_pathwise_pass
            ),
            "calibration_actor_identity_gate_pass": calibration_actor_pass,
            "calibration_validation_pass": calibration_validation_pass,
            "learned_actor_difference_gate_pass": learned_actor_pass,
            "paired_actor_statistics": actor_statistics,
            "changed_action_trace_condition_count": changed_action_conditions,
            "changed_action_trace_environment_count": (
                changed_action_environments
            ),
            "changed_latent_trace_condition_count": changed_latent_conditions,
            "learned_behavior_gate_pass": learned_behavior_pass,
            "deployment_projection_mechanism_gate_pass": (
                deployment_projection_pass
            ),
            "iterative_candidate": iterative_candidate,
            "one_step_projection_reduction_fraction": (
                legacy_projection_reduction
            ),
            "iterative_projection_reduction_gain": iterative_reduction_gain,
            "iterative_projection_reduction_gain_gate_pass": (
                iterative_reduction_gain_pass
            ),
            "strict_reward_improvement_condition_count": (
                strict_reward_improvements
            ),
            "strict_reward_improvement_gate_pass": strict_reward_pass,
            "development_selection_pass": development_selection_pass,
            "minimum_normalized_safety_slack": float(min([
                row["minimum_normalized_safety_slack"]
                for row in condition_rows
            ] + global_slacks)),
            "mean_return_lower_bound": float(np.mean([
                row["return_difference_one_sided_lower"]
                for row in condition_rows
            ])),
            "training_diagnostics": training_status,
        }

    eligible = [
        arm for arm in spec.LEARNED_ARMS
        if arm_status[arm]["development_selection_pass"]
    ]
    selected = (
        max(
            eligible,
            key=lambda arm: (
                arm_status[arm]["minimum_normalized_safety_slack"],
                arm_status[arm]["mean_return_lower_bound"],
                arm,
            ),
        )
        if eligible else None
    )
    return {
        "analysis_version": ANALYSIS_VERSION,
        "status": (
            "learned_candidate_selected"
            if selected is not None else "no_latent_behavior_safe_candidate"
        ),
        "evidence_role": "development_screen_not_confirmatory",
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "selected_arm": selected,
        "selected_arm_spec": None if selected is None else spec.ARMS[selected],
        "eligible_arms": eligible,
        "base_control_arm": spec.BASE_CONTROL_ARM,
        "calibration_arm": spec.CALIBRATION_ARM,
        "matched_comparator_arm": spec.MATCHED_COMPARATOR_ARM,
        "calibration_validation_pass": bool(
            arm_status[spec.CALIBRATION_ARM]["calibration_validation_pass"]
        ),
        "arm_status": arm_status,
        "environment_condition_rows": rows,
        "gate_granularity": "environment_by_disturbance_mode",
        "selection_confidence": spec.SELECTION_CONFIDENCE,
        "bootstrap_draws": spec.BOOTSTRAP_DRAWS,
        "input_sha256": _input_sha256(run),
        "claim_boundary": (
            "The v14.11 projection calibration is compared with a mean-reward "
            "zero-strength continuation and must preserve paired actor tensors, "
            "actions, rewards, and latent-policy traces exactly. It can never be "
            "selected. Every learned arm uses independently registered upper "
            "and lower deployment-frequency dual rates and is compared with "
            "the same-strength, paired-relative-selector zero-dual control. "
            "The constrained update must act on deterministic actor-mean "
            "deployment traces and pass its reward guard. It must "
            "change actor tensors and actions, retain reward noninferiority in "
            "every condition, strictly improve reward in at least one condition, "
            "reduce effective and latent frequency leakage, preserve lower "
            "activity, and satisfy "
            "upper-HF and reconstruction budgets. This development screen cannot "
            "support a confirmatory claim."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_analysis(output_dir: Path, decision: dict[str, Any]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    decision_path = output / "decision.json"
    rendered = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    if decision_path.exists() and decision_path.read_text(
        encoding="utf-8"
    ) != rendered:
        raise RuntimeError("existing v14.11 screen decision differs")
    decision_path.write_text(rendered, encoding="utf-8")
    _write_csv(
        output / "environment_condition_gates.csv",
        decision["environment_condition_rows"],
    )
    lines = [
        "# MuJoCo v14.11 Iterative-Projection Screen",
        "",
        f"- Status: `{decision['status']}`",
        f"- Selected arm: `{decision['selected_arm']}`",
        f"- Calibration valid: `{decision['calibration_validation_pass']}`",
        "- Evidence role: development only; not confirmatory.",
        "- Learned comparator: same projection and paired-relative selector, zero deployment dual.",
        "",
        "| arm | role | comparator | conditions | trained | actor gate | behavior | projection | strict reward | select | paired actor RMS min | min slack |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, status in decision["arm_status"].items():
        actor_gate = (
            status["calibration_actor_identity_gate_pass"]
            if status["arm_role"] == "calibration"
            else status["learned_actor_difference_gate_pass"]
        )
        behavior_gate = (
            status["calibration_pathwise_identity_gate_pass"]
            if status["arm_role"] == "calibration"
            else status["learned_behavior_gate_pass"]
        )
        lines.append(
            f"| {arm} | {status['arm_role']} | {status['comparator_arm']} | "
            f"{status['passed_gate_count']}/{status['total_gate_count']} | "
            f"{status['trained_checkpoint_gate_pass']} | {actor_gate} | "
            f"{behavior_gate} | "
            f"{status['deployment_projection_mechanism_gate_pass']} | "
            f"{status['strict_reward_improvement_gate_pass']} | "
            f"{status['development_selection_pass']} | "
            f"{status['paired_actor_statistics']['combined_minimum']:.6g} | "
            f"{status['minimum_normalized_safety_slack']:.6g} |"
        )
    (output / "screen_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    decision = analyze(args.run_dir)
    write_analysis(args.output_dir, decision)
    print(
        f"mujoco_v14_11_screen status={decision['status']} "
        f"selected={decision['selected_arm']}"
    )


if __name__ == "__main__":
    main()
