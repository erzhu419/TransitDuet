#!/usr/bin/env python3
"""Run grouped v17.13 causal actor-adapter selection on reused paths."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freq_hrl.experiments.mujoco.causal_actor_adapter import (  # noqa: E402
    apply_causal_actor_adapter,
    causal_actor_features,
    fit_causal_actor_adapter,
)
from freq_hrl.experiments.mujoco import (  # noqa: E402
    full_horizon_responsibility_oracle as oracle,
)
from freq_hrl.experiments.reproducibility import (  # noqa: E402
    verify_current_freq_hrl_source_identity,
)
from scripts import mujoco_v17_13_causal_actor_adapter_spec as spec  # noqa: E402
from scripts.train_mujoco_v17_8_causal_fir import (  # noqa: E402
    load_reused_panel,
)


def target_artifact_path(target_root: Path, row: dict[str, Any]) -> Path:
    return (
        Path(target_root)
        / str(row["environment"])
        / str(row["disturbance_mode"])
        / f"seed_{int(row['evaluation_seed'])}"
        / "nearest_feasible_target.npz"
    )


def attach_actor_targets(
    panel: list[dict[str, Any]], target_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in panel:
        row = dict(source)
        total = np.asarray(row["total_action"], dtype=np.float64)
        target_path = target_artifact_path(target_root, row)
        actor_floor = not bool(row["oracle_joint_feasible"])
        if actor_floor:
            if not target_path.is_file():
                raise FileNotFoundError(
                    f"missing v17.12 actor-floor target: {target_path}"
                )
            with np.load(target_path, allow_pickle=False) as arrays:
                expected = {
                    "reference_total_action",
                    "target_upper_action",
                    "target_lower_action",
                    "target_total_action",
                    "total_action_correction",
                }
                if set(arrays.files) != expected:
                    raise ValueError(f"invalid v17.12 target keys: {target_path}")
                reference = np.asarray(
                    arrays["reference_total_action"], dtype=np.float64
                ).copy()
                target_upper = np.asarray(
                    arrays["target_upper_action"], dtype=np.float64
                ).copy()
                target_lower = np.asarray(
                    arrays["target_lower_action"], dtype=np.float64
                ).copy()
                target_total = np.asarray(
                    arrays["target_total_action"], dtype=np.float64
                ).copy()
                correction = np.asarray(
                    arrays["total_action_correction"], dtype=np.float64
                ).copy()
            if any(
                array.shape != total.shape
                or not np.all(np.isfinite(array))
                for array in (
                    reference,
                    target_upper,
                    target_lower,
                    target_total,
                    correction,
                )
            ):
                raise ValueError(f"invalid v17.12 target arrays: {target_path}")
            if (
                np.max(np.abs(reference - total))
                > spec.TARGET_ALIGNMENT_TOLERANCE
                or np.max(np.abs(target_upper + target_lower - target_total))
                > spec.TARGET_ALIGNMENT_TOLERANCE
                or np.max(np.abs(target_total - total - correction))
                > spec.TARGET_ALIGNMENT_TOLERANCE
            ):
                raise ValueError(f"misaligned v17.12 actor target: {target_path}")
        else:
            if target_path.exists():
                raise ValueError(
                    f"unexpected v17.12 target for feasible path: {target_path}"
                )
            correction = np.zeros_like(total)
        executed_correction = (
            np.clip(
                total + correction,
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
            "target_total_correction": correction,
            "target_correction_rms": _rms(correction),
            "target_correction_abs_max": float(np.max(np.abs(correction))),
            "target_executed_correction_rms": _rms(executed_correction),
        })
        rows.append(row)
    floor_count = sum(bool(row["actor_floor"]) for row in rows)
    floor_by_seed = {
        seed: sum(
            bool(row["actor_floor"])
            and int(row["evaluation_seed"]) == int(seed)
            for row in rows
        )
        for seed in spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT_BY_SEED
    }
    if (
        len(rows) != spec.EXPECTED_PATH_COUNT
        or floor_count != spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT
        or any(
            row["environment"] != spec.EXPECTED_ACTOR_FLOOR_ENVIRONMENT
            for row in rows if row["actor_floor"]
        )
        or floor_by_seed != spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT_BY_SEED
    ):
        raise RuntimeError("v17.13 target panel count mismatch")
    return rows


def candidate_id(
    window: int,
    ridge_penalty: float,
    actor_floor_weight: float,
    output_gain: float,
    correction_abs_limit: float,
) -> str:
    return (
        f"actor_fir_w{int(window)}_ridge{float(ridge_penalty):.0e}_"
        f"floorw{float(actor_floor_weight):g}_gain{float(output_gain):.2f}_"
        f"cap{float(correction_abs_limit):.3f}"
    )


def candidate_configs() -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": candidate_id(window, penalty, weight, gain, limit),
            "window": int(window),
            "ridge_penalty": float(penalty),
            "actor_floor_path_weight": float(weight),
            "output_gain": float(gain),
            "correction_abs_limit": float(limit),
        }
        for window in spec.FIR_WINDOWS
        for penalty in spec.RIDGE_PENALTIES
        for weight in spec.ACTOR_FLOOR_PATH_WEIGHTS
        for gain in spec.OUTPUT_GAINS
        for limit in spec.CORRECTION_ABS_LIMITS
    ]


def _fit_model(
    rows: list[dict[str, Any]],
    *,
    window: int,
    ridge_penalty: float,
    actor_floor_path_weight: float,
    statistics_cache: dict[tuple[Any, ...], dict[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    dimension = int(np.asarray(rows[0]["total_action"]).shape[1])
    if not any(bool(row["actor_floor"]) for row in rows):
        return {
            "window": int(window),
            "ridge_penalty": float(ridge_penalty),
            "action_dimension": dimension,
            "proposal_dimension": 2 * dimension,
            "fit_path_count": len(rows),
            "fit_total_path_weight": float(len(rows)),
            "coefficients": np.zeros(
                (int(window), 2 * dimension, dimension), dtype=np.float64
            ),
        }
    if statistics_cache is not None:
        weighted_gram = None
        weighted_cross = None
        total_weight = 0.0
        for row in rows:
            path_weight = (
                float(actor_floor_path_weight)
                if row["actor_floor"] else 1.0
            )
            statistics = statistics_cache[_statistics_key(row, window)]
            gram = path_weight * statistics["gram"]
            cross = path_weight * statistics["cross"]
            weighted_gram = (
                gram.copy() if weighted_gram is None else weighted_gram + gram
            )
            weighted_cross = (
                cross.copy()
                if weighted_cross is None else weighted_cross + cross
            )
            total_weight += path_weight
        if weighted_gram is None or weighted_cross is None:
            raise RuntimeError("v17.13 sufficient statistics are empty")
        raw_gram = weighted_gram / total_weight
        raw_cross = weighted_cross / total_weight
        feature_scale = np.maximum(
            np.sqrt(np.maximum(np.diag(raw_gram), 0.0)),
            spec.FEATURE_SCALE_FLOOR,
        )
        normalized_gram = raw_gram / (
            feature_scale[:, None] * feature_scale[None, :]
        )
        normalized_cross = raw_cross / feature_scale[:, None]
        normalized_coefficients = np.linalg.solve(
            normalized_gram
            + float(ridge_penalty)
            * np.eye(normalized_gram.shape[0], dtype=np.float64),
            normalized_cross,
        )
        coefficients = normalized_coefficients / feature_scale[:, None]
        return {
            "window": int(window),
            "ridge_penalty": float(ridge_penalty),
            "action_dimension": dimension,
            "proposal_dimension": 2 * dimension,
            "fit_path_count": len(rows),
            "fit_total_path_weight": float(total_weight),
            "coefficients": coefficients.reshape(
                int(window), 2 * dimension, dimension
            ),
        }
    return fit_causal_actor_adapter(
        [row["total_action"] for row in rows],
        [row["baseline_upper_action"] for row in rows],
        [row["target_total_correction"] for row in rows],
        [
            float(actor_floor_path_weight) if row["actor_floor"] else 1.0
            for row in rows
        ],
        window=int(window),
        ridge_penalty=float(ridge_penalty),
        feature_scale_floor=spec.FEATURE_SCALE_FLOOR,
    )


def build_statistics_cache(
    panel: list[dict[str, Any]],
) -> dict[tuple[Any, ...], dict[str, np.ndarray]]:
    floor_environments = {
        str(row["environment"]) for row in panel if row["actor_floor"]
    }
    cache: dict[tuple[Any, ...], dict[str, np.ndarray]] = {}
    for row in panel:
        if row["environment"] not in floor_environments:
            continue
        target = np.asarray(
            row["target_total_correction"], dtype=np.float64
        )
        for window in spec.FIR_WINDOWS:
            features = causal_actor_features(
                row["total_action"],
                row["baseline_upper_action"],
                window=int(window),
            )
            sample_count = float(features.shape[0])
            cache[_statistics_key(row, window)] = {
                "gram": features.T @ features / sample_count,
                "cross": features.T @ target / sample_count,
            }
    return cache


def _statistics_key(
    row: dict[str, Any], window: int
) -> tuple[Any, ...]:
    return (
        str(row["environment"]),
        str(row["disturbance_mode"]),
        int(row["evaluation_seed"]),
        int(window),
    )


def _raw_unit_prediction(
    row: dict[str, Any], model: dict[str, Any]
) -> np.ndarray:
    features = causal_actor_features(
        row["total_action"],
        row["baseline_upper_action"],
        window=int(model["window"]),
    )
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    return features @ coefficients.reshape(
        features.shape[1], int(model["action_dimension"])
    )


def _prediction_from_raw(
    row: dict[str, Any],
    raw_unit: np.ndarray,
    *,
    output_gain: float,
    correction_abs_limit: float,
) -> dict[str, Any]:
    total = np.asarray(row["total_action"], dtype=np.float64)
    raw = float(output_gain) * np.asarray(raw_unit, dtype=np.float64)
    trusted = np.clip(
        raw, -float(correction_abs_limit), float(correction_abs_limit)
    )
    corrected = np.clip(
        total + trusted,
        -spec.COMPONENT_SUM_LIMIT,
        spec.COMPONENT_SUM_LIMIT,
    )
    correction = corrected - total
    executed = (
        np.clip(
            corrected,
            -spec.EXECUTED_ACTION_LIMIT,
            spec.EXECUTED_ACTION_LIMIT,
        )
        - np.clip(
            total,
            -spec.EXECUTED_ACTION_LIMIT,
            spec.EXECUTED_ACTION_LIMIT,
        )
    )
    target = np.asarray(row["target_total_correction"], dtype=np.float64)
    valid = bool(
        np.all(np.isfinite(correction))
        and np.max(np.abs(correction))
        <= float(correction_abs_limit) + spec.CORRECTION_BOUND_TOLERANCE
        and np.max(np.abs(corrected))
        <= spec.COMPONENT_SUM_LIMIT + spec.CORRECTION_BOUND_TOLERANCE
    )
    return {
        "valid": valid,
        "corrected_total": corrected,
        "correction": correction,
        "correction_rms": _rms(correction),
        "correction_abs_max": float(np.max(np.abs(correction))),
        "executed_correction_rms": _rms(executed),
        "target_squared_error": float(np.sum(np.square(correction - target))),
        "target_squared_energy": float(np.sum(np.square(target))),
    }


def grouped_prefilter_summaries(
    panel: list[dict[str, Any]],
    statistics_cache: dict[
        tuple[Any, ...], dict[str, np.ndarray]
    ] | None = None,
) -> list[dict[str, Any]]:
    if statistics_cache is None:
        statistics_cache = build_statistics_cache(panel)
    configs = candidate_configs()
    by_base: dict[tuple[int, float, float], list[dict[str, Any]]] = defaultdict(list)
    for config in configs:
        by_base[(
            int(config["window"]),
            float(config["ridge_penalty"]),
            float(config["actor_floor_path_weight"]),
        )].append(config)
    accumulators = {
        str(config["candidate_id"]): _empty_accumulator() for config in configs
    }
    for environment in spec.ENVIRONMENTS:
        environment_rows = [
            row for row in panel if row["environment"] == environment
        ]
        for held_seed in spec.REUSED_SELECTION_SEEDS:
            fit_rows, held_rows = _fold_rows(environment_rows, held_seed)
            for (window, penalty, floor_weight), related in by_base.items():
                model = _fit_model(
                    fit_rows,
                    window=window,
                    ridge_penalty=penalty,
                    actor_floor_path_weight=floor_weight,
                    statistics_cache=statistics_cache,
                )
                for row in held_rows:
                    raw = _raw_unit_prediction(row, model)
                    for config in related:
                        metrics = _prediction_from_raw(
                            row,
                            raw,
                            output_gain=float(config["output_gain"]),
                            correction_abs_limit=float(
                                config["correction_abs_limit"]
                            ),
                        )
                        _accumulate(
                            accumulators[str(config["candidate_id"])],
                            row,
                            metrics,
                        )
    summaries = [
        _finish_accumulator(config, accumulators[str(config["candidate_id"])])
        for config in configs
    ]
    summaries.sort(key=lambda row: str(row["candidate_id"]))
    return summaries


def select_prefilter_candidate_ids(
    summaries: list[dict[str, Any]],
) -> list[str]:
    count = min(spec.PREFILTER_TOP_PER_RANKING, len(summaries))
    fidelity = sorted(summaries, key=lambda row: (
        float(row["actor_floor_target_normalized_mse"]),
        float(row["reference_feasible_correction_rms_mean"]),
        str(row["candidate_id"]),
    ))
    preservation_pool = [
        row for row in summaries
        if float(row["actor_floor_target_normalized_mse"]) <= 1.0
    ] or list(summaries)
    preservation = sorted(preservation_pool, key=lambda row: (
        float(row["reference_feasible_correction_rms_mean"]),
        float(row["actor_floor_target_normalized_mse"]),
        str(row["candidate_id"]),
    ))
    target_scale = max(
        float(fidelity[0]["actor_floor_target_correction_rms_mean"]),
        1e-12,
    )
    balanced = sorted(summaries, key=lambda row: (
        float(row["actor_floor_target_normalized_mse"])
        + 2.0 * float(row["reference_feasible_correction_rms_mean"])
        / target_scale,
        str(row["candidate_id"]),
    ))
    selected = {
        str(row["candidate_id"])
        for ranking in (fidelity, preservation, balanced)
        for row in ranking[:count]
    }
    return sorted(selected)


def grouped_candidate_rows(
    panel: list[dict[str, Any]],
    config: dict[str, Any],
    statistics_cache: dict[
        tuple[Any, ...], dict[str, np.ndarray]
    ] | None = None,
    oracle_executor: ProcessPoolExecutor | None = None,
) -> list[dict[str, Any]]:
    if statistics_cache is None:
        statistics_cache = build_statistics_cache(panel)
    rows: list[dict[str, Any]] = []
    for environment in spec.ENVIRONMENTS:
        environment_rows = [
            row for row in panel if row["environment"] == environment
        ]
        for held_seed in spec.REUSED_SELECTION_SEEDS:
            fit_rows, held_rows = _fold_rows(environment_rows, held_seed)
            model = _fit_model(
                fit_rows,
                window=int(config["window"]),
                ridge_penalty=float(config["ridge_penalty"]),
                actor_floor_path_weight=float(
                    config["actor_floor_path_weight"]
                ),
                statistics_cache=statistics_cache,
            )
            for row in held_rows:
                applied = apply_causal_actor_adapter(
                    row["total_action"],
                    row["baseline_upper_action"],
                    model,
                    output_gain=float(config["output_gain"]),
                    correction_abs_limit=float(
                        config["correction_abs_limit"]
                    ),
                    component_sum_limit=spec.COMPONENT_SUM_LIMIT,
                    executed_action_limit=spec.EXECUTED_ACTION_LIMIT,
                )
                metrics = _prediction_from_raw(
                    row,
                    np.asarray(applied["raw_correction"], dtype=np.float64)
                    / float(config["output_gain"]),
                    output_gain=float(config["output_gain"]),
                    correction_abs_limit=float(
                        config["correction_abs_limit"]
                    ),
                )
                corrected = np.asarray(
                    metrics.pop("corrected_total"), dtype=np.float64
                )
                metrics.pop("correction")
                if metrics["correction_rms"] <= 1e-15:
                    oracle_summary = {
                        "joint_feasible": bool(row["oracle_joint_feasible"]),
                        "upper_power": float(row["oracle_upper_power"]),
                        "lower_power": float(row["oracle_lower_power"]),
                        "status": "unchanged_reference_oracle_reused",
                    }
                else:
                    oracle_summary = (
                        _solve_corrected_oracle(corrected)
                        if oracle_executor is None
                        else oracle_executor.submit(
                            _solve_corrected_oracle, corrected
                        )
                    )
                output_row = {
                    "candidate_id": str(config["candidate_id"]),
                    "environment": str(environment),
                    "disturbance_mode": str(row["disturbance_mode"]),
                    "evaluation_seed": int(held_seed),
                    "actor_floor": bool(row["actor_floor"]),
                    "reference_joint_feasible": bool(
                        row["oracle_joint_feasible"]
                    ),
                    **metrics,
                }
                if isinstance(oracle_summary, dict):
                    _attach_oracle_summary(output_row, oracle_summary)
                else:
                    output_row["_oracle_future"] = oracle_summary
                rows.append(output_row)
    for row in rows:
        future = row.pop("_oracle_future", None)
        if future is not None:
            _attach_oracle_summary(row, future.result())
    if len(rows) != spec.EXPECTED_PATH_COUNT:
        raise RuntimeError("v17.13 grouped OOF path count mismatch")
    return rows


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


def selection_key(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(summary["corrected_joint_feasible_path_count"]),
        int(summary["actor_floor_recovered_path_count"]),
        int(summary["reference_feasible_preserved_path_count"]),
        sum(
            row["recovered_path_count"] == row["path_count"]
            for row in summary["actor_floor_by_seed"].values()
        ),
        -float(summary["actor_floor_target_normalized_mse"]),
        -float(summary["reference_feasible_correction_rms_mean"]),
        -float(summary["correction_abs_maximum"]),
    )


def reused_advancement_gate(summary: dict[str, Any]) -> dict[str, bool]:
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


def fit_final_models(
    panel: list[dict[str, Any]],
    selected: dict[str, Any],
    statistics_cache: dict[
        tuple[Any, ...], dict[str, np.ndarray]
    ] | None = None,
) -> dict[str, Any]:
    if statistics_cache is None:
        statistics_cache = build_statistics_cache(panel)
    models = {}
    for environment in spec.ENVIRONMENTS:
        rows = [row for row in panel if row["environment"] == environment]
        model = _fit_model(
            rows,
            window=int(selected["window"]),
            ridge_penalty=float(selected["ridge_penalty"]),
            actor_floor_path_weight=float(
                selected["actor_floor_path_weight"]
            ),
            statistics_cache=statistics_cache,
        )
        models[environment] = {
            **{key: value for key, value in model.items() if key != "coefficients"},
            "coefficients": np.asarray(
                model["coefficients"], dtype=np.float64
            ).tolist(),
        }
    return models


def run_selection(
    dataset_root: Path, target_root: Path, *, oracle_workers: int = 1
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    panel = attach_actor_targets(load_reused_panel(dataset_root), target_root)
    statistics_cache = build_statistics_cache(panel)
    prefilter = grouped_prefilter_summaries(panel, statistics_cache)
    selected_ids = select_prefilter_candidate_ids(prefilter)
    config_by_id = {
        str(row["candidate_id"]): row for row in candidate_configs()
    }
    full_rows = {}
    full_summaries = []
    worker_count = int(oracle_workers)
    if worker_count < 1:
        raise ValueError("v17.13 oracle worker count must be positive")
    executor_context = (
        ProcessPoolExecutor(max_workers=worker_count)
        if worker_count > 1 else nullcontext(None)
    )
    with executor_context as oracle_executor:
        for candidate in selected_ids:
            rows = grouped_candidate_rows(
                panel,
                config_by_id[candidate],
                statistics_cache,
                oracle_executor,
            )
            full_rows[candidate] = rows
            full_summaries.append(
                summarize_candidate(config_by_id[candidate], rows)
            )
    full_summaries.sort(key=lambda row: str(row["candidate_id"]))
    selected = max(full_summaries, key=selection_key)
    selected_rows = full_rows[str(selected["candidate_id"])]
    floor = [row for row in panel if row["actor_floor"]]
    target_audit = {
        "actor_floor_path_count": len(floor),
        "post_clipping_nonzero_target_path_count": sum(
            row["target_executed_correction_rms"]
            >= spec.EXECUTED_CORRECTION_RMS_MIN_GATE
            for row in floor
        ),
        "target_correction_rms": _distribution([
            row["target_correction_rms"] for row in floor
        ]),
        "target_executed_correction_rms": _distribution([
            row["target_executed_correction_rms"] for row in floor
        ]),
    }
    gate = reused_advancement_gate(selected)
    gate["all_actor_floor_targets_change_executed_action"] = bool(
        target_audit["post_clipping_nonzero_target_path_count"]
        == spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT
    )
    advances = bool(all(gate.values()))
    summary = {
        "status": (
            "causal_actor_adapter_authorizes_closed_loop_fresh_validation"
            if advances
            else "causal_actor_adapter_stops_before_fresh_path_access"
        ),
        "evidence_role": spec.EVIDENCE_ROLE,
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "frozen_core_revision": spec.FROZEN_CORE_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "dataset_root": str(dataset_root),
        "target_root": str(target_root),
        "selection_contract": spec.SELECTION_CONTRACT,
        "candidate_count": len(prefilter),
        "full_oracle_candidate_count": len(full_summaries),
        "oracle_worker_count": worker_count,
        "prefilter_selected_candidate_ids": selected_ids,
        "target_audit": target_audit,
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate": selected,
        "advancement_gate": gate,
        "fresh_path_access_allowed": advances,
        "fresh_validation_paths_accessed": False,
        "selected_out_of_fold_rows": selected_rows,
        "full_oracle_candidate_summaries": full_summaries,
        "prefilter_candidate_summaries": prefilter,
        "runtime_seconds": float(time.perf_counter() - started),
        "claim_boundary": spec.SELECTION_CONTRACT["claim_boundary"],
    }
    model = {
        "status": "v17_13_selected_causal_actor_adapter_fitted",
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "frozen_core_revision": spec.FROZEN_CORE_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "selected_candidate_id": selected["candidate_id"],
        "window": int(selected["window"]),
        "ridge_penalty": float(selected["ridge_penalty"]),
        "actor_floor_path_weight": float(
            selected["actor_floor_path_weight"]
        ),
        "output_gain": float(selected["output_gain"]),
        "correction_abs_limit": float(selected["correction_abs_limit"]),
        "component_sum_limit": spec.COMPONENT_SUM_LIMIT,
        "executed_action_limit": spec.EXECUTED_ACTION_LIMIT,
        "fresh_validation_eligible": advances,
        "environment_models": fit_final_models(
            panel, selected, statistics_cache
        ),
        "claim_boundary": spec.SELECTION_CONTRACT["claim_boundary"],
    }
    return summary, model


def _fold_rows(
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
        raise RuntimeError("v17.13 grouped fold construction mismatch")
    return fit, held


def _solve_corrected_oracle(corrected: np.ndarray) -> dict[str, Any]:
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


def _attach_oracle_summary(
    row: dict[str, Any], oracle_summary: dict[str, Any]
) -> None:
    row.update({
        "corrected_joint_feasible": bool(
            oracle_summary["joint_feasible"]
        ),
        "corrected_upper_power": float(oracle_summary["upper_power"]),
        "corrected_lower_power": float(oracle_summary["lower_power"]),
        "oracle_status": str(oracle_summary["status"]),
    })


def _empty_accumulator() -> dict[str, Any]:
    return {
        "path_count": 0,
        "valid_path_count": 0,
        "actor_floor_path_count": 0,
        "actor_floor_target_error": 0.0,
        "actor_floor_target_energy": 0.0,
        "actor_floor_executed_nonzero_path_count": 0,
        "actor_floor_target_correction_rms_sum": 0.0,
        "reference_feasible_path_count": 0,
        "reference_feasible_correction_rms_sum": 0.0,
        "reference_feasible_correction_rms_maximum": 0.0,
        "correction_abs_maximum": 0.0,
    }


def _accumulate(
    accumulator: dict[str, Any],
    row: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    accumulator["path_count"] += 1
    accumulator["valid_path_count"] += int(bool(metrics["valid"]))
    accumulator["correction_abs_maximum"] = max(
        float(accumulator["correction_abs_maximum"]),
        float(metrics["correction_abs_max"]),
    )
    if row["actor_floor"]:
        accumulator["actor_floor_path_count"] += 1
        accumulator["actor_floor_target_error"] += float(
            metrics["target_squared_error"]
        )
        accumulator["actor_floor_target_energy"] += float(
            metrics["target_squared_energy"]
        )
        accumulator["actor_floor_target_correction_rms_sum"] += float(
            row["target_correction_rms"]
        )
        accumulator["actor_floor_executed_nonzero_path_count"] += int(
            float(metrics["executed_correction_rms"])
            >= spec.EXECUTED_CORRECTION_RMS_MIN_GATE
        )
    if row["oracle_joint_feasible"]:
        accumulator["reference_feasible_path_count"] += 1
        accumulator["reference_feasible_correction_rms_sum"] += float(
            metrics["correction_rms"]
        )
        accumulator["reference_feasible_correction_rms_maximum"] = max(
            float(accumulator[
                "reference_feasible_correction_rms_maximum"
            ]),
            float(metrics["correction_rms"]),
        )


def _finish_accumulator(
    config: dict[str, Any], accumulator: dict[str, Any]
) -> dict[str, Any]:
    floor_count = int(accumulator["actor_floor_path_count"])
    reference_count = int(accumulator["reference_feasible_path_count"])
    target_energy = float(accumulator["actor_floor_target_energy"])
    if floor_count != spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT or target_energy <= 0.0:
        raise RuntimeError("v17.13 prefilter actor-floor denominator mismatch")
    if reference_count != spec.EXPECTED_REFERENCE_FEASIBLE_PATH_COUNT:
        raise RuntimeError("v17.13 prefilter reference denominator mismatch")
    return {
        **config,
        "path_count": int(accumulator["path_count"]),
        "valid_path_count": int(accumulator["valid_path_count"]),
        "actor_floor_path_count": floor_count,
        "actor_floor_target_normalized_mse": float(
            accumulator["actor_floor_target_error"] / target_energy
        ),
        "actor_floor_executed_nonzero_path_count": int(
            accumulator["actor_floor_executed_nonzero_path_count"]
        ),
        "actor_floor_target_correction_rms_mean": float(
            accumulator["actor_floor_target_correction_rms_sum"]
            / floor_count
        ),
        "reference_feasible_path_count": reference_count,
        "reference_feasible_correction_rms_mean": float(
            accumulator["reference_feasible_correction_rms_sum"]
            / reference_count
        ),
        "reference_feasible_correction_rms_maximum": float(
            accumulator["reference_feasible_correction_rms_maximum"]
        ),
        "correction_abs_maximum": float(
            accumulator["correction_abs_maximum"]
        ),
    }


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError("v17.13 distribution requires finite values")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
    }


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(
        np.asarray(values, dtype=np.float64)
    ))))


def write_outputs(
    summary: dict[str, Any], model: dict[str, Any], output_dir: Path
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "selected_model.json").write_text(
        json.dumps(model, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oracle-workers", type=int, default=1)
    args = parser.parse_args()
    source_identity = verify_current_freq_hrl_source_identity(
        code_revision=spec.FROZEN_CORE_REVISION,
        expected_source_manifest_sha256=spec.FROZEN_SOURCE_MANIFEST_SHA256,
        require_verified=True,
    )
    summary, model = run_selection(
        args.dataset_root,
        args.target_root,
        oracle_workers=int(args.oracle_workers),
    )
    summary["source_identity"] = source_identity
    model["source_identity"] = source_identity
    write_outputs(summary, model, args.output_dir)
    selected_summary = summary["selected_candidate"]
    floor_recovered = selected_summary["actor_floor_recovered_path_count"]
    reference_preserved = selected_summary[
        "reference_feasible_preserved_path_count"
    ]
    print(
        f"DONE v17.13 status={summary['status']} "
        f"candidate={summary['selected_candidate_id']} "
        f"floor={floor_recovered}/"
        f"{spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT} "
        f"preserved={reference_preserved}/"
        f"{spec.EXPECTED_REFERENCE_FEASIBLE_PATH_COUNT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
